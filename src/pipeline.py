"""
pipeline.py — builds the conversational RAG pipeline: HuggingFace
embeddings, a Pinecone retriever, a history-aware query rewriter, and the
Groq-backed answer chain.

This is deliberately separated from app.py (Flask) so it can be reused by:
  - app.py            the web app, via create_app()
  - eval/run_eval.py   a standalone evaluation script — no Flask needed
  - tests/             which inject a lightweight FAKE pipeline instead of
                        importing this module at all, so the test suite
                        never needs real Pinecone/Groq credentials.

One index, one namespace, optional per-document filtering
------------------------------------------------------------
Every document in the knowledge base -- whether it was sitting in
data/seed/ at deploy time (seed_data.py) or uploaded later through the
UI (src/documents.py, wired up via the /documents routes in app.py) --
is embedded with the same model and upserted into the same namespace,
DOCUMENTS_NAMESPACE (src/documents.py), on the same Pinecone index. There
is no separate "reference" vs "uploaded" distinction anywhere in this
file, in Pinecone, or in the Postgres document manifest (src/
document_store.py): a document is a document, and every one of them can
be listed, selected, or deleted the same way.

CombinedMedicalRetriever below is what turns a chat question into a
ranked context list from that namespace, optionally scoped to just the
document(s) a person picked in the sidebar (via a Pinecone metadata
filter on doc_id -- see search()) -- see its docstring for why a
similarity floor is applied before anything reaches the model, and for
how an optional cross-encoder reranking pass fits in on top of that.

Follow-up questions and build_conversational_chain()
------------------------------------------------------
A naive "history-aware retriever" setup (LangChain's create_retrieval_chain
+ create_history_aware_retriever used as-is) only uses the rewritten,
context-resolved question for *retrieval* — the final answer-generation
call still receives the user's original, possibly-ambiguous message
("explain in details") rather than what it was rewritten to ("explain
asthma in detail"), and has to re-resolve the reference itself from raw
chat history. That mostly works, but is a real, verifiable weak point on
terse follow-ups (see build_conversational_chain()'s docstring for how
this is confirmed and fixed).
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import os

from pydantic import ConfigDict
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from src.helper import download_hugging_face_embeddings, download_reranker, resolve_page_display
from src.prompt import system_prompt, contextualize_q_system_prompt
from src.documents import DOCUMENTS_NAMESPACE

INDEX_NAME = "pritam-medical-chatbot"
EMBEDDING_DIMENSION = 384  # must match src/helper.py's embedding model (all-MiniLM-L6-v2)

# Cosine similarity floor (Pinecone's index metric — see seed_data.py)
# below which a retrieved chunk is treated as "not actually related to
# the question" and dropped before it ever reaches the LLM. This is
# deliberately conservative: it only exists to catch questions that are
# nowhere in the knowledge base at all (e.g. "what's the capital of
# France" against a set of medical fact sheets), not to second-guess
# whether a topically-related chunk actually answers the specific
# question asked — that finer judgment is left to the LLM via
# src/prompt.py, since it needs real language understanding a cosine
# cutoff can't provide. Tune this against real queries on your own
# Pinecone index if it ever feels too trigger-happy or too lenient.
MIN_SIMILARITY = 0.2


class CombinedMedicalRetriever(BaseRetriever):
    """Searches DOCUMENTS_NAMESPACE -- the one namespace every seeded and
    uploaded document lives in, see src/documents.py -- optionally
    narrowed to a chosen subset of documents, and drops anything below
    MIN_SIMILARITY.

    Why filter at all: the plain `vectorstore.as_retriever()` a single
    namespace search would use always returns its top-k chunks, even when
    the *best* of them is barely related to the question — there's no
    "actually, I've got nothing" option. Handing an LLM an irrelevant
    chunk as "context" is exactly what invites a confident-sounding
    hallucination instead of an honest "I don't know". Filtering those
    out means a clearly off-topic question comes back with an empty
    context list instead, which app.py uses to short-circuit straight to
    an honest "I don't have information about that" — without spending an
    LLM call on it at all (the same "cheap check before the expensive
    one" shape as the emergency guardrail in src/safety.py).

    Scoping to specific documents: search() below takes an optional
    doc_ids list -- when a person checks a subset of documents in the
    sidebar instead of leaving everything selected, app.py passes their
    doc_ids straight through here, and it becomes a Pinecone metadata
    filter (`{"doc_id": {"$in": doc_ids}}`) applied at the vector-search
    step itself, not a post-hoc Python filter over already-fetched
    results — so a narrow selection doesn't spend its k budget on
    candidates from documents that were never going to be shown anyway.
    _get_relevant_documents() (the plain BaseRetriever interface used by
    eval/run_eval.py and anything else that just calls .invoke(query))
    always searches every document, unfiltered — only the chat pipeline
    itself, via whatever's chosen in the UI, ever passes doc_ids.

    A namespace query that raises (Pinecone briefly unavailable, an
    unexpected filter shape) degrades to "no results from that query"
    rather than taking down retrieval entirely.

    Optional reranking: when `reranker` is set (a sentence-transformers
    CrossEncoder — see download_reranker() in src/helper.py), retrieval
    becomes two-stage. Embedding search still runs first and still pulls
    a *wider* pool (rerank_pool_size) instead of going straight to
    k_total; the cross-encoder then re-scores that whole pool by jointly
    encoding each (question, chunk) pair, and only *that* ordering
    decides the final top k_total. Bi-encoder similarity (comparing two
    independently-built embeddings) is a proxy for relevance; a
    cross-encoder scoring the pair together is a substantially stronger
    one, at a cost that only makes sense to pay against an
    already-narrowed pool rather than a whole namespace. If reranking
    fails for any reason (or `reranker` is None — see build_pipeline's
    use_reranker flag), retrieval quietly falls back to plain
    embedding-score ordering rather than losing the answer entirely.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any
    namespaces: List[str]
    k_per_namespace: int = 3
    k_total: int = 4
    min_similarity: float = MIN_SIMILARITY
    reranker: Optional[Any] = None
    rerank_pool_size: int = 12

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.search(query, doc_ids=None)

    def search(self, query: str, doc_ids: Optional[List[str]] = None) -> List[Document]:
        """The actual retrieval entry point build_conversational_chain()
        calls (see the context-assigning lambda at the bottom of this
        file) -- doc_ids is None or empty to search every document, or a
        list of doc_id values (see src/documents.py's ingest_pdf()) to
        search only those.

        This is a plain method rather than something routed through
        BaseRetriever's own .invoke()/**kwargs machinery on purpose:
        LangChain's stable Runnable interface doesn't guarantee arbitrary
        extra kwargs passed to .invoke() reach _get_relevant_documents()
        untouched, and an explicit method here is one less thing to trust
        about a framework internal. _get_relevant_documents() above still
        exists and still works exactly as before for anything that treats
        this as a standard retriever (eval/run_eval.py, tests) -- it's
        just this method with doc_ids=None.
        """
        # Reranking needs a genuinely useful pool to choose from, wider
        # than the k_per_namespace tuned for the no-reranker case.
        fetch_k = max(self.k_per_namespace, self.rerank_pool_size) if self.reranker else self.k_per_namespace

        # Pinecone's filter syntax: None means unfiltered; a populated
        # filter is applied *during* the vector search itself, not as a
        # post-hoc Python-side filter over already-fetched candidates --
        # so a narrow document selection doesn't waste its k budget on
        # chunks from documents that were never in scope.
        search_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None

        scored: List[Tuple[Document, float]] = []
        for namespace in self.namespaces:
            try:
                scored.extend(
                    self.vectorstore.similarity_search_with_score(
                        query, k=fetch_k, namespace=namespace, filter=search_filter
                    )
                )
            except Exception:
                continue

        scored.sort(key=lambda pair: pair[1], reverse=True)
        candidates = [pair for pair in scored if pair[1] >= self.min_similarity]

        if self.reranker and candidates:
            results = self._rerank(query, candidates)
        else:
            results = candidates[: self.k_total]

        # Stashed in metadata (not returned as a separate structure)
        # because BaseRetriever's interface contract is "return
        # List[Document]" -- every caller downstream of a retriever
        # (create_history_aware_retriever, create_stuff_documents_chain)
        # expects exactly that shape, so smuggling extra fields through
        # metadata is the least invasive way to make them visible to
        # app.py's extract_sources() without changing that contract.
        #
        # retrieval_score: rounded to 4 places since raw float32 cosine
        # similarity carries far more precision than is meaningful to
        # show anyone. Always the original embedding cosine score, even
        # when reranking picked the final order -- see _rerank()'s
        # docstring for why.
        #
        # page_display: computed once, here, via the same
        # resolve_page_display() app.py's extract_sources() calls for the
        # citation chips -- see that function's docstring in src/helper.py
        # for why sharing one implementation matters. This is what
        # build_conversational_chain()'s document_prompt shows the LLM
        # alongside each chunk, so the model can name a real page if
        # asked instead of having no way to know one at all.
        for doc, score in results:
            doc.metadata["retrieval_score"] = round(float(score), 4)
            doc.metadata["page_display"] = resolve_page_display(doc.metadata) or "not given"

        return [doc for doc, _ in results]

    def _rerank(self, query: str, candidates: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Re-orders (and re-selects) the top rerank_pool_size candidates
        using the cross-encoder, then returns the best k_total — still as
        (Document, score) pairs so the caller's code above doesn't need to
        know reranking happened at all.

        The score attached to each returned pair is deliberately still
        the *original* embedding cosine similarity, not the
        cross-encoder's own score. A cross-encoder outputs an unbounded
        raw logit on a completely different scale than the [0,1] cosine
        range the UI already displays as "Relevance: NN%" (see
        extract_sources() in app.py) -- turning that logit into a
        percentage would need a calibrated activation function this
        project has no labeled data to tune correctly, and getting it
        wrong would be actively misleading. The cross-encoder's job here
        is to pick better chunks and put them in a better order; the
        cosine score's job is to be an honest, consistently-scaled number
        to show a person.
        """
        pool = candidates[: self.rerank_pool_size]
        pairs = [(query, doc.page_content) for doc, _ in pool]
        try:
            cross_scores = self.reranker.predict(pairs)
        except Exception:
            return candidates[: self.k_total]

        order = sorted(range(len(pool)), key=lambda i: cross_scores[i], reverse=True)
        return [pool[i] for i in order[: self.k_total]]


@dataclass
class RAGPipeline:
    embeddings: Any  # exposes .embed_query(text) -> list[float]
    retriever: Any
    chain: Any        # a LangChain Runnable: supports .invoke() and .stream()
    vectorstore: Any = None  # PineconeVectorStore — used by app.py's /documents routes to add_documents()/delete() uploads
    reranker: Any = None     # sentence-transformers CrossEncoder, or None if use_reranker=False / it failed to load


# Phrases that show up when the query-rewriting LLM (contextualize_q_prompt
# below) slips into commenting on a previous turn instead of producing a
# clean standalone question -- e.g. after an unrelated topic switch, "It
# seems I made an incorrect assumption... I don't have information about
# 'bike'" instead of a rewritten question about the *new* topic. Seen for
# real, not hypothetical — see _looks_like_a_reasonable_rewrite()'s
# docstring. Deliberately narrow and low-risk: these are phrases a genuine
# standalone question would essentially never contain, so this only ever
# rejects rewrites that have already gone wrong, not borderline-but-fine
# ones.
_REWRITE_META_COMMENTARY_MARKERS = (
    "i made an incorrect",
    "i made a mistake",
    "i should have said",
    "i should have answered",
    "it seems i",
    "you are correct",
    "you're correct",
    "i apologize",
    "i'm sorry",
    "my previous answer",
    "my previous response",
    "my apologies",
    "correcting my",
    "to correct myself",
    "to answer your original question",
)


def _looks_like_a_reasonable_rewrite(original: str, rewritten: str) -> bool:
    """Cheap, deterministic sanity check on the query-rewriting LLM's
    output, run before trusting it to drive *both* retrieval and the
    final answer (build_conversational_chain below stakes a lot on this
    rewrite being good, more than the old architecture ever did).

    Confirmed failure mode this guards against: given a chat history that
    jumps between unrelated topics (e.g. a medical question, then "what
    is bike", then a completely unrelated follow-up about an uploaded
    document), the rewriting model can sometimes produce commentary about
    the *previous* answer instead of a clean standalone question for the
    *current* one — e.g. producing something like "It seems I made an
    incorrect assumption, I don't have information about bike" instead of
    rewriting the actual new question. Because that flawed text then
    drives both retrieval and the final answer, the user ends up seeing a
    reply about entirely the wrong topic. contextualize_q_system_prompt
    (src/prompt.py) is written to avoid this directly, but no prompt
    guarantees compliance from an LLM on every single call — this is the
    deterministic backstop: if the rewrite doesn't look like a plausible
    question, fall back to the user's own raw message instead (see
    resolve_standalone_question below), which is always at least a
    faithful account of what they actually asked, even if imperfectly
    contextualized.
    """
    if not rewritten or not rewritten.strip():
        return False
    lowered = rewritten.lower()
    if any(marker in lowered for marker in _REWRITE_META_COMMENTARY_MARKERS):
        return False
    # A rewrite should be a tighter-or-similar restatement of the
    # question, not several sentences of rambling -- generous headroom
    # (a genuine rewrite can reasonably run longer than the original,
    # e.g. a terse "why?" expanding to name a whole condition) while still
    # catching outputs that have clearly ballooned into something else.
    if len(rewritten) > max(300, len(original) * 8):
        return False
    return True


def _retrieve_context(retriever, query: str, document_ids: Optional[List[str]]) -> List[Document]:
    """Calls retriever.search(query, doc_ids=...) when the retriever
    supports it (CombinedMedicalRetriever does -- see its docstring),
    otherwise falls back to the plain BaseRetriever .invoke(query) every
    other retriever (including test doubles like
    tests/test_pipeline_chain.py's RecordingRetriever) already supports.

    Duck-typed on hasattr rather than an isinstance check, so a
    test-only retriever never needs to actually subclass
    CombinedMedicalRetriever just to be usable with this chain -- it only
    needs a .search() method if a test actually wants to exercise
    document-scoped retrieval.
    """
    search = getattr(retriever, "search", None)
    if callable(search):
        return search(query, doc_ids=document_ids)
    return retriever.invoke(query)


def _ensure_prompt_metadata(docs: List[Document]) -> List[Document]:
    """build_conversational_chain()'s document_prompt reads {source} and
    {page_display} out of each retrieved Document's metadata to label it
    for the LLM. CombinedMedicalRetriever.search() already sets both on
    every real chunk, but guarantee they exist here too (with a plain,
    honest fallback) so this chain also works with a minimal retriever
    that was never told about this prompt's specific field names --
    e.g. a test double that only cares about page_content. Mutates in
    place and returns the same list, matching the pattern
    CombinedMedicalRetriever.search() itself already uses for
    retrieval_score/page_display.
    """
    for doc in docs:
        doc.metadata.setdefault("source", "the knowledge base")
        doc.metadata.setdefault("page_display", "not given")
    return docs


def build_conversational_chain(chat_model, retriever):
    """
    Builds the full conversational RAG chain: rewrite the question using
    history if there is any, retrieve with the rewritten question, answer
    using the retrieved context — and critically, the *rewritten* question
    is what both retrieval and answer-generation see, not the user's raw
    follow-up.

    Split out from build_pipeline() (which also needs real Pinecone/Groq
    credentials to build embeddings/vectorstore/chat_model) specifically
    so this composition has real unit test coverage with fake
    chat_model/retriever — see tests/test_pipeline_chain.py — the
    behavior below was verified there and empirically, not just written
    and hoped to work.

    Why not the textbook create_history_aware_retriever +
    create_retrieval_chain combo: that pairing only feeds the rewritten,
    context-resolved question to the *retriever*. The final
    answer-generation call still gets the user's original, possibly
    ambiguous message — e.g. asking "What is asthma?" then "explain in
    details" rewrites to "explain asthma in detail" for retrieval, but
    the answering call still literally sees "explain in details" as the
    human turn, and has to re-resolve on its own, from raw chat history,
    what "in details" is even about. That mostly works, since a capable
    model can usually piece it together, but it's a real, unnecessary
    weak point on terse or ambiguous follow-ups ("explain more", "why?",
    "and in children?") — exactly the "forgets what we were just talking
    about" complaint this function exists to fix. The fix doesn't need an
    extra LLM call: the rewrite step already runs once, this just reuses
    its output for the answer step too instead of throwing it away after
    retrieval.

    Making the rewrite do double duty like this raises the stakes on it
    being *good*, though — a bad rewrite used to only weaken retrieval;
    now it can derail the final answer too. resolve_standalone_question()
    below adds a deterministic fallback for exactly that case: run the
    output through _looks_like_a_reasonable_rewrite() and use the user's
    original raw message instead if it fails, rather than trusting every
    rewrite unconditionally.

    History-aware retrieval's own optimization is preserved: with no
    chat history yet (a conversation's first message), rewriting is
    skipped entirely — there's nothing to resolve — so this costs zero
    extra LLM calls in that case, same as before.

    Scoping to chosen documents: if the dict passed to .invoke()/.stream()
    has a "document_ids" key (app.py sets this from whatever's checked in
    the sidebar; it's absent or None for eval/run_eval.py and any other
    caller that doesn't care), it's threaded through to the retriever via
    _retrieve_context() above, so retrieval only ever considers those
    documents. document_prompt below is what actually shows the model a
    source and page number per chunk (previously it saw raw page_content
    only, with no way to know either) -- src/prompt.py's system_prompt
    tells it how, and how not, to use that.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    query_rewriter = contextualize_q_prompt | chat_model | StrOutputParser()

    def resolve_standalone_question(x):
        if not x.get("chat_history"):
            return x["input"]
        rewritten = query_rewriter.invoke(x)
        if _looks_like_a_reasonable_rewrite(x["input"], rewritten):
            return rewritten
        return x["input"]

    standalone_question_chain = RunnableLambda(resolve_standalone_question)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # Labels each chunk with its source document and page before the
    # model ever sees it -- create_stuff_documents_chain's own default
    # document_prompt is just "{page_content}", which is why the model
    # previously had no way to honestly answer "what page is that on"
    # even though app.py's extract_sources() already had the real page
    # number for the *structured* citation list shown separately in the
    # UI. {source} and {page_display} are guaranteed present on every
    # document context reaches this chain with -- see
    # _ensure_prompt_metadata() above.
    document_prompt = PromptTemplate.from_template("[Source: {source} | Page: {page_display}]\n{page_content}")
    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt, document_prompt=document_prompt)

    # RunnablePassthrough.assign(input=...) deliberately *overwrites* the
    # input key with the resolved standalone question -- everything after
    # this point (retrieval, then answer generation) should see the
    # disambiguated question, never the raw original one. Nothing
    # downstream (app.py, eval/run_eval.py) reads the chain's returned
    # "input" key, so overwriting it here is safe.
    #
    # Deliberately three *method-chained* .assign() calls building one
    # composite structure, NOT `step1 | step2 | step3` piping three
    # separately-built RunnablePassthrough.assign(...) results together --
    # those look equivalent and both invoke() correctly, but only the
    # method-chained form preserves app.py's early-break-on-empty-context
    # optimization under .stream(): piping separate assigns together
    # builds a plain RunnableSequence, whose .stream() was verified (see
    # tests/test_pipeline_chain.py) to still dispatch the final answer
    # LLM call even after a consumer stops pulling right after an empty
    # context chunk -- silently turning "skip the wasted Groq call on an
    # obviously out-of-scope question" back into "waste it anyway."
    # Chaining .assign() calls off the same RunnablePassthrough avoids
    # that regression, confirmed by the same test.
    return (
        RunnablePassthrough.assign(input=standalone_question_chain)
        .assign(
            context=(
                lambda x: _ensure_prompt_metadata(
                    _retrieve_context(retriever, x["input"], x.get("document_ids"))
                )
            )
        )
        .assign(answer=question_answer_chain)
    )


def ensure_index_exists(index_name: str) -> None:
    """Creates the Pinecone index if this is a brand new Pinecone project
    with nothing in it yet (e.g. straight after following DEPLOYMENT.md's
    "create a free Pinecone account" step) -- so a fresh deploy needs zero
    manual Pinecone-console steps, just the API key in the environment.

    Safe to call on every startup: has_index() is one cheap read, and
    create_index() only ever runs the one time it's actually needed.

    Dimension must match the embedding model exactly (see
    EMBEDDING_DIMENSION's comment above) -- a vector index's dimension is
    fixed at creation and can't be changed afterward, so switching
    embedding models later means deleting and recreating the index (and
    re-running seed_data.py / re-uploading documents), not just changing
    this number.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def build_pipeline(index_name: str = INDEX_NAME, k: int = 3, use_reranker: bool = True) -> RAGPipeline:
    embeddings = download_hugging_face_embeddings()

    reranker = None
    if use_reranker:
        try:
            reranker = download_reranker()
        except Exception:
            # Reranking is a quality improvement, not a hard requirement.
            # A model-download hiccup, or a host that genuinely can't
            # spare the RAM for a second small model (see README
            # Troubleshooting), degrades to plain embedding-score
            # retrieval instead of failing app startup entirely.
            reranker = None

    ensure_index_exists(index_name)
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = CombinedMedicalRetriever(
        vectorstore=docsearch,
        namespaces=[DOCUMENTS_NAMESPACE],
        k_per_namespace=k,
        k_total=k + 1,
        reranker=reranker,
    )

    chat_model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
    chain = build_conversational_chain(chat_model, retriever)

    return RAGPipeline(embeddings=embeddings, retriever=retriever, chain=chain, vectorstore=docsearch, reranker=reranker)
