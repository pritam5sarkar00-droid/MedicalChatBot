import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def load_single_pdf(path: str) -> List[Document]:
    """
    Same idea as load_pdf_file(), but for exactly one PDF already sitting
    on disk rather than every *.pdf in a directory. seed_data.py uses
    load_pdf_file() to bulk-load every file under data/seed/ at seed
    time; this is called per-file, on demand, whenever someone uploads a
    new PDF through the app (see src/documents.py) -- both paths funnel
    into the exact same ingest_pdf(), so a seeded document and an
    uploaded one are chunked, tagged, and indexed identically.
    """
    return PyPDFLoader(path).load()



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source', 'page', and 'page_label' in metadata (plus
    the original page_content) — drops noisier per-page metadata some PDF
    loaders attach (producer, creation date, total_pages, etc.) while
    keeping the fields the app actually uses: app.py's extract_sources()
    builds every citation chip shown under an answer from these.

    page vs page_label: 'page' is pypdf's raw, zero-indexed, purely
    sequential position of a page within the file. 'page_label' is the
    PDF's own embedded page numbering (many PDFs define this explicitly,
    e.g. roman numerals for a preface then arabic numerals once the main
    content starts) -- when it's present, it's what's actually *printed*
    on the page, which is what a citation should point to. A book with n
    pages of unnumbered front matter would otherwise have every citation
    off by n once you account for cover pages, a table of contents, etc.
    resolve_page_display() below prefers page_label and only falls back
    to page+1 when a PDF has no embedded labels at all.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        page = doc.metadata.get("page")
        page_label = doc.metadata.get("page_label")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src, "page": page, "page_label": page_label}
            )
        )
    return minimal_docs



def resolve_page_display(metadata: dict) -> Optional[str]:
    """
    The one place this app decides what page number a human -- or the
    model itself, see build_conversational_chain()'s document_prompt in
    src/pipeline.py -- should be told a chunk came from. Both
    app.py's extract_sources() (the citation chips under an answer) and
    the text actually stuffed into the LLM's context call this, so the
    page a citation shows and the page the model was told about can never
    quietly drift apart from the same fallback logic being written twice
    in two places (that exact class of bug is why
    tests/test_helper.py::test_filter_to_minimal_docs_preserves_source_page_and_page_label
    exists -- see its docstring).

    Prefers the PDF's own embedded page_label (see filter_to_minimal_docs'
    docstring for why this can differ from raw position); falls back to
    pypdf's raw zero-indexed page position + 1; returns None only when a
    chunk genuinely has neither (e.g. metadata stripped by an unusual
    loader) -- callers decide how to render that themselves (the UI
    omits the page segment entirely; the LLM-facing prompt says so
    explicitly, so it's never left to guess).
    """
    page_label = metadata.get("page_label")
    if page_label:
        return str(page_label)
    page = metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)
    return None



# Chunk size/overlap as named constants (not just inlined into
# text_split() below) so tests and anything else that needs to reason
# about chunk shape -- e.g. tests/test_documents.py's "uploaded docs use
# the same chunking scheme as everything else" check -- import the real
# number instead of hardcoding a copy that could silently drift out of
# sync with this file.
#
# 1000/150 rather than the original 500/20: 500 characters is roughly
# 80-100 words, which routinely split a single finding or instruction
# across two chunks -- great for keyword recall, bad for "does this one
# retrieved chunk actually contain a complete enough thought to answer
# from and cite accurately." 1000 characters holds a full short section
# of a fact sheet in one piece far more often, and a wider overlap (150,
# up from 20) means a sentence sitting right at a chunk boundary still
# shows up whole in at least one of the two chunks around that boundary
# rather than being truncated in both.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



# ---------------------------------------------------------------------------
# Embeddings + reranker
#
# Both of these can either load a model in-process (sentence-transformers
# + torch -- the original design, still the default) or talk over HTTP to
# inference_service/, a small standalone Flask app that loads those same
# two models exactly once and exposes them as POST /embed and POST
# /rerank. See DEPLOYMENT.md for why: sentence-transformers' torch
# dependency alone is a few hundred MB resident in memory, and loading it
# into the *same* process as Flask/LangChain/the Groq and Pinecone
# clients is what made this app hard to fit into a single free-tier
# host's RAM limit. Splitting it into its own service means it gets its
# own dedicated free instance instead of competing with everything else
# for one.
#
# Which mode is active is controlled by one env var, EMBEDDING_SERVICE_URL
# -- unset (the default) keeps the original in-process behavior, so local
# development, tests, and a single-box deploy are completely unaffected.
# ---------------------------------------------------------------------------

_REMOTE_TIMEOUT_S = 60  # generous on purpose: a free-tier inference
                         # service that has spun down from inactivity can
                         # take 30-60s to wake up and serve its first
                         # request again.


def _embedding_service_url() -> str:
    # Read fresh on every call (not cached at import time) so tests can
    # monkeypatch.setenv("EMBEDDING_SERVICE_URL", ...) per-test.
    return os.environ.get("EMBEDDING_SERVICE_URL", "").rstrip("/")


def _remote_auth_headers() -> dict:
    # inference_service/ sits on its own public URL once deployed, so
    # anyone who finds it could otherwise spend your free compute on
    # arbitrary embedding/rerank calls. Optional and off by default (an
    # empty dict, i.e. no Authorization header at all) to keep local dev
    # and anyone not worried about this friction-free -- set
    # INFERENCE_SERVICE_TOKEN to the same value on both this app and
    # inference_service/ to require it. See DEPLOYMENT.md.
    token = os.environ.get("INFERENCE_SERVICE_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


class RemoteEmbeddings(Embeddings):
    """
    Talks to inference_service's POST /embed instead of loading
    sentence-transformers in this process. Implements the same
    embed_query()/embed_documents() shape as LangChain's own
    HuggingFaceEmbeddings (this subclasses langchain_core's Embeddings
    ABC for exactly that reason), so it's a drop-in swap everywhere an
    Embeddings object is used -- PineconeVectorStore, the semantic cache
    -- without either needing to know or care which kind it was handed.
    """

    def __init__(self, base_url: str):
        self._base_url = base_url

    def _post(self, texts: List[str]) -> List[List[float]]:
        import requests

        last_error: Optional[Exception] = None
        for _attempt in range(2):  # one retry, to ride out a cold start
            try:
                resp = requests.post(
                    f"{self._base_url}/embed",
                    json={"texts": texts},
                    headers=_remote_auth_headers(),
                    timeout=_REMOTE_TIMEOUT_S,
                )
                resp.raise_for_status()
                return resp.json()["embeddings"]
            except requests.RequestException as exc:
                last_error = exc
        raise RuntimeError(f"embedding service at {self._base_url} did not respond: {last_error}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._post(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._post([text])[0]


class RemoteReranker:
    """
    Talks to inference_service's POST /rerank instead of loading a
    CrossEncoder in this process. Exposes the one method
    CombinedMedicalRetriever actually calls -- predict(pairs), the same
    name sentence-transformers' CrossEncoder itself uses -- so it's a
    drop-in swap there too; see _rerank() in src/pipeline.py.
    """

    def __init__(self, base_url: str):
        self._base_url = base_url

    def predict(self, pairs) -> List[float]:
        import requests

        pairs = list(pairs)
        if not pairs:
            return []
        query = pairs[0][0]
        documents = [text for _, text in pairs]
        last_error: Optional[Exception] = None
        for _attempt in range(2):
            try:
                resp = requests.post(
                    f"{self._base_url}/rerank",
                    json={"query": query, "documents": documents},
                    headers=_remote_auth_headers(),
                    timeout=_REMOTE_TIMEOUT_S,
                )
                resp.raise_for_status()
                return resp.json()["scores"]
            except requests.RequestException as exc:
                last_error = exc
        raise RuntimeError(f"reranker service at {self._base_url} did not respond: {last_error}")


#Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    """
    Returns something implementing embed_query()/embed_documents() --
    either a real local HuggingFaceEmbeddings (this model returns 384
    dimensions) or, if EMBEDDING_SERVICE_URL is set, a thin RemoteEmbeddings
    HTTP client pointed at inference_service/. See the module-level
    comment above for why this split exists.
    """
    base_url = _embedding_service_url()
    if base_url:
        return RemoteEmbeddings(base_url)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')



def download_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Loads a cross-encoder for reranking retrieved chunks — see
    CombinedMedicalRetriever in src/pipeline.py for how and why it's used
    -- or, if EMBEDDING_SERVICE_URL is set, returns a RemoteReranker HTTP
    client pointed at inference_service/ instead (same split as
    download_hugging_face_embeddings() above; that service loads and
    serves both models from one place).

    Unlike download_hugging_face_embeddings() above (a *bi*-encoder: the
    question and each chunk are embedded independently, then compared by
    cosine similarity — cheap enough to run against every chunk in the
    index), a cross-encoder scores one (question, chunk) pair *together*
    in a single forward pass, so it can pick up on interactions between
    the two texts that two independently-computed embeddings inherently
    can't capture. That's a meaningfully stronger relevance signal, but
    far too slow to run against a whole namespace — which is why it only
    reranks the already-narrowed candidate pool embedding retrieval
    produces, rather than replacing embedding retrieval entirely.

    ms-marco-MiniLM-L-6-v2 specifically: same MiniLM family and a similar
    six-layer size as the embedding model above, so a free-tier CPU is
    running two small, similarly-cheap models rather than one small model
    plus one heavy one. sentence-transformers already depends on
    everything this needs (torch, transformers) — no new package.

    activation_fn=Sigmoid(): MS MARCO cross-encoders output raw,
    unbounded logits by default (a documented gotcha — see sentence-
    transformers' own docs), not 0–1 scores. This doesn't actually change
    anything CombinedMedicalRetriever does with the output (it only ever
    uses these scores to *sort* candidates, and sigmoid is monotonic, so
    the ranking is identical either way) — set explicitly anyway so the
    numbers are sane if anyone (a debugger, a future feature) ever looks
    at them directly instead of just their relative order.
    """
    base_url = _embedding_service_url()
    if base_url:
        return RemoteReranker(base_url)
    from sentence_transformers import CrossEncoder
    from torch.nn import Sigmoid

    return CrossEncoder(model_name, activation_fn=Sigmoid())
