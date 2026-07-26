"""
Tests for src/pipeline.py's CombinedMedicalRetriever — merges results
across one or more Pinecone namespaces by score, applies the
MIN_SIMILARITY floor that lets app.py short-circuit to an honest "no
information" answer, optionally reranks, and optionally narrows a search
to a chosen subset of documents (doc_ids).

A fake vectorstore stands in for PineconeVectorStore here, so these run
with no network, no API keys, and no real embedding model — see
src/pipeline.py's own module docstring for why this class exists at all.

Production (build_pipeline()) only ever passes one namespace now
(DOCUMENTS_NAMESPACE — every document in the knowledge base, seeded or
uploaded, shares it; see src/documents.py's module docstring for why).
The retriever itself stays genuinely multi-namespace-capable though, so
most tests below still exercise that with two clearly-generic namespace
names ("ns-a"/"ns-b") rather than DOCUMENTS_NAMESPACE — what's under test
in this file is "does merging/reranking/filtering work correctly across
however many namespaces it's given", which doesn't depend on how many
namespaces production happens to configure it with today.
"""

from langchain_core.documents import Document

from src.documents import DOCUMENTS_NAMESPACE
from src.pipeline import CombinedMedicalRetriever

NS_A = "ns-a"
NS_B = "ns-b"


def doc(text, source="book.pdf", doc_id=None):
    metadata = {"source": source}
    if doc_id is not None:
        metadata["doc_id"] = doc_id
    return Document(page_content=text, metadata=metadata)


class FakeVectorStore:
    """results: {namespace: [(Document, score), ...]}. raises_for: set of
    namespaces whose query should raise, to test graceful degradation.

    filter emulates just enough of Pinecone's real behavior to test
    CombinedMedicalRetriever.search()'s doc_ids narrowing: when a filter
    is given, only candidates whose doc_id is in filter["doc_id"]["$in"]
    are returned — same as Pinecone applying the filter *during* the
    vector search itself, not as a separate post-hoc step.
    """

    def __init__(self, results=None, raises_for=None):
        self.results = results or {}
        self.raises_for = raises_for or set()
        self.calls = []  # (query, k, namespace, filter) for every call made

    def similarity_search_with_score(self, query, k=4, namespace=None, filter=None):
        self.calls.append((query, k, namespace, filter))
        if namespace in self.raises_for:
            raise RuntimeError(f"simulated Pinecone failure for namespace={namespace!r}")
        candidates = self.results.get(namespace, [])
        if filter:
            allowed_ids = set(filter.get("doc_id", {}).get("$in", []))
            candidates = [(d, s) for d, s in candidates if d.metadata.get("doc_id") in allowed_ids]
        return candidates[:k]


def make_retriever(vectorstore, **overrides):
    kwargs = dict(vectorstore=vectorstore, namespaces=[NS_A, NS_B], k_per_namespace=3, k_total=4)
    kwargs.update(overrides)
    return CombinedMedicalRetriever(**kwargs)


def test_merges_and_ranks_across_both_namespaces_by_score():
    doc_a = doc("content from ns-a", "a.pdf")
    doc_b = doc("content from ns-b", "b.pdf")
    vs = FakeVectorStore(results={NS_A: [(doc_a, 0.5)], NS_B: [(doc_b, 0.9)]})

    results = make_retriever(vs).invoke("some question")

    assert results == [doc_b, doc_a]  # higher score (0.9) ranked first, regardless of namespace


def test_drops_anything_below_min_similarity():
    relevant = doc("relevant content")
    irrelevant = doc("barely related content")
    vs = FakeVectorStore(results={NS_A: [(relevant, 0.6), (irrelevant, 0.05)]})

    results = make_retriever(vs, min_similarity=0.2).invoke("some question")

    assert results == [relevant]


def test_returns_empty_list_when_nothing_clears_the_floor():
    off_topic = doc("completely unrelated content")
    vs = FakeVectorStore(results={NS_A: [(off_topic, 0.1)]})

    results = make_retriever(vs, min_similarity=0.2).invoke("what's the capital of France?")

    assert results == []


def test_respects_k_total_cap_even_with_many_good_matches():
    docs = [(doc(f"content {i}"), 0.9 - i * 0.01) for i in range(10)]
    vs = FakeVectorStore(results={NS_A: docs})

    # k_per_namespace must be >= k_total here, or the fake store's own
    # per-call slice (simulating Pinecone's own top-k) caps candidates
    # before the merge step's k_total ever gets a chance to matter.
    results = make_retriever(vs, k_per_namespace=10, k_total=4, min_similarity=0.0).invoke("some question")

    assert len(results) == 4
    assert results[0].page_content == "content 0"  # best score kept


def test_queries_each_namespace_with_k_per_namespace():
    vs = FakeVectorStore(results={})
    make_retriever(vs, k_per_namespace=3).invoke("some question")

    namespaces_queried = {call[2] for call in vs.calls}
    assert namespaces_queried == {NS_A, NS_B}
    assert all(call[1] == 3 for call in vs.calls)


def test_a_failed_namespace_does_not_break_retrieval_from_the_other():
    doc_a = doc("still findable", "a.pdf")
    vs = FakeVectorStore(results={NS_A: [(doc_a, 0.7)]}, raises_for={NS_B})

    results = make_retriever(vs).invoke("some question")

    assert results == [doc_a]


def test_empty_namespace_degrades_gracefully():
    """A namespace with nothing indexed in it yet should behave exactly
    like a namespace with zero matches, not an error."""
    doc_a = doc("content", "a.pdf")
    vs = FakeVectorStore(results={NS_A: [(doc_a, 0.7)], NS_B: []})

    results = make_retriever(vs).invoke("some question")

    assert results == [doc_a]


def test_production_default_is_a_single_shared_namespace():
    """build_pipeline() (src/pipeline.py) only ever configures this with
    one namespace -- DOCUMENTS_NAMESPACE, shared by every document in the
    knowledge base regardless of whether it arrived via data/seed/ or an
    upload (see src/documents.py's module docstring for why there's no
    longer a second namespace to merge against in production)."""
    vs = FakeVectorStore(results={})
    make_retriever(vs, namespaces=[DOCUMENTS_NAMESPACE]).invoke("some question")

    assert {call[2] for call in vs.calls} == {DOCUMENTS_NAMESPACE}


def test_default_min_similarity_matches_module_constant():
    from src.pipeline import MIN_SIMILARITY

    vs = FakeVectorStore()
    retriever = make_retriever(vs)
    assert retriever.min_similarity == MIN_SIMILARITY


def test_stashes_retrieval_score_into_each_documents_metadata():
    """extract_sources() (app.py) and the retrieval-transparency view both
    read doc.metadata['retrieval_score'] -- this is the one place that
    value gets set, so a regression here would silently break both."""
    doc_a = doc("content from ns-a", "a.pdf")
    doc_b = doc("content from ns-b", "b.pdf")
    vs = FakeVectorStore(results={NS_A: [(doc_a, 0.5)], NS_B: [(doc_b, 0.876543)]})

    results = make_retriever(vs).invoke("some question")

    assert results[0].metadata["retrieval_score"] == 0.8765  # rounded to 4 places
    assert results[1].metadata["retrieval_score"] == 0.5


def test_stashes_page_display_into_each_documents_metadata():
    """build_conversational_chain()'s document_prompt (src/pipeline.py)
    shows this string to the LLM alongside each chunk, so it's what makes
    'what page is that on?' answerable at all instead of the model having
    no way to know or guessing -- see resolve_page_display()'s docstring
    in src/helper.py."""
    with_label = Document(page_content="x", metadata={"source": "a.pdf", "page_label": "xii"})
    without_label = Document(page_content="y", metadata={"source": "b.pdf", "page": 4})
    no_page_at_all = Document(page_content="z", metadata={"source": "c.pdf"})
    vs = FakeVectorStore(results={NS_A: [(with_label, 0.9), (without_label, 0.8), (no_page_at_all, 0.7)]})

    results = make_retriever(vs, min_similarity=0.0).invoke("some question")

    by_source = {d.metadata["source"]: d.metadata["page_display"] for d in results}
    assert by_source["a.pdf"] == "xii"
    assert by_source["b.pdf"] == "5"  # 0-indexed page 4 -> displayed as 5
    assert by_source["c.pdf"] == "not given"  # never a blank/None shown to the LLM


# ---------------------------------------------------------------------------
# doc_ids — scoping a search to a chosen subset of the knowledge base
# (search(), the entry point build_conversational_chain() actually calls —
# see that method's own docstring in src/pipeline.py for why this isn't
# routed through .invoke() instead)
# ---------------------------------------------------------------------------


def test_search_without_doc_ids_applies_no_filter():
    vs = FakeVectorStore(results={})
    make_retriever(vs).search("some question", doc_ids=None)

    assert all(call[3] is None for call in vs.calls)


def test_search_with_empty_doc_ids_applies_no_filter():
    """[] and None are treated identically -- both mean "search
    everything" (see _normalize_document_ids()'s docstring in app.py,
    which relies on exactly this to collapse both onto one cache scope)."""
    vs = FakeVectorStore(results={})
    make_retriever(vs).search("some question", doc_ids=[])

    assert all(call[3] is None for call in vs.calls)


def test_search_with_doc_ids_builds_a_pinecone_in_filter():
    vs = FakeVectorStore(results={})
    make_retriever(vs).search("some question", doc_ids=["doc-a", "doc-b"])

    assert all(call[3] == {"doc_id": {"$in": ["doc-a", "doc-b"]}} for call in vs.calls)


def test_search_with_doc_ids_actually_narrows_which_documents_come_back():
    from_a = doc("belongs to doc-a", "a.pdf", doc_id="doc-a")
    from_b = doc("belongs to doc-b", "b.pdf", doc_id="doc-b")
    vs = FakeVectorStore(results={NS_A: [(from_a, 0.9), (from_b, 0.8)]})

    results = make_retriever(vs).search("some question", doc_ids=["doc-a"])

    assert results == [from_a]


def test_search_with_doc_ids_matching_nothing_returns_empty_not_an_error():
    from_a = doc("belongs to doc-a", "a.pdf", doc_id="doc-a")
    vs = FakeVectorStore(results={NS_A: [(from_a, 0.9)]})

    results = make_retriever(vs).search("some question", doc_ids=["doc-that-does-not-exist"])

    assert results == []


def test_invoke_never_applies_a_filter_even_with_doc_id_tagged_documents():
    """The standard BaseRetriever.invoke() path (used by eval/run_eval.py
    and anything else treating this as a generic retriever) always
    searches every document -- only the explicit search(doc_ids=...) call
    build_conversational_chain() makes can narrow it. See
    _get_relevant_documents()'s one-line implementation in
    src/pipeline.py."""
    from_a = doc("belongs to doc-a", "a.pdf", doc_id="doc-a")
    from_b = doc("belongs to doc-b", "b.pdf", doc_id="doc-b")
    vs = FakeVectorStore(results={NS_A: [(from_a, 0.9), (from_b, 0.8)]})

    results = make_retriever(vs).invoke("some question")

    assert {d.metadata["source"] for d in results} == {"a.pdf", "b.pdf"}
    assert all(call[3] is None for call in vs.calls)


def test_search_respects_min_similarity_and_k_total_same_as_invoke():
    """doc_ids narrows *which* documents are eligible; it doesn't bypass
    the usual relevance floor or result cap."""
    relevant = doc("relevant", "a.pdf", doc_id="doc-a")
    irrelevant = doc("irrelevant", "a.pdf", doc_id="doc-a")
    vs = FakeVectorStore(results={NS_A: [(relevant, 0.6), (irrelevant, 0.05)]})

    results = make_retriever(vs, min_similarity=0.2).search("some question", doc_ids=["doc-a"])

    assert results == [relevant]


# ---------------------------------------------------------------------------
# Reranking (CombinedMedicalRetriever.reranker) — a sentence-transformers
# CrossEncoder in production (src/helper.py's download_reranker, or
# inference_service/app.py's over HTTP -- see src/helper.py's
# RemoteReranker), a tiny fake here so this logic is testable with no
# model download or network call at all.
# ---------------------------------------------------------------------------


class FakeReranker:
    """scores_by_content: {page_content: cross-encoder score}. Mirrors
    sentence-transformers CrossEncoder.predict(pairs) -> scores (and
    src/helper.py's RemoteReranker.predict(), which has the identical
    signature for exactly this reason)."""

    def __init__(self, scores_by_content, raises=False):
        self.scores_by_content = scores_by_content
        self.raises = raises
        self.last_pairs = None

    def predict(self, pairs):
        self.last_pairs = pairs
        if self.raises:
            raise RuntimeError("simulated cross-encoder failure")
        return [self.scores_by_content.get(passage, 0.0) for _, passage in pairs]


def test_reranker_can_reorder_results_by_a_different_signal_than_cosine():
    # Embedding similarity ranks "b" highest, but the (fake) cross-encoder
    # -- which sees the actual question text, unlike this test's cosine
    # scores -- thinks "a" is the true best match. The final order should
    # follow the reranker, proving it's actually driving selection.
    doc_a = doc("content a")
    doc_b = doc("content b")
    doc_c = doc("content c")
    vs = FakeVectorStore(results={NS_A: [(doc_a, 0.5), (doc_b, 0.9), (doc_c, 0.3)]})
    reranker = FakeReranker({"content a": 9.0, "content b": 1.0, "content c": 5.0})

    results = make_retriever(vs, k_per_namespace=10, k_total=2, reranker=reranker).invoke("some question")

    assert [d.page_content for d in results] == ["content a", "content c"]


def test_reranker_receives_the_actual_query_text():
    vs = FakeVectorStore(results={NS_A: [(doc("content a"), 0.5)]})
    reranker = FakeReranker({"content a": 1.0})

    make_retriever(vs, reranker=reranker).invoke("what is asthma")

    assert reranker.last_pairs == [("what is asthma", "content a")]


def test_reranked_documents_still_carry_their_original_cosine_score():
    # The displayed retrieval_score stays the embedding cosine similarity
    # even when the cross-encoder picked the order -- see _rerank()'s
    # docstring for why (cross-encoder logits aren't on a [0,1] scale).
    doc_a = doc("content a")
    vs = FakeVectorStore(results={NS_A: [(doc_a, 0.73)]})
    reranker = FakeReranker({"content a": 9999.0})  # wildly different scale

    [result] = make_retriever(vs, reranker=reranker).invoke("some question")

    assert result.metadata["retrieval_score"] == 0.73


def test_reranker_still_respects_min_similarity_floor_before_reranking():
    relevant = doc("relevant content")
    irrelevant = doc("irrelevant content")
    vs = FakeVectorStore(results={NS_A: [(relevant, 0.6), (irrelevant, 0.05)]})
    # Reranker would love the "irrelevant" one, but it should never see it
    # -- the similarity floor is applied before reranking, not after.
    reranker = FakeReranker({"relevant content": 1.0, "irrelevant content": 100.0})

    results = make_retriever(vs, reranker=reranker, min_similarity=0.2).invoke("some question")

    assert [d.page_content for d in results] == ["relevant content"]


def test_reranker_pulls_a_wider_pool_than_k_per_namespace():
    docs = [(doc(f"content {i}"), 0.9 - i * 0.01) for i in range(10)]
    vs = FakeVectorStore(results={NS_A: docs})
    reranker = FakeReranker({f"content {i}": float(i) for i in range(10)})

    make_retriever(vs, k_per_namespace=3, reranker=reranker, rerank_pool_size=8).invoke("q")

    # k_per_namespace=3 would normally cap the fetch at 3 -- reranking
    # should have asked the (fake) store for up to rerank_pool_size=8
    # instead, so the reranker actually has something to choose between.
    assert vs.calls[0][1] == 8


def test_no_reranker_falls_back_to_plain_cosine_ordering():
    doc_a = doc("content a")
    doc_b = doc("content b")
    vs = FakeVectorStore(results={NS_A: [(doc_a, 0.5), (doc_b, 0.9)]})

    results = make_retriever(vs, reranker=None).invoke("some question")

    assert [d.page_content for d in results] == ["content b", "content a"]


def test_reranker_failure_falls_back_to_cosine_ordering_instead_of_erroring():
    doc_a = doc("content a")
    doc_b = doc("content b")
    vs = FakeVectorStore(results={NS_A: [(doc_a, 0.5), (doc_b, 0.9)]})
    reranker = FakeReranker({}, raises=True)

    results = make_retriever(vs, reranker=reranker).invoke("some question")

    # Falls back to the cosine-similarity order rather than raising or
    # returning nothing -- a reranker hiccup shouldn't cost the user an
    # answer entirely.
    assert [d.page_content for d in results] == ["content b", "content a"]


def test_reranker_defaults_to_none():
    vs = FakeVectorStore()
    retriever = make_retriever(vs)
    assert retriever.reranker is None
