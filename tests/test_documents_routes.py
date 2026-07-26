"""
Tests for the /documents/* routes (upload, list, delete) and for the
"no relevant context -> honest answer" short-circuit in /get and
/get/stream that those routes exist to feed.

Everything here runs against fakes -- no real Pinecone, Groq, or
Postgres, no network, no API keys. Uploaded PDFs are small *real* PDFs
(pypdf-built slices of one of the bundled data/seed/ documents), so the
upload route's own file-handling and ingestion logic runs for real; only
the vector store on the other end (pipeline.vectorstore) is a fake.
"""

import io
import os

import pytest
from pypdf import PdfReader, PdfWriter

from app import NO_CONTEXT_MESSAGE, create_app
from src.cache import SemanticCache
from src.document_store import InMemoryDocumentStore
from src.documents import DOCUMENTS_NAMESPACE
from src.telemetry import InMemoryTelemetry


# ---------------------------------------------------------------------------
# Fakes — same shapes/spirit as tests/test_app.py, kept local to this file
# so the two test files stay independent of each other.
# ---------------------------------------------------------------------------


class FakeEmbeddings:
    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


class FakeDoc:
    def __init__(self, content, source="reference.pdf", page=41):
        self.page_content = content
        self.metadata = {"source": source, "page": page}


class FakeChain:
    def invoke(self, payload):
        return {"answer": "This is a fake grounded answer.", "context": [FakeDoc("relevant passage")]}

    def stream(self, payload):
        yield {"context": [FakeDoc("relevant passage")]}
        for word in ["This ", "is ", "a ", "fake ", "answer."]:
            yield {"answer": word}


class NoContextChain:
    """Simulates CombinedMedicalRetriever finding nothing above
    MIN_SIMILARITY -- context resolves to an empty list. stream() tracks
    whether anything ever asked it for a 2nd item after that, so tests
    can assert app.py's early-break actually happens instead of just
    checking the final response shape."""

    def __init__(self):
        self.stream_advanced_past_empty_context = False

    def invoke(self, payload):
        return {"answer": "SHOULD_NOT_REACH_USER", "context": []}

    def stream(self, payload):
        yield {"context": []}
        self.stream_advanced_past_empty_context = True
        yield {"answer": "SHOULD_NOT_REACH_USER_STREAMED"}


class FakeVectorStore:
    def __init__(self, raise_on_add=False, raise_on_delete=False):
        self.added = []  # (documents, ids, namespace)
        self.deleted = []  # (ids, namespace)
        self.raise_on_add = raise_on_add
        self.raise_on_delete = raise_on_delete

    def add_documents(self, documents, ids=None, namespace=None):
        if self.raise_on_add:
            raise RuntimeError("simulated Pinecone add failure")
        self.added.append((documents, ids, namespace))
        return ids

    def delete(self, ids=None, namespace=None):
        if self.raise_on_delete:
            raise RuntimeError("simulated Pinecone delete failure")
        self.deleted.append((ids, namespace))


class CountingChain:
    """Returns a different answer every time it's actually invoked, and
    tracks how many times that's happened -- used to prove a cache was
    genuinely bypassed (invoke_count goes up) rather than just checking
    the final answer text, which a stale cache could coincidentally still
    get right."""

    def __init__(self):
        self.invoke_count = 0

    def _answer(self):
        self.invoke_count += 1
        return f"Answer number {self.invoke_count}"

    def invoke(self, payload):
        return {"answer": self._answer(), "context": [FakeDoc("relevant passage")]}

    def stream(self, payload):
        yield {"context": [FakeDoc("relevant passage")]}
        yield {"answer": self._answer()}


class FakePipeline:
    def __init__(self, chain=None, vectorstore=None):
        self.embeddings = FakeEmbeddings()
        self.chain = chain or FakeChain()
        self.vectorstore = vectorstore if vectorstore is not None else FakeVectorStore()


@pytest.fixture
def small_real_pdf_bytes():
    reader = PdfReader(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "seed", "diabetes.pdf")
    )
    writer = PdfWriter()
    for i in range(1, 4):
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def blank_pdf_bytes():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def make_client(pipeline=None, document_store=None, upload_dir=None, monkeypatch=None):
    pipeline = pipeline or FakePipeline()
    document_store = document_store if document_store is not None else InMemoryDocumentStore()
    if upload_dir is not None and monkeypatch is not None:
        import src.documents as documents_module

        monkeypatch.setattr(documents_module, "UPLOAD_DIR", str(upload_dir))
    app = create_app(
        pipeline=pipeline, cache=SemanticCache(), telemetry=InMemoryTelemetry(), document_store=document_store
    )
    app.config["TESTING"] = True
    return app.test_client(), pipeline, document_store


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------


def test_upload_rejects_missing_file(tmp_path, monkeypatch):
    client, _, _ = make_client(upload_dir=tmp_path, monkeypatch=monkeypatch)
    res = client.post("/documents/upload", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    assert res.get_json()["error"] == "no_file"


def test_upload_rejects_non_pdf(tmp_path, monkeypatch):
    client, _, _ = make_client(upload_dir=tmp_path, monkeypatch=monkeypatch)
    res = client.post(
        "/documents/upload",
        data={"file": (io.BytesIO(b"hello"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert res.get_json()["error"] == "invalid_file"


def test_upload_accepts_a_real_pdf_and_indexes_it(small_real_pdf_bytes, tmp_path, monkeypatch):
    client, pipeline, document_store = make_client(upload_dir=tmp_path, monkeypatch=monkeypatch)

    res = client.post(
        "/documents/upload",
        data={"file": (io.BytesIO(small_real_pdf_bytes), "my_report.pdf")},
        content_type="multipart/form-data",
    )

    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["document"]["filename"] == "my_report.pdf"
    assert data["document"]["chunk_count"] > 0

    # Actually reached the vector store, in the shared documents
    # namespace, with matching ids -- not just returned a nice-looking
    # response.
    assert len(pipeline.vectorstore.added) == 1
    documents, ids, namespace = pipeline.vectorstore.added[0]
    assert namespace == DOCUMENTS_NAMESPACE
    assert len(documents) == len(ids) == data["document"]["chunk_count"]

    # And the manifest actually has it, for GET /documents and DELETE.
    stored = document_store.get_document(data["document"]["id"])
    assert stored["filename"] == "my_report.pdf"
    assert stored["vector_ids"] == ids


def test_upload_failure_cleans_up_the_saved_file_and_manifest(small_real_pdf_bytes, tmp_path, monkeypatch):
    pipeline = FakePipeline(vectorstore=FakeVectorStore(raise_on_add=True))
    client, _, document_store = make_client(pipeline=pipeline, upload_dir=tmp_path, monkeypatch=monkeypatch)

    res = client.post(
        "/documents/upload",
        data={"file": (io.BytesIO(small_real_pdf_bytes), "report.pdf")},
        content_type="multipart/form-data",
    )

    assert res.status_code == 500
    assert res.get_json()["error"] == "ingest_failed"
    assert document_store.list_documents() == []
    assert os.listdir(tmp_path) == []  # no orphaned file left behind


def test_upload_of_a_pdf_with_no_extractable_text_is_rejected(blank_pdf_bytes, tmp_path, monkeypatch):
    client, pipeline, document_store = make_client(upload_dir=tmp_path, monkeypatch=monkeypatch)

    res = client.post(
        "/documents/upload",
        data={"file": (io.BytesIO(blank_pdf_bytes), "scanned.pdf")},
        content_type="multipart/form-data",
    )

    assert res.status_code == 500
    assert res.get_json()["error"] == "ingest_failed"
    assert "extractable text" in res.get_json()["message"]
    assert pipeline.vectorstore.added == []
    assert document_store.list_documents() == []


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


def test_list_documents_reflects_the_manifest(tmp_path, monkeypatch):
    document_store = InMemoryDocumentStore()
    document_store.add_document("abc123", "report.pdf", 5, 2, ["abc123::0"])
    client, _, _ = make_client(document_store=document_store, upload_dir=tmp_path, monkeypatch=monkeypatch)

    res = client.get("/documents")

    assert res.status_code == 200
    docs = res.get_json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "report.pdf"
    assert "vector_ids" not in docs[0]  # internal detail, not for the frontend


# ---------------------------------------------------------------------------
# DELETE /documents/<id>
# ---------------------------------------------------------------------------


def test_delete_removes_vectors_and_manifest_entry(tmp_path, monkeypatch):
    document_store = InMemoryDocumentStore()
    document_store.add_document("abc123", "report.pdf", 3, 1, ["abc123::0", "abc123::1", "abc123::2"])
    client, pipeline, document_store = make_client(document_store=document_store, upload_dir=tmp_path, monkeypatch=monkeypatch)

    res = client.delete("/documents/abc123")

    assert res.status_code == 200
    assert res.get_json()["ok"] is True
    assert pipeline.vectorstore.deleted == [(["abc123::0", "abc123::1", "abc123::2"], DOCUMENTS_NAMESPACE)]
    assert document_store.get_document("abc123") is None
    # Also tombstoned -- see src/document_store.py's module docstring for
    # why this matters specifically for a *seeded* document's id: without
    # it, seed_data.py would have no way to tell "deliberately removed"
    # apart from "never seeded yet" on the app's next restart, and would
    # silently re-add it.
    assert document_store.was_deleted("abc123") is True


def test_delete_unknown_id_returns_404(tmp_path, monkeypatch):
    client, _, _ = make_client(upload_dir=tmp_path, monkeypatch=monkeypatch)
    res = client.delete("/documents/does-not-exist")
    assert res.status_code == 404
    assert res.get_json()["error"] == "not_found"


def test_delete_still_cleans_up_manifest_even_if_pinecone_delete_fails(tmp_path, monkeypatch):
    document_store = InMemoryDocumentStore()
    document_store.add_document("abc123", "report.pdf", 1, 1, ["abc123::0"])
    pipeline = FakePipeline(vectorstore=FakeVectorStore(raise_on_delete=True))
    client, _, document_store = make_client(
        pipeline=pipeline, document_store=document_store, upload_dir=tmp_path, monkeypatch=monkeypatch
    )

    res = client.delete("/documents/abc123")

    # The user asked to remove it -- a Pinecone-side hiccup shouldn't
    # leave it permanently stuck in their document list with no way to
    # retry (see the comment in app.py's delete_document route).
    assert res.status_code == 200
    assert document_store.get_document("abc123") is None


# ---------------------------------------------------------------------------
# The "no relevant context" short circuit in /get and /get/stream
# ---------------------------------------------------------------------------


def test_get_overrides_the_answer_when_context_is_empty(tmp_path, monkeypatch):
    pipeline = FakePipeline(chain=NoContextChain())
    client, _, _ = make_client(pipeline=pipeline, upload_dir=tmp_path, monkeypatch=monkeypatch)

    res = client.post("/get", json={"message": "what's the capital of France?", "history": []})
    data = res.get_json()

    assert data["answer"] == NO_CONTEXT_MESSAGE
    assert data["sources"] == []
    assert data["no_info"] is True
    assert "SHOULD_NOT_REACH_USER" not in data["answer"]


def test_stream_short_circuits_on_empty_context_without_advancing_the_chain(tmp_path, monkeypatch):
    chain = NoContextChain()
    pipeline = FakePipeline(chain=chain)
    client, _, _ = make_client(pipeline=pipeline, upload_dir=tmp_path, monkeypatch=monkeypatch)

    res = client.post("/get/stream", json={"message": "what's the capital of France?", "history": []})
    raw = res.get_data(as_text=True)

    assert NO_CONTEXT_MESSAGE in raw
    assert '"no_info": true' in raw
    assert "SHOULD_NOT_REACH_USER_STREAMED" not in raw
    # The real point of this test: app.py must never pull a second item
    # from chain.stream() once it sees an empty context chunk.
    assert chain.stream_advanced_past_empty_context is False


def test_normal_answer_is_not_flagged_as_no_info(tmp_path, monkeypatch):
    client, _, _ = make_client(upload_dir=tmp_path, monkeypatch=monkeypatch)  # default FakeChain has real context
    res = client.post("/get", json={"message": "What is asthma?", "history": []})
    assert res.get_json()["no_info"] is False


def test_no_info_answer_is_cached_and_flagged_on_cache_hit(tmp_path, monkeypatch):
    pipeline = FakePipeline(chain=NoContextChain())
    client, _, _ = make_client(pipeline=pipeline, upload_dir=tmp_path, monkeypatch=monkeypatch)

    first = client.post("/get", json={"message": "what's the capital of France?", "history": []}).get_json()
    second = client.post("/get", json={"message": "what's the capital of France?", "history": []}).get_json()

    assert first["no_info"] is True
    assert second["cached"] is True
    assert second["no_info"] is True
    assert second["answer"] == NO_CONTEXT_MESSAGE


# ---------------------------------------------------------------------------
# document_ids — scoping a question to a chosen subset of the knowledge base
# ---------------------------------------------------------------------------


class RecordingChain:
    """Records the full payload it was invoked/streamed with, so a test
    can assert *what the retriever was actually asked to search*, not
    just what came back."""

    def __init__(self):
        self.invoke_payloads = []
        self.stream_payloads = []

    def invoke(self, payload):
        self.invoke_payloads.append(payload)
        return {"answer": "answer", "context": [FakeDoc("relevant passage")]}

    def stream(self, payload):
        self.stream_payloads.append(payload)
        yield {"context": [FakeDoc("relevant passage")]}
        yield {"answer": "answer"}


def test_get_threads_document_ids_into_the_chain_payload(tmp_path, monkeypatch):
    chain = RecordingChain()
    client, _, _ = make_client(pipeline=FakePipeline(chain=chain), upload_dir=tmp_path, monkeypatch=monkeypatch)

    client.post("/get", json={"message": "What is diabetes?", "history": [], "document_ids": ["doc-a", "doc-b"]})

    assert chain.invoke_payloads[0]["document_ids"] == ["doc-a", "doc-b"]


def test_get_omitting_document_ids_passes_none_through(tmp_path, monkeypatch):
    """A request shaped exactly like one from before this feature existed
    (no document_ids key at all -- e.g. eval/run_eval.py) must still
    search every document, not an empty/no-op selection."""
    chain = RecordingChain()
    client, _, _ = make_client(pipeline=FakePipeline(chain=chain), upload_dir=tmp_path, monkeypatch=monkeypatch)

    client.post("/get", json={"message": "What is diabetes?", "history": []})

    assert chain.invoke_payloads[0]["document_ids"] is None


def test_get_with_junk_document_ids_falls_back_to_none(tmp_path, monkeypatch):
    """Malformed document_ids (wrong type, empty list, non-string entries)
    degrade to "search everything" rather than a 400 -- see
    _normalize_document_ids()'s docstring in app.py."""
    chain = RecordingChain()
    client, _, _ = make_client(pipeline=FakePipeline(chain=chain), upload_dir=tmp_path, monkeypatch=monkeypatch)

    client.post("/get", json={"message": "q1", "history": [], "document_ids": "not-a-list"})
    client.post("/get", json={"message": "q2", "history": [], "document_ids": []})
    client.post("/get", json={"message": "q3", "history": [], "document_ids": [123, None, "  "]})

    assert [p["document_ids"] for p in chain.invoke_payloads] == [None, None, None]


def test_stream_threads_document_ids_into_the_chain_payload(tmp_path, monkeypatch):
    chain = RecordingChain()
    client, _, _ = make_client(pipeline=FakePipeline(chain=chain), upload_dir=tmp_path, monkeypatch=monkeypatch)

    client.post(
        "/get/stream",
        json={"message": "What is diabetes?", "history": [], "document_ids": ["doc-a"]},
    )

    assert chain.stream_payloads[0]["document_ids"] == ["doc-a"]


def test_same_question_different_document_selection_does_not_share_a_cache_hit(tmp_path, monkeypatch):
    """Answering "yes" for {doc-a} and "no" for {doc-b} are different
    facts about different documents -- the semantic cache must never
    blur them together just because the question text embeds the same."""
    chain = CountingChain()
    client, _, _ = make_client(pipeline=FakePipeline(chain=chain), upload_dir=tmp_path, monkeypatch=monkeypatch)

    first = client.post(
        "/get", json={"message": "What does it say?", "history": [], "document_ids": ["doc-a"]}
    ).get_json()
    second = client.post(
        "/get", json={"message": "What does it say?", "history": [], "document_ids": ["doc-b"]}
    ).get_json()

    assert chain.invoke_count == 2  # both actually ran -- neither was a cache hit off the other
    assert first.get("cached") is not True
    assert second.get("cached") is not True
    assert first["answer"] != second["answer"]


def test_same_question_same_document_selection_does_share_a_cache_hit(tmp_path, monkeypatch):
    chain = CountingChain()
    client, _, _ = make_client(pipeline=FakePipeline(chain=chain), upload_dir=tmp_path, monkeypatch=monkeypatch)

    first = client.post(
        "/get", json={"message": "What does it say?", "history": [], "document_ids": ["doc-a", "doc-b"]}
    ).get_json()
    second = client.post(
        "/get", json={"message": "What does it say?", "history": [], "document_ids": ["doc-b", "doc-a"]}
    ).get_json()  # same set, different click/selection order

    assert chain.invoke_count == 1  # second call was a cache hit
    assert second["cached"] is True
    assert second["answer"] == first["answer"]


# ---------------------------------------------------------------------------
# /stats — document knowledge-base counts and the dashboard's daily series
# ---------------------------------------------------------------------------


def test_stats_includes_document_counts_and_daily_series(tmp_path, monkeypatch):
    document_store = InMemoryDocumentStore()
    document_store.add_document("doc1", "a.pdf", 5, 2, ["doc1::0"])
    document_store.add_document("doc2", "b.pdf", 3, 1, ["doc2::0"])
    client, _, _ = make_client(document_store=document_store, upload_dir=tmp_path, monkeypatch=monkeypatch)

    res = client.get("/stats")
    data = res.get_json()

    assert data["documents_indexed"] == 2
    assert data["chunks_indexed"] == 8  # 5 + 3
    assert isinstance(data["daily"], list)
    assert len(data["daily"]) == 14  # default window
    assert set(data["daily"][0].keys()) == {"date", "queries", "avg_ms", "cache_hits"}


def test_stats_document_counts_are_zero_with_no_uploads(tmp_path, monkeypatch):
    client, _, _ = make_client(upload_dir=tmp_path, monkeypatch=monkeypatch)
    data = client.get("/stats").get_json()
    assert data["documents_indexed"] == 0
    assert data["chunks_indexed"] == 0


# ---------------------------------------------------------------------------
# Regression tests: uploading or deleting a document must invalidate the
# semantic cache, or a stale answer (possibly citing a document that no
# longer exists, or missing one that now does) keeps getting served.
# ---------------------------------------------------------------------------


def test_uploading_a_document_invalidates_the_cache(small_real_pdf_bytes, tmp_path, monkeypatch):
    chain = CountingChain()
    pipeline = FakePipeline(chain=chain)
    client, _, _ = make_client(pipeline=pipeline, upload_dir=tmp_path, monkeypatch=monkeypatch)
    question = {"message": "What does my report say about X?", "history": []}

    first = client.post("/get", json=question).get_json()
    assert first["answer"] == "Answer number 1"

    cached_again = client.post("/get", json=question).get_json()
    assert cached_again["cached"] is True
    assert cached_again["answer"] == "Answer number 1"  # served from cache, chain not re-invoked
    assert chain.invoke_count == 1

    client.post(
        "/documents/upload",
        data={"file": (io.BytesIO(small_real_pdf_bytes), "report.pdf")},
        content_type="multipart/form-data",
    )

    after_upload = client.post("/get", json=question).get_json()
    assert after_upload["cached"] is False
    assert after_upload["answer"] == "Answer number 2"  # proves a fresh retrieval actually ran
    assert chain.invoke_count == 2


def test_deleting_a_document_invalidates_the_cache(tmp_path, monkeypatch):
    chain = CountingChain()
    pipeline = FakePipeline(chain=chain)
    document_store = InMemoryDocumentStore()
    document_store.add_document("abc123", "report.pdf", 3, 1, ["abc123::0", "abc123::1", "abc123::2"])
    client, _, _ = make_client(
        pipeline=pipeline, document_store=document_store, upload_dir=tmp_path, monkeypatch=monkeypatch
    )
    question = {"message": "What does my report say about X?", "history": []}

    first = client.post("/get", json=question).get_json()
    assert first["answer"] == "Answer number 1"
    assert client.post("/get", json=question).get_json()["cached"] is True
    assert chain.invoke_count == 1

    client.delete("/documents/abc123")

    after_delete = client.post("/get", json=question).get_json()
    assert after_delete["cached"] is False
    assert after_delete["answer"] == "Answer number 2"  # the pre-delete cached answer is gone
    assert chain.invoke_count == 2


def test_upload_invalidates_the_cache_on_the_streaming_path_too(small_real_pdf_bytes, tmp_path, monkeypatch):
    chain = CountingChain()
    pipeline = FakePipeline(chain=chain)
    client, _, _ = make_client(pipeline=pipeline, upload_dir=tmp_path, monkeypatch=monkeypatch)
    question = {"message": "What does my report say about X?", "history": []}

    client.post("/get/stream", json=question)
    assert chain.invoke_count == 1

    client.post(
        "/documents/upload",
        data={"file": (io.BytesIO(small_real_pdf_bytes), "report.pdf")},
        content_type="multipart/form-data",
    )

    raw = client.post("/get/stream", json=question).get_data(as_text=True)
    assert "Answer number 2" in raw
    assert '"cached": true' not in raw
    assert chain.invoke_count == 2


def test_failed_upload_does_not_invalidate_the_cache(tmp_path, monkeypatch):
    """An upload that's rejected (bad file) never touched the knowledge
    base -- clearing the cache anyway would just be needless churn."""
    chain = CountingChain()
    pipeline = FakePipeline(chain=chain)
    client, _, _ = make_client(pipeline=pipeline, upload_dir=tmp_path, monkeypatch=monkeypatch)
    question = {"message": "What does my report say about X?", "history": []}

    client.post("/get", json=question)
    client.post(
        "/documents/upload",
        data={"file": (io.BytesIO(b"not a pdf"), "notes.txt")},
        content_type="multipart/form-data",
    )

    still_cached = client.post("/get", json=question).get_json()
    assert still_cached["cached"] is True
    assert chain.invoke_count == 1


def test_delete_of_unknown_document_does_not_invalidate_the_cache(tmp_path, monkeypatch):
    chain = CountingChain()
    pipeline = FakePipeline(chain=chain)
    client, _, _ = make_client(pipeline=pipeline, upload_dir=tmp_path, monkeypatch=monkeypatch)
    question = {"message": "What does my report say about X?", "history": []}

    client.post("/get", json=question)
    client.delete("/documents/does-not-exist")  # 404s before ever reaching cache.clear()

    still_cached = client.post("/get", json=question).get_json()
    assert still_cached["cached"] is True
    assert chain.invoke_count == 1


def test_delete_invalidates_the_cache_even_if_pinecone_delete_fails(tmp_path, monkeypatch):
    chain = CountingChain()
    document_store = InMemoryDocumentStore()
    document_store.add_document("abc123", "report.pdf", 1, 1, ["abc123::0"])
    pipeline = FakePipeline(chain=chain, vectorstore=FakeVectorStore(raise_on_delete=True))
    client, _, _ = make_client(
        pipeline=pipeline, document_store=document_store, upload_dir=tmp_path, monkeypatch=monkeypatch
    )
    question = {"message": "What does my report say about X?", "history": []}

    client.post("/get", json=question)
    client.delete("/documents/abc123")  # Pinecone-side delete fails, but manifest cleanup still proceeds

    after_delete = client.post("/get", json=question).get_json()
    assert after_delete["cached"] is False
    assert chain.invoke_count == 2
