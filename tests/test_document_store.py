from src.document_store import InMemoryDocumentStore


def test_init_db_is_a_no_op():
    store = InMemoryDocumentStore()
    store.init_db()  # should not raise


def test_list_documents_empty_initially():
    store = InMemoryDocumentStore()
    assert store.list_documents() == []


def test_add_and_get_document_round_trips_vector_ids():
    store = InMemoryDocumentStore()
    store.add_document(
        doc_id="abc123",
        filename="report.pdf",
        chunk_count=5,
        page_count=2,
        vector_ids=["abc123::0", "abc123::1", "abc123::2", "abc123::3", "abc123::4"],
    )

    doc = store.get_document("abc123")
    assert doc["filename"] == "report.pdf"
    assert doc["chunk_count"] == 5
    assert doc["page_count"] == 2
    assert doc["vector_ids"] == ["abc123::0", "abc123::1", "abc123::2", "abc123::3", "abc123::4"]
    assert "uploaded_at" in doc


def test_get_document_returns_none_for_unknown_id():
    store = InMemoryDocumentStore()
    assert store.get_document("does-not-exist") is None


def test_list_documents_excludes_vector_ids():
    """vector_ids are only ever needed internally (to delete Pinecone
    vectors) -- the GET /documents route hands list_documents() straight
    to the frontend, so leaking the full id list there would be pointless
    payload bloat for something the UI never uses."""
    store = InMemoryDocumentStore()
    store.add_document("abc123", "report.pdf", 5, 2, ["abc123::0"])

    [doc] = store.list_documents()
    assert "vector_ids" not in doc
    assert doc["id"] == "abc123"


def test_list_documents_sorted_most_recent_first():
    store = InMemoryDocumentStore()
    store.add_document("first", "a.pdf", 1, 1, ["first::0"])
    store.add_document("second", "b.pdf", 1, 1, ["second::0"])
    store.add_document("third", "c.pdf", 1, 1, ["third::0"])

    ids_in_order = [d["id"] for d in store.list_documents()]
    assert ids_in_order == ["third", "second", "first"]


def test_delete_document_removes_it():
    store = InMemoryDocumentStore()
    store.add_document("abc123", "report.pdf", 5, 2, ["abc123::0"])
    store.delete_document("abc123")
    assert store.get_document("abc123") is None
    assert store.list_documents() == []


def test_delete_document_is_safe_for_unknown_id():
    store = InMemoryDocumentStore()
    store.delete_document("never-existed")  # should not raise


def test_multiple_documents_are_independent():
    store = InMemoryDocumentStore()
    store.add_document("doc1", "a.pdf", 3, 1, ["doc1::0", "doc1::1", "doc1::2"])
    store.add_document("doc2", "b.pdf", 7, 3, ["doc2::0"])

    store.delete_document("doc1")

    assert store.get_document("doc1") is None
    remaining = store.get_document("doc2")
    assert remaining["filename"] == "b.pdf"
    assert remaining["chunk_count"] == 7


# ---------------------------------------------------------------------------
# mark_deleted / was_deleted — the deletion tombstone seed_data.py checks
# before ever (re-)adding a data/seed/*.pdf document, so an intentional
# deletion from the UI can't be silently undone by the app's own next
# restart. See this module's docstring in src/document_store.py.
# ---------------------------------------------------------------------------


def test_was_deleted_is_false_for_an_id_never_marked():
    store = InMemoryDocumentStore()
    assert store.was_deleted("seed-diabetes") is False


def test_mark_deleted_then_was_deleted_is_true():
    store = InMemoryDocumentStore()
    store.mark_deleted("seed-diabetes")
    assert store.was_deleted("seed-diabetes") is True


def test_mark_deleted_is_independent_per_id():
    store = InMemoryDocumentStore()
    store.mark_deleted("seed-diabetes")
    assert store.was_deleted("seed-asthma") is False


def test_mark_deleted_survives_the_document_no_longer_existing():
    """The realistic order of operations (see app.py's DELETE
    /documents/<id> route): the manifest row is gone by the time anyone
    would check was_deleted() again -- the tombstone must not depend on
    the document still being present in list_documents()/get_document()."""
    store = InMemoryDocumentStore()
    store.add_document("seed-diabetes", "diabetes.pdf", 4, 4, ["seed-diabetes::0"])
    store.delete_document("seed-diabetes")
    store.mark_deleted("seed-diabetes")

    assert store.get_document("seed-diabetes") is None
    assert store.was_deleted("seed-diabetes") is True


def test_mark_deleted_twice_is_safe():
    store = InMemoryDocumentStore()
    store.mark_deleted("seed-diabetes")
    store.mark_deleted("seed-diabetes")  # should not raise
    assert store.was_deleted("seed-diabetes") is True
