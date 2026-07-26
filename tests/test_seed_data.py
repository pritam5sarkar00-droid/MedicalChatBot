"""
Tests for seed_data.py — indexing data/seed/*.pdf into the knowledge base
on first startup (see seed_data.py's own module docstring for the full
picture, including why deterministic ids and the deletion tombstone in
src/document_store.py both exist).

Runs against the *real* data/seed/*.pdf files (small, real PDFs -- see
build_seed_pdfs.py) and a fake vectorstore + InMemoryDocumentStore, so
these tests exercise the real PDF-loading/chunking path with no network,
Pinecone, or API keys.
"""

import os

import pytest

from src.document_store import InMemoryDocumentStore
from src.documents import DOCUMENTS_NAMESPACE
from seed_data import SEED_DIR, _seed_doc_id, seed_default_documents


class FakeVectorStore:
    def __init__(self):
        self.added = []  # (documents, ids, namespace)

    def add_documents(self, documents, ids=None, namespace=None):
        self.added.append((documents, ids, namespace))


def test_seed_doc_id_is_deterministic():
    assert _seed_doc_id("diabetes.pdf") == _seed_doc_id("diabetes.pdf")


def test_seed_doc_id_is_a_readable_slug():
    assert _seed_doc_id("high-blood-pressure.pdf") == "seed-high-blood-pressure"


def test_seed_doc_id_normalizes_case_and_punctuation():
    assert _seed_doc_id("What Is Asthma.PDF") == "seed-what-is-asthma"


def test_seed_doc_id_stays_within_the_postgres_id_column_limit():
    """PostgresDocumentStore's id column is VARCHAR(32) (src/document_store.py)
    -- a long filename must not produce an id Postgres would reject at
    insert time."""
    long_name = "a-very-long-descriptive-filename-someone-might-reasonably-pick.pdf"
    assert len(_seed_doc_id(long_name)) <= 32


def test_seed_doc_id_matches_every_file_actually_bundled_in_data_seed():
    """Guards the specific failure mode that motivated shortening the
    original filenames (e.g. high-blood-pressure-and-older-adults.pdf) --
    see build_seed_pdfs.py and seed_data.py's docstring."""
    for filename in os.listdir(SEED_DIR):
        if filename.lower().endswith(".pdf"):
            assert len(_seed_doc_id(filename)) <= 32, filename


@pytest.fixture
def fresh_stores():
    return FakeVectorStore(), InMemoryDocumentStore()


def test_seeds_every_pdf_in_the_real_seed_directory(fresh_stores):
    vectorstore, document_store = fresh_stores

    seeded = seed_default_documents(vectorstore, document_store)

    real_pdf_count = len([f for f in os.listdir(SEED_DIR) if f.lower().endswith(".pdf")])
    assert seeded == real_pdf_count
    assert len(document_store.list_documents()) == real_pdf_count
    assert len(vectorstore.added) == real_pdf_count


def test_seeded_chunks_land_in_the_shared_documents_namespace(fresh_stores):
    vectorstore, document_store = fresh_stores
    seed_default_documents(vectorstore, document_store)

    assert all(namespace == DOCUMENTS_NAMESPACE for _, _, namespace in vectorstore.added)


def test_seeded_chunks_are_tagged_with_their_deterministic_doc_id(fresh_stores):
    vectorstore, document_store = fresh_stores
    seed_default_documents(vectorstore, document_store)

    for documents, ids, _ in vectorstore.added:
        doc_ids_in_metadata = {c.metadata["doc_id"] for c in documents}
        assert len(doc_ids_in_metadata) == 1  # every chunk from one file shares that file's doc_id
        [doc_id] = doc_ids_in_metadata
        assert all(i.startswith(f"{doc_id}::") for i in ids)


def test_seeding_twice_is_idempotent(fresh_stores):
    """The realistic case on a free-tier host that sleeps and cold-starts
    repeatedly: create_app() runs this on every single startup, so a
    no-op second run (no new Pinecone writes, no duplicate manifest rows)
    is the normal case, not an edge case."""
    vectorstore, document_store = fresh_stores
    first_pass = seed_default_documents(vectorstore, document_store)

    second_pass = seed_default_documents(vectorstore, document_store)

    assert first_pass > 0
    assert second_pass == 0
    assert len(document_store.list_documents()) == first_pass  # not doubled
    assert len(vectorstore.added) == first_pass  # no new Pinecone writes on the second pass


def test_a_deliberately_deleted_seed_document_is_not_reseeded(fresh_stores):
    """The behavior mark_deleted()/was_deleted() exist for (see src/
    document_store.py's module docstring): a person deletes one of the
    seeded documents from the sidebar, then the app restarts (a free-tier
    cold start, or just a redeploy) -- it must stay gone, not silently
    reappear because seed_data.py can no longer find its manifest row."""
    vectorstore, document_store = fresh_stores
    seed_default_documents(vectorstore, document_store)
    seeded_ids = [d["id"] for d in document_store.list_documents()]
    doomed_id = seeded_ids[0]

    document_store.delete_document(doomed_id)
    document_store.mark_deleted(doomed_id)  # what app.py's DELETE route does on every deletion

    reseeded = seed_default_documents(vectorstore, document_store)

    # Nothing comes back: the deleted one is skipped because it's
    # tombstoned, and every *other* seed document is skipped too, exactly
    # as it would be on any ordinary idempotent second run (see
    # test_seeding_twice_is_idempotent) -- it was never removed, so it's
    # still sitting in the manifest from the first pass.
    assert reseeded == 0
    assert document_store.get_document(doomed_id) is None
    assert len(document_store.list_documents()) == len(seeded_ids) - 1


def test_tombstone_alone_prevents_reseeding_even_with_an_otherwise_empty_manifest(fresh_stores):
    """The precise scenario the tombstone exists for: the manifest is
    empty (as if this were a fresh Postgres database that had only ever
    recorded the one deletion), but the tombstone table remembers --
    seed_data.py must still skip that one file, not treat an empty
    manifest as "nothing has ever been seeded, add everything back"."""
    vectorstore, document_store = fresh_stores
    real_files = sorted(f for f in os.listdir(SEED_DIR) if f.lower().endswith(".pdf"))
    doc_id = _seed_doc_id(real_files[0])
    document_store.mark_deleted(doc_id)  # manifest itself is still completely empty at this point

    seeded = seed_default_documents(vectorstore, document_store)

    assert seeded == len(real_files) - 1
    assert document_store.get_document(doc_id) is None


def test_missing_seed_directory_returns_zero_not_an_error(fresh_stores, tmp_path):
    vectorstore, document_store = fresh_stores
    empty_dir = str(tmp_path / "does-not-exist")

    result = seed_default_documents(vectorstore, document_store, seed_dir=empty_dir)

    assert result == 0


def test_non_pdf_files_in_the_seed_directory_are_ignored(fresh_stores, tmp_path):
    vectorstore, document_store = fresh_stores
    (tmp_path / "README.md").write_text("not a pdf")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x01")

    result = seed_default_documents(vectorstore, document_store, seed_dir=str(tmp_path))

    assert result == 0
    assert document_store.list_documents() == []


def test_one_corrupt_pdf_does_not_block_the_others_in_the_same_directory(fresh_stores, tmp_path):
    import shutil

    real_files = [f for f in os.listdir(SEED_DIR) if f.lower().endswith(".pdf")]
    shutil.copy(os.path.join(SEED_DIR, real_files[0]), tmp_path / real_files[0])
    (tmp_path / "corrupt.pdf").write_bytes(b"this is not actually a pdf")

    vectorstore, document_store = fresh_stores
    result = seed_default_documents(vectorstore, document_store, seed_dir=str(tmp_path))

    assert result == 1  # the one real PDF, despite the corrupt one sitting right next to it
    assert document_store.list_documents()[0]["filename"] == real_files[0]


def test_already_present_filename_is_skipped_without_touching_the_vectorstore(fresh_stores):
    """A filename that's already in the manifest (whichever way it got
    there) is left alone -- seed_default_documents() only ever adds,
    never re-embeds or overwrites an existing entry."""
    vectorstore, document_store = fresh_stores
    real_files = sorted(f for f in os.listdir(SEED_DIR) if f.lower().endswith(".pdf"))
    already_there = real_files[0]
    document_store.add_document(_seed_doc_id(already_there), already_there, 1, 1, ["x::0"])

    seed_default_documents(vectorstore, document_store)

    # Every *other* real seed file was still indexed -- only the
    # already-present one was skipped, so the vectorstore was never asked
    # to add anything for it.
    touched_namespaces_doc_ids = {
        c.metadata["doc_id"] for documents, _, _ in vectorstore.added for c in documents
    }
    assert _seed_doc_id(already_there) not in touched_namespaces_doc_ids
