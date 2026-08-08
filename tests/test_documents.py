"""
Tests for src/documents.py — upload validation and single-PDF ingestion.

These build a small *real* PDF on the fly (a 3-page slice of one of the
bundled data/seed/*.pdf files, via pypdf) rather than mocking PyPDFLoader,
so the tests exercise the actual text-extraction + chunking path a real
upload would go through. No network, no Pinecone, no Groq, no API keys —
pypdf and PyPDFLoader both work entirely on local files.
"""

import io
import os

import pytest
from pypdf import PdfReader, PdfWriter
from werkzeug.datastructures import FileStorage

import src.documents as documents
from src.documents import (
    InvalidUpload,
    find_document_file_path,
    ingest_pdf,
    remove_uploaded_file,
    save_and_validate,
)
from src.helper import CHUNK_SIZE


@pytest.fixture
def small_real_pdf_bytes():
    """A small, real, valid PDF with genuine extractable text -- pages 2-4
    of one of the bundled data/seed/ documents, repackaged as a
    standalone file. diabetes.pdf specifically, because it (like every
    seed PDF) has at least 4 pages -- see build_seed_pdfs.py -- which
    this fixture's pages[1:4] slice requires."""
    seed_pdf = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "seed", "diabetes.pdf"
    )
    reader = PdfReader(seed_pdf)
    writer = PdfWriter()
    for i in range(1, 4):
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path, monkeypatch):
    """Redirects UPLOAD_DIR to a pytest tmp_path for every test in this
    file, so nothing ever gets written under the real project's
    data/uploads/ during a test run."""
    monkeypatch.setattr(documents, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


def make_file_storage(content: bytes, filename: str) -> FileStorage:
    return FileStorage(stream=io.BytesIO(content), filename=filename, content_type="application/pdf")


# ---------------------------------------------------------------------------
# save_and_validate
# ---------------------------------------------------------------------------


def test_accepts_a_real_pdf_and_saves_it(small_real_pdf_bytes, isolated_upload_dir):
    doc_id, saved_path, display_name = save_and_validate(make_file_storage(small_real_pdf_bytes, "my_report.pdf"))

    assert display_name == "my_report.pdf"
    assert len(doc_id) == 12
    assert os.path.exists(saved_path)
    assert os.path.dirname(saved_path) == str(isolated_upload_dir)
    with open(saved_path, "rb") as f:
        assert f.read() == small_real_pdf_bytes


def test_rejects_non_pdf_extension(small_real_pdf_bytes):
    with pytest.raises(InvalidUpload):
        save_and_validate(make_file_storage(small_real_pdf_bytes, "notes.txt"))


def test_rejects_empty_file():
    with pytest.raises(InvalidUpload):
        save_and_validate(make_file_storage(b"", "empty.pdf"))


def test_rejects_spoofed_pdf_extension_on_non_pdf_content():
    # A .pdf extension with no %PDF- magic bytes -- e.g. someone renaming
    # a .exe or a .txt to bypass an extension-only check.
    with pytest.raises(InvalidUpload):
        save_and_validate(make_file_storage(b"this is definitely not a pdf", "totally_a_pdf.pdf"))


def test_rejects_oversized_file(small_real_pdf_bytes, monkeypatch):
    monkeypatch.setattr(documents, "MAX_UPLOAD_BYTES", 10)  # tiny limit, no need to allocate 15MB in a test
    with pytest.raises(InvalidUpload):
        save_and_validate(make_file_storage(small_real_pdf_bytes, "report.pdf"))


def test_sanitizes_path_traversal_in_filename(small_real_pdf_bytes):
    _, saved_path, display_name = save_and_validate(make_file_storage(small_real_pdf_bytes, "../../etc/passwd.pdf"))
    assert ".." not in display_name
    assert ".." not in saved_path
    assert display_name.endswith(".pdf")


def test_lone_dot_pdf_filename_still_ends_in_pdf(small_real_pdf_bytes):
    # secure_filename(".pdf") strips down to "pdf" (treats the leading dot
    # as a hidden-file marker) -- save_and_validate should still hand back
    # something that reads as a PDF, not a bare extensionless name.
    _, _, display_name = save_and_validate(make_file_storage(small_real_pdf_bytes, ".pdf"))
    assert display_name.lower().endswith(".pdf")


def test_two_uploads_of_the_same_filename_get_different_ids(small_real_pdf_bytes):
    doc_id_1, path_1, _ = save_and_validate(make_file_storage(small_real_pdf_bytes, "report.pdf"))
    doc_id_2, path_2, _ = save_and_validate(make_file_storage(small_real_pdf_bytes, "report.pdf"))
    assert doc_id_1 != doc_id_2
    assert path_1 != path_2
    assert os.path.exists(path_1) and os.path.exists(path_2)  # neither overwrote the other


# ---------------------------------------------------------------------------
# ingest_pdf
# ---------------------------------------------------------------------------


def test_ingest_produces_nonempty_chunks_with_expected_metadata(small_real_pdf_bytes):
    doc_id, saved_path, display_name = save_and_validate(make_file_storage(small_real_pdf_bytes, "report.pdf"))
    chunks, vector_ids = ingest_pdf(doc_id, saved_path, display_name)

    assert len(chunks) > 0
    assert len(vector_ids) == len(chunks)
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == doc_id
        assert chunk.metadata["source"] == display_name
        assert chunk.page_content.strip() != ""


def test_ingest_vector_ids_are_unique_and_doc_id_prefixed(small_real_pdf_bytes):
    doc_id, saved_path, display_name = save_and_validate(make_file_storage(small_real_pdf_bytes, "report.pdf"))
    chunks, vector_ids = ingest_pdf(doc_id, saved_path, display_name)

    assert len(set(vector_ids)) == len(vector_ids)  # no duplicates
    assert all(v.startswith(f"{doc_id}::") for v in vector_ids)


def test_ingest_uses_the_projects_one_chunking_scheme(small_real_pdf_bytes):
    """Chunk sizes should look like text_split()'s output (<=CHUNK_SIZE
    chars), not raw whole-page text -- otherwise a freshly-uploaded
    document's retrieval quality would silently differ from every other
    document already in the knowledge base (seeded or uploaded alike --
    see src/documents.py's module docstring)."""
    doc_id, saved_path, display_name = save_and_validate(make_file_storage(small_real_pdf_bytes, "report.pdf"))
    chunks, _ = ingest_pdf(doc_id, saved_path, display_name)
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)


# ---------------------------------------------------------------------------
# remove_uploaded_file
# ---------------------------------------------------------------------------


def test_remove_uploaded_file_deletes_only_the_matching_file(small_real_pdf_bytes, isolated_upload_dir):
    doc_id_1, path_1, _ = save_and_validate(make_file_storage(small_real_pdf_bytes, "keep_me.pdf"))
    doc_id_2, path_2, _ = save_and_validate(make_file_storage(small_real_pdf_bytes, "delete_me.pdf"))

    remove_uploaded_file(doc_id_2)

    assert os.path.exists(path_1)
    assert not os.path.exists(path_2)


def test_remove_uploaded_file_is_safe_when_nothing_matches(isolated_upload_dir):
    remove_uploaded_file("no-such-doc-id")  # should not raise


def test_remove_uploaded_file_is_safe_when_directory_does_not_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "UPLOAD_DIR", str(tmp_path / "does-not-exist"))
    remove_uploaded_file("anything")  # should not raise


# ---------------------------------------------------------------------------
# find_document_file_path -- what app.py's GET /documents/<id>/file route
# (click a document in the sidebar to view the actual PDF) resolves a
# document's id to an actual file with. See its docstring for why an
# uploaded and a seeded document need genuinely different lookup logic
# despite behaving identically everywhere else document-related.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_seed_dir(tmp_path, monkeypatch):
    """Redirects SEED_DIR to a pytest tmp_path for tests exercising the
    seed-file half of find_document_file_path(), so nothing ever reads
    the real project's data/seed/*.pdf during a test run -- the same
    isolation isolated_upload_dir (above) gives the upload half."""
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    monkeypatch.setattr(documents, "SEED_DIR", str(seed_dir))
    return seed_dir


def test_finds_an_uploaded_document_by_its_doc_id_prefix(small_real_pdf_bytes, isolated_upload_dir):
    doc_id, saved_path, display_name = save_and_validate(make_file_storage(small_real_pdf_bytes, "report.pdf"))

    found = find_document_file_path(doc_id, display_name)

    assert found == saved_path


def test_finds_a_seed_document_by_its_plain_filename(isolated_seed_dir):
    seed_file = isolated_seed_dir / "diabetes.pdf"
    seed_file.write_bytes(b"%PDF-1.4 fake seed content")

    found = find_document_file_path("seed-diabetes", "diabetes.pdf")

    assert found == str(seed_file)


def test_upload_dir_is_checked_before_seed_dir(small_real_pdf_bytes, isolated_upload_dir, isolated_seed_dir):
    """Not a realistic collision in practice (uploads use random uuid4
    ids, seed documents use deterministic seed-<slug> ids -- see
    seed_data.py's _seed_doc_id() -- so the two id spaces don't actually
    overlap), but the lookup order should still be well-defined rather
    than accidental."""
    doc_id, saved_path, display_name = save_and_validate(make_file_storage(small_real_pdf_bytes, "same-name.pdf"))
    decoy_seed_file = isolated_seed_dir / display_name
    decoy_seed_file.write_bytes(b"%PDF-1.4 this one should NOT be returned")

    found = find_document_file_path(doc_id, display_name)

    assert found == saved_path


def test_returns_none_for_an_id_with_no_matching_file_anywhere(isolated_upload_dir, isolated_seed_dir):
    found = find_document_file_path("no-such-doc-id", "ghost.pdf")

    assert found is None


def test_returns_none_when_seed_dir_itself_does_not_exist(isolated_upload_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "SEED_DIR", str(tmp_path / "does-not-exist-at-all"))

    found = find_document_file_path("seed-whatever", "whatever.pdf")

    assert found is None
