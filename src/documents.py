"""
documents.py — turning a PDF (seeded at deploy time, or uploaded later
through the UI) into indexable chunks.

This is the "front door" for every document in the knowledge base --
there is no longer a separate code path for "the curated reference book"
vs. "something a user uploaded"; both a fresh deploy's data/seed/*.pdf
files (see seed_data.py) and anything POSTed to /documents/upload go
through save_and_validate()/ingest_pdf() the same way and land in the
same DOCUMENTS_NAMESPACE, so they show up, get selected, and get deleted
identically -- see app.py's /documents routes and src/document_store.py.

Everything here is pure file-handling and text-processing, with zero
knowledge of Pinecone/Groq. app.py calls save_and_validate() then
ingest_pdf(), and only *after* that hands the resulting chunks to
pipeline.vectorstore to actually be embedded and upserted (see
src/pipeline.py for the retrieval side of this, and app.py's /documents
routes for the HTTP side).

Keeping Pinecone out of this module means the validation/ingestion logic
here can be unit-tested with a real local PDF and no network or API keys
at all — see tests/test_documents.py.
"""

import os
import uuid
from typing import List, Optional, Tuple

from werkzeug.utils import secure_filename
from langchain_core.documents import Document

from src.helper import load_single_pdf, text_split

# Pinecone's serverless indexes only support delete-by-id, not
# delete-by-metadata-filter -- so every chunk gets an id of the form
# "{doc_id}::{chunk_index}" at ingest time, and those ids are the only
# way a later "remove this document" ever finds its vectors again (see
# src/document_store.py, which is what remembers the id list). Which
# *document* a chunk should be searched under is a separate concern,
# handled by the doc_id metadata tag below plus CombinedMedicalRetriever's
# optional filter (src/pipeline.py) -- not by namespace, since every
# document now shares one namespace.
#
# DOCUMENTS_NAMESPACE lives here (not in src/pipeline.py, which is where
# it actually gets used against Pinecone) so that app.py can import it
# without pulling langchain_pinecone/langchain_groq into every module
# import -- this file has no vector-store dependencies at all, matching
# src/cache.py and src/safety.py, the other "always imported at module
# load time" modules. src/pipeline.py and seed_data.py import it from
# here too, so there is exactly one place this string is ever written
# down.
DOCUMENTS_NAMESPACE = "documents"
ALLOWED_EXTENSION = ".pdf"
PDF_MAGIC_BYTES = b"%PDF-"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB -- generous for a fact-sheet-sized PDF; raise if you
                                      # know your host has the RAM/time budget for bigger files.

UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploads"
)

# Defined here (not in seed_data.py, which is where it's more obviously
# "about") specifically so find_document_file_path() below -- which needs
# to know about both directories to serve a click-to-view request for
# either kind of document -- doesn't have to import seed_data.py to get
# it, which would be a circular import (seed_data.py already imports
# from this module). seed_data.py imports SEED_DIR from here instead.
SEED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "seed"
)


class InvalidUpload(ValueError):
    """Raised for anything wrong with the file itself (bad type, empty,
    too large, unreadable) -- as opposed to an ingestion failure, which
    means the file *was* a valid PDF but something went wrong turning it
    into chunks. app.py maps this to a 400; ingestion failures get a 500.
    """


def _looks_like_pdf(header: bytes) -> bool:
    return header[:5] == PDF_MAGIC_BYTES


def save_and_validate(file_storage) -> Tuple[str, str, str]:
    """
    Validates an uploaded werkzeug FileStorage and saves it to disk.

    Deliberately checks the file's actual magic bytes rather than trusting
    the ".pdf" extension or the browser-supplied content type -- either of
    those is trivial to spoof, and load_single_pdf() below would otherwise
    be the first thing to discover a non-PDF file, several steps too late.

    Returns (doc_id, saved_path, display_name).
    Raises InvalidUpload if the file doesn't pass validation.
    """
    filename = file_storage.filename or ""
    if not filename.lower().endswith(ALLOWED_EXTENSION):
        raise InvalidUpload("Only PDF files are supported.")

    content = file_storage.read()
    if len(content) == 0:
        raise InvalidUpload("That file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise InvalidUpload(f"That file is too large (max {limit_mb}MB).")
    if not _looks_like_pdf(content):
        raise InvalidUpload("That doesn't look like a valid PDF file.")

    display_name = secure_filename(filename) or "document.pdf"
    if not display_name.lower().endswith(ALLOWED_EXTENSION):
        # secure_filename() can strip a lone ".pdf" down to "pdf" (it
        # treats a leading dot as a hidden-file marker) -- make sure the
        # name we show the user and store on disk still reads as a PDF.
        display_name += ALLOWED_EXTENSION

    doc_id = uuid.uuid4().hex[:12]
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_path = os.path.join(UPLOAD_DIR, f"{doc_id}__{display_name}")
    with open(saved_path, "wb") as f:
        f.write(content)

    return doc_id, saved_path, display_name


def ingest_pdf(doc_id: str, path: str, display_name: str) -> Tuple[List[Document], List[str]]:
    """
    Loads one PDF from disk (a freshly-saved upload, or a file sitting in
    data/seed/ at seed time -- this function doesn't care which) and
    splits it using the project's one chunking scheme (see
    src/helper.text_split), so retrieval quality is consistent no matter
    which document a chunk came from.

    Every chunk is tagged doc_id=<doc_id> in its metadata, on top of the
    usual source/page pair -- this is what CombinedMedicalRetriever
    (src/pipeline.py) filters on when a chat request scopes its question
    to a chosen subset of documents, and it's how vector ids are built
    below.

    Returns (chunks, vector_ids), both empty if the PDF had no
    extractable text (e.g. a scanned page image with no text layer) --
    callers should treat an empty result as "nothing to index" rather
    than an error.
    """
    pages = load_single_pdf(path)
    for page in pages:
        page.metadata = {
            "source": display_name,
            "page": page.metadata.get("page"),
            "page_label": page.metadata.get("page_label"),
        }

    chunks = text_split(pages)
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id

    vector_ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
    return chunks, vector_ids


def remove_uploaded_file(doc_id: str) -> None:
    """Deletes the on-disk copy of a previously-uploaded PDF, identified
    by the doc_id prefix save_and_validate() gave its filename. Used by
    app.py's DELETE /documents/<id> route after the document's vectors
    and manifest row are gone, so nothing is left behind on disk. Safe to
    call even if the file was already removed (e.g. a prior failed
    ingestion already cleaned it up) or never existed.
    """
    if not os.path.isdir(UPLOAD_DIR):
        return
    for name in os.listdir(UPLOAD_DIR):
        if name.startswith(f"{doc_id}__"):
            try:
                os.remove(os.path.join(UPLOAD_DIR, name))
            except OSError:
                pass


def find_document_file_path(doc_id: str, filename: str) -> Optional[str]:
    """Resolves a document's id (+ the filename document_store has on
    record for it) to wherever its actual PDF bytes live on disk --
    UPLOAD_DIR for something uploaded through the UI, SEED_DIR for one of
    the bundled data/seed/*.pdf files -- so app.py's GET
    /documents/<id>/file route can serve either kind identically, the
    same "no distinction once it's in the knowledge base" principle as
    everywhere else (see this module's docstring).

    The two live in different places with different naming schemes for
    an unrelated reason (upload filenames are prefixed with their doc_id
    to keep two people's same-named upload from colliding on disk -- see
    save_and_validate(); seed files don't need that, since there's only
    ever one of each committed to the repo) -- this function is what
    hides that difference from every caller.

    Returns None (not an exception) if the manifest says a document
    exists but its file doesn't -- e.g. someone deleted a file out from
    under data/seed/ by hand -- so the route can turn that into a clean
    404 rather than a 500.
    """
    if os.path.isdir(UPLOAD_DIR):
        for name in os.listdir(UPLOAD_DIR):
            if name.startswith(f"{doc_id}__"):
                return os.path.join(UPLOAD_DIR, name)

    seed_path = os.path.join(SEED_DIR, filename)
    if os.path.isfile(seed_path):
        return seed_path

    return None
