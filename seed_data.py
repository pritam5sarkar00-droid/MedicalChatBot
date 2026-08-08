"""
seed_data.py — indexes data/seed/*.pdf into the knowledge base.

Replaces the old store_index.py. The old script loaded every PDF in
data/ in one batch with LangChain's DirectoryLoader and pushed the whole
thing into Pinecone's default namespace as a single, un-tracked blob --
which is exactly why the old reference book couldn't be listed or
deleted from the UI (see README's changelog and app.py's module
docstring). This version ingests data/seed/*.pdf one file at a time,
through the *exact* same save_and_validate()-adjacent path an upload
takes (src/documents.py's ingest_pdf(), the same DOCUMENTS_NAMESPACE,
the same document_store manifest), so every seeded document is a normal,
first-class, listable, deletable document from the moment it's indexed --
there's nothing here a person couldn't also get by uploading the same
PDF through the sidebar.

Two ways this runs:

  1. Automatically, once, on every app startup (create_app() in app.py)
     -- so `git clone` + fill in .env + `python app.py` is enough to get
     a working demo with a non-empty knowledge base, no separate manual
     step required. Safe to run on every restart: seed_default_documents()
     below skips any file whose deterministic id (see _seed_doc_id())
     already has a row in document_store, so a warm restart does no
     Pinecone/embedding work at all beyond the O(n) manifest check.

  2. By hand -- `python seed_data.py` -- useful right after adding a new
     file to data/seed/, or for rebuilding the knowledge base after
     pointing at a brand new, empty Pinecone index.

Deterministic ids, and why deletions have to stay deleted
-----------------------------------------------------------
Every seed file's id is derived from its filename (_seed_doc_id()) rather
than randomly generated the way an upload's is (src/documents.py's
save_and_validate() uses uuid4 for that) -- specifically so re-running
this script, or the app simply restarting, always resolves to the same
id for "diabetes.pdf" and can tell "already indexed" apart from "new
file, needs indexing" with a plain lookup.

That alone isn't quite enough, though: a person can delete a seeded
document from the sidebar exactly like an uploaded one (that's the
point -- see app.py's DELETE /documents/<id> route), which removes its
document_store row. Without anything else, the *next* restart would see
"seed-diabetes has no row" and index it right back -- indistinguishable,
from this script's point of view, from a file that was never seeded in
the first place. document_store.mark_deleted()/was_deleted() (src/
document_store.py) is the tombstone that breaks that ambiguity: app.py
records every deletion there (seeded or uploaded id alike -- it doesn't
need to know which), and seed_default_documents() below checks it before
ever re-adding anything, so a deletion made from the UI stays deleted
across restarts, not just until the free-tier host happens to sleep.
"""

import os
import re
from typing import List

from src.documents import DOCUMENTS_NAMESPACE, SEED_DIR, ingest_pdf


def _seed_doc_id(filename: str) -> str:
    """Deterministic id for a data/seed/*.pdf file, derived from its name
    -- e.g. "high-blood-pressure.pdf" -> "seed-high-blood-pressure". Kept
    under the 32-char id column PostgresDocumentStore uses (src/
    document_store.py) by truncating the slug; if you add a seed file
    with a very long name, prefer a shorter filename over relying on the
    truncation to disambiguate it from another long name that happens to
    share the same first ~26 characters.
    """
    stem = os.path.splitext(filename)[0]
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return f"seed-{slug}"[:32]


def _pending_seed_files(seed_dir: str, document_store) -> List[str]:
    """The filenames in seed_dir that seed_default_documents() would
    actually attempt to index right now -- not already in the manifest,
    not deliberately deleted before, actually a .pdf. Shared by
    has_pending_seed_documents() (a cheap yes/no a caller can check
    *before* paying any cost to seed) and seed_default_documents() itself
    (which does the real work), so the two can never quietly disagree
    about what counts as "needs seeding."
    """
    if not os.path.isdir(seed_dir):
        return []

    already_present = {d["filename"] for d in document_store.list_documents()}
    pending = []
    for filename in sorted(os.listdir(seed_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        if filename in already_present:
            continue
        if document_store.was_deleted(_seed_doc_id(filename)):
            continue
        pending.append(filename)
    return pending


def has_pending_seed_documents(document_store, seed_dir: str = SEED_DIR) -> bool:
    """Cheap check -- one document_store.list_documents() call, no
    Pinecone/embedding-service traffic at all -- for whether
    seed_default_documents() would actually have anything to do.

    Meant to be checked *before* paying for anything expensive on the way
    to seeding, most importantly waiting for a separate inference_service/
    deployment to report healthy (see wait_for_embedding_service() in
    src/helper.py). Without this check, create_app() would wait on that
    every single time the app starts -- including every time a free-tier
    host wakes it from an idle sleep, which can easily be most of its
    restarts -- even though, almost always by then, every seed document
    is already indexed and there is nothing whatsoever left to seed. That
    turns a should-be-instant restart into one that's slow for no reason,
    which is exactly backwards from the free-tier cold-start experience
    this project is trying to keep tolerable.
    """
    return bool(_pending_seed_files(seed_dir, document_store))


def seed_default_documents(vectorstore, document_store, seed_dir: str = SEED_DIR) -> int:
    """Indexes every *.pdf in seed_dir that isn't already in
    document_store and wasn't deliberately deleted from it before (see
    this module's docstring). Returns how many were newly added.

    vectorstore: anything shaped like a PineconeVectorStore -- needs
    .add_documents(docs, ids=, namespace=). This is intentionally just
    the vectorstore, not a whole RAGPipeline, so seeding never needs a
    Groq (or any chat-model) credential -- see this file's __main__
    block, which builds only an embeddings model + a Pinecone connection
    for exactly that reason.

    document_store: the manifest from src/document_store.py (Postgres-
    backed in production, in-memory in tests) -- .list_documents(),
    .was_deleted(id), .add_document(...).

    Never raises: a missing seed_dir, an unreadable or non-PDF-shaped
    file, or a Pinecone hiccup on one file are all logged (by whichever
    caller wrapped this -- app.py's create_app() catches around its own
    call) rather than allowed to take the whole batch, or app startup,
    down with them. One bad file just means one fewer document seeded,
    not zero.
    """
    pending = _pending_seed_files(seed_dir, document_store)
    seeded_count = 0

    for filename in pending:
        doc_id = _seed_doc_id(filename)
        path = os.path.join(seed_dir, filename)
        try:
            chunks, vector_ids = ingest_pdf(doc_id, path, filename)
            if not chunks:
                continue  # no extractable text -- nothing to index
            vectorstore.add_documents(chunks, ids=vector_ids, namespace=DOCUMENTS_NAMESPACE)
            page_count = len({c.metadata.get("page") for c in chunks})
            document_store.add_document(
                doc_id=doc_id,
                filename=filename,
                chunk_count=len(chunks),
                page_count=page_count,
                vector_ids=vector_ids,
            )
            seeded_count += 1
        except Exception:
            # Deliberately no bare `raise` here -- see docstring. The
            # caller (create_app(), or the __main__ block below, which
            # prints instead of logging) decides how loudly to surface
            # this; either way, one broken seed file shouldn't block the
            # other three from indexing.
            import logging

            logging.getLogger("medicare_ai").exception("Failed to seed %s", filename)

    return seeded_count


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    from src.helper import download_hugging_face_embeddings
    from src.pipeline import INDEX_NAME, ensure_index_exists
    from langchain_pinecone import PineconeVectorStore

    print("[1/3] Loading the embedding model (all-MiniLM-L6-v2) ...")
    embeddings = download_hugging_face_embeddings()

    print(f"[2/3] Connecting to Pinecone index '{INDEX_NAME}' (creating it if it doesn't exist yet) ...")
    ensure_index_exists(INDEX_NAME)
    vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

    print(f"[3/3] Seeding *.pdf from {SEED_DIR} ...")
    try:
        from src.document_store import PostgresDocumentStore

        document_store = PostgresDocumentStore()
        document_store.init_db()
    except Exception as e:
        print(f"      Couldn't reach Postgres ({e!r}) -- seeding into an in-memory manifest instead.")
        print("      This still populates Pinecone, but the app's own next startup won't know these")
        print("      documents are already seeded unless DATABASE_URL is reachable by then too.")
        from src.document_store import InMemoryDocumentStore

        document_store = InMemoryDocumentStore()

    count = seed_default_documents(vectorstore, document_store)
    if count:
        print(f"Done. Seeded {count} new document(s). Run `python app.py` (or redeploy) to use them.")
    else:
        print("Done. Nothing new to seed — the knowledge base is already up to date.")
