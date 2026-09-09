"""
document_store.py — the manifest of what's in the knowledge base.

Pinecone stores vectors; it has no concept of "list every distinct
document that's been indexed" or "give me back the ids I used for
document X". This module is that missing manifest, so:

  - GET  /documents         can list everything in the knowledge base,
                             seeded or uploaded alike (app.py)
  - DELETE /documents/<id>  knows exactly which vector ids to remove --
                             required because Pinecone serverless indexes
                             only support delete-by-id, not
                             delete-by-metadata-filter

It also remembers which document ids have been deleted at all, not just
which ones currently exist -- see mark_deleted()/was_deleted() below.
Without that, deleting one of the data/seed/*.pdf documents would look
identical, from seed_data.py's point of view on the *next* startup, to a
document that was simply never seeded yet -- and it would silently come
back. Free-tier hosts commonly restart an idle process on its next
request (Render/Koyeb's free tiers both sleep and cold-start this way),
so "silently" here would likely mean "within a few hours", not some
theoretical edge case. seed_data.py gives every seeded document a
deterministic id (derived from its filename, unlike an upload's random
one) specifically so this check is a simple, unambiguous lookup.

Same two-implementation shape as src/telemetry.py, for the same reason:

  PostgresDocumentStore  the real backend — two more tables
                          (uploaded_documents, deleted_document_ids) in
                          the same Postgres database telemetry already
                          uses, so this piggybacks on infrastructure the
                          project needs anyway rather than introducing a
                          new one.

  InMemoryDocumentStore   a pure-Python drop-in with zero external
                          dependencies, injected via
                          create_app(document_store=InMemoryDocumentStore())
                          in tests — no Postgres required to test the
                          upload/list/delete routes.

psycopg2 is imported lazily, inside PostgresDocumentStore.__init__, not
at module load time — so `from src.document_store import
InMemoryDocumentStore` doesn't require that package to even be installed.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger("medicare_ai")


class PostgresDocumentStore:
    """Real document-manifest backend, backed by PostgreSQL. Reads
    DATABASE_URL the same way src/telemetry.py's PostgresTelemetry does —
    see .env.example.
    """

    def __init__(self, min_conn: int = 1, max_conn: int = 5):
        import psycopg2
        import psycopg2.extras
        from psycopg2 import pool as pg_pool

        self._dict_cursor = psycopg2.extras.RealDictCursor
        self._OperationalError = psycopg2.OperationalError  # <-- stored for retry logic

        database_url = os.environ.get("DATABASE_URL", "")
        self._pool = pg_pool.ThreadedConnectionPool(min_conn, max_conn, dsn=database_url)

    def _get_conn(self):
        """Get a connection from the pool, retrying if pool is temporarily exhausted."""
        for attempt in range(3):
            try:
                return self._pool.getconn()
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        # Should never reach here, but just in case
        raise RuntimeError("Could not obtain DB connection")

    def _execute(self, query, params=None, fetchall=False, fetchone=False, commit=False):
        """
        Execute a SQL query with automatic retry on connection errors.

        If psycopg2.OperationalError is raised (e.g., SSL connection closed
        unexpectedly), the broken connection is discarded and the operation
        is retried with a fresh one from the pool.
        """
        max_retries = 2
        for attempt in range(max_retries + 1):
            conn = None
            try:
                conn = self._get_conn()
                cur = conn.cursor(cursor_factory=self._dict_cursor if (fetchall or fetchone) else None)
                if params is not None:
                    cur.execute(query, params)
                else:
                    cur.execute(query)

                if fetchall:
                    result = cur.fetchall()
                elif fetchone:
                    result = cur.fetchone()
                else:
                    result = None

                if commit:
                    conn.commit()
                cur.close()
                return result

            except Exception as e:
                if conn:
                    # If OperationalError (connection broken), close permanently.
                    # Otherwise release back to pool.
                    try:
                        close_conn = isinstance(e, self._OperationalError)
                        self._pool.putconn(conn, close=close_conn)
                    except Exception:
                        pass
                if attempt < max_retries and isinstance(e, self._OperationalError):
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise

    def init_db(self):
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS uploaded_documents (
                id VARCHAR(32) PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                page_count INTEGER,
                vector_ids TEXT NOT NULL,
                uploaded_at VARCHAR(64) NOT NULL
            )
            """,
            commit=True
        )
        # See this module's docstring: tracks every id that has ever
        # been deleted (not just what's currently absent), so
        # seed_data.py can tell "never seeded yet" apart from
        # "deliberately removed" on a later restart.
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS deleted_document_ids (
                id VARCHAR(64) PRIMARY KEY,
                deleted_at VARCHAR(64) NOT NULL
            )
            """,
            commit=True
        )

    def add_document(self, doc_id: str, filename: str, chunk_count: int, page_count: int, vector_ids: List[str]):
        self._execute(
            """INSERT INTO uploaded_documents (id, filename, chunk_count, page_count, vector_ids, uploaded_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            params=(
                doc_id,
                filename,
                chunk_count,
                page_count,
                # Stored as a JSON string in a TEXT column rather than
                # a native Postgres array -- one fewer type mapping to
                # get right, and this table is never queried *by*
                # vector_ids, only ever read back whole for a known id.
                json.dumps(vector_ids),
                datetime.now(timezone.utc).isoformat(),
            ),
            commit=True
        )

    def list_documents(self) -> List[dict]:
        rows = self._execute(
            "SELECT id, filename, chunk_count, page_count, uploaded_at "
            "FROM uploaded_documents ORDER BY uploaded_at DESC",
            fetchall=True
        )
        return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> Optional[dict]:
        row = self._execute(
            "SELECT * FROM uploaded_documents WHERE id = %s",
            params=(doc_id,),
            fetchone=True
        )
        if not row:
            return None
        doc = dict(row)
        doc["vector_ids"] = json.loads(doc["vector_ids"])
        return doc

    def delete_document(self, doc_id: str):
        self._execute(
            "DELETE FROM uploaded_documents WHERE id = %s",
            params=(doc_id,),
            commit=True
        )

    def mark_deleted(self, doc_id: str):
        """Records that doc_id was deliberately removed, so seed_data.py
        never re-adds it on a later restart -- see this module's
        docstring. Called for every deletion, seeded or uploaded alike
        (app.py's DELETE /documents/<id> route doesn't know or care which
        kind doc_id is); harmless to record for an uploaded document's id
        too, since seed_data.py only ever looks up the deterministic ids
        it itself generates for files under data/seed/, and an upload's
        id (random, from uuid4) will never collide with one of those."""
        self._execute(
            "INSERT INTO deleted_document_ids (id, deleted_at) VALUES (%s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            params=(doc_id, datetime.now(timezone.utc).isoformat()),
            commit=True
        )

    def was_deleted(self, doc_id: str) -> bool:
        row = self._execute(
            "SELECT 1 FROM deleted_document_ids WHERE id = %s",
            params=(doc_id,),
            fetchone=True
        )
        return row is not None


class InMemoryDocumentStore:
    """Pure-Python drop-in used by the test suite — no Postgres, no file
    on disk, no network. Injected via
    create_app(document_store=InMemoryDocumentStore())."""

    def __init__(self):
        self._docs = {}
        self._deleted_ids = set()

    def init_db(self):
        pass  # nothing to create

    def add_document(self, doc_id: str, filename: str, chunk_count: int, page_count: int, vector_ids: List[str]):
        self._docs[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "chunk_count": chunk_count,
            "page_count": page_count,
            "vector_ids": list(vector_ids),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

    def list_documents(self) -> List[dict]:
        docs = [{k: v for k, v in d.items() if k != "vector_ids"} for d in self._docs.values()]
        return sorted(docs, key=lambda d: d["uploaded_at"], reverse=True)

    def get_document(self, doc_id: str) -> Optional[dict]:
        doc = self._docs.get(doc_id)
        return dict(doc) if doc else None

    def delete_document(self, doc_id: str):
        self._docs.pop(doc_id, None)

    def mark_deleted(self, doc_id: str):
        self._deleted_ids.add(doc_id)

    def was_deleted(self, doc_id: str) -> bool:
        return doc_id in self._deleted_ids