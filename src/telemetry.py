"""
telemetry.py — observability for MediCare AI.

Every chat turn gets logged: the question, a latency breakdown (retrieval
vs generation), whether the semantic cache was hit, how many sources were
used, and whether the emergency guardrail fired. That's what powers the
/stats endpoint — real numbers instead of "it feels fast."

Two implementations behind the same small interface
(init_db / log_query / log_feedback / get_stats):

  PostgresTelemetry  the real backend, for any Postgres 12+ server
                      (built and tested against Neon's free tier). Every
                      write goes through a small connection pool
                      (psycopg2's ThreadedConnectionPool) rather than
                      opening a new TCP+TLS connection per query — worth
                      being able to explain in an interview: pooling
                      amortizes the handshake cost across requests
                      instead of paying it every time, which matters more
                      on Postgres than most databases since Postgres
                      connections are relatively expensive (each one is
                      a full OS process on the server side).

  InMemoryTelemetry   a pure-Python drop-in with zero external
                      dependencies, injected via
                      create_app(telemetry=InMemoryTelemetry()) in
                      tests/test_app.py — the exact same
                      dependency-injection pattern already used for the
                      RAG pipeline (src/pipeline.py) and the semantic
                      cache (src/cache.py). This is why the test suite
                      never needs a real Postgres server running.

psycopg2 is imported lazily, inside PostgresTelemetry.__init__, not at
module load time — so `from src.telemetry import InMemoryTelemetry`
doesn't require that package to even be installed.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medicare_ai")


def _zero_filled_daily_series(by_day: dict, days: int) -> list:
    """Shared by both PostgresTelemetry and InMemoryTelemetry's
    get_daily_stats(): given a {date_string: row} map of days that *do*
    have data, returns a list covering every one of the last `days` days
    in order, oldest first, filling in zeros for any day with no queries
    at all. `row` may be a psycopg2 RealDictRow or a plain dict — both
    support the same ["queries"]/["avg_ms"]/["cache_hits"] keys.
    """
    today = datetime.now(timezone.utc).date()
    series = []
    for i in range(days):
        day = today - timedelta(days=(days - 1 - i))
        key = day.isoformat()
        row = by_day.get(key)
        if row is None:
            series.append({"date": key, "queries": 0, "avg_ms": 0, "cache_hits": 0})
        else:
            series.append(
                {
                    "date": key,
                    "queries": row["queries"],
                    "avg_ms": round(float(row["avg_ms"]), 1),
                    "cache_hits": row["cache_hits"],
                }
            )
    return series


class PostgresTelemetry:
    """Real telemetry backend, backed by PostgreSQL.

    Reads a single connection string from the DATABASE_URL environment
    variable (see .env.example) — the same convention Neon, Render,
    Railway, Heroku, and Supabase all use, so it's usually a straight
    copy-paste from the provider's dashboard with no reassembly needed.
    Works with any standard Postgres 12+ server, not just Neon.
    """

    def __init__(self, min_conn: int = 1, max_conn: int = 5):
        import psycopg2
        import psycopg2.extras
        from psycopg2 import pool as pg_pool

        self._psycopg2 = psycopg2
        self._dict_cursor = psycopg2.extras.RealDictCursor

        database_url = os.environ.get("DATABASE_URL", "")
        # ThreadedConnectionPool, not SimpleConnectionPool: Flask can serve
        # requests from multiple threads depending on how it's run (the
        # dev server, or gunicorn with --threads), and SimpleConnectionPool
        # is explicitly documented as not thread-safe.
        self._pool = pg_pool.ThreadedConnectionPool(min_conn, max_conn, dsn=database_url)

    def _connect(self):
        return self._pool.getconn()

    def _release(self, conn):
        # Important: putconn(), not conn.close(). Closing a connection
        # obtained from a psycopg2 pool destroys it instead of returning it
        # to the pool -- the opposite of what pooling is for. (This is a
        # real API difference from mysql-connector-python's pool, where
        # conn.close() correctly returns the connection to that pool.)
        self._pool.putconn(conn)

    def init_db(self):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    rating VARCHAR(8) NOT NULL CHECK (rating IN ('up', 'down')),
                    created_at VARCHAR(64) NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    retrieval_ms REAL,
                    generation_ms REAL,
                    total_ms REAL,
                    num_sources INTEGER,
                    cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
                    emergency BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at VARCHAR(64) NOT NULL
                )
                """
            )
            conn.commit()
            cur.close()
        finally:
            self._release(conn)

    def log_query(self, question, retrieval_ms, generation_ms, total_ms, num_sources, cache_hit, emergency):
        logger.info(
            "query total=%.0fms retrieval=%s generation=%s sources=%s cache_hit=%s emergency=%s",
            total_ms or 0,
            f"{retrieval_ms:.0f}ms" if retrieval_ms is not None else "n/a",
            f"{generation_ms:.0f}ms" if generation_ms is not None else "n/a",
            num_sources,
            cache_hit,
            emergency,
        )
        try:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO query_logs
                       (question, retrieval_ms, generation_ms, total_ms, num_sources, cache_hit, emergency, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        question,
                        retrieval_ms,
                        generation_ms,
                        total_ms,
                        num_sources,
                        # Real bool, not int(bool(...)): Postgres is strict
                        # about types in parameterized queries and psycopg2
                        # adapts Python bool -> SQL boolean natively. MySQL
                        # was lax enough to accept 0/1 into TINYINT; Postgres
                        # generally is not for a genuine BOOLEAN column.
                        bool(cache_hit),
                        bool(emergency),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
                cur.close()
            finally:
                self._release(conn)
        except Exception:
            logger.exception("Failed to write query log")

    def log_feedback(self, question: str, answer: str, rating: str):
        try:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO feedback (question, answer, rating, created_at) VALUES (%s, %s, %s, %s)",
                    (question, answer, rating, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                cur.close()
            finally:
                self._release(conn)
        except Exception:
            logger.exception("Failed to write feedback")

    def get_stats(self) -> dict:
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=self._dict_cursor)

            cur.execute("SELECT COUNT(*) AS c FROM query_logs")
            total = cur.fetchone()["c"]
            if total == 0:
                cur.close()
                return {"total_queries": 0}

            cur.execute("SELECT AVG(total_ms) AS v FROM query_logs")
            avg_total = cur.fetchone()["v"]
            cur.execute("SELECT AVG(retrieval_ms) AS v FROM query_logs WHERE retrieval_ms IS NOT NULL")
            avg_retrieval = cur.fetchone()["v"]
            cur.execute("SELECT AVG(generation_ms) AS v FROM query_logs WHERE generation_ms IS NOT NULL")
            avg_generation = cur.fetchone()["v"]
            # IS TRUE, not "= 1": Postgres won't implicitly compare a
            # boolean column against an integer literal the way MySQL's
            # looser typing allowed.
            cur.execute("SELECT COUNT(*) AS c FROM query_logs WHERE cache_hit IS TRUE")
            cache_hits = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM query_logs WHERE emergency IS TRUE")
            emergencies = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM feedback WHERE rating = 'up'")
            fb_up = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM feedback WHERE rating = 'down'")
            fb_down = cur.fetchone()["c"]
            cur.close()

            return {
                "total_queries": total,
                "avg_total_ms": round(float(avg_total or 0), 1),
                "avg_retrieval_ms": round(float(avg_retrieval or 0), 1),
                "avg_generation_ms": round(float(avg_generation or 0), 1),
                "cache_hit_rate": round(cache_hits / total, 3),
                "cache_hits": cache_hits,
                "emergency_rate": round(emergencies / total, 3),
                "emergency_count": emergencies,
                "feedback_up": fb_up,
                "feedback_down": fb_down,
            }
        finally:
            self._release(conn)

    def get_daily_stats(self, days: int = 14) -> list:
        """Query volume, average latency, and cache hits per day for the
        last `days` days — feeds the dashboard's trend chart
        (static/app.jsx's Dashboard component). Zero-filled for any day
        with no traffic at all, so the chart always shows a full, gap-free
        window instead of jumping straight from one active day to the next.

        created_at is stored as VARCHAR (an ISO-8601 string — see
        log_query above), not a native TIMESTAMP column, so every
        comparison/grouping here explicitly casts it with `::timestamptz`
        rather than relying on Postgres to infer the type. The cutoff
        itself is computed once in Python and passed in as a single
        parameter (a plain timestamp comparison) rather than building a
        dynamic `INTERVAL '%s days'` in SQL, which is a needless source of
        parameterization bugs for what a one-line Python timedelta already
        does clearly.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=self._dict_cursor)
            cur.execute(
                """
                SELECT
                    DATE(created_at::timestamptz) AS day,
                    COUNT(*) AS queries,
                    COALESCE(AVG(total_ms), 0) AS avg_ms,
                    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS cache_hits
                FROM query_logs
                WHERE created_at::timestamptz >= %s::timestamptz
                GROUP BY DATE(created_at::timestamptz)
                ORDER BY day
                """,
                (cutoff,),
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            self._release(conn)

        by_day = {r["day"].isoformat(): r for r in rows}
        return _zero_filled_daily_series(by_day, days)


class InMemoryTelemetry:
    """Pure-Python drop-in used by the test suite — no Postgres, no file
    on disk, no network. Injected via create_app(telemetry=InMemoryTelemetry())."""

    def __init__(self):
        self.feedback = []
        self.query_logs = []

    def init_db(self):
        pass  # nothing to create

    def log_query(self, question, retrieval_ms, generation_ms, total_ms, num_sources, cache_hit, emergency):
        self.query_logs.append(
            {
                "question": question,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                "num_sources": num_sources,
                "cache_hit": bool(cache_hit),
                "emergency": bool(emergency),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def log_feedback(self, question: str, answer: str, rating: str):
        self.feedback.append({"question": question, "answer": answer, "rating": rating})

    def get_stats(self) -> dict:
        total = len(self.query_logs)
        if total == 0:
            return {"total_queries": 0}

        def avg(key):
            values = [q[key] for q in self.query_logs if q[key] is not None]
            return (sum(values) / len(values)) if values else 0

        cache_hits = sum(1 for q in self.query_logs if q["cache_hit"])
        emergencies = sum(1 for q in self.query_logs if q["emergency"])
        fb_up = sum(1 for f in self.feedback if f["rating"] == "up")
        fb_down = sum(1 for f in self.feedback if f["rating"] == "down")

        return {
            "total_queries": total,
            "avg_total_ms": round(avg("total_ms"), 1),
            "avg_retrieval_ms": round(avg("retrieval_ms"), 1),
            "avg_generation_ms": round(avg("generation_ms"), 1),
            "cache_hit_rate": round(cache_hits / total, 3),
            "cache_hits": cache_hits,
            "emergency_rate": round(emergencies / total, 3),
            "emergency_count": emergencies,
            "feedback_up": fb_up,
            "feedback_down": fb_down,
        }

    def get_daily_stats(self, days: int = 14) -> list:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        by_day: dict = {}
        for q in self.query_logs:
            created_at = q.get("created_at")
            if not created_at:
                continue
            when = datetime.fromisoformat(created_at)
            if when < cutoff:
                continue
            key = when.date().isoformat()
            bucket = by_day.setdefault(key, {"queries": 0, "_ms_sum": 0.0, "_ms_count": 0, "cache_hits": 0})
            bucket["queries"] += 1
            if q.get("total_ms") is not None:
                bucket["_ms_sum"] += q["total_ms"]
                bucket["_ms_count"] += 1
            if q.get("cache_hit"):
                bucket["cache_hits"] += 1

        # Reduce each bucket's running sum/count down to the same
        # {queries, avg_ms, cache_hits} shape _zero_filled_daily_series
        # expects (matching what Postgres's AVG() already returns there).
        reduced = {
            key: {
                "queries": b["queries"],
                "avg_ms": (b["_ms_sum"] / b["_ms_count"]) if b["_ms_count"] else 0,
                "cache_hits": b["cache_hits"],
            }
            for key, b in by_day.items()
        }
        return _zero_filled_daily_series(reduced, days)
