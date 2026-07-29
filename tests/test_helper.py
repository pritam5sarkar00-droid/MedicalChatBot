from langchain_core.documents import Document

from src.helper import filter_to_minimal_docs, wait_for_embedding_service, warm_up_embedding_service_async


def test_filter_to_minimal_docs_preserves_source_page_and_page_label():
    """This is a regression test: filter_to_minimal_docs used to keep only
    'source' in metadata, silently dropping 'page'. Since app.py's
    extract_sources() builds every citation chip from these fields, that
    bug meant citations would show a filename but never a page number, for
    every single answer, with no error or test failure anywhere -- it only
    would have shown up by actually reading a live PDF end-to-end. This
    test exists so that gap can never come back unnoticed.

    page_label is kept alongside the raw page index for the same reason:
    dropping it here would silently make extract_sources() fall back to
    raw sequential page numbers for every citation, even for PDFs whose
    embedded page labels diverge from that (front matter, unnumbered
    cover pages, etc.) -- see filter_to_minimal_docs' own docstring."""
    docs = [
        Document(
            page_content="Asthma is a chronic airway condition...",
            metadata={
                "source": "data/seed/diabetes.pdf",
                "page": 22,
                "page_label": "23",
                "producer": "some noisy PDF library metadata",
                "creationdate": "2004-12-18",
            },
        )
    ]

    result = filter_to_minimal_docs(docs)

    assert len(result) == 1
    assert result[0].page_content == docs[0].page_content
    assert result[0].metadata == {"source": "data/seed/diabetes.pdf", "page": 22, "page_label": "23"}


def test_filter_to_minimal_docs_handles_missing_page():
    # Some loaders/document types genuinely have no page number -- should
    # degrade to None, not raise.
    docs = [Document(page_content="text", metadata={"source": "book.pdf"})]
    result = filter_to_minimal_docs(docs)
    assert result[0].metadata == {"source": "book.pdf", "page": None, "page_label": None}


def test_filter_to_minimal_docs_handles_empty_list():
    assert filter_to_minimal_docs([]) == []


# ---------------------------------------------------------------------------
# wait_for_embedding_service -- see its docstring in src/helper.py for the
# startup-race it fixes: seed_data.py's one-time seeding step running
# before a freshly-booting inference_service/ has finished loading its
# models, silently skipping every document and leaving the knowledge base
# looking empty until someone notices and restarts the app.
# ---------------------------------------------------------------------------


def test_returns_true_immediately_when_no_embedding_service_url_is_set(monkeypatch):
    """Single-service mode (embeddings run in-process): there's no
    separate service to wait for, so this must be an instant no-op, never
    adding startup delay for the far more common non-split deployment."""
    monkeypatch.delenv("EMBEDDING_SERVICE_URL", raising=False)

    import time

    start = time.monotonic()
    result = wait_for_embedding_service(timeout_s=5, poll_interval_s=1)
    elapsed = time.monotonic() - start

    assert result is True
    assert elapsed < 0.5


def test_returns_true_once_health_reports_models_loaded(monkeypatch):
    """The realistic case this whole function exists for: the service is
    up but still mid-boot when the wait starts, and finishes loading its
    models a couple seconds in -- this should notice and return as soon
    as that happens, not only at the timeout."""
    import threading
    import time

    from flask import Flask, jsonify

    health_app = Flask(__name__)
    state = {"ready": False}

    @health_app.route("/health")
    def health():
        return jsonify({"status": "ok" if state["ready"] else "loading", "models_loaded": state["ready"]})

    port = 8199
    thread = threading.Thread(
        target=lambda: health_app.run(host="127.0.0.1", port=port, use_reloader=False), daemon=True
    )
    thread.start()
    time.sleep(0.5)

    def become_ready_shortly():
        time.sleep(1)
        state["ready"] = True

    threading.Thread(target=become_ready_shortly, daemon=True).start()

    monkeypatch.setenv("EMBEDDING_SERVICE_URL", f"http://127.0.0.1:{port}")

    start = time.monotonic()
    result = wait_for_embedding_service(timeout_s=15, poll_interval_s=0.3)
    elapsed = time.monotonic() - start

    assert result is True
    assert 0.8 < elapsed < 10  # noticed it becoming ready, didn't just wait for the full timeout


def test_returns_false_after_timeout_when_never_ready(monkeypatch):
    """A service that's reachable but stuck (still loading, or reporting
    unhealthy) must not hang the app's startup forever -- give up after
    timeout_s and let the caller (create_app() in app.py) proceed anyway,
    logging that it timed out rather than blocking indefinitely."""
    import threading
    import time

    from flask import Flask, jsonify

    health_app = Flask(__name__)

    @health_app.route("/health")
    def health():
        return jsonify({"status": "loading", "models_loaded": False})

    port = 8200
    thread = threading.Thread(
        target=lambda: health_app.run(host="127.0.0.1", port=port, use_reloader=False), daemon=True
    )
    thread.start()
    time.sleep(0.5)

    monkeypatch.setenv("EMBEDDING_SERVICE_URL", f"http://127.0.0.1:{port}")

    start = time.monotonic()
    result = wait_for_embedding_service(timeout_s=2, poll_interval_s=0.3)
    elapsed = time.monotonic() - start

    assert result is False
    assert elapsed >= 1.8  # actually waited out the timeout, not an early bail


def test_returns_false_gracefully_when_service_is_unreachable(monkeypatch):
    """A wrong URL, or the service never having been deployed at all,
    must degrade the same way as "reachable but never ready" -- a
    connection failure on every poll, not an unhandled exception that
    would crash app startup."""
    monkeypatch.setenv("EMBEDDING_SERVICE_URL", "http://127.0.0.1:1")  # port 1: nothing listens here

    result = wait_for_embedding_service(timeout_s=2, poll_interval_s=0.3)

    assert result is False


# ---------------------------------------------------------------------------
# warm_up_embedding_service_async -- the complementary, non-blocking nudge
# fired unconditionally at the very start of create_app() (app.py), so a
# sleeping inference_service/ starts waking up immediately rather than
# waiting for the first real chat message to trigger that.
# ---------------------------------------------------------------------------


def test_is_a_true_no_op_with_no_embedding_service_url(monkeypatch):
    import threading
    import time

    monkeypatch.delenv("EMBEDDING_SERVICE_URL", raising=False)
    threads_before = threading.active_count()

    start = time.monotonic()
    warm_up_embedding_service_async()
    elapsed = time.monotonic() - start

    assert elapsed < 0.05  # returned instantly
    assert threading.active_count() == threads_before  # didn't even spawn a thread


def test_never_blocks_even_against_a_slow_to_respond_service(monkeypatch):
    """The whole point: this must return immediately regardless of how
    long the actual ping takes to complete in the background -- a slow
    cold start on the other end should never be something create_app()
    waits on here (that's wait_for_embedding_service()'s job, and only
    when seeding actually needs it)."""
    import threading
    import time

    from flask import Flask, jsonify

    slow_app = Flask(__name__)

    @slow_app.route("/health")
    def health():
        time.sleep(2)
        return jsonify({"status": "ok", "models_loaded": True})

    port = 8299
    thread = threading.Thread(
        target=lambda: slow_app.run(host="127.0.0.1", port=port, use_reloader=False), daemon=True
    )
    thread.start()
    time.sleep(0.5)

    monkeypatch.setenv("EMBEDDING_SERVICE_URL", f"http://127.0.0.1:{port}")

    start = time.monotonic()
    warm_up_embedding_service_async()
    elapsed = time.monotonic() - start

    assert elapsed < 0.5  # nowhere near the 2s the health check itself takes
    time.sleep(2.5)  # let the background thread's ping actually complete, for a clean test teardown


def test_never_raises_even_when_the_service_is_unreachable(monkeypatch):
    monkeypatch.setenv("EMBEDDING_SERVICE_URL", "http://127.0.0.1:1")

    warm_up_embedding_service_async()  # should not raise, in this thread or the background one

    import time

    time.sleep(0.5)  # let the background thread's failed attempt complete silently before the test exits
