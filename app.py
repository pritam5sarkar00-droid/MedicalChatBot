"""
app.py — MediCare AI (Flask backend)

Built by Pritam

Architecture:
  src/pipeline.py    builds the conversational RAG chain (Pinecone + Groq).
                      Only imported lazily, inside create_app(), so tests
                      never have to install/authenticate against it. As of
                      the multi-service split (see DEPLOYMENT.md), the
                      embedding model and reranker it uses may live in
                      this same process or in a separate inference_service/
                      deployment reached over HTTP -- see src/helper.py.
  src/cache.py        semantic cache — skips retrieval+generation entirely
                      on repeat/near-duplicate questions, scoped per the
                      selected document set (see _cache_scope() below).
  src/safety.py       emergency keyword guardrail, runs before the LLM.
  src/telemetry.py     structured logging + Postgres (feedback + latency),
                      with a pure-Python in-memory double for tests.
  seed_data.py        idempotently indexes data/seed/*.pdf through the
                      same path as an upload, so a fresh deploy's knowledge
                      base isn't empty. Runs once automatically on startup
                      (only against real infra, never against the fakes
                      tests inject -- see create_app()) and can also be run
                      by hand: `python seed_data.py`.

The Flask app is built with an app factory (`create_app`) so FAKE
dependencies can be injected in tests instead of hitting real Pinecone,
Groq, or Postgres — see tests/test_app.py. This is the same pattern you'd use
to unit-test any service wrapped around a slow/external/paid dependency.

Every document in the knowledge base -- whether it was baked in via
data/seed/ at deploy time or uploaded later through the UI -- lives in one
Postgres/in-memory manifest (src/document_store.py) and one Pinecone
namespace (DOCUMENTS_NAMESPACE, src/documents.py); there is no separate
"reference" vs "uploaded" code path anywhere in this file.

Request flow for a chat turn:
  1. Browser POSTs JSON {message, history, document_ids} to /get (or
     streams via /get/stream). document_ids is optional; omitting it (or
     sending null/[]) searches every document, which is also exactly what
     a plain {message, history} request from eval/run_eval.py or an older
     client still does.
  2. src/safety.py checks for an emergency phrase — if matched, the LLM is
     skipped entirely and emergency guidance is returned immediately.
  3. The question is embedded once and checked against the semantic cache,
     scoped to the same document_ids so an answer computed under one
     document selection is never served back under a different one.
     A near-duplicate question (cosine similarity above threshold) within
     that scope returns the cached answer instantly, skipping both
     retrieval and generation.
  4. On a cache miss: a history-aware retriever rewrites follow-ups into
     standalone questions, searches Pinecone for the top-k chunks (filtered
     to document_ids when given), and Llama 3.3 70B (via Groq) answers
     using only that context.
  5. The turn (latency breakdown, cache hit, source count) is logged via
     src/telemetry.py, which is what powers /stats.

CORS is enabled only when ALLOWED_ORIGINS is set in the environment (a
comma-separated list of origins) -- see create_app(). Leave it unset for a
single-service deployment where Flask itself serves templates/chat.html
and static/app.jsx; set it when the frontend is hosted separately (e.g. on
Netlify/Vercel) and calls this API cross-origin. See DEPLOYMENT.md.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from langchain_core.messages import AIMessage, HumanMessage
from werkzeug.exceptions import HTTPException

from src.cache import SemanticCache
from src.helper import resolve_page_display
from src.safety import EMERGENCY_MESSAGE, detect_emergency
from src.documents import (
    DOCUMENTS_NAMESPACE,
    InvalidUpload,
    MAX_UPLOAD_BYTES,
    ingest_pdf,
    remove_uploaded_file,
    save_and_validate,
)

load_dotenv()

GENERIC_ERROR_MESSAGE = "Something went wrong on our end. Please try again in a moment."

# Returned verbatim, overriding whatever the LLM generated, whenever
# src/pipeline.py's CombinedMedicalRetriever finds literally nothing above
# MIN_SIMILARITY in the (optionally document_ids-scoped) knowledge base --
# see _is_no_context_answer() and the /get and /get/stream routes below.
# This guarantees an honest answer for clearly-out-of-scope questions
# regardless of how well the LLM follows the equivalent instruction in
# src/prompt.py, and (on the streaming path) skips the LLM call for that
# question entirely -- see the comment in chat_stream().
NO_CONTEXT_MESSAGE = (
    "I don't have information about that in the selected documents. Try "
    "rephrasing your question, selecting more documents to search, or "
    "uploading a PDF that covers this topic and asking again."
)


def _is_no_context_answer(answer: str) -> bool:
    return answer == NO_CONTEXT_MESSAGE


def build_chat_history(raw_history):
    """Turn the JSON history the browser sends into LangChain message
    objects. The browser is the source of truth for conversation state —
    the API itself stays stateless, which keeps the backend simple and
    horizontally scalable (no server-side session store to manage)."""
    messages = []
    for turn in (raw_history or [])[-10:]:  # last 10 turns is plenty of context
        role = turn.get("role")
        content = turn.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "bot":
            messages.append(AIMessage(content=content))
    return messages


def _normalize_document_ids(raw):
    """Turns whatever the frontend sent for the "answer only from these
    documents" selection (see DocumentsPanel in static/app.jsx) into
    either None (search every document -- the default, and what a plain
    {message, history} request with no document_ids key at all still
    gets, so eval/run_eval.py and any older/simpler client keep working
    unmodified) or a non-empty list of string ids.

    Deliberately never raises. A malformed value -- wrong type, an empty
    list because the frontend momentarily had nothing checked, junk
    entries -- is just treated as "no scoping", since a scoping filter is
    a narrowing convenience, not something a chat request should 400 over.
    CombinedMedicalRetriever.search() (src/pipeline.py) treats None and []
    identically, but resolving to a clean None here (rather than passing
    a maybe-empty list through) is what makes _cache_scope() below collapse
    both onto the same "no filter" cache scope.
    """
    if not isinstance(raw, list):
        return None
    ids = [d for d in raw if isinstance(d, str) and d.strip()]
    return ids or None


def _cache_scope(document_ids):
    """A stable SemanticCache scope key (src/cache.py) for a given
    document-id selection: None for "every document", so it lines up with
    the same-shaped requests that predate this feature, and otherwise a
    sorted, comma-joined string so that selecting {B, A} and {A, B} in
    the sidebar -- same set, different click order -- hit the exact same
    cache entry, while two genuinely different subsets never collide."""
    if not document_ids:
        return None
    return ",".join(sorted(document_ids))


def extract_sources(context_docs):
    """De-duplicate retrieved chunks down to a list of {source, page, score}.

    Every document in the knowledge base -- whether it arrived via
    data/seed/ at deploy time or through /documents/upload later -- is
    cited identically here; there is deliberately no flag distinguishing
    the two anymore (see src/documents.py's module docstring for why).

    Page numbers go through src/helper.py's resolve_page_display(), the
    one place this whole app decides what page a citation should show --
    see that function's docstring for the page_label-vs-raw-index
    reasoning. Using it here too (rather than re-deriving the same
    fallback inline) is what guarantees a citation chip and the page the
    model itself was told about (src/pipeline.py's document_prompt) never
    quietly disagree.

    'score' is the retrieval similarity CombinedMedicalRetriever computed
    for this specific chunk (src/pipeline.py) -- shown so a citation can
    be judged rather than trusted blindly: a 0.83 match backing a claim is
    a very different thing from a 0.21 match that barely cleared the
    relevance floor. None for a cached answer, since a cache hit doesn't
    re-run retrieval.
    """
    sources = []
    seen = set()
    for doc in context_docs or []:
        source = doc.metadata.get("source", "the knowledge base")
        # Deliberately not os.path.basename(): that only splits on the
        # separator of whatever OS this code happens to be running on. A
        # document indexed from a Windows machine at some point can leave
        # "data\\seed\\depression.pdf" (backslash) baked into Pinecone's
        # stored metadata; running that through basename() on a Linux
        # server (which only treats "/" as a separator) returns the
        # string unchanged -- producing a second, differently-labeled
        # "duplicate" citation for a source that's already shown under
        # its clean name. Splitting on both separators, on any OS, avoids
        # that regardless of which platform originally built the index.
        label = re.split(r"[\\/]", source)[-1] if source else "the knowledge base"
        page_number = resolve_page_display(doc.metadata)
        score = doc.metadata.get("retrieval_score")
        key = (label, page_number)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": label, "page": page_number, "score": score})
    return sources


def create_app(pipeline=None, cache=None, telemetry=None, document_store=None):
    """Application factory.

    pipeline: an object exposing `.embeddings` (with `.embed_query`),
    `.chain` (a LangChain Runnable supporting `.invoke()`/`.stream()`), and
    `.vectorstore` (supports `.add_documents(docs, ids=, namespace=)` and
    `.delete(ids=, namespace=)` — used by the /documents routes below).
    If not given, the real Pinecone/Groq pipeline is built — this requires
    PINECONE_API_KEY and GROQ_API_KEY.

    telemetry: an object exposing `.init_db()`, `.log_query(...)`,
    `.log_feedback(...)`, `.get_stats()`. If not given, the real Postgres
    backend is built — this requires DATABASE_URL.

    document_store: an object exposing `.init_db()`, `.add_document(...)`,
    `.list_documents()`, `.get_document(id)`, `.delete_document(id)` — the
    manifest of what's been uploaded (see src/document_store.py). If not
    given, the real Postgres backend is built, same as telemetry, and for
    the same reason: this piggybacks on the one Postgres database the app
    already needs rather than introducing new infrastructure.

    Tests pass in fakes for all of these, so the test suite never needs
    real credentials, a real database, or network access, and runs in well
    under a second. That fake-injection is also exactly how create_app()
    decides whether to auto-seed data/seed/*.pdf on startup (see below):
    only when *both* pipeline and document_store are left as None, i.e.
    only when this is genuinely building the real Pinecone/Postgres-backed
    app rather than a test harness's stand-ins.
    """
    # Captured before the two "if x is None: x = build the real thing"
    # blocks below reassign these names -- see the auto-seed call at the
    # end of this function.
    pipeline_was_injected = pipeline is not None
    document_store_was_injected = document_store is not None

    app = Flask(__name__)

    # Defense in depth alongside src/documents.py's own size check: Flask
    # rejects anything over this at the request-body level (a clean 413,
    # handled by the error handler below) before the bytes are even fully
    # read into memory, which matters on a free host where RAM is tight.
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + (1 * 1024 * 1024)

    # Off by default: a single-service deployment (Flask serving
    # templates/chat.html and static/app.jsx itself, the simplest setup --
    # see DEPLOYMENT.md) is same-origin already and needs no CORS at all.
    # Set ALLOWED_ORIGINS (comma-separated) only when the frontend is
    # deployed separately, e.g. a static Netlify/Vercel site calling this
    # API cross-origin as part of the multi-service split.
    allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if allowed_origins:
        CORS(app, origins=allowed_origins, supports_credentials=False)

    if pipeline is None:
        from src.pipeline import build_pipeline  # lazy: keeps this out of the test import path

        pipeline = build_pipeline()

    if cache is None:
        cache = SemanticCache()

    if telemetry is None:
        # Both construction AND init_db() can fail here (the connection
        # pool connects eagerly when constructed, not just when init_db()
        # runs) -- e.g. Neon's free tier scales to zero when idle and can
        # take a moment to wake up, or be briefly unreachable. Telemetry
        # is a nice-to-have, not core functionality, so any failure here
        # falls back to an in-memory stand-in rather than taking the whole
        # chat app down with it: /stats and feedback just won't persist
        # until the app is restarted with a reachable database.
        try:
            from src.telemetry import PostgresTelemetry  # lazy: keeps psycopg2 out of the test import path

            telemetry = PostgresTelemetry()
            telemetry.init_db()
        except Exception:
            app.logger.exception("Could not initialize Postgres telemetry — falling back to in-memory (not persisted)")
            from src.telemetry import InMemoryTelemetry

            telemetry = InMemoryTelemetry()
    else:
        try:
            telemetry.init_db()
        except Exception:
            # An explicitly-injected telemetry (e.g. in tests) that fails
            # to init still shouldn't crash the app -- log_query()/
            # log_feedback() are independently fail-safe too (see
            # src/telemetry.py), so the app degrades to "chat works,
            # telemetry doesn't" instead of "nothing works."
            app.logger.exception("Could not initialize telemetry storage — continuing without it")

    if document_store is None:
        try:
            from src.document_store import PostgresDocumentStore  # lazy: keeps psycopg2 out of the test import path

            document_store = PostgresDocumentStore()
            document_store.init_db()
        except Exception:
            app.logger.exception(
                "Could not initialize Postgres document store — falling back to in-memory "
                "(uploaded documents won't survive a restart until a reachable database is configured)"
            )
            from src.document_store import InMemoryDocumentStore

            document_store = InMemoryDocumentStore()
    else:
        try:
            document_store.init_db()
        except Exception:
            app.logger.exception("Could not initialize document store — continuing without it")

    # Seed the knowledge base from data/seed/*.pdf on first boot so a
    # fresh deploy isn't a blank slate (see seed_data.py -- it ingests
    # through the exact same save_and_validate()/ingest_pdf() path as a
    # real upload, and skips any filename already present, so this is
    # cheap and safe to leave running on every restart). Gated to real
    # infra only (see pipeline_was_injected/document_store_was_injected
    # above) so the test suite's fakes are never silently populated with
    # real seed documents. Never allowed to take the app down: a bad seed
    # PDF or an unreachable Pinecone at boot should degrade to "starts up
    # with an empty-ish knowledge base," not "won't start."
    if not pipeline_was_injected and not document_store_was_injected:
        try:
            from seed_data import seed_default_documents  # lazy: only needed for this one-time step

            seeded = seed_default_documents(pipeline.vectorstore, document_store)
            if seeded:
                app.logger.info("Seeded %d default document(s) into the knowledge base", seeded)
        except Exception:
            app.logger.exception("Seeding default documents failed — continuing without them")

    # In-memory rate limiting is fine for a single free-tier instance; at
    # real scale (multiple workers/instances) you'd point this at Redis
    # instead so limits are shared across processes.
    limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["200 per hour"])

    # Flask's default error pages are HTML (a 404 page, a debug traceback
    # page, etc). This app is a JSON/SSE API behind a JS frontend, so an
    # HTML error page would either show up as broken markup or as a lump
    # of raw text in a chat bubble. This handler makes sure *every* error
    # -- a bad route, a bug in a route, an unhandled exception deep in the
    # RAG pipeline -- comes back as JSON instead. The real exception is
    # still logged server-side; the client only ever sees a safe message.
    @app.errorhandler(Exception)
    def handle_any_error(err):
        if isinstance(err, HTTPException):
            return jsonify({"error": err.name.lower().replace(" ", "_"), "message": err.description}), err.code
        app.logger.exception("Unhandled exception in %s", request.path)
        return jsonify({"error": "server_error", "message": GENERIC_ERROR_MESSAGE}), 500

    @app.route("/")
    def index():
        return render_template("chat.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/stats")
    def stats():
        """Powers both the plain-JSON /stats API and the Dashboard view in
        the frontend (static/app.jsx) — same endpoint, no separate route to
        keep in sync. `daily` and the document counts are computed fresh on
        every call rather than cached: this endpoint is for a human looking
        at a dashboard, not a hot path, so the extra couple of queries
        aren't worth the staleness risk of caching them."""
        payload = telemetry.get_stats()
        payload["cache_size"] = cache.stats()["size"]
        payload["daily"] = telemetry.get_daily_stats()
        documents = document_store.list_documents()
        payload["documents_indexed"] = len(documents)
        payload["chunks_indexed"] = sum(d.get("chunk_count", 0) for d in documents)
        return jsonify(payload)

    @app.route("/get", methods=["POST"])
    @limiter.limit("10 per minute")
    def chat():
        """Synchronous JSON endpoint — simplest to call from a script or a
        test, and what the automated eval harness uses."""
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()
        raw_history = data.get("history", [])
        document_ids = _normalize_document_ids(data.get("document_ids"))
        scope = _cache_scope(document_ids)

        if not msg:
            return jsonify({"answer": "Please type a question.", "sources": [], "emergency": False})

        t_start = time.time()

        if detect_emergency(msg):
            telemetry.log_query(msg, None, None, (time.time() - t_start) * 1000, 0, False, True)
            return jsonify({"answer": EMERGENCY_MESSAGE, "sources": [], "emergency": True, "no_info": False})

        try:
            query_vector = pipeline.embeddings.embed_query(msg)
            cached = cache.get(query_vector, scope=scope)
            if cached:
                elapsed = (time.time() - t_start) * 1000
                telemetry.log_query(msg, 0, 0, elapsed, len(cached["sources"]), True, False)
                return jsonify(
                    {
                        "answer": cached["answer"],
                        "sources": cached["sources"],
                        "emergency": False,
                        "cached": True,
                        "no_info": _is_no_context_answer(cached["answer"]),
                    }
                )

            chat_history = build_chat_history(raw_history)
            t_retrieval_start = time.time()
            response = pipeline.chain.invoke(
                {"input": msg, "chat_history": chat_history, "document_ids": document_ids}
            )
            t_end = time.time()

            context_docs = response.get("context")
            sources = extract_sources(context_docs)
            if not context_docs:
                # CombinedMedicalRetriever (src/pipeline.py) already
                # filtered out anything below MIN_SIMILARITY -- an empty
                # context here means nothing in the selected documents (or
                # the whole knowledge base, if none were selected) was even
                # topically related to this question. Overriding the
                # model's answer guarantees this is always reported
                # honestly, regardless of how well the LLM followed the
                # "say you don't know" instruction in src/prompt.py. (The
                # LLM call above still ran — .invoke() resolves the whole
                # chain in one blocking call, so there's no earlier point
                # to intercept on this synchronous path — but what reaches
                # the user is deterministic either way.)
                answer = NO_CONTEXT_MESSAGE
                sources = []
            else:
                answer = response.get("answer") or ""
                if not answer:
                    raise RuntimeError("Empty response from the model")

            cache.set(query_vector, msg, answer, sources, scope=scope)
            telemetry.log_query(
                msg,
                None,  # retrieval/generation aren't separately timed on the sync path
                (t_end - t_retrieval_start) * 1000,
                (t_end - t_start) * 1000,
                len(sources),
                False,
                False,
            )
            return jsonify(
                {
                    "answer": answer,
                    "sources": sources,
                    "emergency": False,
                    "cached": False,
                    "no_info": _is_no_context_answer(answer),
                }
            )

        except Exception:
            app.logger.exception("Error generating an answer in /get")
            return jsonify(
                {"answer": GENERIC_ERROR_MESSAGE, "sources": [], "emergency": False, "error": True, "no_info": False}
            )

    @app.route("/get/stream", methods=["POST"])
    @limiter.limit("10 per minute")
    def chat_stream():
        """Server-Sent-Events endpoint the frontend actually uses — streams
        the answer token-by-token instead of making the user wait for the
        full response, then sends a final 'done' event with sources."""
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()
        raw_history = data.get("history", [])
        document_ids = _normalize_document_ids(data.get("document_ids"))
        scope = _cache_scope(document_ids)

        def sse(payload):
            return f"data: {json.dumps(payload)}\n\n"

        def generate():
            t_start = time.time()

            if not msg:
                yield sse({"type": "done", "sources": [], "emergency": False, "no_info": False})
                return

            if detect_emergency(msg):
                yield sse({"type": "chunk", "content": EMERGENCY_MESSAGE})
                yield sse({"type": "done", "sources": [], "emergency": True, "no_info": False})
                telemetry.log_query(msg, None, None, (time.time() - t_start) * 1000, 0, False, True)
                return

            # Everything below can fail (Pinecone down, Groq down, a network
            # blip). Once this response has started streaming, Flask/Werkzeug
            # can no longer turn it into a normal 500 JSON response -- the
            # headers are already sent as 200 text/event-stream. So failures
            # here MUST be handled inside the generator itself: caught, logged
            # server-side, and turned into a clean "error" event the frontend
            # already knows how to render as a friendly message. Without this,
            # a mid-stream crash would just silently truncate the response.
            try:
                query_vector = pipeline.embeddings.embed_query(msg)
                cached = cache.get(query_vector, scope=scope)
                if cached:
                    yield sse({"type": "chunk", "content": cached["answer"]})
                    yield sse(
                        {
                            "type": "done",
                            "sources": cached["sources"],
                            "emergency": False,
                            "cached": True,
                            "no_info": _is_no_context_answer(cached["answer"]),
                        }
                    )
                    elapsed = (time.time() - t_start) * 1000
                    telemetry.log_query(msg, 0, 0, elapsed, len(cached["sources"]), True, False)
                    return

                chat_history = build_chat_history(raw_history)
                full_answer = ""
                sources = []
                t_first_chunk = None
                context_checked = False
                no_context = False

                # A plain `for chunk in pipeline.chain.stream(...)` would
                # work too, but breaking out of it early leaves the
                # abandoned generator for the garbage collector to close
                # at some later, unpredictable time -- possibly from a
                # worker thread inside the ThreadPoolExecutor LangChain's
                # own RunnableAssign uses internally for streaming
                # (see langchain_core/runnables/passthrough.py), which can
                # raise "cannot join current thread" while tearing itself
                # down. Harmless (Python reports it as an ignored
                # exception during GC, not a real failure -- everything
                # above still completes correctly), but it's exactly the
                # kind of alarming-looking noise that has no business in
                # production logs. Holding the generator in its own
                # variable and closing it explicitly, immediately, from
                # this same thread and call stack the moment we're done
                # with it avoids that entirely.
                chain_stream = pipeline.chain.stream(
                    {"input": msg, "chat_history": chat_history, "document_ids": document_ids}
                )
                try:
                    for chunk in chain_stream:
                        if "context" in chunk and not context_checked:
                            context_checked = True
                            sources = extract_sources(chunk["context"])
                            if not chunk["context"]:
                                # Nothing cleared MIN_SIMILARITY across the
                                # documents this question was allowed to
                                # search (every document, or just the
                                # selected subset -- see
                                # CombinedMedicalRetriever.search() in
                                # src/pipeline.py) -- stop pulling from this
                                # generator right here rather than letting the
                                # chain proceed into answer generation. This
                                # usually means the Groq call for an answer
                                # never even gets dispatched (build_pipeline's
                                # chain resolves the context stage fully
                                # before generation can start), but that part
                                # is a best-effort latency/cost optimization,
                                # not a guarantee -- see
                                # build_conversational_chain()'s docstring in
                                # src/pipeline.py for a nuance in LangChain's
                                # own streaming internals at 3+ chained
                                # .assign() steps that means it occasionally
                                # still fires anyway. What *is* guaranteed,
                                # unconditionally: breaking here means this
                                # loop never asks the generator for another
                                # chunk, so no answer text -- real or
                                # fabricated -- can ever reach the code below
                                # this point, regardless of what the chain
                                # computed internally after we stopped
                                # watching.
                                no_context = True
                                break
                        if "answer" in chunk:
                            if t_first_chunk is None:
                                t_first_chunk = time.time()
                            full_answer += chunk["answer"]
                            yield sse({"type": "chunk", "content": chunk["answer"]})
                finally:
                    chain_stream.close()

                if no_context:
                    full_answer = NO_CONTEXT_MESSAGE
                    sources = []
                    yield sse({"type": "chunk", "content": full_answer})
                elif not full_answer:
                    raise RuntimeError("Empty response from the model")

                t_end = time.time()
                cache.set(query_vector, msg, full_answer, sources, scope=scope)
                yield sse(
                    {
                        "type": "done",
                        "sources": sources,
                        "emergency": False,
                        "cached": False,
                        "no_info": _is_no_context_answer(full_answer),
                    }
                )

                retrieval_ms = ((t_first_chunk - t_start) * 1000) if t_first_chunk else None
                generation_ms = ((t_end - t_first_chunk) * 1000) if t_first_chunk else None
                telemetry.log_query(
                    msg, retrieval_ms, generation_ms, (t_end - t_start) * 1000, len(sources), False, False
                )

            except Exception:
                app.logger.exception("Error generating a streamed answer for: %r", msg)
                yield sse({"type": "error", "message": GENERIC_ERROR_MESSAGE})
                yield sse({"type": "done", "sources": [], "emergency": False, "no_info": False})
                telemetry.log_query(msg, None, None, (time.time() - t_start) * 1000, 0, False, False)

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    @app.route("/feedback", methods=["POST"])
    def feedback():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        answer = (data.get("answer") or "").strip()
        rating = data.get("rating")

        if rating not in ("up", "down") or not question or not answer:
            return jsonify({"ok": False, "error": "invalid feedback payload"}), 400

        telemetry.log_feedback(question, answer, rating)
        return jsonify({"ok": True})

    @app.route("/documents/upload", methods=["POST"])
    @limiter.limit("10 per hour")
    def upload_document():
        """Add a PDF to the knowledge base. Embedding + upserting runs
        synchronously (no background job queue in this project — see
        README), so this can take anywhere from a couple of seconds to
        roughly a minute for a large file on a slow free-tier CPU; the
        frontend shows an indexing spinner and uses a longer timeout than
        a normal chat turn to match."""
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "no_file", "message": "No file was sent."}), 400

        file_storage = request.files["file"]

        try:
            doc_id, saved_path, display_name = save_and_validate(file_storage)
        except InvalidUpload as e:
            return jsonify({"ok": False, "error": "invalid_file", "message": str(e)}), 400

        try:
            chunks, vector_ids = ingest_pdf(doc_id, saved_path, display_name)
            if not chunks:
                raise RuntimeError(
                    "No extractable text was found in this PDF — it may be a scanned "
                    "image with no text layer."
                )
            pipeline.vectorstore.add_documents(chunks, ids=vector_ids, namespace=DOCUMENTS_NAMESPACE)
        except Exception as e:
            app.logger.exception("Failed to ingest uploaded PDF %r", display_name)
            try:
                os.remove(saved_path)  # don't leave a half-ingested file on disk
            except OSError:
                pass
            message = str(e) if isinstance(e, RuntimeError) else "Couldn't process that PDF. Please try a different file."
            return jsonify({"ok": False, "error": "ingest_failed", "message": message}), 500

        page_count = len({c.metadata.get("page") for c in chunks})
        document_store.add_document(
            doc_id=doc_id,
            filename=display_name,
            chunk_count=len(chunks),
            page_count=page_count,
            vector_ids=vector_ids,
        )

        # The knowledge base just changed -- a question that was cached
        # as "I don't have information about that" five minutes ago might
        # be answerable from this new document now. See SemanticCache.clear().
        cache.clear()

        return jsonify(
            {
                "ok": True,
                "document": {
                    "id": doc_id,
                    "filename": display_name,
                    "chunk_count": len(chunks),
                    "page_count": page_count,
                },
            }
        )

    @app.route("/documents", methods=["GET"])
    def list_documents():
        return jsonify({"documents": document_store.list_documents()})

    @app.route("/documents/<doc_id>", methods=["DELETE"])
    def delete_document(doc_id):
        doc = document_store.get_document(doc_id)
        if not doc:
            return jsonify({"ok": False, "error": "not_found", "message": "No document with that id."}), 404

        try:
            pipeline.vectorstore.delete(ids=doc["vector_ids"], namespace=DOCUMENTS_NAMESPACE)
        except Exception:
            # Logged and swallowed rather than failing the request: if we
            # bail out here, the document stays stuck in the sidebar
            # forever with no way to remove it even though the user asked
            # to. Worst case a few orphaned vectors remain in Pinecone
            # under a namespace the manifest no longer references them
            # from — not ideal, but the app staying usable is more
            # important than that for a project at this scale.
            app.logger.exception("Failed to delete Pinecone vectors for document %s", doc_id)

        document_store.delete_document(doc_id)
        document_store.mark_deleted(doc_id)
        remove_uploaded_file(doc_id)

        # Same reasoning as the upload route, in reverse: a cached answer
        # that leaned on this document's content is now wrong if it's no
        # longer part of the retrievable knowledge base. Cleared
        # unconditionally, even if the Pinecone delete above failed --
        # if those vectors are in fact still there, a fresh retrieval
        # will just find them again, so clearing here is never harmful,
        # only ever either correct or a no-op.
        cache.clear()

        return jsonify({"ok": True})

    return app


if __name__ == "__main__":
    app = create_app()
    # Most free hosts (Render, Hugging Face Spaces, Railway...) inject a PORT
    # env var and expect the app to listen on it.
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
