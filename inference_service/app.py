"""
inference_service/app.py — the embedding + reranking half of the split
deployment (see DEPLOYMENT.md).

A deliberately tiny, standalone Flask app -- no LangChain, no Pinecone,
no Groq, no Postgres, nothing from the rest of this repo. Its only job is
to load sentence-transformers' all-MiniLM-L6-v2 (embeddings) and
cross-encoder/ms-marco-MiniLM-L-6-v2 (reranking) exactly once at startup
and serve them as two endpoints:

  POST /embed   {"texts": ["...", "..."]}  -> {"embeddings": [[...], ...]}
  POST /rerank  {"query": "...", "documents": ["...", "..."]} -> {"scores": [...]}
  GET  /health  -> {"status": "ok", "models_loaded": true}

The main app (src/helper.py's RemoteEmbeddings/RemoteReranker) is the
only intended caller, and only when it's been pointed here via
EMBEDDING_SERVICE_URL -- see that file's module-level comment for the
full picture of why this split exists. This service and the main app
otherwise know nothing about each other beyond that one HTTP contract on
purpose, so either side can be redeployed, rescaled, or even swapped out
independently.

Deployed as its own free-tier instance (Render/Koyeb — see
DEPLOYMENT.md), separate from the main app's instance, so the ~few
hundred MB torch + two small transformer models need resident in memory
get their own dedicated 512MB rather than competing with
Flask/LangChain/the Groq and Pinecone clients for one shared 512MB, which
is what forced the reranker to be disabled in production before this
split existed.

Run directly for local dev / in a container:
    python3 inference_service/app.py
(reads PORT from the environment, defaulting to 8081 -- see
inference_service/Dockerfile)

IMPORTANT: Models are now loaded EAGERLY at startup, not lazily.
This means the container will take longer to become "ready" (usually
10-40s depending on cold cache), but it will never accept a request
before the embedding model is fully resident in memory. The /health
endpoint will return 503 until loading is complete, so Render/Koyeb
keep retrying and don't mark the service live prematurely.
"""

import os
import time
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

# ------------- GLOBALS -------------
_embedder = None
_reranker = None

# Set LOAD_RERANKER=true in your environment ONLY if you have >1GB RAM.
# On 512MB Render instances, leave it false (default) to avoid OOM.
_LOAD_RERANKER = os.getenv("LOAD_RERANKER", "false").lower() == "true"


# ------------- AUTH (mirrors src/helper.py) -------------
def _check_auth():
    """Return True if no token is set, or if the Authorization header matches."""
    expected = os.environ.get("INFERENCE_SERVICE_TOKEN", "")
    if not expected:
        return True
    got = request.headers.get("Authorization", "")
    return got == f"Bearer {expected}"


# ------------- FLASK APP -------------
def create_app():
    app = Flask(__name__)

    @app.route("/")
    def root():
        return jsonify({"service": "inference_service", "status": "ok"})

    @app.route("/health")
    def health():
        """
        Ready only when embedder is loaded AND (if reranker enabled) reranker loaded.
        Returns 503 while loading so the platform keeps polling until ready.
        """
        ready = _embedder is not None and (not _LOAD_RERANKER or _reranker is not None)
        status_code = 200 if ready else 503
        return jsonify({
            "status": "ok" if ready else "loading",
            "models_loaded": ready,
            "reranker_enabled": _LOAD_RERANKER
        }), status_code

    @app.route("/embed", methods=["POST"])
    def embed():
        if not _check_auth():
            return jsonify({"error": "unauthorized"}), 401
        if _embedder is None:
            return jsonify({"error": "models still loading, try again shortly"}), 503

        data = request.get_json(silent=True) or {}
        texts = data.get("texts")
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return jsonify({"error": "'texts' must be a list of strings"}), 400
        if not texts:
            return jsonify({"embeddings": []})

        vectors = _embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return jsonify({"embeddings": vectors.tolist()})

    @app.route("/rerank", methods=["POST"])
    def rerank():
        # If reranker is disabled globally, return 501 immediately
        if not _LOAD_RERANKER:
            return jsonify({
                "error": "reranker is disabled on this instance (set LOAD_RERANKER=true to enable)"
            }), 501

        if not _check_auth():
            return jsonify({"error": "unauthorized"}), 401

        if _reranker is None:
            return jsonify({"error": "reranker still loading, try again shortly"}), 503

        data = request.get_json(silent=True) or {}
        query = data.get("query")
        documents = data.get("documents")
        valid = isinstance(query, str) and isinstance(documents, list) and all(isinstance(d, str) for d in documents)
        if not valid:
            return jsonify({"error": "'query' must be a string and 'documents' a list of strings"}), 400
        if not documents:
            return jsonify({"scores": []})

        scores = _reranker.predict([(query, doc) for doc in documents])
        return jsonify({"scores": [float(s) for s in scores]})

    @app.errorhandler(Exception)
    def handle_error(err):
        if isinstance(err, HTTPException):
            return err
        app.logger.exception("inference_service error")
        return jsonify({"error": "internal error"}), 500

    return app


# ------------- BOOT: LOAD MODELS EAGERLY -------------
print("[inference_service] Loading models at startup...", flush=True)
t0 = time.time()

# Load embedding model
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"[inference_service] Embedding model ready in {time.time() - t0:.1f}s", flush=True)

# Optionally load reranker
if _LOAD_RERANKER:
    from sentence_transformers import CrossEncoder
    from torch.nn import Sigmoid
    print("[inference_service] Loading cross-encoder/ms-marco-MiniLM-L-6-v2 ...", flush=True)
    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", activation_fn=Sigmoid())
    print(f"[inference_service] Reranker ready in {time.time() - t0:.1f}s", flush=True)
else:
    print("[inference_service] Reranker disabled (LOAD_RERANKER not set to 'true')", flush=True)

# Create the Flask app only after models are loaded
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    # For local dev only. In production, use Gunicorn (see Dockerfile).
    # Render sets WEB_CONCURRENCY=1 by default, which is fine.
    app.run(host="0.0.0.0", port=port)