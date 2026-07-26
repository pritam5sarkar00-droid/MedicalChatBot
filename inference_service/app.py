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
"""

import os
import time

from flask import Flask, jsonify, request

_embedder = None
_reranker = None
_models_loaded = False


def _load_models():
    """Loads both models into the module-level globals above. Called
    once, unconditionally, right after create_app() below -- not lazily
    on first request -- so that a request arriving right after startup
    doesn't pay a multi-second model-load penalty on top of whatever a
    free-tier cold start already cost, and so /health only reports ready
    once these are genuinely usable rather than merely "the Flask process
    is up." Takes tens of seconds on a free-tier CPU the first time (both
    models download from HuggingFace's hub and are cached under
    ~/.cache/huggingface/ for every run after that)."""
    global _embedder, _reranker, _models_loaded

    from sentence_transformers import CrossEncoder, SentenceTransformer
    from torch.nn import Sigmoid

    print("[inference_service] loading sentence-transformers/all-MiniLM-L6-v2 ...", flush=True)
    t0 = time.time()
    _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"[inference_service] embedding model ready in {time.time() - t0:.1f}s", flush=True)

    print("[inference_service] loading cross-encoder/ms-marco-MiniLM-L-6-v2 ...", flush=True)
    t0 = time.time()
    # Sigmoid activation for the same reason src/helper.py's
    # download_reranker() uses it locally: MS MARCO cross-encoders output
    # raw, unbounded logits by default, not 0-1 scores. Keeping this
    # consistent with the local/in-process code path matters here more
    # than almost anywhere else in the split -- CombinedMedicalRetriever
    # only ever uses these scores to *sort* candidates (sigmoid is
    # monotonic, so ranking is identical either way), but a caller
    # graphing or thresholding raw scores directly would see very
    # different numbers from the two modes if this ever drifted from
    # src/helper.py's choice.
    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", activation_fn=Sigmoid())
    print(f"[inference_service] reranker ready in {time.time() - t0:.1f}s", flush=True)

    _models_loaded = True


def _check_auth():
    """Mirrors src/helper.py's _remote_auth_headers(): if
    INFERENCE_SERVICE_TOKEN is set in this service's environment, every
    /embed and /rerank request must carry a matching bearer token (set
    the same value on the main app's side too). Unset by default, so
    local dev and low-stakes demo deployments work with zero extra setup
    -- this only exists because, once deployed, this service sits on its
    own public URL that anyone could otherwise send free requests to."""
    expected = os.environ.get("INFERENCE_SERVICE_TOKEN", "")
    if not expected:
        return True
    got = request.headers.get("Authorization", "")
    return got == f"Bearer {expected}"


def create_app():
    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok" if _models_loaded else "loading", "models_loaded": _models_loaded})

    @app.route("/embed", methods=["POST"])
    def embed():
        if not _check_auth():
            return jsonify({"error": "unauthorized"}), 401
        if not _models_loaded:
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
        if not _check_auth():
            return jsonify({"error": "unauthorized"}), 401
        if not _models_loaded:
            return jsonify({"error": "models still loading, try again shortly"}), 503

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
        app.logger.exception("inference_service error")
        return jsonify({"error": "internal error"}), 500

    return app


app = create_app()
_load_models()  # unconditional at import time -- see _load_models()'s
                 # docstring for why this isn't deferred to first-request
                 # or gated behind `if __name__ == "__main__"` (a WSGI
                 # server importing this module, e.g. `gunicorn app:app`,
                 # needs models loaded just as eagerly as running this
                 # file directly does).

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    app.run(host="0.0.0.0", port=port)
