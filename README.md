# MediCare AI

**A production-shaped RAG medical chatbot — built by Pritam**

MediCare AI answers general health questions by retrieving relevant passages
from a knowledge base of medical PDFs and asking an LLM to answer *using only
that retrieved context* (Retrieval-Augmented Generation). Beyond the core RAG
pipeline, it's built with the kind of engineering scaffolding — evaluation,
tests, observability, caching, streaming — that separates a wired-together
demo from a system someone actually thought about running in production.

> ⚠️ **Disclaimer:** This project shares general medical information for
> educational purposes only. It is not a substitute for professional medical
> advice, diagnosis, or treatment.

---

## ✨ Features

- **Conversational RAG** with real multi-turn memory — a follow-up like
  "explain in more detail" is rewritten into a standalone question using
  chat history (e.g. "explain asthma in more detail"), and that resolved
  question drives *both* retrieval *and* the final answer, not just
  retrieval — see "How retrieval works" and Engineering highlights below
  for why that distinction is the whole fix for a real "forgets what we
  were just discussing" bug this project used to have.
- **Two-stage retrieval with cross-encoder reranking** — embedding
  similarity casts a wide net, then a small cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, reusing the
  `sentence-transformers` dependency already needed for embeddings — no
  new package) re-scores that pool by jointly encoding each
  (question, chunk) pair, a meaningfully stronger relevance signal than
  comparing two independently-built embeddings. Toggleable
  (`build_pipeline(use_reranker=...)`) and fails open to plain
  embedding-score retrieval if the model can't load. Can run in-process
  or as its own service (see "Deploying as multiple services" below) —
  the split is what lets both models stay on comfortably within
  free-tier RAM instead of trading quality for headroom.
- **One knowledge base, built from PDFs you add or that ship with it** —
  six small public-domain fact sheets (diabetes, high blood pressure,
  asthma, depression, headache, and high cholesterol — see `data/seed/`) come pre-indexed, and uploading
  more (the paperclip icon in the composer, or the "+" in the sidebar)
  adds to the exact same knowledge base, indexed and cited identically —
  there's no special-cased "reference" document behind the scenes. Every
  document in the sidebar can be clicked to open the actual PDF in a new
  tab (`GET /documents/<id>/file`, served inline so the browser renders
  it directly rather than downloading it) or removed entirely — seeded
  and uploaded documents behave identically for both. If nothing in the
  knowledge base has anything relevant to a question, you get a plain "I
  don't have information about that" instead of a guess — see "How
  retrieval works" below for how that's enforced.
- **Answer from a chosen subset of documents** — tick specific documents
  in the sidebar to scope a question to just those (e.g. "only answer from
  the diabetes fact sheet"), or leave everything ticked to search the
  whole knowledge base. The semantic cache (below) is scoped the same way,
  so an answer computed for one document selection never leaks into a
  different one.
- **Streamed responses** — answers arrive token-by-token over
  Server-Sent-Events instead of making the user wait for the full reply.
- **Semantic response cache** — near-duplicate questions ("what is
  asthma" vs "can you explain asthma to me") hit the same cache entry via
  embedding cosine-similarity, skipping retrieval *and* generation, scoped
  per document selection (above). The whole cache is invalidated the
  moment a document is uploaded or deleted, so a stale answer can never
  keep citing a document that's gone or keep saying "no information"
  about something a new upload now covers — see `SemanticCache.clear()`.
- **Visible source citations** on every answer (document + page number,
  with the retrieval similarity score on hover) — and the page number is
  something the model itself is actually told, not just a UI-only
  afterthought, so it can answer "what page is that on?" directly too.
- **Emergency guardrail** — a keyword check runs *before* the LLM; a
  possible medical/mental-health emergency short-circuits straight to
  real helpline numbers.
- **An Analytics dashboard** (sidebar → Analytics) — query volume over
  the last 14 days, cache hit rate, latency broken down by retrieval vs
  generation, and 👍/👎 feedback, all charted with Chart.js against the
  same `/stats` endpoint below. Built to be the kind of thing worth
  screenshotting for a portfolio, not just a raw JSON blob.
- **Observability** — every turn logs a latency breakdown (retrieval vs
  generation), cache hits, and emergency triggers to Postgres, surfaced via
  a `/stats` endpoint (JSON) and the Analytics dashboard above (charts).
- **Rate limiting** to protect the free LLM quota from abuse.
- **An evaluation harness** (`eval/run_eval.py`) that measures retrieval
  hit-rate, answer keyword coverage, and precision/recall/F1/accuracy on
  the model's answer-or-refuse decision against a fixed test set — real
  numbers, not "it seems to work." See "Running the evaluation harness"
  below.
- **A pytest suite** that runs the whole API with fake dependencies
  (RAG pipeline, cache, and database), so tests need zero API keys, no
  Postgres instance, and run in under a second.
- **Voice input & read-aloud** via the Web Speech API, a ChatGPT-style
  sidebar for multiple saved conversations (`localStorage`), light/dark
  theme, and a 👍/👎 feedback control on every answer.
- Runs entirely on **free-tier infrastructure** — Pinecone, Groq, Neon
  Postgres, and Render (or Koyeb) for either one service or a split
  multi-service deployment, no credit card anywhere — see
  `DEPLOYMENT.md`.

---

## 🧠 How it works

```
data/seed/*.pdf (bundled)                  a PDF you upload in the app
        │  indexed automatically                     │  (any time, from the UI)
        │  on first startup                           │
        ▼                                              ▼
   seed_data.py                              POST /documents/upload
        │  chunks (~1000 chars, 150 overlap)          │  src/documents.py: validate,
        │  → embeddings (in-process or                │  load, chunk (same scheme)
        │     inference_service/, see below)           │
        ▼                                              ▼
   Pinecone index "pritam-medical-chatbot", namespace "documents"  ◄────┘
   one shared namespace — every document lives here identically,
   listed in one manifest (src/document_store.py), deletable the same way
        ▲
        │  CombinedMedicalRetriever (src/pipeline.py): searches the
        │  namespace (optionally filtered to a chosen doc_ids subset —
        │  see "Answer from a chosen subset of documents" above),
        │  optionally reranks with a cross-encoder,
        │  drops anything below a similarity floor
        │
     app.py (Flask)          build_conversational_chain()  →  CombinedMedicalRetriever
        │  ┌─ src/safety.py     (emergency check, runs first)
        │  ├─ src/cache.py      (semantic cache, scoped per document selection)
        │  ├─ src/pipeline.py   (retrieval + Groq generation, on a cache miss —
        │  │                     an empty result short-circuits to an
        │  │                     honest "no information" answer, no LLM call)
        │  ├─ src/document_store.py (manifest of every document — seeded or
        │  │                     uploaded — for GET /documents and
        │  │                     DELETE /documents/<id>)
        │  └─ src/telemetry.py  (logs latency/cache/emergency for every turn)
        ▼
  templates/chat.html + static/app.jsx    ◄── SSE stream ──  openai/gpt-oss-120b
  (chat UI, browser)                                          via Groq
```

Embeddings and reranking (the two boxes needing `sentence-transformers`/
`torch`) can run **in this same process**, or as a separate
`inference_service/` deployment reached over HTTP
(`EMBEDDING_SERVICE_URL`) — see "Deploying as multiple services" below.
Neither the diagram nor the request flow below changes based on which;
it's purely a deployment-time choice.

**Every chat turn, in order:**

1. Browser POSTs `{ message, history, document_ids }` to `/get/stream` (or
   `/get` for a plain JSON response — used by the eval harness and
   tests). `document_ids` is optional — omit it (or leave every sidebar
   checkbox ticked) to search the whole knowledge base.
2. `src/safety.py` checks for an emergency phrase. If matched, the LLM is
   skipped entirely.
3. The question is embedded once; `src/cache.py` checks it against
   previously-answered questions (with the same document selection) by
   cosine similarity. A hit returns instantly, skipping retrieval and
   generation.
4. On a miss: `build_conversational_chain()` (`src/pipeline.py`) rewrites
   the question into a standalone one using chat history if there is any
   (skipped entirely on a conversation's first message — nothing to
   resolve yet). That resolved question drives everything downstream:
   `CombinedMedicalRetriever` searches Pinecone (narrowed to
   `document_ids` if given), optionally reranks the pool with a
   cross-encoder, and drops anything below a similarity floor — and the
   *same* resolved question (not the user's raw, possibly ambiguous
   follow-up) is what the final answering call sees too. If nothing
   clears the similarity floor, the answer becomes a fixed "I don't have
   information about that" — no LLM call needed. Otherwise
   `create_stuff_documents_chain` asks openai/gpt-oss-120b (via Groq) to answer
   using only that context (each chunk labeled with its source and page,
   so the model can name one back if asked) — streamed back
   chunk-by-chunk.
5. `src/telemetry.py` logs the question, latency breakdown, cache hit, and
   source count to Postgres — this is what `/stats` reports.

### 📎 The knowledge base: seeded documents, uploads, and selection

Every document in the knowledge base — whether it's one of the ones
bundled in `data/seed/` (indexed automatically on first startup by
`seed_data.py`) or something uploaded through the paperclip icon or the
"+" in the sidebar — goes through the exact same path: `src/documents.py`
loads and chunks it, embeds it with the same model, and upserts it into
Pinecone's `documents` namespace, tracked in one manifest
(`src/document_store.py`). There's no separate "curated reference" code
path anywhere — every document can be listed, ticked for a specific
question, and deleted from the sidebar identically.

Ticking a subset of documents in the sidebar (instead of leaving
everything ticked) scopes the *next* question to just those — sent as
`document_ids` in the request, filtered at the Pinecone query itself
(`CombinedMedicalRetriever.search()` in `src/pipeline.py`), not as a
slower post-hoc filter over already-fetched results.

A few things worth knowing:

- **Two layers keep "no information" honest, not one.**
  `CombinedMedicalRetriever`'s similarity floor catches the coarse case —
  a question that's nowhere in the selected document(s) at all — before
  an LLM call is even made. `src/prompt.py`'s system prompt handles the
  subtler case: context *was* retrieved (it's topically related) but
  doesn't actually answer what was asked, which needs real language
  understanding, not a cosine cutoff, to judge. Relying on either one
  alone would be weaker — a prompt instruction the model can occasionally
  ignore, or a similarity cutoff that can't tell "related" from "actually
  answers this."
- **Everything is shared, not private to you.** There's no login system
  anywhere in this app (see Future scope), so a document you upload joins
  a knowledge base every visitor to that deployed instance can ask
  questions against — the same trust model the rest of the app already
  has (one shared cache, one shared telemetry log). Don't upload anything
  sensitive. Scoping uploads per-user is a natural next step once accounts
  exist.
- **Every upload or delete wipes the semantic response cache.** Without
  this, a question cached as "I don't have information about that" could
  keep getting served from cache even after you upload a document that
  answers it — or worse, a cached answer that cited a document could keep
  being served after you delete that exact document. `cache.clear()`
  runs on both `/documents/upload` and `DELETE /documents/<id>` (but not
  on a *failed* upload or a 404'd delete, since those never actually
  changed anything worth invalidating for). See
  `tests/test_documents_routes.py`'s cache-invalidation tests for the
  regression coverage.
- **Deleting a bundled `data/seed/` document stays deleted.** Without
  anything else, the *next* restart (a redeploy, or just a free-tier host
  waking back up from an idle sleep) would look identical, from
  `seed_data.py`'s point of view, to "this was never seeded" — and
  silently re-add it. `document_store.mark_deleted()`/`was_deleted()`
  (a small tombstone table) is what makes a deletion permanent regardless
  of how the app's own startup-time seeding behaves afterward — see
  `src/document_store.py`'s module docstring.

### 📊 Analytics dashboard

The sidebar's **Analytics** button swaps the chat view for a small
dashboard (`static/app.jsx`'s `Dashboard` component) built on the same
`/stats` endpoint the plain-JSON API already returns:

- KPI cards — total queries, average response time, cache hit rate,
  documents indexed.
- A 14-day query volume line chart (`telemetry.get_daily_stats()` —
  zero-filled per day so the chart never has gaps, even on days with no
  traffic).
- A retrieval-vs-generation latency bar chart.
- Cache hit/miss and 👍/👎 feedback donut charts.

Charts are drawn with [Chart.js](https://www.chartjs.org/) (one CDN
`<script>` tag in `templates/chat.html`, the same no-build-step pattern
already used for React/Babel/marked — see that file's comment on why the
UMD build specifically, not the ES-module one). Colors are read from
`theme.css`'s CSS custom properties at chart-build time via
`getComputedStyle`, so the dashboard follows light/dark mode automatically
instead of hardcoding a second palette. With zero queries logged yet
(a fresh deployment), the charts that need real data show a plain
"ask a few questions to see this" placeholder instead of an empty grid.

---

## 🛠️ Tech stack

| Layer               | Technology                                                        |
|----------------------|--------------------------------------------------------------------|
| LLM                  | openai/gpt-oss-120b via **Groq** (`langchain-groq`) — free, no card      |
| Orchestration        | LangChain (RAG chains, history-aware retriever, streaming)        |
| Embeddings           | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — in-process or via `inference_service/` |
| Vector database       | Pinecone (serverless, free Starter plan) — one index, one shared namespace for every document |
| Caching              | In-memory semantic cache (numpy cosine similarity), scoped per document selection |
| Backend              | Flask (app-factory pattern), REST + SSE — optionally split into a lightweight API + a separate embedding/reranker service |
| Rate limiting        | Flask-Limiter                                                      |
| Observability        | Python `logging` + PostgreSQL (`query_logs`, `feedback`, `uploaded_documents`, `deleted_document_ids` tables) |
| Testing              | pytest, dependency-injected fakes (pipeline, cache, telemetry, document store) |
| Evaluation           | Custom harness — retrieval hit-rate, answer coverage, faithfulness |
| Frontend             | React 18 + Tailwind CSS (via CDN, no build step), `marked.js`, Chart.js, Web Speech API |
| CI/CD                | GitHub Actions — a test job (every push) + an optional AWS deploy job |
| Hosting              | Render (free) or Koyeb (free), as one service or split across two — see `DEPLOYMENT.md` (local-only, not pushed — see Project structure below) |

Every document — seeded or uploaded — shares the same backend
dependencies: `pypdf` / `langchain_community` for loading, the same
chunking scheme, Flask's own multipart form handling for uploads, and one
Pinecone namespace. The Analytics dashboard added exactly one new
dependency overall: Chart.js, loaded via CDN like the rest of the
frontend — no npm, no build step, nothing to `pip install`.

---

## 📁 Project structure

```
MediCare-AI-Pritam/
├── app.py                  # Flask app factory: routes, streaming, caching, rate limiting, uploads, stats
├── seed_data.py             # Indexes data/seed/*.pdf on first startup (auto) or by hand — replaces store_index.py
├── src/
│   ├── pipeline.py         # Builds the RAG chain (Pinecone + Groq) — reused by app + eval
│   │                        #   incl. CombinedMedicalRetriever (one shared namespace, optional
│   │                        #   doc_ids filter, optional cross-encoder reranking) and
│   │                        #   build_conversational_chain (follow-up question resolution)
│   ├── documents.py         # Upload validation + single-PDF ingestion (no Pinecone knowledge)
│   ├── document_store.py    # Manifest of every document — Postgres + in-memory (like telemetry.py) —
│   │                        #   plus the deletion tombstone seed_data.py checks
│   ├── cache.py             # Semantic (embedding-similarity) response cache, scoped per document
│   │                        #   selection, cleared on upload/delete
│   ├── telemetry.py          # Structured logging + Postgres (latency, cache, feedback, daily stats)
│   ├── safety.py             # Emergency keyword guardrail
│   ├── prompt.py             # System prompts (answering + query rewriting)
│   └── helper.py             # PDF loading/chunking, embedding + reranker models (in-process, or
│                              #   via inference_service/ over HTTP if EMBEDDING_SERVICE_URL is set)
├── inference_service/        # Standalone embedding+reranker service for the split deployment
│   ├── app.py                #   POST /embed, POST /rerank, GET /health — see DEPLOYMENT.md
│   ├── requirements.txt
│   └── Dockerfile
├── eval/
│   ├── testset.json          # Questions grounded in the actual bundled data/seed/*.pdf content
│   └── run_eval.py           # Runs the test set against the live pipeline, writes a report
├── tests/
│   ├── test_safety.py        # Unit tests, no API needed
│   ├── test_cache.py          # Unit tests incl. clear()/invalidation/scoping, no API needed
│   ├── test_helper.py          # Unit tests for the PDF ingestion pipeline, no API needed
│   ├── test_telemetry.py        # Stats aggregation + daily time-series bucketing, no API needed
│   ├── test_eval_metrics.py      # Precision/recall/F1/accuracy confusion-matrix logic, no API needed
│   ├── test_documents.py          # Upload validation + ingestion, using a real local seed PDF
│   ├── test_document_store.py      # In-memory document manifest CRUD + deletion tombstone
│   ├── test_seed_data.py            # data/seed/ ingestion: idempotency, tombstone, bad-file isolation
│   ├── test_pipeline_retrieval.py   # CombinedMedicalRetriever merge/threshold/reranking/doc_ids logic
│   ├── test_pipeline_chain.py        # build_conversational_chain: follow-up resolution, doc_ids threading
│   ├── test_extract_sources.py        # Citation de-duplication + cross-platform path handling
│   ├── test_app.py                    # Flask route tests via fake pipeline/cache/telemetry, no API needed
│   ├── test_documents_routes.py        # /documents/* routes, no-context short circuit, cache invalidation
│   └── conftest.py
├── templates/chat.html       # Flask-rendered frontend shell (loads React/Tailwind/Babel/marked/Chart.js
│                              #   from CDN, mounts <div id="root">) — served at "/" by app.py
├── index.html                 # Same shell, Jinja-free — for deploying the frontend as its own static
│                              #   site (Netlify/Vercel/...) instead of Flask serving it; see DEPLOYMENT.md
├── netlify.toml               # Tells Netlify how to package index.html + static/ in isolation from the
│                              #   backend files sitting right next to them; see DEPLOYMENT.md
├── static/
│   ├── theme.css               # CSS custom properties for light/dark theme
│   └── app.jsx                  # The whole React app: chat, document selection, Analytics dashboard, SSE
├── data/
│   ├── seed/                    # Small bundled PDFs, auto-indexed on first startup (see seed_data.py)
│   └── uploads/                  # User-uploaded PDFs land here (git-ignored, created on first upload)
├── requirements.txt                    # Core deps — no torch; enough for the split main-API deployment
├── requirements-local-inference.txt     # Adds sentence-transformers, for running models in-process
├── requirements-dev.txt
├── Dockerfile                 # All-in-one image (app + models in one process)
├── Dockerfile.api              # Lightweight image for the split deployment's main-API half
├── DEPLOYMENT.md             # Hosting steps — local-only, git-ignored, not pushed (see .gitignore)
└── .github/workflows/{tests.yml, cicd.yaml}
```

---

## 🚀 Getting started (local)

```bash
conda create -n medicare python=3.10 -y
conda activate medicare

# CPU-only torch first -- skips ~1-2GB of unused CUDA/NVIDIA packages that
# `pip install -r requirements-local-inference.txt` would otherwise pull in
# on Linux as a dependency of sentence-transformers, even though nothing
# here uses a GPU. This one extra command is usually the difference
# between a fast, reliable install and one that's slow, huge, or fails on
# a disk- or RAM-constrained machine (including most free-tier hosts).
pip install torch --index-url https://download.pytorch.org/whl/cpu

# requirements-local-inference.txt (which pulls in requirements.txt too)
# runs the embedding + reranker models in this same process -- the
# simplest option for local dev. If you're instead pointing this app at a
# separately-running inference_service/ (EMBEDDING_SERVICE_URL set — see
# DEPLOYMENT.md), `pip install -r requirements.txt` alone is enough; skip
# the torch install above too, since this process won't load either model
# itself.
pip install -r requirements-local-inference.txt
```

> The commands above are **one-time setup**. Every time after this,
> you only need `conda activate medicare` (or `venv\Scripts\activate` /
> `source venv/bin/activate` if you used `venv` instead of conda) — not a
> reinstall. Your terminal prompt should show `(medicare)` or `(venv)`
> when it's active; if it doesn't, activation didn't stick.

```bash
cp .env.example .env
# fill in PINECONE_API_KEY, GROQ_API_KEY, and DATABASE_URL — all free, see below

python app.py             # open http://localhost:8080
```

No separate indexing step needed — the PDFs in `data/seed/` are
indexed automatically the first time `app.py` starts (see `seed_data.py`),
and stay indexed on every restart after that without redoing the work
(it skips anything already in the manifest). To index them by hand
instead (or re-index after adding a new file to `data/seed/`), run
`python seed_data.py` directly.

### Setting up Postgres (free, no credit card)

The chat/RAG pipeline needs `PINECONE_API_KEY` and `GROQ_API_KEY`.
Telemetry (`/stats`, feedback) needs a Postgres database:

1. Create a free account at [neon.tech](https://neon.tech) → **Create a
   project** → free plan (0.5GB storage, no card needed).
2. Neon shows you a ready-made connection string right on the dashboard —
   copy it exactly as given (it already includes `?sslmode=require`, no
   reassembly needed) into your `.env`:
   ```
   DATABASE_URL="postgresql://user:password@ep-xxxx.region.aws.neon.tech/dbname?sslmode=require"
   ```
3. That's it — `app.py` creates the `feedback`, `query_logs`, and
   `uploaded_documents` tables automatically on first run
   (`telemetry.init_db()` and `document_store.init_db()`).

Prefer to develop against a local Postgres instead? Any Postgres 12+
server works — just point `DATABASE_URL` at it, e.g.
`postgresql://user:password@localhost:5432/medicare_ai`.

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

No API keys and no database needed — `tests/test_app.py` and
`tests/test_documents_routes.py` inject fakes for the RAG pipeline
(`FakePipeline`), the vector store, and the database
(`InMemoryTelemetry`, `InMemoryDocumentStore`) instead of calling real
Pinecone/Groq/Postgres, so the whole suite runs in well under a second
and passes in CI with zero secrets configured (see
`.github/workflows/tests.yml`). Upload-related tests still exercise real
PDF parsing — they build a small real PDF on the fly from a few pages of
one of the bundled `data/seed/*.pdf` files rather than mocking that part.

### Running the evaluation harness

```bash
python eval/run_eval.py
```

This *does* need real API keys and a populated index, since it's measuring
the live pipeline. It writes `eval/results.md` — put those numbers in your
resume/report instead of a vague "it works well."

Beyond retrieval hit-rate and answer keyword coverage, it also treats
"should this question be answered or refused?" as a binary classification
problem (ground truth: each question's `expect_no_answer` in
`testset.json`) and reports **precision, recall, F1, and accuracy** on
that decision — a false positive there means the model answered something
it should have refused (hallucination risk); a false negative means it
refused something it should have known (over-caution). See
`compute_confusion_matrix()` in `eval/run_eval.py` for the exact
definitions, and `tests/test_eval_metrics.py` for unit coverage of that
logic that runs with zero credentials.

---

## 🎯 Engineering highlights (why these exist)

A few notes on the choices behind this, in case you get asked "why" in an
interview:

- **App factory + dependency injection**
  (`create_app(pipeline=..., cache=..., telemetry=..., document_store=...)`)
  — the Flask app never hard-codes the RAG pipeline or the database.
  Tests inject fakes for all four — including `InMemoryTelemetry` and
  `InMemoryDocumentStore` standing in for real Postgres, so the suite
  never needs a database running. This is the standard pattern for
  unit-testing anything wrapped around a slow/paid/external dependency (a
  DB, an LLM API, a payment gateway — same idea every time).
- **Semantic cache over exact-match cache** — reuses the embedding model
  that's already loaded, so paraphrased questions still hit the cache.
  Trade-off: it's in-memory and process-local, so it resets on restart and
  doesn't share state across multiple instances. At real scale you'd back
  it with Redis.
- **Streaming (SSE) over a single JSON response** — perceived latency
  matters more than actual latency for chat UX; the user sees the first
  words in ~1s instead of waiting 3-5s for the full answer.
- **Emergency check runs before the LLM, not after** — cheaper, faster,
  and doesn't depend on the LLM behaving safely under adversarial input.
- **"No relevant context" is a retrieval-layer decision, not just a
  prompt instruction** — `CombinedMedicalRetriever` (`src/pipeline.py`)
  applies a similarity floor and returns nothing when a question isn't
  even topically related to either knowledge source, and `app.py`
  overrides the answer with a fixed message when that happens — the same
  "cheap deterministic check before trusting the expensive, occasionally-
  wrong LLM" shape as the emergency guardrail above. On the streaming
  path, breaking out of `.stream()` the moment an empty context chunk
  arrives *usually* also means the Groq generation call is never
  dispatched at all — but see the next bullet for why that specific part
  is a best-effort optimization, not a hard guarantee, and why that
  distinction matters.
- **The rewritten question drives the final answer too, not just
  retrieval** — the textbook LangChain pattern
  (`create_history_aware_retriever` + `create_retrieval_chain`) only
  feeds a history-resolved standalone question to the *retriever*; the
  answer-generation call still receives the user's raw, possibly
  ambiguous follow-up ("explain in details") and has to re-resolve
  "explain what?" itself from raw chat history. Confirmed with a small
  instrumented fake-LLM trace (not just theorized) that this really was
  happening, then fixed by restructuring into three chained
  `RunnablePassthrough.assign()` calls — rewrite, then retrieve with the
  rewrite, then answer with the *same* rewrite (`build_conversational_chain()`
  in `src/pipeline.py`) — so the resolved question does double duty
  instead of being thrown away after retrieval, at zero extra LLM calls.
  Chasing that fix surfaced a genuine LangChain internals nuance worth
  knowing about if you ever build something similar: three
  *method-chained* `.assign().assign().assign()` calls preserve the
  generator laziness the streaming short-circuit above depends on,
  but piping separately-built `.assign(...)` results together with `|`
  does not, even though both produce what looks like the same chain
  under `.invoke()`. Caught by writing a test for it
  (`tests/test_pipeline_chain.py`) rather than assuming — worth
  internalizing as a general lesson: an LCEL composition that returns
  the right answer under `.invoke()` hasn't actually proven anything
  about its `.stream()` behavior, since those two code paths can differ
  in ways that only show up as flaky, hard-to-reproduce test failures.
  Abandoning that generator via an early `break` also surfaced a second,
  smaller nuance worth knowing: LangChain's `RunnableAssign` uses a real
  `ThreadPoolExecutor` internally for `.stream()` (not just plain Python
  generators all the way down), so letting garbage collection close an
  abandoned one at some unpredictable later time — possibly from inside
  one of that pool's own worker threads — can raise a harmless-but-
  alarming-looking `RuntimeError: cannot join current thread` straight
  into the server logs. `app.py`'s `chat_stream()` now holds the
  generator in its own variable and closes it explicitly, immediately,
  from the same thread and call stack it broke out of, instead of
  leaving cleanup to GC's timing.
- **Making one rewrite drive two things doubled the cost of it going
  wrong, so it gets validated, not just trusted** — a follow-up
  question's rewrite can itself occasionally go off the rails, especially
  right after an unrelated topic switch (a real example: after asking an
  off-topic question and getting a hallucinated answer, the *next*,
  completely unrelated question got rewritten into commentary about that
  previous answer instead of a clean standalone question — which then
  corrupted retrieval *and* the final answer, since both now depend on
  the same rewrite). `contextualize_q_system_prompt` (`src/prompt.py`) is
  written to avoid this directly (explicit "if it's a new topic, return
  it unchanged" instruction, explicit "output only the question, no
  commentary" instruction), but a prompt is never a guarantee — a second,
  deterministic layer, `_looks_like_a_reasonable_rewrite()`
  (`src/pipeline.py`), sanity-checks the actual output (empty, wildly
  longer than the original, or containing tells like "I made an
  incorrect assumption" / "to correct myself") and falls back to the
  user's own raw message if it looks broken, rather than trusting every
  rewrite unconditionally. `tests/test_pipeline_chain.py` reproduces the
  original bug almost verbatim as a regression test, not just the
  validator's logic in isolation.
- **A retrieved chunk mentioning the right words isn't the same as it
  answering the question** — a real example: asking "what is bike"
  against the knowledge base retrieved a chunk that only mentions a
  *stationary* bike as equipment used during a cardiac stress test, which
  correctly clears the similarity floor (it's genuinely topically
  related) but doesn't define what a bicycle is — and the model filled
  that gap with general outside knowledge instead of saying so, exactly
  the failure mode `MIN_SIMILARITY` was never meant to catch (see its own
  docstring in `src/pipeline.py`: it only catches "nothing here is even
  topically related," not "this doesn't actually answer what was asked").
  `system_prompt` (`src/prompt.py`) now spells out this specific
  partial-mention case explicitly, with that exact example, rather than
  relying on a generic "don't guess" instruction to cover it implicitly.
- **A citation bug that only showed up on Linux, from data indexed on
  Windows** — `extract_sources()` used to reduce a source path to a
  filename with `os.path.basename()`, which only treats the *current*
  OS's own separator as meaningful. A path recorded with backslashes
  (`data\report.pdf` — leftover in Pinecone's stored metadata from
  whichever machine originally indexed it) passed through
  `os.path.basename()` on a Linux server unchanged, since POSIX doesn't
  treat `\` as a separator — producing a second, differently-labeled
  "duplicate" citation for a source already shown under its clean name.
  Fixed by splitting on both separators explicitly, regardless of which
  OS the code happens to be running on.
- **One shared namespace, one manifest, deterministic ids for what's
  seeded** — every document (seeded or uploaded) lives in the same
  Pinecone namespace and the same Postgres manifest, so "list every
  document" and "delete this one" work identically regardless of where it
  came from (see `src/documents.py`'s module docstring — this replaced an
  earlier design with a separate, unmanaged namespace for the original
  reference book, which is exactly why *that* document couldn't be listed
  or deleted from the UI at all). Pinecone serverless indexes only support
  delete-by-id (not delete-by-metadata-filter), which is why every chunk
  gets an explicit, predictable id (`{doc_id}::{chunk_index}`) at index
  time instead of letting Pinecone generate one — see `src/documents.py`.
  `data/seed/` documents specifically get a *deterministic* id derived
  from their filename (`seed_data.py`'s `_seed_doc_id()`), rather than a
  random one — otherwise a restart re-running the seeding step couldn't
  tell "already indexed" apart from "needs indexing" and would just keep
  adding duplicates.
- **A deletion has to survive the app's own next restart** — deleting one
  of the `data/seed/` documents from the sidebar removes its manifest row,
  which (on its own) is indistinguishable, next time `seed_data.py` runs,
  from "never seeded yet" — and free-tier hosts restart often enough
  (every idle-sleep wake-up, every redeploy) that this isn't a theoretical
  edge case. `document_store.mark_deleted()`/`was_deleted()` — a small
  tombstone table alongside the manifest — is what makes a deletion
  actually stick regardless of how many times the app restarts afterward.
- **Every failure path returns clean JSON/SSE, never a raw error** — a
  global Flask error handler catches anything unhandled and reshapes it
  into `{"error": ..., "message": ...}`; the streaming endpoint wraps its
  generator in its own try/except for the same reason (once an SSE
  response starts, Flask can no longer turn a mid-stream crash into a
  normal error response — it has to be handled inside the generator).
  The frontend mirrors this: a backend `"error"` event renders as a plain
  message bubble, a dropped connection preserves whatever partial answer
  had already streamed in instead of discarding it, and a stalled request
  aborts after 60s instead of hanging forever. The real exception is
  always logged server-side and never shown to the user.
- **Evaluation is separate from testing** — `tests/` checks that the code
  behaves correctly (routing, error handling, caching logic — deterministic,
  fast, run on every push). `eval/` checks that the *model's answers* are
  actually good (retrieval quality, hallucination rate — needs live APIs,
  run manually/periodically). Conflating the two is a common mistake.
- **React via CDN + Babel standalone, not a Vite/webpack build** — deliberate:
  it keeps deployment identical to before (Flask just serves `app.jsx` as a
  static file, no Node/npm anywhere in the stack, no Docker build stage to
  maintain). The honest trade-off: Babel transforms JSX in the browser on
  every page load, which the React docs explicitly say isn't meant for
  production at scale. For this project's traffic that overhead is a
  non-issue; if you were to scale this up, the fix is a proper Vite build
  producing a static bundle Flask still just serves the same way — the
  component code itself wouldn't need to change.
- **In-memory rate limiting** — good enough for a single free-tier
  instance; explicitly not good enough for multiple workers, which is a
  legitimate follow-up question to have an answer ready for (swap in
  Redis-backed storage).

---

## 🩹 Troubleshooting

**`pip install -r requirements.txt` is extremely slow, downloads several
GB, runs out of disk space, or just "never seems to finish":**
This is almost always torch. `sentence-transformers` depends on it, and
on Linux, plain `pip install torch` defaults to a CUDA-enabled build —
~800MB for torch itself, plus another ~1-2GB of NVIDIA/CUDA packages
(`nvidia-cublas`, `nvidia-cudnn`, `triton`, and a dozen more) pulled in
alongside it — even though nothing in this project touches a GPU. Fix:
install the CPU-only build *first*, so pip finds it already satisfied
when it processes the rest of `requirements.txt` and never reaches for
the CUDA one:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
This is also baked into the Dockerfile, so a `docker build` doesn't hit
it — only a bare local `pip install` does. If you've already installed
the CUDA version and want to reclaim the space: `pip uninstall torch
nvidia-cublas-cu13 nvidia-cudnn-cu13 triton` (package names vary by
version — `pip list | grep -i nvidia` shows exactly what's installed),
then run the two commands above.

**A fresh install works one day and breaks the next, or works on your
machine but not a teammate's, with no code changes:**
Most of this project's *direct* dependencies are pinned to an exact
version (`langchain==0.3.26`, etc.), but pip still resolves every
*transitive* dependency (the packages those depend on) freely within
whatever range each pin allows — so two installs done weeks apart, or on
two different machines, aren't guaranteed to produce the identical
dependency tree. If you hit this: `pip freeze > working-versions.txt`
the moment you have a setup that works, commit that alongside
`requirements.txt`, and reinstall from it (`pip install -r
working-versions.txt`) when reproducing the same environment matters more
than getting the newest compatible version of everything. This project
doesn't ship a full lock file by default to keep `requirements.txt`
readable and easy to hand-edit, which is a real trade-off, not an
oversight — worth being able to explain that trade-off if asked.

**Blank white screen when you open the app, with a console error like
`Uncaught SyntaxError: Cannot use import statement outside a module`
coming from Babel's `transformScriptTags.ts`:**
Already fixed, but worth knowing why: `templates/chat.html` used to let
Babel standalone auto-scan the page for `<script type="text/babel"
src="...">` and transform it internally. That auto-scan path goes through
an older code path that can fail silently, *and* newer Babel versions
default the JSX transform to `runtime: "automatic"`, which compiles JSX
into `import {jsx as _jsx} from "react/jsx-runtime"` — a real ES module
import, which throws exactly that error when injected as a plain script
(there's no bundler here to resolve it). Fixed by calling
`Babel.transform()` directly with `runtime: "classic"` explicitly set
(plain `React.createElement(...)` calls, no imports), and by showing any
boot failure on-screen instead of leaving a silent blank page — so if this
ever regresses, you'll see a real error message instead of nothing.

**`pip install` fails with a `ResolutionImpossible` / conflicting
dependencies error mentioning `langchain-core`:**
`langchain-groq` jumped to a new major version (1.x) that requires
`langchain-core>=1.0.0`, which conflicts with `langchain==0.3.26` (which
needs `langchain-core<1.0.0`). `requirements.txt` already pins
`langchain-groq==0.3.8` — the last release on the compatible 0.3.x line —
so if you hit this, make sure you're installing from the `requirements.txt`
in this repo and haven't manually bumped that line.

**`ImportError` from `src/helper.py` (e.g. `cannot import name
'HuggingFaceEmbeddings' from 'langchain.embeddings'`):**
Already fixed, but worth knowing why: the original tutorial code imported
`PyPDFLoader`, `DirectoryLoader`, `HuggingFaceEmbeddings`, and `Document`
straight from the top-level `langchain` package. LangChain reorganized
those into `langchain_community` (integrations), `langchain_text_splitters`,
and `langchain_core` (`Document`) starting around 0.2.x, and the old
top-level paths stopped working entirely by 0.3.x. `src/helper.py` now
imports from the correct locations. If you ever add a new LangChain
integration and get an import error, this reorg is almost always why —
check `langchain_community` and `langchain_core` first.

**`/stats` or `/feedback` are briefly slow, or `/stats` returns a clean
500 (but chat still works):**
Neon's free tier scales its compute to zero after ~5 minutes of
inactivity to stay free forever. Unlike some providers, this doesn't need
manual intervention — Neon auto-resumes the compute on the next query,
usually adding a second or so to that one request while it wakes up. If a
connection attempt lands in the exact moment it's still spinning up and
fails, `/stats` will return a clean 500 (just retry a moment later), and
chat/feedback submission keep working regardless: `log_query()` and
`log_feedback()` both catch and log their own failures instead of
breaking the request (see `src/telemetry.py`) — a telemetry hiccup should
never block the thing people actually came here for.

**Uploaded a PDF, it worked, but after a restart it's gone from "Your
documents" (chat may still occasionally reference it):**
The document manifest (`src/document_store.py`) needs a working
`DATABASE_URL` to persist — same requirement as `/stats` and feedback,
see Setting up Postgres above. Without one, `create_app()` falls back to
`InMemoryDocumentStore`, exactly like telemetry's own Postgres fallback,
which means the *list* of uploaded documents lives only in that process's
memory and resets on restart. The underlying Pinecone vectors aren't
deleted by this — they're just no longer tracked by anything, so they
can't be listed or removed through the UI anymore either. Configure
`DATABASE_URL` to avoid this entirely; it's the same free Neon setup the
rest of the app already needs.

**Upload fails with "No extractable text was found in this PDF":**
That PDF has no text layer — usually a scanned document (pages saved as
images) rather than "real" text, which is common for old paperwork run
through a scanner without OCR. `PyPDFLoader` can only extract text that's
actually stored as text in the file; OCR'ing scanned pages first (e.g.
with a tool like `ocrmypdf`) isn't something this project does
automatically.

**`/documents/upload` returns 413:**
The file is over the 15MB cap (`MAX_UPLOAD_BYTES` in `src/documents.py`,
also enforced via Flask's `MAX_CONTENT_LENGTH`). Split large PDFs or
raise the constant if you control the deployment and know the host has
the RAM to embed that many chunks at once (free-tier hosts are usually
the tight case here — see `DEPLOYMENT.md`'s troubleshooting section for
the same RAM constraint as it applies to the embedding/reranker models).

**Nothing above explains it — a route just isn't behaving:**
Every route in `app.py` is wrapped so failures come back as clean JSON
(`{"error": ..., "message": ...}`) or, for `/get/stream`, a clean
`{"type": "error", "message": ...}` SSE event — never a raw traceback or
an HTML error page. The *real* exception is always logged server-side
(`app.logger.exception(...)`), so check your terminal/host logs for the
actual stack trace; what the browser sees is deliberately sanitized.

---

## 🔭 Future scope

- Swap `localStorage` history for real accounts, persisted in the same
  Postgres database already used for telemetry/feedback — and scope
  uploaded documents per-account at the same time, instead of the current
  shared-namespace model (see "Bringing your own documents" above).
- Redis-backed semantic cache + rate limiting for multi-instance scaling.
- A background job queue for PDF ingestion instead of processing uploads
  synchronously in the request — matters more as upload volume or file
  size grows past what a single free-tier CPU embeds comfortably within
  one HTTP request.
- OCR fallback for scanned PDFs with no text layer (see Troubleshooting).
- Swap the keyword-based emergency guardrail for a small classifier,
  A/B-tested against the current rule-based version.
- A configurable date range on the Analytics dashboard (currently a fixed
  14-day window) and a per-document breakdown of how often each uploaded
  file actually gets cited in an answer.
- Extend `eval/run_eval.py`'s testset.json with questions targeting
  uploaded documents specifically (using `document_ids` to scope them) —
  right now the precision/recall/F1 numbers only reflect the
  bundled `data/seed/` documents.

---

## Author

**Pritam** — extended a Flask + LangChain + Pinecone RAG starting point
with conversational memory (including a real fix for a follow-up-question
context bug), source citations, cross-encoder reranking, a semantic
cache, streaming responses, an emergency safety layer, structured
observability, an evaluation harness with precision/recall/F1 scoring, a
pytest suite, voice I/O, a free (Groq-based) LLM swap, user-uploaded PDF
support with retrieval-grounded "no information" honesty, a charted
Analytics dashboard, and a full frontend redesign.
