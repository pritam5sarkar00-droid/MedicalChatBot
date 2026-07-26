import pytest

from app import create_app
from src.cache import SemanticCache
from src.telemetry import InMemoryTelemetry


class FakeEmbeddings:
    """Deterministic fake embeddings: same text -> same vector, so the
    semantic cache behaves predictably in tests without loading a real
    sentence-transformers model."""

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


class FakeDoc:
    def __init__(self, content, source="reference.pdf", page=41):
        self.page_content = content
        self.metadata = {"source": source, "page": page}


class FakeChain:
    """Stands in for the real LangChain RAG chain."""

    def invoke(self, payload):
        return {
            "answer": "This is a fake grounded answer about the topic.",
            "context": [FakeDoc("some relevant passage")],
        }

    def stream(self, payload):
        yield {"context": [FakeDoc("some relevant passage")]}
        for word in ["This ", "is ", "a ", "fake ", "streamed ", "answer."]:
            yield {"answer": word}


class FakePipeline:
    def __init__(self):
        self.embeddings = FakeEmbeddings()
        self.chain = FakeChain()


class CrashingChain:
    """Simulates a Pinecone/Groq outage or any unexpected pipeline failure."""

    def invoke(self, payload):
        raise RuntimeError("simulated upstream failure")

    def stream(self, payload):
        raise RuntimeError("simulated upstream failure")
        yield  # pragma: no cover — unreachable, but keeps this a generator function


class EmptyAnswerChain:
    """Simulates the model returning zero text — should be treated as a
    failure, not a silent success with a blank bubble."""

    def invoke(self, payload):
        return {"answer": "", "context": [FakeDoc("x")]}

    def stream(self, payload):
        yield {"context": [FakeDoc("x")]}
        # deliberately no "answer" chunks at all


def make_client(chain=None):
    """Builds a Flask test client with every external dependency faked:
    no real Pinecone/Groq (FakePipeline), no real MySQL (InMemoryTelemetry).
    This is the same dependency-injection pattern throughout the test
    suite — see create_app(pipeline=..., cache=..., telemetry=...)."""
    pipeline = FakePipeline()
    if chain is not None:
        pipeline.chain = chain
    app = create_app(pipeline=pipeline, cache=SemanticCache(), telemetry=InMemoryTelemetry())
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def client():
    return make_client()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_empty_message_is_handled(client):
    res = client.post("/get", json={"message": "", "history": []})
    assert res.status_code == 200
    assert res.get_json()["answer"] == "Please type a question."


def test_emergency_short_circuits_the_llm(client):
    res = client.post("/get", json={"message": "I have severe chest pain", "history": []})
    data = res.get_json()
    assert data["emergency"] is True
    assert "112" in data["answer"] and "108" in data["answer"]


def test_normal_question_uses_the_pipeline(client):
    res = client.post("/get", json={"message": "What is asthma?", "history": []})
    data = res.get_json()
    assert data["emergency"] is False
    assert "fake grounded answer" in data["answer"]
    # FakeDoc uses page=41 (0-indexed, like PyPDFLoader) -> shown as page 42
    assert data["sources"][0]["page"] == 42


def test_repeated_question_hits_the_cache(client):
    first = client.post("/get", json={"message": "What is asthma?", "history": []}).get_json()
    second = client.post("/get", json={"message": "What is asthma?", "history": []}).get_json()
    assert first.get("cached") is False
    assert second.get("cached") is True
    assert second["answer"] == first["answer"]


def test_feedback_rejects_invalid_rating(client):
    res = client.post("/feedback", json={"question": "q", "answer": "a", "rating": "sideways"})
    assert res.status_code == 400


def test_feedback_accepts_valid_rating(client):
    res = client.post("/feedback", json={"question": "q", "answer": "a", "rating": "up"})
    assert res.status_code == 200
    assert res.get_json()["ok"] is True


def test_stats_empty_before_any_queries(client):
    res = client.get("/stats")
    assert res.status_code == 200
    assert res.get_json()["total_queries"] == 0


def test_stats_reflect_logged_queries(client):
    client.post("/get", json={"message": "What is diabetes?", "history": []})
    res = client.get("/stats")
    data = res.get_json()
    assert data["total_queries"] >= 1


# ---------------------------------------------------------------------------
# Failure paths — the core requirement here is that NOTHING ever leaks a raw
# traceback, a stack trace, or an HTML error page to the client. Every
# failure must come back as clean, friendly JSON (or a clean SSE "error"
# event), with the real exception only visible in server-side logs.
# ---------------------------------------------------------------------------


def test_get_handles_a_crashing_chain_gracefully():
    client = make_client(chain=CrashingChain())
    res = client.post("/get", json={"message": "What is asthma?", "history": []})
    data = res.get_json()
    assert res.status_code == 200
    assert data["error"] is True
    assert "Something went wrong" in data["answer"]
    assert "RuntimeError" not in data["answer"]
    assert "Traceback" not in data["answer"]


def test_stream_handles_a_crashing_chain_gracefully():
    client = make_client(chain=CrashingChain())
    res = client.post("/get/stream", json={"message": "What is asthma?", "history": []})
    raw = res.get_data(as_text=True)
    assert '"type": "error"' in raw
    assert '"type": "done"' in raw
    assert "Something went wrong" in raw
    assert "RuntimeError" not in raw
    assert "Traceback" not in raw


def test_stream_treats_an_empty_answer_as_a_failure():
    client = make_client(chain=EmptyAnswerChain())
    res = client.post("/get/stream", json={"message": "What is asthma?", "history": []})
    raw = res.get_data(as_text=True)
    assert '"type": "error"' in raw


def test_get_treats_an_empty_answer_as_a_failure():
    client = make_client(chain=EmptyAnswerChain())
    res = client.post("/get", json={"message": "What is asthma?", "history": []})
    data = res.get_json()
    assert data["error"] is True


def test_unknown_route_returns_json_not_html(client):
    res = client.get("/this-route-does-not-exist")
    assert res.status_code == 404
    assert "application/json" in res.content_type
    assert "<html" not in res.get_data(as_text=True).lower()


def test_malformed_json_body_does_not_crash(client):
    res = client.post("/get", data="not valid json", content_type="application/json")
    # get_json(silent=True) swallows the parse error -> treated as an empty
    # message, not a 500.
    assert res.status_code == 200
    assert res.get_json()["answer"] == "Please type a question."


class TelemetryThatFailsToInit(InMemoryTelemetry):
    """Simulates a MySQL instance that's asleep/unreachable at startup
    (e.g. Aiven's free tier auto-powering-off when idle)."""

    def init_db(self):
        raise ConnectionError("simulated: MySQL is unreachable")


def test_app_starts_and_chat_still_works_even_if_telemetry_init_fails():
    pipeline = FakePipeline()
    app = create_app(pipeline=pipeline, cache=SemanticCache(), telemetry=TelemetryThatFailsToInit())
    client = app.test_client()

    res = client.get("/health")
    assert res.status_code == 200

    res = client.post("/get", json={"message": "What is asthma?", "history": []})
    assert res.status_code == 200
    assert "fake grounded answer" in res.get_json()["answer"]
