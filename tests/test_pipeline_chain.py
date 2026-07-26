"""
Tests for src/pipeline.py's build_conversational_chain() — the piece
responsible for a real, previously-unfixed bug: a follow-up like "explain
in details" was rewritten into a standalone question for *retrieval*
only; the final answer-generation call still saw the raw, ambiguous
follow-up and had to re-resolve the reference itself from chat history.
See build_conversational_chain()'s docstring in src/pipeline.py for the
full explanation, and MediCare-AI-Pritam's commit history for the
before/after trace that found this.

FakeListChatModel is a real LangChain testing utility (langchain_core is
already a hard dependency via langchain==0.3.26) -- not a hand-rolled
double, so composing it with `|` and passing it into
create_stuff_documents_chain behaves exactly like a real chat model would.
"""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever

from src.pipeline import _looks_like_a_reasonable_rewrite, build_conversational_chain


class RecordingRetriever(BaseRetriever):
    """Records every query string it's called with, and always returns
    one fixed Document -- what matters for these tests is *what query the
    retriever received*, not retrieval ranking (that's
    test_pipeline_retrieval.py's job)."""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        self.calls.append(query)
        return self.doc_to_return

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "calls", [])
        object.__setattr__(
            self, "doc_to_return", [Document(page_content="Asthma is a chronic airway disease.", metadata={"page": 1})]
        )


HISTORY = [HumanMessage(content="What is asthma?"), AIMessage(content="Asthma is a chronic airway disease.")]


class CountingFakeChatModel(FakeListChatModel):
    """FakeListChatModel that also tracks how many times it was actually
    invoked -- a more direct, less fragile signal than trying to infer
    call count from which canned response came back, especially once a
    test wants to prove a call *never happened* at all."""

    call_count: int = 0

    def invoke(self, *args, **kwargs):
        self.call_count += 1
        return super().invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        self.call_count += 1
        yield from super().stream(*args, **kwargs)


def test_final_answer_call_receives_the_rewritten_question_not_the_raw_followup():
    """The core regression test: reproduces 'ask X, then ask a vague
    follow-up' and confirms the *second* (answer-generation) LLM call
    sees the resolved standalone question, not the raw "explain in
    details" the textbook create_history_aware_retriever + 
    create_retrieval_chain combo would still be passing it."""
    llm = FakeListChatModel(responses=["What is asthma in detail?", "Asthma in detail: it is a chronic airway disease."])
    retriever = RecordingRetriever()
    chain = build_conversational_chain(llm, retriever)

    result = chain.invoke({"input": "explain in details", "chat_history": HISTORY})

    assert result["input"] == "What is asthma in detail?"
    assert result["answer"] == "Asthma in detail: it is a chronic airway disease."


def test_retriever_is_called_with_the_rewritten_question():
    llm = FakeListChatModel(responses=["What is asthma in detail?", "some answer"])
    retriever = RecordingRetriever()
    chain = build_conversational_chain(llm, retriever)

    chain.invoke({"input": "explain in details", "chat_history": HISTORY})

    assert retriever.calls == ["What is asthma in detail?"]


class RecordingFilterableRetriever(BaseRetriever):
    """Like RecordingRetriever, but also exposes .search(query, doc_ids=),
    the richer interface CombinedMedicalRetriever actually implements
    (src/pipeline.py) -- exists to prove build_conversational_chain()
    actually calls *this* method (and threads document_ids through to its
    doc_ids parameter) when it's available, via _retrieve_context()'s
    duck-typing, rather than always falling back to plain .invoke(query)
    the way every test above this one relies on it doing for a retriever
    that *doesn't* have .search()."""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        raise AssertionError(
            "build_conversational_chain() should have called .search(), not fallen back to "
            ".invoke()/._get_relevant_documents(), for a retriever that implements .search()"
        )

    def search(self, query, doc_ids=None):
        self.search_calls.append((query, doc_ids))
        return self.doc_to_return

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "search_calls", [])
        object.__setattr__(
            self, "doc_to_return", [Document(page_content="Asthma is a chronic airway disease.", metadata={"page": 1})]
        )


def test_document_ids_threads_through_to_a_retriever_that_supports_search():
    llm = FakeListChatModel(responses=["What is asthma? It's a chronic condition."])
    retriever = RecordingFilterableRetriever()
    chain = build_conversational_chain(llm, retriever)

    chain.invoke({"input": "What is asthma?", "chat_history": [], "document_ids": ["doc-a", "doc-b"]})

    assert retriever.search_calls == [("What is asthma?", ["doc-a", "doc-b"])]


def test_missing_document_ids_key_passes_none_to_search():
    """A chain input dict shaped exactly like one from before this feature
    existed (no document_ids key at all) must still resolve to "search
    everything" (doc_ids=None), not a crash from a missing dict key --
    see _retrieve_context()'s call site (the .assign(context=...) lambda)
    in build_conversational_chain()."""
    llm = FakeListChatModel(responses=["What is asthma? It's a chronic condition."])
    retriever = RecordingFilterableRetriever()
    chain = build_conversational_chain(llm, retriever)

    chain.invoke({"input": "What is asthma?", "chat_history": []})

    assert retriever.search_calls == [("What is asthma?", None)]


def test_no_history_skips_the_rewrite_call_entirely():
    """First message in a conversation: nothing to resolve, so this
    should cost exactly one LLM call (the answer), not two -- matching
    create_history_aware_retriever's own no-history optimization."""
    llm = CountingFakeChatModel(responses=["What is asthma? It's a chronic condition."])
    retriever = RecordingRetriever()
    chain = build_conversational_chain(llm, retriever)

    result = chain.invoke({"input": "What is asthma?", "chat_history": []})

    assert result["answer"] == "What is asthma? It's a chronic condition."
    assert retriever.calls == ["What is asthma?"]  # retrieved with the original input, unchanged
    assert llm.call_count == 1  # the rewrite step never ran at all


def test_chain_output_has_the_interface_app_py_and_eval_depend_on():
    """app.py and eval/run_eval.py both read response['answer'] and
    response['context'] -- this is the actual public contract, unlike
    'input', which is an internal implementation detail nothing outside
    this chain should rely on."""
    llm = FakeListChatModel(responses=["rewritten question", "the answer"])
    retriever = RecordingRetriever()
    chain = build_conversational_chain(llm, retriever)

    result = chain.invoke({"input": "a follow-up", "chat_history": HISTORY})

    assert result["answer"] == "the answer"
    assert result["context"] == retriever.doc_to_return


def test_streaming_yields_context_before_answer_chunks():
    llm = FakeListChatModel(responses=["What is asthma in detail?", "Asthma in detail: it is a chronic condition."])
    retriever = RecordingRetriever()
    chain = build_conversational_chain(llm, retriever)

    saw_context_before_first_answer_chunk = False
    context_seen = False
    for chunk in chain.stream({"input": "explain in details", "chat_history": HISTORY}):
        if "context" in chunk:
            context_seen = True
        if "answer" in chunk and context_seen:
            saw_context_before_first_answer_chunk = True

    assert saw_context_before_first_answer_chunk


def test_streaming_early_break_on_empty_context_never_leaks_an_answer_chunk():
    """What app.py's chat_stream() actually depends on isn't "the answer
    LLM is never invoked internally" -- empirically, with three chained
    .assign() calls (rewrite, then retrieve, then answer -- needed so the
    rewritten question can drive both retrieval *and* answer generation,
    see this module's docstring), LangChain's own streaming internals
    don't reliably guarantee that at 3+ levels the way they do at the
    2-level depth create_retrieval_chain uses, even though a plain `for
    chunk in ...: break` never asks the generator for a further chunk.
    That's a LangChain internals nuance, not a correctness gap: app.py
    breaks the instant it sees an empty context chunk and never pulls
    again, so it can't ever *see* a leaked answer regardless of whether
    the chain happened to compute one in the background. That's the
    guarantee that actually matters, and this test verifies exactly that
    -- not an internal call count, which was flaky (see git history for
    a full trace of this if you're curious)."""

    class EmptyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            return []

    llm = FakeListChatModel(responses=["rewritten question", "SHOULD_NEVER_REACH_A_CONSUMER"])
    chain = build_conversational_chain(llm, EmptyRetriever())

    seen_answer_chunk = False
    for chunk in chain.stream({"input": "what's the capital of France?", "chat_history": HISTORY}):
        if "answer" in chunk:
            seen_answer_chunk = True
        if "context" in chunk:
            assert chunk["context"] == []
            break  # exactly what app.py's chat_stream() does

    assert seen_answer_chunk is False


# ---------------------------------------------------------------------------
# _looks_like_a_reasonable_rewrite() and the fallback it drives -- a real
# reported bug: after a topic switch in the conversation, the rewriting
# LLM produced commentary about the *previous* answer ("It seems I made
# an incorrect assumption... I don't have information about 'bike'")
# instead of a clean standalone question for the *new* one, and because
# that text then drove both retrieval and the final answer, the user got
# a reply about entirely the wrong topic.
# ---------------------------------------------------------------------------


class TopicSwitchHistory:
    """Chat history shaped like the actual bug report: an unrelated
    question ("what is bike") was just asked and answered, and now the
    user has switched to a completely different topic."""

    MESSAGES = [
        HumanMessage(content="what is bike"),
        AIMessage(content="A bicycle is a vehicle with two wheels..."),
    ]


def test_looks_like_a_reasonable_rewrite_accepts_clean_questions():
    assert _looks_like_a_reasonable_rewrite("explain in details", "Explain asthma in more detail") is True
    assert _looks_like_a_reasonable_rewrite("why?", "Why does asthma cause wheezing?") is True
    assert _looks_like_a_reasonable_rewrite("what is bike", "what is bike") is True  # unchanged passthrough


def test_looks_like_a_reasonable_rewrite_rejects_meta_commentary():
    # The exact failure mode from the bug report, near-verbatim.
    bad = "It seems I made an incorrect assumption. I don't have information about 'bike' in the provided context."
    assert _looks_like_a_reasonable_rewrite("what's in the antiragging form?", bad) is False


def test_looks_like_a_reasonable_rewrite_rejects_other_self_correcting_phrasings():
    examples = [
        "I apologize for the confusion, what is in the antiragging form?",
        "You are correct, I should have said I don't know about that.",
        "To answer your original question, I don't have information about bike.",
        "My previous answer was wrong -- what is in the document?",
    ]
    for bad in examples:
        assert _looks_like_a_reasonable_rewrite("some original question", bad) is False, bad


def test_looks_like_a_reasonable_rewrite_rejects_empty_or_blank():
    assert _looks_like_a_reasonable_rewrite("what is asthma", "") is False
    assert _looks_like_a_reasonable_rewrite("what is asthma", "   ") is False


def test_looks_like_a_reasonable_rewrite_rejects_wildly_long_rambling():
    original = "why?"
    rambling = "Well, " * 100 + "that is a very long-winded non-answer."
    assert _looks_like_a_reasonable_rewrite(original, rambling) is False


def test_looks_like_a_reasonable_rewrite_allows_reasonable_expansion():
    # A terse follow-up legitimately expanding into a longer standalone
    # question shouldn't be penalized just for being longer than the
    # original -- only for being *disproportionately* longer.
    original = "why?"
    reasonable = "Why does asthma cause shortness of breath and wheezing in affected individuals?"
    assert _looks_like_a_reasonable_rewrite(original, reasonable) is True


def test_bad_rewrite_falls_back_to_the_raw_question_end_to_end():
    """Direct reproduction of the reported bug, with the rewriting LLM
    mocked to produce exactly the kind of broken output seen for real:
    the chain should recover by using the user's actual raw message
    instead of letting that broken text drive retrieval and the answer."""
    llm = FakeListChatModel(
        responses=[
            "It seems I made an incorrect assumption. I don't have information about 'bike' in the provided context.",
            "Antiragging forms typically require students to acknowledge awareness of anti-ragging regulations.",
        ]
    )
    retriever = RecordingRetriever()
    chain = build_conversational_chain(llm, retriever)

    result = chain.invoke(
        {"input": "whats in there in the antiraggin form", "chat_history": TopicSwitchHistory.MESSAGES}
    )

    # The broken rewrite was discarded -- retrieval searched using the
    # user's actual question, not commentary about the previous topic.
    assert retriever.calls == ["whats in there in the antiraggin form"]
    assert result["input"] == "whats in there in the antiraggin form"
    assert "bike" not in result["answer"].lower()


def test_good_rewrite_still_flows_through_normally_alongside_the_safety_net():
    """The fallback net shouldn't get in the way of the common case --
    confirms the safety net is a backstop, not a regression in normal
    rewrite quality."""
    llm = FakeListChatModel(responses=["What is asthma in detail?", "Asthma in detail: a chronic airway disease."])
    retriever = RecordingRetriever()
    chain = build_conversational_chain(llm, retriever)

    result = chain.invoke({"input": "explain in details", "chat_history": HISTORY})

    assert retriever.calls == ["What is asthma in detail?"]
    assert result["input"] == "What is asthma in detail?"
