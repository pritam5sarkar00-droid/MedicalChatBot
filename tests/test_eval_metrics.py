"""
Tests for eval/run_eval.py's compute_confusion_matrix() — pure, no
pipeline, no network, no credentials, exercised directly with hand-built
result rows shaped exactly like run()'s real per-question output. See
that function's docstring for the TP/FN/FP/TN definitions.
"""

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "run_eval", os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "run_eval.py")
)
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)
compute_confusion_matrix = run_eval.compute_confusion_matrix


def row(expect_no_answer, refused, error=None):
    return {"expect_no_answer": expect_no_answer, "refused": refused, "error": error}


def test_true_positive_should_answer_and_did():
    result = compute_confusion_matrix([row(expect_no_answer=False, refused=False)])
    assert result["true_positive"] == 1
    assert result["false_negative"] == result["false_positive"] == result["true_negative"] == 0


def test_false_negative_should_answer_but_refused():
    result = compute_confusion_matrix([row(expect_no_answer=False, refused=True)])
    assert result["false_negative"] == 1
    assert result["true_positive"] == result["false_positive"] == result["true_negative"] == 0


def test_false_positive_should_refuse_but_answered():
    result = compute_confusion_matrix([row(expect_no_answer=True, refused=False)])
    assert result["false_positive"] == 1
    assert result["true_positive"] == result["false_negative"] == result["true_negative"] == 0


def test_true_negative_should_refuse_and_did():
    result = compute_confusion_matrix([row(expect_no_answer=True, refused=True)])
    assert result["true_negative"] == 1
    assert result["true_positive"] == result["false_negative"] == result["false_positive"] == 0


def test_errored_rows_are_excluded_entirely():
    results = [
        row(expect_no_answer=False, refused=False),
        {"expect_no_answer": False, "refused": False, "error": "TimeoutError: ..."},
    ]
    result = compute_confusion_matrix(results)
    assert result["true_positive"] == 1  # only the non-errored row counted
    assert sum(result[k] for k in ("true_positive", "false_negative", "false_positive", "true_negative")) == 1


def test_precision_recall_f1_accuracy_on_a_realistic_mixed_set():
    results = [
        row(False, False),  # TP
        row(False, False),  # TP
        row(False, True),  # FN -- over-cautious refusal
        row(True, False),  # FP -- hallucinated an answer it shouldn't have
        row(True, True),  # TN
        row(True, True),  # TN
    ]
    result = compute_confusion_matrix(results)

    assert (result["true_positive"], result["false_negative"], result["false_positive"], result["true_negative"]) == (
        2,
        1,
        1,
        2,
    )
    # precision = TP / (TP + FP) = 2 / 3
    assert result["precision"] == round(2 / 3, 3)
    # recall = TP / (TP + FN) = 2 / 3
    assert result["recall"] == round(2 / 3, 3)
    assert result["f1"] == round(2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)), 3)
    # accuracy = (TP + TN) / total = 4 / 6
    assert result["accuracy"] == round(4 / 6, 3)


def test_perfect_score_when_every_decision_is_correct():
    results = [row(False, False), row(False, False), row(True, True)]
    result = compute_confusion_matrix(results)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["accuracy"] == 1.0


def test_worst_case_every_decision_wrong():
    # Refuses everything it should answer, answers everything it should refuse.
    results = [row(False, True), row(True, False)]
    result = compute_confusion_matrix(results)
    assert result["precision"] == 0.0  # the only "answer" given was a should-refuse case
    assert result["recall"] == 0.0  # the only "should answer" case was refused
    assert result["f1"] is None  # undefined (0/0) rather than a misleading 0.0
    assert result["accuracy"] == 0.0


def test_precision_is_none_when_the_model_never_answered_anything():
    """TP + FP == 0 -- precision (TP / (TP+FP)) is mathematically
    undefined, not zero. Reporting None is honest; reporting 0.0 would
    misleadingly suggest every answer given was wrong, when none were
    given at all."""
    results = [row(True, True), row(True, True)]  # both correctly refused, nothing answered
    result = compute_confusion_matrix(results)
    assert result["precision"] is None
    assert result["recall"] is None  # no should-answer cases existed either
    assert result["accuracy"] == 1.0  # still well-defined: both decisions were correct


def test_empty_results_returns_all_none_or_zero_without_raising():
    result = compute_confusion_matrix([])
    assert result["true_positive"] == 0
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert result["accuracy"] is None


def test_missing_expect_no_answer_defaults_to_should_answer():
    """testset.json only sets expect_no_answer on out-of-scope questions
    (see the JSON file) -- absence means False, not an error."""
    result = compute_confusion_matrix([{"refused": False, "error": None}])
    assert result["true_positive"] == 1
