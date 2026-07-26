"""
eval/run_eval.py — a small evaluation harness for MediCare AI's RAG
pipeline.

Most tutorial clones of this project never measure anything — they just
demo a few questions and call it done. This script runs a fixed test set
against the *live* pipeline and reports real numbers:

  retrieval_hit_rate     — for questions with a known answer in the book,
                            did the retriever actually pull a chunk
                            containing an expected keyword? (tests the
                            retrieval half of RAG in isolation)
  avg_answer_coverage     — of the expected keywords, what fraction showed
                            up in the final generated answer?
  precision / recall / f1 / accuracy
                          — treats "should this question be answered or
                            refused?" (expect_no_answer in testset.json)
                            as a binary classification problem and scores
                            the model's actual answer-or-refuse decision
                            against it. See compute_confusion_matrix()
                            below for the full breakdown (this is a
                            distinct axis from retrieval_hit_rate/
                            avg_answer_coverage above: those grade *what*
                            the model said, this grades *whether* it
                            decided to say anything at all).
  avg_latency_ms          — end-to-end response time per question.

Run this once your .env has real PINECONE_API_KEY / GROQ_API_KEY values
and `python seed_data.py` has populated the index:

    python eval/run_eval.py

It writes eval/results.json (machine-readable) and eval/results.md
(a table you can paste straight into your README/resume/report) —
so any numbers you cite are numbers you actually measured, not vibes.

Scope note: testset.json only exercises the base reference book (the
default Pinecone namespace) — it says nothing about retrieval quality
against user-uploaded documents (the "uploads" namespace, see
src/pipeline.py's CombinedMedicalRetriever and src/documents.py). To
extend this script for that: ingest a known fixture PDF via
src.documents.ingest_pdf(), upsert it into the uploads namespace with
pipeline.vectorstore.add_documents(...), add testset questions whose
expected_keywords only appear in that fixture, and clean up with
pipeline.vectorstore.delete(..., namespace="uploads") afterward so
repeated eval runs don't accumulate leftover vectors.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()  # PINECONE_API_KEY / GROQ_API_KEY -- must run before build_pipeline() below reads them

from src.pipeline import build_pipeline  # noqa: E402

TESTSET_PATH = os.path.join(os.path.dirname(__file__), "testset.json")
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "results.json")
RESULTS_MD = os.path.join(os.path.dirname(__file__), "results.md")

NO_ANSWER_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not in my reference material",
    "i don't know",
    "cannot find",
    "no information",
    "not covered",
    "doesn't appear to be",
    "don't have information",
]


def looks_like_no_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(p in lowered for p in NO_ANSWER_PHRASES)


def compute_confusion_matrix(results: list) -> dict:
    """
    Treats "should this question be answered, or refused?" as a binary
    classification problem, with expect_no_answer (testset.json) as the
    ground-truth label, and scores the model's actual decision against
    it. Pulled out as its own pure function (no pipeline, no network)
    specifically so this logic has real unit test coverage — see
    tests/test_eval_metrics.py — independent of needing live Pinecone/Groq
    credentials to exercise it.

    This is a different axis from retrieval_hit_rate/avg_answer_coverage:
    those grade *what* the model said (did the right content show up),
    this grades *whether* it decided to say anything at all. Both
    failure directions matter and are visible separately here:

        TP: should answer, and did answer
        FN: should answer, but refused          (over-cautious --
                                                   a real question going
                                                   unnecessarily unanswered)
        FP: should refuse, but answered anyway  (hallucination -- exactly
                                                   what MIN_SIMILARITY in
                                                   src/pipeline.py and the
                                                   NO_CONTEXT_MESSAGE
                                                   override in app.py
                                                   exist to prevent)
        TN: should refuse, and did refuse        (correct refusal)

    Precision here reads as "of the times the model gave an answer, how
    often was it actually supposed to" (low precision = hallucinating on
    things it shouldn't touch); recall reads as "of the times it should
    have answered, how often did it actually try" (low recall = too
    trigger-happy about refusing legitimate questions). Rows with an
    unhandled error (see the try/except in run()) are excluded entirely,
    same as retrieval_hit/correct_refusal already are.
    """
    tp = fn = fp = tn = 0
    for r in results:
        if r.get("error"):
            continue
        should_answer = not r.get("expect_no_answer", False)
        did_answer = not r.get("refused", False)
        if should_answer and did_answer:
            tp += 1
        elif should_answer and not did_answer:
            fn += 1
        elif not should_answer and did_answer:
            fp += 1
        else:
            tn += 1

    total = tp + fn + fp + tn
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall)) else None
    accuracy = ((tp + tn) / total) if total else None

    return {
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
    }


def run():
    with open(TESTSET_PATH) as f:
        testset = json.load(f)

    print(f"Loaded {len(testset)} eval questions. Building pipeline (this loads the embedding model)...")
    pipeline = build_pipeline()

    results = []
    for item in testset:
        question = item["question"]
        expected_keywords = [k.lower() for k in item.get("expected_keywords", [])]
        expect_no_answer = item.get("expect_no_answer", False)

        t0 = time.time()
        try:
            response = pipeline.chain.invoke({"input": question, "chat_history": []})
            answer = response.get("answer") or ""
            context_docs = response.get("context", [])
        except Exception as exc:
            # A single flaky API call (rate limit, transient network blip)
            # shouldn't discard every other question's results — record
            # the failure and keep going.
            latency_ms = (time.time() - t0) * 1000
            print(f"  [ERROR] {item['id']}: {question}  -- {type(exc).__name__}: {exc}")
            results.append(
                {
                    "id": item["id"],
                    "category": item.get("category"),
                    "question": question,
                    "latency_ms": round(latency_ms, 1),
                    "expect_no_answer": expect_no_answer,
                    "retrieval_hit": None,
                    "answer_coverage": None,
                    "correct_refusal": None,
                    "refused": None,
                    "num_sources": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        latency_ms = (time.time() - t0) * 1000
        context_text = " ".join((d.page_content or "").lower() for d in context_docs)

        retrieval_hit = any(k in context_text for k in expected_keywords) if expected_keywords else None
        answer_lower = answer.lower()
        matched = [k for k in expected_keywords if k in answer_lower]
        answer_coverage = (len(matched) / len(expected_keywords)) if expected_keywords else None

        # Computed for *every* question, not only the out-of-scope ones --
        # compute_confusion_matrix() needs to know whether an in-scope
        # question got refused too (a legitimate question the model
        # unnecessarily declined to answer is exactly the false-negative
        # case it's meant to catch).
        refused = looks_like_no_answer(answer)
        correct_refusal = refused if expect_no_answer else None

        results.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "question": question,
                "latency_ms": round(latency_ms, 1),
                "expect_no_answer": expect_no_answer,
                "retrieval_hit": retrieval_hit,
                "answer_coverage": round(answer_coverage, 2) if answer_coverage is not None else None,
                "correct_refusal": correct_refusal,
                "refused": refused,
                "num_sources": len(context_docs),
            }
        )
        passed = retrieval_hit or correct_refusal
        tag = "OK" if passed else "CHECK"
        print(f"  [{tag}] {item['id']}: {question}  ({latency_ms:.0f}ms)")

    scored = [r for r in results if r["retrieval_hit"] is not None]
    refusal_scored = [r for r in results if r["correct_refusal"] is not None]
    confusion = compute_confusion_matrix(results)

    summary = {
        "num_questions": len(results),
        "retrieval_hit_rate": round(sum(1 for r in scored if r["retrieval_hit"]) / len(scored), 3) if scored else None,
        "avg_answer_coverage": (
            round(sum(r["answer_coverage"] for r in scored if r["answer_coverage"] is not None) / len(scored), 3)
            if scored
            else None
        ),
        "correct_refusal_rate": (
            round(sum(1 for r in refusal_scored if r["correct_refusal"]) / len(refusal_scored), 3)
            if refusal_scored
            else None
        ),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1) if results else None,
        **confusion,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    with open(RESULTS_MD, "w") as f:
        f.write("# MediCare AI — Evaluation Report\n\n")
        f.write(f"Ran {summary['num_questions']} questions against the live pipeline.\n\n")
        f.write("## Summary\n\n")
        f.write("| Metric | Score |\n|---|---|\n")
        f.write(f"| Retrieval hit-rate | {summary['retrieval_hit_rate']} |\n")
        f.write(f"| Avg. answer keyword coverage | {summary['avg_answer_coverage']} |\n")
        f.write(f"| Correct refusal rate (out-of-scope Qs) | {summary['correct_refusal_rate']} |\n")
        f.write(f"| Avg. latency | {summary['avg_latency_ms']} ms |\n\n")
        f.write(
            "### Answer-or-refuse decision (confusion matrix)\n\n"
            "Scores whether the model correctly decided *to answer at all*, "
            "treating each question's `expect_no_answer` label as ground truth "
            "— see `compute_confusion_matrix()` in this script for the exact "
            "TP/FN/FP/TN definitions.\n\n"
        )
        f.write("| Metric | Score |\n|---|---|\n")
        f.write(f"| Precision | {summary['precision']} |\n")
        f.write(f"| Recall | {summary['recall']} |\n")
        f.write(f"| F1 | {summary['f1']} |\n")
        f.write(f"| Accuracy | {summary['accuracy']} |\n")
        f.write(
            f"| Confusion matrix | TP={summary['true_positive']}, FN={summary['false_negative']}, "
            f"FP={summary['false_positive']}, TN={summary['true_negative']} |\n\n"
        )
        f.write("## Per-question results\n\n")
        f.write("| ID | Category | Retrieval hit | Answer coverage | Refused? | Latency (ms) |\n|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['id']} | {r['category']} | {r['retrieval_hit']} | {r['answer_coverage']} | "
                f"{r['refused']} | {r['latency_ms']} |\n"
            )

    print("\nSummary:", json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_JSON} and {RESULTS_MD}")


if __name__ == "__main__":
    run()
