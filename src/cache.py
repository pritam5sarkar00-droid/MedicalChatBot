"""
cache.py — a small in-memory semantic cache for MediCare AI.

A plain dict cache only helps if someone asks the *exact* same question
twice. This one compares the embedding of a new question against
previously-seen questions and reuses the cached answer if the cosine
similarity clears a threshold — so "what is asthma" and "can you explain
asthma to me" can hit the same entry even though the text differs.

This reuses the embedding model MediCare AI already loads for retrieval, so
there's no extra dependency beyond numpy for the similarity math.

Trade-off worth knowing for an interview: this is process-local (an
in-memory dict), so it resets on restart and doesn't share state across
multiple instances. Fine for a single free-tier deployment; at real scale
you'd back it with Redis (store vectors + a vector index) instead.
"""

import time
import uuid
from collections import OrderedDict
from typing import Optional

import numpy as np


class SemanticCache:
    def __init__(self, threshold: float = 0.93, max_size: int = 200, ttl_seconds: int = 3600):
        self.threshold = threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[str, dict]" = OrderedDict()

    @staticmethod
    def _cosine(a, b) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def get(self, vector, scope: Optional[str] = None) -> Optional[dict]:
        """Return the cached {answer, sources, question, similarity} for the
        closest previous question *within the same scope*, or None if
        nothing clears the threshold.

        scope exists because of per-document selection (see app.py's
        /get and /get/stream, and the sidebar's document checkboxes): the
        same question asked with only "Diabetes" selected and asked again
        with every document selected can have two genuinely different
        correct answers, since they were generated from different
        context. Passing the caller's current selection as scope (app.py
        normalizes it to a stable string) keeps those two cases from ever
        matching each other's cached entry. The default, None, behaves
        exactly like the old scope-less cache -- every call that doesn't
        pass a scope only ever matches other calls that also didn't.
        """
        now = time.time()
        best_key, best_score = None, 0.0

        for key, entry in self._entries.items():
            if entry.get("scope") != scope:
                continue
            if now - entry["ts"] > self.ttl_seconds:
                continue
            score = self._cosine(vector, entry["vector"])
            if score > best_score:
                best_score, best_key = score, key

        if best_key is not None and best_score >= self.threshold:
            self._entries.move_to_end(best_key)
            hit = self._entries[best_key]
            return {
                "answer": hit["answer"],
                "sources": hit["sources"],
                "question": hit["question"],
                "similarity": round(best_score, 4),
            }
        return None

    def set(self, vector, question: str, answer: str, sources: list, scope: Optional[str] = None) -> None:
        # A plain time.time()-based key can collide under concurrent
        # requests (two requests computing the same timestamp+count before
        # either has inserted), silently overwriting one cached entry with
        # another. A UUID makes that impossible regardless of timing or
        # threading model.
        key = uuid.uuid4().hex
        self._entries[key] = {
            "vector": np.asarray(vector, dtype=float),
            "question": question,
            "answer": answer,
            "sources": sources,
            "scope": scope,
            "ts": time.time(),
        }
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)  # evict oldest (simple FIFO/LRU-ish)

    def stats(self) -> dict:
        return {"size": len(self._entries), "threshold": self.threshold, "max_size": self.max_size}

    def clear(self) -> None:
        """Wipes every cached answer.

        Called from app.py whenever the knowledge base itself changes --
        a document gets uploaded or deleted (see the /documents routes).
        A cached answer is only correct for as long as the retrieval
        results it was built from are still accurate; the moment a new
        PDF might answer a previously-"I don't know" question, or a
        deleted PDF's content should no longer back a previously-cached
        answer, every existing entry is a potential stale/wrong answer
        rather than a valid shortcut. Clearing the whole cache is a
        blunter tool than invalidating only the affected entries, but
        it's the only option that's actually guaranteed correct: there's
        no cheap way to know in advance which cached answers a given
        document did or didn't influence.
        """
        self._entries.clear()
