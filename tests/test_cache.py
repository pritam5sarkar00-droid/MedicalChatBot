from src.cache import SemanticCache


def test_cache_miss_when_empty():
    cache = SemanticCache()
    assert cache.get([1.0, 0.0, 0.0]) is None


def test_cache_hit_on_identical_vector():
    cache = SemanticCache(threshold=0.9)
    vec = [1.0, 0.0, 0.0]
    cache.set(vec, "what is asthma", "Asthma is a chronic airway condition.", [{"source": "book.pdf", "page": 23}])

    hit = cache.get(vec)

    assert hit is not None
    assert hit["answer"] == "Asthma is a chronic airway condition."
    assert hit["sources"][0]["page"] == 23


# ---------------------------------------------------------------------------
# scope — keeps an answer computed under one document selection (see
# app.py's _cache_scope()) from ever being served back under a different
# one, even for the exact same question text / embedding.
# ---------------------------------------------------------------------------


def test_same_vector_different_scope_is_a_miss():
    """"What does it say about dosage?" means something completely
    different depending on which document was selected when it was asked
    -- a cache hit across that boundary would silently serve the wrong
    document's answer."""
    cache = SemanticCache(threshold=0.9)
    vec = [1.0, 0.0, 0.0]
    cache.set(vec, "what does it say", "Answer about doc A", [], scope="doc-a")

    assert cache.get(vec, scope="doc-b") is None
    assert cache.get(vec, scope=None) is None


def test_same_vector_same_scope_is_a_hit():
    cache = SemanticCache(threshold=0.9)
    vec = [1.0, 0.0, 0.0]
    cache.set(vec, "what does it say", "Answer about doc A", [], scope="doc-a")

    hit = cache.get(vec, scope="doc-a")

    assert hit is not None
    assert hit["answer"] == "Answer about doc A"


def test_default_scope_is_none_and_isolated_from_explicit_scopes():
    """Every call site that predates document selection (or a request
    that legitimately searches everything -- see _cache_scope()'s
    docstring in app.py) uses scope=None; this must keep behaving exactly
    as it did before scope existed at all, including staying separate
    from any specific document's scope."""
    cache = SemanticCache(threshold=0.9)
    vec = [1.0, 0.0, 0.0]
    cache.set(vec, "q", "answer for everything", [])  # scope defaults to None

    assert cache.get(vec) is not None
    assert cache.get(vec, scope=None) is not None
    assert cache.get(vec, scope="doc-a") is None


def test_near_duplicate_vector_still_respects_scope():
    """Scope isolation and the similarity threshold are independent
    checks -- a near-duplicate vector shouldn't slip through under the
    wrong scope just because it's similar enough in the right one."""
    cache = SemanticCache(threshold=0.9)
    cache.set([1.0, 0.1, 0.0], "q", "answer for doc-a", [], scope="doc-a")

    assert cache.get([0.98, 0.12, 0.01], scope="doc-b") is None
    assert cache.get([0.98, 0.12, 0.01], scope="doc-a") is not None


def test_stats_size_counts_entries_across_all_scopes():
    cache = SemanticCache()
    cache.set([1.0, 0.0, 0.0], "q1", "a1", [], scope="doc-a")
    cache.set([0.0, 1.0, 0.0], "q2", "a2", [], scope="doc-b")
    cache.set([0.0, 0.0, 1.0], "q3", "a3", [])

    assert cache.stats()["size"] == 3


def test_cache_hit_on_near_duplicate_vector():
    cache = SemanticCache(threshold=0.9)
    cache.set([1.0, 0.1, 0.0], "what is asthma", "Asthma is...", [])

    # A slightly different but highly similar vector should still hit
    hit = cache.get([0.98, 0.12, 0.01])
    assert hit is not None


def test_cache_miss_on_dissimilar_vector():
    cache = SemanticCache(threshold=0.9)
    cache.set([1.0, 0.0, 0.0], "q1", "a1", [])
    assert cache.get([0.0, 1.0, 0.0]) is None


def test_cache_respects_max_size():
    cache = SemanticCache(max_size=3)
    for i in range(5):
        cache.set([float(i), 1.0, 0.0], f"q{i}", f"a{i}", [])
    assert len(cache._entries) == 3


def test_cache_stats_shape():
    cache = SemanticCache(threshold=0.9, max_size=50)
    stats = cache.stats()
    assert stats == {"size": 0, "threshold": 0.9, "max_size": 50}


def test_cache_keys_are_uuids_not_derived_from_racy_state():
    """Regression test: entries used to be keyed by
    f'{time.time()}_{len(self._entries)}'. Two concurrent requests can
    both read the same pre-insert length before either has written to the
    dict, so under real thread interleaving that scheme could produce the
    same key twice and silently overwrite one cached answer with another.
    A hard reproduction of that race is inherently flaky to assert on, so
    this test instead pins down the actual fix: keys must be UUIDs (36
    hex-and-dash characters), which are unique independent of timing or
    how many entries already exist -- not derived from any state a second
    concurrent call could also be reading at the same instant."""
    cache = SemanticCache()
    cache.set([1.0, 0.0, 0.0], "q1", "answer one", [])
    cache.set([0.0, 1.0, 0.0], "q2", "answer two", [])

    keys = list(cache._entries.keys())
    assert len(keys) == 2
    for key in keys:
        assert len(key) == 32  # uuid4().hex
        int(key, 16)  # raises ValueError if it isn't valid hex -- i.e. not a real UUID


def test_cache_set_never_collides_across_many_rapid_inserts():
    cache = SemanticCache(max_size=10_000)
    for i in range(2000):
        cache.set([float(i), 0.0, 0.0], f"q{i}", f"answer{i}", [])
    assert len(cache._entries) == 2000  # every key was unique -- nothing silently overwritten


def test_clear_empties_the_cache():
    cache = SemanticCache()
    cache.set([1.0, 0.0, 0.0], "q1", "a1", [])
    cache.set([0.0, 1.0, 0.0], "q2", "a2", [])
    assert cache.stats()["size"] == 2

    cache.clear()

    assert cache.stats()["size"] == 0
    assert cache.get([1.0, 0.0, 0.0]) is None
    assert cache.get([0.0, 1.0, 0.0]) is None


def test_clear_on_an_already_empty_cache_is_safe():
    cache = SemanticCache()
    cache.clear()  # should not raise
    assert cache.stats()["size"] == 0


def test_cache_works_normally_again_after_clear():
    """clear() should reset state, not leave the cache permanently broken."""
    cache = SemanticCache(threshold=0.9)
    cache.set([1.0, 0.0, 0.0], "q1", "a1", [])
    cache.clear()

    cache.set([1.0, 0.0, 0.0], "q1 again", "a1 again", [{"source": "new.pdf", "page": 1}])
    hit = cache.get([1.0, 0.0, 0.0])

    assert hit is not None
    assert hit["answer"] == "a1 again"
