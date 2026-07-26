from src.telemetry import InMemoryTelemetry


def test_stats_empty_when_nothing_logged():
    t = InMemoryTelemetry()
    assert t.get_stats() == {"total_queries": 0}


def test_stats_aggregate_correctly():
    t = InMemoryTelemetry()
    t.log_query("q1", retrieval_ms=100, generation_ms=200, total_ms=300, num_sources=2, cache_hit=False, emergency=False)
    t.log_query("q2", retrieval_ms=None, generation_ms=None, total_ms=5, num_sources=0, cache_hit=True, emergency=False)
    t.log_query("q3", retrieval_ms=None, generation_ms=None, total_ms=10, num_sources=0, cache_hit=False, emergency=True)

    stats = t.get_stats()
    assert stats["total_queries"] == 3
    assert stats["avg_total_ms"] == round((300 + 5 + 10) / 3, 1)
    # retrieval/generation averages should only count entries where they
    # were actually recorded (not the cache-hit/emergency shortcuts, which
    # pass None for those fields)
    assert stats["avg_retrieval_ms"] == 100
    assert stats["avg_generation_ms"] == 200
    assert stats["cache_hit_rate"] == round(1 / 3, 3)
    assert stats["cache_hits"] == 1
    assert stats["emergency_rate"] == round(1 / 3, 3)
    assert stats["emergency_count"] == 1


def test_feedback_counts_split_by_rating():
    t = InMemoryTelemetry()
    t.log_query("q", None, None, 1, 0, False, False)  # need at least one query for get_stats to return counts
    t.log_feedback("q1", "a1", "up")
    t.log_feedback("q2", "a2", "up")
    t.log_feedback("q3", "a3", "down")

    stats = t.get_stats()
    assert stats["feedback_up"] == 2
    assert stats["feedback_down"] == 1


def test_init_db_is_a_no_op():
    # Should never raise -- InMemoryTelemetry has nothing to create.
    InMemoryTelemetry().init_db()


def test_daily_stats_zero_filled_when_nothing_logged():
    t = InMemoryTelemetry()
    daily = t.get_daily_stats(days=14)
    assert len(daily) == 14
    assert all(d["queries"] == 0 and d["avg_ms"] == 0 and d["cache_hits"] == 0 for d in daily)
    # oldest to newest, ISO dates, no gaps
    dates = [d["date"] for d in daily]
    assert dates == sorted(dates)


def test_daily_stats_always_returns_exactly_days_entries():
    t = InMemoryTelemetry()
    t.log_query("q", 100, 100, 200, 1, False, False)
    assert len(t.get_daily_stats(days=7)) == 7
    assert len(t.get_daily_stats(days=30)) == 30


def test_daily_stats_counts_queries_logged_just_now():
    t = InMemoryTelemetry()
    t.log_query("q1", 100, 200, 300, 2, False, False)
    t.log_query("q2", None, None, 100, 1, True, False)

    today = t.get_daily_stats(days=14)[-1]
    assert today["queries"] == 2
    assert today["avg_ms"] == 200.0  # (300 + 100) / 2
    assert today["cache_hits"] == 1


def test_daily_stats_buckets_by_day_and_excludes_out_of_window_entries():
    from datetime import datetime, timedelta, timezone

    t = InMemoryTelemetry()
    now = datetime.now(timezone.utc)

    def inject(days_ago, total_ms, cache_hit):
        t.query_logs.append(
            {
                "question": "q",
                "retrieval_ms": None,
                "generation_ms": None,
                "total_ms": total_ms,
                "num_sources": 1,
                "cache_hit": cache_hit,
                "emergency": False,
                "created_at": (now - timedelta(days=days_ago)).isoformat(),
            }
        )

    inject(days_ago=0, total_ms=1000.0, cache_hit=True)
    inject(days_ago=0, total_ms=2000.0, cache_hit=False)
    inject(days_ago=1, total_ms=500.0, cache_hit=False)
    inject(days_ago=20, total_ms=999.0, cache_hit=False)  # outside a 14-day window

    daily = t.get_daily_stats(days=14)
    assert len(daily) == 14  # the 20-day-old entry doesn't extend the series

    today, yesterday = daily[-1], daily[-2]
    assert today["queries"] == 2
    assert today["avg_ms"] == 1500.0
    assert today["cache_hits"] == 1
    assert yesterday["queries"] == 1
    assert yesterday["avg_ms"] == 500.0

    total_queries_in_series = sum(d["queries"] for d in daily)
    assert total_queries_in_series == 3  # the out-of-window entry is genuinely excluded, not just unbucketed


def test_daily_stats_ignores_entries_with_no_recorded_timestamp():
    # Defensive: a hand-built fake/older log entry missing created_at
    # shouldn't crash get_daily_stats, just be skipped.
    t = InMemoryTelemetry()
    t.query_logs.append(
        {
            "question": "q",
            "retrieval_ms": None,
            "generation_ms": None,
            "total_ms": 100.0,
            "num_sources": 1,
            "cache_hit": False,
            "emergency": False,
        }
    )
    daily = t.get_daily_stats(days=14)  # should not raise
    assert sum(d["queries"] for d in daily) == 0
