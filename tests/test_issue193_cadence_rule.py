"""#193: impact-volume jogging is cadence- and workout-window scoped."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from health_advisor import analysis
from health_advisor import metrics


def _workout(conn, start: str, seconds: int, workout_type: str = "walking"):
    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end = (t0 + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = t0.date().isoformat()
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES (?, ?, ?, ?, ?, 'test', ?)",
        (workout_type, start, end, day, seconds / 60.0, f"w|{start}|{end}"),
    )


def _record(conn, metric: str, start: str, value: float, end: str | None = None):
    day = start[:10]
    end = end or start
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, local_date, "
        "source, origin, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, 'test', 'test', ?)",
        (metric, value, "count" if metric == "step_count" else "mi",
         start, end, day, f"{metric}|{start}|{value}|{end}"),
    )


def test_cadence_is_start_bucket_sum_times_three_and_window_is_type_agnostic(conn):
    _workout(conn, "2026-07-01T12:00:00Z", 40, workout_type="walking")
    _record(conn, "distance_walking_running", "2026-07-01T12:00:00Z", 0.03)
    _record(conn, "distance_walking_running", "2026-07-01T12:00:20Z", 0.03)
    _record(conn, "step_count", "2026-07-01T12:00:00Z", 47.0)
    _record(conn, "step_count", "2026-07-01T12:00:20Z", 140.0 / 3.0)
    conn.commit()

    rows = metrics.bucket_series(conn, "2026-07-01T12:00:00Z",
                                 "2026-07-01T12:01:00Z")

    assert [row["cadence_spm"] for row in rows] == pytest.approx([141.0, 140.0])
    assert [row["is_jog"] for row in rows] == [True, True]


def test_high_cadence_outside_workout_window_is_not_jog(conn):
    _workout(conn, "2026-07-01T12:00:00Z", 20)
    _record(conn, "distance_walking_running", "2026-07-01T11:59:40Z", 0.03)
    _record(conn, "distance_walking_running", "2026-07-01T12:00:00Z", 0.03)
    _record(conn, "step_count", "2026-07-01T11:59:40Z", 47.0)
    _record(conn, "step_count", "2026-07-01T12:00:00Z", 47.0)
    conn.commit()

    rows = metrics.bucket_series(conn, "2026-07-01T11:59:40Z",
                                 "2026-07-01T12:00:20Z")

    assert [row["is_jog"] for row in rows] == [False, True]
    assert [row["is_walk"] for row in rows] == [True, False]


def test_step_count_is_not_distributed_across_buckets(conn):
    _workout(conn, "2026-07-01T12:00:00Z", 60)
    for i in range(3):
        start = f"2026-07-01T12:00:{i * 20:02d}Z"
        _record(conn, "distance_walking_running", start, 0.03)
    _record(conn, "step_count", "2026-07-01T12:00:00Z", 47.0,
            "2026-07-01T12:03:05Z")
    conn.commit()

    rows = metrics.bucket_series(conn, "2026-07-01T12:00:00Z",
                                 "2026-07-01T12:01:00Z")

    assert rows[0]["cadence_spm"] == pytest.approx(141.0)
    assert rows[1]["cadence_spm"] is None
    assert rows[2]["cadence_spm"] is None
    assert [row["is_jog"] for row in rows] == [True, False, False]


def test_implausible_pace_floor_and_raw_tables_are_unchanged(conn):
    _workout(conn, "2026-07-01T12:00:00Z", 20)
    _record(conn, "distance_walking_running", "2026-07-01T12:00:00Z", 0.07)
    _record(conn, "step_count", "2026-07-01T12:00:00Z", 47.0)
    conn.commit()
    counts_before = tuple(conn.execute(
        "SELECT (SELECT COUNT(*) FROM records), "
        "(SELECT COUNT(*) FROM daily_metrics), "
        "(SELECT COUNT(*) FROM insights)").fetchone())

    rows = metrics.bucket_series(conn, "2026-07-01T12:00:00Z",
                                 "2026-07-01T12:00:20Z")
    analysis.impact_volume(conn, "2026-07-01", "2026-07-01", by="day")

    assert rows[0]["is_jog"] is False  # 4.76 min/mi remains below the floor
    counts_after = tuple(conn.execute(
        "SELECT (SELECT COUNT(*) FROM records), "
        "(SELECT COUNT(*) FROM daily_metrics), "
        "(SELECT COUNT(*) FROM insights)").fetchone())
    assert counts_after == counts_before
