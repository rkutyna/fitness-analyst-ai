"""F-84: arbitrate live movement by day and distribute interval samples."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from health_advisor import analysis
from health_advisor import db
from health_advisor import metrics
from tests.fixture_loader import load_f84_wholeday


def _distance(conn, start, end, day, value, source="Demo’s Apple Watch"):
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) "
        "VALUES ('distance_walking_running', ?, 'mi', ?, ?, NULL, ?, ?, "
        "'test', ?)",
        (value, start, end, day, source, f"f84|{start}|{end}|{source}"),
    )


def test_whole_day_arbitration_reaches_a_day_without_a_workout(conn):
    """The iPhone stream on Aug 23 is excluded although that day has no workout."""
    load_f84_wholeday(conn)
    clause, args = db._workout_arbitration(conn, "distance_walking_running")
    row = conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric = ? AND local_date = ? "
        "AND source LIKE '%iPhone'" + clause,
        ("distance_walking_running", "2026-08-23", *args),
    ).fetchone()
    assert row[0] == 0


def test_long_interval_is_distributed_and_not_classified_as_a_jog(conn):
    """A 599-second sample retains distance but cannot become a fast bucket."""
    start = datetime(2026, 8, 30, 12, 0, 1, tzinfo=timezone.utc)
    end = start + timedelta(seconds=599)
    _distance(conn, start.isoformat(), end.isoformat(), "2026-08-30", 0.08)
    conn.commit()

    rows = metrics.bucket_series(
        conn, "2026-08-30T00:00:00Z", "2026-08-31T00:00:00Z")
    assert len(rows) >= 30
    assert sum(row["miles"] for row in rows) == pytest.approx(0.08)
    assert not any(row["is_jog"] for row in rows)


def test_fixture_without_cadence_cannot_classify_jog_volume(conn):
    """The committed F-84 fixture predates the cadence stream.

    The live-vault week-8 oracle is measured separately in the issue; this
    fixture deliberately verifies that distance/HR alone no longer sneak
    through the dose ceiling.
    """
    payload = load_f84_wholeday(conn)
    week = analysis.impact_volume(conn, "2026-08-17", "2026-08-23", by="week")[0]
    days = {
        row["period_start"]: row["jog_minutes"]
        for row in analysis.impact_volume(
            conn, "2026-08-17", "2026-08-23", by="day")
    }

    assert week["jog_minutes"] == pytest.approx(0.0)
    assert all(value == pytest.approx(0.0) for value in days.values())
