"""metrics.bucket_series takes a separate arbitration window (health_advisor#533).

Workout-window device arbitration only matches a workout that STARTS inside
the window it is given.  A caller reading a sub-span that opens after its
workout began (a block within a session) therefore needs to read one span and
arbitrate over another.  The fixture is synthetic: one running workout with a
GymKit stream and a watch stream recording the same distance.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from health_advisor import db
from health_advisor import metrics as mx

WORKOUT = ("2026-09-01T10:00:00Z", "2026-09-01T10:10:00Z")
READ = ("2026-09-01T10:02:00Z", "2026-09-01T10:06:00Z")   # opens after the workout
BUCKETS = 12                                              # 4 min / 20 s
MILES_PER_BUCKET = 0.0333                                 # ~10 min/mi


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    c.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('running', ?, ?, "
        "'2026-09-01', 10.0, 'test', 'arb-window-workout')", WORKOUT)
    start = datetime.fromisoformat(READ[0].replace("Z", "+00:00"))
    for source in ("GymKit", "Demo Apple Watch"):
        for i in range(BUCKETS):
            ts = (start + timedelta(seconds=20 * i)).astimezone(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            c.execute(
                "INSERT INTO records (metric, start_utc, end_utc, local_date, "
                "value, unit, source, dedupe_key) VALUES "
                "('distance_walking_running', ?, ?, '2026-09-01', ?, 'mi', ?, ?)",
                (ts, ts, MILES_PER_BUCKET, source, f"arb|{source}|{ts}"))
    c.commit()
    yield c
    c.close()


def _miles(rows):
    return sum(r["miles"] for r in rows)


def test_default_arbitration_is_the_read_window(conn):
    """No kwarg: the workout starts before the read window, matches nothing,
    and both streams are summed -- the behaviour every caller has today."""
    default = mx.bucket_series(conn, *READ)
    explicit = mx.bucket_series(conn, *READ, arbitration_window=READ)
    assert default == explicit
    assert _miles(default) == pytest.approx(2 * BUCKETS * MILES_PER_BUCKET)


def test_wider_arbitration_window_selects_one_source(conn):
    rows = mx.bucket_series(conn, *READ, arbitration_window=WORKOUT)
    assert len(rows) == BUCKETS
    assert _miles(rows) == pytest.approx(BUCKETS * MILES_PER_BUCKET)
    # Only the read span is returned, however wide the arbitration window.
    assert {r["bucket_start_utc"] for r in rows} <= {
        (datetime(2026, 9, 1, 10, 2, tzinfo=timezone.utc)
         + timedelta(seconds=20 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(BUCKETS)}


def test_arbitration_window_does_not_widen_the_read(conn):
    """The buckets are the read window's, identical to impact_bucket_rows."""
    rows = mx.bucket_series(conn, *READ, arbitration_window=WORKOUT)
    raw = mx.impact_bucket_rows(
        conn, "start_utc >= ? AND start_utc < ?", READ,
        arbitration_window=WORKOUT, arbitration_window_kind="utc")
    assert [r["miles"] for r in rows] == [float(r["mi"]) for r in raw]
