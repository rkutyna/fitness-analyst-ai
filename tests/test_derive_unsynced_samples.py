"""The nightly derive writes no longest_block_min for an unsynced running day.

A running workout whose distance, step and heart-rate series have not synced
has no distance buckets. derive._dial_for_day used to take max([0.0]) over its
sessions, so the day stored a longest block of 0.0 min: an unmeasured value
published as a measured zero. The rule now matches a day with no session: no
row is written. A day with at least one measured session is unchanged, and the
first derive after its samples arrive writes the real value.

Oracles are literals. The measured-day literals (13.3 and 4.0 minutes) were
captured from the code BEFORE this change on the same synthetic vault.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from health_advisor import derive
from health_advisor import metrics as mx

BUCKET_MIN = mx.IMPACT_BUCKET_SECONDS / 60.0
DAY_MEASURED = "2026-07-02"    # 40 jog buckets: longest block 13.3 min
DAY_UNSYNCED = "2026-07-03"    # 40-min run with running_speed samples only
DAY_FEW = "2026-07-04"         # 12 jog buckets: longest block 4.0 min
DAY_REST = "2026-07-05"        # no workout of any kind
BLOCK = "longest_block_min"
UNSYNCED_BASE = "2026-07-03T12:00:00Z"


def _iso(base: str, i: int) -> str:
    t = datetime.fromisoformat(base.replace("Z", "+00:00"))
    return (t + timedelta(seconds=mx.IMPACT_BUCKET_SECONDS * i)
            ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(conn, metric, start, value, unit):
    conn.execute(
        "INSERT INTO records (metric, start_utc, end_utc, local_date, value, "
        "unit, source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, 'test', ?)",
        (metric, start, start, start[:10], value, unit,
         f"{metric}|{start}|{value}"))


def _workout(conn, start, end, local_date, minutes, avg_hr):
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, avg_heart_rate, source, dedupe_key) "
        "VALUES ('running', ?, ?, ?, ?, ?, 'test', ?)",
        (start, end, local_date, minutes, avg_hr, f"w|{start}|running"))


def _jog_buckets(conn, base: str, buckets: int):
    """Samples for `buckets` jog buckets: pace 14, cadence 135, HR 140."""
    for i in range(buckets):
        s = _iso(base, i)
        _record(conn, "distance_walking_running", s, BUCKET_MIN / 14.0, "mi")
        _record(conn, "step_count", s, 135.0 / 3.0, "count")
        _record(conn, "heart_rate", s, 140.0, "count/min")


def _vault(conn):
    _jog_buckets(conn, "2026-07-02T12:00:00Z", 40)
    _workout(conn, "2026-07-02T12:00:00Z", _iso("2026-07-02T12:00:00Z", 40),
             DAY_MEASURED, 40 * BUCKET_MIN, 140.0)
    for i in range(120):          # the unsynced run: running_speed only
        _record(conn, "running_speed", _iso(UNSYNCED_BASE, i), 2.7, "m/s")
    _workout(conn, UNSYNCED_BASE, _iso(UNSYNCED_BASE, 120), DAY_UNSYNCED,
             40.0, None)
    _jog_buckets(conn, "2026-07-04T12:00:00Z", 12)
    _workout(conn, "2026-07-04T12:00:00Z", _iso("2026-07-04T12:00:00Z", 12),
             DAY_FEW, 12 * BUCKET_MIN, 140.0)
    conn.commit()


def _block_value(conn, day: str):
    row = conn.execute(
        "SELECT avg FROM daily_metrics WHERE metric = ? AND date = ?",
        (BLOCK, day)).fetchone()
    return None if row is None else row["avg"]


def test_unsynced_day_writes_no_block_row(conn):
    _vault(conn)
    derive.update_for_days(conn, [DAY_MEASURED, DAY_UNSYNCED, DAY_FEW, DAY_REST])
    conn.commit()

    assert _block_value(conn, DAY_UNSYNCED) is None
    assert _block_value(conn, DAY_REST) is None      # rest day: still no row
    assert _block_value(conn, DAY_MEASURED) == 13.3  # captured before the fix
    assert _block_value(conn, DAY_FEW) == 4.0


def test_later_derive_after_samples_arrive_writes_real_block(conn):
    _vault(conn)
    derive.update_for_days(conn, [DAY_UNSYNCED])
    conn.commit()
    assert _block_value(conn, DAY_UNSYNCED) is None

    # The distance, step and heart-rate samples for the same run now sync.
    _jog_buckets(conn, UNSYNCED_BASE, 40)
    conn.commit()
    derive.update_for_days(conn, [DAY_UNSYNCED])
    conn.commit()

    assert _block_value(conn, DAY_UNSYNCED) == 13.3


def test_stale_zero_block_row_is_removed_by_recompute(conn):
    """A 0.0 row written by the old code is removed, not kept as a claim."""
    _vault(conn)
    derive._upsert(conn, BLOCK, DAY_UNSYNCED, 0.0)
    conn.commit()
    derive.update_for_days(conn, [DAY_UNSYNCED])
    conn.commit()

    assert _block_value(conn, DAY_UNSYNCED) is None


def test_indoor_only_day_writes_no_block_row(conn):
    """An indoor run never gets samples, so it must not write a block row."""
    base = "2026-07-08T12:00:00Z"
    for i in range(120):
        _record(conn, "running_speed", _iso(base, i), 2.7, "m/s")
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, avg_heart_rate, is_indoor, source, dedupe_key) "
        "VALUES ('running', ?, ?, '2026-07-08', 40.0, NULL, 1, 'test', ?)",
        (base, _iso(base, 120), f"w|{base}|running"))
    conn.commit()

    derive.update_for_days(conn, ["2026-07-08"])
    conn.commit()

    assert _block_value(conn, "2026-07-08") is None
