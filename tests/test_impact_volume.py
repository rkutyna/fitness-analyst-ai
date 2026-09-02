"""Impact volume — jog minutes vs walk minutes from raw distance samples.

The point of this metric is that workout *duration* counts walk breaks and so
overstates running impact; these tests pin the pace classification, the bucket
resolution that makes short intervals visible, and the week grouping the ramp
rule is defined on.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from health_advisor import analysis as A
from health_advisor import db
from health_advisor import metrics as mx


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    return conn


def _emit(conn, local_date: str, start_hhmm: str, seconds: int,
          pace_min_per_mi: float, hr: float | None = None,
          cadence_spm: float = 141.0):
    """Write a workout, distance samples, and optional HR/cadence samples."""
    t0 = datetime.fromisoformat(f"{local_date}T{start_hhmm}:00+00:00")
    end = t0 + timedelta(seconds=seconds)
    start = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('other', ?, ?, ?, ?, 'test', ?)",
        (start, end_utc, local_date, seconds / 60.0,
         f"workout|{start}|{end_utc}"),
    )
    per_second_mi = 1.0 / (pace_min_per_mi * 60.0)
    for i in range(seconds):
        ts = (t0 + timedelta(seconds=i)).isoformat()
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('distance_walking_running', ?, 'mi', ?, ?, ?, ?, 't', 't', ?)",
            (per_second_mi, ts, ts, ts, local_date, f"{local_date}-{start_hhmm}-{i}"),
        )
        if hr is not None and i % 5 == 0:
            conn.execute(
                "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
                "start_local, local_date, source, origin, dedupe_key) "
                "VALUES ('heart_rate', ?, 'count/min', ?, ?, ?, ?, 't', 't', ?)",
                (hr, ts, ts, ts, local_date, f"hr-{local_date}-{start_hhmm}-{i}"),
            )
        if cadence_spm and i % mx.IMPACT_BUCKET_SECONDS == 0:
            conn.execute(
                "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
                "start_local, local_date, source, origin, dedupe_key) "
                "VALUES ('step_count', ?, 'count', ?, ?, ?, ?, 't', 't', ?)",
                (cadence_spm / 3.0, ts, ts, ts, local_date,
                 f"steps-{local_date}-{start_hhmm}-{i}"),
            )


def test_jog_and_walk_are_split_by_pace():
    conn = _conn()
    _emit(conn, "2026-07-28", "12:00", 300, 14.0)   # 5 min jogging
    _emit(conn, "2026-07-28", "12:05", 300, 22.0, cadence_spm=0.0)   # 5 min walking
    (row,) = A.impact_volume(conn, "2026-07-28", "2026-07-28", by="day")

    assert row["jog_minutes"] == pytest.approx(5.0, abs=0.4)
    assert row["walk_minutes"] == pytest.approx(5.0, abs=0.4)
    assert row["jog_pace_min_per_mi"] == pytest.approx(14.0, abs=0.5)


def test_walk_break_inside_a_run_is_not_counted_as_impact():
    """The whole reason this metric exists: a 20-minute run/walk session is not
    20 minutes of impact."""
    conn = _conn()
    for i in range(4):                                   # 4 x (2 min jog / 3 min walk)
        _emit(conn, "2026-07-28", f"12:{i * 5:02d}", 120, 14.0)
        _emit(conn, "2026-07-28", f"12:{i * 5 + 2:02d}", 180, 22.0,
              cadence_spm=0.0)
    (row,) = A.impact_volume(conn, "2026-07-28", "2026-07-28", by="day")

    assert row["jog_minutes"] == pytest.approx(8.0, abs=0.7)   # not 20
    assert row["walk_minutes"] == pytest.approx(12.0, abs=0.7)


def test_ninety_second_interval_survives_bucketing():
    """20s buckets exist so Week 1-2 style 90-second jogs don't wash out."""
    conn = _conn()
    _emit(conn, "2026-07-28", "12:00", 90, 13.0)
    _emit(conn, "2026-07-28", "12:01", 90, 24.0, cadence_spm=0.0)
    (row,) = A.impact_volume(conn, "2026-07-28", "2026-07-28", by="day")

    assert row["jog_minutes"] >= 1.0


def test_daily_total_is_not_a_bucket_even_below_the_pace_ceiling():
    """A coarse writer cannot pass by looking like a slow 20-second sample."""
    conn = _conn()
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) VALUES "
        "('distance_walking_running', 0.05, 'mi', ?, ?, ?, ?, ?, 'test', ?)",
        (
            "2026-07-01T12:00:00+00:00",
            "2026-07-02T11:59:59+00:00",
            "2026-07-01 12:00:00",
            "2026-07-01",
            "Future Daily Writer",
            "daily-total-span-guard",
        ),
    )
    conn.commit()

    assert mx.bucket_series(
        conn, "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"
    ) == []
    assert A.impact_volume(conn, "2026-07-01", "2026-07-01", by="day") == []


def test_pace_no_longer_controls_jog_classification():
    """Cadence, not the old pace lane, controls jogging inside a workout."""
    conn = _conn()
    _emit(conn, "2026-07-28", "12:00", 120, 15.5)
    _emit(conn, "2026-07-29", "12:00", 120, 17.0)
    rows = {r["period_start"]: r for r in
            A.impact_volume(conn, "2026-07-28", "2026-07-29", by="day")}

    assert rows["2026-07-28"]["jog_minutes"] > 1.5
    assert rows["2026-07-29"]["jog_minutes"] > 1.5


def test_slow_jog_confirmed_by_heart_rate_counts_as_impact():
    """The Week 5 bug: jogging slowly enough to hold the HR cap fell off the far
    side of the 16 min/mi cutoff and scored as walking. A 10-minute jog at 18
    min/mi with the heart rate of a run is a jog."""
    conn = _conn()
    _emit(conn, "2026-07-29", "12:00", 600, 17.0, hr=140)
    (row,) = A.impact_volume(conn, "2026-07-29", "2026-07-29", by="day")

    assert row["jog_minutes"] == pytest.approx(10.0, abs=0.7)


def test_brisk_walk_at_the_same_pace_is_not_impact():
    """The other half of the rule: pace alone can't tell a 17 min/mi jog from a
    17 min/mi brisk walk — heart rate is what separates them."""
    conn = _conn()
    _emit(conn, "2026-07-29", "12:00", 600, 17.0, hr=105, cadence_spm=0.0)
    (row,) = A.impact_volume(conn, "2026-07-29", "2026-07-29", by="day")

    assert row["jog_minutes"] == 0.0
    assert row["walk_minutes"] == pytest.approx(10.0, abs=0.7)


def test_low_cadence_hiking_is_not_rescued_by_a_high_heart_rate():
    """HR no longer promotes a slow bucket; cadence keeps this hike walking."""
    conn = _conn()
    _emit(conn, "2026-07-29", "12:00", 600, 19.0, hr=145, cadence_spm=0.0)
    (row,) = A.impact_volume(conn, "2026-07-29", "2026-07-29", by="day")

    assert row["jog_minutes"] == 0.0
    assert row["walk_minutes"] == pytest.approx(10.0, abs=0.7)


def test_fast_pace_still_counts_with_no_heart_rate_data():
    """Backward compatibility: most of the 2019-2021 history has distance but no
    usable HR. Unambiguously fast buckets must not need confirming."""
    conn = _conn()
    _emit(conn, "2026-07-29", "12:00", 600, 13.0)      # no hr argument
    (row,) = A.impact_volume(conn, "2026-07-29", "2026-07-29", by="day")

    assert row["jog_minutes"] == pytest.approx(10.0, abs=0.7)


def test_heart_rate_confirmed_jog_is_not_also_counted_as_walking():
    """jog and walk buckets partition the session; a bucket promoted into the jog
    lane has to leave the walk lane."""
    conn = _conn()
    _emit(conn, "2026-07-29", "12:00", 300, 18.0, hr=140)   # jog, HR-confirmed
    _emit(conn, "2026-07-29", "12:05", 300, 22.0, hr=105,
          cadence_spm=0.0)   # walk
    (row,) = A.impact_volume(conn, "2026-07-29", "2026-07-29", by="day")

    assert row["jog_minutes"] == pytest.approx(5.0, abs=0.4)
    assert row["walk_minutes"] == pytest.approx(5.0, abs=0.4)


def test_weeks_are_monday_anchored_with_change_pct():
    conn = _conn()
    _emit(conn, "2026-07-22", "12:00", 600, 14.0)   # Wed, week of Jul 20
    _emit(conn, "2026-07-29", "12:00", 690, 14.0)   # Wed, week of Jul 27 (+15%)
    rows = A.impact_volume(conn, "2026-07-20", "2026-08-02", by="week")

    assert [r["period_start"] for r in rows] == ["2026-07-20", "2026-07-27"]
    assert rows[0]["jog_change_pct"] is None          # no prior period
    assert rows[1]["jog_change_pct"] == pytest.approx(15.0, abs=2.0)


def test_stationary_noise_is_neither_jog_nor_walk():
    conn = _conn()
    _emit(conn, "2026-07-28", "12:00", 300, 120.0,
          cadence_spm=0.0)    # shuffling at a desk
    (row,) = A.impact_volume(conn, "2026-07-28", "2026-07-28", by="day")

    assert row["jog_minutes"] == 0.0
    assert row["walk_minutes"] == 0.0


def test_empty_range_returns_no_rows():
    assert A.impact_volume(_conn(), "2026-01-01", "2026-01-07") == []


def test_invalid_grouping_rejected():
    with pytest.raises(ValueError):
        A.impact_volume(_conn(), "2026-07-01", "2026-07-07", by="month")
