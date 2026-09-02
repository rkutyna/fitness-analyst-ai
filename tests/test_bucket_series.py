"""metrics.bucket_series — the per-bucket view of the same classification
analysis.impact_volume aggregates. The consistency test at the bottom is the
one that matters: it is what stops the two implementations drifting."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from math import nextafter

import pytest

from health_advisor import analysis as A
from health_advisor import db
from health_advisor import metrics as mx


def test_snapshot_has_exactly_one_mixed_mirror_bucket():
    """Pin the raw snapshot fact corrected in #24: one mixed bucket remains.

    NOT marked `live` — that marker is for tests needing a local Ollama model,
    and this one needs the snapshot, which it skips for explicitly below. It was
    tagged `live` when written, which meant it never ran: a test pinning a
    measured fact, silently skipped, is the same defect class as the rest of #24.
    """
    path = db.REPO_ROOT / "data" / "health.db"
    if not path.exists():
        pytest.skip("production snapshot is not present")
    live = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    live.row_factory = sqlite3.Row
    try:
        rows = live.execute(
            """
            WITH bucketed AS (
              SELECT local_date,
                     CAST(strftime('%s', start_utc) / ? AS INTEGER) AS bkt,
                     source
                FROM records
               WHERE metric = 'distance_walking_running'
            )
            SELECT local_date, bkt,
                   SUM(CASE WHEN source = 'Sync Solver' THEN 1 ELSE 0 END) AS mirror_rows,
                   SUM(CASE WHEN source <> 'Sync Solver' OR source IS NULL THEN 1 ELSE 0 END)
                       AS non_mirror_rows
              FROM bucketed
          GROUP BY local_date, bkt
            HAVING mirror_rows > 0 AND non_mirror_rows > 0
          ORDER BY local_date, bkt
            """,
            (mx.IMPACT_BUCKET_SECONDS,),
        ).fetchall()
    finally:
        live.close()

    assert [(row["local_date"], row["mirror_rows"], row["non_mirror_rows"])
            for row in rows] == [("2018-07-22", 1, 1)]


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    yield c
    c.close()


def _add_workout(c, start_utc, local_date, seconds=20, workout_type="other"):
    from datetime import datetime, timedelta
    start = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    end = (start + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    c.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES (?, ?, ?, ?, ?, 'test', ?)",
        (workout_type, start_utc, end, local_date, seconds / 60.0,
         f"w|{start_utc}|{end}|{workout_type}"),
    )


def _add_distance(c, start_utc, local_date, miles, cadence_spm=141.0):
    c.execute(
        "INSERT INTO records (metric, start_utc, end_utc, local_date, value, unit, "
        "source, dedupe_key) VALUES ('distance_walking_running', ?, ?, ?, ?, 'mi', "
        "'test', ?)",
        (start_utc, start_utc, local_date, miles, f"d|{start_utc}|{miles}"))
    _add_workout(c, start_utc, local_date)
    if cadence_spm:
        c.execute(
            "INSERT INTO records (metric, start_utc, end_utc, local_date, value, unit, "
            "source, dedupe_key) VALUES ('step_count', ?, ?, ?, ?, 'count', 'test', ?)",
            (start_utc, start_utc, local_date, cadence_spm / 3.0,
             f"s|{start_utc}|{cadence_spm}"))


def _add_hr(c, start_utc, local_date, bpm):
    c.execute(
        "INSERT INTO records (metric, start_utc, end_utc, local_date, value, unit, "
        "source, dedupe_key) VALUES ('heart_rate', ?, ?, ?, ?, 'count/min', "
        "'test', ?)",
        (start_utc, start_utc, local_date, bpm, f"h|{start_utc}|{bpm}"))


def test_fast_bucket_is_jog_on_pace_alone(conn):
    # 20 s at 10 min/mi = 0.0333 mi. Well inside the 16 min/mi pace lane.
    _add_distance(conn, "2026-07-01T12:00:00Z", "2026-07-01", 0.0333)
    conn.commit()
    rows = mx.bucket_series(conn, "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z")
    assert len(rows) == 1
    assert rows[0]["is_jog"] is True
    assert rows[0]["is_walk"] is False
    assert rows[0]["hr"] is None
    assert rows[0]["pace_min_per_mi"] == pytest.approx(10.0, abs=0.1)
    assert rows[0]["speed_mph"] == pytest.approx(6.0, abs=0.1)


def test_slow_bucket_with_high_hr_is_jog_on_the_hr_lane(conn):
    # 20 s at 17.5 min/mi = 0.01905 mi — too slow for the pace lane, inside the
    # 18 min/mi HR lane. With HR 135 (>=130) it is a jog.
    _add_distance(conn, "2026-07-01T12:00:00Z", "2026-07-01", 0.01905)
    _add_hr(conn, "2026-07-01T12:00:05Z", "2026-07-01", 135.0)
    conn.commit()
    rows = mx.bucket_series(conn, "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z")
    assert rows[0]["is_jog"] is True
    assert rows[0]["hr"] == pytest.approx(135.0)


def test_same_bucket_with_low_hr_is_still_a_jog_at_running_cadence(conn):
    _add_distance(conn, "2026-07-01T12:00:00Z", "2026-07-01", 0.01905)
    _add_hr(conn, "2026-07-01T12:00:05Z", "2026-07-01", 105.0)
    conn.commit()
    rows = mx.bucket_series(conn, "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z")
    # HR is diagnostic only for this dial; cadence is the discriminator.
    assert rows[0]["is_jog"] is True
    assert rows[0]["is_walk"] is False


def test_very_slow_bucket_is_neither_jog_nor_walk(conn):
    # 20 s at 60 min/mi = 0.00556 mi. Slower than the 40 min/mi walk floor:
    # standing or GPS noise, and it must not be counted as walking.
    _add_distance(conn, "2026-07-01T12:00:00Z", "2026-07-01", 0.00556,
                  cadence_spm=0.0)
    conn.commit()
    rows = mx.bucket_series(conn, "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z")
    assert rows[0]["is_jog"] is False
    assert rows[0]["is_walk"] is False


def test_window_is_exclusive_of_end(conn):
    _add_distance(conn, "2026-07-01T12:00:00Z", "2026-07-01", 0.0333)
    _add_distance(conn, "2026-07-01T13:00:00Z", "2026-07-01", 0.0333)
    conn.commit()
    rows = mx.bucket_series(conn, "2026-07-01T12:00:00Z", "2026-07-01T13:00:00Z")
    assert len(rows) == 1
    assert rows[0]["bucket_start_utc"].startswith("2026-07-01T12:00")


def test_empty_window_returns_empty_list(conn):
    rows = mx.bucket_series(conn, "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z")
    assert rows == []


def test_gps_pause_leaves_a_hole_rather_than_a_slow_bucket(conn):
    """A paused GPS produces no distance samples, not slow ones. The gap must
    simply be absent from the series — inventing a 'standing' bucket there
    would put phantom time into every downstream measure."""
    _add_distance(conn, "2026-07-01T12:00:00Z", "2026-07-01", 0.0333)
    # 12:00:20 through 12:01:20 missing entirely — the pause.
    _add_distance(conn, "2026-07-01T12:01:40Z", "2026-07-01", 0.0333)
    conn.commit()
    rows = mx.bucket_series(conn, "2026-07-01T12:00:00Z", "2026-07-01T12:02:00Z")
    assert len(rows) == 2
    assert all(r["is_jog"] for r in rows)


def test_hr_lag_after_a_walk_break_does_not_change_cadence_classification(conn):
    """The dose ceiling uses cadence, so HR lag cannot strip the first rep bucket."""
    # A 17.5 min/mi bucket is still a jog when cadence is at the gait threshold.
    _add_distance(conn, "2026-07-01T12:00:00Z", "2026-07-01", 0.01905)
    _add_hr(conn, "2026-07-01T12:00:05Z", "2026-07-01", 112.0)   # HR hasn't caught up
    _add_distance(conn, "2026-07-01T12:00:20Z", "2026-07-01", 0.01905)
    _add_hr(conn, "2026-07-01T12:00:25Z", "2026-07-01", 138.0)   # now it has
    conn.commit()
    rows = mx.bucket_series(conn, "2026-07-01T12:00:00Z", "2026-07-01T12:01:00Z")
    assert [r["is_jog"] for r in rows] == [True, True]


def test_bucket_minutes_agree_with_impact_volume(conn):
    """The consistency gate. Jog minutes counted bucket-by-bucket must equal
    what impact_volume aggregates for the same day. Two implementations of one
    rule; this is what keeps them from diverging."""
    # A mixed session: six jog buckets, three walk buckets, one standing.
    # t0 MUST correspond to the local_date the rows are given below: bucket_series
    # filters on start_utc while impact_volume filters on local_date, so a fixture
    # where the two disagree compares an empty window against a populated day.
    t0 = 1782907200  # 2026-07-01T12:00:00Z
    from datetime import datetime, timezone
    for i in range(6):
        ts = datetime.fromtimestamp(t0 + i * 20, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _add_distance(conn, ts, "2026-07-01", 0.0333)
    for i in range(6, 9):
        ts = datetime.fromtimestamp(t0 + i * 20, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _add_distance(conn, ts, "2026-07-01", 0.0125)   # 26.7 min/mi — a walk
    ts = datetime.fromtimestamp(t0 + 9 * 20, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _add_distance(conn, ts, "2026-07-01", 0.00556)      # standing
    conn.commit()

    buckets = mx.bucket_series(conn, "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z")
    bucket_jog_min = sum(1 for b in buckets if b["is_jog"]) * (mx.IMPACT_BUCKET_SECONDS / 60.0)

    iv = A.impact_volume(conn, "2026-07-01", "2026-07-01", by="day")
    assert iv, "impact_volume returned nothing for a day with samples"
    assert bucket_jog_min == pytest.approx(iv[0]["jog_minutes"], abs=0.05)

    bucket_walk_min = sum(1 for b in buckets if b["is_walk"]) * (mx.IMPACT_BUCKET_SECONDS / 60.0)
    assert bucket_walk_min == pytest.approx(iv[0]["walk_minutes"], abs=0.05)


def test_multi_day_bucket_and_daily_surfaces_agree_on_the_same_days(conn):
    """The UTC and local-date questions agree when their windows are equivalent.

    The zero-distance day is intentional: it must be absent on both surfaces,
    not become a zero-valued impact row. The transitions within the other days
    also make a one-sided bucket-grid change visible in the jog-minute sum.
    """
    samples = [
        ("2026-07-01T12:00:00Z", "2026-07-01", 0.0333, None),
        ("2026-07-01T12:00:20Z", "2026-07-01", 0.0333, None),
        ("2026-07-01T12:00:40Z", "2026-07-01", 0.0125, None),
        ("2026-07-02T12:00:00Z", "2026-07-02", 0.01905, 135.0),
        ("2026-07-02T12:00:20Z", "2026-07-02", 0.01905, 105.0),
        ("2026-07-02T12:00:40Z", "2026-07-02", 0.00556, None),
        ("2026-07-03T12:00:00Z", "2026-07-03", 0.0, None),
    ]
    for start, local_date, miles, hr in samples:
        _add_distance(conn, start, local_date, miles)
        if hr is not None:
            _add_hr(conn, start, local_date, hr)
    conn.commit()

    buckets = mx.bucket_series(conn, "2026-07-01T00:00:00Z",
                               "2026-07-04T00:00:00Z")
    daily = A.impact_volume(conn, "2026-07-01", "2026-07-03", by="day")

    assert {row["local_date"] for row in buckets} == {"2026-07-01", "2026-07-02"}
    assert all(row["miles"] > 0 for row in buckets)
    assert {row["period_start"] for row in daily} == {"2026-07-01", "2026-07-02"}
    assert sum(row["is_jog"] for row in buckets) * mx.IMPACT_BUCKET_SECONDS / 60.0 == pytest.approx(
        sum(row["jog_minutes"] for row in daily), abs=0.05
    )


def test_classification_matches_impact_volume_at_cadence_and_pace_boundaries(conn):
    bucket_min = mx.IMPACT_BUCKET_SECONDS / 60.0
    cases = [
        ("2026-07-01", 15.0, mx.IMPACT_JOG_CADENCE_MIN, True, False),
        ("2026-07-02", 15.0, nextafter(mx.IMPACT_JOG_CADENCE_MIN, 0.0), False, True),
        ("2026-07-03", 15.0, nextafter(mx.IMPACT_JOG_CADENCE_MIN, float("inf")), True, False),
        ("2026-07-04", mx.IMPACT_IMPLAUSIBLE_PACE_MIN, mx.IMPACT_JOG_CADENCE_MIN, True, False),
        ("2026-07-05", nextafter(mx.IMPACT_IMPLAUSIBLE_PACE_MIN, 0.0), mx.IMPACT_JOG_CADENCE_MIN, False, False),
        ("2026-07-06", 22.0, 0.0, False, True),
        ("2026-07-07", 45.0, mx.IMPACT_JOG_CADENCE_MIN, True, False),
    ]
    for local_date, pace, cadence, expected_jog, expected_walk in cases:
        start = f"{local_date}T12:00:00Z"
        _add_distance(conn, start, local_date, bucket_min / pace, cadence)
    conn.commit()

    buckets = {
        row["local_date"]: row
        for row in mx.bucket_series(conn, "2026-07-01T00:00:00Z", "2026-07-13T00:00:00Z")
    }
    impact = {
        row["period_start"]: row
        for row in A.impact_volume(conn, "2026-07-01", "2026-07-12", by="day")
    }
    for local_date, _, _, expected_jog, expected_walk in cases:
        bucket = buckets[local_date]
        assert (bucket["is_jog"], bucket["is_walk"]) == (expected_jog, expected_walk)
        assert impact[local_date]["jog_minutes"] == pytest.approx(
            (bucket_min if expected_jog else 0.0), abs=0.05
        )
        assert impact[local_date]["walk_minutes"] == pytest.approx(
            (bucket_min if expected_walk else 0.0), abs=0.05
        )


def test_bucket_spanning_local_midnight_matches_impact_volume(conn):
    # The two samples share one UTC bucket but carry different local dates.
    _add_distance(conn, "2026-07-01T23:59:50Z", "2026-07-01", 0.0125)
    _add_distance(conn, "2026-07-01T23:59:55Z", "2026-07-02", 0.0125)
    conn.commit()

    buckets = mx.bucket_series(conn, "2026-07-01T23:59:40Z", "2026-07-02T00:00:20Z")
    assert [row["local_date"] for row in buckets] == ["2026-07-01", "2026-07-02"]
    assert [row["bucket_start_utc"] for row in buckets] == sorted(
        row["bucket_start_utc"] for row in buckets
    )
    assert all(row["is_jog"] is False and row["is_walk"] is True for row in buckets)

    impact = {
        row["period_start"]: row
        for row in A.impact_volume(conn, "2026-07-01", "2026-07-02", by="day")
    }
    for local_date in ("2026-07-01", "2026-07-02"):
        assert impact[local_date]["jog_minutes"] == pytest.approx(0.0)
        assert impact[local_date]["walk_minutes"] == pytest.approx(
            round(mx.IMPACT_BUCKET_SECONDS / 60.0, 1)
        )
