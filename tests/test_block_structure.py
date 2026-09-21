"""The three-signal jog predicate for block structure, and the cadence glitch
ceiling that also protects the impact-volume dial.

Rule (analysis._is_jog_bucket): a bucket counts as jogging if ANY of:
  1. canonical is_jog AND cadence plausible (present, <= 250.0);
  2. jog pace (IMPACT_IMPLAUSIBLE_PACE_MIN..IMPACT_JOG_PACE_MAX) unless the
     heart rate is KNOWN and below IMPACT_JOG_HR_MIN — a missing HR never vetoes;
  3. cadence plausible AND >= BLOCK_EFFORT_CADENCE_MIN AND known HR
     >= IMPACT_JOG_HR_MIN.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from health_advisor import analysis
from health_advisor import metrics


def _bucket(**kw):
    out = {
        "bucket_start_utc": "2001-07-01T12:00:00Z",
        "local_date": "2001-07-01",
        "pace_min_per_mi": 10.0,
        "hr": None,
        "cadence_spm": None,
        "is_jog": False,
    }
    out.update(kw)
    return out


def _buckets(*specs):
    """Consecutive 20-second buckets starting at 2001-07-01T12:00:00Z."""
    start = datetime.fromisoformat("2001-07-01T12:00:00+00:00")
    out = []
    for i, spec in enumerate(specs):
        b = _bucket(**spec)
        b["bucket_start_utc"] = (
            start + timedelta(seconds=20 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append(b)
    return out


def test_cadence_140_counts_as_running_gait():
    # Pace absent so only the running-gait lane can possibly fire.
    b = _bucket(cadence_spm=140.0, is_jog=True, pace_min_per_mi=None)
    assert analysis._is_jog_bucket(b) is True


def test_incline_walk_does_not_count():
    """A slow jog pace and running-effort HR, but cadence ~105 says walking.

    Pace sits above the jog-pace lane, so only the effort lane could count
    it — and its cadence floor is why it does not.
    """
    b = _bucket(pace_min_per_mi=17.0, hr=150.0, cadence_spm=105.0)
    assert analysis._is_jog_bucket(b) is False


def test_fatigued_bucket_counts_on_effort():
    """Just under the gait line at an obvious running HR, pace slower than 16."""
    b = _bucket(pace_min_per_mi=18.0, hr=150.0, cadence_spm=135.0)
    assert analysis._is_jog_bucket(b) is True


def test_fast_pace_with_known_walking_hr_does_not_count():
    """The HR veto: a treadmill pace at an obvious walking heart rate."""
    b = _bucket(pace_min_per_mi=9.0, hr=100.0)
    assert analysis._is_jog_bucket(b) is False


def test_fast_pace_with_missing_hr_counts():
    """A missing HR never vetoes — absence of a reading is not evidence."""
    b = _bucket(pace_min_per_mi=9.0, hr=None)
    assert analysis._is_jog_bucket(b) is True


def test_glitch_cadence_does_not_count():
    """2961 spm is a mis-scaled sample; the plausible ceiling refuses it even
    with canonical is_jog, a running HR, and a pace outside the jog lane."""
    b = _bucket(cadence_spm=2961.0, hr=135.0, is_jog=True,
                pace_min_per_mi=18.0)
    assert analysis._is_jog_bucket(b) is False


def test_glitch_cadence_at_exactly_the_ceiling_counts():
    """The boundary: <= IMPACT_CADENCE_PLAUSIBLE_MAX is plausible."""
    b = _bucket(cadence_spm=metrics.IMPACT_CADENCE_PLAUSIBLE_MAX,
                hr=135.0, is_jog=True, pace_min_per_mi=18.0)
    assert analysis._is_jog_bucket(b) is True


def _workout(conn, start: str, seconds: int):
    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end = (t0 + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = t0.date().isoformat()
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES (?, ?, ?, ?, ?, 'test', ?)",
        ("walking", start, end, day, seconds / 60.0, f"w|{start}|{end}"),
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


def test_impact_bucket_rows_does_not_set_is_jog_for_glitch_cadence(conn):
    """The canonical is_jog inherits the plausible ceiling: a four-figure step
    sample cannot set impact-volume jogging."""
    _workout(conn, "2001-07-01T12:00:00Z", 20)
    _record(conn, "distance_walking_running", "2001-07-01T12:00:00Z", 0.03)
    _record(conn, "step_count", "2001-07-01T12:00:00Z", 2961.0 / 3.0)
    conn.commit()

    rows = metrics.impact_bucket_rows(
        conn, "start_utc >= ? AND start_utc < ?",
        ("2001-07-01T12:00:00Z", "2001-07-01T12:01:00Z"),
        arbitration_window=("2001-07-01T12:00:00Z", "2001-07-01T12:01:00Z"),
        arbitration_window_kind="utc")

    assert len(rows) == 1
    assert rows[0]["cadence_spm"] == pytest.approx(2961.0)
    assert rows[0]["is_jog"] == 0


def test_bridge_spans_a_gap_of_at_most_two_buckets():
    """Two contiguous 16-18 min/mi buckets with HR >= 130 bridge the jog
    segments; the merged block is jog + gap + jog = BLOCK_BRIDGE_MAX_BUCKETS + 2
    buckets long."""
    buckets = _buckets(
        {"pace_min_per_mi": 9.0, "is_jog": True, "cadence_spm": 150.0},
        {"pace_min_per_mi": 17.0, "hr": 140.0},
        {"pace_min_per_mi": 17.0, "hr": 140.0},
        {"pace_min_per_mi": 9.0, "is_jog": True, "cadence_spm": 150.0},
    )
    block = analysis.longest_block_from_buckets(buckets)
    expected_buckets = metrics.BLOCK_BRIDGE_MAX_BUCKETS + 2
    assert block["bridged_min"] == metrics.r(
        expected_buckets * metrics.IMPACT_BUCKET_SECONDS / 60.0, 1)
    assert block["reps"][0]["bridged"] is True


def test_bridge_refuses_a_gap_longer_than_two_buckets():
    bins = metrics.BLOCK_BRIDGE_MAX_BUCKETS + 1
    buckets = _buckets(
        {"pace_min_per_mi": 9.0, "is_jog": True, "cadence_spm": 150.0},
        *({"pace_min_per_mi": 17.0, "hr": 140.0} for _ in range(bins)),
        {"pace_min_per_mi": 9.0, "is_jog": True, "cadence_spm": 150.0},
    )
    block = analysis.longest_block_from_buckets(buckets)
    assert block["bridged_min"] == metrics.r(
        metrics.IMPACT_BUCKET_SECONDS / 60.0, 1)
