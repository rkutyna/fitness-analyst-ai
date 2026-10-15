"""One jog rule for every use of the word "jogging".

A 20-second bucket inside a workout window is a jog bucket iff

    pace >= 5.0 min/mi (plausibility floor, when pace is known)
    AND ( 140 <= cadence <= 250
          OR (130 <= cadence <= 250 AND hr >= 130) )

Jog minutes count those buckets. The longest-block dial chains the very same
buckets, bridging a gap of at most two buckets only when every gap bucket is
still moving at a running heart rate:

    hr >= 130 AND 5.0 <= pace <= 18.0

There is no pace lane any more: pace decides nothing except the floor and,
inside a bridge, "still moving". These tests are written one lane at a time so
that a regression names the lane it broke.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from health_advisor import analysis as A
from health_advisor import db
from health_advisor import metrics as mx

T0 = "2026-07-01T12:00:00Z"
BUCKET_MIN = mx.IMPACT_BUCKET_SECONDS / 60.0


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    yield c
    c.close()


def _iso(offset_buckets: int, extra_seconds: int = 0) -> str:
    t = datetime.fromisoformat(T0.replace("Z", "+00:00"))
    return (t + timedelta(seconds=mx.IMPACT_BUCKET_SECONDS * offset_buckets
                          + extra_seconds)
            ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(conn, metric, start, value, unit):
    conn.execute(
        "INSERT INTO records (metric, start_utc, end_utc, local_date, value, "
        "unit, source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, 'test', ?)",
        (metric, start, start, start[:10], value, unit,
         f"{metric}|{start}|{value}"))


def _session(conn, specs, *, workout=True, workout_type="running"):
    """One workout window of consecutive 20 s buckets.

    Each spec is a dict with ``pace`` (min/mi), ``cadence`` (spm) and ``hr``
    (bpm), any of which may be omitted, or ``None`` for a bucket with no
    distance sample at all (a hole in time). A bucket with a distance sample
    and no ``pace`` key is not producible; give a pace.
    """
    for i, spec in enumerate(specs):
        if spec is None:
            continue
        start = _iso(i)
        _record(conn, "distance_walking_running", start,
                BUCKET_MIN / spec["pace"], "mi")
        if spec.get("cadence") is not None:
            _record(conn, "step_count", start, spec["cadence"] / 3.0, "count")
        if spec.get("hr") is not None:
            _record(conn, "heart_rate", _iso(i, 5), spec["hr"],
                    "count/min")
    if workout:
        conn.execute(
            "INSERT INTO workouts (workout_type, start_utc, end_utc, "
            "local_date, duration_min, source, dedupe_key) "
            "VALUES (?, ?, ?, ?, ?, 'test', ?)",
            (workout_type, T0, _iso(len(specs)), T0[:10],
             len(specs) * BUCKET_MIN, f"w|{T0}|{workout_type}"))
    conn.commit()
    return T0, _iso(len(specs))


def _series(conn, specs, **kw):
    start, end = _session(conn, specs, **kw)
    return start, end, mx.bucket_series(conn, start, end)


def _jog_minutes(conn, start, end):
    return sum(r["is_jog"] for r in mx.bucket_series(conn, start, end)) * BUCKET_MIN


# --- the predicate, lane by lane ------------------------------------------

def _jog(**kw):
    base = dict(in_workout=True, pace_min_per_mi=10.0, cadence_spm=None, hr=None)
    base.update(kw)
    return mx.is_jog_bucket(**base)


def test_incline_walk_is_not_jog():
    # Pace 14 and HR 140 would have made this a jog under the old pace lane;
    # cadence 110 says walking and cadence is the only thing that can say jog.
    assert _jog(pace_min_per_mi=14.0, cadence_spm=110.0, hr=140.0) is False


def test_cadence_140_is_jog_with_or_without_heart_rate():
    assert _jog(cadence_spm=140.0, hr=None) is True
    assert _jog(cadence_spm=140.0, hr=95.0) is True


def test_cadence_135_at_hr_140_is_jog():
    assert _jog(pace_min_per_mi=14.0, cadence_spm=135.0, hr=140.0) is True


def test_cadence_135_needs_a_known_running_heart_rate():
    assert _jog(cadence_spm=135.0, hr=None) is False
    assert _jog(cadence_spm=135.0, hr=129.9) is False
    assert _jog(cadence_spm=135.0, hr=130.0) is True


def test_effort_lane_cadence_floor_is_130():
    assert _jog(cadence_spm=130.0, hr=150.0) is True
    assert _jog(cadence_spm=129.9, hr=150.0) is False


def test_cadence_glitch_ceiling_applies_to_both_lanes():
    assert _jog(cadence_spm=250.0, hr=None) is True
    assert _jog(cadence_spm=250.1, hr=None) is False
    assert _jog(cadence_spm=250.1, hr=150.0) is False
    assert _jog(cadence_spm=2961.0, hr=150.0) is False


def test_pace_alone_is_never_jog():
    # The pace lane (5-16 min/mi unless HR < 130) is gone.
    assert _jog(pace_min_per_mi=15.0, cadence_spm=120.0, hr=140.0) is False
    assert _jog(pace_min_per_mi=9.0, cadence_spm=None, hr=None) is False
    assert _jog(pace_min_per_mi=9.0, cadence_spm=None, hr=150.0) is False


def test_pace_floor_vetoes_both_lanes_when_pace_is_known():
    assert _jog(pace_min_per_mi=mx.IMPACT_IMPLAUSIBLE_PACE_MIN,
                cadence_spm=150.0) is True
    assert _jog(pace_min_per_mi=4.9, cadence_spm=150.0) is False
    assert _jog(pace_min_per_mi=4.9, cadence_spm=135.0, hr=150.0) is False
    # Unknown pace is not evidence against: the floor is "when pace is known".
    assert _jog(pace_min_per_mi=None, cadence_spm=150.0) is True


def test_outside_a_workout_nothing_is_jog():
    assert _jog(in_workout=False, cadence_spm=150.0, hr=150.0) is False
    assert _jog(in_workout=False, cadence_spm=135.0, hr=150.0) is False


# --- the bridge -----------------------------------------------------------

def _bridge(**kw):
    base = dict(pace_min_per_mi=17.0, hr=140.0)
    base.update(kw)
    return mx.is_bridge_bucket(base)


def test_bridge_needs_running_hr_and_a_moving_pace():
    assert _bridge() is True
    assert _bridge(pace_min_per_mi=5.0) is True
    assert _bridge(pace_min_per_mi=18.0) is True
    assert _bridge(pace_min_per_mi=18.1) is False
    assert _bridge(pace_min_per_mi=4.9) is False
    assert _bridge(hr=129.9) is False
    assert _bridge(hr=None) is False


def test_bridge_covers_the_pace_the_old_pace_lane_used_to_cover():
    # The bridge now spans 5-18, not 16-18: slower-than-jog pace down to the
    # floor is "still moving" too.
    assert _bridge(pace_min_per_mi=12.0, hr=140.0) is True


def test_standstill_bucket_does_not_bridge():
    # Standing still keeps HR above 130 but moves nothing: no pace, no bridge.
    assert _bridge(pace_min_per_mi=None, hr=140.0) is False


# --- through the real pipeline: records -> buckets -> jog minutes -> block ---

def test_incline_walk_session_has_no_jog_minutes_and_no_block(conn):
    walk = {"pace": 14.0, "cadence": 110.0, "hr": 140.0}
    start, end, buckets = _series(conn, [walk] * 12)
    assert len(buckets) == 12
    assert not any(b["is_jog"] for b in buckets)
    assert A.impact_volume(conn, "2026-07-01", "2026-07-01", by="day"
                           )[0]["jog_minutes"] == 0.0
    block = A.longest_block(conn, start, end)
    assert block["bridged_min"] == 0.0
    assert block["unbridged_min"] == 0.0


def test_cadence_135_at_hr_140_session_is_jog_and_one_block(conn):
    tired = {"pace": 14.0, "cadence": 135.0, "hr": 140.0}
    start, end, buckets = _series(conn, [tired] * 12)
    assert all(b["is_jog"] for b in buckets)
    day = A.impact_volume(conn, "2026-07-01", "2026-07-01", by="day")[0]
    assert day["jog_minutes"] == pytest.approx(12 * BUCKET_MIN, abs=0.05)
    assert A.longest_block(conn, start, end)["bridged_min"] == pytest.approx(
        12 * BUCKET_MIN, abs=0.05)


def test_pace_15_at_cadence_120_and_hr_140_is_not_a_block_bucket(conn):
    # The old pace lane (5-16 unless HR < 130) made this a block bucket.
    slow = {"pace": 15.0, "cadence": 120.0, "hr": 140.0}
    start, end, buckets = _series(conn, [slow] * 12)
    assert not any(b["is_jog"] for b in buckets)
    assert not any(mx.is_block_bucket(b) for b in buckets)
    assert A.longest_block(conn, start, end)["bridged_min"] == 0.0


def test_pace_without_cadence_is_not_a_block_bucket(conn):
    # No step data at all: pace and HR cannot make a jog.
    start, end, buckets = _series(conn, [{"pace": 9.0, "hr": 150.0}] * 12)
    assert not any(b["is_jog"] for b in buckets)
    assert A.longest_block(conn, start, end)["bridged_min"] == 0.0


def test_standstill_gap_does_not_bridge(conn):
    # HR 140 through the gap but no distance: the buckets do not exist, so
    # time is not contiguous and nothing may be bridged across it.
    jog = {"pace": 9.0, "cadence": 160.0, "hr": 150.0}
    start, end, _ = _series(conn, [jog] * 6 + [None, None] + [jog] * 6)
    block = A.longest_block(conn, start, end)
    assert block["bridged_min"] == pytest.approx(6 * BUCKET_MIN, abs=0.05)
    assert block["reps"][0]["bridged"] is False


def test_standstill_bucket_without_pace_does_not_bridge():
    jog = {"bucket_start_utc": _iso(0), "is_jog": True, "pace_min_per_mi": 9.0,
           "hr": 150.0, "cadence_spm": 160.0}
    standing = {"bucket_start_utc": _iso(1), "is_jog": False,
                "pace_min_per_mi": None, "hr": 140.0, "cadence_spm": 0.0}
    after = {**jog, "bucket_start_utc": _iso(2)}
    block = A.longest_block_from_buckets([jog, standing, after])
    assert block["bridged_min"] == pytest.approx(BUCKET_MIN, abs=0.05)


def test_two_bucket_gap_at_pace_17_and_hr_140_bridges(conn):
    jog = {"pace": 9.0, "cadence": 160.0, "hr": 150.0}
    gap = {"pace": 17.0, "cadence": 120.0, "hr": 140.0}
    start, end, _ = _series(conn, [jog] * 6 + [gap] * 2 + [jog] * 6)
    block = A.longest_block(conn, start, end)
    assert block["bridged_min"] == pytest.approx(14 * BUCKET_MIN, abs=0.05)
    assert block["unbridged_min"] == pytest.approx(6 * BUCKET_MIN, abs=0.05)
    assert block["reps"][0]["bridged"] is True


def test_a_hole_in_time_beside_a_moving_gap_bucket_does_not_bridge(conn):
    # One still-moving bucket, then 20 s with no data at all, then the run
    # resumes: the bucket indices are not contiguous, so nothing is bridged.
    jog = {"pace": 9.0, "cadence": 160.0, "hr": 150.0}
    gap = {"pace": 17.0, "cadence": 120.0, "hr": 140.0}
    start, end, _ = _series(conn, [jog] * 6 + [gap] + [None] + [jog] * 6)
    block = A.longest_block(conn, start, end)
    assert block["bridged_min"] == pytest.approx(6 * BUCKET_MIN, abs=0.05)
    assert block["reps"][0]["bridged"] is False


def test_three_bucket_gap_does_not_bridge(conn):
    jog = {"pace": 9.0, "cadence": 160.0, "hr": 150.0}
    gap = {"pace": 17.0, "cadence": 120.0, "hr": 140.0}
    start, end, _ = _series(conn, [jog] * 6 + [gap] * 3 + [jog] * 6)
    assert A.longest_block(conn, start, end)["bridged_min"] == pytest.approx(
        6 * BUCKET_MIN, abs=0.05)


@pytest.mark.parametrize("gap", [
    {"pace": 19.0, "cadence": 100.0, "hr": 140.0},   # slower than the ceiling
    {"pace": 14.0, "cadence": 100.0, "hr": 120.0},   # moving, but not running HR
    {"pace": 14.0, "cadence": 100.0, "hr": None},    # no HR, no confirmation
])
def test_gap_at_walking_pace_or_low_hr_does_not_bridge(conn, gap):
    jog = {"pace": 9.0, "cadence": 160.0, "hr": 150.0}
    start, end, _ = _series(conn, [jog] * 6 + [gap] * 2 + [jog] * 6)
    assert A.longest_block(conn, start, end)["bridged_min"] == pytest.approx(
        6 * BUCKET_MIN, abs=0.05)


def test_gap_pace_16_to_18_bridge_and_5_to_16_bridge_alike(conn):
    # The bridge range is one range now. A gap at pace 12 and HR 140 bridges
    # exactly as one at pace 17 does.
    jog = {"pace": 9.0, "cadence": 160.0, "hr": 150.0}
    gap = {"pace": 12.0, "cadence": 100.0, "hr": 140.0}
    start, end, _ = _series(conn, [jog] * 6 + [gap] * 2 + [jog] * 6)
    block = A.longest_block(conn, start, end)
    assert block["bridged_min"] == pytest.approx(14 * BUCKET_MIN, abs=0.05)
    # Bridged, not jogged: the old pace lane counted these two buckets as jog
    # and so reported an unbridged 14-bucket run.
    assert block["unbridged_min"] == pytest.approx(6 * BUCKET_MIN, abs=0.05)
    assert block["reps"][0]["bridged"] is True


# --- one rule everywhere -------------------------------------------------

def test_sql_rows_and_the_exported_predicate_agree_on_a_mixed_fixture(conn):
    cases = [
        {"pace": 9.0, "cadence": 160.0, "hr": 150.0},    # C
        {"pace": 9.0, "cadence": 140.0, "hr": None},     # C, no HR
        {"pace": 14.0, "cadence": 135.0, "hr": 140.0},   # C130
        {"pace": 14.0, "cadence": 135.0, "hr": 120.0},   # below the HR line
        {"pace": 14.0, "cadence": 110.0, "hr": 140.0},   # incline walk
        {"pace": 15.0, "cadence": 120.0, "hr": 140.0},   # old pace lane
        {"pace": 9.0, "hr": 150.0},                      # no step data
        {"pace": 4.5, "cadence": 150.0, "hr": 150.0},    # under the floor
        {"pace": 9.0, "cadence": 2961.0, "hr": 150.0},   # glitch cadence
        {"pace": 14.0, "cadence": 130.0, "hr": 130.0},   # both edges
    ]
    start, end = _session(conn, cases)
    rows = mx.impact_bucket_rows(
        conn, "start_utc >= ? AND start_utc < ?", (start, end),
        arbitration_window=(start, end), arbitration_window_kind="utc")
    assert len(rows) == len(cases)
    got = []
    for row in rows:
        pace = BUCKET_MIN / row["mi"]
        expect = mx.is_jog_bucket(
            in_workout=bool(row["in_workout"]), pace_min_per_mi=pace,
            cadence_spm=row["cadence_spm"], hr=row["hr"])
        assert bool(row["is_jog"]) is expect
        got.append(bool(row["is_jog"]))
    assert got == [True, True, True, False, False, False, False, False, False,
                   True]
    # The same buckets as bucket_series, impact_volume and the block dial see
    # them: one answer.
    series = mx.bucket_series(conn, start, end)
    assert [b["is_jog"] for b in series] == got
    day = A.impact_volume(conn, "2026-07-01", "2026-07-01", by="day")[0]
    assert day["jog_minutes"] == pytest.approx(sum(got) * BUCKET_MIN, abs=0.05)


def test_a_jog_bucket_outside_a_workout_window_is_not_jog(conn):
    start, end, buckets = _series(
        conn, [{"pace": 9.0, "cadence": 160.0, "hr": 150.0}] * 6,
        workout=False)
    assert len(buckets) == 6
    assert not any(b["is_jog"] for b in buckets)


def test_a_pace_14_cadence_110_bucket_is_walking_not_jogging(conn):
    _, _, buckets = _series(conn, [{"pace": 14.0, "cadence": 110.0,
                                    "hr": 140.0}] * 3)
    assert all((not b["is_jog"]) and b["is_walk"] for b in buckets)


def test_an_effort_lane_bucket_is_jogging_so_it_is_no_longer_walking(conn):
    _, _, buckets = _series(conn, [{"pace": 14.0, "cadence": 135.0,
                                    "hr": 140.0}] * 3)
    assert all(b["is_jog"] and not b["is_walk"] for b in buckets)


def test_derived_formula_version_moved_so_stored_figures_re_derive():
    assert mx.DERIVED_FORMULA_VERSION == 3


def test_the_retired_pace_lane_constants_are_gone():
    for name in ("IMPACT_JOG_PACE_MAX",):
        assert not hasattr(mx, name), name
    assert not hasattr(A, "_is_jog_bucket")
    assert not hasattr(A, "_is_bridge_bucket")
