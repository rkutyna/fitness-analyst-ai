"""running_form — per-workout measures over jog buckets only.

Read spec section 11.2 before changing any name in this module. These are
deliberately NOT called aerobic decoupling: that construct is defined on
prolonged steady-state efforts and its <5% threshold is a coaching heuristic
from that setting. These runs are 35-50 minute run/walk intervals.
"""
from __future__ import annotations

import math
import itertools

import pytest

from health_advisor import running_form as rf


_BUCKETS = itertools.count()


def _bucket(is_jog, speed_mph, hr, is_walk=None, start=None):
    """A bucket_series row, only the fields running_form reads."""
    if is_walk is None:
        is_walk = not is_jog
    if start is None:
        n = next(_BUCKETS)
        start = f"2026-07-01T12:{n // 3:02d}:{(n % 3) * 20:02d}Z"
    return {"bucket_start_utc": start, "local_date": "2026-07-01",
            "miles": speed_mph / 180.0, "hr": hr, "speed_mph": speed_mph,
            "pace_min_per_mi": (60.0 / speed_mph) if speed_mph else None,
            "is_jog": is_jog, "is_walk": is_walk}


def test_midnight_spanning_bucket_is_counted_once():
    buckets = [_bucket(True, 4.5, 140.0) for _ in range(29)]
    midnight = _bucket(True, 4.5, 140.0, start="2026-07-01T23:59:40Z")
    midnight["local_date"] = "2026-07-01"
    other_half = _bucket(True, 4.5, 142.0, start="2026-07-01T23:59:40Z")
    other_half["local_date"] = "2026-07-02"
    buckets.extend((midnight, other_half))
    out = rf.efficiency_change_from_buckets(buckets)
    assert out["status"] == "ok"
    assert out["jog_minutes"] == pytest.approx(10.0)


def test_flat_effort_has_near_zero_change():
    # 40 jog buckets (13.3 min), identical speed and HR throughout.
    buckets = [_bucket(True, 4.5, 140.0) for _ in range(40)]
    out = rf.efficiency_change_from_buckets(buckets)
    assert out["status"] == "ok"
    assert out["change_pct"] == pytest.approx(0.0, abs=0.01)


def test_hr_rising_at_constant_speed_is_a_negative_change():
    """Efficiency is speed/HR. Same speed, higher HR later => efficiency FELL,
    so change_pct is negative. If this test passes with a positive number the
    implementation has inverted the ratio — see spec 11.4."""
    first = [_bucket(True, 4.5, 140.0) for _ in range(20)]
    second = [_bucket(True, 4.5, 154.0) for _ in range(20)]
    out = rf.efficiency_change_from_buckets(first + second)
    assert out["status"] == "ok"
    assert out["change_pct"] < 0
    # 140 -> 154 is +10% HR; efficiency falls by 1 - 140/154 = 9.09%
    assert out["change_pct"] == pytest.approx(-9.09, abs=0.1)


def test_slowing_at_constant_hr_is_also_a_negative_change():
    first = [_bucket(True, 5.0, 140.0) for _ in range(20)]
    second = [_bucket(True, 4.5, 140.0) for _ in range(20)]
    out = rf.efficiency_change_from_buckets(first + second)
    assert out["change_pct"] == pytest.approx(-10.0, abs=0.1)


def test_walk_buckets_are_excluded_from_both_halves():
    """Walks between jogs must not shift the split. The split is on cumulative
    JOG time; a pile of walk buckets in the middle changes nothing."""
    jog = [_bucket(True, 4.5, 140.0) for _ in range(20)]
    slow_jog = [_bucket(True, 4.5, 154.0) for _ in range(20)]
    walks = [_bucket(False, 2.0, 100.0) for _ in range(30)]
    without = rf.efficiency_change_from_buckets(jog + slow_jog)
    with_walks = rf.efficiency_change_from_buckets(jog + walks + slow_jog)
    assert with_walks["change_pct"] == pytest.approx(without["change_pct"], abs=0.01)


def test_buckets_without_hr_are_skipped_not_zeroed():
    """A missing HR is unknown. Treating it as 0 would make efficiency infinite."""
    buckets = ([_bucket(True, 4.5, 140.0) for _ in range(20)]
               + [_bucket(True, 4.5, None) for _ in range(5)]
               + [_bucket(True, 4.5, 140.0) for _ in range(20)])
    out = rf.efficiency_change_from_buckets(buckets)
    assert out["status"] == "ok"
    assert out["first_half_buckets"] + out["second_half_buckets"] == 40


def test_gps_spike_buckets_are_excluded():
    first = [_bucket(True, 4.5, 140.0) for _ in range(20)]
    second = [_bucket(True, 4.5, 154.0) for _ in range(20)]
    normal = first + second
    spikes = [_bucket(False, speed, 108.0, is_walk=False)
              for speed in (20.0, 21.0, 24.0, 26.0)]

    without_spikes = rf.efficiency_change_from_buckets(normal)
    with_spikes = rf.efficiency_change_from_buckets(first + spikes + second)

    assert with_spikes["change_pct"] == pytest.approx(without_spikes["change_pct"])
    assert with_spikes["buckets_dropped_implausible"] == len(spikes)


def test_plausibility_floor_is_inclusive_at_the_boundary():
    boundary = _bucket(True, 60.0 / 5.0, 108.0)
    just_faster = _bucket(False, 60.0 / math.nextafter(5.0, 0.0), 108.0,
                          is_walk=False)

    out = rf.efficiency_change_from_buckets(
        [_bucket(True, 60.0 / 5.0, 108.0) for _ in range(30)] + [just_faster])

    assert out["jog_minutes"] == pytest.approx(10.0)
    assert out["buckets_dropped_implausible"] == 1


def test_short_session_refuses():
    buckets = [_bucket(True, 4.5, 140.0) for _ in range(20)]  # 6.7 min
    out = rf.efficiency_change_from_buckets(buckets)
    assert out["status"] == "insufficient_jog_time"
    assert "change_pct" not in out


def test_lopsided_halves_refuse():
    """Enough total jog time, but one half has too few HR-bearing buckets.

    32 jog buckets = 10.7 min, clearing MIN_JOG_MINUTES. The split is at 16, and
    the second half carries only 4 buckets with HR — one under MIN_HALF_BUCKETS.
    The HR-less buckets must sit at the START of the second half, or the split
    lands somewhere that leaves both halves covered.
    """
    buckets = ([_bucket(True, 4.5, 140.0) for _ in range(16)]
               + [_bucket(True, 4.5, None) for _ in range(12)]
               + [_bucket(True, 4.5, 140.0) for _ in range(4)])
    out = rf.efficiency_change_from_buckets(buckets)
    assert out["status"] == "insufficient_half_coverage"


def test_no_buckets_refuses_without_raising():
    out = rf.efficiency_change_from_buckets([])
    assert out["status"] == "insufficient_jog_time"
    assert out["jog_minutes"] == 0.0


def test_walk_structure_counts_bouts_not_buckets():
    """Three consecutive walk buckets are ONE bout, not three."""
    b = ([_bucket(True, 4.5, 140.0) for _ in range(20)]
         + [_bucket(False, 2.0, 100.0) for _ in range(3)]
         + [_bucket(True, 4.5, 140.0) for _ in range(20)]
         + [_bucket(False, 2.0, 100.0) for _ in range(3)]
         + [_bucket(True, 4.5, 140.0) for _ in range(5)])
    out = rf.walk_structure_from_buckets(b)
    assert out["status"] == "ok"
    assert out["walk_bouts"] == 2
    assert out["mean_bout_minutes"] == pytest.approx(1.0, abs=0.01)


def test_walk_structure_ignores_implausible_buckets():
    normal = ([_bucket(True, 4.5, 140.0) for _ in range(20)]
              + [_bucket(False, 2.0, 100.0) for _ in range(3)]
              + [_bucket(True, 4.5, 140.0) for _ in range(20)])
    spike = _bucket(False, 24.0, 108.0, is_walk=False)

    without_spike = rf.walk_structure_from_buckets(normal)
    with_spike = rf.walk_structure_from_buckets(normal[:20] + [spike] + normal[20:])

    assert with_spike["walk_fraction"] == without_spike["walk_fraction"]


def test_walk_structure_late_loading_shows_in_the_half_split():
    """All the walking in the second half is the signal this exists to carry."""
    b = ([_bucket(True, 4.5, 140.0) for _ in range(20)]
         + [_bucket(True, 4.5, 140.0) for _ in range(20)]
         + [_bucket(False, 2.0, 100.0) for _ in range(20)])
    out = rf.walk_structure_from_buckets(b)
    assert out["first_half_walk_fraction"] == pytest.approx(0.0, abs=0.01)
    assert out["second_half_walk_fraction"] > 0.4


def test_walk_structure_refuses_on_a_short_session():
    out = rf.walk_structure_from_buckets([_bucket(True, 4.5, 140.0) for _ in range(10)])
    assert out["status"] == "insufficient_jog_time"


def test_walk_structure_with_no_walking_reports_zero_not_refusal():
    b = [_bucket(True, 4.5, 140.0) for _ in range(40)]
    out = rf.walk_structure_from_buckets(b)
    assert out["status"] == "ok"
    assert out["walk_bouts"] == 0
    assert out["walk_fraction"] == 0.0


def test_reference_band_excludes_out_of_band_buckets():
    """Efficiency across mixed paces mostly reflects the pace mix. Banding is
    what makes week-to-week comparison mean anything."""
    in_band = _bucket(True, 60.0 / 14.0, 140.0)     # 14 min/mi — inside 13-15
    too_fast = _bucket(True, 60.0 / 11.0, 165.0)    # 11 min/mi — outside
    too_slow = _bucket(True, 60.0 / 17.0, 120.0)    # 17 min/mi — outside
    kept = rf.in_reference_band([in_band, too_fast, too_slow])
    assert len(kept) == 1
    assert kept[0]["pace_min_per_mi"] == pytest.approx(14.0, abs=0.01)


def test_banded_weekly_refuses_under_three_weeks(tmp_path):
    from health_advisor import db
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    out = rf.banded_weekly(c, "2026-07-01", "2026-07-31")
    assert out["status"] == "insufficient_weeks"
    c.close()


def test_banded_weekly_computes_slope_over_synthetic_weeks():
    """Efficiency improving week over week must produce a positive slope."""
    weeks = [
        {"week_start": "2026-06-01", "efficiency": 0.0300, "mean_hr": 145.0, "buckets": 20},
        {"week_start": "2026-06-08", "efficiency": 0.0310, "mean_hr": 143.0, "buckets": 20},
        {"week_start": "2026-06-15", "efficiency": 0.0320, "mean_hr": 141.0, "buckets": 20},
        {"week_start": "2026-06-22", "efficiency": 0.0330, "mean_hr": 139.0, "buckets": 20},
    ]
    out = rf.trend_from_weeks(weeks)
    assert out["status"] == "ok"
    assert out["efficiency_slope_per_week"] > 0
    assert out["hr_slope_per_week"] < 0
    assert "descriptive" in out["caveat"].lower()


def _insert_running_sample(conn, metric, start_utc, local_date, value, suffix):
    conn.execute(
        "INSERT INTO records (metric, start_utc, end_utc, local_date, value, unit, "
        "source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, 'test', ?)",
        (metric, start_utc, start_utc, local_date, value,
         "mi" if metric == "distance_walking_running" else "count/min",
         f"{metric}|{start_utc}|{suffix}"),
    )


def _insert_nested_running_workouts(conn):
    from health_advisor import db
    from datetime import datetime, timedelta, timezone

    for week, local_date in enumerate(("2026-07-06", "2026-07-13", "2026-07-20", "2026-07-27", "2026-08-03")):
        outer_start = f"{local_date}T12:00:00Z"
        outer_end = f"{local_date}T12:20:00Z"
        inner_start = f"{local_date}T12:02:00Z"
        inner_end = f"{local_date}T12:12:00Z"
        db.insert_workouts(conn, [
            {"workout_type": "running", "start_utc": outer_start, "end_utc": outer_end,
             "local_date": local_date, "duration_min": 20.0, "distance_mi": 20.0 / 14.0,
             "energy_kcal": None, "unit_distance": "mi", "source": "test", "dedupe_key": f"outer-running-{week}"},
            {"workout_type": "running", "start_utc": inner_start, "end_utc": inner_end,
             "local_date": local_date, "duration_min": 10.0, "distance_mi": 10.0 / 14.0,
             "energy_kcal": None, "unit_distance": "mi", "source": "test", "dedupe_key": f"inner-running-{week}"},
        ])
        t0 = datetime.fromisoformat(outer_start.replace("Z", "+00:00"))
        for i in range(60):
            ts = (t0 + timedelta(seconds=20 * i)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _insert_running_sample(conn, "distance_walking_running", ts, local_date, 1.0 / 42.0, f"{week}-{i}")
            _insert_running_sample(conn, "heart_rate", (t0 + timedelta(seconds=20 * i + 5)).strftime("%Y-%m-%dT%H:%M:%SZ"), local_date, 140.0, f"{week}-{i}")
    conn.commit()


def test_banded_weekly_does_not_double_count_a_nested_workout(tmp_path):
    from health_advisor import db
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    _insert_nested_running_workouts(c)
    out = rf.banded_weekly(c, "2026-07-06", "2026-07-27")
    assert out["status"] == "ok"
    # The nested window shares all of its buckets with the outer one; the
    # week's direct bucket count is 60, not 60 + 30.
    assert out["weeks"][0]["buckets"] == 60
    c.close()


def test_personal_reference_refuses_under_five_sessions():
    out = rf.reference_from_changes([-3.0, -5.0, -4.0])
    assert out["status"] == "insufficient_sessions"
    assert out["n_sessions"] == 3


def test_personal_reference_reports_a_spread_not_a_threshold():
    changes = [-2.0, -4.0, -6.0, -3.0, -5.0, -8.0, -1.0]
    out = rf.reference_from_changes(changes)
    assert out["status"] == "ok"
    assert out["n_sessions"] == 7
    assert out["p10"] < out["median_change_pct"] < out["p90"]
    # No literature threshold may appear anywhere in the output.
    assert "threshold" not in out
    assert out["minimum_detectable_change_pct"] > 0


def test_minimum_detectable_change_shrinks_as_sessions_accumulate():
    tight_few = rf.reference_from_changes([-4.0, -4.2, -3.8, -4.1, -3.9])
    tight_many = rf.reference_from_changes([-4.0, -4.2, -3.8, -4.1, -3.9] * 4)
    assert (tight_many["minimum_detectable_change_pct"]
            < tight_few["minimum_detectable_change_pct"])


def test_personal_reference_skips_a_nested_workout(tmp_path):
    from health_advisor import db
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    _insert_nested_running_workouts(c)
    out = rf.personal_reference(c, "2026-07-06", "2026-08-09")
    assert out["status"] == "ok"
    assert out["n_sessions"] == 5
    c.close()


def test_get_run_form_rejects_a_bad_date(tools):
    out = tools.get_run_form(workout_date="not-a-date")
    assert "error" in out


def test_get_run_form_is_registered_and_excluded_from_researcher_tools():
    """get_run_form must NOT reach the deep-dive researcher: it is close enough
    to the week's prescription that the researcher could confirm what it was
    told. Same reasoning that keeps the plan tools out."""
    from health_advisor import llm
    assert "get_run_form" not in llm.RESEARCHER_TOOLS


def _insert_power_run(conn, local_date, start_hour, power, *, route_ref):
    from health_advisor import db
    from datetime import datetime, timedelta, timezone

    start = f"{local_date}T{start_hour:02d}:00:00Z"
    end = (datetime.fromisoformat(start.replace("Z", "+00:00"))
           + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.insert_workouts(conn, [{
        "workout_type": "running", "start_utc": start, "end_utc": end,
        "local_date": local_date, "duration_min": 10.0,
        "distance_mi": 10.0 / 14.0, "energy_kcal": None, "unit_distance": "mi",
        "source": "Demo's Apple Watch", "route_ref": route_ref,
        "dedupe_key": f"power-{local_date}-{start_hour}",
    }])
    for i in range(30):
        ts = (datetime.fromisoformat(start.replace("Z", "+00:00"))
              + timedelta(seconds=20 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        # 1/42 mi per 20-second bucket is exactly 14:00 min/mi.
        for metric, value, unit in (
            ("distance_walking_running", 1.0 / 42.0, "mi"),
            ("heart_rate", 140.0, "count/min"),
            # The production bucket predicate now needs cadence inside the
            # workout window; this remains a settled-state power fixture.
            ("step_count", 47.0, "count"),
            ("running_power", power if i >= 9 else 999.0, "W"),
        ):
            _insert_running_sample(conn, metric, ts, local_date, value,
                                   f"power-{local_date}-{start_hour}-{metric}-{i}")
    conn.commit()


def test_monthly_running_power_uses_matched_pace_after_three_minutes(conn):
    _insert_power_run(conn, "2026-07-14", 8, 200.0, route_ref="run-a.gpx")
    _insert_power_run(conn, "2026-07-21", 8, 220.0, route_ref="run-b.gpx")

    assert rf.monthly_running_power(conn, "2026-07") == pytest.approx(210.0)


def test_monthly_running_power_refuses_fewer_than_two_qualifying_outdoor_runs(conn):
    _insert_power_run(conn, "2026-07-14", 8, 200.0, route_ref="run-a.gpx")
    assert rf.monthly_running_power(conn, "2026-07") is None


def test_monthly_running_power_excludes_treadmill_runs(conn):
    _insert_power_run(conn, "2026-07-14", 8, 200.0, route_ref="run-a.gpx")
    _insert_power_run(conn, "2026-07-21", 8, 400.0, route_ref=None)
    assert rf.monthly_running_power(conn, "2026-07") is None
