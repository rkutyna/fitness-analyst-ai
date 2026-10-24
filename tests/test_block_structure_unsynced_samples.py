"""get_block_structure must not publish an unmeasured block as a measured 0.0.

A running workout whose per-sample distance, step and heart-rate series have not
synced yet has zero distance buckets. analysis.longest_block then reports
0.0 minutes, and the ask tool narrated that as "the longest running block was
0 min". Python owns the truth: an unmeasured session publishes None and says why.

Oracles are literals written from the rule, not the code's own output. The
measured-session literal (test_measured_session_is_byte_identical_to_base) was
captured from the code BEFORE the unsynced branch existed.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from health_advisor import metrics as mx

BUCKET_MIN = mx.IMPACT_BUCKET_SECONDS / 60.0
NOT_SYNCED_NOTE = ("this session's minute-by-minute distance and step samples "
                   "are not in the vault yet, so its blocks cannot be measured")

MEASURED_DAY = "2026-07-02"
UNSYNCED_DAY = "2026-07-03"
MIXED_DAY = "2026-07-04"


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


def _measured_session(conn, base: str, day: str):
    """12 jog buckets (pace 14, cadence 135, HR 140), then the workout row."""
    for i in range(12):
        s = _iso(base, i)
        _record(conn, "distance_walking_running", s, BUCKET_MIN / 14.0, "mi")
        _record(conn, "step_count", s, 135.0 / 3.0, "count")
        _record(conn, "heart_rate", s, 140.0, "count/min")
    _workout(conn, base, _iso(base, 12), day, 12 * BUCKET_MIN, 140.0)


def _unsynced_session(conn, base: str, day: str):
    """A 40-minute run whose only samples are running_speed: no distance,
    step or heart-rate rows fall inside its window."""
    for i in range(120):
        _record(conn, "running_speed", _iso(base, i), 2.7, "m/s")
    _workout(conn, base, _iso(base, 120), day, 40.0, None)


def _no_zero_figures(value, path="$"):
    """Every float 0.0 in the structure, with where it sits."""
    found = []
    if isinstance(value, dict):
        for k, v in value.items():
            found += _no_zero_figures(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found += _no_zero_figures(v, f"{path}[{i}]")
    elif isinstance(value, float) and value == 0.0:
        found.append(path)
    return found


# Captured from the unmodified code, before the unsynced branch was added.
MEASURED_BASE = {
    "day": "2026-07-02",
    "excluded_count": 0,
    "hr_ceiling_for_qualifying": 150.0,
    "longest_block_min": 4.0,
    "qualified_block_min": 4.0,
    "sessions": [{
        "avg_hr_all_jog": 140.0,
        "avg_hr_longest_block": 140.0,
        "avg_hr_session": 140.0,
        "duration_min": 4.0,
        "longest_block_min": 4.0,
        "qualified_block_min": 4.0,
        "reps": [{
            "bridged": False,
            "end_utc": "2026-07-02T12:03:40Z",
            "mean_hr": 140.0,
            "mean_pace_min_per_mi": 14.0,
            "minutes": 4.0,
            "start_utc": "2026-07-02T12:00:00Z",
        }],
        "unbridged_min": 4.0,
        "workout_type": "running",
    }],
}


def test_unsynced_running_session_publishes_none_not_zero(conn, tools):
    _unsynced_session(conn, "2026-07-03T12:00:00Z", UNSYNCED_DAY)
    conn.commit()

    out = tools.get_block_structure(day=UNSYNCED_DAY)

    assert out["sessions"] == [{
        "workout_type": "running",
        "duration_min": 40.0,
        "longest_block_min": None,
        "qualified_block_min": None,
        "unbridged_min": None,
        "avg_hr_longest_block": None,
        "avg_hr_all_jog": None,
        "avg_hr_session": None,
        "reps": [],
        "samples": "not_synced",
        "note": NOT_SYNCED_NOTE,
    }]
    assert out["longest_block_min"] is None
    assert out["qualified_block_min"] is None
    assert out["samples"] == "not_synced"
    assert out["note"] == NOT_SYNCED_NOTE
    # No unmeasured figure may be published as a number, anywhere in the day.
    assert _no_zero_figures(out) == []


def test_measured_session_is_byte_identical_to_base(conn, tools):
    _measured_session(conn, "2026-07-02T12:00:00Z", MEASURED_DAY)
    conn.commit()

    out = tools.get_block_structure(day=MEASURED_DAY)

    assert out == MEASURED_BASE
    assert json.dumps(out, sort_keys=True) == json.dumps(
        MEASURED_BASE, sort_keys=True)
    assert "samples" not in out["sessions"][0]
    assert "note" not in out["sessions"][0]


def test_day_with_measured_and_unsynced_session_reports_the_measured_block(
        conn, tools):
    _measured_session(conn, "2026-07-04T12:00:00Z", MIXED_DAY)
    _unsynced_session(conn, "2026-07-04T18:00:00Z", MIXED_DAY)
    conn.commit()

    out = tools.get_block_structure(day=MIXED_DAY)

    assert len(out["sessions"]) == 2
    measured, unsynced = out["sessions"]
    assert measured["longest_block_min"] == 4.0
    assert "samples" not in measured
    assert unsynced["longest_block_min"] is None
    assert unsynced["samples"] == "not_synced"
    assert unsynced["note"] == NOT_SYNCED_NOTE
    assert out["longest_block_min"] == 4.0
    assert out["qualified_block_min"] == 4.0


# ---- indoor and treadmill sessions: zero buckets that will never sync ----

INDOOR_DAY = "2026-07-08"
INDOOR_NOT_RECORDED_NOTE = ("an indoor or treadmill session records no "
                            "per-sample distance, so its blocks cannot be measured")


def _indoor_session(conn, base: str, day: str, *, is_indoor, fitness_machine):
    """A 40-minute run with running_speed samples only and no distance, step or
    heart-rate rows: what GymKit writes for a treadmill run."""
    for i in range(120):
        _record(conn, "running_speed", _iso(base, i), 2.7, "m/s")
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, avg_heart_rate, is_indoor, fitness_machine, source, "
        "dedupe_key) VALUES ('running', ?, ?, ?, 40.0, NULL, ?, ?, 'test', ?)",
        (base, _iso(base, 120), day, is_indoor, fitness_machine,
         f"w|{base}|running"))


def test_indoor_running_session_publishes_none_as_not_recorded(conn, tools):
    _indoor_session(conn, "2026-07-08T12:00:00Z", INDOOR_DAY,
                    is_indoor=1, fitness_machine=None)
    conn.commit()

    out = tools.get_block_structure(day=INDOOR_DAY)

    assert out["sessions"] == [{
        "workout_type": "running",
        "duration_min": 40.0,
        "longest_block_min": None,
        "qualified_block_min": None,
        "unbridged_min": None,
        "avg_hr_longest_block": None,
        "avg_hr_all_jog": None,
        "avg_hr_session": None,
        "reps": [],
        "samples": "not_recorded",
        "note": INDOOR_NOT_RECORDED_NOTE,
    }]
    assert out["longest_block_min"] is None
    assert out["qualified_block_min"] is None
    assert out["samples"] == "not_recorded"
    assert out["note"] == INDOOR_NOT_RECORDED_NOTE
    assert _no_zero_figures(out) == []


def test_treadmill_machine_alone_is_not_recorded(conn, tools):
    """fitness_machine set while is_indoor is NULL ("the device did not say")
    still means a machine session: it records no distance either."""
    _indoor_session(conn, "2026-07-08T12:00:00Z", INDOOR_DAY,
                    is_indoor=None, fitness_machine="treadmill")
    conn.commit()

    out = tools.get_block_structure(day=INDOOR_DAY)

    assert out["sessions"][0]["samples"] == "not_recorded"
    assert out["samples"] == "not_recorded"
    assert out["longest_block_min"] is None


def test_indoor_day_beside_an_outdoor_unsynced_run_is_not_synced(conn, tools):
    """The day stays "not_synced" while an outdoor run may still sync; the
    indoor session is individually "not_recorded" and never becomes a 0."""
    _indoor_session(conn, "2026-07-08T12:00:00Z", INDOOR_DAY,
                    is_indoor=1, fitness_machine=None)
    _unsynced_session(conn, "2026-07-08T18:00:00Z", INDOOR_DAY)
    conn.commit()

    out = tools.get_block_structure(day=INDOOR_DAY)

    indoor, outdoor = out["sessions"]
    assert indoor["samples"] == "not_recorded"
    assert indoor["note"] == INDOOR_NOT_RECORDED_NOTE
    assert outdoor["samples"] == "not_synced"
    assert outdoor["note"] == NOT_SYNCED_NOTE
    assert out["samples"] == "not_synced"
    assert out["note"] == NOT_SYNCED_NOTE
    assert out["longest_block_min"] is None
    assert _no_zero_figures(out) == []
