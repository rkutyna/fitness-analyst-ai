"""Unsynced running samples publish as unmeasured, never as a measured 0 min.

A running workout whose distance, step and heart-rate series have not synced
has no distance buckets. Two publishers turned that into a number:
  - get_impact_volume's gap-filled period published jog_minutes 0.0 for a day
    with no impact row, which the model narrates as "0 min of jogging";
  - get_run_form's session mode refused with insufficient_jog_time and
    jog_minutes 0.0, which reads as "too little jogging" rather than "not synced".

Rule: a running, walking or hiking workout whose samples have not synced is
unmeasured, so its figures are None and it says ``samples: "not_synced"``. A
day with NO workout at all is a real rest day: its jog minutes are a measured
0.0, exactly as before. Block totals sum measured days only and name the
unsynced ones in ``samples_not_synced_days``.

Oracles are literals written from the rule. The measured-day, measured-session
and rest-day literals were captured from the code BEFORE these publishers
changed, and the controls assert they are still byte-identical.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from health_advisor import metrics as mx

BUCKET_MIN = mx.IMPACT_BUCKET_SECONDS / 60.0
DAY_MEASURED = "2026-07-02"    # 40 jog buckets (13.3 min), measured
DAY_UNSYNCED = "2026-07-03"    # 40-min run with running_speed samples only
DAY_FEW = "2026-07-04"         # 12 jog buckets (4.0 min), under the jog floor
AS_OF = "2026-07-10"

NOT_SYNCED_REFUSAL = {"status": "samples_not_synced",
                      "reason": "samples_not_synced",
                      "jog_minutes": None}


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


def _measured_session(conn, base: str, day: str, buckets: int):
    """Jog buckets at pace 14, cadence 135, HR 140, then the workout row."""
    for i in range(buckets):
        s = _iso(base, i)
        _record(conn, "distance_walking_running", s, BUCKET_MIN / 14.0, "mi")
        _record(conn, "step_count", s, 135.0 / 3.0, "count")
        _record(conn, "heart_rate", s, 140.0, "count/min")
    _workout(conn, base, _iso(base, buckets), day,
             buckets * BUCKET_MIN, 140.0)


def _unsynced_session(conn, base: str, day: str):
    """A 40-minute run whose only samples are running_speed."""
    for i in range(120):
        _record(conn, "running_speed", _iso(base, i), 2.7, "m/s")
    _workout(conn, base, _iso(base, 120), day, 40.0, None)


def _vault_with_sessions(conn):
    conn.execute(
        "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, "
        "last, unit) VALUES ('step_count', ?, 1, 1.0, 1.0, 1.0, 1.0, 1.0, "
        "'count')", (AS_OF,))
    _measured_session(conn, "2026-07-02T12:00:00Z", DAY_MEASURED, 40)
    _unsynced_session(conn, "2026-07-03T12:00:00Z", DAY_UNSYNCED)
    _measured_session(conn, "2026-07-04T12:00:00Z", DAY_FEW, 12)
    conn.commit()


def _period(out: dict, day: str) -> dict:
    return next(p for p in out["periods"] if p["period_start"] == day)


# ---- frozen literals captured from the unmodified code ----
IMPACT_MEASURED_DAY_BASE = json.loads(r'''
{
 "as_of": "2026-07-10",
 "by": "day",
 "count": 1,
 "end": "2026-07-02",
 "jog_cadence_threshold_steps_per_min": 140.0,
 "jog_effort_cadence_threshold_steps_per_min": 130.0,
 "jog_effort_heart_rate_threshold_bpm": 130.0,
 "jog_near_threshold": {
  "buckets_near_cutoff": 0,
  "jog_buckets": 40,
  "note": "0 eligible buckets sit within 10.0 steps/min below the 140 steps/min cadence cutoff. Read a fall in jog_minutes against the sensitivity row above before calling it a fall in training volume.",
  "pct_of_jog_buckets": 0.0,
  "within_steps_per_min": 10.0
 },
 "jog_threshold_sensitivity": [
  {
   "cadence_min_steps_per_min": 130.0,
   "field_metrics": {
    "jog_minutes": "jog_minutes"
   },
   "jog_buckets": 40,
   "jog_minutes": 13.3,
   "live_cutoff": false,
   "presentation": {
    "field": "presentation",
    "metric": "jog_minutes",
    "period": null,
    "value": "13 min"
   }
  },
  {
   "cadence_min_steps_per_min": 139.0,
   "field_metrics": {
    "jog_minutes": "jog_minutes"
   },
   "jog_buckets": 40,
   "jog_minutes": 13.3,
   "live_cutoff": false,
   "presentation": {
    "field": "presentation",
    "metric": "jog_minutes",
    "period": null,
    "value": "13 min"
   }
  },
  {
   "cadence_min_steps_per_min": 140.0,
   "field_metrics": {
    "jog_minutes": "jog_minutes"
   },
   "jog_buckets": 40,
   "jog_minutes": 13.3,
   "live_cutoff": true,
   "presentation": {
    "field": "presentation",
    "metric": "jog_minutes",
    "period": null,
    "value": "13 min"
   }
  },
  {
   "cadence_min_steps_per_min": 141.0,
   "field_metrics": {
    "jog_minutes": "jog_minutes"
   },
   "jog_buckets": 40,
   "jog_minutes": 13.3,
   "live_cutoff": false,
   "presentation": {
    "field": "presentation",
    "metric": "jog_minutes",
    "period": null,
    "value": "13 min"
   }
  },
  {
   "cadence_min_steps_per_min": 150.0,
   "field_metrics": {
    "jog_minutes": "jog_minutes"
   },
   "jog_buckets": 40,
   "jog_minutes": 13.3,
   "live_cutoff": false,
   "presentation": {
    "field": "presentation",
    "metric": "jog_minutes",
    "period": null,
    "value": "13 min"
   }
  }
 ],
 "periods": [
  {
   "days_covered": 1,
   "jog_change_note": "no preceding day inside the requested range",
   "jog_change_pct": null,
   "jog_miles": 0.95,
   "jog_minutes": 13.3,
   "jog_minutes_source": "measured",
   "jog_pace_min_per_mi": 14.0,
   "manual_note": null,
   "metric": "jog_minutes",
   "no_data": false,
   "partial": false,
   "period": "2026-07-02",
   "period_start": "2026-07-02",
   "presentation": {
    "field": "presentation",
    "metric": "jog_minutes",
    "period": "2026-07-02",
    "value": "13 min"
   },
   "walk_miles": 0.0,
   "walk_minutes": 0.0
  }
 ],
 "start": "2026-07-02"
}
''')

RUN_FORM_MEASURED_BASE = json.loads(r'''
{
 "date": "2026-07-02",
 "efficiency_change": {
  "buckets_dropped_implausible": 0,
  "change_pct": 0.0,
  "first_half_buckets": 20,
  "first_half_efficiency": 0.0306,
  "jog_minutes": 13.3,
  "second_half_buckets": 20,
  "second_half_efficiency": 0.0306,
  "status": "ok"
 },
 "found": true,
 "mode": "session",
 "personal_reference": {
  "n_sessions": 0,
  "reason": "0 comparable session(s); need 5",
  "status": "insufficient_sessions"
 },
 "walk_structure": {
  "first_half_walk_fraction": 0.0,
  "jog_minutes": 13.3,
  "mean_bout_minutes": 0.0,
  "second_half_walk_fraction": 0.0,
  "status": "ok",
  "walk_bouts": 0,
  "walk_fraction": 0.0,
  "walk_minutes": 0.0
 }
}
''')

RUN_FORM_FEW_BASE = json.loads(r'''
{
 "date": "2026-07-04",
 "efficiency_change": {
  "jog_minutes": 4.0,
  "reason": "4.0 min of jogging; need 10",
  "status": "insufficient_jog_time"
 },
 "found": true,
 "mode": "session",
 "personal_reference": {
  "n_sessions": 1,
  "reason": "1 comparable session(s); need 5",
  "status": "insufficient_sessions"
 },
 "walk_structure": {
  "jog_minutes": 4.0,
  "reason": "4.0 min of jogging; need 10",
  "status": "insufficient_jog_time"
 }
}
''')

# ---- (a) the unsynced publishers ----

def test_unsynced_day_publishes_jog_minutes_none_not_zero(conn, tools):
    _vault_with_sessions(conn)

    out = tools.get_impact_volume(start=DAY_MEASURED, end=DAY_UNSYNCED, by="day")

    row = _period(out, DAY_UNSYNCED)
    assert row["no_data"] is True
    assert row["jog_minutes"] is None
    assert row["jog_miles"] is None
    assert row["walk_minutes"] is None
    assert row["walk_miles"] is None
    assert row["samples"] == "not_synced"
    assert "presentation" not in row      # no "0 min" rendering either
    # The measured day in the same range is still a number.
    assert _period(out, DAY_MEASURED)["jog_minutes"] == 13.3
    assert _period(out, DAY_MEASURED)["no_data"] is False


def test_unsynced_session_run_form_refuses_as_not_synced(conn, tools):
    _vault_with_sessions(conn)

    out = tools.get_run_form(workout_date=DAY_UNSYNCED)

    assert out["mode"] == "session" and out["found"] is True
    assert out["efficiency_change"] == NOT_SYNCED_REFUSAL
    assert out["walk_structure"] == NOT_SYNCED_REFUSAL
    assert "insufficient_jog_time" not in json.dumps(out)


# ---- (b) controls: measured outputs are byte-identical to the base ----

def test_measured_day_impact_is_byte_identical_to_base(conn, tools):
    _vault_with_sessions(conn)

    out = tools.get_impact_volume(start=DAY_MEASURED, end=DAY_MEASURED, by="day")

    assert out == IMPACT_MEASURED_DAY_BASE
    assert json.dumps(out, sort_keys=True) == json.dumps(
        IMPACT_MEASURED_DAY_BASE, sort_keys=True)


def test_measured_session_run_form_is_byte_identical_to_base(conn, tools):
    _vault_with_sessions(conn)

    out = tools.get_run_form(workout_date=DAY_MEASURED)

    assert out == RUN_FORM_MEASURED_BASE
    assert json.dumps(out, sort_keys=True) == json.dumps(
        RUN_FORM_MEASURED_BASE, sort_keys=True)


def test_few_bucket_session_stays_insufficient_jog_time(conn, tools):
    _vault_with_sessions(conn)

    out = tools.get_run_form(workout_date=DAY_FEW)

    assert out == RUN_FORM_FEW_BASE
    assert out["efficiency_change"] == {
        "status": "insufficient_jog_time",
        "reason": "4.0 min of jogging; need 10",
        "jog_minutes": 4.0}
    assert out["walk_structure"]["status"] == "insufficient_jog_time"
    assert out["walk_structure"]["jog_minutes"] == 4.0


# ---- (c) a true rest day is still a measured zero ----

DAY_REST = "2026-07-05"        # Sunday, no workout of any kind
REST_DAY_BASE = json.loads(r'''
{"days_covered": 1, "jog_change_pct": -100.0, "jog_miles": 0.0,
 "jog_minutes": 0.0, "jog_pace_min_per_mi": null, "metric": "jog_minutes",
 "no_data": true, "partial": false, "period": "2026-07-05",
 "period_start": "2026-07-05",
 "presentation": {"field": "presentation", "metric": "jog_minutes",
  "period": "2026-07-05", "value": "0 min"},
 "walk_miles": 0.0, "walk_minutes": 0.0}
''')


def test_true_rest_day_stays_a_measured_zero(conn, tools):
    _vault_with_sessions(conn)

    out = tools.get_impact_volume(start=DAY_FEW, end=DAY_REST, by="day")

    row = _period(out, DAY_REST)
    assert row["jog_minutes"] == 0.0
    assert row["walk_minutes"] == 0.0
    assert "samples" not in row
    assert "samples_not_synced_days" not in out
    assert row == REST_DAY_BASE


# ---- (d) a block comparison over a week with one unsynced day ----

AS_OF_WEEK = "2026-07-20"
UNSYNCED_IN_RECENT = "2026-07-08"


def test_block_comparison_sums_measured_days_and_names_the_unsynced_one(
        conn, tools):
    conn.execute(
        "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, "
        "last, unit) VALUES ('step_count', ?, 1, 1.0, 1.0, 1.0, 1.0, 1.0, "
        "'count')", (AS_OF_WEEK,))
    _measured_session(conn, "2026-07-01T12:00:00Z", "2026-07-01", 40)  # prior
    _measured_session(conn, "2026-07-07T12:00:00Z", "2026-07-07", 40)  # recent
    _unsynced_session(conn, "2026-07-08T12:00:00Z", UNSYNCED_IN_RECENT)
    _measured_session(conn, "2026-07-09T12:00:00Z", "2026-07-09", 12)  # recent
    conn.commit()

    out = tools.get_impact_volume(start="2026-06-29", end="2026-07-12",
                                  by="week", weeks_per_block=1,
                                  anchor="last_complete_week")

    assert "block_comparison_error" not in out
    recent_week = next(p for p in out["periods"]
                       if p["period_start"] == "2026-07-06")
    assert recent_week["jog_minutes"] == 17.3          # 13.3 + 4.0, measured
    assert recent_week["samples_not_synced_days"] == [UNSYNCED_IN_RECENT]
    blocks = out["block_comparison"]["blocks"]
    recent, prior = blocks["recent"], blocks["prior"]
    assert recent["total"] == 17.3                     # 13.3 + 4.0, literal
    assert prior["total"] == 13.3
    assert recent["samples_not_synced_days"] == [UNSYNCED_IN_RECENT]
    assert "samples_not_synced_days" not in prior


# ---- (e) a week with no measured day is unmeasured, a true rest week is 0.0 ----

WEEK_ALL_UNSYNCED = "2026-07-06"     # Mon..Sun; only an unsynced run on 07-08
WEEK_REST = "2026-07-13"             # no workout of any kind


def test_all_unsynced_week_totals_are_none_not_zero(conn, tools):
    _measured_session(conn, "2026-06-30T12:00:00Z", "2026-06-30", 40)  # earlier
    _unsynced_session(conn, "2026-07-08T12:00:00Z", UNSYNCED_IN_RECENT)
    conn.commit()

    out = tools.get_impact_volume(start="2026-06-29", end="2026-07-12",
                                  by="week")

    week = _period(out, WEEK_ALL_UNSYNCED)
    assert week["samples"] == "not_synced"
    assert week["jog_minutes"] is None
    assert week["jog_miles"] is None
    assert week["walk_minutes"] is None
    assert week["walk_miles"] is None
    assert week["no_data"] is True
    assert week["samples_not_synced_days"] == [UNSYNCED_IN_RECENT]
    assert "presentation" not in week


def test_true_rest_week_total_stays_a_literal_zero(conn, tools):
    _measured_session(conn, "2026-06-30T12:00:00Z", "2026-06-30", 40)  # earlier
    _unsynced_session(conn, "2026-07-08T12:00:00Z", UNSYNCED_IN_RECENT)
    conn.commit()

    out = tools.get_impact_volume(start="2026-06-29", end="2026-07-19", by="week")

    week = _period(out, WEEK_REST)
    assert week["jog_minutes"] == 0.0 and type(week["jog_minutes"]) is float
    assert week["walk_minutes"] == 0.0
    assert "samples" not in week
    assert "samples_not_synced_days" not in week


def test_block_with_all_unsynced_week_has_no_total_and_no_change(conn, tools):
    _measured_session(conn, "2026-07-01T12:00:00Z", "2026-07-01", 40)  # prior
    _unsynced_session(conn, "2026-07-08T12:00:00Z", UNSYNCED_IN_RECENT)
    conn.commit()

    out = tools.get_impact_volume(start="2026-06-29", end="2026-07-12",
                                  by="week", weeks_per_block=1,
                                  anchor="last_complete_week")

    blocks = out["block_comparison"]["blocks"]
    assert blocks["recent"]["total"] is None
    assert blocks["recent"]["mean"] is None
    assert blocks["recent"]["samples_not_synced_days"] == [UNSYNCED_IN_RECENT]
    assert blocks["prior"]["total"] == 13.3
    assert out["block_comparison"]["change"]["total_delta"] is None
    assert out["block_comparison"]["change"]["total_delta_pct"] is None
