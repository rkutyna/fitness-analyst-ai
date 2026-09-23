"""Workout-scoped facts for the run-answering tools (health_advisor#469).

``list_workouts``, ``get_run_form``, ``get_block_structure``, and
``get_briefing``'s workout focus each describe one session, not a metric
series -- ``fact_template.build_fact_set`` excludes them by design (they
carry no ``metric``/``period`` identity). Before this, a "how was my run
today?" turn that fired all four tools still published exactly one
quotable figure. ``build_workout_facts`` widens the closed fact set with an
allowlist of those tools' own fields, keyed by the workout's own identity
(date, type, and a start time only when needed -- see
``fact_template._workout_identity``).

A day is routinely more than one workout -- a run beside a walk, most days
-- so keying by date alone collides every field on that day into one
withheld identity. These tests pin: (1) a run+walk day publishes both
workouts' own numbers, (2) two same-type workouts on one day still both
publish, and (3) collapsing the key back to date-only regresses both.

All data here is synthetic (year 2001); the engine repository is public.
"""
from __future__ import annotations

import pytest

from health_advisor import fact_template


def _list_workouts_record(sequence=1, rows=None):
    if rows is None:
        rows = [{
            "date": "2001-03-04", "type": "running",
            "workout_key": "synthetic-key-1",
            "duration_min": 48.3, "distance_mi": 3.18,
            "avg_heart_rate": 156.0, "max_heart_rate": 171.0,
            "start_time_local": "08:00",
        }]
    return {
        "sequence": sequence,
        "tool_name": "list_workouts",
        "result_elided": False,
        "result": {
            "start": "2001-01-01", "end": "2001-03-04", "count": len(rows),
            "workouts": rows,
        },
    }


def _run_walk_day_rows():
    return [
        {"date": "2001-03-04", "type": "walking",
         "workout_key": "synthetic-walk-1", "duration_min": 15.1,
         "distance_mi": 0.54, "avg_heart_rate": 119.0, "max_heart_rate": 125.0,
         "start_time_local": "09:49"},
        {"date": "2001-03-04", "type": "running",
         "workout_key": "synthetic-run-1", "duration_min": 52.3,
         "distance_mi": 3.07, "avg_heart_rate": 138.0, "max_heart_rate": 167.0,
         "start_time_local": "08:57"},
    ]


def _two_runs_same_day_rows():
    return [
        {"date": "2001-03-04", "type": "running",
         "workout_key": "synthetic-run-am", "duration_min": 20.0,
         "distance_mi": 1.5, "avg_heart_rate": 140.0, "max_heart_rate": 150.0,
         "start_time_local": "06:00"},
        {"date": "2001-03-04", "type": "running",
         "workout_key": "synthetic-run-pm", "duration_min": 30.0,
         "distance_mi": 2.5, "avg_heart_rate": 145.0, "max_heart_rate": 160.0,
         "start_time_local": "18:00"},
    ]


def _get_run_form_record(sequence=2, date="2001-03-04", jog_minutes=44.0):
    return {
        "sequence": sequence,
        "tool_name": "get_run_form",
        "result_elided": False,
        "result": {
            "mode": "session", "found": True, "date": date,
            "efficiency_change": {
                "status": "ok", "change_pct": 3.4,
                "jog_minutes": jog_minutes,
            },
            "walk_structure": {"first_half_walk_fraction": 0.2},
        },
    }


def _get_block_structure_record(sequence=3, day="2001-03-04",
                                bridged=48.0, qualified=None,
                                avg_hr_session=155.7,
                                workout_type="running"):
    sessions = [{
        "workout_type": workout_type, "duration_min": 48.3,
        "longest_block_min": bridged, "qualified_block_min": qualified,
        "avg_hr_session": avg_hr_session,
    }]
    return {
        "sequence": sequence,
        "tool_name": "get_block_structure",
        "result_elided": False,
        "result": {
            "day": day, "sessions": sessions, "excluded_count": 0,
            "longest_block_min": bridged,
            "qualified_block_min": qualified,
        },
    }


def _get_briefing_record(sequence=4, date="2001-03-04"):
    return {
        "sequence": sequence,
        "tool_name": "get_briefing",
        "result_elided": False,
        "result": {
            "scope": "daily",
            "workout_focus": {
                "type": "running", "date": date,
                "duration_min": 48.3, "distance_mi": 3.2,
            },
        },
    }


def _full_ledger():
    return [
        _list_workouts_record(),
        _get_run_form_record(),
        _get_block_structure_record(),
        _get_briefing_record(),
    ]


def _run_key():
    return fact_template.workout_fact_key("2001-03-04|running", "duration_min")


# --- Identity ---------------------------------------------------------
def test_workout_identity_is_date_and_type_when_unambiguous():
    assert fact_template._workout_identity("2001-03-04", "running") == \
        "2001-03-04|running"


def test_workout_identity_adds_start_time_when_given():
    assert fact_template._workout_identity(
        "2001-03-04", "walking", "09:49") == "2001-03-04|walking|09:49"


def test_workout_identity_requires_date_and_type():
    assert fact_template._workout_identity(None, "running") is None
    assert fact_template._workout_identity("2001-03-04", None) is None


# --- Core publication ---------------------------------------------------
def test_build_workout_facts_publishes_duration_distance_max_hr():
    facts = fact_template.build_workout_facts([_list_workouts_record()])
    key = fact_template.workout_fact_key("2001-03-04|running", "duration_min")
    assert facts[key]["value"] == 48.3
    assert facts[key]["unit"] == "min"
    assert fact_template.workout_fact_key(
        "2001-03-04|running", "distance_mi") in facts
    assert facts[fact_template.workout_fact_key(
        "2001-03-04|running", "max_heart_rate")]["value"] == 171.0


def test_build_workout_facts_list_workouts_alone_publishes_no_avg_hr():
    facts = fact_template.build_workout_facts([_list_workouts_record()])
    assert fact_template.workout_fact_key(
        "2001-03-04|running", "avg_heart_rate") not in facts


def test_build_workout_facts_publishes_avg_heart_rate_from_block_structure():
    facts = fact_template.build_workout_facts([_get_block_structure_record()])
    key = fact_template.workout_fact_key("2001-03-04|running", "avg_heart_rate")
    assert facts[key]["value"] == 155.7
    assert facts[key]["unit"] == "bpm"


def test_build_workout_facts_publishes_session_jog_minutes_from_run_form():
    facts = fact_template.build_workout_facts([_get_run_form_record()])
    key = fact_template.workout_fact_key(
        "2001-03-04|running", "session_jog_minutes")
    assert facts[key]["value"] == 44.0


def test_build_workout_facts_publishes_longest_block_from_block_structure():
    facts = fact_template.build_workout_facts(
        [_get_block_structure_record(avg_hr_session=None)])
    key = fact_template.workout_fact_key("2001-03-04|running", "longest_block_min")
    assert facts[key]["value"] == 48.0
    assert fact_template.workout_fact_key(
        "2001-03-04|running", "qualified_block_min") not in facts


# --- A walk's zero "longest running block" is not a run (defect B) -------
def test_walking_workout_with_zero_longest_block_publishes_no_fact():
    """A walk's block_structure solo-day fields can be a genuine 0.

    Narrating it verbatim reads as "your run had a longest running block of
    0 minutes" for a vault that contains no run at all -- the intake tool
    correctly said there was no jogging. The field is withheld, not
    fabricated as non-zero.
    """
    facts = fact_template.build_workout_facts(
        [_get_block_structure_record(workout_type="walking", bridged=0.0,
                                      avg_hr_session=None)])
    assert fact_template.workout_fact_key(
        "2001-03-04|walking", "longest_block_min") not in facts


def test_running_workout_with_zero_longest_block_still_publishes():
    """The same zero is true and meaningful for an actual run."""
    facts = fact_template.build_workout_facts(
        [_get_block_structure_record(workout_type="running", bridged=0.0,
                                      avg_hr_session=None)])
    key = fact_template.workout_fact_key(
        "2001-03-04|running", "longest_block_min")
    assert facts[key]["value"] == 0.0


def test_walking_workout_with_nonzero_longest_block_still_publishes():
    """A walk that contained a real jogging burst still reports it."""
    facts = fact_template.build_workout_facts(
        [_get_block_structure_record(workout_type="walking", bridged=6.5,
                                      avg_hr_session=None)])
    key = fact_template.workout_fact_key(
        "2001-03-04|walking", "longest_block_min")
    assert facts[key]["value"] == 6.5


def test_walking_workout_with_zero_qualified_block_publishes_no_fact():
    facts = fact_template.build_workout_facts(
        [_get_block_structure_record(workout_type="walking", bridged=0.0,
                                      qualified=0.0, avg_hr_session=None)])
    assert fact_template.workout_fact_key(
        "2001-03-04|walking", "qualified_block_min") not in facts


def test_run_form_zero_session_jog_minutes_still_publishes():
    """session_jog_minutes is on the same gated-fields list as the two block

    fields above (see ``_RUNNING_ONLY_ZERO_GATED_FIELDS``), per the task's
    field list, even though get_run_form's own query is running-only today
    (``WHERE w.workout_type = 'running'`` in mcp_server.get_run_form) --
    so this identity can never be non-running and the gate is a no-op here.
    A genuine 0 for the running session it does describe still publishes.
    """
    facts = fact_template.build_workout_facts(
        [_get_run_form_record(jog_minutes=0.0)])
    assert facts[fact_template.workout_fact_key(
        "2001-03-04|running", "session_jog_minutes")]["value"] == 0.0


# --- The defect this session fixes: a run+walk day ----------------------
def test_run_and_walk_same_day_both_publish_duration():
    ledger = [_list_workouts_record(rows=_run_walk_day_rows())]
    facts = fact_template.build_workout_facts(ledger)
    run_key = fact_template.workout_fact_key("2001-03-04|running", "duration_min")
    walk_key = fact_template.workout_fact_key("2001-03-04|walking", "duration_min")
    assert facts[run_key]["value"] == 52.3
    assert facts[walk_key]["value"] == 15.1


def test_run_and_walk_same_day_both_publish_distance_and_max_hr():
    ledger = [_list_workouts_record(rows=_run_walk_day_rows())]
    facts = fact_template.build_workout_facts(ledger)
    for kind, distance, max_hr in (("running", 3.07, 167.0),
                                   ("walking", 0.54, 125.0)):
        assert facts[fact_template.workout_fact_key(
            f"2001-03-04|{kind}", "distance_mi")]["value"] == distance
        assert facts[fact_template.workout_fact_key(
            f"2001-03-04|{kind}", "max_heart_rate")]["value"] == max_hr


def test_run_and_walk_day_run_gets_block_and_avg_hr_walk_does_not():
    ledger = [_list_workouts_record(rows=_run_walk_day_rows()),
              _get_block_structure_record()]
    facts = fact_template.build_workout_facts(ledger)
    run_hr_key = fact_template.workout_fact_key(
        "2001-03-04|running", "avg_heart_rate")
    walk_hr_key = fact_template.workout_fact_key(
        "2001-03-04|walking", "avg_heart_rate")
    assert facts[run_hr_key]["value"] == 155.7
    assert walk_hr_key not in facts
    block_key = fact_template.workout_fact_key(
        "2001-03-04|running", "longest_block_min")
    assert facts[block_key]["value"] == 48.0


def test_two_runs_same_day_both_publish_via_start_time():
    ledger = [_list_workouts_record(rows=_two_runs_same_day_rows())]
    facts = fact_template.build_workout_facts(ledger)
    am_key = fact_template.workout_fact_key(
        "2001-03-04|running|06:00", "duration_min")
    pm_key = fact_template.workout_fact_key(
        "2001-03-04|running|18:00", "duration_min")
    assert facts[am_key]["value"] == 20.0
    assert facts[pm_key]["value"] == 30.0
    # The ambiguous, time-free key must not appear at all -- Python does not
    # pick a side.
    ambiguous_key = fact_template.workout_fact_key(
        "2001-03-04|running", "duration_min")
    assert ambiguous_key not in facts


def test_single_run_that_day_does_not_carry_a_start_time_key():
    # A lone workout of a type keys as plain date+type, matching what
    # get_run_form/get_block_structure/get_briefing publish for it, so
    # duration (list_workouts) and heart rate (get_block_structure) land
    # under the SAME key and merge.
    ledger = [_list_workouts_record(), _get_block_structure_record()]
    facts = fact_template.build_workout_facts(ledger)
    duration_key = fact_template.workout_fact_key(
        "2001-03-04|running", "duration_min")
    hr_key = fact_template.workout_fact_key(
        "2001-03-04|running", "avg_heart_rate")
    assert duration_key in facts and hr_key in facts


def test_block_structure_skips_day_level_fields_when_sessions_ambiguous():
    # Two sessions that day: Python does not guess whose block the
    # day-level "best" figure belongs to.
    record = _get_block_structure_record()
    record["result"]["sessions"].append({
        "workout_type": "walking", "duration_min": 15.0,
        "longest_block_min": 2.0, "qualified_block_min": None,
        "avg_hr_session": 110.0,
    })
    facts = fact_template.build_workout_facts([record])
    assert fact_template.workout_fact_key(
        "2001-03-04|running", "longest_block_min") not in facts
    # Per-session heart rate still publishes for each session's own type.
    assert fact_template.workout_fact_key(
        "2001-03-04|running", "avg_heart_rate") in facts
    assert fact_template.workout_fact_key(
        "2001-03-04|walking", "avg_heart_rate") in facts


# --- get_briefing / duplicate handling -----------------------------------
def test_build_workout_facts_get_briefing_alone_publishes_nothing():
    facts = fact_template.build_workout_facts([_get_briefing_record()])
    assert facts == {}


def test_build_workout_facts_briefing_does_not_shadow_list_workouts():
    facts = fact_template.build_workout_facts(
        [_list_workouts_record(), _get_briefing_record()])
    key = fact_template.workout_fact_key("2001-03-04|running", "distance_mi")
    assert facts[key]["value"] == 3.18


def test_build_workout_facts_agreeing_duplicates_publish():
    ledger = [_list_workouts_record(sequence=1),
              _list_workouts_record(sequence=2)]
    facts = fact_template.build_workout_facts(ledger)
    key = fact_template.workout_fact_key("2001-03-04|running", "duration_min")
    assert facts[key]["value"] == 48.3


def test_build_workout_facts_conflicting_duplicates_withheld():
    other = _list_workouts_record(sequence=2)
    other["result"]["workouts"][0]["duration_min"] = 99.9
    ledger = [_list_workouts_record(sequence=1), other]
    facts = fact_template.build_workout_facts(ledger)
    key = fact_template.workout_fact_key("2001-03-04|running", "duration_min")
    assert key not in facts


def test_build_workout_facts_ignores_other_tools():
    ledger = [{
        "sequence": 1, "tool_name": "get_latest", "result_elided": False,
        "result": {"metric": "resting_hr", "latest_sample": {"value": 55}},
    }]
    assert fact_template.build_workout_facts(ledger) == {}


def test_build_workout_facts_skips_elided_results():
    record = _list_workouts_record()
    record["result_elided"] = True
    assert fact_template.build_workout_facts([record]) == {}


def test_build_workout_facts_empty_ledger_publishes_nothing():
    assert fact_template.build_workout_facts([]) == {}
    assert fact_template.build_workout_facts(None) == {}


def test_workout_fact_key_round_trips():
    key = fact_template.workout_fact_key("2001-03-04|running", "duration_min")
    assert fact_template.parse_workout_fact_key(key) == (
        "2001-03-04|running", "duration_min")


def test_workout_fact_key_requires_both_parts():
    with pytest.raises(ValueError):
        fact_template.workout_fact_key("", "duration_min")
    with pytest.raises(ValueError):
        fact_template.workout_fact_key("2001-03-04|running", "")


def test_parse_workout_fact_key_rejects_metric_fact_key():
    metric_key = fact_template.fact_key("jog_minutes", "2001-03-04", "mean")
    assert fact_template.parse_workout_fact_key(metric_key) is None


def test_parse_fact_key_rejects_workout_fact_key():
    workout_key = fact_template.workout_fact_key(
        "2001-03-04|running", "duration_min")
    assert fact_template.parse_fact_key(workout_key) is None


# --- Mutation guards ------------------------------------------------------
# If build_workout_facts is disabled (returns {}), the richness this issue
# exists to fix regresses silently. Verified red against a stubbed
# `return {}` body before being left green here (see the session report for
# the captured failure output).
def test_build_workout_facts_mutation_guard_nonempty_for_real_ledger():
    facts = fact_template.build_workout_facts(_full_ledger())
    assert len(facts) >= 5


def test_mutation_guard_date_only_key_loses_the_walk(monkeypatch):
    """Collapsing the identity back to date-only regresses the run+walk day.

    This directly demonstrates the defect the orchestrator flagged: with
    ``_workout_identity`` degraded to ignore type (and start time), the
    run and the walk on 2001-03-04 share one identity, their differing
    duration values conflict, and BOTH are withheld -- the exact silent
    failure measured on 14 of 17 real run days. Verified red below; the
    fix keeps this green by keying on type.
    """
    monkeypatch.setattr(fact_template, "_workout_identity",
                        lambda date, workout_type, start_time=None: date)
    ledger = [_list_workouts_record(rows=_run_walk_day_rows())]
    facts = fact_template.build_workout_facts(ledger)
    run_key = fact_template.workout_fact_key("2001-03-04", "duration_min")
    assert run_key not in facts, (
        "date-only keying should collide the run and the walk and withhold "
        "both -- if this key is present, the mutation guard is not "
        "actually exercising the collision"
    )


# --- Gate refusal ---------------------------------------------------------
def test_scan_template_refuses_fabricated_workout_placeholder():
    facts = fact_template.build_workout_facts([_list_workouts_record()])
    bad_field = "{fact|workout=2001-03-04%7Crunning|field=vo2max}"
    scan = fact_template.scan_template(f"Today's run: {bad_field}", facts)
    assert not scan["ok"]
    assert scan["reason"] == "unresolvable placeholder"


def test_scan_template_refuses_fabricated_workout_type():
    # A day where only a run happened; a walk placeholder for that date
    # must still be refused, not silently answered from the run's figures.
    facts = fact_template.build_workout_facts([_list_workouts_record()])
    bad_key = fact_template.workout_fact_key("2001-03-04|walking", "duration_min")
    scan = fact_template.scan_template("Walk: {" + bad_key + "}", facts)
    assert not scan["ok"]
    assert bad_key in scan["unresolved"]


def test_scan_template_accepts_real_workout_placeholder():
    facts = fact_template.build_workout_facts([_list_workouts_record()])
    key = fact_template.workout_fact_key("2001-03-04|running", "duration_min")
    scan = fact_template.scan_template("Duration: {" + key + "}", facts)
    assert scan["ok"]
    interpolated = fact_template.interpolate_template(
        "Duration: {" + key + "}", facts)
    assert interpolated == "Duration: 48.3"


def test_scan_template_lets_the_model_pick_the_run_on_a_run_and_walk_day():
    ledger = [_list_workouts_record(rows=_run_walk_day_rows())]
    facts = fact_template.build_workout_facts(ledger)
    run_key = fact_template.workout_fact_key("2001-03-04|running", "duration_min")
    walk_key = fact_template.workout_fact_key("2001-03-04|walking", "duration_min")
    template = f"Run: {{{run_key}}}. Walk: {{{walk_key}}}."
    scan = fact_template.scan_template(template, facts)
    assert scan["ok"]
    assert fact_template.interpolate_template(template, facts) == \
        "Run: 52.3. Walk: 15.1."


def test_workout_fact_placeholder_prompt_text_is_jargon_free():
    from health_advisor import chat
    lowered = chat._WORKOUT_FACT_PLACEHOLDER_TEXT.lower()
    for term in ("leaf", "leaves", "closed fact set", "citable"):
        assert term not in lowered
    assert "workout=" in chat._WORKOUT_FACT_PLACEHOLDER_TEXT
    assert "date" in lowered and "type" in lowered and "start time" in lowered
