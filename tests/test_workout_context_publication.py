"""How workout context is published (consumer #590, engine half).

``list_workouts`` rows gain ``is_indoor`` and ``fitness_machine``, and, only for
a session recorded with a connected machine that also reported a climb over a
positive distance, ``average_grade_pct``. The closed fact set gains Python-
authored labels for them. Absence is not a fact: a session whose device did not
say publishes nothing about where it was done, and a session with no machine or
no climb publishes no grade -- never a 0.

Everything here is synthetic; the engine repository is public.
"""
from __future__ import annotations

import copy
import sqlite3

import pytest

from health_advisor import db, fact_template, hk_parse, mcp_server
from health_advisor import vault as vault_mod

DAY = "2030-01-05"
METRES_PER_MILE = 1609.344


def _workout(key, *, start_hour=15, distance_mi=3.0, wtype="running", **extra):
    row = {
        "workout_type": wtype,
        "start_utc": f"{DAY}T{start_hour:02d}:00:00+00:00",
        "end_utc": f"{DAY}T{start_hour:02d}:30:00+00:00",
        "local_date": DAY, "duration_min": 30.0, "energy_kcal": None,
        "distance_mi": distance_mi, "unit_distance": "mi" if distance_mi else None,
        "source": "Synthetic Source", "dedupe_key": key, "hk_uuid": key,
    }
    row.update(extra)
    return row


def _listed(tools, **kwargs):
    return tools.list_workouts(start=DAY, end=DAY, **kwargs)


def _by_key(listed):
    return {w["workout_key"]: w for w in listed["workouts"]}


def _facts(listed, sequence=1):
    return fact_template.build_workout_facts([{
        "sequence": sequence, "tool_name": "list_workouts",
        "result_elided": False, "result": listed}])


def _fields(facts, field):
    return {f["workout"]: f for f in facts.values() if f["field"] == field}


# --------------------------------------------------------------------------- #
# list_workouts rows
# --------------------------------------------------------------------------- #
def test_a_row_with_null_context_differs_from_before_only_by_two_null_keys(
        conn, tools):
    db.insert_workouts(conn, [_workout("plain")])
    conn.commit()
    row = _by_key(_listed(tools))["plain"]
    before = {
        "date": DAY, "type": "running", "workout_key": "plain",
        "duration_min": 30.0, "energy_kcal": None, "distance_mi": 3.0,
        "pace_min_per_mi": 10.0,
        "avg_heart_rate": None, "max_heart_rate": None,
        "start_time_local": row["start_time_local"],
        "end_time_local": row["end_time_local"],
        "has_route": False, "n_segments": 0,
    }
    assert {k: v for k, v in row.items()
            if k not in ("is_indoor", "fitness_machine")} == before
    assert row["is_indoor"] is None and row["fitness_machine"] is None
    assert "average_grade_pct" not in row
    # The two new keys come last, so the leading bytes of the row are unchanged.
    assert list(row)[-2:] == ["is_indoor", "fitness_machine"]
    assert list(row)[:-2] == list(before)


@pytest.mark.parametrize("stored, published", [(1, True), (0, False), (None, None)])
def test_is_indoor_is_a_real_boolean_or_null(conn, tools, stored, published):
    db.insert_workouts(conn, [_workout("w", is_indoor=stored)])
    conn.commit()
    value = _by_key(_listed(tools))["w"]["is_indoor"]
    assert value is published


def test_a_null_is_indoor_is_never_rendered_as_outdoors(conn, tools):
    db.insert_workouts(conn, [_workout("w")])
    conn.commit()
    value = _by_key(_listed(tools))["w"]["is_indoor"]
    assert value is None
    assert value is not False


def test_fitness_machine_is_published_as_stored(conn, tools):
    db.insert_workouts(conn, [_workout("w", fitness_machine="treadmill",
                                       is_indoor=1)])
    conn.commit()
    assert _by_key(_listed(tools))["w"]["fitness_machine"] == "treadmill"


def test_grade_is_climb_over_distance_for_a_machine_session(conn, tools):
    climb_m, distance_mi = 150.0, 3.0
    db.insert_workouts(conn, [_workout(
        "w", distance_mi=distance_mi, is_indoor=1, fitness_machine="treadmill",
        elevation_ascended_m=climb_m)])
    conn.commit()
    expected = round(climb_m / (distance_mi * METRES_PER_MILE) * 100.0, 1)
    assert expected == 3.1
    row = _by_key(_listed(tools))["w"]
    assert row["average_grade_pct"] == expected


def test_grade_is_the_same_number_on_a_metric_vault(conn, tools):
    db.insert_workouts(conn, [_workout(
        "w", distance_mi=3.0, fitness_machine="treadmill",
        elevation_ascended_m=150.0)])
    conn.commit()
    vault_mod.set_unit_system(conn, "metric")
    conn.commit()
    row = _by_key(_listed(tools))["w"]
    assert "distance_km" in row and "distance_mi" not in row
    assert row["average_grade_pct"] == 3.1


def test_grade_uses_stored_values_not_rounded_ones(conn, tools):
    # 0.004 mi rounds to 0.0 at two decimals; the grade must still come from
    # the stored distance.
    db.insert_workouts(conn, [_workout(
        "w", distance_mi=0.004, fitness_machine="treadmill",
        elevation_ascended_m=0.5)])
    conn.commit()
    row = _by_key(_listed(tools))["w"]
    assert row["distance_mi"] == 0.0
    assert row["average_grade_pct"] == round(
        0.5 / (0.004 * METRES_PER_MILE) * 100.0, 1)


def test_a_flat_machine_climb_of_zero_is_a_stated_zero_grade(conn, tools):
    db.insert_workouts(conn, [_workout(
        "w", fitness_machine="treadmill", elevation_ascended_m=0.0)])
    conn.commit()
    assert _by_key(_listed(tools))["w"]["average_grade_pct"] == 0.0


@pytest.mark.parametrize("label, extra", [
    ("no machine, climb stored", {"is_indoor": 0, "elevation_ascended_m": 80.0}),
    ("no machine, indoor, climb stored",
     {"is_indoor": 1, "elevation_ascended_m": 80.0}),
    ("indoor, no machine, no climb", {"is_indoor": 1}),
    ("machine, no climb", {"is_indoor": 1, "fitness_machine": "treadmill"}),
    ("machine, climb, no distance",
     {"fitness_machine": "treadmill", "elevation_ascended_m": 80.0,
      "distance_mi": None}),
    ("machine, climb, zero distance",
     {"fitness_machine": "treadmill", "elevation_ascended_m": 80.0,
      "distance_mi": 0.0}),
])
def test_no_grade_without_a_machine_a_climb_and_a_distance(conn, tools, label, extra):
    row_extra = dict(extra)
    distance = row_extra.pop("distance_mi", 3.0)
    db.insert_workouts(conn, [_workout("w", distance_mi=distance, **row_extra)])
    conn.commit()
    row = _by_key(_listed(tools))["w"]
    assert "average_grade_pct" not in row, label
    # Neither is a climb figure published on the row: that stays as it was.
    assert not [k for k in row if "climb" in k or "elevation" in k], label


def test_an_old_vault_read_only_lists_workouts_with_nulls(tmp_path, vault, tools):
    """A read-only connection never migrates. A vault not opened for writing
    since the columns were added still lists, with the context read as not
    stated."""
    conn = vault.connect()
    db.init_db(conn)
    db.insert_workouts(conn, [_workout("old")])
    conn.commit()
    conn.close()
    raw = sqlite3.connect(vault.db_path)
    for column in ("is_indoor", "fitness_machine", "elevation_ascended_m"):
        raw.execute(f"ALTER TABLE workouts DROP COLUMN {column}")
    raw.commit()
    raw.close()
    row = _by_key(_listed(tools))["old"]
    assert row["is_indoor"] is None and row["fitness_machine"] is None
    assert "average_grade_pct" not in row


def test_the_tool_description_says_null_is_not_outdoors():
    doc = mcp_server.list_workouts.__doc__
    assert doc is not None
    assert "null is not outdoors" in doc


# --------------------------------------------------------------------------- #
# The closed fact set
# --------------------------------------------------------------------------- #
def _machine_row(**extra):
    row = {
        "date": DAY, "type": "running", "workout_key": "k",
        "duration_min": 30.0, "distance_mi": 3.0, "max_heart_rate": 150.0,
        "start_time_local": "10:00", "is_indoor": True,
        "fitness_machine": "treadmill", "average_grade_pct": 3.1,
    }
    row.update(extra)
    return row


def _listing(*rows):
    return {"start": DAY, "end": DAY, "count": len(rows),
            "workouts": list(rows)}


def test_a_machine_session_publishes_setting_machine_grade_and_climb_source():
    facts = _facts(_listing(_machine_row()))
    workout = f"{DAY}|running"
    grade = _fields(facts, "average_grade_pct")[workout]
    assert grade["value"] == 3.1 and grade["unit"] == "%"
    assert grade["display"] == "3.1"
    assert grade["source"]["path"] == "$.result.workouts[0].average_grade_pct"
    assert _fields(facts, "setting")[workout]["display"] == "indoors"
    assert _fields(facts, "fitness_machine")[workout]["display"] == "a treadmill"
    assert _fields(facts, "climb_source")[workout]["display"] == (
        "the machine's own climb figure")


@pytest.mark.parametrize("indoor, value, display", [
    (True, "indoor", "indoors"), (False, "outdoor", "outdoors")])
def test_a_stated_setting_is_published(indoor, value, display):
    facts = _facts(_listing(_machine_row(
        is_indoor=indoor, fitness_machine=None, average_grade_pct=None)))
    fact = _fields(facts, "setting")[f"{DAY}|running"]
    assert (fact["value"], fact["display"]) == (value, display)


def test_a_null_setting_publishes_nothing_about_where_it_was_done():
    facts = _facts(_listing(_machine_row(is_indoor=None)))
    assert _fields(facts, "setting") == {}
    assert not [f for f in facts.values()
                if f.get("value") in ("outdoor", "outdoors")
                or f.get("display") == "outdoors"]


def test_no_grade_fact_without_a_machine_even_if_the_row_carries_one():
    facts = _facts(_listing(_machine_row(fitness_machine=None)))
    assert _fields(facts, "average_grade_pct") == {}
    assert _fields(facts, "climb_source") == {}


def test_no_grade_fact_when_the_row_has_no_grade():
    facts = _facts(_listing(_machine_row(average_grade_pct=None)))
    assert _fields(facts, "average_grade_pct") == {}
    assert _fields(facts, "climb_source") == {}
    assert _fields(facts, "fitness_machine")  # the machine is still stated


@pytest.mark.parametrize("bad", [True, "3.1", float("nan"), float("inf")])
def test_a_non_numeric_or_non_finite_grade_is_not_published(bad):
    facts = _facts(_listing(_machine_row(average_grade_pct=bad)))
    assert _fields(facts, "average_grade_pct") == {}


def test_an_outdoor_run_keeps_its_published_facts_and_gains_only_a_setting():
    outdoor = _machine_row(is_indoor=False, fitness_machine=None,
                           average_grade_pct=None)
    outdoor_before = {k: v for k, v in outdoor.items()
                      if k not in ("is_indoor", "fitness_machine",
                                   "average_grade_pct")}
    before = _facts(_listing(outdoor_before))
    after = _facts(_listing(outdoor))
    gained = set(after) - set(before)
    assert {after[k]["field"] for k in gained} == {"setting"}
    assert {k: after[k] for k in before} == before
    assert _fields(after, "average_grade_pct") == {}


def test_null_context_keys_leave_the_facts_identical():
    base = {
        "date": DAY, "type": "running", "workout_key": "k",
        "duration_min": 30.0, "distance_mi": 3.0, "max_heart_rate": 150.0,
        "pace_min_per_mi": 10.0, "start_time_local": "10:00"}
    with_nulls = dict(base, is_indoor=None, fitness_machine=None)
    assert _facts(_listing(with_nulls)) == _facts(_listing(base))
    assert _facts(_listing(base))  # and it is not vacuously empty


def test_an_unknown_machine_kind_is_published_generically_not_echoed():
    facts = _facts(_listing(_machine_row(fitness_machine="skierg")))
    fact = _fields(facts, "fitness_machine")[f"{DAY}|running"]
    assert fact["value"] == "other"
    assert fact["display"] == "a connected fitness machine"
    assert "skierg" not in repr(facts)


def test_context_never_makes_a_workout_appear_on_its_own():
    row = {"date": DAY, "type": "running", "workout_key": "k",
           "is_indoor": True, "fitness_machine": "treadmill",
           "average_grade_pct": 3.1}
    # average_grade_pct is itself a published figure, so use a row with none.
    row.pop("average_grade_pct")
    assert _facts(_listing(row)) == {}


def test_label_facts_are_not_counted_as_figures():
    from health_advisor import chat
    facts = _facts(_listing(_machine_row()))
    counters = chat._ask_fact_counters("how was my run?", facts, facts,
                                       {"placeholders": []}, ())
    figures = {k for k, f in facts.items() if f["field"] not in
               fact_template.WORKOUT_LABEL_FIELDS}
    assert counters["facts_offered"] == len(figures)
    assert fact_template.WORKOUT_LABEL_FIELDS >= {
        "date", "setting", "fitness_machine", "climb_source"}


def test_two_same_type_sessions_on_one_day_keep_their_own_context():
    a = _machine_row(workout_key="a", start_time_local="06:00",
                     fitness_machine="treadmill", average_grade_pct=2.0)
    b = _machine_row(workout_key="b", start_time_local="18:00",
                     is_indoor=False, fitness_machine=None,
                     average_grade_pct=None)
    facts = _facts(_listing(a, b))
    machines = _fields(facts, "fitness_machine")
    settings = _fields(facts, "setting")
    assert set(machines) == {f"{DAY}|running|06:00"}
    assert settings[f"{DAY}|running|06:00"]["value"] == "indoor"
    assert settings[f"{DAY}|running|18:00"]["value"] == "outdoor"


# --------------------------------------------------------------------------- #
# End to end: wire -> vault -> list_workouts -> facts
# --------------------------------------------------------------------------- #
def test_wire_to_facts_for_a_machine_session(conn, tools):
    parsed = hk_parse.parse_payload({
        "protocol_version": 1,
        "device": {"id": "d", "name": "n", "model": "m"},
        "app_version": "v", "batch_id": "b", "batch_sequence": 1,
        "sent_at": "2030-01-05T18:00:00Z", "anchors": [], "samples": [],
        "deletions": [],
        "workouts": [{
            "hk_uuid": "wire-1", "workout_activity_type":
                "HKWorkoutActivityTypeRunning",
            "start": f"{DAY}T15:00:00Z", "end": f"{DAY}T15:30:00Z",
            "duration_min": 30.0, "distance_mi": 3.0,
            "elevation_ascended_m": 150.0, "is_indoor": True,
            "fitness_machine":
                "com.apple.health.fitnessmachinemodel.treadmill",
            "source_revision": {"source_name": "S", "bundle_id": "b"}}],
    })
    db.insert_workouts(conn, parsed["workouts"])
    conn.commit()
    listed = _listed(tools)
    (row,) = listed["workouts"]
    assert (row["is_indoor"], row["fitness_machine"],
            row["average_grade_pct"]) == (True, "treadmill", 3.1)
    facts = _facts(copy.deepcopy(listed))
    assert _fields(facts, "average_grade_pct")[f"{DAY}|running"]["value"] == 3.1
