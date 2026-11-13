"""A swim's distance reads in metres, in every unit system.

The client sends every workout distance in miles, so a pool swim is stored in
miles. ``list_workouts`` publishes a swim's distance as ``distance_m``, whole
metres rounded half up from the stored miles, in place of ``distance_mi`` /
``distance_km``; every other workout type is unchanged. The closed workout fact
set publishes ``distance_m`` with unit ``m``, so a narration of the swim in
miles has no fact to stand on.

Synthetic swims: 1,000 yd (914.4 m) and 1,500 m, 30 minutes each, stored as the
client sends them (unrounded miles) and to three decimals (0.568 / 0.932 mi).
Oracles worked by hand: 914.4 -> 914; 0.568 mi = 914.107 m -> 914;
1,500 m -> 1500; 0.932 mi = 1499.909 m -> 1500.
"""
from __future__ import annotations

import pytest

from health_advisor import fact_template
from health_advisor import vault as vault_mod

METRES_PER_MILE = 1609.344
YARD_MI = 0.9144 / METRES_PER_MILE
DAY_YD, DAY_M, DAY_RIDE, DAY_RUN = "2030-01-07", "2030-01-08", "2030-01-09", "2030-01-10"


@pytest.mark.parametrize("miles, metres", [
    (1000 * YARD_MI, 914), (0.568, 914), (1500 / METRES_PER_MILE, 1500), (0.932, 1500),
    (None, None)])
def test_swim_distance_m_is_whole_metres_half_up(miles, metres):
    assert vault_mod.swim_distance_m(miles) == metres


def _seed(conn, rows):
    for day, kind, miles in rows:
        start, end = f"{day}T12:00:00Z", f"{day}T12:30:00Z"
        conn.execute(
            "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
            "duration_min, energy_kcal, distance_mi, unit_distance, source, dedupe_key) "
            "VALUES (?, ?, ?, ?, 30.0, 250.0, ?, 'mi', 'test', ?)",
            (kind, start, end, day, miles, f"{kind}|{start}|{end}|test"))
    conn.commit()


ROWS = [(DAY_YD, "swimming", 1000 * YARD_MI), (DAY_M, "swimming", 1500 / METRES_PER_MILE),
        (DAY_RIDE, "cycling", 12.4), (DAY_RUN, "running", 3.1)]


def _listed(conn, tools, system):
    _seed(conn, ROWS)
    if system is not None:
        vault_mod.set_unit_system(conn, system)
        conn.commit()
    listed = tools.list_workouts(start=DAY_YD, end=DAY_RUN)
    return {row["date"]: row for row in listed["workouts"]}, listed


@pytest.mark.parametrize("system", [None, "imperial", "metric"])
def test_list_workouts_shows_each_swim_in_metres(conn, tools, system):
    rows, _listed_out = _listed(conn, tools, system)
    for day, metres in ((DAY_YD, 914), (DAY_M, 1500)):
        row = rows[day]
        assert row["distance_m"] == metres
        assert "distance_mi" not in row and "distance_km" not in row
        assert not any(key.startswith(("pace_", "speed_")) for key in row)


def test_rides_and_runs_keep_their_units(conn, tools):
    rows, _ = _listed(conn, tools, None)
    assert rows[DAY_RIDE]["distance_mi"] == 12.4 and "distance_m" not in rows[DAY_RIDE]
    assert rows[DAY_RUN]["distance_mi"] == 3.1 and rows[DAY_RUN]["pace_min_per_mi"] == 9.68


def test_rides_and_runs_keep_km_on_a_metric_vault(conn, tools):
    rows, _ = _listed(conn, tools, "metric")
    assert rows[DAY_RIDE]["distance_km"] == 19.96 and "distance_m" not in rows[DAY_RIDE]
    assert rows[DAY_RUN]["distance_km"] == 4.99


@pytest.mark.parametrize("system", [None, "metric"])
def test_the_longest_swim_fact_is_in_metres(conn, tools, system):
    _rows, listed = _listed(conn, tools, system)
    facts = {fact["id"]: fact for fact in listed["publishable_facts"]}
    window = f"workouts/{DAY_YD}:{DAY_RUN}"
    longest = facts[f"{window}/swimming/longest_distance"]
    assert (longest["value"], longest["unit"]) == (1500, "m")
    ride = facts[f"{window}/cycling/longest_distance"]
    assert ride["unit"] == ("km" if system == "metric" else "mi")


def test_the_closed_fact_set_carries_the_swim_in_metres(conn, tools):
    _rows, listed = _listed(conn, tools, None)
    facts = fact_template.build_workout_facts([{
        "sequence": 1, "tool_name": "list_workouts", "result_elided": False,
        "result": listed}])
    swims = sorted((fact["value"], fact["unit"]) for fact in facts.values()
                   if fact["field"].startswith("distance") and "swimming" in fact["workout"])
    assert swims == [(914, "m"), (1500, "m")]
    assert not [fact for fact in facts.values()
                if "swimming" in fact["workout"] and fact["unit"] in ("mi", "km")]
