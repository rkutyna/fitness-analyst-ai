"""A swim's distance reads in metres, in every unit system.

The client sends every workout distance in miles, so a pool swim is stored in
miles. ``list_workouts`` publishes a swim's distance as ``distance_m``, whole
metres rounded half up from the stored miles, in place of ``distance_mi`` /
``distance_km``; every other workout type is unchanged. The closed workout fact
set publishes ``distance_m`` with unit ``m``, so a narration of the swim in
miles has no fact to stand on.

Synthetic swims, stored as the client sends them: unrounded miles, which the
ingest stores as sent (pinned below through the production ingest), so a pool
distance comes back as itself. Miles rounded to two or three decimals would
not (1,000 m stored as 0.621 mi is 999 m); no live path rounds them.
Oracles worked by hand: 1,000 yd = 914.4 m -> 914; 1,650 yd = 1508.76 m -> 1509.
"""
from __future__ import annotations

import pytest

from health_advisor import db, fact_template, mcp_server
from health_advisor import vault as vault_mod

METRES_PER_MILE = 1609.344
YARD_MI = 0.9144 / METRES_PER_MILE
DAY_YD, DAY_M, DAY_RIDE, DAY_RUN = "2030-01-07", "2030-01-08", "2030-01-09", "2030-01-10"


@pytest.mark.parametrize("miles, metres", [
    *[(m / METRES_PER_MILE, m) for m in (25, 50, 400, 1000, 1500, 3800)],
    (1000 * YARD_MI, 914), (1650 * YARD_MI, 1509), (None, None)])
def test_unrounded_stored_miles_give_back_the_pool_distance(miles, metres):
    assert vault_mod.swim_distance_m(miles) == metres


def test_the_ingest_stores_the_miles_it_was_sent_unrounded(vault, vault_path, monkeypatch):
    """The metres depend on it: a 1,650 yd swim sent as unrounded miles is
    stored bit for bit and listed as 1509 m."""
    from tests.test_hk_ingest import _client, _payload, _workout

    sent = 1650 * YARD_MI
    swim = {**_workout(), "workout_activity_type": "HKWorkoutActivityTypeSwimming",
            "distance_mi": sent}
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(batch_id="swim", workouts=[swim]),
                               headers={"x-health-secret": "hk-secret"})
    assert response.status_code == 200, response.text
    conn = db.connect(vault_path, read_only=True)
    try:
        (stored,) = conn.execute("SELECT distance_mi FROM workouts").fetchone()
    finally:
        conn.close()
    assert stored == sent
    listed = mcp_server.list_workouts(vault, start="2026-08-22", end="2026-08-22")
    assert [row["distance_m"] for row in listed["workouts"]] == [1509]


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


# --------------------------------------------------------------------------- #
# get_briefing: the workout focus, its talking point and its suggestion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("system", [None, "imperial", "metric"])
def test_get_briefing_hands_the_model_a_swim_in_metres(conn, vault, system):
    _seed(conn, [(DAY_YD, "swimming", 1000 * YARD_MI)])
    if system is not None:
        vault_mod.set_unit_system(conn, system)
        conn.commit()
    got = mcp_server.get_briefing(vault, scope="deep", day=DAY_YD)
    focus = got["workout_focus"]
    assert focus["distance_m"] == 914
    assert not {"distance_mi", "distance_km"} & set(focus)
    (point,) = [tp for tp in got["talking_points"] if tp["topic"] == "workout"]
    assert point == {"topic": "workout", "seed": "swimming workout today: 914 m",
                     "numbers": [914]}
    because = [s["because"] for s in got["suggestions"] if "effort" in s["because"]]
    # Where a next-day suggestion was given before (not metric), it is in metres.
    assert because == ([] if system == "metric" else ["recent 914 m effort"])


def test_a_ride_in_the_briefing_keeps_its_miles(conn, vault):
    _seed(conn, [(DAY_RIDE, "cycling", 12.4)])
    got = mcp_server.get_briefing(vault, scope="deep", day=DAY_RIDE)
    assert got["workout_focus"]["distance_mi"] == 12.4
    assert "distance_m" not in got["workout_focus"]
    (point,) = [tp for tp in got["talking_points"] if tp["topic"] == "workout"]
    assert point["seed"].startswith("cycling workout today: 12.4 mi at ")
