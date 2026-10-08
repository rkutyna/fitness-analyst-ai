"""A pace is a foot-travel reading: workout_focus and talking_points compute and
narrate one only for ``normalize.PACE_WORKOUT_TYPES`` (running, walking,
hiking), a speed only for cycling, and every other type that arrives with a
distance (a swim, a row, an uncategorised session) reports the distance and
duration and nothing that reads as a rate (consumer #589).

Each figure expected below is computed from the literal duration and distance
the test seeds, never from the code under test.
"""
import pytest

from health_advisor import analysis as A
from health_advisor import mcp_server as S
from health_advisor import normalize as nz
from tests.conftest import seed_workout

AS_OF = "2026-06-10"
DAY = "2026-06-10"


def _parts(wf, as_of=AS_OF):
    """The minimum briefing shape talking_points reads."""
    return {"coverage": [], "as_of": as_of, "readiness": {}, "trends": {},
            "training_load": {}, "movers": [], "highlights": [],
            "workout_focus": wf}


def _workout_seeds(wf):
    return [t for t in A.talking_points(_parts(wf)) if t["topic"] == "workout"]


# --- a type that is not foot travel or a ride gets no rate ---------------------

@pytest.mark.parametrize("kind", ["swimming", "rowing", "other"])
@pytest.mark.parametrize("metric_units", [False, True])
def test_a_non_foot_non_cycling_workout_with_a_distance_gets_no_pace_or_speed(
        conn, kind, metric_units):
    seed_workout(conn, kind, DAY, duration_min=40.0, distance_mi=0.9)
    wf = A.workout_focus(conn, AS_OF, metric_units=metric_units)
    assert wf["type"] == kind
    pace_key = "pace_min_per_km" if metric_units else "pace_min_per_mi"
    speed_key = "speed_kph" if metric_units else "speed_mph"
    jog_key = "jog_pace_min_per_km" if metric_units else "jog_pace_min_per_mi"
    assert wf[pace_key] is None
    assert wf[speed_key] is None
    assert wf[jog_key] is None
    assert wf["pace_label"] is None
    # The distance and the duration are still reported.
    assert wf["duration_min"] == 40.0
    assert wf["distance_km" if metric_units else "distance_mi"] == (
        1.4 if metric_units else 0.9)


@pytest.mark.parametrize("kind", ["swimming", "rowing", "other"])
def test_talking_points_says_the_distance_of_a_non_foot_workout_with_no_pace(
        conn, kind):
    seed_workout(conn, kind, DAY, duration_min=40.0, distance_mi=0.9)
    seeds = _workout_seeds(A.workout_focus(conn, AS_OF))
    assert seeds == [{"topic": "workout",
                      "seed": f"{kind} workout today: 0.9 mi",
                      "numbers": [0.9]}]
    text = seeds[0]["seed"]
    for word in ("pace", "min/mi", "min/km", "mph", "kph", "blended"):
        assert word not in text


def test_the_distance_only_seed_is_dated_when_the_workout_is_not_today(conn):
    seed_workout(conn, "swimming", "2026-06-09", duration_min=40.0,
                 distance_mi=0.9)
    seeds = _workout_seeds(A.workout_focus(conn, AS_OF))
    assert [s["seed"] for s in seeds] == [
        "swimming workout on 2026-06-09 (not today): 0.9 mi"]


def test_a_days_jog_pace_is_not_attached_to_a_swim(conn, monkeypatch):
    """The day's jog pace belongs to the day's foot travel, not to a swim that
    happened to share the date."""
    monkeypatch.setattr(
        A, "impact_volume",
        lambda *a, **k: [{"jog_pace_min_per_mi": 12.0,
                          "jog_pace_min_per_km": 7.5}])
    seed_workout(conn, "swimming", DAY, duration_min=40.0, distance_mi=0.9)
    assert A.workout_focus(conn, AS_OF)["jog_pace_min_per_mi"] is None
    assert "jog pace" not in _workout_seeds(A.workout_focus(conn, AS_OF))[0]["seed"]


def test_a_workout_with_no_distance_still_says_nothing(conn):
    seed_workout(conn, "swimming", DAY, duration_min=40.0, distance_mi=None)
    assert _workout_seeds(A.workout_focus(conn, AS_OF)) == []


def test_the_recent_effort_nudge_for_a_swim_states_a_distance_not_a_pace(conn):
    seed_workout(conn, "swimming", DAY, duration_min=40.0, distance_mi=0.9)
    wf = A.workout_focus(conn, AS_OF)
    out = A.suggestions({"band": "strong"},
                        {"acwr": 1.0, "acwr_band": "optimal"}, wf)
    assert [s["because"] for s in out] == ["recent 0.9 mi effort; ACWR 1.0 (optimal)"]
    assert "pace" not in " ".join(s["text"] for s in out)


# --- foot travel keeps its blended pace, byte for byte -------------------------

@pytest.mark.parametrize("kind,duration,distance,expected", [
    ("running", 50.0, 4.0, 12.5),
    ("walking", 60.0, 3.0, 20.0),
    ("hiking", 90.0, 3.0, 30.0),
])
def test_foot_travel_keeps_its_blended_pace(conn, kind, duration, distance,
                                            expected):
    seed_workout(conn, kind, DAY, duration_min=duration, distance_mi=distance)
    wf = A.workout_focus(conn, AS_OF)
    assert wf["pace_min_per_mi"] == expected
    assert wf["pace_label"] == "blended"
    assert wf["speed_mph"] is None
    assert wf["distance_mi"] == distance
    assert _workout_seeds(wf) == [{
        "topic": "workout",
        "seed": f"{kind} workout today: {distance} mi at blended pace "
                f"{expected} min/mi",
        "numbers": [distance, expected]}]


def test_foot_travel_pace_in_metric_units_is_the_converted_pace(conn):
    seed_workout(conn, "running", DAY, duration_min=50.0, distance_mi=4.0)
    wf = A.workout_focus(conn, AS_OF, metric_units=True)
    assert wf["pace_min_per_km"] == pytest.approx(12.5 / 1.609344, abs=0.05)
    assert "pace_min_per_mi" not in wf


def test_a_foot_type_is_matched_whatever_its_case(conn):
    seed_workout(conn, "Running", DAY, duration_min=50.0, distance_mi=4.0)
    assert A.workout_focus(conn, AS_OF)["pace_min_per_mi"] == 12.5


# --- a ride keeps its speed -----------------------------------------------------

@pytest.mark.parametrize("kind,duration,distance,expected", [
    ("cycling", 48.0, 12.0, 15.0),
    ("hand_cycling", 30.0, 5.0, 10.0),
])
def test_a_ride_keeps_its_speed_and_gets_no_pace(conn, kind, duration,
                                                 distance, expected):
    seed_workout(conn, kind, DAY, duration_min=duration, distance_mi=distance)
    wf = A.workout_focus(conn, AS_OF)
    assert wf["speed_mph"] == expected
    assert wf["pace_min_per_mi"] is None
    assert wf["pace_label"] is None
    assert wf["jog_pace_min_per_mi"] is None
    assert _workout_seeds(wf) == [{
        "topic": "workout",
        "seed": f"{kind} workout today: {distance} mi at {expected} mph",
        "numbers": [distance, expected]}]


# --- one definition of "types that get a pace" ---------------------------------

def test_the_tool_surface_and_workout_focus_share_one_pace_set():
    assert S._PACE_WORKOUT_TYPES is nz.PACE_WORKOUT_TYPES
    assert nz.PACE_WORKOUT_TYPES == frozenset({"running", "walking", "hiking"})


def test_list_workouts_and_workout_focus_agree_on_which_types_get_a_pace(
        conn, tools, vault):
    kinds = sorted(nz.PACE_WORKOUT_TYPES) + ["swimming", "rowing", "other",
                                             "cycling"]
    for kind in kinds:
        conn.execute("DELETE FROM workouts")
        conn.commit()
        seed_workout(conn, kind, DAY, duration_min=60.0, distance_mi=3.0)
        listed = tools.list_workouts(start=DAY, end=DAY)["workouts"]
        assert len(listed) == 1
        focus = A.workout_focus(conn, AS_OF)
        in_set = kind in nz.PACE_WORKOUT_TYPES
        assert ("pace_min_per_mi" in listed[0]) is in_set, kind
        assert (focus["pace_min_per_mi"] is not None) is in_set, kind
