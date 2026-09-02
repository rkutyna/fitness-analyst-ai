"""A second sighting of a workout must fill its holes, not be thrown away.

Audit P3-5: `insert_workouts` was `INSERT OR IGNORE`, so whichever ingest path
saw a session first owned every column of it forever. The two paths populate
DISJOINT fields — the export XML carries duration/energy/distance but no route
and often no HR; the retired phone path carried the route, the HR summary and
the per-second samples — so on the live DB 638 of 785 workouts have a NULL
avg_heart_rate and 410 have a NULL route_ref, with the missing halves sitting
unused in the other source. The historical capture-reingest helper is retired
with that phone path.

The merge rule is fill-the-holes: an existing non-NULL value is never
overwritten by a later arrival. See the commit message for why that direction
and not the other.
"""
from __future__ import annotations

import pytest

from health_advisor import db


def _w(**over):
    row = dict(
        workout_type="running",
        start_utc="2026-06-22T14:24:24+00:00",
        end_utc="2026-06-22T15:00:25+00:00",
        local_date="2026-06-22",
        duration_min=36.02, energy_kcal=None, distance_mi=None,
        unit_distance=None, source="", route_ref=None,
        avg_heart_rate=None, max_heart_rate=None,
    )
    row.update(over)
    row["dedupe_key"] = db.workout_key(
        row["workout_type"], row["start_utc"], row["end_utc"])
    return row


def _row(conn):
    return conn.execute("SELECT * FROM workouts").fetchone()


def test_second_path_fills_the_columns_the_first_left_null(conn):
    """The real case: export.zip lands duration/energy, HAE lands route + HR."""
    assert db.insert_workouts(conn, [_w(energy_kcal=313.9, source="Demo's Apple Watch")]) == 1
    assert db.insert_workouts(conn, [_w(
        route_ref="2026-06-22T142424Z.gpx", avg_heart_rate=136.6,
        max_heart_rate=171.0, distance_mi=2.017, unit_distance="mi",
        source="GymKit")]) == 0, "a merge is not a new workout"

    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 1
    r = _row(conn)
    assert r["route_ref"] == "2026-06-22T142424Z.gpx"
    assert r["avg_heart_rate"] == 136.6
    assert r["max_heart_rate"] == 171.0
    assert r["distance_mi"] == 2.017
    assert r["unit_distance"] == "mi"
    assert r["energy_kcal"] == 313.9, "the first path's value must survive"


def test_it_fills_in_the_other_arrival_order_too(conn):
    assert db.insert_workouts(conn, [_w(route_ref="r.gpx", avg_heart_rate=136.6)]) == 1
    assert db.insert_workouts(conn, [_w(energy_kcal=313.9, distance_mi=2.017)]) == 0
    r = _row(conn)
    assert (r["route_ref"], r["avg_heart_rate"]) == ("r.gpx", 136.6)
    assert (r["energy_kcal"], r["distance_mi"]) == (313.9, 2.017)


@pytest.mark.parametrize("col,first,second", [
    ("avg_heart_rate", 152.3, 128.6),
    ("max_heart_rate", 184.0, 134.0),
    ("energy_kcal", 313.9, 300.0),
    ("distance_mi", 2.017, 1.9),
    ("duration_min", 36.02, 30.0),
    ("route_ref", "real.gpx", "other.gpx"),
])
def test_an_existing_value_is_never_clobbered(conn, col, first, second):
    """Direction matters. reconcile_workout_heart_rate rewrites avg/max HR from
    the raw sample series; a later export.zip carrying the device's own (wrong)
    summary must not undo that."""
    db.insert_workouts(conn, [_w(**{col: first})])
    db.insert_workouts(conn, [_w(**{col: second})])
    assert _row(conn)[col] == first


def test_a_non_empty_source_is_not_replaced(conn):
    """`source` is deliberately outside the dedupe key because the two paths
    name one session differently; first non-empty wins, as before."""
    db.insert_workouts(conn, [_w(source="Demo's Apple Watch")])
    db.insert_workouts(conn, [_w(source="GymKit")])
    assert _row(conn)["source"] == "Demo's Apple Watch"


def test_an_empty_source_is_filled(conn):
    db.insert_workouts(conn, [_w(source="")])
    db.insert_workouts(conn, [_w(source="GymKit")])
    assert _row(conn)["source"] == "GymKit"


def test_a_different_session_is_still_a_new_row(conn):
    db.insert_workouts(conn, [_w()])
    assert db.insert_workouts(conn, [_w(workout_type="cycling")]) == 1
    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 2


def test_added_count_excludes_merges(conn):
    """`workouts_added` is reported to the phone and logged; a merge must not
    inflate it (total_changes would count the UPDATE)."""
    assert db.insert_workouts(conn, [_w(), _w(workout_type="cycling")]) == 2
    assert db.insert_workouts(conn, [_w(energy_kcal=1.0)]) == 0
    assert db.insert_workouts(conn, [_w(energy_kcal=1.0), _w(workout_type="walking")]) == 1
