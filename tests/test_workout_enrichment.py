"""The workouts heart-rate columns and their migration."""
import sqlite3

from health_advisor import db


def test_migration_adds_hr_columns_to_existing_table(tmp_path):
    # Simulate a pre-existing workouts table without the HR columns.
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE workouts (id INTEGER PRIMARY KEY, workout_type TEXT, "
              "start_utc TEXT, end_utc TEXT, local_date TEXT, dedupe_key TEXT UNIQUE)")
    c.commit()
    c.close()
    conn = db.connect(p)
    db.init_db(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(workouts)")}
    assert {"avg_heart_rate", "max_heart_rate"} <= cols
    conn.close()


def test_insert_workout_round_trips_heart_rate(conn):
    rows = [{
        "workout_type": "running", "start_utc": "2026-06-15T13:17:35+00:00",
        "end_utc": "2026-06-15T13:28:35+00:00", "local_date": "2026-06-15",
        "duration_min": 11.0, "energy_kcal": 141.3, "distance_mi": 1.01,
        "unit_distance": "mi", "source": "Watch", "route_ref": "run.gpx",
        "avg_heart_rate": 166.2, "max_heart_rate": 190.0,
        "dedupe_key": "k1",
    }]
    db.insert_workouts(conn, rows)
    conn.commit()
    r = conn.execute("SELECT avg_heart_rate, max_heart_rate, route_ref FROM workouts "
                     "WHERE dedupe_key='k1'").fetchone()
    assert abs(r["avg_heart_rate"] - 166.2) < 0.1
    assert abs(r["max_heart_rate"] - 190.0) < 0.1
    assert r["route_ref"] == "run.gpx"


def test_insert_workout_without_hr_keys_still_works(conn):
    # Backfill rows don't carry HR keys; insert must tolerate their absence.
    rows = [{
        "workout_type": "walking", "start_utc": "2020-01-01T00:00:00+00:00",
        "end_utc": "2020-01-01T01:00:00+00:00", "local_date": "2020-01-01",
        "duration_min": 60.0, "energy_kcal": 100.0, "distance_mi": 2.0,
        "unit_distance": "mi", "source": "Watch", "route_ref": None,
        "dedupe_key": "k2",
    }]
    assert db.insert_workouts(conn, rows) == 1
    conn.commit()
    r = conn.execute("SELECT avg_heart_rate FROM workouts WHERE dedupe_key='k2'").fetchone()
    assert r["avg_heart_rate"] is None
