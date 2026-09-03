"""Workout arbitration resolves device roles without schema state."""
from __future__ import annotations

import sqlite3

from health_advisor import db
from health_advisor import normalize as nz


def _record(source: str, value: float, key: str) -> dict:
    return {
        "metric": "distance_walking_running",
        "value": value,
        "unit": "mi",
        "start_utc": "2026-08-26T12:00:00+00:00",
        "end_utc": "2026-08-26T12:00:20+00:00",
        "start_local": "2026-08-26 12:00:00",
        "local_date": "2026-08-26",
        "source": source,
        "origin": "test",
        "dedupe_key": key,
    }


def test_source_role_normalizes_nbsp_and_whitespace():
    assert nz.workout_source_role("Demo\xa0Watch  ") == "watch"
    assert nz.workout_source_role("Demo\xa0Phone\t") == "iphone"


def test_arbitration_works_on_plain_sqlite_without_role_column(tmp_path):
    path = tmp_path / "plain.db"
    conn = sqlite3.connect(path)
    db.init_db(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
    assert "source_role" not in columns
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("running", "2026-08-26T12:00:00+00:00",
         "2026-08-26T13:00:00+00:00", "2026-08-26", 60.0,
         "test", "plain-connection-workout"),
    )
    db.insert_records(conn, [
        _record("Demo\xa0Watch  ", 0.01, "plain-watch"),
        _record("Demo Phone", 0.02, "plain-phone"),
    ])
    conn.commit()

    clause, args = db._workout_arbitration(conn, "distance_walking_running")
    kept = conn.execute(
        "SELECT source FROM records WHERE metric = ?" + clause + " ORDER BY source",
        ("distance_walking_running", *args),
    ).fetchall()
    assert [row[0] for row in kept] == ["Demo\xa0Watch  "]
    conn.close()


def test_source_role_vocabulary():
    assert nz.workout_source_role("Demo Watch") == "watch"
    assert nz.workout_source_role("Demo Phone") == "iphone"
    assert nz.workout_source_role("Demo Scale") == "scale"
    assert nz.workout_source_role("Sync Solver") == "mirror"
    assert nz.workout_source_role("GymKit") == "gymkit"
    assert nz.workout_source_role("Demo Watch|Demo Scale") is None
    assert nz.workout_source_role("") is None
