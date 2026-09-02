"""Shared loaders for committed, compressed live-data fixtures."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from health_advisor import db


FIXTURE = Path(__file__).resolve().parent / "fixtures" / \
    "f82_multi_source_live_20260828.json.gz"
F84_FIXTURE = Path(__file__).resolve().parent / "fixtures" / \
    "f84_week8_wholeday_live_20260828.json.gz"


def load_f82_multi_source(conn, path: Path = FIXTURE) -> dict:
    """Load the F-82 fixture into the production workouts/records schema.

    ``workout_id`` is extraction provenance and is dropped from records.
    Records go through ``db.insert_records`` so the real dedupe key is used;
    the overlapping June windows therefore exercise production upsert behavior.
    Fixture workout ids are retained for stable session-window assertions.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    workout_columns = (
        "id", "workout_type", "start_utc", "end_utc", "local_date",
        "duration_min", "energy_kcal", "distance_mi", "unit_distance",
        "source", "route_ref", "dedupe_key", "avg_heart_rate",
        "max_heart_rate", "hk_uuid",
    )
    placeholders = ",".join("?" * len(workout_columns))
    conn.executemany(
        f"INSERT INTO workouts ({','.join(workout_columns)}) "
        f"VALUES ({placeholders})",
        ([workout.get(column) for column in workout_columns]
         for workout in payload["workouts"]),
    )

    records = []
    for record in payload["records"]:
        row = {**record, "start_local": None}
        row.pop("workout_id", None)
        records.append(row)
    db.insert_records(conn, records)
    conn.commit()
    return payload


def load_f84_wholeday(conn, path: Path = F84_FIXTURE) -> dict:
    """Load the complete week-8 whole-day fixture into the production schema.

    Workout ids are retained so the fixture remains useful for session-window
    assertions. Record ``workout_id`` is extraction provenance rather than a
    production column and is dropped before the real insert/upsert path.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    workout_columns = (
        "id", "workout_type", "start_utc", "end_utc", "local_date",
        "duration_min", "energy_kcal", "distance_mi", "unit_distance",
        "source", "route_ref", "dedupe_key", "avg_heart_rate",
        "max_heart_rate", "hk_uuid",
    )
    placeholders = ",".join("?" * len(workout_columns))
    conn.executemany(
        f"INSERT INTO workouts ({','.join(workout_columns)}) "
        f"VALUES ({placeholders})",
        ([workout.get(column) for column in workout_columns]
         for workout in payload["workouts"]),
    )

    records = []
    for record in payload["records"]:
        row = {**record, "start_local": None}
        row.pop("workout_id", None)
        records.append(row)
    db.insert_records(conn, records)
    conn.commit()
    return payload
