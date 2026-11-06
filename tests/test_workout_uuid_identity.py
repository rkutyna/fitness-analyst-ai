"""A HealthKit UUID is the identity of a workout; its type is a name (consumer #713).

The phone sent 71 HealthKit activity types as HKWorkoutActivityTypeOther and a
newer build names them. A re-sent session then hashes to a different
dedupe_key, and the INSERT used to hit the hk_uuid unique index and roll back
the whole batch. These tests pin the rules: the stored key never changes, a
named type replaces a stored one, `other` never replaces a name, and a UUID
match outranks a key match on a UUID-less export row.

Everything here is synthetic; the engine repository is public.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, receiver
from tests.conftest import INGEST_HEADERS, INGEST_TEST_SECRET


UUID = "synthetic-uuid-1"
START = "2030-01-05T15:00:00+00:00"
END = "2030-01-05T15:30:00+00:00"
SOURCE = "Synthetic Source"


def _w(**over):
    row = dict(
        workout_type="other",
        start_utc=START,
        end_utc=END,
        local_date="2030-01-05",
        duration_min=30.0, energy_kcal=None, distance_mi=None,
        unit_distance=None, source=SOURCE, route_ref=None,
        avg_heart_rate=None, max_heart_rate=None,
        elevation_ascended_m=None, elevation_descended_m=None,
        elevation_source=None, is_indoor=None, fitness_machine=None,
        hk_uuid=UUID,
    )
    row.update(over)
    row["dedupe_key"] = db.workout_key(
        row["workout_type"], row["start_utc"], row["end_utc"])
    return row


def _rows(conn):
    return conn.execute(
        "SELECT * FROM workouts ORDER BY id").fetchall()


def test_a_re_sent_session_under_a_name_merges_instead_of_raising(conn):
    """Stored `other`, re-sent `cooldown`, same UUID: the batch survives."""
    assert db.insert_workouts(conn, [_w(workout_type="other")]) == 1
    # Before the fix this raised sqlite3.IntegrityError on workouts.hk_uuid.
    assert db.insert_workouts(conn, [_w(workout_type="cooldown")]) == 0

    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 1
    (r,) = _rows(conn)
    assert r["workout_type"] == "cooldown"
    assert r["dedupe_key"] == db.workout_key("other", START, END), \
        "the key is the first sighting's, and never changes"
    assert r["hk_uuid"] == UUID


def test_an_old_other_re_send_does_not_erase_the_name(conn):
    """Stored `cooldown`, an old build re-sends `other`: the name stays."""
    assert db.insert_workouts(conn, [_w(workout_type="cooldown")]) == 1
    assert db.insert_workouts(conn, [_w(workout_type="other")]) == 0

    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 1
    (r,) = _rows(conn)
    assert r["workout_type"] == "cooldown"
    assert r["dedupe_key"] == db.workout_key("cooldown", START, END)


def test_fill_the_holes_still_applies_across_the_rename(conn):
    """A NULL stored energy fills from the re-send across the rename."""
    assert db.insert_workouts(conn, [_w(workout_type="other", energy_kcal=None)]) == 1
    assert db.insert_workouts(conn, [_w(workout_type="cooldown", energy_kcal=50.0)]) == 0
    (r,) = _rows(conn)
    assert r["energy_kcal"] == 50.0


def test_a_stored_value_is_not_clobbered_across_the_rename(conn):
    """A stored energy of 40.0 stays 40.0 when the renamed re-send says 50.0."""
    assert db.insert_workouts(conn, [_w(workout_type="other", energy_kcal=40.0)]) == 1
    assert db.insert_workouts(conn, [_w(workout_type="cooldown", energy_kcal=50.0)]) == 0
    (r,) = _rows(conn)
    assert r["energy_kcal"] == 40.0, "fill the holes, never clobber"
    assert r["workout_type"] == "cooldown"


def test_a_session_mark_survives_the_rename(conn):
    """The marks table holds the key; a rename must not orphan or break it."""
    db.insert_workouts(conn, [_w(workout_type="other")])
    key = db.workout_key("other", START, END)
    db.mark_workout_not_a_session(conn, key, source="synthetic")

    assert db.insert_workouts(conn, [_w(workout_type="cooldown")]) == 0

    marks = conn.execute("SELECT workout_key FROM workout_session_marks").fetchall()
    assert len(marks) == 1
    (r,) = _rows(conn)
    assert marks[0]["workout_key"] == r["dedupe_key"]
    assert r["workout_type"] == "cooldown"


def test_a_uuid_match_outranks_a_key_match_on_a_legacy_row(conn):
    """An export row (no UUID) owns the cooldown key; the phone row owns the UUID."""
    export_row = _w(workout_type="cooldown", source="Synthetic Export", hk_uuid=None)
    phone_row = _w(workout_type="other", source=SOURCE)
    assert db.insert_workouts(conn, [export_row]) == 1
    assert db.insert_workouts(conn, [phone_row]) == 1

    assert db.insert_workouts(conn, [_w(workout_type="cooldown")]) == 0

    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 2
    by_uuid = conn.execute(
        "SELECT * FROM workouts WHERE hk_uuid = ?", (UUID,)).fetchone()
    assert by_uuid["workout_type"] == "cooldown"
    assert by_uuid["dedupe_key"] == db.workout_key("other", START, END)
    legacy = conn.execute(
        "SELECT * FROM workouts WHERE source = 'Synthetic Export'").fetchone()
    assert legacy["hk_uuid"] is None
    assert legacy["workout_type"] == "cooldown"
    assert legacy["dedupe_key"] == db.workout_key("cooldown", START, END)


def test_the_receiver_accepts_a_renamed_re_send(vault, monkeypatch):
    """End to end: two batches, the second names the type; both land 200."""
    monkeypatch.setattr(receiver, "SHARED_SECRET", INGEST_TEST_SECRET)
    client = TestClient(receiver.create_app(vault), headers=INGEST_HEADERS)

    def payload(batch_id, sequence, activity):
        return {
            "protocol_version": 1,
            "device": {"id": "synthetic-device", "name": "synthetic-phone",
                       "model": "synthetic-model"},
            "app_version": "synthetic-version",
            "batch_id": batch_id,
            "batch_sequence": sequence,
            "sent_at": "2030-01-05T18:00:00Z",
            "anchors": [], "samples": [], "deletions": [],
            "workouts": [{
                "hk_uuid": UUID,
                "workout_activity_type": activity,
                "start": "2030-01-05T15:00:00Z",
                "end": "2030-01-05T15:30:00Z",
                "duration_min": 30.0,
                "source_revision": {"source_name": SOURCE,
                                    "bundle_id": "synthetic.bundle"},
            }],
        }

    with client:
        first = client.post("/v1/ingest", json=payload(
            "synthetic-batch-1", 1, "HKWorkoutActivityTypeOther"))
        second = client.post("/v1/ingest", json=payload(
            "synthetic-batch-2", 2, "HKWorkoutActivityTypeCooldown"))
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    conn = vault.read_only()
    try:
        rows = conn.execute("SELECT workout_type FROM workouts").fetchall()
    finally:
        conn.close()
    assert [r["workout_type"] for r in rows] == ["cooldown"]
