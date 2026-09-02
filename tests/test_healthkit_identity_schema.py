"""M1 slice 1: HealthKit identity and sync-state storage only."""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import db
from health_advisor import vault


HK_UUID = "hk-sleep-uuid-1"


def _record(metric: str, dedupe_key: str, *, hk_uuid: str = HK_UUID) -> dict:
    return {
        "metric": metric,
        "value": 1.0,
        "unit": "min",
        "start_utc": "2026-08-20T01:00:00+00:00",
        "end_utc": "2026-08-20T01:30:00+00:00",
        "start_local": "2026-08-19 21:00:00",
        "local_date": "2026-08-19",
        "source": "HealthKit",
        "origin": "receiver",
        "dedupe_key": dedupe_key,
        "hk_uuid": hk_uuid,
        "hk_type_identifier": "HKCategoryTypeIdentifierSleepAnalysis",
        "source_revision_json": '{"version":"1"}',
        "hk_device_id": "watch-a",
    }


def _workout(*, dedupe_key: str, hk_uuid: str | None = None) -> dict:
    return {
        "workout_type": "running",
        "start_utc": "2026-08-20T12:00:00+00:00",
        "end_utc": "2026-08-20T13:00:00+00:00",
        "local_date": "2026-08-20",
        "duration_min": 60.0,
        "energy_kcal": None,
        "distance_mi": None,
        "unit_distance": "mi",
        "source": "HealthKit",
        "dedupe_key": dedupe_key,
        "hk_uuid": hk_uuid,
    }


def _index(conn, table: str, name: str):
    return conn.execute(
        f"SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (name,),
    ).fetchone()[0]


def test_hk_columns_are_nullable_and_indexes_are_partial(conn):
    record_info = {
        row["name"]: row for row in conn.execute("PRAGMA table_info(records)")
    }
    workout_info = {
        row["name"]: row for row in conn.execute("PRAGMA table_info(workouts)")
    }
    assert all(record_info[name]["notnull"] == 0 for name in (
        "hk_uuid", "hk_type_identifier", "source_revision_json", "hk_device_id",
    ))
    assert workout_info["hk_uuid"]["notnull"] == 0
    assert record_info["dedupe_key"]["notnull"] == 1

    assert "(metric, hk_uuid)" in _index(
        conn, "records", "idx_records_metric_hk_uuid"
    )
    assert "WHERE hk_uuid IS NOT NULL" in _index(
        conn, "records", "idx_records_metric_hk_uuid"
    )
    assert "(hk_uuid)" in _index(conn, "workouts", "idx_workouts_hk_uuid")
    assert "WHERE hk_uuid IS NOT NULL" in _index(
        conn, "workouts", "idx_workouts_hk_uuid"
    )


def test_one_sleep_uuid_expands_into_two_canonical_rows(conn):
    db.insert_records(conn, [
        _record("sleep_asleep", "sleep-asleep-key"),
        _record("sleep_core", "sleep-core-key"),
    ])

    rows = conn.execute(
        "SELECT metric, hk_uuid FROM records WHERE hk_uuid = ? ORDER BY metric",
        (HK_UUID,),
    ).fetchall()
    assert [(row["metric"], row["hk_uuid"]) for row in rows] == [
        ("sleep_asleep", HK_UUID), ("sleep_core", HK_UUID),
    ]


def test_same_metric_and_hk_uuid_is_rejected(conn):
    db.insert_records(conn, [_record("sleep_core", "first-key")])

    with pytest.raises(sqlite3.IntegrityError):
        db.insert_records(conn, [_record("sleep_core", "second-key")])

    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric = 'sleep_core'"
    ).fetchone()[0] == 1


def test_workout_hk_uuid_is_unique_when_present(conn):
    db.insert_workouts(conn, [_workout(dedupe_key="workout-1", hk_uuid="w-1")])

    with pytest.raises(sqlite3.IntegrityError):
        db.insert_workouts(conn, [_workout(dedupe_key="workout-2", hk_uuid="w-1")])


def test_sync_state_is_scoped_by_device_and_type_and_deletions_round_trip(conn):
    conn.executemany(
        "INSERT INTO hk_sync_state "
        "(device_id, type_identifier, anchor_token, last_batch_sequence, "
        "last_batch_id, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("watch-a", "HKQuantityTypeIdentifierStepCount", "a1", 3, "ba", "2026-08-20T00:00:00Z"),
            ("watch-b", "HKQuantityTypeIdentifierStepCount", "b1", 7, "bb", "2026-08-20T00:01:00Z"),
        ],
    )
    conn.execute(
        "INSERT INTO hk_deletions "
        "(device_id, type_identifier, hk_uuid, deleted_at) VALUES (?, ?, ?, ?)",
        ("watch-a", "HKQuantityTypeIdentifierStepCount", "deleted-1", "2026-08-20T00:02:00Z"),
    )

    states = conn.execute(
        "SELECT device_id, anchor_token FROM hk_sync_state "
        "WHERE type_identifier = ? ORDER BY device_id",
        ("HKQuantityTypeIdentifierStepCount",),
    ).fetchall()
    assert [(row["device_id"], row["anchor_token"]) for row in states] == [
        ("watch-a", "a1"), ("watch-b", "b1"),
    ]
    assert conn.execute("SELECT COUNT(*) FROM hk_deletions").fetchone()[0] == 1


def test_old_vault_keeps_rows_while_migrating_hk_columns_and_indexes(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    conn.executescript("""
        CREATE TABLE records (
            id INTEGER PRIMARY KEY, metric TEXT NOT NULL, value REAL,
            unit TEXT, start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
            start_local TEXT, local_date TEXT NOT NULL, source TEXT,
            origin TEXT NOT NULL DEFAULT 'backfill',
            dedupe_key TEXT NOT NULL UNIQUE
        );
        CREATE TABLE workouts (
            id INTEGER PRIMARY KEY, workout_type TEXT NOT NULL,
            start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
            local_date TEXT NOT NULL, duration_min REAL, energy_kcal REAL,
            distance_mi REAL, unit_distance TEXT, source TEXT, route_ref TEXT,
            dedupe_key TEXT NOT NULL UNIQUE
        );
    """)
    conn.execute(
        "INSERT INTO records (metric, value, start_utc, end_utc, local_date, dedupe_key) "
        "VALUES ('step_count', 12, '2026-08-20T00:00:00Z', '2026-08-20T00:01:00Z', '2026-08-20', 'old')"
    )
    conn.commit()
    conn.close()

    conn = db.connect(path)
    db.init_db(conn)
    assert conn.execute("SELECT value FROM records WHERE dedupe_key = 'old'").fetchone()[0] == 12
    assert {row["name"] for row in conn.execute("PRAGMA table_info(records)")} >= {
        "hk_uuid", "hk_type_identifier", "source_revision_json", "hk_device_id",
    }
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' "
        "AND name = 'idx_records_metric_hk_uuid'"
    ).fetchone()[0] == 1
    conn.close()


def test_build_vault_carries_hk_columns_indexes_and_state(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "vault.db"
    conn = db.connect(source)
    db.init_db(conn)
    db.insert_records(conn, [_record("sleep_asleep", "hk-record-key")])
    db.insert_workouts(conn, [_workout(dedupe_key="hk-workout-key", hk_uuid="w-1")])
    conn.execute(
        "INSERT INTO hk_sync_state "
        "(device_id, type_identifier, anchor_token, updated_at) VALUES (?, ?, ?, ?)",
        ("watch-a", "HKCategoryTypeIdentifierSleepAnalysis", "anchor", "2026-08-20T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    vault.build_vault(source, target, measure_gzip=False)
    conn = db.connect(target, read_only=True)
    try:
        row = conn.execute(
            "SELECT hk_uuid, hk_type_identifier, source_revision_json, hk_device_id "
            "FROM records"
        ).fetchone()
        assert tuple(row) == (
            HK_UUID, "HKCategoryTypeIdentifierSleepAnalysis", '{"version":"1"}', "watch-a",
        )
        assert conn.execute("SELECT hk_uuid FROM workouts").fetchone()[0] == "w-1"
        # The anchor must NOT travel — see the _COPY_ORDER comment. build_vault
        # drops raw records outside the D3 allowlist, so an inherited anchor
        # would tell the phone not to resend samples this build discarded.
        assert conn.execute("SELECT COUNT(*) FROM hk_sync_state").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM pragma_index_list('records') "
            "WHERE name = 'idx_records_metric_hk_uuid' AND partial = 1"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM pragma_index_list('workouts') "
            "WHERE name = 'idx_workouts_hk_uuid' AND partial = 1"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_build_vault_does_not_inherit_an_anchor_for_a_dropped_series(tmp_path):
    """The failure the _COPY_ORDER exclusion exists to prevent, end to end.

    basal_energy is outside the D3 raw allowlist, so build_vault drops its raw
    samples. If its anchor came across, the vault would claim to hold samples
    it had just discarded and the phone would never resend them.
    """
    source = tmp_path / "source.db"
    target = tmp_path / "vault.db"
    conn = db.connect(source)
    db.init_db(conn)
    db.insert_records(conn, [_record("basal_energy", "dropped-series-key")])
    conn.execute(
        "INSERT INTO hk_sync_state "
        "(device_id, type_identifier, anchor_token, updated_at) VALUES (?, ?, ?, ?)",
        ("watch-a", "HKQuantityTypeIdentifierBasalEnergyBurned", "anchor-99",
         "2026-08-20T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO hk_deletions "
        "(device_id, type_identifier, hk_uuid, deleted_at) VALUES (?, ?, ?, ?)",
        ("watch-a", "HKQuantityTypeIdentifierBasalEnergyBurned", "gone-1",
         "2026-08-20T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    vault.build_vault(source, target, measure_gzip=False)
    conn = db.connect(target, read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM records WHERE metric = 'basal_energy'"
        ).fetchone()[0] == 0, "precondition: the series should have been dropped"
        assert conn.execute("SELECT COUNT(*) FROM hk_sync_state").fetchone()[0] == 0
        # The tombstone is the opposite case and must survive.
        assert conn.execute("SELECT hk_uuid FROM hk_deletions").fetchone()[0] == "gone-1"
    finally:
        conn.close()
