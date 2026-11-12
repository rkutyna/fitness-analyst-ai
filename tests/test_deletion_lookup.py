"""HealthKit deletions find their rows by SEARCH, and find the same rows.

downstream #734: each deletion looked its UUID up with a predicate that planned
as a SCAN of idx_records_metric_hk_uuid, ~30 ms per deletion on a 14M-row
vault, all under the ingest's write lock; 200 deletions of UUIDs the vault
never held held it for 6 s. The cost oracle is SQLite VM steps at two vault
sizes, never wall-clock.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, receiver

DEVICE = {"id": "dev-a", "name": "iPhone", "model": "iPhone17,1"}
SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"
HEART = "HKQuantityTypeIdentifierHeartRate"


def _payload(batch_id, deletions):
    return {
        "protocol_version": 1, "device": DEVICE, "app_version": "1.0",
        "batch_id": batch_id, "batch_sequence": 1,
        "sent_at": "2026-10-22T13:04:05Z", "anchors": [],
        "samples": [], "deletions": deletions, "workouts": [],
    }


def _row(conn, metric, uuid, type_identifier, device, n):
    start = f"2026-10-0{1 + n % 9}T08:{n % 60:02d}:{n // 60 % 60:02d}+00:00"
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, dedupe_key, hk_uuid, hk_type_identifier, "
        "hk_device_id) VALUES (?, 1, 'u', ?, ?, ?, 's', 'receiver', ?, ?, ?, ?)",
        (metric, start, start, start[:10], f"k|{metric}|{uuid}|{device}|{n}",
         uuid, type_identifier, device))


def _post(vault, monkeypatch, batch_id, deletions):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "s3cret")
    with TestClient(receiver.create_app(vault)) as client:
        response = client.post("/v1/ingest", json=_payload(batch_id, deletions),
                               headers={"x-health-secret": "s3cret"})
    assert response.status_code == 200, response.text
    return response.json()


def test_deletion_removes_exactly_the_rows_a_uuid_lookup_finds(
        conn, vault, monkeypatch):
    # One sleep sample expanded into two canonical metrics shares its UUID.
    _row(conn, "sleep_core", "u-sleep", SLEEP, "dev-a", 1)
    _row(conn, "sleep_in_bed", "u-sleep", SLEEP, "dev-a", 2)
    # Same UUID, another device / another type: not this deletion's rows.
    _row(conn, "sleep_deep", "u-sleep", SLEEP, "dev-b", 3)
    _row(conn, "heart_rate", "u-sleep", HEART, "dev-a", 4)
    _row(conn, "heart_rate", "u-other", HEART, "dev-a", 5)
    conn.commit()

    body = _post(vault, monkeypatch, "del-1", [
        {"hk_uuid": "u-sleep", "type_identifier": SLEEP},
        {"hk_uuid": "u-never-held", "type_identifier": HEART},
    ])
    assert body["deleted"] == 2
    left = {(r["metric"], r["hk_uuid"], r["hk_device_id"]) for r in conn.execute(
        "SELECT metric, hk_uuid, hk_device_id FROM records")}
    assert left == {("sleep_deep", "u-sleep", "dev-b"),
                    ("heart_rate", "u-sleep", "dev-a"),
                    ("heart_rate", "u-other", "dev-a")}
    tombstones = {r["hk_uuid"]: (r["sample_metric"], r["sample_local_date"])
                  for r in conn.execute("SELECT * FROM hk_deletions")}
    assert tombstones == {"u-sleep": ("sleep_core", "2026-10-02"),
                          "u-never-held": (None, None)}


def test_an_empty_vault_still_tombstones_every_deletion(conn, vault, monkeypatch):
    # No stored metric at all: the IN-list is empty and must match nothing.
    body = _post(vault, monkeypatch, "del-empty", [
        {"hk_uuid": "u-1", "type_identifier": HEART}])
    assert body["deleted"] == 0
    assert conn.execute("SELECT COUNT(*) FROM hk_deletions").fetchone()[0] == 1


def _deletion_steps(tmp_path, monkeypatch, size):
    from health_advisor.context import VaultContext
    path = tmp_path / f"v{size}.db"
    seed = db.connect(path)
    db.init_db(seed)
    for n in range(size):
        _row(seed, "heart_rate", f"stored-{n:08d}", HEART, "dev-a", n)
    seed.commit()
    seed.close()
    vault = VaultContext.local(path, user_id="test", writable=True)
    steps = [0]
    real_connect = db.connect

    def counting_connect(*args, **kwargs):
        c = real_connect(*args, **kwargs)

        def tick():
            steps[0] += 1
            return 0
        c.set_progress_handler(tick, 100)
        return c

    monkeypatch.setattr(db, "connect", counting_connect)
    _post(vault, monkeypatch, f"del-{size}",
          [{"hk_uuid": f"unknown-{i}", "type_identifier": HEART}
           for i in range(40)])
    monkeypatch.setattr(db, "connect", real_connect)
    return steps[0]


def test_deletion_cost_does_not_grow_with_the_vault(tmp_path, monkeypatch):
    small = _deletion_steps(tmp_path, monkeypatch, 3_000)
    large = _deletion_steps(tmp_path, monkeypatch, 12_000)
    # A SEARCH is flat in the vault's size; the SCAN it replaced was ~4x here.
    assert large / small < 1.5, (small, large)


def test_a_tombstone_still_refuses_the_sample_arriving_in_a_later_batch(
        conn, vault, monkeypatch):
    body = _post(vault, monkeypatch, "del-first",
                 [{"hk_uuid": "late-uuid", "type_identifier": HEART}])
    assert body["deleted"] == 0 and body["tombstones_added"] == 1
    sample = {"kind": "quantity", "hk_uuid": "late-uuid", "type_identifier": HEART,
              "start": "2026-10-21T08:00:00-04:00", "end": "2026-10-21T08:00:00-04:00",
              "value": 80.0, "unit": "count/min",
              "source_revision": {"source_name": "Apple Watch", "bundle_id": "x"}}
    later = {**_payload("add-later", []), "samples": [sample]}
    monkeypatch.setattr(receiver, "SHARED_SECRET", "s3cret")
    with TestClient(receiver.create_app(vault)) as client:
        r = client.post("/v1/ingest", json=later, headers={"x-health-secret": "s3cret"})
    assert r.status_code == 200 and r.json()["records_added"] == 0
    assert conn.execute("SELECT COUNT(*) FROM records WHERE hk_uuid = 'late-uuid'"
                        ).fetchone()[0] == 0


def _steps_for_deletions(tmp_path, monkeypatch, n):
    from health_advisor.context import VaultContext
    path = tmp_path / f"n{n}.db"
    seed = db.connect(path)
    db.init_db(seed)
    for i in range(2_000):
        _row(seed, "heart_rate", f"stored-{i:08d}", HEART, "dev-a", i)
    seed.commit()
    seed.close()
    steps = [0]
    real_connect = db.connect

    def counting_connect(*args, **kwargs):
        c = real_connect(*args, **kwargs)

        def tick():
            steps[0] += 1
            return 0
        c.set_progress_handler(tick, 100)
        return c

    monkeypatch.setattr(db, "connect", counting_connect)
    _post(VaultContext.local(path, user_id="test", writable=True), monkeypatch,
          f"n-{n}", [{"hk_uuid": f"unknown-{i}", "type_identifier": HEART}
                     for i in range(n)])
    monkeypatch.setattr(db, "connect", real_connect)
    return steps[0]


def test_deletion_cost_is_linear_in_the_number_of_deletions(tmp_path, monkeypatch):
    small = _steps_for_deletions(tmp_path, monkeypatch, 50)
    large = _steps_for_deletions(tmp_path, monkeypatch, 200)
    assert large / small < 6, (small, large)
