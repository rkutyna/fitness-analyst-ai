"""A batch's record tombstones are looked up in one statement, not one per record (#110)."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, receiver
from tests.conftest import INGEST_HEADERS, INGEST_TEST_SECRET

HEART = "HKQuantityTypeIdentifierHeartRate"
SAMPLE_LOOKUP = "FROM hk_deletions WHERE (device_id, type_identifier, hk_uuid) IN"


def _device(name):
    return {"id": name, "name": name, "model": "synthetic-model"}


def _sample(uuid, minute):
    stamp = f"2026-01-02T10:{minute % 60:02d}:00Z"
    return {
        "kind": "quantity", "hk_uuid": uuid, "type_identifier": HEART,
        "start": stamp, "end": stamp, "value": 120.0, "unit": "count/min",
        "source_revision": {"source_name": "Synthetic Watch",
                            "bundle_id": "synthetic.bundle"},
    }


def _payload(*, device="dev-a", samples=(), deletions=(),
             batch_id="b", sequence=1):
    return {
        "protocol_version": 1, "device": _device(device), "app_version": "1",
        "batch_id": batch_id, "batch_sequence": sequence,
        "sent_at": "2026-01-02T12:00:00Z", "anchors": [],
        "samples": list(samples), "deletions": list(deletions),
        "workouts": [],
    }


@pytest.fixture
def statements(monkeypatch):
    seen: list[str] = []
    original = db.connect

    def traced(path, *, read_only=False):
        conn = original(path, read_only=read_only)
        if not read_only:
            conn.set_trace_callback(seen.append)
        return conn

    monkeypatch.setattr(db, "connect", traced)
    return seen


def _client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", INGEST_TEST_SECRET)
    return TestClient(receiver.create_app(vault), headers=INGEST_HEADERS)


def _post(client, payload):
    response = client.post("/v1/ingest", json=payload)
    assert response.status_code == 200, response.text
    return response


def _stored(path, table, column):
    conn = db.connect(path, read_only=True)
    try:
        return {r[0] for r in conn.execute(f"SELECT {column} FROM {table}")}
    finally:
        conn.close()


def _lookups(statements, marker):
    return sum(marker in s for s in statements)


@pytest.mark.parametrize("n", [3, 60])
def test_record_tombstone_lookup_count_does_not_grow_with_the_batch(
        vault, vault_path, monkeypatch, statements, n):
    with _client(vault, monkeypatch) as client:
        _post(client, _payload(deletions=[
            {"hk_uuid": "dead-1", "type_identifier": HEART}], batch_id="del"))
        statements.clear()
        samples = [_sample(f"live-{i}", i) for i in range(n)]
        samples.append(_sample("dead-1", 59))
        _post(client, _payload(samples=samples, batch_id="add", sequence=2))
    assert _lookups(statements, SAMPLE_LOOKUP) == 1
    assert not any("SELECT 1 FROM hk_deletions" in s for s in statements)
    stored = _stored(vault_path, "records", "hk_uuid")
    assert "dead-1" not in stored
    assert {f"live-{i}" for i in range(n)} <= stored


def test_a_tombstone_from_another_device_does_not_filter_a_record(
        vault, vault_path, monkeypatch):
    with _client(vault, monkeypatch) as client:
        _post(client, _payload(device="dev-a", batch_id="del", deletions=[
            {"hk_uuid": "shared-1", "type_identifier": HEART}]))
        _post(client, _payload(device="dev-b", batch_id="add", sequence=2,
                               samples=[_sample("shared-1", 5)]))
    assert "shared-1" in _stored(vault_path, "records", "hk_uuid")


def test_lookup_helpers_agree_with_the_per_key_query_across_chunks(vault):
    conn = vault.connect()
    db.init_db(conn)
    rows = [(f"dev-{i % 3}", f"type-{i % 5}", f"uuid-{i}") for i in range(0, 900, 2)]
    conn.executemany(
        "INSERT INTO hk_deletions (device_id, type_identifier, hk_uuid, "
        "deleted_at) VALUES (?, ?, ?, 'now')", rows)
    asked = [(f"dev-{i % 3}", f"type-{i % 5}", f"uuid-{i}") for i in range(900)]
    asked += [("dev-0", "type-9", "uuid-0"), ("dev-9", "type-0", "uuid-0"),
              (None, "type-0", "uuid-0")]

    def one_by_one(key):
        return conn.execute(
            "SELECT 1 FROM hk_deletions WHERE device_id = ? "
            "AND type_identifier = ? AND hk_uuid = ?", key).fetchone() is not None

    expected = {key for key in asked if one_by_one(key)}
    assert len(expected) == 450
    assert receiver._tombstoned_samples(conn, asked) == expected

    conn.close()


def test_helpers_survive_the_smallest_variable_limit(vault):
    conn = vault.connect()
    db.init_db(conn)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    keys = [("d", "t", f"u{i}") for i in range(2000)]
    assert receiver._tombstoned_samples(conn, keys) == set()
    conn.close()
