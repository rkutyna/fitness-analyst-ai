"""A batch's route tombstones are looked up in one statement, not one scan per route (#110)."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, receiver

ROUTE_TYPE = "HKWorkoutRouteTypeIdentifier"
SOURCE = "synthetic-route-source"
ROUTE_LOOKUP = "FROM hk_deletions WHERE hk_uuid IN"


def _device(name):
    return {"id": name, "name": name, "model": "synthetic-model"}


def _route(uuid):
    return {
        "hk_route_uuid": uuid, "start": "2026-01-02T10:00:00Z",
        "end": "2026-01-02T10:30:00Z", "source_name": SOURCE,
        "start_lat_1dp": 0.1, "start_lon_1dp": 0.2,
        "points": {"t_offset_s": [0.0, 1.0], "altitude_m": [100.0, 100.25],
                   "vertical_accuracy_m": [1.0, 1.0],
                   "horizontal_accuracy_m": [2.0, 2.0]},
    }


def _payload(*, device="dev-a", deletions=(), routes=(),
             batch_id="b", sequence=1):
    return {
        "protocol_version": 1, "device": _device(device), "app_version": "1",
        "batch_id": batch_id, "batch_sequence": sequence,
        "sent_at": "2026-01-02T12:00:00Z", "anchors": [],
        "samples": [], "deletions": list(deletions),
        "workouts": [], "workout_routes": list(routes),
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
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    return TestClient(receiver.create_app(vault))


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


@pytest.mark.parametrize("n", [2, 40])
def test_route_tombstone_lookup_count_does_not_grow_with_the_batch(
        vault, vault_path, monkeypatch, statements, n):
    with _client(vault, monkeypatch) as client:
        # Tombstoned under a different device than the one sending the routes.
        _post(client, _payload(device="dev-a", batch_id="del", deletions=[
            {"hk_uuid": "route-dead", "type_identifier": ROUTE_TYPE}]))
        statements.clear()
        routes = [_route(f"route-{i}") for i in range(n)] + [_route("route-dead")]
        _post(client, _payload(device="dev-b", routes=routes, batch_id="add",
                               sequence=2))
    assert _lookups(statements, ROUTE_LOOKUP) == 1
    assert not any("SELECT 1 FROM hk_deletions" in s for s in statements)
    stored = _stored(vault_path, "workout_routes", "hk_route_uuid")
    assert "route-dead" not in stored
    assert {f"route-{i}" for i in range(n)} <= stored


def test_route_lookup_agrees_with_the_per_route_query_across_chunks(vault):
    conn = vault.connect()
    db.init_db(conn)
    rows = [(f"dev-{i % 3}", f"type-{i % 5}", f"uuid-{i}") for i in range(0, 900, 2)]
    conn.executemany(
        "INSERT INTO hk_deletions (device_id, type_identifier, hk_uuid, "
        "deleted_at) VALUES (?, ?, ?, 'now')", rows)
    uuids = [f"uuid-{i}" for i in range(900)] + ["nowhere", None]
    expected = {u for u in uuids if u is not None and conn.execute(
        "SELECT 1 FROM hk_deletions WHERE hk_uuid = ?", (u,)).fetchone()}
    assert len(expected) == 450
    assert receiver._tombstoned_route_uuids(conn, uuids) == expected
    conn.close()


def test_route_lookup_survives_the_smallest_variable_limit(vault):
    conn = vault.connect()
    db.init_db(conn)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    assert receiver._tombstoned_route_uuids(
        conn, [f"u{i}" for i in range(2000)]) == set()
    conn.close()
