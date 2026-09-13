"""Issue #57 acceptance tests: route ingest and storage."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, hk_parse, receiver, vault


DEVICE = {"id": "synthetic-device", "name": "synthetic-phone", "model": "synthetic-model"}
SOURCE = "synthetic-route-source"
REVISION = {"source_name": SOURCE, "bundle_id": "synthetic.bundle"}


def _route(n: int = 3, *, uuid: str = "route-1", workout_hk_uuid: str | None = None,
           start: str = "2026-01-02T10:00:00Z", extra: dict | None = None) -> dict:
    route = {
        "hk_route_uuid": uuid,
        "start": start,
        "end": f"{start[:10]}T10:30:00Z",
        "source_name": SOURCE,
        "start_lat_1dp": 0.1,
        "start_lon_1dp": 0.2,
        "points": {
            "t_offset_s": [float(i) for i in range(n)],
            "altitude_m": [100.0 + i * 0.25 for i in range(n)],
            "vertical_accuracy_m": [1.0] * n,
            "horizontal_accuracy_m": [2.0] * n,
        },
    }
    if workout_hk_uuid is not None:
        route["workout_hk_uuid"] = workout_hk_uuid
    if extra:
        route.update(extra)
    return route


def _payload(*, routes=None, workouts=None, deletions=None, samples=None,
             batch_id="route-batch", sequence=1):
    return {
        "protocol_version": 1,
        "device": DEVICE,
        "app_version": "synthetic-version",
        "batch_id": batch_id,
        "batch_sequence": sequence,
        "sent_at": "2026-01-02T12:00:00Z",
        "anchors": [],
        "samples": samples or [],
        "deletions": deletions or [],
        "workouts": workouts or [],
        "workout_routes": routes if routes is not None else [],
    }


def _client(vault_context, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    return TestClient(receiver.create_app(vault_context))


def _workout(*, uuid="workout-1", start="2026-01-02T10:00:00Z",
             end="2026-01-02T10:30:00Z"):
    return {
        "hk_uuid": uuid,
        "workout_activity_type": "HKWorkoutActivityTypeRunning",
        "start": start,
        "end": end,
        "duration_min": 30.0,
        "source_revision": REVISION,
    }


def _stored_route(path):
    conn = db.connect(path, read_only=True)
    try:
        return conn.execute("SELECT * FROM workout_routes").fetchone()
    finally:
        conn.close()


def _set_history(vault_context, through):
    conn = vault_context.connect()
    db.init_db(conn)
    vault.set_history_imported_through(conn, through)
    conn.commit()
    conn.close()


def test_done_01_5400_route_round_trips(vault, vault_path, monkeypatch):
    route = _route(5400)
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(routes=[route]))
    assert response.status_code == 200, response.text
    assert response.json()["routes_added"] == 1
    row = _stored_route(vault_path)
    assert row["n_points"] == 5400
    decoded = db.decode_route_points(row["points"], row["encoding"], row["n_points"])
    for field in route["points"]:
        assert decoded[field] == pytest.approx(route["points"][field], rel=1e-6, abs=1e-6)
    assert row["ascent_m"] is None and row["descent_m"] is None
    assert row["method_version"] is None


def test_done_02_route_resends_are_idempotent(vault, vault_path, monkeypatch):
    route = _route()
    with _client(vault, monkeypatch) as client:
        first = client.post("/v1/ingest", json=_payload(routes=[route], batch_id="one"))
        replay = client.post("/v1/ingest", json=_payload(routes=[route], batch_id="one"))
        different_batch = client.post(
            "/v1/ingest", json=_payload(routes=[route], batch_id="two", sequence=2)
        )
    assert first.json()["routes_added"] == 1
    assert replay.json()["applied"] is False
    assert different_batch.json()["routes_added"] == 0
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM workout_routes").fetchone()[0] == 1
    finally:
        conn.close()


def test_done_03_point_limit_and_fixed_bind_count(vault, vault_path, monkeypatch):
    original_connect = db.connect

    def limited_connect(path, *, read_only=False):
        conn = original_connect(path, read_only=read_only)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        return conn

    monkeypatch.setattr(db, "connect", limited_connect)
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(routes=[_route(20_000)]))
    assert response.status_code == 200, response.text
    assert _stored_route(vault_path)["n_points"] == 20_000
    with pytest.raises(hk_parse.PayloadError):
        hk_parse.parse_payload(_payload(routes=[_route(20_001)]))


def test_done_04_no_trace_and_coordinate_precision_are_rejected(vault, monkeypatch):
    for route in (
        _route(extra={"points": {**_route()["points"], "lat": [0.1], "lon": [0.2]}}),
        _route(extra={"start_lat_1dp": 0.12}),
    ):
        with pytest.raises(hk_parse.PayloadError):
            hk_parse.parse_payload(_payload(routes=[route]))
        with _client(vault, monkeypatch) as client:
            response = client.post("/v1/ingest", json=_payload(routes=[route]))
        assert response.status_code == 400


def test_done_05_route_deletion_and_workout_cascade(vault, vault_path, monkeypatch):
    route = _route(workout_hk_uuid="workout-1")
    with _client(vault, monkeypatch) as client:
        added = client.post(
            "/v1/ingest",
            json=_payload(routes=[route], workouts=[_workout()], batch_id="add"),
        )
        assert added.status_code == 200, added.text
        deleted = client.post(
            "/v1/ingest",
            json=_payload(
                deletions=[{
                    "hk_uuid": "route-1",
                    "type_identifier": "HKWorkoutRouteTypeIdentifier",
                }],
                batch_id="delete-route",
                sequence=2,
            ),
        )
    assert deleted.status_code == 200, deleted.text
    conn = db.connect(vault_path, read_only=False)
    try:
        assert conn.execute("SELECT COUNT(*) FROM workout_routes").fetchone()[0] == 0
        conn.execute("DELETE FROM workouts WHERE hk_uuid = ?", ("workout-1",))
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM workout_routes").fetchone()[0] == 0
    finally:
        conn.close()


def test_done_06_time_match_unmatched_report_and_late_attach(vault, vault_path, monkeypatch):
    conn = vault.connect()
    db.init_db(conn)
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("running", "2026-01-02T10:00:00+00:00", "2026-01-02T10:30:00+00:00",
         "2026-01-02", 30.0, SOURCE, "legacy-workout-key"),
    )
    conn.commit()
    conn.close()
    with _client(vault, monkeypatch) as client:
        matched = client.post(
            "/v1/ingest", json=_payload(routes=[_route(uuid="matched")], batch_id="matched")
        )
        unmatched = client.post(
            "/v1/ingest",
            json=_payload(routes=[_route(uuid="later", start="2026-01-03T10:00:00Z")],
                          batch_id="unmatched", sequence=2),
        )
        later_workout = client.post(
            "/v1/ingest",
            json=_payload(
                workouts=[_workout(uuid="later-workout", start="2026-01-03T10:00:00Z",
                                   end="2026-01-03T10:30:00Z")],
                batch_id="later-workout", sequence=3,
            ),
        )
    assert matched.json()["routes_unmatched"] == 0
    assert unmatched.json()["routes_unmatched"] == 1
    assert later_workout.status_code == 200
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute(
            "SELECT workout_id FROM workout_routes WHERE hk_route_uuid='matched'"
        ).fetchone()[0] is not None
        assert conn.execute(
            "SELECT workout_id FROM workout_routes WHERE hk_route_uuid='later'"
        ).fetchone()[0] is not None
    finally:
        conn.close()


def test_done_07_health_capability_and_legacy_batch(vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        health = client.get("/health")
        legacy = client.post("/v1/ingest", json={
            "protocol_version": 1, "device": DEVICE, "app_version": "v",
            "batch_id": "legacy", "batch_sequence": 1, "sent_at": "2026-01-01T00:00:00Z",
            "anchors": [], "samples": [], "deletions": [], "workouts": [],
        })
    assert health.status_code == 200
    assert health.json()["workout_routes_supported"] is True
    assert legacy.status_code == 200
    assert "routes_seen" not in legacy.json()


def test_done_08_build_vault_copies_routes(tmp_path):
    source_path = tmp_path / "source.db"
    target_path = tmp_path / "target.db"
    source = db.connect(source_path)
    db.init_db(source)
    db.insert_workout_routes(source, [{
        "hk_route_uuid": "copy-route", "start_utc": "2026-01-02T10:00:00+00:00",
        "end_utc": "2026-01-02T10:30:00+00:00", "source_name": SOURCE,
        "workout_hk_uuid": None, "start_lat_1dp": 0.1, "start_lon_1dp": 0.2,
        "n_points": 2, "points": {
            "t_offset_s": [0.0, 1.0], "altitude_m": [1.0, 2.0],
            "vertical_accuracy_m": [1.0, 1.0], "horizontal_accuracy_m": [2.0, 2.0],
        },
    }])
    source.commit()
    source.close()
    report = vault.build_vault(source_path, target_path)
    assert report["table_counts"]["workout_routes"] == 1
    target = db.connect(target_path, read_only=True)
    try:
        assert target.execute("SELECT COUNT(*) FROM workout_routes").fetchone()[0] == 1
    finally:
        target.close()


def test_done_09_packer_mutation_is_observable(monkeypatch):
    original = db.pack_route_points

    def drops_last_point(points):
        return original(points)[:-16]

    monkeypatch.setattr(db, "pack_route_points", drops_last_point)
    points = {
        "t_offset_s": [0.0, 1.0], "altitude_m": [1.0, 2.0],
        "vertical_accuracy_m": [1.0, 1.0], "horizontal_accuracy_m": [2.0, 2.0],
    }
    with pytest.raises(ValueError, match="BLOB length"):
        db.unpack_route_points(db.pack_route_points(points), 2)


def test_review_01_negative_accuracy_values_round_trip(vault, vault_path, monkeypatch):
    route = _route(100)
    route["points"]["vertical_accuracy_m"][::10] = [-1.0] * 10
    for index in (0, 33, 66):
        route["points"]["horizontal_accuracy_m"][index] = -1.0
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(routes=[route]))
    assert response.status_code == 200, response.text
    assert response.json()["routes_added"] == 1
    row = _stored_route(vault_path)
    decoded = db.decode_route_points(row["points"], row["encoding"], row["n_points"])
    assert decoded["vertical_accuracy_m"][0] == pytest.approx(-1.0)
    assert decoded["horizontal_accuracy_m"][0] == pytest.approx(-1.0)


def test_review_02_routes_bypass_history_watermark_records_do_not(
    vault, vault_path, monkeypatch
):
    _set_history(vault, "2026-08-20")
    with _client(vault, monkeypatch) as client:
        route_response = client.post(
            "/v1/ingest",
            json=_payload(
                routes=[_route(start="2026-06-22T10:00:00Z")],
                batch_id="route-before-watermark",
            ),
        )
        record_response = client.post(
            "/v1/ingest",
            json=_payload(
                samples=[{
                    "kind": "quantity", "hk_uuid": "record-before-watermark",
                    "type_identifier": "HKQuantityTypeIdentifierHeartRate",
                    "start": "2026-08-01T10:00:00Z",
                    "end": "2026-08-01T10:00:01Z", "value": 120.0,
                    "unit": "count/min",
                    "source_revision": REVISION,
                }],
                batch_id="record-before-watermark", sequence=2,
            ),
        )
    assert route_response.status_code == 200, route_response.text
    assert route_response.json()["routes_added"] == 1
    assert record_response.status_code == 409, record_response.text
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM workout_routes").fetchone()[0] == 1
    finally:
        conn.close()


def test_review_03_method_version_is_integer(vault_path):
    conn = db.connect(vault_path)
    db.init_db(conn)
    conn.commit()
    conn.close()
    conn = db.connect(vault_path, read_only=True)
    try:
        column = conn.execute(
            "SELECT type FROM pragma_table_info('workout_routes') "
            "WHERE name = 'method_version'"
        ).fetchone()
    finally:
        conn.close()
    assert column[0] == "INTEGER"


def test_review_04_empty_routes_are_counted_but_not_stored(vault, vault_path, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest",
            json=_payload(
                routes=[_route(0, uuid="empty-route"), _route(3, uuid="full-route")],
                batch_id="empty-route-batch",
            ),
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["routes_seen"] == 2
    assert body["routes_added"] == 1
    assert body["routes_empty"] == 1
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM workout_routes").fetchone()[0] == 1
        detail = conn.execute(
            "SELECT detail FROM commit_log WHERE key = ?",
            ("healthkit:synthetic-device:empty-route-batch",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert "routes_empty=1" in detail
