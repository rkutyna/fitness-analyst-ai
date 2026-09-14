"""Issue #57 acceptance tests: route ingest and storage."""
from __future__ import annotations

import sqlite3
from datetime import datetime

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


def _legacy_workout(uuid, start, end, source, workout_type="walking"):
    duration_min = (
        datetime.fromisoformat(end) - datetime.fromisoformat(start)
    ).total_seconds() / 60
    return {
        "workout_type": workout_type,
        "start_utc": start,
        "end_utc": end,
        "local_date": start[:10],
        "duration_min": duration_min,
        "energy_kcal": None,
        "distance_mi": None,
        "unit_distance": None,
        "source": source,
        "dedupe_key": f"legacy-{uuid}",
        "hk_uuid": None,
    }


def _db_route(uuid, start, end, source="Synthetic Watch"):
    return {
        "hk_route_uuid": uuid,
        "start_utc": start,
        "end_utc": end,
        "source_name": source,
        "workout_hk_uuid": None,
        "start_lat_1dp": 0.0,
        "start_lon_1dp": 0.0,
        "n_points": 2,
        "points": {
            "t_offset_s": [0.0, 1.0],
            "altitude_m": [100.0, 100.25],
            "vertical_accuracy_m": [1.0, 1.0],
            "horizontal_accuracy_m": [2.0, 2.0],
        },
    }


def test_issue_61_done_01_legacy_source_shapes_attach(vault):
    conn = vault.connect()
    db.init_db(conn)
    workouts = [
        _legacy_workout(
            "shape-a", "2026-01-02T10:00:00+00:00", "2026-01-02T10:30:00+00:00",
            "Synthetic Phone ", "walking",
        ),
        _legacy_workout(
            "shape-b", "2026-01-02T11:00:00+00:00", "2026-01-02T11:30:00+00:00",
            "Synthetic Watch|Synthetic Phone ", "walking",
        ),
        _legacy_workout(
            "shape-c", "2026-01-02T12:00:00+00:00", "2026-01-02T12:30:00+00:00",
            "Synthetic Phone ", "hiking",
        ),
    ]
    routes = [
        _db_route("shape-a-route", "2026-01-02T10:00:00+00:00", "2026-01-02T10:30:00+00:00"),
        _db_route("shape-b-route", "2026-01-02T11:00:00+00:00", "2026-01-02T11:30:00+00:00"),
        _db_route("shape-c-route", "2026-01-02T12:00:00+00:00", "2026-01-02T12:30:00+00:00"),
    ]
    try:
        db.insert_workouts(conn, workouts[:2])
        db.insert_workout_routes(conn, routes[:2])
        db.insert_workout_routes(conn, [routes[2]])
        assert conn.execute(
            "SELECT workout_id FROM workout_routes WHERE hk_route_uuid = ?",
            ("shape-c-route",),
        ).fetchone()[0] is None

        db.insert_workouts(conn, [workouts[2]])
        assert db.attach_unmatched_workout_routes(conn) == 1
        attached = conn.execute(
            "SELECT r.hk_route_uuid, w.workout_type "
            "FROM workout_routes r JOIN workouts w ON w.id = r.workout_id "
            "ORDER BY r.hk_route_uuid"
        ).fetchall()
        assert [(row[0], row[1]) for row in attached] == [
            ("shape-a-route", "walking"),
            ("shape-b-route", "walking"),
            ("shape-c-route", "hiking"),
        ]
    finally:
        conn.close()


def test_issue_61_done_02_early_route_start_overlaps_long_workout(vault):
    conn = vault.connect()
    db.init_db(conn)
    route = _db_route(
        "early-route", "2026-01-02T09:57:16+00:00", "2026-01-02T13:02:00+00:00",
    )
    workout = _legacy_workout(
        "long-workout", "2026-01-02T10:00:00+00:00", "2026-01-02T13:02:00+00:00",
        "Synthetic Phone ", "paddle_sports",
    )
    try:
        db.insert_workout_routes(conn, [route])
        db.insert_workouts(conn, [workout])
        assert db.attach_unmatched_workout_routes(conn) == 1
        assert conn.execute(
            "SELECT workout_id FROM workout_routes WHERE hk_route_uuid = ?",
            ("early-route",),
        ).fetchone()[0] is not None
    finally:
        conn.close()


def test_issue_61_done_03_route_without_90_percent_overlap_stays_unmatched(vault):
    conn = vault.connect()
    db.init_db(conn)
    route = _db_route(
        "unmatched-route", "2026-01-02T10:00:00+00:00", "2026-01-02T10:30:00+00:00",
    )
    workout = _legacy_workout(
        "distant-workout", "2026-01-02T12:18:00+00:00", "2026-01-02T12:48:00+00:00",
        "Synthetic Phone ",
    )
    try:
        db.insert_workouts(conn, [workout])
        db.insert_workout_routes(conn, [route])
        assert db.attach_unmatched_workout_routes(conn) == 0
        assert conn.execute(
            "SELECT workout_id FROM workout_routes WHERE hk_route_uuid = ?",
            ("unmatched-route",),
        ).fetchone()[0] is None
    finally:
        conn.close()


def test_issue_61_done_04_back_to_back_routes_are_order_independent(tmp_path):
    workouts = [
        _legacy_workout(
            "run", "2026-01-02T09:00:00+00:00", "2026-01-02T09:35:00+00:00",
            "Synthetic Phone ", "running",
        ),
        _legacy_workout(
            "walk", "2026-01-02T09:37:30+00:00", "2026-01-02T10:19:30+00:00",
            "Synthetic Phone ", "walking",
        ),
    ]
    routes = [
        _db_route("run-route", "2026-01-02T09:00:00+00:00", "2026-01-02T09:35:00+00:00"),
        _db_route("walk-route", "2026-01-02T09:37:30+00:00", "2026-01-02T10:19:30+00:00"),
    ]
    expected = {"run-route": "running", "walk-route": "walking"}
    for index, route_order in enumerate((routes, list(reversed(routes)))):
        conn = db.connect(tmp_path / f"back-to-back-{index}.db")
        db.init_db(conn)
        try:
            db.insert_workouts(conn, workouts)
            db.insert_workout_routes(conn, route_order)
            rows = conn.execute(
                "SELECT r.hk_route_uuid, w.workout_type "
                "FROM workout_routes r JOIN workouts w ON w.id = r.workout_id"
            ).fetchall()
            actual = {row[0]: row[1] for row in rows}
            assert actual == expected
        finally:
            conn.close()


def test_issue_61_source_contains_route_source_on_equal_overlap(vault):
    conn = vault.connect()
    db.init_db(conn)
    route = _db_route(
        "source-tie-route", "2026-01-02T10:00:00+00:00", "2026-01-02T10:30:00+00:00",
    )
    workouts = [
        _legacy_workout(
            "generic-source", "2026-01-02T09:59:30+00:00", "2026-01-02T10:30:30+00:00",
            "Synthetic Phone ",
        ),
        _legacy_workout(
            "matching-source", "2026-01-02T10:00:00+00:00", "2026-01-02T10:30:00+00:00",
            "  Synthetic   Watch | Synthetic Phone ",
        ),
    ]
    try:
        db.insert_workouts(conn, workouts)
        db.insert_workout_routes(conn, [route])
        matched = conn.execute(
            "SELECT w.dedupe_key FROM workout_routes r "
            "JOIN workouts w ON w.id = r.workout_id "
            "WHERE r.hk_route_uuid = ?", ("source-tie-route",),
        ).fetchone()
        assert matched[0] == "legacy-matching-source"
    finally:
        conn.close()


def test_issue_61_uuid_route_attaches_to_legacy_workout(conn):
    workout = _legacy_workout(
        "legacy-route-parent", "2030-01-05T10:00:00+00:00",
        "2030-01-05T10:30:00+00:00", "Synthetic Watch", "running",
    )
    route = _db_route(
        "uuid-route-to-legacy", "2030-01-05T10:00:00+00:00",
        "2030-01-05T10:30:00+00:00", "Synthetic Watch",
    )
    route["workout_hk_uuid"] = "AAAA-1111"

    db.insert_workouts(conn, [workout])
    db.insert_workout_routes(conn, [route])
    legacy_id = conn.execute(
        "SELECT id FROM workouts WHERE dedupe_key = ?",
        (workout["dedupe_key"],),
    ).fetchone()[0]

    assert conn.execute(
        "SELECT workout_id FROM workout_routes WHERE hk_route_uuid = ?",
        ("uuid-route-to-legacy",),
    ).fetchone()[0] == legacy_id


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


def test_route_anchor_is_advanceable_and_route_deletion_is_handled():
    """The phone's route pass sends an HKWorkoutRouteTypeIdentifier anchor. A
    refused anchor is held by the client, so the backfill would re-send the
    same first route forever; a route deletion noted as unknown would do the
    same through the client's reading of `unhandled`."""
    payload = _payload(routes=[_route()], deletions=[{
        "hk_uuid": "gone-route", "type_identifier": "HKWorkoutRouteTypeIdentifier"}])
    payload["anchors"] = [{"type_identifier": "HKWorkoutRouteTypeIdentifier",
                           "from": None, "to": "anchor-token"}]
    parsed = hk_parse.parse_payload(payload)
    assert parsed["rejected_anchors"] == []
    assert parsed["anchor_results"] == [{
        "index": 0, "type_identifier": "HKWorkoutRouteTypeIdentifier", "accepted": True}]
    assert not any("HKWorkoutRouteTypeIdentifier" in note for note in parsed["unhandled"])
