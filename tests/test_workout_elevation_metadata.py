"""Issue #63 acceptance tests using only synthetic workout data."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, elevation, hk_parse, receiver


DEVICE = {"id": "synthetic-device", "name": "synthetic-phone", "model": "synthetic-model"}
SOURCE = "Synthetic Source"
REVISION = {"source_name": SOURCE, "bundle_id": "synthetic.bundle"}
START = "2026-02-03T10:00:00Z"
END = "2026-02-03T10:30:00Z"


def _workout(*, uuid="synthetic-workout", start=START, end=END,
             elevation_values=None):
    row = {
        "hk_uuid": uuid,
        "workout_activity_type": "HKWorkoutActivityTypeRunning",
        "start": start,
        "end": end,
        "duration_min": 30.0,
        "source_revision": REVISION,
    }
    if elevation_values is not None:
        row.update(elevation_values)
    return row


def _elevation_entry(*, uuid=None, start=START, end=END, source=SOURCE,
                     ascended=11.0, descended=7.0):
    return {
        "hk_uuid": uuid,
        "start": start,
        "end": end,
        "source_name": source,
        "elevation_ascended_m": ascended,
        "elevation_descended_m": descended,
    }


def _payload(*, batch_id="synthetic-batch", workouts=None, workout_elevation=None):
    payload = {
        "protocol_version": 1,
        "device": DEVICE,
        "app_version": "synthetic-version",
        "batch_id": batch_id,
        "batch_sequence": 1,
        "sent_at": "2026-02-03T12:00:00Z",
        "anchors": [],
        "samples": [],
        "deletions": [],
        "workouts": workouts or [],
    }
    if workout_elevation is not None:
        payload["workout_elevation"] = workout_elevation
    return payload


def _client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    return TestClient(receiver.create_app(vault))


def _workout_row(vault):
    conn = vault.read_only()
    try:
        return conn.execute(
            "SELECT * FROM workouts ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()


def test_issue_63_done_01_phone_metadata_is_idempotent_and_old_shape_survives(
    vault, monkeypatch
):
    payload = _payload(
        batch_id="phone-elevation-1",
        workouts=[_workout(elevation_values={"elevation_ascended_m": 17.4,
                                              "elevation_descended_m": 8.2})],
    )
    with _client(vault, monkeypatch) as client:
        first = client.post("/v1/ingest", json=payload)
        replay = client.post(
            "/v1/ingest", json={**payload, "batch_id": "phone-elevation-2"}
        )
    assert first.status_code == replay.status_code == 200
    assert first.json()["workouts_added"] == 1
    assert replay.json()["workouts_added"] == 0
    row = _workout_row(vault)
    assert row["elevation_ascended_m"] == pytest.approx(17.4)
    assert row["elevation_descended_m"] == pytest.approx(8.2)
    assert row["elevation_source"] == "device_metadata"

    old_shape = _payload(
        batch_id="phone-old-shape",
        workouts=[_workout(
            uuid="old-shape-workout", start="2026-02-03T11:00:00Z",
            end="2026-02-03T11:30:00Z",
        )],
    )
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=old_shape)
    assert response.status_code == 200
    conn = vault.read_only()
    try:
        old = conn.execute(
            "SELECT elevation_ascended_m, elevation_descended_m "
            "FROM workouts WHERE hk_uuid = ?", ("old-shape-workout",)
        ).fetchone()
    finally:
        conn.close()
    assert old[0] is None and old[1] is None


def test_issue_63_done_02_backfill_fills_nulls_once_and_counts_unmatched(
    vault, monkeypatch
):
    conn = vault.connect()
    db.init_db(conn)
    db.insert_workouts(conn, [{
        "workout_type": "running", "start_utc": START,
        "end_utc": END, "local_date": "2026-02-03", "duration_min": 30.0,
        "energy_kcal": None, "distance_mi": None, "unit_distance": None,
        "source": "Synthetic Source ", "dedupe_key": "legacy-workout",
        "hk_uuid": None,
    }])
    conn.commit()
    conn.close()

    entries = [
        _elevation_entry(ascended=21.0, descended=13.0),
        _elevation_entry(ascended=99.0, descended=88.0),
        _elevation_entry(
            start="2026-02-03T12:00:00Z", end="2026-02-03T12:30:00Z",
            ascended=4.0, descended=3.0,
        ),
    ]
    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest", json=_payload(
                batch_id="elevation-backfill", workout_elevation=entries
            )
        )
    assert response.status_code == 200
    body = response.json()
    assert body["workout_elevation_seen"] == 3
    assert body["workout_elevation_matched"] == 2
    assert body["workout_elevation_updated"] == 1
    assert body["workout_elevation_unmatched"] == 1
    row = _workout_row(vault)
    assert row["elevation_ascended_m"] == pytest.approx(21.0)
    assert row["elevation_descended_m"] == pytest.approx(13.0)
    assert row["elevation_source"] == "device_metadata"


@pytest.mark.parametrize("entry_uuid", ["AAAA-1111", None])
def test_issue_63_done_02_uuid_and_legacy_entries_match_legacy_workout(
    conn, entry_uuid
):
    db.insert_workouts(conn, [{
        "workout_type": "running", "start_utc": "2030-01-05T15:00:00+00:00",
        "end_utc": "2030-01-05T15:30:00+00:00", "local_date": "2030-01-05",
        "duration_min": 30.0, "energy_kcal": None, "distance_mi": None,
        "unit_distance": None, "source": "Watch", "dedupe_key": "oracle",
        "hk_uuid": None,
    }])

    result = db.attach_workout_elevation(conn, [{
        "hk_uuid": entry_uuid,
        "start_utc": "2030-01-05T15:00:00+00:00",
        "end_utc": "2030-01-05T15:30:00+00:00",
        "source_name": "Watch",
        "elevation_ascended_m": 10.0,
        "elevation_descended_m": None,
    }])

    assert result == {
        "seen": 1, "matched": 1, "updated": 1, "unmatched": 0,
    }


def test_issue_63_done_02_different_workout_uuid_stays_unmatched(conn):
    db.insert_workouts(conn, [{
        "workout_type": "running", "start_utc": "2030-01-05T15:00:00+00:00",
        "end_utc": "2030-01-05T15:30:00+00:00", "local_date": "2030-01-05",
        "duration_min": 30.0, "energy_kcal": None, "distance_mi": None,
        "unit_distance": None, "source": "Watch", "dedupe_key": "other",
        "hk_uuid": "BBBB-2222",
    }])

    result = db.attach_workout_elevation(conn, [{
        "hk_uuid": "AAAA-1111",
        "start_utc": "2030-01-05T15:00:00+00:00",
        "end_utc": "2030-01-05T15:30:00+00:00",
        "source_name": "Watch",
        "elevation_ascended_m": 10.0,
        "elevation_descended_m": None,
    }])

    assert result == {
        "seen": 1, "matched": 0, "updated": 0, "unmatched": 1,
    }


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_issue_63_done_03_invalid_backfill_elevation_is_payload_error(value):
    payload = _payload(
        workout_elevation=[_elevation_entry(ascended=value)]
    )
    with pytest.raises(hk_parse.PayloadError):
        hk_parse.parse_payload(payload)


def test_issue_63_done_03_unknown_backfill_field_is_payload_error():
    entry = _elevation_entry()
    entry["unexpected"] = "reject-me"
    with pytest.raises(hk_parse.PayloadError):
        hk_parse.parse_payload(_payload(workout_elevation=[entry]))


def test_issue_63_done_04_health_advertises_workout_elevation(vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["workout_elevation_supported"] is True
    assert body["workout_elevation_backfill_generation"] == (
        elevation.WORKOUT_ELEVATION_BACKFILL_GENERATION)
    assert isinstance(body["workout_elevation_backfill_generation"], int)
    assert body["workout_elevation_backfill_generation"] >= 2


def test_issue_63_done_05_workout_climb_prefers_device_then_route_then_none(vault):
    conn = vault.connect()
    db.init_db(conn)
    db.insert_workouts(conn, [{
        "workout_type": "running", "start_utc": START, "end_utc": END,
        "local_date": "2026-02-03", "duration_min": 30.0,
        "energy_kcal": None, "distance_mi": None, "unit_distance": None,
        "source": SOURCE, "dedupe_key": "device-workout", "hk_uuid": "device",
        "elevation_ascended_m": 17.4, "elevation_descended_m": 8.2,
    }, {
        "workout_type": "running", "start_utc": START, "end_utc": END,
        "local_date": "2026-02-03", "duration_min": 30.0,
        "energy_kcal": None, "distance_mi": None, "unit_distance": None,
        "source": SOURCE, "dedupe_key": "route-workout", "hk_uuid": "route",
    }, {
        "workout_type": "running", "start_utc": START, "end_utc": END,
        "local_date": "2026-02-03", "duration_min": 30.0,
        "energy_kcal": None, "distance_mi": None, "unit_distance": None,
        "source": SOURCE, "dedupe_key": "empty-workout", "hk_uuid": "empty",
    }])
    points = {
        "t_offset_s": [float(i) for i in range(20)],
        "altitude_m": [100.0] * 10 + [110.0] * 10,
        "vertical_accuracy_m": [1.0] * 20,
        "horizontal_accuracy_m": [2.0] * 20,
    }
    db.insert_workout_routes(conn, [{
        "hk_route_uuid": "synthetic-route", "start_utc": START,
        "end_utc": "2026-02-03T10:00:20Z", "source_name": SOURCE,
        "workout_hk_uuid": "route", "start_lat_1dp": 0.0,
        "start_lon_1dp": 0.0, "n_points": 20, "points": points,
    }])
    ids = {
        row["dedupe_key"]: row["id"]
        for row in conn.execute("SELECT id, dedupe_key FROM workouts")
    }
    assert elevation.workout_climb(conn, ids["device-workout"]) == {
        "ascended_m": 17.4, "descended_m": 8.2, "source": "device_metadata"
    }
    # A direct insert that names no source still stores a labelled total, as
    # the conflict path does; workout_climb alone cannot see the column.
    stored = {
        row["dedupe_key"]: row["elevation_source"]
        for row in conn.execute("SELECT dedupe_key, elevation_source FROM workouts")
    }
    assert stored == {"device-workout": "device_metadata",
                      "route-workout": None, "empty-workout": None}
    route_result = elevation.workout_climb(conn, ids["route-workout"])
    assert route_result["source"] == "route_estimate"
    assert route_result["ascended_m"] > 0
    assert elevation.workout_climb(conn, ids["empty-workout"]) == {
        "status": "no_elevation"
    }
    conn.close()
