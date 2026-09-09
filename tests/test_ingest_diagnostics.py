"""Structured HealthKit rejection evidence and its read-only MCP tool."""
from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from health_advisor import db, hk_parse, mcp_server, receiver


DAY = "2026-08-01"
UNKNOWN_APPLE_UNIT = "mL/(kg·fortnight)"


def _payload():
    return {
        "protocol_version": 1,
        "device": {"id": "synthetic-device", "name": "Synthetic Phone",
                    "model": "Synthetic Model"},
        "app_version": "test",
        "batch_id": "synthetic-batch",
        "batch_sequence": 1,
        "sent_at": f"{DAY}T12:00:00Z",
        "anchors": [],
        "samples": [{
            "kind": "quantity",
            "hk_uuid": "synthetic-vo2-point",
            "type_identifier": "HKQuantityTypeIdentifierVO2Max",
            "start": f"{DAY}T09:00:00Z",
            "end": f"{DAY}T09:00:01Z",
            "value": 42.0,
            "unit": UNKNOWN_APPLE_UNIT,
            "source_revision": {
                "source_name": "Synthetic Watch",
                "bundle_id": "synthetic.health",
            },
        }],
        "deletions": [],
        "workouts": [],
    }


def test_unknown_unit_is_structured_and_explained_by_the_tool(
    vault, vault_path, monkeypatch
):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with TestClient(receiver.create_app(vault)) as client:
        response = client.post("/v1/ingest", json=_payload())
    assert response.status_code == 200, response.text

    tool = mcp_server.build_tools(vault)["get_ingest_diagnostics"]
    out = tool("vo2_max", DAY, DAY)

    assert out["arrived"] == 1
    assert out["stored"] == 0
    assert out["rejected"] == {"unit_mismatch": 1}
    assert out["rejection_details"][0]["reason"] == "unit_mismatch"
    assert out["rejection_details"][0]["metric"] == "vo2_max"
    assert out["rejection_details"][0]["unit"] == UNKNOWN_APPLE_UNIT

    conn = db.connect(vault_path, read_only=True)
    try:
        row = conn.execute(
            "SELECT metric, reason, local_date, source, device_id, batch_id, unit "
            "FROM ingest_diagnostics"
        ).fetchone()
    finally:
        conn.close()
    assert tuple(row) == (
        "vo2_max", "unit_mismatch", DAY, "Synthetic Watch",
        "synthetic-device", "synthetic-batch", UNKNOWN_APPLE_UNIT,
    )


def test_empty_diagnostics_are_explicit(conn, tools):
    out = tools.get_ingest_diagnostics("metric-never-seen", DAY, DAY)

    assert out == {
        "metric": "metric-never-seen",
        "start": DAY,
        "end": DAY,
        "arrived": 0,
        "stored": 0,
        "rejected": {},
        "rejection_details": [],
        "last_stored_timestamp": {},
        "stored_by_source": [],
    }

    db.insert_records(conn, [{
        "metric": "step_count", "value": 1.0, "unit": "count",
        "start_utc": f"{DAY}T09:00:00Z", "end_utc": f"{DAY}T09:00:01Z",
        "start_local": f"{DAY} 09:00:00", "local_date": DAY,
        "source": "Synthetic Watch", "origin": "healthkit",
        "dedupe_key": "diagnostics-shape-record",
    }])
    conn.commit()
    nonempty = tools.get_ingest_diagnostics("step_count", DAY, DAY)
    assert nonempty["arrived"] == 1
    assert nonempty["stored"] == 1
    assert nonempty["stored_by_source"] == [{
        "source": "Synthetic Watch", "count": 1,
        "last_stored_timestamp": f"{DAY}T09:00:00Z",
    }]


def test_only_rejections_write_diagnostics(vault, vault_path, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    payload = _payload()
    accepted = []
    for i in range(100):
        sample = copy.deepcopy(payload["samples"][0])
        sample.update({
            "hk_uuid": f"accepted-{i}",
            "type_identifier": "HKQuantityTypeIdentifierStepCount",
            "value": i + 1.0,
            "unit": "count",
            "start": f"{DAY}T09:{i // 60:02d}:{i % 60:02d}Z",
            "end": f"{DAY}T09:{i // 60:02d}:{i % 60:02d}Z",
        })
        accepted.append(sample)
    rejected = []
    for i in range(2):
        sample = copy.deepcopy(payload["samples"][0])
        sample.update({
            "hk_uuid": f"rejected-{i}",
            "value": 42.0,
            "unit": UNKNOWN_APPLE_UNIT,
            "start": f"{DAY}T12:0{i}:00Z",
            "end": f"{DAY}T12:0{i}:01Z",
        })
        rejected.append(sample)
    payload.update({"batch_id": "accepted-and-rejected", "samples": accepted + rejected})

    with TestClient(receiver.create_app(vault)) as client:
        response = client.post("/v1/ingest", json=payload)
    assert response.status_code == 200, response.text

    conn = db.connect(vault_path, read_only=True)
    try:
        diagnostic_count = conn.execute(
            "SELECT COUNT(*) FROM ingest_diagnostics"
        ).fetchone()[0]
    finally:
        conn.close()
    assert diagnostic_count == 2


def test_rejected_sample_date_is_derived_from_timestamp():
    payload = _payload()
    sample = payload["samples"][0]
    sample.update({
        "local_date": "2099-01-01",
        "start": f"{DAY}T09:00:00Z",
        "end": f"{DAY}T09:00:01Z",
    })
    parsed = hk_parse.parse_payload(payload)
    assert parsed["rejections"][0]["local_date"] == DAY


def test_duplicate_daily_totals_do_not_create_point_diagnostics(
    vault, vault_path, monkeypatch
):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    payload = _payload()
    payload.update({
        "batch_id": "duplicate-daily-totals",
        "samples": [],
        "deletions": [],
        "daily_totals": [{
            "type_identifier": "HKQuantityTypeIdentifierStepCount",
            "local_date": DAY, "value": 100, "unit": "count",
            "interval": "day", "state": "provisional",
            "queried_at": f"{DAY}T12:00:00Z",
        }] * 2,
    })
    parsed = hk_parse.parse_payload(payload)
    assert [row["_point_index"] for row in parsed["daily_totals"]] == [0, 1]

    with TestClient(receiver.create_app(vault)) as client:
        response = client.post("/v1/ingest", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["daily_totals_added"] == 2

    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM hk_daily_total_revisions"
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM ingest_diagnostics"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_untyped_rejection_retry_is_idempotent(conn):
    row = {
        "batch_id": "malformed-retry", "point_kind": "sample",
        "point_index": 0, "metric": None, "type_identifier": None,
        "local_date": DAY, "source": None, "device_id": "synthetic-device",
        "hk_uuid": None, "unit": None,
        "reason": "malformed", "detail": "missing required field(s)",
    }
    db.log_ingest_diagnostics(conn, [row, row])
    assert conn.execute(
        "SELECT COUNT(*) FROM ingest_diagnostics"
    ).fetchone()[0] == 1


def test_parser_types_the_other_point_rejection_paths():
    cases = [
        (lambda sample: sample.update(
            {"type_identifier": "HKQuantityTypeIdentifierFutureMetric"}),
         "unknown_type"),
        (lambda sample: sample.pop("value"), "malformed"),
        (lambda sample: sample.update({"kind": "future-kind"}), "malformed"),
    ]
    for mutate, expected in cases:
        payload = _payload()
        mutate(payload["samples"][0])
        parsed = hk_parse.parse_payload(payload)
        assert parsed["rejections"][0]["reason"] == expected

    payload = _payload()
    payload["daily_totals"] = [{
        "type_identifier": "HKQuantityTypeIdentifierFutureTotal",
        "local_date": DAY,
        "value": 1,
        "unit": "count",
        "interval": "day",
        "state": "provisional",
        "queried_at": f"{DAY}T12:00:00Z",
    }]
    parsed = hk_parse.parse_payload(payload)
    assert parsed["rejections"][-1]["reason"] == "unknown_type"
    assert parsed["rejections"][-1]["point_kind"] == "daily_total"
