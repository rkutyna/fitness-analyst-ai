"""HealthKit receiver robustness.

Malformed HealthKit envelopes are rejected before any database mutation. This
module keeps the boundary/error coverage that used to be exercised through the
retired JSON writer, but sends only HealthKit-direct shapes.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, hk_parse, receiver


def _base(**overrides):
    payload = {
        "protocol_version": 1,
        "device": {"id": "watch", "name": "Watch", "model": "test"},
        "app_version": "1", "batch_id": "robustness", "batch_sequence": 1,
        "sent_at": "2026-08-22T12:00:00Z", "anchors": [],
        "samples": [], "deletions": [], "workouts": [],
    }
    payload.update(overrides)
    return payload


MALFORMED = [
    pytest.param([1, 2, 3], id="top-level-list"),
    pytest.param("just a string", id="top-level-string"),
    pytest.param({"protocol_version": 1}, id="missing-envelope-fields"),
    pytest.param(_base(extra=True), id="unknown-envelope-field"),
    pytest.param(_base(device="watch"), id="device-is-string"),
    pytest.param(_base(anchors={}), id="anchors-is-object"),
    pytest.param(_base(deletions={}), id="deletions-is-object"),
    pytest.param(_base(workouts={}), id="workouts-is-object"),
    pytest.param(_base(samples=[{"kind": "quantity", "extra": True}]),
                 id="sample-has-unknown-field-early"),
    pytest.param(_base(samples=[{"kind": "quantity", "extra": True}]),
                 id="sample-has-unknown-field"),
]


@pytest.fixture
def client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with TestClient(receiver.create_app(vault)) as c:
        yield c


@pytest.mark.parametrize("payload", MALFORMED)
def test_parse_payload_raises_payload_error(payload):
    with pytest.raises(hk_parse.PayloadError):
        hk_parse.parse_payload(payload)


@pytest.mark.parametrize("payload", MALFORMED)
def test_malformed_payload_is_400_not_500(client, payload):
    response = client.post("/v1/ingest", json=payload)
    assert response.status_code == 400, response.text
    assert "malformed payload" in response.json()["detail"].lower()


@pytest.mark.parametrize("payload", MALFORMED)
def test_malformed_payload_writes_nothing(client, vault_path, payload):
    client.post("/v1/ingest", json=payload)
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM commit_log").fetchone()[0] == 0
    finally:
        conn.close()


def test_invalid_json_is_400_and_writes_nothing(client, vault_path):
    response = client.post("/v1/ingest", content=b"{not json",
                              headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert "malformed payload" in response.json()["detail"].lower()
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_good_payload_still_ingests(client, vault_path):
    payload = _base(
        samples=[{
            "kind": "quantity", "hk_uuid": "step-1",
            "type_identifier": "HKQuantityTypeIdentifierStepCount",
            "start": "2026-08-22T08:00:00-04:00",
            "end": "2026-08-22T08:01:00-04:00", "value": 120,
            "unit": "count",
            "source_revision": {"source_name": "Watch", "bundle_id": "test"},
        }],
    )
    response = client.post("/v1/ingest", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["records_added"] == 1


def test_empty_payload_is_accepted_not_rejected(client):
    assert client.post("/v1/ingest", json=_base()).status_code == 200
