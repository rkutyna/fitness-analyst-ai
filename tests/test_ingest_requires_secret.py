"""#105: /v1/ingest refuses with 401 when no shared secret is configured.

The ask routes already refused an empty secret; ingest used to skip the check,
so a launcher that never set one served unauthenticated writes. Both now share
one rule: no secret configured means every secret-checked route answers 401.
"""
from __future__ import annotations

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from health_advisor import receiver
from tests.conftest import INGEST_HEADERS, INGEST_TEST_SECRET


def _payload(batch_id="requires-secret"):
    return {
        "protocol_version": 1,
        "device": {"id": "watch", "name": "Watch", "model": "test"},
        "app_version": "1", "batch_id": batch_id, "batch_sequence": 1,
        "sent_at": "2026-08-22T12:00:00Z", "anchors": [],
        "samples": [], "deletions": [], "workouts": [],
    }


def _ingest_rows(vault) -> int:
    conn = vault.connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0]
    finally:
        conn.close()


@pytest.mark.parametrize("headers", [
    pytest.param({}, id="no-header"),
    pytest.param({"x-health-secret": ""}, id="empty-header"),
    pytest.param(INGEST_HEADERS, id="a-header-nobody-configured"),
])
def test_ingest_refuses_401_when_no_secret_is_configured(
        vault, monkeypatch, headers):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with TestClient(receiver.create_app(vault)) as client:
        before = _ingest_rows(vault)
        response = client.post("/v1/ingest", json=_payload(), headers=headers)
        assert response.status_code == 401
        assert _ingest_rows(vault) == before


def test_empty_secret_refusal_has_the_ask_routes_shape(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with pytest.raises(Exception) as ask:
        receiver._require_ask_secret(None)
    with TestClient(receiver.create_app(vault)) as client:
        response = client.post("/v1/ingest", json=_payload())
    assert response.status_code == ask.value.status_code == 401
    assert response.json()["detail"] == ask.value.detail


def test_the_ingest_guard_does_not_run_for_an_unconfigured_secret(
        vault, monkeypatch):
    calls = []

    def guard():
        calls.append("guard")
        return JSONResponse({"detail": "guard"}, status_code=503)

    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with TestClient(receiver.create_app(vault, ingest_guard=guard)) as client:
        response = client.post("/v1/ingest", json=_payload())
    assert response.status_code == 401
    assert calls == []


def test_ingest_with_a_configured_secret_still_requires_the_header(
        vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", INGEST_TEST_SECRET)
    with TestClient(receiver.create_app(vault)) as client:
        assert client.post("/v1/ingest", json=_payload()).status_code == 401
        wrong = client.post("/v1/ingest", json=_payload(),
                            headers={"x-health-secret": "not-the-secret"})
        assert wrong.status_code == 401
        ok = client.post("/v1/ingest", json=_payload(), headers=INGEST_HEADERS)
        assert ok.status_code == 200


def test_health_stays_open_without_a_secret(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with TestClient(receiver.create_app(vault)) as client:
        assert client.get("/health").status_code == 200
