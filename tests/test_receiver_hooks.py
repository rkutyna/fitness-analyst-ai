"""Tests for the receiver's deployment extension hooks."""
from __future__ import annotations

import sys
from types import SimpleNamespace

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from health_advisor import receiver


def test_ingest_guard_runs_before_the_body_is_parsed(vault, monkeypatch):
    calls = []

    def guard():
        calls.append("guard")
        return JSONResponse({"detail": "temporarily unavailable"}, status_code=503)

    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with TestClient(receiver.create_app(vault, ingest_guard=guard)) as client:
        response = client.post(
            "/v1/ingest",
            content=b"this is deliberately not JSON",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "temporarily unavailable"
    assert calls == ["guard"]


def test_health_extra_is_merged_into_health_payload(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with TestClient(receiver.create_app(
            vault, health_extra=lambda: {"deployment": "ready"})) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["deployment"] == "ready"
    assert response.json()["ok"] is True


def test_main_honours_an_explicit_app_factory(monkeypatch, tmp_path):
    captured = {}
    context = object()

    monkeypatch.setattr(
        receiver.VaultContext, "local",
        lambda *args, **kwargs: context,
    )

    def app_factory(ctx, **kwargs):
        captured["ctx"] = ctx
        captured["kwargs"] = kwargs
        return object()

    uvicorn = SimpleNamespace(
        run=lambda app, **kwargs: captured.update(app=app, run_kwargs=kwargs)
    )
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    assert receiver.main(
        ["--vault", str(tmp_path / "vault.db")],
        app_factory=app_factory,
    ) == 0
    assert captured["ctx"] is context
    assert captured["app"] is not None
