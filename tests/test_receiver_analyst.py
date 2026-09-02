from __future__ import annotations

import asyncio
import json
import sys
import threading
from types import SimpleNamespace

import httpx
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from health_advisor import db, receiver
from health_advisor import analyst_sandbox as sb
from health_advisor.analyst_envelope import Envelope, Refusal
from health_advisor.context import VaultContext


def _envelope() -> Envelope:
    return Envelope(
        run_id="run-id",
        question="unused",
        code_sha256="code-sha",
        vault_sha256="vault-sha",
        vault_version=0,
        ledger={"query_count": 1, "tables_read": ["daily_metrics"],
                "rows_read": 2},
        tables=({
            "name": "weekly_steps",
            "columns": ("week", "steps"),
            "units": ("count", "count"),
            "rows": ((1, 12),),
            "row_count": 1,
        },),
        counts={"rows": 1, "cells": 2, "numeric_tokens": 2, "bytes": 100},
    )


def _response_payload(response):
    return response.json() if hasattr(response, "json") else json.loads(response.body)


def _fake_complete(reply: str = "```python\nemit('weekly_steps', ...)\n```"):
    calls: list[str] = []

    def complete(prompt: str, **kwargs) -> str:
        calls.append(prompt)
        return reply

    complete.calls = calls
    return complete


def _fake_executor():
    return object()


def _fake_run_code(code, vault_path, run_dir, executor, *, limits=None):
    return _envelope()


def test_analyst_success_adds_refused_and_removes_exact_run_record_key(
        vault, conn, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "secret")
    complete = _fake_complete()
    app = receiver.create_app(
        vault, analyst_complete_fn=complete, analyst_run_code_fn=_fake_run_code,
        analyst_executor_factory=_fake_executor)
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyst", json={"question": "What changed?"},
            headers={"x-health-secret": "secret"})

    assert response.status_code == 200
    body = _response_payload(response)
    assert body["refused"] is False
    assert "run_record_path" not in body["provenance"]
    assert body["code"] == "emit('weekly_steps', ...)"
    assert body["tables"][0]["rows"] == [[1, 12]]
    assert complete.calls


def test_analyst_auth_and_question_validation(vault, conn, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "secret")
    with TestClient(receiver.create_app(vault)) as client:
        assert client.post("/v1/analyst", json={"question": "q"}).status_code == 401
        assert client.post(
            "/v1/analyst", json={"question": "q"},
            headers={"x-health-secret": "wrong"}).status_code == 401
        assert client.post(
            "/v1/analyst", json={},
            headers={"x-health-secret": "secret"}).status_code == 422
        assert client.post(
            "/v1/analyst", json={"question": "  "},
            headers={"x-health-secret": "secret"}).status_code == 422


def test_analyst_refusal_is_a_200_payload_unchanged(vault, conn, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "secret")
    refusal = Refusal("table 'x' exceeds row cap: 500 > 200")

    def always_refuses(code, vault_path, run_dir, executor, *, limits=None):
        return refusal

    app = receiver.create_app(
        vault, analyst_complete_fn=_fake_complete(),
        analyst_run_code_fn=always_refuses,
        analyst_executor_factory=_fake_executor)
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyst", json={"question": "Too much?"},
            headers={"x-health-secret": "secret"})

    assert response.status_code == 200
    body = _response_payload(response)
    assert body["refused"] is True
    assert body["reason"] == refusal.reason
    assert body["remediation"]


def test_analyst_missing_sandbox_returns_503_before_model_call(
        vault, conn, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "secret")

    def missing_sandbox():
        raise RuntimeError(
            "/usr/bin/sandbox-exec not found — SeatbeltExecutor is macOS-only")

    def should_not_complete(prompt):
        raise AssertionError("the model must not be called without a sandbox")

    app = receiver.create_app(
        vault, analyst_complete_fn=should_not_complete,
        analyst_run_code_fn=_fake_run_code,
        analyst_executor_factory=missing_sandbox)
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyst", json={"question": "Will this run?"},
            headers={"x-health-secret": "secret"})

    assert response.status_code == 503
    body = _response_payload(response)
    # `detail` is the key the iOS client reads, and it is the same key FastAPI's
    # HTTPException gives the 429 beside it. Pinned here because this body is
    # the entire user-visible surface of analyst mode on a Linux host.
    assert "sandbox-exec" in body["detail"]
    assert "reason" not in body


def test_main_explicitly_selects_transient_executor(monkeypatch, tmp_path):
    selected = {}
    monkeypatch.delenv("HEALTH_ADVISOR_ANALYST_EXECUTOR", raising=False)
    monkeypatch.setattr(
        receiver.VaultContext, "local", lambda *args, **kwargs: object()
    )

    def fake_create_app(ctx, **kwargs):
        selected.update(kwargs)
        return object()

    uvicorn = SimpleNamespace(
        run=lambda app, **kwargs: selected.update(uvicorn_kwargs=kwargs)
    )
    monkeypatch.setattr(receiver, "create_app", fake_create_app)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    assert receiver.main([
        "--vault", str(tmp_path / "vault.db"),
        "--analyst-executor", "transient",
    ]) == 0
    assert selected["analyst_executor_factory"] is sb.TransientUnitExecutor


def test_analyst_uses_the_bound_context_vault(
        vault, vault_path, conn, tmp_path, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "secret")
    other_path = tmp_path / "other.db"
    other_conn = db.connect(other_path)
    db.init_db(other_conn)
    other_conn.close()
    other = VaultContext.local(other_path, user_id="other", writable=True)
    seen: list = []

    def record_path(code, path, run_dir, executor, *, limits=None):
        seen.append(path)
        return _envelope()

    for context in (vault, other):
        response = receiver._analyst(
            context, object(), b'{"question":"Which vault?"}', "secret",
            complete_fn=_fake_complete(), run_code_fn=record_path,
            executor_factory=_fake_executor)
        assert response.status_code == 200

    assert seen == [vault.db_path, other.db_path]
    assert seen[0] == vault_path
    assert seen[0] != seen[1]


def test_analyst_permit_rejects_second_request_immediately(vault, conn, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "secret")
    started = threading.Event()
    release = threading.Event()

    def slow_run(ctx, question, **kwargs):
        started.set()
        assert release.wait(5), "first analyst request did not get released"
        return JSONResponse({"refused": True, "reason": "test",
                             "remediation": "test"})

    monkeypatch.setattr(receiver, "_run_analyst", slow_run)
    app = receiver.create_app(vault)

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.post(
                "/v1/analyst", json={"question": "first"},
                headers={"x-health-secret": "secret"}))
            await asyncio.to_thread(started.wait, 5)
            assert started.is_set(), "first analyst request never entered the run"
            second = await client.post(
                "/v1/analyst", json={"question": "second"},
                headers={"x-health-secret": "secret"})
            release.set()
            first_response = await first
        return first_response, second

    first_response, second_response = asyncio.run(exercise())
    assert first_response.status_code == 200
    assert second_response.status_code == 429
