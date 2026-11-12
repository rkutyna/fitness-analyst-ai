"""A held vault lock answers 503 and stalls nothing else; ingest traces timings.

downstream #734: a phone sync holds the vault's write lock in bursts. A request
that waited out the busy timeout escaped as a bare 500, /v1/ask waited for the
lock ON the event loop (stalling every other request), and the ingest trace
carried no timings to show who held the lock.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from health_advisor import chat, db, receiver

HEART = "HKQuantityTypeIdentifierHeartRate"
HEADERS = {"x-health-secret": "s3cret"}


def _payload(batch_id, n=3):
    samples = [{
        "kind": "quantity", "hk_uuid": f"{batch_id}-{i}", "type_identifier": HEART,
        "start": f"2026-10-21T08:00:{i:02d}-04:00",
        "end": f"2026-10-21T08:00:{i:02d}-04:00", "value": 80.0 + i,
        "unit": "count/min",
        "source_revision": {"source_name": "Apple Watch", "bundle_id": "x"},
    } for i in range(n)]
    return {"protocol_version": 1,
            "device": {"id": "dev-a", "name": "iPhone", "model": "iPhone17,1"},
            "app_version": "1.0", "batch_id": batch_id, "batch_sequence": 1,
            "sent_at": "2026-10-21T13:04:05Z", "anchors": [],
            "samples": samples, "deletions": [], "workouts": []}


@pytest.fixture
def app(vault, conn, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "s3cret")
    monkeypatch.setattr(db, "BUSY_TIMEOUT_MS", 300)
    return receiver.create_app(vault)


def _counts(path):
    c = sqlite3.connect(path)
    try:
        return tuple(c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                     for t in ("records", "commit_log", "conversations",
                               "conversation_turns"))
    finally:
        c.close()


@pytest.fixture
def hold_lock(vault_path):
    holders = []

    def take(kind):
        c = sqlite3.connect(vault_path, isolation_level=None)
        c.execute(f"BEGIN {kind}")
        holders.append(c)
    yield take
    for c in holders:
        c.rollback()
        c.close()


# ------------------------------------------------------------------- trace (F)
def test_ingest_ok_trace_carries_lock_timings(app, capfd):
    with TestClient(app) as client:
        r = client.post("/v1/ingest", json=_payload("t-1"), headers=HEADERS)
    assert r.status_code == 200, r.text
    line = next(l for l in capfd.readouterr().err.splitlines()
                if l.startswith("ingest-trace ingest-ok"))
    for field in ("lock_wait_ms=", "lock_hold_ms=", "derive_hold_ms=",
                  "slowest_step=", "slowest_step_ms="):
        assert field in line, (field, line)
    assert "ingest-slow-lock" not in capfd.readouterr().err


def test_a_hold_over_the_threshold_emits_a_warning(app, capfd, monkeypatch):
    monkeypatch.setattr(receiver, "LOCK_HOLD_WARN_MS", -1)
    with TestClient(app) as client:
        assert client.post("/v1/ingest", json=_payload("t-2"),
                           headers=HEADERS).status_code == 200
    err = capfd.readouterr().err
    assert "ingest-trace ingest-slow-lock" in err and "threshold_ms=-1" in err


def test_timing_code_cannot_fail_an_ingest(app, capfd, monkeypatch):
    def boom(self):
        raise RuntimeError("timer broke")
    monkeypatch.setattr(receiver._StepClock, "slowest", boom)
    with TestClient(app) as client:
        assert client.post("/v1/ingest", json=_payload("t-3"),
                           headers=HEADERS).status_code == 200
    assert "ingest-trace ingest-ok" in capfd.readouterr().err


# ---------------------------------------------------------------- busy (G)
def _assert_busy(response):
    assert response.status_code == 503, response.text
    assert response.headers["retry-after"] == receiver.VAULT_BUSY_RETRY_AFTER_SECONDS
    assert response.json() == {"detail": receiver.VAULT_BUSY_DETAIL}


def test_a_held_write_lock_answers_503_on_ingest_and_ask_storing_nothing(
        app, vault_path, hold_lock):
    before = _counts(vault_path)
    hold_lock("IMMEDIATE")
    with TestClient(app) as client:
        _assert_busy(client.post("/v1/ingest", json=_payload("b-1"), headers=HEADERS))
        _assert_busy(client.post("/v1/ask", json={"question": "how did I sleep?"},
                                 headers=HEADERS))
    assert _counts(vault_path) == before   # nothing stored: safe to re-send


def test_ask_on_an_existing_conversation_is_503_and_stores_no_turn(
        app, vault, vault_path, hold_lock):
    conversation = chat.create_conversation(vault)["id"]
    before = _counts(vault_path)
    hold_lock("IMMEDIATE")
    with TestClient(app) as client:
        _assert_busy(client.post(
            "/v1/ask", json={"conversation_id": conversation, "question": "q?"},
            headers=HEADERS))
    assert _counts(vault_path) == before


def test_a_held_exclusive_lock_answers_503_on_conversation(app, hold_lock):
    with TestClient(app) as client:     # start-up first: it opens the vault
        hold_lock("EXCLUSIVE")
        _assert_busy(client.get("/v1/conversation", headers=HEADERS))


def test_other_operational_errors_stay_500(app, monkeypatch):
    def broken(*a, **k):
        raise sqlite3.OperationalError("no such table: nowhere")
    monkeypatch.setattr(chat, "list_recent_turns", broken)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/v1/conversation", headers=HEADERS).status_code == 500


def test_the_event_loop_keeps_serving_while_ask_waits_for_the_lock(
        app, monkeypatch, hold_lock):
    monkeypatch.setattr(db, "BUSY_TIMEOUT_MS", 2_000)
    hold_lock("IMMEDIATE")   # writers wait; readers (/health) are not blocked

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://t") as client:
            started = time.monotonic()
            ask = asyncio.create_task(client.post(
                "/v1/ask", json={"question": "q?"}, headers=HEADERS))
            await asyncio.sleep(0.2)          # ask is now waiting on the lock
            health = await client.get("/health")
            health_after = time.monotonic() - started
            ask_response = await ask
            return health.status_code, health_after, ask_response.status_code
    status, health_after, ask_status = asyncio.run(scenario())
    assert status == 200
    assert health_after < 1.5, health_after   # did not wait out ask's 2 s
    assert ask_status == 503
