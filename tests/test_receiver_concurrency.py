"""The HealthKit receiver must not block the event loop.

The body is read asynchronously, while parsing and SQLite work run in
Starlette's threadpool so `/health` remains responsive during a POST.
"""
from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from health_advisor import db, hk_parse, receiver


@pytest.fixture
def app(vault):
    return receiver.create_app(vault)


@pytest.fixture
def client(app, tmp_path, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with TestClient(app) as c:
        yield c


def _payload(n: int = 1):
    samples = []
    for i in range(n):
        hh, mm = divmod(i, 60)
        samples.append({"kind": "quantity", "hk_uuid": f"step-{i}",
                        "type_identifier": "HKQuantityTypeIdentifierStepCount",
                        "start": f"2026-07-30T{hh:02d}:{mm:02d}:00-04:00",
                        "end": f"2026-07-30T{hh:02d}:{mm:02d}:30-04:00",
                        "value": 10 + i, "unit": "count",
                        "source_revision": {"source_name": "Watch",
                                              "bundle_id": "test"}})
    return {"protocol_version": 1,
            "device": {"id": "watch", "name": "Watch", "model": "test"},
            "app_version": "1", "batch_id": f"batch-{n}", "batch_sequence": 1,
            "sent_at": "2026-07-30T12:00:00Z", "anchors": [],
            "samples": samples, "deletions": [], "workouts": []}


# --------------------------------------------------------------------------- #
# Off the event loop
# --------------------------------------------------------------------------- #
def _endpoint(app):
    for r in app.routes:
        if getattr(r, "path", None) == "/v1/ingest":
            return r.endpoint
    raise AssertionError("no /v1/ingest route")


def test_ingest_endpoint_is_sync_so_starlette_threadpools_it(app):
    assert not asyncio.iscoroutinefunction(_endpoint(app)), (
        "an `async def` /v1/ingest runs its blocking DB work ON the event loop"
    )


def test_ingest_work_runs_off_the_event_loop_thread(app, vault_path, conn, tmp_path, monkeypatch):
    """Hard proof: the parse/DB work happens on a thread that is NOT the thread
    running the event loop, and /health answers while /v1/ingest is mid-flight."""
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    # httpx.ASGITransport does not run lifespan; the conn fixture initializes the schema.

    started = threading.Event()
    release = threading.Event()
    handler_thread: list[threading.Thread] = []
    real_parse = hk_parse.parse_payload

    def slow_parse(payload):
        handler_thread.append(threading.current_thread())
        started.set()
        release.wait(2)          # bounded, so a regression fails rather than hangs
        return real_parse(payload)

    monkeypatch.setattr(hk_parse, "parse_payload", slow_parse)

    async def _run():
        loop_thread = threading.current_thread()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            post = asyncio.create_task(ac.post("/v1/ingest", json=_payload(3)))
            await asyncio.to_thread(started.wait, 5)
            assert started.is_set(), "/v1/ingest never reached the parse"
            assert handler_thread[0] is not loop_thread, (
                "the handler ran on the event loop thread — it is still async def "
                "or has been wrapped back onto the loop"
            )
            # ...and the loop is free enough to serve another request meanwhile.
            health = await asyncio.wait_for(ac.get("/health"), timeout=5)
            assert health.status_code == 200
            release.set()
            resp = await post
            assert resp.status_code == 200, resp.text

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# Chunked record insertion inside one HealthKit transaction
# --------------------------------------------------------------------------- #
def test_records_are_written_in_bounded_chunks(client, vault_path, monkeypatch):
    monkeypatch.setattr(receiver, "INGEST_CHUNK", 10)
    sizes: list[int] = []
    real_insert = db.insert_records

    def spy(conn, rows):
        rows = list(rows)
        sizes.append(len(rows))
        return real_insert(conn, rows)

    monkeypatch.setattr(db, "insert_records", spy)

    response = client.post("/v1/ingest", json=_payload(25))
    assert response.status_code == 200, response.text
    assert sizes == [10, 10, 5], f"one unbounded insert, not chunks: {sizes}"
    assert response.json()["records_added"] == 25

    conn = db.connect(vault_path)
    assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 25
    conn.close()


def test_chunking_does_not_lose_the_daily_rollup(client, vault_path, monkeypatch):
    monkeypatch.setattr(receiver, "INGEST_CHUNK", 4)
    assert client.post("/v1/ingest", json=_payload(9)).status_code == 200
    conn = db.connect(vault_path)
    row = conn.execute(
        "SELECT count, sum FROM daily_metrics WHERE metric='step_count'"
    ).fetchone()
    conn.close()
    assert row["count"] == 9
    assert row["sum"] == sum(10 + i for i in range(9))


def test_ingest_chunk_default_is_bounded():
    assert 0 < receiver.INGEST_CHUNK <= 20_000


# --------------------------------------------------------------------------- #
# Reader busy_timeout
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("read_only", [True, False])
def test_busy_timeout_is_30s(tmp_path, read_only):
    p = tmp_path / "bt.db"
    db.connect(p).close()
    conn = db.connect(p, read_only=read_only)
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000
    finally:
        conn.close()
