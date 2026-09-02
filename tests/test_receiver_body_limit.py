"""A request body must be bounded.

Audit P3-5: the receiver shows `Memory: 1G (peak 1.7G)` for 2-8 requests a day,
because the raw body, the decoded JSON and the built record dicts are all live
at once and nothing caps any of them.

The other half of P3-5 was a `MemoryMax=` on the systemd unit, asserted here
against `systemd/health-receiver.service`. That unit is not part of this repo,
so the assertion has no subject and was removed rather than faked; the process
limit below is the half that lives in Python.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, receiver


@pytest.fixture
def client(vault, tmp_path, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    monkeypatch.setattr(receiver, "MAX_BODY_BYTES", 500)
    with TestClient(receiver.create_app(vault)) as c:
        yield c


def _log(vault_path):
    conn = db.connect(vault_path)
    try:
        return conn.execute("SELECT kind, detail FROM ingest_log ORDER BY id DESC").fetchall()
    finally:
        conn.close()


def test_oversized_body_is_413(client):
    big = b'{"data": {"metrics": []}, "pad": "' + b"x" * 2000 + b'"}'
    r = client.post("/v1/ingest", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413, r.text
    assert "too large" in r.json()["detail"].lower()


def test_oversized_body_is_rejected_by_declared_content_length(client):
    """Cheapest rejection: refuse on the header, before reading a single byte."""
    r = client.post("/v1/ingest", content=b"x" * 4000,
                    headers={"content-type": "application/json", "content-length": "4000"})
    assert r.status_code == 413


def test_oversized_chunked_body_is_still_413(client):
    """A chunked upload declares no Content-Length; the cap must hold anyway."""
    def gen():
        for _ in range(20):
            yield b"x" * 100

    r = client.post("/v1/ingest", content=gen(), headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_oversized_body_is_logged(client, vault_path):
    client.post("/v1/ingest", content=b"x" * 4000, headers={"content-type": "application/json"})
    rows = _log(vault_path)
    assert rows and rows[0]["kind"] == "reject"
    assert "too large" in rows[0]["detail"].lower()


def test_a_body_under_the_cap_still_ingests(client):
    r = client.post("/v1/ingest", json={"protocol_version": 1,
                                        "device": {"id": "d", "name": "n", "model": "m"},
                                        "app_version": "1", "batch_id": "b",
                                        "batch_sequence": 1, "sent_at": "2026-08-22T00:00:00Z",
                                        "anchors": [], "samples": [], "deletions": [],
                                        "workouts": []})
    assert r.status_code == 200, r.text


def test_default_cap_admits_the_largest_batch_ever_observed():
    """1,052,330 records on 2026-06-24, ~180 MB of JSON at ~180 B/point."""
    assert receiver.MAX_BODY_BYTES >= 200 * 1024 * 1024
    assert receiver.MAX_BODY_BYTES <= 512 * 1024 * 1024
