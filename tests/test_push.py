from __future__ import annotations

import base64
import json
import logging
import sqlite3
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from fastapi.testclient import TestClient

from health_advisor import db, push, receiver


def _decode_segment(segment: str) -> dict:
    padded = segment + "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))


def _sender(tmp_path: Path, *, handler=None, endpoint="https://api.push.apple.com"):
    private_key = ec.generate_private_key(ec.SECP256R1())
    key_path = tmp_path / "throwaway-apns-key.p8"
    key_path.write_bytes(private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    client = None
    if handler is not None:
        client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = push.APNsSender(
        key_path=key_path,
        key_id="TESTKEY1",
        team_id="TESTTEAM1",
        topic="com.example.HealthAdvisor",
        endpoint=endpoint,
        http_client=client,
    )
    return sender, private_key


def test_device_tokens_are_created_fresh_and_preserve_existing_rows(tmp_path):
    path = tmp_path / "existing-vault.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE preserved_rows (value TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO preserved_rows VALUES (?)", ("still-here",))
    conn.commit()
    before = conn.execute("SELECT COUNT(*) FROM preserved_rows").fetchone()[0]
    conn.close()

    migrated = db.connect(path)
    db.init_db(migrated)
    after = migrated.execute("SELECT COUNT(*) FROM preserved_rows").fetchone()[0]
    assert before == 1
    assert after == 1
    assert migrated.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
        "AND name='device_tokens'"
    ).fetchone()[0] == 1
    migrated.close()

    fresh = db.connect(tmp_path / "fresh-vault.db")
    db.init_db(fresh)
    assert fresh.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
        "AND name='device_tokens'"
    ).fetchone()[0] == 1
    fresh.close()


def test_registering_same_token_twice_is_one_row_with_new_last_seen(
        vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "shared-secret")
    seen: list[str] = []

    def clock():
        value = f"2026-01-01T00:00:{len(seen):02d}+00:00"
        seen.append(value)
        return value

    monkeypatch.setattr(db, "utcnow_iso", clock)
    with TestClient(receiver.create_app(vault)) as client:
        headers = {"x-health-secret": "shared-secret"}
        assert client.post(
            "/v1/push/register",
            json={"device_token": "token-one", "environment": "sandbox"},
            headers=headers,
        ).status_code == 200
        assert client.post(
            "/v1/push/register",
            json={"device_token": "token-one", "environment": "sandbox"},
            headers=headers,
        ).status_code == 200

    conn = vault.connect()
    row = conn.execute(
        "SELECT token, first_seen_at, last_seen_at, apns_environment "
        "FROM device_tokens"
    ).fetchone()
    count = conn.execute("SELECT COUNT(*) FROM device_tokens").fetchone()[0]
    conn.close()
    assert count == 1
    assert dict(row)["token"] == "token-one"
    assert dict(row)["apns_environment"] == "sandbox"
    assert dict(row)["first_seen_at"] != dict(row)["last_seen_at"]
    assert dict(row)["last_seen_at"] == seen[-1]


def test_registration_requires_shared_secret_and_writes_no_row(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "shared-secret")
    with TestClient(receiver.create_app(vault)) as client:
        response = client.post(
            "/v1/push/register",
            json={"device_token": "token-without-auth", "environment": "sandbox"},
        )
    assert response.status_code == 401
    conn = vault.connect()
    assert conn.execute("SELECT COUNT(*) FROM device_tokens").fetchone()[0] == 0
    conn.close()


def test_apns_jwt_has_es256_header_team_claim_and_verifiable_signature(tmp_path):
    sender, private_key = _sender(tmp_path)
    token = sender.jwt(issued_at=1_735_689_600)
    header_segment, claims_segment, signature_segment = token.split(".")
    header = _decode_segment(header_segment)
    claims = _decode_segment(claims_segment)
    padded = signature_segment + "=" * (-len(signature_segment) % 4)
    signature = base64.urlsafe_b64decode(padded)

    assert header == {"alg": "ES256", "kid": "TESTKEY1"}
    assert claims["iss"] == "TESTTEAM1"
    assert claims["iat"] == 1_735_689_600
    assert len(signature) == 64
    private_key.public_key().verify(
        encode_dss_signature(
            int.from_bytes(signature[:32], "big"),
            int.from_bytes(signature[32:], "big"),
        ),
        f"{header_segment}.{claims_segment}".encode("ascii"),
        ec.ECDSA(hashes.SHA256()),
    )


def test_serialized_push_payload_contains_only_ready_signal_and_turn_id(tmp_path):
    captured = {}

    def handler(request):
        captured["body"] = request.content
        captured["headers"] = request.headers
        return httpx.Response(200, request=request)

    sender, _ = _sender(tmp_path, handler=handler)
    question = "How did my resting heart rate change after Tuesday's run?"
    answer = "Your resting heart rate improved by 4 bpm."
    figure = "resting_heart_rate=52"
    assert sender.send("abcdef0123456789", "turn-123") is True
    serialized = captured["body"].decode("utf-8")
    payload = json.loads(serialized)

    assert "turn-123" in serialized
    assert question not in serialized
    assert answer not in serialized
    assert figure not in serialized
    assert set(payload) == {"aps", "turn_id"}
    assert set(payload["aps"]) == {"alert", "sound"}
    assert captured["headers"]["apns-topic"] == "com.example.HealthAdvisor"
    assert captured["headers"]["apns-push-type"] == "alert"
    assert captured["headers"]["apns-priority"] == "10"
    assert captured["headers"]["authorization"].startswith("bearer ")


@pytest.mark.parametrize("failure", [
    "status",
    "timeout",
    "dns",
])
def test_push_failures_are_announced_and_swallowed(tmp_path, caplog, failure):
    def handler(request):
        if failure == "status":
            return httpx.Response(503, request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("stub timeout", request=request)
        raise httpx.ConnectError("stub DNS failure", request=request)

    sender, _ = _sender(tmp_path, handler=handler)
    with caplog.at_level(logging.WARNING, logger=push.__name__):
        result = sender.send("abcdef0123456789", "turn-123")
    assert result is False
    assert "APNs push failed" in caplog.text
    assert "abcdef0123456789" not in caplog.text


@pytest.mark.parametrize("endpoint", [
    "https://api.push.apple.com.attacker.example",
    "http://api.push.apple.com",
])
def test_disapproved_apns_endpoints_are_refused(endpoint):
    with pytest.raises(ValueError):
        push.validate_apns_endpoint(endpoint)


def test_apns_configuration_has_no_ambient_defaults():
    with pytest.raises(RuntimeError, match="HA_APNS_KEY_PATH"):
        push.APNsConfig.from_env({})
