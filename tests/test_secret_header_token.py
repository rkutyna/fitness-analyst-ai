"""X-Health-Secret carries a derived auth token, never the root secret.

The shared secret is also the HKDF root of the D23 body keys, so a header that
carries it lets any intermediary reading request headers derive both body keys.
The header therefore carries ``body_aead.auth_token(secret)``: HKDF under its
own info label, one-way, and useless as a key root. ``HA_SECRET_HEADER_MODE``
decides whether the raw secret is still accepted for clients that predate the
token (``raw_or_token``, the default) or refused (``token_only``).
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from health_advisor import body_aead, receiver


SECRET = "header-token-test-secret-value"
TOKEN = body_aead.auth_token(SECRET)
REQUIRES = (receiver._require_ask_secret, receiver._require_ingest_secret)


def _mode(monkeypatch, mode: str | None) -> None:
    if mode is None:
        monkeypatch.delenv("HA_SECRET_HEADER_MODE", raising=False)
    else:
        monkeypatch.setenv("HA_SECRET_HEADER_MODE", mode)


def _refusal(require, presented) -> HTTPException:
    with pytest.raises(HTTPException) as caught:
        require(presented)
    assert caught.value.status_code == 401
    return caught.value


@pytest.mark.parametrize("secret, expected", [
    ("test-secret-0123456789abcdef",
     "zkCXPI3v9YC-pia1hKgHWxcjGvu80Our3_xc3bFfqFo"),
    ("  padded secret \n",
     "Km55zNcrAJwZUBodK6lijHodqq8eDbdu5yTzrNtu8Mw"),
])
def test_auth_token_matches_the_client_contract_vectors(secret, expected):
    assert body_aead.auth_token(secret) == expected


def test_the_token_is_not_a_key_root():
    """Keys derived from what the header carries are not the body keys."""
    assert TOKEN != SECRET.strip()
    assert body_aead.derive_keys(TOKEN) != body_aead.derive_keys(SECRET)
    assert body_aead.key_id(TOKEN) != body_aead.key_id(SECRET)
    keys = body_aead.derive_keys(SECRET)
    assert TOKEN.encode("ascii") not in (keys.request, keys.response)
    assert body_aead.auth_token(TOKEN) != TOKEN


@pytest.mark.parametrize("mode", [None, "", "raw_or_token", "token_only"])
def test_token_authenticates_in_every_mode(monkeypatch, mode):
    _mode(monkeypatch, mode)
    monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
    for require in REQUIRES:
        require(TOKEN)


@pytest.mark.parametrize("mode", [None, "", "raw_or_token"])
def test_raw_secret_is_accepted_while_the_mode_allows_it(monkeypatch, mode):
    _mode(monkeypatch, mode)
    monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
    for require in REQUIRES:
        require(SECRET)


def test_raw_secret_is_refused_in_token_only_with_a_clear_detail(monkeypatch):
    _mode(monkeypatch, "token_only")
    monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
    for require in REQUIRES:
        refused = _refusal(require, SECRET)
        assert "token_only" in refused.detail
        assert "auth token" in refused.detail


@pytest.mark.parametrize("mode", ["raw_or_token", "token_only"])
@pytest.mark.parametrize("presented", [
    body_aead.auth_token("some-other-secret"),
    TOKEN[:-1] + ("A" if TOKEN[-1] != "A" else "B"),
    TOKEN + "=",
    "",
    None,
])
def test_a_wrong_token_is_refused(monkeypatch, mode, presented):
    _mode(monkeypatch, mode)
    monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
    for require in REQUIRES:
        assert _refusal(require, presented).detail == "missing or bad shared secret"


def test_an_empty_ask_secret_refuses_even_the_token_of_an_empty_secret(monkeypatch):
    _mode(monkeypatch, "raw_or_token")
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    _refusal(receiver._require_ask_secret, body_aead.auth_token(""))


@pytest.mark.parametrize("value", ["sometimes", "token", "RAW_OR_TOKEN", " token_only"])
def test_an_invalid_mode_refuses_at_startup(monkeypatch, vault, value):
    monkeypatch.setenv("HA_D23_MODE", "off")
    monkeypatch.setenv("HA_SECRET_HEADER_MODE", value)
    monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
    with pytest.raises(RuntimeError, match="HA_SECRET_HEADER_MODE"):
        receiver.create_app(vault)


def _empty_batch() -> bytes:
    return json.dumps({
        "protocol_version": 1,
        "device": {"id": "d", "name": "n", "model": "m"},
        "app_version": "1", "batch_id": "token-round-trip",
        "batch_sequence": 1, "sent_at": "2026-01-01T00:00:00Z",
        "anchors": [], "samples": [], "deletions": [], "workouts": [],
    }).encode("utf-8")


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def test_d23_round_trip_with_a_token_header(monkeypatch, vault):
    """Sealed request in, sealed response out, the header carrying only a token.

    The response is sealed with keys from the server's own secret. Were it
    sealed with keys from the header value, opening it with the secret would
    fail and opening it with the token would succeed; both are asserted.
    """
    monkeypatch.setenv("HA_D23_MODE", "required")
    _mode(monkeypatch, "token_only")
    monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
    app = receiver.create_app(vault)
    headers = {"x-health-secret": TOKEN, "content-type": body_aead.CONTENT_TYPE}
    with TestClient(app) as client:
        wire = body_aead.seal(SECRET, "request", "POST", b"/v1/ingest",
                              _empty_batch(), _now_ms())
        ingest = client.post("/v1/ingest", content=wire, headers=headers)
        conversation = client.get("/v1/conversation",
                                  headers={"x-health-secret": TOKEN})
        raw = client.get("/v1/conversation",
                         headers={"x-health-secret": SECRET})

    assert ingest.status_code == 200
    assert ingest.headers["content-type"] == body_aead.CONTENT_TYPE
    body = json.loads(body_aead.open_(SECRET, "response", "POST",
                                      b"/v1/ingest", ingest.content))
    assert body["batch_id"] == "token-round-trip"
    with pytest.raises(body_aead.D23Error):
        body_aead.open_(TOKEN, "response", "POST", b"/v1/ingest",
                        ingest.content)

    assert conversation.status_code == 200
    assert json.loads(body_aead.open_(
        SECRET, "response", "GET", b"/v1/conversation",
        conversation.content))["turns"] == []

    assert raw.status_code == 401
    refused = json.loads(body_aead.open_(
        SECRET, "response", "GET", b"/v1/conversation", raw.content))
    assert "token_only" in refused["detail"]


def test_request_bodies_sealed_with_header_derived_keys_are_refused(monkeypatch, vault):
    """A client that keyed D23 from the token instead of the secret fails."""
    monkeypatch.setenv("HA_D23_MODE", "required")
    _mode(monkeypatch, "token_only")
    monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
    app = receiver.create_app(vault)
    wire = body_aead.seal(TOKEN, "request", "POST", b"/v1/ingest",
                          _empty_batch(), _now_ms())
    with TestClient(app) as client:
        response = client.post(
            "/v1/ingest", content=wire,
            headers={"x-health-secret": TOKEN,
                     "content-type": body_aead.CONTENT_TYPE})
    assert response.status_code == 400
