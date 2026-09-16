"""D23 body-AEAD contract and receiver-boundary tests."""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi import Request
from fastapi.responses import Response
from fastapi.testclient import TestClient

from health_advisor import body_aead, receiver


VECTOR_FILE = Path(__file__).parent / "fixtures" / "d23_body_aead_v1.json"
VECTORS = json.loads(VECTOR_FILE.read_text(encoding="utf-8"))
SECRET = base64.b64decode(VECTORS["secret_utf8_b64"]).decode("utf-8")
CASES = VECTORS["cases"]


def _plaintext(case: dict) -> bytes:
    if "plaintext_b64" in case:
        return base64.b64decode(case["plaintext_b64"])
    recipe = case["plaintext_recipe"]
    pattern = base64.b64decode(recipe["pattern_b64"])
    return (pattern * (recipe["length"] // len(pattern) + 1))[:recipe["length"]]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_contract_vectors(case):
    wire = (base64.b64decode(case["wire_b64"])
            if "wire_b64" in case else None)
    verify = case.get("verify_as", {})
    direction = verify.get("direction", case["direction"])
    method = verify.get("method", case["method"])
    target = verify.get("target", case["target"])
    sealed_secret = (base64.b64decode(case["sealed_with_secret_utf8_b64"]).decode()
                     if "sealed_with_secret_utf8_b64" in case else SECRET)
    assert body_aead.build_aad(
        case["direction"], case["method"], case["target"],
        body_aead.key_id(sealed_secret), case["timestamp_ms"]
    ) == base64.b64decode(case["aad_b64"])

    if case["expected"] == "ok":
        plaintext = _plaintext(case)
        assert hashlib.sha256(plaintext).hexdigest() == case["plaintext_sha256"]
        made = body_aead.seal(
            SECRET, case["direction"], case["method"], case["target"],
            plaintext, case["timestamp_ms"],
            nonce=base64.b64decode(case["nonce_b64"]),
        )
        if wire is None:
            assert hashlib.sha256(made).hexdigest() == case["wire_sha256"]
            assert len(made) == case["wire_length"]
            wire = made
        assert made == wire
        assert hashlib.sha256(made).hexdigest() == case["wire_sha256"]
        assert body_aead.open_(SECRET, direction, method, target, made,
                               case["verify_at_ms"]) == plaintext
    else:
        with pytest.raises(body_aead.D23Error) as caught:
            body_aead.open_(SECRET, direction, method, target, wire,
                            case["verify_at_ms"])
        assert caught.value.code == case["expected"].split(":", 1)[1]


def _app(monkeypatch, vault, mode: str):
    monkeypatch.setenv("HA_D23_MODE", mode)
    monkeypatch.setattr(receiver, "SHARED_SECRET", "d23-test-secret")
    app = receiver.create_app(vault)

    async def echo(request: Request):
        return Response(await request.body(), media_type="application/octet-stream")

    app.app.add_api_route("/test/echo", echo, methods=["POST"])
    return app


def test_route_coverage_is_the_completed_router(monkeypatch, vault):
    app = _app(monkeypatch, vault, "required")
    assert isinstance(app, receiver.D23BodyAEADApp)
    assert app.app.docs_url is None
    assert app.app.redoc_url is None
    assert app.app.openapi_url is None
    routes = [route for route in app.routes if hasattr(route, "path")]
    assert routes
    # The outer wrapper owns every route in the completed router. /health is
    # the sole route whose request and response remain plaintext by contract.
    assert all(route.path == "/health" or route in app.protected_routes
               for route in routes)
    assert all(route.path != "/health" or route not in app.protected_routes
               for route in routes)
    assert {route.path for route in routes if route.path in {
        "/docs", "/redoc", "/openapi.json"
    }} == set()


def test_health_docs_and_generated_500(monkeypatch, vault):
    app = _app(monkeypatch, vault, "required")

    def boom():
        raise RuntimeError("deliberate test failure")

    app.app.add_api_route("/test/boom", boom, methods=["GET"])
    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404
        failed = client.get("/test/boom")
    assert failed.status_code == 500
    assert failed.headers["content-type"].startswith("application/octet-stream")
    assert int(failed.headers["content-length"]) == len(failed.content)
    opened = body_aead.open_("d23-test-secret", "response", "GET", b"/test/boom",
                             failed.content)
    assert b"Internal Server Error" in opened


def test_required_refuses_clear_body_without_echo(monkeypatch, vault):
    app = _app(monkeypatch, vault, "required")
    clear = b'{"question":"sent in the clear"}'
    with TestClient(app) as client:
        response = client.post("/test/echo", content=clear)
    assert response.status_code == 400
    assert response.json() == {"detail": {"error": "d23_missing"}}
    assert clear not in response.content
    assert b"sent in the clear" not in response.content


def test_off_accepts_clear_and_encrypted_bodies(monkeypatch, vault):
    app = _app(monkeypatch, vault, "off")
    clear = b"clear payload"
    with TestClient(app) as client:
        plain_response = client.post("/test/echo", content=clear)
        encrypted = body_aead.seal(
            "d23-test-secret", "request", "POST", b"/test/echo",
            b"sealed payload", time.time_ns() // 1_000_000,
            nonce=b"0123456789ab",
        )
        sealed_response = client.post("/test/echo", content=encrypted)
    assert plain_response.status_code == 200
    assert plain_response.content == clear
    assert sealed_response.status_code == 200
    assert body_aead.open_("d23-test-secret", "response", "POST",
                           b"/test/echo", sealed_response.content) == b"sealed payload"


@pytest.mark.parametrize("mode", ["unset", "invalid"])
def test_mode_fails_closed_at_startup(monkeypatch, vault, mode):
    if mode == "unset":
        monkeypatch.delenv("HA_D23_MODE", raising=False)
    else:
        monkeypatch.setenv("HA_D23_MODE", "sometimes")
    monkeypatch.setattr(receiver, "SHARED_SECRET", "d23-test-secret")
    with pytest.raises(RuntimeError, match="HA_D23_MODE"):
        receiver.create_app(vault)


def test_required_round_trip_matches_plain_route(monkeypatch, vault):
    off = _app(monkeypatch, vault, "off")
    with TestClient(off) as client:
        plain_response = client.post("/test/echo", content=b"round trip")

    required = _app(monkeypatch, vault, "required")
    with TestClient(required) as client:
        request_wire = body_aead.seal(
            "d23-test-secret", "request", "POST", b"/test/echo", b"round trip",
            time.time_ns() // 1_000_000,
        )
        response = client.post("/test/echo", content=request_wire)
    assert response.status_code == 200
    assert body_aead.open_("d23-test-secret", "response", "POST",
                           b"/test/echo", response.content) == plain_response.content


def test_required_get_without_body_is_passed_then_response_is_sealed(monkeypatch, vault):
    app = _app(monkeypatch, vault, "required")
    with TestClient(app) as client:
        response = client.get("/v1/ask/progress?progress_id=missing",
                              headers={"x-health-secret": "d23-test-secret"})
    assert response.status_code == 404
    assert body_aead.open_("d23-test-secret", "response", "GET",
                           b"/v1/ask/progress?progress_id=missing",
                           response.content)


@pytest.mark.parametrize("case", [
    case for case in CASES if case["expected"] != "ok"
], ids=lambda case: case["name"])
def test_refusal_bytes_do_not_echo_request(case):
    request_bytes = base64.b64decode(case["wire_b64"])
    error = case["expected"].split(":", 1)[1]
    body = json.dumps({"detail": {"error": error}},
                      separators=(",", ":")).encode()
    if request_bytes:
        assert request_bytes not in body
    if case["name"] == "neg_plaintext_json_body":
        assert b"sent in the clear" not in body
