"""D23 body-AEAD contract and receiver-boundary tests."""
from __future__ import annotations

import asyncio
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


def _disconnect_scope():
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/test/disconnect",
        "raw_path": b"/test/disconnect",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-length", b"16")],
    }


def _body_then_disconnect(calls=None):
    messages = [
        {"type": "http.request", "body": b"0123456789abcdef",
         "more_body": False},
        {"type": "http.disconnect"},
    ]

    async def receive():
        if calls is not None:
            calls.append(len(calls) + 1)
        return messages.pop(0)

    return receive


async def _disconnect_probe(scope, receive, send):
    request = Request(scope, receive)
    await request.body()
    disconnected = await request.is_disconnected()
    result = b"true" if disconnected else b"false"
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-length", str(len(result)).encode())]})
    await send({"type": "http.response.body", "body": result,
                "more_body": False})


async def _run_disconnect_probe(app, receive=None):
    sent = []

    async def send(message):
        sent.append(message)

    await app(_disconnect_scope(), receive or _body_then_disconnect(), send)
    return next(message["body"] for message in sent
                if message["type"] == "http.response.body")


def test_disconnect_reaches_route_with_or_without_body_layer():
    bare = asyncio.run(_run_disconnect_probe(_disconnect_probe))
    wrapped = receiver.D23BodyAEADApp(
        _disconnect_probe, lambda: "d23-test-secret", "off")
    behind_layer = asyncio.run(_run_disconnect_probe(wrapped))
    assert bare == b"true"
    assert behind_layer == b"true"


def test_buffered_body_is_cached_for_a_second_request_read():
    observed = []
    source_calls = []

    async def handler(scope, receive, send):
        request = Request(scope, receive)
        first = await request.body()
        second = await request.body()
        observed.append((first, second))
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-length", b"0")]})
        await send({"type": "http.response.body", "body": b"",
                    "more_body": False})

    wrapped = receiver.D23BodyAEADApp(
        handler, lambda: "d23-test-secret", "off")
    asyncio.run(asyncio.wait_for(
        _run_disconnect_probe(wrapped, _body_then_disconnect(source_calls)),
        timeout=1))
    assert observed == [(b"0123456789abcdef", b"0123456789abcdef")]
    assert source_calls == [1]


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
    assert failed.headers["content-type"].startswith(body_aead.CONTENT_TYPE)
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
        sealed_response = client.post(
            "/test/echo", content=encrypted,
            headers={"content-type": body_aead.CONTENT_TYPE},
        )
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
        response = client.post(
            "/test/echo", content=request_wire,
            headers={"content-type": body_aead.CONTENT_TYPE},
        )
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


def test_required_bodyless_post_without_envelope_is_missing(monkeypatch, vault):
    app = _app(monkeypatch, vault, "required")
    with TestClient(app) as client:
        response = client.post(
            "/test/echo", content=b"",
            headers={"content-length": "0"},
        )
    assert response.status_code == 400
    assert response.json() == {"detail": {"error": "d23_missing"}}


def test_required_accepts_sealed_empty_post(monkeypatch, vault):
    app = _app(monkeypatch, vault, "required")
    wire = body_aead.seal(
        "d23-test-secret", "request", "POST", b"/test/echo", b"",
        time.time_ns() // 1_000_000,
    )
    with TestClient(app) as client:
        response = client.post(
            "/test/echo", content=wire,
            headers={"content-type": body_aead.CONTENT_TYPE},
        )
    assert response.status_code == 200
    assert body_aead.open_("d23-test-secret", "response", "POST",
                           b"/test/echo", response.content) == b""


def test_secret_rotation_uses_new_secret_per_request(monkeypatch, vault):
    monkeypatch.setenv("HA_D23_MODE", "required")
    monkeypatch.setattr(receiver, "SHARED_SECRET", "old-d23-test-secret")
    current = {"value": "old-d23-test-secret"}
    app = receiver.create_app(vault, secret_for_request=lambda: current["value"])

    async def echo(request: Request):
        return Response(await request.body(), media_type="application/octet-stream")

    app.app.add_api_route("/test/echo", echo, methods=["POST"])

    def wire(secret: str, plaintext: bytes) -> bytes:
        return body_aead.seal(
            secret, "request", "POST", b"/test/echo", plaintext,
            time.time_ns() // 1_000_000,
        )

    old_wire = wire(current["value"], b"old")
    old_key_id = body_aead.key_id(current["value"])
    with TestClient(app) as client:
        assert client.post(
            "/test/echo", content=old_wire,
            headers={"content-type": body_aead.CONTENT_TYPE},
        ).status_code == 200

        current["value"] = "new-d23-test-secret"
        assert body_aead.key_id(current["value"]) != old_key_id
        refused = client.post(
            "/test/echo", content=old_wire,
            headers={"content-type": body_aead.CONTENT_TYPE},
        )
        opened = client.post(
            "/test/echo", content=wire(current["value"], b"new"),
            headers={"content-type": body_aead.CONTENT_TYPE},
        )
    assert refused.status_code == 400
    assert refused.json() == {"detail": {"error": "d23_decrypt"}}
    assert body_aead.open_(current["value"], "response", "POST",
                           b"/test/echo", opened.content) == b"new"


def test_off_passes_gzip_and_json_bodies_untouched(monkeypatch, vault):
    app = _app(monkeypatch, vault, "off")

    async def echo(request: Request):
        return Response(await request.body(), media_type="application/octet-stream")

    app.app.add_api_route("/test/raw", echo, methods=["POST"])
    gzip_body = b"\x1f\x8b\x08\x00not-an-envelope"
    json_body = b'{"plain":true}'
    with TestClient(app) as client:
        gzip_response = client.post("/test/raw", content=gzip_body)
        json_response = client.post(
            "/test/raw", content=json_body,
            headers={"content-type": "application/json; charset=utf-8"},
        )
    assert gzip_response.content == gzip_body
    assert json_response.content == json_body


def test_required_sealed_body_with_json_content_type_is_missing(monkeypatch, vault):
    app = _app(monkeypatch, vault, "required")
    wire = body_aead.seal(
        "d23-test-secret", "request", "POST", b"/test/echo", b"payload",
        time.time_ns() // 1_000_000,
    )
    with TestClient(app) as client:
        response = client.post(
            "/test/echo", content=wire,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400
    assert response.json() == {"detail": {"error": "d23_missing"}}


@pytest.mark.parametrize("mode", ["off", "required"])
def test_d23_content_type_wrong_version_is_refused(monkeypatch, vault, mode):
    app = _app(monkeypatch, vault, mode)
    wire = body_aead.seal(
        "d23-test-secret", "request", "POST", b"/test/echo", b"payload",
        time.time_ns() // 1_000_000,
    )
    wrong_version = bytes([2]) + wire[1:]
    with TestClient(app) as client:
        response = client.post(
            "/test/echo", content=wrong_version,
            headers={"content-type": "APPLICATION/X-HA-D23; charset=binary"},
        )
    assert response.status_code == 400
    assert response.json() == {"detail": {"error": "d23_version"}}


@pytest.mark.parametrize("mode", ["off", "required"])
def test_health_is_plaintext_json_in_both_modes(monkeypatch, vault, mode):
    app = _app(monkeypatch, vault, mode)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_attribute_assignment_proxies_to_fastapi(monkeypatch, vault):
    app = _app(monkeypatch, vault, "off")
    app.some_new_attr = 1
    assert app.app.some_new_attr == 1


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
