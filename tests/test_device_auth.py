"""Per-device request signatures (T4): enrolment, verification, revocation.

Every key here is generated per test run; none is a real device's.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from fastapi.testclient import TestClient

from health_advisor import body_aead, device_auth, receiver


SECRET = "device-auth-test-secret-value"
TOKEN = body_aead.auth_token(SECRET)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


class Phone:
    """A test stand-in for one enrolled phone."""

    def __init__(self):
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.public_x963 = self.key.public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint)
        self.kid = device_auth.kid_for(self.public_x963)

    def sign(self, method: str, target: bytes, wire: bytes, *,
             ts: int | None = None, kid: str | None = None) -> dict:
        ts = _now_ms() if ts is None else ts
        kid = self.kid if kid is None else kid
        message = device_auth.signing_input(
            kid, ts, method, target, hashlib.sha256(wire).hexdigest())
        r, s = decode_dss_signature(
            self.key.sign(message, ec.ECDSA(hashes.SHA256())))
        raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        return {"x-ha-device-kid": kid, "x-ha-device-ts": str(ts),
                "x-ha-device-sig": _b64url(raw)}


def _sealed(method: str, target: bytes, plaintext: bytes) -> bytes:
    return body_aead.seal(SECRET, "request", method, target, plaintext, _now_ms())


def _open(response, method: str, target: bytes) -> dict:
    if response.headers.get("content-type") == body_aead.CONTENT_TYPE:
        return json.loads(body_aead.open_(SECRET, "response", method, target,
                                          response.content))
    return response.json()


def _batch(batch_id: str) -> bytes:
    return json.dumps({
        "protocol_version": 1,
        "device": {"id": "d", "name": "n", "model": "m"},
        "app_version": "1", "batch_id": batch_id,
        "batch_sequence": 1, "sent_at": "2026-01-01T00:00:00Z",
        "anchors": [], "samples": [], "deletions": [], "workouts": [],
    }).encode("utf-8")


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "state" / "devices.json"


@pytest.fixture
def make_app(monkeypatch, vault, registry_path):
    def _make(mode: str | None, d23: str = "required"):
        registry_path.parent.mkdir(exist_ok=True)
        monkeypatch.setenv("HA_D23_MODE", d23)
        monkeypatch.setenv("HA_SECRET_HEADER_MODE", "token_only")
        monkeypatch.setenv("HA_DEVICE_REGISTRY_FILE", str(registry_path))
        if mode is None:
            monkeypatch.delenv("HA_DEVICE_AUTH_MODE", raising=False)
        else:
            monkeypatch.setenv("HA_DEVICE_AUTH_MODE", mode)
        monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
        return receiver.create_app(vault)
    return _make


def _enrol(client, phone: Phone, *, key_storage: str = "keychain",
           signer: Phone | None = None, seal: bool = True, token: str = TOKEN):
    target = device_auth.ENROL_UPGRADE_PATH.encode("ascii")
    plaintext = json.dumps({"v": 1, "public_key": _b64url(phone.public_x963),
                            "key_storage": key_storage}).encode("utf-8")
    wire = _sealed("POST", target, plaintext) if seal else plaintext
    headers = {"x-health-secret": token}
    if seal:
        headers["content-type"] = body_aead.CONTENT_TYPE
    headers.update((signer or phone).sign("POST", target, wire, kid=phone.kid))
    response = client.post(device_auth.ENROL_UPGRADE_PATH, content=wire,
                           headers=headers)
    return response, _open(response, "POST", target)


def _signed_get(client, phone: Phone, path: str = "/v1/conversation", **sign_kw):
    headers = {"x-health-secret": TOKEN}
    headers.update(phone.sign("GET", path.encode("ascii"), b"", **sign_kw))
    return client.get(path, headers=headers)


def _signed_ingest(client, phone: Phone, batch_id: str, *,
                   signed_plaintext: bytes | None = None, **sign_kw):
    target = b"/v1/ingest"
    wire = _sealed("POST", target, _batch(batch_id))
    signed_wire = wire if signed_plaintext is None else _sealed(
        "POST", target, signed_plaintext)
    headers = {"x-health-secret": TOKEN, "content-type": body_aead.CONTENT_TYPE}
    headers.update(phone.sign("POST", target, signed_wire, **sign_kw))
    return client.post("/v1/ingest", content=wire, headers=headers)


def _error(response) -> str:
    return response.json()["detail"]["error"]


# ------------------------------------------------------------------ modes


@pytest.mark.parametrize("value", ["on", "ACCEPT", " required", "true", "0"])
def test_an_invalid_mode_refuses_at_startup(make_app, value):
    with pytest.raises(RuntimeError, match="HA_DEVICE_AUTH_MODE"):
        make_app(value)


@pytest.mark.parametrize("mode", ["accept", "required"])
def test_an_enabled_mode_without_a_registry_refuses_at_startup(
        make_app, monkeypatch, vault, mode):
    make_app(mode)  # sanity: the same settings start with the registry named
    monkeypatch.delenv("HA_DEVICE_REGISTRY_FILE")
    with pytest.raises(RuntimeError, match="HA_DEVICE_REGISTRY_FILE"):
        receiver.create_app(vault)


def test_an_unreadable_registry_refuses_at_startup(make_app, registry_path):
    registry_path.parent.mkdir(exist_ok=True)
    registry_path.write_text("{not json")
    with pytest.raises(RuntimeError, match="HA_DEVICE_REGISTRY_FILE"):
        make_app("accept")


@pytest.mark.parametrize("mode", [None, "", "off"])
def test_off_reads_no_header_and_adds_no_route(make_app, mode):
    """off is today's receiver: device headers, even garbage ones, change no byte."""
    app = make_app(mode)
    assert app.device_auth is None
    paths = {getattr(route, "path", None) for route in app.routes}
    assert device_auth.ENROL_UPGRADE_PATH not in paths
    phone = Phone()
    garbage = {"x-ha-device-kid": "not-a-kid", "x-ha-device-ts": "x",
               "x-ha-device-sig": "!!"}
    with TestClient(app) as client:
        plain = client.get("/v1/conversation", headers={"x-health-secret": TOKEN})
        noisy = client.get("/v1/conversation",
                           headers={"x-health-secret": TOKEN, **garbage})
        signed = _signed_get(client, phone)
        enrol, _ = _enrol(client, phone)
        stock_404 = client.post("/v1/no-such-route", content=_sealed(
            "POST", b"/v1/no-such-route", b"{}"),
            headers={"x-health-secret": TOKEN,
                     "content-type": body_aead.CONTENT_TYPE})
    assert plain.status_code == noisy.status_code == signed.status_code == 200
    assert (_open(plain, "GET", b"/v1/conversation")
            == _open(noisy, "GET", b"/v1/conversation")
            == _open(signed, "GET", b"/v1/conversation"))
    assert {k: v for k, v in plain.headers.items() if k != "content-length"}.keys() \
        == {k: v for k, v in noisy.headers.items() if k != "content-length"}.keys()
    # The enrolment route answers exactly as any unknown route does.
    assert enrol.status_code == stock_404.status_code == 404
    assert (_open(enrol, "POST", device_auth.ENROL_UPGRADE_PATH.encode())
            == _open(stock_404, "POST", b"/v1/no-such-route"))


# ------------------------------------------------------- enrol then sign


def test_enrol_through_the_token_channel_then_signed_requests_pass(
        make_app, registry_path):
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        response, body = _enrol(client, phone, key_storage="secure_enclave")
        assert response.status_code == 200, body
        assert body == {"ok": True, "kid": phone.kid, "created": True,
                        "mode": "accept"}
        again, body = _enrol(client, phone)
        assert again.status_code == 200 and body["created"] is False

        get = _signed_get(client, phone)
        query = _signed_get(client, phone, "/v1/conversation?limit=5")
        ingest = _signed_ingest(client, phone, "signed-ingest")
        # accept: the token alone still works.
        unsigned = client.get("/v1/conversation",
                              headers={"x-health-secret": TOKEN})

    assert get.status_code == 200
    assert query.status_code == 200
    assert ingest.status_code == 200
    assert _open(ingest, "POST", b"/v1/ingest")["batch_id"] == "signed-ingest"
    assert unsigned.status_code == 200
    stored = json.loads(registry_path.read_text())
    assert [d["kid"] for d in stored["devices"]] == [phone.kid]
    assert stored["devices"][0]["key_storage"] == "secure_enclave"
    assert stored["devices"][0]["via"] == "upgrade"
    assert stored["devices"][0]["revoked_at"] is None


def test_enrolment_needs_the_token(make_app):
    app = make_app("accept")
    with TestClient(app) as client:
        response, body = _enrol(client, Phone(), token="wrong")
    assert response.status_code == 401


def test_enrolment_must_arrive_sealed(make_app, registry_path):
    """The token crosses the edge in a header; only a sealed body proves the secret."""
    app = make_app("accept", d23="off")
    with TestClient(app) as client:
        response, body = _enrol(client, Phone(), seal=False)
    assert response.status_code == 400
    assert body["detail"]["error"] == "device_enrol_unsealed"
    assert not registry_path.exists()


def test_enrolment_needs_proof_of_possession(make_app, registry_path):
    app = make_app("accept")
    phone, impostor = Phone(), Phone()
    with TestClient(app) as client:
        response, body = _enrol(client, phone, signer=impostor)
    assert response.status_code == 401
    assert body["detail"]["error"] == "device_sig_bad"
    assert not registry_path.exists()


# ------------------------------------------------------------ refusals


@pytest.mark.parametrize("mode", ["accept", "required"])
def test_a_bad_signature_is_refused(make_app, mode):
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        assert _enrol(client, phone)[0].status_code == 200
    app = make_app(mode)
    headers = {"x-health-secret": TOKEN}
    headers.update(phone.sign("GET", b"/v1/conversation", b""))
    sig = bytearray(base64.urlsafe_b64decode(headers["x-ha-device-sig"] + "=="))
    sig[-1] ^= 0x01
    headers["x-ha-device-sig"] = _b64url(bytes(sig))
    with TestClient(app) as client:
        response = client.get("/v1/conversation", headers=headers)
        other_key = client.get("/v1/conversation", headers={
            "x-health-secret": TOKEN,
            **Phone().sign("GET", b"/v1/conversation", b"", kid=phone.kid)})
    assert response.status_code == 401 and _error(response) == "device_sig_bad"
    assert other_key.status_code == 401 and _error(other_key) == "device_sig_bad"


@pytest.mark.parametrize("mode", ["accept", "required"])
def test_a_wrong_kid_is_refused(make_app, mode):
    app = make_app(mode)
    stranger = Phone()  # never enrolled
    with TestClient(app) as client:
        response = _signed_get(client, stranger)
    assert response.status_code == 401
    assert _error(response) == "device_unknown_kid"


def test_a_tampered_body_is_refused(make_app):
    """The signature covers the body: a sealed body swapped under it fails."""
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        assert _enrol(client, phone)[0].status_code == 200
        tampered = _signed_ingest(client, phone, "what-was-sent",
                                  signed_plaintext=_batch("what-was-signed"))
    assert tampered.status_code == 401
    assert _error(tampered) == "device_sig_bad"


def test_a_signature_for_another_method_or_target_is_refused(make_app):
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        assert _enrol(client, phone)[0].status_code == 200
        wrong_target = client.get("/v1/conversation", headers={
            "x-health-secret": TOKEN,
            **phone.sign("GET", b"/v1/conversation?limit=1", b"")})
        wrong_method = client.get("/v1/conversation", headers={
            "x-health-secret": TOKEN,
            **phone.sign("DELETE", b"/v1/conversation", b"")})
    assert _error(wrong_target) == "device_sig_bad"
    assert _error(wrong_method) == "device_sig_bad"


@pytest.mark.parametrize("offset_s, expected", [
    (-601, "device_sig_stale"), (601, "device_sig_stale"),
    (-3600, "device_sig_stale"), (-590, None), (590, None),
])
def test_the_timestamp_window_is_d23s(make_app, offset_s, expected):
    assert device_auth.SKEW_SECONDS == body_aead.SKEW_SECONDS == 600
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        assert _enrol(client, phone)[0].status_code == 200
        response = _signed_get(client, phone, ts=_now_ms() + offset_s * 1000)
    if expected is None:
        assert response.status_code == 200
    else:
        assert response.status_code == 401 and _error(response) == expected


@pytest.mark.parametrize("headers", [
    {"x-ha-device-kid": "A" * 22},
    {"x-ha-device-kid": "A" * 22, "x-ha-device-ts": "1"},
    {"x-ha-device-kid": "short", "x-ha-device-ts": "1", "x-ha-device-sig": "AA"},
    {"x-ha-device-kid": "A" * 22, "x-ha-device-ts": "-1",
     "x-ha-device-sig": _b64url(b"\x01" * 64)},
    {"x-ha-device-kid": "A" * 22, "x-ha-device-ts": "1",
     "x-ha-device-sig": _b64url(b"\x01" * 63)},
])
def test_partial_or_malformed_headers_are_refused_not_ignored(make_app, headers):
    app = make_app("accept")
    with TestClient(app) as client:
        response = client.get("/v1/conversation",
                              headers={"x-health-secret": TOKEN, **headers})
    assert response.status_code == 401
    assert _error(response) == "device_sig_malformed"


# ------------------------------------------------------------- required


def test_required_retires_the_secret_as_an_authenticator(make_app):
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        assert _enrol(client, phone)[0].status_code == 200
    app = make_app("required")
    newcomer = Phone()
    with TestClient(app) as client:
        unsigned = client.get("/v1/conversation", headers={"x-health-secret": TOKEN})
        unsigned_ingest = client.post(
            "/v1/ingest", content=_sealed("POST", b"/v1/ingest", _batch("u")),
            headers={"x-health-secret": TOKEN,
                     "content-type": body_aead.CONTENT_TYPE})
        signed = _signed_get(client, phone)
        signed_ingest = _signed_ingest(client, phone, "required-ingest")
        health = client.get("/health")
        reconfirm, reconfirm_body = _enrol(client, phone)
        new_key, new_key_body = _enrol(client, newcomer)
        newcomer_get = _signed_get(client, newcomer)
    assert unsigned.status_code == 401 and _error(unsigned) == "device_sig_missing"
    assert _error(unsigned_ingest) == "device_sig_missing"
    assert signed.status_code == 200
    assert signed_ingest.status_code == 200
    assert health.status_code == 200
    assert reconfirm.status_code == 200 and reconfirm_body["created"] is False
    assert new_key.status_code == 403
    assert new_key_body["detail"]["error"] == "device_enrol_closed"
    assert _error(newcomer_get) == "device_unknown_kid"


def test_required_covers_routes_mounted_after_create_app(make_app):
    """A deployment's own routes sit behind the same layer."""
    app = make_app("required")

    @app.app.get("/v1/deployment-extra")
    def extra():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/v1/deployment-extra",
                              headers={"x-health-secret": TOKEN})
    assert _error(response) == "device_sig_missing"


# ----------------------------------------------------------- revocation


def test_revocation_refuses_one_device_and_leaves_the_other(
        make_app, registry_path, capsys):
    app = make_app("accept")
    lost, kept = Phone(), Phone()
    with TestClient(app) as client:
        assert _enrol(client, lost)[0].status_code == 200
        assert _enrol(client, kept, key_storage="secure_enclave")[0].status_code == 200
    app = make_app("required")
    with TestClient(app) as client:
        assert _signed_get(client, lost).status_code == 200
        # The operator revokes while the receiver runs; no restart.
        assert device_auth.main(["--registry", str(registry_path),
                                 "revoke", lost.kid]) == 0
        assert "other devices unaffected" in capsys.readouterr().out
        refused = _signed_get(client, lost)
        still = _signed_get(client, kept)
        re_enrol, re_enrol_body = _enrol(client, lost)
    assert refused.status_code == 401 and _error(refused) == "device_revoked"
    assert still.status_code == 200
    assert re_enrol.status_code == 403
    assert re_enrol_body["detail"]["error"] == "device_revoked"

    assert device_auth.main(["--registry", str(registry_path),
                             "revoke", lost.kid]) == 0
    assert "already revoked" in capsys.readouterr().out
    assert device_auth.main(["--registry", str(registry_path),
                             "revoke", "A" * 22]) == 1
    assert device_auth.main(["--registry", str(registry_path), "list"]) == 0
    listing = capsys.readouterr().out
    assert f"{lost.kid}  revoked" in listing
    assert f"{kept.kid}  active" in listing


def test_accept_mode_still_admits_a_revoked_phone_without_a_signature(
        make_app, registry_path):
    """Why revocation needs `required`: in accept the token alone still works."""
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        assert _enrol(client, phone)[0].status_code == 200
        device_auth.DeviceRegistry(registry_path).revoke(phone.kid)
        assert _error(_signed_get(client, phone)) == "device_revoked"
        unsigned = client.get("/v1/conversation", headers={"x-health-secret": TOKEN})
    assert unsigned.status_code == 200


def test_a_registry_edited_by_hand_to_mismatch_fails_closed(make_app, registry_path):
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        assert _enrol(client, phone)[0].status_code == 200
        data = json.loads(registry_path.read_text())
        data["devices"][0]["public_key"] = _b64url(Phone().public_x963)
        registry_path.write_text(json.dumps(data))
        response = _signed_get(client, phone)
    assert response.status_code == 503
    assert _error(response) == "device_registry_unreadable"


# ------------------------------------------------------ the wire contract


def test_the_signing_input_vector():
    """The same bytes the iOS client's DeviceCredentialTests pins."""
    # The generator point of P-256: a public constant, nobody's key.
    g = bytes.fromhex(
        "04"
        "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"
        "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5")
    assert device_auth.kid_for(g) == "aYvqY9xEo0RmP_FCmuoQhA"
    assert device_auth.body_sha256(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert device_auth.signing_input(
        "aYvqY9xEo0RmP_FCmuoQhA", 1700000000000, "post",
        b"/v1/ingest?x=1", device_auth.body_sha256(b"{}")) == (
        b"ha-device-sig-v1\naYvqY9xEo0RmP_FCmuoQhA\n1700000000000\nPOST\n"
        b"/v1/ingest?x=1\n"
        b"44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a")
    with pytest.raises(ValueError):
        device_auth.signing_input("k", 1, "GET", b"/a\nb", "00")
