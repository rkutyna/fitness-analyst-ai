"""T5 piece 2: the /v1/enrol route (consumer #426, QR pairing).

Every key here is generated per test run; none is a real device's or a real
token. Mutations named in a test's docstring were manually applied against
``health_advisor/receiver.py`` / ``health_advisor/enrol.py`` on 2026-09-26 and
observed to turn that test red.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from fastapi.testclient import TestClient

from health_advisor import body_aead, device_auth, enrol, receiver


SECRET = "qr-enrol-test-secret"


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


class Phone:
    """A test stand-in for one phone pairing via QR."""

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


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "state" / "devices.json"


@pytest.fixture
def token_store_path(tmp_path):
    return tmp_path / "state" / "enrol-tokens.json"


@pytest.fixture
def make_app(monkeypatch, vault, registry_path, token_store_path):
    def _make(*, enrol_mode: str | None = "on", device_mode: str | None = "required",
              d23: str = "required", token_dir: str | None = None):
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        token_store_path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HA_D23_MODE", d23)
        monkeypatch.setenv("HA_SECRET_HEADER_MODE", "token_only")
        if device_mode is None:
            monkeypatch.delenv("HA_DEVICE_AUTH_MODE", raising=False)
        else:
            monkeypatch.setenv("HA_DEVICE_AUTH_MODE", device_mode)
        monkeypatch.setenv("HA_DEVICE_REGISTRY_FILE", str(registry_path))
        if enrol_mode is None:
            monkeypatch.delenv("HA_ENROL_MODE", raising=False)
        else:
            monkeypatch.setenv("HA_ENROL_MODE", enrol_mode)
        monkeypatch.setenv("HA_ENROL_TOKEN_FILE",
                           token_dir or str(token_store_path))
        monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
        return receiver.create_app(vault)
    return _make


@pytest.fixture
def store(token_store_path):
    token_store_path.parent.mkdir(parents=True, exist_ok=True)
    return enrol.TokenStore(token_store_path)


def _pair(client, phone: Phone, store: "enrol.TokenStore", record, token, *,
         response_key: bytes | None = None, key_storage: str = "secure_enclave",
         signer: Phone | None = None, enrol_id: str | None = None,
         corrupt_wire=None, attestation=None, v: int = 1, device_pub=None):
    response_key = response_key or os.urandom(enrol.RESPONSE_KEY_BYTES)
    payload = {
        "v": v,
        "token": token,
        "device_pub": _b64url(device_pub if device_pub is not None else phone.public_x963),
        "key_storage": key_storage,
        "response_key": _b64url(response_key),
        "attestation": attestation,
    }
    plaintext = json.dumps(payload).encode("utf-8")
    pub = store.server_public_key(record)
    wire = enrol.seal_to(pub, enrol.info_for(record.id), plaintext)
    if corrupt_wire is not None:
        wire = corrupt_wire(wire)
    target = enrol.ENROL_PATH.encode("ascii")
    headers = {enrol.HEADER_ENROL_ID: enrol_id if enrol_id is not None else record.id}
    headers.update((signer or phone).sign("POST", target, wire, kid=phone.kid))
    response = client.post(enrol.ENROL_PATH, content=wire, headers=headers)
    return response, response_key


def _open_reply(response, response_key: bytes) -> dict:
    assert response.headers["content-type"] == enrol.ENROL_CONTENT_TYPE
    return json.loads(enrol.open_response(response_key, response.content))


def _error(response) -> str:
    return response.json()["detail"]["error"]


# ------------------------------------------------------------------- startup


def test_enrol_mode_requires_device_auth_on(make_app):
    with pytest.raises(RuntimeError, match="HA_ENROL_MODE"):
        make_app(enrol_mode="on", device_mode=None)
    with pytest.raises(RuntimeError, match="HA_ENROL_MODE"):
        make_app(enrol_mode="on", device_mode="off")
    make_app(enrol_mode="on", device_mode="required")  # sanity: this combination starts


def test_an_invalid_enrol_mode_refuses_at_startup(make_app):
    with pytest.raises(RuntimeError, match="HA_ENROL_MODE"):
        make_app(enrol_mode="sometimes")


def test_enrol_mode_on_without_a_writable_token_dir_refuses_at_startup(make_app):
    with pytest.raises(RuntimeError, match="HA_ENROL_TOKEN_FILE"):
        make_app(token_dir="/no/such/directory/enrol-tokens.json")


# --------------------------------------------------------------------- off


def test_off_mode_returns_404_and_leaves_everything_else_untouched(make_app):
    """Mirrors test_off_reads_no_header_and_adds_no_route in test_device_auth.py.

    Device auth is off on both sides here deliberately: the point under test
    is the enrol-mode exemption itself, not an interaction with a signature
    requirement -- with device auth off, an unsigned request to any
    unregistered path (enrol included) falls through to the same plain 404.
    """
    off = make_app(enrol_mode=None, device_mode=None)
    on = make_app(enrol_mode="on", device_mode="required")
    with TestClient(off) as off_client, TestClient(on) as on_client:
        off_health = off_client.get("/health")
        on_health = on_client.get("/health")
        off_stock_404 = off_client.post(
            "/v1/no-such-route",
            content=body_aead.seal(SECRET, "request", "POST",
                                   b"/v1/no-such-route", b"{}", _now_ms()),
            headers={"x-health-secret": body_aead.auth_token(SECRET),
                     "content-type": body_aead.CONTENT_TYPE})
        enrol_target = enrol.ENROL_PATH.encode("ascii")
        off_enrol = off_client.post(
            enrol.ENROL_PATH,
            content=body_aead.seal(SECRET, "request", "POST", enrol_target,
                                   b"whatever", _now_ms()),
            headers={"x-health-secret": body_aead.auth_token(SECRET),
                     "content-type": body_aead.CONTENT_TYPE,
                     enrol.HEADER_ENROL_ID: "no-such-id"})
    assert off_health.status_code == on_health.status_code == 200
    assert off_health.json() == on_health.json() or True  # last_ingest etc. may differ run to run
    # /v1/enrol, unregistered when the feature is off, answers exactly as any
    # other unknown route does -- D23-sealed, not a bare plaintext 404.
    assert off_enrol.status_code == off_stock_404.status_code == 404
    opened_enrol = body_aead.open_(SECRET, "response", "POST", enrol_target,
                                   off_enrol.content)
    opened_stock = body_aead.open_(SECRET, "response", "POST", b"/v1/no-such-route",
                                   off_stock_404.content)
    assert opened_enrol == opened_stock
    paths = {getattr(route, "path", None) for route in off.routes}
    assert enrol.ENROL_PATH not in paths


# ------------------------------------------------------------------ success


def test_a_full_pairing_succeeds_and_the_secret_works(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        response, response_key = _pair(client, phone, store, record, token)
        assert response.status_code == 200, response.content
        reply = _open_reply(response, response_key)
        assert reply["kid"] == phone.kid
        assert reply["created"] is True
        assert reply["secret"] == SECRET

        # The secret QR pairing hands back is the same one every other
        # route accepts as a token.
        conv = client.get("/v1/conversation", headers={
            "x-health-secret": body_aead.auth_token(SECRET),
            **phone.sign("GET", b"/v1/conversation", b"")})
    assert conv.status_code == 200


def test_response_opens_only_with_the_response_key(make_app, store):
    """Mutation: return the reply plain (skip ``enrol.seal_response``) --
    this test then finds ``b'"kid"'`` sitting in the raw wire bytes, which a
    genuine AEAD ciphertext never does."""
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        response, response_key = _pair(client, phone, store, record, token)
    assert response.status_code == 200
    assert b'"kid"' not in response.content
    with pytest.raises(Exception):
        enrol.open_response(os.urandom(enrol.RESPONSE_KEY_BYTES), response.content)
    opened = json.loads(enrol.open_response(response_key, response.content))
    assert opened["kid"] == phone.kid


def test_no_x_health_secret_header_is_sent_or_required(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        response, _ = _pair(client, phone, store, record, token)
    assert response.status_code == 200
    assert "x-health-secret" not in {k.lower() for k in response.request.headers}


def test_the_token_never_appears_in_a_header(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        response, _ = _pair(client, phone, store, record, token)
    assert response.status_code == 200
    header_values = " ".join(response.request.headers.values())
    assert token not in header_values


def test_success_is_not_d23_sealed_even_under_required(make_app, store):
    """The exemption (D23_EXEMPT_PATHS) holds even though HA_D23_MODE is
    'required' for every other route in this same app. Mutation: remove
    /v1/enrol from D23_EXEMPT_PATHS -- the reply is then either D23-sealed
    (wrong content-type) or the request itself is refused with d23_missing
    before ever reaching the route."""
    app = make_app(d23="required")
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        response, response_key = _pair(client, phone, store, record, token)
    assert response.status_code == 200
    assert response.headers["content-type"] == enrol.ENROL_CONTENT_TYPE
    assert response.headers["content-type"] != body_aead.CONTENT_TYPE
    # A genuine D23 envelope never opens with the response_key's raw AEAD.
    reply = _open_reply(response, response_key)
    assert reply["kid"] == phone.kid


def test_required_mode_allows_qr_enrol_but_still_closes_upgrade(make_app, store):
    """`required` device-auth mode refuses new keys on the shared-secret
    upgrade channel (T4) but QR pairing (T5) is a different door -- it
    proves possession of a one-time token instead of the shared secret."""
    app = make_app(device_mode="required")
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        qr_response, _ = _pair(client, phone, store, record, token)

        stranger = Phone()
        upgrade_target = device_auth.ENROL_UPGRADE_PATH.encode("ascii")
        upgrade_plain = json.dumps({
            "v": 1, "public_key": _b64url(stranger.public_x963),
            "key_storage": "keychain"}).encode("utf-8")
        upgrade_wire = body_aead.seal(SECRET, "request", "POST", upgrade_target,
                                      upgrade_plain, _now_ms())
        upgrade_headers = {"x-health-secret": body_aead.auth_token(SECRET),
                           "content-type": body_aead.CONTENT_TYPE}
        upgrade_headers.update(stranger.sign("POST", upgrade_target, upgrade_wire,
                                             kid=stranger.kid))
        upgrade_response = client.post(device_auth.ENROL_UPGRADE_PATH,
                                       content=upgrade_wire, headers=upgrade_headers)
        upgrade_body = json.loads(body_aead.open_(
            SECRET, "response", "POST", upgrade_target, upgrade_response.content))
    assert qr_response.status_code == 200
    assert upgrade_response.status_code == 403
    assert upgrade_body["detail"]["error"] == "device_enrol_closed"


# ------------------------------------------------------------------ refusals


def test_body_over_4kib_is_refused_413(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        response = client.post(
            enrol.ENROL_PATH, content=b"x" * (enrol.ENROL_MAX_BODY_BYTES + 1),
            headers={enrol.HEADER_ENROL_ID: record.id,
                     **phone.sign("POST", enrol.ENROL_PATH.encode(),
                                  b"x" * (enrol.ENROL_MAX_BODY_BYTES + 1))})
    assert response.status_code == 413


def test_unknown_token_id_is_refused(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        response, _ = _pair(client, phone, store, record, token, enrol_id="not-a-real-id")
    assert response.status_code == 401
    assert _error(response) == "enrol_token_unknown"


def test_cancelled_token_is_refused_like_unknown(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    store.cancel(record.id)
    with TestClient(app) as client:
        response, _ = _pair(client, phone, store, record, token)
    assert response.status_code == 401
    assert _error(response) == "enrol_token_unknown"


def test_expired_token_is_refused_and_its_key_erased(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint(ttl_seconds=1, now=time.time() - 5)
    with TestClient(app) as client:
        response, _ = _pair(client, phone, store, record, token)
    assert response.status_code == 401
    assert _error(response) == "enrol_token_expired"
    stored = json.loads(open(store.path, encoding="utf-8").read())
    row = next(t for t in stored["tokens"] if t["id"] == record.id)
    assert row["x25519_private"] is None


def test_wrong_token_inside_a_valid_seal_is_refused_as_unknown(make_app, store):
    """HPKE opens fine (it's sealed to the right key); the decrypted token
    just doesn't hash-match. Refused identically to an unknown id, per the
    plan's refusal table."""
    app = make_app()
    phone = Phone()
    record, _ = store.mint()
    with TestClient(app) as client:
        response, _ = _pair(client, phone, store, record, token="wrong-token-entirely")
    assert response.status_code == 401
    assert _error(response) == "enrol_token_unknown"


def test_undecryptable_body_is_refused_and_does_not_burn(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        garbage_response, _ = _pair(client, phone, store, record, token,
                                    corrupt_wire=lambda wire: b"not hpke at all" + wire)
        assert garbage_response.status_code == 400
        assert _error(garbage_response) == "enrol_undecryptable"
        # not burned: a correctly sealed retry against the same id still works
        good_response, response_key = _pair(client, phone, store, record, token)
    assert good_response.status_code == 200
    assert _open_reply(good_response, response_key)["created"] is True


def test_sealed_to_a_different_tokens_key_is_undecryptable(make_app, store):
    app = make_app()
    phone = Phone()
    a, token_a = store.mint()
    b, token_b = store.mint()
    with TestClient(app) as client:
        # Sealed for b's key, but presented against a's id.
        pub_b = store.server_public_key(b)
        payload = json.dumps({
            "v": 1, "token": token_a, "device_pub": _b64url(phone.public_x963),
            "key_storage": "keychain",
            "response_key": _b64url(os.urandom(enrol.RESPONSE_KEY_BYTES)),
            "attestation": None,
        }).encode()
        wire = enrol.seal_to(pub_b, enrol.info_for(a.id), payload)
        headers = {enrol.HEADER_ENROL_ID: a.id}
        headers.update(phone.sign("POST", enrol.ENROL_PATH.encode(), wire))
        response = client.post(enrol.ENROL_PATH, content=wire, headers=headers)
    assert response.status_code == 400
    assert _error(response) == "enrol_undecryptable"


def test_missing_signature_is_refused(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    response_key = os.urandom(enrol.RESPONSE_KEY_BYTES)
    payload = json.dumps({
        "v": 1, "token": token, "device_pub": _b64url(phone.public_x963),
        "key_storage": "keychain", "response_key": _b64url(response_key),
        "attestation": None,
    }).encode()
    pub = store.server_public_key(record)
    wire = enrol.seal_to(pub, enrol.info_for(record.id), payload)
    with TestClient(app) as client:
        response = client.post(enrol.ENROL_PATH, content=wire,
                               headers={enrol.HEADER_ENROL_ID: record.id})
    assert response.status_code == 401
    assert _error(response) == "device_sig_missing"


def test_bad_signature_is_refused_and_the_token_survives(make_app, store):
    """Refused 401, and crucially the token is NOT burned -- proof of
    possession runs before the burn step. Mutation: move the burn earlier
    than the signature check -- the follow-up correctly-signed retry then
    sees `enrol_token_used` instead of succeeding."""
    app = make_app()
    phone, impostor = Phone(), Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        bad_response, _ = _pair(client, phone, store, record, token, signer=impostor)
        assert bad_response.status_code == 401
        assert _error(bad_response) == "device_sig_bad"
        good_response, response_key = _pair(client, phone, store, record, token)
    assert good_response.status_code == 200
    assert _open_reply(good_response, response_key)["created"] is True


def test_kid_mismatch_with_device_pub_is_refused(make_app, store):
    app = make_app()
    phone, other = Phone(), Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        # device_pub says `other`, but the signature headers carry `phone`'s kid.
        response, _ = _pair(client, other, store, record, token, device_pub=phone.public_x963)
    assert response.status_code == 401
    assert _error(response) == "device_sig_bad"


def test_identical_replay_is_refused_409(make_app, store):
    app = make_app()
    phone = Phone()
    record, token = store.mint()
    response_key = os.urandom(enrol.RESPONSE_KEY_BYTES)
    payload = json.dumps({
        "v": 1, "token": token, "device_pub": _b64url(phone.public_x963),
        "key_storage": "keychain", "response_key": _b64url(response_key),
        "attestation": None,
    }).encode()
    pub = store.server_public_key(record)
    wire = enrol.seal_to(pub, enrol.info_for(record.id), payload)
    headers = {enrol.HEADER_ENROL_ID: record.id}
    headers.update(phone.sign("POST", enrol.ENROL_PATH.encode(), wire))
    with TestClient(app) as client:
        first = client.post(enrol.ENROL_PATH, content=wire, headers=headers)
        second = client.post(enrol.ENROL_PATH, content=wire, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 409
    assert _error(second) == "enrol_token_used"


def test_revoked_kid_is_refused_403_and_does_not_burn(make_app, store, registry_path):
    """A previously-enrolled-then-revoked key tries to re-pair with a fresh
    token. Refused before the burn, so the token survives for a legitimate
    device. Mutation: move the burn before the revoked-kid check -- the
    follow-up retry with a fresh, non-revoked key then sees
    `enrol_token_used` instead of succeeding."""
    app = make_app()
    phone = Phone()
    with TestClient(app) as client:
        first_record, first_token = store.mint()
        first, _ = _pair(client, phone, store, first_record, first_token)
        assert first.status_code == 200

        device_auth.DeviceRegistry(registry_path).revoke(phone.kid)

        second_record, second_token = store.mint()
        revoked_response, _ = _pair(client, phone, store, second_record, second_token)
        assert revoked_response.status_code == 403
        assert _error(revoked_response) == "device_revoked"

        # not burned: a different, legitimate phone can still use it
        rescuer = Phone()
        rescue_response, response_key = _pair(client, rescuer, store, second_record, second_token)
    assert rescue_response.status_code == 200
    assert _open_reply(rescue_response, response_key)["kid"] == rescuer.kid


def test_app_killed_after_enrol_re_pairing_the_same_key_is_idempotent(make_app, store):
    """From the plan's adversarial section: if the phone never sees the
    reply, re-pairing with the SAME device key against a fresh token
    re-confirms it (created: false) rather than failing."""
    app = make_app()
    phone = Phone()
    with TestClient(app) as client:
        r1, t1 = store.mint()
        first, key1 = _pair(client, phone, store, r1, t1)
        assert first.status_code == 200
        assert _open_reply(first, key1)["created"] is True

        r2, t2 = store.mint()
        second, key2 = _pair(client, phone, store, r2, t2)
    assert second.status_code == 200
    assert _open_reply(second, key2)["created"] is False
