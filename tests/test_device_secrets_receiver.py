"""T5 piece 9, E3: per-device D23 secrets at the receiver (consumer #426).

Every key here is generated per test run; none is a real device's or a real
token. Mutations named in a test's docstring were manually applied against
``health_advisor/receiver.py`` on 2026-09-26 and observed to turn that test
red.
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

from health_advisor import body_aead, device_auth, device_secrets, enrol, receiver


SECRET = "p9-receiver-test-instance-secret"
_MISSING = object()


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


class Phone:
    """A test stand-in for one phone with an enrolled P-256 device key."""

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


def _seal(secret: str, method: str, target: bytes, plaintext: bytes) -> bytes:
    return body_aead.seal(secret, "request", method, target, plaintext, _now_ms())


def _open(response, secret: str, method: str, target: bytes) -> dict:
    if response.headers.get("content-type") == body_aead.CONTENT_TYPE:
        return json.loads(body_aead.open_(secret, "response", method, target,
                                          response.content))
    return response.json()


def _token(secret: str) -> str:
    return body_aead.auth_token(secret)


def _batch(batch_id: str) -> bytes:
    return json.dumps({
        "protocol_version": 1,
        "device": {"id": "d", "name": "n", "model": "m"},
        "app_version": "1", "batch_id": batch_id,
        "batch_sequence": 1, "sent_at": "2026-01-01T00:00:00Z",
        "anchors": [], "samples": [], "deletions": [], "workouts": [],
    }).encode("utf-8")


def _signed_get(client, phone: Phone, secret: str, path: str = "/v1/conversation",
                **sign_kw) -> object:
    headers = {"x-health-secret": _token(secret)}
    headers.update(phone.sign("GET", path.encode("ascii"), b"", **sign_kw))
    return client.get(path, headers=headers)


def _signed_ingest(client, phone: Phone, secret: str, batch_id: str, *,
                   seal_under: str | None = None, **sign_kw):
    target = b"/v1/ingest"
    wire = _seal(seal_under if seal_under is not None else secret,
                "POST", target, _batch(batch_id))
    headers = {"x-health-secret": _token(secret),
               "content-type": body_aead.CONTENT_TYPE}
    headers.update(phone.sign("POST", target, wire, **sign_kw))
    return client.post("/v1/ingest", content=wire, headers=headers)


def _error(response) -> str:
    return response.json()["detail"]["error"]


def _pair(client, phone: Phone, store: "enrol.TokenStore", record, token, *,
         response_key: bytes | None = None, key_storage: str = "secure_enclave",
         signer: Phone | None = None, enrol_id: str | None = None):
    response_key = response_key or os.urandom(enrol.RESPONSE_KEY_BYTES)
    payload = {
        "v": 1, "token": token,
        "device_pub": _b64url(phone.public_x963),
        "key_storage": key_storage,
        "response_key": _b64url(response_key),
        "attestation": None,
    }
    plaintext = json.dumps(payload).encode("utf-8")
    pub = store.server_public_key(record)
    wire = enrol.seal_to(pub, enrol.info_for(record.id), plaintext)
    target = enrol.ENROL_PATH.encode("ascii")
    headers = {enrol.HEADER_ENROL_ID: enrol_id if enrol_id is not None else record.id}
    headers.update((signer or phone).sign("POST", target, wire, kid=phone.kid))
    response = client.post(enrol.ENROL_PATH, content=wire, headers=headers)
    return response, response_key


def _open_reply(response, response_key: bytes) -> dict:
    assert response.headers["content-type"] == enrol.ENROL_CONTENT_TYPE
    return json.loads(enrol.open_response(response_key, response.content))


def _pair_and_get_secret(client, store: "enrol.TokenStore", phone: Phone | None = None):
    phone = phone or Phone()
    record, token = store.mint()
    response, response_key = _pair(client, phone, store, record, token)
    assert response.status_code == 200, response.content
    reply = _open_reply(response, response_key)
    return phone, reply["secret"]


def _upgrade(client, phone: Phone, secret: str, *, device_mode: str) -> object:
    target = device_auth.ENROL_UPGRADE_PATH.encode("ascii")
    plaintext = json.dumps({"v": 1, "public_key": _b64url(phone.public_x963),
                            "key_storage": "secure_enclave"}).encode("utf-8")
    wire = _seal(secret, "POST", target, plaintext)
    headers = {"x-health-secret": _token(secret), "content-type": body_aead.CONTENT_TYPE}
    headers.update(phone.sign("POST", target, wire, kid=phone.kid))
    return client.post(device_auth.ENROL_UPGRADE_PATH, content=wire, headers=headers)


# --------------------------------------------------------------- fixtures


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "state" / "devices.json"


@pytest.fixture
def token_store_path(tmp_path):
    return tmp_path / "state" / "enrol-tokens.json"


@pytest.fixture
def secrets_path(tmp_path):
    return tmp_path / "state" / "device-secrets.json"


@pytest.fixture
def store(token_store_path):
    token_store_path.parent.mkdir(parents=True, exist_ok=True)
    return enrol.TokenStore(token_store_path)


@pytest.fixture
def make_app(monkeypatch, vault, registry_path, token_store_path, secrets_path):
    def _make(*, device_mode: str | None = "required", enrol_mode: str | None = "on",
              d23: str = "required", secret_mode: str | None = "per_device",
              secrets_file=_MISSING):
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        token_store_path.parent.mkdir(parents=True, exist_ok=True)
        secrets_path.parent.mkdir(parents=True, exist_ok=True)
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
        monkeypatch.setenv("HA_ENROL_TOKEN_FILE", str(token_store_path))
        if secret_mode is None:
            monkeypatch.delenv("HA_DEVICE_SECRET_MODE", raising=False)
        else:
            monkeypatch.setenv("HA_DEVICE_SECRET_MODE", secret_mode)
        path = str(secrets_path) if secrets_file is _MISSING else secrets_file
        if path is None:
            monkeypatch.delenv("HA_DEVICE_SECRETS_FILE", raising=False)
        else:
            monkeypatch.setenv("HA_DEVICE_SECRETS_FILE", path)
        monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
        return receiver.create_app(vault)
    return _make


# ---------------------------------------------------------------------- T1


@pytest.mark.parametrize("secret_mode", [None, "instance"])
def test_t1_instance_mode_is_fully_inert(make_app, secrets_path, store, secret_mode):
    """The single most important test here: a live instance runs with
    HA_DEVICE_SECRET_MODE unset. Nothing about piece 9 may change a single
    byte of its behaviour.

    Mutation: read/validate the per-device store unconditionally, regardless
    of mode (drop the ``if secret_mode == "per_device":`` guard around
    ``device_secrets.store_from_env()`` in ``create_app``) -- a garbage
    HA_DEVICE_SECRETS_FILE would then refuse startup even in 'instance'
    mode, and this test's ``make_app`` call would raise instead of
    returning.
    """
    secrets_path.write_text("{not json")
    app = make_app(secret_mode=secret_mode, secrets_file=str(secrets_path))
    assert app.device_secrets is None
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        response, response_key = _pair(client, phone, store, record, token)
        assert response.status_code == 200, response.content
        reply = _open_reply(response, response_key)
        assert reply["secret"] == SECRET
        conv = _signed_get(client, phone, SECRET)
    assert conv.status_code == 200


# ---------------------------------------------------------------------- T2


def test_t2_per_device_without_device_auth_refuses_at_startup(make_app):
    with pytest.raises(RuntimeError, match="HA_DEVICE_SECRET_MODE"):
        make_app(secret_mode="per_device", device_mode=None, enrol_mode=None)
    with pytest.raises(RuntimeError, match="HA_DEVICE_SECRET_MODE"):
        make_app(secret_mode="per_device", device_mode="off", enrol_mode=None)
    make_app(secret_mode="per_device", device_mode="required")  # sanity: starts


# ---------------------------------------------------------------------- T3


def test_t3_the_qr_secret_is_per_device_and_works_end_to_end(make_app, store):
    """Mutation: in ``_enrol_qr``, keep calling ``secret_for_request()``
    (the instance secret) even when a device-secret store is configured --
    ``secret != SECRET`` then fails immediately."""
    app = make_app()
    with TestClient(app) as client:
        phone, secret = _pair_and_get_secret(client, store)
        assert secret != SECRET

        get_resp = _signed_get(client, phone, secret)
        ingest_resp = _signed_ingest(client, phone, secret, "p9-t3")
    assert get_resp.status_code == 200
    assert ingest_resp.status_code == 200
    opened = _open(get_resp, secret, "GET", b"/v1/conversation")
    assert "turns" in opened
    # The response opens only under the phone's own secret.
    with pytest.raises(body_aead.D23Error):
        body_aead.open_(SECRET, "response", "GET", b"/v1/conversation",
                        get_resp.content)


# ---------------------------------------------------------------------- T4


def test_t4_unsigned_requests_never_select_a_device_secret(make_app, store):
    """An unsigned request never resolves a kid, so the per-device lookup
    (gated on a verified kid) can never run for it -- even when the caller
    happens to hold a valid per-device secret and its token.

    Mutation: build a reverse token->secret map and select a device secret
    by matching X-Health-Secret alone, without requiring a verified
    signature first -- the unsigned GET would then succeed (200) instead of
    401, and the unsigned D23-sealed-under-S ingest would open successfully
    instead of failing d23_decrypt.
    """
    app = make_app(device_mode="accept")
    with TestClient(app) as client:
        phone, secret = _pair_and_get_secret(client, store)

        unsigned_get = client.get(
            "/v1/conversation", headers={"x-health-secret": _token(secret)})
        wire_under_s = _seal(secret, "POST", b"/v1/ingest", _batch("t4"))
        unsigned_ingest = client.post(
            "/v1/ingest", content=wire_under_s,
            headers={"x-health-secret": _token(secret),
                     "content-type": body_aead.CONTENT_TYPE})
    assert unsigned_get.status_code == 401
    # d23="required" seals every response, errors included, under whatever
    # secret this request resolved to -- the instance secret here, since an
    # unsigned request never verifies a kid.
    assert (_open(unsigned_get, SECRET, "GET", b"/v1/conversation")["detail"]
           == receiver._BAD_SECRET_DETAIL)
    assert unsigned_ingest.status_code == 400
    assert _error(unsigned_ingest) == "d23_decrypt"


# ---------------------------------------------------------------------- T5


def test_t5_revocation_refuses_signed_and_unsigned_in_required(
        make_app, store, registry_path):
    """Mutation: resolve the per-device secret from a raw header-supplied
    kid instead of the kid ``device_auth.authenticate`` actually verified
    (which itself checks revocation) -- a revoked phone's signed request
    would then still be admitted instead of refused ``device_revoked``."""
    app = make_app(device_mode="required")
    with TestClient(app) as client:
        phone, secret = _pair_and_get_secret(client, store)
        assert _signed_get(client, phone, secret).status_code == 200
        assert device_auth.main(
            ["--registry", str(registry_path), "revoke", phone.kid]) == 0
        signed_after = _signed_get(client, phone, secret)
        unsigned_after = client.get(
            "/v1/conversation", headers={"x-health-secret": _token(secret)})
    assert signed_after.status_code == 401
    assert _error(signed_after) == "device_revoked"
    assert unsigned_after.status_code == 401
    assert _error(unsigned_after) == "device_sig_missing"


# ---------------------------------------------------------------------- T6


def test_t6_a_forged_kid_claim_cannot_borrow_another_kids_secret(make_app, store):
    """B signs a request whose header claims kid=A. ``device_auth`` verifies
    the signature against A's registered public key, which B's private key
    cannot produce a valid signature for, so this is refused before any
    secret is ever chosen -- A's own per-device secret is never touched.
    """
    app = make_app()
    with TestClient(app) as client:
        phone_a, secret_a = _pair_and_get_secret(client, store)
        phone_b, _secret_b = _pair_and_get_secret(client, store)
        target = b"/v1/conversation"
        headers = {"x-health-secret": _token(secret_a)}
        headers.update(phone_b.sign("GET", target, b"", kid=phone_a.kid))
        response = client.get("/v1/conversation", headers=headers)
    assert response.status_code == 401
    assert _error(response) == "device_sig_bad"


# ---------------------------------------------------------------------- T7


def test_t7_a_upgrade_only_phone_stays_on_the_instance_secret_beside_a_qr_phone(
        make_app, store):
    """A T4-style phone (enrolled via the shared-secret upgrade route, never
    through QR) has no entry in the device-secret store.

    Mutation: treat a missing per-device entry as a refusal instead of a
    fallback to the instance secret -- the legacy phone's SECRET-token
    requests would then break the moment per-device mode is turned on for
    the instance, even though it was never QR-paired.
    """
    app = make_app(device_mode="accept")
    with TestClient(app) as client:
        qr_phone, secret = _pair_and_get_secret(client, store)

        legacy_phone = Phone()
        enrol_resp = _upgrade(client, legacy_phone, SECRET, device_mode="accept")
        assert enrol_resp.status_code == 200, enrol_resp.content

        legacy_get = _signed_get(client, legacy_phone, SECRET)
        qr_get = _signed_get(client, qr_phone, secret)
    assert legacy_get.status_code == 200
    assert qr_get.status_code == 200


# ---------------------------------------------------------------------- T8


def test_t8_a_per_device_phone_presenting_the_instance_secret_is_admitted(
        make_app, store):
    """The rollback-then-re-pair case needs this fallback, not a refusal.

    Mutation: refuse (401/403) a signed, per-device-enrolled phone whose
    presented token doesn't match ITS OWN per-device secret, instead of
    silently leaving `secret` on the instance value -- a correctly signed
    request presenting a valid token (just not this kid's own) would then
    be refused instead of admitted on the instance secret.
    """
    app = make_app()
    with TestClient(app) as client:
        phone, secret = _pair_and_get_secret(client, store)
        assert secret != SECRET
        fallback = _signed_get(client, phone, SECRET)
    assert fallback.status_code == 200
    opened = _open(fallback, SECRET, "GET", b"/v1/conversation")
    assert "turns" in opened


# ---------------------------------------------------------------------- T9


def test_t9_a_re_pair_rotates_the_secret(make_app, store):
    """Mutation: have ``DeviceSecretStore.issue`` return the existing
    secret for an already-known kid instead of always minting a fresh one
    -- ``new_secret != old_secret`` would then fail, and the old secret
    would keep working after a re-pair meant to retire it."""
    app = make_app()
    phone = Phone()
    with TestClient(app) as client:
        record1, token1 = store.mint()
        response1, key1 = _pair(client, phone, store, record1, token1)
        old_secret = _open_reply(response1, key1)["secret"]

        record2, token2 = store.mint()
        response2, key2 = _pair(client, phone, store, record2, token2)
        reply2 = _open_reply(response2, key2)
        new_secret = reply2["secret"]
        assert reply2["created"] is False
        assert new_secret != old_secret

        old_get = _signed_get(client, phone, old_secret)
        new_get = _signed_get(client, phone, new_secret)
    assert old_get.status_code == 401
    assert new_get.status_code == 200


# --------------------------------------------------------------------- T10


def test_t10_the_request_secret_reaches_sync_and_async_routes_not_unsigned(
        make_app, store):
    """A sync ``def`` route runs on a worker thread via
    ``run_in_threadpool``; an async route runs on the event loop where the
    contextvar was set. Both must see the per-device secret; an unsigned
    request (no verified kid at all) must see only the instance secret.

    Mutation: read the D23-selected secret through a plain ``threading.local``
    instead of a ``ContextVar`` -- the async route (running on the event
    loop's task, not the worker thread the middleware itself executes on for
    this call) would then see the instance secret instead of S.
    """
    # accept, not required: the "unsigned sees SECRET" arm needs an unsigned
    # request to be admitted at all -- required refuses it outright before
    # any secret is chosen for a route to observe.
    app = make_app(device_mode="accept")

    @app.app.get("/v1/deployment-secret-probe-sync")
    def sync_probe():
        return {"token": body_aead.auth_token(receiver._shared_secret_for_request())}

    @app.app.get("/v1/deployment-secret-probe-async")
    async def async_probe():
        return {"token": body_aead.auth_token(receiver._shared_secret_for_request())}

    with TestClient(app) as client:
        phone, secret = _pair_and_get_secret(client, store)
        signed_sync = _signed_get(client, phone, secret,
                                  "/v1/deployment-secret-probe-sync")
        signed_async = _signed_get(client, phone, secret,
                                   "/v1/deployment-secret-probe-async")
        unsigned = client.get("/v1/deployment-secret-probe-sync",
                              headers={"x-health-secret": _token(SECRET)})

    sync_body = _open(signed_sync, secret, "GET", b"/v1/deployment-secret-probe-sync")
    async_body = _open(signed_async, secret, "GET", b"/v1/deployment-secret-probe-async")
    unsigned_body = _open(unsigned, SECRET, "GET", b"/v1/deployment-secret-probe-sync")
    assert sync_body["token"] == _token(secret)
    assert async_body["token"] == _token(secret)
    assert unsigned_body["token"] == _token(SECRET)


# --------------------------------------------------------------------- T11


@pytest.mark.parametrize("device_mode", ["accept", "required"])
def test_t11_a_qr_phone_reconfirms_via_upgrade_under_its_own_secret(
        make_app, store, device_mode):
    """Mutation: keep skipping ``authenticate()`` entirely on the upgrade
    path even when a device-secret store is configured (drop the
    ``or self.device_secrets is not None`` half of the upgrade-path
    condition) -- ``device_kid`` then stays None, no per-device secret is
    ever selected, and the D23 layer tries to open this body (sealed under
    S) with the instance secret instead -- ``d23_decrypt``, not 200.
    """
    app = make_app(device_mode=device_mode)
    with TestClient(app) as client:
        phone, secret = _pair_and_get_secret(client, store)
        response = _upgrade(client, phone, secret, device_mode=device_mode)
    assert response.status_code == 200, response.content
    body = _open(response, secret, "POST", device_auth.ENROL_UPGRADE_PATH.encode())
    assert body == {"ok": True, "kid": phone.kid, "created": False,
                    "mode": device_mode}


# --------------------------------------------------------------------- T12


def test_t12_a_store_corrupted_after_startup_fails_closed_only_for_signed(
        make_app, store, secrets_path):
    """Mutation: swallow the read error in the per-device lookup and fall
    back to the instance secret -- the signed request would then silently
    open under the wrong secret's derivation attempt (or succeed on the
    instance secret) instead of refusing 503."""
    app = make_app(device_mode="accept")
    with TestClient(app) as client:
        phone, secret = _pair_and_get_secret(client, store)
        secrets_path.write_text("{not json")

        signed = _signed_get(client, phone, secret)
        unsigned = client.get("/v1/conversation",
                              headers={"x-health-secret": _token(SECRET)})
    assert signed.status_code == 503
    assert _error(signed) == "device_secrets_unreadable"
    assert unsigned.status_code == 200


# --------------------------------------------------------------------- T13


def test_t13_a_store_write_failure_during_qr_pairing_refuses_without_fallback(
        make_app, store, monkeypatch):
    """Mutation: fall back to ``secret_for_request()`` (the instance secret)
    when ``issue`` raises, instead of refusing -- the pairing would then
    silently downgrade a per_device instance's phone to the shared secret
    it exists to avoid handing out."""
    app = make_app()

    def _boom(self, kid):
        raise OSError("disk full")

    monkeypatch.setattr(device_secrets.DeviceSecretStore, "issue", _boom)
    phone = Phone()
    record, token = store.mint()
    with TestClient(app) as client:
        response, _response_key = _pair(client, phone, store, record, token)
    assert response.status_code == 503
    body = response.json()
    assert body == {"detail": {"error": "device_secrets_unwritable"}}
