"""The verification scope (#652): a device key that may call only the probe.

A registry entry with ``"scope": "verify"`` is refused on every route but
``GET /v1/verify/probe``; the probe reads no vault, checks the shared secret
after the device layer exactly as other routes do, and answers a fixed body.
Only the host-side ``verify-enrol`` command writes a scope, and neither
enrolment route will confirm or pair a scoped key. Entries with no scope are
full devices, exactly as before.

Every key here is generated per test run.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import re
import socket
import stat
import threading
import time

import pytest
from cryptography.hazmat.primitives import serialization
from fastapi.testclient import TestClient

from health_advisor import body_aead, device_auth, enrol, receiver
from tests.test_device_auth import Phone, _enrol, _signed_get
from tests.test_enrol_route import Phone as QRPhone, _pair

SECRET = "device-auth-test-secret-value"  # tests.test_device_auth's
OTHER_SECRET = "another-instance-secret-value"
TOKEN = body_aead.auth_token(SECRET)
OTHER_TOKEN = body_aead.auth_token(OTHER_SECRET)
PROBE = device_auth.VERIFY_PROBE_PATH
SECRET_REFUSED = {"detail": "missing or bad shared secret"}


@pytest.fixture
def state(tmp_path):
    path = tmp_path / "state"
    path.mkdir()
    return path


@pytest.fixture
def make_app(monkeypatch, vault, state):
    def _make(mode: str | None = "required", *, d23: str = "required",
              enrol_on: bool = False, per_device: bool = False):
        monkeypatch.setenv("HA_D23_MODE", d23)
        monkeypatch.setenv("HA_SECRET_HEADER_MODE", "token_only")
        monkeypatch.setenv("HA_DEVICE_REGISTRY_FILE", str(state / "devices.json"))
        if mode is None:
            monkeypatch.delenv("HA_DEVICE_AUTH_MODE", raising=False)
        else:
            monkeypatch.setenv("HA_DEVICE_AUTH_MODE", mode)
        if enrol_on:
            monkeypatch.setenv("HA_ENROL_MODE", "on")
            monkeypatch.setenv("HA_ENROL_TOKEN_FILE", str(state / "enrol-tokens.json"))
        else:
            monkeypatch.delenv("HA_ENROL_MODE", raising=False)
        if per_device:
            monkeypatch.setenv("HA_DEVICE_SECRET_MODE", "per_device")
            monkeypatch.setenv("HA_DEVICE_SECRETS_FILE", str(state / "device-secrets.json"))
        else:
            monkeypatch.delenv("HA_DEVICE_SECRET_MODE", raising=False)
        monkeypatch.setattr(receiver, "SHARED_SECRET", SECRET)
        return receiver.create_app(vault)
    return _make


def _cli(state, *args) -> int:
    return device_auth.main(["--registry", str(state / "devices.json"), *args])


@pytest.fixture
def verify_key(state, capsys):
    """Enrol a verify key through the real CLI; return the loaded private key."""
    key_file = state / "verify-probe.key"
    assert _cli(state, "verify-enrol", "--key-file", str(key_file)) == 0
    capsys.readouterr()
    return device_auth.load_key_file(str(key_file))


def _headers(key, method: str, path: str, token: str = TOKEN, **kw) -> dict:
    headers = {"x-health-secret": token}
    headers.update(device_auth.sign_request(key, method, path.encode("ascii"), **kw))
    return headers


def _body(response, method: str, path: str, secret: str = SECRET):
    if response.headers.get("content-type") == body_aead.CONTENT_TYPE:
        return json.loads(body_aead.open_(secret, "response", method,
                                          path.encode("ascii"), response.content))
    return response.json()


def _error(response):
    return response.json()["detail"]["error"]


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


# ------------------------------------------------------------ route table


@pytest.mark.parametrize("mode", ["accept", "required"])
def test_a_verify_key_is_refused_on_every_route_but_the_probe(make_app, verify_key, mode):
    """Walks the app's own route table, so a route added later is covered.

    Mutation: drop the ``device.permits`` check in DeviceAuth.authenticate ->
    the routes answer for themselves (200s, 400s, 422s) and this goes red.
    """
    app = make_app(mode, enrol_on=True, per_device=True)

    @app.app.get("/v1/deployment-extra")  # a deployment's own route, mounted later
    def extra():
        return {"ok": True}

    exempt = set(app.exempt_paths)
    assert exempt == {"/health", enrol.ENROL_PATH}, exempt
    checked = []
    with TestClient(app) as client:
        for route in app.routes:
            methods = getattr(route, "methods", None)
            if not methods or route.path in exempt:
                continue
            for method in sorted(methods):
                if (method, route.path) == ("GET", PROBE):
                    continue
                path = _concrete(route.path)
                response = client.request(method, path,
                                          headers=_headers(verify_key, method, path))
                assert (response.status_code, _error(response)) == (
                    403, "device_scope_forbidden"), (method, path, response.content)
                checked.append((method, route.path))
        for method, path in (("HEAD", PROBE), ("POST", PROBE), ("PUT", PROBE),
                             ("DELETE", PROBE), ("GET", PROBE + "/"),
                             ("GET", "/v1/verify/probe2"), ("GET", "/v1/no-such-route")):
            response = client.request(method, path, headers=_headers(verify_key, method, path))
            assert response.status_code == 403, (method, path)
            if method != "HEAD":  # a HEAD answer carries no body to read
                assert _error(response) == "device_scope_forbidden", (method, path)
        probe = client.get(PROBE, headers=_headers(verify_key, "GET", PROBE))
    assert probe.status_code == 200
    covered = {path for _, path in checked}
    for must in (device_auth.ENROL_UPGRADE_PATH, "/v1/ingest", "/v1/conversation",
                 "/v1/device-token", "/v1/deployment-extra"):
        assert must in covered, must
    assert len(checked) >= 10, checked


def test_the_scope_check_runs_after_the_signature(make_app, verify_key):
    """A forged or stale verify signature gets the signature's refusal, not the scope's."""
    app = make_app("required")
    with TestClient(app) as client:
        stale = client.get("/v1/conversation", headers=_headers(
            verify_key, "GET", "/v1/conversation", now_ms=time.time_ns() // 1_000_000 - 3_600_000))
        stale_probe = client.get(PROBE, headers=_headers(
            verify_key, "GET", PROBE, now_ms=time.time_ns() // 1_000_000 - 3_600_000))
        wrong_target = client.get(PROBE, headers=_headers(verify_key, "GET", "/v1/conversation"))
    assert _error(stale) == _error(stale_probe) == "device_sig_stale"
    assert _error(wrong_target) == "device_sig_bad"


def test_authenticate_refuses_a_scoped_device_given_no_path(state, verify_key):
    registry = device_auth.DeviceRegistry(state / "devices.json")
    auth = device_auth.DeviceAuth("required", registry)
    headers = {k.lower(): v for k, v in
               device_auth.sign_request(verify_key, "GET", PROBE.encode()).items()}
    with pytest.raises(device_auth.DeviceAuthError) as refused:
        auth.authenticate(headers.get, method="GET", target=PROBE.encode(), body=b"")
    assert (refused.value.code, refused.value.status) == ("device_scope_forbidden", 403)
    assert auth.authenticate(headers.get, method="GET", target=PROBE.encode(),
                             body=b"", path=PROBE) is not None


# ------------------------------------------------------------- the probe


@pytest.mark.parametrize("d23", ["required", "off"])
def test_the_probe_answers_own_secret_200_fixed_body_and_cross_secret_401_secret_body(
        make_app, verify_key, d23):
    """Mutation: make the probe return vault content (e.g. add rows or the
    vault path to its body) -> the exact-body assertion goes red. Mutation:
    drop ``_require_ask_secret`` from the probe -> the cross-secret probe
    answers 200 and goes red."""
    app = make_app("required", d23=d23)
    with TestClient(app) as client:
        own = client.get(PROBE, headers=_headers(verify_key, "GET", PROBE))
        cross = client.get(PROBE, headers=_headers(verify_key, "GET", PROBE, token=OTHER_TOKEN))
        missing = client.get(PROBE, headers={
            k: v for k, v in _headers(verify_key, "GET", PROBE).items()
            if k != "x-health-secret"})
    assert own.status_code == 200
    assert _body(own, "GET", PROBE) == {"ok": True}
    assert cross.status_code == 401
    assert _body(cross, "GET", PROBE) == SECRET_REFUSED
    assert missing.status_code == 401
    assert _body(missing, "GET", PROBE) == SECRET_REFUSED


def test_the_probe_still_needs_a_device_signature_in_required(make_app):
    app = make_app("required")
    with TestClient(app) as client:
        unsigned = client.get(PROBE, headers={"x-health-secret": TOKEN})
    assert (unsigned.status_code, _error(unsigned)) == (401, "device_sig_missing")


def test_a_phone_may_call_the_probe_too(make_app):
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        response, _ = _enrol(client, phone)
        assert response.status_code == 200
        probe = _signed_get(client, phone, PROBE)
    assert probe.status_code == 200
    assert _body(probe, "GET", PROBE) == {"ok": True}


@pytest.mark.parametrize("mode", [None, "", "off"])
def test_off_has_no_probe_route(make_app, mode):
    app = make_app(mode)
    paths = {getattr(route, "path", None) for route in app.routes}
    assert PROBE not in paths


# -------------------------------------------- enrolment cannot widen a scope


def test_the_upgrade_route_will_not_confirm_a_verify_key(make_app, state, verify_key):
    """Accept mode even admits NEW keys through upgrade; a scoped one still is not.

    Mutation: drop the scope refusal in DeviceRegistry.enrol -> 200 and red.
    """
    app = make_app("accept")
    public = device_auth._public_point(verify_key)
    target = device_auth.ENROL_UPGRADE_PATH.encode("ascii")
    plaintext = json.dumps({"v": 1, "public_key": device_auth.b64url_encode(public),
                            "key_storage": "keychain"}).encode()
    wire = body_aead.seal(SECRET, "request", "POST", target, plaintext, time.time_ns() // 1_000_000)
    before = (state / "devices.json").read_bytes()
    with TestClient(app) as client:
        response = client.post(device_auth.ENROL_UPGRADE_PATH, content=wire, headers={
            "content-type": body_aead.CONTENT_TYPE,
            **_headers(verify_key, "POST", device_auth.ENROL_UPGRADE_PATH, body=wire)})
    assert response.status_code == 403
    assert _body(response, "POST", device_auth.ENROL_UPGRADE_PATH)["detail"]["error"] \
        == "device_scope_forbidden"
    assert (state / "devices.json").read_bytes() == before


def test_qr_pairing_will_not_pair_a_verify_key_and_leaves_the_token_live(
        make_app, state, verify_key):
    """Mutation: drop the scope pre-check in _enrol_qr -> the token burns
    (status 'used') before the registry refuses, and this goes red."""
    app = make_app("required", enrol_on=True, per_device=True)
    store = enrol.TokenStore(state / "enrol-tokens.json")
    record, token = store.mint()
    impostor = QRPhone()
    impostor.key = verify_key
    impostor.public_x963 = device_auth._public_point(verify_key)
    impostor.kid = device_auth.kid_for(impostor.public_x963)
    with TestClient(app) as client:
        response, _ = _pair(client, impostor, store, record, token)
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "device_scope_forbidden"
    assert store.status(record.id).state == "live"
    assert not (state / "device-secrets.json").exists()


def test_registry_enrol_refuses_a_scoped_key_and_enrol_scoped_refuses_a_phone(state):
    registry = device_auth.DeviceRegistry(state / "devices.json")
    phone, verifier = Phone(), Phone()
    registry.enrol(phone.public_x963, via="token", key_storage="keychain", allow_new=True)
    device, created = registry.enrol_scoped(verifier.public_x963, scope="verify")
    assert created and device.scope == "verify"
    assert registry.enrol_scoped(verifier.public_x963, scope="verify") == (device, False)
    for allow_new in (True, False):
        with pytest.raises(device_auth.DeviceAuthError, match="device_scope_forbidden"):
            registry.enrol(verifier.public_x963, via="token", key_storage="keychain",
                           allow_new=allow_new)
    # A phone can be neither downgraded to verify nor re-scoped.
    with pytest.raises(device_auth.DeviceAuthError, match="device_scope_conflict"):
        registry.enrol_scoped(phone.public_x963, scope="verify")
    with pytest.raises(ValueError):
        registry.enrol_scoped(Phone().public_x963, scope="full")
    with pytest.raises(ValueError):
        registry.enrol_scoped(Phone().public_x963, scope=None)
    assert registry.get(phone.kid).scope is None
    registry.revoke(verifier.kid)
    with pytest.raises(device_auth.DeviceAuthError, match="device_revoked"):
        registry.enrol_scoped(verifier.public_x963, scope="verify")


# ------------------------------------------------ scope-less entries unchanged


def test_a_phone_entry_is_written_without_a_scope_key(state):
    """Mutation: write asdict(device) unfiltered -> "scope": null appears and an
    engine that predates scopes could no longer read a phone-only registry."""
    registry = device_auth.DeviceRegistry(state / "devices.json")
    registry.enrol(Phone().public_x963, via="token", key_storage="keychain", allow_new=True)
    registry.enrol_scoped(Phone().public_x963, scope="verify")
    entries = json.loads((state / "devices.json").read_text())["devices"]
    phone = [e for e in entries if e["via"] == "token"][0]
    verifier = [e for e in entries if e["via"] == device_auth.VERIFY_VIA][0]
    assert set(phone) == {"kid", "public_key", "enrolled_at", "via", "key_storage",
                          "revoked_at"}
    assert verifier["scope"] == "verify"


def test_an_unknown_scope_in_the_registry_fails_closed(make_app, state):
    registry = device_auth.DeviceRegistry(state / "devices.json")
    registry.enrol_scoped(Phone().public_x963, scope="verify")
    data = json.loads((state / "devices.json").read_text())
    data["devices"][0]["scope"] = "admin"
    (state / "devices.json").write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="HA_DEVICE_REGISTRY_FILE"):
        make_app("required")


def test_a_phone_beside_a_verify_key_is_unchanged(make_app, state, verify_key):
    app = make_app("accept")
    phone = Phone()
    with TestClient(app) as client:
        response, _ = _enrol(client, phone)
        assert response.status_code == 200
        conv = _signed_get(client, phone, "/v1/conversation")
    assert conv.status_code == 200


def test_revoking_a_verify_key_refuses_the_probe(make_app, state, verify_key):
    kid = device_auth.kid_for(device_auth._public_point(verify_key))
    assert _cli(state, "revoke", "--", kid) == 0
    app = make_app("required")
    with TestClient(app) as client:
        probe = client.get(PROBE, headers=_headers(verify_key, "GET", PROBE))
    assert (probe.status_code, _error(probe)) == (401, "device_revoked")


# ------------------------------------------------------------- the CLI


def test_verify_enrol_writes_a_600_key_never_prints_it_and_is_idempotent(state, capsys):
    key_file = state / "verify-probe.key"
    assert _cli(state, "verify-enrol", "--key-file", str(key_file)) == 0
    first = capsys.readouterr()
    assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600
    pem = key_file.read_text()
    assert "PRIVATE KEY" in pem
    body = pem.split("-----")[2].strip()
    for line in body.splitlines():
        assert line not in first.out + first.err
    entries = json.loads((state / "devices.json").read_text())["devices"]
    assert len(entries) == 1 and entries[0]["scope"] == "verify"
    assert entries[0]["kid"] in first.out and "enrolled" in first.out
    assert _cli(state, "verify-enrol", "--key-file", str(key_file)) == 0
    assert "already enrolled" in capsys.readouterr().out
    assert key_file.read_text() == pem
    assert len(json.loads((state / "devices.json").read_text())["devices"]) == 1


def test_verify_enrol_refuses_a_loose_key_file_and_a_phones_key(state, capsys):
    loose = state / "loose.key"
    loose.write_text("x")
    loose.chmod(0o644)
    assert _cli(state, "verify-enrol", "--key-file", str(loose)) == 1
    # A key already enrolled as a phone is refused, and the file stays.
    phone_key = state / "phone.key"
    phone = Phone()
    fd = os.open(phone_key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, phone.key.private_bytes(serialization.Encoding.PEM,
                                         serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()))
    os.close(fd)
    registry = device_auth.DeviceRegistry(state / "devices.json")
    registry.enrol(phone.public_x963, via="token", key_storage="keychain", allow_new=True)
    assert _cli(state, "verify-enrol", "--key-file", str(phone_key)) == 1
    assert "device_scope_conflict" in capsys.readouterr().err
    assert registry.get(phone.kid).scope is None
    assert phone_key.exists()


def test_verify_enrol_removes_a_fresh_key_the_registry_refused(state, monkeypatch):
    def refuse(self, public, *, scope):
        raise device_auth.DeviceAuthError("device_scope_conflict", 409)
    monkeypatch.setattr(device_auth.DeviceRegistry, "enrol_scoped", refuse)
    key_file = state / "verify-probe.key"
    assert _cli(state, "verify-enrol", "--key-file", str(key_file)) == 1
    assert not key_file.exists()


def test_verify_enrol_will_not_follow_a_symlink(state, tmp_path):
    """Mutation: drop O_NOFOLLOW from load_key_file -> the linked key enrols."""
    target = tmp_path / "elsewhere.key"
    device_auth._write_new_key_file(str(target))
    link = state / "verify-probe.key"
    link.symlink_to(target)
    assert _cli(state, "verify-enrol", "--key-file", str(link)) == 1
    assert not (state / "devices.json").exists()
    dangling = state / "dangling.key"
    dangling.symlink_to(tmp_path / "nowhere.key")
    assert _cli(state, "verify-enrol", "--key-file", str(dangling)) == 1
    assert not (tmp_path / "nowhere.key").exists()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def serve():
    import uvicorn

    servers = []

    def _serve(app) -> str:
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                               log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        assert server.started
        servers.append((server, thread))
        return f"http://127.0.0.1:{port}"

    yield _serve
    for server, thread in servers:
        server.should_exit = True
        thread.join(10)


def _probe_cli(state, base_url, stdin_text, capsys, key_file=None):
    key_file = key_file or state / "verify-probe.key"
    real_stdin = device_auth.sys.stdin
    device_auth.sys.stdin = io.StringIO(stdin_text)
    try:
        rc = device_auth.main(["verify-probe", "--key-file", str(key_file),
                               "--base-url", base_url])
    finally:
        device_auth.sys.stdin = real_stdin
    captured = capsys.readouterr()
    return rc, captured.out.strip(), captured.err


@pytest.mark.parametrize("d23", ["required", "off"])
def test_verify_probe_over_the_wire_own_200_cross_401_secret(
        make_app, state, verify_key, serve, capsys, d23):
    """The CLI signs, sends, opens a sealed reply with the instance secret and
    prints only "<status> <class>". Mutation: classify by status alone (drop
    the secret-detail check) -> the cross line is no longer 'secret-refused'."""
    base = serve(make_app("required", d23=d23))
    header_line = f'header = "X-Health-Secret: {TOKEN}"\n'
    rc, own, err = _probe_cli(state, base, header_line, capsys)
    assert (rc, own) == (0, "200 ok"), err
    rc, cross, err = _probe_cli(state, base, OTHER_TOKEN + "\n", capsys)
    assert (rc, cross) == (0, "401 secret-refused"), err
    for printed in (own, cross):
        assert TOKEN not in printed and OTHER_TOKEN not in printed


def test_verify_probe_reports_an_unenrolled_or_revoked_key_as_device_auth(
        make_app, state, verify_key, serve, capsys):
    base = serve(make_app("required"))
    kid = device_auth.kid_for(device_auth._public_point(verify_key))
    assert _cli(state, "revoke", "--", kid) == 0
    capsys.readouterr()
    rc, line, _ = _probe_cli(state, base, TOKEN + "\n", capsys)
    assert (rc, line) == (0, "401 device-auth device_revoked")
    stranger = state / "stranger.key"
    device_auth._write_new_key_file(str(stranger))
    rc, line, _ = _probe_cli(state, base, TOKEN + "\n", capsys, key_file=stranger)
    assert (rc, line) == (0, "401 device-auth device_unknown_kid")


def test_verify_probe_refuses_bad_input_without_sending(state, verify_key, capsys):
    for base in ("http://127.0.0.1:1/v1/ingest", "http://127.0.0.1:1/?q=1",
                 "file:///etc/passwd", "http://user@127.0.0.1:1"):
        rc, line, _ = _probe_cli(state, base, TOKEN + "\n", capsys)
        assert (rc, line) == (2, ""), base
    rc, line, _ = _probe_cli(state, "http://127.0.0.1:1", "not a token line!\n", capsys)
    assert (rc, line) == (2, "")
    key_file = state / "verify-probe.key"
    key_file.chmod(0o640)
    rc, line, _ = _probe_cli(state, "http://127.0.0.1:1", TOKEN + "\n", capsys)
    assert (rc, line) == (2, "")


def test_classify_probe_never_echoes_a_body():
    assert device_auth.classify_probe(200, b'{"ok": true}') == "ok"
    assert device_auth.classify_probe(200, b'{"ok": true, "rows": [1]}') == "unexpected-body"
    assert device_auth.classify_probe(401, json.dumps(SECRET_REFUSED).encode()) == "secret-refused"
    assert device_auth.classify_probe(
        401, b'{"detail": {"error": "device_sig_missing"}}') == "device-auth device_sig_missing"
    assert device_auth.classify_probe(
        500, b'{"detail": {"error": "LEAK Me"}}') == "error unrecognised"
    assert device_auth.classify_probe(404, b"<html>CANARY</html>") == "unrecognised"


# ------------------------------------------------- rollback: verify-remove


@dataclasses.dataclass(frozen=True)
class _DeviceBeforeScopes:
    """Device as an engine predating #652 defines it: no ``scope`` field."""
    kid: str
    public_key: str
    enrolled_at: str
    via: str
    key_storage: str
    revoked_at: str | None = None


def _load_like_an_engine_before_scopes(path) -> int:
    data = json.loads(path.read_text())
    return len([_DeviceBeforeScopes(**raw) for raw in data["devices"]])


def test_verify_remove_leaves_the_registry_byte_identical_to_phones_only(
        state, capsys):
    """Mutation: remove_scoped revokes instead of deleting -> the scoped entry
    stays, the bytes differ and the pre-scope loader raises: red."""
    path = state / "devices.json"
    registry = device_auth.DeviceRegistry(path)
    for _ in range(2):
        registry.enrol(Phone().public_x963, via="token", key_storage="keychain",
                       allow_new=True)
    phones_only = path.read_bytes()
    key_file = state / "verify-probe.key"
    assert _cli(state, "verify-enrol", "--key-file", str(key_file)) == 0
    kid = device_auth.kid_for(device_auth._public_point(
        device_auth.load_key_file(str(key_file))))
    # The availability failure being fixed (#652 review M2), and its survival
    # of revocation: an engine before scopes cannot load either registry.
    with pytest.raises(TypeError, match="scope"):
        _load_like_an_engine_before_scopes(path)
    assert _cli(state, "revoke", "--", kid) == 0
    with pytest.raises(TypeError, match="scope"):
        _load_like_an_engine_before_scopes(path)
    capsys.readouterr()
    assert _cli(state, "verify-remove", f"--kid={kid}") == 0
    out = capsys.readouterr()
    assert f"removed {kid}" in out.out
    for line in key_file.read_text().split("-----")[2].strip().splitlines():
        assert line not in out.out + out.err
    assert path.read_bytes() == phones_only
    assert _load_like_an_engine_before_scopes(path) == 2
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_verify_remove_refuses_a_phone_and_an_unknown_kid_leaving_the_file(
        state, capsys):
    """Mutation: drop the scope-is-None refusal -> the phone's entry is deleted
    and the call exits 0: red."""
    path = state / "devices.json"
    registry = device_auth.DeviceRegistry(path)
    phone = Phone()
    registry.enrol(phone.public_x963, via="token", key_storage="keychain",
                   allow_new=True)
    registry.enrol_scoped(Phone().public_x963, scope="verify")
    before = path.read_bytes()
    capsys.readouterr()
    assert _cli(state, "verify-remove", f"--kid={phone.kid}") == 1
    err = capsys.readouterr().err
    assert "full device" in err and "nothing removed" in err
    assert path.read_bytes() == before
    with pytest.raises(device_auth.DeviceAuthError, match="device_not_scoped"):
        registry.remove_scoped(phone.kid)
    assert _cli(state, "verify-remove", f"--kid={Phone().kid}") == 1
    assert "no enrolled device" in capsys.readouterr().err
    assert path.read_bytes() == before


def _dash_leading_phone() -> Phone:
    """A key whose kid starts with '-' (about one in 64): argparse reads such
    a word as an option, which made the CLI calls above fail at random."""
    for _ in range(5000):
        phone = Phone()
        if phone.kid.startswith("-"):
            return phone
    raise AssertionError("no dash-leading kid in 5000 keys")


def test_a_dash_leading_scoped_kid_is_revoked_and_removed_through_the_cli(
        state, capsys):
    """The calling forms this file and the consumer's instances.sh use
    (``revoke -- K``, ``verify-remove --kid=K``) work for the one kid in 64
    that starts with '-'. Mutation: the separate-word forms (``revoke K``,
    ``--kid K``) -> argparse exits 2: red."""
    path = state / "devices.json"
    registry = device_auth.DeviceRegistry(path)
    scoped = _dash_leading_phone()
    registry.enrol_scoped(scoped.public_x963, scope="verify")
    assert scoped.kid.startswith("-")
    capsys.readouterr()
    assert _cli(state, "revoke", "--", scoped.kid) == 0
    assert _cli(state, "verify-remove", f"--kid={scoped.kid}") == 0
    assert f"removed {scoped.kid}" in capsys.readouterr().out
    assert scoped.kid not in path.read_text()
