"""Per-device request signatures: the device credential layer.

Each phone holds a P-256 key that never leaves it (a Secure Enclave key where
the hardware has one). It enrols the public half once, and from then on every
request it sends carries three headers:

    X-HA-Device-Kid   the key id, base64url(SHA-256(X9.63 public key)[:16])
    X-HA-Device-Ts    milliseconds since the Unix epoch, decimal
    X-HA-Device-Sig   base64url of the raw 64-byte ECDSA-P256-SHA256 signature
                      (r || s, big-endian) over ``signing_input(...)``

The signed string is six lines joined by ``\\n`` with no trailing newline::

    ha-device-sig-v1
    <kid>
    <ts>
    <METHOD>
    <target>            raw path, plus "?" and the raw query when there is one
    <sha256-hex>        of the request body exactly as it crossed the wire

"Exactly as it crossed the wire" means the D23 envelope when the body is
sealed, so a signature can be checked before anything is decrypted, and an
empty body hashes as the empty string. No field can contain a newline: the kid
is base64url, the timestamp is digits, the method is an HTTP token, a request
target has no line breaks (the verifier refuses one that does) and the hash is
hex. The form is therefore unambiguous without length prefixes.

The timestamp window is D23's own skew window (``body_aead.SKEW_SECONDS``, 600
s either side). A request that one layer admits on clock grounds the other
admits too, so a phone with a drifting clock sees one diagnosis rather than
two. The window bounds replay; it does not prevent replay inside it, which is
the job of a replay cache (a separate item), exactly as for D23 today.

``HA_DEVICE_AUTH_MODE`` selects what the receiver does with all this:

    off       (the default, and what unset or empty means) nothing: no header
              is read, the enrolment route does not exist, and every response
              is byte-identical to a receiver that predates this module.
    accept    a request that carries the headers must verify; one that carries
              none is admitted on the shared-secret token as before. New keys
              may enrol through the upgrade route.
    required  every request except ``/health`` and the enrolment route must
              carry a valid signature from an enrolled, unrevoked key. The
              upgrade route re-confirms keys already enrolled and refuses new
              ones, so a holder of the shared secret alone can neither
              authenticate nor add a device.

Anything else refuses at startup. In ``accept`` a revoked key is refused only
when it signs; the shared secret still opens the door without a signature --
true only for a phone on the instance secret. A per-device secret (T5 piece
9, consumer #426, ``device_secrets.py``) is selected by the DERIVED auth
token after a signature has already verified, never by the token alone, so
once that signature is refused for a revoked key its own per-device secret
is simply never chosen; the "signature only" carve-out above does not extend
to it.
Revocation is effective in ``required``, which is the point of that mode.

The registry is a small JSON file per instance (``HA_DEVICE_REGISTRY_FILE``),
not a vault table. It holds public keys only, and it must stay writable and
durable while the vault is checked out: a revocation written into a plaintext
checkout that has not been checked back in would be lost with the process.
Writes take an exclusive ``flock`` on a sidecar lock file and replace the file
atomically, so the receiver (enrolling) and an operator (revoking) cannot lose
each other's change. The receiver re-reads the file when its stat signature
changes, so a revocation takes effect on the next request without a restart.

Verification scope (#652). A registry entry may carry ``"scope": "verify"``.
Such a device is refused with ``device_scope_forbidden`` (403) on every route
except ``GET /v1/verify/probe`` (``VERIFY_PROBE_PATH``), and only after its
signature has verified, so the scope says nothing to a caller who cannot sign.
The probe route exists only while device authentication is on; it reads no
vault and answers the fixed body ``{"ok": true}`` once the shared secret has
been checked exactly as every other route checks it. An operator can then
sign one probe with an instance's own secret (200) and one with another
instance's secret (401, the secret refusal body) and so measure secret
isolation under ``required`` without a phone. A scoped entry is created only
by the host-side ``verify-enrol`` command below, which can create nothing
else; neither enrolment route will confirm, re-enrol, upgrade or attach a
per-device secret to a scoped key, and ``verify-enrol`` will not touch an
entry that is already a phone's. An entry with no scope is a full device,
exactly as before, and is written without a ``scope`` key, so a registry with
no scoped entry is byte-identical to one written before this existed. An
engine that predates scopes refuses to load a registry holding one
(``Device(**raw)`` is strict): it fails closed instead of treating a verify
key as a phone. Revocation (``revoke``) and the timestamp window apply to a
scoped key exactly as to a phone's. Because ``revoke`` keeps the entry, rolling
an instance back past scopes first needs ``verify-remove --kid K``, which
deletes a scoped entry outright (and refuses a phone's), leaving the registry
byte-identical to one that never held it.

Seams left for later items: the enrolment route's body carries a version so a
QR-token enrolment can share this registry; ``Device.via`` records which
channel enrolled a key; and nothing here depends on how D23 derives its keys.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

from . import body_aead
from . import dates
from .wire import b64url_decode, b64url_encode


MODES = ("off", "accept", "required")
SIG_VERSION = b"ha-device-sig-v1"
HEADER_KID = "x-ha-device-kid"
HEADER_TS = "x-ha-device-ts"
HEADER_SIG = "x-ha-device-sig"
SKEW_SECONDS = body_aead.SKEW_SECONDS
KID_BYTES = 16
PUBLIC_KEY_BYTES = 65  # uncompressed X9.63: 0x04 || X || Y
SIGNATURE_BYTES = 64  # raw r || s
ENROL_UPGRADE_PATH = "/v1/enrol/upgrade"
VERIFY_PROBE_PATH = "/v1/verify/probe"
SCOPE_VERIFY = "verify"
SCOPES = (SCOPE_VERIFY,)
# What each scope may call, as exact (METHOD, decoded path) pairs. A scope
# missing from this table may call nothing.
SCOPE_ROUTES = {
    SCOPE_VERIFY: frozenset({("GET", VERIFY_PROBE_PATH)}),
}
VERIFY_PROBE_BODY = {"ok": True}
VERIFY_KEY_STORAGE = "host_file"
VERIFY_VIA = "host"
REGISTRY_VERSION = 1
KEY_STORAGE_VALUES = ("secure_enclave", "keychain")

_KID_RE = re.compile(r"^[A-Za-z0-9_-]{22}$")
_TS_RE = re.compile(r"^[0-9]{1,16}$")
_METHOD_RE = re.compile(r"^[A-Z]{1,16}$")


class DeviceAuthError(Exception):
    """A refusal, carrying the stable wire code and the HTTP status."""

    def __init__(self, code: str, status: int = 401):
        self.code = code
        self.status = status
        super().__init__(code)


def device_auth_mode() -> str:
    """Return the configured mode, refusing an invalid value at startup."""
    mode = os.environ.get("HA_DEVICE_AUTH_MODE", "")
    if mode == "":
        return "off"
    if mode in MODES:
        return mode
    raise RuntimeError(
        "device authentication refuses to start: HA_DEVICE_AUTH_MODE is set to "
        f"invalid value {mode!r}; set it to 'off', 'accept' or 'required', or "
        "leave it unset for 'off'."
    )


def kid_for(public_key_x963: bytes) -> str:
    """The key id of an uncompressed P-256 public key."""
    return b64url_encode(hashlib.sha256(public_key_x963).digest()[:KID_BYTES])


def body_sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def signing_input(kid: str, timestamp_ms: int | str, method: str,
                  target: bytes | str, body_sha256_hex: str) -> bytes:
    """The exact bytes a device signs. See the module docstring."""
    if isinstance(target, str):
        target = target.encode("utf-8")
    if b"\n" in target or b"\r" in target:
        raise ValueError("a request target cannot contain a line break")
    return b"\n".join((
        SIG_VERSION,
        kid.encode("ascii"),
        str(int(timestamp_ms)).encode("ascii"),
        method.upper().encode("ascii"),
        target,
        body_sha256_hex.encode("ascii"),
    ))


def load_public_key(public_key_x963: bytes):
    """Parse an uncompressed P-256 point, refusing anything else."""
    from cryptography.hazmat.primitives.asymmetric import ec

    if len(public_key_x963) != PUBLIC_KEY_BYTES or public_key_x963[0] != 0x04:
        raise ValueError("expected an uncompressed P-256 public key")
    return ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(), public_key_x963)


def verify_signature(public_key_x963: bytes, message: bytes, signature: bytes) -> bool:
    """True when ``signature`` (raw r || s) is valid for ``message``."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        encode_dss_signature,
    )

    if len(signature) != SIGNATURE_BYTES:
        return False
    r = int.from_bytes(signature[:32], "big")
    s = int.from_bytes(signature[32:], "big")
    if r == 0 or s == 0:
        return False
    try:
        load_public_key(public_key_x963).verify(
            encode_dss_signature(r, s), message, ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError):
        return False
    return True


@dataclass(frozen=True)
class SignedHeaders:
    kid: str
    timestamp_ms: int
    signature: bytes


def parse_headers(get_header) -> SignedHeaders | None:
    """Read the three headers through ``get_header(name) -> str | None``.

    None means the request carries none of them. Some but not all, or any
    malformed value, is a refusal rather than "unsigned": a request that tried
    to sign and failed must not quietly fall back to the shared secret.
    """
    values = [get_header(name) for name in (HEADER_KID, HEADER_TS, HEADER_SIG)]
    if all(value is None for value in values):
        return None
    kid, ts, sig = values
    if kid is None or ts is None or sig is None:
        raise DeviceAuthError("device_sig_malformed")
    if not _KID_RE.fullmatch(kid) or not _TS_RE.fullmatch(ts):
        raise DeviceAuthError("device_sig_malformed")
    try:
        signature = b64url_decode(sig)
    except ValueError as exc:
        raise DeviceAuthError("device_sig_malformed") from exc
    if len(signature) != SIGNATURE_BYTES:
        raise DeviceAuthError("device_sig_malformed")
    return SignedHeaders(kid=kid, timestamp_ms=int(ts), signature=signature)


def check_signature(public_key_x963: bytes, signed: SignedHeaders, *,
                    method: str, target: bytes, body_hash: str,
                    now_ms: int | None = None) -> None:
    """Raise unless ``signed`` is a fresh, valid signature over this request.

    ``body_hash`` is ``body_sha256`` of the body as it crossed the wire.
    """
    if now_ms is None:
        now_ms = time.time_ns() // 1_000_000
    if abs(now_ms - signed.timestamp_ms) > SKEW_SECONDS * 1000:
        raise DeviceAuthError("device_sig_stale")
    if not _METHOD_RE.fullmatch(method.upper()):
        raise DeviceAuthError("device_sig_malformed")
    try:
        message = signing_input(signed.kid, signed.timestamp_ms, method,
                                target, body_hash)
    except ValueError as exc:
        raise DeviceAuthError("device_sig_malformed") from exc
    if not verify_signature(public_key_x963, message, signed.signature):
        raise DeviceAuthError("device_sig_bad")


# ---------------------------------------------------------------- registry


@dataclass(frozen=True)
class Device:
    kid: str
    public_key: str  # base64url of the X9.63 point
    enrolled_at: str
    via: str
    key_storage: str
    revoked_at: str | None = None
    # None is a full device (a phone). Anything else is one of SCOPES and
    # limits the device to SCOPE_ROUTES[scope].
    scope: str | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def permits(self, method: str, path: str | None) -> bool:
        """True when this device may call ``method path`` at all."""
        if self.scope is None:
            return True
        return (method.upper(), path) in SCOPE_ROUTES.get(self.scope, frozenset())

    def public_key_bytes(self) -> bytes:
        return b64url_decode(self.public_key)


@contextlib.contextmanager
def exclusive_lock(path: str | os.PathLike):
    """An exclusive ``flock`` on ``path``'s sidecar lock file.

    Shared with ``enrol.TokenStore`` (T5, #426): both files live beside a
    user's vault and need the same reader/writer discipline, so the locking
    primitive is factored here rather than duplicated. flock is scoped to the
    open file description, so two callers in the same process each get their
    own fd and correctly serialise against each other, not just across
    processes.
    """
    lock_path = os.fspath(path) + ".lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def atomic_write_json(path: str | os.PathLike, payload: str, *,
                       temp_prefix: str = ".tmp.") -> None:
    """Write ``payload`` to ``path`` atomically, mode 600, fsynced through the directory.

    Shared with ``enrol.TokenStore``: write-then-``os.replace`` under the
    caller's own ``exclusive_lock`` is the pattern both registries need, so it
    lives here once rather than twice. Caller is responsible for holding the
    lock; this function only does the write.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, temp_path = tempfile.mkstemp(prefix=temp_prefix, dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_path)
        raise
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class DeviceRegistry:
    """The enrolled devices of one instance, in one JSON file."""

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._devices: dict[str, Device] = {}

    # -- reading

    def _stat_signature(self) -> tuple | None:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    @staticmethod
    def _parse(text: str) -> dict[str, Device]:
        data = json.loads(text)
        if not isinstance(data, dict) or data.get("v") != REGISTRY_VERSION:
            raise ValueError("unrecognised device registry version")
        devices: dict[str, Device] = {}
        for raw in data.get("devices", []):
            device = Device(**raw)
            if kid_for(device.public_key_bytes()) != device.kid:
                raise ValueError(f"device {device.kid!r} does not match its key")
            if device.scope is not None and device.scope not in SCOPES:
                raise ValueError(f"device {device.kid!r} has an unknown scope")
            devices[device.kid] = device
        return devices

    def _read_file(self) -> dict[str, Device]:
        try:
            with open(self.path, encoding="utf-8") as handle:
                return self._parse(handle.read())
        except FileNotFoundError:
            return {}

    def devices(self) -> dict[str, Device]:
        """The current registry, re-read only when the file has changed.

        An unreadable file raises: for an authentication table, failing
        closed is the only safe reading of "I cannot tell who is enrolled".
        """
        with self._lock:
            signature = self._stat_signature()
            if signature != self._signature or signature is None:
                self._devices = self._read_file()
                self._signature = signature
            return dict(self._devices)

    def get(self, kid: str) -> Device | None:
        return self.devices().get(kid)

    # -- writing

    def _exclusive(self):
        return exclusive_lock(self.path)

    @staticmethod
    def _record(device: Device) -> dict:
        # A full device is written without a scope key, so a registry with no
        # scoped entry stays byte-identical to one that predates scopes.
        record = asdict(device)
        if record["scope"] is None:
            del record["scope"]
        return record

    def _write(self, devices: dict[str, Device]) -> None:
        payload = json.dumps({
            "v": REGISTRY_VERSION,
            "devices": [self._record(device) for device in devices.values()],
        }, indent=2, sort_keys=True) + "\n"
        atomic_write_json(self.path, payload, temp_prefix=".devices.")

    def enrol(self, public_key_x963: bytes, *, via: str, key_storage: str,
              allow_new: bool) -> tuple[Device, bool]:
        """Record a key; return (device, created). Idempotent for a known key.

        A revoked key is never re-admitted. ``allow_new=False`` confirms keys
        already enrolled and refuses the rest, which is ``required`` mode. A
        scoped key is never confirmed here: a phone's enrolment channel must
        not be able to vouch for, or hand a secret to, a verification key.
        """
        load_public_key(public_key_x963)
        kid = kid_for(public_key_x963)
        with self._exclusive():
            devices = self._read_file()
            existing = devices.get(kid)
            if existing is not None:
                if not existing.active:
                    raise DeviceAuthError("device_revoked", 403)
                if existing.scope is not None:
                    raise DeviceAuthError("device_scope_forbidden", 403)
                return existing, False
            if not allow_new:
                raise DeviceAuthError("device_enrol_closed", 403)
            device = Device(kid=kid, public_key=b64url_encode(public_key_x963),
                            enrolled_at=dates.utc_now(), via=via,
                            key_storage=key_storage)
            devices[kid] = device
            self._write(devices)
            return device, True

    def enrol_scoped(self, public_key_x963: bytes, *,
                     scope: str) -> tuple[Device, bool]:
        """Record a SCOPED key; return (device, created). Host-side only.

        The only writer of a scope. It can create a scoped entry and confirm
        the same scoped entry again; it refuses a revoked key, and it refuses
        a key already enrolled under another scope or none (a phone), so it
        can neither mint a full device nor change what an existing one is.
        """
        if scope not in SCOPES:
            raise ValueError(f"unknown device scope {scope!r}")
        load_public_key(public_key_x963)
        kid = kid_for(public_key_x963)
        with self._exclusive():
            devices = self._read_file()
            existing = devices.get(kid)
            if existing is not None:
                if not existing.active:
                    raise DeviceAuthError("device_revoked", 403)
                if existing.scope != scope:
                    raise DeviceAuthError("device_scope_conflict", 409)
                return existing, False
            device = Device(kid=kid, public_key=b64url_encode(public_key_x963),
                            enrolled_at=dates.utc_now(), via=VERIFY_VIA,
                            key_storage=VERIFY_KEY_STORAGE, scope=scope)
            devices[kid] = device
            self._write(devices)
            return device, True

    def revoke(self, kid: str) -> tuple[Device, bool]:
        """Revoke one key; return (device, already_revoked). Others untouched."""
        with self._exclusive():
            devices = self._read_file()
            existing = devices.get(kid)
            if existing is None:
                raise KeyError(kid)
            if not existing.active:
                return existing, True
            revoked = Device(**{**asdict(existing), "revoked_at": dates.utc_now()})
            devices[kid] = revoked
            self._write(devices)
            return revoked, False

    def remove_scoped(self, kid: str) -> Device:
        """Delete one SCOPED entry outright; return it. Host-side only.

        The rollback path for #652: an engine that predates scopes refuses to
        load a registry holding any scoped entry, and ``revoke`` keeps the
        entry, so a revoked verify key would still take every phone offline
        after a rollback. Deleting it leaves the registry exactly as if the
        key had never been enrolled. A full device (a phone) is never deleted
        here -- phones are revoked, so their refusal stays on record -- and an
        unknown kid is an error, not a silent no-op.
        """
        with self._exclusive():
            devices = self._read_file()
            existing = devices.get(kid)
            if existing is None:
                raise KeyError(kid)
            if existing.scope is None:
                raise DeviceAuthError("device_not_scoped", 409)
            del devices[kid]
            self._write(devices)
            return existing



def registry_from_env() -> DeviceRegistry:
    """The registry named by HA_DEVICE_REGISTRY_FILE, checked at startup."""
    path = os.environ.get("HA_DEVICE_REGISTRY_FILE", "")
    if not path:
        raise RuntimeError(
            "device authentication refuses to start: HA_DEVICE_AUTH_MODE is "
            "on but HA_DEVICE_REGISTRY_FILE is unset; point it at a writable, "
            "persistent path for this instance's enrolled devices.")
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory) or not os.access(directory, os.W_OK):
        raise RuntimeError(
            "device authentication refuses to start: the directory of "
            f"HA_DEVICE_REGISTRY_FILE ({directory}) is missing or not writable.")
    registry = DeviceRegistry(path)
    try:
        registry.devices()
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "device authentication refuses to start: HA_DEVICE_REGISTRY_FILE "
            f"is unreadable ({exc}).") from exc
    return registry


class DeviceAuth:
    """What the ASGI layer needs: a mode and a registry."""

    def __init__(self, mode: str, registry: DeviceRegistry):
        if mode not in ("accept", "required"):
            raise ValueError(f"DeviceAuth needs 'accept' or 'required', not {mode!r}")
        self.mode = mode
        self.registry = registry

    def authenticate(self, get_header, *, method: str, target: bytes,
                     body: bytes, now_ms: int | None = None,
                     path: str | None = None) -> str | None:
        """Return the verified kid, None for an admitted unsigned request.

        ``path`` is the decoded route path the request will be dispatched
        on. A scoped device is refused unless ``(method, path)`` is in its
        scope's table, and a scoped device with no path is refused outright.
        """
        signed = parse_headers(get_header)
        if signed is None:
            if self.mode == "required":
                raise DeviceAuthError("device_sig_missing")
            return None
        try:
            device = self.registry.get(signed.kid)
        except (ValueError, TypeError, OSError) as exc:
            raise DeviceAuthError("device_registry_unreadable", 503) from exc
        if device is None:
            raise DeviceAuthError("device_unknown_kid")
        if not device.active:
            raise DeviceAuthError("device_revoked")
        check_signature(device.public_key_bytes(), signed, method=method,
                        target=target, body_hash=body_sha256(body),
                        now_ms=now_ms)
        if not device.permits(method, path):
            raise DeviceAuthError("device_scope_forbidden", 403)
        return device.kid


# --------------------------------------------------------------------- CLI
#
# verify-enrol and verify-probe are the host half of the verification scope
# (#652). The private key is a PKCS#8 PEM file, created O_EXCL|O_NOFOLLOW at
# mode 600 and refused at any looser mode. Nothing here ever prints it, the
# shared secret, the auth token or a response body: verify-probe prints one
# line, "<status> <class>", where class is ok, unexpected-body,
# secret-refused, device-auth <code>, error <code>, unreachable or
# unrecognised, and only a bare lowercase code is ever echoed.

PROBE_MAX_RESPONSE_BYTES = 64 * 1024
_TOKEN_LINE_RE = re.compile(r'^header = "X-Health-Secret: ([A-Za-z0-9_\-]{1,256})"$')
_BARE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]{1,256}$")
_CODE_RE = re.compile(r"^[a-z0-9_]{1,48}$")


def _write_new_key_file(path: str) -> bytes:
    """Create a fresh P-256 key at ``path`` (which must not exist); return its public point."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM,
                            serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, pem)
        os.fsync(fd)
    finally:
        os.close(fd)
    return key.public_key().public_bytes(serialization.Encoding.X962,
                                         serialization.PublicFormat.UncompressedPoint)


def load_key_file(path: str):
    """Load a verification private key, refusing a link or any group/other access."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if os.fstat(fd).st_mode & 0o077:
            raise PermissionError("the key file must be mode 600")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            data = handle.read()
    finally:
        if fd >= 0:
            os.close(fd)
    key = serialization.load_pem_private_key(data, password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey) or key.curve.name != "secp256r1":
        raise ValueError("the key file does not hold a P-256 private key")
    return key


def _public_point(key) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(serialization.Encoding.X962,
                                         serialization.PublicFormat.UncompressedPoint)


def sign_request(key, method: str, target: bytes, body: bytes = b"", *,
                 now_ms: int | None = None) -> dict[str, str]:
    """The three device headers for one request, signed with ``key``."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        decode_dss_signature,
    )

    kid = kid_for(_public_point(key))
    ts = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    message = signing_input(kid, ts, method, target, body_sha256(body))
    r, s = decode_dss_signature(key.sign(message, ec.ECDSA(hashes.SHA256())))
    return {"X-HA-Device-Kid": kid, "X-HA-Device-Ts": str(ts),
            "X-HA-Device-Sig": b64url_encode(r.to_bytes(32, "big") + s.to_bytes(32, "big"))}


def _verify_enrol(registry: DeviceRegistry, key_file: str) -> int:
    key_file = os.path.abspath(key_file)
    created_file = False
    try:
        if os.path.lexists(key_file):
            public = _public_point(load_key_file(key_file))
        else:
            public = _write_new_key_file(key_file)
            created_file = True
    except (OSError, ValueError, TypeError) as exc:
        print(f"refusing: the key file cannot be used ({type(exc).__name__}: "
              f"{getattr(exc, 'strerror', None) or exc})", file=sys.stderr)
        return 1
    try:
        device, created = registry.enrol_scoped(public, scope=SCOPE_VERIFY)
    except (DeviceAuthError, OSError, ValueError, TypeError) as exc:
        if created_file:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(key_file)
        code = exc.code if isinstance(exc, DeviceAuthError) else type(exc).__name__
        print(f"refusing: the registry would not take this key ({code}); "
              "nothing was enrolled", file=sys.stderr)
        return 1
    state = "enrolled" if created else "already enrolled"
    print(f"{state}: verify-scoped device {device.kid} (may call only "
          f"GET {VERIFY_PROBE_PATH}); private key kept in {key_file}, mode 600, "
          "never printed")
    return 0


def _verify_remove(registry: DeviceRegistry, kid: str) -> int:
    try:
        device = registry.remove_scoped(kid)
    except KeyError:
        print(f"refusing: no enrolled device has kid {kid!r}; nothing removed",
              file=sys.stderr)
        return 1
    except DeviceAuthError as exc:
        print(f"refusing: {kid} is a full device, not a verify-scoped key "
              f"({exc.code}); revoke a phone instead; nothing removed",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"refusing: the registry cannot be read or written "
              f"({type(exc).__name__}); nothing removed", file=sys.stderr)
        return 1
    print(f"removed {device.kid} (scope {device.scope}) from the registry; "
          "other devices unaffected")
    return 0


def _read_token(stream) -> str | None:
    line = stream.readline(1024).strip()
    found = _TOKEN_LINE_RE.fullmatch(line)
    if found:
        return found.group(1)
    if _BARE_TOKEN_RE.fullmatch(line):
        return line
    return None


def classify_probe(status: int, body: bytes) -> str:
    """Reduce one probe response to its class; never echoes the body."""
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        parsed = None
    if status == 200:
        return "ok" if parsed == VERIFY_PROBE_BODY else "unexpected-body"
    detail = parsed.get("detail") if isinstance(parsed, dict) else None
    if isinstance(detail, dict) and isinstance(detail.get("error"), str):
        code = detail["error"] if _CODE_RE.fullmatch(detail["error"]) else "unrecognised"
        return ("device-auth " if code.startswith("device_") else "error ") + code
    if detail == "missing or bad shared secret":
        return "secret-refused"
    return "unrecognised"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # never follow: report the 3xx
        return None


def _probe_once(key, base_url: str, token: str, open_secret) -> str:
    parts = urllib.parse.urlsplit(base_url)
    if (parts.scheme not in ("http", "https") or not parts.netloc
            or parts.path not in ("", "/") or parts.query or parts.fragment
            or "@" in parts.netloc):
        raise ValueError("--base-url must be scheme://host:port and nothing else")
    target = VERIFY_PROBE_PATH.encode("ascii")
    headers = sign_request(key, "GET", target)
    headers["X-Health-Secret"] = token
    request = urllib.request.Request(
        f"{parts.scheme}://{parts.netloc}{VERIFY_PROBE_PATH}",
        method="GET", headers=headers)
    # No proxy: a signed request and a token go to the named host only.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    try:
        with opener.open(request, timeout=10) as response:
            status = response.status
            content_type = response.headers.get("content-type", "")
            body = response.read(PROBE_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        status = exc.code
        content_type = exc.headers.get("content-type", "") if exc.headers else ""
        body = exc.read(PROBE_MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, OSError):
        return "000 unreachable"
    if len(body) > PROBE_MAX_RESPONSE_BYTES:
        return f"{status} unrecognised"
    if content_type.split(";", 1)[0].strip().lower() == body_aead.CONTENT_TYPE:
        secret = open_secret()
        if not secret:
            return f"{status} unrecognised"
        try:
            body = body_aead.open_(secret, "response", "GET", target, body,
                                   require_version=True)
        except body_aead.D23Error:
            return f"{status} unrecognised"
    return f"{status} {classify_probe(status, body)}"


def _instance_secret_for_open() -> str:
    # The receiver's own loader, so a sealed response is opened with exactly
    # the secret this instance seals with. Imported here: the CLI's other
    # commands must not need the receiver's dependencies.
    from . import receiver

    return receiver._instance_secret()


def _verify_probe(key_file: str, base_url: str, stdin) -> int:
    token = _read_token(stdin)
    if token is None:
        print("refusing: stdin must carry one X-Health-Secret token line "
              "(secret_header.py output, or the bare token)", file=sys.stderr)
        return 2
    try:
        key = load_key_file(key_file)
    except (OSError, ValueError, TypeError) as exc:
        print(f"refusing: the key file cannot be used ({type(exc).__name__})",
              file=sys.stderr)
        return 2
    try:
        print(_probe_once(key, base_url, token, _instance_secret_for_open))
    except ValueError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m health_advisor.device_auth",
        description="List or revoke an instance's enrolled device keys, or "
                    "enrol and use a verify-scoped probe key (#652).")
    parser.add_argument("--registry", default=os.environ.get("HA_DEVICE_REGISTRY_FILE"),
                        help="the registry file (default: HA_DEVICE_REGISTRY_FILE)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print every enrolled key, one per line")
    revoke = sub.add_parser("revoke", help="revoke one key; other devices are unaffected")
    revoke.add_argument("kid")
    venrol = sub.add_parser(
        "verify-enrol",
        help=f"create (or reuse) a host key file, mode 600, and enrol it as a "
             f"verify-scoped device that may call only GET {VERIFY_PROBE_PATH}; "
             "never prints the key")
    venrol.add_argument("--key-file", required=True)
    vprobe = sub.add_parser(
        "verify-probe",
        help=f"sign one GET {VERIFY_PROBE_PATH} with a verify key; the "
             "X-Health-Secret token is read from stdin; prints '<status> <class>'")
    vprobe.add_argument("--key-file", required=True)
    vprobe.add_argument("--base-url", required=True)
    vremove = sub.add_parser(
        "verify-remove",
        help="delete one verify-scoped entry from the registry outright (run "
             "before rolling an instance back to an engine older than #652, "
             "which cannot load a scoped entry); refuses a phone's key")
    vremove.add_argument("--kid", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify-probe":
        return _verify_probe(args.key_file, args.base_url, sys.stdin)
    if not args.registry:
        parser.error("no registry: pass --registry or set HA_DEVICE_REGISTRY_FILE")
    registry = DeviceRegistry(args.registry)

    if args.command == "verify-enrol":
        return _verify_enrol(registry, args.key_file)

    if args.command == "verify-remove":
        return _verify_remove(registry, args.kid)

    if args.command == "list":
        devices = registry.devices()
        if not devices:
            print("no devices enrolled")
        for device in devices.values():
            state = f"revoked {device.revoked_at}" if device.revoked_at else "active"
            scope = f"  scope {device.scope}" if device.scope is not None else ""
            print(f"{device.kid}  {state}  enrolled {device.enrolled_at}  "
                  f"via {device.via}  key {device.key_storage}{scope}")
        return 0

    try:
        device, already = registry.revoke(args.kid)
    except KeyError:
        print(f"no enrolled device has kid {args.kid!r}; nothing revoked",
              file=sys.stderr)
        return 1
    if already:
        print(f"{device.kid} was already revoked at {device.revoked_at}")
    else:
        print(f"revoked {device.kid} at {device.revoked_at}; other devices unaffected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
