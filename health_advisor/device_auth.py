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

Seams left for later items: the enrolment route's body carries a version so a
QR-token enrolment can share this registry; ``Device.via`` records which
channel enrolled a key; and nothing here depends on how D23 derives its keys.
"""
from __future__ import annotations

import argparse
import base64
import binascii
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
from dataclasses import asdict, dataclass

from . import body_aead


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


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    if not isinstance(text, str) or not re.fullmatch(r"[A-Za-z0-9_-]*", text):
        raise ValueError("not base64url")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("not base64url") from exc


def kid_for(public_key_x963: bytes) -> str:
    """The key id of an uncompressed P-256 public key."""
    return _b64url(hashlib.sha256(public_key_x963).digest()[:KID_BYTES])


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
        signature = _b64url_decode(sig)
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

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def public_key_bytes(self) -> bytes:
        return _b64url_decode(self.public_key)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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

    def _write(self, devices: dict[str, Device]) -> None:
        payload = json.dumps({
            "v": REGISTRY_VERSION,
            "devices": [asdict(device) for device in devices.values()],
        }, indent=2, sort_keys=True) + "\n"
        atomic_write_json(self.path, payload, temp_prefix=".devices.")

    def enrol(self, public_key_x963: bytes, *, via: str, key_storage: str,
              allow_new: bool) -> tuple[Device, bool]:
        """Record a key; return (device, created). Idempotent for a known key.

        A revoked key is never re-admitted. ``allow_new=False`` confirms keys
        already enrolled and refuses the rest, which is ``required`` mode.
        """
        load_public_key(public_key_x963)
        kid = kid_for(public_key_x963)
        with self._exclusive():
            devices = self._read_file()
            existing = devices.get(kid)
            if existing is not None:
                if not existing.active:
                    raise DeviceAuthError("device_revoked", 403)
                return existing, False
            if not allow_new:
                raise DeviceAuthError("device_enrol_closed", 403)
            device = Device(kid=kid, public_key=_b64url(public_key_x963),
                            enrolled_at=_utc_now(), via=via,
                            key_storage=key_storage)
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
            revoked = Device(**{**asdict(existing), "revoked_at": _utc_now()})
            devices[kid] = revoked
            self._write(devices)
            return revoked, False


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
                     body: bytes, now_ms: int | None = None) -> str | None:
        """Return the verified kid, None for an admitted unsigned request."""
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
        return device.kid


# --------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m health_advisor.device_auth",
        description="List or revoke an instance's enrolled device keys.")
    parser.add_argument("--registry", default=os.environ.get("HA_DEVICE_REGISTRY_FILE"),
                        help="the registry file (default: HA_DEVICE_REGISTRY_FILE)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print every enrolled key, one per line")
    revoke = sub.add_parser("revoke", help="revoke one key; other devices are unaffected")
    revoke.add_argument("kid")
    args = parser.parse_args(argv)
    if not args.registry:
        parser.error("no registry: pass --registry or set HA_DEVICE_REGISTRY_FILE")
    registry = DeviceRegistry(args.registry)

    if args.command == "list":
        devices = registry.devices()
        if not devices:
            print("no devices enrolled")
        for device in devices.values():
            state = f"revoked {device.revoked_at}" if device.revoked_at else "active"
            print(f"{device.kid}  {state}  enrolled {device.enrolled_at}  "
                  f"via {device.via}  key {device.key_storage}")
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
