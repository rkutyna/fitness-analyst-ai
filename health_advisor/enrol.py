"""Single-use enrolment tokens for QR pairing (T5, consumer #426).

An operator mints a token for one phone to scan: a 256-bit random secret and
a fresh, token-scoped X25519 keypair, TTL-capped at ``MAX_TTL_SECONDS``. The
phone reaches ``/v1/enrol`` (a separate module; this one holds no route) with
a body sealed under HPKE base mode to the token's public key, carrying the
token itself, the phone's device public key, and a symmetric ``response_key``
the phone generated so the server can seal its reply back without needing any
key of its own.

Why the token round-trips inside the sealed body rather than staying only in
a header: the header-visible id is just a lookup key. Binding the secret
itself into the HPKE-sealed, signed plaintext means the server never needs to
keep the raw token — only ``sha256(token)`` — while still proving the caller
who can decrypt the body is the same caller who was handed the QR.

The store is one JSON file (``HA_ENROL_TOKEN_FILE``), record shape::

    {id, token_sha256, x25519_private|null, created_at, expires_at,
     state: live|used|cancelled|expired, kid|null, used_at|null}

``x25519_private`` is erased (set to ``null``) the moment a token stops being
live — at burn, at cancel, and lazily at the first touch past its expiry —
so a captured token file never yields a live decryption key for a dead token.
Only the token's hash is ever written; the plaintext token exists in memory
only for the length of ``mint()``'s return value and the caller's own use of
it (printed once by the CLI).

Locking and atomic replace are ``device_auth.exclusive_lock`` /
``device_auth.atomic_write_json`` — the same primitives ``DeviceRegistry``
uses for ``devices.json``, factored out there rather than reimplemented here
(both files sit beside one vault and need the same reader/writer discipline).
"""
from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import hashlib
import json
import os
import re
import secrets
import sys
import time
from dataclasses import asdict, dataclass, replace

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hpke, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import device_auth


STORE_VERSION = 1
TOKEN_BYTES = 32          # 256-bit single-use token
RESPONSE_KEY_BYTES = 32   # AES-256-GCM key the phone supplies for the reply
MAX_TTL_SECONDS = 900     # the engine's hard cap; a caller cannot ask for more
ENROL_TOKEN_TTL_SECONDS = 900
NONCE_BYTES = 12
STATES = ("live", "used", "cancelled", "expired")
HPKE_INFO_PREFIX = b"ha-enrol-v1\n"
ENROL_PATH = "/v1/enrol"
HEADER_ENROL_ID = "x-ha-enrol-id"
ENROL_CONTENT_TYPE = "application/x-ha-enrol-response"
ENROL_MAX_BODY_BYTES = 4096

_HPKE_SUITE = hpke.Suite(hpke.KEM.X25519, hpke.KDF.HKDF_SHA256,
                         hpke.AEAD.CHACHA20_POLY1305)


class EnrolError(Exception):
    """A refusal, carrying the stable wire code and the HTTP status."""

    def __init__(self, code: str, status: int = 401):
        self.code = code
        self.status = status
        super().__init__(code)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    if not isinstance(text, str) or not re.fullmatch(r"[A-Za-z0-9_-]*", text):
        raise ValueError("not base64url")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("not base64url") from exc


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def token_hash(token_text: str) -> str:
    """The hash stored for a token, keyed on its wire form (base64url text) --
    the same bytes the phone actually sends inside the sealed payload, not
    the raw entropy underneath it. ``mint()`` and the route's compare must
    agree on which one they hash; this is the one function both call."""
    return hashlib.sha256(token_text.encode("utf-8")).hexdigest()


def info_for(enrol_id: str) -> bytes:
    """The HPKE ``info`` bound to one token: the id, so a body sealed for one
    token cannot silently be replayed against another's key."""
    return HPKE_INFO_PREFIX + enrol_id.encode("ascii")


# ------------------------------------------------------------------ HPKE


def open_sealed(private_key_b64: str, info: bytes, wire: bytes) -> bytes:
    """Open an HPKE base-mode wire (``enc(32) || ct``) sealed to this token's key.

    Raises ``EnrolError("enrol_undecryptable", 400)`` for any failure — a bad
    key, truncated wire, wrong info, or an authentication failure all look
    the same to a caller, and none of them burn the token.
    """
    try:
        sk = X25519PrivateKey.from_private_bytes(_b64url_decode(private_key_b64))
    except (ValueError, TypeError) as exc:
        raise EnrolError("enrol_undecryptable", 400) from exc
    try:
        return _HPKE_SUITE.decrypt(wire, sk, info=info)
    except (InvalidTag, ValueError, TypeError) as exc:
        raise EnrolError("enrol_undecryptable", 400) from exc


def seal_to(public_key_b64: str, info: bytes, plaintext: bytes) -> bytes:
    """The sender side of ``open_sealed`` — used by the operator's ``pair``
    tooling (and by tests standing in for the phone), never by the server."""
    pk = X25519PublicKey.from_public_bytes(_b64url_decode(public_key_b64))
    return _HPKE_SUITE.encrypt(plaintext, pk, info=info)


def seal_response(response_key: bytes, plaintext: bytes) -> bytes:
    """AES-256-GCM seal of the enrolment response, under the phone's own key.

    HPKE here is one-directional (base mode, no PSK, no bidirectional
    context), so the reply is sealed under a symmetric key the phone
    generated and sent inside its own HPKE-sealed request — the server never
    holds a long-term key of its own for this. Wire = ``nonce(12) || ct``.
    """
    if len(response_key) != RESPONSE_KEY_BYTES:
        raise ValueError(f"response_key must be {RESPONSE_KEY_BYTES} bytes")
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(response_key).encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def open_response(response_key: bytes, wire: bytes) -> bytes:
    """The phone's side of ``seal_response`` — kept here for tests; the real
    reader is the iOS client (T5 piece 6)."""
    if len(wire) < NONCE_BYTES:
        raise ValueError("truncated response")
    nonce, ciphertext = wire[:NONCE_BYTES], wire[NONCE_BYTES:]
    return AESGCM(response_key).decrypt(nonce, ciphertext, None)


# ------------------------------------------------------------------ store


@dataclass(frozen=True)
class EnrolToken:
    id: str
    token_sha256: str
    x25519_private: str | None  # base64url of the raw 32-byte scalar; erased when not live
    created_at: str
    expires_at: float           # epoch seconds
    state: str
    kid: str | None = None
    used_at: str | None = None

    @property
    def live(self) -> bool:
        return self.state == "live"


class TokenStore:
    """The enrolment tokens of one instance, in one JSON file.

    Mirrors ``device_auth.DeviceRegistry`` in shape (stat-signature caching,
    ``exclusive_lock``/``atomic_write_json`` for writes) but is a distinct
    file and a distinct record type: the kid-to-token link belongs here, not
    in ``devices.json``, so an older engine reading the device registry never
    sees a field it doesn't recognise (``Device(**raw)`` is strict).
    """

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._signature: tuple | None = None
        self._tokens: dict[str, EnrolToken] = {}
        # Test-only hook: when set, burn() sleeps for this many seconds after
        # reading the file and before checking state, so a concurrency test
        # can force two threads to overlap inside what should be one
        # exclusive critical section. Production code never sets this.
        self._test_race_delay: float = 0.0

    # -- reading

    def _stat_signature(self) -> tuple | None:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    @staticmethod
    def _parse(text: str) -> dict[str, EnrolToken]:
        data = json.loads(text)
        if not isinstance(data, dict) or data.get("v") != STORE_VERSION:
            raise ValueError("unrecognised enrolment token store version")
        tokens: dict[str, EnrolToken] = {}
        for raw in data.get("tokens", []):
            record = EnrolToken(**raw)
            if record.state not in STATES:
                raise ValueError(f"token {record.id!r} has an unknown state {record.state!r}")
            tokens[record.id] = record
        return tokens

    def _read_file(self) -> dict[str, EnrolToken]:
        try:
            with open(self.path, encoding="utf-8") as handle:
                return self._parse(handle.read())
        except FileNotFoundError:
            return {}

    def tokens(self) -> dict[str, EnrolToken]:
        """The current store, re-read only when the file has changed.

        An unreadable or corrupt file raises — failing closed, as
        ``DeviceRegistry.devices()`` does, is the only safe reading of "I
        cannot tell what tokens exist."
        """
        signature = self._stat_signature()
        if signature != self._signature or signature is None:
            self._tokens = self._read_file()
            self._signature = signature
        return dict(self._tokens)

    def get(self, id: str) -> EnrolToken | None:
        return self.tokens().get(id)

    # -- writing

    def _exclusive(self):
        return device_auth.exclusive_lock(self.path)

    def _write(self, tokens: dict[str, EnrolToken]) -> None:
        payload = json.dumps({
            "v": STORE_VERSION,
            "tokens": [asdict(token) for token in tokens.values()],
        }, indent=2, sort_keys=True) + "\n"
        device_auth.atomic_write_json(self.path, payload, temp_prefix=".enrol-tokens.")

    def mint(self, *, ttl_seconds: int = ENROL_TOKEN_TTL_SECONDS,
             now: float | None = None) -> tuple[EnrolToken, str]:
        """Create a token; return (record, plaintext token). Never re-read
        by the caller: the plaintext exists nowhere in the store."""
        now = time.time() if now is None else now
        ttl_seconds = min(ttl_seconds, MAX_TTL_SECONDS)
        token = secrets.token_bytes(TOKEN_BYTES)
        token_text = _b64url(token)
        private_key = X25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption())
        record = EnrolToken(
            id=_b64url(secrets.token_bytes(12)),
            token_sha256=token_hash(token_text),
            x25519_private=_b64url(private_bytes),
            created_at=_utc_now_iso(),
            expires_at=now + ttl_seconds,
            state="live",
        )
        with self._exclusive():
            tokens = self._read_file()
            tokens[record.id] = record
            self._write(tokens)
        return record, token_text

    def server_public_key(self, record: EnrolToken) -> str:
        """The token's public key, base64url, for the QR payload."""
        private_key = X25519PrivateKey.from_private_bytes(
            _b64url_decode(record.x25519_private))
        public_bytes = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return _b64url(public_bytes)

    def _effective(self, record: EnrolToken, now: float) -> EnrolToken:
        if record.state == "live" and now >= record.expires_at:
            return replace(record, state="expired", x25519_private=None)
        return record

    def get_live(self, id: str, *, now: float | None = None) -> EnrolToken:
        """Lookup + expiry: the first two steps of the enrolment ordering.

        Returns the still-live record (private key intact) or raises. A
        token discovered expired here is persisted as expired, with its key
        erased, before the refusal is raised — the key does not wait for a
        second touch to be destroyed.
        """
        now = time.time() if now is None else now
        with self._exclusive():
            tokens = self._read_file()
            record = tokens.get(id)
            if record is None or record.state == "cancelled":
                raise EnrolError("enrol_token_unknown", 401)
            if record.state == "used":
                raise EnrolError("enrol_token_used", 409)
            if record.state == "expired":
                raise EnrolError("enrol_token_expired", 401)
            effective = self._effective(record, now)
            if effective.state == "expired":
                tokens[id] = effective
                self._write(tokens)
                raise EnrolError("enrol_token_expired", 401)
            return record

    def burn(self, id: str, *, kid: str, now: float | None = None) -> EnrolToken:
        """Mark a live token used, erase its key, record the enrolling kid.

        Exactly one caller wins a race for the same id: the whole
        read-check-write happens under ``exclusive_lock``, and the loser sees
        the winner's ``used`` state and refuses.
        """
        now = time.time() if now is None else now
        with self._exclusive():
            tokens = self._read_file()
            if self._test_race_delay:
                time.sleep(self._test_race_delay)
            record = tokens.get(id)
            if record is None or record.state == "cancelled":
                raise EnrolError("enrol_token_unknown", 401)
            if record.state == "used":
                raise EnrolError("enrol_token_used", 409)
            effective = self._effective(record, now) if record.state == "live" else record
            if effective.state != "live":
                tokens[id] = effective
                self._write(tokens)
                raise EnrolError("enrol_token_expired", 401)
            burned = replace(record, state="used", x25519_private=None,
                             kid=kid, used_at=_utc_now_iso())
            tokens[id] = burned
            self._write(tokens)
            return burned

    def cancel(self, id: str) -> EnrolToken:
        """Cancel a live token: operator abort or Ctrl-C. Erases its key."""
        with self._exclusive():
            tokens = self._read_file()
            record = tokens.get(id)
            if record is None:
                raise KeyError(id)
            if record.state != "live":
                return record
            cancelled = replace(record, state="cancelled", x25519_private=None)
            tokens[id] = cancelled
            self._write(tokens)
            return cancelled

    def status(self, id: str, *, now: float | None = None) -> EnrolToken:
        record = self.get(id)
        if record is None:
            raise KeyError(id)
        return self._effective(record, time.time() if now is None else now)

    def list(self, *, now: float | None = None) -> list[EnrolToken]:
        now = time.time() if now is None else now
        return [self._effective(record, now)
                for record in sorted(self.tokens().values(), key=lambda t: t.created_at)]

    def which(self, kid: str) -> EnrolToken | None:
        """The token record that enrolled ``kid``, if any — the audit trail
        ``devices.json`` deliberately does not carry (see the module
        docstring on why ``Device`` gains no new field for this)."""
        for record in self.tokens().values():
            if record.kid == kid:
                return record
        return None


def store_from_env() -> TokenStore:
    """The store named by HA_ENROL_TOKEN_FILE, checked at startup.

    Mirrors ``device_auth.registry_from_env``: an enabled feature with no
    usable directory, or an unreadable file, refuses to start rather than
    silently running with a lookup that always fails.
    """
    path = os.environ.get("HA_ENROL_TOKEN_FILE", "")
    if not path:
        raise RuntimeError(
            "QR enrolment refuses to start: HA_ENROL_MODE is on but "
            "HA_ENROL_TOKEN_FILE is unset; point it at a writable, "
            "persistent path for this instance's enrolment tokens.")
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory) or not os.access(directory, os.W_OK):
        raise RuntimeError(
            "QR enrolment refuses to start: the directory of "
            f"HA_ENROL_TOKEN_FILE ({directory}) is missing or not writable.")
    store = TokenStore(path)
    try:
        store.tokens()
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "QR enrolment refuses to start: HA_ENROL_TOKEN_FILE is "
            f"unreadable ({exc}).") from exc
    return store


# --------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m health_advisor.enrol",
        description="Mint, inspect, or cancel single-use QR enrolment tokens.")
    parser.add_argument("--store", default=os.environ.get("HA_ENROL_TOKEN_FILE"),
                        help="the token store file (default: HA_ENROL_TOKEN_FILE)")
    sub = parser.add_subparsers(dest="command", required=True)

    mint = sub.add_parser("mint", help="create a token and print it once, as JSON")
    mint.add_argument("--ttl", type=int, default=ENROL_TOKEN_TTL_SECONDS,
                      help=f"seconds until expiry, capped at {MAX_TTL_SECONDS}")

    status = sub.add_parser("status", help="show one token's current state")
    status.add_argument("id")

    cancel = sub.add_parser("cancel", help="cancel a live token; erases its key")
    cancel.add_argument("id")

    sub.add_parser("list", help="list every token this store knows about")

    which = sub.add_parser("which", help="find the token that enrolled a device kid")
    which.add_argument("kid")

    args = parser.parse_args(argv)
    if not args.store:
        parser.error("no store: pass --store or set HA_ENROL_TOKEN_FILE")
    store = TokenStore(args.store)

    if args.command == "mint":
        record, token_text = store.mint(ttl_seconds=args.ttl)
        print(json.dumps({
            "id": record.id,
            "token": token_text,
            "server_pub": store.server_public_key(record),
            "expires": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.expires_at)),
        }))
        return 0

    if args.command == "status":
        try:
            record = store.status(args.id)
        except KeyError:
            print(f"no token {args.id!r}", file=sys.stderr)
            return 1
        print(f"{record.id}  {record.state}  expires {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(record.expires_at))}"
              f"{'  kid ' + record.kid if record.kid else ''}")
        return 0

    if args.command == "cancel":
        try:
            record = store.cancel(args.id)
        except KeyError:
            print(f"no token {args.id!r}; nothing cancelled", file=sys.stderr)
            return 1
        print(f"{record.id} is now {record.state}")
        return 0

    if args.command == "list":
        records = store.list()
        if not records:
            print("no tokens")
        for record in records:
            print(f"{record.id}  {record.state}  expires "
                  f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(record.expires_at))}"
                  f"{'  kid ' + record.kid if record.kid else ''}")
        return 0

    if args.command == "which":
        record = store.which(args.kid)
        if record is None:
            print(f"no token on record enrolled kid {args.kid!r}", file=sys.stderr)
            return 1
        print(f"{args.kid} was enrolled by token {record.id} (used {record.used_at})")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
