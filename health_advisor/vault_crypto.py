"""Streaming envelope encryption for SQLite vault files.

The SQLite layer deliberately knows nothing about this module.  A caller
checks out a vault by decrypting it to a working path, uses ``db.connect`` on
that plaintext path, and encrypts the resulting file again when checking it
back in.

The envelope is intentionally a small, versioned format rather than a
filesystem trick.  Both versions use AES-256-GCM and a freshly generated
256-bit data key (DEK) per envelope, wrapped by a provider-held 256-bit
key-encryption key (KEK).  Plaintext is processed in bounded chunks; neither
encryption nor decryption materialises the vault in memory.  Each chunk nonce
is generated independently with ``secrets.token_bytes(NONCE_SIZE)``.  NIST SP
800-38D limits AES-GCM to 2**32 invocations per key with random 96-bit IVs;
the envelope therefore permits at most 2**31 chunks plus one footer invocation
per data key, a one-bit safety margin below that bound.  With the default 1 MiB
chunk size, the enforced plaintext ceiling is 2**41 bytes (2 TiB), or
2**31 chunks.

Version 1 (read only) puts the wrapped data key inside the public header and
binds that whole header into every chunk's AAD, so moving an envelope under a
different KEK re-encrypts the body.

Version 2 (written by ``encrypt_vault``) separates the two::

    MAGIC | u32 | body header JSON        immutable: vault_id, generation,
                                          sizes, chunk size, dek_id, footer nonce
    chunk records ... | FOOTER             AAD = v2 label + body header bytes
    KEYBLOCK_MAGIC | key block JSON | u32 length | KEYBLOCK_MAGIC

The key block holds 1..n wrap slots ``{kind, kid, nonce, wrapped}``.  Each
slot wraps the DEK under one KEK with AAD binding the SHA-256 of the body
header and the slot's own ``kind``/``kid``; the whole key block is then
authenticated by an HMAC under a key derived from the DEK.  The body is
encrypted under a second key derived from the DEK (HKDF-SHA256, distinct
labels), so nothing in the body depends on any KEK.  Re-wrapping under a new
KEK (``rewrap_vault``) therefore rewrites only the trailing key block: the
header, every chunk and the footer are byte-identical before and after.

The KEK is whatever a ``KeyProvider`` returns.  Today that is the operator's
master key (slot kind ``"master"``).  A provider may declare ``key_kind`` and
``key_id``; a user-held vault key is such a provider with a different kind,
and nothing else in the format changes.

This module implements the cipher only, and delivers exactly that: encryption
at rest, with a per-vault data key wrapped by a provider-held master key.  It
is not a key-management or access-control system.  Both ``KeyProvider``
implementations return the raw master key to whichever caller asks, there is
no role boundary or policy around who may unwrap a vault, and the ``actor``
and ``purpose`` recorded in the mandatory unwrap audit are caller-supplied
strings, validated for shape and not for identity -- the audit log records
what a caller claimed, not an authenticated identity.  That gap is documented
rather than papered over: a key-file provider dressed in a KMS-shaped
interface would be strictly worse, because it would look like a boundary that
is not there.  It is revisited once a hosting provider's KMS and role model
are known (#30).

Envelopes are also complete, immutable snapshots -- each one replaces the
last in full.  There is no encrypted delta log, no restore manifest, and no
retention policy for earlier envelopes, so there is no point-in-time recovery
and no online, resumable migration path across a fleet of vaults; a schema
change means decrypting, rewriting, and re-encrypting every vault in one
pass.  Also deferred rather than half-built, until a hosting deployment's
retention story is decided (#7).

See SECURITY.md for both gaps stated as part of the vault's data-handling
posture.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import errno
import ctypes
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


MAGIC = b"HAVLTENC"
FOOTER_MAGIC = b"HAVLTFTR"
KEYBLOCK_MAGIC = b"HAVLTKEY"
# The version ``encrypt_vault`` writes.  ``decrypt_vault`` reads every entry
# of SUPPORTED_FORMAT_VERSIONS; a version-1 envelope becomes version 2 at its
# next check-in (a plain ``encrypt_vault``) or by ``rewrap_vault``.
FORMAT_VERSION = 2
LEGACY_FORMAT_VERSION = 1
SUPPORTED_FORMAT_VERSIONS = (LEGACY_FORMAT_VERSION, FORMAT_VERSION)
KEYBLOCK_FORMAT = "health-advisor-vault-keys"
KDF_NAME = "HKDF-SHA256"
MASTER_KEY_KIND = "master"
MAX_KEY_BLOCK_SIZE = 64 * 1024
MAX_KEY_SLOTS = 16
DEK_ID_SIZE = 16
CIPHER_NAME = "AES-256-GCM"
DEFAULT_CHUNK_SIZE = 1024 * 1024
MAX_HEADER_SIZE = 1024 * 1024
MAX_CHUNK_COUNT = 1 << 31
MAX_PLAINTEXT_SIZE = MAX_CHUNK_COUNT * DEFAULT_CHUNK_SIZE
MAX_GENERATION = (1 << 63) - 1
NONCE_SIZE = 12
KEY_SIZE = 32
TAG_SIZE = 16
MASTER_KEY_ENV = "HEALTH_ADVISOR_MASTER_KEY"
AUDIT_LOG_ENV = "HEALTH_ADVISOR_VAULT_AUDIT_LOG"
STAGING_PREFIX = ".health-advisor-vault-"
STAGING_SUFFIX = ".staging"
STAGING_FILE_MODE = 0o600


class VaultCryptoError(Exception):
    """Base class for errors which should be shown as a clean CLI failure."""


class VaultFormatError(VaultCryptoError):
    """The envelope is not a valid version-1 vault envelope."""


class TamperError(VaultCryptoError):
    """Authenticated envelope content failed validation."""


class WrongMasterKeyError(VaultCryptoError):
    """The provider key could not unwrap the data key."""


class AuditError(VaultCryptoError):
    """The mandatory unwrap audit could not be written."""


class DurableReplaceStatus(str, Enum):
    """What is known about the directory entry after an atomic replacement."""

    DURABLE = "durable"
    DIRECTORY_SYNC_FAILED = "directory_sync_failed"
    DIRECTORY_SYNC_UNSUPPORTED = "directory_sync_unsupported"


class KeyProvider(Protocol):
    """Source of the key-encryption key (KEK) used to wrap vault data keys.

    The method keeps its historical name: today every provider returns the
    operator's master key.  **This is the seam for a user-held vault key.**
    A provider may also carry two optional attributes, ``key_kind`` and
    ``key_id`` (lower-case labels, default ``"master"``), which name the
    version-2 key slot it writes and the only slot it will try to open.  A
    provider that returns a user-held key under ``key_kind = "user"`` needs
    nothing else from this module: ``encrypt_vault`` writes its slot at a
    check-in, and ``rewrap_vault`` moves an existing envelope under it by
    rewriting only the key block.  Version-1 envelopes have a single implicit
    ``master`` slot.
    """

    def get_master_key(self) -> bytes:
        """Return the 32-byte KEK without writing it to the vault."""


_SLOT_LABEL = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


def _slot_identity(provider: Any) -> tuple[str, str]:
    """The ``(kind, kid)`` key-slot label a provider wraps and unwraps under."""
    kind = getattr(provider, "key_kind", MASTER_KEY_KIND)
    kid = getattr(provider, "key_id", MASTER_KEY_KIND)
    for value, field in ((kind, "key_kind"), (kid, "key_id")):
        if not isinstance(value, str) or not _SLOT_LABEL.match(value):
            raise VaultCryptoError(
                f"key provider {field} must be a lower-case label of at most 64 characters"
            )
    return kind, kid


def _default_audit_log() -> Path:
    """Return the host audit location, always outside a vault directory."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "HealthAdvisor" / "vault-unwrapping.jsonl"
    state = os.environ.get("XDG_STATE_HOME")
    root = Path(state) if state else Path.home() / ".local" / "state"
    return root / "health_advisor" / "vault-unwrapping.jsonl"


# This is a required destination, not an optional callback.  The environment
# override is useful for a host's central log and for isolated tests, but an
# empty value falls back to the default rather than disabling auditing.
AUDIT_LOG_PATH = Path(os.environ.get(AUDIT_LOG_ENV) or _default_audit_log())


class EnvKeyProvider:
    """Read a base64- or hex-encoded 256-bit master key from the environment."""

    def __init__(self, env_var: str = MASTER_KEY_ENV) -> None:
        self.env_var = env_var

    def get_master_key(self) -> bytes:
        raw = os.environ.get(self.env_var)
        if not raw:
            raise VaultCryptoError(
                f"environment variable {self.env_var} is not set; "
                "a 256-bit master key is required"
            )
        try:
            return _decode_key(raw, source=self.env_var)
        except ValueError as exc:
            raise VaultCryptoError(str(exc)) from None


class KeychainKeyProvider:
    """Read a base64- or hex-encoded master key from the macOS Keychain.

    The ``security`` command returns the generic password value; the keychain
    item itself is not a file beside the vault.  This provider intentionally
    does not create or update keychain items.
    """

    def __init__(self, service: str = "health-advisor-vault-master", account: str = "health-advisor") -> None:
        self.service = service
        self.account = account

    def get_master_key(self) -> bytes:
        try:
            result = subprocess.run(
                ["/usr/bin/security", "find-generic-password", "-a", self.account,
                 "-s", self.service, "-w"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            raise VaultCryptoError("macOS security CLI is not available") from None
        except subprocess.CalledProcessError as exc:
            raise VaultCryptoError(
                "could not read the vault master key from macOS Keychain "
                f"(service {self.service!r}, account {self.account!r})"
            ) from exc
        try:
            return _decode_key(result.stdout.strip(), source="macOS Keychain")
        except ValueError as exc:
            raise VaultCryptoError(str(exc)) from None


def _decode_key(value: str, *, source: str) -> bytes:
    """Decode the two portable representations accepted by both providers."""
    candidate = value.strip()
    if len(candidate) == KEY_SIZE * 2:
        try:
            decoded = bytes.fromhex(candidate)
        except ValueError:
            decoded = b""
        if len(decoded) == KEY_SIZE:
            return decoded
    try:
        padded = candidate + "=" * (-len(candidate) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeEncodeError, binascii.Error):
        decoded = b""
    if len(decoded) != KEY_SIZE:
        raise ValueError(
            f"{source} must contain a 32-byte key encoded as hex or base64"
        )
    return decoded


def _master_key(provider: KeyProvider) -> bytes:
    try:
        key = provider.get_master_key()
    except AttributeError:
        raise VaultCryptoError("key provider must implement get_master_key()") from None
    if not isinstance(key, bytes) or len(key) != KEY_SIZE:
        raise VaultCryptoError("key provider must return exactly a 32-byte master key")
    return key


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _unb64(value: Any, field: str) -> bytes:
    if not isinstance(value, str):
        raise VaultFormatError(f"header field {field!r} is not base64 text")
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.b64decode(
            padded.encode("ascii"), altchars=b"-_", validate=True
        )
    except (UnicodeEncodeError, binascii.Error, ValueError):
        raise VaultFormatError(f"header field {field!r} is not valid base64") from None
    return decoded


def _header_bytes(header: dict[str, Any]) -> bytes:
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_identity(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise VaultCryptoError(f"{field} must be a non-empty string of at most 512 characters")
    return value


def _validate_common(header: Any) -> dict[str, Any]:
    """Checks shared by every envelope version: identity, sizes, generation."""
    if not isinstance(header, dict):
        raise VaultFormatError("header is not an object")
    if (header.get("format") != "health-advisor-vault"
            or header.get("version") not in SUPPORTED_FORMAT_VERSIONS
            or isinstance(header.get("version"), bool)):
        raise VaultFormatError("unsupported vault envelope format or version")
    if header.get("cipher") != CIPHER_NAME:
        raise VaultFormatError("unsupported vault cipher")
    vault_id = header.get("vault_id")
    if not isinstance(vault_id, str) or not vault_id or len(vault_id) > 512:
        raise VaultFormatError("header has no valid vault_id")
    chunk_size = header.get("chunk_size")
    size = header.get("plaintext_size")
    count = header.get("chunk_count")
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)
            or chunk_size < 4096 or chunk_size > 16 * 1024 * 1024):
        raise VaultFormatError("header has an invalid chunk_size")
    if (not isinstance(size, int) or isinstance(size, bool)
            or size < 0 or size > MAX_PLAINTEXT_SIZE):
        raise VaultFormatError("header has an invalid plaintext_size")
    if (not isinstance(count, int) or isinstance(count, bool) or count < 0):
        raise VaultFormatError("header has an invalid chunk_count")
    if count > MAX_CHUNK_COUNT:
        raise VaultFormatError("header has too many chunks")
    expected_count = (size + chunk_size - 1) // chunk_size if size else 0
    if count != expected_count:
        raise VaultFormatError("header chunk count does not match plaintext size")
    generation = header.get("generation")
    if generation is not None and (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
        or generation > MAX_GENERATION
    ):
        raise VaultFormatError("header has an invalid generation")
    return header


_V2_HEADER_FIELDS = frozenset({
    "format", "version", "cipher", "kdf", "chunk_size", "plaintext_size",
    "chunk_count", "generation", "vault_id", "dek_id", "footer_nonce",
})


def _validate_header(header: Any) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    """Validate a public header of either version.

    Returns ``(header, canonical bytes, wrapped data key, nonces)``.  For
    version 1 the wrapped key and ``wrap_nonce + footer_nonce`` come from the
    header itself.  A version-2 header carries no key material -- the wrapped
    key lives in the trailing key block -- so it returns ``b""`` and the
    footer nonce alone.
    """
    header = _validate_common(header)
    if header["version"] == LEGACY_FORMAT_VERSION:
        wrapped = _unb64(header.get("wrapped_data_key"), "wrapped_data_key")
        wrap_nonce = _unb64(header.get("wrap_nonce"), "wrap_nonce")
        footer_nonce = _unb64(header.get("footer_nonce"), "footer_nonce")
        if len(wrapped) != KEY_SIZE + TAG_SIZE:
            raise VaultFormatError("wrapped data key has an invalid length")
        if len(wrap_nonce) != NONCE_SIZE or len(footer_nonce) != NONCE_SIZE:
            raise VaultFormatError("header nonce has an invalid length")
        nonces = wrap_nonce + footer_nonce
    else:
        # Version 2 is strict about its field set: a header is immutable and
        # authenticated, so an unexpected field is a writer bug or an attack,
        # never a benign extension.
        if set(header) != _V2_HEADER_FIELDS:
            raise VaultFormatError("version-2 header has missing or unexpected fields")
        if header["generation"] is None:
            raise VaultFormatError("version-2 header requires a generation")
        if header.get("kdf") != KDF_NAME:
            raise VaultFormatError("unsupported vault key derivation")
        if len(_unb64(header.get("dek_id"), "dek_id")) != DEK_ID_SIZE:
            raise VaultFormatError("header dek_id has an invalid length")
        footer_nonce = _unb64(header.get("footer_nonce"), "footer_nonce")
        if len(footer_nonce) != NONCE_SIZE:
            raise VaultFormatError("header nonce has an invalid length")
        wrapped = b""
        nonces = footer_nonce
    encoded = _header_bytes(header)
    if len(encoded) > MAX_HEADER_SIZE:
        raise VaultFormatError("header is too large")
    return header, encoded, wrapped, nonces


def inspect_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate only the public header.  No provider or key is used."""
    with Path(path).open("rb") as handle:
        header_bytes = _read_header(handle)
    header = _parse_header_bytes(header_bytes)
    _validate_header(header)
    return header


def _read_exact(handle: Any, size: int, description: str) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise TamperError(f"vault is truncated while reading {description}")
    return value


def _read_header(handle: Any) -> bytes:
    if _read_exact(handle, len(MAGIC), "magic") != MAGIC:
        raise VaultFormatError("not a Health Advisor vault envelope")
    raw_length = _read_exact(handle, 4, "header length")
    (header_length,) = struct.unpack(">I", raw_length)
    if header_length == 0 or header_length > MAX_HEADER_SIZE:
        raise VaultFormatError("invalid vault header length")
    return _read_exact(handle, header_length, "header")


def _parse_header_bytes(header_bytes: bytes) -> dict[str, Any]:
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise VaultFormatError("vault header is not valid JSON") from None
    _validate_header(header)
    return header


def _wrap_aad(vault_id: str) -> bytes:
    return b"health-advisor data-key wrap v1\0" + vault_id.encode("utf-8")


def _chunk_aad(header_bytes: bytes, index: int) -> bytes:
    return b"health-advisor chunk v1\0" + header_bytes + struct.pack(">Q", index)


def _footer_aad(header_bytes: bytes) -> bytes:
    return b"health-advisor footer v1\0" + header_bytes


# --------------------------------------------------------------- version 2
#
# Key separation: the DEK itself never keys a cipher.  HKDF derives the body
# key (chunks and footer) and the key-block MAC key under distinct labels, so
# the key block's HMAC spends none of the body key's AES-GCM invocation budget.

def _hkdf(key: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=KEY_SIZE, salt=None, info=info).derive(key)


def _v2_body_key(data_key: bytes) -> bytes:
    return _hkdf(data_key, b"health-advisor vault v2 body key")


def _v2_mac_key(data_key: bytes) -> bytes:
    return _hkdf(data_key, b"health-advisor vault v2 key-block mac")


def _chunk_aad_v2(header_bytes: bytes, index: int) -> bytes:
    # The body header only: no wrapped key and no KEK-dependent byte, which
    # is what lets a re-wrap leave every chunk valid.
    return b"health-advisor chunk v2\0" + header_bytes + struct.pack(">Q", index)


def _footer_aad_v2(header_bytes: bytes) -> bytes:
    return b"health-advisor footer v2\0" + header_bytes


def _slot_aad_v2(header_digest: bytes, kind: str, kid: str) -> bytes:
    return (b"health-advisor data-key wrap v2\0" + header_digest + b"\0"
            + kind.encode("ascii") + b"\0" + kid.encode("ascii"))


def _key_block_mac(mac_key: bytes, fields: dict[str, Any]) -> bytes:
    unsigned = {name: value for name, value in fields.items() if name != "mac"}
    return hmac.new(
        mac_key, b"health-advisor key block v2\0" + _header_bytes(unsigned), hashlib.sha256
    ).digest()


def _build_header_v2(
    *, vault_id: str, generation: int, chunk_size: int, plaintext_size: int,
    chunk_count: int, dek_id: bytes, footer_nonce: bytes,
) -> dict[str, Any]:
    return {
        "format": "health-advisor-vault",
        "version": FORMAT_VERSION,
        "cipher": CIPHER_NAME,
        "kdf": KDF_NAME,
        "chunk_size": chunk_size,
        "plaintext_size": plaintext_size,
        "chunk_count": chunk_count,
        "generation": generation,
        "vault_id": vault_id,
        "dek_id": _b64(dek_id),
        "footer_nonce": _b64(footer_nonce),
    }


def _build_key_block(
    header_bytes: bytes, data_key: bytes, providers: Sequence[KeyProvider]
) -> bytes:
    """Wrap ``data_key`` under every provider's KEK and authenticate the block."""
    if not providers:
        raise VaultCryptoError("at least one key provider is required")
    if len(providers) > MAX_KEY_SLOTS:
        raise VaultCryptoError(f"at most {MAX_KEY_SLOTS} key slots are supported")
    digest = hashlib.sha256(header_bytes).digest()
    slots = []
    seen: set[tuple[str, str]] = set()
    for provider in providers:
        kind, kid = _slot_identity(provider)
        if (kind, kid) in seen:
            raise VaultCryptoError(f"duplicate key slot {kind}/{kid}")
        seen.add((kind, kid))
        nonce = secrets.token_bytes(NONCE_SIZE)
        wrapped = AESGCM(_master_key(provider)).encrypt(
            nonce, data_key, _slot_aad_v2(digest, kind, kid)
        )
        slots.append({"kind": kind, "kid": kid, "nonce": _b64(nonce), "wrapped": _b64(wrapped)})
    fields: dict[str, Any] = {
        "format": KEYBLOCK_FORMAT,
        "version": FORMAT_VERSION,
        "header_sha256": _b64(digest),
        "slots": slots,
    }
    fields["mac"] = _b64(_key_block_mac(_v2_mac_key(data_key), fields))
    payload = _header_bytes(fields)
    if len(payload) > MAX_KEY_BLOCK_SIZE:
        raise VaultCryptoError("generated key block is too large")
    return KEYBLOCK_MAGIC + payload + struct.pack(">I", len(payload)) + KEYBLOCK_MAGIC


_KEY_BLOCK_FIELDS = frozenset({"format", "version", "header_sha256", "slots", "mac"})
_SLOT_FIELDS = frozenset({"kind", "kid", "nonce", "wrapped"})
_KEY_BLOCK_TRAILER = 4 + len(KEYBLOCK_MAGIC)


def _parse_key_block(payload: bytes) -> dict[str, Any]:
    """Structural validation only; authentication needs the DEK."""
    try:
        fields = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TamperError("vault key block is not valid JSON") from None
    if not isinstance(fields, dict) or set(fields) != _KEY_BLOCK_FIELDS:
        raise TamperError("vault key block has missing or unexpected fields")
    # One encoding per key block: duplicate JSON keys, whitespace or reordering
    # would otherwise be malleable bytes outside the MAC's canonical input.
    if _header_bytes(fields) != payload:
        raise TamperError("vault key block is not canonically encoded")
    if fields["format"] != KEYBLOCK_FORMAT or fields["version"] != FORMAT_VERSION:
        raise TamperError("unsupported vault key block format or version")
    try:
        digest = _unb64(fields["header_sha256"], "header_sha256")
        mac = _unb64(fields["mac"], "mac")
    except VaultFormatError as exc:
        raise TamperError(str(exc)) from None
    if len(digest) != hashlib.sha256().digest_size or len(mac) != hashlib.sha256().digest_size:
        raise TamperError("vault key block digest or MAC has an invalid length")
    slots = fields["slots"]
    if not isinstance(slots, list) or not 1 <= len(slots) <= MAX_KEY_SLOTS:
        raise TamperError("vault key block has an invalid number of slots")
    seen: set[tuple[str, str]] = set()
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != _SLOT_FIELDS:
            raise TamperError("vault key slot has missing or unexpected fields")
        kind, kid = slot["kind"], slot["kid"]
        if not (isinstance(kind, str) and _SLOT_LABEL.match(kind)
                and isinstance(kid, str) and _SLOT_LABEL.match(kid)):
            raise TamperError("vault key slot has an invalid kind or kid")
        if (kind, kid) in seen:
            raise TamperError("vault key block repeats a slot")
        seen.add((kind, kid))
        try:
            nonce = _unb64(slot["nonce"], "nonce")
            wrapped = _unb64(slot["wrapped"], "wrapped")
        except VaultFormatError as exc:
            raise TamperError(str(exc)) from None
        if len(nonce) != NONCE_SIZE or len(wrapped) != KEY_SIZE + TAG_SIZE:
            raise TamperError("vault key slot has an invalid nonce or wrapped key length")
    return fields


def _read_key_block(handle: Any) -> tuple[int, dict[str, Any]]:
    """Locate the trailing key block from end of file.

    Returns its start offset -- where the authenticated body must end -- and
    its structurally valid fields.  Anything appended after it, and any cut
    through it, breaks the trailer and fails here.
    """
    handle.seek(0, os.SEEK_END)
    end = handle.tell()
    if end < _KEY_BLOCK_TRAILER:
        raise TamperError("vault is truncated: no key block")
    handle.seek(end - _KEY_BLOCK_TRAILER)
    trailer = _read_exact(handle, _KEY_BLOCK_TRAILER, "key block trailer")
    if trailer[4:] != KEYBLOCK_MAGIC:
        raise TamperError("vault key block is missing, or the envelope is truncated or extended")
    (length,) = struct.unpack(">I", trailer[:4])
    if length == 0 or length > MAX_KEY_BLOCK_SIZE:
        raise TamperError("vault key block has an invalid length")
    start = end - _KEY_BLOCK_TRAILER - length - len(KEYBLOCK_MAGIC)
    if start < 0:
        raise TamperError("vault is truncated inside its key block")
    handle.seek(start)
    raw = _read_exact(handle, len(KEYBLOCK_MAGIC) + length, "key block")
    if raw[:len(KEYBLOCK_MAGIC)] != KEYBLOCK_MAGIC:
        raise TamperError("vault key block is missing its leading magic")
    return start, _parse_key_block(raw[len(KEYBLOCK_MAGIC):])


def _verify_key_block_mac(fields: dict[str, Any], data_key: bytes) -> None:
    expected = _key_block_mac(_v2_mac_key(data_key), fields)
    if not hmac.compare_digest(expected, _unb64(fields["mac"], "mac")):
        raise TamperError("vault key block authentication failed")


def _unwrap_v2(fields: dict[str, Any], header_bytes: bytes, provider: KeyProvider) -> bytes:
    """Return the DEK from the provider's slot, then authenticate the block."""
    digest = hashlib.sha256(header_bytes).digest()
    if not hmac.compare_digest(_unb64(fields["header_sha256"], "header_sha256"), digest):
        raise TamperError("vault key block belongs to a different envelope header")
    kind, kid = _slot_identity(provider)
    slot = next((s for s in fields["slots"] if s["kind"] == kind and s["kid"] == kid), None)
    if slot is None:
        raise WrongMasterKeyError(f"envelope has no key slot for {kind}/{kid}")
    try:
        data_key = AESGCM(_master_key(provider)).decrypt(
            _unb64(slot["nonce"], "nonce"), _unb64(slot["wrapped"], "wrapped"),
            _slot_aad_v2(digest, kind, kid),
        )
    except InvalidTag as exc:
        raise WrongMasterKeyError(
            "master key is wrong (or the wrapped data key is corrupt)"
        ) from exc
    _verify_key_block_mac(fields, data_key)
    return data_key


class _BodyWriter:
    """Encrypt plaintext chunks, in order, into records plus a footer."""

    def __init__(self, output: Any, key: bytes, header_bytes: bytes) -> None:
        self._output = output
        self._cipher = AESGCM(key)
        self._header_bytes = header_bytes
        self._chain = hashlib.sha256()
        self.chunks = 0
        self.size = 0

    def write(self, plaintext: bytes) -> None:
        nonce = secrets.token_bytes(NONCE_SIZE)
        ciphertext = self._cipher.encrypt(
            nonce, plaintext, _chunk_aad_v2(self._header_bytes, self.chunks)
        )
        record_prefix = struct.pack(">QI", self.chunks, len(ciphertext)) + nonce
        self._output.write(record_prefix)
        self._output.write(ciphertext)
        self._chain.update(record_prefix)
        self._chain.update(ciphertext)
        self.chunks += 1
        self.size += len(plaintext)

    def finish(self, footer_nonce: bytes) -> None:
        payload = struct.pack(">QQ", self.chunks, self.size) + self._chain.digest()
        footer = self._cipher.encrypt(footer_nonce, payload, _footer_aad_v2(self._header_bytes))
        self._output.write(FOOTER_MAGIC)
        self._output.write(struct.pack(">I", len(footer)))
        self._output.write(footer)


def _staging_path(destination: Path) -> Path:
    """Create a restrictive, identifiable staging file beside ``destination``.

    A crash can leave one of these files behind with plaintext or a partial
    envelope. ``find_staging_files`` lists such leftovers and
    ``remove_staging_files`` removes them after inspection; there is no resume
    protocol for them.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f"{STAGING_PREFIX}{destination.name}-",
        suffix=STAGING_SUFFIX,
        dir=destination.parent,
    )
    try:
        os.fchmod(fd, STAGING_FILE_MODE)
    finally:
        os.close(fd)
    return Path(name)


def find_staging_files(directory: str | os.PathLike[str]) -> list[Path]:
    """Return identifiable crash-leftover staging paths in one directory."""
    root = Path(directory)
    return sorted(root.glob(f"{STAGING_PREFIX}*{STAGING_SUFFIX}"))


def remove_staging_files(directory: str | os.PathLike[str]) -> list[Path]:
    """Remove regular-file staging leftovers and return the paths removed.

    Symlinks are intentionally not followed or removed by this cleanup helper.
    """
    removed: list[Path] = []
    for path in find_staging_files(directory):
        if path.is_symlink() or not path.is_file():
            continue
        path.unlink()
        removed.append(path)
    return removed


def _durable_replace(staging: Path, destination: Path) -> DurableReplaceStatus:
    """Install a file and classify the directory durability outcome.

    Once ``os.replace`` returns, the destination is installed. A directory
    sync failure is therefore reported rather than raised as an installation
    failure. Some platforms cannot open a directory for syncing at all; that
    is distinct from a directory sync that was attempted and failed.
    """
    with staging.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(staging, destination)
    try:
        directory_fd = os.open(destination.parent, os.O_RDONLY)
    except OSError:
        # The file itself is durable; some platforms do not allow opening a
        # directory for fsync.  macOS, the local target, does.
        return DurableReplaceStatus.DIRECTORY_SYNC_UNSUPPORTED
    directory_durable = True
    try:
        try:
            os.fsync(directory_fd)
        except OSError:
            directory_durable = False
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            directory_durable = False
    return (
        DurableReplaceStatus.DURABLE
        if directory_durable
        else DurableReplaceStatus.DIRECTORY_SYNC_FAILED
    )


def _check_paths(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise VaultCryptoError(f"source is not a regular file: {source}")
    try:
        if destination.exists() and os.path.samefile(source, destination):
            raise VaultCryptoError("source and destination must be different files")
    except FileNotFoundError:
        pass


def _check_sqlite_sidecars(source: Path) -> None:
    """Refuse encryption while SQLite has a live journal sidecar."""
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = source.with_name(source.name + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise VaultCryptoError(
                f"source has a live SQLite sidecar: {sidecar.name}; "
                "close the database before encrypting"
            )


def _append_audit_event(*, event: dict[str, Any], source: Path) -> None:
    """Append and fsync one audit event, failing closed if it cannot be written."""
    path = AUDIT_LOG_PATH
    operation = "unwrap" if event.get("event") == "vault_unwrap" else "replace"
    try:
        resolved_log = path.resolve()
        resolved_parent = source.parent.resolve()
        if resolved_log == source.resolve() or resolved_log.is_relative_to(resolved_parent):
            raise AuditError(f"{operation} audit log must be outside the vault directory")
        if path.exists() and not path.is_file():
            raise AuditError(f"{operation} audit log must be a regular file")
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise AuditError(f"{operation} audit log must not be a symlink") from exc
            raise
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise AuditError(f"{operation} audit log must be a regular file")
            if file_stat.st_nlink > 1:
                raise AuditError(f"{operation} audit log must have exactly one hard link")
            os.fchmod(fd, 0o600)
            payload = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
            written = os.write(fd, payload)
            if written != len(payload):
                raise AuditError(f"{operation} audit log event was only partially written")
            os.fsync(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)
    except AuditError:
        raise
    except OSError as exc:
        raise AuditError(f"cannot append {operation} audit log at {path}: {exc}") from exc


def _append_unwrap_audit(*, vault_id: str, actor: str, purpose: str, source: Path) -> None:
    """Append a required audit event before any data-key unwrap is attempted."""
    _append_audit_event(
        event={
            "event": "vault_unwrap",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "purpose": purpose,
            "user": vault_id,
            "vault": vault_id,
        },
        source=source,
    )


def _record_replace_status(
    *,
    status: DurableReplaceStatus,
    vault_id: str,
    actor: str,
    purpose: str,
    source: Path,
    destination: Path,
) -> None:
    """Record replacement durability without undoing an installed replacement.

    This runs after ``os.replace``. The audit event is the durable record of
    the status; if that record cannot itself be written, warn with both facts
    and preserve the already-committed destination.
    """
    event = {
        "event": "vault_replace",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "purpose": purpose,
        "user": vault_id,
        "vault": vault_id,
        "destination": str(destination),
        "durability": status.value,
    }
    try:
        _append_audit_event(event=event, source=source)
    except AuditError as exc:
        warnings.warn(
            f"vault replacement installed at {destination}, but its "
            f"durability status {status.value!r} could not be audited: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _next_generation(destination: Path, vault_id: str) -> int:
    """Return the next generation for a valid envelope at ``destination``."""
    if not destination.is_file():
        return 1
    try:
        previous = inspect_header(destination)
    except (OSError, VaultCryptoError):
        return 1
    if previous.get("vault_id") != vault_id:
        return 1
    generation = previous.get("generation")
    if generation is None:
        return 1
    if generation >= MAX_GENERATION:
        raise VaultCryptoError("vault generation cannot be incremented further")
    return generation + 1


def encrypt_vault(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    provider: KeyProvider,
    vault_id: str,
    actor: str,
    purpose: str,
    generation: int | None = None,
) -> None:
    """Stream-encrypt ``src`` into an atomically replaced version-2 envelope.

    New envelopes carry a positive generation.  When ``generation`` is omitted,
    replacing a valid current envelope at ``dst`` (of either version)
    increments its generation; otherwise generation 1 is used.  Callers
    storing versions under separate object names must supply their own
    monotonically increasing generation.  Replacing a version-1 envelope this
    way is its conversion: a check-in needs no other step.  The data key is
    wrapped in one slot under ``provider``'s KEK.  Plaintext is limited to
    2,199,023,255,552 bytes (2 TiB) and 2,147,483,648 chunks per data key.
    """
    source = Path(src)
    destination = Path(dst)
    _check_paths(source, destination)
    _check_sqlite_sidecars(source)
    vault_id = _validate_identity(vault_id, "vault_id")
    _validate_identity(actor, "actor")
    _validate_identity(purpose, "purpose")
    plaintext_size = source.stat().st_size
    chunk_count = (plaintext_size + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE if plaintext_size else 0
    if plaintext_size > MAX_PLAINTEXT_SIZE or chunk_count > MAX_CHUNK_COUNT:
        raise VaultCryptoError(
            f"source exceeds vault limits ({MAX_PLAINTEXT_SIZE} bytes or "
            f"{MAX_CHUNK_COUNT} chunks)"
        )
    if generation is None:
        generation = _next_generation(destination, vault_id)
    elif (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
        or generation > MAX_GENERATION
    ):
        raise VaultCryptoError(
            f"generation must be an integer from 1 to {MAX_GENERATION}"
        )
    data_key = secrets.token_bytes(KEY_SIZE)
    footer_nonce = secrets.token_bytes(NONCE_SIZE)
    header = _build_header_v2(
        vault_id=vault_id, generation=generation, chunk_size=DEFAULT_CHUNK_SIZE,
        plaintext_size=plaintext_size, chunk_count=chunk_count,
        dek_id=secrets.token_bytes(DEK_ID_SIZE), footer_nonce=footer_nonce,
    )
    header_bytes = _header_bytes(header)
    if len(header_bytes) > MAX_HEADER_SIZE:
        raise VaultCryptoError("generated vault header is too large")
    key_block = _build_key_block(header_bytes, data_key, [provider])
    staging = _staging_path(destination)
    committed = False
    try:
        with source.open("rb") as source_handle, staging.open("wb") as output:
            output.write(MAGIC)
            output.write(struct.pack(">I", len(header_bytes)))
            output.write(header_bytes)
            body = _BodyWriter(output, _v2_body_key(data_key), header_bytes)
            for index in range(chunk_count):
                plaintext = source_handle.read(DEFAULT_CHUNK_SIZE)
                expected_size = min(DEFAULT_CHUNK_SIZE, plaintext_size - index * DEFAULT_CHUNK_SIZE)
                if len(plaintext) != expected_size:
                    raise VaultCryptoError("source changed while it was being encrypted")
                body.write(plaintext)
            if source_handle.read(1):
                raise VaultCryptoError("source changed size while it was being encrypted")
            body.finish(footer_nonce)
            output.write(key_block)
            output.flush()
            os.fsync(output.fileno())
        replace_status = _durable_replace(staging, destination)
        committed = True
        _record_replace_status(
            status=replace_status, vault_id=vault_id, actor=actor, purpose=purpose,
            source=source, destination=destination,
        )
    finally:
        if not committed:
            staging.unlink(missing_ok=True)


def _read_authenticated_body(
    input_handle: Any,
    header_bytes: bytes,
    data_key: bytes,
    footer_nonce: bytes,
    chunk_size: int,
    plaintext_size: int,
    chunk_count: int,
    *,
    write_plaintext: Path | None = None,
    sink: Callable[[bytes], None] | None = None,
    chunk_aad: Callable[[bytes, int], bytes] = _chunk_aad,
    footer_aad: Callable[[bytes], bytes] = _footer_aad,
    body_end: int | None = None,
) -> None:
    """Authenticate one complete body, optionally writing its plaintext.

    The caller performs one pass without ``write_plaintext`` before creating a
    named staging file.  A later pass may write only content whose complete
    envelope was already authenticated.  ``sink`` receives each chunk's
    plaintext in memory (the version-1 conversion re-encrypts it without it
    ever reaching disk); a caller using it must discard its output unless this
    function returns.  ``body_end`` is where the footer must end: end of file
    for version 1, the key block's offset for version 2.
    """
    output = write_plaintext.open("wb") if write_plaintext is not None else None
    try:
        chain = hashlib.sha256()
        for expected_index in range(chunk_count):
            record_prefix = _read_exact(input_handle, 8 + 4 + NONCE_SIZE, "chunk record")
            index, ciphertext_size = struct.unpack(">QI", record_prefix[:12])
            nonce = record_prefix[12:]
            if index != expected_index:
                # A dropped chunk puts the footer where a chunk record should
                # be, and its magic then reads as an enormous index. Report
                # that as truncation rather than as a reordering bug.
                if record_prefix.startswith(FOOTER_MAGIC):
                    raise TamperError(
                        f"vault is missing chunks: the footer appears where "
                        f"chunk {expected_index} of {chunk_count} should be"
                    )
                raise TamperError(
                    f"vault chunks are out of order: expected {expected_index}, got {index}"
                )
            expected_plaintext_size = min(
                chunk_size, plaintext_size - expected_index * chunk_size
            )
            if ciphertext_size != expected_plaintext_size + TAG_SIZE:
                raise TamperError("vault chunk has an invalid ciphertext length")
            ciphertext = _read_exact(input_handle, ciphertext_size, "chunk ciphertext")
            try:
                plaintext = AESGCM(data_key).decrypt(
                    nonce, ciphertext, chunk_aad(header_bytes, expected_index)
                )
            except InvalidTag as exc:
                raise TamperError("vault ciphertext authentication failed") from exc
            if output is not None:
                output.write(plaintext)
            if sink is not None:
                sink(plaintext)
            chain.update(record_prefix)
            chain.update(ciphertext)
        if _read_exact(input_handle, len(FOOTER_MAGIC), "footer magic") != FOOTER_MAGIC:
            raise TamperError("vault footer is missing or the body is truncated")
        footer_size = struct.unpack(">I", _read_exact(input_handle, 4, "footer length"))[0]
        if footer_size != _FOOTER_SIZE:
            raise TamperError("vault footer has an invalid length")
        footer = _read_exact(input_handle, footer_size, "footer")
        try:
            footer_payload = AESGCM(data_key).decrypt(
                footer_nonce, footer, footer_aad(header_bytes)
            )
        except InvalidTag as exc:
            raise TamperError("vault header or footer authentication failed") from exc
        if len(footer_payload) != 8 + 8 + hashlib.sha256().digest_size:
            raise TamperError("vault footer payload has an invalid length")
        footer_count, footer_plaintext_size = struct.unpack(">QQ", footer_payload[:16])
        if (footer_count != chunk_count
                or footer_plaintext_size != plaintext_size
                or footer_payload[16:] != chain.digest()):
            raise TamperError("vault body is reordered, truncated, or otherwise tampered")
        if body_end is None:
            if input_handle.read(1):
                raise TamperError("vault has trailing data after its authenticated footer")
        elif input_handle.tell() != body_end:
            raise TamperError("vault has unauthenticated data between its footer and key block")
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    finally:
        if output is not None:
            output.close()


_FOOTER_SIZE = 8 + 8 + hashlib.sha256().digest_size + TAG_SIZE
_RECORD_OVERHEAD = 8 + 4 + NONCE_SIZE + TAG_SIZE


def _body_length(chunk_count: int, plaintext_size: int) -> int:
    """Bytes from the end of the header to the end of the footer."""
    return (chunk_count * _RECORD_OVERHEAD + plaintext_size
            + len(FOOTER_MAGIC) + 4 + _FOOTER_SIZE)


def _open_data_key(
    handle: Any, header_bytes: bytes, header: dict[str, Any], provider: KeyProvider,
) -> tuple[bytes, dict[str, Any]]:
    """Return the key that decrypts this envelope's body, plus the read plan.

    The unwrap audit has already been written by the caller.  For version 1
    the key is the DEK itself; for version 2 it is the HKDF body key, and the
    key block (MAC included) is authenticated before it is returned.
    """
    _, _, wrapped_data_key, nonces = _validate_header(header)
    if header["version"] == LEGACY_FORMAT_VERSION:
        master_key = _master_key(provider)
        try:
            data_key = AESGCM(master_key).decrypt(
                nonces[:NONCE_SIZE], wrapped_data_key, _wrap_aad(header["vault_id"])
            )
        except InvalidTag as exc:
            raise WrongMasterKeyError(
                "master key is wrong (or the wrapped data key is corrupt)"
            ) from exc
        return data_key, {
            "footer_nonce": nonces[NONCE_SIZE:], "chunk_aad": _chunk_aad,
            "footer_aad": _footer_aad, "body_end": None,
        }
    body_end, fields = _read_key_block(handle)
    data_key = _unwrap_v2(fields, header_bytes, provider)
    return _v2_body_key(data_key), {
        "footer_nonce": nonces, "chunk_aad": _chunk_aad_v2,
        "footer_aad": _footer_aad_v2, "body_end": body_end,
    }


def decrypt_vault(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    provider: KeyProvider,
    actor: str,
    purpose: str,
    expected_generation: int | None = None,
) -> None:
    """Stream-decrypt an envelope into an atomically replaced plaintext file.

    Reads version 1 and version 2.  The complete encrypted body and footer
    (and, for version 2, the key block) are authenticated before any named
    plaintext staging file is created.  If ``expected_generation`` is
    supplied, an envelope without a generation or with an older generation is
    refused.  Version-1 envelopes created before generation support remain
    readable when no expected generation is supplied.  Plaintext is limited to
    2,199,023,255,552 bytes (2 TiB) and 2,147,483,648 chunks per data key.
    """
    source = Path(src)
    destination = Path(dst)
    _check_paths(source, destination)
    _validate_identity(actor, "actor")
    _validate_identity(purpose, "purpose")
    if expected_generation is not None and (
        not isinstance(expected_generation, int)
        or isinstance(expected_generation, bool)
        or expected_generation < 1
        or expected_generation > MAX_GENERATION
    ):
        raise VaultCryptoError(
            f"expected_generation must be an integer from 1 to {MAX_GENERATION}"
        )
    staging: Path | None = None
    committed = False
    try:
        with source.open("rb") as input_handle:
            header_bytes = _read_header(input_handle)
            header = _parse_header_bytes(header_bytes)
            body_offset = input_handle.tell()
            vault_id = header["vault_id"]
            generation = header.get("generation")
            if expected_generation is not None and (
                generation is None or generation < expected_generation
            ):
                raise TamperError(
                    "vault envelope generation is older than the expected generation"
                )

            # The audit is intentionally before provider access and before the
            # first plaintext chunk can be written.  A failed audit aborts the
            # operation instead of becoming a best-effort side effect.
            _append_unwrap_audit(
                vault_id=vault_id, actor=actor, purpose=purpose, source=source
            )
            body_key, plan = _open_data_key(input_handle, header_bytes, header, provider)

            chunk_size = header["chunk_size"]
            plaintext_size = header["plaintext_size"]
            chunk_count = header["chunk_count"]
            input_handle.seek(body_offset)
            _read_authenticated_body(
                input_handle, header_bytes, body_key, plan["footer_nonce"],
                chunk_size, plaintext_size, chunk_count,
                chunk_aad=plan["chunk_aad"], footer_aad=plan["footer_aad"],
                body_end=plan["body_end"],
            )
            input_handle.seek(body_offset)
            staging = _staging_path(destination)
            _read_authenticated_body(
                input_handle, header_bytes, body_key, plan["footer_nonce"],
                chunk_size, plaintext_size, chunk_count,
                write_plaintext=staging,
                chunk_aad=plan["chunk_aad"], footer_aad=plan["footer_aad"],
                body_end=plan["body_end"],
            )
            replace_status = _durable_replace(staging, destination)
            committed = True
            _record_replace_status(
                status=replace_status, vault_id=vault_id, actor=actor, purpose=purpose,
                source=source, destination=destination,
            )
    finally:
        if staging is not None and not committed:
            staging.unlink(missing_ok=True)


# ------------------------------------------------------------------ re-wrap

_DARWIN_CLONE_NOFOLLOW = 0x0001
_LINUX_FICLONE = 0x40049409


def _clone_file(source: Path, staging: Path) -> bool:
    """Make ``staging`` a copy-on-write clone of ``source`` where supported.

    ``staging`` exists (0600, empty) on entry and on a False return.  A clone
    shares the body's blocks, so a version-2 re-wrap writes only its new key
    block.  Where the filesystem cannot clone, the caller copies instead:
    the same ciphertext bytes, O(n) I/O and still no cryptography over the
    body.
    """
    if sys.platform == "darwin":
        try:
            clonefile = ctypes.CDLL(None, use_errno=True).clonefile
        except (OSError, AttributeError):
            return False
        clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
        clonefile.restype = ctypes.c_int
        # clonefile(2) refuses an existing destination, so the staging name is
        # released first; nothing else can take it without failing the clone.
        staging.unlink()
        if clonefile(os.fsencode(source), os.fsencode(staging), _DARWIN_CLONE_NOFOLLOW) == 0:
            os.chmod(staging, STAGING_FILE_MODE)
            return True
        os.close(os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, STAGING_FILE_MODE))
        return False
    if sys.platform.startswith("linux"):
        with source.open("rb") as src_handle, staging.open("r+b") as dst_handle:
            try:
                fcntl.ioctl(dst_handle.fileno(), _LINUX_FICLONE, src_handle.fileno())
                return True
            except OSError:
                return False
    return False


def _rewrap_v2_in_staging(
    staging: Path, *, provider: KeyProvider, new_providers: Sequence[KeyProvider],
    actor: str, purpose: str, source: Path,
) -> str:
    """Replace the key block of the version-2 envelope copy at ``staging``.

    Reads only the header, the key block and the footer.  The footer is
    decrypted to prove that the unwrapped key belongs to this body and that
    the body has the length its header declares; the chunks themselves are
    not read, which is the point, and are authenticated at the next decrypt.
    """
    with staging.open("r+b") as handle:
        header_bytes = _read_header(handle)
        header = _parse_header_bytes(header_bytes)
        if header["version"] != FORMAT_VERSION:
            raise TamperError("vault envelope changed version while it was being re-wrapped")
        body_offset = handle.tell()
        key_block_offset, fields = _read_key_block(handle)
        if body_offset + _body_length(header["chunk_count"], header["plaintext_size"]) != key_block_offset:
            raise TamperError("vault body length does not match its header (truncated or extended)")
        _append_unwrap_audit(
            vault_id=header["vault_id"], actor=actor, purpose=purpose, source=source
        )
        data_key = _unwrap_v2(fields, header_bytes, provider)
        footer_offset = key_block_offset - (len(FOOTER_MAGIC) + 4 + _FOOTER_SIZE)
        handle.seek(footer_offset)
        if (_read_exact(handle, len(FOOTER_MAGIC), "footer magic") != FOOTER_MAGIC
                or struct.unpack(">I", _read_exact(handle, 4, "footer length"))[0] != _FOOTER_SIZE):
            raise TamperError("vault footer is missing or the body is truncated")
        try:
            payload = AESGCM(_v2_body_key(data_key)).decrypt(
                _unb64(header["footer_nonce"], "footer_nonce"),
                _read_exact(handle, _FOOTER_SIZE, "footer"),
                _footer_aad_v2(header_bytes),
            )
        except InvalidTag as exc:
            raise TamperError("vault header or footer authentication failed") from exc
        if struct.unpack(">QQ", payload[:16]) != (header["chunk_count"], header["plaintext_size"]):
            raise TamperError("vault footer does not match its header")
        new_block = _build_key_block(header_bytes, data_key, new_providers)
        # Self-check before anything is installed: every new slot must open.
        _, new_fields = _read_key_block(_BytesTail(new_block))
        for new_provider in new_providers:
            if _unwrap_v2(new_fields, header_bytes, new_provider) != data_key:
                raise VaultCryptoError("re-wrapped key block does not round-trip")
        handle.truncate(key_block_offset)
        handle.seek(key_block_offset)
        handle.write(new_block)
        handle.flush()
        os.fsync(handle.fileno())
    return header["vault_id"]


class _BytesTail:
    """A minimal seekable reader over bytes, for re-parsing a built key block."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._pos = offset if whence == os.SEEK_SET else len(self._data) + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int) -> bytes:
        value = self._data[self._pos:self._pos + size]
        self._pos += len(value)
        return value


def _convert_v1_into_staging(
    source: Path, staging: Path, *, provider: KeyProvider,
    new_providers: Sequence[KeyProvider], actor: str, purpose: str,
) -> str:
    """Re-encrypt a version-1 envelope as version 2, ciphertext to ciphertext.

    Version 1 binds its wrapped key into every chunk's AAD, so this is the one
    re-wrap that must touch the body.  Each chunk is decrypted in memory and
    re-encrypted under a fresh DEK; no plaintext is written anywhere.  The
    whole version-1 body and footer are authenticated before the caller may
    install the result.
    """
    with source.open("rb") as handle, staging.open("wb") as output:
        header_bytes = _read_header(handle)
        header = _parse_header_bytes(header_bytes)
        if header["version"] != LEGACY_FORMAT_VERSION:
            raise TamperError("vault envelope changed version while it was being converted")
        _append_unwrap_audit(
            vault_id=header["vault_id"], actor=actor, purpose=purpose, source=source
        )
        old_key, plan = _open_data_key(handle, header_bytes, header, provider)
        data_key = secrets.token_bytes(KEY_SIZE)
        footer_nonce = secrets.token_bytes(NONCE_SIZE)
        new_header = _build_header_v2(
            vault_id=header["vault_id"], generation=header.get("generation") or 1,
            chunk_size=header["chunk_size"], plaintext_size=header["plaintext_size"],
            chunk_count=header["chunk_count"], dek_id=secrets.token_bytes(DEK_ID_SIZE),
            footer_nonce=footer_nonce,
        )
        new_header_bytes = _header_bytes(new_header)
        key_block = _build_key_block(new_header_bytes, data_key, new_providers)
        output.write(MAGIC)
        output.write(struct.pack(">I", len(new_header_bytes)))
        output.write(new_header_bytes)
        body = _BodyWriter(output, _v2_body_key(data_key), new_header_bytes)
        _read_authenticated_body(
            handle, header_bytes, old_key, plan["footer_nonce"],
            header["chunk_size"], header["plaintext_size"], header["chunk_count"],
            sink=body.write,
        )
        body.finish(footer_nonce)
        output.write(key_block)
        output.flush()
        os.fsync(output.fileno())
    return header["vault_id"]


def rewrap_vault(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str] | None = None,
    *,
    provider: KeyProvider,
    new_providers: KeyProvider | Sequence[KeyProvider],
    actor: str,
    purpose: str,
) -> int:
    """Move an envelope under different KEK(s); returns its format version before.

    ``provider`` opens the current envelope; ``new_providers`` (one provider,
    or up to MAX_KEY_SLOTS with distinct ``key_kind``/``key_id``) each get a
    slot in the result, which replaces ``dst`` (default: ``src`` itself)
    atomically.  A failure at any point leaves the previous envelope intact.

    * Version 2: only the trailing key block is rewritten.  The staging copy
      is a copy-on-write clone where the filesystem supports one, so the cost
      is O(header); elsewhere it is a byte copy with no cryptography over the
      body.  Header, chunks, footer and generation are byte-identical.
    * Version 1: converted to version 2 by re-encrypting ciphertext to
      ciphertext in memory -- the O(body) case, taken once per vault.

    The caller serialises this against check-ins of the same envelope (a
    check-in that lands in between would otherwise be overwritten by the older
    body this copied).  No plaintext is written to disk on either path.
    """
    source = Path(src)
    destination = Path(dst) if dst is not None else source
    if not source.is_file():
        raise VaultCryptoError(f"source is not a regular file: {source}")
    _validate_identity(actor, "actor")
    _validate_identity(purpose, "purpose")
    providers = [new_providers] if hasattr(new_providers, "get_master_key") else list(new_providers)
    if not providers:
        raise VaultCryptoError("at least one new key provider is required")
    with source.open("rb") as handle:
        version = _parse_header_bytes(_read_header(handle))["version"]
    staging = _staging_path(destination)
    committed = False
    try:
        if version == LEGACY_FORMAT_VERSION:
            vault_id = _convert_v1_into_staging(
                source, staging, provider=provider, new_providers=providers,
                actor=actor, purpose=purpose,
            )
        else:
            if not _clone_file(source, staging):
                shutil.copyfile(source, staging)
            vault_id = _rewrap_v2_in_staging(
                staging, provider=provider, new_providers=providers,
                actor=actor, purpose=purpose, source=source,
            )
        replace_status = _durable_replace(staging, destination)
        committed = True
        _record_replace_status(
            status=replace_status, vault_id=vault_id, actor=actor, purpose=purpose,
            source=source, destination=destination,
        )
    finally:
        if not committed:
            staging.unlink(missing_ok=True)
    return version


def _provider_from_args(args: argparse.Namespace) -> KeyProvider:
    if args.provider == "env":
        return EnvKeyProvider(args.key_env)
    return KeychainKeyProvider(args.service, args.account)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encrypt, decrypt, or inspect a Health Advisor vault envelope.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_provider_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--provider", choices=("env", "keychain"), default="env")
        command.add_argument("--key-env", default=MASTER_KEY_ENV)
        command.add_argument("--service", default="health-advisor-vault-master")
        command.add_argument("--account", default="health-advisor")

    encrypt = subparsers.add_parser("encrypt", help="stream-encrypt a plaintext vault")
    encrypt.add_argument("src")
    encrypt.add_argument("dst")
    encrypt.add_argument("--vault-id", required=True)
    encrypt.add_argument("--actor", required=True)
    encrypt.add_argument("--purpose", required=True)
    encrypt.add_argument("--generation", type=int)
    add_provider_options(encrypt)

    decrypt = subparsers.add_parser("decrypt", help="stream-decrypt an encrypted vault")
    decrypt.add_argument("src")
    decrypt.add_argument("dst")
    decrypt.add_argument("--actor", required=True)
    decrypt.add_argument("--purpose", required=True)
    decrypt.add_argument("--expected-generation", type=int)
    add_provider_options(decrypt)

    rewrap = subparsers.add_parser(
        "rewrap",
        help="move an envelope under a new key: rewrites only the key block "
             "(version 2) or converts version 1 to version 2",
    )
    rewrap.add_argument("src")
    rewrap.add_argument("--dst", help="write here instead of replacing SRC")
    rewrap.add_argument("--actor", required=True)
    rewrap.add_argument("--purpose", required=True)
    rewrap.add_argument("--new-key-env", required=True,
                        help="environment variable holding the new 256-bit key")
    add_provider_options(rewrap)

    inspect = subparsers.add_parser("inspect", help="print the public envelope header")
    inspect.add_argument("src")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_header(args.src), indent=2, sort_keys=True))
        elif args.command == "encrypt":
            encrypt_vault(
                args.src, args.dst, provider=_provider_from_args(args),
                vault_id=args.vault_id, actor=args.actor, purpose=args.purpose,
                generation=args.generation,
            )
        elif args.command == "rewrap":
            before = rewrap_vault(
                args.src, args.dst, provider=_provider_from_args(args),
                new_providers=EnvKeyProvider(args.new_key_env),
                actor=args.actor, purpose=args.purpose,
            )
            print(f"rewrap: version {before} -> version {FORMAT_VERSION}")
        else:
            decrypt_vault(
                args.src, args.dst, provider=_provider_from_args(args),
                actor=args.actor, purpose=args.purpose,
                expected_generation=args.expected_generation,
            )
    except (OSError, VaultCryptoError) as exc:
        print(f"vault_crypt: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
