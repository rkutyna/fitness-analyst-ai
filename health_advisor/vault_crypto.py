"""Streaming envelope encryption for SQLite vault files.

The SQLite layer deliberately knows nothing about this module.  A caller
checks out a vault by decrypting it to a working path, uses ``db.connect`` on
that plaintext path, and encrypts the resulting file again when checking it
back in.

The envelope is intentionally a small, versioned format rather than a
filesystem trick.  Version 1 uses AES-256-GCM, a freshly generated 256-bit
data key, and a provider-held 256-bit master key which wraps that data key.
Plaintext is processed in bounded chunks; neither encryption nor decryption
materialises the vault in memory.  In ``encrypt_vault``, each chunk nonce is
generated independently with ``secrets.token_bytes(NONCE_SIZE)``.  NIST SP
800-38D limits AES-GCM to 2**32 invocations per key with random 96-bit IVs;
the envelope therefore permits at most 2**31 chunks plus one footer invocation
per data key, a one-bit safety margin below that bound.  With the default 1 MiB
chunk size, the enforced plaintext ceiling is 2**41 bytes (2 TiB), or
2**31 chunks.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
import struct
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MAGIC = b"HAVLTENC"
FOOTER_MAGIC = b"HAVLTFTR"
FORMAT_VERSION = 1
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
    """Source of the master key used to wrap vault data keys."""

    def get_master_key(self) -> bytes:
        """Return the 32-byte master key without writing it to the vault."""


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


def _validate_header(header: Any) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    if not isinstance(header, dict):
        raise VaultFormatError("header is not an object")
    if header.get("format") != "health-advisor-vault" or header.get("version") != FORMAT_VERSION:
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
    wrapped = _unb64(header.get("wrapped_data_key"), "wrapped_data_key")
    wrap_nonce = _unb64(header.get("wrap_nonce"), "wrap_nonce")
    footer_nonce = _unb64(header.get("footer_nonce"), "footer_nonce")
    if len(wrapped) != KEY_SIZE + TAG_SIZE:
        raise VaultFormatError("wrapped data key has an invalid length")
    if len(wrap_nonce) != NONCE_SIZE or len(footer_nonce) != NONCE_SIZE:
        raise VaultFormatError("header nonce has an invalid length")
    encoded = _header_bytes(header)
    if len(encoded) > MAX_HEADER_SIZE:
        raise VaultFormatError("header is too large")
    return header, encoded, wrapped, wrap_nonce + footer_nonce


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
    """Stream-encrypt ``src`` into an atomically replaced envelope at ``dst``.

    New envelopes carry a positive generation.  When ``generation`` is omitted,
    replacing a valid current envelope at ``dst`` increments its generation;
    otherwise generation 1 is used.  Callers storing versions under separate
    object names must supply their own monotonically increasing generation.
    Plaintext is limited to 2,199,023,255,552 bytes (2 TiB) and
    2,147,483,648 chunks per data key.
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
    master_key = _master_key(provider)
    wrap_nonce = secrets.token_bytes(NONCE_SIZE)
    footer_nonce = secrets.token_bytes(NONCE_SIZE)
    wrapped_data_key = AESGCM(master_key).encrypt(
        wrap_nonce, data_key, _wrap_aad(vault_id)
    )
    header = {
        "format": "health-advisor-vault",
        "version": FORMAT_VERSION,
        "cipher": CIPHER_NAME,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "plaintext_size": plaintext_size,
        "chunk_count": chunk_count,
        "generation": generation,
        "vault_id": vault_id,
        "wrap_nonce": _b64(wrap_nonce),
        "wrapped_data_key": _b64(wrapped_data_key),
        "footer_nonce": _b64(footer_nonce),
    }
    header_bytes = _header_bytes(header)
    if len(header_bytes) > MAX_HEADER_SIZE:
        raise VaultCryptoError("generated vault header is too large")
    staging = _staging_path(destination)
    committed = False
    try:
        chain = hashlib.sha256()
        with source.open("rb") as source_handle, staging.open("wb") as output:
            output.write(MAGIC)
            output.write(struct.pack(">I", len(header_bytes)))
            output.write(header_bytes)
            for index in range(chunk_count):
                plaintext = source_handle.read(DEFAULT_CHUNK_SIZE)
                expected_size = min(DEFAULT_CHUNK_SIZE, plaintext_size - index * DEFAULT_CHUNK_SIZE)
                if len(plaintext) != expected_size:
                    raise VaultCryptoError("source changed while it was being encrypted")
                nonce = secrets.token_bytes(NONCE_SIZE)
                ciphertext = AESGCM(data_key).encrypt(
                    nonce, plaintext, _chunk_aad(header_bytes, index)
                )
                record_prefix = struct.pack(">QI", index, len(ciphertext)) + nonce
                output.write(record_prefix)
                output.write(ciphertext)
                chain.update(record_prefix)
                chain.update(ciphertext)
            if source_handle.read(1):
                raise VaultCryptoError("source changed size while it was being encrypted")
            footer_payload = struct.pack(">QQ", chunk_count, plaintext_size) + chain.digest()
            footer = AESGCM(data_key).encrypt(
                footer_nonce, footer_payload, _footer_aad(header_bytes)
            )
            output.write(FOOTER_MAGIC)
            output.write(struct.pack(">I", len(footer)))
            output.write(footer)
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
) -> None:
    """Authenticate one complete body, optionally writing its plaintext.

    The caller performs one pass without ``write_plaintext`` before creating a
    named staging file.  A later pass may write only content whose complete
    envelope was already authenticated.
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
                    nonce, ciphertext, _chunk_aad(header_bytes, expected_index)
                )
            except InvalidTag as exc:
                raise TamperError("vault ciphertext authentication failed") from exc
            if output is not None:
                output.write(plaintext)
            chain.update(record_prefix)
            chain.update(ciphertext)
        if _read_exact(input_handle, len(FOOTER_MAGIC), "footer magic") != FOOTER_MAGIC:
            raise TamperError("vault footer is missing or the body is truncated")
        footer_size = struct.unpack(">I", _read_exact(input_handle, 4, "footer length"))[0]
        if footer_size != 8 + 8 + hashlib.sha256().digest_size + TAG_SIZE:
            raise TamperError("vault footer has an invalid length")
        footer = _read_exact(input_handle, footer_size, "footer")
        try:
            footer_payload = AESGCM(data_key).decrypt(
                footer_nonce, footer, _footer_aad(header_bytes)
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
        if input_handle.read(1):
            raise TamperError("vault has trailing data after its authenticated footer")
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    finally:
        if output is not None:
            output.close()


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

    The complete encrypted body and footer are authenticated before any named
    plaintext staging file is created.  If ``expected_generation`` is supplied,
    an envelope without a generation or with an older generation is refused.
    Version-1 envelopes created before generation support remain readable when
    no expected generation is supplied.  Plaintext is limited to
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
            _, _, wrapped_data_key, nonces = _validate_header(header)
            wrap_nonce = nonces[:NONCE_SIZE]
            footer_nonce = nonces[NONCE_SIZE:]
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
            master_key = _master_key(provider)
            try:
                data_key = AESGCM(master_key).decrypt(
                    wrap_nonce, wrapped_data_key, _wrap_aad(vault_id)
                )
            except InvalidTag as exc:
                raise WrongMasterKeyError(
                    "master key is wrong (or the wrapped data key is corrupt)"
                ) from exc

            chunk_size = header["chunk_size"]
            plaintext_size = header["plaintext_size"]
            chunk_count = header["chunk_count"]
            body_offset = input_handle.tell()
            _read_authenticated_body(
                input_handle, header_bytes, data_key, footer_nonce,
                chunk_size, plaintext_size, chunk_count,
            )
            input_handle.seek(body_offset)
            staging = _staging_path(destination)
            _read_authenticated_body(
                input_handle, header_bytes, data_key, footer_nonce,
                chunk_size, plaintext_size, chunk_count,
                write_plaintext=staging,
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
