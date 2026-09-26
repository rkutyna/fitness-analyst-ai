"""The D23 AES-256-GCM body envelope.

This module deliberately has no web-framework dependency. The envelope is
opaque to intermediaries, while its clear timestamp is authenticated again in
the AAD so changing it cannot extend the replay window.
"""
from __future__ import annotations

import base64
import os
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


VERSION = 1  # Framing is signaled by CONTENT_TYPE; no byte range is reserved as a discriminator.
CONTENT_TYPE = "application/x-ha-d23"
HKDF_SALT = b"ha-d23-body-aead-v1"
INFO_REQUEST = b"ha/d23/v1/body/request"
INFO_RESPONSE = b"ha/d23/v1/body/response"
INFO_KEY_ID = b"ha/d23/v1/key-id"
# The value a client sends as its authentication header. It is derived from the
# shared secret under its own info label, so an intermediary that reads request
# headers learns a bearer token but nothing from which the body keys above (or
# the secret itself) can be computed.
INFO_AUTH_TOKEN = b"ha/d23/v1/auth-token"
AUTH_TOKEN_BYTES = 32
KEY_BYTES = 32
KEY_ID_BYTES = 8
NONCE_BYTES = 12
TAG_BYTES = 16
HEADER_BYTES = 1 + 8 + NONCE_BYTES
SKEW_SECONDS = 600


class D23Error(ValueError):
    """A protocol refusal, identified by the stable wire-facing error code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class DerivedKeys:
    """The two directional AES keys derived for one trimmed secret."""

    request: bytes
    response: bytes

    def __iter__(self):
        return iter((self.request, self.response))

    def __getitem__(self, direction: str) -> bytes:
        if direction == "request":
            return self.request
        if direction == "response":
            return self.response
        raise KeyError(direction)


@dataclass(frozen=True)
class _DerivedMaterial:
    keys: DerivedKeys
    key_id: str
    auth_token: str


_DERIVED_CACHE_LIMIT = 64
_DERIVED_CACHE: OrderedDict[bytes, _DerivedMaterial] = OrderedDict()
# Guards every read-modify-write of _DERIVED_CACHE. Routes call auth_token /
# derive_keys from worker threads (FastAPI's run_in_threadpool), so a cache
# hit's get()+move_to_end() must be atomic with a concurrent miss's
# insert+evict: without the lock, a hit's move_to_end() can run after another
# thread has already evicted that exact entry, and OrderedDict.move_to_end on
# a missing key raises KeyError (a live 500, not a hypothetical).
_DERIVED_CACHE_LOCK = threading.Lock()


def _secret_bytes(secret: str) -> bytes:
    return secret.strip().encode("utf-8")


def _hkdf(secret: str, info: bytes, length: int) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(), length=length, salt=HKDF_SALT, info=info
    ).derive(_secret_bytes(secret))


def _derived_material(secret: str) -> _DerivedMaterial:
    """Return cached derivations keyed by the exact trimmed secret bytes."""
    secret_b = _secret_bytes(secret)
    with _DERIVED_CACHE_LOCK:
        material = _DERIVED_CACHE.get(secret_b)
        if material is not None:
            _DERIVED_CACHE.move_to_end(secret_b)
            return material

        material = _DerivedMaterial(
            keys=DerivedKeys(
                request=_hkdf(secret, INFO_REQUEST, KEY_BYTES),
                response=_hkdf(secret, INFO_RESPONSE, KEY_BYTES),
            ),
            key_id=base64.urlsafe_b64encode(
                _hkdf(secret, INFO_KEY_ID, KEY_ID_BYTES)
            ).decode("ascii").rstrip("="),
            auth_token=base64.urlsafe_b64encode(
                _hkdf(secret, INFO_AUTH_TOKEN, AUTH_TOKEN_BYTES)
            ).decode("ascii").rstrip("="),
        )
        _DERIVED_CACHE[secret_b] = material
        _DERIVED_CACHE.move_to_end(secret_b)
        while len(_DERIVED_CACHE) > _DERIVED_CACHE_LIMIT:
            _DERIVED_CACHE.popitem(last=False)
        return material


def derive_keys(secret: str) -> DerivedKeys:
    """Derive the request and response AES-256 keys from ``secret``."""
    return _derived_material(secret).keys


def auth_token(secret: str) -> str:
    """Return the unpadded base64url authentication token for ``secret``.

    This is what a client presents in its secret header instead of the secret.
    It is one-way: neither the secret nor the body keys can be derived from it.
    """
    return _derived_material(secret).auth_token


def key_id(secret: str) -> str:
    """Return the unpadded base64url identifier for ``secret``."""
    return _derived_material(secret).key_id


def _field(value: str | bytes, *, ascii_only: bool = False) -> bytes:
    if isinstance(value, bytes):
        return value
    return value.encode("ascii" if ascii_only else "utf-8")


def build_aad(direction: str, method: str, target: str | bytes,
              kid: str, timestamp_ms: int) -> bytes:
    """Build the versioned, four-field, big-endian length-prefixed AAD."""
    try:
        direction_b = _field(direction, ascii_only=True)
        method_b = _field(method, ascii_only=True)
        target_b = _field(target)
        kid_b = _field(kid, ascii_only=True)
        parts = [bytes([VERSION])]
        for value in (direction_b, method_b, target_b, kid_b):
            parts.extend((struct.pack(">I", len(value)), value))
        parts.append(struct.pack(">Q", timestamp_ms))
        return b"".join(parts)
    except (UnicodeEncodeError, struct.error, OverflowError) as exc:
        raise ValueError("invalid D23 AAD fields") from exc


def _key(keys: DerivedKeys, direction: str) -> bytes:
    try:
        return keys[direction]
    except KeyError as exc:
        raise ValueError("invalid D23 direction") from exc


def seal(secret: str, direction: str, method: str, target: str | bytes,
         plaintext: bytes, timestamp_ms: int, nonce: bytes | None = None) -> bytes:
    """Seal ``plaintext`` in a D23 envelope.

    Production callers must leave ``nonce`` as ``None`` so it is drawn from
    ``os.urandom(12)``. Supplying a nonce is for deterministic tests only;
    reusing a (key, nonce) pair under AES-GCM leaks the XOR of two plaintexts
    and the authentication subkey.
    """
    if nonce is None:
        nonce = os.urandom(NONCE_BYTES)
    if len(nonce) != NONCE_BYTES:
        raise ValueError("D23 nonce must be 12 bytes")
    if not isinstance(plaintext, bytes):
        raise TypeError("D23 plaintext must be bytes")
    aad = build_aad(direction, method, target, key_id(secret), timestamp_ms)
    ciphertext_tag = AESGCM(_key(derive_keys(secret), direction)).encrypt(
        nonce, plaintext, aad
    )
    return bytes([VERSION]) + struct.pack(">Q", timestamp_ms) + nonce + ciphertext_tag


def open_(secret: str, direction: str, method: str, target: str | bytes,
          wire: bytes, now_ms: int | None = None, *,
          require_version: bool = False) -> bytes:
    """Open a D23 envelope, checking version, length, skew, then its tag."""
    if not wire:
        raise D23Error("d23_missing")
    if wire[0] != VERSION:
        # The ASGI layer supplies the content type before asking for a strict
        # envelope parse. Preserve the primitive's historical clear-text
        # refusal for direct callers that have no such framing signal.
        if require_version or wire[0] < 0x20 or wire[0] > ord("~"):
            raise D23Error("d23_version")
        raise D23Error("d23_missing")
    if len(wire) < HEADER_BYTES + TAG_BYTES:
        raise D23Error("d23_truncated")
    timestamp_ms = struct.unpack(">Q", wire[1:9])[0]
    if now_ms is None:
        now_ms = time.time_ns() // 1_000_000
    if abs(now_ms - timestamp_ms) > SKEW_SECONDS * 1000:
        raise D23Error("d23_skew")
    aad = build_aad(direction, method, target, key_id(secret), timestamp_ms)
    try:
        return AESGCM(_key(derive_keys(secret), direction)).decrypt(
            wire[9:21], wire[21:], aad
        )
    except (InvalidTag, ValueError, TypeError) as exc:
        raise D23Error("d23_decrypt") from exc


unseal = open_
