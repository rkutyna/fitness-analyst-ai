"""The D23 vector file's own oracle, checked against sources outside this repo.

`tests/fixtures/d23_body_aead_v1.json` is the contract two languages implement
against. Everything downstream of it — the server layer, the iOS layer, and the
byte-for-byte agreement between them — inherits any mistake it contains, and
inherits it *consistently*, which is the dangerous kind: both implementations
would agree with each other and with the file, and all of it would be wrong.

So the file is pinned from underneath, against things that were not written here:

- **RFC 5869 Appendix A**, the published HKDF-SHA256 test vectors, including the
  zero-length-salt case (A.3), which is the branch a hand-rolled extract step
  most often gets wrong.
- **The published AES-256-GCM tag** for the all-zero key, nonce and empty
  plaintext.
- The `cryptography` library's own `HKDF`, as a second opinion on the same
  derivation the reference script rolls by hand.

And from above: every `ok` case is opened with an AAD rebuilt from the case's
published fields, so a file whose `aad_b64` disagrees with its own
`aad_layout` cannot pass.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORS = REPO_ROOT / "tests" / "fixtures" / "d23_body_aead_v1.json"
REFERENCE = REPO_ROOT / "scripts" / "d23_reference.py"


def _reference():
    spec = importlib.util.spec_from_file_location("d23_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _reference()
doc = json.loads(VECTORS.read_text(encoding="utf-8"))


# RFC 5869, Appendix A. Test cases 1-3 for SHA-256.
RFC_5869 = [
    pytest.param(
        bytes.fromhex("0b" * 22),
        bytes.fromhex("000102030405060708090a0b0c"),
        bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"),
        42,
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865",
        id="A.1",
    ),
    pytest.param(
        bytes(range(80)),
        bytes(range(0x60, 0x60 + 80)),
        bytes(range(0xB0, 0xB0 + 80)),
        82,
        "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c"
        "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71"
        "cc30c58179ec3e87c14c01d5c1f3434f1d87",
        id="A.2",
    ),
    pytest.param(
        bytes.fromhex("0b" * 22), b"", b"", 42,
        "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d"
        "9d201395faa4b61a96c8",
        id="A.3-zero-length-salt",
    ),
]


@pytest.mark.parametrize("ikm,salt,info,length,expected", RFC_5869)
def test_the_reference_hkdf_matches_rfc_5869(ikm, salt, info, length, expected):
    assert ref.hkdf_sha256(ikm, salt, info, length).hex() == expected


def test_aes_256_gcm_matches_the_published_all_zero_tag():
    assert AESGCM(bytes(32)).encrypt(bytes(12), b"", b"").hex() == (
        "530f8afbc74536b9a963b4f1c4cb738b")


def test_the_reference_hkdf_agrees_with_the_library():
    secret = base64.b64decode(doc["secret_utf8_b64"])
    salt = base64.b64decode(doc["hkdf_salt_b64"])
    for key, length in (("info_request_b64", 32), ("info_response_b64", 32),
                        ("info_key_id_b64", doc["key_id_bytes"])):
        info = base64.b64decode(doc[key])
        assert ref.hkdf_sha256(secret, salt, info, length) == HKDF(
            algorithm=hashes.SHA256(), length=length, salt=salt, info=info,
        ).derive(secret), key


def test_the_published_keys_and_key_id_are_what_the_parameters_derive():
    secret = base64.b64decode(doc["secret_utf8_b64"])
    salt = base64.b64decode(doc["hkdf_salt_b64"])

    def derive(info_key, length):
        return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt,
                    info=base64.b64decode(doc[info_key])).derive(secret)

    assert base64.b64encode(derive("info_request_b64", 32)).decode() == doc["key_request_b64"]
    assert base64.b64encode(derive("info_response_b64", 32)).decode() == doc["key_response_b64"]
    assert base64.urlsafe_b64encode(
        derive("info_key_id_b64", doc["key_id_bytes"])).decode().rstrip("=") == doc["key_id"]


def test_the_two_directions_and_two_secrets_are_separated():
    """A copy-pasted info string, or a key id that ignores the secret."""
    assert doc["info_request_b64"] != doc["info_response_b64"]
    assert doc["key_request_b64"] != doc["key_response_b64"]
    assert ref.key_id(ref.VECTOR_SECRET) != ref.key_id(ref.OTHER_SECRET)
    assert ref.body_key(ref.VECTOR_SECRET, "request") != \
        ref.body_key(ref.OTHER_SECRET, "request")


def _aad(case, version):
    """Build the AAD from the published layout, not from the reference module."""
    out = bytes([version])
    for field in (case["direction"], case["method"], case["target"], case["key_id"]):
        raw = field.encode("utf-8")
        out += struct.pack(">I", len(raw)) + raw
    return out + struct.pack(">Q", case["timestamp_ms"])


@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c["name"]) for c in doc["cases"]
             if c["expected"] == "ok" and "wire_b64" in c])
def test_every_ok_case_opens_under_its_published_parameters(case):
    wire = base64.b64decode(case["wire_b64"])
    assert hashlib.sha256(wire).hexdigest() == case["wire_sha256"]
    assert len(wire) == case["wire_length"]
    assert wire[0] == doc["version"]
    assert struct.unpack(">Q", wire[1:9])[0] == case["timestamp_ms"]

    aad = _aad(case, doc["version"])
    assert base64.b64encode(aad).decode() == case["aad_b64"], (
        "the file's aad_b64 disagrees with its own aad_layout")

    key = base64.b64decode(doc[f"key_{case['direction']}_b64"])
    header = doc["header_bytes"]
    plain = AESGCM(key).decrypt(wire[9:header], wire[header:], aad)
    assert base64.b64encode(plain).decode() == case["plaintext_b64"]


def test_the_large_case_is_a_reproducible_recipe():
    case = next(c for c in doc["cases"] if c["name"] == "request_large_1_2mb")
    recipe = case["plaintext_recipe"]
    pattern = base64.b64decode(recipe["pattern_b64"])
    body = (pattern * (recipe["length"] // len(pattern) + 1))[:recipe["length"]]
    assert len(body) == case["plaintext_length"] > 1024 * 1024
    assert hashlib.sha256(body).hexdigest() == case["plaintext_sha256"]

    wire, aad = ref.seal(ref.VECTOR_SECRET, case["direction"], case["method"],
                         case["target"].encode(), body, case["timestamp_ms"],
                         base64.b64decode(case["nonce_b64"]))
    assert hashlib.sha256(wire).hexdigest() == case["wire_sha256"]
    assert base64.b64encode(aad).decode() == case["aad_b64"]


def test_the_case_set_still_covers_what_the_contract_requires():
    """A vector deleted in a hurry is how a contract quietly narrows."""
    names = {c["name"] for c in doc["cases"]}
    required = {
        "request_empty_body", "request_one_byte", "request_large_1_2mb",
        "request_non_ascii_utf8", "request_query_bearing_target",
        "response_small_json", "skew_boundary_inclusive",
        "neg_wrong_key", "neg_expired_timestamp", "neg_tampered_aad_target",
        "neg_wrong_version_byte", "neg_tampered_wire_timestamp",
        "neg_direction_confusion",
    }
    assert required <= names, sorted(required - names)
    assert sum(1 for c in doc["cases"] if c["expected"] != "ok") >= 10
