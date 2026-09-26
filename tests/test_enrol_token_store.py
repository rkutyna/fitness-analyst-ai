"""T5 piece 1: the single-use enrolment token store (consumer #426).

Every mutation named in a test's docstring was manually applied and observed
to turn that test red before this file was finalised (device 20260926):

- ``test_second_burn_is_refused``: dropping the ``state == "used"`` check in
  ``TokenStore.burn`` (returning the burned record unconditionally) turns
  this red.
- ``test_expiry_boundary``: changing ``now >= record.expires_at`` to
  ``now > record.expires_at`` in ``TokenStore._effective``/``get_live`` turns
  the exact-boundary assertion red.
- ``test_concurrent_burn_exactly_one_winner``: replacing ``self._exclusive()``
  with ``contextlib.nullcontext()`` turns this red (observed 5/5 "ok" instead
  of 1/5).
- ``test_mint_never_writes_the_plaintext_token``: writing the raw token
  instead of its hash in ``TokenStore.mint`` turns this red.
- ``test_private_key_is_erased_after_burn``: removing
  ``x25519_private=None`` from the ``replace(...)`` call in ``burn`` turns
  this red.
- ``test_ttl_is_capped_at_the_engine_maximum``: removing the
  ``min(ttl_seconds, MAX_TTL_SECONDS)`` clamp in ``mint`` turns this red.
- ``test_store_file_is_mode_600_and_an_unreadable_file_raises``: dropping the
  ``os.chmod(temp_path, 0o600)`` call in ``device_auth.atomic_write_json``,
  or swallowing the parse error in ``TokenStore.tokens``, turns this red.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
import os
import stat
import threading
import time

import pytest

from health_advisor import enrol


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "state" / "enrol-tokens.json"


@pytest.fixture
def store(store_path):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    return enrol.TokenStore(store_path)


# ------------------------------------------------------------- mint / shape


def test_mint_returns_a_live_token_with_a_server_public_key(store):
    record, token = store.mint()
    assert record.state == "live"
    assert record.kid is None
    assert record.used_at is None
    assert len(enrol._b64url_decode(token)) == enrol.TOKEN_BYTES
    pub = store.server_public_key(record)
    assert len(enrol._b64url_decode(pub)) == 32


def test_mint_never_writes_the_plaintext_token(store, store_path):
    """The store is meant to hold only sha256(token); see the module
    docstring. Mutation: have ``mint`` write ``token_text`` in place of its
    hash — this test then finds the token substring in the file."""
    _, token = store.mint()
    raw = store_path.read_text()
    assert token not in raw
    assert "token_sha256" in raw
    assert "x25519_private" in raw  # present while live


def test_ttl_is_capped_at_the_engine_maximum(store):
    now = 1_000_000.0
    requested = enrol.MAX_TTL_SECONDS * 10
    record, _ = store.mint(ttl_seconds=requested, now=now)
    assert record.expires_at - now == enrol.MAX_TTL_SECONDS
    assert enrol.MAX_TTL_SECONDS < requested


def test_store_file_is_mode_600_and_an_unreadable_file_raises(store, store_path):
    store.mint()
    mode = stat.S_IMODE(os.stat(store_path).st_mode)
    assert mode == 0o600

    store_path.write_text("{not json")
    fresh = enrol.TokenStore(store_path)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        fresh.tokens()


# --------------------------------------------------------------- expiry


def test_expiry_boundary(store):
    now = 1_000.0
    record, _ = store.mint(ttl_seconds=10, now=now)
    just_before = store.get_live(record.id, now=record.expires_at - 0.001)
    assert just_before.state == "live"
    assert just_before.x25519_private is not None

    with pytest.raises(enrol.EnrolError) as caught:
        store.get_live(record.id, now=record.expires_at)
    assert caught.value.code == "enrol_token_expired"
    assert caught.value.status == 401

    # The key is erased the moment expiry is discovered, not on a later touch.
    with open(store.path, encoding="utf-8") as handle:
        persisted = json.load(handle)
    stored = next(t for t in persisted["tokens"] if t["id"] == record.id)
    assert stored["state"] == "expired"
    assert stored["x25519_private"] is None


def test_expired_token_cannot_be_burned(store):
    record, _ = store.mint(ttl_seconds=1, now=1_000.0)
    with pytest.raises(enrol.EnrolError) as caught:
        store.burn(record.id, kid="kid-x", now=1_002.0)
    assert caught.value.code == "enrol_token_expired"


# ------------------------------------------------------------------- burn


def test_second_burn_is_refused(store):
    record, _ = store.mint()
    burned = store.burn(record.id, kid="kid-1")
    assert burned.state == "used"
    assert burned.kid == "kid-1"

    with pytest.raises(enrol.EnrolError) as caught:
        store.burn(record.id, kid="kid-2")
    assert caught.value.code == "enrol_token_used"
    assert caught.value.status == 409


def test_private_key_is_erased_after_burn(store, store_path):
    record, _ = store.mint()
    burned = store.burn(record.id, kid="kid-1")
    assert burned.x25519_private is None
    raw = json.loads(store_path.read_text())
    stored = next(t for t in raw["tokens"] if t["id"] == record.id)
    assert stored["x25519_private"] is None
    assert stored["kid"] == "kid-1"
    assert stored["used_at"] is not None


def test_unknown_and_cancelled_tokens_refuse_the_same_way(store):
    with pytest.raises(enrol.EnrolError) as unknown:
        store.get_live("no-such-id")
    assert unknown.value.code == "enrol_token_unknown"

    record, _ = store.mint()
    store.cancel(record.id)
    with pytest.raises(enrol.EnrolError) as cancelled:
        store.get_live(record.id)
    assert cancelled.value.code == "enrol_token_unknown"


def test_cancel_erases_the_key_and_is_idempotent(store, store_path):
    record, _ = store.mint()
    cancelled = store.cancel(record.id)
    assert cancelled.state == "cancelled"
    assert cancelled.x25519_private is None
    again = store.cancel(record.id)
    assert again.state == "cancelled"


def test_concurrent_burn_exactly_one_winner(store):
    """Five threads race to burn the same token. Real exclusivity (flock,
    held for the whole read-check-write) means exactly one succeeds; the
    ``_test_race_delay`` hook widens the window so the race is not a matter
    of luck. See the module docstring for the mutation that turns this red.
    """
    record, _ = store.mint()
    store._test_race_delay = 0.05
    results: list[str] = []
    lock = threading.Lock()

    def worker():
        try:
            store.burn(record.id, kid="racer")
            outcome = "ok"
        except enrol.EnrolError as exc:
            outcome = exc.code
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count("ok") == 1
    assert results.count("enrol_token_used") == 4


def test_removing_exclusivity_lets_both_threads_win(store):
    """The mutation itself, executed: with ``_exclusive`` replaced by a
    no-op, the same race produces more than one winner — proving the
    previous test's lock is load-bearing, not incidental."""
    record, _ = store.mint()
    store._test_race_delay = 0.05
    store._exclusive = lambda: contextlib.nullcontext()
    results: list[str] = []
    lock = threading.Lock()

    def worker():
        try:
            store.burn(record.id, kid="racer")
            outcome = "ok"
        except enrol.EnrolError as exc:
            outcome = exc.code
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count("ok") > 1, (
        "expected the no-op lock to let more than one racer win; if this "
        "fails, flock is doing the same job under a different guise")


# ------------------------------------------------------------ HPKE + reply


def test_hpke_seal_and_open_round_trip(store):
    record, token = store.mint()
    pub = store.server_public_key(record)
    payload = json.dumps({"v": 1, "token": token}).encode("utf-8")
    wire = enrol.seal_to(pub, enrol.info_for(record.id), payload)
    opened = enrol.open_sealed(record.x25519_private, enrol.info_for(record.id), wire)
    assert opened == payload


def test_hpke_open_refuses_a_wire_sealed_to_a_different_token(store):
    a, _ = store.mint()
    b, _ = store.mint()
    pub_b = store.server_public_key(b)
    wire = enrol.seal_to(pub_b, enrol.info_for(b.id), b"payload")
    with pytest.raises(enrol.EnrolError) as caught:
        enrol.open_sealed(a.x25519_private, enrol.info_for(a.id), wire)
    assert caught.value.code == "enrol_undecryptable"
    assert caught.value.status == 400


def test_hpke_open_refuses_the_wrong_info(store):
    record, _ = store.mint()
    pub = store.server_public_key(record)
    wire = enrol.seal_to(pub, enrol.info_for("some-other-id"), b"payload")
    with pytest.raises(enrol.EnrolError):
        enrol.open_sealed(record.x25519_private, enrol.info_for(record.id), wire)


def test_response_seal_open_round_trip_and_wrong_key_fails():
    key = os.urandom(enrol.RESPONSE_KEY_BYTES)
    plaintext = json.dumps({"kid": "abc", "secret": "s", "created": True}).encode()
    wire = enrol.seal_response(key, plaintext)
    assert enrol.open_response(key, wire) == plaintext
    with pytest.raises(Exception):
        enrol.open_response(os.urandom(enrol.RESPONSE_KEY_BYTES), wire)


def test_rfc9180_a2_suite_round_trip():
    """RFC 9180 Appendix A.2 names DHKEM(X25519, HKDF-SHA256), HKDF-SHA256,
    ChaCha20Poly1305 as a base-mode ciphersuite — exactly what ``enrol.py``
    pins via ``hpke.Suite(KEM.X25519, KDF.HKDF_SHA256, AEAD.CHACHA20_POLY1305)``.

    This is a round trip against that exact suite, not a byte-for-byte replay
    of the RFC's published KAT: the installed ``cryptography`` HPKE API used
    by production code (``hpke.Suite.encrypt``/``decrypt``) draws its
    ephemeral sender key from the OS RNG with no way to pin it to the
    published ``skEm``, and does not expose per-message AAD, which the
    official vector's ciphertext was sealed under. Reproducing the vector's
    exact ciphertext bytes would need bypassing the public API for the
    internal ``_encrypt_with_aad``/``_decrypt_with_aad`` bindings *and* the
    vector's exact hex constants transcribed from the RFC text — this
    environment has no network access to fetch and verify those against the
    published spec, so this test instead pins the suite identity and proves
    it decrypts what it seals, using the real recipient key type (X25519)
    RFC 9180 A.2 specifies. Flagged in the T5 handoff as a gap: verify
    byte-level interop against RFC 9180 Appendix A.2 directly when network
    access is available.
    """
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    suite = enrol._HPKE_SUITE
    info = b"Ode on a Grecian Urn"  # the info string RFC 9180's Appendix A examples use
    sk = X25519PrivateKey.generate()
    pk = sk.public_key()
    plaintext = b"Beauty is truth, truth beauty"
    wire = suite.encrypt(plaintext, pk, info=info)
    assert len(wire) == 32 + len(plaintext) + 16  # enc(32) || ct || 16-byte AEAD tag
    opened = suite.decrypt(wire, sk, info=info)
    assert opened == plaintext


def test_hpke_open_a_cryptokit_sealed_fixture_from_ios():
    """Cross-stack interop: bytes sealed by the iOS client's CryptoKit
    ``HPKE.Sender`` (``.Curve25519_SHA256_ChachaPoly``) to a fixed recipient
    key open here to the identical plaintext, and a response sealed by
    ``seal_response`` opens with its ``response_key``. If this goes red the
    two wire formats have drifted apart (consumer #426 T5, piece 6)."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "t5_cryptokit_hpke_fixture.json")
        .read_text(encoding="utf-8"))
    info = enrol.info_for(fixture["enrol_id"])
    assert info == fixture["info_text"].encode("ascii")
    wire = enrol._b64url_decode(fixture["request"]["wire_b64url"])
    opened = enrol.open_sealed(fixture["recipient_private_x25519_raw_b64url"], info, wire)
    assert opened == fixture["request"]["plaintext_json"].encode("utf-8")
    tampered = wire[:-1] + bytes([wire[-1] ^ 1])
    with pytest.raises(Exception):
        enrol.open_sealed(fixture["recipient_private_x25519_raw_b64url"], info, tampered)
    response_key = enrol._b64url_decode(fixture["response"]["response_key_b64url"])
    response_wire = enrol._b64url_decode(fixture["response"]["wire_b64url"])
    assert enrol.open_response(response_key, response_wire) == \
        fixture["response"]["plaintext_json"].encode("utf-8")


# ------------------------------------------------------------------- CLI


def test_cli_mint_prints_json_once(store_path, capsys):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    rc = enrol.main(["--store", str(store_path), "mint"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert set(payload) == {"id", "token", "server_pub", "expires"}


def test_cli_status_list_which_and_cancel(store_path, capsys):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = enrol.TokenStore(store_path)
    record, _ = store.mint()

    rc = enrol.main(["--store", str(store_path), "status", record.id])
    assert rc == 0
    assert "live" in capsys.readouterr().out

    store.burn(record.id, kid="kid-xyz")
    rc = enrol.main(["--store", str(store_path), "which", "kid-xyz"])
    assert rc == 0
    assert record.id in capsys.readouterr().out

    record2, _ = store.mint()
    rc = enrol.main(["--store", str(store_path), "cancel", record2.id])
    assert rc == 0
    assert "cancelled" in capsys.readouterr().out

    rc = enrol.main(["--store", str(store_path), "list"])
    assert rc == 0
    listed = capsys.readouterr().out
    assert record.id in listed and record2.id in listed
