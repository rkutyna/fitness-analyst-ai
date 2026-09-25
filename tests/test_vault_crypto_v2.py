"""Envelope version 2: a separately authenticated key block (consumer #427, T2).

The claims under test, each of which has a mutation that turns it red:

* re-wrapping a version-2 envelope rewrites ONLY the trailing key block --
  header, every chunk and the footer are byte-identical, and the number of
  AES-GCM operations does not grow with the body (binding the body to the KEK
  makes the post-rewrap decrypt fail);
* the key block is authenticated as a whole -- removing a slot, editing the
  MAC, or re-encoding the JSON fails closed (skipping the MAC check lets the
  slot removal through);
* a key block from another envelope, a cut anywhere, or bytes spliced in fail
  closed;
* version 1 is still read, converts at its next check-in or by re-wrap, and an
  interrupted conversion leaves the previous envelope intact with no
  plaintext on disk.

Every key here is a fixed test value, never a real key.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from health_advisor import vault_crypto as crypto


class Provider:
    def __init__(self, key: bytes, kind: str | None = None, kid: str | None = None) -> None:
        self.key = key
        if kind is not None:
            self.key_kind = kind
        if kid is not None:
            self.key_id = kid

    def get_master_key(self) -> bytes:
        return self.key


OLD = Provider(b"o" * 32)
NEW = Provider(b"n" * 32)

# Written by the version-1 engine (main at 2729c3a) with key b"k" * 32,
# vault_id "test-vault", generation 5; plaintext below.
V1_FIXTURE_PLAINTEXT = b"version-1 fixture written by the v1 engine"
V1_FIXTURE = base64.b64decode(
    "SEFWTFRFTkMAAAE8eyJjaHVua19jb3VudCI6MSwiY2h1bmtfc2l6ZSI6MTA0ODU3NiwiY2lwaGVyIjoiQUVTLTI1"
    "Ni1HQ00iLCJmb290ZXJfbm9uY2UiOiJyZnZXOVFYSEdIWm5TTHF4IiwiZm9ybWF0IjoiaGVhbHRoLWFkdmlzb3It"
    "dmF1bHQiLCJnZW5lcmF0aW9uIjo1LCJwbGFpbnRleHRfc2l6ZSI6NDIsInZhdWx0X2lkIjoidGVzdC12YXVsdCIs"
    "InZlcnNpb24iOjEsIndyYXBfbm9uY2UiOiJGY0RBQ3JQQThaaW5WaDhMIiwid3JhcHBlZF9kYXRhX2tleSI6Ildx"
    "VGQxTVZ0SUJPNkdkMDlzWm0yYy0xWFM1RDZISDhfbk5zbHR4a1loTERxYUs2ZnRIajJsaXZoTlQ1aVdkdXcifQAA"
    "AAAAAAAAAAAAOmrHFSpHQc8egjuKzqQLF2Am/YZVZp9t89gin1RhFbT4BQDHHAAhx5ZOLuKe8mmUBDDRNF45HOTf"
    "lasRYm3BkVTw9DejgiJIQVZMVEZUUgAAAEBjA5Kn/efz3uD2bMad4dkyY7WQtk2rHW9ogQMx7ywCc3qPsOixpXy8"
    "pII0GVtjXboZ7x5lsDtdJ/kgNEGSzdxF"
)


@pytest.fixture
def vault_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(crypto, "AUDIT_LOG_PATH", tmp_path / "audit" / "vault-unwrapping.jsonl")
    directory = tmp_path / "vault"
    directory.mkdir()
    return directory


def _plaintext(size: int) -> bytes:
    # Distinct per chunk, so a chunk that decrypts in the wrong place is visible.
    return bytes((i * 31 + i // crypto.DEFAULT_CHUNK_SIZE) % 251 for i in range(size))


def _write_v1(source: Path, destination: Path, key: bytes, vault_id: str, generation: int) -> None:
    """Version-1 writer, byte-for-byte the format of main's encrypt_vault.

    Kept here, not in the module, so production code has no downgrade path.
    ``test_v1_test_writer_matches_the_pinned_v1_fixture`` ties it to real bytes.
    """
    plaintext = source.read_bytes()
    chunk = crypto.DEFAULT_CHUNK_SIZE
    count = (len(plaintext) + chunk - 1) // chunk if plaintext else 0
    data_key = secrets.token_bytes(32)
    wrap_nonce = secrets.token_bytes(12)
    footer_nonce = secrets.token_bytes(12)
    header = {
        "format": "health-advisor-vault", "version": 1, "cipher": crypto.CIPHER_NAME,
        "chunk_size": chunk, "plaintext_size": len(plaintext), "chunk_count": count,
        "generation": generation, "vault_id": vault_id,
        "wrap_nonce": crypto._b64(wrap_nonce),
        "wrapped_data_key": crypto._b64(crypto.AESGCM(key).encrypt(
            wrap_nonce, data_key, crypto._wrap_aad(vault_id))),
        "footer_nonce": crypto._b64(footer_nonce),
    }
    header_bytes = crypto._header_bytes(header)
    out = bytearray(crypto.MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes)
    chain = crypto.hashlib.sha256()
    for index in range(count):
        nonce = secrets.token_bytes(12)
        ciphertext = crypto.AESGCM(data_key).encrypt(
            nonce, plaintext[index * chunk:(index + 1) * chunk],
            crypto._chunk_aad(header_bytes, index))
        prefix = struct.pack(">QI", index, len(ciphertext)) + nonce
        out += prefix + ciphertext
        chain.update(prefix)
        chain.update(ciphertext)
    footer = crypto.AESGCM(data_key).encrypt(
        footer_nonce, struct.pack(">QQ", count, len(plaintext)) + chain.digest(),
        crypto._footer_aad(header_bytes))
    out += crypto.FOOTER_MAGIC + struct.pack(">I", len(footer)) + footer
    destination.write_bytes(bytes(out))


def _layout(blob: bytes) -> dict[str, int]:
    """Offsets of a version-2 envelope: header end, footer start, key block start."""
    header_length = struct.unpack(">I", blob[8:12])[0]
    header_end = 12 + header_length
    key_block_length = struct.unpack(">I", blob[-12:-8])[0]
    key_block_start = len(blob) - 12 - key_block_length - len(crypto.KEYBLOCK_MAGIC)
    return {
        "header_end": header_end,
        "footer_start": key_block_start - (len(crypto.FOOTER_MAGIC) + 4 + crypto._FOOTER_SIZE),
        "key_block_start": key_block_start,
    }


def _key_block(blob: bytes) -> dict:
    start = _layout(blob)["key_block_start"] + len(crypto.KEYBLOCK_MAGIC)
    return json.loads(blob[start:-12])


def _with_key_block(blob: bytes, fields: dict, *, canonical: bool = True) -> bytes:
    payload = (crypto._header_bytes(fields) if canonical
               else json.dumps(fields, sort_keys=True).encode())
    body = blob[:_layout(blob)["key_block_start"]]
    return (body + crypto.KEYBLOCK_MAGIC + payload + struct.pack(">I", len(payload))
            + crypto.KEYBLOCK_MAGIC)


def _encrypt(vault_dir: Path, name: str, size: int, provider=OLD, vault_id="vault-a"):
    source = vault_dir / f"{name}.db"
    source.write_bytes(_plaintext(size))
    envelope = vault_dir / f"{name}.enc"
    crypto.encrypt_vault(source, envelope, provider=provider, vault_id=vault_id,
                         actor="test", purpose="t2")
    return source, envelope


def _decrypt(envelope: Path, provider, out: Path | None = None) -> bytes:
    out = out or envelope.with_suffix(".out")
    crypto.decrypt_vault(envelope, out, provider=provider, actor="test", purpose="t2")
    try:
        return out.read_bytes()
    finally:
        out.unlink()


# ------------------------------------------------------------------ reading v1

def test_v1_test_writer_matches_the_pinned_v1_fixture(vault_dir):
    """The pinned bytes came from the real v1 engine; the helper must match
    their structure exactly, or the helper's round trips prove nothing."""
    pinned = vault_dir / "pinned.enc"
    pinned.write_bytes(V1_FIXTURE)
    assert _decrypt(pinned, Provider(b"k" * 32)) == V1_FIXTURE_PLAINTEXT

    source = vault_dir / "same.db"
    source.write_bytes(V1_FIXTURE_PLAINTEXT)
    ours = vault_dir / "ours.enc"
    _write_v1(source, ours, b"k" * 32, "test-vault", 5)
    theirs_header, ours_header = crypto.inspect_header(pinned), crypto.inspect_header(ours)
    assert set(theirs_header) == set(ours_header)
    assert {k: v for k, v in theirs_header.items() if "nonce" not in k and "wrapped" not in k} \
        == {k: v for k, v in ours_header.items() if "nonce" not in k and "wrapped" not in k}
    assert len(ours.read_bytes()) == len(V1_FIXTURE)
    assert _decrypt(ours, Provider(b"k" * 32)) == V1_FIXTURE_PLAINTEXT


def test_multi_chunk_v1_envelope_still_decrypts(vault_dir):
    source = vault_dir / "v1.db"
    source.write_bytes(_plaintext(3 * crypto.DEFAULT_CHUNK_SIZE + 5))
    envelope = vault_dir / "v1.enc"
    _write_v1(source, envelope, OLD.key, "vault-a", 2)
    assert crypto.inspect_header(envelope)["version"] == 1
    assert _decrypt(envelope, OLD) == source.read_bytes()


# ------------------------------------------------------------------ writing v2

@pytest.mark.parametrize("size", [0, 1, crypto.DEFAULT_CHUNK_SIZE, 2 * crypto.DEFAULT_CHUNK_SIZE + 9])
def test_v2_round_trip_and_the_header_carries_no_key_material(vault_dir, size):
    source, envelope = _encrypt(vault_dir, f"rt{size}", size)
    header = crypto.inspect_header(envelope)
    assert header["version"] == 2 and header["generation"] == 1
    assert set(header) == crypto._V2_HEADER_FIELDS
    assert _decrypt(envelope, OLD) == source.read_bytes()
    slots = _key_block(envelope.read_bytes())["slots"]
    assert [(s["kind"], s["kid"]) for s in slots] == [("master", "master")]


def test_v2_header_refuses_an_unexpected_field():
    header = crypto._build_header_v2(
        vault_id="v", generation=1, chunk_size=crypto.DEFAULT_CHUNK_SIZE, plaintext_size=0,
        chunk_count=0, dek_id=b"d" * 16, footer_nonce=b"f" * 12)
    crypto._validate_header(header)
    with pytest.raises(crypto.VaultFormatError, match="unexpected fields"):
        crypto._validate_header({**header, "wrapped_data_key": "AAAA"})


# ------------------------------------------------------------------ conversion

def test_v1_converts_to_v2_at_its_next_check_in(vault_dir):
    """A check-in is a plain encrypt_vault over the existing envelope."""
    envelope = vault_dir / "vault.enc"
    envelope.write_bytes(V1_FIXTURE)
    plaintext = vault_dir / "checked-out.db"
    crypto.decrypt_vault(envelope, plaintext, provider=Provider(b"k" * 32),
                         actor="test", purpose="checkout")
    crypto.encrypt_vault(plaintext, envelope, provider=Provider(b"k" * 32),
                         vault_id="test-vault", actor="test", purpose="checkin")
    header = crypto.inspect_header(envelope)
    assert header["version"] == 2
    assert header["generation"] == 6  # the v1 envelope's 5, incremented
    assert _decrypt(envelope, Provider(b"k" * 32)) == V1_FIXTURE_PLAINTEXT


def test_v1_rewrap_converts_ciphertext_to_ciphertext(vault_dir):
    source = vault_dir / "v1.db"
    source.write_bytes(_plaintext(2 * crypto.DEFAULT_CHUNK_SIZE + 77))
    envelope = vault_dir / "v1.enc"
    _write_v1(source, envelope, OLD.key, "vault-a", 4)
    assert crypto.rewrap_vault(envelope, provider=OLD, new_providers=NEW,
                               actor="test", purpose="convert") == 1
    header = crypto.inspect_header(envelope)
    assert (header["version"], header["generation"], header["vault_id"]) == (2, 4, "vault-a")
    assert _decrypt(envelope, NEW) == source.read_bytes()
    with pytest.raises(crypto.WrongMasterKeyError):
        _decrypt(envelope, OLD)
    assert crypto.find_staging_files(vault_dir) == []


# ------------------------------------------------------------ O(header) re-wrap

@pytest.mark.parametrize("clone", [True, False], ids=["clone", "copy"])
def test_v2_rewrap_rewrites_only_the_key_block(vault_dir, monkeypatch, clone):
    source, envelope = _encrypt(vault_dir, "big", 5 * crypto.DEFAULT_CHUNK_SIZE + 3)
    before = envelope.read_bytes()
    if not clone:
        monkeypatch.setattr(crypto, "_clone_file", lambda src, dst: False)

    # Count cipher operations: they must not scale with the five chunks.
    counts = {"encrypt": 0, "decrypt": 0}
    real_encrypt, real_decrypt = crypto.AESGCM.encrypt, crypto.AESGCM.decrypt

    def counting(name, real):
        def wrapper(self, *args):
            counts[name] += 1
            return real(self, *args)
        return wrapper

    header_before = crypto.inspect_header(envelope)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(crypto.AESGCM, "encrypt", counting("encrypt", real_encrypt))
        patch.setattr(crypto.AESGCM, "decrypt", counting("decrypt", real_decrypt))
        assert crypto.rewrap_vault(envelope, provider=OLD, new_providers=NEW,
                                   actor="test", purpose="rotate") == 2
    # unwrap + footer check + self-check of the new slot; one new wrap.
    assert counts == {"encrypt": 1, "decrypt": 3}

    after = envelope.read_bytes()
    boundary = _layout(before)["key_block_start"]
    assert _layout(after)["key_block_start"] == boundary
    assert after[:boundary] == before[:boundary]          # header, chunks, footer
    assert after[boundary:] != before[boundary:]          # a new key block
    assert crypto.inspect_header(envelope) == header_before  # generation unchanged too
    assert _decrypt(envelope, NEW) == source.read_bytes()
    with pytest.raises(crypto.WrongMasterKeyError):
        _decrypt(envelope, OLD)
    assert crypto.find_staging_files(vault_dir) == []
    assert oct(envelope.stat().st_mode & 0o777) == oct(crypto.STAGING_FILE_MODE)


def test_rewrap_to_a_separate_destination_leaves_the_source(vault_dir):
    source, envelope = _encrypt(vault_dir, "src", 1000)
    before = envelope.read_bytes()
    target = vault_dir / "rotated.enc"
    crypto.rewrap_vault(envelope, target, provider=OLD, new_providers=NEW,
                        actor="test", purpose="rotate")
    assert envelope.read_bytes() == before
    assert _decrypt(target, NEW) == source.read_bytes()


def test_multiple_slots_each_open_alone_and_kinds_do_not_cross(vault_dir):
    """The T3 seam: a provider of another kind is just another slot."""
    user = Provider(b"u" * 32, kind="user")
    source, envelope = _encrypt(vault_dir, "slots", 4096)
    crypto.rewrap_vault(envelope, provider=OLD, new_providers=[OLD, user],
                        actor="test", purpose="add-user-slot")
    assert _decrypt(envelope, OLD) == source.read_bytes()
    assert _decrypt(envelope, user) == source.read_bytes()
    with pytest.raises(crypto.WrongMasterKeyError, match="master key is wrong"):
        _decrypt(envelope, Provider(b"x" * 32, kind="user"))
    with pytest.raises(crypto.WrongMasterKeyError, match="no key slot for recovery/master"):
        _decrypt(envelope, Provider(b"u" * 32, kind="recovery"))
    with pytest.raises(crypto.VaultCryptoError, match="duplicate key slot"):
        crypto.rewrap_vault(envelope, provider=OLD, new_providers=[OLD, NEW],
                            actor="test", purpose="dup")
    with pytest.raises(crypto.VaultCryptoError, match="lower-case label"):
        crypto.rewrap_vault(envelope, provider=OLD, new_providers=Provider(b"u" * 32, kind="User"),
                            actor="test", purpose="bad-label")


# ------------------------------------------------------------------ fail closed

def _two_slot_envelope(vault_dir):
    source, envelope = _encrypt(vault_dir, "tamper", 2 * crypto.DEFAULT_CHUNK_SIZE + 1)
    crypto.rewrap_vault(envelope, provider=OLD,
                        new_providers=[OLD, Provider(b"u" * 32, kind="user")],
                        actor="test", purpose="setup")
    return source, envelope


def _mutate_slot_wrapped(fields):
    wrapped = bytearray(base64.urlsafe_b64decode(fields["slots"][0]["wrapped"]))
    wrapped[0] ^= 1
    fields["slots"][0]["wrapped"] = base64.urlsafe_b64encode(bytes(wrapped)).decode()


def _mutate_drop_slot(fields):
    fields["slots"] = [s for s in fields["slots"] if s["kind"] == "master"]


def _mutate_mac(fields):
    mac = bytearray(base64.urlsafe_b64decode(fields["mac"]))
    mac[-1] ^= 1
    fields["mac"] = base64.urlsafe_b64encode(bytes(mac)).decode()


def _mutate_relabel(fields):
    fields["slots"][1]["kid"] = "other"


@pytest.mark.parametrize("mutation", [
    _mutate_slot_wrapped, _mutate_drop_slot, _mutate_mac, _mutate_relabel,
], ids=["slot-bytes", "slot-removed", "mac", "slot-relabelled"])
def test_a_tampered_key_block_fails_closed(vault_dir, mutation):
    _, envelope = _two_slot_envelope(vault_dir)
    fields = _key_block(envelope.read_bytes())
    mutation(fields)
    envelope.write_bytes(_with_key_block(envelope.read_bytes(), fields))
    with pytest.raises((crypto.TamperError, crypto.WrongMasterKeyError)):
        _decrypt(envelope, OLD)
    with pytest.raises((crypto.TamperError, crypto.WrongMasterKeyError)):
        crypto.rewrap_vault(envelope, provider=OLD, new_providers=NEW,
                            actor="test", purpose="tampered")


def test_a_removed_slot_is_caught_by_the_key_block_mac(vault_dir):
    """The remaining master slot is itself intact; only the MAC notices."""
    _, envelope = _two_slot_envelope(vault_dir)
    fields = _key_block(envelope.read_bytes())
    _mutate_drop_slot(fields)
    envelope.write_bytes(_with_key_block(envelope.read_bytes(), fields))
    with pytest.raises(crypto.TamperError, match="key block authentication failed"):
        _decrypt(envelope, OLD)


def test_a_non_canonical_key_block_fails_closed(vault_dir):
    _, envelope = _two_slot_envelope(vault_dir)
    fields = _key_block(envelope.read_bytes())
    envelope.write_bytes(_with_key_block(envelope.read_bytes(), fields, canonical=False))
    with pytest.raises(crypto.TamperError, match="canonically encoded"):
        _decrypt(envelope, OLD)


@pytest.mark.parametrize("donor", ["other-vault", "same-vault-other-generation"])
def test_a_key_block_swapped_in_from_another_envelope_fails_closed(vault_dir, donor):
    """Same master key on both sides, so only the header binding can refuse."""
    source, envelope = _encrypt(vault_dir, "victim", 3000, vault_id="vault-a")
    if donor == "other-vault":
        _, other = _encrypt(vault_dir, "donor", 3000, vault_id="vault-b")
    else:
        other = vault_dir / "donor.enc"
        other.write_bytes(envelope.read_bytes())
        crypto.encrypt_vault(source, other, provider=OLD, vault_id="vault-a",
                             actor="test", purpose="t2")
    swapped = _with_key_block(envelope.read_bytes(), _key_block(other.read_bytes()))
    envelope.write_bytes(swapped)
    with pytest.raises(crypto.TamperError, match="different envelope header"):
        _decrypt(envelope, OLD)
    with pytest.raises(crypto.TamperError, match="different envelope header"):
        crypto.rewrap_vault(envelope, provider=OLD, new_providers=NEW,
                            actor="test", purpose="swapped")


@pytest.mark.parametrize("cut", [
    "mid-body", "drop-last-chunk", "before-footer", "mid-footer", "mid-key-block",
    "no-key-block", "last-byte",
])
def test_a_truncated_envelope_fails_closed(vault_dir, cut):
    _, envelope = _encrypt(vault_dir, "trunc", 3 * crypto.DEFAULT_CHUNK_SIZE + 10)
    blob = envelope.read_bytes()
    layout = _layout(blob)
    key_block = blob[layout["key_block_start"]:]
    record = 24 + crypto.DEFAULT_CHUNK_SIZE + 16
    if cut == "mid-body":
        mutated = blob[:layout["header_end"] + record + 100]
    elif cut == "drop-last-chunk":
        # Remove a whole record but keep footer and key block: the key block is
        # found from the end, so the body walk must notice the gap.
        last = layout["footer_start"] - (24 + 10 + 16)
        mutated = blob[:last] + blob[layout["footer_start"]:]
    elif cut == "before-footer":
        mutated = blob[:layout["footer_start"]] + key_block
    elif cut == "mid-footer":
        mutated = blob[:layout["footer_start"] + 20]
    elif cut == "mid-key-block":
        mutated = blob[:layout["key_block_start"] + 30]
    elif cut == "no-key-block":
        mutated = blob[:layout["key_block_start"]]
    else:
        mutated = blob[:-1]
    envelope.write_bytes(mutated)
    with pytest.raises(crypto.VaultCryptoError):
        _decrypt(envelope, OLD)
    with pytest.raises(crypto.VaultCryptoError):
        crypto.rewrap_vault(envelope, provider=OLD, new_providers=NEW,
                            actor="test", purpose="truncated")
    assert envelope.read_bytes() == mutated
    assert crypto.find_staging_files(vault_dir) == []


@pytest.mark.parametrize("where", ["between-footer-and-key-block", "after-key-block"])
def test_spliced_bytes_fail_closed(vault_dir, where):
    _, envelope = _encrypt(vault_dir, "splice", 5000)
    blob = envelope.read_bytes()
    boundary = _layout(blob)["key_block_start"]
    if where == "after-key-block":
        mutated = blob + b"\0"
    else:
        mutated = blob[:boundary] + b"\0" * 16 + blob[boundary:]
    envelope.write_bytes(mutated)
    with pytest.raises(crypto.TamperError):
        _decrypt(envelope, OLD)


def test_a_tampered_body_is_not_laundered_by_a_rewrap(vault_dir):
    """Re-wrap reads no chunk, so it cannot vouch for one: the corruption
    survives the re-wrap and the next decrypt still refuses it."""
    _, envelope = _encrypt(vault_dir, "body", 2 * crypto.DEFAULT_CHUNK_SIZE)
    blob = bytearray(envelope.read_bytes())
    blob[_layout(bytes(blob))["header_end"] + 40] ^= 1
    envelope.write_bytes(bytes(blob))
    crypto.rewrap_vault(envelope, provider=OLD, new_providers=NEW, actor="test", purpose="t2")
    with pytest.raises(crypto.TamperError, match="authentication failed"):
        _decrypt(envelope, NEW)


# -------------------------------------------------------- interrupted conversion

def test_an_interrupted_v1_conversion_keeps_the_v1_envelope(vault_dir, monkeypatch):
    source = vault_dir / "v1.db"
    source.write_bytes(_plaintext(3 * crypto.DEFAULT_CHUNK_SIZE))
    envelope = vault_dir / "v1.enc"
    _write_v1(source, envelope, OLD.key, "vault-a", 3)
    before = envelope.read_bytes()
    real_write = crypto._BodyWriter.write

    def fail_on_third_chunk(self, plaintext):
        if self.chunks == 2:
            raise OSError("simulated crash mid-conversion")
        real_write(self, plaintext)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(crypto._BodyWriter, "write", fail_on_third_chunk)
        with pytest.raises(OSError, match="simulated crash"):
            crypto.rewrap_vault(envelope, provider=OLD, new_providers=NEW,
                                actor="test", purpose="convert")
    assert envelope.read_bytes() == before
    assert crypto.find_staging_files(vault_dir) == []
    assert _decrypt(envelope, OLD) == source.read_bytes()


def test_an_interrupted_check_in_keeps_the_v1_envelope(vault_dir, monkeypatch):
    envelope = vault_dir / "vault.enc"
    envelope.write_bytes(V1_FIXTURE)
    plaintext = vault_dir / "plain.db"
    plaintext.write_bytes(b"newer rows")

    def failing_replace(staging, destination):
        raise OSError("simulated failure before install")

    monkeypatch.setattr(crypto, "_durable_replace", failing_replace)
    with pytest.raises(OSError, match="before install"):
        crypto.encrypt_vault(plaintext, envelope, provider=Provider(b"k" * 32),
                             vault_id="test-vault", actor="test", purpose="checkin")
    assert envelope.read_bytes() == V1_FIXTURE
    assert crypto.find_staging_files(vault_dir) == []


@pytest.mark.parametrize("version", [1, 2])
def test_a_hard_kill_mid_rewrap_keeps_the_old_envelope_and_writes_no_plaintext(
    vault_dir, version,
):
    """os._exit skips every finally: whatever staging survives must hold
    ciphertext only, and the installed envelope must be the old one."""
    source = vault_dir / "kill.db"
    marker = b"PLAINTEXT-MARKER-" * 4096
    source.write_bytes((marker * (3 * crypto.DEFAULT_CHUNK_SIZE // len(marker) + 1))
                       [:3 * crypto.DEFAULT_CHUNK_SIZE])
    envelope = vault_dir / "kill.enc"
    if version == 1:
        _write_v1(source, envelope, OLD.key, "vault-a", 1)
    else:
        crypto.encrypt_vault(source, envelope, provider=OLD, vault_id="vault-a",
                             actor="test", purpose="t2")
    before = envelope.read_bytes()

    child = os.fork()
    if child == 0:  # pragma: no cover - the child exits through os._exit
        if version == 1:
            real_write = crypto._BodyWriter.write

            def die_mid_body(self, plaintext):
                if self.chunks == 2:
                    os._exit(0)
                real_write(self, plaintext)

            crypto._BodyWriter.write = die_mid_body
        else:
            def die_before_install(staging, destination):
                os._exit(0)

            crypto._durable_replace = die_before_install
        try:
            crypto.rewrap_vault(envelope, provider=OLD, new_providers=NEW,
                                actor="test", purpose="kill")
        except BaseException:
            os._exit(2)
        os._exit(3)
    _, status = os.waitpid(child, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    assert envelope.read_bytes() == before
    leftovers = crypto.find_staging_files(vault_dir)
    assert leftovers, "the kill should have stranded a staging file"
    for leftover in leftovers:
        assert b"PLAINTEXT-MARKER" not in leftover.read_bytes()
        assert leftover.stat().st_mode & 0o077 == 0
    assert _decrypt(envelope, OLD) == source.read_bytes()


def test_cli_rewrap_moves_an_envelope_to_a_new_key(vault_dir, tmp_path):
    source, envelope = _encrypt(vault_dir, "cli", 3000)
    environment = os.environ.copy()
    environment["TEST_OLD_KEK"] = base64.b64encode(OLD.key).decode()
    environment["TEST_NEW_KEK"] = NEW.key.hex()
    environment[crypto.AUDIT_LOG_ENV] = str(tmp_path / "cli-audit" / "audit.jsonl")
    result = subprocess.run(
        [sys.executable, "scripts/vault_crypt.py", "rewrap", str(envelope),
         "--key-env", "TEST_OLD_KEK", "--new-key-env", "TEST_NEW_KEK",
         "--actor", "operator", "--purpose", "rotate"],
        check=True, env=environment, capture_output=True, text=True,
    )
    assert "version 2 -> version 2" in result.stdout
    assert _decrypt(envelope, NEW) == source.read_bytes()
