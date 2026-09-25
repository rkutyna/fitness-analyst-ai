"""Rollback detection against a remembered envelope identity (consumer #427, T7).

A caller that remembers the ``envelope_identity`` it last saw can refuse to be
served an older envelope.  Claims under test, each with a mutation that turns
it red:

* an older envelope (a kept copy of an earlier generation) is refused, before
  any key is used or any plaintext written;
* a SAME-generation envelope with different content -- an older state
  re-encrypted up to the generation the caller saw -- is refused.  Compare the
  generation alone and ``test_same_generation_other_content_is_diverged``
  goes red;
* a re-wrap keeps the identity (same generation AND same body header), so it
  is never flagged, and a fresh-data-key re-encryption at the next generation
  (the user-key migration's shape) reads ``newer``.  Compare the digest
  without the generation and ``test_rewrap_is_the_same_identity`` still
  passes but ``test_migration_with_a_fresh_data_key_is_newer`` goes red;
* a v1 -> v2 conversion keeps the generation under a fresh data key, so it
  reads ``diverged``: a caller must not take its first identity from a
  version-1 envelope (pinned here so a change to that rule is deliberate).

Every key is a fixed test value.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from health_advisor import vault_crypto as crypto
from tests.test_vault_crypto_v2 import _write_v1


VK = bytes(range(32))
MASTER = b"m" * 32


class Master:
    def get_master_key(self) -> bytes:
        return MASTER


@pytest.fixture
def vault_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(crypto, "AUDIT_LOG_PATH", tmp_path / "audit" / "vault-unwrapping.jsonl")
    directory = tmp_path / "vault"
    directory.mkdir()
    return directory


def _write(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


def _encrypt(directory: Path, content: bytes, provider, **kw) -> Path:
    envelope = directory / "vault.enc"
    crypto.encrypt_vault(_write(directory, "plain.db", content), envelope, provider=provider,
                         vault_id="test-vault", actor="test", purpose="test", **kw)
    return envelope


def _audit_lines() -> int:
    path = crypto.AUDIT_LOG_PATH
    return len(path.read_text().splitlines()) if path.exists() else 0


def test_identity_is_keyless_and_matches_the_key_block_digest(vault_dir):
    envelope = _encrypt(vault_dir, b"state one", crypto.MemoryKeyProvider(VK))
    identity = crypto.envelope_identity(envelope)
    assert identity["vault_id"] == "test-vault"
    assert identity["generation"] == 1
    with envelope.open("rb") as handle:
        crypto._read_header(handle)
        _, block = crypto._read_key_block(handle)
    assert base64.urlsafe_b64decode(block["header_sha256"] + "==").hex() == identity["header_sha256"]


def test_newer_envelope_is_accepted(vault_dir):
    envelope = _encrypt(vault_dir, b"state one", Master())
    seen = crypto.envelope_identity(envelope)
    _encrypt(vault_dir, b"state two", Master())
    current = crypto.envelope_identity(envelope)
    assert current["generation"] == seen["generation"] + 1
    assert crypto.compare_envelope_identity(seen, current) == "newer"
    crypto.decrypt_vault(envelope, vault_dir / "out.db", provider=Master(), actor="test",
                         purpose="test", expected_identity=seen)
    assert (vault_dir / "out.db").read_bytes() == b"state two"


def test_older_envelope_is_refused_before_any_key_is_used(vault_dir):
    envelope = _encrypt(vault_dir, b"state one", Master())
    kept = vault_dir / "kept.enc"
    kept.write_bytes(envelope.read_bytes())           # the operator keeps a copy
    _encrypt(vault_dir, b"state two -- a deletion", Master())
    seen = crypto.envelope_identity(envelope)
    envelope.write_bytes(kept.read_bytes())           # ...and serves it later
    assert crypto.compare_envelope_identity(seen, crypto.envelope_identity(envelope)) == "older"
    before = _audit_lines()
    with pytest.raises(crypto.RollbackError):
        crypto.decrypt_vault(envelope, vault_dir / "out.db", provider=Master(), actor="test",
                             purpose="test", expected_identity=seen)
    assert isinstance(crypto.RollbackError("x"), crypto.TamperError)
    assert not (vault_dir / "out.db").exists()
    assert _audit_lines() == before, "refused before the unwrap audit, i.e. before any key"
    assert not crypto.find_staging_files(vault_dir)


def test_same_generation_other_content_is_diverged(vault_dir):
    """The attack generation-only comparison misses: restore generation N-1,
    re-encrypt it once, and it is generation N again with the deletion
    undone."""
    envelope = _encrypt(vault_dir, b"state one", Master())
    _encrypt(vault_dir, b"state two -- a deletion", Master())
    seen = crypto.envelope_identity(envelope)
    assert seen["generation"] == 2
    _encrypt(vault_dir, b"state one", Master(), generation=2)   # old state, same number
    current = crypto.envelope_identity(envelope)
    assert current["generation"] == seen["generation"]
    assert crypto.compare_envelope_identity(seen, current) == "diverged"
    with pytest.raises(crypto.RollbackError):
        crypto.decrypt_vault(envelope, vault_dir / "out.db", provider=Master(), actor="test",
                             purpose="test", expected_identity=seen)
    assert not (vault_dir / "out.db").exists()


def test_rewrap_is_the_same_identity(vault_dir):
    """A re-wrap keeps the generation AND the body header, so it is not a
    rollback: moving the vault under another key must never be flagged."""
    envelope = _encrypt(vault_dir, b"state one", Master())
    seen = crypto.envelope_identity(envelope)
    crypto.rewrap_vault(envelope, provider=Master(),
                        new_providers=crypto.MemoryKeyProvider(VK), actor="test", purpose="test")
    assert [s["kind"] for s in crypto.key_slots(envelope)] == ["user"]
    current = crypto.envelope_identity(envelope)
    assert current == seen
    assert crypto.compare_envelope_identity(seen, current) == "same"
    crypto.decrypt_vault(envelope, vault_dir / "out.db", provider=crypto.MemoryKeyProvider(VK),
                         actor="test", purpose="test", expected_identity=seen)
    assert (vault_dir / "out.db").read_bytes() == b"state one"


def test_migration_with_a_fresh_data_key_is_newer(vault_dir):
    """The user-key migration re-encrypts under a fresh data key: the body
    header changes, and the generation moves on with it."""
    envelope = _encrypt(vault_dir, b"state one", Master())
    seen = crypto.envelope_identity(envelope)
    _encrypt(vault_dir, b"state one", crypto.MemoryKeyProvider(VK))  # same content, fresh DEK
    current = crypto.envelope_identity(envelope)
    assert current["header_sha256"] != seen["header_sha256"]
    assert crypto.compare_envelope_identity(seen, current) == "newer"
    crypto.decrypt_vault(envelope, vault_dir / "out.db", provider=crypto.MemoryKeyProvider(VK),
                         actor="test", purpose="test", expected_identity=seen)


def test_another_vault_is_diverged(vault_dir):
    envelope = _encrypt(vault_dir, b"state one", Master())
    seen = dict(crypto.envelope_identity(envelope), vault_id="another-vault")
    assert crypto.compare_envelope_identity(seen, crypto.envelope_identity(envelope)) == "diverged"


def test_v1_conversion_keeps_the_generation_and_so_reads_diverged(vault_dir):
    envelope = vault_dir / "v1.enc"
    _write_v1(_write(vault_dir, "v1.db", b"v1 state"), envelope, MASTER, "test-vault", 4)
    seen = crypto.envelope_identity(envelope)
    assert seen["generation"] == 4
    crypto.rewrap_vault(envelope, provider=Master(), new_providers=Master(),
                        actor="test", purpose="test")
    current = crypto.envelope_identity(envelope)
    assert crypto.inspect_header(envelope)["version"] == 2
    assert current["generation"] == 4
    assert crypto.compare_envelope_identity(seen, current) == "diverged"


@pytest.mark.parametrize("seen", [
    "x", {}, {"vault_id": "test-vault", "generation": 0, "header_sha256": "0" * 64},
    {"vault_id": "test-vault", "generation": True, "header_sha256": "0" * 64},
    {"vault_id": "test-vault", "generation": 1, "header_sha256": "0" * 63},
    {"vault_id": "test-vault", "generation": 1, "header_sha256": "G" * 64},
    {"vault_id": "", "generation": 1, "header_sha256": "0" * 64},
])
def test_a_malformed_expected_identity_is_refused(vault_dir, seen):
    envelope = _encrypt(vault_dir, b"state one", Master())
    with pytest.raises(crypto.VaultCryptoError):
        crypto.compare_envelope_identity(seen, crypto.envelope_identity(envelope))
    with pytest.raises(crypto.VaultCryptoError):
        crypto.decrypt_vault(envelope, vault_dir / "out.db", provider=Master(), actor="test",
                             purpose="test", expected_identity=seen)
    assert not (vault_dir / "out.db").exists()
