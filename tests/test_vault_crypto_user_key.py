"""The user-held vault key's engine seam (consumer #427, T3).

Claims under test, each with a mutation that turns it red:

* an envelope written under ``MemoryKeyProvider`` carries exactly one ``user``
  slot and no ``master`` slot, so no master-key provider opens it (default the
  provider's kind to ``"master"`` and the S3 test goes red);
* a wrong user key is refused whether it names its own slot label or forges
  the right one;
* ``verify_vault`` authenticates every chunk and writes nothing (skip the body
  pass and the tamper test goes red);
* ``key_slots`` reads the labels without a key, for both versions.

Every key here is a fixed test value, never a real key.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from health_advisor import vault_crypto as crypto


VK = bytes(range(32))
OTHER_VK = bytes(range(1, 33))
MASTER = b"m" * 32


class MasterProvider:
    def get_master_key(self) -> bytes:
        return MASTER


@pytest.fixture
def vault_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(crypto, "AUDIT_LOG_PATH", tmp_path / "audit" / "vault-unwrapping.jsonl")
    directory = tmp_path / "vault"
    directory.mkdir()
    return directory


def _plaintext(directory: Path, size: int = 3 * crypto.DEFAULT_CHUNK_SIZE + 17) -> Path:
    source = directory / "plain.db"
    source.write_bytes(bytes((i * 7) % 253 for i in range(size)))
    return source


def _encrypt(directory: Path, provider) -> Path:
    envelope = directory / "vault.enc"
    crypto.encrypt_vault(_plaintext(directory), envelope, provider=provider,
                         vault_id="test-vault", actor="test", purpose="test")
    return envelope


def test_user_key_id_is_stable_distinct_and_a_valid_label():
    kid = crypto.user_key_id(VK)
    assert kid == crypto.user_key_id(VK)
    assert kid != crypto.user_key_id(OTHER_VK)
    assert kid.startswith("vk-") and len(kid) == 3 + 16
    assert VK.hex()[:16] not in kid
    with pytest.raises(crypto.VaultCryptoError):
        crypto.user_key_id(b"short")


def test_memory_provider_writes_only_a_user_slot(vault_dir):
    envelope = _encrypt(vault_dir, crypto.MemoryKeyProvider(VK))
    assert crypto.key_slots(envelope) == [
        {"kind": "user", "kid": crypto.user_key_id(VK)}]


def test_s3_no_master_key_provider_opens_a_user_keyed_envelope(vault_dir):
    """S3: once an envelope is written under the user's key, the operator's
    master key -- under its own label, or relabelled as a user slot -- opens
    nothing."""
    envelope = _encrypt(vault_dir, crypto.MemoryKeyProvider(VK))
    out = vault_dir / "out.db"
    with pytest.raises(crypto.WrongMasterKeyError):
        crypto.decrypt_vault(envelope, out, provider=MasterProvider(),
                             actor="test", purpose="test")
    relabelled = crypto.MemoryKeyProvider(MASTER, key_kind="user",
                                          key_id=crypto.user_key_id(VK))
    with pytest.raises(crypto.WrongMasterKeyError):
        crypto.decrypt_vault(envelope, out, provider=relabelled,
                             actor="test", purpose="test")
    assert not out.exists()
    crypto.decrypt_vault(envelope, out, provider=crypto.MemoryKeyProvider(VK),
                         actor="test", purpose="test")
    assert out.read_bytes() == (vault_dir / "plain.db").read_bytes()


def test_wrong_user_key_is_refused(vault_dir):
    envelope = _encrypt(vault_dir, crypto.MemoryKeyProvider(VK))
    with pytest.raises(crypto.WrongMasterKeyError, match="no key slot"):
        crypto.verify_vault(envelope, provider=crypto.MemoryKeyProvider(OTHER_VK),
                            actor="test", purpose="test")
    forged = crypto.MemoryKeyProvider(OTHER_VK, key_id=crypto.user_key_id(VK))
    with pytest.raises(crypto.WrongMasterKeyError):
        crypto.verify_vault(envelope, provider=forged, actor="test", purpose="test")


def test_verify_vault_authenticates_every_chunk_and_writes_nothing(vault_dir):
    envelope = _encrypt(vault_dir, crypto.MemoryKeyProvider(VK))
    before = sorted(p.name for p in vault_dir.iterdir())
    header = crypto.verify_vault(envelope, provider=crypto.MemoryKeyProvider(VK),
                                 actor="test", purpose="test")
    assert header["vault_id"] == "test-vault"
    assert sorted(p.name for p in vault_dir.iterdir()) == before
    blob = bytearray(envelope.read_bytes())
    # A byte in the middle of the second chunk: the key block and footer are
    # untouched, so only a pass over the body can see it.
    blob[12 + 400 + crypto.DEFAULT_CHUNK_SIZE + 5000] ^= 0x01
    envelope.write_bytes(bytes(blob))
    with pytest.raises(crypto.TamperError):
        crypto.verify_vault(envelope, provider=crypto.MemoryKeyProvider(VK),
                            actor="test", purpose="test")


def test_key_slots_reads_both_versions_and_follows_a_rewrap(vault_dir):
    envelope = _encrypt(vault_dir, MasterProvider())
    assert crypto.key_slots(envelope) == [{"kind": "master", "kid": "master"}]
    crypto.rewrap_vault(envelope, provider=MasterProvider(),
                        new_providers=crypto.MemoryKeyProvider(VK),
                        actor="test", purpose="test")
    assert [s["kind"] for s in crypto.key_slots(envelope)] == ["user"]


def test_memory_provider_never_prints_its_key_and_can_be_wiped():
    provider = crypto.MemoryKeyProvider(VK)
    assert VK.hex() not in repr(provider) and str(list(VK)) not in repr(provider)
    assert provider.matches(VK) and not provider.matches(OTHER_VK)
    provider.wipe()
    with pytest.raises(crypto.VaultCryptoError, match="wiped"):
        provider.get_master_key()


def test_slots_cli_prints_json(vault_dir):
    envelope = _encrypt(vault_dir, crypto.MemoryKeyProvider(VK))
    result = subprocess.run(
        [sys.executable, "-m", "health_advisor.vault_crypto", "slots", str(envelope)],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(Path(crypto.__file__).resolve().parents[1])},
    )
    assert json.loads(result.stdout) == [{"kind": "user", "kid": crypto.user_key_id(VK)}]
