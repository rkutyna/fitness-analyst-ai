"""Contract tests for the streaming vault envelope."""
from __future__ import annotations

import base64
import errno
import json
import os
import sqlite3
import stat
import struct
import subprocess
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

from health_advisor import vault_crypto as crypto


class FixedProvider:
    def __init__(self, key: bytes) -> None:
        self.key = key
        self.calls = 0

    def get_master_key(self) -> bytes:
        self.calls += 1
        return self.key


def _write_source(path: Path, size: int = 2 * crypto.DEFAULT_CHUNK_SIZE + 123) -> bytes:
    value = bytes((i * 17 + 3) % 256 for i in range(size))
    path.write_bytes(value)
    return value


def _set_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    audit = tmp_path / "audit" / "vault-unwrapping.jsonl"
    monkeypatch.setattr(crypto, "AUDIT_LOG_PATH", audit)
    return audit


def _encrypt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    source = vault_dir / "plain.db"
    ciphertext = vault_dir / "plain.db.enc"
    original = _write_source(source)
    audit = _set_audit(monkeypatch, tmp_path)
    provider = FixedProvider(b"m" * 32)
    crypto.encrypt_vault(
        source, ciphertext, provider=provider, vault_id="user-123",
        actor="worker-7", purpose="analytics-checkout",
    )
    return source, ciphertext, original, audit


def test_round_trip_is_byte_identical_and_inspect_needs_no_key(tmp_path, monkeypatch):
    source, ciphertext, original, audit = _encrypt(tmp_path, monkeypatch)
    header = crypto.inspect_header(ciphertext)
    assert header["version"] == 1
    assert header["vault_id"] == "user-123"
    assert header["chunk_count"] == 3
    replace_event = json.loads(audit.read_text().splitlines()[0])
    assert replace_event["event"] == "vault_replace"
    assert replace_event["durability"] == crypto.DurableReplaceStatus.DURABLE.value

    restored = tmp_path / "restored.db"
    crypto.decrypt_vault(
        ciphertext, restored, provider=FixedProvider(b"m" * 32),
        actor="worker-7", purpose="analytics-checkout",
    )
    assert restored.read_bytes() == original
    event = json.loads(audit.read_text().splitlines()[-2])
    assert event["event"] == "vault_unwrap"
    assert event["actor"] == "worker-7"
    assert event["purpose"] == "analytics-checkout"
    assert event["user"] == "user-123"
    assert event["vault"] == "user-123"


def test_older_envelope_is_refused_when_expected_generation_is_newer(tmp_path, monkeypatch):
    """A valid older object cannot replace the generation the caller expects."""
    _set_audit(monkeypatch, tmp_path)
    vault_dir = tmp_path / "generation-vault"
    vault_dir.mkdir()
    source = vault_dir / "source.db"
    encrypted = vault_dir / "current.enc"
    older = vault_dir / "older.enc"
    source.write_bytes(b"first version")
    provider = FixedProvider(b"g" * 32)

    crypto.encrypt_vault(
        source, encrypted, provider=provider, vault_id="generation-vault",
        actor="worker", purpose="generation-test",
    )
    older.write_bytes(encrypted.read_bytes())
    assert crypto.inspect_header(older)["generation"] == 1

    source.write_bytes(b"second version")
    crypto.encrypt_vault(
        source, encrypted, provider=provider, vault_id="generation-vault",
        actor="worker", purpose="generation-test",
    )
    current_generation = crypto.inspect_header(encrypted)["generation"]
    assert current_generation == 2

    encrypted.write_bytes(older.read_bytes())
    restored = vault_dir / "restored.db"
    child = os.fork()
    if child == 0:  # pragma: no cover - the child reports through its exit code
        try:
            crypto.decrypt_vault(
                encrypted, restored, provider=provider, actor="worker",
                purpose="generation-test", expected_generation=current_generation,
            )
        except crypto.TamperError:
            os._exit(0)
        except BaseException:
            os._exit(2)
        os._exit(1)

    _, status = os.waitpid(child, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    assert not restored.exists()
    assert crypto.find_staging_files(vault_dir) == []


def test_hard_kill_during_verification_leaves_no_plaintext_staging(tmp_path, monkeypatch):
    """The real process-kill boundary occurs before named staging exists."""
    _set_audit(monkeypatch, tmp_path)
    vault_dir = tmp_path / "crash-vault"
    vault_dir.mkdir()
    source = vault_dir / "source.db"
    encrypted = vault_dir / "source.enc"
    restored = vault_dir / "restored.db"
    _write_source(source, 3 * crypto.DEFAULT_CHUNK_SIZE + 17)
    provider = FixedProvider(b"h" * 32)
    crypto.encrypt_vault(
        source, encrypted, provider=provider, vault_id="crash-vault",
        actor="worker", purpose="crash-test",
    )

    child = os.fork()
    if child == 0:  # pragma: no cover - the child exits intentionally
        real_decrypt = crypto.AESGCM.decrypt
        calls = 0

        def kill_after_two_chunks(cipher, nonce, ciphertext, associated_data):
            nonlocal calls
            calls += 1
            plaintext = real_decrypt(cipher, nonce, ciphertext, associated_data)
            # Call 1 unwraps the data key; calls 2 and 3 authenticate chunks.
            if calls == 3:
                os._exit(0)
            return plaintext

        crypto.AESGCM.decrypt = kill_after_two_chunks
        try:
            crypto.decrypt_vault(
                encrypted, restored, provider=provider, actor="worker",
                purpose="crash-test",
            )
        except BaseException:
            os._exit(2)
        os._exit(3)

    _, status = os.waitpid(child, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    assert not restored.exists()
    assert crypto.find_staging_files(vault_dir) == []


def test_generation_is_optional_for_legacy_headers():
    header = {
        "format": "health-advisor-vault",
        "version": crypto.FORMAT_VERSION,
        "cipher": crypto.CIPHER_NAME,
        "chunk_size": crypto.DEFAULT_CHUNK_SIZE,
        "plaintext_size": 0,
        "chunk_count": 0,
        "vault_id": "legacy-vault",
        "wrap_nonce": base64.urlsafe_b64encode(b"n" * crypto.NONCE_SIZE).decode(),
        "wrapped_data_key": base64.urlsafe_b64encode(
            b"w" * (crypto.KEY_SIZE + crypto.TAG_SIZE)
        ).decode(),
        "footer_nonce": base64.urlsafe_b64encode(b"f" * crypto.NONCE_SIZE).decode(),
    }
    validated, _, _, _ = crypto._validate_header(header)
    assert "generation" not in validated


def test_envelope_generated_before_generation_field_still_decrypts(tmp_path, monkeypatch):
    """Pinned version-1 fixture: its authenticated header has no generation."""
    legacy_envelope = base64.b64decode(
        "SEFWTFRFTkMAAAExeyJjaHVua19jb3VudCI6MSwiY2h1bmtfc2l6ZSI6MTA0ODU3NiwiY2lwaGVyIjoiQUVTLTI1"
        "Ni1HQ00iLCJmb290ZXJfbm9uY2UiOiJEQXdNREF3TURBd01EQXdNIiwiZm9ybWF0IjoiaGVhbHRoLWFkdmlzb3It"
        "dmF1bHQiLCJwbGFpbnRleHRfc2l6ZSI6MTQsInZhdWx0X2lkIjoibGVnYWN5LWZpeHR1cmUiLCJ2ZXJzaW9uIjox"
        "LCJ3cmFwX25vbmNlIjoiREF3TURBd01EQXdNREF3TSIsIndyYXBwZWRfZGF0YV9rZXkiOiJRbklITTBVeWJwQXdh"
        "dXdCVjd2RXI2SWFhYXpDbDJMYVBweEoyeGZIdW0tcUJSdXppbnBaUFJ0WFpJUWxyQVVrIn0AAAAAAAAAAAAAAB4M"
        "DAwMDAwMDAwMDAwDTS0fshdCmSndXWgEyYHD7/xVs3V12R9b6P9YqjFIQVZMVEZUUgAAAEBvKEp+0W5v/kClKR12"
        "rFZtgr2UmlBTwqqSeclleH0XkKQY9QmfLI1dLtem+kMj1qECcXtotU97d9ibRDcBp92X"
    )
    _set_audit(monkeypatch, tmp_path)
    vault_dir = tmp_path / "legacy-vault"
    vault_dir.mkdir()
    encrypted = vault_dir / "legacy.enc"
    restored = vault_dir / "legacy.db"
    encrypted.write_bytes(legacy_envelope)

    crypto.decrypt_vault(
        encrypted, restored, provider=FixedProvider(b"k" * 32),
        actor="fixture", purpose="compatibility",
    )
    assert restored.read_bytes() == b"legacy-fixture"


def test_plaintext_and_chunk_limits_are_enforced(tmp_path, monkeypatch):
    # The encrypt-side check reads the module constants at call time, so it is
    # exercised against a lowered ceiling: a sparse file at the real 2 TiB
    # limit is refused with EFBIG by some CI filesystems (ubuntu-latest /tmp).
    # The header check below runs against the real constants and needs no file.
    monkeypatch.setattr(crypto, "MAX_CHUNK_COUNT", 4)
    monkeypatch.setattr(crypto, "MAX_PLAINTEXT_SIZE", 4 * crypto.DEFAULT_CHUNK_SIZE)
    source = tmp_path / "oversized.db"
    with source.open("wb") as handle:
        handle.truncate(crypto.MAX_PLAINTEXT_SIZE + 1)
    with pytest.raises(crypto.VaultCryptoError, match="exceeds vault limits"):
        crypto.encrypt_vault(
            source, tmp_path / "oversized.enc", provider=FixedProvider(b"l" * 32),
            vault_id="limit-vault", actor="worker", purpose="limit-test",
        )

    monkeypatch.undo()
    header = {
        "format": "health-advisor-vault",
        "version": crypto.FORMAT_VERSION,
        "cipher": crypto.CIPHER_NAME,
        "chunk_size": 4096,
        "plaintext_size": crypto.MAX_PLAINTEXT_SIZE,
        "chunk_count": crypto.MAX_CHUNK_COUNT + 1,
        "vault_id": "limit-vault",
        "wrap_nonce": base64.urlsafe_b64encode(b"n" * crypto.NONCE_SIZE).decode(),
        "wrapped_data_key": base64.urlsafe_b64encode(
            b"w" * (crypto.KEY_SIZE + crypto.TAG_SIZE)
        ).decode(),
        "footer_nonce": base64.urlsafe_b64encode(b"f" * crypto.NONCE_SIZE).decode(),
    }
    with pytest.raises(crypto.VaultFormatError, match="too many chunks"):
        crypto._validate_header(header)


def test_round_trip_of_real_sqlite_file(tmp_path, monkeypatch):
    audit = _set_audit(monkeypatch, tmp_path)
    vault_dir = tmp_path / "sqlite-vault"
    vault_dir.mkdir()
    source = vault_dir / "sqlite-vault.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE measurements (day TEXT, value REAL)")
        connection.executemany(
            "INSERT INTO measurements VALUES (?, ?)",
            [("2026-08-22", 42.5), ("2026-08-23", 43.0)],
        )
    original = source.read_bytes()
    encrypted = vault_dir / "sqlite-vault.enc"
    restored = vault_dir / "sqlite-vault-restored.db"
    crypto.encrypt_vault(
        source, encrypted, provider=FixedProvider(b"s" * 32), vault_id="sqlite-user",
        actor="worker", purpose="sqlite-round-trip",
    )
    crypto.decrypt_vault(
        encrypted, restored, provider=FixedProvider(b"s" * 32),
        actor="worker", purpose="sqlite-round-trip",
    )
    assert restored.read_bytes() == original
    with sqlite3.connect(restored) as connection:
        assert connection.execute("SELECT SUM(value) FROM measurements").fetchone()[0] == 85.5
    assert audit.exists()


def test_wrong_key_fails_cleanly_and_audits_attempt(tmp_path, monkeypatch):
    _, ciphertext, _, audit = _encrypt(tmp_path, monkeypatch)
    with pytest.raises(crypto.WrongMasterKeyError, match="master key is wrong"):
        crypto.decrypt_vault(
            ciphertext, tmp_path / "wrong.db", provider=FixedProvider(b"x" * 32),
            actor="worker-7", purpose="test-wrong-key",
        )
    assert json.loads(audit.read_text().splitlines()[-1])["purpose"] == "test-wrong-key"


@pytest.mark.parametrize("mutation", ["body", "header", "truncate", "reorder"])
def test_tampering_reordering_and_truncation_fail(tmp_path, monkeypatch, mutation):
    _, ciphertext, _, _ = _encrypt(tmp_path, monkeypatch)
    mutated = tmp_path / f"{mutation}.enc"
    content = bytearray(ciphertext.read_bytes())
    magic_size = len(crypto.MAGIC)
    header_size = struct.unpack(">I", content[magic_size:magic_size + 4])[0]
    body_start = magic_size + 4 + header_size
    if mutation == "header":
        content[magic_size + 4 + 10] ^= 1
    elif mutation == "body":
        content[body_start + 30] ^= 1
    elif mutation == "truncate":
        content = content[:-1]
    else:
        # Parse two complete records and exchange their bytes.  Their
        # authenticated indexes travel with them, so the decryptor rejects the
        # first unexpected index before producing an output destination.
        records = []
        cursor = body_start
        for _ in range(3):
            start = cursor
            index, size = struct.unpack(">QI", content[cursor:cursor + 12])
            cursor += 24 + size
            records.append((start, cursor))
        first = bytes(content[records[0][0]:records[0][1]])
        second = bytes(content[records[1][0]:records[1][1]])
        content[records[0][0]:records[0][1]] = second
        # The records are deliberately equal-sized in this fixture.
        content[records[1][0]:records[1][1]] = first
    mutated.write_bytes(content)
    with pytest.raises(crypto.VaultCryptoError):
        crypto.decrypt_vault(
            mutated, tmp_path / f"{mutation}.db", provider=FixedProvider(b"m" * 32),
            actor="worker-7", purpose=f"test-{mutation}",
        )


def test_master_key_is_not_in_ciphertext(tmp_path, monkeypatch):
    _, ciphertext, _, _ = _encrypt(tmp_path, monkeypatch)
    assert b"m" * 32 not in ciphertext.read_bytes()


def test_crypto_path_is_bounded_to_chunks(tmp_path, monkeypatch):
    """The implementation reads/writes one 1 MiB chunk, never the vault."""
    vault_dir = tmp_path / "bounded-vault"
    vault_dir.mkdir()
    source = vault_dir / "large.db"
    source.write_bytes(os.urandom(8 * crypto.DEFAULT_CHUNK_SIZE + 17))
    encrypted = vault_dir / "large.enc"
    restored = vault_dir / "large-restored.db"
    _set_audit(monkeypatch, tmp_path)
    tracemalloc.start()
    crypto.encrypt_vault(
        source, encrypted, provider=FixedProvider(b"b" * 32), vault_id="bounded-user",
        actor="worker", purpose="bounded-test",
    )
    crypto.decrypt_vault(
        encrypted, restored, provider=FixedProvider(b"b" * 32),
        actor="worker", purpose="bounded-test",
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert restored.read_bytes() == source.read_bytes()
    # This leaves room for Python/cryptography overhead while being far below
    # the 8 MiB source. The implementation's explicit read size is the primary
    # guarantee; this test is a regression check against whole-file reads.
    assert peak < 4 * crypto.DEFAULT_CHUNK_SIZE


def test_env_provider_accepts_hex_and_base64(monkeypatch):
    key = bytes(range(32))
    monkeypatch.setenv("TEST_VAULT_KEY", key.hex())
    assert crypto.EnvKeyProvider("TEST_VAULT_KEY").get_master_key() == key
    monkeypatch.setenv("TEST_VAULT_KEY", base64.urlsafe_b64encode(key).decode())
    assert crypto.EnvKeyProvider("TEST_VAULT_KEY").get_master_key() == key


def test_cli_inspect_does_not_need_master_key(tmp_path, monkeypatch):
    source, ciphertext, _, _ = _encrypt(tmp_path, monkeypatch)
    environment = os.environ.copy()
    environment.pop(crypto.MASTER_KEY_ENV, None)
    result = subprocess.run(
        [sys.executable, "scripts/vault_crypt.py", "inspect", str(ciphertext)],
        capture_output=True, text=True, check=True, env=environment,
    )
    inspected = json.loads(result.stdout)
    assert inspected["vault_id"] == "user-123"
    assert source.exists()


def test_cli_encrypt_and_decrypt_with_env_provider(tmp_path):
    vault_dir = tmp_path / "cli-vault"
    vault_dir.mkdir()
    source = vault_dir / "source.db"
    encrypted = vault_dir / "source.enc"
    restored = vault_dir / "restored.db"
    source.write_bytes(b"SQLite format 3\000" + os.urandom(7000))
    audit = tmp_path / "cli-audit" / "vault-unwrapping.jsonl"
    environment = os.environ.copy()
    environment[crypto.MASTER_KEY_ENV] = base64.b64encode(b"c" * 32).decode()
    environment[crypto.AUDIT_LOG_ENV] = str(audit)
    subprocess.run(
        [sys.executable, "scripts/vault_crypt.py", "encrypt", str(source), str(encrypted),
         "--vault-id", "cli-user", "--actor", "cli-worker", "--purpose", "cli-test"],
        check=True, env=environment, capture_output=True, text=True,
    )
    subprocess.run(
        [sys.executable, "scripts/vault_crypt.py", "decrypt", str(encrypted), str(restored),
         "--actor", "cli-worker", "--purpose", "cli-test"],
        check=True, env=environment, capture_output=True, text=True,
    )
    assert restored.read_bytes() == source.read_bytes()
    assert json.loads(audit.read_text().splitlines()[-2])["actor"] == "cli-worker"


def test_a_missing_chunk_says_truncation_rather_than_reordering(tmp_path, monkeypatch):
    """A dropped chunk puts the footer where a chunk record belongs, and its
    magic then unpacks as an enormous index. Reporting that as "out of order"
    sends a reader hunting for a reordering bug that is really a truncation.
    """
    _set_audit(monkeypatch, tmp_path)
    provider = FixedProvider(os.urandom(32))
    # The vault lives in its own directory: the audit log may not sit inside it.
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    source = vault_dir / "source.bin"
    _write_source(source, crypto.DEFAULT_CHUNK_SIZE * 3 + 128)
    encrypted = vault_dir / "source.enc"
    crypto.encrypt_vault(source, encrypted, provider=provider, vault_id="v1",
                         actor="test", purpose="unit")

    blob = encrypted.read_bytes()
    header_len = struct.unpack(">I", blob[8:12])[0]
    pos = 12 + header_len
    records = []
    for _ in range(4):
        _index, size = struct.unpack(">QI", blob[pos:pos + 12])
        record = blob[pos:pos + 12 + crypto.NONCE_SIZE + size]
        records.append(record)
        pos += len(record)
    truncated = vault_dir / "truncated.enc"
    truncated.write_bytes(blob[:12 + header_len] + b"".join(records[:-1]) + blob[pos:])

    with pytest.raises(crypto.TamperError) as excinfo:
        crypto.decrypt_vault(truncated, vault_dir / "out.bin", provider=provider,
                             actor="test", purpose="unit")
    message = str(excinfo.value)
    assert "missing chunks" in message
    assert "out of order" not in message


def test_directory_fsync_failure_does_not_report_failed_install(tmp_path, monkeypatch):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    source = vault_dir / "source.db"
    destination = vault_dir / "source.enc"
    _write_source(source, 123)
    _set_audit(monkeypatch, tmp_path)
    provider = FixedProvider(b"d" * 32)

    real_fsync = crypto.os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(crypto.os, "fsync", fail_directory_fsync)
    try:
        crypto.encrypt_vault(
            source, destination, provider=provider, vault_id="fsync-user",
            actor="worker", purpose="fsync-test",
        )
    except OSError as exc:
        pytest.fail(f"installed destination was reported as failed: {exc}")
    assert destination.exists(), "destination must remain installed after directory fsync failure"


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (None, crypto.DurableReplaceStatus.DURABLE),
        ("fsync", crypto.DurableReplaceStatus.DIRECTORY_SYNC_FAILED),
        ("open", crypto.DurableReplaceStatus.DIRECTORY_SYNC_UNSUPPORTED),
    ],
)
def test_durable_replace_classifies_directory_sync_outcomes(
    tmp_path, monkeypatch, failure, expected,
):
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.write_bytes(b"installed")

    if failure == "fsync":
        real_fsync = crypto.os.fsync

        def fail_directory_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EIO, "directory fsync failed")
            return real_fsync(fd)

        monkeypatch.setattr(crypto.os, "fsync", fail_directory_fsync)
    elif failure == "open":
        real_open = crypto.os.open

        def reject_directory_open(path, flags, *args):
            if Path(path) == destination.parent:
                raise OSError(errno.ENOTSUP, "directory open unsupported")
            return real_open(path, flags, *args)

        monkeypatch.setattr(crypto.os, "open", reject_directory_open)

    assert crypto._durable_replace(staging, destination) is expected
    assert destination.read_bytes() == b"installed"


@pytest.mark.parametrize("status", list(crypto.DurableReplaceStatus))
def test_encrypt_consumes_each_durable_replace_status(tmp_path, monkeypatch, status):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    source = vault_dir / "source.db"
    destination = vault_dir / "source.enc"
    _write_source(source, 123)
    audit = _set_audit(monkeypatch, tmp_path)
    real_replace = crypto._durable_replace

    def replace_with_status(staging, destination):
        real_replace(staging, destination)
        return status

    monkeypatch.setattr(crypto, "_durable_replace", replace_with_status)
    crypto.encrypt_vault(
        source, destination, provider=FixedProvider(b"s" * 32), vault_id="status-user",
        actor="worker", purpose="status-test",
    )

    event = json.loads(audit.read_text().splitlines()[-1])
    assert event["event"] == "vault_replace"
    assert event["durability"] == status.value
    assert destination.exists()


def test_decrypt_consumes_durable_replace_status(tmp_path, monkeypatch):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    source = vault_dir / "source.db"
    encrypted = vault_dir / "source.enc"
    restored = vault_dir / "restored.db"
    original = _write_source(source, 123)
    _set_audit(monkeypatch, tmp_path)
    provider = FixedProvider(b"r" * 32)
    crypto.encrypt_vault(
        source, encrypted, provider=provider, vault_id="decrypt-status-user",
        actor="worker", purpose="status-test",
    )
    status = crypto.DurableReplaceStatus.DIRECTORY_SYNC_UNSUPPORTED
    real_replace = crypto._durable_replace

    def replace_with_status(staging, destination):
        real_replace(staging, destination)
        return status

    monkeypatch.setattr(crypto, "_durable_replace", replace_with_status)
    crypto.decrypt_vault(
        encrypted, restored, provider=provider, actor="worker", purpose="status-test",
    )

    assert restored.read_bytes() == original
    event = json.loads(crypto.AUDIT_LOG_PATH.read_text().splitlines()[-1])
    assert event["event"] == "vault_replace"
    assert event["durability"] == status.value


def test_installed_vault_survives_replace_audit_failure(tmp_path, monkeypatch):
    """Post-install audit failure cannot turn a committed install into an error."""
    audit = tmp_path / "audit" / "vault-unwrapping.jsonl"
    monkeypatch.setattr(crypto, "AUDIT_LOG_PATH", audit)

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    source = vault_dir / "plain.db"
    source.write_bytes(bytes((i * 17 + 3) % 256 for i in range(5000)))
    destination = vault_dir / "plain.db.enc"

    real_replace = crypto._durable_replace

    def fake_replace(staging, destination):
        real_replace(staging, destination)
        return crypto.DurableReplaceStatus.DIRECTORY_SYNC_FAILED

    monkeypatch.setattr(crypto, "_durable_replace", fake_replace)

    real_append = crypto._append_audit_event

    def fake_append(*, event, source):
        if event.get("event") == "vault_replace":
            raise crypto.AuditError("simulated: audit log unwritable at install time")
        return real_append(event=event, source=source)

    monkeypatch.setattr(crypto, "_append_audit_event", fake_append)

    raised = None
    with pytest.warns(RuntimeWarning, match="could not be audited"):
        try:
            crypto.encrypt_vault(
                source, destination, provider=FixedProvider(b"m" * 32),
                vault_id="user-123", actor="worker-7", purpose="analytics-checkout",
            )
        except Exception as exc:  # pragma: no cover - assertion below reports it
            raised = exc

    print("\n--- PROBE RESULT ---")
    print("destination exists on disk :", destination.exists())
    print("destination size           :", destination.stat().st_size if destination.exists() else None)
    print("call raised                :", type(raised).__name__ if raised else "no")
    print("--------------------")
    assert raised is None
    assert destination.exists()
    assert destination.stat().st_size > 0


def test_staging_files_have_identifiable_names_and_restrictive_mode(tmp_path):
    destination = tmp_path / "vault.enc"
    staging = crypto._staging_path(destination)
    try:
        assert staging.name.startswith(f"{crypto.STAGING_PREFIX}{destination.name}-")
        assert staging.name.endswith(crypto.STAGING_SUFFIX)
        assert stat.S_IMODE(staging.stat().st_mode) == 0o600
        assert crypto.find_staging_files(tmp_path) == [staging]
    finally:
        assert crypto.remove_staging_files(tmp_path) == [staging]


def test_audit_symlink_is_refused(tmp_path, monkeypatch):
    _, ciphertext, _, _ = _encrypt(tmp_path, monkeypatch)
    target = tmp_path / "real-audit.jsonl"
    target.touch()
    audit_link = tmp_path / "audit-link.jsonl"
    audit_link.symlink_to(target)
    monkeypatch.setattr(crypto, "AUDIT_LOG_PATH", audit_link)

    with pytest.raises(crypto.AuditError) as excinfo:
        crypto.decrypt_vault(
            ciphertext, tmp_path / "restored.db", provider=FixedProvider(b"m" * 32),
            actor="worker", purpose="symlink-audit-test",
        )
    assert str(excinfo.value) == "unwrap audit log must not be a symlink"


def test_audit_hard_link_is_refused(tmp_path, monkeypatch):
    _, ciphertext, _, _ = _encrypt(tmp_path, monkeypatch)
    audit_link = tmp_path / "hard-link-audit.jsonl"
    os.link(ciphertext, audit_link)
    monkeypatch.setattr(crypto, "AUDIT_LOG_PATH", audit_link)

    with pytest.raises(crypto.AuditError) as excinfo:
        crypto.decrypt_vault(
            ciphertext, tmp_path / "restored.db", provider=FixedProvider(b"m" * 32),
            actor="worker", purpose="hard-link-audit-test",
        )
    assert str(excinfo.value) == "unwrap audit log must have exactly one hard link"


def test_encrypt_refuses_live_sqlite_sidecar(tmp_path):
    source = tmp_path / "health.db"
    destination = tmp_path / "health.enc"
    _write_source(source, 123)
    sidecar = source.with_name(source.name + "-wal")
    sidecar.write_bytes(b"live wal")

    with pytest.raises(crypto.VaultCryptoError) as excinfo:
        crypto.encrypt_vault(
            source, destination, provider=FixedProvider(b"w" * 32),
            vault_id="sidecar-user", actor="worker", purpose="sidecar-test",
        )
    assert str(excinfo.value) == (
        "source has a live SQLite sidecar: health.db-wal; "
        "close the database before encrypting"
    )


@pytest.mark.parametrize(
    ("label", "size"),
    [("empty", 0), ("sub-chunk", crypto.DEFAULT_CHUNK_SIZE - 1),
     ("exact-chunk", crypto.DEFAULT_CHUNK_SIZE)],
)
def test_round_trip_edge_sizes_are_byte_identical(tmp_path, monkeypatch, label, size):
    vault_dir = tmp_path / label
    vault_dir.mkdir()
    source = vault_dir / "source.db"
    encrypted = vault_dir / "source.enc"
    restored = vault_dir / "restored.db"
    original = _write_source(source, size)
    _set_audit(monkeypatch, tmp_path)
    provider = FixedProvider(b"e" * 32)

    crypto.encrypt_vault(
        source, encrypted, provider=provider, vault_id=f"{label}-user",
        actor="worker", purpose="edge-size-test",
    )
    crypto.decrypt_vault(
        encrypted, restored, provider=FixedProvider(b"e" * 32),
        actor="worker", purpose="edge-size-test",
    )
    assert restored.read_bytes() == original


def test_keychain_provider_uses_absolute_security_path(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=base64.b64encode(b"k" * 32).decode() + "\n")

    monkeypatch.setattr(crypto.subprocess, "run", fake_run)
    assert crypto.KeychainKeyProvider().get_master_key() == b"k" * 32
    assert calls[0][0][0] == "/usr/bin/security"


def test_keychain_provider_fails_cleanly_when_absolute_security_is_missing(monkeypatch):
    def missing_security(*args, **kwargs):
        raise FileNotFoundError(errno.ENOENT, "No such file or directory")

    monkeypatch.setattr(crypto.subprocess, "run", missing_security)
    with pytest.raises(crypto.VaultCryptoError) as excinfo:
        crypto.KeychainKeyProvider().get_master_key()
    assert str(excinfo.value) == "macOS security CLI is not available"
