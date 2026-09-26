"""T5 piece 9, E2: the per-device D23 secret store (consumer #426).

Every mutation named in a test's docstring was manually applied against
``health_advisor/device_secrets.py`` and observed to turn that test red
before this file was finalised (device 20260926).
"""
from __future__ import annotations

import json
import stat
import threading

import pytest

from health_advisor import device_secrets


# ---------------------------------------------------------------- mode


@pytest.mark.parametrize("value", [None, ""])
def test_unset_or_empty_mode_is_instance(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("HA_DEVICE_SECRET_MODE", raising=False)
    else:
        monkeypatch.setenv("HA_DEVICE_SECRET_MODE", value)
    assert device_secrets.device_secret_mode() == "instance"


def test_per_device_mode_is_recognised(monkeypatch):
    monkeypatch.setenv("HA_DEVICE_SECRET_MODE", "per_device")
    assert device_secrets.device_secret_mode() == "per_device"


@pytest.mark.parametrize("value", ["PER_DEVICE", " per_device", "on", "both"])
def test_an_invalid_mode_refuses(monkeypatch, value):
    """Mutation: accept any string (drop the ``mode in MODES`` check) --
    then ``"both"`` would return truthy nonsense instead of raising."""
    monkeypatch.setenv("HA_DEVICE_SECRET_MODE", value)
    with pytest.raises(RuntimeError, match="HA_DEVICE_SECRET_MODE"):
        device_secrets.device_secret_mode()


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def store_path(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    return d / "device-secrets.json"


@pytest.fixture
def store(store_path):
    return device_secrets.DeviceSecretStore(store_path)


# ---------------------------------------------------------------- issue


def test_issue_writes_43_chars_at_0600(store, store_path):
    """Mutation: write the payload with a plain ``open(path, "w")`` instead
    of ``device_auth.atomic_write_json`` -- the file then inherits the
    process umask (644 under the common 022) instead of the 600
    ``tempfile.mkstemp`` + explicit ``chmod`` combination guarantees."""
    record = store.issue("kid-a")
    assert len(record.secret) == 43
    mode = stat.S_IMODE(store_path.stat().st_mode)
    assert mode == 0o600
    on_disk = json.loads(store_path.read_text())
    assert on_disk["secrets"][0]["secret"] == record.secret


def test_second_issue_rotates(store):
    """Mutation: have ``issue`` return the existing record for a known kid
    instead of always minting a fresh one -- the two secrets would then be
    equal and a leaked old secret would survive a deliberate re-pair."""
    first = store.issue("kid-a")
    second = store.issue("kid-a")
    assert first.secret != second.secret
    assert store.get("kid-a").secret == second.secret


def test_get_sees_an_external_write(store_path):
    """Mutation: cache the parsed store forever after the first read (e.g.
    only ever check the stat signature when it is ``None``) -- the second
    ``get`` would then still return ``None`` for ``kid-b``."""
    store = device_secrets.DeviceSecretStore(store_path)
    other = device_secrets.DeviceSecretStore(store_path)
    store.issue("kid-a")  # gives `store` a real, non-None cached signature
    assert store.get("kid-b") is None
    external = other.issue("kid-b")
    assert store.get("kid-b") == external


def test_kids_never_returns_secrets(store):
    store.issue("kid-a")
    store.issue("kid-b")
    kids = store.kids()
    assert set(kids) == {"kid-a", "kid-b"}
    assert all(isinstance(k, str) and "secret" not in k for k in kids)


# ---------------------------------------------------------------- parsing


def test_wrong_version_raises(store_path):
    store_path.write_text(json.dumps({"v": 2, "secrets": []}))
    store = device_secrets.DeviceSecretStore(store_path)
    with pytest.raises(ValueError):
        store.get("kid-a")


def test_unknown_field_raises(store_path):
    store_path.write_text(json.dumps({
        "v": 1,
        "secrets": [{"kid": "kid-a", "secret": "x" * 43,
                    "issued_at": "2026-01-01T00:00:00Z",
                    "unexpected_field": "surprise"}],
    }))
    store = device_secrets.DeviceSecretStore(store_path)
    with pytest.raises((ValueError, TypeError)):
        store.get("kid-a")


def test_duplicate_kid_raises(store_path):
    record = {"kid": "kid-a", "secret": "x" * 43,
              "issued_at": "2026-01-01T00:00:00Z"}
    store_path.write_text(json.dumps({"v": 1, "secrets": [record, record]}))
    store = device_secrets.DeviceSecretStore(store_path)
    with pytest.raises(ValueError, match="duplicate kid"):
        store.get("kid-a")


# ------------------------------------------------------------ concurrency


def test_concurrent_issue_for_two_kids_keeps_both(store):
    """Two threads mint secrets for two different kids at once. The
    read-modify-write must happen entirely inside ``exclusive_lock`` --
    reading the file only after the lock is held -- or one writer's snapshot
    (taken before the other's write lands) can overwrite the other's kid
    clean out of the file. ``_test_race_delay`` widens the window so this is
    not a matter of luck.

    Mutation: move the ``self._read_file()`` call in ``issue`` to before
    ``with self._exclusive():`` (read outside the lock). Observed: one of
    the two kids goes missing from the final file, 5/5 runs.
    """
    store._test_race_delay = 0.05
    results: dict[str, device_secrets.DeviceSecret] = {}
    lock = threading.Lock()

    def worker(kid: str):
        record = store.issue(kid)
        with lock:
            results[kid] = record

    threads = [threading.Thread(target=worker, args=(kid,))
              for kid in ("kid-a", "kid-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert set(results) == {"kid-a", "kid-b"}
    assert store.get("kid-a") == results["kid-a"]
    assert store.get("kid-b") == results["kid-b"]
    assert set(store.kids()) == {"kid-a", "kid-b"}


# ------------------------------------------------------------------- forget


def test_forget_removes_one_kid_and_reports_whether_it_was_present(store):
    store.issue("kid-a")
    store.issue("kid-b")
    assert store.forget("kid-a") is True
    assert store.get("kid-a") is None
    assert store.get("kid-b") is not None
    assert store.forget("kid-a") is False


# ---------------------------------------------------------------------- CLI


def test_cli_list_never_prints_a_secret(store, store_path, capsys):
    """Mutation: print ``dataclasses.asdict(record)`` (or the record itself)
    in the ``list`` command instead of just kid + issued_at -- the raw
    secret string would then appear in captured stdout."""
    record = store.issue("kid-a")
    rc = device_secrets.main(["--store", str(store_path), "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "kid-a" in out
    assert record.secret not in out


def test_cli_forget_reports_success_and_failure(store, store_path, capsys):
    store.issue("kid-a")
    assert device_secrets.main(["--store", str(store_path), "forget", "kid-a"]) == 0
    assert "forgot" in capsys.readouterr().out
    assert device_secrets.main(["--store", str(store_path), "forget", "kid-a"]) == 1
    assert "nothing" in capsys.readouterr().err


# ------------------------------------------------------------- store_from_env


def test_store_from_env_requires_the_path(monkeypatch):
    monkeypatch.delenv("HA_DEVICE_SECRETS_FILE", raising=False)
    with pytest.raises(RuntimeError, match="HA_DEVICE_SECRETS_FILE"):
        device_secrets.store_from_env()


def test_store_from_env_refuses_an_unwritable_directory(monkeypatch):
    monkeypatch.setenv("HA_DEVICE_SECRETS_FILE",
                       "/no/such/directory/device-secrets.json")
    with pytest.raises(RuntimeError, match="HA_DEVICE_SECRETS_FILE"):
        device_secrets.store_from_env()


def test_store_from_env_refuses_a_0644_file(monkeypatch, store_path):
    """Mutation: drop the ``mode & 0o077`` startup check -- a group/other
    readable file full of live D23 secrets would then start clean."""
    store_path.write_text(json.dumps({"v": 1, "secrets": []}))
    store_path.chmod(0o644)
    monkeypatch.setenv("HA_DEVICE_SECRETS_FILE", str(store_path))
    with pytest.raises(RuntimeError, match="0o077|mode 644|chmod 600"):
        device_secrets.store_from_env()


def test_store_from_env_refuses_an_unreadable_file(monkeypatch, store_path):
    store_path.write_text("{not json")
    store_path.chmod(0o600)
    monkeypatch.setenv("HA_DEVICE_SECRETS_FILE", str(store_path))
    with pytest.raises(RuntimeError, match="HA_DEVICE_SECRETS_FILE"):
        device_secrets.store_from_env()


def test_store_from_env_succeeds_on_a_fresh_600_directory(monkeypatch, store_path):
    monkeypatch.setenv("HA_DEVICE_SECRETS_FILE", str(store_path))
    got = device_secrets.store_from_env()
    assert isinstance(got, device_secrets.DeviceSecretStore)
    assert got.kids() == []
