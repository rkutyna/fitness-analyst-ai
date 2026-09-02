from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from health_advisor import analyst_runner
from health_advisor import analyst_sandbox as sb


def test_bidirectional_query_bridge_forwards_bytes_and_exits_on_eof():
    child_peer, accepted_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    parent_peer, parent_half = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    bridge = threading.Thread(
        target=sb.bridge_bidirectional, args=(accepted_child, parent_half)
    )
    bridge.start()
    try:
        child_peer.sendall(b"request-from-child")
        assert parent_peer.recv(1024) == b"request-from-child"
        parent_peer.sendall(b"response-from-parent")
        assert child_peer.recv(1024) == b"response-from-parent"

        child_peer.shutdown(socket.SHUT_WR)
        assert parent_peer.recv(1) == b""
        parent_peer.shutdown(socket.SHUT_WR)
        bridge.join(timeout=2)
        assert not bridge.is_alive()
    finally:
        for sock in (child_peer, accepted_child, parent_peer, parent_half):
            sock.close()


def _user_systemd_available() -> bool:
    if sys.platform != "linux":
        return False
    if not shutil.which("bwrap") or not shutil.which("systemd-run"):
        return False
    try:
        probe = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode in (0, 1, 2, 3)


LINUX_TRANSIENT = pytest.mark.skipif(
    not _user_systemd_available(),
    reason="needs Linux bwrap and a responsive user systemd manager",
)


def _bare_executor() -> sb.TransientUnitExecutor:
    executor = object.__new__(sb.TransientUnitExecutor)
    executor.BWRAP = "/usr/bin/bwrap"
    executor.SYSTEMD_RUN = "systemd-run"
    executor.SYSTEMCTL = "systemctl"
    executor._python = sys.executable
    executor._venv_dir = sys.prefix
    executor._pyenv_prefix = sys.base_prefix
    executor._pkg_dir = None
    return executor


def test_failed_transient_unit_exposes_named_systemd_result(monkeypatch):
    executor = _bare_executor()
    def fake_run(argv, **kwargs):
        assert argv == [
            "systemctl", "--user", "show", "analyst-test.service",
            "-p", "Result", "--value",
        ]
        return SimpleNamespace(returncode=0, stdout="timeout\n", stderr="")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    assert executor._read_unit_result("analyst-test.service") == "timeout"
    with pytest.raises(sb.TransientUnitError, match=r"Result=timeout"):
        executor._require_ordinary_unit_result("timeout")


@LINUX_TRANSIENT
def _user_bus_available() -> bool:
    """Can `systemd-run --user` actually reach a user session bus?

    These tests drive a transient user unit, which needs a running systemd
    user session and its D-Bus socket. A container (CI) has neither, and
    systemd-run fails with "Failed to connect to bus: No medium found". That
    is the environment being unable to run the test — not a result — so it
    skips rather than failing, and rather than being quietly deleted.
    """
    if not shutil.which("systemd-run"):
        return False
    try:
        return subprocess.run(
            ["systemd-run", "--user", "--quiet", "--collect", "--wait",
             "--", "true"],
            capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


pytestmark = pytest.mark.skipif(
    not _user_bus_available(),
    reason="needs a systemd user session bus (systemd-run --user); "
           "unavailable in containers — run on a Linux desktop session")


def test_transient_query_round_trip_is_parent_observed(tmp_path):
    vault = tmp_path / "vault.db"
    import sqlite3

    connection = sqlite3.connect(vault)
    connection.execute("CREATE TABLE probe (value INTEGER)")
    connection.execute("INSERT INTO probe VALUES (42)")
    connection.commit()
    connection.close()

    executor = sb.TransientUnitExecutor()
    result = analyst_runner.run_analyst_code(
        "row = conn.execute('SELECT value FROM probe').fetchone()\n"
        "emit('probe', ['value'], ['count'], [[row[0]]])\n",
        str(vault), str(tmp_path / "caller-run"), executor,
    )
    assert result.tables[0]["rows"] == ((42,),)
    assert result.ledger["parent_observed"] is True
    assert result.ledger["rows_read"] == 1


@LINUX_TRANSIENT
def test_transient_child_cannot_reach_repo_or_vault(tmp_path):
    package = tmp_path / "production-repo"
    (package / "data").mkdir(parents=True)
    (package / "systemd").mkdir()
    vault_like = package / "data" / "health.db"
    secret_like = package / "systemd" / "receiver.env"
    vault_like.write_text("vault-shaped", encoding="utf-8")
    secret_like.write_text("secret-shaped", encoding="utf-8")

    vault = tmp_path / "real-vault.db"
    vault.write_bytes(b"SQLite format 3\x00")
    parent, child = socket.socketpair()
    query_fd = child.detach()
    code = f"""
import json
facts = {{}}
for name, path in (("vault", {str(vault_like)!r}),
                   ("secret", {str(secret_like)!r}),
                   ("real_vault", {str(vault)!r})):
    try:
        open(path, "rb").read()
        facts[name] = "unsafe"
    except Exception:
        facts[name] = "blocked"
emit("isolation", ["facts"], ["text"], [[json.dumps(facts, sort_keys=True)]])
"""
    try:
        result = sb.TransientUnitExecutor(pkg_dir=str(package)).run_with_named_query_channel(
            code, str(tmp_path / "caller-run"), query_fd,
            runner_source=analyst_runner._named_socket_runner_source("code.py"),
            limits=sb.RunLimits(wall_clock_s=10),
        )
    finally:
        parent.close()
    assert result.returncode == 0, result.stderr
    facts = json.loads(result.fd3_as_json()["tables"][0]["rows"][0][0])
    assert facts == {"real_vault": "blocked", "secret": "blocked", "vault": "blocked"}
    assert not result.run_dir.exists()


@LINUX_TRANSIENT
def test_transient_runs_do_not_leave_owned_directories(tmp_path):
    executor = sb.TransientUnitExecutor()
    for index in range(2):
        parent, child = socket.socketpair()
        query_fd = child.detach()
        try:
            result = executor.run_with_named_query_channel(
                "pass", str(tmp_path / f"caller-{index}"), query_fd,
                runner_source=analyst_runner._named_socket_runner_source("code.py"),
                limits=sb.RunLimits(wall_clock_s=10),
            )
        finally:
            parent.close()
        assert result.returncode == 0, result.stderr
        assert not result.run_dir.exists()
