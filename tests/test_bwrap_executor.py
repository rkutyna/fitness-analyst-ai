from __future__ import annotations

import inspect
import json
import os
import shutil
import sqlite3
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from health_advisor import analyst_sandbox as sb
from health_advisor import analyst_envelope
from health_advisor import analyst_runner
from health_advisor import receiver


def _argv_executor(*, venv: str, pkg: str, pyenv: str) -> sb.BwrapExecutor:
    executor = object.__new__(sb.BwrapExecutor)
    executor.BWRAP = "/usr/bin/bwrap"
    executor._python = "/opt/python/bin/python3.11"
    executor._venv_dir = venv
    executor._pyenv_prefix = pyenv
    executor._pkg_dir = pkg
    executor._real_home = "/home/tester"
    return executor


def test_bwrap_argv_has_usr_loader_links_and_no_vault_bind():
    executor = _argv_executor(
        venv="/repo/.venv",
        pkg="/repo",
        pyenv="/opt/python",
    )
    run_dir = Path("/tmp/runs/run with spaces")
    work_dir = run_dir / "work"
    argv = executor._build_argv(
        run_dir_real=run_dir,
        work_dir_real=work_dir,
        runner_path=run_dir / "runner.py",
    )

    assert argv == [
        "/usr/bin/bwrap",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
        "--ro-bind", "/opt/python", "/opt/python",
        "--ro-bind", "/repo", "/repo",
        "--ro-bind", str(run_dir), str(run_dir),
        "--bind", str(work_dir), str(work_dir),
        "--proc", "/proc",
        "--dev", "/dev",
        "--chdir", str(work_dir),
        "--unshare-all", "--die-with-parent", "--new-session",
        "/opt/python/bin/python3.11", "-I", str(run_dir / "runner.py"),
    ]
    assert "/tmp/vault with spaces.db" not in argv
    assert argv[argv.index("--bind") + 1:argv.index("--bind") + 3] == [
        str(work_dir), str(work_dir)]


def test_bwrap_argv_adds_a_separate_venv_bind_only_when_needed():
    executor = _argv_executor(
        venv="/outside/.venv",
        pkg="/repo",
        pyenv="/opt/python",
    )
    argv = executor._build_argv(
        run_dir_real=Path("/tmp/run"),
        work_dir_real=Path("/tmp/run/work"),
        runner_path=Path("/tmp/run/runner.py"),
    )
    assert ["--ro-bind", "/outside/.venv", "/outside/.venv"] == (
        argv[argv.index("--ro-bind", argv.index("/repo") + 1):
             argv.index("--ro-bind", argv.index("/repo") + 1) + 3]
    )


def test_bwrap_constructor_resolves_paths_and_rejects_missing_binary(monkeypatch):
    monkeypatch.setattr(sb.BwrapExecutor, "BWRAP", "/definitely/not/bwrap")
    with pytest.raises(RuntimeError, match=r"BwrapExecutor.*Linux-only"):
        sb.BwrapExecutor(python_executable="/tmp/python", pkg_dir="/tmp/pkg")


def test_bwrap_constructor_realpaths_symlinked_python_and_package(tmp_path, monkeypatch):
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("", encoding="utf-8")
    python_real = tmp_path / "real-python"
    python_real.write_text("", encoding="utf-8")
    python_link = tmp_path / "python-link"
    python_link.symlink_to(python_real)
    pkg_real = tmp_path / "real-package"
    pkg_real.mkdir()
    pkg_link = tmp_path / "package-link"
    pkg_link.symlink_to(pkg_real, target_is_directory=True)
    monkeypatch.setattr(sb.BwrapExecutor, "BWRAP", str(bwrap))
    monkeypatch.setattr(sb.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sb.sys, "base_prefix", str(tmp_path / "pyenv"))

    executor = sb.BwrapExecutor(
        python_executable=str(python_link), pkg_dir=str(pkg_link)
    )
    assert executor._python == str(python_real)
    assert executor._pkg_dir == str(pkg_real)
    assert executor._venv_dir == str(tmp_path / "venv")
    assert executor._pyenv_prefix == str(tmp_path / "pyenv")


def test_bwrap_run_passes_constructed_command_and_honors_run_limits(
        tmp_path, monkeypatch):
    executor = _argv_executor(
        venv="/repo/.venv", pkg="/repo", pyenv="/opt/python"
    )
    vault_path = tmp_path / "vault.db"
    vault_path.write_bytes(b"")
    popen_call = {}
    streams = []

    class FakeProcess:
        pid = 4242
        returncode = -9

        def __init__(self):
            self.stdout = open(os.devnull, "rb")
            self.stderr = open(os.devnull, "rb")
            streams.extend((self.stdout, self.stderr))

        def wait(self, timeout):
            return None

    def fake_popen(argv, **kwargs):
        popen_call["argv"] = argv
        popen_call["kwargs"] = kwargs
        return FakeProcess()

    class FakeSelector:
        def __init__(self):
            self.entries = {}

        def register(self, fileobj, events, data):
            self.entries[fileobj] = SimpleNamespace(fileobj=fileobj, data=data)

        def unregister(self, fileobj):
            self.entries.pop(fileobj, None)

        def get_map(self):
            return self.entries

        def select(self, timeout):
            keys = list(self.entries.values())
            return [(key, None) for key in keys]

    killed = []
    monkeypatch.setattr(sb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sb.selectors, "DefaultSelector", FakeSelector)
    monkeypatch.setattr(sb.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    limits = sb.RunLimits(wall_clock_s=0.0, max_fd3_bytes=4)

    result = executor.run("pass", str(vault_path), str(tmp_path / "run"), limits)

    assert result.timed_out is True
    assert result.killed_group is True
    assert killed == [(4242, sb.signal.SIGKILL)]
    assert popen_call["argv"] == executor._build_argv(
        run_dir_real=result.run_dir,
        work_dir_real=result.run_dir / "work",
        runner_path=result.run_dir / "runner.py",
    )
    assert popen_call["kwargs"]["pass_fds"] == (sb.FD_OUT,)
    assert popen_call["kwargs"]["start_new_session"] is True
    assert popen_call["kwargs"]["cwd"] == str(result.run_dir / "work")
    assert popen_call["kwargs"]["env"]["TMPDIR"] == str(result.run_dir / "work")
    runner_source = result.run_dir.joinpath("runner.py").read_text()
    assert "conn = None" in runner_source
    assert str(vault_path) not in runner_source


def test_bwrap_query_argv_has_no_pkg_bind_and_runs_from_run_dir(tmp_path):
    pkg_dir = tmp_path / "production-repo"
    executor = _argv_executor(
        venv="/repo/.venv", pkg=str(pkg_dir), pyenv="/opt/python"
    )
    run_dir = tmp_path / "run"
    work_dir = run_dir / "work"
    argv = executor._build_query_argv(
        run_dir_real=run_dir,
        work_dir_real=work_dir,
        runner_path=run_dir / "runner.py",
    )

    # Exact argv captured before issue #37; the production query namespace is
    # deliberately unchanged by the plain-path fix.
    assert argv == [
        "/usr/bin/bwrap",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
        "--ro-bind", "/opt/python", "/opt/python",
        "--ro-bind", "/repo/.venv", "/repo/.venv",
        "--ro-bind", str(run_dir), str(run_dir),
        "--bind", str(work_dir), str(work_dir),
        "--proc", "/proc",
        "--dev", "/dev",
        "--chdir", str(run_dir),
        "--unshare-all", "--die-with-parent", "--new-session",
        "/opt/python/bin/python3.11", "-I", str(run_dir / "runner.py"),
    ]
    assert str(pkg_dir) not in argv
    assert ["--ro-bind", "/repo/.venv", "/repo/.venv"] == (
        argv[argv.index("--ro-bind", argv.index("/opt/python") + 1):
             argv.index("--ro-bind", argv.index("/opt/python") + 1) + 3]
    )
    assert argv[argv.index("--chdir") + 1] == str(run_dir)
    assert argv[-1] == str(run_dir / "runner.py")


def test_bwrap_query_channel_writes_runner_verbatim_and_passes_high_fds(
        tmp_path, monkeypatch):
    executor = _argv_executor(
        venv="/repo/.venv", pkg="/production-repo", pyenv="/opt/python"
    )
    parent, child = socket.socketpair()
    query_fd = child.detach()
    popen_call = {}
    streams = []

    class FakeProcess:
        pid = 5252
        returncode = 0

        def __init__(self):
            self.stdout = open(os.devnull, "rb")
            self.stderr = open(os.devnull, "rb")
            streams.extend((self.stdout, self.stderr))

        def wait(self, timeout):
            return None

    def fake_popen(argv, **kwargs):
        popen_call["argv"] = argv
        popen_call["kwargs"] = kwargs
        return FakeProcess()

    class FakeSelector:
        def __init__(self):
            self.entries = {}

        def register(self, fileobj, events, data):
            self.entries[fileobj] = SimpleNamespace(fileobj=fileobj, data=data)

        def unregister(self, fileobj):
            self.entries.pop(fileobj, None)

        def get_map(self):
            return self.entries

        def select(self, timeout):
            return [(key, None) for key in list(self.entries.values())]

    monkeypatch.setattr(sb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sb.selectors, "DefaultSelector", FakeSelector)
    runner_source = "runner source must remain byte-for-byte unchanged"
    try:
        result = executor.run_with_query_channel(
            "code source",
            str(tmp_path / "run"),
            query_fd,
            runner_source=runner_source,
            limits=sb.RunLimits(wall_clock_s=0.0),
        )
    finally:
        parent.close()

    assert result.timed_out is True
    assert result.run_dir.joinpath("code.py").read_text() == "code source"
    assert result.run_dir.joinpath("runner.py").read_text() == runner_source
    assert str(executor._pkg_dir) not in popen_call["argv"]
    assert popen_call["kwargs"]["cwd"] == str(result.run_dir)
    pass_fds = popen_call["kwargs"]["pass_fds"]
    assert len(pass_fds) == 4
    assert pass_fds[0] >= 10 and pass_fds[1] >= 10
    assert sb.FD_OUT in pass_fds
    assert sb.FD_QUERY in pass_fds
    assert popen_call["kwargs"]["start_new_session"] is True


def test_bwrap_query_channel_rejects_unusable_query_fd(tmp_path):
    executor = _argv_executor(
        venv="/repo/.venv", pkg="/production-repo", pyenv="/opt/python"
    )
    with pytest.raises((OSError, ValueError)):
        executor.run_with_query_channel(
            "pass",
            str(tmp_path / "run"),
            -1,
            runner_source="pass",
            limits=sb.RunLimits(wall_clock_s=1.0),
        )


@pytest.mark.parametrize(
    ("platform", "exists", "expected"),
    [
        ("darwin", False, "seatbelt"),
        ("linux", True, "bwrap"),
    ],
)
def test_default_executor_selects_the_platform_substrate(
        monkeypatch, platform, exists, expected):
    monkeypatch.setattr(sb.sys, "platform", platform)
    monkeypatch.setattr(sb.os.path, "exists", lambda path: exists)
    seatbelt = object()
    bwrap = object()
    monkeypatch.setattr(sb, "SeatbeltExecutor", lambda: seatbelt)
    monkeypatch.setattr(sb, "BwrapExecutor", type(
        "FakeBwrap", (), {"BWRAP": "/usr/bin/bwrap", "__new__": lambda cls: bwrap}
    ))

    assert sb.default_executor() is (seatbelt if expected == "seatbelt" else bwrap)


def test_default_executor_fails_closed_without_a_supported_substrate(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "linux")
    monkeypatch.setattr(sb.os.path, "exists", lambda path: False)
    with pytest.raises(sb.NoExecutorAvailable):
        sb.default_executor()


def test_receiver_uses_default_factory_and_maps_unavailable_sandbox_to_503(
        monkeypatch, tmp_path):
    assert inspect.signature(receiver._run_analyst).parameters[
        "executor_factory"].default is sb.default_executor
    assert inspect.signature(receiver._analyst).parameters[
        "executor_factory"].default is sb.default_executor
    assert inspect.signature(receiver.create_app).parameters[
        "analyst_executor_factory"].default is sb.default_executor

    # Force the no-substrate case by platform, NOT by replacing SeatbeltExecutor.
    # Patching that class only reaches `default_executor()`'s darwin branch, so
    # on Linux the bwrap branch ran instead, a real sandbox was constructed, and
    # this test proceeded into run_analyst and died opening a nonexistent vault.
    # It passed on macOS and failed on the box — measured 2026-08-30. The thing
    # under test is the NoExecutorAvailable -> 503 mapping, which is
    # platform-independent, so the trigger must be too.
    monkeypatch.setattr(sb.sys, "platform", "nosuchos")
    response = receiver._run_analyst(
        SimpleNamespace(db_path=tmp_path / "vault.db"),
        "this must not reach the model",
        complete_fn=lambda prompt: (_ for _ in ()).throw(
            AssertionError("model called without a sandbox")),
    )
    assert response.status_code == 503
    body = json.loads(response.body)
    assert body["detail"].startswith("analyst sandbox unavailable: ")
    assert "nosuchos" in body["detail"]


LINUX_BWRAP = pytest.mark.skipif(
    sys.platform != "linux" or not shutil.which("bwrap"),
    reason="needs bwrap on Linux",
)


def _linux_pids_using_run_dir(run_dir: Path) -> list[int]:
    """Return live host PIDs whose command line belongs to this sandbox run."""
    if sys.platform != "linux":
        return []
    needle = os.fsencode(str(run_dir))
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            command_line = (entry / "cmdline").read_bytes()
            state = next(
                line for line in (entry / "status").read_text().splitlines()
                if line.startswith("State:")
            )
        except (OSError, StopIteration, ValueError):
            continue
        state_fields = state.split()
        if (needle in command_line and len(state_fields) > 1
                and state_fields[1] != "Z"):
            pids.append(pid)
    return pids


def _pid_namespace_unavailable(result) -> bool:
    """Recognize bwrap's inability to create namespaces, not other failures."""
    stderr = result.stderr.decode("utf-8", errors="replace").lower()
    return (
        not result.timed_out
        and result.returncode not in (None, 0)
        and "namespace" in stderr
        and any(token in stderr for token in (
            "operation not permitted", "permission denied", "not allowed",
        ))
    )


@LINUX_BWRAP
def test_bwrap_query_channel_fd4_probe_and_production_pkg_isolation(tmp_path):
    pkg_dir = tmp_path / "production-repo"
    (pkg_dir / "data").mkdir(parents=True)
    (pkg_dir / "systemd").mkdir()
    vault_like = pkg_dir / "data" / "health.db"
    secret_like = pkg_dir / "systemd" / "receiver.env"
    vault_like.write_bytes(b"SQLite format 3\x00secret")
    secret_like.write_text("RECEIVER_HMAC=secret", encoding="utf-8")

    run_dir = tmp_path / "query-run"
    parent, child = socket.socketpair()
    query_fd = child.detach()
    code = f"""
import json, os
facts = {{}}
for name, path in (("vault", {str(vault_like)!r}),
                   ("secret", {str(secret_like)!r})):
    try:
        open(path, "rb").read()
        facts[name] = "unsafe"
    except Exception:
        facts[name] = "blocked"
os.write(4, b"fd4-survived")
os.write(1, json.dumps(facts).encode())
"""
    try:
        result = sb.BwrapExecutor(pkg_dir=str(pkg_dir)).run_with_query_channel(
            code,
            str(run_dir),
            query_fd,
            runner_source=analyst_runner._runner_source("code.py"),
            limits=sb.RunLimits(wall_clock_s=5.0),
        )
        probe = parent.recv(len(b"fd4-survived"))
    finally:
        parent.close()

    assert result.returncode == 0, result.stderr
    assert probe == b"fd4-survived"
    # The probe reports on STDOUT, not fd 3. fd 3 belongs to the runner, which
    # appends its own envelope there, so reading the probe from fd 3 yielded two
    # concatenated JSON documents and `fd3_as_json()` returned None — measured on
    # a Linux host 2026-08-30, where the buffer was literally
    # b'{"vault": "blocked", "secret": "blocked"}{"tables":[]}'. The isolation was
    # working; only the channel the assertion read was wrong.
    assert json.loads(result.stdout) == {"vault": "blocked", "secret": "blocked"}


@LINUX_BWRAP
def test_bwrap_query_channel_carries_runner_query_traffic(tmp_path):
    parent, child = socket.socketpair()
    query_fd = child.detach()
    requests = []

    def serve_one_query():
        with parent:
            header = parent.recv(4)
            size = struct.unpack("!I", header)[0]
            request = json.loads(parent.recv(size).decode())
            requests.append(request)
            response = json.dumps({
                "ok": True,
                "description": [["value"]],
                "rows": [[42]],
            }, separators=(",", ":")).encode()
            parent.sendall(struct.pack("!I", len(response)) + response)

    server = threading.Thread(target=serve_one_query)
    server.start()
    code = (
        "row = conn.execute('SELECT 42').fetchone()\n"
        "emit('probe', ['value'], ['count'], [[row[0]]])\n"
    )
    try:
        result = sb.BwrapExecutor().run_with_query_channel(
            code,
            str(tmp_path / "query-traffic"),
            query_fd,
            runner_source=analyst_runner._runner_source("code.py"),
            limits=sb.RunLimits(wall_clock_s=5.0),
        )
    finally:
        server.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert requests == [{"sql": "SELECT 42", "params": []}]
    assert result.fd3_as_json() == {
        "tables": [{
            "name": "probe",
            "columns": ["value"],
            "units": ["count"],
            "rows": [[42]],
        }]
    }


@LINUX_BWRAP
def test_bwrap_real_sandbox_smoke_enforces_grants_and_fd3(tmp_path):
    vault_path = tmp_path / "vault.db"
    conn = sqlite3.connect(vault_path)
    conn.execute("CREATE TABLE probe (value INTEGER)")
    conn.execute("INSERT INTO probe VALUES (7)")
    conn.commit()
    conn.close()

    run_dir = tmp_path / "run with spaces"
    (run_dir / "parent.txt").mkdir(parents=True)
    # Replace the directory-shaped setup with the parent-owned file after the
    # parent directory has been created; the child must be able to read it but
    # not modify the run-directory parent.
    (run_dir / "parent.txt").rmdir()
    (run_dir / "parent.txt").write_text("parent", encoding="utf-8")
    code = f"""
import socket, sqlite3
facts = {{}}
facts["parent_read"] = int(open({str(run_dir / 'parent.txt')!r}).read() == "parent")
try:
    open({str(run_dir / 'parent.txt')!r}, "w").write("changed")
    facts["parent_write"] = 0
except Exception:
    facts["parent_write"] = 1
open({str(run_dir / 'work' / 'work-child.txt')!r}, "w").write("allowed")
facts["work_write"] = int(open({str(run_dir / 'work' / 'work-child.txt')!r}).read() == "allowed")
facts["vault_rows"] = conn.execute("SELECT COUNT(*) FROM probe").fetchone()[0]
# Nothing binds the vault into this namespace any more (#37). A raw connect
# to the host path therefore either fails or creates a fresh, empty file on
# bubblewrap's private root -- measured on CI: CREATE TABLE "succeeds" there.
# So the probe is whether the raw path can see the REAL vault's table; the
# parent checks the host file itself after the run.
try:
    c = sqlite3.connect({str(vault_path)!r})
    c.execute("SELECT COUNT(*) FROM probe").fetchone()
    facts["vault_raw_blocked"] = 0
except Exception:
    facts["vault_raw_blocked"] = 1
try:
    c.execute("CREATE TABLE denied (value INTEGER)")
    c.commit()
except Exception:
    pass
try:
    socket.gethostbyname("openrouter.ai")
    facts["network"] = 0
except Exception:
    facts["network"] = 1
emit("sandbox", list(facts), ["count"] * len(facts), [list(facts.values())])
"""
    # The former plain run() body read conn directly.  The query-channel
    # runner supplies the parent-owned proxy and validates its fd-3 envelope.
    # Paths are absolute because the bwrap query child starts in run_dir
    # (_build_query_argv --chdir), not in work/ as the plain path did; the
    # binds sit at identical paths on both sides of the namespace.
    result = analyst_runner.run_analyst_code(
        code, str(vault_path), str(run_dir), sb.BwrapExecutor())
    assert isinstance(result, analyst_envelope.Envelope)
    assert result.tables[0]["name"] == "sandbox"
    assert result.tables[0]["rows"] == ((1, 1, 1, 1, 1, 1),)
    assert result.ledger["parent_observed"] is True
    # The host vault is exactly as the parent built it: the probe table, and
    # no "denied" table from the child's raw connect.
    host = sqlite3.connect(vault_path)
    tables = {row[0] for row in host.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    host.close()
    assert tables == {"probe"}


@LINUX_BWRAP
def test_bwrap_kills_forked_setsid_descendant_at_deadline(tmp_path):
    vault_path = tmp_path / "vault.db"
    vault_path.write_bytes(b"")
    run_dir = tmp_path / "setsid_escape"
    marker = run_dir / "work" / "marker.txt"
    # The parent must OUTLIVE the wall clock, or this tests nothing. As first
    # written it forked and fell off the end of the script, so the run returned
    # returncode=0 in milliseconds and `timed_out` was False — the assertion
    # below failed on the box while the sandbox was working perfectly. Holding
    # the parent open is what actually exercises the timeout and the group kill,
    # which is the path a runaway child would really take.
    code = """
import os, time
pid = os.fork()
if pid == 0:
    os.setsid()
    for i in range(15):
        with open("marker.txt", "a") as f:
            f.write(str(i) + "\\n")
            f.flush()
        time.sleep(1)
    os._exit(0)
time.sleep(30)
"""
    result = sb.BwrapExecutor().run(
        code, str(vault_path), str(run_dir), sb.RunLimits(wall_clock_s=2.0)
    )
    if _pid_namespace_unavailable(result):
        pytest.skip("bubblewrap could not create the PID namespace on this host")
    assert result.timed_out is True
    assert result.killed_group is True
    assert marker.exists(), "expected the child to write before the timeout"
    lines_after_run = marker.read_text().splitlines()
    time.sleep(1)
    assert marker.read_text().splitlines() == lines_after_run
    survivors = _linux_pids_using_run_dir(run_dir)
    survivor_count = len(survivors)
    print(f"\n[setsid PID namespace] survivor_count={survivor_count}")
    try:
        assert survivor_count == 0, (
            f"PID namespace left {survivor_count} live sandbox process(es): "
            f"{survivors}"
        )
    finally:
        for pid in survivors:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
