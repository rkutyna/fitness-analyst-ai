"""A1 — the analyst-mode sandbox executor (#115 / M7).

Spec: docs/product/reviews/analyst-mode-proposal.md — read §2 (the runtime),
§4.6 (the split run directory), and §8's A1 task definition before touching
this file.

This module is the **only substrate-aware seam** (§2.4):

    Analyst loop  ---> Executor (protocol)   .run(code, vault_path, run_dir, limits)
                   +-> SeatbeltExecutor   (this file, macOS)
                   +-> BwrapExecutor      (this file, Linux)
                   +-> FargateExecutor    (production, unbuilt)

`RawResult` is deliberately **untrusted bytes** — the fd-3 payload is handed
back exactly as the child wrote it, with no envelope parsing, no whitelist,
no caps beyond a raw byte ceiling. Turning those bytes into a validated
`Envelope` (columns, units, refusals) is A2's job (`analyst_envelope.py`,
unbuilt). Treating every executor's output identically is what makes the
substrate swappable.

What this module does implement, per A1's `Must implement` (§8):

- A seatbelt (`sandbox-exec`) profile per §2.5, generated into **parent-owned**
  space, with `os.path.realpath()` resolved on every path written into it —
  this worktree symlinks `.venv` and `data/health.db` into the main checkout,
  and an unresolved symlink in the profile silently denies (§2.3 failure 3).
- A bubblewrap mount namespace on Linux with the same split run-directory,
  minimal environment, read-only vault, and fd-3 result-channel contracts.
- An absolute interpreter path, `python -I`, `cwd` inside the child-writable
  `work/` directory, and an `env -i`-style minimal environment — `PATH`,
  `TMPDIR`, `HOME` all pointed at `work/`. None of the parent's own
  environment (secrets included) is inherited.
- The **split run directory** (§4.6): only `$RUNDIR/work` is child-writable.
  `profile.sb`, `code.py`, and every parent-owned record live outside the
  child's write grant, so nothing the child does can rewrite its own
  provenance or race a parent read.
- The result envelope's raw bytes arrive **only on fd 3** — never as a file
  the child could rewrite, per review finding 5.
- **Process-group kill on timeout**: the child is spawned with
  `start_new_session=True` (its own session/process group), and a timeout
  kills the whole group with `os.killpg`, because a forked grandchild
  otherwise survives the direct child's death and can go on holding a SQLite
  lock (review finding 6, §3.5).
- The vault connection is opened **by the parent-authored runner bootstrap**,
  read-only, from a `realpath`-resolved literal the harness bakes into the
  generated `runner.py` — the same way `profile.sb` bakes in paths. The
  user's/model's own code (`code.py`) receives the live `conn` object as a
  bound global; it is never handed the path string itself. (§3.1: "the
  generated code never receives a vault path".) The full **ledgered** wrapper
  around this connection — the authorizer, the refusal on zero reads — is
  A2's `analyst_ledger.py`; this module hands over a plain read-only
  `sqlite3.Connection`.

KNOWN, UNBLOCKED GAP (found during this work, not defended against here): a
forked descendant that calls `os.setsid()` before the wall-clock deadline
fires leaves the process group `os.killpg` targets — permanently. Nothing in
the seatbelt profile constrains `setsid` (it is a plain POSIX syscall, not
filesystem or network activity), and `process-fork` is allowed by design.
Measured directly: such a descendant keeps running, and keeps writing,
several seconds after `run()` has already returned with `killed_group=True`
for everything that *was* still in the tracked group. See
`tests/test_analyst_sandbox.py::test_KNOWN_GAP_forked_setsid_descendant_escapes_process_group_kill`
for the reproduction. Closing this needs tracking real descendant PIDs
(e.g. via `EVFILT_PROC`/`kqueue`) rather than trusting one process-group id
across the whole run — out of scope for A1, and not attempted here.
"""
from __future__ import annotations

import fcntl
import json
import os
import select
import selectors
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #

DEFAULT_WALL_CLOCK_S = 60.0

# A raw-channel sanity bound, not the envelope's cap. §4.5 sets the *envelope*
# byte cap (65,536) as a property of the validated payload — that's A2's job
# (analyst_envelope.py, unbuilt). This module has no schema at all: it exists
# so a hostile child cannot grow the parent's memory unboundedly by writing
# gigabytes to fd 3. Re-used at the same value on purpose (there is exactly
# one number in the spec for "how much fd-3 data is reasonable"), but this is
# the executor deciding "small enough to be worth handing to a validator",
# not the validator itself.
DEFAULT_MAX_FD3_BYTES = 65_536

# Past this many bytes we stop *storing* fd-3 output but keep draining the
# pipe (discarding what we read) so a hostile writer never blocks forever
# waiting on a full pipe buffer — that would otherwise turn "oversized write"
# into "hang until the 60s timeout", which is a worse failure mode to debug.
_FD3_DRAIN_CEILING = DEFAULT_MAX_FD3_BYTES * 32  # 2 MiB before we stop caring

_CHUNK = 65_536

# The only fd, beyond stdin/stdout/stderr, the child is handed.
FD_OUT = 3
FD_QUERY = 4


@dataclass(frozen=True)
class RunLimits:
    wall_clock_s: float = DEFAULT_WALL_CLOCK_S
    max_fd3_bytes: int = DEFAULT_MAX_FD3_BYTES


@dataclass
class RawResult:
    """Untrusted bytes and process facts. §2.4: A2's validator is meant to
    treat every executor's `RawResult` identically — that symmetry is what
    makes `Executor` swappable for a Fargate implementation later."""

    fd3_bytes: bytes
    fd3_oversized: bool
    stdout: bytes
    stderr: bytes
    returncode: int | None
    timed_out: bool
    killed_group: bool
    duration_s: float
    pgid: int
    run_dir: Path

    def fd3_as_json(self):
        """Best-effort `json.loads` of the fd-3 payload. A convenience for
        tests and callers that want a quick look; NOT the hardened A2
        grammar (no NaN/Infinity/duplicate-key handling, no caps beyond the
        raw byte ceiling above). Returns None on any parse failure."""
        try:
            return json.loads(self.fd3_bytes.decode("utf-8"))
        except Exception:
            return None


class Executor(Protocol):
    """The only substrate-aware seam (§2.4)."""

    def run(
        self,
        code: str,
        vault_path: str,
        run_dir: str,
        limits: RunLimits | None = None,
    ) -> RawResult:
        ...


class NoExecutorAvailable(RuntimeError):
    """No supported sandbox substrate is available on this host."""


class TransientUnitError(RuntimeError):
    """The user-systemd transient unit could not run the analyst sandbox."""


def bridge_bidirectional(left: socket.socket, right: socket.socket) -> None:
    """Copy opaque bytes in both directions until both peers reach EOF.

    This deliberately has no framing or analyst-specific knowledge.  The
    parent remains the endpoint that understands query frames; this function
    is only the transport crossing the user-manager process boundary.
    """
    stop = threading.Event()

    def _shutdown(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, ValueError):
            pass

    def _pump(source: socket.socket, destination: socket.socket) -> None:
        try:
            while not stop.is_set():
                readable, _, _ = select.select([source], [], [], 0.1)
                if not readable:
                    continue
                chunk = source.recv(_CHUNK)
                if not chunk:
                    try:
                        destination.shutdown(socket.SHUT_WR)
                    except (OSError, ValueError):
                        pass
                    return
                destination.sendall(chunk)
        except (OSError, ValueError):
            stop.set()
            _shutdown(left)
            _shutdown(right)

    threads = [
        threading.Thread(target=_pump, args=(left, right), daemon=True),
        threading.Thread(target=_pump, args=(right, left), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


# --------------------------------------------------------------------------- #
# The seatbelt profile (§2.5)
# --------------------------------------------------------------------------- #

# `(literal "$VAULT")` rather than `(subpath ...)`: the vault is one file, not
# a directory, and `literal` is the tighter grant.
_PROFILE_TEMPLATE = """\
(version 1)
(deny default)
(allow process-exec* process-fork sysctl-read mach-lookup)
(allow file-read*)
(deny  file-read-data (subpath "{real_home}"))
(allow file-read-data (subpath "{pyenv_prefix}") (subpath "{venv_dir}")
                      (subpath "{pkg_dir}") (literal "{vault}"))
(allow file-read* file-write* (subpath "{work_dir}"))
(deny network*)
"""


def _scheme_quote(path: str) -> str:
    """Escape a path for embedding in a seatbelt (Scheme) string literal."""
    return path.replace("\\", "\\\\").replace('"', '\\"')


def build_profile(
    *,
    real_home: str,
    pyenv_prefix: str,
    venv_dir: str,
    pkg_dir: str,
    vault_path: str,
    work_dir: str,
) -> str:
    """Render §2.5's profile. Every argument here must already be a
    `os.path.realpath()` result — this function does not resolve symlinks
    itself, because generating the profile is where the harness's other
    knowledge (which paths are meant to be parent-owned vs. child-writable)
    lives, and re-resolving here would hide a caller's mistake instead of
    catching it."""
    return _PROFILE_TEMPLATE.format(
        real_home=_scheme_quote(real_home),
        pyenv_prefix=_scheme_quote(pyenv_prefix),
        venv_dir=_scheme_quote(venv_dir),
        pkg_dir=_scheme_quote(pkg_dir),
        vault=_scheme_quote(vault_path),
        work_dir=_scheme_quote(work_dir),
    )


# --------------------------------------------------------------------------- #
# The parent-authored runner bootstrap
# --------------------------------------------------------------------------- #

# This is NOT `health_advisor/analyst_runner.py` (that file, with the
# ledgered `conn` and the real `emit()`, is A2's deliverable). This is the
# minimal bootstrap A1 needs to exercise the executor itself: it opens the
# vault read-only (so "the vault handle is opened by the parent, and
# generated code never receives a path" is true even in A1, before the
# ledger exists), then execs the run's `code.py` with `conn` bound and
# nothing path-shaped bound anywhere.
_RUNNER_TEMPLATE = """\
import sqlite3 as _sqlite3

try:
    conn = _sqlite3.connect("file:{vault_uri}?mode=ro", uri=True)
except Exception:
    conn = None

with open({code_path!r}, "r", encoding="utf-8") as _f:
    _src = _f.read()

_globals = {{"__name__": "__main__", "__builtins__": __builtins__, "conn": conn}}
# Compiled with the absolute path as the filename (not "code.py") so a
# traceback's linecache lookup can actually find the source: cwd is
# work/, and code.py lives one level up, in the parent-owned $RUNDIR.
exec(compile(_src, {code_path!r}, "exec"), _globals)
"""


def _build_runner(*, vault_real: str, code_path_real: str) -> str:
    vault_uri = urllib.parse.quote(vault_real)
    return _RUNNER_TEMPLATE.format(vault_uri=vault_uri, code_path=code_path_real)


# --------------------------------------------------------------------------- #
# SeatbeltExecutor
# --------------------------------------------------------------------------- #


class SeatbeltExecutor:
    """Runs code under `/usr/bin/sandbox-exec` on macOS. §2.2's candidate A,
    "measured working" — §2.6 is explicit that this is a *measured
    configuration*, not a proven confinement boundary."""

    SANDBOX_EXEC = "/usr/bin/sandbox-exec"

    def __init__(self, python_executable: str | None = None, pkg_dir: str | None = None):
        # realpath'd immediately (spec failure 3): this worktree symlinks
        # `.venv` into the main checkout, and `sys.executable` inside a
        # pyenv-created venv resolves clean out of `.venv/bin` into the
        # pyenv install itself (measured: .venv/bin/python -> a symlink to
        # ~/.pyenv/versions/3.11.15/bin/python3.11). We exec the realpath so
        # the exec target and the profile's allow-rule name the same vnode;
        # execing the un-resolved symlink path (which sits under $HOME) would
        # be denied by the profile's own `(deny file-read-data (subpath
        # "$HOME"))` rule before python ever got to run.
        self._python = os.path.realpath(python_executable or sys.executable)
        self._venv_dir = os.path.realpath(sys.prefix)
        self._pyenv_prefix = os.path.realpath(sys.base_prefix)
        self._pkg_dir = os.path.realpath(pkg_dir or str(REPO_ROOT))
        self._real_home = os.path.realpath(os.path.expanduser("~"))
        if not os.path.exists(self.SANDBOX_EXEC):
            raise RuntimeError(
                f"{self.SANDBOX_EXEC} not found — SeatbeltExecutor is macOS-only "
                "(§2.2 candidate A)."
            )

    # -- profile / run-dir plumbing ---------------------------------------- #

    def _prepare_run_dir(self, run_dir: str) -> tuple[Path, Path]:
        run_dir_p = Path(run_dir)
        run_dir_p.mkdir(parents=True, exist_ok=True)
        work_dir_p = run_dir_p / "work"
        work_dir_p.mkdir(parents=True, exist_ok=True)
        # realpath'd *after* creation: on macOS the system temp root itself is
        # a symlink (/tmp -> /private/tmp), and pytest's own tmp_path already
        # lives under /private/var/folders/... but nothing guarantees a
        # caller's run_dir doesn't route through another symlinked
        # component. Resolve what we actually created, not what we were
        # asked for.
        run_dir_real = Path(os.path.realpath(run_dir_p))
        work_dir_real = Path(os.path.realpath(work_dir_p))
        return run_dir_real, work_dir_real

    # -- the run ------------------------------------------------------------ #

    def run(
        self,
        code: str,
        vault_path: str,
        run_dir: str,
        limits: RunLimits | None = None,
    ) -> RawResult:
        limits = limits or RunLimits()
        run_dir_real, work_dir_real = self._prepare_run_dir(run_dir)
        vault_real = os.path.realpath(vault_path)

        # Parent-owned artifacts (§4.6): profile.sb, code.py, and the
        # generated runner all live directly under $RUNDIR, never under
        # work/, so the child's write grant (work/ only) cannot touch them.
        code_path = run_dir_real / "code.py"
        code_path.write_text(code, encoding="utf-8")

        runner_src = _build_runner(vault_real=vault_real, code_path_real=str(code_path))
        runner_path = run_dir_real / "runner.py"
        runner_path.write_text(runner_src, encoding="utf-8")

        profile_text = build_profile(
            real_home=self._real_home,
            pyenv_prefix=self._pyenv_prefix,
            venv_dir=self._venv_dir,
            pkg_dir=self._pkg_dir,
            vault_path=vault_real,
            work_dir=str(work_dir_real),
        )
        profile_path = run_dir_real / "profile.sb"
        profile_path.write_text(profile_text, encoding="utf-8")

        argv = [
            self.SANDBOX_EXEC,
            "-f",
            str(profile_path),
            self._python,
            "-I",
            str(runner_path),
        ]

        # env -i, by construction: we build the dict from nothing rather than
        # `os.environ.copy()`, so no parent secret (an API key, a token in
        # the test's own env) is ever a candidate for inheritance.
        env = {
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(work_dir_real),
            "HOME": str(work_dir_real),
        }

        r_fd, w_fd = os.pipe()

        def _preexec() -> None:
            # Runs in the forked child, before sandbox-exec's own exec. This
            # is the one place a specific fd number is guaranteed: dup2 the
            # pipe's write end onto fd 3 so it survives past the close_fds
            # sweep (which we tell to keep only fd 3 via pass_fds).
            #
            # Order matters here and it is not cosmetic: `os.pipe()` hands
            # out the next free descriptors, and the read end (`r_fd`) is
            # created first, so it is `r_fd` that is more likely to land on
            # a low number — including, sometimes, FD_OUT (3) itself. Close
            # `r_fd` *before* touching FD_OUT: closing it after would close
            # whatever the dup2 below just placed at fd 3 rather than the
            # read end, silently destroying the channel while leaving fd 3
            # open for the next thing that asks for a low fd — which, one
            # layer up, is the vault's own sqlite3 connection. That failure
            # was measured directly: fd 3 came back with `fstat` mode
            # 0o100444 — the *vault file's* permissions, not a pipe — every
            # write to it then raised `OSError: Bad file descriptor` because
            # the pipe write end had already been torn down.
            os.close(r_fd)
            if w_fd != FD_OUT:
                os.dup2(w_fd, FD_OUT)
                os.close(w_fd)

        start = time.monotonic()
        proc = subprocess.Popen(
            argv,
            cwd=str(work_dir_real),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(FD_OUT,),
            preexec_fn=_preexec,
            start_new_session=True,  # §3.5 — its own process group
            close_fds=True,
        )
        os.close(w_fd)  # parent must not hold the write end open, or EOF
        # on the read end never arrives even after the child exits.
        pgid = proc.pid  # start_new_session=True makes the child its own
        # process-group leader, so pgid == pid at spawn time.

        fd3_buf = bytearray()
        fd3_oversized = False
        stdout_buf = bytearray()
        stderr_buf = bytearray()

        sel = selectors.DefaultSelector()
        sel.register(r_fd, selectors.EVENT_READ, "fd3")
        sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
        sel.register(proc.stderr, selectors.EVENT_READ, "stderr")

        timed_out = False
        killed_group = False
        deadline = start + limits.wall_clock_s

        def _drain_ready(timeout: float) -> None:
            nonlocal fd3_oversized
            for key, _ in sel.select(timeout=timeout):
                fileobj = key.fileobj
                fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
                try:
                    chunk = os.read(fd, _CHUNK)
                except OSError:
                    chunk = b""
                if not chunk:
                    sel.unregister(fileobj)
                    try:
                        if not isinstance(fileobj, int):
                            fileobj.close()
                    except Exception:
                        pass
                    continue
                if key.data == "fd3":
                    if len(fd3_buf) < _FD3_DRAIN_CEILING:
                        fd3_buf.extend(chunk)
                        if len(fd3_buf) > limits.max_fd3_bytes:
                            fd3_oversized = True
                    else:
                        fd3_oversized = True  # discard: draining only
                elif key.data == "stdout":
                    stdout_buf.extend(chunk)
                else:
                    stderr_buf.extend(chunk)

        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            _drain_ready(min(remaining, 0.25))

        if timed_out:
            try:
                # Use the pgid captured at spawn time, NOT
                # os.getpgid(proc.pid) looked up now. Measured directly: once
                # the *direct* child has already exited (its own script ran
                # to completion while a forked descendant lingers — the
                # exact review-finding-6 shape), the direct child is a
                # zombie, and os.getpgid() on an exited pid raises
                # ProcessLookupError on this OS — which silently skipped the
                # kill entirely, leaving every surviving descendant alone.
                # The stored pgid needs no such lookup: start_new_session
                # guarantees pgid == pid at spawn, permanently, regardless
                # of whether that pid is later reaped.
                os.killpg(pgid, signal.SIGKILL)
                killed_group = True
            except ProcessLookupError:
                pass
            except PermissionError:
                # Measured: once the group's leader is a zombie (exited but
                # not yet reaped) and nothing else remains *in that group*,
                # killpg can come back EPERM rather than ESRCH on this OS.
                # Nothing changes about what this means for the caller:
                # either way, this executor has no further lever over
                # whatever is still alive outside the original group (see
                # the module docstring's note on setsid escapes).
                pass
            # Grace period: drain whatever the dying processes still flush,
            # and reap the direct child, without hanging the harness.
            grace_deadline = time.monotonic() + 5.0
            while sel.get_map() and time.monotonic() < grace_deadline:
                _drain_ready(0.25)

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass

        for fileobj in (r_fd, proc.stdout, proc.stderr):
            try:
                if isinstance(fileobj, int):
                    os.close(fileobj)
                else:
                    fileobj.close()
            except Exception:
                pass

        duration_s = time.monotonic() - start

        return RawResult(
            fd3_bytes=bytes(fd3_buf[: limits.max_fd3_bytes]) if fd3_oversized else bytes(fd3_buf),
            fd3_oversized=fd3_oversized,
            stdout=bytes(stdout_buf),
            stderr=bytes(stderr_buf),
            returncode=proc.returncode,
            timed_out=timed_out,
            killed_group=killed_group,
            duration_s=duration_s,
            pgid=pgid,
            run_dir=run_dir_real,
        )


# --------------------------------------------------------------------------- #
# BwrapExecutor
# --------------------------------------------------------------------------- #


class BwrapExecutor:
    """Runs code under bubblewrap on Linux.

    The mount namespace exposes only the runtime, package, vault, and run
    directory paths needed by the runner.  The read-only run-directory bind is
    deliberately followed by the nested writable ``work`` bind: that ordering
    is what preserves the §4.6 split-run-directory contract.
    """

    BWRAP = "/usr/bin/bwrap"

    def __init__(self, python_executable: str | None = None, pkg_dir: str | None = None):
        # Keep path resolution identical to SeatbeltExecutor.  In particular,
        # the venv and pyenv interpreter may be symlinks, and bwrap must bind
        # and exec the same real paths or the kernel resolves the target
        # outside the namespace's visible tree.
        self._python = os.path.realpath(python_executable or sys.executable)
        self._venv_dir = os.path.realpath(sys.prefix)
        self._pyenv_prefix = os.path.realpath(sys.base_prefix)
        self._pkg_dir = os.path.realpath(pkg_dir or str(REPO_ROOT))
        self._real_home = os.path.realpath(os.path.expanduser("~"))
        if not os.path.exists(self.BWRAP):
            raise RuntimeError(
                f"{self.BWRAP} not found — BwrapExecutor is Linux-only."
            )

    @staticmethod
    def _is_subpath(path: str, parent: str) -> bool:
        try:
            Path(path).relative_to(parent)
        except ValueError:
            return False
        return True

    def _prepare_run_dir(self, run_dir: str) -> tuple[Path, Path]:
        run_dir_p = Path(run_dir)
        run_dir_p.mkdir(parents=True, exist_ok=True)
        work_dir_p = run_dir_p / "work"
        work_dir_p.mkdir(parents=True, exist_ok=True)
        run_dir_real = Path(os.path.realpath(run_dir_p))
        work_dir_real = Path(os.path.realpath(work_dir_p))
        return run_dir_real, work_dir_real

    def _build_argv(
        self,
        *,
        vault_real: str,
        run_dir_real: Path,
        work_dir_real: Path,
        runner_path: Path,
    ) -> list[str]:
        argv = [
            self.BWRAP,
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/sbin", "/sbin",
            "--ro-bind", self._pyenv_prefix, self._pyenv_prefix,
            "--ro-bind", self._pkg_dir, self._pkg_dir,
        ]
        if not (
            self._is_subpath(self._venv_dir, self._pkg_dir)
            or self._is_subpath(self._venv_dir, self._pyenv_prefix)
        ):
            argv.extend(["--ro-bind", self._venv_dir, self._venv_dir])
        argv.extend([
            "--ro-bind", vault_real, vault_real,
            "--ro-bind", str(run_dir_real), str(run_dir_real),
            "--bind", str(work_dir_real), str(work_dir_real),
            "--proc", "/proc",
            "--dev", "/dev",
            "--chdir", str(work_dir_real),
            "--unshare-all", "--die-with-parent", "--new-session",
            self._python, "-I", str(runner_path),
        ])
        return argv

    def _build_query_argv(
        self,
        *,
        run_dir_real: Path,
        work_dir_real: Path,
        runner_path: Path,
    ) -> list[str]:
        """Build the query-channel namespace without any repository bind."""
        argv = [
            self.BWRAP,
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/sbin", "/sbin",
            "--ro-bind", self._pyenv_prefix, self._pyenv_prefix,
        ]
        if not self._is_subpath(self._venv_dir, self._pyenv_prefix):
            argv.extend(["--ro-bind", self._venv_dir, self._venv_dir])
        argv.extend([
            "--ro-bind", str(run_dir_real), str(run_dir_real),
            "--bind", str(work_dir_real), str(work_dir_real),
            "--proc", "/proc",
            "--dev", "/dev",
            "--chdir", str(run_dir_real),
            "--unshare-all", "--die-with-parent", "--new-session",
            self._python, "-I", str(runner_path),
        ])
        return argv

    @staticmethod
    def _high_fd(fd: int) -> int:
        """Move a channel descriptor away from the fixed child slots."""
        duplicate = fcntl.fcntl(fd, fcntl.F_DUPFD, 10)
        os.close(fd)
        return duplicate

    def _collect_query_result(
        self,
        proc,
        r_fd: int,
        run_dir_real: Path,
        limits,
        start: float,
    ) -> RawResult:
        """Drain the fd-3/stdout/stderr channels with run()'s semantics."""
        fd3_buf = bytearray()
        fd3_oversized = False
        stdout_buf = bytearray()
        stderr_buf = bytearray()

        sel = selectors.DefaultSelector()
        sel.register(r_fd, selectors.EVENT_READ, "fd3")
        sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
        sel.register(proc.stderr, selectors.EVENT_READ, "stderr")

        timed_out = False
        killed_group = False
        deadline = start + limits.wall_clock_s

        def _drain_ready(timeout: float) -> None:
            nonlocal fd3_oversized
            for key, _ in sel.select(timeout=timeout):
                fileobj = key.fileobj
                fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
                try:
                    chunk = os.read(fd, _CHUNK)
                except OSError:
                    chunk = b""
                if not chunk:
                    sel.unregister(fileobj)
                    try:
                        if not isinstance(fileobj, int):
                            fileobj.close()
                    except Exception:
                        pass
                    continue
                if key.data == "fd3":
                    if len(fd3_buf) < _FD3_DRAIN_CEILING:
                        fd3_buf.extend(chunk)
                        if len(fd3_buf) > limits.max_fd3_bytes:
                            fd3_oversized = True
                    else:
                        fd3_oversized = True
                elif key.data == "stdout":
                    stdout_buf.extend(chunk)
                else:
                    stderr_buf.extend(chunk)

        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            _drain_ready(min(remaining, 0.25))

        if timed_out:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                killed_group = True
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
            grace_deadline = time.monotonic() + 5.0
            while sel.get_map() and time.monotonic() < grace_deadline:
                _drain_ready(0.25)

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass

        for fileobj in (r_fd, proc.stdout, proc.stderr):
            try:
                if isinstance(fileobj, int):
                    os.close(fileobj)
                else:
                    fileobj.close()
            except Exception:
                pass

        return RawResult(
            fd3_bytes=(bytes(fd3_buf[:limits.max_fd3_bytes])
                       if fd3_oversized else bytes(fd3_buf)),
            fd3_oversized=fd3_oversized,
            stdout=bytes(stdout_buf),
            stderr=bytes(stderr_buf),
            returncode=proc.returncode,
            timed_out=timed_out,
            killed_group=killed_group,
            duration_s=time.monotonic() - start,
            pgid=proc.pid,
            run_dir=run_dir_real,
        )

    def run_with_query_channel(
        self,
        code: str,
        run_dir: str,
        query_fd: int,
        *,
        runner_source: str,
        limits,
    ) -> RawResult:
        """Run analyst code with parent-mediated SQL on fd 4."""
        limits = limits or RunLimits()
        run_dir_real, work_dir_real = self._prepare_run_dir(run_dir)

        code_path = run_dir_real / "code.py"
        code_path.write_text(code, encoding="utf-8")
        runner_path = run_dir_real / "runner.py"
        runner_path.write_bytes(runner_source.encode("utf-8"))

        argv = self._build_query_argv(
            run_dir_real=run_dir_real,
            work_dir_real=work_dir_real,
            runner_path=runner_path,
        )
        env = {
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(work_dir_real),
            "HOME": str(work_dir_real),
        }

        out_r, out_w = os.pipe()
        query_high = query_fd
        try:
            # Do this before any dup2: callers may have handed us fd 3 or 4,
            # and os.pipe() may also choose either fixed slot.
            out_r = self._high_fd(out_r)
            out_w = self._high_fd(out_w)
            query_high = self._high_fd(query_fd)
        except Exception:
            for fd in (out_r, out_w, query_high):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

        def _preexec() -> None:
            os.close(out_r)
            os.dup2(out_w, FD_OUT)
            os.close(out_w)
            os.dup2(query_high, FD_QUERY)
            os.close(query_high)

        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(run_dir_real),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Keep both source fds available to preexec_fn and both fixed
                # destinations alive through the final exec.
                pass_fds=(out_w, query_high, FD_OUT, FD_QUERY),
                preexec_fn=_preexec,
                start_new_session=True,
                close_fds=True,
            )
        except Exception:
            for fd in (out_r, out_w, query_high):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

        os.close(out_w)
        os.close(query_high)
        return self._collect_query_result(
            proc, out_r, run_dir_real, limits, start
        )

    def run(
        self,
        code: str,
        vault_path: str,
        run_dir: str,
        limits: RunLimits | None = None,
    ) -> RawResult:
        limits = limits or RunLimits()
        run_dir_real, work_dir_real = self._prepare_run_dir(run_dir)
        vault_real = os.path.realpath(vault_path)

        # Parent-owned artifacts (§4.6): code.py and the generated runner live
        # directly under $RUNDIR, never under work/.
        code_path = run_dir_real / "code.py"
        code_path.write_text(code, encoding="utf-8")
        runner_src = _build_runner(vault_real=vault_real, code_path_real=str(code_path))
        runner_path = run_dir_real / "runner.py"
        runner_path.write_text(runner_src, encoding="utf-8")

        argv = self._build_argv(
            vault_real=vault_real,
            run_dir_real=run_dir_real,
            work_dir_real=work_dir_real,
            runner_path=runner_path,
        )

        # env -i, by construction: no parent secret is inherited.
        env = {
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(work_dir_real),
            "HOME": str(work_dir_real),
        }

        r_fd, w_fd = os.pipe()

        def _preexec() -> None:
            # Runs in the forked child, before bwrap's own exec.  Close the
            # read end before touching fd 3: os.pipe() may have returned r_fd
            # as FD_OUT, and closing it after dup2 would close the new channel.
            os.close(r_fd)
            if w_fd != FD_OUT:
                os.dup2(w_fd, FD_OUT)
                os.close(w_fd)

        start = time.monotonic()
        proc = subprocess.Popen(
            argv,
            cwd=str(work_dir_real),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(FD_OUT,),
            preexec_fn=_preexec,
            start_new_session=True,
            close_fds=True,
        )
        os.close(w_fd)
        pgid = proc.pid

        fd3_buf = bytearray()
        fd3_oversized = False
        stdout_buf = bytearray()
        stderr_buf = bytearray()

        sel = selectors.DefaultSelector()
        sel.register(r_fd, selectors.EVENT_READ, "fd3")
        sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
        sel.register(proc.stderr, selectors.EVENT_READ, "stderr")

        timed_out = False
        killed_group = False
        deadline = start + limits.wall_clock_s

        def _drain_ready(timeout: float) -> None:
            nonlocal fd3_oversized
            for key, _ in sel.select(timeout=timeout):
                fileobj = key.fileobj
                fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
                try:
                    chunk = os.read(fd, _CHUNK)
                except OSError:
                    chunk = b""
                if not chunk:
                    sel.unregister(fileobj)
                    try:
                        if not isinstance(fileobj, int):
                            fileobj.close()
                    except Exception:
                        pass
                    continue
                if key.data == "fd3":
                    if len(fd3_buf) < _FD3_DRAIN_CEILING:
                        fd3_buf.extend(chunk)
                        if len(fd3_buf) > limits.max_fd3_bytes:
                            fd3_oversized = True
                    else:
                        fd3_oversized = True
                elif key.data == "stdout":
                    stdout_buf.extend(chunk)
                else:
                    stderr_buf.extend(chunk)

        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            _drain_ready(min(remaining, 0.25))

        if timed_out:
            try:
                # Use the pgid captured at spawn time, not a lookup after the
                # direct child may have become a zombie.
                os.killpg(pgid, signal.SIGKILL)
                killed_group = True
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
            grace_deadline = time.monotonic() + 5.0
            while sel.get_map() and time.monotonic() < grace_deadline:
                _drain_ready(0.25)

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass

        for fileobj in (r_fd, proc.stdout, proc.stderr):
            try:
                if isinstance(fileobj, int):
                    os.close(fileobj)
                else:
                    fileobj.close()
            except Exception:
                pass

        duration_s = time.monotonic() - start

        return RawResult(
            fd3_bytes=bytes(fd3_buf[: limits.max_fd3_bytes]) if fd3_oversized else bytes(fd3_buf),
            fd3_oversized=fd3_oversized,
            stdout=bytes(stdout_buf),
            stderr=bytes(stderr_buf),
            returncode=proc.returncode,
            timed_out=timed_out,
            killed_group=killed_group,
            duration_s=duration_s,
            pgid=pgid,
            run_dir=run_dir_real,
        )


class TransientUnitExecutor(BwrapExecutor):
    """Run the query-channel bwrap inside a user-systemd transient unit.

    The receiver's systemd service may have a mount namespace from which a
    second unprivileged user namespace is forbidden.  The user manager forks
    this unit outside that namespace, so bwrap can create its namespace there.
    The caller-provided ``run_dir`` is intentionally accepted for interface
    parity only: it is not child-visible storage.  We create the run directory
    under the shared per-user runtime tree so the user-manager child can see it.
    """

    SYSTEMD_RUN = "systemd-run"
    SYSTEMCTL = "systemctl"

    def __init__(self, python_executable: str | None = None,
                 pkg_dir: str | None = None,
                 bwrap_executable: str | None = None,
                 systemd_run_executable: str | None = None,
                 systemctl_executable: str | None = None):
        # pkg_dir is accepted for construction parity with BwrapExecutor, but
        # is deliberately not stored or bound: this query path needs neither
        # the repository nor the vault.
        del pkg_dir
        self.BWRAP = bwrap_executable or BwrapExecutor.BWRAP
        self.SYSTEMD_RUN = systemd_run_executable or self.SYSTEMD_RUN
        self.SYSTEMCTL = systemctl_executable or self.SYSTEMCTL
        self._python = os.path.realpath(python_executable or sys.executable)
        self._venv_dir = os.path.realpath(sys.prefix)
        self._pyenv_prefix = os.path.realpath(sys.base_prefix)
        self._pkg_dir = None
        for executable, label in (
            (self.BWRAP, "bwrap"), (self.SYSTEMD_RUN, "systemd-run"),
            (self.SYSTEMCTL, "systemctl"),
        ):
            if not (os.path.isabs(executable) and os.path.exists(executable)) \
                    and shutil.which(executable) is None:
                raise RuntimeError(
                    f"{executable} not found — TransientUnitExecutor needs "
                    f"{label} on Linux."
                )

    @staticmethod
    def _seconds(value: float) -> str:
        return f"{value:g}s"

    def _prepare_transient_run_dir(self, ignored_run_dir: str) -> tuple[Path, Path]:
        # Do not use ignored_run_dir: receiver.py creates it under /tmp, and
        # PrivateTmp makes that location an unsafe cross-namespace transport.
        del ignored_run_dir
        base = os.environ.get("XDG_RUNTIME_DIR")
        if not base:
            base = os.path.join(os.path.expanduser("~"), ".local", "state")
        base_path = Path(base)
        base_path.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="ha-", dir=base))
        run_dir.chmod(0o700)
        work_dir = run_dir / "work"
        work_dir.mkdir(mode=0o700)
        work_dir.chmod(0o700)
        return run_dir, work_dir

    def _transient_query_argv(
            self, *, run_dir: Path, work_dir: Path, runner_path: Path,
            query_socket: Path, result_path: Path) -> list[str]:
        argv = self._build_query_argv(
            run_dir_real=run_dir, work_dir_real=work_dir,
            runner_path=runner_path)
        python_index = argv.index(self._python)
        argv[python_index:python_index] = [
            "--clearenv",
            "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "TMPDIR", str(work_dir),
            "--setenv", "HOME", str(work_dir),
            "--setenv", "ANALYST_QUERY_SOCKET", str(query_socket),
            "--setenv", "ANALYST_RESULT_PATH", str(result_path),
        ]
        return argv

    def _kill_unit(self, unit: str) -> None:
        try:
            subprocess.run(
                [self.SYSTEMCTL, "--user", "kill", "--kill-who=all", unit],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=2, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _read_unit_result(self, unit: str) -> str:
        try:
            shown = subprocess.run(
                [self.SYSTEMCTL, "--user", "show", unit, "-p", "Result", "--value"],
                capture_output=True, text=True, timeout=2, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TransientUnitError(
                f"Result=unavailable: could not inspect transient unit {unit}: {exc}"
            ) from exc
        result = shown.stdout.strip() if shown.returncode == 0 else ""
        if not result:
            detail = shown.stderr.strip() or "systemctl show returned no Result"
            raise TransientUnitError(f"Result=unavailable: {detail}")
        return result

    @staticmethod
    def _require_ordinary_unit_result(result: str) -> None:
        if result not in ("success", "exit-code"):
            raise TransientUnitError(
                f"Result={result}: transient analyst unit failed"
            )

    def _collect_process(self, proc, unit: str, limits: RunLimits,
                         start: float) -> tuple[bytes, bytes, bool, bool]:
        stdout_buf = bytearray()
        stderr_buf = bytearray()
        sel = selectors.DefaultSelector()
        if proc.stdout is not None:
            sel.register(proc.stdout, selectors.EVENT_READ, stdout_buf)
        if proc.stderr is not None:
            sel.register(proc.stderr, selectors.EVENT_READ, stderr_buf)
        deadline = start + limits.wall_clock_s
        timed_out = False
        killed = False
        while sel.get_map() or proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                self._kill_unit(unit)
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                killed = True
                break
            for key, _ in sel.select(min(remaining, 0.25)):
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), _CHUNK)
                except OSError:
                    chunk = b""
                if not chunk:
                    sel.unregister(stream)
                    stream.close()
                else:
                    key.data.extend(chunk)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._kill_unit(unit)
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            killed = True
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except (AttributeError, OSError):
                pass
        return bytes(stdout_buf), bytes(stderr_buf), timed_out, killed

    def run_with_named_query_channel(
            self, code: str, run_dir: str, query_fd: int, *,
            runner_source: str, limits: RunLimits | None = None) -> RawResult:
        """Run bwrap from a transient unit, bridging its named socket."""
        limits = limits or RunLimits()
        run_dir_real, work_dir_real = self._prepare_transient_run_dir(run_dir)
        query_parent = None
        query_server = None
        accepted = None
        bridge_thread = None
        proc = None
        unit = f"health-advisor-analyst-{uuid.uuid4().hex}"
        start = time.monotonic()
        timed_out = False
        killed = False
        try:
            query_parent = socket.socket(fileno=query_fd)
            query_parent.setblocking(True)
            # Keep the socket basename short as AF_UNIX paths are capped at
            # roughly 108 bytes, while the user-selected runtime directory
            # may itself be nested deeply. The run directory is already
            # unique, and this adds a second unguessable component.
            query_socket = run_dir_real / f"q-{uuid.uuid4().hex[:8]}.sock"
            query_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            query_server.bind(str(query_socket))
            query_socket.chmod(0o600)
            query_server.listen(1)
            query_server.settimeout(0.2)
            code_path = run_dir_real / "code.py"
            runner_path = run_dir_real / "runner.py"
            result_path = work_dir_real / "envelope.json"
            code_path.write_text(code, encoding="utf-8")
            runner_path.write_text(runner_source, encoding="utf-8")
            bwrap_argv = self._transient_query_argv(
                run_dir=run_dir_real, work_dir=work_dir_real,
                runner_path=runner_path, query_socket=query_socket,
                result_path=result_path)
            argv = [
                self.SYSTEMD_RUN, "--user", "--pipe", "--wait", "--collect",
                f"--unit={unit}", "-p", f"RuntimeMaxSec={self._seconds(limits.wall_clock_s)}",
                "--", *bwrap_argv,
            ]

            def _bridge_worker():
                nonlocal accepted
                try:
                    while query_server is not None:
                        try:
                            accepted, _ = query_server.accept()
                            bridge_bidirectional(accepted, query_parent)
                            return
                        except socket.timeout:
                            if proc is not None and proc.poll() is not None:
                                return
                except OSError:
                    return

            # The client needs the caller's session bus/runtime environment;
            # bwrap immediately clears it and installs only the five variables
            # above, so receiver secrets do not reach analyst code.
            client_env = os.environ.copy()
            proc = subprocess.Popen(
                argv, env=client_env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True, close_fds=True,
            )
            bridge_thread = threading.Thread(target=_bridge_worker, daemon=True)
            bridge_thread.start()
            stdout, stderr, timed_out, killed = self._collect_process(
                proc, unit, limits, start)
            result = self._read_unit_result(unit)
            self._require_ordinary_unit_result(result)
            try:
                size = result_path.stat().st_size
                oversized = size > limits.max_fd3_bytes
                with result_path.open("rb") as envelope_file:
                    fd3_bytes = envelope_file.read(limits.max_fd3_bytes)
            except FileNotFoundError:
                fd3_bytes = b""
                oversized = False
            return RawResult(
                fd3_bytes=fd3_bytes, fd3_oversized=oversized,
                stdout=stdout, stderr=stderr, returncode=proc.returncode,
                timed_out=timed_out, killed_group=killed,
                duration_s=time.monotonic() - start, pgid=proc.pid,
                run_dir=run_dir_real,
            )
        except OSError as exc:
            raise TransientUnitError(
                f"Result=systemd-run-refused: could not start transient unit: {exc}"
            ) from exc
        finally:
            if query_server is not None:
                try:
                    query_server.close()
                except OSError:
                    pass
            for sock in (accepted, query_parent):
                if sock is None:
                    continue
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except (OSError, ValueError):
                    pass
            if bridge_thread is not None:
                bridge_thread.join(timeout=2)
            if query_parent is not None:
                query_parent.close()
            if query_server is not None:
                try:
                    query_socket.unlink()
                except (FileNotFoundError, OSError):
                    pass
            shutil.rmtree(run_dir_real, ignore_errors=True)

    def run(self, code: str, vault_path: str, run_dir: str,
            limits: RunLimits | None = None) -> RawResult:
        raise TypeError("TransientUnitExecutor requires the analyst query channel")


# Descriptive compatibility name for callers that refer to the dispatch
# mechanism rather than the unit's lifecycle.
SystemdRunExecutor = TransientUnitExecutor


def default_executor() -> Executor:
    """Return the host's supported sandbox executor, or fail closed."""
    if sys.platform == "darwin":
        return SeatbeltExecutor()
    if sys.platform == "linux" and os.path.exists(BwrapExecutor.BWRAP):
        return BwrapExecutor()
    raise NoExecutorAvailable(
        f"no supported analyst sandbox available on {sys.platform}: "
        "BwrapExecutor is Linux-only and SeatbeltExecutor is macOS-only"
    )
