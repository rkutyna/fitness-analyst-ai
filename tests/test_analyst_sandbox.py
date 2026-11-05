"""A1's escape corpus (#115 / M7, the analyst-mode design, restated in SECURITY.md
§8's A1 task). Every measurement here is a real `sandbox-exec` invocation
against the real `data/health.db` fixture (read-only throughout — nothing
here ever opens it writable) or a synthetic vault under `tmp_path`.

§2.6 is explicit that fourteen probes are "a configuration test... not a
confinement proof" — this file is the corpus the review demanded instead: 9
classes, each scored as attempted/blocked, not a raw pass count. Where a
class does not reach 100% under the profile exactly as specified in §2.5,
that is reported as a measured fact (see `test_attack_corpus_class_coverage`
and its module-level `KNOWN_GAPS`), not hidden by narrowing the corpus —
review finding 4's whole point was that a narrow corpus lies by omission.

Issue #26: this file used to pay a real 60s sleep in the default run — the
kill-mechanism test ran the *production default* (`DEFAULT_WALL_CLOCK_S`)
end to end on `while True: pass`. The mechanism under test (the deadline
check and `os.killpg` in `SeatbeltExecutor.run`/`BwrapExecutor.run`) does not
branch on the configured value, so a short configured timeout
(`RunLimits(wall_clock_s=2.0)`, see `test_wall_clock_configured_timeout_kills_within_tolerance`
below) exercises the identical code path in a couple of seconds instead of a
minute. `test_default_wall_clock_constant_is_60s` is the fast, separate guard
against the production default silently drifting — the thing the old
end-to-end test also happened to catch, without needing to sleep for it.
"""
from __future__ import annotations

import os
import re
import select
import sqlite3
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from health_advisor import analyst_sandbox as sb
from health_advisor import analyst_envelope
from health_advisor import analyst_ledger
from health_advisor import analyst_runner

# The vault these tests run against. Configurable on purpose: a public engine
# should not hardcode one person's file. Point HA_TEST_VAULT at any populated
# vault -- `python -m health_advisor.demo --out <path>` builds a synthetic one --
# and the suite below runs against it instead of skipping.
REAL_VAULT = Path(
    os.environ.get("HA_TEST_VAULT", sb.REPO_ROOT / "data" / "health.db"))

# Two real paths inside the invoking user's home directory, for the probes that
# must attempt something genuinely outside the sandbox. They are resolved HERE
# and baked into the child's source as literals on purpose: the TOCTOU probes
# below rewrite $HOME before expanding it, so a `~` expanded inside the sandbox
# would name the remapped root and test nothing. Nothing here is machine- or
# user-specific -- the corpus runs identically for whoever invokes it.
HOME_WRITE_PROBE = os.path.join(os.path.expanduser("~"), "pwned_by_sandbox_test.txt")
HOME_READ_PROBE = os.path.join(os.path.expanduser("~"), ".zshrc")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def executor():
    """The substrate-aware seam: seatbelt on macOS, bubblewrap on Linux.

    Instantiating SeatbeltExecutor unconditionally made this whole file
    macOS-only, so on a Linux runner with a vault present it errored rather
    than exercising the executor that platform actually ships.
    """
    if sys.platform == "darwin":
        return sb.SeatbeltExecutor()
    if sys.platform.startswith("linux"):
        return sb.BwrapExecutor()
    pytest.skip(f"no sandbox executor for platform {sys.platform!r}")


@pytest.fixture
def vault_path() -> str:
    """The real vault at `data/health.db`, read-only throughout this file.

    Never opened writable by anything in this module — the whole point of the
    corpus is that the *sandboxed child* cannot write it either, and asserting
    that against a stand-in built inside the test would prove nothing about the
    real permission bits (Done when 2). A `tmp_path` toy would also be too
    small to time anything: `test_sandbox_overhead_vs_bare_python` and
    `test_normal_vault_read_still_works` both need a populated database.

    No vault ships with this repository, so every test that asks for one skips
    rather than failing. The guard is existence alone: drop a real vault (or a
    generated demo one — `python -m health_advisor.demo`) at that path and the
    whole corpus runs again with no edit here. Do NOT replace this with a
    synthetic fixture; that would turn a measurement into a configuration test,
    which is exactly what §2.6 rejected.
    """
    if not REAL_VAULT.exists():
        pytest.skip(
            f"requires a populated vault at {REAL_VAULT}; generate one with "
            "`python -m health_advisor.demo` (see the README roadmap) and these "
            "tests run automatically")
    return str(REAL_VAULT)


@pytest.fixture
def query_vault_path(tmp_path) -> str:
    """A small synthetic demo vault for query-channel assertions."""
    from health_advisor.demo import build_demo_vault

    path = tmp_path / "demo.db"
    build_demo_vault(path, days=2)
    return str(path)


def _run(
    executor: sb.SeatbeltExecutor,
    code: str,
    vault: str,
    run_dir: Path,
    **limit_kwargs,
) -> sb.RawResult:
    limits = sb.RunLimits(**limit_kwargs) if limit_kwargs else None
    return executor.run(code, vault, str(run_dir), limits=limits)


def _is_unsafe(
    res: sb.RawResult, *, host_artifact: Path | None = None
) -> bool:
    """Return whether a probe produced a host-visible escape.

    Namespace-sensitive filesystem probes must provide the parent-chosen host
    path they attempted to affect. Their child output is deliberately ignored:
    bubblewrap can report a successful write to a namespace-local path that is
    not an escape. Probes without a host artifact retain the original
    stdout/fd-3 convention.
    """
    if host_artifact is not None:
        return os.path.exists(host_artifact)

    haystack = res.stdout + res.fd3_bytes
    return b"UNSAFE" in haystack


def test_parent_side_host_artifact_scoring_ignores_namespace_local_write(tmp_path):
    artifact = tmp_path / "host_artifact.txt"
    result = sb.RawResult(
        fd3_bytes=b"",
        fd3_oversized=False,
        stdout=b"UNSAFE wrote a namespace-local file",
        stderr=b"",
        returncode=0,
        timed_out=False,
        killed_group=False,
        duration_s=0.0,
        pgid=0,
        run_dir=tmp_path,
        reaped=True,
    )

    assert _is_unsafe(result, host_artifact=artifact) is False
    artifact.write_text("x", encoding="utf-8")
    assert _is_unsafe(result, host_artifact=artifact) is True


# --------------------------------------------------------------------------- #
# Scoring: BLOCKED has to be earned (#109)
#
# `_is_unsafe` answers "did the attack succeed?" and nothing else, so a probe
# that never reached its attack (NameError, failed import, missing interpreter)
# used to be counted BLOCKED. A row is now BLOCKED only on positive evidence
# that the probe ran inside the sandbox and was stopped there BY THE SANDBOX:
# every probe prints BLOCKED from a bare `except Exception`, so the marker alone
# proves only that the attack failed. The line must also carry the sandbox's
# own denial reason for that row and executor (`_DENIAL` below).
# --------------------------------------------------------------------------- #


class _EscapeStillOpen(AssertionError):
    """The desired property of a KNOWN_GAP does not hold (yet)."""


BLOCKED = "BLOCKED"
UNSAFE = "UNSAFE"
NOT_CONFIRMED = "NOT CONFIRMED"
NOT_APPLICABLE = "NOT APPLICABLE"


def _score_case(
    res: sb.RawResult,
    *,
    cls: str,
    host_artifact: Path | None = None,
    kill_expected: bool = False,
    network_control_ok: bool = True,
    evidence: bytes | None = None,
    target_present: bool = True,
    control_ok: bool | None = None,
    control_failure: str = NOT_CONFIRMED,
) -> str:
    """Classify one corpus run as BLOCKED, UNSAFE, NOT APPLICABLE or NOT CONFIRMED.

    UNSAFE is whatever `_is_unsafe` says, and is checked first, so nothing
    below can hide an escape. BLOCKED needs positive evidence that the probe
    executed inside the sandbox and was stopped there BY THE SANDBOX:

    (i)  the child exited 0, printed its own BLOCKED marker, and the line
         carries `evidence`, the sandbox's own denial reason for this row and
         executor (Seatbelt answers EPERM, "Operation not permitted"; the
         query channel's parent-side authorizer answers "not authorized"). A
         probe stopped by anything else -- host permissions (EACCES), a
         missing target (ENOENT, under Seatbelt), a probe bug -- prints
         BLOCKED from the same bare `except Exception`, and does not count.
         With no `evidence` nothing counts. For a row with a parent-chosen
         host artifact the child's UNSAFE line also proves it ran (a
         bubblewrap namespace-local write is not an escape) as long as the
         artifact is absent;
    (ii) the row is `kill_expected` (the forked-descendant rows, which spin on
         purpose so the deadline has to do the work): the run timed out, its
         process group was confirmed killed, and nothing printed UNSAFE.

    NOT APPLICABLE is earned too, and means the row measures nothing on this
    run: the host target does not exist for this user (`target_present`), or
    the unsandboxed positive control failed for a row whose `control_failure`
    says that is a property of the host (dyld injection on a host that strips
    it). The roll-up pins which rows may land here.

    Anything else -- a crash, a missing interpreter, a profile that failed to
    load, a timeout nobody expected, a denial with the wrong reason -- is NOT
    CONFIRMED. Network rows additionally need the positive control: on an
    offline host every connect fails, sandbox or not.
    """
    if _is_unsafe(res, host_artifact=host_artifact):
        return UNSAFE
    if cls == "network" and not network_control_ok:
        return NOT_CONFIRMED
    if kill_expected:
        if res.timed_out and res.killed_group:
            return BLOCKED
        return NOT_CONFIRMED
    if res.timed_out or res.returncode != 0:
        return NOT_CONFIRMED
    haystack = res.stdout + res.fd3_bytes
    if not target_present:
        return NOT_APPLICABLE
    if control_ok is False:
        return control_failure
    if host_artifact is not None and b"UNSAFE" in haystack:
        return BLOCKED
    if b"BLOCKED" not in haystack or evidence is None:
        return NOT_CONFIRMED
    return BLOCKED if re.search(evidence, haystack) else NOT_CONFIRMED


def _network_control() -> bool:
    """Run the network attacks' own operations OUTSIDE the sandbox.

    The network rows print BLOCKED for any exception, so on an offline runner
    all five would pass without the sandbox having done anything. If the
    unsandboxed host cannot connect and resolve, those rows prove nothing.
    """
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        socket.gethostbyname("openrouter.ai")
    except OSError:
        return False
    return True


def _raw(tmp_path: Path, **overrides) -> sb.RawResult:
    fields = dict(
        fd3_bytes=b"", fd3_oversized=False, stdout=b"", stderr=b"",
        returncode=0, timed_out=False, killed_group=False, duration_s=0.0,
        pgid=0, run_dir=tmp_path, reaped=True,
    )
    fields.update(overrides)
    return sb.RawResult(**fields)


# --------------------------------------------------------------------------- #
# The sandbox's own denial, per row and executor
#
# SEATBELT: measured by running every corpus row under SeatbeltExecutor on
# macOS and reading its line. Seatbelt answers a denied file, socket or exec
# with EPERM ("Operation not permitted"); host permissions answer EACCES and a
# missing path ENOENT, which is how they are told apart. The row entries are
# the measured lines of probes that surface no errno.
#
# BWRAP: measured by running the corpus under BwrapExecutor in the Linux CI
# job. A host path that is not bound answers ENOENT, a read-only bind EROFS,
# the unshared network ENETUNREACH.
#
# QUERY CHANNEL: the rows that attack `conn` run through the analyst query
# channel, where `conn` is a proxy to a parent-owned, ledgered, read-only
# connection. The denial there is the parent's SQLite authorizer, identical on
# both executors, and it must also appear in that connection's own record of
# refused actions (`_QUERY_DENIED_ACTION`).
# --------------------------------------------------------------------------- #

EPERM = rb"\[Errno 1\] Operation not permitted"
_NOT_AUTHORIZED = rb"BLOCKED AnalystQueryError DatabaseError: not authorized"

_DENIAL_DEFAULT = {
    "seatbelt": EPERM,
    "bwrap": rb"\[Errno (?:1|2|30)\] ",
}

_DENIAL = {
    ("network", "dns_resolution"): {
        "seatbelt": rb"BLOCKED gaierror \[Errno 8\] ", "bwrap": rb"BLOCKED gaierror "},
    ("network", "curl_child_process"): {
        "seatbelt": rb"BLOCKED rc= 7 ", "bwrap": rb"BLOCKED rc= (?:6|7) "},
    ("network", "outbound_tcp_connect"): {"bwrap": rb"\[Errno (?:1|101)\] "},
    ("network", "udp_sendto"): {"bwrap": rb"\[Errno (?:1|101)\] "},
    # SQLite's SQLITE_AUTH for an open the profile refused (the vault is under
    # $HOME). NOT "attempt to write a readonly database": that is what the
    # same write gets unsandboxed against the 0444 vault, and also what a
    # sandboxed write to a DAC-writable file outside work/ gets.
    ("vault_integrity", "writable_open_and_create_table"): {
        "seatbelt": rb"BLOCKED DatabaseError authorization denied",
        "bwrap": rb"BLOCKED OperationalError unable to open database file"},
    ("vault_integrity", "pragma_journal_mode_wal"): {
        "seatbelt": _NOT_AUTHORIZED, "bwrap": _NOT_AUTHORIZED},
    ("vault_integrity", "attach_and_write"): {
        "seatbelt": _NOT_AUTHORIZED, "bwrap": _NOT_AUTHORIZED},
    # VACUUM INTO attaches its target internally; the authorizer refuses that
    # ATTACH while the statement runs, which SQLite reports as SQLITE_AUTH.
    ("vault_integrity", "vacuum_into_exfil"): {
        "seatbelt": rb"BLOCKED AnalystQueryError DatabaseError: authorization denied",
        "bwrap": rb"BLOCKED AnalystQueryError DatabaseError: authorization denied"},
    ("inherited_descriptors", "enumerate_and_read_beyond_known"): {
        "seatbelt": rb"BLOCKED no unexpected readable fd", "bwrap": rb"BLOCKED no unexpected readable fd"},
    ("inherited_descriptors", "parent_only_secret_not_leaked"): {
        "seatbelt": rb"BLOCKED no secret leak", "bwrap": rb"BLOCKED no secret leak"},
    ("inherited_descriptors", "raw_vault_fd_write_bypassing_conn"): {
        "seatbelt": rb"BLOCKED no descriptor refers to the vault",
        "bwrap": rb"BLOCKED no descriptor refers to the vault"},
    ("binaries", "setuid_sudo"): {
        "bwrap": rb"effective uid is not 0|nosuid|no new privileges|\[Errno 1\] "},
    ("binaries", "setuid_ping_raw_socket"): {
        "seatbelt": rb"sendto: Operation not permitted",
        "bwrap": rb"Network is unreachable|Operation not permitted"},
    ("binaries", "dyld_insert_libraries_injection"): {
        "seatbelt": rb"BLOCKED no injection observed", "bwrap": rb"BLOCKED no injection observed"},
    ("binaries", "system_binary_reads_forbidden_path"): {
        "seatbelt": rb"Operation not permitted",
        "bwrap": rb"No such file or directory|Operation not permitted"},
}

# The authorizer action each SQL-level query-channel row must have been refused
# with, as recorded by the parent's ledgered connection.
_QUERY_DENIED_ACTION = {
    ("vault_integrity", "pragma_journal_mode_wal"): sqlite3.SQLITE_PRAGMA,
    ("vault_integrity", "attach_and_write"): sqlite3.SQLITE_ATTACH,
    ("vault_integrity", "vacuum_into_exfil"): sqlite3.SQLITE_ATTACH,
}


def _denial_evidence(cls: str, name: str, kind: str) -> bytes:
    """The line a BLOCKED verdict for this row must carry under this executor."""
    return _DENIAL.get((cls, name), {}).get(kind, _DENIAL_DEFAULT[kind])


def _host_targets(vault_real: str) -> dict[tuple[str, str], str]:
    """What each row attacks on the host. A row whose target this user cannot
    see (`os.path.lexists` is False) measures nothing: /var/root/.bash_history
    is invisible to a non-root user, and /Users/Shared or /dev/rdisk0 do not
    exist on Linux. Such a row is NOT APPLICABLE unless it printed UNSAFE."""
    return {
        ("network", "unix_domain_socket"): "/var/run/mDNSResponder",
        ("network", "curl_child_process"): "/usr/bin/curl",
        ("filesystem_write", "write_into_repo"): str(sb.REPO_ROOT),
        ("vault_integrity", "writable_open_and_create_table"): vault_real,
        ("binaries", "setuid_sudo"): "/usr/bin/sudo",
        ("binaries", "setuid_ping_raw_socket"): "/sbin/ping",
        ("binaries", "dyld_insert_libraries_injection"): "/usr/lib/libgmalloc.dylib",
        ("binaries", "system_binary_reads_forbidden_path"): HOME_READ_PROBE,
        ("paths", "users_shared_dir_readable"): "/Users/Shared",
        ("paths", "another_users_home_var_root"): "/var/root/.bash_history",
        ("paths", "device_file_read"): "/dev/rdisk0",
        ("paths", "device_file_write"): "/dev/rdisk0",
    }


# Rows whose BLOCKED line reports an ABSENCE (no fd leaked, no library loaded,
# no descriptor on the vault) prove something only if the same probe, run
# OUTSIDE the sandbox with the thing it looks for handed to it, finds it.
# Read-only probes only: no write probe is ever run unsandboxed. The value is
# the verdict when the control fails: a probe that cannot see a leak that is
# there is broken (NOT CONFIRMED); a host that strips DYLD_* from the binary
# itself (SIP) cannot measure the profile at all.
_CONTROLLED = {
    ("inherited_descriptors", "enumerate_and_read_beyond_known"): NOT_CONFIRMED,
    ("inherited_descriptors", "parent_only_secret_not_leaked"): NOT_CONFIRMED,
    ("inherited_descriptors", "raw_vault_fd_write_bypassing_conn"): NOT_CONFIRMED,
    ("binaries", "dyld_insert_libraries_injection"): NOT_APPLICABLE,
}


def _unsandboxed_control(code: str, work: Path, readable_fd_path: Path | str) -> bool:
    """Run a read-only probe with no sandbox, handing it one readable fd above
    4 (what a leak would look like); True when it reports UNSAFE there."""
    work.mkdir(parents=True, exist_ok=True)
    fd = os.open(readable_fd_path, os.O_RDONLY)
    while fd <= 4:
        higher = os.dup(fd)
        os.close(fd)
        fd = higher
    try:
        done = subprocess.run([sys.executable, "-c", "conn = None\n" + code], cwd=work,
                              pass_fds=(fd,), capture_output=True, timeout=30)
    finally:
        os.close(fd)
    return done.returncode == 0 and b"UNSAFE" in done.stdout


def _run_query_channel(
    executor, code: str, vault: str, run_dir: Path, **limit_kwargs,
) -> tuple[sb.RawResult, tuple[int, ...]]:
    """Run `code` through the analyst query channel and return the child's raw
    result with the authorizer actions the parent's connection refused.

    The same dispatch `analyst_runner.run_analyst_code` uses: the parent opens
    the vault read-only behind the ledgered wrapper, the child gets the `conn`
    proxy over a socket, and every request is served by `_service_query`. The
    envelope step is left out because these rows are scored on the child's
    own line, which `run_analyst_code` does not return.
    """
    limits = analyst_runner.AnalystLimits(**limit_kwargs) if limit_kwargs else None
    parent_conn = analyst_ledger.open_ledgered(vault)
    query_parent, query_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    query_child_fd = query_child.detach()
    outcome: list[sb.RawResult] = []
    failure: list[BaseException] = []

    def _worker() -> None:
        try:
            outcome.append(analyst_runner._invoke_executor(
                executor, code, str(run_dir), query_child_fd, limits, vault))
        except BaseException as exc:      # reported below, never swallowed
            failure.append(exc)
        finally:
            try:
                os.close(query_child_fd)
            except OSError:
                pass

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    query_parent.setblocking(False)
    pending = bytearray()
    try:
        while worker.is_alive():
            ready, _, _ = select.select([query_parent], [], [], 0.25)
            if not ready:
                continue
            try:
                more, _reason = analyst_runner._service_query(
                    query_parent, parent_conn, analyst_runner.DEFAULT_MAX_QUERY_ROWS, pending)
            except (ConnectionError, analyst_runner._AnalystQueryError, ValueError):
                break
            if not more:
                break
    finally:
        query_parent.close()
        worker.join(timeout=30)
        denied = tuple(action for action, _arg in parent_conn.denied_events)
        parent_conn.close()
    if failure:
        raise failure[0]
    assert outcome, "the query-channel executor returned no result"
    return outcome[0], denied


def test_probe_that_crashed_before_its_attack_is_not_confirmed(tmp_path):
    crashed = _raw(tmp_path, returncode=-1, stderr=b"NameError: name 'x'")
    assert _score_case(crashed, cls="filesystem_write", evidence=EPERM) == NOT_CONFIRMED


def test_probe_that_printed_nothing_is_not_confirmed(tmp_path):
    assert _score_case(_raw(tmp_path), cls="paths", evidence=EPERM) == NOT_CONFIRMED


def test_probe_that_exited_nonzero_after_printing_blocked_is_not_confirmed(tmp_path):
    odd = _raw(tmp_path, returncode=1,
               stdout=b"BLOCKED PermissionError [Errno 1] Operation not permitted")
    assert _score_case(odd, cls="paths", evidence=EPERM) == NOT_CONFIRMED


def test_probe_that_reported_the_sandbox_denial_and_exited_cleanly_is_blocked(tmp_path):
    ok = _raw(tmp_path, stdout=b"BLOCKED PermissionError [Errno 1] Operation not permitted")
    assert _score_case(ok, cls="paths", evidence=EPERM) == BLOCKED
    via_fd3 = _raw(tmp_path, fd3_bytes=b"BLOCKED [Errno 1] Operation not permitted")
    assert _score_case(via_fd3, cls="paths", evidence=EPERM) == BLOCKED
    # The marker alone is not enough: no evidence, or a different reason.
    assert _score_case(ok, cls="paths") == NOT_CONFIRMED
    bare = _raw(tmp_path, stdout=b"BLOCKED PermissionError denied")
    assert _score_case(bare, cls="paths", evidence=EPERM) == NOT_CONFIRMED


def test_an_unsafe_marker_wins_even_on_a_killed_run(tmp_path):
    bad = _raw(tmp_path, returncode=-9, timed_out=True, killed_group=True,
               fd3_bytes=b"UNSAFE descendant survived deadline")
    assert _score_case(bad, cls="forked_descendants",
                       kill_expected=True) == UNSAFE


def test_kill_expected_row_is_blocked_only_by_a_confirmed_group_kill(tmp_path):
    killed = _raw(tmp_path, returncode=-9, timed_out=True, killed_group=True)
    assert _score_case(killed, cls="forked_descendants",
                       kill_expected=True) == BLOCKED
    # Timed out but no confirmed group kill.
    unkilled = _raw(tmp_path, returncode=-9, timed_out=True, killed_group=False)
    assert _score_case(unkilled, cls="forked_descendants",
                       kill_expected=True) == NOT_CONFIRMED
    # Exited by itself: the deadline never did the work this row exists for.
    early = _raw(tmp_path, returncode=0)
    assert _score_case(early, cls="forked_descendants",
                       kill_expected=True) == NOT_CONFIRMED


def test_a_timeout_is_not_blocked_unless_the_row_expects_a_kill(tmp_path):
    killed = _raw(tmp_path, returncode=-9, timed_out=True, killed_group=True)
    assert _score_case(killed, cls="paths", evidence=EPERM) == NOT_CONFIRMED


def test_host_artifact_row_needs_proof_the_probe_ran(tmp_path):
    artifact = tmp_path / "host_artifact.txt"
    ran = _raw(tmp_path, stdout=b"BLOCKED PermissionError [Errno 1] Operation not permitted")
    assert _score_case(ran, cls="filesystem_write", host_artifact=artifact,
                       evidence=EPERM) == BLOCKED
    # A bubblewrap namespace-local write prints UNSAFE but leaves the host
    # untouched: the probe demonstrably ran, and nothing escaped.
    local = _raw(tmp_path, stdout=b"UNSAFE wrote a namespace-local file")
    assert _score_case(local, cls="filesystem_write", host_artifact=artifact,
                       evidence=EPERM) == BLOCKED
    silent = _raw(tmp_path)
    assert _score_case(silent, cls="filesystem_write", host_artifact=artifact,
                       evidence=EPERM) == NOT_CONFIRMED
    artifact.write_text("x", encoding="utf-8")
    assert _score_case(ran, cls="filesystem_write", host_artifact=artifact,
                       evidence=EPERM) == UNSAFE


NETWORK_ROWS = {"outbound_tcp_connect", "dns_resolution", "curl_child_process",
                "unix_domain_socket", "udp_sendto"}


def test_network_rows_are_not_confirmed_when_the_positive_control_fails(
        tmp_path, vault_path):
    cases = _corpus(tmp_path, vault_path)
    network = [c for c in cases if c["class"] == "network"]
    assert {c["name"] for c in network} == NETWORK_ROWS
    blocked = _raw(tmp_path, stdout=b"BLOCKED PermissionError [Errno 1] Operation not permitted")
    for case in network:
        assert _score_case(blocked, cls="network", evidence=EPERM,
                           network_control_ok=True) == BLOCKED
        assert _score_case(blocked, cls="network", evidence=EPERM,
                           network_control_ok=False) == NOT_CONFIRMED
    # Other classes do not depend on the network control.
    assert _score_case(blocked, cls="paths", evidence=EPERM,
                       network_control_ok=False) == BLOCKED


def test_network_control_fails_when_the_host_cannot_connect(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(socket, "create_connection", refuse)
    assert _network_control() is False


def test_network_control_fails_when_the_host_cannot_resolve(monkeypatch):
    class _Conn:
        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _Conn())

    def refuse(*args, **kwargs):
        raise socket.gaierror("no dns")

    monkeypatch.setattr(socket, "gethostbyname", refuse)
    assert _network_control() is False


def test_kill_expected_is_set_on_exactly_the_three_forked_rows(tmp_path, vault_path):
    cases = _corpus(tmp_path, vault_path)
    flagged = {c["name"] for c in cases if c.get("kill_expected")}
    assert flagged == {"descendant_holds_fd3_open",
                       "descendant_holds_vault_conn_open",
                       "three_plain_descendants"}


# The read probe every `paths` row uses, run here with NO sandbox at all.
_UNSANDBOXED_READ_PROBE = """
try:
    with open(PATH) as f:
        print("UNSAFE", f.read()[:20])
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
"""


@pytest.mark.parametrize("kind", ["seatbelt", "bwrap"])
def test_a_probe_stopped_by_host_permissions_is_not_blocked(tmp_path, kind):
    """An unsandboxed child that gets EACCES on a mode-0 file prints BLOCKED
    from its bare `except`. Run for real, outside any sandbox: it must not
    score BLOCKED under either executor's evidence."""
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-0 file; the case needs an ordinary user")
    target = tmp_path / "mode0.txt"
    target.write_text("x")
    target.chmod(0)
    done = subprocess.run([sys.executable, "-c", f"PATH = {str(target)!r}\n" + _UNSANDBOXED_READ_PROBE],
                          capture_output=True, timeout=30)
    assert done.stdout.startswith(b"BLOCKED PermissionError [Errno 13] Permission denied"), done.stdout
    res = _raw(tmp_path, stdout=done.stdout)
    evidence = _denial_evidence("paths", "another_users_home_var_root", kind)
    assert _score_case(res, cls="paths", evidence=evidence) == NOT_CONFIRMED
    assert _score_case(res, cls="paths") == NOT_CONFIRMED            # no evidence, no BLOCKED


def test_a_missing_target_is_not_a_seatbelt_denial(tmp_path):
    done = subprocess.run([sys.executable, "-c", f"PATH = {str(tmp_path / 'absent')!r}\n" + _UNSANDBOXED_READ_PROBE],
                          capture_output=True, timeout=30)
    assert done.stdout.startswith(b"BLOCKED FileNotFoundError [Errno 2] No such file or directory"), done.stdout
    res = _raw(tmp_path, stdout=done.stdout)
    assert _score_case(res, cls="paths", evidence=_denial_evidence("paths", "device_file_read", "seatbelt")) \
        == NOT_CONFIRMED
    # A target this user cannot see measures nothing, whatever the line says.
    assert _score_case(res, cls="paths", evidence=EPERM, target_present=False) == NOT_APPLICABLE


def test_the_measured_seatbelt_denial_is_blocked(tmp_path):
    """The literal line the device-read row printed under Seatbelt."""
    res = _raw(tmp_path, stdout=b"BLOCKED PermissionError [Errno 1] Operation not permitted: '/dev/rdisk0'\n")
    assert _score_case(res, cls="paths", evidence=_denial_evidence("paths", "device_file_read", "seatbelt")) \
        == BLOCKED


def test_not_applicable_needs_its_proof_and_never_hides_an_escape(tmp_path):
    missing = _raw(tmp_path, stdout=b"BLOCKED FileNotFoundError [Errno 2] No such file or directory\n")
    assert _score_case(missing, cls="paths", evidence=EPERM, target_present=False) == NOT_APPLICABLE
    assert _score_case(missing, cls="paths", evidence=EPERM, target_present=True) == NOT_CONFIRMED
    escaped = _raw(tmp_path, stdout=b"UNSAFE listed ['x']\n")
    assert _score_case(escaped, cls="paths", evidence=EPERM, target_present=False) == UNSAFE
    assert _score_case(escaped, cls="binaries", evidence=EPERM, control_ok=False,
                       control_failure=NOT_APPLICABLE) == UNSAFE
    # A run that did not finish cleanly is never NOT APPLICABLE.
    crashed = _raw(tmp_path, returncode=1, stdout=missing.stdout)
    assert _score_case(crashed, cls="paths", evidence=EPERM, target_present=False) == NOT_CONFIRMED


def test_an_absence_row_counts_only_with_its_unsandboxed_control(tmp_path):
    line = _raw(tmp_path, stdout=b"BLOCKED no secret leak\n")
    evidence = _denial_evidence("inherited_descriptors", "parent_only_secret_not_leaked", "seatbelt")
    assert _score_case(line, cls="inherited_descriptors", evidence=evidence, control_ok=True) == BLOCKED
    assert _score_case(line, cls="inherited_descriptors", evidence=evidence, control_ok=False) == NOT_CONFIRMED
    dyld = _raw(tmp_path, stdout=b"BLOCKED no injection observed 0 b'hi\\n'\n")
    assert _score_case(dyld, cls="binaries", evidence=rb"BLOCKED no injection observed", control_ok=False,
                       control_failure=NOT_APPLICABLE) == NOT_APPLICABLE


def test_the_fd_probes_see_a_leak_when_there_is_one(tmp_path):
    """The positive control the `inherited_descriptors` absence rows rest on:
    outside the sandbox, handed the readable fd each one looks for, each probe
    reports UNSAFE. The vault-descriptor probe is handed a descriptor on its
    own target, a stand-in vault that is never written."""
    secret = tmp_path / "parent_secret.txt"
    stand_in_vault = tmp_path / "stand_in_vault.db"
    stand_in_vault.write_bytes(b"not a real vault")
    cases = {c["name"]: c for c in _corpus(tmp_path, str(stand_in_vault), str(stand_in_vault))}
    for name in ("enumerate_and_read_beyond_known", "parent_only_secret_not_leaked"):
        assert _unsandboxed_control(cases[name]["code"], tmp_path / name, secret), name
    vault_row = cases["raw_vault_fd_write_bypassing_conn"]
    assert _unsandboxed_control(vault_row["code"], tmp_path / "vault_fd", stand_in_vault)
    # And it does not cry wolf: a descriptor on some other file is not the vault.
    assert not _unsandboxed_control(vault_row["code"], tmp_path / "other_fd", secret)
    assert stand_in_vault.read_bytes() == b"not a real vault"


def test_query_channel_rows_are_exactly_the_conn_and_vault_descriptor_rows(tmp_path, vault_path):
    """The rows that attack `conn` or a vault descriptor run where a `conn`
    exists. On the plain path the child is handed neither, so there they
    could only ever fail for want of a target."""
    cases = _corpus(tmp_path, vault_path)
    query_rows = {(c["class"], c["name"]) for c in cases if c["channel"] == "query"}
    assert query_rows == {
        ("vault_integrity", "pragma_journal_mode_wal"),
        ("vault_integrity", "attach_and_write"),
        ("vault_integrity", "vacuum_into_exfil"),
        ("inherited_descriptors", "raw_vault_fd_write_bypassing_conn"),
    }
    assert set(_QUERY_DENIED_ACTION) <= query_rows
    plain_conn_users = [c["name"] for c in cases
                        if c["channel"] == "plain" and "conn.execute" in c["code"]
                        and "if conn is not None" not in c["code"]]
    # The toctou placeholder is scored by its own query-channel check below.
    assert plain_conn_users == ["vault_symlink_resolved_consistently"]


# =========================================================================== #
# Done when 2 — the vault's real permission bits
# =========================================================================== #


def test_vault_realpath_mode_is_0o444(vault_path):
    real = os.path.realpath(vault_path)
    mode = os.stat(real).st_mode & 0o777
    assert mode == 0o444, f"expected 0o444, measured {oct(mode)} on {real}"


# =========================================================================== #
# Done when 3/4 — process-group kill, lock release, wall-clock timeout
# =========================================================================== #


def _pgrep_group_survivors(pgid: int) -> list[str]:
    r = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True)
    return [line for line in r.stdout.splitlines() if line.strip()]


def test_process_group_kill_leaves_no_survivors_and_releases_a_lock(executor, vault_path, tmp_path):
    """§3.5 / review finding 6: killing only the direct child leaves a forked
    grandchild free to keep running — and, if it is the one holding a SQLite
    write transaction open, free to keep blocking a writer. This test makes
    a descendant actually acquire a write lock (on a file inside the
    child-writable `work/`, since the vault itself is never writable even
    from inside the sandbox — that grant does not exist) and confirms both
    that `os.killpg` reaches it and that the lock is gone afterward.
    """
    run_dir = tmp_path / "killtest"
    code = """
import os, sqlite3, time

conn2 = sqlite3.connect("held.db")
conn2.execute("CREATE TABLE t (a int)")
conn2.commit()

pid = os.fork()
if pid == 0:
    # Grandchild: open its own handle to the same file, hold a RESERVED
    # lock via an uncommitted write, and outlive any *direct-child* kill.
    c = sqlite3.connect("held.db")
    c.execute("BEGIN IMMEDIATE")
    c.execute("INSERT INTO t VALUES (1)")
    time.sleep(120)
    os._exit(0)

# The direct process also spins, so it needs the timeout too, not just a
# natural exit — matching review finding 6's actual shape (a live process
# at the moment of the kill, not one that already returned).
while True:
    pass
"""
    limits = sb.RunLimits(wall_clock_s=3.0)
    t0 = time.monotonic()
    res = _run(executor, code, vault_path, run_dir, wall_clock_s=limits.wall_clock_s)
    kill_elapsed_s = time.monotonic() - t0

    assert res.timed_out is True
    assert res.killed_group is True

    survivors = _pgrep_group_survivors(res.pgid)
    survivor_count = len(survivors)

    held_db = run_dir / "work" / "held.db"
    assert held_db.exists()
    t1 = time.monotonic()
    conn = sqlite3.connect(str(held_db), timeout=5.0)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("INSERT INTO t VALUES (2)")
    conn.commit()
    conn.close()
    lock_release_probe_s = time.monotonic() - t1

    print(
        f"\n[Done-when 3] kill_elapsed={kill_elapsed_s:.3f}s "
        f"survivors={survivor_count} lock_release_probe={lock_release_probe_s:.4f}s"
    )
    assert survivor_count == 0, f"pgrep -g {res.pgid} found survivors: {survivors}"
    assert lock_release_probe_s < 5.0


def test_default_wall_clock_constant_is_60s():
    """Fast guard against `DEFAULT_WALL_CLOCK_S` silently drifting.

    Issue #26: the timing test below no longer exercises the production
    default end to end (that cost a real 60s sleep in every default run with
    a vault present). The deadline/kill logic in `SeatbeltExecutor.run` and
    `BwrapExecutor.run` does not branch on the configured value — `deadline =
    start + limits.wall_clock_s` either way — so a short configured timeout
    proves the mechanism just as well. What a short-timeout test *cannot*
    catch is someone changing the shipped default itself; this assertion is
    the whole cost of covering that, and it needs no executor, no vault, and
    no sleep.
    """
    assert sb.DEFAULT_WALL_CLOCK_S == 60.0


def test_wall_clock_configured_timeout_kills_within_tolerance(executor, vault_path, tmp_path):
    """Done when 4 (A1): the kill mechanism, measured for real, on
    `while True: pass` — against a short *configured* timeout rather than the
    60s production default (issue #26). See `test_default_wall_clock_constant_is_60s`
    for the default value itself."""
    run_dir = tmp_path / "configured_timeout"
    code = "while True:\n    pass\n"
    configured_s = 2.0
    t0 = time.monotonic()
    res = _run(executor, code, vault_path, run_dir, wall_clock_s=configured_s)
    elapsed = time.monotonic() - t0

    print(f"\n[Done-when 4] configured-timeout({configured_s}s) kill_elapsed={elapsed:.2f}s")
    assert res.timed_out is True
    assert res.killed_group is True
    assert configured_s - 0.5 <= elapsed <= configured_s + 2.5, (
        f"expected {configured_s}s (+2.5s/-0.5s), measured {elapsed:.2f}s"
    )
    survivors = _pgrep_group_survivors(res.pgid)
    assert survivors == []


# =========================================================================== #
# Done when 5 — sandbox overhead vs. bare `python -c pass`
# =========================================================================== #


def test_sandbox_overhead_vs_bare_python(executor, vault_path, tmp_path):
    n = 20
    bare_ms = []
    sandboxed_ms = []

    for i in range(n):
        t0 = time.monotonic()
        subprocess.run([executor._python, "-I", "-c", "pass"], check=True)
        bare_ms.append((time.monotonic() - t0) * 1000)

    for i in range(n):
        run_dir = tmp_path / f"overhead_{i}"
        t0 = time.monotonic()
        _run(executor, "pass\n", vault_path, run_dir)
        sandboxed_ms.append((time.monotonic() - t0) * 1000)

    bare_median = statistics.median(bare_ms)
    sandboxed_median = statistics.median(sandboxed_ms)
    overhead_ms = sandboxed_median - bare_median

    print(
        f"\n[Done-when 5] bare_median={bare_median:.2f}ms "
        f"sandboxed_median={sandboxed_median:.2f}ms overhead={overhead_ms:.2f}ms (n={n})"
    )
    # Sanity bound only — this is a measurement to report, not a strict gate.
    # §2.3 measured ~10ms; a factor-of-50 regression would indicate something
    # is structurally wrong (e.g. the profile forcing a slow path), not
    # ordinary machine-to-machine variance.
    assert overhead_ms < 500


# =========================================================================== #
# MUST IMPLEMENT — the vault handle is opened by the parent; generated code
# never receives a path (§3.1, stronger than T-003)
# =========================================================================== #


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt query channel")
def test_generated_code_receives_conn_not_a_path(query_vault_path, tmp_path):
    code = """
assert callable(conn.execute)
assert not any(
    isinstance(value, str) and ("health.db" in value or value == %r)
    for value in globals().values()
)
row = conn.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()
emit("connection_check", ["row_count"], ["count"], [[row[0]]])
""" % (os.path.realpath(query_vault_path),)
    res = analyst_runner.run_analyst_code(
        code, query_vault_path, str(tmp_path / "no_path_leak"),
        sb.SeatbeltExecutor())
    assert isinstance(res, analyst_envelope.Envelope)
    assert res.tables[0]["name"] == "connection_check"
    assert res.tables[0]["rows"][0][0] > 0


def _seatbelt_query_probe(code: str, bound_root: Path, tmp_path: Path):
    """Invoke the same Seatbelt dispatch used by ``run_analyst_code``."""
    executor = sb.SeatbeltExecutor(pkg_dir=str(bound_root))
    # Make the synthetic bound root the denied home subtree. The run directory
    # remains outside it, so the bootstrap itself can still be read.
    executor._real_home = os.path.realpath(bound_root)
    query_r, query_w = os.pipe()
    try:
        return analyst_runner._invoke_executor(
            executor, code, str(tmp_path / "run"), query_r,
            sb.RunLimits(wall_clock_s=5.0), str(bound_root / "data" / "demo.db"),
        )
    finally:
        for fd in (query_r, query_w):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt regression")
def test_seatbelt_query_channel_denies_secret_in_bound_root(tmp_path):
    bound_root = tmp_path / "bound-root"
    secret_path = bound_root / "systemd" / "receiver.env"
    secret_path.parent.mkdir(parents=True)
    secret_path.write_text("synthetic-secret", encoding="utf-8")

    result = _seatbelt_query_probe(
        f"""
try:
    open({str(secret_path)!r}, "rb").read()
    print("UNSAFE secret")
except Exception:
    print("BLOCKED secret")
""",
        bound_root,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert b"BLOCKED secret" in result.stdout
    assert b"UNSAFE secret" not in result.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt regression")
def test_seatbelt_query_channel_denies_vault_in_bound_root(tmp_path):
    bound_root = tmp_path / "bound-root"
    vault_path = bound_root / "data" / "demo.db"
    vault_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(str(vault_path))
    connection.execute("CREATE TABLE marker (value TEXT)")
    connection.execute("INSERT INTO marker VALUES ('synthetic')")
    connection.commit()
    connection.close()
    os.chmod(vault_path, 0o444)

    result = _seatbelt_query_probe(
        f"""
import sqlite3
try:
    direct = sqlite3.connect({str(vault_path)!r})
    direct.execute("SELECT name FROM sqlite_master").fetchall()
    direct.close()
    print("UNSAFE vault")
except Exception:
    print("BLOCKED vault")
""",
        bound_root,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert b"BLOCKED vault" in result.stdout
    assert b"UNSAFE vault" not in result.stdout


# =========================================================================== #
# Done when 1 — the attack corpus: 9 classes, each attempted -> blocked
# =========================================================================== #

# Two classes, each with one sub-case, this profile — implemented exactly as
# §2.5 specifies, with no tightening beyond it — does not fully close.
# `/etc/passwd` is already named in §2.3/§9.3 as a "named limit" of this
# exact configuration: the profile denies `file-read-data` only under
# `$HOME`, so anything outside `$HOME` falls through to the broad
# `(allow file-read*)` grant. `/Users/Shared` is a sibling of `$HOME`, not a
# subpath of it, so it is readable for the identical reason `/etc/passwd`
# is — this is the same gap, at a different path, not a new one. Recorded
# here rather than removed from the corpus (review finding 4's point
# exactly): a corpus that quietly drops the case it fails is worse than one
# that reports 8/9 classes at 100% and says which one is not, and why.
#
# The second entry (#2) is host-dependent rather than deterministic: measured
# fully blocked (4/4 `binaries` cases) on a current macOS/arm64 host, and
# NOT blocked (3/4, `DYLD_INSERT_LIBRARIES` injection succeeds, with
# `libgmalloc` observed loaded into the child) on GitHub's `macos-latest`
# runner image. That means this profile's denial of dyld injection relies on
# a protection present on one host and absent on the other, rather than on
# the profile itself -- see SECURITY.md and #2 for the open question of
# which. It does not reproduce under bubblewrap, so it is scoped to Seatbelt
# only below. The deficit is not the same on every affected host, so there is
# no fixed expected count to assert here the way there is for the `paths`
# entry above. There IS a floor, and asserting it is the whole difference
# between a named exception and a blanket one: at most ONE `binaries` case may
# be unblocked, and it must be this exact injection. See the assertion at the
# end of test_attack_corpus_class_coverage. Without that floor, listing the
# class in KNOWN_GAPS deletes its assertion outright and all four cases could
# regress while the corpus still reported green -- which is what #2's own
# `Done when` means by "not a blanket exemption".
KNOWN_GAPS = {
    ("paths", "users_shared_dir_readable"),
    ("binaries", "dyld_insert_libraries_injection"),
    ("process", "forked_setsid_descendant_escapes_process_group_kill"),
}

EXECUTOR_SCOPES = {
    "seatbelt-only": frozenset({"seatbelt"}),
    "bwrap-only": frozenset({"bwrap"}),
    "both": frozenset({"seatbelt", "bwrap"}),
}
KNOWN_GAP_EXECUTORS = {
    ("paths", "users_shared_dir_readable"): frozenset({"seatbelt"}),
    ("binaries", "dyld_insert_libraries_injection"): frozenset({"seatbelt"}),
    ("process", "forked_setsid_descendant_escapes_process_group_kill"):
        frozenset({"seatbelt"}),
}
SEATBELT_ONLY = "seatbelt-only"

# The one `binaries` case the Seatbelt gap (#2) covers. Naming it is what makes
# the KNOWN_GAPS entry an exception rather than an exemption: listing a class in
# KNOWN_GAPS removes its 100%-blocked assertion outright, so without this the
# other three cases could regress unnoticed.
BINARIES_GAP_CASE = "dyld_insert_libraries_injection"
SETSID_GAP = ("process", "forked_setsid_descendant_escapes_process_group_kill")


def test_setsid_known_gap_is_seatbelt_only():
    assert SETSID_GAP in KNOWN_GAPS
    assert KNOWN_GAP_EXECUTORS[SETSID_GAP] == frozenset({"seatbelt"})


def _binaries_gap_violations(details: list[str]) -> list[str]:
    """Reasons the observed `binaries` shortfall exceeds the named gap.

    An empty list means the shortfall is within #2's exception: either nothing
    unblocked (a host that denies the injection -- measured 4/4 on macOS 26.6.2
    arm64) or exactly the one named case (measured 3/4 on `macos-latest`). Both
    are expected; anything else is a regression the entry must not absorb.

    Pure so it can be exercised on a host where the gap does not reproduce --
    otherwise the assertion is vacuous on the developer machine and only ever
    evaluated where it cannot be tested.
    """
    unblocked = [d for d in details if d.startswith("binaries/")]
    problems: list[str] = []
    if len(unblocked) > 1:
        problems.append(
            f"{len(unblocked)} unblocked binaries cases, at most 1 permitted: "
            f"{unblocked}"
        )
    problems.extend(
        f"unpermitted binaries gap (only {BINARIES_GAP_CASE} is): {d}"
        for d in unblocked
        if not d.startswith(f"binaries/{BINARIES_GAP_CASE}:")
    )
    return problems


def _corpus(tmp_root: Path, vault: str, query_vault: str | None = None) -> list[dict]:
    """Build the attack corpus. Each entry is self-contained: the `code` is
    everything the sandboxed process runs. Namespace-sensitive filesystem
    entries also carry a parent-chosen `host_artifact`; those entries are
    scored from the host after the run, because a successful write can remain
    inside bubblewrap's namespace. All other entries are responsible for
    printing "UNSAFE ..." if (and only if) the attack succeeded, and
    "BLOCKED ..." when it was stopped. Every row carries an `applies_to` scope,
    and only applicable rows enter the roll-up. A row counts as blocked only
    on the evidence `_score_case` requires; anything else is NOT CONFIRMED and
    fails the roll-up (#109). The forked-descendant rows are `kill_expected`:
    the deadline kill itself is what blocks them. Rows marked
    `channel="query"` attack `conn` or a vault descriptor and run through the
    analyst query channel (`_run_query_channel`) against `query_vault`; the
    rest run on the plain path, which hands the child neither."""

    secret_path = tmp_root / "parent_secret.txt"
    secret_path.write_text("TOP-SECRET-VALUE")

    vault_real = os.path.realpath(vault)

    cases: list[dict] = []

    def add(
        cls: str,
        name: str,
        code: str,
        *,
        limits: dict | None = None,
        host_artifact: Path | None = None,
        parent_fd_path: Path | None = None,
        applies_to: str = "both",
        kill_expected: bool = False,
        channel: str = "plain",
        control_fd_path: Path | str | None = None,
    ) -> None:
        if applies_to not in EXECUTOR_SCOPES:
            raise ValueError(f"unknown executor scope: {applies_to!r}")
        if channel not in ("plain", "query"):
            raise ValueError(f"unknown channel: {channel!r}")
        cases.append({
            "class": cls,
            "name": name,
            "code": code,
            "limits": limits or {},
            "host_artifact": host_artifact,
            "parent_fd_path": parent_fd_path,
            "applies_to": applies_to,
            "kill_expected": kill_expected,
            "channel": channel,
            "control_fd_path": control_fd_path,
        })

    # (i) network -------------------------------------------------------- #
    add("network", "outbound_tcp_connect", """
import socket
try:
    socket.create_connection(("1.1.1.1", 443), timeout=3)
    print("UNSAFE connected")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("network", "dns_resolution", """
import socket
try:
    socket.gethostbyname("openrouter.ai")
    print("UNSAFE resolved")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("network", "curl_child_process", """
import subprocess
r = subprocess.run(["/usr/bin/curl", "-s", "-m", "3", "https://1.1.1.1"], capture_output=True)
if r.returncode == 0 and len(r.stdout) > 0:
    print("UNSAFE", r.returncode, len(r.stdout))
else:
    print("BLOCKED rc=", r.returncode, len(r.stdout))
""")
    add("network", "unix_domain_socket", """
import socket
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect("/var/run/mDNSResponder")
    print("UNSAFE connected")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("network", "udp_sendto", """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"x", ("1.1.1.1", 53))
    print("UNSAFE sent")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")

    # (ii) filesystem write ------------------------------------------------ #
    add("filesystem_write", "write_into_repo", f"""
try:
    with open({str(sb.REPO_ROOT / "PWNED_BY_TEST")!r}, "w") as f:
        f.write("x")
    print("UNSAFE wrote repo file")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    home_write_probe = Path(os.path.realpath(
        Path(HOME_WRITE_PROBE).with_name(
            f"pwned_by_sandbox_test_{tmp_root.name}.txt")))
    add("filesystem_write", "write_into_home", f"""
try:
    with open({str(home_write_probe)!r}, "w") as f:
        f.write("x")
    print("UNSAFE wrote home file")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""", host_artifact=home_write_probe)
    tmp_write_probe = tmp_root / "pwned_by_sandbox_test.txt"
    add("filesystem_write", "write_into_tmp_outside_work", f"""
try:
    with open({str(tmp_write_probe)!r}, "w") as f:
        f.write("x")
    print("UNSAFE wrote /tmp file")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""", host_artifact=tmp_write_probe)
    add("filesystem_write", "overwrite_parent_owned_code_py", """
try:
    with open("../code.py", "w") as f:
        f.write("HACKED")
    print("UNSAFE overwrote code.py")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("filesystem_write", "create_new_file_in_rundir_parent", """
try:
    with open("../evil_new_file.txt", "w") as f:
        f.write("x")
    print("UNSAFE created file in RUNDIR")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")

    # (iii) vault integrity ------------------------------------------------ #
    # The plain path hands the child no `conn`, so the three rows that attack
    # `conn` run through the analyst query channel (`channel="query"`), where
    # `conn` is the proxy to the parent's ledgered, read-only connection.
    add("vault_integrity", "writable_open_and_create_table", f"""
import sqlite3
try:
    c = sqlite3.connect({vault_real!r})
    c.execute("CREATE TABLE evil (a int)")
    c.commit()
    print("UNSAFE wrote vault")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("vault_integrity", "pragma_journal_mode_wal", """
try:
    conn.execute("PRAGMA journal_mode=WAL")
    print("UNSAFE set WAL")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""", channel="query")
    attach_probe = tmp_root / "probe_attach_evil.db"
    add("vault_integrity", "attach_and_write", f"""
try:
    conn.execute("ATTACH DATABASE {str(attach_probe)!r} AS evil")
    conn.execute("CREATE TABLE evil.x (a int)")
    print("UNSAFE attached and wrote")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""", host_artifact=attach_probe, channel="query")
    vacuum_probe = tmp_root / "probe_vacuum_exfil.db"
    add("vault_integrity", "vacuum_into_exfil", f"""
try:
    conn.execute("VACUUM INTO {str(vacuum_probe)!r}")
    print("UNSAFE vacuumed a copy out")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""", host_artifact=vacuum_probe, channel="query")

    # (iv) inherited descriptors ------------------------------------------- #
    add("inherited_descriptors", "enumerate_and_read_beyond_known", """
import os
unsafe = False
for name in sorted(os.listdir("/dev/fd"), key=lambda s: int(s) if s.isdigit() else -1):
    try:
        fd = int(name)
    except ValueError:
        continue
    if fd <= 4:  # 0-2 std, 3 output pipe, 4 our own vault conn: all expected
        continue
    try:
        data = os.read(fd, 4096)
        if data:
            unsafe = True
            print("UNSAFE readable fd", fd, data[:80])
    except Exception:
        pass
if not unsafe:
    print("BLOCKED no unexpected readable fd")
""")
    add("inherited_descriptors", "parent_only_secret_not_leaked", f"""
import os
found = False
for name in os.listdir("/dev/fd"):
    try:
        fd = int(name)
    except ValueError:
        continue
    if fd <= 3:
        continue
    try:
        data = os.read(fd, 4096)
        if b"TOP-SECRET-VALUE" in data:
            found = True
            print("UNSAFE leaked secret via fd", fd)
    except Exception:
        pass
if not found:
    print("BLOCKED no secret leak")
""", parent_fd_path=secret_path)
    # Bypassing `conn` needs a descriptor on the vault file itself. Any such
    # descriptor in the child is the escape, readable or writable, since reads
    # through it would skip the parent's ledger and authorizer; the probe
    # matches every open descriptor against the vault's device and inode.
    # Run on the query channel, the path that carries vault data to the child,
    # so the target is the vault that channel serves.
    query_vault_real = os.path.realpath(query_vault or vault)
    vault_stat = os.stat(query_vault_real)
    add("inherited_descriptors", "raw_vault_fd_write_bypassing_conn", f"""
import os
target = ({vault_stat.st_dev!r}, {vault_stat.st_ino!r})
hits = []
for name in os.listdir("/dev/fd"):
    try:
        fd = int(name)
        st = os.fstat(fd)
    except (ValueError, OSError):
        continue
    if (st.st_dev, st.st_ino) == target:
        hits.append(fd)
if hits:
    print("UNSAFE holds a vault descriptor", sorted(hits))
else:
    print("BLOCKED no descriptor refers to the vault")
""", channel="query", control_fd_path=query_vault_real)

    # (v) forked descendants ------------------------------------------------ #
    # These three share one property: the *direct* process spins (so the
    # timeout — not a natural exit — is what has to do the work) while a
    # forked descendant is the one actually holding something interesting.
    add("forked_descendants", "descendant_holds_fd3_open", """
import os, time
pid = os.fork()
if pid == 0:
    time.sleep(4)
    os.write(3, b"UNSAFE descendant survived deadline")
    os._exit(0)
while True:
    pass
""", limits={"wall_clock_s": 3.0}, kill_expected=True)
    vault_conn_marker = (tmp_root / "corpus" / "forked_descendants"
                         / "descendant_holds_vault_conn_open" / "work"
                         / "post_timeout_marker.txt")
    add("forked_descendants", "descendant_holds_vault_conn_open", f"""
import os, time
pid = os.fork()
if pid == 0:
    if conn is not None:
        cur = conn.execute("SELECT COUNT(*) FROM daily_metrics")
        cur.fetchone()
    time.sleep(4)
    with open({str(vault_conn_marker)!r}, "w", encoding="utf-8") as marker:
        marker.write("descendant survived deadline")
    os._exit(0)
while True:
    pass
""", limits={"wall_clock_s": 3.0}, host_artifact=vault_conn_marker,
        kill_expected=True)
    plain_descendant_marker = (tmp_root / "corpus" / "forked_descendants"
                               / "three_plain_descendants" / "work"
                               / "post_timeout_marker.txt")
    add("forked_descendants", "three_plain_descendants", f"""
import os, time
for _ in range(3):
    pid = os.fork()
    if pid == 0:
        time.sleep(4)
        with open({str(plain_descendant_marker)!r}, "a", encoding="utf-8") as marker:
            marker.write("descendant survived deadline\\n")
        os._exit(0)
while True:
    pass
""", limits={"wall_clock_s": 3.0}, host_artifact=plain_descendant_marker,
        kill_expected=True)

    # (vi) path / TOCTOU ---------------------------------------------------- #
    # These three are asserted structurally below (they need filesystem
    # setup/verification the generic runner can't express as a single
    # UNSAFE/BLOCKED line), so they are added as placeholders whose actual
    # verification lives in test_toctou_* and is folded into the same
    # counters by that test. See _record_toctou_result.
    add("toctou", "preplanted_code_py_is_overwritten", "print('marker: this file must not run')")
    add("toctou", "preplanted_profile_sb_is_overwritten", "print('ok')",
        applies_to=SEATBELT_ONLY)
    add("toctou", "vault_symlink_resolved_consistently", "r = conn.execute('SELECT 1').fetchone(); print('BLOCKED', r)")
    add("toctou", "home_tmpdir_env_remap_has_no_effect", f"""
import os
os.environ["HOME"] = "/"
os.environ["TMPDIR"] = "/"
try:
    with open({HOME_READ_PROBE!r}) as f:
        print("UNSAFE", f.read()[:20])
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")

    # (vii) binaries ---------------------------------------------------------- #
    add("binaries", "setuid_sudo", """
import subprocess
try:
    r = subprocess.run(["/usr/bin/sudo", "-n", "whoami"], capture_output=True, timeout=5)
    if r.returncode == 0 and b"root" in r.stdout:
        print("UNSAFE became root", r.stdout)
    else:
        print("BLOCKED rc=", r.returncode, r.stdout, r.stderr[:150])
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("binaries", "setuid_ping_raw_socket", """
import subprocess
try:
    r = subprocess.run(["/sbin/ping", "-c", "1", "-t", "2", "1.1.1.1"], capture_output=True, timeout=5)
    out = r.stdout + r.stderr
    if b"1 packets transmitted, 1 packets received" in out:
        print("UNSAFE", out[:200])
    else:
        print("BLOCKED rc=", r.returncode, out[:200])
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("binaries", "dyld_insert_libraries_injection", """
import subprocess, os
env = dict(os.environ)
env["DYLD_INSERT_LIBRARIES"] = "/usr/lib/libgmalloc.dylib"
try:
    r = subprocess.run(["/bin/echo", "hi"], capture_output=True, env=env, timeout=5)
    combined = r.stdout + r.stderr
    if b"GuardMalloc" in combined or b"libgmalloc" in combined:
        print("UNSAFE injected", combined[:200])
    else:
        print("BLOCKED no injection observed", r.returncode, combined[:80])
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("binaries", "system_binary_reads_forbidden_path", f"""
import subprocess
r = subprocess.run(["/bin/cat", {HOME_READ_PROBE!r}], capture_output=True, timeout=5)
if r.returncode == 0 and len(r.stdout) > 0:
    print("UNSAFE", r.stdout[:100])
else:
    print("BLOCKED rc=", r.returncode, r.stderr[:150])
""")

    # (viii) paths ------------------------------------------------------------ #
    add("paths", "users_shared_dir_readable", """
import os
try:
    d = os.listdir("/Users/Shared")
    print("UNSAFE listed", d[:5])
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("paths", "another_users_home_var_root", """
try:
    with open("/var/root/.bash_history") as f:
        print("UNSAFE", f.read()[:20])
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("paths", "device_file_read", """
try:
    with open("/dev/rdisk0", "rb") as f:
        data = f.read(512)
    print("UNSAFE read", len(data))
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("paths", "device_file_write", """
try:
    with open("/dev/rdisk0", "r+b") as f:
        f.write(b"x")
    print("UNSAFE wrote device")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")

    # (ix) fd-3 abuse ------------------------------------------------------- #
    # Per A1's own note: fd 3 is intentionally writable, so these are never
    # "blocked" by a write failure — they are scored on whether the PARENT
    # (this test, standing in for A2's future validator) correctly refuses
    # to treat the garbage as a valid result. See test_fd3_abuse below,
    # which is where these three are actually scored; they are listed here
    # for the corpus table's completeness.
    add("fd3_abuse", "oversized_10mib_write", """
import os
os.write(3, b"x" * (10 * 1024 * 1024))
print("wrote 10 MiB to fd 3")
""")
    add("fd3_abuse", "non_json_write", """
import os
os.write(3, b"not json at all {{{")
""")
    add("fd3_abuse", "write_after_close", """
import os
os.write(3, b'{"ok": true}')
os.close(3)
try:
    os.write(3, b"more")
    print("wrote after close (unexpected)")
except OSError:
    print("close then write raised, as expected")
""")

    return cases


def _executor_kind(executor) -> str:
    if isinstance(executor, sb.SeatbeltExecutor):
        return "seatbelt"
    if isinstance(executor, sb.BwrapExecutor):
        return "bwrap"
    raise TypeError(f"unsupported corpus executor: {type(executor).__name__}")


def _case_applies(case: dict, executor) -> bool:
    return _executor_kind(executor) in EXECUTOR_SCOPES[case["applies_to"]]


def _known_gap_applies(cls: str, executor) -> bool:
    return any(
        gap[0] == cls
        and _executor_kind(executor) in KNOWN_GAP_EXECUTORS.get(
            gap, EXECUTOR_SCOPES["both"])
        for gap in KNOWN_GAPS
    )


def _skip_if_not_applicable(executor, scope: str) -> None:
    if _executor_kind(executor) not in EXECUTOR_SCOPES[scope]:
        pytest.skip(f"corpus case is {scope}, not applicable to this executor")


def test_attack_corpus_class_coverage(executor, vault_path, tmp_path):
    # The query channel's parent opens its vault through `db.connect`, which
    # this suite refuses for the production vault (tests/conftest.py), so the
    # query-channel rows run against a synthetic read-only vault. What they
    # attack -- the parent's authorizer and the child's descriptors -- does
    # not depend on the vault's contents.
    from health_advisor.demo import build_demo_vault

    query_vault = tmp_path / "query_vault.db"
    build_demo_vault(query_vault, days=2)
    os.chmod(query_vault, 0o444)
    cases = _corpus(tmp_path, vault_path, str(query_vault))
    assert len(cases) == 35
    assert all(case["applies_to"] in EXECUTOR_SCOPES for case in cases)

    kind = _executor_kind(executor)
    targets = _host_targets(os.path.realpath(vault_path))
    readable_fd_path = next(c["parent_fd_path"] for c in cases if c["parent_fd_path"] is not None)
    in_scope: Counter[str] = Counter()
    applicable: Counter[str] = Counter()
    blocked: Counter[str] = Counter()
    details: list[str] = []
    not_applicable: list[str] = []
    rows: list[str] = []
    network_control_ok = _network_control()
    if not network_control_ok:
        details.append(
            "network/*: NOT CONFIRMED — the unsandboxed positive control "
            "(TCP connect to 1.1.1.1:443 and a DNS lookup) failed on this host")

    for case in cases:
        cls, name = case["class"], case["name"]
        if cls in ("toctou", "fd3_abuse"):
            # Scored by their own dedicated tests below (they need real
            # filesystem setup / different success criteria than a plain
            # UNSAFE/BLOCKED marker) — still counted here for the table.
            continue
        if not _case_applies(case, executor):
            continue
        run_dir = tmp_path / "corpus" / cls / name
        host_artifact = case["host_artifact"]
        parent_fd = None
        if host_artifact is not None:
            assert not os.path.exists(host_artifact), (
                "host probe artifact already exists before the sandbox run"
            )
        if case["parent_fd_path"] is not None:
            parent_fd = os.open(case["parent_fd_path"], os.O_RDONLY)
            while parent_fd <= 4:
                higher_fd = os.dup(parent_fd)
                os.close(parent_fd)
                parent_fd = higher_fd
            os.set_inheritable(parent_fd, True)
        evidence = _denial_evidence(cls, name, kind)
        try:
            if case["channel"] == "query":
                res, refused = _run_query_channel(
                    executor, case["code"], str(query_vault), run_dir, **case["limits"])
                expected_refusal = _QUERY_DENIED_ACTION.get((cls, name))
                if expected_refusal is not None and expected_refusal not in refused:
                    evidence = None     # the parent's authorizer did not refuse it
            else:
                res = _run(executor, case["code"], vault_path, run_dir, **case["limits"])
        finally:
            if parent_fd is not None:
                os.close(parent_fd)
        target = targets.get((cls, name))
        control_ok = None
        if (cls, name) in _CONTROLLED:
            control_ok = _unsandboxed_control(
                case["code"], tmp_path / "control" / cls / name,
                case["control_fd_path"] or readable_fd_path)
        verdict = _score_case(
            res, cls=cls, host_artifact=host_artifact,
            kill_expected=case["kill_expected"],
            network_control_ok=network_control_ok,
            evidence=evidence,
            target_present=target is None or os.path.lexists(target),
            control_ok=control_ok,
            control_failure=_CONTROLLED.get((cls, name), NOT_CONFIRMED))
        first_line = (res.stdout + res.fd3_bytes).strip().split(b"\n")[0][:160]
        rows.append(f"{cls}/{name} [{case['channel']}]: {verdict} — {first_line!r}")
        in_scope[cls] += 1
        if verdict == NOT_APPLICABLE:
            not_applicable.append(f"{cls}/{name}")
            continue
        applicable[cls] += 1
        if verdict == BLOCKED:
            blocked[cls] += 1
        elif verdict == NOT_CONFIRMED:
            details.append(
                f"{cls}/{name}: NOT CONFIRMED — returncode={res.returncode} "
                f"timed_out={res.timed_out} killed_group={res.killed_group} "
                f"stdout={res.stdout!r} stderr={res.stderr[:200]!r}")
        elif host_artifact is not None:
            details.append(f"{cls}/{name}: NOT BLOCKED — host artifact exists")
        else:
            details.append(f"{cls}/{name}: NOT BLOCKED — stdout={res.stdout!r}")

    # TOCTOU class, scored structurally (see test_toctou_* below for the
    # actual mechanics; this reproduces the same checks so the one summary
    # table is complete and self-contained).
    toctou_cases = [
        case for case in cases
        if case["class"] == "toctou" and _case_applies(case, executor)
    ]
    in_scope["toctou"] += len(toctou_cases)
    applicable["toctou"] += len(toctou_cases)
    blocked["toctou"] += _toctou_score(
        executor, vault_path, tmp_path, toctou_cases)

    # fd3_abuse class, scored on the *parent's* refusal per the note above.
    fd3_cases = [
        case for case in cases
        if case["class"] == "fd3_abuse" and _case_applies(case, executor)
    ]
    in_scope["fd3_abuse"] += len(fd3_cases)
    applicable["fd3_abuse"] += len(fd3_cases)
    if fd3_cases:
        blocked["fd3_abuse"] += _fd3_abuse_score(
            executor, vault_path, tmp_path)

    # binaries/paths/etc. already counted above via the loop.

    total_in_scope = sum(in_scope.values())
    total_applicable = sum(applicable.values())
    total_blocked = sum(blocked.values())

    lines = ["", f"executor: {kind}", "class | in scope | measured | blocked | rate"]
    for cls in sorted(in_scope):
        i, a, b = in_scope[cls], applicable[cls], blocked[cls]
        rate = f"{b/a:.0%}" if a else "-"
        lines.append(f"{cls:22s} | {i:8d} | {a:8d} | {b:7d} | {rate}")
    lines.append(f"{'TOTAL':22s} | {total_in_scope:8d} | {total_applicable:8d} | {total_blocked:7d} | "
                 f"{total_blocked/total_applicable:.0%}")
    print("\n".join(lines))
    print("ROWS (verdict — first output line):\n" + "\n".join(rows))
    if not_applicable:
        print("NOT APPLICABLE (measure nothing on this run):\n" + "\n".join(not_applicable))
    if details:
        print("NOT BLOCKED:\n" + "\n".join(details))

    assert len(cases) == 35, f"corpus has {len(cases)} cases, need 35"
    expected_in_scope = {"seatbelt": 35, "bwrap": 34}[kind]
    assert total_in_scope == expected_in_scope, (
        f"corpus has {total_in_scope} cases in scope, need {expected_in_scope}"
    )
    assert len(in_scope) == 9, f"corpus spans {len(in_scope)} classes, need 9"

    # Which rows may measure nothing is pinned by name, so the category cannot
    # absorb a row whose sandbox denial went missing: a row lands here only
    # through a declared host target that this user cannot see, or through a
    # control whose failure is a property of the host.
    host_dependent = ({f"{c}/{n}" for c, n in targets}
                      | {f"{c}/{n}" for (c, n), v in _CONTROLLED.items() if v == NOT_APPLICABLE})
    assert set(not_applicable) <= host_dependent, sorted(set(not_applicable) - host_dependent)

    for cls in applicable:
        expected_full = not _known_gap_applies(cls, executor)
        if expected_full:
            assert blocked[cls] == applicable[cls], (
                f"class {cls!r} is not 100% blocked ({blocked[cls]}/{applicable[cls]}) "
                "and is not in KNOWN_GAPS"
            )
    # The one documented Seatbelt exception, asserted explicitly rather than
    # silently excluded: users_shared_dir_readable is measured NOT blocked.
    if _known_gap_applies("paths", executor):
        assert blocked["paths"] == applicable["paths"] - 1

    # The Seatbelt `binaries` gap (#2), pinned the same way but by NAME rather
    # than by count, because the host disagrees with the runner: 4/4 blocked on
    # a current macOS/arm64 host, 3/4 on `macos-latest`. Both satisfy this.
    # Anything else does not.
    if _known_gap_applies("binaries", executor):
        problems = _binaries_gap_violations(details)
        assert not problems, "; ".join(problems)


# --------------------------------------------------------------------------- #
# (vi) TOCTOU / path — scoring, and the same checks as standalone tests
# --------------------------------------------------------------------------- #


def _toctou_preplanted_code_is_overwritten(executor, vault_path, run_dir: Path) -> bool:
    """§4.6: the profile is generated into parent-owned space and `run()`
    refuses a pre-existing parent-owned `code.py`. Pre-plant a malicious file
    at the exact path `run()` is about to write to, then confirm the write is
    refused before the pre-planted file can be executed or changed."""
    run_dir.mkdir(parents=True, exist_ok=True)
    code_path = run_dir / "code.py"
    code_path.write_text("print('UNSAFE ran the preplanted file')")
    try:
        _run(executor, "print('BLOCKED ran the real code')", vault_path, run_dir)
    except (FileExistsError, sb.TransientUnitError):
        return code_path.is_file() and "UNSAFE" in code_path.read_text()
    return False


def _toctou_preplanted_profile_is_overwritten(executor, vault_path, run_dir: Path) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_path = run_dir / "profile.sb"
    profile_path.write_text("(version 1)\n(allow default)\n")  # would-be wide-open profile
    try:
        _run(executor, "print('BLOCKED ran under the real profile')", vault_path, run_dir)
    except (FileExistsError, sb.TransientUnitError):
        profile_text = profile_path.read_text()
        return "allow default" in profile_text and "deny network*" not in profile_text
    return False


def _toctou_vault_symlink_consistency(executor, tmp_root: Path) -> bool:
    """The parent resolves the vault link when a run starts, never later.

    The plain Seatbelt path gives the child no vault at all (#24), so the
    probe runs through the query channel, where the parent opens the vault:
    a run started against link->A reads A, a run started after the swap
    reads B, and neither run's artifacts name either target's real path.
    """
    from health_advisor.demo import build_demo_vault

    vault_a = tmp_root / "toctou_vault_a.db"
    vault_b = tmp_root / "toctou_vault_b.db"
    build_demo_vault(vault_a, days=2)
    build_demo_vault(vault_b, days=5)
    link_path = tmp_root / "toctou_vault_link.db"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    os.symlink(vault_a, link_path)

    code = ("row = conn.execute('SELECT COUNT(*) FROM daily_metrics').fetchone(); "
            "emit('toctou', ['row_count'], ['count'], [[row[0]]])")

    def count_for(run_dir: Path) -> int | None:
        res = analyst_runner.run_analyst_code(
            code, str(link_path), str(run_dir), executor)
        if not isinstance(res, analyst_envelope.Envelope) or not res.tables:
            return None
        return res.tables[0]["rows"][0][0]

    run_dir1 = tmp_root / "toctou_symlink_1"
    count_a = count_for(run_dir1)
    link_path.unlink()
    os.symlink(vault_b, link_path)
    count_b = count_for(tmp_root / "toctou_symlink_2")

    # The FIRST run's own artifacts must not name either target: swapping
    # the link afterward must not retroactively alter a run that happened.
    real_a, real_b = os.path.realpath(vault_a), os.path.realpath(vault_b)
    unaltered = True
    for artifact in run_dir1.rglob("*"):
        if artifact.is_file():
            text = artifact.read_text(errors="replace")
            if real_a in text or real_b in text:
                unaltered = False

    return (count_a is not None and count_b is not None
            and count_a != count_b and unaltered)


def _toctou_home_tmpdir_remap_ignored(executor, vault_path, run_dir: Path) -> bool:
    code = f"""
import os
os.environ["HOME"] = "/"
os.environ["TMPDIR"] = "/"
try:
    with open({HOME_READ_PROBE!r}) as f:
        print("UNSAFE", f.read()[:20])
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
"""
    res = _run(executor, code, vault_path, run_dir)
    evidence = _denial_evidence("toctou", "home_tmpdir_env_remap_has_no_effect", _executor_kind(executor))
    return _score_case(res, cls="toctou", evidence=evidence) == BLOCKED


def _toctou_score(
    executor, vault_path, tmp_path: Path, cases: list[dict]
) -> int:
    checks = {
        "preplanted_code_py_is_overwritten":
            lambda: _toctou_preplanted_code_is_overwritten(
                executor, vault_path, tmp_path / "toctou_score_1"),
        "preplanted_profile_sb_is_overwritten":
            lambda: _toctou_preplanted_profile_is_overwritten(
                executor, vault_path, tmp_path / "toctou_score_2"),
        "vault_symlink_resolved_consistently":
            lambda: _toctou_vault_symlink_consistency(executor, tmp_path),
        "home_tmpdir_env_remap_has_no_effect":
            lambda: _toctou_home_tmpdir_remap_ignored(
                executor, vault_path, tmp_path / "toctou_score_4"),
    }
    return sum(1 for case in cases if checks[case["name"]]())


def test_toctou_preplanted_code_py_is_overwritten(executor, vault_path, tmp_path):
    assert _toctou_preplanted_code_is_overwritten(executor, vault_path, tmp_path / "run")


def test_toctou_preplanted_profile_sb_is_overwritten(executor, vault_path, tmp_path):
    _skip_if_not_applicable(executor, SEATBELT_ONLY)
    assert _toctou_preplanted_profile_is_overwritten(executor, vault_path, tmp_path / "run")


def test_toctou_vault_symlink_resolved_consistently(executor, tmp_path):
    assert _toctou_vault_symlink_consistency(executor, tmp_path)


def test_toctou_home_tmpdir_env_remap_has_no_effect(executor, vault_path, tmp_path):
    assert _toctou_home_tmpdir_remap_ignored(executor, vault_path, tmp_path / "run")


# --------------------------------------------------------------------------- #
# Parent-owned artefacts — exclusive creation for every executor path
# --------------------------------------------------------------------------- #


def _plant_parent_artifact_symlink(run_dir: Path, artifact: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir.parent / f"{artifact}.target"
    target.write_bytes(b"sentinel")
    (run_dir / artifact).symlink_to(target)
    return target


@pytest.mark.parametrize("artifact", ["code.py", "runner.py", "profile.sb"])
def test_seatbelt_run_rejects_preexisting_parent_artifact_symlinks(
        tmp_path, monkeypatch, artifact):
    executor = object.__new__(sb.SeatbeltExecutor)
    executor._python = sys.executable
    executor._real_home = str(tmp_path)
    executor._pyenv_prefix = str(tmp_path)
    executor._venv_dir = str(tmp_path)
    executor._pkg_dir = None
    run_dir = tmp_path / "seatbelt-run"
    target = _plant_parent_artifact_symlink(run_dir, artifact)

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(sb.subprocess, "Popen", unexpected_popen)
    with pytest.raises(FileExistsError):
        executor.run("pass", "unused-vault", str(run_dir))
    assert (run_dir / artifact).is_symlink()
    assert target.read_bytes() == b"sentinel"


@pytest.mark.parametrize("artifact", ["code.py", "runner.py"])
def test_bwrap_query_run_rejects_preexisting_parent_artifact_symlinks(
        tmp_path, monkeypatch, artifact):
    executor = object.__new__(sb.BwrapExecutor)
    run_dir = tmp_path / "bwrap-query-run"
    target = _plant_parent_artifact_symlink(run_dir, artifact)

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(sb.subprocess, "Popen", unexpected_popen)
    with pytest.raises(FileExistsError):
        executor.run_with_query_channel(
            "pass", str(run_dir), -1, runner_source="pass",
            limits=sb.RunLimits())
    assert (run_dir / artifact).is_symlink()
    assert target.read_bytes() == b"sentinel"


@pytest.mark.parametrize("artifact", ["code.py", "runner.py"])
def test_bwrap_plain_run_rejects_preexisting_parent_artifact_symlinks(
        tmp_path, monkeypatch, artifact):
    executor = object.__new__(sb.BwrapExecutor)
    run_dir = tmp_path / "bwrap-plain-run"
    target = _plant_parent_artifact_symlink(run_dir, artifact)

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(sb.subprocess, "Popen", unexpected_popen)
    with pytest.raises(FileExistsError):
        executor.run("pass", "unused-vault", str(run_dir))
    assert (run_dir / artifact).is_symlink()
    assert target.read_bytes() == b"sentinel"


@pytest.mark.parametrize("artifact", ["code.py", "runner.py"])
def test_transient_query_run_rejects_preexisting_parent_artifact_symlinks(
        tmp_path, monkeypatch, artifact):
    executor = object.__new__(sb.TransientUnitExecutor)
    run_dir = tmp_path / "transient-query-run"
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True)
    target = _plant_parent_artifact_symlink(run_dir, artifact)
    monkeypatch.setattr(
        sb.TransientUnitExecutor,
        "_prepare_transient_run_dir",
        lambda self, ignored: (run_dir, work_dir),
    )

    query_parent, query_child = socket.socketpair()
    real_socket = sb.socket.socket

    class FakeQueryServer:
        def bind(self, path):
            Path(path).touch()

        def listen(self, backlog):
            pass

        def settimeout(self, timeout):
            pass

        def close(self):
            pass

    def socket_factory(*args, **kwargs):
        if args[:2] == (socket.AF_UNIX, socket.SOCK_STREAM):
            return FakeQueryServer()
        return real_socket(*args, **kwargs)

    monkeypatch.setattr(sb.socket, "socket", socket_factory)
    monkeypatch.setattr(sb.shutil, "rmtree", lambda path, ignore_errors: None)

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(sb.subprocess, "Popen", unexpected_popen)
    try:
        with pytest.raises(sb.TransientUnitError, match="File exists"):
            executor.run_with_named_query_channel(
                "pass", "ignored-run-dir", query_child.detach(),
                runner_source="pass", limits=sb.RunLimits())
    finally:
        query_parent.close()
    assert (run_dir / artifact).is_symlink()
    assert target.read_bytes() == b"sentinel"


# The fd-3 clamp, as literals: 0 at the floor, 2 MiB at the ceiling
# (`_FD3_DRAIN_CEILING`, 32 x the 65,536-byte default). These were compared with
# `analyst_runner._fd3_limit` alone, which delegates to the same `RunLimits`
# clamp, so a comparison of the two pins no value at all.
FD3_CAP_CASES = [            # (configured max_fd3_bytes, effective cap)
    (-1, 0),
    (0, 0),
    (65_535, 65_535),
    (65_536, 65_536),
    (65_537, 65_537),
    (2_097_152, 2_097_152),
    (2_097_153, 2_097_152),
    (10**9, 2_097_152),
]


def test_run_limits_canonicalize_fd3_and_wall_clock_limits():
    assert sb.DEFAULT_MAX_FD3_BYTES == 65_536
    assert sb.RunLimits().max_fd3_bytes == 65_536
    for configured, cap in FD3_CAP_CASES:
        assert sb.RunLimits(max_fd3_bytes=configured).max_fd3_bytes == cap, configured
        assert analyst_runner._fd3_limit(configured) == cap, configured
    assert sb.RunLimits(wall_clock_s=-1).wall_clock_s == 0.0


def test_raw_executor_uses_canonical_fd3_cap(executor, vault_path, tmp_path):
    # (configured limit, bytes the child writes, bytes the parent keeps, oversized)
    cases = [
        (-1, 0, 0, False),
        (0, 0, 0, False),
        (65_535, 65_535, 65_535, False),
        (65_536, 65_536, 65_536, False),
        (65_537, 65_537, 65_537, False),
        (2_097_152, 2_097_152, 2_097_152, False),
        (2_097_153, 2_097_153, 2_097_152, True),       # the ceiling clamps the configured limit
    ]
    for index, (configured_limit, payload_size, kept, oversized) in enumerate(cases):
        result = executor.run(
            f"import os\nos.write(3, b'x' * {payload_size})\n",
            vault_path,
            str(tmp_path / f"fd3-limit-{index}"),
            limits=sb.RunLimits(max_fd3_bytes=configured_limit),
        )
        assert isinstance(result.fd3_bytes, bytes)
        assert result.fd3_oversized is oversized, configured_limit
        assert result.fd3_bytes == b"x" * kept, (configured_limit, len(result.fd3_bytes))


# --------------------------------------------------------------------------- #
# (ix) fd-3 abuse — scored on the parent's refusal, per A1's own note
# --------------------------------------------------------------------------- #


def _fd3_abuse_score(executor, vault_path, tmp_path: Path) -> int:
    n = 0

    res = _run(executor, "import os\nos.write(3, b'x' * (10 * 1024 * 1024))\n",
                vault_path, tmp_path / "fd3_oversized", max_fd3_bytes=65_536)
    if res.fd3_oversized and len(res.fd3_bytes) <= 65_536:
        n += 1

    res = _run(executor, "import os\nos.write(3, b'not json at all {{{')\n",
                vault_path, tmp_path / "fd3_nonjson")
    if res.fd3_as_json() is None and not res.fd3_oversized:
        n += 1

    res = _run(
        executor,
        "import os\nos.write(3, b'{\"ok\": true}')\nos.close(3)\n"
        "try:\n    os.write(3, b'more')\nexcept OSError:\n    pass\n",
        vault_path, tmp_path / "fd3_write_after_close",
    )
    if res.fd3_as_json() == {"ok": True} and not res.timed_out:
        n += 1

    return n


def test_fd3_abuse_oversized_write_is_capped_not_trusted(executor, vault_path, tmp_path):
    res = _run(executor, "import os\nos.write(3, b'x' * (10 * 1024 * 1024))\n",
               vault_path, tmp_path / "run", max_fd3_bytes=65_536)
    assert res.fd3_oversized is True
    assert len(res.fd3_bytes) <= 65_536


def test_fd3_abuse_non_json_is_not_silently_accepted(executor, vault_path, tmp_path):
    res = _run(executor, "import os\nos.write(3, b'not json at all {{{')\n",
               vault_path, tmp_path / "run")
    assert res.fd3_as_json() is None


def test_fd3_abuse_write_after_close_does_not_hang_or_corrupt(executor, vault_path, tmp_path):
    res = _run(
        executor,
        "import os\nos.write(3, b'{\"ok\": true}')\nos.close(3)\n"
        "try:\n    os.write(3, b'more')\nexcept OSError:\n    pass\n",
        vault_path, tmp_path / "run",
    )
    assert res.timed_out is False
    assert res.fd3_as_json() == {"ok": True}


# --------------------------------------------------------------------------- #
# Query-channel fd-3 validation — the parent must own the acceptance decision
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt query channel")
def test_query_channel_emitted_oversize_is_parent_refusal(
        executor, query_vault_path, tmp_path):
    code = """
columns = [f"value_{i}" for i in range(200)]
units = ["count"] * 200
rows = [[0] * 200 for _ in range(200)]
emit("oversized", columns, units, rows)
"""
    result = analyst_runner.run_analyst_code(
        code, query_vault_path, str(tmp_path / "query-oversized"), executor,
        limits=analyst_runner.AnalystLimits(
            max_fd3_bytes=sb.DEFAULT_MAX_FD3_BYTES),
    )
    assert isinstance(result, analyst_envelope.Refusal)
    assert result.reason == (
        f"analyst output exceeds {sb.DEFAULT_MAX_FD3_BYTES} bytes")


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt query channel")
def test_query_channel_runs_validate_for_emitted_invalid_unit(
        executor, query_vault_path, tmp_path, monkeypatch):
    calls = []
    original_validate = analyst_envelope.validate

    def spy_validate(raw, **kwargs):
        calls.append(raw)
        return original_validate(raw, **kwargs)

    monkeypatch.setattr(analyst_envelope, "validate", spy_validate)
    result = analyst_runner.run_analyst_code(
        "emit('bad_unit', ['value'], ['not_a_catalog_unit'], [[0]])",
        query_vault_path, str(tmp_path / "query-invalid-unit"), executor,
    )
    assert calls
    assert isinstance(result, analyst_envelope.Refusal)
    assert "unit 'not_a_catalog_unit' is not in" in result.reason


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt query channel")
def test_query_channel_does_not_accept_raw_fd3_payload_alongside_emit(
        executor, query_vault_path, tmp_path):
    code = """
import os
os.write(3, b'{\"tables\":[]}')
emit("legitimate", ["value"], ["count"], [[0]])
"""
    result = analyst_runner.run_analyst_code(
        code, query_vault_path, str(tmp_path / "query-raw-fd3"), executor,
    )
    # Measured on the host (2026-09-03): under the query channel the payload
    # descriptor is a parent-chosen high fd, not 3 -- fd 3 is closed in the
    # child, so the raw write raises EBADF before emit() runs and the parent
    # refuses with its own EXEC_FAILED reason. The raw bytes never reach the
    # parent as a payload, which is the property this test pins.
    assert isinstance(result, analyst_envelope.Refusal)
    assert result.reason.startswith("EXEC_FAILED")
    assert "Bad file descriptor" in result.reason


# =========================================================================== #
# Sanity: the lockdown does not also break legitimate vault reads
# =========================================================================== #


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt query channel")
def test_normal_vault_read_still_works(query_vault_path, tmp_path):
    res = analyst_runner.run_analyst_code(
        "row = conn.execute('SELECT COUNT(*) FROM daily_metrics').fetchone(); "
        "emit('daily_metrics', ['row_count'], ['count'], [[row[0]]])",
        query_vault_path, str(tmp_path / "run"), sb.SeatbeltExecutor(),
    )
    assert isinstance(res, analyst_envelope.Envelope)
    assert res.tables[0]["name"] == "daily_metrics"
    assert res.tables[0]["rows"][0][0] > 0


# =========================================================================== #
# KNOWN GAP, found during this work and NOT in the scored corpus above: on
# macOS, a forked descendant that calls os.setsid() before the timeout fires
# leaves the original process group entirely, and os.killpg on the group this
# executor tracks can no longer reach it. This is a stronger version of
# review finding 6 than §3.5 defends against: `setsid` is not filesystem or
# network activity, so nothing in the seatbelt profile constrains it, and
# `process-fork` is (deliberately) allowed. Linux bubblewrap runs the child in
# its own PID namespace, where killing the namespace init also kills this
# descendant; the inverse is tested in tests/test_bwrap_executor.py.
# =========================================================================== #


@pytest.mark.skipif(sys.platform != "darwin", reason="setsid process-group gap is macOS-only")
@pytest.mark.xfail(
    strict=True, raises=_EscapeStillOpen,
    reason="engine #18: a setsid'd descendant leaves the killed process group",
)
def test_KNOWN_GAP_forked_setsid_descendant_escapes_process_group_kill(executor, vault_path, tmp_path):
    """States the property we WANT: after the deadline kill, a forked
    descendant stops writing. Today it does not on macOS, so this is a strict
    xfail. When the gap closes the test XPASSes, strict mode turns that into a
    failure, and the failure is the prompt to delete this marker (and the
    KNOWN_GAPS entry). Only `_EscapeStillOpen` counts as the expected failure:
    a broken precondition is an ordinary failure, not a hidden xfail."""
    run_dir = tmp_path / "setsid_escape"
    marker = run_dir / "work" / "marker.txt"
    code = """
import os, time
pid = os.fork()
if pid == 0:
    os.setsid()
    for i in range(15):
        with open("marker.txt", "a") as f:
            f.write(str(i) + "\\n")
        time.sleep(1)
    os._exit(0)
"""
    try:
        res = _run(executor, code, vault_path, run_dir, wall_clock_s=2.0)
        assert res.timed_out is True
        # The descendant has to have really started, or "it stopped writing"
        # would be vacuously true.
        deadline = time.monotonic() + 2.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "the descendant never wrote its first marker line"

        # `run()` has already returned. Does the marker still advance?
        before = len(marker.read_text().splitlines())
        time.sleep(3.0)
        after = len(marker.read_text().splitlines())
        if after > before:
            raise _EscapeStillOpen(
                f"the descendant kept writing after the deadline kill "
                f"({before} -> {after} marker lines)")
    finally:
        # An escapee is outside the tracked group; find it by its run
        # directory and kill it by pid, or it outlives the test run.
        found = subprocess.run(
            ["pgrep", "-f", str(run_dir)], capture_output=True, text=True
        ).stdout.split()
        for pid_s in found:
            try:
                os.kill(int(pid_s), 9)
            except (ProcessLookupError, ValueError):
                pass


# --------------------------------------------------------------------------- #
# #2 — the Seatbelt `binaries` gap is a NAMED exception, not a class exemption
#
# These run on every host, including one where the injection is blocked and the
# corpus assertion above is therefore vacuous. That is the point: #2's own
# `Done when` asks for an entry "naming the exact configurations affected — not
# a blanket exemption", and listing a class in KNOWN_GAPS deletes its assertion
# entirely. Without a floor, all four binaries cases could regress green.
# --------------------------------------------------------------------------- #

_DYLD = "binaries/dyld_insert_libraries_injection: NOT BLOCKED — stdout=''"
_OTHER = "binaries/execve_system_binary: NOT BLOCKED — stdout=''"


def test_binaries_gap_allows_a_host_that_blocks_everything():
    """macOS 26.6.2 arm64, measured 4/4 — nothing unblocked."""
    assert _binaries_gap_violations([]) == []
    assert _binaries_gap_violations(["paths/users_shared_dir_readable: NOT BLOCKED"]) == []


def test_binaries_gap_allows_exactly_the_named_case():
    """`macos-latest`, measured 3/4 — the one case #2 characterises."""
    assert _binaries_gap_violations([_DYLD]) == []


def test_binaries_gap_rejects_a_second_unblocked_case():
    """The regression the KNOWN_GAPS entry must NOT absorb."""
    problems = _binaries_gap_violations([_DYLD, _OTHER])
    assert problems, "a second unblocked binaries case must be reported"
    assert any("at most 1 permitted" in p for p in problems)


def test_binaries_gap_rejects_a_different_single_case():
    """One unblocked case, but not the named one — still a regression."""
    problems = _binaries_gap_violations([_OTHER])
    assert problems, "an unnamed binaries case must be reported"
    assert any("unpermitted binaries gap" in p for p in problems)
