"""A1's escape corpus (#115 / M7, docs/product/reviews/analyst-mode-proposal.md
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

This suite is intentionally slow in one place:
`test_wall_clock_default_timeout_kills_within_tolerance` runs the *default*
60s wall clock for real, because A1's `Done when` 4 asks for the default to
be measured, not a scaled-down stand-in.
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from health_advisor import analyst_sandbox as sb

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


def _run(
    executor: sb.SeatbeltExecutor,
    code: str,
    vault: str,
    run_dir: Path,
    **limit_kwargs,
) -> sb.RawResult:
    limits = sb.RunLimits(**limit_kwargs) if limit_kwargs else None
    return executor.run(code, vault, str(run_dir), limits=limits)


def _is_unsafe(res: sb.RawResult) -> bool:
    """Corpus convention: an attack that *succeeded* prints a line starting
    with "UNSAFE" — to stdout or, if it made it that far, into the fd-3
    payload. Anything else (a caught exception printing "BLOCKED...", or an
    uncaught exception that killed the child with a nonzero exit code) is
    blocked. This treats "the process died before it could misbehave" as a
    pass, which matches how every probe in §2.3 itself was read."""
    haystack = res.stdout + res.fd3_bytes
    return b"UNSAFE" in haystack


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


def test_wall_clock_default_timeout_kills_within_tolerance(executor, vault_path, tmp_path):
    """Done when 4: the *default* (60s) wall clock, measured for real, on
    `while True: pass`. This is deliberately not scaled down."""
    run_dir = tmp_path / "default_timeout"
    code = "while True:\n    pass\n"
    t0 = time.monotonic()
    res = _run(executor, code, vault_path, run_dir)  # default RunLimits(): 60.0s
    elapsed = time.monotonic() - t0

    print(f"\n[Done-when 4] default-timeout kill_elapsed={elapsed:.2f}s")
    assert res.timed_out is True
    assert res.killed_group is True
    assert 55.0 <= elapsed <= 65.0, f"expected 60s +/- 5s, measured {elapsed:.2f}s"
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


def test_generated_code_receives_conn_not_a_path(executor, vault_path, tmp_path):
    run_dir = tmp_path / "no_path_leak"
    code = """
import json, os
names = sorted(k for k in dict(globals()) if not k.startswith("_"))
leaked_path = False
for k, v in list(globals().items()):
    if isinstance(v, str) and ("health.db" in v or v == %r):
        leaked_path = True
row = conn.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()
os.write(3, json.dumps({
    "globals": names, "leaked_path": leaked_path, "count": row[0],
}).encode())
""" % (os.path.realpath(vault_path),)
    res = _run(executor, code, vault_path, run_dir)
    payload = res.fd3_as_json()
    assert payload is not None, res.stderr
    assert payload["leaked_path"] is False
    assert "conn" in payload["globals"]
    assert payload["count"] > 0  # proves conn is a live, working connection


# =========================================================================== #
# Done when 1 — the attack corpus: 9 classes, each attempted -> blocked
# =========================================================================== #

# The one class, and the one sub-case, this profile — implemented exactly as
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
KNOWN_GAPS = {
    ("paths", "users_shared_dir_readable"),
}


def _corpus(tmp_root: Path, vault: str) -> list[dict]:
    """Build the attack corpus. Each entry is self-contained: the `code` is
    everything the sandboxed process runs, and it is responsible for
    printing "UNSAFE ..." if (and only if) the attack succeeded. Everything
    else is scored as blocked (`_is_unsafe` above) — including an uncaught
    exception that simply kills the child, which is exactly how most of
    these actually resolve in practice (see the module docstring)."""

    secret_path = tmp_root / "parent_secret.txt"
    secret_path.write_text("TOP-SECRET-VALUE")

    vault_real = os.path.realpath(vault)

    cases: list[dict] = []

    def add(cls: str, name: str, code: str, *, limits: dict | None = None) -> None:
        cases.append({"class": cls, "name": name, "code": code, "limits": limits or {}})

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
    add("filesystem_write", "write_into_home", f"""
try:
    with open({HOME_WRITE_PROBE!r}, "w") as f:
        f.write("x")
    print("UNSAFE wrote home file")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("filesystem_write", "write_into_tmp_outside_work", """
try:
    with open("/tmp/pwned_by_sandbox_test.txt", "w") as f:
        f.write("x")
    print("UNSAFE wrote /tmp file")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
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
""")
    add("vault_integrity", "attach_and_write", """
try:
    conn.execute("ATTACH DATABASE '/tmp/probe_attach_evil.db' AS evil")
    conn.execute("CREATE TABLE evil.x (a int)")
    print("UNSAFE attached and wrote")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")
    add("vault_integrity", "vacuum_into_exfil", """
try:
    conn.execute("VACUUM INTO '/tmp/probe_vacuum_exfil.db'")
    print("UNSAFE vacuumed a copy out")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")

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
""")
    add("inherited_descriptors", "raw_vault_fd_write_bypassing_conn", """
import os
try:
    os.write(4, b"corrupt")
    print("UNSAFE wrote raw vault fd")
except Exception as e:
    print("BLOCKED", type(e).__name__, e)
""")

    # (v) forked descendants ------------------------------------------------ #
    # These three share one property: the *direct* process spins (so the
    # timeout — not a natural exit — is what has to do the work) while a
    # forked descendant is the one actually holding something interesting.
    add("forked_descendants", "descendant_holds_fd3_open", """
import os, time
pid = os.fork()
if pid == 0:
    time.sleep(60)
    os._exit(0)
while True:
    pass
""", limits={"wall_clock_s": 3.0})
    add("forked_descendants", "descendant_holds_vault_conn_open", """
import os, time
pid = os.fork()
if pid == 0:
    cur = conn.execute("SELECT COUNT(*) FROM daily_metrics")
    cur.fetchone()
    time.sleep(60)
    os._exit(0)
while True:
    pass
""", limits={"wall_clock_s": 3.0})
    add("forked_descendants", "three_plain_descendants", """
import os, time
for _ in range(3):
    pid = os.fork()
    if pid == 0:
        time.sleep(60)
        os._exit(0)
while True:
    pass
""", limits={"wall_clock_s": 3.0})

    # (vi) path / TOCTOU ---------------------------------------------------- #
    # These three are asserted structurally below (they need filesystem
    # setup/verification the generic runner can't express as a single
    # UNSAFE/BLOCKED line), so they are added as placeholders whose actual
    # verification lives in test_toctou_* and is folded into the same
    # counters by that test. See _record_toctou_result.
    add("toctou", "preplanted_code_py_is_overwritten", "print('marker: this file must not run')")
    add("toctou", "preplanted_profile_sb_is_overwritten", "print('ok')")
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


def test_attack_corpus_class_coverage(executor, vault_path, tmp_path):
    cases = _corpus(tmp_path, vault_path)

    attempted: Counter[str] = Counter()
    blocked: Counter[str] = Counter()
    details: list[str] = []

    for case in cases:
        cls, name = case["class"], case["name"]
        if cls in ("toctou", "fd3_abuse"):
            # Scored by their own dedicated tests below (they need real
            # filesystem setup / different success criteria than a plain
            # UNSAFE/BLOCKED marker) — still counted here for the table.
            continue
        run_dir = tmp_path / "corpus" / cls / name
        res = _run(executor, case["code"], vault_path, run_dir, **case["limits"])
        unsafe = _is_unsafe(res)
        attempted[cls] += 1
        if not unsafe:
            blocked[cls] += 1
        else:
            details.append(f"{cls}/{name}: NOT BLOCKED — stdout={res.stdout!r}")

    # TOCTOU class, scored structurally (see test_toctou_* below for the
    # actual mechanics; this reproduces the same checks so the one summary
    # table is complete and self-contained).
    attempted["toctou"] += 4
    blocked["toctou"] += _toctou_score(executor, vault_path, tmp_path)

    # fd3_abuse class, scored on the *parent's* refusal per the note above.
    attempted["fd3_abuse"] += 3
    blocked["fd3_abuse"] += _fd3_abuse_score(executor, vault_path, tmp_path)

    # binaries/paths/etc. already counted above via the loop.

    total_attempted = sum(attempted.values())
    total_blocked = sum(blocked.values())

    lines = ["", "class | attempted | blocked | rate"]
    for cls in sorted(attempted):
        a, b = attempted[cls], blocked[cls]
        lines.append(f"{cls:22s} | {a:9d} | {b:7d} | {b/a:.0%}")
    lines.append(f"{'TOTAL':22s} | {total_attempted:9d} | {total_blocked:7d} | {total_blocked/total_attempted:.0%}")
    print("\n".join(lines))
    if details:
        print("NOT BLOCKED:\n" + "\n".join(details))

    assert total_attempted >= 34, f"corpus has {total_attempted} attempts, need >= 34"
    assert len(attempted) == 9, f"corpus spans {len(attempted)} classes, need 9"

    for cls in attempted:
        expected_full = not any(g[0] == cls for g in KNOWN_GAPS)
        if expected_full:
            assert blocked[cls] == attempted[cls], (
                f"class {cls!r} is not 100% blocked ({blocked[cls]}/{attempted[cls]}) "
                "and is not in KNOWN_GAPS"
            )
    # The one documented exception, asserted explicitly rather than silently
    # excluded: users_shared_dir_readable is measured NOT blocked.
    assert blocked["paths"] == attempted["paths"] - 1


# --------------------------------------------------------------------------- #
# (vi) TOCTOU / path — scoring, and the same checks as standalone tests
# --------------------------------------------------------------------------- #


def _toctou_preplanted_code_is_overwritten(executor, vault_path, run_dir: Path) -> bool:
    """§4.6: the profile is generated into parent-owned space and `run()`
    always regenerates `code.py` fresh. Pre-plant a malicious file at the
    exact path `run()` is about to write to, then confirm the ACTUALLY
    EXECUTED code is the one just passed in, never the pre-planted one."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "code.py").write_text("print('UNSAFE ran the preplanted file')")
    res = _run(executor, "print('BLOCKED ran the real code')", vault_path, run_dir)
    return b"BLOCKED" in res.stdout and b"UNSAFE" not in res.stdout


def _toctou_preplanted_profile_is_overwritten(executor, vault_path, run_dir: Path) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "profile.sb").write_text("(version 1)\n(allow default)\n")  # would-be wide-open profile
    res = _run(executor, "print('BLOCKED ran under the real profile')", vault_path, run_dir)
    # If the preplanted wide-open profile had survived, network would work.
    net_code_run_dir = run_dir  # same dir, second call would reuse a regenerated profile
    ok_ran = b"BLOCKED" in res.stdout
    # Confirm the actual profile.sb on disk is OUR generated one, not the stub.
    profile_text = (run_dir / "profile.sb").read_text()
    ok_regenerated = "deny network*" in profile_text and "allow default" not in profile_text
    return ok_ran and ok_regenerated


def _toctou_vault_symlink_consistency(executor, tmp_root: Path) -> bool:
    vault_a = tmp_root / "toctou_vault_a.db"
    vault_b = tmp_root / "toctou_vault_b.db"
    for p, marker in ((vault_a, "AAAA"), (vault_b, "BBBB")):
        c = sqlite3.connect(str(p))
        c.execute("CREATE TABLE marker (v TEXT)")
        c.execute("INSERT INTO marker VALUES (?)", (marker,))
        c.commit()
        c.close()
        os.chmod(p, 0o444)
    link_path = tmp_root / "toctou_vault_link.db"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    os.symlink(vault_a, link_path)

    code = "r = conn.execute('SELECT v FROM marker').fetchone(); print('BLOCKED', r[0])"
    run_dir1 = tmp_root / "toctou_symlink_1"
    res1 = _run(executor, code, str(link_path), run_dir1)
    read_a = b"AAAA" in res1.stdout

    link_path.unlink()
    os.symlink(vault_b, link_path)
    run_dir2 = tmp_root / "toctou_symlink_2"
    res2 = _run(executor, code, str(link_path), run_dir2)
    read_b = b"BBBB" in res2.stdout

    # The FIRST run's own generated artifacts must still name vault_a's
    # realpath — swapping the link afterward must not retroactively alter
    # a run that already happened.
    runner1 = (run_dir1 / "runner.py").read_text()
    unaltered = os.path.realpath(vault_a) in runner1 and os.path.realpath(vault_b) not in runner1

    return read_a and read_b and unaltered


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
    return not _is_unsafe(res)


def _toctou_score(executor, vault_path, tmp_path: Path) -> int:
    checks = [
        _toctou_preplanted_code_is_overwritten(executor, vault_path, tmp_path / "toctou_score_1"),
        _toctou_preplanted_profile_is_overwritten(executor, vault_path, tmp_path / "toctou_score_2"),
        _toctou_vault_symlink_consistency(executor, tmp_path),
        _toctou_home_tmpdir_remap_ignored(executor, vault_path, tmp_path / "toctou_score_4"),
    ]
    return sum(1 for c in checks if c)


def test_toctou_preplanted_code_py_is_overwritten(executor, vault_path, tmp_path):
    assert _toctou_preplanted_code_is_overwritten(executor, vault_path, tmp_path / "run")


def test_toctou_preplanted_profile_sb_is_overwritten(executor, vault_path, tmp_path):
    assert _toctou_preplanted_profile_is_overwritten(executor, vault_path, tmp_path / "run")


def test_toctou_vault_symlink_resolved_consistently(executor, tmp_path):
    assert _toctou_vault_symlink_consistency(executor, tmp_path)


def test_toctou_home_tmpdir_env_remap_has_no_effect(executor, vault_path, tmp_path):
    assert _toctou_home_tmpdir_remap_ignored(executor, vault_path, tmp_path / "run")


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


# =========================================================================== #
# Sanity: the lockdown does not also break legitimate vault reads
# =========================================================================== #


def test_normal_vault_read_still_works(executor, vault_path, tmp_path):
    res = _run(
        executor,
        "r = conn.execute('SELECT COUNT(*) FROM daily_metrics').fetchone(); print('rows', r[0])",
        vault_path, tmp_path / "run",
    )
    assert res.returncode == 0, res.stderr
    assert b"rows " in res.stdout
    count = int(res.stdout.split(b"rows ")[1].split(b"\n")[0])
    assert count > 0


# =========================================================================== #
# KNOWN GAP, found during this work and NOT in the scored corpus above: a
# forked descendant that calls os.setsid() before the timeout fires leaves
# the original process group entirely, and os.killpg on the group this
# executor tracks can no longer reach it. This is a stronger version of
# review finding 6 than §3.5 defends against: `setsid` is not filesystem or
# network activity, so nothing in the seatbelt profile constrains it, and
# `process-fork` is (deliberately) allowed. Reported here as a real,
# unblocked escape rather than narrowed out of the suite — see this
# session's report for the full write-up.
# =========================================================================== #


def test_KNOWN_GAP_forked_setsid_descendant_escapes_process_group_kill(executor, vault_path, tmp_path):
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
    res = _run(executor, code, vault_path, run_dir, wall_clock_s=2.0)
    assert res.timed_out is True

    # Give the escaped, setsid'd grandchild time to keep writing well past
    # the point `run()` already returned.
    time.sleep(4.0)
    assert marker.exists(), "expected the escaped descendant to still be alive and writing"
    lines_written = len(marker.read_text().splitlines())
    assert lines_written >= 2, (
        f"expected the setsid'd descendant to keep running after the parent's "
        f"kill attempt; observed {lines_written} marker line(s) — if this now "
        f"fails, the escape may have been closed and this test (and the "
        f"module docstring / report) should be updated, not silently deleted"
    )

    survivors = subprocess.run(
        ["pgrep", "-f", str(run_dir)], capture_output=True, text=True
    ).stdout.strip()
    assert survivors, "expected a still-alive escaped descendant found by pgrep -f"

    # Clean up: the escapee is no longer in our tracked group, so we have to
    # find and kill it by pid directly, or it outlives the whole test run.
    for pid_s in survivors.splitlines():
        try:
            os.kill(int(pid_s), 9)
        except ProcessLookupError:
            pass
