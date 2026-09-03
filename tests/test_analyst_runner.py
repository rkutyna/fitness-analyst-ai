"""End-to-end checks for the parent-owned analyst query boundary."""
from __future__ import annotations

import os
import fcntl
import json
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from health_advisor import analyst_envelope as env
from health_advisor import analyst_runner as runner
from health_advisor import analyst_sandbox as sandbox


def _build_vault(path: Path, rows: int = 4) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(rows)])
    conn.commit()
    conn.close()


def _build_shape_vault(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE daily_metrics ("
        "metric TEXT, date TEXT, count INTEGER, sum REAL, avg REAL, "
        "min REAL, max REAL, last REAL, unit TEXT)"
    )
    conn.executemany(
        "INSERT INTO daily_metrics "
        "(metric, date, count, sum, avg, min, max, last, unit) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("step_count", "2026-01-01", 2, 30.0, 15.0, 10.0, 20.0, 20.0, "count"),
            ("step_count", "2026-01-02", 2, 50.0, 25.0, 20.0, 30.0, 30.0, "count"),
            ("jog_minutes", "2026-01-01", 1, 5.0, 5.0, 5.0, 5.0, 5.0, "min"),
        ],
    )
    conn.commit()
    conn.close()


class LocalChannelExecutor:
    """Portable child-process executor with the production channel shape."""

    def __init__(self):
        self.last = None

    @staticmethod
    def _high(fd: int) -> int:
        duplicate = fcntl.fcntl(fd, fcntl.F_DUPFD, 10)
        os.close(fd)
        return duplicate

    def run_with_query_channel(self, code, run_dir, query_fd, *,
                               runner_source, limits):
        import selectors

        run = Path(run_dir)
        run.mkdir(parents=True, exist_ok=True)
        (run / "code.py").write_text(code, encoding="utf-8")
        (run / "runner.py").write_text(runner_source, encoding="utf-8")
        out_r, out_w = os.pipe()
        out_r = self._high(out_r)
        out_w = self._high(out_w)
        query_fd = self._high(query_fd)

        def preexec():
            os.close(out_r)
            os.dup2(out_w, 3)
            os.close(out_w)
            os.dup2(query_fd, 4)
            os.close(query_fd)

        proc = subprocess.Popen(
            [sys.executable, "-I", str(run / "runner.py")], cwd=str(run),
            env={"PATH": "/usr/bin:/bin", "HOME": str(run),
                 "TMPDIR": str(run), "ANALYST_QUERY_FD": "4"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=(3, 4), preexec_fn=preexec,
            start_new_session=True,
        )
        os.close(out_w)
        os.close(query_fd)
        buffers = {"fd3": bytearray(), "stdout": bytearray(),
                   "stderr": bytearray()}
        sel = selectors.DefaultSelector()
        sel.register(out_r, selectors.EVENT_READ, "fd3")
        sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
        sel.register(proc.stderr, selectors.EVENT_READ, "stderr")
        start = time.monotonic()
        timeout = getattr(limits, "wall_clock_s", 60.0)
        while sel.get_map():
            if time.monotonic() - start > timeout:
                os.killpg(proc.pid, 9)
                break
            for key, _ in sel.select(0.05):
                fileobj = key.fileobj
                fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
                chunk = os.read(fd, 65536)
                if not chunk:
                    sel.unregister(fileobj)
                    continue
                buffers[key.data].extend(chunk)
        proc.wait(timeout=5)
        os.close(out_r)
        proc.stdout.close()
        proc.stderr.close()
        self.last = sandbox.RawResult(
            fd3_bytes=bytes(buffers["fd3"]), fd3_oversized=False,
            stdout=bytes(buffers["stdout"]), stderr=bytes(buffers["stderr"]),
            returncode=proc.returncode, timed_out=False, killed_group=False,
            duration_s=time.monotonic() - start, pgid=proc.pid, run_dir=run,
        )
        return self.last


@pytest.fixture
def vault_path(tmp_path):
    path = tmp_path / "vault.db"
    _build_vault(path)
    return path


@pytest.fixture
def executor():
    return LocalChannelExecutor()


@pytest.fixture
def shape_vault(tmp_path):
    path = tmp_path / "shape-vault.db"
    _build_shape_vault(path)
    return path


def _run(code, vault_path, tmp_path, executor, **kwargs):
    return runner.run_analyst_code(
        code, str(vault_path), str(tmp_path / "run"), executor,
        limits=runner.AnalystLimits(**kwargs) if kwargs else None,
    )


def test_hostile_fabricated_numbers_use_parent_ledger(vault_path, tmp_path,
                                                       executor, monkeypatch):
    opened = []
    original = runner.ledger.open_ledgered

    def capture(path):
        connection = original(path)
        opened.append(connection)
        return connection

    monkeypatch.setattr(runner.ledger, "open_ledgered", capture)
    result = _run(
        """
fake_ledger = {'query_count': 99, 'rows_read': 99, 'tables_read': ['t']}
print('CHILD_FAKE_LEDGER=' + repr(fake_ledger), file=__import__('sys').stderr)
emit('fabricated', ['value'], ['count'], [[999]])
""",
        vault_path, tmp_path, executor,
    )
    assert isinstance(result, env.Refusal)
    assert result.reason == "emitted 1 numeric tables from 0 vault tables and 0 reads"
    assert opened[0].ledger.as_dict() == {
        "query_count": 0, "rows_read": 0, "tables_read": [], "columns_read": []
    }
    assert "CHILD_FAKE_LEDGER" in executor.last.stderr.decode()
    print(f"\n[runner] hostile_refusal={result.reason} "
          f"child_fake_ledger=query_count=99,rows_read=99,tables_read=['t'] "
          f"parent_ledger={opened[0].ledger.as_dict()}")


def test_vault_path_is_absent_from_child_and_proxy(vault_path, tmp_path, executor):
    result = _run(
        """
import os, sqlite3, sys
print('ENV=' + repr(dict(os.environ)), file=sys.stderr)
print('PROXY=' + repr(conn.__dict__) + ' ' + repr(conn), file=sys.stderr)
try:
    sqlite3.connect('/definitely/not/the/vault.db')
except Exception as exc:
    print('GUESSED_OPEN=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
rows = conn.execute('SELECT v FROM t LIMIT 1').fetchall()
emit('observed', ['value'], ['count'], [[rows[0][0]]])
""",
        vault_path, tmp_path, executor,
    )
    assert isinstance(result, env.Envelope)
    diagnostics = executor.last.stderr.decode()
    assert str(vault_path) not in runner.RUNNER_TEMPLATE
    assert str(vault_path) not in diagnostics
    assert "_conn" not in diagnostics
    assert "GUESSED_OPEN=OperationalError:" in diagnostics
    assert "AnalystQueryProxy fd=4" in diagnostics
    print("\n[runner] path_template=False path_env=False path_proxy=False "
          "guessed_open=OperationalError: unable to open database file")


def test_sql_error_is_catchable_and_50_queries_do_not_deadlock(
    vault_path, tmp_path, executor,
):
    code = """
import sys
try:
    conn.execute('SELECT missing FROM t')
except Exception as exc:
    print('CAUGHT=' + type(exc).__name__, file=sys.stderr)
for _ in range(50):
    print('diagnostic ' + ('x' * 1000), file=sys.stderr)
    conn.execute('SELECT v FROM t LIMIT 1').fetchall()
emit('queries', ['value'], ['count'], [[50]])
"""
    started = time.monotonic()
    result = _run(code, vault_path, tmp_path, executor)
    elapsed = time.monotonic() - started
    assert isinstance(result, env.Envelope)
    assert result.ledger["query_count"] == 50
    assert "CAUGHT=AnalystQueryError" in executor.last.stderr.decode()
    print(f"\n[runner] sequential_queries=50 elapsed_s={elapsed:.4f}")


def test_median_round_trip_latency_of_20_queries(vault_path, tmp_path, executor,
                                                  monkeypatch):
    timings = []
    original = runner._service_query

    def timed(sock, conn, max_rows, pending):
        started = time.perf_counter()
        result = original(sock, conn, max_rows, pending)
        if result[0]:
            timings.append((time.perf_counter() - started) * 1000)
        return result

    monkeypatch.setattr(runner, "_service_query", timed)
    result = _run(
        """
for _ in range(20):
    conn.execute('SELECT v FROM t LIMIT 1').fetchall()
emit('latency', ['value'], ['count'], [[20]])
""",
        vault_path, tmp_path, executor,
    )
    assert isinstance(result, env.Envelope)
    assert len(timings) == 20
    median_ms = statistics.median(timings)
    print(f"\n[runner] proxy_queries=20 median_service_ms={median_ms:.3f}")


def test_parent_row_cap_is_typed_refusal(vault_path, tmp_path, executor):
    result = _run(
        """
try:
    conn.execute('SELECT v FROM t').fetchall()
except Exception:
    pass
""",
        vault_path, tmp_path, executor, max_query_rows=3,
    )
    assert isinstance(result, env.Refusal)
    assert result.reason == "query result exceeds row cap: 4 > 3"
    print(f"\n[runner] configured_row_cap=3 reason={result.reason}")


def test_last_for_sum_metric_is_a_typed_parent_refusal(
        shape_vault, tmp_path, executor):
    result = _run(
        """
rows = conn.execute(
    "SELECT date, last FROM daily_metrics "
    "WHERE metric = 'step_count' ORDER BY date"
).fetchall()
total = sum(row[1] for row in rows if row[1] is not None)
emit('total_steps_last_shaped', ['total_steps'], ['count'], [[total]])
""",
        shape_vault, tmp_path, executor,
    )
    assert isinstance(result, env.Refusal)
    assert result.reason == (
        "analyst query refused: daily_metrics.last is invalid for "
        "sum-shaped metric(s) 'step_count'; select daily_metrics.sum"
    )


def test_last_for_last_metric_still_executes(shape_vault, tmp_path, executor):
    result = _run(
        """
rows = conn.execute(
    "SELECT date, last FROM daily_metrics "
    "WHERE metric = 'jog_minutes' ORDER BY date"
).fetchall()
emit('jog_minutes', ['minutes'], ['min'], [[row[1]] for row in rows])
""",
        shape_vault, tmp_path, executor,
    )
    assert isinstance(result, env.Envelope)
    assert result.ledger["query_count"] == 1
    assert result.ledger["columns_read"] == [
        "daily_metrics.date", "daily_metrics.last", "daily_metrics.metric",
    ]


def test_nonzero_child_stderr_is_carried_as_quoted_diagnostic(
        vault_path, tmp_path, executor, capsys):
    """The real bwrap namespace failure must not disappear with the rc."""
    result = _run(
        '''import os
os.write(2, b"bwrap: No permissions to create a new namespace, "
             b"likely because the kernel does not allow non-privileged "
             b"user namespaces.\\n")
os._exit(1)
''',
        vault_path, tmp_path, executor,
    )

    assert isinstance(result, env.Refusal)
    assert result.reason == "analyst process exited with status 1"
    assert result.diagnostic == (
        'quoted child stderr (untrusted tail): "bwrap: No permissions to '
        'create a new namespace, likely because the kernel does not allow '
        'non-privileged user namespaces."'
    )
    refusal_json = json.dumps(result.to_dict(), sort_keys=True)
    assert json.loads(refusal_json)["diagnostic"] == result.diagnostic
    with capsys.disabled():
        print(f"\n[runner] refusal_json={refusal_json}")


def test_ten_megabytes_of_child_stderr_has_a_hard_diagnostic_cap(
        vault_path, tmp_path, executor):
    result = _run(
        "import os\n"
        "os.write(2, b'head ' + b'x' * (10 * 1024 * 1024) + b'TAIL')\n"
        "os._exit(1)\n",
        vault_path, tmp_path, executor,
    )

    assert isinstance(result, env.Refusal)
    assert result.reason == "analyst process exited with status 1"
    assert result.diagnostic is not None
    assert len(result.diagnostic.encode("utf-8")) <= runner.MAX_DIAGNOSTIC_BYTES
    assert "TAIL" in result.diagnostic
    assert "head" not in result.diagnostic
    assert result.diagnostic.endswith('"')


def test_hostile_child_stderr_round_trips_without_controls_or_terminal_breaks(
        vault_path, tmp_path, executor):
    hostile = b'prefix\x00\x08\x1b[31mred\nnext\r\tend number=1234\x7f'
    result = _run(
        "import os\n"
        f"os.write(2, {hostile!r})\n"
        "os._exit(1)\n",
        vault_path, tmp_path, executor,
    )

    assert isinstance(result, env.Refusal)
    assert result.diagnostic is not None
    assert all(char.isprintable() for char in result.diagnostic)
    assert not any(control in result.diagnostic
                   for control in ("\x00", "\x08", "\x1b", "\n", "\r", "\t", "\x7f"))
    assert "red" in result.diagnostic
    assert "1234" not in result.diagnostic
    assert json.loads(json.dumps(result.to_dict()))["diagnostic"] == result.diagnostic


def test_named_refusal_paths_keep_their_parent_authored_reasons(
        vault_path, tmp_path):
    class CannotStart:
        def run_with_query_channel(self, code, run_dir, query_fd, *,
                                   runner_source, limits):
            os.close(query_fd)
            raise sandbox.TransientUnitError(
                "Result=systemd-run-refused: could not start transient unit: "
                "[Errno 2] No such file or directory: 'systemd-run'")

    not_started = runner.run_analyst_code(
        "pass", str(vault_path), str(tmp_path / "not-started"), CannotStart())
    assert isinstance(not_started, env.Refusal)
    assert "TransientUnitError: Result=systemd-run-refused: could not start " \
           "transient unit" in not_started.reason

    class TimedOut:
        def run_with_query_channel(self, code, run_dir, query_fd, *,
                                   runner_source, limits):
            os.close(query_fd)
            return sandbox.RawResult(
                fd3_bytes=b"", fd3_oversized=False, stdout=b"",
                stderr=b"RuntimeMaxSec backstop", returncode=-9,
                timed_out=True, killed_group=True, duration_s=0.01,
                pgid=0, run_dir=Path(run_dir))

    timed_out = runner.run_analyst_code(
        "pass", str(vault_path), str(tmp_path / "timed-out"), TimedOut())
    assert isinstance(timed_out, env.Refusal)
    assert timed_out.reason == "analyst run timed out"


def test_generated_template_has_no_vault_path_literal(vault_path):
    source = runner._runner_source("/parent-owned/code.py")
    assert str(vault_path) not in source
    assert "sqlite3.connect" not in source
    assert "_conn" not in source
