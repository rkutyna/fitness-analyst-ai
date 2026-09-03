"""Parent-owned runtime for analyst-mode code.

The analyst process receives a query proxy, not a SQLite object.  The parent
owns the read-only connection and services framed requests from the proxy.
Only the child-produced ``{"tables": [...]}`` value is sent on fd 3; the
ledger used for validation is always read in this process.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import selectors
import select
import signal
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import analyst_envelope as envelope
from . import analyst_ledger as ledger
from . import analyst_sandbox as sandbox

__all__ = [
    "AnalystLimits",
    "DEFAULT_MAX_QUERY_ROWS",
    "MAX_DIAGNOSTIC_BYTES",
    "QueryRowLimitExceeded",
    "RUNNER_TEMPLATE",
    "run_analyst_code",
]


DEFAULT_MAX_QUERY_ROWS = 10_000
FD_OUT = 3
FD_QUERY = 4
_FRAME_HEADER = 4
_MAX_QUERY_FRAME = 1_048_576
_CHUNK = 65_536

# Child stderr is useful for diagnosing substrate failures, but it is an
# untrusted display channel. Keep the entire field (marker and quotes included)
# below a hard byte ceiling; this is deliberately independent of the fd-3
# envelope cap.
MAX_DIAGNOSTIC_BYTES = 600
_DIAGNOSTIC_PREFIX = 'quoted child stderr (untrusted tail): "'
_DIAGNOSTIC_SUFFIX = '"'


@dataclass(frozen=True)
class AnalystLimits:
    """Optional runner limits, including the parent-only query row cap."""

    wall_clock_s: float = sandbox.DEFAULT_WALL_CLOCK_S
    max_fd3_bytes: int = sandbox.DEFAULT_MAX_FD3_BYTES
    max_query_rows: int = DEFAULT_MAX_QUERY_ROWS


class QueryRowLimitExceeded(RuntimeError):
    """Raised in the child when the parent refuses an oversized result."""


# This source is intentionally independent of vault_path.  ``code_path`` is
# parent-authored and is the only path embedded in the generated bootstrap.
RUNNER_TEMPLATE = r'''import base64 as _base64
import json as _json
import os as _os
import struct as _struct
import sys as _sys
import threading as _threading
import traceback as _traceback

_QUERY_FD = __query_fd__
_FD_OUT = __out_fd__
_QUERY_LOCK = _threading.Lock()

class AnalystQueryError(RuntimeError):
    pass

class QueryRowLimitExceeded(AnalystQueryError):
    pass

def _write_all(_fd, _data):
    _view = memoryview(_data)
    while _view:
        _n = _os.write(_fd, _view)
        _view = _view[_n:]

def _read_exact(_fd, _size):
    _parts = []
    _remaining = _size
    while _remaining:
        _part = _os.read(_fd, _remaining)
        if not _part:
            raise AnalystQueryError("query channel closed")
        _parts.append(_part)
        _remaining -= len(_part)
    return b"".join(_parts)

def _decode_value(_value):
    if isinstance(_value, dict) and set(_value) == {"__bytes__"}:
        return _base64.b64decode(_value["__bytes__"].encode("ascii"))
    return _value

class _ProxyCursor:
    def __init__(self, _description, _rows):
        self.description = tuple(tuple(_item) for _item in _description)
        self._rows = [tuple(_decode_value(_v) for _v in _row) for _row in _rows]
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        _row = self.fetchone()
        if _row is None:
            raise StopIteration
        return _row

    def fetchone(self):
        if self._index >= len(self._rows):
            return None
        _row = self._rows[self._index]
        self._index += 1
        return _row

    def fetchmany(self, _size=None):
        if _size is None:
            _size = 1
        _end = min(self._index + _size, len(self._rows))
        _rows = self._rows[self._index:_end]
        self._index = _end
        return _rows

    def fetchall(self):
        _rows = self._rows[self._index:]
        self._index = len(self._rows)
        return _rows

class _QueryProxy:
    def __init__(self, _fd):
        self._fd = _fd

    def __repr__(self):
        return "<AnalystQueryProxy fd=%d>" % self._fd

    def execute(self, sql, params=()):
        if not isinstance(sql, str):
            raise TypeError("sql must be a string")
        _params = list(params)
        _request = _json.dumps({"sql": sql, "params": _params},
                               separators=(",", ":"),
                               allow_nan=False).encode("utf-8")
        _frame = _struct.pack("!I", len(_request)) + _request
        with _QUERY_LOCK:
            _write_all(self._fd, _frame)
            _size = _struct.unpack("!I", _read_exact(self._fd, 4))[0]
            if _size > 1048576:
                raise AnalystQueryError("query response is too large")
            _response = _json.loads(_read_exact(self._fd, _size).decode("utf-8"))
        if not _response.get("ok"):
            _kind = _response.get("error_type", "AnalystQueryError")
            _message = _response.get("error", "query failed")
            if _kind == "QueryRowLimitExceeded":
                raise QueryRowLimitExceeded(_message)
            raise AnalystQueryError(_message)
        return _ProxyCursor(_response["description"], _response["rows"])

conn = _QueryProxy(_QUERY_FD)
_tables = []

def emit(name, columns, units, rows):
    _tables.append({"name": name, "columns": columns, "units": units,
                    "rows": rows})

with open(__code_path__, "r", encoding="utf-8") as _code_file:
    _source = _code_file.read()
_globals = {"__name__": "__main__", "__builtins__": __builtins__,
            "conn": conn, "emit": emit}
try:
    exec(compile(_source, __code_path__, "exec"), _globals)
except BaseException:
    _traceback.print_exc(file=_sys.stderr)
finally:
    _payload = _json.dumps({"tables": _tables}, allow_nan=False,
                           separators=(",", ":")).encode("utf-8")
    _write_all(_FD_OUT, _payload)
    _os.close(_FD_OUT)
'''


class _AnalystQueryError(RuntimeError):
    pass


def _runner_source(code_path: str, query_fd: int = FD_QUERY,
                   out_fd: int = FD_OUT) -> str:
    return (RUNNER_TEMPLATE.replace("__code_path__", repr(code_path))
            .replace("__query_fd__", str(query_fd))
            .replace("__out_fd__", str(out_fd)))


# The transient user-systemd path cannot inherit fd 3 or fd 4.  Keep this as a
# separate source template so the established fd-based executors remain
# byte-for-byte unchanged; its frames are deliberately identical (!I header
# followed by a JSON body in both directions).
NAMED_SOCKET_RUNNER_TEMPLATE = r'''import base64 as _base64
import json as _json
import os as _os
import socket as _socket
import struct as _struct
import sys as _sys
import threading as _threading
import traceback as _traceback

_QUERY_SOCKET = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
_QUERY_SOCKET.connect(_os.environ["ANALYST_QUERY_SOCKET"])
_RESULT_PATH = _os.environ["ANALYST_RESULT_PATH"]
_QUERY_LOCK = _threading.Lock()

class AnalystQueryError(RuntimeError):
    pass

class QueryRowLimitExceeded(AnalystQueryError):
    pass

def _read_exact(_sock, _size):
    _parts = []
    _remaining = _size
    while _remaining:
        _part = _sock.recv(_remaining)
        if not _part:
            raise AnalystQueryError("query channel closed")
        _parts.append(_part)
        _remaining -= len(_part)
    return b"".join(_parts)

def _decode_value(_value):
    if isinstance(_value, dict) and set(_value) == {"__bytes__"}:
        return _base64.b64decode(_value["__bytes__"].encode("ascii"))
    return _value

class _ProxyCursor:
    def __init__(self, _description, _rows):
        self.description = tuple(tuple(_item) for _item in _description)
        self._rows = [tuple(_decode_value(_v) for _v in _row) for _row in _rows]
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        _row = self.fetchone()
        if _row is None:
            raise StopIteration
        return _row

    def fetchone(self):
        if self._index >= len(self._rows):
            return None
        _row = self._rows[self._index]
        self._index += 1
        return _row

    def fetchmany(self, _size=None):
        if _size is None:
            _size = 1
        _end = min(self._index + _size, len(self._rows))
        _rows = self._rows[self._index:_end]
        self._index = _end
        return _rows

    def fetchall(self):
        _rows = self._rows[self._index:]
        self._index = len(self._rows)
        return _rows

class _QueryProxy:
    def __repr__(self):
        return "<AnalystQueryProxy named-socket>"

    def execute(self, sql, params=()):
        if not isinstance(sql, str):
            raise TypeError("sql must be a string")
        _request = _json.dumps(
            {"sql": sql, "params": list(params)},
            separators=(",", ":"), allow_nan=False).encode("utf-8")
        _frame = _struct.pack("!I", len(_request)) + _request
        with _QUERY_LOCK:
            _QUERY_SOCKET.sendall(_frame)
            _size = _struct.unpack("!I", _read_exact(_QUERY_SOCKET, 4))[0]
            if _size > 1048576:
                raise AnalystQueryError("query response is too large")
            _response = _json.loads(
                _read_exact(_QUERY_SOCKET, _size).decode("utf-8"))
        if not _response.get("ok"):
            _kind = _response.get("error_type", "AnalystQueryError")
            _message = _response.get("error", "query failed")
            if _kind == "QueryRowLimitExceeded":
                raise QueryRowLimitExceeded(_message)
            raise AnalystQueryError(_message)
        return _ProxyCursor(_response["description"], _response["rows"])

conn = _QueryProxy()
_tables = []

def emit(name, columns, units, rows):
    _tables.append({"name": name, "columns": columns, "units": units,
                    "rows": rows})

with open(__code_path__, "r", encoding="utf-8") as _code_file:
    _source = _code_file.read()
_globals = {"__name__": "__main__", "__builtins__": __builtins__,
            "conn": conn, "emit": emit}
try:
    exec(compile(_source, __code_path__, "exec"), _globals)
except BaseException:
    _traceback.print_exc(file=_sys.stderr)
finally:
    _payload = _json.dumps({"tables": _tables}, allow_nan=False,
                           separators=(",", ":")).encode("utf-8")
    with open(_RESULT_PATH, "wb") as _result_file:
        _result_file.write(_payload)
    _QUERY_SOCKET.close()
'''


def _named_socket_runner_source(code_path: str) -> str:
    return NAMED_SOCKET_RUNNER_TEMPLATE.replace("__code_path__", repr(code_path))


def _frame(payload: dict) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    if len(body) > _MAX_QUERY_FRAME:
        raise ValueError("query response is too large")
    return len(body).to_bytes(_FRAME_HEADER, "big") + body


def _send_frame(sock: socket.socket, payload: dict) -> None:
    sock.sendall(_frame(payload))


def _emitted_nothing(fd3_bytes: bytes) -> bool:
    """Whether the child's payload carries no table at all."""
    try:
        payload = json.loads(fd3_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and not payload.get("tables")


def _reduce_diagnostic(stderr: bytes, limit: int = 600) -> str:
    """A child traceback, stripped of digits before it can reach a prompt.

    Diagnostics ARE a data channel: the child writes them and a repair turn
    sends them to the provider. Every run of ASCII digits becomes `#`, so
    `no such column: xyz` survives while `assert 1234 == 5678` becomes
    `assert # == #` -- the shape a repair needs, without carrying the
    athlete's figures outward.
    """
    text = stderr.decode("utf-8", "replace").strip()
    tail = text[-limit:] if len(text) > limit else text
    return re.sub(r"\d+", "#", tail).replace("\n", " | ")


def _child_stderr_diagnostic(stderr: bytes) -> str | None:
    """Make a bounded, display-only diagnostic from untrusted child bytes.

    The tail is selected before decoding so a child cannot make the parent
    retain an unbounded traceback. Invalid UTF-8 becomes a replacement
    character, non-printing characters (including newlines and ANSI escape
    controls) are removed, and numeric runs are redacted before the value can
    be exposed outside the refusal headline.
    """
    if not stderr:
        return None

    marker_bytes = (_DIAGNOSTIC_PREFIX + _DIAGNOSTIC_SUFFIX).encode("utf-8")
    tail_budget = MAX_DIAGNOSTIC_BYTES - len(marker_bytes)
    raw_tail = stderr[-tail_budget:]
    text = raw_tail.decode("utf-8", "replace")
    text = "".join(char for char in text if char.isprintable())
    text = re.sub(r"\d+", "#", text)

    # A replacement character can use more bytes than the malformed source
    # byte it represents. Truncate by UTF-8 bytes, without splitting a code
    # point, so the hard cap remains true after sanitisation.
    kept: list[str] = []
    used = 0
    for char in text:
        char_bytes = len(char.encode("utf-8"))
        if used + char_bytes > tail_budget:
            break
        kept.append(char)
        used += char_bytes
    return _DIAGNOSTIC_PREFIX + "".join(kept) + _DIAGNOSTIC_SUFFIX


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_metadata(conn) -> int:
    """Set parent-only SQLite metadata without making it an analyst query."""
    real = conn._conn
    real.set_authorizer(None)
    try:
        real.execute("PRAGMA temp_store = MEMORY")
        return int(real.execute("PRAGMA user_version").fetchone()[0])
    finally:
        real.set_authorizer(conn._authorize)


def _limits_for_executor(limits: Any):
    if limits is None:
        return sandbox.RunLimits()
    if isinstance(limits, sandbox.RunLimits):
        return limits
    return sandbox.RunLimits(
        wall_clock_s=float(getattr(limits, "wall_clock_s", sandbox.DEFAULT_WALL_CLOCK_S)),
        max_fd3_bytes=int(getattr(limits, "max_fd3_bytes", sandbox.DEFAULT_MAX_FD3_BYTES)),
    )


def _max_query_rows(limits: Any) -> int:
    if limits is None:
        return DEFAULT_MAX_QUERY_ROWS
    if isinstance(limits, dict):
        value = limits.get("max_query_rows", DEFAULT_MAX_QUERY_ROWS)
    else:
        value = getattr(limits, "max_query_rows", DEFAULT_MAX_QUERY_ROWS)
    value = int(value)
    if value < 1:
        raise ValueError("max_query_rows must be positive")
    return value


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"unsupported SQLite value type: {type(value).__name__}")


def _decode_params(value):
    if not isinstance(value, list):
        raise TypeError("params must be a JSON array")
    result = []
    for item in value:
        if isinstance(item, (type(None), bool, int, float, str)):
            result.append(item)
        elif isinstance(item, dict) and set(item) == {"__bytes__"}:
            result.append(base64.b64decode(item["__bytes__"].encode("ascii")))
        else:
            raise TypeError("params contain an unsupported value")
    return result


def _service_query(sock: socket.socket, conn, max_rows: int,
                   pending: bytearray) -> tuple[bool, str | None]:
    try:
        chunk = sock.recv(_CHUNK)
    except BlockingIOError:
        return True, None
    if not chunk:
        if pending:
            raise _AnalystQueryError("query request has a truncated frame")
        return False, None
    pending.extend(chunk)
    if len(pending) < _FRAME_HEADER:
        return True, None
    size = int.from_bytes(pending[:_FRAME_HEADER], "big")
    if size > _MAX_QUERY_FRAME:
        raise _AnalystQueryError("query request is too large")
    if len(pending) < _FRAME_HEADER + size:
        return True, None
    body = bytes(pending[_FRAME_HEADER:_FRAME_HEADER + size])
    del pending[:_FRAME_HEADER + size]
    request = json.loads(body.decode("utf-8"))
    try:
        sql = request["sql"]
        params = _decode_params(request.get("params", []))
        if not isinstance(sql, str):
            raise TypeError("sql must be a string")
        cursor = conn.execute(sql, params)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            reason = f"query result exceeds row cap: {len(rows)} > {max_rows}"
            _send_frame(sock, {"ok": False, "error_type": "QueryRowLimitExceeded",
                                "error": reason})
            return True, reason
        encoded_rows = [[_json_value(value) for value in row] for row in rows]
        description = [list(item) for item in (cursor.description or ())]
        _send_frame(sock, {"ok": True, "description": description,
                           "rows": encoded_rows})
        return True, None
    except QueryRowLimitExceeded as exc:
        reason = str(exc)
        _send_frame(sock, {"ok": False, "error_type": "QueryRowLimitExceeded",
                            "error": reason})
        return True, reason
    except Exception as exc:
        _send_frame(sock, {"ok": False, "error_type": "AnalystQueryError",
                            "error": f"{type(exc).__name__}: {exc}"})
        return True, None


def _high_fd(fd: int) -> int:
    """Move a channel end away from the fixed child descriptors."""
    duplicate = fcntl.fcntl(fd, fcntl.F_DUPFD, 10)
    os.close(fd)
    return duplicate


def _run_seatbelt(executor, code: str, run_dir: str, query_fd: int,
                  limits: Any, vault_path: str):
    """Seatbelt invocation with fd 3 output and fd 4 query input/output."""
    del vault_path
    run_dir_real, work_dir_real = executor._prepare_run_dir(run_dir)
    code_path = run_dir_real / "code.py"
    code_path.write_text(code, encoding="utf-8")
    runner_path = run_dir_real / "runner.py"
    # Written below, once the real descriptor numbers are known -- see the
    # comment on the Popen call.
    profile_path = run_dir_real / "profile.sb"
    profile_text = sandbox.build_profile(
        real_home=executor._real_home,
        pyenv_prefix=executor._pyenv_prefix,
        venv_dir=executor._venv_dir,
        pkg_dir=None,
        work_dir=str(work_dir_real),
    )
    # The child must not be able to read the parent-owned profile back as an
    # information channel.
    quoted_profile = sandbox._scheme_quote(str(profile_path))
    profile_text += f'\n(deny file-read-data (literal "{quoted_profile}"))\n'
    profile_path.write_text(profile_text, encoding="utf-8")

    argv = [executor.SANDBOX_EXEC, "-f", str(profile_path), executor._python,
            "-I", str(runner_path)]
    env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(work_dir_real),
           "HOME": str(work_dir_real), "ANALYST_QUERY_FD": str(FD_QUERY)}

    out_r, out_w = os.pipe()
    out_r = _high_fd(out_r)
    out_w = _high_fd(out_w)
    query_fd = _high_fd(query_fd)

    # The child is told its descriptor numbers instead of having them dup2'd
    # into fixed slots by a `preexec_fn`. `preexec_fn` is documented as unsafe
    # in a process with threads, and `run_analyst_code` calls this from a worker
    # thread so the parent can service queries while the child runs -- which is
    # exactly how this failed the first time it met a real seatbelt executor:
    # the child died with EBADF before exec, and the whole run came back as
    # "analyst executor failed: OSError: [Errno 9] Bad file descriptor".
    # `pass_fds` alone keeps the descriptors open across the close-fds sweep,
    # and the template already parameterises both numbers.
    runner_path.write_text(
        _runner_source(str(code_path), query_fd=query_fd, out_fd=out_w),
        encoding="utf-8")

    start = time.monotonic()
    proc = subprocess.Popen(
        argv, cwd=str(work_dir_real), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=(out_w, query_fd),
        start_new_session=True, close_fds=True,
    )
    os.close(out_w)
    os.close(query_fd)

    fd3_buf = bytearray()
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    fd3_oversized = False
    sel = selectors.DefaultSelector()
    sel.register(out_r, selectors.EVENT_READ, "fd3")
    sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, "stderr")
    run_limits = _limits_for_executor(limits)
    deadline = start + run_limits.wall_clock_s
    timed_out = False

    while sel.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        for key, _ in sel.select(min(remaining, 0.25)):
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
                if len(fd3_buf) < sandbox._FD3_DRAIN_CEILING:
                    fd3_buf.extend(chunk)
                    if len(fd3_buf) > run_limits.max_fd3_bytes:
                        fd3_oversized = True
                else:
                    fd3_oversized = True
            elif key.data == "stdout":
                stdout_buf.extend(chunk)
            else:
                stderr_buf.extend(chunk)

    killed_group = False
    if timed_out:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            killed_group = True
        except (ProcessLookupError, PermissionError):
            pass
        grace_deadline = time.monotonic() + 5
        while sel.get_map() and time.monotonic() < grace_deadline:
            for key, _ in sel.select(0.25):
                fileobj = key.fileobj
                fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
                try:
                    chunk = os.read(fd, _CHUNK)
                except OSError:
                    chunk = b""
                if not chunk:
                    sel.unregister(fileobj)
                    continue
                if key.data == "fd3":
                    if len(fd3_buf) < sandbox._FD3_DRAIN_CEILING:
                        fd3_buf.extend(chunk)
                elif key.data == "stdout":
                    stdout_buf.extend(chunk)
                else:
                    stderr_buf.extend(chunk)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    for fileobj in (out_r, proc.stdout, proc.stderr):
        try:
            if isinstance(fileobj, int):
                os.close(fileobj)
            else:
                fileobj.close()
        except Exception:
            pass
    return sandbox.RawResult(
        fd3_bytes=(bytes(fd3_buf[:run_limits.max_fd3_bytes])
                   if fd3_oversized else bytes(fd3_buf)),
        fd3_oversized=fd3_oversized, stdout=bytes(stdout_buf),
        stderr=bytes(stderr_buf), returncode=proc.returncode,
        timed_out=timed_out, killed_group=killed_group,
        duration_s=time.monotonic() - start, pgid=proc.pid,
        run_dir=run_dir_real,
    )


def _invoke_executor(executor, code: str, run_dir: str, query_fd: int,
                     limits: Any, vault_path: str):
    if isinstance(executor, sandbox.SeatbeltExecutor):
        return _run_seatbelt(executor, code, run_dir, query_fd, limits,
                             vault_path)
    if isinstance(executor, sandbox.TransientUnitExecutor):
        return executor.run_with_named_query_channel(
            code, run_dir, query_fd,
            runner_source=_named_socket_runner_source("code.py"),
            limits=_limits_for_executor(limits))
    method = getattr(executor, "run_with_query_channel", None)
    if method is None:
        raise TypeError("executor must support the analyst query channel")
    return method(code, run_dir, query_fd,
                  runner_source=_runner_source("code.py"),
                  limits=_limits_for_executor(limits))


def run_analyst_code(
    code: str,
    vault_path: str,
    run_dir: str,
    executor,
    *,
    limits=None,
) -> "envelope.Envelope | envelope.Refusal":
    """Run arbitrary analyst Python with parent-mediated SQL execution."""
    # Compile in the parent before spending a sandbox run. The model available
    # under D15's provider pin is a flash model, and measured against real
    # questions it produces a syntax error often enough that this is the common
    # path, not the rare one (#194). Catching it here costs nothing, names the
    # line, and gives the repair turn something exact to fix instead of a
    # traceback recovered from the child's stderr.
    try:
        compile(code, "<analyst>", "exec")
    except SyntaxError as exc:
        return envelope.Refusal(
            f"SYNTAX_ERROR: the generated code does not parse -- line "
            f"{exc.lineno}: {exc.msg}. Offending text: "
            f"{(exc.text or '').strip()[:120]!r}")
    max_rows = _max_query_rows(limits)
    parent_conn = ledger.open_ledgered(vault_path)
    query_parent, query_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    query_child_fd = query_child.detach()
    raw_result: list[Any] = []
    worker_error: list[BaseException] = []

    def _worker():
        try:
            raw_result.append(_invoke_executor(
                executor, code, run_dir, query_child_fd, limits, vault_path))
        except BaseException as exc:
            worker_error.append(exc)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    query_parent.setblocking(False)
    cap_reason = None
    pending = bytearray()
    outcome = None
    try:
        while worker.is_alive():
            ready, _, _ = select.select([query_parent], [], [], 0.25)
            if not ready:
                continue
            try:
                more, reason = _service_query(
                    query_parent, parent_conn, max_rows, pending)
                if reason is not None:
                    cap_reason = reason
                if not more:
                    break
            except (ConnectionError, _AnalystQueryError, json.JSONDecodeError):
                break
    finally:
        query_parent.close()
        worker.join(timeout=6)
    parent_ledger = parent_conn.ledger.as_dict()
    # This ledger was accumulated by THIS process's authorizer, against a
    # connection the child never held. The flag is what lets a caller say so
    # honestly rather than defaulting to the child-asserted caveat of #226.
    parent_ledger["parent_observed"] = True
    try:
        parent_version = _parent_metadata(parent_conn)
        vault_sha = _hash_file(vault_path)
        code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        run_id = uuid.uuid4().hex
        if worker_error:
            outcome = envelope.Refusal(
                f"analyst executor failed: {type(worker_error[0]).__name__}: "
                f"{worker_error[0]}")
        elif not raw_result:
            outcome = envelope.Refusal("analyst executor returned no result")
        else:
            result = raw_result[0]
            if cap_reason is not None:
                outcome = envelope.Refusal(cap_reason)
            elif result.timed_out:
                outcome = envelope.Refusal("analyst run timed out")
            elif result.fd3_oversized:
                outcome = envelope.Refusal(
                    f"analyst output exceeds {sandbox.DEFAULT_MAX_FD3_BYTES} bytes")
            elif result.returncode not in (0, None):
                outcome = envelope.Refusal(
                    f"analyst process exited with status {result.returncode}",
                    diagnostic=_child_stderr_diagnostic(result.stderr))
            elif not result.fd3_bytes:
                outcome = envelope.Refusal("analyst produced no fd-3 envelope")
            elif _emitted_nothing(result.fd3_bytes) and result.stderr:
                # The child caught an exception, printed it, and emitted an
                # empty envelope -- which validates perfectly and renders as
                # silence. Measured on the first real run: the model's SQL
                # carried a stray combining character, every query raised, and
                # the user was shown provenance and code with no answer and no
                # reason. An empty result is a legitimate answer only when
                # nothing went wrong; when the child also wrote to stderr it is
                # a failure and must say so.
                outcome = envelope.Refusal(
                    "EXEC_FAILED: the analyst code raised before emitting any "
                    "table. Reduced diagnostic: "
                    + _reduce_diagnostic(result.stderr))
            else:
                outcome = envelope.validate(
                    result.fd3_bytes, run_id=run_id, question="",
                    code_sha256=code_sha, vault_sha256=vault_sha,
                    vault_version=parent_version, ledger=parent_ledger)
    finally:
        parent_conn.close()
    return outcome
