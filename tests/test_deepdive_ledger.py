"""Tests for the server-owned deep-dive tool-call ledger."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import threading

import httpx
import pytest

from health_advisor import chat
from health_advisor import deepdive_mcp as D
from health_advisor import ledger_lock
from health_advisor import llm
from scripts import trace_researcher_answer as trace


def _server_tool(server, name):
    return next(t.fn for t in server._tool_manager.list_tools() if t.name == name)


def _ledger_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_direct_deepdive_call_writes_name_arguments_result_and_sequence(
        conn, vault, tmp_path):
    """This fails before the ledger change: build_server has no ledger path and
    a direct server tool call creates no call record.
    """
    ledger_path = tmp_path / "run_ledger.jsonl"
    server = D.build_server(vault, ledger_path=str(ledger_path))

    result = _server_tool(server, "get_latest")("body_mass")

    records = _ledger_records(ledger_path)
    assert len(records) == 1
    assert records[0]["tool_name"] == "get_latest"
    assert records[0]["arguments"] == {"metric": "body_mass"}
    assert records[0]["result"] == {k: v for k, v in result.items()
                                     if k != "_ledger"}
    assert result["_ledger"] == {"sequence": 1}
    assert records[0]["sequence"] == 1
    assert records[0]["result_elided"] is False


def test_ledger_result_elision_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "LEDGER_RESULT_MAX_BYTES", 16)
    path = tmp_path / "large.jsonl"
    ledger = D._CallLedger(str(path))
    wrapped = D._ledger_wrapper("synthetic", lambda: {"payload": "x" * 100}, ledger)

    wrapped()

    record = _ledger_records(path)[0]
    assert record["result_elided"] is True
    assert record["result"]["_elided"] is True
    assert record["result"]["bytes"] > 16
    assert record["result"]["sha256"]


def test_ledger_cannot_be_the_vault(vault):
    with pytest.raises(ValueError, match="separate from the vault"):
        D.build_server(vault, ledger_path=str(vault.db_path))


def test_absent_figure_is_not_traceable_from_synthetic_ledger(tmp_path):
    """The check is intentionally inline: #45 owns the eventual verifier."""
    path = tmp_path / "synthetic.jsonl"
    path.write_text(json.dumps({
        "sequence": 1,
        "tool_name": "get_workouts",
        "arguments": {"period": "7d"},
        "result": {"jog_minutes": 42.5},
        "result_elided": False,
    }) + "\n")
    records = _ledger_records(path)

    def contains(value, figure):
        if isinstance(value, dict):
            return any(contains(v, figure) for v in value.values())
        if isinstance(value, list):
            return any(contains(v, figure) for v in value)
        return value == figure

    figures = [42.5, 99.0]
    traceable = [figure for figure in figures
                 if any(contains(record["result"], figure) for record in records)]
    assert traceable == [42.5]
    assert 99.0 not in traceable


def test_ledger_path_reaches_launcher_on_argv(vault, tmp_path):
    ledger = str(tmp_path / "explicit.jsonl")
    values = llm._deepdive_mcp_config(vault, ledger_path=ledger)
    joined = " ".join(value for value in values if "ledger" in value)
    assert f'"--ledger", "{ledger}"' in joined


def test_scratch_run_gets_sibling_ledger_argv(vault, tmp_path):
    scratch = str(tmp_path / "deepdive_scratch.json")
    values = llm._deepdive_mcp_config(vault, scratch_path=scratch, task_id=7)
    joined = " ".join(values)
    assert '"--scratch", "' + scratch + '"' in joined
    assert '"--ledger", "' + str(tmp_path / "deepdive_scratch_ledger.jsonl") + '"' in joined


def test_ledger_wrapper_publishes_unambiguous_claim_period_vocabulary():
    result = {"metric": "jog_minutes", "period": {
        "start": "2026-07-27", "end": "2026-08-17",
        "period_starts": ["2026-07-27", "2026-08-03",
                           "2026-08-10", "2026-08-17"],
    }}

    vocabulary = D._claim_period_vocabulary(result)

    assert vocabulary == [{
        "metric": "jog_minutes",
        "claim_period": "2026-07-27:2026-08-23",
        "ledger_period": result["period"],
    }]


# The published claim_period must span the last bucket by the spacing the
# payload itself shows. Measured 2026-08-24: a hardcoded +6 days published
# "2026-06-29:2026-07-08" for a day-spaced block whose period ended
# 2026-07-02. The model is instructed to copy this string verbatim, so the
# gate would have ACCEPTED a claim whose period overstated its own window by
# six days -- a verified figure with a wrong scope, which is the exact failure
# the claim channel exists to prevent.
def _vocab(start, end, starts):
    from health_advisor import deepdive_mcp
    payload = {"result": {"block": {"metric": "jog_minutes", "mean": 1.0,
               "period": {"start": start, "end": end, "period_starts": starts}}}}
    return deepdive_mcp._claim_period_vocabulary(payload)[0]["claim_period"]


def test_claim_period_spans_the_last_bucket_by_measured_spacing():
    # weekly — the shape of the live 2026-08-24 run, which verified 3/3
    assert _vocab("2026-06-29", "2026-07-20",
                  ["2026-06-29", "2026-07-06", "2026-07-13", "2026-07-20"]
                  ) == "2026-06-29:2026-07-26"
    # daily — the case a hardcoded week overstated by six days
    assert _vocab("2026-06-29", "2026-07-02",
                  ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"]
                  ) == "2026-06-29:2026-07-02"
    # fortnightly — spacing is read, not assumed
    assert _vocab("2026-06-01", "2026-06-15",
                  ["2026-06-01", "2026-06-15"]) == "2026-06-01:2026-06-28"


def test_claim_period_with_one_bucket_uses_the_payloads_own_end():
    """Spacing is unknowable from a single start; the payload's end is honest."""
    assert _vocab("2026-06-29", "2026-07-05", ["2026-06-29"]) == "2026-06-29:2026-07-05"


# --- Atomic line writes (#103) and the reader's lock (#116) ------------------
#
# The ledger's reader treats one undecodable line as "the whole file is
# unreadable", so a reader that catches the file mid-append would blank every
# citation. The writer therefore hands the kernel one complete line in a single
# write, under the ledger's lock.
#
# One write is not enough on Linux: a read there does not wait for a write in
# progress, and an unlocked reader saw a partial last line in 11 of 50 runs on
# overlayfs and 14 of 50 on ext4 (#116), every partial snapshot ending on a
# 4096-byte page boundary while the writer issued exactly one full-length write
# per record. So the append holds an exclusive flock and every reader a shared
# one (health_advisor.ledger_lock).


def test_each_record_reaches_the_file_in_exactly_one_write(tmp_path, monkeypatch):
    """The record is one complete newline-terminated line in one os.write."""
    import os as _os

    writes = []
    real_write = _os.write

    def spy(fd, data):
        writes.append(bytes(data))
        return real_write(fd, data)

    monkeypatch.setattr(D.os, "write", spy)
    ledger = D._CallLedger(str(tmp_path / "ledger.jsonl"))

    # Larger than any stdio buffer, so a buffered json.dump would split it.
    ledger.append("big", {"n": 1}, {"text": "x" * 100_000})
    ledger.append("small", {}, {"ok": True})

    assert len(writes) == 2
    for data in writes:
        assert data.endswith(b"\n") and data.count(b"\n") == 1
        json.loads(data)
    assert b"".join(writes) == (tmp_path / "ledger.jsonl").read_bytes()


def test_a_concurrent_reader_never_sees_a_partial_line(tmp_path):
    """Hammer the writer from threads while a reader parses every snapshot.

    The reader reads the way every engine reader of the ledger does, under
    ``ledger_lock.lock_for_read``. Without that lock this fails on Linux (#116).
    """
    import threading

    path = tmp_path / "ledger.jsonl"
    ledger = D._CallLedger(str(path))
    per_thread, n_threads = 25, 4
    done = threading.Event()
    problems = []

    def reader():
        while not done.is_set():
            try:
                with open(path, "rb") as fh:
                    ledger_lock.lock_for_read(fh)
                    data = fh.read()
            except FileNotFoundError:
                continue
            if not data:
                continue
            if not data.endswith(b"\n"):
                problems.append("partial tail")
                return
            try:
                for line in data.splitlines():
                    json.loads(line)
            except ValueError:
                problems.append("undecodable line")
                return

    def writer(i):
        for j in range(per_thread):
            ledger.append("t", {"i": i, "j": j}, {"text": "y" * 60_000})

    watcher = threading.Thread(target=reader)
    watcher.start()
    workers = [threading.Thread(target=writer, args=(i,))
               for i in range(n_threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    done.set()
    watcher.join()

    assert problems == []
    sequences = [r["sequence"] for r in _ledger_records(path)]
    assert sequences == list(range(1, per_thread * n_threads + 1))


# --- Deterministic: an append in progress, held the way the writer holds it ---
#
# The hammer test above needs the race to happen, and on macOS it never did.
# These do not need a race. Each one holds the ledger the way the writer does
# mid-append -- the exclusive flock taken and the first half of the line on
# disk -- and requires the reader under test to wait for the rest. A reader
# that skips the lock returns at once on every platform, with the torn tail:
# [] from the verifier, an empty index, a skipped sequence or a decode error.

# An unlocked reader of a two-record file returns in milliseconds; a locked one
# stays blocked for as long as the lock is held, so this cannot pass by luck.
_BLOCKED_FOR_S = 0.3


def _record(sequence):
    return {"sequence": sequence, "tool_name": "get_latest",
            "arguments": {"metric": "body_mass"},
            "result": {"metric": "body_mass", "value": 80.0 + sequence},
            "result_elided": False}


def _line(record):
    return (json.dumps(record, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")


@contextlib.contextmanager
def _append_in_progress(path, record):
    """Hold ``path`` mid-append: exclusive flock, half of ``record`` written.

    Leaving the block writes the other half and closes, releasing the lock.
    """
    line = _line(record)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line[:len(line) // 2])
        yield
        os.write(fd, line[len(line) // 2:])
    finally:
        os.close(fd)


def _start(fn):
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # surfaced by the test, not lost here
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def _index_sequences(path):
    text = llm._ledger_index_text(path)
    return [int(n) for n in re.findall(r"^sequence (\d+) — ", text, re.M)]


_READERS = {
    "chat._read_ledger (the verifier)":
        (lambda p: [r["sequence"] for r in chat._read_ledger(p)], [1, 2]),
    "llm._ledger_index_text (the model's index)":
        (_index_sequences, [1, 2]),
    "_CallLedger continuing a reused path":
        (lambda p: D._CallLedger(p)._next_sequence, 3),
    "scripts/trace_researcher_answer.read_ledger":
        (lambda p: [r["sequence"] for r in trace.read_ledger(p)], [1, 2]),
}


@pytest.mark.parametrize("name", sorted(_READERS))
def test_every_ledger_reader_waits_for_an_append_in_progress(tmp_path, name):
    read, expected = _READERS[name]
    path = str(tmp_path / "ledger.jsonl")
    with open(path, "wb") as fh:
        fh.write(_line(_record(1)))

    with _append_in_progress(path, _record(2)):
        thread, box = _start(lambda: read(path))
        thread.join(_BLOCKED_FOR_S)
        returned_mid_append = not thread.is_alive()
        seen_mid_append = dict(box)
    thread.join(10)

    assert not returned_mid_append, (
        f"{name} read the ledger mid-append and returned {seen_mid_append}")
    assert "error" not in box, box.get("error")
    assert box["value"] == expected


def test_the_tool_loop_sequence_scan_waits_for_an_append_in_progress(
        tmp_path, monkeypatch, vault):
    """``tool_loop`` numbers its calls by scanning the ledger it continues."""
    # Isolate the scan: the loop's own _CallLedger would otherwise read the
    # file first and do the waiting for it.
    monkeypatch.setattr(llm, "_ledgered", lambda reg, ledger_path: reg)
    monkeypatch.setattr(llm, "_registry", lambda ctx, include=None: {})
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setattr(llm, "OPENROUTER_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "coreweave/fp8")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_SORT", "")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")
    turns = {"n": 0}

    def reply(message):
        return httpx.Response(200, json={
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return reply({"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": "get_latest", "arguments": "{}"}}]})
        return reply({"role": "assistant", "content": "done"})

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    path = str(tmp_path / "ledger.jsonl")
    with open(path, "wb") as fh:
        fh.write(_line(_record(1)))
    numbered = []

    with _append_in_progress(path, _record(2)):
        thread, box = _start(lambda: llm.tool_loop(
            "q", ctx=vault, tools=[], max_turns=3, ledger_path=path,
            on_tool_call=lambda name, sequence: numbered.append(sequence)))
        thread.join(_BLOCKED_FOR_S)
        returned_mid_append = not thread.is_alive()
    thread.join(10)

    assert not returned_mid_append
    assert "error" not in box, box.get("error")
    assert numbered == [3]


def test_an_append_waits_while_a_reader_holds_the_ledger(tmp_path):
    """The writer's half of the protocol: no write lands under a reader's lock."""
    path = tmp_path / "ledger.jsonl"
    ledger = D._CallLedger(str(path))
    with open(path, "ab") as reader:
        fcntl.flock(reader.fileno(), fcntl.LOCK_SH)
        thread, box = _start(lambda: ledger.append("t", {}, {"ok": True}))
        thread.join(_BLOCKED_FOR_S)
        appended_under_read_lock = not thread.is_alive()
        size_under_read_lock = path.stat().st_size
    thread.join(10)

    assert not appended_under_read_lock
    assert size_under_read_lock == 0
    assert box == {"value": 1}
    assert [r["sequence"] for r in _ledger_records(path)] == [1]
