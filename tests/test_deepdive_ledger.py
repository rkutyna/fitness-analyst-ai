"""Tests for the server-owned deep-dive tool-call ledger."""
from __future__ import annotations

import json

import pytest

from health_advisor import deepdive_mcp as D
from health_advisor import llm


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


# --- Atomic line writes (#103) ----------------------------------------------
#
# The ledger's reader treats one undecodable line as "the whole file is
# unreadable", so a reader that catches the file mid-append would blank every
# citation. The writer therefore hands the kernel one complete line in a single
# write, under the ledger's lock.


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
    """Hammer the writer from threads while a reader parses every snapshot."""
    import threading

    path = tmp_path / "ledger.jsonl"
    ledger = D._CallLedger(str(path))
    per_thread, n_threads = 25, 4
    done = threading.Event()
    problems = []

    def reader():
        while not done.is_set():
            try:
                data = path.read_bytes()
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
