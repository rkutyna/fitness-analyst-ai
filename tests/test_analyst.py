"""Tests for health_advisor/analyst.py -- the analyst-mode CLI.

No real model and no real sandbox are ever touched here: every test injects
a fake `complete_fn` and a fake `run_code_fn` (matching analyst_runner's
stated `run_analyst_code(code, vault_path, run_dir, executor, *, limits)`
signature). `tests/conftest.py`'s autouse `_isolate_db`/`_ollama_backend`
fixtures keep this suite off the production DB and off the codex backend by
default; the codex-refusal test explicitly opts back into "codex".
"""
from __future__ import annotations

import io
import json

import pytest

from health_advisor import analyst
from health_advisor import db as dbmod
from health_advisor import llm
from health_advisor.analyst_envelope import Envelope, Refusal


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def vault_path(tmp_path):
    """A tiny, schema-initialized vault file on disk (path, not connection --
    analyst.py takes a path from argv, never an open connection)."""
    p = tmp_path / "vault.db"
    conn = dbmod.connect(p)
    dbmod.init_db(conn)
    conn.close()
    return str(p)


def _fake_complete(reply: str):
    """A complete_fn stand-in with a call counter, fenced-code reply."""
    calls: list[str] = []

    def _complete(prompt: str, **kwargs) -> str:
        calls.append(prompt)
        return reply

    _complete.calls = calls
    return _complete


def _sample_envelope(*, ledger=None) -> Envelope:
    table = {
        "name": "resting_hr_by_week",
        "columns": ("week", "mean_bpm"),
        "units": ("count", "count/min"),
        "rows": ((1, 52.5), (2, 53.25)),
        "row_count": 2,
    }
    return Envelope(
        run_id="child-run-id",
        question="unused",
        code_sha256="child-code-sha",
        vault_sha256="child-vault-sha",
        vault_version=0,
        ledger=ledger if ledger is not None else {
            "query_count": 3, "tables_read": ["daily_metrics"], "rows_read": 14,
        },
        tables=(table,),
        counts={"rows": 2, "cells": 4, "numeric_tokens": 4, "bytes": 120},
    )


# --------------------------------------------------------------------------- #
# 1. The codex backend is refused
# --------------------------------------------------------------------------- #
def test_codex_backend_is_refused(monkeypatch, vault_path, capsys):
    monkeypatch.setattr(llm, "BACKEND", "codex")
    rc = analyst.main(["--vault", vault_path, "--question", "q"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "codex" in err.lower()
    assert "refuses" in err.lower() or "refused" in err.lower()


def test_assert_analyst_backend_approved_raises_for_codex(monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "codex")
    with pytest.raises(RuntimeError, match="codex"):
        analyst.assert_analyst_backend_approved()


def test_assert_analyst_backend_approved_ok_for_ollama(monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "ollama")
    analyst.assert_analyst_backend_approved()  # must not raise


# --------------------------------------------------------------------------- #
# 2. A successful run renders every envelope cell and no invented number
# --------------------------------------------------------------------------- #
def test_successful_run_renders_every_envelope_cell(tmp_path, vault_path):
    envelope = _sample_envelope()
    complete_fn = _fake_complete("```python\nemit('resting_hr_by_week', ...)\n```")
    run_calls: list[tuple] = []

    def fake_run_code(code, vpath, run_dir, executor, *, limits=None):
        run_calls.append((code, vpath, run_dir, executor, limits))
        return envelope

    out = io.StringIO()
    run_dir = str(tmp_path / "run")
    rc = analyst.run_analyst(
        "did resting HR rise?", vault_path, run_dir,
        complete_fn=complete_fn, run_code_fn=fake_run_code,
        executor=object(), out=out)

    assert rc == 0
    text = out.getvalue()

    # The question and both table cells are present, verbatim.
    assert "did resting HR rise?" in text
    assert "resting_hr_by_week" in text
    for row in envelope.tables[0]["rows"]:
        for cell in row:
            assert str(cell) in text

    # Column headers and units are shown.
    assert "week" in text
    assert "mean_bpm" in text
    assert "count/min" in text

    # The ledger figures actually reached from the envelope are shown.
    assert "3" in text  # query_count
    assert "daily_metrics" in text
    assert "14" in text  # rows_read

    # The model-written code is echoed back.
    assert "emit(" in text

    # run_code_fn was called exactly once (no refusal, no retry).
    assert len(run_calls) == 1
    # complete_fn was called exactly once too.
    assert len(complete_fn.calls) == 1


def test_ledger_is_labelled_child_asserted_by_default():
    text = analyst.ledger_trust_label({"query_count": 1})
    assert "child-asserted" in text
    assert "226" in text  # cites the tracking issue


def test_ledger_is_labelled_parent_observed_when_reported():
    text = analyst.ledger_trust_label({"parent_observed": True})
    assert text == "parent-observed"


# --------------------------------------------------------------------------- #
# 3. A refusal triggers exactly ONE retry and no more
# --------------------------------------------------------------------------- #
def test_refusal_triggers_exactly_one_retry(tmp_path, vault_path):
    refusal = Refusal("emitted 1 numeric tables from 0 vault tables and 0 reads")
    complete_fn = _fake_complete("```python\nemit('x', ...)\n```")
    run_calls: list[str] = []

    def always_refuses(code, vpath, run_dir, executor, *, limits=None):
        run_calls.append(code)
        return refusal

    out = io.StringIO()
    rc = analyst.run_analyst(
        "some question", vault_path, str(tmp_path / "run"),
        complete_fn=complete_fn, run_code_fn=always_refuses,
        executor=object(), out=out)

    assert rc != 0
    # Exactly two attempts: the first code, and exactly one repair.
    assert len(run_calls) == 2
    assert len(complete_fn.calls) == 2
    # The repair prompt carries the first refusal's reason.
    assert refusal.reason in complete_fn.calls[1]

    text = out.getvalue()
    assert refusal.reason in text
    assert "Remediation" in text


def test_refusal_after_repair_prints_reason_and_remediation(tmp_path, vault_path):
    """A second refusal (repair did not help) is still reported cleanly, not
    retried again."""
    first = Refusal("table 'x' exceeds row cap: 500 > 200")
    second = Refusal("envelope exceeds 65536 bytes (70000 bytes)")
    complete_fn = _fake_complete("```python\nemit('x', ...)\n```")
    results = iter([first, second])

    def fake_run_code(code, vpath, run_dir, executor, *, limits=None):
        return next(results)

    out = io.StringIO()
    rc = analyst.run_analyst(
        "q", vault_path, str(tmp_path / "run"),
        complete_fn=complete_fn, run_code_fn=fake_run_code,
        executor=object(), out=out)

    assert rc != 0
    text = out.getvalue()
    assert second.reason in text
    assert first.reason not in text  # only the FINAL refusal is reported


# --------------------------------------------------------------------------- #
# 4. The run record contains the code hash and the ledger
# --------------------------------------------------------------------------- #
def test_run_record_contains_code_hash_and_ledger(tmp_path, vault_path):
    import hashlib

    code_reply = "```python\nemit('resting_hr_by_week', ...)\n```"
    envelope = _sample_envelope()
    complete_fn = _fake_complete(code_reply)

    def fake_run_code(code, vpath, run_dir, executor, *, limits=None):
        return envelope

    run_dir = tmp_path / "run"
    rc = analyst.run_analyst(
        "q", vault_path, str(run_dir),
        complete_fn=complete_fn, run_code_fn=fake_run_code,
        executor=object(), out=io.StringIO())
    assert rc == 0

    record_path = run_dir / "run_record.json"
    assert record_path.exists()
    record = json.loads(record_path.read_text())

    expected_code = analyst.extract_code(code_reply)
    expected_sha = hashlib.sha256(expected_code.encode("utf-8")).hexdigest()

    assert record["code"]["initial"] == expected_code
    assert record["run_id"]
    assert record["question"] == "q"
    assert record["ledger"] == dict(envelope.ledger)
    assert record["result"]["tables"][0]["name"] == "resting_hr_by_week"
    assert record["vault_sha256"]
    assert record["started_at"] and record["finished_at"]

    # The printed code_sha256 in stdout matches the hash of the code that was
    # actually run.
    out = io.StringIO()
    analyst.run_analyst(
        "q", vault_path, str(tmp_path / "run2"),
        complete_fn=_fake_complete(code_reply), run_code_fn=fake_run_code,
        executor=object(), out=out)
    assert expected_sha in out.getvalue()


def test_run_record_records_both_prompts_on_repair(tmp_path, vault_path):
    refusal = Refusal("emitted 1 numeric tables from 0 vault tables and 0 reads")
    envelope = _sample_envelope()
    complete_fn = _fake_complete("```python\nemit('x', ...)\n```")
    outcomes = iter([refusal, envelope])

    def fake_run_code(code, vpath, run_dir, executor, *, limits=None):
        return next(outcomes)

    run_dir = tmp_path / "run"
    rc = analyst.run_analyst(
        "q", vault_path, str(run_dir),
        complete_fn=complete_fn, run_code_fn=fake_run_code,
        executor=object(), out=io.StringIO())
    assert rc == 0

    record = json.loads((run_dir / "run_record.json").read_text())
    assert record["prompts"]["initial"] is not None
    assert record["prompts"]["repair"] is not None
    assert refusal.reason in record["prompts"]["repair"]
    assert record["code"]["repair"] is not None


# --------------------------------------------------------------------------- #
# 5. --json output round-trips
# --------------------------------------------------------------------------- #
def test_json_output_round_trips_on_success(tmp_path, vault_path):
    envelope = _sample_envelope()
    complete_fn = _fake_complete("```python\nemit('x', ...)\n```")

    def fake_run_code(code, vpath, run_dir, executor, *, limits=None):
        return envelope

    out = io.StringIO()
    rc = analyst.run_analyst(
        "did it rise?", vault_path, str(tmp_path / "run"),
        complete_fn=complete_fn, run_code_fn=fake_run_code,
        executor=object(), json_output=True, out=out)
    assert rc == 0

    payload = json.loads(out.getvalue())  # must parse cleanly
    assert payload["question"] == "did it rise?"
    assert payload["tables"][0]["name"] == "resting_hr_by_week"
    assert payload["tables"][0]["rows"] == [[1, 52.5], [2, 53.25]]
    assert payload["provenance"]["ledger"]["query_count"] == 3
    assert "child-asserted" in payload["provenance"]["ledger"]["provenance"]
    assert payload["code"]


def test_json_output_round_trips_on_refusal(tmp_path, vault_path):
    refusal = Refusal(
        "table 'x' exceeds row cap: 500 > 200",
        'quoted child stderr (untrusted tail): "safe detail"',
    )
    complete_fn = _fake_complete("```python\nemit('x', ...)\n```")

    def always_refuses(code, vpath, run_dir, executor, *, limits=None):
        return refusal

    out = io.StringIO()
    rc = analyst.run_analyst(
        "q", vault_path, str(tmp_path / "run"),
        complete_fn=complete_fn, run_code_fn=always_refuses,
        executor=object(), json_output=True, out=out)
    assert rc != 0

    payload = json.loads(out.getvalue())
    assert payload["refused"] is True
    assert payload["reason"] == refusal.reason
    assert payload["diagnostic"] == refusal.diagnostic
    assert payload["remediation"]


# --------------------------------------------------------------------------- #
# 6. A missing --vault or --question fails cleanly
# --------------------------------------------------------------------------- #
def test_missing_vault_fails_cleanly(capsys):
    rc = analyst.main(["--question", "q"])
    assert isinstance(rc, int)
    assert rc != 0
    assert "--vault" in capsys.readouterr().err


def test_missing_question_fails_cleanly(vault_path, capsys):
    rc = analyst.main(["--vault", vault_path])
    assert isinstance(rc, int)
    assert rc != 0
    assert "--question" in capsys.readouterr().err


def test_missing_both_fails_cleanly(capsys):
    rc = analyst.main([])
    assert isinstance(rc, int)
    assert rc != 0


# --------------------------------------------------------------------------- #
# extract_code / refusal_guidance -- small, direct unit coverage
# --------------------------------------------------------------------------- #
def test_extract_code_pulls_fenced_block():
    reply = "Here is the code:\n```python\nx = 1\nemit('t', [], [], [])\n```\nDone."
    assert analyst.extract_code(reply) == "x = 1\nemit('t', [], [], [])"


def test_extract_code_falls_back_to_whole_reply_when_unfenced():
    reply = "x = 1\nemit('t', [], [], [])"
    assert analyst.extract_code(reply) == reply


def test_refusal_guidance_matches_zero_read():
    reason = "emitted 1 numeric tables from 0 vault tables and 0 reads"
    guidance = analyst.refusal_guidance(reason)
    assert guidance == analyst.REFUSAL_REMEDIATION["ZERO_READ"]


def test_refusal_guidance_falls_back_to_full_table_when_unmatched():
    guidance = analyst.refusal_guidance("envelope is not valid JSON: whoops")
    for code, remediation in analyst.REFUSAL_REMEDIATION.items():
        assert remediation in guidance


# --------------------------------------------------------------------------- #
# analyst_runner absence -- the module must import and run cleanly with a
# fake injected, whether or not analyst_runner.py exists on disk.
# --------------------------------------------------------------------------- #
def test_module_imports_without_analyst_runner(monkeypatch):
    """`_load_run_analyst_code` is only ever called when no fake is injected;
    this checkout may or may not have analyst_runner.py, and either way the
    module itself must already be imported cleanly (asserted implicitly by
    every test above having imported `analyst` at collection time)."""
    assert callable(analyst.run_analyst)
    assert callable(analyst.main)
