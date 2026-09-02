from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import trace_researcher_answer as trace


def _write_ledger(path, *results):
    path.write_text("\n".join(json.dumps({
        "sequence": index,
        "tool_name": "synthetic_tool",
        "arguments": {},
        "result": result,
        "result_elided": elided,
    }) for index, (result, elided) in enumerate(results, 1)) + "\n")


def test_main_reports_traceable_and_absent_figures_without_model(
        monkeypatch, tmp_path, capsys):
    ledger = tmp_path / "researcher.jsonl"
    _write_ledger(ledger, ({"jog_minutes": 42.5}, False),
                  ({"jog_minutes": 99}, True))
    calls = {}

    def fake_schemas(ctx):
        calls["ctx"] = ctx
        return [{"name": "synthetic"}]

    def fake_tool_loop(question, **kwargs):
        calls["question"] = question
        calls["kwargs"] = kwargs
        return "The jog total was 42.5 minutes; the missing figure is 99 minutes."

    monkeypatch.setattr(trace.llm, "tool_schemas", fake_schemas)
    monkeypatch.setattr(trace.llm, "tool_loop", fake_tool_loop)

    rc = trace.main([
        "--db", str(tmp_path / "vault.sqlite"),
        "--question", "How many jog minutes?",
        "--backend", "ollama",
        "--ledger-out", str(ledger),
    ])

    output = capsys.readouterr().out
    assert rc == 1
    assert "TRACEABILITY: 1 of 2 figures traceable" in output
    assert "sequence=1 tool='synthetic_tool' field='$.jog_minutes'" in output
    assert "UNTRACEABLE: token='99'" in output
    assert "missing figure is 99 minutes" in output
    assert calls["question"] == "How many jog minutes?"
    assert calls["kwargs"]["tools"] == [{"name": "synthetic"}]
    assert calls["kwargs"]["ledger_path"] == str(ledger)
    assert calls["ctx"].db_path == tmp_path / "vault.sqlite"
    assert trace.llm.BACKEND == "ollama"


def test_unrelated_result_field_collision_is_rejected():
    """A number in a named but unrelated field is not evidence for the figure."""
    records = [{
        "sequence": 1,
        "tool_name": "sleep_window",
        "arguments": {"days": 100},
        "result": {"days": 100, "window_start": "2026-05-01"},
        "result_elided": False,
    }]

    traceable, misses = trace.trace_answer(
        "The jog total was 100 minutes this month.", records)

    assert traceable == 0
    assert [miss["token"] for miss in misses] == ["100"]


def test_live_jog_ledger_traceability_uses_result_fields_and_arguments():
    ledger_path = Path(__file__).parent / "fixtures" / "jog_ledger_live_20260824.jsonl"
    records = trace.read_ledger(ledger_path)
    answer = (
        "Average weekly jog minutes:\n"
        "- Last 4 complete weeks (Jul 27-Aug 23): **50.1 min/week**\n"
        "- 4 weeks before (Jun 29-Jul 26): **50.2 min/week**\n"
        "That's essentially unchanged: **down 0.1 min/week**."
    )

    traceable, misses, matches = trace._trace_answer(answer, records)

    assert traceable == 5
    assert misses == []
    assert sum(1 for match in matches if match["category"] == "TRACEABLE-RESULT") == 3
    assert {(match["token"], match["path"]) for match in matches
            if match["category"] == "TRACEABLE-RESULT"
            and match["token"] in {"50.1", "50.2"}} == {
                ("50.1", "$.block_comparison.blocks.recent.mean"),
                ("50.2", "$.block_comparison.blocks.prior.mean"),
            }
    assert [(match["token"], match["path"]) for match in matches
            if match["category"] == "TRACEABLE-RESULT"
            and match["token"] == "0.1"] == [
                ("0.1", "$.block_comparison.change.total_delta_pct_note")
            ]
    argument_matches = [match for match in matches
                        if match["category"] == "TRACEABLE-ARG"]
    assert len(argument_matches) == 2
    assert {match["token"] for match in argument_matches} == {"4"}
    assert {match["argument"] for match in argument_matches} == {"weeks_per_block"}
    assert not any(match["token"] in {"27", "23", "29", "26"}
                   for match in matches)


def test_main_requires_explicit_db(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        trace.main([
            "--question", "How many jog minutes?",
            "--backend", "codex",
            "--ledger-out", str(tmp_path / "ledger.jsonl"),
        ])
    assert exc.value.code == 2
    assert "--db" in capsys.readouterr().err
