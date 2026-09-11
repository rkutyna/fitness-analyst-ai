"""Tests for fact-template verification tier reporting."""
from __future__ import annotations

from health_advisor import chat, fact_template, llm


def test_fact_template_tiers_distinguish_zero_and_resolved_placeholders(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = [{
        "sequence": 1,
        "tool_name": "analyst_query",
        "arguments": {},
        "result": {"tables": [{
            "name": "synthetic_table",
            "columns": ["item", "value"],
            "units": ["text", "count"],
            "rows": [["synthetic-a", 11], ["synthetic-b", 22]],
            "row_count": 2,
        }]},
    }]
    facts = fact_template.build_attachment_facts(ledger)
    first_key = fact_template.attachment_fact_key(
        "synthetic_table", "value", "synthetic-a")
    second_key = fact_template.attachment_fact_key(
        "synthetic_table", "value", "synthetic-b")
    responses = iter([
        "acknowledged",
        "Add {advice:3 repetitions} after the session.",
        "acknowledged",
        "The synthetic values are {" + first_key + "} and {" +
        second_key + "}.",
    ])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    zero = chat.answer_question(vault, "What should I do next?")
    resolved = chat.answer_question(vault, "What are the synthetic values?")

    assert zero["verification"]["tier_counts"] == {
        "path": 0, "metric": None,
    }
    assert zero["verification"]["tier1_path_bound"] == 0
    assert zero["verification"]["tier2_metric_recomputed"] is None
    assert resolved["verification"]["tier_counts"] == {
        "path": 2, "metric": None,
    }
    assert resolved["verification"]["tier1_path_bound"] == 2
    assert resolved["verification"]["tier2_metric_recomputed"] is None
