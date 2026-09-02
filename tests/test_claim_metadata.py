"""Coupling and metric-less claim regressions for published tool metadata."""

from health_advisor import deepdive_mcp as D
from health_advisor import deepdive_verify as DV
from health_advisor import mcp_server


def test_weekly_string_period_enters_ledger_vocabulary():
    result = {"metric": "sleep_asleep", "weeks": [{
        "week_start": "2026-08-10",
        "period": "2026-08-10:2026-08-16",
        "mean": 461.57,
    }]}
    assert D._claim_period_vocabulary(result) == [{
        "metric": "sleep_asleep",
        "claim_period": "2026-08-10:2026-08-16",
        "ledger_period": "2026-08-10:2026-08-16",
    }]


def test_workout_type_count_is_a_metricless_claim_leaf():
    ledger = [{
        "sequence": 1,
        "tool_name": "list_workouts",
        "arguments": {"start": "2026-08-01", "end": "2026-08-31"},
        "result": {"workout_counts": [{"type": "running", "count": 5}]},
        "result_elided": False,
    }]
    path = "$.result.workout_counts[0].count"
    honest = {"metric": None, "period": None, "field": "count", "value": 5,
              "source": {"sequence": 1, "path": path}}
    metric_claim = {**honest, "metric": "running"}

    scopes = DV._ledger_scopes(ledger[0])
    count_scope = next(entry for entry in scopes if entry["path"] == path)
    assert count_scope["metric"] is None
    assert DV._resolve_ledger_value(ledger, honest)["ok"] is True
    assert DV._resolve_ledger_value(ledger, metric_claim)["ok"] is False


def test_claim_metadata_rules_are_coupled_to_prompts_and_tool_docs():
    from health_advisor.chat import ASK_CLAIM_INSTRUCTIONS
    from health_advisor.llm import _RESEARCH_CLAIM_INSTRUCTIONS

    prompts = (ASK_CLAIM_INSTRUCTIONS, _RESEARCH_CLAIM_INSTRUCTIONS)
    for sentence in (DV.weekly_claim_metadata_sentence(),
                     DV.metric_ownership_sentence(),
                     DV.subjective_claim_metadata_sentence(),
                     DV.workout_count_claim_metadata_sentence()):
        assert all(sentence in prompt for prompt in prompts)

    weekly_doc = " ".join(mcp_server.get_weekly_series.__doc__.split())
    subjective_doc = " ".join(mcp_server.get_subjective.__doc__.split())
    workout_doc = " ".join(mcp_server.list_workouts.__doc__.split())
    assert " ".join(DV.weekly_claim_metadata_sentence().split()) in weekly_doc
    assert " ".join(DV.metric_ownership_sentence().split()) in weekly_doc
    assert " ".join(DV.subjective_claim_metadata_sentence().split()) in subjective_doc
    assert " ".join(DV.workout_count_claim_metadata_sentence().split()) in workout_doc
