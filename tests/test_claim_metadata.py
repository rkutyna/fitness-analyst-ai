"""Coupling and metric-less claim regressions for published tool metadata."""

from health_advisor import deepdive_mcp as D
from health_advisor import deepdive_verify as DV
from health_advisor import claim_contract as CC
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


def test_metricless_predicate_covers_missing_none_and_blank_values():
    assert CC.is_metricless_metric({}.get("metric"))
    assert CC.is_metricless_metric(None)
    assert CC.is_metricless_metric("")
    assert CC.is_metricless_metric("  ")
    assert not CC.is_metricless_metric("jog_minutes")


def test_blank_metric_binds_metricless_result_leaf_by_exact_path():
    ledger = [{
        "sequence": 1,
        "tool_name": "list_workouts",
        "arguments": {},
        "result": {"workout_counts": [{"type": "running", "count": 5}]},
        "result_elided": False,
    }]
    claim = {
        "metric": "",
        "period": None,
        "field": "count",
        "value": 5,
        "source": {"sequence": 1,
                   "path": "$.result.workout_counts[0].count"},
    }

    verdict = DV._resolve_ledger_value(ledger, claim)

    assert verdict["ok"] is True
    assert verdict["tier"] == "path"


def test_metricless_encodings_agree_on_a_metric_bearing_result_leaf():
    """Absent, null and blank-after-strip are ONE state, everywhere.

    This replaces test_blank_metric_does_not_bind_metric_bearing_result_leaf,
    which asserted that `metric: ""` is refused where an omitted key binds.
    That asymmetry IS the defect #43 reports, one table over: measured against
    unpatched main on this very fixture, an absent key and `None` both bind and
    `""` alone is refused. Pinning it kept absence-versus-emptiness alive in the
    place the fix was least likely to look.

    A claim asserting a WRONG metric is still refused — that is the property
    the old test was reaching for, and it is asserted explicitly below.
    """
    ledger = [{
        "sequence": 1,
        "tool_name": "synthetic",
        "arguments": {},
        "result": {"metric": "jog_minutes", "period": "2026-08-21",
                   "value": 12, "field_metrics": {"value": "jog_minutes"}},
        "result_elided": False,
    }]

    def verdict(**kw):
        claim = {"period": "2026-08-21", "field": "value", "value": 12,
                 "source": {"sequence": 1, "path": "$.result.value"}}
        claim.update(kw)
        return DV._resolve_ledger_value(ledger, claim)

    # Every encoding of "no metric" agrees with an omitted key.
    omitted = verdict()
    assert omitted["ok"] is True
    for blank in (None, "", "   ", "\t"):
        assert verdict(metric=blank)["ok"] is omitted["ok"], (
            f"metric={blank!r} disagrees with an omitted key")

    # Blankness buys no leniency: a wrong metric is still refused.
    wrong = verdict(metric="steps")
    assert wrong["ok"] is False
    assert wrong["reason"] == "claim metric does not match ledger field"

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
