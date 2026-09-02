from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from health_advisor import db as dbmod
from health_advisor import deepdive_verify as DV
from health_advisor import llm


FIXTURE = Path(__file__).parent / "fixtures/jog_ledger_live_20260824_claims.jsonl"
ANSWER_FIXTURE = Path(__file__).parent / "fixtures/jog_answer_live_20260824.json"

# This file describes the resolver's DEFAULT contract. Two of its refusals are
# deliberately lifted by Method C (`HA_ASK_VALUE_REBIND`), which rebinds a claim
# whose value/field/metric/period are right and whose pointer is wrong — see
# `tests/test_ledger_value_rebind.py`, which restates both with the flag on.
# Pin the flag down here so an exported environment variable cannot quietly
# change what these tests measure.


@pytest.fixture(autouse=True)
def value_rebind_off(monkeypatch):
    monkeypatch.delenv("HA_ASK_VALUE_REBIND", raising=False)


def _ledger():
    return [json.loads(FIXTURE.read_text())]


def _live_answer():
    return json.loads(ANSWER_FIXTURE.read_text())


def _blocks(ledger):
    return ledger[0]["result"]["block_comparison"]["blocks"]


def _claim(*, metric, period, field, value, path, sequence=1):
    return {"metric": metric, "period": period, "field": field,
            "value": value, "source": {"sequence": sequence, "path": path}}


def test_researcher_claim_channel_rejects_all_four_adversarial_sentences():
    ledger = _ledger()
    blocks = _blocks(ledger)
    week = blocks["prior"]["weeks"][0]
    cases = [
        ("Your resting heart rate is 201 bpm.", _claim(
            metric="resting_heart_rate", period=blocks["prior"]["period"],
            field="total", value=201,
            path="$.result.block_comparison.blocks.prior.total")),
        ("You slept 45.3 hours.", _claim(
            metric="sleep_hours", period=week["period"], field="sleep_hours",
            value=45.3,
            path="$.result.block_comparison.blocks.prior.weeks[0].value")),
        ("You ran 7 times this month.", _claim(
            metric="jog_minutes", period=week["period"], field="days_covered",
            value=7,
            path="$.result.block_comparison.blocks.prior.weeks[0].days_covered")),
        ("Jogging increased 100%.", _claim(
            metric="jog_minutes", period=blocks["recent"]["period"],
            field="percent_change", value=100,
            path="$.result.block_comparison.blocks.recent.not_a_field")),
    ]

    verdicts = [DV.verify_research_claims(text, [claim], ledger)
                for text, claim in cases]

    assert all(not verdict["ok"] for verdict in verdicts)
    assert [verdict["reason"] for verdict in verdicts] == [
        "claim metric does not match ledger field",
        "claim field does not match ledger path",
        "claim metric does not match ledger field",
        "ledger path not found",
    ]


def test_real_jog_answer_figures_verify_through_claim_channel():
    ledger = _ledger()
    blocks = _blocks(ledger)
    claims = [
        _claim(metric="jog_minutes", period=blocks["recent"]["period"],
               field="mean", value=50.1,
               path="$.result.block_comparison.blocks.recent.mean"),
        _claim(metric="jog_minutes", period=blocks["prior"]["period"],
               field="mean", value=50.2,
               path="$.result.block_comparison.blocks.prior.mean"),
        _claim(metric="jog_minutes", period=blocks["recent"]["period"],
               field="weeks_per_block", value=4,
               path="$.result.block_comparison.weeks_per_block"),
        _claim(metric="jog_minutes", period="comparison", field="mean_delta",
               value=-0.1,
               path="$.result.block_comparison.change.mean_delta"),
    ]

    verdict = DV.verify_research_claims(
        """Average weekly jog minutes:
        - Last 4 complete weeks (Jul 27-Aug 23): 50.1 min/week
        - 4 weeks before (Jun 29-Jul 26): 50.2 min/week
        That's essentially unchanged: down 0.1 min/week.""",
        claims, ledger)

    assert verdict["ok"]
    assert verdict["figures_verified"] == 5
    assert verdict["figures_total"] == 5
    assert sum(number["ok"] for number in verdict["verdict"]["numbers"]) == 4


def test_unmodified_live_answer_fixture_verifies_three_of_three():
    answer = _live_answer()
    verdict = DV.verify_research_claims(answer["text"], answer["claims"], _ledger())

    assert verdict["ok"] is True
    assert verdict["grounded"] is True
    assert verdict["unsupported"] == []
    assert [number["ok"] for number in verdict["verdict"]["numbers"]] == [
        True, True, True]


def test_plausible_but_wrong_periods_name_published_vocabulary():
    answer = _live_answer()
    ledger = _ledger()
    actual = ledger[0]["result"]["block_comparison"]["blocks"]["recent"]["period"]

    for period in (
        "2026-07-27:2026-08-22",
        {**actual, "end": "2026-08-23"},
    ):
        claim = deepcopy(answer["claims"][0])
        claim["period"] = period
        verdict = DV.verify_research_claims("Recent mean 50.1.", [claim], ledger)

        assert not verdict["ok"]
        assert "published period vocabulary" in verdict["reason"]
        assert "2026-07-27:2026-08-23" in verdict["verdict"]["numbers"][0][
            "published_periods"]


def test_research_grounding_allowlists_block_ordinals_but_not_measurements():
    claim = {"metric": "jog_minutes", "period": "30d", "field": "mean",
             "value": 50.1}

    ok, unsupported = DV._research_grounding(
        "The last 4 complete weeks averaged 50.1 minutes.", [claim])
    assert ok and unsupported == []

    ok, unsupported = DV._research_grounding(
        "Jogging increased by 4 minutes over the last 4 complete weeks.", [claim])
    assert not ok and unsupported == ["4"]


def test_researcher_claim_rejects_unrelated_days_100_collision():
    ledger = [{"sequence": 1, "tool_name": "synthetic", "arguments": {},
               "result": {"days": 100}, "result_elided": False}]
    claim = _claim(metric="jog_minutes", period="30d", field="jog_minutes",
                   value=100, path="$.result.days")

    verdict = DV.verify_research_claims("Jogging increased 100%.", [claim], ledger)

    assert not verdict["ok"]
    assert verdict["reason"] == "claim field does not match ledger path"


def test_researcher_claim_naming_absent_sequence_path_is_rejected():
    ledger = _ledger()
    blocks = _blocks(ledger)
    claim = _claim(metric="jog_minutes", period=blocks["recent"]["period"],
                   field="mean", value=50.1,
                   path="$.result.block_comparison.blocks.recent.missing")

    verdict = DV.verify_research_claims("Recent mean 50.1.", [claim], ledger)

    assert not verdict["ok"]
    assert verdict["reason"] == "ledger path not found"


def test_researcher_claim_citing_elided_result_is_rejected_with_distinct_reason():
    ledger = _ledger()
    ledger[0]["result"] = {"_elided": True, "bytes": ledger[0]["result_bytes"]}
    ledger[0]["result_elided"] = True
    claim = _claim(metric="jog_minutes", period="30d", field="mean",
                   value=50.1, path="$.result.block_comparison.blocks.recent.mean")

    verdict = DV.verify_research_claims("Recent mean 50.1.", [claim], ledger)

    assert not verdict["ok"]
    assert verdict["reason"] == "ledger result is elided"
    assert verdict["reason"] != "ledger path not found"


def test_researcher_and_coach_share_claim_shape_and_one_resolver():
    payload = {"metric": "jog_minutes", "period": "2026-08-17",
               "field": "jog_minutes", "value": 68.3}
    claim = {key: payload[key] for key in DV.SCOPED_CLAIM_FIELDS}

    assert DV.resolve_ledger_value is DV.resolve_payload_value
    assert DV.SCOPED_CLAIM_FIELDS == frozenset(
        {"metric", "period", "field", "value"})
    assert set(claim) == DV.SCOPED_CLAIM_FIELDS
    assert DV.resolve_payload_value([payload], claim)["ok"]


def test_list_workouts_claim_uses_identified_row_and_rejects_fabrication():
    """A workout leaf is scoped by the server-published row identity, not a
    metric-series label.  The value check must still reject fabrication."""
    workout_key = dbmod.workout_key(
        "running", "2026-08-15T12:00:00Z", "2026-08-15T13:08:48Z")
    ledger = [{
        "sequence": 1,
        "tool_name": "list_workouts",
        "arguments": {"start": "2026-08-01", "end": "2026-08-31"},
        "result": {"workouts": [{
            "workout_key": workout_key,
            "date": "2026-08-15",
            "type": "running",
            "duration_min": 68.8,
            "distance_mi": 3.85,
        }]},
        "result_elided": False,
    }]
    path = "$.result.workouts[0].duration_min"
    honest = {
        "metric": None, "period": None, "field": "duration_min", "value": 68.8,
        "source": {"sequence": 1, "path": path},
    }
    fabricated = {**honest, "value": 99.9}

    accepted = DV._resolve_ledger_value(ledger, honest)
    rejected = DV._resolve_ledger_value(ledger, fabricated)

    assert accepted["ok"] is True
    assert accepted["workout_key"] == workout_key
    assert rejected["ok"] is False
    assert rejected["reason"] == "claim value does not match ledger field"


def test_metricless_result_binds_by_exact_path_without_a_metric_label():
    ledger = [{
        "sequence": 1, "tool_name": "get_latest", "arguments": {},
        "result": {"value": 68.8}, "result_elided": False,
    }]
    claim = {
        "metric": None, "period": None, "field": "value", "value": 68.8,
        "source": {"sequence": 1, "path": "$.result.value"},
    }

    verdict = DV._resolve_ledger_value(ledger, claim)

    assert verdict["ok"] is True
    assert verdict["tier"] == "path"


def test_metricless_argument_path_is_still_refused():
    ledger = [{
        "sequence": 1, "tool_name": "get_latest",
        "arguments": {"value": 68.8},
        "result": {"value": 68.8}, "result_elided": False,
    }]
    claim = {
        "metric": None, "period": None, "field": "value", "value": 68.8,
        "source": {"sequence": 1, "path": "$.arguments.value"},
    }

    verdict = DV._resolve_ledger_value(ledger, claim)

    assert verdict["ok"] is False
    assert verdict["reason"].startswith(
        "claim cites a tool argument, not a result:")


def test_metricless_exact_path_requires_exact_value():
    ledger = [{
        "sequence": 1, "tool_name": "get_latest", "arguments": {},
        "result": {"period": "2026-08-21", "jog_minutes": 55.3},
        "result_elided": False,
    }]
    claim = {
        "metric": None, "period": "2026-08-21", "field": "jog_minutes",
        "value": 55.3001,
        "source": {"sequence": 1, "path": "$.result.jog_minutes"},
    }

    verdict = DV._resolve_ledger_value(ledger, claim)

    assert verdict["ok"] is False
    assert verdict["reason"] == "claim value does not match ledger field"


def test_metricless_ambiguous_path_is_refused(monkeypatch):
    ledger = [{
        "sequence": 1, "tool_name": "synthetic", "arguments": {},
        "result": {"value": 68.8}, "result_elided": False,
    }]
    entry = {"metric": None, "period": None, "field": "value",
             "value": 68.8, "workout_key": None,
             "path": "$.result.value", "kind": "result"}
    monkeypatch.setattr(DV, "_ledger_scopes", lambda record: [entry, entry])
    claim = {
        "metric": None, "period": None, "field": "value", "value": 68.8,
        "source": {"sequence": 1, "path": "$.result.value"},
    }

    verdict = DV._resolve_ledger_value(ledger, claim)

    assert verdict["ok"] is False
    assert verdict["reason"] == "ambiguous ledger path"


def test_metricless_false_accept_check_refuses_wrong_path_or_value():
    ledger = [{
        "sequence": 1, "tool_name": "synthetic",
        "arguments": {"answer": 42},
        "result": {"score": 42, "other": 7},
        "result_elided": False,
    }]

    cases = [
        {"metric": None, "period": None, "field": "score", "value": 43,
         "source": {"sequence": 1, "path": "$.result.score"}},
        {"metric": None, "period": None, "field": "score", "value": 42,
         "source": {"sequence": 1, "path": "$.result.other"}},
        {"metric": None, "period": None, "field": "score", "value": 99,
         "source": {"sequence": 1, "path": "$.result.score"}},
    ]

    verdicts = [DV.verify_number(None, claim, payload=ledger)
                for claim in cases]

    assert all(not verdict["ok"] for verdict in verdicts)


def test_research_answer_without_claims_or_tool_calls_is_not_grounded():
    verdict = DV.verify_research_claims("The result is 50.1.", [], [])

    assert not verdict["ok"]
    assert verdict["reason"] == "research answer has no tool-call ledger"


def test_research_loops_return_structured_claim_channel(monkeypatch, vault):
    raw = json.dumps({"text": "Mean 50.1.", "claims": [{
        "metric": "jog_minutes", "period": "30d", "field": "mean",
        "value": 50.1, "source": {"sequence": 1, "path": "$.result.mean"},
    }]})
    monkeypatch.setattr(llm, "BACKEND", "codex")
    monkeypatch.setattr(llm, "_codex_exec", lambda *args, **kwargs: raw)

    tool_result = llm.tool_loop("question", ctx=vault, tools=[])
    research_result = llm.research_loop(
        "question", ctx=vault, extra_tools={}, compact_state=lambda: "state")

    for result in (tool_result, research_result):
        assert result.text == "Mean 50.1."
        assert result["claims"][0]["source"]["sequence"] == 1
        assert result == "Mean 50.1."


def test_the_workout_claim_shape_that_verifies_is_the_shape_the_model_is_told_to_write():
    """The verifier accepts a metric-less workout claim. Nothing makes the model
    write one except the instructions, so the two must be pinned together.

    #96 landed a verifier that accepted `metric: None` for a workout row while
    both claim-instruction blocks still said "name the metric" — measured
    inert: three plausible metric names all refused, only omission verified.
    That is #93's finding recurring (an operation vocabulary living where the
    model cannot read it), so it is pinned rather than remembered.
    """
    from health_advisor.chat import ASK_CLAIM_INSTRUCTIONS
    from health_advisor.llm import _RESEARCH_CLAIM_INSTRUCTIONS

    for name, text in (("ASK_CLAIM_INSTRUCTIONS", ASK_CLAIM_INSTRUCTIONS),
                       ("_RESEARCH_CLAIM_INSTRUCTIONS", _RESEARCH_CLAIM_INSTRUCTIONS)):
        assert "list_workouts" in text, (
            f"{name} never names the tool whose rows need the exception")
        assert "workout_key" in text, (
            f"{name} does not mention workout_key, the identity the verifier keys on")
        assert "omit" in text.lower(), (
            f"{name} does not tell the model to omit metric for a workout row, "
            "so the verifier's workout path is unreachable in production")
