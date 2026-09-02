"""#75: Python-owned presentation leaves are rendered and claimable."""
from __future__ import annotations

import json

import pytest

from health_advisor import deepdive_mcp
from health_advisor import deepdive_verify as DV
from health_advisor import fact_template
from health_advisor import mcp_server
from health_advisor import metrics as mx
from health_advisor.context import VaultContext
from tests.conftest import seed_metric


def test_formatter_handles_both_duration_families_and_all_hour_origins():
    assert mx.format_presentation("sleep_asleep", 439.69) == "7 h 20 m"
    assert mx.format_presentation("sleep_time_in_bed", 462.96) == "7 h 43 m"
    assert mx.format_presentation("sleep_bedtime", 11.694) == "11:41 PM"
    assert mx.format_presentation("sleep_midpoint", 15.434) == "3:26 AM"
    assert mx.format_presentation("sleep_wake_time", 7.173) == "7:10 AM"
    assert mx.format_presentation("sleep_midpoint_sd_28d", 1.019) == "± 1 h 01 m"
    assert mx.format_presentation("wear_hours", 20.0) == "20 h"
    assert mx.format_presentation("sleep_timing_interval_regularity", 89.0) is None


def test_mcp_presentation_leaf_reaches_the_ledger_and_is_claimable(conn, tools):
    seed_metric(conn, "sleep_asleep", "2026-08-20", [439.69])
    result = tools.get_daily_series("sleep_asleep", "2026-08-20", "2026-08-20")
    leaf = result["points"][0]["presentation"]
    assert leaf == {
        "metric": "sleep_asleep", "period": "2026-08-20",
        "field": "presentation", "value": "7 h 20 m",
    }

    ledger = [{"sequence": 1, "tool_name": "get_daily_series",
               "arguments": {}, "result": result,
               "result_elided": False}]
    claim = {**leaf, "source": {"sequence": 1,
                                 "path": "$.result.points[0].presentation.value"}}
    verdict = DV.verify_number(None, claim, payload=ledger)
    assert verdict["ok"] is True
    assert verdict["actual"] == "7 h 20 m"

    prose = "You got 7 h 20 m asleep."
    grounded = DV.verify_coach_claims(None, prose, [claim], payload=ledger)
    assert grounded["ok"] is True


def test_presentation_claim_does_not_accept_a_reformatted_string(conn, tools):
    seed_metric(conn, "sleep_asleep", "2026-08-20", [439.69])
    result = tools.get_daily_series("sleep_asleep", "2026-08-20", "2026-08-20")
    ledger = [{"sequence": 1, "tool_name": "get_daily_series",
               "arguments": {}, "result": result,
               "result_elided": False}]
    claim = {"metric": "sleep_asleep", "period": "2026-08-20",
             "field": "presentation", "value": "7h 20m",
             "source": {"sequence": 1,
                        "path": "$.result.points[0].presentation.value"}}
    assert DV.verify_number(None, claim, payload=ledger)["ok"] is False


def test_non_unit_fields_never_publish_metric_unit_leaves():
    node = {
        "value": 439.69,
        "delta_pct": 11.51,
        "total_delta_pct": 25.0,
        "trend_per_week": -3.2,
    }

    mcp_server._add_stat_presentations(node, "sleep_asleep", "90d")

    assert node["presentations"]["value"]["value"] == "7 h 20 m"
    assert "delta_pct" not in node["presentations"]
    assert "total_delta_pct" not in node["presentations"]
    assert "trend_per_week" not in node["presentations"]
    assert mx.format_presentation(
        "sleep_asleep", 11.51, field="delta_pct") is None


def test_signed_duration_fields_preserve_sign_and_magnitude():
    assert mx.format_presentation(
        "sleep_asleep", -45.0, field="delta_vs_baseline") == "-45 m"
    assert mx.format_presentation(
        "sleep_asleep", 45.0, field="delta_vs_baseline") == "+45 m"


@pytest.mark.live
def test_real_ledger_fact_set_has_only_field_correct_presentations(tmp_path):
    """Exercise the real ledger/fact-template arm against the read-only snapshot."""
    ctx = VaultContext.local("data/health.db", user_id="presentation-live")
    ledger = deepdive_mcp._CallLedger(str(tmp_path / "calls.jsonl"))
    tools = mcp_server.build_tools(ctx)
    wrapped = {
        name: deepdive_mcp._ledger_wrapper(name, tools[name], ledger)
        for name in ("summarize_metric", "get_daily_series",
                     "get_sleep_regularity")
    }

    asleep = wrapped["summarize_metric"]("sleep_asleep", period="90d")
    awake = wrapped["summarize_metric"]("sleep_awake", period="all")
    stand = wrapped["summarize_metric"]("apple_stand_time", period="30d")
    series = wrapped["get_daily_series"](
        "sleep_asleep", start="2026-07-01", end="2026-08-29")
    regularity = wrapped["get_sleep_regularity"](
        start="2026-07-01", end="2026-08-29")

    assert asleep["delta_pct"] is not None
    assert "delta_pct" not in asleep["presentations"]
    assert asleep["trend_per_week"] is not None
    assert "trend_per_week" not in asleep["presentations"]
    for result in (asleep, awake, stand):
        for field, leaf in result.get("presentations", {}).items():
            assert field not in mx._NON_UNIT_PRESERVING_FIELDS
            assert leaf["value"] == mx.format_presentation(
                result["metric"], result[field], field=field)
    assert awake["delta_vs_baseline"] < 0
    assert awake["presentations"]["delta_vs_baseline"]["value"] != "0 m"
    assert stand["delta_vs_baseline"] < 0
    assert stand["presentations"]["delta_vs_baseline"]["value"] != "0 m"

    for point in series["points"]:
        assert point["presentation"]["value"] == mx.format_presentation(
            "sleep_asleep", point["value"], field="value")
    midpoint = regularity["midpoint_variability"]
    if midpoint.get("presentation") is not None:
        assert midpoint["presentation"]["value"] == mx.format_presentation(
            "sleep_midpoint_sd_28d", midpoint["latest_sd_hours"],
            field="latest_sd_hours")

    with open(tmp_path / "calls.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    facts = fact_template.build_fact_set(records)
    for fact in facts.values():
        expected = mx.format_presentation(
            fact["metric"], fact["value"], field=fact["field"])
        assert fact["display"] == (expected if expected is not None
                                    else str(fact["value"]))
