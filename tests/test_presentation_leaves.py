"""#75: Python-owned presentation leaves are rendered and claimable."""
from __future__ import annotations

import json
import re

import pytest

from health_advisor import deepdive_mcp
from health_advisor import deepdive_verify as DV
from health_advisor import fact_template
from health_advisor import mcp_server
from health_advisor import metrics as mx
from health_advisor import normalize
from health_advisor.context import VaultContext
from health_advisor.numeric_tokens import NUM_RE
from tests.conftest import seed_metric


def test_formatter_handles_both_duration_families_and_all_hour_origins():
    assert mx.format_presentation("sleep_asleep", 439.69) == "7 h 20 m"
    assert mx.format_presentation("sleep_time_in_bed", 462.96) == "7 h 43 m"
    assert mx.format_presentation("sleep_bedtime", 11.694) == "11:41 PM"
    assert mx.format_presentation("sleep_midpoint", 15.434) == "3:26 AM"
    assert mx.format_presentation("sleep_wake_time", 7.173) == "7:10 AM"
    assert mx.format_presentation("sleep_midpoint_sd_28d", 1.019) == "± 1 h 01 m"
    assert mx.format_presentation("wear_hours", 20.0) == "20 h"
    assert mx.format_presentation("sleep_timing_interval_regularity", 89.0) == "89"


def test_catalog_coverage_records_the_20_of_92_baseline_and_new_split():
    """The pre-change 20/72 gap is pinned beside the post-change 92/0 split."""
    legacy_duration_clock = frozenset({
        "apple_exercise_time", "apple_stand_time", "jog_minutes",
        "longest_block_min", "mindful_minutes", "sleep_asleep",
        "sleep_awake", "sleep_awake_longest", "sleep_bedtime", "sleep_core",
        "sleep_deep", "sleep_in_bed", "sleep_latency", "sleep_midpoint",
        "sleep_midpoint_sd_28d", "sleep_rem", "sleep_time_in_bed",
        "sleep_wake_time", "time_in_daylight", "wear_hours",
    })
    catalog = set(normalize.known_metrics())
    assert len(catalog) == 92
    assert (len(legacy_duration_clock),
            len(catalog - legacy_duration_clock)) == (20, 72)

    covered = {
        metric for metric in catalog
        if any(mx.format_presentation(metric, 12345.6789, field=field)
               is not None for field in mx._PRESENTATION_FIELDS)
    }
    assert (len(covered), len(catalog - covered)) == (92, 0)

    # Every catalog field is exercised. A raw-string mutation of any renderer
    # group exposes more than the two-decimal presentation contract.
    for metric in sorted(catalog):
        for field in mx._PRESENTATION_FIELDS:
            rendered = mx.format_presentation(
                metric, 12345.6789, field=field)
            if rendered is None:
                continue
            for token in re.findall(r"-?[\d,]+(?:\.\d+)?", rendered):
                decimals = token.split(".", 1)[1] if "." in token else ""
                assert len(decimals) <= mx.PRESENTATION_MAX_DECIMALS


def test_numeric_groups_round_without_a_unit_and_preserve_the_owned_number():
    """The display is the NUMERAL only; the unit selects precision, nothing more.

    The live defect read "Your longest run on record is 7.45228 miles." — the
    word "miles" is the model's, around the published figure. Appending the
    catalog unit here would render "7.45 mi miles", and for storage units it
    puts vault vocabulary into prose: "Your rate is 60 count/min."
    """
    samples = (
        ("distance_walking_running", 7.45228, "7.45", 0.0051),
        ("distance_cycling", 12.3456789, "12.35", 0.0051),
        ("resting_heart_rate", 52.37, "52", 0.5),
        ("step_count", 12345, "12,345", 0.5),
        ("body_mass", 164.25, "164.2", 0.051),
    )
    for metric, value, expected, tolerance in samples:
        rendered = mx.format_presentation(metric, value)
        assert rendered == expected
        # No unit suffix, ever, for these groups: the whole string must parse
        # as one number. Durations and clocks are the deliberate exception and
        # are pinned separately above.
        assert rendered.replace(",", "").lstrip("+-").replace(".", "", 1).isdigit()
        token = NUM_RE.findall(rendered)[0]
        assert float(token.replace(",", "")) == pytest.approx(
            value, abs=tolerance)


def test_unknown_metrics_and_non_unit_fields_stay_unrendered():
    assert mx.format_presentation("not_in_catalog", 12.345) is None
    for field in mx._NON_UNIT_PRESERVING_FIELDS:
        assert mx.format_presentation(
            "distance_walking_running", 12.345, field=field) is None


def test_numeric_tokens_bind_unit_suffixes_and_grouped_thousands():
    assert NUM_RE.findall("7.45 mi") == ["7.45"]
    assert NUM_RE.findall("52 bpm") == ["52"]
    assert NUM_RE.findall("12,345 steps") == ["12,345"]
    for rendered in ("7.45 mi", "52 count/min", "12,345 count"):
        assert len(NUM_RE.findall(rendered)) == 1


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
