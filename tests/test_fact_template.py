"""Tests for the closed fact-set ask narration arm."""
from __future__ import annotations

from pathlib import Path

import pytest

from health_advisor import fact_template, mcp_server, metrics
from health_advisor.context import VaultContext
from tests.conftest import PROD_DB_PATH


def _ledger(*, value=50.1, period="2026-08-17", display="50 m",
            unit="min"):
    return [{
        "sequence": 1,
        "tool_name": "synthetic_metric",
        "arguments": {},
        "result": {
            "metric": "jog_minutes",
            "unit": unit,
            "period": period,
            "mean": value,
            "presentation": {
                "metric": "jog_minutes", "period": period,
                "field": "presentation", "value": display,
            },
        },
    }]


def _weekly_ledger():
    return [{
        "sequence": 1,
        "tool_name": "get_weekly_series",
        "arguments": {},
        "result": {
            "metric": "jog_minutes",
            "weeks": [{"week_start": "2026-08-17", "mean": 50.1,
                       "unit": "min"}],
        },
    }]


@pytest.mark.parametrize("publisher", [
    fact_template.build_fact_set,
    fact_template.build_attachment_facts,
], ids=["build_fact_set", "build_attachment_facts"])
@pytest.mark.parametrize("case,expected", [
    ("single", True),
    ("identical", True),
    ("conflicting", False),
    ("same_value_different_presentation", True),
], ids=["single", "identical", "conflicting", "same-value-different-presentation"])
def test_fact_publishers_resolve_duplicate_values(publisher, case, expected):
    """Duplicate identity is safe when its owned values resolve uniquely.

    A wrong implementation that always publishes would fail the conflicting
    row; one that always drops duplicates would fail the identical and
    same-value-different-presentation rows. The single row catches accidental
    rejection of ordinary, non-duplicate facts. The row-4 case is deliberately
    explicit: passing by dropping the key would make a broken record-comparison
    guard look correct, so this asserts that the key is actually published.
    """
    if publisher is fact_template.build_fact_set:
        ledgers = {
            "single": _ledger(),
            "identical": _ledger() + _ledger(),
            "conflicting": _ledger() + _ledger(value=50.2),
            "same_value_different_presentation": (
                _ledger(display="50 m", unit="min")
                + _ledger(display="50.1 minutes", unit="minutes")),
        }
        key = fact_template.fact_key("jog_minutes", "2026-08-17", "mean")
    else:
        ledgers = {
            "single": _attachment_ledger([["same-day", 63]]),
            "identical": (_attachment_ledger([["same-day", 63]])
                          + _attachment_ledger([["same-day", 63]])),
            "conflicting": (_attachment_ledger([["same-day", 63]])
                            + _attachment_ledger([["same-day", 64]])),
            "same_value_different_presentation": (
                _attachment_ledger([["same-day", 63]],
                                    units=["date", "count/min"])
                + _attachment_ledger([["same-day", 63]],
                                     units=["date", "beats/min"])),
        }
        key = fact_template.attachment_fact_key(
            "resting_rate", "rate", "same-day")

    facts = publisher(ledgers[case])
    assert (key in facts) is expected


def test_weekly_mean_without_period_uses_week_start_as_fact_period():
    """The exact recovered key prevents a generic or invented period passing."""
    facts = fact_template.build_fact_set(_weekly_ledger())
    key = fact_template.fact_key("jog_minutes", "2026-08-17", "mean")

    assert key in facts
    assert facts[key]["value"] == 50.1


def test_fact_key_round_trips_the_natural_identity_tuple():
    period = {"start": "2026-08-10", "end": "2026-08-17"}
    key = fact_template.fact_key("jog_minutes", period, "mean")

    assert fact_template.parse_fact_key(key) == (
        "jog_minutes", period, "mean")


def test_fact_set_is_closed_and_pairs_the_published_presentation():
    facts = fact_template.build_fact_set(_ledger())
    key = fact_template.fact_key("jog_minutes", "2026-08-17", "mean")

    assert facts[key]["value"] == 50.1
    assert facts[key]["display"] == "50 m"
    plausible_missing = fact_template.fact_key(
        "jog_minutes", "2026-08-18", "mean")
    assert plausible_missing not in facts
    assert fact_template.interpolate_template(
        "Run: {" + plausible_missing + "}.", facts) is None


def test_week_and_day_periods_publish_human_period_labels():
    week_period = "2026-08-10:2026-08-16"
    day_period = "2026-08-07"
    block_period = {
        "start": "2026-07-27", "end": "2026-08-17",
        "period_starts": ["2026-07-27", "2026-08-03",
                           "2026-08-10", "2026-08-17"],
    }
    week_facts = fact_template.build_fact_set(_ledger(period=week_period))
    day_facts = fact_template.build_fact_set(_ledger(period=day_period))
    block_facts = fact_template.build_fact_set(_ledger(period=block_period))

    week_key = fact_template.fact_key("jog_minutes", week_period,
                                      "period_label")
    day_key = fact_template.fact_key("jog_minutes", day_period,
                                     "period_label")
    block_key = fact_template.fact_key("jog_minutes", block_period,
                                       "period_label")
    assert week_facts[week_key]["value"] == "the week of August 10"
    assert day_facts[day_key]["value"] == "Fri Aug 7"
    assert block_facts[block_key]["value"] == "the last 4 weeks"


def test_period_label_placeholder_interpolates_date_without_digit_refusal():
    period = "2026-08-10:2026-08-16"
    facts = fact_template.build_fact_set(_ledger(period=period))
    key = fact_template.fact_key("jog_minutes", period, "period_label")
    template = "Training for {" + key + "}."

    assert fact_template.scan_template(template, facts)[
        "digits_outside_placeholders"] is False
    assert fact_template.interpolate_template(template, facts) == (
        "Training for the week of August 10.")


def test_unknown_period_shape_publishes_no_label():
    period = {"kind": "unknown", "window": "not a date"}
    known_period = "2026-08-07"
    facts = {
        **fact_template.build_fact_set(_ledger(period=known_period)),
        **fact_template.build_fact_set(_ledger(period=period)),
    }

    known_key = fact_template.fact_key("jog_minutes", known_period,
                                      "period_label")
    assert facts[known_key]["value"] == "Fri Aug 7"
    assert not any(fact.get("field") == "period_label"
                   and fact.get("period") == period
                   for fact in facts.values())


def test_digit_inside_placeholder_is_allowed_but_digit_in_prose_is_refused():
    facts = fact_template.build_fact_set(_ledger())
    key = fact_template.fact_key("jog_minutes", "2026-08-17", "mean")

    assert fact_template.interpolate_template("Run: {" + key + "}.", facts) \
        == "Run: 50 m."
    assert fact_template.template_refused(
        "Run: {" + key + "} on day 2.", facts)


def test_bare_day_beside_weekday_is_prose_digit_and_is_refused():
    facts = fact_template.build_fact_set(_ledger())

    assert fact_template.template_refused("Tuesday, 26 minutes.", facts)


def test_digit_template_is_never_interpolated_or_returned():
    facts = fact_template.build_fact_set(_ledger())
    key = fact_template.fact_key("jog_minutes", "2026-08-17", "mean")

    assert fact_template.interpolate_template(
        "The result is 2: {" + key + "}.", facts) is None


def test_advice_slot_allows_prescriptive_digits_and_surfaces_its_span():
    facts = fact_template.build_fact_set(_ledger())
    advice = []

    rendered = fact_template.interpolate_template(
        "Try {advice:3 sets of 10 reps}.", facts,
        advice_quantities=advice)

    assert rendered == "Try 3 sets of 10 reps."
    assert advice == ["3 sets of 10 reps"]
    assert fact_template.scan_template(
        "Try {advice:3 sets of 10 reps}.", facts)["digits_outside_placeholders"] is False


def test_advice_slot_cannot_bypass_vault_metric_verification():
    facts = fact_template.build_fact_set(_ledger())

    scan = fact_template.scan_template(
        "Try {advice:3 sets after your jog_minutes reaches 50}.", facts)

    assert scan["ok"] is False
    assert scan["reason"] == "advice slot references the user's own data"
    assert fact_template.interpolate_template(
        "Try {advice:3 sets after your jog_minutes reaches 50}.", facts) is None

    metric_only = fact_template.scan_template(
        "Try {advice:3 sets after jog_minutes training}.", facts)
    assert metric_only["ok"] is False
    assert metric_only["reason"] == "advice slot references vault metric jog_minutes"


def test_advice_slot_mixed_with_fact_keeps_both_channels_separate():
    facts = fact_template.build_fact_set(_ledger())
    key = fact_template.fact_key("jog_minutes", "2026-08-17", "mean")
    advice = []

    rendered = fact_template.interpolate_template(
        "You logged {" + key + "}; add {advice:3 sets of 10 reps}.", facts,
        advice_quantities=advice)

    assert rendered == "You logged 50 m; add 3 sets of 10 reps."
    assert advice == ["3 sets of 10 reps"]
    assert fact_template.scan_template(
        "You logged {" + key + "}; add {advice:3 sets of 10 reps}.", facts
    )["placeholders"] == [key]


def _attachment_ledger(rows, *, units=None, table_name="resting_rate"):
    return [{
        "sequence": 7,
        "tool_name": "analyst_query",
        "arguments": {},
        "result": {"tables": [{
            "name": table_name,
            "columns": ["day", "rate"],
            "units": units or ["date", "count/min"],
            "rows": rows,
            "row_count": len(rows),
        }]},
    }]


def test_attachment_facts_publish_verbatim_cells_and_python_trends():
    ledger = _attachment_ledger([
        ["2026-08-01", 63], ["2026-08-02", 60],
    ])
    facts = fact_template.build_attachment_facts(ledger)
    cell_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-01")
    last_cell_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-02")
    first_key = fact_template.attachment_trend_key(
        "resting_rate", "rate", "first")
    last_key = fact_template.attachment_trend_key(
        "resting_rate", "rate", "last")
    delta_key = fact_template.attachment_trend_key(
        "resting_rate", "rate", "delta")
    direction_key = fact_template.attachment_trend_key(
        "resting_rate", "rate", "direction")

    assert facts[cell_key]["value"] == 63
    assert facts[cell_key]["display"] == "63"
    assert facts[cell_key]["unit"] == "count/min"
    assert facts[cell_key]["source"] == {
        "sequence": 7,
        "path": "$.result.tables[0].rows[0][1]",
    }
    assert facts[last_cell_key]["value"] == 60
    assert facts[last_cell_key]["display"] == "60"
    assert facts[last_cell_key]["unit"] == "count/min"
    assert facts[first_key]["value"] == 63
    assert facts[first_key]["display"] == "63"
    assert facts[first_key]["unit"] == "count/min"
    assert facts[last_key]["value"] == 60
    assert facts[last_key]["display"] == "60"
    assert facts[delta_key]["value"] == -3
    assert facts[delta_key]["display"] == "-3"
    assert facts[delta_key]["unit"] == "count/min"
    assert facts[direction_key]["value"] == facts[direction_key]["display"]
    assert facts[direction_key]["unit"] is None


def test_attachment_single_row_has_cells_but_no_trends():
    facts = fact_template.build_attachment_facts(
        _attachment_ledger([["2026-08-01", 63]]))

    assert fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-01") in facts
    assert not any("trend=" in key for key in facts)


def test_attachment_non_numeric_column_has_cells_but_no_trends():
    facts = fact_template.build_attachment_facts(_attachment_ledger([
        ["2026-08-01", 63], ["2026-08-02", "missing"],
    ]))

    assert len(facts) == 2
    assert not any("trend=" in key for key in facts)


def test_attachment_table_missing_required_key_is_skipped():
    ledger = _attachment_ledger([["2026-08-01", 63]])
    del ledger[0]["result"]["tables"][0]["units"]

    assert fact_template.build_attachment_facts(ledger) == {}


def test_attachment_duplicate_cell_keys_are_dropped():
    # Duplicate row keys are doubly ambiguous: the cell key names two cells,
    # and the key column is not strictly monotonic, so the table has no
    # chronology for a trend either. Everything ambiguous is omitted.
    ledger = _attachment_ledger([
        ["same-day", 63], ["same-day", 60],
    ])

    facts = fact_template.build_attachment_facts(ledger)

    assert fact_template.attachment_fact_key(
        "resting_rate", "rate", "same-day") not in facts
    assert not any("trend=" in key for key in facts)


def test_attachment_descending_table_trends_are_chronological():
    # Real envelope row keys are numeric periods; a newest-first table must
    # not flip the direction word. Trend facts read oldest-to-newest and
    # their source paths point at the rows actually used.
    descending = fact_template.build_attachment_facts(
        _attachment_ledger([[20260802, 60], [20260801, 63]]))
    ascending = fact_template.build_attachment_facts(
        _attachment_ledger([[20260801, 63], [20260802, 60]]))

    for stat in ("first", "last", "delta", "direction"):
        key = fact_template.attachment_trend_key("resting_rate", "rate", stat)
        assert descending[key]["value"] == ascending[key]["value"]
    first_key = fact_template.attachment_trend_key(
        "resting_rate", "rate", "first")
    assert descending[first_key]["value"] == 63
    assert descending[first_key]["source"]["paths"] == [
        "$.result.tables[0].rows[1][1]",
        "$.result.tables[0].rows[0][1]",
    ]


def test_attachment_non_monotonic_keys_publish_cells_but_no_trends():
    facts = fact_template.build_attachment_facts(
        _attachment_ledger([[2, 63], [1, 60], [3, 58]]))

    assert fact_template.attachment_fact_key("resting_rate", "rate", 2) in facts
    assert not any("trend=" in key for key in facts)


def test_attachment_keys_round_trip_and_do_not_collide_with_fact_keys():
    cell_key = fact_template.attachment_fact_key(
        "resting|rate", "count/min", "2026-08-01")
    trend_key = fact_template.attachment_trend_key(
        "resting|rate", "count/min", "delta")

    assert fact_template.parse_attachment_fact_key(cell_key) == (
        "resting|rate", "count/min", "2026-08-01")
    assert fact_template.parse_attachment_trend_key(trend_key) == (
        "resting|rate", "count/min", "delta")
    assert cell_key != fact_template.fact_key(
        "resting|rate", "count/min", "2026-08-01")
    assert trend_key != fact_template.fact_key(
        "resting|rate", "count/min", "delta")


@pytest.mark.live
def test_real_health_db_presentation_is_interpolated_byte_for_byte():
    """Use a read-only real snapshot result, not a hand-written display."""
    ctx = VaultContext.local(Path(PROD_DB_PATH), user_id="fact-template-live")
    tool = mcp_server.build_tools(ctx)["get_daily_series"]
    result = tool("jog_minutes", start="2026-08-17", end="2026-08-21")
    point = next(point for point in result["points"]
                 if point.get("value") is not None
                 and point.get("presentation"))
    ledger = [{"sequence": 1, "tool_name": "get_daily_series",
               "arguments": {}, "result": result}]
    facts = fact_template.build_fact_set(ledger)
    key = fact_template.fact_key("jog_minutes", point["date"], "value")

    assert facts[key]["display"] == metrics.format_presentation(
        "jog_minutes", point["value"])
    assert fact_template.interpolate_template("Value {" + key + "}.", facts) \
        == "Value " + metrics.format_presentation(
            "jog_minutes", point["value"]) + "."


def test_digit_free_advice_slot_unwraps_to_plain_prose_without_label():
    facts = fact_template.build_fact_set(_ledger())
    template = "{advice:Keep up your current routine; the trend is stable}."

    scan = fact_template.scan_template(template, facts)
    advice = []
    rendered = fact_template.interpolate_template(
        template, facts, advice_quantities=advice)

    # Digit-free content was always legal as prose: no refusal, but also no
    # exemption and no "coaching guidance" label for it to hide behind.
    assert scan["ok"] is True
    assert scan["advice_quantities"] == []
    assert advice == []
    assert rendered == "Keep up your current routine; the trend is stable."
