"""``summarize_metric``'s ``trend_per_week`` and ``delta_vs_baseline`` are citable.

consumer #578 (the sweep behind consumer #576): both leaves sat beside the
published ``mean``/``recent_avg`` figures but belonged to no metric, so the fact
set had no slot for "is it trending?" or "how does the last week compare?".

Two things had to be true before they could be published honestly:

* ``trend_per_week`` is a RATE. Rendered as the metric's own unit it would read
  as a level ("283" steps). It is rendered signed with its unit word and "per
  week" in the figure itself ("+283.2 steps per week").
* ``delta_vs_baseline`` of a clock-time metric is a duration. The formatter
  rendered a bedtime that moved 18 minutes earlier as the clock time
  "-11:42 AM"; it is "-18 min".
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from health_advisor import chat, fact_template, llm
from health_advisor import metrics as mx
from tests.conftest import seed_metric

START = "2026-07-01"
DAYS = 30
LAST = "2026-07-30"
ASK = "What is the trend in my step count?"


def _ledger(tools, *calls):
    out = []
    for sequence, (name, arguments) in enumerate(calls, start=1):
        out.append({"sequence": sequence, "tool_name": name,
                    "arguments": arguments, "result_elided": False,
                    "result": json.loads(json.dumps(
                        getattr(tools, name)(**arguments)))})
    return out


def _mine(facts, metric, field):
    return [f for f in facts.values()
            if f["metric"] == metric and f["field"] == field]


@pytest.fixture
def steps(conn):
    # +100 steps a day, exactly: a trend of +700 a week, and a recent week
    # whose mean is the baseline's mean plus a computable gap.
    values = [8000.0 + 100.0 * i for i in range(DAYS)]
    seed_metric(conn, "step_count", START, values)
    return values


def test_trend_and_change_are_published_with_figures_computed_from_the_seed(
        steps, tools):
    ledger = _ledger(tools, ("summarize_metric", {"metric": "step_count"}))
    facts = fact_template.build_fact_set(ledger)
    period = ledger[0]["result"]["period"]

    (trend,) = _mine(facts, "step_count", "trend_per_week")
    (delta,) = _mine(facts, "step_count", "delta_vs_baseline")
    recent, baseline = steps[-7:], steps[:-7]
    assert trend["period"] == delta["period"] == period
    assert trend["value"] == pytest.approx(
        np.polyfit(range(DAYS), steps, 1)[0] * 7, abs=0.01) == pytest.approx(700.0)
    assert delta["value"] == pytest.approx(
        np.mean(recent) - np.mean(baseline), abs=0.01)
    # A rate says so; a change is signed.
    assert trend["display"] == "+700 steps per week"
    assert delta["display"].startswith("+") and "per week" not in delta["display"]
    leaves = ledger[0]["result"]["presentations"]
    assert trend["display"] == leaves["trend_per_week"]["value"]
    assert delta["display"] == leaves["delta_vs_baseline"]["value"]
    assert fact_template.publish_completeness(ledger, facts) == set()


def test_the_rate_is_never_a_bare_level_for_any_unit_kind(conn, tools):
    seed_metric(conn, "resting_heart_rate", START,
                [60.0 - 0.1 * i for i in range(DAYS)])
    seed_metric(conn, "sleep_asleep", START, [420.0 + 2 * i for i in range(DAYS)])
    seed_metric(conn, "sleep_bedtime", START, [11.0 + 0.01 * i for i in range(DAYS)])
    shown = {}
    for metric in ("resting_heart_rate", "sleep_asleep", "sleep_bedtime"):
        ledger = _ledger(tools, ("summarize_metric", {"metric": metric}))
        facts = fact_template.build_fact_set(ledger)
        (trend,) = _mine(facts, metric, "trend_per_week")
        (delta,) = _mine(facts, metric, "delta_vs_baseline")
        shown[metric] = (trend["display"], delta["display"])
        assert trend["display"].endswith(" per week")
    assert shown["resting_heart_rate"][0] == "-0.7 bpm per week"
    assert shown["sleep_asleep"][0] == "+14 min per week"
    # A bedtime that drifts later by 0.07 h a week is minutes, not a clock time.
    assert shown["sleep_bedtime"][0] == "+4 min per week"
    for text in shown["sleep_bedtime"]:
        assert "AM" not in text and "PM" not in text


def test_a_change_of_a_clock_time_is_a_duration():
    assert mx.format_presentation(
        "sleep_bedtime", -0.3, field="delta_vs_baseline") == "-18 min"
    assert mx.format_presentation(
        "sleep_wake_time", 0.5, field="delta_vs_baseline") == "+30 min"
    assert mx.format_presentation(
        "sleep_midpoint_sd_28d", 0.3, field="delta_vs_baseline") == "+18 min"
    # The levels keep their own notation.
    assert mx.format_presentation("sleep_bedtime", 0.5) == "12:30 PM"
    assert mx.format_presentation("sleep_midpoint_sd_28d", 1.019) == "± 1 h 01 min"


def test_a_rate_that_rounds_to_nothing_has_no_sign():
    assert mx.format_presentation(
        "resting_heart_rate", -0.004, field="trend_per_week") == "0 bpm per week"
    assert mx.format_presentation(
        "step_count", 0.0, field="trend_per_week") == "0 steps per week"


def test_publishing_them_withholds_nothing_published_today(steps, tools):
    others = [("get_latest", {"metric": "step_count"}),
              ("get_daily_series", {"metric": "step_count"}),
              ("get_weekly_series", {"metric": "step_count", "start": START,
                                     "end": LAST}),
              ("compare_periods", {"metric": "step_count", "period_a": "7d",
                                   "period_b": f"{START}:2026-07-14"}),
              ("summarize_metric", {"metric": "step_count", "period": "7d"})]
    summary = ("summarize_metric", {"metric": "step_count"})
    without = _ledger(tools, *others)
    combined = _ledger(tools, summary, *others)

    before = fact_template.build_fact_set(without)
    after = fact_template.build_fact_set(combined)

    for key, fact in before.items():
        assert key in after, f"{key} stopped being published"
        assert after[key]["display"] == fact["display"]
    # The two 7d/30d summaries each add their own pair, and nothing else new
    # is a value (period labels aside).
    added = [f for k, f in after.items() if k not in before
             and f["field"] != "period_label"]
    assert {f["field"] for f in added} <= {
        "trend_per_week", "delta_vs_baseline", "mean", "median", "min", "max",
        "std", "recent_avg", "baseline_avg", "delta_pct"}
    assert fact_template.publish_completeness(combined, after) == set()


def _run(monkeypatch, vault, ledger, responses):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    replies = iter(responses)
    prompts = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **k: prompts.append(prompt) or next(replies))
    return chat.answer_question(vault, ASK, as_of=LAST, capture=[]), prompts


def test_template_citing_the_trend_slot_renders_and_verifies(
        monkeypatch, vault, steps, tools):
    ledger = _ledger(tools, ("summarize_metric", {"metric": "step_count"}))
    facts = fact_template.build_fact_set(ledger)
    (trend,) = _mine(facts, "step_count", "trend_per_week")
    label = fact_template.fact_key("step_count", trend["period"], "period_label")
    template = "Over {%s} your step count moved {%s}." % (label, trend["key"])

    out, _ = _run(monkeypatch, vault, ledger, ["acknowledged", template])

    assert out["mode"] == "narration"
    assert out["verification"]["cause"] == "ok"
    assert "+700 steps per week" in out["text"]


def test_denial_over_the_same_ledger_is_still_refused_and_offered_the_slot(
        monkeypatch, vault, steps, tools):
    ledger = _ledger(tools, ("summarize_metric", {"metric": "step_count"}))
    denial = ("That is not among the figures this lookup returned. "
              "{status:gathered_data_uncited}")
    out, prompts = _run(monkeypatch, vault, ledger,
                        ["acknowledged", denial, denial])

    assert out["mode"] == "fallback"
    assert out["verification"]["cause"] == "denied_available_figure"
    period = ledger[0]["result"]["period"]
    assert fact_template.fact_key(
        "step_count", period, "trend_per_week") in prompts[2]
