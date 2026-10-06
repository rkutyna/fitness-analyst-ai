"""A ``get_briefing`` readiness factor's ``current`` and ``baseline`` are citable.

consumer #578 (the sweep behind consumer #576): ``readiness.factors[].current``
and ``.baseline`` were owned by their metric but carried no period, so no key
could be built and the fact set was empty while the available-figure gate could
see the figure. Each now carries the period that is TRUE for it, because the two
are different statistics over different windows:

* ``current`` is the MEAN of the (up to 3) days ending at the factor's latest
  day;
* ``baseline`` is the MEDIAN of the days before those, back at most 28 days.

Publishing either under the single ``date`` would let the model say "on <date>"
about an average. ``readiness.score`` and ``readiness.components`` stay
unpublished on purpose (see ``fact_template.build_briefing_facts``).
"""
from __future__ import annotations

import json
import statistics
from datetime import date, timedelta

import pytest

from health_advisor import chat, fact_template, llm
from tests.conftest import seed_metric

START = "2026-07-01"
DAYS = 40
LAST = (date.fromisoformat(START) + timedelta(days=DAYS - 1)).isoformat()
METRICS = ("heart_rate_variability", "resting_heart_rate", "sleep_asleep")
ASK = "What is my resting heart rate?"


def _series(metric: str) -> list[float]:
    """An invented series whose last three days differ from the rest."""
    base = {"heart_rate_variability": 50.0, "resting_heart_rate": 58.0,
            "sleep_asleep": 420.0}[metric]
    values = [base + (i % 5) for i in range(DAYS)]
    values[-3:] = [base + 10, base + 12, base + 14]
    return values


@pytest.fixture
def seeded(conn):
    for metric in METRICS:
        seed_metric(conn, metric, START, _series(metric))
    return conn


def _ledger(tools, *calls) -> list[dict]:
    """One ledger record per (tool, arguments), as the ask path records them."""
    out = []
    for sequence, (name, arguments) in enumerate(calls, start=1):
        result = getattr(tools, name)(**arguments)
        out.append({"sequence": sequence, "tool_name": name,
                    "arguments": arguments, "result_elided": False,
                    "result": json.loads(json.dumps(result))})
    return out


def _factors(ledger) -> dict[str, dict]:
    return {f["component"]: f
            for f in ledger[0]["result"]["readiness"]["factors"]}


def _mine(facts, metric, field):
    return [f for f in facts.values()
            if f["metric"] == metric and f["field"] == field]


def test_current_and_baseline_are_published_under_their_own_true_windows(
        seeded, tools):
    ledger = _ledger(tools, ("get_briefing", {}))
    factors = _factors(ledger)
    facts = fact_template.build_fact_set(ledger)

    for component, metric in (("hrv", "heart_rate_variability"),
                              ("rhr", "resting_heart_rate")):
        values = dict(zip(
            [(date.fromisoformat(START) + timedelta(days=i)).isoformat()
             for i in range(DAYS)], _series(metric)))
        last3 = [values[(date.fromisoformat(LAST) - timedelta(days=i)).isoformat()]
                 for i in (2, 1, 0)]
        before = [v for d, v in values.items()
                  if (date.fromisoformat(LAST) - date.fromisoformat(d)).days >= 3
                  and (date.fromisoformat(LAST) - date.fromisoformat(d)).days <= 30]

        (current,) = _mine(facts, metric, "current")
        (baseline,) = _mine(facts, metric, "baseline")
        # Computed here from the seed, not read back from the tool.
        assert current["value"] == pytest.approx(statistics.mean(last3), abs=0.01)
        assert baseline["value"] == pytest.approx(statistics.median(before), abs=0.01)
        # The mean covers three days, never the single "date" of the factor.
        start_c, end_c = current["period"].split(":")
        assert end_c == factors[component]["date"] == LAST
        assert (date.fromisoformat(end_c) - date.fromisoformat(start_c)).days == 2
        # The median ends BEFORE the days that make up the current mean.
        start_b, end_b = baseline["period"].split(":")
        assert end_b < start_c
        assert baseline["period"] != current["period"]
        # Display is the leaf the engine's one formatter wrote beside the value.
        leaf = factors[component]["presentations"]
        assert current["display"] == leaf["current"]["value"]
        assert baseline["display"] == leaf["baseline"]["value"]
        # Each window can be named for the reader.
        for fact in (current, baseline):
            label = fact_template.fact_key(metric, fact["period"], "period_label")
            assert facts[label]["display"].startswith("from ")

    (sleep,) = _mine(facts, "sleep_asleep", "current")
    assert _mine(facts, "sleep_asleep", "baseline") == []     # no baseline exists
    assert sleep["display"] == factors["sleep"]["presentations"]["current"]["value"]
    assert "h" in sleep["display"] and "min" in sleep["display"]
    assert fact_template.publish_completeness(ledger, facts) == set()


def test_the_ledger_flattening_gives_each_leaf_its_own_window(seeded, tools):
    """The verifier and the ledger index see the same windows the facts do."""
    from health_advisor import deepdive_verify as dv

    ledger = _ledger(tools, ("get_briefing", {}))
    factors = _factors(ledger)["rhr"]
    entries = {e["path"]: e for e in dv._ledger_scopes(ledger[0])}
    base = "$.result.readiness.factors[%d]." % (
        ledger[0]["result"]["readiness"]["factors"].index(factors))
    assert entries[base + "current"]["period"] == factors["field_periods"]["current"]
    assert entries[base + "baseline"]["period"] == factors["field_periods"]["baseline"]
    assert entries[base + "current"]["metric"] == "resting_heart_rate"
    # An untouched sibling keeps the node's own (absent) period.
    assert entries[base + "pct"]["period"] is None


def test_a_gap_before_the_latest_day_moves_the_windows_with_the_data(conn, tools):
    # The newest row is two days before the others stopped: the window follows
    # the data, not the calendar, and the baseline ends before it starts.
    for metric in METRICS:
        values = _series(metric)[:-2]
        seed_metric(conn, metric, START, values)
    ledger = _ledger(tools, ("get_briefing", {}))
    facts = fact_template.build_fact_set(ledger)
    last = (date.fromisoformat(LAST) - timedelta(days=2)).isoformat()
    (current,) = _mine(facts, "resting_heart_rate", "current")
    assert current["period"].endswith(":" + last)


def test_the_withheld_readiness_figures_stay_withheld(seeded, tools):
    ledger = _ledger(tools, ("get_briefing", {}))
    facts = fact_template.build_fact_set(ledger)
    paths = {f["source"].get("path", "") for f in facts.values()}
    assert not any(".components." in p or p.endswith(".readiness.score")
                   for p in paths)
    assert not any(".readiness.factors[" in p and p.endswith((".pct", ".target"))
                   for p in paths)


def test_publishing_them_withholds_nothing_published_today(seeded, tools):
    """get_briefing beside get_latest, summarize_metric and get_daily_series."""
    others = []
    for metric in METRICS:
        others += [("get_latest", {"metric": metric}),
                   ("summarize_metric", {"metric": metric}),
                   ("summarize_metric", {"metric": metric, "period": "7d"}),
                   ("get_daily_series", {"metric": metric, "start": "2026-08-01",
                                         "end": LAST})]
    without = _ledger(tools, *others)
    combined = _ledger(tools, ("get_briefing", {}), *others)

    before = fact_template.build_fact_set(without)
    after = fact_template.build_fact_set(combined)

    assert before, "the comparison ledger published nothing"
    for key, fact in before.items():
        assert key in after, f"{key} stopped being published"
        assert after[key]["value"] == fact["value"]
        assert after[key]["display"] == fact["display"]
    added = {k: f for k, f in after.items() if k not in before}
    assert {f["field"] for f in added.values()} <= {
        "current", "baseline", "period_label"}
    assert len([f for f in added.values() if f["field"] != "period_label"]) == 5
    assert fact_template.publish_completeness(combined, after) == set()


def test_no_other_tool_publishes_the_new_field_names(seeded, tools):
    """``current`` and ``baseline`` are only ever the briefing's own fields."""
    combined = _ledger(tools, ("get_briefing", {}),
                       ("get_latest", {"metric": "resting_heart_rate"}),
                       ("summarize_metric", {"metric": "resting_heart_rate"}),
                       ("get_daily_series", {"metric": "resting_heart_rate"}),
                       ("get_weekly_series", {"metric": "resting_heart_rate",
                                              "start": "2026-07-06", "end": LAST}))
    facts = fact_template.build_fact_set(combined)
    mine = [f for f in facts.values() if f["field"] in ("current", "baseline")]
    assert mine and {f["source"]["sequence"] for f in mine} == {1}


# --- narration, end to end, with the model stubbed ---------------------------


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


def test_template_citing_the_current_slot_renders_and_verifies(
        monkeypatch, vault, seeded, tools):
    ledger = _ledger(tools, ("get_briefing", {}))
    facts = fact_template.build_fact_set(ledger)
    (current,) = _mine(facts, "resting_heart_rate", "current")
    label = fact_template.fact_key(
        "resting_heart_rate", current["period"], "period_label")
    template = "Your resting heart rate averaged {%s} %s." % (
        current["key"], "{" + label + "}")

    out, _ = _run(monkeypatch, vault, ledger, ["acknowledged", template])

    assert out["mode"] == "narration"
    assert out["verification"]["cause"] == "ok"
    assert current["display"] in out["text"]
    assert facts[label]["display"] in out["text"]


def test_denial_over_the_same_ledger_is_still_refused_and_offered_the_slot(
        monkeypatch, vault, seeded, tools):
    ledger = _ledger(tools, ("get_briefing", {}))
    denial = ("That is not among the figures this lookup returned. "
              "{status:gathered_data_uncited}")
    out, prompts = _run(monkeypatch, vault, ledger,
                        ["acknowledged", denial, denial])

    assert out["mode"] == "fallback"
    assert out["verification"]["cause"] == "denied_available_figure"
    facts = fact_template.build_fact_set(ledger)
    (current,) = _mine(facts, "resting_heart_rate", "current")
    assert current["key"] in prompts[2]
