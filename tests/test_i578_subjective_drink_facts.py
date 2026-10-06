"""``get_subjective``'s drink counts are citable facts.

consumer #578 (the sweep behind consumer #576): a check-in row's
``alcohol_drinks`` and ``caffeine_drinks`` were not in the row's
``field_metrics``, so they belonged to no metric. The fact set had no slot for
them while the available-figure gate matched them by field name and refused a
denial. They are the same kind of daily value as the ratings beside them, so
they are owned the same way, and their display comes from the one formatter that
also renders ``get_daily_series`` -- a half drink reads "2.5" in both.
"""
from __future__ import annotations

import json

import pytest

from health_advisor import chat, fact_template, llm

ASK = "How many alcohol drinks did I log?"
DAYS = {"2026-08-01": (0, 3), "2026-08-02": (1, 2), "2026-08-03": (2.5, 1)}


@pytest.fixture
def logged(tools):
    for day, (alcohol, caffeine) in DAYS.items():
        assert tools.log_subjective(
            day, alcohol_drinks=alcohol, caffeine_drinks=caffeine, stress=2)["ok"]
    return tools


def _ledger(tools, *calls):
    out = []
    for sequence, (name, arguments) in enumerate(calls, start=1):
        out.append({"sequence": sequence, "tool_name": name,
                    "arguments": arguments, "result_elided": False,
                    "result": json.loads(json.dumps(
                        getattr(tools, name)(**arguments)))})
    return out


SUBJECTIVE = ("get_subjective",
              {"start_date": "2026-08-01", "end_date": "2026-08-03"})


def _own(facts, metric, day):
    return [f for f in facts.values()
            if f["metric"] == metric and f["period"] == day and f["field"] == metric]


def test_each_days_drink_counts_are_published_with_python_displays(logged):
    ledger = _ledger(logged, SUBJECTIVE)
    facts = fact_template.build_fact_set(ledger)

    for day, (alcohol, caffeine) in DAYS.items():
        (a,) = _own(facts, "alcohol_drinks", day)
        (c,) = _own(facts, "caffeine_drinks", day)
        assert (a["value"], c["value"]) == (alcohol, caffeine)
        assert a["unit"] == c["unit"] == "drinks"
    # No invented decimals on whole counts, and none lost on a half drink.
    assert _own(facts, "alcohol_drinks", "2026-08-01")[0]["display"] == "0"
    assert _own(facts, "caffeine_drinks", "2026-08-01")[0]["display"] == "3"
    assert _own(facts, "alcohol_drinks", "2026-08-03")[0]["display"] == "2.5"
    # The display is the leaf the engine's formatter wrote beside the row.
    row = ledger[0]["result"]["days"][2]
    assert row["presentations"]["alcohol_drinks"]["value"] == "2.5"
    assert fact_template.publish_completeness(ledger, facts) == set()
    # The ratings already published stay as they were.
    assert {f["field"] for f in facts.values()
            if f["metric"] == "subjective_stress"} >= {"stress"}


def test_a_null_count_is_not_published(tools):
    tools.log_subjective("2026-08-01", stress=3)
    ledger = _ledger(tools, ("get_subjective", {"start_date": "2026-08-01",
                                                "end_date": "2026-08-01"}))
    facts = fact_template.build_fact_set(ledger)
    assert not [f for f in facts.values() if "drinks" in f["metric"]]


def test_the_same_day_agrees_with_get_daily_series_and_nothing_is_withheld(logged):
    ledger = _ledger(logged, SUBJECTIVE,
                     ("get_daily_series", {"metric": "alcohol_drinks"}),
                     ("get_daily_series", {"metric": "caffeine_drinks"}),
                     ("summarize_metric", {"metric": "alcohol_drinks"}))
    facts = fact_template.build_fact_set(ledger)
    series_only = fact_template.build_fact_set(ledger[1:])

    for day in DAYS:
        for metric in ("alcohol_drinks", "caffeine_drinks"):
            (mine,) = _own(facts, metric, day)
            (theirs,) = [f for f in facts.values()
                         if f["metric"] == metric and f["period"] == day
                         and f["field"] == "value"]
            assert mine["value"] == theirs["value"]
            assert mine["display"] == theirs["display"]
    # Everything the series ledger published is still published, unchanged.
    for key, fact in series_only.items():
        assert facts[key]["display"] == fact["display"]
    assert fact_template.publish_completeness(ledger, facts) == set()


def _run(monkeypatch, vault, ledger, responses):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    replies = iter(responses)
    prompts = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **k: prompts.append(prompt) or next(replies))
    return chat.answer_question(vault, ASK, as_of="2026-08-03", capture=[]), prompts


def test_template_citing_the_drink_slot_renders_and_verifies(
        monkeypatch, vault, logged):
    ledger = _ledger(logged, SUBJECTIVE)
    key = fact_template.fact_key("alcohol_drinks", "2026-08-03", "alcohol_drinks")
    label = fact_template.fact_key("alcohol_drinks", "2026-08-03", "period_label")
    # The engine appends the unit word to a bare count, so the template
    # leaves it out (a template that adds its own noun gets both).
    template = "On {%s} you had {%s} of alcohol." % (label, key)

    out, _ = _run(monkeypatch, vault, ledger, ["acknowledged", template])

    assert out["mode"] == "narration"
    assert out["verification"]["cause"] == "ok"
    assert out["text"] == "On Mon Aug 3 you had 2.5 drinks of alcohol."


def test_denial_over_the_same_ledger_is_still_refused_and_offered_the_slot(
        monkeypatch, vault, logged):
    ledger = _ledger(logged, SUBJECTIVE)
    denial = ("That is not among the figures this lookup returned. "
              "{status:gathered_data_uncited}")
    out, prompts = _run(monkeypatch, vault, ledger,
                        ["acknowledged", denial, denial])

    assert out["mode"] == "fallback"
    assert out["verification"]["cause"] == "denied_available_figure"
    assert fact_template.fact_key(
        "alcohol_drinks", "2026-08-03", "alcohol_drinks") in prompts[2]


def test_the_formatter_does_not_round_a_drink_count_to_a_whole_drink():
    from health_advisor import metrics as mx

    assert mx.format_presentation("alcohol_drinks", 2.5) == "2.5"
    assert mx.format_presentation("caffeine_drinks", 3.0) == "3"
    assert mx.format_presentation("alcohol_drinks", 1.1666, field="mean") == "1.17"
