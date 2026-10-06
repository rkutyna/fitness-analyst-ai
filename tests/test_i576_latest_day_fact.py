"""A ``get_latest`` answer publishes its latest-day figure as a citable fact.

consumer #576: when ``get_latest`` was the only figure-bearing tool call, the
fact set was empty -- the ``latest_day`` value belonged to no metric in the
ledger flattening -- so the model held a figure it had no legal slot to state
and wrote a denial, which the available-figure gate then (rightly) refused.
"""
from __future__ import annotations

import pytest

from health_advisor import chat, fact_template, llm
from tests.conftest import seed_metric

ASK = "What is my VO2 max right now?"
AS_OF = "2026-08-21"


def _ledger(tools, metric: str) -> list[dict]:
    """The ledger of a model that listed the metrics and asked for the latest."""
    return [
        {"sequence": 1, "tool_name": "list_available_metrics",
         "arguments": {}, "result": tools.list_available_metrics()},
        {"sequence": 2, "tool_name": "get_latest",
         "arguments": {"metric": metric}, "result": tools.get_latest(metric)},
    ]


def _latest_day_facts(facts: dict, metric: str, day: str) -> dict[str, dict]:
    return {k: v for k, v in facts.items()
            if v["metric"] == metric and v["period"] == day}


@pytest.fixture
def vo2(conn, tools):
    seed_metric(conn, "vo2_max", "2026-08-10", [40.0, 41.0, 41.5, 42.0])
    return _ledger(tools, "vo2_max")


def test_latest_only_ledger_for_a_last_metric_publishes_the_value_and_label(vo2):
    result = vo2[1]["result"]
    assert result["agg"] == "last"
    assert result["latest_sample"] is None
    assert result["latest_sample_status"]["status"] == "unavailable"
    day = result["latest_day"]["date"]

    facts = fact_template.build_fact_set(vo2)

    mine = _latest_day_facts(facts, "vo2_max", day)
    assert set(v["field"] for v in mine.values()) == {"value", "period_label"}
    value = next(v for v in mine.values() if v["field"] == "value")
    # The display is the presentation Python computed, not a second rendering.
    assert value["display"] == result["latest_day"]["presentation"]["value"]
    assert value["value"] == result["latest_day"]["value"]
    label = next(v for v in mine.values() if v["field"] == "period_label")
    assert label["display"]
    # Nothing the tool did not return is published, and the eligibility walk
    # (the independent reader) agrees with the publisher.
    assert fact_template.publish_completeness(vo2, facts) == set()


def test_latest_only_ledger_for_a_sum_metric_publishes_the_value_and_label(
        conn, tools):
    seed_metric(conn, "step_count", "2026-08-10", [8000, 9000, 10234])
    ledger = _ledger(tools, "step_count")
    result = ledger[1]["result"]
    assert result["agg"] == "sum"
    day = result["latest_day"]["date"]

    facts = fact_template.build_fact_set(ledger)

    mine = _latest_day_facts(facts, "step_count", day)
    assert set(v["field"] for v in mine.values()) == {"value", "period_label"}
    value = next(v for v in mine.values() if v["field"] == "value")
    assert value["display"] == result["latest_day"]["presentation"]["value"]
    assert fact_template.publish_completeness(ledger, facts) == set()


def test_a_metric_with_no_latest_day_publishes_nothing(conn, tools):
    # The metric exists in the vocabulary but has no stored day.
    seed_metric(conn, "fixture_metric", "2026-08-10", [1.0])
    result = tools.get_latest("vo2_max")
    ledger = [{"sequence": 1, "tool_name": "get_latest",
               "arguments": {"metric": "vo2_max"}, "result": result}]
    assert result.get("latest_day") is None

    assert fact_template.build_fact_set(ledger) == {}
    assert fact_template.eligible_fact_keys(ledger) == set()


def test_latest_sample_is_not_published_as_a_second_value_for_the_day(
        conn, tools):
    """The day's aggregate is the cited figure; the sample half stays as is."""
    from tests.test_daily_last import _record
    from health_advisor import db as dbmod
    _record(conn, "heart_rate", 61.0, "2026-07-21T08:00:00+00:00")
    _record(conn, "heart_rate", 58.0, "2026-07-21T09:00:00+00:00")
    dbmod.recompute_daily_metrics(conn, pairs=[("heart_rate", "2026-07-21")])
    conn.commit()
    ledger = _ledger(tools, "heart_rate")
    result = ledger[1]["result"]
    assert result["latest_sample"]["value"] == 58.0

    facts = fact_template.build_fact_set(ledger)

    values = [f for f in facts.values() if f["field"] == "value"]
    assert [f["value"] for f in values] == [result["latest_day"]["value"]]


def _run(monkeypatch, vault, ledger, responses):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    replies = iter(responses)
    prompts = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **k: prompts.append(prompt) or next(replies))
    capture = []
    result = chat.answer_question(vault, ASK, as_of=AS_OF, capture=capture)
    return result, capture, prompts


def test_template_citing_the_latest_day_slot_renders_and_verifies(
        monkeypatch, vault, vo2):
    result = vo2[1]["result"]
    day = result["latest_day"]["date"]
    value_key = fact_template.fact_key("vo2_max", day, "value")
    label_key = fact_template.fact_key("vo2_max", day, "period_label")
    template = "Your aerobic fitness reading was {%s} {%s}." % (
        value_key, label_key)

    out, capture, _ = _run(monkeypatch, vault, vo2, ["acknowledged", template])

    assert out["mode"] == "narration"
    assert out["verification"]["cause"] == "ok"
    assert result["latest_day"]["presentation"]["value"] in out["text"]


def test_denial_over_the_latest_day_figure_is_still_refused(
        monkeypatch, vault, vo2):
    denial = ("That is not among the figures this lookup returned. "
              "{status:gathered_data_uncited}")
    out, capture, prompts = _run(
        monkeypatch, vault, vo2, ["acknowledged", denial, denial])

    assert out["mode"] == "fallback"
    assert out["verification"]["cause"] == "denied_available_figure"
    day = vo2[1]["result"]["latest_day"]["date"]
    # The retry is now offered the slot it was previously missing.
    assert fact_template.fact_key("vo2_max", day, "value") in prompts[2]


def test_denial_is_legitimate_when_there_is_no_latest_day(
        monkeypatch, vault, conn, tools):
    seed_metric(conn, "fixture_metric", "2026-08-10", [1.0])
    ledger = [{"sequence": 1, "tool_name": "get_latest",
               "arguments": {"metric": "vo2_max"},
               "result": tools.get_latest("vo2_max")}]
    denial = "I have no recorded reading of your aerobic fitness."

    out, _, _ = _run(monkeypatch, vault, ledger, ["acknowledged", denial, denial])

    assert out["verification"]["cause"] != "denied_available_figure"
