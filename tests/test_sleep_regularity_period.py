"""A published sleep-regularity period is the data's span, not the query's (#357).

Called with no arguments, ``get_sleep_regularity`` asks from its all-of-history
default. Its SD presentation leaf used to carry that request window as its
period, so the fact set labelled the figure "from <default start> to <end>" on
a vault holding three weeks of nights, and a refusal stated that label as the
span of the user's data. The period must start no earlier than the metric's
first record, and must be the window the published value was computed from.
"""
from __future__ import annotations

import json
from datetime import date

from health_advisor import deepdive_mcp, demo, fact_template, mcp_server
from health_advisor import sleep_regularity as SR
from health_advisor.context import VaultContext


END = "2026-06-30"


def _no_argument_call(tmp_path, monkeypatch, days: int):
    db_path = tmp_path / f"demo-{days}.db"
    demo.build_demo_vault(db_path, days=days, end_date=END, read_only=False)
    monkeypatch.setattr(mcp_server, "_today",
                        lambda local_timezone=None: date.fromisoformat(END))
    ctx = VaultContext.local(db_path, user_id=f"demo-{days}")
    ledger_path = tmp_path / f"calls-{days}.jsonl"
    ledger = deepdive_mcp._CallLedger(ledger_path)
    tool = mcp_server.build_tools(ctx)["get_sleep_regularity"]
    result = deepdive_mcp._ledger_wrapper(
        "get_sleep_regularity", tool, ledger)()
    records = [json.loads(line)
               for line in ledger_path.read_text().splitlines()]
    with ctx.read_only() as conn:
        first, last, nights = conn.execute(
            "SELECT min(date), max(date), count(*) FROM daily_metrics "
            "WHERE metric = 'sleep_midpoint' AND last IS NOT NULL").fetchone()
    return result, fact_template.build_fact_set(records), first, last, nights


def _span(period: str) -> tuple[str, str]:
    start, end = period.split(":")
    return start, end


def test_short_vault_sd_period_starts_at_its_first_night(tmp_path, monkeypatch):
    result, facts, first, last, nights = _no_argument_call(
        tmp_path, monkeypatch, days=SR.MIN_WINDOW_DAYS + 5)
    # The request window is the all-of-history default; that is the premise.
    assert result["start"] < first
    assert nights >= SR.MIN_WINDOW_DAYS
    midpoint = result["midpoint_variability"]
    assert midpoint["status"] == "ok"

    sd_facts = [fact for fact in facts.values()
                if fact["metric"] == "sleep_midpoint_sd_28d"]
    # Non-vacuous: the value and its period label are both published.
    assert {fact["field"] for fact in sd_facts} >= {"latest_sd_hours",
                                                    "period_label"}
    for fact in sd_facts:
        start, end = _span(fact["period"])
        assert first <= start <= end <= last, fact
    assert midpoint["presentation"]["period"] == f"{first}:{last}"
    label = next(fact["value"] for fact in sd_facts
                 if fact["field"] == "period_label")
    start_day = date.fromisoformat(first)
    assert label.startswith(f"from {start_day:%a %b} {start_day.day} ")


def test_long_vault_sd_period_is_the_latest_28_day_window(tmp_path, monkeypatch):
    result, facts, first, last, _ = _no_argument_call(
        tmp_path, monkeypatch, days=60)
    window = result["midpoint_variability"]["latest_window"]
    assert window["end"] == last
    assert first < window["start"]
    assert (date.fromisoformat(window["end"])
            - date.fromisoformat(window["start"])).days < SR.WINDOW_DAYS
    periods = {fact["period"] for fact in facts.values()
               if fact["metric"] == "sleep_midpoint_sd_28d"}
    assert periods == {f"{window['start']}:{window['end']}"}
