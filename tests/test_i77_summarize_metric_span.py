"""#77 item 4a: `summarize_metric` must publish the DATA's span, not the
caller's requested range.

A spec like '2020-01-01:2039-12-31' over a 60-day vault used to come back
with `period` set to that raw 20-year spec, and every downstream label (the
period_label fact, in particular) rendered a 60-day mean as if it covered two
decades. The metrics layer already returns the true start/end of the rows it
summarised (`metrics.stats` -> `dates[0]`/`dates[-1]`), so no extra query is
needed to fix this — the fix is publishing that, and keeping the raw spec
around (as `requested_period`) so nothing is lost.
"""
from __future__ import annotations

import json

from health_advisor import deepdive_mcp
from health_advisor import fact_template
from health_advisor import mcp_server
from tests.conftest import seed_metric

_VAULT_START = "2029-11-05"
_VAULT_END = "2030-01-03"  # 60 days: Nov 5 .. Jan 3 inclusive


def _seed_60_days(conn):
    seed_metric(conn, "step_count", _VAULT_START, [1000.0] * 60)


def test_wide_explicit_spec_publishes_the_data_span(conn, tools):
    """A caller-requested range far wider than the vault must not be echoed
    back as the stat's period -- the stat was computed over 60 days, not 20
    years."""
    _seed_60_days(conn)
    out = tools.summarize_metric("step_count", "2020-01-01:2039-12-31")

    assert out["period"] == f"{_VAULT_START}:{_VAULT_END}"
    assert out["start"] == _VAULT_START
    assert out["end"] == _VAULT_END
    assert out["requested_period"] == "2020-01-01:2039-12-31"
    # nothing else about the existing contract moved
    assert out["metric"] == "step_count"
    assert out["n_days"] == 60
    assert out["mean"] == 1000.0


def test_spec_inside_the_data_publishes_itself_unchanged(conn, tools):
    """When the requested range is already inside the data, the published
    period is exactly the requested range -- this is the case that must not
    regress while the wide-spec case above gets fixed."""
    _seed_60_days(conn)
    inside_spec = "2029-11-10:2029-12-01"
    out = tools.summarize_metric("step_count", inside_spec)

    assert out["period"] == inside_spec
    assert out["start"] == "2029-11-10"
    assert out["end"] == "2029-12-01"
    assert out["requested_period"] == inside_spec


def test_period_label_no_longer_names_2020_or_2039(conn, vault, tmp_path):
    """The period_label fact built from the published period must describe
    the actual 60-day span, never the requested 2020-2039 window."""
    _seed_60_days(conn)
    ledger = deepdive_mcp._CallLedger(str(tmp_path / "calls.jsonl"))
    raw_tools = mcp_server.build_tools(vault)
    wrapped = deepdive_mcp._ledger_wrapper(
        "summarize_metric", raw_tools["summarize_metric"], ledger)

    out = wrapped("step_count", period="2020-01-01:2039-12-31")
    assert out["period"] == f"{_VAULT_START}:{_VAULT_END}"

    with open(tmp_path / "calls.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    facts = fact_template.build_fact_set(records)

    label_facts = [f for f in facts.values() if f["field"] == "period_label"]
    assert label_facts, "expected at least one period_label fact"
    for fact in label_facts:
        assert "2020" not in fact["value"]
        assert "2039" not in fact["value"]
