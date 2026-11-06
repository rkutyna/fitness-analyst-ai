"""An ask's ``as_of`` pins every tool's default window (downstream issue #596).

``chat.answer_question(..., as_of=D)`` tells the model that D is the current
date and the most recent day with data. Before this test existed nothing told
the TOOLS: a tool called with its default window anchored on the vault's
latest day, or on the wall clock, so a backdated ask on a vault holding data
after D handed the model figures from after D. A replay against a fresh
snapshot is only a drop-in for an old one if ``as_of`` really pins the answer.

The audit drives the real ask path end to end -- ``answer_question``, the real
``llm.tool_loop``, the real registry and the real call ledger -- with a fake
model whose single gather turn calls every offered tool once with only its
REQUIRED arguments (dates a model told "today is D" would send). Every
optional argument is left at its default, because the default window is what
is under test. Then it walks every ledgered result and collects every ISO date
after D, anywhere in a key or a value.

The mechanism under test: ``answer_question`` binds the registry to
``ctx.at(as_of)``, every tool takes its default window end and its "latest"
day from ``analysis.request_as_of(conn, ctx)``, and an explicit date after the
as-of is clamped to it. Two further tests below hold the stronger forms: the
full vault under the as-of answers exactly as a copy truncated at it, and the
registry hands engine tools, host tools and replacements the as-of.

A NEW coach tool must be added to :func:`required_args`, so the surface cannot
grow without being audited.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import date, timedelta

import pytest

from health_advisor import chat, demo, llm
from health_advisor.context import VaultContext

END = demo.DEFAULT_END_DATE                     # last day of data in the vault
AS_OF = (date.fromisoformat(END) - timedelta(days=20)).isoformat()
VAULT_DAYS = 60
METRIC = "resting_heart_rate"
QUESTION = "How is my resting heart rate doing?"

# Keys whose value is a true fact about the vault FILE rather than a figure for
# a period, and so may name a day after the session's as-of. Keyed by tool and
# by the JSON path the walk reports (list indices collapsed to `[]`).
#
# - list_available_metrics `metrics[].last_date`: the discovery tool states the
#   span each metric is held for. It is the vault's own extent, not a window
#   any figure was computed over; `first_date` beside it is the same kind of
#   fact. Clipping it would make the tool misreport what the file holds.
ALLOWED = {
    ("list_available_metrics", "metrics[].last_date"),
}

_ISO_DAY = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def required_args(ref: str) -> dict[str, dict]:
    """The arguments a model told that ``ref`` is today sends each coach tool.

    Only REQUIRED parameters are given. A required date is the one a model
    would pick for "today", "the last four weeks", "this month".
    """
    day = date.fromisoformat(ref)
    four_weeks = (day - timedelta(days=27)).isoformat()
    eight_weeks = (day - timedelta(days=55)).isoformat()
    return {
        "list_available_metrics": {},
        "get_daily_series": {"metric": METRIC},
        "summarize_metric": {"metric": METRIC},
        "compare_periods": {"metric": METRIC, "period_a": "7d",
                            "period_b": "28d"},
        "get_intraday": {"metric": "heart_rate", "day": ref},
        "get_hr_zones": {"day": ref},
        "list_workouts": {},
        "get_workout_segments": {"day": ref},
        "get_impact_volume": {"start": four_weeks, "end": ref},
        "get_sleep_regularity": {},
        "get_training_load_detail": {},
        "get_run_form": {},
        "get_briefing": {},
        "get_latest": {"metric": METRIC},
        "correlate_metrics": {"metric_x": METRIC, "metric_y": "step_count"},
        "scan_correlations": {"target": METRIC},
        "get_subjective": {"start_date": four_weeks, "end_date": ref},
        "food_lookup": {},
        "food_meal_total": {"items": []},
        "get_weekly_series": {"metric": METRIC, "start": eight_weeks,
                              "end": ref},
        "get_block_comparison": {"metric": METRIC, "block_weeks": 2,
                                 "as_of": ref},
        "get_block_structure": {"day": ref},
        "get_weekly_readiness": {},
        "get_benchmark_series": {},
        "get_monthly_running_power": {"month": ref[:7]},
        # No analyst_query_fn is passed here, so the tool answers with a
        # ``refused`` dict and never reaches the analyst sandbox.
        "analyst_query": {"question": QUESTION},
    }


def _ask_tool_results(ctx: VaultContext, ledger_path, monkeypatch, *,
                      as_of: str | None, ref: str) -> dict[str, object]:
    """One real ask; returns ``{tool_name: result}`` from its call ledger."""
    table = required_args(ref)
    offered: list[str] = []
    posts = []

    def fake_post(payload, timeout):
        posts.append(payload)
        if len(posts) > 1:
            return {"message": {"role": "assistant", "content": "Noted."}}
        names = [tool["function"]["name"] for tool in payload["tools"]]
        offered.extend(n for n in names if n != llm.SUBMIT_ANSWER_NAME)
        return {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": name, "arguments": table[name]}}
            for name in offered if name in table]}}

    monkeypatch.setattr(llm, "_ollama_chat_post", fake_post)
    chat.answer_question(ctx, QUESTION, as_of=as_of,
                         ledger_path=str(ledger_path))

    host = set(llm.HOST_SUPPLIED_TOOLS)
    assert set(offered) - host == set(table), (
        "required_args() must name exactly the coach tools the ask offers: "
        f"unaudited {sorted(set(offered) - host - set(table))}, "
        f"stale {sorted(set(table) - set(offered))}")
    results: dict[str, object] = {}
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            name = record["tool_name"]
            assert name not in results, f"{name} ledgered twice"
            assert not record["result_elided"], (
                f"{name}: result elided from the ledger, so it cannot be walked")
            results[name] = record["result"]
    assert set(results) == set(table), sorted(set(table) ^ set(results))
    errors = {name: result for name, result in results.items()
              if isinstance(result, dict) and "_error_type" in result}
    assert not errors, errors
    return results


def _dates_after(node, horizon: str, path: str = ""):
    """Yield ``(path, day)`` for every ISO day after ``horizon`` in ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _dates_after(str(key), horizon, child + "<key>")
            yield from _dates_after(value, horizon, child)
    elif isinstance(node, list):
        for item in node:
            yield from _dates_after(item, horizon, path + "[]")
    elif isinstance(node, str):
        for match in _ISO_DAY.findall(node):
            try:
                date.fromisoformat(match)
            except ValueError:
                continue
            if match > horizon:
                yield path, match


def offenders(results: dict[str, object], horizon: str) -> dict[str, dict]:
    """``{tool: {path: [first, last, count]}}`` of every unallowed late day."""
    found: dict[str, dict] = {}
    for name, result in sorted(results.items()):
        for path, day in _dates_after(result, horizon):
            if (name, path) in ALLOWED:
                continue
            seen = found.setdefault(name, {}).setdefault(path, [day, day, 0])
            seen[0], seen[1] = min(seen[0], day), max(seen[1], day)
            seen[2] += 1
    return found


@pytest.fixture(scope="module")
def built_vault(tmp_path_factory):
    """Sixty days of synthetic data ending on END; the ask is 20 days earlier."""
    path = tmp_path_factory.mktemp("as_of_binds") / "demo.db"
    demo.build_demo_vault(path, days=VAULT_DAYS, read_only=False)
    return path


@pytest.fixture
def session(built_vault, tmp_path):
    copy = tmp_path / "vault.db"
    shutil.copy(built_vault, copy)
    return VaultContext.local(copy, user_id="test")


def test_no_default_window_reaches_past_the_asks_as_of(session, tmp_path,
                                                       monkeypatch):
    results = _ask_tool_results(session, tmp_path / "ledger.jsonl",
                                monkeypatch, as_of=AS_OF, ref=AS_OF)
    found = offenders(results, AS_OF)
    assert not found, (
        f"tools published days after the ask's as_of {AS_OF}:\n"
        + json.dumps(found, indent=1, sort_keys=True))


def test_the_audit_sees_a_late_day(session, tmp_path, monkeypatch):
    """The walk is not vacuous: the allow-listed key DOES carry a late day,
    so a walk that found nothing would fail here rather than pass above."""
    results = _ask_tool_results(session, tmp_path / "ledger.jsonl",
                                monkeypatch, as_of=AS_OF, ref=AS_OF)
    late = list(_dates_after(results["list_available_metrics"], AS_OF))
    assert ("metrics[].last_date", END) in late


def _truncated_copy(source, dest, horizon: str) -> None:
    """``source`` with every dated row after ``horizon`` deleted."""
    import os
    import sqlite3
    shutil.copy(source, dest)
    os.chmod(dest, 0o644)
    conn = sqlite3.connect(dest)
    try:
        cutoff_utc = (date.fromisoformat(horizon) + timedelta(days=1)).isoformat()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")]
        for table in tables:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            col = next((c for c in ("date", "local_date", "day") if c in cols),
                       None)
            if col is not None:
                conn.execute(f"DELETE FROM {table} WHERE {col} > ?", (horizon,))
            elif "start_utc" in cols:
                conn.execute(f"DELETE FROM {table} WHERE start_utc >= ?",
                             (cutoff_utc,))
        conn.commit()
    finally:
        conn.close()


def test_the_full_vault_as_of_d_answers_as_a_vault_that_ends_on_d(
        built_vault, tmp_path):
    """Stronger than the date walk: under the as-of, every coach tool returns
    exactly what it returns on a copy of the vault with every later row
    deleted. A figure computed from a later row with no date on it would pass
    the walk and fail here. The vault's extent is the one allowed difference."""
    from health_advisor import mcp_server as S
    full = tmp_path / "full.db"
    shutil.copy(built_vault, full)
    short = tmp_path / "short.db"
    _truncated_copy(built_vault, short, AS_OF)
    tools_full = S.build_tools(
        VaultContext.local(full, user_id="test").provider_facing().at(AS_OF))
    tools_short = S.build_tools(
        VaultContext.local(short, user_id="test").provider_facing().at(AS_OF))
    differ = []
    for name, kwargs in required_args(AS_OF).items():
        if name not in tools_full:          # analyst_query is not an MCP tool
            continue
        a = json.dumps(tools_full[name](**kwargs), sort_keys=True, default=repr)
        b = json.dumps(tools_short[name](**kwargs), sort_keys=True, default=repr)
        if a != b:
            differ.append(name)
    assert differ == ["list_available_metrics"]


def test_answer_question_binds_its_as_of_into_every_tool_context(
        session, tmp_path, monkeypatch):
    """The registry the ask builds hands engine tools, host tools and host
    replacements a context whose ``as_of`` is the ask's. Spied at the bind,
    which every one of the three goes through; no model is called."""
    from health_advisor import mcp_server as S

    bound: dict[str, object] = {}
    real_bind = S._bind

    def spy(fn, ctx):
        bound[fn.__name__] = ctx.as_of
        return real_bind(fn, ctx)

    called: dict[str, object] = {}

    def get_planned_session(ctx: VaultContext, day: str = "") -> dict:
        """Host tool stand-in: the session planned for a day."""
        called["get_planned_session"] = ctx.as_of
        return {"day": day, "planned": None}

    def get_latest(ctx: VaultContext, metric: str) -> dict:
        """Host replacement stand-in for the engine's get_latest."""
        called["get_latest"] = ctx.as_of
        return {"metric": metric}

    posts = []

    def fake_post(payload, timeout):
        posts.append(payload)
        if len(posts) > 1:
            return {"message": {"role": "assistant", "content": "Noted."}}
        return {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_planned_session",
                          "arguments": {"day": AS_OF}}},
            {"function": {"name": "get_latest",
                          "arguments": {"metric": METRIC}}}]}}

    monkeypatch.setattr(S, "_bind", spy)
    monkeypatch.setattr(llm, "_ollama_chat_post", fake_post)
    llm.register_tools(get_planned_session)
    llm.register_replacement(get_latest)
    try:
        chat.answer_question(session, QUESTION, as_of=AS_OF,
                             ledger_path=str(tmp_path / "ledger.jsonl"))
    finally:
        llm.unregister_tools("get_planned_session")
        llm.unregister_replacements("get_latest")
    engine = {"get_daily_series", "summarize_metric", "list_workouts",
              "get_briefing", "get_sleep_regularity"}
    assert engine <= set(bound)
    assert {name: bound[name] for name in engine} == dict.fromkeys(engine, AS_OF)
    assert bound["get_planned_session"] == AS_OF      # host tool
    assert bound["get_latest"] == AS_OF               # host replacement
    assert called == {"get_planned_session": AS_OF, "get_latest": AS_OF}
