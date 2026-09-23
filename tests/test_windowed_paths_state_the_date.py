"""health_advisor#428 Done-when 3 and 5: enumerate every windowed/period-
bearing prompt-building path FROM CODE and assert each one states the
vault's current date.

Background: engine commit fa01c62 (#70) made the ask path state the vault's
current date via ``chat._render_ask_calendar_dates``, called once at
``chat.py:2336``. #428 exists because a live answer was three weeks stale
with entirely correct figures -- the model invented the PERIOD, not a
number, and every grounding gate passed. Fixing the ask path alone is
explicitly NOT sufficient (#428's "Not done when"); this file is the
instrument that says which paths are covered and which are not. It does not
fix anything.

Discovery criterion (stated once, used by every test below): a "windowed
prompt-building path" is a call, inside ``health_advisor.chat``, to a
model-invoking function (``llm.complete``, ``llm.tool_loop``, or
``agents.run_model``) whose ENCLOSING function's parameters include a vault
handle (``ctx`` or ``conn``) together with a window-shaped parameter
(``as_of``, or any parameter whose name contains ``window``). That is the
mechanical shape of "a function that can produce a windowed/period-bearing
answer": it holds the vault to read a period FROM, and a way to bound WHICH
period -- exactly the two things a date statement in its prompt would be
protecting. A judge or scorer that only reads back an already-produced
answer (no vault handle) does not qualify, and correctly so: it cannot
invent a period, it can only grade one that already exists.

The rest of the package was checked by hand and excluded from the scan
because it cannot even take part: ``grep -rn "llm\\.complete(\\|llm\\.tool_loop("
health_advisor/*.py`` finds calls only in ``chat.py`` (and the definitions
in ``agents.py``); ``agents.run_model`` itself has ``ctx`` but no
``as_of``/window parameter, and ``analyst.build_repair_prompt`` /
``analyst_prompt.build_analyst_prompt`` take neither a vault handle nor a
window -- they write Python analysis code against an explicit schema
summary, not a narrated answer bound to a period.
"""
from __future__ import annotations

import ast
import inspect
from collections import Counter

import pytest

from health_advisor import chat
from health_advisor import fact_template
from health_advisor import llm
from health_advisor import vault as vaultmod
from tests.conftest import seed_metric


@pytest.fixture(autouse=True)
def _seed_a_metric_so_the_vault_is_not_day_zero(conn):
    """Every test here must reach a real prompt-building call. An empty
    vault answers with the day-zero status text
    (``chat._no_data_answer``) before any prompt is ever built, which would
    make every test below vacuously pass without exercising anything --
    the "instrument that measures nothing" failure mode this file exists to
    avoid."""
    seed_metric(conn, "fixture_metric", "2026-01-01", [1])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_MODEL_CALL_TARGETS = {
    ("llm", "complete"), ("llm", "tool_loop"), ("agents", "run_model"),
}


def _takes_vault_handle_and_window(funcdef: ast.FunctionDef) -> bool:
    names = set()
    for arglist in (funcdef.args.posonlyargs, funcdef.args.args,
                    funcdef.args.kwonlyargs):
        names.update(arg.arg for arg in arglist)
    has_vault_handle = "ctx" in names or "conn" in names
    has_window = "as_of" in names or any("window" in n for n in names)
    return has_vault_handle and has_window


def _discover_windowed_prompt_call_sites() -> list[dict]:
    """Statically enumerate qualifying model-invoking call sites in
    ``health_advisor.chat``, per the module docstring's criterion."""
    source = inspect.getsource(chat)
    tree = ast.parse(source)
    stack: list[ast.FunctionDef] = []
    sites: list[dict] = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):  # noqa: N802 (ast API name)
            stack.append(node)
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):  # noqa: N802 (ast API name)
            func = node.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and (func.value.id, func.attr) in _MODEL_CALL_TARGETS
                    and stack
                    and _takes_vault_handle_and_window(stack[-1])):
                sites.append({
                    "function": stack[-1].name,
                    "def_line": stack[-1].lineno,
                    "call_line": node.lineno,
                    "target": f"{func.value.id}.{func.attr}",
                })
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sorted(sites, key=lambda s: s["call_line"])


# The discovered shape as of this file's writing. A per-path test below
# exists for each entry. If discovery's total for a function changes --
# a new call site added, one removed, one merged -- this manifest goes out
# of sync and the test that checks it fails LOUDLY, the same guarantee
# ``xfail(strict=True)`` gives per-path, applied to the discovery step
# itself: silently leaving a new call site untested is exactly the failure
# mode #428 is about.
_EXPECTED_CALL_SITE_COUNTS = {
    "_answer_question_inner": 2,
    "_answer_fact_template": 3,
    "_try_span_suppression": 1,
}


def test_discovery_finds_a_plausible_number_of_windowed_paths():
    """Done-when 3: the path list comes from code, not a hand-written tuple.

    A hand-typed list of four names would pass trivially and prove nothing;
    require the AST walk to actually find call sites, and require enough of
    them that an empty or near-empty result (discovery silently broken, or
    the engine having dropped these branches) cannot pass unnoticed.
    """
    sites = _discover_windowed_prompt_call_sites()
    assert len(sites) >= 4, (
        "windowed prompt call-site discovery collapsed to "
        f"{len(sites)} site(s); expected at least 4. This either means "
        f"discovery is broken or the engine restructured these paths -- "
        f"either way this must be investigated, not silenced. sites={sites}")
    functions = {site["function"] for site in sites}
    assert functions >= {"_answer_question_inner", "_answer_fact_template"}, (
        f"expected core narration functions missing from discovery: {sites}")


def test_discovered_call_sites_match_the_covered_manifest():
    """Every discovered call site must have a per-path test in this file.

    If a future change adds a third ``llm.complete`` call inside
    ``_try_span_suppression``, or a wholly new windowed-narration function,
    this fails with the new shape printed -- forcing this file's per-path
    tests to be updated rather than silently leaving the new path unmeasured.
    """
    sites = _discover_windowed_prompt_call_sites()
    counts = dict(Counter(site["function"] for site in sites))
    assert counts == _EXPECTED_CALL_SITE_COUNTS, (
        "discovered windowed-prompt call sites changed shape; update this "
        f"file's per-path tests to match.\ndiscovered={counts}\n"
        f"expected={_EXPECTED_CALL_SITE_COUNTS}\nsites={sites}")


# ---------------------------------------------------------------------------
# Per-path assertions
#
# One test per call site in _EXPECTED_CALL_SITE_COUNTS above (6 total: 2 in
# _answer_question_inner, 3 in _answer_fact_template, 1 in
# _try_span_suppression). Each asserts the prompt sent to the model on that
# specific call states the vault's current date. The three that currently
# fail are marked xfail(strict=True) -- that list IS the finding.
# ---------------------------------------------------------------------------


def _analyst_resting_rate_ledger():
    return [{
        "sequence": 1,
        "tool_name": "analyst_query",
        "arguments": {},
        "result": {"tables": [{
            "name": "resting_rate",
            "columns": ["day", "rate"],
            "units": ["date", "count/min"],
            "rows": [["2026-08-01", 63], ["2026-08-02", 60]],
            "row_count": 2,
        }]},
    }]


# --- _answer_question_inner: call site #1, the first-draft llm.tool_loop ---

def test_answer_question_first_draft_prompt_states_the_date(
        monkeypatch, vault, conn):
    """_answer_question_inner call site #1 (llm.tool_loop, first draft).

    This is the path #439/#70 already fixed: the shared ``prompt`` built in
    _answer_question_inner runs through _render_ask_calendar_dates before
    any tool_loop call.
    """
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "0")
    calls: list[str] = []
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "tool_loop", lambda prompt, **k: calls.append(prompt) or "")

    chat.answer_question(vault, "How am I doing?", as_of="2026-02-03")

    assert calls, "the first-draft tool_loop call never happened"
    assert "2026-02-03" in calls[0]


# --- _answer_question_inner: call site #2, the retry-draft llm.tool_loop ---

def test_answer_question_retry_draft_prompt_states_the_date(
        monkeypatch, vault, conn):
    """_answer_question_inner call site #2 (llm.tool_loop, retry draft).

    The retry always fires when the first draft comes back empty (no gate
    to enable it), so returning "" twice deterministically reaches it. The
    retry prompt is `prompt + "..."` -- a superstring of the same dated
    prompt -- so this should already pass; it is included so the manifest
    above has one test per discovered call site, not three-out-of-four.
    """
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "0")
    calls: list[str] = []
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "tool_loop", lambda prompt, **k: calls.append(prompt) or "")

    chat.answer_question(vault, "How am I doing?", as_of="2026-02-03")

    assert len(calls) >= 2, "the retry-draft tool_loop call never happened"
    assert "2026-02-03" in calls[1]


# --- _answer_fact_template: call site #1, the tool-gathering llm.tool_loop -

def test_fact_template_gather_prompt_states_the_date(monkeypatch, vault, conn):
    """_answer_fact_template call site #1 (llm.tool_loop, tool-gather turn).

    Reached via the shared ``prompt`` from _answer_question_inner (same
    dated string, plus a gather-only suffix), so this should already pass.
    """
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    calls: list[str] = []
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **k: calls.append(prompt) or "acknowledged")

    chat.answer_question(vault, "How am I doing?", as_of="2026-02-03")

    assert calls, "the fact-template gather tool_loop call never happened"
    assert "2026-02-03" in calls[0]


# NOTE on the two fact-template xfails below, added by the orchestrator on
# review: whether these paths SHOULD state the date is an OPEN DESIGN QUESTION,
# not a settled defect, and `xfail(strict=True)` should not be read as deciding
# it. The final-narration prompt instructs the model to write a *placeholder*
# for any date it wants in prose
# ({fact|metric=...|period=...|field=period_label}), which is Python-owned and
# is a STRONGER guarantee than stating a date the model could then paraphrase.
# On that reading, the narration turn not knowing today is correct by design,
# and #428's stale window enters upstream -- at the gather turn that chooses
# the window, which DOES state the date (call site #1, passing).
#
# These stay xfail(strict=True) because the mechanical claim in each reason
# string is true and worth pinning: if someone adds the date here, the xfail
# flips and forces the question to be answered rather than drifted into. What
# must not happen is a future session reading "xfail" as "known bug, go fix
# it". See health_advisor#428.

# --- _answer_fact_template: call site #2, the final-narration llm.tool_loop

@pytest.mark.xfail(
    strict=True,
    reason="health_advisor#428: the fact-template final-narration prompt "
           "('You are writing the final answer...') is built fresh in "
           "_answer_fact_template and never calls "
           "_render_ask_calendar_dates, unlike the gather prompt it "
           "follows.")
def test_fact_template_final_narration_prompt_states_the_date(
        monkeypatch, vault, conn):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = [{
        "sequence": 1, "tool_name": "synthetic_metric", "arguments": {},
        "result": {
            "metric": "jog_minutes", "unit": "min", "period": "2026-08-17",
            "mean": 50.1,
            "presentation": {"metric": "jog_minutes", "period": "2026-08-17",
                              "field": "presentation", "value": "50 m"},
        },
    }]
    key = fact_template.fact_key("jog_minutes", "2026-08-17", "mean")
    responses = iter(["acknowledged", "Your run was {" + key + "}."])
    calls: list[str] = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **k: calls.append(prompt) or next(responses))

    chat.answer_question(vault, "How was my run?", as_of="2026-02-03")

    assert len(calls) >= 2, "the final-narration tool_loop call never happened"
    assert "2026-02-03" in calls[1]


# --- _answer_fact_template: call site #3, the repair-retry llm.tool_loop --

@pytest.mark.xfail(
    strict=True,
    reason="health_advisor#428: the fact-template repair prompt is "
           "`final_prompt + refusal detail`, so it inherits the same gap "
           "as the final-narration prompt and never states a current "
           "date.")
def test_fact_template_repair_prompt_states_the_date(monkeypatch, vault, conn):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = _analyst_resting_rate_ledger()
    cell_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-02")
    # First narration attempt states a bare digit outside a placeholder,
    # which the template gate refuses -- forcing exactly one repair retry.
    responses = iter([
        "acknowledged",
        "Your rate is 60 today.",
        "Your rate is {" + cell_key + "}.",
    ])
    calls: list[str] = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **k: calls.append(prompt) or next(responses))

    chat.answer_question(
        vault, "What is my resting heart rate today?", as_of="2026-02-03")

    assert len(calls) >= 3, "the repair-retry tool_loop call never happened"
    assert "2026-02-03" in calls[2]


# --- _try_span_suppression: the single llm.complete regeneration call -----

@pytest.mark.xfail(
    strict=True,
    reason="health_advisor#428: _span_regeneration_prompt only permits "
           "copying a date already present in the redacted question/draft "
           "-- it never states the vault's current date itself.")
def test_span_suppression_prompt_states_the_date(monkeypatch, vault, conn):
    monkeypatch.setenv("HA_ASK_SPAN_SUPPRESS", "1")
    calls: list[str] = []
    monkeypatch.setattr(
        llm, "complete",
        lambda prompt, **k: calls.append(prompt) or "Rewritten answer.")
    # A synthetic failed-but-mostly-verified attempt: _suppression_allowed
    # requires verified/(verified+unverified) >= 0.5, and _verified_claims
    # requires a claim/number pair with number["ok"] True.
    attempt = {
        "prose": "Your run covered 5 miles today.",
        "claims": [{"value": 5, "source": {"sequence": 1}}],
        "verification": {
            "verdict": {"numbers": [{"ok": True, "claimed": 5}]},
            "unsupported": [],
        },
        "ledger": [],
    }

    chat._try_span_suppression(
        vault, "How far did I run?", attempt, as_of="2026-02-03")

    assert calls, "the span-regeneration llm.complete call never happened"
    assert "2026-02-03" in calls[0]
