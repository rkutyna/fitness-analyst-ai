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

The scan covers EVERY module of the ``health_advisor`` package, not only
``chat.py``, and matches a model call by name whether it is reached as an
attribute (``llm.complete``) or as a bare name (an injected ``complete``
callable, as ``analyst.run_analyst`` uses). Every model-invoking call site
the scan finds is pinned in ``_EXPECTED_MODEL_CALL_SITES`` together with
whether it qualifies, so a new model call ANYWHERE in the engine fails
loudly until someone decides which side of the criterion it is on. As of
health_advisor#428 the non-qualifying sites are: ``chat._ask_judge`` (grades
an answer already produced; no vault handle), ``agents.run_model`` (a
generic transport wrapper with ``ctx`` but no window -- its prompts are
assembled by the CONSUMER's briefing and review paths, which is where their
date statement has to live), and ``analyst.run_analyst`` (writes Python
analysis code against a schema summary; holds a vault path but no window).

How each qualifying site is anchored to the vault's current date (#428
Done-when 3):

* the ask first draft, its retry, and the fact-template gather turn state
  the date in the prompt (``_render_ask_calendar_dates``);
* the fact-template final narration and its repair do NOT, deliberately:
  the prompt forbids digits and ISO dates in prose and routes every period
  through a Python-rendered placeholder, so a stated date would be a digit
  trap. They are anchored in Python instead -- ``_mark_stale_window``
  refuses a narration whose cited periods end well before the data -- and
  the tests below assert that anchor on each of the two call sites;
* the span-suppression regeneration has neither. Its prompt's contract is
  that no failed figure's digits reach the model, which a date line breaks
  (a year such as 2026 contains the figure 26 -- pinned by
  ``test_span_suppression_rewrites_from_only_verified_claims``). It stays an
  open, measured gap; it runs only under ``HA_ASK_SPAN_SUPPRESS``.
"""
from __future__ import annotations

import ast
import pathlib
from collections import Counter

import pytest

import health_advisor
from health_advisor import chat
from health_advisor import db as dbmod
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

# A call is model-invoking when it names one of these functions, reached as
# ``llm.<name>`` / ``agents.<name>`` or as a bare (imported or injected) name.
_MODEL_CALL_NAMES = {"complete", "tool_loop", "run_model"}
_MODEL_CALL_MODULES = {"llm", "agents"}


def _takes_vault_handle_and_window(funcdef: ast.FunctionDef) -> bool:
    names = set()
    for arglist in (funcdef.args.posonlyargs, funcdef.args.args,
                    funcdef.args.kwonlyargs):
        names.update(arg.arg for arg in arglist)
    has_vault_handle = "ctx" in names or "conn" in names
    has_window = "as_of" in names or any("window" in n for n in names)
    return has_vault_handle and has_window


def _model_call_name(func: ast.expr) -> str | None:
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            and func.value.id in _MODEL_CALL_MODULES
            and func.attr in _MODEL_CALL_NAMES):
        return f"{func.value.id}.{func.attr}"
    if isinstance(func, ast.Name) and func.id in _MODEL_CALL_NAMES:
        return func.id
    return None


def _discover_model_call_sites() -> list[dict]:
    """Every model-invoking call site in every ``health_advisor`` module.

    ``llm.py`` itself is skipped: it DEFINES the model calls, and its own
    internals reach the transports, not these names.
    """
    package_dir = pathlib.Path(health_advisor.__file__).parent
    sites: list[dict] = []
    for path in sorted(package_dir.glob("*.py")):
        module = path.stem
        if module == "llm":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack: list[ast.FunctionDef] = []

        class _Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):  # noqa: N802 (ast API name)
                stack.append(node)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):  # noqa: N802 (ast API name)
                target = _model_call_name(node.func)
                if target is not None:
                    enclosing = stack[-1] if stack else None
                    sites.append({
                        "module": module,
                        "function": (f"{module}.{enclosing.name}"
                                     if enclosing else f"{module}.<module>"),
                        "call_line": node.lineno,
                        "target": target,
                        "qualifies": bool(
                            enclosing is not None
                            and _takes_vault_handle_and_window(enclosing)),
                    })
                self.generic_visit(node)

        _Visitor().visit(tree)
    return sorted(sites, key=lambda s: (s["module"], s["call_line"]))


def _discover_windowed_prompt_call_sites() -> list[dict]:
    """Statically enumerate qualifying model-invoking call sites, package-wide,
    per the module docstring's criterion."""
    return [site for site in _discover_model_call_sites() if site["qualifies"]]


# The discovered shape as of this file's writing. A per-path test below
# exists for each entry. If discovery's total for a function changes --
# a new call site added, one removed, one merged -- this manifest goes out
# of sync and the test that checks it fails LOUDLY, the same guarantee
# ``xfail(strict=True)`` gives per-path, applied to the discovery step
# itself: silently leaving a new call site untested is exactly the failure
# mode #428 is about.
_EXPECTED_CALL_SITE_COUNTS = {
    "chat._answer_question_inner": 2,
    "chat._answer_fact_template": 3,
    "chat._try_span_suppression": 1,
}

# Every model-invoking call site in the package, qualifying or not, and why
# the non-qualifying ones are out (see the module docstring).
_EXPECTED_MODEL_CALL_SITES = {
    **{name: (count, True) for name, count in _EXPECTED_CALL_SITE_COUNTS.items()},
    "chat._ask_judge": (1, False),
    "agents.run_model": (2, False),
    "analyst.run_analyst": (2, False),
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
    assert functions >= {"chat._answer_question_inner",
                         "chat._answer_fact_template"}, (
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


def test_every_model_call_in_the_package_is_classified():
    """No model call anywhere in the engine escapes the criterion unseen.

    The windowed manifest above can only notice a new call site that
    already qualifies; this one notices a model call in ANY module -- a new
    briefing path, a second judge, an injected ``complete`` -- and forces a
    decision about which side of the criterion it belongs on.
    """
    sites = _discover_model_call_sites()
    shape: dict[str, tuple[int, bool]] = {}
    for site in sites:
        count, _ = shape.get(site["function"], (0, site["qualifies"]))
        shape[site["function"]] = (count + 1, site["qualifies"])
    assert shape == _EXPECTED_MODEL_CALL_SITES, (
        "the engine's model-invoking call sites changed; classify the new "
        f"one(s) in this file.\ndiscovered={shape}\n"
        f"expected={_EXPECTED_MODEL_CALL_SITES}\nsites={sites}")


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


# NOTE on the two fact-template xfails below. Whether these paths should
# state the date in the PROMPT was left open when this file was written; #428
# Done-when 4 answered it the other way. The final-narration prompt forbids
# digits and ISO dates in prose and routes any period through a Python-owned
# ``{fact|...|field=period_label}`` placeholder, so a stated date would be a
# digit trap. Instead both call sites are anchored to the vault's date IN
# PYTHON: ``_mark_stale_window`` refuses a template whose latest cited period
# ends more than ``STALE_WINDOW_DAYS`` before its metrics' most recent data.
# The two ``..._is_anchored_by_the_stale_window_check`` tests at the end of
# this file assert that anchor per call site.
#
# These stay xfail(strict=True) because the mechanical claim in each reason
# string is still true: if someone adds the date here, the xfail flips and
# forces the digit-trap question to be measured rather than drifted into.
# See health_advisor#428.

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


# ---------------------------------------------------------------------------
# The Python anchor for fact-template call sites #2 and #3 (#428 Done-when 4)
#
# The final narration and its repair do not state the date in the prompt
# (see the NOTE above); the stale-window marker is what binds them to the
# vault's current date. One test per call site, so removing the marker from
# either one turns exactly that test red.
# ---------------------------------------------------------------------------

def _seed_current_resting_rate(vault):
    conn = vault.connect()
    dbmod.init_db(conn)
    # Resting heart rate recorded daily through the as-of below.
    seed_metric(conn, "resting_heart_rate", "2026-07-01", [60] * 52)
    conn.close()


def _two_week_resting_rate_ledger():
    ledger = []
    for sequence, period in enumerate(
            ("2026-07-20:2026-07-26", "2026-08-10:2026-08-16"), start=1):
        ledger.append({
            "sequence": sequence, "tool_name": "synthetic_metric",
            "arguments": {},
            "result": {
                "metric": "resting_heart_rate", "period": period,
                "mean": 61 + sequence, "unit": "bpm",
                "presentation": {"metric": "resting_heart_rate",
                                 "period": period, "field": "presentation",
                                 "value": f"{61 + sequence} bpm"},
            },
        })
    return ledger


def _run_resting_rate_templates(monkeypatch, vault, templates):
    replies = iter(["acknowledged", *templates])
    ledger = _two_week_resting_rate_ledger()
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *a, **k: next(replies))
    capture: list = []
    result = chat.answer_question(
        vault, "How has my resting heart rate been?", as_of="2026-08-21",
        capture=capture)
    return result, capture


_STALE_KEY = fact_template.fact_key(
    "resting_heart_rate", "2026-07-20:2026-07-26", "mean")
_CURRENT_KEY = fact_template.fact_key(
    "resting_heart_rate", "2026-08-10:2026-08-16", "mean")


def test_fact_template_final_narration_is_anchored_by_the_stale_window_check(
        monkeypatch, vault):
    """Call site #2: a first draft citing only a window that ended 26 days
    before the data is refused with the stale-window cause."""
    _seed_current_resting_rate(vault)
    result, capture = _run_resting_rate_templates(monkeypatch, vault, [
        "Your resting heart rate averaged {" + _STALE_KEY + "}.",
        "Your resting heart rate averaged {" + _CURRENT_KEY + "}.",
    ])

    assert capture[0]["verification"]["cause"] == "stale_window"
    assert result["mode"] == "narration"
    assert result["verification"]["cause"] == "ok"


def test_fact_template_repair_is_anchored_by_the_stale_window_check(
        monkeypatch, vault):
    """Call site #3: a repair that cites the stale window again is refused
    too, so one failed attempt is never the way past the check."""
    _seed_current_resting_rate(vault)
    stale = "Your resting heart rate averaged {" + _STALE_KEY + "}."
    result, capture = _run_resting_rate_templates(
        monkeypatch, vault, [stale, stale])

    assert capture[1]["verification"]["cause"] == "stale_window"
    assert result["mode"] == "fallback"
    assert result["verification"]["cause"] == "stale_window"
