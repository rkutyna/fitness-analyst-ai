"""health_advisor#557 -- declared facts, and no instruction text to a user.

Two defects from one live answer. (1) A host tool's result (a fitness plan)
carried real, citable values but no metric identity, so none became facts and
the answer degraded. (2) The engine's own steering note ("... none of it
cleared the bar ... Do not say the vault has no data ... a follow-up tool such
as summarize_metric ...") was repeated to a user as narration and every gate
passed it.

Both are pinned here at the seams that matter: the generic
``publishable_facts`` convention (the engine attaches no meaning to it), and a
deterministic post-check that sends any answer repeating model-facing
instructions to the fallback.
"""
from __future__ import annotations

import pytest

from health_advisor import chat, fact_template, llm, steering

QUESTION = "What does my plan have me doing this week?"

# The exact note the live model repeated, as the pre-#557 engine built it.
OLD_STEERING_NOTE = (
    "This turn's tools (get_plan_overview, get_week_plan) returned real vault "
    "data, but none of it cleared the bar to publish as a citable figure. Do "
    "not say the vault has no data or no history -- it has data. Say that a "
    "specific number needs a follow-up tool such as summarize_metric, "
    "get_weekly_series, list_workouts."
)
FORBIDDEN = ("cleared the bar", "Do not say", "do not say", "get_week_plan",
             "get_plan_overview", "summarize_metric", "get_weekly_series",
             "list_workouts")


def _record(sequence, tool, result):
    return {"sequence": sequence, "tool_name": tool, "result": result,
            "result_elided": False}


def _plan_result():
    return {
        "week_title": "Plan week",
        "days": [{"date": "2026-08-31", "weekday": "Mon"}],
        DECLARED: [
            {"id": "wk/2026-08-31/2026-08-31/title",
             "label": "Session title on Monday, August 31",
             "value": "Easy run"},
            {"id": "wk/2026-08-31/2026-08-31/jog_minutes",
             "label": "Jog minutes planned on Monday, August 31",
             "value": 25, "unit": "min"},
        ],
    }


DECLARED = fact_template.DECLARED_FACTS_RESULT_KEY


# --- the generic convention -------------------------------------------------

def test_declared_facts_publish_string_and_number_leaves():
    facts = fact_template.build_declared_facts(
        [_record(3, "any_host_tool", _plan_result())])
    title = facts[fact_template.declared_fact_key(
        "wk/2026-08-31/2026-08-31/title")]
    minutes = facts[fact_template.declared_fact_key(
        "wk/2026-08-31/2026-08-31/jog_minutes")]
    assert title["display"] == "Easy run" and title["value"] == "Easy run"
    assert minutes["display"] == "25" and minutes["unit"] == "min"
    assert minutes["value"] == 25
    assert minutes["label"].startswith("Jog minutes planned")
    assert minutes["source"] == {
        "sequence": 3, "path": "$.result.publishable_facts[1].value"}
    # No metric identity is invented for a declared fact.
    assert "metric" not in minutes and "period" not in minutes


def test_declared_facts_engine_knows_no_tool_names():
    """Any tool name works: the convention keys on the result, not the tool."""
    a = fact_template.build_declared_facts(
        [_record(1, "one_tool", _plan_result())])
    b = fact_template.build_declared_facts(
        [_record(1, "another_tool", _plan_result())])
    assert a == b and len(a) == 2


@pytest.mark.parametrize("entry", [
    "not a dict",
    {"id": "", "label": "x", "value": 1},
    {"id": "a", "label": "", "value": 1},
    {"id": "a", "label": "x", "value": None},
    {"id": "a", "label": "x", "value": True},
    {"id": "a", "label": "x", "value": float("nan")},
    {"id": "a", "label": "x", "value": float("inf")},
    {"id": "a", "label": "x", "value": "  "},
    {"id": "a", "label": "x", "value": "v" * 201},
    {"id": "a", "label": "x", "value": [1]},
    {"id": "a", "label": "x", "value": 1, "unit": 5},
    {"id": "a\nb", "label": "x", "value": 1},
])
def test_declared_facts_skip_malformed_entries_without_repairing(entry):
    ledger = [_record(1, "t", {DECLARED: [entry]})]
    assert fact_template.build_declared_facts(ledger) == {}


def test_declared_facts_conflicting_duplicates_are_withheld_agreeing_kept():
    def rec(seq, value):
        return _record(seq, "t", {DECLARED: [
            {"id": "same", "label": "x", "value": value}]})
    key = fact_template.declared_fact_key("same")
    assert key in fact_template.build_declared_facts([rec(1, 5), rec(2, 5)])
    assert fact_template.build_declared_facts([rec(1, 5), rec(2, 6)]) == {}


def test_declared_facts_read_only_a_non_elided_results_top_level_list():
    nested = {"inner": {DECLARED: [{"id": "n", "label": "x", "value": 1}]}}
    elided = {**_record(1, "t", _plan_result()), "result_elided": True}
    args_only = {"sequence": 2, "tool_name": "t", "result": {},
                 "arguments": {DECLARED: [
                     {"id": "a", "label": "x", "value": 1}]},
                 "result_elided": False}
    assert fact_template.build_declared_facts(
        [_record(3, "t", nested), elided, args_only]) == {}


def test_declared_facts_guidance_only_when_present():
    facts = fact_template.build_declared_facts(
        [_record(1, "t", _plan_result())])
    assert "LABELLED FACTS" in fact_template.declared_facts_guidance(facts)
    assert fact_template.declared_facts_guidance({}) == ""


# --- declared facts through the whole ask path ------------------------------

def _run_template_turn(monkeypatch, vault, tmp_path, ledger, responses):
    it = iter(responses)
    prompts = []

    def fake_loop(prompt, *args, **kwargs):
        prompts.append(prompt)
        return next(it)

    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_loop", fake_loop)
    result = chat._answer_fact_template(
        vault, QUESTION, "gather", [], str(tmp_path / "ledger.jsonl"))
    return result, prompts


def test_a_plan_answer_interpolates_declared_title_and_minutes(
        vault, tmp_path, monkeypatch):
    title = fact_template.declared_fact_key("wk/2026-08-31/2026-08-31/title")
    mins = fact_template.declared_fact_key(
        "wk/2026-08-31/2026-08-31/jog_minutes")
    template = (f"On Monday you have {{{title}}}: a jog of {{{mins}}} minutes.")
    result, prompts = _run_template_turn(
        monkeypatch, vault, tmp_path, [_record(1, "any_host_tool",
                                               _plan_result())],
        ["ack", template])
    assert result["mode"] == "narration", result["verification"]
    assert result["text"] == "On Monday you have Easy run: a jog of 25 minutes."
    assert result["verification"]["ok"] and result["verification"]["cause"] == "ok"
    assert [f["display"] for f in result["figures"]] == ["Easy run", "25"]
    # Declared facts suppress the "nothing cleared the bar" status entirely.
    assert "status:gathered_data_uncited" not in prompts[-1]
    assert "LABELLED FACTS" in prompts[-1]


def test_a_digit_in_prose_beside_a_declared_fact_is_still_refused(
        vault, tmp_path, monkeypatch):
    title = fact_template.declared_fact_key("wk/2026-08-31/2026-08-31/title")
    bad = f"{{{title}}} for 25 minutes."
    result, _ = _run_template_turn(
        monkeypatch, vault, tmp_path,
        [_record(1, "any_host_tool", _plan_result())], ["ack", bad, bad])
    assert result["mode"] == "fallback"


# --- (2) the leak -----------------------------------------------------------

def _no_figure_ledger():
    # Real data, nothing publishable: the shape that used to trigger the note.
    return [_record(1, "get_week_plan", {"found": True, "days": [
        {"date": "2026-08-31", "weekday": "Mon", "title": "Easy run"}]})]


def _assert_no_steering(text):
    for phrase in FORBIDDEN:
        assert phrase not in text, phrase
    assert steering.leak(text) is None


def test_verbatim_steering_note_from_the_model_goes_to_fallback(
        vault, tmp_path, monkeypatch):
    result, _ = _run_template_turn(
        monkeypatch, vault, tmp_path, _no_figure_ledger(),
        ["ack", OLD_STEERING_NOTE, OLD_STEERING_NOTE])
    assert result["mode"] == "fallback"
    assert result["verification"]["ok"] is False
    assert result["verification"]["reason"] == steering.REASON
    _assert_no_steering(result["text"])


def test_a_partly_paraphrased_note_still_goes_to_fallback(
        vault, tmp_path, monkeypatch):
    echoed = ("Here is what I found. Do not say the vault has no data -- "
              "it has data, but a specific number needs a follow-up.")
    result, _ = _run_template_turn(
        monkeypatch, vault, tmp_path, _no_figure_ledger(),
        ["ack", echoed, echoed])
    assert result["mode"] == "fallback"
    _assert_no_steering(result["text"])


def test_a_leaking_first_draft_gets_one_repair_and_a_clean_retry_publishes(
        conn, vault, tmp_path, monkeypatch):
    result, _ = _run_template_turn(
        monkeypatch, vault, tmp_path, _no_figure_ledger(),
        ["ack", OLD_STEERING_NOTE,
         "Here is where things stand. {status:gathered_data_uncited}"])
    assert result["mode"] == "narration"
    assert result["verification"].get("retry") is True
    _assert_no_steering(result["text"])


def test_the_status_fact_a_model_quotes_is_a_user_sentence(
        vault, tmp_path, monkeypatch):
    """The sanctioned path: the model uses the status slot; the user reads a
    sentence that is safe by construction, so it is not flagged."""
    slot = "{status:gathered_data_uncited}"
    result, prompts = _run_template_turn(
        monkeypatch, vault, tmp_path, _no_figure_ledger(),
        ["ack", f"Here is where things stand. {slot}"])
    assert "status:gathered_data_uncited" in prompts[-1]
    assert result["mode"] == "narration"
    assert "Your data is on file" in result["text"]
    _assert_no_steering(result["text"])


def test_a_leak_in_a_conversational_reply_goes_to_fallback(
        vault, tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda *a, **k: "Python will validate and may publish that reply "
                        "directly, thanks!")
    result = chat._answer_fact_template(
        vault, "thanks!", "gather", [], str(tmp_path / "ledger.jsonl"))
    assert result["mode"] != "narration" or steering.leak(result["text"]) is None
    assert steering.leak(result["text"]) is None


def test_backstop_replaces_any_leaking_narration_with_the_fallback(
        vault, monkeypatch):
    """The other arms (and any future path) are covered at the exit."""
    monkeypatch.setattr(chat, "_record_question", lambda *a, **k: None)
    monkeypatch.setattr(chat, "_answer_question_inner", lambda *a, **k: {
        "text": OLD_STEERING_NOTE, "mode": "narration", "tool_trace": [],
        "verification": {"ok": True, "grounded": True, "cause": "ok"},
        "figures": [{"key": "k"}]})
    result = chat.answer_question(vault, QUESTION)
    assert result["mode"] == "fallback"
    assert result["verification"]["ok"] is False
    assert "figures" not in result
    _assert_no_steering(result["text"])


def test_backstop_leaves_clean_narration_alone(vault, monkeypatch):
    monkeypatch.setattr(chat, "_record_question", lambda *a, **k: None)
    payload = {"text": "Your Monday run is easy, then a rest day.",
               "mode": "narration", "tool_trace": [],
               "verification": {"ok": True, "cause": "ok"}}
    monkeypatch.setattr(chat, "_answer_question_inner",
                        lambda *a, **k: dict(payload))
    assert chat.answer_question(vault, QUESTION) == payload


# --- the detector itself ----------------------------------------------------

@pytest.mark.parametrize("text", [
    "You have an easy run on Monday.",
    "Rest on Tuesday, then a longer jog on Thursday.",
    "Your plan is on file, but I have no specific figure to quote.",
    "Try a cite-worthy source next time.",  # 'cite' is ordinary English
    # A date range is a run of numbers that the prompt's own example also
    # contains; an honest answer about that week must not be flagged for it.
    "Last week, 2026-08-10:2026-08-16, you averaged more sleep per night.",
    "Weekly running volume: 2026-08-10 to 2026-08-16 versus 2026-08-03 to "
    "2026-08-09.",
    "",
])
def test_ordinary_answers_are_not_flagged(text):
    assert steering.leak(text) is None


@pytest.mark.parametrize("text", [
    "I used get_week_plan to read it.",
    "The GET_WEEK_PLAN tool said so.",
    "It cleared the bar, so I cite it.",
    "Do not say that.",
    "See {fact|metric=jog_minutes|period=x|field=mean}.",
    "See {pub:wk/x} for details.",
    "The closed fact set has nothing.",
    "status:gathered_data_uncited",
    "out_of_corpus_domain",
])
def test_markers_and_tool_names_are_flagged(text):
    assert steering.leak(text) is not None


def test_every_registered_instruction_is_flagged_when_repeated():
    """The registry is the guard: any run of six words from ANY registered
    steering text is caught, wherever it sits in the answer."""
    texts = steering.registered_texts()
    assert len(texts) >= 15
    for text in texts:
        words = text.split()
        starts = [i for i in range(len(words) - steering.SHINGLE_WORDS + 1)
                  if tuple(words[i:i + steering.SHINGLE_WORDS])
                  in steering._TEXTS[text]]
        assert starts, f"no checkable run in a registered text: {text[:60]}"
        start = starts[len(starts) // 2]
        snippet = " ".join(words[start:start + 8])
        answer = f"Sure, your week looks fine. {snippet} Have a good run."
        assert steering.leak(answer) is not None, snippet


def test_no_instruction_the_ask_path_sends_names_a_tool_a_user_could_read():
    """The status sentence, the one steering-shaped text a user DOES read, is
    checked against the whole mechanism rather than a phrase list."""
    facts = fact_template.build_gathered_data_status_fact(
        ledger_has_data=True, figures_published=False,
        tool_names=llm.COACH_TOOLS)
    (fact,) = facts.values()
    assert steering.leak(fact["display"]) is None
    for name in llm.COACH_TOOLS:
        assert name not in fact["display"]
