"""Conversation identity, durability, isolation, and append-only history."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from health_advisor import chat
from health_advisor import db as dbmod
from health_advisor import deepdive_verify as DV
from health_advisor import fact_template
from health_advisor import llm
from health_advisor.context import VaultContext, VaultOwnershipError
from tests.conftest import seed_metric


def test_first_question_prompt_is_unchanged_without_history(monkeypatch, vault,
                                                            conn):
    prompts = []
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda prompt, **kwargs: prompts.append(prompt) or "")

    chat.answer_question(vault, "How am I doing?")

    expected = (
        "You are the user's personal health coach. Answer the user's question "
        "directly and honestly using the supplied read-only health tools. Call "
        "the most relevant tool(s), check their scope and caveats, and do not "
        "invent a number, metric, activity, or period. If the data cannot answer "
        "the question, say so without guessing.\n\nUSER QUESTION:\n"
        "How am I doing?\n\n" + chat.ASK_CLAIM_INSTRUCTIONS
    )
    assert prompts[0] == expected


def test_fact_template_flag_zero_keeps_the_existing_prose_path(
        monkeypatch, vault, conn):
    """An explicit flag-off call still sends the legacy claim prompt/loop."""
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "0")
    prompts = []
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda prompt, **kwargs: prompts.append(prompt) or "")

    result = chat.answer_question(vault, "How am I doing?")

    assert result["mode"] == "fallback"
    assert len(prompts) == 2
    assert chat.ASK_CLAIM_INSTRUCTIONS in prompts[0]
    assert "CLOSED FACT SET" not in prompts[0]


def test_fact_template_flag_runs_gather_then_closed_set_narration(
        monkeypatch, vault):
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
    calls = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: calls.append((prompt, kwargs)) or next(responses))

    result = chat.answer_question(vault, "How was my run?")

    assert result["mode"] == "narration"
    assert result["text"] == "Your run was 50 m."
    assert result["verification"]["template_compliant"] is True
    assert result["verification"]["narration_counts_comparable"] is False
    assert len(calls) == 2
    assert key in calls[1][0]
    assert "VO2" in calls[1][0]
    assert "last 4 weeks" in calls[1][0]
    assert "ISO dates" in calls[1][0]
    assert ("When a date or period name belongs in prose, use "
            "{fact|metric=...|period=...|field=period_label}") in calls[1][0]
    assert ("Activity for {fact|metric=jog_minutes|period=s:2026-08-10:2026-08-16|"
            "field=period_label}.") in calls[1][0]
    assert calls[0][1]["tools"] == []
    assert calls[1][1]["tools"] == []


def test_period_label_fact_is_excluded_from_figures_total(monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    period = "2026-08-10:2026-08-16"
    ledger = [{
        "sequence": 1, "tool_name": "synthetic_metric", "arguments": {},
        "result": {
            "metric": "jog_minutes", "unit": "min", "period": period,
            "mean": 50.1,
            "presentation": {"metric": "jog_minutes", "period": period,
                              "field": "presentation", "value": "50 m"},
        },
    }]
    value_key = fact_template.fact_key("jog_minutes", period, "mean")
    label_key = fact_template.fact_key("jog_minutes", period, "period_label")
    responses = iter([
        "acknowledged",
        "During {" + label_key + "}, your run was {" + value_key + "}.",
    ])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    result = chat.answer_question(vault, "How was my run?")

    assert result["mode"] == "narration"
    assert result["text"] == "During the week of August 10, your run was 50 m."
    assert result["verification"]["figures_total"] == 1
    assert result["verification"]["figures_verified"] == 1


def test_fact_template_advice_quantities_reach_public_verification(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = _analyst_attachment_ledger()
    responses = iter([
        "acknowledged",
        "Add {advice:3 sets of 10 reps} after your run.",
    ])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    result = chat.answer_question(vault, "What should I add to my routine?")

    assert result["mode"] == "narration"
    assert result["text"] == "Add 3 sets of 10 reps after your run."
    assert result["verification"]["figures_verified"] == 0
    assert result["verification"]["figures_total"] == 0
    assert result["verification"]["advice_quantities"] == [
        "3 sets of 10 reps"
    ]
    # Advice spans are substance: the empty-narration retry must NOT fire.
    assert "retry" not in result["verification"]


def test_zero_figure_template_with_facts_retries_and_names_unused_facts(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = _analyst_attachment_ledger()
    cell_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-02")
    responses = iter([
        "acknowledged",
        "I found the relevant record.",
        "Your rate is {" + cell_key + "}.",
    ])
    calls = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: calls.append(prompt) or next(responses))

    capture = []
    result = chat.answer_question(vault, "What is my resting heart rate?",
                                  capture=capture)

    assert result["mode"] == "narration"
    assert result["verification"]["retry"] is True
    assert [entry["attempt"] for entry in capture] == [1, 2]
    assert "UNUSED FACT NAMES" in calls[2]
    assert cell_key in calls[2]


def test_still_empty_narration_for_present_metric_falls_back(
        monkeypatch, vault, conn):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    seed_metric(conn, "step_count", "2026-08-08", list(range(14)))
    ledger = _analyst_attachment_ledger()
    responses = iter([
        "acknowledged",
        "I don't have data for step count.",
        "I don't have data for step count.",
    ])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    capture = []
    result = chat.answer_question(
        vault, "What are my steps?", as_of="2026-08-21", capture=capture)

    assert result["mode"] == "fallback"
    assert result["verification"]["reason"] == (
        "empty narration names a metric whose coverage is not missing")
    assert [entry["attempt"] for entry in capture] == [1, 2]
    assert len(capture) <= 2


def _available_jog_ledger():
    return [{
        "sequence": 1,
        "tool_name": "get_impact_volume",
        "arguments": {"start": "2026-08-17", "end": "2026-08-23",
                       "by": "week"},
        "result": {"metric": "jog_minutes",
                    "period": "2026-08-17:2026-08-23",
                    "jog_minutes": 12.5, "unit": "min"},
    }]


def test_denial_of_available_figure_is_refused_and_retry_gets_fact_keys(
        monkeypatch, vault):
    """A digit-free denial cannot pass over a ledger fact."""
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = _available_jog_ledger()
    period = ledger[0]["result"]["period"]
    key = fact_template.fact_key("jog_minutes", period, "jog_minutes")
    denial = "I don't have a recorded value for your jog minutes this week."
    responses = iter(["acknowledged", denial, denial])
    calls = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: calls.append(prompt) or next(responses))

    capture = []
    result = chat.answer_question(
        vault, "How many jog minutes did I do this week?",
        as_of="2026-08-23", capture=capture)

    assert result["mode"] == "fallback"
    assert result["verification"]["cause"] == "denied_available_figure"
    assert result["verification"]["reason"] == (
        "narration denied an available figure")
    assert capture[0]["verification"]["cause"] == "denied_available_figure"
    assert capture[1]["verification"]["cause"] == "denied_available_figure"
    assert "AVAILABLE FACT KEYS FOR THE ASKED METRIC" in calls[2]
    assert key in calls[2]


def test_denial_without_a_returned_value_remains_a_valid_empty_answer(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = [{
        **_available_jog_ledger()[0],
        "result": {"metric": "jog_minutes",
                    "period": "2026-08-17:2026-08-23",
                    "jog_minutes": None, "unit": "min"},
    }]
    denial = "I don't have a recorded value for your jog minutes this week."
    responses = iter(["acknowledged", denial])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    result = chat.answer_question(
        vault, "How many jog minutes did I do this week?",
        as_of="2026-08-23")

    assert result["mode"] == "narration"
    assert result["verification"]["cause"] == "ok"


@pytest.mark.parametrize("kind,text,expected", [
    ("prose", "I don't have a recorded value for your jog minutes this week.", True),
    ("prose", "You logged 12.5 jog minutes this week. This does not include walk breaks.", False),
    ("prose", "12.5 jog minutes this week. Your plan does not include a fourth run.", False),
    ("prose", "You logged 12.5 jog minutes this week.", False),
    ("template", "You logged {KEY} this week; this does not include walk breaks.", False),
])
def test_denial_phrase_requires_no_asked_metric_figure(kind, text, expected):
    """Denial vocabulary cannot override a figure carried by the answer."""
    ledger = _available_jog_ledger()
    period = ledger[0]["result"]["period"]
    key = fact_template.fact_key("jog_minutes", period, "jog_minutes")
    text = text.replace("{KEY}", "{" + key + "}")
    facts = fact_template.build_fact_set(ledger)
    question = "How many jog minutes did I do this week?"
    if kind == "template":
        carries_figure = chat._template_has_asked_metric_figure(
            question, text, facts)
    else:
        carries_figure = chat._prose_has_asked_metric_figure(
            "jog_minutes",
            ([{"metric": "jog_minutes", "period": period,
               "field": "jog_minutes", "value": 12.5,
               "source": {"sequence": 1, "path": "$.result.jog_minutes"}}]
             if "12.5" in text else []),
            ({"verdict": {"numbers": [{"ok": True,
                                         "metric": "jog_minutes",
                                         "field": "jog_minutes"}]}}
             if "12.5" in text else {}))
    assert chat._denied_available_figure(
        question, text, ledger,
        answer_has_asked_metric_figure=carries_figure,
        facts=facts) is expected


@pytest.mark.parametrize("text,empty_expected,denial_expected", [
    ("I don't have your jog minutes for this week.", False, True),
    ("I don't have a recorded figure for your jog minutes this week.",
     False, True),
    ("I don't have any jog minute data for this week, so I can't provide "
     "that figure.", True, False),
    ("Jog minute data is unavailable for this week, so I cannot provide "
     "that figure.", True, False),
    ("I don't have a record of your jog minutes for this week, so I can't "
     "provide that figure.", True, False),
    ("I don't have a recorded value for the jog minutes metric family for "
     "this week.", False, True),
])
def test_measured_denial_wordings_keep_the_regex_split(
        text, empty_expected, denial_expected):
    """Measured denial prose stays on its intended closed-list gate."""
    assert bool(chat._EMPTY_NARRATION_RE.search(text)) is empty_expected
    assert bool(chat._AVAILABLE_FIGURE_DENIAL_RE.search(text)) is denial_expected


def test_genuinely_empty_gather_allows_empty_narration_without_retry(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = [{
        "sequence": 1, "tool_name": "list_workouts", "arguments": {},
        "result": {"count": 0, "workouts": [], "note": "no data"},
        "result_elided": False,
    }]
    responses = iter([
        "acknowledged",
        "I don't have data for this question.",
    ])
    calls = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: calls.append(prompt) or next(responses))

    capture = []
    result = chat.answer_question(vault, "What was my longest run?",
                                  capture=capture)

    assert result["mode"] == "narration"
    assert result["text"] == "I don't have data for this question."
    assert [entry["attempt"] for entry in capture] == [1]
    assert len(calls) == 2


def test_fact_template_repair_budget_is_hard_limited_to_two_attempts(
        monkeypatch, vault, conn):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    seed_metric(conn, "step_count", "2026-08-08", list(range(14)))
    ledger = _analyst_attachment_ledger()
    responses = iter([
        "acknowledged",
        "I don't have data for step count.",
        "I don't have data for step count.",
        "this must never be requested",
    ])
    calls = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: calls.append(prompt) or next(responses))

    capture = []
    result = chat.answer_question(vault, "What are my steps?", capture=capture)

    assert result["mode"] == "fallback"
    assert len(capture) == 2
    assert [entry["attempt"] for entry in capture] == [1, 2]
    # One gather call plus one initial template and one repair template.
    assert len(calls) == 3


def _analyst_attachment_ledger():
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


def test_fact_template_analyst_attachment_narration_uses_cells_and_direction(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = _analyst_attachment_ledger()
    facts = fact_template.build_attachment_facts(ledger)
    first_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-01")
    last_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-02")
    direction_key = fact_template.attachment_trend_key(
        "resting_rate", "rate", "direction")
    delta_key = fact_template.attachment_trend_key(
        "resting_rate", "rate", "delta")
    template = (f"Your resting heart rate has {{{direction_key}}} slightly, "
                f"from {{{first_key}}} to {{{last_key}}}.")
    responses = iter(["acknowledged", template])
    calls = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: calls.append((prompt, kwargs))
        or next(responses))

    result = chat.answer_question(vault, "How has my resting heart rate changed?")

    direction = facts[direction_key]["display"]
    assert result["mode"] == "narration"
    assert "63" in result["text"] and "60" in result["text"]
    assert direction in result["text"]
    assert facts[delta_key]["value"] == 60 - 63
    assert direction == facts[direction_key]["value"]
    # The direction token changes with the sign, without baking an English
    # direction word into this test's assertion.
    positive = fact_template.build_attachment_facts([{
        **ledger[0],
        "result": {"tables": [{
            **ledger[0]["result"]["tables"][0],
            "rows": [["2026-08-01", 60], ["2026-08-02", 63]],
        }]},
    }])
    unchanged = fact_template.build_attachment_facts([{
        **ledger[0],
        "result": {"tables": [{
            **ledger[0]["result"]["tables"][0],
            "rows": [["2026-08-01", 60], ["2026-08-02", 60]],
        }]},
    }])
    assert facts[delta_key]["value"] < 0
    assert direction != positive[direction_key]["display"]
    assert direction != unchanged[direction_key]["display"]
    assert len(calls) == 2


def test_fact_template_repair_prompt_carries_exact_digit_refusal(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = _analyst_attachment_ledger()
    cell_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-02")
    responses = iter([
        "acknowledged",
        "Your rate is 60 today.",
        "Your rate is {" + cell_key + "}.",
    ])
    calls = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: calls.append((prompt, kwargs))
        or next(responses))

    capture = []
    result = chat.answer_question(
        vault, "What is my resting heart rate today?", capture=capture)

    assert result["mode"] == "narration"
    assert result["text"] == "Your rate is 60."
    assert [entry["attempt"] for entry in capture] == [1, 2]
    assert capture[0]["verification"]["reason"] == (
        "digit outside placeholder")
    assert capture[1]["verification"]["ok"] is True
    repair = calls[2][0]
    assert "FAILING TEMPLATE:\nYour rate is 60 today." in repair
    assert ("EXACT GATE REFUSAL:\n"
            "digit outside placeholder; offending span: '60'" in repair)
    assert "Fix only this reported issue" in repair
    assert cell_key in repair


def test_fact_template_repair_prompt_carries_exact_unresolved_key(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = _analyst_attachment_ledger()
    cell_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-02")
    unresolved_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-03")
    responses = iter([
        "acknowledged",
        "Your rate is {" + unresolved_key + "}.",
        "Your rate is {" + cell_key + "}.",
    ])
    calls = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: calls.append((prompt, kwargs))
        or next(responses))

    capture = []
    result = chat.answer_question(
        vault, "What is my resting heart rate today?", capture=capture)

    assert result["mode"] == "narration"
    assert ("unresolvable placeholder; unresolved placeholder key: '"
            + unresolved_key + "'") in calls[2][0]


def test_fact_template_double_rejection_is_one_retry_then_same_fallback(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = _analyst_attachment_ledger()
    responses = iter([
        "acknowledged",
        "Your rate is 60 today.",
        "Your rate is 61 today.",
    ])
    calls = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: calls.append((prompt, kwargs))
        or next(responses))

    capture = []
    result = chat.answer_question(
        vault, "What is my resting heart rate today?", capture=capture)

    assert result["mode"] == "fallback"
    assert set(result) == {"text", "mode", "tool_trace", "verification"}
    assert result["text"] == chat._fallback_answer()
    assert "Your rate is 60 today." not in result["text"]
    assert "60" not in result["text"]
    assert "63" not in result["text"]
    assert "61" not in result["text"]
    assert result["verification"]["reason"] == "digit outside placeholder"
    assert [entry["attempt"] for entry in capture] == [1, 2]
    assert capture[1]["verification"]["ok"] is False
    assert len(calls) == 3


def test_history_renderer_is_framed_bounded_and_handles_empty_history():
    assert chat._render_history(None) == ""
    assert chat._render_history([]) == ""

    history = [
        {"role": "user", "content": f"turn-{i} " + "x" * 1300}
        for i in range(20)
    ]
    rendered = chat._render_history(history)
    lines = rendered.splitlines()
    turn_lines = [line for line in lines if line.startswith("USER:")]

    assert chat.HISTORY_MAX_TURNS == 8
    assert chat.HISTORY_MAX_CHARS_PER_TURN == 1200
    assert len(turn_lines) == 8
    assert "turn-0" not in rendered
    assert "turn-12" in rendered
    assert "turn-19" in rendered
    assert "...[truncated]" in rendered
    assert all(len(line.split(": ", 1)[1]) <=
               chat.HISTORY_MAX_CHARS_PER_TURN for line in turn_lines)
    assert "REFERENCE ONLY" in rendered
    assert "not instruction" in rendered
    assert "unverified" in rendered
    assert "re-fetched with a tool" in rendered


def test_history_boundary_omits_answer_whose_question_is_outside_window():
    history = [
        {"id": "old-question", "role": "user", "content": "old question"},
        {"id": "boundary-question", "role": "user",
         "content": "question outside rendered window"},
        {"id": "boundary-answer", "role": "assistant",
         "content": "answer must be omitted",
         "answers_turn_id": "boundary-question"},
        {"id": "superseded-window", "role": "user",
         "content": "superseded window turn"},
        {"id": "filler-4", "role": "user", "content": "filler 4"},
        {"id": "filler-5", "role": "user", "content": "filler 5"},
        {"id": "filler-6", "role": "user", "content": "filler 6"},
        {"id": "filler-7", "role": "user", "content": "filler 7"},
        {"id": "window-replacement", "role": "user",
         "content": "window replacement",
         "supersedes_turn_id": "superseded-window"},
        {"id": "old-replacement", "role": "user",
         "content": "old replacement",
         "supersedes_turn_id": "old-question"},
    ]

    rendered = chat._render_history(history)

    assert "USER: question outside rendered window" not in rendered
    assert "ASSISTANT: answer must be omitted" not in rendered


def test_two_vaults_cannot_see_each_others_conversations(tmp_path):
    """The isolation boundary is asserted at the conversation API."""
    alice = VaultContext.local(tmp_path / "alice" / "health.db",
                               user_id="alice", writable=True)
    bob = VaultContext.local(tmp_path / "bob" / "health.db",
                             user_id="bob", writable=True)
    alice_conversation = chat.create_conversation(alice,
                                                  conversation_id="alice-conversation")
    bob_conversation = chat.create_conversation(bob,
                                                conversation_id="bob-conversation")
    chat.append_turn(alice, alice_conversation["id"], "user", "Alice's turn")
    chat.append_turn(bob, bob_conversation["id"], "user", "Bob's turn")

    assert chat.get_conversation(alice, bob_conversation["id"]) is None
    assert chat.get_conversation(bob, alice_conversation["id"]) is None
    assert [c["id"] for c in chat.list_conversations(alice)] == ["alice-conversation"]
    assert [c["id"] for c in chat.list_conversations(bob)] == ["bob-conversation"]


def test_ownership_boundary_on_one_file_allows_unclaimed_refuses_claimed(tmp_path):
    """Unclaimed files are open; claiming makes a different session unwelcome."""
    path = tmp_path / "shared" / "health.db"
    alice = VaultContext.local(path, user_id="alice", writable=True)
    bob = VaultContext.local(path, user_id="bob")

    chat.create_conversation(alice, conversation_id="secret")
    assert [c["id"] for c in chat.list_conversations(bob)] == ["secret"]

    alice.claim()
    with pytest.raises(VaultOwnershipError):
        bob.read_only()


def test_chat_reads_empty_vault_without_conversation_tables(tmp_path):
    """Conversation reads tolerate a vault created before this feature."""
    path = tmp_path / "older" / "health.db"
    conn = dbmod.connect(path)
    conn.execute("CREATE TABLE legacy (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    ctx = VaultContext.local(path, writable=True)

    assert chat.list_conversations(ctx) == []
    assert chat.get_conversation(ctx, "missing") is None
    assert chat.list_turns(ctx, "missing") == []
    with pytest.raises(KeyError, match=r"unknown conversation: missing"):
        chat.append_turn(ctx, "missing", "user", "hello")


def test_conversation_survives_process_restart(tmp_path):
    """A new process can reopen the same vault and recover ordered turns."""
    path = tmp_path / "restart" / "health.db"
    script = """
import json
import sys
from health_advisor import chat
from health_advisor.context import VaultContext

ctx = VaultContext.local(sys.argv[1], user_id="restart", writable=True)
conversation = chat.create_conversation(ctx, conversation_id="restart-conversation")
chat.append_turn(ctx, conversation["id"], "user", "first")
chat.append_turn(ctx, conversation["id"], "assistant", "second")
print(json.dumps({"id": conversation["id"]}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    conversation_id = json.loads(result.stdout)["id"]

    reopened = VaultContext.local(path, user_id="restart")
    turns = chat.list_turns(reopened, conversation_id)
    assert [(turn["sequence"], turn["content"]) for turn in turns] == [
        (1, "first"), (2, "second")
    ]


def test_turns_are_append_only_and_corrections_are_new_turns(conn, vault):
    """The old event remains readable when a later turn corrects it."""
    conversation = chat.create_conversation(vault, conversation_id="immutable")
    original = chat.append_turn(vault, conversation["id"], "assistant", "old answer")

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(
            "UPDATE conversation_turns SET content = 'mutated' WHERE id = ?",
            (original["id"],),
        )
    # SQLite's ABORT trigger refuses the statement but leaves the surrounding
    # transaction open; release that transaction before the next writer.
    conn.rollback()

    correction = chat.append_turn(
        vault,
        conversation["id"],
        "assistant",
        "corrected answer",
        supersedes_turn_id=original["id"],
    )
    turns = chat.list_turns(vault, conversation["id"])
    assert [turn["content"] for turn in turns] == ["old answer", "corrected answer"]
    assert correction["sequence"] == 2
    assert correction["supersedes_turn_id"] == original["id"]


def test_render_history_pairs_answers_excludes_superseded_and_undelivered_turns(
        vault):
    """The store keeps the log; the prompt receives only the current transcript."""
    conversation = chat.create_conversation(vault, conversation_id="fixture-fdf79fd3")
    q3 = chat.append_turn(
        vault, conversation["id"], "user",
        "How does this compare to the last couple weeks?",
    )
    q4 = chat.append_turn(
        vault, conversation["id"], "user",
        "How does this compare to the night before?",
    )
    a5 = chat.append_turn(
        vault, conversation["id"], "assistant",
        "Compared with the night before, it was 555.68 versus 343.32.",
        answers_turn_id=q4["id"],
    )
    a6 = chat.append_turn(
        vault, conversation["id"], "assistant",
        "The average was 453.29 across 12 recorded nights.",
        answers_turn_id=q3["id"],
    )
    q7 = chat.append_turn(
        vault, conversation["id"], "user",
        "How has my running volume been over the last two weeks?",
    )
    a8 = chat.append_turn(
        vault, conversation["id"], "assistant",
        "Fallback: I couldn't verify a grounded answer.",
        answers_turn_id=q7["id"],
        client_disconnected_at="2026-08-25T00:13:26+00:00",
    )

    stored = chat.list_turns(vault, conversation["id"])
    assert [turn["id"] for turn in stored] == [
        q3["id"], q4["id"], a5["id"], a6["id"], q7["id"], a8["id"]
    ]
    assert a5["answers_turn_id"] == q4["id"]
    assert a6["answers_turn_id"] == q3["id"]
    assert a8["client_disconnected_at"]

    rendered = chat._render_history(stored)
    assert rendered.splitlines()[2:7] == [
        "USER: How does this compare to the last couple weeks?",
        "ASSISTANT: The average was 453.29 across 12 recorded nights.",
        "USER: How does this compare to the night before?",
        "ASSISTANT: Compared with the night before, it was 555.68 versus 343.32.",
        "USER: How has my running volume been over the last two weeks?",
    ]
    assert "Fallback: I couldn't verify a grounded answer." not in rendered


def test_superseded_turn_stays_in_store_but_not_rendered(vault):
    conversation = chat.create_conversation(vault, conversation_id="superseded-render")
    old = chat.append_turn(vault, conversation["id"], "user", "old question")
    chat.append_turn(
        vault, conversation["id"], "user", "replacement question",
        supersedes_turn_id=old["id"],
    )

    stored = chat.list_turns(vault, conversation["id"])
    assert [turn["content"] for turn in stored] == [
        "old question", "replacement question"
    ]
    rendered = chat._render_history(stored)
    assert "USER: old question" not in rendered
    assert "USER: replacement question" in rendered


ASK_LEDGER_FIXTURE = (Path(__file__).parent
                      / "fixtures/jog_ledger_live_20260824_claims.jsonl")

# The response shape `/v1/ask` hands to clients verbatim. A key added to either
# set is a new public field, and an unverified draft's prose reaching one of
# them is the failure the capture channel exists to avoid.
ASK_RESULT_KEYS = {"text", "mode", "tool_trace", "verification"}
ASK_VERIFICATION_KEYS = {
    "ok", "grounded", "unsupported", "reason", "verdict", "structural_claims",
    "figures_verified", "figures_total", "tier_counts", "tier1_path_bound",
    "tier2_metric_recomputed", "tool_calls", "judge_score", "cause",
}
CAPTURE_KEYS = {"attempt", "prose", "claims", "verification", "judge_score",
                "ledger"}


def _verified_draft():
    """A draft whose single claim resolves against the recorded live ledger."""
    ledger = [json.loads(ASK_LEDGER_FIXTURE.read_text(encoding="utf-8"))]
    period = ledger[0]["result"]["block_comparison"]["blocks"]["recent"]["period"]
    draft = llm.ResearchResponse("Your recent jogging averaged 50.1 minutes.")
    draft.claims = [{
        "metric": "jog_minutes", "period": period, "field": "mean",
        "value": 50.1,
        "source": {"sequence": 1,
                   "path": "$.result.block_comparison.blocks.recent.mean"},
    }]
    return ledger, draft


def test_capture_records_the_single_attempt_of_a_first_pass_success(
        monkeypatch, vault, conn):
    ledger, draft = _verified_draft()
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: 100)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: draft)

    capture = []
    result = chat.answer_question(vault, "How is my jogging?", capture=capture)

    assert result["mode"] == "narration"
    assert len(capture) == 1
    entry = capture[0]
    assert set(entry) == CAPTURE_KEYS
    assert entry["attempt"] == 1
    assert entry["prose"] == "Your recent jogging averaged 50.1 minutes."
    assert entry["claims"] == draft.claims
    assert entry["judge_score"] == 100
    assert entry["ledger"] == ledger
    assert entry["verification"]["ok"] is True
    assert entry["verification"]["figures_verified"] == 1


def test_capture_records_the_failed_draft_and_the_retry(monkeypatch, vault, conn):
    ledger, draft = _verified_draft()
    drafts = iter([llm.ResearchResponse(""), draft])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: 100)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: next(drafts))

    capture = []
    result = chat.answer_question(vault, "How is my jogging?", capture=capture)

    assert result["verification"]["retry"] is True
    assert [entry["attempt"] for entry in capture] == [1, 2]
    assert all(set(entry) == CAPTURE_KEYS for entry in capture)
    assert capture[0]["prose"] == ""
    assert capture[0]["claims"] is None
    assert capture[1]["prose"] == "Your recent jogging averaged 50.1 minutes."
    assert capture[1]["claims"] == draft.claims
    assert capture[1]["judge_score"] == 100


def test_unrun_judge_is_none_in_capture_and_response(monkeypatch, vault, conn):
    verifications = iter([
        _failed_verification(figures_verified=0, figures_total=1,
                             reason="first gate failure"),
        _failed_verification(figures_verified=0, figures_total=1,
                             reason="retry gate failure"),
    ])
    drafts = iter([llm.ResearchResponse("first draft"),
                   llm.ResearchResponse("retry draft")])
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger",
                        lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))
    monkeypatch.setattr(chat, "_ask_judge",
                        lambda *args, **kwargs: pytest.fail("judge must not run"))

    capture = []
    result = chat.answer_question(vault, "What happened?", capture=capture)

    assert result["mode"] == "fallback"
    assert result["verification"]["judge_score"] is None
    assert [entry["judge_score"] for entry in capture] == [None, None]


def test_ask_cause_keeps_a_real_zero_distinct_from_no_judge():
    verification = {"ok": True}
    loop_outcomes = []
    assert chat._ask_cause(verification, ledger=[{"sequence": 1}],
                           loop_outcomes=loop_outcomes,
                           judge_score=None) == "ok"
    assert chat._ask_cause(verification, ledger=[{"sequence": 1}],
                           loop_outcomes=loop_outcomes,
                           judge_score=0) == "judge_refused"


def _failed_verification(*, figures_verified, figures_total, reason,
                         unsupported=None):
    return {
        "ok": False,
        "grounded": False,
        "unsupported": unsupported or [],
        "reason": reason,
        "figures_verified": figures_verified,
        "figures_total": figures_total,
    }


def test_retry_returns_the_more_verified_failed_attempt(monkeypatch, vault, conn):
    verifications = iter([
        _failed_verification(figures_verified=1, figures_total=2,
                             reason="attempt one retained a verified figure"),
        _failed_verification(figures_verified=0, figures_total=0,
                             reason="degenerate repeated-token generation"),
    ])
    drafts = iter([llm.ResearchResponse("first draft", [{}]),
                   llm.ResearchResponse("retry draft")])
    ledger = [{"sequence": 1}]
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))

    result = chat.answer_question(vault, "What did I do?", capture=[])

    assert result["mode"] == "fallback"
    assert result["verification"]["figures_verified"] == 1
    assert result["verification"]["reason"] == (
        "attempt one retained a verified figure")
    assert result["verification"]["retry"] is True


def test_retry_returns_attempt_two_when_python_verifies_more_figures(
        monkeypatch, vault, conn):
    verifications = iter([
        _failed_verification(figures_verified=0, figures_total=1,
                             reason="attempt one failed"),
        _failed_verification(figures_verified=1, figures_total=1,
                             reason="retry retained a verified figure"),
    ])
    drafts = iter([llm.ResearchResponse("first draft"),
                   llm.ResearchResponse("retry draft", [{}])])
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))

    result = chat.answer_question(vault, "What did I do?")

    assert result["verification"]["figures_verified"] == 1
    assert result["verification"]["reason"] == (
        "retry retained a verified figure")


def _partial_failure(*, unsupported=("26",)):
    return {
        "ok": False,
        "grounded": False,
        "unsupported": list(unsupported),
        "reason": "one claim failed",
        "verdict": {"numbers": [
            {"ok": True, "claimed": 50.1},
            {"ok": False, "claimed": 26, "reason": "wrong value"},
        ]},
        "figures_verified": 1,
        "figures_total": 2,
    }


def _verified_rewrite():
    return {
        "ok": True,
        "grounded": True,
        "unsupported": [],
        "reason": "",
        "verdict": {"numbers": [{"ok": True, "claimed": 50.1}]},
        "figures_verified": 1,
        "figures_total": 1,
    }


def test_regeneration_redaction_preserves_date_shapes_but_strips_figures():
    text = (
        "Ran on July 26 for 3.04 mi, on 18 Aug for 36.1 min, and on "
        "August 18th with 144 bpm. 2026-08-21T09:30:00-04:00 is also "
        "a date, as are Tuesday 08-18 and August 2026."
    )

    redacted = chat._redact_regeneration_figures(text)

    assert "July 26" in redacted
    assert "18 Aug" in redacted
    assert "August 18th" in redacted
    assert "2026-08-21" in redacted
    assert "2026-08-21T09:30:00-04:00" in redacted
    assert "Tuesday 08-18" in redacted
    assert "August 2026" in redacted
    assert "3.04" not in redacted
    assert "36.1" not in redacted
    assert "144" not in redacted


def test_regeneration_prompt_forbids_new_dates():
    prompt = chat._span_regeneration_prompt(
        "What was my longest run last month?", "It was on July 26 for 3.04 mi.",
        [{"value": 3.04}],
    )

    assert "Copy dates exactly from the redacted question or draft" in prompt
    assert "do not state any date that is not present" in prompt
    assert "never invent a date" in prompt
    assert "July 26" in prompt
    draft = prompt.split("DRAFT (figures removed):", 1)[1].split(
        "PYTHON-VERIFIED CLAIMS:", 1)[0]
    assert "3.04" not in draft


def test_unverified_figure_scan_allows_failed_day_inside_preserved_date():
    verification = {
        "unsupported": ["26"],
        "verdict": {"numbers": [{"ok": False, "claimed": 26}]},
    }

    assert not chat._contains_unverified_figure(
        "Your longest run was on July 26.", verification)
    assert chat._contains_unverified_figure(
        "Your longest run was on July 26 and lasted 26 minutes.",
        verification)


def test_weekday_beside_a_bare_figure_is_not_a_date_span():
    """'Tuesday, 26' must not read as a date: the bare-day fallback let a
    failed figure beside a weekday word slip the rescan (caught at review,
    2026-08-29). A weekday needs a month or a numeric date to count."""
    verification = {
        "unsupported": ["26"],
        "verdict": {"numbers": [{"ok": False, "claimed": 26}]},
    }

    assert chat._contains_unverified_figure(
        "Monday, 15 minutes; Tuesday, 26 minutes were logged.", verification)
    assert chat._contains_unverified_figure(
        "You logged Sat 26 of them.", verification)


def test_span_suppression_rewrites_from_only_verified_claims(
        monkeypatch, vault, conn):
    verified_claim = {"metric": "jog_minutes", "period": "recent",
                      "field": "mean", "value": 50.1,
                      "source": {"sequence": 1, "path": "$.result.mean"}}
    failed_claim = {"metric": "jog_minutes", "period": "recent",
                    "field": "mean", "value": 26,
                    "source": {"sequence": 1, "path": "$.result.mean"}}
    claims = [verified_claim, failed_claim]
    verifications = iter([_partial_failure(), _partial_failure(),
                          _verified_rewrite()])
    drafts = iter([llm.ResearchResponse(
        "Jogging averaged 50.1 minutes and lasted 26 minutes.", claims),
                   llm.ResearchResponse(
                       "Jogging was 50.1 minutes and lasted 26 minutes.",
                       claims)])
    prompts = []
    reverify_args = []

    def verify(conn_arg, prose, claim_arg, ledger_arg, **kwargs):
        reverify_args.append((prose, claim_arg, ledger_arg))
        return next(verifications)

    monkeypatch.setenv("HA_ASK_SPAN_SUPPRESS", "1")
    monkeypatch.setattr(chat, "_verify_ask_answer", verify)
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))
    monkeypatch.setattr(llm, "complete",
                        lambda prompt, **kwargs: prompts.append(prompt)
                        or "Your jogging trend was steady.")

    result = chat.answer_question(vault, "How did 26 compare with my jogging?")

    assert result["mode"] == "narration"
    assert result["verification"]["span_suppressed"] is True
    assert result["text"] == "Your jogging trend was steady."
    assert len(prompts) == 1
    assert '"value": 50.1' in prompts[0]
    assert '"value": 26' not in prompts[0]
    assert "26" not in prompts[0]
    assert reverify_args[2][0] == result["text"]
    assert reverify_args[2][1] == [verified_claim]
    assert reverify_args[2][2] == [{"sequence": 1}]


def test_span_suppression_withholds_a_denial_of_an_available_figure(
        monkeypatch, vault, conn):
    """Suppression cannot publish a digit-free denial as a successful answer."""
    verified_claim = {"metric": "jog_minutes", "period": "recent",
                      "field": "mean", "value": 50.1}
    attempt = {
        "prose": "The draft had one unsupported claim.",
        "claims": [verified_claim, {"metric": "jog_minutes", "value": 26}],
        "verification": _partial_failure(),
        "ledger": _available_jog_ledger(),
    }
    monkeypatch.setenv("HA_ASK_SPAN_SUPPRESS", "1")
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: {
                            "ok": True, "grounded": True,
                            "unsupported": [], "reason": "",
                            "verdict": {"numbers": []},
                            "figures_verified": 0, "figures_total": 0,
                        })
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs:
                        "I don't have your jog minutes for this week.")
    derived_causes = []
    real_ask_cause = chat._ask_cause
    def derive_cause(*args, **kwargs):
        cause = real_ask_cause(*args, **kwargs)
        derived_causes.append(cause)
        return "computed-" + cause
    monkeypatch.setattr(chat, "_ask_cause", derive_cause)

    verification, text, attempts, failures = chat._try_span_suppression(
        vault, "How many jog minutes did I do this week?", attempt,
        as_of="2026-08-23")

    assert text is None
    assert verification["ok"] is False
    assert verification["cause"] == "computed-denied_available_figure"
    assert derived_causes == [
        "denied_available_figure", "denied_available_figure"]
    assert attempts == failures == 2


def test_span_suppression_failure_twice_refuses_whole_answer(
        monkeypatch, vault, conn):
    claims = [{"value": 50.1}, {"value": 26}]
    failed = _partial_failure()
    verifications = iter([failed, failed, failed, failed])
    drafts = iter([llm.ResearchResponse("draft", claims),
                   llm.ResearchResponse("retry", claims)])
    rewrites = iter(["rewrite still says 26", "rewrite still says 26"])
    monkeypatch.setenv("HA_ASK_SPAN_SUPPRESS", "1")
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))
    monkeypatch.setattr(llm, "complete",
                        lambda *args, **kwargs: next(rewrites))

    result = chat.answer_question(vault, "How did I do?")

    assert result["mode"] == "fallback"
    assert result["verification"]["span_suppression"] == "failed"
    assert result["verification"]["span_suppression_attempts"] == 2
    assert result["verification"]["span_suppression_failures"] == 2
    assert "26" not in result["text"]


def test_all_claims_failed_never_attempts_span_suppression(
        monkeypatch, vault, conn):
    claims = [{"value": 26}]
    failed = {
        **_failed_verification(figures_verified=0, figures_total=1,
                               reason="all claims failed", unsupported=["26"]),
        "verdict": {"numbers": [{"ok": False, "claimed": 26}]},
    }
    verifications = iter([failed, failed])
    drafts = iter([llm.ResearchResponse("draft", claims),
                   llm.ResearchResponse("retry", claims)])
    regeneration_calls = []
    monkeypatch.setenv("HA_ASK_SPAN_SUPPRESS", "1")
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))
    monkeypatch.setattr(llm, "complete",
                        lambda *args, **kwargs: regeneration_calls.append(args)
                        or pytest.fail("regeneration must not run"))

    result = chat.answer_question(vault, "How did I do?")

    assert result["mode"] == "fallback"
    assert not regeneration_calls


def test_majority_failed_figures_stay_a_whole_answer_refusal(
        monkeypatch, vault, conn):
    claims = [{"value": 50.1}, {"value": 26}, {"value": 27}]
    failed = {
        "ok": False, "grounded": False,
        "unsupported": ["26", "27"], "reason": "most claims failed",
        "verdict": {"numbers": [
            {"ok": True, "claimed": 50.1},
            {"ok": False, "claimed": 26},
            {"ok": False, "claimed": 27},
        ]},
        "figures_verified": 1, "figures_total": 3,
    }
    verifications = iter([failed, failed])
    drafts = iter([llm.ResearchResponse("draft", claims),
                   llm.ResearchResponse("retry", claims)])
    regeneration_calls = []
    monkeypatch.setenv("HA_ASK_SPAN_SUPPRESS", "1")
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))
    monkeypatch.setattr(llm, "complete",
                        lambda *args, **kwargs: regeneration_calls.append(args)
                        or pytest.fail("majority failure must not regenerate"))

    result = chat.answer_question(vault, "How did I do?")

    assert result["mode"] == "fallback"
    assert "span_suppression" not in result["verification"]
    assert not regeneration_calls


def test_span_suppression_flag_off_keeps_whole_answer_fallback(
        monkeypatch, vault, conn):
    claims = [{"value": 50.1}, {"value": 26}]
    failed = _partial_failure()
    verifications = iter([failed, failed])
    drafts = iter([llm.ResearchResponse("draft", claims),
                   llm.ResearchResponse("retry", claims)])
    regeneration_calls = []
    monkeypatch.delenv("HA_ASK_SPAN_SUPPRESS", raising=False)
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))
    monkeypatch.setattr(llm, "complete",
                        lambda *args, **kwargs: regeneration_calls.append(args)
                        or pytest.fail("flag-off regeneration must not run"))

    result = chat.answer_question(vault, "How did I do?")

    assert result["mode"] == "fallback"
    assert "span_suppressed" not in result["verification"]
    assert not regeneration_calls


def test_question_log_records_the_question_and_verdict_only(
        monkeypatch, vault, conn, tmp_path):
    """HA_ASK_QUESTION_LOG appends question + Python's verdict, never prose."""
    verifications = iter([
        _failed_verification(figures_verified=1, figures_total=2,
                             reason="first failed"),
        _failed_verification(figures_verified=0, figures_total=1,
                             reason="retry failed"),
    ])
    drafts = iter([llm.ResearchResponse("secret draft prose", [{}]),
                   llm.ResearchResponse("secret retry prose")])
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))
    log_path = tmp_path / "questions.jsonl"
    monkeypatch.setenv("HA_ASK_QUESTION_LOG", str(log_path))

    chat.answer_question(vault, "How did I sleep?")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["question"] == "How did I sleep?"
    assert row["mode"] == "fallback"
    assert row["figures_verified"] == 1
    assert "secret" not in lines[0]
    assert "text" not in row and "prose" not in row


def test_question_log_off_by_default_writes_nothing(
        monkeypatch, vault, conn, tmp_path):
    verifications = iter([
        _failed_verification(figures_verified=0, figures_total=1, reason="f1"),
        _failed_verification(figures_verified=0, figures_total=1, reason="f2"),
    ])
    drafts = iter([llm.ResearchResponse("draft"),
                   llm.ResearchResponse("retry")])
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *args, **kwargs: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(drafts))
    monkeypatch.delenv("HA_ASK_QUESTION_LOG", raising=False)
    before = set(tmp_path.rglob("*"))

    chat.answer_question(vault, "How did I sleep?")

    assert set(tmp_path.rglob("*")) == before


def test_degenerate_repeated_token_has_a_distinct_verification_reason():
    degenerate = "\n".join("- ......" for _ in range(22))
    verdict = DV.verify_coach_claims(None, degenerate, None, payload=[])

    assert verdict["ok"] is False
    assert verdict["reason"] == "degenerate repeated-token generation"

    ordinary = DV.verify_coach_claims(
        None, "I ran 3 miles, but no claim was filed.", None, payload=[])
    assert ordinary["reason"] == "numbered coach prose has no structured claims"

    legitimate = DV.verify_coach_claims(
        None, "\n".join("- Rest day" for _ in range(8)), None, payload=[])
    assert legitimate["ok"] is True


def test_capture_adds_no_key_to_the_returned_response(monkeypatch, vault, conn):
    """The draft channel is out-of-band; the client-facing shape does not move."""
    # `HA_ASK_VALUE_REBIND` (Method C) declares one additional verification
    # field, `rebind_counts`, and only while it is on. That is asserted in
    # `tests/test_ledger_value_rebind.py`; the default shape is asserted here,
    # so hold the flag down rather than inherit the ambient environment.
    monkeypatch.delenv("HA_ASK_VALUE_REBIND", raising=False)
    ledger, draft = _verified_draft()
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: 100)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: draft)

    without = chat.answer_question(vault, "How is my jogging?")
    capture = []
    with_capture = chat.answer_question(vault, "How is my jogging?",
                                        capture=capture)

    assert set(without) == ASK_RESULT_KEYS
    assert set(with_capture) == ASK_RESULT_KEYS
    assert set(without["verification"]) == ASK_VERIFICATION_KEYS
    assert set(with_capture["verification"]) == ASK_VERIFICATION_KEYS
    assert with_capture == without
    assert capture

    retry_drafts = iter([llm.ResearchResponse(""), draft])
    monkeypatch.setattr(llm, "tool_loop", lambda *a, **k: next(retry_drafts))
    retried = chat.answer_question(vault, "How is my jogging?", capture=[])

    assert set(retried) == ASK_RESULT_KEYS
    assert set(retried["verification"]) == ASK_VERIFICATION_KEYS | {"retry"}


def test_retry_feedback_names_every_unsupported_token_once():
    """The blob this replaced left the enumerated leftovers unrepaired; each
    one now gets its own sentence, deduplicated and in reported order."""
    rendered = chat._retry_feedback({
        "ok": False,
        "reason": "prose number is not in claims",
        "unsupported": ["26", "3", "26"],
    })

    assert "Reason: prose number is not in claims." in rendered
    assert "You wrote 26 in your text and filed no claim for it." in rendered
    assert "You wrote 3 in your text and filed no claim for it." in rendered
    assert rendered.count("You wrote 26") == 1
    assert rendered.index("You wrote 26") < rendered.index("You wrote 3")
    assert "REMOVE the number from your text" in rendered


def test_retry_feedback_reports_a_failed_claim_with_its_field_hint():
    rendered = chat._retry_feedback({
        "ok": False,
        "reason": "claim field does not match ledger path",
        "unsupported": [],
        "verdict": {"numbers": [
            {"ok": True, "claimed": 50.1, "reason": ""},
            {"ok": False, "claimed": 3.07,
             "reason": "claim field does not match ledger path",
             "actual_field": "total_distance_mi"},
        ]},
    })

    assert ("Your claim of 3.07 failed: claim field does not match ledger "
            "path.") in rendered
    assert "The field at that path is 'total_distance_mi'." in rendered
    assert "50.1" not in rendered


def test_retry_feedback_says_the_leaf_is_unlabelled_when_no_metric_hint():
    """The verifier reports actual_metric=None for a leaf no metric key labels.
    Echoing that as "metric is None" would invite `metric: null`; the model is
    told to omit the slot and cite the path instead."""
    for failure in ({"ok": False, "claimed": 3.04, "actual_metric": None,
                     "reason": "claim metric does not match ledger field"},
                    {"ok": False, "claimed": 3.04,
                     "reason": "claim metric does not match ledger field"}):
        rendered = chat._retry_feedback(
            {"ok": False, "reason": "claim metric does not match ledger field",
             "unsupported": [], "verdict": {"numbers": [failure]}})
        assert ("That leaf carries no metric; omit metric and cite the exact "
                "path.") in rendered
        assert "None" not in rendered

    labelled = chat._retry_feedback({
        "ok": False, "reason": "claim metric does not match ledger field",
        "unsupported": [],
        "verdict": {"numbers": [
            {"ok": False, "claimed": 3.04, "actual_metric": "jog_minutes",
             "reason": "claim metric does not match ledger field"}]},
    })
    assert "That leaf's metric is 'jog_minutes'." in labelled
    assert "carries no metric" not in labelled


def test_the_retry_prompt_names_the_leftover_token(monkeypatch, vault, conn):
    """The second tool_loop call must receive the pointed feedback, not the
    verification dict as JSON."""
    ledger, draft = _verified_draft()
    drafts = iter([llm.ResearchResponse("You ran 26 miles."), draft])
    prompts = []

    def tool_loop(prompt, *args, **kwargs):
        prompts.append(prompt)
        return next(drafts)

    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: 100)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", tool_loop)

    result = chat.answer_question(vault, "How far did I run?")

    assert result["verification"]["retry"] is True
    assert len(prompts) == 2
    retry_prompt = prompts[1]
    assert "Your previous draft failed Python verification or the judge." \
        in retry_prompt
    assert "You wrote 26 in your text and filed no claim for it." in retry_prompt
    assert "fixes every reported issue." in retry_prompt
    assert '"unsupported"' not in retry_prompt


def test_ask_claim_instructions_cover_the_battery_s_prompt_side_gaps():
    """Structural digits, model-side unit conversion, mis-labelled context
    fields, and row indices each produced a live refusal; the workout exception
    that the leaf rule generalizes stays exactly where it was.

    The metric rule is deliberately narrower than "a metric key on the row or
    an ancestor": measured live 2026-08-27, that phrasing sent the model into
    `claim metric does not match ledger field` on every sibling field of an
    impact period row (jog_miles, days_covered, jog_pace_min_per_mi), because
    the resolver labels only the field the row's metric key owns."""
    text = " ".join(chat.ASK_CLAIM_INSTRUCTIONS.split())
    assert "Every digit sequence in text is checked" in text
    assert "State every figure in exactly the units the tool publishes" in text
    assert ("Name a `metric` ONLY on a leaf whose own field is that series' "
            "value") in text
    assert ("takes a claim with `metric` OMITTED and the exact path as its "
            "source, even when the row carries a `metric` key") in text
    assert ("`get_weekly_series` publishes each row's inclusive Monday-Sunday "
            "`period` as `YYYY-MM-DD:YYYY-MM-DD`; copy that exact string into "
            "a claim and never invent a period from `week_start`") in text
    # A style rule, not a gate disclosure: "Tuesday 08-18" tokenized as 08 and
    # 18 and sank an answer whose 14/14 figures verified (battery 2026-08-27).
    assert ("either in full ISO form (2026-08-18) or with its month name "
            "(Aug 18)") in text
    assert "copy the index of the exact row you read from the result" in text
    assert "EXCEPTION — workout rows" in text
    assert "OMIT `metric` entirely" in text
    # The strip's negative space is never documented to the model: telling it
    # dates and clock times are exempt is telling it where the gate is blind.
    assert "AM" not in text and "clock" not in text


def test_new_tables_are_created_for_an_older_schema(tmp_path):
    """This feature adds tables, not columns to an existing table."""
    path = tmp_path / "older.db"
    conn = dbmod.connect(path)
    conn.execute("CREATE TABLE legacy (id INTEGER PRIMARY KEY)")
    conn.commit()
    dbmod.init_db(conn)
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"conversations", "conversation_turns"} <= tables
    conn.close()


def test_no_data_regex_requires_a_data_object_not_coaching_prose():
    # The live false positive of 2026-08-31: slotless coaching prose tripped
    # the bare "cannot find" branch, then "weight" matched body_mass's alias.
    coaching = ("If you cannot find a sturdy table for inverted rows, add "
                "weight with dumbbells instead. Nothing is missing from a "
                "simple plan.")
    assert chat._EMPTY_NARRATION_RE.search(coaching) is None

    for genuine in (
        "I don't have information about your longest run for last month.",
        "I don't have data for step count.",
        "I cannot compare your running volume because the data for the two "
        "periods is not available.",
        "No records of workouts exist for that month.",
    ):
        assert chat._EMPTY_NARRATION_RE.search(genuine) is not None, genuine


def test_pure_advice_answer_narrates_without_a_tool_ledger(monkeypatch, vault):
    # Live 2026-08-31: a conversational follow-up's gather made no tool calls
    # ("write this up into a circuit"), and the arm refused with 'ask answer
    # has no tool-call ledger' despite the answer claiming zero vault facts.
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    responses = iter([
        "",
        "Circuit: {advice:3 rounds} of squats, push-ups, rows; rest "
        "{advice:60 seconds} between exercises.",
    ])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    result = chat.answer_question(vault, "Can you write this as a circuit?")

    assert result["mode"] == "narration"
    assert result["verification"]["advice_quantities"] == [
        "3 rounds", "60 seconds"]
    assert result["verification"]["figures_total"] == 0


def test_zero_advice_prose_without_a_ledger_still_falls_back(
        monkeypatch, vault):
    # The guard the exemption must not open: a lazy no-gather qualitative
    # answer on a data question carries no advice spans and stays refused.
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    responses = iter([
        "",
        "Your training is going well overall.",
        "Your training is going well overall.",
    ])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    result = chat.answer_question(vault, "How is my training going?")

    assert result["mode"] == "fallback"
