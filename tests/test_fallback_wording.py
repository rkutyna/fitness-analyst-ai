"""The refusal text an end user reads is plain language, not internal jargon.

``chat._fallback_answer`` renders it from the verification dict.  The dict keeps
the raw internal reason for developers; the text never repeats it, never
repeats a figure or quoted text from the refused answer, and does not use the
words of the verification machinery.
"""
from __future__ import annotations

import re

import pytest

from health_advisor import chat, fact_template, steering

BANNED = ("draft", "placeholder", "python", "verdict", "verif", "ledger",
          "template", "narration", "slot")

OPENING = "I couldn't give you an answer I could check against your data."
CLOSING = "Try asking about one thing at a time, or a shorter period."

# Every exact reason the ask path can hand to the fallback, with a phrase the
# plain sentence must contain.
KNOWN_REASONS = [
    ("answer truncated", "cut off"),
    ("digit outside placeholder", "number I couldn't trace"),
    (fact_template.NUMBER_WORD_REASON, "number I couldn't trace"),
    (fact_template.ADVICE_QUANTITY_REASON, "amount"),
    ("ask answer has no tool-call ledger", "didn't look anything up"),
    (steering.REASON, "wandered away"),
    ("malformed placeholder", "wasn't put together correctly"),
    ("unresolvable placeholder", "wasn't put together correctly"),
    ("empty advice slot", "wasn't put together correctly"),
    ("empty conversational answer", "wasn't put together correctly"),
    ("conversational answer contains a placeholder",
     "wasn't put together correctly"),
    ("conversational answer states something the user did", "claim"),
    ("conversational answer praises a recorded result", "claim"),
    ("conversational answer makes the user data the subject of a statement",
     "claim"),
    ("citation verification refused", "source"),
    ("citation verifier is unavailable on the chat path", "source"),
    (chat._DENIED_AVAILABLE_FIGURE_REASON, "no figure for this"),
    (chat._WITHHELD_ELIGIBLE_FIGURE_REASON, "left out"),
    ("empty narration does not name a missing metric family",
     "some of your data was missing"),
    ("empty narration names a metric whose coverage is not missing",
     "some of your data was missing"),
    (chat._WINDOW_DATA_UNAVAILABLE_REASON, "don't cover the period"),
]

# Reasons that embed answer-derived specifics, keyed by their closed cause.
CAUSE_CASES = [
    ("unsupported_period_phrase",
     "narration names a period longer than the data: 'over 40 weeks' needs 280",
     "longer than your data"),
    ("stale_window",
     "narration cites a period that ended before the recent data: 2026-07-26",
     "ended well before your most recent data"),
    ("contradicted_day_count",
     "narration contradicts its own day itemisation; stated 5, itemised 3",
     "day count that didn't match"),
    ("restated_unit",
     "narration restates a rendered unit; placeholder 'k' shows 61 bpm",
     "unit that was already shown"),
]


def _assert_plain(text: str) -> None:
    lowered = text.lower()
    for word in BANNED:
        assert word not in lowered, f"{word!r} leaked into: {text}"
    assert not text.startswith("Fallback")
    assert text.startswith(OPENING)
    assert text.endswith(CLOSING)


@pytest.mark.parametrize("reason, phrase", KNOWN_REASONS)
def test_every_known_reason_gets_a_plain_sentence(reason, phrase):
    text = chat._fallback_answer({"ok": False, "reason": reason})

    _assert_plain(text)
    assert phrase in text
    assert not re.search(r"\d", text)
    assert reason not in text


def test_known_reasons_cover_the_exact_reason_table():
    """A reason added to the table without a case here is caught."""
    table = set(chat._fallback_reason_sentences())
    assert table <= {reason for reason, _ in KNOWN_REASONS}


@pytest.mark.parametrize("cause, reason, phrase", CAUSE_CASES)
def test_cause_labelled_reasons_never_repeat_the_answers_specifics(
        cause, reason, phrase):
    text = chat._fallback_answer(
        {"ok": False, "cause": cause, "reason": reason})

    _assert_plain(text)
    assert phrase in text
    assert not re.search(r"\d", text)
    for fragment in ("40 weeks", "280", "2026-07-26", "61 bpm", "itemised"):
        assert fragment not in text


@pytest.mark.parametrize("reason", [
    "ledger path not found",
    "claim value does not match ledger field",
    "draft value 987.5 was rejected",
    "synthetic internal reason with 'quoted draft text' and 42",
    "",
])
def test_unknown_or_missing_reason_gets_only_the_generic_sentences(reason):
    text = chat._fallback_answer({"ok": False, "reason": reason})

    _assert_plain(text)
    assert text == f"{OPENING} {CLOSING}"
    assert not re.search(r"\d", text)


@pytest.mark.parametrize("verification", [None, {}, {"reason": None},
                                          "not a dict"])
def test_missing_verification_is_the_generic_text(verification):
    assert chat._fallback_answer(verification) == f"{OPENING} {CLOSING}"


def test_unsupported_tokens_are_counted_never_repeated():
    one = chat._fallback_answer({"unsupported": ["14"],
                                 "reason": "claim value does not match"})
    two = chat._fallback_answer({"unsupported": ["14", "97", "14"]})

    for text in (one, two):
        assert text.startswith(OPENING) and text.endswith(CLOSING)
        for word in BANNED:
            assert word not in text.lower()
    assert "one figure" in one and "14" not in one and "97" not in one
    assert "2 figures" in two and "14" not in two and "97" not in two


def test_a_known_reason_wins_over_the_figure_count():
    text = chat._fallback_answer({
        "unsupported": ["14"], "reason": "digit outside placeholder"})

    assert "number I couldn't trace" in text
    assert "one figure" not in text


def test_the_raw_reason_stays_on_the_result_metadata(monkeypatch):
    """The user text is plain; the developer reason is untouched."""
    monkeypatch.setattr(steering, "leak", lambda text: "matched")
    result = chat._refuse_steering_leak({
        "text": "anything", "mode": "narration",
        "verification": {"ok": True, "reason": ""}})

    assert result["mode"] == "fallback"
    assert result["verification"]["reason"] == steering.REASON
    assert result["verification"]["cause"] == "gate_refused"
    assert steering.REASON not in result["text"]
    _assert_plain(result["text"])
