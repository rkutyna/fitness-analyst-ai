"""health_advisor#495 -- invisible and combining characters in model prose.

Model prose sometimes carried U+0308 COMBINING DIAERESIS or a zero-width /
format character right before a digit, usually after a doubled space. The
shared tokenizer reads straight through such marks, so every gate verified the
figure and the stray diacritic PUBLISHED. ``numeric_tokens.
normalise_model_prose`` is the one publish-time normaliser; these tests pin
what it removes, what it must never touch, and that every /v1/ask narration
path applies it BEFORE its gate, so the gate reads exactly what publishes.

Every figure and date here is invented; only the shape of the defect is real.

Paths covered, one test each (the model call sites are enumerated from code in
``tests/test_windowed_paths_state_the_date.py``; the only sites NOT listed are
``chat._ask_judge``, ``agents.run_model`` and ``analyst.run_analyst``, none of
which produces prose that /v1/ask publishes):

* the fact-template gather turn's conversational reply;
* the fact-template narration and its repair (``_narrate_audit`` reaches the
  same function, so it is covered by these two);
* the prose (claim-channel) first draft and its retry;
* the span-suppression regeneration.
"""
from __future__ import annotations

import unicodedata

import pytest

from health_advisor import chat
from health_advisor import fact_template
from health_advisor import llm
from health_advisor.numeric_tokens import normalise_model_prose
from tests.conftest import seed_metric


# ---------------------------------------------------------------------------
# The function.
# ---------------------------------------------------------------------------

def test_the_two_observed_strings_normalise():
    assert normalise_model_prose("on  \u03083 of  \u03084") == "on 3 of 4"
    assert normalise_model_prose("by  \u200b10-20%") == "by 10-20%"


@pytest.mark.parametrize("invisible", [
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u200e", "\u00ad",
])
def test_every_format_character_is_removed_before_a_digit(invisible):
    assert normalise_model_prose(f"ran  {invisible}12 km") == "ran 12 km"
    assert normalise_model_prose(f"ran {invisible}12 km") == "ran 12 km"
    assert normalise_model_prose(f"1{invisible}54 bpm") == "154 bpm"


@pytest.mark.parametrize("mark", ["\u0300", "\u0301", "\u0304", "\u0308",
                                  "\u0336", "\u036f"])
def test_a_mark_on_no_letter_is_removed(mark):
    assert normalise_model_prose(f"on  {mark}3 nights") == "on 3 nights"
    assert normalise_model_prose(f"5{mark} km") == "5 km"
    assert normalise_model_prose(f"({mark}3)") == "(3)"
    assert normalise_model_prose(f"{mark}3 nights") == "3 nights"


def test_accented_letters_survive_precomposed_and_decomposed():
    precomposed = "caf\u00e9 ni\u00f1o \u00fcber"
    decomposed = "cafe\u0301 nin\u0303o u\u0308ber"
    assert normalise_model_prose(precomposed) == precomposed
    assert normalise_model_prose(decomposed) == decomposed
    # A mark on a letter survives even when a digit follows the letter.
    assert normalise_model_prose("e\u0301 3") == "e\u0301 3"
    assert normalise_model_prose("x\u03083") == "x\u03083"
    # No normalisation form is applied: decomposed stays decomposed.
    assert (unicodedata.normalize("NFC", decomposed)
            != normalise_model_prose(decomposed))


def test_visible_typography_survives_byte_for_byte():
    text = "14\u201315 min \u2014 about 20\u00b0C, 3\u00d7400 m, \u22485 km \u2713 \u2717 \u2022 done"
    assert normalise_model_prose(text) == text


def test_paragraphs_and_untouched_spacing_survive():
    text = "First  paragraph.\n\n  Indented second.\n\tTabbed."
    assert normalise_model_prose(text) == text
    assert (normalise_model_prose("One.\n\n\u200bTwo  \u03083.\n\nThree.")
            == "One.\n\nTwo 3.\n\nThree.")


def test_non_string_and_empty_input_pass_through():
    assert normalise_model_prose(None) is None
    assert normalise_model_prose("") == ""


def test_it_is_idempotent():
    once = normalise_model_prose("on  \u03083 of  \u200b\u03084, cafe\u0301")
    assert normalise_model_prose(once) == once


# ---------------------------------------------------------------------------
# The /v1/ask paths. Each stubs the model with MARKED prose and asserts the
# published text is the clean prose, and that the gate's record (the capture)
# holds the same clean text, i.e. the gate read what published.
# ---------------------------------------------------------------------------

QUESTION = "How often did I cycle last week?"
AS_OF = "2031-03-09"
PERIOD = "2031-03-02:2031-03-08"
MARKS = ("\u200b", "\u0308", "\u2060", "\ufeff")


@pytest.fixture
def _seeded(conn):
    seed_metric(conn, "step_count", "2031-03-08", [1000])


def _ride_ledger() -> list[dict]:
    return [{
        "sequence": 1,
        "tool_name": "get_impact_volume",
        "arguments": {"start": "2031-03-02", "end": "2031-03-08",
                      "by": "week"},
        "result": {"metric": "ride_minutes", "period": PERIOD,
                   "ride_minutes": 77.5, "unit": "min"},
    }]


def _key() -> str:
    return fact_template.fact_key("ride_minutes", PERIOD, "ride_minutes")


def _assert_clean(text: str) -> None:
    assert not any(mark in text for mark in MARKS), repr(text)
    assert "  " not in text, repr(text)


MARKED_PROSE = ("You cycled on  \u0308two days last week:  \u200bTuesday "
                "and Friday.")
CLEAN_PROSE = "You cycled on two days last week: Tuesday and Friday."


def _template_arm(monkeypatch, vault, templates):
    replies = iter(["acknowledged", *templates])
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: _ride_ledger())
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: next(replies))
    capture: list[dict] = []
    result = chat.answer_question(vault, QUESTION, as_of=AS_OF,
                                  capture=capture)
    return result, capture


def _prose_arm(monkeypatch, vault, answers):
    replies = iter(answers)
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "0")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: _ride_ledger())
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: 95)
    capture: list[dict] = []
    result = chat.answer_question(vault, QUESTION, as_of=AS_OF,
                                  capture=capture)
    return result, capture


def _marked_template() -> str:
    # The observed shape exactly: doubled space, then the artifact, then the
    # figure -- here a Python-interpolated one.
    return ("You logged  \u200b{" + _key() + "} on  \u0308two rides:  "
            "\u2060Tuesday and Friday.")


def test_conversational_reply_publishes_clean(monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop", lambda *args, **kwargs:
        "Hello!  \u200bI'm ready to help. What would you like to look "
        "\ufeffinto?")
    result = chat.answer_question(vault, "hi there")
    assert result["mode"] == "narration"
    assert result["verification"]["cause"] == "conversational"
    assert result["text"] == ("Hello! I'm ready to help. What would you "
                              "like to look into?")


def test_template_first_attempt_publishes_clean(monkeypatch, vault, _seeded):
    result, capture = _template_arm(monkeypatch, vault, [_marked_template()])
    assert result["mode"] == "narration"
    assert result["text"].startswith("You logged 77.5")
    assert "on two rides: Tuesday and Friday." in result["text"]
    _assert_clean(result["text"])
    _assert_clean(capture[0]["prose"])


def test_template_repair_publishes_clean(monkeypatch, vault, _seeded):
    result, capture = _template_arm(
        monkeypatch, vault,
        ["You cycled 3 times last week.", _marked_template()])
    assert capture[0]["verification"]["cause"] == "gate_refused"
    assert result["mode"] == "narration"
    assert result["verification"]["retry"] is True
    assert result["text"].startswith("You logged 77.5")
    _assert_clean(result["text"])
    _assert_clean(capture[1]["prose"])


def test_prose_first_draft_publishes_clean(monkeypatch, vault, _seeded):
    result, capture = _prose_arm(monkeypatch, vault, [MARKED_PROSE])
    assert result["mode"] == "narration"
    assert result["text"] == CLEAN_PROSE
    assert capture[0]["prose"] == CLEAN_PROSE


def test_prose_retry_publishes_clean(monkeypatch, vault, _seeded):
    result, capture = _prose_arm(
        monkeypatch, vault,
        ["You cycled 41.0 minutes last week.", MARKED_PROSE])
    assert not capture[0]["verification"]["ok"]
    assert result["mode"] == "narration"
    assert result["verification"]["retry"] is True
    assert result["text"] == CLEAN_PROSE
    assert capture[1]["prose"] == CLEAN_PROSE


def test_span_suppression_regeneration_publishes_clean(monkeypatch, vault,
                                                       _seeded):
    claim = {"metric": "jog_minutes", "period": "recent", "field": "mean",
             "value": 50.1,
             "source": {"sequence": 1, "path": "$.result.mean"}}
    failed = {**claim, "value": 26}
    partial = {
        "ok": False, "grounded": False, "unsupported": ["26"],
        "reason": "one claim failed",
        "verdict": {"numbers": [{"ok": True, "claimed": 50.1},
                                {"ok": False, "claimed": 26,
                                 "reason": "wrong value"}]},
        "figures_verified": 1, "figures_total": 2,
    }
    verified = {
        "ok": True, "grounded": True, "unsupported": [], "reason": "",
        "verdict": {"numbers": [{"ok": True, "claimed": 50.1}]},
        "figures_verified": 1, "figures_total": 1,
    }
    verifications = iter([partial, dict(partial), verified])
    reverified: list[str] = []
    drafts = iter([
        llm.ResearchResponse("Jogging averaged 50.1 minutes and lasted 26 minutes.",
                             [claim, failed]),
        llm.ResearchResponse("Jogging was 50.1 minutes and lasted 26 minutes.",
                             [claim, failed]),
    ])

    def verify(conn_arg, prose, claims, ledger, **kwargs):
        reverified.append(prose)
        return next(verifications)

    monkeypatch.setenv("HA_ASK_SPAN_SUPPRESS", "1")
    monkeypatch.setattr(chat, "_verify_ask_answer", verify)
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: next(drafts))
    monkeypatch.setattr(llm, "complete", lambda prompt, **kwargs:
                        "Your jogging  \u200btrend was  \u0308steady.")

    result = chat.answer_question(vault, "How did 26 compare with my jogging?",
                                  as_of=AS_OF)

    assert result["verification"].get("span_suppressed") is True, result
    assert result["text"] == "Your jogging trend was steady."
    assert reverified[2] == result["text"]


def test_an_emoji_zwj_sequence_survives():
    # Measured: 2 of the 8 capture prose fields carrying a format character
    # were a runner-with-gender-sign emoji, whose joiner is the glyph.
    runner = "\U0001f3c3\u200d\u2642\ufe0f"
    text = "Ready when you are! " + runner + "\U0001f4a4"
    assert normalise_model_prose(text) == text
    # A joiner beside ASCII is still the artifact.
    assert normalise_model_prose("up to  \u200d15 min") == "up to 15 min"
    assert normalise_model_prose("1\u200c5 km") == "15 km"
