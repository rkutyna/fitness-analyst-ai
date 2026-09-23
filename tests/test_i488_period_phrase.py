"""A gate for relative-period phrases the model writes itself (#488).

The ask path's fact-template prompt tells the model to put any period in a
Python-rendered ``{...|field=period_label}`` placeholder rather than writing
one itself.  A tester saw the model do it anyway: "Over the past month, your
sleep has been fairly consistent." on a vault holding 21 days of data --
every figure verified, but the period phrase was the model's own words and
nothing checked it. ``chat._mark_unsupported_period_phrase`` is the post-scan
marker that catches this, modelled on ``chat._mark_restated_rendered_unit``.
"""
from __future__ import annotations

import pytest

from health_advisor import chat
from health_advisor import db as dbmod
from health_advisor import fact_template
from health_advisor import llm
from tests.conftest import seed_metric

O21_SENTENCE = (
    "Over the past month, your sleep has been fairly consistent.")
O21_QUESTION = "How has my sleep been?"


# --------------------------------------------------------------------- #
# Direct unit tests of the marker.
# --------------------------------------------------------------------- #

def _assert_o21_refused_at_21_days():
    """The measured O21 sentence, refused against a 21-day vault.

    The question ("How has my sleep been?") names no period itself, so the
    exemption below does not apply and this is unchanged by it.

    Factored into a helper (rather than inlined in the test body) so the
    mutation test below can run the *same* assertions under a stubbed marker
    and show them go red -- proving this isn't a vacuous check.
    """
    verification = {"ok": True}
    fired = chat._mark_unsupported_period_phrase(
        verification, text=O21_SENTENCE, template=O21_SENTENCE, facts={},
        history_days=21, question=O21_QUESTION)

    assert fired is True
    assert verification["ok"] is False
    assert verification["grounded"] is False
    reason = verification["reason"]
    assert "past month" in reason
    assert "28" in reason
    assert "21" in reason
    assert verification["unsupported_period_phrase"] == {
        "phrase": "over the past month",
        "needed_days": 28,
        "history_days": 21,
    }


def test_o21_sentence_refused_at_21_days_history():
    _assert_o21_refused_at_21_days()


def test_o21_sentence_passes_at_400_days_history():
    verification = {"ok": True}
    fired = chat._mark_unsupported_period_phrase(
        verification, text=O21_SENTENCE, template=O21_SENTENCE, facts={},
        history_days=400, question=O21_QUESTION)

    assert fired is False
    assert verification == {"ok": True}


def test_past_few_weeks_passes_at_21_days_history():
    template = "Over the past few weeks, your training load has climbed."
    verification = {"ok": True}
    fired = chat._mark_unsupported_period_phrase(
        verification, text=template, template=template, facts={},
        history_days=21, question="How has my training load been?")

    # "few weeks" needs 14 days; 21 days of history covers it.
    assert fired is False
    assert verification == {"ok": True}


def test_past_couple_of_months_refused_at_21_days_history():
    template = "Over the past couple of months, your resting HR has dropped."
    verification = {"ok": True}
    fired = chat._mark_unsupported_period_phrase(
        verification, text=template, template=template, facts={},
        history_days=21, question="How has my resting heart rate changed?")

    assert fired is True
    assert verification["ok"] is False
    assert "couple of months" in verification["reason"]
    assert "56" in verification["reason"]
    assert "21" in verification["reason"]


def test_no_period_phrase_passes():
    template = "Your training load has been climbing steadily."
    verification = {"ok": True}
    fired = chat._mark_unsupported_period_phrase(
        verification, text=template, template=template, facts={},
        history_days=21, question="How has my training load been?")

    assert fired is False
    assert verification == {"ok": True}


def test_ignores_phrase_inside_rendered_placeholder_value():
    """A period phrase Python renders into a placeholder is not the model's.

    The template still carries the placeholder token -- the interpolated
    ``text`` is what would contain "over the past month" here, rendered from
    a ``period_label`` fact -- but the marker scans the TEMPLATE, mirroring
    ``_mark_restated_rendered_unit``'s own precedent (it accepts ``text`` but
    never reads it). Placeholders are Python-interpolated, so this proves the
    detection can never see Python-owned rendered content, only the model's
    own prose.
    """
    template = (
        "Progress during "
        "{fact|metric=jog_minutes|period=s:2026-07-25:2026-08-21|"
        "field=period_label} was solid."
    )
    # What the template above would render to, if a period_label fact's
    # display happened to read like a relative-period phrase.
    rendered_text = "Progress during over the past month was solid."
    verification = {"ok": True}

    fired = chat._mark_unsupported_period_phrase(
        verification, text=rendered_text, template=template, facts={},
        history_days=21, question="How has my jogging been going?")

    assert fired is False
    assert verification == {"ok": True}


def test_marker_noops_when_history_days_unknown():
    verification = {"ok": True}
    fired = chat._mark_unsupported_period_phrase(
        verification, text=O21_SENTENCE, template=O21_SENTENCE, facts={},
        history_days=None, question=O21_QUESTION)

    assert fired is False
    assert verification == {"ok": True}


# --------------------------------------------------------------------- #
# The question's own calendar phrase is exempt: it is Python's resolved
# window, echoed back, not the model inventing one (orchestrator review).
# --------------------------------------------------------------------- #

def test_question_naming_the_same_period_exempts_the_phrase():
    """"How did I sleep last month?" makes "Over the past month" an echo.

    Same O21 sentence, same 21-day vault that refuses it above -- the only
    difference is a question that names "month" itself, so this is not the
    model inventing a period the data cannot support.
    """
    verification = {"ok": True}
    fired = chat._mark_unsupported_period_phrase(
        verification, text=O21_SENTENCE, template=O21_SENTENCE, facts={},
        history_days=21, question="How did I sleep last month?")

    assert fired is False
    assert verification == {"ok": True}


def test_question_naming_last_week_exempts_it_even_at_two_days_history():
    """The #16 cycling shape (tests/test_ask_day_count_gate.py): "last week"
    in the answer to "How often did I cycle last week?" is not refused even
    though the vault's own daily_metrics history is only 2 days -- the real
    evidence is the cited tool result, and Python already resolved "last
    week" from the question itself.
    """
    template = "Last week you cycled on two days: Tuesday and Friday."
    verification = {"ok": True}
    fired = chat._mark_unsupported_period_phrase(
        verification, text=template, template=template, facts={},
        history_days=2, question="How often did I cycle last week?")

    assert fired is False
    assert verification == {"ok": True}


def test_question_naming_a_different_unit_does_not_exempt():
    """Exemption compares the phrase's UNIT, not just "some period phrase
    was in the question": a question about weeks does not license the model
    inventing an unsupported month claim.
    """
    template = "Over the past month, your sleep has been fairly consistent."
    verification = {"ok": True}
    fired = chat._mark_unsupported_period_phrase(
        verification, text=template, template=template, facts={},
        history_days=21, question="How did I sleep last week?")

    assert fired is True
    assert verification["ok"] is False
    assert "past month" in verification["reason"]


# --------------------------------------------------------------------- #
# Mutation: stub the marker to always return False and watch the refusal
# tests above go red.
# --------------------------------------------------------------------- #

def test_mutation_stubbed_marker_breaks_o21_refusal(monkeypatch):
    monkeypatch.setattr(chat, "_mark_unsupported_period_phrase",
                        lambda *args, **kwargs: False)

    with pytest.raises(AssertionError):
        _assert_o21_refused_at_21_days()


# --------------------------------------------------------------------- #
# Integration-shaped test through the same fact-template entry point the
# unit marker's own tests use (tests/test_ask_rendered_units.py).
# --------------------------------------------------------------------- #

def _hr_ledger(period, value=62, display="62 bpm"):
    return [{
        "sequence": 1,
        "tool_name": "synthetic_metric",
        "arguments": {},
        "result": {
            "metric": "resting_heart_rate",
            "period": period,
            "mean": value,
            "unit": "bpm",
            "presentation": {
                "metric": "resting_heart_rate",
                "period": period,
                "field": "presentation",
                "value": display,
            },
        },
    }]


def _template_arm(monkeypatch, vault, ledger, templates, as_of):
    replies = iter(["acknowledged", *templates])
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(replies))
    capture = []
    result = chat.answer_question(
        vault, "How has my resting heart rate been?", as_of=as_of,
        capture=capture)
    return result, capture


def test_unsupported_period_phrase_integration(monkeypatch, vault):
    """A 21-day vault refuses "past month", the retry lands, cause names it.

    The vault's real ``daily_metrics`` rows are what ``history_days`` is
    measured from (via ``cold_start._first_date``); the mocked ledger below
    is the separate closed fact set the drafts are templated against -- the
    same split ``test_ask_rendered_units.py`` uses for the unit marker.
    """
    conn = vault.connect()
    dbmod.init_db(conn)
    seed_metric(conn, "step_count", "2026-08-01", [100] * 21)
    conn.close()

    period = "2026-07-25:2026-08-21"
    key = fact_template.fact_key("resting_heart_rate", period, "mean")
    ledger = _hr_ledger(period)
    bad = (f"Over the past month, your resting heart rate averaged "
           f"{{{key}}} bpm.")
    good = f"Your resting heart rate has averaged {{{key}}} bpm recently."

    result, capture = _template_arm(monkeypatch, vault, ledger, [bad, good],
                                    as_of="2026-08-21")

    assert capture[0]["verification"]["ok"] is False
    assert capture[0]["verification"]["cause"] == "unsupported_period_phrase"
    assert "past month" in capture[0]["verification"]["reason"]
    assert "28" in capture[0]["verification"]["reason"]
    assert "21" in capture[0]["verification"]["reason"]

    assert result["mode"] == "narration"
    assert result["verification"]["retry"] is True
    assert result["verification"]["cause"] == "ok"
    assert "past month" not in result["text"]
    assert "62" in result["text"]
