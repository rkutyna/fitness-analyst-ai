"""A narration whose window ended well before the data is refused (health_advisor#428).

A live answer was three weeks stale with every figure correct: the model
chose the PERIOD, every claim was bound, and no gate asked whether the window
was current. ``chat._mark_stale_window`` is the post-scan marker that asks: it
takes the latest period END the template cites (from the Python-published
``period=`` field of its placeholder keys), compares it with the most recent
day with data among the cited metrics, and refuses when the gap exceeds
``chat.STALE_WINDOW_DAYS``. It is modelled on #488's
``_mark_unsupported_period_phrase`` and repaired through the same loop.
"""
from __future__ import annotations

from datetime import date

import pytest

from health_advisor import chat
from health_advisor import db as dbmod
from health_advisor import fact_template
from health_advisor import llm
from health_advisor.calendar_window import resolve_window
from tests.conftest import seed_metric

AS_OF = date(2026, 8, 21)
HORIZONS = {"resting_heart_rate": AS_OF}
STALE = "2026-07-20:2026-07-26"     # ends 26 days before AS_OF
CURRENT = "2026-08-10:2026-08-16"   # ends 5 days before AS_OF


def _facts(*periods):
    facts = {}
    for period in periods:
        key = fact_template.fact_key("resting_heart_rate", period, "mean")
        facts[key] = {"key": key, "metric": "resting_heart_rate",
                      "period": period, "field": "mean", "value": 60}
    return facts


def _template(*periods):
    return " and ".join(
        "{" + fact_template.fact_key("resting_heart_rate", p, "mean") + "}"
        for p in periods)


def _mark(template, facts, question="How has my resting heart rate been?",
          horizons=None, verification=None, as_of=AS_OF):
    verification = {"ok": True} if verification is None else verification
    fired = chat._mark_stale_window(
        verification, template=template, facts=facts,
        metric_horizons=HORIZONS if horizons is None else horizons,
        question=question, resolved_window=resolve_window(question, as_of))
    return fired, verification


# --------------------------------------------------------------------- #
# The marker.
# --------------------------------------------------------------------- #

def _assert_stale_window_refused():
    """Factored out so the mutation test can re-run it under a stub."""
    fired, verification = _mark(_template(STALE), _facts(STALE))

    assert fired is True
    assert verification["ok"] is False
    assert verification["grounded"] is False
    assert "2026-07-26" in verification["reason"]
    assert "2026-08-21" in verification["reason"]
    assert verification["stale_window"] == {
        "latest_cited_end": "2026-07-26",
        "most_recent_data": "2026-08-21",
        "gap_days": 26,
        "threshold_days": chat.STALE_WINDOW_DAYS,
    }


def test_stale_window_is_refused():
    _assert_stale_window_refused()


def test_current_window_passes():
    fired, verification = _mark(_template(CURRENT), _facts(CURRENT))

    assert fired is False
    assert verification == {"ok": True}


def test_latest_cited_end_decides_so_a_comparison_with_an_old_block_passes():
    fired, _ = _mark(_template(STALE, CURRENT), _facts(STALE, CURRENT))

    assert fired is False


def test_threshold_boundary():
    at = "2026-08-01:2026-08-07"      # 14 days before AS_OF
    past = "2026-07-31:2026-08-06"    # 15 days before AS_OF

    assert _mark(_template(at), _facts(at))[0] is False
    assert _mark(_template(past), _facts(past))[0] is True


@pytest.mark.parametrize("question", [
    "How was my resting heart rate in July?",
    "How was August?",
    "What was my resting heart rate on Jul 22?",
    "How did 2025 compare?",
    "How was my resting heart rate last year?",
])
def test_a_question_naming_an_older_period_is_exempt(question):
    fired, verification = _mark(_template(STALE), _facts(STALE),
                                question=question)

    assert fired is False
    assert verification == {"ok": True}


def test_a_resolved_calendar_window_caps_the_anchor():
    """"last month" asked on Aug 21 is July; a July window is current for
    it, a June window is not."""
    question = "How was my resting heart rate last month?"
    july = "2026-07-01:2026-07-31"
    june = "2026-06-01:2026-06-30"

    assert _mark(_template(july), _facts(july), question=question)[0] is False
    fired, verification = _mark(_template(june), _facts(june),
                                question=question)
    assert fired is True
    assert verification["stale_window"]["most_recent_data"] == "2026-07-31"


def test_a_rolling_period_phrase_is_not_exempt():
    """#488 exempts any period phrase the question names, because its check
    is about the phrase's LENGTH. Staleness is about where a window ENDS, and
    "the last few weeks" ends now -- so an old window answering it is the
    #428 defect itself, not an exemption."""
    fired, _ = _mark(_template(STALE), _facts(STALE),
                     question="How was my resting heart rate over the last "
                              "few weeks?")

    assert fired is True


# A question about one past day or event names no present: the old window
# it cites is its answer. Each of these was refused at a 26-day gap before
# the check became default-exempt (adversarial review of #428).
SPECIFIC_PAST_QUESTIONS = [
    "How did I sleep on 8/29?",
    "How did I sleep the night before my half marathon?",
    "What did I do 3 weeks ago on Saturday?",
    "What was my HR on my last long run?",
]


def _assert_specific_past_question_exempt(question):
    fired, verification = _mark(_template(STALE), _facts(STALE),
                                question=question)
    assert fired is False, question
    assert verification == {"ok": True}, question


def _assert_specific_past_questions_exempt():
    for question in SPECIFIC_PAST_QUESTIONS:
        _assert_specific_past_question_exempt(question)


@pytest.mark.parametrize("question", SPECIFIC_PAST_QUESTIONS)
def test_a_question_about_a_specific_past_day_is_exempt(question):
    _assert_specific_past_question_exempt(question)


@pytest.mark.parametrize("period", [
    "2026-07-27:2026-08-02",    # ends 19 days before AS_OF
    "2026-07-28:2026-08-03",    # ends 18 days before AS_OF
])
def test_the_historical_rolling_true_positive_is_still_refused(period):
    fired, verification = _mark(
        _template(period), _facts(period),
        question="How much have I been running over the last two weeks?")

    assert fired is True
    assert verification["stale_window"]["gap_days"] in (18, 19)


# The present/rolling questions the window battery asks: every one stays
# subject to the check.
@pytest.mark.parametrize("question", [
    "How has my running been lately?",
    "Is my resting heart rate trending the right way?",
    "How has my sleep been recently?",
    "Have I been more active than usual?",
    "How consistent have I been with my training?",
    "Is my weight moving?",
    "How is my cardio fitness doing?",
    "What has changed in my heart rate variability?",
    # #70's reproducer.
    "How's my running been?",
])
def test_a_present_or_rolling_question_is_subject_to_the_check(question):
    exempt, _cap = chat._stale_window_scope(question,
                                            resolve_window(question, AS_OF))
    assert exempt is False
    assert _mark(_template(STALE), _facts(STALE), question=question)[0] is True


def test_mutation_present_marker_always_true_refuses_the_past_day(
        monkeypatch):
    monkeypatch.setattr(chat, "_asks_about_present", lambda question: True)

    with pytest.raises(AssertionError):
        _assert_specific_past_questions_exempt()


def test_a_sparse_metric_is_measured_against_its_own_latest_day():
    """A metric last recorded a month ago is current at its last reading."""
    horizons = {"resting_heart_rate": date(2026, 7, 26)}

    assert _mark(_template(STALE), _facts(STALE), horizons=horizons)[0] is False


def test_unknown_metric_horizon_never_refuses():
    assert _mark(_template(STALE), _facts(STALE), horizons={})[0] is False


def test_an_already_refused_draft_is_left_alone():
    verification = {"ok": False, "reason": "digit outside placeholder"}
    fired, verification = _mark(_template(STALE), _facts(STALE),
                                verification=verification)

    assert fired is False
    assert verification == {"ok": False, "reason": "digit outside placeholder"}


def test_placeholders_outside_the_fact_set_are_not_read():
    assert _mark(_template(STALE), {})[0] is False


@pytest.mark.parametrize("period, end", [
    ("2026-08-03", date(2026, 8, 3)),
    ("2026-08-03:2026-08-09", date(2026, 8, 9)),
    ({"start": "2026-07-01", "end": "2026-07-31"}, date(2026, 7, 31)),
    # A block node's ``end`` can be its last bucket's START; the bucket
    # starts win, as in fact_template._period_label.
    ({"start": "2026-07-13", "end": "2026-08-03",
      "period_starts": ["2026-07-13", "2026-07-20", "2026-07-27",
                        "2026-08-03"]}, date(2026, 8, 9)),
    (["2026-08-01", "2026-08-02"], date(2026, 8, 2)),
    ("recent", None),
])
def test_fact_period_end(period, end):
    assert chat._fact_period_end(period) == end


def test_mutation_stubbed_marker_breaks_the_refusal(monkeypatch):
    monkeypatch.setattr(chat, "_mark_stale_window",
                        lambda *args, **kwargs: False)

    with pytest.raises(AssertionError):
        _assert_stale_window_refused()


def test_cause_is_closed_and_ranked_after_the_period_phrase():
    assert "stale_window" in chat.ASK_CAUSES
    ledger = [{"sequence": 1}]
    assert chat._ask_cause({"ok": False}, ledger=ledger, loop_outcomes=[],
                           stale_window=True) == "stale_window"
    assert chat._ask_cause({"ok": False}, ledger=ledger, loop_outcomes=[],
                           unsupported_period_phrase=True,
                           stale_window=True) == "unsupported_period_phrase"


def test_fallback_names_the_cause_without_the_draft_dates():
    text = chat._fallback_answer({
        "ok": False, "cause": "stale_window",
        "reason": "narration cites a period that ended before the recent "
                  "data: the latest period cited ends 2026-07-26",
    })

    assert "ended well before your most recent data" in text
    assert "2026-07-26" not in text


# --------------------------------------------------------------------- #
# Through the ask path, against a real vault.
# --------------------------------------------------------------------- #

def _ledger():
    ledger = []
    for sequence, period in enumerate((STALE, CURRENT), start=1):
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


def _ask(monkeypatch, vault, question, templates):
    conn = vault.connect()
    dbmod.init_db(conn)
    seed_metric(conn, "resting_heart_rate", "2026-07-01", [60] * 52)
    conn.close()
    replies = iter(["acknowledged", *templates])
    ledger = _ledger()
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *a, **k: next(replies))
    capture: list = []
    result = chat.answer_question(vault, question, as_of=AS_OF.isoformat(),
                                  capture=capture)
    return result, capture


def test_stale_draft_is_repaired_to_the_current_window(monkeypatch, vault):
    stale = "Your resting heart rate averaged {" + fact_template.fact_key(
        "resting_heart_rate", STALE, "mean") + "}."
    current = "Your resting heart rate averaged {" + fact_template.fact_key(
        "resting_heart_rate", CURRENT, "mean") + "}."

    result, capture = _ask(monkeypatch, vault,
                           "How has my resting heart rate been?",
                           [stale, current])

    assert capture[0]["verification"]["cause"] == "stale_window"
    assert capture[0]["verification"]["stale_window"]["gap_days"] == 26
    assert result["mode"] == "narration"
    assert result["verification"]["retry"] is True
    assert result["verification"]["cause"] == "ok"
    assert "63" in result["text"]


def test_a_question_naming_july_is_answered_by_the_july_window(
        monkeypatch, vault):
    stale = "Your resting heart rate averaged {" + fact_template.fact_key(
        "resting_heart_rate", STALE, "mean") + "}."

    result, capture = _ask(monkeypatch, vault,
                           "How was my resting heart rate in July?", [stale])

    assert len(capture) == 1
    assert result["mode"] == "narration"
    assert result["verification"]["cause"] == "ok"
