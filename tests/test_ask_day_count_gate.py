"""#16 — the day-count check wired into the /v1/ask publishing path.

Every figure, date and weekday here is INVENTED. The two real failures came
from one person's real health data in a private repo; this engine repo is
public, so only the SHAPE of the defect is reproduced.

`tests/test_day_count_check.py` already pins what `agents.day_count_check`
decides. These tests pin something different and narrower: that a decision it
makes actually withholds an answer from `/v1/ask`, under the cause that names
the defect, on all four attempts — and, just as importantly, that a correct
answer still publishes.

The house failure mode this file is written against: "a self-contradicting
answer was withheld" passes trivially if the answer was withheld for some other
reason, and a gate that refuses everything passes it too. So every refusal test
asserts on the CAUSE, not the mode, and every refusal fixture is paired with a
control that differs by one word — the stated count — and must publish.
"""
from __future__ import annotations

import pytest

from health_advisor import chat
from health_advisor import fact_template
from health_advisor import llm


QUESTION = "How often did I cycle last week?"
AS_OF = "2031-03-09"

# One invented week of invented riding. The period is what fact keys are built
# from, so it has to agree with the ledger the arm reads.
PERIOD = "2031-03-02:2031-03-08"


def _ride_ledger() -> list[dict]:
    return [{
        "sequence": 1,
        "tool_name": "get_impact_volume",
        "arguments": {"start": "2031-03-02", "end": "2031-03-08",
                      "by": "week"},
        "result": {"metric": "ride_minutes", "period": PERIOD,
                   "ride_minutes": 77.5, "unit": "min"},
    }]


# The defect: the count says three, the same sentence itemises two. Digit-free
# on purpose — the point under test is the day count, and a fixture that also
# trips the figure-grounding gate could not tell the two refusals apart.
CONTRADICTING = "You cycled on three days last week: Tuesday and Friday."

# The control, one word away: the count now matches the itemisation. Anything
# that refuses this refuses a correct answer.
CONSISTENT = "You cycled on two days last week: Tuesday and Friday."

# The other direction of correctness: a third itemised day makes "three" right.
CONSISTENT_THREE = (
    "You cycled on three days last week: Tuesday, Friday and Saturday.")

# Fails the FIGURE gate, not this one — an unclaimed number. Used to drive the
# prose arm into its retry without touching the day-count gate.
UNGROUNDED = "You cycled 41.0 minutes last week."


def _prose_arm(monkeypatch, vault, answers, *, judge=95):
    """Drive the legacy prose arm with a scripted sequence of model answers."""
    replies = iter(answers)
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "0")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: _ride_ledger())
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: judge)
    capture: list[dict] = []
    result = chat.answer_question(vault, QUESTION, as_of=AS_OF,
                                  capture=capture)
    return result, capture


def _template_arm(monkeypatch, vault, templates):
    """Drive the closed-fact template arm; the first reply is the gather turn."""
    replies = iter(["acknowledged", *templates])
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: _ride_ledger())
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: next(replies))
    capture: list[dict] = []
    result = chat.answer_question(vault, QUESTION, as_of=AS_OF,
                                  capture=capture)
    return result, capture


def _ride_key() -> str:
    return fact_template.fact_key("ride_minutes", PERIOD, "ride_minutes")


def _template(sentence: str) -> str:
    """A compliant template: every digit lives inside a placeholder."""
    return "You logged {" + _ride_key() + "}. " + sentence


# --------------------------------------------------------------------------
# The closed vocabulary.
# --------------------------------------------------------------------------

def test_the_cause_is_in_the_closed_ask_vocabulary():
    """A cause the taxonomy does not contain is a cause nothing can chart."""
    assert "contradicted_day_count" in chat.ASK_CAUSES


def test_the_cause_outranks_the_generic_gate_verdict_but_not_a_denial():
    """Pin the precedence, because both new orderings are silent if wrong.

    A refused answer is not-ok either way, so `gate_refused` would be returned
    for a day-count refusal if the new branch sat below it — the mode would be
    identical and only the cause would be wrong.
    """
    refused = {"ok": False}
    ledger = _ride_ledger()

    assert chat._ask_cause(refused, ledger=ledger, loop_outcomes=[],
                           contradicted_day_count=True) == \
        "contradicted_day_count"
    assert chat._ask_cause(refused, ledger=ledger, loop_outcomes=[],
                           denied_available_figure=True,
                           contradicted_day_count=True) == \
        "denied_available_figure"
    assert chat._ask_cause(refused, ledger=ledger,
                           loop_outcomes=[]) == "gate_refused"


# --------------------------------------------------------------------------
# The marker: what it writes, and what it refuses to overwrite.
# --------------------------------------------------------------------------

def test_the_refusal_reason_carries_the_finding_detail():
    """A label alone is not auditable; the numbers and the span must survive."""
    verification = {"ok": True, "grounded": True, "reason": ""}

    assert chat._mark_contradicted_day_count(verification,
                                             text=CONTRADICTING) is True

    assert verification["ok"] is False
    assert verification["grounded"] is False
    reason = verification["reason"]
    assert reason.startswith("narration contradicts its own day itemisation")
    assert "stated 3" in reason
    assert "itemised 2" in reason
    assert "three days last week: Tuesday and Friday" in reason
    finding = verification["day_count"]["findings"][0]
    assert (finding["stated"], finding["itemised"]) == (3, 2)
    assert finding["span"] == "three days last week: Tuesday and Friday"


def test_a_consistent_answer_is_left_completely_alone():
    verification = {"ok": True, "grounded": True, "reason": ""}

    assert chat._mark_contradicted_day_count(verification,
                                             text=CONSISTENT) is False

    assert verification == {"ok": True, "grounded": True, "reason": ""}


def test_the_gate_does_not_steal_an_earlier_refusals_reason():
    """Precedence is deliberate: the first refusal keeps the repair detail.

    Without this, a template refused for an unresolved placeholder would be
    reported as a day-count contradiction and the retry would be told to fix
    the wrong thing.
    """
    verification = {"ok": False, "grounded": False,
                    "reason": "digit outside placeholder"}

    assert chat._mark_contradicted_day_count(verification,
                                             text=CONTRADICTING) is False

    assert verification["reason"] == "digit outside placeholder"
    assert "day_count" not in verification


# --------------------------------------------------------------------------
# Prose arm — attempt 1.
# --------------------------------------------------------------------------

def test_prose_first_attempt_contradiction_is_withheld_under_its_own_cause(
        monkeypatch, vault, conn):
    result, capture = _prose_arm(monkeypatch, vault,
                                 [CONTRADICTING, CONTRADICTING])

    # Not merely "not published": published under this cause, with the detail.
    assert capture[0]["verification"]["cause"] == "contradicted_day_count"
    assert "stated 3" in capture[0]["verification"]["reason"]
    assert result["mode"] == "fallback"
    assert CONTRADICTING not in result["text"]
    assert result["verification"]["cause"] == "contradicted_day_count"


def test_prose_publishes_the_control_that_differs_only_in_the_count(
        monkeypatch, vault, conn):
    """The anti-vacuity pin for the test above: one word apart, and it ships.

    If this fixture stopped publishing, the refusal test above would still
    pass while the gate had become worthless.
    """
    result, capture = _prose_arm(monkeypatch, vault, [CONSISTENT])

    assert result["mode"] == "narration"
    assert result["text"] == CONSISTENT
    assert result["verification"]["cause"] == "ok"
    assert "day_count" not in result["verification"]
    assert [entry["attempt"] for entry in capture] == [1]


def test_prose_publishes_a_three_day_count_that_itemises_three_days(
        monkeypatch, vault, conn):
    """The gate must read the itemisation, not the word "three"."""
    result, _ = _prose_arm(monkeypatch, vault, [CONSISTENT_THREE])

    assert result["mode"] == "narration"
    assert result["verification"]["cause"] == "ok"


# --------------------------------------------------------------------------
# Prose arm — the retry. The hole a first-attempt-only wiring would leave.
# --------------------------------------------------------------------------

def test_prose_retry_contradiction_is_withheld(monkeypatch, vault, conn):
    """A contradiction the REPAIR turn introduces is refused the same way.

    Attempt 1 fails on an unclaimed figure, so the day-count gate is provably
    not what refused it — that is what rules out "withheld for another
    reason" for attempt 2's verdict, which is recorded separately.
    """
    result, capture = _prose_arm(monkeypatch, vault,
                                 [UNGROUNDED, CONTRADICTING])

    assert capture[0]["verification"]["cause"] == "gate_refused"
    assert capture[0]["verification"]["unsupported"] == ["41.0"]
    assert capture[1]["verification"]["cause"] == "contradicted_day_count"
    assert "stated 3" in capture[1]["verification"]["reason"]
    assert result["mode"] == "fallback"
    assert CONTRADICTING not in result["text"]


def test_prose_retry_still_publishes_a_consistent_repair(monkeypatch, vault,
                                                         conn):
    """The control for the retry test: the same script, a correct repair.

    Without this, the retry test above would pass on an arm that simply never
    publishes anything after a failed first attempt.
    """
    result, capture = _prose_arm(monkeypatch, vault, [UNGROUNDED, CONSISTENT])

    assert capture[0]["verification"]["cause"] == "gate_refused"
    assert result["mode"] == "narration"
    assert result["text"] == CONSISTENT
    assert result["verification"]["retry"] is True
    assert result["verification"]["cause"] == "ok"


def test_prose_retry_feedback_names_the_count_and_the_itemisation(
        monkeypatch, vault, conn):
    """The refusal detail has to reach the model, or the repair is a guess."""
    prompts: list[str] = []
    replies = iter([CONTRADICTING, CONTRADICTING])
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "0")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: _ride_ledger())
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: prompts.append(prompt) or next(replies))
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: 95)

    chat.answer_question(vault, QUESTION, as_of=AS_OF)

    assert "contradicts its own day itemisation" in prompts[1]
    assert "stated 3, itemised 2" in prompts[1]


# --------------------------------------------------------------------------
# Fact-template arm — both attempts.
# --------------------------------------------------------------------------

def test_template_first_attempt_contradiction_is_withheld(monkeypatch, vault,
                                                           conn):
    result, capture = _template_arm(
        monkeypatch, vault,
        [_template(CONTRADICTING), _template(CONTRADICTING)])

    assert capture[0]["verification"]["cause"] == "contradicted_day_count"
    assert "stated 3" in capture[0]["verification"]["reason"]
    assert result["mode"] == "fallback"
    assert "Tuesday and Friday" not in result["text"]
    assert result["verification"]["cause"] == "contradicted_day_count"


def test_template_publishes_the_control_that_differs_only_in_the_count(
        monkeypatch, vault, conn):
    result, _ = _template_arm(monkeypatch, vault, [_template(CONSISTENT)])

    assert result["mode"] == "narration"
    assert result["verification"]["cause"] == "ok"
    assert CONSISTENT in result["text"]
    # The interpolated figure is Python's, and it is still there — the gate
    # refuses answers, it never edits one.
    assert "77.5" in result["text"]


def test_template_retry_contradiction_is_withheld(monkeypatch, vault, conn):
    """Attempt 1 fails on a bare digit; attempt 2 fails on the day count.

    Same construction as the prose retry test, and for the same reason: the
    two attempts must fail for provably different causes, or this test could
    pass on a first-attempt-only wiring.
    """
    result, capture = _template_arm(
        monkeypatch, vault,
        ["You cycled 3 times last week.", _template(CONTRADICTING)])

    assert capture[0]["verification"]["cause"] == "gate_refused"
    assert capture[1]["verification"]["cause"] == "contradicted_day_count"
    assert "stated 3" in capture[1]["verification"]["reason"]
    assert result["mode"] == "fallback"
    assert "Tuesday and Friday" not in result["text"]


def test_template_retry_still_publishes_a_consistent_repair(monkeypatch, vault,
                                                            conn):
    result, capture = _template_arm(
        monkeypatch, vault,
        ["You cycled 3 times last week.", _template(CONSISTENT)])

    assert capture[0]["verification"]["cause"] == "gate_refused"
    assert result["mode"] == "narration"
    assert CONSISTENT in result["text"]
    assert result["verification"]["cause"] == "ok"


# --------------------------------------------------------------------------
# The invariant the whole issue turns on.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("answer", [
    CONSISTENT,
    CONSISTENT_THREE,
    "You cycled on three key days last week: Tuesday's intervals, Friday's "
    "tempo, and the long ride.",
    "Aim for three riding days next week: Tuesday and Friday are your usual "
    "slots.",
    "You rode more this week than in any of the last three weeks: Tuesday and "
    "Friday were the sessions.",
])
def test_ordinary_coaching_prose_still_publishes(monkeypatch, vault, conn,
                                                 answer):
    """0 false positives in 189 is the entire value; a gate that refuses good
    answers destroys it. These are the idiom shapes nearest the defect."""
    result, _ = _prose_arm(monkeypatch, vault, [answer])

    assert result["mode"] == "narration", result["verification"]
    assert result["verification"]["cause"] == "ok"
