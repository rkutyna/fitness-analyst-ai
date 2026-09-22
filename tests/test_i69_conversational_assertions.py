"""#69 item 3: the conversational gate refuses claims, not metric names.

A no-ledger conversational reply is exempt from numeric verification because
it states nothing about the vault. The old rule refused any canonical metric
spelling, and measured on the deployed shape it refused seven capability menus
out of seven refusals while publishing "You ran twelve miles on Saturday.".
The oracle below was written before the rule, independently of it; every line
is synthetic or a recorded model reply that carries no user data.
"""
from __future__ import annotations

import pytest

from health_advisor import chat, fact_template, llm


# Recorded verbatim in #69's measurement comment: two of the seven refused
# capability menus (the issue quotes these two in full; the other five were
# not reproduced in the tracker). Both were refused by the metric-name ban.
RECORDED_MENUS = (
    "Hello! I'm ready to help. Go ahead and ask me anything about your health "
    "and training data — sleep, heart rate, running form, training load, "
    "nutrition, and more. What would you like to look into?",
    "Recovery — resting heart rate, sleep regularity, readiness, training "
    "load (ACWR)",
)

MUST_ADMIT = RECORDED_MENUS + (
    # The hard two: "your <metric>" in a capability sense.
    "Hi! I can help you look at your sleep, heart rate, runs and training "
    "plan. What would you like to know?",
    "Hello! Ask me about your resting heart rate, your weekly running volume, "
    "or how last night's sleep went.",
    "You're welcome — happy to help anytime.",
    "Got it. Let me know if you want to look at anything in your data.",
)

NUMBER_WORD_CLAIMS = (
    "You ran twelve miles on Saturday.",
    "Your resting heart rate is fifty-two beats per minute.",
    "You slept seven hours last night.",
)
DIGIT_CLAIMS = (
    "Great job on your 5K!",
    "Your VO2 max is improving.",
)
WORDLESS_CLAIMS = (
    "Your resting heart rate has been dropping, which is a great sign.",
    "Your sleep was better last night than the night before.",
    "You've been running more this week.",
    "Your heart rate looks normal.",
    "You hit a new personal best on your run.",
)
MUST_REFUSE = NUMBER_WORD_CLAIMS + DIGIT_CLAIMS + WORDLESS_CLAIMS


@pytest.mark.parametrize("reply", MUST_ADMIT)
def test_capability_and_courtesy_replies_are_admitted(reply):
    assert fact_template.conversational_violation(reply, {}) == ""


@pytest.mark.parametrize("reply", MUST_REFUSE)
def test_claims_about_the_user_are_refused(reply):
    assert fact_template.conversational_violation(reply, {}) != ""


@pytest.mark.parametrize("reply", NUMBER_WORD_CLAIMS)
def test_spelled_figures_are_refused_as_numbers(reply):
    # Its own layer, independent of the assertion rule below.
    assert fact_template.number_words(reply)
    assert (fact_template.conversational_violation(reply, {})
            == "number word outside placeholder")


@pytest.mark.parametrize("reply", NUMBER_WORD_CLAIMS + WORDLESS_CLAIMS)
def test_assertion_rule_refuses_every_digit_free_claim_on_its_own(reply):
    # Defence in depth: the number-word lines are ALSO claims by shape, so
    # neither layer is the only thing standing between them and publication.
    assert fact_template.conversational_assertion(reply, {}) != ""


def test_a_number_word_is_refused_without_any_claim_shape():
    reply = "Hello! Ask me about the last twelve weeks."
    assert fact_template.conversational_assertion(reply, {}) == ""
    assert (fact_template.conversational_violation(reply, {})
            == "number word outside placeholder")


@pytest.mark.parametrize("reply", (
    "If you slept badly, ask me about recovery.",
    "Let me know what you decided about the plan.",
    "Have a great week!",
    "Keep up the great work!",
    "One of the best places to start is your sleep.",
    "Take a couple of days easy if you like.",
))
def test_hypotheticals_wishes_and_vague_quantifiers_are_admitted(reply):
    assert fact_template.conversational_violation(reply, {}) == ""


@pytest.mark.parametrize("text,expected", (
    ("twelve miles", ["twelve"]),
    ("fifty-two beats", ["fifty"]),
    ("one hundred and five", ["hundred", "five"]),
    ("a dozen runs", ["dozen"]),
    ("on the twelfth", ["twelfth"]),
    ("one mile", ["one mile"]),
    ("an hour and a half", ["and a half"]),
    ("twice this week", ["twice"]),
))
def test_number_words_finds_cardinals_ordinals_and_fractions(text, expected):
    assert fact_template.number_words(text) == expected


@pytest.mark.parametrize("text", (
    "one of your runs", "once you have synced", "a couple of days",
    "first, the second half of the week", "your half-marathon plan",
    "the six minute walk test distance", "zero in on sleep",
    "someone, anyone, tone, often, weight",
))
def test_number_words_leaves_pronouns_idioms_and_metric_names_alone(text):
    assert fact_template.number_words(text) == []


def test_ledger_backed_narration_does_not_run_the_assertion_rule():
    # Narration asserts facts by design, through placeholders. Pinned so the
    # conversational rule cannot leak into scan_template unnoticed.
    ledger = [{
        "sequence": 1, "tool_name": "synthetic_metric", "arguments": {},
        "result": {
            "metric": "jog_minutes", "unit": "min", "period": "2026-08-17",
            "mean": 50.1,
            "presentation": {"metric": "jog_minutes", "period": "2026-08-17",
                             "field": "presentation", "value": "50 m"},
        },
    }]
    facts = fact_template.build_fact_set(ledger)
    key = fact_template.fact_key("jog_minutes", "2026-08-17", "mean")
    template = "Your jog minutes were {" + key + "}, and you ran well."
    assert fact_template.conversational_assertion(template, facts) != ""
    assert fact_template.scan_template(template, facts)["ok"] is True


def _conversational(monkeypatch, vault, reply):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: reply)
    return chat.answer_question(vault, "hi there")


@pytest.mark.parametrize("reply", RECORDED_MENUS)
def test_recorded_menus_publish_on_the_deployed_shape(monkeypatch, vault,
                                                      reply):
    result = _conversational(monkeypatch, vault, reply)
    assert result["mode"] == "narration"
    assert result["verification"]["cause"] == "conversational"
    assert result["text"] == reply


@pytest.mark.parametrize("reply", (
    "You ran twelve miles on Saturday.",
    "Your heart rate looks normal.",
))
def test_claims_are_withheld_on_the_deployed_shape(monkeypatch, vault, reply):
    result = _conversational(monkeypatch, vault, reply)
    assert result["mode"] == "fallback"
    assert result["verification"]["cause"] == "conversational_refused"
    assert reply not in result["text"]
