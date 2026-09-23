"""The health_advisor#483 oracle for the advice-slot claim rule.

This pins the exact MUST-ADMIT / MUST-REFUSE sentences from the issue body,
built BEFORE any rule change so it can score the old and the new rule
independently. The advice slot (``{advice:...}``) skips numeric verification,
so the only thing standing between it and an unverified claim about the
user's data is ``fact_template._advice_violation``.

The rule under test was, until engine #66, "does the span contain the word
`your`, or name a vault metric" -- a pronoun-and-noun test. #66/#483 replaced
it with ``conversational_assertion`` (#69's claim-shaped test, reused rather
than re-invented per health_advisor#483's routing note): a claim needs a
subject AND a predicate, not just a pronoun or a noun.
"""
from __future__ import annotations

import re

import pytest

from health_advisor import fact_template


# --- The oracle, verbatim from health_advisor#483's Done-when 2 -------------
#
# Advice that asserts nothing about the user's own data, however naturally
# second-person, must be admitted.
MUST_ADMIT = [
    "Ease back your mileage for a week and see if the ache settles.",
    "Check the outer edge of your shoes for wear.",
    "If the pain sharpens or lingers into the next day, see a physio.",
    # Review, 2026-09-23: the ordinary shapes of pain advice. A rule that
    # refused "you have <word>" to catch a diagnosis refused these too, and
    # the same predicate guards greetings ("You have a coach here...").
    "You have a few options: rest, change shoes, or see a podiatrist.",
    "If you have pain that lasts more than two weeks, see a physiotherapist.",
    "You have to ease back your mileage for a week.",
]

# A claim about the user's data -- a figure, a trend, or a diagnosis -- must
# be refused, "your" or no "your". health_advisor#483's Done-when 4 (the
# coaching-stance check) adds the diagnosis and figure-in-prose shapes: the
# fix must not let the model assert a figure, a trend or a diagnosis about
# the user even when phrased as if it were advice.
MUST_REFUSE = [
    "Your cadence is too low, which is why your feet hurt.",
    "Your mileage jumped 40% this week.",
    "You ran 30 miles last week.",
    # Done-when 4: stance lines a fix must not admit.
    "Your pace dropped this week.",
    "Your mileage is too high.",
]

# A present-tense diagnosis carries no digit, no metric name and no past-tense
# verb, so nothing lexical separates it from "You have a few options". It is a
# KNOWN GAP, pinned so it cannot be forgotten. The stance against diagnosing
# lives in the prompt, not in this predicate. Note also that scan_template
# runs this rule only on advice spans that carry a digit, so a digit-free
# diagnosis never reaches it in the advice slot anyway.
KNOWN_GAP = [
    "You have plantar fasciitis.",
]


def _old_pronoun_and_noun_rule(content: str) -> str:
    """The pre-#66 rule: refuse `\\byour\\b`, or any canonical metric name.

    Reproduced here (not imported) because engine #66 replaced it outright;
    this is what ``_advice_violation`` used to be, kept only so the oracle can
    score the "before" state without reverting the real module. Metric names
    are left out of the reproduction because none of the oracle's lines name
    one -- the historical rule's metric arm is orthogonal to what this issue
    is about.
    """
    if re.search(r"\byour\b", content, re.IGNORECASE):
        return "advice slot references the user's own data"
    return ""


def test_before_the_fix_the_bare_your_rule_fails_the_oracle_both_ways():
    """The state this issue reports: measure before touching anything.

    The old rule is wrong in both directions on this oracle: it refuses every
    MUST-ADMIT line that says "your" (plain second-person coaching) and admits
    a MUST-REFUSE claim about the user's data that happens not to say "your",
    the dangerous miss, since that is exactly what the advice-slot exemption
    must not carry.
    """
    wrongly_refused = [s for s in MUST_ADMIT if _old_pronoun_and_noun_rule(s)]
    wrongly_admitted = [s for s in MUST_REFUSE
                         if not _old_pronoun_and_noun_rule(s)]

    assert wrongly_refused == [
        "Ease back your mileage for a week and see if the ache settles.",
        "Check the outer edge of your shoes for wear.",
        "You have to ease back your mileage for a week.",
    ]
    assert wrongly_admitted == [
        "You ran 30 miles last week.",
    ]


def test_after_the_fix_every_must_admit_line_is_admitted():
    for line in MUST_ADMIT:
        violation = fact_template._advice_violation(line, None)
        assert violation == "", (line, violation)


def test_after_the_fix_every_must_refuse_line_is_refused():
    for line in MUST_REFUSE:
        violation = fact_template._advice_violation(line, None)
        assert violation != "", line


@pytest.mark.xfail(strict=True, reason="known gap: a digit-free present-tense "
                   "diagnosis is lexically indistinguishable from 'You have a "
                   "few options'; see the KNOWN_GAP comment")
def test_known_gap_a_bare_diagnosis_is_not_refused():
    for line in KNOWN_GAP:
        assert fact_template._advice_violation(line, None) != "", line


def test_advice_slot_end_to_end_matches_the_oracle():
    """The same oracle through the public surface the model's draft hits.

    ``scan_template`` only runs ``_advice_violation`` on advice content that
    contains a digit -- a digit-free span is unwrapped as plain prose instead
    (a separate, already-shipped decision, 2026-08-31: it was never going to
    trip "digit outside placeholder" either way, so the slot earns it no
    exemption). None of this oracle's lines happen to carry a digit, so this
    end-to-end check uses digit-bearing paraphrases that keep each line's
    claim shape, to prove the full ``scan_template``/``interpolate_template``
    plumbing invokes the rule when the model actually needs the exemption.
    The digit-free lines themselves are pinned directly against
    ``_advice_violation`` above, which is what the issue calls "the rule".
    """
    facts = fact_template.build_fact_set([])
    admit_with_digits = [
        "Ease back your mileage by 10% for a week and see if the ache "
        "settles.",
        "Check the outer edge of your shoes for wear every 2 weeks.",
    ]
    refuse_with_digits = [
        "Your mileage jumped 40% this week.",
        "You ran 30 miles last week.",
    ]

    for line in admit_with_digits:
        template = "{advice:" + line + "}"
        scan = fact_template.scan_template(template, facts)
        assert scan["ok"] is True, (line, scan.get("reason"))
        assert fact_template.interpolate_template(template, facts) == line

    for line in refuse_with_digits:
        template = "{advice:" + line + "}"
        scan = fact_template.scan_template(template, facts)
        assert scan["ok"] is False, line
        assert fact_template.interpolate_template(template, facts) is None


def test_mutation_reverting_to_the_bare_your_rule_reddens_the_oracle(
        monkeypatch):
    """Prove the pinned test depends on the real rule, not a static oracle.

    Swap ``_advice_violation`` back to the pre-#66 pronoun-and-noun test and
    show the MUST-ADMIT assertions go red -- the same mutation check
    health_advisor#483 asks for, run here instead of by editing the module
    and reverting it, so the tree never carries a broken intermediate state.
    """
    monkeypatch.setattr(
        fact_template, "_advice_violation",
        lambda content, facts=None: _old_pronoun_and_noun_rule(content))

    failures = [line for line in MUST_ADMIT
                if fact_template._advice_violation(line, None) != ""]

    assert failures == [
        "Ease back your mileage for a week and see if the ache settles.",
        "Check the outer edge of your shoes for wear.",
        "You have to ease back your mileage for a week.",
    ], "mutation did not redden the oracle -- the pinned test is vacuous"
