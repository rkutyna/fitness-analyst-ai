"""An audit that could not run must never narrate as one that found nothing.

#368. `_audit_fallback` derived its sentence from `checks` alone, so a report
with `checks: []` — which is what an audit that could not run produces — said
"found no flagged checks", asserting that something was examined and nothing was
wrong. On a tester deployment the claims register is absent permanently and by
design, so that reassuring falsehood would have been narrated to every user on
every ask. It is composed in Python, so no grounding check, template gate or
claim scan sees it.
"""
from __future__ import annotations

import pytest

from health_advisor.chat import _audit_fallback


NAME = "conclusions_still_hold_mini"


def _report(checks, **extra):
    return {"name": NAME, "checks": checks, **extra}


def test_ran_and_found_nothing_still_says_so():
    text = _audit_fallback(_report([{"check_id": "weight_loss_slope", "flag": False}]))
    assert text == ("The deterministic conclusions-still-hold audit found no "
                    "flagged checks.")


def test_flagged_checks_are_named():
    text = _audit_fallback(_report([{"check_id": "weight_loss_slope", "flag": True}]))
    assert "flagged: weight_loss_slope" in text


@pytest.mark.parametrize("unavailable", [
    _report([], status="claims_register_unavailable",
            register_status="no claims register on this deployment"),
    _report([], status="claims_register_unavailable"),
    _report([]),
])
def test_an_audit_that_examined_nothing_never_claims_a_clean_result(unavailable):
    """The whole point: absence of checks is not evidence of absence of findings."""
    text = _audit_fallback(unavailable)
    assert "found no flagged checks" not in text
    assert "examined nothing" in text
    assert "neither confirmed nor contradicted" in text


def test_the_two_states_are_not_the_same_sentence():
    ran_clean = _audit_fallback(_report([{"check_id": "w", "flag": False}]))
    never_ran = _audit_fallback(_report([], status="claims_register_unavailable"))
    assert ran_clean != never_ran


def test_a_stated_reason_is_carried_verbatim_rather_than_invented():
    text = _audit_fallback(_report(
        [], register_status="no claims register on this deployment"))
    assert "no claims register on this deployment" in text
