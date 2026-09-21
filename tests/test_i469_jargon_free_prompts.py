"""Model-facing refusal/repair prompts must read as plain English.

health_advisor#469: a fact-template refusal built from internal vocabulary
("the closed fact set has no citable metric leaves for it") reached the model,
which paraphrased it straight into a user-visible answer ("neither provides a
citable metric leaf"). Python owns the prose the user eventually sees only
indirectly here — it owns the *evidence*, but the model chooses words, and a
prompt written in engineering jargon is a paraphrase away from surfacing that
jargon to the user. These strings must stay legible without the codebase's
internal vocabulary.
"""
from __future__ import annotations

from health_advisor import chat

_BANNED_TERMS = ("leaf", "leaves", "closed fact set", "citable")


def _assert_jargon_free(text: str) -> None:
    lowered = text.lower()
    for term in _BANNED_TERMS:
        assert term not in lowered, f"{term!r} found in model-facing text: {text!r}"


def test_unused_fact_prompt_with_gathered_data_is_jargon_free():
    # No published facts, but one tool call returned real data — the exact
    # shape that produced the leak (#469).
    ledger = [{
        "sequence": 1,
        "tool_name": "list_workouts",
        "result": {"workouts": [{"date": "2001-01-05", "duration_min": 35.1}]},
        "result_elided": False,
    }]
    text = chat._unused_fact_prompt({}, ledger)
    _assert_jargon_free(text)
    # Still informative: names the tool whose data went unused.
    assert "list_workouts" in text


def test_unused_fact_prompt_with_absent_facts_is_jargon_free():
    facts = {"fact|metric=jog_minutes|period=s:2001-01-05|field=value": {}}
    text = chat._unused_fact_prompt(facts, [])
    _assert_jargon_free(text)


def test_unused_fact_prompt_with_nothing_gathered_is_jargon_free():
    text = chat._unused_fact_prompt({}, [])
    _assert_jargon_free(text)


def test_retry_repair_instructions_constant_is_jargon_free():
    _assert_jargon_free(chat._RETRY_REPAIR_INSTRUCTIONS)


def test_retry_feedback_named_metric_is_jargon_free():
    verification = {
        "reason": "unsupported claim",
        "unsupported": [],
        "verdict": {"numbers": [{
            "ok": False, "claimed": 35.1,
            "reason": "claim value does not match ledger field",
            "path": "$.result.workouts[0].duration_min",
            "actual": 48.3, "actual_field": "duration_min",
            "actual_metric": "jog_minutes",
        }]},
    }
    _assert_jargon_free(chat._retry_feedback(verification))


def test_retry_feedback_unlabelled_row_is_jargon_free():
    verification = {
        "reason": "unsupported claim",
        "unsupported": [],
        "verdict": {"numbers": [{
            "ok": False, "claimed": 35.1,
            "reason": "claim metric does not match published field",
            "path": "$.result.workouts[0].duration_min",
            "actual_field": "duration_min", "actual_metric": None,
        }]},
    }
    _assert_jargon_free(chat._retry_feedback(verification))
