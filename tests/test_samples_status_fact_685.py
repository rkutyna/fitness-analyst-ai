"""Samples-status facts: a session whose minute data is not measurable.

consumer #685 (last Done-when line). ``get_block_structure``,
``get_impact_volume`` and ``get_run_form`` publish null plus a leaf
``samples`` of "not_synced" or "not_recorded" for such a session or day. Under
the closed fact set (HA_ASK_FACT_TEMPLATE) nothing carried that state, so a
turn could only say "0 min" or "no data". These tests pin the two status facts
that carry it, their exact user-facing sentences, their guidance, and the
end-to-end narration path with the model stubbed.
"""
from __future__ import annotations

from health_advisor import chat, fact_template, llm, steering

NOT_SYNCED_KEY = "status:samples_not_synced"
NOT_RECORDED_KEY = "status:samples_not_recorded"
NOT_SYNCED_SENTENCE = (
    "That session's minute-by-minute data hasn't synced to your vault yet, so "
    "its continuous blocks and jog minutes can't be measured. They'll be "
    "measured once your watch finishes syncing.")
NOT_RECORDED_SENTENCE = (
    "That session was recorded indoors or on a machine, which records no "
    "minute-by-minute distance, so its continuous blocks and jog minutes can't "
    "be measured.")


def _record(result, *, sequence=1, tool_name="get_block_structure"):
    return {"sequence": sequence, "tool_name": tool_name,
            "result_elided": False, "result": result}


def _nested_session_record(samples):
    """A session nested under ``sessions[0]``, as get_block_structure returns."""
    return _record({"sessions": [{"date": "2026-08-01", "samples": samples,
                                  "continuous_blocks": None,
                                  "jog_minutes": None}]})


def _top_level_record(samples):
    return _record({"date": "2026-08-01", "samples": samples,
                    "continuous_blocks": None})


# --- detection: any depth, the right key, the right sentence ---------------

def test_not_synced_nested_in_sessions_publishes_the_exact_sentence():
    facts = fact_template.build_samples_status_facts(
        [_nested_session_record("not_synced")])
    assert set(facts) == {NOT_SYNCED_KEY}
    assert facts[NOT_SYNCED_KEY] == {
        "key": NOT_SYNCED_KEY, "display": NOT_SYNCED_SENTENCE,
        "value": "not_synced"}


def test_not_synced_at_top_level_is_detected_too():
    facts = fact_template.build_samples_status_facts(
        [_top_level_record("not_synced")])
    assert facts[NOT_SYNCED_KEY]["display"] == NOT_SYNCED_SENTENCE


def test_not_recorded_publishes_the_other_fact_only():
    facts = fact_template.build_samples_status_facts(
        [_nested_session_record("not_recorded")])
    assert set(facts) == {NOT_RECORDED_KEY}
    assert facts[NOT_RECORDED_KEY]["display"] == NOT_RECORDED_SENTENCE
    assert facts[NOT_RECORDED_KEY]["value"] == "not_recorded"


def test_both_statuses_in_one_ledger_publish_both_facts():
    facts = fact_template.build_samples_status_facts(
        [_nested_session_record("not_synced"),
         _top_level_record("not_recorded")])
    assert set(facts) == {NOT_SYNCED_KEY, NOT_RECORDED_KEY}


def test_ledger_with_neither_status_publishes_nothing():
    ledger = [_record({"sessions": [{"date": "2026-08-01",
                                     "samples": "ok",
                                     "continuous_blocks": [{"min": 12}]}]})]
    assert fact_template.build_samples_status_facts(ledger) == {}
    assert fact_template.samples_status_guidance({}) == ""


def test_elided_record_is_not_walked():
    elided = {"sequence": 1, "tool_name": "get_block_structure",
              "result_elided": True,
              "result": {"sessions": [{"samples": "not_synced"}]}}
    assert fact_template.build_samples_status_facts([elided]) == {}


# --- the sentences are answers, not instructions ---------------------------

def test_sentences_contain_no_digits_and_no_tool_names():
    for sentence in (NOT_SYNCED_SENTENCE, NOT_RECORDED_SENTENCE):
        assert not any(ch.isdigit() for ch in sentence)
        assert "get_" not in sentence
        assert steering.leak(sentence) is None


# --- guidance and registration ---------------------------------------------

def test_guidance_names_the_key_and_forbids_a_zero():
    facts = fact_template.build_samples_status_facts(
        [_nested_session_record("not_synced")])
    guidance = fact_template.samples_status_guidance(facts)
    assert guidance
    assert "SAMPLES STATUS" in guidance
    assert "{%s}" % NOT_SYNCED_KEY in guidance
    assert "never state a 0" in guidance


def test_guidance_names_each_key_present():
    facts = fact_template.build_samples_status_facts(
        [_nested_session_record("not_synced"),
         _top_level_record("not_recorded")])
    guidance = fact_template.samples_status_guidance(facts)
    assert "{%s}" % NOT_SYNCED_KEY in guidance
    assert "{%s}" % NOT_RECORDED_KEY in guidance


def test_both_keys_are_status_facts():
    facts = fact_template.build_samples_status_facts(
        [_nested_session_record("not_synced"),
         _top_level_record("not_recorded")])
    for key, fact in facts.items():
        assert fact_template.is_status_fact(key, fact) is True


def test_guidance_strings_are_registered_in_steering_texts():
    guidance = fact_template.samples_status_guidance(
        fact_template.build_samples_status_facts(
            [_nested_session_record("not_synced"),
             _top_level_record("not_recorded")]))
    for line in guidance.splitlines():
        assert line in fact_template.STEERING_TEXTS
    assert len(guidance.splitlines()) == 2


# --- end to end: the model narrates the status, the gate passes ------------

ASK = "How did my runs go this week?"
AS_OF = "2026-08-21"


def test_end_to_end_status_fact_is_narrated_and_passes_the_gate(
        monkeypatch, vault, conn):
    from tests.conftest import seed_metric
    seed_metric(conn, "step_count", "2026-06-01", [8000.0] * 60)
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = [_nested_session_record("not_synced")]
    prompts = []
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    replies = iter(["acknowledged", "{%s}" % NOT_SYNCED_KEY])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **k: prompts.append(prompt) or next(replies))

    out = chat.answer_question(vault, ASK, as_of=AS_OF, capture=[])

    assert "SAMPLES STATUS" in prompts[-1]
    assert NOT_SYNCED_KEY in prompts[-1]
    assert out["mode"] == "narration"
    assert out["verification"]["cause"] == "ok"
    assert NOT_SYNCED_SENTENCE in out["text"]
