"""Publishing gates for Python-rendered units and empty paragraphs."""
from __future__ import annotations

from health_advisor import chat
from health_advisor import fact_template
from health_advisor import llm


PERIODS = (
    ("2026-08-17:2026-08-23", 88, "1 h 28 m"),
    ("2026-08-24:2026-08-30", 71, "1 h 11 m"),
    ("2026-08-31:2026-09-06", 86, "1 h 26 m"),
)


def _duration_ledger(rows=PERIODS):
    return [{
        "sequence": index,
        "tool_name": "synthetic_metric",
        "arguments": {},
        "result": {
            "metric": "jog_minutes",
            "period": period,
            "mean": value,
            "unit": "min",
            "presentation": {
                "metric": "jog_minutes",
                "period": period,
                "field": "presentation",
                "value": display,
            },
        },
    } for index, (period, value, display) in enumerate(rows, start=1)]


def _key(period):
    return fact_template.fact_key("jog_minutes", period, "mean")


def _template_arm(monkeypatch, vault, ledger, templates):
    replies = iter(["acknowledged", *templates])
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(replies))
    capture = []
    result = chat.answer_question(vault, "How did my recent duration change?",
                                  capture=capture)
    return result, capture


def test_rendered_duration_unit_restatement_retries_measured_shape(
        monkeypatch, vault):
    ledger = _duration_ledger()
    keys = [_key(period) for period, _, _ in PERIODS]
    labels = [fact_template.fact_key("jog_minutes", period, "period_label")
              for period, _, _ in PERIODS]
    bad = (
        f"During {{{labels[0]}}}, you logged {{{keys[0]}}} minutes for the "
        f"week, including {{{keys[1]}}} minutes in {{{labels[1]}}} and "
        f"{{{keys[2]}}} minutes in {{{labels[2]}}}."
    )
    good = (f"The three reported durations were {{{keys[0]}}}, {{{keys[1]}}}, "
            f"and {{{keys[2]}}}.")

    result, capture = _template_arm(monkeypatch, vault, ledger, [bad, good])

    assert capture[0]["verification"]["ok"] is False
    assert capture[0]["verification"]["cause"] == "restated_unit"
    assert "narration restates a rendered unit" in \
        capture[0]["verification"]["reason"]
    assert result["mode"] == "narration"
    assert result["verification"]["retry"] is True
    assert result["verification"]["template_compliant"] is True
    assert result["verification"]["figures_verified"] == 3
    assert result["verification"]["figures_total"] == 3
    assert result["verification"]["unsupported"] == []
    assert result["verification"]["tier_counts"] == {"path": 3,
                                                        "metric": None}
    assert "minutes" not in result["text"]


def test_rendered_unit_families_refuse_only_their_matching_words():
    cases = (
        ("1 m", "minutes"),
        ("1 h 28 m", "mins"),
        ("1 min", "minute"),
        ("1 h", "hrs"),
    )
    for display, word in cases:
        key = "fact|metric=fixture|period=s:period|field=mean"
        verification = {"ok": True, "grounded": True, "reason": ""}
        facts = {key: {"display": display}}

        assert chat._mark_restated_rendered_unit(
            verification, text=f"Value {display} {word}.",
            template=f"Value {{{key}}} {word}.", facts=facts)
        assert verification["reason"].startswith(
            "narration restates a rendered unit")


def test_pace_compound_is_not_mistaken_for_a_duration_unit():
    key = "fact|metric=fixture|period=s:period|field=pace"
    verification = {"ok": True, "grounded": True, "reason": ""}
    facts = {key: {"display": "14 min/mi"}}

    assert not chat._mark_restated_rendered_unit(
        verification, text=f"Pace {{{key}}} minutes per mile.",
        template=f"Pace {{{key}}} minutes per mile.", facts=facts)
    assert verification == {"ok": True, "grounded": True, "reason": ""}


def test_correct_duration_answer_keeps_grounding_unchanged(monkeypatch, vault):
    ledger = _duration_ledger(PERIODS[:1])
    key = _key(PERIODS[0][0])
    template = f"The logged duration was {{{key}}} overall."

    result, capture = _template_arm(monkeypatch, vault, ledger, [template])

    assert result["mode"] == "narration"
    assert result["verification"]["template_compliant"] is True
    assert result["verification"]["figures_verified"] == 1
    assert result["verification"]["figures_total"] == 1
    assert result["verification"]["tier_counts"] == {"path": 1,
                                                        "metric": None}
    assert result["verification"]["unsupported"] == []
    assert [entry["attempt"] for entry in capture] == [1]


def test_punctuation_only_paragraph_is_removed_before_publication(
        monkeypatch, vault):
    ledger = _duration_ledger(PERIODS[:1])
    key = _key(PERIODS[0][0])
    template = (f"The logged duration was {{{key}}}.\n\n.\n\n"
                "The record is complete.")

    result, _ = _template_arm(monkeypatch, vault, ledger, [template])

    assert result["mode"] == "narration"
    assert result["text"] == (
        "The logged duration was 1 h 28 m.\n\nThe record is complete."
    )
    assert "\n\n.\n\n" not in result["text"]
