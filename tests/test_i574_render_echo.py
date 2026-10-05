"""Two rendering defects that doubled text for a reader (consumer #574).

Obs 1. A unit rendered twice, in two spellings: the template wrote the fact's
own unit after the slot ("{vo2_max} mL/min·kg") and the renderer, which did not
recognise that spelling as the unit it was about to append, added its own
("N mL/min/kg mL/min·kg"). The renderer owns units, so the control is the
renderer recognising every spelling the engine itself uses for a unit, and the
suffix it appends is spelled exactly as the fact's ``unit``.

Obs 2. A status sentence rendered twice: the model copied a status slot's own
display text into its prose and then added the slot as well.

Every value here is synthetic.
"""
from __future__ import annotations

import pytest

from health_advisor import fact_template, metrics, normalize

WEEK = "2030-03-04:2030-03-10"


def _metric_facts(metric, **fields):
    ledger = [{"sequence": 1, "tool_name": "synthetic_summary", "arguments": {},
               "result": {"metric": metric, "period": WEEK, **fields}}]
    return fact_template.build_fact_set(ledger)


def _render(metric, field, value, tail, head="The figure is "):
    facts = _metric_facts(metric, **{field: value})
    key = "{" + fact_template.fact_key(metric, WEEK, field) + "}"
    return fact_template.interpolate_template(head + key + tail, facts)


# ---------------------------------------------------------------------------
# Obs 1: one spelling, and an echoed unit renders once
# ---------------------------------------------------------------------------

def test_the_vo2_max_suffix_is_spelled_as_the_facts_unit():
    fact = next(iter(_metric_facts("vo2_max", mean=41.25).values()))
    assert fact["unit"] == "mL/min·kg"
    suffix, _words = metrics.unit_suffix(fact["unit"], metric="vo2_max",
                                         field="mean")
    assert suffix == fact["unit"]
    assert _render("vo2_max", "mean", 41.25, ".") == (
        "The figure is 41.25 mL/min·kg.")


@pytest.mark.parametrize("spelling", [
    "mL/min·kg", "mL/min/kg", "mL/kg/min", "ml/kg/min",
    "ml/(kg·min)", "mL/(kg·min)",
])
def test_a_vo2_max_unit_written_after_the_slot_renders_once(spelling):
    # The exact shape of the live defect, then every spelling the engine has
    # used for this unit (normalize's table and the Apple export's).
    rendered = _render("vo2_max", "mean", 41.25,
                       f" {spelling}, based on a watch.",
                       head="Your estimate is ")
    assert rendered == f"Your estimate is 41.25 {spelling}, based on a watch."


@pytest.mark.parametrize("metric,field,value,spelling,display", [
    ("distance_walking_running", "mean", 5.24, "mi", "5.24"),
    ("distance_walking_running", "mean", 5.24, "miles", "5.24"),
    ("resting_heart_rate", "mean", 50, "bpm", "50"),
    ("resting_heart_rate", "mean", 50, "beats per minute", "50"),
    ("resting_heart_rate", "mean", 50, "count/min", "50"),
    ("active_energy", "mean", 94, "kcal", "94"),
    ("active_energy", "mean", 94, "calories", "94"),
    ("body_mass", "mean", 196.1, "lb", "196.1"),
])
def test_another_units_echo_in_any_engine_spelling_renders_once(
        metric, field, value, spelling, display):
    rendered = _render(metric, field, value, f" {spelling} overall.")
    assert rendered == f"The figure is {display} {spelling} overall."


@pytest.mark.parametrize("tail", [".", " overall.", ", then it fell."])
def test_a_template_with_no_trailing_unit_is_unchanged(tail):
    assert _render("vo2_max", "mean", 41.25, tail) == (
        f"The figure is 41.25 mL/min·kg{tail}")
    assert _render("body_mass", "mean", 196.1, tail) == (
        f"The figure is 196.1 lb{tail}")


def test_a_word_that_only_resembles_a_unit_is_not_swallowed():
    # "min" is a unit word, but not this fact's: the model's words stay and
    # nothing is stripped, exactly as before the echo guard knew compound forms.
    facts = _metric_facts("step_count", mean=14074)
    key = "{" + fact_template.fact_key("step_count", WEEK, "mean") + "}"
    rendered = fact_template.interpolate_template(
        f"You walked {key} min later than usual.", facts)
    assert "min later than usual." in rendered
    assert rendered.startswith("You walked 14,074")
    # And a compound that merely begins like this unit is not its echo: the
    # model's words stay and the unit is still stated once by Python.
    rendered = _render("vo2_max", "mean", 41.25, " mL/day later.")
    assert rendered == "The figure is 41.25 mL/min\u00b7kg mL/day later."


def test_a_compound_that_is_another_units_form_is_not_this_units_echo():
    # "mL/kg/min" after a body-mass slot is not a body-mass unit, and "lb"
    # is not appended after a unit word the template already wrote.
    rendered = _render("body_mass", "mean", 196.1, " mL/kg/min later.")
    assert rendered == "The figure is 196.1 mL/kg/min later."


def _echo_cases():
    cases = []
    for unit, (suffix, _words) in metrics._UNIT_SUFFIXES.items():
        field = "delta_pct" if unit == "%" else "mean"
        cases.append((unit, None, field))
    for metric in metrics._COUNT_NOUNS:
        cases.append(("count", metric, "mean"))
    for metric in metrics._RATE_NOUNS:
        cases.append(("count/min", metric, "mean"))
    return cases


@pytest.mark.parametrize("unit,metric,field", _echo_cases())
def test_every_engine_spelling_of_every_unit_is_recognised_as_an_echo(
        unit, metric, field):
    forms = metrics.unit_echo_forms(unit, metric=metric, field=field)
    suffix, _words = metrics.unit_suffix(unit, metric=metric, field=field)
    # The renderer's own suffix and the fact's own unit are always forms.
    assert suffix.lower() in forms
    if unit != "count" and not field.endswith("_pct"):
        assert unit.lower() in forms
    fact = {"key": "k", "metric": metric, "field": field, "unit": unit,
            "display": "12.5"}
    for form in sorted(forms):
        separator = "" if form == "%" else " "
        rendered = fact_template.interpolate_template(
            "Value {k}" + separator + form + ".", {"k": fact})
        assert rendered == "Value 12.5" + separator + form + ".", (unit, form)


# ---------------------------------------------------------------------------
# Obs 2: a status sentence the prose already carries renders once
# ---------------------------------------------------------------------------

def _cold_start_status_text_fact():
    entries = [
        {"path": "$.result.readiness.cold_start.status", "value": "partial"},
        {"path": "$.result.readiness.cold_start.status_text",
         "value": "Too few days have synced, so readiness starts a little later."},
    ]
    candidates = dict(fact_template._cold_start_entries(entries, sequence=1))
    key = "$.result.readiness.cold_start.status_text"
    return {key: candidates[key]}


def _status_kinds():
    kinds = [("gathered_data_uncited",
              fact_template.build_gathered_data_status_fact(
                  ledger_has_data=True, figures_published=False))]
    for status in [*fact_template._EVIDENCE_STATUS_SENTENCES, "other_reason"]:
        kinds.append((f"evidence_{status}",
                      fact_template.build_evidence_status_fact(status)))
    kinds.append(("cold_start_status_text", _cold_start_status_text_fact()))
    return kinds


STATUS_KINDS = _status_kinds()
KIND_IDS = [name for name, _facts in STATUS_KINDS]


def _only(facts):
    (key, fact), = facts.items()
    return key, fact


@pytest.mark.parametrize("name,facts", STATUS_KINDS, ids=KIND_IDS)
def test_a_status_slot_whose_text_the_prose_repeats_renders_once(name, facts):
    key, fact = _only(facts)
    text = fact["display"]
    template = f"{text} {{{key}}}"
    assert fact_template.interpolate_template(template, facts) == text
    # Whatever reads the scan still sees the status as present.
    scan = fact_template.scan_template(template, facts)
    assert scan["ok"] is True
    assert scan["placeholders"] == [key]


@pytest.mark.parametrize("name,facts", STATUS_KINDS, ids=KIND_IDS)
def test_a_status_slot_with_the_text_before_and_words_after_renders_once(
        name, facts):
    key, fact = _only(facts)
    text = fact["display"]
    template = f"{{{key}}} Then {text} Ask again later."
    assert fact_template.interpolate_template(template, facts) == (
        f"Then {text} Ask again later.")


@pytest.mark.parametrize("name,facts", STATUS_KINDS, ids=KIND_IDS)
def test_a_status_slot_alone_renders_once_unchanged(name, facts):
    key, fact = _only(facts)
    assert fact_template.interpolate_template(
        f"{{{key}}}", facts) == fact["display"]
    assert fact_template.interpolate_template(
        f"Note: {{{key}}}", facts) == f"Note: {fact['display']}"


@pytest.mark.parametrize("name,facts", STATUS_KINDS, ids=KIND_IDS)
def test_prose_that_only_partly_overlaps_the_status_text_keeps_the_slot(
        name, facts):
    key, fact = _only(facts)
    text = fact["display"]
    partial = text[:len(text) // 2].rstrip()
    template = f"{partial} {{{key}}}"
    assert fact_template.interpolate_template(template, facts) == (
        f"{partial} {text}")
    # Same words, one changed: no fuzzy matching.
    changed = text.lower() if text.lower() != text else text.upper()
    assert changed != text
    assert fact_template.interpolate_template(
        f"{changed} {{{key}}}", facts) == f"{changed} {text}"


def test_a_status_text_that_differs_only_in_whitespace_still_counts_as_verbatim():
    facts = fact_template.build_gathered_data_status_fact(
        ledger_has_data=True, figures_published=False)
    key, fact = _only(facts)
    rewrapped = fact["display"].replace(", but ", ",\n but  ", 1)
    assert fact_template.interpolate_template(
        f"{rewrapped} {{{key}}}", facts) == rewrapped


def test_only_a_status_sentence_is_dropped_never_another_text_fact():
    facts = {"pub:rest": {"key": "pub:rest", "value": "Rest day.",
                          "display": "Rest day.", "unit": None}}
    assert fact_template.interpolate_template(
        "Rest day. {pub:rest}", facts) == "Rest day. Rest day."
    # The cold-start token leaf is a word, not a sentence: never dropped.
    entries = [{"path": "$.result.readiness.cold_start.status",
                "value": "partial"}]
    facts = dict(fact_template._cold_start_entries(
        entries + [{"path": "$.result.readiness.cold_start.status_text",
                    "value": "Starts later."}], sequence=1))
    token_key = "$.result.readiness.cold_start.status"
    assert fact_template.interpolate_template(
        f"The status is partial: {{{token_key}}}", facts) == (
        "The status is partial: partial")
