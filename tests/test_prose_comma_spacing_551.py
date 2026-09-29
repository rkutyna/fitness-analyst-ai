"""health_advisor#551 -- a comma glued to the next word in model prose.

The model sometimes writes ``{fact|...},which``: the placeholder renders, the
comma follows it, and the next word is glued on ("Thursday,which"). The one
publish-time normaliser, ``numeric_tokens.normalise_model_prose``, gives a
comma that is immediately followed by a letter one space. It is applied to the
template BEFORE grounding, so no rendered offset or claim span can shift.

Numbers are untouched (a digit follows the comma in ``56,459``), and so is the
placeholder grammar: every fact key percent-escapes its components, comma
included, so a key never contains a comma-letter pair.
"""
from __future__ import annotations

from health_advisor import fact_template
from health_advisor.numeric_tokens import normalise_model_prose


def test_comma_after_a_placeholder_gets_a_space():
    assert (normalise_model_prose("on {fact|x},which")
            == "on {fact|x}, which")
    assert (normalise_model_prose("missed on Tuesday,with rest days")
            == "missed on Tuesday, with rest days")


def test_numbers_with_thousands_separators_are_unchanged():
    for text in ("56,459 steps", "1,234", "from 1,000,000 to 2,500.5",
                 "-13,900.25 kJ", "a 1,234km week"):
        assert normalise_model_prose(text) == text


def test_commas_with_a_space_or_at_the_ends_are_unchanged():
    for text in ("one, two", "one , two", "end,",
                 ",", "a,\nb", "a,—b", "a,(b)"):
        assert normalise_model_prose(text) == text


def test_the_rule_also_applies_when_other_normalisation_is_a_no_op():
    # No invisible character anywhere: the early-return path must still apply it.
    assert normalise_model_prose("a,b") == "a, b"
    # ...and alongside a removal site.
    assert normalise_model_prose("a,b  ̈3 c") == "a, b 3 c"


def test_it_is_idempotent_with_the_comma_rule():
    once = normalise_model_prose("on {fact|x},which,and 1,234,then")
    assert once == "on {fact|x}, which, and 1,234, then"
    assert normalise_model_prose(once) == once


def test_no_fact_key_contains_a_comma_letter_pair():
    """The placeholder grammar cannot be broken by the comma rule."""
    awkward = "a,b,c d,e"
    keys = [
        fact_template.fact_key(awkward, "2026-09-21", awkward),
        fact_template.fact_key("sleep_hours", {"start": "2026-09-21,x",
                                               "end": "2026-09-27"}, "mean"),
        fact_template.attachment_fact_key(awkward, awkward, awkward),
        fact_template.attachment_trend_key(awkward, awkward, "delta"),
        fact_template.workout_fact_key(awkward, awkward),
        fact_template.citation_fact_key(1, awkward, 2),
    ]
    for key in keys:
        assert "," not in key, key
        template = "on {" + key + "},which"
        assert normalise_model_prose(template) == "on {" + key + "}, which"
