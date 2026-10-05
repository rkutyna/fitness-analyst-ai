"""A duration reads "min", never a bare "m" (consumer #574, observation 3).

"Your longest continuous run was 35 m" reads as metres. The one duration
formatter, ``metrics.format_presentation``, now writes "35 min" and
"1 h 26 min"; the recognisers that decide whether a template already said the
unit accept the new spelling and keep accepting the old bare "m".

Every value here is synthetic.
"""
from __future__ import annotations

import re

import pytest

from health_advisor import chat, fact_template, metrics, normalize

WEEK = "2030-03-04:2030-03-10"
_BARE_M_TAIL = re.compile(r"(?:\d|\s)m\s*$")


# ---------------------------------------------------------------------------
# The formatter
# ---------------------------------------------------------------------------

# The existing rule, unchanged: whole hours drop the minutes ("1 h"); hours
# with minutes zero-pad the minutes ("1 h 06 min"); under an hour is "N min".
@pytest.mark.parametrize("minutes,expected", [
    (0, "0 min"),
    (35, "35 min"),
    (59, "59 min"),
    (60, "1 h"),
    (61, "1 h 01 min"),
    (86, "1 h 26 min"),
    (477, "7 h 57 min"),
])
@pytest.mark.parametrize("metric", [
    "jog_minutes", "sleep_asleep", "longest_block_min", "duration_min",
])
def test_the_duration_formatter_writes_min(metric, minutes, expected):
    assert metrics.format_presentation(metric, minutes) == expected


@pytest.mark.parametrize("hours,expected", [
    (0.0, "0 min"), (0.5, "30 min"), (7.95, "7 h 57 min"), (8.0, "8 h"),
])
def test_an_hour_valued_duration_writes_min(hours, expected):
    assert metrics.format_presentation("wear_hours", hours) == expected


def test_signed_and_dispersion_durations_write_min():
    assert metrics.format_presentation(
        "sleep_asleep", -45.0, field="delta_vs_baseline") == "-45 min"
    assert metrics.format_presentation(
        "sleep_asleep", 86.0, field="delta_vs_baseline") == "+1 h 26 min"
    assert metrics.format_presentation(
        "sleep_midpoint_sd_28d", 1.019) == "± 1 h 01 min"


@pytest.mark.parametrize("minutes,expected", [
    (35, "35 min"), (60, "1h"), (61, "1h 1min"), (555.68, "9h 16min"),
])
def test_the_compact_form_writes_min_too(minutes, expected):
    assert metrics.format_presentation(
        "duration_min", minutes, compact=True) == expected


def test_no_minute_unit_display_ends_in_a_bare_m():
    """Sweep: every metric and field the formatter renders, over values that
    land in every branch (zero, under an hour, whole hours, hours + minutes).
    A length metric renders a bare number, so a trailing ``m`` here can only
    be a duration written the old way."""
    rendered_durations = 0
    for metric in normalize.known_metrics():
        for field in sorted(metrics._PRESENTATION_FIELDS):
            for value in (0.0, 35.0, 59.4, 60.0, 86.0, 477.0, 12345.6789):
                for compact in (False, True):
                    text = metrics.format_presentation(
                        metric, value, field=field, compact=compact)
                    if text is None:
                        continue
                    assert not _BARE_M_TAIL.search(text), (
                        metric, field, value, compact, text)
                    if text.endswith("min"):
                        rendered_durations += 1
    # The sweep must have reached the duration renderer, not passed vacuously.
    assert rendered_durations > 100


def test_no_minute_unit_fact_display_ends_in_a_bare_m():
    """The same sweep through the fact set, where a display with no published
    leaf is re-rendered by ``fact_template._display_value``."""
    checked = 0
    for metric in sorted(metrics._DURATION_MINUTE_METRICS):
        for value in (0.0, 35.0, 86.0, 477.0):
            ledger = [{"sequence": 1, "tool_name": "synthetic_summary",
                       "arguments": {},
                       "result": {"metric": metric, "period": WEEK,
                                  "mean": value}}]
            facts = [fact for fact in
                     fact_template.build_fact_set(ledger).values()
                     if fact["field"] == "mean"]
            assert len(facts) == 1
            checked += 1
            assert facts[0]["display"].endswith(("min", "h")), facts[0]
            assert not _BARE_M_TAIL.search(facts[0]["display"]), facts[0]
    assert checked == 4 * len(metrics._DURATION_MINUTE_METRICS)


# ---------------------------------------------------------------------------
# The recognisers: a unit word after a duration slot is still an echo
# ---------------------------------------------------------------------------

def _restated(display, following):
    key = "fact|metric=fixture|period=s:period|field=mean"
    verification = {"ok": True, "grounded": True, "reason": ""}
    refused = chat._mark_restated_rendered_unit(
        verification, text=f"Value {display} {following}.",
        template=f"Value {{{key}}} {following}.",
        facts={key: {"display": display}})
    return refused, verification


@pytest.mark.parametrize("display", ["35 min", "1 h 26 min", "0 min"])
@pytest.mark.parametrize("word", ["min", "mins", "minute", "minutes", "m"])
def test_a_minute_word_after_a_min_duration_is_a_restated_unit(display, word):
    refused, verification = _restated(display, word)
    assert refused is True
    assert verification["ok"] is False
    assert verification["reason"].startswith(
        "narration restates a rendered unit")


@pytest.mark.parametrize("display,word", [
    ("2 h", "h"), ("2 h", "hours"), ("2 h", "hrs"),
    # A display stored before the wording changed is still recognised.
    ("1 h 26 m", "minutes"), ("35 m", "m"), ("35 m", "min"),
])
def test_other_duration_tails_still_refuse_their_own_words(display, word):
    assert _restated(display, word)[0] is True


@pytest.mark.parametrize("display,following", [
    ("35 min", "of running"),
    ("35 min", "more than last week"),
    ("1 h 26 min", "on average"),
    ("14 min/mi", "minutes per mile"),
    ("2 h", "minutes"),
])
def test_an_unrelated_word_after_a_duration_slot_is_not_an_echo(
        display, following):
    refused, verification = _restated(display, following)
    assert refused is False
    assert verification == {"ok": True, "grounded": True, "reason": ""}


@pytest.mark.parametrize("tail,rendered", [
    # The slot's own bare figure: the template already says the unit, in
    # whichever spelling, so the renderer adds nothing.
    (" min overall.", "35 min overall."),
    (" mins overall.", "35 mins overall."),
    (" minutes overall.", "35 minutes overall."),
    (" m overall.", "35 m overall."),
    # No unit written: the renderer appends the unit's spelling, "min".
    (" overall.", "35 min overall."),
    (".", "35 min."),
])
def test_a_bare_minute_figure_is_followed_by_one_unit(tail, rendered):
    fact = {"key": "k", "metric": "workout", "field": "duration_min",
            "unit": "min", "display": "35"}
    assert fact_template.interpolate_template(
        "{k}" + tail, {"k": fact}) == rendered


# ---------------------------------------------------------------------------
# A distance is not a duration
# ---------------------------------------------------------------------------

def test_a_length_in_metres_is_untouched():
    assert metrics.format_presentation("running_stride_length", 1.07) == "1.07"
    suffix, words = metrics.unit_suffix("m", metric="running_stride_length",
                                        field="mean")
    assert suffix == "m"
    assert "metres" in words and "minutes" not in words
    assert "m" in metrics.unit_echo_forms("m")
    assert "min" not in metrics.unit_echo_forms("m")


def test_a_distance_in_miles_is_untouched():
    assert metrics.format_presentation(
        "distance_walking_running", 5.24) == "5.24"
    assert metrics.unit_suffix("mi")[0] == "mi"
    fact = {"key": "k", "metric": "distance_walking_running",
            "field": "mean", "unit": "mi", "display": "5.24"}
    assert fact_template.interpolate_template(
        "You covered {k} today.", {"k": fact}) == (
            "You covered 5.24 mi today.")


def test_a_metres_figure_still_gets_metres_not_min():
    fact = {"key": "k", "metric": "running_stride_length", "field": "mean",
            "unit": "m", "display": "1.07"}
    assert fact_template.interpolate_template(
        "Your stride was {k} long.", {"k": fact}) == (
            "Your stride was 1.07 m long.")
