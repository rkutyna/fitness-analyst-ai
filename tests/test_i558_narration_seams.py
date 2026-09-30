"""The interpolation seam and the number gate for narration (#558).

Three defect classes read off 56 live answers that all passed verification:

C  Date-slot abuse: "from from Tue Jun 9", "In from Mon Jun 1 to Tue Jun 30",
   "on from Mon Aug 31 to Mon Aug 31", a four-week window called "the week",
   and (plan days) "pace.." / a title running into the next sentence.
D  Units dropped from interpolated figures: "your average weight was 196.1".
G  Model-authored numbers that no tool published, in advice and in prose.

Every template here is digit-free prose around real placeholders, and every
fact is built by ``build_fact_set`` / ``build_workout_facts`` from a
ledger-shaped record, so a change to the publisher shows up here too.
"""
from __future__ import annotations

import pytest

from health_advisor import chat, fact_template, llm, metrics, normalize
from tests.conftest import seed_metric

WEEK = "2026-08-24:2026-08-30"


def _ledger(metric="body_mass", period=WEEK, **fields):
    fields = fields or {"mean": 196.1}
    return [{
        "sequence": 1, "tool_name": "synthetic_summary", "arguments": {},
        "result": {"metric": metric, "period": period, **fields},
    }]


def _facts(metric="body_mass", period=WEEK, **fields):
    return fact_template.build_fact_set(_ledger(metric, period, **fields))


def _slot(metric, period, field):
    return "{" + fact_template.fact_key(metric, period, field) + "}"


def _render(template_with_L, period, *, metric="body_mass"):
    """Interpolate a template whose ``{L}`` is the period_label of ``period``."""
    facts = _facts(metric, period, mean=1.0)
    template = template_with_L.replace(
        "{L}", _slot(metric, period, "period_label"))
    return fact_template.interpolate_template(template, facts)


# ---------------------------------------------------------------------------
# Class C: the date slot
# ---------------------------------------------------------------------------

def test_a_single_day_range_labels_as_that_day():
    assert fact_template._period_label("2026-08-31:2026-08-31") == "Mon Aug 31"
    assert fact_template._period_label(
        {"start": "2026-08-31", "end": "2026-08-31"}) == "Mon Aug 31"


# (census text that was published, template the model wrote, period, expected)
CENSUS_DATE_SLOTS = [
    pytest.param(
        "Over the last three months, from from Tue Jun 9 to Mon Aug 31",
        "Over the last three months, from {L}, your heart rate was steady.",
        "2026-06-09:2026-08-31",
        "Over the last three months, from Tue Jun 9 to Mon Aug 31, "
        "your heart rate was steady.",
        id="from-from"),
    pytest.param(
        "In from Mon Jun 1 to Tue Jun 30",
        "In {L}, your weight held steady.",
        "2026-06-01:2026-06-30",
        "From Mon Jun 1 to Tue Jun 30, your weight held steady.",
        id="in-from-at-sentence-start"),
    pytest.param(
        "a strength training session on from Mon Aug 31 to Mon Aug 31",
        "You had a strength training session on {L}.",
        "2026-08-31:2026-08-31",
        "You had a strength training session on Mon Aug 31.",
        id="on-from-single-day"),
    pytest.param(
        "jogging for 0 m on from Mon Aug 31 to Mon Aug 31",
        "You went jogging for a while on {L}.",
        "2026-08-31:2026-08-31",
        "You went jogging for a while on Mon Aug 31.",
        id="on-from-single-day-jogging"),
    pytest.param(
        "As of from Mon Aug 31 to Mon Aug 31",
        "As of {L}, you have logged no running.",
        "2026-08-31:2026-08-31",
        "As of Mon Aug 31, you have logged no running.",
        id="as-of-single-day"),
    pytest.param(
        "the week from from Mon Jul 6 to Sun Aug 2",
        "Your load was steady in the week from {L}.",
        "2026-07-06:2026-08-02",
        "Your load was steady in the period from Mon Jul 6 to Sun Aug 2.",
        id="four-weeks-called-the-week"),
    pytest.param(
        "for from Mon Jun 1 to Mon Jun 29",
        "Your sleep was consistent for {L}.",
        "2026-06-01:2026-06-29",
        "Your sleep was consistent from Mon Jun 1 to Mon Jun 29.",
        id="for-from"),
    pytest.param(
        "the week the week of August 24",
        "During the week {L} you slept well.",
        "2026-08-24:2026-08-30",
        "During the week of August 24 you slept well.",
        id="doubled-the-week-of"),
    pytest.param(
        "a block called the week",
        "The week from {L} was steady.",
        "2026-07-06:2026-08-02",
        "The period from Mon Jul 6 to Sun Aug 2 was steady.",
        id="sentence-initial-the-week"),
]


@pytest.mark.parametrize("published,template,period,expected",
                         CENSUS_DATE_SLOTS)
def test_census_date_slots_render_grammatically(published, template, period,
                                                expected):
    rendered = _render(template, period)
    assert rendered == expected
    for doubled in ("from from", "in from", "on from", "for from",
                    "In from", "the week from", "the the"):
        assert doubled not in rendered
    assert published not in rendered


def test_a_multi_week_window_is_never_called_the_week():
    block = {"start": "2026-07-27", "end": "2026-08-23",
             "period_starts": ["2026-07-27", "2026-08-03", "2026-08-10",
                               "2026-08-17"]}
    rendered = _render("In the week {L} you were steady.", block)
    assert rendered == ("In the 4 weeks from Mon Jul 27 to Sun Aug 23 "
                        "you were steady.")


def test_a_correct_lead_in_is_left_exactly_as_written():
    for template, expected in [
        ("Over the period {L}, steady.",
         "Over the period from Tue Jun 9 to Tue Aug 4, steady."),
        ("Steady from {L}.", "Steady from Tue Jun 9 to Tue Aug 4."),
    ]:
        assert _render(template, "2026-06-09:2026-08-04") == expected
    assert _render("Steady on {L}.", "2026-08-31") == "Steady on Mon Aug 31."


# Plan days (#557): titles are sentences, and the model guesses the edges.
def _plan_facts():
    return {
        "pub:rest": {"key": "pub:rest", "value": "Rest day.",
                     "display": "Rest day.", "unit": None},
        "pub:easy": {"key": "pub:easy", "unit": None,
                     "value": "Easy jog at a comfortable, conversational pace.",
                     "display": "Easy jog at a comfortable, "
                                "conversational pace."},
        "pub:strength": {"key": "pub:strength", "unit": None,
                         "value": "Full-body strength (strength)",
                         "display": "Full-body strength (strength)"},
    }


@pytest.mark.parametrize("template,expected", [
    ("Saturday is {pub:rest}.", "Saturday is Rest day."),
    ("Friday is {pub:easy}.",
     "Friday is Easy jog at a comfortable, conversational pace."),
    ("Saturday is {pub:rest}. Sunday is not.",
     "Saturday is Rest day. Sunday is not."),
    ("Saturday is {pub:rest}!", "Saturday is Rest day."),
    ("You have {pub:strength} On Sunday you rest.",
     "You have Full-body strength (strength). On Sunday you rest."),
    ("You have {pub:strength}, then rest.",
     "You have Full-body strength (strength), then rest."),
    ("You have {pub:strength}.", "You have Full-body strength (strength)."),
], ids=["period-doubled", "long-title-period-doubled", "period-then-sentence",
        "exclaim-after-period", "title-runs-into-sentence",
        "comma-untouched", "title-without-period-keeps-template-period"])
def test_a_text_display_and_the_template_share_one_sentence_edge(
        template, expected):
    rendered = fact_template.interpolate_template(template, _plan_facts())
    assert rendered == expected
    assert ".." not in rendered


# ---------------------------------------------------------------------------
# Class D: units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric,field,value,template,expected", [
    # The census examples, by metric and field.
    ("body_mass", "mean", 196.1,
     "Your average weight was {K}.", "Your average weight was 196.1 lb."),
    ("body_mass", "delta_pct", 0.54,
     "The overall change was {K}.", "The overall change was 0.54%."),
    ("distance_walking_running", "mean", 5.24,
     "You covered about {K} of walking and running.",
     "You covered about 5.24 mi of walking and running."),
    ("vo2_max", "mean", 41.02,
     "Your oxygen fitness is {K}.", "Your oxygen fitness is 41.02 mL/min/kg."),
    ("active_energy", "mean", 94,
     "Your active energy averaged {K}.", "Your active energy averaged 94 kcal."),
    ("resting_heart_rate", "mean", 50,
     "Your resting heart rate averaged {K}.",
     "Your resting heart rate averaged 50 bpm."),
    ("step_count", "mean", 14074,
     "You averaged {K} on a typical day.",
     "You averaged 14,074 steps on a typical day."),
    ("longest_block_min", "delta_pct", -4.04,
     "Your longest block decreased by {K}.",
     "Your longest block decreased by -4.04%."),
    ("jog_minutes", "delta_pct", 21.13,
     "Jogging volume rose by {K} against baseline.",
     "Jogging volume rose by 21.13% against baseline."),
    # A percent-change field is a percentage whatever the series' unit is.
    ("body_mass", "total_delta_pct", 1.2,
     "Weight moved {K}.", "Weight moved 1.2%."),
])
def test_a_bare_figure_interpolates_with_its_unit(
        metric, field, value, template, expected):
    facts = _facts(metric, **{field: value})
    key = _slot(metric, WEEK, field)
    assert fact_template.interpolate_template(
        template.replace("{K}", key), facts) == expected


@pytest.mark.parametrize("template,expected", [
    ("Your weight was {K} pounds.", "Your weight was 196.1 pounds."),
    ("Your weight was {K} lb.", "Your weight was 196.1 lb."),
    ("Your weight was {K}lbs today.", "Your weight was 196.1 lbs today."),
    # A different unit word is the model's claim; a second unit is not added.
    ("Your weight was {K} kilograms.", "Your weight was 196.1 kilograms."),
    # An ordinal suffix and a compound are not a unit slot.
    ("The {K}th reading.", "The 196.1th reading."),
    ("A {K}-pound reading.", "A 196.1-pound reading."),
], ids=["pounds", "lb", "glued-lbs", "other-unit", "ordinal", "compound"])
def test_a_unit_the_template_already_wrote_is_not_repeated(template, expected):
    facts = _facts("body_mass", mean=196.1)
    key = _slot("body_mass", WEEK, "mean")
    assert fact_template.interpolate_template(
        template.replace("{K}", key), facts) == expected


def test_a_percent_word_after_a_percent_change_is_not_repeated():
    facts = _facts("body_mass", delta_pct=0.54)
    key = _slot("body_mass", WEEK, "delta_pct")
    for follow, expected in [(" percent", "0.54 percent"), ("%", "0.54%"),
                             (" of it", "0.54% of it")]:
        assert fact_template.interpolate_template(
            "Change: " + key + follow + ".", facts) == "Change: " + expected + "."


def test_a_percent_change_fact_carries_the_percent_unit():
    fact = _facts("body_mass", delta_pct=0.54)[
        fact_template.fact_key("body_mass", WEEK, "delta_pct")]
    assert fact["unit"] == "%"


def test_a_workout_figure_interpolates_with_its_unit():
    record = {
        "sequence": 1, "tool_name": "list_workouts", "result_elided": False,
        "result": {"start": "2026-08-31", "end": "2026-08-31", "count": 1,
                   "workouts": [{
                       "date": "2026-08-31", "type": "strength",
                       "workout_key": "k1", "duration_min": 50.3,
                       "distance_mi": 0.0, "avg_heart_rate": 99.0,
                       "max_heart_rate": 125.0, "start_time_local": "08:00"}]},
    }
    facts = fact_template.build_workout_facts([record])
    duration = next(k for k in facts if k.endswith("field=duration_min"))
    heart = next(k for k in facts if k.endswith("field=max_heart_rate"))
    assert fact_template.interpolate_template(
        "A session of {" + duration + "} reached {" + heart + "} at most.",
        facts) == "A session of 50.3 min reached 125 bpm at most."


# Metrics whose bare figure is deliberately unitless: Python has no word for
# them. Every other catalogued metric must render with a unit, so adding a
# unit to the catalog without a suffix goes red here, not in a live answer.
_UNITLESS_BY_DESIGN = {
    "body_mass_index", "stand_hour", "breathing_disturbances",
    "apple_sleeping_breathing_disturbances", "hr_load_proxy",
    "physical_effort", "subjective_stress", "subjective_soreness",
    "subjective_energy", "subjective_sleep_quality", "workout_effort",
}


def test_every_unit_bearing_catalog_metric_interpolates_with_its_unit():
    """Sweep the catalog: bare display + unit-bearing metric => a unit appears."""
    bare = []
    for metric, spec in normalize.CATALOG.items():
        unit = spec["unit"]
        display = metrics.format_presentation(metric, 12.5, field="mean")
        if display is None:
            display = metrics.format_numeric(12.5)
        fact = {"key": "k", "metric": metric, "field": "mean", "unit": unit,
                "display": display}
        rendered = fact_template.interpolate_template(
            "Value {k} here.", {"k": fact})
        head = rendered[len("Value "):-len(" here.")]
        if head == display and fact_template._BARE_DISPLAY_RE.match(display):
            bare.append(metric)
    assert set(bare) <= _UNITLESS_BY_DESIGN, sorted(
        set(bare) - _UNITLESS_BY_DESIGN)
    # And the allowlist is not stale: each name here really renders bare.
    assert _UNITLESS_BY_DESIGN <= set(bare), sorted(
        _UNITLESS_BY_DESIGN - set(bare))


def test_a_display_that_already_carries_its_unit_is_untouched():
    facts = {"k": {"key": "k", "metric": "sleep_asleep", "field": "mean",
                   "unit": "min", "display": "7 h 53 m"}}
    assert fact_template.interpolate_template("Slept {k} a night.", facts) == (
        "Slept 7 h 53 m a night.")


# ---------------------------------------------------------------------------
# Class G: numbers the model wrote
# ---------------------------------------------------------------------------

CENSUS_ADVICE = [
    pytest.param("Start with {advice:2-3 short runs per week, increasing each "
                 "run by 5-10 minutes weekly}.", id="runs-per-week"),
    pytest.param("Aim for {advice:a bedtime window of about 30 minutes} and a "
                 "wake window to match.", id="bedtime-window"),
]


def _scan(template, facts=None, question="How consistent has my sleep been?"):
    return fact_template.scan_template(
        template, facts or {}, question=question, unbacked_numbers=True)


@pytest.mark.parametrize("template", CENSUS_ADVICE)
def test_the_census_advice_quantities_fail_the_scan(template):
    # Before #558 both passed: the advice slot admitted any digits.
    assert fact_template.scan_template(template, {})["ok"] is True
    scan = _scan(template)
    assert scan["ok"] is False
    assert scan["reason"] == fact_template.ADVICE_QUANTITY_REASON
    assert scan["unbacked_numbers"]


STRENGTH_QUESTION = "What strength work should I do this week?"
WEIGHT_QUESTION = "How has my weight changed since I stopped running?"


@pytest.mark.parametrize("content", [
    "3 sets of 8 to 12 reps", "3 sets of 10 reps", "3 rounds",
    "60 seconds", "60-90 seconds", "20–30 seconds", "3x10", "3 x 10",
    "two sets of eight reps", "8 reps per leg", "2-3 rounds",
    "3 sets of 8 per leg", "3 repetitions",
])
def test_a_strength_prescription_is_the_one_advice_quantity_admitted(content):
    # ... in a strength or plan context (#558, scoped 2026-09-30).
    assert _scan("Try {advice:" + content + "}.",
                 question=STRENGTH_QUESTION)["ok"] is True


@pytest.mark.parametrize("content", [
    "2-3 short runs per week", "5-10 minutes weekly",
    "about 30 minutes", "8 hours each night", "10 to 20 pounds",
    "5 miles", "twice a week", "three sessions", "a 15-minute walk",
    "20 to 30 minutes of warm-up", "3 sets in 20 minutes",
])
def test_every_other_advice_quantity_is_refused(content):
    scan = _scan("Try {advice:" + content + "}.", question=STRENGTH_QUESTION)
    assert scan["ok"] is False, content
    assert scan["reason"] == fact_template.ADVICE_QUANTITY_REASON


# --- the prescription exemption is scoped to strength or the plan (#558) -----

C2_ANSWER = ("Your weight trend is on file. Coaching guidance: "
             "{advice:3 sets of 8-12 reps for each movement, on separate days}")
SPELLED_PRESCRIPTION = ("Do three sets of eight to twelve reps for each "
                        "movement, with sixty seconds of rest.")


def _plan_day_facts():
    key = fact_template.declared_fact_key("plan/2026-09-30/title")
    return {key: {"key": key, "label": "The plan day's session title",
                  "value": "Full-body strength", "unit": None,
                  "display": "Full-body strength",
                  "source": {"sequence": 1, "path": "$"}}}


def _strength_workout_facts(workout_type="traditional_strength_training"):
    workout = "2026-09-28|" + workout_type
    key = fact_template.workout_fact_key(workout, "duration")
    return {key: {"key": key, "workout": workout, "field": "duration",
                  "value": 40, "unit": "min", "display": "40 minutes",
                  "source": {"sequence": 1, "path": "$"}}}


def test_a_prescription_on_a_weight_question_is_refused():
    # Census 2, persona C, question C2: the answer ended in a strength
    # prescription the question never asked for.
    scan = _scan(C2_ANSWER, question=WEIGHT_QUESTION)
    assert scan["ok"] is False
    assert scan["reason"] == fact_template.ADVICE_QUANTITY_REASON
    assert "8" in scan["unbacked_numbers"]


def test_the_same_prescription_on_a_strength_question_passes():
    scan = _scan(C2_ANSWER, question=STRENGTH_QUESTION)
    assert scan["ok"] is True
    assert scan["unbacked_numbers"] == []


def test_a_spelled_prescription_follows_the_same_scope():
    refused = _scan(SPELLED_PRESCRIPTION, question=WEIGHT_QUESTION)
    assert refused["ok"] is False
    assert refused["reason"] == fact_template.NUMBER_WORD_REASON
    assert _scan(SPELLED_PRESCRIPTION, question=STRENGTH_QUESTION)["ok"] is True


def test_plan_day_facts_make_the_context_whatever_the_question_says():
    facts = _plan_day_facts()
    assert _scan(C2_ANSWER, facts, question="What is next?")["ok"] is True
    assert _scan(SPELLED_PRESCRIPTION, facts, question="Hi")["ok"] is True


def test_a_strength_session_fact_makes_the_context():
    facts = _strength_workout_facts()
    assert _scan(C2_ANSWER, facts, question=WEIGHT_QUESTION)["ok"] is True
    # A run is not a strength session.
    running = _strength_workout_facts("running")
    assert _scan(C2_ANSWER, running, question=WEIGHT_QUESTION)["ok"] is False


@pytest.mark.parametrize("question", [
    "What strength work should I do this week?",
    "Can you write this up into a circuit?",
    "How many reps should I do?",
    "What does my plan say for Thursday?",
    "What should I add to my routine?",
    "Is lifting weights worth it?",
    "Should I go to the gym on rest days?",
])
def test_strength_and_plan_vocabulary_is_the_context(question):
    assert fact_template.strength_or_plan_context(question, {}) is True


@pytest.mark.parametrize("question", [
    "How has my weight changed since I stopped running?",
    "How consistent has my sleep been?",
    "How much did I run over the last two weeks?",
    "What was my resting heart rate this week?",
    "Did I train enough last month?",
    "", None,
])
def test_other_questions_are_not(question):
    assert fact_template.strength_or_plan_context(question, {}) is False


def test_unknown_context_grants_no_exemption():
    # A caller that passes neither a question nor facts gets no exemption.
    scan = fact_template.scan_template(
        "Try {advice:3 sets of 10 reps}.", {}, unbacked_numbers=True)
    assert scan["ok"] is False


def test_a_spelled_number_in_prose_is_refused_like_a_digit():
    scan = _scan("That is a total of eight sessions.")
    assert scan["ok"] is False
    assert scan["reason"] == fact_template.NUMBER_WORD_REASON
    assert scan["unbacked_numbers"] == ["eight"]
    # A digit-free advice slot is prose, so the same rule reaches inside it.
    assert _scan("Try {advice:five easy runs}.")["ok"] is False


def test_incidental_numbers_have_a_stated_rule():
    ok = [
        # An ordinal numbers a list position; it states no quantity.
        "Your third and fourth weeks were steadier.",
        # The user's own window, named back.
        "Over the last two weeks you stayed steady.",
        # 'one' alone is a pronoun, and first/second are never counted.
        "The one you asked about came second.",
        # A metric name is a name, not a number.
        "Your six minute walk test distance is on file.",
    ]
    for text in ok:
        scan = _scan(text, question="How much did I run over the last two weeks?")
        assert scan["ok"] is True, text
    refused = [
        "Compared to the previous two nights you slept longer.",
        "All other days had zero jogging minutes.",
        "You rated your sleep four out of five.",
    ]
    for text in refused:
        assert _scan(text, question="How did I sleep last night?")["ok"] is False


def test_a_digit_echoed_from_the_question_is_still_refused_in_prose():
    # The user's "4 weeks" licenses the SPELLED "four", never a typed digit.
    q = "How much did I run over the last 4 weeks?"
    assert _scan("Over the last four weeks you ran steadily.", question=q)["ok"]
    assert _scan("Over the last 4 weeks you ran steadily.", question=q)[
        "ok"] is False


def test_the_default_scan_keeps_the_old_rule_for_other_callers():
    template = "Start with {advice:2-3 short runs per week}. Five is plenty."
    assert fact_template.scan_template(template, {})["ok"] is True


# The ask path, end to end: both census answers end in fallback, and the
# repair turn is told which span to remove.
@pytest.fixture(autouse=True)
def _seed_model_path_vault(conn):
    seed_metric(conn, "fixture_metric", "2026-01-01", [1])


@pytest.mark.parametrize("draft,span", [
    ("Start with {advice:2-3 short runs per week, increasing each run by "
     "5-10 minutes weekly}.", "2-3"),
    ("Aim for {advice:a bedtime window of about 30 minutes}.", "30"),
])
def test_the_ask_path_refuses_the_census_advice_and_names_the_span(
        monkeypatch, vault, draft, span):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    prompts = []

    def loop(*args, **kwargs):
        prompts.append(args[0])
        return draft

    monkeypatch.setattr(llm, "tool_loop", loop)

    result = chat.answer_question(vault, "What should I change about my sleep?")

    assert result["mode"] == "fallback"
    assert result["verification"]["reason"] == (
        fact_template.ADVICE_QUANTITY_REASON)
    assert "advice_quantities" in result["verification"]
    assert any(span in prompt and "EXACT GATE REFUSAL" in prompt
               for prompt in prompts[1:])
    assert "2-3" not in result["text"] and "30 minutes" not in result["text"]


def test_a_strength_circuit_is_still_answered(monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda *a, **k: "Circuit: {advice:3 rounds} of squats and rows, rest "
                        "{advice:60 seconds} between exercises.")

    result = chat.answer_question(vault, "What should I add to my routine?")

    assert result["mode"] == "narration"
    assert result["verification"]["advice_quantities"] == [
        "3 rounds", "60 seconds"]


def test_the_ask_path_refuses_a_prescription_on_a_weight_question(
        monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *a, **k: C2_ANSWER)

    result = chat.answer_question(vault, WEIGHT_QUESTION)

    assert result["mode"] == "fallback"
    assert result["verification"]["reason"] == (
        fact_template.ADVICE_QUANTITY_REASON)
    assert "8-12" not in result["text"]
