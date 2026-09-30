"""Census classes E and F, and the rating scale (health_advisor#558, part 2).

E. The aggregate a question asks for was never published, so the model refused
   ("not directly available") or dropped it ("You completed strength training
   sessions", no count). Each test here runs the census question through the
   real ask path with a stubbed ``tool_loop``: the gather turn calls the real
   tools, in the real ledger wrapper, with the arguments the census model chose;
   the narration turn sees the closed fact set and returns a template. The
   expected value in every assertion is computed by SQL on the test vault, not
   by the builder under test.

F. "The last month" asked on 2026-08-31 was resolved to the previous CALENDAR
   month, so the tools were told July and every gate agreed, because the
   window they measured against was the one Python had resolved wrongly.

Rating scale. A self-rating's scale is a Python fact; "out of five" is no
longer a number the model authored.

All data here is synthetic; the engine repository is public.
"""
from __future__ import annotations

import json
import re

import pytest

from health_advisor import chat, fact_template, llm, mcp_server, subjective
from health_advisor.calendar_window import resolve_window
from datetime import date

from tests.conftest import seed_metric, seed_workout

AS_OF = "2026-08-31"
FACT_SET_HEAD = "CLOSED FACT SET (Python ledger facts for this answer only):\n"
STEPS_LAST_WEEK = [5588.0, 4360.0, 3970.0, 4674.0, 4535.0, 1299.0, 2088.0]

# (type, date, duration_min, distance_mi). Aug 1 is one day outside "the last
# month" (30 days ending Aug 31 start on Aug 2), and July is outside it too.
WORKOUTS = [
    ("running", "2026-07-15", 80.0, 9.5),
    ("running", "2026-08-01", 70.0, 7.5),
    ("running", "2026-08-05", 48.0, 5.12),
    ("running", "2026-08-12", 62.3, 7.44),
    ("running", "2026-08-25", 55.0, 6.08),
    ("running", "2026-08-27", 60.3, 6.46),
    ("running", "2026-08-30", 73.0, 7.67),
    ("walking", "2026-08-20", 30.0, 1.5),
    ("traditional_strength_training", "2026-07-20", 50.0, None),
    ("traditional_strength_training", "2026-08-01", 51.0, None),
    ("traditional_strength_training", "2026-08-04", 44.7, None),
    ("traditional_strength_training", "2026-08-11", 48.0, None),
    ("traditional_strength_training", "2026-08-18", 50.3, None),
    ("traditional_strength_training", "2026-08-31", 46.0, None),
]


@pytest.fixture
def seeded(conn):
    seed_metric(conn, "step_count", "2026-07-01", [8000.0] * 54)      # to Aug 23
    seed_metric(conn, "step_count", "2026-08-24", STEPS_LAST_WEEK)     # to Aug 30
    seed_metric(conn, "step_count", "2026-08-31", [6000.0])
    for row in WORKOUTS:
        seed_workout(conn, *row)
    return conn


def _sql(conn, query, *args):
    return conn.execute(query, args).fetchall()


def _ask(monkeypatch, vault, question, calls, template):
    """Run the ask path; return (result, facts the narration turn saw, ledger)."""
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    seen = {}

    def fake_tool_loop(prompt, *, ledger_path=None, tool_names=None, **kwargs):
        if tool_names:
            reg = llm._ledgered(
                llm._registry(vault, include=llm.COACH_TOOLS), ledger_path)
            for name, args in calls:
                reg[name][0](**args)
            return "gathered"
        seen["prompt"] = prompt
        return template(seen["facts"]) if callable(template) else template

    real_render = fact_template.render_fact_set

    def spy_render(facts):
        seen["facts"] = facts
        return real_render(facts)

    monkeypatch.setattr(llm, "tool_loop", fake_tool_loop)
    monkeypatch.setattr(fact_template, "render_fact_set", spy_render)
    capture: list = []
    result = chat.answer_question(vault, question, as_of=AS_OF,
                                  capture=capture)
    return result, seen.get("facts", {}), capture[0]["ledger"] if capture else []


def _declared(facts, suffix):
    hits = [fact for key, fact in facts.items()
            if key.startswith("pub:workouts/") and key.endswith(suffix)]
    assert len(hits) == 1, (suffix, sorted(facts))
    return hits[0]


def _key(facts, suffix):
    return next(key for key in facts
                if key.startswith("pub:workouts/") and key.endswith(suffix))


# --------------------------------------------------------------------------
# E1. "How many strength sessions did I do in the last month?"
# --------------------------------------------------------------------------

def test_strength_session_count_is_a_fact(monkeypatch, vault, seeded):
    expected = _sql(
        seeded, "SELECT COUNT(*) FROM workouts WHERE workout_type = "
        "'traditional_strength_training' AND local_date BETWEEN "
        "'2026-08-02' AND '2026-08-31'")[0][0]
    assert expected == 4                       # Aug 4, 11, 18, 31

    def template(facts):
        key = _key(facts, "/traditional_strength_training/count")
        return "You completed {%s} strength sessions in the last month." % key

    result, facts, _ledger = _ask(
        monkeypatch, vault, "How many strength sessions did I do in the last "
        "month?", [("list_workouts", {"start": "2026-07-01",
                                      "end": "2026-07-31"})], template)
    fact = _declared(facts, "/traditional_strength_training/count")
    assert fact["value"] == expected
    assert result["mode"] == "narration", result
    assert result["text"] == ("You completed 4 strength sessions in the last "
                              "month.")


def test_count_covers_the_whole_range_not_the_truncated_rows(vault, seeded):
    tools = mcp_server.build_tools(vault)
    result = tools["list_workouts"]("2026-08-02", "2026-08-31", limit=2)
    assert result["truncated"] is True and result["count"] == 2
    facts = fact_template.build_declared_facts(
        [{"sequence": 1, "tool_name": "list_workouts", "result": result}])
    counts = {key: fact["value"] for key, fact in facts.items()
              if key.endswith("/count")}
    expected = dict(_sql(
        seeded, "SELECT workout_type, COUNT(*) FROM workouts WHERE "
        "local_date BETWEEN '2026-08-02' AND '2026-08-31' GROUP BY 1"))
    for kind, n in expected.items():
        assert counts[f"pub:workouts/2026-08-02:2026-08-31/{kind}/count"] == n
    assert counts["pub:workouts/2026-08-02:2026-08-31/all/count"] == sum(
        expected.values())


def test_the_last_month_is_the_thirty_days_ending_on_as_of(monkeypatch, vault,
                                                          seeded):
    _result, _facts, ledger = _ask(
        monkeypatch, vault, "How many strength sessions did I do in the last "
        "month?", [("list_workouts", {"start": "2026-07-01",
                                      "end": "2026-07-31"})], "Nothing.")
    args = ledger[0]["arguments"]
    assert (args["start"], args["end"]) == ("2026-08-02", "2026-08-31")


def test_two_strength_types_publish_one_family_count(vault, conn):
    seed_metric(conn, "step_count", "2026-08-01", [1000.0] * 31)
    for day, kind in (("2026-08-03", "traditional_strength_training"),
                      ("2026-08-10", "functional_strength_training"),
                      ("2026-08-17", "functional_strength_training"),
                      ("2026-08-24", "running")):
        seed_workout(conn, kind, day, 40.0, 3.0 if kind == "running" else None)
    result = mcp_server.build_tools(vault)["list_workouts"](
        "2026-08-02", "2026-08-31")
    facts = fact_template.build_declared_facts(
        [{"sequence": 1, "tool_name": "list_workouts", "result": result}])
    key = "pub:workouts/2026-08-02:2026-08-31/strength/count"
    expected = _sql(conn, "SELECT COUNT(*) FROM workouts WHERE workout_type "
                          "LIKE '%strength%' AND local_date BETWEEN "
                          "'2026-08-02' AND '2026-08-31'")[0][0]
    assert facts[key]["value"] == expected == 3


# --------------------------------------------------------------------------
# E2. "How many steps did I average last week?"
# --------------------------------------------------------------------------

def test_last_weeks_average_steps_is_a_fact(monkeypatch, vault, seeded):
    expected = _sql(
        seeded, "SELECT AVG(sum) FROM daily_metrics WHERE metric = "
        "'step_count' AND date BETWEEN '2026-08-24' AND '2026-08-30'"
    )[0][0]
    assert expected == pytest.approx(sum(STEPS_LAST_WEEK) / 7)

    def template(facts):
        key = fact_template.fact_key(
            "step_count", "2026-08-24:2026-08-30", "mean")
        assert key in facts
        return "You averaged {%s} last week." % key

    result, facts, _ = _ask(
        monkeypatch, vault, "How many steps did I average last week?",
        [("get_daily_series", {"metric": "step_count", "start": "2026-08-24",
                               "end": "2026-08-30"})], template)
    fact = facts[fact_template.fact_key(
        "step_count", "2026-08-24:2026-08-30", "mean")]
    assert fact["value"] == pytest.approx(expected, abs=0.01)
    assert result["mode"] == "narration", result
    assert result["text"] == "You averaged 3,788 steps last week."


def test_a_series_mean_agrees_with_summarize_metric(vault, seeded):
    tools = mcp_server.build_tools(vault)
    daily = tools["get_daily_series"]("step_count", "2026-08-24", "2026-08-30")
    summary = tools["summarize_metric"]("step_count", "2026-08-24:2026-08-30")
    ledger = [{"sequence": 1, "tool_name": "get_daily_series",
               "result": daily},
              {"sequence": 2, "tool_name": "summarize_metric",
               "result": summary}]
    key = fact_template.fact_key(
        "step_count", "2026-08-24:2026-08-30", "mean")
    # Both tools published the key with the same value, so it is not withheld.
    assert fact_template.build_fact_set(ledger)[key]["value"] == \
        summary["mean"] == daily["window_summary"]["mean"]


# --------------------------------------------------------------------------
# E3. "What was my longest run in the last two months?"
# --------------------------------------------------------------------------

def test_longest_run_and_its_date_are_facts(monkeypatch, vault, seeded):
    window = ("2026-07-01", "2026-08-31")
    by_distance = _sql(
        seeded, "SELECT local_date, distance_mi FROM workouts WHERE "
        "workout_type = 'running' AND local_date BETWEEN ? AND ? "
        "ORDER BY distance_mi DESC LIMIT 1", *window)[0]
    by_duration = _sql(
        seeded, "SELECT local_date, duration_min FROM workouts WHERE "
        "workout_type = 'running' AND local_date BETWEEN ? AND ? "
        "ORDER BY duration_min DESC LIMIT 1", *window)[0]
    assert tuple(by_distance) == ("2026-07-15", 9.5)
    assert tuple(by_duration) == ("2026-07-15", 80.0)

    def template(facts):
        return ("Your longest run was {%s} on {%s}, {%s} long."
                % (_key(facts, "/running/longest_distance"),
                   _key(facts, "/running/longest_distance_date"),
                   _key(facts, "/running/longest_duration")))

    result, facts, _ = _ask(
        monkeypatch, vault, "What was my longest run in the last two months?",
        [("list_workouts", {"start": window[0], "end": window[1]})], template)
    assert _declared(facts, "/running/longest_distance")["value"] == \
        by_distance[1]
    assert _declared(facts, "/running/longest_duration")["value"] == \
        by_duration[1]
    day = date.fromisoformat(by_distance[0])
    assert _declared(facts, "/running/longest_distance_date")["value"] == \
        f"{day.strftime('%a %b')} {day.day}"
    assert result["mode"] == "narration", result
    assert result["text"] == ("Your longest run was 9.5 mi on Wed Jul 15, "
                              "80 min long.")


def test_a_metric_vault_publishes_kilometres(monkeypatch, vault, seeded):
    from health_advisor import vault as vaultmod
    vaultmod.set_unit_system(seeded, "metric")
    seeded.commit()
    result = mcp_server.build_tools(vault)["list_workouts"](
        "2026-08-02", "2026-08-31")
    facts = fact_template.build_declared_facts(
        [{"sequence": 1, "tool_name": "list_workouts", "result": result}])
    longest = facts["pub:workouts/2026-08-02:2026-08-31/running/"
                    "longest_distance"]
    expected_mi = _sql(seeded, "SELECT MAX(distance_mi) FROM workouts WHERE "
                       "workout_type = 'running' AND local_date BETWEEN "
                       "'2026-08-02' AND '2026-08-31'")[0][0]
    assert longest["unit"] == "km"
    assert longest["value"] == pytest.approx(expected_mi * 1.609344, abs=0.01)
    paces = [fact for fact in fact_template.build_workout_facts(
        [{"sequence": 1, "tool_name": "list_workouts", "result": result}]
    ).values() if fact["field"] == "pace"]
    assert paces and all(fact["unit"] == "min/km" for fact in paces)


# --------------------------------------------------------------------------
# E4. "What pace have my recent runs been at?" -- pace, and the dates
# --------------------------------------------------------------------------

def test_each_runs_pace_and_date_are_facts(monkeypatch, vault, seeded):
    runs = _sql(seeded, "SELECT local_date, duration_min, distance_mi FROM "
                        "workouts WHERE workout_type = 'running' AND "
                        "local_date BETWEEN '2026-08-25' AND '2026-08-31' "
                        "ORDER BY local_date")
    assert len(runs) == 3

    def template(facts):
        parts = []
        for day, _duration, _distance in runs:
            base = f"fact|workout={day}%7Crunning|field="
            parts.append("On {%sdate} you ran {%sdistance_mi} in "
                         "{%sduration_min}, a pace of {%space}."
                         % (base, base, base, base))
        return " ".join(parts)

    result, facts, _ = _ask(
        monkeypatch, vault, "What pace have my recent runs been at?",
        [("list_workouts", {"start": "2026-08-25", "end": "2026-08-31"})],
        template)
    expected_lines = []
    for day, duration, distance in runs:
        pace = duration / distance
        fact = facts[f"fact|workout={day}%7Crunning|field=pace"]
        assert fact["value"] == pytest.approx(pace, abs=0.01)
        assert fact["unit"] == "min/mi"
        seconds = int(round(pace * 60))
        assert fact["display"] == f"{seconds // 60}:{seconds % 60:02d} min/mi"
        stamp = date.fromisoformat(day)
        expected_lines.append(
            f"On {stamp.strftime('%a %b')} {stamp.day} you ran "
            f"{distance:g} mi in {duration:g} min, a pace of "
            f"{fact['display']}.")
    assert result["mode"] == "narration", result
    assert result["text"] == " ".join(expected_lines)
    assert "On  " not in result["text"] and "On 6" not in result["text"]


def test_a_walk_and_a_zero_distance_row_publish_no_pace(vault, conn):
    seed_metric(conn, "step_count", "2026-08-01", [1000.0] * 31)
    seed_workout(conn, "running", "2026-08-10", 30.0, 0.0)
    seed_workout(conn, "cycling", "2026-08-11", 30.0, 10.0)
    seed_workout(conn, "walking", "2026-08-12", 20.0, 1.0)
    result = mcp_server.build_tools(vault)["list_workouts"](
        "2026-08-01", "2026-08-31")
    rows = {row["type"]: row for row in result["workouts"]}
    assert "pace_min_per_mi" not in rows["running"]      # no distance
    assert "pace_min_per_mi" not in rows["cycling"]      # not a pace sport
    assert rows["walking"]["pace_min_per_mi"] == 20.0


# --------------------------------------------------------------------------
# F. The window
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question,start,end", [
    ("How consistent has my sleep been over the last month?",
     "2026-08-02", "2026-08-31"),
    ("How was my sleep in the past month?", "2026-08-02", "2026-08-31"),
    ("What did I do in the last week?", "2026-08-25", "2026-08-31"),
])
def test_the_last_month_or_week_is_rolling_and_ends_on_as_of(
        question, start, end):
    window = resolve_window(question, date.fromisoformat(AS_OF))
    assert (window.start, window.end) == (start, end)


@pytest.mark.parametrize("question,start,end", [
    ("How did I sleep last month?", "2026-07-01", "2026-07-31"),
    ("How many steps did I average last week?", "2026-08-24", "2026-08-30"),
])
def test_a_bare_last_month_or_week_is_still_the_calendar_period(
        question, start, end):
    window = resolve_window(question, date.fromisoformat(AS_OF))
    assert (window.start, window.end) == (start, end)


def test_a_model_window_a_month_early_is_corrected_on_a_dated_tool(
        monkeypatch, vault, seeded):
    """The census: sleep consistency, "the last month", model chose July."""
    _result, _facts, ledger = _ask(
        monkeypatch, vault,
        "How consistent has my sleep been over the last month?",
        [("get_sleep_regularity", {"start": "2026-07-01",
                                   "end": "2026-07-31"})], "Nothing.")
    record = ledger[0]
    assert (record["arguments"]["start"], record["arguments"]["end"]) == (
        "2026-08-02", "2026-08-31")
    assert record["window_override"]["model_sent_window"] == {
        "start": "2026-07-01", "end": "2026-07-31"}
    assert record["window_override"]["phrase"] == "the last month"


def test_a_model_window_a_month_early_is_refused_on_a_period_tool(
        monkeypatch, vault, seeded):
    """``summarize_metric`` takes no start/end, so it cannot be corrected: the
    stale-window gate refuses an answer that cites the July period."""
    question = "How steady have my steps been over the last month?"

    def template(facts):
        return "Your step average was {%s}." % fact_template.fact_key(
            "step_count", "2026-07-04:2026-07-31", "mean")

    result, _facts, _ = _ask(
        monkeypatch, vault, question,
        [("summarize_metric", {"metric": "step_count",
                               "period": "2026-07-04:2026-07-31"})], template)
    assert result["mode"] == "fallback"
    assert result["verification"]["cause"] == "stale_window"


def test_the_same_answer_on_the_window_python_resolved_is_accepted(
        monkeypatch, vault, seeded):
    question = "How steady have my steps been over the last month?"
    period = "2026-08-02:2026-08-31"

    def template(facts):
        return "Your step average was {%s}." % fact_template.fact_key(
            "step_count", period, "mean")

    result, _facts, _ = _ask(
        monkeypatch, vault, question,
        [("summarize_metric", {"metric": "step_count", "period": period})],
        template)
    assert result["mode"] == "narration", result


# --------------------------------------------------------------------------
# The scale of a rating
# --------------------------------------------------------------------------

def _rating_facts():
    ledger = [{"sequence": 1, "tool_name": "summarize_metric", "result": {
        "metric": "subjective_sleep_quality", "unit": "score",
        "period": "2026-08-25:2026-08-31", "mean": 3.5, "n_days": 7}}]
    facts = fact_template.build_fact_set(ledger)
    return {**facts, **fact_template.build_rating_scale_facts(facts)}


def test_a_rating_metric_publishes_its_scale():
    facts = _rating_facts()
    scale = facts["scale:subjective_sleep_quality"]
    assert (scale["scale_min"], scale["scale_max"]) == (
        subjective.RATING_MIN, subjective.RATING_MAX) == (1, 5)
    assert scale["display"] == "out of 5"
    assert fact_template.rating_scale_guidance(facts)


def test_no_rating_read_means_no_scale_fact():
    ledger = [{"sequence": 1, "tool_name": "summarize_metric", "result": {
        "metric": "step_count", "unit": "count",
        "period": "2026-08-25:2026-08-31", "mean": 4000.0, "n_days": 7}}]
    facts = fact_template.build_fact_set(ledger)
    assert fact_template.build_rating_scale_facts(facts) == {}
    assert fact_template.rating_scale_guidance(facts) == ""


def _scan(text, facts):
    return fact_template.scan_template(
        text, facts, question="How did I sleep last night?",
        unbacked_numbers=True)


MEAN_KEY = "{fact|metric=subjective_sleep_quality|period=s:2026-08-25:" \
           "2026-08-31|field=mean}"


@pytest.mark.parametrize("phrase", [
    "out of five", "on a five-point scale", "on a scale of one to five",
    "on a scale from one to five",
])
def test_a_published_scale_may_be_spelled_but_only_that_scale(phrase):
    facts = _rating_facts()
    assert _scan(f"You rated your sleep {MEAN_KEY} {phrase}.", facts)["ok"]
    assert not _scan(f"You rated your sleep {MEAN_KEY} {phrase}.",
                     {k: v for k, v in facts.items()
                      if not k.startswith("scale:")})["ok"]


@pytest.mark.parametrize("text", [
    f"You rated it {MEAN_KEY} out of ten.",
    f"You rated it {MEAN_KEY}, and hit it on four out of five nights.",
    f"You rated it {MEAN_KEY}. Aim for seven to nine hours of sleep.",
])
def test_the_scale_exemption_covers_nothing_else(text):
    assert not _scan(text, _rating_facts())["ok"]


def test_the_scale_placeholder_renders_python_s_scale():
    facts = _rating_facts()
    text = f"You rated your sleep {MEAN_KEY} {{scale:subjective_sleep_quality}}."
    assert fact_template.interpolate_template(text, facts) == \
        "You rated your sleep 3.5 out of 5."
