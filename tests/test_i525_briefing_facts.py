"""Tests for health_advisor#525: get_briefing publishes citable figures, and
a Python-owned status covers the "data exists but nothing published"
gap -- see fact_template.build_briefing_facts / build_gathered_data_status_fact
and the module comment above them for the full design rationale.
"""
from __future__ import annotations

from health_advisor import chat, fact_template, llm, metrics
from tests.conftest import seed_metric


def _briefing_record(sequence=1, as_of="2026-08-21", *,
                     highlights=None, components=None, long_term=None,
                     extra_result=None):
    result = {
        "as_of": as_of, "scope": "daily",
        "coverage": [{"metric": "step_count", "status": "ok"}],
        "readiness": {
            "status": "ok", "score": 62, "band": "steady",
            "as_of": as_of, "latest_date": as_of, "stale_days": 0,
            "components": components if components is not None else {},
            "factors": [],
        },
        "movers": [], "movers_status": "nothing_moved",
        "movers_status_text": "nothing moved",
        "long_term": long_term if long_term is not None else [],
        "highlights": highlights if highlights is not None else [],
        "workout_focus": None,
        "suggestions": [], "talking_points": [],
    }
    if extra_result:
        result.update(extra_result)
    return {
        "sequence": sequence, "tool_name": "get_briefing",
        "result_elided": False, "result": result,
    }


def _rich_briefing_record(sequence=1, as_of="2026-08-21"):
    return _briefing_record(
        sequence=sequence, as_of=as_of,
        highlights=[
            {"metric": "step_count", "kind": "all_time_high",
             "value": 21500, "date": "2026-08-19"},
        ],
        components={"hrv": 58.0, "rhr": 71.0, "sleep": 64.0},
        long_term=[
            {"metric": "resting_heart_rate", "unit": "bpm",
             "this_month_avg": 55.2, "vs_3mo": -2.1, "vs_6mo": 3.4},
        ],
    )


# --- (a) a rich get_briefing ledger publishes real figures -----------------

def test_build_briefing_facts_publishes_highlight():
    facts = fact_template.build_briefing_facts([_rich_briefing_record()])
    key = fact_template.fact_key("step_count", "2026-08-19", "all_time_high")
    assert key in facts
    assert facts[key]["value"] == 21500
    assert facts[key]["metric"] == "step_count"


def test_build_briefing_facts_excludes_readiness_components():
    """Reverted after a live rendering (orchestrator review of b1e218a):
    a pseudo-metric key name stops a KEY collision, but not the model
    reading a bare subscore next to "resting heart rate" as the
    physiological value -- a live sample narrated "a resting heart rate of
    {fact|metric=briefing_readiness_rhr|...}", rendering a 0-100 subscore
    (e.g. 54) as if it were a bpm reading. See the module comment above
    build_briefing_facts for the full reasoning."""
    facts = fact_template.build_briefing_facts([_rich_briefing_record()])
    assert not any(str(key).startswith("fact|metric=briefing_readiness_")
                  for key in facts)
    for fact in facts.values():
        assert not str(fact.get("metric", "")).startswith(
            "briefing_readiness_")


def test_build_briefing_facts_excludes_daily_readiness_score_and_band():
    """The retired daily composite (get_weekly_readiness's docstring: "the
    daily 0-100 composite was retired") must not become citable through a
    different door."""
    facts = fact_template.build_briefing_facts([_rich_briefing_record()])
    for fact in facts.values():
        assert fact["metric"] != "readiness"
    keys_text = " ".join(facts)
    assert "field=score" not in keys_text
    assert "field=band" not in keys_text


def test_build_briefing_facts_publishes_long_term_month_avg():
    facts = fact_template.build_briefing_facts([_rich_briefing_record()])
    start, end = metrics.parse_period("30d", "2026-08-21")
    period = f"{start}:{end}"
    avg_key = fact_template.fact_key(
        "resting_heart_rate", period, "briefing_month_avg")
    assert facts[avg_key]["value"] == 55.2


def test_build_briefing_facts_excludes_long_term_comparison_fields():
    """vs_3mo/vs_6mo/vs_12mo stay unpublished: analysis.long_term compares
    this month to the single 30-day window 90/180/365 days ago (not a
    3/6/12-month AVERAGE), and a live sample narrated it as "above your
    three-month average" regardless -- plus the value has no recorded sign
    convention or `%` marker, so it rendered as bare digits. See the module
    comment above build_briefing_facts."""
    facts = fact_template.build_briefing_facts([_rich_briefing_record()])
    for key in facts:
        parsed = fact_template.parse_fact_key(key)
        if parsed is not None:
            assert not parsed[2].startswith("briefing_vs_")
    assert not any(fact["value"] in (-2.1, 3.4) for fact in facts.values())


def test_build_briefing_facts_does_not_publish_workout_focus_fields():
    """See the _WORKOUT_TOOLS module comment: these duplicate list_workouts
    at coarser rounding and are handled (excluded) there, not here."""
    record = _briefing_record(extra_result={
        "workout_focus": {"type": "running", "date": "2026-08-21",
                          "duration_min": 48.3, "distance_mi": 3.2},
    })
    facts = fact_template.build_briefing_facts([record])
    assert facts == {}


def test_build_briefing_facts_ignores_other_tools():
    ledger = [{"sequence": 1, "tool_name": "summarize_metric",
              "result_elided": False,
              "result": {"metric": "step_count", "period": "2026-08-21",
                         "mean": 9000}}]
    assert fact_template.build_briefing_facts(ledger) == {}


def test_build_briefing_facts_agreeing_duplicates_publish():
    ledger = [_rich_briefing_record(sequence=1),
              _rich_briefing_record(sequence=2)]
    facts = fact_template.build_briefing_facts(ledger)
    key = fact_template.fact_key("step_count", "2026-08-19", "all_time_high")
    assert facts[key]["value"] == 21500


def test_build_briefing_facts_conflicting_highlight_withheld():
    a = _rich_briefing_record(sequence=1)
    b = _rich_briefing_record(sequence=2)
    b["result"]["highlights"][0]["value"] = 999
    facts = fact_template.build_briefing_facts([a, b])
    key = fact_template.fact_key("step_count", "2026-08-19", "all_time_high")
    assert key not in facts


# --- (a, cont'd) the two published leaves render like a sibling tool's -----
#
# A synthetic vault, the REAL get_briefing tool, and the REAL sibling tools
# (get_latest, summarize_metric) that would answer a follow-up about the
# same metric -- not hand-built ledger dicts -- so this is not just checking
# that fact_template calls the same formatter, it is checking that the
# actual rendered strings agree.
def test_briefing_highlight_renders_like_get_latest(conn, tools):
    # An unbroken rise with the peak on the LAST day, so the all-time-high
    # date is the same day get_latest calls "latest" -- the two tools then
    # describe the identical (metric, day, value) triple.
    seed_metric(conn, "step_count", "2025-07-18",
               list(range(7000, 7000 + 400 * 10, 10)))
    out = tools.get_briefing(scope="deep", day="2026-08-21")
    ledger = [{"sequence": 1, "tool_name": "get_briefing",
              "result_elided": False, "result": out}]
    facts = fact_template.build_briefing_facts(ledger)
    key = fact_template.fact_key("step_count", "2026-08-21", "all_time_high")
    assert key in facts, sorted(facts)

    sibling = tools.get_latest(metric="step_count")
    assert sibling["latest_day"]["date"] == "2026-08-21"
    assert facts[key]["value"] == sibling["latest_day"]["value"]
    assert (facts[key]["display"]
           == sibling["latest_day"]["presentation"]["value"])


def test_briefing_month_avg_renders_like_summarize_metric(conn, tools):
    # A flat 400-day series: the requested 30d window and summarize_metric's
    # own data-clamped window coincide exactly, so both tools average the
    # identical rows.
    seed_metric(conn, "vo2_max", "2025-07-18", [41.5] * 400)
    out = tools.get_briefing(scope="deep", day="2026-08-21")
    ledger = [{"sequence": 1, "tool_name": "get_briefing",
              "result_elided": False, "result": out}]
    facts = fact_template.build_briefing_facts(ledger)
    start, end = metrics.parse_period("30d", "2026-08-21")
    period = f"{start}:{end}"
    key = fact_template.fact_key("vo2_max", period, "briefing_month_avg")
    assert key in facts, sorted(facts)

    sibling = tools.summarize_metric(metric="vo2_max", period="30d")
    assert sibling["period"] == period
    assert facts[key]["value"] == sibling["mean"]
    assert facts[key]["display"] == sibling["presentations"]["mean"]["value"]


def test_briefing_highlight_distance_renders_like_get_latest(conn, tools):
    """A second highlight metric, with a non-integer unit-preserving value
    (miles, one decimal place) -- distinct from step_count's bare count."""
    seed_metric(conn, "distance_walking_running", "2025-07-18",
               [round(3.0 + i * 0.01, 2) for i in range(400)])
    out = tools.get_briefing(scope="deep", day="2026-08-21")
    ledger = [{"sequence": 1, "tool_name": "get_briefing",
              "result_elided": False, "result": out}]
    facts = fact_template.build_briefing_facts(ledger)
    key = fact_template.fact_key(
        "distance_walking_running", "2026-08-21", "all_time_high")
    assert key in facts, sorted(facts)

    sibling = tools.get_latest(metric="distance_walking_running")
    assert sibling["latest_day"]["date"] == "2026-08-21"
    assert facts[key]["value"] == sibling["latest_day"]["value"]
    assert (facts[key]["display"]
           == sibling["latest_day"]["presentation"]["value"])


# --- (b) data exists, nothing publishable -> Python-owned status fact ------

def test_gathered_data_status_fact_when_nothing_publishable():
    fact = fact_template.build_gathered_data_status_fact(
        ledger_has_data=True, figures_published=False,
        tool_names=["get_briefing"])
    key = "status:gathered_data_uncited"
    assert key in fact
    text = fact[key]["display"]
    assert "get_briefing" in text
    assert "summarize_metric" in text
    # The text INSTRUCTS the model not to claim absence; it must not itself
    # assert absence as a bare claim outside that instruction.
    assert "do not say the vault has no data" in text.lower()
    assert text.lower().count("no data") == 1


def test_gathered_data_status_fact_names_every_tool_that_ran():
    fact = fact_template.build_gathered_data_status_fact(
        ledger_has_data=True, figures_published=False,
        tool_names=["get_briefing", "list_workouts"])
    text = fact["status:gathered_data_uncited"]["display"]
    assert "get_briefing" in text and "list_workouts" in text


def test_gathered_data_status_guidance_tells_model_to_quote_it():
    facts = fact_template.build_gathered_data_status_fact(
        ledger_has_data=True, figures_published=False, tool_names=[])
    guidance = fact_template.gathered_data_status_guidance(facts)
    assert "status:gathered_data_uncited" in guidance
    assert "never state or imply" in guidance.lower()


# --- (c) a thin/empty vault must still be allowed to say so ----------------

def test_gathered_data_status_fact_absent_without_data():
    assert fact_template.build_gathered_data_status_fact(
        ledger_has_data=False, figures_published=False,
        tool_names=["get_briefing"]) == {}


def test_gathered_data_status_fact_absent_when_figures_were_published():
    """A turn that DID publish a figure needs no status fact, even if it
    also ran a tool that produced nothing extra."""
    assert fact_template.build_gathered_data_status_fact(
        ledger_has_data=True, figures_published=True,
        tool_names=["get_briefing"]) == {}


def test_build_briefing_facts_on_empty_vault_publishes_nothing():
    """A genuinely thin vault's get_briefing (establishing_baseline,
    insufficient_history, empty highlights/long_term) must not manufacture
    a figure."""
    record = _briefing_record(
        as_of="2026-01-05",
        highlights=[], components={}, long_term=[],
        extra_result={
            "readiness": {
                "status": "establishing_baseline", "score": None,
                "band": None, "as_of": "2026-01-05", "latest_date": None,
                "stale_days": None, "components": {}, "factors": [],
            },
        })
    assert fact_template.build_briefing_facts([record]) == {}


# --- (d) a "no data" narration against a data-bearing ledger is refused ----
#
# health_advisor#525's live repro: the production capture's actual denial ("I don't have
# any recorded health data in your vault to summarize") and one of its four
# offline samples ("I have no logged figures for your activity metrics,
# workout sessions, or health trends") both passed every gate before this
# fix -- 0 figures, 0 claims, nothing to contradict. This pins the fix at
# the full chat.answer_question level: the retry's grounding check
# (chat._empty_narration_is_grounded, gated by chat._EMPTY_NARRATION_RE)
# must now refuse that exact sentence.

_REPRO_DENIAL = ("I have no logged figures for your activity metrics, "
                 "workout sessions, or health trends.")


def test_broad_denial_against_a_data_bearing_ledger_is_refused(
        monkeypatch, vault, conn):
    # `conn`'s fixture initializes the schema on disk at `vault`'s path (the
    # coverage read inside `_empty_narration_is_grounded` needs a real vault
    # file), and it is seeded with real rows so `chat._no_data_answer`'s
    # day-zero short-circuit does not pre-empt the mocked ledger below --
    # this test is specifically about a vault that HAS data.
    seed_metric(conn, "step_count", "2026-06-01", [8000.0] * 60)
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = [_rich_briefing_record()]
    responses = iter(["acknowledged", _REPRO_DENIAL, _REPRO_DENIAL])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    capture = []
    result = chat.answer_question(
        vault, "Tell me about my health?", as_of="2026-08-21",
        capture=capture)

    assert result["mode"] != "narration"
    assert _REPRO_DENIAL not in result["text"]


def test_original_live_capture_denial_wording_is_refused(monkeypatch, vault,
                                                          conn):
    """The literal sentence from the issue's live capture."""
    seed_metric(conn, "step_count", "2026-06-01", [8000.0] * 60)
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    denial = ("I'm sorry, but I don't have any recorded health data in "
              "your vault to summarize.")
    ledger = [_rich_briefing_record()]
    responses = iter(["acknowledged", denial, denial])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    result = chat.answer_question(
        vault, "Tell me about my health?", as_of="2026-08-21")

    assert result["mode"] != "narration"
    assert denial not in result["text"]


def test_broadened_regex_still_spares_genuine_coaching_prose():
    """Pinned false positive from 2026-08-31 (test_chat.py's sibling test);
    must still hold after adding figures/metrics to the noun list."""
    coaching = ("If you cannot find a sturdy table for inverted rows, add "
               "weight with dumbbells instead. Nothing is missing from a "
               "simple plan.")
    assert chat._EMPTY_NARRATION_RE.search(coaching) is None


def test_broadened_regex_catches_the_two_offline_repro_samples():
    assert chat._EMPTY_NARRATION_RE.search(_REPRO_DENIAL) is not None
    assert chat._EMPTY_NARRATION_RE.search(
        "there are no specific health metrics available in your records"
    ) is not None
