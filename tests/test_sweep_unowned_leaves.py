"""Sweep: which tool figures does the ask path's fact set leave unpublished?

consumer #576 was one instance of a class: a tool returns a figure a user would
ask for, the fact set publishes no slot for it, and a separate gate (the
available-figure denial check) can see the figure anyway. This test sweeps the
whole ask-path tool surface against one synthetic vault and PINS today's state,
so a tool that gains a new unpublished figure, or a new "gate sees a value with
no slot" metric, fails here with the tool and the path.

How it measures, and why the two halves are independent:

* Oracle. A plain JSON walk of each tool's real result (recorded through the
  ask path's own ledger wrapper) collects every numeric leaf. It knows nothing
  about metric ownership.
* Publisher. A leaf counts as published when a fact built by any of the six
  builders the ask path unions (metric, attachment, workout, briefing, declared,
  citation) names that leaf's path as its source, or when another published fact
  carries the same number under this leaf's own name (agreeing duplicates keep
  one source).
* Gate. ``chat._ledger_has_asked_metric_value`` is asked about every catalogue
  metric; a metric is reported when the gate says yes and none of the leaves it
  counted is a published source.

Every unpublished leaf must match a rule below. The labels are a human review of
the path, not something the code derives:

* ``UM``  user-meaningful: a figure a user plausibly asks for (a value, mean,
  total, change, count of sessions, duration, a percentage). A KNOWN GAP.
* ``AMB`` ambiguous: reviewed, could be asked for, could be noise.
* ``DW``  deliberately withheld: the fact-template code documents why.
* ``ST``  structural: coverage counts, thresholds, configuration, echoed
  arguments, ordinals. Not a gap.

A new leaf that matches no rule fails (classify it). A UM, AMB or DW rule that
no longer matches any unpublished leaf also fails: that gap was fixed, so delete
the rule and the pin ratchets. Structural rules are not checked for staleness.
"""
from __future__ import annotations

from unittest import mock

import pytest

from health_advisor import llm, mcp_server
from tests import sweep_unowned_leaves_lib as lib

KNOWN: dict[str, dict[str, str]] = {
    "list_available_metrics": {
        "ST": """
            .count .metrics[].n_days
        """,
    },
    "get_daily_series": {
        "ST": """
            .n .window_summary.n_days
        """,
    },
    "summarize_metric": {
        "UM": """
            .delta_vs_baseline .trend_per_week
        """,
        "ST": """
            .n_days .recent_window_days
        """,
    },
    "get_weekly_series": {
        "UM": """
            .total
        """,
        "AMB": """
            .expected_training_effect_bpm.max .expected_training_effect_bpm.min .noise_floor.rho
            .noise_floor.sd_day .weeks[].mdc95 .weeks[].rho .weeks[].sd_day
        """,
        "ST": """
            .count .expected_training_effect_bpm.citation.year .noise_floor.n_consecutive_pairs
            .noise_floor.n_days .noise_floor.window_days .weeks[].n_days
        """,
    },
    "get_block_comparison": {
        "UM": """
            .diff
        """,
        "AMB": """
            .blocks.previous.sd .blocks.recent.sd .mdc95
        """,
        "ST": """
            .block_weeks .blocks.previous.n .blocks.recent.n .blocks.previous.required_n
            .blocks.recent.required_n
        """,
    },
    "compare_periods": {
        "UM": """
            .mean_delta .mean_delta_pct
        """,
        "ST": """
            .period_a.n_days .period_b.n_days
        """,
    },
    "get_intraday": {
        "UM": """
            .buckets[].value
        """,
        "ST": """
            .bucket_hours .buckets[].hour .buckets[].n
        """,
    },
    "get_hr_zones": {
        "UM": """
            .windows[].above[].minutes .windows[].above[].pct_samples .windows[].avg_heart_rate
            .windows[].bands[].minutes .windows[].bands[].pct_samples .windows[].max_heart_rate
        """,
        "AMB": """
            .windows[].covered_min .windows[].min_heart_rate .windows[].uncovered_min
            .windows[].window_min
        """,
        "ST": """
            .count .excluded_count .thresholds[] .windows[].above[].n_samples
            .windows[].above[].threshold .windows[].bands[].n_samples
            .windows[].bands[].upper_inclusive .windows[].bands[].lower_exclusive
            .windows[].n_samples
        """,
    },
    "list_workouts": {
        "UM": """
            .count .total_in_range .workout_counts[].count .workouts[].energy_kcal
        """,
        "DW": """
            .workouts[].avg_heart_rate
        """,
        "ST": """
            .excluded_count .limit .workouts[].n_segments
        """,
    },
    "get_workout_segments": {
        "UM": """
            .workouts[].alternate_segmentations[].segments[].avg_heart_rate
            .workouts[].alternate_segmentations[].segments[].duration_min
            .workouts[].alternate_segmentations[].segments[].max_heart_rate
            .workouts[].duration_min .workouts[].elapsed_min .workouts[].segments[].duration_min
            .workouts[].segments[].avg_heart_rate .workouts[].segments[].max_heart_rate
        """,
        "AMB": """
            .count .workouts[].alternate_segmentations[].covered_min
            .workouts[].auto_pause_count .workouts[].covered_min
        """,
        "ST": """
            .excluded_count .workouts[].alternate_segmentations[].n_segments
            .workouts[].alternate_segmentations[].segments[].n .workouts[].n_segments
            .workouts[].segments[].n
        """,
    },
    "get_impact_volume": {
        "UM": """
            .periods[].jog_miles .periods[].jog_pace_min_per_mi .periods[].walk_miles
            .periods[].walk_minutes .periods[].jog_change_pct
            .block_comparison.change.mean_delta .block_comparison.change.total_delta
            .block_comparison.change.total_delta_pct
        """,
        "AMB": """
            .jog_threshold_sensitivity[].jog_minutes
        """,
        "ST": """
            .count .jog_cadence_threshold_steps_per_min .jog_near_threshold.buckets_near_cutoff
            .jog_near_threshold.jog_buckets .jog_near_threshold.pct_of_jog_buckets
            .jog_near_threshold.within_steps_per_min
            .jog_threshold_sensitivity[].cadence_min_steps_per_min
            .jog_threshold_sensitivity[].jog_buckets .periods[].days_covered
            .block_comparison.blocks.prior.weeks[].days_covered
            .block_comparison.blocks.prior.weeks[].days_expected
            .block_comparison.blocks.recent.weeks[].days_covered
            .block_comparison.blocks.recent.weeks[].days_expected
            .block_comparison.completeness.partial_trailing_week.days_covered
            .block_comparison.completeness.partial_trailing_week.days_expected
            .block_comparison.weeks_per_block
        """,
    },
    "get_sleep_regularity": {
        "UM": """
            .interval_regularity.match_pct .plan_compliance.inside_anchor
            .plan_compliance.inside_anchor_pct
        """,
        "AMB": """
            .calibration.bias_sd_minutes .calibration.limits_of_agreement[]
            .calibration.mean_bias_minutes .calibration.mean_proxy_sleep_minutes
            .calibration.mean_staged_sleep_minutes .cosinor.acrophase_days .cosinor.amplitude
            .cosinor.drift_hours_per_year .cosinor.mesor .midpoint_variability.days[].sd_hours
            .plan_compliance.past_limit .plan_compliance.social_nights
        """,
        "ST": """
            .calibration.n_days .cosinor.era_days .cosinor.eras_found .cosinor.n_days
            .interval_regularity.n_pairs .plan_compliance.anchor_h .plan_compliance.n_nights
            .plan_compliance.social_limit_h
        """,
    },
    "get_training_load_detail": {
        "UM": """
            .days[].load .live_acwr.acute_7d .live_acwr.acwr .live_acwr.chronic_weekly_avg
        """,
        "AMB": """
            .live_acwr.workout_mix_28d[].count
        """,
        "ST": """
            .days[].sessions .days[].sessions_nested_skipped .days[].sessions_without_hr
            .days_unknown .live_acwr.min_required .live_acwr.n_days .live_acwr.window_days
            .live_acwr.n_recent_days .live_acwr.n_worn_days
        """,
    },
    "get_run_form": {
        "UM": """
            .efficiency_change.change_pct .walk_structure.mean_bout_minutes
            .walk_structure.walk_bouts .walk_structure.walk_fraction
            .walk_structure.walk_minutes
        """,
        "AMB": """
            .efficiency_change.first_half_efficiency .efficiency_change.second_half_efficiency
            .personal_reference.median_change_pct
            .personal_reference.minimum_detectable_change_pct .personal_reference.p10
            .personal_reference.p90 .walk_structure.first_half_walk_fraction
            .walk_structure.second_half_walk_fraction
        """,
        "ST": """
            .efficiency_change.buckets_dropped_implausible .efficiency_change.first_half_buckets
            .efficiency_change.second_half_buckets .personal_reference.n_sessions
            .band_min_per_mi[]
        """,
    },
    "get_briefing": {
        "UM": """
            .talking_points[].numbers[] .training_load.acute_7d .training_load.acwr
            .training_load.chronic_weekly_avg .trends.hrv_per_week .trends.rhr_per_week
            .workout_focus.energy_kcal
        """,
        "AMB": """
            .readiness.factors[].pct .readiness.factors[].target
            .training_load.workout_mix_28d[].count
        """,
        "DW": """
            .readiness.components.hrv .readiness.components.rhr .readiness.components.sleep
            .readiness.score .workout_focus.duration_min .long_term[].vs_3mo
        """,
        "ST": """
            .coverage[].n_days .coverage[].recent_days .coverage[].recent_fraction
            .coverage[].window_days .readiness.factors[].age_days .readiness.stale_days
            .training_load.n_days .training_load.n_recent_days .training_load.n_worn_days
            .trends.early_warning.hrv_threshold_per_week
            .trends.early_warning.rhr_threshold_per_week .trends.early_warning.window_days
        """,
    },
    "correlate_metrics": {
        "UM": """
            .pearson_r .spearman_rho
        """,
        "AMB": """
            .coverage_x_pct .coverage_y_pct .pearson_ci95[] .pearson_p .spearman_ci95[]
            .spearman_p
        """,
        "ST": """
            .dropped_low_wear .lag_days .n_pairs .window_days
        """,
    },
    "scan_correlations": {
        "UM": """
            .results[].pearson_r .results[].spearman_rho
        """,
        "AMB": """
            .passed_fdr_count .results[].q_value .results[].spearman_p .tested_count
        """,
        "ST": """
            .fdr_q .lags[] .results[].dropped_low_wear .results[].lag_days .results[].n_pairs
        """,
    },
    "get_subjective": {
        "AMB": """
            .count
        """,
    },
    "food_lookup": {
        "UM": """
            .items[].carb_g .items[].fat_g .items[].kcal .items[].protein_g .items[].serving_g
        """,
        "AMB": """
            .count
        """,
        "ST": """
            .items[].confirmed
        """,
    },
    "food_meal_total": {
        "UM": """
            .carb_g .fat_g .items[].kcal .kcal .protein_g
        """,
        "ST": """
            .items[].confirmed .items[].servings
        """,
    },
    "get_block_structure": {
        "UM": """
            .sessions[].avg_hr_all_jog .sessions[].avg_hr_longest_block .sessions[].duration_min
        """,
        "AMB": """
            .hr_ceiling_for_qualifying .sessions[].reps[].mean_hr
            .sessions[].reps[].mean_pace_min_per_mi .sessions[].reps[].minutes
            .sessions[].unbridged_min
        """,
        "ST": """
            .excluded_count
        """,
    },
    "get_weekly_readiness": {
        "UM": """
            .score
        """,
        "AMB": """
            .components.hrv .components.rhr .components.sleep
        """,
        "ST": """
            .continuity.active_days .continuity.days .continuity.walk_minutes_floor .n_days
        """,
    },
    "get_benchmark_series": {
        "UM": """
            .stages[].median_hr_last_two_min .stages[].pace_min_per_mi
        """,
        "AMB": """
            .count .heat_effect_bpm_per_c.value .stages[].dew_point_c .stages[].temp_c
        """,
        "ST": """
            .heat_effect_bpm_per_c.citation.year .stages[].stage
        """,
    },
    "get_monthly_running_power": {
        "UM": """
            .mean_power_w
        """,
        "ST": """
            .pace_band_min_per_mi[]
        """,
    },
    "get_workout_coverage": {
        "ST": """
            .unmatched_routes .weather_status.fetched .weather_status.no_route
            .weather_status.pending .with_route .with_weather .workouts
        """,
    },
    "get_ingest_diagnostics": {
        "AMB": """
            .arrived .stored .stored_by_source[].count
        """,
    },
}

# Metrics for which the denial gate sees a figure that no published fact cites,
# per tool. Pinned both ways: a new entry is a new #576-shaped hole, a missing
# one is a fix that should shrink this table.
GATE_SEES_WITHOUT_SLOT: dict[str, set[str]] = {
    "get_intraday": {"heart_rate", "step_count"},
}

GAP_LABELS = ("UM", "AMB", "DW")


def _rules() -> list[tuple[str, str, str]]:
    return [(tool, pat, label)
            for tool, by_label in KNOWN.items()
            for label, text in by_label.items()
            for pat in text.split()]


def classify(tool: str, pattern: str) -> tuple[str, str] | None:
    """(label, rule) for the most specific rule covering this leaf pattern."""
    best = None
    for rule_tool, rule, label in _rules():
        if rule_tool != tool:
            continue
        if pattern == rule or pattern.startswith(rule + "."):
            if best is None or len(rule) > len(best[1]):
                best = (label, rule)
    return best


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep")
    ctx, days = lib.build_vault(root / "sweep.db")
    registry = lib.base_registry(ctx)
    runs = []
    # The one tool the demo vault cannot feed (it needs route-backed outdoor
    # runs with power records) has its data function stubbed; the tool, its
    # result shape and the ledger path are real.
    with mock.patch.object(mcp_server.RF, "monthly_running_power",
                           lambda *a, **k: 205.25):
        for tool, label, args in lib.calls(days):
            ledger = lib.run_one(registry, tool, args, root / "ledger.jsonl")
            result = ledger[0]["result"]
            analysis = lib.analyse(ledger)
            runs.append({"tool": tool, "label": label, "args": args,
                         "result": result, **analysis})
    return runs


def _where(run) -> str:
    return f"{run['tool']}({run['label']})"


def test_the_sweep_exercises_every_ask_exposed_engine_tool_with_real_data(sweep):
    exposed = {n for n in llm.COACH_TOOLS
               if n not in lib.ASK_EXPOSED_SKIP
               and n not in llm.HOST_SUPPLIED_TOOLS}
    covered = {run["tool"] for run in sweep}
    assert exposed - covered == set(), (
        f"ask-exposed tools with no sweep call: {sorted(exposed - covered)}")
    for run in sweep:
        result = run["result"]
        assert not run["elided"], f"{_where(run)}: result was elided"
        assert not (isinstance(result, dict) and result.get("error")), (
            f"{_where(run)} returned an error, so the sweep measured nothing: "
            f"{result.get('error')}")
        assert run["leaves"], f"{_where(run)} returned no numeric leaf"


def test_the_instrument_can_see_a_published_figure(sweep):
    """Guards the oracle against reading everything as unpublished."""
    by = {(r["tool"], r["label"]): r for r in sweep}
    latest = by[("get_latest", "metric=vo2_max")]
    assert [l["path"] for l in latest["leaves"]] == ["$.result.latest_day.value"]
    assert latest["unpublished"] == [], (
        "get_latest's latest_day.value is not published as a fact; this is the "
        "#576 defect (the ownership rule in deepdive_verify._ledger_scopes)")
    series = by[("get_daily_series", "metric=step_count")]
    assert sum(l["published"] for l in series["leaves"]) > 50


def test_no_tool_gains_an_unclassified_unpublished_leaf(sweep):
    unclassified: dict[tuple[str, str], str] = {}
    for run in sweep:
        for leaf in run["unpublished"]:
            if classify(run["tool"], leaf["pattern"].replace("$.result", "")) is None:
                unclassified.setdefault(
                    (run["tool"], leaf["pattern"]),
                    f"{leaf['path']} = {leaf['value']!r} (call {run['label']!r}, "
                    f"{'owned by a metric but no period/presentation key' if leaf['owned'] else 'owned by no metric'})")
    assert not unclassified, (
        "These numeric tool-result leaves are not published as fact-set slots "
        "and are not in the reviewed KNOWN table. Either publish them (an "
        "ownership rule in deepdive_verify._ledger_scopes, a field_metrics map "
        "or publishable_facts in the tool) or classify them in KNOWN with the "
        "label UM/AMB/DW/ST after reading the path:\n  "
        + "\n  ".join(f"{tool}: {pat}  e.g. {ex}"
                       for (tool, pat), ex in sorted(unclassified.items())))


def test_every_known_gap_rule_still_matches_an_unpublished_leaf(sweep):
    matched = set()
    for run in sweep:
        for leaf in run["unpublished"]:
            hit = classify(run["tool"], leaf["pattern"].replace("$.result", ""))
            if hit:
                matched.add((run["tool"], hit[1]))
    stale = [f"{tool}: {pat} ({label})" for tool, pat, label in _rules()
             if label in GAP_LABELS and (tool, pat) not in matched]
    assert not stale, (
        "These known-gap rules match no unpublished leaf any more, so the gap "
        "was fixed (or the leaf moved). Delete the rule so a regression of it "
        "fails:\n  " + "\n  ".join(sorted(stale)))


def test_the_denial_gate_never_sees_a_figure_that_has_no_slot_beyond_the_pin(sweep):
    seen: dict[str, set[str]] = {}
    for run in sweep:
        if run["gate_no_slot"]:
            seen.setdefault(run["tool"], set()).update(run["gate_no_slot"])
    new = {tool: sorted(metrics - GATE_SEES_WITHOUT_SLOT.get(tool, set()))
           for tool, metrics in seen.items()
           if metrics - GATE_SEES_WITHOUT_SLOT.get(tool, set())}
    assert not new, (
        "chat._ledger_has_asked_metric_value sees a value for these metrics "
        "but no published fact cites any leaf it saw, so a denial would be "
        "refused while the model has no slot to state the figure (#576 shape): "
        f"{new}")
    fixed = {tool: sorted(metrics - seen.get(tool, set()))
             for tool, metrics in GATE_SEES_WITHOUT_SLOT.items()
             if metrics - seen.get(tool, set())}
    assert not fixed, (
        f"these pinned gate holes no longer reproduce, remove them from "
        f"GATE_SEES_WITHOUT_SLOT: {fixed}")
