"""Tool figures the closed fact set had no slot for (downstream issue #578).

``tests/test_sweep_unowned_leaves.py`` pins which numeric tool leaves the ask
path's fact set leaves unpublished. Each section below is one item of that
issue: the figures a user plausibly asks for, published now, checked against
LITERAL expected keys and values on the sweep's own synthetic vault (the
invented catalog rows and demo workouts in ``sweep_unowned_leaves_lib``),
never against a value recomputed by the code under test.
"""
from __future__ import annotations

import pytest

from health_advisor import fact_template
from tests import sweep_unowned_leaves_lib as lib

PUB = fact_template.declared_fact_key


@pytest.fixture(scope="module")
def vault(tmp_path_factory):
    root = tmp_path_factory.mktemp("i578")
    ctx, days = lib.build_vault(root / "sweep.db")
    return {"registry": lib.base_registry(ctx), "days": days, "root": root}


def _ledger(vault, tool, args):
    return lib.run_one(vault["registry"], tool, args,
                       vault["root"] / "ledger.jsonl")


def _unpublished_paths(ledger) -> set[str]:
    return {leaf["path"] for leaf in lib.analyse(ledger)["unpublished"]}


# --- item 9: food totals ------------------------------------------------------

OATS = {"kcal": 150.0, "protein_g": 5.0, "carb_g": 20.0, "fat_g": 3.0,
        "serving_g": 100.0}


def test_food_lookup_publishes_each_items_per_serving_figures(vault):
    ledger = _ledger(vault, "food_lookup", {"query": "oats"})
    facts = fact_template.build_declared_facts(ledger)
    got = {key: (fact["value"], fact["unit"], fact["source"].get("paths"))
           for key, fact in facts.items()}
    assert got == {
        PUB("food/sweep-oats/per_serving/kcal"):
            (150.0, "kcal", ["$.result.items[0].kcal"]),
        PUB("food/sweep-oats/per_serving/protein_g"):
            (5.0, "g", ["$.result.items[0].protein_g"]),
        PUB("food/sweep-oats/per_serving/carb_g"):
            (20.0, "g", ["$.result.items[0].carb_g"]),
        PUB("food/sweep-oats/per_serving/fat_g"):
            (3.0, "g", ["$.result.items[0].fat_g"]),
        PUB("food/sweep-oats/per_serving/serving_g"):
            (100.0, "g", ["$.result.items[0].serving_g"]),
    }
    kcal = facts[PUB("food/sweep-oats/per_serving/kcal")]
    assert kcal["label"] == ("Calories in one serving (1 cup) of Sweep oats, "
                             "from the food catalog (source: estimate)")
    assert kcal["display"] == "150"
    assert _unpublished_paths(ledger) == {"$.result.count",
                                          "$.result.items[0].confirmed"}


def test_food_meal_total_publishes_the_totals_and_each_items_calories(vault):
    ledger = _ledger(vault, "food_meal_total", {"items": [
        {"item_key": "sweep-oats", "servings": 1.5},
        {"item_key": "sweep-yogurt", "servings": 1}]})
    facts = fact_template.build_declared_facts(ledger)
    by_tail = {key.split("/", 3)[-1] if key.startswith(PUB("food/meal/"))
               else key: (fact["value"], fact["source"]["paths"])
               for key, fact in facts.items()}
    # 1.5 x (150 kcal, 5 g, 20 g, 3 g) + 1 x (110 kcal, 5 g, 20 g, 3 g)
    assert by_tail == {
        "kcal": (335.0, ["$.result.kcal"]),
        "protein_g": (12.5, ["$.result.protein_g"]),
        "carb_g": (50.0, ["$.result.carb_g"]),
        "fat_g": (7.5, ["$.result.fat_g"]),
        PUB("food/sweep-oats/servings/1.5/kcal"):
            (225.0, ["$.result.items[0].kcal"]),
        PUB("food/sweep-yogurt/servings/1/kcal"):
            (110.0, ["$.result.items[1].kcal"]),
    }
    total = next(f for k, f in facts.items() if k.endswith("/kcal")
                 and k.startswith(PUB("food/meal/")))
    assert total["label"] == (
        "Total calories in the meal of 1.5 x Sweep oats (1 cup) + 1 x Sweep "
        "yogurt (1 cup), summed from the food catalog (weakest source: "
        "estimate)")
    assert _unpublished_paths(ledger) == {
        "$.result.items[0].confirmed", "$.result.items[0].servings",
        "$.result.items[1].confirmed", "$.result.items[1].servings"}


def test_the_same_meal_twice_publishes_once_and_another_meal_apart(vault):
    meal = {"items": [{"item_key": "sweep-oats", "servings": 2}]}
    other = {"items": [{"item_key": "sweep-oats", "servings": 3}]}
    ledger = (_ledger(vault, "food_meal_total", meal)
              + _ledger(vault, "food_meal_total", meal)
              + _ledger(vault, "food_meal_total", other))
    for seq, record in enumerate(ledger):
        record["sequence"] = seq
    totals = sorted(f["value"] for k, f in
                    fact_template.build_declared_facts(ledger).items()
                    if k.startswith(PUB("food/meal/")) and k.endswith("/kcal"))
    assert totals == [300.0, 450.0]


@pytest.mark.parametrize("restates", [
    "$.result.kcal_total",          # names no leaf
    "$.result.protein_g",           # a leaf with a different number
    "$.result.kcal.x",              # descends through a number
    "$.result.kcal extra",          # a parse that would drop characters
    "kcal",                         # not a result path
    7,                              # not a string
])
def test_a_restated_path_that_does_not_hold_the_figure_drops_the_entry(
        restates):
    record = {"sequence": 1, "tool_name": "host_tool", "result_elided": False,
              "result": {"kcal": 335.0, "protein_g": 12.5,
                         fact_template.DECLARED_FACTS_RESULT_KEY: [
                             {"id": "meal/kcal", "label": "Meal calories",
                              "value": 335.0, "unit": "kcal",
                              "restates": restates},
                             {"id": "meal/plain", "label": "No restated leaf",
                              "value": 1.0}]}}
    facts = fact_template.build_declared_facts([record])
    assert set(facts) == {PUB("meal/plain")}
    assert facts[PUB("meal/plain")]["source"] == {
        "sequence": 1, "path": "$.result.publishable_facts[1].value"}


# --- item 5: list_workouts counts and energy ----------------------------------

WINDOW = "workouts/2026-06-03:2026-08-31"


def test_list_workouts_counts_name_the_leaves_they_restate(vault):
    ledger = _ledger(vault, "list_workouts", {"limit": 3})
    facts = fact_template.build_declared_facts(ledger)
    counts = {key: (fact["value"], fact["source"].get("paths"))
              for key, fact in facts.items() if key.endswith("/count")}
    assert counts == {
        PUB(f"{WINDOW}/all/count"): (65, ["$.result.total_in_range"]),
        PUB(f"{WINDOW}/cycling/count"):
            (12, ["$.result.workout_counts[0].count"]),
        PUB(f"{WINDOW}/running/count"):
            (31, ["$.result.workout_counts[1].count"]),
        PUB(f"{WINDOW}/traditional_strength_training/count"):
            (11, ["$.result.workout_counts[2].count"]),
        PUB(f"{WINDOW}/walking/count"):
            (11, ["$.result.workout_counts[3].count"]),
    }


def test_list_workouts_rows_publish_their_energy(vault):
    ledger = _ledger(vault, "list_workouts", {"limit": 3})
    facts = fact_template.build_workout_facts(ledger)
    energy = {key: (fact["value"], fact["unit"], fact["source"]["path"])
              for key, fact in facts.items() if key.endswith("energy_kcal")}
    assert energy == {
        fact_template.workout_fact_key(
            "2026-08-31|traditional_strength_training", "energy_kcal"):
            (117.9, "kcal", "$.result.workouts[0].energy_kcal"),
        fact_template.workout_fact_key("2026-08-27|running", "energy_kcal"):
            (397.7, "kcal", "$.result.workouts[1].energy_kcal"),
        fact_template.workout_fact_key("2026-08-25|running", "energy_kcal"):
            (334.4, "kcal", "$.result.workouts[2].energy_kcal"),
    }


def test_list_workouts_page_size_stays_unpublished(vault):
    """``count`` is the number of rows ``limit`` let through (3 here), not the
    range's 65: it must never be citable as "how many workouts"."""
    ledger = _ledger(vault, "list_workouts", {"limit": 3})
    assert ledger[0]["result"]["count"] == 3
    unpublished = _unpublished_paths(ledger)
    assert unpublished == {
        "$.result.count", "$.result.limit", "$.result.excluded_count",
        *(f"$.result.workouts[{i}].{field}" for i in range(3)
          for field in ("avg_heart_rate", "n_segments"))}


# --- item 8: heart-rate zones -------------------------------------------------

def test_hr_zones_publish_time_above_and_in_band_per_window(vault):
    run_day, _ = vault["days"]
    assert run_day == "2026-08-27"
    ledger = _ledger(vault, "get_hr_zones", {"day": run_day})
    facts = fact_template.build_declared_facts(ledger)
    day = "hr_zones/2026-08-27/day"
    run = "hr_zones/2026-08-27/workout/running/17:38:00"
    got = {key: (fact["value"], fact["unit"], fact["source"]["paths"])
           for key, fact in facts.items()}
    expected = {
        PUB(f"{day}/above/150/minutes"):
            (3.0, "min", ["$.result.windows[0].above[1].minutes"]),
        PUB(f"{day}/above/150/pct_samples"):
            (5.1, "%", ["$.result.windows[0].above[1].pct_samples"]),
        PUB(f"{day}/band/min-135/minutes"):
            (28.0, "min", ["$.result.windows[0].bands[0].minutes"]),
        PUB(f"{day}/avg_heart_rate"):
            (132.0, "bpm", ["$.result.windows[0].avg_heart_rate"]),
        PUB(f"{run}/above/150/minutes"):
            (3.0, "min", ["$.result.windows[1].above[1].minutes"]),
        PUB(f"{run}/above/150/pct_samples"):
            (5.7, "%", ["$.result.windows[1].above[1].pct_samples"]),
        PUB(f"{run}/band/135-150/minutes"):
            (42.0, "min", ["$.result.windows[1].bands[1].minutes"]),
        PUB(f"{run}/band/170-max/minutes"):
            (0.0, "min", ["$.result.windows[1].bands[4].minutes"]),
        PUB(f"{run}/avg_heart_rate"):
            (141.0, "bpm", ["$.result.windows[1].avg_heart_rate"]),
        PUB(f"{run}/max_heart_rate"):
            (153.0, "bpm", ["$.result.windows[1].max_heart_rate"]),
    }
    assert {k: got.get(k) for k in expected} == expected
    # 2 windows x (avg, max, 4 thresholds x 2, 5 bands x 2)
    assert len(facts) == 40
    assert facts[PUB(f"{run}/above/150/minutes")]["label"] == (
        "Minutes with heart rate above 150 bpm during the running workout "
        "starting 17:38:00 on Thu Aug 27")
    # The day window's samples cover 73 of 1380 minutes: its minutes say so.
    assert facts[PUB(f"{day}/above/150/minutes")]["label"].endswith(
        "(counted only where heart-rate samples exist; they do not cover the "
        "whole window)")
    assert not any(path.endswith(("minutes", "pct_samples", "heart_rate"))
                   and not path.endswith("min_heart_rate")
                   for path in _unpublished_paths(ledger))


def test_hr_zones_custom_thresholds_publish_under_their_own_edges(vault):
    ledger = _ledger(vault, "get_hr_zones", {
        "day": "2026-08-27", "workout_type": "running",
        "thresholds": "120,140,160"})
    facts = fact_template.build_declared_facts(ledger)
    run = "hr_zones/2026-08-27/workout/running/17:38:00"
    assert facts[PUB(f"{run}/above/140/minutes")]["value"] == 41.33
    assert facts[PUB(f"{run}/band/120-140/minutes")]["value"] == 6.67
    assert PUB(f"{run}/above/150/minutes") not in facts


# --- item 4: impact volume's sibling figures ----------------------------------

def test_impact_volume_siblings_publish_under_their_own_ids(vault):
    ledger = _ledger(vault, "get_impact_volume", {
        "start": "2026-07-27", "end": "2026-08-30", "weeks_per_block": 2})
    facts = fact_template.build_declared_facts(ledger)
    week = "impact/2026-08-03:2026-08-09"
    blocks = "impact/blocks/2026-08-03:2026-08-16/2026-08-17:2026-08-30/jog_minutes"
    got = {key: (fact["value"], fact["unit"], fact["display"],
                 fact["source"]["paths"]) for key, fact in facts.items()}
    expected = {
        PUB(f"{week}/jog_miles"):
            (7.65, "mi", "7.65", ["$.result.periods[1].jog_miles"]),
        PUB(f"{week}/jog_pace_min_per_mi"):
            (10.4, "min/mi", "10:24 min/mi",
             ["$.result.periods[1].jog_pace_min_per_mi"]),
        PUB(f"{week}/walk_miles"):
            (11.24, "mi", "11.24", ["$.result.periods[1].walk_miles"]),
        PUB(f"{week}/walk_minutes"):
            (198.3, "min", "198.3", ["$.result.periods[1].walk_minutes"]),
        PUB(f"{week}/jog_change_pct"):
            (0.8, "%", "0.8", ["$.result.periods[1].jog_change_pct"]),
        PUB(f"{blocks}/mean_delta"):
            (24.7, "min", "24.7",
             ["$.result.block_comparison.change.mean_delta"]),
        PUB(f"{blocks}/total_delta"):
            (49.4, "min", "49.4",
             ["$.result.block_comparison.change.total_delta"]),
        PUB(f"{blocks}/total_delta_pct"):
            (27.5, "%", "27.5",
             ["$.result.block_comparison.change.total_delta_pct"]),
    }
    assert {k: got.get(k) for k in expected} == expected
    # The first week has no predecessor in range: its change is null and
    # publishes nothing.
    assert PUB("impact/2026-07-27:2026-08-02/jog_change_pct") not in facts
    assert facts[PUB(f"{week}/jog_miles")]["label"] == (
        "Distance covered while jogging (minutes the tool classifies as "
        "jogging) in the week of Mon Aug 3 to Sun Aug 9")
    assert facts[PUB(f"{blocks}/mean_delta")]["label"] == (
        "Change in average weekly jog minutes from the 2 weeks Mon Aug 3 to "
        "Sun Aug 16 to the 2 weeks Mon Aug 17 to Sun Aug 30")


def test_impact_volume_siblings_never_inherit_the_jog_minutes_metric(vault):
    """No sibling leaf becomes a ``jog_minutes`` metric fact (miles labelled as
    minutes); the metric facts are still exactly the period rows' jog
    minutes and the block weeks' values."""
    ledger = _ledger(vault, "get_impact_volume", {
        "start": "2026-07-27", "end": "2026-08-30", "weeks_per_block": 2})
    metric_paths = {fact["source"]["path"] for fact in
                    fact_template.build_fact_set(ledger).values()
                    if (fact.get("source") or {}).get("path")}
    assert all(path.endswith((".jog_minutes", ".value", ".mean", ".total"))
               for path in metric_paths), sorted(metric_paths)
    assert _unpublished_paths(ledger) & {
        f"$.result.periods[{i}].{field}" for i in range(5)
        for field in ("jog_miles", "jog_pace_min_per_mi", "walk_miles",
                      "walk_minutes", "jog_change_pct")} == set()


def test_a_period_with_no_samples_publishes_no_zero():
    from health_advisor import mcp_server
    out = {"by": "week", "periods": [
        {"period": "2026-08-03:2026-08-09", "no_data": True, "partial": False,
         "jog_miles": 0.0, "walk_miles": 0.0, "walk_minutes": 0.0,
         "jog_pace_min_per_mi": None, "jog_change_pct": None}]}
    assert mcp_server._impact_facts(out, metric_units=False) == []
