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
