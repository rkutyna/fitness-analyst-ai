"""food: the nutrition reference catalog — full-replace writes, provenance
tiers, and meal arithmetic that refuses to present a partial macro sum as a
total."""
import math

import pytest

from health_advisor import db as dbmod
from health_advisor import food as F


@pytest.fixture()
def wconn(tmp_path):
    conn = dbmod.connect(tmp_path / "t.db")
    dbmod.init_db(conn)
    yield conn
    conn.close()


def add_yogurt(conn, **over):
    kw = dict(item_key="tj-strained-greek-yogurt-plain",
              display_name="Strained Thick & Creamy Greek Yogurt, Plain",
              brand="Trader Joe's", aliases="greek yogurt|strained yogurt",
              serving_desc="3/4 cup", serving_g=170.0, kcal=170.0,
              protein_g=15.0, carb_g=7.0, fat_g=9.0,
              source="web", source_detail="https://example.test/tj-yogurt")
    kw.update(over)
    return F.add(conn, **kw)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_init_db_creates_the_catalog_with_its_columns(wconn):
    cols = {r["name"] for r in wconn.execute("PRAGMA table_info(food_catalog)")}
    assert cols == {"item_key", "display_name", "brand", "aliases",
                    "serving_desc", "serving_g", "kcal", "protein_g", "carb_g",
                    "fat_g", "source", "source_detail", "confirmed",
                    "verified_at", "notes"}


# --------------------------------------------------------------------------- #
# add() — validation
# --------------------------------------------------------------------------- #
def test_add_stores_and_returns_the_row(wconn):
    row = add_yogurt(wconn)
    assert row["kcal"] == 170.0 and row["protein_g"] == 15.0
    assert row["source"] == "web"
    assert row["confirmed"] == 0          # not confirmed unless said so
    assert row["verified_at"]


@pytest.mark.parametrize("over, msg", [
    ({"source": "vibes"}, "source"),
    ({"source_detail": "   "}, "source_detail"),
    ({"kcal": None}, "kcal"),
    ({"kcal": -1}, "kcal"),
    ({"protein_g": -2}, "protein_g"),
    ({"serving_g": 0}, "serving_g"),
    ({"item_key": "Has Spaces"}, "item_key"),
    ({"item_key": "UPPER"}, "item_key"),
    ({"item_key": ""}, "item_key"),
    ({"display_name": "  "}, "display_name"),
    ({"serving_desc": ""}, "serving_desc"),
    ({"confirmed": 2}, "confirmed"),
    ({"kcal": float("nan")}, "finite"),
    ({"fat_g": float("inf")}, "finite"),
])
def test_add_rejects_bad_input(wconn, over, msg):
    with pytest.raises(ValueError, match=msg):
        add_yogurt(wconn, **over)
    assert F.get(wconn, "tj-strained-greek-yogurt-plain") is None


def test_source_detail_is_required_the_way_manual_jog_why_is(wconn):
    with pytest.raises(ValueError):
        add_yogurt(wconn, source_detail="")


# --------------------------------------------------------------------------- #
# add() — full replace, not partial upsert
# --------------------------------------------------------------------------- #
def test_rewriting_an_item_clears_fields_that_were_not_supplied(wconn):
    add_yogurt(wconn)
    row = F.add(wconn, item_key="tj-strained-greek-yogurt-plain",
                display_name="TJ Nonfat Greek Yogurt", serving_desc="1 cup",
                kcal=120.0, source="label_text", source_detail="tub, 2026-08-17")
    # Everything not passed is now NULL — a catalog row is one complete
    # statement read off one label, never a merge of two.
    assert row["protein_g"] is None and row["carb_g"] is None
    assert row["fat_g"] is None and row["serving_g"] is None
    assert row["brand"] is None and row["aliases"] is None
    assert row["kcal"] == 120.0 and row["source"] == "label_text"


def test_relabelling_an_estimate_cannot_keep_its_estimated_macros(wconn):
    """The provenance bug that killed partial upsert: a guess must not acquire
    a better source label while its guessed numbers survive underneath."""
    F.add(wconn, item_key="tj-adobo-pork-shoulder",
          display_name="TJ Adobo Pork Shoulder", serving_desc="4 oz raw",
          kcal=210.0, protein_g=17.0, carb_g=4.0, fat_g=14.0,
          source="estimate", source_detail="proxied off Al Pastor diced pork")
    row = F.add(wconn, item_key="tj-adobo-pork-shoulder",
                display_name="TJ Adobo Pork Shoulder", serving_desc="4 oz raw",
                kcal=230.0, source="web", source_detail="https://example.test/x")
    assert row["source"] == "web"
    assert row["protein_g"] is None, "estimated macros survived a relabel"
    assert row["fat_g"] is None


def test_verified_at_moves_on_every_write(wconn):
    first = add_yogurt(wconn)["verified_at"]
    second = add_yogurt(wconn, kcal=171.0)["verified_at"]
    assert second >= first


def test_confirmed_round_trips(wconn):
    assert add_yogurt(wconn, confirmed=1)["confirmed"] == 1


# --------------------------------------------------------------------------- #
# search()
# --------------------------------------------------------------------------- #
def test_search_finds_by_display_name_case_insensitively(wconn):
    add_yogurt(wconn)
    assert len(F.search(wconn, "STRAINED")) == 1


def test_search_finds_by_alias(wconn):
    add_yogurt(wconn)
    hits = F.search(wconn, "greek yogurt")
    assert [h["item_key"] for h in hits] == ["tj-strained-greek-yogurt-plain"]


def test_empty_and_whitespace_queries_return_the_whole_catalog(wconn):
    add_yogurt(wconn)
    add_yogurt(wconn, item_key="tj-mini-flour-tortillas",
               display_name="Organic Mini Flour Tortillas", aliases=None)
    assert len(F.search(wconn, "")) == 2
    assert len(F.search(wconn, "   ")) == 2


def test_wildcards_in_the_query_are_escaped_not_honoured(wconn):
    add_yogurt(wconn)
    # Naive LIKE interpolation makes both of these match everything.
    assert F.search(wconn, "%") == []
    assert F.search(wconn, "_") == []
    add_yogurt(wconn, item_key="whey-100", display_name="100% Whey", aliases=None)
    assert [h["item_key"] for h in F.search(wconn, "100%")] == ["whey-100"]


def test_search_miss_returns_empty(wconn):
    add_yogurt(wconn)
    assert F.search(wconn, "adobo pork") == []


# --------------------------------------------------------------------------- #
# totals()
# --------------------------------------------------------------------------- #
def test_totals_sums_fractional_servings(wconn):
    add_yogurt(wconn)                                   # 170 kcal, 15/7/9
    out = F.totals(wconn, [("tj-strained-greek-yogurt-plain", 1.5)])
    assert out["kcal"] == pytest.approx(255.0)
    assert out["protein_g"] == pytest.approx(22.5)
    assert out["incomplete"] == []
    assert len(out["items"]) == 1


def test_a_missing_macro_makes_that_total_unknown_not_partial(wconn):
    """One item with 15 g protein and one with unknown protein must not
    report 15 g — that reads as a complete number and is not one."""
    add_yogurt(wconn)
    F.add(wconn, item_key="mystery-pita", display_name="Pita",
          serving_desc="1 large", kcal=220.0, carb_g=44.0,
          source="estimate", source_detail="bakery, no label")
    out = F.totals(wconn, [("tj-strained-greek-yogurt-plain", 1),
                           ("mystery-pita", 1)])
    assert out["kcal"] == pytest.approx(390.0)      # kcal is NOT NULL, always sums
    assert out["protein_g"] is None
    assert out["fat_g"] is None
    assert out["carb_g"] == pytest.approx(51.0)     # both known
    assert set(out["incomplete"]) == {"protein_g", "fat_g"}


def test_weakest_source_is_the_worst_tier_present(wconn):
    # Given best-first, a reversed comparison returns 'label_photo' and fails.
    F.add(wconn, item_key="a", display_name="A", serving_desc="1", kcal=1,
          source="label_photo", source_detail="photo")
    F.add(wconn, item_key="b", display_name="B", serving_desc="1", kcal=1,
          source="web", source_detail="url")
    F.add(wconn, item_key="c", display_name="C", serving_desc="1", kcal=1,
          source="estimate", source_detail="guess")
    assert F.totals(wconn, [("a", 1), ("b", 1), ("c", 1)])["weakest_source"] == "estimate"
    assert F.totals(wconn, [("a", 1), ("b", 1)])["weakest_source"] == "web"
    assert F.totals(wconn, [("a", 1)])["weakest_source"] == "label_photo"


def test_zero_serving_items_do_not_drag_down_the_weakest_tier(wconn):
    F.add(wconn, item_key="a", display_name="A", serving_desc="1", kcal=100,
          protein_g=10, carb_g=1, fat_g=1, source="label_photo", source_detail="p")
    F.add(wconn, item_key="c", display_name="C", serving_desc="1", kcal=50,
          source="estimate", source_detail="guess")
    out = F.totals(wconn, [("a", 1), ("c", 0)])
    assert out["kcal"] == pytest.approx(100.0)
    assert out["weakest_source"] == "label_photo"
    assert out["incomplete"] == []          # c's NULL macros contribute nothing


def test_totals_rejects_unknown_item_empty_list_and_bad_servings(wconn):
    add_yogurt(wconn)
    with pytest.raises(ValueError, match="unknown"):
        F.totals(wconn, [("no-such-item", 1)])
    with pytest.raises(ValueError, match="empty"):
        F.totals(wconn, [])
    with pytest.raises(ValueError, match="servings"):
        F.totals(wconn, [("tj-strained-greek-yogurt-plain", -1)])
    with pytest.raises(ValueError, match="finite"):
        F.totals(wconn, [("tj-strained-greek-yogurt-plain", math.inf)])
