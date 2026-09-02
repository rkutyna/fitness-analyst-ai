"""MCP food_lookup / food_catalog_add / food_meal_total on a temp DB."""
import pytest


YOGURT = dict(item_key="tj-strained-greek-yogurt-plain",
              display_name="Strained Thick & Creamy Greek Yogurt, Plain",
              serving_desc="3/4 cup", kcal=170.0, protein_g=15.0, carb_g=7.0,
              fat_g=9.0, source="web", source_detail="https://example.test/y",
              brand="Trader Joe's", aliases="greek yogurt", confirmed=1)


def _db_without_the_catalog(path):
    """The production DB as it is before this feature's first write: the file
    exists and is populated, but food_catalog is not in it. Read tools open
    read-only and never run DDL (see mcp_server.py get_subjective), so they
    cannot create it on the way past."""
    from health_advisor import db as dbmod
    conn = dbmod.connect(path)
    dbmod.init_db(conn)
    conn.execute("DROP TABLE food_catalog")
    conn.commit()
    conn.close()


def test_lookup_before_the_table_exists_returns_empty_not_a_crash(tools, vault_path):
    _db_without_the_catalog(vault_path)
    out = tools.food_lookup("anything")
    assert out["count"] == 0
    assert out["items"] == []
    assert "note" in out


def test_lookup_does_not_swallow_a_misconfigured_db_path(tools):
    """A missing vault file must be loud — an empty catalog is the wrong
    answer to 'your database is not there'."""
    import sqlite3
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        tools.food_lookup("anything")


def test_add_then_lookup_round_trip(tools):
    out = tools.food_catalog_add(**YOGURT)
    assert out["ok"] is True
    assert out["stored"]["confirmed"] == 1

    got = tools.food_lookup("greek yogurt")
    assert got["count"] == 1
    assert got["items"][0]["kcal"] == 170.0
    assert got["items"][0]["source"] == "web"
    assert got["items"][0]["verified_at"]


def test_empty_query_lists_the_whole_catalog(tools):
    tools.food_catalog_add(**YOGURT)
    assert tools.food_lookup("")["count"] == 1


def test_add_validation_error_is_structured_not_raised(tools):
    out = tools.food_catalog_add(**{**YOGURT, "source": "vibes"})
    assert out["ok"] is False
    assert "source" in out["error"]


def test_add_requires_source_detail(tools):
    out = tools.food_catalog_add(**{**YOGURT, "source_detail": "  "})
    assert out["ok"] is False
    assert "source_detail" in out["error"]


def test_meal_total_sums_and_reports_the_weakest_tier(tools):
    tools.food_catalog_add(**YOGURT)
    tools.food_catalog_add(item_key="tj-adobo-pork-shoulder",
                         display_name="TJ Adobo Pork Shoulder",
                         serving_desc="4 oz raw", kcal=210.0,
                         source="estimate",
                         source_detail="proxied off Al Pastor diced pork")
    out = tools.food_meal_total([
        {"item_key": "tj-strained-greek-yogurt-plain", "servings": 1},
        {"item_key": "tj-adobo-pork-shoulder", "servings": 2},
    ])
    assert out["ok"] is True
    assert out["kcal"] == pytest.approx(590.0)
    assert out["weakest_source"] == "estimate"
    assert out["protein_g"] is None          # pork has no protein figure
    assert "protein_g" in out["incomplete"]


def test_meal_total_unknown_item_is_structured(tools):
    tools.food_catalog_add(**YOGURT)
    out = tools.food_meal_total([{"item_key": "nope", "servings": 1}])
    assert out["ok"] is False
    assert "nope" in out["error"]


def test_meal_total_rejects_malformed_items(tools):
    tools.food_catalog_add(**YOGURT)
    assert tools.food_meal_total([{"servings": 1}])["ok"] is False
    assert tools.food_meal_total(
        [{"item_key": "tj-strained-greek-yogurt-plain",
          "servings": "two"}])["ok"] is False
    assert tools.food_meal_total([])["ok"] is False


def test_meal_total_before_the_table_exists_is_structured(tools, vault_path):
    _db_without_the_catalog(vault_path)
    out = tools.food_meal_total([{"item_key": "x", "servings": 1}])
    assert out["ok"] is False
    assert out["error"]
