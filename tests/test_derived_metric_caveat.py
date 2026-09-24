"""health_advisor#434: the catalogue's caveat on body_fat_percentage /
lean_body_mass (both pure functions of body_mass on this device) must be
machine-readable, must stop correlate.py from presenting a pair coupled
through it as an independent finding, and must travel to every MCP surface
that publishes one of these metrics' values, series or correlations.

This module also carries the mutation check from the issue: with
`derived_from` removed from the catalogue, or the accessor stubbed to always
return None, the tests below that depend on it must go red.
"""
from __future__ import annotations

from health_advisor import correlate as C
from health_advisor import normalize as nz
from tests.conftest import seed_metric

BODY_MASS_CAVEAT_METRICS = ("body_fat_percentage", "lean_body_mass")


# --------------------------------------------------------------------------- #
# 1. catalogue accessors
# --------------------------------------------------------------------------- #
def test_derived_from_accessor():
    assert nz.derived_from("body_fat_percentage") == "body_mass"
    assert nz.derived_from("lean_body_mass") == "body_mass"
    assert nz.derived_from("body_mass") is None
    assert nz.derived_from("step_count") is None


def test_caveat_accessor():
    for m in BODY_MASS_CAVEAT_METRICS:
        c = nz.caveat(m)
        assert c is not None and len(c) > 0
        assert c == nz.CATALOG[m]["caveat"]
    assert nz.caveat("body_mass") is None
    assert nz.caveat("step_count") is None


# --------------------------------------------------------------------------- #
# 2. correlate.py honours it
# --------------------------------------------------------------------------- #
def test_derived_pair_both_directions_and_shared_source():
    assert C.derived_pair("body_fat_percentage", "body_mass")
    assert C.derived_pair("body_mass", "body_fat_percentage")
    assert C.derived_pair("lean_body_mass", "body_mass")
    assert C.derived_pair("body_mass", "lean_body_mass")
    # both derived from the same source, neither derived from the other
    assert C.derived_pair("body_fat_percentage", "lean_body_mass")
    assert C.derived_pair("lean_body_mass", "body_fat_percentage")
    # unrelated pair
    assert not C.derived_pair("step_count", "body_fat_percentage")
    assert not C.derived_pair("body_mass", "step_count")


def test_related_pair_wiring_is_driven_by_derived_from_not_group(monkeypatch):
    """Isolates correlate.py's OR-wiring from the group-equality check: these
    two synthetic metrics are in different catalog groups, so only the
    derived_from path can flag them as related."""
    monkeypatch.setitem(nz.CATALOG, "fake_source",
                        {"unit": "x", "agg": "last", "group": "alpha"})
    monkeypatch.setitem(nz.CATALOG, "fake_derived",
                        {"unit": "x", "agg": "last", "group": "beta",
                         "derived_from": "fake_source"})
    assert C.derived_pair("fake_source", "fake_derived")
    assert C._related_pair("fake_source", "fake_derived")
    assert C._related_pair("fake_derived", "fake_source")


def _seed_body_trio(conn, n=30):
    # Non-constant, non-degenerate series so scipy's correlate() does not
    # decline on a zero-variance guard.
    bm = [150.0 + 0.3 * i for i in range(n)]
    bf = [25.0 - 0.01 * i for i in range(n)]
    lbm = [b * (1 - f / 100.0) for b, f in zip(bm, bf)]
    seed_metric(conn, "body_mass", "2026-01-01", bm)
    seed_metric(conn, "body_fat_percentage", "2026-01-01", bf)
    seed_metric(conn, "lean_body_mass", "2026-01-01", lbm)


def test_scan_flags_derived_pair_as_related_and_carries_caveat(conn):
    _seed_body_trio(conn)
    tests = C.scan(conn, "body_fat_percentage", "2026-01-01", "2026-01-30", lags=(0,))
    rows = {t["metric"]: t for t in tests if t["lag_days"] == 0}
    assert rows["body_mass"]["related_group"] is True
    assert any(nz.caveat("body_fat_percentage") in c for c in rows["body_mass"]["caveats"])
    assert rows["lean_body_mass"]["related_group"] is True
    assert any(nz.caveat("body_fat_percentage") in c or nz.caveat("lean_body_mass") in c
              for c in rows["lean_body_mass"]["caveats"])


def test_hypotheses_flags_derived_pair_and_carries_caveat(conn):
    _seed_body_trio(conn)
    rows = C.test_hypotheses(conn, [
        {"metric_x": "body_mass", "metric_y": "body_fat_percentage",
         "lag_days": 0, "window": "all"},
        {"metric_x": "body_fat_percentage", "metric_y": "lean_body_mass",
         "lag_days": 0, "window": "all"},
    ])
    assert rows[0]["related_group"] is True
    assert rows[0]["caveats"]
    assert rows[1]["related_group"] is True
    assert rows[1]["caveats"]


# --------------------------------------------------------------------------- #
# 3. MCP publish surfaces
# --------------------------------------------------------------------------- #
def test_correlate_metrics_tool_carries_caveat_and_not_independent_flag(conn, tools):
    _seed_body_trio(conn)
    out = tools.correlate_metrics("body_mass", "body_fat_percentage", 0, "all")
    assert out["status"] == "ok"
    joined = " ".join(out["caveats"])
    assert nz.caveat("body_fat_percentage") in joined
    assert "not an independent finding" in joined


def test_correlate_metrics_tool_carries_caveat_even_against_unrelated_metric(conn, tools):
    _seed_body_trio(conn)
    seed_metric(conn, "step_count", "2026-01-01", [1000.0 + 5 * i for i in range(30)])
    out = tools.correlate_metrics("step_count", "body_fat_percentage", 0, "all")
    assert out["status"] == "ok"
    joined = " ".join(out["caveats"])
    assert nz.caveat("body_fat_percentage") in joined


def test_scan_correlations_tool_flags_related_group_for_derived_pair(conn, tools):
    _seed_body_trio(conn)
    out = tools.scan_correlations("body_fat_percentage", "all", "0", 50)
    rows = {r["metric"]: r for r in out["results"] if r["lag_days"] == 0}
    assert rows["body_mass"]["related_group"] is True
    assert rows["lean_body_mass"]["related_group"] is True


def test_get_daily_series_carries_caveat(conn, tools):
    seed_metric(conn, "body_fat_percentage", "2026-01-01", [25.0 - 0.1 * i for i in range(10)])
    out = tools.get_daily_series("body_fat_percentage", "2026-01-01", "2026-01-10")
    assert out["caveat"] == nz.caveat("body_fat_percentage")


def test_summarize_metric_carries_caveat(conn, tools):
    seed_metric(conn, "lean_body_mass", "2026-01-01", [110.0 + 0.2 * i for i in range(30)])
    out = tools.summarize_metric("lean_body_mass", "30d")
    assert out["caveat"] == nz.caveat("lean_body_mass")


def test_compare_periods_carries_caveat(conn, tools):
    seed_metric(conn, "body_fat_percentage", "2026-01-01", [25.0 - 0.1 * i for i in range(60)])
    out = tools.compare_periods("body_fat_percentage", "2026-01-01:2026-01-30",
                                "2026-01-31:2026-03-01")
    assert out["caveat"] == nz.caveat("body_fat_percentage")


def test_get_latest_carries_caveat(conn, tools):
    seed_metric(conn, "body_fat_percentage", "2026-01-01", [25.0, 24.9, 24.8])
    out = tools.get_latest("body_fat_percentage")
    assert out["caveat"] == nz.caveat("body_fat_percentage")


def test_get_weekly_series_carries_caveat(conn, tools):
    seed_metric(conn, "lean_body_mass", "2026-01-01", [110.0 + 0.1 * i for i in range(60)])
    out = tools.get_weekly_series("lean_body_mass", "2026-01-01", "2026-02-28")
    assert out["caveat"] == nz.caveat("lean_body_mass")


def test_get_block_comparison_carries_caveat(conn, tools):
    seed_metric(conn, "body_fat_percentage", "2026-01-01", [25.0 - 0.02 * i for i in range(60)])
    out = tools.get_block_comparison("body_fat_percentage", 4, "2026-03-01")
    assert out.get("caveat") == nz.caveat("body_fat_percentage")


def test_metrics_without_a_catalogue_caveat_publish_none(conn, tools):
    seed_metric(conn, "step_count", "2026-01-01", [1000.0 + i for i in range(10)])
    out = tools.get_daily_series("step_count", "2026-01-01", "2026-01-10")
    assert "caveat" not in out
