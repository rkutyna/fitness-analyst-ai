"""Tools must refuse input they didn't understand (audit P1-4, P1-5).

The product's premise is that every number the coach says came from a tool, so
a tool that answers confidently about a window it invented is the worst failure
mode available: the agent cannot detect it and narrates the result as fact.
Asked to compare June against July, the old parser silently compared the last
30 days against themselves and reported "0% change".
"""
from __future__ import annotations

import pytest

from health_advisor import metrics as M
from tests.conftest import seed_metric


@pytest.fixture
def db(conn):
    """Seed 60 days of steps in this test's vault."""
    seed_metric(conn, "step_count", "2026-05-01", [6000 + i for i in range(60)])


# --------------------------------------------------------------------------- #
# P1-4: period specs
# --------------------------------------------------------------------------- #
def test_parse_period_rejects_a_spec_it_cannot_parse():
    for spec in ("june", "last month", "30", "d30", "12x"):
        with pytest.raises(ValueError):
            M.parse_period(spec, "2026-07-29")


def test_parse_period_still_accepts_the_documented_specs():
    assert M.parse_period("30d", "2026-07-29")[0] == "2026-06-30"
    assert M.parse_period("all", "2026-07-29") == (None, "2026-07-29")


def test_parse_range_rejects_a_reversed_explicit_range():
    with pytest.raises(ValueError):
        M.parse_range("2026-07-29:2026-06-01", "2026-07-29")


def test_parse_range_rejects_a_non_date_in_an_explicit_range():
    with pytest.raises(ValueError):
        M.parse_range("june:july", "2026-07-29")


def test_summarize_metric_reports_a_bad_period(db, tools):
    out = tools.summarize_metric("step_count", "june")
    assert "error" in out
    assert "june" in out["error"]
    assert "n_days" not in out


def test_compare_periods_reports_a_bad_period(db, tools):
    out = tools.compare_periods("step_count", "june", "july")
    assert "error" in out
    assert "mean_delta" not in out


def test_compare_periods_will_not_call_an_empty_window_a_change(db, tools):
    # period_b has no data: the old code reported period_a's whole mean as the
    # delta, which reads as an enormous real change.
    out = tools.compare_periods("step_count", "30d", "2020-01-01:2020-02-01")
    assert out["mean_delta"] is None
    assert out["mean_delta_pct"] is None
    assert "note" in out


def test_compare_periods_still_compares_two_real_windows(db, tools):
    out = tools.compare_periods("step_count", "2026-06-01:2026-06-15",
                            "2026-05-01:2026-05-15")
    assert out["mean_delta"] is not None
    assert out["period_a"]["n_days"] == 15


def test_correlate_metrics_reports_a_bad_period(db, tools):
    out = tools.correlate_metrics("step_count", "step_count", period="june")
    assert "error" in out


# --------------------------------------------------------------------------- #
# P1-5: date parameters
# --------------------------------------------------------------------------- #
def test_get_intraday_rejects_a_relative_day(db, tools):
    out = tools.get_intraday("step_count", "yesterday")
    assert "error" in out
    assert "buckets" not in out


def test_get_intraday_rejects_an_impossible_date(db, tools):
    assert "error" in tools.get_intraday("step_count", "2026-07-32")


def test_get_daily_series_rejects_a_reversed_range(db, tools):
    out = tools.get_daily_series("step_count", start="2026-06-30", end="2026-06-01")
    assert "error" in out
    assert "points" not in out


def test_get_daily_series_rejects_a_relative_date(db, tools):
    assert "error" in tools.get_daily_series("step_count", start="last week")


def test_list_workouts_rejects_relative_dates(db, tools):
    out = tools.list_workouts("last week", "yesterday")
    assert "error" in out
    assert "workouts" not in out


def test_list_workouts_rejects_a_reversed_range(db, tools):
    assert "error" in tools.list_workouts("2026-07-01", "2026-06-01")


def test_get_subjective_rejects_a_reversed_range(db, tools):
    out = tools.get_subjective("2026-07-29", "2026-07-01")
    assert out["ok"] is False


def test_get_impact_volume_rejects_a_reversed_range(db, tools):
    assert "error" in tools.get_impact_volume("2026-07-29", "2026-07-01")


def test_valid_dates_still_work(db, tools):
    out = tools.get_daily_series("step_count", start="2026-05-01", end="2026-05-10")
    assert out["n"] == 10
    assert "error" not in out
