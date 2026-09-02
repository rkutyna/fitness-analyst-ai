from datetime import date
import numpy as np
from health_advisor import metrics as M
from tests.conftest import seed_metric


def test_parse_period_30d():
    start, end = M.parse_period("30d", "2026-06-10")
    assert end == "2026-06-10"
    assert start == "2026-05-12"  # 30 inclusive days


def test_slope_per_week_positive():
    dates = ["2026-06-01", "2026-06-08", "2026-06-15"]
    vals = [10.0, 17.0, 24.0]  # +7/week
    assert round(M.slope_per_week(dates, vals), 1) == 7.0


def test_baseline_excludes_recent_and_uses_median():
    vals = [50, 50, 50, 50, 999]  # last is the "recent" excluded point
    assert M.baseline(vals, exclude_recent=1, window=28) == 50.0


def test_pct_change():
    assert M.pct_change(110, 100) == 10.0
    assert M.pct_change(10, 0) is None


def test_series_reads_right_column(conn):
    seed_metric(conn, "step_count", "2026-06-01", [100, 200, 300])
    dates, vals, unit = M.series(conn, "step_count", "2026-06-01", "2026-06-03")
    assert vals == [100.0, 200.0, 300.0]
    assert unit == "count"
