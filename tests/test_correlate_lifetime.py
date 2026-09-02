"""Regression test for lifetime correlation windows."""
from __future__ import annotations

from health_advisor import correlate as C
from health_advisor import metrics as M

from tests.conftest import seed_metric


def test_scan_all_period_resolves_earliest_metric_date(conn):
    values = [float(i) for i in range(12)]
    seed_metric(conn, "step_count", "2026-06-01", values)
    seed_metric(conn, "active_energy", "2026-06-01", values)

    start_iso, end_iso = M.parse_period("all", "2026-06-12")
    results = C.scan(conn, "active_energy", start_iso, end_iso, lags=(0,))

    step = next(row for row in results if row["metric"] == "step_count")
    assert step["n_pairs"] == 12
    assert step["pearson_r"] == 1.0
