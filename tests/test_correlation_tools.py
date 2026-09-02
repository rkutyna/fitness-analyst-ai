"""MCP-level tests for correlate_metrics / scan_correlations (test_get_briefing_tool style)."""
from __future__ import annotations

import re

from tests.conftest import seed_metric, seed_workout


def test_correlate_metrics_lagged(conn, tools):
    seed_metric(conn, "step_count", "2026-06-01", [float(i) for i in range(30)])
    seed_metric(conn, "active_energy", "2026-06-02", [float(i * 10 + 3) for i in range(30)])
    out = tools.correlate_metrics("step_count", "active_energy", lag_days=1, period="30d")
    assert out["status"] == "ok"
    assert out["pearson_r"] == 1.0
    assert out["n_pairs"] == 30
    assert "day D-1" in out["lag_semantics"]
    assert any("causation" in c for c in out["caveats"])


def test_correlate_metrics_unknown_metric(conn, tools):
    seed_metric(conn, "step_count", "2026-06-01", [1.0] * 10)
    out = tools.correlate_metrics("nope", "step_count")
    assert "error" in out


def test_correlate_metrics_insufficient(conn, tools):
    seed_metric(conn, "step_count", "2026-06-01", [1.0, 2.0, 3.0])
    seed_metric(conn, "active_energy", "2026-06-01", [5.0, 6.0, 7.0])
    out = tools.correlate_metrics("step_count", "active_energy", period="30d")
    assert out["status"] == "insufficient_data"
    assert "pearson_r" not in out


def test_scan_correlations_tool(conn, tools):
    n = 40
    sleep = [400.0 + (i % 7) * 10 for i in range(n)]
    seed_metric(conn, "sleep_asleep", "2026-06-01", sleep)
    seed_metric(conn, "time_in_daylight", "2026-06-01", [s / 4 for s in sleep])
    seed_metric(conn, "headphone_audio_exposure", "2026-06-01",
                [70.0 + (i % 3) for i in range(n)])
    out = tools.scan_correlations("sleep_asleep", period="60d", lags="0,1", max_results=5)
    assert out["tested_count"] >= 2
    assert out["results"][0]["metric"] == "time_in_daylight"
    assert out["passed_fdr_count"] >= 1
    assert "q_value" in out["results"][0]
    assert len(out["results"]) <= 5


def test_scan_correlations_malformed_lags(conn, tools):
    seed_metric(conn, "step_count", "2026-06-01", [float(i) for i in range(10)])
    out = tools.scan_correlations("step_count", lags="0,x")
    assert "error" in out and "lags" in out["error"]


def test_list_workouts_has_local_times(conn, tools):
    seed_workout(conn, "running", "2026-07-10", 45.0, 5.0, avg_heart_rate=140.0)
    out = tools.list_workouts(start="2026-07-01", end="2026-07-15")
    w = out["workouts"][0]
    assert re.fullmatch(r"\d{2}:\d{2}", w["start_time_local"])
    assert re.fullmatch(r"\d{2}:\d{2}", w["end_time_local"])
