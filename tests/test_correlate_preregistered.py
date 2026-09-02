"""Regression tests for lifetime and pre-registered correlation statistics."""
from __future__ import annotations

import pytest

from health_advisor import correlate as C
from tests.conftest import seed_metric


def test_test_hypotheses_returns_effect_ci_and_fdr_for_declared_set(conn):
    values = [float(i) for i in range(12)]
    seed_metric(conn, "cardio_recovery", "2026-06-01", values)
    seed_metric(conn, "body_fat_percentage", "2026-06-01", values)

    results = C.test_hypotheses(conn, [
        {"metric_x": "cardio_recovery", "metric_y": "body_fat_percentage",
         "lag_days": 0, "window": "2026-06-01:2026-06-12"},
        {"metric_x": "cardio_recovery", "metric_y": "body_fat_percentage",
         "lag_days": 0, "window": "2026-06-01:2026-06-05"},
    ])

    assert len(results) == 2
    tested = results[0]
    assert tested["status"] == "ok"
    assert tested["rho"] == 1.0
    assert tested["p"] < 0.001
    assert tested["q"] < 0.001
    assert tested["passed_fdr"] is True
    assert tested["n_pairs"] == 12
    assert tested["pearson_ci95"][0] <= tested["pearson_r"] <= tested["pearson_ci95"][1]
    assert tested["dropped_low_wear"] == 0
    assert tested["related_group"] is False

    untestable = results[1]
    assert untestable["status"] == "insufficient_data"
    assert untestable["n_pairs"] == 5
    assert untestable["q"] is None
    assert untestable["passed_fdr"] is False
    assert untestable["dropped_low_wear"] == 0


def test_test_hypotheses_uses_period_window_and_flags_tautology(conn):
    n = 12
    seed_metric(conn, "sleep_asleep", "2026-06-01", [400.0 + i for i in range(n)])
    seed_metric(conn, "sleep_deep", "2026-06-01", [80.0 + i for i in range(n)])

    results = C.test_hypotheses(conn, [{
        "x": "sleep_deep", "y": "sleep_asleep", "lag_days": 0,
        "window": "all",
    }])

    assert results[0]["status"] == "ok"
    assert results[0]["n_pairs"] == n
    assert results[0]["related_group"] is True
