"""Tests for the gated fixed-length metric block comparison."""
from __future__ import annotations

from datetime import date, timedelta

from health_advisor import analysis as A
from tests.conftest import seed_metric


def _seed_last_series(conn, metric: str, as_of: date, values: list[float]) -> None:
    start = as_of - timedelta(days=len(values) - 1)
    for offset, value in enumerate(values):
        day = (start + timedelta(days=offset)).isoformat()
        conn.execute(
            "INSERT INTO daily_metrics "
            "(metric, date, count, sum, avg, min, max, last, unit) "
            "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)",
            (metric, day, value + 100.0, value + 100.0, value, value,
             value, "count/min"),
        )
    conn.commit()


def test_block_comparison_uses_last_and_gates_the_mdc(tools, conn, monkeypatch):
    as_of = date.today()
    values = [100.0] * 28 + [99.33] * 28
    _seed_last_series(conn, "resting_heart_rate", as_of, values)
    monkeypatch.setattr(
        A, "metric_noise_floor",
        lambda conn, metric, as_of: {
            "sd_day": 3.86, "rho": 0.16,
        },
    )

    result = tools.get_block_comparison(
        "resting_heart_rate", block_weeks=4, as_of=as_of.isoformat())

    assert result["blocks"]["previous"]["mean"] == 100.0
    assert result["blocks"]["recent"]["mean"] == 99.33
    assert result["diff"] == -0.67
    assert result["mdc95"] == 2.38
    assert result["exceeds_mdc95"] is False


def test_block_comparison_refuses_a_block_below_coverage(tools, conn):
    as_of = date.today()
    seed_metric(conn, "resting_heart_rate", (as_of - timedelta(days=27)).isoformat(),
                [60.0] * 28)
    conn.execute(
        "DELETE FROM daily_metrics WHERE metric = ? AND date > ?",
        ("resting_heart_rate", (as_of - timedelta(days=8)).isoformat()),
    )
    conn.commit()

    result = tools.get_block_comparison(
        "resting_heart_rate", block_weeks=4, as_of=as_of.isoformat())

    assert result["status"] == "insufficient_coverage"
    assert result["blocks"]["recent"]["n"] == 20
    assert "mean" not in result["blocks"]["recent"]
    assert result["diff"] is None
    assert result["exceeds_mdc95"] is None


def test_block_comparison_mdc_gate_is_a_real_verdict(tools, conn, monkeypatch):
    as_of = date.today()
    values = [100.0] * 28 + [99.33] * 28
    _seed_last_series(conn, "resting_heart_rate", as_of, values)
    monkeypatch.setattr(
        A, "metric_noise_floor",
        lambda conn, metric, as_of: {"sd_day": 3.86, "rho": 0.16},
    )

    result = tools.get_block_comparison(
        "resting_heart_rate", block_weeks=4, as_of=as_of.isoformat())

    assert abs(result["diff"]) < result["mdc95"]
    assert result["exceeds_mdc95"] is False


def test_block_comparison_verdict_moves_with_the_floor(tools, conn, monkeypatch):
    """The MDC gate is a computation, not a constant: the same 0.67 bpm move
    exceeds a floor of 0 and does not exceed a floor of 1000.

    Committed form of the review's mutation check: a comparison whose verdict
    ignored mdc95 would pass the exact-number test above and fail here."""
    as_of = date.today()
    _seed_last_series(conn, "resting_heart_rate", as_of, [100.0] * 28 + [99.33] * 28)
    monkeypatch.setattr(
        A, "metric_noise_floor",
        lambda conn, metric, as_of: {"sd_day": 3.86, "rho": 0.16},
    )
    monkeypatch.setattr(A, "mdc95", lambda sd_day, rho, days: 0.0)
    assert tools.get_block_comparison(
        "resting_heart_rate", block_weeks=4, as_of=as_of.isoformat(),
    )["exceeds_mdc95"] is True
    monkeypatch.setattr(A, "mdc95", lambda sd_day, rho, days: 1000.0)
    assert tools.get_block_comparison(
        "resting_heart_rate", block_weeks=4, as_of=as_of.isoformat(),
    )["exceeds_mdc95"] is False
