"""Stage 4: declared metric presentation converts one vault at a time."""
from __future__ import annotations

import pytest

from health_advisor import benchmark, db, mcp_server as S, vault as V
from health_advisor.context import VaultContext


DAY = "2026-08-01"


def _vault(path, system=None):
    ctx = VaultContext.local(path, user_id="test", writable=True)
    conn = ctx.connect()
    db.init_db(conn)
    rows = [
        ("distance_walking_running", 2.0, "mi", "sum"),
        ("body_mass", 180.0, "lb", "last"),
        ("sleeping_wrist_temperature", 68.0, "degF", "avg"),
        ("walking_speed", 3.0, "mi/hr", "avg"),
        ("active_energy", 100.0, "kcal", "sum"),
        ("step_count", 1000.0, "count", "sum"),
        ("heart_rate", 140.0, "count/min", "mean"),
    ]
    for metric, value, unit, _ in rows:
        conn.execute(
            "INSERT INTO daily_metrics "
            "(metric, date, count, sum, avg, min, max, last, unit) "
            "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)",
            (metric, DAY, value, value, value, value, value, unit),
        )
    for metric, value, unit in (
        ("distance_walking_running", 2.0, "mi"),
        ("heart_rate", 140.0, "count/min"),
    ):
        ts = f"{DAY}T12:00:00+00:00"
        conn.execute(
            "INSERT INTO records "
            "(metric, value, unit, start_utc, end_utc, start_local, local_date, "
            "source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, ?, 'test', ?)",
            (metric, value, unit, ts, ts, ts[:-6], DAY, f"{metric}-1"),
        )
    if system is not None:
        V.set_unit_system(conn, system)
    conn.commit()
    conn.close()
    return ctx


def test_metric_and_imperial_vaults_relabel_the_same_three_quantities(tmp_path):
    imperial = S.build_tools(_vault(tmp_path / "imperial.db", "imperial"))
    metric = S.build_tools(_vault(tmp_path / "metric.db", "metric"))

    def readings(tools):
        return {
            name: tools["get_latest"](name)["latest_day"]
            for name in ("distance_walking_running", "body_mass",
                         "sleeping_wrist_temperature")
        }

    old, new = readings(imperial), readings(metric)
    assert [(old[m]["unit"], old[m]["value"], new[m]["unit"], new[m]["value"])
            for m in old] == [
        ("mi", 2.0, "km", 3.22),
        ("lb", 180.0, "kg", 81.65),
        ("degF", 68.0, "degC", 20.0),
    ]
    assert new["distance_walking_running"]["value"] / V.UNIT_CONVERSION_FACTORS[
        "distance_mi_to_km"] == pytest.approx(
            old["distance_walking_running"]["value"], abs=0.01)


def test_undeclared_vault_matches_explicit_imperial_for_all_generic_paths(tmp_path):
    undeclared = S.build_tools(_vault(tmp_path / "undeclared.db"))
    imperial = S.build_tools(_vault(tmp_path / "imperial.db", "imperial"))
    calls = [
        ("list_available_metrics", ()),
        ("get_daily_series", ("distance_walking_running", DAY, DAY)),
        ("summarize_metric", ("distance_walking_running", "all")),
        ("compare_periods", ("distance_walking_running", f"{DAY}:{DAY}",
                              f"{DAY}:{DAY}")),
        ("get_intraday", ("distance_walking_running", DAY)),
        ("get_latest", ("distance_walking_running",)),
    ]
    for name, args in calls:
        assert undeclared[name](*args) == imperial[name](*args)

    # The cross-comparison above is necessary and NOT sufficient, measured
    # 2026-09-05 by mutating the guard to convert regardless of the declared
    # system -- the exact defect this test is named for. BOTH payloads then
    # convert, still match each other, and this test stayed GREEN while an
    # undeclared vault silently published km. Its oracle moved with the bug.
    #
    # So anchor to the stored units and values, which no such mutation moves.
    for metric, unit, value in (("distance_walking_running", "mi", 2.0),
                                ("body_mass", "lb", 180.0),
                                ("sleeping_wrist_temperature", "degF", 68.0)):
        latest = undeclared["get_latest"](metric)["latest_day"]
        assert (latest["unit"], latest["value"]) == (unit, pytest.approx(value)), (
            f"an undeclared vault must publish stored {unit}, got {latest}")


def test_metric_gate_keeps_counts_rates_and_minutes_unchanged(tmp_path):
    imperial = S.build_tools(_vault(tmp_path / "imperial.db", "imperial"))
    metric = S.build_tools(_vault(tmp_path / "metric.db", "metric"))

    old = imperial["get_daily_series"]("step_count", DAY, DAY)
    new = metric["get_daily_series"]("step_count", DAY, DAY)
    assert new == old

    old = imperial["get_daily_series"]("heart_rate", DAY, DAY)
    new = metric["get_daily_series"]("heart_rate", DAY, DAY)
    assert new == old


def test_benchmark_series_converts_stored_pace_only_at_metric_boundary(conn):
    benchmark.record(conn, date=DAY, stage=1, pace="10:00",
                     median_hr_last_two_min=140)
    old = benchmark.series(conn)
    new = benchmark.series(conn, metric_units=True)
    assert old[0]["pace_min_per_mi"] == 10.0
    assert new[0]["pace_min_per_km"] == pytest.approx(
        10.0 * V.UNIT_CONVERSION_FACTORS[
            "pace_min_per_mi_to_min_per_km"])
    assert "pace_min_per_mi" not in new[0]


def test_tool_conversion_does_not_change_stored_values(tmp_path):
    ctx = _vault(tmp_path / "metric.db", "metric")
    tools = S.build_tools(ctx)
    conn = ctx.read_only()
    try:
        before = conn.execute(
            "SELECT metric, date, count, sum, avg, min, max, last, unit "
            "FROM daily_metrics ORDER BY metric, date"
        ).fetchall()
    finally:
        conn.close()
    tools["get_daily_series"]("body_mass", DAY, DAY)
    tools["get_latest"]("distance_walking_running")
    conn = ctx.read_only()
    try:
        after = conn.execute(
            "SELECT metric, date, count, sum, avg, min, max, last, unit "
            "FROM daily_metrics ORDER BY metric, date"
        ).fetchall()
        stored = conn.execute(
            "SELECT metric, value FROM records WHERE metric IN "
            "('distance_walking_running', 'heart_rate') ORDER BY metric"
        ).fetchall()
        daily = conn.execute(
            "SELECT metric, avg FROM daily_metrics WHERE metric IN "
            "('body_mass', 'distance_walking_running') ORDER BY metric"
        ).fetchall()
    finally:
        conn.close()
    assert before == after
    assert [(r["metric"], r["value"]) for r in stored] == [
        ("distance_walking_running", 2.0), ("heart_rate", 140.0)]
    assert [(r["metric"], r["avg"]) for r in daily] == [
        ("body_mass", 180.0), ("distance_walking_running", 2.0)]
