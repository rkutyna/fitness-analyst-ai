"""get_weekly_series gives a metric-unit vault the same unit treatment as
get_daily_series: converted values, the metric unit label, nothing converted
twice, and an imperial or undeclared vault unchanged."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from health_advisor import db, mcp_server as S, vault as V
from health_advisor.context import VaultContext

START = date(2026, 6, 1)          # a Monday
DAYS = 42                         # six full Monday-Sunday weeks
SERIES = {
    "body_mass": ("lb", "last"),
    "sleeping_wrist_temperature": ("degF", "avg"),
    "distance_walking_running": ("mi", "sum"),
}


def _value(metric_index: int, i: int) -> float:
    return 150.0 + 20 * metric_index + (i % 7) * 1.5 + (i // 7) * 0.4 + (i % 3)


def _vault(path, system):
    ctx = VaultContext.local(path, user_id="test", writable=True)
    conn = ctx.connect()
    db.init_db(conn)
    for k, (metric, (unit, _)) in enumerate(SERIES.items()):
        for i in range(DAYS):
            day = (START + timedelta(days=i)).isoformat()
            v = _value(k, i)
            conn.execute(
                "INSERT INTO daily_metrics "
                "(metric, date, count, sum, avg, min, max, last, unit) "
                "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)",
                (metric, day, v, v, v, v, v, unit))
    if system is not None:
        V.set_unit_system(conn, system)
    conn.commit()
    conn.close()
    return S.build_tools(ctx)


def _weekly(tools, metric):
    end = (START + timedelta(days=DAYS - 1)).isoformat()
    return tools["get_weekly_series"](metric, START.isoformat(), end)


@pytest.mark.parametrize("metric", sorted(SERIES))
def test_metric_vault_weekly_mean_is_the_converted_storage_mean(tmp_path, metric):
    imperial = _weekly(_vault(tmp_path / "i.db", "imperial"), metric)
    metric_out = _weekly(_vault(tmp_path / "m.db", "metric"), metric)
    unit = SERIES[metric][0]
    twin = V._METRIC_UNIT_TWINS[unit][0]
    assert len(metric_out["weeks"]) == len(imperial["weeks"]) == 6
    for old, new in zip(imperial["weeks"], metric_out["weeks"]):
        expected, expected_unit = V.convert_for_unit_system(
            old["mean"], unit, "metric")
        assert new["unit"] == expected_unit == twin
        assert new["mean"] == pytest.approx(expected, abs=0.006)
        assert new["mean"] != old["mean"]
        # Same week, same days: only the presentation changed.
        assert (new["period"], new["n_days"]) == (old["period"], old["n_days"])
        # The presentation leaf renders the converted value, as the daily path
        # does, and nothing is converted twice.
        assert new["presentation"]["value"] != old["presentation"]["value"]


def test_weekly_mean_agrees_with_the_daily_series_for_the_same_week(tmp_path):
    tools = _vault(tmp_path / "m.db", "metric")
    daily = tools["get_daily_series"](
        "body_mass", START.isoformat(), (START + timedelta(days=6)).isoformat())
    week = _weekly(tools, "body_mass")["weeks"][0]
    mean_of_daily = sum(p["value"] for p in daily["points"]) / 7
    assert daily["unit"] == week["unit"] == "kg"
    assert week["mean"] == pytest.approx(mean_of_daily, abs=0.02)


def test_sum_total_and_noise_floor_are_in_the_declared_unit_too(tmp_path):
    imperial = _weekly(_vault(tmp_path / "i.db", "imperial"),
                       "distance_walking_running")
    metric_out = _weekly(_vault(tmp_path / "m.db", "metric"),
                         "distance_walking_running")
    factor = V.UNIT_CONVERSION_FACTORS["distance_mi_to_km"]
    for old, new in zip(imperial["weeks"], metric_out["weeks"]):
        assert new["total"] == pytest.approx(old["total"] * factor, abs=0.02)
        assert new["mdc95"] == pytest.approx(old["mdc95"] * factor, abs=0.02)
        assert new["sd_day"] == pytest.approx(old["sd_day"] * factor, abs=0.02)
    assert metric_out["noise_floor"]["sd_day"] == pytest.approx(
        imperial["noise_floor"]["sd_day"] * factor, abs=0.02)
    assert metric_out["total"] == pytest.approx(imperial["total"] * factor,
                                                abs=0.05)


def test_a_temperature_spread_is_scaled_not_offset(tmp_path):
    imperial = _weekly(_vault(tmp_path / "i.db", "imperial"),
                       "sleeping_wrist_temperature")
    metric_out = _weekly(_vault(tmp_path / "m.db", "metric"),
                         "sleeping_wrist_temperature")
    scale = V.UNIT_CONVERSION_FACTORS["temperature_degF_to_degC"]
    for old, new in zip(imperial["weeks"], metric_out["weeks"]):
        assert new["mdc95"] == pytest.approx(old["mdc95"] * scale, abs=0.02)


@pytest.mark.parametrize("metric", sorted(SERIES))
def test_undeclared_matches_imperial_and_keeps_stored_units(tmp_path, metric):
    undeclared = _weekly(_vault(tmp_path / "u.db", None), metric)
    imperial = _weekly(_vault(tmp_path / "i.db", "imperial"), metric)
    assert undeclared == imperial
    # Anchored to the stored value, which a convert-always mutation cannot move.
    assert undeclared["weeks"][0]["unit"] == SERIES[metric][0]
    stored = [_value(list(SERIES).index(metric), i) for i in range(7)]
    assert undeclared["weeks"][0]["mean"] == pytest.approx(
        sum(stored) / 7, abs=0.006)


def test_unitless_metric_is_identical_under_a_metric_declaration(tmp_path):
    def build(system):
        ctx = VaultContext.local(tmp_path / f"{system}.db", user_id="t",
                                 writable=True)
        conn = ctx.connect()
        db.init_db(conn)
        for i in range(DAYS):
            day = (START + timedelta(days=i)).isoformat()
            conn.execute(
                "INSERT INTO daily_metrics "
                "(metric, date, count, sum, avg, unit, source_kind) "
                "VALUES ('step_count', ?, 1, ?, ?, 'count', 'cumulative')",
                (day, 1000.0 + i, 1000.0 + i))
        V.set_unit_system(conn, system)
        conn.commit()
        conn.close()
        return S.build_tools(ctx)
    end = (START + timedelta(days=DAYS - 1)).isoformat()
    a = build("imperial")["get_weekly_series"]("step_count", START.isoformat(), end)
    b = build("metric")["get_weekly_series"]("step_count", START.isoformat(), end)
    assert a == b


# The field of a weekly row that carries the same figure the daily-series
# tool's weekly downsample reports for that week, keyed by the metric's reducer.
_AGG_FIELD = {"sum": "total", "last": "latest", "mean": "mean"}


@pytest.mark.parametrize("system", ["imperial", "metric"])
@pytest.mark.parametrize("metric", sorted(SERIES))
def test_weekly_rows_name_the_reducer_and_carry_the_downsampled_figure(
        tmp_path, monkeypatch, metric, system):
    tools = _vault(tmp_path / f"{system}.db", system)
    end = (START + timedelta(days=DAYS - 1)).isoformat()
    # Force the daily-series tool onto its weekly downsample for this range.
    monkeypatch.setattr(S, "MAX_SERIES_POINTS", 7)
    daily = tools["get_daily_series"](metric, START.isoformat(), end)
    weekly = _weekly(tools, metric)
    assert daily["downsampled"] is True
    assert len(daily["points"]) == len(weekly["weeks"]) == 6
    expected_agg = SERIES[metric][1].replace("avg", "mean")
    for point, row in zip(daily["points"], weekly["weeks"]):
        assert row["agg"] == daily["downsample_agg"] == expected_agg
        assert row["week_start"] == point["date"]
        assert row["n_days"] == point["days"]
        # In stored units the two figures are equal. A metric vault converts
        # the weekly figure after it is rounded to 0.01 and the downsample
        # before, so the two may differ by that one rounding step.
        tolerance = 1e-9 if system == "imperial" else 0.0101
        assert row[_AGG_FIELD[row["agg"]]] == pytest.approx(
            point["value"], abs=tolerance)
        # The weekly mean is kept for every metric, whatever its reducer.
        assert "mean" in row
        # A figure that does not describe this metric is not published.
        for field in set(_AGG_FIELD.values()) - {"mean", _AGG_FIELD[row["agg"]]}:
            assert field not in row


def test_the_weekly_latest_is_the_weeks_final_reading_not_its_mean(tmp_path):
    tools = _vault(tmp_path / "i.db", "imperial")
    row = _weekly(tools, "body_mass")["weeks"][0]
    stored = [_value(list(SERIES).index("body_mass"), i) for i in range(7)]
    assert row["latest"] == pytest.approx(stored[-1], abs=0.006)
    assert row["latest"] != row["mean"]
