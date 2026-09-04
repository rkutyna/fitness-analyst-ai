from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from health_advisor import analysis as A
from health_advisor import db, metrics as M, running_form as RF, vault as V
from health_advisor import mcp_server as S
from health_advisor.context import VaultContext


def _seed_running_vault(path, unit_system):
    ctx = VaultContext.local(path, user_id="test", writable=True)
    conn = ctx.connect()
    db.init_db(conn)
    conn.execute(
        "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, last, unit) "
        "VALUES ('step_count', '2026-08-01', 1, 1, 1, 1, 1, 1, 'count')"
    )
    start = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
    end = start + timedelta(minutes=1)
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, distance_mi, unit_distance, source, dedupe_key) "
        "VALUES ('running', ?, ?, '2026-08-01', 1, 0.1, 'mi', 'test', 'run')",
        (start.isoformat(), end.isoformat()),
    )
    for i in range(3):
        sample = start + timedelta(seconds=20 * i)
        stamp = sample.isoformat()
        conn.execute(
            "INSERT INTO records (metric, start_utc, end_utc, local_date, value, unit, "
            "source, dedupe_key) VALUES ('distance_walking_running', ?, ?, "
            "'2026-08-01', 0.0333, 'mi', 'test', ?)",
            (stamp, stamp, f"distance-{i}"),
        )
        conn.execute(
            "INSERT INTO records (metric, start_utc, end_utc, local_date, value, unit, "
            "source, dedupe_key) VALUES ('step_count', ?, ?, '2026-08-01', "
            "47, 'count', 'test', ?)",
            (stamp, stamp, f"steps-{i}"),
        )
    if unit_system is not None:
        V.set_unit_system(conn, unit_system)
    conn.commit()
    conn.close()
    return ctx


def test_metric_and_undeclared_vaults_change_labels_not_physical_values(tmp_path):
    legacy = _seed_running_vault(tmp_path / "legacy.db", None)
    metric = _seed_running_vault(tmp_path / "metric.db", "metric")
    factor = V.UNIT_CONVERSION_FACTORS["distance_mi_to_km"]

    def views(ctx):
        metric_units = ctx.settings()["unit_system"] == "metric"
        tools = S.build_tools(ctx)
        conn = ctx.read_only()
        try:
            impact = tools["get_impact_volume"](
                "2026-08-01", "2026-08-01", by="day")["periods"][0]
            focus = A.workout_focus(conn, "2026-08-01", metric_units=metric_units)
            buckets = M.bucket_series(
                conn, "2026-08-01T12:00:00Z", "2026-08-01T12:01:00Z",
                metric_units=metric_units,
            )
            collapsed = RF._collapse_bucket_rows(buckets, metric_units)
            return impact, focus, buckets[0], collapsed[0]
        finally:
            conn.close()

    old_impact, old_focus, old_bucket, old_collapsed = views(legacy)
    new_impact, new_focus, new_bucket, new_collapsed = views(metric)

    assert "jog_miles" in old_impact and "jog_km" not in old_impact
    assert "jog_km" in new_impact and "jog_miles" not in new_impact
    assert new_impact["jog_km"] / factor == pytest.approx(old_impact["jog_miles"], abs=0.01)

    assert "pace_min_per_mi" in old_focus and "pace_min_per_km" not in old_focus
    assert "pace_min_per_km" in new_focus and "pace_min_per_mi" not in new_focus
    assert new_focus["pace_min_per_km"] * factor == pytest.approx(
        old_focus["pace_min_per_mi"], abs=0.1)

    assert "speed_mph" in old_bucket and "speed_kph" not in old_bucket
    assert "speed_kph" in new_bucket and "speed_mph" not in new_bucket
    assert new_bucket["speed_kph"] / factor == pytest.approx(old_bucket["speed_mph"])
    assert new_bucket["pace_min_per_km"] * factor == pytest.approx(
        old_bucket["pace_min_per_mi"])
    assert "speed_kph" in new_collapsed and "speed_mph" not in new_collapsed
    assert "pace_min_per_km" in new_collapsed and "pace_min_per_mi" not in new_collapsed
