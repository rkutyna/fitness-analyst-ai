"""movers reads the wear-hours map once per call, not once per metric (#110)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from health_advisor import analysis as A
from tests.conftest import seed_metric

AS_OF = "2026-06-18"
START = (date.fromisoformat(AS_OF) - timedelta(days=A.MOVER_WINDOW_DAYS - 1)
         ).isoformat()
METRICS = ("step_count", "active_energy", "flights_climbed",
           "distance_walking_running")
WEAR_MAP_SQL = "v FROM daily_metrics WHERE metric = 'wear_hours'"


@pytest.fixture
def seeded(conn):
    n = A.MOVER_WINDOW_DAYS
    for i, metric in enumerate(METRICS):
        seed_metric(conn, metric, START, [100 + i] * (n - 7) + [300 + i] * 7,
                    counts=[5] * 6 + [40000] * (n - 6))
    # Wear hours cover most days but not all, and a few are under the minimum.
    hours = [12.0 if d % 5 else 3.0 for d in range(n)]
    hours = [h for d, h in enumerate(hours) if d not in (2, 9)]
    days = [d for d in range(n) if d not in (2, 9)]
    for day, h in zip(days, hours):
        seed_metric(conn, "wear_hours",
                    (date.fromisoformat(START) + timedelta(days=day)).isoformat(),
                    [h])
    # A consolidated day with wear hours, and one without.
    for day in (days[3], 2):
        conn.execute(
            "UPDATE daily_metrics SET source_kind = 'apple_consolidated' "
            "WHERE metric = 'step_count' AND date = ?",
            ((date.fromisoformat(START) + timedelta(days=day)).isoformat(),))
    conn.commit()
    return conn


def _wear_map_reads(conn) -> int:
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        A.movers(conn, as_of=AS_OF, scope="deep", _include_cold_start=False)
    finally:
        conn.set_trace_callback(None)
    return sum(WEAR_MAP_SQL in s for s in statements)


def test_movers_reads_the_wear_hours_map_once(seeded):
    assert _wear_map_reads(seeded) == 1


def test_movers_without_a_long_enough_metric_reads_none(conn):
    seed_metric(conn, "step_count", START, [100] * 5)
    assert _wear_map_reads(conn) == 0


@pytest.mark.parametrize("metric", METRICS)
def test_a_window_wide_map_selects_the_same_rows_as_the_per_metric_read(
        seeded, metric):
    rows = A._daily_load_rows(seeded, metric, AS_OF, A.MOVER_WINDOW_DAYS)
    wide = A._wear_hours_map(seeded, START, AS_OF)
    assert A._wear_eligible(seeded, metric, AS_OF, rows, wide) == (
        A._wear_eligible(seeded, metric, AS_OF, rows))
    assert A._worn_rows(seeded, metric, AS_OF, rows, wide) == (
        A._worn_rows(seeded, metric, AS_OF, rows))
    assert A._worn_values(seeded, metric, AS_OF, rows, wide) == (
        A._worn_values(seeded, metric, AS_OF, rows))


def test_the_consolidated_day_without_wear_hours_really_leaves_the_window(seeded):
    rows = A._daily_load_rows(seeded, "step_count", AS_OF, A.MOVER_WINDOW_DAYS)
    kept = A._wear_eligible(seeded, "step_count", AS_OF, rows)
    assert len(kept) == len(rows) - 1


def test_movers_payload_is_unchanged_by_the_shared_read(seeded):
    shared = A.movers(seeded, as_of=AS_OF, scope="deep",
                      _include_cold_start=False)
    assert list(shared), "fixture should produce movers"
    expected = []
    for metric in METRICS:
        rows = A._daily_load_rows(seeded, metric, AS_OF, A.MOVER_WINDOW_DAYS)
        expected.append((metric, A._worn_values(seeded, metric, AS_OF, rows)))
    by_metric = {row["metric"]: row for row in shared}
    for metric, vals in expected:
        if metric in by_metric:
            assert by_metric[metric]["n_days"] == len(vals)
    json.dumps(list(shared), sort_keys=True)
