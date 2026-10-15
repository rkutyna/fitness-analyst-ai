"""Cold-start builds only the requested surface's probe set (#110)."""
from __future__ import annotations

import json

import pytest

from health_advisor import cold_start
from tests.conftest import seed_metric

AS_OF = "2026-01-20"
MOVERS_SCAN = "SELECT DISTINCT metric FROM daily_metrics"


@pytest.fixture
def seeded(conn):
    start = "2026-01-01"
    seed_metric(conn, "heart_rate_variability", start, [50 + i for i in range(20)])
    seed_metric(conn, "resting_heart_rate", start, [60 + i % 3 for i in range(20)])
    seed_metric(conn, "sleep_asleep", start, [450] * 20)
    seed_metric(conn, "active_energy", start, [500 + i for i in range(20)])
    seed_metric(conn, "step_count", start, [1000 + i for i in range(20)])
    return conn


def _movers_scans(conn, **kwargs) -> int:
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        cold_start.describe(conn, AS_OF, **kwargs)
    finally:
        conn.set_trace_callback(None)
    return sum(MOVERS_SCAN in s for s in statements)


@pytest.mark.parametrize("surface", ["readiness", "training_load", "coverage"])
def test_other_surfaces_do_not_run_the_movers_scan(seeded, surface):
    assert _movers_scans(seeded, surfaces={surface}) == 0


def test_correlate_does_not_run_the_movers_scan(seeded):
    assert _movers_scans(seeded, metric_x="step_count",
                         metric_y="active_energy", surfaces={"correlate"}) == 0


def test_movers_surface_still_runs_its_scan_once(seeded):
    assert _movers_scans(seeded, surfaces={"movers"}) >= 1


@pytest.mark.parametrize("surface", ["readiness", "training_load", "movers"])
def test_a_single_surface_block_equals_the_full_describe_block(seeded, surface):
    full = cold_start.describe(seeded, AS_OF)
    alone = cold_start.describe(seeded, AS_OF, surfaces={surface})
    assert (json.dumps(alone[surface], sort_keys=True)
            == json.dumps(full[surface], sort_keys=True))
