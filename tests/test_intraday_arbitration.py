"""get_intraday must arbitrate sources the same way the daily read path does.

`daily_metrics` is built through `db._arbitration()`, which drops a mirror
source once the Apple devices took over and drops whole-day estimates wearing a
sample's clothes. `get_intraday` aggregated raw `records` with no such filter,
so the same metric and day could come back inflated — measured across the live
DB on 2026-08-09: 3,114 (metric, day) pairs disagree, worst case 430x on
step_count, and `basal_energy` was still affected as recently as 2026-07-21.

That is the worst failure mode this project has. The agent cannot tell the two
apart: get_daily_series says one number, get_intraday says another, and both
look authoritative.

These exercise `step_count`, which is what the evidence is about. They briefly
ran on `distance_walking_running` while step_count sat outside the D3 raw
allowlist; #16 put it back in (bucketed at five minutes), so the reason for the
detour is gone.
"""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import db as D
from health_advisor import mcp_server as S


def _seed(conn, rows):
    for metric, value, ts, day, source in rows:
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES (?, ?, 'count', ?, ?, ?, ?, ?, 'test', ?)",
            (metric, value, ts, ts, ts, day, source,
             f"{metric}|{ts}|{source}|{value}"),
        )
    conn.commit()
    D.recompute_daily_metrics(conn, full=True)
    for pair in D.arbitrated_pairs(conn):
        D.recompute_daily_metrics(conn, pairs=[pair])
    conn.commit()


@pytest.fixture
def mirrored_db(conn, vault_path):
    """A day after the mirror cutoff where both a mirror and Apple wrote it."""
    path = vault_path
    day = "2019-06-24"
    rows = []
    for h in range(6):
        ts = f"{day}T{10 + h:02d}:00:00"
        rows.append(("step_count", 500.0, ts, day, "Demo's Apple Watch"))
        rows.append(("step_count", 500.0, ts, day, "Sync Solver"))
    _seed(conn, rows)
    return path, day


def test_intraday_total_matches_the_arbitrated_daily_total(mirrored_db, tools):
    path, day = mirrored_db
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    daily = conn.execute(
        "SELECT sum FROM daily_metrics WHERE metric='step_count' AND date=?",
        (day,)).fetchone()["sum"]
    raw = conn.execute(
        "SELECT SUM(value) s FROM records WHERE metric='step_count' "
        "AND local_date=?", (day,)).fetchone()["s"]
    conn.close()

    assert raw == pytest.approx(daily * 2), "fixture must actually double-count"

    out = tools.get_intraday("step_count", day, bucket_hours=24)
    total = sum(b["value"] for b in out["buckets"])

    assert total == pytest.approx(daily), (
        f"intraday total {total} disagrees with the daily read path {daily}")


def test_intraday_buckets_are_not_inflated_by_the_mirror_source(mirrored_db, tools):
    _, day = mirrored_db
    out = tools.get_intraday("step_count", day, bucket_hours=1)
    # Six hours of 500 each, not 1000.
    values = [b["value"] for b in out["buckets"] if b["value"]]
    assert values == [pytest.approx(500.0)] * 6, values


def test_intraday_still_returns_data_when_no_arbitration_applies(conn, vault_path,
                                                                tools):
    path = vault_path
    day = "2026-08-01"
    rows = [("step_count", 100.0, f"{day}T{9 + h:02d}:00:00", day,
             "Demo's Apple Watch") for h in range(4)]
    _seed(conn, rows)

    out = tools.get_intraday("step_count", day, bucket_hours=24)

    assert sum(b["value"] for b in out["buckets"]) == pytest.approx(400.0)


def test_intraday_leaves_averaged_metrics_alone(conn, tools):
    """Arbitration is a cumulative-metric rule; an averaged metric must be
    untouched by it, mirror source or not."""
    day = "2019-06-24"
    rows = []
    for h in range(4):
        ts = f"{day}T{10 + h:02d}:00:00"
        rows.append(("heart_rate", 60.0, ts, day, "Demo's Apple Watch"))
        rows.append(("heart_rate", 80.0, ts, day, "Sync Solver"))
    _seed(conn, rows)

    out = tools.get_intraday("heart_rate", day, bucket_hours=24)

    assert out["buckets"][0]["value"] == pytest.approx(70.0)


def test_intraday_refuses_a_series_the_vault_does_not_carry(conn, tools):
    """D3/T-006. `basal_energy` has raw rows in the full snapshot and none in a
    vault, so the tool answers from the contract rather than from what this
    particular database happens to hold — otherwise it works in development and
    returns a silent empty day in production."""
    day = "2026-08-01"
    _seed(conn, [("basal_energy", 100.0, f"{day}T09:00:00", day,
                  "Demo's Apple Watch")])

    out = tools.get_intraday("basal_energy", day)

    assert out["status"] == "unavailable"
    assert out["reason"] == "raw_series_not_in_vault"
    assert "buckets" not in out, "an empty bucket list is the failure, not the answer"
    assert "get_daily_series" in out["detail"], "say what to use instead"
