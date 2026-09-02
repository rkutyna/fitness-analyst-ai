"""Daily point-in-time aggregates must preserve the final raw observation.

The audit found that body-mass days with multiple scale readings were served
as their mean even though the catalog says ``agg: last``.  These tests pin the
schema migration, both recompute paths, deterministic tie-breaking, and the
analytic column selection that exposed the wrong value.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys

import pytest

from health_advisor import db as dbmod
from health_advisor import derive
from health_advisor import metrics


def _record(conn, metric, value, start, end=None, day="2026-07-21", source="test"):
    end = end or start
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, dedupe_key) VALUES (?, ?, 'lb', ?, ?, ?, "
        "?, 'receiver', ?)",
        (metric, value, start, end, day, source,
         f"{metric}|{value}|{start}|{end}|{source}"),
    )


def test_init_adds_last_to_a_preexisting_daily_metrics_table(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE daily_metrics (metric TEXT, date TEXT, count INTEGER, "
                 "sum REAL, avg REAL, min REAL, max REAL, unit TEXT, "
                 "PRIMARY KEY (metric, date))")
    dbmod.init_db(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(daily_metrics)")}
    assert "last" in columns
    conn.close()


def test_recompute_stores_the_last_sample_not_the_daily_mean(conn):
    _record(conn, "body_mass", 190.59, "2026-07-21T08:04:00+00:00")
    _record(conn, "body_mass", 188.49, "2026-07-21T12:58:00+00:00")

    dbmod.recompute_daily_metrics(conn, pairs=[("body_mass", "2026-07-21")])
    row = conn.execute("SELECT avg, last FROM daily_metrics").fetchone()
    assert row["avg"] == pytest.approx(189.54)
    assert row["last"] == 188.49
    assert metrics.value_col("body_mass") == "last"


def test_last_has_a_total_order_when_start_times_tie(conn):
    start = "2026-07-21T12:00:00+00:00"
    _record(conn, "body_mass", 180.0, start, "2026-07-21T12:00:01+00:00")
    _record(conn, "body_mass", 181.0, start, "2026-07-21T12:00:02+00:00")

    dbmod.recompute_daily_metrics(conn, pairs=[("body_mass", "2026-07-21")])
    assert conn.execute("SELECT last FROM daily_metrics").fetchone()[0] == 181.0


def test_full_and_incremental_rebuilds_populate_the_same_last(conn):
    _record(conn, "body_mass", 190.0, "2026-07-21T08:00:00+00:00")
    _record(conn, "body_mass", 189.0, "2026-07-21T09:00:00+00:00")
    dbmod.recompute_daily_metrics(conn, full=True)
    full = conn.execute("SELECT last FROM daily_metrics").fetchone()[0]

    conn.execute("UPDATE daily_metrics SET last = NULL")
    dbmod.recompute_daily_metrics(conn, pairs=[("body_mass", "2026-07-21")])
    incremental = conn.execute("SELECT last FROM daily_metrics").fetchone()[0]
    assert full == incremental == 189.0


def test_last_uses_the_same_arbitrated_rows_as_the_other_columns(conn):
    _record(conn, "step_count", 100.0, "2026-07-21T08:00:00+00:00", source="Demo’s Apple Watch")
    _record(conn, "step_count", 900.0, "2026-07-21T09:00:00+00:00", source="Sync Solver")

    dbmod.recompute_daily_metrics(conn, full=True)
    row = conn.execute("SELECT sum, last FROM daily_metrics").fetchone()
    assert row["sum"] == row["last"] == 100.0


def test_derived_rows_populate_last_for_schema_consistency(conn):
    derive._upsert(conn, "sleep_bedtime", "2026-07-21", 10.5)
    row = conn.execute("SELECT avg, last FROM daily_metrics").fetchone()
    assert row["last"] == row["avg"] == 10.5


def test_backfill_is_dry_by_default_idempotent_and_changes_only_last(tmp_path):
    path = tmp_path / "backfill.db"
    conn = dbmod.connect(path)
    dbmod.init_db(conn)
    _record(conn, "body_mass", 190.0, "2026-07-21T08:00:00+00:00")
    _record(conn, "body_mass", 189.0, "2026-07-21T09:00:00+00:00")
    dbmod.recompute_daily_metrics(conn, pairs=[("body_mass", "2026-07-21")])
    conn.execute("UPDATE daily_metrics SET last = NULL")
    conn.commit()
    before = tuple(conn.execute(
        "SELECT count, sum, avg, min, max, last, unit FROM daily_metrics").fetchone())
    conn.close()

    script = "scripts/backfill_daily_last.py"
    dry = subprocess.run([sys.executable, script, "--db", str(path)],
                         capture_output=True, text=True, check=True)
    assert "dry-run: no changes written" in dry.stdout
    conn = dbmod.connect(path, read_only=True)
    assert tuple(conn.execute(
        "SELECT count, sum, avg, min, max, last, unit FROM daily_metrics").fetchone()) == before
    conn.close()

    subprocess.run([sys.executable, script, "--apply", "--db", str(path)], check=True)
    conn = dbmod.connect(path, read_only=True)
    assert conn.execute("SELECT last FROM daily_metrics").fetchone()[0] == 189.0
    conn.close()
    verified = subprocess.run(
        [sys.executable, "scripts/verify_daily_metrics.py", "--db", str(path)],
        capture_output=True, text=True, check=True)
    assert "OK" in verified.stdout


def test_daily_last_is_the_same_sample_the_raw_table_ends_on(conn):
    """`daily_metrics.last` means value-at-latest-timestamp, and the check is
    that it agrees with the raw row it was taken from.

    Asserted against the tables rather than through `get_latest`, because
    `body_mass` has no raw samples in a D3 vault (T-006) and the tool therefore
    reports the sample half as unavailable there. The invariant is a property of
    ingestion, not of the tool, so it is pinned where it still holds."""
    _record(conn, "body_mass", 190.0, "2026-07-21T08:00:00+00:00")
    _record(conn, "body_mass", 189.0, "2026-07-21T09:00:00+00:00")
    dbmod.recompute_daily_metrics(conn, pairs=[("body_mass", "2026-07-21")])
    conn.commit()

    stored = conn.execute(
        "SELECT last FROM daily_metrics WHERE metric='body_mass' "
        "AND date='2026-07-21'").fetchone()["last"]
    raw = conn.execute(
        "SELECT value FROM records WHERE metric='body_mass' "
        "AND local_date='2026-07-21' "
        "ORDER BY start_utc DESC, end_utc DESC, id DESC LIMIT 1").fetchone()["value"]
    assert stored == raw == 189.0


def test_get_latest_says_the_sample_is_unavailable_rather_than_null(conn, tools):
    """D3/T-006. The daily aggregate travels with every vault; the raw sample
    does not. `latest_sample: null` alone would be a claim about the data — that
    the last reading carries no timestamp — so the tool has to say which it is."""
    _record(conn, "body_mass", 189.0, "2026-07-21T09:00:00+00:00")
    dbmod.recompute_daily_metrics(conn, pairs=[("body_mass", "2026-07-21")])
    conn.commit()

    result = tools.get_latest("body_mass")

    assert result["latest_day"]["value"] == 189.0, "the aggregate is still there"
    assert result["latest_sample"] is None
    assert result["latest_sample_status"]["status"] == "unavailable"
    assert result["latest_sample_status"]["reason"] == "raw_series_not_in_vault"


def test_get_latest_still_carries_the_sample_for_an_allowlisted_series(conn, tools):
    _record(conn, "heart_rate", 61.0, "2026-07-21T08:00:00+00:00")
    _record(conn, "heart_rate", 58.0, "2026-07-21T09:00:00+00:00")
    dbmod.recompute_daily_metrics(conn, pairs=[("heart_rate", "2026-07-21")])
    conn.commit()

    result = tools.get_latest("heart_rate")

    assert result["latest_sample"]["value"] == 58.0
    assert "latest_sample_status" not in result
