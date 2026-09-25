"""Consumer #430 (M6) step 1: migration_id + prior_sum, and their revert.

D7/D19 decision brief 20260925 item 2: historical rows are written `settled`,
but only once a `migration_id` column exists AND each revision row records the
`daily_metrics.sum` it overwrote (`prior_sum`). The reason is D3 (consumer
#37): behind a compaction watermark a non-allowlisted series' day has no raw
rows left, so delete-and-recompute (`db._without_frozen_pairs`) cannot restore
it -- only a recorded prior value can, which is what this file's acceptance
test exercises against the REAL compaction/freeze mechanism, not a hand-set
flag.

Step 1 only: the history pull, staging import, report script and `--apply`
(build steps 2-4 of the brief's "what happens next") are not built here.
"""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import db
from health_advisor import vault as V


WATERMARK = "2026-07-22"
FLIGHTS = "flights_climbed"


def _raw(metric, day, key, value=3.0, *, origin="receiver", source="test",
         hk_uuid=None, hour=8, unit="count"):
    ts = f"{day}T{hour:02d}:00:00+00:00"
    return {"metric": metric, "value": value, "unit": unit, "start_utc": ts,
            "end_utc": ts, "start_local": f"{day} {hour:02d}:00:00",
            "local_date": day, "source": source, "origin": origin,
            "dedupe_key": key, "hk_uuid": hk_uuid}


def _dm(conn, metric, day):
    row = conn.execute(
        "SELECT count, sum, avg, min, max, last, unit, source_kind "
        "FROM daily_metrics WHERE metric = ? AND date = ?", (metric, day)
    ).fetchone()
    return dict(row) if row else None


def _historical(metric, day, value, *, unit="count", device_id="migration-1",
                 queried_at="2026-09-25T09:00:00-04:00"):
    return {"metric": metric, "local_date": day, "value": value, "unit": unit,
            "interval": "day", "device_id": device_id, "queried_at": queried_at}


def _hk_totals_row(conn, metric, day):
    row = conn.execute(
        "SELECT state, migration_id, value FROM hk_daily_totals "
        "WHERE metric = ? AND local_date = ?", (metric, day)).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# The brief's acceptance test: a frozen flights_climbed day
# --------------------------------------------------------------------------- #
def test_frozen_flights_day_migrates_then_reverts_byte_identical(conn):
    V.declare_vault(conn)
    db.insert_records(conn, [
        _raw(FLIGHTS, "2026-07-10", "f1", 3.0, hk_uuid="u-f1"),
        _raw(FLIGHTS, "2026-07-10", "f2", 5.0, hk_uuid="u-f2", hour=12),
    ])
    db.recompute_daily_metrics(conn, pairs=[(FLIGHTS, "2026-07-10")])
    conn.commit()
    before = _dm(conn, FLIGHTS, "2026-07-10")
    assert before == {"count": 2, "sum": 8.0, "avg": 4.0, "min": 3.0,
                      "max": 5.0, "last": 5.0, "unit": "count",
                      "source_kind": "records"}

    # Freeze it through the REAL compaction mechanism: flights_climbed is not
    # in VAULT_RAW_SERIES, so compact() removes its raw rows for this day.
    assert V.compact(conn, through=WATERMARK) == {FLIGHTS: 2}
    assert V.frozen_through(conn) == WATERMARK
    assert V.is_frozen(FLIGHTS, "2026-07-10", WATERMARK)
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric = ?", (FLIGHTS,)
    ).fetchone()[0] == 0  # no raw rows survive: delete-and-recompute cannot rebuild this

    written = db.write_historical_consolidated_totals(
        conn, [_historical(FLIGHTS, "2026-07-10", 9.0)],
        migration_id="i430-flights-v1")
    assert written == 1

    migrated = _dm(conn, FLIGHTS, "2026-07-10")
    assert migrated["sum"] == 9.0
    assert migrated["source_kind"] == "apple_consolidated"
    # Untouched by apply_consolidated_totals (D19): count/avg/min/max/last stay
    # exactly as the pre-freeze records-derived row left them.
    assert migrated["count"] == before["count"]
    assert migrated["avg"] == before["avg"]
    assert migrated["min"] == before["min"]
    assert migrated["max"] == before["max"]
    assert migrated["last"] == before["last"]

    totals_row = _hk_totals_row(conn, FLIGHTS, "2026-07-10")
    assert totals_row == {"state": "settled", "migration_id": "i430-flights-v1",
                          "value": 9.0}

    result = db.revert_historical_migration(conn, "i430-flights-v1")
    assert result == {"restored": 1, "deleted": 0}
    assert _dm(conn, FLIGHTS, "2026-07-10") == before
    assert conn.execute(
        "SELECT COUNT(*) FROM hk_daily_totals WHERE metric = ?", (FLIGHTS,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM hk_daily_total_revisions WHERE metric = ?",
        (FLIGHTS,)).fetchone()[0] == 0


# --------------------------------------------------------------------------- #
# A migrated day that was never frozen reverts identically too
# --------------------------------------------------------------------------- #
def test_unfrozen_day_migrates_then_reverts_byte_identical(conn):
    # step_count IS in VAULT_RAW_SERIES: nothing here is ever frozen. A
    # migration must restore this case exactly as well as the frozen one.
    V.declare_vault(conn)
    db.insert_records(conn, [
        _raw("step_count", "2019-06-01", "s1", 4000.0, unit="count"),
        _raw("step_count", "2019-06-01", "s2", 5000.0, unit="count", hour=14),
    ])
    db.recompute_daily_metrics(conn, pairs=[("step_count", "2019-06-01")])
    conn.commit()
    before = _dm(conn, "step_count", "2019-06-01")
    assert before["sum"] == 9000.0 and before["source_kind"] == "records"
    assert V.frozen_through(conn) is None

    db.write_historical_consolidated_totals(
        conn, [_historical("step_count", "2019-06-01", 7200.0)],
        migration_id="i430-steps-v1")
    migrated = _dm(conn, "step_count", "2019-06-01")
    assert migrated["sum"] == 7200.0
    assert migrated["source_kind"] == "apple_consolidated"

    result = db.revert_historical_migration(conn, "i430-steps-v1")
    assert result == {"restored": 1, "deleted": 0}
    assert _dm(conn, "step_count", "2019-06-01") == before


# --------------------------------------------------------------------------- #
# Reverting one migration leaves another migration and every ordinary
# (NULL migration_id) settled row untouched
# --------------------------------------------------------------------------- #
def test_revert_is_scoped_to_its_own_migration_id(conn):
    V.declare_vault(conn)
    db.insert_records(conn, [
        _raw("step_count", "2020-01-01", "a1", 1000.0, unit="count"),
        _raw("step_count", "2020-01-02", "a2", 2000.0, unit="count"),
    ])
    db.recompute_daily_metrics(
        conn, pairs=[("step_count", "2020-01-01"), ("step_count", "2020-01-02")])
    conn.commit()
    before_day1 = _dm(conn, "step_count", "2020-01-01")
    before_day2 = _dm(conn, "step_count", "2020-01-02")

    # An ordinary live D19 pull, migration_id NULL, on a THIRD day.
    db.insert_daily_totals(conn, [{
        "metric": "step_count", "local_date": "2026-08-25", "value": 10173.0,
        "unit": "count", "interval": "day", "state": "settled",
        "device_id": "dev-1", "queried_at": "2026-08-26T09:00:00-04:00"}],
        batch_id="live-batch")
    db.apply_consolidated_totals(
        conn, pairs=[("step_count", "2026-08-25")])
    ordinary_before = _dm(conn, "step_count", "2026-08-25")
    assert ordinary_before["source_kind"] == "apple_consolidated"

    db.write_historical_consolidated_totals(
        conn, [_historical("step_count", "2020-01-01", 900.0)],
        migration_id="i430-a")
    db.write_historical_consolidated_totals(
        conn, [_historical("step_count", "2020-01-02", 1800.0)],
        migration_id="i430-b")

    result = db.revert_historical_migration(conn, "i430-a")
    assert result == {"restored": 1, "deleted": 0}

    # migration "a"'s day is back to its pre-migration state.
    assert _dm(conn, "step_count", "2020-01-01") == before_day1
    assert conn.execute(
        "SELECT COUNT(*) FROM hk_daily_totals WHERE migration_id = 'i430-a'"
    ).fetchone()[0] == 0

    # migration "b"'s day is untouched.
    day2 = _dm(conn, "step_count", "2020-01-02")
    assert day2["sum"] == 1800.0 and day2["source_kind"] == "apple_consolidated"
    assert _hk_totals_row(conn, "step_count", "2020-01-02")["migration_id"] == "i430-b"

    # The ordinary settled row (migration_id NULL) is untouched.
    assert _dm(conn, "step_count", "2026-08-25") == ordinary_before
    assert _hk_totals_row(conn, "step_count", "2026-08-25")["migration_id"] is None


# --------------------------------------------------------------------------- #
# The settled triggers still refuse UPDATE/DELETE on an ordinary settled row
# --------------------------------------------------------------------------- #
def test_ordinary_settled_row_still_rejects_update_and_delete(conn):
    db.insert_daily_totals(conn, [{
        "metric": "step_count", "local_date": "2026-08-25", "value": 10173.0,
        "unit": "count", "interval": "day", "state": "settled",
        "device_id": "dev-1", "queried_at": "2026-08-26T09:00:00-04:00"}],
        batch_id="b1")

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE hk_daily_totals SET value = 1.0 "
            "WHERE metric = 'step_count' AND local_date = '2026-08-25'")
    with pytest.raises(sqlite3.IntegrityError, match="not deletable"):
        conn.execute(
            "DELETE FROM hk_daily_totals "
            "WHERE metric = 'step_count' AND local_date = '2026-08-25'")


def test_migration_row_permits_update_and_delete(conn):
    """The other side of the same trigger: a migration_id row is not stuck."""
    db.write_historical_consolidated_totals(
        conn, [_historical("step_count", "2019-01-01", 100.0)],
        migration_id="i430-c")
    # Re-run under the same id with a corrected value -- an UPDATE the OLD
    # trigger body would have refused outright.
    db.write_historical_consolidated_totals(
        conn, [_historical("step_count", "2019-01-01", 150.0)],
        migration_id="i430-c")
    assert _hk_totals_row(conn, "step_count", "2019-01-01")["value"] == 150.0
    # And a revert, which DELETEs it.
    db.revert_historical_migration(conn, "i430-c")
    assert _hk_totals_row(conn, "step_count", "2019-01-01") is None


# --------------------------------------------------------------------------- #
# The ALTER path is idempotent against a vault that predates these columns
# --------------------------------------------------------------------------- #
def test_migration_columns_added_idempotently_to_an_old_shape_vault(vault_path):
    raw = sqlite3.connect(vault_path)
    raw.executescript(
        """
        CREATE TABLE hk_daily_totals (
            metric TEXT NOT NULL, local_date TEXT NOT NULL, value REAL NOT NULL,
            unit TEXT NOT NULL, interval TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('provisional','settled')),
            device_id TEXT NOT NULL, queried_at TEXT NOT NULL,
            first_seen_at TEXT NOT NULL, settled_at TEXT,
            PRIMARY KEY (metric, local_date)
        );
        CREATE TABLE hk_daily_total_revisions (
            id INTEGER PRIMARY KEY, metric TEXT NOT NULL, local_date TEXT NOT NULL,
            from_value REAL, to_value REAL NOT NULL, from_state TEXT,
            to_state TEXT NOT NULL, lag_days INTEGER NOT NULL,
            batch_id TEXT NOT NULL, recorded_at TEXT NOT NULL
        );
        INSERT INTO hk_daily_totals VALUES
            ('step_count', '2026-08-01', 5000.0, 'count', 'day', 'settled',
             'dev-1', '2026-08-02T09:00:00', '2026-08-01T09:00:00',
             '2026-08-02T09:00:00');
        """
    )
    raw.commit()
    raw.close()

    conn = sqlite3.connect(vault_path)
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.init_db(conn)  # idempotent: a second call must not raise

    have_totals = [r[1] for r in conn.execute("PRAGMA table_info(hk_daily_totals)")]
    have_rev = [r[1] for r in conn.execute(
        "PRAGMA table_info(hk_daily_total_revisions)")]
    assert have_totals.count("migration_id") == 1
    assert have_rev.count("migration_id") == 1
    assert have_rev.count("prior_sum") == 1
    assert have_rev.count("prior_source_kind") == 1

    # The pre-existing row survives, with the new column reading NULL --
    # an old row is an ordinary live pull, not a migration.
    row = conn.execute(
        "SELECT value, migration_id FROM hk_daily_totals "
        "WHERE metric = 'step_count' AND local_date = '2026-08-01'"
    ).fetchone()
    assert row["value"] == 5000.0
    assert row["migration_id"] is None

    # And the recreated trigger still refuses this old, ordinary settled row.
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE hk_daily_totals SET value = 1.0 "
            "WHERE metric = 'step_count' AND local_date = '2026-08-01'")
    conn.close()
