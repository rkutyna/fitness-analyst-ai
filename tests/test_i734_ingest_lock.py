"""Downstream #734: the ingest's month recount is sargable, and init_db's
fast path takes no write lock on a current vault.

Every oracle here is a hand-written literal. None of them is read back from
the code under test.
"""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import db

SCHEMA_KEY = "schema_fingerprint"


def _insert_records(conn: sqlite3.Connection, rows) -> None:
    """rows: (metric, local_date, source) tuples; each gets a unique dedupe_key."""
    for i, (metric, local_date, source) in enumerate(rows):
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "local_date, source, origin, dedupe_key) "
            "VALUES (?, 60.0, 'count/min', ?, ?, ?, ?, 'backfill', ?)",
            (metric, f"{local_date}T12:00:00Z", f"{local_date}T12:01:00Z",
             local_date, source, f"i734|{i}|{metric}|{local_date}|{source}"),
        )
    conn.commit()


def _msm(conn: sqlite3.Connection, metric: str, month: str) -> list[tuple]:
    rows = conn.execute(
        "SELECT source, n FROM metric_source_months "
        "WHERE metric = ? AND month = ? ORDER BY source",
        (metric, month),
    ).fetchall()
    return [tuple(r) for r in rows]


def _stamp(conn: sqlite3.Connection):
    row = conn.execute(
        "SELECT value FROM vault_meta WHERE key = ?", (SCHEMA_KEY,)
    ).fetchone()
    return None if row is None else row[0]


def test_month_recount_counts_exactly_the_rows_whose_local_date_starts_with_the_month(conn):
    _insert_records(conn, [
        ("heart_rate", "2026-11-01", "Watch A"),
        ("heart_rate", "2026-11-30", "Watch A"),
        ("heart_rate", "2026-11-15", "Watch B"),
        ("heart_rate", "2026-11X", None),
        ("heart_rate", "2026-11", "Watch A"),
        ("heart_rate", "2026-10-31", "Watch A"),
        ("heart_rate", "2026-12-01", "Watch A"),
        ("heart_rate", "2026-1", "Watch A"),
        ("heart_rate", "2026-09-30", "Watch A"),
        ("heart_rate", "2026-10-01", "Watch A"),
        ("resting_heart_rate", "2026-11-02", "Watch A"),
    ])

    written = db.rebuild_metric_source_months(
        conn, pairs=[("heart_rate", "2026-11-07"), ("heart_rate", "2026-09-15")]
    )

    # 2026-11: '' (the NULL source) 1, 'Watch A' 3 (11-01, 11-30, "2026-11"),
    # 'Watch B' 1. Excluded: 10-31, 12-01, "2026-1", the other metric.
    assert _msm(conn, "heart_rate", "2026-11") == [
        ("", 1), ("Watch A", 3), ("Watch B", 1),
    ]
    # 2026-09: upper bound "2026-0:"; only 09-30 counts, not 10-01.
    assert _msm(conn, "heart_rate", "2026-09") == [("Watch A", 1)]
    # Three (source) groups for November and one for September.
    assert written == 3 + 1


def test_month_recount_seeks_the_local_date_index(conn):
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM records WHERE " + db._MONTH_RECOUNT_WHERE,
        db._month_recount_params("heart_rate", "2026-11"),
    ).fetchall()
    details = [str(r[-1]) for r in plan]
    assert any(
        "idx_records_metric_localdate" in d and "local_date>" in d for d in details
    ), details


def test_second_init_db_takes_no_write_lock(conn, vault_path):
    c2 = db.connect(vault_path)
    holder = sqlite3.connect(vault_path, isolation_level=None)
    try:
        c2.execute("PRAGMA busy_timeout = 0")
        holder.execute("BEGIN IMMEDIATE")
        db.init_db(c2)  # must not raise: the fast path only reads
    finally:
        holder.execute("ROLLBACK")
        holder.close()
        c2.close()


def test_drifted_trigger_body_still_migrates(conn, vault_path):
    conn.execute("DROP TRIGGER hk_daily_totals_settled_no_delete")
    conn.execute(
        "CREATE TRIGGER hk_daily_totals_settled_no_delete "
        "BEFORE DELETE ON hk_daily_totals WHEN OLD.state = 'settled' "
        "BEGIN SELECT RAISE(ABORT, 'old'); END"
    )
    conn.commit()

    # Proof that the drifted vault takes the full path: under a held write
    # lock with no busy wait, init_db must fail on its first write.
    c2 = db.connect(vault_path)
    holder = sqlite3.connect(vault_path, isolation_level=None)
    try:
        c2.execute("PRAGMA busy_timeout = 0")
        holder.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError):
            db.init_db(c2)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
        c2.close()

    db.init_db(conn)
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
        "AND name = 'hk_daily_totals_settled_no_delete'"
    ).fetchone()[0]
    assert "migration_id IS NULL" in sql


def test_vault_without_stamp_takes_full_path(conn):
    stamp_before = _stamp(conn)
    assert stamp_before is not None
    conn.execute("DELETE FROM vault_meta WHERE key = ?", (SCHEMA_KEY,))
    conn.execute("DROP INDEX idx_records_metric_localdate")
    conn.commit()

    db.init_db(conn)

    index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' "
        "AND name = 'idx_records_metric_localdate'"
    ).fetchone()
    assert index is not None
    assert _stamp(conn) == stamp_before


def test_changed_migration_code_is_not_mistaken_for_current(conn, monkeypatch):
    patched = {t: dict(cols) for t, cols in db._ADDED_COLUMNS.items()}
    patched["records"]["zz_i734_probe"] = "TEXT"
    monkeypatch.setattr(db, "_ADDED_COLUMNS", patched)

    db.init_db(conn)

    names = [r[1] for r in conn.execute("PRAGMA table_info(records)").fetchall()]
    assert "zz_i734_probe" in names


def test_read_only_connection_is_unchanged(conn, vault_path):
    stamp_before = _stamp(conn)
    ro = db.connect(vault_path, read_only=True)
    try:
        assert db.init_db(ro) is None
    finally:
        ro.close()
    assert _stamp(conn) == stamp_before
