"""`init_db` fills `metric_source_months` when a migration creates it.

A vault that predates the table gains an EMPTY `metric_source_months` on its
first `init_db`, while its raw `records` already span years. The receiver then
maintains only the months each batch touches, so instrument eras vanish. The
fix rebuilds the table once, in the call that creates it, on a vault that is
not compacted.
"""
from __future__ import annotations

import pytest

from health_advisor import db as dbmod
from health_advisor import vault as V

# Hand-computed oracle: (metric, month, source, n). Each n is the number of
# records seeded for that combination in `_seed_records` below.
EXPECTED = [
    ("heart_rate", "2026-01", "Phone", 1),
    ("heart_rate", "2026-01", "Watch", 3),
    ("heart_rate", "2026-02", "Phone", 2),
    ("heart_rate", "2026-02", "Watch", 2),
    ("heart_rate", "2026-03", "Phone", 3),
    ("heart_rate", "2026-03", "Watch", 1),
    ("step_count", "2026-01", "Phone", 2),
    ("step_count", "2026-01", "Watch", 1),
    ("step_count", "2026-02", "Phone", 1),
    ("step_count", "2026-02", "Watch", 4),
    ("step_count", "2026-03", "Phone", 2),
    ("step_count", "2026-03", "Watch", 2),
]


def _seed_records(conn):
    """Insert raw records so that per-month counts match EXPECTED exactly."""
    k = 0
    for metric, month, source, n in EXPECTED:
        for i in range(1, n + 1):
            k += 1
            day = f"{month}-{i:02d}"
            ts = f"{day}T12:00:00+00:00"
            conn.execute(
                "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
                "start_local, local_date, source, origin, dedupe_key) "
                "VALUES (?, 1.0, 'count', ?, ?, ?, ?, ?, 'test', ?)",
                (metric, ts, ts, ts, day, source, f"{metric}|{ts}|{source}|{k}"))
    conn.commit()


def _msm_rows(conn):
    return sorted(tuple(r) for r in conn.execute(
        "SELECT metric, month, source, n FROM metric_source_months"))


def test_dropped_table_is_rebuilt_in_full_by_init_db(conn):
    _seed_records(conn)
    conn.execute("DROP TABLE metric_source_months")
    conn.commit()

    dbmod.init_db(conn)

    assert _msm_rows(conn) == sorted(EXPECTED)
    assert len(_msm_rows(conn)) == 12


def test_partial_existing_table_is_not_rebuilt_by_init_db(conn):
    _seed_records(conn)
    dbmod.rebuild_metric_source_months(conn, full=True)
    assert len(_msm_rows(conn)) == 12
    # Make the table partial: drop every row of one month, leaving it present.
    conn.execute("DELETE FROM metric_source_months WHERE month = '2026-02'")
    conn.commit()
    before = _msm_rows(conn)
    assert len(before) == 8

    dbmod.init_db(conn)

    assert _msm_rows(conn) == before
    assert not any(r[1] == "2026-02" for r in _msm_rows(conn))


def test_empty_vault_gets_table_with_no_rows_and_no_error(conn):
    conn.execute("DROP TABLE metric_source_months")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0

    dbmod.init_db(conn)

    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='metric_source_months'").fetchone() is not None
    assert conn.execute(
        "SELECT COUNT(*) FROM metric_source_months").fetchone()[0] == 0


def test_compacted_vault_is_not_rebuilt_by_init_db(conn, monkeypatch):
    _seed_records(conn)
    # The compaction watermark, as vault.compact() writes it (vault.py ~831).
    conn.execute(
        "INSERT OR REPLACE INTO vault_meta (key, value) "
        "VALUES ('compacted_through', '2026-01-31')")
    conn.execute("DROP TABLE metric_source_months")
    conn.commit()
    assert V.compacted_through(conn) == "2026-01-31"

    def _refuse(*args, **kwargs):
        raise AssertionError("rebuild must not run on a compacted vault")

    monkeypatch.setattr(dbmod, "rebuild_metric_source_months", _refuse)

    dbmod.init_db(conn)

    assert conn.execute(
        "SELECT COUNT(*) FROM metric_source_months").fetchone()[0] == 0


def test_old_vault_with_unnamed_source_rows_migrates_without_error(tmp_path):
    """A records row with no source must not stop init_db: the rebuild counts
    it under the empty source name, the convention `NULLIF(source, '')` uses
    for "no source" elsewhere in the schema."""
    path = tmp_path / "old.db"
    conn = dbmod.connect(path)
    conn.executescript("""
        CREATE TABLE records (
            id INTEGER PRIMARY KEY, metric TEXT NOT NULL, value REAL,
            unit TEXT, start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
            start_local TEXT, local_date TEXT NOT NULL, source TEXT,
            origin TEXT NOT NULL DEFAULT 'backfill',
            dedupe_key TEXT NOT NULL UNIQUE
        );
    """)
    conn.executemany(
        "INSERT INTO records (metric, value, start_utc, end_utc, local_date, "
        "source, dedupe_key) VALUES (?, 1, ?, ?, ?, ?, ?)",
        [("step_count", "2026-08-20T00:00:00Z", "2026-08-20T00:01:00Z",
          "2026-08-20", None, "a"),
         ("step_count", "2026-08-21T00:00:00Z", "2026-08-21T00:01:00Z",
          "2026-08-21", None, "b"),
         ("step_count", "2026-08-21T00:02:00Z", "2026-08-21T00:03:00Z",
          "2026-08-21", "Watch", "c")])
    conn.commit()
    dbmod.init_db(conn)
    rows = sorted(tuple(r) for r in conn.execute(
        "SELECT metric, month, source, n FROM metric_source_months"))
    assert rows == [("step_count", "2026-08", "", 2),
                    ("step_count", "2026-08", "Watch", 1)]
    conn.close()
