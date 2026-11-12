"""A sync-sized write transaction does not spill and lock readers out.

downstream #734: with the library's default page cache a ~1,000-record batch
spilled dirty pages mid-transaction, which takes the EXCLUSIVE lock, so plain
readers were refused for the rest of the transaction rather than only during
its commit. Journal mode and synchronous are unchanged.
"""
from __future__ import annotations

import hashlib
import sqlite3
import uuid

import pytest

from health_advisor import db


def _fill(conn, n, offset=0):
    conn.executemany(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, dedupe_key, hk_uuid) VALUES "
        "('heart_rate', 1, 'u', ?, ?, '2026-10-01', 's', 'test', ?, ?)",
        [(f"t{offset + i}", f"t{offset + i}", f"k{offset + i}",
          str(uuid.UUID(bytes=hashlib.md5(str(offset + i).encode()).digest())))
         for i in range(n)])


def _reader_refused_mid_transaction(path) -> bool:
    writer = db.connect(path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        _fill(writer, 2_000, offset=10_000_000)   # a sync-sized batch
        reader = sqlite3.connect(path, timeout=0)
        try:
            reader.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()
            return False
        except sqlite3.OperationalError as exc:
            assert "locked" in str(exc)
            return True
        finally:
            reader.close()
            writer.rollback()
    finally:
        writer.close()


@pytest.fixture(scope="module")
def big_vault(tmp_path_factory):
    path = tmp_path_factory.mktemp("cache") / "v.db"
    conn = db.connect(path)
    db.init_db(conn)
    _fill(conn, 60_000)   # random UUIDs: a batch dirties many index pages
    conn.commit()
    conn.close()
    return path


def test_writable_connections_get_the_larger_cache_and_readers_do_not(big_vault):
    w = db.connect(big_vault)
    r = db.connect(big_vault, read_only=True)
    try:
        assert w.execute("PRAGMA cache_size").fetchone()[0] == -db.WRITABLE_CACHE_KIB
        assert r.execute("PRAGMA cache_size").fetchone()[0] != -db.WRITABLE_CACHE_KIB
        assert w.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        w.close()
        r.close()


def test_a_sync_sized_batch_leaves_readers_able_to_read(big_vault):
    assert not _reader_refused_mid_transaction(big_vault)


def test_control_the_stock_cache_would_lock_readers_out(big_vault, monkeypatch):
    # The same transaction with the library default proves the test can fail.
    monkeypatch.setattr(db, "WRITABLE_CACHE_KIB", 2_000)
    assert _reader_refused_mid_transaction(big_vault)
