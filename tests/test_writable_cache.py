"""The ingest's write transaction does not spill and lock readers out, and
only the ingest's connection gets the cache that buys this.

downstream #734: with the library's default page cache a thousand-record batch
spilled dirty pages mid-transaction, which takes the EXCLUSIVE lock, so plain
readers were refused for the rest of the transaction rather than only during
its commit. A large cache on EVERY writable connection fixed that but is also
filled by plain reads, so a process with many connections open multiplied
it; the cache is therefore opt-in, on the ingest's main connection only.
Journal mode and synchronous are unchanged.
"""
from __future__ import annotations

import hashlib
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, derive, receiver

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
    db.use_ingest_cache(writer)
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


def _default_cache_size(path) -> int:
    plain = sqlite3.connect(path)
    try:
        return plain.execute("PRAGMA cache_size").fetchone()[0]
    finally:
        plain.close()


def test_ordinary_connections_keep_the_library_default_cache(big_vault):
    default = _default_cache_size(big_vault)
    assert default != -db.INGEST_CACHE_KIB
    w = db.connect(big_vault)
    r = db.connect(big_vault, read_only=True)
    try:
        assert w.execute("PRAGMA cache_size").fetchone()[0] == default
        assert r.execute("PRAGMA cache_size").fetchone()[0] == default
        assert w.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert w.execute("PRAGMA synchronous").fetchone()[0] == 1   # NORMAL
    finally:
        w.close()
        r.close()


def test_use_ingest_cache_changes_only_the_cache(big_vault):
    w = db.connect(big_vault)
    try:
        db.use_ingest_cache(w)
        assert w.execute("PRAGMA cache_size").fetchone()[0] == -db.INGEST_CACHE_KIB
        page = w.execute("PRAGMA page_size").fetchone()[0]
        assert (w.execute("PRAGMA cache_spill").fetchone()[0]
                == db.INGEST_CACHE_KIB * 1024 // page)
        assert w.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert w.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        w.close()


def test_only_the_ingest_main_connection_gets_the_ingest_cache(
        vault, conn, monkeypatch):
    """The main transaction's connection has the large cache; the derive
    connection, opened by the same ingest, keeps the default."""
    seen: dict[str, int] = {}
    insert_records, update_after_ingest = db.insert_records, derive.update_after_ingest

    def spy_insert(c, *a, **k):
        seen["main"] = c.execute("PRAGMA cache_size").fetchone()[0]
        return insert_records(c, *a, **k)

    def spy_derive(c, *a, **k):
        seen["derive"] = c.execute("PRAGMA cache_size").fetchone()[0]
        return update_after_ingest(c, *a, **k)

    monkeypatch.setattr(db, "insert_records", spy_insert)
    monkeypatch.setattr(derive, "update_after_ingest", spy_derive)
    monkeypatch.setattr(receiver, "SHARED_SECRET", "s3cret")
    payload = {
        "protocol_version": 1,
        "device": {"id": "dev-a", "name": "iPhone", "model": "iPhone17,1"},
        "app_version": "1.0", "batch_id": "cache-1", "batch_sequence": 1,
        "sent_at": "2026-10-21T13:04:05Z", "anchors": [], "deletions": [],
        "workouts": [],
        "samples": [{
            "kind": "quantity", "hk_uuid": "cache-1-0",
            "type_identifier": "HKQuantityTypeIdentifierHeartRate",
            "start": "2026-10-21T08:00:00-04:00",
            "end": "2026-10-21T08:00:00-04:00", "value": 80.0,
            "unit": "count/min",
            "source_revision": {"source_name": "Apple Watch", "bundle_id": "x"},
        }],
    }
    with TestClient(receiver.create_app(vault)) as client:
        r = client.post("/v1/ingest", json=payload,
                        headers={"x-health-secret": "s3cret"})
    assert r.status_code == 200, r.text
    assert seen["main"] == -db.INGEST_CACHE_KIB
    assert seen["derive"] == _default_cache_size(vault.db_path)


def test_a_sync_sized_batch_leaves_readers_able_to_read(big_vault):
    assert not _reader_refused_mid_transaction(big_vault)


def test_control_the_stock_cache_would_lock_readers_out(big_vault, monkeypatch):
    # The same transaction with a library-default-sized cache proves the test
    # can fail.
    monkeypatch.setattr(db, "INGEST_CACHE_KIB", 2_000)
    assert _reader_refused_mid_transaction(big_vault)
