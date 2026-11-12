"""Lock waits: the owner check, the connect pragmas, a per-request budget.

downstream #734: (1) `_owner_of` read a busy error as "unclaimed", skipping the
ownership check; (2) `db.connect` ran `PRAGMA journal_mode` before setting the
busy timeout, so a writable connect waited Python's default 5 s; (3) a request
that opens several connections could wait the full busy timeout on each.
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from health_advisor import db


@pytest.fixture
def exclusive(vault_path, conn):
    holder = sqlite3.connect(vault_path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    yield
    holder.rollback()
    holder.close()


def test_a_busy_vault_is_not_read_as_unclaimed(vault, exclusive, monkeypatch):
    monkeypatch.setattr(db, "BUSY_TIMEOUT_MS", 200)
    with pytest.raises(sqlite3.OperationalError) as info:
        vault.read_only()
    assert db.is_busy_error(info.value)


def test_a_writable_connect_waits_the_engine_timeout_not_five_seconds(
        vault_path, exclusive, monkeypatch):
    monkeypatch.setattr(db, "BUSY_TIMEOUT_MS", 300)
    started = time.monotonic()
    with pytest.raises(sqlite3.OperationalError):
        db.connect(vault_path)
    # The wait being excluded is Python's default 5 s; 300 ms plus any load.
    assert time.monotonic() - started < 4.0


def test_a_busy_budget_bounds_every_connection_opened_inside_it(
        vault_path, conn, monkeypatch):
    monkeypatch.setattr(db, "BUSY_TIMEOUT_MS", 30_000)
    holder = sqlite3.connect(vault_path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with db.busy_budget(700):
            c = db.connect(vault_path)
            assert c.execute("PRAGMA busy_timeout").fetchone()[0] <= 700
            started = time.monotonic()
            with pytest.raises(sqlite3.OperationalError):
                c.execute("BEGIN IMMEDIATE")
            # Far under the 30 s the budget replaces, whatever the load.
            assert time.monotonic() - started < 10.0
            c.close()
    finally:
        holder.rollback()
        holder.close()


def test_the_budget_has_a_floor_and_is_unset_outside_its_block(vault_path, conn):
    with db.busy_budget(0):
        assert db.busy_timeout_ms() == db.BUSY_BUDGET_FLOOR_MS
    assert db.busy_timeout_ms() == db.BUSY_TIMEOUT_MS
    c = db.connect(vault_path)
    try:
        assert c.execute("PRAGMA busy_timeout").fetchone()[0] == db.BUSY_TIMEOUT_MS
    finally:
        c.close()
