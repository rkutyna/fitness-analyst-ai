"""Single definitions for the helpers that used to be copied across modules (#101).

Each collapsed helper gets two kinds of check: the retired private names are
gone (so a second copy cannot quietly reappear), and the surviving helper gives
the same answer the old copies gave over a table of inputs.
"""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import chat, db, facts


# --- table and column probes -------------------------------------------------

def _old_has_table(conn, name):
    """The body every retired copy shared (db, chat, facts, the two scripts)."""
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,)).fetchone() is not None


@pytest.mark.parametrize("row_factory", [None, sqlite3.Row])
def test_has_table_matches_old_copy(row_factory):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = row_factory
    conn.execute("CREATE TABLE present (a INTEGER)")
    conn.execute("CREATE VIEW only_a_view AS SELECT 1 AS a")
    for name in ["present", "absent", "only_a_view", "PRESENT", ""]:
        assert db.has_table(conn, name) == _old_has_table(conn, name), name
    assert db.has_table(conn, "present") is True
    assert db.has_table(conn, "absent") is False
    assert db.has_table(conn, "only_a_view") is False


@pytest.mark.parametrize("row_factory", [None, sqlite3.Row])
def test_has_column_reads_pragma_by_position(row_factory):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = row_factory
    conn.execute("CREATE TABLE daily_metrics (metric TEXT, source_kind TEXT)")
    assert db.has_column(conn, "daily_metrics", "source_kind") is True
    assert db.has_column(conn, "daily_metrics", "metric") is True
    assert db.has_column(conn, "daily_metrics", "missing") is False
    # A table that does not exist has no columns, so the probe answers False.
    assert db.has_column(conn, "no_such_table", "source_kind") is False


def test_retired_table_probes_have_no_definition():
    import scripts.build_source_provenance as provenance
    import scripts.verify_daily_metrics as verify
    assert not hasattr(db, "_has_table")
    assert not hasattr(chat, "_has_table")
    assert not hasattr(facts, "_has_table")
    assert not hasattr(provenance, "_has_table")
    assert not hasattr(verify, "_has_table")
    assert not hasattr(verify, "_has_column")
