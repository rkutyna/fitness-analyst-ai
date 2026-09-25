from __future__ import annotations

import pytest

from health_advisor import db
from health_advisor import vault as V


def test_declare_empty_vault_on_fresh_schema_makes_is_vault_true(conn):
    """A schema-initialized-but-empty vault (what `claim()` produces) is a
    legitimate D3 vault once declared this way — consumer #521."""
    assert V.is_vault(conn) is False

    V.declare_empty_vault(conn)

    assert V.is_vault(conn) is True


def test_declare_empty_vault_lets_compaction_run_without_the_declared_vault_error(
    vault_path, conn
):
    """The engine-level half of #521's Done-when #1: `run_compaction` on a
    bootstrap-created empty vault must not raise
    ValueError("compaction requires a declared vault")."""
    V.declare_empty_vault(conn)
    conn.commit()
    conn.close()

    report = V.run_compaction(vault_path)

    # No raw device rows were ever ingested, so there is nothing to compact —
    # but critically, no ValueError was raised getting here.
    assert report["status"] == "no_raw_rows"


def test_declare_empty_vault_refuses_a_vault_with_a_records_row(conn):
    """Done-when #2: refuses when `records` (the table D3/compact govern) is
    non-empty."""
    assert db.insert_records(conn, [{
        "metric": "basal_energy",
        "value": 1.0,
        "unit": "kcal",
        "start_utc": "2026-07-22T12:00:00+00:00",
        "end_utc": "2026-07-22T12:00:00+00:00",
        "start_local": "2026-07-22 08:00:00",
        "local_date": "2026-07-22",
        "source": "test",
        "origin": "receiver",
        "dedupe_key": "not-empty",
    }]) == 1

    with pytest.raises(ValueError, match="not empty"):
        V.declare_empty_vault(conn)

    assert V.is_vault(conn) is False


def test_declare_empty_vault_refuses_on_other_raw_tables_too(conn):
    """`records` is not the only table `build_vault` copies. A vault that
    already carries, say, a `daily_metrics` row — never having gone through
    `build_vault`'s filter — is not empty either, even though `compact()`
    itself never touches `daily_metrics`."""
    conn.execute(
        "INSERT INTO daily_metrics (metric, date, count) "
        "VALUES ('step_count', '2026-07-22', 1)"
    )
    conn.commit()

    with pytest.raises(ValueError, match="daily_metrics"):
        V.declare_empty_vault(conn)


def test_declare_empty_vault_is_idempotent_while_still_empty(conn):
    """Calling it twice on a vault nothing has since written to must not
    raise, and must leave it declared."""
    V.declare_empty_vault(conn)
    V.declare_empty_vault(conn)

    assert V.is_vault(conn) is True
