"""A table is declared in exactly one place.

#124. `_ADDED_TABLES` in db.py and schema.sql could both declare a table, and
`_apply_table_migrations` runs at the top of `init_db` — before the
`executescript(schema.sql)` below it. Both declarations are CREATE TABLE IF NOT
EXISTS, so the db.py copy always won and schema.sql's became a dead no-op, on
fresh vaults as well as existing ones.

Measured 2026-08-26: adding a sentinel column to db.py's `plan_projections`
copy alone put that column on a freshly created vault. Editing the schema.sql
copy alone had no effect on any vault, ever, and nothing was checking.

`_ADDED_TABLES` is now empty — `plan_projections` is declared only in
schema.sql. These tests keep it that way and pin the reasoning that made
emptying it safe.
"""

import sqlite3

import pytest

from health_advisor import db

def test_no_table_is_declared_in_both_places():
    """The invariant #124 exists to protect.

    A key here that is also in schema.sql silently wins over the file a reader
    would go to. If this fails, delete the db.py copy — not the schema.sql one.
    """
    schema = db.SCHEMA_PATH.read_text().lower()
    both = [t for t in db._ADDED_TABLES
            if f"create table if not exists {t.lower()}" in schema
            or f"create table {t.lower()}" in schema]
    assert not both, (
        f"{both} are declared in BOTH db.py's _ADDED_TABLES and schema.sql. "
        f"db.py's copy runs first and wins; schema.sql's is a dead no-op."
    )


def test_added_tables_is_empty_so_schema_sql_is_the_single_declaration():
    """Documents the current end state, and fails loudly if a key comes back.

    Not a style assertion: a new key is only correct for a table that must exist
    before `_apply_column_migrations` runs, and that is rare enough to be worth
    re-reading db.py's comment for. Deleting this test is the right move when
    such a table genuinely arrives.
    """
    assert db._ADDED_TABLES == {}, (
        "a table was added to _ADDED_TABLES — it must NOT also be in schema.sql "
        "(see test_no_table_is_declared_in_both_places) and db.py's comment "
        "explains the only case that warrants it"
    )


def test_schema_sql_alone_brings_an_existing_vault_up_to_date(tmp_path):
    """The reasoning that made emptying _ADDED_TABLES safe.

    `executescript(schema.sql)` is unconditional in init_db and every table
    there is IF NOT EXISTS, so a vault that predates a table gains it with no
    migration entry at all. Modelled by building a real vault and dropping the
    projection tables back off it — a hand-written "legacy" schema woulddiffer from
    the real one and test the fixture rather than the code.

    If this fails, executescript stopped being unconditional and the comment in
    db.py needs correcting again.
    """
    path = tmp_path / "vault.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    try:
        conn.execute("DROP TABLE plan_projections")
        conn.commit()
        before = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "plan_projections" not in before

        db.init_db(conn)
        after = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()

    assert "plan_projections" in after, (
        "plan_projections did not come back from schema.sql alone, so removing "
        "db.py's copy was not safe after all"
    )
    assert after > before


def test_plan_projections_keeps_its_provenance_check(tmp_path):
    """The declaration that survived is the one with the constraints.

    A projection row carries either a conversation_turn_id or a parsed
    file/line, never both and never neither (G-07). That CHECK lived in both
    copies; it must still be enforced now that only one remains.
    """
    path = tmp_path / "vault.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO plan_projections (projection_id, week_start, "
                "payload_json, schema_version, created_at) "
                "VALUES ('p1', '2026-08-17', '{}', 1, '2026-08-26T00:00:00Z')")
    finally:
        conn.close()
