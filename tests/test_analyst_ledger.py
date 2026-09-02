"""analyst_ledger -- the authorizer-backed ledgered connection.

Exercises the ledgered `conn` analyst-mode code would receive, entirely
in-process (no sandbox needed for any of this -- the ledger is a property of
the sqlite3 connection, not of the process it runs in).

Covers A2 `Done when` 1 and 2 from
the analyst-mode design, restated in SECURITY.md:

  1. the zero-read refusal: code that emits numeric tables without ever
     touching `conn` is refused, with the exact reason string, and the
     ledger it was refused against reports zero on every count;
  2. ledger accuracy: for >=10 known queries, the authorizer-recorded
     `tables_read` set matches the expected set exactly -- including one
     subquery and one view (S9.2's evidence that this is the database
     reporting, not statement-text parsing) -- plus an ATTACH attempt,
     asserted denied.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from health_advisor import analyst_envelope as env
from health_advisor import analyst_ledger as led


def _build_probe_db(path) -> None:
    """A small, custom schema -- deliberately not the project schema, since
    this module is schema-agnostic and the interesting cases (a view, a
    subquery, a join, a CTE) are cheaper to set up directly."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE t (a INTEGER, b INTEGER);
        INSERT INTO t VALUES (1, 10), (2, 20), (3, 30);
        CREATE TABLE t2 (x INTEGER, y INTEGER);
        INSERT INTO t2 VALUES (1, 100), (2, 200);
        CREATE VIEW v AS SELECT a FROM t;
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def probe_db_path(tmp_path):
    path = tmp_path / "probe.db"
    _build_probe_db(path)
    return path


# --------------------------------------------------------------------------- #
# Done when 1 -- the zero-read refusal
# --------------------------------------------------------------------------- #

def test_zero_read_refusal_fires_with_exact_reason(probe_db_path):
    """Hostile code that emits numeric tables while never touching `conn`."""
    conn = led.open_ledgered(probe_db_path)
    try:
        # Deliberately: no conn.execute(...) call anywhere on this path.
        ledger_before = conn.ledger
        assert ledger_before.query_count == 0
        assert ledger_before.rows_read == 0
        assert ledger_before.tables_read == ()
        assert ledger_before.columns_read == ()

        # The hostile payload: two numeric tables, fabricated from nothing.
        payload = {
            "tables": [
                {"name": "t1", "columns": ["v"], "units": ["count"],
                 "rows": [[1]]},
                {"name": "t2", "columns": ["v"], "units": ["count"],
                 "rows": [[2]]},
            ]
        }
        raw = json.dumps(payload).encode("utf-8")

        result = env.validate(
            raw,
            run_id="run-zero-read",
            question="how many steps yesterday?",
            code_sha256="c" * 64,
            vault_sha256="v" * 64,
            vault_version=1,
            ledger=conn.ledger.as_dict(),
        )
    finally:
        conn.close()

    assert isinstance(result, env.Refusal)
    assert result.reason == ("emitted 2 numeric tables from 0 vault tables "
                             "and 0 reads")

    # And the ledger this was refused against is zero on every count named in
    # the spec's `Done when` 1: query_count and rows_read.
    assert ledger_before.query_count == 0
    assert ledger_before.rows_read == 0


def test_zero_read_refusal_does_not_fire_once_the_vault_was_read(probe_db_path):
    """The mirror case: a run that DID read the vault is not refused by this
    gate (it may of course still be refused by unrelated grammar checks)."""
    conn = led.open_ledgered(probe_db_path)
    try:
        conn.execute("SELECT a FROM t").fetchall()
        payload = {"tables": [{"name": "t1", "columns": ["v"],
                                "units": ["count"], "rows": [[1]]}]}
        raw = json.dumps(payload).encode("utf-8")
        result = env.validate(
            raw, run_id="run-real-read", question="q",
            code_sha256="c" * 64, vault_sha256="v" * 64, vault_version=1,
            ledger=conn.ledger.as_dict(),
        )
    finally:
        conn.close()

    assert isinstance(result, env.Envelope)


# --------------------------------------------------------------------------- #
# Done when 2 -- ledger accuracy, >=10 known queries, 10/10
# --------------------------------------------------------------------------- #

# (label, sql, expected tables_read). Every one of these is the database
# reporting what it actually touched -- not a guess from the SQL text, which
# is exactly what the view and subquery cases are here to demonstrate: `v`'s
# query mentions only `v` in its FROM clause, yet the authorizer also reports
# the base table `t` it expands to.
KNOWN_QUERIES = [
    ("plain_select", "SELECT a FROM t", frozenset({"t"})),
    ("plain_select_where", "SELECT a, b FROM t WHERE b > 15", frozenset({"t"})),
    ("other_table", "SELECT x FROM t2", frozenset({"t2"})),
    ("join", "SELECT t.a, t2.y FROM t JOIN t2 ON t.a = t2.x", frozenset({"t", "t2"})),
    ("view", "SELECT a FROM v", frozenset({"t", "v"})),
    ("scalar_subquery", "SELECT a FROM t WHERE a = (SELECT MAX(a) FROM t)",
     frozenset({"t"})),
    ("cte", "WITH cte AS (SELECT a FROM t) SELECT a FROM cte", frozenset({"t"})),
    ("count_star", "SELECT COUNT(*) FROM t", frozenset({"t"})),
    ("select_one_limit", "SELECT 1 FROM t LIMIT 1", frozenset({"t"})),
    ("rowid_subquery",
     "SELECT a FROM t WHERE rowid IN (SELECT rowid FROM t WHERE b > 10)",
     frozenset({"t"})),
]


@pytest.mark.parametrize("label,sql,expected", KNOWN_QUERIES,
                         ids=[c[0] for c in KNOWN_QUERIES])
def test_ledger_accuracy(probe_db_path, label, sql, expected):
    conn = led.open_ledgered(probe_db_path)
    try:
        conn.execute(sql).fetchall()
        observed = set(conn.ledger.tables_read)
    finally:
        conn.close()
    assert observed == expected, f"{label}: {sql!r} -> {observed} != {expected}"


def test_ledger_accuracy_matches_expected_for_every_known_query(probe_db_path):
    """The Done-when-2 number, computed directly rather than left to pytest's
    per-case pass/fail count."""
    matches = 0
    for _label, sql, expected in KNOWN_QUERIES:
        conn = led.open_ledgered(probe_db_path)
        try:
            conn.execute(sql).fetchall()
            if set(conn.ledger.tables_read) == expected:
                matches += 1
        finally:
            conn.close()
    assert matches == len(KNOWN_QUERIES) == 10


def test_attach_is_denied_outright(probe_db_path, tmp_path):
    """S9.2: SQLITE_ATTACH denial raises `sqlite3.DatabaseError: not
    authorized`, and this connection refuses it rather than merely recording
    it -- the vault's authorizer is the boundary, not just the ledger."""
    other_db = tmp_path / "other.db"
    other_conn = sqlite3.connect(other_db)
    other_conn.execute("CREATE TABLE ext (z INTEGER)")
    other_conn.execute("INSERT INTO ext VALUES (99)")
    other_conn.commit()
    other_conn.close()

    conn = led.open_ledgered(probe_db_path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            conn.execute(f"ATTACH DATABASE '{other_db}' AS ext_db")
        # A denied ATTACH is not a successful query: it must not inflate the
        # ledger with a read that never happened.
        assert conn.ledger.query_count == 0
        assert "ext" not in conn.ledger.tables_read
    finally:
        conn.close()


def test_pragma_is_denied_outright(probe_db_path):
    """S9.2: PRAGMAs surface as SQLITE_PRAGMA and are refused the same way
    ATTACH is -- e.g. a hostile `PRAGMA journal_mode=WAL` attempting to make
    the read-only mount misbehave (S3.2's fourth reason the vault stays
    unwritable)."""
    conn = led.open_ledgered(probe_db_path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            conn.execute("PRAGMA table_info(t)")
        assert conn.ledger.query_count == 0
    finally:
        conn.close()


def test_null_column_and_empty_column_events_do_not_crash(probe_db_path):
    """S9.2: a rowid/aggregate-shaped read can emit an authorizer event whose
    column is falsy (measured while building this module: `SELECT 1 FROM t
    LIMIT 1` reports `('t', '', None)` -- table is real, column is an empty
    string, dbname is None). The ledger must tolerate this rather than
    raising or mis-recording an empty-string column as a real one."""
    conn = led.open_ledgered(probe_db_path)
    try:
        conn.execute("SELECT 1 FROM t LIMIT 1").fetchall()
        assert conn.ledger.tables_read == ("t",)
        # No column made it into columns_read: '' is falsy and is not a real
        # column name that should show up next to 't'.
        assert conn.ledger.columns_read == ()
    finally:
        conn.close()


def test_rows_read_counts_via_fetchall(probe_db_path):
    conn = led.open_ledgered(probe_db_path)
    try:
        rows = conn.execute("SELECT a FROM t").fetchall()
        assert len(rows) == 3
        assert conn.ledger.rows_read == 3
        assert conn.ledger.query_count == 1
    finally:
        conn.close()


def test_rows_read_counts_via_iteration(probe_db_path):
    conn = led.open_ledgered(probe_db_path)
    try:
        seen = 0
        for _row in conn.execute("SELECT a FROM t"):
            seen += 1
        assert seen == 3
        assert conn.ledger.rows_read == 3
    finally:
        conn.close()


def test_cursor_style_execute_is_also_counted(probe_db_path):
    """`conn.cursor().execute(...)` must be ledgered exactly like
    `conn.execute(...)` -- a caller should not be able to dodge the ledger by
    choosing the other equivalent API."""
    conn = led.open_ledgered(probe_db_path)
    try:
        cur = conn.cursor()
        rows = cur.execute("SELECT a FROM t").fetchall()
        assert len(rows) == 3
        assert conn.ledger.query_count == 1
        assert conn.ledger.rows_read == 3
        assert conn.ledger.tables_read == ("t",)
    finally:
        conn.close()


def test_open_ledgered_does_not_leak_the_vault_path(probe_db_path):
    """S3.1: 'the generated code never receives a vault path'. This wrapper
    should add no attribute that hands the path back out."""
    conn = led.open_ledgered(probe_db_path)
    try:
        for attr in ("path", "db_path", "vault_path", "_path", "name"):
            assert not hasattr(conn, attr), (
                f"LedgeredConnection exposes {attr!r}, which would leak the "
                "vault path back to analyst code")
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Regression: reading SQLite's own catalog must not satisfy the zero-read gate
# --------------------------------------------------------------------------- #

def test_catalog_read_is_allowed_but_cannot_satisfy_the_zero_read_gate(
        probe_db_path):
    """The bypass this gate exists to stop, found 2026-08-30 -- and the
    over-correction an adversarial review then measured.

    Before the fix, `SELECT 1 FROM sqlite_master LIMIT 1` produced
    query_count=1 and rows_read=1 while touching nothing about the athlete --
    enough for fabricated numbers to pass a gate whose entire purpose is to
    prove the vault was consulted.

    The first fix DENIED catalog reads. That broke legitimate SQL that a plain
    SELECT does not exercise: CREATE TEMP TABLE and ANALYZE both fail on the
    catalog. So the catalog is READ-able and merely excluded from
    `tables_read`, which is what the gate reads.
    """
    conn = led.open_ledgered(probe_db_path)
    try:
        # Allowed -- no exception.
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
        # A temp table is a legitimate analysis shape and must still work.
        conn.execute("CREATE TEMP TABLE scratch AS SELECT a FROM t")

        ledger = conn.ledger.as_dict()
        # The catalog is NOT a vault table, so it cannot satisfy the gate.
        assert "sqlite_master" not in ledger["tables_read"]
    finally:
        conn.close()


def test_zero_read_gate_refuses_when_no_vault_table_was_read():
    """Defence in depth: the gate itself rejects a ledger that reports reads
    but names no vault table, independent of the authorizer's denial."""
    result = env.validate(
        json.dumps({"tables": [
            {"name": "t1", "columns": ["v"], "units": ["count"],
             "rows": [[1]]}]}).encode("utf-8"),
        run_id="run-catalog-only",
        question="how many steps yesterday?",
        code_sha256="c" * 64,
        vault_sha256="v" * 64,
        vault_version=1,
        ledger={"query_count": 1, "rows_read": 1,
                "tables_read": [], "columns_read": []},
    )
    assert isinstance(result, env.Refusal)
    assert "0 vault tables" in result.reason


def test_denial_record_is_bounded_so_a_refused_loop_cannot_grow_the_parent(
        probe_db_path):
    """A denied action costs the child nothing and used to cost the parent a
    list entry for ever (#232's shape, found on the corpus side first).

    Measured before the bound: 20,000 denied PRAGMAs accumulated 1.29 MB in
    the parent in under a second, and the child can spend its whole wall-clock
    budget doing it. The growth is parent-side, so no sandbox profile sees it.
    """
    conn = led.open_ledgered(probe_db_path)
    try:
        attempts = led.MAX_RETAINED_DENIALS * 5
        for _ in range(attempts):
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute("PRAGMA page_size")
        state = conn._state
        # Retention is capped ...
        assert len(state.denied) == led.MAX_RETAINED_DENIALS
        # ... while the count that matters stays exact.
        assert state.denied_count == attempts
    finally:
        conn.close()


def test_denial_entries_cannot_carry_unbounded_child_text():
    """Bounding the NUMBER of records does nothing while a single record's
    SIZE is child-controlled.

    `arg1` is the ATTACH filename or the PRAGMA name -- both written by the
    analyst's own code. Measured before this bound: 100 retained attempts, each
    carrying a 1 MB path, held 100,000,000 bytes in the parent while sitting
    comfortably inside the entry-count cap. Two different bounds, both needed.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a INTEGER)")
    wrapped = led.wrap_connection(conn)
    big = "A" * 100_000
    for _ in range(led.MAX_RETAINED_DENIALS + 10):
        with pytest.raises(sqlite3.DatabaseError):
            wrapped.execute(f"ATTACH DATABASE '{big}' AS x")

    state = wrapped._state
    assert len(state.denied) == led.MAX_RETAINED_DENIALS
    assert state.denied_count == led.MAX_RETAINED_DENIALS + 10
    retained = sum(len(arg or "") for _action, arg in state.denied)
    # Every entry is bounded, so the whole record is bounded by construction.
    per_entry_max = led.MAX_RETAINED_DENIAL_ARG + len("...[truncated]")
    assert all(len(arg or "") <= per_entry_max for _a, arg in state.denied)
    assert retained <= led.MAX_RETAINED_DENIALS * per_entry_max
    # And the truncation is never silent.
    assert any((arg or "").endswith("...[truncated]") for _a, arg in state.denied)
