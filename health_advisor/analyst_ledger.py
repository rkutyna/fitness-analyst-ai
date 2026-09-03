"""analyst_ledger.py -- an authorizer-backed, ledgered read-only connection.

Analyst mode is the mode where the model writes the Python
(the analyst-mode design, restated in SECURITY.md). The ``conn`` handed to
that code must let a run's provenance be established from something the model
cannot spoof: SQLite's own ``set_authorizer`` callback fires per access with
``(action, arg1, arg2, dbname, trigger)``, and for ``SQLITE_READ`` that is
``(table, column)`` -- the database reporting what was actually touched, not a
regex guessing from statement text (S1.2). That closes the "which tables were
read" half of provenance. It does NOT prove the emitted number came *from*
those reads (S1.3 -- open, and explicitly not this module's job to close: a
program can run ``SELECT 1 FROM daily_metrics LIMIT 1`` and still emit a
constant. The ledger raises the cost of fabrication; it does not eliminate it).

This module reuses the SHAPE of the project's existing call ledgers
(``deepdive_mcp._CallLedger``/``_ledger_wrapper``, ``llm._ledgered``) -- an
append-only record fed by the real call, never by anything the wrapped code
claims about itself -- but the source of truth here is SQLite's authorizer
rather than a wrapped Python callable, because analyst code issues arbitrary
SQL rather than calling named tools, so there is no call boundary to wrap.

Measured on this venv's Python (sqlite3 3.51.0) the night of 2026-08-29,
recorded in the proposal's S9.2 and reconfirmed here while building this
module (not re-derived as a security claim -- used as given):

  * a view read reports BOTH layers: the view name and the base table, each
    as its own SQLITE_READ event;
  * a CTE read reports only the base table, never the CTE's own name;
  * a rowid/aggregate subquery can emit an event whose ``column`` is an empty
    string and whose ``dbname`` is None (``('t', '', None)`` for
    ``SELECT 1 FROM t LIMIT 1``) -- this module tolerates a falsy column and a
    None dbname rather than assuming either is always populated, or it would
    crash on ordinary SQL;
  * ``SQLITE_ATTACH`` (action 24) and ``SQLITE_PRAGMA`` (action 19) are
    refused outright here by returning ``SQLITE_DENY``, rather than merely
    recorded -- confirmed to raise ``sqlite3.DatabaseError: not authorized``.
    Because ATTACH is refused, this ledger never needs to disambiguate a
    cross-database read attributed to an attached db's own name (the
    "laundered into main" concern in S1.2/S9.2): that path cannot be reached.
"""
from __future__ import annotations

import sqlite3
import weakref
from dataclasses import dataclass, field

from . import db as dbmod

__all__ = [
    "LedgeredConnection",
    "LedgerSummary",
    "open_ledgered",
    "wrap_connection",
]


# Actions this connection refuses outright rather than merely recording
# (analyst-mode-proposal.md S9.2: "the sandbox connection refuses ATTACH and
# PRAGMA outright rather than merely logging them"). Persistent write actions
# are also refused so `wrap_connection()` cannot preserve write access from a
# caller that opened its input connection in writable mode.
_DENIED_ACTIONS = frozenset({sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_PRAGMA})

# `wrap_connection()` also accepts fixture connections that were opened
# writable.  SQLite's authorizer is the narrowest way to make those wrappers
# read-only while retaining the useful ability to create temporary tables.
# Persistent-database write actions are denied below; actions against the
# connection-local `temp` database remain available for scratch work.
_WRITE_ACTION_NAMES = (
    "SQLITE_ALTER_TABLE", "SQLITE_CREATE_INDEX", "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VTABLE", "SQLITE_CREATE_VIEW",
    "SQLITE_DELETE", "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE",
    "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VTABLE", "SQLITE_DROP_VIEW",
    "SQLITE_INSERT", "SQLITE_REINDEX", "SQLITE_UPDATE",
)
_DENIED_WRITE_ACTIONS = frozenset(
    action for name in _WRITE_ACTION_NAMES
    if (action := getattr(sqlite3, name, None)) is not None
)

# SQLite's own catalog is readable and tells the analyst nothing about the
# athlete. Reading it must therefore NOT satisfy the zero-read gate: measured
# 2026-08-30, `SELECT 1 FROM sqlite_master LIMIT 1` produced query_count=1 and
# rows_read=1, which was enough to let fabricated numbers through a gate whose
# whole purpose is to prove the vault was consulted. Catalog reads are recorded
# separately so provenance keeps them while the gate cannot be satisfied by
# them.
_CATALOG_PREFIX = "sqlite_"

# How many denial events are kept for diagnostics. Everything beyond this is
# counted, not stored -- see the comment on `_LedgerState.denied`.
MAX_RETAINED_DENIALS = 100

# And how much of each event's argument is kept. Bounding the NUMBER of records
# does nothing while a single record's SIZE is child-controlled, which is the
# hole the first version of this cap left wide open: the argument recorded for
# SQLITE_ATTACH is the *filename* and for SQLITE_PRAGMA the pragma name, both
# written by the analyst's own code. Measured 2026-08-30, after the corpus
# track hit the identical shape in its own refusal record and said to check
# here: 100 retained `ATTACH DATABASE '<1 MB string>'` attempts held
# **100,000,000 bytes** in the parent, comfortably inside a cap that counted
# only entries. The count bound and the size bound are two different bounds and
# both are required.
MAX_RETAINED_DENIAL_ARG = 120


# Real SQLite objects live only in these parent-side registries.  The wrapper
# instances contain no connection/cursor reference, so inspecting an object
# received across a trust boundary cannot recover the object underneath it.
# Weak keys also avoid retaining a closed wrapper solely because its backing
# connection's authorizer callback is still installed.
_CONNECTIONS = weakref.WeakKeyDictionary()
_CURSORS = weakref.WeakKeyDictionary()
_AUTHORIZERS = weakref.WeakKeyDictionary()


def _connection_for(wrapper: "LedgeredConnection") -> sqlite3.Connection:
    try:
        return _CONNECTIONS[wrapper]
    except KeyError as exc:
        raise sqlite3.ProgrammingError("ledgered connection is closed") from exc


def _cursor_for(wrapper):
    try:
        return _CURSORS[wrapper]
    except KeyError as exc:
        raise sqlite3.ProgrammingError("ledgered cursor is closed") from exc


def _forget_cursor(wrapper, real_cursor=None):
    if real_cursor is None:
        real_cursor = _CURSORS.pop(wrapper, None)
    else:
        _CURSORS.pop(wrapper, None)
    if real_cursor is not None:
        real_cursor.close()


class _ParentMetadataCursor:
    """The tiny result surface needed by the unchanged parent runner."""

    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _ParentMetadataConnection:
    """A non-SQLite compatibility handle for the runner's metadata probe.

    The runner is deliberately out of scope for this change and still asks
    its connection for ``_conn`` while reading parent-owned metadata.  Keep
    that private protocol working without returning the real connection:
    this handle accepts only the two fixed metadata PRAGMAs and its
    ``set_authorizer`` method cannot alter the real connection's authorizer.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: "LedgeredConnection"):
        self._owner = owner

    def set_authorizer(self, callback) -> None:
        # The method exists only for the runner's unchanged private protocol.
        # It intentionally cannot disable the authorizer on the real object.
        return None

    def execute(self, sql: str, params=()) -> _ParentMetadataCursor:
        if params or sql not in ("PRAGMA temp_store = MEMORY",
                                 "PRAGMA user_version"):
            raise sqlite3.ProgrammingError(
                "parent metadata handle does not execute arbitrary SQL")
        owner = self._owner
        real_conn = _connection_for(owner)
        authorizer = _AUTHORIZERS[owner]
        real_conn.set_authorizer(None)
        try:
            row = real_conn.execute(sql).fetchone()
        finally:
            real_conn.set_authorizer(authorizer)
        return _ParentMetadataCursor(row)


def _is_catalog(table: str) -> bool:
    return table.lower().startswith(_CATALOG_PREFIX)


@dataclass
class _LedgerState:
    """Mutable accumulator fed only by sqlite3's own authorizer and cursors --
    never by anything the wrapped analyst code claims about itself."""

    query_count: int = 0
    rows_read: int = 0
    tables_read: set[str] = field(default_factory=set)
    catalog_tables_read: set[str] = field(default_factory=set)
    columns_read: set[tuple[str, str]] = field(default_factory=set)
    # Bounded on purpose. A denied action costs the CHILD nothing -- the
    # authorizer refuses before any work happens -- while the parent paid one
    # list entry per attempt, for ever. Measured 2026-08-30: 20,000 denied
    # PRAGMAs accumulated 1.29 MB here in under a second, and the child can
    # spend its whole wall-clock budget doing it. The growth is in the PARENT,
    # so no sandbox profile can see it; `seatbelt bounds syscalls, not cycles`
    # arriving as a concrete instance. Found by running a loop, after the
    # corpus track hit the same shape in its own refusal record (#232).
    #
    # The retained sample is for diagnostics; `denied_count` is the number that
    # actually matters and an int costs nothing to keep exactly.
    denied: list[tuple[int, str | None]] = field(default_factory=list)
    denied_count: int = 0

    def summary(self) -> "LedgerSummary":
        return LedgerSummary(
            query_count=self.query_count,
            rows_read=self.rows_read,
            tables_read=tuple(sorted(self.tables_read)),
            columns_read=tuple(sorted(
                f"{table}.{column}" for table, column in self.columns_read)),
        )


@dataclass(frozen=True)
class LedgerSummary:
    """The parent-computed, model-invisible provenance for one run -- the
    ``ledger`` block of the envelope (analyst-mode-proposal.md S4.5:
    ``ledger: query_count, tables_read[], columns_read[], rows_read``)."""

    query_count: int
    rows_read: int
    tables_read: tuple[str, ...]
    columns_read: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "query_count": self.query_count,
            "rows_read": self.rows_read,
            "tables_read": list(self.tables_read),
            "columns_read": list(self.columns_read),
        }


class _CountingCursor:
    """Wraps a real cursor so every row a caller actually consumes -- by
    iteration, ``fetchone``, ``fetchmany`` or ``fetchall`` -- is counted
    exactly once, regardless of which of those the analyst code happens to
    use to drain its results."""

    def __init__(self, cursor: sqlite3.Cursor, state: _LedgerState):
        self._state = state
        _CURSORS[self] = cursor

    def __iter__(self):
        return self

    def __next__(self):
        row = next(_cursor_for(self))
        self._state.rows_read += 1
        return row

    def fetchone(self):
        row = _cursor_for(self).fetchone()
        if row is not None:
            self._state.rows_read += 1
        return row

    def fetchmany(self, size=None):
        cursor = _cursor_for(self)
        rows = (cursor.fetchmany(size) if size is not None
                else cursor.fetchmany())
        self._state.rows_read += len(rows)
        return rows

    def fetchall(self):
        rows = _cursor_for(self).fetchall()
        self._state.rows_read += len(rows)
        return rows

    @property
    def description(self):
        return _cursor_for(self).description

    @property
    def rowcount(self):
        return _cursor_for(self).rowcount

    @property
    def lastrowid(self):
        return _cursor_for(self).lastrowid

    def execute(self, sql: str, params=()) -> "_CountingCursor":
        cursor = _cursor_for(self)
        result = cursor.execute(sql, params)
        self._state.query_count += 1
        return _CountingCursor(result, self._state)

    def executemany(self, sql: str, seq_of_params) -> "_CountingCursor":
        cursor = _cursor_for(self)
        result = cursor.executemany(sql, seq_of_params)
        self._state.query_count += 1
        return _CountingCursor(result, self._state)

    def close(self) -> None:
        _forget_cursor(self)


class _LedgeredCursor:
    """``conn.cursor().execute(...)`` counted the same way ``conn.execute``
    is -- so a caller that prefers the explicit-cursor style gets the same
    ledger, not a silent gap in it."""

    def __init__(self, cursor: sqlite3.Cursor, state: _LedgerState):
        self._state = state
        _CURSORS[self] = cursor

    def execute(self, sql: str, params=()) -> _CountingCursor:
        cur = _cursor_for(self).execute(sql, params)
        self._state.query_count += 1
        return _CountingCursor(cur, self._state)

    @property
    def description(self):
        return _cursor_for(self).description

    @property
    def rowcount(self):
        return _cursor_for(self).rowcount

    @property
    def lastrowid(self):
        return _cursor_for(self).lastrowid

    def close(self) -> None:
        _forget_cursor(self)


class LedgeredConnection:
    """The ``conn`` analyst-mode code receives: read-only, authorizer-backed,
    and -- regardless of how its input connection was opened --
    unable to hand its own vault path back out (analyst-mode-proposal.md
    S3.1: "the generated code never receives a vault path"; this wrapper adds
    no path-revealing attribute of its own).
    """

    def __init__(self, real_conn: sqlite3.Connection):
        self._state = _LedgerState()
        _CONNECTIONS[self] = real_conn

        # A weak-reference callback avoids making a cycle from the real
        # connection back to this wrapper through SQLite's authorizer slot.
        wrapper_ref = weakref.ref(self)

        def authorize(action, arg1, arg2, dbname, trigger):
            wrapper = wrapper_ref()
            if wrapper is None:
                return sqlite3.SQLITE_DENY
            return wrapper._authorize(action, arg1, arg2, dbname, trigger)

        _AUTHORIZERS[self] = authorize
        real_conn.set_authorizer(authorize)

    def _authorize(self, action, arg1, arg2, dbname, trigger):
        if (action in _DENIED_ACTIONS
                or (action in _DENIED_WRITE_ACTIONS and dbname != "temp")):
            self._state.denied_count += 1
            if len(self._state.denied) < MAX_RETAINED_DENIALS:
                # Truncated, not stored whole: `arg1` is child-authored for
                # both denied actions. Never a silent truncation -- the marker
                # says the entry was cut, so a reader cannot mistake a prefix
                # for the whole argument.
                if arg1 is not None and len(arg1) > MAX_RETAINED_DENIAL_ARG:
                    arg1 = arg1[:MAX_RETAINED_DENIAL_ARG] + "...[truncated]"
                self._state.denied.append((action, arg1))
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ:
            table, column = arg1, arg2
            if table:
                if _is_catalog(table):
                    # RECORDED, NOT DENIED -- and the distinction was measured
                    # rather than reasoned. Denying catalog reads outright does
                    # not break a plain SELECT (SQLite prepares those without an
                    # authorizer read here), which is why denial looked safe and
                    # was tried first. But an adversarial code review measured
                    # what a plain SELECT does not exercise: `CREATE TEMP TABLE`
                    # fails with "access to temp.sqlite_temp_master.ROWID is
                    # prohibited" and `ANALYZE` fails on sqlite_master -- both
                    # legitimate shapes for a real analysis. A denial that
                    # breaks ordinary work is a defect, not a safeguard.
                    #
                    # The hole this closes is unchanged: catalog reads stay OUT
                    # of `tables_read`, so a run that consulted only SQLite's
                    # own catalog still presents an empty vault-table set and
                    # the zero-read gate refuses it. Provenance keeps the fact
                    # that the catalog was read, separately.
                    self._state.catalog_tables_read.add(table)
                else:
                    self._state.tables_read.add(table)
                    if column:
                        self._state.columns_read.add((table, column))
        return sqlite3.SQLITE_OK

    def execute(self, sql: str, params=()) -> _CountingCursor:
        cur = _connection_for(self).execute(sql, params)
        self._state.query_count += 1
        return _CountingCursor(cur, self._state)

    def executemany(self, sql: str, seq_of_params) -> _CountingCursor:
        cur = _connection_for(self).executemany(sql, seq_of_params)
        self._state.query_count += 1
        return _CountingCursor(cur, self._state)

    def cursor(self) -> _LedgeredCursor:
        return _LedgeredCursor(_connection_for(self).cursor(), self._state)

    def close(self) -> None:
        real_conn = _CONNECTIONS.pop(self, None)
        _AUTHORIZERS.pop(self, None)
        if real_conn is not None:
            real_conn.close()

    @property
    def _conn(self) -> _ParentMetadataConnection:
        """Return only the restricted parent-metadata compatibility handle."""
        return _ParentMetadataConnection(self)

    def __enter__(self) -> "LedgeredConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @property
    def ledger(self) -> LedgerSummary:
        """A snapshot of this run's provenance so far. Safe to read at any
        point; it never mutates and never touches the vault itself."""
        return self._state.summary()

    @property
    def denied_events(self) -> tuple[tuple[int, str | None], ...]:
        """(action_code, arg1) for every statement this connection refused --
        exposed for tests; not part of the envelope."""
        return tuple(self._state.denied)


def wrap_connection(real_conn: sqlite3.Connection) -> LedgeredConnection:
    """Wrap an already-open connection with persistent writes disabled.

    Fixture callers may pass a writable connection; the wrapper's authorizer
    denies every persistent-database write while retaining temporary scratch
    tables. Production code should prefer ``open_ledgered``, which also owns
    the read-only open.
    """
    return LedgeredConnection(real_conn)


def open_ledgered(vault_path) -> LedgeredConnection:
    """Open ``vault_path`` read-only and return the ledgered wrapper analyst
    code is bound to.

    Reuses ``db.connect(path, read_only=True)`` (db.py:50-54) rather than
    reimplementing the ``file:...?mode=ro`` URI open, so this module inherits
    the same read-only guarantee and pragmas as every other read-only caller
    in the project, and so ``tests/conftest.py``'s production-DB guard (which
    patches ``db.connect``) covers this path too.
    """
    real_conn = dbmod.connect(vault_path, read_only=True)
    return LedgeredConnection(real_conn)
