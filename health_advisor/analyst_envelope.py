"""analyst_envelope.py -- the closed grammar analyst-mode output must satisfy.

Analyst mode lets model-written code answer a question the curated tools
cannot. What that code emits crosses a process boundary as untrusted bytes on
fd 3 (the analyst-mode design; its runtime half is restated in SECURITY.md), and this module
is the validator on the far side: raw bytes in, either a validated
``Envelope`` or a typed ``Refusal`` out. Nothing in between is trusted.
``Refusal.reason`` is a parent-authored headline and is never a fragment of
the untrusted child payload. A refusal may additionally carry ``diagnostic``:
a separate, capped, sanitised, explicitly quoted tail of child stderr for
human display only. It is not part of ``reason`` and is not an evidence or
numeric channel.

Hand-rolled imperative validation, deliberately -- not a JSON-Schema library.
``llm.py`` (~line 931) records why for the project's other synthetic schema
(``submit_answer``): "Encoding the full grammar in JSON Schema would move a
verification decision out of Python, which is the one rule." The same
argument applies here, more sharply, because this schema gates numbers that
reach a provider. ``ENVELOPE_SCHEMA_HINT`` below is exactly that -- a hint a
future prompt builder may show the model -- and it enforces nothing; every
rule below is checked by this module's own Python.

The caps and the grammar come from S4.4 (the whitelist: only numeric cells,
whitelist-matched column/table names, and catalog-vetted units cross into
narration) and S4.5 (closing four permissive JSON defaults, all *measured*
there): ``NaN``/``Infinity`` accepted silently by ``json.loads``, duplicate
keys silently overwriting, ``\\d`` matching non-ASCII digits, and
``-0.0 == 0.0`` while ``repr`` differs.

**Truncation is not implemented.** The first design trimmed an oversized
table; a later review found claims from a truncated table unsound, and S4.5
now says a half-state is worse than none: **a cap breach is a refusal, in
full**, not a shortened envelope. This module never returns a partially
populated ``Envelope``.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from . import normalize

__all__ = [
    "Envelope",
    "Refusal",
    "CAPS",
    "MAX_ROWS_PER_TABLE",
    "MAX_TABLES",
    "MAX_CELLS",
    "MAX_NUMERIC_TOKENS",
    "MAX_ENVELOPE_BYTES",
    "NAME_RE",
    "ALLOWED_UNITS",
    "ALLOWED_TOP_LEVEL_KEYS",
    "ENVELOPE_SCHEMA_HINT",
    "validate",
]


# --------------------------------------------------------------------------- #
# Caps -- S4.5's table, asserted here as constants rather than argued about.
# The proposal itself flags most of these as asserted, not measured (S9.5);
# only the cell cap has an argument (S1.4: the model must not be the one
# sizing the grounding pool). Wall-clock (60s) and diagnostics-reduction
# (200 chars) are NOT this module's concern -- they belong to the sandbox
# lifecycle (A1) and the repair-turn loop (A3) respectively.
# --------------------------------------------------------------------------- #
MAX_ROWS_PER_TABLE = 200
MAX_TABLES = 4
MAX_CELLS = 2_000
MAX_NUMERIC_TOKENS = 200
MAX_ENVELOPE_BYTES = 65_536

CAPS = {
    "max_rows_per_table": MAX_ROWS_PER_TABLE,
    "max_tables": MAX_TABLES,
    "max_cells": MAX_CELLS,
    "max_numeric_tokens": MAX_NUMERIC_TOKENS,
    "max_envelope_bytes": MAX_ENVELOPE_BYTES,
}

# Column/table name grammar (S4.4): lower-snake, 1-31 chars total, ASCII only.
# `fullmatch` (not `match`) so a trailing control character (e.g. "abc\n")
# cannot ride past `$`, which is permitted to match just before a trailing
# newline under `re.match`/`search` but not under `fullmatch` -- confirmed
# empirically while building this module: `NAME_RE.fullmatch("abc\n")` is
# None, `NAME_RE.match("abc\n")` is not.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,30}$")


def _name_hint(name) -> str:
    """Why a name failed NAME_RE, in words the repair retry can act on.

    The refusal reason is the only feedback the model's single repair attempt
    receives (#255): 2 of 6 natural phrasings died on names 1-2 chars over the
    cap because "fails the naming grammar" never said what the grammar was.
    """
    if not isinstance(name, str):
        return f"not a string: {type(name).__name__}"
    if len(name) > 31:
        return f"{len(name)} chars, max 31 — use a shorter name"
    return "lowercase letter first, then lowercase/digits/underscore only"

# The real unit vocabulary (S4.4: "must be in normalize.CATALOG's unit set"),
# read from the catalog rather than copied -- CATALOG is 91 metrics / 26
# distinct units as of this module's writing, and a hardcoded copy would
# silently drift from it. Computed once at import time.
ALLOWED_UNITS = frozenset(entry["unit"] for entry in normalize.CATALOG.values())

_TABLE_KEYS = frozenset({"name", "columns", "units", "rows"})

# --------------------------------------------------------------------------- #
# The raw child payload's top-level shape.
#
# The Envelope diagram in S4.5 lists run_id, question, code_sha256,
# vault_sha256, vault_version, ledger, tables[] and counts -- but every field
# except `tables` is explicitly parent-owned or parent-computed (S4.5, S4.7):
# the child never sends them. And S3.1's CURRENT `emit` signature is
# `emit(name, columns, rows, unit=None)` -- no `note` parameter. (S4.4 talks
# about demoting a `note` channel to the run record; that sentence describes
# the FIRST draft's `emit`, which S3.1 already revised to drop `note`
# entirely. This module follows S3.1, the later and more specific section,
# over S4.4's older phrasing -- flagged in this project's report as a
# resolved ambiguity, not a silent guess.)
#
# So the only key a well-formed child payload may carry is "tables". Anything
# else -- a stray "note", a hostile "stdout", an injected "claims" -- is an
# unknown top-level key and is refused outright, never silently dropped.
# --------------------------------------------------------------------------- #
ALLOWED_TOP_LEVEL_KEYS = frozenset({"tables"})


# --------------------------------------------------------------------------- #
# The envelope schema -- a HINT, not an enforcer. See module docstring.
# --------------------------------------------------------------------------- #
ENVELOPE_SCHEMA_HINT = {
    "tables": [
        {
            "name": "lower_snake_case, 1-31 chars, unique in the envelope",
            "columns": ["lower_snake_case, 1-31 chars, unique in the table"],
            "units": ["one per column, must be a health_advisor.normalize "
                      "canonical unit"],
            "rows": [["one finite number per column, same order as columns"]],
        }
    ],
}


@dataclass(frozen=True)
class Refusal:
    """A typed failure with a safe headline and optional child diagnosis.

    ``reason`` remains parent-authored and never contains a fragment of the
    untrusted payload. ``diagnostic`` is the deliberate exception for the
    sandbox's separately bounded and sanitised child-stderr tail; callers may
    show it as quoted diagnostic detail, but must not treat it as evidence or
    as a source of numbers.
    """

    reason: str
    diagnostic: str | None = None

    def to_dict(self) -> dict:
        """Return the refusal wire shape, omitting absent optional detail."""
        result = {"refused": True, "reason": self.reason}
        if self.diagnostic is not None:
            result["diagnostic"] = self.diagnostic
        return result

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.reason


@dataclass(frozen=True)
class Envelope:
    """A fully validated run result. Every field on this object has already
    passed the grammar below; nothing downstream needs to re-check it."""

    run_id: str
    question: str
    code_sha256: str
    vault_sha256: str
    vault_version: int
    ledger: dict
    tables: tuple  # tuple of {"name","columns","units","rows","row_count"}
    counts: dict

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "code_sha256": self.code_sha256,
            "vault_sha256": self.vault_sha256,
            "vault_version": self.vault_version,
            "ledger": dict(self.ledger),
            "tables": [
                {
                    "name": t["name"],
                    "columns": list(t["columns"]),
                    "units": list(t["units"]),
                    "rows": [list(row) for row in t["rows"]],
                    "row_count": t["row_count"],
                }
                for t in self.tables
            ],
            "counts": dict(self.counts),
        }


class _GrammarRefusal(Exception):
    """Internal control flow only: raised deep inside nested validation loops
    to short-circuit to a single refusal point, and always caught inside this
    module. Never a JSON-Schema evaluator -- every check that raises this is
    plain imperative Python reading the parsed structure directly."""


def _refuse(reason: str):
    raise _GrammarRefusal(reason)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """``object_pairs_hook`` -- S4.5: ``json.loads`` silently keeps the LAST
    of a duplicate key by default (measured: ``{"a":1,"a":2}`` -> ``{'a':2}``
    with no error). Applies at every nesting level, so a duplicated "name" or
    "columns" key inside one table entry is caught exactly like a duplicated
    top-level "tables" key."""
    seen: set[str] = set()
    result: dict = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key: {key!r}")
        seen.add(key)
        result[key] = value
    return result


def _reject_json_constant(name: str):
    """``parse_constant`` -- S4.5: bare ``json.loads`` accepts the extension
    tokens ``NaN``/``Infinity``/``-Infinity`` and yields a non-finite float
    with no error. Refuse the token during parsing, before any cell-level
    finiteness check would even run."""
    raise ValueError(f"disallowed JSON constant: {name}")


def _token_key(value) -> str:
    """A canonical dedup key for the distinct-numeric-token cap (S4.5/S1.4):
    the model must not be the one sizing its own grounding pool. `3` (int)
    and `3.0` (float) are kept distinct, since they would render as distinct
    tokens in narration."""
    if isinstance(value, int):
        return f"i:{value}"
    return f"f:{value!r}"


def _validate_cell(table_name: str, row_i: int, col_j: int, cell):
    """One cell: finite ``int``/``float`` only, ``bool`` explicitly excluded,
    ``-0.0`` normalised to ``0.0``.

    ``bool`` MUST be checked before ``(int, float)``: JSON ``true``/``false``
    decode to Python ``bool``, and ``bool`` is a subclass of ``int``
    (``isinstance(True, int)`` is ``True``), so a naive numeric check accepts
    a boolean as `1`/`0`. That is the "booleans-as-integers" case in the A2
    grammar corpus, kept as a real code path here rather than only a comment.

    Cells arrive as native JSON number types ONLY -- a string cell is refused
    outright, never coerced. This is deliberate and load-bearing: Python's
    numeric coercions accept far more than ASCII digits --
    ``int('\\u0663')`` and ``float('\\u0663')`` (Arabic-Indic digit three)
    both return ``3`` -- so a validator that tried to "helpfully" parse a
    string cell would silently launder a unicode-numeral string into a
    legitimate number. Never calling ``int()``/``float()`` on untrusted cell
    text removes that whole class rather than trying to out-guess it.
    """
    if isinstance(cell, bool):
        _refuse(f"table {table_name!r} row {row_i} col {col_j}: "
                "boolean is not a numeric cell")
    if not isinstance(cell, (int, float)):
        _refuse(f"table {table_name!r} row {row_i} col {col_j}: "
                f"non-numeric cell ({type(cell).__name__})")
    if isinstance(cell, float):
        if not math.isfinite(cell):
            _refuse(f"table {table_name!r} row {row_i} col {col_j}: "
                    "non-finite numeric cell")
        if cell == 0.0 and math.copysign(1.0, cell) < 0:
            cell = 0.0  # -0.0 normalised to 0.0 (S4.5)
    return cell


def _validate_table(raw_table, seen_table_names: set) -> dict:
    if not isinstance(raw_table, dict):
        _refuse(f"table entry must be a JSON object, got "
                f"{type(raw_table).__name__}")

    unknown = set(raw_table) - _TABLE_KEYS
    if unknown:
        _refuse(f"table contains disallowed key(s): {sorted(unknown)}")
    missing = _TABLE_KEYS - set(raw_table)
    if missing:
        _refuse(f"table is missing required key(s): {sorted(missing)}")

    name = raw_table["name"]
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        _refuse(f"table name {name!r} fails the naming grammar "
                f"{NAME_RE.pattern} ({_name_hint(name)})")
    if name in seen_table_names:
        _refuse(f"duplicate table name {name!r}")
    seen_table_names.add(name)

    columns = raw_table["columns"]
    if not isinstance(columns, list) or not columns:
        _refuse(f"table {name!r}: 'columns' must be a non-empty JSON array")
    for c in columns:
        if not isinstance(c, str) or not NAME_RE.fullmatch(c):
            _refuse(f"table {name!r}: column name {c!r} fails the naming "
                    f"grammar {NAME_RE.pattern} ({_name_hint(c)})")
    if len(set(columns)) != len(columns):
        _refuse(f"table {name!r}: duplicate column name(s) in {columns!r}")

    units = raw_table["units"]
    if not isinstance(units, list):
        _refuse(f"table {name!r}: 'units' must be a JSON array")
    if len(units) != len(columns):
        _refuse(f"table {name!r}: len(units)={len(units)} != "
                f"len(columns)={len(columns)}")
    for u in units:
        if u not in ALLOWED_UNITS:
            _refuse(f"table {name!r}: unit {u!r} is not in "
                    "normalize.CATALOG's unit vocabulary")

    rows = raw_table["rows"]
    if not isinstance(rows, list):
        _refuse(f"table {name!r}: 'rows' must be a JSON array")
    if len(rows) > MAX_ROWS_PER_TABLE:
        _refuse(f"table {name!r} exceeds row cap: {len(rows)} > "
                f"{MAX_ROWS_PER_TABLE}")

    validated_rows = []
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(columns):
            got = len(row) if isinstance(row, list) else type(row).__name__
            _refuse(f"table {name!r} row {i}: expected {len(columns)} "
                    f"cells, got {got}")
        validated_rows.append(tuple(
            _validate_cell(name, i, j, cell) for j, cell in enumerate(row)))

    return {
        "name": name,
        "columns": tuple(columns),
        "units": tuple(units),
        "rows": tuple(validated_rows),
        "row_count": len(validated_rows),
    }


def _validate_payload(payload, *, run_id, question, code_sha256,
                       vault_sha256, vault_version, ledger, raw_bytes: int
                       ) -> Envelope:
    if not isinstance(payload, dict):
        _refuse(f"envelope must be a JSON object, got "
                f"{type(payload).__name__}")

    unknown = set(payload) - ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        _refuse(f"envelope contains disallowed top-level key(s): "
                f"{sorted(unknown)}")
    if "tables" not in payload:
        _refuse("envelope is missing required key 'tables'")

    raw_tables = payload["tables"]
    if not isinstance(raw_tables, list):
        _refuse(f"'tables' must be a JSON array, got "
                f"{type(raw_tables).__name__}")
    if len(raw_tables) > MAX_TABLES:
        _refuse(f"too many tables: {len(raw_tables)} > {MAX_TABLES}")

    tables = []
    seen_table_names: set = set()
    total_cells = 0
    numeric_tokens: set = set()

    for raw_table in raw_tables:
        table = _validate_table(raw_table, seen_table_names)
        tables.append(table)
        total_cells += table["row_count"] * len(table["columns"])
        for row in table["rows"]:
            for cell in row:
                numeric_tokens.add(_token_key(cell))

    if total_cells > MAX_CELLS:
        _refuse(f"envelope exceeds cell cap: {total_cells} > {MAX_CELLS}")

    if len(numeric_tokens) > MAX_NUMERIC_TOKENS:
        _refuse(f"envelope exceeds distinct numeric-token cap: "
                f"{len(numeric_tokens)} > {MAX_NUMERIC_TOKENS}")

    # The refusal that is the actual new gate (S1.2): a run whose envelope
    # contains ANY numeric cell while the ledger shows zero vault reads.
    query_count = ledger.get("query_count", 0)
    rows_read = ledger.get("rows_read", 0)
    # `tables_read` excludes SQLite's own catalog (analyst_ledger._is_catalog),
    # so a run that only read sqlite_master cannot satisfy this gate: it reports
    # a query and a row while having consulted nothing about the athlete.
    vault_tables = ledger.get("tables_read") or ()
    if total_cells > 0 and (query_count == 0 or rows_read == 0
                            or not vault_tables):
        _refuse(f"emitted {len(tables)} numeric tables from "
                f"{len(vault_tables)} vault tables and {rows_read} reads")

    counts = {
        "rows": sum(t["row_count"] for t in tables),
        "cells": total_cells,
        "numeric_tokens": len(numeric_tokens),
        "bytes": raw_bytes,
    }

    return Envelope(
        run_id=run_id,
        question=question,
        code_sha256=code_sha256,
        vault_sha256=vault_sha256,
        vault_version=vault_version,
        ledger=dict(ledger),
        tables=tuple(tables),
        counts=counts,
    )


def validate(raw: bytes, *, run_id: str, question: str, code_sha256: str,
             vault_sha256: str, vault_version: int, ledger: dict
             ) -> "Envelope | Refusal":
    """Validate one run's raw fd-3 bytes against the closed grammar.

    ``raw`` is exactly what a child process wrote -- untrusted, unparsed,
    unread until this call. ``run_id``, ``question``, ``code_sha256``,
    ``vault_sha256``, ``vault_version`` and ``ledger`` are all parent-owned or
    parent-computed (S4.5/S4.7/S1.2) and are trusted as given; this function
    does not re-derive them, only attaches them to a validated envelope.

    ``ledger`` is a plain dict with ``query_count``/``rows_read`` (and
    typically ``tables_read``/``columns_read``) -- e.g.
    ``analyst_ledger.LedgeredConnection.ledger.as_dict()`` -- kept as a dict
    rather than importing ``analyst_ledger`` so this module has no dependency
    on sqlite at all and can be tested, and used, without ever opening a
    connection.

    Returns an ``Envelope`` on success. On ANY failure -- byte cap, invalid
    UTF-8, invalid JSON, a grammar violation, or the zero-read refusal --
    returns a ``Refusal`` naming exactly one reason. There is no partial
    result: a cap breach refuses the whole envelope, never a truncated one.
    """
    if not isinstance(raw, (bytes, bytearray)):
        return Refusal(f"envelope must be raw bytes, got {type(raw).__name__}")

    raw = bytes(raw)
    if len(raw) > MAX_ENVELOPE_BYTES:
        return Refusal(f"envelope exceeds {MAX_ENVELOPE_BYTES} bytes "
                        f"({len(raw)} bytes)")

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return Refusal(f"envelope is not valid UTF-8: {exc}")

    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys,
                              parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        return Refusal(f"envelope is not valid JSON: {exc}")

    try:
        return _validate_payload(
            payload, run_id=run_id, question=question,
            code_sha256=code_sha256, vault_sha256=vault_sha256,
            vault_version=vault_version, ledger=ledger, raw_bytes=len(raw))
    except _GrammarRefusal as exc:
        return Refusal(str(exc))
    except (TypeError, ValueError, KeyError, IndexError, RecursionError) as exc:
        # Fail closed on anything the checks above did not anticipate, rather
        # than let a malformed payload raise out of the validator.
        return Refusal(f"envelope failed validation: "
                        f"{type(exc).__name__}: {exc}")
