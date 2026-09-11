"""Parent-owned correctness checks for analyst envelopes.

The analyst is allowed to choose a table shape, but it is not allowed to choose
what a known quantity means.  A declaration is carried by the table name (the
envelope grammar deliberately has no model-authored metadata channel).  This
module owns the declaration vocabulary and recomputes declared values with the
deterministic analysis layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from . import analysis
from . import db
from . import normalize

# The envelope rounds published analysis values to one decimal place for impact
# minutes.  Summing already-rounded daily cache rows can differ from rounding a
# direct weekly bucket total by one displayed tenth.  One tenth is therefore
# the explicit rounding tolerance; the one-day-per-week error is orders of
# magnitude larger and remains a refusal.
ROUNDING_TOLERANCE = 0.1

# These are quantity *families*.  ``weekly_series:<metric>`` and
# ``block_comparison:<metric>:<weeks>`` are parameterized members of one family.
STANDARD_QUANTITIES = (
    "jog_minutes_per_week",
    "weekly_series:<metric>",
    "block_comparison:<metric>:<weeks>",
    "block_structure",
    "weekly_readiness",
    "sleep_asleep_per_week",
)
STANDARD_QUANTITY_COUNT = len(STANDARD_QUANTITIES)


@dataclass(frozen=True)
class Declaration:
    quantity: str
    kind: str
    metric: str | None = None
    block_weeks: int | None = None


@dataclass(frozen=True)
class CrossCheck:
    verification: str
    quantity: str | None = None
    reason: str | None = None
    oracle: Any = None
    envelope: Any = None

    @property
    def ok(self) -> bool:
        return self.verification == "cross_checked"


def _declaration_from_name(name: str) -> Declaration | None:
    """Decode the short, grammar-safe table-name declaration wire format."""
    if name == "jog_minutes_per_week":
        return Declaration(name, "jog")
    if name == "weekly_readiness":
        return Declaration(name, "weekly_readiness")
    if name == "block_structure":
        return Declaration(name, "block_structure")
    if name == "sleep_asleep_per_week":
        return Declaration(name, "weekly", "sleep_asleep")

    # ``weekly_<metric>`` is the compact wire form of weekly_series:<metric>.
    # Also accept the unshortened form when it happens to fit the envelope's
    # name grammar; this keeps the protocol readable in small test fixtures.
    for prefix in ("weekly_", "weekly_series_"):
        if name.startswith(prefix):
            metric = name[len(prefix):]
            if metric in normalize.CATALOG:
                return Declaration(f"weekly_series:{metric}", "weekly", metric)

    # bc_<metric>_<positive integer> is the compact form of a block comparison.
    match = re.fullmatch(r"bc_(.+)_([1-9][0-9]*)", name)
    if match and match.group(1) in normalize.CATALOG:
        weeks = int(match.group(2))
        return Declaration(
            f"block_comparison:{match.group(1)}:{weeks}",
            "block_comparison", match.group(1), weeks)
    return None


def declared_quantity(table_name: str) -> str | None:
    declaration = _declaration_from_name(table_name)
    return declaration.quantity if declaration else None


def parse_declaration(table_name: str) -> Declaration | None:
    """Return the parent-owned declaration represented by a table name."""
    return _declaration_from_name(table_name)


def _column(table: dict, *names: str) -> int | None:
    columns = list(table["columns"])
    for name in names:
        if name in columns:
            return columns.index(name)
    return None


def _value_column(table: dict, *names: str) -> int | None:
    index = _column(table, *names)
    if index is not None:
        return index
    # A declaration is still useful with a concise two-column table.
    return 1 if len(table["columns"]) == 2 else None


def _as_date(value: Any) -> date | None:
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    text = str(integer)
    if len(text) == 8:
        try:
            return date.fromisoformat(f"{text[:4]}-{text[4:6]}-{text[6:]}")
        except ValueError:
            return None
    if len(text) == 6:  # year*100 + ISO week, e.g. 202633
        try:
            return date.fromisocalendar(int(text[:4]), int(text[4:]), 1)
        except ValueError:
            return None
    return None


def _period(row: tuple, table: dict) -> str | None:
    index = _column(table, "date_yyyymmdd", "week_start", "period_start",
                    "as_of_yyyymmdd", "iso_week", "week")
    if index is None:
        index = 0 if row else None
    if index is None:
        return None
    day = _as_date(row[index])
    return day.isoformat() if day else None


def _bounds(table: dict) -> tuple[str, str] | None:
    periods = [_period(row, table) for row in table["rows"]]
    periods = [p for p in periods if p is not None]
    if not periods:
        return None
    start = min(periods)
    # Weekly declarations use complete Monday-Sunday spans.  For daily
    # declarations this is harmless and preserves the observed day range.
    end = (date.fromisoformat(max(periods)) + timedelta(days=6)).isoformat()
    return start, end


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _within_tolerance(actual: float, expected: float) -> bool:
    # Round the subtraction itself so binary tails cannot turn an exact
    # one-tenth boundary into a spurious disagreement.
    return round(abs(actual - expected), 10) <= ROUNDING_TOLERANCE


def _compare_rows(table: dict, expected: list[dict], value_keys: tuple[str, ...]) -> tuple[bool, Any, Any, str | None]:
    """Compare period-keyed analysis rows against a numeric envelope table."""
    actual_period = {}
    value_index = _value_column(table, *value_keys)
    if value_index is None:
        return False, expected, None, f"declared table lacks a value column ({', '.join(value_keys)})"
    for row in table["rows"]:
        period = _period(row, table)
        value = _number(row[value_index])
        if period is None or value is None or period in actual_period:
            return False, expected, None, "declared table has an invalid or duplicate period"
        actual_period[period] = value
    oracle = {r["period_start"]: r for r in expected}
    if set(actual_period) != set(oracle):
        return False, expected, actual_period, "declared periods differ from recomputed periods"
    for period, value in actual_period.items():
        expected_value = _number(oracle[period].get(value_keys[0]))
        if expected_value is None or not _within_tolerance(value, expected_value):
            return False, expected, actual_period, f"value disagreement at {period}"
    return True, expected, actual_period, None


def _check_weekly(conn, declaration: Declaration, table: dict) -> CrossCheck:
    bounds = _bounds(table)
    if bounds is None:
        # Nothing was recomputed, so nothing was verified. An empty or
        # period-less declared table must not read as a clean cross-check:
        # a gate that reports success over zero comparisons is the inverted
        # instrument, and "unverified" is the honest label for it.
        return CrossCheck("unverified", declaration.quantity,
                          reason="declared table has no parseable period "
                                 "column; nothing was recomputed")
    if declaration.kind == "jog":
        expected = analysis.impact_volume(conn, *bounds, by="week")
        keys = ("jog_minutes", "minutes")
    else:
        expected = [
            {**row, "period_start": row["week_start"]}
            for row in analysis.weekly_series(conn, declaration.metric, *bounds)
        ]
        keys = ("mean", declaration.metric)
    ok, oracle, actual, reason = _compare_rows(table, expected, keys)
    return CrossCheck("cross_checked" if ok else "refused_disagreement",
                      declaration.quantity, reason=reason, oracle=oracle,
                      envelope=actual)


def _check_block_comparison(conn, declaration: Declaration, table: dict) -> CrossCheck:
    index = _column(table, "as_of_yyyymmdd", "date_yyyymmdd", "week_start")
    if index is None or len(table["rows"]) != 1:
        return CrossCheck("refused_disagreement", declaration.quantity,
                          reason="block comparison needs one row and as_of_yyyymmdd")
    as_of = _as_date(table["rows"][0][index])
    if as_of is None:
        return CrossCheck("refused_disagreement", declaration.quantity,
                          reason="invalid block comparison as_of date")
    oracle = analysis.block_comparison(conn, declaration.metric,
                                       declaration.block_weeks, as_of.isoformat())
    row = table["rows"][0]
    actual = {}
    for key in ("recent_mean", "previous_mean", "diff", "mdc95"):
        idx = _column(table, key)
        if idx is not None:
            actual[key] = _number(row[idx])
    expected = {
        "recent_mean": oracle.get("blocks", {}).get("recent", {}).get("mean"),
        "previous_mean": oracle.get("blocks", {}).get("previous", {}).get("mean"),
        "diff": oracle.get("diff"), "mdc95": oracle.get("mdc95"),
    }
    if set(actual) != {key for key, value in expected.items() if value is not None}:
        reason = "block comparison columns do not match the recomputed result"
        return CrossCheck("refused_disagreement", declaration.quantity, reason,
                          oracle, actual)
    ok = all(_within_tolerance(actual[k], expected[k]) for k in actual)
    return CrossCheck("cross_checked" if ok else "refused_disagreement",
                      declaration.quantity,
                      reason=None if ok else "block comparison value disagreement",
                      oracle=oracle, envelope=actual)


def _check_block_structure(conn, table: dict) -> CrossCheck:
    """Check the canonical one-row-per-day block-structure projection."""
    if not table["rows"]:
        return CrossCheck("unverified", "block_structure",
                          reason="declared table is empty; nothing was recomputed")
    date_index = _column(table, "date_yyyymmdd", "period_start")
    longest_index = _column(table, "longest_block_min")
    qualified_index = _column(table, "qualified_block_min")
    if date_index is None or longest_index is None or qualified_index is None:
        return CrossCheck("refused_disagreement", "block_structure",
                          reason="block structure needs date, longest, and qualified columns")

    actual = {}
    for row in table["rows"]:
        day = _as_date(row[date_index])
        longest = _number(row[longest_index])
        qualified = _number(row[qualified_index])
        if day is None or longest is None or qualified is None or day.isoformat() in actual:
            return CrossCheck("refused_disagreement", "block_structure",
                              reason="invalid or duplicate block-structure row")
        actual[day.isoformat()] = (longest, qualified)

    oracle = {}
    for day in actual:
        session_rows = conn.execute(
            "SELECT start_utc, end_utc FROM workouts WHERE local_date = ? "
            "AND workout_type IN ('running', 'walking', 'hiking') ORDER BY start_utc",
            (day,),
        ).fetchall()
        blocks = [analysis.longest_block(conn, row[0], row[1]) for row in session_rows]
        longest = max((block["bridged_min"] for block in blocks), default=0.0)
        qualified_values = [block["qualified_min"] for block in blocks
                           if block["qualified_min"] is not None]
        qualified = max(qualified_values, default=0.0)
        oracle[day] = (longest, qualified)
    ok = set(actual) == set(oracle) and all(
        _within_tolerance(actual[day][i], oracle[day][i])
        for day in actual for i in (0, 1)
    )
    return CrossCheck("cross_checked" if ok else "refused_disagreement",
                      "block_structure",
                      reason=None if ok else "block structure value disagreement",
                      oracle=oracle, envelope=actual)


def _check_weekly_readiness(conn, table: dict) -> CrossCheck:
    bounds = _bounds(table)
    if bounds is None or len(table["rows"]) != 1:
        return CrossCheck("refused_disagreement", "weekly_readiness",
                          reason="weekly readiness needs one dated row")
    as_of = bounds[0]
    oracle = analysis.weekly_readiness(conn, as_of)
    idx = _value_column(table, "score")
    value = _number(table["rows"][0][idx]) if idx is not None else None
    expected = _number(oracle.get("score"))
    ok = value is not None and expected is not None and _within_tolerance(value, expected)
    return CrossCheck("cross_checked" if ok else "refused_disagreement",
                      "weekly_readiness",
                      reason=None if ok else "weekly readiness score disagreement",
                      oracle=oracle, envelope=value)


def cross_check(vault_path: str | Path, envelope) -> CrossCheck:
    """Recompute every declared table, or mark an undeclared table unverified."""
    declarations = [(_declaration_from_name(table["name"]), table)
                    for table in envelope.tables]
    if not declarations or all(declaration is None for declaration, _ in declarations):
        return CrossCheck("unverified", reason="result table has no standard quantity declaration")

    conn = db.connect(vault_path, read_only=True)
    try:
        saw_unverified = False
        for declaration, table in declarations:
            if declaration is None:
                saw_unverified = True
                continue
            if declaration.kind in {"jog", "weekly"}:
                result = _check_weekly(conn, declaration, table)
            elif declaration.kind == "block_comparison":
                result = _check_block_comparison(conn, declaration, table)
            elif declaration.kind == "weekly_readiness":
                result = _check_weekly_readiness(conn, table)
            elif declaration.kind == "block_structure":
                result = _check_block_structure(conn, table)
            else:
                result = CrossCheck("unverified", declaration.quantity)
            if result.verification == "refused_disagreement":
                return result
        if saw_unverified:
            return CrossCheck("unverified", reason="at least one result table lacks a standard declaration")
        if len(declarations) == 1:
            return result
        return CrossCheck("cross_checked", quantity="multiple standard quantities")
    finally:
        conn.close()


# Readable spelling for callers that use ``crosscheck`` as one word.
crosscheck = cross_check


__all__ = [
    "ROUNDING_TOLERANCE", "STANDARD_QUANTITIES", "STANDARD_QUANTITY_COUNT",
    "CrossCheck", "cross_check", "declared_quantity",
    "crosscheck", "parse_declaration",
]
