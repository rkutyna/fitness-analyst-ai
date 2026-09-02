"""Append-only observations of what the user says happened in a session.

An observation preserves the pipeline's computed snapshot beside the user's
statement.  It never writes ``records`` or ``daily_metrics`` and never updates
an existing row; a revised statement is a new row with a new ``stated_at``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

from . import analysis
from . import db
from . import metrics

SCOPES = ("week", "day", "session")
FIELDS = ("jog_minutes", "jogged", "longest_block_min", "structure")
EVIDENCE = ("recall", "segments", "device")


def _date(value: str, name: str = "local_date") -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be YYYY-MM-DD") from None


def _range(start: str, end: str) -> tuple[str, str]:
    a, b = _date(start, "start"), _date(end, "end")
    if a > b:
        raise ValueError("start must not be after end")
    return a.isoformat(), b.isoformat()


def _timestamp(value: str | None, name: str) -> str:
    if value is None:
        return db.utcnow_iso()
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from None
    return value


def _validate(*, scope: str, local_date: str, workout_key: str, field: str,
              computed_value: Any, stated_value: Any, stated_text: Any,
              agrees: Any, evidence: str) -> None:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of: {', '.join(SCOPES)}")
    day = _date(local_date)
    if scope == "week" and day.weekday() != 0:
        raise ValueError("local_date must be the Monday when scope is 'week'")
    if field not in FIELDS:
        raise ValueError(f"field must be one of: {', '.join(FIELDS)}")
    if scope != "session" and workout_key:
        raise ValueError("workout_key must be empty for week/day observations")
    if scope == "session" and not workout_key:
        raise ValueError("workout_key is required for a session observation")
    if stated_value is None and stated_text is None:
        raise ValueError("provide stated_value or stated_text")
    if stated_value is not None and stated_text is not None:
        raise ValueError("provide only one of stated_value or stated_text")
    if stated_value is not None:
        try:
            float(stated_value)
        except (TypeError, ValueError):
            raise ValueError("stated_value must be numeric") from None
    if computed_value is not None:
        try:
            float(computed_value)
        except (TypeError, ValueError):
            raise ValueError("computed_value must be numeric") from None
    if agrees not in (0, 1, False, True):
        raise ValueError("agrees must be 0 or 1")
    if evidence not in EVIDENCE:
        raise ValueError(f"evidence must be one of: {', '.join(EVIDENCE)}")


def _record(conn: sqlite3.Connection, *, scope: str, local_date: str,
            workout_key: str = "", field: str,
            computed_value: float | None = None,
            stated_value: float | None = None, stated_text: str | None = None,
            agrees: int, evidence: str, note: str | None = None,
            computed_at: str | None = None, stated_at: str | None = None) -> None:
    """Implementation shared by the public function and the loader."""
    _validate(scope=scope, local_date=local_date, workout_key=workout_key,
              field=field, computed_value=computed_value,
              stated_value=stated_value, stated_text=stated_text,
              agrees=agrees, evidence=evidence)
    computed_at = _timestamp(computed_at, "computed_at")
    stated_at = _timestamp(stated_at, "stated_at")
    conn.execute(
        """
        INSERT OR IGNORE INTO session_observation
            (scope, local_date, workout_key, field, computed_value,
             stated_value, stated_text, agrees, computed_at, stated_at,
             evidence, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (scope, local_date, workout_key, field,
         float(computed_value) if computed_value is not None else None,
         float(stated_value) if stated_value is not None else None,
         stated_text, int(bool(agrees)), computed_at, stated_at, evidence, note),
    )
    conn.commit()


def record(conn: sqlite3.Connection, *, scope: str, local_date: str,
           field: str, computed_value: float | None = None,
           stated_value: float | None = None, stated_text: str | None = None,
           agrees: int, evidence: str, note: str | None = None,
           computed_at: str | None = None, stated_at: str | None = None,
           workout_key: str = "") -> None:
    """Append one observation; a later statement is a new row.

    ``workout_key`` is optional in the call shape for week/day observations and
    is required by validation for session observations.
    """
    _record(conn, scope=scope, local_date=local_date, workout_key=workout_key,
            field=field, computed_value=computed_value,
            stated_value=stated_value, stated_text=stated_text, agrees=agrees,
            evidence=evidence, note=note, computed_at=computed_at,
            stated_at=stated_at)


def _latest_rows(conn: sqlite3.Connection, start: str, end: str) -> list[dict]:
    start, end = _range(start, end)
    rows = conn.execute(
        """
        SELECT * FROM (
            SELECT so.*, ROW_NUMBER() OVER (
                PARTITION BY scope, local_date, workout_key, field
                ORDER BY stated_at DESC
            ) AS rn
            FROM session_observation AS so
            WHERE local_date BETWEEN ? AND ?
        ) WHERE rn = 1
        ORDER BY local_date, scope, workout_key, field
        """, (start, end)).fetchall()
    return [dict(row) for row in rows]


def disagreements(conn: sqlite3.Connection, start: str, end: str) -> list[dict]:
    """Return latest claims in a range where the user and Python disagree.

    Numeric ``delta`` is ``computed_value - stated_value``: positive means the
    pipeline reported more than the user's statement.  A text-only structure has
    no scalar delta, but its computed scalar and statement remain in the row.
    Sessions with no observation are absent, not disagreements.
    """
    out = []
    for row in _latest_rows(conn, start, end):
        if row["agrees"] != 0:
            continue
        row.pop("rn", None)
        row["delta"] = (
            round(float(row["computed_value"]) - float(row["stated_value"]), 6)
            if row["computed_value"] is not None and row["stated_value"] is not None
            else None
        )
        out.append(row)
    return out


def coverage(conn: sqlite3.Connection, start: str, end: str) -> dict:
    """Count latest asked claims, confirmed claims, and corrections."""
    rows = _latest_rows(conn, start, end)
    return {
        "asked": len(rows),
        "confirmed": sum(row["agrees"] == 1 for row in rows),
        "corrected": sum(row["agrees"] == 0 for row in rows),
    }


def _session_line(conn: sqlite3.Connection, workout: sqlite3.Row) -> dict:
    start, end = workout["start_utc"], workout["end_utc"]
    buckets = metrics.bucket_series(conn, start, end)
    block = analysis.longest_block(conn, start, end)
    jog_minutes = round(sum(bool(row["is_jog"]) for row in buckets)
                        * metrics.IMPACT_BUCKET_SECONDS / 60.0, 1)
    return {
        "date": workout["local_date"],
        "workout_key": workout["dedupe_key"],
        "type": workout["workout_type"],
        "duration_min": workout["duration_min"],
        "jog_minutes": jog_minutes,
        "longest_block_min": block["bridged_min"],
        "qualified_block_min": block["qualified_min"],
        "reps": block["reps"],
        "observations": [],
    }


def week_lines(conn: sqlite3.Connection, monday: str) -> list[dict]:
    """Return one computed presentation line per workout in Monday's week."""
    monday_date = _date(monday, "monday")
    if monday_date.weekday() != 0:
        raise ValueError("monday must be a Monday in YYYY-MM-DD")
    end = (monday_date + timedelta(days=6)).isoformat()
    workouts = conn.execute(
        "SELECT * FROM workouts WHERE local_date BETWEEN ? AND ? "
        "ORDER BY start_utc", (monday_date.isoformat(), end)).fetchall()
    lines = [_session_line(conn, workout) for workout in workouts]
    for line in lines:
        rows = conn.execute(
            "SELECT * FROM session_observation WHERE scope = 'session' "
            "AND local_date = ? AND workout_key = ? ORDER BY stated_at DESC",
            (line["date"], line["workout_key"]),
        ).fetchall()
        seen = set()
        for row in rows:
            if row["field"] not in seen:
                line["observations"].append(dict(row))
                seen.add(row["field"])
        line["asked"] = bool(line["observations"])
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to the vault")
    sub = parser.add_subparsers(dest="command", required=True)
    p_week = sub.add_parser("week-lines")
    p_week.add_argument("monday")
    p_range = sub.add_parser("disagreements")
    p_range.add_argument("start")
    p_range.add_argument("end")
    p_cov = sub.add_parser("coverage")
    p_cov.add_argument("start")
    p_cov.add_argument("end")
    p_record = sub.add_parser("record")
    p_record.add_argument("--scope", required=True, choices=SCOPES)
    p_record.add_argument("--local-date", required=True)
    p_record.add_argument("--workout-key", default="")
    p_record.add_argument("--field", required=True, choices=FIELDS)
    p_record.add_argument("--computed-value", type=float)
    p_record.add_argument("--stated-value", type=float)
    p_record.add_argument("--stated-text")
    p_record.add_argument("--agrees", required=True, type=int, choices=(0, 1))
    p_record.add_argument("--evidence", required=True, choices=EVIDENCE)
    p_record.add_argument("--note")
    p_record.add_argument("--computed-at")
    p_record.add_argument("--stated-at")
    args = parser.parse_args(argv)
    conn = db.connect(args.db, read_only=args.command != "record")
    try:
        if args.command == "record":
            db.init_db(conn)
            record(conn, scope=args.scope, local_date=args.local_date,
                   workout_key=args.workout_key, field=args.field,
                   computed_value=args.computed_value,
                   stated_value=args.stated_value, stated_text=args.stated_text,
                   agrees=args.agrees, evidence=args.evidence, note=args.note,
                   computed_at=args.computed_at, stated_at=args.stated_at)
            result = {"recorded": True}
        elif args.command == "week-lines":
            result = week_lines(conn, args.monday)
        elif args.command == "disagreements":
            result = disagreements(conn, args.start, args.end)
        elif args.command == "coverage":
            result = coverage(conn, args.start, args.end)
        else:
            parser.error("record is available through observations.record")
    finally:
        conn.close()
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
