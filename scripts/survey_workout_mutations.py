#!/usr/bin/env python3
"""Survey workout HR summaries that no longer agree with raw HR samples.

This is deliberately a read-only survey.  It delegates the sample-window and
evidence-gate logic to ``db.reconcile_workout_heart_rate`` with ``dry_run`` so
this script cannot drift from the production reconciliation path.

The records table has no workout identifier.  Consequently this can recover
only average/max heart rate by windowing records on a workout's start/end;
``duration_min`` remains the value stored on workouts and is not recomputed or
validated here.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_advisor import db


DEFAULT_SINCE = "2026-06-22"
_HR_FIGURE = re.compile(
    r"(?:avg(?:erage)?\s+HR|average\s+heart\s+rate(?:\s+of)?)\s*[:=]?\s*"
    r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _scope_workouts(conn, since: str, until: str | None) -> dict[int, dict[str, Any]]:
    clauses = ["local_date >= ?"]
    params: list[str] = [since]
    if until:
        clauses.append("local_date <= ?")
        params.append(until)
    sql = (
        "SELECT id, local_date, workout_type FROM workouts WHERE "
        + " AND ".join(clauses)
    )
    return {
        row["id"]: dict(row)
        for row in conn.execute(sql, tuple(params)).fetchall()
    }


def _brief_reference(conn, day: str, old_avg: float | None) -> dict[str, Any]:
    """Return evening-brief HR figures and whether one equals the old average."""
    rows = conn.execute(
        "SELECT id, text FROM insights "
        "WHERE date = ? AND lower(coalesce(tags, '')) LIKE '%evening%' "
        "ORDER BY id",
        (day,),
    ).fetchall()
    figures: list[float] = []
    ids: list[int] = []
    for row in rows:
        ids.append(row["id"])
        figures.extend(float(value) for value in _HR_FIGURE.findall(row["text"]))
    matched = (
        old_avg is not None
        and any(abs(value - old_avg) <= 0.005 for value in figures)
    )
    return {"insight_ids": ids, "brief_avg_hr": figures, "brief_old_match": matched}


def survey(conn, since: str = DEFAULT_SINCE, until: str | None = None) -> dict[str, Any]:
    """Survey reconciliation-qualified differences without writing to conn."""
    scoped = _scope_workouts(conn, since, until)
    moved: list[dict[str, Any]] = []

    def report(workout, samples) -> None:
        w = dict(workout)
        s = dict(samples)
        meta = scoped[w["id"]]
        brief = _brief_reference(conn, w["local_date"], w["avg_heart_rate"])
        moved.append(
            {
                "id": w["id"],
                "local_date": w["local_date"],
                "workout_type": meta["workout_type"],
                "old_avg": w["avg_heart_rate"],
                "new_avg": s["a"],
                "avg_delta": (
                    None
                    if w["avg_heart_rate"] is None
                    else s["a"] - w["avg_heart_rate"]
                ),
                "old_max": w["max_heart_rate"],
                "new_max": s["m"],
                "max_delta": (
                    None
                    if w["max_heart_rate"] is None
                    else s["m"] - w["max_heart_rate"]
                ),
                "sample_count": s["n"],
                **brief,
            }
        )

    # This is the production windowing and evidence gate.  dry_run is
    # essential: the connection is read-only and the survey must never update
    # workouts.  duration_min is used only by that existing coverage gate; it
    # is not recomputed from records.
    db.reconcile_workout_heart_rate(
        conn,
        since=since,
        until=until,
        dry_run=True,
        report=report,
    )
    return {"since": since, "until": until, "surveyed": len(scoped), "moved": moved}


def _fmt(value: float | None, digits: int = 2) -> str:
    return "missing" if value is None else f"{value:.{digits}f}"


def print_report(result: dict[str, Any]) -> None:
    moved = result["moved"]
    print(f"workouts surveyed: {result['surveyed']} (local_date >= {result['since']}"
          + (f", <= {result['until']})" if result["until"] else ")"))
    print(f"rows differing under reconcile_workout_heart_rate: {len(moved)}")
    print("duration_min: not recomputed (records has no workout identifier; HR only is window-recoverable)")

    if not moved:
        print("no moved rows to cross-reference with evening briefs")
        return

    dates = Counter(row["local_date"] for row in moved)
    types = Counter(row["workout_type"] for row in moved)
    print("by local date: " + ", ".join(f"{day}={count}" for day, count in sorted(dates.items())))
    print("by workout type: " + ", ".join(f"{kind}={count}" for kind, count in sorted(types.items())))

    numeric_deltas = [
        (abs(row[field]), row["id"], field, row[field])
        for row in moved
        for field in ("avg_delta", "max_delta")
        if row[field] is not None
    ]
    if numeric_deltas:
        largest = max(numeric_deltas)
        print(
            "largest absolute numeric delta: "
            f"{largest[3]:+.2f} bpm ({largest[2].removesuffix('_delta')}, workout id {largest[1]})"
        )
    else:
        print("largest absolute numeric delta: none (all moved summaries were missing)")

    brief_matches = 0
    print("moved rows:")
    for row in moved:
        if row["brief_old_match"]:
            brief_matches += 1
        brief = (
            f"insight ids {row['insight_ids']}, avg HR {row['brief_avg_hr']}, "
            f"old-value match={'yes' if row['brief_old_match'] else 'no'}"
            if row["insight_ids"]
            else "no evening insight"
        )
        print(
            f"  id={row['id']} {row['local_date']} {row['workout_type']}: "
            f"avg {_fmt(row['old_avg'])} -> {_fmt(row['new_avg'])} "
            f"(delta={_fmt(row['avg_delta'])}); "
            f"max {_fmt(row['old_max'], 1)} -> {_fmt(row['new_max'], 1)} "
            f"(delta={_fmt(row['max_delta'], 1)}); {brief}"
        )
    print(f"moved rows whose evening brief quoted the old average: {brief_matches}")
    if len(moved) == 1:
        print("cluster assessment: one row cannot establish a date or workout-type cluster")
    else:
        date_leader, date_count = dates.most_common(1)[0]
        type_leader, type_count = types.most_common(1)[0]
        print(f"date concentration: {date_leader}={date_count}/{len(moved)}")
        print(f"type concentration: {type_leader}={type_count}/{len(moved)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/health.db", help="SQLite database path")
    parser.add_argument("--since", default=DEFAULT_SINCE)
    parser.add_argument("--until")
    args = parser.parse_args()

    # Do not call init_db or any write-capable helper.  The live services use
    # this database, so the survey's only permitted connection is read-only.
    conn = db.connect(Path(args.db), read_only=True)
    try:
        print_report(survey(conn, since=args.since, until=args.until))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
