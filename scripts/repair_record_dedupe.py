#!/usr/bin/env python3
"""One-off repair for the pre-upsert dedupe key (audit P0-1).

Until 2026-07-29 `record_key()` hashed the sample's *value*, so every Health
Auto Export re-send that recomputed a quantity inserted a second row for one
physical sample and `recompute_daily_metrics` summed both.

Two things have to happen to the stored rows, and both matter:

1. **Collapse** the duplicate groups — same (metric, start, end, source),
   different value — keeping the newest row (highest id), which is the sender's
   latest correction.
2. **Rewrite every dedupe_key** of the affected metrics to the new scheme.
   Without this the fix is worse than the bug: a re-send (or a fresh
   export.zip) computes a window-only key that matches nothing stored, so
   INSERT ... ON CONFLICT has no conflict to resolve and duplicates the row
   again — for the entire history, not just the recent window.

Dry run by default; --apply writes. Back up first (scripts/backup_health.sh)
and rehearse on a copy: this rewrites real history.

    ./.venv/bin/python scripts/repair_record_dedupe.py --db /path/to/copy.db
    ./.venv/bin/python scripts/repair_record_dedupe.py --db /path/to/copy.db --apply
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod  # noqa: E402


def affected_metrics(conn) -> list[str]:
    """Metrics whose natural key is the window, present in this DB."""
    rows = conn.execute("SELECT DISTINCT metric FROM records").fetchall()
    return sorted(m["metric"] for m in rows
                  if not dbmod._value_identifies_sample(m["metric"]))


def duplicate_groups(conn, metric: str) -> tuple[int, int]:
    """(groups, extra_rows) for one metric under the window-only natural key."""
    row = conn.execute(
        """
        SELECT COUNT(*) groups, COALESCE(SUM(c - 1), 0) extra FROM (
            SELECT COUNT(*) c FROM records WHERE metric = ?
            GROUP BY start_utc, end_utc, source HAVING c > 1
        )
        """,
        (metric,),
    ).fetchone()
    return row["groups"], row["extra"]


# The losing rows of every duplicate group: same window and source, not the
# newest. Ranking in one window-function pass keeps this linear — the obvious
# `id NOT IN (SELECT MAX(id) ... GROUP BY ...)` formulation re-scans millions of
# rows per day and never finishes on basal_energy.
_LOSERS = """
    SELECT id, local_date, value FROM (
        SELECT id, local_date, value,
               ROW_NUMBER() OVER (PARTITION BY start_utc, end_utc, source
                                  ORDER BY id DESC) rn
        FROM records WHERE metric = ?
    ) WHERE rn > 1
"""


def day_deltas(conn, metric: str) -> list[tuple[str, float, float]]:
    """(date, current_sum, deduped_sum) for days this repair would change."""
    dropped = {
        r["local_date"]: r["v"]
        for r in conn.execute(
            f"SELECT local_date, SUM(value) v FROM ({_LOSERS}) GROUP BY local_date",
            (metric,),
        ).fetchall()
    }
    if not dropped:
        return []
    holes = ",".join("?" * len(dropped))
    out = []
    for r in conn.execute(
        f"SELECT local_date, SUM(value) cur FROM records WHERE metric = ? "
        f"AND local_date IN ({holes}) GROUP BY local_date ORDER BY local_date",
        (metric, *dropped),
    ).fetchall():
        out.append((r["local_date"], r["cur"], r["cur"] - dropped[r["local_date"]]))
    return out


def collapse(conn, metric: str) -> int:
    """Delete all but the newest row of each duplicate group. Returns rows gone."""
    cur = conn.execute(
        f"DELETE FROM records WHERE id IN (SELECT id FROM ({_LOSERS}))", (metric,)
    )
    return cur.rowcount


def rekey(conn, metric: str) -> int:
    """Rewrite dedupe_key under the current scheme. Returns rows touched."""
    conn.create_function("new_record_key", 6, dbmod.record_key, deterministic=True)
    cur = conn.execute(
        "UPDATE records SET dedupe_key = new_record_key("
        "metric, start_utc, end_utc, value, unit, source) WHERE metric = ?",
        (metric,),
    )
    return cur.rowcount


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(LOCAL_DB_PATH))
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--show-days", type=int, default=8,
                    help="how many changed days to print per metric")
    args = ap.parse_args()

    conn = dbmod.connect(args.db)
    metrics = affected_metrics(conn)
    print(f"db: {args.db}")
    print(f"window-keyed metrics present: {len(metrics)}\n")

    total_groups = total_extra = 0
    plan = []
    for metric in metrics:
        groups, extra = duplicate_groups(conn, metric)
        total_groups += groups
        total_extra += extra
        if not groups:
            continue
        deltas = day_deltas(conn, metric)
        plan.append((metric, groups, extra, deltas))
        worst = max(deltas, key=lambda d: abs(d[1] - d[2]), default=None)
        print(f"{metric}: {groups} groups, {extra} extra rows, "
              f"{len(deltas)} days change")
        for d, cur, dedup in deltas[:args.show_days]:
            pct = (cur - dedup) / dedup * 100 if dedup else float("nan")
            print(f"    {d}  {cur:12.2f} -> {dedup:12.2f}  ({pct:+.2f}%)")
        if len(deltas) > args.show_days:
            print(f"    ... {len(deltas) - args.show_days} more")
        if worst:
            print(f"    worst: {worst[0]} {worst[1]:.2f} -> {worst[2]:.2f}")
        print()

    print(f"TOTAL: {total_groups} duplicate groups, {total_extra} rows to remove")

    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply.")
        return 0

    t0 = time.time()
    deleted = rekeyed = 0
    pairs = []
    for metric, _g, _e, deltas in plan:
        pairs += [(metric, d) for d, _c, _n in deltas]
    for metric in metrics:
        deleted += collapse(conn, metric)
        rekeyed += rekey(conn, metric)
        print(f"  {metric}: collapsed, {rekeyed} keys rewritten so far "
              f"({time.time() - t0:.0f}s)")
    written = dbmod.recompute_daily_metrics(conn, pairs=pairs)
    conn.commit()
    print(f"\napplied: {deleted} rows deleted, {rekeyed} keys rewritten, "
          f"{written} daily_metrics rows recomputed in {time.time() - t0:.0f}s")

    dupes_left = sum(duplicate_groups(conn, m)[0] for m in metrics)
    print(f"duplicate groups remaining: {dupes_left}")
    return 0 if dupes_left == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
