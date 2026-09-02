#!/usr/bin/env python3
"""Re-aggregate the days where two sources described the same movement (P0-2).

`recompute_daily_metrics` now resolves cross-source overlap (see
normalize.MIRROR_SOURCES / SAMPLE_CEILING), but the stored `daily_metrics`
rows were written before it did — every affected day still carries the summed
total. This recomputes exactly those days; `records` is not touched.

Dry run by default; --apply writes.

    ./.venv/bin/python scripts/repair_source_arbitration.py --db /path/to/copy.db
    ./.venv/bin/python scripts/repair_source_arbitration.py --apply
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod  # noqa: E402


def current(conn, metric: str, date: str):
    row = conn.execute(
        "SELECT sum FROM daily_metrics WHERE metric = ? AND date = ?", (metric, date)
    ).fetchone()
    return row["sum"] if row else None


def arbitrated(conn, metric: str, date: str):
    clause, extra = dbmod._arbitration(conn, metric, date)
    return conn.execute(
        f"SELECT SUM(value) s FROM records WHERE metric = ? AND local_date = ?{clause}",
        (metric, date, *extra),
    ).fetchone()["s"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(LOCAL_DB_PATH))
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--show", type=int, default=10, help="changed days to print per metric")
    args = ap.parse_args()

    t0 = time.time()
    conn = dbmod.connect(args.db)
    pairs = dbmod.arbitrated_pairs(conn)
    print(f"db: {args.db}")
    print(f"{len(pairs)} (metric, day) pairs touched by arbitration "
          f"({time.time() - t0:.0f}s to find)\n")

    changed: dict[str, list] = {}
    for metric, date in pairs:
        cur, new = current(conn, metric, date), arbitrated(conn, metric, date)
        if cur is None or new is None or abs(cur - new) < 1e-6:
            continue
        changed.setdefault(metric, []).append((date, cur, new))

    total_removed = 0.0
    for metric, rows in sorted(changed.items()):
        removed = sum(c - n for _d, c, n in rows)
        total_removed += removed
        print(f"{metric}: {len(rows)} days change, {removed:,.0f} removed "
              f"({rows[0][0]} .. {rows[-1][0]})")
        for d, c, n in rows[:args.show]:
            print(f"    {d}  {c:12.2f} -> {n:12.2f}  ({(c - n) / n * 100:+.1f}%)"
                  if n else f"    {d}  {c:12.2f} -> {n}")
        if len(rows) > args.show:
            print(f"    ... {len(rows) - args.show} more")
        print()

    print(f"TOTAL: {sum(len(v) for v in changed.values())} days, "
          f"{total_removed:,.0f} phantom units removed")

    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply.")
        return 0

    written = dbmod.recompute_daily_metrics(conn, pairs=pairs)
    conn.commit()
    print(f"\napplied: {written} daily_metrics rows recomputed "
          f"in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
