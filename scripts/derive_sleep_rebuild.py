#!/usr/bin/env python3
"""Re-attribute sleep records by session and re-derive the sleep metrics (E7-1, E7-2).

`local_date` was assigned per sample as the date the SAMPLE ends. Correct while
HealthKit gave one span per night; wrong since 2026, when a night became 20-40
samples averaging 19.7 min and every sample ending before midnight went to the
PREVIOUS date. A day's sleep total became two half-nights and sleep_bedtime was
clipped at midnight.

This is a RE-DERIVE, not a re-ingest: the timestamps, values and dedupe_keys in
`records` are untouched. Only `local_date` — a column computed from those
timestamps — is corrected, and only for sleep stage records inside a
midnight-crossing episode.

    ./scripts/backup_health.sh
    ./.venv/bin/python scripts/derive_sleep_rebuild.py --dry-run --from 2016-01-01
    ./.venv/bin/python scripts/derive_sleep_rebuild.py --from 2016-01-01

Read the dry run before writing. 2021-2022 is the untested middle (61-75 min
samples, partially affected) — read that stretch specifically.

I2 guard: derived metrics with a 28-day trailing window (sleep_midpoint_sd_28d,
sleep_timing_interval_regularity, hr_load_proxy) read OTHER days' derived rows,
so the re-derive runs in ASCENDING date order over a SUPERSET that includes the
28 days before the earliest affected date. Verify immediately afterwards with
`scripts/verify_daily_metrics.py --derived-days 0`; anything it reports gets
re-derived before you commit.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod          # noqa: E402
from health_advisor import derive               # noqa: E402

WINDOW_PAD_DAYS = 28     # the longest trailing window any derived metric uses


def _span(conn, start: str) -> tuple[str, str]:
    ph = ",".join("?" * len(derive._STAGE_METRICS))
    row = conn.execute(
        f"SELECT MIN(local_date), MAX(local_date) FROM records "
        f"WHERE metric IN ({ph}) AND local_date >= ?",
        (*derive._STAGE_METRICS, start)).fetchone()
    return row[0], row[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(LOCAL_DB_PATH))
    ap.add_argument("--from", dest="start", default="2016-01-01")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the moves and change nothing")
    ap.add_argument("--limit", type=int, default=40, help="sample rows to print")
    args = ap.parse_args()

    conn = dbmod.connect(args.db, read_only=args.dry_run)
    lo, hi = _span(conn, args.start)
    if lo is None:
        print(f"no sleep records on or after {args.start}")
        return 0
    print(f"sleep records span {lo} .. {hi}")

    t0 = time.time()
    moves = derive.reattribute_sleep(conn, lo, hi, apply=not args.dry_run)
    by_year = Counter(old[:4] for _, old, _ in moves)
    forward = sum(1 for _, old, new in moves if new > old)
    print(f"{len(moves):,} record(s) change local_date ({time.time() - t0:.1f}s); "
          f"{forward:,} move forward, {len(moves) - forward:,} backward")
    for year in sorted(by_year):
        print(f"   {year}: {by_year[year]:,}")

    if not moves:
        print("nothing to do.")
        return 0

    days = sorted({d for _, old, new in moves for d in (old, new)})
    print(f"\n{len(days):,} affected date(s), {days[0]} .. {days[-1]}")
    print(f"sample of {min(args.limit, len(moves))}:")
    for rid, old, new in moves[:args.limit]:
        print(f"   record {rid}: {old} -> {new}")

    if args.dry_run:
        print("\ndry run — nothing written. Re-run without --dry-run to apply.")
        return 0

    # Both sides of every move: a record leaving date D makes D's stored rollup
    # too large and D+1's too small.
    pairs = sorted(derive.pairs_for_moves(moves))
    n = dbmod.recompute_daily_metrics(conn, pairs=pairs)
    conn.commit()
    print(f"\nrecomputed {n:,} daily_metrics pair(s) from records")

    # I2 guard, both directions. Backward: a trailing-window metric reads the 28
    # days BEFORE its own date, so the re-derive must start 28 days before the
    # earliest affected date or it computes from rows that are not there yet.
    # Forward: the same dependency means a moved bedtime keeps changing answers
    # for 28 days AFTER it, so stopping at the last affected date leaves stale
    # rows behind — measured, that is exactly what happened on the first
    # rehearsal (2026-08-16's midpoint SD and regularity, caught by
    # verify_daily_metrics.py --derived-days 0). Run to the end of the series.
    first = (date.fromisoformat(days[0]) - timedelta(days=WINDOW_PAD_DAYS)).isoformat()
    superset = sorted(d for d in derive.all_source_days(conn) if d >= first)
    t1 = time.time()
    written = derive.update_for_days(conn, superset)
    conn.commit()
    print(f"re-derived {written:,} value(s) across {len(superset):,} day(s) "
          f"({time.time() - t1:.1f}s), from {superset[0] if superset else '-'}")
    print("\nNow run:  ./.venv/bin/python scripts/verify_daily_metrics.py --derived-days 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
