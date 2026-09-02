#!/usr/bin/env python3
"""Populate `metric_source_months` in an existing database (T-006).

Optional. Nothing is wrong without it: `analysis.instrument_eras_status` reads
raw `records` when the derived table is empty, so era detection is correct on
any database that still has its samples. What this buys is speed — the
aggregate over 13.9M raw rows takes ~30 s, and `build_vault.py` pays it on every
build until the source carries the table. Run it once and every later vault
build reads a 2,197-row table instead.

Additive: creates one table and inserts into it. It writes nothing to `records`,
`daily_metrics`, or anything else. Still — this is a real database, so it is
dry-run by default and prints what it would write.

    ./.venv/bin/python scripts/build_source_provenance.py --db data/health.db
    ./.venv/bin/python scripts/build_source_provenance.py --db data/health.db --apply
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(LOCAL_DB_PATH))
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default: report only)")
    args = ap.parse_args(argv)

    conn = dbmod.connect(args.db, read_only=not args.apply)
    try:
        if args.apply:
            dbmod.init_db(conn)
        existing = conn.execute(
            "SELECT COUNT(*) FROM metric_source_months").fetchone()[0] \
            if _has_table(conn) else 0

        started = time.perf_counter()
        rows = conn.execute(
            "SELECT metric, substr(local_date, 1, 7) AS month, source, COUNT(*) n "
            "FROM records GROUP BY metric, month, source").fetchall()
        elapsed = time.perf_counter() - started

        metrics = {r["metric"] for r in rows}
        months = {r["month"] for r in rows}
        print(f"db          : {args.db}")
        print(f"existing    : {existing:,} rows")
        print(f"computed    : {len(rows):,} rows over {len(metrics)} metrics, "
              f"{len(months)} months ({elapsed:.1f} s)")

        if not args.apply:
            print("\ndry run — nothing written. Pass --apply to write.")
            return 0

        written = dbmod.rebuild_metric_source_months(conn, full=True)
        print(f"written     : {written:,} rows")
        return 0
    finally:
        conn.close()


def _has_table(conn) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='metric_source_months'").fetchone() is not None


if __name__ == "__main__":
    raise SystemExit(main())
