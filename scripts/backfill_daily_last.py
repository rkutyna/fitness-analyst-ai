#!/usr/bin/env python3
"""Fill the additive daily_metrics.last column from raw records.

Dry-run is the default. Pass ``--apply`` to update only rows whose ``last`` is
NULL; all other daily aggregate columns are left untouched. Derived metrics
have no source records, so their existing one-value ``avg`` is used as the
equivalent last value. The operation is idempotent and safe to rerun.

    ./.venv/bin/python scripts/backfill_daily_last.py [--db PATH]
    ./.venv/bin/python scripts/backfill_daily_last.py --apply [--db PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod  # noqa: E402


def _targets(conn):
    """Materialize desired values, including arbitration, without table writes."""
    conn.execute("""
        CREATE TEMP TABLE rebuilt_last AS
        SELECT metric, local_date AS date, value AS last
        FROM (
            SELECT metric, local_date, value,
                   ROW_NUMBER() OVER (
                       PARTITION BY metric, local_date
                       ORDER BY start_utc DESC, end_utc DESC, id DESC
                   ) AS rn
            FROM records
        ) WHERE rn = 1
    """)
    # Match recompute_daily_metrics: cumulative days with source arbitration
    # must not get a last sample from a row excluded from their other columns.
    for metric, day in dbmod.arbitrated_pairs(conn):
        clause, extra = dbmod._arbitration(conn, metric, day)
        conn.execute(
            """
            UPDATE rebuilt_last SET last = (
                SELECT value FROM records
                WHERE metric = ? AND local_date = ?""" + clause + """
                ORDER BY start_utc DESC, end_utc DESC, id DESC LIMIT 1)
            WHERE metric = ? AND date = ?
            """,
            (metric, day, *extra, metric, day),
        )
    conn.execute("""
        CREATE TEMP TABLE daily_last_targets AS
        SELECT d.rowid AS daily_rowid, d.metric, d.date,
               CASE WHEN r.metric IS NULL THEN d.avg ELSE r.last END AS desired,
               d.avg
        FROM daily_metrics d
        LEFT JOIN rebuilt_last r ON r.metric = d.metric AND r.date = d.date
        WHERE d.last IS NULL
          AND CASE WHEN r.metric IS NULL THEN d.avg ELSE r.last END IS NOT NULL
    """)
    return conn.execute(
        "SELECT COUNT(*), SUM(desired IS NOT avg) FROM daily_last_targets"
    ).fetchone()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(LOCAL_DB_PATH))
    ap.add_argument("--apply", action="store_true", help="commit the backfill")
    args = ap.parse_args()

    conn = dbmod.connect(args.db, read_only=not args.apply)
    try:
        if args.apply:
            # The migration is explicit and only happens on the committing path.
            dbmod.init_db(conn)
        else:
            columns = {row["name"] for row in conn.execute(
                "PRAGMA table_info(daily_metrics)")}
            if "last" not in columns:
                raise RuntimeError(
                    "daily_metrics.last is absent; initialize the database before "
                    "running a dry-run")
        rows, changed = _targets(conn)
        print(f"rows that would change (NULL last): {rows:,}")
        print(f"rows whose last differs from avg: {changed or 0:,}")
        if args.apply and rows:
            conn.execute("""
                UPDATE daily_metrics
                SET last = (SELECT desired FROM daily_last_targets
                            WHERE daily_rowid = daily_metrics.rowid)
                WHERE rowid IN (SELECT daily_rowid FROM daily_last_targets)
            """)
            conn.commit()
            print(f"applied: {rows:,} daily_metrics.last values")
        elif args.apply:
            print("applied: 0 daily_metrics.last values")
        else:
            print("dry-run: no changes written")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
