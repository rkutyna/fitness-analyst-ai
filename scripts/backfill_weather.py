#!/usr/bin/env python3
"""Run the scheduled workout weather and route-elevation enrichment.

The deployment owns scheduling (for example, a host timer); this command only
needs an explicit vault path. Packed ``workout_routes`` rows are preferred and
legacy GPX files are used only when no usable matched route row exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_advisor import db as dbmod  # noqa: E402
from health_advisor import weather as wx  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True,
                    help="SQLite vault to enrich")
    ap.add_argument("--routes-dir", default=None,
                    help="directory containing legacy GPX route references")
    ap.add_argument("--since", default=None,
                    help="local_date lower bound")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="perform no fetches and make no database writes")
    args = ap.parse_args(argv)

    conn = dbmod.connect(args.db)
    try:
        dbmod.init_db(conn)
        counts = wx.enrich_workouts(
            conn,
            since=args.since,
            limit=args.limit,
            dry_run=args.dry_run,
            routes_dir=args.routes_dir,
        )
    finally:
        conn.close()

    print("enrichment " + " ".join(f"{key}={counts[key]}" for key in (
        "fetched", "pending", "no_route", "elevation_filled", "fetch_calls")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
