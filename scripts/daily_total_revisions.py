#!/usr/bin/env python3
"""How many consolidated daily totals changed on a re-pull, and by how much?

The read side of health_advisor#220 Done-when 1: the settle lag N is chosen
from this distribution, not assumed. Prints `db.daily_total_revision_report`
as JSON. A settle lag at or below a metric's `last_change_lag` would have
frozen a value that later moved; `max_observed_lag` says how far the evidence
reaches. Read-only.

    ./.venv/bin/python scripts/daily_total_revisions.py --vault PATH
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from health_advisor import db  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vault", required=True, help="vault database path")
    args = ap.parse_args(argv)
    conn = db.connect(args.vault, read_only=True)
    try:
        report = db.daily_total_revision_report(conn)
    finally:
        conn.close()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
