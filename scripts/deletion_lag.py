#!/usr/bin/env python3
"""How late does a HealthKit deletion arrive after the sample's own day?

This is the figure the compaction window has to be designed against (#37, D13).
Compacting to a lag shorter than the observed deletion tail discards raw that a
correction still needs, and the correction then has nothing to correct.

**It is not the same as late ARRIVAL**, which is what was measured for G-04
(p95 4-5 days, max 11.3) and which bounds a related but different behaviour.
ARCHITECTURE.md §D9 names mistaking one for the other as the way a compaction
window silently drops a correction, so this script deliberately reports only
deletions.

Undated tombstones are counted and excluded, never treated as zero: a deletion
for a sample this vault never held has no date to measure from, and folding it
in as "no lag" would pull the distribution toward zero — the direction that
makes an unsafely short window look safe.

    ./.venv/bin/python scripts/deletion_lag.py --vault PATH
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from health_advisor import db  # noqa: E402


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank; with a handful of deletions an interpolated value would
    read as more precision than the sample supports."""
    if not values:
        return float("nan")
    rank = max(1, min(len(values), round(pct / 100.0 * len(values))))
    return sorted(values)[rank - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vault", required=True, help="path to the vault to measure")
    ap.add_argument("--by-metric", action="store_true",
                    help="break the distribution down per metric")
    args = ap.parse_args()

    conn = db.connect(args.vault, read_only=True)
    try:
        have = {r["name"] for r in conn.execute("PRAGMA table_info(hk_deletions)")}
        if not have:
            print(f"{args.vault}: no hk_deletions table — not a HealthKit vault.")
            return 0
        if "sample_local_date" not in have:
            # Reported, not raised: a vault written before the instrumentation
            # existed is the expected case for a while, and its tombstones can
            # never be back-filled. Saying so is more useful than a traceback.
            print(f"{args.vault}: tombstones predate the deletion-lag columns.")
            print("The vault gains them by additive migration on its next")
            print("ingest under current code — restart the receiver. Deletions")
            print("recorded before that point are not recoverable.")
            return 0
        rows = conn.execute(
            "SELECT sample_local_date, sample_metric, deleted_at "
            "FROM hk_deletions").fetchall()
    finally:
        conn.close()

    dated, undated = [], 0
    for row in rows:
        if not row["sample_local_date"]:
            undated += 1
            continue
        deleted_day = dt.date.fromisoformat(row["deleted_at"][:10])
        sample_day = dt.date.fromisoformat(row["sample_local_date"])
        dated.append(((deleted_day - sample_day).days, row["sample_metric"]))

    print(f"tombstones      : {len(rows)}")
    print(f"  dated         : {len(dated)}")
    print(f"  undated       : {undated}  (sample never held here; excluded)")
    if not dated:
        print("\nNo dated deletions yet. This measurement is calendar-bound —")
        print("it accumulates only while the phone is syncing.")
        return 0

    lags = [lag for lag, _ in dated]
    print(f"\nlag in days from the sample's local date to the deletion:")
    print(f"  n             : {len(lags)}")
    print(f"  min / median  : {min(lags)} / {_percentile(lags, 50):g}")
    print(f"  p95 / max     : {_percentile(lags, 95):g} / {max(lags)}")

    if args.by_metric:
        per: dict[str, list[int]] = {}
        for lag, metric in dated:
            per.setdefault(metric or "(unknown)", []).append(lag)
        print("\nper metric:")
        for metric, values in sorted(per.items()):
            print(f"  {metric:28s} n={len(values):4d}  "
                  f"median={_percentile(values, 50):g}  max={max(values)}")

    print("\nThis is deletion lag, NOT arrival lag. Do not compare it to the")
    print("G-04 figures (p95 4-5 d, max 11.3 d) — those measure a different")
    print("behaviour, and ARCHITECTURE.md §D9 names confusing the two as how a")
    print("compaction window silently drops a correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
