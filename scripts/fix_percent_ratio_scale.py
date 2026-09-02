#!/usr/bin/env python
"""One-off repair: rescale percent-ratio metrics stored as a 0–1 fraction to percent.

HealthKit reported OxygenSaturation and AppleWalkingSteadiness as a 0–1 ratio while
Health Auto Export sends them as 0–100 percent; the canonical unit is '%'. Ingest now
canonicalizes via normalize.canonical_value, but records imported before that fix stay
fractions (blood_oxygen_saturation 2026-06-09/10; walking_steadiness all backfill).

For each such record this multiplies value by 100 and recomputes its dedupe_key (which
encodes value, so a future re-backfill of the same sample still dedupes against it),
then re-aggregates the affected daily_metrics rows. Idempotent: only rows with
value <= 1.0 are touched, so a real percent value is never re-scaled and re-running is a
no-op. `main()` writes a timestamped DB backup before mutating.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime, timezone

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod
from health_advisor import normalize as nz

METRICS = tuple(sorted(nz.PERCENT_RATIO_METRICS))


def rescale_fraction_records(conn: sqlite3.Connection) -> dict:
    """Rescale every percent-ratio record with value <= 1.0 to percent (×100),
    recompute its dedupe_key, and re-aggregate the affected daily_metrics. Returns
    counts. Operates on the open connection; the caller owns backup + commit."""
    placeholders = ",".join("?" for _ in METRICS)
    rows = conn.execute(
        f"SELECT id, metric, value, unit, start_utc, end_utc, source, local_date "
        f"FROM records WHERE metric IN ({placeholders}) AND value <= 1.0",
        METRICS,
    ).fetchall()
    pairs: set[tuple[str, str]] = set()
    per_metric: dict[str, int] = {}
    for r in rows:
        new_value = r["value"] * 100.0
        new_key = dbmod.record_key(r["metric"], r["start_utc"], r["end_utc"],
                                   new_value, r["unit"], r["source"] or "")
        conn.execute("UPDATE records SET value = ?, dedupe_key = ? WHERE id = ?",
                     (new_value, new_key, r["id"]))
        pairs.add((r["metric"], r["local_date"]))
        per_metric[r["metric"]] = per_metric.get(r["metric"], 0) + 1
    written = dbmod.recompute_daily_metrics(conn, pairs=pairs)
    return {"records": len(rows), "pairs": len(pairs), "daily_rows": written,
            "per_metric": per_metric}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-path", default=str(LOCAL_DB_PATH))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    placeholders = ",".join("?" for _ in METRICS)
    ro = dbmod.connect(args.db_path, read_only=True)
    preview = ro.execute(
        f"SELECT metric, COUNT(*) n, MIN(value) mn, MAX(value) mx "
        f"FROM records WHERE metric IN ({placeholders}) AND value <= 1.0 GROUP BY metric",
        METRICS,
    ).fetchall()
    ro.close()
    total = sum(r["n"] for r in preview)
    print(f"Fraction records to rescale (value <= 1.0) across {METRICS}: {total}")
    for r in preview:
        print(f"  {r['metric']}: {r['n']} record(s), value {r['mn']}–{r['mx']} → ×100")
    if args.dry_run:
        print("dry-run: no changes written.")
        return
    if total == 0:
        print("nothing to do.")
        return

    backup = f"{args.db_path}.bak-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    shutil.copy2(args.db_path, backup)
    print(f"Backed up DB → {backup}")

    conn = dbmod.connect(args.db_path)
    try:
        stats = rescale_fraction_records(conn)
        conn.commit()
    finally:
        conn.close()
    print(f"Rescaled {stats['records']} record(s) "
          f"({stats['per_metric']}); recomputed {stats['daily_rows']} daily_metrics "
          f"row(s) over {stats['pairs']} (metric, day) pair(s).")


if __name__ == "__main__":
    main()
