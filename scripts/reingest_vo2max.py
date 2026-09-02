#!/usr/bin/env python3
"""Recover VO2 max readings that were dropped at ingest, from an Apple Health CSV.

Between 2026-07-31 and 2026-08-16 every vo2_max reading was discarded: the source
began writing the unit as 'ml/(kg·min)' where normalize's table held 'mL/min·kg',
so convert_unit() raised and the point was dropped. Sixty consecutive receiver
batches, one point each. The unit lookup is fixed (F6-1) and live readings resume
on their own, but the sixteen days in between are only in Apple Health.

This reads the CSV that Health exports for a single metric:

    Date/Time,VO2 Max (ml/(kg·min)),Sources
    2026-08-01 15:40:07,37.28,<name>’s Apple Watch

Rows are keyed exactly as the receiver keys them -- db.record_key() over
(metric, start_utc, end_utc, value, source) -- so re-running this, or Health Auto
Export later re-sending the same reading, updates one row rather than adding a
second. Verified against the 2026-08-16 reading, which arrived through the live
path after the fix and whose key this script reproduces byte for byte.

Note the source string carries a NON-BREAKING space ('<name>’s Apple\xa0Watch').
It is part of the identity; normalising it to a plain space would silently
duplicate every row on the next live send.

Usage:
    ./.venv/bin/python scripts/reingest_vo2max.py --csv FILE            # dry run
    ./.venv/bin/python scripts/reingest_vo2max.py --csv FILE --apply    # writes
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db, normalize as nz          # noqa: E402

METRIC = "vo2_max"
ORIGIN = "backfill"        # honest provenance: a manual export, not the live feed

# Health's CSV writes local wall time with NO offset, while every other ingest
# path here relies on the timestamp carrying its own. Attach the zone rather than
# a fixed -04:00: the same export in November would be EST and every row would
# land an hour out, which for a metric Apple stamps mid-morning is a silent
# wrong-day error at the boundaries. The --verify pass below proves the choice.
LOCAL_TZ = ZoneInfo("America/New_York")
CSV_STAMP = "%Y-%m-%d %H:%M:%S"


def parse_csv(path: Path) -> list[dict]:
    """CSV rows -> record dicts, canonicalized the way the ingest path does."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if len(header) < 3 or "Date/Time" not in header[0]:
            raise SystemExit(f"unexpected header: {header!r}")
        raw_unit = header[1].split("(", 1)[1].rstrip(")") if "(" in header[1] else ""
        convert, unit = nz.unit_converter(METRIC, raw_unit)
        for line in reader:
            if not line or not line[0].strip():
                continue
            stamp, qty, source = line[0], line[1], line[2]
            dt = datetime.strptime(stamp.strip(), CSV_STAMP).replace(tzinfo=LOCAL_TZ)
            start_utc = end_utc = nz.to_utc_iso(dt)
            value = nz.canonical_value(METRIC, convert(float(qty)))
            rows.append(dict(
                metric=METRIC, value=value, unit=unit,
                start_utc=start_utc, end_utc=end_utc,
                start_local=nz.local_naive(dt), local_date=nz.local_date_of(dt),
                source=source, origin=ORIGIN,
                dedupe_key=db.record_key(METRIC, start_utc, end_utc, value,
                                         unit, source),
            ))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--db", default=LOCAL_DB_PATH)
    ap.add_argument("--apply", action="store_true",
                    help="write to the database (otherwise report and exit)")
    args = ap.parse_args()

    rows = parse_csv(args.csv)
    print(f"{len(rows)} readings in {args.csv.name}: "
          f"{rows[0]['local_date']} -> {rows[-1]['local_date']}")

    conn = db.connect(args.db)

    # Identity is (start_utc, value), NOT dedupe_key.
    #
    # dedupe_key looked like the obvious choice and is the wrong one: commit
    # 7097ac3 (2026-07-31) changed record_key to a source-stable identity, and
    # its repair script -- scripts/repair_record_dedupe_stability.py -- is
    # dry-run by default and was never run. So every record written before that
    # afternoon still carries an old-scheme key that today's record_key does not
    # reproduce. Keying the overlap check on dedupe_key therefore reports 27 of
    # 28 CSV readings as "new", 14 of which are already in the database, and
    # inserting them would duplicate the whole July series.
    #
    # A VO2 max sample is one instantaneous reading, so its timestamp and value
    # are its identity regardless of which key scheme wrote the row.
    stored = {(r["start_utc"], round(r["value"], 6)): r for r in conn.execute(
        "SELECT start_utc, start_local, value, dedupe_key FROM records "
        "WHERE metric = ?", (METRIC,))}

    def ident(r):
        return (r["start_utc"], round(r["value"], 6))

    overlap = [r for r in rows if ident(r) in stored]
    print(f"  overlap check  : {len(overlap)} of {len(rows)} already stored "
          f"(matched on timestamp + value)")
    if not overlap:
        raise SystemExit("ABORT: not one CSV reading matches a stored row. The "
                         "timezone or the parse is wrong — writing now would "
                         "duplicate the entire series.")
    for r in overlap:                       # wall-clock must agree too
        was = stored[ident(r)]
        if was["start_local"] != r["start_local"]:
            raise SystemExit(f"ABORT: {r['local_date']} local time disagrees "
                             f"(csv {r['start_local']}, db {was['start_local']}).")

    stale_keyed = sum(1 for r in overlap if stored[ident(r)]["dedupe_key"]
                      != r["dedupe_key"])
    if stale_keyed:
        print(f"  note           : {stale_keyed} stored rows carry pre-7097ac3 "
              f"keys (see comment above; not this script's business to fix)")

    new = [r for r in rows if ident(r) not in stored]
    days = sorted({r["local_date"] for r in new})
    print(f"  already stored : {len(rows) - len(new)}")
    print(f"  to insert      : {len(new)} across {len(days)} days")
    for d in days:
        vals = [r["value"] for r in new if r["local_date"] == d]
        print(f"    {d}  {', '.join(str(v) for v in vals)}")

    if not new:
        print("nothing to do.")
        return 0
    if not args.apply:
        print("\ndry run — re-run with --apply to write.")
        return 0

    added = db.insert_records(conn, new)
    pairs = [(METRIC, d) for d in days]
    touched = db.recompute_daily_metrics(conn, pairs=pairs)
    conn.commit()
    print(f"\ninserted {added} records; recomputed {touched} daily_metrics rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
