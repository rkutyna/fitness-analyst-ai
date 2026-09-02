#!/usr/bin/env python3
"""Fill (metric, local_date) pairs the receiver never delivered, from an export.zip.

Why not `backfill.run()`: a full re-run re-adds coarse backfill records for days
the receiver already replaced with fine samples, double-counting them (see the
docstring on backfill.run). This fills only pairs the DB has *zero* records for,
so a collision is impossible by construction — a pair with any existing record is
skipped and reported, never merged.

The live case it was written for: 2026-07-17 had 1,354 ActiveEnergyBurned records
in HealthKit and none in the DB, which took ACWR dark (analysis.training_load
reported insufficient_history, n_days=27).

    python -m scripts.repair_missing_days --zip export.zip \
        --metric active_energy --date 2026-07-17            # report only
    python -m scripts.repair_missing_days ... --apply       # write
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import backfill, db, derive

BATCH = 5000


def existing_pairs(conn, pairs):
    """Subset of `pairs` that already has at least one record row."""
    return {(m, d) for m, d in pairs
            if conn.execute("SELECT 1 FROM records WHERE metric = ? AND local_date = ? LIMIT 1",
                            (m, d)).fetchone()}


def day_sums(conn, pairs):
    """Current daily_metrics.sum per (metric, date) — the before/after evidence
    that a merge added the missing hours and did not double-count the present ones."""
    out = {}
    for m, d in pairs:
        r = conn.execute("SELECT sum, count FROM daily_metrics WHERE metric = ? AND date = ?",
                         (m, d)).fetchone()
        out[(m, d)] = (r["sum"], r["count"]) if r else (None, None)
    return out


def run(zip_path, metrics, dates, db_path=LOCAL_DB_PATH, apply=False,
        inner=backfill.INNER_XML, replace=False, xml_path=None) -> dict:
    """replace=False fills only pairs with ZERO existing records, so a collision is
    impossible by construction.

    replace=True rebuilds PARTIALLY covered days: the mid-July HAE outage left days
    holding only a few hours (2026-07-15 held 00:06-04:57 and 18.2 kcal against
    HealthKit's 828.8). Those cannot be *merged* — measured on 2026-07-16 and -20,
    ZERO of 2,614 export records share a dedupe_key with the receiver rows already
    stored, because the receiver samples far finer than the export (45,483 records
    for 07-23 vs the export's 3,351). Adding them would double-count: 07-16 would
    read 1,278 kcal against a true 790. So the day's existing records are DELETED
    and rebuilt from the export instead.

    This deletes real history. Two guards: a pair whose export side is empty is
    never touched (deleting data and putting nothing back is the one unrecoverable
    mistake here), and per-day sums are reported before and after."""
    conn = db.connect(db_path)
    db.init_db(conn)

    wanted = {(m, d) for m in metrics for d in dates}
    occupied = existing_pairs(conn, wanted)
    target = wanted if replace else wanted - occupied
    if replace:
        for m, d in sorted(occupied):
            print(f"  replace {m} {d}: existing records will be deleted", file=sys.stderr)
    else:
        for m, d in sorted(occupied):
            print(f"  skip {m} {d}: records already present", file=sys.stderr)
    if not target:
        conn.close()
        return dict(scanned=0, matched=0, added=0, skipped_occupied=len(occupied),
                    per_pair={}, before={}, after={})

    before = day_sums(conn, wanted)

    dates_of = {d for _, d in target}
    buf: list[dict] = []
    held: dict = {}          # replace mode: pair -> rows, kept until the scan proves non-empty
    per_pair: Counter = Counter()
    scanned = matched = added = deleted = 0

    with backfill.open_xml(zip_path, xml_path, inner) as stream:
        it = ET.iterparse(stream, events=("start", "end"))
        _, root = next(it)
        for event, elem in it:
            if event != "end":
                continue
            if elem.tag == "Record":
                scanned += 1
                # Cheap prefilter on the raw attribute before normalizing: the
                # export is ~1.2 GB and only a handful of days are wanted.
                if (elem.get("startDate") or "")[:10] in dates_of:
                    for row in backfill._record_rows(elem.attrib):
                        key = (row["metric"], row["local_date"])
                        if key in target:
                            (held.setdefault(key, []) if replace else buf).append(row)
                            per_pair[key] += 1
                            matched += 1
            elem.clear()
            # Replace holds everything: nothing may be deleted until the whole
            # export has been read and the replacement set is known to be complete.
            if not replace and len(buf) >= BATCH:
                if apply:
                    added += db.insert_records(conn, buf)
                buf.clear()
            if scanned % backfill.ROOT_CLEAR_EVERY == 0:
                root.clear()
                print(f"... {scanned:,} records scanned (matched {matched:,})", file=sys.stderr)

    if replace:
        empty = sorted(target - set(held))
        for m, d in empty:
            print(f"  REFUSED {m} {d}: export has no records — not deleting", file=sys.stderr)
        if apply:
            for key, rows in sorted(held.items()):
                origins = [r[0] for r in conn.execute(
                    "SELECT DISTINCT origin FROM records WHERE metric = ? AND local_date = ?",
                    key)]
                for o in origins:
                    deleted += db.delete_records_for_pairs(conn, [key], origin=o)
                added += db.insert_records(conn, rows)
    elif buf and apply:
        added += db.insert_records(conn, buf)

    if apply and matched:
        conn.commit()
        n = db.recompute_daily_metrics(conn, pairs=sorted(per_pair))
        derive.update_after_ingest(conn, sorted(dates_of), "backfill")
        conn.commit()
        db.log_ingest(conn, "repair", "records", matched, added,
                      detail=f"mode={'replace' if replace else 'fill'} "
                             f"pairs={sorted(per_pair)} deleted={deleted} "
                             f"daily_metric_rows={n}")
        conn.commit()

    after = day_sums(conn, wanted)
    conn.close()
    return dict(scanned=scanned, matched=matched, added=added, deleted=deleted,
                skipped_occupied=len(occupied), per_pair=dict(per_pair),
                before=before, after=after)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", dest="zip_path", default="export.zip")
    p.add_argument("--db", dest="db_path", default=str(LOCAL_DB_PATH))
    p.add_argument("--inner", default=backfill.INNER_XML)
    p.add_argument("--metric", action="append", required=True,
                   help="canonical metric name; repeatable")
    p.add_argument("--date", action="append", required=True,
                   help="local date YYYY-MM-DD; repeatable")
    p.add_argument("--apply", action="store_true",
                   help="write to the DB (default: report what would be added)")
    p.add_argument("--replace", action="store_true",
                   help="REBUILD partially covered days: delete the day's existing "
                        "records and re-ingest from the export. Deletes real history; "
                        "a pair the export cannot supply is refused, not emptied.")
    a = p.parse_args()

    s = run(a.zip_path, a.metric, a.date, db_path=a.db_path, apply=a.apply,
            inner=a.inner, replace=a.replace)
    print("\n" + ("Repair complete:" if a.apply else "Dry run (no writes):"))
    for k in ("scanned", "matched", "added", "deleted", "skipped_occupied"):
        print(f"  {k:20s} {s[k]:,}")
    for (m, d), n in sorted(s["per_pair"].items()):
        print(f"  {m} {d}: {n:,} records")
    if a.replace and s["before"]:
        print("\n  daily sum before -> after (each day should land ON the export")
        print("  total; a day that lands near before+export means a delete missed):")
        for k in sorted(s["before"]):
            b, _ = s["before"][k]
            aft, _ = s["after"][k]
            fmt = lambda v: "  --  " if v is None else f"{v:7.1f}"
            print(f"    {k[0]} {k[1]}: {fmt(b)} -> {fmt(aft)}")
    if not a.apply and s["matched"]:
        print("\nRe-run with --apply to write.")


if __name__ == "__main__":
    main()
