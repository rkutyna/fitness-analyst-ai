#!/usr/bin/env python3
"""Migrate every record to the source-stable dedupe key, collapsing the
duplicates the un-migrated keys already let in.

Background. Commit 7097ac3 (2026-07-31 13:35) changed `db.record_key` to a
source-stable identity because the old key was derived from the canonical metric
name and unit, so any edit to normalize.py's catalog silently changed the key of
samples already stored. Its repair, scripts/repair_record_dedupe_stability.py,
is dry-run by default and was never run. As of 2026-08-16, **1,039,072 records**
still carry a key today's record_key does not reproduce.

Two consequences, and the second one is not hypothetical:

1. `insert_records` merges ON CONFLICT(dedupe_key). For an un-migrated row the
   conflict cannot fire, so the same sample arriving again is INSERTED rather
   than updated. "Re-backfill adds 0 rows" is false for those rows.

2. That already happened, on the afternoon of the deploy. **19,633 rows on
   2026-07-31 are exact duplicates** — same metric, timestamp, value, unit,
   source and origin, differing only in row id — because a batch already stored
   under the old key was re-sent after the change and matched nothing. The day's
   totals are inflated 3.3% to 36.9%: step_count 9,941 stored against 9,376 real,
   flights_climbed 21 against 15.3.

Why this script and not the existing one. repair_record_dedupe_stability.py
takes one metric and ABORTS when two rows would share a new key, which is the
correct refusal for a script whose job is only to re-key: collapsing rows means
deleting data. 26 of 77 metrics abort that way. This script does the deletion
deliberately, and only where the rows are genuinely identical.

The guard that matters. A collision group is collapsed ONLY when every row in it
agrees on value, unit, source, origin and local_date. A group whose values differ
is not a duplicate, and by default it is reported and skipped, never touched.

`--collapse-revisions` handles the one principled exception. On a metric whose
key EXCLUDES value (cumulative: the window is the identity), two rows with the
same window and source but different quantities are a revision, not two samples
— and insert_records already resolves revisions as "the newer send wins"
(ON CONFLICT ... SET value = excluded.value). With the flag, such a group keeps
its LATEST row, which is what would have been stored had the key matched. It
still refuses when the rows differ in unit, source, origin or local_date, and
when value is part of the metric's identity (there a collision means the key is
too weak, which is a different bug and must not be papered over). On the live
database this affected exactly 2 groups: one basal_energy, one stand_hour.

    ./.venv/bin/python scripts/repair_dedupe_key_migration.py                    # dry run
    ./.venv/bin/python scripts/repair_dedupe_key_migration.py --collapse-revisions --apply

Rehearse on a copy first — `sqlite3 data/health.db "VACUUM INTO '/tmp/x.db';"`
takes 17 s and the full run 6 minutes. The rehearsal is what caught the missing
re-derive step: recompute_daily_metrics does not touch metrics DERIVED from
records, and sleep_awakenings for 2026-07-31 read 9.0 because the duplicated
sleep_awake spans were each counted.

Back up first. One transaction, so an interrupted run rolls back.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod          # noqa: E402

IDENTITY_FIELDS = ("value", "unit", "source", "origin", "local_date")


def _new_key(row, metric: str) -> str:
    """The key today's ingest paths would compute for this row.

    source_metric/source_value are passed as the canonical metric and the stored
    value, which is what the retired Health Auto Export receiver path produced
    (its rows used record_key with no source overrides). Verified 2026-08-16
    against live rows of three origin/metric combinations.
    """
    return dbmod.record_key(row["metric"], row["start_utc"], row["end_utc"],
                            row["value"], row["unit"], row["source"] or "",
                            source_metric=metric, source_value=row["value"])


def survey(conn, collapse_revisions: bool = False) -> dict:
    """Group every record by the key it WOULD have. Read-only."""
    metrics = [r[0] for r in conn.execute(
        "SELECT DISTINCT metric FROM records ORDER BY metric")]
    rekey: list[tuple[str, int]] = []      # (new_key, id)
    collapse: list[tuple[int, list[int]]] = []   # (keep_id, [drop_ids])
    unsafe: list[dict] = []
    revisions: list[dict] = []
    for metric in metrics:
        owners: dict[str, list] = {}
        for row in conn.execute(
                "SELECT id, metric, value, unit, start_utc, end_utc, source, "
                "origin, local_date, dedupe_key FROM records WHERE metric = ? "
                "ORDER BY id", (metric,)):
            owners.setdefault(_new_key(row, metric), []).append(row)

        value_is_identity = dbmod._value_identifies_sample(metric)
        for key, group in owners.items():
            if len(group) > 1:
                identities = {tuple(r[f] for f in IDENTITY_FIELDS) for r in group}
                ids = [r["id"] for r in group]
                if len(identities) > 1:
                    others = {tuple(r[f] for f in IDENTITY_FIELDS if f != "value")
                              for r in group}
                    if value_is_identity or len(others) > 1 or not collapse_revisions:
                        # Either the rows differ in something other than value,
                        # or value is part of this metric's identity (so they are
                        # genuinely different samples and the collision is a key
                        # weakness), or we were not asked to resolve revisions.
                        unsafe.append({"metric": metric, "key": key, "ids": ids,
                                       "identities": sorted(identities, key=str)})
                        continue
                    # Same window, same source, different quantity, on a metric
                    # whose key excludes value: this is a REVISION, and
                    # insert_records resolves revisions as "the newer send
                    # wins" (ON CONFLICT ... SET value = excluded.value). Keep
                    # the latest row, which is what would have been stored had
                    # the key matched in the first place.
                    *drop, keep = ids
                    revisions.append({"metric": metric, "kept": keep, "dropped": drop})
                    collapse.append((keep, drop))
                    if group[-1]["dedupe_key"] != key:
                        rekey.append((key, keep))
                    continue
                keep, *drop = ids
                collapse.append((keep, drop))
                # The surviving row still needs the new key if it lacks it.
                if group[0]["dedupe_key"] != key:
                    rekey.append((key, keep))
            else:
                row = group[0]
                if row["dedupe_key"] != key:
                    rekey.append((key, row["id"]))
    return {"rekey": rekey, "collapse": collapse, "unsafe": unsafe,
            "revisions": revisions}


def affected_pairs(conn, drop_ids: list[int]) -> list[tuple[str, str]]:
    """(metric, local_date) pairs whose aggregate changes when rows are deleted."""
    pairs = set()
    for i in range(0, len(drop_ids), 500):
        chunk = drop_ids[i:i + 500]
        holes = ",".join("?" * len(chunk))
        for r in conn.execute(
                f"SELECT DISTINCT metric, local_date FROM records "
                f"WHERE id IN ({holes})", chunk):
            pairs.add((r["metric"], r["local_date"]))
    return sorted(pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(LOCAL_DB_PATH))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--collapse-revisions", action="store_true",
                    help="also collapse same-window groups that differ ONLY in "
                         "value, on metrics whose key excludes value, keeping "
                         "the latest row (insert_records' own upsert rule)")
    args = ap.parse_args()

    conn = dbmod.connect(args.db, read_only=not args.apply)
    before = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    plan = survey(conn, collapse_revisions=args.collapse_revisions)
    drop_ids = [i for _, drops in plan["collapse"] for i in drops]
    pairs = affected_pairs(conn, drop_ids) if drop_ids else []

    print(f"records                : {before:,}")
    print(f"keys to rewrite        : {len(plan['rekey']):,}")
    print(f"duplicate rows to drop : {len(drop_ids):,} "
          f"in {len(plan['collapse']):,} group(s)")
    print(f"aggregates to recompute: {len(pairs):,} (metric, date) pair(s)")
    print(f"revisions collapsed    : {len(plan['revisions']):,} group(s) "
          f"(latest kept)")
    for rev in plan["revisions"][:10]:
        print(f"  {rev['metric']}: kept {rev['kept']}, dropped {rev['dropped']}")
    print(f"refused as not-duplicate: {len(plan['unsafe']):,} group(s)")
    for u in plan["unsafe"][:10]:
        print(f"  {u['metric']} ids={u['ids']}")
        for ident in u["identities"]:
            print(f"    {ident}")
    if pairs:
        dates = sorted({d for _, d in pairs})
        print(f"dates touched          : {dates[0]} -> {dates[-1]} "
              f"({len(dates)} day(s))")

    if not args.apply:
        print("\ndry run — re-run with --apply to write.")
        return 0

    if plan["unsafe"]:
        print("\nrefusing to write while non-duplicate collisions exist; "
              "resolve them first.")
        return 2

    conn.execute("BEGIN")
    try:
        if drop_ids:
            for i in range(0, len(drop_ids), 500):
                chunk = drop_ids[i:i + 500]
                holes = ",".join("?" * len(chunk))
                conn.execute(f"DELETE FROM records WHERE id IN ({holes})", chunk)
        # Re-key after the deletes: a surviving row may be taking a key its
        # now-deleted twin was holding, and the column is UNIQUE.
        conn.executemany("UPDATE records SET dedupe_key = ? WHERE id = ?",
                         plan["rekey"])
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    if pairs:
        dbmod.recompute_daily_metrics(conn, pairs=pairs)
        conn.commit()
        # Derived metrics are computed FROM records, not aggregated from them, so
        # recompute_daily_metrics does not touch them. Found in rehearsal:
        # sleep_awakenings for 2026-07-31 read 9.0 because the duplicated
        # sleep_awake spans were each counted; it re-derives to 5.0.
        #
        # Re-derive in ascending date order over a superset that includes the 28
        # days before the earliest affected date. update_for_days deletes the
        # trailing-window metrics (midpoint SD, interval regularity, HR load) on
        # every run and only restores them when they recompute, and their inputs
        # are OTHER days' rows — so a re-derive over a bare subset can drop rows
        # it cannot rebuild. That is I2, which already ate 2026-08-06..08-08.
        from datetime import date, timedelta
        from health_advisor import derive
        days = sorted({d for _, d in pairs})
        first = date.fromisoformat(days[0]) - timedelta(days=28)
        span = [(first + timedelta(days=i)).isoformat()
                for i in range((date.fromisoformat(days[-1]) - first).days + 1)]
        derive.update_for_days(conn, span)
        conn.commit()
        print(f"re-derived {len(span)} day(s) "
              f"({span[0]} -> {span[-1]}, includes the 28-day lead-in)")

    after = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    print(f"\napplied. records {before:,} -> {after:,} "
          f"({before - after:,} removed); {len(plan['rekey']):,} keys rewritten; "
          f"{len(pairs):,} aggregate(s) recomputed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
