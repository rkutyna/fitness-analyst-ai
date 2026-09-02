"""One-shot migration: rekey workouts to the source-independent dedupe_key.

`db.workout_key()` used to hash `source`, which the two ingest paths report
differently for the same physical session (export: "<name>'s Apple Watch";
HAE: "GymKit", "GymKit|<name>'s Apple Watch", "<name>'s iPhone "). Every full
export therefore re-added workouts the receiver already had.

This rewrites every workouts.dedupe_key with the new formula, merges the
duplicate pairs that the old formula let through, and re-points
workout_events.workout_key (recomputing each event's own dedupe_key) so
segments stay attached to the surviving row.

Merge rule: the survivor is the row carrying a reconciled avg_heart_rate
(the receiver writes it; the export does not), else the lowest id. Every
nullable column the survivor is missing is filled from its twin.

    python scripts/migrate_workout_keys.py --dry-run
    python scripts/migrate_workout_keys.py --apply
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_advisor import db  # noqa: E402

# Columns filled from the twin when the survivor's value is NULL.
MERGEABLE = ("duration_min", "energy_kcal", "distance_mi", "unit_distance",
             "source", "route_ref", "avg_heart_rate", "max_heart_rate")


def plan(conn):
    """Return (groups, event_updates, event_drops) without touching the DB."""
    rows = conn.execute(
        "SELECT id, workout_type, start_utc, end_utc, dedupe_key, "
        + ", ".join(MERGEABLE) + " FROM workouts ORDER BY id").fetchall()

    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[db.workout_key(r["workout_type"], r["start_utc"], r["end_utc"])].append(r)

    merges, rekeys = [], []
    old_to_new: dict[str, str] = {}
    for new_key, members in groups.items():
        survivor = next((m for m in members if m["avg_heart_rate"] is not None),
                        members[0])
        fills = {}
        for col in MERGEABLE:
            if survivor[col] is None:
                donor = next((m[col] for m in members if m[col] is not None), None)
                if donor is not None:
                    fills[col] = donor
        losers = [m for m in members if m["id"] != survivor["id"]]
        for m in members:
            old_to_new[m["dedupe_key"]] = new_key
        entry = dict(new_key=new_key, survivor=survivor, losers=losers, fills=fills)
        (merges if losers else rekeys).append(entry)

    # Re-point events; a merge can make two events collide on the new key.
    events = conn.execute(
        "SELECT id, workout_key, event_type, start_utc, duration_min, dedupe_key "
        "FROM workout_events ORDER BY id").fetchall()
    seen: set[str] = set()
    updates, drops, orphans = [], [], []
    for e in events:
        new_parent = old_to_new.get(e["workout_key"])
        if new_parent is None:
            orphans.append(e["id"])
            continue
        new_key = db.workout_event_key(new_parent, e["event_type"],
                                       e["start_utc"], e["duration_min"])
        if new_key in seen:
            drops.append(e["id"])
        else:
            seen.add(new_key)
            updates.append((new_parent, new_key, e["id"]))
    return merges, rekeys, updates, drops, orphans


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/health.db")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="explicit no-op; the default when --apply is absent")
    args = ap.parse_args()

    conn = db.connect(args.db)
    merges, rekeys, updates, drops, orphans = plan(conn)

    print(f"workouts: {len(merges) + len(rekeys)} sessions after migration "
          f"({len(merges)} merged from duplicates, {len(rekeys)} rekeyed in place)")
    for m in merges:
        s = m["survivor"]
        print(f"  MERGE {s['start_utc'][:10]} {s['workout_type']:<9} "
              f"keep id={s['id']} ({s['source']!r}) "
              f"drop {[l['id'] for l in m['losers']]} "
              f"fills={ {k: round(v, 3) if isinstance(v, float) else v for k, v in m['fills'].items()} }")
    print(f"workout_events: {len(updates)} re-pointed, {len(drops)} dropped as "
          f"post-merge duplicates, {len(orphans)} orphaned (left untouched)")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        conn.close()
        return

    with conn:  # single transaction; rolls back on any error
        # 1. losers go first, so the survivor can claim the shared new key
        loser_ids = [l["id"] for m in merges for l in m["losers"]]
        conn.executemany("DELETE FROM workouts WHERE id = ?", [(i,) for i in loser_ids])
        conn.executemany("DELETE FROM workout_events WHERE id = ?", [(i,) for i in drops])
        # 2. survivors: new key + any columns inherited from the twin
        for entry in merges + rekeys:
            s, fills = entry["survivor"], entry["fills"]
            sets = ", ".join(f"{c} = :{c}" for c in fills)
            sql = "UPDATE workouts SET dedupe_key = :k" + (f", {sets}" if sets else "") \
                  + " WHERE id = :id"
            conn.execute(sql, {"k": entry["new_key"], "id": s["id"], **fills})
        # 3. events follow their parent
        conn.executemany(
            "UPDATE workout_events SET workout_key = ?, dedupe_key = ? WHERE id = ?",
            updates)

    n_w = conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0]
    n_e = conn.execute("SELECT COUNT(*) FROM workout_events").fetchone()[0]
    linked = conn.execute(
        "SELECT COUNT(*) FROM workout_events e JOIN workouts w "
        "ON w.dedupe_key = e.workout_key").fetchone()[0]
    print(f"\nAPPLIED. workouts={n_w:,} workout_events={n_e:,} "
          f"(linked to a parent: {linked:,})")
    conn.close()


if __name__ == "__main__":
    main()
