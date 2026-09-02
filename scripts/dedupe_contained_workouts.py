#!/usr/bin/env python3
"""Remove workout rows that are sub-workouts of a session already stored (#150).

The phone emits some workouts as top-level sessions even when they are
fragments of another workout in the same batch — each with its own hk_uuid, so
nothing upstream refuses them. The 2026-08-25 treadmill run arrived as one
57.6-min row plus two 4-min, 0.2-mi rows nested inside it. `workout_key` is
type|start|end, so a contained row hashes differently from its container by
construction and no key over those fields can collide them; the ingest path now
refuses them (`db.contained_by`), and this script cleans up what was stored
before that landed.

**Same-source containment only.** Measured on data/health.db: 48 same-type
containment pairs exist, and 47 of them are CROSS-source — ErgData nested
inside the Apple Watch, mostly rowing, 2019-2021 — two devices legitimately
recording one session, both rows real historical record. This script refuses to
touch those, and prints them so the refusal is visible rather than assumed.
Sources are compared to each other, never to a literal: the stored strings
carry a curly apostrophe and a NO-BREAK SPACE.

Dry run by default; --apply writes. The vault path is a required argument —
there is no ambient default, because the vault this is aimed at is not the one
in this checkout.

BEFORE --apply ON A LIVE VAULT: back it up (scripts/backup_health.sh), and stop
the receiver. The live vault is single-writer with a DELETE journal.

    ./.venv/bin/python scripts/dedupe_contained_workouts.py /path/to/health.db
    ./.venv/bin/python scripts/dedupe_contained_workouts.py /path/to/health.db --apply
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_advisor import db as dbmod  # noqa: E402

# Same-type strict containment. `o.id <> i.id` plus the strictness clause keeps
# a row from containing itself and keeps identical spans (one session, one key)
# out of the result.
PAIRS_SQL = """
SELECT i.id            AS inner_id,
       i.workout_type  AS workout_type,
       i.local_date    AS local_date,
       i.start_utc     AS inner_start,
       i.end_utc       AS inner_end,
       i.duration_min  AS inner_min,
       i.distance_mi   AS inner_mi,
       i.source        AS inner_source,
       i.dedupe_key    AS inner_key,
       o.id            AS outer_id,
       o.start_utc     AS outer_start,
       o.end_utc       AS outer_end,
       o.duration_min  AS outer_min,
       o.distance_mi   AS outer_mi,
       o.source        AS outer_source
FROM workouts i
JOIN workouts o
  ON o.id <> i.id
 AND o.workout_type = i.workout_type
 AND o.start_utc <= i.start_utc
 AND o.end_utc   >= i.end_utc
 AND (o.start_utc < i.start_utc OR o.end_utc > i.end_utc)
{where}
ORDER BY i.local_date, i.start_utc
"""


def containment_pairs(conn: sqlite3.Connection, since: str | None = None,
                      until: str | None = None) -> list[dict]:
    where, params = [], []
    if since:
        where.append("i.local_date >= ?")
        params.append(since)
    if until:
        where.append("i.local_date <= ?")
        params.append(until)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return [dict(r) for r in conn.execute(PAIRS_SQL.format(where=clause), params)]


def split(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """(same-source, cross-source). Compared value-to-value; a blank source on
    either side is never "the same source" — it cannot prove anything."""
    same, cross = [], []
    for p in pairs:
        i_src, o_src = p["inner_source"] or "", p["outer_source"] or ""
        (same if (i_src and i_src == o_src) else cross).append(p)
    return same, cross


def _fmt(p: dict) -> str:
    return (f"  {p['local_date']}  {p['workout_type']:<10} "
            f"id={p['inner_id']:<5} {p['inner_start']} "
            f"{(p['inner_min'] or 0):6.2f}min {(p['inner_mi'] or 0):6.3f}mi"
            f"   INSIDE id={p['outer_id']} {p['outer_start']}..{p['outer_end']} "
            f"{(p['outer_min'] or 0):6.2f}min")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vault", help="path to the health.db to inspect (required; "
                                  "no default — back it up first)")
    ap.add_argument("--apply", action="store_true",
                    help="delete the fragment rows (default: dry run, read-only)")
    ap.add_argument("--since", help="only consider inner rows on/after this local_date")
    ap.add_argument("--until", help="only consider inner rows on/before this local_date")
    args = ap.parse_args()

    path = Path(args.vault)
    if not path.exists():
        print(f"no such vault: {path}", file=sys.stderr)
        return 2

    conn = dbmod.connect(path, read_only=not args.apply)
    try:
        total = conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0]
        pairs = containment_pairs(conn, args.since, args.until)
        same, cross = split(pairs)
        scope = " ".join(filter(None, [f"since={args.since}" if args.since else "",
                                       f"until={args.until}" if args.until else ""]))
        print(f"{path}: {total} workouts{(' [' + scope + ']') if scope else ''}")
        print(f"same-type containment pairs: {len(pairs)}  "
              f"(same-source {len(same)}, cross-source {len(cross)})")

        if cross:
            print(f"\nKEPT — {len(cross)} cross-source pair(s): two devices recording "
                  f"one session, both rows are real record.")
            for p in cross[:20]:
                print(_fmt(p) + f"   [{p['inner_source']!r} inside {p['outer_source']!r}]")
            if len(cross) > 20:
                print(f"  ... and {len(cross) - 20} more")

        # A row can be contained by more than one outer row (nested chains); the
        # delete set is the distinct inner ids, not the pair count.
        victims = sorted({p["inner_id"]: p for p in same}.items())
        print(f"\n{'DELETING' if args.apply else 'WOULD DELETE'} — "
              f"{len(victims)} same-source fragment row(s):")
        if not victims:
            print("  (none)")
        for _, p in victims:
            n_ev = conn.execute(
                "SELECT COUNT(*) FROM workout_events WHERE workout_key = ?",
                (p["inner_key"],)).fetchone()[0]
            print(_fmt(p) + f"   src={p['inner_source']!r}  events={n_ev}")

        if not args.apply:
            print("\ndry run — nothing written. Re-run with --apply (after a backup).")
            return 0

        for wid, p in victims:
            conn.execute("DELETE FROM workout_events WHERE workout_key = ?",
                         (p["inner_key"],))
            conn.execute("DELETE FROM workouts WHERE id = ?", (wid,))
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0]
        left = containment_pairs(conn, args.since, args.until)
        left_same, left_cross = split(left)
        print(f"\ndeleted {len(victims)} row(s): {total} -> {after} workouts")
        print(f"containment pairs now: {len(left)} "
              f"(same-source {len(left_same)}, cross-source {len(left_cross)})")
        if left_same:
            print("WARNING: same-source pairs remain — nested chains? re-run to check.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
