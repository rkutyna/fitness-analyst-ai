#!/usr/bin/env python3
"""Repair legacy keys made unstable by canonical-unit normalization.

The current ingest code uses source-native identity. Rows already in the
database do not retain that identity separately, so this narrowly-scoped pass
uses the stored canonical metric and raw stored value as the best available
identity. It is intended for metrics whose source name is the canonical name
(notably ``dietary_energy_consumed``).

Dry-run is the default. ``--apply`` is required to write, and the pass aborts
if two existing rows would collide under the new key; it never silently drops
data. Rehearse on a copy first. This is deliberately not a whole-database
re-key: the production database has millions of records and a complete
source-identity migration requires retaining native IDs.

Example (after a backup and a copy rehearsal):

    ./.venv/bin/python scripts/repair_record_dedupe_stability.py \
        --db /path/to/health.db --metric dietary_energy_consumed --apply
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod  # noqa: E402


def _rows(conn, metric: str):
    return conn.execute(
        "SELECT id, metric, value, unit, start_utc, end_utc, source, dedupe_key "
        "FROM records WHERE metric = ? ORDER BY id", (metric,)
    )


def plan(conn, metric: str) -> tuple[list[tuple[int, str, str]], dict[str, list[int]]]:
    updates = []
    owners: dict[str, list[int]] = defaultdict(list)
    for row in _rows(conn, metric):
        new_key = dbmod.record_key(
            row["metric"], row["start_utc"], row["end_utc"], row["value"],
            row["unit"], row["source"] or "", source_metric=metric,
            source_value=row["value"],
        )
        owners[new_key].append(row["id"])
        if new_key != row["dedupe_key"]:
            updates.append((row["id"], row["dedupe_key"], new_key))
    collisions = {key: ids for key, ids in owners.items() if len(ids) > 1}
    return updates, collisions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(LOCAL_DB_PATH))
    ap.add_argument("--metric", default="dietary_energy_consumed")
    ap.add_argument("--apply", action="store_true", help="commit the re-key")
    args = ap.parse_args()

    conn = dbmod.connect(args.db, read_only=not args.apply)
    try:
        updates, collisions = plan(conn, args.metric)
        print(f"metric: {args.metric}")
        print(f"rows whose key would change: {len(updates):,}")
        print(f"stable-key collisions: {len(collisions):,}")
        if collisions:
            print("aborted: resolve collisions before applying")
            return 2
        if not args.apply:
            print("dry-run: no changes written")
            return 0
        conn.executemany(
            "UPDATE records SET dedupe_key = ? WHERE id = ?",
            [(new_key, row_id) for row_id, _old_key, new_key in updates],
        )
        conn.commit()
        print(f"applied: {len(updates):,} record keys")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
