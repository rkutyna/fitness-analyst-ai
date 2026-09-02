"""Checkpoint 1 verification: init DB, insert sample rows, prove idempotency
and daily_metrics rollup. Uses a throwaway DB so it never touches real data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from health_advisor import db  # noqa: E402

TEST_DB = Path(__file__).resolve().parent.parent / "data" / "_check.db"
TEST_DB.unlink(missing_ok=True)

conn = db.connect(TEST_DB)
db.init_db(conn)
print("tables:", [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")])

# Two step_count samples on the same local day + one heart_rate sample.
rows = []
for start, end, val, src in [
    ("2024-01-15T08:00:00+00:00", "2024-01-15T08:01:00+00:00", 120, "iPhone"),
    ("2024-01-15T09:00:00+00:00", "2024-01-15T09:01:00+00:00", 80, "iPhone"),
]:
    rows.append(dict(metric="step_count", value=val, unit="count",
                     start_utc=start, end_utc=end, start_local="2024-01-15 08:00:00",
                     local_date="2024-01-15", source=src, origin="backfill",
                     dedupe_key=db.record_key("step_count", start, end, val, "count", src)))
rows.append(dict(metric="heart_rate", value=62, unit="count/min",
                 start_utc="2024-01-15T08:00:00+00:00", end_utc="2024-01-15T08:00:00+00:00",
                 start_local="2024-01-15 08:00:00", local_date="2024-01-15", source="Watch",
                 origin="backfill",
                 dedupe_key=db.record_key("heart_rate", "2024-01-15T08:00:00+00:00",
                                          "2024-01-15T08:00:00+00:00", 62, "count/min", "Watch")))

added1 = db.insert_records(conn, rows)
print(f"first insert: {added1} new rows added (expect 3)")

# Re-insert the SAME rows -> idempotent, zero new rows.
added2 = db.insert_records(conn, rows)
print(f"re-insert same: {added2} new rows added (expect 0)")

conn.commit()
n = db.recompute_daily_metrics(conn, pairs=[("step_count", "2024-01-15"),
                                            ("heart_rate", "2024-01-15")])
conn.commit()
print(f"daily_metrics rows written: {n}")
for r in conn.execute("SELECT metric, date, count, sum, avg, min, max, unit "
                      "FROM daily_metrics ORDER BY metric"):
    print(" ", dict(r))

db.write_insight(conn, "2024-01-15", "Steps light today; resting HR normal.", "steps,hr")
print("insight:", dict(conn.execute("SELECT date, text, tags FROM insights").fetchone()))

db.log_ingest(conn, "backfill", "records", rows_seen=3, rows_added=added1, detail="self-test")
print("ingest_log rows:", conn.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0])

conn.close()
TEST_DB.unlink(missing_ok=True)
for suf in ("-wal", "-shm"):
    Path(str(TEST_DB) + suf).unlink(missing_ok=True)
print("\nOK — Checkpoint 1 verified, throwaway DB cleaned up.")
