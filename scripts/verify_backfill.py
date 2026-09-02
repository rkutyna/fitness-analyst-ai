"""Checkpoint 2 verification: counts, date range, hand cross-checks of
daily_metrics against an independent re-aggregation straight from records,
and a look at workouts (energy/distance/route)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from health_advisor import db  # noqa

conn = db.connect("data/health.db", read_only=True)

def one(q, *a): return conn.execute(q, a).fetchone()

print("=== TABLE COUNTS ===")
for t in ("records", "workouts", "daily_metrics", "insights", "ingest_log"):
    print(f"  {t:15s} {one(f'SELECT COUNT(*) FROM {t}')[0]:,}")
print("  distinct metrics:", one("SELECT COUNT(DISTINCT metric) FROM records")[0])
r = one("SELECT MIN(local_date), MAX(local_date) FROM records")
print("  date range:", r[0], "->", r[1])

print("\n=== TOP METRICS BY ROW COUNT ===")
for row in conn.execute("SELECT metric, COUNT(*) c FROM records GROUP BY metric "
                        "ORDER BY c DESC LIMIT 12"):
    print(f"  {row['c']:>9,}  {row['metric']}")

print("\n=== CROSS-CHECK: daily_metrics vs raw records re-aggregation ===")
# Pick 3 (metric, date) pairs with data and recompute independently from records.
pairs = conn.execute("""
    SELECT metric, date FROM daily_metrics
    WHERE metric IN ('step_count','active_energy','heart_rate','sleep_asleep','resting_heart_rate')
      AND count > 1 ORDER BY date DESC LIMIT 6""").fetchall()
ok = True
for p in pairs:
    dm = one("SELECT count,sum,avg,min,max FROM daily_metrics WHERE metric=? AND date=?",
             p["metric"], p["date"])
    raw = one("SELECT COUNT(*) c, SUM(value) s, AVG(value) a, MIN(value) mn, MAX(value) mx "
              "FROM records WHERE metric=? AND local_date=?", p["metric"], p["date"])
    match = (dm["count"] == raw["c"] and abs((dm["sum"] or 0) - (raw["s"] or 0)) < 1e-6)
    ok = ok and match
    print(f"  [{'OK' if match else 'MISMATCH'}] {p['metric']} {p['date']}: "
          f"dm(count={dm['count']}, sum={dm['sum']:.2f}, avg={dm['avg']:.2f}) "
          f"raw(count={raw['c']}, sum={raw['s']:.2f})")

print("\n=== SAMPLE DAILY VALUES (most recent dates) ===")
for m in ("step_count", "active_energy", "resting_heart_rate", "heart_rate_variability",
          "vo2_max", "sleep_asleep", "sleep_in_bed"):
    row = conn.execute("SELECT date, count, sum, avg, min, max, unit FROM daily_metrics "
                       "WHERE metric=? ORDER BY date DESC LIMIT 1", (m,)).fetchone()
    if row:
        print(f"  {m:22s} {row['date']}  sum={row['sum']:.1f} avg={row['avg']:.1f} "
              f"min={row['min']:.1f} max={row['max']:.1f} n={row['count']} {row['unit']}")

print("\n=== WORKOUTS (recent 5) ===")
for w in conn.execute("SELECT workout_type, local_date, duration_min, energy_kcal, "
                      "distance_mi, route_ref FROM workouts ORDER BY start_utc DESC LIMIT 5"):
    rt = (w["route_ref"] or "").split("/")[-1]
    print(f"  {w['local_date']} {w['workout_type']:28s} {w['duration_min'] or 0:6.1f}min "
          f"{w['energy_kcal'] or 0:7.1f}kcal {w['distance_mi'] or 0:6.2f}mi {rt}")
print("  workouts with energy:", one("SELECT COUNT(*) FROM workouts WHERE energy_kcal IS NOT NULL")[0])
print("  workouts with route :", one("SELECT COUNT(*) FROM workouts WHERE route_ref IS NOT NULL")[0])

print("\nCROSS-CHECK RESULT:", "ALL OK" if ok else "MISMATCH FOUND")
conn.close()
