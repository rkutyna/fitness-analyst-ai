"""Checkpoint 4: exercise every MCP tool against the real DB and print compact
results. Calls the tool functions directly (no protocol layer)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import mcp_server as S  # noqa: E402
from health_advisor.context import VaultContext  # noqa: E402

# One session against the local checkout's vault. Every tool below is the bound
# form the model would be handed, so this exercises the real binding too.
M = SimpleNamespace(**S.build_tools(
    VaultContext.local(LOCAL_DB_PATH, writable=True)))


def show(title, obj, trunc=None):
    print(f"\n### {title}")
    s = json.dumps(obj, indent=2, default=str)
    if trunc and len(s) > trunc:
        s = s[:trunc] + f"\n  ... (truncated, full len {len(s)})"
    print(s)


m = M.list_available_metrics()
print(f"### list_available_metrics -> {m['count']} metrics")
for x in m["metrics"][:6]:
    print("   ", x)

show("get_daily_series(step_count, last 90d)",
     {**M.get_daily_series("step_count"), "points": "..."})
ds = M.get_daily_series("step_count")
print("   first 3 points:", ds["points"][:3], "downsampled:", ds["downsampled"])

show("get_daily_series(heart_rate, 1y -> downsample check)",
     {k: v for k, v in M.get_daily_series("heart_rate",
        start="2025-06-01", end="2026-06-11").items() if k != "points"})

show("summarize_metric(resting_heart_rate, 90d)", M.summarize_metric("resting_heart_rate", "90d"))
show("summarize_metric(step_count, 30d)", M.summarize_metric("step_count", "30d"))
show("summarize_metric(vo2_max, all)", M.summarize_metric("vo2_max", "all"))

show("compare_periods(step_count, 30d vs 30d-ago-explicit)",
     M.compare_periods("step_count", "2026-05-13:2026-06-11", "2026-04-13:2026-05-12"))

show("get_intraday(heart_rate, 2026-06-11)", M.get_intraday("heart_rate", "2026-06-11"))
show("get_intraday(step_count, 2026-06-11) [first buckets]",
     {**M.get_intraday("step_count", "2026-06-11"),
      "buckets": M.get_intraday("step_count", "2026-06-11")["buckets"][:5]})

show("list_workouts(last 90d, limit 5)", M.list_workouts(limit=5))
show("get_latest(vo2_max)", M.get_latest("vo2_max"))
show("get_latest(heart_rate)", M.get_latest("heart_rate"))

show("write_insight (test, then read back)",
     M.write_insight("2026-06-11", "TEST insight from tool test.", "test,checkpoint4"))

# error handling
show("unknown metric handling", M.summarize_metric("not_a_metric", "30d"))
show("bad date for write_insight", M.write_insight("not-a-date", "x"))

# cleanup the test insight
conn = M.db.connect("data/health.db")
conn.execute("DELETE FROM insights WHERE tags LIKE '%checkpoint4%'")
conn.commit(); conn.close()
print("\n(cleaned up test insight)")
print("\nALL TOOLS EXERCISED.")
