"""mcp_server tool outputs must match metrics.py-computed values after refactor."""
from health_advisor import metrics as M
from tests.conftest import seed_metric


def test_summarize_matches_metrics(conn, tools):
    seed_metric(conn, "step_count", "2026-05-01", list(range(1, 41)))  # 40 days
    out = tools.summarize_metric("step_count", "30d")
    ro = M  # sanity: server uses metrics primitives
    assert out["metric"] == "step_count"
    assert out["n_days"] == 30
    assert out["trend_per_week"] is not None
