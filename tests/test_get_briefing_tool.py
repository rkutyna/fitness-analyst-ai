from tests.conftest import seed_metric


def test_get_briefing_returns_sections(conn, tools):
    seed_metric(conn, "step_count", "2026-05-01", [6000] * 23 + [9000] * 7)
    out = tools.get_briefing(scope="daily", day="2026-05-30")
    assert "talking_points" in out and "readiness" in out
