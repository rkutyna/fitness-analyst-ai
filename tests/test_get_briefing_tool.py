from tests.conftest import seed_metric


def test_get_briefing_returns_sections(conn, tools):
    seed_metric(conn, "step_count", "2026-05-01", [6000] * 23 + [9000] * 7)
    seed_metric(conn, "resting_heart_rate", "2026-05-01", [60] * 4)
    seed_metric(conn, "heart_rate_variability", "2026-05-01", [50] * 4)
    out = tools.get_briefing(scope="daily", day="2026-05-30")
    assert "talking_points" in out and "readiness" in out
    assert out["readiness"]["cold_start"]["status"] == "establishing_baseline"
    assert out["readiness"]["cold_start"]["starts_on_day"] == 17
