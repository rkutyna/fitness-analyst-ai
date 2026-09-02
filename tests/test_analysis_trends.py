from health_advisor import analysis as A
from tests.conftest import seed_metric


def test_early_warning_fires_on_rhr_up_hrv_down(conn):
    seed_metric(conn, "resting_heart_rate", "2026-05-10", [50 + i for i in range(21)])  # rising
    seed_metric(conn, "heart_rate_variability", "2026-05-10", [80 - i for i in range(21)])  # falling
    tr = A.trends(conn, as_of="2026-05-30")
    assert tr["early_warning"]["flag"] is True


def test_early_warning_quiet_on_noise(conn):
    seed_metric(conn, "resting_heart_rate", "2026-05-10", [55, 54, 56, 55, 54, 56] * 4)
    seed_metric(conn, "heart_rate_variability", "2026-05-10", [60, 61, 59, 60, 61, 59] * 4)
    tr = A.trends(conn, as_of="2026-06-02")
    assert tr["early_warning"]["flag"] is False
