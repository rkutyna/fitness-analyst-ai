from datetime import date, timedelta

from health_advisor import analysis as A
from health_advisor import cold_start
from health_advisor import correlate as C
from tests.conftest import seed_metric


def test_cold_start_matches_the_real_surface_day_walk(conn):
    start = "2026-01-01"
    seed_metric(conn, "heart_rate_variability", start, [50 + i for i in range(90)])
    seed_metric(conn, "resting_heart_rate", start, [60 + i % 3 for i in range(90)])
    seed_metric(conn, "sleep_asleep", start, [450] * 90)
    seed_metric(conn, "active_energy", start, [500 + i for i in range(90)])
    seed_metric(conn, "step_count", start, [1000 + i for i in range(90)])

    first = {}
    for i in range(1, 91):
        as_of = (date.fromisoformat(start) + timedelta(days=i - 1)).isoformat()
        readiness = A.readiness(conn, as_of=as_of)
        load = A.training_load(conn, as_of=as_of)
        xs, ys, _ = C.paired_series(conn, "step_count", "active_energy", 0,
                                     start, as_of)
        correlation = C.correlate(xs, ys)
        if "readiness" not in first and readiness["status"] == "ok":
            first["readiness"] = i
        if "training_load" not in first and load["status"] == "ok":
            first["training_load"] = i
        if "correlate" not in first and correlation["status"] != "insufficient_data":
            first["correlate"] = len(xs)

    measured = cold_start.describe(conn, "2026-01-09",
                                   metric_x="step_count",
                                   metric_y="active_energy")
    assert measured["days_of_history"] == 9
    assert measured["readiness"]["starts_on_day"] == first["readiness"] == 17
    assert measured["training_load"]["starts_on_day"] == first["training_load"] == 21
    assert measured["correlate"]["starts_on_day"] == first["correlate"] == 8


def test_readiness_partial_names_missing_hrv(conn):
    seed_metric(conn, "resting_heart_rate", "2026-01-01", [60] * 90)
    out = A.readiness(conn, as_of="2026-03-31")
    assert out["status"] == "partial"
    assert out["status_text"] == (
        "partial: no HRV source in this vault; readiness needs both HRV and resting HR"
    )
    assert out["cold_start"]["status"] == "partial"


def test_movers_distinguishes_short_window_from_flat_window(conn):
    seed_metric(conn, "step_count", "2026-01-01", [1000] * 6)
    short = A.movers(conn, as_of="2026-01-06")
    assert short["status"] == "insufficient_history"
    assert short["status_text"] == (
        "movers starts on day 21 of data; you are on day 6 (2026-01-06)"
    )

    seed_metric(conn, "active_energy", "2026-01-01", [500] * 28)
    flat = A.movers(conn, as_of="2026-01-28")
    assert flat["status"] == "nothing_moved"
    assert flat["status_text"] == (
        "nothing has moved by more than 1.5 SD in the last 28 days"
    )


def test_empty_vault_publishes_no_data_blocks_not_empty_dicts(tmp_path):
    """A vault with zero records is the tester's day zero (found by the
    consumer suite: readiness() indexed status_text on an empty dict)."""
    import sqlite3
    from health_advisor import db, cold_start, analysis
    conn = sqlite3.connect(tmp_path / "empty.db")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    out = cold_start.describe(conn, "2026-09-09")
    assert out["days_of_history"] == 0
    for surface in ("readiness", "training_load", "correlate", "coverage", "movers"):
        block = out[surface]
        assert block["day_now"] == 0 and block["starts_on_day"] is None
        assert "no health data has synced yet" in block["status_text"]
    readiness = analysis.readiness(conn, "2026-09-09")
    assert readiness["status_text"].startswith("no health data has synced yet")


def test_probe_tolerates_a_daily_metrics_table_without_source_kind():
    import sqlite3
    from health_advisor import cold_start
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE daily_metrics (metric TEXT, date TEXT, count INTEGER, "
                 "sum REAL, avg REAL, min REAL, max REAL, last REAL, unit TEXT)")
    conn.execute("CREATE TABLE workouts (workout_type TEXT, local_date TEXT)")
    conn.execute("CREATE TABLE vault_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO daily_metrics VALUES ('resting_heart_rate','2026-09-01',1,60,60,60,60,60,'count/min')")
    probe = cold_start._probe(conn, "2026-09-09", "2026-10-01", {"resting_heart_rate"})
    assert probe.execute("SELECT count(*) FROM daily_metrics").fetchone()[0] >= 1
