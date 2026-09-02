"""hr_load — a session-scoped heart-rate load proxy.

NOT called TRIMP. The shape is Banister's, but HR_rest and HR_max here are
estimated from observational data rather than measured under the protocols the
published formula assumes, and HR during run/walk intervals is not
continuous-exercise intensity. See spec section 11.2.
"""
from __future__ import annotations

import pytest

from health_advisor import hr_load as hl


def test_session_load_rises_with_intensity():
    easy = hl.session_load(30.0, 130.0, 55.0, 185.0)
    hard = hl.session_load(30.0, 165.0, 55.0, 185.0)
    assert hard > easy


def test_session_load_rises_with_duration():
    short = hl.session_load(20.0, 145.0, 55.0, 185.0)
    long = hl.session_load(60.0, 145.0, 55.0, 185.0)
    assert long == pytest.approx(3.0 * short, rel=1e-6)


def test_session_load_includes_the_064_factor():
    """The published male formula is D * ratio * 0.64 * e^(1.92 * ratio). An
    implementation that drops 0.64 gives a number ~1.56x too large. Spec 11.4."""
    import math
    ratio = (145.0 - 55.0) / (185.0 - 55.0)
    expected = 30.0 * ratio * 0.64 * math.exp(1.92 * ratio)
    assert hl.session_load(30.0, 145.0, 55.0, 185.0) == pytest.approx(expected, rel=1e-9)


def test_session_load_is_none_when_an_input_is_missing():
    assert hl.session_load(30.0, None, 55.0, 185.0) is None
    assert hl.session_load(30.0, 145.0, None, 185.0) is None
    assert hl.session_load(None, 145.0, 55.0, 185.0) is None


def test_session_load_is_none_on_a_degenerate_hr_range():
    assert hl.session_load(30.0, 145.0, 185.0, 185.0) is None


def test_hr_below_rest_clamps_to_zero_not_negative():
    assert hl.session_load(30.0, 50.0, 55.0, 185.0) == 0.0


def test_a_day_with_no_hr_coverage_is_unknown_not_zero(tmp_path):
    """analysis.py line 62 records what zero-instead-of-unknown already cost:
    a week of non-wear reported as 'acwr 0.08, detraining'."""
    from health_advisor import db
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    c.execute("INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
              "duration_min, source, dedupe_key) VALUES ('running', "
              "'2026-07-01T12:00:00Z', '2026-07-01T12:40:00Z', '2026-07-01', 40.0, "
              "'test', 'w1')")
    c.commit()
    rows = hl.daily_load(c, "2026-07-01", "2026-07-01")
    assert len(rows) == 1
    assert rows[0]["status"] == "unknown"
    assert rows[0]["load"] is None
    assert rows[0]["sessions_without_hr"] == 1
    c.close()


def test_a_day_with_a_covered_session_reports_a_load(tmp_path):
    from datetime import datetime, timedelta, timezone
    from health_advisor import db
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    c.execute("INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
              "duration_min, source, dedupe_key) VALUES ('running', "
              "'2026-07-01T12:00:00Z', '2026-07-01T12:40:00Z', '2026-07-01', 40.0, "
              "'test', 'w1')")
    t0 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    for i in range(60):
        ts = (t0 + timedelta(seconds=i * 30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        c.execute("INSERT INTO records (metric, start_utc, end_utc, local_date, value, "
                  "unit, source, dedupe_key) VALUES ('heart_rate', ?, ?, '2026-07-01', "
                  "?, 'count/min', 'test', ?)", (ts, ts, 145.0, f"h{i}"))
    c.commit()
    rows = hl.daily_load(c, "2026-07-01", "2026-07-01",
                         hr_rest=55.0, hr_max=185.0)
    assert rows[0]["status"] == "ok"
    assert rows[0]["load"] > 0
    assert rows[0]["sessions"] == 1
    assert rows[0]["sessions_without_hr"] == 0
    c.close()


def test_nested_workout_is_skipped_and_outer_load_is_counted_once(tmp_path):
    from datetime import datetime, timedelta, timezone
    from health_advisor import db
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    c.execute("INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
              "duration_min, source, dedupe_key) VALUES ('running', "
              "'2026-07-01T12:00:00Z', '2026-07-01T12:37:54Z', '2026-07-01', "
              "37.9, 'test', 'outer')")
    c.execute("INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
              "duration_min, source, dedupe_key) VALUES ('running', "
              "'2026-07-01T12:10:00Z', '2026-07-01T12:11:18Z', '2026-07-01', "
              "1.3, 'test', 'inner')")
    t0 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    for i in range(0, 2274, 2):
        ts = (t0 + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        c.execute("INSERT INTO records (metric, start_utc, end_utc, local_date, value, "
                  "unit, source, dedupe_key) VALUES ('heart_rate', ?, ?, '2026-07-01', "
                  "145.0, 'count/min', 'test', ?)", (ts, ts, f"h{i}"))
    c.commit()
    rows = hl.daily_load(c, "2026-07-01", "2026-07-01", hr_rest=55.0, hr_max=185.0)
    assert rows[0]["load"] == pytest.approx(
        hl.session_load(37.9, 145.0, 55.0, 185.0), abs=0.005)
    assert rows[0]["sessions"] == 2
    assert rows[0]["sessions_nested_skipped"] == 1
    c.close()


def test_front_loaded_hr_samples_do_not_cover_a_long_session(tmp_path):
    from datetime import datetime, timedelta, timezone
    from health_advisor import db
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    c.execute("INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
              "duration_min, source, dedupe_key) VALUES ('running', "
              "'2026-07-01T12:00:00Z', '2026-07-01T14:00:00Z', '2026-07-01', "
              "120.0, 'test', 'w1')")
    t0 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    for i in range(30):
        ts = (t0 + timedelta(seconds=i * 20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        c.execute("INSERT INTO records (metric, start_utc, end_utc, local_date, value, "
                  "unit, source, dedupe_key) VALUES ('heart_rate', ?, ?, '2026-07-01', "
                  "145.0, 'count/min', 'test', ?)", (ts, ts, f"h{i}"))
    c.commit()
    rows = hl.daily_load(c, "2026-07-01", "2026-07-01", hr_rest=55.0, hr_max=185.0)
    assert rows[0]["status"] == "unknown"
    assert rows[0]["load"] is None
    assert rows[0]["sessions_without_hr"] == 1
    c.close()


def test_load_is_never_computed_from_whole_day_samples(tmp_path):
    """The load must come from workout windows only. HR recorded outside any
    workout — sleep, sitting, errands — must not contribute."""
    from datetime import datetime, timedelta, timezone
    from health_advisor import db
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    t0 = datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc)   # 03:00, no workout
    for i in range(200):
        ts = (t0 + timedelta(seconds=i * 30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        c.execute("INSERT INTO records (metric, start_utc, end_utc, local_date, value, "
                  "unit, source, dedupe_key) VALUES ('heart_rate', ?, ?, '2026-07-01', "
                  "?, 'count/min', 'test', ?)", (ts, ts, 60.0, f"h{i}"))
    c.commit()
    rows = hl.daily_load(c, "2026-07-01", "2026-07-01", hr_rest=55.0, hr_max=185.0)
    assert rows == [] or rows[0]["load"] in (None, 0.0)
    c.close()


# ---------------------------------------------------------------------------
# hr_load_proxy as an ACWR input.
#
# These four tests were written against scripts/compare_acwr_inputs.py, a
# one-off migration diagnostic that carried its OWN reimplementation of the
# acute:chronic ratio so the two candidate inputs could be compared before the
# swap. That script is not part of this repo, and the swap it existed to
# justify has since happened: analysis.training_load prefers hr_load_proxy.
# The invariants it protected are asserted here against the shipped engine
# instead, which is where they now live.
# ---------------------------------------------------------------------------


def test_acwr_names_the_input_it_actually_used_and_invents_no_band(conn):
    """No hr_load_proxy in the vault means no hr_load-derived ratio.

    The diagnostic reported `acwr_hr_load: None, band_hr_load: None` for such a
    day rather than quietly banding the other input's number. The engine's
    version of that honesty is `load_metric`: it falls back to active_energy
    and says so, and with nothing to fall back to it produces no band at all.
    """
    from datetime import date, timedelta
    from health_advisor import analysis as A
    from tests.conftest import seed_metric

    assert A.training_load(conn, as_of="2026-07-01") == {
        "status": "no_load_metric", "acwr": None}

    start = (date(2026, 7, 1) - timedelta(days=27)).isoformat()
    seed_metric(conn, "active_energy", start, [400.0] * 28)
    tl = A.training_load(conn, as_of="2026-07-01")
    assert tl["status"] == "ok"
    assert tl["load_metric"] == "active_energy"      # named, not assumed
    assert tl["acwr_band"] == "sweet-spot"
    assert conn.execute("SELECT COUNT(*) FROM daily_metrics WHERE metric = "
                        "'hr_load_proxy'").fetchone()[0] == 0


def test_acwr_refuses_a_window_with_too_few_measured_hr_load_days(conn):
    """Two measured days inside the window is not a chronic baseline.

    The gate is the same one hr_load.py's unknown-is-not-zero doctrine exists
    to feed: absent days stay absent, so a short history cannot be averaged
    into a ratio that reads as a training collapse.
    """
    from health_advisor import analysis as A
    from tests.conftest import seed_metric

    seed_metric(conn, "hr_load_proxy", "2026-06-20", [200.0])
    seed_metric(conn, "hr_load_proxy", "2026-07-01", [100.0])
    tl = A.training_load(conn, as_of="2026-07-01")
    assert tl["status"] == "insufficient_history"
    assert tl["acwr"] is None
    assert tl["n_days"] == 2
    assert tl.get("acwr_band") is None


def test_acwr_averages_over_measured_days_not_over_the_calendar(conn):
    """Both sides are means over the days that were measured, scaled to a week.

    25 measured days in the 28-day window: five in the acute week at 100, and
    twenty older days at 10. Mean-based, acute is 100*7 = 700 against a chronic
    weekly 196, so 3.57. Divide by the calendar instead — acute sum 500 against
    a 28-day sum scaled to a week (175) — and the same data reads 2.86, a
    quarter lower, because the unmeasured days silently vote zero.
    """
    from datetime import date, timedelta
    from health_advisor import analysis as A
    from tests.conftest import seed_metric

    end = date(2026, 7, 1)
    for i in range(5):                      # acute week: 5 of 7 days measured
        seed_metric(conn, "hr_load_proxy",
                    (end - timedelta(days=i)).isoformat(), [100.0])
    for i in range(7, 27):                  # chronic tail: 20 more days
        seed_metric(conn, "hr_load_proxy",
                    (end - timedelta(days=i)).isoformat(), [10.0])

    tl = A.training_load(conn, as_of="2026-07-01")
    assert tl["status"] == "ok"
    assert tl["load_metric"] == "hr_load_proxy"
    assert tl["n_days"] == 25 and tl["n_recent_days"] == 5
    assert tl["acute_7d"] == pytest.approx(700.0)
    assert tl["chronic_weekly_avg"] == pytest.approx(196.0)
    assert tl["acwr"] == pytest.approx(3.57, abs=0.005)
    assert tl["acwr"] != pytest.approx(2.86, abs=0.005)   # the calendar answer
    assert tl["acwr_band"] == "ramping-fast"


def test_hr_load_proxy_is_cataloged():
    from health_advisor import normalize as nz
    assert "hr_load_proxy" in nz.CATALOG
    entry = nz.CATALOG["hr_load_proxy"]
    assert entry["agg"] in ("sum", "mean", "last")
    assert entry.get("group")


def test_unknown_days_write_no_row_at_all(tmp_path):
    """Absence, not zero. A row of 0 would read as a rest day to every
    downstream consumer, including the ACWR that drives the back-off advice."""
    from health_advisor import db, derive
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    c.execute("INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
              "duration_min, source, dedupe_key) VALUES ('running', "
              "'2026-07-01T12:00:00Z', '2026-07-01T12:40:00Z', '2026-07-01', 40.0, "
              "'test', 'w1')")
    c.commit()
    derive.update_for_days(c, ["2026-07-01"])
    c.commit()
    row = c.execute("SELECT * FROM daily_metrics WHERE metric = 'hr_load_proxy' "
                    "AND date = '2026-07-01'").fetchone()
    assert row is None
    c.close()


def test_get_training_load_detail_rejects_a_bad_date(tools):
    out = tools.get_training_load_detail(start="nope")
    assert "error" in out


def test_get_training_load_detail_is_available_to_the_researcher():
    from health_advisor import llm
    assert "get_training_load_detail" in llm.RESEARCHER_TOOLS
