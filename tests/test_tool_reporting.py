"""The smaller tool-contract defects from audit P3-8.

Each of these is a tool answering confidently with something other than what it
was asked for: a weekly mean labelled as a total, a parameter that does nothing,
a truncated list that looks complete, a scope the server never understood. The
agent cannot detect any of them from the outside, which is what makes them
worth a test apiece.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from health_advisor import metrics as M
from tests.conftest import seed_metric, seed_workout


# --------------------------------------------------------------------------- #
# get_daily_series downsampling: a week of a cumulative metric is a SUM
# --------------------------------------------------------------------------- #
def test_downsample_sums_a_cumulative_metric():
    """The live bug: 2026-07-20's week of step_count totalled 55,376 and came
    back as 7,910.89 — the daily mean, under a tool whose own output says
    agg='sum'."""
    dates = [(datetime(2026, 7, 20) + timedelta(days=i)).date().isoformat()
             for i in range(7)]
    vals = [7132.46, 5190.69, 9188.76, 8296.94, 12272.18, 4470.31, 8824.91]
    (pt,) = M.downsample_weekly(dates, vals, "sum")
    assert pt["date"] == "2026-07-20"
    assert pt["value"] == pytest.approx(55376.25, abs=0.05)
    assert pt["days"] == 7


def test_downsample_means_a_rate_metric():
    dates = [(datetime(2026, 7, 20) + timedelta(days=i)).date().isoformat()
             for i in range(7)]
    (pt,) = M.downsample_weekly(dates, [60.0] * 6 + [74.0], "mean")
    assert pt["value"] == pytest.approx(62.0, abs=0.01)


def test_downsample_takes_the_week_s_final_reading_for_a_last_metric():
    """body_mass's daily aggregate is the day's last weigh-in; the week's is the
    week's last weigh-in, not an average of them."""
    dates = ["2026-07-20", "2026-07-22", "2026-07-24"]
    (pt,) = M.downsample_weekly(dates, [180.0, 179.0, 177.0], "last")
    assert pt["value"] == 177.0
    assert pt["days"] == 3


def test_downsample_marks_a_partial_week():
    """A partial week's SUM is not comparable to a full week's, so the day count
    travels with the point."""
    pts = M.downsample_weekly(["2026-07-24", "2026-07-25"], [10.0, 10.0], "sum")
    assert pts[0]["days"] == 2


def test_daily_series_downsampled_week_totals_match_the_days(conn, tools):
    seed_metric(conn, "step_count", "2024-01-01", [1000.0] * 500)
    out = tools.get_daily_series("step_count", start="2024-01-01", end="2025-05-14")
    assert out["downsampled"] is True
    assert out["downsample_agg"] == "sum"
    full = [p for p in out["points"] if p["days"] == 7]
    assert full and all(p["value"] == 7000.0 for p in full)


# --------------------------------------------------------------------------- #
# get_intraday: bucket_hours was a dead parameter
# --------------------------------------------------------------------------- #
def _seed_hr_hours(conn, day: str, per_hour: dict[int, float]):
    for hour, value in per_hour.items():
        ts = f"{day}T{hour:02d}:30:00+00:00"
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('heart_rate', ?, 'count/min', ?, ?, ?, ?, 't', 't', ?)",
            (value, ts, ts, f"{day}T{hour:02d}:30:00", day, f"h-{day}-{hour}"))
    conn.commit()


@pytest.fixture
def intraday_db(conn):
    # daily_metrics too: get_intraday gates on _metric_exists, which reads it.
    seed_metric(conn, "heart_rate", "2026-07-26", [85.0])
    _seed_hr_hours(conn, "2026-07-26", {0: 60.0, 1: 70.0, 2: 80.0,
                                        3: 90.0, 4: 100.0, 5: 110.0})


def test_intraday_bucket_hours_actually_buckets(intraday_db, tools):
    """Verified in the audit: output was byte-identical for 1 and 3."""
    one = tools.get_intraday("heart_rate", "2026-07-26", bucket_hours=1)
    three = tools.get_intraday("heart_rate", "2026-07-26", bucket_hours=3)
    assert len(one["buckets"]) == 6
    assert len(three["buckets"]) == 2
    assert three["buckets"][0]["hour"] == 0
    assert three["buckets"][0]["value"] == pytest.approx(70.0)   # mean of 60/70/80
    assert three["bucket_hours"] == 3


def test_intraday_rejects_a_bucket_size_that_does_not_divide_the_day(intraday_db, tools):
    for bad in (0, 5, 25, -1):
        assert "error" in tools.get_intraday("heart_rate", "2026-07-26", bucket_hours=bad)


def test_intraday_sums_a_cumulative_metric_across_the_bucket(conn, tools):
    """`distance_walking_running` rather than `step_count` because T-006 makes
    get_intraday refuse series outside the D3 raw allowlist. Both are
    cumulative, which is the property being pinned."""
    seed_metric(conn, "distance_walking_running", "2026-07-26", [300.0])
    for hour in (0, 1, 2):
        ts = f"2026-07-26T{hour:02d}:30:00+00:00"
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) VALUES "
            "('distance_walking_running', 100, 'mi', ?, ?, ?, '2026-07-26', "
            "'t', 't', ?)",
            (ts, ts, f"2026-07-26T{hour:02d}:30:00", f"s-{hour}"))
    conn.commit()
    out = tools.get_intraday("distance_walking_running", "2026-07-26",
                             bucket_hours=3)
    assert out["buckets"][0]["value"] == 300.0


# --------------------------------------------------------------------------- #
# list_workouts: a truncated list must say so
# --------------------------------------------------------------------------- #
@pytest.fixture
def many_workouts(conn):
    for i in range(12):
        seed_workout(conn, "running", f"2026-07-{i + 1:02d}", 30.0, 3.0)


def test_list_workouts_flags_truncation(many_workouts, tools):
    out = tools.list_workouts(start="2026-07-01", end="2026-07-31", limit=5)
    assert out["count"] == 5
    assert out["truncated"] is True
    assert out["total_in_range"] == 12
    assert out["workout_counts"] == [{"type": "running", "count": 12}]
    assert "note" in out


def test_list_workouts_does_not_cry_truncation_when_complete(many_workouts, tools):
    out = tools.list_workouts(start="2026-07-01", end="2026-07-31", limit=50)
    assert out["truncated"] is False
    assert out["total_in_range"] == 12
    assert out["workout_counts"] == [{"type": "running", "count": 12}]


# --------------------------------------------------------------------------- #
# get_briefing: an unknown scope was echoed back as if it had been honoured
# --------------------------------------------------------------------------- #
def test_get_briefing_rejects_an_unknown_scope(conn, tools):
    seed_metric(conn, "step_count", "2026-07-01", [8000.0] * 20)
    out = tools.get_briefing(scope="weekly")
    assert "error" in out
    assert "daily" in out["error"] and "deep" in out["error"]


def test_get_briefing_still_accepts_its_two_scopes(conn, tools):
    seed_metric(conn, "step_count", "2026-07-01", [8000.0] * 20)
    for scope in ("daily", "deep"):
        assert "error" not in tools.get_briefing(scope=scope)


# --------------------------------------------------------------------------- #
# log_subjective: a check-in cannot be about a day that hasn't happened
# --------------------------------------------------------------------------- #
def test_log_subjective_rejects_a_future_day(conn, tools):
    future = (datetime.now().date() + timedelta(days=1)).isoformat()
    out = tools.log_subjective(day=future, energy=4)
    assert out["ok"] is False
    assert "future" in out["error"]


def test_log_subjective_still_accepts_today(conn, tools):
    today = datetime.now().date().isoformat()
    assert tools.log_subjective(day=today, energy=4)["ok"] is True


def test_log_subjective_rejects_a_malformed_day(conn, tools):
    out = tools.log_subjective(day="yesterday", energy=4)
    assert out["ok"] is False


# --------------------------------------------------------------------------- #
# get_subjective is a READ tool — it must not open the DB writable
# --------------------------------------------------------------------------- #
def test_get_subjective_opens_the_db_read_only(conn, tools, monkeypatch):
    """It ran init_db (DDL) on the 3.6 GB production database on every read."""
    from health_advisor import db as dbmod

    opened: list[bool] = []
    real = dbmod.connect

    def spy(path=None, *, read_only=False):
        opened.append(read_only)
        return real(path, read_only=read_only)

    monkeypatch.setattr(dbmod, "connect", spy)
    out = tools.get_subjective("2026-07-01", "2026-07-07")
    assert opened == [True]
    assert out["count"] == 0


def test_get_subjective_survives_a_db_without_the_table(vault_path, tools):
    """Read-only means it can no longer CREATE TABLE its way out of a missing
    table, so the missing table has to be an answer rather than a crash."""
    p = vault_path
    import sqlite3
    sqlite3.connect(p).close()
    out = tools.get_subjective("2026-07-01", "2026-07-07")
    assert out["count"] == 0
    assert out["days"] == []
