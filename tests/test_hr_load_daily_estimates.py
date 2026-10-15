"""daily_load estimates a day's HR references once, not once per workout (#110)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from health_advisor import db
from health_advisor import hr_load as hl

DAYS = ("2026-07-01", "2026-07-02")


def _hr_sample(c, ts, value, tag):
    stamp = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    c.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) "
        "VALUES ('heart_rate', ?, 'count/min', ?, ?, ?, ?, 't', 't', ?)",
        (value, stamp, stamp, stamp, stamp[:10], f"hr-{tag}-{stamp}"))


@pytest.fixture
def c(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    db.init_db(conn)
    for i, day in enumerate(DAYS):
        conn.execute(
            "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, "
            "last, unit) VALUES ('resting_heart_rate', ?, 1, ?, ?, ?, ?, ?, "
            "'count/min')", (day, 55 + i, 55 + i, 55 + i, 55 + i, 55 + i))
    # Three separate sessions on the first day, one on the second.
    for day, hour in ((DAYS[0], 6), (DAYS[0], 9), (DAYS[0], 14), (DAYS[1], 7)):
        t0 = datetime.fromisoformat(f"{day}T{hour:02d}:00:00")
        t1 = t0 + timedelta(minutes=20)
        conn.execute(
            "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
            "duration_min, source, dedupe_key) VALUES ('running', ?, ?, ?, 20.0, "
            "'t', ?)", (t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        t1.strftime("%Y-%m-%dT%H:%M:%SZ"), day,
                        f"w-{day}-{hour}"))
        for i in range(60):
            _hr_sample(conn, t0 + timedelta(seconds=20 * i),
                       130.0 + (hour % 5) * 6 + i % 7, f"w{hour}")
    # A trailing window rich enough for an HR-max estimate (needs >= 100).
    for i in range(150):
        _hr_sample(conn, datetime.fromisoformat("2026-06-20T10:00:00")
                   + timedelta(seconds=30 * i), 120.0 + i % 70, "bg")
    conn.commit()
    return conn


def _count_calls(monkeypatch):
    calls = {"rest": [], "peak": []}
    real_rest, real_peak = hl.hr_rest_estimate, hl.hr_max_estimate

    def rest(conn, day):
        calls["rest"].append(day)
        return real_rest(conn, day)

    def peak(conn, day):
        calls["peak"].append(day)
        return real_peak(conn, day)

    monkeypatch.setattr(hl, "hr_rest_estimate", rest)
    monkeypatch.setattr(hl, "hr_max_estimate", peak)
    return calls


def test_three_workouts_on_one_day_estimate_each_reference_once(c, monkeypatch):
    calls = _count_calls(monkeypatch)
    rows = hl.daily_load(c, DAYS[0], DAYS[0])
    assert rows[0]["sessions"] == 3 and rows[0]["load"] is not None
    assert calls == {"rest": [DAYS[0]], "peak": [DAYS[0]]}


def test_each_day_is_estimated_once_across_a_range(c, monkeypatch):
    calls = _count_calls(monkeypatch)
    hl.daily_load(c, DAYS[0], DAYS[1])
    assert sorted(calls["rest"]) == list(DAYS)
    assert sorted(calls["peak"]) == list(DAYS)


def test_explicit_references_are_never_estimated(c, monkeypatch):
    calls = _count_calls(monkeypatch)
    hl.daily_load(c, DAYS[0], DAYS[1], hr_rest=55.0, hr_max=185.0)
    assert calls == {"rest": [], "peak": []}


def test_the_payload_equals_per_workout_estimation(c):
    """Passing each day's own estimates in explicitly is the old per-workout
    computation; the memoised range call must publish the same rows."""
    expected = []
    for day in DAYS:
        expected += hl.daily_load(
            c, day, day, hr_rest=hl.hr_rest_estimate(c, day),
            hr_max=hl.hr_max_estimate(c, day))
    got = hl.daily_load(c, DAYS[0], DAYS[1])
    assert [r["load"] for r in got] and all(r["load"] for r in got)
    assert json.dumps(got, sort_keys=True) == json.dumps(expected, sort_keys=True)
