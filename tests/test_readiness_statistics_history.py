"""#696 E4: readiness on a vault whose HRV and resting-HR history is Apple daily
statistics only (no raw samples) for the baseline, with the last three days raw.

This is a new tester's situation after the history backfill. The oracles are
hand-written literals: HRV current 40.0 (raw, last 3 days), baseline 50.0
(statistic average, 28 days); resting HR baseline 60.0 (statistic most_recent).
The negative control drops the statistics and must fall back to
establishing_baseline, which shows the statistics are what made the difference.
"""
from __future__ import annotations

from datetime import date, timedelta

from health_advisor import analysis as A
from health_advisor import db as dbmod

_HRV = "heart_rate_variability"
_RHR = "resting_heart_rate"
_AS_OF = "2026-10-26"
_STAT_DAYS = [(date(2026, 9, 26) + timedelta(days=i)).isoformat()
              for i in range(28)]          # 2026-09-26 .. 2026-10-23
_RAW_DAYS = ["2026-10-24", "2026-10-25", "2026-10-26"]


def _seed_raw(conn, metric, unit, local_date, values):
    """Raw samples straight into `records`, then the ordinary recompute.
    Copied from tests/test_hk_daily_stats.py so the raw side is exact."""
    for i, value in enumerate(values):
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, dedupe_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'seed', ?)",
            (metric, value, unit, f"{local_date}T12:0{i}:00Z",
             f"{local_date}T12:0{i}:30Z", f"{local_date}T08:0{i}:00",
             local_date, f"seed-{metric}-{local_date}-{i}"))
    dbmod.recompute_daily_metrics(conn, pairs=[(metric, local_date)])
    conn.commit()


def _write_stats(conn, metric, unit, statistic, value):
    rows = [{"metric": metric, "local_date": day, "value": value,
             "unit": unit, "statistic": statistic, "device_id": "dev-1",
             "queried_at": "2026-10-26T09:00:00-04:00"} for day in _STAT_DAYS]
    dbmod.insert_backfill_daily_stats(conn, rows, batch_id="stats-history")
    dbmod.recompute_daily_metrics(
        conn, pairs=[(metric, day) for day in _STAT_DAYS])
    conn.commit()


def _seed_raw_last_three(conn):
    for day in _RAW_DAYS:
        _seed_raw(conn, _HRV, "ms", day, [40.0])
        _seed_raw(conn, _RHR, "count/min", day, [60.0])


def _factor(rd, component):
    matches = [f for f in rd["factors"] if f["component"] == component]
    assert len(matches) == 1, rd["factors"]
    return matches[0]


def test_statistics_history_scores_readiness(conn):
    """28 days of statistics plus 3 raw days: readiness is scored, and the
    HRV factor compares raw 40.0 against the statistic baseline 50.0."""
    _write_stats(conn, _HRV, "ms", "average", 50.0)
    _write_stats(conn, _RHR, "count/min", "most_recent", 60.0)
    _seed_raw_last_three(conn)

    rd = A.readiness(conn, as_of=_AS_OF)

    assert rd["status"] != "establishing_baseline", rd
    hrv = _factor(rd, "hrv")
    assert hrv["current"] == 40.0
    assert hrv["baseline"] == 50.0
    assert hrv["stale"] is False
    rhr = _factor(rd, "rhr")
    assert rhr["baseline"] == 60.0
    assert rhr["current"] == 60.0
    assert rhr["stale"] is False


def test_without_statistics_the_same_raw_days_establish_no_baseline(conn):
    """Negative control: identical raw days, no statistics rows. Three raw days
    are not a baseline, so the statistics are what made test 1 score."""
    _seed_raw_last_three(conn)

    rd = A.readiness(conn, as_of=_AS_OF)

    assert rd["status"] == "establishing_baseline", rd
    assert rd.get("score") is None
