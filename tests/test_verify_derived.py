"""verify_daily_metrics.py must check the metrics it used to skip.

DERIVED_METRICS — sleep timing, wear hours, midpoint SD, interval regularity,
HR load — are computed straight into daily_metrics and have no rows in
`records`, so the rebuild-and-diff pass excludes them. That left the one script
that claims to verify daily_metrics unable to see the metrics the sleep
re-derive (E7-1) is about to rewrite, while the plan cited it as the safety
gate for exactly that work.

On its first run against the live database this pass found a real three-day
hole: 2026-08-06 through 08-08 carry sleep_bedtime but none of the three
trailing-window derived metrics, with both neighbours intact.
"""
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from health_advisor import db as dbmod
from health_advisor import derive


def _sleep_night(conn, wake_day: str, bedtime_hour: int = 23, hours: float = 7.5):
    """Write one night's sleep_asleep samples ending on the morning of wake_day."""
    wake = datetime.fromisoformat(f"{wake_day}T00:00:00+00:00") + timedelta(hours=7)
    start = wake - timedelta(hours=hours)
    t, n = start, 0
    while t < wake:
        end = min(t + timedelta(minutes=20), wake)
        mins = (end - t).total_seconds() / 60
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('sleep_asleep', ?, 'min', ?, ?, ?, ?, 'test', 'test', ?)",
            (mins, t.isoformat(), end.isoformat(),
             t.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d"),
             f"{wake_day}-{n}"))
        t, n = end, n + 1


@pytest.fixture
def derived_db(tmp_path):
    path = tmp_path / "derived.db"
    conn = dbmod.connect(path)
    dbmod.init_db(conn)
    for i in range(6):
        day = (datetime(2026, 8, 3, tzinfo=timezone.utc) + timedelta(days=i)).strftime("%Y-%m-%d")
        _sleep_night(conn, day)
    conn.commit()
    dbmod.recompute_daily_metrics(conn, full=True)
    derive.update_for_days(conn, derive.all_source_days(conn))
    conn.commit()
    return path, conn


def _run(path, *extra):
    return subprocess.run(
        [sys.executable, "scripts/verify_daily_metrics.py", "--db", str(path),
         "--derived-days", "0", *extra],
        capture_output=True, text=True)


def test_a_consistent_database_passes_both_passes(derived_db):
    path, conn = derived_db
    conn.close()
    out = _run(path)
    assert "derived metrics: re-derived" in out.stdout
    assert out.returncode == 0, out.stdout


def test_a_corrupted_derived_value_is_reported(derived_db):
    path, conn = derived_db
    conn.execute("UPDATE daily_metrics SET last = last + 3.0 "
                 "WHERE metric = 'sleep_bedtime'")
    conn.commit()
    conn.close()
    out = _run(path)
    assert out.returncode == 1
    assert "derived discrepancy" in out.stdout
    assert "sleep_bedtime" in out.stdout


def test_a_missing_derived_row_is_reported(derived_db):
    """The live defect: the row was never written, and nothing noticed."""
    path, conn = derived_db
    conn.execute("DELETE FROM daily_metrics WHERE metric = 'sleep_bedtime' "
                 "AND date = (SELECT MAX(date) FROM daily_metrics "
                 "            WHERE metric = 'sleep_bedtime')")
    conn.commit()
    conn.close()
    out = _run(path)
    assert out.returncode == 1
    assert "sleep_bedtime" in out.stdout


def test_the_old_pass_still_reports_a_records_level_discrepancy(derived_db):
    path, conn = derived_db
    conn.execute("UPDATE daily_metrics SET sum = sum + 100 "
                 "WHERE metric = 'sleep_asleep'")
    conn.commit()
    conn.close()
    out = _run(path)
    assert out.returncode == 1
    assert "vs a rebuild from records" in out.stdout
