"""Sleep-timing derivation: session merge, midnight wrap, naps, awakenings."""
from __future__ import annotations

from datetime import datetime, timedelta
import uuid

import pytest

from health_advisor import db as dbmod
from health_advisor import derive
from health_advisor import derive as D
from health_advisor import normalize as nz


def iv(metric: str, start: str, minutes: float) -> D.Interval:
    s = datetime.fromisoformat(start)
    return D.Interval(s, s + timedelta(minutes=minutes), metric)


def test_midnight_wrap_bedtime_and_wake():
    # bed 22:30 (prev day) -> wake 06:30: one asleep block across midnight
    out = D.compute_sleep_timing([iv("sleep_asleep", "2026-07-09 22:30:00", 480)],
                                 "2026-07-10")
    assert abs(out["sleep_bedtime"] - 10.5) < 1e-6      # hours since prev-day noon
    assert abs(out["sleep_wake_time"] - 6.5) < 1e-6     # hours since midnight
    assert abs(out["sleep_midpoint"] - 14.5) < 1e-6     # 02:30 = 14.5h after noon
    assert abs(out["sleep_time_in_bed"] - 480) < 1e-6


def test_after_midnight_bedtime_is_continuous():
    # bed 00:30 same day -> 12.5, NOT a wrap to a small number
    out = D.compute_sleep_timing([iv("sleep_asleep", "2026-07-10 00:30:00", 360)],
                                 "2026-07-10")
    assert abs(out["sleep_bedtime"] - 12.5) < 1e-6


def test_evening_doze_only_returns_none():
    # only sleep that day: 23:18-23:45 doze attributed to its own end-day
    out = D.compute_sleep_timing([iv("sleep_asleep", "2026-07-10 23:18:00", 27)],
                                 "2026-07-10")
    assert out is None


def test_nap_excluded_by_gap_merge():
    night = [iv("sleep_asleep", "2026-07-09 23:00:00", 420)]      # 23:00-06:00
    nap = [iv("sleep_asleep", "2026-07-10 14:00:00", 60)]         # 14:00-15:00
    out = D.compute_sleep_timing(night + nap, "2026-07-10")
    assert abs(out["sleep_wake_time"] - 6.0) < 1e-6               # nap ignored
    assert abs(out["sleep_time_in_bed"] - 420) < 1e-6


def test_awakenings_and_longest():
    ivs = [
        iv("sleep_asleep", "2026-07-09 23:00:00", 120),   # 23:00-01:00
        iv("sleep_awake",  "2026-07-10 01:00:00", 20),    # 01:00-01:20
        iv("sleep_asleep", "2026-07-10 01:20:00", 280),   # 01:20-06:00
        iv("sleep_awake",  "2026-07-10 03:00:00", 0.5),   # sub-minute: ignored
    ]
    out = D.compute_sleep_timing(ivs, "2026-07-10")
    assert out["sleep_awakenings"] == 1.0
    assert abs(out["sleep_awake_longest"] - 20) < 1e-6
    assert abs(out["sleep_latency"] - 0.0) < 1e-6
    assert abs(out["sleep_wake_time"] - 6.0) < 1e-6      # enveloped awake must not clip end
    assert abs(out["sleep_time_in_bed"] - 420) < 1e-6


def test_enveloped_interval_does_not_clip_session_end():
    # in_bed envelops the night; a late-starting short stage ends before in_bed
    ivs = [
        iv("sleep_in_bed",  "2026-07-09 22:00:00", 540),   # 22:00-07:00
        iv("sleep_asleep",  "2026-07-09 22:30:00", 480),   # 22:30-06:30
    ]
    out = D.compute_sleep_timing(ivs, "2026-07-10")
    assert abs(out["sleep_wake_time"] - 7.0) < 1e-6       # in_bed's end governs
    assert abs(out["sleep_time_in_bed"] - 540) < 1e-6


def test_latency_from_leading_awake():
    ivs = [
        iv("sleep_awake",  "2026-07-09 22:00:00", 30),
        iv("sleep_asleep", "2026-07-09 22:30:00", 480),
    ]
    out = D.compute_sleep_timing(ivs, "2026-07-10")
    assert abs(out["sleep_latency"] - 30) < 1e-6
    assert abs(out["sleep_bedtime"] - 10.0) < 1e-6       # session starts 22:00


def test_no_intervals_returns_none():
    assert D.compute_sleep_timing([], "2026-07-10") is None


def test_catalog_registered():
    expected_agg = {
        "sleep_bedtime": "mean",
        "sleep_wake_time": "mean",
        "sleep_midpoint": "mean",
        "sleep_time_in_bed": "mean",
        "sleep_awakenings": "mean",
        "sleep_awake_longest": "mean",
        "sleep_latency": "mean",
        "wear_hours": "mean",
        "sleep_midpoint_sd_28d": "mean",
        "sleep_timing_interval_regularity": "mean",
        "hr_load_proxy": "sum",
        # The plan's dial (E8-8). "last", not "sum" or "mean": the row IS the
        # day's value, already aggregated across the day's sessions — jog
        # minutes summed, longest block maxed — so re-aggregating it would be
        # aggregating an aggregate.
        "jog_minutes": "last",
        "longest_block_min": "last",
    }
    assert set(expected_agg) == set(D.DERIVED_METRICS)
    for m in D.DERIVED_METRICS:
        assert m in nz.CATALOG, m
        assert nz.agg_for(m) == expected_agg[m]
    assert nz.CATALOG["sleep_bedtime"]["group"] == "sleep_timing"
    assert nz.CATALOG["wear_hours"]["group"] == "coverage"


def _rec(conn, metric: str, start_local: str, minutes: float, local_date: str):
    """Insert one raw interval record (test helper; satisfies NOT NULLs)."""
    start_utc = start_local.replace(" ", "T") + "+00:00"   # tz value irrelevant here
    end_utc = start_utc                                     # not used by derive
    # Use a unique key per record to allow duplicates with the same timestamp
    dedupe_key = f"{metric}|{start_local}|{minutes}|{uuid.uuid4()}"
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, start_local, "
        "local_date, source, origin, dedupe_key) VALUES (?, ?, 'min', ?, ?, ?, ?, "
        "'test', 'receiver', ?)",
        (metric, minutes, start_utc, end_utc, start_local, local_date, dedupe_key))
    conn.commit()


def _seed_night(conn, day="2026-07-10"):
    _rec(conn, "sleep_asleep", "2026-07-09 23:00:00", 120, day)
    _rec(conn, "sleep_awake",  "2026-07-10 01:00:00", 20, day)
    _rec(conn, "sleep_asleep", "2026-07-10 01:20:00", 280, day)


def test_wear_hours_counts_distinct_hours(conn):
    for h in ("08", "08", "12", "23"):   # 3 distinct hours, one dup
        _rec(conn, "heart_rate", f"2026-07-10 {h}:15:00", 0, "2026-07-10")
    assert D.wear_hours(conn, "2026-07-10") == 3.0
    assert D.wear_hours(conn, "2026-07-11") is None


def test_update_for_days_writes_daily_metrics(conn):
    _seed_night(conn)
    _rec(conn, "heart_rate", "2026-07-10 08:00:00", 0, "2026-07-10")
    n = D.update_for_days(conn, ["2026-07-10"])
    conn.commit()
    assert n == 8   # 7 sleep metrics + wear_hours
    row = conn.execute("SELECT avg, unit FROM daily_metrics WHERE metric = ? AND date = ?",
                       ("sleep_bedtime", "2026-07-10")).fetchone()
    assert abs(row["avg"] - 11.0) < 1e-6     # 23:00 -> 11h after prev-day noon
    assert row["unit"] == "h"
    assert conn.execute(
        "SELECT 1 FROM daily_metrics WHERE metric = "
        "'sleep_timing_interval_regularity' AND date = '2026-07-10'"
    ).fetchone() is None


def test_update_for_days_idempotent_and_deletes_stale(conn):
    _seed_night(conn)
    D.update_for_days(conn, ["2026-07-10"])
    D.update_for_days(conn, ["2026-07-10"])   # re-run: same rows, no dupes
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM daily_metrics WHERE date = '2026-07-10'").fetchone()[0]
    assert n == 7    # no heart_rate seeded -> no wear_hours row
    conn.execute("DELETE FROM records WHERE metric LIKE 'sleep%'")
    D.update_for_days(conn, ["2026-07-10"])
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM daily_metrics WHERE date = '2026-07-10'").fetchone()[0]
    assert n == 0    # stale derived rows removed


def test_all_source_days(conn):
    _seed_night(conn, "2026-07-10")
    _rec(conn, "heart_rate", "2026-07-12 08:00:00", 0, "2026-07-12")
    assert sorted(D.all_source_days(conn)) == ["2026-07-10", "2026-07-12"]


def test_update_after_ingest_never_raises(conn, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("derive bug")
    monkeypatch.setattr(D, "update_for_days", boom)
    n = D.update_after_ingest(conn, ["2026-07-10"], "receiver")
    assert n == 0
    row = conn.execute("SELECT kind, detail FROM ingest_log ORDER BY id DESC LIMIT 1").fetchone()
    assert row["kind"] == "derive_error" and "derive bug" in row["detail"]


def test_update_after_ingest_happy_path(conn):
    _seed_night(conn)
    assert D.update_after_ingest(conn, ["2026-07-10"], "receiver") == 7


# --- a swallowed derive failure must still be visible ----------------------
# update_after_ingest is right to swallow: the raw records are the truth and
# losing a batch would be worse than losing a derived metric. But the receiver
# then returned ok:true with no hint, and nothing ever read the ingest_log row,
# so sleep timing, wear, regularity and hr_load_proxy could silently stop being
# written and the first sign would be a briefing quietly missing a number.

def test_update_after_ingest_reports_the_error_to_a_caller_that_asks(conn,
                                                                    monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("derive bug")
    monkeypatch.setattr(D, "update_for_days", boom)

    errors: list[str] = []
    n = D.update_after_ingest(conn, ["2026-07-10"], "receiver", errors=errors)

    assert n == 0
    assert len(errors) == 1
    assert "derive bug" in errors[0]


def test_update_after_ingest_reports_nothing_on_the_happy_path(conn):
    _seed_night(conn)
    errors: list[str] = []
    assert D.update_after_ingest(conn, ["2026-07-10"], "receiver",
                                 errors=errors) == 7
    assert errors == []


def test_update_after_ingest_still_works_without_the_errors_argument(conn,
                                                                    monkeypatch):
    """Every existing caller passes three positional args; that must keep working."""
    def boom(*a, **k):
        raise RuntimeError("derive bug")
    monkeypatch.setattr(D, "update_for_days", boom)
    assert D.update_after_ingest(conn, ["2026-07-10"], "receiver") == 0


# --- E8-8: the plan's dial gets a daily row ----------------------------------
# jog_minutes and longest_block_min were computed on demand from `records` and
# never stored. correlate.paired_series reads daily_metrics and nothing else, so
# no hypothesis, correlation or ACWR variant could reference the number the ramp
# is defined on — which is why none of the twelve pre-registered questions was
# about running.

def _run_workout(conn, day, start_h=10, minutes=20, pace_min=12.0, hr=140):
    """A synthetic run: one distance sample per 20s bucket at a steady pace."""
    from datetime import datetime, timedelta, timezone
    t0 = datetime(*[int(x) for x in day.split("-")], start_h, tzinfo=timezone.utc)
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, distance_mi, unit_distance, source, avg_heart_rate, dedupe_key) "
        "VALUES ('running', ?, ?, ?, ?, ?, 'mi', 'test', ?, ?)",
        (t0.isoformat(), (t0 + timedelta(minutes=minutes)).isoformat(), day,
         float(minutes), minutes / pace_min, float(hr), f"w-{day}-{start_h}"))
    per_bucket_mi = (20.0 / 60.0) / pace_min
    for i in range(int(minutes * 3)):
        t = t0 + timedelta(seconds=20 * i)
        for metric, value in (("distance_walking_running", per_bucket_mi),
                              ("heart_rate", float(hr)),
                              # The impact-volume dial now requires the
                              # cadence oracle as well as distance.
                              ("step_count", 47.0)):
            conn.execute(
                "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
                "start_local, local_date, source, origin, dedupe_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'test', 'test', ?)",
                (metric, value, "mi" if "distance" in metric else "count/min",
                 t.isoformat(), t.isoformat(), t.strftime("%Y-%m-%d %H:%M:%S"),
                 day, f"{metric}-{day}-{start_h}-{i}"))


def test_derive_writes_the_dial(tmp_path):
    conn = dbmod.connect(tmp_path / "dial.db")
    dbmod.init_db(conn)
    _run_workout(conn, "2026-08-20", minutes=20)
    conn.commit()
    dbmod.recompute_daily_metrics(conn, full=True)
    derive.update_for_days(conn, ["2026-08-20"])
    conn.commit()
    row = conn.execute("SELECT last FROM daily_metrics WHERE metric='longest_block_min' "
                       "AND date='2026-08-20'").fetchone()
    assert row and row["last"] == pytest.approx(20.0, abs=0.5)
    jog = conn.execute("SELECT last FROM daily_metrics WHERE metric='jog_minutes' "
                       "AND date='2026-08-20'").fetchone()
    assert jog and jog["last"] > 15.0


def test_the_dial_is_computed_not_copied(tmp_path):
    """A hard-coded implementation would pass the test above."""
    conn = dbmod.connect(tmp_path / "dial2.db")
    dbmod.init_db(conn)
    _run_workout(conn, "2026-08-20", minutes=9)
    conn.commit()
    dbmod.recompute_daily_metrics(conn, full=True)
    derive.update_for_days(conn, ["2026-08-20"])
    conn.commit()
    row = conn.execute("SELECT last FROM daily_metrics WHERE metric='longest_block_min' "
                       "AND date='2026-08-20'").fetchone()
    assert row["last"] == pytest.approx(9.0, abs=0.5)


def test_a_multi_session_day_takes_the_max_block_and_summed_minutes(tmp_path):
    """2026-07-17 carries two running workouts. The block is the longest one —
    the question it answers is how long he can run continuously — while jog
    minutes are the day's total."""
    conn = dbmod.connect(tmp_path / "dial3.db")
    dbmod.init_db(conn)
    _run_workout(conn, "2026-08-20", start_h=8, minutes=6)
    _run_workout(conn, "2026-08-20", start_h=17, minutes=14)
    conn.commit()
    dbmod.recompute_daily_metrics(conn, full=True)
    derive.update_for_days(conn, ["2026-08-20"])
    conn.commit()
    block = conn.execute("SELECT last FROM daily_metrics WHERE metric='longest_block_min' "
                         "AND date='2026-08-20'").fetchone()["last"]
    jog = conn.execute("SELECT last FROM daily_metrics WHERE metric='jog_minutes' "
                       "AND date='2026-08-20'").fetchone()["last"]
    assert block == pytest.approx(14.0, abs=0.5)      # max, not 20
    assert jog == pytest.approx(20.0, abs=1.0)        # sum


def test_a_day_with_no_running_writes_no_dial_row(tmp_path):
    """Absence, not zero — the wear_hours convention, not sleep_awakenings'."""
    conn = dbmod.connect(tmp_path / "dial4.db")
    dbmod.init_db(conn)
    derive.update_for_days(conn, ["2026-08-20"])
    conn.commit()
    for metric in derive.DIAL_METRICS:
        row = conn.execute("SELECT last FROM daily_metrics WHERE metric=? AND date=?",
                           (metric, "2026-08-20")).fetchone()
        assert row is None, metric


def test_the_dial_metrics_are_catalogued(tmp_path):
    """Uncatalogued they fall through as unmanaged with agg 'mean' and no unit,
    silently breaking value_col, the MCP metadata and every correlation."""
    from health_advisor import metrics as mx
    from health_advisor import normalize as nz
    for metric in derive.DIAL_METRICS:
        assert metric in nz.CATALOG, metric
        assert nz.CATALOG[metric]["unit"] == "min"
        assert mx.value_col(metric) == "last"


def test_the_dial_is_reachable_by_a_correlation(tmp_path):
    """The whole point of storing it: correlate reads daily_metrics only."""
    from health_advisor import correlate as C
    conn = dbmod.connect(tmp_path / "dial5.db")
    dbmod.init_db(conn)
    from datetime import date, timedelta
    d0 = date(2026, 7, 1)
    for i in range(12):
        _run_workout(conn, (d0 + timedelta(days=i)).isoformat(), minutes=8 + i)
    conn.commit()
    dbmod.recompute_daily_metrics(conn, full=True)
    derive.update_for_days(conn, [(d0 + timedelta(days=i)).isoformat() for i in range(12)])
    conn.commit()
    dates, vals, _ = mx_series(conn)
    assert len(vals) == 12 and max(vals) > min(vals)


def mx_series(conn):
    from health_advisor import metrics as mx
    return mx.series(conn, "longest_block_min", "2026-07-01", "2026-07-12")
