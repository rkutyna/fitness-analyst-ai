"""Workout heart-rate reconciliation against the raw sample series.

The retired phone path occasionally sent a workout summary whose avgHeartRate /
maxHeartRate contradicted the heart_rate samples it sent for the same window
(observed 2026-07-22: summary 129/134 vs 434 samples averaging 152, peak 184).
The samples are the source of truth when they densely cover the workout.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from health_advisor import db


def _seed_workout(conn, *, start, end, local_date, duration_min,
                  avg=None, max_=None, key="w1"):
    db.insert_workouts(conn, [{
        "workout_type": "running", "start_utc": start, "end_utc": end,
        "local_date": local_date, "duration_min": duration_min,
        "energy_kcal": 300.0, "distance_mi": 2.6, "unit_distance": "mi",
        "source": "Watch", "route_ref": None,
        "avg_heart_rate": avg, "max_heart_rate": max_, "dedupe_key": key,
    }])
    conn.commit()


def _seed_hr(conn, *, start, span_min, values):
    """Insert heart-rate samples evenly spaced across `span_min` from `start`."""
    t0 = datetime.fromisoformat(start)
    step = timedelta(minutes=span_min / max(len(values) - 1, 1))
    rows = []
    for i, v in enumerate(values):
        iso = (t0 + step * i).isoformat()
        rows.append({
            "metric": "heart_rate", "value": float(v), "unit": "count/min",
            "start_utc": iso, "end_utc": iso, "start_local": iso[:19],
            "local_date": iso[:10], "source": "Watch", "origin": "receiver",
            "dedupe_key": f"hr-{i}-{iso}",
        })
    db.insert_records(conn, rows)
    conn.commit()


def _hr(conn, key="w1"):
    r = conn.execute("SELECT avg_heart_rate a, max_heart_rate m FROM workouts "
                     "WHERE dedupe_key = ?", (key,)).fetchone()
    return r["a"], r["m"]


START, END, DAY = "2026-07-22T14:23:01+00:00", "2026-07-22T14:59:16+00:00", "2026-07-22"
HARD = [140] * 14 + [160] * 14 + [184, 184]          # n=30, avg 152.27, max 184


def test_reconcile_replaces_summary_hr_that_contradicts_dense_samples(conn):
    _seed_workout(conn, start=START, end=END, local_date=DAY, duration_min=36.25,
                  avg=129.0, max_=134.0)
    _seed_hr(conn, start=START, span_min=36.0, values=HARD)

    assert db.reconcile_workout_heart_rate(conn) == 1

    avg, mx = _hr(conn)
    assert abs(avg - 152.27) < 0.1
    assert mx == 184.0


def test_reconcile_leaves_agreeing_summary_untouched(conn):
    _seed_workout(conn, start=START, end=END, local_date=DAY, duration_min=36.25,
                  avg=152.3, max_=184.0)
    _seed_hr(conn, start=START, span_min=36.0, values=HARD)

    assert db.reconcile_workout_heart_rate(conn) == 0
    assert _hr(conn) == (152.3, 184.0)


def test_reconcile_tolerates_small_disagreement_as_aggregation_noise(conn):
    # 3 bpm between the device's mean and ours is rounding/windowing, not the bug
    # we're hunting. Only a material contradiction is worth overwriting.
    _seed_workout(conn, start=START, end=END, local_date=DAY, duration_min=36.25,
                  avg=155.3, max_=184.0)
    _seed_hr(conn, start=START, span_min=36.0, values=HARD)

    assert db.reconcile_workout_heart_rate(conn) == 0
    assert _hr(conn) == (155.3, 184.0)


def test_reconcile_ignores_a_sparse_sample_series(conn):
    # A handful of stray samples is weaker evidence than the device's own summary.
    _seed_workout(conn, start=START, end=END, local_date=DAY, duration_min=36.25,
                  avg=129.0, max_=134.0)
    _seed_hr(conn, start=START, span_min=36.0, values=[150, 160, 170, 180, 184])

    assert db.reconcile_workout_heart_rate(conn) == 0
    assert _hr(conn) == (129.0, 134.0)


def test_reconcile_can_be_scoped_to_the_days_a_payload_touched(conn):
    other_start, other_end = "2026-07-23T14:23:01+00:00", "2026-07-23T14:59:16+00:00"
    _seed_workout(conn, start=START, end=END, local_date=DAY, duration_min=36.25,
                  avg=129.0, max_=134.0, key="w1")
    _seed_hr(conn, start=START, span_min=36.0, values=HARD)
    _seed_workout(conn, start=other_start, end=other_end, local_date="2026-07-23",
                  duration_min=36.25, avg=129.0, max_=134.0, key="w2")
    _seed_hr(conn, start=other_start, span_min=36.0, values=HARD)

    assert db.reconcile_workout_heart_rate(conn, local_dates={DAY}) == 1

    assert abs(_hr(conn, "w1")[0] - 152.27) < 0.1
    assert _hr(conn, "w2") == (129.0, 134.0)      # untouched: different day


def test_reconcile_fills_in_a_missing_summary_from_samples(conn):
    # Backfill rows carry no HR at all; samples are strictly better than nothing.
    _seed_workout(conn, start=START, end=END, local_date=DAY, duration_min=36.25,
                  avg=None, max_=None)
    _seed_hr(conn, start=START, span_min=36.0, values=HARD)

    assert db.reconcile_workout_heart_rate(conn) == 1
    assert abs(_hr(conn)[0] - 152.27) < 0.1


def test_reconcile_ignores_samples_covering_only_part_of_the_workout(conn):
    # 30 dense samples, but all inside the first 5 min of a 36-min run: the mean
    # of a warmup is not the mean of the workout, so the summary stands.
    _seed_workout(conn, start=START, end=END, local_date=DAY, duration_min=36.25,
                  avg=129.0, max_=134.0)
    _seed_hr(conn, start=START, span_min=5.0, values=HARD)

    assert db.reconcile_workout_heart_rate(conn) == 0
    assert _hr(conn) == (129.0, 134.0)
