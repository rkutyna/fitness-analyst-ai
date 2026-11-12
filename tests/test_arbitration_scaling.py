"""Workout-window distance arbitration: linear cost, unchanged survivors.

downstream #734: the winner tests were EXISTS subqueries correlated to every
candidate row, so one recompute of a workout day cost O(n^2) in its samples
and held the ingest's write lock for seconds. The oracle for cost here is a
count of SQLite VM steps at n and 4n, never wall-clock, so the test cannot rot
with machine speed. The survivor oracles are written out by hand.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from health_advisor import db

M = "distance_walking_running"
WATCH, PHONE, GYM = "Test Apple Watch", "Test iPhone", "GymKit"


def _iso(day: str, hh: int, mm: int, ss: int = 0) -> str:
    return datetime(*map(int, day.split("-")), hh, mm, ss,
                    tzinfo=timezone.utc).isoformat()


def _rec(conn, day, start_iso, source, value=0.001, tag=""):
    start = datetime.fromisoformat(start_iso)
    end = (start + timedelta(seconds=1)).isoformat()
    cur = conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, dedupe_key) "
        "VALUES (?, ?, 'mi', ?, ?, ?, ?, 'test', ?)",
        (M, value, start_iso, end, day, source,
         f"{M}|{start_iso}|{end}|{source}|{tag}"))
    return cur.lastrowid


def _workout(conn, day, start_iso, end_iso, kind="walking"):
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES (?, ?, ?, ?, 60, 'x', ?)",
        (kind, start_iso, end_iso, day, db.workout_key(kind, start_iso, end_iso)))


def _survivors(conn, day, **window):
    clause, args = db._arbitration(conn, M, day, **window)
    return {r[0] for r in conn.execute(
        f"SELECT id FROM records WHERE metric = ? AND local_date = ?{clause}",
        (M, day, *args))}


def _assert_all_modes(conn, day, expected):
    assert _survivors(conn, day) == expected
    assert _survivors(conn, day, arbitration_window=(day, day),
                      arbitration_window_kind="local_date") == expected
    db.recompute_daily_metrics(conn, pairs=[(M, day)])
    row = conn.execute("SELECT count FROM daily_metrics WHERE metric = ? "
                       "AND date = ?", (M, day)).fetchone()
    assert row["count"] == len(expected)


# --------------------------------------------------------------- oracles
def test_gymkit_workout_beats_watch_and_phone_only_inside_its_window(conn):
    day = "2026-10-20"
    _workout(conn, day, _iso(day, 17, 0), _iso(day, 18, 0), "indoor_walk")
    _workout(conn, day, _iso(day, 20, 0), _iso(day, 20, 15))
    gym = _rec(conn, day, _iso(day, 17, 55), GYM)        # last minutes only
    _rec(conn, day, _iso(day, 17, 5), WATCH)             # loses to GymKit
    _rec(conn, day, _iso(day, 17, 6), PHONE)             # loses
    watch_late = _rec(conn, day, _iso(day, 20, 5), WATCH)  # no GymKit there
    _rec(conn, day, _iso(day, 20, 6), PHONE)             # loses to the watch
    _assert_all_modes(conn, day, {gym, watch_late})


def test_overlapping_workouts_each_keep_their_own_winner(conn):
    day = "2026-10-21"
    _workout(conn, day, _iso(day, 17, 0), _iso(day, 18, 0))            # A
    _workout(conn, day, _iso(day, 17, 30), _iso(day, 18, 30), "running")  # B
    w_a = _rec(conn, day, _iso(day, 17, 10), WATCH)   # A only, A has no GymKit
    _rec(conn, day, _iso(day, 17, 11), PHONE)         # A has a watch: loses
    _rec(conn, day, _iso(day, 17, 40), WATCH)         # in A and B; B has GymKit
    g_b = _rec(conn, day, _iso(day, 18, 10), GYM)     # B only
    _rec(conn, day, _iso(day, 18, 20), WATCH)         # B: loses to GymKit
    _rec(conn, day, _iso(day, 18, 40), PHONE)         # outside: day rule
    _assert_all_modes(conn, day, {w_a, g_b})


def test_phone_only_day_keeps_every_phone_sample(conn):
    day = "2026-10-22"
    _workout(conn, day, _iso(day, 17, 0), _iso(day, 18, 0))
    ids = {_rec(conn, day, _iso(day, 17, m), PHONE) for m in range(0, 50, 7)}
    ids.add(_rec(conn, day, _iso(day, 19, 0), PHONE))
    _assert_all_modes(conn, day, ids)


def test_day_without_a_workout_drops_phone_for_watch(conn):
    day = "2026-10-23"
    watch = {_rec(conn, day, _iso(day, 12, m), WATCH) for m in range(5)}
    for m in range(5):
        _rec(conn, day, _iso(day, 12, m, 30), PHONE)
    _assert_all_modes(conn, day, watch)


def test_pre_cutoff_day_is_left_alone(conn):
    day = "2026-08-01"
    _workout(conn, day, _iso(day, 17, 0), _iso(day, 18, 0), "indoor_walk")
    ids = {_rec(conn, day, _iso(day, 17, 5), src) for src in (GYM, WATCH, PHONE)}
    _assert_all_modes(conn, day, ids)


# ----------------------------------------------------------------- scaling
def _steps_for_recompute(tmp_path, name, n, *, gymkit, phone_only=False):
    path = tmp_path / f"{name}-{n}.db"
    conn = db.connect(path)
    try:
        db.init_db(conn)
        day = "2026-11-03"
        start = datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc)
        _workout(conn, day, start.isoformat(),
                 (start + timedelta(seconds=n + 60)).isoformat())
        main_source = PHONE if phone_only else WATCH
        for i in range(n):
            _rec(conn, day, (start + timedelta(seconds=i)).isoformat(), main_source)
        for i in range(0, n, 10):
            _rec(conn, day, (start + timedelta(seconds=i)).isoformat(), PHONE,
                 tag="p")
        if gymkit:   # GymKit only at the very end of the window
            for i in range(5):
                _rec(conn, day, (start + timedelta(seconds=n + i)).isoformat(),
                     GYM)
        conn.commit()
        steps = [0]

        def tick():
            steps[0] += 1
            return 0

        conn.set_progress_handler(tick, 100)
        db.recompute_daily_metrics(conn, pairs=[(M, day)])
        conn.set_progress_handler(None, 100)
        conn.rollback()
        return steps[0]
    finally:
        conn.close()


@pytest.mark.parametrize("name,kwargs", [
    ("no-gymkit", {"gymkit": False}),
    ("gymkit-at-end", {"gymkit": True}),
    ("phone-only", {"gymkit": False, "phone_only": True}),
])
def test_arbitration_cost_is_linear_in_the_days_samples(tmp_path, name, kwargs):
    small = _steps_for_recompute(tmp_path, name, 400, **kwargs)
    large = _steps_for_recompute(tmp_path, name, 1600, **kwargs)
    # Linear is ~4x for 4x the samples; the correlated form measured ~16x.
    assert large / small < 6, (name, small, large)
