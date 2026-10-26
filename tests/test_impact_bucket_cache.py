"""The cross-request cache for ``impact_buckets.impact_bucket_rows_py`` (#692 S2-4b).

The cache key is a digest of every field of ``BucketInputs``. These tests pin:

* a repeat call is a hit (``_compute`` runs once) and returns the first call's
  rows; mutating a returned row list never changes a later hit;
* any change inside the window (a new sample, a mark on the workout) is a miss,
  and the result equals the SQL reference;
* a change outside the window is still a hit;
* the row budget evicts least-recently-used entries first;
* eight threads reading the same and different windows get SQL-equal results.

Every expected value comes from ``metrics._impact_bucket_rows_sql``, the frozen
reference, or from a literal, never from the cache's own output.
"""
from __future__ import annotations

import dataclasses
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from health_advisor import db
from health_advisor import impact_buckets as ib
from health_advisor import metrics as mx
from tests.test_impact_buckets_py import (
    LOCAL_WINDOW, POST_DAY, PRE_DAY, UTC_WINDOW, _distance, _iso, _steps,
    _workout)

DAY_ARGS = (POST_DAY, POST_DAY)
OTHER_DAY = "2026-09-03"


@pytest.fixture(autouse=True)
def _fresh_cache():
    ib.clear_impact_bucket_cache()
    yield
    ib.clear_impact_bucket_cache()


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "vault.db")
    db.init_db(connection)
    yield connection
    connection.close()


@pytest.fixture
def compute_calls(monkeypatch):
    """Count how many times the pure ``_compute`` actually runs."""
    real = ib._compute
    calls = []

    def spy(*args, **kwargs):
        calls.append(args[1:3])
        return real(*args, **kwargs)

    monkeypatch.setattr(ib, "_compute", spy)
    return calls


def _workout_vault(conn):
    """A marked-or-not workout with a jog-cadence bucket inside it on POST_DAY."""
    key = _workout(conn, _iso(POST_DAY, "12:00:00"), _iso(POST_DAY, "12:10:00"),
                   POST_DAY)
    _distance(conn, 0.05, _iso(POST_DAY, "12:03:00"), _iso(POST_DAY, "12:03:20"),
              POST_DAY, "Apple Watch")
    _steps(conn, 60.0, _iso(POST_DAY, "12:03:00"), _iso(POST_DAY, "12:03:20"),
           POST_DAY, "Apple Watch")
    conn.commit()
    return key


def _sql(conn, predicate=LOCAL_WINDOW, args=DAY_ARGS):
    return mx._impact_bucket_rows_sql(conn, predicate, args)


def _py(conn, predicate=LOCAL_WINDOW, args=DAY_ARGS):
    return ib.impact_bucket_rows_py(conn, predicate, args)


def test_repeat_call_is_a_hit_and_mutating_its_rows_changes_nothing(
        conn, compute_calls):
    _workout_vault(conn)
    reference = _sql(conn)
    assert reference, "the fixture must produce rows"

    first = _py(conn)
    assert first == reference
    assert len(compute_calls) == 1
    assert ib.impact_bucket_cache_info()["misses"] == 1

    second = _py(conn)
    assert len(compute_calls) == 1, "a repeat call must not recompute"
    assert second == first
    info = ib.impact_bucket_cache_info()
    assert (info["hits"], info["misses"]) == (1, 1)

    # The caller owns its rows: mutate every field of the returned list.
    second[0]["mi"] = 999.0
    second[0]["is_jog"] = 42
    second.append({"bkt": -1})
    second.clear()
    third = _py(conn)
    assert third == reference
    assert len(compute_calls) == 1
    assert ib.impact_bucket_cache_info()["hits"] == 2


def test_sample_inside_the_window_is_a_miss_and_matches_sql(conn, compute_calls):
    _workout_vault(conn)
    before = _py(conn)
    assert before == _sql(conn)

    _distance(conn, 0.4, _iso(POST_DAY, "15:00:00"), _iso(POST_DAY, "15:00:40"),
              POST_DAY, "Apple Watch", tag="new")
    conn.commit()
    after = _py(conn)
    assert len(compute_calls) == 2, "a new sample in the window must miss"
    assert after == _sql(conn)
    assert after != before


def test_marking_the_workout_not_a_session_is_a_miss_and_matches_sql(
        conn, compute_calls):
    key = _workout_vault(conn)
    before = _py(conn)
    assert before == _sql(conn)
    assert before[0]["in_workout"] == 1 and before[0]["is_jog"] == 1

    db.mark_workout_not_a_session(conn, key)
    conn.commit()
    after = _py(conn)
    assert len(compute_calls) == 2, "a new mark must change the key"
    assert after == _sql(conn)
    assert after[0]["in_workout"] == 0 and after[0]["is_jog"] == 0


def test_sample_outside_the_window_is_still_a_hit(conn, compute_calls):
    _workout_vault(conn)
    first = _py(conn)

    _distance(conn, 0.3, _iso(OTHER_DAY, "08:00:00"), _iso(OTHER_DAY, "08:00:00"),
              OTHER_DAY, "Apple Watch", tag="outside")
    conn.commit()
    second = _py(conn)
    assert len(compute_calls) == 1, "a sample outside the window must not miss"
    assert second == first == _sql(conn)


def test_row_budget_evicts_least_recently_used_first(conn, compute_calls,
                                                     monkeypatch):
    # Each day holds one pre-cutoff point sample, so each call is one row.
    days = ["2026-03-02", "2026-03-03", "2026-03-04"]
    for n, day in enumerate(days):
        _distance(conn, 0.5 + n, _iso(day, "09:00:05"), _iso(day, "09:00:05"),
                  day, "Apple Watch", tag=f"p{n}")
    conn.commit()
    monkeypatch.setattr(ib, "IMPACT_BUCKET_CACHE_MAX_ROWS", 2)

    def call(day):
        return ib.impact_bucket_rows_py(conn, LOCAL_WINDOW, (day, day))

    a, b, c = days
    assert len(call(a)) == 1 and len(call(b)) == 1       # misses: A, B
    assert len(call(a)) == 1                             # hit: A is now newest
    call(c)                                              # miss: evicts B (LRU)
    assert ib.impact_bucket_cache_info()["entries"] == 2
    misses_before = ib.impact_bucket_cache_info()["misses"]
    call(a)                                              # hit: A survived
    assert ib.impact_bucket_cache_info()["misses"] == misses_before
    call(b)                                              # miss: B was evicted
    info = ib.impact_bucket_cache_info()
    assert info["misses"] == misses_before + 1
    assert info["rows"] <= 2
    assert len(compute_calls) == 4                       # A, B, C, B again


def test_entry_larger_than_the_budget_is_returned_but_not_stored(
        conn, compute_calls, monkeypatch):
    _distance(conn, 0.2, _iso(PRE_DAY, "09:00:05"), _iso(PRE_DAY, "09:00:05"),
              PRE_DAY, "Apple Watch", tag="a")
    _distance(conn, 0.3, _iso(PRE_DAY, "10:00:05"), _iso(PRE_DAY, "10:00:05"),
              PRE_DAY, "Apple Watch", tag="b")
    conn.commit()
    monkeypatch.setattr(ib, "IMPACT_BUCKET_CACHE_MAX_ROWS", 1)

    rows = _py(conn, LOCAL_WINDOW, (PRE_DAY, PRE_DAY))
    assert len(rows) == 2 and rows == _sql(conn, LOCAL_WINDOW, (PRE_DAY, PRE_DAY))
    assert ib.impact_bucket_cache_info()["entries"] == 0
    _py(conn, LOCAL_WINDOW, (PRE_DAY, PRE_DAY))
    assert len(compute_calls) == 2


def test_eight_threads_match_the_sql_reference(tmp_path):
    path = tmp_path / "vault.db"
    main = db.connect(path)
    db.init_db(main)
    _workout_vault(main)
    _distance(main, 0.6, _iso(OTHER_DAY, "07:00:00"), _iso(OTHER_DAY, "07:01:00"),
              OTHER_DAY, "Apple Watch", tag="other")
    main.commit()
    windows = [
        (LOCAL_WINDOW, DAY_ARGS),
        (LOCAL_WINDOW, (OTHER_DAY, OTHER_DAY)),
        (UTC_WINDOW, (_iso(POST_DAY, "12:00:00"), _iso(POST_DAY, "12:02:00"))),
        (LOCAL_WINDOW, (POST_DAY, OTHER_DAY)),
    ]
    reference = [_sql(main, p, a) for p, a in windows]
    main.close()

    def worker(seed):
        conn = db.connect(path)
        try:
            out = []
            for step in range(12):
                idx = (seed + step) % len(windows)
                predicate, args = windows[idx]
                out.append((idx, ib.impact_bucket_rows_py(conn, predicate, args)))
            return out
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(8)))
    mismatches = [idx for run in results for idx, rows in run
                  if rows != reference[idx]]
    assert mismatches == []
    info = ib.impact_bucket_cache_info()
    assert info["hits"] + info["misses"] == 8 * 12


def test_digest_is_over_every_field_of_the_inputs(conn):
    _workout_vault(conn)
    inputs = ib._fetch_inputs(conn, LOCAL_WINDOW, DAY_ARGS)
    base = ib._inputs_digest(inputs)
    assert ib._inputs_digest(ib._fetch_inputs(conn, LOCAL_WINDOW, DAY_ARGS)) == base
    assert len(base) == 32

    first = inputs.distance_rows[0]
    nudged = (first[:3] + (math.nextafter(first[3], math.inf),) + first[4:],)
    changes = {
        "cutoff": dataclasses.replace(inputs, cutoff="2026-01-01"),
        "distance value bit": dataclasses.replace(
            inputs, distance_rows=nudged + inputs.distance_rows[1:]),
        "heart": dataclasses.replace(inputs, heart_rows=((POST_DAY, 70.0, 1),)),
        "steps": dataclasses.replace(inputs, step_rows=()),
        "workouts": dataclasses.replace(inputs, workout_rows=()),
        "arbitration": dataclasses.replace(
            inputs, arbitration=dataclasses.replace(
                inputs.arbitration, watch_days=("1999-01-01",))),
    }
    for name, changed in changes.items():
        assert ib._inputs_digest(changed) != base, name


def test_digest_refuses_types_it_cannot_encode():
    with pytest.raises(TypeError):
        ib._canonical((object(),))
