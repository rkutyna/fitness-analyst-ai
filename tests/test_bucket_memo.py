"""The request-scoped memo for ``metrics.impact_bucket_rows``.

One read of a day or a week asks for the same workout's buckets once per figure
built from them. Inside ``metrics.bucket_memo()`` each distinct call runs the
bucketing query once. These tests pin the four properties that make that safe:
the computation count drops, a reused result is an independent copy, nothing is
cached outside the block, and nested blocks share one memo. The last test pins
that every argument that selects rows is in the key, so two different windows
never share a result.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta

import pytest

from health_advisor import analysis as A
from health_advisor import db
from health_advisor import metrics as mx

DAY_A = "2026-03-02"
DAY_B = "2026-03-04"
UTC = "start_utc >= ? AND start_utc < ?"
LOCAL = "local_date BETWEEN ? AND ?"


def _insert(conn, metric, value, unit, ts, day, key):
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 't', 't', ?)",
        (metric, value, unit, ts, ts, ts, day, key))


def _session(conn, day, seconds, pace_min_per_mi, cadence_spm, hr):
    """One workout at noon UTC with per-second distance, HR and 20-s steps."""
    t0 = datetime.fromisoformat(f"{day}T12:00:00+00:00")
    start = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (t0 + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) "
        "VALUES ('running', ?, ?, ?, ?, 'test', ?)",
        (start, end, day, seconds / 60.0, f"workout|{start}|{end}"))
    per_second = 1.0 / (pace_min_per_mi * 60.0)
    for i in range(seconds):
        ts = (t0 + timedelta(seconds=i)).isoformat()
        tag = f"{day}-{i}"
        _insert(conn, "distance_walking_running", per_second, "mi", ts, day,
                f"d-{tag}")
        if i % 5 == 0:
            _insert(conn, "heart_rate", hr, "count/min", ts, day, f"h-{tag}")
        if i % mx.IMPACT_BUCKET_SECONDS == 0:
            _insert(conn, "step_count", cadence_spm / 3.0, "count", ts, day,
                    f"s-{tag}")
    return start, end


def _vault(path, sessions):
    conn = db.connect(path)
    db.init_db(conn)
    windows = [_session(conn, *s) for s in sessions]
    conn.commit()
    conn.close()
    return windows


@pytest.fixture
def vault(tmp_path):
    """Two different runs on two days, so the two windows' rows differ."""
    path = tmp_path / "vault.db"
    windows = _vault(path, [(DAY_A, 300, 12.0, 160.0, 140.0),
                            (DAY_B, 240, 10.0, 170.0, 150.0)])
    conn = db.connect(path, read_only=True)
    yield conn, windows
    conn.close()


@pytest.fixture
def computations(monkeypatch):
    """Every run of the bucketing query, with its window arguments."""
    seen = []
    real = mx._impact_bucket_rows_uncached

    def counting(conn, window_predicate, window_args, **kwargs):
        seen.append((window_predicate, tuple(window_args)))
        return real(conn, window_predicate, window_args, **kwargs)

    monkeypatch.setattr(mx, "_impact_bucket_rows_uncached", counting)
    return seen


def _fresh(conn, predicate, args, **kwargs):
    return copy.deepcopy(
        mx._impact_bucket_rows_sql(conn, predicate, args, **kwargs))


def test_the_memo_runs_each_distinct_window_once(vault, computations):
    conn, (win_a, win_b) = vault
    expected_a = _fresh(conn, UTC, win_a)
    computations.clear()
    with mx.bucket_memo():
        first = mx.bucket_series(conn, *win_a)
        again = mx.bucket_series(conn, *win_a)
        rows = mx.impact_bucket_rows(conn, UTC, win_a,
                                     arbitration_window=win_a,
                                     arbitration_window_kind="utc")
        mx.impact_bucket_rows(conn, UTC, win_b,
                              arbitration_window_kind="utc")
        A.impact_volume(conn, DAY_A, DAY_B)
        A.impact_volume(conn, DAY_A, DAY_B)
    assert computations == [(UTC, win_a), (UTC, win_b), (LOCAL, (DAY_A, DAY_B))]
    assert first and first == again
    assert rows == expected_a
    assert any(r["is_jog"] for r in rows)

    computations.clear()
    for _ in range(3):
        mx.bucket_series(conn, *win_a)
    assert len(computations) == 3


def test_a_reused_result_is_an_independent_copy(vault, computations):
    conn, (win_a, _) = vault
    expected = _fresh(conn, UTC, win_a)
    computations.clear()
    with mx.bucket_memo():
        first = mx.impact_bucket_rows(conn, UTC, win_a)
        first[0]["mi"] = -1.0
        first[0]["is_jog"] = 99
        del first[1]["hr"]
        first.append({"bkt": 0})
        second = mx.impact_bucket_rows(conn, UTC, win_a)
        assert second == expected
        second.clear()
        third = mx.impact_bucket_rows(conn, UTC, win_a)
        assert third == expected
        assert third is not second and third[0] is not first[0]
    assert len(computations) == 1


def test_nothing_is_cached_outside_the_block(vault, computations):
    conn, (win_a, _) = vault
    assert mx._BUCKET_MEMO.get() is None
    mx.impact_bucket_rows(conn, UTC, win_a)
    mx.impact_bucket_rows(conn, UTC, win_a)
    assert len(computations) == 2

    with mx.bucket_memo():
        mx.impact_bucket_rows(conn, UTC, win_a)
    assert mx._BUCKET_MEMO.get() is None
    mx.impact_bucket_rows(conn, UTC, win_a)
    assert len(computations) == 4

    with pytest.raises(RuntimeError):
        with mx.bucket_memo():
            mx.impact_bucket_rows(conn, UTC, win_a)
            raise RuntimeError("the block failed")
    assert mx._BUCKET_MEMO.get() is None
    mx.impact_bucket_rows(conn, UTC, win_a)
    assert len(computations) == 6


def test_a_nested_block_reuses_the_outer_memo(vault, computations):
    conn, (win_a, win_b) = vault
    with mx.bucket_memo():
        outer = mx._BUCKET_MEMO.get()
        mx.impact_bucket_rows(conn, UTC, win_a)
        with mx.bucket_memo():
            assert mx._BUCKET_MEMO.get() is outer
            mx.impact_bucket_rows(conn, UTC, win_a)
            mx.impact_bucket_rows(conn, UTC, win_b)
        assert mx._BUCKET_MEMO.get() is outer
        mx.impact_bucket_rows(conn, UTC, win_b)
    assert mx._BUCKET_MEMO.get() is None
    assert computations == [(UTC, win_a), (UTC, win_b)]


def test_every_argument_that_selects_rows_is_in_the_key(tmp_path, vault,
                                                        computations):
    """Each call inside one block equals the same call made with no memo."""
    conn, (win_a, win_b) = vault
    other_path = tmp_path / "other.db"
    _vault(other_path, [(DAY_A, 300, 22.0, 100.0, 100.0)])
    other = db.connect(other_path, read_only=True)
    session = (win_a[0], win_b[1])
    # A block read inside a session: the read window is a sub-span, and source
    # arbitration is decided over the whole session (as a decoupling block
    # does). Same arbitration window as call 4, different rows.
    block = (win_a[0], "2026-03-02T12:02:30Z")
    calls = [
        (conn, UTC, win_a, {}),
        (conn, UTC, win_b, {}),
        (conn, LOCAL, (DAY_A, DAY_A), {}),
        (conn, LOCAL, (DAY_A, DAY_B), {}),
        (conn, UTC, win_a, {"arbitration_window_kind": "utc"}),
        (conn, UTC, win_a, {"arbitration_window": session,
                            "arbitration_window_kind": "utc"}),
        (other, UTC, win_a, {}),
        (conn, UTC, block, {"arbitration_window": win_a,
                            "arbitration_window_kind": "utc"}),
    ]
    try:
        expected = [_fresh(c, p, a, **k) for c, p, a, k in calls]
        assert expected[0] != expected[1]
        assert expected[0] != expected[6]
        assert expected[7] and expected[7] != expected[4]
        computations.clear()
        with mx.bucket_memo():
            got = [mx.impact_bucket_rows(c, p, a, **k) for c, p, a, k in calls]
    finally:
        other.close()
    assert len(computations) == len(calls)
    for index, (rows, want) in enumerate(zip(got, expected)):
        assert rows == want, f"call {index} received another window's rows"
