"""``impact_buckets`` splits into ``_fetch_inputs`` (every read) and ``_compute`` (pure).

Three properties are pinned here:

* The inputs are one plain immutable value: frozen, hashable, made only of
  tuples, ints, floats, strings and None, with no sqlite3.Row, dict or list.
* ``_compute`` needs no connection. The inputs are fetched, the connection is
  closed, and ``_compute`` still returns the rows the public call returns. The
  rows are also checked against a hand-computed literal, so a bug that moved
  the reads and the arithmetic together would still fail.
* A hidden read in ``_compute`` is caught: the mutation that makes it query a
  module-level connection turns the closed-connection test red.
"""
from __future__ import annotations

import dataclasses
import inspect
import sqlite3

import pytest

from health_advisor import db
from health_advisor import impact_buckets as ib
from health_advisor import metrics as mx
from tests.test_impact_buckets_py import (
    LOCAL_WINDOW, POST_DAY, _distance, _heart, _iso, _record, _steps, _workout)

WORKOUT_A = (_iso(POST_DAY, "11:00:00"), _iso(POST_DAY, "11:30:00"))
WORKOUT_B = (_iso(POST_DAY, "12:00:00"), _iso(POST_DAY, "12:10:00"))
DAY_ARGS = (POST_DAY, POST_DAY)


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "vault.db")
    db.init_db(connection)
    yield connection
    connection.close()


def _build_vault(conn):
    """Every rule at once: day rule, GymKit workout rule, a marked workout, heart and steps.

    Hand-computed survivors on POST_DAY (local_date window):
    * GymKit 0.25 mi point at 11:05:00, inside workout A: kept.
    * Apple Watch 0.5 mi point and iPhone 0.9 mi point at 11:05:00: the
      workout rule drops both (a GymKit sample starts inside A).
    * Apple Watch 0.6 mi interval 12:03:10-12:04:10 in marked workout B: kept.
    * iPhone 0.3 mi point at 15:00:00: the day rule drops it (a watch sample
      records the day).
    So the surviving distance is 0.25 + 0.6 = 0.85 miles.
    """
    key_a = _workout(conn, *WORKOUT_A, POST_DAY)
    key_b = _workout(conn, *WORKOUT_B, POST_DAY)
    _distance(conn, 0.25, _iso(POST_DAY, "11:05:00"),
              _iso(POST_DAY, "11:05:00"), POST_DAY, "GymKit", tag="g")
    _distance(conn, 0.5, _iso(POST_DAY, "11:05:00"),
              _iso(POST_DAY, "11:05:00"), POST_DAY, "Apple Watch", tag="w")
    _distance(conn, 0.9, _iso(POST_DAY, "11:05:00"),
              _iso(POST_DAY, "11:05:00"), POST_DAY, "iPhone", tag="p")
    _distance(conn, 0.6, _iso(POST_DAY, "12:03:10"),
              _iso(POST_DAY, "12:04:10"), POST_DAY, "Apple Watch", tag="iv")
    _distance(conn, 0.3, _iso(POST_DAY, "15:00:00"),
              _iso(POST_DAY, "15:00:00"), POST_DAY, "iPhone", tag="late")
    _steps(conn, 60.0, _iso(POST_DAY, "12:03:10"),
           _iso(POST_DAY, "12:03:20"), POST_DAY, "Apple Watch", tag="sw")
    _steps(conn, 40.0, _iso(POST_DAY, "12:03:10"),
           _iso(POST_DAY, "12:03:20"), POST_DAY, "iPhone", tag="sp")
    _heart(conn, 100.0, _iso(POST_DAY, "11:05:00"), POST_DAY, tag="h1")
    _heart(conn, 120.0, _iso(POST_DAY, "12:03:10"), POST_DAY, tag="h2")
    db.mark_workout_not_a_session(conn, key_b)
    conn.commit()
    return key_a


def _walk(value, path="inputs"):
    """Assert every leaf is a plain scalar or tuple, and no container is a Row, dict or list."""
    assert not isinstance(value, (sqlite3.Row, dict, list)), path
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _walk(getattr(value, field.name), f"{path}.{field.name}")
    elif isinstance(value, tuple):
        for i, item in enumerate(value):
            _walk(item, f"{path}[{i}]")
    else:
        assert value is None or type(value) in (bool, int, float, str), (
            path, type(value))


def test_compute_after_closing_the_connection_matches_the_public_call(conn):
    _build_vault(conn)
    public = ib.impact_bucket_rows_py(conn, LOCAL_WINDOW, DAY_ARGS)
    reference = mx._impact_bucket_rows_sql(conn, LOCAL_WINDOW, DAY_ARGS)
    inputs = ib._fetch_inputs(conn, LOCAL_WINDOW, DAY_ARGS,
                              arbitration_window=None,
                              arbitration_window_kind="local_date")
    conn.close()

    rows = ib._compute(inputs, LOCAL_WINDOW, DAY_ARGS,
                       arbitration_window=None,
                       arbitration_window_kind="local_date")
    assert rows == public
    assert rows == reference
    # Hand-computed: 0.25 (GymKit point) + 0.6 (watch interval) survive.
    assert sum(row["mi"] for row in rows) == pytest.approx(0.85, abs=1e-12)
    # 60 watch steps + 40 phone steps in one 20 s bucket: the larger source, 60 * 3.
    cadences = [row["cadence_spm"] for row in rows
                if row["cadence_spm"] is not None]
    assert 180.0 in cadences


def test_inputs_are_one_plain_frozen_value(conn):
    _build_vault(conn)
    inputs = ib._fetch_inputs(conn, LOCAL_WINDOW, DAY_ARGS,
                              arbitration_window=None,
                              arbitration_window_kind="local_date")
    assert isinstance(inputs, ib.BucketInputs)
    assert isinstance(inputs.arbitration, ib.ArbitrationInputs)
    _walk(inputs)
    # The samples read are there: the distance rows, heart rows, steps and the
    # labelled arbitration sources.
    assert len(inputs.distance_rows) == 5
    assert len(inputs.heart_rows) == 2
    assert len(inputs.step_rows) == 2
    assert inputs.arbitration.iphone == ("iPhone",)
    assert inputs.arbitration.gymkit == ("GymKit",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.cutoff = "1900-01-01"


def test_two_fetches_of_one_window_are_equal_and_hash_alike(conn):
    _build_vault(conn)
    first = ib._fetch_inputs(conn, LOCAL_WINDOW, DAY_ARGS,
                             arbitration_window=None,
                             arbitration_window_kind="local_date")
    second = ib._fetch_inputs(conn, LOCAL_WINDOW, DAY_ARGS,
                              arbitration_window=None,
                              arbitration_window_kind="local_date")
    assert first == second
    assert hash(first) == hash(second)


def test_compute_takes_no_connection():
    params = inspect.signature(ib._compute).parameters
    assert "conn" not in params
    assert not any(p.annotation is sqlite3.Connection
                   for p in params.values())


def test_span_guard_drops_twelve_hour_and_longer_samples_and_null_bounds(conn):
    """The span guard moved from SQL into ``_fetch_inputs``; its rule is pinned here.

    The guard keeps a distance sample only when end - start < 43200 s (12 h,
    a hand-written literal below). Hand-computed on POST_DAY:
    * 00:10:00-00:10:30 (30 s): kept.
    * 00:00:00-11:59:59 (43199 s): kept, one second under the limit.
    * 00:00:00-12:00:00 (43200 s): dropped, the limit itself.
    * 00:00:00-18:00:00 (64800 s): dropped.
    * 09:00:00 with an unparseable end: dropped, strftime gives NULL and the
      SQL comparison is NULL (end_utc is NOT NULL, so the text is not NULL).
    The two survivors carry 0.1 mi each, so the buckets hold 0.2 mi.
    """
    assert ib._SPAN_S == 12 * 60 * 60
    _distance(conn, 0.1, _iso(POST_DAY, "00:10:00"),
              _iso(POST_DAY, "00:10:30"), POST_DAY, "Apple Watch", tag="short")
    _distance(conn, 0.1, _iso(POST_DAY, "00:00:00"),
              _iso(POST_DAY, "11:59:59"), POST_DAY, "Apple Watch", tag="under")
    _distance(conn, 0.1, _iso(POST_DAY, "00:00:00"),
              _iso(POST_DAY, "12:00:00"), POST_DAY, "Apple Watch", tag="limit")
    _distance(conn, 0.1, _iso(POST_DAY, "00:00:00"),
              _iso(POST_DAY, "18:00:00"), POST_DAY, "Apple Watch", tag="long")
    _record(conn, "distance_walking_running", 0.1, "mi",
            _iso(POST_DAY, "09:00:00"), "not-a-time", POST_DAY, "Apple Watch",
            "null")
    conn.commit()

    inputs = ib._fetch_inputs(conn, LOCAL_WINDOW, DAY_ARGS,
                              arbitration_window=None,
                              arbitration_window_kind="local_date")
    assert [row[2] for row in inputs.distance_rows] == [
        _iso(POST_DAY, "00:10:00"), _iso(POST_DAY, "00:00:00")]

    public = ib.impact_bucket_rows_py(conn, LOCAL_WINDOW, DAY_ARGS)
    assert public == mx._impact_bucket_rows_sql(conn, LOCAL_WINDOW, DAY_ARGS)
    assert sum(row["mi"] for row in public) == pytest.approx(0.2, abs=1e-12)
