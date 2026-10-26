"""``impact_buckets.impact_bucket_rows_py`` must return the SQL's rows exactly.

Each test builds a small synthetic vault, runs the Python port and the engine's
SQL (``metrics._impact_bucket_rows_sql``) over the same window, and asserts
the two lists are equal, value for value, including the floats. The arithmetic
cases also assert a literal computed by hand from the sample durations, so a bug
that moved both implementations together would still fail here.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from health_advisor import db
from health_advisor import impact_buckets as ib
from health_advisor import metrics as mx

POST_DAY = "2026-09-02"      # after the default arbitration cutoff, 2026-08-21
PRE_DAY = "2026-03-02"       # before it
UTC_WINDOW = "start_utc >= ? AND start_utc < ?"
LOCAL_WINDOW = "local_date BETWEEN ? AND ?"
BUCKET_S = mx.IMPACT_BUCKET_SECONDS


def _iso(day: str, hms: str) -> str:
    return f"{day}T{hms}Z"


def _record(conn, metric, value, unit, start, end, day, source, tag):
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?)",
        (metric, value, unit, start, end, start, day, source,
         f"{tag}|{metric}|{start}|{end}|{source}"))


def _distance(conn, value, start, end, day, source, tag="d"):
    _record(conn, "distance_walking_running", value, "mi", start, end, day,
            source, tag)


def _steps(conn, value, start, end, day, source, tag="s"):
    _record(conn, "step_count", value, "count", start, end, day, source, tag)


def _heart(conn, value, start, day, tag="h"):
    _record(conn, "heart_rate", value, "count/min", start, start, day,
            "test", tag)


def _workout(conn, start, end, day):
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES "
        "('running', ?, ?, ?, ?, 'test', ?)",
        (start, end, day, (
            datetime.fromisoformat(end.replace("Z", "+00:00"))
            - datetime.fromisoformat(start.replace("Z", "+00:00"))
        ).total_seconds() / 60.0, db.workout_key("running", start, end)))
    return db.workout_key("running", start, end)


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "vault.db")
    db.init_db(connection)
    yield connection
    connection.close()


def _both(conn, predicate, args, **kwargs):
    """The Python rows and the SQL rows for one call."""
    return (ib.impact_bucket_rows_py(conn, predicate, args, **kwargs),
            mx._impact_bucket_rows_sql(conn, predicate, args, **kwargs))


def _bucket_of(hms_utc: str, day: str) -> int:
    """The 20-second bucket index for a UTC wall-clock time on ``day``."""
    epoch = int(datetime.fromisoformat(
        _iso(day, hms_utc).replace("Z", "+00:00")).timestamp())
    return epoch // BUCKET_S


def test_post_cutoff_interval_spans_buckets_in_proportion(conn):
    """A 60-second, 0.6-mile interval starting at :10 overlaps four buckets.

    The overlaps are 10, 20, 20 and 10 seconds, so the shares are 0.1, 0.2, 0.2
    and 0.1 miles. The SQL must agree with the Python port on every row.
    """
    start, end = _iso(POST_DAY, "12:00:10"), _iso(POST_DAY, "12:01:10")
    _distance(conn, 0.6, start, end, POST_DAY, "Apple Watch")
    conn.commit()

    py, sql = _both(conn, UTC_WINDOW, (_iso(POST_DAY, "12:00:00"),
                                       _iso(POST_DAY, "12:02:00")))
    assert py == sql
    assert [row["bkt"] for row in py] == [
        _bucket_of("12:00:00", POST_DAY) + i for i in range(4)]
    assert [row["mi"] for row in py] == [pytest.approx(v, abs=1e-12)
                                         for v in (0.1, 0.2, 0.2, 0.1)]


def test_pre_cutoff_samples_stay_one_bucket(conn):
    """Before the cutoff a sample is one bucket, whatever its duration.

    The point sample keeps its full 0.5 miles, and the 60-second interval keeps
    its full 1.0 miles in the bucket where it starts.
    """
    _distance(conn, 0.5, _iso(PRE_DAY, "09:00:05"), _iso(PRE_DAY, "09:00:05"),
              PRE_DAY, "Apple Watch", tag="point")
    _distance(conn, 1.0, _iso(PRE_DAY, "09:10:10"), _iso(PRE_DAY, "09:11:10"),
              PRE_DAY, "Apple Watch", tag="interval")
    conn.commit()

    py, sql = _both(conn, LOCAL_WINDOW, (PRE_DAY, PRE_DAY))
    assert py == sql
    assert [row["bkt"] for row in py] == [
        _bucket_of("09:00:00", PRE_DAY), _bucket_of("09:10:00", PRE_DAY)]
    assert [row["mi"] for row in py] == [0.5, 1.0]


def test_two_devices_on_one_walk_take_the_larger_step_rate(conn):
    """Two devices counting the same walk: the cadence is the larger, not the sum.

    Inside one workout, the watch counts 60 steps and the phone 40 in the same
    20-second bucket. The rate is 60 * 3 = 180 steps/min, not 300. The bucket
    also has a plausible pace and a running cadence, so it is a jog.
    """
    workout_start = _iso(POST_DAY, "12:00:00")
    workout_end = _iso(POST_DAY, "12:10:00")
    _workout(conn, workout_start, workout_end, POST_DAY)
    bucket_start = _iso(POST_DAY, "12:03:00")
    bucket_end = _iso(POST_DAY, "12:03:20")
    _distance(conn, 0.05, bucket_start, bucket_end, POST_DAY, "Apple Watch")
    _steps(conn, 60.0, bucket_start, bucket_end, POST_DAY, "Apple Watch",
           tag="watch")
    _steps(conn, 40.0, bucket_start, bucket_end, POST_DAY, "iPhone",
           tag="phone")
    conn.commit()

    py, sql = _both(conn, UTC_WINDOW, (workout_start, workout_end))
    assert py == sql
    assert len(py) == 1
    assert py[0]["cadence_spm"] == 180.0
    assert py[0]["in_workout"] == 1
    assert py[0]["is_jog"] == 1


def test_workout_marked_not_a_session_loses_its_jog(conn):
    """A marked workout stops counting as in-workout, so its bucket is not a jog."""
    workout_start = _iso(POST_DAY, "12:00:00")
    workout_end = _iso(POST_DAY, "12:10:00")
    key = _workout(conn, workout_start, workout_end, POST_DAY)
    bucket_start = _iso(POST_DAY, "12:03:00")
    bucket_end = _iso(POST_DAY, "12:03:20")
    _distance(conn, 0.05, bucket_start, bucket_end, POST_DAY, "Apple Watch")
    _steps(conn, 60.0, bucket_start, bucket_end, POST_DAY, "Apple Watch")
    conn.commit()

    before_py, before_sql = _both(conn, LOCAL_WINDOW, (POST_DAY, POST_DAY))
    assert before_py == before_sql
    assert before_py[0]["is_jog"] == 1

    db.mark_workout_not_a_session(conn, key)
    conn.commit()
    after_py, after_sql = _both(conn, LOCAL_WINDOW, (POST_DAY, POST_DAY))
    assert after_py == after_sql
    assert after_py[0]["in_workout"] == 0
    assert after_py[0]["is_jog"] == 0


def test_local_date_window_selects_only_its_days(conn):
    _distance(conn, 0.2, _iso(POST_DAY, "08:00:00"), _iso(POST_DAY, "08:00:00"),
              POST_DAY, "Apple Watch", tag="day")
    _distance(conn, 0.3, _iso("2026-09-03", "08:00:00"),
              _iso("2026-09-03", "08:00:00"), "2026-09-03", "Apple Watch",
              tag="next")
    conn.commit()

    py, sql = _both(conn, LOCAL_WINDOW, (POST_DAY, POST_DAY))
    assert py == sql
    assert {row["local_date"] for row in py} == {POST_DAY}


def test_utc_sub_window_keeps_samples_that_start_inside_it(conn):
    """A UTC sub-window selects by start time; each sample is spread over all its buckets.

    The 60-second interval starts at :10 inside the sub-window and its whole
    0.6 miles are kept, including the buckets after the window's end.
    """
    start, end = _iso(POST_DAY, "12:00:10"), _iso(POST_DAY, "12:01:10")
    _distance(conn, 0.6, start, end, POST_DAY, "Apple Watch")
    conn.commit()

    window = (_iso(POST_DAY, "12:00:00"), _iso(POST_DAY, "12:00:30"))
    py, sql = _both(conn, UTC_WINDOW, window,
                    arbitration_window=(_iso(POST_DAY, "11:00:00"),
                                        _iso(POST_DAY, "13:00:00")),
                    arbitration_window_kind="utc")
    assert py == sql
    assert len(py) == 4
    assert sum(row["mi"] for row in py) == pytest.approx(0.6, abs=1e-12)


def test_day_rule_drops_iphone_when_a_watch_records_the_day(conn):
    """Post-cutoff, the iPhone's distance loses to the watch on the same local date.

    Only the watch's 0.4 miles survive, in the bucket where both samples sit.
    """
    stamp = _iso(POST_DAY, "10:00:00")
    _distance(conn, 0.4, stamp, stamp, POST_DAY, "Apple Watch", tag="w")
    _distance(conn, 0.7, stamp, stamp, POST_DAY, "iPhone", tag="p")
    conn.commit()

    py, sql = _both(conn, LOCAL_WINDOW, (POST_DAY, POST_DAY))
    assert py == sql
    assert [row["mi"] for row in py] == [0.4]


def test_workout_rule_drops_iphone_and_watch_inside_a_gymkit_workout(conn):
    """Inside a post-cutoff workout with GymKit, iPhone and watch samples lose.

    Only the GymKit sample's 0.25 miles survive.
    """
    workout_start = _iso(POST_DAY, "11:00:00")
    workout_end = _iso(POST_DAY, "11:30:00")
    _workout(conn, workout_start, workout_end, POST_DAY)
    stamp = _iso(POST_DAY, "11:05:00")
    _distance(conn, 0.25, stamp, stamp, POST_DAY, "GymKit", tag="g")
    _distance(conn, 0.5, stamp, stamp, POST_DAY, "Apple Watch", tag="w")
    _distance(conn, 0.9, stamp, stamp, POST_DAY, "iPhone", tag="p")
    conn.commit()

    py, sql = _both(conn, LOCAL_WINDOW, (POST_DAY, POST_DAY))
    assert py == sql
    assert [row["mi"] for row in py] == [0.25]


def test_unknown_predicate_delegates_to_the_sql(conn):
    _distance(conn, 0.3, _iso(POST_DAY, "08:00:00"), _iso(POST_DAY, "08:00:00"),
              POST_DAY, "Apple Watch")
    conn.commit()
    py, sql = _both(conn, "local_date >= ? AND local_date <= ?",
                    (POST_DAY, POST_DAY))
    assert py == sql
    assert py


def test_bad_arbitration_kind_raises_as_the_sql_does(conn):
    with pytest.raises(ValueError):
        ib.impact_bucket_rows_py(conn, LOCAL_WINDOW, (POST_DAY, POST_DAY),
                                 arbitration_window_kind="bogus")
    with pytest.raises(ValueError):
        mx._impact_bucket_rows_sql(conn, LOCAL_WINDOW,
                                   (POST_DAY, POST_DAY),
                                        arbitration_window_kind="bogus")
