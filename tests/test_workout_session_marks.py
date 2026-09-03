"""Explicit, durable user corrections for device-recorded workout rows."""
from __future__ import annotations

import pytest

from health_advisor import db
from health_advisor import metrics as mx


DAY = "2026-08-01"
SHORT_KEY = db.workout_key(
    "running", f"{DAY}T10:00:00Z", f"{DAY}T10:01:00Z"
)
LONG_KEY = db.workout_key(
    "running", f"{DAY}T10:01:25Z", f"{DAY}T10:03:25Z"
)


def _workout(start: str, end: str, duration_min: float) -> dict:
    return {
        "workout_type": "running",
        "start_utc": start,
        "end_utc": end,
        "local_date": DAY,
        "duration_min": duration_min,
        "energy_kcal": None,
        "distance_mi": None,
        "unit_distance": "mi",
        "source": "synthetic-source",
        "dedupe_key": db.workout_key("running", start, end),
    }


def _seed_false_start(conn) -> None:
    assert db.insert_workouts(conn, [
        _workout(f"{DAY}T10:00:00Z", f"{DAY}T10:01:00Z", 1.0),
        _workout(f"{DAY}T10:01:25Z", f"{DAY}T10:03:25Z", 2.0),
    ]) == 2
    db.insert_workout_events(conn, [
        {
            "workout_key": SHORT_KEY, "event_type": "segment",
            "start_utc": f"{DAY}T10:00:00Z", "end_utc": f"{DAY}T10:01:00Z",
            "duration_min": 1.0,
            "dedupe_key": db.workout_event_key(
                SHORT_KEY, "segment", f"{DAY}T10:00:00Z", 1.0),
        },
        {
            "workout_key": LONG_KEY, "event_type": "segment",
            "start_utc": f"{DAY}T10:01:25Z", "end_utc": f"{DAY}T10:03:25Z",
            "duration_min": 2.0,
            "dedupe_key": db.workout_event_key(
                LONG_KEY, "segment", f"{DAY}T10:01:25Z", 2.0),
        },
    ])
    conn.commit()


def test_mark_is_durable_attributed_and_does_not_rewrite_device_row(
        conn, tools, vault_path):
    _seed_false_start(conn)
    before = tuple(conn.execute(
        "SELECT workout_type, start_utc, end_utc, local_date, duration_min, "
        "energy_kcal, distance_mi, unit_distance, source, route_ref, "
        "avg_heart_rate, max_heart_rate, dedupe_key, hk_uuid "
        "FROM workouts WHERE dedupe_key = ?", (SHORT_KEY,)).fetchone())

    marked = tools.mark_workout_not_a_session(
        SHORT_KEY, reason="false start", source="user",
        marked_at="2026-08-02T12:00:00+00:00")
    assert marked["ok"] is True
    assert marked["mark"] == "not_a_session"
    assert marked["source"] == "user"
    assert marked["marked_at"] == "2026-08-02T12:00:00+00:00"
    assert marked["already_marked"] is False

    ro = db.connect(vault_path, read_only=True)
    try:
        mark = ro.execute(
            "SELECT workout_id, workout_key, mark, source, marked_at, reason "
            "FROM workout_session_marks WHERE workout_key = ?", (SHORT_KEY,)
        ).fetchone()
        after = tuple(ro.execute(
            "SELECT workout_type, start_utc, end_utc, local_date, duration_min, "
            "energy_kcal, distance_mi, unit_distance, source, route_ref, "
            "avg_heart_rate, max_heart_rate, dedupe_key, hk_uuid "
            "FROM workouts WHERE dedupe_key = ?", (SHORT_KEY,)).fetchone())
    finally:
        ro.close()

    assert dict(mark) == {
        "workout_id": mark["workout_id"], "workout_key": SHORT_KEY,
        "mark": "not_a_session", "source": "user",
        "marked_at": "2026-08-02T12:00:00+00:00", "reason": "false start",
    }
    assert after == before
    repeated = tools.mark_workout_not_a_session(
        SHORT_KEY, reason="different text", source="other",
        marked_at="2026-08-03T12:00:00+00:00")
    assert repeated["already_marked"] is True
    assert repeated["source"] == "user"


def test_list_and_segments_exclude_exactly_the_marked_false_start(
        conn, tools):
    _seed_false_start(conn)
    before = tools.list_workouts(start=DAY, end=DAY)
    assert before["count"] == 2
    assert before["total_in_range"] == 2
    assert before["excluded_count"] == 0
    assert before["workout_counts"] == [{"type": "running", "count": 2}]
    long_before = next(w for w in before["workouts"] if w["workout_key"] == LONG_KEY)
    assert tools.get_workout_segments(DAY)["count"] == 2

    assert tools.mark_workout_not_a_session(SHORT_KEY)["ok"] is True
    after = tools.list_workouts(start=DAY, end=DAY)
    assert after["count"] == 1
    assert after["total_in_range"] == 1
    assert after["excluded_count"] == 1
    assert after["workout_counts"] == [{"type": "running", "count": 1}]
    assert [w["workout_key"] for w in after["workouts"]] == [LONG_KEY]
    assert after["workouts"][0] == long_before

    detailed = tools.get_workout_segments(DAY)
    assert detailed["count"] == 1
    assert detailed["excluded_count"] == 1
    assert detailed["workouts"][0]["duration_min"] == pytest.approx(2.0)


def test_marked_workout_window_is_removed_from_impact_volume(conn, tools):
    _seed_false_start(conn)
    rows = []
    for start in (
            "10:00:00", "10:00:20", "10:00:40",
            "10:02:00", "10:02:20", "10:02:40",
            "10:03:00", "10:03:20"):
        ts = f"{DAY}T{start}Z"
        for metric, value, unit in (
                ("distance_walking_running", 0.02, "mi"),
                ("step_count", 50.0, "count")):
            rows.append({
                "metric": metric, "value": value, "unit": unit,
                "start_utc": ts, "end_utc": ts, "start_local": ts,
                "local_date": DAY, "source": "synthetic-source",
                "origin": "test", "dedupe_key": db.record_key(
                    metric, ts, ts, value, unit, "synthetic-source"),
            })
    db.insert_records(conn, rows)
    conn.commit()

    before = tools.get_impact_volume(start=DAY, end=DAY, by="day")["periods"][0]
    assert tools.mark_workout_not_a_session(SHORT_KEY)["ok"] is True
    after = tools.get_impact_volume(start=DAY, end=DAY, by="day")["periods"][0]

    assert before["jog_minutes"] - after["jog_minutes"] == pytest.approx(1.0)
    assert after["jog_minutes"] < before["jog_minutes"]
    assert conn.execute(
        "SELECT COUNT(*) FROM workouts WHERE dedupe_key = ?", (SHORT_KEY,)
    ).fetchone()[0] == 1


def test_mark_does_not_change_arbitrated_distance_rows(conn, tools):
    """Session identity marks must not alter the device-source arbitration."""
    day = "2026-08-25"
    start = f"{day}T11:00:00Z"
    end = f"{day}T11:01:00Z"
    key = db.workout_key("running", start, end)
    workout = _workout(start, end, 1.0)
    workout["local_date"] = day
    db.insert_workouts(conn, [workout])
    rows = []
    for source, value in (("synthetic-phone", 0.01), ("GymKit", 0.02)):
        ts = start
        rows.append({
            "metric": "distance_walking_running", "value": value, "unit": "mi",
            "start_utc": ts, "end_utc": ts, "start_local": ts,
            "local_date": day, "source": source, "origin": "test",
            "dedupe_key": db.record_key(
                "distance_walking_running", ts, ts, value, "mi", source),
        })
    rows.append({
        "metric": "step_count", "value": 50.0, "unit": "count",
        "start_utc": start, "end_utc": start, "start_local": start,
        "local_date": day, "source": "synthetic", "origin": "test",
        "dedupe_key": db.record_key(
            "step_count", start, start, 50.0, "count", "synthetic"),
    })
    db.insert_records(conn, rows)
    conn.commit()

    def admitted_distance_rows():
        clause, args = db._workout_arbitration(
            conn, "distance_walking_running",
            arbitration_window=(start, end), arbitration_window_kind="utc",
        )
        return [tuple(row) for row in conn.execute(
            "SELECT source, value FROM records "
            "WHERE metric = ? AND start_utc >= ? AND start_utc < ?" + clause
            + " ORDER BY source", ("distance_walking_running", start, end, *args)
        ).fetchall()]

    before_distance = admitted_distance_rows()
    before_buckets = mx.impact_bucket_rows(
        conn, "local_date BETWEEN ? AND ?", (day, day)
    )
    assert before_distance == [("GymKit", 0.02)]
    assert len(before_buckets) == 1
    assert before_buckets[0]["in_workout"] == 1
    assert before_buckets[0]["is_jog"] == 1

    assert tools.mark_workout_not_a_session(key)["ok"] is True
    after_distance = admitted_distance_rows()
    after_buckets = mx.impact_bucket_rows(
        conn, "local_date BETWEEN ? AND ?", (day, day)
    )

    assert after_distance == before_distance
    assert len(after_distance) == 1
    assert after_buckets[0]["in_workout"] == 0
    assert after_buckets[0]["is_jog"] == 0
    assert sum(r["is_jog"] for r in before_buckets) - sum(
        r["is_jog"] for r in after_buckets
    ) == 1
