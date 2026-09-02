"""The monthly treadmill benchmark is a small, comparable time series."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from health_advisor import benchmark, db


def _record_hr(conn, start: datetime, values: list[float], local_date: str) -> None:
    rows = []
    for i, value in enumerate(values):
        ts = (start + timedelta(seconds=20 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(("heart_rate", value, "count/min", ts, ts, local_date,
                     f"heart-rate|{ts}|{i}"))
    conn.executemany(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, local_date, "
        "source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, 'test', ?)", rows,
    )
    conn.commit()


def _treadmill_workout(conn, local_date: str, start: str) -> None:
    db.insert_workouts(conn, [{
        "workout_type": "running",
        "start_utc": start,
        "end_utc": (datetime.fromisoformat(start.replace("Z", "+00:00"))
                    + timedelta(minutes=35)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_date": local_date,
        "duration_min": 35.0,
        "distance_mi": 0.0,
        "energy_kcal": None,
        "unit_distance": "mi",
        "source": "GymKit",
        "route_ref": None,
        "dedupe_key": f"benchmark-{local_date}",
    }])
    conn.commit()


def test_four_stage_run_round_trips_and_series_aligns_dates(conn):
    for stage, pace in enumerate(("15:00", "14:00", "13:00", "12:00"), 1):
        benchmark.record(
            conn, date="2026-08-25", stage=stage, pace=pace,
            median_hr_last_two_min=130 + stage, talk_test="comfortable",
            temp_c=22.0, dew_point_c=15.0, notes="same treadmill",
        )
    benchmark.record(conn, date="2026-09-22", stage=1, pace="15:00",
                     median_hr_last_two_min=128, talk_test="comfortable")

    series = benchmark.series(conn)
    assert [(row["date"], row["stage"]) for row in series] == [
        ("2026-08-25", 1), ("2026-08-25", 2),
        ("2026-08-25", 3), ("2026-08-25", 4),
        ("2026-09-22", 1),
    ]
    assert series[0]["pace_min_per_mi"] == pytest.approx(15.0)
    assert series[3]["pace_min_per_mi"] == pytest.approx(12.0)
    assert series[0]["talk_test"] == "comfortable"
    assert series[0]["temp_c"] == pytest.approx(22.0)


def test_stopped_early_stores_completed_stages_only(conn):
    benchmark.record(conn, date="2026-08-25", stage=1, pace="15:00",
                     median_hr_last_two_min=140, talk_test="comfortable")
    benchmark.record(conn, date="2026-08-25", stage=2, pace="14:00",
                     median_hr_last_two_min=151, talk_test="not sure",
                     notes="stopped before stage 3")

    rows = benchmark.series(conn)
    assert [row["stage"] for row in rows] == [1, 2]
    assert all(row["median_hr_last_two_min"] is not None for row in rows)
    assert all(row["median_hr_last_two_min"] != 0 for row in rows)


def test_the_stage_median_comes_from_records_not_from_the_caller(conn):
    local_date = "2026-08-25"
    _treadmill_workout(conn, local_date, "2026-08-25T12:00:00Z")
    # Protocol stage 1 starts after the eight-minute warm-up. The last two
    # minutes are 12:10--12:12; their median is 142.5, not the typed 999.
    _record_hr(conn, datetime(2026, 8, 25, 12, 8, tzinfo=timezone.utc),
               list(range(130, 136)) + [140, 141, 142, 143, 144, 145], local_date)

    benchmark.record(conn, date=local_date, stage=1, pace="15:00",
                     median_hr_last_two_min=999)

    assert benchmark.series(conn)[0]["median_hr_last_two_min"] == pytest.approx(
        142.5, abs=0.5,
    )


def test_typed_median_is_used_only_when_no_raw_records_exist(conn):
    benchmark.record(conn, date="2026-08-25", stage=1, pace="15:00",
                     median_hr_last_two_min=141)
    assert benchmark.series(conn)[0]["median_hr_last_two_min"] == pytest.approx(141)


def test_the_stored_median_says_how_it_was_obtained(conn):
    # A protocol-derived window is an inference about session structure, not a
    # measurement. Storing it indistinguishably from an explicitly-bounded one
    # hands a later reader four numbers that look equally solid.
    local_date = "2026-08-25"
    _treadmill_workout(conn, local_date, "2026-08-25T12:00:00Z")
    _record_hr(conn, datetime(2026, 8, 25, 12, 8, tzinfo=timezone.utc),
               [140] * 12, local_date)

    benchmark.record(conn, date=local_date, stage=1, pace="15:00")
    assert benchmark.series(conn)[0]["median_source"] == "records:protocol"

    benchmark.record(conn, date=local_date, stage=1, pace="15:00",
                     stage_start_utc="2026-08-25T12:08:00Z")
    assert benchmark.series(conn)[0]["median_source"] == "records:explicit"


def test_a_typed_median_is_labelled_as_typed(conn):
    # Python did not own this number. The series must show that, because a
    # typed median is not comparable with a measured one.
    benchmark.record(conn, date="2026-08-25", stage=1, pace="15:00",
                     median_hr_last_two_min=140)
    row = benchmark.series(conn)[0]
    assert row["median_source"] == "typed"
    assert row["median_hr_last_two_min"] == 140
