"""Public workout-for-window matcher tests using synthetic vault data."""
from __future__ import annotations

from datetime import datetime

import pytest

from health_advisor import db


def _workout(key: str, start: str, end: str, source: str) -> dict:
    duration_min = (
        datetime.fromisoformat(end) - datetime.fromisoformat(start)
    ).total_seconds() / 60
    return {
        "workout_type": "running",
        "start_utc": start,
        "end_utc": end,
        "local_date": start[:10],
        "duration_min": duration_min,
        "energy_kcal": None,
        "distance_mi": None,
        "unit_distance": None,
        "source": source,
        "dedupe_key": key,
        "hk_uuid": None,
    }


def _insert_workout(conn, key: str, start: str, end: str,
                    source: str = "Synthetic Watch") -> int:
    db.insert_workouts(conn, [_workout(key, start, end, source)])
    return conn.execute(
        "SELECT id FROM workouts WHERE dedupe_key = ?", (key,)
    ).fetchone()[0]


WINDOW_START = "2099-01-01T10:10:00+00:00"
WINDOW_END = "2099-01-01T10:30:00+00:00"


def test_window_matching_one_workout_returns_id(conn):
    workout_id = _insert_workout(
        conn, "one", WINDOW_START, WINDOW_END,
    )

    assert db.workout_for_window(conn, WINDOW_START, WINDOW_END) == workout_id


def test_largest_overlap_wins(conn):
    larger = _insert_workout(
        conn, "larger", "2099-01-01T10:09:00+00:00",
        "2099-01-01T10:31:00+00:00",
    )
    _insert_workout(
        conn, "smaller", "2099-01-01T10:10:00+00:00",
        "2099-01-01T10:29:00+00:00", "Other Synthetic Watch",
    )

    assert db.workout_for_window(conn, WINDOW_START, WINDOW_END) == larger


@pytest.mark.parametrize(
    ("workout_start", "workout_end", "window_start", "window_end"),
    [
        ("2099-01-01T10:00:00+00:00", "2099-01-01T10:30:00+00:00",
         "2099-01-01T10:00:00+00:00", "2099-01-01T10:26:00+00:00"),
        ("2099-01-01T10:00:00+00:00", "2099-01-01T10:26:00+00:00",
         "2099-01-01T10:00:00+00:00", "2099-01-01T10:30:00+00:00"),
    ],
)
def test_less_than_90_percent_coverage_returns_none(
    conn, workout_start, workout_end, window_start, window_end,
):
    _insert_workout(conn, "under-covered", workout_start, workout_end)

    assert db.workout_for_window(conn, window_start, window_end) is None


def test_equal_overlap_source_name_breaks_tie(conn):
    _insert_workout(
        conn, "other-source", "2099-01-01T10:00:00+00:00",
        "2099-01-01T10:30:00+00:00", "Other Synthetic Device",
    )
    preferred = _insert_workout(
        conn, "preferred-source", "2099-01-01T09:59:00+00:00",
        "2099-01-01T10:31:00+00:00", "Preferred Synthetic Device",
    )

    assert db.workout_for_window(
        conn, "2099-01-01T10:00:00+00:00", "2099-01-01T10:30:00+00:00",
        source_name=" Preferred   Synthetic ",
    ) == preferred


def test_public_wrapper_matches_private_matcher(conn):
    workout_id = _insert_workout(
        conn, "parity", "2099-01-01T10:09:00+00:00",
        "2099-01-01T10:31:00+00:00", "Parity Synthetic Device",
    )
    item = {
        "start_utc": WINDOW_START,
        "end_utc": WINDOW_END,
        "source_name": "Parity Synthetic",
    }

    assert db.workout_for_window(
        conn, item["start_utc"], item["end_utc"],
        source_name=item["source_name"],
    ) == db._workout_parent_id(
        conn, item, uuid_key="workout_hk_uuid",
    ) == workout_id
