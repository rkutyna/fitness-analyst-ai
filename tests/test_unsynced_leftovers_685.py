"""Consumer #685 leftovers (engine half, after 023ce1c).

1. get_impact_volume's description must name both unmeasured states, not_synced
   and not_recorded, the way get_block_structure's description does.
2. A walk-only day whose walks have no samples is unmeasured, not a measured
   0.0 "no running session". A walk-only day whose walk HAS samples (it walked,
   measured, no jog block) keeps the old 0.0 dict byte-for-byte, and so does a
   day with no workouts at all.

Oracles are literals written from the rule, not the code's own output.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from health_advisor import metrics as mx
from health_advisor import mcp_server

BUCKET_MIN = mx.IMPACT_BUCKET_SECONDS / 60.0
NOT_SYNCED_NOTE = ("this session's minute-by-minute distance and step samples "
                   "are not in the vault yet, so its blocks cannot be measured")
NO_SESSION_NOTE = "no session with measurable block structure on this day"

WALK_UNSYNCED_DAY = "2026-07-15"
WALK_MEASURED_DAY = "2026-07-16"
NO_WORKOUT_DAY = "2026-07-17"


def _iso(base: str, i: int) -> str:
    t = datetime.fromisoformat(base.replace("Z", "+00:00"))
    return (t + timedelta(seconds=mx.IMPACT_BUCKET_SECONDS * i)
            ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(conn, metric, start, value, unit):
    conn.execute(
        "INSERT INTO records (metric, start_utc, end_utc, local_date, value, "
        "unit, source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, 'test', ?)",
        (metric, start, start, start[:10], value, unit,
         f"{metric}|{start}|{value}"))


def _walk(conn, base: str, day: str, *, samples: bool):
    """A 12-minute walk. With samples: distance at 20 min/mi and 100 steps/min,
    a walk that is measured and has no jog bucket. Without: no rows at all."""
    if samples:
        for i in range(36):
            s = _iso(base, i)
            _record(conn, "distance_walking_running", s, BUCKET_MIN / 20.0, "mi")
            _record(conn, "step_count", s, 100.0 / 3.0, "count")
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, avg_heart_rate, source, dedupe_key) "
        "VALUES ('walking', ?, ?, ?, ?, NULL, 'test', ?)",
        (base, _iso(base, 36), day, 36 * BUCKET_MIN, f"w|{base}|walking"))


def test_impact_volume_description_names_not_synced_and_not_recorded():
    doc = " ".join(mcp_server.get_impact_volume.__doc__.split())
    assert '"not_synced"' in doc
    assert '"not_recorded"' in doc


def test_unsynced_walk_only_day_is_none_not_a_measured_zero(conn, tools):
    _walk(conn, "2026-07-15T12:00:00Z", WALK_UNSYNCED_DAY, samples=False)
    conn.commit()

    out = tools.get_block_structure(day=WALK_UNSYNCED_DAY)

    assert out["longest_block_min"] is None
    assert out["samples"] == "not_synced"
    assert out["note"] == NOT_SYNCED_NOTE
    assert out["sessions"] == []
    assert out["qualified_block_min"] is None
    assert out["qualified_is_longest"] is None
    assert out["excluded_count"] == 0


def test_measured_walk_only_day_keeps_todays_dict_literal(conn, tools):
    _walk(conn, "2026-07-16T12:00:00Z", WALK_MEASURED_DAY, samples=True)
    conn.commit()

    out = tools.get_block_structure(day=WALK_MEASURED_DAY)

    assert out == {
        "day": "2026-07-16",
        "sessions": [],
        "longest_block_min": 0.0,
        "excluded_count": 0,
        "qualified_block_min": None,
        "qualified_is_longest": None,
        "note": NO_SESSION_NOTE,
    }
    assert "samples" not in out


def test_day_with_no_workouts_keeps_todays_dict_literal(conn, tools):
    conn.commit()

    out = tools.get_block_structure(day=NO_WORKOUT_DAY)

    assert out == {
        "day": "2026-07-17",
        "sessions": [],
        "longest_block_min": 0.0,
        "excluded_count": 0,
        "qualified_block_min": None,
        "qualified_is_longest": None,
        "note": NO_SESSION_NOTE,
    }
