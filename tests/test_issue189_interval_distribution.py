"""Regression coverage for #189's post-boundary interval attribution."""
from __future__ import annotations

import pytest

from health_advisor import analysis
from tests.fixture_loader import load_f84_wholeday


def test_post_boundary_intervals_distribute_inside_workouts(conn):
    """Long walking-window samples remain walking and do not create blocks."""
    load_f84_wholeday(conn)

    by_day = {
        row["period_start"]: row
        for row in analysis.impact_volume(
            conn, "2026-08-21", "2026-08-21", by="day"
        )
    }
    aug21 = by_day["2026-08-21"]
    assert aug21["jog_minutes"] == pytest.approx(0.0)
    assert aug21["jog_miles"] == pytest.approx(0.0)
    assert aug21["walk_minutes"] == pytest.approx(40.7, abs=0.05)
    assert aug21["walk_miles"] == pytest.approx(1.92, abs=0.01)

    workouts = {
        row["id"]: row
        for row in conn.execute(
            "SELECT id, start_utc, end_utc FROM workouts "
            "WHERE id IN (809, 810, 808, 811)"
        )
    }
    blocks = {
        workout_id: analysis.longest_block(
            conn, workout["start_utc"], workout["end_utc"]
        )
        for workout_id, workout in workouts.items()
    }

    for workout_id in (809, 810):
        assert blocks[workout_id]["reps"] == []
        assert blocks[workout_id]["qualified_min"] is None

    # The Thursday 8-on/3-off x3 and Saturday 10-on/3-off x2 runs are the
    # external block oracle recorded in #190; the attribution fix must not move
    # either session's governing block.
    assert blocks[808]["bridged_min"] == pytest.approx(8.0)
    assert blocks[808]["qualified_min"] == pytest.approx(8.0)
    assert blocks[811]["bridged_min"] == pytest.approx(10.0)
    assert blocks[811]["qualified_min"] == pytest.approx(10.0)


def test_no_post_boundary_bucket_implies_impossible_travel(conn):
    """The general form of #189, found on F-82 workout 815 (2026-08-28).

    A collapsed interval sample is visible without any oracle: it implies a
    pace no human produces. Before this fix, 815 carried buckets at 1.74 and
    0.75 min/mi holding 0.638 mi -- 18.5% of the workout -- inside 40 seconds,
    and the 5.0 min/mi floor then DISCARDED that distance, so the defect
    presented as an undercount rather than as an absurd figure. Asserting the
    floor over the whole fixture catches the class, not just the two days two
    issues happened to name.
    """
    from health_advisor import metrics as mx
    from tests.fixture_loader import load_f82_multi_source

    load_f82_multi_source(conn)
    rows = mx.impact_bucket_rows(
        conn, "local_date BETWEEN ? AND ?", ("2026-06-01", "2026-12-31"))
    bucket_min = mx.IMPACT_BUCKET_SECONDS / 60.0
    paces = [bucket_min / r["mi"] for r in rows if (r["mi"] or 0) > 0]
    impossible = [p for p in paces if p < mx.IMPACT_IMPLAUSIBLE_PACE_MIN]
    assert impossible == [], (
        f"{len(impossible)} bucket(s) imply faster-than-human travel; "
        f"fastest {min(impossible):.2f} min/mi")
