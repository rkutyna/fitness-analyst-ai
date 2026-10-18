"""The bedtime anchor is a caller's parameter; the engine constant is a default.

``compliance_from_bedtimes`` and ``plan_compliance`` used to read the module
constant ``BEDTIME_ANCHOR_H`` with no way to pass another. A caller that holds
its own plan now supplies that plan's anchor on the since-noon scale. Omitting
it keeps the engine default, so no existing caller changes.

The values here are generic clock times (22:00, 00:00, and the existing default
23:00). The engine is public, so no deployment anchor is written into it.
"""
from __future__ import annotations

import pytest

from health_advisor import metrics as mx
from health_advisor import sleep_regularity as sr

BEDTIMES = [10.5, 11.0, 11.25, 12.25, 12.75]  # 22:30, 23:00, 23:15, 00:15, 00:45


def test_default_anchor_is_the_engine_constant():
    out = sr.compliance_from_bedtimes(BEDTIMES)
    assert mx.BEDTIME_ANCHOR_H == 11.0  # the default itself is unchanged
    assert out["anchor_h"] == mx.BEDTIME_ANCHOR_H
    assert out["inside_anchor"] == 2     # 22:30 and exactly 23:00
    assert out["social_nights"] == 2     # 23:15 and 00:15
    assert out["past_limit"] == 1        # 00:45


def test_a_passed_anchor_moves_the_inside_band():
    out = sr.compliance_from_bedtimes(BEDTIMES, anchor_h=10.0)
    assert out["anchor_h"] == 10.0
    assert out["inside_anchor"] == 0     # nothing at or before 22:00
    assert out["social_nights"] == 4     # everything from 22:00 to 00:30
    assert out["past_limit"] == 1
    assert out["inside_anchor_pct"] == 0.0


def test_the_social_band_starts_just_after_the_passed_anchor():
    # Anchor at midnight: the boundary itself is inside, one minute later is social.
    out = sr.compliance_from_bedtimes(
        [11.75, 12.0, 12.25, 12.5, 12.501], anchor_h=12.0)
    assert out["inside_anchor"] == 2     # 11.75 and 12.0 are at or before the anchor
    assert out["social_nights"] == 2     # 12.25 and exactly 12.5
    assert out["past_limit"] == 1        # one minute past 00:30


def test_the_passed_anchor_is_echoed_on_the_empty_refusal_too():
    out = sr.compliance_from_bedtimes([], anchor_h=10.0)
    assert out["n_nights"] == 0
    assert out["inside_anchor_pct"] is None
    assert out["anchor_h"] == 10.0


def test_plan_compliance_passes_the_anchor_through(conn):
    values = [10.5, 11.0, 11.25]
    for i, value in enumerate(values, start=16):
        day = f"2026-08-{i:02d}"
        conn.execute(
            "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, last, unit) "
            "VALUES ('sleep_bedtime', ?, 1, ?, ?, ?, ?, ?, 'hr')",
            (day, value, value, value, value, value),
        )
    conn.commit()

    default = sr.plan_compliance(conn, "2026-08-16", "2026-08-18")
    assert default["anchor_h"] == mx.BEDTIME_ANCHOR_H
    assert default["inside_anchor"] == 2

    moved = sr.plan_compliance(conn, "2026-08-16", "2026-08-18", anchor_h=10.0)
    assert moved["anchor_h"] == 10.0
    assert moved["inside_anchor"] == 0
    assert moved["social_nights"] == 3
    assert moved["n_nights"] == 3


@pytest.mark.parametrize("anchor_h", [10.0, 12.0])
def test_inside_and_past_anchor_partition_every_night(anchor_h):
    out = sr.compliance_from_bedtimes(BEDTIMES, anchor_h=anchor_h)
    past_anchor = sum(1 for b in BEDTIMES if b > anchor_h)
    assert out["inside_anchor"] + out["social_nights"] + out["past_limit"] == len(BEDTIMES)
    assert out["social_nights"] + out["past_limit"] == past_anchor
