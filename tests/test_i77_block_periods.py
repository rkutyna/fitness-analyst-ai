"""engine #77 item 4b: two block tools must not name days the data doesn't cover.

(A) analysis.block_comparison published the full requested window even when the
recent block ran past the last date actually present -- as_of a few days after
the vault's last row still produced a period string reaching to `as_of`, with
no flag distinguishing it from a fully-covered block. Fixed to clamp the
published period to the rows the block actually used and to say so via
"partial".

(B) deepdive_mcp._claim_period_vocabulary's single-bucket fallback used the
block-level period dict's own "end", which for a weeks_per_block=1 block is
just that block's *start* (period_starts has one element, so "end" ==
"start"). That published claim_period '<day>:<day>' for a seven-day total.
Fixed to prefer the underlying row's own (possibly clamped-for-partial)
period string, falling back to start+6 only when no row is available.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from health_advisor import deepdive_mcp as D
from tests.conftest import seed_metric


# --------------------------------------------------------------------------
# (A) analysis.block_comparison
# --------------------------------------------------------------------------

def test_recent_block_period_clamps_to_the_last_date_present(tools, conn, monkeypatch):
    """Three as_of values past a fixed data end, matching the #77 measurement:
    the published 'recent' period must end at the data end (not at as_of), and
    'partial' must say so, while n/28 keeps behaving as it always did."""
    monkeypatch.setattr(
        "health_advisor.analysis.metric_noise_floor",
        lambda conn, metric, as_of: {"sd_day": 3.86, "rho": 0.16},
    )
    data_end = date(2029, 12, 31)
    # 56 days of history ending at data_end: enough that the *previous* 28-day
    # block is always fully covered regardless of how far as_of drifts past
    # data_end, so only the recent block's clamp is under test.
    seed_metric(conn, "resting_heart_rate",
                (data_end - timedelta(days=55)).isoformat(), [60.0] * 56)

    cases = [
        (data_end, 28, False),               # as_of == data end: full window
        (data_end + timedelta(days=3), 25, True),   # 3 days past
        (data_end + timedelta(days=5), 23, True),   # 5 days past
    ]
    for as_of, expected_n, expected_partial in cases:
        result = tools.get_block_comparison(
            "resting_heart_rate", block_weeks=4, as_of=as_of.isoformat())
        recent = result["blocks"]["recent"]
        window_start = as_of - timedelta(days=27)
        assert recent["n"] == expected_n, as_of
        assert recent["partial"] is expected_partial, as_of
        assert recent["period"] == f"{window_start.isoformat()}:{data_end.isoformat()}", as_of
        # The previous block never runs past the data: never partial.
        assert result["blocks"]["previous"]["partial"] is False, as_of
        assert result["blocks"]["previous"]["n"] == 28, as_of
        # A mean is still published in every one of these cases (the gate is
        # coverage, 75%, not exact completeness).
        assert "mean" in recent


def test_fully_empty_recent_block_is_still_refused_unpartialled(tools, conn):
    """The existing insufficient_coverage path (fully empty block) is untouched:
    no mean, no 'partial' key invented for a block that was refused outright."""
    as_of = date(2029, 12, 31)
    seed_metric(conn, "resting_heart_rate",
                (as_of - timedelta(days=27)).isoformat(), [60.0] * 28)

    result = tools.get_block_comparison(
        "resting_heart_rate", block_weeks=4,
        as_of=(as_of + timedelta(days=40)).isoformat())

    assert result["status"] == "insufficient_coverage"
    assert "mean" not in result["blocks"]["recent"]
    assert "partial" not in result["blocks"]["recent"]


# --------------------------------------------------------------------------
# (B) deepdive_mcp._claim_period_vocabulary for weeks_per_block=1
# --------------------------------------------------------------------------

def _one_week_block(start: str, row_period: str) -> dict:
    """The shape mcp_server._impact_block_comparison's block() emits for a
    single-row (weeks_per_block=1) block -- verified by reading that function,
    which this test scope may not edit."""
    return {
        "metric": "jog_minutes",
        "period": {"start": start, "end": start, "period_starts": [start]},
        "period_starts": [start],
        "weeks": [{"metric": "jog_minutes", "period": row_period,
                   "field": "jog_minutes", "value": 30.0,
                   "days_covered": 7, "days_expected": 7,
                   "partial": False, "no_data": False}],
        "total": 30.0, "mean": 30.0,
    }


def test_one_week_block_claim_period_spans_seven_days_not_start_start():
    block = _one_week_block("2029-12-17", "2029-12-17:2029-12-23")

    vocabulary = D._claim_period_vocabulary(block)

    assert vocabulary[0]["claim_period"] == "2029-12-17:2029-12-23"
    start, end = vocabulary[0]["claim_period"].split(":")
    span_days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    assert span_days == 7


def test_one_week_block_claim_period_uses_the_rows_own_clamped_span():
    """A partial trailing week (near the vault's data end) must publish ITS
    clamped span, not an invented full week and not start:start."""
    block = _one_week_block("2029-12-24", "2029-12-24:2029-12-28")
    block["weeks"][0]["days_covered"] = 5
    block["weeks"][0]["partial"] = True

    vocabulary = D._claim_period_vocabulary(block)

    assert vocabulary[0]["claim_period"] == "2029-12-24:2029-12-28"


def _pin_as_of(conn, day: str) -> None:
    """Give the test vault a horizon at `day` (analysis._as_of reads
    MAX(date) from daily_metrics, and otherwise falls back to the real host
    date -- which would put every 2029/2030 fixture date after the horizon)."""
    conn.execute(
        "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, last, unit) "
        "VALUES ('step_count', ?, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 'count')",
        (day,),
    )
    conn.commit()


def _emit_jog_minutes(conn, local_date: str, minutes: float):
    """One workout of `minutes` jog time on `local_date`, at a cadence/pace the
    classifier scores as jogging -- the same construction test_impact_volume_tool
    uses for its block-comparison fixtures."""
    t0 = datetime.fromisoformat(f"{local_date}T12:00:00+00:00")
    n_buckets = round(minutes * 3)
    per_bucket_mi = 20.0 / (12.0 * 60.0)
    end = t0 + timedelta(seconds=n_buckets * 20)
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('running', ?, ?, ?, ?, 'test', ?)",
        (t0.isoformat(), end.isoformat(), local_date, n_buckets / 3.0,
         f"i77-workout-{local_date}"),
    )
    for i in range(n_buckets):
        ts = (t0 + timedelta(seconds=i * 20)).isoformat()
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('distance_walking_running', ?, 'mi', ?, ?, ?, ?, 't', 't', ?)",
            (per_bucket_mi, ts, ts, ts, local_date, f"i77-{local_date}-{i}"))
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('step_count', 47.0, 'count', ?, ?, ?, ?, 't', 't', ?)",
            (ts, ts, ts, local_date, f"i77-steps-{local_date}-{i}"))
    conn.commit()


def test_real_get_impact_volume_one_week_block_full_week_spans_seven_days(
        tools, conn):
    _pin_as_of(conn, "2029-12-28")
    _emit_jog_minutes(conn, "2029-12-10", 30.0)  # prior (full week)
    _emit_jog_minutes(conn, "2029-12-17", 30.0)  # recent for last_complete_week
    _emit_jog_minutes(conn, "2029-12-24", 20.0)  # trailing, clipped partial below
    conn.commit()

    out = tools.get_impact_volume(
        "2029-12-10", "2029-12-28", by="week", weeks_per_block=1,
        anchor="last_complete_week")

    vocabulary = D._claim_period_vocabulary(out["block_comparison"])
    by_metric_period = {item["claim_period"] for item in vocabulary}
    # last_complete_week drops the partial 12-24 week, so both published
    # one-week blocks are full, undamaged weeks.
    assert "2029-12-17:2029-12-23" in by_metric_period
    assert "2029-12-10:2029-12-16" in by_metric_period
    for claim_period in by_metric_period:
        start, end = claim_period.split(":")
        assert (date.fromisoformat(end) - date.fromisoformat(start)).days + 1 == 7


def test_real_get_impact_volume_one_week_block_partial_trailing_week_is_clamped(
        tools, conn):
    _pin_as_of(conn, "2029-12-28")
    _emit_jog_minutes(conn, "2029-12-17", 30.0)  # prior (full week)
    _emit_jog_minutes(conn, "2029-12-24", 20.0)  # recent: clipped to 5 days
    conn.commit()

    out = tools.get_impact_volume(
        "2029-12-10", "2029-12-28", by="week", weeks_per_block=1,
        anchor="last_day_with_data")

    recent = out["block_comparison"]["blocks"]["recent"]
    assert recent["period_starts"] == ["2029-12-24"]

    vocabulary = D._claim_period_vocabulary(out["block_comparison"])
    recent_claims = [item["claim_period"] for item in vocabulary
                     if item["claim_period"].startswith("2029-12-24")]
    # The explicit end (2029-12-28) clips this week to 5 days: 24..28. The old
    # code published '2029-12-24:2029-12-24' here (start:start).
    assert recent_claims == ["2029-12-24:2029-12-28"]


def _claim_periods(payload):
    return {item["claim_period"] for item in D._claim_period_vocabulary(payload)}


def test_republished_period_resolves_the_same_regardless_of_walk_order():
    """A presentation leaf republishing a block's period dict WITHOUT its rows
    must get the row-derived end even when it is walked before the block --
    and after a JSON round-trip, where object identity is gone."""
    import json
    period = {"start": "2029-12-24", "end": "2029-12-24",
              "period_starts": ["2029-12-24"]}
    block = {"metric": "jog_minutes", "period": dict(period),
             "weeks": [{"metric": "jog_minutes",
                        "period": "2029-12-24:2029-12-28"}]}
    leaf = {"metric": "jog_minutes", "period": dict(period), "value": 1}
    payload = {"a_presentations": leaf, "b_block": block}  # leaf walked first
    assert _claim_periods(json.loads(json.dumps(payload))) == {
        "2029-12-24:2029-12-28"}


def test_multi_week_block_with_a_partial_trailing_week_ends_on_its_row():
    """The spacing rule (last start + 6) overshoots a clamped trailing week;
    the last row's own period is exact."""
    period = {"start": "2029-12-10", "end": "2029-12-24",
              "period_starts": ["2029-12-10", "2029-12-17", "2029-12-24"]}
    block = {"metric": "jog_minutes", "period": period,
             "weeks": [{"period": "2029-12-10:2029-12-16"},
                       {"period": "2029-12-17:2029-12-23"},
                       {"period": "2029-12-24:2029-12-27"}]}
    assert _claim_periods(block) == {"2029-12-10:2029-12-27"}
