"""engine #77 last edge: a block's period_label must not name days past the
data when its trailing week is partial.

``mcp_server._impact_block_comparison``'s ``block()`` publishes a block's
``period`` as bucket STARTS only (``{"start", "end", "period_starts"}``, where
``end`` is itself a start, not a real end) beside a ``weeks`` list whose rows
are already clamped to the days actually present -- with
``anchor="last_day_with_data"`` the last row can be a partial week
(``2029-12-24:2029-12-27`` instead of a full ``2029-12-24:2029-12-30``).
``fact_template._period_starts_label`` used to always name the block's span
end as ``period_starts[-1] + 6``, which is up to six days past the data for a
partial trailing week. ``deepdive_mcp._claim_period_vocabulary`` was fixed to
read the last row's own clamped end for its ``claim_period`` vocabulary; this
closes the same gap for the label a user actually reads.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from health_advisor import deepdive_mcp as D
from health_advisor import fact_template


def _block_ledger(rows: list[dict], *, include_weeks: bool = True):
    """A ledger record wrapping the exact node shape block() emits (metric,
    period-as-starts, and a weeks list all on the same node) -- built the way
    tests/test_fact_template.py's `_ledger` helper builds a synthetic result,
    and matching the shape tests/test_i77_block_periods.py's `_one_week_block`
    verified against block() itself.
    """
    starts = [row["start"] for row in rows]
    period = {"start": starts[0], "end": starts[-1], "period_starts": starts}
    values = [row["value"] for row in rows]
    total = round(sum(values), 1)
    mean = round(total / len(values), 1)
    result = {
        "metric": "jog_minutes",
        "period": period,
        "period_starts": starts,
        "total": total,
        "mean": mean,
    }
    if include_weeks:
        result["weeks"] = [
            {"metric": "jog_minutes", "period": row["period"],
             "field": "jog_minutes", "value": row["value"],
             "days_covered": row["days_covered"], "days_expected": 7,
             "partial": row["partial"], "no_data": False}
            for row in rows
        ]
    ledger = [{
        "sequence": 1,
        "tool_name": "get_impact_volume",
        "arguments": {},
        "result": result,
    }]
    return ledger, period


_PRIOR_THREE_FULL_WEEKS = [
    {"start": "2029-12-10", "period": "2029-12-10:2029-12-16",
     "value": 30.0, "days_covered": 7, "partial": False},
    {"start": "2029-12-17", "period": "2029-12-17:2029-12-23",
     "value": 30.0, "days_covered": 7, "partial": False},
]


def test_partial_trailing_week_labels_the_rows_own_clamped_end():
    """2029-12-27 is a Thursday: the block's last week only has 4 of its 7
    days, and the label must say Thu Dec 27, not Sun Dec 30 (start + 6)."""
    rows = _PRIOR_THREE_FULL_WEEKS + [
        {"start": "2029-12-24", "period": "2029-12-24:2029-12-27",
         "value": 20.0, "days_covered": 4, "partial": True},
    ]
    ledger, period = _block_ledger(rows)

    facts = fact_template.build_fact_set(ledger)

    assert date(2029, 12, 27).strftime("%a") == "Thu"
    key = fact_template.fact_key("jog_minutes", period, "period_label")
    assert facts[key]["value"] == "the 3 weeks from Mon Dec 10 to Thu Dec 27"
    assert "Dec 30" not in facts[key]["value"]


def test_full_trailing_week_keeps_todays_label():
    """A full, unclamped trailing week's own end (Dec 30) already equals
    start + 6, so the override is a no-op and the label is unchanged."""
    rows = _PRIOR_THREE_FULL_WEEKS + [
        {"start": "2029-12-24", "period": "2029-12-24:2029-12-30",
         "value": 28.0, "days_covered": 7, "partial": False},
    ]
    ledger, period = _block_ledger(rows)

    facts = fact_template.build_fact_set(ledger)

    key = fact_template.fact_key("jog_minutes", period, "period_label")
    assert facts[key]["value"] == "the 3 weeks from Mon Dec 10 to Sun Dec 30"


def test_block_node_without_weeks_keeps_todays_label():
    """No `weeks` sibling on the node -- e.g. a bare block period republished
    elsewhere without its rows -- means no row end to read, so the label
    keeps naming start + 6 exactly as it did before this change."""
    rows = _PRIOR_THREE_FULL_WEEKS + [
        {"start": "2029-12-24", "period": "2029-12-24:2029-12-27",
         "value": 20.0, "days_covered": 4, "partial": True},
    ]
    ledger, period = _block_ledger(rows, include_weeks=False)

    facts = fact_template.build_fact_set(ledger)

    key = fact_template.fact_key("jog_minutes", period, "period_label")
    assert facts[key]["value"] == "the 3 weeks from Mon Dec 10 to Sun Dec 30"


# --------------------------------------------------------------------------
# End-to-end: a real get_impact_volume call, anchor="last_day_with_data".
# Helpers copied from tests/test_i77_block_periods.py's shape (that file may
# not be edited by this task's scope).
# --------------------------------------------------------------------------

def _pin_as_of(conn, day: str) -> None:
    conn.execute(
        "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, last, unit) "
        "VALUES ('step_count', ?, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 'count')",
        (day,),
    )
    conn.commit()


def _emit_jog_minutes(conn, local_date: str, minutes: float):
    t0 = datetime.fromisoformat(f"{local_date}T12:00:00+00:00")
    n_buckets = round(minutes * 3)
    per_bucket_mi = 20.0 / (12.0 * 60.0)
    end = t0 + timedelta(seconds=n_buckets * 20)
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('running', ?, ?, ?, ?, 'test', ?)",
        (t0.isoformat(), end.isoformat(), local_date, n_buckets / 3.0,
         f"i77-label-workout-{local_date}"),
    )
    for i in range(n_buckets):
        ts = (t0 + timedelta(seconds=i * 20)).isoformat()
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('distance_walking_running', ?, 'mi', ?, ?, ?, ?, 't', 't', ?)",
            (per_bucket_mi, ts, ts, ts, local_date, f"i77-label-{local_date}-{i}"))
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('step_count', 47.0, 'count', ?, ?, ?, ?, 't', 't', ?)",
            (ts, ts, ts, local_date, f"i77-label-steps-{local_date}-{i}"))
    conn.commit()


def test_real_get_impact_volume_period_label_matches_claim_period_vocabulary(
        tools, conn):
    """The label a user reads and the vocabulary the model is told to copy
    verbatim must name the same end date for the same partial trailing week."""
    _pin_as_of(conn, "2029-12-27")
    _emit_jog_minutes(conn, "2029-11-19", 30.0)
    _emit_jog_minutes(conn, "2029-11-26", 30.0)
    _emit_jog_minutes(conn, "2029-12-03", 30.0)
    _emit_jog_minutes(conn, "2029-12-10", 30.0)
    _emit_jog_minutes(conn, "2029-12-17", 30.0)
    _emit_jog_minutes(conn, "2029-12-24", 20.0)  # trailing, clipped to 4 days

    out = tools.get_impact_volume(
        "2029-11-19", "2029-12-27", by="week", weeks_per_block=3,
        anchor="last_day_with_data")

    recent = out["block_comparison"]["blocks"]["recent"]
    assert recent["period_starts"][-1] == "2029-12-24"
    assert recent["weeks"][-1]["period"] == "2029-12-24:2029-12-27"

    ledger = [{"sequence": 1, "tool_name": "get_impact_volume",
               "arguments": {}, "result": out}]
    facts = fact_template.build_fact_set(ledger)
    label_key = fact_template.fact_key(
        "jog_minutes", recent["period"], "period_label")
    label = facts[label_key]["value"]

    vocabulary = D._claim_period_vocabulary(out["block_comparison"])
    recent_claim = next(
        item["claim_period"] for item in vocabulary
        if item["ledger_period"] == recent["period"])
    claim_end = date.fromisoformat(recent_claim.split(":")[1])

    assert recent_claim == "2029-12-10:2029-12-27"
    assert claim_end == date(2029, 12, 27)
    assert label == "the 3 weeks from Mon Dec 10 to Thu Dec 27"
