"""The receiver/backfill seam must replace coarse rows exactly once."""
from __future__ import annotations

from health_advisor import db


def _row(metric, value, day, *, origin, start):
    end = start
    return {
        "metric": metric, "value": value, "unit": "count",
        "start_utc": start, "end_utc": end,
        "start_local": f"{day} 08:00:00", "local_date": day,
        "source": "Watch", "origin": origin,
        "dedupe_key": f"{origin}|{metric}|{day}|{start}|{value}",
    }


def test_evicting_backfill_pairs_preserves_live_rows_and_other_pairs(conn):
    target = ("step_count", "2026-07-30")
    other = ("step_count", "2026-07-31")
    db.insert_records(conn, [
        _row("step_count", 100.0, target[1], origin="backfill",
             start="2026-07-30T08:00:00+00:00"),
        _row("step_count", 7.0, target[1], origin="receiver",
             start="2026-07-30T09:00:00+00:00"),
        _row("step_count", 200.0, other[1], origin="backfill",
             start="2026-07-31T08:00:00+00:00"),
    ])

    evicted = db.delete_records_for_pairs(conn, [target, target], origin="backfill")

    assert evicted == 1
    rows = conn.execute(
        "SELECT metric, local_date, value, origin FROM records ORDER BY local_date"
    ).fetchall()
    assert [(r["local_date"], r["value"], r["origin"]) for r in rows] == [
        (target[1], 7.0, "receiver"),
        (other[1], 200.0, "backfill"),
    ]

    db.recompute_daily_metrics(conn, pairs=[target, other])
    aggregate = conn.execute(
        "SELECT date, count, sum FROM daily_metrics ORDER BY date"
    ).fetchall()
    assert [(r["date"], r["count"], r["sum"]) for r in aggregate] == [
        (target[1], 1, 7.0),
        (other[1], 1, 200.0),
    ]
