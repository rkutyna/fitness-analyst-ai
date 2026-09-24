"""engine#78: a records-free ingest batch must not pay for
`db.rebuild_metric_source_months`.

`rebuild_metric_source_months` (health_advisor/db.py) derives
`metric_source_months` solely from the `records` table — it counts raw
sample rows per (metric, month, source). A daily-totals-only `/v1/ingest`
batch never writes to `records`, so recomputing that count for the pairs it
touches would just recount a slice of `records` that did not change: safe to
skip, not merely cheap to skip. A batch that DOES carry samples must still
trigger the rebuild for the months those samples land in.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from health_advisor import db as dbmod
from health_advisor import receiver

DEVICE = {"id": "dev-1", "name": "iPhone", "model": "iPhone17,1"}
HEART = "HKQuantityTypeIdentifierHeartRate"


def _sample(uuid, start, end, value, unit="count/min", type_identifier=HEART):
    return {
        "kind": "quantity", "hk_uuid": uuid, "type_identifier": type_identifier,
        "start": start, "end": end, "value": value, "unit": unit,
        "source_revision": {"source_name": "Apple Watch",
                             "bundle_id": "com.apple.Health"},
    }


def _total(metric_type="HKQuantityTypeIdentifierStepCount",
           local_date="2026-08-25", value=10173.0):
    return {
        "type_identifier": metric_type, "local_date": local_date,
        "value": value, "unit": "count", "interval": "day",
        "state": "provisional", "queried_at": "2026-08-26T09:00:00-04:00",
    }


def _payload(*, samples=(), daily_totals=(), batch_id="batch-1", sequence=1):
    return {
        "protocol_version": 1, "device": DEVICE, "app_version": "1.0",
        "batch_id": batch_id, "batch_sequence": sequence,
        "sent_at": "2026-08-26T13:04:05Z",
        "anchors": [], "samples": list(samples), "deletions": [],
        "workouts": [], "daily_totals": list(daily_totals),
    }


def _client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "hk-secret")
    return TestClient(receiver.create_app(vault))


def test_records_free_batch_does_not_rebuild_metric_source_months(
    vault, monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        dbmod, "rebuild_metric_source_months",
        lambda *a, **k: calls.append((a, k)) or 0,
    )

    payload = _payload(daily_totals=[_total()], batch_id="daily-only")
    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest", json=payload,
            headers={"x-health-secret": "hk-secret"},
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert calls == [], (
        "a daily-totals-only batch wrote no records, so "
        "rebuild_metric_source_months should not have run at all"
    )


def test_batch_with_records_still_rebuilds_metric_source_months(
    vault, monkeypatch,
):
    calls = []
    real_rebuild = dbmod.rebuild_metric_source_months

    def _spy(conn, *a, **k):
        calls.append((a, k))
        return real_rebuild(conn, *a, **k)

    monkeypatch.setattr(dbmod, "rebuild_metric_source_months", _spy)

    sample = _sample("hr-1", "2026-08-25T09:00:00-04:00",
                      "2026-08-25T09:00:01-04:00", 145.0)
    payload = _payload(samples=[sample], daily_totals=[_total()],
                        batch_id="records-and-totals")
    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest", json=payload,
            headers={"x-health-secret": "hk-secret"},
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert len(calls) == 1
    (args, kwargs) = calls[0]
    pairs = kwargs.get("pairs") if "pairs" in kwargs else args[0]
    # Scoped to what the sample actually touched (heart_rate, the day it
    # landed on) — not to step_count, which only appeared in daily_totals
    # and never wrote a `records` row.
    assert set(pairs) == {("heart_rate", "2026-08-25")}


SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"


def _sleep(uuid, start, end, value):
    return {
        "kind": "category", "hk_uuid": uuid, "type_identifier": SLEEP,
        "start": start, "end": end, "value": value, "unit": None,
        "source_revision": {"source_name": "Apple Watch",
                             "bundle_id": "com.apple.Health"},
    }


def _source_months(vault):
    conn = dbmod.connect(vault.db_path)
    try:
        return sorted(tuple(row) for row in conn.execute(
            "SELECT metric, month, source, n FROM metric_source_months "
            "WHERE metric LIKE 'sleep_%'"))
    finally:
        conn.close()


def test_sleep_reattribution_across_a_month_boundary_rebuilds_both_months(
    vault, monkeypatch,
):
    """`derive.reattribute_sleep(apply=True)` rewrites `records.local_date`.

    An in-bed interval ingested alone on Aug 31 is its own session and stays
    there; the asleep interval that arrives in the next batch joins it into
    one session ending Sep 1, so the in-bed record MOVES to Sep 1. That move
    is this batch's only `records` effect on sleep_in_bed, and it changes the
    per-month count in BOTH months -- August loses a row, September gains
    one -- so both must be rebuilt, or `metric_source_months` keeps an
    in-bed row in August that `records` no longer holds.
    """
    calls = []
    real_rebuild = dbmod.rebuild_metric_source_months

    def _spy(conn, *a, **k):
        calls.append(k.get("pairs") if "pairs" in k else a[0])
        return real_rebuild(conn, *a, **k)

    monkeypatch.setattr(dbmod, "rebuild_metric_source_months", _spy)

    in_bed = _sleep("bed-1", "2026-08-31T23:30:00-04:00",
                    "2026-08-31T23:55:00-04:00",
                    "HKCategoryValueSleepAnalysisInBed")
    asleep = _sleep("core-1", "2026-09-01T00:05:00-04:00",
                    "2026-09-01T06:00:00-04:00",
                    "HKCategoryValueSleepAnalysisAsleepCore")
    with _client(vault, monkeypatch) as client:
        for n, sample in enumerate((in_bed, asleep), start=1):
            response = client.post(
                "/v1/ingest",
                json=_payload(samples=[sample], batch_id=f"sleep-{n}",
                              sequence=n),
                headers={"x-health-secret": "hk-secret"},
            )
            assert response.status_code == 200
            assert response.json()["ok"] is True

    conn = dbmod.connect(vault.db_path)
    try:
        moved = conn.execute(
            "SELECT local_date FROM records WHERE metric = 'sleep_in_bed'"
        ).fetchall()
    finally:
        conn.close()
    assert [row[0] for row in moved] == ["2026-09-01"], (
        "precondition: the in-bed record must have been reattributed")

    second_batch_months = {(metric, day[:7]) for metric, day in calls[-1]}
    assert ("sleep_in_bed", "2026-08") in second_batch_months
    assert ("sleep_in_bed", "2026-09") in second_batch_months

    # The invariant the scoped rebuild must keep: identical to a full recount.
    scoped = _source_months(vault)
    conn = dbmod.connect(vault.db_path)
    try:
        real_rebuild(conn, full=True)
        conn.commit()
    finally:
        conn.close()
    assert scoped == _source_months(vault)
