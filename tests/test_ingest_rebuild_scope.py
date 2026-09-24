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
