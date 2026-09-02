"""A real receiver ingest must refresh the additive daily last value."""
from __future__ import annotations

import json

from starlette.requests import Request

from health_advisor import db, receiver


def _request():
    return Request({
        "type": "http", "method": "POST", "path": "/v1/ingest",
        "query_string": b"", "headers": [], "client": ("test", 1),
        "server": ("test", 80), "scheme": "http", "http_version": "1.1",
    })


def test_receiver_incremental_ingest_populates_last(vault, vault_path, monkeypatch):
    """`body_mass` is load-bearing here and must not be swapped for an
    allowlisted series.

    It is deliberately NOT in `vault.VAULT_RAW_SERIES`, which makes this the
    test that proves D3 restricts which series are kept as raw samples and not
    which get a daily aggregate. An attempt to filter the receiver's write set
    to the allowlist (2026-08-22) dropped these rows *and* their (metric, date)
    pairs, so ~93 series stopped getting `daily_metrics` updates entirely and
    silently — weight among them. The change was found only because re-pointing
    this test from `body_mass` to `step_count` was what kept the suite green.
    """
    path = vault_path
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    payload = {
        "protocol_version": 1,
        "device": {"id": "scale", "name": "Scale", "model": "test"},
        "app_version": "1.0", "batch_id": "body-mass-1", "batch_sequence": 1,
        "sent_at": "2026-07-30T14:00:00Z",
        "anchors": [{"type_identifier": "HKQuantityTypeIdentifierBodyMass",
                      "from": None, "to": "anchor-1"}],
        "samples": [
            {"kind": "quantity", "hk_uuid": "mass-1",
             "type_identifier": "HKQuantityTypeIdentifierBodyMass",
             "start": "2026-07-30T08:00:00-04:00",
             "end": "2026-07-30T08:00:01-04:00", "value": 190.0,
             "unit": "lb", "source_revision": {"source_name": "Scale",
                                                   "bundle_id": "test"}},
            {"kind": "quantity", "hk_uuid": "mass-2",
             "type_identifier": "HKQuantityTypeIdentifierBodyMass",
             "start": "2026-07-30T09:00:00-04:00",
             "end": "2026-07-30T09:00:01-04:00", "value": 189.0,
             "unit": "lb", "source_revision": {"source_name": "Scale",
                                                   "bundle_id": "test"}},
        ], "deletions": [], "workouts": [],
    }

    response = receiver._healthkit_ingest(vault, _request(),
                                          json.dumps(payload).encode(), None)
    assert response.status_code == 200

    conn = db.connect(path, read_only=True)
    try:
        row = conn.execute(
            "SELECT avg, last FROM daily_metrics "
            "WHERE metric = 'body_mass' AND date = '2026-07-30'"
        ).fetchone()
    finally:
        conn.close()
    assert row["avg"] == 189.5
    assert row["last"] == 189.0
