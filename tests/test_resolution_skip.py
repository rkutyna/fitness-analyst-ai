"""A sample-resolution series is not re-measured on every ingest batch.

downstream #734: on a receiver vault, each step/distance batch measured the
metric's whole history under the write lock and then discarded the answer when
the series was already marked as sample resolution. The skip must leave
vault_meta byte-identical, and a bucketed series must still be measured.
"""
from __future__ import annotations

import json

from health_advisor import vault


def _meta(conn):
    return [tuple(r) for r in conn.execute(
        "SELECT key, value FROM vault_meta ORDER BY key")]


def _set_marks(conn, buckets, raw):
    conn.executemany(
        "INSERT INTO vault_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [(vault.VAULT_META_BUCKET_SECONDS, json.dumps(buckets)),
         (vault.VAULT_META_RAW_SERIES, json.dumps(raw))])
    conn.commit()


def _measured(monkeypatch):
    calls = []
    real = vault._recorded_resolution_seconds

    def spy(conn, metric):
        calls.append(metric)
        return real(conn, metric)
    monkeypatch.setattr(vault, "_recorded_resolution_seconds", spy)
    return calls


def test_a_raw_series_batch_skips_the_measurement(conn, monkeypatch):
    _set_marks(conn, {}, ["distance_walking_running", "step_count"])
    before = _meta(conn)
    calls = _measured(monkeypatch)
    assert vault.mark_receiver_resolutions(
        conn, mapped_metrics=["step_count", "distance_walking_running"]) == {}
    assert calls == []
    assert _meta(conn) == before


def test_a_bucketed_series_is_still_measured(conn, monkeypatch):
    _set_marks(conn, {"step_count": 20}, ["distance_walking_running"])
    calls = _measured(monkeypatch)
    vault.mark_receiver_resolutions(
        conn, mapped_metrics=["step_count", "distance_walking_running"])
    assert calls == ["step_count"]
