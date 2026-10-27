"""#696 E3: Apple's daily STATISTICS for history days, written fill-only.

A history_backfill batch may carry `daily_stats` for the non-sum metrics
(HRV = day average, resting heart rate = most recent). The stored statistic
reaches daily_metrics only on a day with no raw samples; raw data always wins.

Oracles are hand-written literals (48.5, 61, 30, 99, ...), never the code's own
output. Records are seeded straight into `records` so the raw side is exact.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from health_advisor import db as dbmod
from health_advisor import receiver

_DEVICE = {"id": "dev-1", "name": "iPhone", "model": "iPhone17,1"}
_HRV = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
_RHR = "HKQuantityTypeIdentifierRestingHeartRate"
_STEPS = "HKQuantityTypeIdentifierStepCount"


def _stat(**overrides):
    row = {
        "type_identifier": _HRV,
        "local_date": "2025-01-10",
        "value": 48.5,
        "unit": "ms",
        "statistic": "average",
        "queried_at": "2026-10-26T09:00:00-04:00",
    }
    row.update(overrides)
    return row


def _payload(*stats, batch_id="backfill", backfill=True, extra=None):
    payload = {
        "protocol_version": 1,
        "device": _DEVICE,
        "app_version": "1.0",
        "batch_id": batch_id,
        "batch_sequence": 1,
        "sent_at": "2026-10-26T09:00:00Z",
        "anchors": [], "samples": [], "deletions": [], "workouts": [],
        "daily_stats": list(stats),
    }
    if backfill:
        payload["history_backfill"] = True
    payload.update(extra or {})
    return payload


def _client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "hk-secret")
    return TestClient(receiver.create_app(vault))


def _post(client, payload):
    return client.post("/v1/ingest", json=payload,
                       headers={"x-health-secret": "hk-secret"})


def _seed_raw(conn, metric, unit, local_date, values):
    """Raw samples straight into `records`, then the ordinary recompute."""
    for i, value in enumerate(values):
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, dedupe_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'seed', ?)",
            (metric, value, unit, f"{local_date}T12:0{i}:00Z",
             f"{local_date}T12:0{i}:30Z", f"{local_date}T08:0{i}:00",
             local_date, f"seed-{metric}-{local_date}-{i}"))
    dbmod.recompute_daily_metrics(conn, pairs=[(metric, local_date)])
    conn.commit()


def _dm(conn, metric, day):
    return conn.execute(
        "SELECT count, sum, avg, min, max, last, unit, source_kind, "
        "derived_version FROM daily_metrics WHERE metric = ? AND date = ?",
        (metric, day)).fetchone()


def test_backfill_stats_fill_days_that_have_no_raw_samples(
        conn, vault, monkeypatch):
    """HRV 2025-01-10 = 48.5 average, resting HR 2025-01-10 = 61 most recent.
    Neither day has a raw sample, so both land in daily_metrics as statistics."""
    with _client(vault, monkeypatch) as client:
        response = _post(client, _payload(
            _stat(),
            _stat(type_identifier=_RHR, value=61.0, unit="count/min",
                  statistic="most_recent"),
            batch_id="stats-1"))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["daily_stats_backfill_skipped"] == {
        "backfill_history_vault": 0,
        "backfill_inside_window": 0,
        "backfill_existing_skip": 0,
    }
    # Values never echo into the acknowledgement.
    assert "48.5" not in str(body) and "61.0" not in str(body)
    stored = {(r["metric"], r["statistic"], r["value"]) for r in conn.execute(
        "SELECT metric, statistic, value FROM hk_daily_stats")}
    assert stored == {("heart_rate_variability", "average", 48.5),
                      ("resting_heart_rate", "most_recent", 61.0)}

    hrv = _dm(conn, "heart_rate_variability", "2025-01-10")
    assert (hrv["count"], hrv["avg"], hrv["unit"]) == (0, 48.5, "ms")
    assert hrv["source_kind"] == "apple_statistics"
    assert hrv["derived_version"] == 3
    rhr = _dm(conn, "resting_heart_rate", "2025-01-10")
    assert (rhr["count"], rhr["last"], rhr["unit"]) == (0, 61.0, "count/min")
    assert rhr["source_kind"] == "apple_statistics"
    assert rhr["derived_version"] == 3


def test_raw_samples_win_over_a_backfilled_statistic(conn, vault, monkeypatch):
    """Raw HRV 2025-02-01 = 28 and 32 (avg 30, count 2). A backfilled statistic
    of 99 for the same day is stored but never reaches daily_metrics."""
    _seed_raw(conn, "heart_rate_variability", "ms", "2025-02-01", [28.0, 32.0])
    before = tuple(_dm(conn, "heart_rate_variability", "2025-02-01"))
    assert before[:3] == (2, 60.0, 30.0)

    with _client(vault, monkeypatch) as client:
        response = _post(client, _payload(
            _stat(local_date="2025-02-01", value=99.0), batch_id="stats-raw"))

    assert response.status_code == 200, response.text
    assert conn.execute(
        "SELECT value FROM hk_daily_stats WHERE local_date = '2025-02-01'"
    ).fetchone()[0] == 99.0
    after = _dm(conn, "heart_rate_variability", "2025-02-01")
    assert tuple(after) == before
    assert after["count"] == 2 and after["avg"] == 30.0
    assert after["source_kind"] == "records"


def test_a_later_raw_sample_on_a_stats_only_day_takes_the_day_back(
        conn, vault, monkeypatch):
    """After a statistic fills 2025-01-10, one raw HRV sample of 40 for that day
    goes through the normal recompute: the row becomes records-derived."""
    with _client(vault, monkeypatch) as client:
        assert _post(client, _payload(_stat(), batch_id="stats-first")
                     ).status_code == 200
    assert _dm(conn, "heart_rate_variability", "2025-01-10")["source_kind"] \
        == "apple_statistics"

    _seed_raw(conn, "heart_rate_variability", "ms", "2025-01-10", [40.0])

    row = _dm(conn, "heart_rate_variability", "2025-01-10")
    assert (row["count"], row["avg"]) == (1, 40.0)
    assert row["source_kind"] == "records"


def test_a_statistic_for_an_existing_stat_day_is_counted_and_skipped(
        conn, vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        assert _post(client, _payload(_stat(), batch_id="stats-a")
                     ).status_code == 200
        response = _post(client, _payload(_stat(value=77.0), batch_id="stats-b"))

    assert response.status_code == 200, response.text
    assert response.json()["daily_stats_backfill_skipped"][
        "backfill_existing_skip"] == 1
    assert conn.execute("SELECT value FROM hk_daily_stats").fetchone()[0] == 48.5


def test_a_statistic_inside_the_repull_window_is_counted_and_not_written(
        conn, vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = _post(client, _payload(
            _stat(local_date="2026-10-20"), batch_id="stats-window"))

    assert response.status_code == 200, response.text
    assert response.json()["daily_stats_backfill_skipped"][
        "backfill_inside_window"] == 1
    assert conn.execute("SELECT COUNT(*) FROM hk_daily_stats").fetchone()[0] == 0


def test_daily_stats_on_a_non_backfill_batch_is_a_400(conn, vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = _post(client, _payload(_stat(), batch_id="stats-live",
                                          backfill=False))

    assert response.status_code == 400
    assert "daily_stats is accepted only in a history_backfill batch" \
        in response.json()["detail"]
    assert conn.execute("SELECT COUNT(*) FROM hk_daily_stats").fetchone()[0] == 0


def test_a_wrong_statistic_for_a_metric_is_a_400(conn, vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = _post(client, _payload(
            _stat(statistic="most_recent"), batch_id="stats-wrong"))

    assert response.status_code == 400
    assert "is not the statistic for heart_rate_variability" \
        in response.json()["detail"]
    assert conn.execute("SELECT COUNT(*) FROM hk_daily_stats").fetchone()[0] == 0


def test_a_sum_metric_in_daily_stats_is_a_400(conn, vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = _post(client, _payload(
            _stat(type_identifier=_STEPS, value=10173.0, unit="count",
                  statistic="average"),
            batch_id="stats-steps"))

    assert response.status_code == 400
    assert "is not a daily statistic metric" in response.json()["detail"]
    assert conn.execute("SELECT COUNT(*) FROM hk_daily_stats").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()[0] == 0


def test_the_stats_table_refuses_update_and_delete(conn):
    conn.execute(
        "INSERT INTO hk_daily_stats (metric, local_date, value, unit, statistic, "
        "device_id, queried_at, batch_id, first_seen_at) VALUES "
        "('heart_rate_variability', '2025-01-10', 48.5, 'ms', 'average', "
        "'dev-1', '2026-10-26T09:00:00-04:00', 'b1', '2026-10-26T13:00:00Z')")
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError) as updated:
        conn.execute("UPDATE hk_daily_stats SET value = 99.0")
    assert "a daily statistic is immutable" in str(updated.value)

    with pytest.raises(sqlite3.IntegrityError) as deleted:
        conn.execute("DELETE FROM hk_daily_stats")
    assert "daily statistics are not deletable" in str(deleted.value)

    assert conn.execute("SELECT value FROM hk_daily_stats").fetchone()[0] == 48.5


def test_init_db_gives_an_existing_vault_the_stats_table(conn):
    conn.execute("DROP TABLE hk_daily_stats")
    conn.commit()
    dbmod.init_db(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
        "AND name = 'hk_daily_stats'").fetchone()[0] == 1
