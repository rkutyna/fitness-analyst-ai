"""Engine #28: the prerequisites for switching vault compaction on.

Compaction deletes a non-allowlisted series' raw rows behind a watermark, which
makes that series' daily rows copies the engine cannot rebuild. Before these
guards, ordinary code paths wrote to them anyway: an incremental recompute
deleted them (count == 0), a D19 re-pull stripped them to `sum`, a full
recompute erased their provenance, a late sample failed its whole batch with a
500 on every retry, and a deletion of one left an undated tombstone. Each test
here states the pre-fix outcome it would have produced.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from health_advisor import analysis, db, receiver
from health_advisor import vault as V


WATERMARK = "2026-07-22"
DEVICE = {"id": "watch-a", "name": "iPhone", "model": "iPhone17,1"}
REVISION = {"source_name": "Apple Watch", "bundle_id": "com.apple.Health"}
HEART = "HKQuantityTypeIdentifierHeartRate"
BASAL = "HKQuantityTypeIdentifierBasalEnergyBurned"
FLIGHTS = "HKQuantityTypeIdentifierFlightsClimbed"
H = {"x-health-secret": "hk-secret"}


def _raw(metric, day, key, value=600.0, *, origin="receiver", source="test",
         hk_uuid=None, hour=12):
    ts = f"{day}T{hour:02d}:00:00+00:00"
    return {"metric": metric, "value": value, "unit": "kcal", "start_utc": ts,
            "end_utc": ts, "start_local": f"{day} {hour:02d}:00:00",
            "local_date": day, "source": source, "origin": origin,
            "dedupe_key": key, "hk_uuid": hk_uuid}


def _dm(conn, metric, day):
    row = conn.execute(
        "SELECT count, sum, avg, min, max, last, source_kind FROM daily_metrics "
        "WHERE metric = ? AND date = ?", (metric, day)).fetchone()
    return dict(row) if row else None


def _compacted_vault(conn):
    """basal_energy on a day behind the watermark and one after it, plus
    allowlisted heart_rate behind it; then compact."""
    V.declare_vault(conn)
    db.insert_records(conn, [
        _raw("basal_energy", "2026-07-10", "b1", 1.0, hk_uuid="u-b1", hour=8),
        _raw("basal_energy", "2026-07-10", "b2", 2.0, hk_uuid="u-b2", hour=9),
        _raw("basal_energy", "2026-07-30", "b3", 5.0, hk_uuid="u-b3"),
        _raw("heart_rate", "2026-07-10", "h1", 100.0),
    ])
    db.recompute_daily_metrics(conn, pairs=[
        ("basal_energy", "2026-07-10"), ("basal_energy", "2026-07-30"),
        ("heart_rate", "2026-07-10")])
    db.rebuild_metric_source_months(conn, full=True)
    conn.commit()
    before = _dm(conn, "basal_energy", "2026-07-10")
    assert V.compact(conn, through=WATERMARK) == {"basal_energy": 2}
    return before


# --------------------------------------------------------------------------- #
# Freeze (build step 1): findings 3, 4 and 5
# --------------------------------------------------------------------------- #
def test_frozen_through_is_none_off_a_compacted_vault(conn):
    assert V.frozen_through(conn) is None
    V.declare_vault(conn)
    assert V.frozen_through(conn) is None
    V.compact(conn, through=WATERMARK)
    assert V.frozen_through(conn) == WATERMARK
    assert V.is_frozen("basal_energy", WATERMARK, WATERMARK)
    assert not V.is_frozen("basal_energy", "2026-07-23", WATERMARK)
    assert not V.is_frozen("heart_rate", "2026-07-01", WATERMARK)


def test_incremental_recompute_keeps_a_frozen_row(conn):
    """Finding 5: pre-fix, count == 0 deleted this row (48% of the table on
    the ten-year clone)."""
    before = _compacted_vault(conn)
    assert before["count"] == 2 and before["sum"] == 3.0

    db.recompute_daily_metrics(conn, pairs=[("basal_energy", "2026-07-10"),
                                            ("basal_energy", "2026-07-30")])

    assert _dm(conn, "basal_energy", "2026-07-10") == before
    # The un-frozen day and the allowlisted series are still re-aggregated.
    assert _dm(conn, "basal_energy", "2026-07-30")["sum"] == 5.0
    logged = conn.execute(
        "SELECT rows_seen, detail FROM ingest_log WHERE kind = 'frozen' "
        "ORDER BY id DESC").fetchone()
    assert logged["rows_seen"] == 1
    assert "basal_energy@2026-07-10" in logged["detail"]
    assert f"compacted_through={WATERMARK}" in logged["detail"]


def test_unfreezing_the_pair_loses_the_row(conn, monkeypatch):
    """Mutation: the guard is what keeps the row -- without it, it is gone."""
    _compacted_vault(conn)
    monkeypatch.setattr(V, "is_frozen", lambda *a, **k: False)
    db.recompute_daily_metrics(conn, pairs=[("basal_energy", "2026-07-10")])
    assert _dm(conn, "basal_energy", "2026-07-10") is None


def test_checkin_row_behind_the_watermark_is_still_rebuilt(conn):
    """A check-in is not a copy of anything; its raw row is where it starts."""
    V.declare_vault(conn)
    V.compact(conn, through=WATERMARK)
    db.insert_records(conn, [_raw("mood", "2026-07-01", "c1", 4.0,
                                  origin="checkin", source="checkin")])
    db.recompute_daily_metrics(conn, pairs=[("mood", "2026-07-01")])
    assert _dm(conn, "mood", "2026-07-01")["last"] == 4.0


def test_full_recompute_keeps_frozen_provenance(conn):
    """Finding 4: pre-fix, the full rebuild dropped basal_energy's months and
    instrument_eras_status went ok -> unavailable/raw_series_not_in_vault."""
    _compacted_vault(conn)
    before = conn.execute(
        "SELECT * FROM metric_source_months ORDER BY 1, 2, 3").fetchall()
    eras_before = analysis.instrument_eras_status(
        conn, "basal_energy", "2026-07-01", "2026-07-31")

    db.recompute_daily_metrics(conn, full=True)
    conn.commit()

    after = conn.execute(
        "SELECT * FROM metric_source_months ORDER BY 1, 2, 3").fetchall()
    assert [tuple(r) for r in after] == [tuple(r) for r in before]
    assert analysis.instrument_eras_status(
        conn, "basal_energy", "2026-07-01", "2026-07-31") == eras_before
    assert eras_before["status"] == "ok"


def test_frozen_month_merges_rather_than_recounts(conn):
    """The watermark's own month keeps its count and still gains new sources;
    months after it, and allowlisted series, are recounted exactly."""
    _compacted_vault(conn)
    db.insert_records(conn, [
        _raw("basal_energy", "2026-07-31", "b4", 1.0, source="Phone"),
        _raw("basal_energy", "2026-08-02", "b5", 1.0),
    ])
    db.rebuild_metric_source_months(conn, pairs=[
        ("basal_energy", "2026-07-31"), ("basal_energy", "2026-08-02")])
    rows = {(r["month"], r["source"]): r["n"] for r in conn.execute(
        "SELECT month, source, n FROM metric_source_months "
        "WHERE metric = 'basal_energy'")}
    # 2026-07 held 3 test rows; 2 were compacted. A recount would say 1.
    assert rows == {("2026-07", "test"): 3, ("2026-07", "Phone"): 1,
                    ("2026-08", "test"): 1}


def test_repull_on_a_frozen_day_keeps_its_raw_aggregates(vault, monkeypatch):
    """Finding 3: pre-fix, a settled D19 re-pull deleted the pair and
    re-inserted count 0 with avg/min/max/last NULL; only sum survived."""
    monkeypatch.setattr(receiver, "SHARED_SECRET", "hk-secret")
    total = lambda v, state, q: {  # noqa: E731
        "type_identifier": FLIGHTS, "local_date": "2026-07-10", "value": v,
        "unit": "count", "interval": "day", "state": state, "queried_at": q}
    with TestClient(receiver.create_app(vault)) as client:
        assert client.post("/v1/ingest", headers=H, json=_payload(
            [_sample("f1", FLIGHTS, "2026-07-10", "08", 3.0, "count"),
             _sample("f2", FLIGHTS, "2026-07-10", "12", 5.0, "count")],
            totals=[total(8.0, "provisional", "2026-07-11T09:00:00-04:00")],
        )).status_code == 200
        conn = vault.connect()
        V.declare_vault(conn)
        V.compact(conn, through=WATERMARK)
        conn.close()
        assert client.post("/v1/ingest", headers=H, json=_payload(
            totals=[total(9.0, "settled", "2026-07-21T09:00:00-04:00")],
            batch="b2")).status_code == 200
    conn = vault.connect()
    assert _dm(conn, "flights_climbed", "2026-07-10") == {
        "count": 2, "sum": 9.0, "avg": 4.0, "min": 3.0, "max": 5.0,
        "last": 5.0, "source_kind": "apple_consolidated"}


# --------------------------------------------------------------------------- #
# Receiver (build steps 2 and 3): findings 1 and 2
# --------------------------------------------------------------------------- #
def _sample(uuid, tid, day, hh, value, unit):
    return {"kind": "quantity", "hk_uuid": uuid, "type_identifier": tid,
            "start": f"{day}T{hh}:00:00-04:00", "end": f"{day}T{hh}:00:30-04:00",
            "value": value, "unit": unit, "source_revision": REVISION}


def _payload(samples=(), deletions=(), totals=(), batch="b1"):
    return {"protocol_version": 1, "device": DEVICE, "app_version": "1.0",
            "batch_id": batch, "batch_sequence": 1,
            "sent_at": "2026-08-22T13:04:05Z",
            "anchors": [{"type_identifier": HEART, "from": None,
                         "to": f"anchor-{batch}"}],
            "samples": list(samples), "deletions": list(deletions),
            "workouts": [], "daily_totals": list(totals)}


def _ingest_vault(vault, monkeypatch, samples):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "hk-secret")
    client = TestClient(receiver.create_app(vault))
    assert client.post("/v1/ingest", headers=H,
                       json=_payload(samples)).status_code == 200
    conn = vault.connect()
    V.declare_vault(conn)
    V.compact(conn, through=WATERMARK)
    conn.close()
    return client


def test_late_sample_is_a_409_and_the_rest_of_the_batch_lands(vault, monkeypatch):
    """Finding 1: pre-fix, 500 on the batch and on every retry, and the
    heart-rate sample beside the late one never landed."""
    client = _ingest_vault(vault, monkeypatch, [
        _sample("b-old", BASAL, "2026-07-10", "08", 0.5, "kcal")])
    late = _payload([_sample("b-late", BASAL, "2026-07-12", "09", 0.7, "kcal"),
                     _sample("hr-now", HEART, "2026-08-21", "10", 120.0,
                             "count/min")], batch="b2")

    first = client.post("/v1/ingest", headers=H, json=late)
    retry = client.post("/v1/ingest", headers=H, json=late)

    assert first.status_code == 409
    body = first.json()
    assert body["applied"] is True
    assert body["detail"].startswith(f"compacted through {WATERMARK};")
    assert body["refusal"] == {
        "reason": "behind_compaction_watermark", "compacted_through": WATERMARK,
        "samples_refused": 1, "deletions_refused": 0,
        "metrics": ["basal_energy"], "date_min": "2026-07-12",
        "date_max": "2026-07-12", "retryable": False}
    assert retry.status_code == 200 and retry.json()["reason"] == "already_applied"
    conn = vault.connect()
    assert _dm(conn, "heart_rate", "2026-08-21")["count"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE hk_uuid = 'b-late'").fetchone()[0] == 0
    diag = conn.execute(
        "SELECT metric, local_date, reason FROM ingest_diagnostics").fetchall()
    assert [tuple(r) for r in diag] == [
        ("basal_energy", "2026-07-12", "behind_compaction_watermark")]
    assert V.compaction_status(conn)["status"] == "compacted_clean"


def test_late_allowlisted_sample_is_not_refused(vault, monkeypatch):
    client = _ingest_vault(vault, monkeypatch, [])
    r = client.post("/v1/ingest", headers=H, json=_payload(
        [_sample("hr-old", HEART, "2026-07-10", "10", 99.0, "count/min")],
        batch="b2"))
    assert r.status_code == 200


def test_deletion_of_a_compacted_sample_is_dated_and_refused(vault, monkeypatch):
    """Finding 2: pre-fix, 200 with the sum silently unchanged and a tombstone
    whose sample_local_date and sample_metric were NULL."""
    client = _ingest_vault(vault, monkeypatch, [
        _sample("b1", BASAL, "2026-07-10", "08", 1.0, "kcal"),
        _sample("b2", BASAL, "2026-07-10", "09", 2.0, "kcal")])
    conn = vault.connect()
    before = _dm(conn, "basal_energy", "2026-07-10")
    assert V.compacted_sample(conn, "b1") == {
        "local_date": "2026-07-10", "metric": "basal_energy"}
    conn.close()

    r = client.post("/v1/ingest", headers=H, json=_payload(
        deletions=[{"hk_uuid": "b1", "type_identifier": BASAL},
                   {"hk_uuid": "never-held", "type_identifier": BASAL}],
        batch="b2"))

    assert r.status_code == 409
    assert r.json()["refusal"]["deletions_refused"] == 1
    conn = vault.connect()
    tombs = {row["hk_uuid"]: (row["sample_local_date"], row["sample_metric"])
             for row in conn.execute("SELECT * FROM hk_deletions")}
    assert tombs == {"b1": ("2026-07-10", "basal_energy"),
                     "never-held": (None, None)}
    # The daily row is frozen: the deletion is refused for the aggregate.
    assert _dm(conn, "basal_energy", "2026-07-10") == before


def test_deletion_of_an_unknown_uuid_is_still_a_200(vault, monkeypatch):
    client = _ingest_vault(vault, monkeypatch, [])
    r = client.post("/v1/ingest", headers=H, json=_payload(
        deletions=[{"hk_uuid": "nobody", "type_identifier": BASAL}], batch="b2"))
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# Driver (build step 5): D1 window, D4 VACUUM
# --------------------------------------------------------------------------- #
def _driver_vault(path):
    conn = db.connect(path)
    db.init_db(conn)
    V.declare_vault(conn)
    rows = []
    for d in range(1, 29):
        day = f"2026-07-{d:02d}"
        rows += [_raw("basal_energy", day, f"b{d}-{i}", 1.0, hk_uuid=f"u{d}-{i}",
                      hour=i % 24) | {"start_utc": f"{day}T00:{i // 60:02d}:"
                                                   f"{i % 60:02d}+00:00"}
                 for i in range(400)]
    rows.append(_raw("basal_energy", "2026-08-21", "last", 1.0))
    db.insert_records(conn, rows)
    conn.commit()
    conn.close()


def test_run_compaction_uses_a_30_day_window_from_the_last_sync(tmp_path):
    path = tmp_path / "v.db"
    _driver_vault(path)
    report = V.run_compaction(path, today="2026-08-25")
    assert report["last_sync"] == "2026-08-21"
    assert report["through"] == "2026-07-22"
    assert report["deleted_total"] == 22 * 400
    assert report["vacuumed"] is True
    assert report["bytes_after"] < report["bytes_before"]
    conn = db.connect(path, read_only=True)
    assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0
    assert V.compacted_sample(conn, "u5-7") == {
        "local_date": "2026-07-05", "metric": "basal_energy"}
    assert conn.execute(
        "SELECT COUNT(*) FROM ingest_log WHERE source = 'vault' "
        "AND kind = 'vacuum'").fetchone()[0] == 1


def test_run_compaction_without_vacuum_frees_no_bytes(tmp_path):
    path = tmp_path / "v.db"
    _driver_vault(path)
    report = V.run_compaction(path, today="2026-08-25", vacuum=False)
    assert report["deleted_total"] > 0
    # The pages go to the freelist: the file does not shrink (it grows by the
    # compacted_samples rows), which is why D4 makes VACUUM part of the job.
    assert report["bytes_after"] >= report["bytes_before"]
    conn = db.connect(path, read_only=True)
    assert conn.execute("PRAGMA freelist_count").fetchone()[0] > 0


def test_run_compaction_refuses_a_window_under_the_repull_floor(tmp_path):
    path = tmp_path / "v.db"
    _driver_vault(path)
    with pytest.raises(ValueError, match="floor"):
        V.run_compaction(path, window_days=7)
    with pytest.raises(ValueError, match="floor"):
        V.run_compaction(path, through="2026-08-14")
    with pytest.raises(ValueError, match="floor"):
        V.run_compaction(path, window_days=20, repull_window_days=20)


def test_compaction_window_clears_the_receiver_repull_window():
    assert V.COMPACTION_WINDOW_DAYS == 30
    assert V.COMPACTION_MIN_WINDOW_DAYS > receiver.DAILY_TOTAL_REPULL_WINDOW_DAYS


def test_run_compaction_dry_run_writes_nothing(tmp_path):
    path = tmp_path / "v.db"
    _driver_vault(path)
    before = path.read_bytes()
    report = V.run_compaction(path, today="2026-08-25", dry_run=True)
    assert report["would_delete"] == 22 * 400
    assert path.read_bytes() == before


def test_compact_migrates_a_vault_that_predates_the_table(conn):
    V.declare_vault(conn)
    conn.execute("DROP TABLE compacted_samples")
    db.insert_records(conn, [_raw("basal_energy", "2026-07-01", "x", hk_uuid="ux")])
    V.compact(conn, through=WATERMARK)
    assert V.compacted_sample(conn, "ux")["local_date"] == "2026-07-01"


def test_compacted_sample_key_is_stable_and_signed_64_bit():
    key = V.compacted_sample_key("0F2A5E4C-1B2D-4C3E-9F00-1234567890AB")
    assert key == V.compacted_sample_key("0F2A5E4C-1B2D-4C3E-9F00-1234567890AB")
    assert -(2 ** 63) <= key < 2 ** 63
    conn = sqlite3.connect(":memory:")
    assert conn.execute("SELECT ?", (key,)).fetchone()[0] == key
