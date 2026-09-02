from __future__ import annotations

import subprocess
import sys

import pytest

from health_advisor import db, vault
from health_advisor.vault import VAULT_RAW_SERIES, build_vault, format_report


def _record(metric: str, value: float, day: str, n: int) -> dict:
    start = f"{day}T00:00:{n:02d}+00:00"
    return {
        "metric": metric,
        "value": value,
        "unit": "count",
        "start_utc": start,
        "end_utc": start,
        "start_local": start[:-6],
        "local_date": day,
        "source": "test",
        "origin": "backfill",
        "dedupe_key": f"{metric}-{day}-{n}",
    }


def _source_db(path) -> None:
    conn = db.connect(path)
    db.init_db(conn)
    db.insert_records(conn, [
        _record("heart_rate", 140, "2026-08-20", 1),
        _record("distance_walking_running", 0.01, "2026-08-20", 2),
        _record("active_energy", 500, "2026-08-20", 3),
        _record("sleep_asleep", 420, "2026-08-20", 4),
    ])
    conn.execute(
        "INSERT INTO daily_metrics "
        "(metric,date,count,sum,avg,min,max,last,unit) VALUES "
        "('wear_hours','2026-08-20',1,16,16,16,16,16,'h')"
    )
    conn.commit()
    conn.close()


def _consolidated_source(path) -> None:
    conn = db.connect(path)
    db.init_db(conn)
    db.insert_records(conn, [_record("step_count", 10, "2026-08-20", 1)])
    db.recompute_daily_metrics(conn, full=True)
    conn.commit()
    conn.close()


def _ingest_consolidated_total(path) -> None:
    conn = db.connect(path)
    db.insert_daily_totals(conn, [{
        "metric": "step_count",
        "local_date": "2026-08-20",
        "value": 6.0,
        "unit": "count",
        "interval": "day",
        "state": "provisional",
        "device_id": "test-device",
        "queried_at": "2026-08-21T09:00:00",
    }], batch_id="daily-test")
    db.apply_consolidated_totals(conn)
    conn.commit()
    conn.close()


def _verify(path):
    return subprocess.run(
        [sys.executable, "scripts/verify_daily_metrics.py", "--db", str(path),
         "--derived-days", "0"],
        capture_output=True, text=True,
    )


def test_allowlist_contains_every_d3_raw_dependency_and_is_single_source():
    """Exact equality, not containment.

    `<=` let a series be added to the allowlist without anything noticing, which
    matters because the allowlist is repeated in ARCHITECTURE.md D3 and in the
    invariants reference. A change that turns this red is a change that has to
    visit those too."""
    assert VAULT_RAW_SERIES == {
        "heart_rate",
        "distance_walking_running",
        "running_power",
        "sleep_asleep",
        "sleep_awake",
        "sleep_in_bed",
        "step_count",
    }


def test_build_filters_records_but_keeps_derived_tables_and_reports_drops(tmp_path):
    source = tmp_path / "source.db"
    vault = tmp_path / "vault.db"
    _source_db(source)

    report = build_vault(source, vault, batch_size=2)

    assert report["records_seen"] == 4
    assert report["records_copied"] == 3
    assert report["records_dropped"] == 1
    assert report["dropped_by_metric"] == {"active_energy": 1}
    assert report["copied_by_metric"] == {
        "distance_walking_running": 1,
        "heart_rate": 1,
        "sleep_asleep": 1,
    }
    assert report["history_imported_through"] == "2026-08-20"
    text = format_report(report)
    assert "dropped raw series (visible omissions):" in text
    assert "active_energy: 1" in text

    conn = db.connect(vault, read_only=True)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 3
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric = 'active_energy'"
    ).fetchone()[0] == 0
    # Derived data is not rebuilt from the filtered raw table and remains present.
    assert conn.execute(
        "SELECT last FROM daily_metrics WHERE metric = 'wear_hours'"
    ).fetchone()[0] == 16
    assert conn.execute(
        "SELECT value FROM vault_meta WHERE key = 'history_imported_through'"
    ).fetchone()[0] == "2026-08-20"
    conn.close()


def test_build_refuses_to_overwrite_without_explicit_replace(tmp_path):
    source = tmp_path / "source.db"
    vault = tmp_path / "vault.db"
    _source_db(source)
    vault.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        build_vault(source, vault)
    assert vault.read_bytes() == b"sentinel"


def test_build_uses_latest_derived_date_when_source_has_no_raw_rows(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "vault.db"
    conn = db.connect(source)
    db.init_db(conn)
    conn.execute(
        "INSERT INTO daily_metrics "
        "(metric,date,count,sum,avg,min,max,last,unit) "
        "VALUES ('heart_rate','2025-01-02',1,120,120,120,120,120,'count/min')"
    )
    conn.commit()
    conn.close()

    report = build_vault(source, target, measure_gzip=False)

    assert report["history_imported_through"] == "2025-01-02"
    conn = db.connect(target, read_only=True)
    try:
        assert conn.execute(
            "SELECT value FROM vault_meta "
            "WHERE key = 'history_imported_through'"
        ).fetchone()[0] == "2025-01-02"
    finally:
        conn.close()


def test_a_bucketed_series_is_stored_coarser_and_says_so(tmp_path):
    """`distance_walking_running` arrives at one sample per second and every
    consumer reduces it to 20 s, so the vault stores the 20 s bucket."""
    from health_advisor.vault import VAULT_BUCKET_SECONDS, raw_resolution_seconds

    source = tmp_path / "source.db"
    vault = tmp_path / "vault.db"
    conn = db.connect(source)
    db.init_db(conn)
    # Six seconds of walking inside one 20 s window, plus one in the next.
    db.insert_records(conn, [
        _record("distance_walking_running", 0.001, "2026-08-20", n)
        for n in range(1, 7)
    ] + [_record("distance_walking_running", 0.002, "2026-08-20", 25)])
    conn.commit()
    conn.close()

    report = build_vault(source, vault, measure_gzip=False)

    assert raw_resolution_seconds("distance_walking_running") == 20
    assert raw_resolution_seconds("heart_rate") == 0
    info = report["bucketed_by_metric"]["distance_walking_running"]
    assert (info["raw"], info["buckets"], info["seconds"]) == (7, 2, 20)
    # It is stored coarser, not omitted — calling it "dropped" is what somebody
    # would later believe.
    assert "distance_walking_running" not in report["dropped_by_metric"]
    assert "bucketed raw series" in format_report(report)

    conn = db.connect(vault, read_only=True)
    try:
        rows = conn.execute(
            "SELECT value, start_utc FROM records "
            "WHERE metric='distance_walking_running' ORDER BY start_utc").fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert rows[0]["value"] == pytest.approx(0.006), "the bucket sums its samples"
    assert rows[1]["value"] == pytest.approx(0.002)
    assert rows[0]["start_utc"].endswith("00:00:01+00:00"), \
        "the bucket keeps its earliest real sample, not a synthetic boundary"


def test_bucketing_is_per_source_so_arbitration_survives(tmp_path):
    """Two devices' samples must not collapse into one row.

    `db._arbitration` resolves cross-source overlap by dropping a mirror source.
    A bucket that merged them would make that impossible forever — F3-1 baked
    in rather than merely present.
    """
    source = tmp_path / "source.db"
    vault = tmp_path / "vault.db"
    conn = db.connect(source)
    db.init_db(conn)
    rows = []
    for n in (1, 2, 3):
        for who in ("Demo's Apple Watch", "Sync Solver"):
            row = _record("distance_walking_running", 0.001, "2026-08-20", n)
            row["source"] = who
            row["dedupe_key"] = f"d-{who}-{n}"
            rows.append(row)
    db.insert_records(conn, rows)
    conn.commit()
    conn.close()

    build_vault(source, vault, measure_gzip=False)

    conn = db.connect(vault, read_only=True)
    try:
        got = {r["source"]: r["value"] for r in conn.execute(
            "SELECT source, value FROM records "
            "WHERE metric='distance_walking_running'")}
    finally:
        conn.close()

    assert set(got) == {"Demo's Apple Watch", "Sync Solver"}, \
        "the two sources were merged; arbitration can no longer tell them apart"
    assert got["Demo's Apple Watch"] == pytest.approx(0.003)
    assert got["Sync Solver"] == pytest.approx(0.003)


def test_18_rebuild_preserves_consolidated_totals(tmp_path):
    """18. Live totals survive and are reapplied after a source rebuild."""
    source = tmp_path / "source.db"
    vault_path = tmp_path / "vault.db"
    _consolidated_source(source)
    build_vault(source, vault_path, measure_gzip=False)
    _ingest_consolidated_total(vault_path)

    build_vault(source, vault_path, replace=True, measure_gzip=False)

    conn = db.connect(vault_path, read_only=True)
    try:
        total = conn.execute(
            "SELECT value FROM hk_daily_totals "
            "WHERE metric = 'step_count' AND local_date = '2026-08-20'"
        ).fetchone()
        metric = conn.execute(
            "SELECT sum, source_kind FROM daily_metrics "
            "WHERE metric = 'step_count' AND date = '2026-08-20'"
        ).fetchone()
    finally:
        conn.close()
    assert total["value"] == 6.0
    assert (metric["sum"], metric["source_kind"]) == (6.0, "apple_consolidated")
    result = _verify(vault_path)
    assert result.returncode == 0, result.stdout


def test_18a_rebuild_that_would_drop_totals_refuses_and_old_vault_survives(
        tmp_path, monkeypatch):
    """18a. A lossy rebuild is refused before it can replace the vault."""
    source = tmp_path / "source.db"
    vault_path = tmp_path / "vault.db"
    _consolidated_source(source)
    build_vault(source, vault_path, measure_gzip=False)
    _ingest_consolidated_total(vault_path)

    real_history_at = vault._history_at
    history = real_history_at(vault_path)
    # Keep the old count so the build-time assertion has an affirmative
    # expectation, but disable the rows that would otherwise be restored.
    broken_history = (*history[:5], history[5], [], [], {})
    monkeypatch.setattr(vault, "_history_at", lambda path: broken_history)

    with pytest.raises(ValueError, match=r"drop 1 consolidated daily total"):
        build_vault(source, vault_path, replace=True, measure_gzip=False)

    assert vault_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute(
            "SELECT value FROM hk_daily_totals "
            "WHERE metric = 'step_count' AND local_date = '2026-08-20'"
        ).fetchone()[0] == 6.0
        assert conn.execute(
            "SELECT sum, source_kind FROM daily_metrics "
            "WHERE metric = 'step_count' AND date = '2026-08-20'"
        ).fetchone()[:] == (6.0, "apple_consolidated")
    finally:
        conn.close()
    result = _verify(vault_path)
    assert result.returncode == 0, result.stdout
    assert "consolidated rows: 1 checked, 0 discrepancy(ies)" in result.stdout


def test_18b_daily_totals_expected_from_survives_rebuild(tmp_path):
    """18b. The check-6 expectation is carried into the rebuilt vault."""
    source = tmp_path / "source.db"
    vault_path = tmp_path / "vault.db"
    _consolidated_source(source)
    build_vault(source, vault_path, measure_gzip=False)
    _ingest_consolidated_total(vault_path)

    build_vault(source, vault_path, replace=True, measure_gzip=False)

    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute(
            "SELECT value FROM vault_meta "
            "WHERE key = 'daily_totals_expected_from:step_count'"
        ).fetchone()[0] == "2026-08-20"
    finally:
        conn.close()


def test_rebuild_never_moves_the_watermark_onto_live_healthkit_days(tmp_path):
    """A refreshed snapshot must not wall off a day the phone is already syncing.

    Without the cap this is a permanent, silent wedge rather than a bad day:
    `_healthkit_ingest` refuses any batch containing the walled-off day, and the
    client commits its anchor only after a 2xx, so it retries the identical
    batch forever. Measured 2026-08-22 against the real ingest path.
    """
    source = tmp_path / "snapshot.db"
    conn = db.connect(source)
    db.init_db(conn)
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, dedupe_key) VALUES "
        "('heart_rate', 60, 'count/min', '2026-08-21T10:00:00Z', "
        "'2026-08-21T10:00:00Z', '2026-08-21', 'Watch', 'backfill', 'k1')")
    conn.commit()
    conn.close()

    vault_path = tmp_path / "vault.db"
    first = vault.build_vault(source, vault_path, measure_gzip=False)
    assert first["history_imported_through"] == "2026-08-21"

    # The phone syncs a day of live data into the vault.
    conn = db.connect(vault_path)
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, hk_uuid, dedupe_key) VALUES "
        "('heart_rate', 70, 'count/min', '2026-08-22T08:00:00Z', "
        "'2026-08-22T08:00:00Z', '2026-08-22', 'Watch', 'healthkit', "
        "'uuid-live', 'k-live')")
    conn.commit()
    conn.close()

    # The snapshot is refreshed and now reaches the day the phone is syncing.
    conn = db.connect(source)
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, dedupe_key) VALUES "
        "('heart_rate', 61, 'count/min', '2026-08-22T10:00:00Z', "
        "'2026-08-22T10:00:00Z', '2026-08-22', 'Watch', 'backfill', 'k2')")
    conn.commit()
    conn.close()

    second = vault.build_vault(source, vault_path, replace=True,
                               measure_gzip=False)
    assert second["history_imported_through"] == "2026-08-21", (
        "the watermark reached a day already carrying HealthKit-direct rows")

    conn = db.connect(vault_path, read_only=True)
    try:
        assert vault.history_imported_through(conn) == "2026-08-21"
    finally:
        conn.close()


def test_watermark_cap_clears_when_every_day_carries_live_data(tmp_path):
    """No day left to declare means no declaration, not a false one."""
    source = tmp_path / "snapshot.db"
    conn = db.connect(source)
    db.init_db(conn)
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, dedupe_key) VALUES "
        "('heart_rate', 60, 'count/min', '2026-08-22T10:00:00Z', "
        "'2026-08-22T10:00:00Z', '2026-08-22', 'Watch', 'backfill', 'k1')")
    conn.commit()
    conn.close()

    vault_path = tmp_path / "vault.db"
    vault.build_vault(source, vault_path, measure_gzip=False)
    conn = db.connect(vault_path)
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, hk_uuid, dedupe_key) VALUES "
        "('heart_rate', 70, 'count/min', '2026-08-22T08:00:00Z', "
        "'2026-08-22T08:00:00Z', '2026-08-22', 'Watch', 'healthkit', "
        "'uuid-live', 'k-live')")
    conn.commit()
    conn.close()

    second = vault.build_vault(source, vault_path, replace=True,
                               measure_gzip=False)
    assert second["history_imported_through"] == "2026-08-21"
