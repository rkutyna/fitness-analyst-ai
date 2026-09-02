"""D3 as an enforced contract (T-006).

Under D3 a vault carries every derived table but raw `records` only for the six
series that sample-level analysis needs. The failure this file exists to stop is
not a wrong number — it is a *right-looking empty one*. A tool that returns no
samples for `basal_energy` is saying "you have no data", when the truth is
"those samples are not in this vault". The VO2max ingest defect went unnoticed
for weeks on exactly that confusion.

So absence has to be explainable everywhere it can occur, and the tests below
are written against `basal_energy` — 5.7M rows in the snapshot, none in a vault,
and the largest series that will ever produce this answer.
"""
from __future__ import annotations

import pytest

from health_advisor import analysis as A
from health_advisor import db as dbmod
from health_advisor import vault as V
from tests.conftest import seed_metric


def _record(conn, metric, value, ts, day, source="Demo's Apple Watch"):
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) "
        "VALUES (?, ?, 'kcal', ?, ?, ?, ?, ?, 'test', ?)",
        (metric, value, ts, ts, ts, day, source, f"{metric}|{ts}|{source}|{value}"),
    )


# --------------------------------------------------------------------------- #
# the allowlist is the contract, not the database's contents
# --------------------------------------------------------------------------- #
def test_basal_energy_is_not_a_vault_raw_series():
    assert not V.raw_series_available("basal_energy")
    assert V.raw_series_available("heart_rate")


def test_intraday_refuses_basal_energy_even_when_the_rows_are_there(conn, tools):
    """The refusal is driven by the contract, not by what this database holds.

    If it were driven by the rows, every raw-dependent feature would work in
    development against the full snapshot and go quietly empty in production —
    which is the one place nobody is watching.
    """
    day = "2026-07-21"
    seed_metric(conn, "basal_energy", day, [1800.0])
    for hour in range(3):
        _record(conn, "basal_energy", 600.0, f"{day}T{hour:02d}:30:00+00:00", day)
    conn.commit()
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric='basal_energy'"
    ).fetchone()[0] == 3, "the rows are present; the refusal is not about that"

    out = tools.get_intraday("basal_energy", day)

    assert out["status"] == "unavailable"
    assert out["reason"] == "raw_series_not_in_vault"
    assert "buckets" not in out
    # Name the way forward, or the agent's only option is to try again.
    assert "get_daily_series" in out["detail"] or "summarize_metric" in out["detail"]
    assert "heart_rate" in out["raw_series_in_vault"]


def test_the_daily_aggregate_for_basal_energy_is_still_answerable(conn, tools):
    """The refusal is scoped to sample-level access. Losing the daily series
    too would make `unavailable` indistinguishable from 'we do not have it',
    which is the confusion this whole task is about."""
    seed_metric(conn, "basal_energy", "2026-07-21", [1800.0, 1810.0])
    conn.commit()

    out = tools.summarize_metric("basal_energy", "30d")

    assert out.get("status") != "unavailable"
    assert out["mean"] == pytest.approx(1805.0)


def test_vault_compacts_non_allowlisted_raw_records_but_keeps_aggregates(conn):
    """Non-allowlisted rows are transient until the watermark passes them."""
    day = "2026-07-22"
    V.declare_vault(conn)
    seed_metric(conn, "basal_energy", day, [1800.0])
    row = {
        "metric": "basal_energy", "value": 600.0, "unit": "kcal",
        "start_utc": f"{day}T12:00:00+00:00",
        "end_utc": f"{day}T12:00:00+00:00",
        "start_local": f"{day} 08:00:00", "local_date": day,
        "source": "test", "origin": "receiver",
        "dedupe_key": "d3-non-allowlisted-raw",
    }

    assert dbmod.insert_records(conn, [row]) == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE dedupe_key = ?",
        (row["dedupe_key"],),
    ).fetchone()[0] == 1
    assert V.compact(conn, through=day) == {"basal_energy": 1}
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE dedupe_key = ?",
        (row["dedupe_key"],),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT sum FROM daily_metrics WHERE metric = ? AND date = ?",
        ("basal_energy", day),
    ).fetchone()[0] == pytest.approx(1800.0)


def test_compaction_watermark_is_forward_and_violations_are_checkable(conn):
    day = "2026-07-22"
    later = "2026-07-23"
    V.declare_vault(conn)
    row = {
        "metric": "basal_energy", "value": 600.0, "unit": "kcal",
        "start_utc": f"{day}T12:00:00+00:00",
        "end_utc": f"{day}T12:00:00+00:00",
        "start_local": f"{day} 08:00:00", "local_date": day,
        "source": "test", "origin": "receiver",
        "dedupe_key": "compact-watermark-row",
    }
    dbmod.insert_records(conn, [row])
    assert V.uncompacted_violations(conn) == []
    assert V.compact(conn, through=later) == {"basal_energy": 1}
    assert V.compacted_through(conn) == later
    assert V.compact(conn, through=day) == {}
    assert V.compacted_through(conn) == later

    # Bypass the writer guard to prove the invariant is observable even if a
    # future writer is faulty or someone edits the SQLite file directly.
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("basal_energy", 601.0, "kcal", f"{day}T13:00:00+00:00",
         f"{day}T13:00:00+00:00", f"{day} 09:00:00", day, "test",
         "receiver", "compact-violation-row"),
    )
    assert [row["dedupe_key"] for row in V.uncompacted_violations(conn)] == [
        "compact-violation-row"
    ]

    with pytest.raises(ValueError, match="compacted_through"):
        dbmod.insert_records(conn, [{**row, "dedupe_key": "guarded-old-row"}])


def test_compaction_leaves_allowlisted_raw_and_daily_metrics_unchanged(conn):
    day = "2026-07-22"
    V.declare_vault(conn)
    seed_metric(conn, "basal_energy", day, [1800.0])
    seed_metric(conn, "heart_rate", day, [72.0])
    dbmod.insert_records(conn, [{
        "metric": "basal_energy", "value": 600.0, "unit": "kcal",
        "start_utc": f"{day}T12:00:00+00:00", "end_utc": f"{day}T12:00:00+00:00",
        "start_local": f"{day} 08:00:00", "local_date": day,
        "source": "test", "origin": "receiver", "dedupe_key": "compact-basal",
    }, {
        "metric": "heart_rate", "value": 145.0, "unit": "count/min",
        "start_utc": f"{day}T12:00:00+00:00", "end_utc": f"{day}T12:00:00+00:00",
        "start_local": f"{day} 08:00:00", "local_date": day,
        "source": "test", "origin": "healthkit", "dedupe_key": "compact-heart",
    }])
    before = conn.execute(
        "SELECT metric, count, sum, avg, min, max, last, unit "
        "FROM daily_metrics WHERE date = ? ORDER BY metric", (day,)
    ).fetchall()

    assert V.compact(conn, through=day) == {"basal_energy": 1}
    after = conn.execute(
        "SELECT metric, count, sum, avg, min, max, last, unit "
        "FROM daily_metrics WHERE date = ? ORDER BY metric", (day,)
    ).fetchall()
    assert after == before
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric = 'heart_rate'"
    ).fetchone()[0] == 1


def test_declared_vault_full_rebuild_preserves_partial_non_allowlisted_history(conn):
    V.declare_vault(conn)
    seed_metric(conn, "body_mass", "2026-07-20", [180.0, 181.0])
    _record(conn, "body_mass", 182.0, "2026-07-22T09:00:00+00:00", "2026-07-22")
    conn.commit()
    before = conn.execute(
        "SELECT metric, date, count, sum, avg, min, max, last, unit "
        "FROM daily_metrics WHERE metric = 'body_mass' ORDER BY date"
    ).fetchall()

    dbmod.recompute_daily_metrics(conn, full=True)

    after = conn.execute(
        "SELECT metric, date, count, sum, avg, min, max, last, unit "
        "FROM daily_metrics WHERE metric = 'body_mass' ORDER BY date"
    ).fetchall()
    assert after == before


def test_plain_snapshot_accepts_non_allowlisted_raw_records(tmp_path):
    """Backfill's undeclared full snapshot is not subject to D3 filtering."""
    path = tmp_path / "snapshot.db"
    conn = dbmod.connect(path)
    conn.executescript(dbmod.SCHEMA_PATH.read_text())
    dbmod.init_db(conn)
    row = {
        "metric": "basal_energy", "value": 600.0, "unit": "kcal",
        "start_utc": "2026-07-22T12:00:00+00:00",
        "end_utc": "2026-07-22T12:00:00+00:00",
        "start_local": "2026-07-22 08:00:00", "local_date": "2026-07-22",
        "source": "test", "origin": "backfill",
        "dedupe_key": "plain-snapshot-raw",
    }
    assert dbmod.insert_records(conn, [row]) == 1
    assert conn.execute(
        "SELECT metric FROM records WHERE dedupe_key = ?",
        (row["dedupe_key"],),
    ).fetchone()[0] == "basal_energy"
    conn.close()


# --------------------------------------------------------------------------- #
# provenance survives the filter
# --------------------------------------------------------------------------- #
def _two_instruments(conn, metric):
    """Six months of watch, then six of phone — a change that sticks."""
    for month, source in [(f"2022-0{m}", "Demo's Apple Watch") for m in range(1, 7)] + \
                         [(f"2022-{m:02d}", "Demo's iPhone") for m in range(7, 13)]:
        for day in range(1, 8):
            _record(conn, metric, 100.0, f"{month}-{day:02d}T09:00:00+00:00",
                    f"{month}-{day:02d}", source)
    conn.commit()


def test_instrument_eras_reads_the_derived_provenance_table(conn):
    _two_instruments(conn, "basal_energy")
    dbmod.rebuild_metric_source_months(conn, full=True)

    out = A.instrument_eras_status(conn, "basal_energy", "2022-01-01", "2022-12-31")

    assert out["status"] == "ok"
    assert out["provenance"] == "metric_source_months"
    assert out["boundaries"] == ["2022-07-01"]


def test_instrument_eras_falls_back_to_records_before_the_table_is_built(conn):
    """Every database that predates the table is in this state, and `records`
    is the authority the table is built from — so this is not a way around the
    contract, it is the same answer from the same evidence."""
    _two_instruments(conn, "basal_energy")

    out = A.instrument_eras_status(conn, "basal_energy", "2022-01-01", "2022-12-31")

    assert out["status"] == "ok"
    assert out["provenance"] == "records"
    assert out["boundaries"] == ["2022-07-01"]


def test_instrument_eras_says_unavailable_rather_than_no_boundaries(conn):
    """The distinction this task exists for. `[]` licenses averaging across the
    whole series; 'we cannot see instruments here' does not."""
    seed_metric(conn, "basal_energy", "2022-01-01", [1800.0] * 30)
    conn.commit()

    out = A.instrument_eras_status(conn, "basal_energy", "2022-01-01", "2022-12-31")

    assert out["status"] == "unavailable"
    assert out["boundaries"] == []
    assert out["provenance"] is None


def test_detect_eras_carries_the_provenance_it_had(conn):
    from health_advisor import history as H

    seed_metric(conn, "basal_energy", "2022-01-01", [1800.0] * 60)
    conn.commit()
    eras = H.detect_eras(conn, "basal_energy")

    assert eras, "a gap-free daily series is still one era"
    assert all(e["instrument_provenance"] == "unavailable" for e in eras)

    _two_instruments(conn, "basal_energy")
    dbmod.rebuild_metric_source_months(conn, full=True)
    eras = H.detect_eras(conn, "basal_energy")
    assert all(e["instrument_provenance"] == "metric_source_months" for e in eras)


def test_a_built_vault_carries_provenance_for_series_whose_samples_it_drops(
        conn, vault_path, tmp_path):
    """The point of persisting it: `basal_energy` samples do not travel, and
    without this the vault could not tell that its instrument ever changed."""
    _two_instruments(conn, "basal_energy")
    seed_metric(conn, "basal_energy", "2022-01-01", [1800.0] * 30)
    conn.commit()
    conn.close()

    out = tmp_path / "vault.db"
    V.build_vault(vault_path, out)

    vconn = dbmod.connect(out, read_only=True)
    try:
        assert vconn.execute(
            "SELECT COUNT(*) FROM records WHERE metric='basal_energy'"
        ).fetchone()[0] == 0, "the samples must not travel"
        status = A.instrument_eras_status(vconn, "basal_energy",
                                          "2022-01-01", "2022-12-31")
    finally:
        vconn.close()

    assert status["status"] == "ok"
    assert status["provenance"] == "metric_source_months"
    assert status["boundaries"] == ["2022-07-01"]


# --------------------------------------------------------------------------- #
# the full rebuild cannot delete what it cannot reconstruct
# --------------------------------------------------------------------------- #
def test_full_rebuild_leaves_series_it_has_no_raw_rows_for(conn):
    """In a vault, `records` holds six series and `daily_metrics` holds about a
    hundred. `DELETE FROM daily_metrics` followed by a rebuild would destroy the
    other ninety-odd and report success."""
    seed_metric(conn, "basal_energy", "2026-07-21", [1800.0])   # aggregate only
    _record(conn, "heart_rate", 61.0, "2026-07-21T09:00:00+00:00", "2026-07-21")
    conn.commit()

    dbmod.recompute_daily_metrics(conn, full=True)

    kept = conn.execute(
        "SELECT sum FROM daily_metrics WHERE metric='basal_energy'").fetchone()
    assert kept is not None, "a full rebuild deleted an aggregate it cannot rebuild"
    assert kept["sum"] == pytest.approx(1800.0)
    assert conn.execute(
        "SELECT COUNT(*) FROM daily_metrics WHERE metric='heart_rate'"
    ).fetchone()[0] == 1


def test_full_rebuild_records_what_it_rebuilt_and_what_it_skipped(conn):
    """`Not done when: the tool returns an empty series and logs a warning
    nobody reads` cuts both ways — a partial rebuild that says nothing is
    indistinguishable from a complete one until somebody asks."""
    seed_metric(conn, "basal_energy", "2026-07-21", [1800.0])
    _record(conn, "heart_rate", 61.0, "2026-07-21T09:00:00+00:00", "2026-07-21")
    conn.commit()

    dbmod.recompute_daily_metrics(conn, full=True)

    detail = conn.execute(
        "SELECT detail FROM ingest_log WHERE kind='rebuild' "
        "ORDER BY id DESC LIMIT 1").fetchone()["detail"]
    assert "rebuilt=1" in detail
    assert "left_intact=1" in detail
    assert "basal_energy" in detail, "name the series that were left alone"


def test_get_latest_says_when_its_sample_is_an_aggregate(conn, tools):
    """D9. `distance_walking_running` is stored as 20-second sums, so the
    'latest sample' is a window total whose timestamp is its earliest sample.
    Reporting that as an instantaneous reading would be a claim about a moment
    that never happened."""
    seed_metric(conn, "distance_walking_running", "2026-08-01", [3.1])
    _record(conn, "distance_walking_running", 0.001, "2026-08-01T09:00:00+00:00",
            "2026-08-01")
    seed_metric(conn, "heart_rate", "2026-08-01", [61.0])
    _record(conn, "heart_rate", 61.0, "2026-08-01T09:00:00+00:00", "2026-08-01")
    conn.commit()

    assert tools.get_latest("distance_walking_running")["latest_sample"][
        "resolution_seconds"] == 20
    assert tools.get_latest("heart_rate")["latest_sample"][
        "resolution_seconds"] == 0, "heart_rate is stored as recorded"


def test_the_vault_bucket_width_is_the_width_consumers_read():
    """Two copies of one number is how the vault ends up storing a resolution
    nothing reads any more."""
    from health_advisor import metrics as mx

    assert V.VAULT_BUCKET_SECONDS["distance_walking_running"] is \
        mx.IMPACT_BUCKET_SECONDS
