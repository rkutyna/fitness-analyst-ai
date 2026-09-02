from __future__ import annotations

from health_advisor import db
from health_advisor import vault as V


def _raw(metric: str, day: str, key: str) -> dict[str, str | float | None]:
    timestamp = f"{day}T12:00:00+00:00"
    return {
        "metric": metric,
        "value": 600.0,
        "unit": "kcal",
        "start_utc": timestamp,
        "end_utc": timestamp,
        "start_local": f"{day} 08:00:00",
        "local_date": day,
        "source": "test",
        "origin": "receiver",
        "dedupe_key": key,
    }


def test_compaction_status_reports_never_compacted(conn):
    assert V.compaction_status(conn)["status"] == "not_a_declared_vault"

    V.declare_vault(conn)
    assert db.insert_records(
        conn, [_raw("basal_energy", "2026-07-22", "never-compacted")]
    ) == 1

    state = V.compaction_status(conn)

    assert V.uncompacted_violations(conn) == []
    assert state == {
        "status": "never_compacted",
        "is_declared_vault": True,
        "watermark_exists": False,
        "compacted_through": None,
        "non_allowlisted_raw_total": 1,
        "non_allowlisted_raw_behind_watermark": 0,
    }


def test_compaction_status_reports_compacted_clean(conn):
    V.declare_vault(conn)
    assert db.insert_records(
        conn, [_raw("basal_energy", "2026-07-23", "clean-after-watermark")]
    ) == 1
    V.compact(conn, through="2026-07-22")

    state = V.compaction_status(conn)

    assert V.uncompacted_violations(conn) == []
    assert state == {
        "status": "compacted_clean",
        "is_declared_vault": True,
        "watermark_exists": True,
        "compacted_through": "2026-07-22",
        "non_allowlisted_raw_total": 1,
        "non_allowlisted_raw_behind_watermark": 0,
    }


def test_compaction_status_reports_compacted_violation(conn):
    V.declare_vault(conn)
    V.compact(conn, through="2026-07-22")
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        tuple(_raw("basal_energy", "2026-07-22", "behind-watermark").values()),
    )

    state = V.compaction_status(conn)

    assert len(V.uncompacted_violations(conn)) == 1
    assert state == {
        "status": "compacted_violated",
        "is_declared_vault": True,
        "watermark_exists": True,
        "compacted_through": "2026-07-22",
        "non_allowlisted_raw_total": 1,
        "non_allowlisted_raw_behind_watermark": 1,
    }
