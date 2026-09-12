"""Per-row formula versions protect derived history from stale reads."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from health_advisor import db
from health_advisor import derive
from health_advisor import metrics
from health_advisor import analyst_runner
from health_advisor import mcp_server
from health_advisor.context import VaultContext
from tests.test_derive import _seed_night


def _db(tmp_path):
    conn = db.connect(tmp_path / "versioned.db")
    db.init_db(conn)
    return conn


def test_version_bump_rederives_exactly_source_backed_days(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    try:
        _seed_night(conn, "2026-07-10")
        derive._upsert(conn, "sleep_bedtime", "2026-07-10", 99.0)
        conn.execute(
            "UPDATE daily_metrics SET derived_version = ? "
            "WHERE metric = 'sleep_bedtime' AND date = ?",
            (metrics.DERIVED_FORMULA_VERSION, "2026-07-10"),
        )
        conn.commit()

        monkeypatch.setattr(metrics, "DERIVED_FORMULA_VERSION",
                            metrics.DERIVED_FORMULA_VERSION + 1)
        before = derive.derived_version_report(conn)
        result = derive.rederive_stale(conn)

        row = conn.execute(
            "SELECT avg, derived_version FROM daily_metrics "
            "WHERE metric = 'sleep_bedtime' AND date = '2026-07-10'"
        ).fetchone()
        assert result["rederived_days"] == 1
        assert before == {"fresh": 0, "stale": 1, "unrecomputable": 0}
        assert row["avg"] == pytest.approx(11.0)
        assert row["derived_version"] == metrics.DERIVED_FORMULA_VERSION
        assert result["after"]["stale"] == 0
    finally:
        conn.close()


def test_source_less_stale_row_is_preserved_reported_and_not_read(
        tmp_path, monkeypatch):
    conn = _db(tmp_path)
    try:
        derive._upsert(conn, "jog_minutes", "2018-01-02", 42.0)
        old_version = metrics.DERIVED_FORMULA_VERSION
        monkeypatch.setattr(metrics, "DERIVED_FORMULA_VERSION", old_version + 1)
        conn.execute(
            "UPDATE daily_metrics SET derived_version = ? "
            "WHERE metric = 'jog_minutes' AND date = '2018-01-02'",
            (old_version,),
        )
        conn.commit()

        result = derive.rederive_stale(conn)
        row = conn.execute(
            "SELECT last, derived_version FROM daily_metrics "
            "WHERE metric = 'jog_minutes' AND date = '2018-01-02'"
        ).fetchone()
        dates, values, _ = metrics.series(
            conn, "jog_minutes", "2018-01-01", "2018-01-03")

        assert result["before"] == {"fresh": 0, "stale": 0, "unrecomputable": 1}
        assert result["after"] == result["before"]
        assert row["last"] == 42.0
        assert row["derived_version"] == old_version
        assert dates == [] and values == []
    finally:
        conn.close()


def test_analyst_projection_excludes_stale_and_marks_legacy_rows(
        tmp_path, monkeypatch):
    conn = _db(tmp_path)
    try:
        derive._upsert(conn, "jog_minutes", "2026-07-10", 42.0)
        derive._upsert(conn, "jog_minutes", "2026-07-11", 43.0)
        derive._upsert(conn, "jog_minutes", "2026-07-12", 44.0)
        conn.execute(
            "UPDATE daily_metrics SET derived_version = ? "
            "WHERE metric = 'jog_minutes' AND date = '2026-07-10'",
            (metrics.DERIVED_FORMULA_VERSION - 1,),
        )
        conn.execute(
            "UPDATE daily_metrics SET derived_version = NULL "
            "WHERE metric = 'jog_minutes' AND date = '2026-07-12'"
        )
        conn.commit()
        sql = analyst_runner._current_daily_metrics_sql(
            "SELECT date, last, verification_status FROM daily_metrics "
            "WHERE metric = 'jog_minutes' ORDER BY date")
        rows = conn.execute(sql).fetchall()
        assert [(row["date"], row["last"], row["verification_status"])
                for row in rows] == [
            ("2026-07-11", 43.0, metrics.VERIFIED_STATUS),
            ("2026-07-12", 44.0, metrics.UNVERIFIED_LEGACY_STATUS),
        ]
    finally:
        conn.close()


def test_pre_stamp_rows_are_mismatched_not_fresh(tmp_path):
    conn = _db(tmp_path)
    try:
        _seed_night(conn, "2026-07-10")
        conn.execute(
            "INSERT INTO daily_metrics "
            "(metric, date, count, sum, avg, min, max, last, unit) "
            "VALUES ('sleep_bedtime', '2026-07-10', 1, 99, 99, 99, 99, 99, 'h')"
        )
        conn.commit()
        assert derive.derived_version_report(conn) == {
            "fresh": 0, "stale": 1, "unrecomputable": 0
        }
    finally:
        conn.close()


def test_additive_migration_preserves_populated_rows(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE daily_metrics ("
        "metric TEXT NOT NULL, date TEXT NOT NULL, count INTEGER NOT NULL, "
        "sum REAL, avg REAL, min REAL, max REAL, last REAL, unit TEXT, "
        "PRIMARY KEY (metric, date))"
    )
    conn.execute(
        "INSERT INTO daily_metrics VALUES "
        "('jog_minutes', '2018-01-02', 1, 42, 42, 42, 42, 42, 'min')"
    )
    conn.commit()
    conn.close()

    migrated = db.connect(path)
    try:
        assert migrated.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
        db.init_db(migrated)
        columns = {row[1] for row in migrated.execute(
            "PRAGMA table_info(daily_metrics)")}
        row = migrated.execute(
            "SELECT count, sum, avg, min, max, last, unit, derived_version "
            "FROM daily_metrics").fetchone()
        assert "derived_version" in columns
        assert tuple(row) == (1, 42.0, 42.0, 42.0, 42.0, 42.0, "min", None)
    finally:
        migrated.close()


def _seed_jog_minutes(path, version):
    conn = db.connect(path)
    db.init_db(conn)
    try:
        values = [3.15] * 27 + [3.25]
        for offset, value in enumerate(values):
            day = (date(2026, 5, 1) + timedelta(days=offset)).isoformat()
            conn.execute(
                "INSERT INTO daily_metrics "
                "(metric, date, count, sum, avg, min, max, last, unit, "
                "derived_version) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
                ("jog_minutes", day, value, value, value, value, value,
                 "min", version),
            )
        conn.commit()
    finally:
        conn.close()


def test_weekly_series_serves_28_legacy_days_with_an_explicit_marker(tmp_path):
    path = tmp_path / "legacy-jog.db"
    _seed_jog_minutes(path, None)
    ctx = VaultContext.local(path)
    result = mcp_server.build_tools(ctx)["get_weekly_series"](
        "jog_minutes", "2026-05-01", "2026-05-28")

    assert result["status"] == metrics.UNVERIFIED_LEGACY_STATUS
    assert sum(row["n_days"] for row in result["weeks"]) == 28
    assert sum(row["total"] for row in result["weeks"]) == pytest.approx(88.3)
    assert all(row["verification_status"] ==
               metrics.UNVERIFIED_LEGACY_STATUS for row in result["weeks"])


def test_weekly_series_marks_current_version_differently(tmp_path):
    path = tmp_path / "current-jog.db"
    _seed_jog_minutes(path, metrics.DERIVED_FORMULA_VERSION)
    ctx = VaultContext.local(path)
    result = mcp_server.build_tools(ctx)["get_weekly_series"](
        "jog_minutes", "2026-05-01", "2026-05-28")

    assert result["status"] == metrics.VERIFIED_STATUS
    assert result["status"] != metrics.UNVERIFIED_LEGACY_STATUS
    assert sum(row["total"] for row in result["weeks"]) == pytest.approx(88.3)
    assert all(row["verification_status"] == metrics.VERIFIED_STATUS
               for row in result["weeks"])
