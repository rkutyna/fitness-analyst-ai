"""A backfill that recomputes a (metric, day) from source and finds nothing
must not leave the prior-formula-version row behind: it is hidden from every
read, nothing can ever replace it, and the version check would count it stale
forever. A row whose source is gone is the unrecomputable class and stays."""
from __future__ import annotations

import pytest

from health_advisor import db, derive, metrics
from tests.test_derive import _rec, _seed_night

BAD_NIGHT = "2026-07-05"        # a 31-hour merged in-bed span: the guard refuses it
NEXT_DAY = "2026-07-06"         # a normal night whose regularity row read BAD_NIGHT
GOOD_DAY = "2026-07-10"         # a normal night nothing is wrong with
NO_SOURCE_DAY = "2018-01-02"    # no records at all

PER_NIGHT = ("sleep_awake_longest", "sleep_awakenings", "sleep_bedtime",
             "sleep_latency", "sleep_midpoint", "sleep_time_in_bed",
             "sleep_wake_time")


def _seed_31_hour_night(conn, day):
    # Four back-to-back asleep intervals: 10 h + 10 h + 10 h + 1 h = 31 h.
    for start, minutes in (("2026-07-04 20:00:00", 600),
                           ("2026-07-05 06:00:00", 600),
                           ("2026-07-05 16:00:00", 600),
                           ("2026-07-06 02:00:00", 60)):
        _rec(conn, "sleep_asleep", start, minutes, day)


def _vault(tmp_path, monkeypatch):
    path = tmp_path / "orphans.db"
    conn = db.connect(path)
    db.init_db(conn)
    _seed_31_hour_night(conn, BAD_NIGHT)
    _rec(conn, "sleep_asleep", "2026-07-06 22:00:00", 120, NEXT_DAY)
    _rec(conn, "sleep_asleep", "2026-07-07 00:20:00", 280, NEXT_DAY)
    _seed_night(conn, GOOD_DAY)
    old = metrics.DERIVED_FORMULA_VERSION
    # Rows the previous formula wrote, including ones the current rules decline.
    for m in PER_NIGHT:
        derive._upsert(conn, m, BAD_NIGHT, 7.0)
    derive._upsert(conn, derive.REGULARITY_METRIC, NEXT_DAY, 55.0)
    derive._upsert(conn, "sleep_bedtime", GOOD_DAY, 99.0)
    derive._upsert(conn, "jog_minutes", NO_SOURCE_DAY, 42.0)
    conn.execute("UPDATE daily_metrics SET derived_version = ?", (old,))
    conn.commit()
    monkeypatch.setattr(metrics, "DERIVED_FORMULA_VERSION", old + 1)
    return path, conn, old


def _rows(conn):
    return [tuple(r) for r in conn.execute(
        "SELECT metric, date, count, sum, avg, min, max, last, unit, "
        "derived_version FROM daily_metrics ORDER BY metric, date")]


def _row(conn, metric, day):
    return conn.execute(
        "SELECT * FROM daily_metrics WHERE metric = ? AND date = ?",
        (metric, day)).fetchone()


def test_guard_premise_the_31_hour_night_derives_nothing(tmp_path, monkeypatch):
    _, conn, _ = _vault(tmp_path, monkeypatch)
    try:
        derive.update_for_days(conn, [BAD_NIGHT])
        for m in PER_NIGHT:
            current = conn.execute(
                "SELECT 1 FROM daily_metrics WHERE metric = ? AND date = ? "
                "AND derived_version = ?",
                (m, BAD_NIGHT, metrics.DERIVED_FORMULA_VERSION)).fetchone()
            assert current is None, m
    finally:
        conn.close()


def test_backfill_removes_prior_rows_the_recompute_leaves_empty(
        tmp_path, monkeypatch):
    _, conn, _ = _vault(tmp_path, monkeypatch)
    try:
        before = derive.derived_version_report(conn)
        assert before == {"fresh": 0, "stale": 9, "unrecomputable": 1}
        result = derive.rederive_stale(conn)
        conn.commit()

        for m in PER_NIGHT:
            assert _row(conn, m, BAD_NIGHT) is None, m
        assert _row(conn, derive.REGULARITY_METRIC, NEXT_DAY) is None
        assert result["after"]["stale"] == 0
        assert result["after"]["unrecomputable"] == 1
        # A prior row the recompute DOES rebuild is rewritten and stamped.
        good = _row(conn, "sleep_bedtime", GOOD_DAY)
        assert good["derived_version"] == metrics.DERIVED_FORMULA_VERSION
        assert good["avg"] == pytest.approx(11.0)
    finally:
        conn.close()


def test_a_row_with_no_remaining_source_is_kept_and_stays_marked(
        tmp_path, monkeypatch):
    _, conn, old = _vault(tmp_path, monkeypatch)
    try:
        derive.rederive_stale(conn)
        kept = _row(conn, "jog_minutes", NO_SOURCE_DAY)
        assert kept["last"] == 42.0
        assert kept["derived_version"] == old
        assert derive.derived_version_report(conn)["unrecomputable"] == 1
        # Even when a run is handed that day directly, with nothing to
        # recompute it from, the row is not touched.
        derive.update_for_days(conn, [NO_SOURCE_DAY])
        conn.execute(
            "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, "
            "max, last, unit, derived_version) VALUES "
            "('sleep_bedtime', ?, 1, 5, 5, 5, 5, 5, 'h', ?)",
            (NO_SOURCE_DAY, old))
        derive.update_for_days(conn, [NO_SOURCE_DAY])
        orphan = _row(conn, "sleep_bedtime", NO_SOURCE_DAY)
        assert orphan is not None and orphan["last"] == 5.0
        assert orphan["derived_version"] == old
    finally:
        conn.close()


def test_a_second_backfill_changes_nothing(tmp_path, monkeypatch, capsys):
    path, conn, _ = _vault(tmp_path, monkeypatch)
    conn.close()
    derive.main(["--backfill", "--db", str(path)])
    first_out = capsys.readouterr().out
    conn = db.connect(path)
    try:
        after_first = _rows(conn)
        report_first = derive.derived_version_report(conn)
    finally:
        conn.close()
    derive.main(["--backfill", "--db", str(path)])
    second_out = capsys.readouterr().out
    conn = db.connect(path)
    try:
        assert _rows(conn) == after_first
        assert derive.derived_version_report(conn) == report_first
        assert report_first["stale"] == 0
    finally:
        conn.close()
    assert "stale=0" in first_out.splitlines()[-1]
    assert "stale=0" in second_out.splitlines()[-1]


def test_an_unstamped_row_follows_the_same_rule_as_a_stamped_one(
        tmp_path, monkeypatch):
    _, conn, _ = _vault(tmp_path, monkeypatch)
    try:
        conn.execute("UPDATE daily_metrics SET derived_version = NULL "
                     "WHERE (metric = 'sleep_latency' AND date = ?) "
                     "OR (metric = 'jog_minutes' AND date = ?)",
                     (BAD_NIGHT, NO_SOURCE_DAY))
        conn.commit()
        derive.rederive_stale(conn)
        # Source here, recompute empty: gone. Source gone: kept as legacy.
        assert _row(conn, "sleep_latency", BAD_NIGHT) is None
        legacy = _row(conn, "jog_minutes", NO_SOURCE_DAY)
        assert legacy is not None and legacy["derived_version"] is None
        assert derive.derived_version_report(conn)["stale"] == 0
    finally:
        conn.close()
