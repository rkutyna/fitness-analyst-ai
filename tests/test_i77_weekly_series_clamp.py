"""engine #77 item 1: weekly_series must not publish a week past the data or
past the caller's end.

Before the fix, `period` was always `monday:monday+6` regardless of what data
existed or what `end` the caller asked for -- a four-day vault (Mon..Thu)
still published a period ending the following Sunday, and that string is what
an answering model copies verbatim into a claim (`get_weekly_series`'s own
docstring says as much).

The clamp is against the SERIES' OWN global first/last fetched date, not
against the entries inside each individual week: clamping per-bucket would
shrink every mid-history week of a sparse metric (body_mass weighed twice a
week, say) to its own first/last logged day and mislabel a complete week of
observation as partial. Only the week containing the series' first date, and
the week containing its last date, can ever come out shorter than 7 days.
"""
from __future__ import annotations

from pathlib import Path

from health_advisor import analysis, db


def _seed_step_count(conn, dates: list[str]) -> None:
    for d in dates:
        conn.execute(
            "INSERT INTO daily_metrics "
            "(date, metric, count, sum, avg, unit, source_kind) "
            "VALUES (?,?,?,?,?,?,?)",
            (d, "step_count", 1, 1000.0, 1000.0, "count", "cumulative"))
    conn.commit()


def _vault(tmp_path: Path):
    conn = db.connect(tmp_path / "i77-weekly.db")
    db.init_db(conn)
    return conn


def test_trailing_partial_week_clamps_to_the_last_day_with_data(tmp_path):
    """Vault covers only Mon..Thu (2029-12-31..2030-01-03); the trailing week
    must not claim through the following Sunday."""
    conn = _vault(tmp_path)
    _seed_step_count(conn, [
        "2029-12-31", "2030-01-01", "2030-01-02", "2030-01-03"])

    rows = analysis.weekly_series(conn, "step_count", "2029-12-31", "2030-01-06")

    assert len(rows) == 1
    row = rows[0]
    assert row["week_start"] == "2029-12-31"
    assert row["period"] == "2029-12-31:2030-01-03"
    assert row["n_days"] == 4
    assert row["partial"] is True
    conn.close()


def test_trailing_partial_week_also_clamps_to_the_callers_end(tmp_path):
    """Same vault, but the caller's own `end` cuts the week short of the data
    that exists -- the period must not run past what was asked for either."""
    conn = _vault(tmp_path)
    _seed_step_count(conn, [
        "2029-12-31", "2030-01-01", "2030-01-02", "2030-01-03"])

    rows = analysis.weekly_series(conn, "step_count", "2029-12-31", "2030-01-02")

    assert len(rows) == 1
    row = rows[0]
    assert row["week_start"] == "2029-12-31"
    assert row["period"] == "2029-12-31:2030-01-02"
    assert row["n_days"] == 3
    assert row["partial"] is True
    conn.close()


def test_a_full_week_of_data_stays_monday_to_sunday_and_is_not_partial(tmp_path):
    """Control case: data on all 7 days of the week must still publish the
    full Monday-Sunday span with partial=False."""
    conn = _vault(tmp_path)
    _seed_step_count(conn, [
        "2029-12-31", "2030-01-01", "2030-01-02", "2030-01-03",
        "2030-01-04", "2030-01-05", "2030-01-06"])

    rows = analysis.weekly_series(conn, "step_count", "2029-12-31", "2030-01-06")

    assert len(rows) == 1
    row = rows[0]
    assert row["week_start"] == "2029-12-31"
    assert row["period"] == "2029-12-31:2030-01-06"
    assert row["n_days"] == 7
    assert row["partial"] is False
    conn.close()


def test_sparse_middle_week_stays_a_full_week_not_its_own_tue_fri_span(tmp_path):
    """A metric logged only twice a week (e.g. body_mass) must not have an
    interior week clamped to its own first/last logged day. Three consecutive
    full weeks, data on Tue and Fri of each, with the query range running
    past both ends of the data -- only the leading and trailing weeks (which
    contain the series' own first/last date) may come out partial; the
    middle week is a complete, ordinarily-observed week and must publish the
    full Monday-Sunday span."""
    conn = _vault(tmp_path)
    _seed_step_count(conn, [
        "2030-01-01", "2030-01-04",   # week 1: Tue, Fri (monday 2029-12-31)
        "2030-01-08", "2030-01-11",   # week 2: Tue, Fri (monday 2030-01-07)
        "2030-01-15", "2030-01-18",   # week 3: Tue, Fri (monday 2030-01-14)
    ])

    rows = analysis.weekly_series(conn, "step_count", "2029-12-30", "2030-01-20")

    assert len(rows) == 3
    week1, week2, week3 = rows

    assert week1["week_start"] == "2029-12-31"
    assert week1["period"] == "2030-01-01:2030-01-06"
    assert week1["n_days"] == 2
    assert week1["partial"] is True

    assert week2["week_start"] == "2030-01-07"
    assert week2["period"] == "2030-01-07:2030-01-13"
    assert week2["n_days"] == 2
    assert week2["partial"] is False

    assert week3["week_start"] == "2030-01-14"
    assert week3["period"] == "2030-01-14:2030-01-18"
    assert week3["n_days"] == 2
    assert week3["partial"] is True
    conn.close()
