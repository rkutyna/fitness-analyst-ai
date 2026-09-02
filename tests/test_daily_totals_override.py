"""D19: Apple's consolidated total survives every recompute, and changes nothing else.

Assertions 8-12 and 12a-12c of the design. Step 1 of the build order, so nothing
here goes through the ingest path: totals are written with
`db.insert_daily_totals` directly, which is what the receiver will call.

The load-bearing one is assertion 8. `_healthkit_ingest` recomputes
(step_count, day) on every batch carrying a raw step sample, so a consolidated
value written once at ingest would be rebuilt into the double-count by the very
next sync. The override has to be a property of the recompute.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from health_advisor import analysis
from health_advisor import db as dbmod
from health_advisor import metrics as mx

DAY = "2026-08-25"
APPLE = 10173.0


def _total(metric="step_count", local_date=DAY, value=APPLE, unit="count",
           state="provisional", queried_at="2026-08-26T09:00:00"):
    return {"metric": metric, "local_date": local_date, "value": value,
            "unit": unit, "interval": "day", "state": state,
            "device_id": "dev-1", "queried_at": queried_at}


def _record(conn, metric, value, start_utc, local_date=DAY, unit="count",
            source="Apple Watch"):
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, dedupe_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'receiver', ?)",
        (metric, value, unit, start_utc, start_utc, local_date, source,
         f"{metric}|{value}|{start_utc}|{source}"))


def _dm(conn, metric="step_count", day=DAY):
    return conn.execute(
        "SELECT * FROM daily_metrics WHERE metric = ? AND date = ?",
        (metric, day)).fetchone()


# --- 8. a raw sample arriving after a consolidated total does not revert it ---

def test_a_later_raw_sample_does_not_revert_the_consolidated_total(conn):
    _record(conn, "step_count", 4000.0, f"{DAY}T09:00:00Z")
    dbmod.recompute_daily_metrics(conn, pairs=[("step_count", DAY)])
    dbmod.insert_daily_totals(conn, [_total()], batch_id="b1")
    dbmod.recompute_daily_metrics(conn, pairs=[("step_count", DAY)])
    assert _dm(conn)["sum"] == APPLE

    # The next sync brings another raw sample for the same day — exactly what
    # receiver.py:522 recomputes on every batch.
    _record(conn, "step_count", 2500.0, f"{DAY}T17:00:00Z")
    dbmod.recompute_daily_metrics(conn, pairs=[("step_count", DAY)])

    row = _dm(conn)
    assert row["sum"] == APPLE, "the double-count was rebuilt on top of Apple's total"
    assert row["source_kind"] == "apple_consolidated"
    assert row["count"] == 2, "count is the raw record count and must have moved"


# --- 9. a full rebuild does not revert it ------------------------------------

def test_a_full_rebuild_does_not_revert_the_consolidated_total(conn):
    _record(conn, "step_count", 4000.0, f"{DAY}T09:00:00Z")
    _record(conn, "step_count", 2500.0, f"{DAY}T17:00:00Z")
    dbmod.insert_daily_totals(conn, [_total()], batch_id="b1")

    dbmod.recompute_daily_metrics(conn, full=True)

    row = _dm(conn)
    assert row["sum"] == APPLE
    assert row["source_kind"] == "apple_consolidated"
    assert row["count"] == 2


# --- 10. a day with no raw samples still gets a row --------------------------

def test_a_day_with_no_raw_samples_gets_an_honest_row(conn):
    dbmod.insert_daily_totals(conn, [_total()], batch_id="b1")
    dbmod.recompute_daily_metrics(conn, pairs=[("step_count", DAY)])

    row = _dm(conn)
    assert row is not None
    assert row["sum"] == APPLE
    assert row["count"] == 0, "count is never synthesised: no raw samples means 0"
    assert row["avg"] is None and row["min"] is None
    assert row["max"] is None and row["last"] is None
    assert row["unit"] == "count"
    assert row["source_kind"] == "apple_consolidated"


# --- 11. only `sum` moves ----------------------------------------------------

def test_only_sum_moves_every_other_column_stays_records_derived(conn):
    _record(conn, "step_count", 4000.0, f"{DAY}T09:00:00Z")
    _record(conn, "step_count", 2500.0, f"{DAY}T17:00:00Z")
    dbmod.recompute_daily_metrics(conn, pairs=[("step_count", DAY)])
    before = dict(_dm(conn))

    dbmod.insert_daily_totals(conn, [_total()], batch_id="b1")
    dbmod.recompute_daily_metrics(conn, pairs=[("step_count", DAY)])
    after = dict(_dm(conn))

    assert before["sum"] == 6500.0 and after["sum"] == APPLE
    for column in ("count", "avg", "min", "max", "last", "unit"):
        assert after[column] == before[column], column
    # Stated in the design rather than discovered as a bug later.
    assert after["avg"] * after["count"] != after["sum"]


# --- 12. nothing touches the jog dial ----------------------------------------

def test_a_consolidated_distance_total_does_not_move_the_jog_dial(conn):
    """`impact_bucket_rows` reads `records` directly and must never see this."""
    start = f"{DAY}T12:00:00Z"
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('other', ?, ?, ?, 0.33, "
        "'test', ?)", (start, f"{DAY}T12:00:20Z", DAY, f"w|{start}"))
    _record(conn, "distance_walking_running", 0.0333, start, unit="mi")
    _record(conn, "step_count", 47.0, start)
    dbmod.recompute_daily_metrics(conn, full=True)
    before = mx.impact_bucket_rows(conn, "local_date BETWEEN ? AND ?", (DAY, DAY))
    assert before, "fixture produced no buckets; the assertion would be vacuous"

    dbmod.insert_daily_totals(
        conn, [_total(metric="distance_walking_running", value=4.2, unit="mi")],
        batch_id="b1")
    dbmod.recompute_daily_metrics(conn, pairs=[("distance_walking_running", DAY)])
    assert _dm(conn, "distance_walking_running")["sum"] == 4.2

    after = mx.impact_bucket_rows(conn, "local_date BETWEEN ? AND ?", (DAY, DAY))
    assert after == before


# --- 12a / 12b. the wear gate ------------------------------------------------

WINDOW_END = "2026-08-28"
WINDOW_START = (date.fromisoformat(WINDOW_END) - timedelta(days=27)).isoformat()
CONSOLIDATED_DAY = "2026-08-20"


def _seed_wear_window(conn, *, wear_hours_on_consolidated_day: bool,
                      label: str = "apple_consolidated"):
    """28 dense step_count days, one of which is a consolidated total with
    `count = 0` — the shape a consolidated day with no raw samples has."""
    d = date.fromisoformat(WINDOW_START)
    end = date.fromisoformat(WINDOW_END)
    while d <= end:
        day = d.isoformat()
        consolidated = day == CONSOLIDATED_DAY
        conn.execute(
            "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, "
            "max, last, unit, source_kind) VALUES ('step_count', ?, ?, ?, 1.0, "
            "1.0, 1.0, 1.0, 'count', ?)",
            (day, 0 if consolidated else 400,
             APPLE if consolidated else 6000.0,
             label if consolidated else "records"))
        if not consolidated or wear_hours_on_consolidated_day:
            conn.execute(
                "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, "
                "max, last, unit) VALUES ('wear_hours', ?, 1, 18.0, 18.0, 18.0, "
                "18.0, 18.0, 'h')", (day,))
        d += timedelta(days=1)
    conn.commit()
    return analysis._daily_load_rows(conn, "step_count", WINDOW_END, 28)


def test_12a_a_consolidated_day_with_measured_wear_is_kept(conn):
    """`wear_hours` decides where it exists — it is the measurement, not the
    proxy — so the day survives despite `count = 0`.

    NOTE, against the design's wording: this is a no-regression assertion, not a
    before/after one. The design says "today the same fixture drops it"; it does
    not. `_worn_rows` consults density only for a day with no `wear_hours` row,
    so this day was kept before the change too. The test below is the one that
    covers the behaviour that actually moved.
    """
    rows = _seed_wear_window(conn, wear_hours_on_consolidated_day=True)
    assert CONSOLIDATED_DAY in {d for d, _, _ in rows}

    worn = analysis._worn_rows(conn, "step_count", WINDOW_END, rows)

    assert CONSOLIDATED_DAY in {d for d, _, _ in worn}
    assert (CONSOLIDATED_DAY, APPLE, 0) in worn
    assert APPLE in analysis._worn_values(conn, "step_count", WINDOW_END, rows)


def test_12a_the_label_alone_changes_nothing_when_wear_is_measured(conn):
    """The `source_kind` term must not disturb a day that has wear evidence."""
    rows = _seed_wear_window(conn, wear_hours_on_consolidated_day=True)
    labelled = analysis._worn_rows(conn, "step_count", WINDOW_END, rows)
    conn.execute("UPDATE daily_metrics SET source_kind = 'records' "
                 "WHERE metric = 'step_count' AND date = ?", (CONSOLIDATED_DAY,))
    conn.commit()

    assert analysis._worn_rows(conn, "step_count", WINDOW_END, rows) == labelled


def test_12b_a_consolidated_day_with_no_wear_evidence_leaves_the_window(conn):
    """Absent from the worn set AND from the count a caller sizes against.

    A day that is dropped but still counted votes "not worn" without evidence.
    `count = 0` on a consolidated row is an honest raw count, not a wear signal,
    so the density proxy is testing a property the value no longer has.
    """
    rows = _seed_wear_window(conn, wear_hours_on_consolidated_day=False)
    assert CONSOLIDATED_DAY in {d for d, _, _ in rows}

    eligible = analysis._wear_eligible(conn, "step_count", WINDOW_END, rows)
    worn = analysis._worn_rows(conn, "step_count", WINDOW_END, rows)

    assert CONSOLIDATED_DAY not in {d for d, _, _ in eligible}
    assert len(eligible) == len(rows) - 1, "it must leave the denominator too"
    assert CONSOLIDATED_DAY not in {d for d, _, _ in worn}
    assert APPLE not in analysis._worn_values(conn, "step_count", WINDOW_END, rows)


def test_12b_without_the_label_the_same_day_stays_in_the_denominator(conn):
    """The before state, so this test proves the fix rather than describing it.

    Unlabelled, the day is judged by density, fails it, and is dropped from the
    worn set while remaining in the window a caller counts.
    """
    rows = _seed_wear_window(conn, wear_hours_on_consolidated_day=False,
                             label="records")

    eligible = analysis._wear_eligible(conn, "step_count", WINDOW_END, rows)
    worn = analysis._worn_rows(conn, "step_count", WINDOW_END, rows)

    assert len(eligible) == len(rows)
    assert CONSOLIDATED_DAY in {d for d, _, _ in eligible}
    assert CONSOLIDATED_DAY not in {d for d, _, _ in worn}


def test_12b_the_denominator_half_is_visible_through_training_load(conn):
    """The only caller whose minimum is sized from `len(rows)`.

    D19 writes no consolidated total for an ACWR metric — hr_load_proxy,
    active_energy, apple_exercise_time — so this exercises the mechanism on a
    metric the design does not touch, which is the only place it is observable:
    `movers` sizes its minimum against a constant.
    """
    d = date.fromisoformat(WINDOW_START)
    end = date.fromisoformat(WINDOW_END)
    while d <= end:
        day = d.isoformat()
        consolidated = day == CONSOLIDATED_DAY
        conn.execute(
            "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, "
            "max, last, unit, source_kind) VALUES ('active_energy', ?, ?, ?, "
            "1.0, 1.0, 1.0, 1.0, 'kcal', ?)",
            (day, 0 if consolidated else 400, 500.0,
             "apple_consolidated" if consolidated else "records"))
        if not consolidated:
            conn.execute(
                "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, "
                "max, last, unit) VALUES ('wear_hours', ?, 1, 18.0, 18.0, 18.0, "
                "18.0, 18.0, 'h')", (day,))
        d += timedelta(days=1)
    conn.commit()

    loaded = analysis.training_load(conn, WINDOW_END)

    assert loaded["load_metric"] == "active_energy"
    # 28 seeded days, one excluded: the day neither votes "not worn" nor raises
    # the bar the worn days have to clear.
    assert loaded["n_days"] == 27
    assert loaded["status"] == "ok", loaded


# --- 12c. the override never runs on an early return -------------------------

def test_12c_the_override_runs_on_both_early_returns(conn):
    """Against the branch-end placement this fails twice.

    `records` is empty, so `full=True` returns at `if not rebuildable` and
    `pairs=[]` returns at `if not pairs`. Both are states in which a
    consolidated total can sit in `hk_daily_totals` behind a stale
    double-counted `daily_metrics` row.
    """
    conn.execute(
        "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, "
        "last, unit) VALUES ('step_count', ?, 2, 16673.0, 8336.5, 4000.0, "
        "12673.0, 12673.0, 'count')", (DAY,))
    dbmod.insert_daily_totals(conn, [_total()], batch_id="b1")
    conn.commit()

    assert dbmod.recompute_daily_metrics(conn, pairs=[]) == 0
    row = _dm(conn)
    assert row["sum"] == APPLE, "the pairs=[] early return skipped the override"
    assert row["source_kind"] == "apple_consolidated"

    conn.execute("UPDATE daily_metrics SET sum = 16673.0, "
                 "source_kind = 'records' WHERE metric = 'step_count'")
    assert dbmod.recompute_daily_metrics(conn, full=True) == 0
    row = _dm(conn)
    assert row["sum"] == APPLE, "the not-rebuildable early return skipped it"
    assert row["source_kind"] == "apple_consolidated"


def test_the_override_is_idempotent(conn):
    dbmod.insert_daily_totals(conn, [_total()], batch_id="b1")
    first = dbmod.apply_consolidated_totals(conn)
    second = dbmod.apply_consolidated_totals(conn)
    assert first == second == 1
    assert conn.execute(
        "SELECT COUNT(*) c FROM daily_metrics").fetchone()["c"] == 1


def test_a_database_without_the_totals_table_is_untouched(conn):
    """The runtime guard, the way `_arbitration` guards on `workouts`."""
    conn.execute("DROP TABLE hk_daily_totals")
    _record(conn, "step_count", 4000.0, f"{DAY}T09:00:00Z")

    assert dbmod.recompute_daily_metrics(conn, pairs=[("step_count", DAY)]) == 1

    row = _dm(conn)
    assert row["sum"] == 4000.0
    assert row["source_kind"] == "records"
