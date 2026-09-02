"""Dedupe-key semantics for records.

Health Auto Export re-transmits overlapping windows on every sync and
recomputes a sample's quantity between sends. The natural key of a cumulative
device sample is its window — (metric, start, end, source) — so a re-send must
UPDATE that row, never insert a second one. Instantaneous samples and nutrition
entries are the opposite: two different values at one timestamp are two real
observations, and collapsing them would destroy data.
"""
from __future__ import annotations

from health_advisor import backfill
from health_advisor import db as dbmod


STEP_START = "2026-07-04T15:00:00+00:00"
STEP_END = "2026-07-04T15:05:00+00:00"
WATCH = "Demo’s Apple Watch"


def _row(metric, value, unit, source=WATCH, start=STEP_START, end=STEP_END,
         origin="receiver"):
    return dict(
        metric=metric, value=value, unit=unit, start_utc=start, end_utc=end,
        start_local="2026-07-04 11:00:00", local_date="2026-07-04",
        source=source, origin=origin,
        dedupe_key=dbmod.record_key(metric, start, end, value, unit, source),
    )


def _rows(conn, metric):
    return conn.execute(
        "SELECT value FROM records WHERE metric = ? ORDER BY id", (metric,)
    ).fetchall()


# --------------------------------------------------------------------------- #
# Cumulative samples: one window, one row.
# --------------------------------------------------------------------------- #
def test_recomputed_cumulative_resend_updates_the_row(conn):
    dbmod.insert_records(conn, [_row("step_count", 120.0, "count")])
    dbmod.insert_records(conn, [_row("step_count", 137.0, "count")])

    assert [r["value"] for r in _rows(conn, "step_count")] == [137.0]


def test_recomputed_cumulative_resend_is_not_counted_as_added(conn):
    dbmod.insert_records(conn, [_row("step_count", 120.0, "count")])

    assert dbmod.insert_records(conn, [_row("step_count", 137.0, "count")]) == 0


def test_cumulative_resend_does_not_inflate_the_daily_sum(conn):
    dbmod.insert_records(conn, [_row("step_count", 120.0, "count")])
    dbmod.insert_records(conn, [_row("step_count", 137.0, "count")])
    dbmod.recompute_daily_metrics(conn, pairs=[("step_count", "2026-07-04")])

    row = conn.execute(
        "SELECT count, sum FROM daily_metrics WHERE metric = 'step_count'"
    ).fetchone()
    assert (row["count"], row["sum"]) == (1, 137.0)


def test_distinct_windows_of_a_cumulative_metric_stay_distinct(conn):
    dbmod.insert_records(conn, [_row("step_count", 120.0, "count")])
    dbmod.insert_records(conn, [_row("step_count", 90.0, "count",
                                     start=STEP_END, end="2026-07-04T15:10:00+00:00")])

    assert sorted(r["value"] for r in _rows(conn, "step_count")) == [90.0, 120.0]


def test_same_window_from_a_different_source_stays_distinct(conn):
    dbmod.insert_records(conn, [_row("step_count", 120.0, "count")])
    dbmod.insert_records(conn, [_row("step_count", 118.0, "count",
                                     source="Demo’s iPhone")])

    assert len(_rows(conn, "step_count")) == 2


# --------------------------------------------------------------------------- #
# Instantaneous samples and nutrition: same timestamp, genuinely distinct rows.
# --------------------------------------------------------------------------- #
def test_two_heart_rate_samples_at_one_timestamp_are_both_kept(conn):
    # 150 such groups exist in the live backfill: instantaneous readings, not
    # a re-send of one sample.
    dbmod.insert_records(conn, [_row("heart_rate", 61.0, "count/min",
                                     start=STEP_START, end=STEP_START)])
    dbmod.insert_records(conn, [_row("heart_rate", 148.0, "count/min",
                                     start=STEP_START, end=STEP_START)])

    assert sorted(r["value"] for r in _rows(conn, "heart_rate")) == [61.0, 148.0]


def test_two_foods_logged_at_one_timestamp_are_both_kept(conn):
    dbmod.insert_records(conn, [_row("dietary_energy_consumed", 250.0, "kcal",
                                     source="MyFitnessPal")])
    dbmod.insert_records(conn, [_row("dietary_energy_consumed", 400.0, "kcal",
                                     source="MyFitnessPal")])

    assert sorted(r["value"] for r in _rows(conn, "dietary_energy_consumed")) \
        == [250.0, 400.0]


# --------------------------------------------------------------------------- #
# The key itself.
# --------------------------------------------------------------------------- #
def test_record_key_ignores_value_for_a_cumulative_metric():
    a = dbmod.record_key("step_count", STEP_START, STEP_END, 120.0, "count", WATCH)
    b = dbmod.record_key("step_count", STEP_START, STEP_END, 137.0, "count", WATCH)

    assert a == b


def test_record_key_separates_values_for_an_instantaneous_metric():
    a = dbmod.record_key("heart_rate", STEP_START, STEP_START, 61.0, "count/min", WATCH)
    b = dbmod.record_key("heart_rate", STEP_START, STEP_START, 148.0, "count/min", WATCH)

    assert a != b


def test_record_key_separates_values_for_a_nutrition_metric():
    a = dbmod.record_key("dietary_protein", STEP_START, STEP_END, 12.0, "g", "MFP")
    b = dbmod.record_key("dietary_protein", STEP_START, STEP_END, 30.0, "g", "MFP")

    assert a != b


def test_record_key_uses_source_native_identity_for_normalized_samples():
    old = dbmod.record_key(
        "dietary_energy_consumed", STEP_START, STEP_END, 500.0, "Cal", "MFP",
        source_metric="dietary_energy_consumed", source_value=500.0,
    )
    current = dbmod.record_key(
        "dietary_energy_consumed", STEP_START, STEP_END, 500.0, "kcal", "MFP",
        source_metric="dietary_energy_consumed", source_value=500.0,
    )

    assert old == current


def test_record_key_source_metric_survives_catalog_renames():
    before = dbmod.record_key(
        "uncatalogued_metric", STEP_START, STEP_END, 3.0, "count", WATCH,
        source_metric="HKQuantityTypeIdentifierExampleMetric", source_value=3.0,
    )
    after = dbmod.record_key(
        "example_metric", STEP_START, STEP_END, 3.0, "count", WATCH,
        source_metric="HKQuantityTypeIdentifierExampleMetric", source_value=3.0,
    )

    assert before == after


def test_backfill_key_uses_the_healthkit_type_and_raw_value():
    attrs = {
        "type": "HKQuantityTypeIdentifierBodyMass", "value": "180.0",
        "unit": "lb", "startDate": "2026-07-04 11:00:00 -0400",
        "endDate": "2026-07-04 11:00:00 -0400", "sourceName": "Scale",
    }
    row = next(backfill._record_rows(attrs))
    expected = dbmod.record_key(
        row["metric"], row["start_utc"], row["end_utc"], row["value"],
        row["unit"], row["source"], source_metric=attrs["type"],
        source_value=attrs["value"],
    )
    assert row["dedupe_key"] == expected
