"""get_impact_volume's period arithmetic and threshold sensitivity (audit P3-8).

analysis.impact_volume classifies the buckets; this is about what the TOOL says
around that number. Three separate ways the week-over-week figure misled:

  - a week with no distance samples produced no row at all, so the next week's
    "vs the previous period" silently spanned the gap;
  - the current week was compared against a whole one all week long (live:
    -71.3% on a Wednesday, no flag);
  - the 16 min/mi jog cutoff sits inside the pace band Week 5 is training
    toward, so slowing toward the plan's own target can drop buckets out of
    jog_minutes entirely.

The numbers themselves must not move: a consistency test pins the tool's
periods to analysis.impact_volume's.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from health_advisor import analysis as A


BLOCK_WEEK_VALUES = {
    "2026-06-15": 39.0,
    "2026-06-22": 17.3,
    "2026-06-29": 45.3,
    "2026-07-06": 16.7,
    "2026-07-13": 71.0,
    "2026-07-20": 68.0,
    "2026-07-27": 46.7,
    "2026-08-03": 32.7,
    "2026-08-10": 52.7,
    "2026-08-17": 68.3,
}


def _emit_impact_buckets(conn, local_date: str, minutes: float):
    """Emit one valid 20-second jogging row per bucket.

    Keeping this fixture at the shared bucket resolution makes the expected
    weekly values come through analysis.impact_volume, rather than bypassing
    the classifier with a stored daily metric or a mocked result.
    """
    t0 = datetime.fromisoformat(f"{local_date}T12:00:00+00:00")
    n_buckets = round(minutes * 3)
    per_bucket_mi = 20.0 / (12.0 * 60.0)
    end = t0 + timedelta(seconds=n_buckets * 20)
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('running', ?, ?, ?, ?, 'test', ?)",
        (t0.isoformat(), end.isoformat(), local_date, n_buckets / 3.0,
         f"block-workout-{local_date}"),
    )
    for i in range(n_buckets):
        ts = (t0 + timedelta(seconds=i * 20)).isoformat()
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('distance_walking_running', ?, 'mi', ?, ?, ?, ?, 't', 't', ?)",
            (per_bucket_mi, ts, ts, ts, local_date,
             f"block-{local_date}-{i}"),
        )
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('step_count', 47.0, 'count', ?, ?, ?, ?, 't', 't', ?)",
            (ts, ts, ts, local_date, f"block-steps-{local_date}-{i}"),
        )
    conn.commit()


@pytest.fixture
def blockdb(conn):
    for week, minutes in BLOCK_WEEK_VALUES.items():
        _emit_impact_buckets(conn, week, minutes)


def _emit(conn, local_date: str, start_hhmm: str, seconds: int,
          pace_min_per_mi: float, hr: float | None = None,
          cadence_spm: float = 141.0):
    """One distance sample per second at a steady pace (as test_impact_volume),
    plus a heart-rate sample every 5s when `hr` is given."""
    t0 = datetime.fromisoformat(f"{local_date}T{start_hhmm}:00+00:00")
    end = t0 + timedelta(seconds=seconds)
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('running', ?, ?, ?, ?, 'test', ?)",
        (t0.isoformat(), end.isoformat(), local_date, seconds / 60.0,
         f"workout-{local_date}-{start_hhmm}"),
    )
    per_second_mi = 1.0 / (pace_min_per_mi * 60.0)
    for i in range(seconds):
        ts = (t0 + timedelta(seconds=i)).isoformat()
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES ('distance_walking_running', ?, 'mi', ?, ?, ?, ?, 't', 't', ?)",
            (per_second_mi, ts, ts, ts, local_date,
             f"{local_date}-{start_hhmm}-{i}"))
        if hr is not None and i % 5 == 0:
            conn.execute(
                "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
                "start_local, local_date, source, origin, dedupe_key) "
                "VALUES ('heart_rate', ?, 'count/min', ?, ?, ?, ?, 't', 't', ?)",
                (hr, ts, ts, ts, local_date,
                 f"hr-{local_date}-{start_hhmm}-{i}"))
        if cadence_spm and i % 20 == 0:
            conn.execute(
                "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
                "start_local, local_date, source, origin, dedupe_key) "
                "VALUES ('step_count', ?, 'count', ?, ?, ?, ?, 't', 't', ?)",
                (cadence_spm / 3.0, ts, ts, ts, local_date,
                 f"steps-{local_date}-{start_hhmm}-{i}"))
    conn.commit()


@pytest.fixture
def gapdb(conn):
    """Three Mondays: two with jogging, the middle week with nothing at all."""
    _emit(conn, "2026-06-01", "12:00", 600, 12.0)     # week of 06-01: 10 min
    #      week of 06-08: no samples whatsoever
    _emit(conn, "2026-06-15", "12:00", 600, 12.0)     # week of 06-15: 10 min


def test_a_week_with_no_samples_is_a_zero_row_not_a_missing_one(gapdb, tools):
    out = tools.get_impact_volume("2026-06-01", "2026-06-21", by="week")
    starts = [p["period_start"] for p in out["periods"]]
    assert starts == ["2026-06-01", "2026-06-08", "2026-06-15"]
    gap = out["periods"][1]
    assert gap["jog_minutes"] == 0.0
    assert gap["no_data"] is True


def test_the_week_after_a_gap_does_not_compare_across_it(gapdb, tools):
    """It used to read '0% change vs the previous period' against a week two
    weeks back, which is not the previous period."""
    out = tools.get_impact_volume("2026-06-01", "2026-06-21", by="week")
    after = out["periods"][2]
    assert after["jog_change_pct"] is None
    assert "jog_change_note" in after


@pytest.fixture
def partialdb(conn):
    """A whole week, then two days of the following one — the live shape."""
    monday = date(2026, 6, 1)
    for i in range(7):
        _emit(conn, (monday + timedelta(days=i)).isoformat(), "12:00", 600, 12.0)
    for i in range(2):
        _emit(conn, (monday + timedelta(days=7 + i)).isoformat(), "12:00", 600, 12.0)


def test_a_partial_week_is_labelled_and_not_given_a_change_pct(partialdb, tools):
    out = tools.get_impact_volume("2026-06-01", "2026-06-09", by="week")
    full, part = out["periods"]
    assert full["partial"] is False
    assert full["days_covered"] == 7
    assert part["partial"] is True
    assert part["days_covered"] == 2
    # 2 days of the same daily volume against 7 is a 71% "drop" that never
    # happened; the tool must not state it as a change.
    assert part["jog_change_pct"] is None
    assert "2" in part["jog_change_note"] and "7" in part["jog_change_note"]


def test_a_partial_week_still_reports_its_minutes(partialdb, tools):
    part = tools.get_impact_volume("2026-06-01", "2026-06-09", by="week")["periods"][1]
    assert part["jog_minutes"] == pytest.approx(20.0, abs=0.5)


def test_full_week_change_pct_is_unchanged(conn, tools):
    _emit(conn, "2026-06-01", "12:00", 600, 12.0)     # 10 min
    _emit(conn, "2026-06-08", "12:00", 1200, 12.0)    # 20 min
    out = tools.get_impact_volume("2026-06-01", "2026-06-14", by="week")
    assert out["periods"][1]["jog_change_pct"] == pytest.approx(100.0, abs=1.0)
    assert out["periods"][1]["partial"] is False


def test_tool_periods_still_match_analysis_exactly(partialdb, vault_path, tools):
    """The tool wraps analysis.impact_volume; it must not restate its numbers."""
    from health_advisor import db as dbmod
    c = dbmod.connect(vault_path, read_only=True)
    try:
        rows = A.impact_volume(c, "2026-06-01", "2026-06-09", by="week")
    finally:
        c.close()
    got = tools.get_impact_volume("2026-06-01", "2026-06-09", by="week")["periods"]
    for a, b in zip(rows, got):
        for key in ("jog_minutes", "jog_miles", "jog_pace_min_per_mi",
                    "walk_minutes", "walk_miles"):
            assert a[key] == b[key], key


def test_four_week_blocks_pin_complete_and_last_day_anchors(blockdb, tools):
    """The same explicit range must preserve both valid readings of the window."""
    complete = tools.get_impact_volume(
        "2026-06-15", "2026-08-21", by="week", weeks_per_block=4,
        anchor="last_complete_week")
    last_day = tools.get_impact_volume(
        "2026-06-15", "2026-08-21", by="week", weeks_per_block=4,
        anchor="last_day_with_data")

    c = complete["block_comparison"]
    assert c["blocks"]["recent"]["period_starts"] == [
        "2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10"]
    assert c["blocks"]["prior"]["period_starts"] == [
        "2026-06-22", "2026-06-29", "2026-07-06", "2026-07-13"]
    assert c["blocks"]["recent"]["total"] == pytest.approx(200.1)
    assert c["blocks"]["recent"]["mean"] == pytest.approx(50.0)
    assert c["blocks"]["prior"]["total"] == pytest.approx(150.3)
    assert c["blocks"]["prior"]["mean"] == pytest.approx(37.6)
    assert c["change"]["total_delta"] == round(
        c["blocks"]["recent"]["total"] - c["blocks"]["prior"]["total"], 1)
    assert c["change"]["total_delta_pct"] == round(
        c["change"]["total_delta"] / c["blocks"]["prior"]["total"] * 100, 1)
    assert c["change"]["total_delta_pct"] == 33.1
    assert c["change"]["total_delta_pct"] is not None
    assert "mean_delta_pct" not in c["change"]

    d = last_day["block_comparison"]
    assert d["blocks"]["recent"]["period_starts"] == [
        "2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17"]
    assert d["blocks"]["prior"]["period_starts"] == [
        "2026-06-29", "2026-07-06", "2026-07-13", "2026-07-20"]
    assert d["blocks"]["recent"]["total"] == pytest.approx(200.4)
    assert d["blocks"]["recent"]["mean"] == pytest.approx(50.1)
    assert d["blocks"]["prior"]["total"] == pytest.approx(201.0)
    assert d["blocks"]["prior"]["mean"] == pytest.approx(50.2)
    assert d["change"]["mean_delta"] == round(
        d["blocks"]["recent"]["mean"] - d["blocks"]["prior"]["mean"], 1)
    assert d["change"]["total_delta_pct"] is None
    assert "not resolvable" in d["change"]["total_delta_pct_note"]


def test_block_output_names_anchor_and_partial_completeness(blockdb, tools):
    complete = tools.get_impact_volume(
        "2026-06-15", "2026-08-21", by="week", weeks_per_block=4,
        anchor="last_complete_week")["block_comparison"]
    last_day = tools.get_impact_volume(
        "2026-06-15", "2026-08-21", by="week", weeks_per_block=4,
        anchor="last_day_with_data")["block_comparison"]

    dropped = complete["completeness"]
    assert complete["anchor"] == "last_complete_week"
    assert complete["anchor_end"] == "2026-08-16"
    assert dropped["end_default"] is False
    assert "date.today()" in dropped["rule"]
    partial = dropped["partial_trailing_week"]
    assert partial["period_start"] == "2026-08-17"
    assert partial["days_covered"] == 5
    assert partial["days_expected"] == 7
    assert partial["partial"] is True
    assert partial["included"] is False
    assert "dropped" in partial["reason"]

    included = last_day["completeness"]["partial_trailing_week"]
    assert last_day["anchor"] == "last_day_with_data"
    assert last_day["anchor_end"] == "2026-08-21"
    assert included["period_start"] == "2026-08-17"
    assert included["days_covered"] == 5
    assert included["days_expected"] == 7
    assert included["partial"] is True
    assert included["included"] is True
    assert "included" in included["reason"]


def test_block_output_exposes_each_value_used_by_the_mean(blockdb, tools):
    out = tools.get_impact_volume(
        "2026-06-15", "2026-08-21", by="week", weeks_per_block=4,
        anchor="last_complete_week")["block_comparison"]
    recent = out["blocks"]["recent"]
    assert [(w["metric"], w["period"], w["field"], w["value"])
            for w in recent["weeks"]] == [
        ("jog_minutes", "2026-07-20", "jog_minutes", 68.0),
        ("jog_minutes", "2026-07-27", "jog_minutes", 46.7),
        ("jog_minutes", "2026-08-03", "jog_minutes", 32.7),
        ("jog_minutes", "2026-08-10", "jog_minutes", 52.7),
    ]
    assert recent["mean"] == pytest.approx(
        sum(w["value"] for w in recent["weeks"]) / 4, abs=0.05)


# --------------------------------------------------------------------------- #
# threshold sensitivity
# --------------------------------------------------------------------------- #
@pytest.fixture
def cliffdb(conn):
    """10 min at 12 min/mi (comfortably jogging) and 10 min at 15.5 min/mi —
    inside the cutoff, but one small slowdown from falling out of it."""
    _emit(conn, "2026-06-01", "12:00", 600, 12.0)
    _emit(conn, "2026-06-01", "12:10", 600, 15.5)


def test_sensitivity_reports_jog_minutes_at_neighbouring_cutoffs(cliffdb, tools):
    out = tools.get_impact_volume("2026-06-01", "2026-06-07", by="week")
    sens = out["jog_threshold_sensitivity"]
    at = {s["cadence_min_steps_per_min"]: s["jog_minutes"] for s in sens}
    assert 140.0 in at and 139.0 in at and 150.0 in at
    # Both buckets carry 141 steps/min: lowering the edge to 139 keeps both;
    # raising it above the observed cadence drops both.
    assert at[140.0] == pytest.approx(20.0, abs=0.5)
    assert at[150.0] == pytest.approx(0.0, abs=0.5)


def test_sensitivity_at_the_live_cutoff_equals_the_reported_minutes(cliffdb, tools):
    """If these ever disagree the sensitivity block is measuring something else
    than the classifier it claims to be probing."""
    out = tools.get_impact_volume("2026-06-01", "2026-06-07", by="week")
    total = sum(p["jog_minutes"] for p in out["periods"])
    at_live = next(s["jog_minutes"] for s in out["jog_threshold_sensitivity"]
                   if s["cadence_min_steps_per_min"] == A.IMPACT_JOG_CADENCE_MIN)
    assert at_live == pytest.approx(total, abs=0.35)


@pytest.fixture
def artifact_cliffdb(conn):
    """A classifier-excluded GPS artifact alongside two ordinary jog lanes."""
    _emit(conn, "2026-06-01", "12:00", 600, 12.0)
    _emit(conn, "2026-06-01", "12:10", 600, 15.5)
    _emit(conn, "2026-06-01", "12:20", 600, 2.0)  # impossible-speed artifact


def test_sensitivity_live_cutoff_uses_the_same_bucket_set(artifact_cliffdb, tools):
    """The live sensitivity row must equal the headline classifier exactly."""
    out = tools.get_impact_volume("2026-06-01", "2026-06-07", by="week")
    total = sum(p["jog_minutes"] for p in out["periods"])
    at_live = next(s["jog_minutes"] for s in out["jog_threshold_sensitivity"]
                   if s["cadence_min_steps_per_min"] == A.IMPACT_JOG_CADENCE_MIN)
    assert at_live == total


@pytest.fixture
def hrcliffdb(conn):
    """10 min at 12 min/mi, and 10 min at 18 min/mi with a running heart rate —
    the Week 5 shape: past the pace cutoff, but jogging on the HR evidence."""
    _emit(conn, "2026-06-01", "12:00", 600, 12.0, hr=145)
    _emit(conn, "2026-06-01", "12:10", 600, 18.0, hr=140)


def test_sensitivity_at_the_live_cutoff_ignores_heart_rate(hrcliffdb, tools):
    """Heart rate is not a second promotion lane for impact volume."""
    out = tools.get_impact_volume("2026-06-01", "2026-06-07", by="week")
    total = sum(p["jog_minutes"] for p in out["periods"])
    at_live = next(s["jog_minutes"] for s in out["jog_threshold_sensitivity"]
                   if s["cadence_min_steps_per_min"] == A.IMPACT_JOG_CADENCE_MIN)
    assert total == pytest.approx(20.0, abs=0.7)      # both have running cadence
    assert at_live == pytest.approx(total, abs=0.35)


def test_sensitivity_reports_the_share_sitting_near_the_cliff(cliffdb, tools):
    out = tools.get_impact_volume("2026-06-01", "2026-06-07", by="week")
    near = out["jog_near_threshold"]
    assert near["within_steps_per_min"] == 10.0
    assert near["pct_of_jog_buckets"] == pytest.approx(0.0, abs=2.0)
    assert "note" in near
