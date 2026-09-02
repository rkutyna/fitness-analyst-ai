from datetime import date, timedelta

import pytest

from health_advisor import history as H
from health_advisor import analysis as A
from tests.conftest import seed_metric


def _days(start, count):
    first = date.fromisoformat(start)
    return [(first + timedelta(days=i)).isoformat() for i in range(count)]


def _seed_sources(conn, metric, rows):
    for i, (day, source) in enumerate(rows):
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (metric, 1.0, "count", f"{day}T00:00:00Z", f"{day}T00:01:00Z",
             f"{day}T00:00:00", day, source, "test", f"source-{i}"),
        )
    conn.commit()


def _month(month: str, source: str, n: int = 8) -> list[tuple[str, str]]:
    """n days of one source in one month — enough to clear ERA_MIN_MONTH_SAMPLES."""
    return [(f"{month}-{d:02d}", source) for d in range(1, n + 1)]


def test_instrument_eras_reports_monthly_dominant_source_changes(conn):
    _seed_sources(conn, "step_count",
                  _month("2020-01", "watch") + _month("2020-02", "phone")
                  + _month("2020-03", "phone"))

    assert A.instrument_eras(
        conn, "step_count", "2020-01-01", "2020-03-31"
    ) == ["2020-02-01"]


def test_detect_eras_splits_continuous_series_at_instrument_change(conn):
    # The series never gaps — 90 consecutive days — and still changes
    # instrument underneath. That is the case the >14-day gap rule cannot see.
    seed_metric(conn, "step_count", "2020-01-01", list(range(90)))
    _seed_sources(conn, "step_count",
                  _month("2020-01", "watch") + _month("2020-02", "phone")
                  + _month("2020-03", "phone"))

    eras = H.detect_eras(conn, "step_count")

    assert [(e["start"], e["end"]) for e in eras] == [
        ("2020-01-01", "2020-01-31"),
        ("2020-02-01", "2020-03-30"),
    ]


def test_detect_eras_splits_long_gap_but_not_missed_week(conn):
    seed_metric(conn, "resting_heart_rate", "2020-01-01", [55] * 10)
    seed_metric(conn, "resting_heart_rate", "2020-01-26", [56] * 10)
    seed_metric(conn, "step_count", "2020-01-01", [8000] * 27)

    eras = H.detect_eras(conn, "resting_heart_rate")
    assert [(e["start"], e["end"], e["n"]) for e in eras] == [
        ("2020-01-01", "2020-01-10", 10),
        ("2020-01-26", "2020-02-04", 10),
    ]
    assert eras[0]["coverage_density"] == 1.0
    assert len(H.detect_eras(conn, "step_count")) == 1


def test_reference_ranges_are_provenanced_and_direction_aware(conn):
    values = list(range(50, 80))
    seed_metric(conn, "resting_heart_rate", "2020-01-01", values)

    out = H.reference_ranges(conn, "resting_heart_rate", direction="lower")
    era = out["eras"][0]
    assert era["central"]["value"] == 64.5
    assert era["floor"]["value"] < era["ceiling"]["value"]
    assert era["best_sustained"]["value"] == 64.5
    for item in (era["central"], era["floor"], era["ceiling"],
                 era["best_sustained"]):
        assert item["era"] == 1
        assert item["n"] > 0
        assert item["start"] <= item["end"]

    with pytest.raises(ValueError, match="direction"):
        H.reference_ranges(conn, "resting_heart_rate")


def test_tiny_era_does_not_turn_one_sample_into_a_range(conn):
    seed_metric(conn, "vo2_max", "2020-01-01", [43.5])
    era = H.reference_ranges(conn, "vo2_max", direction="higher")["eras"][0]
    assert era["central"]["value"] == 43.5
    assert era["floor"] is None
    assert era["ceiling"] is None


def test_trajectory_marks_sparse_bucket_without_emitting_its_low_value(conn):
    seed_metric(conn, "step_count", "2020-01-01", [1000] * 10)
    seed_metric(conn, "step_count", "2020-02-01", [9000] * 60)

    out = H.trajectory(conn, "step_count", bucket="month")
    by_bucket = {p["bucket_start"]: p for p in out["points"]}
    assert by_bucket["2020-01-01"]["value"] is None
    assert by_bucket["2020-01-01"]["coverage_density"] < 0.7
    assert by_bucket["2020-02-01"]["value"] == 261000.0
    assert by_bucket["2020-02-01"]["coverage_density"] == 1.0


def test_sustained_periods_return_only_numbers_and_provenance(conn):
    values = [8000] * 30 + [2000] * 5 + [9000] * 30
    seed_metric(conn, "step_count", "2020-01-01", values)

    out = H.sustained_periods(
        conn, "step_count", threshold=7000, direction="higher", window_days=30,
        min_days=24, min_fraction=0.8,
    )
    assert out["periods"]
    period = out["periods"][0]
    assert period["start"] == "2020-01-01"
    assert period["typical"]["value"] == 8000.0
    assert period["typical"]["n"] >= 24
    assert period["era"] == 1
    assert "success" not in str(out).lower()


def test_a_single_sample_month_cannot_declare_an_instrument_change(conn):
    # Measured on the live DB before this guard existed: body_mass produced
    # EIGHT eras, four of them one reading, because a month with a single
    # weigh-in lets one sample decide which source is "dominant". history.py
    # refuses to average across a boundary, so a spurious boundary silently
    # destroys a real series — a worse error than the one being fixed.
    _rec(conn, "body_mass", "2026-01", "Scale A", 30)
    _rec(conn, "body_mass", "2026-02", "Scale B", 1)     # one stray reading
    _rec(conn, "body_mass", "2026-03", "Scale A", 30)
    assert A.instrument_eras(conn, "body_mass", "2026-01-01", "2026-03-31") == []


def test_a_change_that_sticks_is_an_instrument_change(conn):
    # The real one: the watch stopped recording steps in July 2022 and the
    # phone took over, with no gap in the daily series at all.
    _rec(conn, "step_count", "2022-05", "Watch", 40)
    _rec(conn, "step_count", "2022-06", "Watch", 40)
    _rec(conn, "step_count", "2022-07", "iPhone", 40)
    _rec(conn, "step_count", "2022-08", "iPhone", 40)
    assert A.instrument_eras(conn, "step_count", "2022-05-01", "2022-08-31") \
        == ["2022-07-01"]


def test_a_one_month_blip_is_not_an_instrument_change(conn):
    # A loaner, a trip, a dead battery. The source came back.
    _rec(conn, "step_count", "2023-01", "Watch", 40)
    _rec(conn, "step_count", "2023-02", "iPhone", 40)
    _rec(conn, "step_count", "2023-03", "Watch", 40)
    assert A.instrument_eras(conn, "step_count", "2023-01-01", "2023-03-31") == []


def _rec(conn, metric: str, month: str, source: str, n: int) -> None:
    """n records for one metric/month/source, one per day from the 1st."""
    conn.executemany(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "local_date, source, origin, dedupe_key) VALUES (?, 1, 'count', ?, ?, "
        "?, ?, 'test', ?)",
        [(metric, f"{month}-{d:02d}T12:00:00Z", f"{month}-{d:02d}T12:00:00Z",
          f"{month}-{d:02d}", source, f"{metric}|{month}|{source}|{d}")
         for d in range(1, n + 1)])
    conn.commit()
