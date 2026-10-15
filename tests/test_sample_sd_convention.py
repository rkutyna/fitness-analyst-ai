"""One standard-deviation convention for published spread figures (#94, C1).

Every published ``std``/``sd`` is the SAMPLE standard deviation (ddof 1): the
days a figure is computed over are a sample of the athlete's days, not the
whole population of them. The one deliberate exception is the movers effect
size, whose denominator and its published ``baseline_sd`` stay population
(ddof 0) by decision, so ``effect_sd`` remains ``diff / baseline_sd``.
"""
from __future__ import annotations

import statistics

import pytest

from health_advisor import analysis as A
from health_advisor import metrics as mx
from tests.conftest import seed_metric

VALUES = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]   # pstdev 2.0, stdev 2.14
DATES = [f"2026-06-0{i + 1}" for i in range(len(VALUES))]


def test_stats_publishes_the_sample_standard_deviation():
    st = mx.stats(DATES, VALUES)
    assert st["std"] == round(statistics.stdev(VALUES), 2) == 2.14
    assert st["std"] != round(statistics.pstdev(VALUES), 2)


def test_stats_of_one_day_has_no_standard_deviation():
    # A sample of one has no spread to estimate; 0.0 would claim a flat series.
    st = mx.stats(DATES[:1], VALUES[:1])
    assert st["std"] is None
    assert st["mean"] == VALUES[0]


def test_block_sd_and_stats_std_agree_on_the_same_days(conn):
    # Two Monday weeks of the same eight-value pattern; the block sd and the
    # stats std must be the same statistic over the same values.
    vals = VALUES[:7] * 2
    seed_metric(conn, "resting_heart_rate", "2026-06-01", vals)
    out = A.block_comparison(conn, "resting_heart_rate", 1, "2026-06-14")
    recent = out["blocks"]["recent"]
    dates, series, _ = mx.series(conn, "resting_heart_rate",
                                 *recent["period"].split(":"))
    assert recent["sd"] == mx.stats(dates, series)["std"]


def test_the_movers_baseline_sd_stays_population(conn):
    seeded = [6000 + 300 * (i % 5) for i in range(23)] + [9000] * 7
    seed_metric(conn, "step_count", "2026-05-01", seeded)
    hit = next(m for m in A.movers(conn, as_of="2026-05-30", scope="deep")
               if m["metric"] == "step_count")
    # The movers window is the last MOVER_WINDOW_DAYS days; its final week is
    # the recent mean and the days before it are the baseline.
    window = seeded[-A.MOVER_WINDOW_DAYS:]
    base_vals = window[:-7]
    assert hit["baseline_sd"] == pytest.approx(
        statistics.pstdev(base_vals), abs=0.01)
    assert hit["baseline_sd"] != pytest.approx(
        statistics.stdev(base_vals), abs=0.01)
