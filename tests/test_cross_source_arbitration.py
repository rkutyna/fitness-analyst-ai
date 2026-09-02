"""Cross-source arbitration for cumulative metrics (audit P0-2).

Apple Health resolves overlapping cumulative samples by source before showing a
daily total; this pipeline summed everything it was given, so a day written by
two sources counted the same movement twice. Two distinct shapes of that bug
exist in the archive and are resolved separately:

* a third-party app mirroring a whole-day total into HealthKit alongside the
  Apple devices' own samples (Sync Solver, 2017-2020), and
* a single sample carrying a whole-day estimate inside an otherwise normal
  stream (RENPHO's BMR blob at weigh-in, 2026) — which HAE sometimes labels
  with the watch's own name, so it cannot be resolved by source alone.
"""
from __future__ import annotations

import pytest

from health_advisor import db as dbmod

WATCH = "Demo’s Apple Watch"
MERGED = "Demo’s Apple Watch|RENPHO Health"
MIRROR = "Sync Solver"


def _add(conn, metric, day, source, values, unit="kcal", start_hour=8):
    """Insert one record per value, a second apart, all on `day`."""
    rows = []
    for i, v in enumerate(values):
        ts = f"{day}T{start_hour:02d}:{i // 60:02d}:{i % 60:02d}+00:00"
        rows.append(dict(
            metric=metric, value=float(v), unit=unit, start_utc=ts, end_utc=ts,
            start_local=f"{day} {start_hour:02d}:00:00", local_date=day,
            source=source, origin="receiver",
            dedupe_key=dbmod.record_key(metric, ts, ts, v, unit, source),
        ))
    dbmod.insert_records(conn, rows)


def _daily(conn, metric, day):
    dbmod.recompute_daily_metrics(conn, pairs=[(metric, day)])
    return conn.execute(
        "SELECT count, sum FROM daily_metrics WHERE metric = ? AND date = ?",
        (metric, day),
    ).fetchone()


# --------------------------------------------------------------------------- #
# A whole-day estimate dropped into a live stream.
# --------------------------------------------------------------------------- #
def test_renpho_whole_day_blob_is_excluded_from_the_daily_sum(conn):
    _add(conn, "basal_energy", "2026-07-09", WATCH, [0.02] * 100)
    _add(conn, "basal_energy", "2026-07-09", "RENPHO Health", [371.8] * 6, start_hour=15)

    # approx, not ==: this is an accumulated float sum (100 x 0.02). It lands
    # on exactly 2.0 on arm64 and on 2.0000000000000013 on x86_64, so an
    # equality assertion here passes by luck of the platform. What the test
    # is actually about is WHICH rows survived arbitration, not the last bit
    # of the sum.
    assert _daily(conn, "basal_energy", "2026-07-09")["sum"] == pytest.approx(2.0)


def test_blob_is_excluded_even_when_labelled_with_the_watch(conn):
    # HAE merges source names, so the blob and real watch samples share a label.
    _add(conn, "basal_energy", "2026-07-21", WATCH, [0.02] * 100)
    _add(conn, "basal_energy", "2026-07-21", MERGED, [365.0] * 5, start_hour=12)
    _add(conn, "basal_energy", "2026-07-21", MERGED, [0.02] * 50, start_hour=17)

    row = _daily(conn, "basal_energy", "2026-07-21")
    # approx for the same reason as above; `count` below is the assertion
    # that carries the meaning — 150 stream samples kept, 5 blobs dropped.
    assert row["sum"] == pytest.approx(3.0)
    assert row["count"] == 150


def test_a_whole_day_estimate_is_kept_when_it_is_the_only_data(conn):
    # Nothing to replace it with: a coarse number beats no number.
    _add(conn, "basal_energy", "2026-07-09", "RENPHO Health", [371.8, 371.8], start_hour=15)

    assert _daily(conn, "basal_energy", "2026-07-09")["sum"] == 743.6


# --------------------------------------------------------------------------- #
# A third-party app mirroring whole-day totals (Sync Solver).
# --------------------------------------------------------------------------- #
def test_apple_wins_over_the_mirror_once_the_watch_exists(conn):
    _add(conn, "step_count", "2019-03-10", MIRROR, [616.0], unit="count")
    _add(conn, "step_count", "2019-03-10", "Demo’s Iphone", [300.0, 329.0], unit="count")

    assert _daily(conn, "step_count", "2019-03-10")["sum"] == 629.0


def test_the_mirror_wins_before_the_watch_exists(conn):
    # Phone-only counts badly undercount the Fitbit era.
    _add(conn, "step_count", "2017-06-01", MIRROR, [14656.0], unit="count")
    _add(conn, "step_count", "2017-06-01", "Demos iphone", [4000.0, 4715.0], unit="count")

    assert _daily(conn, "step_count", "2017-06-01")["sum"] == 14656.0


def test_the_mirror_is_kept_when_no_apple_source_wrote_that_day(conn):
    _add(conn, "basal_energy", "2018-04-02", MIRROR, [1691.0])

    assert _daily(conn, "basal_energy", "2018-04-02")["sum"] == 1691.0


# --------------------------------------------------------------------------- #
# Scope: only cumulative metrics, and both recompute paths agree.
# --------------------------------------------------------------------------- #
def test_instantaneous_metrics_are_left_alone(conn):
    # Two devices reading heart rate is two observations of one truth: the mean
    # over both is right, and arbitration would throw half the samples away.
    _add(conn, "heart_rate", "2026-07-21", WATCH, [60.0, 62.0], unit="count/min")
    _add(conn, "heart_rate", "2026-07-21", MERGED, [400.0], unit="count/min",
         start_hour=12)

    assert _daily(conn, "heart_rate", "2026-07-21")["count"] == 3


def test_full_rebuild_agrees_with_the_incremental_path(conn):
    _add(conn, "basal_energy", "2026-07-09", WATCH, [0.02] * 100)
    _add(conn, "basal_energy", "2026-07-09", "RENPHO Health", [371.8] * 6, start_hour=15)
    _add(conn, "step_count", "2017-06-01", MIRROR, [14656.0], unit="count")
    _add(conn, "step_count", "2017-06-01", "Demos iphone", [4000.0], unit="count")

    incremental = {
        (m, d): _daily(conn, m, d)["sum"]
        for m, d in [("basal_energy", "2026-07-09"), ("step_count", "2017-06-01")]
    }
    dbmod.recompute_daily_metrics(conn, full=True)
    rebuilt = {
        (r["metric"], r["date"]): r["sum"]
        for r in conn.execute("SELECT metric, date, sum FROM daily_metrics")
    }

    assert rebuilt == incremental
