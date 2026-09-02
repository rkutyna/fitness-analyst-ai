from health_advisor import analysis as A
from tests.conftest import seed_metric


def test_movers_flags_big_change_only(conn):
    # step_count jumps ~+50% recently; active_energy flat
    seed_metric(conn, "step_count", "2026-05-01", [6000] * 23 + [9000] * 7)
    seed_metric(conn, "active_energy", "2026-05-01", [500] * 30)
    movers = A.movers(conn, as_of="2026-05-30", scope="deep")
    metrics = {m["metric"] for m in movers}
    assert "step_count" in metrics
    assert "active_energy" not in metrics  # below threshold


def test_movers_excludes_newly_resumed_metric(conn):
    # Only ~6 days of data (e.g. a freshly-paired watch) -> not enough window.
    seed_metric(conn, "blood_oxygen_saturation", "2026-05-25", [95, 96, 97, 96, 95, 97])
    movers = A.movers(conn, as_of="2026-05-30", scope="deep")
    assert "blood_oxygen_saturation" not in {m["metric"] for m in movers}


def test_movers_skips_artifact_scale_swings(conn):
    # Near-zero baseline then a jump -> explosive % is an artifact, not a finding.
    seed_metric(conn, "flights_climbed", "2026-05-01", [0.01] * 23 + [50] * 7)
    movers = A.movers(conn, as_of="2026-05-30", scope="deep")
    assert "flights_climbed" not in {m["metric"] for m in movers}


def test_movers_respects_topk_daily(conn):
    for i, m in enumerate(["step_count", "active_energy", "flights_climbed",
                           "distance_walking_running"]):
        seed_metric(conn, m, "2026-05-01", [100] * 23 + [300] * 7)  # all +200%
    movers = A.movers(conn, as_of="2026-05-30", scope="daily")
    assert len(movers) <= 3


def test_movers_skips_metric_without_enough_worn_history(conn):
    # 7 dense worn days over 21 coarse non-wear days: not enough consistent-wear
    # history to trust a recent-vs-baseline comparison (the backfill→live-sync
    # case). The metric must be skipped rather than reported as a huge mover.
    seed_metric(conn, "active_energy", "2026-05-22",
                [120] * 21 + [500] * 7,
                counts=[15] * 21 + [40000] * 7)
    movers = A.movers(conn, as_of="2026-06-18", scope="deep")
    assert "active_energy" not in {m["metric"] for m in movers}


def test_movers_baseline_ignores_unworn_days(conn):
    # Truly flat on worn days (500), but the first 7 days are non-wear with an
    # artifactually low value. Including them would fabricate a ~+43% "rise";
    # excluding them (wear gate) leaves a flat metric -> not a mover.
    seed_metric(conn, "active_energy", "2026-05-22",
                [50] * 7 + [500] * 21,
                counts=[12] * 7 + [40000] * 21)
    movers = A.movers(conn, as_of="2026-06-18", scope="deep")
    assert "active_energy" not in {m["metric"] for m in movers}


# --- effect size, not raw % (audit P2-4) ------------------------------------ #

_NOISY_BASE = [2, 6] * 10 + [2]          # 21 days, mean ~3.9, sd ~2.0


def test_movers_ignores_a_swing_inside_the_metric_own_noise(conn):
    # The live #1 mover: walking_asymmetry_percentage "UP 85.9%" at t≈1.4, read
    # by the narrator as an injury signal. A metric that swings 2-6 every day
    # sitting at 6 for a week is not news, whatever the percentage says.
    seed_metric(conn, "walking_asymmetry_percentage", "2026-05-03",
                _NOISY_BASE + [6] * 7)
    movers = A.movers(conn, as_of="2026-05-30", scope="deep")
    assert "walking_asymmetry_percentage" not in {m["metric"] for m in movers}


def test_movers_still_reports_a_change_that_clears_the_noise(conn):
    # Same day-to-day variance, a shift that is genuinely outside it.
    seed_metric(conn, "walking_asymmetry_percentage", "2026-05-03",
                _NOISY_BASE + [10] * 7)
    movers = A.movers(conn, as_of="2026-05-30", scope="deep")
    hit = next(m for m in movers if m["metric"] == "walking_asymmetry_percentage")
    assert hit["effect_sd"] >= 1.5


def test_movers_rank_by_effect_not_percentage(conn):
    # step_count moves less in % terms but far more relative to its own spread.
    seed_metric(conn, "flights_climbed", "2026-05-03",
                [60, 140] * 10 + [60] + [170] * 7)        # +73%, noisy
    seed_metric(conn, "step_count", "2026-05-03",
                [98, 102] * 10 + [98] + [130] * 7)        # +30%, steady
    movers = A.movers(conn, as_of="2026-05-30", scope="deep")
    assert [m["metric"] for m in movers][0] == "step_count"
    assert abs(movers[0]["pct"]) < abs(movers[1]["pct"])   # ranked despite less %


def test_movers_clamp_the_biggest_movers_instead_of_dropping_them(conn):
    # A metric that genuinely multiplied was excluded by MOVER_MAX_PCT, so the
    # ceiling silently hid the largest real changes.
    seed_metric(conn, "time_in_daylight", "2026-05-03",
                [9, 11] * 10 + [9] + [700] * 7)           # ~+6900%
    movers = A.movers(conn, as_of="2026-05-30", scope="deep")
    hit = next(m for m in movers if m["metric"] == "time_in_daylight")
    assert hit["pct"] == A.MOVER_MAX_PCT
    assert hit["pct_capped"] is True
    assert hit["pct_uncapped"] > A.MOVER_MAX_PCT


def test_a_low_activity_day_is_not_a_non_wear_day(conn):
    # flights_climbed writes a record per flight, so its sample count tracks how
    # much was climbed, not whether the watch was on. Judging wear by density
    # therefore deletes exactly the quiet days a downward mover is made of: live,
    # it removed flights_climbed, time_in_daylight and walking_asymmetry_
    # percentage from the briefing altogether. wear_hours says all 28 days were
    # worn, and it outranks the proxy.
    vals = [60, 140] * 10 + [60] + [8] * 7
    seed_metric(conn, "flights_climbed", "2026-05-03", vals,
                counts=[int(v) for v in vals])
    seed_metric(conn, "wear_hours", "2026-05-03", [24.0] * 28)
    movers = A.movers(conn, as_of="2026-05-30", scope="deep")
    assert "flights_climbed" in {m["metric"] for m in movers}


def test_movers_ignores_days_whose_value_column_is_null(conn):
    # Live: the two most recent daily_metrics rows carry a NULL `last` for every
    # metric, and resting_heart_rate is a last-valued metric. _daily_load_rows
    # read the value column raw — unlike mx.series, which filters NULLs — so
    # movers() raised TypeError on the production DB and took the whole briefing
    # down with it. A NULL day is an absent day, not a zero.
    seed_metric(conn, "resting_heart_rate", "2026-05-01", [60] * 23 + [75] * 7)
    conn.execute("UPDATE daily_metrics SET last = NULL WHERE date >= '2026-05-29'")
    conn.commit()
    A.movers(conn, as_of="2026-05-30", scope="deep")   # must not raise


def test_highlights_returns_list(conn):
    seed_metric(conn, "step_count", "2026-05-01", list(range(1000, 1030)))
    hl = A.highlights(conn, as_of="2026-05-29")
    assert isinstance(hl, list)
