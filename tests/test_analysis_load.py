from datetime import date, timedelta

from health_advisor import analysis as A
from tests.conftest import seed_metric


def test_acwr_sweet_spot_when_steady(conn):
    seed_metric(conn, "active_energy", "2026-05-01", [500] * 30)  # steady load
    tl = A.training_load(conn, as_of="2026-05-30")
    assert tl["status"] == "ok"
    assert tl["acwr_band"] == "sweet-spot"
    assert abs(tl["acwr"] - 1.0) < 0.05


def test_acwr_ramping_when_recent_spike(conn):
    seed_metric(conn, "active_energy", "2026-05-01", [200] * 23 + [900] * 7)  # acute spike
    tl = A.training_load(conn, as_of="2026-05-30")
    assert tl["acwr"] > 1.5
    assert tl["acwr_band"] == "ramping-fast"


def test_acwr_insufficient_history(conn):
    seed_metric(conn, "active_energy", "2026-06-01", [500] * 10)
    tl = A.training_load(conn, as_of="2026-06-10")
    assert tl["status"] == "insufficient_history"


def test_acwr_insufficient_wear_when_chronic_window_mostly_unworn(conn):
    # 28 days present, but only the last 8 were densely sampled (watch worn).
    # The first 20 are coarse backfill-style days (few samples). ACWR must not
    # be reported off a baseline of non-wear days.
    seed_metric(conn, "active_energy", "2026-05-22",
                [120] * 20 + [500] * 8,
                counts=[15] * 20 + [40000] * 8)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["status"] == "insufficient_wear"
    assert tl["acwr"] is None
    assert tl["n_worn_days"] == 8


def test_acwr_excludes_unworn_days_from_chronic_baseline(conn):
    # 7 coarse non-wear days then 21 dense worn days, all at the same true
    # load (500). The non-wear days carry an artifactually low value (50) and
    # must be excluded so they don't deflate the chronic baseline and inflate
    # the ratio. With them excluded, acute == chronic -> ACWR ~ 1.0.
    seed_metric(conn, "active_energy", "2026-05-22",
                [50] * 7 + [500] * 21,
                counts=[12] * 7 + [40000] * 21)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["status"] == "ok"
    assert abs(tl["acwr"] - 1.0) < 0.05  # not ~1.29 (which including the 50s gives)


def test_no_deload_nudge_when_wear_history_insufficient(conn):
    # The live bug: a dense recent week over a sparse backfill baseline fired a
    # "ramped quickly — take a lighter day" nudge every day. With wear gating,
    # ACWR is withheld and that nudge must not appear.
    seed_metric(conn, "active_energy", "2026-05-22",
                [150] * 21 + [500] * 7,
                counts=[15] * 21 + [40000] * 7)
    tl = A.training_load(conn, as_of="2026-06-18")
    sugg = A.suggestions({"status": "establishing_baseline"}, tl)
    assert tl["status"] == "insufficient_wear"
    assert not any("ramped" in s["text"] for s in sugg)


def _seed_sparse(conn, metric, start, values):
    """Seed only the days whose value is not None, leaving the rest genuinely
    absent from daily_metrics — how a workout-scoped metric actually lands."""
    d0 = date.fromisoformat(start)
    for i, v in enumerate(values):
        if v is None:
            continue
        seed_metric(conn, metric, (d0 + timedelta(days=i)).isoformat(), [v],
                    counts=40000)


def _seed_with_gap(conn, start, values, counts, gap_dates):
    """seed_metric, then delete the named days — a metric absent for a day
    (the live 2026-07-17 active_energy gap), not merely sparse."""
    seed_metric(conn, "active_energy", start, values, counts=counts)
    for d in gap_dates:
        conn.execute("DELETE FROM daily_metrics WHERE metric='active_energy' AND date=?", (d,))
    conn.commit()


def test_acwr_survives_a_single_missing_day(conn):
    # The live bug: active_energy was absent for exactly one day (2026-07-17),
    # so a 28-day window held 27 rows and ACWR went dark for a month.
    _seed_with_gap(conn, "2026-05-22", [500] * 28, 40000, ["2026-06-04"])
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["status"] == "ok"
    assert abs(tl["acwr"] - 1.0) < 0.05


def test_acwr_still_withheld_when_history_is_genuinely_short(conn):
    # Relaxing the gate must not let a two-week-old watch report ACWR.
    seed_metric(conn, "active_energy", "2026-06-01", [500] * 14, counts=40000)
    tl = A.training_load(conn, as_of="2026-06-14")
    assert tl["status"] == "insufficient_history"


# --- wear gate, the direction that matters (audit P2-3) --------------------- #

def test_wear_gate_bites_when_the_recent_week_is_the_unworn_one(conn):
    # The live repro: 21 densely-sampled days at 500 kcal, then a week with the
    # watch off logging 30 kcal/day off the phone. Calibrating the threshold
    # from the recent window made it collapse — every one of the 28 days read as
    # worn and the briefing announced "acwr 0.08, detraining", i.e. it turned a
    # week of non-wear into a training finding.
    seed_metric(conn, "active_energy", "2026-05-22",
                [500] * 21 + [30] * 7,
                counts=[40000] * 21 + [15] * 7)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["acwr"] is None
    assert tl.get("acwr_band") is None
    assert tl["status"] == "insufficient_recent"
    assert tl["n_worn_days"] == 21          # not 28


def test_wear_reference_is_measured_over_a_long_history(conn):
    # Watch worn all spring, then off for the whole 28-day window while the
    # phone keeps logging a trickle. A threshold calibrated from the window
    # itself has nothing dense left to compare against, so it declares the
    # non-wear days worn and reports ACWR off them.
    seed_metric(conn, "active_energy", "2026-03-01", [500] * 60, counts=40000)
    seed_metric(conn, "active_energy", "2026-04-30", [40] * 40, counts=12)
    tl = A.training_load(conn, as_of="2026-06-08")
    assert tl["status"] == "insufficient_wear"
    assert tl["n_worn_days"] == 0


def test_wear_hours_catch_a_week_off_the_wrist_that_density_cannot(conn):
    # Sample density is a proxy for wear; wear_hours (distinct local hours with a
    # heart_rate sample) is the thing itself. Here the record granularity never
    # changes — the density gate is blind — but the watch was off all week.
    seed_metric(conn, "active_energy", "2026-05-22", [500] * 21 + [30] * 7,
                counts=40000)
    seed_metric(conn, "wear_hours", "2026-05-22", [24.0] * 21 + [0.5] * 7)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["acwr"] is None
    assert tl["status"] == "insufficient_recent"
    assert tl["n_worn_days"] == 21


def test_acute_window_excludes_non_wear_days_too(conn):
    # Genuinely flat 500 kcal/day, but the watch was off for two days this week.
    # Chronic already ignored them; acute did not, so a flat block of training
    # read as "detraining" (0.73) purely because of two days of non-wear.
    seed_metric(conn, "active_energy", "2026-05-22",
                [500] * 23 + [30, 30] + [500] * 3,
                counts=[40000] * 23 + [15, 15] + [40000] * 3)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["status"] == "ok"
    assert abs(tl["acwr"] - 1.0) < 0.05
    assert tl["acwr_band"] == "sweet-spot"
    assert tl["n_recent_days"] == 5          # the two non-wear days are not load


def test_acwr_withheld_when_the_week_has_too_few_worn_days(conn):
    # Three non-wear days leaves four measured days: not a week of training, and
    # scaling four days to seven would invent the other three.
    seed_metric(conn, "active_energy", "2026-05-22",
                [500] * 22 + [30, 30, 30] + [500] * 3,
                counts=[40000] * 22 + [15, 15, 15] + [40000] * 3)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["acwr"] is None
    assert tl["status"] == "insufficient_recent"


def test_acute_window_is_calendar_bound_not_last_seven_present_rows(conn):
    # A gap inside the acute week must not drag an 8th calendar day into it.
    # Days 1-21 are 100; the last 7 calendar days are 500 with one absent.
    # Acute must reflect the 500s alone, never the 100 from day 22-back.
    _seed_with_gap(conn, "2026-05-22", [100] * 21 + [500] * 7, 40000, ["2026-06-15"])
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["status"] == "ok"
    assert abs(tl["acute_7d"] - 3500) < 1  # 500/day * 7, not 100 blended in


# --- hr_load_proxy as the ACWR input (swapped 2026-08-09) -------------------
#
# hr_load_proxy is workout-scoped: derive.py writes a row only for days that
# had a session with usable HR. A rest day therefore has NO row, and the
# generic "absent means unknown" rule would read a rest week as missing data
# rather than as the zero training load it actually was. Over the 40 days to
# 2026-08-09 that left ACWR computable on 8 of them. Zero-filling worn rest
# days restores it to 40/40.

def test_workout_scoped_load_counts_a_worn_rest_day_as_zero(conn):
    # 28 worn days; sessions on only 14 of them. Without zero-fill only 14 days
    # are present, which is under ACWR_MIN_CHRONIC_DAYS and reports
    # insufficient_history. The rest days are measured zeros, not absences.
    loads = [100.0 if i % 2 == 0 else None for i in range(28)]
    _seed_sparse(conn, "hr_load_proxy", "2026-05-22", loads)
    seed_metric(conn, "wear_hours", "2026-05-22", [24.0] * 28)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["status"] == "ok"
    assert tl["n_days"] == 28
    assert abs(tl["acwr"] - 1.0) < 0.15   # steady alternating pattern


def test_workout_scoped_load_does_not_zero_fill_an_unworn_day(conn):
    # The doctrine hr_load.py states: missing is unknown, not zero. A day the
    # watch was off is unmeasured and must stay absent, or a non-wear stretch
    # reads as a training collapse. Same 14 sessions, but the 14 rest days
    # carry no wear signal at all -> not enough days, ACWR withheld.
    loads = [100.0 if i % 2 == 0 else None for i in range(28)]
    _seed_sparse(conn, "hr_load_proxy", "2026-05-22", loads)
    seed_metric(conn, "wear_hours", "2026-05-22",
                [24.0 if i % 2 == 0 else 2.0 for i in range(28)])
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["acwr"] is None
    assert tl["status"] == "insufficient_history"


def test_workout_scoped_load_reads_a_rest_week_as_detraining(conn):
    # 21 days of daily sessions, then a week off with the watch on. Zero-filled
    # that is a real acute drop; treating the rest week as absent would instead
    # compare the training weeks to themselves and report sweet-spot.
    loads = [100.0] * 21 + [None] * 7
    _seed_sparse(conn, "hr_load_proxy", "2026-05-22", loads)
    seed_metric(conn, "wear_hours", "2026-05-22", [24.0] * 28)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["status"] == "ok"
    assert tl["acwr_band"] == "detraining"


def test_active_energy_absent_days_are_still_not_zero_filled(conn):
    # active_energy is a whole-day total, so an absent day is a dropped sync,
    # not a rest day. Zero-filling it would invent a crash week. Only metrics
    # named as workout-scoped get the zero-fill.
    vals = [500.0 if i < 21 else None for i in range(28)]
    _seed_sparse(conn, "active_energy", "2026-05-22", vals)
    seed_metric(conn, "wear_hours", "2026-05-22", [24.0] * 28)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["acwr"] is None


def test_acwr_prefers_hr_load_proxy_over_active_energy(conn):
    # The swap itself: when both inputs exist, the intensity-aware one wins.
    seed_metric(conn, "active_energy", "2026-05-22", [500.0] * 28)
    seed_metric(conn, "hr_load_proxy", "2026-05-22", [100.0] * 28)
    seed_metric(conn, "wear_hours", "2026-05-22", [24.0] * 28)
    tl = A.training_load(conn, as_of="2026-06-18")
    assert tl["load_metric"] == "hr_load_proxy"
