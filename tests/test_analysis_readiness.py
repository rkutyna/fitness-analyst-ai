from health_advisor import analysis as A
from tests.conftest import seed_metric


RESCALE_DATE = "2026-07-31"


def test_readiness_establishing_when_sparse(conn):
    seed_metric(conn, "resting_heart_rate", "2026-06-05", [60, 60, 60, 60])  # 4 days
    seed_metric(conn, "heart_rate_variability", "2026-06-05", [45, 45, 45, 45])
    rd = A.readiness(conn, as_of="2026-06-08")
    assert rd["status"] == "establishing_baseline"
    assert rd.get("score") is None


def test_readiness_sleep_only_does_not_produce_confident_score(conn):
    # Plenty of sleep history but NO HRV/RHR baseline -> no confident readiness.
    seed_metric(conn, "sleep_asleep", "2026-05-01", [450] * 30)
    rd = A.readiness(conn, as_of="2026-05-30")
    assert rd["status"] == "establishing_baseline"
    assert rd.get("score") is None


def test_readiness_red_when_rhr_up_hrv_down(conn):
    # 30 days baseline then a bad stretch. Deliberately three days, not one:
    # readiness now describes a 3-day state (audit P3-6), and a single low night
    # inside normal variation is not evidence of being under-recovered.
    seed_metric(conn, "resting_heart_rate", "2026-05-01",
                [55] * 27 + [70] * 3)       # RHR up and staying up (bad)
    seed_metric(conn, "heart_rate_variability", "2026-05-01",
                [60] * 27 + [35] * 3)       # HRV cratered (bad)
    seed_metric(conn, "sleep_asleep", "2026-05-01",
                [450] * 27 + [300] * 3)     # short sleep
    rd = A.readiness(conn, as_of="2026-05-30")
    assert rd["status"] == "ok"
    assert rd["band"] == "red"
    assert rd["score"] < 34


def test_readiness_green_when_recovered(conn):
    seed_metric(conn, "resting_heart_rate", "2026-05-01", [55] * 27 + [50] * 3)
    seed_metric(conn, "heart_rate_variability", "2026-05-01", [60] * 27 + [80] * 3)
    seed_metric(conn, "sleep_asleep", "2026-05-01", [450] * 30)
    rd = A.readiness(conn, as_of="2026-05-30")
    assert rd["band"] == "green"


def test_missing_rhr_holds_the_prior_readiness_band(conn):
    seed_metric(conn, "resting_heart_rate", "2026-06-01", [55] * 28)
    seed_metric(conn, "heart_rate_variability", "2026-06-01", [60] * 30)
    seed_metric(conn, "sleep_asleep", "2026-06-01", [450] * 30)

    rd = A.readiness(conn, as_of="2026-06-30")

    assert rd["status"] == "partial"
    assert rd["band"] == "amber"
    assert "rhr" in rd["note"]
    assert "absent" in rd["note"]


def test_missing_hrv_holds_the_prior_readiness_band(conn):
    seed_metric(conn, "resting_heart_rate", "2026-06-01", [55] * 30)
    seed_metric(conn, "heart_rate_variability", "2026-06-01", [60] * 28)
    seed_metric(conn, "sleep_asleep", "2026-06-01", [450] * 30)

    rd = A.readiness(conn, as_of="2026-06-30")

    assert rd["status"] == "partial"
    assert rd["band"] == "amber"
    assert "hrv" in rd["note"]
    assert "absent" in rd["note"]


def test_a_trend_line_refuses_to_cross_the_rescale():
    line = A.readiness_rescale_refusal(today="2026-07-08", week_ago="2026-07-01")
    assert line is None or "instrument changed" in line


def test_a_trend_line_refuses_an_actual_crossing_of_the_rescale():
    line = A.readiness_rescale_refusal(today="2026-08-07", week_ago="2026-07-30")
    assert line and "instrument changed" in line


def test_the_rescale_date_is_recorded_where_the_code_can_see_it():
    assert A.SUBSCORE_K_RESCALED_ON == RESCALE_DATE


# --- not a step function (audit P3-6) --------------------------------------- #

def test_one_noisy_day_cannot_swing_readiness_across_the_scale(conn):
    # Measured HRV daily CV is 17.1% and SUBSCORE_K = 2.5 saturated at ±20%, so
    # the HRV subscore — 40% of the composite — was effectively binary. One low
    # night moved the score 20 points; the live series swung green→amber→red→
    # green on consecutive days.
    seed_metric(conn, "resting_heart_rate", "2026-05-01", [55] * 30)
    seed_metric(conn, "heart_rate_variability", "2026-05-01", [60] * 29 + [35])
    seed_metric(conn, "sleep_asleep", "2026-05-01", [450] * 30)
    calm = A.readiness(conn, as_of="2026-05-29")["score"]
    dip = A.readiness(conn, as_of="2026-05-30")["score"]
    assert abs(dip - calm) < 15
    assert dip < calm            # still moves in the right direction


def test_band_holds_through_an_oscillating_series(conn):
    # A metric whose day-to-day spread is 17% of its mean must not relabel the
    # user every morning. Same underlying state all week -> same label all week.
    hrv = [60, 78, 45, 70, 50, 65, 55] * 4
    seed_metric(conn, "heart_rate_variability", "2026-05-01", hrv)
    seed_metric(conn, "resting_heart_rate", "2026-05-01", [55] * 28)
    seed_metric(conn, "sleep_asleep", "2026-05-01", [450] * 28)
    bands = [A.readiness(conn, as_of=f"2026-05-{d}")["band"] for d in range(22, 29)]
    assert len(set(bands)) == 1, bands


def test_band_is_sticky_at_the_boundary_but_still_moves():
    assert A._sticky_band(68, "amber") == "amber"     # 1 point over: noise
    assert A._sticky_band(72, "amber") == "green"     # clear of it: real
    assert A._sticky_band(65, "green") == "green"     # 2 points under: stays
    assert A._sticky_band(60, "green") == "amber"
    assert A._sticky_band(36, "red") == "red"
    assert A._sticky_band(40, "red") == "amber"
    assert A._sticky_band(75, None) == "green"        # no history: plain bands
    assert A._sticky_band(20, None) == "red"


# --- staleness (audit P2-2): a dead watch must not read green forever -------- #

def _seed_recovery(conn, start="2026-05-01", days=30):
    seed_metric(conn, "resting_heart_rate", start, [55] * days)
    seed_metric(conn, "heart_rate_variability", start, [60] * days)
    seed_metric(conn, "sleep_asleep", start, [450] * days)


def test_readiness_is_stale_when_the_watch_stopped_reporting(conn):
    # 30 good days ending 2026-05-30, then nothing. Ten days later the same
    # green score was still being reported as today's.
    _seed_recovery(conn)
    rd = A.readiness(conn, as_of="2026-06-09")
    assert rd["status"] == "stale"
    assert rd["score"] is None
    assert rd["band"] is None
    assert rd["stale_days"] == 10
    assert rd["latest_date"] == "2026-05-30"
    assert rd["as_of"] == "2026-06-09"


def test_readiness_still_scores_on_yesterdays_data(conn):
    # The 05:00 brief runs before the phone syncs: one day old is normal, not
    # stale. It must still score, and say how old the data is.
    _seed_recovery(conn)
    rd = A.readiness(conn, as_of="2026-05-31")
    assert rd["status"] in ("ok", "partial")
    assert rd["score"] is not None
    assert rd["stale_days"] == 1
    assert rd["latest_date"] == "2026-05-30"


def test_readiness_factors_carry_their_source_date(conn):
    _seed_recovery(conn)
    rd = A.readiness(conn, as_of="2026-05-30")
    assert rd["factors"]
    for f in rd["factors"]:
        assert f["date"] == "2026-05-30"
        assert f["age_days"] == 0


def test_readiness_publishes_metric_ownership_for_composite_and_factors(conn):
    _seed_recovery(conn)
    rd = A.readiness(conn, as_of="2026-05-30")

    assert rd["field_metrics"] == {"score": "readiness"}
    assert rd["components"]["field_metrics"] == {"sleep": "readiness"}
    assert {f["component"]: f["field_metrics"] for f in rd["factors"]} == {
        "hrv": {"current": "heart_rate_variability",
                "baseline": "heart_rate_variability"},
        "rhr": {"current": "resting_heart_rate",
                "baseline": "resting_heart_rate"},
        "sleep": {"current": "sleep_asleep"},
    }


def test_readiness_drops_only_the_stale_component(conn):
    # HRV keeps flowing, resting HR stops three days back (the live case: the
    # last-valued metrics end two days before the rest of the DB). The score
    # must come from the fresh input alone, not from a three-day-old RHR.
    seed_metric(conn, "heart_rate_variability", "2026-05-01", [60] * 30)
    seed_metric(conn, "resting_heart_rate", "2026-05-01", [55] * 27)
    seed_metric(conn, "sleep_asleep", "2026-05-01", [450] * 30)
    rd = A.readiness(conn, as_of="2026-05-30")
    assert rd["status"] == "partial"
    assert "hrv" in rd["components"]
    assert "rhr" not in rd["components"]
    rhr = next(f for f in rd["factors"] if f["component"] == "rhr")
    assert rhr["stale"] is True
    assert rhr["age_days"] == 3


def test_stale_readiness_is_distinguished_from_no_baseline(conn):
    # Four days of history is not a stale watch, it's a new one.
    seed_metric(conn, "resting_heart_rate", "2026-06-05", [60] * 4)
    seed_metric(conn, "heart_rate_variability", "2026-06-05", [45] * 4)
    rd = A.readiness(conn, as_of="2026-06-30")
    assert rd["status"] == "establishing_baseline"


def test_stale_readiness_gets_a_check_your_watch_suggestion(conn):
    _seed_recovery(conn)
    rd = A.readiness(conn, as_of="2026-06-09")
    sugg = A.suggestions(rd, {"status": "no_load_metric", "acwr": None})
    assert any("10 days" in s["because"] for s in sugg)
