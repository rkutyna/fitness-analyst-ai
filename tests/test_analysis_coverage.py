from health_advisor import analysis as A
from health_advisor import metrics as mx
from tests.conftest import seed_metric


def test_coverage_marks_sparse_and_active(conn):
    seed_metric(conn, "step_count", "2026-04-01", list(range(1, 71)))  # 70 days, ends ~06-09
    seed_metric(conn, "resting_heart_rate", "2026-06-04", [60, 61, 59, 60])  # 4 days -> sparse
    cov = A.coverage(conn, as_of="2026-06-09")
    by = {c["metric"]: c for c in cov}
    assert by["step_count"]["status"] == "active"
    assert by["resting_heart_rate"]["status"] == "sparse"
    assert by["heart_rate_variability"]["status"] == "missing"


def test_coverage_survives_a_single_missing_day(conn):
    # The live bug: active_energy was absent for exactly one day and a
    # 2,532-day metric was labelled "sparse" — the same all-or-nothing
    # brittleness that took ACWR dark, in the section that tells the narrator
    # which numbers to distrust.
    seed_metric(conn, "step_count", "2026-04-01", list(range(1, 71)))
    conn.execute("DELETE FROM daily_metrics WHERE metric='step_count' AND date=?",
                 ("2026-06-05",))
    conn.commit()
    by = {c["metric"]: c for c in A.coverage(conn, as_of="2026-06-09")}
    assert by["step_count"]["status"] == "active"
    assert by["step_count"]["recent_days"] == 13
    assert by["step_count"]["window_days"] == A.SPARSE_DAYS
    assert 0.9 < by["step_count"]["recent_fraction"] < 1.0


def test_coverage_is_sparse_when_most_of_the_window_is_missing(conn):
    # Half a window present is genuinely thin and must still say so.
    seed_metric(conn, "respiratory_rate", "2026-05-01", [14] * 40)
    for d in ("2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05",
              "2026-06-06", "2026-06-07", "2026-06-08"):
        conn.execute("DELETE FROM daily_metrics WHERE metric='respiratory_rate' "
                     "AND date=?", (d,))
    conn.commit()
    by = {c["metric"]: c for c in A.coverage(conn, as_of="2026-06-09")}
    assert by["respiratory_rate"]["status"] == "sparse"


def test_coverage_does_not_count_days_with_no_usable_value(conn):
    # Live, the newest two rows of every metric carry a NULL value column.
    # Counting them made coverage report resting_heart_rate fresh through today
    # while readiness was correctly treating it as two days stale.
    seed_metric(conn, "resting_heart_rate", "2026-05-27", [60] * 14)
    # Null the column this metric actually reads, resolved from the catalog
    # rather than hardcoded: resting_heart_rate moved from 'last' to 'mean' on
    # 2026-08-01, which silently made a hardcoded `last = NULL` a no-op and the
    # test vacuous. The behaviour under test is "a day with no usable value
    # does not count as covered", whichever column carries it.
    col = mx.value_col("resting_heart_rate")
    conn.execute(f"UPDATE daily_metrics SET {col} = NULL WHERE date >= '2026-06-08'")
    conn.commit()
    by = {c["metric"]: c for c in A.coverage(conn, as_of="2026-06-09")}
    assert by["resting_heart_rate"]["last_date"] == "2026-06-07"
    assert by["resting_heart_rate"]["recent_days"] == 12


def test_coverage_does_not_claim_a_metric_covers_an_unobserved_as_of_day(conn):
    seed_metric(conn, "step_count", "2026-06-01", list(range(8)))
    by = {c["metric"]: c for c in A.coverage(conn, as_of="2026-06-09")}
    assert by["step_count"].get("covers_as_of") is False


def test_a_daily_metric_absent_only_on_as_of_is_not_called_sparse(conn):
    """The normal morning state of a live vault must not read as "too thin".

    Fourteen consecutive days, then ask about the day after the last one: the
    metric is missing exactly `as_of` and nothing else. Measured 2026-08-24 on
    the snapshot, treating that as `sparse` handed `talking_points` six vitals
    at 0.93 density as COVERAGE_THIN. `covers_as_of` states the bound honestly;
    the status must stay `active`, and one day is not past a daily cadence.
    """
    seed_metric(conn, "step_count", "2026-06-01", list(range(14)))
    row = {c["metric"]: c for c in A.coverage(conn, as_of="2026-06-15")}["step_count"]
    assert row["covers_as_of"] is False
    assert row["behind"] is False
    assert row["status"] == "active"
    assert row["recent_fraction"] >= A.COVERAGE_MIN_FRACTION


def test_coverage_uses_metric_cadence_for_a_short_intermittent_gap(conn):
    # The last observation is three days old, but this metric normally arrives
    # every five days. It is not cadence-late yet (density may still be thin).
    seed_metric(conn, "vo2_max", "2026-05-01", [40])
    seed_metric(conn, "vo2_max", "2026-05-06", [41])
    seed_metric(conn, "vo2_max", "2026-05-11", [42])
    by = {c["metric"]: c for c in A.coverage(conn, as_of="2026-05-14")}
    assert by["vo2_max"]["status"] != "stale"


def test_coverage_marks_stale(conn):
    seed_metric(conn, "vo2_max", "2022-01-01", [40, 41])  # ancient
    cov = A.coverage(conn, as_of="2026-06-09")
    by = {c["metric"]: c for c in cov}
    assert by["vo2_max"]["status"] == "stale"


# --- talking_points must report the statuses that mean "a metric stopped" -----
# F6-2, audit part 6. The filter asked for ("sparse", "establishing"); coverage()
# returns only missing/stale/sparse/active. So "establishing" was unreachable and
# the two statuses meaning a tracked metric had gone silent were both excluded.
# vo2_max was dropped at ingest for 16 days while coverage() flagged it daily.

def _parts(coverage_rows, as_of="2026-08-16"):
    """The minimum briefing shape talking_points reads — every key it touches."""
    return {"coverage": coverage_rows, "as_of": as_of, "readiness": {},
            "trends": {}, "training_load": {}, "movers": [],
            "highlights": [], "workout_focus": None}


def test_a_stale_metric_reaches_the_narrator_with_its_last_seen_date():
    seeds = A.talking_points(_parts([
        {"metric": "vo2_max", "status": "stale", "last_date": "2026-07-31"}]))
    hit = [s for s in seeds if "vo2_max" in s["seed"]]
    assert hit, "a stale metric produced no seed"
    assert "2026-07-31" in hit[0]["seed"]


def test_a_missing_metric_reaches_the_narrator():
    seeds = A.talking_points(_parts([
        {"metric": "heart_rate_variability", "status": "missing", "last_date": None}]))
    assert any("heart_rate_variability" in s["seed"] for s in seeds)


def test_sparse_is_still_reported_separately_from_stopped():
    seeds = A.talking_points(_parts([
        {"metric": "body_mass", "status": "sparse", "last_date": "2026-08-16"},
        {"metric": "vo2_max", "status": "stale", "last_date": "2026-07-31"}]))
    text = " | ".join(s["seed"] for s in seeds)
    assert "thin/sparse data for: body_mass" in text
    assert "vo2_max has stopped arriving" in text


def test_an_active_metric_produces_no_coverage_seed():
    seeds = A.talking_points(_parts([
        {"metric": "step_count", "status": "active", "last_date": "2026-08-16"}]))
    assert not [s for s in seeds if s["topic"] == "coverage"]


def test_every_filtered_status_is_one_coverage_can_actually_return():
    """The invariant that would have caught F6-2 the day it was written.

    The old filter named "establishing", which coverage() has never emitted, so
    the filter silently matched nothing and the real statuses were excluded."""
    assert set(A.COVERAGE_STOPPED) <= set(A.COVERAGE_STATUSES)
    assert set(A.COVERAGE_THIN) <= set(A.COVERAGE_STATUSES)
    # And the vocabulary is the truth: coverage() emits nothing outside it.
    assert set(A.COVERAGE_STATUSES) == {"missing", "stale", "sparse", "active"}


def test_coverage_only_ever_emits_the_declared_vocabulary(conn):
    seed_metric(conn, "step_count", "2026-04-01", list(range(1, 71)))
    seed_metric(conn, "resting_heart_rate", "2026-06-04", [60, 61, 59, 60])
    for c in A.coverage(conn, as_of="2026-06-09"):
        assert c["status"] in A.COVERAGE_STATUSES
