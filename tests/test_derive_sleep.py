"""Sleep is attributed by session, not by sample (E7-1), and absence is not zero (E7-2).

`local_date` was assigned per sample as the date the SAMPLE ends. That was right
while HealthKit gave one span per night (2017 samples average 427 min). It is
wrong now: a 2026 night is 20-40 samples averaging 19.7 min, so every sample
ending before midnight is filed under the PREVIOUS date. Two consequences, both
live: a day's `sleep_asleep` total is two half-nights, and `sleep_bedtime` is
clipped at midnight because compute_sleep_timing only ever saw one date's rows.

These tests run against `tests/fixtures/sleep_nights.json` — real records from
six nights across four eras — never the production DB (conftest.py forbids it).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from health_advisor import derive as D

FIXTURE = Path(__file__).parent / "fixtures" / "sleep_nights.json"


@pytest.fixture(scope="module")
def nights() -> dict:
    return json.loads(FIXTURE.read_text())["data"]


def _seed(conn, records: list[dict]) -> None:
    """Insert captured raw records verbatim, local_date included.

    Verbatim matters: the defect IS the stored local_date, so a helper that
    recomputed it would test the fix against its own output.
    """
    for r in records:
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (r["metric"], r["value"], r["unit"], r["start_utc"], r["end_utc"],
             r["start_local"], r["local_date"], r["source"], r["origin"],
             r["dedupe_key"]))
    conn.commit()


def _night(conn, nights, day):
    _seed(conn, nights[day]["records"])
    D.reattribute_sleep(conn, day, day, apply=True)
    conn.commit()
    return D.compute_sleep_timing(D._sleep_intervals(conn, day), day)


def _clock(hours_since_prev_noon: float) -> str:
    """Render sleep_bedtime (hours since previous-day noon) as a wall clock."""
    t = datetime(2000, 1, 1, 12) + timedelta(hours=hours_since_prev_noon)
    return t.strftime("%H:%M")


# --- the defect, on the night that measured it ----------------------------

def test_a_pre_midnight_onset_is_not_clipped(conn, nights):
    # Night of 2026-07-13 -> 07-14. The stored sleep_bedtime is 11.97 h after
    # 07-13 noon, i.e. 23:58 — which is not when he went to bed, it is the
    # first sample that happened to survive the date filter. The session
    # actually begins 2026-07-13 23:26:58.
    out = _night(conn, nights, "2026-07-14")
    assert _clock(out["sleep_bedtime"]) == "23:26"
    assert out["sleep_bedtime"] < nights["2026-07-14"]["stored_derived"]["sleep_bedtime"]


def test_wake_time_is_unchanged(conn, nights):
    # Only the LEADING edge of a night was clipped: the session end was always
    # after midnight and so always inside the day's rows. If a re-attribution
    # moves a wake time, it has regrouped something it should not have.
    for day in ("2026-07-14", "2026-08-15", "2026-07-05"):
        stored = nights[day]["stored_derived"]["sleep_wake_time"]
        out = _night(conn, nights, day)
        assert out["sleep_wake_time"] == pytest.approx(stored, abs=1 / 60)
        conn.execute("DELETE FROM records")
        conn.commit()


def test_asleep_total_is_one_night_not_two(conn, nights):
    # daily_metrics.sleep_asleep.sum[D] was the post-midnight half of the night
    # ending on D plus the pre-midnight half of the night ending on D+1.
    _seed(conn, nights["2026-07-14"]["records"])
    before = conn.execute(
        "SELECT SUM(value) FROM records WHERE metric = 'sleep_asleep' "
        "AND local_date = '2026-07-14'").fetchone()[0]
    D.reattribute_sleep(conn, "2026-07-14", "2026-07-14", apply=True)
    conn.commit()
    after = conn.execute(
        "SELECT SUM(value) FROM records WHERE metric = 'sleep_asleep' "
        "AND local_date = '2026-07-14'").fetchone()[0]
    # One night, bounded by the session the timing metrics report.
    out = D.compute_sleep_timing(D._sleep_intervals(conn, "2026-07-14"), "2026-07-14")
    assert after <= out["sleep_time_in_bed"]
    assert after != pytest.approx(before)


def test_a_night_that_begins_after_midnight_does_not_move(conn, nights):
    # 2026-07-05's episode starts 2026-07-04 23:53 — the control case. Any rule
    # that moves records here is moving them on the calendar, not on the data.
    _seed(conn, nights["2026-07-05"]["records"])
    moves = D.reattribute_sleep(conn, "2026-07-05", "2026-07-05")
    assert moves == []


# --- the eras the change has to survive ------------------------------------

@pytest.mark.parametrize("day", ["2017-03-15", "2021-11-10"])
def test_the_historical_eras_are_untouched(conn, nights, day):
    # 2017 is one span per night, so end-dating was already session-dating.
    # 2021's 61-minute samples are the untested middle: measured, zero moves.
    _seed(conn, nights[day]["records"])
    assert D.reattribute_sleep(conn, day, day) == []


def test_the_untested_middle_moves_only_what_crosses_midnight(conn, nights):
    # 2022-05-18's neighbourhood holds one genuinely midnight-crossing episode
    # (21:00 -> next evening). Exactly its two pre-midnight samples move, and
    # they move forward by one day — never backward.
    _seed(conn, nights["2022-05-18"]["records"])
    moves = D.reattribute_sleep(conn, "2022-05-17", "2022-05-19")
    assert len(moves) == 2
    for _id, old, new in moves:
        assert new > old
        assert (datetime.fromisoformat(new) - datetime.fromisoformat(old)).days == 1


# --- the gap threshold -----------------------------------------------------

@pytest.mark.parametrize("gap", [15, 30, 60, 90, 120])
def test_the_gap_threshold_does_not_change_the_rebuild_on_real_nights(
        conn, nights, gap):
    # The 2026 nights have no internal gap anywhere near these thresholds, so
    # every one of them must produce the same answer on the REAL series. A
    # rebuild that is sensitive to the threshold here is sessionising noise.
    #
    # 18, not 8: the fixture window holds two midnight-crossing episodes, and
    # asking about 07-14 settles the record for its neighbours too (8 samples
    # move 07-13 -> 07-14, 10 move 07-14 -> 07-15). The number that matters is
    # that it is the same 18 at every threshold.
    _seed(conn, nights["2026-07-14"]["records"])
    moves = D.reattribute_sleep(conn, "2026-07-14", "2026-07-14", gap_min=gap)
    assert len(moves) == 18
    assert {(old, new) for _, old, new in moves} == {
        ("2026-07-13", "2026-07-14"), ("2026-07-14", "2026-07-15")}


def test_the_gap_threshold_is_actually_honoured(conn):
    # Guards the test above: a synthetic night with a 45-minute internal gap
    # must split at 30 and merge at 60. Without this, an implementation that
    # ignores `gap_min` entirely passes the parametrized test.
    ivs = [D.Interval(datetime(2026, 1, 1, 22, 0), datetime(2026, 1, 1, 23, 0), "sleep_asleep"),
           D.Interval(datetime(2026, 1, 1, 23, 45), datetime(2026, 1, 2, 6, 0), "sleep_asleep")]
    assert len(D.sleep_sessions(ivs, gap_min=30)) == 2
    assert len(D.sleep_sessions(ivs, gap_min=60)) == 1


# --- E7-2: absence is not zero --------------------------------------------

def test_awakenings_are_none_when_the_metric_was_never_recorded(conn, nights):
    # sleep_awake exists for 2019 and 2026 only. 2,444 of 2,535 stored zeros
    # (96%) meant "not measured", not "slept through" — and wear_hours in this
    # same module already returns None for exactly this reason.
    out = _night(conn, nights, "2017-03-15")
    assert "sleep_awakenings" not in out
    assert "sleep_awake_longest" not in out


def test_awakenings_are_reported_when_the_metric_is_present(conn, nights):
    out = _night(conn, nights, "2026-07-14")
    assert out["sleep_awakenings"] > 0
    assert out["sleep_awake_longest"] > 0


def test_a_measured_night_with_no_qualifying_awakening_reports_zero(conn):
    # The distinction that matters: the metric IS present that day, but every
    # awake span is under MIN_AWAKE_MIN. That is a real zero and must not be
    # refused along with the unmeasured ones.
    ivs = [D.Interval(datetime(2026, 1, 1, 23, 0), datetime(2026, 1, 2, 6, 0), "sleep_asleep"),
           D.Interval(datetime(2026, 1, 2, 3, 0), datetime(2026, 1, 2, 3, 0, 20), "sleep_awake")]
    out = D.compute_sleep_timing(ivs, "2026-01-02")
    assert out["sleep_awakenings"] == 0.0
    assert out["sleep_awake_longest"] == 0.0


def test_update_for_days_writes_no_awakenings_row_when_unmeasured(conn, nights):
    _seed(conn, nights["2017-03-15"]["records"])
    D.update_for_days(conn, ["2017-03-15"])
    conn.commit()
    rows = {r[0] for r in conn.execute(
        "SELECT metric FROM daily_metrics WHERE date = '2017-03-15'")}
    assert "sleep_bedtime" in rows
    assert "sleep_awakenings" not in rows
    assert "sleep_awake_longest" not in rows


# --- the rewrite itself ----------------------------------------------------

def test_reattribution_is_idempotent(conn, nights):
    _seed(conn, nights["2026-07-14"]["records"])
    first = D.reattribute_sleep(conn, "2026-07-14", "2026-07-14", apply=True)
    conn.commit()
    assert first
    assert D.reattribute_sleep(conn, "2026-07-14", "2026-07-14", apply=True) == []


def test_reattribution_reports_the_pairs_it_invalidated(conn, nights):
    # Moving a record from D to D+1 makes the stored daily_metrics rollup wrong
    # on BOTH dates. A caller that recomputes only one of them leaves a stale
    # aggregate behind, which is the exact class of defect verify_daily_metrics
    # exists to catch.
    _seed(conn, nights["2026-07-14"]["records"])
    moves = D.reattribute_sleep(conn, "2026-07-14", "2026-07-14", apply=True)
    pairs = D.pairs_for_moves(moves, ["sleep_asleep"])
    assert ("sleep_asleep", "2026-07-13") in pairs
    assert ("sleep_asleep", "2026-07-14") in pairs


def test_reattribution_never_touches_a_non_sleep_record(conn, nights):
    _seed(conn, nights["2026-07-14"]["records"])
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, start_local, "
        "local_date, source, origin, dedupe_key) VALUES "
        "('heart_rate', 60, 'count/min', '2026-07-13T23:30:00+00:00', "
        "'2026-07-13T23:30:00+00:00', '2026-07-13 23:30:00', '2026-07-13', "
        "'test', 'receiver', 'hr-key')")
    conn.commit()
    D.reattribute_sleep(conn, "2026-07-14", "2026-07-14", apply=True)
    conn.commit()
    assert conn.execute(
        "SELECT local_date FROM records WHERE dedupe_key = 'hr-key'"
    ).fetchone()[0] == "2026-07-13"
