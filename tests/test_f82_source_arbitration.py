"""F-82: select one HealthKit device stream inside an affected workout."""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import analysis
from health_advisor import db
from health_advisor import metrics
from tests.fixture_loader import load_f82_multi_source


def _ratio(conn, workout_id: int, clause: str = "", args=()) -> float | None:
    workout = conn.execute(
        "SELECT start_utc, end_utc, distance_mi FROM workouts WHERE id = ?",
        (workout_id,),
    ).fetchone()
    if workout["distance_mi"] is None:
        return None
    total = conn.execute(
        "SELECT SUM(value) FROM records "
        "WHERE metric = 'distance_walking_running' "
        "AND start_utc >= ? AND start_utc < ?" + clause,
        (workout["start_utc"], workout["end_utc"], *args),
    ).fetchone()[0]
    return total / workout["distance_mi"]


def test_fixture_characterises_duplicate_source_streams(conn):
    payload = load_f82_multi_source(conn)
    row = conn.execute(
        "SELECT SUM(value) AS summed, COUNT(*) AS rows, "
        "COUNT(DISTINCT source) AS sources FROM records "
        "WHERE metric = 'distance_walking_running' "
        "AND start_utc >= '2026-08-25T14:06:41+00:00' "
        "AND start_utc < '2026-08-25T15:04:18+00:00'",
    ).fetchone()
    assert row["rows"] == 4343
    assert row["sources"] == 3
    assert row["summed"] == pytest.approx(7.2799, abs=0.0001)
    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 13
    assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 42416
    assert payload["provenance"]["dedupe_evidence"][
        "all_workouts_rows_equal_distinct_keys"]


def test_all_thirteen_ratios_preserve_controls_and_correct_inflation(conn):
    payload = load_f82_multi_source(conn)
    before = {int(wid): _ratio(conn, int(wid)) for wid in payload["expected"]}
    clause, args = db._workout_arbitration(conn, "distance_walking_running")
    after = {int(wid): _ratio(conn, int(wid), clause, args)
             for wid in payload["expected"]}

    expected_after = {
        809: 1.0802, 810: 1.0000, 812: 1.0000, 815: 1.0001,
    }
    for wid, expected in expected_after.items():
        assert after[wid] == pytest.approx(expected, abs=0.0002)
        assert after[wid] != pytest.approx(before[wid], abs=0.0001)
    for wid in (746, 748, 762, 803, 804, 805, 806, 807, 808):
        assert after[wid] == before[wid]

    # Workout 809 is the no-GymKit control day: the whole-day iPhone-vs-Watch
    # rule still excludes the iPhone stream and keeps the Watch stream.
    day = "2026-08-21"
    watch = conn.execute(
        "SELECT DISTINCT source FROM records WHERE metric = ? AND "
        "local_date = ? AND source NOT LIKE '%|%' AND source LIKE '%Apple%'",
        ("distance_walking_running", day),
    ).fetchone()[0]
    iphone = conn.execute(
        "SELECT DISTINCT source FROM records WHERE metric = ? AND "
        "local_date = ? AND source NOT LIKE '%|%' AND source LIKE '%iPhone%'",
        ("distance_walking_running", day),
    ).fetchone()[0]
    kept_watch = conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric = ? AND local_date = ? "
        "AND source = ?" + clause,
        ("distance_walking_running", day, watch, *args),
    ).fetchone()[0]
    dropped_iphone = conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric = ? AND local_date = ? "
        "AND source = ?" + clause,
        ("distance_walking_running", day, iphone, *args),
    ).fetchone()[0]
    assert kept_watch > 0
    assert dropped_iphone == 0


def test_gymkit_day_scopes_gymkit_to_window_but_watch_wins_outside(conn):
    """An out-of-window Watch row survives beside in-window GymKit rows."""
    load_f82_multi_source(conn)
    day = "2026-08-25"
    watch = conn.execute(
        "SELECT DISTINCT source FROM records WHERE metric = ? AND "
        "local_date = ? AND source NOT LIKE '%|%' AND source LIKE '%Apple%'",
        ("distance_walking_running", day),
    ).fetchone()[0]
    iphone = conn.execute(
        "SELECT DISTINCT source FROM records WHERE metric = ? AND "
        "local_date = ? AND source NOT LIKE '%|%' AND source LIKE '%iPhone%'",
        ("distance_walking_running", day),
    ).fetchone()[0]
    synthetic = [
        # These match the measured out-of-window split: Watch 0.92728 mi and
        # iPhone 0.49555 mi.
        (0.92728, watch, "f82-synthetic-watch"),
        (0.49555, iphone, "f82-synthetic-iphone"),
    ]
    conn.executemany(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) VALUES "
        "('distance_walking_running', ?, 'mi', '2026-08-25T20:00:00Z', "
        "'2026-08-25T20:00:00Z', NULL, ?, ?, 'test', ?)",
        [(value, day, source, key) for value, source, key in synthetic],
    )
    conn.commit()

    clause, args = db._workout_arbitration(conn, "distance_walking_running")
    watch_row = conn.execute(
        "SELECT value FROM records WHERE dedupe_key = ?" + clause,
        ("f82-synthetic-watch", *args),
    ).fetchone()
    iphone_row = conn.execute(
        "SELECT value FROM records WHERE dedupe_key = ?" + clause,
        ("f82-synthetic-iphone", *args),
    ).fetchone()
    assert watch_row[0] == pytest.approx(0.92728)
    assert iphone_row is None

    total_miles = conn.execute(
        "SELECT SUM(value) FROM records WHERE metric = ? AND local_date = ?" + clause,
        ("distance_walking_running", day, *args),
    ).fetchone()[0]
    # Match the metre conversion used by the recorded Apple consolidated oracle.
    total_metres = total_miles * 1609.25
    assert total_miles == pytest.approx(4.37713, abs=0.00001)
    assert total_metres == pytest.approx(7043.9, abs=0.2)
    assert round((total_metres / 6791.8 - 1) * 100, 1) == 3.7


def test_read_window_does_not_import_a_watch_winner_from_outside_the_window(conn):
    """A session query only lets movement in that query choose its winner."""
    load_f82_multi_source(conn)
    start = "2026-08-26T20:01:00+00:00"
    end = "2026-08-26T20:01:20+00:00"
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('running', ?, ?, ?, 0.33, "
        "'test', ?)",
        (start, end, "2026-08-26", f"f82-window|{start}|{end}"),
    )
    # The Watch row is on the same local day but outside the requested session;
    # the iPhone row is the only device inside it.
    conn.executemany(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) VALUES "
        "('distance_walking_running', ?, 'mi', ?, ?, NULL, ?, ?, 'test', ?)",
        [
            (0.01, "2026-08-26T20:00:00+00:00", "2026-08-26T20:00:20+00:00",
             "2026-08-26", "Demo's Apple Watch", "f82-outside-watch"),
            (0.02, start, end, "2026-08-26", "Demo's iPhone", "f82-window-iphone"),
        ],
    )
    conn.commit()

    rows = metrics.bucket_series(conn, start, end)
    assert sum(row["miles"] for row in rows) == pytest.approx(0.02)


@pytest.mark.parametrize("source", ["GymKit|Demo's Apple Watch", ""])
def test_post_cutoff_opaque_source_labels_are_refused_loudly(conn, source):
    """Neither merged nor empty labels may silently enter device arbitration."""
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) VALUES "
        "('distance_walking_running', 0.1, 'mi', '2026-08-26T20:00:00+00:00', "
        "'2026-08-26T20:00:20+00:00', NULL, '2026-08-26', ?, 'test', ?)",
        (source, f"f82-invalid|{source}"),
    )
    conn.commit()

    with pytest.raises(ValueError, match="refuses.*source label"):
        db._workout_arbitration(conn, "distance_walking_running")


def test_pre_cutoff_arbitration_leaves_fixture_rows_untouched(conn):
    load_f82_multi_source(conn)
    day = conn.execute(
        "SELECT local_date FROM records WHERE local_date < ? "
        "AND metric = 'distance_walking_running' ORDER BY local_date DESC LIMIT 1",
        ("2026-08-21",),
    ).fetchone()[0]
    clause, args = db._workout_arbitration(conn, "distance_walking_running")
    total = conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric = ? AND local_date = ?" + clause,
        ("distance_walking_running", day, *args),
    ).fetchone()[0]
    expected = conn.execute(
        "SELECT COUNT(*) FROM records WHERE metric = ? AND local_date = ?",
        ("distance_walking_running", day),
    ).fetchone()[0]
    assert total == expected


def test_arbitration_is_bounded_and_survives_a_low_sqlite_variable_limit(conn):
    load_f82_multi_source(conn)
    clause, args = db._workout_arbitration(conn, "distance_walking_running")
    # The generated arbitration predicate has one fixed cutoff bind, never one
    # bind for each of the hundreds of loser rows in the fixture.
    assert len(args) == 1
    assert clause.count("?") == 1
    old_limit = conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 16)
    try:
        rows = analysis.mx.impact_bucket_rows(
            conn, "local_date BETWEEN ? AND ?", ("2026-08-25", "2026-08-25"))
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old_limit)
    assert rows


def test_metric_path_uses_one_selected_distance_stream_for_volume_and_blocks(
    conn, monkeypatch,
):
    load_f82_multi_source(conn)
    workout = conn.execute(
        "SELECT start_utc, end_utc FROM workouts WHERE id = 815"
    ).fetchone()
    with monkeypatch.context() as patch:
        patch.setattr(db, "_workout_arbitration", lambda *a, **k: ("", []))
        before_volume = analysis.impact_volume(
            conn, "2026-08-25", "2026-08-25", by="day")[0]
        before_block = analysis.longest_block(
            conn, workout["start_utc"], workout["end_utc"])

    after_volume = analysis.impact_volume(
        conn, "2026-08-25", "2026-08-25", by="day")[0]
    after_block = analysis.longest_block(
        conn, workout["start_utc"], workout["end_utc"])

    # This fixture predates the cadence stream, so neither volume result can
    # classify a bucket as jogging. The pace-only block dial still sees the
    # distributed duplicate stream; keeping these assertions separate is the
    # #193 scope guard.
    assert before_volume["jog_minutes"] == pytest.approx(0.0, abs=0.1)
    assert before_block["bridged_min"] == pytest.approx(57.3, abs=0.1)
    # 22.3, and IT IS SIX MINUTES TOO HIGH. The athlete supplied the segment times for
    # this session on 2026-08-28: it was the 4x4 test, four reps of 4:02/4:02/
    # 4:03/4:04 = 16.2 min of jogging. The "16.2-minute ground truth" an earlier
    # version of this comment cited WAS real ground truth, and a 2026-08-28
    # review that dismissed it as an artifact of the old attribution was wrong.
    #
    # Attributed to his segments, the dial is accurate where it counts: the four
    # reps measure 4.0/3.7/4.0/4.0 = 15.7 against his 16.2. The excess is
    # 5.0 min from SEGMENT 1 -- an 11:23 warmup WALK at 15'12"/mi and HR 99,
    # counted as jogging because the <=16 min/mi lane has no HR floor -- plus
    # ~1.6 min from the recoveries and cooldown. That is #193, not #189.
    #
    # #189 did not cause it, it UNMASKED it: before the fix this workout carried
    # two physically impossible buckets (1.74 and 0.75 min/mi) holding 0.638 mi
    # inside 40 seconds, which the 5.0 min/mi floor DISCARDED, and 25 bucket
    # slots sat empty -- coincidentally suppressing about as much as the warmup
    # over-count added. 17.3 was two errors cancelling. After the fix: 173
    # buckets, 0 missing, 0 implausible, 3.450 mi conserved at ratio 1.000.
    #
    # So this 22.3 was pinned as MEASURED-AND-KNOWN-WRONG. The fixture has no
    # step_count records, so the settled cadence rule reports zero here; the
    # live-vault measurement is the contaminated ~24-minute result described
    # by #193, not a tuning oracle. Do not "fix" it by reverting #189.
    # The '62-minute' figure remains the inflated duplicate-stream result.
    assert after_volume["jog_minutes"] == pytest.approx(0.0, abs=0.1)
    assert len(after_block["reps"]) == 5
    assert after_block["bridged_min"] == pytest.approx(5.0, abs=0.2)


def test_each_inflated_session_has_a_finite_selected_block_structure(conn):
    payload = load_f82_multi_source(conn)
    expected_blocks = {809: 0.0, 810: 0.0, 812: 0.0, 815: 5.0}
    for workout_id, expected in expected_blocks.items():
        workout = conn.execute(
            "SELECT start_utc, end_utc FROM workouts WHERE id = ?",
            (workout_id,),
        ).fetchone()
        block = analysis.longest_block(
            conn, workout["start_utc"], workout["end_utc"])
        if expected:
            assert block["reps"]
        else:
            assert block["reps"] == []
        assert block["bridged_min"] == pytest.approx(expected, abs=0.2)
    assert set(payload["multi_source_workout_ids"]) == {
        746, 748, 762, 809, 810, 812, 815
    }


def test_arbitrated_pairs_reports_only_the_three_changed_dates(conn):
    load_f82_multi_source(conn)
    assert db.arbitrated_pairs(conn) == [
        ("distance_walking_running", "2026-08-21"),
        ("distance_walking_running", "2026-08-22"),
        ("distance_walking_running", "2026-08-25"),
    ]
