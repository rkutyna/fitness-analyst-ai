"""F-82: select one HealthKit device stream inside an affected workout."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from health_advisor import analysis
from health_advisor import db
from health_advisor import metrics
from tests.fixture_loader import load_f82_multi_source


def _seed_cadence_where_the_retired_pace_lane_fired(conn, workout):
    """Give the cadence-blind fixture a step stream, so a jog bucket can exist.

    The fixture has no step_count, and since the one jog rule (#604) only a
    bucket with a cadence can be a jog bucket, so without this every block
    figure here reads zero whether or not the rule works. A synthetic 150
    steps/min sample goes into each bucket of the workout's window that the
    retired pace lane counted (pace 5-16 min/mi, heart rate not known to be under
    130), which restores the buckets the block expectations were recorded
    against. Returns the number of buckets seeded.
    """
    seeded = 0
    for bucket in metrics.bucket_series(
            conn, workout["start_utc"], workout["end_utc"]):
        pace, hr = bucket["pace_min_per_mi"], bucket["hr"]
        if not (pace is not None and 5.0 <= pace <= 16.0
                and not (hr is not None and hr < 130.0)):
            continue
        start = bucket["bucket_start_utc"]
        cur = conn.execute(
            "INSERT OR IGNORE INTO records (metric, value, unit, start_utc, "
            "end_utc, local_date, source, origin, dedupe_key) VALUES "
            "('step_count', ?, 'count', ?, ?, ?, 'cadence-seed', 'test', ?)",
            (150.0 / 3.0, start, start, bucket["local_date"],
             f"f82-cadence|{start}"))
        seeded += cur.rowcount
    conn.commit()
    return seeded


def _clear_cadence_seed(conn):
    conn.execute("DELETE FROM records WHERE source = 'cadence-seed'")
    conn.commit()


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


def test_utc_arbitration_bounds_many_post_floor_workouts_by_local_date(conn):
    """The UTC overlap remains exact, but the candidate scan uses its index."""
    load_f82_multi_source(conn)
    start = "2026-08-25T00:00:00+00:00"
    end = "2026-09-01T00:00:00+00:00"
    extra_workouts = []
    for i in range(256):
        day = date(2026, 9, 2) + timedelta(days=i)
        extra_workouts.append((
            "running", f"{day.isoformat()}T12:00:00+00:00",
            f"{day.isoformat()}T13:00:00+00:00", day.isoformat(),
            f"f82-scale-workout-{i}",
        ))
    conn.executemany(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "source, dedupe_key) VALUES (?, ?, ?, ?, 'test', ?)",
        extra_workouts,
    )
    conn.commit()

    clause, args = db._workout_arbitration(
        conn, "distance_walking_running",
        arbitration_window=(start, end), arbitration_window_kind="utc",
    )
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT 1 FROM records "
        "WHERE metric = ? AND start_utc >= ? AND start_utc < ?" + clause,
        ("distance_walking_running", start, end, *args),
    ).fetchall()
    workout_steps = [row[3] for row in plan if "workout" in row[3]]

    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] >= 250
    assert any(
        "idx_workouts_localdate" in detail
        and "local_date>? AND local_date<?" in detail
        for detail in workout_steps
    ), workout_steps


def test_arbitration_uses_roles_for_renamed_devices(conn):
    """A generic watch/phone pair still selects the watch stream."""
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("running", "2026-08-26T12:00:00+00:00",
         "2026-08-26T13:00:00+00:00", "2026-08-26", 60.0,
         "test", "renamed-device-workout"),
    )
    rows = []
    for source, value in (("Demo Watch", 0.01), ("Demo Phone", 0.02)):
        rows.append({
            "metric": "distance_walking_running", "value": value, "unit": "mi",
            "start_utc": "2026-08-26T12:00:00+00:00",
            "end_utc": "2026-08-26T12:00:20+00:00",
            "start_local": "2026-08-26 12:00:00", "local_date": "2026-08-26",
            "source": source, "origin": "test",
            "dedupe_key": f"renamed-device-{source}",
        })
    db.insert_records(conn, rows)

    clause, args = db._workout_arbitration(conn, "distance_walking_running")
    kept = conn.execute(
        "SELECT source FROM records WHERE metric = ?" + clause,
        ("distance_walking_running", *args),
    ).fetchall()
    assert [row[0] for row in kept] == ["Demo Watch"]


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
    # The generated arbitration predicate has one bind for each distinct raw
    # post-cutoff label, never one bind for each of the loser rows.
    assert set(args) == {"Demo Apple Watch", "Demo iPhone", "GymKit"}
    assert len(args) == len(set(args)) == 3
    assert clause.count("?") == 3
    # The bucket query itself binds a fixed, small number of values — none of
    # them proportional to the rows it reads, which is the property this
    # ceiling exists to hold. It was 16 until the step series gained its own
    # sample-span guard (one more constant, bound the same way the distance
    # series binds it). Raise this only alongside a bind that is likewise
    # constant; a bind that grows with the data must be rejected here rather
    # than accommodated, because the platform with the low limit is not the
    # one this suite runs on.
    old_limit = conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 17)
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

        before_miles = sum(row["miles"] for row in metrics.bucket_series(
            conn, workout["start_utc"], workout["end_utc"]))

    after_volume = analysis.impact_volume(
        conn, "2026-08-25", "2026-08-25", by="day")[0]
    after_block = analysis.longest_block(
        conn, workout["start_utc"], workout["end_utc"])
    after_miles = sum(row["miles"] for row in metrics.bucket_series(
        conn, workout["start_utc"], workout["end_utc"]))

    # This fixture predates the cadence stream, so neither volume result can
    # classify a bucket as jogging -- and since the one jog rule (#604) the block
    # dial cannot either: pace no longer makes a jog bucket, so a session with no
    # step data has no block. The block figures below were 57.3 (the duplicate
    # streams, pace-only) and 5.0 (the selected stream, five pace-lane reps)
    # while the dial still had a pace lane; they are 0.0 and no reps now, with
    # or without arbitration. That would leave this test unable to see the
    # arbitration at all, so the bucket rows -- the one series volume and blocks
    # both read -- are asserted directly: the duplicate streams sum to 7.28 mi
    # in this workout's window and the selected stream to its recorded 3.45 mi.
    assert before_miles == pytest.approx(7.2799, abs=0.001)
    assert after_miles == pytest.approx(3.4495, abs=0.001)
    assert before_volume["jog_minutes"] == pytest.approx(0.0, abs=0.1)
    assert before_block["bridged_min"] == pytest.approx(0.0, abs=0.1)
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
    assert after_block["reps"] == []
    assert after_block["bridged_min"] == pytest.approx(0.0, abs=0.2)

    # Positive control. Zero block minutes with or without arbitration is what a
    # cadence-blind fixture reads and also what a jog rule that never fires
    # reads, so the zeros above cannot tell the two apart. Give the fixture a
    # synthetic step stream in the buckets the retired pace lane counted and the
    # recorded figures must come back, and must still tell the duplicate streams
    # (one 57.3-minute block) from the selected stream (five 5.0-minute reps).
    with monkeypatch.context() as patch:
        patch.setattr(db, "_workout_arbitration", lambda *a, **k: ("", []))
        assert _seed_cadence_where_the_retired_pace_lane_fired(conn, workout) > 0
        seeded_before = analysis.longest_block(
            conn, workout["start_utc"], workout["end_utc"])
        seeded_before_volume = analysis.impact_volume(
            conn, "2026-08-25", "2026-08-25", by="day")[0]
    _clear_cadence_seed(conn)
    assert _seed_cadence_where_the_retired_pace_lane_fired(conn, workout) > 0
    seeded_after = analysis.longest_block(
        conn, workout["start_utc"], workout["end_utc"])
    seeded_after_volume = analysis.impact_volume(
        conn, "2026-08-25", "2026-08-25", by="day")[0]
    assert seeded_before["bridged_min"] == pytest.approx(57.3, abs=0.1)
    assert seeded_before_volume["jog_minutes"] == pytest.approx(57.3, abs=0.1)
    assert len(seeded_after["reps"]) == 5
    assert seeded_after["bridged_min"] == pytest.approx(5.0, abs=0.2)
    assert seeded_after_volume["jog_minutes"] == pytest.approx(20.0, abs=0.1)


def test_each_inflated_session_has_a_finite_selected_block_structure(conn):
    payload = load_f82_multi_source(conn)
    # 815 read 5.0 while the block dial had a pace lane; the fixture carries no
    # step data, so under the one jog rule (#604) every session here has no
    # block. The selected stream is checked on the bucket rows instead: each
    # session's bucket miles equal its recorded distance (809's 1.0802 is the
    # known whole-day Watch-vs-iPhone residue, see the ratios test above).
    expected_blocks = {809: 0.0, 810: 0.0, 812: 0.0, 815: 0.0}
    expected_ratio = {809: 1.0802, 810: 1.0, 812: 1.0, 815: 1.0001}
    for workout_id, expected in expected_blocks.items():
        workout = conn.execute(
            "SELECT start_utc, end_utc, distance_mi FROM workouts WHERE id = ?",
            (workout_id,),
        ).fetchone()
        block = analysis.longest_block(
            conn, workout["start_utc"], workout["end_utc"])
        assert block["reps"] == []
        assert block["bridged_min"] == pytest.approx(expected, abs=0.2)
        miles = sum(row["miles"] for row in metrics.bucket_series(
            conn, workout["start_utc"], workout["end_utc"]))
        assert miles / workout["distance_mi"] == pytest.approx(
            expected_ratio[workout_id], abs=0.0005)
    assert set(payload["multi_source_workout_ids"]) == {
        746, 748, 762, 809, 810, 812, 815
    }

    # Positive control: the same four sessions with a synthetic step stream in the
    # buckets the retired pace lane counted, which is where the recorded block
    # figures were read. 815 is the only one of the four with any such bucket; its
    # five 5.0-minute reps are the selected stream's block structure, and the
    # three controls stay at no block.
    expected_seeded = {809: 0.0, 810: 0.0, 812: 0.0, 815: 5.0}
    for workout_id, expected in expected_seeded.items():
        workout = conn.execute(
            "SELECT start_utc, end_utc FROM workouts WHERE id = ?",
            (workout_id,),
        ).fetchone()
        seeded = _seed_cadence_where_the_retired_pace_lane_fired(conn, workout)
        block = analysis.longest_block(
            conn, workout["start_utc"], workout["end_utc"])
        assert (seeded > 0) == (expected > 0)
        assert bool(block["reps"]) == (expected > 0)
        assert block["bridged_min"] == pytest.approx(expected, abs=0.2)
        _clear_cadence_seed(conn)


def test_arbitrated_pairs_reports_only_the_three_changed_dates(conn):
    load_f82_multi_source(conn)
    assert db.arbitrated_pairs(conn) == [
        ("distance_walking_running", "2026-08-21"),
        ("distance_walking_running", "2026-08-22"),
        ("distance_walking_running", "2026-08-25"),
    ]
