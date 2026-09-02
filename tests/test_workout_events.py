"""Workout segment/lap events: backfill parsing, idempotency, and the
get_workout_segments MCP tool."""
import pytest

from health_advisor import backfill, db


XML = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="20.0"
          durationUnit="min" sourceName="Watch"
          startDate="2026-07-01 06:00:00 -0400" endDate="2026-07-01 06:20:00 -0400">
  <WorkoutEvent type="HKWorkoutEventTypeSegment" date="2026-07-01 06:00:00 -0400"
                duration="10.0" durationUnit="min"/>
  <WorkoutEvent type="HKWorkoutEventTypeSegment" date="2026-07-01 06:10:00 -0400"
                duration="9.5" durationUnit="min"/>
  <WorkoutEvent type="HKWorkoutEventTypeLap" date="2026-07-01 06:05:00 -0400"
                duration="300.0" durationUnit="sec"/>
  <WorkoutEvent type="HKWorkoutEventTypePause" date="2026-07-01 06:10:00 -0400"/>
  <WorkoutEvent type="HKWorkoutEventTypeResume" date="2026-07-01 06:10:30 -0400"/>
  <WorkoutEvent type="HKWorkoutEventTypeMotionPaused" date="2026-07-01 06:03:00 -0400"/>
  <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" sum="180"/>
 </Workout>
 <Workout workoutActivityType="HKWorkoutActivityTypeYoga" duration="30.0"
          durationUnit="min" sourceName="Watch"
          startDate="2026-07-01 08:00:00 -0400" endDate="2026-07-01 08:30:00 -0400"/>
 <Record type="HKQuantityTypeIdentifierStepCount" value="100" unit="count"
         sourceName="Watch" startDate="2026-07-01 07:00:00 -0400"
         endDate="2026-07-01 07:10:00 -0400"/>
</HealthData>
"""


@pytest.fixture
def backfilled(tmp_path, vault_path):
    xml = tmp_path / "export.xml"
    xml.write_text(XML)
    dbp = vault_path
    summary = backfill.run(xml_path=str(xml), db_path=str(dbp))
    return dbp, summary


def test_backfill_parses_workout_events(backfilled):
    dbp, summary = backfilled
    assert summary["workout_events_seen"] == 6
    assert summary["workout_events_added"] == 6
    conn = db.connect(dbp, read_only=True)
    rows = conn.execute(
        "SELECT event_type, start_utc, duration_min FROM workout_events "
        "ORDER BY start_utc, event_type").fetchall()
    conn.close()
    by_type = {}
    for r in rows:
        by_type.setdefault(r["event_type"], []).append(r)
    assert len(by_type["segment"]) == 2
    assert by_type["segment"][0]["duration_min"] == pytest.approx(10.0)
    # sec durations convert to minutes
    assert by_type["lap"][0]["duration_min"] == pytest.approx(5.0)
    # instantaneous events keep NULL duration, camelCase converts to snake_case
    assert by_type["pause"][0]["duration_min"] is None
    assert "motion_paused" in by_type


def test_events_link_to_parent_workout(backfilled):
    dbp, _ = backfilled
    conn = db.connect(dbp, read_only=True)
    n = conn.execute(
        "SELECT COUNT(*) FROM workout_events e JOIN workouts w "
        "ON w.dedupe_key = e.workout_key WHERE w.workout_type = 'running'"
    ).fetchone()[0]
    conn.close()
    assert n == 6


def test_backfill_rerun_adds_no_events(backfilled, tmp_path):
    dbp, _ = backfilled
    summary2 = backfill.run(xml_path=str(tmp_path / "export.xml"), db_path=str(dbp))
    assert summary2["workout_events_added"] == 0


def test_workouts_only_skips_records_and_daily_metrics(tmp_path):
    xml = tmp_path / "export.xml"
    xml.write_text(XML)
    dbp = tmp_path / "wo.db"
    summary = backfill.run(xml_path=str(xml), db_path=str(dbp), workouts_only=True)
    assert summary["workout_events_added"] == 6
    assert summary["records_seen"] == 0
    conn = db.connect(dbp, read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()[0] == 0
    conn.close()


def _seed_hr(conn, start_hhmmss_utc: str, values: list[float]):
    """One heart_rate record per value, 30s apart from start (2026-07-01 UTC)."""
    from datetime import datetime, timedelta, timezone
    t = datetime.fromisoformat(f"2026-07-01T{start_hhmmss_utc}+00:00")
    rows = []
    for i, v in enumerate(values):
        s = (t + timedelta(seconds=30 * i))
        iso = s.astimezone(timezone.utc).isoformat()
        rows.append(dict(metric="heart_rate", value=v, unit="count/min",
                         start_utc=iso, end_utc=iso, start_local=None,
                         local_date="2026-07-01", source="Watch", origin="receiver",
                         dedupe_key=db.record_key("heart_rate", iso, iso, v, "count/min", "W")))
    db.insert_records(conn, rows)
    conn.commit()


@pytest.fixture
def tool_db(backfilled):
    dbp, _ = backfilled
    conn = db.connect(dbp)
    # HR samples inside the first segment (10:00:00Z-10:10:00Z): avg 150, max 160
    _seed_hr(conn, "10:00:00", [140, 150, 160])
    return dbp


def test_get_workout_segments_tool(tool_db, tools):
    out = tools.get_workout_segments("2026-07-01")
    assert out["count"] == 2
    run = next(w for w in out["workouts"] if w["type"] == "running")
    # The 06:05 lap sits inside the 06:00 and 06:10 segments — a second
    # partition of the same 20 minutes, not a third split of it.
    assert run["n_segments"] == 2
    assert run["covered_min"] == pytest.approx(19.5)
    assert [a["n_segments"] for a in run["alternate_segmentations"]] == [1]
    seg1 = run["segments"][0]
    assert seg1["type"] == "segment"
    assert seg1["duration_min"] == pytest.approx(10.0)
    assert seg1["avg_heart_rate"] == 150
    assert seg1["max_heart_rate"] == 160
    assert {p["event"] for p in run["pauses"]} == {"pause", "resume"}
    assert run["auto_pause_count"] == 1
    yoga = next(w for w in out["workouts"] if w["type"] == "yoga")
    assert yoga["n_segments"] == 0
    assert "no segments stored" in yoga["note"]


def test_get_workout_segments_type_filter_and_empty_day(tool_db, tools):
    out = tools.get_workout_segments("2026-07-01", workout_type="yoga")
    assert out["count"] == 1 and out["workouts"][0]["type"] == "yoga"
    empty = tools.get_workout_segments("2026-07-02")
    assert empty["count"] == 0 and "no workouts" in empty["note"]
    bad = tools.get_workout_segments("july 1")
    assert "error" in bad


def test_list_workouts_reports_n_segments(tool_db, tools):
    out = tools.list_workouts(start="2026-07-01", end="2026-07-01")
    by_type = {w["type"]: w for w in out["workouts"]}
    # Same count the detail tool gives, so the two tools can't contradict.
    assert by_type["running"]["n_segments"] == 2
    assert by_type["yoga"]["n_segments"] == 0
