"""One physical workout, two historical ingest sources, two `source` strings.

The full export reports `sourceName="Demo's Apple Watch"`; the retired phone
path reported whatever the nested samples carried — observed in real payloads as
"GymKit", "GymKit|Demo's Apple Watch", and "Demo's iPhone " for the very same
session. The dedupe key must therefore not depend on `source`, or every fresh
export.zip re-adds the workouts the receiver already stored (seen 2026-07-28:
5 duplicate pairs).
"""
from health_advisor import backfill, db


START, END = "2026-06-22 10:24:24 -0400", "2026-06-22 11:00:25 -0400"

XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="36.02"
          durationUnit="min" sourceName="Demo&#8217;s Apple Watch"
          startDate="{START}" endDate="{END}">
  <WorkoutEvent type="HKWorkoutEventTypeSegment" date="2026-06-22 10:24:24 -0400"
                duration="18.0" durationUnit="min"/>
  <WorkoutEvent type="HKWorkoutEventTypeSegment" date="2026-06-22 10:42:24 -0400"
                duration="18.0" durationUnit="min"/>
 </Workout>
</HealthData>
"""


def _receiver_workout(source: str) -> dict:
    """The same historical receiver workout, already in storage shape."""
    start_utc = "2026-06-22T14:24:24+00:00"
    end_utc = "2026-06-22T15:00:25+00:00"
    return {
        "workout_type": "running", "start_utc": start_utc, "end_utc": end_utc,
        "local_date": "2026-06-22", "duration_min": 36.0167,
        "distance_mi": 2.017, "unit_distance": "mi", "energy_kcal": 313.9,
        "avg_heart_rate": 136.6, "max_heart_rate": 171.0, "source": source,
        "dedupe_key": db.workout_key("running", start_utc, end_utc),
    }


def _ingest_receiver_history(dbp, source: str) -> int:
    conn = db.connect(dbp)
    db.init_db(conn)
    added = db.insert_workouts(conn, [_receiver_workout(source)])
    conn.commit()
    conn.close()
    return added


def test_workout_key_separates_distinct_sessions():
    """Dropping source must not collapse genuinely different workouts."""
    base = ("running", "2026-06-22T14:24:24+00:00", "2026-06-22T15:00:25+00:00")
    assert db.workout_key(*base) != db.workout_key("cycling", *base[1:])
    assert db.workout_key(*base) != db.workout_key(
        "running", "2026-06-22T14:30:00+00:00", base[2])
    assert db.workout_key(*base) != db.workout_key(
        "running", base[1], "2026-06-22T15:30:00+00:00")


def test_export_then_receiver_yields_one_workout(tmp_path):
    xml = tmp_path / "export.xml"
    xml.write_text(XML)
    dbp = tmp_path / "health.db"
    backfill.run(xml_path=str(xml), db_path=str(dbp), workouts_only=True)

    assert _ingest_receiver_history(dbp, "GymKit|Demo’s Apple\xa0Watch") == 0

    conn = db.connect(dbp, read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 1
    conn.close()


def test_receiver_then_export_yields_one_workout_keeping_segments(tmp_path):
    """Real arrival order: the phone syncs daily, the export lands later. The
    export must attach its segments to the row the receiver already wrote."""
    xml = tmp_path / "export.xml"
    xml.write_text(XML)
    dbp = tmp_path / "health.db"
    _ingest_receiver_history(dbp, "Demo’s iPhone ")
    summary = backfill.run(xml_path=str(xml), db_path=str(dbp), workouts_only=True)
    assert summary["workouts_added"] == 0

    conn = db.connect(dbp, read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 1
    # the receiver's reconciled HR survives ...
    assert conn.execute("SELECT avg_heart_rate FROM workouts").fetchone()[0] == 136.6
    # ... and the export's segments hang off that same row
    n = conn.execute(
        "SELECT COUNT(*) FROM workout_events e JOIN workouts w "
        "ON w.dedupe_key = e.workout_key").fetchone()[0]
    conn.close()
    assert n == 2
