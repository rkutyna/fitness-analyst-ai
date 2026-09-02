"""Storage and reading for the monthly treadmill benchmark (W7-7)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median


STAGE_MINUTES = 4
WARMUP_MINUTES = 8
RECOVERY_MINUTES = 2


def _pace_minutes(pace: str | float | int) -> float:
    if isinstance(pace, str) and ":" in pace:
        minutes, seconds = pace.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    return float(pace)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _stage_window(conn, date: str, stage: int,
                  stage_start_utc: str | None,
                  stage_end_utc: str | None) -> tuple[datetime, datetime] | None:
    """Find the four-minute stage window from explicit bounds or the protocol.

    The explicit arguments make the raw-data boundary unambiguous for imported
    benchmark logs. The fallback uses the treadmill running workout's start;
    when a test or manual log has no workout row, the first HR sample anchors
    the protocol. The median itself is never supplied by this helper.
    """
    if stage_start_utc:
        start = _parse_timestamp(stage_start_utc)
        end = _parse_timestamp(stage_end_utc) if stage_end_utc else start + timedelta(minutes=STAGE_MINUTES)
        return start, end

    workout = conn.execute(
        "SELECT start_utc FROM workouts WHERE workout_type = 'running' "
        "AND local_date = ? ORDER BY route_ref IS NOT NULL, start_utc LIMIT 1",
        (date,),
    ).fetchone()
    if workout:
        session_start = _parse_timestamp(workout["start_utc"])
    else:
        first = conn.execute(
            "SELECT start_utc FROM records WHERE metric = 'heart_rate' "
            "AND local_date = ? ORDER BY start_utc LIMIT 1", (date,),
        ).fetchone()
        if not first:
            return None
        session_start = _parse_timestamp(first["start_utc"])

    offset = WARMUP_MINUTES + (stage - 1) * (STAGE_MINUTES + RECOVERY_MINUTES)
    start = session_start + timedelta(minutes=offset)
    return start, start + timedelta(minutes=STAGE_MINUTES)


def _median_from_records(conn, date: str, stage: int,
                         stage_start_utc: str | None,
                         stage_end_utc: str | None) -> tuple[float | None, bool]:
    rows = conn.execute(
        "SELECT value, start_utc FROM records WHERE metric = 'heart_rate' "
        "AND local_date = ? ORDER BY start_utc", (date,),
    ).fetchall()
    if not rows:
        return None, False

    window = _stage_window(conn, date, stage, stage_start_utc, stage_end_utc)
    if window is None:
        return None, True
    stage_start, stage_end = window
    final_start = stage_end - timedelta(minutes=2)
    values = [float(row["value"]) for row in rows
              if row["value"] is not None
              and final_start <= _parse_timestamp(row["start_utc"]) < stage_end]
    return (float(median(values)) if values else None), True


def record(conn, *, date: str, stage: int, pace: str | float | int,
           median_hr_last_two_min: float | None = None,
           talk_test: str | None = None, temp_c: float | None = None,
           dew_point_c: float | None = None, notes: str | None = None,
           stage_start_utc: str | None = None,
           stage_end_utc: str | None = None) -> None:
    """Upsert one completed benchmark stage.

    A manually typed median is a fallback for a stage with no raw HR records.
    Whenever records exist, Python recomputes the final-two-minute median and
    ignores the caller's number (the project's "Python owns the truth" rule).
    """
    if not 1 <= int(stage) <= 4:
        raise ValueError("benchmark stage must be between 1 and 4")
    computed, records_exist = _median_from_records(
        conn, date, int(stage), stage_start_utc, stage_end_utc,
    )
    median_hr = computed if records_exist else median_hr_last_two_min
    # Say how the number was obtained. "records:protocol" means the window was
    # inferred from the published stage structure rather than measured, so a
    # session that ran to a different shape yields a plausible wrong median
    # with nothing on its face to show it. "typed" means Python did not own
    # this number at all — which is worth seeing when the series is compared.
    if not records_exist:
        median_source = "typed"
    elif stage_start_utc:
        median_source = "records:explicit"
    else:
        median_source = "records:protocol"
    conn.execute(
        """
        INSERT INTO benchmark
            (date, stage, pace_min_per_mi, median_hr_last_two_min, talk_test,
             temp_c, dew_point_c, notes, median_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, stage) DO UPDATE SET
            pace_min_per_mi = excluded.pace_min_per_mi,
            median_hr_last_two_min = excluded.median_hr_last_two_min,
            talk_test = excluded.talk_test,
            temp_c = excluded.temp_c,
            dew_point_c = excluded.dew_point_c,
            notes = excluded.notes,
            median_source = excluded.median_source
        """,
        (date, int(stage), _pace_minutes(pace), median_hr, talk_test, temp_c,
         dew_point_c, notes, median_source),
    )
    conn.commit()


def series(conn) -> list[dict]:
    """Return completed benchmark stages in date/stage order."""
    rows = conn.execute(
        "SELECT date, stage, pace_min_per_mi, median_hr_last_two_min, talk_test, "
        "temp_c, dew_point_c, notes, median_source "
        "FROM benchmark ORDER BY date, stage"
    ).fetchall()
    return [dict(row) for row in rows]
