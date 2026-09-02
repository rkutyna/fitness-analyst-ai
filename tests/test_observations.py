"""The session-observation instrument: what the user said, beside what Python computed.

These tests were written against a committed fixture of one real week (week 8)
and a one-off loader script that replayed that week's recorded answers. Neither
is part of this repository — the fixture is personal data and the loader is a
historical migration of it. The instrument itself is here, so the observations
below are made through its own public API (`observations.record`, and the CLI)
rather than replayed from a file. The invariants are the ones the week-8 seed
case protected: an observation never touches `records` or `daily_metrics`, a
disagreement carries BOTH figures, coverage separates confirmed from corrected,
and a week line is one per session keyed by a stable workout key.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from health_advisor import observations


ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# A week of statements, in the shape the week-8 seed case had: nine claims,
# seven confirming what Python computed and two correcting it — one with a
# number, one with a structure the user described in words.
# --------------------------------------------------------------------------- #

CONFIRMED = [
    ("2026-08-17", "jogged", 1.0),
    ("2026-08-18", "jog_minutes", 4.0),
    ("2026-08-19", "jogged", 1.0),
    ("2026-08-20", "jog_minutes", 6.5),
    ("2026-08-21", "jogged", 1.0),
    ("2026-08-23", "jogged", 1.0),
    ("2026-08-24", "longest_block_min", 9.3),
]


def _seed_week_of_statements(conn) -> None:
    for day, field, value in CONFIRMED:
        observations.record(
            conn, scope="day", local_date=day, field=field,
            computed_value=value, stated_value=value, agrees=1,
            evidence="recall",
            computed_at="2026-08-28T12:00:00+00:00",
            stated_at=f"{day}T20:00:00+00:00")
    # A correction with no scalar to correct: the user describes the structure
    # of the session in words. The computed figure stays on the row.
    observations.record(
        conn, scope="day", local_date="2026-08-22", field="structure",
        computed_value=10.0, stated_text="10 on 3 off x2", agrees=0,
        evidence="recall",
        computed_at="2026-08-28T12:00:00+00:00",
        stated_at="2026-08-22T20:00:00+00:00")
    # A numeric correction: Python said 22.3 jog minutes, the user says 16.2.
    observations.record(
        conn, scope="day", local_date="2026-08-25", field="jog_minutes",
        computed_value=22.3, stated_value=16.2, agrees=0, evidence="segments",
        computed_at="2026-08-28T12:00:00+00:00",
        stated_at="2026-08-25T20:00:00+00:00")


def _seed_pipeline_rows(conn) -> None:
    """Rows in the two tables an observation must never touch."""
    conn.executemany(
        "INSERT INTO records (metric, start_utc, end_utc, local_date, value, unit,"
        " source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, 'test', ?)",
        [("heart_rate", "2026-08-21T12:00:00Z", "2026-08-21T12:00:00Z",
          "2026-08-21", 138.0, "count/min", "h|1"),
         ("step_count", "2026-08-25T12:00:00Z", "2026-08-25T12:00:20Z",
          "2026-08-25", 47.0, "count", "s|1")])
    conn.executemany(
        "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, unit)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [("jog_minutes", "2026-08-21", 1, 8.0, 8.0, 8.0, 8.0, "min"),
         ("step_count", "2026-08-25", 1, 9001.0, 9001.0, 9001.0, 9001.0, "count")])
    conn.commit()


def _fingerprint(conn):
    """Count AND content for both raw tables.

    A count alone cannot see a row that was MODIFIED rather than added, and on
    an empty table a bare count comparison passes vacuously — which is how the
    first report of this invariant read (0 before, 0 after). Seeding real rows
    and hashing them is what makes it a test.
    """
    out = {}
    for table, cols in (("records", "id, metric, start_utc, value, source"),
                        ("daily_metrics", "metric, date, count, sum, avg")):
        rows = conn.execute(
            f"SELECT {cols} FROM {table} ORDER BY 1, 2").fetchall()
        out[table] = (len(rows), hash(tuple(tuple(r) for r in rows)))
    return out


# --------------------------------------------------------------------------- #
# A synthetic week of sessions, for the week-line computation.
# --------------------------------------------------------------------------- #


def _add_workout(conn, key: str, workout_type: str, start_utc: str,
                 minutes: float) -> None:
    end = (datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
           + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date,"
        " duration_min, source, dedupe_key) VALUES (?, ?, ?, ?, ?, 'test', ?)",
        (workout_type, start_utc, end, start_utc[:10], float(minutes), key))


def _add_buckets(conn, start_utc: str, n: int, miles: float,
                 cadence_spm: float, hr: float | None = None) -> None:
    """`n` consecutive 20-second buckets from `start_utc`.

    A bucket is a jog when its cadence clears the gait threshold (140 spm) and
    it falls inside a workout window, so `cadence_spm=0` gives a walking bucket
    at the same pace. Cadence is carried by `step_count` rows, which
    bucket_series sums per bucket and scales by three.
    """
    t0 = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    for i in range(n):
        ts = (t0 + timedelta(seconds=20 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute(
            "INSERT INTO records (metric, start_utc, end_utc, local_date, value,"
            " unit, source, dedupe_key) VALUES ('distance_walking_running',"
            " ?, ?, ?, ?, 'mi', 'test', ?)",
            (ts, ts, ts[:10], miles, f"d|{ts}"))
        if cadence_spm:
            conn.execute(
                "INSERT INTO records (metric, start_utc, end_utc, local_date,"
                " value, unit, source, dedupe_key) VALUES ('step_count',"
                " ?, ?, ?, ?, 'count', 'test', ?)",
                (ts, ts, ts[:10], cadence_spm / 3.0, f"s|{ts}"))
        if hr is not None:
            conn.execute(
                "INSERT INTO records (metric, start_utc, end_utc, local_date,"
                " value, unit, source, dedupe_key) VALUES ('heart_rate',"
                " ?, ?, ?, ?, 'count/min', 'test', ?)",
                (ts, ts, ts[:10], hr, f"h|{ts}"))


def _seed_week_of_sessions(conn) -> None:
    """Three sessions in the week of Monday 2026-08-17, TWO of them on one day.

    The same-day pair is the point: a day-scoped computation would give both
    Monday lines 7.0 jog minutes, and the whole instrument rests on a line
    describing exactly one session.
    """
    # Monday morning: 15 jog buckets (5.0 min) at HR 140, then a walking tail.
    _add_workout(conn, "w|mon-am", "running", "2026-08-17T12:00:00Z", 10)
    _add_buckets(conn, "2026-08-17T12:00:00Z", 15, 0.0333, 141.0, hr=140.0)
    _add_buckets(conn, "2026-08-17T12:05:00Z", 15, 0.0200, 0.0)
    # Monday evening: a separate, shorter session — 6 jog buckets (2.0 min),
    # with no heart rate, so its block cannot be qualified against the ceiling.
    _add_workout(conn, "w|mon-pm", "running", "2026-08-17T18:00:00Z", 3)
    _add_buckets(conn, "2026-08-17T18:00:00Z", 6, 0.0333, 141.0)
    # Wednesday: a walk. Same pace lane, cadence below the gait threshold.
    _add_workout(conn, "w|wed", "walking", "2026-08-19T12:00:00Z", 5)
    _add_buckets(conn, "2026-08-19T12:00:00Z", 15, 0.0200, 0.0)
    conn.commit()


# --------------------------------------------------------------------------- #
# The invariants
# --------------------------------------------------------------------------- #


def test_recording_observations_never_writes_records_or_daily_metrics(conn):
    """The append-only guarantee, asserted against tables that have rows in them.

    An observation is what the user said; it is not evidence, and it must never
    edit the pipeline's own tables. The seeded rows exist so the comparison is
    not 0 == 0.
    """
    _seed_pipeline_rows(conn)
    before = _fingerprint(conn)
    assert before["records"][0] == 2 and before["daily_metrics"][0] == 2, (
        "the guard must have rows to guard")

    _seed_week_of_statements(conn)

    assert _fingerprint(conn) == before
    assert conn.execute(
        "SELECT COUNT(*) FROM session_observation").fetchone()[0] == 9


def test_disagreements_are_queryable_with_both_figures(conn):
    _seed_week_of_statements(conn)
    rows = observations.disagreements(conn, "2026-08-17", "2026-08-25")
    assert [(row["local_date"], row["computed_value"], row["stated_value"])
            for row in rows] == [
                # Both figures survive on the row: a correction never
                # overwrites what the pipeline computed at the time.
                ("2026-08-22", 10.0, None),
                ("2026-08-25", 22.3, 16.2),
            ]
    assert rows[1]["delta"] == pytest.approx(6.1)   # computed - stated
    assert rows[1]["evidence"] == "segments"
    assert rows[0]["stated_text"] == "10 on 3 off x2"
    assert rows[0]["delta"] is None                 # a structure has no scalar
    # The seven confirmations are not disagreements, and a day nobody was asked
    # about is absent rather than a disagreement.
    assert len(rows) == 2
    assert observations.disagreements(conn, "2026-08-26", "2026-08-31") == []


def test_coverage_distinguishes_confirmed_from_corrected(conn):
    _seed_week_of_statements(conn)
    assert observations.coverage(conn, "2026-08-17", "2026-08-25") == {
        "asked": 9, "confirmed": 7, "corrected": 2,
    }
    confirmed = [row for row in conn.execute(
        "SELECT local_date, agrees FROM session_observation "
        "WHERE field = 'jogged' ORDER BY local_date")]
    assert [(row[0], row[1]) for row in confirmed] == [
        ("2026-08-17", 1), ("2026-08-19", 1),
        ("2026-08-21", 1), ("2026-08-23", 1),
    ]


def test_week_lines_are_one_per_session_and_use_stable_workout_keys(conn):
    _seed_week_of_sessions(conn)
    lines = observations.week_lines(conn, "2026-08-17")

    assert len(lines) == 3
    assert [(line["date"], line["workout_key"], line["type"]) for line in lines] == [
        ("2026-08-17", "w|mon-am", "running"),
        ("2026-08-17", "w|mon-pm", "running"),
        ("2026-08-19", "w|wed", "walking"),
    ]
    # The key is the workout's dedupe key — stable across a re-ingest — not a
    # rowid, which an observation recorded today could not survive.
    assert [line["workout_key"] for line in lines] == [
        row[0] for row in conn.execute(
            "SELECT dedupe_key FROM workouts ORDER BY start_utc")]

    # Per session, not per day: 15 jog buckets and 6 jog buckets at 20 s each.
    # Day-scoped, both Monday lines would read 7.0.
    assert lines[0]["jog_minutes"] == pytest.approx(5.0)
    assert lines[1]["jog_minutes"] == pytest.approx(2.0)
    assert lines[2]["jog_minutes"] == pytest.approx(0.0)   # a walk is not a jog
    assert lines[0]["longest_block_min"] == pytest.approx(5.0)
    assert lines[1]["longest_block_min"] == pytest.approx(2.0)
    assert lines[2]["longest_block_min"] == pytest.approx(0.0)
    # The governing dial needs heart rate; the evening session has none, so it
    # is None rather than silently qualified.
    assert lines[0]["qualified_block_min"] == pytest.approx(5.0)
    assert lines[1]["qualified_block_min"] is None


def test_a_week_line_carries_only_its_own_session_observations(conn):
    _seed_week_of_sessions(conn)
    observations.record(
        conn, scope="session", local_date="2026-08-17",
        workout_key="w|mon-pm", field="jog_minutes",
        computed_value=2.0, stated_value=3.0, agrees=0, evidence="recall")

    lines = {line["workout_key"]: line for line in
             observations.week_lines(conn, "2026-08-17")}
    assert lines["w|mon-pm"]["asked"] is True
    assert [row["stated_value"] for row in lines["w|mon-pm"]["observations"]] == [3.0]
    # The other session that day was not asked about, and must not inherit it.
    assert lines["w|mon-am"]["asked"] is False
    assert lines["w|mon-am"]["observations"] == []


def test_revised_statement_is_a_new_row_and_latest_wins(conn):
    observations.record(
        conn, scope="day", local_date="2026-08-25", field="jog_minutes",
        computed_value=22.3, stated_value=16.2, agrees=0, evidence="segments",
        computed_at="2026-08-28T12:00:00+00:00",
        stated_at="2026-08-28T12:01:00+00:00",
    )
    observations.record(
        conn, scope="day", local_date="2026-08-25", field="jog_minutes",
        computed_value=22.3, stated_value=17.0, agrees=0, evidence="device",
        computed_at="2026-08-28T12:00:00+00:00",
        stated_at="2026-08-29T12:01:00+00:00",
    )
    rows = conn.execute(
        "SELECT stated_value, stated_at FROM session_observation "
        "WHERE local_date = '2026-08-25' ORDER BY stated_at").fetchall()
    assert [(row[0], row[1]) for row in rows] == [
        (16.2, "2026-08-28T12:01:00+00:00"),
        (17.0, "2026-08-29T12:01:00+00:00"),
    ]
    assert observations.disagreements(conn, "2026-08-25", "2026-08-25")[0]["stated_value"] == 17.0


def _cli(vault_path, *args) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "health_advisor.observations",
         "--db", str(vault_path), *args],
        cwd=ROOT, check=True, capture_output=True, text=True)
    return proc.stdout


def test_cli_replay_is_idempotent_and_changes_only_observations(vault_path, conn):
    """Replaying the same statement must not double it, or touch anything else.

    This is what made re-running the week-8 loader safe, and it is a property of
    the INSERT OR IGNORE in `observations._record` plus the CLI's argument
    wiring — not of any particular loader script.
    """
    _seed_pipeline_rows(conn)
    before = _fingerprint(conn)

    record_args = ("record", "--scope", "day", "--local-date", "2026-08-25",
                   "--field", "jog_minutes", "--computed-value", "22.3",
                   "--stated-value", "16.2", "--agrees", "0",
                   "--evidence", "segments",
                   "--computed-at", "2026-08-28T12:00:00+00:00",
                   "--stated-at", "2026-08-28T12:01:00+00:00")
    assert json.loads(_cli(vault_path, *record_args)) == {"recorded": True}
    assert json.loads(_cli(vault_path, *record_args)) == {"recorded": True}

    assert conn.execute(
        "SELECT COUNT(*) FROM session_observation").fetchone()[0] == 1
    assert _fingerprint(conn) == before

    reported = json.loads(_cli(vault_path, "coverage", "2026-08-17", "2026-08-25"))
    assert reported == {"asked": 1, "confirmed": 0, "corrected": 1}


@pytest.mark.parametrize("kwargs", [
    {"scope": "nope", "local_date": "2026-08-25", "field": "jog_minutes"},
    {"scope": "week", "local_date": "2026-08-18", "field": "jog_minutes"},
    {"scope": "session", "local_date": "2026-08-25", "field": "jog_minutes"},
])
def test_validation_is_explicit(kwargs, conn):
    with pytest.raises(ValueError):
        observations.record(conn, **kwargs, stated_value=1, agrees=1, evidence="recall")
