"""The monthly treadmill benchmark is a small, comparable time series."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from health_advisor import benchmark, db


def _record_hr(conn, start: datetime, values: list[float], local_date: str) -> None:
    rows = []
    for i, value in enumerate(values):
        ts = (start + timedelta(seconds=20 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(("heart_rate", value, "count/min", ts, ts, local_date,
                     f"heart-rate|{ts}|{i}"))
    conn.executemany(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, local_date, "
        "source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, 'test', ?)", rows,
    )
    conn.commit()


def _treadmill_workout(conn, local_date: str, start: str) -> None:
    db.insert_workouts(conn, [{
        "workout_type": "running",
        "start_utc": start,
        "end_utc": (datetime.fromisoformat(start.replace("Z", "+00:00"))
                    + timedelta(minutes=35)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_date": local_date,
        "duration_min": 35.0,
        "distance_mi": 0.0,
        "energy_kcal": None,
        "unit_distance": "mi",
        "source": "GymKit",
        "route_ref": None,
        "dedupe_key": f"benchmark-{local_date}",
    }])
    conn.commit()


def test_four_stage_run_round_trips_and_series_aligns_dates(conn):
    for stage, pace in enumerate(("15:00", "14:00", "13:00", "12:00"), 1):
        benchmark.record(
            conn, date="2026-08-25", stage=stage, pace=pace,
            median_hr_last_two_min=130 + stage, talk_test="comfortable",
            temp_c=22.0, dew_point_c=15.0, notes="same treadmill",
        )
    benchmark.record(conn, date="2026-09-22", stage=1, pace="15:00",
                     median_hr_last_two_min=128, talk_test="comfortable")

    series = benchmark.series(conn)
    assert [(row["date"], row["stage"]) for row in series] == [
        ("2026-08-25", 1), ("2026-08-25", 2),
        ("2026-08-25", 3), ("2026-08-25", 4),
        ("2026-09-22", 1),
    ]
    assert series[0]["pace_min_per_mi"] == pytest.approx(15.0)
    assert series[3]["pace_min_per_mi"] == pytest.approx(12.0)
    assert series[0]["talk_test"] == "comfortable"
    assert series[0]["temp_c"] == pytest.approx(22.0)


def test_stopped_early_stores_completed_stages_only(conn):
    benchmark.record(conn, date="2026-08-25", stage=1, pace="15:00",
                     median_hr_last_two_min=140, talk_test="comfortable")
    benchmark.record(conn, date="2026-08-25", stage=2, pace="14:00",
                     median_hr_last_two_min=151, talk_test="not sure",
                     notes="stopped before stage 3")

    rows = benchmark.series(conn)
    assert [row["stage"] for row in rows] == [1, 2]
    assert all(row["median_hr_last_two_min"] is not None for row in rows)
    assert all(row["median_hr_last_two_min"] != 0 for row in rows)


def test_the_stage_median_comes_from_records_not_from_the_caller(conn):
    local_date = "2026-08-25"
    _treadmill_workout(conn, local_date, "2026-08-25T12:00:00Z")
    # Protocol stage 1 starts after the eight-minute warm-up. The last two
    # minutes are 12:10--12:12; their median is 142.5, not the typed 999.
    _record_hr(conn, datetime(2026, 8, 25, 12, 8, tzinfo=timezone.utc),
               list(range(130, 136)) + [140, 141, 142, 143, 144, 145], local_date)

    benchmark.record(conn, date=local_date, stage=1, pace="15:00",
                     median_hr_last_two_min=999)

    assert benchmark.series(conn)[0]["median_hr_last_two_min"] == pytest.approx(
        142.5, abs=0.5,
    )


def test_typed_median_is_used_only_when_no_raw_records_exist(conn):
    benchmark.record(conn, date="2026-08-25", stage=1, pace="15:00",
                     median_hr_last_two_min=141)
    assert benchmark.series(conn)[0]["median_hr_last_two_min"] == pytest.approx(141)


def test_the_stored_median_says_how_it_was_obtained(conn):
    # A protocol-derived window is an inference about session structure, not a
    # measurement. Storing it indistinguishably from an explicitly-bounded one
    # hands a later reader four numbers that look equally solid.
    local_date = "2026-08-25"
    _treadmill_workout(conn, local_date, "2026-08-25T12:00:00Z")
    _record_hr(conn, datetime(2026, 8, 25, 12, 8, tzinfo=timezone.utc),
               [140] * 12, local_date)

    benchmark.record(conn, date=local_date, stage=1, pace="15:00")
    assert benchmark.series(conn)[0]["median_source"] == "records:protocol"

    benchmark.record(conn, date=local_date, stage=1, pace="15:00",
                     stage_start_utc="2026-08-25T12:08:00Z")
    assert benchmark.series(conn)[0]["median_source"] == "records:explicit"


def test_a_typed_median_is_labelled_as_typed(conn):
    # Python did not own this number. The series must show that, because a
    # typed median is not comparable with a measured one.
    benchmark.record(conn, date="2026-08-25", stage=1, pace="15:00",
                     median_hr_last_two_min=140)
    row = benchmark.series(conn)[0]
    assert row["median_source"] == "typed"
    assert row["median_hr_last_two_min"] == 140


# Issue #81: an empty `benchmark` table must never read as "no run ever
# happened" -- it says only that no stage row is stored, in words, and states
# a deployment-supplied run count beside it or says that count is unknown.


def test_get_benchmark_series_status_names_the_row_count_and_marks_runs_unknown(tools, conn):
    out = tools.get_benchmark_series()
    assert out["count"] == 0
    assert out["status"] == (
        "`benchmark` holds 0 row(s); how many runs have been recorded is "
        "UNKNOWN (no external count was supplied -- this is not evidence of "
        "zero runs).")
    # The status must never claim zero runs happened.
    assert "0 run" not in out["status"]


def test_get_benchmark_series_status_states_a_supplied_run_count(tools, conn):
    benchmark.record(conn, date="2026-08-25", stage=1, pace="15:00",
                     median_hr_last_two_min=140)
    out = tools.get_benchmark_series(runs_recorded=3)
    assert out["count"] == 1
    assert out["status"] == "`benchmark` holds 1 row(s); 3 run(s) recorded."


def test_get_benchmark_series_status_is_not_the_bare_count(tools, conn):
    """Mutation for #81: returning the bare `count` with no `status` string
    is exactly the defect the issue reports (an empty table reading as
    'never run'). This must go red against that regression."""
    out = tools.get_benchmark_series()
    assert "status" in out, (
        "get_benchmark_series returned a bare count with no status string -- "
        "this is issue #81's defect: an empty table now reads as 'never run'"
    )


# Consumer #571: the protocol gained optional stages beyond the fourth, so the
# stage bound is one named constant, enforced identically by record() and by
# the table's CHECK, and an existing vault's old CHECK is migrated.


OLD_BENCHMARK_DDL = """
CREATE TABLE benchmark (
    date TEXT NOT NULL,
    stage INTEGER NOT NULL CHECK (stage BETWEEN 1 AND 4),
    pace_min_per_mi REAL NOT NULL,
    median_hr_last_two_min REAL,
    talk_test TEXT,
    temp_c REAL,
    dew_point_c REAL,
    notes TEXT,
    median_source TEXT,
    PRIMARY KEY (date, stage)
)
"""


def _table_sql(conn) -> str:
    return conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'benchmark'"
    ).fetchone()[0]


def test_record_accepts_every_stage_up_to_the_named_maximum(conn):
    assert benchmark.MAX_STAGE == 8
    for stage in range(1, benchmark.MAX_STAGE + 1):
        benchmark.record(conn, date="2030-01-05", stage=stage, pace="10:00",
                         median_hr_last_two_min=120 + stage)
    assert [r["stage"] for r in benchmark.series(conn)] == list(range(1, 9))


@pytest.mark.parametrize("stage", [0, 9, -1])
def test_record_refuses_a_stage_outside_the_bound(conn, stage):
    with pytest.raises(ValueError, match="between 1 and 8"):
        benchmark.record(conn, date="2030-01-05", stage=stage, pace="10:00",
                         median_hr_last_two_min=120)
    assert benchmark.series(conn) == []


def test_the_table_check_and_the_constant_cannot_drift(conn):
    # A fresh vault's CHECK must name the same bound record() enforces.
    assert f"BETWEEN 1 AND {benchmark.MAX_STAGE}" in _table_sql(conn).upper()
    with pytest.raises(Exception, match="CHECK"):
        conn.execute("INSERT INTO benchmark (date, stage, pace_min_per_mi) "
                     "VALUES ('2030-01-05', ?, 10.0)", (benchmark.MAX_STAGE + 1,))
    conn.execute("INSERT INTO benchmark (date, stage, pace_min_per_mi) "
                 "VALUES ('2030-01-05', ?, 10.0)", (benchmark.MAX_STAGE,))


def test_the_tool_docstring_states_the_same_bound():
    from health_advisor import mcp_server
    assert f"1 to {benchmark.MAX_STAGE}" in mcp_server.record_benchmark.__doc__


def test_an_explicit_window_stage_five_is_stored_as_explicit_records(conn):
    local_date = "2030-01-05"
    start = datetime(2030, 1, 5, 12, 0, tzinfo=timezone.utc)
    _record_hr(conn, start, [150.0] * 6 + [160.0, 162.0, 164.0, 166.0, 168.0, 170.0],
               local_date)
    benchmark.record(conn, date=local_date, stage=5, pace="11:07",
                     median_hr_last_two_min=999,
                     stage_start_utc="2030-01-05T12:00:00Z",
                     stage_end_utc="2030-01-05T12:04:00Z")
    row = benchmark.series(conn)[0]
    assert row["stage"] == 5
    assert row["median_source"] == "records:explicit"
    assert row["median_hr_last_two_min"] != 999


def test_a_vault_with_the_old_check_is_rebuilt_without_losing_a_row(tmp_path):
    path = tmp_path / "old.db"
    c = db.connect(path)
    db.init_db(c)
    c.execute("DROP TABLE benchmark")
    c.execute(OLD_BENCHMARK_DDL)
    c.execute("CREATE INDEX idx_benchmark_date ON benchmark (date)")
    for date in ("2030-01-05", "2030-02-02"):
        for stage in range(1, 5):
            c.execute(
                "INSERT INTO benchmark VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (date, stage, 15.0 - stage, 130.0 + stage, f"talk {stage}",
                 21.5, 12.25, f"note {date} {stage}",
                 ("typed", "records:explicit", "records:protocol", None)[stage - 1]),
            )
    c.commit()
    assert "BETWEEN 1 AND 4" in _table_sql(c)
    with pytest.raises(Exception, match="CHECK"):
        c.execute("INSERT INTO benchmark (date, stage, pace_min_per_mi) "
                  "VALUES ('2030-01-05', 5, 10.0)")
    c.rollback()
    before = c.execute("SELECT rowid, * FROM benchmark ORDER BY rowid").fetchall()
    before = [tuple(r) for r in before]
    c.close()

    c = db.connect(path)
    db.init_db(c)
    after = [tuple(r) for r in
             c.execute("SELECT rowid, * FROM benchmark ORDER BY rowid").fetchall()]
    assert after == before and len(after) == 8
    assert f"BETWEEN 1 AND {benchmark.MAX_STAGE}" in _table_sql(c).upper()
    # the date index and the (date, stage) key survive the rebuild
    assert "idx_benchmark_date" in {
        r[1] for r in c.execute("PRAGMA index_list(benchmark)")}
    with pytest.raises(Exception, match="UNIQUE|PRIMARY"):
        c.execute("INSERT INTO benchmark (date, stage, pace_min_per_mi) "
                  "VALUES ('2030-01-05', 1, 10.0)")
    c.rollback()
    benchmark.record(c, date="2030-02-02", stage=5, pace="11:07",
                     median_hr_last_two_min=150)
    assert (len(benchmark.series(c)), after == before) == (9, True)
    sql_before = _table_sql(c)
    c.close()

    # A second open is a no-op: same DDL, same rows.
    c = db.connect(path)
    db.init_db(c)
    assert _table_sql(c) == sql_before
    assert len(benchmark.series(c)) == 9
    assert [tuple(r) for r in c.execute(
        "SELECT rowid, * FROM benchmark WHERE stage <= 4 ORDER BY rowid")
    ] == before
    c.close()
