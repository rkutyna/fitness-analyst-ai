"""Synthetic acceptance tests for scheduled weather/elevation enrichment."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from health_advisor import db, elevation, weather


def _payload(day: str) -> dict:
    hours = [f"{day}T{hour:02d}:00" for hour in range(24)]
    return {"hourly": {
        "time": hours,
        "temperature_2m": [70.0] * 24,
        "relative_humidity_2m": [40.0] * 24,
        "dew_point_2m": [50.0] * 24,
        "wind_speed_10m": [5.0] * 24,
    }}


def _points() -> dict[str, list[float]]:
    return {
        "t_offset_s": [float(i) for i in range(20)],
        "altitude_m": [100.0 + float(i) for i in range(20)],
        "vertical_accuracy_m": [1.0] * 20,
        "horizontal_accuracy_m": [1.0] * 20,
    }


def _seed_vault(tmp_path: Path):
    conn = db.connect(tmp_path / "synthetic-vault.db")
    db.init_db(conn)
    workouts = [
        (1, "2026-09-01", 95.0, "A"),
        (2, "2026-09-02", 30.0, "B"),
        (3, "2026-09-03", 40.0, "C"),
    ]
    for workout_id, day, duration, key in workouts:
        conn.execute(
            "INSERT INTO workouts (id, workout_type, start_utc, end_utc, local_date, "
            "duration_min, route_ref, dedupe_key) VALUES (?, 'running', ?, ?, ?, ?, NULL, ?)",
            (workout_id, f"{day}T10:00:00+00:00", f"{day}T12:00:00+00:00",
             day, duration, key),
        )
    for workout_id, day in ((1, "2026-09-01"), (3, "2026-09-03")):
        conn.execute(
            "INSERT INTO workout_routes (workout_id, hk_route_uuid, start_utc, end_utc, "
            "n_points, start_lat_1dp, start_lon_1dp, encoding, points) "
            "VALUES (?, ?, ?, ?, 20, 10.0, 20.0, ?, ?)",
            (workout_id, f"route-{workout_id}", f"{day}T10:00:00+00:00",
             f"{day}T12:00:00+00:00", db.ROUTE_POINT_ENCODING,
             db.pack_route_points(_points())),
        )
    conn.commit()
    return conn


def _statuses(conn):
    return {
        row["workout_id"]: (row["status"], row["attempts"])
        for row in conn.execute(
            "SELECT workout_id, status, attempts FROM workout_weather_status"
        )
    }


def test_done_01_synthetic_vault_has_explicit_weather_outcomes(tmp_path):
    conn = _seed_vault(tmp_path)
    calls = []

    def fetch(lat, lon, day):
        calls.append((lat, lon, day))
        return None if day == "2026-09-03" else _payload(day)

    counts = weather.enrich_workouts(conn, fetch=fetch, delay=lambda _seconds: None)
    assert counts == {"fetched": 1, "pending": 1, "no_route": 1,
                      "elevation_filled": 2, "fetch_calls": 2}
    assert calls == [(10.0, 20.0, "2026-09-01"), (10.0, 20.0, "2026-09-03")]
    assert conn.execute(
        "SELECT COUNT(*) FROM workout_weather WHERE workout_id = 1"
    ).fetchone()[0] == 4
    assert [row["offset_min"] for row in conn.execute(
        "SELECT offset_min FROM workout_weather WHERE workout_id = 1 ORDER BY offset_min"
    )] == [0, 30, 60, 90]
    assert all((row["lat"], row["lon"]) == (10.0, 20.0) for row in conn.execute(
        "SELECT lat, lon FROM workout_weather WHERE workout_id = 1"
    ))
    assert conn.execute(
        "SELECT COUNT(*) FROM workout_weather WHERE workout_id = 2"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM workout_weather WHERE workout_id = 3 "
        "AND temp_f IS NULL"
    ).fetchone()[0] == 2
    assert _statuses(conn) == {1: ("fetched", 1), 2: ("no_route", 0), 3: ("pending", 1)}


def test_done_02_pending_retry_is_idempotent_and_counts_attempts(tmp_path):
    conn = _seed_vault(tmp_path)
    available = {"2026-09-01": _payload("2026-09-01"), "2026-09-03": None}
    first_calls = []

    def first_fetch(lat, lon, day):
        first_calls.append((lat, lon, day))
        return available[day]

    weather.enrich_workouts(conn, fetch=first_fetch, delay=lambda _seconds: None)
    available["2026-09-03"] = _payload("2026-09-03")
    second_calls = []

    def second_fetch(lat, lon, day):
        second_calls.append((lat, lon, day))
        return available[day]

    counts = weather.enrich_workouts(conn, fetch=second_fetch, delay=lambda _seconds: None)
    assert second_calls == [(10.0, 20.0, "2026-09-03")]
    assert counts["fetched"] == 1 and counts["fetch_calls"] == 1
    assert _statuses(conn)[1] == ("fetched", 1)
    assert _statuses(conn)[3] == ("fetched", 2)


def test_done_03_same_location_and_day_is_one_archive_call(tmp_path):
    conn = _seed_vault(tmp_path)
    for workout_id, key in ((4, "D"), (5, "E")):
        conn.execute(
            "INSERT INTO workouts (id, workout_type, start_utc, end_utc, local_date, "
            "duration_min, route_ref, dedupe_key) VALUES (?, 'running', ?, ?, ?, 30, NULL, ?)",
            (workout_id, "2026-09-04T10:00:00+00:00", "2026-09-04T11:00:00+00:00",
             "2026-09-04", key),
        )
        conn.execute(
            "INSERT INTO workout_routes (workout_id, hk_route_uuid, start_utc, end_utc, "
            "n_points, start_lat_1dp, start_lon_1dp, encoding, points) "
            "VALUES (?, ?, '2026-09-04T10:00:00+00:00', '2026-09-04T11:00:00+00:00', "
            "20, 10.04, 20.04, ?, ?)",
            (workout_id, f"route-{workout_id}", db.ROUTE_POINT_ENCODING,
             db.pack_route_points(_points())),
        )
    conn.commit()
    calls = []
    counts = weather.enrich_workouts(
        conn, fetch=lambda lat, lon, day: (calls.append((lat, lon, day)) or _payload(day)),
        delay=lambda _seconds: None,
    )
    assert calls.count((10.0, 20.0, "2026-09-04")) == 1
    assert counts["fetch_calls"] == 3  # A, C, and the shared D/E day


def test_done_04_elevation_fill_is_versioned_and_idempotent(tmp_path):
    conn = _seed_vault(tmp_path)
    weather.enrich_workouts(conn, fetch=lambda _lat, _lon, day: _payload(day),
                            delay=lambda _seconds: None)
    for row in conn.execute("SELECT encoding, points, n_points, ascent_m, descent_m, method_version "
                            "FROM workout_routes"):
        expected = elevation.compute_elevation(
            db.decode_route_points(row["points"], row["encoding"], row["n_points"]))
        assert row["ascent_m"] == pytest.approx(expected["ascent_m"], abs=1e-6)
        assert row["descent_m"] == pytest.approx(expected["descent_m"], abs=1e-6)
        assert row["method_version"] == 1
    assert weather.enrich_workouts(conn, fetch=lambda *_args: pytest.fail("no fetch"),
                                   delay=lambda _seconds: None)["elevation_filled"] == 0
    conn.execute("UPDATE workout_routes SET method_version = 0 WHERE id = 1")
    conn.commit()
    assert weather.enrich_workouts(conn, fetch=lambda *_args: pytest.fail("no fetch"),
                                   delay=lambda _seconds: None)["elevation_filled"] == 1


def test_done_05_dry_run_makes_no_fetches_or_writes(tmp_path):
    conn = _seed_vault(tmp_path)
    before = {
        name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        for name in ("workout_weather", "workout_weather_status", "workout_routes")
    }
    counts = weather.enrich_workouts(
        conn,
        fetch=lambda *_args: pytest.fail("dry-run called fetch"),
        delay=lambda _seconds: pytest.fail("dry-run slept"),
        dry_run=True,
    )
    after = {
        name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        for name in before
    }
    assert counts["fetch_calls"] == 0
    assert before == after


def test_done_06_cli_requires_db_and_has_no_database_default(tmp_path, capsys):
    from scripts import backfill_weather

    with pytest.raises(SystemExit) as exc:
        backfill_weather.main([])
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err
    source = Path(backfill_weather.__file__).read_text()
    assert "LOCAL_DB_PATH" not in source
    assert "default=str(LOCAL_DB_PATH)" not in source

    path = tmp_path / "empty-vault.db"
    conn = db.connect(path)
    db.init_db(conn)
    conn.close()
    assert backfill_weather.main(["--db", str(path), "--dry-run"]) == 0
    assert capsys.readouterr().out.strip().count("\n") == 0


def test_done_07_gpx_fallback_works_but_packed_route_wins(tmp_path):
    conn = _seed_vault(tmp_path)
    gpx = tmp_path / "legacy.gpx"
    gpx.write_text(
        '<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        '<trkpt lat="11.1" lon="21.2"><time>2026-09-05T10:00:00+00:00</time></trkpt>'
        '</trkseg></trk></gpx>'
    )
    conn.execute("UPDATE workouts SET route_ref = ? WHERE id = 1", (str(gpx),))
    conn.execute(
        "INSERT INTO workouts (id, workout_type, start_utc, end_utc, local_date, "
        "duration_min, route_ref, dedupe_key) VALUES (6, 'running', ?, ?, ?, 30, ?, 'F')",
        ("2026-09-05T10:00:00+00:00", "2026-09-05T11:00:00+00:00", "2026-09-05", str(gpx)),
    )
    conn.commit()
    calls = []
    weather.enrich_workouts(
        conn, fetch=lambda lat, lon, day: (calls.append((lat, lon, day)) or _payload(day)),
        delay=lambda _seconds: None,
    )
    assert (10.0, 20.0, "2026-09-01") in calls
    assert (11.1, 21.2, "2026-09-05") in calls
    assert tuple(conn.execute(
        "SELECT lat, lon FROM workout_weather WHERE workout_id = 1"
    ).fetchone()) == (10.0, 20.0)
    assert tuple(conn.execute(
        "SELECT lat, lon FROM workout_weather WHERE workout_id = 6"
    ).fetchone()) == (11.1, 21.2)


def test_done_08_result_contains_all_run_counts(tmp_path):
    conn = _seed_vault(tmp_path)
    result = weather.enrich_workouts(conn, fetch=lambda *_args: None,
                                     delay=lambda _seconds: None)
    assert {"fetched", "pending", "no_route", "elevation_filled", "fetch_calls"} <= result.keys()
    assert result["fetched"] == 0 and result["pending"] == 2
    assert result["no_route"] == 1 and result["fetch_calls"] == 2


def test_done_09_mutation_ignoring_packed_routes_is_caught(tmp_path, monkeypatch):
    conn = _seed_vault(tmp_path)
    monkeypatch.setattr(weather, "_workout_samples", lambda *_args: [])
    result = weather.enrich_workouts(conn, fetch=lambda *_args: pytest.fail("mutant fetched"),
                                     delay=lambda _seconds: None)
    assert result["no_route"] == 3
    assert conn.execute(
        "SELECT status FROM workout_weather_status WHERE workout_id = 1"
    ).fetchone()[0] == "no_route"


def _insert_route(conn, workout_id: int, *, start: str, lat: float = 10.0,
                  lon: float = 20.0) -> None:
    conn.execute(
        "INSERT INTO workout_routes (workout_id, hk_route_uuid, start_utc, "
        "end_utc, n_points, start_lat_1dp, start_lon_1dp, encoding, points) "
        "VALUES (?, ?, ?, ?, 20, ?, ?, ?, ?)",
        (workout_id, f"review-route-{workout_id}", start, start,
         lat, lon, db.ROUTE_POINT_ENCODING, db.pack_route_points(_points())),
    )


def test_review_01_no_route_is_reconsidered_when_route_arrives(tmp_path):
    conn = db.connect(tmp_path / "late-route.db")
    db.init_db(conn)
    conn.execute(
        "INSERT INTO workouts (id, workout_type, start_utc, end_utc, local_date, "
        "duration_min, route_ref, dedupe_key) VALUES "
        "(1, 'running', '2026-09-06T10:00:00+00:00', "
        "'2026-09-06T10:30:00+00:00', '2026-09-06', 30, NULL, 'late-route')"
    )
    conn.commit()

    first_calls = []
    first = weather.enrich_workouts(
        conn, fetch=lambda *args: first_calls.append(args) or _payload(args[2]),
        delay=lambda _seconds: None,
    )
    assert first["no_route"] == 1 and first["fetch_calls"] == 0
    assert conn.execute(
        "SELECT status FROM workout_weather_status WHERE workout_id = 1"
    ).fetchone()[0] == "no_route"

    _insert_route(conn, 1, start="2026-09-06T10:00:00+00:00")
    conn.commit()
    second_calls = []
    second = weather.enrich_workouts(
        conn, fetch=lambda *args: second_calls.append(args) or _payload(args[2]),
        delay=lambda _seconds: None,
    )
    assert second_calls == [(10.0, 20.0, "2026-09-06")]
    assert second["fetched"] == 1 and second["fetch_calls"] == 1
    assert conn.execute(
        "SELECT status FROM workout_weather_status WHERE workout_id = 1"
    ).fetchone()[0] == "fetched"
    assert tuple(conn.execute(
        "SELECT lat, lon FROM workout_weather WHERE workout_id = 1 LIMIT 1"
    ).fetchone()) == (10.0, 20.0)

    third_calls = []
    weather.enrich_workouts(
        conn, fetch=lambda *args: third_calls.append(args) or _payload(args[2]),
        delay=lambda _seconds: None,
    )
    assert third_calls == []


def test_review_02_archive_day_is_each_sample_utc_date(tmp_path):
    conn = db.connect(tmp_path / "utc-day.db")
    db.init_db(conn)
    conn.execute(
        "INSERT INTO workouts (id, workout_type, start_utc, end_utc, local_date, "
        "duration_min, route_ref, dedupe_key) VALUES "
        "(1, 'running', '2026-09-01T23:40:00+00:00', "
        "'2026-09-02T00:20:00+00:00', '2026-09-01', 40, NULL, 'utc-1'), "
        "(2, 'running', '2026-09-02T01:00:00+00:00', "
        "'2026-09-02T01:40:00+00:00', '2026-09-01', 40, NULL, 'utc-2')"
    )
    _insert_route(conn, 1, start="2026-09-01T23:40:00+00:00")
    _insert_route(conn, 2, start="2026-09-02T01:00:00+00:00")
    conn.commit()

    calls = []
    weather.enrich_workouts(
        conn,
        fetch=lambda *args: calls.append(args) or _payload(args[2]),
        delay=lambda _seconds: None,
    )
    assert calls == [
        (10.0, 20.0, "2026-09-01"),
        (10.0, 20.0, "2026-09-02"),
    ]
    assert conn.execute(
        "SELECT COUNT(*) FROM workout_weather WHERE temp_f IS NOT NULL"
    ).fetchone()[0] == 4
    assert conn.execute(
        "SELECT COUNT(*) FROM workout_weather WHERE workout_id = 2"
    ).fetchone()[0] == 2


def test_review_03_relative_gpx_never_uses_ambient_paths(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "ambient-path.db")
    db.init_db(conn)
    conn.execute(
        "INSERT INTO workouts (id, workout_type, start_utc, end_utc, local_date, "
        "duration_min, route_ref, dedupe_key) VALUES "
        "(1, 'running', '2026-09-07T10:00:00+00:00', "
        "'2026-09-07T10:30:00+00:00', '2026-09-07', 30, 'ambient.gpx', 'ambient')"
    )
    conn.commit()
    (tmp_path / "ambient.gpx").write_text("not a route")
    monkeypatch.chdir(tmp_path)

    calls = []
    result = weather.enrich_workouts(
        conn, fetch=lambda *args: calls.append(args), delay=lambda _seconds: None,
    )
    assert result["no_route"] == 1 and result["fetch_calls"] == 0
    assert calls == []
    source = Path(weather.__file__).read_text()
    assert "Path.cwd" not in source
    assert '"data" / "routes"' not in source


def test_review_04_legacy_weather_rows_reconcile_without_or_with_route(tmp_path):
    conn = db.connect(tmp_path / "legacy-weather.db")
    db.init_db(conn)
    conn.execute(
        "INSERT INTO workouts (id, workout_type, start_utc, end_utc, local_date, "
        "duration_min, route_ref, dedupe_key) VALUES "
        "(1, 'running', '2026-09-08T10:00:00+00:00', "
        "'2026-09-08T10:30:00+00:00', '2026-09-08', 30, NULL, 'legacy-complete'), "
        "(2, 'running', '2026-09-09T10:00:00+00:00', "
        "'2026-09-09T10:30:00+00:00', '2026-09-09', 30, NULL, 'legacy-pending')"
    )
    _insert_route(conn, 2, start="2026-09-09T10:00:00+00:00")
    weather.upsert_weather(conn, [{
        "workout_id": 1, "offset_min": 0, "lat": 10.0, "lon": 20.0,
        "observed_utc": "2026-09-08T10:00:00+00:00", "temp_f": 70.0,
        "humidity_pct": 40.0, "dew_point_f": 50.0, "wind_kmh": 5.0,
        "source": weather.SOURCE, "fetched_utc": "2026-09-08T12:00:00+00:00",
    }, {
        "workout_id": 2, "offset_min": 0, "lat": 10.0, "lon": 20.0,
        "observed_utc": "2026-09-09T10:00:00+00:00", "temp_f": None,
        "humidity_pct": None, "dew_point_f": None, "wind_kmh": None,
        "source": weather.SOURCE, "fetched_utc": "2026-09-09T12:00:00+00:00",
    }])
    conn.commit()

    calls = []
    result = weather.enrich_workouts(
        conn,
        fetch=lambda *args: calls.append(args) or _payload(args[2]),
        delay=lambda _seconds: None,
    )
    assert calls == [(10.0, 20.0, "2026-09-09")]
    assert result["fetched"] == 2 and result["fetch_calls"] == 1
    assert conn.execute(
        "SELECT status, attempts FROM workout_weather_status WHERE workout_id = 1"
    ).fetchone()[:] == ("fetched", 0)
    assert conn.execute(
        "SELECT status, attempts FROM workout_weather_status WHERE workout_id = 2"
    ).fetchone()[:] == ("fetched", 1)


def test_no_write_transaction_is_open_while_the_archive_is_called(tmp_path):
    """The vault's write lock must never be held across a network call: the
    receiver shares the file and gives up after its 30 s busy_timeout."""
    conn = _seed_vault(tmp_path)
    # Two route workouts on different days, so the second fetch follows the
    # first workout's weather upsert and status write.
    open_during_fetch = []

    def fetch(lat, lon, day):
        open_during_fetch.append(conn.in_transaction)
        return _payload(day)

    weather.enrich_workouts(conn, fetch=fetch, delay=lambda _seconds: None)
    assert len(open_during_fetch) >= 2
    assert open_during_fetch == [False] * len(open_during_fetch)
