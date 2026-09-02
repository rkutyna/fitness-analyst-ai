"""weather — join workout GPS routes to historical conditions.

The point of this module is that heat claims about the athlete's training were called
"unmeasurable" on 2026-08-15 and were not: every workout carries a route_ref,
and data/routes/ holds a 1 Hz GPX. See
docs/audits/results/AUDIT-1-race-reality-2026-08-15.md section 6.

Two things these tests defend that are easy to get wrong:
  - coordinates are ROUNDED before they leave the machine;
  - a long session is sampled more than once, because a 214-minute hike crosses
    real weather and a 35-minute run does not.
"""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import db as dbmod
from health_advisor import weather as wx


GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="42.42787776784456" lon="-71.18691461662054">
      <time>2026-08-15T16:21:28+00:00</time>
    </trkpt>
    <trkpt lat="42.43101010101010" lon="-71.19000000000000">
      <time>2026-08-15T16:51:28+00:00</time>
    </trkpt>
    <trkpt lat="42.44000000000000" lon="-71.20000000000000">
      <time>2026-08-15T17:25:28+00:00</time>
    </trkpt>
  </trkseg></trk>
</gpx>
"""


@pytest.fixture()
def routes_dir(tmp_path):
    d = tmp_path / "routes"
    d.mkdir()
    (d / "run.gpx").write_text(GPX)
    return d


# --------------------------------------------------------------------------- #
# reading the track
# --------------------------------------------------------------------------- #
def test_sample_points_returns_first_point_for_a_short_workout(routes_dir):
    pts = wx.sample_points(routes_dir / "run.gpx", duration_min=35.0)
    assert len(pts) == 1
    assert pts[0].offset_min == 0
    assert pts[0].time_utc == "2026-08-15T16:21:28+00:00"


def test_sample_points_adds_a_sample_every_30_minutes_for_long_workouts(routes_dir):
    pts = wx.sample_points(routes_dir / "run.gpx", duration_min=69.0)
    assert [p.offset_min for p in pts] == [0, 30]
    # each sample takes the trackpoint nearest its mark
    assert pts[1].time_utc == "2026-08-15T16:51:28+00:00"


def test_a_trailing_mark_near_the_end_is_dropped(routes_dir):
    """A mark is kept only if half a sampling interval remains after it.
    Otherwise a 35-minute run gets a second sample five minutes from the end,
    describing air the first sample already described."""
    assert [p.offset_min for p in wx.sample_points(routes_dir / "run.gpx", 35.0)] == [0]
    assert [p.offset_min for p in wx.sample_points(routes_dir / "run.gpx", 44.0)] == [0]
    assert [p.offset_min for p in wx.sample_points(routes_dir / "run.gpx", 45.0)] == [0, 30]


def test_a_long_hike_is_sampled_across_its_whole_span(routes_dir):
    """The 214-minute Jul 8 hike is the case this exists for."""
    offsets = [p.offset_min for p in wx.sample_points(routes_dir / "run.gpx", 214.0)]
    assert offsets == [0, 30, 60, 90, 120, 150, 180]


def test_sample_points_on_a_missing_file_returns_nothing(tmp_path):
    assert wx.sample_points(tmp_path / "nope.gpx", duration_min=40.0) == []


def test_a_route_file_declaring_entities_is_refused(tmp_path):
    """Route files are our own output, but ElementTree expands internal
    entities and a real GPX has no DTD, so refusing one costs nothing."""
    p = tmp_path / "bomb.gpx"
    p.write_text(
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE gpx [<!ENTITY a "aaaaaaaaaa">]>\n'
        '<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        '<trkpt lat="42.4" lon="-71.2"><time>2026-08-15T16:00:00+00:00</time></trkpt>'
        '</trkseg></trk></gpx>'
    )
    assert wx.sample_points(p, duration_min=40.0) == []


def test_sample_points_on_a_gpx_without_trackpoints_returns_nothing(tmp_path):
    p = tmp_path / "empty.gpx"
    p.write_text('<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"></gpx>')
    assert wx.sample_points(p, duration_min=40.0) == []


# --------------------------------------------------------------------------- #
# privacy: what leaves the machine
# --------------------------------------------------------------------------- #
def test_coordinates_are_rounded_before_they_leave_the_machine(routes_dir):
    """ERA5's grid is ~9 km, coarser than 0.1 deg, so rounding costs nothing --
    and it means the request carries a metro cell, not a street."""
    pts = wx.sample_points(routes_dir / "run.gpx", duration_min=35.0)
    lat, lon = wx.coarsen(pts[0].lat, pts[0].lon)
    assert (lat, lon) == (42.4, -71.2)
    assert wx.COORD_PRECISION == 1


def test_the_request_url_never_contains_full_precision(routes_dir):
    pts = wx.sample_points(routes_dir / "run.gpx", duration_min=35.0)
    url = wx.archive_url(pts[0].lat, pts[0].lon, "2026-08-15")
    assert "42.42787" not in url and "-71.18691" not in url
    assert "latitude=42.4" in url and "longitude=-71.2" in url


# --------------------------------------------------------------------------- #
# picking the hour out of the response
# --------------------------------------------------------------------------- #
PAYLOAD = {
    "hourly": {
        "time": ["2026-08-15T15:00", "2026-08-15T16:00", "2026-08-15T17:00"],
        "temperature_2m": [76.0, 78.0, 79.0],
        "relative_humidity_2m": [44, 38, 36],
        "dew_point_2m": [52.0, 50.3, 49.0],
        "wind_speed_10m": [7.0, 8.2, 9.0],
    }
}


def test_reading_conditions_picks_the_hour_containing_the_sample():
    c = wx.conditions_at(PAYLOAD, "2026-08-15T16:21:28+00:00")
    assert c["dew_point_f"] == 50.3
    assert c["temp_f"] == 78.0
    assert c["humidity_pct"] == 38


def test_reading_conditions_truncates_rather_than_rounding_the_hour():
    """16:51 is still the 16:00 observation, not the 17:00 one."""
    assert wx.conditions_at(PAYLOAD, "2026-08-15T16:51:28+00:00")["dew_point_f"] == 50.3


def test_reading_conditions_returns_none_when_the_hour_is_absent():
    assert wx.conditions_at(PAYLOAD, "2026-08-15T23:00:00+00:00") is None


def test_reading_conditions_survives_a_null_in_the_series():
    payload = {"hourly": {"time": ["2026-08-15T16:00"], "temperature_2m": [None],
                          "relative_humidity_2m": [38], "dew_point_2m": [50.3],
                          "wind_speed_10m": [8.2]}}
    c = wx.conditions_at(payload, "2026-08-15T16:10:00+00:00")
    assert c["temp_f"] is None and c["dew_point_f"] == 50.3


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(tmp_path / "t.db")
    dbmod.init_db(c)
    c.execute(
        "INSERT INTO workouts (id, workout_type, start_utc, end_utc, local_date, "
        "duration_min, route_ref, dedupe_key) VALUES "
        "(1,'running','2026-08-15T16:21:28+00:00','2026-08-15T17:30:16+00:00',"
        "'2026-08-15',68.8,'run.gpx','k1')"
    )
    c.commit()
    return c


def test_upsert_is_idempotent(conn):
    row = dict(workout_id=1, offset_min=0, lat=42.4, lon=-71.2,
               observed_utc="2026-08-15T16:00:00+00:00", temp_f=78.0,
               humidity_pct=38, dew_point_f=50.3, wind_kmh=8.2,
               source="open-meteo-era5", fetched_utc="2026-08-16T00:00:00+00:00")
    wx.upsert_weather(conn, [row])
    wx.upsert_weather(conn, [row])
    assert conn.execute("SELECT COUNT(*) FROM workout_weather").fetchone()[0] == 1


def test_upsert_replaces_on_refetch(conn):
    base = dict(workout_id=1, offset_min=0, lat=42.4, lon=-71.2,
                observed_utc="2026-08-15T16:00:00+00:00", temp_f=None,
                humidity_pct=None, dew_point_f=None, wind_kmh=None,
                source="open-meteo-era5", fetched_utc="2026-08-16T00:00:00+00:00")
    wx.upsert_weather(conn, [base])
    wx.upsert_weather(conn, [{**base, "dew_point_f": 50.3,
                              "fetched_utc": "2026-08-21T00:00:00+00:00"}])
    rows = conn.execute("SELECT dew_point_f, fetched_utc FROM workout_weather").fetchall()
    assert len(rows) == 1
    assert rows[0]["dew_point_f"] == 50.3

def test_a_pending_row_is_distinguishable_from_a_missing_one(conn):
    """ERA5 lags ~5 days. 'asked, not yet available' is a null reading with a
    non-null fetched_utc -- otherwise the backfill re-asks forever or gives up."""
    wx.upsert_weather(conn, [dict(
        workout_id=1, offset_min=0, lat=42.4, lon=-71.2,
        observed_utc="2026-08-15T16:00:00+00:00", temp_f=None, humidity_pct=None,
        dew_point_f=None, wind_kmh=None, source="open-meteo-era5",
        fetched_utc="2026-08-16T00:00:00+00:00")])
    assert wx.pending_workout_ids(conn) == [1]
    wx.upsert_weather(conn, [dict(
        workout_id=1, offset_min=0, lat=42.4, lon=-71.2,
        observed_utc="2026-08-15T16:00:00+00:00", temp_f=78.0, humidity_pct=38,
        dew_point_f=50.3, wind_kmh=8.2, source="open-meteo-era5",
        fetched_utc="2026-08-21T00:00:00+00:00")])
    assert wx.pending_workout_ids(conn) == []


def test_deleting_a_workout_takes_its_weather_with_it(conn):
    wx.upsert_weather(conn, [dict(
        workout_id=1, offset_min=0, lat=42.4, lon=-71.2,
        observed_utc="2026-08-15T16:00:00+00:00", temp_f=78.0, humidity_pct=38,
        dew_point_f=50.3, wind_kmh=8.2, source="open-meteo-era5",
        fetched_utc="2026-08-16T00:00:00+00:00")])
    conn.execute("DELETE FROM workouts WHERE id = 1")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM workout_weather").fetchone()[0] == 0


# --------------------------------------------------------------------------- #
# reading it back
# --------------------------------------------------------------------------- #
def test_for_workout_reports_the_session_summary(conn):
    rows = [dict(workout_id=1, offset_min=o, lat=42.4, lon=-71.2,
                 observed_utc=f"2026-08-15T{16+i}:00:00+00:00", temp_f=t,
                 humidity_pct=h, dew_point_f=d, wind_kmh=8.0,
                 source="open-meteo-era5", fetched_utc="2026-08-21T00:00:00+00:00")
            for i, (o, t, h, d) in enumerate([(0, 78.0, 38, 50.3), (30, 80.0, 36, 51.5)])]
    wx.upsert_weather(conn, rows)
    got = wx.for_workout(conn, 1)
    assert got["n_samples"] == 2
    assert got["dew_point_f"] == pytest.approx(50.9, abs=0.05)   # mean over the session
    assert got["temp_f_max"] == 80.0


def test_for_workout_returns_none_when_nothing_was_stored(conn):
    assert wx.for_workout(conn, 1) is None


def test_for_workout_ignores_a_pending_row(conn):
    wx.upsert_weather(conn, [dict(
        workout_id=1, offset_min=0, lat=42.4, lon=-71.2,
        observed_utc="2026-08-15T16:00:00+00:00", temp_f=None, humidity_pct=None,
        dew_point_f=None, wind_kmh=None, source="open-meteo-era5",
        fetched_utc="2026-08-16T00:00:00+00:00")])
    assert wx.for_workout(conn, 1) is None
