"""Outdoor conditions during a workout, joined through its GPX track.

WHY THIS EXISTS. Audit part 1 (2026-08-15) recorded "no weather data of any kind
in health.db" and concluded the plan's heat claims were unmeasurable. That was
wrong, and withdrawn the same day: `workouts.route_ref` points at a 1 Hz GPX in
data/routes/, and a trackpoint carries position and time, which is everything a
historical weather API needs. See
docs/audits/results/AUDIT-1-race-reality-2026-08-15.md section 6.

DEW POINT IS THE POINT. Temperature is the number people quote and humidity is
the number that does the work; dew point combines them into the one that
tracks heat strain. The audit's prototype found r=0.82 between dew point and
cardiac drift across five pace-matched sessions -- five points and no
controls, so a lead rather than a result, but it is also why "run early on hot
days" may be backwards in some climates: in a New England August the
temperature swings across the day and the dew point barely moves.

WHAT THIS CANNOT DO. ERA5 is a ~9 km reanalysis grid, not a weather station at
the athlete's shoulder. It will not see shade, a breeze off a pond, or the
difference between asphalt and trail. It is a description of the air mass, and
the right resolution for "was that session hot" -- not for explaining a single
bad mile.

PRIVACY. Coordinates are rounded to COORD_PRECISION before they are stored or
sent. ERA5's own grid is coarser than the rounding, so the request loses no
accuracy and carries a metro-area cell rather than a street address.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SOURCE = "open-meteo-era5"

#: Decimal places kept on latitude/longitude. 1 dp is ~11 km at this latitude,
#: coarser than ERA5's own grid, so rounding costs no accuracy at all.
COORD_PRECISION = 1

#: A workout longer than this gets more than one sample.
SAMPLE_EVERY_MIN = 30

#: Fields requested, in the order the API returns them.
HOURLY_FIELDS = ("temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m")

_FIELD_MAP = {
    "temp_f": "temperature_2m",
    "humidity_pct": "relative_humidity_2m",
    "dew_point_f": "dew_point_2m",
    "wind_kmh": "wind_speed_10m",
}

_GPX_NS = {"g": "http://www.topografix.com/GPX/1/1"}


@dataclass(frozen=True)
class TrackSample:
    """One point we will ask the weather about."""
    offset_min: int
    lat: float
    lon: float
    time_utc: str


# --------------------------------------------------------------------------- #
# Reading the track
# --------------------------------------------------------------------------- #
def _reject_doctype(raw: bytes) -> None:
    """Refuse a route file carrying a DTD.

    These GPX files are written by routes.py from the receiver payload, so they
    are our own output rather than untrusted input, and stdlib ElementTree does
    not fetch external entities. What it does do is expand *internal* ones,
    which is the billion-laughs footgun. A real GPX track has no DTD, so
    refusing one outright costs nothing and needs no new dependency.
    """
    head = raw[:4096].lstrip()
    lowered = head.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ET.ParseError("route files must not declare a DTD or entities")


def _trackpoints(gpx_path: Path) -> list[tuple[datetime, float, float]]:
    """Every (time, lat, lon) in the file, in order. Points without a <time>
    are skipped -- we cannot ask about a moment we do not have."""
    try:
        raw = Path(gpx_path).read_bytes()
        _reject_doctype(raw)
        tree = ET.ElementTree(ET.fromstring(raw))
    except (OSError, ET.ParseError, ValueError):
        return []
    out: list[tuple[datetime, float, float]] = []
    for pt in tree.iter():
        if not pt.tag.endswith("trkpt"):
            continue
        lat, lon = pt.get("lat"), pt.get("lon")
        if lat is None or lon is None:
            continue
        stamp = None
        for child in pt:
            if child.tag.endswith("time") and child.text:
                stamp = child.text.strip()
                break
        if stamp is None:
            continue
        try:
            when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        out.append((when.astimezone(timezone.utc), float(lat), float(lon)))
    out.sort(key=lambda r: r[0])
    return out


def sample_points(gpx_path: str | Path, duration_min: float) -> list[TrackSample]:
    """Points to ask the weather about: the first, then one every
    SAMPLE_EVERY_MIN of elapsed time, each taking the trackpoint nearest its
    mark. A 35-minute run yields one sample; a 214-minute hike yields eight,
    because that hike crosses real weather and the run does not."""
    points = _trackpoints(Path(gpx_path))
    if not points:
        return []
    start = points[0][0]
    # A mark is kept only if at least half a sampling interval of the session
    # remains after it. Without that, a 35-minute run picks up a second sample
    # five minutes from the end, describing air the first sample already
    # described -- two rows, one observation, and a mean that double-counts the
    # end of the run.
    duration = max(float(duration_min), 0.0)
    marks = [0] + [o for o in range(SAMPLE_EVERY_MIN, int(duration) + 1, SAMPLE_EVERY_MIN)
                   if duration - o >= SAMPLE_EVERY_MIN / 2]
    out: list[TrackSample] = []
    for offset in marks:
        target = start + timedelta(minutes=offset)
        when, lat, lon = min(points, key=lambda r: abs((r[0] - target).total_seconds()))
        out.append(TrackSample(
            offset_min=offset, lat=lat, lon=lon,
            time_utc=when.isoformat(),
        ))
    return out


def coarsen(lat: float, lon: float) -> tuple[float, float]:
    """Round a coordinate to what actually leaves the machine."""
    return round(lat, COORD_PRECISION), round(lon, COORD_PRECISION)


# --------------------------------------------------------------------------- #
# Asking
# --------------------------------------------------------------------------- #
def archive_url(lat: float, lon: float, day: str) -> str:
    """Build the request. Coordinates are coarsened HERE rather than by the
    caller, so there is no path that sends full precision by forgetting to."""
    lat, lon = coarsen(lat, lon)
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": day, "end_date": day,
        "hourly": ",".join(HOURLY_FIELDS),
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    })
    return f"{ARCHIVE_URL}?{query}"


def fetch_day(lat: float, lon: float, day: str, *, timeout: float = 30.0) -> dict | None:
    """One day of hourly conditions, or None if the call fails. Network errors
    are not exceptional here -- ERA5 lags ~5 days and a recent workout simply
    is not published yet -- so the caller records a pending row and moves on."""
    try:
        with urllib.request.urlopen(archive_url(lat, lon, day), timeout=timeout) as fh:
            return json.load(fh)
    except Exception:
        return None


def conditions_at(payload: dict, time_utc: str) -> dict[str, Any] | None:
    """Pull the hour containing `time_utc` out of an archive response.

    Truncates rather than rounds: 16:51 is described by the 16:00 observation,
    not the 17:00 one, because an hourly reanalysis value labels the hour it
    opens rather than the instant nearest it.
    """
    hourly = (payload or {}).get("hourly") or {}
    times = hourly.get("time") or []
    when = datetime.fromisoformat(time_utc.replace("Z", "+00:00"))
    key = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00")
    try:
        i = times.index(key)
    except ValueError:
        return None
    out: dict[str, Any] = {}
    for name, field in _FIELD_MAP.items():
        series = hourly.get(field) or []
        out[name] = series[i] if i < len(series) else None
    out["observed_utc"] = when.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0).isoformat()
    return out


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
_COLUMNS = ("workout_id", "offset_min", "lat", "lon", "observed_utc",
            "temp_f", "humidity_pct", "dew_point_f", "wind_kmh",
            "source", "fetched_utc")


def upsert_weather(conn, rows: Iterable[dict]) -> int:
    """Insert or replace rows keyed on (workout_id, offset_min). Idempotent, so
    a re-fetch after ERA5 catches up overwrites the pending row in place."""
    payload = [tuple(r.get(c) for c in _COLUMNS) for r in rows]
    if not payload:
        return 0
    placeholders = ", ".join("?" * len(_COLUMNS))
    conn.executemany(
        f"INSERT OR REPLACE INTO workout_weather ({', '.join(_COLUMNS)}) "
        f"VALUES ({placeholders})",
        payload,
    )
    conn.commit()
    return len(payload)


def pending_workout_ids(conn) -> list[int]:
    """Workouts we asked about and got nothing for -- ERA5 had not published
    the day yet. These are what a re-run should retry, and the reason a failed
    fetch writes a row at all instead of leaving a hole."""
    rows = conn.execute(
        "SELECT DISTINCT workout_id FROM workout_weather "
        " WHERE dew_point_f IS NULL AND temp_f IS NULL "
        " ORDER BY workout_id"
    ).fetchall()
    return [r[0] for r in rows]


def unfetched_workouts(conn, *, since: str | None = None) -> list[dict]:
    """Route-bearing workouts with no weather row at all, plus any still
    pending. `since` is a local_date lower bound."""
    clauses = ["w.route_ref IS NOT NULL"]
    params: list[Any] = []
    if since:
        clauses.append("w.local_date >= ?")
        params.append(since)
    rows = conn.execute(
        "SELECT w.id, w.local_date, w.start_utc, w.duration_min, w.route_ref "
        "  FROM workouts w "
        "  LEFT JOIN workout_weather x "
        "    ON x.workout_id = w.id AND x.dew_point_f IS NOT NULL "
        f" WHERE {' AND '.join(clauses)} AND x.workout_id IS NULL "
        " ORDER BY w.start_utc",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def for_workout(conn, workout_id: int) -> dict[str, Any] | None:
    """Session-level summary: means over the samples, plus the hottest and the
    muggiest moment. Returns None when nothing usable was stored -- a pending
    row is not an answer."""
    rows = conn.execute(
        "SELECT temp_f, humidity_pct, dew_point_f, wind_kmh FROM workout_weather "
        " WHERE workout_id = ? AND dew_point_f IS NOT NULL",
        (workout_id,),
    ).fetchall()
    if not rows:
        return None

    def mean(col):
        vals = [r[col] for r in rows if r[col] is not None]
        return sum(vals) / len(vals) if vals else None

    def biggest(col):
        vals = [r[col] for r in rows if r[col] is not None]
        return max(vals) if vals else None

    return {
        "n_samples": len(rows),
        "temp_f": mean("temp_f"),
        "temp_f_max": biggest("temp_f"),
        "humidity_pct": mean("humidity_pct"),
        "dew_point_f": mean("dew_point_f"),
        "dew_point_f_max": biggest("dew_point_f"),
        "wind_kmh": mean("wind_kmh"),
    }


# --------------------------------------------------------------------------- #
# The one interpretive helper
# --------------------------------------------------------------------------- #
#: Dew point bands, in Fahrenheit. These are the conventional comfort
#: descriptors used in US weather reporting, not a physiological threshold, and
#: they ship as labels rather than as anything the plan is allowed to act on.
DEW_POINT_BANDS = ((55, "dry"), (60, "comfortable"), (65, "sticky"),
                   (70, "humid"), (75, "oppressive"))


def dew_point_label(dew_point_f: float | None) -> str | None:
    """Plain-language band for a dew point. Descriptive only."""
    if dew_point_f is None:
        return None
    for ceiling, label in DEW_POINT_BANDS:
        if dew_point_f < ceiling:
            return label
    return "dangerous"
