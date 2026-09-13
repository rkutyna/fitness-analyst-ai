"""Outdoor conditions during a workout, joined through its route or GPX track.

WHY THIS EXISTS. Workout weather is useful context for heat-strain analysis:
the route start point and workout time are enough to query a historical archive.
Packed route rows are the primary source; legacy GPX references remain a
compatibility fallback.

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
import time
from typing import Any, Callable, Iterable, Sequence

from . import db as dbmod
from . import elevation

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


def route_samples(start_utc: str, duration_min: float, lat: float, lon: float) -> list[TrackSample]:
    """Make the fixed-offset samples for a packed ``workout_routes`` row.

    Unlike the legacy GPX sampler, the route contract has already given us the
    workout's start point. Every offset deliberately keeps that one point; no
    point-level coordinate is ever sent to the archive.
    """
    try:
        start = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return []
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    try:
        duration = max(float(duration_min), 0.0)
        lat, lon = coarsen(float(lat), float(lon))
    except (TypeError, ValueError):
        return []
    marks = range(0, int(duration) + 1, SAMPLE_EVERY_MIN)
    return [TrackSample(
        offset_min=offset,
        lat=lat,
        lon=lon,
        time_utc=(start + timedelta(minutes=offset)).isoformat(),
    ) for offset in marks]


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


_DEFAULT_FETCH_DAY = fetch_day


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
        "SELECT DISTINCT x.workout_id FROM workout_weather x "
        " LEFT JOIN workout_weather_status s ON s.workout_id = x.workout_id "
        " WHERE s.status = 'pending' "
        "    OR (s.status IS NULL AND x.dew_point_f IS NULL AND x.temp_f IS NULL) "
        " ORDER BY x.workout_id"
    ).fetchall()
    return [r[0] for r in rows]


def unfetched_workouts(conn, *, since: str | None = None) -> list[dict]:
    """Workouts not yet settled by the enrichment run.

    A missing route is intentionally included: the run must record
    ``no_route`` rather than silently dropping the workout. Settled statuses
    are excluded, while ``pending`` is retried on the next scheduled run.
    """
    clauses = [
        "(s.status IS NULL OR s.status = 'pending' OR "
        "(s.status = 'no_route' AND EXISTS ("
        "SELECT 1 FROM workout_routes r WHERE r.workout_id = w.id)))"
    ]
    params: list[Any] = []
    if since:
        clauses.append("w.local_date >= ?")
        params.append(since)
    rows = conn.execute(
        "SELECT w.id, w.local_date, w.start_utc, w.duration_min, w.route_ref "
        "  FROM workouts w "
        "  LEFT JOIN workout_weather_status s ON s.workout_id = w.id "
        f" WHERE {' AND '.join(clauses)} "
        " ORDER BY w.start_utc",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


# The archive is deliberately not called for every sample. A single response
# covers all offsets in a workout, and this cache also shares a request between
# workouts that have the same rounded start point and local day.
SLEEP_BETWEEN_CALLS = 0.35


def _gpx_path(route_ref: str | None, routes_dir: str | Path | None) -> Path | None:
    if not route_ref:
        return None
    ref = Path(route_ref)
    candidates = []
    if ref.is_absolute():
        candidates.append(ref)
    if routes_dir is not None and not ref.is_absolute():
        candidates.append(Path(routes_dir) / ref)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _workout_samples(
    conn,
    workout: dict,
    routes_dir: str | Path | None,
) -> list[TrackSample]:
    """Locate one workout, preferring its packed route over any GPX file."""
    route = conn.execute(
        "SELECT start_lat_1dp, start_lon_1dp FROM workout_routes "
        "WHERE workout_id = ? ORDER BY id LIMIT 1",
        (workout["id"],),
    ).fetchone()
    if route is not None and route[0] is not None and route[1] is not None:
        return route_samples(workout["start_utc"], workout["duration_min"] or 0.0,
                             route[0], route[1])

    path = _gpx_path(workout.get("route_ref"), routes_dir)
    if path is None:
        return []
    samples = sample_points(path, workout["duration_min"] or 0.0)
    if not samples:
        return []
    # GPX remains a compatibility fallback, but it follows the same privacy
    # rule as packed routes: only its first point is used for every offset.
    lat, lon = coarsen(samples[0].lat, samples[0].lon)
    return [TrackSample(s.offset_min, lat, lon, s.time_utc) for s in samples]


def _status_row(conn, workout_id: int):
    return conn.execute(
        "SELECT status, attempts FROM workout_weather_status WHERE workout_id = ?",
        (workout_id,),
    ).fetchone()


def _existing_weather_is_complete(conn, workout_id: int) -> bool:
    """Recognize rows written before the durable status table existed."""
    rows = conn.execute(
        "SELECT temp_f FROM workout_weather WHERE workout_id = ?",
        (workout_id,),
    ).fetchall()
    return bool(rows) and all(row["temp_f"] is not None for row in rows)


def _store_status(conn, workout_id: int, status: str, checked_utc: str, attempts: int) -> None:
    conn.execute(
        "INSERT INTO workout_weather_status (workout_id, status, checked_utc, attempts) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(workout_id) DO UPDATE SET status=excluded.status, "
        "checked_utc=excluded.checked_utc, attempts=excluded.attempts",
        (workout_id, status, checked_utc, attempts),
    )
    # Commit each status at once. Left open, this write transaction would span the
    # next workout's archive call (up to 30 s) and hold the vault's write lock
    # against the receiver, whose busy_timeout is also 30 s.
    conn.commit()


def _fill_route_elevation(conn, *, dry_run: bool) -> int:
    rows = conn.execute(
        "SELECT id, encoding, points, n_points FROM workout_routes "
        "WHERE method_version IS NULL OR method_version < ?",
        (elevation.METHOD_VERSION,),
    ).fetchall()
    filled = 0
    for row in rows:
        points = dbmod.decode_route_points(row["points"], row["encoding"], row["n_points"])
        result = elevation.compute_elevation(points)
        if not dry_run:
            conn.execute(
                "UPDATE workout_routes SET ascent_m = ?, descent_m = ?, method_version = ? "
                "WHERE id = ?",
                (result["ascent_m"], result["descent_m"], elevation.METHOD_VERSION, row["id"]),
            )
        filled += 1
    return filled


def enrich_workouts(
    conn,
    *,
    since: str | None = None,
    limit: int | None = None,
    fetch: Callable[[float, float, str], dict | None] = fetch_day,
    dry_run: bool = False,
    routes_dir: str | Path | None = None,
    delay: Callable[[float], None] | None = None,
) -> dict[str, int]:
    """Fill route elevation and weather in one idempotent scheduled pass.

    ``routes_dir`` and ``delay`` are optional compatibility/test seams. The
    public work selection is the packed route table; ``routes_dir`` only lets
    old GPX references remain readable while they are being retired.
    """
    todo = unfetched_workouts(conn, since=since)
    if limit is not None:
        todo = todo[:max(limit, 0)]

    result = {"fetched": 0, "pending": 0, "no_route": 0,
              "elevation_filled": 0, "fetch_calls": 0}
    cache: dict[tuple[float, float, str], dict | None] = {}
    sleep = delay if delay is not None else time.sleep
    checked_utc = dbmod.utcnow_iso()
    network_calls = 0
    if fetch is _DEFAULT_FETCH_DAY:
        # Keep the default signature while allowing tests and embedders to
        # replace the module fetcher without having to pass it explicitly.
        fetch = fetch_day

    for workout in todo:
        prior = _status_row(conn, workout["id"])
        prior_attempts = int(prior["attempts"]) if prior is not None else 0

        # The status table was added after workout_weather. Preserve complete
        # rows from the old script even when their GPX has since disappeared.
        if prior is None and _existing_weather_is_complete(conn, workout["id"]):
            result["fetched"] += 1
            if not dry_run:
                _store_status(conn, workout["id"], "fetched", checked_utc, 0)
            continue

        samples = _workout_samples(conn, workout, routes_dir)
        if not samples:
            result["no_route"] += 1
            if not dry_run:
                _store_status(conn, workout["id"], "no_route", checked_utc, prior_attempts)
            continue

        if dry_run:
            # A dry run is deliberately unable to know whether the archive has
            # data, because learning that would require making the forbidden
            # network call. It reports the route as pending without persisting
            # anything.
            result["pending"] += 1
            continue

        rows = []
        attempts = prior_attempts + 1
        for sample in samples:
            lat, lon = coarsen(sample.lat, sample.lon)
            day = sample.time_utc[:10]
            key = (lat, lon, day)
            if key not in cache:
                if network_calls:
                    sleep(SLEEP_BETWEEN_CALLS)
                cache[key] = fetch(lat, lon, day)
                network_calls += 1
            payload = cache[key]
            reading = conditions_at(payload, sample.time_utc) if payload else None
            rows.append({
                "workout_id": workout["id"], "offset_min": sample.offset_min,
                "lat": lat, "lon": lon,
                "observed_utc": (reading or {}).get("observed_utc"),
                "temp_f": (reading or {}).get("temp_f"),
                "humidity_pct": (reading or {}).get("humidity_pct"),
                "dew_point_f": (reading or {}).get("dew_point_f"),
                "wind_kmh": (reading or {}).get("wind_kmh"),
                "source": SOURCE, "fetched_utc": checked_utc,
            })
        fetched = all(row["temp_f"] is not None for row in rows)
        status = "fetched" if fetched else "pending"
        result[status] += 1
        if not dry_run:
            upsert_weather(conn, rows)
            _store_status(conn, workout["id"], status, checked_utc, attempts)

    result["fetch_calls"] = network_calls
    result["elevation_filled"] = _fill_route_elevation(conn, dry_run=dry_run)
    if not dry_run:
        conn.commit()
    return result


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
