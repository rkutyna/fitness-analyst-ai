"""Write workout GPS routes to GPX files on disk. The receiver payload carries
the actual track points (unlike the backfill, which only had a path reference),
so we persist them as standard GPX and store the filename in workouts.route_ref."""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

DEFAULT_ROUTES_DIR = Path(__file__).resolve().parent.parent / "data" / "routes"


def _filename(workout: dict) -> str:
    """Deterministic per-workout name so re-ingesting overwrites, not duplicates."""
    stamp = re.sub(r"[^0-9]", "", workout.get("start_utc", ""))[:14]
    wtype = re.sub(r"[^a-z0-9]+", "-", (workout.get("workout_type") or "workout").lower())
    return f"{workout.get('local_date', 'unknown')}_{wtype}_{stamp}.gpx"


def write_gpx(workout: dict, routes_dir: str | Path = DEFAULT_ROUTES_DIR) -> str | None:
    """Write the workout's route_points to a GPX file; return the filename
    (stored as route_ref) or None when there's no route."""
    pts = workout.get("route_points") or []
    if not pts:
        return None
    routes_dir = Path(routes_dir)
    routes_dir.mkdir(parents=True, exist_ok=True)
    name = _filename(workout)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="health_advisor" '
        'xmlns="http://www.topografix.com/GPX/1/1">',
        f"  <trk><name>{escape(str(workout.get('workout_type', 'workout')))}</name>",
        "    <trkseg>",
    ]
    for p in pts:
        if p.get("lat") is None or p.get("lon") is None:
            continue
        lines.append(f'      <trkpt lat="{p["lat"]}" lon="{p["lon"]}">')
        if p.get("ele") is not None:
            lines.append(f"        <ele>{p['ele']}</ele>")
        if p.get("time"):
            lines.append(f"        <time>{escape(p['time'])}</time>")
        lines.append("      </trkpt>")
    lines += ["    </trkseg>", "  </trk>", "</gpx>", ""]
    (routes_dir / name).write_text("\n".join(lines))
    return name
