#!/usr/bin/env python3
"""Fill workout_weather from each workout's GPX track (audit part 1, section 6).

Audit part 1 recorded the plan's heat claims as "unmeasurable" and withdrew it
the same day: workouts.route_ref points at a 1 Hz GPX, and a trackpoint carries
position and time, which is all a historical weather API needs. Its prototype
found r=0.82 between dew point and cardiac drift across five pace-matched
sessions -- a lead, not a result, but not one worth leaving unmeasured.

    ./.venv/bin/python scripts/backfill_weather.py --since 2026-06-22
    ./.venv/bin/python scripts/backfill_weather.py --dry-run
    ./.venv/bin/python scripts/backfill_weather.py --retry-pending

WHAT IT SENDS. One request per workout-day per sampled point, to Open-Meteo's
free ERA5 archive, carrying a coordinate ROUNDED to ~11 km (weather.coarsen).
That is coarser than ERA5's own grid, so the rounding loses no accuracy. It is
still an approximate location leaving the machine on every workout -- which is
why this is a script you run rather than something wired into ingest.

ERA5 LAGS ~5 DAYS. A workout too recent to be published gets a row with NULL
readings and a real fetched_utc, which --retry-pending picks up later. That is
deliberately different from having no row at all, which means "never asked".

WRITES TO THE DB. Run ./scripts/backup_health.sh first if you care.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod        # noqa: E402
from health_advisor import weather as wx      # noqa: E402
from health_advisor import routes as rt       # noqa: E402

# Open-Meteo asks for courtesy on the free tier; this is well inside it.
SLEEP_BETWEEN_CALLS = 0.35


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(LOCAL_DB_PATH))
    ap.add_argument("--routes-dir", default=str(rt.DEFAULT_ROUTES_DIR))
    ap.add_argument("--since", default=None,
                    help="local_date lower bound (routes only exist from 2026-06-22)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retry-pending", action="store_true",
                    help="re-ask only for workouts whose readings came back empty")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be fetched; make no requests, write nothing")
    args = ap.parse_args()

    conn = dbmod.connect(args.db)
    dbmod.init_db(conn)
    routes_dir = Path(args.routes_dir)

    todo = wx.unfetched_workouts(conn, since=args.since)
    if args.retry_pending:
        pending = set(wx.pending_workout_ids(conn))
        todo = [w for w in todo if w["id"] in pending]
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        print("nothing to fetch — every route-bearing workout already has weather")
        return 0

    missing_gpx = written = pending = 0
    for w in todo:
        gpx = routes_dir / (w["route_ref"] or "")
        samples = wx.sample_points(gpx, w["duration_min"] or 0.0)
        if not samples:
            missing_gpx += 1
            print(f"  {w['local_date']}  workout {w['id']:>4}  "
                  f"no usable track ({w['route_ref']})")
            continue

        if args.dry_run:
            lat, lon = wx.coarsen(samples[0].lat, samples[0].lon)
            print(f"  {w['local_date']}  workout {w['id']:>4}  "
                  f"{len(samples)} sample(s) @ {lat},{lon}")
            continue

        rows = []
        for s in samples:
            day = s.time_utc[:10]
            payload = wx.fetch_day(s.lat, s.lon, day)
            time.sleep(SLEEP_BETWEEN_CALLS)
            reading = wx.conditions_at(payload, s.time_utc) if payload else None
            lat, lon = wx.coarsen(s.lat, s.lon)
            rows.append({
                "workout_id": w["id"], "offset_min": s.offset_min,
                "lat": lat, "lon": lon,
                "observed_utc": (reading or {}).get("observed_utc"),
                "temp_f": (reading or {}).get("temp_f"),
                "humidity_pct": (reading or {}).get("humidity_pct"),
                "dew_point_f": (reading or {}).get("dew_point_f"),
                "wind_kmh": (reading or {}).get("wind_kmh"),
                "source": wx.SOURCE, "fetched_utc": dbmod.utcnow_iso(),
            })

        wx.upsert_weather(conn, rows)
        got = [r for r in rows if r["dew_point_f"] is not None]
        if got:
            written += 1
            dew = sum(r["dew_point_f"] for r in got) / len(got)
            temp = [r["temp_f"] for r in got if r["temp_f"] is not None]
            print(f"  {w['local_date']}  workout {w['id']:>4}  "
                  f"{len(got)}/{len(rows)} sample(s)  "
                  f"{(sum(temp)/len(temp)) if temp else float('nan'):5.1f}F  "
                  f"dew {dew:5.1f}F  {wx.dew_point_label(dew)}")
        else:
            pending += 1
            print(f"  {w['local_date']}  workout {w['id']:>4}  "
                  f"pending — ERA5 has not published this day yet")

    verb = "would fetch" if args.dry_run else "fetched"
    print(f"\n{verb} {len(todo)} workout(s): {written} written, "
          f"{pending} pending, {missing_gpx} without a usable track")
    if pending:
        print("re-run with --retry-pending in a few days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
