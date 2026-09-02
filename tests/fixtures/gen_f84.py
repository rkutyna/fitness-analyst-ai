#!/usr/bin/env python
"""Build a SYNTHETIC tests/fixtures/f84_week8_wholeday_live_20260828.json.gz.

Shape follows tests/fixture_loader.py::load_f84_wholeday: a whole week of
day-level movement, not just workout windows, so the whole-day Watch-vs-iPhone
rule can be exercised on a day with no workout at all (2026-08-23).

Every device name, timestamp and distance is invented.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "f84_week8_wholeday_live_20260828.json.gz"

BUCKET = 20
WATCH = "Demo Apple Watch"
IPHONE = "Demo iPhone"

records: list[dict] = []
workouts: list[dict] = []


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00")


def epoch(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(
        tzinfo=timezone.utc).timestamp())


def add_record(metric, value, unit, start_s, end_s, local_date, source, tag):
    records.append({
        "dedupe_key": f"demo|{tag}",
        "end_utc": iso(end_s),
        "local_date": local_date,
        "metric": metric,
        "origin": "healthkit",
        "source": source,
        "start_utc": iso(start_s),
        "unit": unit,
        "value": value,
    })


def emit_stream(*, source, local_date, first_bucket, targets, window_start,
                window_end, short_counts, long_fraction=0.4,
                max_long_buckets=15, tag):
    """Same realisation as the F-82 generator: part of each run's distance is
    carried by one interval sample spanning the whole run, the rest by short
    in-bucket samples. The long samples are what #189's distribution has to
    spread; collapsing one back into its first bucket is immediately visible.
    """
    n = len(targets)
    runs = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and targets[j + 1] == targets[i]:
            j += 1
        runs.append((i, j))
        i = j + 1

    carried = [0.0] * n
    seq = 0
    for run_start, run_end in runs:
        pos = run_start
        while pos <= run_end:
            stop = min(pos + max_long_buckets - 1, run_end)
            start_s = (first_bucket + pos) * BUCKET
            end_s = (first_bucket + stop + 1) * BUCKET
            length = stop - pos + 1
            if (start_s >= window_start and end_s <= window_end
                    and length >= 2 and targets[pos] > 0):
                value = round(long_fraction * targets[pos] * length, 9)
                seq += 1
                add_record("distance_walking_running", value, "mi",
                           start_s, end_s, local_date, source,
                           f"{tag}|long|{seq}")
                for k in range(pos, stop + 1):
                    carried[k] += value / length
            pos = stop + 1

    for index, target in enumerate(targets):
        bucket_start = (first_bucket + index) * BUCKET
        lo = max(bucket_start, window_start)
        hi = min(bucket_start + BUCKET, window_end)
        count = short_counts[index]
        remaining = target - carried[index]
        if count <= 0 or remaining <= 0:
            continue
        each = round(remaining / count, 9)
        for k in range(count):
            offset = min(lo + k * max(1, (hi - lo) // max(count, 1)), hi - 1)
            add_record("distance_walking_running", each, "mi",
                       offset, offset + 1, local_date, source,
                       f"{tag}|s|{index}|{k}")
        carried[index] += each * count
    return carried


def add_hr(local_date, first_bucket, n_buckets, bpm, tag):
    for index in range(n_buckets):
        ts = (first_bucket + index) * BUCKET + 4
        add_record("heart_rate", bpm, "count/min", ts, ts, local_date,
                   WATCH, f"{tag}|hr|{index}")


def add_workout(*, wid, wtype, start_s, end_s, local_date, distance_mi,
                avg_hr, max_hr):
    workouts.append({
        "avg_heart_rate": avg_hr,
        "dedupe_key": f"{wtype}|{iso(start_s)}|{iso(end_s)}",
        "distance_mi": distance_mi,
        "duration_min": round((end_s - start_s) / 60.0, 2),
        "end_utc": iso(end_s),
        "energy_kcal": round((end_s - start_s) / 60.0 * 9.6, 1),
        "hk_uuid": f"demo-workout-{wid}",
        "id": wid,
        "local_date": local_date,
        "max_heart_rate": max_hr,
        "route_ref": None,
        "source": WATCH,
        "start_utc": iso(start_s),
        "unit_distance": "mi",
        "workout_type": wtype,
    })


CUTOFF = "2026-08-21"


def long_fraction(day: str) -> float:
    return 0.4 if day >= CUTOFF else 0.0


def session(*, wid, day, hhmm, layout, values, hr, tag, wtype="running",
            phone_ratio=0.82, shorts=5):
    """One workout window; `layout` is a list of per-bucket keys into `values`."""
    start_s = epoch(f"{day}T{hhmm}:00")
    start_s -= start_s % BUCKET
    n = len(layout)
    end_s = start_s + n * BUCKET
    first = start_s // BUCKET
    targets = [values[key] for key in layout]
    emit_stream(source=WATCH, local_date=day, first_bucket=first,
                targets=targets, window_start=start_s, window_end=end_s,
                short_counts=[shorts] * n, long_fraction=long_fraction(day),
                tag=f"{tag}-watch")
    # The second device stream, and the multi-bucket interval sample, both
    # begin at the HealthKit-direct boundary. Before it there is one stream of
    # point samples, which is why half of week 8 looks nothing like the other
    # half -- the boundary falls inside the week.
    if day >= CUTOFF:
        emit_stream(source=IPHONE, local_date=day, first_bucket=first,
                    targets=[round(v * phone_ratio, 9) for v in targets],
                    window_start=start_s, window_end=end_s,
                    short_counts=[shorts] * n, tag=f"{tag}-phone")
    add_hr(day, first, n, hr, tag)
    add_workout(wid=wid, wtype=wtype, start_s=start_s, end_s=end_s,
                local_date=day, distance_mi=round(sum(targets), 3),
                avg_hr=hr, max_hr=hr + 20.0)


# --- the two block oracles -------------------------------------------------
# 8 on / 3 off x3 on the Thursday, 10 on / 3 off x2 on the Saturday. A block
# is 24 (or 30) contiguous jog-paced buckets; the recoveries are nine buckets
# wide, far beyond the two-bucket bridge, so the blocks cannot merge.
session(wid=808, day="2026-08-20", hhmm="06:40",
        layout=(["r"] * 24 + ["e"] * 9) * 2 + ["r"] * 24,
        values={"r": 0.035, "e": 0.012}, hr=148.0, tag="w808")
session(wid=811, day="2026-08-22", hhmm="07:05",
        layout=["r"] * 30 + ["e"] * 9 + ["r"] * 30,
        values={"r": 0.033, "e": 0.011}, hr=146.0, tag="w811")

# --- the Friday: two walking sessions, no block structure at all -----------
# Every bucket is above the walking floor and below the jog lane, so
# 2026-08-21 is 122 walk buckets and no reps.
WALK = 0.015738
session(wid=809, day="2026-08-21", hhmm="07:12", layout=["w"] * 62,
        values={"w": WALK}, hr=112.0, tag="w809", wtype="walking")
session(wid=810, day="2026-08-21", hhmm="17:30", layout=["w"] * 60,
        values={"w": WALK}, hr=115.0, tag="w810", wtype="walking")

# --- the rest of week 8 ----------------------------------------------------
session(wid=806, day="2026-08-17", hhmm="06:55", layout=["w"] * 84,
        values={"w": 0.0161}, hr=118.0, tag="w806", wtype="walking")
session(wid=807, day="2026-08-18", hhmm="07:20", layout=["w"] * 70,
        values={"w": 0.0154}, hr=116.0, tag="w807", wtype="walking")
session(wid=812, day="2026-08-19", hhmm="06:48", layout=["w"] * 96,
        values={"w": 0.0149}, hr=120.0, tag="w812", wtype="walking")

# --- whole-day background movement ----------------------------------------
# The point of a whole-day capture: both devices write outside any workout,
# and 2026-08-23 has no workout at all. Every background bucket sits BELOW the
# walking floor (bucket_min / 40 min/mi), so this movement is real in the
# vault without changing any day's classified minutes.
WEEK = ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20",
        "2026-08-21", "2026-08-22", "2026-08-23"]
for day in WEEK:
    for hour in (9, 12, 15, 19):
        start_s = epoch(f"{day}T{hour:02d}:00:00")
        first = start_s // BUCKET
        n = 45
        emit_stream(source=WATCH, local_date=day, first_bucket=first,
                    targets=[0.0041] * n, window_start=start_s,
                    window_end=start_s + n * BUCKET,
                    short_counts=[2] * n, long_fraction=long_fraction(day),
                    tag=f"bg-{day}-{hour}-watch")
        if day >= CUTOFF:
            emit_stream(source=IPHONE, local_date=day, first_bucket=first,
                        targets=[0.0034] * n, window_start=start_s,
                        window_end=start_s + n * BUCKET,
                        short_counts=[2] * n, tag=f"bg-{day}-{hour}-phone")

spans = {"0": 0, "1-59": 0, "60-249": 0, ">=250": 0}
for row in records:
    if row["metric"] != "distance_walking_running":
        continue
    span = epoch(row["end_utc"][:19]) - epoch(row["start_utc"][:19])
    key = ("0" if span == 0 else "1-59" if span < 60
           else "60-249" if span < 250 else ">=250")
    spans[key] += 1

payload = {
    "provenance": {
        "extracted_utc": "synthetic",
        "host": "synthetic",
        "oracle": {
            "note": (
                "SYNTHETIC. The live week-8 oracle was the athlete's own "
                "recollection of each session and cannot be invented, so it "
                "is not reproduced here. What the fixture does carry is the "
                "structure the block dial is measured against: 8 on / 3 off "
                "x3 on 2026-08-20 (workout 808) and 10 on / 3 off x2 on "
                "2026-08-22 (workout 811). The cadence stream is absent on "
                "purpose, so classified jog volume is zero for the whole "
                "week -- that is the assertion, not an artifact."),
            "per_day_actual": {"2026-08-21": 0, "2026-08-22": 0,
                               "2026-08-23": 0},
            "per_day_reported_2026_08_28": {day: 0.0 for day in WEEK},
            "quote": "n/a -- synthetic fixture, no recorded statement",
            "source": "generated",
            "week_total_jog_minutes": 0,
        },
        "path": "generated; no vault was read",
        "span_distribution_distance": spans,
        "why": (
            "SYNTHETIC replacement for the live F-84 whole-day capture. The "
            "conditions the tests measure are reproduced: whole-day movement "
            "from both devices including a day with no workout (2026-08-23), "
            "distance carried partly by interval samples spanning many "
            "20-second buckets so #189's distribution is exercised, and two "
            "sessions whose block structure is the external oracle #190 "
            "recorded."),
        "workout_columns": [
            "id", "workout_type", "start_utc", "end_utc", "local_date",
            "duration_min", "energy_kcal", "distance_mi", "unit_distance",
            "source", "route_ref", "dedupe_key", "avg_heart_rate",
            "max_heart_rate", "hk_uuid"],
    },
    "records": records,
    "workouts": sorted(workouts, key=lambda w: w["id"]),
}

with gzip.open(OUT, "wt", encoding="utf-8") as handle:
    json.dump(payload, handle)
print("records", len(records), "distinct",
      len({r["dedupe_key"] for r in records}),
      "workouts", len(workouts), "spans", spans,
      "size", OUT.stat().st_size)
