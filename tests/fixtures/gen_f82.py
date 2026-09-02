#!/usr/bin/env python
"""Build a SYNTHETIC tests/fixtures/f82_multi_source_live_20260828.json.gz.

Shape follows tests/fixture_loader.py::load_f82_multi_source. Every device
name, timestamp and distance is invented; the numbers are chosen so the
conditions F-82 and #189 were written to catch are reproduced:

  * three concurrent device streams inside one post-cutoff workout window,
    so the un-arbitrated distance is inflated and the arbitrated one is not;
  * a no-GymKit post-cutoff day where the whole-day Watch-vs-iPhone rule is
    the only thing that excludes the phone;
  * pre-cutoff multi-source workouts that arbitration must leave alone;
  * interval samples that span many 20-second buckets, so the #189
    distribution is genuinely exercised: collapsing one back into a single
    bucket puts ~0.4 mi into 20 seconds and trips the implausible-pace floor.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "f82_multi_source_live_20260828.json.gz"

BUCKET = 20
WATCH = "Demo Apple Watch"
IPHONE = "Demo iPhone"
GYMKIT = "GymKit"

records: list[dict] = []
workouts: list[dict] = []


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00")


def epoch(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(
        tzinfo=timezone.utc).timestamp())


def add_record(metric, value, unit, start_s, end_s, local_date, source,
               workout_id, tag):
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
        "workout_id": workout_id,
    })


def emit_stream(*, workout_id, source, local_date, first_bucket, targets,
                window_start, window_end, short_counts, long_fraction=0.4,
                max_long_buckets=15, tag):
    """Realise a per-bucket distance target vector as records.

    `targets[i]` is the distance the bucket `first_bucket + i` must end up
    holding from this source. A run of equal targets is carried partly by one
    long interval sample spanning the whole run -- the #189 case -- and the
    rest by short one-second samples inside each bucket.
    """
    n = len(targets)
    # Long samples cover only whole buckets fully inside the window.
    runs: list[tuple[int, int]] = []
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
                           start_s, end_s, local_date, source, workout_id,
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
            offset = lo + (k * max(1, (hi - lo) // max(count, 1)))
            offset = min(offset, hi - 1)
            add_record("distance_walking_running", each, "mi",
                       offset, offset + 1, local_date, source, workout_id,
                       f"{tag}|s|{index}|{k}")
        carried[index] += each * count
    return carried


def add_hr(workout_id, local_date, first_bucket, n_buckets, bpm, tag,
           per_bucket=1):
    for index in range(n_buckets):
        for k in range(per_bucket):
            ts = (first_bucket + index) * BUCKET + 3 + k * 5
            add_record("heart_rate", bpm, "count/min", ts, ts, local_date,
                       WATCH, workout_id, f"{tag}|hr|{index}|{k}")


def add_workout(*, wid, wtype, start_s, end_s, local_date, distance_mi,
                source, avg_hr, max_hr):
    workouts.append({
        "avg_heart_rate": avg_hr,
        "dedupe_key": f"{wtype}|{iso(start_s)}|{iso(end_s)}",
        "distance_mi": distance_mi,
        "duration_min": round((end_s - start_s) / 60.0, 2),
        "end_utc": iso(end_s),
        "energy_kcal": round((end_s - start_s) / 60.0 * 9.4, 1),
        "hk_uuid": f"demo-workout-{wid}",
        "id": wid,
        "local_date": local_date,
        "max_heart_rate": max_hr,
        "route_ref": None,
        "source": source,
        "start_utc": iso(start_s),
        "unit_distance": "mi",
        "workout_type": wtype,
    })


# --------------------------------------------------------------------------
# Workout 815 -- 2026-08-25, the GymKit day with three concurrent streams.
# --------------------------------------------------------------------------
W815_START = epoch("2026-08-25T14:06:41")
W815_END = epoch("2026-08-25T15:04:18")
B815 = W815_START // BUCKET
N815 = ((W815_END - 1) // BUCKET) - B815 + 1        # 173

GYM_TOTAL = 3.44985          # the arbitrated in-window distance
WINDOW_TOTAL = 7.2799        # all three streams, un-arbitrated
J, W, F = 0.030, 0.014725, 0.00065
T0 = 0.002

# Positions 1..172 are the block layout; position 0 is the partial first
# bucket and is deliberately below the jog lane in the un-arbitrated view,
# so the un-arbitrated chain is 172 buckets (57.3 min) rather than 173.
layout = ["f"] + (["j"] * 15 + ["w"] * 4 + ["j"] * 12 + ["w"] * 4
                  + ["j"] * 12 + ["w"] * 4 + ["j"] * 12 + ["w"] * 4
                  + ["j"] * 9 + ["w"] * 96)
assert len(layout) == N815, (len(layout), N815)

gym_targets = [{"f": F, "j": J, "w": W}[kind] for kind in layout]
gym_targets[-1] = round(GYM_TOTAL - sum(gym_targets[:-1]), 9)
combined = [T0] + [round((WINDOW_TOTAL - T0) / (N815 - 1), 9)] * (N815 - 1)
other = [round(combined[i] - gym_targets[i], 9) for i in range(N815)]
watch_targets = [round(v * 0.55, 9) for v in other]
phone_targets = [round(v - round(v * 0.55, 9), 9) for v in other]
assert all(v > 0 for v in watch_targets + phone_targets)

# 4343 rows inside this window, across the three sources.
TARGET_ROWS = 4343
before_815 = len(records)
counts = {source: [8] * N815 for source in (GYMKIT, WATCH, IPHONE)}
# 8 short samples per bucket per source plus the long interval samples comes
# to 4197; the window carries 4343, so the first 146 GymKit buckets take one
# more.
for _i in range(146):
    counts[GYMKIT][_i] = 9
gym_carried = emit_stream(
    workout_id=815, source=GYMKIT, local_date="2026-08-25",
    first_bucket=B815, targets=gym_targets, window_start=W815_START,
    window_end=W815_END, short_counts=counts[GYMKIT], tag="w815-gym")
watch_carried = emit_stream(
    workout_id=815, source=WATCH, local_date="2026-08-25",
    first_bucket=B815, targets=watch_targets, window_start=W815_START,
    window_end=W815_END, short_counts=counts[WATCH], tag="w815-watch")
phone_carried = emit_stream(
    workout_id=815, source=IPHONE, local_date="2026-08-25",
    first_bucket=B815, targets=phone_targets, window_start=W815_START,
    window_end=W815_END, short_counts=counts[IPHONE], tag="w815-phone")
print("w815 rows before padding:", len(records) - before_815)
add_hr(815, "2026-08-25", B815, N815, 141.0, "w815")

print("gym carried total", sum(gym_carried))
print("window total", sum(gym_carried) + sum(watch_carried) + sum(phone_carried))


# --------------------------------------------------------------------------
# The remaining twelve workouts.
# --------------------------------------------------------------------------
def simple_workout(*, wid, day, hhmm, n_buckets, sources, per_bucket,
                   distance_mi, hr, tag, wtype="running"):
    """One aligned window whose per-source per-bucket distance is constant."""
    start_s = epoch(f"{day}T{hhmm}:00")
    start_s -= start_s % BUCKET
    end_s = start_s + n_buckets * BUCKET
    first = start_s // BUCKET
    # Before the HealthKit-direct boundary every sample is a point sample and
    # the bucketer deliberately preserves one bucket per row, so a
    # multi-bucket interval sample would be a shape the era never produced.
    long_fraction = 0.4 if day >= "2026-08-21" else 0.0
    for source in sources:
        emit_stream(
            workout_id=wid, source=source, local_date=day, first_bucket=first,
            targets=[per_bucket[source]] * n_buckets, window_start=start_s,
            window_end=end_s, short_counts=[6] * n_buckets,
            long_fraction=long_fraction,
            tag=f"{tag}-{source.replace(' ', '')}")
    add_hr(wid, day, first, n_buckets, hr, tag)
    add_workout(wid=wid, wtype=wtype, start_s=start_s, end_s=end_s,
                local_date=day, distance_mi=distance_mi, source=WATCH,
                avg_hr=hr, max_hr=hr + 22.0)
    return start_s, end_s


add_workout(wid=815, wtype="running", start_s=W815_START, end_s=W815_END,
            local_date="2026-08-25",
            distance_mi=round(GYM_TOTAL / 1.0001, 5), source=WATCH,
            avg_hr=141.0, max_hr=166.0)

# 809: the no-GymKit post-cutoff day. Watch and iPhone both write; only the
# whole-day rule excludes the phone. Every arbitrated bucket is below the
# jog lane, so the session has no block structure at all.
simple_workout(wid=809, day="2026-08-21", hhmm="07:12", n_buckets=120,
               sources=(WATCH, IPHONE), per_bucket={WATCH: 0.015,
                                                    IPHONE: 0.0122},
               distance_mi=round(120 * 0.015 / 1.0802, 5), hr=118.0,
               tag="w809", wtype="walking")
# 810 and 812: the second post-cutoff day, same shape.
simple_workout(wid=810, day="2026-08-22", hhmm="08:04", n_buckets=90,
               sources=(WATCH, IPHONE), per_bucket={WATCH: 0.016,
                                                    IPHONE: 0.0131},
               distance_mi=round(90 * 0.016, 5), hr=121.0,
               tag="w810", wtype="walking")
simple_workout(wid=812, day="2026-08-22", hhmm="17:26", n_buckets=75,
               sources=(WATCH, IPHONE), per_bucket={WATCH: 0.0145,
                                                    IPHONE: 0.0118},
               distance_mi=round(75 * 0.0145, 5), hr=115.0,
               tag="w812", wtype="walking")

# 746, 748, 762: pre-cutoff multi-source sessions. Arbitration must not touch
# them, and 746/748 overlap so the extraction captured the overlap twice.
w746 = simple_workout(wid=746, day="2026-06-15", hhmm="06:30", n_buckets=150,
                      sources=(WATCH, IPHONE),
                      per_bucket={WATCH: 0.0161, IPHONE: 0.0143},
                      distance_mi=2.4, hr=132.0, tag="w746")
simple_workout(wid=748, day="2026-06-18", hhmm="06:40", n_buckets=140,
               sources=(WATCH, IPHONE),
               per_bucket={WATCH: 0.0158, IPHONE: 0.0139},
               distance_mi=2.2, hr=134.0, tag="w748")
simple_workout(wid=762, day="2026-06-22", hhmm="07:05", n_buckets=160,
               sources=(WATCH, IPHONE, GYMKIT),
               per_bucket={WATCH: 0.0155, IPHONE: 0.0136, GYMKIT: 0.0121},
               distance_mi=2.5, hr=136.0, tag="w762")

# 803-808: single-stream controls.
CONTROLS = [
    (803, "2026-07-02", "06:45", 130, 0.0168, 2.18, 138.0),
    (804, "2026-07-09", "06:50", 120, 0.0172, 2.06, 140.0),
    (805, "2026-07-16", "06:35", 145, 0.0164, 2.37, 137.0),
    (806, "2026-07-23", "06:55", 110, 0.0175, 1.92, 142.0),
    (807, "2026-07-30", "06:48", 135, 0.0166, 2.24, 139.0),
    (808, "2026-08-06", "06:52", 125, 0.0170, 2.12, 141.0),
]
for wid, day, hhmm, n_buckets, per_bucket, distance_mi, hr in CONTROLS:
    simple_workout(wid=wid, day=day, hhmm=hhmm, n_buckets=n_buckets,
                   sources=(WATCH,), per_bucket={WATCH: per_bucket},
                   distance_mi=distance_mi, hr=hr, tag=f"w{wid}")

# The overlap the loader's docstring describes: 746's tail was extracted a
# second time under 748, so the file carries duplicate rows that the real
# dedupe key collapses on insert.
overlap = [dict(row, workout_id=748) for row in records
           if row["workout_id"] == 746][:320]

distinct = {row["dedupe_key"] for row in records}
TARGET_RECORDS = 42416
filler = TARGET_RECORDS - len(distinct)
if filler < 0:
    raise SystemExit(f"over budget by {-filler} records")
for index in range(filler):
    ts = epoch("2026-06-10T09:00:00") + index * 5
    add_record("heart_rate", 62.0 + index % 9, "count/min", ts, ts,
               "2026-06-10", WATCH, None, f"rest-hr|{index}")

records.extend(overlap)

expected = {}
for workout in workouts:
    wid = workout["id"]
    rows = [r for r in records if r["workout_id"] == wid
            and r["metric"] == "distance_walking_running"]
    sources = sorted({r["source"] for r in rows})
    summed = round(sum(r["value"] for r in rows), 5)
    expected[str(wid)] = {
        "distance_mi": workout["distance_mi"],
        "local_date": workout["local_date"],
        "n_sources": len(sources),
        "ratio": round(summed / workout["distance_mi"], 4),
        "role": ("multi_source" if len(sources) > 1 else "control"),
        "sources": sources,
        "summed_sample_mi": summed,
        "workout_type": workout["workout_type"],
    }

per_workout = {}
for workout in workouts:
    wid = workout["id"]
    rows = [r for r in records if r["workout_id"] == wid]
    per_workout[str(wid)] = {
        "distinct_dedupe_keys": len({r["dedupe_key"] for r in rows}),
        "rows": len(rows),
    }

payload = {
    "control_workout_ids": [803, 804, 805, 806, 807, 808],
    "expected": expected,
    "multi_source_workout_ids": [746, 748, 762, 809, 810, 812, 815],
    "provenance": {
        "dedupe_evidence": {
            "all_workouts_rows_equal_distinct_keys": all(
                v["rows"] == v["distinct_dedupe_keys"]
                for k, v in per_workout.items()),
            "file_level_distinct_dedupe_keys": len(
                {r["dedupe_key"] for r in records}),
            "file_level_rows": len(records),
            "note": ("per-workout extraction; a row inside two overlapping "
                     "workout windows is written once per window"),
            "per_workout": per_workout,
            "why_file_level_counts_differ": (
                "746 and 748 overlap, so the shared tail appears under both "
                "workout ids; the production dedupe key collapses it on "
                "insert"),
        },
        "extracted_utc": "synthetic",
        "host": "synthetic",
        "path": "generated; no vault was read",
        "selection": ("thirteen workouts: seven with concurrent device "
                      "streams, six single-stream controls"),
        "why": (
            "SYNTHETIC replacement for the live F-82 capture, which held real "
            "device names and real movement and could not be published. The "
            "conditions the tests measure are reproduced rather than copied: "
            "three concurrent streams inside 815's post-cutoff window, a "
            "no-GymKit post-cutoff day (809) where only the whole-day rule "
            "excludes the phone, pre-cutoff multi-source sessions arbitration "
            "must leave alone, and multi-bucket interval samples so #189's "
            "distribution is exercised rather than assumed."),
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
      "workouts", len(workouts), "size", OUT.stat().st_size)
