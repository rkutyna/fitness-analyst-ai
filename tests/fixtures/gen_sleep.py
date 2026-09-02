#!/usr/bin/env python
"""Build a SYNTHETIC tests/fixtures/sleep_nights.json.

Shape and semantics follow tests/fixtures/capture.py::capture_sleep_nights:
raw `records` rows for each night and its two neighbours, with `local_date`
assigned as the date the SAMPLE ends (the defect E7-1 fixes), plus the
`stored_derived` values the pre-fix pipeline produced from those rows.

Every timestamp, duration and device name here is invented.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from health_advisor import derive as D  # noqa: E402

UTC_OFFSET = timedelta(hours=4)   # local = UTC-04:00 for this synthetic vault
SOURCE = "Demo Watch"
ORIGIN = "healthkit"

_next_id = 1000


def chain(start: datetime, spec: list[tuple[float, str]]) -> list[dict]:
    """Contiguous stage samples from `start`; spec is (minutes, metric)."""
    global _next_id
    out = []
    cursor = start
    for minutes, metric in spec:
        end = cursor + timedelta(seconds=round(minutes * 60))
        _next_id += 1
        out.append({
            "id": _next_id,
            "metric": metric,
            "value": round(minutes, 4),
            "unit": "min",
            "start_utc": (cursor + UTC_OFFSET).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "end_utc": (end + UTC_OFFSET).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "start_local": cursor.strftime("%Y-%m-%d %H:%M:%S"),
            # THE DEFECT: filed under the date the SAMPLE ends.
            "local_date": end.date().isoformat(),
            "source": SOURCE,
            "origin": ORIGIN,
            "dedupe_key": f"demo-sleep|{metric}|{cursor:%Y%m%dT%H%M%S}",
        })
        cursor = end
    return out


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


# --- night builders ---------------------------------------------------------

def modern_night(start: str, spec: list[tuple[float, str]]) -> list[dict]:
    return chain(dt(start), spec)


def stage_run(total_min: float, n: int, awake_every: int = 4,
              awake_min: float = 3.5) -> list[tuple[float, str]]:
    """`n` contiguous samples covering `total_min`, with periodic awakenings."""
    spec: list[tuple[float, str]] = []
    awake_slots = [i for i in range(n) if i and i % awake_every == 0]
    asleep_total = total_min - awake_min * len(awake_slots)
    asleep_each = asleep_total / (n - len(awake_slots))
    for index in range(n):
        if index in awake_slots:
            spec.append((awake_min, "sleep_awake"))
        else:
            spec.append((asleep_each, "sleep_asleep"))
    return spec


records: dict[str, list[dict]] = {}

# --- 2026-07-14: the measured win ------------------------------------------
# Truncated tail of the night that ENDS on 07-13 (its pre-midnight half sits
# on 07-12 and is outside the captured window, exactly as in a real capture).
n0714 = modern_night("2026-07-13 00:00:00", stage_run(391.0, 16))
# Episode A. True onset 23:26:58 on the 13th. Eight short onset samples end
# before midnight and are therefore filed under 07-13.
epA = modern_night("2026-07-13 23:26:58",
                   [(4.1, "sleep_asleep"), (4.1, "sleep_awake"),
                    (4.1, "sleep_asleep"), (4.1, "sleep_asleep"),
                    (4.1, "sleep_awake"), (4.1, "sleep_asleep"),
                    (4.1, "sleep_asleep"), (4.1, "sleep_asleep")]
                   + stage_run(407.4333, 20))
# Episode B: the next night. Ten samples end before midnight on 07-14.
epB = modern_night("2026-07-14 22:47:20",
                   [(7.2, "sleep_asleep")] * 10 + stage_run(372.6667, 18))
# The leading edge of the night that ends on 07-16; its session is truncated
# at the capture boundary, so it must NOT move.
n0716 = modern_night("2026-07-15 23:11:00",
                     [(15.0, "sleep_asleep"), (15.0, "sleep_asleep"),
                      (16.0, "sleep_asleep")])
records["2026-07-14"] = n0714 + epA + epB + n0716

# --- 2026-08-15: the wake-time invariance check ----------------------------
records["2026-08-15"] = (
    modern_night("2026-08-14 00:04:00", stage_run(382.0, 15))
    + modern_night("2026-08-14 23:12:00",
                   [(6.5, "sleep_asleep")] * 8 + stage_run(414.0, 21))
    + modern_night("2026-08-15 23:04:00",
                   [(18.0, "sleep_asleep"), (18.0, "sleep_asleep"),
                    (18.0, "sleep_asleep")])
)

# --- 2026-07-05: the control, an episode with no sample ending pre-midnight -
records["2026-07-05"] = (
    modern_night("2026-07-03 23:47:00",
                 [(21.0, "sleep_asleep")] + stage_run(392.0, 19))
    + modern_night("2026-07-04 23:53:00",
                   [(22.0, "sleep_asleep")] + stage_run(380.0, 19))
    + modern_night("2026-07-05 23:51:00",
                   [(24.0, "sleep_asleep")] + stage_run(365.0, 18))
)

# --- 2017-03-15: one span per night ----------------------------------------
records["2017-03-15"] = (
    chain(dt("2017-03-13 23:20:00"), [(431.0, "sleep_in_bed")])
    + chain(dt("2017-03-13 23:31:00"), [(414.0, "sleep_asleep")])
    + chain(dt("2017-03-14 23:15:00"), [(427.0, "sleep_in_bed")])
    + chain(dt("2017-03-14 23:24:00"), [(409.0, "sleep_asleep")])
    + chain(dt("2017-03-15 23:32:00"), [(419.0, "sleep_in_bed")])
    + chain(dt("2017-03-15 23:42:00"), [(401.0, "sleep_asleep")])
)

# --- 2021-11-10: the untested middle, 61-minute samples --------------------
def middle_night(start: str, first: float, n: int, each: float) -> list[dict]:
    return chain(dt(start), [(first, "sleep_asleep")]
                 + [(each, "sleep_asleep")] * n)


records["2021-11-10"] = (
    middle_night("2021-11-08 23:38:00", 63.0, 5, 61.0)
    + middle_night("2021-11-09 23:40:00", 62.0, 5, 61.0)
    + middle_night("2021-11-10 23:44:00", 62.0, 5, 61.0)
)

# --- 2022-05-18: the long midnight-crossing episode ------------------------
records["2022-05-18"] = (
    middle_night("2022-05-16 23:52:00", 70.0, 4, 75.0)
    # 21:00 on the 17th through 18:00 on the 18th: the 21-hour episode the
    # two-day attribution padding exists for. Exactly two samples end before
    # midnight.
    + chain(dt("2022-05-17 21:00:00"),
            [(75.0, "sleep_in_bed")] * 16 + [(60.0, "sleep_in_bed")])
    + middle_night("2022-05-18 23:50:00", 70.0, 4, 75.0)
)

NOTES = {
    "2026-07-14": ("the measured win: true onset 23:26:58 on the 13th, "
                   "stored ~23:59"),
    "2026-08-15": "a recent night, for the wake-time invariance check",
    "2026-07-05": ("a night whose first sample already crosses midnight — "
                   "nothing to move"),
    "2017-03-15": "one span per night: end-dating is already correct here",
    "2021-11-10": "the untested middle — 61 min samples, zero moves",
    "2022-05-18": ("the untested middle — 75 min samples, and the one "
                   "21-hour midnight-crossing episode"),
}


def pre_fix_derived(rows: list[dict], day: str) -> dict:
    """What the OLD per-sample attribution produced for `day`.

    Intervals are grouped by the stored `local_date`, which is the defect;
    this is the number the fix has to be measured against.
    """
    ivs = []
    for row in rows:
        if row["metric"] not in D._STAGE_METRICS or row["local_date"] != day:
            continue
        start = datetime.fromisoformat(row["start_local"])
        ivs.append(D.Interval(start, start + timedelta(minutes=row["value"]),
                              row["metric"]))
    ivs.sort(key=lambda i: i.start)
    out = D.compute_sleep_timing(ivs, day) or {}
    return {k: round(v, 4) for k, v in out.items()}


data = {}
for day, rows in records.items():
    prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    kept = sorted((r for r in rows if prev <= r["local_date"] <= nxt),
                  key=lambda r: (r["start_local"], r["id"]))
    data[day] = {
        "note": NOTES[day],
        "window": [prev, nxt],
        "stored_derived": pre_fix_derived(kept, day),
        "records": kept,
    }

payload = {
    "provenance": {
        "captured": "synthetic",
        "source": "invented; no vault was read",
        "script": "generated for the public extraction, not by capture.py",
        "function": "capture_sleep_nights",
        "docstring": (
            "SYNTHETIC replacement for the live capture. The real fixture held "
            "genuine sleep timestamps back to 2017 and could not be published. "
            "Shape, semantics and every condition the E7-1/E7-2 tests measure "
            "are reproduced: `local_date` is assigned per sample as the date "
            "the SAMPLE ends, each night is captured with its D-1 and D+1 "
            "neighbours because the fix regroups across exactly that boundary, "
            "and `stored_derived` is what the pre-fix per-sample attribution "
            "computed from these rows."),
    },
    "data": data,
}

(ROOT / "tests/fixtures/sleep_nights.json").write_text(
    json.dumps(payload, indent=1, sort_keys=True) + "\n")
for day, entry in data.items():
    print(day, len(entry["records"]), entry["stored_derived"])
