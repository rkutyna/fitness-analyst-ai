#!/usr/bin/env python3
"""Capture live figures the audits published, as fixtures tests can run against.

`tests/conftest.py` makes opening the production database an AssertionError, so
every validation number the eight-part audit produced -- the block lengths, the
"ran hot" backtest, the resting-HR revision series -- was unrunnable as a test
and could only be re-checked by hand. That is how two published figures came to
not reproduce in the first place.

This is a SCRIPT, not a test: it opens the production DB read-only and writes
JSON under tests/fixtures/. Run it when a fixture needs refreshing, and commit
the result. Each file carries a `provenance` block naming the query, the capture
date, and the figure it is expected to reproduce, so a fixture that stops
matching its own audit is visible rather than silently authoritative.

    ./.venv/bin/python tests/fixtures/capture.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from health_advisor import metrics as mx     # noqa: E402

OUT = Path(__file__).resolve().parent
DB = ROOT / "data" / "health.db"

# Workouts the audits' block figures were measured on. Expected values are from
# AUDIT-1 §2 and AUDIT-2, independently reproduced 2026-08-16 by a second
# implementation written from PLAN.md's prose alone.
BLOCK_WORKOUTS = {
    "2026-06-22": {"expect_bridged_min": 6.7,
                   "note": "AUDIT-2 validation case"},
    "2026-07-22": {"expect_avg_hr_longest_block": 169.2,
                   "note": "block mean HR above the 150 ceiling — must not qualify"},
    "2026-08-15": {"expect_unbridged_min": 4.0, "expect_bridged_min": 9.3,
                   "expect_avg_hr_longest_block": 145.0,
                   "note": "the bridge rule changes exactly this one session"},
    "2026-07-17": {"note": "two running workouts on one day — daily dial semantics"},
}


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def capture_blocks(conn) -> dict:
    """Bucket series per running workout — the only input longest_block needs.

    Buckets rather than raw records: a 68-minute run is 4,702 records but 201
    buckets, and the block rule is defined on buckets. Keeping the fixture at
    the level the algorithm consumes makes it readable and diffable.
    """
    out = {}
    for day, meta in BLOCK_WORKOUTS.items():
        rows = conn.execute(
            "SELECT id, start_utc, end_utc, duration_min, avg_heart_rate "
            "FROM workouts WHERE local_date = ? AND workout_type = 'running' "
            "ORDER BY start_utc", (day,)).fetchall()
        sessions = []
        for w in rows:
            buckets = mx.bucket_series(conn, w["start_utc"], w["end_utc"])
            sessions.append({
                "workout_id": w["id"],
                "start_utc": w["start_utc"], "end_utc": w["end_utc"],
                "duration_min": w["duration_min"],
                "avg_hr_session": w["avg_heart_rate"],
                "buckets": [dict(b) for b in buckets],
            })
        out[day] = {**meta, "sessions": sessions}
    return out


def capture_ran_hot(conn) -> dict:
    """The seven sessions the easy-band warning actually fired on (F4-1).

    Six were false. The backtest needs each session's whole-session average and
    its jog-only average, which are the two numbers the fix swaps between.
    """
    dates = ["2026-07-02", "2026-07-17", "2026-07-19", "2026-07-26",
             "2026-08-09", "2026-08-12", "2026-08-13"]
    out = {}
    for day in dates:
        rows = conn.execute(
            "SELECT id, workout_type, start_utc, end_utc, duration_min, "
            "avg_heart_rate, max_heart_rate FROM workouts "
            "WHERE local_date = ? ORDER BY duration_min DESC", (day,)).fetchall()
        sessions = []
        for w in rows:
            buckets = mx.bucket_series(conn, w["start_utc"], w["end_utc"])
            jog = [b["hr"] for b in buckets if b["is_jog"] and b["hr"] is not None]
            sessions.append({
                "workout_id": w["id"], "workout_type": w["workout_type"],
                "duration_min": w["duration_min"],
                "avg_hr_session": w["avg_heart_rate"],
                "max_hr_session": w["max_heart_rate"],
                "avg_hr_all_jog": round(sum(jog) / len(jog), 2) if jog else None,
                "n_jog_buckets": len(jog),
            })
        out[day] = sessions
    return out


def capture_revision_series(conn) -> dict:
    """resting_heart_rate / walking_heart_rate_average over the plan window.

    Apple re-emits both as same-day revisions; the catalog averaged the drafts.
    AUDIT-6 §2: disagrees on 27 of 54 days, 07-11 stored 58.2 but settled 63.0,
    08-02 stored 60.0 but settled 58.0.
    """
    out = {}
    for metric in ("resting_heart_rate", "walking_heart_rate_average"):
        out[metric] = [dict(r) for r in conn.execute(
            "SELECT date, count, avg, last FROM daily_metrics "
            "WHERE metric = ? AND date >= '2026-06-22' ORDER BY date", (metric,))]
    return out


def capture_body_mass(conn) -> dict:
    """Every weigh-in in the plan window, for the fitted-slope work (F5-1)."""
    return {"readings": [dict(r) for r in conn.execute(
        "SELECT local_date, start_local, value FROM records "
        "WHERE metric = 'body_mass' AND local_date >= '2026-06-22' "
        "ORDER BY start_utc")]}


# Nights the sleep-attribution work (E7-1) is measured on. Two 2026 nights where
# the episode crosses midnight and is split into 20-40 samples; one 2026 night
# that does not cross; and three nights from the eras the change has to survive
# — 2017 (one span per night, 427 min average), 2021 and 2022 (the untested
# middle, 61-75 min samples). Captured as RAW records, because the defect is in
# how a raw sample is dated and a fixture of the derived output would encode it.
SLEEP_NIGHTS = [
    ("2026-07-14", "the measured win: true onset 23:19 on the 13th, stored ~00:0x"),
    ("2026-08-15", "a recent night, for the wake-time invariance check"),
    ("2026-07-05", "a night whose episode begins after midnight — must not move"),
    ("2017-03-15", "one span per night: end-dating is already correct here"),
    ("2021-11-10", "the untested middle — 61 min samples, partially affected"),
    ("2022-05-18", "the untested middle — 75 min samples, partially affected"),
]


def capture_sleep_nights(conn) -> dict:
    """Raw sleep records around each night in SLEEP_NIGHTS (E7-1, E7-2).

    `local_date` is assigned per sample as the date the SAMPLE ends. That was
    right when HealthKit gave one span per night; in 2026 a night is 20-40
    samples, so every sample ending before midnight is filed under the previous
    date. Each night is captured with its neighbours (D-1 and D+1) because the
    fix regroups samples across exactly that boundary, and a one-day window
    could not show the regrouping at all.
    """
    out = {}
    for day, note in SLEEP_NIGHTS:
        prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        rows = conn.execute(
            "SELECT id, metric, value, unit, start_utc, end_utc, start_local, "
            "local_date, source, origin, dedupe_key FROM records "
            "WHERE metric IN ('sleep_asleep', 'sleep_awake', 'sleep_in_bed') "
            "AND local_date BETWEEN ? AND ? ORDER BY start_local, id",
            (prev, nxt)).fetchall()
        stored = {r["metric"]: r["last"] for r in conn.execute(
            "SELECT metric, last FROM daily_metrics WHERE date = ? "
            "AND metric LIKE 'sleep_%'", (day,))}
        out[day] = {"note": note, "window": [prev, nxt],
                    "stored_derived": stored,
                    "records": [dict(r) for r in rows]}
    return out


def main() -> int:
    if not DB.exists():
        raise SystemExit(f"no database at {DB}")
    conn = _conn()
    captures = {
        "blocks": capture_blocks,
        "ran_hot": capture_ran_hot,
        "revision_series": capture_revision_series,
        "body_mass": capture_body_mass,
        "sleep_nights": capture_sleep_nights,
    }
    for name, fn in captures.items():
        payload = {
            "provenance": {
                "captured": date.today().isoformat(),
                "source": "data/health.db (read-only)",
                "script": "tests/fixtures/capture.py",
                "function": fn.__name__,
                "docstring": (fn.__doc__ or "").strip(),
            },
            "data": fn(conn),
        }
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
