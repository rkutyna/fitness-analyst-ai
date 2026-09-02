"""A session-scoped heart-rate load proxy — an intensity-aware replacement for
active_energy as the input to analysis.training_load's acute:chronic ratio.

NOT NAMED TRIMP, DELIBERATELY. The functional form is Banister's, but the
published formula assumes HR_rest measured under a defined resting protocol and
HR_max from maximal exercise testing. Both are estimated here from observational
data, and heart rate during run/walk intervals is not continuous-exercise
intensity. It is a proxy, it is named as one, and it should be reported as one.

WORKOUT-SCOPED, NEVER DAILY. Summing over all of a day's heart-rate samples
would fold in sleep, sitting, errands and sensor artifacts. Load is computed per
workout and summed per day.

MISSING IS UNKNOWN, NOT ZERO. A day whose sessions lack heart-rate coverage
reports load None with status 'unknown'. analysis.py line 62 records what the
other choice already cost once: a week of non-wear reported as
"acwr 0.08, detraining".
"""
from __future__ import annotations

import math

import numpy as np

from . import metrics as mx

MIN_SESSION_HR_SAMPLES = 30
MIN_HR_COVERAGE_FRACTION = 0.5
HR_MAX_PERCENTILE = 99.5
HR_MAX_WINDOW_DAYS = 365
BANISTER_MALE_SCALE = 0.64
BANISTER_EXP = 1.92


def session_load(duration_min, mean_hr, hr_rest, hr_max) -> float | None:
    """Banister-form load for one session. None if any input is missing or the
    HR range is degenerate."""
    if None in (duration_min, mean_hr, hr_rest, hr_max):
        return None
    if hr_max <= hr_rest:
        return None
    ratio = (float(mean_hr) - float(hr_rest)) / (float(hr_max) - float(hr_rest))
    ratio = max(0.0, ratio)   # below resting is zero load, never negative
    return (float(duration_min) * ratio * BANISTER_MALE_SCALE
            * math.exp(BANISTER_EXP * ratio))


def hr_max_estimate(conn, as_of: str) -> float | None:
    """99.5th percentile of heart_rate over a trailing window.

    A percentile rather than the raw maximum: one artifactual 210 bpm sample
    would otherwise set the ceiling for a year. This is still an ESTIMATE — a
    true HR_max needs a maximal effort, and nothing here guarantees one
    occurred.
    """
    rows = conn.execute(
        "SELECT value FROM records WHERE metric = 'heart_rate' "
        "AND local_date <= ? AND local_date >= date(?, ?)",
        (as_of, as_of, f"-{HR_MAX_WINDOW_DAYS} days")).fetchall()
    vals = [r["value"] for r in rows if r["value"] is not None]
    if len(vals) < 100:
        rows = conn.execute(
            "SELECT value FROM records WHERE metric = 'heart_rate' "
            "AND local_date <= ?", (as_of,)).fetchall()
        vals = [r["value"] for r in rows if r["value"] is not None]
    if len(vals) < 100:
        return None
    return float(np.percentile(np.array(vals, dtype=float), HR_MAX_PERCENTILE))


def hr_rest_estimate(conn, as_of: str) -> float | None:
    """Rolling baseline of resting_heart_rate. Drifts with fitness, which is
    correct — but note it also means a historical load recomputed today can
    differ slightly from the same day computed months ago."""
    rows = conn.execute(
        "SELECT last FROM daily_metrics WHERE metric = 'resting_heart_rate' "
        "AND date <= ? AND last IS NOT NULL ORDER BY date DESC LIMIT 60",
        (as_of,)).fetchall()
    vals = [r["last"] for r in rows][::-1]
    if not vals:
        return None
    return mx.baseline(vals, exclude_recent=0, window=28)


def _session_mean_hr(conn, start_utc: str, end_utc: str, duration_min: float):
    row = conn.execute(
        "SELECT AVG(value) AS m, COUNT(*) AS n, "
        "(julianday(MAX(start_utc)) - julianday(MIN(start_utc))) * 1440.0 "
        "AS span_min FROM records "
        "WHERE metric = 'heart_rate' AND start_utc >= ? AND start_utc < ?",
        (start_utc, end_utc)).fetchone()
    if not row or not row["n"] or row["n"] < MIN_SESSION_HR_SAMPLES:
        return None, int(row["n"] or 0) if row else 0
    if duration_min is None:
        return None, int(row["n"])
    if (row["span_min"] or 0.0) < MIN_HR_COVERAGE_FRACTION * float(duration_min):
        return None, int(row["n"])
    return float(row["m"]), int(row["n"])


def daily_load(conn, start: str, end: str,
               hr_rest: float | None = None,
               hr_max: float | None = None) -> list[dict]:
    """Per-day summed session load. One row per day that had any workout.

    Days with workouts but no usable heart-rate coverage report load None and
    status 'unknown' — they are days we cannot measure, not days of no training.
    """
    workouts = conn.execute(
        "SELECT local_date, start_utc, end_utc, duration_min FROM workouts "
        "WHERE local_date BETWEEN ? AND ? ORDER BY local_date, start_utc",
        (start, end)).fetchall()

    by_day: dict[str, dict] = {}
    kept_by_day: dict[str, list[tuple[str, str]]] = {}
    for w in workouts:
        day = w["local_date"]
        d = by_day.setdefault(day, {"date": day, "load": 0.0, "sessions": 0,
                                    "sessions_without_hr": 0,
                                    "sessions_nested_skipped": 0, "_any": False})
        d["sessions"] += 1
        kept = kept_by_day.setdefault(day, [])
        if any(s <= w["start_utc"] and w["end_utc"] <= e for s, e in kept):
            d["sessions_nested_skipped"] += 1
            continue
        kept.append((w["start_utc"], w["end_utc"]))
        mean_hr, _n = _session_mean_hr(conn, w["start_utc"], w["end_utc"],
                                       w["duration_min"])
        rest = hr_rest if hr_rest is not None else hr_rest_estimate(conn, day)
        peak = hr_max if hr_max is not None else hr_max_estimate(conn, day)
        load = session_load(w["duration_min"], mean_hr, rest, peak)
        if load is None:
            d["sessions_without_hr"] += 1
        else:
            d["load"] += load
            d["_any"] = True

    out = []
    for day in sorted(by_day):
        d = by_day[day]
        any_load = d.pop("_any")
        d["status"] = "ok" if any_load else "unknown"
        d["load"] = mx.r(d["load"], 2) if any_load else None
        out.append(d)
    return out
