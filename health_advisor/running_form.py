"""Per-workout running form measures, over jog buckets only.

WHAT THIS IS NOT. These are not "aerobic decoupling". That construct is defined
on prolonged, reasonably steady efforts, and its familiar <5% threshold is a
coaching heuristic from that setting. The sessions these measures were built
for are 35-50 minute run/walk intervals containing 10-25 minutes of actual
jogging. Naming a proxy after the
validated construct it approximates is how an unvalidated number acquires
borrowed authority, so these measures are named for what they compute and no
literature threshold ships with them. See the design spec, section 11.2.

WHAT IT IS. Efficiency is speed_mph / heart_rate — never HR / pace, which is
dimensionally inverted and moves the wrong way. Halves are split on cumulative
JOG time, not clock time: splitting a run/walk session at the halfway clock mark
measures where the walk breaks fell, not cardiac drift.

WHAT IT CANNOT CONTROL FOR. Terrain, grade, heat, humidity, surface and GPS
quality are all absent from the export. Every output here is descriptive.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np

from . import metrics as mx
from . import vault as V

MIN_JOG_MINUTES = 10.0   # below this a half is too short to compare against
MIN_HALF_BUCKETS = 5     # HR-bearing jog buckets needed in EACH half


def _pace_min_per_mi(bucket, metric_units: bool = False):
    if metric_units and bucket.get("pace_min_per_km") is not None:
        return bucket["pace_min_per_km"] * V.UNIT_CONVERSION_FACTORS["distance_mi_to_km"]
    return bucket.get("pace_min_per_mi")


def _speed_mph(bucket, metric_units: bool = False):
    if metric_units and bucket.get("speed_kph") is not None:
        return bucket["speed_kph"] / V.UNIT_CONVERSION_FACTORS["distance_mi_to_km"]
    return bucket.get("speed_mph")


def _collapse_bucket_rows(buckets, metric_units: bool = False):
    """Collapse local-date splits of one UTC bucket before measuring it."""
    grouped = {}
    for bucket in buckets:
        grouped.setdefault(bucket["bucket_start_utc"], []).append(bucket)
    out = []
    bucket_min = mx.IMPACT_BUCKET_SECONDS / 60.0
    for start in sorted(grouped):
        rows = grouped[start]
        miles = sum(float(row.get("miles") or 0.0) for row in rows)
        hrs = [float(row["hr"]) for row in rows if row.get("hr") is not None]
        hr = sum(hrs) / len(hrs) if hrs else None
        pace = bucket_min / miles if miles > 0 else None
        is_jog = bool(
            pace is not None
            and pace >= mx.IMPACT_IMPLAUSIBLE_PACE_MIN
            and (pace <= mx.IMPACT_JOG_PACE_MAX
                 or (pace <= mx.IMPACT_JOG_HR_PACE_MAX
                     and (hr or 0.0) >= mx.IMPACT_JOG_HR_MIN)))
        is_walk = bool(not is_jog and pace is not None
                       and pace >= mx.IMPACT_IMPLAUSIBLE_PACE_MIN
                       and pace <= mx.IMPACT_WALK_PACE_MAX)
        speed = (miles / (bucket_min / 60.0)) if miles > 0 else None
        out.append({"bucket_start_utc": start,
                    "local_date": rows[0].get("local_date"),
                    "miles": miles,
                    "hr": hr,
                    **({
                        "speed_kph": (speed * V.UNIT_CONVERSION_FACTORS[
                            "distance_mi_to_km"] if speed is not None else None),
                        "pace_min_per_km": (pace / V.UNIT_CONVERSION_FACTORS[
                            "distance_mi_to_km"] if pace is not None else None),
                    } if metric_units else {
                        "speed_mph": speed,
                        "pace_min_per_mi": pace,
                    }),
                    "is_jog": is_jog,
                    "is_walk": is_walk})
    return out


def _efficiency(buckets, metric_units: bool = False) -> float | None:
    """Mean speed/HR over buckets that have both. None if none do."""
    vals = [_speed_mph(b, metric_units) / b["hr"]
            for b in buckets
            if b.get("hr") and _speed_mph(b, metric_units)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def efficiency_change_from_buckets(buckets, metric_units: bool = False) -> dict:
    """Percentage change in speed/HR efficiency, first vs second half of
    cumulative jog time. Negative means efficiency fell across the session.

    Pure function over bucket_series rows so it can be tested without a DB.
    """
    buckets = _collapse_bucket_rows(buckets, metric_units)
    buckets_dropped_implausible = sum(
        1 for b in buckets
        if not b.get("is_jog") and not b.get("is_walk")
        and _pace_min_per_mi(b, metric_units) is not None
        and _pace_min_per_mi(b, metric_units) < mx.IMPACT_IMPLAUSIBLE_PACE_MIN
    )
    jog = [b for b in buckets if b.get("is_jog")]
    jog_minutes = len(jog) * (mx.IMPACT_BUCKET_SECONDS / 60.0)
    if jog_minutes < MIN_JOG_MINUTES:
        return {"status": "insufficient_jog_time",
                "reason": f"{jog_minutes:.1f} min of jogging; need "
                          f"{MIN_JOG_MINUTES:.0f}",
                "jog_minutes": round(jog_minutes, 1)}

    mid = len(jog) // 2
    first, second = jog[:mid], jog[mid:]
    n_first = sum(1 for b in first if b.get("hr") and _speed_mph(b, metric_units))
    n_second = sum(1 for b in second if b.get("hr") and _speed_mph(b, metric_units))
    if n_first < MIN_HALF_BUCKETS or n_second < MIN_HALF_BUCKETS:
        return {"status": "insufficient_half_coverage",
                "reason": f"{n_first} and {n_second} usable buckets in the two "
                          f"halves; need {MIN_HALF_BUCKETS} in each",
                "jog_minutes": round(jog_minutes, 1)}

    e1, e2 = _efficiency(first, metric_units), _efficiency(second, metric_units)
    return {
        "status": "ok",
        "change_pct": mx.r((e2 - e1) / e1 * 100.0),
        "first_half_efficiency": mx.r(e1, 4),
        "second_half_efficiency": mx.r(e2, 4),
        "first_half_buckets": n_first,
        "second_half_buckets": n_second,
        "jog_minutes": round(jog_minutes, 1),
        "buckets_dropped_implausible": buckets_dropped_implausible,
    }


def jog_efficiency_change(conn, start_utc: str, end_utc: str, *,
                          metric_units: bool = False) -> dict:
    """efficiency_change_from_buckets over one workout's window."""
    return efficiency_change_from_buckets(
        mx.bucket_series(conn, start_utc, end_utc, metric_units=metric_units),
        metric_units=metric_units)


def walk_structure_from_buckets(buckets, metric_units: bool = False) -> dict:
    """Walk fraction and bout structure across the session, and how it is
    distributed early vs late.

    This exists because the efficiency measure excludes walk buckets, and in a
    run/walk session the walking IS part of the signal: walks getting longer or
    more frequent late is exactly what fatigue looks like here. Excluding them
    from the efficiency ratio is correct (a walk makes pace enormous and would
    swamp the number); discarding them entirely is not.

    Bouts are runs of consecutive walk buckets, so a three-bucket walk is one
    bout of a minute, not three events. Halves split on cumulative jog time, the
    same as efficiency_change_from_buckets, so the two are directly comparable.
    """
    buckets = _collapse_bucket_rows(buckets, metric_units)
    moving = [b for b in buckets if b.get("is_jog") or b.get("is_walk")]
    jog = [b for b in moving if b.get("is_jog")]
    bucket_min = mx.IMPACT_BUCKET_SECONDS / 60.0
    jog_minutes = len(jog) * bucket_min
    if jog_minutes < MIN_JOG_MINUTES:
        return {"status": "insufficient_jog_time",
                "reason": f"{jog_minutes:.1f} min of jogging; need "
                          f"{MIN_JOG_MINUTES:.0f}",
                "jog_minutes": round(jog_minutes, 1)}

    bouts, in_bout = 0, False
    for b in moving:
        if b.get("is_walk"):
            if not in_bout:
                bouts += 1
                in_bout = True
        else:
            in_bout = False

    walk_buckets = sum(1 for b in moving if b.get("is_walk"))
    walk_minutes = walk_buckets * bucket_min

    # Split the MOVING sequence at the bucket where cumulative jog time passes
    # halfway, so the halves line up with efficiency_change_from_buckets.
    half_jog, seen, split_at = len(jog) / 2.0, 0, len(moving)
    for i, b in enumerate(moving):
        if b.get("is_jog"):
            seen += 1
            if seen >= half_jog:
                split_at = i + 1
                break
    first, second = moving[:split_at], moving[split_at:]

    def _frac(seq):
        return (sum(1 for b in seq if b.get("is_walk")) / len(seq)) if seq else 0.0

    return {
        "status": "ok",
        "walk_fraction": mx.r(walk_buckets / len(moving), 3),
        "walk_bouts": bouts,
        "mean_bout_minutes": mx.r(walk_minutes / bouts, 2) if bouts else 0.0,
        "first_half_walk_fraction": mx.r(_frac(first), 3),
        "second_half_walk_fraction": mx.r(_frac(second), 3),
        "walk_minutes": round(walk_minutes, 1),
        "jog_minutes": round(jog_minutes, 1),
    }


def walk_structure(conn, start_utc: str, end_utc: str, *,
                   metric_units: bool = False) -> dict:
    """walk_structure_from_buckets over one workout's window."""
    return walk_structure_from_buckets(
        mx.bucket_series(conn, start_utc, end_utc, metric_units=metric_units),
        metric_units=metric_units)


# Where Week 5's third gear landed. PARTLY POST HOC: chosen from an observed
# training week, so it is a descriptive reference, not an independent standard.
# Moving it makes the series before and after incomparable — if it ever moves,
# say so where the number is reported.
REFERENCE_PACE_BAND = (13.0, 15.0)   # min/mi, inclusive
MIN_BAND_BUCKETS = 10                # in-band jog buckets needed in a week
MIN_TREND_WEEKS = 3

CAVEAT = ("Descriptive only: nothing here controls for terrain, grade, heat, "
          "humidity, surface or GPS quality, none of which are in the export. "
          "The reference band was chosen from an observed training week.")


def in_reference_band(buckets, band=REFERENCE_PACE_BAND, *,
                      metric_units: bool = False) -> list[dict]:
    """Jog buckets whose pace falls inside the reference band, and which carry
    both HR and speed."""
    buckets = _collapse_bucket_rows(buckets, metric_units)
    lo, hi = band
    return [b for b in buckets
            if b.get("is_jog") and b.get("hr") and _speed_mph(b, metric_units)
            and _pace_min_per_mi(b, metric_units) is not None
            and lo <= _pace_min_per_mi(b, metric_units) <= hi]


def trend_from_weeks(weeks) -> dict:
    """Slopes over a list of weekly rows. Pure, so it is testable without a DB."""
    if len(weeks) < MIN_TREND_WEEKS:
        return {"status": "insufficient_weeks",
                "reason": f"{len(weeks)} week(s) with >={MIN_BAND_BUCKETS} "
                          f"in-band buckets; need {MIN_TREND_WEEKS}",
                "band_min_per_mi": list(REFERENCE_PACE_BAND),
                "weeks": weeks, "caveat": CAVEAT}
    dates = [w["week_start"] for w in weeks]
    return {
        "status": "ok",
        "band_min_per_mi": list(REFERENCE_PACE_BAND),
        "weeks": weeks,
        "efficiency_slope_per_week": mx.slope_per_week(
            dates, [w["efficiency"] for w in weeks]),
        "hr_slope_per_week": mx.slope_per_week(
            dates, [w["mean_hr"] for w in weeks]),
        "caveat": CAVEAT,
    }


def banded_weekly(conn, start: str, end: str, *,
                  metric_units: bool = False) -> dict:
    """Weekly mean efficiency (speed/HR) and mean HR within the reference pace
    band, Monday-anchored, plus their slopes.

    One bucket_series call per running workout rather than one over the whole
    range: bucket_series is workout-scale by design.
    """
    runs = conn.execute(
        "SELECT start_utc, end_utc, local_date FROM workouts "
        "WHERE workout_type = 'running' AND local_date BETWEEN ? AND ? "
        "ORDER BY start_utc", (start, end)).fetchall()

    # Deduplicate by bucket identity. The DB contains NESTED workouts — on
    # 2026-06-18 a 1.3-minute run sits entirely inside a 37.9-minute one, sharing
    # every bucket — so accumulating per workout double-counts the overlap. Found
    # while verifying Task 2: the naive sum exceeded impact_volume for that day,
    # which is impossible for a subset of the same day's buckets.
    by_week: dict[str, dict[tuple, dict]] = {}
    for w in runs:
        band = in_reference_band(
            mx.bucket_series(conn, w["start_utc"], w["end_utc"],
                             metric_units=metric_units),
            metric_units=metric_units)
        if not band:
            continue
        d = date.fromisoformat(w["local_date"])
        key = (d - timedelta(days=d.weekday())).isoformat()
        week = by_week.setdefault(key, {})
        for b in band:
            week[(b["local_date"], b["bucket_start_utc"])] = b

    weeks = []
    for key in sorted(by_week):
        bs = list(by_week[key].values())
        if len(bs) < MIN_BAND_BUCKETS:
            continue
        weeks.append({
            "week_start": key,
            "efficiency": mx.r(sum(_speed_mph(b, metric_units) / b["hr"]
                                    for b in bs) / len(bs), 4),
            "mean_hr": mx.r(sum(b["hr"] for b in bs) / len(bs)),
            "buckets": len(bs),
        })
    return trend_from_weeks(weeks)


MIN_REFERENCE_SESSIONS = 5


def reference_from_changes(changes) -> dict:
    """The athlete's own distribution of efficiency change, and the smallest
    change that would stand out from it.

    No literature threshold is applied. The <5% figure associated with aerobic
    decoupling belongs to prolonged steady-state efforts and would put borrowed
    authority on a different quantity. What IS answerable from this data is
    whether a session sits outside the athlete's own normal range.

    minimum_detectable_change_pct is 2 * SEM: roughly the smallest shift that
    would clear the noise in that athlete's own history. It falls as sessions
    accumulate, which is the honest form of "this gets better with more data".
    """
    vals = [c for c in changes if c is not None]
    if len(vals) < MIN_REFERENCE_SESSIONS:
        return {"status": "insufficient_sessions",
                "reason": f"{len(vals)} comparable session(s); need "
                          f"{MIN_REFERENCE_SESSIONS}",
                "n_sessions": len(vals)}
    arr = np.array(vals, dtype=float)
    sem = float(arr.std(ddof=1)) / (len(arr) ** 0.5)
    return {
        "status": "ok",
        "n_sessions": len(vals),
        "median_change_pct": mx.r(np.median(arr)),
        "p10": mx.r(np.percentile(arr, 10)),
        "p90": mx.r(np.percentile(arr, 90)),
        "minimum_detectable_change_pct": mx.r(2.0 * sem),
    }


def personal_reference(conn, start: str, end: str,
                       exclude_start_utc: str | None = None, *,
                       metric_units: bool = False) -> dict:
    """reference_from_changes over every comparable run in a date range.

    `exclude_start_utc` leaves a session out, so a run can be compared against
    a reference range it did not itself help define.
    """
    runs = conn.execute(
        "SELECT start_utc, end_utc FROM workouts WHERE workout_type = 'running' "
        "AND local_date BETWEEN ? AND ? ORDER BY start_utc", (start, end)).fetchall()
    changes = []
    kept: list[tuple[str, str]] = []
    for w in runs:
        if exclude_start_utc and w["start_utc"] == exclude_start_utc:
            continue
        # Skip a window nested inside one already counted. The DB has these —
        # 2026-06-18 carries a 1.3-minute run wholly inside a 37.9-minute one —
        # and letting both in would put a near-duplicate session into the
        # reference distribution the current run is judged against.
        if any(s <= w["start_utc"] and w["end_utc"] <= e for s, e in kept):
            continue
        out = jog_efficiency_change(conn, w["start_utc"], w["end_utc"],
                                    metric_units=metric_units)
        if out["status"] == "ok":
            changes.append(out["change_pct"])
            kept.append((w["start_utc"], w["end_utc"]))
    return reference_from_changes(changes)


# P1-10 / W7-6: watts are the adopted economy readout. Keep the existing
# 13--15 min/mi band so month-to-month power is not a disguised pace change.
RUNNING_POWER_LAG_SECONDS = 3 * 60


def _epoch_seconds(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00"))
                       .astimezone(timezone.utc).timestamp())


def _running_power_buckets(conn, start_utc: str, end_utc: str) -> dict[tuple[str, int], float]:
    rows = conn.execute(
        "SELECT local_date, CAST(strftime('%s', start_utc) / ? AS INT) AS bkt, "
        "AVG(value) AS power FROM records "
        "WHERE metric = 'running_power' AND start_utc >= ? AND start_utc < ? "
        "GROUP BY local_date, bkt",
        (mx.IMPACT_BUCKET_SECONDS, start_utc, end_utc),
    ).fetchall()
    return {(row["local_date"], int(row["bkt"])): float(row["power"])
            for row in rows if row["power"] is not None}


def _matched_power_for_run(conn, start_utc: str, end_utc: str,
                           pace_band: tuple[float, float], *,
                           metric_units: bool = False) -> float | None:
    buckets = mx.bucket_series(conn, start_utc, end_utc,
                               metric_units=metric_units)
    powers = _running_power_buckets(conn, start_utc, end_utc)
    if not buckets:
        return None

    lo, hi = pace_band
    matched = []
    previous_start = None
    block_elapsed = 0
    for bucket in buckets:
        start = _epoch_seconds(bucket["bucket_start_utc"])
        contiguous = previous_start is not None and start - previous_start == mx.IMPACT_BUCKET_SECONDS
        if not bucket.get("is_jog"):
            block_elapsed = 0
        elif contiguous:
            block_elapsed += mx.IMPACT_BUCKET_SECONDS
        else:
            block_elapsed = 0

        if (bucket.get("is_jog")
                and block_elapsed >= RUNNING_POWER_LAG_SECONDS
                and _pace_min_per_mi(bucket, metric_units) is not None
                and lo <= _pace_min_per_mi(bucket, metric_units) <= hi):
            key = (bucket["local_date"], start // mx.IMPACT_BUCKET_SECONDS)
            if key in powers:
                matched.append(powers[key])
        previous_start = start

    return sum(matched) / len(matched) if matched else None


def monthly_running_power(conn, month: str,
                           pace_band: tuple[float, float] = REFERENCE_PACE_BAND, *,
                           metric_units: bool = False) -> float | None:
    """Mean matched-pace power for a month, or ``None`` with fewer than two runs.

    Only route-backed Apple Watch running workouts are outdoor evidence. Each
    qualifying run contributes one mean, so a long run cannot dominate the
    month. The first three minutes of every continuous jog block are excluded
    because HR-lag analysis showed those buckets are not settled-state evidence
    (AUDIT-1 §6).
    """
    first = date.fromisoformat(f"{month}-01")
    if first.month == 12:
        next_month = date(first.year + 1, 1, 1)
    else:
        next_month = date(first.year, first.month + 1, 1)
    end_month = next_month.isoformat()

    runs = conn.execute(
        "SELECT start_utc, end_utc, local_date FROM workouts "
        "WHERE workout_type = 'running' AND route_ref IS NOT NULL "
        "AND local_date >= ? AND local_date < ? ORDER BY start_utc",
        (first.isoformat(), end_month),
    ).fetchall()
    kept: list[tuple[str, str]] = []
    run_means = []
    for run in runs:
        if any(start <= run["start_utc"] and run["end_utc"] <= end
               for start, end in kept):
            continue
        value = _matched_power_for_run(
            conn, run["start_utc"], run["end_utc"], pace_band,
            metric_units=metric_units,
        )
        if value is not None:
            run_means.append(value)
            kept.append((run["start_utc"], run["end_utc"]))

    if len(run_means) < 2:
        return None
    return mx.r(sum(run_means) / len(run_means), 2)
