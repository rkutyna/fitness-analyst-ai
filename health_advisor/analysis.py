"""Deterministic Briefing builder — the analytical 'brain'. Everything the model
might get wrong (which metrics matter, the math, what moved) is computed here.
Read-only over the DB; returns plain JSON-able dicts."""
from __future__ import annotations

import math
import sqlite3
import statistics
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import db
from . import metrics as mx
from . import vault as V
from .metrics import WEAR_MIN_HOURS

# Literature figures are data, not instructions for the model to calculate.
# Keep the citation beside every published value so renderers can show it.
LITERATURE_FIGURES = {
    "heat_effect_bpm_per_c": {
        "value": 1.0,
        "unit": "bpm/°C",
        "citation": {
            "author": "K B Pandolf; E Cafarelli; B J Noble; K F Metz",
            "year": 1975,
            "venue": "Arch Phys Med Rehabil",
            "pmid": "1200826",
        },
    },
    "expected_training_effect_bpm": {
        "min": 0.0,
        "max": 3.0,
        "unit": "bpm",
        "citation": {
            "author": "A K Reimers; G Knapp; C D Reimers",
            "year": 2018,
            "venue": "J Clin Med",
            "doi": "10.3390/jcm7120503",
        },
    },
}

# --- tunable constants (kept here so tuning is one place) -------------------- #
VITALS = ["resting_heart_rate", "heart_rate_variability", "sleep_asleep",
          "respiratory_rate", "blood_oxygen_saturation", "step_count",
          "active_energy", "vo2_max", "body_mass"]
SPARSE_DAYS = 14
STALE_DAYS = 14
# "Sparse" is a fraction of the window, not every day of it. Requiring all 14
# labelled a 2,532-day metric sparse because it was absent for one day — the
# same all-or-nothing brittleness that took ACWR dark, in the section that tells
# the narrator which numbers to distrust.
COVERAGE_MIN_FRACTION = 0.7
COVERAGE_CADENCE_DAYS = 60

# The complete vocabulary coverage() can return, named once so a consumer cannot
# filter on a status that will never occur. It could, and did: talking_points()
# selected ("sparse", "establishing") — a status coverage() has never emitted —
# while excluding "stale" and "missing", the two that mean a tracked metric has
# gone silent. vo2_max was dropped at ingest for 16 days and no seed said so.
# Audit part 6, F6-2.
COVERAGE_STATUSES = ("missing", "stale", "sparse", "active")
COVERAGE_STOPPED = ("stale", "missing")     # the metric is not arriving
COVERAGE_THIN = ("sparse",)                 # arriving, but too little to trend
assert set(COVERAGE_STOPPED) <= set(COVERAGE_STATUSES)
assert set(COVERAGE_THIN) <= set(COVERAGE_STATUSES)

READINESS_WEIGHTS = {"hrv": 0.40, "rhr": 0.35, "sleep": 0.25}
READINESS_MIN_BASELINE_DAYS = 14
READINESS_BASELINE_WINDOW = 28
SLEEP_TARGET_MIN = 7.5 * 60
# A subscore moves SUBSCORE_K points per 1% deviation from baseline. At 2.5 it
# saturated at ±20%, and HRV's measured daily CV is 17.1% — so the HRV subscore,
# 40% of the composite, was very nearly binary. Together with a 3-day mean for
# `current` and hysteresis on the band, one bad night no longer relabels the day.
SUBSCORE_K = 1.25
# E8-4: this is an instrument change, not a new athlete state. Trend lines must
# not compare scores from opposite sides of the SUBSCORE_K rescale.
# Deployment-history anchor carried over from the first deployment (the date
# its stored scores were rescaled); a fresh vault has no scores before it.
# Making this per-vault is tracked in issue #6.
SUBSCORE_K_RESCALED_ON = "2026-07-31"
READINESS_SMOOTH_DAYS = 3
GREEN, RED = 67, 34
BAND_HYSTERESIS = 3
READINESS_HYSTERESIS_DAYS = 7
# How old the newest reading may be and still describe "today". The 05:00 brief
# runs before the phone syncs, so yesterday's value is normal; anything older is
# a watch that stopped, and a score built on it is a stale green light.
READINESS_MAX_AGE_DAYS = 1

ACWR_LOAD_METRICS = ["hr_load_proxy", "active_energy", "apple_exercise_time"]
# Metrics that exist only on days a session happened. For these, a day with no
# row is a rest day — a measured zero — not missing data, PROVIDED the watch was
# worn enough to know nothing happened. Everything else keeps the default rule
# that an absent day is unknown (a dropped sync), because a whole-day total like
# active_energy is never legitimately absent on a day the watch was on.
#
# Without this, swapping ACWR onto hr_load_proxy computes the mean load over
# TRAINING days rather than over the week — a different quantity, which cannot
# see the thing ACWR exists to see: training the same sessions across fewer
# days. Measured over the 40 days to 2026-08-09 it also left ACWR computable on
# only 8 of them, against 40/40 with the zero-fill.
ACWR_WORKOUT_SCOPED_METRICS = frozenset({"hr_load_proxy"})
ACWR_WINDOW_DAYS = 28
# Days that must be *present* in the 28-day window. Deliberately below the window
# length: a metric can be absent for a day (a dropped HAE sync, not a rest day)
# and requiring all 28 made ACWR all-or-nothing — one missing day took it dark
# for the four weeks it took that day to age out. Both load figures are means
# scaled to a week, so a gap costs precision, not validity.
ACWR_MIN_CHRONIC_DAYS = 21
ACWR_MIN_ACUTE_DAYS = 5
# Wear gating: ACWR is only trustworthy when both load figures reflect days the
# watch was actually worn. A day counts as "worn" if its sample density
# (daily_metrics.count) is at least WEAR_DENSITY_FRACTION of what a worn day
# looks like for that metric — self-calibrating per metric, and a no-op when
# data is uniformly dense. Require most of the present days to be worn before
# reporting ACWR; otherwise a dense recent week over a sparse/intermittent
# baseline (the backfill→live-sync transition) reads as a false "ramping-fast".
#
# The reference density is measured over a LONG history, not the 28-day window:
# calibrating it from the window meant the window could recalibrate it. When the
# recent week was the sparse one the threshold collapsed with it, every day
# passed, and a week of non-wear was reported as "acwr 0.08, detraining" — the
# exact failure AGENTS.md names (low watch-wear masquerading as a real trend),
# with the gate silently a no-op. A high quantile rather than the median, so a
# metric that is genuinely absent on most days still has a worn-day yardstick.
#
# Density is only a proxy, and a poor one for metrics whose record count tracks
# volume rather than wear (flights_climbed writes a record per flight; live it
# spans 4 to 13,664 samples/day at a constant 24 h of wear). derive.wear_hours
# measures the thing itself — distinct local hours with a heart_rate sample — so
# where it exists it decides, and density is the fallback for days it doesn't
# cover (it only starts 2026-06-09; the backfill era before that still needs the
# proxy). Absence of a wear_hours row is not evidence of non-wear, because the
# derivation may simply never have run for that day.
WEAR_DENSITY_FRACTION = 0.25
WEAR_REF_WINDOW_DAYS = 180
WEAR_REF_QUANTILE = 0.9
WEAR_HOURS_METRIC = "wear_hours"
# WEAR_MIN_HOURS is defined once in metrics.py (imported above) so this module
# and correlate.py cannot numerically diverge on what counts as "worn" — #127.
ACWR_WORN_FRACTION = 0.75

WORKOUT_FOCUS_LOOKBACK_DAYS = 2

MOVER_THRESHOLD_PCT = 15.0
MOVER_TOPK = {"daily": 3, "deep": 8}
# Guards so the "surprise me" channel doesn't surface artifacts: require most of
# the 28-day window present (excludes newly-resumed metrics like a new watch's
# gait stats), skip near-zero baselines, and cap absurd % swings.
MOVER_MIN_WINDOW_DAYS = 21
MOVER_BASE_FLOOR = 1e-6
# A % change says nothing about whether the metric moved further than it moves
# anyway. walking_asymmetry_percentage "UP 85.9%" was the #1 talking point at
# t≈1.4 — noise, read to the user as an injury signal. A mover must clear this
# many standard deviations of its own baseline, and movers are ranked by that
# standardized effect rather than by percentage.
MOVER_MIN_EFFECT_SD = 1.5
MOVER_EFFECT_CAP = 99.0      # a flat baseline gives an infinite effect; keep it JSON-safe
# The ceiling CLAMPS the reported percentage; it used to drop the row, which
# excluded exactly the metrics that moved most. A change beyond 100x is not the
# body changing — it is a unit or scale artifact — and is still dropped.
MOVER_MAX_PCT = 500.0
MOVER_ARTIFACT_PCT = 10000.0

LONGTERM_METRICS = ["step_count", "active_energy", "distance_walking_running",
                    "vo2_max", "body_mass"]


# --- impact volume ----------------------------------------------------------
# Workout duration overstates running impact ~2x because it counts walk breaks.
# The dial that matters for injury risk is time actually spent jogging, so we
# bucket the distance samples and classify each bucket by cadence inside a
# workout window. Pace remains a field used by the separate block dial.
# 20s is short enough to resolve a 90-second interval and long enough that GPS
# jitter on a single sample can't flip a bucket.
# Re-exported so existing callers of analysis.IMPACT_* keep working; the
# definitions live in metrics.py alongside bucket_series.
from .metrics import (  # noqa: F401  — re-export
    IMPACT_BUCKET_SECONDS, IMPACT_JOG_HR_MIN, IMPACT_JOG_HR_PACE_MAX,
    IMPACT_IMPLAUSIBLE_PACE_MIN, IMPACT_JOG_CADENCE_MIN,
    IMPACT_JOG_PACE_MAX, IMPACT_WALK_PACE_MAX,
)


def _as_of(conn, as_of: str | None) -> str:
    if as_of:
        return as_of
    row = conn.execute("SELECT MAX(date) FROM daily_metrics").fetchone()
    return row[0] if row and row[0] else _today(conn).isoformat()


def _today(conn) -> date:
    """Return today's date in the vault's declared zone, or the host date."""
    local_timezone = V.local_timezone(conn)
    return (datetime.now(ZoneInfo(local_timezone)).date()
            if local_timezone else date.today())


def _bucket_index(bucket_start_utc: str) -> int:
    """Bucket ordinal, so 'contiguous' means no missing time rather than merely
    adjacent in a list that may have gaps."""
    ts = datetime.strptime(bucket_start_utc, "%Y-%m-%dT%H:%M:%SZ")
    return int(ts.replace(tzinfo=timezone.utc).timestamp()) // mx.IMPACT_BUCKET_SECONDS


def _is_jog_bucket(b: dict) -> bool:
    """Pace-only, and it DELIBERATELY differs from metrics' canonical is_jog.

    Do not "unify" this with `metrics.impact_bucket_rows`'s is_jog. The volume
    rule is cadence >= IMPACT_JOG_CADENCE_MIN inside any workout window. Block
    structure does not use that classification: it remains pace-only, and a
    16-18 min/mi bucket at high HR is a *bridge* (see _is_bridge_bucket), which
    maintains continuity without being counted as jog.

    Measured 2026-08-26 over the five most recent running sessions: 795 buckets,
    canonical is_jog = 436, this predicate = 368, **68 disagreements — and all
    68 are bridge buckets.** Zero unexplained. The two rules answer different
    questions and agree everywhere the questions coincide.

    scripts/luna_audit.sh reported this divergence as duplicate implementation
    of one computation. It is not, and the measurement above is why. Making them
    identical would move 22.7 jog-minutes across those five sessions into the
    block rule, which governs the ramp.
    """
    p = b.get("pace_min_per_mi")
    return p is not None and mx.IMPACT_IMPLAUSIBLE_PACE_MIN <= p <= mx.IMPACT_JOG_PACE_MAX


def _is_bridge_bucket(b: dict) -> bool:
    p, hr = b.get("pace_min_per_mi"), b.get("hr")
    return (p is not None and mx.IMPACT_JOG_PACE_MAX < p <= mx.BLOCK_BRIDGE_PACE_MAX
            and (hr or 0.0) >= mx.BLOCK_BRIDGE_HR_MIN)


def longest_block_from_buckets(buckets: list[dict], bridge: bool = True,
                               hr_ceiling: float | None = None) -> dict:
    """The block structure of one session, given its bucket series.

    Pure so it can be tested against captured buckets — a 68-minute run is 4,702
    records but 201 buckets, and the rule is defined on buckets.

    Returns:
      unbridged_min          longest chain of jog buckets, bridge rule NOT applied
      bridged_min            longest chain with the bridge rule applied
      qualified_min          longest bridged block whose mean HR is at or below
                             the vault-configured ceiling, else None.
                             THIS is the dial P2-1 made governing.
      avg_hr_longest_block   mean HR over the longest bridged block
      reps                   every block, longest first
    """
    bm = mx.IMPACT_BUCKET_SECONDS / 60.0
    bk = sorted(({**b, "_i": _bucket_index(b["bucket_start_utc"])} for b in buckets),
                key=lambda b: b["_i"])

    segs: list[list[int]] = []          # [start, end] positions into bk
    i, n = 0, len(bk)
    while i < n:
        if _is_jog_bucket(bk[i]):
            j = i
            while (j + 1 < n and bk[j + 1]["_i"] == bk[j]["_i"] + 1
                   and _is_jog_bucket(bk[j + 1])):
                j += 1
            segs.append([i, j])
            i = j + 1
        else:
            i += 1

    def _merge(segments: list[list[int]]) -> list[list[int]]:
        if not segments:
            return []
        out = [segments[0]]
        for s in segments[1:]:
            prev = out[-1]
            gap = list(range(prev[1] + 1, s[0]))
            contiguous = all(bk[k]["_i"] == bk[k - 1]["_i"] + 1
                             for k in range(prev[1] + 1, s[0] + 1))
            if (contiguous and 1 <= len(gap) <= mx.BLOCK_BRIDGE_MAX_BUCKETS
                    and all(_is_bridge_bucket(bk[k]) for k in gap)):
                out[-1] = [prev[0], s[1]]
            else:
                out.append(s)
        return out

    def _describe(seg: list[int], was_bridged: bool) -> dict:
        span = bk[seg[0]:seg[1] + 1]
        hrs = [b["hr"] for b in span if b.get("hr") is not None]
        paces = [b["pace_min_per_mi"] for b in span if b.get("pace_min_per_mi")]
        return {
            "start_utc": span[0]["bucket_start_utc"],
            "end_utc": span[-1]["bucket_start_utc"],
            "minutes": mx.r(len(span) * bm, 1),
            "mean_hr": mx.r(sum(hrs) / len(hrs), 1) if hrs else None,
            "mean_pace_min_per_mi": mx.r(sum(paces) / len(paces), 1) if paces else None,
            "bridged": was_bridged,
        }

    plain_len = max((s[1] - s[0] + 1 for s in segs), default=0)
    merged = _merge(segs) if bridge else list(segs)
    bridged_sets = {(s[0], s[1]) for s in segs}
    reps = sorted((_describe(s, (s[0], s[1]) not in bridged_sets) for s in merged),
                  key=lambda r: r["minutes"], reverse=True)

    best = reps[0] if reps else None
    if hr_ceiling is None:
        hr_ceiling = 155.0  # legacy default for direct pure-function callers
    qualified = [r for r in reps
                 if r["mean_hr"] is not None and r["mean_hr"] <= hr_ceiling]
    return {
        "unbridged_min": mx.r(plain_len * bm, 1),
        "bridged_min": best["minutes"] if best else 0.0,
        # None, not 0.0: "no block qualified" and "a user ran for zero minutes" are
        # different facts and the ramp treats them differently.
        "qualified_min": qualified[0]["minutes"] if qualified else None,
        "avg_hr_longest_block": best["mean_hr"] if best else None,
        "reps": reps,
    }


def longest_block(conn, start_utc: str, end_utc: str, bridge: bool = True) -> dict:
    """Block structure for one session window. See longest_block_from_buckets.

    This is the governing dial for whether the plan progresses: the longest
    continuous block at or below the vault-configured HR ceiling. Until this
    existed it was computed nowhere and derived by hand at
    the review, which is a governor in name only. F2-1 / W7-2.
    """
    return longest_block_from_buckets(
        mx.bucket_series(conn, start_utc, end_utc), bridge=bridge,
        hr_ceiling=mx.block_qualify_hr_max(conn),
    )


NOISE_FLOOR_WINDOW_DAYS = 365
NOISE_FLOOR_MIN_PAIRS = 20


def metric_noise_floor(conn, metric: str, as_of: str,
                       window_days: int = NOISE_FLOOR_WINDOW_DAYS) -> dict:
    """Day-to-day noise of a metric, and the smallest change that is not noise.

    `sd_day` is estimated from CONSECUTIVE-DAY DIFFERENCES rather than from the
    raw spread of the series: SD(diff)/sqrt(2). A raw SD absorbs any real drift
    over the window and so overstates the noise — on resting heart rate over the
    plan window it reads 4.32 against 4.04 from the differences, and the whole
    point of a noise floor is that it is the measurement's, not the trend's.
    This is what audit part 6 used (published SD_day 4.01, rho 0.12).

    `mdc95` is for a comparison of two means of `n_days` each:

        SE   = sd_day / sqrt(n) * sqrt((1 + rho) / (1 - rho))
        MDC  = 1.96 * sqrt(2) * SE

    The autocorrelation inflation matters: consecutive days are not independent
    observations, and ignoring it makes a weekly mean look more precise than it
    is. The per-vault computation gives a WEEKLY mean MDC against the expected
    0–3 bpm training effect reported by Reimers, Knapp & Reimers (2018, J Clin
    Med, DOI 10.3390/jcm7120503). Reporting resting HR weekly can therefore
    miss a small effect, which is why the cadence moved to 4-week blocks where
    the effect sits inside the floor.

    Returns sd_day/rho/mdc95 = None when the window has too few consecutive
    pairs to estimate them. A noise floor guessed from six days is worse than
    admitting there isn't one.
    """
    start = (date.fromisoformat(as_of) - timedelta(days=window_days - 1)).isoformat()
    days, vals, _ = mx.series(conn, metric, start, as_of)
    pairs = [(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)
             if (date.fromisoformat(days[i + 1])
                 - date.fromisoformat(days[i])).days == 1]
    out = {"metric": metric, "n_days": len(vals), "n_consecutive_pairs": len(pairs),
           "window_days": window_days, "sd_day": None, "rho": None}
    if len(pairs) < NOISE_FLOOR_MIN_PAIRS:
        return out
    sd_day = statistics.stdev(pairs) / math.sqrt(2.0)
    rho = 0.0
    if len(vals) > 2:
        mean = statistics.fmean(vals)
        num = sum((vals[i] - mean) * (vals[i + 1] - mean) for i in range(len(vals) - 1))
        den = sum((v - mean) ** 2 for v in vals)
        rho = (num / den) if den else 0.0
    out["sd_day"] = mx.r(sd_day, 2)
    out["rho"] = mx.r(rho, 2)
    return out


def mdc95(sd_day: float, rho: float, n_days: int) -> float:
    """Smallest detectable change between two means of `n_days` each."""
    rho = min(max(rho, 0.0), 0.95)      # a negative or near-1 rho is not usable here
    se = (sd_day / math.sqrt(n_days)) * math.sqrt((1 + rho) / (1 - rho))
    return 1.96 * math.sqrt(2.0) * se


def weekly_series(conn, metric: str, start: str, end: str) -> list[dict]:
    """Monday-anchored weekly means, each with its day count and noise floor.

    The weekly-mean reporting rule mandates weekly means and nothing produced one:
    `summarize_metric` takes periods like '30d', `compare_periods` takes two
    windows and returns a delta with no noise floor attached. So every weekly
    mean in every review was computed ad hoc in-session — which is how the
    64.0 turned out to be the week of 07-13 MINUS its highest day, silently, and
    how week-06-review.md's 66.9 reproduces as neither 67.74 nor 66.43. This is
    a "Python owns the truth" gap and it is the root cause of the
    non-reproducing figures rather than a separate defect. F6-4.

    `n_days` is on every row on purpose: a mean of two days and a mean of seven
    are not the same claim, and the reviews that went wrong did so by not
    saying which they had. `mdc95` is the smallest week-to-week change that is
    not noise — quote a delta smaller than it as movement and you are reading
    the instrument, not the athlete.
    """
    floor = metric_noise_floor(conn, metric, end)
    days, vals, unit = mx.series(conn, metric, start, end)
    weeks: dict[str, list[float]] = {}
    for d, v in zip(days, vals):
        day = date.fromisoformat(d)
        monday = (day - timedelta(days=day.weekday())).isoformat()
        weeks.setdefault(monday, []).append(v)
    out = []
    for monday in sorted(weeks):
        vs = weeks[monday]
        week_end = (date.fromisoformat(monday) + timedelta(days=6)).isoformat()
        out.append({
            "week_start": monday,
            "period": f"{monday}:{week_end}",
            "mean": mx.r(statistics.fmean(vs), 2),
            "n_days": len(vs),
            "unit": unit,
            "sd_day": floor["sd_day"],
            "rho": floor["rho"],
            "mdc95": (mx.r(mdc95(floor["sd_day"], floor["rho"], len(vs)), 2)
                      if floor["sd_day"] is not None else None),
        })
    return out


def session_hr_scopes(conn, start_utc: str, end_utc: str,
                      session_avg: float | None = None) -> dict:
    """All three "average heart rates" of a session, each named for its scope.

    Three quantities have all been called "average HR" here and they run up to
    20 bpm apart on one session — 2026-08-15 is 123.4 whole-session against
    143.6 over jog buckets, and the session grades outside the easy band on one
    and inside on the other. F6-5.

    Lives here rather than in metrics.py because the block figure needs
    longest_block(), and metrics.py cannot import analysis without a cycle.
    """
    out = mx.session_hr_figures(conn, start_utc, end_utc, session_avg)
    block = longest_block(conn, start_utc, end_utc)
    out["avg_hr_longest_block"] = block["avg_hr_longest_block"]
    out["longest_block_min"] = block["bridged_min"]
    out["qualified_block_min"] = block["qualified_min"]
    return out


def impact_volume(conn, start: str, end: str, by: str = "week", *,
                  metric_units: bool = False) -> list[dict]:
    """Minutes actually spent jogging vs walking between two local dates.

    Groups raw distance_walking_running samples into fixed time buckets and
    classifies jogging by start-bucket cadence inside a workout window. Pace
    remains diagnostic and belongs to the separate block dial. Returns one row
    per week (Monday anchored) or per day. Distances are miles, durations
    minutes.
    """
    if by not in ("week", "day"):
        raise ValueError("by must be 'week' or 'day'")
    bucket_min = IMPACT_BUCKET_SECONDS / 60.0
    bucket_rows = mx.impact_bucket_rows(
        conn, "local_date BETWEEN ? AND ?", (start, end)
    )
    grouped: dict[str, dict[str, float | int]] = {}
    for bucket in bucket_rows:
        period = (_week_start(bucket["local_date"]) if by == "week"
                  else bucket["local_date"])
        totals = grouped.setdefault(period, {
            "jog_buckets": 0, "jog_mi": 0.0,
            "walk_buckets": 0, "walk_mi": 0.0,
        })
        if bucket["is_jog"]:
            totals["jog_buckets"] += 1
            totals["jog_mi"] += bucket["mi"]
        if bucket["is_walk"]:
            totals["walk_buckets"] += 1
            totals["walk_mi"] += bucket["mi"]
    rows = [{"period": period, **totals}
            for period, totals in sorted(grouped.items())]

    # Hand-entered minutes for sessions the watch could not measure (F2-4).
    # GymKit produces no per-sample distance, so Jun 27 and Jul 16 — both real
    # running sessions — score 0.0 here. Measured ALWAYS wins; a manual value
    # is a fallback for a period with none, and the two are never blended,
    # which is the unlabelled-blend defect (19c5edc) this must not repeat.
    manual = db.manual_jog_by_day(conn, start, end)
    measured_days = {r["period"] for r in rows} if by == "day" else set()

    out: list[dict] = []
    prev_jog: float | None = None
    distance_key = "jog_km" if metric_units else "jog_miles"
    for r in rows:
        jog_min = round((r["jog_buckets"] or 0) * bucket_min, 1)
        jog_distance = round(
            (r["jog_mi"] or 0.0)
            * (V.UNIT_CONVERSION_FACTORS["distance_mi_to_km"]
               if metric_units else 1.0), 2)
        source, note = "measured", None
        if by == "week":
            covered = [d for d in manual if _week_start(d) == r["period"]
                       and d not in measured_days]
            if covered:
                jog_min = round(jog_min + sum(manual[d]["minutes"] for d in covered), 1)
                source = "partly_manual"
                note = "; ".join(f"{d} {manual[d]['note']}" for d in sorted(covered))
        change = None
        if prev_jog:            # no % off a zero base
            change = round((jog_min - prev_jog) / prev_jog * 100, 1)
        out.append({
            "period_start": r["period"],
            "jog_minutes": jog_min,
            distance_key: jog_distance,
            "jog_pace_min_per_mi": (round((r["jog_buckets"] * bucket_min) / r["jog_mi"], 1)
                                    if r["jog_mi"] else None),
            "walk_minutes": round((r["walk_buckets"] or 0) * bucket_min, 1),
            "walk_miles": round(r["walk_mi"] or 0.0, 2),
            "jog_change_pct": change,
            # Guard 2: every consumer that uses a manual value says so.
            "jog_minutes_source": source,
            "manual_note": note,
        })
        prev_jog = jog_min

    # Periods the measured query produced NO row for, but a manual entry covers.
    if by == "day":
        for day in sorted(set(manual) - measured_days):
            if not (start <= day <= end):
                continue
            out.append({
                "period_start": day,
                "jog_minutes": manual[day]["minutes"],
                distance_key: None, "jog_pace_min_per_mi": None,
                "walk_minutes": 0.0, "walk_miles": 0.0, "jog_change_pct": None,
                "jog_minutes_source": "manual",
                "manual_note": manual[day]["note"],
            })
        out.sort(key=lambda d: d["period_start"])
    else:
        seen = {r["period_start"] for r in out}
        for day in sorted(manual):
            wk = _week_start(day)
            if wk in seen or not (start <= day <= end):
                continue
            out.append({
                "period_start": wk,
                "jog_minutes": manual[day]["minutes"],
                distance_key: None, "jog_pace_min_per_mi": None,
                "walk_minutes": 0.0, "walk_miles": 0.0, "jog_change_pct": None,
                "jog_minutes_source": "partly_manual",
                "manual_note": f"{day} {manual[day]['note']}",
            })
            seen.add(wk)
        out.sort(key=lambda d: d["period_start"])
    return out


def _week_start(day: str) -> str:
    """Monday-anchored week start. This IS impact_volume's week grouping — it
    used to be a Python copy of `date(local_date, 'weekday 0', '-6 days')` kept
    in step by hand, and became the only implementation when the bucket rule
    moved to metrics.impact_bucket_rows."""
    d = date.fromisoformat(day)
    return (d - timedelta(days=d.weekday())).isoformat()


def coverage(conn, as_of: str | None = None) -> list[dict]:
    """How well each vital is covered as of `as_of`. A day counts only if it
    carries a usable value — the same definition of "present" the rest of the
    module uses, so coverage cannot call a metric fresh that readiness is
    treating as stale. ``covers_as_of`` is separate from the density status:
    an intermittent metric may be healthy even when it has no value on this
    particular day. Cadence is measured from usable observations only and is
    used internally to avoid calling that normal gap late."""
    as_of = _as_of(conn, as_of)
    as_of_date = date.fromisoformat(as_of)
    cutoff = (as_of_date - timedelta(days=SPARSE_DAYS - 1)).isoformat()
    cadence_start = (as_of_date -
                     timedelta(days=COVERAGE_CADENCE_DAYS - 1)).isoformat()
    out = []
    for m in VITALS:
        col = mx.value_col(m)
        row = conn.execute(
            f"SELECT MIN(date) f, MAX(date) l, COUNT(*) n FROM daily_metrics "
            f"WHERE metric = ? AND date <= ? AND {col} IS NOT NULL",
            (m, as_of)).fetchone()
        if not row or row["n"] == 0:
            out.append({"metric": m, "status": "missing", "n_days": 0,
                        "first_date": None, "last_date": None,
                        "recent_days": 0, "window_days": SPARSE_DAYS,
                        "recent_fraction": 0.0, "covers_as_of": False,
                        "behind": False})
            continue
        recent_n = conn.execute(
            f"SELECT COUNT(*) FROM daily_metrics WHERE metric = ? AND date >= ? "
            f"AND date <= ? AND {col} IS NOT NULL",
            (m, cutoff, as_of)).fetchone()[0]
        last = row["l"]
        covers_as_of = conn.execute(
            f"SELECT 1 FROM daily_metrics WHERE metric = ? AND date = ? "
            f"AND {col} IS NOT NULL LIMIT 1", (m, as_of)).fetchone() is not None
        typical_gap = _coverage_typical_gap(conn, m, col, cadence_start, as_of)
        gap = (as_of_date - date.fromisoformat(last)).days
        stale = gap > STALE_DAYS
        fraction = recent_n / SPARSE_DAYS
        # Behind means STRICTLY past this metric's own cadence. A daily metric
        # absent on `as_of` alone is not behind -- that is the normal state of a
        # vault whose day has not finished syncing, every morning.
        #
        # It is a separate boolean and NOT a status. Folding it into `sparse`
        # was measured on 2026-08-24: `gap >= typical` labels 23 metric-days in
        # 30 days behind against 8 for `gap > typical`, and on the stub day it
        # handed talking_points (:1284) six vitals at 0.93 density as
        # COVERAGE_THIN -- "arriving, but too little to trend" said of a metric
        # with 13 of 14 days present. A status light wired to the wrong fact.
        behind = (not covers_as_of and typical_gap is not None
                  and gap > typical_gap)
        if stale:
            status = "stale"
        elif fraction < COVERAGE_MIN_FRACTION:
            status = "sparse"
        else:
            status = "active"
        out.append({"metric": m, "status": status, "n_days": row["n"],
                    "first_date": row["f"], "last_date": last,
                    "recent_days": recent_n, "window_days": SPARSE_DAYS,
                    "recent_fraction": round(fraction, 2),
                    "covers_as_of": covers_as_of, "behind": behind})
    return out


def _coverage_typical_gap(conn, metric: str, col: str,
                          start: str, end: str) -> float | None:
    """Median gap between usable observations for one metric.

    This is deliberately an internal decision statistic. It must not become a
    number in the briefing payload: ``last_date`` and ``covers_as_of`` carry
    the externally useful freshness bound without widening the model's pool of
    unscoped numbers.
    """
    rows = conn.execute(
        f"SELECT date FROM daily_metrics WHERE metric = ? AND date BETWEEN ? AND ? "
        f"AND {col} IS NOT NULL ORDER BY date", (metric, start, end)).fetchall()
    dates = [date.fromisoformat(r[0]) for r in rows]
    gaps = [(later - earlier).days for earlier, later in zip(dates, dates[1:])]
    return statistics.median(gaps) if gaps else None


def _recent_and_baseline(conn, metric, as_of, window=READINESS_BASELINE_WINDOW):
    """Return (current_value, baseline_value, n_baseline_days, latest_date).

    `latest_date` is the day `current` actually came from — it is NOT `as_of`.
    Callers must compare the two: a metric that stopped arriving keeps returning
    its final value forever, and a readiness score built on it reads as today's.

    `current` is the mean of the last READINESS_SMOOTH_DAYS days rather than a
    single reading: HRV's measured daily CV is 17.1%, so one night decided the
    band. The baseline excludes exactly the days that made `current`.
    """
    start = (date.fromisoformat(as_of) - timedelta(days=window)).isoformat()
    dates, vals, _ = mx.series(conn, metric, start, as_of)
    if not vals:
        return None, None, 0, None
    # Smooth over calendar days, not over rows: with a gap, the last three rows
    # can span a week and "recent" would quietly mean something else.
    cutoff = (date.fromisoformat(dates[-1])
              - timedelta(days=READINESS_SMOOTH_DAYS - 1)).isoformat()
    recent = [v for d, v in zip(dates, vals) if d >= cutoff][-READINESS_SMOOTH_DAYS:]
    current = sum(recent) / len(recent)
    base = mx.baseline(vals, exclude_recent=len(recent), window=window)
    return current, base, len(vals) - len(recent), dates[-1]


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def _age_days(as_of: str, day: str | None) -> int | None:
    if not day:
        return None
    return (date.fromisoformat(as_of) - date.fromisoformat(day)).days


_READINESS_COMPONENT_METRICS = {
    "hrv": "heart_rate_variability",
    "rhr": "resting_heart_rate",
    "sleep": "sleep_asleep",
}


def _sticky_band(score: float, prev: str | None) -> str:
    """Band for `score`, given yesterday's band. Crossing a boundary takes
    BAND_HYSTERESIS points more than staying put, so a label survives the noise
    that is left after smoothing instead of being renegotiated every morning."""
    h = 0 if prev is None else BAND_HYSTERESIS
    green_edge = GREEN - h if prev == "green" else GREEN + h
    red_edge = RED + h if prev == "red" else RED - h
    if score >= green_edge:
        return "green"
    if score < red_edge:
        return "red"
    return "amber"


def _readiness_subscores(conn, as_of: str):
    """(subs, factors, baselined, ages) for one day — the raw material of a
    readiness score, with no banding applied."""
    subs, factors = {}, []
    ages: dict[str, int] = {}          # component -> age of its source day
    baselined: set[str] = set()        # components with enough history to score

    def _record(component, cur, day, subscore, extra):
        age = _age_days(as_of, day)
        f = {"component": component, "current": mx.r(cur), "date": day,
             "age_days": age, **extra}
        metric = _READINESS_COMPONENT_METRICS[component]
        f["field_metrics"] = {"current": metric}
        if "baseline" in extra:
            f["field_metrics"]["baseline"] = metric
        stale = age is None or age > READINESS_MAX_AGE_DAYS
        f["stale"] = stale
        factors.append(f)
        if age is not None:
            ages[component] = age
        if not stale:
            subs[component] = subscore

    cur, base, n, day = _recent_and_baseline(conn, "heart_rate_variability", as_of)
    if base is not None and n >= READINESS_MIN_BASELINE_DAYS:
        baselined.add("hrv")
        dev = mx.pct_change(cur, base) or 0.0
        _record("hrv", cur, day, _clamp(50 + SUBSCORE_K * dev),
                {"baseline": mx.r(base), "pct": dev})

    cur, base, n, day = _recent_and_baseline(conn, "resting_heart_rate", as_of)
    if base is not None and n >= READINESS_MIN_BASELINE_DAYS:
        baselined.add("rhr")
        dev = mx.pct_change(cur, base) or 0.0
        _record("rhr", cur, day, _clamp(50 - SUBSCORE_K * dev),
                {"baseline": mx.r(base), "pct": dev})

    cur, _, _, day = _recent_and_baseline(conn, "sleep_asleep", as_of)
    if cur is not None:
        _record("sleep", cur, day, _clamp(100 * cur / SLEEP_TARGET_MIN),
                {"target": SLEEP_TARGET_MIN,
                 "pct": mx.pct_change(cur, SLEEP_TARGET_MIN)})

    factors.sort(key=lambda f: abs(f.get("pct") or 0), reverse=True)
    return subs, factors, baselined, ages


def _composite(subs: dict) -> int:
    w = {k: READINESS_WEIGHTS[k] for k in subs}
    return round(sum(subs[k] * w[k] for k in subs) / sum(w.values()))


def readiness_rescale_refusal(*, today: str, week_ago: str) -> str | None:
    """Refuse a readiness comparison that crosses the score rescale.

    The old and new subscores are different instruments. Returns the refusal
    text when the two dates straddle SUBSCORE_K_RESCALED_ON, else None — a
    same-era comparison needs no warning, and callers still decide whether both
    scores exist. Lives here rather than in coach_brief because the rescale is a
    property of the score, not of one surface that renders it.
    """
    boundary = date.fromisoformat(SUBSCORE_K_RESCALED_ON)
    today_d = date.fromisoformat(today)
    week_ago_d = date.fromisoformat(week_ago)
    crosses = (today_d < boundary <= week_ago_d
               or week_ago_d < boundary <= today_d)
    if crosses:
        return ("Readiness trend unavailable: instrument changed on "
                f"{SUBSCORE_K_RESCALED_ON}.")
    return None


def _prior_band(conn, as_of: str) -> str | None:
    """Yesterday's band, itself smoothed over the preceding days so the label
    does not depend on where the walk happens to start."""
    prev = None
    for i in range(READINESS_HYSTERESIS_DAYS, 0, -1):
        day = (date.fromisoformat(as_of) - timedelta(days=i)).isoformat()
        subs, _, _, _ = _readiness_subscores(conn, day)
        if {"hrv", "rhr"} & set(subs):
            prev = _sticky_band(_composite(subs), prev)
    return prev


def readiness(conn, as_of: str | None = None) -> dict:
    """Recovery readiness for `as_of`, or an explicit refusal to score.

    `status` is one of:
      ok                     both autonomic inputs fresh and scored
      partial                one autonomic input scored (the other missing/stale)
      stale                  the inputs exist but predate `as_of` -- score is None
                             and `stale_days` says by how much
      establishing_baseline  not enough history to have a baseline yet
    Every response carries `as_of`, `latest_date` (the newest source day used or
    available), `stale_days` (their difference) and dated `factors`, so a
    consumer can always render "N days old" instead of implying "today".
    """
    as_of = _as_of(conn, as_of)
    subs, factors, baselined, ages = _readiness_subscores(conn, as_of)
    fresh_age = min((ages[k] for k in subs if k in ages), default=None)
    any_age = min(ages.values(), default=None)

    # Recovery readiness is fundamentally an autonomic signal: without HRV or
    # resting-HR baseline, a lone sleep value must NOT produce a confident score.
    autonomic = {"hrv", "rhr"} & set(subs)
    if not autonomic:
        # Distinguish "the watch stopped" from "the watch is new". Both refuse to
        # score; only one of them is something the user can fix today.
        if baselined:
            stale_days = min(ages[k] for k in baselined if k in ages)
            latest = (date.fromisoformat(as_of) - timedelta(days=stale_days)).isoformat()
            return {"status": "stale", "score": None, "band": None,
                    "as_of": as_of, "latest_date": latest,
                    "stale_days": stale_days,
                    "max_age_days": READINESS_MAX_AGE_DAYS,
                    "note": f"recovery inputs are {stale_days} days old "
                            f"(latest {latest}); not scored",
                    "factors": factors}
        return {"status": "establishing_baseline", "score": None, "band": None,
                "as_of": as_of, "latest_date": None, "stale_days": any_age,
                "note": "recovery readiness needs HRV / resting-HR history; "
                        "still building a baseline", "factors": factors}

    score = _composite(subs)
    prior = _prior_band(conn, as_of)
    if len(autonomic) < 2:
        # E8-4: renormalising a partial score can promote an unchanged person.
        # Keep the last complete band's label, and say which input was absent so
        # a held label cannot be mistaken for a complete, fresh assessment.
        absent = sorted({"hrv", "rhr"} - autonomic)
        band = prior if prior is not None else _sticky_band(score, None)
        note = (f"{', '.join(absent)} component absent; "
                + (f"holding prior {band} band" if prior is not None
                   else "no prior band available; partial score used"))
    else:
        band = _sticky_band(score, prior)
        note = None
    status = "ok" if len(autonomic) == 2 else "partial"
    latest = ((date.fromisoformat(as_of) - timedelta(days=fresh_age)).isoformat()
              if fresh_age is not None else None)
    components = {k: round(v) for k, v in subs.items()}
    if "sleep" in components:
        components["field_metrics"] = {"sleep": "readiness"}
    out = {"status": status, "score": score, "band": band,
           "as_of": as_of, "latest_date": latest, "stale_days": fresh_age,
           "components": components, "factors": factors,
           "field_metrics": {"score": "readiness"}}
    if note:
        out["note"] = note
    return out


# --- readiness becomes weekly, plus a two-day alert (P8-4, W7-4) -----------
#
# The daily 0-100 composite was retired at the Week 7 review. 53 scored days
# produced 40 amber, 13 green and red NEVER — red would have needed a 3-day mean
# resting HR of 77 against a 60 baseline — and both reachable bands license the
# same session. A number that cannot reach a third of its range, and whose two
# reachable values imply the same action, is not informing a decision.
#
# What replaces it: a WEEKLY read, where seven days of averaging make the number
# mean something; and an alert that stays silent unless a component crosses a
# detectable threshold on two consecutive days. One bad morning is noise. Two in
# a row is the smallest pattern worth interrupting a user for — and anything looser
# is the daily composite again under another name.
READINESS_ALERT_DAYS = 2
# Thresholds are deviations from the user's own baseline, not absolutes, and they
# are set where the deviation exceeds what the instrument's day-to-day noise can
# produce. See metric_noise_floor(): a change smaller than the MDC is not a
# change, it is the measurement.
READINESS_ALERT_RHR_PCT = 10.0     # resting HR this far ABOVE baseline
READINESS_ALERT_HRV_PCT = -20.0    # HRV this far BELOW baseline
_ALERT_RULES = {
    "rhr": ("resting heart rate", READINESS_ALERT_RHR_PCT, 1),
    "hrv": ("heart-rate variability", READINESS_ALERT_HRV_PCT, -1),
}


def readiness_alert(conn, as_of: str | None = None) -> dict | None:
    """The one thing worth interrupting a day for, or None.

    None is the expected answer and is not a failure: an alert that fires most
    days is the amber-every-morning composite this replaced.
    """
    as_of = _as_of(conn, as_of)
    days = [(date.fromisoformat(as_of) - timedelta(days=i)).isoformat()
            for i in range(READINESS_ALERT_DAYS)]
    for component, (label, threshold, sign) in _ALERT_RULES.items():
        crossings = []
        for day in days:
            factors = _readiness_subscores(conn, day)[1]
            f = next((x for x in factors
                      if x["component"] == component and not x["stale"]), None)
            if f is None or f.get("pct") is None:
                break
            if (f["pct"] - threshold) * sign < 0:
                break
            crossings.append(f)
        if len(crossings) == READINESS_ALERT_DAYS:
            return {"component": label, "metric": component,
                    "days": READINESS_ALERT_DAYS,
                    "current": crossings[0]["current"],
                    "baseline": crossings[0].get("baseline"),
                    "pct": mx.r(crossings[0]["pct"]),
                    "as_of": as_of}
    return None


def weekly_readiness(conn, as_of: str | None = None) -> dict:
    """Readiness as a week, which is the scale it can carry.

    Averages each component's daily subscore over the seven days ending at
    `as_of` and reports the mean composite with the day count behind it. No
    band and no cue: the Week 7 review dropped the daily label because both
    reachable bands licensed the same session, and re-attaching one to a weekly
    mean would put it straight back.
    """
    as_of = _as_of(conn, as_of)
    start = (date.fromisoformat(as_of) - timedelta(days=6)).isoformat()
    scores, per_component = [], {}
    for i in range(7):
        day = (date.fromisoformat(as_of) - timedelta(days=i)).isoformat()
        subs = _readiness_subscores(conn, day)[0]
        if not subs:
            continue
        scores.append(_composite(subs))
        for k, v in subs.items():
            per_component.setdefault(k, []).append(v)
    return {
        "week_start": start, "week_end": as_of, "n_days": len(scores),
        "score": mx.r(sum(scores) / len(scores), 1) if scores else None,
        "components": {k: mx.r(sum(v) / len(v), 1)
                       for k, v in sorted(per_component.items())},
    }


def weekly_readiness_trend(conn, as_of: str | None = None) -> str | None:
    """This week's readiness against last week's, or a refusal.

    Week over week, not day over day: P8-4 retired the daily composite, and a
    daily comparison of daily composites is that number again under another
    name. Returns None when either week is unscoreable — silence, never a
    direction inferred from one usable day.
    """
    as_of = _as_of(conn, as_of)
    prev = (date.fromisoformat(as_of) - timedelta(days=7)).isoformat()
    refusal = readiness_rescale_refusal(today=as_of, week_ago=prev)
    if refusal:
        return refusal
    now, before = weekly_readiness(conn, as_of), weekly_readiness(conn, prev)
    if now["score"] is None or before["score"] is None:
        return None
    if min(now["n_days"], before["n_days"]) < 4:
        return None          # a "week" of three days is not a week
    delta = now["score"] - before["score"]
    return (f"Readiness this week {now['score']:.0f}/100 "
            f"({now['n_days']} days) vs {before['score']:.0f} last week "
            f"({delta:+.0f}).")


# --- the floor is an action, so continuity is measured as movement (P3-2) --
#
# "No day under 3,000 steps" was retired: an OUTCOME cannot be achieved on a
# bad day by deciding to, and an ACTION can — which is the whole mechanism the
# user reports ("it helps, but not by making me walk — knowing it's there
# takes pressure off a bad day"). The floor is now "a 15-minute walk, or 15
# minutes of deliberate movement at home."
#
# Only the walk half leaves a trace. This reports what it can see and says so;
# it must never render a day with no evidence as a miss, because that grades
# the user on the instrument's blind spot rather than on what they did.
FLOOR_WALK_MINUTES = 15.0


def movement_floor_days(conn, start: str, end: str) -> list[dict]:
    """Per-day movement evidence over [start, end], inclusive."""
    walk = {r["period_start"]: r["walk_minutes"]
            for r in impact_volume(conn, start, end, by="day")}
    sessions = {r[0] for r in conn.execute(
        "SELECT DISTINCT local_date FROM workouts WHERE local_date BETWEEN ? AND ?",
        (start, end))}
    out, day = [], date.fromisoformat(start)
    last = date.fromisoformat(end)
    while day <= last:
        iso = day.isoformat()
        wm = walk.get(iso, 0.0)
        out.append({"date": iso, "walk_minutes": wm,
                    "session": iso in sessions,
                    "active": bool(iso in sessions or wm >= FLOOR_WALK_MINUTES)})
        day += timedelta(days=1)
    return out


EARLY_WARNING_DAYS = 21
EARLY_WARNING_RHR_PER_WEEK_CONVENTION = 1.0   # bpm/week rising
EARLY_WARNING_HRV_PER_WEEK_CONVENTION = -1.0  # ms/week falling
EARLY_WARNING_THRESHOLD_BASIS = (
    "operational convention; not a literature-derived fit"
)


def _slope_over(conn, metric, as_of, days):
    start = (date.fromisoformat(as_of) - timedelta(days=days - 1)).isoformat()
    dates, vals, _ = mx.series(conn, metric, start, as_of)
    return mx.slope_per_week(dates, vals), len(vals)


def trends(conn, as_of: str | None = None) -> dict:
    """Three-week slopes and early-warning flags.

    The per-week cutoffs are operational conventions, not fitted or
    literature-derived thresholds; the basis is returned beside the flags.
    """
    as_of = _as_of(conn, as_of)
    rhr_slope, rhr_n = _slope_over(conn, "resting_heart_rate", as_of, EARLY_WARNING_DAYS)
    hrv_slope, hrv_n = _slope_over(conn, "heart_rate_variability", as_of, EARLY_WARNING_DAYS)

    rhr_rising = (rhr_slope is not None
                  and rhr_slope >= EARLY_WARNING_RHR_PER_WEEK_CONVENTION)
    hrv_falling = (hrv_slope is not None
                   and hrv_slope <= EARLY_WARNING_HRV_PER_WEEK_CONVENTION)
    flag = bool(rhr_rising or hrv_falling) and rhr_n >= 10 and hrv_n >= 10

    return {
        "rhr_per_week": mx.r(rhr_slope),
        "hrv_per_week": mx.r(hrv_slope),
        "early_warning": {
            "flag": flag,
            "rhr_rising": rhr_rising,
            "hrv_falling": hrv_falling,
            "window_days": EARLY_WARNING_DAYS,
            "rhr_threshold_per_week": EARLY_WARNING_RHR_PER_WEEK_CONVENTION,
            "hrv_threshold_per_week": EARLY_WARNING_HRV_PER_WEEK_CONVENTION,
            "threshold_basis": EARLY_WARNING_THRESHOLD_BASIS,
        },
    }


def _daily_load_rows(conn, metric, as_of, days):
    """(date, value, sample_count) per present day over the window, oldest first.
    Days the metric is absent for simply aren't returned, so callers must select
    the acute sub-window by date rather than by row position. The count is the
    wear-density signal used to gate ACWR.

    A day whose value column is NULL is absent, not zero — mx.series has always
    filtered those and this must agree, or the two disagree about what a day is.
    Live, the newest two daily_metrics rows carry a NULL `last` for ~58 metrics,
    which fed None into movers' arithmetic and crashed the whole briefing."""
    start = (date.fromisoformat(as_of) - timedelta(days=days - 1)).isoformat()
    col = mx.value_col(metric)
    rows = conn.execute(
        f"SELECT date d, {col} v, count c FROM daily_metrics WHERE metric = ? "
        f"AND date BETWEEN ? AND ? AND {col} IS NOT NULL ORDER BY date",
        (metric, start, as_of)).fetchall()
    return [(r["d"], r["v"], r["c"]) for r in rows]


def _acwr_load_rows(conn, metric, as_of, days):
    """_daily_load_rows, plus zero rows for worn rest days on a workout-scoped
    metric. ACWR-only on purpose: `movers` shares _daily_load_rows and asks a
    different question of it (how much did this metric move), where inventing
    days would change what it reports.

    The wear check is what keeps hr_load.py's "missing is unknown, not zero"
    doctrine intact: a day the watch was off stays absent, so a non-wear stretch
    can never be read as a training collapse. A day the watch was ON and no
    session happened is a rest day, and its training load is zero by observation.

    Synthesized rows carry a count of 0, which never reaches the density
    fallback in _worn_rows: they exist only because wear_hours covers them, so
    that function takes the measured-hours branch for every one of them.
    """
    rows = _daily_load_rows(conn, metric, as_of, days)
    if metric not in ACWR_WORKOUT_SCOPED_METRICS:
        return rows
    start = (date.fromisoformat(as_of) - timedelta(days=days - 1)).isoformat()
    hours = _wear_hours_map(conn, start, as_of)
    have = {d for d, _, _ in rows}
    d, last = date.fromisoformat(start), date.fromisoformat(as_of)
    while d <= last:
        day = d.isoformat()
        if day not in have and hours.get(day, 0.0) >= WEAR_MIN_HOURS:
            rows.append((day, 0.0, 0))
        d += timedelta(days=1)
    return sorted(rows)


def _wear_reference(conn, metric, as_of, window=WEAR_REF_WINDOW_DAYS) -> float:
    """The sample density of a worn day for this metric, from a long history.

    Deliberately independent of the window being judged: a threshold derived
    from the days under test cannot detect that those days are the unworn ones.
    """
    start = (date.fromisoformat(as_of) - timedelta(days=window - 1)).isoformat()
    counts = [r[0] for r in conn.execute(
        "SELECT count FROM daily_metrics WHERE metric = ? AND date BETWEEN ? AND ? "
        "AND count IS NOT NULL ORDER BY count", (metric, start, as_of)).fetchall()]
    if not counts:
        return 0.0
    i = int(WEAR_REF_QUANTILE * (len(counts) - 1))
    return float(counts[i])


def _wear_hours_map(conn, start: str, end: str) -> dict[str, float]:
    """date -> hours the watch was worn, for the days that carry the signal."""
    col = mx.value_col(WEAR_HOURS_METRIC)
    return {r["d"]: r["v"] for r in conn.execute(
        f"SELECT date d, {col} v FROM daily_metrics WHERE metric = ? "
        f"AND date BETWEEN ? AND ? AND {col} IS NOT NULL",
        (WEAR_HOURS_METRIC, start, end)).fetchall()}


def _consolidated_days(conn, metric: str, start: str, end: str) -> set[str]:
    """The dates in [start, end] whose `sum` is Apple's consolidated total.

    Empty on any database that predates D19's `source_kind` column — including
    the read-only snapshot, which is never migrated — so every caller below
    reduces exactly to its pre-D19 behaviour there.
    """
    try:
        return {r[0] for r in conn.execute(
            "SELECT date FROM daily_metrics WHERE metric = ? "
            "AND date BETWEEN ? AND ? AND source_kind = 'apple_consolidated'",
            (metric, start, end)).fetchall()}
    except sqlite3.OperationalError:
        return set()


def _wear_eligible(conn, metric, as_of, rows: list[tuple]) -> list[tuple]:
    """`rows`, minus the days that carry no wear evidence of EITHER kind (D19).

    Call this before sizing a denominator from `len(rows)`. A consolidated day
    with no `wear_hours` row leaves the window entirely rather than merely
    failing the worn test: dropped-but-still-counted is a day voting "not worn"
    without evidence, and both `training_load` (min_worn from len(rows)) and
    `movers` size a minimum against a count this would otherwise inflate.

    Why the density proxy cannot stand in for it: the proxy exists because a
    sparse day's `sum` *understates* the day, and on a consolidated row that
    premise is false outright — Apple's total is complete however few raw
    samples we happened to receive, and `count` on such a row is the honest raw
    count, 0 included. But a consolidated total carries no wear information
    either: it can be phone-only steps on a day the watch sat on a charger,
    which is the failure AGENTS.md names. So neither answer is available and the
    day is not asked.

    Expected to be unreachable in practice — `derive.wear_hours` starts
    2026-06-09 and D19 is forward-only from a cutover well after that, so every
    day D19 can produce is inside wear_hours' coverage. Specified anyway,
    because "expected to be unreachable" is precisely how silent behaviour gets
    written.
    """
    if not rows:
        return []
    consolidated = _consolidated_days(conn, metric, rows[0][0], as_of)
    if not consolidated:
        return list(rows)
    hours = _wear_hours_map(conn, rows[0][0], as_of)
    return [row for row in rows
            if row[0] not in consolidated or row[0] in hours]


def _worn_rows(conn, metric, as_of, rows: list[tuple]) -> list[tuple]:
    """The subset of [(date, value, sample_count)] rows that describe a day the
    watch was actually worn. Measured hours where they exist, sample density
    otherwise. A no-op when the window is uniformly worn, so it only bites at a
    wear discontinuity — in either direction.

    On a consolidated day (D19) `wear_hours` decides where it exists, exactly as
    for any other day — it is the measurement rather than the proxy — and where
    it does not, `_wear_eligible` has already removed the day, because `count`
    on such a row is not a wear-density signal. This is NOT
    "consolidated implies worn".
    """
    if not rows:
        return []
    rows = _wear_eligible(conn, metric, as_of, rows)
    if not rows:
        return []
    hours = _wear_hours_map(conn, rows[0][0], as_of)
    floor = WEAR_DENSITY_FRACTION * _wear_reference(conn, metric, as_of)
    return [(d, v, c) for d, v, c in rows
            if (hours[d] >= WEAR_MIN_HOURS if d in hours else (c or 0) >= floor)]


def _worn_values(conn, metric, as_of, rows: list[tuple]) -> list[float]:
    return [v for _, v, _ in _worn_rows(conn, metric, as_of, rows)]


def _band_acwr(ratio: float) -> str:
    if ratio < 0.8:
        return "detraining"
    if ratio <= 1.3:
        return "sweet-spot"
    if ratio <= 1.5:
        return "caution"
    return "ramping-fast"


def training_load(conn, as_of: str | None = None) -> dict:
    as_of = _as_of(conn, as_of)
    load_metric = next((m for m in ACWR_LOAD_METRICS if mx.metric_exists(conn, m)), None)
    if not load_metric:
        return {"status": "no_load_metric", "acwr": None}

    # _wear_eligible before any len(rows) is taken: a day with no wear evidence
    # of either kind must leave the DENOMINATOR as well as the worn set, or it
    # votes "not worn" without evidence against min_worn below (D19 §Q2). A
    # provable no-op today — ACWR_LOAD_METRICS is hr_load_proxy/active_energy/
    # apple_exercise_time and D19 writes none of them — and here so that it
    # stays correct if that ever stops being true.
    rows = _wear_eligible(conn, load_metric, as_of,
                          _acwr_load_rows(conn, load_metric, as_of,
                                          ACWR_WINDOW_DAYS))
    if len(rows) < ACWR_MIN_CHRONIC_DAYS:
        return {"status": "insufficient_history", "acwr": None,
                "load_metric": load_metric, "n_days": len(rows),
                "min_required": ACWR_MIN_CHRONIC_DAYS,
                "window_days": ACWR_WINDOW_DAYS}

    # Wear gate: keep only days sampled densely enough to describe a worn day.
    # BOTH load figures are means over worn days (scaled to a week): filtering
    # only the chronic side made a flat 500 kcal/day with three non-wear days
    # report "acwr 0.59, detraining" — a comparison between a measured week and
    # an unmeasured one. With a fully-dense window this reduces exactly to the
    # classic sum(28d)/4 weekly average.
    worn_rows = _worn_rows(conn, load_metric, as_of, rows)
    worn = [v for _, v, _ in worn_rows]
    min_worn = math.ceil(ACWR_WORN_FRACTION * len(rows))
    if len(worn) < min_worn:
        return {"status": "insufficient_wear", "acwr": None,
                "load_metric": load_metric, "n_worn_days": len(worn),
                "n_days": len(rows), "min_required": min_worn}

    # Acute window is bound by calendar date, not by row position: with a gap in
    # the last week, rows[-7:] would reach back into an 8th day and blend an
    # older load into "this week". Both sides are day-means scaled to a week so
    # they stay comparable when either window is missing a day.
    acute_start = (date.fromisoformat(as_of) - timedelta(days=6)).isoformat()
    recent = [v for d, v, _ in worn_rows if d >= acute_start]
    if len(recent) < ACWR_MIN_ACUTE_DAYS:
        return {"status": "insufficient_recent", "acwr": None,
                "load_metric": load_metric, "n_recent_days": len(recent),
                "n_worn_days": len(worn), "n_days": len(rows),
                "min_required": ACWR_MIN_ACUTE_DAYS}

    acute = (sum(recent) / len(recent)) * 7
    chronic = (sum(worn) / len(worn)) * 7
    ratio = acute / chronic if chronic else None
    if ratio is None:
        return {"status": "no_load", "acwr": None, "load_metric": load_metric}

    # workout mix (best-effort; empty list if no workouts table rows)
    wstart = (date.fromisoformat(as_of) - timedelta(days=27)).isoformat()
    wrows = conn.execute(
        "SELECT workout_type, COUNT(*) n FROM workouts WHERE local_date BETWEEN ? AND ? "
        "GROUP BY workout_type ORDER BY n DESC", (wstart, as_of)).fetchall()
    mix = [{"type": w["workout_type"], "count": w["n"]} for w in wrows]

    return {
        "status": "ok",
        "load_metric": load_metric,
        "acwr": mx.r(ratio),
        "acwr_band": _band_acwr(ratio),
        "acute_7d": mx.r(acute),
        "chronic_weekly_avg": mx.r(chronic),
        "n_worn_days": len(worn),
        "n_days": len(rows),
        "n_recent_days": len(recent),
        "workout_mix_28d": mix,
    }


def workout_focus(conn, as_of: str | None = None, *,
                  metric_units: bool = False) -> dict | None:
    """Most recent workout within the lookback window.
    Running-style workouts get pace; cycling gets speed so a bike ride can never
    be narrated as a running pace.
    On a same-day tie the longest workout (by duration) is chosen."""
    as_of = _as_of(conn, as_of)
    start = (date.fromisoformat(as_of)
             - timedelta(days=WORKOUT_FOCUS_LOOKBACK_DAYS)).isoformat()
    row = conn.execute(
        "SELECT workout_type, local_date, duration_min, distance_mi, energy_kcal "
        "FROM workouts WHERE local_date BETWEEN ? AND ? "
        "ORDER BY local_date DESC, duration_min DESC LIMIT 1",
        (start, as_of)).fetchone()
    if not row:
        return None
    dist = row["distance_mi"]
    dur = row["duration_min"]
    cycling = _is_cycling_type(row["workout_type"])
    # duration / distance on a run/walk session is a BLENDED pace — it counts
    # the walk breaks — and this dict reaches the agent verbatim through
    # get_briefing. On 2026-08-09 the workout row gave 17.0 min/mi while the
    # day's jog buckets gave 14.2: handing over a bare "17.0 min/mi" understates
    # the single number the plan is written about. Label it, and carry the day's
    # real jog pace alongside where the samples support one. coach_brief.
    # day_actuals already does exactly this; the briefing path did not.
    day = row["local_date"]
    jog_pace = None
    if not cycling:
        try:
            rows = impact_volume(conn, day, day, by="day", metric_units=metric_units)
            jog_pace = rows[0].get("jog_pace_min_per_mi") if rows else None
        except Exception:  # noqa: BLE001 — a briefing must not die for a label
            jog_pace = None
    pace_key = "pace_min_per_km" if metric_units else "pace_min_per_mi"
    speed_key = "speed_kph" if metric_units else "speed_mph"
    pace = (None if cycling else dur / dist if dist and dur else None)
    speed = (dist / (dur / 60) if cycling and dist and dur else None)
    if metric_units:
        factor = V.UNIT_CONVERSION_FACTORS["distance_mi_to_km"]
        pace = mx.r(pace / factor, 1) if pace is not None else None
        speed = mx.r(speed * factor, 1) if speed is not None else None
    else:
        pace = mx.r(pace, 1) if pace is not None else None
        speed = mx.r(speed, 1) if speed is not None else None
    out = {
        "type": row["workout_type"],
        "date": day,
        "distance_mi": mx.r(dist, 1) if dist is not None else None,
        "duration_min": mx.r(dur, 1) if dur is not None else None,
        "energy_kcal": mx.r(row["energy_kcal"]) if row["energy_kcal"] is not None else None,
        pace_key: pace,
        "pace_label": None if cycling else "blended",
        "jog_pace_min_per_mi": jog_pace,
        speed_key: speed,
    }
    return out


def _is_cycling_type(workout_type: str | None) -> bool:
    value = (workout_type or "").lower()
    return value in {"cycling", "stationary_bike", "bike", "spinning"} \
        or "cycling" in value or "bike" in value


def _all_metrics(conn):
    rows = conn.execute("SELECT DISTINCT metric FROM daily_metrics").fetchall()
    return [r["metric"] for r in rows]


def movers(conn, as_of: str | None = None, scope: str = "daily") -> list[dict]:
    as_of = _as_of(conn, as_of)
    start = (date.fromisoformat(as_of) - timedelta(days=27)).isoformat()
    found = []
    for m in _all_metrics(conn):
        dates, present_vals, unit = mx.series(conn, m, start, as_of)
        if len(present_vals) < MOVER_MIN_WINDOW_DAYS:
            continue
        # Compare recent vs baseline over WORN days only, so a sparse/intermittent
        # baseline (e.g. backfill before live sync began) can't fabricate a swing.
        # Require most of the window to be worn, or there's no trustworthy baseline.
        vals = _worn_values(conn, m, as_of, _daily_load_rows(conn, m, as_of, 28))
        if len(vals) < MOVER_MIN_WINDOW_DAYS:
            continue
        rn = max(1, min(7, len(vals) // 3))
        recent_vals, base_vals = vals[-rn:], vals[:-rn]
        recent = sum(recent_vals) / len(recent_vals)
        base = sum(base_vals) / max(1, len(base_vals))
        if abs(base) < MOVER_BASE_FLOOR:
            continue
        pct = mx.pct_change(recent, base)
        if pct is None or abs(pct) < MOVER_THRESHOLD_PCT:
            continue
        if abs(pct) > MOVER_ARTIFACT_PCT:
            continue        # a 100x jump is a unit/scale change, not a finding
        # Effect size: how far the recent mean sits from the baseline in units of
        # the baseline's own day-to-day spread. A perfectly flat baseline makes
        # any move unambiguous, so it takes the cap rather than dividing by zero.
        sd = statistics.pstdev(base_vals) if len(base_vals) > 1 else 0.0
        diff = recent - base
        if sd > 0:
            effect = max(-MOVER_EFFECT_CAP, min(MOVER_EFFECT_CAP, diff / sd))
        else:
            effect = math.copysign(MOVER_EFFECT_CAP, diff) if diff else 0.0
        if abs(effect) < MOVER_MIN_EFFECT_SD:
            continue
        capped = abs(pct) > MOVER_MAX_PCT
        found.append({"metric": m, "unit": unit,
                      "direction": "up" if pct > 0 else "down",
                      "pct": math.copysign(MOVER_MAX_PCT, pct) if capped else pct,
                      "pct_uncapped": pct, "pct_capped": capped,
                      "effect_sd": mx.r(effect), "baseline_sd": mx.r(sd),
                      "recent_avg": mx.r(recent),
                      "baseline_avg": mx.r(base), "n_days": len(vals)})
    found.sort(key=lambda x: (abs(x["effect_sd"]), abs(x["pct"])), reverse=True)
    return found[:MOVER_TOPK.get(scope, 3)]


def long_term(conn, as_of: str | None = None) -> list[dict]:
    as_of = _as_of(conn, as_of)
    out = []
    for m in LONGTERM_METRICS:
        if not mx.metric_exists(conn, m):
            continue
        now_s, now_e = mx.parse_period("30d", as_of)
        _, now_vals, unit = mx.series(conn, m, now_s, now_e)
        if not now_vals:
            continue
        now_mean = sum(now_vals) / len(now_vals)
        entry = {"metric": m, "unit": unit, "this_month_avg": mx.r(now_mean)}
        for label, days in (("vs_3mo", 90), ("vs_6mo", 180), ("vs_12mo", 365)):
            past_e = (date.fromisoformat(as_of) - timedelta(days=days)).isoformat()
            past_s = (date.fromisoformat(past_e) - timedelta(days=29)).isoformat()
            _, pv, _ = mx.series(conn, m, past_s, past_e)
            if pv:
                pmean = sum(pv) / len(pv)
                entry[label] = mx.pct_change(now_mean, pmean)
        out.append(entry)
    return out


def highlights(conn, as_of: str | None = None) -> list[dict]:
    """True positive facts for grounded encouragement: all-time PRs hit recently."""
    as_of = _as_of(conn, as_of)
    out = []
    recent_start = (date.fromisoformat(as_of) - timedelta(days=6)).isoformat()
    for m in ["step_count", "active_energy", "distance_walking_running", "vo2_max"]:
        if not mx.metric_exists(conn, m):
            continue
        col = mx.value_col(m)
        row = conn.execute(
            f"SELECT date, MAX({col}) v FROM daily_metrics WHERE metric = ?",
            (m,)).fetchone()
        if row and row["date"] and row["date"] >= recent_start:
            out.append({"metric": m, "kind": "all_time_high",
                        "value": mx.r(row["v"]), "date": row["date"]})
    return out


def suggestions(readiness_d: dict, load_d: dict,
                workout_d: dict | None = None) -> list[dict]:
    """Safe, reversible, non-medical nudges with a `because` citing numbers."""
    out = []
    band = readiness_d.get("band")
    if readiness_d.get("status") == "stale":
        # The one readiness state the user can act on immediately.
        n = readiness_d.get("stale_days")
        out.append({"text": "No recent recovery data — check the watch is charged, "
                            "worn overnight, and syncing.",
                    "because": f"newest HRV / resting-HR reading is {n} days old "
                               f"({readiness_d.get('latest_date')})"})
    elif readiness_d.get("status") == "establishing_baseline":
        out.append({"text": "Recovery metrics are still building a baseline — a few "
                            "more days of wear and these get meaningful.",
                    "because": "fewer than 14 baseline days available"})
    elif band == "red":
        f = readiness_d.get("factors", [])
        why = "; ".join(f"{x['component']} {x.get('pct')}%" for x in f[:2])
        out.append({"text": "Consider an easy or rest day; your body looks "
                            "under-recovered.", "because": why or "readiness in red band"})
    elif band == "green" and load_d.get("acwr") is not None and load_d["acwr"] < 0.9:
        out.append({"text": "You're fresh and recent volume is modest — a good day "
                            "to push if you want.",
                    "because": f"readiness {readiness_d.get('score')}, "
                               f"ACWR {load_d.get('acwr')}"})

    # When a distance workout is present, the specific next-day target below carries
    # the ramping-fast guidance instead, so skip this generic nudge to avoid repetition.
    if (load_d.get("acwr_band") == "ramping-fast"
            and not (workout_d and workout_d.get("distance_mi"))):
        out.append({"text": "You've ramped training load quickly — watch for niggles "
                            "and maybe slot in a lighter day.",
                    "because": f"ACWR {load_d.get('acwr')} (>1.5)"})

    if workout_d and workout_d.get("distance_mi"):
        dist = workout_d["distance_mi"]
        acwr_band = load_d.get("acwr_band")
        acwr = load_d.get("acwr")
        why = f"recent {dist} mi effort"
        if acwr is not None:
            why += f"; ACWR {acwr} ({acwr_band})"
        # The written plan (docs/fitness/week-NN.md) owns progression — ≤10%/week,
        # deload cadence, rest days. The coach never prescribes distance; it only
        # says how to *approach* tomorrow's planned session. A red recovery read or
        # a fast ramp argue for backing off; anything else defers to the plan as
        # written. (An earlier version computed ±10% distance targets here, which
        # contradicted rest days and the plan's growth cap.)
        if acwr_band == "ramping-fast" or readiness_d.get("band") == "red":
            out.append({"text": "Keep tomorrow easy — trim the planned session or "
                                "take the rest day, not another hard push.",
                        "because": why})
        elif acwr_band == "caution":
            out.append({"text": "Hold steady tomorrow — do the planned session as "
                                "written, nothing extra.", "because": why})
        else:
            out.append({"text": "You're fresh — a good day to do tomorrow's planned "
                                "session exactly as written; the plan owns "
                                "progression, so don't add to it.", "because": why})
    return out


def talking_points(parts: dict) -> list[dict]:
    """Ordered factual seeds the narrator walks 1:1."""
    tp = []
    cov = {c["metric"]: c for c in parts["coverage"]}
    sparse = [m for m, c in cov.items() if c["status"] in COVERAGE_THIN]
    stopped = [(m, c) for m, c in cov.items() if c["status"] in COVERAGE_STOPPED]

    wf = parts.get("workout_focus")
    pace_key = ("pace_min_per_mi" if wf and "pace_min_per_mi" in wf
                else "pace_min_per_km")
    speed_key = ("speed_mph" if wf and "speed_mph" in wf
                 else "speed_kph")
    if wf and wf.get("distance_mi") and (wf.get(pace_key) or wf.get(speed_key)):
        # Date-stamp the seed: the focus workout may be up to 2 days old, and an
        # undated seed reads as "today" to the narrator (observed fabrication:
        # a rest-day briefing praising the previous day's ride as today's).
        when = "today" if wf.get("date") == parts.get("as_of") \
            else f"on {wf.get('date')} (not today)"
        if wf.get(speed_key) is not None:
            speed_unit = "mph" if speed_key == "speed_mph" else "kph"
            seed = (f"{wf['type']} workout {when}: {wf['distance_mi']} mi at "
                    f"{wf[speed_key]} {speed_unit}")
            numbers = [wf["distance_mi"], wf[speed_key]]
        elif wf.get("jog_pace_min_per_mi") is not None:
            # Say the jog pace where the day has one: it is the number the ramp
            # and the "slow down" lever are both defined on. The blended figure
            # stays available in the dict, it just isn't what gets narrated.
            seed = (f"{wf['type']} workout {when}: {wf['distance_mi']} mi, "
                    f"jog pace {wf['jog_pace_min_per_mi']} min/mi "
                    f"(walk breaks excluded)")
            numbers = [wf["distance_mi"], wf["jog_pace_min_per_mi"]]
        else:
            pace_unit = "min/mi" if pace_key == "pace_min_per_mi" else "min/km"
            seed = (f"{wf['type']} workout {when}: {wf['distance_mi']} mi at "
                    f"blended pace {wf[pace_key]} {pace_unit}")
            numbers = [wf["distance_mi"], wf[pace_key]]
        tp.append({"topic": "workout", "seed": seed, "numbers": numbers})

    rd = parts["readiness"]
    if rd.get("score") is not None:
        tp.append({"topic": "readiness",
                   "seed": f"recovery readiness {rd['score']}/100 ({rd['band']})",
                   "numbers": [rd["score"]]})
    elif rd.get("status") == "stale":
        tp.append({"topic": "readiness",
                   "seed": f"no recovery data for {rd.get('stale_days')} days "
                           f"(newest reading {rd.get('latest_date')}) — readiness "
                           f"not scored",
                   "numbers": [n for n in [rd.get("stale_days")] if n is not None]})
    elif rd.get("status") == "establishing_baseline":
        tp.append({"topic": "readiness",
                   "seed": "recovery metrics still establishing a baseline",
                   "numbers": []})

    ew = parts["trends"].get("early_warning", {})
    if ew.get("flag"):
        tp.append({"topic": "early_warning",
                   "seed": "resting HR rising and/or HRV falling over ~3 weeks — "
                           "watch recovery",
                   "numbers": [n for n in (parts["trends"].get("rhr_per_week"),
                                           parts["trends"].get("hrv_per_week"))
                               if n is not None]})

    tl = parts["training_load"]
    if tl.get("acwr") is not None:
        tp.append({"topic": "load",
                   "seed": f"training load ACWR {tl['acwr']} ({tl['acwr_band']})",
                   "numbers": [tl["acwr"]]})

    for mv in parts["movers"]:
        tp.append({"topic": "mover",
                   "seed": f"{mv['metric']} {mv['direction']} {mv['pct']}% vs baseline",
                   "numbers": [mv["pct"], mv["recent_avg"], mv["baseline_avg"]]})

    for hl in parts["highlights"]:
        tp.append({"topic": "highlight",
                   "seed": f"new recent high for {hl['metric']}: {hl['value']}",
                   "numbers": [hl["value"]]})

    if sparse:
        tp.append({"topic": "coverage",
                   "seed": f"thin/sparse data for: {', '.join(sparse)}", "numbers": []})

    # Date-stamped, for the same reason the workout seed is: an undated "no data"
    # seed reads as "since forever" and gives no way to tell a one-week gap from
    # a two-month one.
    for m, c in stopped:
        when = (f"last reading {c['last_date']}" if c.get("last_date")
                else "never any reading")
        tp.append({"topic": "coverage",
                   "seed": f"{m} has stopped arriving — {when}", "numbers": []})
    return tp


def build_briefing(conn, scope: str = "daily", as_of: str | None = None, *,
                   metric_units: bool = False) -> dict:
    as_of = _as_of(conn, as_of)
    rd = readiness(conn, as_of)
    tl = training_load(conn, as_of)
    wf = workout_focus(conn, as_of, metric_units=metric_units)
    parts = {
        "as_of": as_of,
        "scope": scope,
        "coverage": coverage(conn, as_of),
        "readiness": rd,
        "trends": trends(conn, as_of),
        "training_load": tl,
        "movers": movers(conn, as_of, scope),
        "long_term": long_term(conn, as_of) if scope == "deep" else [],
        "suggestions": suggestions(rd, tl, wf),
        "highlights": highlights(conn, as_of),
        "workout_focus": wf,
    }
    parts["talking_points"] = talking_points(parts)
    return parts


# A month must carry at least this many samples before its dominant source
# counts, and a new source must hold for at least this many qualifying months
# before it is called an era. Both guards exist because the unguarded rule
# measured badly on real data: body_mass produced eight eras, four of them a
# SINGLE reading, because a month with one weigh-in lets one sample decide the
# "dominant" source. Declaring an instrument change on one sample is a worse
# error than the one this is fixing — history.py refuses to average across a
# boundary, so a spurious boundary silently destroys a real series.
ERA_MIN_MONTH_SAMPLES = 5
ERA_MIN_PERSIST_MONTHS = 2


def _source_months(conn, metric: str, start: str, end: str):
    """(rows, provenance) of monthly source counts, or (None, None) if neither
    source of provenance has anything for this metric.

    `metric_source_months` is preferred and is the only one that exists in a D3
    vault. Raw `records` is consulted when the derived table has not been built
    yet — every database that predates it is in that state, and the raw table is
    the authority the derived one is built from, so reading it is not a way
    around the contract.
    """
    rows = conn.execute(
        "SELECT month, source, n FROM metric_source_months "
        "WHERE metric = ? AND month BETWEEN ? AND ? "
        "ORDER BY month, n DESC, source",
        (metric, start[:7], end[:7]),
    ).fetchall()
    if rows:
        return rows, "metric_source_months"
    rows = conn.execute(
        "SELECT substr(local_date, 1, 7) AS month, source, COUNT(*) AS n "
        "FROM records WHERE metric = ? AND local_date BETWEEN ? AND ? "
        "GROUP BY month, source ORDER BY month, n DESC, source",
        (metric, start, end),
    ).fetchall()
    if rows:
        return rows, "records"
    return None, None


def instrument_eras_status(conn, metric: str, start: str, end: str) -> dict:
    """Instrument-era boundaries, or an explicit statement that they are unknown.

    `history.py` refuses to average across a >14-day data gap, but the watch
    leaving in 2022 changed the instrument underneath a series that never
    gapped — a series that never gaps can still change instrument (F3-2).
    `daily_metrics` carries no source column, so provenance comes from
    `metric_source_months` (or from raw `records`, which is what that table is
    built from).

    **When neither exists the answer is `unavailable`, never an empty list.**
    Those are different claims: "no instrument change happened" licenses
    averaging across the whole series, and "we cannot see whether one happened"
    does not. Under D3 most series' samples never reach a vault, so this
    distinction is the difference between a correct refusal and F3-2 happening
    again silently.

    Source values are grouped exactly as stored: a compound source string is one
    source class, not an invitation to infer or de-duplicate its contributing
    devices. This reports WHEN the instrument changed, which is true and useful,
    and deliberately does not attempt the hourly-maximum reconstruction an
    independent methodology review ruled NOT USABLE (F3-1) — Apple does not
    publish its overlap-resolution algorithm. Ties break deterministically by
    source text.
    """
    rows, provenance = _source_months(conn, metric, start, end)
    if rows is None:
        return {**V.raw_unavailable(metric,
                                        needed_for="instrument era detection"),
                "boundaries": [], "provenance": None}

    dominant, totals = {}, {}
    for row in rows:
        month, source, n = row["month"], row["source"], row["n"]
        totals[month] = totals.get(month, 0) + n
        dominant.setdefault(month, source)
    months = [m for m in sorted(dominant) if totals[m] >= ERA_MIN_MONTH_SAMPLES]

    boundaries, previous = [], None
    for i, month in enumerate(months):
        source = dominant[month]
        if previous is not None and source != previous:
            # Only a change that STICKS is an instrument change. One month of
            # a different dominant source is a trip, a loaner, a dead battery.
            ahead = [dominant[m] for m in months[i:i + ERA_MIN_PERSIST_MONTHS]]
            if len(ahead) == ERA_MIN_PERSIST_MONTHS and len(set(ahead)) == 1:
                boundaries.append(f"{month}-01")
                previous = source
            continue
        previous = source
    return {"status": "ok", "metric": metric, "provenance": provenance,
            "boundaries": boundaries}


def instrument_eras(conn, metric: str, start: str, end: str) -> list[str]:
    """The boundaries alone. `unavailable` collapses to `[]` here, so callers
    that must tell the two apart use :func:`instrument_eras_status`."""
    return instrument_eras_status(conn, metric, start, end)["boundaries"]
