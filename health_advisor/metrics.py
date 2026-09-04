"""Shared analytic primitives over the health DB. Extracted from mcp_server.py so
the MCP tools and analysis.py share ONE implementation (no numeric divergence).
All functions are pure given a connection; callers manage the connection."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np

from . import normalize as nz

MAX_SERIES_POINTS = 400

# Single definition of the watch-wear floor: a day counts as "worn" at or above
# 12 hours of wear_hours (or the density proxy where wear_hours doesn't cover
# it). analysis.py and correlate.py both compare daily wear against this same
# threshold, so it lives here once and each module imports it rather than
# keeping its own copy — see #127 (F-50), and #53 (numeric_tokens.NUM_RE) for
# the precedent. analysis.WEAR_MIN_HOURS and correlate.WEAR_MIN_HOURS are this
# same object, not equal copies.
WEAR_MIN_HOURS = 12.0

# Presentation semantics are deliberately metric-specific.  In particular,
# ``unit == "h"`` is not enough to choose a renderer: bedtime and midpoint are
# continuous hours after the previous noon, wake time is hours after midnight,
# midpoint SD is a duration, and wear_hours is a duration.  Keep this table
# explicit so a new hour-valued metric cannot silently become a clock time.
_DURATION_MINUTE_METRICS = frozenset({
    "apple_exercise_time", "apple_stand_time", "time_in_daylight",
    "sleep_in_bed", "sleep_asleep", "sleep_awake", "sleep_rem",
    "sleep_deep", "sleep_core", "sleep_time_in_bed", "sleep_awake_longest",
    "sleep_latency", "jog_minutes", "longest_block_min", "mindful_minutes",
})
_PREVIOUS_NOON_CLOCK_METRICS = frozenset({"sleep_bedtime", "sleep_midpoint"})
_MIDNIGHT_CLOCK_METRICS = frozenset({"sleep_wake_time"})
_SIGNED_DURATION_HOUR_METRICS = frozenset({"sleep_midpoint_sd_28d"})
_DURATION_HOUR_METRICS = frozenset({"wear_hours"})

# A presentation renderer is selected by the field as well as the metric.
# Keep this table explicit: a new result field must not silently inherit the
# metric renderer, which would make a percentage or rate look like a duration.
_UNIT_PRESERVING_FIELDS = frozenset({
    "value", "mean", "median", "min", "max", "std", "latest", "sum",
    "total", "recent_avg", "baseline_avg", "jog_minutes", "latest_sd_hours",
})
_SIGNED_UNIT_PRESERVING_FIELDS = frozenset({
    "delta_vs_baseline", "mean_delta", "total_delta",
})
_NON_UNIT_PRESERVING_FIELDS = frozenset({
    "delta_pct", "total_delta_pct", "trend_per_week",
})
_PRESENTATION_FIELDS = (_UNIT_PRESERVING_FIELDS
                        | _SIGNED_UNIT_PRESERVING_FIELDS
                        | _NON_UNIT_PRESERVING_FIELDS)


def format_presentation(metric: str, value, *, clock_24: bool = False,
                        compact: bool = False, field: str = "value") -> str | None:
    """Render one stored metric value for a person, deterministically.

    This is the one presentation formatter for the tool/ledger surfaces.  It
    owns the unit conversion and rounding; callers only publish its returned
    string and never reconstruct it from a raw value.  The field table above
    decides whether the metric renderer applies at all. Unknown metrics and
    fields return ``None`` rather than guessing from a unit label.
    """
    if field not in _PRESENTATION_FIELDS or field in _NON_UNIT_PRESERVING_FIELDS:
        return None
    signed_field = field in _SIGNED_UNIT_PRESERVING_FIELDS
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    def duration(total_minutes: float, prefix: str = "") -> str:
        sign = ""
        if signed_field:
            sign = "+" if total_minutes > 0 else "-" if total_minutes < 0 else ""
            total = int(round(abs(total_minutes)))
        else:
            total = max(0, int(round(total_minutes)))
        prefix = sign + prefix
        hours, minutes = divmod(total, 60)
        if hours:
            if compact:
                return (f"{prefix}{hours}h" if not minutes else
                        f"{prefix}{hours}h {minutes}m")
            return (f"{prefix}{hours} h" if not minutes else
                    f"{prefix}{hours} h {minutes:02d} m")
        if metric == "duration_min":
            return f"{prefix}{minutes} min"
        return (f"{prefix}{minutes}m" if compact else
                f"{prefix}{minutes} m")

    if metric in _DURATION_MINUTE_METRICS:
        return duration(value)
    if metric == "duration_min":
        return duration(value)
    if metric in _DURATION_HOUR_METRICS:
        return duration(value * 60)
    if metric in _SIGNED_DURATION_HOUR_METRICS:
        return duration(value * 60, prefix="± ")
    if metric in _PREVIOUS_NOON_CLOCK_METRICS:
        total = int((12.0 + value) * 60) % (24 * 60)
    elif metric in _MIDNIGHT_CLOCK_METRICS:
        total = int(value * 60) % (24 * 60)
    else:
        return None
    hour24, minute = divmod(total, 60)
    sign = ""
    if signed_field:
        sign = "+" if value > 0 else "-" if value < 0 else ""
    if clock_24:
        return f"{sign}{hour24:02d}:{minute:02d}"
    suffix = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12 or 12
    return f"{sign}{hour12}:{minute:02d} {suffix}"


def presentation_leaf(metric: str, period, value, *, field: str = "value") -> dict | None:
    """Return the claimable leaf beside a raw value, if it has one."""
    rendered = format_presentation(metric, value, field=field)
    if rendered is None:
        return None
    return {"metric": metric, "period": period, "field": "presentation",
            "value": rendered}


def presentation_clock_parts(metric: str, value) -> tuple[int, int] | None:
    """Return 24-hour clock components from the canonical clock rendering."""
    rendered = format_presentation(metric, value)
    if rendered is None or " " not in rendered or ":" not in rendered:
        return None
    clock, suffix = rendered.rsplit(" ", 1)
    try:
        hour12, minute = (int(part) for part in clock.split(":", 1))
    except (TypeError, ValueError):
        return None
    hour24 = hour12 % 12
    if suffix == "PM":
        hour24 += 12
    return hour24, minute

# The HR model is intentionally a module-level namespace: consumers import
# this module as ``hr_model`` rather than copying thresholds into their own
# surface. The easy jogging band is 145–155 bpm, with 155 as the
# block-qualification ceiling. That ceiling is a convention, not a
# physiological threshold; nothing physiological happens at exactly 155.
# Ordinary jog sessions use a separate 160-bpm session cap. A continuous test
# or benchmark may reach 165 bpm when the talk test still permits full
# sentences.
#
# ALL FIVE MOVED TOGETHER on 2026-08-30, settling P4-2 (band 130–145 →
# 145–155, session cap 155 → 160, qualification ceiling 150 → 155, test
# allowance 160 → 165). These bands were tuned against one athlete's record
# during development, not derived from population physiology: for that
# athlete the old band sat at 64–71% of a corroborated max of 204 — a
# recovery zone, not an aerobic-base one. What settled it was a measurement,
# not the HRR arithmetic: Pa:HR decoupling of +1.38% over a 30.7-minute block
# at a mean of 150.5 bpm on confirmed flat ground (2026-08-29), closing at
# 155–156 with pace still improving.
#
# The band is on probation: >5% decoupling on two consecutive long runs sends
# the qualification ceiling back to 150.
EASY_JOG_HR_MIN = 145.0
EASY_JOG_HR_MAX = 155.0
EASY_JOG_CEILING = 155.0
JOG_SESSION_HR_CAP = 160
TEST_BENCHMARK_HR_CAP = 165

# Impact-volume bucket classification. Lives here rather than in analysis.py so
# analysis.impact_volume and metrics.bucket_series cannot drift apart.
IMPACT_BUCKET_SECONDS = 20
# Samples longer than the 20-second activity bucket are distributed over the
# buckets they cover below. The existing 12-hour boundary remains the guard
# for daily-total-style rows: it protects the pre-June-2026 history without
# pretending that a 599-second live interval is a 20-second observation.
IMPACT_MAX_SAMPLE_SPAN_SECONDS = 12 * 60 * 60
IMPACT_JOG_PACE_MAX = 16.0     # min/mi; retained for the separate block dial
IMPACT_WALK_PACE_MAX = 40.0    # slower than this is standing/GPS noise, not walking
# No lower pace bound existed here, so a car journey read as jogging: 2026-06-03
# carries a chain of buckets at 29-59 mph with no heart rate and no workout, and
# they were 43% of that week's jog minutes. A sub-5-minute mile is not
# plausible travel on foot. Buckets faster than this are neither jogging nor
# walking — the same treatment the 40 min/mi ceiling gives to standing and GPS
# noise.
IMPACT_IMPLAUSIBLE_PACE_MIN = 5.0  # min/mi; faster than this is not human travel

# Jogging is now a workout-scoped cadence classification. The pace constants
# remain part of the bucket payload and the separate block dial; they do not
# classify impact-volume jogging.
IMPACT_JOG_CADENCE_MIN = 140.0  # steps/min; walk/run gait transition; six oracles land within a minute of it
# Retained for the separate block bridge predicate; not an impact-volume lane.
IMPACT_JOG_HR_PACE_MAX = 18.0  # min/mi; block bridge's slow-bucket ceiling
IMPACT_JOG_HR_MIN = 130.0      # bpm; block bridge's slow-bucket floor

# --- the bridge rule: how a bucket is classified, item 3 ----------------------
# A continuous jog is not a continuous chain of <=16 min/mi buckets. GPS drops a
# bucket, a road crossing slows one, and a single 20-second sample splits a run
# in two. On 2026-08-15 exactly that happened: a 9.3-minute continuous jog was
# reported as 4.0 — the best session on record read as the worst — because
# two buckets in the middle fell to 16-18 min/mi.
#
# So up to TWO consecutive buckets at 16-18 min/mi MAY bridge two jog segments,
# but only with HR >= 130 confirming the athlete was still running through
# them, only when the bucket indices are genuinely contiguous (no missing
# time), and only when flanked by jog buckets on BOTH sides — a trailing slow
# tail does not extend a block. Backtested across every running session since
# 2026-06-22 the bridge changes exactly one number, which is the point: it is
# a repair for a specific artifact, not a loosening.
BLOCK_BRIDGE_MAX_BUCKETS = 2
BLOCK_BRIDGE_PACE_MAX = 18.0   # min/mi; same ceiling as the HR-confirmed jog lane
BLOCK_BRIDGE_HR_MIN = 130.0    # bpm; same floor, and for the same reason

# The ceiling a continuous block's MEAN heart rate must clear to count toward the
# ramp (P2-1, adopted at the Week 7 review 2026-08-16).
#
# This is NOT the session cap. JOG_SESSION_HR_CAP (155) governs whether an
# ordinary jog session was run too hard; this governs whether a block counts
# as evidence the athlete can run that long EASY. Two numbers, two jobs, and
# collapsing them is the F4-4 defect this file exists to prevent.
#
# Written as its own literal ON PURPOSE, even though it currently equals
# EASY_JOG_CEILING. Aliasing it (`= EASY_JOG_CEILING`) would mean that moving
# the planning band silently moves the ramp's evidence bar too — the same
# coupling defect as writing the number four times, just running the other way.
# If the two ever have to move together, that is a decision someone makes here,
# once, in writing.
#
# Audit part 4 measured 150 at 63% HRR / 75% HRmax and recorded plainly that
# nothing physiological happens at it — a defensible mid-range convention, not
# a threshold.
#
# THIS IS THAT DECISION, MADE ONCE, IN WRITING (decided 2026-08-30, settling
# P4-2). The two DO move together this time, and deliberately — the ramp's
# evidence bar was the specific thing under discussion, not a side effect of
# the planning band moving. On 2026-08-29 a 30.7-minute unbroken block came
# back qualified_min=None because its MEAN was 150.5, half a beat over this
# literal, so the best session in the plan's history scored zero on the ramp.
# Decoupling on that same block was +1.38% on flat ground, which is the
# evidence that 150 was the wrong bar rather than 150.5 being too hot.
#
# The ceiling is a per-vault setting (vault_meta 'block_qualify_hr_max').
# DEFAULT_BLOCK_QUALIFY_HR_MAX is the labelled legacy default an undeclared
# vault resolves to, so figures do not move when a vault has never declared
# one; it is a default, not a personal parameter, and consumers that only
# need the default read this name rather than a per-vault value.
DEFAULT_BLOCK_QUALIFY_HR_MAX = 155.0


def block_qualify_hr_max(conn) -> float:
    """Return the vault ceiling, or DEFAULT_BLOCK_QUALIFY_HR_MAX when undeclared."""
    from . import vault

    configured = vault.block_qualify_hr_max(conn)
    return configured if configured is not None else DEFAULT_BLOCK_QUALIFY_HR_MAX

# Bedtime plan bands from P7-1, adopted at the Week 7 review on 2026-08-16.
# Values are hours since the previous day's noon: 23:00 is 11.0 and 00:30 is
# 12.5. The middle band is deliberately ungraded because the data cannot tell
# a social night from a scrolling night.
BEDTIME_ANCHOR_H = 11.0
BEDTIME_SOCIAL_LIMIT_H = 12.5


def impact_bucket_rows(conn, window_predicate: str,
                       window_args: tuple[str, str], *,
                       arbitration_window: tuple[str, str] | None = None,
                       arbitration_window_kind: str = "local_date") -> list[dict]:
    """Return the shared classified impact buckets for a parameterised window.

    ``metrics.bucket_series`` and ``analysis.impact_volume`` both depend on
    this one b/h/c rule.  Callers supply their question's window predicate:
    bucket_series uses the UTC timestamp window, while impact_volume uses the
    inclusive local-date window.  The bucket width, positive-distance HAVING
    clause, span guard, pace fields, and cadence/workout-window jog rule are
    deliberately not caller parameters.

    Workout-window device arbitration is applied to the distance CTE before
    bucketing, so the same selected stream feeds ``impact_volume`` and
    ``bucket_series``. Post-boundary interval samples are distributed across
    every 20-second UTC bucket they overlap, in proportion to the overlap
    duration, whether or not they fall inside a workout window. This prevents
    a long HealthKit interval from becoming one fictitious fast bucket while
    retaining its distance. Point samples and rows before the vault's cutoff
    remain one bucket. The 12-hour span guard still rejects daily-total-style
    rows, and all of this is shared by both callers.

    ``arbitration_window`` scopes the source-presence tests used by
    ``db._workout_arbitration``. It defaults to the same two bounds supplied to
    this query; ``bucket_series`` overrides the kind to ``utc`` while the
    local-date impact-volume callers retain the default. A row outside the
    requested session therefore cannot decide which device wins inside it.
    This scopes raw-sample selection only; it does not reconstruct a daily
    distance total.

    `window_predicate` is interpolated into the SQL, so it must be a literal
    written here in the source. It is not a place to put anything that reached
    the process from a tool argument or a request.
    """
    bucket_min = IMPACT_BUCKET_SECONDS / 60.0
    implausible_mi_ceiling = bucket_min / IMPACT_IMPLAUSIBLE_PACE_MIN
    if arbitration_window is None:
        arbitration_window = window_args
    from . import db as dbmod
    workout_cutoff = dbmod.workout_source_arbitration_cutoff(conn).replace("'", "''")
    # These are the two literal windows used by the production callers. Keep
    # their repeated record filters in one CTE so the workout lookup can use
    # the indexed local-date candidate bound without adding duplicate binds.
    # The candidate bound is a day either side of the window, so a workout
    # spanning more than a day would be missed by in_workout/is_jog. Measured
    # on the reference vault: 805 workouts, the longest 11.8 h, and identical
    # bucket rows before and after on five windows (2026-09-03).
    known_window = window_predicate in {
        "local_date BETWEEN ? AND ?", "start_utc >= ? AND start_utc < ?",
    }
    workout_active_condition = dbmod.workout_mark_condition(conn, "w")
    if known_window:
        impact_window_cte = (
            "impact_window(start_value, end_value) AS (VALUES (?, ?)),"
        )
        candidate_start = (date.fromisoformat(window_args[0][:10])
                           - timedelta(days=1)).isoformat()
        candidate_end = (date.fromisoformat(window_args[1][:10])
                         + timedelta(days=1)).isoformat()
        workout_candidates_cte = (
            "workout_candidates AS MATERIALIZED ("
            "SELECT w.* FROM workouts AS w "
            "WHERE local_date >= ? AND local_date <= ? "
            f"AND {workout_active_condition}),"
        )
        candidate_window_args = (candidate_start, candidate_end)
        if window_predicate == "local_date BETWEEN ? AND ?":
            query_record_scope = (
                " AND records.local_date >= "
                "(SELECT start_value FROM impact_window)"
                " AND records.local_date <= "
                "(SELECT end_value FROM impact_window)"
            )
        else:
            query_record_scope = (
                " AND records.start_utc >= "
                "(SELECT start_value FROM impact_window)"
                " AND records.start_utc < "
                "(SELECT end_value FROM impact_window)"
            )
        record_window_filter = query_record_scope
        cte_window_args = window_args
        repeated_window_args = ()
        workout_source = "workout_candidates"
    else:
        impact_window_cte = ""
        workout_candidates_cte = ""
        query_record_scope = ""
        record_window_filter = " AND " + window_predicate
        cte_window_args = ()
        candidate_window_args = ()
        repeated_window_args = window_args
        workout_source = "workouts"
    source_clause, source_args = dbmod._workout_arbitration(
        conn, "distance_walking_running",
        arbitration_window=arbitration_window,
        arbitration_window_kind=arbitration_window_kind)
    rows = conn.execute(
        f"""
        WITH RECURSIVE {impact_window_cte} {workout_candidates_cte} distance AS (
          SELECT local_date, value,
                 CAST(strftime('%s', start_utc) AS INTEGER) AS start_s,
                 CAST(strftime('%s', end_utc) AS INTEGER) AS end_s,
                 CAST(strftime('%s', start_utc) / {IMPACT_BUCKET_SECONDS} AS INT)
                   AS first_bkt,
                 CASE WHEN local_date < '{workout_cutoff}'
                      THEN CAST(strftime('%s', start_utc) / {IMPACT_BUCKET_SECONDS} AS INT)
                      ELSE CAST((CAST(strftime('%s', end_utc) AS INTEGER) - 1)
                                / {IMPACT_BUCKET_SECONDS} AS INT)
                  END AS last_bkt,
                 CASE WHEN local_date < '{workout_cutoff}'
                      THEN 1 ELSE 0 END AS preserve_one_bucket
            FROM records
           WHERE metric = 'distance_walking_running'
             AND {window_predicate}
             {source_clause}
             AND CAST(strftime('%s', end_utc) AS INTEGER)
                 - CAST(strftime('%s', start_utc) AS INTEGER) < ?
        ), expanded AS (
          SELECT local_date, value, start_s, end_s, first_bkt AS bkt, last_bkt,
                 preserve_one_bucket
            FROM distance
          UNION ALL
          SELECT local_date, value, start_s, end_s, bkt + 1, last_bkt,
                 preserve_one_bucket
            FROM expanded
           WHERE bkt < last_bkt
        ), b AS (
          SELECT local_date, bkt,
                 SUM(CASE WHEN preserve_one_bucket OR end_s <= start_s
                          THEN value
                          WHEN end_s > start_s
                          THEN value * (
                               MIN(end_s, (bkt + 1) * {IMPACT_BUCKET_SECONDS})
                               - MAX(start_s, bkt * {IMPACT_BUCKET_SECONDS})
                             ) / (end_s - start_s)
                          ELSE value END) AS mi
            FROM expanded
        GROUP BY local_date, bkt
          HAVING SUM(CASE WHEN preserve_one_bucket OR end_s <= start_s
                          THEN value
                          ELSE value * (
                               MIN(end_s, (bkt + 1) * {IMPACT_BUCKET_SECONDS})
                               - MAX(start_s, bkt * {IMPACT_BUCKET_SECONDS})
                             ) / (end_s - start_s) END) > 0
        ),
        h AS (
          SELECT local_date,
                 CAST(strftime('%s', start_utc) / {IMPACT_BUCKET_SECONDS} AS INT) AS bkt,
                 AVG(value) AS hr
            FROM records
           WHERE metric = 'heart_rate'
             {record_window_filter}
        GROUP BY local_date, bkt
        ),
        s AS (
          SELECT CAST(strftime('%s', start_utc) / {IMPACT_BUCKET_SECONDS} AS INT) AS bkt,
                 SUM(value) * 3.0 AS cadence_spm
            FROM records
           WHERE metric = 'step_count'
             {record_window_filter}
        GROUP BY bkt
        ),
        c AS (
          SELECT b.bkt AS bkt, b.local_date AS local_date, b.mi AS mi,
                 h.hr AS hr,
                 s.cadence_spm AS cadence_spm,
                 CASE WHEN EXISTS (
                                     SELECT 1 FROM {workout_source} w
                                       WHERE {workout_active_condition}
                                     AND CAST(strftime('%s', w.start_utc) AS INTEGER)
                                               <= b.bkt * {IMPACT_BUCKET_SECONDS}
                                     AND CAST(strftime('%s', w.end_utc) AS INTEGER)
                                       > b.bkt * {IMPACT_BUCKET_SECONDS}
                            ) THEN 1 ELSE 0 END AS in_workout,
                 CASE WHEN b.mi <= ?
                            AND s.cadence_spm >= ?
                            AND EXISTS (
                                  SELECT 1 FROM {workout_source} w
                                   WHERE {workout_active_condition}
                                     AND CAST(strftime('%s', w.start_utc) AS INTEGER)
                                           <= b.bkt * {IMPACT_BUCKET_SECONDS}
                                     AND CAST(strftime('%s', w.end_utc) AS INTEGER)
                                       > b.bkt * {IMPACT_BUCKET_SECONDS}
                            )
                      THEN 1 ELSE 0 END AS is_jog
            FROM b LEFT JOIN h
              ON h.local_date = b.local_date AND h.bkt = b.bkt
            LEFT JOIN s ON s.bkt = b.bkt
        )
        SELECT bkt, local_date, mi, hr, cadence_spm, in_workout, is_jog,
               CASE WHEN is_jog = 0 AND mi >= ? AND mi <= ?
                    THEN 1 ELSE 0 END AS is_walk
          FROM c
      ORDER BY bkt, local_date
        """,
            (*cte_window_args, *candidate_window_args, *window_args,
             *source_args,
             IMPACT_MAX_SAMPLE_SPAN_SECONDS,
             *repeated_window_args,
             *repeated_window_args,
         implausible_mi_ceiling, IMPACT_JOG_CADENCE_MIN,
         bucket_min / IMPACT_WALK_PACE_MAX,
         implausible_mi_ceiling),
    ).fetchall()
    return [dict(row) for row in rows]


def bucket_series(conn, start_utc: str, end_utc: str, *,
                  metric_units: bool = False) -> list[dict]:
    """Per-bucket view of the jog/walk classification, over a UTC window.

    Same rule as analysis.impact_volume — that function aggregates these
    buckets; this returns them. Scoped by timestamp rather than by local date
    so a single workout's window can be asked for directly: a 50-minute run is
    ~150 buckets. Do NOT call this over a multi-year range.

    `end_utc` is exclusive. Values are unrounded; round at the presentation
    edge. `speed_mph` is carried explicitly so no caller has to invert pace —
    efficiency is speed/HR, and a ratio built on pace moves the wrong way.
    """
    if metric_units:
        from . import vault as V

    bucket_min = IMPACT_BUCKET_SECONDS / 60.0
    rows = impact_bucket_rows(
        conn, "start_utc >= ? AND start_utc < ?", (start_utc, end_utc),
        arbitration_window=(start_utc, end_utc),
        arbitration_window_kind="utc")

    out: list[dict] = []
    for row in rows:
        mi = float(row["mi"])
        hr = float(row["hr"]) if row["hr"] is not None else None
        pace = bucket_min / mi if mi > 0 else None
        is_jog = bool(row["is_jog"])
        is_walk = bool(row["is_walk"])
        out.append({
            "bucket_start_utc": datetime.fromtimestamp(
                int(row["bkt"]) * IMPACT_BUCKET_SECONDS, timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "local_date": row["local_date"],
            "miles": mi,
            "hr": hr,
            "cadence_spm": (float(row["cadence_spm"])
                            if row["cadence_spm"] is not None else None),
            **({
                "speed_kph": (mi / (bucket_min / 60.0))
                             * V.UNIT_CONVERSION_FACTORS["distance_mi_to_km"]
                             if mi > 0 else None,
                "pace_min_per_km": (pace / V.UNIT_CONVERSION_FACTORS[
                    "distance_mi_to_km"] if pace is not None else None),
            } if metric_units else {
                "speed_mph": (mi / (bucket_min / 60.0)) if mi > 0 else None,
                "pace_min_per_mi": pace,
            }),
            "is_jog": is_jog,
            "is_walk": is_walk,
        })
    return out


def session_hr_figures(conn, start_utc: str, end_utc: str,
                       session_avg: float | None = None) -> dict:
    """The distinct "average heart rates" of one session, each named for its scope.

    Three different quantities have all been called "average HR" here, and they
    are up to 20 bpm apart on the same session (2026-08-15: 123.4 whole-session
    vs 143.6 over jog buckets). A session can grade inside the easy band on one
    and outside on the other, so an unlabelled figure is not a number, it is a
    coin flip. F6-5, audit part 6 — same defect class as commit 19c5edc, where
    the briefing handed the agent a blended pace with no label.

    Returns, all optional:
      avg_hr_session  -- whole session INCLUDING prescribed walk breaks. This is
                         what workouts.avg_heart_rate holds; passed in rather
                         than recomputed so the two cannot disagree.
      avg_hr_all_jog  -- mean over jog buckets only. Runs HIGHER than the
                         session mean, because walk breaks drag that one down.
      n_jog_buckets   -- how much jogging the figure rests on. Two buckets is
                         not a session average and the caller should say so.

    `avg_hr_longest_block` (the mean over the single longest continuous block)
    is the third scope and arrives with analysis.longest_block().
    """
    buckets = bucket_series(conn, start_utc, end_utc)
    jog = [b["hr"] for b in buckets if b["is_jog"] and b["hr"] is not None]
    return {
        "avg_hr_session": r(session_avg, 1) if session_avg is not None else None,
        "avg_hr_all_jog": r(sum(jog) / len(jog), 1) if jog else None,
        "n_jog_buckets": len(jog),
        # Zero buckets means no per-sample distance at all, which is a different
        # state from "distance exists and none of it was jogging" — the first is
        # unclassifiable, the second is a walk. Callers need to tell them apart.
        "n_buckets": len(buckets),
    }


def r(x, nd=2):
    """Round to a JSON-safe python float (handles numpy / None / NaN)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return round(f, nd)


def agg(metric: str) -> str:
    return nz.agg_for(metric)


def value_col(metric: str) -> str:
    aggregation = agg(metric)
    return "sum" if aggregation == "sum" else "last" if aggregation == "last" else "avg"


def metric_exists(conn, metric: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM daily_metrics WHERE metric = ? LIMIT 1", (metric,)
    ).fetchone() is not None


def anchor_end(conn, metric: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(date) FROM daily_metrics WHERE metric = ?", (metric,)
    ).fetchone()
    return row[0] if row and row[0] else None


def parse_period(spec: str, end_iso: str):
    """'30d','12w','6m','1y','all' -> (start_iso|None, end_iso). Inclusive.

    Raises ValueError on anything else. It used to fall back to 30d, so
    compare_periods('june','july') compared the last 30 days against themselves
    and answered "0% change" — a fabricated window the agent cannot detect."""
    spec = (spec or "30d").strip().lower()
    end = date.fromisoformat(end_iso)
    if spec in ("all", "max", "lifetime"):
        return None, end_iso
    try:
        num, unit = int(spec[:-1]), spec[-1]
        mult = {"d": 1, "w": 7, "m": 30, "y": 365}[unit]
    except (ValueError, KeyError):
        raise ValueError(
            f"could not parse period {spec!r}; use a count and a unit "
            "('30d', '12w', '6m', '1y'), 'all', or an explicit "
            "'YYYY-MM-DD:YYYY-MM-DD' range"
        ) from None
    if num < 1:
        raise ValueError(f"period {spec!r} must cover at least one day")
    start = end - timedelta(days=num * mult - 1)
    return start.isoformat(), end_iso


def parse_range(spec: str, anchor_end_iso: str):
    """A period spec or an explicit 'YYYY-MM-DD:YYYY-MM-DD' range. Raises
    ValueError if the range isn't two real dates in order."""
    spec = (spec or "").strip()
    if ":" in spec:
        a, b = (part.strip() for part in spec.split(":", 1))
        try:
            if date.fromisoformat(a) > date.fromisoformat(b):
                raise ValueError(f"range {spec!r} ends before it starts")
        except ValueError as e:
            raise ValueError(
                str(e) if "ends before" in str(e)
                else f"range {spec!r} must be two YYYY-MM-DD dates"
            ) from None
        return a, b
    return parse_period(spec, anchor_end_iso)


# How far two segment boundaries may sit apart and still be the same seam.
# start_utc is stored truncated to the second while end_utc keeps fractions, so
# a perfectly contiguous pair reads as ~1s of overlap.
SEGMENT_SEAM_TOLERANCE_S = 2.0


def segment_chains(events, tolerance_s: float = SEGMENT_SEAM_TOLERANCE_S) -> list[list]:
    """Split segment/lap events into non-overlapping chains, best first.

    The watch records several *independent* partitions of one workout — the
    2026-07-26 run has a 5-split chain and a 4-split chain, each covering the
    whole 36.1 minutes. Concatenating them describes a session twice as long as
    the one that happened, with splits nested inside other splits.

    A chain is a run of splits that don't overlap: each event joins the chain
    whose current end it fits most tightly, and starts a new one if it overlaps
    every open chain. A gap (a pause) is not an overlap, so it stays in the same
    chain instead of starting a rival partition.

    Returns chains ordered by minutes covered, then by how finely they divide
    the session; `events` are mappings with start_utc/end_utc/duration_min.
    """
    def _dt(v):
        return datetime.fromisoformat(v.replace("Z", "+00:00")) if v else None

    chains: list[list] = []
    ends: list[datetime] = []
    for e in sorted(events, key=lambda e: (e["start_utc"], e["duration_min"] or 0)):
        start = _dt(e["start_utc"])
        end = _dt(e["end_utc"]) or (
            start + timedelta(minutes=float(e["duration_min"] or 0)))
        fits = [i for i, prev_end in enumerate(ends)
                if (start - prev_end).total_seconds() >= -tolerance_s]
        if fits:
            i = min(fits, key=lambda i: abs((start - ends[i]).total_seconds()))
            chains[i].append(e)
            ends[i] = max(ends[i], end)
        else:
            chains.append([e])
            ends.append(end)
    # Rival partitions of the same session cover the same minutes to within
    # float noise, so compare coverage at the stored precision and let the
    # finer-grained chain win the tie — more splits is more signal.
    return sorted(chains, reverse=True, key=lambda c: (
        round(sum(x["duration_min"] or 0 for x in c), 2), len(c)))


def downsample_weekly(dates, vals, how: str) -> list[dict]:
    """Collapse a daily series to Monday-anchored weeks using the metric's OWN
    aggregation.

    It used to take the mean regardless, so a cumulative metric came back as a
    daily average wearing a weekly label: step_count's 2026-07-20 week totalled
    55,376 and was reported as 7,910.89 under a payload that also said
    agg='sum'. Nothing downstream could tell the difference.

    'days' rides along on every point because the first and last week of a range
    are usually partial, and a partial week's SUM is not comparable to a full
    one's.
    """
    buckets: dict[str, list[float]] = {}
    for d, v in zip(dates, vals):
        day = date.fromisoformat(d)
        wk = (day - timedelta(days=day.weekday())).isoformat()
        buckets.setdefault(wk, []).append(v)
    if how == "sum":
        reduce = np.sum
    elif how == "last":
        # 'last' metrics (body_mass) store the day's final reading; the week's
        # is the week's final reading, not an average of the dailies.
        def reduce(vs):
            return vs[-1]
    else:
        reduce = np.mean
    return [{"date": k, "value": r(reduce(vs)), "days": len(vs)}
            for k, vs in sorted(buckets.items())]


# --- time in heart-rate zones ----------------------------------------------
# The watch samples heart rate irregularly (2-6 s during a workout, minutes
# apart at rest), so "minutes in a band" has to be duration-weighted: each
# sample owns the time until the next one. Capped, because the same arithmetic
# applied to a 20-minute hole in the samples would invent 20 minutes in
# whichever band happened to be last — the failure this tool exists to avoid.
HR_SAMPLE_MAX_GAP_S = 60.0


def parse_thresholds(spec) -> list[float]:
    """'135,150,155,170' (or a list) -> ascending band edges. Raises ValueError.

    Never falls back to a default set: a coach reading "12 minutes over 150"
    must be able to trust that 150 is the number that was asked for.
    """
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        parts = list(spec or [])
    try:
        vals = [float(p) for p in parts]
    except (TypeError, ValueError):
        raise ValueError(
            f"could not parse thresholds {spec!r}; use ascending bpm numbers, "
            "e.g. '135,150,155,170'") from None
    if not vals:
        raise ValueError("thresholds must name at least one bpm boundary, "
                         "e.g. '150'")
    if any(v <= 0 for v in vals):
        raise ValueError(f"thresholds must be positive bpm values, got {vals}")
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError(f"thresholds must be strictly ascending, got {vals}")
    return vals


def hr_bands(thresholds: list[float]) -> list[dict]:
    """Half-open bands (lower, upper] from ascending edges.

    Upper-inclusive on purpose: the plan says "at or under 150", so 150 itself
    is compliant and belongs to the band that ENDS at 150.
    """
    bands = []
    lower = None
    for t in thresholds:
        label = f"<={r(t, 0):g}" if lower is None else f">{r(lower, 0):g}-{r(t, 0):g}"
        bands.append({"label": label, "lower_exclusive": lower, "upper_inclusive": t})
        lower = t
    bands.append({"label": f">{r(lower, 0):g}", "lower_exclusive": lower,
                  "upper_inclusive": None})
    return bands


def zone_minutes(samples, window_end, thresholds,
                 max_gap_s: float = HR_SAMPLE_MAX_GAP_S) -> dict:
    """Minutes and sample counts per HR band over a window.

    `samples` is [(datetime, bpm)] in any order; `window_end` is the datetime
    the last sample's slice is truncated at. Each sample owns the interval to
    the next sample, clipped to `max_gap_s`. What the clip discards is reported
    as `uncovered_min` rather than absorbed, so a band's minutes are always a
    lower bound backed by real samples and the agent can see the shortfall.
    """
    edges = parse_thresholds(thresholds)
    bands = [dict(b, minutes=0.0, n_samples=0) for b in hr_bands(edges)]
    pts = sorted(samples, key=lambda s: s[0])
    covered = uncovered = 0.0
    for i, (ts, value) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else window_end
        span = max(0.0, (nxt - ts).total_seconds())
        slice_s = min(span, max_gap_s)
        covered += slice_s
        uncovered += span - slice_s
        idx = next((j for j, e in enumerate(edges) if value <= e), len(edges))
        bands[idx]["minutes"] += slice_s / 60.0
        bands[idx]["n_samples"] += 1
    n = len(pts)
    for b in bands:
        b["minutes"] = r(b["minutes"], 2)
        b["pct_samples"] = r(b["n_samples"] / n * 100, 1) if n else None
    vals = [v for _, v in pts]
    above = []
    for e in edges:
        mins = sum(b["minutes"] for b in bands
                   if b["lower_exclusive"] is not None and b["lower_exclusive"] >= e)
        cnt = sum(b["n_samples"] for b in bands
                  if b["lower_exclusive"] is not None and b["lower_exclusive"] >= e)
        above.append({"threshold": r(e, 0), "minutes": r(mins, 2),
                      "n_samples": cnt,
                      "pct_samples": r(cnt / n * 100, 1) if n else None})
    return {
        "n_samples": n,
        "avg_heart_rate": r(float(np.mean(vals)), 0) if vals else None,
        "min_heart_rate": r(min(vals), 0) if vals else None,
        "max_heart_rate": r(max(vals), 0) if vals else None,
        "covered_min": r(covered / 60.0, 2),
        "uncovered_min": r(uncovered / 60.0, 2),
        "bands": bands,
        "above": above,
    }


def series(conn, metric, start_iso, end_iso):
    col = value_col(metric)
    q = (f"SELECT date, {col} AS v, unit FROM daily_metrics "
         f"WHERE metric = ? AND date BETWEEN ? AND ? AND {col} IS NOT NULL ORDER BY date")
    rows = conn.execute(q, (metric, start_iso or "0000-01-01", end_iso)).fetchall()
    dates = [row["date"] for row in rows]
    vals = [row["v"] for row in rows]
    unit = rows[0]["unit"] if rows else nz.canonical_unit(metric, None)
    return dates, vals, unit


def stats(dates, vals):
    if not vals:
        return {"n_days": 0}
    arr = np.array(vals, dtype=float)
    imin, imax = int(arr.argmin()), int(arr.argmax())
    return {
        "n_days": len(vals),
        "start": dates[0], "end": dates[-1],
        "mean": r(arr.mean()), "median": r(np.median(arr)),
        "min": r(arr.min()), "min_date": dates[imin],
        "max": r(arr.max()), "max_date": dates[imax],
        "std": r(arr.std()),
    }


def slope_per_week(dates, vals):
    """Linear slope per week over a daily series; None if <3 points."""
    if len(vals) < 3:
        return None
    d0 = date.fromisoformat(dates[0])
    x = np.array([(date.fromisoformat(d) - d0).days for d in dates], dtype=float)
    slope = float(np.polyfit(x, np.array(vals, dtype=float), 1)[0])
    return slope * 7


def baseline(vals, exclude_recent=1, window=28):
    """Robust baseline (median) over a trailing window, excluding the most recent
    `exclude_recent` points. None if not enough history remains."""
    if exclude_recent:
        hist = vals[:-exclude_recent]
    else:
        hist = list(vals)
    hist = hist[-window:]
    if not hist:
        return None
    return float(np.median(np.array(hist, dtype=float)))


def pct_change(recent, base):
    if base in (None, 0):
        return None
    return r((recent - base) / base * 100)
