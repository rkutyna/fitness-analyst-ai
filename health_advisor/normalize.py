"""Canonical metric vocabulary + mappings from BOTH ingestion paths.

This is the single most important correctness surface in the project: the
backfill sees HealthKit identifiers (HKQuantityTypeIdentifierStepCount) and the
Health Auto Export app sends snake_case (step_count). Both must land on ONE
canonical name with ONE unit so the schema, agent and dashboard stay coherent.

Everything that maps a name or a unit lives here and nowhere else.

Canonical unit policy (chosen to match the Apple export so the 2.3M-row
backfill needs no numeric conversion):
  - distance (records): miles ('mi')      - mass: pounds ('lb')
  - energy: kilocalories ('kcal')         - temperature: degF
  - workouts.distance_mi is miles too — there is no exception. (This line used
    to claim a km column; the schema has never had one.)

Incoming Health Auto Export values are converted INTO these canonical units by
convert_unit()/unit_converter() below — genuinely converted, with the arithmetic
in UNIT_FAMILIES. A unit that cannot be placed raises UnitError and the points
are dropped and logged rather than stored under a label that is not theirs.
(Until 2026-07-31 this paragraph was a claim rather than a description: the only
"conversion" was Apple's Cal -> kcal rename at factor 1.0, and everything else
was relabeled with the number untouched. See audit P3-5.)
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Iterable, Optional

# --------------------------------------------------------------------------- #
# Datetime parsing — Apple export format: '2024-01-15 08:30:00 -0400'
# --------------------------------------------------------------------------- #
_APPLE_DT = "%Y-%m-%d %H:%M:%S %z"


def parse_apple_datetime(s: str) -> datetime:
    """Parse Apple's space-separated offset datetime into an aware datetime."""
    return datetime.strptime(s.strip(), _APPLE_DT)


def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def local_date_of(dt: datetime) -> str:
    """Local calendar day (YYYY-MM-DD) in the timestamp's own offset."""
    return dt.strftime("%Y-%m-%d")


def local_naive(dt: datetime) -> str:
    """Wall-clock local time as naive 'YYYY-MM-DD HH:MM:SS' (offset dropped).
    Stored so SQLite strftime('%H', start_local) yields the true LOCAL hour for
    intraday bucketing without any timezone normalization."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
# Metric catalog: canonical_name -> (unit, agg, group, display)
#   agg: how the metric should be summarized over a day
#        'sum'  cumulative (steps, energy, distance, minutes)
#        'mean' instantaneous sample (heart rate, respiratory rate)
#        'last' point-in-time state (body mass, vo2 max)
# Only metrics that need a fixed unit/agg are listed; unmapped quantity types
# are auto-named and default to ('<native unit>', 'mean').
# --------------------------------------------------------------------------- #
CATALOG: dict[str, dict] = {
    # activity / energy
    "step_count":                 {"unit": "count",      "agg": "sum",  "group": "activity"},
    "distance_walking_running":   {"unit": "mi",         "agg": "sum",  "group": "activity"},
    "distance_cycling":           {"unit": "mi",         "agg": "sum",  "group": "activity"},
    "flights_climbed":            {"unit": "count",      "agg": "sum",  "group": "activity"},
    "active_energy":              {"unit": "kcal",       "agg": "sum",  "group": "activity"},
    "basal_energy":               {"unit": "kcal",       "agg": "sum",  "group": "activity"},
    "apple_exercise_time":        {"unit": "min",        "agg": "sum",  "group": "activity"},
    "apple_stand_time":           {"unit": "min",        "agg": "sum",  "group": "activity"},
    "time_in_daylight":           {"unit": "min",        "agg": "sum",  "group": "activity"},
    "physical_effort":            {"unit": "kcal/hr·kg", "agg": "mean", "group": "activity"},
    # heart
    "heart_rate":                 {"unit": "count/min",  "agg": "mean", "group": "heart"},
    # 'last', not 'mean' — REVISED 2026-08-17 (F6-3, audit part 6), superseding
    # the 2026-08-01 choice below.
    #
    # These two are not multi-sample series. Apple emits each as same-day
    # REVISIONS of a single statistic: every record for a day shares one
    # start_utc and carries a progressively later end_utc, the last being the
    # settled value. So averaging them averages the DRAFTS. Measured over the
    # plan window the stored value disagreed with Apple's settled value on 27 of
    # 54 days, by up to 5.33 bpm (resting) and 10.83 (walking), and two of the
    # three "plan-era lows" that the weekly-mean reporting rule was written to correct
    # were artifacts of it — 07-11's 58.2 is really 63.0 — while the one real
    # low, 08-02's 58.0, was stored as 60.0 and never noticed.
    #
    # It also inflated the noise floor, which decides what the plan can detect
    # at all: day-to-day SD measured from consecutive-day differences is 4.04 on
    # the settled series and 4.79 on the draft means, reproducing the audit's
    # published 4.01 vs 4.75. On a weekly mean that is an MDC of 4.7 vs 5.5 bpm
    # against an expected training effect of 0-3.
    #
    # Scope is exactly these two metrics: the signature (one distinct start_utc,
    # n distinct end_utc) was tested across every 'mean' metric and found in
    # resting_heart_rate 45/46 days, walking_heart_rate_average 38/38, and 0 of
    # 25 others — heart_rate, heart_rate_variability and respiratory_rate are
    # genuine multi-sample series where 'mean' is correct.
    #
    # No migration and no re-derive: daily_metrics.last is already populated for
    # all 1,153 / 965 days with zero nulls, so this only re-routes value_col().
    #
    # The superseded 2026-08-01 note, kept because the reasoning was sound given
    # what was known then: "'mean', not 'last': the watch records several
    # resting-heart-rate readings a day and the daily average is the more stable
    # of the two ... It also ends a real inconsistency — the deep dive was
    # quoting the daily average while the briefing quoted `last`." The
    # inconsistency was real; the resolution picked the wrong side of it,
    # because nobody had yet noticed the readings were revisions rather than
    # samples.
    "resting_heart_rate":         {"unit": "count/min",  "agg": "last", "group": "heart"},
    "walking_heart_rate_average": {"unit": "count/min",  "agg": "last", "group": "heart"},
    "heart_rate_variability":     {"unit": "ms",         "agg": "mean", "group": "heart"},
    "vo2_max":                    {"unit": "mL/min·kg",  "agg": "last", "group": "heart"},
    # respiratory / vitals
    "respiratory_rate":           {"unit": "count/min",  "agg": "mean", "group": "vitals"},
    "blood_oxygen_saturation":    {"unit": "%",          "agg": "mean", "group": "vitals"},
    "body_mass":                  {"unit": "lb",         "agg": "last", "group": "body"},
    "body_mass_index":            {"unit": "count",      "agg": "last", "group": "body"},
    "height":                     {"unit": "ft",         "agg": "last", "group": "body"},
    "sleeping_wrist_temperature": {"unit": "degF",       "agg": "mean", "group": "vitals"},
    # mobility
    "walking_speed":              {"unit": "mi/hr",      "agg": "mean", "group": "mobility"},
    "running_speed":              {"unit": "mi/hr",      "agg": "mean", "group": "mobility"},
    "walking_step_length":        {"unit": "in",         "agg": "mean", "group": "mobility"},
    "walking_steadiness":         {"unit": "%",          "agg": "last", "group": "mobility"},
    "stair_ascent_speed":         {"unit": "ft/s",       "agg": "mean", "group": "mobility"},
    "stair_descent_speed":        {"unit": "ft/s",       "agg": "mean", "group": "mobility"},
    # audio
    "environmental_audio_exposure": {"unit": "dBASPL",   "agg": "mean", "group": "audio"},
    "headphone_audio_exposure":     {"unit": "dBASPL",   "agg": "mean", "group": "audio"},
    # sleep (derived to minutes; attributed to wake day) — see SLEEP_VALUE_MAP
    "sleep_in_bed":               {"unit": "min",        "agg": "sum",  "group": "sleep"},
    "sleep_asleep":               {"unit": "min",        "agg": "sum",  "group": "sleep"},
    "sleep_awake":                {"unit": "min",        "agg": "sum",  "group": "sleep"},
    "sleep_rem":                  {"unit": "min",        "agg": "sum",  "group": "sleep"},
    "sleep_deep":                 {"unit": "min",        "agg": "sum",  "group": "sleep"},
    "sleep_core":                 {"unit": "min",        "agg": "sum",  "group": "sleep"},
    # derived (health_advisor/derive.py) — computed from interval records.
    # bedtime/midpoint: hours since PREVIOUS-day noon (22:30 -> 10.5, 00:30 -> 12.5,
    # continuous across midnight); wake_time: hours since midnight of the wake day.
    "sleep_bedtime":       {"unit": "h",     "agg": "mean", "group": "sleep_timing"},
    "sleep_wake_time":     {"unit": "h",     "agg": "mean", "group": "sleep_timing"},
    "sleep_midpoint":      {"unit": "h",     "agg": "mean", "group": "sleep_timing"},
    "sleep_midpoint_sd_28d": {"unit": "h", "agg": "mean", "group": "sleep_timing"},
    "sleep_timing_interval_regularity": {"unit": "%", "agg": "mean", "group": "sleep_timing"},
    "sleep_time_in_bed":   {"unit": "min",   "agg": "mean", "group": "sleep_timing"},
    "sleep_awakenings":    {"unit": "count", "agg": "mean", "group": "sleep_timing"},
    "sleep_awake_longest": {"unit": "min",   "agg": "mean", "group": "sleep_timing"},
    "sleep_latency":       {"unit": "min",   "agg": "mean", "group": "sleep_timing"},
    "wear_hours":          {"unit": "h",     "agg": "mean", "group": "coverage"},
    "hr_load_proxy":       {"unit": "au",    "agg": "sum",  "group": "training"},
    # The plan's own dial, written nightly by derive.py. Both are DERIVED: the
    # row IS the day's value, not a sample to be averaged, so agg is "last".
    # Catalogued because an uncatalogued metric falls through as unmanaged with
    # agg "mean" and no unit, which silently breaks metrics.value_col(), the MCP
    # metadata and every correlation that reads them. E8-8.
    "jog_minutes":         {"unit": "min",   "agg": "last", "group": "training"},
    "longest_block_min":   {"unit": "min",   "agg": "last", "group": "training"},
    # mindfulness / events
    "mindful_minutes":            {"unit": "min",        "agg": "sum",  "group": "mind"},
    "stand_hour":                 {"unit": "count",      "agg": "sum",  "group": "activity"},
    # heart / recovery (watch-derived nightly or post-workout -> wear-filtered groups)
    "cardio_recovery":            {"unit": "count/min",  "agg": "mean", "group": "heart"},
    "breathing_disturbances":     {"unit": "count",      "agg": "mean", "group": "sleep"},
    "apple_sleeping_breathing_disturbances":
                                  {"unit": "count",      "agg": "mean", "group": "sleep"},
    # body composition (scale readings: point-in-time states)
    # RENPHO-only, and on this device information-free (F5-2, audit part 5).
    # Measured over the plan window: body_fat% = -20.404 + 0.2175 * lb with
    # R^2 = 0.991 and residual SD 0.032 pp, and lean_body_mass is identically
    # body_mass * (1 - body_fat/100) to within 0.011 lb. Every pound lost is
    # reported as 0.376 lb of lean mass BY CONSTRUCTION, not by measurement.
    # The weight-adjusted body-fat trend is -0.0011 pp/week, CI [-0.0066,
    # +0.0044] — nothing beyond the scale weight. Kept and ingested because the
    # readings are real; caveated so the next consumer does not have to
    # rediscover that they carry no independent information.
    "body_fat_percentage":        {"unit": "%",          "agg": "last", "group": "body",
                                   "caveat": "derived from scale weight on this "
                                             "device (R^2 0.991); not an independent "
                                             "measurement"},
    "lean_body_mass":             {"unit": "lb",         "agg": "last", "group": "body",
                                   "caveat": "identically body_mass * (1 - "
                                             "body_fat/100); carries no information "
                                             "beyond body_mass"},
    # workout dynamics — deliberately NOT in a wear-filtered group: these exist
    # only when the watch was worn during the session, so all-day wear_hours is
    # the wrong exclusion signal for them.
    "running_power":              {"unit": "W",          "agg": "mean", "group": "workout"},
    "running_ground_contact_time":{"unit": "ms",         "agg": "mean", "group": "workout"},
    "running_stride_length":      {"unit": "m",          "agg": "mean", "group": "workout"},
    "running_vertical_oscillation":{"unit": "cm",        "agg": "mean", "group": "workout"},
    "cycling_cadence":            {"unit": "count/min",  "agg": "mean", "group": "workout"},
    "cycling_power":              {"unit": "W",          "agg": "mean", "group": "workout"},
    "cycling_speed":              {"unit": "mi/hr",      "agg": "mean", "group": "workout"},
    # mobility (cont.)
    "walking_asymmetry_percentage":
                                  {"unit": "%",          "agg": "mean", "group": "mobility"},
    "walking_double_support_percentage":
                                  {"unit": "%",          "agg": "mean", "group": "mobility"},
    "number_of_times_fallen":     {"unit": "count",      "agg": "sum",  "group": "mobility"},
    # 6-minute walk test: two names from the two ingest eras (backfill auto-derived
    # vs Health Auto Export). Cataloged separately; unifying them needs a records
    # migration — see ledger follow-ups.
    # Since #143 the HealthKit identifier is mapped EXPLICITLY to the
    # ...walking... name, so no new row lands under the auto-derived one. The
    # entry below stays for the 58 rows written 2020-09-29..2022-06-30, which
    # only a migration can move.
    "six_minute_walk_test_distance":
                                  {"unit": "m",          "agg": "last", "group": "mobility"},
    "six_minute_walking_test_distance":
                                  {"unit": "m",          "agg": "last", "group": "mobility"},
    # audio (cont.)
    "environmental_sound_reduction":
                                  {"unit": "dBASPL",     "agg": "mean", "group": "audio"},
    # nutrition (sparse 2019-2022 logging; daily intake totals)
    "dietary_energy_consumed":    {"unit": "kcal",       "agg": "sum",  "group": "nutrition"},
    "dietary_carbohydrates":      {"unit": "g",          "agg": "sum",  "group": "nutrition"},
    "dietary_protein":            {"unit": "g",          "agg": "sum",  "group": "nutrition"},
    "dietary_fat_total":          {"unit": "g",          "agg": "sum",  "group": "nutrition"},
    "dietary_fat_saturated":      {"unit": "g",          "agg": "sum",  "group": "nutrition"},
    "dietary_fat_monounsaturated":{"unit": "g",          "agg": "sum",  "group": "nutrition"},
    "dietary_fat_polyunsaturated":{"unit": "g",          "agg": "sum",  "group": "nutrition"},
    "dietary_fiber":              {"unit": "g",          "agg": "sum",  "group": "nutrition"},
    "dietary_sugar":              {"unit": "g",          "agg": "sum",  "group": "nutrition"},
    "dietary_sodium":             {"unit": "mg",         "agg": "sum",  "group": "nutrition"},
    "dietary_potassium":          {"unit": "mg",         "agg": "sum",  "group": "nutrition"},
    "dietary_calcium":            {"unit": "mg",         "agg": "sum",  "group": "nutrition"},
    "dietary_iron":               {"unit": "mg",         "agg": "sum",  "group": "nutrition"},
    "dietary_cholesterol":        {"unit": "mg",         "agg": "sum",  "group": "nutrition"},
    "dietary_vitamin_c":          {"unit": "mg",         "agg": "sum",  "group": "nutrition"},
    "dietary_water":              {"unit": "mL",         "agg": "sum",  "group": "nutrition"},
    # subjective check-in (nightly Telegram conversation -> subjective table,
    # mirrored to records/daily_metrics with source/origin 'checkin').
    # Ratings are 1-5 integers; drinks are counts. Not watch-derived, so the
    # correlation wear filter must NOT apply (groups deliberately outside
    # correlate.WATCH_GROUPS).
    "subjective_stress":          {"unit": "score",      "agg": "mean", "group": "subjective"},
    "subjective_soreness":        {"unit": "score",      "agg": "mean", "group": "subjective"},
    "subjective_energy":          {"unit": "score",      "agg": "mean", "group": "subjective"},
    "subjective_sleep_quality":   {"unit": "score",      "agg": "mean", "group": "subjective"},
    "caffeine_drinks":            {"unit": "drinks",     "agg": "sum",  "group": "intake"},
    "alcohol_drinks":             {"unit": "drinks",     "agg": "sum",  "group": "intake"},
    # Apple workout effort score (1-10, rated on the watch post-session; the
    # Estimated variant is auto-derived). Not seen in the snapshot yet
    # (2026-07-16 scan: zero records in export.zip + receiver payloads) —
    # mapped now so it flows the moment HAE starts sending it.
    "workout_effort":             {"unit": "score",      "agg": "mean", "group": "workout"},
}

# --------------------------------------------------------------------------- #
# HealthKit QUANTITY identifier -> canonical name (explicit for the messy ones)
# --------------------------------------------------------------------------- #
HK_QUANTITY: dict[str, str] = {
    "HKQuantityTypeIdentifierStepCount": "step_count",
    "HKQuantityTypeIdentifierDistanceWalkingRunning": "distance_walking_running",
    "HKQuantityTypeIdentifierDistanceCycling": "distance_cycling",
    "HKQuantityTypeIdentifierFlightsClimbed": "flights_climbed",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy",
    "HKQuantityTypeIdentifierBasalEnergyBurned": "basal_energy",
    "HKQuantityTypeIdentifierAppleExerciseTime": "apple_exercise_time",
    "HKQuantityTypeIdentifierAppleStandTime": "apple_stand_time",
    "HKQuantityTypeIdentifierTimeInDaylight": "time_in_daylight",
    "HKQuantityTypeIdentifierPhysicalEffort": "physical_effort",
    "HKQuantityTypeIdentifierHeartRate": "heart_rate",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_heart_rate",
    "HKQuantityTypeIdentifierWalkingHeartRateAverage": "walking_heart_rate_average",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "heart_rate_variability",
    "HKQuantityTypeIdentifierVO2Max": "vo2_max",
    "HKQuantityTypeIdentifierRespiratoryRate": "respiratory_rate",
    "HKQuantityTypeIdentifierOxygenSaturation": "blood_oxygen_saturation",
    "HKQuantityTypeIdentifierBodyMass": "body_mass",
    "HKQuantityTypeIdentifierBodyMassIndex": "body_mass_index",
    "HKQuantityTypeIdentifierHeight": "height",
    "HKQuantityTypeIdentifierAppleSleepingWristTemperature": "sleeping_wrist_temperature",
    "HKQuantityTypeIdentifierWalkingSpeed": "walking_speed",
    "HKQuantityTypeIdentifierRunningSpeed": "running_speed",
    "HKQuantityTypeIdentifierWalkingStepLength": "walking_step_length",
    "HKQuantityTypeIdentifierAppleWalkingSteadiness": "walking_steadiness",
    "HKQuantityTypeIdentifierEnvironmentalAudioExposure": "environmental_audio_exposure",
    "HKQuantityTypeIdentifierHeadphoneAudioExposure": "headphone_audio_exposure",
    "HKQuantityTypeIdentifierWorkoutEffortScore": "workout_effort",
    "HKQuantityTypeIdentifierEstimatedWorkoutEffortScore": "workout_effort",
    # ----------------------------------------------------------------- #
    # Added 2026-08-27 for #143. Every one of these was delivered by the
    # retired Health Auto Export path and requested by the iOS client — but
    # `hk_parse._type_info` resolves ONLY through this dict and has no
    # auto-derive fallback, so an unmapped identifier is not stored under a
    # second name: the sample is dropped into `unhandled`. Two of these
    # (BodyFatPercentage, RunningPower) were already in the client's request
    # list and were being discarded on arrival — which is exactly why the
    # first HealthKit week showed 17 distinct `records` metrics rather than
    # the 19 the client asked for.
    #
    # Names are the ones the existing rows already use, NOT the ones
    # `_camel_to_snake` would derive. Three would have forked a live series:
    "HKQuantityTypeIdentifierBodyFatPercentage": "body_fat_percentage",
    "HKQuantityTypeIdentifierLeanBodyMass": "lean_body_mass",
    "HKQuantityTypeIdentifierRunningPower": "running_power",
    "HKQuantityTypeIdentifierRunningStrideLength": "running_stride_length",
    "HKQuantityTypeIdentifierRunningVerticalOscillation": "running_vertical_oscillation",
    "HKQuantityTypeIdentifierRunningGroundContactTime": "running_ground_contact_time",
    "HKQuantityTypeIdentifierWalkingAsymmetryPercentage": "walking_asymmetry_percentage",
    "HKQuantityTypeIdentifierWalkingDoubleSupportPercentage":
        "walking_double_support_percentage",
    "HKQuantityTypeIdentifierStairAscentSpeed": "stair_ascent_speed",
    "HKQuantityTypeIdentifierStairDescentSpeed": "stair_descent_speed",
    # …and these three are the forks. Auto-derivation would have produced
    # "heart_rate_recovery_one_minute" (0 rows), "apple_sleeping_breathing_
    # disturbances" (1 row) and "six_minute_walk_test_distance" (last written
    # 2022-06-30) beside the series the coach actually reads.
    "HKQuantityTypeIdentifierHeartRateRecoveryOneMinute": "cardio_recovery",
    "HKQuantityTypeIdentifierAppleSleepingBreathingDisturbances":
        "breathing_disturbances",
    "HKQuantityTypeIdentifierSixMinuteWalkTestDistance":
        "six_minute_walking_test_distance",
}

# Units that the export reports per-metric but that we relabel/convert to canonical.
# value_factor multiplies the raw value; unit is the canonical unit label.
UNIT_RELABEL: dict[str, tuple[float, str]] = {
    # Apple "Cal" == kilocalorie; keep the number, relabel the unit.
    "Cal": (1.0, "kcal"),
}

# --------------------------------------------------------------------------- #
# Sleep category value -> list of canonical metrics it contributes to.
# 'Asleep*' all feed sleep_asleep (total) AND, when staged, the stage metric.
# --------------------------------------------------------------------------- #
HK_SLEEP_TYPE_IDENTIFIER = "HKCategoryTypeIdentifierSleepAnalysis"

SLEEP_VALUE_MAP: dict[str, list[str]] = {
    "HKCategoryValueSleepAnalysisInBed": ["sleep_in_bed"],
    "HKCategoryValueSleepAnalysisAsleepUnspecified": ["sleep_asleep"],
    "HKCategoryValueSleepAnalysisAsleepCore": ["sleep_asleep", "sleep_core"],
    "HKCategoryValueSleepAnalysisAsleepDeep": ["sleep_asleep", "sleep_deep"],
    "HKCategoryValueSleepAnalysisAsleepREM": ["sleep_asleep", "sleep_rem"],
    "HKCategoryValueSleepAnalysisAwake": ["sleep_awake"],
}

# Other category types we keep. 'flag' = 1 when value in positive-set else 0;
# 'duration' = minutes between start/end; 'count' = 1 per event.
HK_CATEGORY: dict[str, dict] = {
    "HKCategoryTypeIdentifierMindfulSession":
        {"metric": "mindful_minutes", "mode": "duration"},
    "HKCategoryTypeIdentifierAppleStandHour":
        {"metric": "stand_hour", "mode": "flag",
         "positive": {"HKCategoryValueAppleStandHourStood"}},
}

# --------------------------------------------------------------------------- #
# Workout activity type -> canonical label
# --------------------------------------------------------------------------- #
def workout_label(activity_type: str) -> str:
    return _camel_to_snake(activity_type.replace("HKWorkoutActivityType", ""))


# Health Auto Export sends human display names ("Outdoor Run", "Indoor Cycle")
# rather than HK activity types. Map them onto the SAME canonical vocabulary the
# backfill produced from HK types (running/walking/cycling/...), so type-based
# queries and workout_mix don't fragment across the two ingestion paths.
_WORKOUT_QUALIFIERS = ("outdoor", "indoor", "pool", "open water")
_WORKOUT_KEYWORDS = (
    ("run", "running"), ("walk", "walking"), ("hik", "hiking"),
    ("cycl", "cycling"), ("bik", "cycling"), ("swim", "swimming"),
    ("row", "rowing"), ("elliptical", "elliptical"),
)


def workout_name_to_canonical(name: str) -> str:
    """Canonical workout type from an HAE display name or HK type."""
    raw = (name or "").strip()
    if not raw:
        return "other"
    if raw.startswith("HKWorkoutActivityType"):
        return workout_label(raw)
    s = raw.lower()
    for q in _WORKOUT_QUALIFIERS:
        if s.startswith(q + " "):
            s = s[len(q) + 1:]
            break
    for kw, canon in _WORKOUT_KEYWORDS:
        if kw in s:
            return canon
    return _camel_to_snake(s).replace(" ", "_").strip("_")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_camel_re1 = re.compile(r"(.)([A-Z][a-z]+)")
_camel_re2 = re.compile(r"([a-z0-9])([A-Z])")


def _camel_to_snake(name: str) -> str:
    s = _camel_re1.sub(r"\1_\2", name)
    s = _camel_re2.sub(r"\1_\2", s)
    return s.lower()


def hk_quantity_to_canonical(hk_type: str) -> str:
    """Explicit map first; otherwise auto-derive a snake_case name."""
    if hk_type in HK_QUANTITY:
        return HK_QUANTITY[hk_type]
    return _camel_to_snake(hk_type.replace("HKQuantityTypeIdentifier", ""))


def canonical_unit(metric: str, raw_unit: Optional[str]) -> str:
    if metric in CATALOG:
        return CATALOG[metric]["unit"]
    return raw_unit or ""


def agg_for(metric: str) -> str:
    return CATALOG.get(metric, {}).get("agg", "mean")


# --------------------------------------------------------------------------- #
# Cross-source arbitration for cumulative metrics (audit P0-2).
#
# Apple Health resolves overlapping cumulative samples by source before showing
# a daily total. This pipeline sums whatever it was given, so a day written by
# two sources counted the same movement twice. The whole-day offenders below
# were found in the archive by comparing per-source daily totals against
# physiology; workout-window device priority is a separate, date-gated rule.
# --------------------------------------------------------------------------- #

# Apps that mirror a whole-day total into HealthKit next to the Apple devices'
# own samples, so both describe the same day. The date is when Apple's record
# becomes the better one — before it, the third party is what was actually worn
# (the phone alone badly undercounts) and it wins instead. The archive's first
# Apple Watch sample is 2019-01-16.
MIRROR_SOURCES = {"Sync Solver": "2019-01-16"}

# HealthKit-direct ingestion began on this date. Unlike MIRROR_SOURCES, this
# is not a whole-day source precedence rule: concurrent samples inside one
# workout window need an activity-device winner. Device roles are resolved from
# the complete source label at query time; a pipe-joined label is deliberately
# not split.
# Deployment default carried over from the first deployment; making this a
# per-vault setting is tracked in issue #6.
WORKOUT_SOURCE_ARBITRATION_FROM = "2026-08-21"

# The largest value one honest sample of a metric carries. Above this the row
# is a whole-day estimate wearing a sample's clothes: RENPHO's scale writes its
# BMR figure at weigh-in as a handful of ~370 kcal points, while nine years of
# Apple basal samples never exceed 81.1 kcal. Health Auto Export sometimes
# labels those points with the watch's own name ("<name>'s Apple Watch|RENPHO
# Health"), so source alone cannot separate them — the magnitude can.
# Only metrics with a demonstrated offender are listed; the rest are unbounded.
SAMPLE_CEILING = {"basal_energy": 150.0}


def is_mirror_source(source: str | None) -> bool:
    return any(m in (source or "") for m in MIRROR_SOURCES)


def mirror_loses_from(source: str | None) -> str | None:
    """The date from which this mirror source stops being the better record."""
    for name, cutoff in MIRROR_SOURCES.items():
        if name in (source or ""):
            return cutoff
    return None


WORKOUT_SOURCE_ROLES = frozenset({"watch", "iphone", "scale", "mirror", "gymkit"})


def workout_source_role(source: str | None, *, product_type: str | None = None) -> str | None:
    """Resolve a complete source label to its stable device role.

    Source names are user-editable and localized, so role words are only a
    fallback for exports that omit HealthKit's product type. A merged or empty
    label has no single device identity and returns ``None``.
    """
    label = " ".join((source or "").replace("\xa0", " ").split())
    if "|" in label:
        return None
    if label == "GymKit":
        return "gymkit"

    if is_mirror_source(label):
        return "mirror"

    product = " ".join((product_type or "").replace("\xa0", " ").split()).casefold()
    compact_product = re.sub(r"[^\w]+", " ", product).strip()
    if re.search(r"(?:^| )(?:watch|applewatch)(?: |$)", compact_product):
        return "watch"
    if re.search(r"(?:^| )(?:iphone|phone|mobile)(?: |$)", compact_product):
        return "iphone"

    words = re.sub(r"[^\w]+", " ", label.casefold()).split()
    if any(word in words for word in (
        "watch", "wrist", "reloj", "uhr", "montre", "relógio", "relogio",
    )):
        return "watch"
    if any(word in words for word in (
        "iphone", "phone", "mobile", "smartphone", "telefon", "teléfono",
        "telefono", "téléphone",
    )):
        return "iphone"
    if any(word in words for word in (
        "scale", "balance", "waage", "báscula", "bascula", "balanza",
    )):
        return "scale"
    return None


def workout_source_winners(sources: set[str]) -> set[str]:
    """Return source labels authoritative within one workout window."""
    roles = {workout_source_role(source): source for source in sources}
    if "gymkit" in roles:
        return {source for source in sources
                if workout_source_role(source) == "gymkit"}
    if "watch" in roles and "iphone" in roles:
        return {source for source in sources
                if workout_source_role(source) == "watch"}
    return set()


def apply_unit(raw_value: float, raw_unit: Optional[str]) -> tuple[float, Optional[str]]:
    """Relabel/convert known units (e.g. Cal -> kcal). Returns (value, unit).

    Legacy, still used by the backfill path where the export's units already ARE
    the canonical ones. New code should use unit_converter()/to_canonical(),
    which knows the target metric and therefore can actually convert.
    """
    if raw_unit in UNIT_RELABEL:
        factor, unit = UNIT_RELABEL[raw_unit]
        return raw_value * factor, unit
    return raw_value, raw_unit


# --------------------------------------------------------------------------- #
# Real unit conversion (audit P3-5).
#
# The header of this module has always claimed that incoming values are
# "converted INTO these canonical units". They were not. `apply_unit` handled
# exactly one case (Apple's "Cal" -> "kcal", a rename at factor 1.0) and
# `canonical_unit` then stamped the catalog's label onto whatever number had
# arrived. km in, "mi" out, digits unchanged.
#
# Each convertible unit is (family, factor-to-base, offset-to-base), so a value
# in base units is `v * factor + offset` and the inverse is exact. The offset
# exists for exactly one family — temperature is affine, not proportional, and
# treating degC->degF as a scale factor gets 37C wrong by 34 degrees.
# --------------------------------------------------------------------------- #
class UnitError(ValueError):
    """A unit we cannot place, or a conversion between different families.

    Raised rather than guessed. A number whose unit we do not understand is
    worse than no number: it is indistinguishable from a good one once stored,
    and every trend, correlation and threshold downstream will believe it.
    """


# family -> {unit: (factor_to_base, offset_to_base)}
UNIT_FAMILIES: dict[str, dict[str, tuple[float, float]]] = {
    "length": {                                    # base: metre
        "m": (1.0, 0.0), "km": (1000.0, 0.0), "cm": (0.01, 0.0), "mm": (0.001, 0.0),
        "in": (0.0254, 0.0), "ft": (0.3048, 0.0), "yd": (0.9144, 0.0),
        "mi": (1609.344, 0.0),
    },
    "mass": {                                      # base: kilogram
        "kg": (1.0, 0.0), "g": (0.001, 0.0), "mg": (1e-6, 0.0),
        "lb": (0.45359237, 0.0), "oz": (0.028349523125, 0.0),
        "st": (6.35029318, 0.0),
    },
    "energy": {                                    # base: kilocalorie
        # Apple writes "Cal" for the kilocalorie (the dietary Calorie); HealthKit's
        # lowercase "cal" is the small calorie, 1/1000 of it. The two are one
        # keystroke and three orders of magnitude apart, so both are explicit.
        "kcal": (1.0, 0.0), "Cal": (1.0, 0.0), "cal": (0.001, 0.0),
        "kJ": (0.23900573613767, 0.0), "J": (0.00023900573613767, 0.0),
    },
    "temperature": {                               # base: degree Celsius
        "degC": (1.0, 0.0), "degF": (5.0 / 9.0, -160.0 / 9.0),
    },
    "speed": {                                     # base: metre/second
        "m/s": (1.0, 0.0), "km/hr": (1000.0 / 3600.0, 0.0), "kph": (1000.0 / 3600.0, 0.0),
        "mi/hr": (0.44704, 0.0), "mph": (0.44704, 0.0), "ft/s": (0.3048, 0.0),
    },
    "time": {                                      # base: minute
        "min": (1.0, 0.0), "s": (1.0 / 60.0, 0.0), "sec": (1.0 / 60.0, 0.0),
        "ms": (1.0 / 60000.0, 0.0), "hr": (60.0, 0.0), "h": (60.0, 0.0),
        "d": (1440.0, 0.0),
    },
    "volume": {                                    # base: millilitre
        "mL": (1.0, 0.0), "L": (1000.0, 0.0),
        "fl_oz": (29.5735295625, 0.0), "cup": (236.5882365, 0.0),
    },
    "power": {                                     # base: watt
        "W": (1.0, 0.0), "kW": (1000.0, 0.0), "hp": (745.6998715822702, 0.0),
    },
    "frequency": {                                 # base: count/minute
        "count/min": (1.0, 0.0), "bpm": (1.0, 0.0), "count/s": (60.0, 0.0),
    },
}

_UNIT_INDEX: dict[str, tuple[str, float, float]] = {
    u: (fam, f, o) for fam, units in UNIT_FAMILIES.items() for u, (f, o) in units.items()
}

# Units with no dimension to convert along, or composite rates that only ever
# arrive in one spelling. An identity conversion is the only valid one for these;
# anything else is a genuine error, not a missing table entry.
DIMENSIONLESS_UNITS: frozenset[str] = frozenset({
    "", "count", "%", "score", "drinks", "dBASPL",
    # au is the arbitrary-unit marker for model outputs that must never be
    # compared against a published figure.
    "au",
    "kcal/hr·kg", "mL/min·kg",
})


# --- Spelling tolerance -------------------------------------------------------
# 2026-07-31 to 2026-08-16, sixty consecutive receiver batches dropped every
# vo2_max reading: the source started writing "ml/(kg·min)" where the table held
# "mL/min·kg". Same unit, different spelling. Rather than add the one variant and
# wait for the next one, resolution falls back through progressively looser forms.
#
# Case is folded LAST and only where it is unambiguous. "Cal" (kilocalorie) and
# "cal" (small calorie) are one keystroke and three orders of magnitude apart —
# see the energy family above — so any folded key covering two different
# conversions is dropped from the folded index and stays an error. Guessing
# between them would be a 1000x silent corruption.

def _structural_unit(unit: str) -> str:
    """Spelling-independent form: no parens, no spaces, denominators sorted.

    "ml/(kg·min)", "ml/min·kg" and "ml/kg/min" all become "ml/kg·min". A '·' or
    '*' after the first '/' binds to the denominator, which is how HealthKit
    writes composite rates.
    """
    u = unit.replace("(", "").replace(")", "").replace(" ", "")
    u = u.replace("*", "·").replace("⋅", "·").replace("×", "·")
    if "/" not in u:
        return u
    head, *rest = u.split("/")
    den = [tok for part in rest for tok in part.split("·") if tok]
    return head + "/" + "·".join(sorted(den)) if den else head


def _build_loose_index() -> tuple[dict, dict, set, set]:
    """Structural and case-folded lookup tables, with collisions excluded."""
    struct: dict[str, tuple[str, float, float]] = {}
    folded: dict[str, tuple[str, float, float]] = {}
    for u, spec in _UNIT_INDEX.items():
        for table, key in ((struct, _structural_unit(u)),
                           (folded, _structural_unit(u).casefold())):
            if key in table and table[key] != spec:
                table[key] = None          # ambiguous: refuse rather than guess
            elif key not in table:
                table[key] = spec
    s_dimless = {_structural_unit(u) for u in DIMENSIONLESS_UNITS}
    f_dimless = {_structural_unit(u).casefold() for u in DIMENSIONLESS_UNITS}
    return ({k: v for k, v in struct.items() if v is not None},
            {k: v for k, v in folded.items() if v is not None},
            s_dimless, f_dimless)


_STRUCT_INDEX, _FOLDED_INDEX, _STRUCT_DIMLESS, _FOLDED_DIMLESS = _build_loose_index()


def _resolve_unit(unit: str) -> tuple[str, float, float] | None:
    """The conversion spec for a unit, trying exact then looser spellings."""
    for table, key in ((_UNIT_INDEX, unit),
                       (_STRUCT_INDEX, _structural_unit(unit)),
                       (_FOLDED_INDEX, _structural_unit(unit).casefold())):
        hit = table.get(key)
        if hit is not None:
            return hit
    return None


def _is_dimensionless(unit: str) -> bool:
    return (unit in DIMENSIONLESS_UNITS
            or _structural_unit(unit) in _STRUCT_DIMLESS
            or _structural_unit(unit).casefold() in _FOLDED_DIMLESS)


def is_convertible_unit(unit: Optional[str]) -> bool:
    """Whether this unit is one convert_unit() can reason about at all."""
    u = (unit or "").strip()
    return _resolve_unit(u) is not None or _is_dimensionless(u)


def convert_unit(value: float, from_unit: Optional[str], to_unit: Optional[str]) -> float:
    """Convert `value` from one unit to another. Raises UnitError if it cannot."""
    a, b = (from_unit or "").strip(), (to_unit or "").strip()
    if a == b:
        return value
    # Same unit, different spelling: identity, not a conversion. This is the
    # vo2_max case — "ml/(kg·min)" -> "mL/min·kg" carries no factor.
    if _is_dimensionless(a) and _is_dimensionless(b) \
            and _structural_unit(a).casefold() == _structural_unit(b).casefold():
        return value
    fa, fb = _resolve_unit(a), _resolve_unit(b)
    if fa is None or fb is None:
        unknown = a if fa is None else b
        raise UnitError(f"unknown unit {unknown!r} (converting {a!r} -> {b!r})")
    if fa[0] != fb[0]:
        raise UnitError(f"cannot convert {a!r} ({fa[0]}) to {b!r} ({fb[0]})")
    base = value * fa[1] + fa[2]
    return (base - fb[2]) / fb[1]


def unit_converter(metric: str, raw_unit: Optional[str]):
    """Return (convert, canonical_unit) for a metric's incoming unit.

    Resolved ONCE per metric rather than per point: the unit is declared on the
    metric object, so an unconvertible one is a single decision about the whole
    series, and a single log line instead of thousands.

    Raises UnitError when the incoming unit cannot be placed. The caller drops
    those points and records why — this is the unit-boundary behavior of the
    retired Health Auto Export path.
    """
    target = canonical_unit(metric, raw_unit)
    src = (raw_unit or "").strip()
    if not src or src == (target or ""):
        # No unit declared, or already canonical. An undeclared unit is trusted
        # to be the catalog's: both ingest paths have always assumed this, and
        # the alternative is dropping every unitless series we already store.
        return (lambda v: v), (target or "")
    convert_unit(1.0, src, target)          # probe: raises UnitError if impossible
    return (lambda v: convert_unit(v, src, target)), target


def to_canonical(metric: str, value: float, raw_unit: Optional[str]) -> tuple[float, str]:
    """Convenience: convert one value into `metric`'s canonical unit."""
    convert, unit = unit_converter(metric, raw_unit)
    return convert(value), unit


# Metrics whose canonical unit is '%' (0–100) but which HealthKit reports as a 0–1
# ratio, while Health Auto Export sends them already as a percent. A real reading for
# any of these is never <= 1% (SpO2 ~70–100, steadiness "Low/OK" well above 1, double
# support while walking ~15–45%), so a value <= 1 is unambiguously a fraction and is
# scaled to percent. Scaling is idempotent (a 0–100 value is left untouched), so it
# is safe on both paths and on re-ingest.
#
# walking_asymmetry_percentage has the same HK-fraction/HAE-percent split but CANNOT
# be disambiguated by value: true sub-1% asymmetry readings exist on the percent path
# (observed 0.0–1.0 alongside 2–48 in live data), so the <=1 rule would corrupt them.
# Its pre-2026-06-11 history stays fraction-scaled; recent-window analyses self-heal
# once the 28-day window is all live data.
PERCENT_RATIO_METRICS: frozenset[str] = frozenset(
    {"blood_oxygen_saturation", "walking_steadiness",
     "walking_double_support_percentage",
     # Added 2026-08-27 (#143) with the HealthKit type. The vault's 43 rows run
     # 20.2–21.6; HKUnit.percent() would have written 0.202–0.216 beside them, a
     # silent 100x fork inside one series name. Body fat below 1% is not
     # survivable — essential fat alone is ~3% — so the <= 1 rule is
     # unambiguous here in a way it is not for walking asymmetry below.
     "body_fat_percentage"})

# The same 0–1/percent split, for the ONE metric the value rule cannot decide.
#
# `HKUnit.percent()` is a 0–1 ratio by definition, so every '%' quantity on the
# HealthKit-direct path is a fraction — that is a property of the path, not a
# guess about the number. PERCENT_RATIO_METRICS recovers it by value and is
# idempotent, which is what lets it also serve the retired HAE path.
#
# walking_asymmetry_percentage cannot use it: true sub-1% readings exist on the
# percent side (the vault holds 0.0–1.0 alongside 2–48), so a <= 1 rule would
# corrupt real data. Keyed on the type identifier instead, where the path is
# known and no heuristic is needed. The measured fork it closes: 17,133 backfill
# rows span 0.0–1.0 (fraction) and 773 receiver rows span 0.0–100.0 (percent).
HK_PERCENT_RATIO_TYPES: frozenset[str] = frozenset(
    {"HKQuantityTypeIdentifierWalkingAsymmetryPercentage"})


_FLOAT_TAIL_ULPS = 2


def _snap_float_tail(value: float | None) -> float | None:
    """Remove binary round-off immediately before a value is stored.

    This is deliberately much tighter than the 1e-6 audit query.  The observed
    tails are one or two ULPs (for example ``0.29 * 100.0`` is
    ``28.999999999999996``), while a meaningful fractional measurement such as
    62.5 is many orders of magnitude outside this boundary.  Zero is excluded
    because a tiny converted quantity can be a real near-zero measurement, not
    an integer representation tail.  The snap belongs here, at Python's
    canonicalization boundary, so neither the client nor the aggregate renderer
    authors a number.
    """
    if value is None or not math.isfinite(value):
        return value
    nearest = float(round(value))
    if nearest != 0.0 and abs(value - nearest) <= (
            _FLOAT_TAIL_ULPS * max(math.ulp(value), math.ulp(nearest))):
        return nearest
    return value


def canonical_value(metric: str, value: float) -> float:
    """Canonicalize a sample's scale and remove representational float tails.

    Percent-ratio metrics (see PERCENT_RATIO_METRICS) are first scaled from a
    0–1 fraction to percent, then passed through the narrow float-tail policy.
    """
    if metric in PERCENT_RATIO_METRICS and value is not None and value <= 1.0:
        value = value * 100.0
    return _snap_float_tail(value)


def hk_canonical_value(type_identifier: str, metric: str, value: float) -> float:
    """`canonical_value` for a sample known to have arrived from HealthKit.

    Identical except for HK_PERCENT_RATIO_TYPES, which are scaled by identifier
    rather than by value. The two rules are deliberately exclusive: applying
    both to one metric would scale a 0.005 fraction to 50%.
    """
    if type_identifier in HK_PERCENT_RATIO_TYPES:
        value = value if value is None else value * 100.0
        return _snap_float_tail(value)
    return canonical_value(metric, value)


# The Health Auto Export vocabulary — its snake_case aliases, its sleep-stage
# value map, and `hae_name_to_canonical` — was deleted with that ingest path
# (#36, #38). Its lesson was not deleted, because it is the reason the function
# below exists:
#
#   `hae_name_to_canonical` passed an unmapped name through unchanged, which is
#   the right call — a name we have not mapped is still real data, and dropping
#   it would be worse than storing it under its own name. But the result may or
#   may not be a name this project knows, so callers had to check
#   `is_known_metric()` and surface the miss. They did not, and **two
#   vocabulary forks reached production unnoticed**: the `six_minute_walk*` and
#   `*breathing_disturbances` pairs, both still in CATALOG because unifying
#   them now needs a records migration.
#
# Both forks are closed at the *source* as of #143: HK_QUANTITY maps each
# identifier explicitly onto the name the live series uses, so nothing new
# arrives under the auto-derived one. The stale halves stay in CATALOG for the
# rows already written under them — 58 and 1 respectively.
#
# The other half of that lesson is `hk_quantity_to_canonical`'s auto-derive,
# which `hk_parse` deliberately does NOT use: it resolves through HK_QUANTITY
# alone, so an unmapped identifier is dropped into `unhandled` rather than
# quietly opening a third fork. That is a *visible* absence, and #143 is the
# argument for it — the two identifiers that were requested but unmapped were
# discarded for five days without a single error.
#
# `tests/test_catalog_coverage.py` pins both pairs so that removing one to
# tidy up fails loudly instead of silently reopening the fork.


def is_known_metric(metric: str) -> bool:
    """Whether this canonical name is one the catalog actually defines.

    Everything downstream — the unit, the daily aggregation rule, the wear
    filter, the correlation groups — is looked up by this name and silently
    defaults when it is missing (canonical_unit falls back to the raw unit,
    agg_for to 'mean', group to none). A metric that is not here is not
    wrong, but it is unmanaged, and that should be someone's decision rather
    than an accident.
    """
    return metric in CATALOG


def known_metrics() -> list[str]:
    return sorted(CATALOG.keys())
