"""Generate a fully synthetic demo vault that matches this repo's schema.

Nobody who clones this repository has ten years of Apple Health data, and the
author's database can never ship. Without a generator there is no way for a
stranger to run anything, no way for CI to exercise the analysis layer against
realistic data shapes, and no differential correctness gate.

    python -m health_advisor.demo --out demo.db --days 730

Everything here is invented. No real person, device or measurement is
reproduced.

WHAT "REALISTIC" MEANS HERE
---------------------------
The point is not plausible-looking noise; it is that the deterministic analysis
layer returns sensible answers when pointed at the result. So the generator
produces the *shapes* the engine keys off:

* raw ``records`` at the resolution each consumer actually reads — 20-second
  distance/step/heart-rate samples inside sessions, because
  ``metrics.impact_bucket_rows`` classifies jogging from 20-second cadence, and
  hourly background samples outside them;
* a weekly rhythm (weekends move more and later), a slow seasonal term, and a
  multi-year fitness trend, so trends, weekly series and correlations have
  something real to find rather than white noise;
* two instrument eras — a phone-only stretch, then a watch — so
  ``metric_source_months`` and instrument-era detection see a genuine change;
* two devices writing the SAME movement over the arbitration window, which is
  the correctness-critical path a single-source vault tests not at all;
* sleep written as sessions and attributed to the day the session ENDS, which
  is the invariant ``derive.reattribute_sleep`` exists to maintain.

DEVICE NAMES
------------
``DEMO_WATCH`` / ``DEMO_PHONE`` end in "Apple Watch" and "iPhone" *on purpose*,
and that suffix is load-bearing rather than decorative: cross-source distance
arbitration in ``db._workout_arbitration`` matches ``LIKE '%Apple Watch'`` and
``LIKE '%iPhone'``. A vault whose devices are called "Demo Watch" and "Demo
Phone" would double-count every overlapping distance sample and would exercise
none of that logic — the arbitration would silently never fire, which is the
worst of the available outcomes. The names carry no personal identity, which is
the actual requirement; the prefix is generic and the suffix is a device model.

DETERMINISM
-----------
``build_demo_vault`` is a pure function of ``(days, seed, end_date)``. Two runs
produce the same rows, so :func:`content_digest` can be used as a correctness
gate: change something in the aggregation layer and the digest moves.

The FILE is not byte-identical between runs — ``db.init_db`` stamps a wall-clock
``vault_meta.created_at``, and SQLite page layout is not a guaranteed function
of the inserts. Compare :func:`content_digest`, never the bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import sqlite3
import sys
from datetime import date, datetime, timedelta
import os
from pathlib import Path

from . import db
from . import derive
from . import normalize as nz
from . import vault as vaultmod

# --------------------------------------------------------------------------- #
# Invented identities. Nothing here refers to a real person or device.
# --------------------------------------------------------------------------- #
DEMO_WATCH = "Demo Apple Watch"
DEMO_PHONE = "Demo iPhone"
DEMO_SCALE = "Demo Scale"
CHECKIN_SOURCE = "checkin"      # matches subjective.CHECKIN_ORIGIN

DEFAULT_DAYS = 730
DEFAULT_SEED = 42
# A FIXED anchor, not `today`. "Same seed, same content" has to survive the
# calendar, and the analysis layer anchors its windows to the latest data date
# anyway, so a fixed end costs nothing. It deliberately sits after
# normalize.WORKOUT_SOURCE_ARBITRATION_FROM so the generated window crosses the
# cutover and the two-device arbitration is actually reachable.
DEFAULT_END_DATE = "2026-08-31"
DEMO_TIMEZONE = "UTC"           # local == UTC, so local_date needs no offset math

BUCKET_S = 20                   # metrics.IMPACT_BUCKET_SECONDS
_INSERT_CHUNK = 5000


# --------------------------------------------------------------------------- #
# Row helpers
# --------------------------------------------------------------------------- #
def _utc(dt: datetime) -> str:
    """The exact spelling backfill produces: ISO-8601 with an explicit offset."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _local(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _rec(metric: str, value: float, start: datetime, end: datetime, source: str,
         *, local_date: str | None = None, origin: str = "backfill",
         unit: str | None = None) -> dict:
    """One `records` row, keyed exactly as an ingest path would key it."""
    unit = unit if unit is not None else nz.canonical_unit(metric, None)
    value = round(float(value), 6)
    start_utc, end_utc = _utc(start), _utc(end)
    ld = local_date or start.date().isoformat()
    return {
        "metric": metric, "value": value, "unit": unit,
        "start_utc": start_utc, "end_utc": end_utc,
        "start_local": _local(start), "local_date": ld,
        "source": source, "origin": origin,
        "dedupe_key": db.record_key(metric, start_utc, end_utc, value, unit, source),
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _align(dt: datetime) -> datetime:
    """Snap to a 20-second boundary so samples land on impact-bucket edges."""
    return dt.replace(second=(dt.second // BUCKET_S) * BUCKET_S, microsecond=0)


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
# Monday-anchored weekly rhythm. Wednesday and Friday are deliberately empty:
# a plan with no rest days produces a training-load series with no variance.
_WEEKLY_PLAN = {
    0: "traditional_strength_training",
    1: "running",
    2: "walking",
    3: "running",
    5: "running",
    6: "cycling",
}
_SKIP_PROBABILITY = 0.12


class _Generator:
    """Holds the RNG and the slowly-moving state (fitness, weight, fatigue).

    A class rather than free functions because the interesting realism is in
    the *carry-over*: yesterday's session shows up in today's soreness and
    resting heart rate, which is what gives ``correlate_metrics`` and
    ``scan_correlations`` a real association to find instead of noise.
    """

    def __init__(self, days: int, seed: int, end_date: str):
        self.days = days
        self.rng = random.Random(seed)
        self.end = date.fromisoformat(end_date)
        self.start = self.end - timedelta(days=days - 1)
        # Instrument era: a phone-only stretch, then a watch appears. Capped so
        # a ten-year vault does not spend three years without a heart rate.
        self.watch_from = self.start + timedelta(days=min(days // 5, 240))
        self.records: list[dict] = []
        self.workouts: list[dict] = []
        self.subjective: list[tuple] = []
        # carry-over state
        self.prev_load = 0.0
        self.prev_sleep_h = 7.5
        self.body_mass = 186.0

    # -- small deterministic helpers ---------------------------------------
    def _n(self, mu: float, sigma: float) -> float:
        return self.rng.gauss(mu, sigma)

    def build(self) -> None:
        for i in range(self.days):
            self._day(i, self.start + timedelta(days=i))

    # -- one day ------------------------------------------------------------
    def _day(self, i: int, d: date) -> None:
        rng = self.rng
        fitness = i / max(1, self.days - 1)          # 0 -> 1 over the window
        yday = d.timetuple().tm_yday
        season = math.sin(2 * math.pi * (yday - 80) / 365.25)   # peaks in summer
        weekend = d.weekday() >= 5
        has_watch = d >= self.watch_from
        primary = DEMO_WATCH if has_watch else DEMO_PHONE
        day_s = d.isoformat()
        midnight = datetime(d.year, d.month, d.day)
        first = len(self.records)

        # A watch that is off the wrist is the single most common real-world
        # data defect, and correlate.py's wear filter exists for it. 5% of days
        # get partial coverage so the filter has something to drop.
        low_wear = has_watch and rng.random() < 0.05

        sessions = self._sessions(d, fitness, weekend, has_watch, primary)
        busy = [(w["_start"], w["_end"]) for w in sessions]

        walk = self._walk_block(d, weekend, primary, busy)
        busy += [(walk["start"], walk["end"])] if walk else []

        self._background(d, midnight, fitness, season, weekend, primary, busy)
        self._heart_rate_background(midnight, fitness, low_wear, busy, has_watch)

        if has_watch:
            self._sleep(d, fitness)
            self._daily_points(d, midnight, fitness, season, has_watch)
        self._body(d, midnight, fitness, has_watch)
        self._nutrition(d, midnight, weekend)

        exercise_min = sum(w["duration_min"] for w in sessions)
        exercise_min += walk["minutes"] if walk else 0.0
        if exercise_min:
            self.records.append(_rec(
                "apple_exercise_time", round(exercise_min, 1),
                midnight + timedelta(hours=23), midnight + timedelta(hours=23),
                primary, local_date=day_s))

        if has_watch:
            self._mirror_to_phone(d, first)

        load = sum(w["_load"] for w in sessions)
        if has_watch:
            self._checkin(d, load, weekend)
        self.prev_load = load

    # -- workouts -----------------------------------------------------------
    def _sessions(self, d: date, fitness: float, weekend: bool,
                  has_watch: bool, source: str) -> list[dict]:
        rng = self.rng
        kind = _WEEKLY_PLAN.get(d.weekday())
        # In the phone-only era there is no watch to record a session; only the
        # occasional walk was logged. That is what an instrument era looks like.
        if kind is None or rng.random() < _SKIP_PROBABILITY:
            return []
        if not has_watch:
            if kind != "walking":
                return []
        base_hour = 9 if weekend else 18
        start = _align(datetime(d.year, d.month, d.day, base_hour)
                       + timedelta(minutes=rng.randint(-45, 45)))
        if kind == "running":
            return [self._run(start, fitness, weekend, source)]
        if kind == "walking":
            # No heart rate before the watch: a phone records the movement and
            # nothing about the heart. An era that carries HR is not an era.
            return [self._walk_workout(start, source, with_hr=has_watch)]
        if kind == "cycling":
            return [self._ride(start, fitness, source)]
        return [self._strength(start, source)]

    def _run(self, start: datetime, fitness: float, weekend: bool,
             source: str) -> dict:
        """A run recorded the way a watch records one: 20-second distance, step
        and heart-rate samples, with a walking warm-up and cool-down."""
        rng = self.rng
        total_min = _clamp(24 + 26 * fitness + self._n(0, 5), 16, 78)
        if weekend:
            total_min *= 1.35
        warm, cool = 4.0, 3.0
        jog_min = max(6.0, total_min - warm - cool)
        pace = _clamp(11.9 - 1.9 * fitness + self._n(0, 0.4), 7.6, 13.6)
        cadence = _clamp(157 + 11 * fitness + self._n(0, 3.5), 145, 183)
        hr_base = _clamp(139 + 9 * fitness + self._n(0, 4), 124, 160)

        segs = [(warm, 18.5, 111.0, 108.0),       # (minutes, pace, cadence, hr)
                (jog_min, pace, cadence, hr_base),
                (cool, 20.0, 104.0, 118.0)]
        return self._session_from_segments(
            start, "running", source, segs, drift=9.0)

    def _walk_workout(self, start: datetime, source: str, *,
                      with_hr: bool = True) -> dict:
        mins = _clamp(30 + self._n(0, 8), 18, 62)
        return self._session_from_segments(
            start, "walking", source,
            [(mins, _clamp(18.0 + self._n(0, 1.2), 15.0, 24.0), 112.0,
              _clamp(101 + self._n(0, 5), 88, 118))], drift=3.0,
            with_hr=with_hr)

    def _session_from_segments(self, start: datetime, kind: str, source: str,
                               segments, *, drift: float,
                               with_hr: bool = True) -> dict:
        """Expand (minutes, pace, cadence, hr) segments into 20-second samples.

        Cadence is the discriminator ``impact_bucket_rows`` classifies jogging
        on (>= metrics.IMPACT_JOG_CADENCE_MIN steps/min inside a workout
        window), so the warm-up and cool-down segments come out as walking and
        the middle as jogging — inside one session, which is exactly the case
        ``analysis.longest_block`` was written for.
        """
        t = start
        hrs: list[float] = []
        miles = 0.0
        elapsed = 0.0
        total = sum(s[0] for s in segments)
        # NOTE the loop variable is `hr_base`, not `hr`: naming it `hr` shadowed
        # the `with_hr` flag's earlier spelling and silently emitted heart rate
        # in the phone-only era, which the era test caught.
        for mins, pace, cadence, hr_base in segments:
            n = max(1, int(round(mins * 60 / BUCKET_S)))
            for _ in range(n):
                end = t + timedelta(seconds=BUCKET_S)
                mi = (BUCKET_S / 60.0) / pace * (1 + self._n(0, 0.05))
                steps = cadence / 3.0 * (1 + self._n(0, 0.02))
                frac = elapsed / max(1e-9, total)
                bpm = _clamp(hr_base + drift * frac + self._n(0, 2.5), 60, 196)
                self.records.append(
                    _rec("distance_walking_running", mi, t, end, source))
                self.records.append(_rec("step_count", steps, t, end, source))
                if with_hr:
                    self.records.append(
                        _rec("heart_rate", round(bpm, 1), t, t, source,
                             unit="count/min"))
                    hrs.append(bpm)
                miles += mi
                t = end
                elapsed += BUCKET_S / 60.0
        duration = round((t - start).total_seconds() / 60.0, 2)
        kcal = round(duration * (7.6 if kind == "running" else 4.4), 1)
        self._energy_samples(start, t, kcal, source)
        return self._workout_row(kind, start, t, duration, kcal, miles, hrs,
                                 source, "mi")

    def _ride(self, start: datetime, fitness: float, source: str) -> dict:
        """Cycling: a distinct workout type on a distinct distance metric, so
        `workout_mix` and per-type queries have more than one answer."""
        duration = round(_clamp(45 + 20 * fitness + self._n(0, 12), 25, 105), 2)
        end = _align(start + timedelta(minutes=duration))
        duration = round((end - start).total_seconds() / 60.0, 2)
        speed = _clamp(13.5 + 2.5 * fitness + self._n(0, 1.0), 9.0, 20.0)
        miles = duration / 60.0 * speed
        hrs = self._sparse_hr(start, end, _clamp(128 + self._n(0, 6), 108, 152),
                              source, step_s=60)
        self.records.append(_rec("distance_cycling", miles, start, end, source))
        kcal = round(duration * 8.1, 1)
        self._energy_samples(start, end, kcal, source)
        return self._workout_row("cycling", start, end, duration, kcal, miles,
                                 hrs, source, "mi")

    def _strength(self, start: datetime, source: str) -> dict:
        duration = round(_clamp(38 + self._n(0, 8), 20, 70), 2)
        end = _align(start + timedelta(minutes=duration))
        duration = round((end - start).total_seconds() / 60.0, 2)
        hrs = self._sparse_hr(start, end, _clamp(112 + self._n(0, 6), 95, 138),
                              source, step_s=60)
        kcal = round(duration * 5.2, 1)
        self._energy_samples(start, end, kcal, source)
        return self._workout_row("traditional_strength_training", start, end,
                                 duration, kcal, None, hrs, source, None)

    def _sparse_hr(self, start: datetime, end: datetime, base: float,
                   source: str, *, step_s: int) -> list[float]:
        hrs: list[float] = []
        t = start
        while t < end:
            bpm = _clamp(base + self._n(0, 6), 55, 190)
            self.records.append(_rec("heart_rate", round(bpm, 1), t, t, source,
                                     unit="count/min"))
            hrs.append(bpm)
            t += timedelta(seconds=step_s)
        return hrs

    def _energy_samples(self, start: datetime, end: datetime, kcal: float,
                        source: str) -> None:
        """Active energy in five-minute slices, the way a watch emits it."""
        span = (end - start).total_seconds()
        n = max(1, int(span // 300))
        per = kcal / n
        for k in range(n):
            a = start + timedelta(seconds=k * 300)
            b = min(end, a + timedelta(seconds=300))
            self.records.append(_rec("active_energy", per, a, b, source))

    def _workout_row(self, kind, start, end, duration, kcal, miles, hrs,
                     source, unit_distance) -> dict:
        avg = round(sum(hrs) / len(hrs), 1) if hrs else None
        mx_hr = round(max(hrs), 1) if hrs else None
        row = {
            "workout_type": kind,
            "start_utc": _utc(start), "end_utc": _utc(end),
            "local_date": start.date().isoformat(),
            "duration_min": duration, "energy_kcal": kcal,
            "distance_mi": round(miles, 4) if miles is not None else None,
            "unit_distance": unit_distance, "source": source,
            "route_ref": None, "avg_heart_rate": avg, "max_heart_rate": mx_hr,
            "dedupe_key": db.workout_key(kind, _utc(start), _utc(end)),
            "hk_uuid": None,
        }
        self.workouts.append({k: v for k, v in row.items()})
        row["_start"], row["_end"] = start, end
        # A crude Banister-shaped stand-in, used only to drive the next day's
        # soreness. hr_load.py computes the real number from the samples above.
        row["_load"] = duration * ((avg or 100) / 100.0) ** 2
        return row

    # -- background movement -------------------------------------------------
    def _walk_block(self, d: date, weekend: bool, source: str, busy) -> dict | None:
        """A deliberate walk, recorded at 20-second resolution but NOT as a
        workout. Without it `impact_volume` reports no walking at all: hourly
        background samples are far too coarse to fall in the walking pace band.
        """
        rng = self.rng
        if rng.random() < 0.15:
            return None
        hour = 15 if weekend else 12
        start = _align(datetime(d.year, d.month, d.day, hour)
                       + timedelta(minutes=rng.randint(0, 50)))
        minutes = _clamp(24 + self._n(0, 7), 12, 46)
        end = start + timedelta(seconds=int(minutes * 60 // BUCKET_S) * BUCKET_S)
        if any(s < end and start < e for s, e in busy):
            return None
        pace = _clamp(18.5 + self._n(0, 1.5), 15.0, 26.0)
        t = start
        while t < end:
            nxt = t + timedelta(seconds=BUCKET_S)
            self.records.append(_rec(
                "distance_walking_running",
                (BUCKET_S / 60.0) / pace * (1 + self._n(0, 0.06)), t, nxt, source))
            self.records.append(_rec(
                "step_count", 113.0 / 3.0 * (1 + self._n(0, 0.03)), t, nxt, source))
            t = nxt
        return {"start": start, "end": end,
                "minutes": round((end - start).total_seconds() / 60.0, 1)}

    def _background(self, d: date, midnight: datetime, fitness: float,
                    season: float, weekend: bool, source: str, busy) -> None:
        """Hourly steps / distance / energy for the rest of the waking day."""
        rng = self.rng
        base = 6200 + 1400 * (1 if weekend else 0) + 900 * season \
            + 700 * fitness + self._n(0, 900)
        base = max(900.0, base)
        hours = list(range(7, 23))
        # A fixed diurnal shape, jittered — mornings and early evenings move.
        shape = [0.5, 0.9, 1.1, 1.0, 0.8, 1.2, 1.1, 0.9,
                 0.8, 0.9, 1.1, 1.3, 1.1, 0.8, 0.5, 0.3]
        weights = [w * (1 + rng.uniform(-0.25, 0.25)) for w in shape]
        total_w = sum(weights)
        for hour, w in zip(hours, weights):
            a = midnight + timedelta(hours=hour)
            b = a + timedelta(hours=1)
            if any(s < b and a < e for s, e in busy):
                continue
            steps = base * w / total_w
            if steps < 1:
                continue
            self.records.append(_rec("step_count", round(steps, 1), a, b, source))
            self.records.append(_rec(
                "distance_walking_running", steps * 0.000445, a, b, source))
            self.records.append(_rec(
                "active_energy", round(steps * 0.035 + self._n(0, 3), 1), a, b,
                source))
        # Basal energy every two hours. Each sample stays under
        # normalize.SAMPLE_CEILING['basal_energy'] (150) — a larger one would be
        # read as a whole-day estimate and arbitrated away.
        for hour in range(0, 24, 2):
            a = midnight + timedelta(hours=hour)
            self.records.append(_rec(
                "basal_energy", round(_clamp(62 + self._n(0, 4), 40, 95), 1),
                a, a + timedelta(hours=2), source))
        self.records.append(_rec(
            "flights_climbed", float(rng.randint(2, 16)),
            midnight + timedelta(hours=18), midnight + timedelta(hours=19), source))

    def _mirror_to_phone(self, d: date, first: int) -> None:
        """The second device, from the arbitration cutover onward.

        The phone writes its own distance stream for the SAME movement the
        watch recorded. That overlap is the correctness-critical path — a vault
        with one source exercises none of `db._workout_arbitration`, which drops
        these rows in favour of the watch both for the whole day and inside
        workout windows. `first` is this day's first record index, so the scan
        stays linear over the day rather than over the whole vault.
        """
        if d.isoformat() < nz.WORKOUT_SOURCE_ARBITRATION_FROM:
            return
        mirrored = []
        for r in self.records[first:]:
            if r["metric"] != "distance_walking_running" or r["source"] != DEMO_WATCH:
                continue
            value = round(r["value"] * (1 + self.rng.uniform(-0.12, 0.12)), 6)
            mirrored.append({
                **r, "source": DEMO_PHONE, "value": value,
                "dedupe_key": db.record_key(
                    r["metric"], r["start_utc"], r["end_utc"], value, r["unit"],
                    DEMO_PHONE),
            })
        self.records.extend(mirrored)

    def _heart_rate_background(self, midnight: datetime, fitness: float,
                               low_wear: bool, busy, has_watch: bool) -> None:
        if not has_watch:
            return
        rest = 60 - 6 * fitness
        # 20:00-22:00 is the charger. A vault where wear_hours is always 24 is
        # a vault where the wear filter can never be observed working.
        last = 14 if low_wear else 24
        for hour in range(6 if low_wear else 0, last):
            a = midnight + timedelta(hours=hour)
            if 20 <= hour < 22 and not low_wear:
                continue
            if any(s <= a < e for s, e in busy):
                continue
            if hour < 6:
                bpm = rest - 4 + self._n(0, 2.5)          # asleep
            elif hour < 9 or hour >= 21:
                bpm = rest + 9 + self._n(0, 5)
            else:
                bpm = rest + 20 + self._n(0, 9)
            self.records.append(_rec("heart_rate", round(_clamp(bpm, 40, 175), 1),
                                     a, a, DEMO_WATCH, unit="count/min"))

    # -- sleep ---------------------------------------------------------------
    def _sleep(self, d: date, fitness: float) -> None:
        """One overnight session, written as the intervals a watch writes.

        Attribution is the invariant: every record of the session carries the
        local_date the session ENDS on, which is what `derive.reattribute_sleep`
        would rewrite them to. A vault built the naive way (attribute each
        sample to the day it ends) splits a night in two at midnight.
        """
        rng = self.rng
        prev = d - timedelta(days=1)
        weekend_night = prev.weekday() >= 4          # Fri/Sat nights run later
        bed_min = 22 * 60 + 40 + (55 if weekend_night else 0) + rng.randint(-40, 70)
        bed = datetime(prev.year, prev.month, prev.day) + timedelta(minutes=bed_min)
        hours = _clamp(7.5 + self._n(0, 0.75) - (0.4 if weekend_night else 0.0),
                       4.6, 10.2)
        wake = bed + timedelta(minutes=round(hours * 60))
        if wake.date() != d:                          # keep the invariant exact
            wake = datetime(d.year, d.month, d.day, 6, 30)
            if wake <= bed:
                return
        day_s = d.isoformat()
        total_min = (wake - bed).total_seconds() / 60.0
        self.records.append(_rec("sleep_in_bed", total_min, bed, wake, DEMO_WATCH,
                                 local_date=day_s))

        latency = _clamp(9 + self._n(0, 5), 2, 32)
        awake_n = rng.choices([0, 1, 2, 3], weights=[18, 40, 30, 12])[0]
        cuts = sorted(rng.uniform(0.15, 0.9) for _ in range(awake_n))
        t = bed + timedelta(minutes=latency)
        span = (wake - t).total_seconds() / 60.0
        asleep_total = 0.0
        for frac in cuts + [1.0]:
            seg_end = bed + timedelta(minutes=latency + span * frac)
            mins = (seg_end - t).total_seconds() / 60.0
            if seg_end < t:                 # cuts landed too close together
                seg_end = t
                mins = 0.0
            if mins > 0.5:
                self.records.append(_rec("sleep_asleep", mins, t, seg_end,
                                         DEMO_WATCH, local_date=day_s))
                asleep_total += mins
            if frac == 1.0:
                break
            awake_len = _clamp(4 + self._n(0, 3), 1.5, 16)
            self.records.append(_rec("sleep_awake", awake_len, seg_end,
                                     seg_end + timedelta(minutes=awake_len),
                                     DEMO_WATCH, local_date=day_s))
            t = seg_end + timedelta(minutes=awake_len)

        # Stage breakdown. Not in derive._STAGE_METRICS (it would double-count
        # the session), but real vaults carry it and the briefing reads it.
        deep = asleep_total * _clamp(0.17 + self._n(0, 0.03), 0.08, 0.27)
        rem = asleep_total * _clamp(0.22 + self._n(0, 0.04), 0.10, 0.34)
        core = max(0.0, asleep_total - deep - rem)
        for metric, mins in (("sleep_deep", deep), ("sleep_rem", rem),
                             ("sleep_core", core)):
            if mins > 0.5:
                self.records.append(_rec(metric, mins, bed, wake, DEMO_WATCH,
                                         local_date=day_s))
        self.prev_sleep_h = asleep_total / 60.0

    # -- daily point metrics --------------------------------------------------
    def _daily_points(self, d: date, midnight: datetime, fitness: float,
                      season: float, has_watch: bool) -> None:
        """resting_heart_rate and walking_heart_rate_average are written as
        same-day REVISIONS — one start_utc, progressively later end_utc — which
        is why normalize.CATALOG gives them agg 'last' rather than 'mean'. A
        generator that emits one row a day would make that distinction
        untestable."""
        rng = self.rng
        day_s = d.isoformat()
        settled = _clamp(60.5 - 6.0 * fitness + 0.9 * (self.prev_load / 40.0)
                         + self._n(0, 2.1), 44, 78)
        for k, hours in enumerate((8, 16, 24)):
            draft = settled + (self._n(0, 1.6) if hours < 24 else 0.0)
            self.records.append(_rec(
                "resting_heart_rate", round(draft, 1), midnight,
                midnight + timedelta(hours=hours), DEMO_WATCH,
                local_date=day_s, unit="count/min"))
        walking = _clamp(settled + 38 + self._n(0, 3), 70, 130)
        for hours in (12, 24):
            self.records.append(_rec(
                "walking_heart_rate_average",
                round(walking + (self._n(0, 1.4) if hours < 24 else 0.0), 1),
                midnight, midnight + timedelta(hours=hours), DEMO_WATCH,
                local_date=day_s, unit="count/min"))

        t = midnight + timedelta(hours=6, minutes=45)
        self.records.append(_rec(
            "heart_rate_variability",
            round(_clamp(42 + 14 * fitness - 5 * (self.prev_load / 60.0)
                         + self._n(0, 6), 12, 105), 1), t, t, DEMO_WATCH))
        self.records.append(_rec(
            "respiratory_rate", round(_clamp(14.6 + self._n(0, 0.9), 10, 20), 2),
            t, t, DEMO_WATCH))
        self.records.append(_rec(
            "blood_oxygen_saturation",
            round(_clamp(97.0 + self._n(0, 1.0), 90, 100), 1), t, t, DEMO_WATCH))
        self.records.append(_rec(
            "time_in_daylight",
            round(_clamp(48 + 55 * season + self._n(0, 25), 0, 240), 1),
            midnight + timedelta(hours=12), midnight + timedelta(hours=13),
            DEMO_WATCH))
        if d.weekday() == 2:
            v = midnight + timedelta(hours=20)
            self.records.append(_rec(
                "vo2_max", round(_clamp(37.5 + 6.5 * fitness + self._n(0, 0.5),
                                        28, 58), 1), v, v, DEMO_WATCH))
        if rng.random() < 0.2:
            m = midnight + timedelta(hours=21, minutes=rng.randint(0, 50))
            self.records.append(_rec(
                "mindful_minutes", float(rng.choice([5, 10, 10, 15, 20])),
                m, m + timedelta(minutes=10), DEMO_WATCH))

    def _body(self, d: date, midnight: datetime, fitness: float,
              has_watch: bool) -> None:
        """A third source. Scale readings are 'last' metrics with no arbitration
        against the watch, so they exercise multi-source provenance
        (metric_source_months) without contending for the same movement."""
        if self.rng.random() > 0.72:
            return
        target = 186.0 - 13.0 * fitness
        self.body_mass += (target - self.body_mass) * 0.12 + self._n(0, 0.35)
        lb = round(_clamp(self.body_mass, 140, 240), 1)
        t = midnight + timedelta(hours=7, minutes=15)
        self.records.append(_rec("body_mass", lb, t, t, DEMO_SCALE))
        # The documented RENPHO relation: body fat on this class of scale is a
        # function of weight, not an independent measurement (CATALOG caveat).
        bf = round(_clamp(-20.404 + 0.2175 * lb, 5, 45), 2)
        self.records.append(_rec("body_fat_percentage", bf, t, t, DEMO_SCALE))
        self.records.append(_rec("lean_body_mass", round(lb * (1 - bf / 100), 2),
                                 t, t, DEMO_SCALE))
        self.records.append(_rec("body_mass_index",
                                 round(lb / (70.0 ** 2) * 703, 2), t, t, DEMO_SCALE))

    def _nutrition(self, d: date, midnight: datetime, weekend: bool) -> None:
        if self.rng.random() > 0.55:
            return
        t = midnight + timedelta(hours=20, minutes=30)
        kcal = _clamp(2280 + (250 if weekend else 0) + self._n(0, 340), 1300, 4200)
        for metric, value in (
                ("dietary_energy_consumed", kcal),
                ("dietary_protein", kcal * 0.21 / 4),
                ("dietary_carbohydrates", kcal * 0.45 / 4),
                ("dietary_fat_total", kcal * 0.34 / 9),
                ("dietary_fiber", _clamp(24 + self._n(0, 7), 6, 60)),
                ("dietary_water", _clamp(2100 + self._n(0, 500), 500, 4500))):
            self.records.append(_rec(metric, round(value, 1), t, t, "Demo Food Log"))

    def _checkin(self, d: date, load: float, weekend: bool) -> None:
        """The nightly self-report, mirrored into `records` exactly as
        subjective.log does it (source and origin 'checkin').

        Its values are DRIVEN by the same latent state as the sensor series —
        sleep and yesterday's load — so `correlate_metrics` and
        `scan_correlations` have a real association to recover rather than a
        coin flip. That is the whole reason for including it.
        """
        rng = self.rng
        if rng.random() > 0.85:
            return
        day_s = d.isoformat()
        sleep_h = self.prev_sleep_h
        # Signal, then a lot of noise. The associations are meant to be
        # RECOVERABLE, not overwhelming: with tighter noise the whole scan came
        # back at |rho| ~ 0.6 and 43 of 98 pairs passing FDR, which is not what
        # a real self-report series looks like and would teach a reader to
        # expect effects this project's own analysis says are not there.
        energy = _clamp(round(1.9 + 0.36 * sleep_h - 0.006 * self.prev_load
                              + self._n(0, 1.05)), 1, 5)
        soreness = _clamp(round(1.7 + 0.011 * self.prev_load
                                + self._n(0, 1.0)), 1, 5)
        stress = _clamp(round(3.3 - 0.14 * sleep_h + (0.5 if not weekend else 0.0)
                              + self._n(0, 1.05)), 1, 5)
        quality = _clamp(round(1.4 + 0.34 * sleep_h + self._n(0, 0.95)), 1, 5)
        caffeine = float(rng.choices([0, 1, 2, 3, 4], [5, 25, 40, 22, 8])[0])
        alcohol = float(rng.choices([0, 1, 2, 3], [62, 22, 12, 4])[0])
        fields = {"subjective_energy": energy, "subjective_soreness": soreness,
                  "subjective_stress": stress, "subjective_sleep_quality": quality,
                  "caffeine_drinks": caffeine, "alcohol_drinks": alcohol}
        t = datetime(d.year, d.month, d.day, 20, 0, 0)
        for metric, value in fields.items():
            self.records.append(_rec(metric, float(value), t, t, CHECKIN_SOURCE,
                                     local_date=day_s, origin=CHECKIN_SOURCE))
        self.subjective.append((day_s, int(stress), int(soreness), int(energy),
                                int(quality), caffeine, alcohol,
                                f"{day_s}T20:00:00+00:00"))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_demo_vault(path: str | Path, *, days: int = DEFAULT_DAYS,
                     seed: int = DEFAULT_SEED,
                     end_date: str = DEFAULT_END_DATE,
                     replace: bool = True,
                     read_only: bool = True) -> dict:
    """Build a synthetic vault at `path` and return a summary report.

    Deterministic in `(days, seed, end_date)`: two builds with the same
    arguments hold the same rows. See :func:`content_digest`.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    date.fromisoformat(end_date)            # validate early, not mid-build
    path = Path(path)
    if path.exists():
        if not replace:
            raise FileExistsError(f"{path} exists (pass replace=True to overwrite)")
        path.unlink()
        for suffix in ("-journal", "-wal", "-shm"):
            side = path.with_name(path.name + suffix)
            if side.exists():
                side.unlink()

    gen = _Generator(days, seed, end_date)
    gen.build()

    conn = db.connect(path)
    try:
        db.init_db(conn)
        vaultmod.set_local_timezone(conn, DEMO_TIMEZONE)
        vaultmod.set_unit_system(conn, "imperial")
        conn.executemany(
            "INSERT OR REPLACE INTO vault_meta (key, value) VALUES (?, ?)",
            [("demo_vault", "1"), ("demo_seed", str(seed)),
             ("demo_days", str(days)), ("demo_end_date", end_date)])

        for i in range(0, len(gen.records), _INSERT_CHUNK):
            db.insert_records(conn, gen.records[i:i + _INSERT_CHUNK])
        db.insert_workouts(conn, gen.workouts)
        conn.executemany(
            "INSERT OR REPLACE INTO subjective (date, stress, soreness, energy, "
            "sleep_quality, caffeine_drinks, alcohol_drinks, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", gen.subjective)
        conn.commit()

        # daily_metrics is a CACHE of records; build it the way the engine does,
        # so cross-source arbitration is applied rather than reimplemented here.
        db.recompute_daily_metrics(conn, full=True)
        conn.commit()
        derive.update_for_days(conn, [
            (gen.start + timedelta(days=i)).isoformat() for i in range(days)])
        conn.commit()
        db.log_ingest(conn, "demo", "generate", len(gen.records),
                      len(gen.records),
                      f"synthetic vault: days={days} seed={seed} end={end_date}")
        conn.commit()
        report = summarize(conn)
    finally:
        conn.close()
    # A vault is read-only by convention once built: analysis never writes to it,
    # and the sandbox executor's defence-in-depth check asserts mode 0444. A
    # rebuild is the way to change a demo vault, which is why `replace` clears
    # the file rather than expecting to write into it.
    if read_only:
        os.chmod(path, 0o444)
    report.update({"path": str(path), "days": days, "seed": seed,
                   "start_date": gen.start.isoformat(), "end_date": end_date,
                   "watch_from": gen.watch_from.isoformat(),
                   "mode": oct(os.stat(path).st_mode & 0o777)})
    return report


_COUNT_TABLES = ("records", "workouts", "daily_metrics", "metric_source_months",
                 "subjective", "ingest_log")


def summarize(conn: sqlite3.Connection) -> dict:
    """Row counts and coverage for a built vault."""
    out: dict = {"rows": {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                          for t in _COUNT_TABLES}}
    out["metrics"] = conn.execute(
        "SELECT COUNT(DISTINCT metric) FROM daily_metrics").fetchone()[0]
    out["sources"] = [r[0] for r in conn.execute(
        "SELECT DISTINCT source FROM records ORDER BY source")]
    out["workout_types"] = [r[0] for r in conn.execute(
        "SELECT DISTINCT workout_type FROM workouts ORDER BY workout_type")]
    row = conn.execute(
        "SELECT MIN(date), MAX(date) FROM daily_metrics").fetchone()
    out["daily_metrics_span"] = [row[0], row[1]]
    return out


_DIGEST_QUERIES = (
    ("records",
     "SELECT metric, value, unit, start_utc, end_utc, start_local, local_date, "
     "source, origin, dedupe_key FROM records ORDER BY dedupe_key"),
    ("workouts",
     "SELECT workout_type, start_utc, end_utc, local_date, duration_min, "
     "energy_kcal, distance_mi, unit_distance, source, avg_heart_rate, "
     "max_heart_rate, dedupe_key FROM workouts ORDER BY dedupe_key"),
    ("daily_metrics",
     "SELECT metric, date, count, sum, avg, min, max, last, unit, source_kind "
     "FROM daily_metrics ORDER BY metric, date"),
    ("metric_source_months",
     "SELECT metric, month, source, n FROM metric_source_months "
     "ORDER BY metric, month, source"),
    ("subjective",
     "SELECT date, stress, soreness, energy, sleep_quality, caffeine_drinks, "
     "alcohol_drinks FROM subjective ORDER BY date"),
)


def content_digest(conn: sqlite3.Connection) -> str:
    """A stable digest of a vault's LOGICAL content.

    Not the file bytes: `vault_meta.created_at` is a wall clock and SQLite page
    layout is not a documented function of the inserts, so two logically
    identical vaults differ on disk. This hashes the rows instead, with floats
    rendered at a fixed precision, which is the thing a correctness gate
    actually wants to compare.
    """
    h = hashlib.sha256()
    for name, sql in _DIGEST_QUERIES:
        h.update(f"\n##{name}\n".encode())
        for row in conn.execute(sql):
            h.update("|".join(
                "" if v is None else
                (f"{v:.9g}" if isinstance(v, float) else str(v))
                for v in row).encode())
            h.update(b"\n")
    return h.hexdigest()


def digest_file(path: str | Path) -> str:
    """`content_digest` for a vault on disk (opened read-only)."""
    conn = db.connect(path, read_only=True)
    try:
        return content_digest(conn)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def format_report(report: dict) -> str:
    lines = [
        f"demo vault: {report['path']}",
        f"  window        {report['start_date']} .. {report['end_date']} "
        f"({report['days']} days, seed {report['seed']})",
        f"  watch from    {report['watch_from']} "
        f"(before it: phone-only instrument era)",
        f"  sources       {', '.join(report['sources'])}",
        f"  workouts      {', '.join(report['workout_types'])}",
        f"  metrics       {report['metrics']} distinct series in daily_metrics",
    ]
    for table, n in report["rows"].items():
        lines.append(f"  {table:<20} {n:>9,}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m health_advisor.demo",
        description="Generate a synthetic demo vault matching this repo's schema.")
    parser.add_argument("--out", "--db", dest="out", required=True,
                        help="destination SQLite path")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"days of history to generate (default: {DEFAULT_DAYS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed; fixes the content (default: {DEFAULT_SEED})")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE,
                        help=f"last local date (default: {DEFAULT_END_DATE})")
    parser.add_argument("--no-replace", action="store_true",
                        help="refuse to overwrite an existing file")
    parser.add_argument("--writable", action="store_true",
                        help="leave the vault writable (0644). By default it is "
                             "chmod 0444, matching how a vault is deployed.")
    parser.add_argument("--digest", action="store_true",
                        help="also print the logical content digest")
    args = parser.parse_args(argv)

    report = build_demo_vault(args.out, days=args.days, seed=args.seed,
                              end_date=args.end_date,
                              replace=not args.no_replace,
                              read_only=not args.writable)
    print(format_report(report))
    if args.digest:
        print(f"  digest              {digest_file(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
