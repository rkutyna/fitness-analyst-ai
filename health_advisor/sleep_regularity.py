"""Sleep timing regularity over the whole sleep-timing series.

WHAT THIS IS NOT. It is not the Sleep Regularity Index. Published SRI is built
from minute-level sleep/wake state and is scaled -100..+100. This reconstructs
one [bedtime, wake_time] interval per night, which fills sleep latency and all
wake-after-sleep-onset as sleep, misses naps and split sleep, and treats in-bed
as asleep. It is a different quantity and is named as one; reporting it as SRI
would invite comparison against population values that do not apply. See the
design spec, section 11.2.

Plan compliance is a separate three-band measure over the same bedtime
encoding: inside the 23:00 anchor, the ungraded 23:00-00:30 social-night band,
and past the 00:30 limit. Its canonical plan constants live in metrics.py.

The sleep-timing series is typically the largest coherent one in the DB and
runs continuously through gaps in watch coverage, which can make it the only
variable that links a much earlier era of a training history to the present
one.

ENCODING. From derive.py: sleep_bedtime and sleep_midpoint are hours since the
PREVIOUS day's noon (continuous across midnight); sleep_wake_time is hours
since midnight of the wake day; the row's date is the WAKE day. Reconciling
those two origins is the single easiest thing to get wrong here, and getting it
wrong shifts every interval by 12 hours while still producing plausible output.

Pairing is on local civil dates, never on elapsed hours: DST gives 23- and
25-hour days, and a naive 24-hour offset walks off the schedule at each
transition.

WINDOW ANCHORING, AND WHAT IT COSTS. Each night is scored inside the 24 hours
ENDING at noon on the wake day, so an ordinary night sits in the middle of its
window rather than straddling a boundary. The cost is that a daytime sleep
cannot be fully represented: an 11:00-19:00 interval falls almost entirely
outside the window anchored on the night it belongs to, and only its last hour
is scored. So this measure cannot return a near-zero score for a fully inverted
schedule — the floor in practice is around 60%. That is acceptable here (the
subject is a conventional night sleeper) but it means the number is a
discriminator between more and less regular, not an absolute on a 0-100 scale.
One more reason not to call it SRI.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
from scipy.optimize import curve_fit

from . import metrics as mx
from . import history
from .metrics import BEDTIME_ANCHOR_H, BEDTIME_SOCIAL_LIMIT_H

# Consecutive-night pairs needed to report at all — distinct from
# correlate.CORRELATION_MIN_PAIRS, a different quantity, must not be unified.
SLEEP_REGULARITY_MIN_PAIRS = 14
MINUTES_PER_DAY = 1440


def night_interval(day: str, bedtime_h: float, wake_h: float):
    """(sleep_start, sleep_end) as naive local datetimes for one wake day.

    `day` is the WAKE day. `bedtime_h` is hours since the previous day's noon,
    so 11.0 is 23:00 the night before and 12.5 is 00:30 on the wake day.
    `wake_h` is hours since midnight of the wake day.
    """
    d = date.fromisoformat(day)
    prev_noon = datetime(d.year, d.month, d.day, 12, 0) - timedelta(days=1)
    return (prev_noon + timedelta(hours=float(bedtime_h)),
            datetime(d.year, d.month, d.day) + timedelta(hours=float(wake_h)))


def _asleep_mask(day: str, bedtime_h: float, wake_h: float) -> np.ndarray:
    """Minute-resolution asleep/awake mask for the 24h ending at noon on `day`.

    The window runs noon-to-noon so a normal night sits in the middle of it
    rather than straddling a boundary.
    """
    d = date.fromisoformat(day)
    window_start = datetime(d.year, d.month, d.day, 12, 0) - timedelta(days=1)
    start, end = night_interval(day, bedtime_h, wake_h)
    mask = np.zeros(MINUTES_PER_DAY, dtype=bool)
    lo = int((start - window_start).total_seconds() // 60)
    hi = int((end - window_start).total_seconds() // 60)
    lo, hi = max(0, lo), min(MINUTES_PER_DAY, hi)
    if hi > lo:
        mask[lo:hi] = True
    return mask


def interval_regularity(nights) -> dict:
    """Percentage of minutes whose asleep/awake state matches the state 24h
    later, over consecutive-night pairs.

    `nights` is [(wake_day_iso, bedtime_h, wake_h)]. Only pairs of nights one
    civil day apart are compared — a gap in the record is not a pair, and
    comparing across it would invent a schedule that was never observed.
    """
    by_day = {}
    for day, bed, wake in nights:
        if bed is None or wake is None:
            continue
        by_day[day] = _asleep_mask(day, bed, wake)

    matches, n_pairs = [], 0
    for day in sorted(by_day):
        nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        if nxt not in by_day:
            continue
        n_pairs += 1
        matches.append(float((by_day[day] == by_day[nxt]).mean()))

    if n_pairs < SLEEP_REGULARITY_MIN_PAIRS:
        return {"status": "insufficient_nights",
                "reason": f"{n_pairs} consecutive-night pair(s); need "
                          f"{SLEEP_REGULARITY_MIN_PAIRS}",
                "n_pairs": n_pairs}
    return {"status": "ok",
            "match_pct": mx.r(float(np.mean(matches)) * 100.0),
            "n_pairs": n_pairs}


def nights_from_db(conn, start: str, end: str):
    """[(wake_day, bedtime_h, wake_h)] from daily_metrics, wake day ordered."""
    rows = conn.execute(
        "SELECT b.date AS day, b.last AS bed, w.last AS wake "
        "  FROM daily_metrics b "
        "  JOIN daily_metrics w ON w.date = b.date AND w.metric = 'sleep_wake_time' "
        " WHERE b.metric = 'sleep_bedtime' AND b.date BETWEEN ? AND ? "
        " ORDER BY b.date", (start, end)).fetchall()
    return [(r["day"], r["bed"], r["wake"]) for r in rows
            if r["bed"] is not None and r["wake"] is not None]


MIN_CALIBRATION_DAYS = 10


def agreement_from_pairs(pairs) -> dict:
    """Bland-Altman style agreement between proxy and staged sleep minutes.

    `pairs` is [(proxy_minutes, staged_minutes)]. Positive bias means the
    interval proxy counts MORE sleep than the stages do, which is the expected
    direction — the proxy fills sleep latency and wake-after-sleep-onset. But
    the direction is measured, not assumed: an earlier draft of the spec
    asserted an upward bias as if it were established, and it was a hypothesis.
    """
    if len(pairs) < MIN_CALIBRATION_DAYS:
        return {"status": "insufficient_overlap",
                "reason": f"{len(pairs)} day(s) with both measures; need "
                          f"{MIN_CALIBRATION_DAYS}",
                "n_days": len(pairs)}
    proxy = np.array([p for p, _ in pairs], dtype=float)
    staged = np.array([s for _, s in pairs], dtype=float)
    diff = proxy - staged
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    return {
        "status": "ok",
        "n_days": len(pairs),
        "mean_proxy_sleep_minutes": mx.r(proxy.mean(), 1),
        "mean_staged_sleep_minutes": mx.r(staged.mean(), 1),
        "mean_bias_minutes": mx.r(bias, 1),
        "bias_sd_minutes": mx.r(sd, 1),
        "limits_of_agreement": [mx.r(bias - 1.96 * sd, 1), mx.r(bias + 1.96 * sd, 1)],
    }


def calibrate_against_stages(conn, start: str, end: str) -> dict:
    """agreement_from_pairs over every day carrying both the reconstructed
    interval and a sleep_asleep total."""
    rows = conn.execute(
        "SELECT b.date AS day, b.last AS bed, w.last AS wake, a.sum AS staged "
        "  FROM daily_metrics b "
        "  JOIN daily_metrics w ON w.date = b.date AND w.metric = 'sleep_wake_time' "
        "  JOIN daily_metrics a ON a.date = b.date AND a.metric = 'sleep_asleep' "
        " WHERE b.metric = 'sleep_bedtime' AND b.date BETWEEN ? AND ? "
        " ORDER BY b.date", (start, end)).fetchall()
    pairs = []
    for r in rows:
        if r["bed"] is None or r["wake"] is None or r["staged"] is None:
            continue
        s, e = night_interval(r["day"], r["bed"], r["wake"])
        proxy_min = (e - s).total_seconds() / 60.0
        if proxy_min <= 0:
            continue
        # sleep_asleep is stored in MINUTES (normalize.CATALOG unit 'min',
        # agg 'sum'; the live mean is 459 min/day). No conversion.
        pairs.append((proxy_min, float(r["staged"])))
    return agreement_from_pairs(pairs)


MIN_WINDOW_DAYS = 21     # of WINDOW_DAYS present before an SD is reportable
WINDOW_DAYS = 28
MIN_COSINOR_DAYS = 60


def rolling_sd(vals) -> float | None:
    """SD of a window of sleep midpoints, in hours. None if too thin."""
    clean = [v for v in vals if v is not None]
    if len(clean) < MIN_WINDOW_DAYS:
        return None
    return float(np.std(np.array(clean, dtype=float), ddof=1))


def midpoint_variability(conn, start: str, end: str) -> dict:
    """Rolling 28-day SD of sleep_midpoint. Lower is more regular."""
    rows = conn.execute(
        "SELECT date, last AS v FROM daily_metrics "
        "WHERE metric = 'sleep_midpoint' "
        "AND date > date(?, '-28 days') AND date <= ? "
        "AND last IS NOT NULL ORDER BY date", (start, end)).fetchall()
    if len(rows) < MIN_WINDOW_DAYS:
        return {"status": "insufficient_window",
                "reason": f"{len(rows)} day(s) of sleep_midpoint; need "
                          f"{MIN_WINDOW_DAYS}",
                "days": []}
    dates = [r["date"] for r in rows if start <= r["date"] <= end]
    out = []
    for day in dates:
        window_rows = conn.execute(
            "SELECT last AS v FROM daily_metrics "
            "WHERE metric = 'sleep_midpoint' "
            "AND date > date(?, '-28 days') AND date <= ? "
            "AND last IS NOT NULL ORDER BY date", (day, day)).fetchall()
        window = [r["v"] for r in window_rows]
        sd = rolling_sd(window)
        if sd is not None:
            out.append({"date": day, "sd_hours": mx.r(sd, 3)})
    if not out:
        return {"status": "insufficient_window",
                "reason": "no window reached the minimum day count", "days": []}
    return {"status": "ok", "days": out, "latest_sd_hours": out[-1]["sd_hours"],
            "field_metrics": {"latest_sd_hours": "sleep_midpoint_sd_28d"}}


def _cosinor_model(t, mesor, amp, phase, drift):
    return mesor + drift * t / 365.25 + amp * np.sin(2 * np.pi * t / 365.25 + phase)


def cosinor(dates, values) -> dict:
    """Annual cosinor fit of a midpoint series: level, seasonal amplitude, and
    linear drift in hours per year.

    Drift is the part that matters for a decade-long series — it is bedtime
    creep, separated from the seasonal swing rather than confounded with it.
    """
    clean = sorted(((d, v) for d, v in zip(dates, values) if v is not None),
                   key=lambda item: item[0])
    eras = history.split_eras([d for d, _ in clean], [v for _, v in clean])
    if not eras:
        return {"status": "insufficient_days", "reason": "0 day(s); need 60",
                "n_days": 0, "eras_found": 0}
    era_dates, era_values = max(eras, key=lambda era: len(era[0]))
    if len(era_dates) < MIN_COSINOR_DAYS:
        return {"status": "insufficient_days",
                "reason": f"{len(era_dates)} day(s); need {MIN_COSINOR_DAYS}",
                "n_days": len(era_dates), "eras_found": len(eras)}
    d0 = date.fromisoformat(era_dates[0])
    t = np.array([(date.fromisoformat(d) - d0).days for d in era_dates], dtype=float)
    y = np.array(era_values, dtype=float)
    try:
        popt, _ = curve_fit(_cosinor_model, t, y,
                            p0=[float(y.mean()), 0.5, 0.0, 0.0], maxfev=20000)
    except (RuntimeError, ValueError) as e:
        return {"status": "fit_failed", "reason": str(e),
                "n_days": len(era_dates), "eras_found": len(eras)}
    mesor, amp, phase, drift = popt
    if amp < 0:
        amp = -amp
        phase += np.pi
    return {
        "status": "ok",
        "n_days": len(era_dates),
        "era_start": era_dates[0],
        "era_end": era_dates[-1],
        "era_days": len(era_dates),
        "eras_found": len(eras),
        "mesor": mx.r(mesor, 3),
        "amplitude": mx.r(amp, 3),
        "acrophase_days": mx.r(float((np.pi / 2 - phase) % (2 * np.pi)
                                      / (2 * np.pi) * 365.25), 1),
        "drift_hours_per_year": mx.r(drift, 3),
    }


def compliance_from_bedtimes(bedtimes) -> dict:
    """Count the plan's bedtime bands. Pure, for testing.

    The social-night band is deliberately not classified as either a hit or a
    failure, so there is no single "compliance" figure to report and the key is
    named for what it actually counts: ``inside_anchor_pct`` is the share of
    nights at or before the anchor. The remainder is NOT the complement — it is
    the social band plus the past-limit band, and only the second of those is a
    shortfall. A caller that reads one percentage and infers the rest has
    re-created the defect this function was rewritten to remove.
    """
    clean = [b for b in bedtimes if b is not None]
    if not clean:
        return {
            "n_nights": 0,
            "inside_anchor": 0,
            "social_nights": 0,
            "past_limit": 0,
            "inside_anchor_pct": None,
            "anchor_h": BEDTIME_ANCHOR_H,
            "social_limit_h": BEDTIME_SOCIAL_LIMIT_H,
        }
    inside = sum(1 for b in clean if b <= BEDTIME_ANCHOR_H)
    social = sum(1 for b in clean
                 if BEDTIME_ANCHOR_H < b <= BEDTIME_SOCIAL_LIMIT_H)
    past = sum(1 for b in clean if b > BEDTIME_SOCIAL_LIMIT_H)
    return {
        "n_nights": len(clean),
        "inside_anchor": inside,
        "social_nights": social,
        "past_limit": past,
        "inside_anchor_pct": mx.r(inside / len(clean) * 100.0),
        "anchor_h": BEDTIME_ANCHOR_H,
        "social_limit_h": BEDTIME_SOCIAL_LIMIT_H,
    }


def plan_compliance(conn, start: str, end: str) -> dict:
    """Report the inside-anchor, social-night, and past-limit bands."""
    rows = conn.execute(
        "SELECT last AS v FROM daily_metrics WHERE metric = 'sleep_bedtime' "
        "AND date BETWEEN ? AND ? AND last IS NOT NULL", (start, end)).fetchall()
    return compliance_from_bedtimes([r["v"] for r in rows])
