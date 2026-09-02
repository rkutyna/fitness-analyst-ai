"""Long-horizon, era-aware history analysis.

The daily table is not one continuous observation of every metric.  A watch
metric can disappear for years and then reappear, so a statistic over the
whole date range would silently compare two unrelated observation eras.  This
module makes era provenance part of every returned statistic and leaves prose
or judgement to its caller.

Dates are local calendar dates, as they are in ``daily_metrics``.  An era uses
the following deliberately conservative seam rule: observations whose dates
are more than 14 days apart start a new era.  A missed week therefore remains
inside an era, while the 2023--2025 watch-data void does not.  The threshold
is configurable for tests or a metric with unusually sparse expected data,
but callers should retain the default unless they have a reason not to.

Reference values use the median and the 5th/95th percentiles when an era has
at least five observations.  The raw minimum and maximum are intentionally
not emitted, and tiny eras have no percentile range: one artifact must not
become a personal record.  A sustained reference is the best 30-calendar-day rolling
median with at least 21 observed days (70% coverage).  Direction is not
present in the current shared metric catalog, so direction-sensitive APIs
require ``direction='higher'`` or ``direction='lower'`` explicitly.  That
intentional error on omission is safer than guessing that lower is better for
one metric and higher for another.

The trajectory defaults to calendar quarters.  Its aggregate follows the
metric catalog's aggregation (sum, mean, or last), and a bucket with less
than 70% calendar-day coverage has ``value=None`` rather than pretending that
its partial total is a real low.  Sustained periods use the same 30-day,
70%-coverage rolling rule and require at least 80% of observed values to meet
the caller's threshold.  These names describe measurements only; this module
does not emit success/failure or past-versus-present judgements.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
import numpy as np

from . import analysis as an
from . import metrics as mx
from . import normalize as nz


DEFAULT_GAP_DAYS = 14
REFERENCE_FLOOR_PERCENTILE = 5
REFERENCE_CEILING_PERCENTILE = 95
MIN_PERCENTILE_SAMPLES = 5
SUSTAINED_WINDOW_DAYS = 30
SUSTAINED_MIN_DAYS = 21
SUSTAINED_MIN_FRACTION = 0.70
SUSTAINED_THRESHOLD_FRACTION = 0.80
TRAJECTORY_MIN_DENSITY = 0.70


def _check_gap(gap_days: int) -> int:
    if isinstance(gap_days, bool) or not isinstance(gap_days, int) or gap_days < 1:
        raise ValueError("gap_days must be a positive integer")
    return gap_days


def _check_fraction(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be between 0 and 1") from None
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def split_eras(dates, values, gap_days: int = DEFAULT_GAP_DAYS):
    """Split paired observations at gaps larger than ``gap_days``."""
    _check_gap(gap_days)
    if not dates:
        return []
    eras = []
    current_dates = [dates[0]]
    current_values = [values[0]]
    for d, value, previous in zip(dates[1:], values[1:], dates):
        if (date.fromisoformat(d) - date.fromisoformat(previous)).days > gap_days:
            eras.append((current_dates, current_values))
            current_dates = []
            current_values = []
        current_dates.append(d)
        current_values.append(value)
    eras.append((current_dates, current_values))
    return eras


def _direction(metric: str, direction: str | None) -> str:
    """Resolve direction, consulting a future catalog field before failing.

    ``normalize.CATALOG`` currently has no direction field.  Supporting one
    here keeps this module aligned if it gains one later, while refusing to
    invent a direction today.  ``metric`` is included in the error because a
    caller wiring several references should be able to find the omission.
    """
    chosen = direction
    if chosen is None:
        chosen = nz.CATALOG.get(metric, {}).get("direction")
    if chosen is None:
        raise ValueError(
            f"direction is required for {metric!r}; pass 'higher' or 'lower'"
        )
    chosen = str(chosen).strip().lower()
    aliases = {"high": "higher", "max": "higher", "up": "higher",
               "low": "lower", "min": "lower", "down": "lower"}
    chosen = aliases.get(chosen, chosen)
    if chosen not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    return chosen


def _date_span(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days + 1


def _provenance(value, era_no: int, start: str, end: str, n: int, **extra) -> dict:
    out = {
        "value": mx.r(value),
        "era": era_no,
        "start": start,
        "end": end,
        "n": int(n),
    }
    out.update(extra)
    return out


def _history_series(conn, metric: str):
    """Read a series from either sqlite3.Row or default tuple connections."""
    col = mx.value_col(metric)
    rows = conn.execute(
        f"SELECT date, {col} AS v, unit FROM daily_metrics "
        "WHERE metric = ? AND " + col + " IS NOT NULL ORDER BY date",
        (metric,),
    ).fetchall()
    if not rows:
        return [], [], nz.canonical_unit(metric, None)

    def get(row, key: str, index: int):
        try:
            return row[key]
        except (IndexError, KeyError, TypeError):
            return row[index]

    dates = [get(row, "date", 0) for row in rows]
    values = [get(row, "v", 1) for row in rows]
    unit = get(rows[0], "unit", 2) or nz.canonical_unit(metric, None)
    return dates, values, unit


def _era_rows(conn, metric: str, gap_days: int = DEFAULT_GAP_DAYS):
    """Return ``(dates, values, era_number)`` rows without crossing seams."""
    _check_gap(gap_days)
    dates, vals, unit = _history_series(conn, metric)
    if not dates:
        return [], [], [], unit, {"status": "ok", "provenance": None,
                                  "boundaries": []}
    # A continuous daily series can still cross an instrument seam.  The
    # daily rollup has no source column, so consult provenance as a second
    # refusal boundary rather than averaging across it (F3-2; F3-1).  When
    # provenance is unavailable the eras are still returned — the gap-based
    # split is real on its own — but every one of them carries that fact, so a
    # reader is never told "one continuous era" when what is true is "one era as
    # far as we can see, and we cannot see instruments here".
    era_status = an.instrument_eras_status(conn, metric, dates[0], dates[-1])
    source_boundaries = era_status["boundaries"]
    eras = []
    for era_dates, era_values in split_eras(dates, [float(v) for v in vals], gap_days):
        current_dates = [era_dates[0]]
        current_values = [era_values[0]]
        for d, value, previous in zip(era_dates[1:], era_values[1:], era_dates):
            crossed_source = any(previous < boundary <= d for boundary in source_boundaries)
            if crossed_source:
                eras.append((current_dates, current_values))
                current_dates = []
                current_values = []
            current_dates.append(d)
            current_values.append(value)
        eras.append((current_dates, current_values))
    return eras, dates, vals, unit, era_status


def detect_eras(conn, metric: str, gap_days: int = DEFAULT_GAP_DAYS) -> list[dict]:
    """Segment one metric into contiguous coverage eras.

    ``n`` counts non-null daily values, while ``span_days`` is the inclusive
    calendar span from the first to last observation.  Density is therefore a
    coverage measure rather than an assertion that missing days had zero
    values.  Empty metrics return an empty list.
    """
    eras, _, _, unit, era_status = _era_rows(conn, metric, gap_days)
    out = []
    for number, (dates, values) in enumerate(eras, 1):
        start, end = dates[0], dates[-1]
        span_days = _date_span(start, end)
        out.append({
            "metric": metric,
            "era": number,
            "start": start,
            "end": end,
            "date_span": {"start": start, "end": end},
            "n": len(values),
            "span_days": span_days,
            "coverage_density": mx.r(len(values) / span_days, 3),
            "unit": unit,
            # Where the seam evidence came from, per era. "unavailable" means
            # this era may straddle an instrument change nobody can see — not
            # that it does not.
            "instrument_provenance": era_status["provenance"]
            if era_status["status"] == "ok" else "unavailable",
        })
    return out


def _best_sustained(dates: list[str], values: list[float], era_no: int,
                    direction: str, window_days: int, min_days: int,
                    min_fraction: float) -> dict | None:
    candidates = []
    for end_index, end_iso in enumerate(dates):
        end_day = date.fromisoformat(end_iso)
        if (end_day - date.fromisoformat(dates[0])).days + 1 < window_days:
            continue
        window_start = end_day - timedelta(days=window_days - 1)
        points = [
            (d, v) for d, v in zip(dates[:end_index + 1], values[:end_index + 1])
            if date.fromisoformat(d) >= window_start
        ]
        if len(points) < min_days or len(points) / window_days < min_fraction:
            continue
        candidate = float(np.median([v for _, v in points]))
        candidates.append((candidate, points))
    if not candidates:
        return None
    if direction == "higher":
        best_value = max(item[0] for item in candidates)
    else:
        best_value = min(item[0] for item in candidates)
    # A tied best level is represented by its earliest sustained window, so
    # output remains deterministic without introducing a second preference.
    points = min(item[1] for item in candidates if item[0] == best_value)
    return _provenance(
        best_value, era_no, points[0][0], points[-1][0], len(points),
        window_days=window_days,
        coverage_density=mx.r(len(points) / window_days, 3),
    )


def reference_ranges(conn, metric: str, *, direction: str | None = None,
                     gap_days: int = DEFAULT_GAP_DAYS,
                     sustained_window_days: int = SUSTAINED_WINDOW_DAYS,
                     sustained_min_days: int = SUSTAINED_MIN_DAYS,
                     sustained_min_fraction: float = SUSTAINED_MIN_FRACTION) -> dict:
    """Return robust, per-era achieved references for ``metric``.

    The central value is a median; floor and ceiling are the 5th and 95th
    percentiles.  ``direction`` controls the best sustained reference and is
    required unless the catalog supplies a direction.  Every statistic has
    ``era``, ``start``, ``end``, and ``n`` provenance fields.
    """
    chosen = _direction(metric, direction)
    if isinstance(sustained_window_days, bool) or sustained_window_days < 1:
        raise ValueError("sustained_window_days must be positive")
    if isinstance(sustained_min_days, bool) or sustained_min_days < 1:
        raise ValueError("sustained_min_days must be positive")
    min_fraction = _check_fraction(sustained_min_fraction, "sustained_min_fraction")
    eras, _, _, unit, _status = _era_rows(conn, metric, gap_days)
    result = {
        "metric": metric,
        "unit": unit,
        "direction": chosen,
        "gap_days": gap_days,
        "eras": [],
    }
    for era_no, (dates, values) in enumerate(eras, 1):
        arr = np.asarray(values, dtype=float)
        start, end = dates[0], dates[-1]
        common = {"era_no": era_no, "start": start, "end": end, "n": len(values)}
        sustained = _best_sustained(
            dates, values, era_no, chosen, int(sustained_window_days),
            int(sustained_min_days), min_fraction,
        )
        floor = ceiling = None
        if len(values) >= MIN_PERCENTILE_SAMPLES:
            floor = _provenance(
                np.percentile(arr, REFERENCE_FLOOR_PERCENTILE), **common,
                percentile=REFERENCE_FLOOR_PERCENTILE,
            )
            ceiling = _provenance(
                np.percentile(arr, REFERENCE_CEILING_PERCENTILE), **common,
                percentile=REFERENCE_CEILING_PERCENTILE,
            )
        result["eras"].append({
            "metric": metric,
            "era": era_no,
            "start": start,
            "end": end,
            "n": len(values),
            "central": _provenance(np.median(arr), **common),
            "floor": floor,
            "ceiling": ceiling,
            "best_sustained": sustained,
        })
    return result


def _bucket_bounds(day: date, bucket: str) -> tuple[date, date]:
    bucket = bucket.strip().lower()
    if bucket == "month":
        start = day.replace(day=1)
        return start, day.replace(day=monthrange(day.year, day.month)[1])
    if bucket == "quarter":
        month = ((day.month - 1) // 3) * 3 + 1
        start = date(day.year, month, 1)
        end_month = month + 2
        return start, date(day.year, end_month, monthrange(day.year, end_month)[1])
    raise ValueError("bucket must be 'month' or 'quarter'")


def trajectory(conn, metric: str, *, bucket: str = "quarter",
               gap_days: int = DEFAULT_GAP_DAYS,
               min_density: float = TRAJECTORY_MIN_DENSITY) -> dict:
    """Return compact, coverage-qualified calendar aggregates over all eras.

    Buckets are kept separate at era boundaries even if a calendar bucket
    happens to contain observations from both sides.  Sparse buckets retain
    their count and density but expose no aggregate ``value``.
    """
    density_limit = _check_fraction(min_density, "min_density")
    eras, _, _, unit, _status = _era_rows(conn, metric, gap_days)
    points = []
    agg = mx.agg(metric)
    for era_no, (dates, values) in enumerate(eras, 1):
        grouped: dict[tuple[date, date], list[float]] = {}
        grouped_dates: dict[tuple[date, date], list[str]] = {}
        for d, value in zip(dates, values):
            bounds = _bucket_bounds(date.fromisoformat(d), bucket)
            grouped.setdefault(bounds, []).append(value)
            grouped_dates.setdefault(bounds, []).append(d)
        for (bucket_start, bucket_end), bucket_values in sorted(grouped.items()):
            bucket_dates = grouped_dates[(bucket_start, bucket_end)]
            n = len(bucket_values)
            density = n / _date_span(bucket_start.isoformat(), bucket_end.isoformat())
            if agg == "sum":
                raw_value = float(np.sum(bucket_values))
            elif agg == "last":
                raw_value = float(bucket_values[-1])
            else:
                raw_value = float(np.mean(bucket_values))
            points.append({
                "metric": metric,
                "era": era_no,
                "bucket_start": bucket_start.isoformat(),
                "bucket_end": bucket_end.isoformat(),
                "start": bucket_dates[0],
                "end": bucket_dates[-1],
                "n": n,
                "coverage_density": mx.r(density, 3),
                "value": mx.r(raw_value) if density >= density_limit else None,
                "status": "covered" if density >= density_limit else "sparse",
            })
    return {
        "metric": metric,
        "unit": unit,
        "bucket": bucket.strip().lower(),
        "gap_days": gap_days,
        "min_density": density_limit,
        "points": points,
    }


def _qualifies(value: float, threshold: float, direction: str) -> bool:
    return value >= threshold if direction == "higher" else value <= threshold


def sustained_periods(conn, metric: str, *, threshold: float,
                      direction: str | None = None,
                      gap_days: int = DEFAULT_GAP_DAYS,
                      window_days: int = SUSTAINED_WINDOW_DAYS,
                      min_days: int = SUSTAINED_MIN_DAYS,
                      min_fraction: float = SUSTAINED_MIN_FRACTION,
                      threshold_fraction: float = SUSTAINED_THRESHOLD_FRACTION) -> dict:
    """Find rolling periods meeting an explicit activity threshold.

    Each 30-day candidate must have at least ``min_days`` observations and
    ``min_fraction`` calendar coverage; at least ``threshold_fraction`` of
    those observations must meet the threshold.  Overlapping candidates in
    one era are merged.  ``typical`` is the median of all observations in the
    merged period and carries the period's era/date-span/sample provenance.
    """
    chosen = _direction(metric, direction)
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1:
        raise ValueError("window_days must be a positive integer")
    if isinstance(min_days, bool) or not isinstance(min_days, int) or min_days < 1:
        raise ValueError("min_days must be a positive integer")
    coverage_fraction = _check_fraction(min_fraction, "min_fraction")
    qualifying_fraction = _check_fraction(threshold_fraction, "threshold_fraction")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("threshold must be numeric") from None
    eras, _, _, unit, _status = _era_rows(conn, metric, gap_days)
    periods = []
    for era_no, (dates, values) in enumerate(eras, 1):
        candidates = []
        for end_iso in dates:
            end_day = date.fromisoformat(end_iso)
            if (end_day - date.fromisoformat(dates[0])).days + 1 < window_days:
                continue
            window_start = end_day - timedelta(days=window_days - 1)
            points = [
                (d, v) for d, v in zip(dates, values)
                if window_start <= date.fromisoformat(d) <= end_day
            ]
            if len(points) < min_days or len(points) / window_days < coverage_fraction:
                continue
            qualifying = sum(_qualifies(v, threshold, chosen) for _, v in points)
            if qualifying / len(points) >= qualifying_fraction:
                candidates.append((window_start, end_day))
        if not candidates:
            continue
        merged = []
        for start, end in candidates:
            if merged and start <= merged[-1][1] + timedelta(days=1):
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for start, end in merged:
            points = [
                (d, v) for d, v in zip(dates, values)
                if start <= date.fromisoformat(d) <= end
            ]
            qualifying = sum(_qualifies(v, threshold, chosen) for _, v in points)
            actual_start, actual_end = points[0][0], points[-1][0]
            periods.append({
                "metric": metric,
                "era": era_no,
                "start": actual_start,
                "end": actual_end,
                "n": len(points),
                "coverage_density": mx.r(len(points) / ((end - start).days + 1), 3),
                "qualifying_n": qualifying,
                "qualifying_fraction": mx.r(qualifying / len(points), 3),
                "threshold": mx.r(threshold),
                "direction": chosen,
                "typical": _provenance(
                    np.median([v for _, v in points]), era_no,
                    actual_start, actual_end, len(points),
                ),
            })
    return {
        "metric": metric,
        "unit": unit,
        "direction": chosen,
        "threshold": mx.r(threshold),
        "window_days": window_days,
        "min_days": min_days,
        "min_fraction": coverage_fraction,
        "threshold_fraction": qualifying_fraction,
        "gap_days": gap_days,
        "periods": periods,
    }


# A descriptive alias for callers that prefer the task's terminology.
long_arc_trajectory = trajectory


__all__ = [
    "DEFAULT_GAP_DAYS", "detect_eras", "reference_ranges", "trajectory",
    "long_arc_trajectory", "sustained_periods",
]
