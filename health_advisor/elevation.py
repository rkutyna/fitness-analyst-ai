"""Versioned elevation gain/loss from route altitude samples.

The public computation is deliberately independent of route storage.  It takes
timestamped samples, filters by vertical accuracy, smooths by time, and then
applies the version-1 hysteresis rule.  ``route_climb`` is only a small reader
for the optional route table.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import math
import sqlite3
import struct
from typing import Any


# Version-1 method constants.  Keep these together: changing one changes the
# meaning of every figure produced by this module.
METHOD_VERSION = 1
ROLLING_MEDIAN_WINDOW_SECONDS = 15.0
HYSTERESIS_THRESHOLD_M = 3.0
MIN_USED_POINTS = 10
METRES_TO_FEET = 3.280839895013123

# Bump when a matcher or ingest change can fill entries an earlier backfill
# could not. Generation 1 was the original backfill; generation 2 adds the
# UUID-to-overlap fallback for workouts stored without a UUID.
WORKOUT_ELEVATION_BACKFILL_GENERATION = 2

__all__ = ["compute_elevation", "route_climb", "workout_climb",
           "WORKOUT_ELEVATION_BACKFILL_GENERATION"]


def _timestamp(value: Any) -> float:
    """Convert supported timestamp values to seconds on a common time scale."""
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        result = dt.timestamp()
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        result = dt.timestamp()
    else:
        raise TypeError(f"unsupported timestamp type: {type(value).__name__}")
    if not math.isfinite(result):
        raise ValueError("timestamps must be finite")
    return result


def _sample_value(sample: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in sample:
            return sample[name]
    joined = ", ".join(names)
    raise KeyError(f"sample is missing one of: {joined}")


def _normalise_samples(
    samples_or_times: Iterable[Any],
    altitudes: Iterable[Any] | None,
    vertical_accuracy_m: Iterable[Any] | None,
) -> tuple[list[tuple[float, float, float | None]], bool, int]:
    """Return filtered rows, whether every input had accuracy, and input count."""
    if altitudes is None and isinstance(samples_or_times, Mapping):
        # Also accept the natural array bundle used by the route decoder:
        # ``{"t_offset_s": [...], "altitude_m": [...], ...}``.
        time_values = _sample_value(
            samples_or_times, "t", "time", "timestamp", "t_offset_s")
        altitude_values = _sample_value(
            samples_or_times, "altitude_m", "altitude", "elevation_m")
        if isinstance(time_values, Iterable) and not isinstance(time_values, (str, bytes)):
            accuracy_values = next(
                (samples_or_times[name] for name in
                 ("vertical_accuracy_m", "vertical_accuracy", "accuracy_m")
                 if name in samples_or_times),
                None,
            )
            return _normalise_samples(time_values, altitude_values, accuracy_values)
        samples_or_times = [samples_or_times]

    if altitudes is not None:
        times = list(samples_or_times)
        elevations = list(altitudes)
        if len(times) != len(elevations):
            raise ValueError("times and altitudes must have equal length")
        if vertical_accuracy_m is None:
            accuracies: list[Any] = [None] * len(times)
        else:
            accuracies = list(vertical_accuracy_m)
            if len(accuracies) != len(times):
                raise ValueError("vertical accuracy must match sample count")
        rows = list(zip(times, elevations, accuracies))
    else:
        rows = []
        for sample in samples_or_times:
            if isinstance(sample, Mapping):
                time_value = _sample_value(sample, "t", "time", "timestamp", "t_offset_s")
                altitude_value = _sample_value(sample, "altitude_m", "altitude", "elevation_m")
                accuracy_value = next(
                    (sample[name] for name in
                     ("vertical_accuracy_m", "vertical_accuracy", "accuracy_m")
                     if name in sample),
                    None,
                )
            else:
                try:
                    size = len(sample)
                except TypeError as exc:
                    raise TypeError("samples must be mappings or sequences") from exc
                if size not in (2, 3):
                    raise ValueError("a sample sequence must contain 2 or 3 values")
                time_value, altitude_value = sample[0], sample[1]
                accuracy_value = sample[2] if size == 3 else None
            rows.append((time_value, altitude_value, accuracy_value))

    n_points = len(rows)
    normalised: list[tuple[float, float, float | None]] = []
    accuracy_presence: list[bool] = []
    for time_value, altitude_value, accuracy_value in rows:
        timestamp = _timestamp(time_value)
        altitude = float(altitude_value)
        if not math.isfinite(altitude):
            raise ValueError("altitudes must be finite")
        accuracy = None if accuracy_value is None else float(accuracy_value)
        if accuracy is not None and not math.isfinite(accuracy):
            # A non-finite accuracy cannot establish that a sample is accurate.
            # Treat it as absent, matching the no-accuracy retention rule.
            accuracy = None
        accuracy_presence.append(accuracy is not None)
        if accuracy is not None and (accuracy < 0.0 or accuracy > 10.0):
            continue
        normalised.append((timestamp, altitude, accuracy))
    normalised.sort(key=lambda row: row[0])
    all_have_accuracy = bool(rows) and all(accuracy_presence)
    return normalised, all_have_accuracy, n_points


def _rolling_medians(rows: Sequence[tuple[float, float, float | None]]) -> list[float]:
    """Centred timestamp median in O(n log n).

    ``window`` contains exactly the samples whose timestamps are within half
    the named 15-second window of the current timestamp.  The sorted value
    list makes insertion, removal, and median lookup logarithmic/constant
    respectively; the two timestamp pointers only move forward.
    """
    half_window = ROLLING_MEDIAN_WINDOW_SECONDS / 2.0
    times = [row[0] for row in rows]
    values = [row[1] for row in rows]
    sorted_window: list[float] = []
    medians: list[float] = []
    left = 0
    right = 0

    for center, timestamp in enumerate(times):
        lower = timestamp - half_window
        upper = timestamp + half_window
        while left < len(rows) and times[left] < lower:
            old_value = values[left]
            position = bisect_left(sorted_window, old_value)
            del sorted_window[position]
            left += 1
        while right < len(rows) and times[right] <= upper:
            insort(sorted_window, values[right])
            right += 1
        # The current sample is always in the window, including duplicate-time
        # samples, so this cannot be empty for a valid center.
        middle = len(sorted_window) // 2
        if len(sorted_window) % 2:
            medians.append(sorted_window[middle])
        else:
            medians.append((sorted_window[middle - 1] + sorted_window[middle]) / 2.0)
    return medians


def _validate_state(initial_state: tuple[float, int] | None) -> tuple[float, int] | None:
    if initial_state is None:
        return None
    if not isinstance(initial_state, tuple) or len(initial_state) != 2:
        raise ValueError("initial_state must be the (ext, d) tuple returned by compute_elevation")
    ext, direction = float(initial_state[0]), int(initial_state[1])
    if not math.isfinite(ext) or direction not in (-1, 0, 1):
        raise ValueError("initial_state must contain a finite ext and d in {-1, 0, 1}")
    return ext, direction


def _hysteresis(altitudes: Sequence[float], initial_state: tuple[float, int] | None) -> tuple[float, float, tuple[float, int]]:
    """Accumulate version-1 gain/loss and return the final ``(ext, d)`` state."""
    if initial_state is None:
        ext = float(altitudes[0])
        direction = 0
    else:
        ext, direction = initial_state
    ascent = 0.0
    descent = 0.0
    threshold = HYSTERESIS_THRESHOLD_M

    for altitude in altitudes if initial_state is not None else altitudes[1:]:
        if direction == 0:
            if altitude - ext >= threshold:
                ascent += altitude - ext
                ext = altitude
                direction = 1
            elif ext - altitude >= threshold:
                descent += ext - altitude
                ext = altitude
                direction = -1
        elif direction == 1:
            if altitude > ext:
                ascent += altitude - ext
                ext = altitude
            elif ext - altitude >= threshold:
                # The entire reversal move counts.  The threshold confirms the
                # turn; it is not subtracted from the physical descent.
                descent += ext - altitude
                ext = altitude
                direction = -1
        else:
            if altitude < ext:
                descent += ext - altitude
                ext = altitude
            elif altitude - ext >= threshold:
                ascent += altitude - ext
                ext = altitude
                direction = 1
    return ascent, descent, (ext, direction)


def compute_elevation(
    samples_or_times: Iterable[Any],
    altitudes: Iterable[Any] | None = None,
    vertical_accuracy_m: Iterable[Any] | None = None,
    *,
    initial_state: tuple[float, int] | None = None,
    return_state: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], tuple[float, int] | None]:
    """Compute version-1 elevation figures from timestamped altitude samples.

    Pass either samples such as ``{"t": seconds, "altitude_m": metres,
    "vertical_accuracy_m": metres}`` or three parallel iterables
    ``times, altitudes, vertical_accuracy_m``.  Timestamps may be numeric
    seconds or ISO-8601 strings.  Samples are sorted by timestamp before the
    centred 15-second median is calculated.

    ``initial_state`` and ``return_state=True`` support a split route.  The
    state is exactly ``(ext, d)``.  Smoothing is done independently per call,
    so a split loses the median window across its boundary; callers should
    bound that seam effect (the test suite does).  The ordinary result has no
    state field, keeping its persisted shape stable.
    """
    state = _validate_state(initial_state)
    rows, all_have_accuracy, n_points = _normalise_samples(
        samples_or_times, altitudes, vertical_accuracy_m)
    medians = _rolling_medians(rows)
    result: dict[str, Any] = {
        "ascent_m": 0.0,
        "descent_m": 0.0,
        "net_m": 0.0,
        "start_alt_m": None,
        "end_alt_m": None,
        "min_alt_m": None,
        "max_alt_m": None,
        "n_points": n_points,
        "n_used": len(rows),
        "accuracy_filtered": all_have_accuracy,
        "method_version": METHOD_VERSION,
        "ascent_ft": 0.0,
        "descent_ft": 0.0,
    }
    final_state: tuple[float, int] | None = state
    if medians:
        ascent, descent, final_state = _hysteresis(medians, state)
        result.update({
            "ascent_m": ascent,
            "descent_m": descent,
            "net_m": medians[-1] - medians[0],
            "start_alt_m": medians[0],
            "end_alt_m": medians[-1],
            "min_alt_m": min(medians),
            "max_alt_m": max(medians),
            "ascent_ft": ascent * METRES_TO_FEET,
            "descent_ft": descent * METRES_TO_FEET,
        })
    if return_state:
        return result, final_state
    return result


def _decode_points(blob: bytes | bytearray | memoryview) -> tuple[list[float], ...]:
    raw = bytes(blob)
    if len(raw) == 0 or len(raw) % 16:
        raise ValueError("route points must contain four equal float32 arrays")
    count = len(raw) // 16
    values = struct.unpack(f"<{count * 4}f", raw)
    return tuple(list(values[offset * count:(offset + 1) * count])
                 for offset in range(4))


def _window_samples(route_rows, start_utc: str,
                    end_utc: str) -> list[tuple[float, float, float]]:
    """Decode ``(start_utc, points)`` route rows into samples inside the window.

    The window is half-open. Callers choose which routes to pass; this only
    turns their packed points into ``(timestamp, altitude, accuracy)``.
    """
    window_start = _timestamp(start_utc)
    window_end = _timestamp(end_utc)
    samples: list[tuple[float, float, float]] = []
    for route_start_utc, points in route_rows:
        offsets, elevations, accuracies, _horizontal = _decode_points(points)
        route_start = _timestamp(route_start_utc)
        for offset, altitude, accuracy in zip(offsets, elevations, accuracies):
            timestamp = route_start + offset
            if window_start <= timestamp < window_end:
                samples.append((timestamp, altitude, accuracy))
    return samples


def route_climb(conn: sqlite3.Connection, start_utc: str, end_utc: str) -> dict[str, Any]:
    """Compute elevation for route points in the half-open UTC window.

    This adapter tolerates an older database with no ``workout_routes`` table
    by returning ``{"status": "no_route"}``.  When present, it assumes
    ``workout_routes(start_utc, end_utc, encoding, points BLOB)``.  ``points``
    contains four sequential little-endian float32 arrays of equal length:
    ``t_offset_s, altitude_m, vertical_accuracy_m, horizontal_accuracy_m``.
    The horizontal-accuracy array is intentionally ignored by this elevation
    computation.  ``end_utc`` is exclusive, as it is for other UTC windows in
    the package.
    """
    if start_utc >= end_utc:
        raise ValueError("end_utc must be after start_utc")
    present = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'workout_routes'"
    ).fetchone()
    if present is None:
        return {"status": "no_route"}

    rows = conn.execute(
        "SELECT start_utc, points FROM workout_routes "
        "WHERE start_utc < ? AND end_utc > ? ORDER BY start_utc",
        (end_utc, start_utc),
    ).fetchall()
    samples = _window_samples(rows, start_utc, end_utc)
    if not samples:
        return {"status": "no_route"}
    result = compute_elevation(samples)
    if result["n_used"] < MIN_USED_POINTS:
        return {"status": "too_few_points"}
    return result


def workout_climb(conn: sqlite3.Connection, workout_id: int) -> dict[str, Any]:
    """Read the authoritative climb for one workout.

    Device totals win whenever either stored device value is present. Only a
    workout with no device total falls back to the route-derived whole-window
    estimate; a missing or unusable route has no elevation.
    """
    row = conn.execute(
        "SELECT start_utc, end_utc, elevation_ascended_m, "
        "elevation_descended_m, elevation_source "
        "FROM workouts WHERE id = ?",
        (workout_id,),
    ).fetchone()
    if row is None:
        return {"status": "no_elevation"}
    if row[2] is not None or row[3] is not None:
        return {
            "ascended_m": row[2],
            "descended_m": row[3],
            "source": "device_metadata",
        }
    present = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'workout_routes'"
    ).fetchone()
    if present is None:
        return {"status": "no_elevation"}
    # Routes attached to this workout, not every route in its window: an
    # unmatched route that merely overlaps is not this workout's evidence.
    route_rows = conn.execute(
        "SELECT start_utc, points FROM workout_routes "
        "WHERE workout_id = ? ORDER BY start_utc",
        (workout_id,),
    ).fetchall()
    samples = _window_samples(route_rows, row[0], row[1])
    if not samples:
        return {"status": "no_elevation"}
    route = compute_elevation(samples)
    if route["n_used"] < MIN_USED_POINTS:
        return {"status": "no_elevation"}
    return {
        "ascended_m": route["ascent_m"],
        "descended_m": route["descent_m"],
        "source": "route_estimate",
    }
