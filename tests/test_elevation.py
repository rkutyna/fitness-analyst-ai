"""Seeded, synthetic acceptance tests for version-1 elevation."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import random
import struct
import sqlite3

import pytest

from health_advisor import elevation


def _noise(seed: int, count: int, amplitude: float) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-amplitude, amplitude) for _ in range(count)]


def _paired_uniform_noise(seed: int, count: int, amplitude: float) -> list[float]:
    """Seeded uniform bounded jitter, paired so a flat median stays flat."""
    rng = random.Random(seed)
    noise = []
    for _ in range((count + 1) // 2):
        value = rng.uniform(-amplitude, amplitude)
        noise.extend((value, -value))
    return noise[:count]


def _three_hills() -> tuple[list[float], list[float], list[float]]:
    n = 200 * 3
    times = list(range(n))
    noise = _noise(1, n, 5.0)
    altitudes = []
    for i in range(n):
        phase = i % 200
        if phase < 10:
            height = 0.0
        elif phase < 100:
            height = (phase - 10) / 90.0
        elif phase < 110:
            height = 1.0
        else:
            height = 1.0 - (phase - 110) / 90.0
        altitudes.append(100.0 * height + noise[i])
    return times, altitudes, [5.0] * n


def _flat() -> tuple[list[float], list[float], list[float]]:
    altitudes = _paired_uniform_noise(5802, 3000, 5.0)
    return list(range(3000)), altitudes, [5.0] * 3000


def _monotone_climb() -> tuple[list[float], list[float], list[float]]:
    n = 500
    noise = _noise(47, n, 5.0)
    altitudes = [50.0 * min(i, 400) / 400.0 + noise[i] for i in range(n)]
    return list(range(n)), altitudes, [5.0] * n


def _one_reversal() -> tuple[list[float], list[float], list[float]]:
    n = 360
    noise = _noise(5804, n, 1.0)
    altitudes = []
    for i in range(n):
        if i <= 180:
            ideal = 100.0 - 36.0 * i / 180.0
        else:
            ideal = 64.0 + 18.0 * (i - 180) / 179.0
        altitudes.append(ideal + noise[i])
    return list(range(n)), altitudes, [1.0] * n


def _rolling_hills() -> tuple[list[float], list[float], list[float]]:
    n = 3000
    noise = _noise(5805, n, 1.0)
    altitudes = []
    for i in range(n):
        phase = (i % 150) / 75.0
        height = phase if phase <= 1.0 else 2.0 - phase
        altitudes.append(6.0 * height + noise[i])
    return list(range(n)), altitudes, [1.0] * n


def _compute(profile):
    return elevation.compute_elevation(*profile)


def _assert_arithmetic(result: dict) -> None:
    assert abs(result["ascent_m"] - result["descent_m"] - result["net_m"]) <= 3.0


def test_three_hundred_metre_hills():
    result = _compute(_three_hills())
    print(f"\n[Done-when 1] ascent={result['ascent_m']:.3f} descent={result['descent_m']:.3f} m")
    assert result["ascent_m"] == pytest.approx(300.0, abs=15.0)
    assert result["descent_m"] == pytest.approx(300.0, abs=15.0)
    _assert_arithmetic(result)


def test_flat_track_rejects_small_noise():
    result = _compute(_flat())
    print(f"\n[Done-when 2] flat ascent={result['ascent_m']:.3f} m")
    assert result["ascent_m"] < 10.0
    _assert_arithmetic(result)


def test_monotone_fifty_metre_climb():
    result = _compute(_monotone_climb())
    print(f"\n[Done-when 3] ascent={result['ascent_m']:.3f} descent={result['descent_m']:.3f} m")
    assert result["ascent_m"] == pytest.approx(50.0, abs=3.0)
    assert result["descent_m"] < 3.0
    assert result["net_m"] == pytest.approx(50.0, abs=3.0)
    _assert_arithmetic(result)


@pytest.mark.parametrize("profile_factory", [_three_hills, _flat, _monotone_climb])
def test_split_carries_extreme_and_direction_with_bounded_smoothing_seam(profile_factory):
    times, altitudes, accuracies = profile_factory()
    midpoint = len(times) // 2
    whole = elevation.compute_elevation(times, altitudes, accuracies)
    first, state = elevation.compute_elevation(
        times[:midpoint], altitudes[:midpoint], accuracies[:midpoint], return_state=True)
    second = elevation.compute_elevation(
        times[midpoint:], altitudes[midpoint:], accuracies[midpoint:], initial_state=state)
    seam_error = abs((first["ascent_m"] + second["ascent_m"]) - whole["ascent_m"])
    print(f"\n[Done-when 4] {profile_factory.__name__} ascent seam error={seam_error:.3f} m")
    assert seam_error <= 1.0
    _assert_arithmetic(whole)


def test_accuracy_filtering_and_missing_accuracy():
    times, altitudes, accuracies = _monotone_climb()
    accuracies[::10] = [25.0] * len(accuracies[::10])
    filtered = elevation.compute_elevation(times, altitudes, accuracies)
    absent = elevation.compute_elevation(times, altitudes)
    print(f"\n[Done-when 5] n_points={filtered['n_points']} n_used={filtered['n_used']}")
    assert filtered["n_points"] == len(times)
    assert filtered["n_used"] == len(times) - len(times[::10])
    assert filtered["accuracy_filtered"] is True
    assert absent["n_used"] == len(times)
    assert absent["accuracy_filtered"] is False


def _continuous_hill(t: int) -> float:
    phase = t % 200
    if phase < 10:
        height = 0.0
    elif phase < 100:
        height = (phase - 10) / 90.0
    elif phase < 110:
        height = 1.0
    else:
        height = 1.0 - (phase - 110) / 90.0
    return 100.0 * height


def test_irregular_timestamps_match_one_hz_profile():
    rng = random.Random(5806)
    one_hz_times = list(range(600))
    noise_by_time = _noise(5807, 600, 2.0)
    one_hz_altitudes = [_continuous_hill(t) + noise_by_time[t] for t in one_hz_times]
    irregular_times = []
    t = 0
    while t < 600:
        irregular_times.append(t)
        t += rng.randint(1, 5)
    irregular_altitudes = [one_hz_altitudes[t] for t in irregular_times]
    one_hz = elevation.compute_elevation(one_hz_times, one_hz_altitudes, [2.0] * len(one_hz_times))
    irregular = elevation.compute_elevation(
        irregular_times, irregular_altitudes, [2.0] * len(irregular_times))
    difference = abs(one_hz["ascent_m"] - irregular["ascent_m"])
    print(f"\n[Done-when 6] 1Hz={one_hz['ascent_m']:.3f} irregular={irregular['ascent_m']:.3f} diff={difference:.3f} m")
    assert difference <= 2.0
    _assert_arithmetic(one_hz)
    _assert_arithmetic(irregular)


def test_one_reversal_counts_the_whole_move():
    times, altitudes, accuracies = _one_reversal()
    result, state = elevation.compute_elevation(
        times, altitudes, accuracies, return_state=True)
    lower_bound = result["end_alt_m"] - result["min_alt_m"]
    print(f"\n[Done-when 9] ascent={result['ascent_m']:.3f} descent={result['descent_m']:.3f} m")
    assert result["descent_m"] == pytest.approx(36.0, abs=1.5)
    assert result["ascent_m"] == pytest.approx(18.0, abs=1.5)
    assert result["ascent_m"] >= lower_bound - 1e-6
    assert abs((result["ascent_m"] - result["descent_m"]) -
               (state[0] - result["start_alt_m"])) < 1e-6
    _assert_arithmetic(result)


def test_rolling_small_hills_do_not_halve_gain():
    result = _compute(_rolling_hills())
    print(f"\n[Done-when 10] ascent={result['ascent_m']:.3f} m")
    assert result["ascent_m"] == pytest.approx(120.0, abs=12.0)
    _assert_arithmetic(result)


@pytest.mark.parametrize("profile_factory", [
    _three_hills, _flat, _monotone_climb, _one_reversal, _rolling_hills,
])
def test_arithmetic_invariant_on_every_profile(profile_factory):
    result, state = elevation.compute_elevation(*profile_factory(), return_state=True)
    _assert_arithmetic(result)
    assert abs((result["ascent_m"] - result["descent_m"]) -
               (state[0] - result["start_alt_m"])) < 1e-6


def test_result_is_versioned_and_has_unit_convenience_fields():
    result = _compute(_one_reversal())
    assert result["method_version"] == 1
    assert result["ascent_ft"] == pytest.approx(result["ascent_m"] * elevation.METRES_TO_FEET)
    assert result["descent_ft"] == pytest.approx(result["descent_m"] * elevation.METRES_TO_FEET)
    assert not hasattr(elevation, "climb")
    assert not hasattr(elevation, "elevation_climb")


def test_route_climb_without_routes_table_is_a_status():
    conn = sqlite3.connect(":memory:")
    try:
        assert elevation.route_climb(conn, "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z") == {
            "status": "no_route"
        }
    finally:
        conn.close()


def test_route_climb_decodes_four_float32_arrays_and_half_open_window():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE workout_routes (start_utc TEXT, end_utc TEXT, encoding TEXT, points BLOB)")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    offsets = list(range(20))
    altitudes = [float(i) for i in range(20)]
    accuracies = [5.0] * 20
    horizontal = [2.0] * 20
    blob = b"".join(struct.pack("<20f", *array) for array in
                     (offsets, altitudes, accuracies, horizontal))
    conn.execute("INSERT INTO workout_routes VALUES (?, ?, ?, ?)", (
        start.isoformat().replace("+00:00", "Z"),
        (start + timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
        "float32le", blob,
    ))
    conn.commit()
    try:
        result = elevation.route_climb(
            conn, "2026-01-01T00:00:05Z", "2026-01-01T00:00:15Z")
        assert result["n_points"] == 10
        assert result["n_used"] == 10
        assert result["method_version"] == 1
        assert elevation.route_climb(
                conn, "2026-01-01T00:00:20Z", "2026-01-01T00:00:21Z") == {
                "status": "no_route"
            }
    finally:
        conn.close()


def test_mutation_zero_threshold_is_detected(monkeypatch):
    """The flat-track acceptance test must go red if T is mutated to zero."""
    monkeypatch.setattr(elevation, "HYSTERESIS_THRESHOLD_M", 0.0)
    result = _compute(_flat())
    with pytest.raises(AssertionError):
        assert result["ascent_m"] < 10.0


def test_mutation_move_minus_threshold_is_detected():
    """A local mutation model proves the reversal lower-bound guard catches it."""
    times, altitudes, accuracies = _one_reversal()
    rows, _has_accuracy, _n_points = elevation._normalise_samples(times, altitudes, accuracies)
    medians = elevation._rolling_medians(rows)
    correct, _descent, _state = elevation._hysteresis(medians, None)
    mutated = correct - elevation.HYSTERESIS_THRESHOLD_M
    lower_bound = medians[-1] - min(medians)
    with pytest.raises(AssertionError):
        assert mutated >= lower_bound - 1e-6


def test_twenty_thousand_points_under_one_second():
    times = list(range(20_000))
    altitudes = [0.5 * ((i % 40) / 40.0) for i in times]
    started = datetime.now(timezone.utc)
    result = elevation.compute_elevation(times, altitudes)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\n[Done-when 12] 20,000 samples elapsed={elapsed:.4f} s")
    assert result["n_used"] == 20_000
    assert elapsed < 1.0
