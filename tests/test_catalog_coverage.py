"""Every metric observed in the live DB is cataloged (added 2026-07-15 after the
correlation-tools review found 34 auto-named metrics bypassing group semantics)."""
from __future__ import annotations

import pytest

from health_advisor import normalize as nz

# The 34 formerly-uncataloged metrics (from the live DB survey, 2026-07-15).
NEWLY_CATALOGED = [
    "apple_sleeping_breathing_disturbances", "body_fat_percentage",
    "breathing_disturbances", "cardio_recovery",
    "cycling_cadence", "cycling_power", "cycling_speed",
    "dietary_calcium", "dietary_carbohydrates", "dietary_cholesterol",
    "dietary_energy_consumed", "dietary_fat_monounsaturated",
    "dietary_fat_polyunsaturated", "dietary_fat_saturated", "dietary_fat_total",
    "dietary_fiber", "dietary_iron", "dietary_potassium", "dietary_protein",
    "dietary_sodium", "dietary_sugar", "dietary_vitamin_c", "dietary_water",
    "environmental_sound_reduction", "lean_body_mass", "number_of_times_fallen",
    "running_ground_contact_time", "running_power", "running_stride_length",
    "running_vertical_oscillation", "six_minute_walk_test_distance",
    "six_minute_walking_test_distance", "walking_asymmetry_percentage",
    "walking_double_support_percentage",
]


def test_all_observed_metrics_cataloged():
    for m in NEWLY_CATALOGED:
        assert m in nz.CATALOG, m
        entry = nz.CATALOG[m]
        assert entry["agg"] in ("sum", "mean", "last"), m
        assert entry.get("group"), m


def test_key_semantics():
    # daily intake totals are summed
    assert nz.agg_for("dietary_protein") == "sum"
    assert nz.agg_for("dietary_energy_consumed") == "sum"
    # scale readings are point-in-time states
    assert nz.agg_for("body_fat_percentage") == "last"
    assert nz.agg_for("lean_body_mass") == "last"
    # nightly/recovery watch metrics land in wear-filtered groups
    assert nz.CATALOG["cardio_recovery"]["group"] == "heart"
    assert nz.CATALOG["breathing_disturbances"]["group"] == "sleep"
    # workout dynamics are NOT wear-filtered (exist only when worn in-session)
    from health_advisor.correlate import WATCH_GROUPS
    assert nz.CATALOG["running_power"]["group"] not in WATCH_GROUPS
    assert nz.CATALOG["cycling_power"]["group"] not in WATCH_GROUPS


def test_is_known_metric_answers_for_the_catalog():
    """`is_known_metric` is how a caller surfaces a miss instead of storing it
    unmanaged. Everything downstream — unit, aggregation rule, wear filter,
    correlation group — is looked up by name and silently defaults when the
    name is missing, so an unanswered miss is invisible rather than loud.
    """
    assert nz.is_known_metric("step_count")
    assert nz.is_known_metric("heart_rate")
    assert not nz.is_known_metric("blood_glucose_wibble")


@pytest.mark.parametrize("fork", [
    ("apple_sleeping_breathing_disturbances", "breathing_disturbances"),
    ("six_minute_walk_test_distance", "six_minute_walking_test_distance"),
])
def test_the_known_vocabulary_forks_are_both_cataloged(fork):
    """Regression pin: they are both in CATALOG on purpose (unifying them needs
    a records migration). This test exists so that removing one to 'clean up'
    fails loudly rather than silently reopening the fork.

    Restored 2026-08-22 with the reasoning intact. The original lived in
    `test_unknown_hae_metric.py` and was deleted whole when the Health Auto
    Export path was retired (#36) — the file was named for a retired ingest
    path, but this pin is about the metric vocabulary, which outlives it. The
    membership assertion survived in `test_all_observed_metrics_cataloged`;
    the reason did not, and the reason is the part that stops the next cleanup.
    """
    a, b = fork
    assert nz.is_known_metric(a) and nz.is_known_metric(b)
