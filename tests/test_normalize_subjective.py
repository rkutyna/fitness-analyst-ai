"""Catalog + ingest-name mappings for check-in metrics and Apple workout effort."""
from health_advisor import normalize as nz


def test_subjective_ratings_cataloged_as_mean_scores():
    for m in ("subjective_stress", "subjective_soreness",
              "subjective_energy", "subjective_sleep_quality"):
        assert nz.CATALOG[m] == {"unit": "score", "agg": "mean", "group": "subjective"}
        assert nz.agg_for(m) == "mean"


def test_drink_counts_cataloged_as_summed_intake():
    for m in ("caffeine_drinks", "alcohol_drinks"):
        assert nz.CATALOG[m] == {"unit": "drinks", "agg": "sum", "group": "intake"}
        assert nz.agg_for(m) == "sum"


def test_workout_effort_mappings_both_hk_variants():
    assert nz.HK_QUANTITY["HKQuantityTypeIdentifierWorkoutEffortScore"] == "workout_effort"
    assert nz.HK_QUANTITY["HKQuantityTypeIdentifierEstimatedWorkoutEffortScore"] == "workout_effort"
    assert nz.CATALOG["workout_effort"] == {"unit": "score", "agg": "mean", "group": "workout"}


def test_subjective_not_wear_filtered():
    """Check-in metrics are self-reported, not watch-derived — the correlation
    wear filter must not drop low-wear days for them."""
    from health_advisor import correlate as C
    assert not C._needs_wear("subjective_stress")
    assert not C._needs_wear("caffeine_drinks")
