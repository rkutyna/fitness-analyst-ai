"""Public, static typed fixture for the plan-log projection tests."""
from __future__ import annotations

from health_advisor.plan_model import Week


SYNTHETIC_FIXTURE_WEEK = Week.from_dict({
    "schema_version": 1,
    "week_start": "2026-08-17",
    "rules": [{
        "kind": "session",
        "scope": {
            "week": "2026-08-17",
            "days": ["2026-08-18"],
            "session": "easy-run",
            "modality": "running",
        },
        "stated": {
            "start": "2026-08-18",
            "end": "2026-08-24",
            "include_start": True,
            "include_end": True,
        },
        "statement": {
            "type": "stated",
            "value": {"minutes": 30, "intensity": "easy"},
        },
        "provenance": {
            "type": "parsed",
            "file": "fixture/plan-week-2026-08-17.md",
            "line": 12,
        },
        "enforced_from": None,
        "acceptance_date": "2026-08-17",
        "payload": {"source": "fixture", "priority": 1},
    }],
    "provenance": {
        "type": "parsed",
        "file": "fixture/week-manifest-2026-08-17.json",
        "line": 1,
    },
    "grading_policy": {
        "version": "fixture-policy-v1",
        "effective_date": "2026-08-17",
        "over_volume_factor": 1.30,
        "under_volume_factor": 0.45,
        "jog_credit_factor": 0.55,
        "block_credit_factor": 0.60,
        "qualify_min_minutes": 25,
        "qualify_min_avg_hr": 105,
        "qualify_min_kcal": 125,
        "non_endurance_types": sorted([
            "traditional_strength_training", "functional_strength_training",
            "core_training", "high_intensity_interval_training", "yoga",
            "pilates", "barre", "tai_chi", "mind_and_body", "flexibility",
            "cooldown", "preparation_and_recovery", "gymnastics", "wrestling",
            "boxing", "kickboxing", "martial_arts", "fencing", "fitness_gaming",
        ]),
    },
})
