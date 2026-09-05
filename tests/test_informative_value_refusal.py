from __future__ import annotations

import copy

from health_advisor import chat
from health_advisor import deepdive_verify as DV


ATTEMPT_0_LEDGER = [{
    "sequence": 1,
    "tool_name": "list_workouts",
    "arguments": {},
    "result": {"workouts": [
        {"workout_key": "w0", "distance_mi": 1.00,
         "duration_min": 10.0, "avg_heart_rate": 100.0},
        {"workout_key": "w1", "distance_mi": 2.00,
         "duration_min": 20.0, "avg_heart_rate": 110.0},
        {"workout_key": "w2", "distance_mi": 2.50,
         "duration_min": 30.0, "avg_heart_rate": 115.0},
        {"workout_key": "w3", "distance_mi": 3.00,
         "duration_min": 40.0, "avg_heart_rate": 120.0},
        {"workout_key": "w4", "distance_mi": 3.50,
         "duration_min": 50.0, "avg_heart_rate": 125.0},
        {"workout_key": "w5", "distance_mi": 3.85,
         "duration_min": 68.8, "avg_heart_rate": 123.0},
        {"workout_key": "w6", "distance_mi": 4.94,
         "duration_min": 205.7, "avg_heart_rate": 141.0},
    ]},
    "result_elided": False,
}]

ATTEMPT_0_CLAIMS = [
    {"metric": None, "period": None, "field": "distance_mi", "value": 1.00,
     "source": {"sequence": 1, "path": "$.result.workouts[0].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 10.0,
     "source": {"sequence": 1, "path": "$.result.workouts[0].duration_min"}},
    {"metric": None, "period": None, "field": "avg_heart_rate", "value": 100.0,
     "source": {"sequence": 1, "path": "$.result.workouts[0].avg_heart_rate"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 2.00,
     "source": {"sequence": 1, "path": "$.result.workouts[1].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 20.0,
     "source": {"sequence": 1, "path": "$.result.workouts[1].duration_min"}},
    {"metric": None, "period": None, "field": "avg_heart_rate", "value": 110.0,
     "source": {"sequence": 1, "path": "$.result.workouts[1].avg_heart_rate"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 2.50,
     "source": {"sequence": 1, "path": "$.result.workouts[2].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 30.0,
     "source": {"sequence": 1, "path": "$.result.workouts[2].duration_min"}},
    {"metric": None, "period": None, "field": "avg_heart_rate", "value": 115.0,
     "source": {"sequence": 1, "path": "$.result.workouts[2].avg_heart_rate"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 3.00,
     "source": {"sequence": 1, "path": "$.result.workouts[3].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 40.0,
     "source": {"sequence": 1, "path": "$.result.workouts[3].duration_min"}},
    {"metric": None, "period": None, "field": "avg_heart_rate", "value": 120.0,
     "source": {"sequence": 1, "path": "$.result.workouts[3].avg_heart_rate"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 3.50,
     "source": {"sequence": 1, "path": "$.result.workouts[4].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 50.0,
     "source": {"sequence": 1, "path": "$.result.workouts[4].duration_min"}},
    {"metric": None, "period": None, "field": "avg_heart_rate", "value": 125.0,
     "source": {"sequence": 1, "path": "$.result.workouts[4].avg_heart_rate"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 3.85,
     "source": {"sequence": 1, "path": "$.result.workouts[5].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 68.8,
     "source": {"sequence": 1, "path": "$.result.workouts[5].duration_min"}},
    {"metric": None, "period": None, "field": "avg_heart_rate", "value": 123.0,
     "source": {"sequence": 1, "path": "$.result.workouts[5].avg_heart_rate"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 4.94,
     "source": {"sequence": 1, "path": "$.result.workouts[6].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 205.7,
     "source": {"sequence": 1, "path": "$.result.workouts[6].duration_min"}},
    {"metric": None, "period": None, "field": "avg_heart_rate", "value": 141.0,
     "source": {"sequence": 1, "path": "$.result.workouts[6].avg_heart_rate"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 3.85,
     "source": {"sequence": 1, "path": "$.result.workouts[6].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 68.8,
     "source": {"sequence": 1, "path": "$.result.workouts[6].duration_min"}},
    {"metric": None, "period": None, "field": "avg_heart_rate", "value": 123.0,
     "source": {"sequence": 1, "path": "$.result.workouts[6].avg_heart_rate"}},
]

ATTEMPT_1_LEDGER = [{
    "sequence": 1,
    "tool_name": "list_workouts",
    "arguments": {},
    "result": {"workouts": [
        {"workout_key": "a0", "distance_mi": 1.00, "duration_min": 10.0},
        {"workout_key": "a1", "distance_mi": 3.82, "duration_min": 184.5},
        {"workout_key": "a2", "distance_mi": 3.18, "duration_min": 48.3},
        {"workout_key": "a3", "distance_mi": 3.40, "duration_min": 40.0},
        {"workout_key": "a4", "distance_mi": 3.60, "duration_min": 42.0},
        {"workout_key": "a5", "distance_mi": 4.00, "duration_min": 44.0},
        {"workout_key": "a6", "distance_mi": 4.20, "duration_min": 46.0},
        {"workout_key": "a7", "distance_mi": 4.40, "duration_min": 35.0},
        {"workout_key": "a8", "distance_mi": 5.84, "duration_min": 37.0},
        {"workout_key": "a9", "distance_mi": 2.87, "duration_min": 49.8},
    ]},
    "result_elided": False,
}]

ATTEMPT_1_CLAIMS = [
    {"metric": None, "period": None, "field": "distance_mi", "value": 1.00,
     "source": {"sequence": 1, "path": "$.result.workouts[0].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 10.0,
     "source": {"sequence": 1, "path": "$.result.workouts[0].duration_min"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 3.82,
     "source": {"sequence": 1, "path": "$.result.workouts[1].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 184.5,
     "source": {"sequence": 1, "path": "$.result.workouts[1].duration_min"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 3.18,
     "source": {"sequence": 1, "path": "$.result.workouts[2].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 48.3,
     "source": {"sequence": 1, "path": "$.result.workouts[2].duration_min"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 3.40,
     "source": {"sequence": 1, "path": "$.result.workouts[3].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 40.0,
     "source": {"sequence": 1, "path": "$.result.workouts[3].duration_min"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 3.60,
     "source": {"sequence": 1, "path": "$.result.workouts[4].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 42.0,
     "source": {"sequence": 1, "path": "$.result.workouts[4].duration_min"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 2.87,
     "source": {"sequence": 1, "path": "$.result.workouts[8].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 49.8,
     "source": {"sequence": 1, "path": "$.result.workouts[8].duration_min"}},
    {"metric": None, "period": None, "field": "distance_mi", "value": 3.18,
     "source": {"sequence": 1, "path": "$.result.workouts[1].distance_mi"}},
    {"metric": None, "period": None, "field": "duration_min", "value": 48.3,
     "source": {"sequence": 1, "path": "$.result.workouts[1].duration_min"}},
]


def _verdicts(ledger, claims):
    return [DV.verify_number(None, claim, payload=ledger) for claim in claims]


def _assert_informative_refusals(verification, expected):
    refused = [result for result in verification if not result["ok"]]
    assert len(refused) == len(expected)
    assert all(result["ok"] is False for result in refused)
    assert {result["reason"] for result in refused} == {
        "claim value does not match ledger field"}
    assert [(result["path"], result["actual"]) for result in refused] == expected
    feedback = chat._retry_feedback({
        "ok": False,
        "reason": refused[0]["reason"],
        "unsupported": [],
        "verdict": {"numbers": verification},
    })
    for path, actual in expected:
        assert f"The cited path {path} holds {actual!r}, not the claimed value." \
            in feedback


def test_measured_bidirectional_index_mismatches_are_informative_refusals():
    attempt_0 = _verdicts(ATTEMPT_0_LEDGER, ATTEMPT_0_CLAIMS)
    assert sum(result["ok"] for result in attempt_0) == 21
    assert sum(not result["ok"] for result in attempt_0) == 3
    _assert_informative_refusals(attempt_0, [
        ("$.result.workouts[6].distance_mi", 4.94),
        ("$.result.workouts[6].duration_min", 205.7),
        ("$.result.workouts[6].avg_heart_rate", 141.0),
    ])
    print("attempt-0: 21 bind, 3 refused")

    attempt_1 = _verdicts(ATTEMPT_1_LEDGER, ATTEMPT_1_CLAIMS)
    assert sum(result["ok"] for result in attempt_1) == 10
    assert sum(not result["ok"] for result in attempt_1) == 4
    _assert_informative_refusals(attempt_1, [
        ("$.result.workouts[8].distance_mi", 5.84),
        ("$.result.workouts[8].duration_min", 37.0),
        ("$.result.workouts[1].distance_mi", 3.82),
        ("$.result.workouts[1].duration_min", 184.5),
    ])
    print("attempt-1: 10 bind, 4 refused")


def test_matching_mutated_fixture_claim_still_binds():
    claims = copy.deepcopy(ATTEMPT_0_CLAIMS)
    claims[21]["value"] = 4.94
    result = DV.verify_number(None, claims[21], payload=ATTEMPT_0_LEDGER)

    assert result["ok"] is True
    print("negative case: 1 mutated claim binds")
