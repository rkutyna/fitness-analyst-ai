from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from health_advisor import backfill, db, hk_parse
from health_advisor import normalize as nz


DEVICE = {"id": "device-1", "name": "iPhone", "model": "iPhone17,1"}
REVISION = {"source_name": "Apple Watch", "bundle_id": "com.apple.Health", "version": "26.0"}


def _payload(*samples, **overrides):
    payload = {
        "protocol_version": 1,
        "device": DEVICE,
        "app_version": "1.0.0+7",
        "batch_id": "batch-1",
        "batch_sequence": 1842,
        "sent_at": "2026-08-22T13:04:05Z",
        "anchors": [{
            "type_identifier": "HKQuantityTypeIdentifierStepCount",
            "from": None,
            "to": "anchor-next",
        }],
        "samples": list(samples),
        "deletions": [{"hk_uuid": "deleted-1", "type_identifier":
                       "HKQuantityTypeIdentifierStepCount"}],
        "workouts": [],
    }
    payload.update(overrides)
    return payload


def _quantity(**overrides):
    sample = {
        "kind": "quantity", "hk_uuid": "quantity-1",
        "type_identifier": "HKQuantityTypeIdentifierStepCount",
        "start": "2026-08-22T08:00:00-04:00",
        "end": "2026-08-22T08:05:00-04:00", "value": 120, "unit": "count",
        "source_revision": REVISION,
        "device": {"name": "Apple Watch", "model": "Watch7,5"},
    }
    sample.update(overrides)
    return sample


def _sleep(**overrides):
    sample = {
        "kind": "category", "hk_uuid": "sleep-1",
        "type_identifier": nz.HK_SLEEP_TYPE_IDENTIFIER,
        "start": "2026-08-21T23:50:00-04:00",
        "end": "2026-08-22T00:20:00-04:00",
        "value": "HKCategoryValueSleepAnalysisAsleepCore", "unit": None,
        "source_revision": REVISION,
        "device": {"name": "Apple Watch", "model": "Watch7,5"},
    }
    sample.update(overrides)
    return sample


def _workout(**overrides):
    workout = {
        "hk_uuid": "workout-1",
        "workout_activity_type": "HKWorkoutActivityTypeRunning",
        "start": "2026-08-22T23:30:00-04:00",
        "end": "2026-08-23T00:00:00-04:00",
        "duration_min": 30.0,
        "energy_kcal": 300.0,
        "distance_mi": 3.0,
        "avg_heart_rate": 145.0,
        "max_heart_rate": 170.0,
        "source_revision": REVISION,
    }
    workout.update(overrides)
    return workout


def _daily_total(**overrides):
    total = {
        "type_identifier": "HKQuantityTypeIdentifierStepCount",
        "local_date": "2026-08-25",
        "value": 10173,
        "unit": "count",
        "interval": "day",
        "state": "provisional",
        "queried_at": "2026-08-26T09:00:00-04:00",
    }
    total.update(overrides)
    return total


def test_realistic_batch_expands_sleep_and_carries_identity():
    out = hk_parse.parse_payload(_payload(_quantity(), _sleep()))

    assert [(row["metric"], row["value"], row["unit"])
            for row in out["records"]] == [
        ("step_count", 120.0, "count"),
        ("sleep_asleep", 30.0, "min"),
        ("sleep_core", 30.0, "min"),
    ]
    assert {row["metric"] for row in out["records"] if row["hk_uuid"] == "sleep-1"} == {
        "sleep_asleep", "sleep_core"
    }
    assert all(row["origin"] == "healthkit" for row in out["records"])
    assert all(row["hk_device_id"] == "device-1" for row in out["records"])
    assert out["records"][0]["hk_type_identifier"] == "HKQuantityTypeIdentifierStepCount"
    assert out["deletions"] == [{"hk_uuid": "deleted-1",
                                  "type_identifier": "HKQuantityTypeIdentifierStepCount"}]
    assert out["anchors"][0]["to"] == "anchor-next"
    assert out["batch_id"] == "batch-1"
    assert out["batch_sequence"] == 1842
    assert out["device_id"] == "device-1"
    assert out["pairs"] == {
        ("step_count", "2026-08-22"),
        ("sleep_asleep", "2026-08-22"),
        ("sleep_core", "2026-08-22"),
    }
    assert out["unhandled"] == []


def test_client_local_date_is_ignored_and_offset_wins():
    row = hk_parse.parse_payload(_payload(
        _quantity(start="2026-08-21T23:00:00-04:00",
                  end="2026-08-21T23:05:00-04:00", local_date="2099-01-01")
    ))["records"][0]
    assert row["local_date"] == "2026-08-21"
    assert row["start_local"] == "2026-08-21 23:00:00"


def test_unconvertible_unit_is_dropped_and_logged():
    with pytest.raises(nz.UnitError):
        nz.unit_converter("distance_walking_running", "furlong")
    out = hk_parse.parse_payload(_payload(_quantity(
        type_identifier="HKQuantityTypeIdentifierDistanceWalkingRunning",
        value=5, unit="furlong",
    )))
    assert out["records"] == []
    assert "unknown unit" in out["unhandled"][0]


def test_unit_is_converted_not_relabelled():
    row = hk_parse.parse_payload(_payload(_quantity(
        type_identifier="HKQuantityTypeIdentifierDistanceWalkingRunning",
        value=5, unit="km",
    )))['records'][0]
    assert row["unit"] == "mi"
    assert row["value"] == pytest.approx(3.10685596)


def test_unknown_type_does_not_reject_batch():
    out = hk_parse.parse_payload(_payload(_quantity(
        type_identifier="HKQuantityTypeIdentifierFutureMetric",
    )))
    assert out["records"] == []
    assert "unknown type_identifier" in out["unhandled"][0]
    assert out["batch_id"] == "batch-1"


def test_category_unit_may_be_omitted():
    sample = _sleep()
    sample.pop("unit")
    assert len(hk_parse.parse_payload(_payload(sample))["records"]) == 2


def test_unknown_field_rejects_whole_batch():
    with pytest.raises(hk_parse.PayloadError):
        hk_parse.parse_payload(_payload(_quantity(not_a_contract_field=True)))


def test_malformed_sample_does_not_kill_batch():
    out = hk_parse.parse_payload(_payload(
        _quantity(hk_uuid="bad", start="not-a-date"),
        _quantity(hk_uuid="good"),
    ))
    assert [row["hk_uuid"] for row in out["records"]] == ["good"]
    assert len(out["unhandled"]) == 1


def test_malformed_sample_object_does_not_kill_batch():
    out = hk_parse.parse_payload(_payload("not-an-object", _quantity()))
    assert [row["hk_uuid"] for row in out["records"]] == ["quantity-1"]
    assert len(out["unhandled"]) == 1


def test_parser_has_no_healthkit_identifier_table():
    source = __import__("inspect").getsource(hk_parse)
    assert "HKQuantityTypeIdentifier" not in source
    assert "HKCategoryValueSleepAnalysis" not in source


def test_a_sample_with_no_device_is_ordinary_data_not_a_malformed_one():
    """HealthKit returns no device for a great many samples.

    The watch writes one; a manual entry, and plenty of third-party sources, do
    not. The wire contract lists per-sample `device` as optional, and requiring
    it routed every such sample into `unhandled` — a silent, permanent loss of
    real data that looks like the phone never sent it. Absent and explicit null
    both mean "no device".
    """
    for label, sample in (("absent", _quantity()), ("null", _quantity(device=None))):
        if label == "absent":
            sample.pop("device")
        out = hk_parse.parse_payload(_payload(sample))
        assert out["unhandled"] == [], f"device {label} was treated as malformed"
        assert len(out["records"]) == 1
        # Falls back to the envelope's device, which is always present.
        assert out["records"][0]["hk_device_id"] == DEVICE["id"]


def test_source_revision_needs_only_the_two_fields_the_contract_requires():
    """version / product_type / operating_system_version are optional.

    HealthKit does not always populate them, and demanding them is the same
    silent-attrition defect as requiring `device`.
    """
    out = hk_parse.parse_payload(_payload(_quantity(
        source_revision={"source_name": "Apple Watch", "bundle_id": "com.apple.Health"})))
    assert out["unhandled"] == []
    assert len(out["records"]) == 1


def test_a_missing_required_field_is_named_in_the_reason():
    """"missing required field" without saying which is an absence nobody can
    explain — the one kind of answer this project is not allowed to give."""
    sample = _quantity()
    del sample["value"]
    out = hk_parse.parse_payload(_payload(sample))
    assert out["records"] == []
    assert "value" in out["unhandled"][0], out["unhandled"]


def test_workout_is_canonicalized_to_the_db_row_shape_and_start_date():
    out = hk_parse.parse_payload(_payload(workouts=[_workout()]))
    row = out["workouts"][0]
    assert set(row) == set(db.WORKOUT_COLS)
    assert row["workout_type"] == "running"
    assert row["local_date"] == "2026-08-22"
    assert row["unit_distance"] == "mi"
    assert row["dedupe_key"] == db.workout_key(
        "running", "2026-08-23T03:30:00+00:00", "2026-08-23T04:00:00+00:00")
    assert out["workout_dates"] == {"2026-08-22"}


def test_workout_key_merges_with_the_backfill_row_for_the_same_session(conn):
    xml = ET.fromstring(
        '<Workout workoutActivityType="HKWorkoutActivityTypeRunning" '
        'startDate="2026-08-22 23:30:00 -0400" '
        'endDate="2026-08-23 00:00:00 -0400" duration="30.0" '
        'sourceName="Apple Watch" />')
    backfill_row = backfill._workout_row(xml)
    healthkit_row = hk_parse.parse_payload(
        _payload(workouts=[_workout()]))["workouts"][0]
    assert healthkit_row["dedupe_key"] == backfill_row["dedupe_key"]
    assert db.insert_workouts(conn, [backfill_row]) == 1
    assert db.insert_workouts(conn, [healthkit_row]) == 0
    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 1


def test_workout_unknown_field_rejects_the_whole_batch():
    with pytest.raises(hk_parse.PayloadError):
        hk_parse.parse_payload(_payload(workouts=[_workout(extra=True)]))


def test_malformed_workout_is_unhandled_without_failing_the_batch():
    out = hk_parse.parse_payload(_payload(workouts=[
        _workout(start="not-a-date"), _workout(hk_uuid="good-workout")]))
    assert [row["hk_uuid"] for row in out["workouts"]] == ["good-workout"]
    assert len(out["unhandled"]) == 1
    assert out["unhandled"][0].startswith("workouts[0]")


def test_non_object_workout_rejects_the_batch():
    with pytest.raises(hk_parse.PayloadError):
        hk_parse.parse_payload(_payload(workouts=["not-an-object"]))


def test_payload_without_daily_totals_still_parses():
    """19. A new server remains compatible with an old phone envelope."""
    out = hk_parse.parse_payload(_payload())
    assert out["daily_totals"] == []
    assert out["daily_total_dates"] == set()


def test_daily_total_unknown_field_rejects_the_batch():
    """20. Consolidated rows have the same strict nested-field contract."""
    with pytest.raises(hk_parse.PayloadError):
        hk_parse.parse_payload(_payload(
            daily_totals=[_daily_total(not_a_contract_field=True)]))


def test_unknown_daily_total_type_is_unhandled():
    """21. Unknown HealthKit identifiers are visible, never stored."""
    out = hk_parse.parse_payload(_payload(daily_totals=[_daily_total(
        type_identifier="HKQuantityTypeIdentifierFutureMetric")]))
    assert out["daily_totals"] == []
    assert out["daily_total_dates"] == set()
    assert "unknown type_identifier" in out["unhandled"][-1]
