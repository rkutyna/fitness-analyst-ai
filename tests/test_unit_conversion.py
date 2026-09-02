"""Unit conversion is wired into the HealthKit-direct parser.

The receiver stores canonical units, so a locale or source-unit change cannot
silently relabel a number without converting it.
"""
from __future__ import annotations

import pytest

from health_advisor import hk_parse, normalize as nz


def _payload(type_identifier: str, unit: str, value: float = 5.0):
    return {
        "protocol_version": 1,
        "device": {"id": "watch", "name": "Watch", "model": "test"},
        "app_version": "1", "batch_id": "conversion", "batch_sequence": 1,
        "sent_at": "2026-07-30T12:00:00Z", "anchors": [],
        "samples": [{
            "kind": "quantity", "hk_uuid": "sample-1",
            "type_identifier": type_identifier,
            "start": "2026-07-30T08:00:00-04:00",
            "end": "2026-07-30T08:01:00-04:00", "value": value,
            "unit": unit,
            "source_revision": {"source_name": "Watch", "bundle_id": "test"},
        }],
        "deletions": [], "workouts": [],
    }


@pytest.mark.parametrize("value,frm,to,expect", [
    (5.0, "km", "mi", 3.106855),
    (1.0, "mi", "km", 1.609344),
    (5.0, "m", "mi", 0.00310686),
    (70.0, "kg", "lb", 154.3236),
    (154.0, "lb", "kg", 69.8532),
    (1.8, "m", "ft", 5.905512),
    (0.0, "degC", "degF", 32.0),
    (100.0, "degC", "degF", 212.0),
    (98.6, "degF", "degC", 37.0),
    (1000.0, "kJ", "kcal", 239.0057),
    (1.0, "Cal", "kcal", 1.0),
    (10.0, "km/hr", "mi/hr", 6.213712),
    (1.0, "hr", "min", 60.0),
    (1.0, "L", "mL", 1000.0),
    (2.54, "cm", "in", 1.0),
])
def test_convert(value, frm, to, expect):
    assert nz.convert_unit(value, frm, to) == pytest.approx(expect, rel=1e-5)


def test_convert_is_identity_for_the_same_unit():
    assert nz.convert_unit(7.5, "mi", "mi") == 7.5


def test_convert_round_trips():
    for v, a, b in [(5.0, "km", "mi"), (70.0, "kg", "lb"), (21.0, "degC", "degF")]:
        assert nz.convert_unit(nz.convert_unit(v, a, b), b, a) == pytest.approx(v)


@pytest.mark.parametrize("frm,to", [
    ("km", "kg"), ("degC", "mi"), ("furlong", "mi"), ("mi", "parsec"),
])
def test_an_impossible_conversion_is_loud(frm, to):
    with pytest.raises(nz.UnitError):
        nz.convert_unit(1.0, frm, to)


def test_kilometres_become_miles_not_a_relabel():
    parsed = hk_parse.parse_payload(
        _payload("HKQuantityTypeIdentifierDistanceWalkingRunning", "km", 5.0))
    rec = parsed["records"][0]
    assert rec["unit"] == "mi"
    assert rec["value"] == pytest.approx(3.106855, rel=1e-5)
    assert parsed["unhandled"] == []


def test_kilograms_become_pounds():
    rec = hk_parse.parse_payload(
        _payload("HKQuantityTypeIdentifierBodyMass", "kg", 70.0))["records"][0]
    assert rec["unit"] == "lb"
    assert rec["value"] == pytest.approx(154.3236, rel=1e-5)


def test_celsius_becomes_fahrenheit_affinely():
    rec = hk_parse.parse_payload(
        _payload("HKQuantityTypeIdentifierAppleSleepingWristTemperature",
                 "degC", 37.0))["records"][0]
    assert rec["unit"] == "degF"
    assert rec["value"] == pytest.approx(98.6, rel=1e-4)


@pytest.mark.parametrize("type_identifier,unit,value", [
    ("HKQuantityTypeIdentifierDistanceWalkingRunning", "mi", 3.2),
    ("HKQuantityTypeIdentifierBodyMass", "lb", 168.4),
    ("HKQuantityTypeIdentifierActiveEnergyBurned", "kcal", 512.0),
    ("HKQuantityTypeIdentifierHeartRate", "count/min", 61.0),
    ("HKQuantityTypeIdentifierAppleExerciseTime", "min", 44.0),
])
def test_imperial_input_is_untouched(type_identifier, unit, value):
    rec = hk_parse.parse_payload(
        _payload(type_identifier, unit, value))["records"][0]
    assert rec["value"] == value


def test_apples_cal_is_still_kcal_at_factor_one():
    rec = hk_parse.parse_payload(
        _payload("HKQuantityTypeIdentifierActiveEnergyBurned", "Cal", 512.0)
    )["records"][0]
    assert (rec["value"], rec["unit"]) == (512.0, "kcal")


def test_an_unconvertible_unit_drops_the_points_and_says_so():
    parsed = hk_parse.parse_payload(
        _payload("HKQuantityTypeIdentifierDistanceWalkingRunning", "furlong", 5.0))
    assert parsed["records"] == []
    assert len(parsed["unhandled"]) == 1
    assert "furlong" in parsed["unhandled"][0]
    assert "mi" in parsed["unhandled"][0]


def test_a_wrong_family_unit_drops_the_points():
    parsed = hk_parse.parse_payload(
        _payload("HKQuantityTypeIdentifierBodyMass", "km", 70.0))
    assert parsed["records"] == []
    assert parsed["unhandled"] and "km" in parsed["unhandled"][0]


def test_every_catalog_unit_is_a_unit_the_converter_understands():
    unknown = sorted({c["unit"] for c in nz.CATALOG.values()
                      if not nz.is_convertible_unit(c["unit"])})
    assert unknown == []
