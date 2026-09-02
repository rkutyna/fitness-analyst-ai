"""Unit lookup tolerates spelling, and refuses to guess when spelling is meaning.

2026-07-31 to 2026-08-16: sixty consecutive receiver batches dropped every
vo2_max reading because the source started writing 'ml/(kg·min)' where the
table held 'mL/min·kg'. Same unit, different string. Audit part 6, F6-1.

The second half of this file is the more important half. 'Cal' (kilocalorie)
and 'cal' (small calorie) differ only in case and by 1000x, so the fix must NOT
fold case blindly — an ambiguous spelling has to stay an error."""
import pytest

from health_advisor import hk_parse, normalize as nz


# --- the defect ---------------------------------------------------------------

@pytest.mark.parametrize("unit", [
    "ml/(kg·min)",      # what the source started sending on 2026-07-31
    "mL/min·kg",        # what the table held
    "ML/(KG*MIN)",      # case + separator variant
    "ml/kg/min",        # double-slash variant
    "mL / min · kg",    # spaced
])
def test_every_vo2max_spelling_resolves(unit):
    assert nz.is_convertible_unit(unit)


def test_the_two_live_spellings_convert_as_an_identity():
    # Same unit, so no factor may be applied in either direction.
    assert nz.convert_unit(42.0, "ml/(kg·min)", "mL/min·kg") == 42.0
    assert nz.convert_unit(42.0, "mL/min·kg", "ml/(kg·min)") == 42.0


def test_the_other_composite_rate_is_covered_too():
    assert nz.is_convertible_unit("kcal/(kg·hr)")
    assert nz.convert_unit(3.0, "kcal/(kg·hr)", "kcal/hr·kg") == 3.0


# --- the safety property ------------------------------------------------------

def test_capital_cal_and_lowercase_cal_stay_1000x_apart():
    assert nz.convert_unit(1.0, "Cal", "kcal") == 1.0
    assert nz.convert_unit(1.0, "cal", "kcal") == pytest.approx(0.001)


def test_an_ambiguous_case_folding_is_refused_not_guessed():
    # 'CAL' could fold to either. Raising is the only safe answer.
    with pytest.raises(nz.UnitError):
        nz.convert_unit(1.0, "CAL", "kcal")


def test_a_genuinely_unknown_unit_still_raises():
    with pytest.raises(nz.UnitError):
        nz.convert_unit(1.0, "furlongs/fortnight", "mL/min·kg")
    assert not nz.is_convertible_unit("furlongs/fortnight")


def test_no_loose_key_covers_two_different_conversions():
    """The collision guard itself: no loose key resolves to two conversions.

    Checked at each index's own strictness — _STRUCT_INDEX is case-sensitive
    (so 'Cal' and 'cal' are separate entries there and both are kept),
    _FOLDED_INDEX is not (so neither survives)."""
    for key, spec in nz._STRUCT_INDEX.items():
        rivals = {s for u, s in nz._UNIT_INDEX.items() if nz._structural_unit(u) == key}
        assert len(rivals) == 1, f"structural {key!r} covers {rivals}"
    for key, spec in nz._FOLDED_INDEX.items():
        rivals = {s for u, s in nz._UNIT_INDEX.items()
                  if nz._structural_unit(u).casefold() == key}
        assert len(rivals) == 1, f"folded {key!r} covers {rivals}"


def test_the_calorie_collision_is_what_the_guard_excluded():
    # Both survive the case-sensitive index...
    assert "Cal" in nz._STRUCT_INDEX and "cal" in nz._STRUCT_INDEX
    # ...and neither survives the folded one, which is why 'CAL' raises.
    assert "cal" not in nz._FOLDED_INDEX


# --- no regression on the existing table --------------------------------------

@pytest.mark.parametrize("value,src,dst,expected", [
    (1.0, "kg", "lb", 2.2046226218487757),
    (1.0, "km/hr", "m/s", 0.2777777777777778),
    (60.0, "count/min", "bpm", 60.0),
    (1.0, "hr", "min", 60.0),
    (212.0, "degF", "degC", 100.0),
])
def test_existing_conversions_are_unchanged(value, src, dst, expected):
    assert nz.convert_unit(value, src, dst) == pytest.approx(expected)


def test_healthkit_ingest_accepts_the_reading_that_was_being_dropped():
    payload = {
        "protocol_version": 1,
        "device": {"id": "watch", "name": "Watch", "model": "test"},
        "app_version": "1", "batch_id": "vo2", "batch_sequence": 1,
        "sent_at": "2026-08-16T12:15:00Z", "anchors": [],
        "samples": [{
            "kind": "quantity", "hk_uuid": "vo2-1",
            "type_identifier": "HKQuantityTypeIdentifierVO2Max",
            "start": "2026-08-16T08:15:00-04:00",
            "end": "2026-08-16T08:15:01-04:00", "value": 37.55,
            "unit": "ml/(kg·min)",
            "source_revision": {"source_name": "Watch", "bundle_id": "test"},
        }], "deletions": [], "workouts": [],
    }
    rec = hk_parse.parse_payload(payload)["records"][0]
    assert rec["metric"] == "vo2_max"
    assert rec["value"] == 37.55
    assert rec["unit"] == "mL/min·kg"


def test_structural_form_sorts_denominators_not_numerators():
    assert nz._structural_unit("ml/(kg·min)") == nz._structural_unit("ml/min·kg")
    assert nz._structural_unit("kg/min") != nz._structural_unit("min/kg")
