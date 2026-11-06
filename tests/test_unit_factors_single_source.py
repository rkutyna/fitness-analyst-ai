"""One definition site for the shared unit conversion factors (#93).

`normalize.UNIT_FAMILIES` and `vault.UNIT_CONVERSION_FACTORS` both need the
feet, pound and mph factors. They are defined once in `health_advisor/unit_factors.py`
and both tables are built from it. The oracles below are the values the two
tables held BEFORE that change (copied from the original source), so a changed
factor is caught even though the tables now read the same constant.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from health_advisor import normalize as nz
from health_advisor import unit_factors as uf
from health_advisor import vault as V

PKG = Path(__file__).resolve().parent.parent / "health_advisor"

# Factors that were written as literals in more than one table.
SHARED_LITERALS = {
    "FT_TO_M": 0.3048,
    "LB_TO_KG": 0.45359237,
    "MPH_TO_M_PER_S": 0.44704,
}

# Oracle 1: the original vault table, captured before the change.
ORIGINAL_VAULT_TABLE = {
    "distance_mi_to_km": 1.609344,
    "energy_kcal_to_kj": 4.184,
    "pace_min_per_mi_to_min_per_km": 0.621371192237334,
    "speed_mi_per_hr_to_km_per_hr": 1.609344,
    "mass_lb_to_kg": 0.45359237,
    "temperature_degF_to_degC": 0.5555555555555556,
    "temperature_degF_to_degC_offset": -32.0,
    "length_ft_to_m": 0.3048,
    "length_in_to_cm": 2.54,
    "speed_ft_per_s_to_m_per_s": 0.3048,
}

# Oracle 2: the original normalize entries that carry a shared factor.
ORIGINAL_NORMALIZE_SHARED = {
    ("length", "ft"): (0.3048, 0.0),
    ("mass", "lb"): (0.45359237, 0.0),
    ("speed", "mi/hr"): (0.44704, 0.0),
    ("speed", "mph"): (0.44704, 0.0),
    ("speed", "ft/s"): (0.3048, 0.0),
}

# Oracle 3: sha256 of the full original repr of each table, so no entry can
# drift unnoticed, not just the shared ones.
ORIGINAL_UNIT_FAMILIES_SHA256 = (
    "b2f963cd53272506a973710efebec1393c7b4c98bc5c4c4e722d0f4271a9e4e1"
)
ORIGINAL_VAULT_TABLE_SHA256 = (
    "ecc474935b43da1eeca196d98b2efccc66f51330580da03f739d7884007e5d14"
)


def _float_literal_files(value: float) -> set[str]:
    """Names of every health_advisor/*.py file holding this float literal."""
    hits: set[str] = set()
    for path in sorted(PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, float)
                    and node.value == value):
                hits.add(path.name)
                break
    return hits


def test_each_shared_factor_literal_lives_in_exactly_one_file():
    for name, value in SHARED_LITERALS.items():
        files = _float_literal_files(value)
        assert files == {"unit_factors.py"}, (
            f"{value!r} ({name}) must be defined only in unit_factors.py, "
            f"found in {sorted(files)}"
        )


def test_unit_factors_constants_hold_the_original_values():
    for name, value in SHARED_LITERALS.items():
        assert repr(getattr(uf, name)) == repr(value), name


def test_vault_table_matches_original_literals_byte_for_byte():
    assert V.UNIT_CONVERSION_FACTORS.keys() == ORIGINAL_VAULT_TABLE.keys()
    for key, original in ORIGINAL_VAULT_TABLE.items():
        assert repr(V.UNIT_CONVERSION_FACTORS[key]) == repr(original), key
    assert hashlib.sha256(
        repr(V.UNIT_CONVERSION_FACTORS).encode()
    ).hexdigest() == ORIGINAL_VAULT_TABLE_SHA256


def test_normalize_table_matches_original_literals_byte_for_byte():
    for (fam, unit), (factor, offset) in ORIGINAL_NORMALIZE_SHARED.items():
        got_factor, got_offset = nz.UNIT_FAMILIES[fam][unit]
        assert repr(got_factor) == repr(factor), (fam, unit)
        assert repr(got_offset) == repr(offset), (fam, unit)
    assert hashlib.sha256(
        repr(nz.UNIT_FAMILIES).encode()
    ).hexdigest() == ORIGINAL_UNIT_FAMILIES_SHA256


def test_consumers_reference_the_single_constant():
    # Consumer comparison: each shared entry in both tables is the same value
    # as the unit_factors constant, and is that very object (not a copy).
    ft = nz.UNIT_FAMILIES["length"]["ft"][0]
    assert repr(ft) == repr(uf.FT_TO_M) and ft is uf.FT_TO_M
    assert repr(V.UNIT_CONVERSION_FACTORS["length_ft_to_m"]) == repr(uf.FT_TO_M)
    assert V.UNIT_CONVERSION_FACTORS["length_ft_to_m"] is uf.FT_TO_M
    assert repr(nz.UNIT_FAMILIES["speed"]["ft/s"][0]) == repr(uf.FT_TO_M)
    assert V.UNIT_CONVERSION_FACTORS["speed_ft_per_s_to_m_per_s"] is uf.FT_TO_M

    lb = nz.UNIT_FAMILIES["mass"]["lb"][0]
    assert repr(lb) == repr(uf.LB_TO_KG) and lb is uf.LB_TO_KG
    assert V.UNIT_CONVERSION_FACTORS["mass_lb_to_kg"] is uf.LB_TO_KG

    for unit in ("mi/hr", "mph"):
        mph = nz.UNIT_FAMILIES["speed"][unit][0]
        assert repr(mph) == repr(uf.MPH_TO_M_PER_S) and mph is uf.MPH_TO_M_PER_S
