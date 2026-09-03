"""Module settings must not carry deployment-specific tuning literals."""
from __future__ import annotations

import ast
import re
from datetime import date
from pathlib import Path

import pytest

from health_advisor import analysis, db, vault as vaultmod

PACKAGE = Path(__file__).parents[1] / "health_advisor"
ISO_DATE = re.compile(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}\Z")


def _module_assignments(path: Path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        assignments = []
        if isinstance(node, ast.Assign):
            assignments = node.targets
        elif isinstance(node, ast.AnnAssign):
            assignments = [node.target]
        for target in assignments:
            if isinstance(target, ast.Name):
                yield target.id, node.value, node.lineno


def test_deployment_parameters_are_not_module_level_literals():
    date_literals = []
    hr_ceiling_literals = []
    for module in ("normalize.py", "metrics.py"):
        for name, value, line in _module_assignments(PACKAGE / module):
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if ISO_DATE.fullmatch(value.value):
                    date_literals.append(f"{module}:{line}:{name}")
            if (name == "BLOCK_QUALIFY_HR_MAX"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, (int, float))
                    and 140 <= value.value <= 220):
                hr_ceiling_literals.append(f"{module}:{line}:{name}")

    assert date_literals == [], date_literals
    assert hr_ceiling_literals == [], hr_ceiling_literals


def test_settings_round_trip_and_legacy_defaults(conn, vault):
    assert vault.settings()["workout_source_arbitration_from"] is None
    assert vault.settings()["block_qualify_hr_max"] is None
    assert db.workout_source_arbitration_cutoff(conn) == "2026-08-21"

    vaultmod.set_workout_source_arbitration_from(conn, "2020-01-02")
    vaultmod.set_block_qualify_hr_max(conn, 140)
    conn.commit()

    settings = vault.settings()
    assert settings["workout_source_arbitration_from"] == "2020-01-02"
    assert settings["block_qualify_hr_max"] == pytest.approx(140.0)
    assert db.workout_source_arbitration_cutoff(conn) == "2020-01-02"


def test_every_date_is_an_explicit_vault_setting(conn):
    vaultmod.set_workout_source_arbitration_from(conn, date.min.isoformat())
    conn.commit()
    assert db.workout_source_arbitration_cutoff(conn) == date.min.isoformat()


def test_block_reader_uses_the_vault_ceiling(conn, vault, monkeypatch):
    buckets = [{
        "bucket_start_utc": "2020-01-02T00:00:00Z",
        "pace_min_per_mi": 10.0,
        "hr": 150.0,
    }]
    monkeypatch.setattr(analysis.mx, "bucket_series", lambda *args: buckets)

    assert analysis.longest_block(conn, "start", "end")["qualified_min"] == 0.3
    vaultmod.set_block_qualify_hr_max(conn, 140)
    conn.commit()
    assert analysis.longest_block(conn, "start", "end")["qualified_min"] is None


def test_a_vault_without_vault_meta_resolves_both_legacy_defaults():
    """The reference snapshot predates vault_meta and is opened read-only.

    It has no settings by construction; both readers must resolve the legacy
    defaults instead of raising "no such table: vault_meta" (which took every
    arbitration and impact read down on 2026-09-03).
    """
    import sqlite3
    from health_advisor import db as dbmod, metrics as mx, vault as vt
    bare = sqlite3.connect(":memory:")
    assert vt.workout_source_arbitration_from(bare) is None
    assert vt.block_qualify_hr_max(bare) is None
    assert dbmod.workout_source_arbitration_cutoff(bare) == "2026-08-21"
    assert mx.block_qualify_hr_max(bare) == 155.0
