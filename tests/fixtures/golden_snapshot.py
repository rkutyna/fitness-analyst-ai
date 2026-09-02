"""Build the small, read-only source slice used by the golden-figure tests.

The normal test guard deliberately rejects the production database. This helper
copies only the rows needed by ``test_golden_figures`` into a session temporary
database, so the regression net can exercise the real analysis code without
weakening that guard or opening the 4.4 GB snapshot in a test.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from health_advisor import db


# Resolved from the package, not written down: an absolute path here would be
# one machine's, and this file already skips cleanly when the snapshot is absent.
PRODUCTION_SNAPSHOT = db.REPO_ROOT / "data" / "health.db"

_DAILY_RANGES = {
    # RHR's weekly_series() noise floor looks back one year from the review.
    "resting_heart_rate": ("2025-08-17", "2026-08-16"),
    "step_count": ("2026-07-20", "2026-08-16"),
    "body_mass": ("2026-07-20", "2026-08-16"),
    "vo2_max": ("2026-07-20", "2026-08-16"),
}
_RECORD_METRICS = ("body_mass", "distance_walking_running", "heart_rate")
_RECORD_RANGE = ("2026-07-20", "2026-08-16")
_WORKOUT_RANGE = ("2026-07-20", "2026-08-16")
_SCOPED_RECORD_RANGE = ("2025-09-01", "2026-07-19")


def _copy_rows(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    where: str = "",
    args: tuple[object, ...] = (),
) -> None:
    """Copy the columns the target and the snapshot have in common.

    The snapshot is a fixed file from 2026-08-21 and the schema moves on
    without it, so a target column the snapshot predates is expected — it
    takes its NULL and the copy carries on, the same rule
    `vault._copy_table` uses for an additive column.

    A missing column that is NOT NULL with no default is a different thing:
    the snapshot cannot supply a value the schema insists on, and copying
    would either fail on insert or, worse, quietly need a fabricated one. That
    still raises. The guard is about columns that matter, not about the schema
    never changing.
    """
    target_info = list(target.execute(f"PRAGMA table_info({table})"))
    columns = [r[1] for r in target_info]
    source_columns = {
        r[1] for r in source.execute(f"PRAGMA table_info({table})")
    }
    required = {r[1] for r in target_info if r[3] and r[4] is None}
    missing = set(columns) - source_columns
    if missing & required:
        raise RuntimeError(
            f"snapshot table {table} lacks required columns: "
            f"{sorted(missing & required)}"
        )
    columns = [c for c in columns if c in source_columns]
    quoted = ", ".join(f'"{c}"' for c in columns)
    placeholders = ", ".join("?" for _ in columns)
    rows = source.execute(
        f'SELECT {quoted} FROM "{table}" {where}', args
    ).fetchall()
    target.executemany(
        f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})',
        [tuple(row[c] for c in columns) for row in rows],
    )


def build_golden_database(path: Path) -> Path:
    """Create a temporary database containing the golden-test source slice."""
    source = sqlite3.connect(f"file:{PRODUCTION_SNAPSHOT}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(path)
    try:
        target.row_factory = sqlite3.Row
        db.init_db(target)

        for metric, (start, end) in _DAILY_RANGES.items():
            _copy_rows(
                source,
                target,
                "daily_metrics",
                "WHERE metric = ? AND date BETWEEN ? AND ?",
                (metric, start, end),
            )

        start, end = _RECORD_RANGE
        for metric in _RECORD_METRICS:
            _copy_rows(
                source,
                target,
                "records",
                "WHERE metric = ? AND local_date BETWEEN ? AND ?",
                (metric, start, end),
            )

        start, end = _WORKOUT_RANGE
        _copy_rows(
            source,
            target,
            "workouts",
            "WHERE local_date BETWEEN ? AND ? ORDER BY start_utc",
            (start, end),
        )

        # Keep this explicit: an empty manual-jog table is part of the measured
        # snapshot semantics, and impact_volume() consults it on every call.
        _copy_rows(target, target, "manual_jog", "WHERE 1 = 0")
        target.commit()
    finally:
        source.close()
        target.close()
    return path


def build_scoped_database(path: Path) -> Path:
    """Create the source slice used by payload-scoped verification tests."""
    source = sqlite3.connect(f"file:{PRODUCTION_SNAPSHOT}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(path)
    try:
        target.row_factory = sqlite3.Row
        db.init_db(target)
        start, end = _SCOPED_RECORD_RANGE
        for metric in ("distance_walking_running", "heart_rate"):
            _copy_rows(
                source, target, "records",
                "WHERE metric = ? AND local_date BETWEEN ? AND ?",
                (metric, start, end),
            )
        _copy_rows(target, target, "manual_jog", "WHERE 1 = 0")
        target.commit()
    finally:
        source.close()
        target.close()
    return path
