"""Nightly subjective check-in storage (stress/soreness/energy/sleep quality,
caffeine, alcohol, food line, running niggle, talk test, notes). One row per
local day in `subjective` with partial
upsert; numeric fields are ALSO mirrored into records/daily_metrics
(source/origin 'checkin', evict-then-insert like the receiver) so summaries,
correlations, and Grafana see them like any other metric. Notes stay in the
subjective table only. sleep_quality is last night's sleep on the wake day.
The food line is the nightly convenience measure from P5-4; running niggle and
talk test are jog-day-only fields from P8-3/P4-5. The original 1-5 soreness
series remains unchanged for hikes and other activities (W7-3)."""
from __future__ import annotations

import sqlite3
from datetime import date

from . import db
from . import normalize as nz

RATING_FIELDS = ("stress", "soreness", "energy", "sleep_quality")
COUNT_FIELDS = ("caffeine_drinks", "alcohol_drinks")
NUMERIC_FIELDS = RATING_FIELDS + COUNT_FIELDS

METRIC_NAMES = {
    "stress": "subjective_stress",
    "soreness": "subjective_soreness",
    "energy": "subjective_energy",
    "sleep_quality": "subjective_sleep_quality",
    "caffeine_drinks": "caffeine_drinks",
    "alcohol_drinks": "alcohol_drinks",
}

CHECKIN_ORIGIN = "checkin"  # records.source AND records.origin for mirrored rows


def _validate(day: str, fields: dict, notes) -> None:
    try:
        date.fromisoformat(day)
    except ValueError:
        raise ValueError("day must be YYYY-MM-DD")
    for f in RATING_FIELDS:
        v = fields.get(f)
        if v is None:
            continue
        if float(v) != int(v) or not 1 <= int(v) <= 5:
            raise ValueError(f"{f} must be an integer 1-5")
    for f in COUNT_FIELDS:
        v = fields.get(f)
        if v is not None and float(v) < 0:
            raise ValueError(f"{f} must be >= 0")
    if fields.get("jog_niggle") not in (None, "y", "n"):
        raise ValueError("jog_niggle must be 'y', 'n', or NULL")
    if fields.get("talk_test") not in (
            None, "comfortable", "not_sure", "not_comfortable"):
        raise ValueError(
            "talk_test must be 'comfortable', 'not_sure', 'not_comfortable', or NULL")
    if all(v is None for v in fields.values()) and not notes:
        raise ValueError("nothing to store — provide at least one field")


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def get_day(conn: sqlite3.Connection, day: str) -> dict | None:
    return _row_to_dict(conn.execute(
        "SELECT * FROM subjective WHERE date = ?", (day,)).fetchone())


def get_range(conn: sqlite3.Connection, start: str, end: str) -> list[dict]:
    rows = []
    for raw in conn.execute(
            "SELECT * FROM subjective WHERE date BETWEEN ? AND ? ORDER BY date",
            (start, end)):
        row = dict(raw)
        row["period"] = row["date"]
        row["field_metrics"] = {
            field: METRIC_NAMES[field]
            for field in RATING_FIELDS
            if row.get(field) is not None
        }
        rows.append(row)
    return rows


def log(conn: sqlite3.Connection, day: str, *, stress=None, soreness=None,
        energy=None, sleep_quality=None, caffeine_drinks=None,
        alcohol_drinks=None, food_note: str | None = None,
        jog_niggle: str | None = None, jog_niggle_detail: str | None = None,
        talk_test: str | None = None, notes: str | None = None) -> dict:
    """Partial upsert + mirror. Unprovided fields keep their prior value.
    Raises ValueError on invalid input. Commits. Returns the stored row."""
    fields = {"stress": stress, "soreness": soreness, "energy": energy,
              "sleep_quality": sleep_quality, "caffeine_drinks": caffeine_drinks,
              "alcohol_drinks": alcohol_drinks, "food_note": food_note,
              "jog_niggle": jog_niggle, "jog_niggle_detail": jog_niggle_detail,
              "talk_test": talk_test}
    _validate(day, fields, notes)

    conn.execute(
        """
        INSERT INTO subjective (date, stress, soreness, energy, sleep_quality,
                                caffeine_drinks, alcohol_drinks, food_note,
                                jog_niggle, jog_niggle_detail, talk_test, notes,
                                updated_at)
        VALUES (:date, :stress, :soreness, :energy, :sleep_quality,
                :caffeine_drinks, :alcohol_drinks, :food_note, :jog_niggle,
                :jog_niggle_detail, :talk_test, :notes, :updated_at)
        ON CONFLICT(date) DO UPDATE SET
            stress          = COALESCE(excluded.stress, subjective.stress),
            soreness        = COALESCE(excluded.soreness, subjective.soreness),
            energy          = COALESCE(excluded.energy, subjective.energy),
            sleep_quality   = COALESCE(excluded.sleep_quality, subjective.sleep_quality),
            caffeine_drinks = COALESCE(excluded.caffeine_drinks, subjective.caffeine_drinks),
            alcohol_drinks  = COALESCE(excluded.alcohol_drinks, subjective.alcohol_drinks),
            food_note       = COALESCE(excluded.food_note, subjective.food_note),
            jog_niggle      = COALESCE(excluded.jog_niggle, subjective.jog_niggle),
            jog_niggle_detail = COALESCE(excluded.jog_niggle_detail, subjective.jog_niggle_detail),
            talk_test       = COALESCE(excluded.talk_test, subjective.talk_test),
            notes           = COALESCE(excluded.notes, subjective.notes),
            updated_at      = excluded.updated_at
        """,
        {"date": day, **fields, "notes": notes, "updated_at": db.utcnow_iso()},
    )

    # Mirror provided numerics into records (evict-then-insert so a re-log
    # replaces, never blends) and recompute the affected daily aggregates.
    pairs, rows = [], []
    for field in NUMERIC_FIELDS:
        value = fields[field]
        if value is None:
            continue
        metric = METRIC_NAMES[field]
        unit = nz.CATALOG[metric]["unit"]
        # Deterministic timestamp: the check-in describes the day, not a
        # moment — a fixed nominal 20:00 marker (the +00:00 offset label
        # notwithstanding) keeps re-logs idempotent and intraday bucketing
        # harmless. Nothing consumes this value as real UTC; local_date
        # drives aggregation.
        start_utc = f"{day}T20:00:00+00:00"
        db.delete_records_for_pairs(conn, [(metric, day)], origin=CHECKIN_ORIGIN)
        rows.append({
            "metric": metric, "value": float(value), "unit": unit,
            "start_utc": start_utc, "end_utc": start_utc,
            "start_local": f"{day} 20:00:00", "local_date": day,
            "source": CHECKIN_ORIGIN, "origin": CHECKIN_ORIGIN,
            "dedupe_key": db.record_key(metric, start_utc, start_utc,
                                        float(value), unit, CHECKIN_ORIGIN),
        })
        pairs.append((metric, day))
    if rows:
        db.insert_records(conn, rows)
        db.recompute_daily_metrics(conn, pairs=pairs)
    conn.commit()
    return get_day(conn, day)
