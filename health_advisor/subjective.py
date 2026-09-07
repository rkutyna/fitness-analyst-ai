"""Subjective check-in storage (stress/soreness/energy/sleep quality, waist
circumference, caffeine, alcohol, food line, running niggle, talk test, notes).
One row per local day in subjective with partial
upsert; numeric fields are ALSO mirrored into records/daily_metrics
(source/origin 'checkin', evict-then-insert like the receiver) so summaries,
correlations, and Grafana see them like any other metric. Notes stay in the
subjective table only. sleep_quality is last night's sleep on the wake day.
The waist field is a self-report and its 14-day cadence is evaluated by
measurement_status; missed windows remain absent. The food line is the nightly
convenience measure from P5-4; running niggle and talk test are jog-day-only
fields from P8-3/P4-5. The original 1-5 soreness series remains unchanged for
hikes and other activities (W7-3)."""
from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta

from . import db
from . import normalize as nz

RATING_FIELDS = ("stress", "soreness", "energy", "sleep_quality")
COUNT_FIELDS = ("caffeine_drinks", "alcohol_drinks")
BODY_MEASUREMENT_METRIC = "waist_circumference"
BODY_MEASUREMENT_FIELD = "waist_circumference"
BODY_MEASUREMENT_CADENCE_DAYS = 14

NUMERIC_FIELDS = RATING_FIELDS + COUNT_FIELDS + (BODY_MEASUREMENT_FIELD,)

METRIC_NAMES = {
    "stress": "subjective_stress",
    "soreness": "subjective_soreness",
    "energy": "subjective_energy",
    "sleep_quality": "subjective_sleep_quality",
    "caffeine_drinks": "caffeine_drinks",
    "alcohol_drinks": "alcohol_drinks",
    BODY_MEASUREMENT_FIELD: BODY_MEASUREMENT_METRIC,
}

CHECKIN_ORIGIN = "checkin"  # records.source AND records.origin for mirrored rows

# This is deliberately a cadence, not a plan date.  The caller supplies the
# plan's effective start, so no person's schedule is embedded in the public
# engine.  A window is complete when it contains a self-report; an unobserved
# closed window is a computable miss.
SELF_REPORT_CADENCES = {BODY_MEASUREMENT_METRIC: BODY_MEASUREMENT_CADENCE_DAYS}


def _as_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be YYYY-MM-DD") from None


def cadence_windows(
        start: str, as_of: str, *,
        metric: str = BODY_MEASUREMENT_METRIC) -> list[dict]:
    """Return computable cadence windows through ``as_of``.

    The final item may be open (its end is after ``as_of``).  Callers can
    therefore distinguish an open ``due`` window from a closed ``missed`` one
    without asking a model to interpret a review sentence.
    """
    if metric not in SELF_REPORT_CADENCES:
        raise ValueError(f"no self-report cadence for {metric!r}")
    first, cutoff = _as_date(start, "start"), _as_date(as_of, "as_of")
    if cutoff < first:
        raise ValueError("as_of must not precede start")
    step = timedelta(days=SELF_REPORT_CADENCES[metric])
    window_start = first
    windows = []
    while window_start <= cutoff:
        windows.append({
            "metric": metric,
            "start": window_start.isoformat(),
            "end": (window_start + step - timedelta(days=1)).isoformat(),
            "cadence_days": step.days,
        })
        window_start += step
    return windows


def measurement_status(conn: sqlite3.Connection, start: str, as_of: str,
                       *, metric: str = BODY_MEASUREMENT_METRIC) -> list[dict]:
    """Classify each cadence window as ``complete``, ``due`` or ``missed``.

    Only the existence of a value in the self-report substrate completes a
    window.  No placeholder or zero is written for a missed window.
    """
    windows = cadence_windows(start, as_of, metric=metric)
    field = next((f for f, m in METRIC_NAMES.items() if m == metric), None)
    if field is None:
        raise ValueError(f"no self-report field for {metric!r}")
    rows = conn.execute(
        f"SELECT date FROM subjective WHERE date BETWEEN ? AND ? "
        f"AND {field} IS NOT NULL ORDER BY date",
        (start, as_of),
    ).fetchall()
    observed = {row["date"] for row in rows}
    cutoff = _as_date(as_of, "as_of")
    out = []
    for window in windows:
        window_end = date.fromisoformat(window["end"])
        present = sorted(day for day in observed
                         if window["start"] <= day <= window["end"])
        status = ("complete" if present else
                  "missed" if window_end < cutoff else "due")
        out.append({**window, "status": status,
                    "observed_dates": present})
    return out


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


def get_range(conn: sqlite3.Connection, start: str, end: str,
              *, unit_system: str | None = None) -> list[dict]:
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
        if row.get(BODY_MEASUREMENT_FIELD) is not None:
            row["field_metrics"][BODY_MEASUREMENT_FIELD] = BODY_MEASUREMENT_METRIC
            from . import vault
            row[BODY_MEASUREMENT_FIELD], row["waist_circumference_unit"] = \
                vault.convert_for_unit_system(
                    row[BODY_MEASUREMENT_FIELD],
                    nz.CATALOG[BODY_MEASUREMENT_METRIC]["unit"], unit_system)
        rows.append(row)
    return rows


def log(conn: sqlite3.Connection, day: str, *, stress=None, soreness=None,
        energy=None, sleep_quality=None, caffeine_drinks=None,
        alcohol_drinks=None, food_note: str | None = None,
        jog_niggle: str | None = None, jog_niggle_detail: str | None = None,
        talk_test: str | None = None, waist_circumference=None,
        waist_circumference_unit: str | None = None,
        unit_system: str | None = None, notes: str | None = None) -> dict:
    """Partial upsert + mirror. Unprovided fields keep their prior value.
    Raises ValueError on invalid input. Commits. Returns the stored row."""
    if waist_circumference is not None:
        try:
            waist_circumference = float(waist_circumference)
        except (TypeError, ValueError):
            raise ValueError(
                "waist_circumference must be a finite positive number") from None
        if not math.isfinite(waist_circumference) or waist_circumference <= 0:
            raise ValueError(
                "waist_circumference must be a finite positive number")
        input_unit = waist_circumference_unit
        if input_unit is None:
            input_unit = "cm" if unit_system == "metric" else nz.CATALOG[
                BODY_MEASUREMENT_METRIC]["unit"]
        waist_circumference = nz.convert_unit(
            waist_circumference, input_unit,
            nz.CATALOG[BODY_MEASUREMENT_METRIC]["unit"])

    fields = {"stress": stress, "soreness": soreness, "energy": energy,
              "sleep_quality": sleep_quality, "caffeine_drinks": caffeine_drinks,
              "alcohol_drinks": alcohol_drinks, "food_note": food_note,
              "jog_niggle": jog_niggle, "jog_niggle_detail": jog_niggle_detail,
              "talk_test": talk_test,
              BODY_MEASUREMENT_FIELD: waist_circumference}
    _validate(day, fields, notes)

    conn.execute(
        """
        INSERT INTO subjective (date, stress, soreness, energy, sleep_quality,
                                caffeine_drinks, alcohol_drinks, food_note,
                                jog_niggle, jog_niggle_detail, talk_test,
                                waist_circumference, notes, updated_at)
        VALUES (:date, :stress, :soreness, :energy, :sleep_quality,
                :caffeine_drinks, :alcohol_drinks, :food_note, :jog_niggle,
                :jog_niggle_detail, :talk_test, :waist_circumference,
                :notes, :updated_at)
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
            waist_circumference = COALESCE(excluded.waist_circumference,
                                           subjective.waist_circumference),
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
