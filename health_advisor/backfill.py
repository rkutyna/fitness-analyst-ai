"""One-time historical backfill from Apple Health 'Export All Health Data'.

**A migration entry point, not a runtime module.** Nothing in the package
imports it, and `tests/test_snapshot_migration.py` fails if that changes. The
distinction matters because a full re-run after the receiver has been live
double-counts every day the receiver already replaced with fine samples — so
this must be something a person invokes deliberately, never something a code
path reaches on its own. `--workouts-only` is the safe refresh.

Streams export.xml (hundreds of MB to >1 GB) with iterparse + clear(), so memory
stays flat. Populates records + workouts, then rebuilds daily_metrics. Idempotent:
re-running never duplicates rows (dedupe_key + INSERT OR IGNORE).

Usage:
    python -m health_advisor.backfill --zip export.zip --db data/health.db
    python -m health_advisor.backfill --xml /path/to/export.xml --db data/health.db
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

from . import db
from . import derive
from . import normalize as nz

INNER_XML = "apple_health_export/export.xml"
BATCH = 100_000          # rows buffered before executemany
ROOT_CLEAR_EVERY = 200_000  # elements between root.clear() to bound memory

SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"


@contextmanager
def open_xml(zip_path: str | None, xml_path: str | None, inner: str):
    if xml_path:
        f = open(xml_path, "rb")
        try:
            yield f
        finally:
            f.close()
    else:
        zf = zipfile.ZipFile(zip_path)
        f = zf.open(inner, "r")
        try:
            yield f
        finally:
            f.close()
            zf.close()


_UNIT_RESOLUTIONS: dict = {}


def _unit_resolution(metric: str, raw_unit):
    """`(convert, canonical_unit)` for a (metric, unit), or None if unplaceable.

    Resolved once per pair, as the receiver's parser does with
    `nz.unit_converter`, so the two ingest paths convert identically. The
    refusal is announced once per pair rather than once per sample.
    """
    key = (metric, raw_unit)
    if key not in _UNIT_RESOLUTIONS:
        try:
            _UNIT_RESOLUTIONS[key] = nz.unit_converter(metric, raw_unit)
        except nz.UnitError as exc:
            print(f"dropping {metric} samples: {exc}", file=sys.stderr)
            _UNIT_RESOLUTIONS[key] = None
    return _UNIT_RESOLUTIONS[key]


def _record_rows(attrib: dict):
    """Yield 0+ normalized record-row dicts from one <Record> element's attribs."""
    rtype = attrib.get("type", "")
    raw_value = attrib.get("value")
    start_s, end_s = attrib.get("startDate"), attrib.get("endDate")
    source = attrib.get("sourceName", "")
    if not start_s or not end_s:
        return
    try:
        start_dt = nz.parse_apple_datetime(start_s)
        end_dt = nz.parse_apple_datetime(end_s)
    except ValueError:
        return
    start_utc = nz.to_utc_iso(start_dt)
    end_utc = nz.to_utc_iso(end_dt)
    start_local = nz.local_naive(start_dt)

    # --- sleep: category -> duration minutes, attributed to WAKE (end) day ---
    if rtype == SLEEP_TYPE:
        metrics = nz.SLEEP_VALUE_MAP.get(raw_value or "")
        if not metrics:
            return
        minutes = (end_dt - start_dt).total_seconds() / 60.0
        local_date = nz.local_date_of(end_dt)
        for metric in metrics:
            yield _mk_row(metric, minutes, "min", start_utc, end_utc, start_local,
                          local_date, source, source_metric=f"{rtype}:{metric}",
                          source_value=raw_value)
        return

    # --- other known category types (mindful duration, stand-hour flag) ---
    if rtype in nz.HK_CATEGORY:
        spec = nz.HK_CATEGORY[rtype]
        local_date = nz.local_date_of(start_dt)
        if spec["mode"] == "duration":
            val = (end_dt - start_dt).total_seconds() / 60.0
            unit = "min"
        elif spec["mode"] == "flag":
            val = 1.0 if raw_value in spec.get("positive", set()) else 0.0
            unit = "count"
        else:  # count
            val = 1.0
            unit = "count"
        yield _mk_row(spec["metric"], val, unit, start_utc, end_utc, start_local,
                      local_date, source, source_metric=f"{rtype}:{spec['metric']}",
                      source_value=raw_value)
        return

    # --- quantity types ---
    if rtype.startswith("HKQuantityTypeIdentifier"):
        if raw_value is None:
            return
        try:
            value = float(raw_value)
        except ValueError:
            return
        metric = nz.hk_quantity_to_canonical(rtype)
        resolved = _unit_resolution(metric, attrib.get("unit"))
        if resolved is None:
            return          # a unit that cannot be placed is dropped, not stored
        convert, unit = resolved
        value = nz.canonical_value(metric, convert(value))
        local_date = nz.local_date_of(start_dt)
        yield _mk_row(metric, value, unit, start_utc, end_utc, start_local, local_date,
                      source, source_metric=rtype, source_value=raw_value)
        return
    # everything else (unmapped category events) -> skipped


def _mk_row(metric, value, unit, start_utc, end_utc, start_local, local_date, source,
            *, source_metric=None, source_value=None):
    return dict(
        metric=metric, value=value, unit=unit, start_utc=start_utc, end_utc=end_utc,
        start_local=start_local, local_date=local_date, source=source, origin="backfill",
        dedupe_key=db.record_key(metric, start_utc, end_utc, value, unit, source,
                                 source_metric=source_metric, source_value=source_value),
    )


_EVENT_TYPE_PREFIX = "HKWorkoutEventType"


def _event_rows(elem, workout_dedupe_key):
    """Yield workout_events rows from a <Workout>'s <WorkoutEvent> children
    (segments, laps, pause/resume markers). Durations arrive in minutes
    (durationUnit="min"); instantaneous events have none."""
    for child in elem:
        if child.tag != "WorkoutEvent":
            continue
        raw_type = child.get("type", "")
        date_s = child.get("date")
        if not raw_type.startswith(_EVENT_TYPE_PREFIX) or not date_s:
            continue
        try:
            start_dt = nz.parse_apple_datetime(date_s)
        except ValueError:
            continue
        # SegmentEvent -> segment, MotionPaused -> motion_paused
        name = raw_type[len(_EVENT_TYPE_PREFIX):]
        etype = "".join(("_" + c.lower()) if c.isupper() else c for c in name).lstrip("_")
        duration_min = _to_float(child.get("duration"))
        unit = (child.get("durationUnit") or "min").lower()
        if duration_min is not None:
            if unit.startswith("sec"):
                duration_min /= 60.0
            elif unit.startswith("hour") or unit == "hr":
                duration_min *= 60.0
        start_utc = nz.to_utc_iso(start_dt)
        end_utc = None
        if duration_min is not None:
            end_utc = nz.to_utc_iso(start_dt + timedelta(minutes=duration_min))
        yield dict(
            workout_key=workout_dedupe_key, event_type=etype, start_utc=start_utc,
            end_utc=end_utc, duration_min=duration_min,
            dedupe_key=db.workout_event_key(workout_dedupe_key, etype, start_utc, duration_min),
        )


def _workout_row(elem):
    a = elem.attrib
    start_s, end_s = a.get("startDate"), a.get("endDate")
    if not start_s or not end_s:
        return None
    try:
        start_dt = nz.parse_apple_datetime(start_s)
        end_dt = nz.parse_apple_datetime(end_s)
    except ValueError:
        return None
    label = nz.workout_label(a.get("workoutActivityType", "Unknown"))
    start_utc, end_utc = nz.to_utc_iso(start_dt), nz.to_utc_iso(end_dt)
    duration_min = _to_float(a.get("duration"))

    energy_kcal = None
    distance_mi = None
    unit_distance = None
    route_ref = None
    for child in elem:
        if child.tag == "WorkoutStatistics":
            ctype = child.get("type", "")
            if ctype == "HKQuantityTypeIdentifierActiveEnergyBurned":
                energy_kcal = _to_float(child.get("sum"))  # 'Cal' == kcal
            elif ctype in ("HKQuantityTypeIdentifierDistanceWalkingRunning",
                           "HKQuantityTypeIdentifierDistanceCycling"):
                miles = _to_float(child.get("sum"))  # export distance is already 'mi'
                if miles is not None:
                    distance_mi = miles
                    unit_distance = "mi"
        elif child.tag == "WorkoutRoute":
            for gc in child:
                if gc.tag == "FileReference":
                    route_ref = gc.get("path")
    return dict(
        workout_type=label, start_utc=start_utc, end_utc=end_utc,
        local_date=nz.local_date_of(start_dt), duration_min=duration_min,
        energy_kcal=energy_kcal, distance_mi=distance_mi, unit_distance=unit_distance,
        source=a.get("sourceName", ""), route_ref=route_ref,
        dedupe_key=db.workout_key(label, start_utc, end_utc),
    )


def _to_float(s):
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def _reattribute_sleep(conn) -> int:
    """Apply session attribution over every stored sleep-stage day; commit.

    Returns the number of records whose `local_date` moved. Idempotent: the
    rule is a function of the timestamps, so a second run moves nothing.
    """
    ph = ",".join("?" * len(derive._STAGE_METRICS))
    first, last = conn.execute(
        f"SELECT MIN(local_date), MAX(local_date) FROM records "
        f"WHERE metric IN ({ph})", derive._STAGE_METRICS).fetchone()
    if first is None:
        return 0
    moves = derive.reattribute_sleep(conn, first, last, apply=True)
    conn.commit()
    return len(moves)


def _reconcile_workout_hr(conn, span) -> int:
    """Reconcile workout avg/max HR over the backfilled local_date span; commit.

    The receiver runs `db.reconcile_workout_heart_rate` after every batch; a
    backfill applies the same rule over the range it wrote (#96). An export.zip
    workout carries no HR summary, so on first backfill this mostly FILLS a
    missing figure from the samples; a summary that contradicts a dense series
    is overruled exactly as at ingest, behind the same evidence gates.
    Idempotent: a second run changes nothing. A failure is reported and logged,
    never raised -- every row the backfill wrote is already committed.
    """
    if not span:
        return 0
    try:
        n = db.reconcile_workout_heart_rate(conn, since=span[0], until=span[1])
        conn.commit()
        return n
    except Exception as exc:  # noqa: BLE001 - the backfilled rows are durable
        if conn.in_transaction:
            conn.rollback()
        print(f"workout HR reconciliation failed: {exc!r}", file=sys.stderr)
        db.log_ingest(conn, "backfill", "reconcile_hr_error", 0, 0, repr(exc)[:500])
        return 0


def run(zip_path=None, xml_path=None, *, db_path, inner=INNER_XML,
        workouts_only=False) -> dict:
    """workouts_only=True ingests only workouts + workout_events and leaves
    records/daily_metrics untouched. Use it to refresh a live DB from a new
    export.zip: a FULL re-run would re-add coarse backfill records for days the
    receiver has already replaced with fine samples (double-counting them)."""
    conn = db.connect(db_path)
    db.init_db(conn)

    rec_buf: list[dict] = []
    wk_buf: list[dict] = []
    ev_buf: list[dict] = []
    rec_seen = rec_added = wk_seen = wk_added = ev_seen = ev_added = skipped = 0
    n = 0
    # The local_date span of the workouts and heart-rate samples this run
    # backfilled: the range the workout HR reconciliation covers (#96).
    hr_span: list[str] = []

    def _widen(day):
        if day:
            hr_span[:] = [min(hr_span[0], day), max(hr_span[1], day)] \
                if hr_span else [day, day]

    with open_xml(zip_path, xml_path, inner) as stream:
        it = ET.iterparse(stream, events=("start", "end"))
        _, root = next(it)  # HealthData root
        for event, elem in it:
            if event != "end":
                continue
            tag = elem.tag
            if tag == "Record":
                if workouts_only:
                    elem.clear()
                    n += 1
                    continue
                rows = list(_record_rows(elem.attrib))
                if rows:
                    rec_buf.extend(rows)
                    rec_seen += len(rows)
                    for row in rows:
                        if row["metric"] == "heart_rate":
                            _widen(row["local_date"])
                else:
                    skipped += 1
                elem.clear()
            elif tag == "Workout":
                wr = _workout_row(elem)
                if wr:
                    wk_buf.append(wr)
                    wk_seen += 1
                    _widen(wr["local_date"])
                    evs = list(_event_rows(elem, wr["dedupe_key"]))
                    ev_buf.extend(evs)
                    ev_seen += len(evs)
                elem.clear()
            else:
                n += 1
                continue
            n += 1

            if len(rec_buf) >= BATCH:
                rec_added += db.insert_records(conn, rec_buf)
                conn.commit()
                rec_buf.clear()
            if len(wk_buf) >= BATCH:
                wk_added += db.insert_workouts(conn, wk_buf)
                conn.commit()
                wk_buf.clear()
            if len(ev_buf) >= BATCH:
                ev_added += db.insert_workout_events(conn, ev_buf)
                conn.commit()
                ev_buf.clear()
            if n % ROOT_CLEAR_EVERY == 0:
                root.clear()
                print(f"... {n:,} elements  (records seen {rec_seen:,})", file=sys.stderr)

    if rec_buf:
        rec_added += db.insert_records(conn, rec_buf)
    if wk_buf:
        wk_added += db.insert_workouts(conn, wk_buf)
    if ev_buf:
        ev_added += db.insert_workout_events(conn, ev_buf)
    conn.commit()

    sleep_moved = 0
    if workouts_only:
        dm = 0
    else:
        # Session attribution is the rule the receiver applies to every batch:
        # a stage sample belongs to the day its sleep SESSION ends, not the day
        # the sample itself ends. It runs before the full rebuild below, which
        # recomputes every aggregate and the per-month source counts from
        # `records`, so the moved rows need no pair-scoped recompute here.
        sleep_moved = _reattribute_sleep(conn)
        print("Rebuilding daily_metrics ...", file=sys.stderr)
        dm = db.recompute_daily_metrics(conn, full=True)
        conn.commit()
        derived = derive.update_after_ingest(conn, derive.all_source_days(conn), "backfill")
        print(f"Derived {derived} sleep-timing/wear rows", file=sys.stderr)

    hr_reconciled = _reconcile_workout_hr(conn, hr_span)

    db.log_ingest(conn, "backfill", "records", rec_seen, rec_added,
                  detail=f"workouts_seen={wk_seen} workouts_added={wk_added} "
                         f"workout_events_seen={ev_seen} workout_events_added={ev_added} "
                         f"skipped_records={skipped} daily_metric_rows={dm} "
                         f"sleep_records_reattributed={sleep_moved} "
                         f"workouts_hr_reconciled={hr_reconciled}")
    summary = dict(records_seen=rec_seen, records_added=rec_added,
                   workouts_seen=wk_seen, workouts_added=wk_added,
                   workout_events_seen=ev_seen, workout_events_added=ev_added,
                   skipped_records=skipped, daily_metric_rows=dm,
                   sleep_records_reattributed=sleep_moved,
                   workouts_hr_reconciled=hr_reconciled)
    conn.close()
    return summary


def main():
    p = argparse.ArgumentParser(description="Apple Health backfill into SQLite.")
    p.add_argument("--zip", dest="zip_path", default=None, help="path to export.zip")
    p.add_argument("--xml", dest="xml_path", default=None, help="path to export.xml")
    p.add_argument("--db", dest="db_path", required=True,
                   help="path to the vault to backfill into")
    p.add_argument("--inner", dest="inner", default=INNER_XML)
    p.add_argument("--workouts-only", action="store_true",
                   help="ingest only workouts + workout_events; safe on a live DB "
                        "(a full re-run would re-add records the receiver evicted)")
    args = p.parse_args()
    if not args.zip_path and not args.xml_path:
        args.zip_path = "export.zip"
    summary = run(zip_path=args.zip_path, xml_path=args.xml_path,
                  db_path=args.db_path, inner=args.inner,
                  workouts_only=args.workouts_only)
    print("\nBackfill complete:")
    for k, v in summary.items():
        print(f"  {k:20s} {v:,}")


if __name__ == "__main__":
    main()
