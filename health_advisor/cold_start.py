"""Deterministic explanations for surfaces that are still warming up.

This module only measures the existing analytical guards.  It does not own a
second set of readiness, ACWR, correlation, or mover thresholds.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

def _first_date(conn: sqlite3.Connection, as_of: str) -> str | None:
    row = conn.execute(
        "SELECT MIN(date) FROM daily_metrics WHERE date <= ?", (as_of,)
    ).fetchone()
    return row[0] if row and row[0] else None


def _probe(conn: sqlite3.Connection, as_of: str, horizon: str,
           metrics: set[str]) -> sqlite3.Connection:
    """Make a small read-only analytical copy with dense future rows.

    A cold-start answer has to be useful before the first qualifying day is
    present.  The probe carries the latest observed row for each requested
    metric forward, then asks the real surface about each candidate day.  The
    source vault is never changed, and the surface's own calculation remains
    the authority for the flip.
    """
    probe = sqlite3.connect(":memory:")
    probe.row_factory = sqlite3.Row
    probe.executescript(
        """
        CREATE TABLE daily_metrics (
            metric TEXT NOT NULL, date TEXT NOT NULL, count INTEGER,
            sum REAL, avg REAL, min REAL, max REAL, last REAL, unit TEXT,
            source_kind TEXT, PRIMARY KEY (metric, date)
        );
        CREATE TABLE workouts (workout_type TEXT, local_date TEXT);
        CREATE TABLE vault_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    if metrics:
        marks = ",".join("?" for _ in metrics)
        # Fixture vaults (and pre-D19 schemas) may lack source_kind; select
        # only the columns that exist and let the probe carry NULL for the rest.
        have = {row[1] for row in conn.execute("PRAGMA table_info(daily_metrics)")}
        wanted = ["metric", "date", "count", "sum", "avg", "min", "max", "last",
                  "unit", "source_kind"]
        select = ", ".join(c if c in have else f"NULL AS {c}" for c in wanted)
        rows = conn.execute(
            f"SELECT {select} "
            f"FROM daily_metrics WHERE date <= ? AND metric IN ({marks}) "
            "ORDER BY date, metric", (as_of, *sorted(metrics))
        ).fetchall()
    else:
        rows = []
    for row in rows:
        probe.execute(
            "INSERT INTO daily_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(row),
        )
    for row in conn.execute("SELECT workout_type, local_date FROM workouts"):
        probe.execute("INSERT INTO workouts VALUES (?, ?)", tuple(row))
    try:
        for row in conn.execute("SELECT key, value FROM vault_meta"):
            probe.execute("INSERT INTO vault_meta VALUES (?, ?)", tuple(row))
    except sqlite3.OperationalError:
        pass

    d0 = date.fromisoformat(as_of)
    end = date.fromisoformat(horizon)
    for metric in metrics:
        latest = probe.execute(
            "SELECT metric, date, count, sum, avg, min, max, last, unit, source_kind "
            "FROM daily_metrics WHERE metric = ? ORDER BY date DESC LIMIT 1",
            (metric,),
        ).fetchone()
        if latest is None:
            continue
        d = date.fromisoformat(latest["date"]) + timedelta(days=1)
        while d <= end:
            probe.execute(
                "INSERT OR IGNORE INTO daily_metrics "
                "(metric, date, count, sum, avg, min, max, last, unit, source_kind) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (latest["metric"], d.isoformat(), latest["count"], latest["sum"],
                 latest["avg"], latest["min"], latest["max"], latest["last"],
                 latest["unit"], latest["source_kind"]),
            )
            d += timedelta(days=1)
    probe.commit()
    return probe


def _starts_on_day(conn: sqlite3.Connection, first: str, as_of: str,
                   surface: str, metric_x: str | None,
                   metric_y: str | None) -> int | None:
    from . import analysis as A
    from . import correlate as C

    horizon = (date.fromisoformat(as_of) + timedelta(days=365)).isoformat()
    needed = {
        "readiness": {"heart_rate_variability", "resting_heart_rate",
                       "sleep_asleep"},
        "training_load": set(A.ACWR_LOAD_METRICS) | {"wear_hours"},
        "correlate": {m for m in (metric_x, metric_y) if m},
        "movers": {row[0] for row in conn.execute(
            "SELECT DISTINCT metric FROM daily_metrics WHERE date <= ?", (as_of,)
        )},
    }[surface]
    probe = _probe(conn, as_of, horizon, needed)
    try:
        for day_number in range(1, 366):
            candidate = (date.fromisoformat(first)
                         + timedelta(days=day_number - 1)).isoformat()
            if surface == "readiness":
                result = A.readiness(probe, candidate,
                                     _include_cold_start=False)
                ready = result["status"] == "ok"
            elif surface == "training_load":
                result = A.training_load(probe, candidate,
                                         _include_cold_start=False)
                ready = result["status"] == "ok"
            elif surface == "movers":
                result = A.movers(probe, candidate, _include_cold_start=False)
                ready = result["status"] != "insufficient_history"
            else:
                if not metric_x or not metric_y:
                    return None
                xs, ys, _ = C.paired_series(
                    probe, metric_x, metric_y, 0, first, candidate)
                ready = len(xs) >= C.CORRELATION_MIN_PAIRS
            if ready:
                return day_number
    finally:
        probe.close()
    return None


def _status_text(surface: str, status: str, starts_on_day: int | None,
                 day_now: int, as_of: str, reason: str | None,
                 pairs_now: int | None = None) -> str | None:
    if surface == "readiness" and status == "partial":
        return reason
    if surface == "movers" and status == "nothing_moved":
        from . import analysis as A
        return ("nothing has moved by more than "
                f"{A.MOVER_MIN_EFFECT_SD:g} SD in the last {A.MOVER_WINDOW_DAYS} days")
    if starts_on_day is None:
        return reason
    if surface == "correlate":
        return (f"correlate starts after {starts_on_day} paired days; "
                f"you have {pairs_now or 0} paired days ({as_of})")
    return (f"{surface.replace('_', ' ')} starts on day {starts_on_day} of data; "
            f"you are on day {day_now} ({as_of})")


_NO_DATA_MINIMUMS = {
    "readiness": ("at least 14 days of HRV and resting HR", "establishing_baseline"),
    "training_load": ("at least 21 days of workouts", "insufficient_history"),
    "correlate": ("at least 8 paired days", "insufficient_data"),
    "coverage": ("the first synced day", "available"),
    "movers": ("28 days of data", "insufficient_history"),
}


def _no_data_block(surface: str, as_of: str) -> dict:
    """The block for a vault with no records: honest, dated, and not a flip."""
    needs, status = _NO_DATA_MINIMUMS.get(surface, ("more data", "insufficient_data"))
    return {
        "status": status,
        "starts_on_day": None,
        "day_now": 0,
        "as_of": as_of,
        "starts_on_date": None,
        "reason": "no health data has synced to this vault yet",
        "status_text": (f"no health data has synced yet; {surface.replace('_', ' ')} starts after "
                        f"{needs}"),
    }


def describe(conn: sqlite3.Connection, as_of: str,
             *, metric_x: str | None = None,
             metric_y: str | None = None,
             surfaces: set[str] | None = None) -> dict:
    """Describe cold-start state for the analytical surfaces.

    ``starts_on_day`` is found by asking the real surface on a projected dense
    continuation of the observed vault.  It is therefore a measured flip, not
    a copy of a minimum-days constant.
    """
    requested = set(surfaces or {
        "readiness", "training_load", "correlate", "coverage", "movers"
    })
    first = _first_date(conn, as_of)
    if first is None:
        # A vault with no records at all — a tester's day zero, the case this
        # module exists for. Nothing can be walked, so no flip day is claimed;
        # the block says so in words and gives the documented minimum as a
        # floor, never as the measured start.
        return {"days_of_history": 0,
                **{surface: _no_data_block(surface, as_of)
                   for surface in requested}}
    day_now = (date.fromisoformat(as_of) - date.fromisoformat(first)).days + 1
    from . import analysis as A
    from . import correlate as C

    readiness = (A.readiness(conn, as_of, _include_cold_start=False)
                 if "readiness" in requested else None)
    load = (A.training_load(conn, as_of, _include_cold_start=False)
            if "training_load" in requested else None)
    readiness_start = (_starts_on_day(conn, first, as_of, "readiness", None, None)
                       if "readiness" in requested else None)
    load_start = (_starts_on_day(conn, first, as_of, "training_load", None, None)
                  if "training_load" in requested else None)

    correlation = {"status": "not_requested"}
    correlation_start = None
    pairs_now = None
    if "correlate" in requested and metric_x and metric_y:
        xs, ys, _ = C.paired_series(conn, metric_x, metric_y, 0, first, as_of)
        correlation_result = C.correlate(xs, ys)
        pairs_now = len(xs)
        correlation_start = _starts_on_day(
            conn, first, as_of, "correlate", metric_x, metric_y)
        correlation = {"status": correlation_result["status"],
                       "n_pairs": pairs_now}

    movers = (A.movers(conn, as_of, _include_cold_start=False)
              if "movers" in requested else None)
    mover_start = (_starts_on_day(conn, first, as_of, "movers", None, None)
                   if "movers" in requested else None)

    readiness_reason = None
    if readiness and readiness["status"] == "partial":
        component = "hrv" if "hrv" not in readiness.get("components", {}) else "rhr"
        metric = {"hrv": "heart_rate_variability",
                  "rhr": "resting_heart_rate"}[component]
        label = "HRV" if component == "hrv" else "resting HR"
        present = conn.execute(
            "SELECT 1 FROM daily_metrics WHERE metric = ? AND date <= ? LIMIT 1",
            (metric, as_of),
        ).fetchone() is not None
        condition = (f"no {label} source in this vault" if not present
                     else f"{label} source is stale")
        readiness_reason = (f"partial: {condition}; readiness needs both HRV "
                            "and resting HR")
    elif readiness and readiness["status"] == "establishing_baseline":
        readiness_reason = "readiness needs both HRV and resting HR history"
    elif readiness and readiness["status"] == "stale":
        readiness_reason = "readiness inputs are stale"

    def block(surface, status, starts, reason=None, pairs=None):
        text = _status_text(surface, status, starts, day_now, as_of, reason, pairs)
        return {
            "status": status,
            "starts_on_day": starts,
            "day_now": day_now,
            "as_of": as_of,
            "starts_on_date": ((date.fromisoformat(first)
                                 + timedelta(days=starts - 1)).isoformat()
                                if starts is not None else None),
            "reason": reason,
            "status_text": text,
            **({"pairs_now": pairs} if pairs is not None else {}),
        }

    out = {"days_of_history": day_now}
    if readiness:
        out["readiness"] = block("readiness", readiness["status"],
                                 readiness_start, readiness_reason)
    if load:
        load_reason = ("training load has no qualifying history yet"
                       if load["status"] != "ok" else None)
        out["training_load"] = block("training_load", load["status"],
                                      load_start, load_reason)
    if "correlate" in requested:
        out["correlate"] = block(
            "correlate", correlation["status"], correlation_start,
            f"fewer than {C.CORRELATION_MIN_PAIRS} paired days"
            if correlation["status"] == "insufficient_data" else None,
            pairs_now)
    if "coverage" in requested:
        out["coverage"] = block(
            "coverage", "available", 1,
            "coverage reports active or missing metrics from day 1")
    if movers is not None:
        mover_status = movers["status"]
        mover_reason = ("the mover window is not long enough"
                        if mover_status == "insufficient_history" else None)
        out["movers"] = block("movers", mover_status, mover_start, mover_reason)
    return out


# The short name is useful to callers that want the helper's public contract.
cold_start = describe
