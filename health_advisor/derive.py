"""Derived daily metrics — sleep timing and wear coverage — computed from
interval records and upserted into daily_metrics. Incremental: called with the
affected local days right after recompute_daily_metrics. Ingest paths go
through update_after_ingest(), which must never raise.

Encodings: sleep_bedtime/sleep_midpoint are hours since PREVIOUS-day noon
(continuous across midnight); sleep_wake_time is hours since midnight of the
wake day. sleep_latency is a floor — the watch often only starts recording
near sleep onset.

CLI:  python -m health_advisor.derive --backfill   (derive all history)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from . import db
from . import normalize as nz

GAP_MERGE_MIN = 90.0   # sleep intervals closer than this merge into one session
MIN_AWAKE_MIN = 1.0    # awake segments shorter than this aren't awakenings
WEAR_METRIC = "heart_rate"

SLEEP_METRICS = ("sleep_bedtime", "sleep_wake_time", "sleep_midpoint",
                 "sleep_time_in_bed", "sleep_awakenings", "sleep_awake_longest",
                 "sleep_latency")
MIDPOINT_SD_METRIC = "sleep_midpoint_sd_28d"
REGULARITY_METRIC = "sleep_timing_interval_regularity"
HR_LOAD_METRIC = "hr_load_proxy"
# The plan's dial. Jog minutes and continuous-block length were computed on
# demand from `records` and never stored, and correlate.paired_series reads
# `daily_metrics` and nothing else — so no hypothesis, correlation or ACWR
# variant could reference the number the ramp is defined on. That is why not one
# of the twelve pre-registered questions was about running, and why active_energy
# (R^2 0.678 with steps, 0.117 with jog minutes) stood in for training load.
# E8-8; precondition for P8-1.
JOG_MINUTES_METRIC = "jog_minutes"
BLOCK_METRIC = "longest_block_min"
DIAL_METRICS = (JOG_MINUTES_METRIC, BLOCK_METRIC)
DERIVED_METRICS = SLEEP_METRICS + ("wear_hours", MIDPOINT_SD_METRIC,
                                   REGULARITY_METRIC, HR_LOAD_METRIC) + DIAL_METRICS

# Only asleep/awake/in_bed spans: core/deep/rem records duplicate the
# sleep_asleep intervals and would double-count session content.
_STAGE_METRICS = ("sleep_asleep", "sleep_awake", "sleep_in_bed")


@dataclass
class Interval:
    start: datetime
    end: datetime
    metric: str
    rec_id: int | None = None      # set when the interval came from a records row


@dataclass
class Session:
    """One merged sleep episode: its members, its span, and the date it ends on."""
    members: list[Interval]
    start: datetime
    end: datetime

    @property
    def end_date(self) -> str:
        return self.end.date().isoformat()


def sleep_sessions(intervals: list[Interval],
                   gap_min: float = GAP_MERGE_MIN) -> list[Session]:
    """Merge intervals into episodes, splitting wherever the gap exceeds
    `gap_min`. Extracted so attribution and timing cannot drift apart: the
    session boundaries that decide which DATE a sample belongs to are the same
    boundaries that decide what the night's bedtime and wake time are."""
    if not intervals:
        return []
    ivs = sorted(intervals, key=lambda i: i.start)
    out = [Session([ivs[0]], ivs[0].start, ivs[0].end)]
    for it in ivs[1:]:
        # `end` is the running MAX, not the previous member's end: an enveloping
        # in_bed span must not be closed by a shorter stage that ends inside it.
        if (it.start - out[-1].end).total_seconds() / 60 > gap_min:
            out.append(Session([it], it.start, it.end))
        else:
            out[-1].members.append(it)
            out[-1].end = max(out[-1].end, it.end)
    return out


def compute_sleep_timing(intervals: list[Interval], day: str) -> dict | None:
    """Timing metrics for one wake-day. The main sleep session is the longest
    run of intervals after merging gaps <= GAP_MERGE_MIN (naps fall outside).
    Returns None when there are no intervals."""
    if not intervals:
        return None
    sessions = sleep_sessions(intervals)
    ses = max(sessions, key=lambda s: (s.end - s.start).total_seconds())
    main, start, end = ses.members, ses.start, ses.end
    midnight = datetime.fromisoformat(day)          # 00:00 of the wake day
    if start >= midnight + timedelta(hours=12):   # main session starts after wake-day noon:
        return None                                # an evening doze, not overnight sleep
    noon_prev = midnight - timedelta(hours=12)      # previous-day 12:00
    awakes = [it for it in main if it.metric == "sleep_awake"
              and (it.end - it.start).total_seconds() / 60 >= MIN_AWAKE_MIN]
    asleep = [it for it in main if it.metric == "sleep_asleep"]
    mid = start + (end - start) / 2
    out = {
        "sleep_bedtime": (start - noon_prev).total_seconds() / 3600,
        "sleep_wake_time": (end - midnight).total_seconds() / 3600,
        "sleep_midpoint": (mid - noon_prev).total_seconds() / 3600,
        "sleep_time_in_bed": (end - start).total_seconds() / 60,
    }
    # Absence is not zero (E7-2). `sleep_awake` exists for 2019 and 2026 only,
    # yet float(len(awakes)) was written unconditionally — so 2,444 of 2,535
    # stored zeros (96%) meant "the watch does not report this", not "he slept
    # through". wear_hours() in this module already returns None for exactly
    # this reason; awakenings now do too. The test is whether the metric was
    # recorded ANYWHERE that day, not whether it survived MIN_AWAKE_MIN — a
    # measured night whose only wakings were sub-minute is a real zero.
    if any(it.metric == "sleep_awake" for it in intervals):
        out["sleep_awakenings"] = float(len(awakes))
        out["sleep_awake_longest"] = max(
            ((it.end - it.start).total_seconds() / 60 for it in awakes), default=0.0)
    if asleep:
        out["sleep_latency"] = max(
            0.0, (asleep[0].start - start).total_seconds() / 60)
    return out


# --- session attribution (E7-1) --------------------------------------------
#
# `local_date` is assigned at ingest as the local date the SAMPLE ends
# (backfill.py:73, and in the retired Health Auto Export path). That was
# correct while HealthKit gave one
# span per night — measured, sleep_in_bed averaged 374.0 min in 2016, 442.8 in
# 2017, 396.0 in 2018 — and it is wrong now: 2026 samples average 19.7 min at
# 20-40 per night, so every sample ending before midnight is filed under the
# PREVIOUS date. A day's sleep total became two half-nights, and sleep_bedtime
# was clipped at midnight because compute_sleep_timing only ever saw one date's
# rows.
#
# The fix generalises the existing rule rather than replacing it: attribute to
# the date the SESSION ends. On a one-span night that is identical to the old
# behaviour, which is why 2016-2021 measures as zero moves. Only a sample inside
# a midnight-crossing episode moves, and it only ever moves forward.
#
# It rewrites `local_date`, a derived column — the timestamps, values and
# dedupe_keys are untouched, and `local_date` is not part of db.record_key(), so
# nothing is re-identified. Callers must recompute daily_metrics for BOTH the
# old and new dates; see pairs_for_moves().

_ATTRIBUTION_PAD_DAYS = 2


def _reattribution_window(conn, start_day: str, end_day: str) -> list[Interval]:
    """Sleep intervals over [start_day - 2, end_day + 2].

    The padding is what makes the answer independent of the window: a session
    is sessionised from complete membership or not at all, and the longest
    episode in this dataset is 21 h (2022-05-18). Moves are then emitted only
    for records at least one day inside the padding, so no session is ever
    judged from a truncated view of itself.
    """
    lo = (date.fromisoformat(start_day) - timedelta(days=_ATTRIBUTION_PAD_DAYS)).isoformat()
    hi = (date.fromisoformat(end_day) + timedelta(days=_ATTRIBUTION_PAD_DAYS)).isoformat()
    ph = ",".join("?" * len(_STAGE_METRICS))
    rows = conn.execute(
        f"SELECT id, metric, value, start_local, local_date FROM records "
        f"WHERE metric IN ({ph}) AND local_date BETWEEN ? AND ? "
        f"AND start_local IS NOT NULL ORDER BY start_local, id",
        (*_STAGE_METRICS, lo, hi)).fetchall()
    out = []
    for row in rows:
        start = datetime.fromisoformat(row["start_local"])
        out.append(Interval(start, start + timedelta(minutes=float(row["value"] or 0)),
                            row["metric"], row["id"]))
    return out


def reattribute_sleep(conn, start_day: str, end_day: str, *,
                      gap_min: float = GAP_MERGE_MIN,
                      apply: bool = False) -> list[tuple[int, str, str]]:
    """Records whose session ends on a different date than the sample does.

    Returns `(record_id, old_local_date, new_local_date)`, and writes them when
    `apply` is set. Caller commits. Idempotent: a second run over the same span
    returns nothing, because the rule is a function of the timestamps alone.
    """
    ivs = _reattribution_window(conn, start_day, end_day)
    if not ivs:
        return []
    # One day inside the padding: any session containing such a record is
    # wholly inside the loaded window, so its end is the real one.
    lo = (date.fromisoformat(start_day) - timedelta(days=_ATTRIBUTION_PAD_DAYS - 1)).isoformat()
    hi = (date.fromisoformat(end_day) + timedelta(days=_ATTRIBUTION_PAD_DAYS - 1)).isoformat()
    stored = {r.rec_id: r for r in ivs}
    current = {row["id"]: row["local_date"] for row in conn.execute(
        f"SELECT id, local_date FROM records WHERE id IN "
        f"({','.join('?' * len(stored))})", tuple(stored))}

    moves: list[tuple[int, str, str]] = []
    for ses in sleep_sessions(ivs, gap_min=gap_min):
        for it in ses.members:
            old = current[it.rec_id]
            if old == ses.end_date or not (lo <= old <= hi):
                continue
            moves.append((it.rec_id, old, ses.end_date))
    if apply and moves:
        conn.executemany("UPDATE records SET local_date = ? WHERE id = ?",
                         [(new, rid) for rid, _, new in moves])
    return moves


def pairs_for_moves(moves, metrics=_STAGE_METRICS) -> set[tuple[str, str]]:
    """The (metric, date) pairs a set of moves invalidated.

    BOTH sides: a record leaving date D makes D's stored rollup too large and
    D+1's too small, so recomputing only the destination leaves a stale
    aggregate behind — the exact defect verify_daily_metrics.py exists to catch.
    """
    dates = {d for _, old, new in moves for d in (old, new)}
    return {(m, d) for m in metrics for d in dates}


def _sleep_intervals(conn, day: str) -> list[Interval]:
    ph = ",".join("?" * len(_STAGE_METRICS))
    rows = conn.execute(
        f"SELECT metric, value, start_local FROM records "
        f"WHERE metric IN ({ph}) AND local_date = ? AND start_local IS NOT NULL "
        f"ORDER BY start_local", (*_STAGE_METRICS, day)).fetchall()
    out = []
    for row in rows:
        start = datetime.fromisoformat(row["start_local"])
        out.append(Interval(start, start + timedelta(minutes=float(row["value"] or 0)),
                            row["metric"]))
    return out


def wear_hours(conn, day: str) -> float | None:
    """Distinct local hours with >=1 heart_rate sample; None when no samples
    (no row is written for unworn days — absence, not zero)."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT strftime('%H', start_local)) FROM records "
        "WHERE metric = ? AND local_date = ? AND start_local IS NOT NULL",
        (WEAR_METRIC, day)).fetchone()
    return float(row[0]) if row and row[0] else None


def _upsert(conn, metric: str, day: str, value: float) -> None:
    conn.execute(
        "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, last, unit) "
        "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(metric, date) DO UPDATE SET count=1, sum=excluded.sum, "
        "avg=excluded.avg, min=excluded.min, max=excluded.max, last=excluded.last, "
        "unit=excluded.unit",
        (metric, day, value, value, value, value, value,
         nz.canonical_unit(metric, None)))


def _midpoint_sd_for_day(conn, day: str) -> float | None:
    """Rolling SD of sleep_midpoint over the WINDOW_DAYS ending at `day`.

    Imported lazily so derive.py does not gain a hard dependency on scipy at
    import time — sleep_regularity pulls in scipy.optimize, and derive runs on
    the ingest path where an import error would be the worst possible failure.
    """
    from . import sleep_regularity as sr
    rows = conn.execute(
        "SELECT last FROM daily_metrics WHERE metric = 'sleep_midpoint' "
        "AND date > date(?, '-28 days') AND date <= ? AND last IS NOT NULL "
        "ORDER BY date", (day, day)).fetchall()
    return sr.rolling_sd([r["last"] for r in rows])


def _interval_regularity_for_day(conn, day: str) -> float | None:
    """Trailing 28-night interval regularity, or None when it refuses."""
    from . import sleep_regularity as sr
    rows = conn.execute(
        "SELECT b.date AS day, b.last AS bed, w.last AS wake "
        "FROM daily_metrics b JOIN daily_metrics w "
        "ON w.date = b.date AND w.metric = 'sleep_wake_time' "
        "WHERE b.metric = 'sleep_bedtime' "
        "AND b.date > date(?, '-28 days') AND b.date <= ? "
        "AND b.last IS NOT NULL AND w.last IS NOT NULL ORDER BY b.date",
        (day, day)).fetchall()
    nights = [(r["day"], r["bed"], r["wake"]) for r in rows]
    out = sr.interval_regularity(nights)
    return out.get("match_pct") if out.get("status") == "ok" else None


def _hr_load_for_day(conn, day: str) -> float | None:
    """Session-scoped HR load for one day, or None when it cannot be measured.

    Lazy import for the same reason as the midpoint SD: this runs on the ingest
    path, where an import error is the worst possible failure.
    """
    from . import hr_load as hl
    rows = hl.daily_load(conn, day, day)
    if not rows or rows[0]["status"] != "ok":
        return None
    return rows[0]["load"]


def _dial_for_day(conn, day: str) -> dict:
    """Jog minutes and longest continuous block for one local day.

    Absence writes NO row rather than a zero — a day with no running is not a
    day with a zero-length block, and `wear_hours` above already established
    that convention. (`sleep_awakenings` writes 0.0 for absence and 96% of its
    stored zeros mean "not measured"; that is E7-2, and this does not repeat it.)

    Across a multi-session day jog minutes SUM and the block is the MAX: the
    question the block answers is how long he can run continuously, and
    2026-07-17 carries two running workouts.
    """
    from . import analysis as A          # lazy: analysis imports metrics, not derive
    out: dict[str, float] = {}
    rows = A.impact_volume(conn, day, day, by="day")
    jog = rows[0].get("jog_minutes") if rows else None
    if jog is not None:
        out[JOG_MINUTES_METRIC] = float(jog)
    blocks = [A.longest_block(conn, w["start_utc"], w["end_utc"])["bridged_min"]
              for w in conn.execute(
                  "SELECT start_utc, end_utc FROM workouts WHERE local_date = ?",
                  (day,))]
    if blocks:
        out[BLOCK_METRIC] = max(blocks)
    return out


def update_for_days(conn, days) -> int:
    """Recompute derived rows for the given local days. Idempotent; removes
    derived rows that are no longer computable. Caller commits."""
    written = 0
    prepared = []
    for day in sorted(set(days)):
        vals = compute_sleep_timing(_sleep_intervals(conn, day), day) or {}
        wh = wear_hours(conn, day)
        if wh is not None:
            vals["wear_hours"] = wh
        vals.update(_dial_for_day(conn, day))
        prepared.append((day, vals))
        for m, v in vals.items():
            _upsert(conn, m, day, v)
            written += 1

    midpoint_sds = {
        day: _midpoint_sd_for_day(conn, day) for day, _ in prepared
    }
    for day, vals in prepared:
        # Compute the trailing-window metrics BEFORE deciding what to delete.
        # They were previously deleted unconditionally and restored only if they
        # recomputed non-None — and their inputs are OTHER days' derived rows,
        # so a bulk re-derive that reached a day before its 28-day window
        # existed deleted rows it could not rebuild and never came back. That is
        # I2: a clean three-day hole at 2026-08-06 -> 08-08, found 2026-08-16.
        # Ordering the caller's days ascending is still required; this makes the
        # delete honest rather than relying on that discipline alone.
        window = {}
        sd = midpoint_sds[day]
        if sd is not None:
            window[MIDPOINT_SD_METRIC] = sd
        regularity = _interval_regularity_for_day(conn, day)
        if regularity is not None:
            window[REGULARITY_METRIC] = regularity
        load = _hr_load_for_day(conn, day)
        if load is not None:
            window[HR_LOAD_METRIC] = load
        for m in DERIVED_METRICS:
            if m not in vals and m not in window:
                conn.execute("DELETE FROM daily_metrics WHERE metric = ? AND date = ?",
                             (m, day))
        for m, v in window.items():
            _upsert(conn, m, day, v)
    return written


def all_source_days(conn) -> list[str]:
    ph = ",".join("?" * (len(_STAGE_METRICS) + 1))
    rows = conn.execute(
        f"SELECT DISTINCT local_date FROM records WHERE metric IN ({ph})",
        (*_STAGE_METRICS, WEAR_METRIC)).fetchall()
    return [r[0] for r in rows]


def update_after_ingest(conn, days, source: str,
                        errors: list[str] | None = None) -> int:
    """Ingest-path wrapper: a derive bug must never fail an ingest request.
    Logs failures to ingest_log and returns 0.

    Swallowing is correct — the raw records are the truth, and losing a whole
    batch to a derived-metric bug would be the worse trade. But swallowing
    silently is not: the ingest_log row this writes was never read by anything,
    so sleep timing, wear, regularity and hr_load_proxy could stop being written
    and the first sign would be a briefing quietly missing a number. Pass
    `errors` to get the failure back and say so upstream; the receiver does.
    """
    try:
        n = update_for_days(conn, days)
        conn.commit()
        return n
    except Exception as e:  # noqa: BLE001 — deliberate catch-all at the ingest boundary
        conn.rollback()
        detail = repr(e)[:500]
        db.log_ingest(conn, source, "derive_error", 0, 0, detail)
        if errors is not None:
            errors.append(detail)
        return 0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Derive sleep-timing/wear daily metrics.")
    ap.add_argument("--backfill", action="store_true", help="derive all history")
    ap.add_argument("--db", required=True, help="path to the vault to derive into")
    args = ap.parse_args()
    if not args.backfill:
        ap.error("nothing to do: pass --backfill (incremental runs happen at ingest)")
    conn = db.connect(args.db)
    try:
        db.init_db(conn)
        days = all_source_days(conn)
        n = update_for_days(conn, days)
        conn.commit()
        print(f"derived {n} metric-day rows over {len(days)} days")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
