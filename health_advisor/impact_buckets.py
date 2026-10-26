"""Python port of the impact bucket query in ``metrics._impact_bucket_rows_uncached``.

The SQL computes classified 20-second buckets with one recursive CTE. This
module returns the same rows from indexed raw-row reads: the distribution of
each sample across the buckets it overlaps, the per-bucket sums, the step rate
and the heart-rate mean are all computed here.

Two things must match SQLite bit for bit, so they are reproduced rather than
approximated:

* ``SUM`` and ``AVG`` use SQLite's Kahan-Babuska-Neumaier compensated sum
  (``_kbn_add``), and the values are added in the order SQLite scans them. The
  recursive expansion emits rows breadth-first: every sample's first bucket,
  then every sample's second bucket, and so on. ``GROUP BY`` keeps that order
  within a group, so each group is summed by ``(level, scan index)``. The
  distance and step loops inline the same compensated-sum arithmetic as
  ``_kbn_add`` for speed; the two must stay identical.
* Timestamps are compared as text where the SQL compares text (the workout
  containment test and the arbitration bounds), and parsed to epoch seconds
  only through ``strftime('%s', ...)``, which is exactly what the SQL does.

Only the two window predicates the SQL treats as known have a Python path. Any
other predicate is delegated to ``metrics._impact_bucket_rows_uncached``.
"""
from __future__ import annotations

import bisect
from datetime import date, datetime, timedelta

from . import db as dbmod
from . import metrics as mx

KNOWN_WINDOW_PREDICATES = frozenset({
    "local_date BETWEEN ? AND ?",
    "start_utc >= ? AND start_utc < ?",
})

_BUCKET_S = mx.IMPACT_BUCKET_SECONDS
_SPAN_S = mx.IMPACT_MAX_SAMPLE_SPAN_SECONDS
_BUCKET_MIN = _BUCKET_S / 60.0


def _kbn_add(acc: list, value: float) -> None:
    """Add one value to a ``[total, comp, count]`` accumulator, as the SQL does."""
    s = acc[0]
    t = s + value
    if abs(s) > abs(value):
        acc[1] += (s - t) + value
    else:
        acc[1] += (value - t) + s
    acc[0] = t
    acc[2] += 1


def _trunc_div(numerator: int, denominator: int) -> int:
    """C-style integer division (toward zero), which is what SQLite's ``/`` does on integers."""
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def _first_bucket(row) -> int:
    """The bucket a sample's first overlap falls in (row = local_date, value, start_s, end_s)."""
    return _trunc_div(row[2], _BUCKET_S)


def _workout_candidates(conn, window_args):
    """Epoch bounds of the workouts the SQL considers: local_date within a day of the window."""
    start = (date.fromisoformat(window_args[0][:10])
             - timedelta(days=1)).isoformat()
    end = (date.fromisoformat(window_args[1][:10])
           + timedelta(days=1)).isoformat()
    active = dbmod.workout_mark_condition(conn, "w")
    rows = conn.execute(
        "SELECT CAST(strftime('%s', w.start_utc) AS INTEGER) AS start_s, "
        "CAST(strftime('%s', w.end_utc) AS INTEGER) AS end_s "
        "FROM workouts AS w WHERE w.local_date >= ? AND w.local_date <= ? "
        f"AND {active}",
        (start, end)).fetchall()
    return [(s, e) for s, e in rows if s is not None and e is not None]


def _in_workout_ranges(candidates):
    """Merged half-open bucket-index ranges [lo, hi) where a bucket starts inside a workout.

    A bucket with index b is in a workout (start_s <= b*W < end_s) exactly when
    b is in [ceil(start_s / W), ceil(end_s / W)).
    """
    ranges = []
    for start_s, end_s in candidates:
        lo = -((-start_s) // _BUCKET_S)
        hi = -((-end_s) // _BUCKET_S)
        if lo < hi:
            ranges.append((lo, hi))
    ranges.sort()
    merged: list[list[int]] = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    starts = [lo for lo, _ in merged]
    return starts, merged


def _is_in_workout(bkt, index) -> bool:
    starts, merged = index
    i = bisect.bisect_right(starts, bkt) - 1
    return i >= 0 and bkt < merged[i][1]


def _arbitration_losers(conn, rows, arbitration_window, kind, cutoff):
    """Return the set of row indices that source arbitration removes.

    Mirrors ``db._workout_arbitration_query`` for the distance metric. A
    sample loses when either:

    * it is a post-cutoff iPhone sample and a watch sample on the same
      local date is in the arbitration window (the day rule), or
    * it sits inside a post-cutoff workout (the workout rule) and is an iPhone
      sample while a watch or GymKit sample starts inside that workout, or is a
      watch sample while a GymKit sample does.

    The labels are resolved by ``db._workout_source_labels``, so an unusable
    post-cutoff label raises the same ``ValueError`` the SQL path raises.
    """
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'workouts'"
    ).fetchone() is None:
        return set()
    if len(arbitration_window) != 2:
        raise ValueError("arbitration_window must be (start, end)")
    if kind not in ("utc", "local_date"):
        raise ValueError("arbitration_window_kind must be 'utc' or 'local_date'")
    aw_start, aw_end = arbitration_window
    if (kind == "utc" and aw_start >= aw_end) or \
            (kind == "local_date" and aw_start > aw_end):
        relation = "start < end" if kind == "utc" else "start <= end"
        raise ValueError(f"arbitration_window must have {relation}")

    def scoped(alias):
        if kind == "local_date":
            return (f" AND {alias}.local_date >= ? AND {alias}.local_date <= ?",
                    [aw_start, aw_end])
        return (f" AND {alias}.start_utc >= ? AND {alias}.start_utc < ?",
                [aw_start, aw_end])

    labels = dbmod._workout_source_labels(
        conn, source_table="records", date=None,
        arbitration_window=arbitration_window,
        arbitration_window_kind=kind, scoped=scoped)
    iphone = set(labels["iphone"])
    watch = set(labels["watch"])
    gymkit = set(labels["gymkit"])
    losers: set[int] = set()

    # The day rule: local dates holding a watch sample in the arbitration window.
    watch_days: set[str] = set()
    if watch:
        marks = ",".join("?" for _ in watch)
        scope = ("local_date >= ? AND local_date <= ?" if kind == "local_date"
                 else "start_utc >= ? AND start_utc < ?")
        watch_days = {row[0] for row in conn.execute(
            "SELECT DISTINCT local_date FROM records "
            "WHERE metric = 'distance_walking_running' "
            f"AND source IN ({marks}) AND {scope}",
            (*watch, aw_start, aw_end)).fetchall()}

    # The workout rule: post-cutoff workouts in the arbitration window.
    if kind == "local_date":
        workouts = conn.execute(
            "SELECT start_utc, end_utc FROM workouts AS workout "
            "WHERE workout.local_date >= ? AND workout.local_date >= ? "
            "AND workout.local_date <= ?",
            (cutoff, aw_start, aw_end)).fetchall()
    else:
        start_day = (datetime.fromisoformat(aw_start.replace("Z", "+00:00"))
                     .date() - timedelta(days=1)).isoformat()
        end_day = (datetime.fromisoformat(aw_end.replace("Z", "+00:00"))
                   .date() + timedelta(days=1)).isoformat()
        workouts = conn.execute(
            "SELECT start_utc, end_utc FROM workouts AS workout "
            "WHERE workout.local_date >= ? AND workout.local_date >= ? "
            "AND workout.local_date <= ? AND workout.start_utc >= ? "
            "AND workout.start_utc < ?",
            (start_day, cutoff, end_day, aw_start, aw_end)).fetchall()
    workouts = [(s, e) for s, e in workouts]
    if workouts and (gymkit or watch):
        lo = min(s for s, _ in workouts)
        hi = max(e for _, e in workouts)
        kinds = sorted(gymkit | watch)
        marks = ",".join("?" for _ in kinds)
        gym_starts: list[str] = []
        watch_starts: list[str] = []
        for start_utc, source in conn.execute(
                "SELECT start_utc, source FROM records "
                "WHERE metric = 'distance_walking_running' "
                f"AND start_utc >= ? AND start_utc < ? AND source IN ({marks})",
                (lo, hi, *kinds)).fetchall():
            if source in gymkit:
                gym_starts.append(start_utc)
            if source in watch:
                watch_starts.append(start_utc)
        gym_starts.sort()
        watch_starts.sort()

        def any_start(starts, lo_text, hi_text):
            idx = bisect.bisect_left(starts, lo_text)
            return idx < len(starts) and starts[idx] < hi_text

        order = sorted(range(len(rows)), key=lambda i: rows[i][2])
        row_starts = [rows[i][2] for i in order]
        for w_start, w_end in workouts:
            has_gymkit = any_start(gym_starts, w_start, w_end)
            has_watch = any_start(watch_starts, w_start, w_end)
            if not (has_gymkit or has_watch):
                continue
            first = bisect.bisect_left(row_starts, w_start)
            last = bisect.bisect_left(row_starts, w_end)
            for i in order[first:last]:
                source = rows[i][1]
                if (source in iphone and (has_gymkit or has_watch)) or \
                        (source in watch and has_gymkit):
                    losers.add(i)

    for i, (local_date, source, _start, _value, _s, _e) in enumerate(rows):
        if local_date >= cutoff and source in iphone and local_date in watch_days:
            losers.add(i)
    return losers


def impact_bucket_rows_py(conn, window_predicate: str,
                          window_args: tuple[str, str], *,
                          arbitration_window: tuple[str, str] | None = None,
                          arbitration_window_kind: str = "local_date") -> list[dict]:
    """Return the same rows as ``metrics.impact_bucket_rows``, computed in Python.

    Each row is a dict with keys bkt, local_date, mi, hr, cadence_spm,
    in_workout, is_jog and is_walk, in the order the SQL selects them.
    """
    if window_predicate not in KNOWN_WINDOW_PREDICATES:
        return mx._impact_bucket_rows_uncached(
            conn, window_predicate, window_args,
            arbitration_window=arbitration_window,
            arbitration_window_kind=arbitration_window_kind)
    if arbitration_window is None:
        arbitration_window = window_args
    cutoff = dbmod.workout_source_arbitration_cutoff(conn)

    # The same order as the SQL's indexed scan of records: (local_date, id) for
    # the local window, (start_utc, id) for the UTC window.
    scan_key = ("local_date" if window_predicate.startswith("local_date")
                else "start_utc")

    # Distance: the samples in the window that survive source arbitration and
    # the span guard. Arbitration runs in Python (_arbitration_losers) rather
    # than as the per-row NOT EXISTS subquery the SQL uses, which dominated the
    # cost of the query.
    fetched = conn.execute(
        "SELECT local_date, source, start_utc, value, "
        "CAST(strftime('%s', start_utc) AS INTEGER) AS start_s, "
        "CAST(strftime('%s', end_utc) AS INTEGER) AS end_s "
        "FROM records WHERE metric = 'distance_walking_running' "
        f"AND {window_predicate} "
        "AND CAST(strftime('%s', end_utc) AS INTEGER) "
        "- CAST(strftime('%s', start_utc) AS INTEGER) < ? "
        f"ORDER BY {scan_key}, id",
        (*window_args, _SPAN_S)).fetchall()
    losers = _arbitration_losers(
        conn, fetched, arbitration_window, arbitration_window_kind, cutoff)
    # A NULL value contributes to no SUM, so a bucket of only NULLs never
    # exists; dropping those rows here is the same result.
    distance = [(row[0], row[3], row[4], row[5])
                for i, row in enumerate(fetched)
                if i not in losers and row[3] is not None
                and row[4] is not None and row[5] is not None]

    # Summation order. A sample contributes to bucket b at recursion level
    # b - first, and SQLite's GROUP BY sees the contributions to one bucket in
    # (level, scan index) order. Within one bucket that is (-first, scan index),
    # so visiting the samples once in that order reproduces each group's order.
    mi_sums: dict[tuple[str, int], list] = {}
    for idx in sorted(range(len(distance)),
                      key=lambda i: (-_first_bucket(distance[i]), i)):
        local_date, value, start_s, end_s = distance[idx]
        first = _first_bucket(distance[idx])
        if local_date < cutoff:
            last = first
            whole = True
        else:
            last = _trunc_div(end_s - 1, _BUCKET_S)
            whole = end_s <= start_s
        for bkt in range(first, max(first, last) + 1):
            if whole:
                share = value
            else:
                share = (value * (min(end_s, (bkt + 1) * _BUCKET_S)
                                  - max(start_s, bkt * _BUCKET_S))
                         / (end_s - start_s))
            key = (local_date, bkt)
            acc = mi_sums.get(key)
            if acc is None:
                acc = mi_sums[key] = [0.0, 0.0]
            total = acc[0]
            t = total + share
            if abs(total) > abs(share):
                acc[1] += (total - t) + share
            else:
                acc[1] += (share - t) + total
            acc[0] = t
    buckets = {}
    for key, acc in mi_sums.items():
        total = acc[0] + acc[1]
        if total > 0:
            buckets[key] = total

    # Heart rate: the mean of the samples whose start falls in each bucket.
    heart_sums: dict[tuple[str, int], list] = {}
    for local_date, value, start_s in conn.execute(
            "SELECT local_date, value, "
            "CAST(strftime('%s', start_utc) AS INTEGER) "
            "FROM records WHERE metric = 'heart_rate' "
            f"AND {window_predicate} ORDER BY {scan_key}, id",
            window_args).fetchall():
        if start_s is None:
            continue
        key = (local_date, _trunc_div(start_s, _BUCKET_S))
        acc = heart_sums.get(key)
        if acc is None:
            acc = heart_sums[key] = [0.0, 0.0, 0]
        if value is not None:
            _kbn_add(acc, value)

    # Step cadence: each bucket takes the largest per-source step total. The
    # same summation order as the distance buckets, grouped by (bucket, source).
    step_rows = conn.execute(
        "SELECT source, value, "
        "CAST(strftime('%s', start_utc) AS INTEGER) AS start_s, "
        "CAST(strftime('%s', end_utc) AS INTEGER) AS end_s "
        "FROM records WHERE metric = 'step_count' "
        f"AND {window_predicate} "
        "AND CAST(strftime('%s', end_utc) AS INTEGER) "
        "- CAST(strftime('%s', start_utc) AS INTEGER) < ? "
        f"ORDER BY {scan_key}, id",
        (*window_args, _SPAN_S)).fetchall()
    step_rows = [row for row in step_rows
                 if row[1] is not None and row[2] is not None
                 and row[3] is not None]
    per_source: dict[tuple[int, object], list] = {}
    for idx in sorted(range(len(step_rows)),
                      key=lambda i: (-_trunc_div(step_rows[i][2], _BUCKET_S), i)):
        source, value, start_s, end_s = step_rows[idx]
        first = _trunc_div(start_s, _BUCKET_S)
        whole = end_s <= start_s
        last = first if whole else _trunc_div(end_s - 1, _BUCKET_S)
        for bkt in range(first, max(first, last) + 1):
            if whole:
                share = value
            else:
                share = (value * (min(end_s, (bkt + 1) * _BUCKET_S)
                                  - max(start_s, bkt * _BUCKET_S))
                         / (end_s - start_s))
            key = (bkt, source)
            acc = per_source.get(key)
            if acc is None:
                acc = per_source[key] = [0.0, 0.0]
            total = acc[0]
            t = total + share
            if abs(total) > abs(share):
                acc[1] += (total - t) + share
            else:
                acc[1] += (share - t) + total
            acc[0] = t
    best_steps: dict[int, float] = {}
    for (bkt, _source), acc in per_source.items():
        total = acc[0] + acc[1]
        if bkt not in best_steps or total > best_steps[bkt]:
            best_steps[bkt] = total
    cadence = {bkt: total * (60.0 / _BUCKET_S)
               for bkt, total in best_steps.items()}

    workout_index = _in_workout_ranges(_workout_candidates(conn, window_args))

    out: list[dict] = []
    for (local_date, bkt) in sorted(buckets, key=lambda k: (k[1], k[0])):
        mi = buckets[(local_date, bkt)]
        heart = heart_sums.get((local_date, bkt))
        hr = ((heart[0] + heart[1]) / heart[2]
              if heart is not None and heart[2] else None)
        cadence_spm = cadence.get(bkt)
        inside = _is_in_workout(bkt, workout_index)
        pace = _BUCKET_MIN / mi if mi > 0 else None
        jog = mx.is_jog_bucket(
            in_workout=inside, pace_min_per_mi=pace,
            cadence_spm=cadence_spm, hr=hr)
        walk_mi_min = _BUCKET_MIN / mx.IMPACT_WALK_PACE_MAX
        implausible_mi_ceiling = _BUCKET_MIN / mx.IMPACT_IMPLAUSIBLE_PACE_MIN
        out.append({
            "bkt": bkt,
            "local_date": local_date,
            "mi": mi,
            "hr": hr,
            "cadence_spm": cadence_spm,
            "in_workout": 1 if inside else 0,
            "is_jog": 1 if jog else 0,
            "is_walk": (1 if (not jog and walk_mi_min <= mi
                              <= implausible_mi_ceiling) else 0),
        })
    return out
