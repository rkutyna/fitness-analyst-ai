#!/usr/bin/env python3
"""Check that daily_metrics still agrees with records (audit day 2).

`daily_metrics` is a cache: every tool, dashboard and briefing reads it, and
nothing reads `records` directly. It is maintained incrementally by the
receiver, so any path that writes records without recomputing the right
(metric, date) pairs makes it silently stale — and stale here means the coach
narrates a wrong number with full confidence.

This rebuilds the aggregate from `records` into a temp table and diffs it
against the stored table. On a D3-filtered vault, divergences from a dropped
series (category one) or from a bucketed series whose exact sum is preserved
are legitimate; full-resolution divergences and bucketed sum mismatches are
genuine (category two). Read-only; exits 1 for category two or derived-metric
discrepancies so it can run from a timer or CI. An unfiltered database keeps
the existing behavior: every raw aggregate divergence is category two.

    ./.venv/bin/python scripts/verify_daily_metrics.py [--db PATH] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod  # noqa: E402
from health_advisor.derive import DERIVED_METRICS  # noqa: E402
from health_advisor import vault  # noqa: E402

TOL = 1e-6

# Bucketed series preserve their sum *mathematically*, not bit-for-bit: the same
# values summed in a different grouping order differ in the last place. Measured
# on vault #1 (2026-08-25), 2333 of 2346 bucketed rows were bit-identical and the
# other 13 spanned 1.1e-16 to 2.0e-16 relative — every one inside a single ULP.
#
# This is not a relaxation of what counts as a discrepancy, because there is
# nothing to hide in the gap. The smallest *genuine* discrepancy is losing one
# sample from a day, which is 1/6148 = 1.6e-4 relative — a factor of 8e11 above
# the observed noise. 1e-9 sits in that gap with room on both sides: 5e6 times
# the measured ceiling, 733 times the theoretical worst case (n*eps = 1.4e-12
# for n=6148, which is why 1e-12 would be too tight), and 1.6e5 times smaller
# than one lost sample.
SUM_REL_TOL = 1e-9


def sums_match(stored, rebuilt) -> bool:
    """Whether two sums agree to within float summation-order noise."""
    if stored is None or rebuilt is None:
        return stored is rebuilt
    if stored == rebuilt:
        return True
    denominator = max(abs(stored), abs(rebuilt))
    return denominator > 0 and abs(stored - rebuilt) / denominator <= SUM_REL_TOL

# What derived_diffs() knows how to recompute. Kept beside the function and
# checked against derive.DERIVED_METRICS at run time: a derived metric absent
# here is a metric written nightly and verified by nothing.
_DERIVED_COVERED = set(DERIVED_METRICS)


def diffs(conn) -> list[dict]:
    """Rows where the stored aggregate disagrees with a fresh rebuild.

    derive.py's sleep-timing and wear_hours metrics are computed straight into
    daily_metrics and have no rows in `records`, so they are excluded rather
    than reported as missing.
    """
    holes = ",".join("?" * len(DERIVED_METRICS))
    conn.execute("""
        CREATE TEMP TABLE rebuilt AS
        SELECT metric, local_date AS date, COUNT(*) count, SUM(value) sum,
               AVG(value) avg, MIN(value) min, MAX(value) max,
               MAX(value) FILTER (WHERE rn = 1) last, MAX(unit) unit
        FROM (
            SELECT metric, local_date, value, unit,
                   ROW_NUMBER() OVER (
                       PARTITION BY metric, local_date
                       ORDER BY start_utc DESC, end_utc DESC, id DESC
                   ) rn
            FROM records
        ) GROUP BY metric, local_date
    """)
    # Days where two sources describe the same movement don't aggregate by a
    # plain GROUP BY — redo those the way recompute_daily_metrics does, or
    # every arbitrated day would be reported as a discrepancy.
    for metric, date in dbmod.arbitrated_pairs(conn):
        clause, extra = dbmod._arbitration(conn, metric, date)
        conn.execute(
            f"""
            UPDATE rebuilt SET (count, sum, avg, min, max, last) = (
                SELECT COUNT(*), SUM(value), AVG(value), MIN(value), MAX(value),
                       (SELECT value FROM records
                        WHERE metric = ? AND local_date = ?{clause}
                        ORDER BY start_utc DESC, end_utc DESC, id DESC LIMIT 1)
                FROM records WHERE metric = ? AND local_date = ?{clause})
            WHERE metric = ? AND date = ?
            """,
            (metric, date, *extra, metric, date, *extra, metric, date),
        )
    return conn.execute(
        f"""
        SELECT COALESCE(r.metric, d.metric) metric,
               COALESCE(r.date, d.date) date,
               d.count s_count, r.count r_count,
               d.sum s_sum, r.sum r_sum, d.avg s_avg, r.avg r_avg,
               d.last s_last, r.last r_last
        FROM rebuilt r FULL OUTER JOIN daily_metrics d
          ON d.metric = r.metric AND d.date = r.date
        WHERE COALESCE(r.metric, d.metric) NOT IN ({holes})
          AND (r.metric IS NULL OR d.metric IS NULL
           OR d.count <> r.count
           OR (d.sum IS NULL) IS NOT (r.sum IS NULL)
           OR d.sum <> r.sum
           OR ABS(COALESCE(d.avg, 0) - COALESCE(r.avg, 0)) > ?
           OR ABS(COALESCE(d.last, 0) - COALESCE(r.last, 0)) > ?)
        ORDER BY date DESC, metric
        """,
        (*DERIVED_METRICS, TOL, TOL),
    ).fetchall()


def classify_diffs(conn, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split raw aggregate divergences into legitimate and genuine rows.

    The D3 contract is allowlist-based, not data-presence-based. A non-
    allowlisted metric has deliberately lost its raw rows. An allowlisted
    bucketed metric retains only coarser rows, so its count/avg/last may differ
    while its sum must be preserved to within summation-order noise — see
    `sums_match`, and note that "preserved" is a claim about the value, not about
    its bit pattern. Allowlisted, unbucketed metrics retain full raw resolution
    and every divergence remains genuine.
    """
    if not vault.is_vault(conn):
        return [], rows

    legitimate: list[dict] = []
    genuine: list[dict] = []
    for row in rows:
        metric = row["metric"]
        if not vault.raw_series_available(metric):
            legitimate.append(row)
        elif (vault.raw_resolution_seconds(metric)
              and sums_match(row["s_sum"], row["r_sum"])):
            legitimate.append(row)
        else:
            genuine.append(row)
    return legitimate, genuine


def derived_diffs(conn, limit: int, days: list[str]) -> list[dict]:
    """Rows where a stored DERIVED metric disagrees with a fresh re-derive.

    The rebuild above deliberately skips DERIVED_METRICS: sleep timing, wear
    hours, midpoint SD, interval regularity and HR load are computed straight
    into daily_metrics and have no rows in `records` to rebuild from. That left
    them unverified by the one script that claims to verify daily_metrics —
    including every metric the sleep re-derive (E7-1) is about to rewrite.

    So re-run the deriver's own read-only functions and compare. This is not a
    second implementation: it calls exactly what update_for_days() calls, minus
    the writes, so a bug in the deriver is NOT hidden by a matching bug here.
    What it catches is the case that matters — stored values that no longer
    follow from the records underneath them.
    """
    from health_advisor import derive

    holes = ",".join("?" * len(DERIVED_METRICS))
    out: list[dict] = []
    for day in days:
        want = derive.compute_sleep_timing(derive._sleep_intervals(conn, day), day) or {}
        wh = derive.wear_hours(conn, day)
        if wh is not None:
            want["wear_hours"] = wh
        for metric, fn in ((derive.MIDPOINT_SD_METRIC, derive._midpoint_sd_for_day),
                           (derive.REGULARITY_METRIC, derive._interval_regularity_for_day),
                           (derive.HR_LOAD_METRIC, derive._hr_load_for_day)):
            val = fn(conn, day)
            if val is not None:
                want[metric] = val
        want.update(derive._dial_for_day(conn, day))

        stored = {r["metric"]: r["last"] for r in conn.execute(
            f"SELECT metric, last FROM daily_metrics WHERE date = ? "
            f"AND metric IN ({holes})", (day, *DERIVED_METRICS))}

        for metric in sorted(set(want) | set(stored)):
            s, r = stored.get(metric), want.get(metric)
            if s is None or r is None or abs(s - r) > TOL:
                out.append({"date": day, "metric": metric, "s_last": s, "r_last": r,
                            "s_count": None, "r_count": None,
                            "s_sum": None, "r_sum": None, "s_avg": None, "r_avg": None})
                if len(out) >= limit:
                    return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(LOCAL_DB_PATH))
    ap.add_argument("--limit", type=int, default=40, help="max discrepancies to print")
    ap.add_argument("--derived-days", type=int, default=120,
                    help="how many recent days to re-derive and check (0 = all)")
    args = ap.parse_args()

    t0 = time.time()
    conn = dbmod.connect(args.db, read_only=True)
    bad = diffs(conn)
    n_records, n_daily = (
        conn.execute("SELECT COUNT(*) FROM records").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()[0],
    )
    print(f"{args.db}: {n_records:,} records -> {n_daily:,} daily_metrics rows "
          f"({time.time() - t0:.1f}s)")

    from health_advisor import derive
    # Every derived metric must be covered by the pass above, or a metric can be
    # written nightly and verified by nothing — which is the hole this whole
    # second pass exists to close. Checked at runtime rather than trusted.
    _uncovered = set(derive.DERIVED_METRICS) - _DERIVED_COVERED
    if _uncovered:
        raise SystemExit(f"verify is blind to derived metric(s): {sorted(_uncovered)}"
                         " — add them to derived_diffs()")
    source_days = sorted(derive.all_source_days(conn))
    check_days = source_days if args.derived_days == 0 else source_days[-args.derived_days:]
    t1 = time.time()
    bad_derived = derived_diffs(conn, args.limit, check_days)
    print(f"derived metrics: re-derived {len(check_days):,} day(s) "
          f"({time.time() - t1:.1f}s)"
          + ("" if args.derived_days == 0 else f" — pass --derived-days 0 for all "
                                              f"{len(source_days):,}"))

    legitimate, genuine = classify_diffs(conn, bad)
    print(f"category one (legitimate): {len(legitimate)}")
    print(f"category two (genuine): {len(genuine)}")

    if not genuine and not bad_derived:
        if legitimate:
            print("OK — no category-two discrepancies; D3-filtered divergences "
                  "are legitimate, and derived metrics match a fresh re-derive.")
        else:
            print("OK — daily_metrics matches a full rebuild from records, and the "
                  "derived metrics match a fresh re-derive.")
        return 0

    if bad:
        print(f"\n{len(bad)} discrepancy(ies) vs a rebuild from records"
              f"{' (truncated)' if len(bad) >= args.limit else ''}:")
        for r in bad[:args.limit]:
            print(f"  {r['date']}  {r['metric']:<28} "
                  f"count {r['s_count']} vs {r['r_count']}  "
                  f"sum {r['s_sum']} vs {r['r_sum']}  avg {r['s_avg']} vs {r['r_avg']}  "
                  f"last {r['s_last']} vs {r['r_last']}")
        print("\n(stored vs rebuilt). Repair with recompute_daily_metrics(full=True).")

    if bad_derived:
        print(f"\n{len(bad_derived)} derived discrepancy(ies)"
              f"{' (truncated)' if len(bad_derived) == args.limit else ''}:")
        for r in bad_derived:
            print(f"  {r['date']}  {r['metric']:<28} "
                  f"stored {r['s_last']} vs re-derived {r['r_last']}")
        print("\nRepair with derive.update_for_days(conn, days).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
