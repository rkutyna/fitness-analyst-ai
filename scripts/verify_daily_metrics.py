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

D19 (#218): a `daily_metrics` row labelled `source_kind = 'apple_consolidated'`
carries Apple's own consolidated daily total in `sum`, which is NOT derivable
from `records` and must not be reported as a mismatch against a rebuild. That
one column is handed to `consolidated_diffs`, which checks it against the
`hk_daily_totals` row it was copied from — an INTEGRITY check, not a check that
Apple's number is right. `count`, `avg` and `last` stay records-derived and stay
compared here. A database with no `source_kind` column behaves exactly as it did
before D19, down to the printed bytes.

    ./.venv/bin/python scripts/verify_daily_metrics.py [--db PATH] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
# Module-qualified: `diffs()` binds a local named `date` in its arbitration
# loop, and a bare `from datetime import date` would sit there as a shadowed
# name waiting for someone to use the class inside that function.
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._vault import LOCAL_DB_PATH  # noqa: E402
from health_advisor import db as dbmod  # noqa: E402
from health_advisor.derive import DERIVED_METRICS  # noqa: E402
from health_advisor.hk_parse import D19_TOTAL_METRICS  # noqa: E402
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

# The label `db.apply_consolidated_totals` writes on a row whose `sum` is
# Apple's figure rather than a sum over `records` (D19; `schema.sql`,
# `daily_metrics.source_kind`).
CONSOLIDATED = "apple_consolidated"

# The metrics D19 pulls from Apple as consolidated daily totals (#218 top
# banner). This is imported from the parser, the wire's single metric catalog,
# rather than duplicated here. It is deliberately not read out of
# `hk_daily_totals`; check 6 exists precisely to catch that table being empty.


def _has_table(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,)).fetchone() is not None


def _has_column(conn, table: str, column: str) -> bool:
    """Whether `table.column` exists. `main` opens the database read-only and
    never calls `init_db`, so a pre-D19 vault genuinely lacks these — detect, do
    not assume. Same probe `_apply_column_migrations` uses at `db.py:161`, and
    indexed the same way: a PRAGMA row is a plain tuple unless the caller set
    `row_factory`, so take column 1 by position rather than by name."""
    return any(row[1] == column
               for row in conn.execute(f"PRAGMA table_info({table})"))


def d19_storage_present(conn) -> bool:
    """Whether this database knows about consolidated totals at all.

    Keyed on the COLUMN, not the table: `source_kind` is what says whether a
    stored `sum` is Apple's figure, so a database that has the label but has
    lost `hk_daily_totals` is not a pre-D19 database — it is a post-D19 database
    missing its totals, which is a discrepancy (check 5), not a reason to skip.
    """
    return _has_column(conn, "daily_metrics", "source_kind")


def diffs(conn) -> list[dict]:
    """Rows where the stored aggregate disagrees with a fresh rebuild.

    derive.py's sleep-timing and wear_hours metrics are computed straight into
    daily_metrics and have no rows in `records`, so they are excluded rather
    than reported as missing.

    So is the `sum` of a row labelled `apple_consolidated` — but only that one
    column, and only for that one label. See the comment in the WHERE below.
    """
    holes = ",".join("?" * len(DERIVED_METRICS))
    # On a database that predates D19 the column does not exist, so substitute
    # the literal every row would have had. The predicate then reduces to the
    # pre-D19 one term for term.
    kind = ("COALESCE(d.source_kind, 'records')"
            if _has_column(conn, "daily_metrics", "source_kind") else "'records'")
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
          -- D19: on a consolidated row `sum` is Apple's figure and is NOT
          -- derivable from `records`; consolidated_diffs() checks it against
          -- hk_daily_totals instead. count/avg/last remain records-derived and
          -- stay checked here: three of the FOUR columns this predicate
          -- actually compares. (min/max/unit are built into the `rebuilt`
          -- SELECT above and have never been compared by anything at all — a
          -- pre-existing gap, filed separately; D19 §Q2 and item 1a. It is
          -- pinned by test 16a so that closing it is a deliberate act.)
          --
          -- A consolidated total for a day with no raw samples has nothing to
          -- rebuild at all, so that pair is handed over whole rather than
          -- reported as a row missing from `records`.
          AND NOT (r.metric IS NULL AND {kind} = '{CONSOLIDATED}')
          AND (r.metric IS NULL OR d.metric IS NULL
           OR d.count <> r.count
           OR ({kind} <> '{CONSOLIDATED}'
               AND ((d.sum IS NULL) IS NOT (r.sum IS NULL) OR d.sum <> r.sum))
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


def _finding(metric, day, check: int, fatal: bool, detail: str) -> dict:
    return {"metric": metric, "date": day, "check": check, "fatal": fatal,
            "detail": detail}


def consolidated_pairs_checked(conn) -> int:
    """How many (metric, date) pairs checks 1-5 examined — the denominator for
    the counter line, so "0 discrepancies" over 0 pairs cannot read as a pass."""
    if not d19_storage_present(conn):
        return 0
    pairs = {(r["metric"], r["date"]) for r in conn.execute(
        "SELECT metric, date FROM daily_metrics WHERE source_kind = ?",
        (CONSOLIDATED,))}
    if _has_table(conn, "hk_daily_totals"):
        pairs |= {(r["metric"], r["local_date"]) for r in conn.execute(
            "SELECT metric, local_date FROM hk_daily_totals")}
    return len(pairs)


def consolidated_diffs(conn) -> list[dict]:
    """INTEGRITY of the consolidated override. NOT a check that the value is right.

    Read that twice before trusting a green run. This compares
    `daily_metrics.sum` against the `hk_daily_totals` row that `sum` was
    **copied from**, so it answers exactly one question — *did the apply step
    run, and is it still in front?* It cannot answer these:

      * **Apple returned a wrong number.** Both sides are the same number. No
        automated check can catch this and none ever will; the only oracle is
        the Health app on the phone, read by hand (#218 Done-when 3). Taking
        Apple's figure as the truth is what #218 decided, and this is what that
        decision costs. A green run here is not a validated number.
      * **The unit was converted wrongly at parse time.** `hk_daily_totals.unit`
        records what we *decided* the unit was, so a wrong conversion is
        consistent with itself on both sides. That belongs at the parse seam
        (step 3), not here.

    What it does catch, in both directions, because each direction is a
    different real failure:

      1. a received total with no `daily_metrics` row at all;
      2. a `daily_metrics` row that exists but is not labelled — the override
         did not run, or a later recompute overwrote it;
      3. a labelled row whose `sum` no longer matches the total it came from
         (`sums_match`, not `==`, for the reason the SUM_REL_TOL comment at
         the head of this file measures out);
      4. a labelled row whose `unit` no longer matches. This is the ONLY unit
         check anywhere in this script — see test 16a;
      5. a label with nothing behind it: a total deleted from `hk_daily_totals`
         by hand. Re-running the override will not bring it back.
      6. **affirmative** — if `vault_meta.daily_totals_expected_from:<metric>`
         is set, that metric's totals must EXIST across the window it opens.
         Checks 1-5 are all
         satisfied by an empty, agreeing world, which is exactly the state a
         rebuild that dropped the totals produces (every row back to
         `source_kind = 'records'`, `hk_daily_totals` empty). Check 5 cannot
         see it — there is no label left to find. Check 6 is built the other way
         round so that absence FAILS instead of passing.
    """
    if not d19_storage_present(conn):
        return []
    totals = {}
    if _has_table(conn, "hk_daily_totals"):
        totals = {(r["metric"], r["local_date"]): r for r in conn.execute(
            "SELECT metric, local_date, value, unit FROM hk_daily_totals")}

    out: list[dict] = []
    for metric, day in sorted(totals):
        want = totals[(metric, day)]
        row = conn.execute(
            "SELECT sum, unit, source_kind FROM daily_metrics "
            "WHERE metric = ? AND date = ?", (metric, day)).fetchone()
        if row is None:
            out.append(_finding(metric, day, 1, True,
                                f"hk_daily_totals holds {want['value']} "
                                f"{want['unit']} but daily_metrics has no row "
                                f"for this pair"))
            continue
        if row["source_kind"] != CONSOLIDATED:
            out.append(_finding(
                metric, day, 2, True,
                f"daily_metrics.source_kind is {row['source_kind']!r}, not "
                f"{CONSOLIDATED!r}: sum {row['sum']} is still records-derived, "
                f"so Apple's total {want['value']} is not the number readers see"))
            continue
        if not sums_match(row["sum"], want["value"]):
            out.append(_finding(metric, day, 3, True,
                                f"sum {row['sum']} does not match the total it "
                                f"was copied from ({want['value']})"))
        if row["unit"] != want["unit"]:
            out.append(_finding(metric, day, 4, True,
                                f"unit {row['unit']!r} does not match the total "
                                f"it was copied from ({want['unit']!r})"))

    orphaned = set()
    for row in conn.execute(
            "SELECT metric, date, sum FROM daily_metrics WHERE source_kind = ? "
            "ORDER BY date, metric", (CONSOLIDATED,)):
        if (row["metric"], row["date"]) not in totals:
            orphaned.add((row["metric"], row["date"]))
            out.append(_finding(
                row["metric"], row["date"], 5, True,
                f"labelled {CONSOLIDATED} with sum {row['sum']}, but "
                f"hk_daily_totals has no row behind it"))

    out.extend(_expected_totals_diffs(conn, totals, orphaned))
    return out


def _as_date(value):
    """A local YYYY-MM-DD, or None if it is not one."""
    try:
        return dt.date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _expected_totals_diffs(conn, totals: dict, orphaned: set) -> list[dict]:
    """Check 6 — the affirmative one. See `consolidated_diffs`.

    How it fails when it fails, stated because an unstated failure mode is a
    trap: a phone that was genuinely off for the whole window produces the same
    shape as a dropped table. So missing days are reported either way, and this
    is **fatal only when the run of missing days starts at
    `daily_totals_expected_from:<metric>` itself** — the dropped-table shape.
    That day is present by construction (`insert_daily_totals` sets the key from
    the first accepted row for that metric), so its absence means something
    removed it. Scattered gaps downstream of it are printed and are not a
    failure. A phone off since the epoch is genuinely indistinguishable from a
    dropped table and will fail; that is the correct direction to be wrong in,
    and one look at `hk_daily_total_revisions` separates the two.
    """
    out: list[dict] = []
    for metric in D19_TOTAL_METRICS:
        row = conn.execute(
            "SELECT value FROM vault_meta WHERE key = ?",
            (f"daily_totals_expected_from:{metric}",)
        ).fetchone() if _has_table(conn, "vault_meta") else None
        # A metric whose first total has never arrived is not yet expected.
        if row is None:
            continue
        # `vault_meta` is free-form TEXT and this key arrives from a payload. A
        # traceback out of a read-only checker is a worse answer than a finding,
        # and an unreadable expectation is itself the thing check 6 is here to
        # notice: it means nothing is being expected of this metric.
        epoch = _as_date(row["value"])
        if epoch is None:
            out.append(_finding(
                metric, row["value"], 6, True,
                f"vault_meta.daily_totals_expected_from:{metric} is "
                f"{row['value']!r}, which is not a date — check 6 cannot run, "
                f"so nothing is verifying that this metric's consolidated "
                f"totals are still present"))
            continue
        last = conn.execute("SELECT MAX(date) FROM daily_metrics WHERE metric = ?",
                            (metric,)).fetchone()[0]
        if last is None:
            continue
        # `- 1`: the newest day is still open on the phone and its total has not
        # been pulled yet, so expecting one there would fire every single run.
        newest = _as_date(last)
        if newest is None:
            continue
        end = newest - dt.timedelta(days=1)
        if end < epoch:
            continue
        window = [epoch + dt.timedelta(days=n)
                  for n in range((end - epoch).days + 1)]
        # A day check 5 already owns is not evidence about the phone: its total
        # was removed from under a live label. Reporting it here as well would
        # print "consistent with the phone being off" about a day we know was
        # deleted, which is worse than not printing it.
        missing = [d for d in window
                   if (metric, d.isoformat()) not in totals
                   and (metric, d.isoformat()) not in orphaned]
        if not missing:
            continue
        leading = 0
        while (leading < len(missing)
               and missing[leading] == epoch + dt.timedelta(days=leading)):
            leading += 1
        shown = ", ".join(d.isoformat() for d in missing[:8])
        if len(missing) > 8:
            shown += f", … (+{len(missing) - 8} more)"
        if leading:
            out.append(_finding(
                metric, missing[0].isoformat(), 6, True,
                f"{len(missing)} of {len(window)} expected day(s) in "
                f"{epoch.isoformat()}..{end.isoformat()} have no hk_daily_totals "
                f"row, and the run starts at the epoch itself "
                f"({epoch.isoformat()}..{missing[leading - 1].isoformat()}, "
                f"{leading} day(s)) — the shape of totals that were LOST, not of "
                f"a phone that was off: {shown}"))
        else:
            out.append(_finding(
                metric, missing[0].isoformat(), 6, False,
                f"{len(missing)} of {len(window)} expected day(s) in "
                f"{epoch.isoformat()}..{end.isoformat()} have no hk_daily_totals "
                f"row, scattered (the epoch {epoch.isoformat()} itself is "
                f"present) — consistent with the phone being off; reported, not "
                f"failed: {shown}"))
    return out


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

    # D19. Printed only when the database has the storage: a pre-D19 vault must
    # produce byte-identical output to what this script produced before D19
    # (test 17), and a counter line reading "0 checked" would be both new bytes
    # and a claim about a thing that does not exist.
    d19 = d19_storage_present(conn)
    bad_consolidated = consolidated_diffs(conn) if d19 else []
    fatal_consolidated = [r for r in bad_consolidated if r["fatal"]]
    if d19:
        print(f"consolidated rows: {consolidated_pairs_checked(conn)} checked, "
              f"{len(bad_consolidated)} discrepancy(ies) "
              f"({len(fatal_consolidated)} fatal) — INTEGRITY only: "
              f"daily_metrics.sum against the hk_daily_totals row it was copied "
              f"from. This does NOT check that Apple's figure is correct; only "
              f"the Health app can (#218).")

    if bad_consolidated:
        # Truncated like the other two lists. An override that never ran at all
        # produces one finding per metric per day, and a screen of ten thousand
        # identical lines is output nobody acts on.
        print(f"\n{len(bad_consolidated)} consolidated-total discrepancy(ies)"
              f"{' (truncated)' if len(bad_consolidated) > args.limit else ''}:")
        for r in bad_consolidated[:args.limit]:
            print(f"  [{'FAIL' if r['fatal'] else 'note'}] check {r['check']}  "
                  f"{r['date']}  {r['metric']:<28} {r['detail']}")
        checks = {r["check"] for r in bad_consolidated}
        # Three repairs, not one: checks 1-4 mean the value is still on disk and
        # the override has to be re-run; check 5 means it is gone from under a
        # label; check 6 means it is gone entirely. Naming
        # apply_consolidated_totals for the last two would be actively
        # misleading — it is a no-op on both.
        if checks & {1, 2, 3, 4}:
            print("\nchecks 1-4 — a received total did not reach the cache. "
                  "Repair with db.apply_consolidated_totals(conn).")
        if 5 in checks:
            print("\ncheck 5 — a label with nothing behind it: a row was removed "
                  "from hk_daily_totals by hand. Re-running the override will "
                  "NOT bring it back; find out what deleted it.")
        if any(r["check"] == 6 and r["fatal"] for r in bad_consolidated):
            print("\ncheck 6 — expected totals are GONE, contiguously from the day "
                  "this deployment started receiving them. apply_consolidated_totals "
                  "is a no-op: there is nothing left to apply. Restore the vault "
                  "from backup, or re-pull that window from the phone. "
                  "hk_daily_total_revisions separates this from a phone that was "
                  "off since the epoch, which looks the same from here.")
        elif 6 in checks:
            print("\ncheck 6 — the missing days above are scattered, which is what a "
                  "phone that was off looks like. Reported, not failed; nothing to "
                  "repair unless you know the phone was on.")

    if not genuine and not bad_derived and not fatal_consolidated:
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
