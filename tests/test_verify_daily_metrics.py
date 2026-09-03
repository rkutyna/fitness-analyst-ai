"""D3-aware classification for the daily-metrics verification script."""
from __future__ import annotations

import subprocess
import sys

from health_advisor import db
from health_advisor import derive
from health_advisor import vault
from scripts import verify_daily_metrics as V


DAY = "2026-08-20"


def _record(metric: str, value: float, n: int, day: str = DAY) -> dict:
    start = f"{day}T00:00:{n:02d}+00:00"
    return {
        "metric": metric,
        "value": value,
        "unit": "count/min" if metric == "heart_rate" else "count",
        "start_utc": start,
        "end_utc": start,
        "start_local": start[:-6],
        "local_date": day,
        "source": "test",
        "origin": "backfill",
        "dedupe_key": f"{metric}-{day}-{n}",
    }


def _database(tmp_path, metric: str, values: list[float], *, d3: bool = True,
              name: str | None = None):
    path = tmp_path / f"{name or metric}.db"
    conn = db.connect(path)
    db.init_db(conn)
    db.insert_records(conn, [_record(metric, value, n)
                             for n, value in enumerate(values, start=1)])
    db.recompute_daily_metrics(conn, full=True)
    derive.update_for_days(conn, derive.all_source_days(conn))
    if d3:
        vault.declare_vault(conn)
    conn.commit()
    return path, conn


def _run(path, *extra):
    return subprocess.run(
        [sys.executable, "scripts/verify_daily_metrics.py", "--db", str(path),
         "--derived-days", "0", *extra],
        capture_output=True, text=True,
    )


def _assert_categories(result, *, legitimate: int, genuine: int,
                       returncode: int):
    assert result.returncode == returncode, result.stdout
    assert f"category one (legitimate): {legitimate}" in result.stdout
    assert f"category two (genuine): {genuine}" in result.stdout


def test_d3_bucketed_step_count_sum_identical_count_diverges(tmp_path):
    path, conn = _database(tmp_path, "step_count", [3.0, 7.0])
    conn.execute(
        "UPDATE daily_metrics SET count = 1, avg = 10.0, last = 10.0 "
        "WHERE metric = 'step_count'"
    )
    conn.commit()
    conn.close()

    result = _run(path)
    _assert_categories(result, legitimate=1, genuine=0, returncode=0)


def test_d3_full_resolution_heart_rate_divergence_is_genuine(tmp_path):
    path, conn = _database(tmp_path, "heart_rate", [100.0, 110.0])
    conn.execute(
        "UPDATE daily_metrics SET last = 999.0 "
        "WHERE metric = 'heart_rate'"
    )
    conn.commit()
    conn.close()

    result = _run(path)
    _assert_categories(result, legitimate=0, genuine=1, returncode=1)


def test_d3_bucketed_step_count_sum_mismatch_is_genuine(tmp_path):
    path, conn = _database(tmp_path, "step_count", [3.0, 7.0])
    conn.execute(
        "UPDATE daily_metrics SET sum = sum + 1.0 "
        "WHERE metric = 'step_count'"
    )
    conn.commit()
    conn.close()

    result = _run(path)
    _assert_categories(result, legitimate=0, genuine=1, returncode=1)


def test_d3_non_allowlisted_metric_divergence_is_legitimate(tmp_path):
    path, conn = _database(tmp_path, "active_energy", [500.0])
    conn.execute("DELETE FROM records WHERE metric = 'active_energy'")
    conn.commit()
    conn.close()

    result = _run(path)
    _assert_categories(result, legitimate=1, genuine=0, returncode=0)


def test_unfiltered_database_discrepancy_remains_genuine(tmp_path):
    path, conn = _database(tmp_path, "heart_rate", [100.0], d3=False)
    conn.execute(
        "UPDATE daily_metrics SET last = 999.0 "
        "WHERE metric = 'heart_rate'"
    )
    conn.commit()
    conn.close()

    result = _run(path)
    _assert_categories(result, legitimate=0, genuine=1, returncode=1)


def test_sums_match_spans_the_measured_gap_between_noise_and_signal():
    """A bucketed sum is preserved mathematically, not bit-for-bit.

    Both bounds are measured on vault #1 (2026-08-25), not chosen: the 13
    bucketed rows whose sums differed at all spanned 1.1e-16 to 2.0e-16
    relative, and the smallest genuine discrepancy — one lost sample from a
    6148-sample day — is 1.6e-4. `SUM_REL_TOL` has to sit between them, and
    this pins that it does.
    """
    # The real pairs, verbatim from the vault. Every one is float noise.
    for stored, rebuilt in [
            (16175.252213674705, 16175.252213674703),
            (7.4579325800000005, 7.45793258),
            (0.8650181173354062, 0.8650181173354061),
            (11.096000030999999, 11.096000031),
            (4470.311930993558, 4470.311930993559)]:
        assert V.sums_match(stored, rebuilt), (stored, rebuilt)

    # One lost sample out of 6148 must still be caught — and so must a
    # discrepancy a thousand times smaller than that.
    total = 7232.461375806046
    assert not V.sums_match(total, total * (1 - 1 / 6148))
    assert not V.sums_match(total, total * (1 - 1e-7))

    # The threshold itself, from both sides.
    assert V.sums_match(1000.0, 1000.0 * (1 + 1e-10))
    assert not V.sums_match(1000.0, 1000.0 * (1 + 1e-8))

    # A None is not equal to a number, and two Nones are not a mismatch.
    assert V.sums_match(None, None)
    assert not V.sums_match(1.0, None)
    assert not V.sums_match(None, 1.0)


# --------------------------------------------------------------------------- #
# D19 (#218) — Apple's consolidated daily totals.
#
# The shape being tested: on a `daily_metrics` row labelled
# `source_kind = 'apple_consolidated'`, `sum` is Apple's own figure and is NOT
# derivable from `records`. Handing that one column to `consolidated_diffs` has
# to fix BOTH halves of a pre-D19 asymmetry, and the second half is the
# dangerous one:
#
#   * step_count and distance_walking_running are in vault.VAULT_RAW_SERIES, so
#     before this change a consolidated row went CATEGORY TWO — loud, wrong.
#   * flights_climbed is NOT in that set, so `classify_diffs` called every
#     divergence legitimate and the row was SILENTLY SWALLOWED. A row the
#     check cannot derive must be RECOGNISED, not waved through.
#
# Several tests below therefore run the pre-D19 script as well as this one, so
# they prove the change rather than describe it.
# --------------------------------------------------------------------------- #

import contextlib  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# D19 step 1. The last commit before the correctness net learned about
# consolidated rows — the "today" that tests 13, 14 and 17 compare against.
BASELINE_SHA = "2fb6bc0"
TOTALS_UNIT = "count"


def _baseline_source() -> str:
    proc = subprocess.run(
        ["git", "show", f"{BASELINE_SHA}:scripts/verify_daily_metrics.py"],
        cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"pre-D19 script unavailable: {proc.stderr.strip()}")
    return proc.stdout


def _current_source() -> str:
    return (REPO / "scripts" / "verify_daily_metrics.py").read_text()


def _run_source(source: str, path, *extra):
    """Run one version of the script in-process and capture what it printed.

    In-process rather than as a subprocess so that the pre-D19 source can be
    run straight out of git without writing it anywhere. `__file__` is set to
    the real path because the module resolves the repo root from it.
    """
    namespace = {"__name__": "_verify_under_test",
                 "__file__": str(REPO / "scripts" / "verify_daily_metrics.py")}
    exec(compile(source, namespace["__file__"], "exec"), namespace)
    argv = ["verify_daily_metrics.py", "--db", str(path), "--derived-days", "0",
            *extra]
    buf, saved = io.StringIO(), sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(buf):
            code = namespace["main"]()
    finally:
        sys.argv = saved
    return code, buf.getvalue()


def _add_total(conn, metric: str, day: str, value: float, *,
               unit: str = TOTALS_UNIT, state: str = "provisional",
               batch: str = "b1") -> None:
    """One consolidated total, written the way the receiver will write it."""
    db.insert_daily_totals(conn, [{
        "metric": metric, "local_date": day, "value": value, "unit": unit,
        "interval": "day", "state": state, "device_id": "dev-1",
        "queried_at": f"{day}T23:00:00+00:00",
    }], batch_id=batch)


def _consolidated(tmp_path, metric: str, *, records=(3.0, 7.0), total=42.0,
                  d3: bool = True, name: str | None = None):
    """A day whose records sum to 10 and whose Apple total says 42.

    The two disagree on purpose: that disagreement is the whole point of D19 —
    the raw per-source samples double-count, Apple's consolidated series does
    not — and it is what makes the row a discrepancy before the change.
    """
    path, conn = _database(tmp_path, metric, list(records), d3=d3,
                           name=name or f"cons-{metric}")
    _add_total(conn, metric, DAY, total)
    db.apply_consolidated_totals(conn)
    conn.commit()
    return path, conn


def test_changed_provisional_to_settled_total_is_not_a_verifier_discrepancy(
        tmp_path):
    """A changed Apple total remains clean when the cache follows the pull."""
    path, conn = _database(tmp_path, "step_count", [3.0, 7.0], d3=False,
                           name="changed-settle")
    _add_total(conn, "step_count", DAY, 42.0, batch="provisional")
    db.apply_consolidated_totals(conn)
    _add_total(conn, "step_count", DAY, 84.0, state="settled",
               batch="settled")
    db.apply_consolidated_totals(conn)
    assert conn.execute(
        "SELECT sum, source_kind FROM daily_metrics WHERE metric = 'step_count' "
        "AND date = ?", (DAY,)).fetchone()[:] == (84.0, "apple_consolidated")
    conn.commit()
    conn.close()

    result = _run(path)
    _assert_categories(result, legitimate=0, genuine=0, returncode=0)
    assert "consolidated rows: 1 checked, 0 discrepancy(ies) (0 fatal)" \
        in result.stdout


def _multi_day(tmp_path, days, *, metric="step_count", total=42.0,
               skip_totals=(), name="multi"):
    """`days` of records for one metric, with a consolidated total on each day
    except those in `skip_totals` (which never had one, as opposed to having
    lost one — check 5 and check 6 are different failures)."""
    path = tmp_path / f"{name}.db"
    conn = db.connect(path)
    db.init_db(conn)
    db.insert_records(conn, [_record(metric, value, n, day)
                             for day in days
                             for n, value in enumerate((3.0, 7.0), start=1)])
    db.recompute_daily_metrics(conn, full=True)
    derive.update_for_days(conn, derive.all_source_days(conn))
    vault.declare_vault(conn)
    for day in days:
        if day not in skip_totals:
            _add_total(conn, metric, day, total)
    db.apply_consolidated_totals(conn)
    conn.commit()
    return path, conn


# --- 13: a consolidated row is not a discrepancy at all --------------------- #

@pytest.mark.parametrize("metric", ["step_count", "flights_climbed"])
def test_13_consolidated_row_is_not_a_discrepancy(tmp_path, metric):
    """13. `main()` returns 0 on a consolidated row whose sum differs from the
    records sum — and the pre-D19 script is asserted too, so this proves the
    fix. Note the two metrics fail DIFFERENTLY before the change: step_count
    goes category two, flights_climbed is swallowed as category one."""
    path, conn = _consolidated(tmp_path, metric)
    conn.close()

    before_code, before_out = _run_source(_baseline_source(), path)
    if metric == "step_count":
        assert before_code == 1, before_out
        assert "category two (genuine): 1" in before_out
    else:
        assert before_code == 0, before_out
        assert "category one (legitimate): 1" in before_out

    result = _run(path)
    _assert_categories(result, legitimate=0, genuine=0, returncode=0)
    assert "consolidated rows: 1 checked, 0 discrepancy(ies) (0 fatal)" \
        in result.stdout
    assert "INTEGRITY only" in result.stdout


# --- 14: a consolidated row that disagrees with its total IS reported ------- #

@pytest.mark.parametrize("metric", ["step_count", "flights_climbed"])
def test_14_consolidated_sum_that_left_its_total_behind_is_reported(tmp_path,
                                                                    metric):
    """14. Edit 1 is not a blanket exemption: corrupt `daily_metrics.sum` on a
    consolidated pair and `main()` returns 1 naming the pair.

    This is also the proof that `flights_climbed` is no longer silently
    swallowed. Before the change the same corruption on `flights_climbed`
    returned 0 as "category one (legitimate)" — a wrong number in the cache,
    reported as fine.
    """
    path, conn = _consolidated(tmp_path, metric)
    conn.execute("UPDATE daily_metrics SET sum = 999.0 WHERE metric = ?",
                 (metric,))
    conn.commit()
    conn.close()

    before_code, before_out = _run_source(_baseline_source(), path)
    if metric == "step_count":
        assert before_code == 1, before_out          # loud, for the wrong reason
    else:
        assert before_code == 0, before_out          # SILENTLY SWALLOWED
        assert "category one (legitimate): 1" in before_out

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "[FAIL] check 3" in result.stdout
    assert metric in result.stdout and DAY in result.stdout
    assert "999.0" in result.stdout and "42.0" in result.stdout
    assert "db.apply_consolidated_totals(conn)" in result.stdout


def test_14b_a_row_that_lost_its_label_is_reported(tmp_path):
    """14, check 2. A recompute that overwrote the override leaves the label
    behind as well as the value, so the pair reads as records-derived. That is
    the failure `apply_consolidated_totals` running after EVERY recompute
    exists to prevent, and this is what catches it if it ever stops."""
    path, conn = _consolidated(tmp_path, "step_count")
    conn.execute("UPDATE daily_metrics SET source_kind = 'records', sum = 10.0 "
                 "WHERE metric = 'step_count'")
    conn.commit()
    conn.close()

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "[FAIL] check 2" in result.stdout
    assert "still records-derived" in result.stdout


def test_14c_a_total_with_no_daily_metrics_row_is_reported(tmp_path):
    """14, check 1. A received total that never reached the cache at all."""
    path, conn = _consolidated(tmp_path, "flights_climbed")
    conn.execute("DELETE FROM daily_metrics WHERE metric = 'flights_climbed'")
    conn.commit()
    conn.close()

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "[FAIL] check 1" in result.stdout
    assert "daily_metrics has no row" in result.stdout


# --- 15: a label with nothing behind it ------------------------------------ #

def test_15_label_with_no_backing_total_is_reported(tmp_path):
    """15. Check 5 — a total deleted from `hk_daily_totals` by hand. NOT the
    rebuild landmine; 15a is that, and 15a proves this check cannot catch it."""
    path, conn = _consolidated(tmp_path, "flights_climbed")
    conn.execute("DELETE FROM hk_daily_totals")
    conn.commit()
    conn.close()

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "[FAIL] check 5" in result.stdout
    assert "no row behind it" in result.stdout
    # The repair line must NOT name apply_consolidated_totals: it is a no-op
    # here, and offering it would send an operator to re-run something that
    # cannot possibly help.
    assert "Re-running the override will NOT bring it back" in result.stdout
    assert "db.apply_consolidated_totals(conn)" not in result.stdout


def test_15_check_6_does_not_re_report_a_day_check_5_already_owns(tmp_path):
    """A day whose total was deleted from under a live label is evidence of a
    deletion, not of a phone that was off. Check 6 must stay quiet about it, or
    the operator reads two findings for one fact and the second one is wrong."""
    day = DAYS[3]
    path, conn = _multi_day(tmp_path, DAYS, name="owned")
    conn.execute("DELETE FROM hk_daily_totals WHERE local_date = ?", (day,))
    conn.commit()

    findings = V.consolidated_diffs(conn)
    conn.close()
    assert [(f["check"], f["date"]) for f in findings] == [(5, day)]

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "check 6" not in result.stdout
    assert "phone being off" not in result.stdout


# --- 15a: the rebuild shape, and why check 5 cannot see it ------------------ #

DAYS = [f"2026-08-{d:02d}" for d in range(10, 20)]


def _post_rebuild(conn):
    """The exact state a rebuild that dropped the totals leaves behind:
    `hk_daily_totals` empty, every label back to 'records', and `sum` carrying
    the double-counted records figure again."""
    conn.execute("DELETE FROM hk_daily_totals")
    db.recompute_daily_metrics(conn, full=True)
    conn.commit()


def test_15a_i_check_5_is_blind_to_the_rebuild_shape(tmp_path):
    """15a(i), and it is the point of the pair. On the post-rebuild state,
    checks 1-5 see an empty, agreeing world and `consolidated_diffs` returns
    ZERO rows. If someone later "fixes" check 5 to cover the rebuild, this
    assertion fails and tells them why it cannot: there is no label left to
    find. Check 6 is the one that does this job, and it is built the other way
    round — see 15a(ii)."""
    path, conn = _multi_day(tmp_path, DAYS, name="rebuild-i")
    _post_rebuild(conn)
    # Without the affirmative expectation there is nothing left to notice.
    conn.execute("DELETE FROM vault_meta "
                 "WHERE key = 'daily_totals_expected_from:step_count'")
    conn.commit()

    assert V.consolidated_diffs(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM daily_metrics "
                        "WHERE source_kind = 'apple_consolidated'").fetchone()[0] == 0
    conn.close()

    result = _run(path)
    assert result.returncode == 0, result.stdout
    assert "consolidated rows: 0 checked, 0 discrepancy(ies)" in result.stdout


def test_15a_ii_check_6_catches_the_rebuild_shape(tmp_path):
    """15a(ii). With `vault_meta.daily_totals_expected_from:<metric>` set — which
    `insert_daily_totals` sets on the first accepted row and which survives a
    rebuild — the same state fails, naming the contiguous range."""
    path, conn = _multi_day(tmp_path, DAYS, name="rebuild-ii")
    _post_rebuild(conn)
    assert conn.execute(
        "SELECT value FROM vault_meta "
        "WHERE key = 'daily_totals_expected_from:step_count'"
    ).fetchone()["value"] == DAYS[0]
    conn.close()

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "[FAIL] check 6" in result.stdout
    # The window stops one day short of the newest day: that day is still open
    # on the phone and its total has not been pulled yet.
    assert f"{DAYS[0]}..{DAYS[-2]}" in result.stdout
    assert "9 of 9 expected day(s)" in result.stdout
    assert "the run starts at the epoch itself" in result.stdout
    # The repair line must not send an operator to re-run the override: there is
    # nothing left for it to apply.
    assert "apply_consolidated_totals is a no-op" in result.stdout
    assert "db.apply_consolidated_totals(conn)" not in result.stdout


# --- 15b: a scattered gap is a phone that was off, not a lost table --------- #

def test_15b_check_6_does_not_fire_on_a_scattered_gap(tmp_path):
    """15b. Three non-adjacent days with no total: reported, `main()` returns 0.
    A phone that was genuinely off produces exactly this shape."""
    missing = [DAYS[2], DAYS[4], DAYS[6]]
    path, conn = _multi_day(tmp_path, DAYS, skip_totals=missing, name="gap")
    conn.close()

    result = _run(path)
    assert result.returncode == 0, result.stdout
    assert "[note] check 6" in result.stdout
    assert "[FAIL]" not in result.stdout
    assert "3 of 9 expected day(s)" in result.stdout
    assert "reported, not failed" in result.stdout
    for day in missing:
        assert day in result.stdout


# --- 16: the other three columns are still records-derived ----------------- #

@pytest.mark.parametrize("column, value", [("count", 99), ("avg", 99.0),
                                           ("last", 99.0)])
def test_16_count_avg_and_last_stay_checked_on_a_consolidated_row(tmp_path,
                                                                  column, value):
    """16. THREE columns, not five, and not seven. `sum` is handed over;
    `count`, `avg` and `last` stay records-derived and stay compared. Run on an
    undeclared database so the verdict comes from the corrupted column alone
    and not from `classify_diffs`' allowlist."""
    path, conn = _consolidated(tmp_path, "step_count", d3=False,
                               name=f"col-{column}")
    conn.commit()
    conn.close()
    assert _run(path).returncode == 0, "fixture is clean before the corruption"

    conn = db.connect(path)
    conn.execute(f"UPDATE daily_metrics SET {column} = ? "
                 f"WHERE metric = 'step_count'", (value,))
    conn.commit()
    conn.close()

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "category two (genuine): 1" in result.stdout


# --- 16a: all seven stored aggregate columns are compared ------------------- #

def test_16a_unit_is_checked_on_a_consolidated_row(tmp_path):
    """16a, first half. A consolidated row's unit remains checked by check 4."""
    path, conn = _consolidated(tmp_path, "step_count", name="unit-cons")
    conn.execute("UPDATE daily_metrics SET unit = 'furlong' "
                 "WHERE metric = 'step_count'")
    conn.commit()
    conn.close()

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "[FAIL] check 4" in result.stdout
    assert "furlong" in result.stdout


def test_16a_unit_disagreement_on_a_records_row_is_reported(tmp_path):
    """16a, second half. Unit is an ungated hard comparison on records rows."""
    path, conn = _database(tmp_path, "step_count", [3.0, 7.0], name="unit-rec")
    conn.execute("UPDATE daily_metrics SET unit = 'furlong' "
                 "WHERE metric = 'step_count'")
    conn.commit()
    conn.close()

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "unit disagreements: 1" in result.stdout
    assert "furlong" in result.stdout


def test_16a_null_unit_disagreement_on_a_records_row_is_reported(tmp_path):
    """16a, unit comparison treats NULL versus a value as disagreement."""
    path, conn = _database(tmp_path, "step_count", [3.0, 7.0], name="unit-null")
    conn.execute("UPDATE daily_metrics SET unit = NULL "
                 "WHERE metric = 'step_count'")
    conn.commit()
    conn.close()

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "unit disagreements: 1" in result.stdout


def test_16a_min_and_max_are_checked_after_the_era_start(tmp_path):
    """16a, third half. The explicit flag enables the two extrema checks."""
    path, conn = _database(tmp_path, "step_count", [3.0, 7.0], name="minmax")
    conn.execute("UPDATE daily_metrics SET min = -1.0, max = 12345.0 "
                 "WHERE metric = 'step_count'")
    conn.commit()
    conn.close()

    result = _run(path, "--minmax-from", "2026-01-01")
    assert result.returncode == 1, result.stdout
    assert "min disagreements: 1" in result.stdout
    assert "max disagreements: 1" in result.stdout


def test_16a_min_and_max_before_the_era_are_historical(tmp_path):
    """16a, fourth half. A pre-era mismatch is counted, never reported."""
    path, conn = _database(tmp_path, "step_count", [3.0, 7.0], name="historical")
    conn.execute("UPDATE daily_metrics SET min = -1.0, max = 12345.0 "
                 "WHERE metric = 'step_count'")
    conn.commit()
    conn.close()

    result = _run(path, "--minmax-from", "2026-08-21")
    assert result.returncode == 0, result.stdout
    assert "min disagreements: 0" in result.stdout
    assert "max disagreements: 0" in result.stdout
    assert "historical, min/max not compared: 1 rows" in result.stdout
    assert "-1.0" not in result.stdout


def test_16a_min_and_max_are_untouched_without_the_flag(tmp_path):
    """16a, fifth half. Omitting the flag preserves today's min/max behavior."""
    path, conn = _database(tmp_path, "step_count", [3.0, 7.0], name="no-minmax")
    conn.execute("UPDATE daily_metrics SET min = -1.0, max = 12345.0 "
                 "WHERE metric = 'step_count'")
    conn.commit()
    conn.close()

    result = _run(path)
    assert result.returncode == 0, result.stdout
    assert "min/max not compared: --minmax-from not supplied" in result.stdout
    assert "-1.0" not in result.stdout


# --- 17: a pre-D19 database is not merely handled, it is untouched --------- #

def _strip_timings(text: str) -> str:
    """Mask the two elapsed-time figures the script prints. They are the only
    non-deterministic bytes in its output."""
    return re.sub(r"\(\d+\.\d+s\)", "(TIMEs)", text)


def test_17_a_pre_d19_database_preserves_legacy_output(tmp_path):
    """17. Neither `hk_daily_totals` nor `daily_metrics.source_kind`: the output
    keeps the pre-D19 verdict and all old check output. The new comparison status
    lines are additive, so remove those before comparing the legacy bytes."""
    path, conn = _database(tmp_path, "step_count", [3.0, 7.0], name="pre-d19")
    conn.execute("DROP TABLE hk_daily_total_revisions")
    conn.execute("DROP TABLE hk_daily_totals")
    conn.execute("ALTER TABLE daily_metrics DROP COLUMN source_kind")
    conn.commit()
    assert not V.d19_storage_present(conn)
    conn.close()

    old_code, old_out = _run_source(_baseline_source(), path)
    new_code, new_out = _run_source(_current_source(), path)
    new_legacy = re.sub(
        r"^(unit disagreements:.*|min/max not compared:.*)\n", "",
        _strip_timings(new_out), flags=re.MULTILINE)
    assert (new_code, new_legacy) == (old_code, _strip_timings(old_out))
    assert "consolidated" not in new_out


def _final_select(source: str) -> str:
    """The SELECT at the end of `diffs()`, verbatim from a version of the file.

    Extracted rather than copied: a copy would drift, and the whole point is to
    compare what the file actually says today against what it said before D19.
    """
    body = source.split("def diffs(")[1].split("\ndef ")[0]
    marker = ('rows = conn.execute(\n        f"""'
              if 'rows = conn.execute(\n        f"""' in body
              else 'return conn.execute(\n        f"""')
    return body.split(marker)[1].split('"""')[0]


def test_the_d19_predicate_is_equivalent_to_the_pre_d19_one_without_the_column(
        tmp_path):
    """On a database with no `source_kind`, edit 1 must be a no-op — not
    "usually", exhaustively.

    A historical vault can have exactly this shape, so this is the property
    that makes running the script against it safe. The interesting
    cases are the NULL ones: `(d.sum IS NULL) IS NOT (r.sum IS NULL)` is the
    term the D19 edit had to wrap, and a hand-edited three-valued predicate is
    where this kind of change goes wrong. So drive every combination of
    present/absent row and NULL/equal/different column rather than a sample.

    SQLite does NOT constant-fold `'records' = 'apple_consolidated'` — the two
    statements compile to different bytecode (243 vs 253 opcodes) — so identical
    output has to be shown, not assumed from the source text.
    """
    holes = ",".join("?" * len(derive.DERIVED_METRICS))
    old = _final_select(_baseline_source()).format(holes=holes)
    new = _final_select(_current_source()).format(
        holes=holes, kind="'records'", CONSOLIDATED="apple_consolidated")
    assert old != new, "the predicates are textually identical; nothing to prove"

    conn = db.connect(tmp_path / "predicate.db")
    db.init_db(conn)
    # The pre-D19 shape: the column the edit keys on does not exist.
    conn.execute("ALTER TABLE daily_metrics DROP COLUMN source_kind")
    conn.execute("CREATE TEMP TABLE rebuilt (metric TEXT, date TEXT, count, "
                 "sum, avg, min, max, last, unit)")

    # `daily_metrics.count` is NOT NULL, so a stored count never is; the
    # rebuilt side can be, and that asymmetry is part of what is being pinned.
    values = [None, 0.0, 1.0, 2.0]
    n = 0
    for d_count, r_count in [(a, b) for a in values[1:] for b in values]:
        for d_sum, r_sum in [(a, b) for a in values for b in values]:
            day = f"2026-01-{1 + (n % 28):02d}"
            metric = f"m{n}"
            # Four presence combinations: both sides, stored only, rebuilt only,
            # and — via the loop skipping it — neither.
            side = n % 3
            if side != 2:
                conn.execute(
                    "INSERT INTO daily_metrics (metric, date, count, sum, avg, "
                    "last, unit) VALUES (?, ?, ?, ?, ?, ?, 'count')",
                    (metric, day, d_count, d_sum, d_sum, d_count))
            if side != 1:
                conn.execute(
                    "INSERT INTO rebuilt (metric, date, count, sum, avg, last, "
                    "unit) VALUES (?, ?, ?, ?, ?, ?, 'count')",
                    (metric, day, r_count, r_sum, r_sum, r_count))
            n += 1
    conn.commit()
    assert n == 192

    args = (*derive.DERIVED_METRICS, V.TOL, V.TOL)
    before = [tuple(r) for r in conn.execute(old, args)]
    after = [tuple(r) for r in conn.execute(new, args)]
    conn.close()
    assert before, "the fixture produced no discrepancies at all"
    assert after == before


def test_the_consolidated_list_is_truncated_like_the_other_two(tmp_path):
    """An override that never ran produces one finding per metric per day. The
    other two discrepancy lists honour `--limit`; this one has to as well, or a
    real failure arrives as ten thousand identical lines — which is output
    nobody acts on."""
    days = [f"2026-07-{d:02d}" for d in range(1, 26)]
    path, conn = _multi_day(tmp_path, days, name="truncate")
    # Every pair loses its label: 25 check-2 findings.
    conn.execute("UPDATE daily_metrics SET source_kind = 'records'")
    conn.commit()
    conn.close()

    result = subprocess.run(
        [sys.executable, "scripts/verify_daily_metrics.py", "--db", str(path),
         "--derived-days", "0", "--limit", "5"],
        capture_output=True, text=True)
    assert result.returncode == 1, result.stdout
    assert "25 consolidated-total discrepancy(ies) (truncated):" in result.stdout
    assert result.stdout.count("[FAIL] check 2") == 5


def test_an_unreadable_expectation_is_a_finding_not_a_traceback(tmp_path):
    """`daily_totals_expected_from:<metric>` is free-form TEXT that arrives from a
    payload. If it is unreadable, check 6 cannot run — and "check 6 is not
    running" is exactly the silence check 6 exists to break, so it has to be
    said out loud rather than raised as a stack trace out of a read-only
    checker."""
    path, conn = _multi_day(tmp_path, DAYS, name="bad-epoch")
    conn.execute("UPDATE vault_meta SET value = 'soon' "
                 "WHERE key = 'daily_totals_expected_from:step_count'")
    conn.commit()
    conn.close()

    result = _run(path)
    assert result.returncode == 1, result.stdout
    assert "Traceback" not in result.stderr, result.stderr
    assert "[FAIL] check 6" in result.stdout
    assert "which is not a date" in result.stdout
