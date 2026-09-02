"""Correlation / lag-analysis primitives over daily_metrics — metrics.py style:
pure functions given a connection; callers manage the connection. scipy
supplies p-values; every float that reaches JSON goes through mx.r.

Lag semantics EVERYWHERE: lag_days = L pairs x from day D-L with y from day D.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from scipy import stats as sps

from . import metrics as mx
from . import normalize as nz
from .metrics import WEAR_MIN_HOURS

# The pair count below which a correlation declines to answer — distinct from
# sleep_regularity.SLEEP_REGULARITY_MIN_PAIRS, a different quantity (consecutive-
# night pairs) that must not be unified with this one. See #127 (F-50).
CORRELATION_MIN_PAIRS = 8
FDR_Q = 0.10

# Var(z_s) ~= 1.06/(n-3) for a Fisher-transformed Spearman rho. Used by the
# reported interval AND by pairs_needed_for_power; named once so the two cannot
# drift apart.
SPEARMAN_VAR_INFLATION = 1.06
# metric groups measured by the watch: pairing drops low-wear days for these
WATCH_GROUPS = {"vitals", "heart", "sleep", "sleep_timing"}
# pairs that are trivially coupled regardless of catalog group
TRIVIAL_PAIRS = {
    frozenset({"step_count", "distance_walking_running"}),
    frozenset({"active_energy", "apple_exercise_time"}),
    frozenset({"active_energy", "physical_effort"}),
    # sleep containment tautologies: the derived timing metrics live in group
    # 'sleep_timing' while stage totals live in 'sleep', so the same-group check
    # misses that time_in_bed/awakenings are computed FROM those stages.
    frozenset({"sleep_time_in_bed", "sleep_asleep"}),
    frozenset({"sleep_time_in_bed", "sleep_in_bed"}),
    frozenset({"sleep_time_in_bed", "sleep_core"}),
    frozenset({"sleep_time_in_bed", "sleep_rem"}),
    frozenset({"sleep_time_in_bed", "sleep_deep"}),
    frozenset({"sleep_time_in_bed", "sleep_awake"}),
    frozenset({"sleep_awakenings", "sleep_awake"}),
    frozenset({"sleep_awake_longest", "sleep_awake"}),
    frozenset({"sleep_awakenings", "sleep_awake_longest"}),
}


def correlate(xs, ys) -> dict:
    """Pearson (+Fisher 95% CI) and Spearman for aligned arrays, with guards."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = int(len(xs))
    if n < CORRELATION_MIN_PAIRS:
        return {"status": "insufficient_data", "n_pairs": n,
                "min_required": CORRELATION_MIN_PAIRS}
    # ptp (max-min) is exactly 0.0 for identical values; np.std of repeated
    # floats can be ~1e-16 and slip an exact-equality check (found live: a
    # constant 'height' series reached scipy and returned NaN p-values).
    if float(np.ptp(xs)) == 0.0 or float(np.ptp(ys)) == 0.0:
        return {"status": "undefined_constant_series", "n_pairs": n}
    pearson_r, pearson_p = sps.pearsonr(xs, ys)
    rho, spearman_p = sps.spearmanr(xs, ys)
    z = np.arctanh(np.clip(pearson_r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    # Every surface here reports the SPEARMAN rho, so it needs a Spearman
    # interval. Fisher-transforming pearson_r and printing that beside rho was
    # not an interval for the number displayed, and it contradicted the
    # significance claim on its own line: three of twelve hypotheses printed an
    # interval excluding zero next to p > 0.05, worst on subjective_soreness,
    # an ordinal at its floor on 77% of nights where Pearson r is a
    # floor-versus-not comparison with arbitrary category spacing. That
    # construction went to Telegram on 2026-08-12. E8-1, audit part 8.
    #
    # Var(z_s) ~= 1.06/(n-3) is the same Spearman variance inflation used by
    # pairs_needed_for_power below; the two must not drift apart.
    z_s = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    se_s = np.sqrt(SPEARMAN_VAR_INFLATION / (n - 3))
    return {
        "status": "ok", "n_pairs": n,
        "pearson_r": mx.r(pearson_r, 3), "pearson_p": mx.r(pearson_p, 4),
        "pearson_ci95": [mx.r(np.tanh(z - 1.96 * se), 3),
                         mx.r(np.tanh(z + 1.96 * se), 3)],
        "spearman_rho": mx.r(rho, 3), "spearman_p": mx.r(spearman_p, 4),
        "spearman_ci95": [mx.r(np.tanh(z_s - 1.96 * se_s), 3),
                          mx.r(np.tanh(z_s + 1.96 * se_s), 3)],
    }


def bh_fdr(pvals, q: float = FDR_Q):
    """Benjamini-Hochberg step-up. Returns (q_values, passed) in input order."""
    m = len(pvals)
    if m == 0:
        return [], []
    order = np.argsort(pvals)
    qvals = np.empty(m)
    running = 1.0
    for back, idx in enumerate(reversed(order)):
        rank = m - back                       # 1-based rank of this p-value
        running = min(running, pvals[idx] * m / rank)
        qvals[idx] = running
    return [float(v) for v in qvals], [bool(v <= q) for v in qvals]


def _needs_wear(metric: str) -> bool:
    return (metric != "wear_hours"
            and nz.CATALOG.get(metric, {}).get("group") in WATCH_GROUPS)


def _resolve_start(conn, metric_x: str, metric_y: str, lag_days: int,
                   start_iso: str | None) -> str:
    """Resolve an open-ended window to the first y-day that can form a pair.

    ``parse_period('all', ...)`` deliberately returns ``None`` for the start.
    A correlation needs a concrete y-day range, and the usable beginning is
    the later of y's first day and x's first day shifted by the lag.  Doing
    this here keeps the deliberate ``None`` contract in metrics.py while
    avoiding a live failure in the correlation tools.
    """
    if start_iso is not None:
        return start_iso
    first = {}
    for metric in (metric_x, metric_y):
        row = conn.execute(
            "SELECT MIN(date) FROM daily_metrics WHERE metric = ?", (metric,)
        ).fetchone()
        if not row or row[0] is None:
            raise ValueError(
                f"cannot resolve an 'all' correlation window: no data for {metric!r}"
            )
        first[metric] = date.fromisoformat(row[0])
    lag = timedelta(days=lag_days)
    return max(first[metric_y], first[metric_x] + lag).isoformat()


def paired_series(conn, metric_x: str, metric_y: str, lag_days: int,
                  start_iso: str | None, end_iso: str):
    """Aligned daily pairs (x from day D-lag_days, y from day D) over
    [start_iso, end_iso] of y-days, inner-joined; drops days where wear_hours
    is known and < WEAR_MIN_HOURS for watch-derived metrics. Returns
    (xs, ys, meta)."""
    start_iso = _resolve_start(conn, metric_x, metric_y, lag_days, start_iso)
    lag = timedelta(days=lag_days)
    x_start = (date.fromisoformat(start_iso) - lag).isoformat()
    x_end = (date.fromisoformat(end_iso) - lag).isoformat()
    dxs, vxs, _ = mx.series(conn, metric_x, x_start, x_end)
    dys, vys, _ = mx.series(conn, metric_y, start_iso, end_iso)
    xmap = dict(zip(dxs, vxs))
    wear: dict[str, float] = {}
    if _needs_wear(metric_x) or _needs_wear(metric_y):
        wd, wv, _ = mx.series(conn, "wear_hours", x_start, end_iso)
        wear = dict(zip(wd, wv))
    xs, ys, dropped = [], [], 0
    for d, y in zip(dys, vys):
        d_x = (date.fromisoformat(d) - lag).isoformat()
        if d_x not in xmap:
            continue
        low_x = _needs_wear(metric_x) and wear.get(d_x, 24.0) < WEAR_MIN_HOURS
        low_y = _needs_wear(metric_y) and wear.get(d, 24.0) < WEAR_MIN_HOURS
        if low_x or low_y:
            dropped += 1
            continue
        xs.append(xmap[d_x])
        ys.append(y)
    window_days = (date.fromisoformat(end_iso) - date.fromisoformat(start_iso)).days + 1
    meta = {
        "n_pairs": len(xs), "window_days": window_days,
        "coverage_x_pct": mx.r(100.0 * len(dxs) / window_days, 1),
        "coverage_y_pct": mx.r(100.0 * len(dys) / window_days, 1),
        "dropped_low_wear": dropped,
    }
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), meta


def scan(conn, target: str, start_iso: str, end_iso: str, lags=(0, 1)) -> list[dict]:
    """Test every candidate metric against `target` at each lag; screening
    statistic is Spearman (robust); BH-FDR corrects all tests jointly.
    Sorted by |rho| desc. wear_hours is a coverage artifact, not a candidate."""
    rows = conn.execute(
        "SELECT DISTINCT metric FROM daily_metrics WHERE metric NOT IN (?, 'wear_hours')",
        (target,)).fetchall()
    tgroup = nz.CATALOG.get(target, {}).get("group")
    tests: list[dict] = []
    for row in rows:
        m = row["metric"]
        for lag in lags:
            xs, ys, meta = paired_series(conn, m, target, lag, start_iso, end_iso)
            res = correlate(xs, ys)
            # spearman_p None (NaN from scipy) would break the joint FDR sort —
            # belt and braces on top of the constant-series guard.
            if res["status"] != "ok" or res.get("spearman_p") is None:
                continue
            mgroup = nz.CATALOG.get(m, {}).get("group")
            related = ((tgroup is not None and mgroup == tgroup)
                       or frozenset({m, target}) in TRIVIAL_PAIRS)
            tests.append({
                "metric": m, "lag_days": lag, "related_group": related,
                "n_pairs": res["n_pairs"], "spearman_rho": res["spearman_rho"],
                "spearman_p": res["spearman_p"], "pearson_r": res["pearson_r"],
                "dropped_low_wear": meta["dropped_low_wear"],
            })
    qvals, passed = bh_fdr([t["spearman_p"] for t in tests])
    for t, qv, ok in zip(tests, qvals, passed):
        t["q_value"] = mx.r(qv, 4)
        t["passed_fdr"] = ok
    tests.sort(key=lambda t: -abs(t["spearman_rho"] or 0.0))
    return tests


def _hypothesis_window(conn, spec: dict, metric_y: str,
                       end_iso: str | None) -> tuple[str | None, str]:
    """Resolve a hypothesis window without silently choosing a date range."""
    window = spec.get("window", spec.get("period"))
    if window is None:
        raise ValueError("each hypothesis must declare a 'window'")
    if isinstance(window, (tuple, list)):
        if len(window) != 2:
            raise ValueError("a hypothesis window range must contain start and end")
        start_iso, range_end = window
        try:
            if date.fromisoformat(start_iso) > date.fromisoformat(range_end):
                raise ValueError("hypothesis window ends before it starts")
        except (TypeError, ValueError):
            raise ValueError(
                f"hypothesis window {window!r} must contain two YYYY-MM-DD dates"
            ) from None
        return start_iso, range_end
    if not isinstance(window, str):
        raise ValueError(
            f"hypothesis window {window!r} must be a period or date range"
        )
    if ":" in window:
        return mx.parse_range(window, end_iso or "9999-12-31")
    anchor = spec.get("end_iso") or end_iso or mx.anchor_end(conn, metric_y)
    if anchor is None:
        raise ValueError(
            f"cannot resolve hypothesis period {window!r}: no data for {metric_y!r}"
        )
    return mx.parse_period(window, anchor)


def _related_pair(metric_x: str, metric_y: str) -> bool:
    """Whether the pair is a same-group or explicitly known tautology."""
    xgroup = nz.CATALOG.get(metric_x, {}).get("group")
    ygroup = nz.CATALOG.get(metric_y, {}).get("group")
    return ((xgroup is not None and xgroup == ygroup)
            or frozenset({metric_x, metric_y}) in TRIVIAL_PAIRS)


def test_hypotheses(conn, specs, end_iso: str | None = None,
                    q: float = FDR_Q) -> list[dict]:
    """Test only a caller-declared set of correlation hypotheses.

    Each mapping must declare ``metric_x`` (or ``x``), ``metric_y`` (or
    ``y``), ``lag_days``, and ``window``.  A window is a period understood by
    :func:`metrics.parse_period`, such as ``'12w'`` or ``'all'``, or an
    explicit ``'YYYY-MM-DD:YYYY-MM-DD'`` range (a two-item sequence is also
    accepted).  Periods use the y metric's latest date as their anchor unless
    ``end_iso`` or the spec's ``end_iso`` is supplied.

    BH-FDR is applied to the p-values of testable hypotheses only.  Untestable
    declarations remain in input order with their correlation status rather
    than disappearing from the result.
    """
    if not 0 < q < 1:
        raise ValueError(f"q must be between 0 and 1, got {q!r}")
    rows: list[dict] = []
    test_indices: list[int] = []
    pvals: list[float] = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("each hypothesis must be a mapping")
        metric_x = spec.get("metric_x", spec.get("x"))
        metric_y = spec.get("metric_y", spec.get("y"))
        if not metric_x or not metric_y:
            raise ValueError("each hypothesis must declare x and y metrics")
        if "lag_days" not in spec:
            raise ValueError("each hypothesis must declare lag_days")
        try:
            lag_days = int(spec["lag_days"])
        except (TypeError, ValueError):
            raise ValueError(f"invalid hypothesis lag_days: {spec['lag_days']!r}") from None
        start_iso, range_end = _hypothesis_window(conn, spec, metric_y, end_iso)
        start_iso = _resolve_start(conn, metric_x, metric_y, lag_days, start_iso)
        xs, ys, meta = paired_series(
            conn, metric_x, metric_y, lag_days, start_iso, range_end
        )
        res = correlate(xs, ys)
        row = {
            "metric_x": metric_x, "metric_y": metric_y, "lag_days": lag_days,
            "window": [start_iso, range_end],
            "related_group": _related_pair(metric_x, metric_y),
            "status": res["status"], "n_pairs": res["n_pairs"],
            "rho": res.get("spearman_rho"), "p": res.get("spearman_p"),
            "q": None, "q_value": None, "passed_fdr": False,
            "pearson_r": res.get("pearson_r"),
            "pearson_p": res.get("pearson_p"),
            "pearson_ci95": res.get("pearson_ci95"),
            "spearman_rho": res.get("spearman_rho"),
            "spearman_p": res.get("spearman_p"),
            "dropped_low_wear": meta["dropped_low_wear"],
        }
        rows.append(row)
        if res["status"] == "ok" and res.get("spearman_p") is not None:
            test_indices.append(len(rows) - 1)
            pvals.append(res["spearman_p"])

    qvals, passed = bh_fdr(pvals, q=q)
    for index, qv, ok in zip(test_indices, qvals, passed):
        rows[index]["q"] = mx.r(qv, 4)
        rows[index]["q_value"] = mx.r(qv, 4)
        rows[index]["passed_fdr"] = ok
    return rows


def min_detectable_rho(n_pairs: int, n_tests: int, q: float = FDR_Q) -> float:
    """Return the smallest absolute Pearson rho detectable by the best test.

    The best-ranked hypothesis must meet the two-sided t-test threshold
    ``p <= q / n_tests``.  This is a sensitivity boundary, not a guarantee of
    power: it inverts the correlation test's nominal p-value threshold.

    NOTE (2026-08-16, audit part 8): ``q / n_tests`` is Benjamini-Hochberg's
    RANK-1 threshold — the most stringent point of the step-up.  A hypothesis
    rejected at rank k faces ``k*q/n_tests`` instead, so this is the worst case
    rather than the operative one.  At n=30, m=6, q=0.10 the rank-1 boundary is
    0.434; at rank 6 it is 0.306.  Renderers say "could only have detected
    0.43", which is true of the best-ranked test and pessimistic for the rest.
    """
    if n_pairs <= 2:
        raise ValueError("n_pairs must be greater than 2")
    if n_tests < 1:
        raise ValueError("n_tests must be at least 1")
    if not 0 < q < 1:
        raise ValueError(f"q must be between 0 and 1, got {q!r}")
    df = n_pairs - 2
    alpha = q / n_tests
    t_critical = sps.t.ppf(1.0 - alpha / 2.0, df)
    return float(t_critical / np.sqrt(t_critical ** 2 + df))


def pairs_needed_for_power(target_rho: float, n_tests: int, q: float = FDR_Q,
                           power: float = 0.80) -> int:
    """Pairs needed to detect `target_rho` with `power`, not merely to clear the
    threshold if it appears exactly.

    `min_detectable_rho` is a SENSITIVITY boundary — the rho at which an
    observed correlation would just pass. Inverting it answers "how big must
    the effect look", which is about 50% power, and using it as an
    answerable-by date understates the wait by roughly 1.8x (47 pairs vs 87
    here at rho=0.35, m=6).

    Fisher-z sample size with the 1.06 Spearman variance inflation
    (Var(z_s) ~= 1.06/(n-3)). Calibrated against simulation: at rho=0.35,
    alpha=0.025 this returns 79 where 4,000-replication simulation of the
    Spearman test gives 83 — agreement to ~5%.

    CORRECTED 2026-08-16 (audit part 8): that sentence used to end "erring
    conservative", which is backwards. 79 < 83 means the formula asks for FOUR
    FEWER pairs than the simulation says are needed — it is 5% OPTIMISTIC
    against its own calibration. It is conservative only against the uninflated
    Fisher figure (75), which is not what the comparison was measuring. The
    1.06 belongs outside the squared term, where it is; folding it into the
    numerator would give 92 and 83 instead of 87 and 79.

    A SECOND, LARGER CAVEAT, also 2026-08-16: this returns a NOMINAL pair
    count. It is not calendar days, and it is not effective sample size. Daily
    wearable series are autocorrelated — measured on this athlete, phi ranges
    from -0.17 (alcohol) to +0.62 (soreness) — so the usable days required for
    an effective n of 87 run from 71 to 136 depending on the pair. See the
    window ceiling in deepdive_levers' module docstring: with a 90-day rolling
    window most of these targets cannot be reached at all.

    CONSERVATIVE ALPHA, deliberately. `alpha = q / n_tests` is Benjamini-
    Hochberg's rank-1 threshold — the most stringent point of the step-up, and
    Bonferroni-like for a single test. It is NOT an exact BH-FDR power
    calculation; that would require simulating all hypotheses jointly through
    `bh_fdr`. It is used because `min_detectable_rho` already assumes the same
    alpha, so the two figures stay commensurable, and because a date that
    arrives early is the failure mode this function exists to remove.
    """
    if not 0 < target_rho < 1:
        raise ValueError(f"target_rho must be in (0, 1), got {target_rho!r}")
    if n_tests < 1:
        raise ValueError("n_tests must be at least 1")
    if not 0 < q < 1:
        raise ValueError(f"q must be between 0 and 1, got {q!r}")
    if not 0 < power < 1:
        raise ValueError(f"power must be between 0 and 1, got {power!r}")
    alpha = q / n_tests
    z_alpha = sps.norm.ppf(1.0 - alpha / 2.0)
    z_power = sps.norm.ppf(power)
    return int(np.ceil(SPEARMAN_VAR_INFLATION * ((z_alpha + z_power) /
                               np.arctanh(target_rho)) ** 2 + 3))
