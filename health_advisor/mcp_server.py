"""Health MCP server — curated, parameterized, read-only-by-default tools over
stdio so the agent can spawn it. The agent reaches the DB ONLY through these tools
(no raw SQL). Every tool computes server-side on a bounded copy and returns a
COMPACT summary — never a huge raw result set.

Computation model: each tool runs a bounded query, computes with SQL/numpy, and
returns small structured JSON (stats, trends, <=~400 points). The model reasons
over the summary; the firehose never enters its context.

Run:  python -m health_advisor.mcp_server --vault PATH   (stdio)

There is no module-global server and no module-global database path. Each tool
below is a plain function whose first parameter is the session's VaultContext;
`build_server(ctx)` binds them to one session and returns a FastMCP for it. One
process can therefore serve two users at once without either of them being able
to reach the other's vault, which a module global made impossible (T-003).
"""
from __future__ import annotations

import functools
import inspect
import sqlite3
from datetime import date, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

import numpy as np
from mcp.server.fastmcp import FastMCP

from . import analysis as A
from . import correlate as C
from . import benchmark
from . import db
from . import food as fd
from . import hr_load as HL
from . import metrics as mx
from . import normalize as nz
from . import running_form as RF
from . import sleep_regularity as SR
from . import subjective as subj
from . import vault as V
from .context import RAW_SAMPLES, VaultContext

# Every tool is registered here as an unbound function taking the session's
# vault as its first argument. Nothing is bound to a server or a path at import
# time — that binding is what `build_server` does, once per session.
_TOOLS: list = []


def tool(fn):
    """Mark a function as one of this server's tools. First parameter is `ctx`."""
    _TOOLS.append(fn)
    return fn


def _bind(fn, ctx: VaultContext):
    """The tool as the model sees it: `ctx` bound away and out of the schema.

    FastMCP derives each tool's JSON schema from the signature, so the bound
    function must not advertise `ctx` — naming a vault is not the model's
    business. Setting `__signature__` also stops `inspect.signature` from
    following `functools.wraps`'s `__wrapped__` back to the unbound function,
    which would put the parameter straight back into the schema. `__wrapped__`
    itself is kept, so `inspect.unwrap(bound)` still names the real tool.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def bound(*args, **kwargs):
        return fn(ctx, *args, **kwargs)

    bound.__signature__ = sig.replace(parameters=list(sig.parameters.values())[1:])
    return bound


def build_tools(ctx: VaultContext) -> dict:
    """`{name: callable}` for one session. Calling one reads that vault, only."""
    return {fn.__name__: _bind(fn, ctx) for fn in _TOOLS}


def build_server(ctx: VaultContext, *, name: str = "health-advisor",
                 include: "frozenset[str] | tuple[str, ...] | None" = None) -> FastMCP:
    """A FastMCP bound to one session. `include` narrows the surface — the
    researcher gets a strict subset (llm.RESEARCHER_TOOLS), and D5 wants that
    asymmetry expressed here rather than enforced by convention elsewhere."""
    server = FastMCP(name)
    for tool_name, bound in build_tools(ctx).items():
        if include is not None and tool_name not in include:
            continue
        server.tool()(bound)
    return server

MAX_WORKOUTS = 200

# The scopes build_briefing actually distinguishes. Taken from analysis.py's own
# table so a new scope there cannot be rejected here.
BRIEFING_SCOPES = tuple(A.MOVER_TOPK)

# --------------------------------------------------------------------------- #
# helpers — delegates to metrics.py primitives
# --------------------------------------------------------------------------- #
_r = mx.r


def _literature_figure(name: str) -> dict:
    """Return a cited literature figure without sharing its citation mapping."""
    figure = A.LITERATURE_FIGURES[name]
    return {**figure, "citation": dict(figure["citation"])}


_agg = mx.agg
_value_col = mx.value_col
_anchor_end = mx.anchor_end
_parse_period = mx.parse_period
_parse_range = mx.parse_range
_series = mx.series
_stats = mx.stats
_metric_exists = mx.metric_exists
MAX_SERIES_POINTS = mx.MAX_SERIES_POINTS


# One source of truth, deliberately not a copy (#215). `metrics` owns which
# fields are in the metric's own unit, which are signed, and which are not
# unit-preserving at all — and `format_presentation` gates on those groups. A
# second list here would have to be kept in step by hand, which is the drift
# shape #53 closed for the numeric tokenizer.
#
# It had already drifted before this line was written: `latest_sd_hours` was
# added to the grouped table in `metrics` and not to the copy here, on the very
# first change after the copy existed. Nothing broke, because that field is
# published directly rather than through the stat loop below — which is exactly
# how this class of defect stays invisible until it isn't.
_PRESENTATION_FIELDS = mx._PRESENTATION_FIELDS


def _add_presentation(node: dict, metric: str, period, value, *, field: str) -> None:
    """Publish the Python-owned rendering as a first-class claim leaf."""
    leaf = mx.presentation_leaf(metric, period, value, field=field)
    if leaf is not None:
        node["presentation"] = leaf


def _add_stat_presentations(node: dict, metric: str, period) -> None:
    """Add leaves for the metric-owned statistics in one result object."""
    leaves = {}
    for field in _PRESENTATION_FIELDS:
        value = node.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            leaf = mx.presentation_leaf(metric, period, value, field=field)
            if leaf is not None:
                leaves[field] = leaf
    if leaves:
        node["presentations"] = leaves




def _n_segments_by_workout(conn, keys: list[str]) -> dict[str, int]:
    """Splits per workout, counted the same way get_workout_segments reports
    them — one partition, not every stored event. A raw COUNT(*) here said 9
    where the detail tool showed 5, which is the kind of contradiction the
    agent resolves by picking one at random."""
    if not keys:
        return {}
    rows = conn.execute(
        f"SELECT workout_key, start_utc, end_utc, duration_min FROM workout_events "
        f"WHERE event_type IN ('segment', 'lap') "
        f"AND workout_key IN ({','.join('?' * len(keys))})", keys).fetchall()
    by_workout: dict[str, list] = {}
    for r in rows:
        by_workout.setdefault(r["workout_key"], []).append(dict(r))
    return {k: len(mx.segment_chains(v)[0]) for k, v in by_workout.items()}


@tool
def mark_workout_not_a_session(ctx: VaultContext, workout_key: str,
                               reason: str = "", source: str = "user",
                               marked_at: str | None = None) -> dict:
    """Mark one stored workout as not a real session.

    This is explicit user correction only: it never deletes or edits the
    device-recorded workout. The stable ``workout_key`` comes from
    ``list_workouts``; ``source`` and ``marked_at`` make the correction
    attributable and durable. Repeating the call is idempotent.
    """
    conn = ctx.connect()
    try:
        db.init_db(conn)
        mark = db.mark_workout_not_a_session(
            conn, workout_key, source=source, reason=reason or None,
            marked_at=marked_at,
        )
        conn.commit()
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()
    return {"ok": True, **mark}


def _bad_dates(**named: str | None) -> str | None:
    """Validate ISO date parameters, returning an error message or None.

    Every date parameter on every tool goes through this. Relative words
    ('yesterday', 'last week') and impossible dates used to fall through to a
    query that matched nothing and returned "no data" — which the agent
    narrates as "nothing recorded that day". A tool must not confuse "I didn't
    understand you" with "it didn't happen". Pairs named start/end are also
    checked for order.
    """
    for name, value in named.items():
        if value is None:
            continue
        try:
            date.fromisoformat(value)
        except (ValueError, TypeError):
            return (f"{name}={value!r} is not a YYYY-MM-DD date. Dates must be "
                    "explicit — relative words like 'yesterday' are not resolved here.")
    start, end = named.get("start"), named.get("end")
    if start and end and start > end:
        return f"start={start!r} is after end={end!r}"
    return None


def _as_timezone(local_timezone: str | tzinfo | None) -> tzinfo | None:
    """Resolve a declared IANA zone once at a tool boundary."""
    if local_timezone is None or isinstance(local_timezone, tzinfo):
        return local_timezone
    return ZoneInfo(local_timezone)


def _local_hhmm(utc_iso: str | None,
                local_timezone: str | tzinfo | None = None,
                seconds: bool = False) -> str | None:
    """UTC ISO timestamp -> local wall clock 'HH:MM'.

    An undeclared vault passes None, preserving the historical host-timezone
    behavior. A declared IANA zone or tzinfo makes the rendering per-vault.
    seconds=True gives 'HH:MM:SS' (segment boundaries need the precision).
    """
    if not utc_iso:
        return None
    try:
        dt = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    local_tz = _as_timezone(local_timezone)
    rendered = dt.astimezone() if local_tz is None else dt.astimezone(local_tz)
    return rendered.strftime("%H:%M:%S" if seconds else "%H:%M")


# --------------------------------------------------------------------------- #
# read tools
# --------------------------------------------------------------------------- #
@tool
def list_available_metrics(ctx: VaultContext) -> dict:
    """List every metric that has data, with its unit, group, aggregation type
    (sum/mean/last), date span and number of days. Call this first to discover
    what can be queried."""
    conn = ctx.read_only()
    try:
        rows = conn.execute(
            "SELECT metric, MIN(date) f, MAX(date) l, COUNT(*) n, MAX(unit) u "
            "FROM daily_metrics GROUP BY metric ORDER BY metric"
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        cat = nz.CATALOG.get(r["metric"], {})
        out.append({
            "metric": r["metric"],
            "group": cat.get("group", "other"),
            "unit": cat.get("unit", r["u"]),
            "agg": cat.get("agg", "mean"),
            "first_date": r["f"], "last_date": r["l"], "n_days": r["n"],
        })
    return {"count": len(out), "metrics": out}


@tool
def get_daily_series(ctx: VaultContext, metric: str, start: str | None = None, end: str | None = None) -> dict:
    """Daily values for a metric between start and end (YYYY-MM-DD; default = last
    90 days of available data). Automatically uses the right daily aggregate
    (sum for cumulative metrics like steps/energy, average for rates like heart
    rate). A range longer than 400 days is downsampled to Monday-anchored weekly
    buckets reduced the SAME way — a week of steps is that week's total, not its
    daily average. When 'downsampled' is true each point carries 'days'; a point
    with days < 7 is a partial week and must not be compared against a full one.
    Dates must be explicit YYYY-MM-DD — an 'error' comes back otherwise."""
    if err := _bad_dates(start=start, end=end):
        return {"error": err}
    conn = ctx.read_only()
    try:
        if not _metric_exists(conn, metric):
            return {"error": f"unknown metric {metric!r}. Call list_available_metrics."}
        anchor = end or _anchor_end(conn, metric)
        if start is None:
            start = (date.fromisoformat(anchor) - timedelta(days=89)).isoformat()
        dates, vals, unit = _series(conn, metric, start, anchor)
    finally:
        conn.close()
    downsampled = False
    points = [{"date": d, "value": _r(v)} for d, v in zip(dates, vals)]
    for point in points:
        _add_presentation(point, metric, point["date"], point["value"],
                          field="value")
    out = {"metric": metric, "unit": unit, "agg": _agg(metric),
           "start": start, "end": anchor}
    if len(points) > MAX_SERIES_POINTS:
        # Weekly resample, reduced the same way the metric's dailies are: a
        # weekly MEAN of a cumulative metric is not that week's total.
        downsampled = True
        points = mx.downsample_weekly(dates, vals, _agg(metric))
        for point in points:
            _add_presentation(point, metric, point["date"], point["value"],
                              field="value")
        out["bucket"] = "week"
        out["downsample_agg"] = _agg(metric)
        out["note"] = (
            f"weekly buckets (Monday-anchored), each the {_agg(metric)} of its "
            "days; 'days' < 7 marks a partial week that is not comparable to a "
            "full one")
    return {**out, "downsampled": downsampled, "n": len(points), "points": points}


@tool
def summarize_metric(ctx: VaultContext, metric: str, period: str = "30d") -> dict:
    """Summarize a metric over a period ('30d','12w','6m','1y','all'). Returns
    mean/median/min/max, recent-vs-baseline change, and a linear trend per week.
    Anchored to the metric's most recent data. A period the server can't parse
    comes back as an 'error' — it is never silently replaced with a default."""
    conn = ctx.read_only()
    try:
        if not _metric_exists(conn, metric):
            return {"error": f"unknown metric {metric!r}. Call list_available_metrics."}
        anchor = _anchor_end(conn, metric)
        try:
            # parse_range, NOT parse_period: parse_period rejects explicit
            # ranges, yet its own error message tells the caller to "use ... an
            # explicit 'YYYY-MM-DD:YYYY-MM-DD' range". A researcher following
            # that advice got the identical error back and had no way forward.
            start_iso, end_iso = mx.parse_range(period, anchor)
        except ValueError as e:
            return {"error": str(e)}
        dates, vals, unit = _series(conn, metric, start_iso, end_iso)
    finally:
        conn.close()
    if not vals:
        return {"metric": metric, "period": period, "n_days": 0,
                "note": "no data in this period"}
    out = {"metric": metric, "unit": unit, "agg": _agg(metric), "period": period}
    out.update(_stats(dates, vals))
    # recent vs baseline
    n = len(vals)
    rn = max(1, min(7, n // 3))
    recent, baseline = vals[-rn:], (vals[:-rn] or vals)
    rmean, bmean = float(np.mean(recent)), float(np.mean(baseline))
    out["recent_window_days"] = rn
    out["recent_avg"] = _r(rmean)
    out["baseline_avg"] = _r(bmean)
    out["delta_vs_baseline"] = _r(rmean - bmean)
    out["delta_pct"] = _r((rmean - bmean) / bmean * 100) if bmean else None
    # linear trend per week
    out["trend_per_week"] = _r(mx.slope_per_week(dates, vals))
    _add_stat_presentations(out, metric, period)
    return out


@tool
def compare_periods(ctx: VaultContext, metric: str, period_a: str, period_b: str) -> dict:
    """Compare a metric across two windows. Each window is either an explicit
    'YYYY-MM-DD:YYYY-MM-DD' range or a period like '30d' (anchored to the
    metric's latest date). Returns stats for each window and the delta.

    An unparseable window is an 'error', never a default. If either window has
    no data the deltas come back null with a 'note' — an empty window is not a
    change of the other window's whole size."""
    conn = ctx.read_only()
    try:
        if not _metric_exists(conn, metric):
            return {"error": f"unknown metric {metric!r}. Call list_available_metrics."}
        anchor = _anchor_end(conn, metric)
        try:
            a0, a1 = _parse_range(period_a, anchor)
            b0, b1 = _parse_range(period_b, anchor)
        except ValueError as e:
            return {"error": str(e)}
        da, va, unit = _series(conn, metric, a0, a1)
        db_, vb, _ = _series(conn, metric, b0, b1)
    finally:
        conn.close()
    sa, sb = _stats(da, va), _stats(db_, vb)
    out = {"metric": metric, "unit": unit,
           "period_a": {"spec": period_a, "range": [a0, a1], **sa},
           "period_b": {"spec": period_b, "range": [b0, b1], **sb}}
    _add_stat_presentations(out["period_a"], metric, f"{a0}:{a1}")
    _add_stat_presentations(out["period_b"], metric, f"{b0}:{b1}")
    if not va or not vb:
        empty = period_a if not va else period_b
        return {**out, "mean_delta": None, "mean_delta_pct": None,
                "note": f"no data in {empty!r}; the two windows cannot be compared"}
    out["mean_delta"] = _r(sa["mean"] - sb["mean"])
    out["mean_delta_pct"] = _r((sa["mean"] - sb["mean"]) / sb["mean"] * 100) \
        if sb["mean"] else None
    return out


@tool
def get_intraday(ctx: VaultContext, metric: str, day: str, bucket_hours: int = 1) -> dict:
    """Intraday pattern for a metric on a single LOCAL day (YYYY-MM-DD), bucketed
    by local hour. Uses sum for cumulative metrics, average for rates. Returns up
    to 24 compact buckets — for time-of-day analysis. 'day' must be an explicit
    YYYY-MM-DD; a relative word comes back as an 'error', not an empty day.

    bucket_hours widens the bucket and must divide 24 (1, 2, 3, 4, 6, 8, 12, 24);
    anything else is an 'error'. 'hour' is the bucket's first local hour. Note
    that a wider bucket dilutes an event: an 08:00 hour can average 140.6 bpm
    over a run that spent 14.7 minutes above 150. For anything about
    an HR cap use get_hr_zones, which scopes to the workout itself."""
    if err := _bad_dates(day=day):
        return {"error": err}
    try:
        bucket_hours = int(bucket_hours)
    except (TypeError, ValueError):
        return {"error": f"bucket_hours={bucket_hours!r} must be a whole number of hours"}
    if bucket_hours < 1 or 24 % bucket_hours:
        return {"error": f"bucket_hours={bucket_hours} must divide 24 evenly "
                         "(1, 2, 3, 4, 6, 8, 12, 24)"}
    # D3: a vault carries sample-level rows only for the series that need them.
    # For anything else the honest answer is "not in this vault", not an empty
    # day — an empty day reads as "you did not move", which is a wrong fact
    # rather than a missing one.
    if not V.raw_series_available(metric):
        return V.raw_unavailable(metric, needed_for="intraday buckets")
    conn = ctx.read_only()
    try:
        if not _metric_exists(conn, metric):
            return {"error": f"unknown metric {metric!r}. Call list_available_metrics."}
        sqlagg = "SUM(value)" if _agg(metric) == "sum" else "AVG(value)"
        # Arbitrate exactly as the daily read path does. `daily_metrics` is
        # built through db._arbitration(), which drops a mirror source once the
        # Apple devices took over and drops whole-day estimates wearing a
        # sample's clothes. Reading raw `records` here skipped all of that, so
        # this tool and get_daily_series disagreed on 3,114 (metric, day) pairs
        # across the live DB — worst case 430x on step_count, and basal_energy
        # still affected as recently as 2026-07-21. Two tools answering the
        # same question differently is the failure the agent cannot detect.
        arb, arb_params = db._arbitration(
            conn, metric, day,
            arbitration_window=(day, day),
            arbitration_window_kind="local_date")
        # Bucket in SQL rather than re-reducing hourly rows in python: averaging
        # per-hour averages would weight a 3-sample hour like a 900-sample one.
        rows = conn.execute(
            f"SELECT (CAST(strftime('%H', start_local) AS INT) / ?) * ? h, "
            f"{sqlagg} v, COUNT(*) n "
            f"FROM records WHERE metric = ? AND local_date = ? AND start_local IS NOT NULL"
            f"{arb} "
            f"GROUP BY h ORDER BY h",
            (bucket_hours, bucket_hours, metric, day, *arb_params)).fetchall()
        unit = conn.execute(
            f"SELECT MAX(unit) FROM records WHERE metric = ? AND local_date = ?{arb}",
            (metric, day, *arb_params)).fetchone()[0]
    finally:
        conn.close()
    if not rows:
        return {"metric": metric, "day": day, "bucket_hours": bucket_hours,
                "note": "no data for this day", "buckets": []}
    buckets = [{"hour": r["h"], "value": _r(r["v"]), "n": r["n"]} for r in rows]
    for bucket in buckets:
        _add_presentation(bucket, metric, day, bucket["value"], field="value")
    return {"metric": metric, "day": day, "unit": unit or nz.canonical_unit(metric, None),
            "agg": _agg(metric), "bucket_hours": bucket_hours, "buckets": buckets}


HR_ZONE_DEFAULT_THRESHOLDS = "135,150,155,170"


def _local_window_utc(day: str, hhmm: str, label: str,
                      local_timezone: str | tzinfo | None = None) -> str:
    """'HH:MM' on a local day -> the UTC ISO string records are stored with."""
    try:
        h, m = (int(p) for p in hhmm.strip().split(":", 1))
        naive = datetime.fromisoformat(day).replace(hour=h, minute=m)
    except (TypeError, ValueError):
        raise ValueError(f"{label}={hhmm!r} must be a local 'HH:MM' time") from None
    local_tz = _as_timezone(local_timezone)
    aware = (naive.astimezone() if local_tz is None
             else naive.replace(tzinfo=local_tz))
    return aware.astimezone(timezone.utc).isoformat()


def _utc_bound(ts: str) -> str:
    """A timestamp in the exact form `records.start_utc` is stored in.

    Window bounds are compared as SQL strings, and '...12:00:00Z' sorts AFTER
    the equivalent '...12:00:00+00:00', so an unnormalized bound silently drops
    samples at the edge instead of failing. Live rows are uniformly '+00:00',
    which is precisely why a stray Z would go unnoticed."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")) \
            .astimezone(timezone.utc).isoformat()
    except (AttributeError, TypeError, ValueError):
        return ts


def _hr_samples(conn, *, day: str | None = None,
                start_utc: str | None = None, end_utc: str | None = None):
    """Raw heart_rate samples as [(datetime, bpm)] — by local day, or by an
    explicit UTC half-open [start, end) window."""
    if start_utc is not None:
        rows = conn.execute(
            "SELECT value, start_utc FROM records WHERE metric = 'heart_rate' "
            "AND start_utc >= ? AND start_utc < ? ORDER BY start_utc",
            (_utc_bound(start_utc), _utc_bound(end_utc))).fetchall()
    else:
        rows = conn.execute(
            "SELECT value, start_utc FROM records WHERE metric = 'heart_rate' "
            "AND local_date = ? ORDER BY start_utc", (day,)).fetchall()
    out = []
    for r in rows:
        try:
            out.append((datetime.fromisoformat(r["start_utc"].replace("Z", "+00:00")),
                        float(r["value"])))
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def _hr_zone_window(conn, source: str, label: str | None, start_utc: str,
                    end_utc: str, thresholds, *, day: str | None = None,
                    local_timezone: tzinfo | None = None) -> dict:
    samples = (_hr_samples(conn, day=day) if source == "day"
               else _hr_samples(conn, start_utc=start_utc, end_utc=end_utc))
    end_dt = datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
    z = mx.zone_minutes(samples, end_dt, thresholds)
    span = _elapsed_min(start_utc, end_utc)
    out = {"source": source, "label": label,
           "start_time_local": _local_hhmm(start_utc, local_timezone, seconds=True),
           "end_time_local": _local_hhmm(end_utc, local_timezone, seconds=True),
           "window_min": span, **z}
    if span and z["covered_min"] < 0.9 * span:
        out["note"] = (
            f"heart-rate samples cover only {z['covered_min']} of the window's "
            f"{span} min — the band minutes are a floor, not the whole window.")
    return out


@tool
def get_hr_zones(ctx: VaultContext, day: str, workout_type: str | None = None,
                 start_time: str | None = None, end_time: str | None = None,
                 thresholds: str = HR_ZONE_DEFAULT_THRESHOLDS) -> dict:
    """Time spent in each heart-rate band on a LOCAL day (YYYY-MM-DD) — the tool
    that answers "how long was I above 150?".

    Use this for any question about an HR CAP or ceiling. list_workouts reports
    only the session average and peak, and get_intraday's hourly buckets mix the
    run with the minutes either side of it: both can report "in band"
    (avg 144, hour-08 mean 140.6) while 41% of the run was over 150.

    'thresholds' is a comma-separated ascending list of bpm edges (default
    '135,150,155,170'); it is never silently replaced with a default. Bands are
    upper-INCLUSIVE — 'at or under 150' is exactly the '<=150' band plus
    everything below it, and a sample of exactly 150 is compliant. 'above' gives
    the direct answer per threshold: minutes and percentage of samples strictly
    above it.

    Scope: pass workout_type for one session's zones (canonical name from
    list_workouts), or start_time/end_time as local 'HH:MM' for an explicit
    window. With neither, 'windows' holds the whole day first and then each
    workout on it. Minutes are duration-weighted per sample and clipped at
    60 s, so a gap in the samples shows up as 'uncovered_min' instead of being
    charged to the last band seen — check it before quoting band minutes as the
    whole window."""
    if err := _bad_dates(day=day):
        return {"error": err}
    try:
        edges = mx.parse_thresholds(thresholds)
    except ValueError as e:
        return {"error": str(e)}
    if (start_time is None) != (end_time is None):
        return {"error": "start_time and end_time must be given together"}
    if start_time and workout_type:
        return {"error": "pass either workout_type or start_time/end_time, not both"}

    local_timezone = _as_timezone(ctx.settings()["local_timezone"])
    conn = ctx.read_only()
    try:
        if start_time:
            try:
                s_utc = _local_window_utc(day, start_time, "start_time",
                                          local_timezone)
                e_utc = _local_window_utc(day, end_time, "end_time",
                                          local_timezone)
            except ValueError as e:
                return {"error": str(e)}
            if s_utc >= e_utc:
                return {"error": f"start_time={start_time!r} is not before "
                                 f"end_time={end_time!r}"}
            windows = [_hr_zone_window(
                conn, "explicit", f"{start_time}-{end_time}", s_utc, e_utc,
                edges, local_timezone=local_timezone)]
            return {"day": day, "thresholds": [mx.r(e, 0) for e in edges],
                    "count": len(windows), "windows": windows}

        marked = db.workout_mark_condition(conn, "w", marked=True)
        sql = ("SELECT w.workout_type, w.start_utc, w.end_utc FROM workouts AS w "
               "WHERE w.local_date = ? AND "
               f"{db.workout_mark_condition(conn, 'w')}")
        args: list = [day]
        if workout_type:
            sql += " AND w.workout_type = ?"
            args.append(workout_type)
        wrows = conn.execute(sql + " ORDER BY start_utc", args).fetchall()
        excluded_args: list = [day]
        excluded_sql = ("SELECT COUNT(*) FROM workouts AS w WHERE w.local_date = ? "
                        f"AND {marked}")
        if workout_type:
            excluded_sql += " AND w.workout_type = ?"
            excluded_args.append(workout_type)
        excluded = conn.execute(excluded_sql, excluded_args).fetchone()[0]

        windows = []
        if workout_type is None:
            row = conn.execute(
                "SELECT MIN(start_utc) a, MAX(start_utc) b FROM records "
                "WHERE metric = 'heart_rate' AND local_date = ?", (day,)).fetchone()
            if row and row["a"]:
                windows.append(_hr_zone_window(
                    conn, "day", day, row["a"], row["b"], edges, day=day,
                    local_timezone=local_timezone))
        for w in wrows:
            if not w["end_utc"]:
                continue
            windows.append(_hr_zone_window(
                conn, "workout", w["workout_type"], w["start_utc"],
                w["end_utc"], edges, local_timezone=local_timezone))
    finally:
        conn.close()

    out = {"day": day, "thresholds": [mx.r(e, 0) for e in edges],
           "count": len(windows), "excluded_count": excluded,
           "windows": windows}
    if not windows:
        out["note"] = (
            f"no workout of type {workout_type!r} on {day}" if workout_type else
            f"no heart-rate samples on {day}")
    return out


@tool
def list_workouts(ctx: VaultContext, start: str | None = None, end: str | None = None,
                  limit: int = 50) -> dict:
    """List workouts between start and end (YYYY-MM-DD; default = last 90 days),
    most recent first. Returns type, duration (min), energy (kcal), distance
    (mi), average/max heart rate (bpm), local start/end times (HH:MM), whether
    a GPS route exists, workout_key, and n_segments — how many splits get_workout_segments
    reports for that session (0 = none stored; it counts one partition, not
    every stored event, so the two tools always agree).

    'workout_key' is this session's stable server-side identity. A workout is
    not part of any metric series, so a claim citing a number from one of these
    rows must OMIT 'metric' — the row's workout_key is what identifies it.
    Naming a metric for a workout field is refused, and a per-session value
    must never be restated as a daily-metric series.

    'total_in_range' is how many unmarked workouts the range actually holds;
    'excluded_count' reports marked rows in the same range so the exclusion is
    visible.
    'truncated' says whether 'limit' cut the list short. Check truncated before
    counting sessions or totalling minutes: the returned rows are the most
    recent ones, not the range. 'workout_counts' is the full-range count by
    workout type. Distance and energy use the vault's declared display units:
    miles/kcal for imperial and kilometres/kJ for metric. `list_workouts` publishes
    full-range per-type counts as
    `workout_counts: [{type, count}]`; cite the `count` leaf at its exact path
    with `metric` omitted, and never count the possibly truncated `workouts`
    rows. Dates must be explicit YYYY-MM-DD — an 'error' comes back otherwise."""
    if err := _bad_dates(start=start, end=end):
        return {"error": err}
    limit = max(1, min(int(limit), MAX_WORKOUTS))
    settings = ctx.settings()
    local_timezone = _as_timezone(settings["local_timezone"])
    declared_unit_system = settings["unit_system"]
    metric_units = declared_unit_system == "metric"
    display_system = declared_unit_system or "imperial"
    display_units = settings["units"] or V.UNIT_SYSTEMS["imperial"]
    conn = ctx.read_only()
    try:
        active = db.workout_mark_condition(conn, "w")
        marked = db.workout_mark_condition(conn, "w", marked=True)
        if end is None:
            row = conn.execute(
                "SELECT MAX(w.local_date) FROM workouts AS w"
            ).fetchone()
            end = row[0] or date.today().isoformat()
        if start is None:
            start = (date.fromisoformat(end) - timedelta(days=89)).isoformat()
        rows = conn.execute(
            "SELECT w.local_date, w.workout_type, w.duration_min, w.energy_kcal, "
            "w.distance_mi, w.route_ref, w.avg_heart_rate, w.max_heart_rate, "
            "w.start_utc, w.end_utc, w.dedupe_key "
            f"FROM workouts AS w WHERE w.local_date BETWEEN ? AND ? AND {active} "
            "ORDER BY start_utc DESC LIMIT ?", (start, end, limit)).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) FROM workouts AS w WHERE w.local_date BETWEEN ? AND ? "
            f"AND {active}",
            (start, end)).fetchone()[0]
        type_counts = conn.execute(
            f"SELECT w.workout_type, COUNT(*) AS n FROM workouts AS w "
            f"WHERE w.local_date BETWEEN ? AND ? AND {active} "
            "GROUP BY w.workout_type ORDER BY w.workout_type", (start, end)).fetchall()
        excluded = conn.execute(
            f"SELECT COUNT(*) FROM workouts AS w WHERE w.local_date BETWEEN ? AND ? "
            f"AND {marked}", (start, end)).fetchone()[0]
        n_segments = _n_segments_by_workout(conn, [r["dedupe_key"] for r in rows])
    finally:
        conn.close()
    out = []
    for r in rows:
        # The ledger verifier uses this server-owned identity to keep a
        # per-session claim attached to this exact workouts row. It is
        # deliberately the database dedupe key, whose natural key omits
        # source so cross-source sightings remain one physical session.
        row = {
            "date": r["local_date"], "type": r["workout_type"],
            "workout_key": r["dedupe_key"],
            "duration_min": _r(r["duration_min"], 1),
        }
        if metric_units:
            row["energy_kj"] = _r(
                r["energy_kcal"] * V.UNIT_CONVERSION_FACTORS["energy_kcal_to_kj"], 1
            ) if r["energy_kcal"] is not None else None
            row["distance_km"] = _r(
                r["distance_mi"] * V.UNIT_CONVERSION_FACTORS["distance_mi_to_km"], 2
            ) if r["distance_mi"] is not None else None
        else:
            row["energy_kcal"] = _r(r["energy_kcal"], 1)
            row["distance_mi"] = _r(r["distance_mi"], 2)
        row.update({
            "avg_heart_rate": _r(r["avg_heart_rate"], 0),
            "max_heart_rate": _r(r["max_heart_rate"], 0),
            "start_time_local": _local_hhmm(r["start_utc"], local_timezone),
            "end_time_local": _local_hhmm(r["end_utc"], local_timezone),
            "has_route": bool(r["route_ref"]),
            "n_segments": n_segments.get(r["dedupe_key"], 0),
        })
        out.append(row)
    res = {"start": start, "end": end, "count": len(out),
           "total_in_range": total, "truncated": total > len(out),
           "excluded_count": excluded,
           "limit": limit, "workout_counts": [
               {"type": r["workout_type"], "count": r["n"]}
               for r in type_counts],
           "units": {"system": display_system,
                     "declared": declared_unit_system is not None,
                     "distance": display_units["distance"],
                     "energy": display_units["energy"]},
           "workouts": out}
    if res["truncated"]:
        res["note"] = (
            f"showing the {len(out)} most recent of {total} workouts in this "
            f"range — this is NOT the whole range. Raise 'limit' (max "
            f"{MAX_WORKOUTS}) or narrow the dates before counting or totalling.")
    return res


MAX_SEGMENTS = 120


def _render_split(conn, e, n: int, local_timezone: tzinfo | None = None) -> dict:
    """One split, with the heart rate measured over its own window."""
    hr = conn.execute(
        "SELECT AVG(value) a, MAX(value) m FROM records WHERE "
        "metric = 'heart_rate' AND start_utc >= ? AND start_utc < ?",
        (e["start_utc"], e["end_utc"])).fetchone() if e["end_utc"] else None
    return {
        "n": n, "type": e["event_type"],
        "start_local": _local_hhmm(e["start_utc"], local_timezone, seconds=True),
        "end_local": _local_hhmm(e["end_utc"], local_timezone, seconds=True),
        "duration_min": _r(e["duration_min"], 2),
        "avg_heart_rate": _r(hr["a"], 0) if hr else None,
        "max_heart_rate": _r(hr["m"], 0) if hr else None,
    }


def _elapsed_min(start_utc: str, end_utc: str) -> float | None:
    """Wall-clock length of the session. `duration_min` is ACTIVE time, which a
    long pause pushes far below the span the splits actually cover."""
    try:
        return _r((datetime.fromisoformat(end_utc) -
                   datetime.fromisoformat(start_utc)).total_seconds() / 60, 1)
    except (TypeError, ValueError):
        return None


def _covered_min(splits) -> float | None:
    """Minutes the splits account for — compare against the workout's own
    duration to see whether a partition is complete."""
    return _r(sum(s["duration_min"] or 0 for s in splits), 2) if splits else None


@tool
def get_workout_segments(ctx: VaultContext, day: str, workout_type: str | None = None) -> dict:
    """Segment/lap breakdown of each workout on a LOCAL day (YYYY-MM-DD) —
    the watch's splits and intervals. Per split: local start and end time,
    duration (min), and average/max heart rate over that window (from raw
    samples). Pause/resume times and auto-pause counts are included so gaps are
    visible. Optional workout_type filters to one activity (canonical name from
    list_workouts).

    'segments' is ONE consistent partition of the session — the watch often
    stores several rival segmentations of the same workout (a 5-split and a
    4-split view of the same 36 minutes), and any of them describes the whole
    session on its own. Rivals appear in 'alternate_segmentations'. Never merge
    them or add their durations together: that invents a session twice as long
    as the one that happened, with splits nested inside other splits. Check
    'covered_min' against 'duration_min' before calling splits complete.

    Segments come from the periodic full Apple Health export, not the nightly
    phone sync — a recent workout may legitimately have none until the next
    export backfill; n_segments in list_workouts shows which workouts have them.
    Use get_intraday('heart_rate', day) for whole-day HR context instead."""
    try:
        date.fromisoformat(day)
    except ValueError:
        return {"error": "day must be YYYY-MM-DD"}
    local_timezone = _as_timezone(ctx.settings()["local_timezone"])
    conn = ctx.read_only()
    try:
        active = db.workout_mark_condition(conn, "w")
        marked = db.workout_mark_condition(conn, "w", marked=True)
        sql = ("SELECT w.workout_type, w.start_utc, w.end_utc, w.duration_min, "
               "w.dedupe_key FROM workouts AS w WHERE w.local_date = ? "
               f"AND {active}")
        args: list = [day]
        if workout_type:
            sql += " AND w.workout_type = ?"
            args.append(workout_type)
        wrows = conn.execute(sql + " ORDER BY start_utc", args).fetchall()
        excluded_args: list = [day]
        excluded_sql = ("SELECT COUNT(*) FROM workouts AS w WHERE w.local_date = ? "
                        f"AND {marked}")
        if workout_type:
            excluded_sql += " AND w.workout_type = ?"
            excluded_args.append(workout_type)
        excluded = conn.execute(excluded_sql, excluded_args).fetchone()[0]
        workouts = []
        for w in wrows:
            evs = conn.execute(
                "SELECT event_type, start_utc, end_utc, duration_min FROM workout_events "
                "WHERE workout_key = ? ORDER BY start_utc, duration_min LIMIT ?",
                (w["dedupe_key"], MAX_SEGMENTS)).fetchall()
            splits, pauses = [], []
            counts: dict[str, int] = {}
            for e in evs:
                counts[e["event_type"]] = counts.get(e["event_type"], 0) + 1
                if e["event_type"] in ("segment", "lap"):
                    splits.append(dict(e))
                elif e["event_type"] in ("pause", "resume"):
                    pauses.append({"event": e["event_type"],
                                   "at_local": _local_hhmm(
                                       e["start_utc"], local_timezone, seconds=True)})

            # One workout carries several rival partitions; report one of them.
            chains = mx.segment_chains(splits)
            rendered = [[_render_split(conn, e, i, local_timezone)
                         for i, e in enumerate(chain, 1)]
                        for chain in chains]
            primary = rendered[0] if rendered else []
            elapsed = _elapsed_min(w["start_utc"], w["end_utc"])
            entry = {
                "type": w["workout_type"],
                "start_time_local": _local_hhmm(w["start_utc"], local_timezone),
                "duration_min": _r(w["duration_min"], 1),
                "elapsed_min": elapsed,
                "n_segments": len(primary), "segments": primary,
                "covered_min": _covered_min(primary),
                "pauses": pauses,
                "auto_pause_count": counts.get("motion_paused", 0),
                "note": None if primary else
                        "no segments stored for this workout (live phone syncs "
                        "don't carry them; they arrive with the next full-export backfill)",
            }
            if len(rendered) > 1:
                entry["alternate_segmentations"] = [
                    {"n_segments": len(c), "covered_min": _covered_min(c),
                     "segments": c} for c in rendered[1:]]
                entry["note"] = (
                    f"the watch stored {len(rendered)} independent segmentations of "
                    "this workout; 'segments' is the one that covers it best — the "
                    "rest are in alternate_segmentations. Never combine them.")
            elif primary and elapsed and entry["covered_min"] < 0.9 * elapsed:
                # Measured against elapsed time, not duration_min: duration_min
                # is ACTIVE time, so on a paused session (8.6 active inside 38.2
                # elapsed) splits legitimately exceed it.
                entry["note"] = (
                    f"splits cover {entry['covered_min']} of the session's "
                    f"{elapsed} elapsed min — report them as partial.")
            workouts.append(entry)
    finally:
        conn.close()
    if not workouts:
        return {"day": day, "count": 0, "excluded_count": excluded, "workouts": [],
                "note": "no workouts on this day" + (f" of type {workout_type!r}" if workout_type else "")}
    return {"day": day, "count": len(workouts), "excluded_count": excluded,
            "workouts": workouts}


# How far below the cadence cutoff still counts as "one bad week from dropping out".
IMPACT_NEAR_THRESHOLD_STEPS_PER_MIN = 10.0
# Cadence thresholds probed either side of the live one, in steps/min.
IMPACT_SENSITIVITY_OFFSETS = (-10.0, -1.0, 0.0, 1.0, 10.0)


def _impact_period_keys(start: str, end: str, by: str) -> list[date]:
    """Every period the range touches, including ones with no samples at all.

    analysis.impact_volume emits a row only where data exists, so a week off
    vanished and the next week's 'vs the previous period' quietly spanned the
    gap. The ramp rule is week-over-week; a missing week has to be a zero, not
    an absence."""
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if by == "day":
        return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    keys, k = [], d0 - timedelta(days=d0.weekday())   # Monday, as analysis groups
    while k <= d1:
        keys.append(k)
        k += timedelta(days=7)
    return keys


def _impact_days_covered(key: date, by: str, start: str, end: str,
                         *, as_of: str | None = None) -> int:
    """Days of this period that are inside the requested range AND have already
    happened. 5 of 7 on a Friday is why a mid-week week-over-week number reads
    as a collapse."""
    span = 1 if by == "day" else 7
    first = max(key, date.fromisoformat(start))
    horizon = date.fromisoformat(as_of) if as_of else date.today()
    last = min(key + timedelta(days=span - 1), date.fromisoformat(end), horizon)
    return max(0, (last - first).days + 1)


def _impact_change_pct(periods: list[dict], by: str) -> None:
    """Recompute jog_change_pct in place over the gap-filled sequence, and
    withhold it — with the reason — where it would not mean what it says."""
    unit = "day" if by == "day" else "week"
    full = 1 if by == "day" else 7
    prev = None
    for p in periods:
        note = None
        if p["partial"]:
            note = (f"this {unit} covers {p['days_covered']} of its {full} days "
                    f"so far; comparing a part-{unit} against a whole one "
                    f"reports a drop that has not happened")
        elif prev is None:
            note = f"no preceding {unit} inside the requested range"
        elif prev["partial"]:
            note = f"the preceding {unit} is partial, so the two are not comparable"
        elif not prev["jog_minutes"]:
            note = (f"the preceding {unit} recorded no jogging; a percentage "
                    "change off zero is undefined")
        if note:
            p["jog_change_pct"] = None
            p["jog_change_note"] = note
        else:
            p["jog_change_pct"] = _r(
                (p["jog_minutes"] - prev["jog_minutes"]) / prev["jog_minutes"] * 100, 1)
        prev = p


def _jog_threshold_sensitivity(conn, start: str, end: str) -> dict | None:
    """How much of jog_minutes depends on exactly where the cadence cutoff sits.

    Uses the shared bucket set and varies only cadence. Workout scope and the
    implausible-pace floor remain fixed, so the live threshold row is exactly
    the same classification as ``impact_volume``. ``jog_minutes`` is the only
    series value in each sensitivity row and is recomputed at that cadence
    edge; ``cadence_min_steps_per_min``, ``jog_buckets``, and ``live_cutoff``
    are context about the sensitivity calculation and remain unowned.
    """
    bucket_min = mx.IMPACT_BUCKET_SECONDS / 60.0
    live = mx.IMPACT_JOG_CADENCE_MIN
    implausible_mi_ceiling = bucket_min / mx.IMPACT_IMPLAUSIBLE_PACE_MIN
    edges = sorted({round(live + off, 1) for off in IMPACT_SENSITIVITY_OFFSETS
                    if live + off > 0})
    near_floor = max(live - IMPACT_NEAR_THRESHOLD_STEPS_PER_MIN, 0.1)
    bucket_rows = mx.impact_bucket_rows(
        conn, "local_date BETWEEN ? AND ?", (start, end)
    )

    def is_jog_at(row: dict, cadence_min: float) -> bool:
        """Apply one cadence edge to the shared bucket set."""
        mi = row["mi"]
        return (mi <= implausible_mi_ceiling
                and row["in_workout"]
                and (row["cadence_spm"] or 0.0) >= cadence_min)

    counts = [sum(is_jog_at(row, edge) for row in bucket_rows) for edge in edges]
    # "near the cliff" means an otherwise eligible bucket within the diagnostic
    # cadence band below the live threshold.
    near = sum(
        row["in_workout"]
        and row["mi"] <= implausible_mi_ceiling
        and near_floor <= (row["cadence_spm"] or 0.0) < live
        for row in bucket_rows
    )
    at_live = counts[edges.index(live)] if live in edges else 0
    if not at_live and not any(counts):
        return None

    def sensitivity_row(edge: float, count: int) -> dict:
        row = {
            "cadence_min_steps_per_min": edge,
            "jog_buckets": count,
            "jog_minutes": _r(count * bucket_min, 1),
            "live_cutoff": edge == live,
            # jog_minutes is a series value at this cadence edge. The other
            # three fields above describe the diagnostic cutoff and buckets;
            # they are context, not separately owned metric values.
            "field_metrics": {"jog_minutes": "jog_minutes"},
        }
        # Keep this unscoped table from stealing the ordinary period table's
        # sole presentation leaf when fact_template pairs display metadata.
        _add_presentation(row, "jog_minutes", None, row["jog_minutes"],
                          field="jog_minutes")
        return row

    return {
        "sensitivity": [sensitivity_row(e, c) for e, c in zip(edges, counts)],
        "near": {
            "within_steps_per_min": IMPACT_NEAR_THRESHOLD_STEPS_PER_MIN,
            "jog_buckets": at_live,
            "buckets_near_cutoff": near,
            "pct_of_jog_buckets": _r(near / at_live * 100, 1) if at_live else None,
            "note": (
                f"{near} eligible buckets sit within "
                f"{IMPACT_NEAR_THRESHOLD_STEPS_PER_MIN} steps/min below the "
                f"{live:g} steps/min cadence cutoff. Read a fall in "
                "jog_minutes against the sensitivity row above before "
                "calling it a fall in training volume."),
        },
    }


def _impact_periods(rows: list[dict], start: str, end: str,
                    by: str, *, as_of: str | None = None) -> list[dict]:
    """Return the gap-filled impact periods with their completeness labels.

    Both the ordinary impact tool and its block-comparison mode need exactly
    this sequence. Keeping it here means the aggregate cannot accidentally
    lose the tool-layer completeness rule that analysis.impact_volume does not
    emit itself.
    """
    if not rows:
        return []

    # The query should already be bounded by this same horizon, but keep the
    # pure presentation helper defensive: a future period must never become a
    # published zero merely because a caller supplied an end beyond as_of.
    horizon = date.fromisoformat(as_of) if as_of else date.today()
    start_date = date.fromisoformat(start)
    end_date = min(date.fromisoformat(end), horizon)
    if start_date > horizon:
        return []
    end = end_date.isoformat()

    have = {r["period_start"]: r for r in rows}
    periods = []
    for key in _impact_period_keys(start, end, by):
        iso = key.isoformat()
        bucket_end = min(key + timedelta(days=6), end_date).isoformat()
        covered = _impact_days_covered(key, by, start, end, as_of=as_of)
        row = have.get(iso) or {
            "period_start": iso, "jog_minutes": 0.0, "jog_miles": 0.0,
            "jog_pace_min_per_mi": None, "walk_minutes": 0.0, "walk_miles": 0.0,
            "jog_change_pct": None,
        }
        periods.append({**row, "no_data": iso not in have,
                        # This labels only the canonical jog-minute series.
                        # Other fields in the row are impact context and stay
                        # available for Tier 1 path provenance.
                        "metric": "jog_minutes",
                        "period": (iso if by == "day"
                                   else f"{iso}:{bucket_end}"),
                        "days_covered": covered,
                        "partial": covered < (1 if by == "day" else 7)})
    _impact_change_pct(periods, by)
    return periods


def _impact_block_comparison(periods: list[dict], start: str, end: str,
                             weeks_per_block: int, anchor: str,
                             *, as_of: str | None = None) -> dict:
    """Aggregate adjacent weekly jog-minute blocks from labelled tool rows."""
    anchor_aliases = {
        "last_complete_week": "last_complete_week",
        "last_day_with_data": "last_day_with_data",
        # These short forms are accepted for callers, but the canonical names
        # are always emitted so a response never leaves the convention implicit.
        "complete_week": "last_complete_week",
        "data_end": "last_day_with_data",
    }
    canonical_anchor = anchor_aliases.get(anchor)
    if canonical_anchor is None:
        return {"error": (
            "anchor must be 'last_complete_week' or 'last_day_with_data'"
        )}
    if len(periods) < 2 * weeks_per_block:
        return {"error": (
            f"the requested range has {len(periods)} weekly periods; "
            f"at least {2 * weeks_per_block} are required for two "
            f"{weeks_per_block}-week blocks"
        )}

    trailing = periods[-1]
    trailing_partial = bool(trailing["partial"])
    if canonical_anchor == "last_complete_week":
        anchor_index = len(periods) - 1
        while anchor_index >= 0 and periods[anchor_index]["partial"]:
            anchor_index -= 1
        if anchor_index < 0:
            return {"error": "the requested range contains no complete week"}
    else:
        anchor_index = len(periods) - 1

    recent_start = anchor_index - weeks_per_block + 1
    prior_start = recent_start - weeks_per_block
    if prior_start < 0:
        return {"error": (
            f"the requested range does not include {weeks_per_block} weeks "
            "before the selected anchor"
        )}
    recent = periods[recent_start:anchor_index + 1]
    prior = periods[prior_start:recent_start]

    # Only the explicitly selected last-day anchor may contain a partial week.
    # A leading partial week or a partial predecessor would make the comparison
    # a different question and must be rejected rather than normalized away.
    allowed_partial = (canonical_anchor == "last_day_with_data"
                       and recent[-1]["partial"])
    if any(p["partial"] for p in prior) or any(p["partial"] for p in recent[:-1]):
        return {"error": (
            "the selected blocks contain a partial non-trailing week; "
            "choose an explicit range containing complete Monday-Sunday weeks"
        )}
    if recent[-1]["partial"] and not allowed_partial:
        return {"error": "the selected trailing week is partial"}

    def block(rows: list[dict]) -> dict:
        values = [float(row["jog_minutes"]) for row in rows]
        total = round(sum(values), 1)
        mean = round(total / len(values), 1)
        starts = [row["period_start"] for row in rows]
        return {
            "metric": "jog_minutes",
            "period": {
                "start": starts[0], "end": starts[-1],
                "period_starts": starts,
            },
            "period_starts": starts,
            "weeks": [
                {"metric": "jog_minutes", "period": row["period"],
                 "field": "jog_minutes", "value": row["jog_minutes"],
                 "days_covered": row["days_covered"],
                 "days_expected": 7, "partial": row["partial"],
                 "no_data": row["no_data"]}
                for row in rows
            ],
            "total": total,
            "mean": mean,
        }

    recent_block, prior_block = block(recent), block(prior)
    total_delta = round(recent_block["total"] - prior_block["total"], 1)
    mean_delta = round(recent_block["mean"] - prior_block["mean"], 1)
    prior_total = prior_block["total"]
    change_note = None
    if not prior_total:
        change_pct = None
        change_note = "a percentage change off a zero prior total is undefined"
    elif abs(mean_delta) <= 0.1:
        change_pct = None
        change_note = (
            "the published mean changed by at most one 0.1-minute rounding unit; "
            "the percentage is not resolvable at the payload's precision"
        )
    else:
        change_pct = round(total_delta / prior_total * 100, 1)

    if canonical_anchor == "last_complete_week":
        anchor_period = periods[anchor_index]
        anchor_end = (date.fromisoformat(anchor_period["period_start"])
                      + timedelta(days=6)).isoformat()
        trailing_reason = (
            "dropped because anchor='last_complete_week'; it covers "
            f"{trailing['days_covered']} of 7 days inside the explicit range"
            if trailing_partial else
            "no partial trailing week was present in the explicit range"
        )
        trailing_included = False if trailing_partial else None
    else:
        anchor_end = end
        trailing_reason = (
            "included because anchor='last_day_with_data'; it is the trailing "
            f"week ending at the explicit end and covers {trailing['days_covered']} "
            "of 7 days"
            if trailing_partial else
            "included; the explicit end falls on the complete trailing week"
        )
        trailing_included = True

    change = {
        "mean_delta": mean_delta,
        "total_delta": total_delta,
        "total_delta_pct": change_pct,
    }
    if change_note:
        change["total_delta_pct_note"] = change_note

    return {
        "metric": "jog_minutes",
        "weeks_per_block": weeks_per_block,
        "anchor": canonical_anchor,
        "anchor_end": anchor_end,
        "requested_range": {"start": start, "end": end},
        "blocks": {"recent": recent_block, "prior": prior_block},
        "change": change,
        "completeness": {
            "rule": (
                "days_covered is the number of days inside the explicit start/end "
                "range and no later than the session as_of date; a week is complete only "
                "when days_covered is 7"
            ),
            "as_of": as_of or date.today().isoformat(),
            "end_default": False,
            "partial_trailing_week": {
                "period_start": trailing["period_start"],
                "days_covered": trailing["days_covered"],
                "days_expected": 7,
                "partial": trailing_partial,
                "included": trailing_included,
                "reason": trailing_reason,
            },
        },
    }


@tool
def get_impact_volume(ctx: VaultContext, start: str, end: str, by: str = "week",
                      weeks_per_block: int | None = None,
                      anchor: str = "last_complete_week") -> dict:
    """Minutes actually spent JOGGING vs walking over a date range — the real
    impact-volume dial for the plan's absolute weekly jog-minute dose ceiling.

    Use this instead of summing workout durations: a run/walk workout's duration
    counts its walk breaks, which overstates impact by roughly 2x. This buckets
    90-second interval still resolves, and counts a bucket as jogging when its
    start-bucket ``step_count`` sum multiplied by three is at least 140
    steps/min and the bucket start lies inside any workout window. Workout type
    and heart rate do not change that classification. The implausible-pace floor
    and positive distance guard remain in force. Returns per-week (Monday
    anchored) or per-day rows with jog_minutes, jog_miles, average jog pace,
    walking equivalents, and jog_change_pct vs the previous period.

    Jog minutes, not workout duration, are the volume dial; the two are not
    convertible. The plan's governing dials are the longest continuous jog
    block at HR <= 150 and the absolute weekly jog-minute dose ceiling.
    jog_change_pct is diagnostic context for explaining week-over-week change,
    not a governing rule. It is null whenever it would not mean what it says —
    a partial period, a partial predecessor, or a predecessor with no jogging —
    and 'jog_change_note' says which. Never substitute your own division in
    that case: the current week is partial all week, and 2 days measured against
    7 reads as a 71% drop that did not happen. 'partial' and 'days_covered' say
    how much of the period has happened; 'no_data' marks a period with no
    distance samples at all (those are emitted as zeros so the sequence has no
    gaps to compare across).

    'jog_threshold_sensitivity' gives jog_minutes recomputed at CADENCE cutoffs
    either side of the live one, and 'jog_near_threshold' the share of eligible
    buckets just below that cutoff. A drop in jog_minutes can therefore reflect
    the hard cadence edge rather than less running — check these before
    reporting a fall in volume.

    When ``weeks_per_block`` is set (for example, ``4``), the same tool also
    returns ``block_comparison``: two adjacent blocks of that many Monday-
    anchored weeks, their per-week ``jog_minutes`` inputs, means, totals, and
    mean and total deltas plus a total-based percentage change. The percentage
    is withheld when the prior total is zero or the published mean change is
    only rounding noise. ``anchor`` must be ``last_complete_week`` or
    ``last_day_with_data``. The former drops a partial trailing week; the latter
    includes it. ``start`` and ``end`` are required and never default to today;
    the completeness output states the session's ``as_of`` horizon.

    Note that walk_minutes covers ALL ambulatory movement the watch saw, not
    just deliberate walks, so it runs far higher than session time — don't
    report it as exercise volume. Dates are local YYYY-MM-DD."""
    for label, value in (("start", start), ("end", end)):
        try:
            date.fromisoformat(value)
        except ValueError:
            return {"error": f"{label} must be YYYY-MM-DD"}
    if start > end:
        return {"error": "start must be on or before end"}
    conn = ctx.read_only()
    try:
        as_of = A._as_of(conn, None)
        as_of_date = date.fromisoformat(as_of)
        requested_start = date.fromisoformat(start)
        if requested_start > as_of_date:
            return {"start": start, "end": end, "by": by, "as_of": as_of,
                    "count": 0, "periods": [],
                    "reason": "window_after_as_of"}

        effective_end = min(date.fromisoformat(end), as_of_date).isoformat()
        rows = A.impact_volume(conn, start, effective_end, by=by)
        sens = _jog_threshold_sensitivity(conn, start, effective_end)
        # Fill periods analysis had no rows for, then recompute the change
        # column over the complete sequence. The same labelled rows feed block
        # comparison. Do this while the read-only connection is open only
        # because the source rows are computed here; the helper itself is pure.
        periods = _impact_periods(rows, start, effective_end, by, as_of=as_of)
    except ValueError as e:
        return {"error": str(e)}
    finally:
        conn.close()
    if not rows:
        return {"start": start, "end": end, "by": by, "as_of": as_of,
                "count": 0, "periods": [],
                "note": "no distance samples in this range"}

    out = {"start": start, "end": end, "by": by, "as_of": as_of,
           "count": len(periods), "periods": periods,
           "jog_cadence_threshold_steps_per_min": A.IMPACT_JOG_CADENCE_MIN}
    for row in periods:
        if "jog_minutes" in row:
            _add_presentation(row, "jog_minutes", row.get("period"),
                              row.get("jog_minutes"), field="jog_minutes")
    if sens:
        out["jog_threshold_sensitivity"] = sens["sensitivity"]
        out["jog_near_threshold"] = sens["near"]
    if weeks_per_block is not None:
        if by != "week":
            out["block_comparison_error"] = (
                "weeks_per_block is only valid when by='week'"
            )
        elif (isinstance(weeks_per_block, bool)
              or not isinstance(weeks_per_block, int)
              or weeks_per_block < 1):
            out["block_comparison_error"] = "weeks_per_block must be a positive integer"
        else:
            comparison = _impact_block_comparison(
                periods, start, end, weeks_per_block, anchor, as_of=as_of
            )
            if "error" in comparison:
                out["block_comparison_error"] = comparison["error"]
            else:
                for block in (comparison.get("blocks") or {}).values():
                    _add_stat_presentations(block, "jog_minutes",
                                            block.get("period"))
                out["block_comparison"] = comparison
    return out


@tool
def get_sleep_regularity(ctx: VaultContext, start: str | None = None, end: str | None = None) -> dict:
    """How regular the user's sleep TIMING is, over the whole sleep-timing series.

    'interval_regularity.match_pct' is NOT the Sleep Regularity Index and must
    not be compared against published SRI values. Real SRI is computed from
    minute-level sleep/wake state and is scaled -100..+100. This reconstructs
    one [bedtime, wake_time] interval per night and reports the percentage of
    minutes matching the state 24 hours later, so it counts sleep latency and
    every mid-night awakening as sleep. 'calibration' quantifies exactly that:
    it compares the reconstructed interval against real staged sleep on the
    ~1,000 days that carry both, and gives the mean bias in minutes with limits
    of agreement. Against 1,060 days carrying real sleep stages the proxy runs
    +36.2 minutes high on average, with limits of agreement from -248 to +320
    minutes — it tracks on average and disagrees badly on any single night.
    Quote the bias whenever you quote the score.

    'midpoint_variability' is the rolling 28-day SD of sleep midpoint in hours
    — lower is more regular, and it is the cleanest week-to-week signal here.
    'cosinor' separates the seasonal swing from linear drift: read
    'drift_hours_per_year' as bedtime creep with the seasonal component removed.
    'plan_compliance' reports three configured bedtime bands: nights inside
    the 11:00 PM anchor, social nights from after 23:00 through 00:30, and
    nights past the 00:30 limit. The middle band is NOT graded either way — the
    plan allows social nights to run to 12:30 and nothing in the data tells a
    social night from a scrolling night. 'inside_anchor_pct' counts only the
    first band, so it is not a compliance score and its complement is not a
    shortfall: report all three counts, and never call a social night a miss.

    The sleep-timing series is typically the largest coherent one in the DB and
    runs continuously through gaps in watch coverage, which can make it the only
    variable that links a much earlier era of a training history to the present
    one. Sleep is attributed to
    the WAKE day. Dates are local YYYY-MM-DD; both default to all of history."""
    if err := _bad_dates(start=start, end=end):
        return {"error": err}
    start = start or "2016-01-01"
    end = end or date.today().isoformat()
    if start > end:
        return {"error": "start must be on or before end"}
    conn = ctx.read_only()
    try:
        out = {
            "start": start, "end": end,
            "interval_regularity": SR.interval_regularity(
                SR.nights_from_db(conn, start, end)),
            "calibration": SR.calibrate_against_stages(conn, start, end),
            "midpoint_variability": SR.midpoint_variability(conn, start, end),
            "cosinor": SR.cosinor(*_midpoint_series(conn, start, end)),
            "plan_compliance": SR.plan_compliance(conn, start, end),
            "not_sri": ("Reconstructed from bedtime/wake timing, not from "
                        "minute-level sleep/wake state. Not comparable to "
                        "published SRI values."),
        }
        midpoint = out.get("midpoint_variability") or {}
        leaf = mx.presentation_leaf(
            "sleep_midpoint_sd_28d", f"{start}:{end}",
            midpoint.get("latest_sd_hours"), field="latest_sd_hours")
        if leaf is not None:
            midpoint["presentation"] = leaf
        return out
    finally:
        conn.close()


def _midpoint_series(conn, start: str, end: str):
    rows = conn.execute(
        "SELECT date, last AS v FROM daily_metrics WHERE metric = 'sleep_midpoint' "
        "AND date BETWEEN ? AND ? AND last IS NOT NULL ORDER BY date",
        (start, end)).fetchall()
    return [r["date"] for r in rows], [r["v"] for r in rows]


@tool
def get_training_load_detail(ctx: VaultContext, start: str | None = None,
                             end: str | None = None) -> dict:
    """Daily session-scoped training load, and what the live ACWR is built on.

    'hr_load_proxy' is an intensity-aware load measure computed per WORKOUT and
    summed per day. It is NOT TRIMP: the functional form is Banister's, but
    HR_rest and HR_max here are estimated from observational data rather than
    measured under the protocols the published formula assumes, and heart rate
    during run/walk intervals is not continuous-exercise intensity. Its units
    are arbitrary — compare it against itself over time, never against a
    published load figure.

    A day with workouts but no usable heart-rate coverage reports
    status 'unknown' and load null. That is NOT a rest day, and must never be
    read or reported as one: a week of non-wear once produced 'acwr 0.08,
    detraining' from exactly that confusion. 'sessions_without_hr' says how many
    sessions could not be measured.

    'live_acwr' is what get_briefing reports, and SINCE 2026-08-09 it is computed
    from hr_load_proxy. It was computed from active_energy before that date.
    THE TWO ARE NOT INTERCHANGEABLE and any insight, review or briefing written
    before 2026-08-09 quotes the active_energy figure: never compare an ACWR
    number from an older insight against one from this tool. Over the 40 days to
    2026-08-09 the two inputs agreed on the band 32% of the time — active_energy
    read 'sweet-spot' on all 40, including the mid-July week where jog minutes
    more than quadrupled, because it counts all daily movement and cannot see
    intensity. hr_load_proxy read 'ramping-fast' through that week.

    For ACWR only, a day with no session but with the watch worn counts as a
    zero, because a rest day's training load is zero by observation. A day the
    watch was OFF stays absent and is never zeroed.

    Dates are local YYYY-MM-DD; both default to the last 90 days."""
    if err := _bad_dates(start=start, end=end):
        return {"error": err}
    end = end or date.today().isoformat()
    start = start or (date.fromisoformat(end) - timedelta(days=90)).isoformat()
    if start > end:
        return {"error": "start must be on or before end"}
    conn = ctx.read_only()
    try:
        days = HL.daily_load(conn, start, end)
        return {
            "start": start, "end": end,
            "days": days,
            "days_unknown": sum(1 for d in days if d["status"] == "unknown"),
            "live_acwr": A.training_load(conn, as_of=end),
            "live_acwr_input": A.ACWR_LOAD_METRICS[0],
            "note": ("live_acwr has been computed from hr_load_proxy since "
                     "2026-08-09; before that date it came from active_energy "
                     "and the two are not comparable. Units are arbitrary."),
        }
    finally:
        conn.close()


@tool
def get_run_form(ctx: VaultContext, workout_date: str | None = None, start: str | None = None,
                 end: str | None = None) -> dict:
    """Running form for one session, or the banded weekly trend over a range.

    THREE THINGS TO KNOW BEFORE QUOTING ANY NUMBER FROM THIS TOOL.

    First, 'efficiency_change_pct' is NOT aerobic decoupling and the familiar
    <5% rule does not apply to it. Decoupling is defined on prolonged
    steady-state efforts; these are 35-50 minute run/walk sessions containing
    10-25 minutes of jogging. Efficiency here is speed/HR over jog buckets only,
    compared between the first and second half of cumulative JOG time — so walk
    breaks cannot decide the answer by where they happened to fall. Negative
    means efficiency fell across the session.

    Second, judge a session against 'personal_reference', not against any
    published figure. That block is the user's own distribution of the same
    measure, with 'minimum_detectable_change_pct' being roughly the smallest
    shift that clears the noise in their own history. A change smaller than that
    is not a finding. The reference excludes the session being reported, so it
    is not compared against itself.

    Third, 'walk_structure' is where late fatigue usually shows first in a
    run/walk session: compare 'first_half_walk_fraction' with
    'second_half_walk_fraction'. The efficiency ratio has to exclude walk
    buckets or pace swamps it, so this is the part it cannot see.

    Nothing here controls for terrain, grade, heat, humidity, surface or GPS
    quality — none of which are in the Apple Health export. Every number is
    descriptive. The 13-15 min/mi reference band used for the weekly trend was
    chosen from an observed training week and is not an independent standard.

    Pass workout_date for one session, start+end for the weekly trend, or
    nothing for the most recent run. Dates are local YYYY-MM-DD."""
    if err := _bad_dates(workout_date=workout_date, start=start, end=end):
        return {"error": err}
    conn = ctx.read_only()
    try:
        if start and end:
            if start > end:
                return {"error": "start must be on or before end"}
            return {"mode": "trend", **RF.banded_weekly(conn, start, end)}

        active = db.workout_mark_condition(conn, "w")
        if workout_date:
            row = conn.execute(
                "SELECT w.start_utc, w.end_utc, w.local_date FROM workouts AS w "
                "WHERE w.workout_type = 'running' AND w.local_date = ? "
                f"AND {active} ORDER BY w.duration_min DESC LIMIT 1",
                (workout_date,)).fetchone()
        else:
            row = conn.execute(
                "SELECT w.start_utc, w.end_utc, w.local_date FROM workouts AS w "
                f"WHERE w.workout_type = 'running' AND {active} "
                "ORDER BY w.start_utc DESC LIMIT 1").fetchone()
        if not row:
            return {"found": False,
                    "error": f"no running workout on {workout_date}"
                             if workout_date else "no running workouts on record"}

        return {
            "mode": "session",
            "found": True,
            "date": row["local_date"],
            "efficiency_change": RF.jog_efficiency_change(
                conn, row["start_utc"], row["end_utc"]),
            "walk_structure": RF.walk_structure(
                conn, row["start_utc"], row["end_utc"]),
            "personal_reference": RF.personal_reference(
                conn, "2026-01-01", row["local_date"],
                exclude_start_utc=row["start_utc"]),
        }
    finally:
        conn.close()


@tool
def get_briefing(ctx: VaultContext, scope: str = "daily", day: str | None = None) -> dict:
    """Deterministic health briefing: coverage, recovery readiness, trends,
    training load (ACWR), movers, long-term deltas, safe suggestions, highlights,
    and ordered talking points. scope='daily' (compact) or 'deep' (full) — any
    other scope is an 'error', not a near-enough default. 'day' defaults to the
    latest data date. Read-only; precomputed — prefer this over assembling
    metrics by hand."""
    if err := _bad_dates(day=day):
        return {"error": err}
    # Unvalidated, an unknown scope fell through to daily's mover count and no
    # long-term section, then echoed 'scope': 'weekly' back — a deep briefing
    # the agent had every reason to believe it had received.
    if scope not in BRIEFING_SCOPES:
        return {"error": f"scope={scope!r} is not a briefing scope; use "
                         f"{' or '.join(repr(s) for s in BRIEFING_SCOPES)}"}
    conn = ctx.read_only()
    try:
        return A.build_briefing(conn, scope=scope, as_of=day)
    finally:
        conn.close()


@tool
def get_latest(ctx: VaultContext, metric: str) -> dict:
    """Most recent reading for a metric: the latest daily aggregate plus the
    most recent stored sample (value + local timestamp).

    `latest_sample.resolution_seconds` says what that sample IS. 0 means it is
    a sample as the device recorded it. A positive number means the vault stores
    that series aggregated into windows of that width (D9), so the value is a
    sum over the window and the timestamp is its earliest sample — calling it
    "the latest reading" would be a claim about an instant that never happened.
    """
    conn = ctx.read_only()
    try:
        if not _metric_exists(conn, metric):
            return {"error": f"unknown metric {metric!r}. Call list_available_metrics."}
        col = _value_col(metric)
        dm = conn.execute(
            f"SELECT date, {col} v, unit FROM daily_metrics WHERE metric = ? "
            f"ORDER BY date DESC LIMIT 1", (metric,)).fetchone()
        # Keep the raw sample on the same local day and with the same total
        # order as daily_metrics.last. Otherwise a last-valued metric could
        # report two different readings under one label when a malformed
        # local_date or a timestamp tie put the rows on different paths.
        # Two independent reasons the sample half may be withheld, and they
        # are not the same reason: D3 (the series is not in this vault at all)
        # and RAW_SAMPLES (this session's output reaches a provider). Both must
        # say which, or "no sample" reads as a claim about the data.
        raw = conn.execute(
            "SELECT value, unit, start_local FROM records WHERE metric = ? "
            "AND local_date = ? ORDER BY start_utc DESC, end_utc DESC, id DESC "
            "LIMIT 1", (metric, dm["date"])).fetchone() \
            if dm and V.raw_series_available(metric) and ctx.can(RAW_SAMPLES) \
            else None
    finally:
        conn.close()
    out = {
        "metric": metric, "agg": _agg(metric),
        "latest_day": {"date": dm["date"], "value": _r(dm["v"]), "unit": dm["unit"]} if dm else None,
        "latest_sample": {"value": _r(raw["value"]), "unit": raw["unit"],
                          "local_time": raw["start_local"],
                          "resolution_seconds": V.raw_resolution_seconds(metric)}
        if raw else None,
    }
    if out["latest_day"] is not None:
        _add_presentation(out["latest_day"], metric, dm["date"],
                          out["latest_day"]["value"], field="value")
    if out["latest_sample"] is not None:
        _add_presentation(out["latest_sample"], metric, dm["date"],
                          out["latest_sample"]["value"], field="value")
    if not V.raw_series_available(metric):
        # `latest_sample: null` alone would say "the last reading has no
        # timestamp", which is a claim about the data. Say which it is.
        out["latest_sample_status"] = V.raw_unavailable(
            metric, needed_for="the most recent raw sample")
    elif not ctx.can(RAW_SAMPLES):
        out["latest_sample_status"] = {
            "status": "withheld",
            "metric": metric,
            "reason": "provider_facing_session",
            "detail": (
                "This session's output reaches a model provider, so it does not "
                "receive values identifying a single stored reading. The daily "
                "aggregate above is the answer to use; nothing about it is "
                "withheld."
            ),
        }
    return out


# --------------------------------------------------------------------------- #
# correlation tools
# --------------------------------------------------------------------------- #
def _corr_window(conn, metric_y: str, period: str) -> tuple[str, str]:
    """Resolve a period spec to [start, end] anchored on metric_y's latest date."""
    anchor = _anchor_end(conn, metric_y)
    start_iso, end_iso = _parse_period(period, anchor)
    if start_iso is None:  # 'all'
        start_iso = conn.execute(
            "SELECT MIN(date) FROM daily_metrics WHERE metric = ?", (metric_y,)
        ).fetchone()[0]
    return start_iso, end_iso


@tool
def correlate_metrics(ctx: VaultContext, metric_x: str, metric_y: str, lag_days: int = 0,
                      period: str = "90d") -> dict:
    """Correlation between two daily metrics with optional lag. lag_days=1
    tests 'x on day D-1 vs y on day D' (e.g. yesterday's activity vs tonight's
    sleep); lag_days=0 is same-day. Pairs days present in BOTH series over the
    period ('30d','12w','6m','1y','all'; anchored to metric_y's latest date);
    watch-derived metrics exclude days with wear_hours < 12. Returns Pearson r
    (with 95% CI), Spearman rho, p-values, n_pairs, coverage, and caveats.
    Correlation is NOT causation — report it as an association."""
    conn = ctx.read_only()
    try:
        for m in (metric_x, metric_y):
            if not _metric_exists(conn, m):
                return {"error": f"unknown metric {m!r}. Call list_available_metrics."}
        lag_days = max(0, min(int(lag_days), 7))
        try:
            start_iso, end_iso = _corr_window(conn, metric_y, period)
        except ValueError as e:
            return {"error": str(e)}
        xs, ys, meta = C.paired_series(conn, metric_x, metric_y, lag_days,
                                       start_iso, end_iso)
    finally:
        conn.close()
    res = C.correlate(xs, ys)
    lag_semantics = ("same day" if lag_days == 0 else
                     f"{metric_x} on day D-{lag_days} vs {metric_y} on day D")
    caveats = []
    if res["status"] == "ok":
        if res["n_pairs"] < 20:
            caveats.append(f"only {res['n_pairs']} paired days — low confidence")
        for name, cov in ((metric_x, meta["coverage_x_pct"]),
                          (metric_y, meta["coverage_y_pct"])):
            if cov is not None and cov < 80:
                caveats.append(f"{name} present on only {cov}% of window days")
        if meta["dropped_low_wear"]:
            caveats.append(f"{meta['dropped_low_wear']} days excluded for wear_hours < 12")
        caveats.append("correlation is not causation — report as association")
    return {"metric_x": metric_x, "metric_y": metric_y, "lag_days": lag_days,
            "lag_semantics": lag_semantics, "period": period,
            "window": [start_iso, end_iso], **meta, **res, "caveats": caveats}


@tool
def scan_correlations(ctx: VaultContext, target: str, period: str = "90d", lags: str = "0,1",
                      max_results: int = 15) -> dict:
    """Sweep ALL metrics against `target` at each lag (comma-separated day
    lags, e.g. '0,1'; lag 1 = candidate on day D-1 vs target on day D) and
    rank associations by |Spearman rho|. All p-values are jointly corrected
    with Benjamini-Hochberg FDR (q=0.10): only trust passed_fdr=true, and use
    tested_count to report honestly ('3 of 41 associations survived
    correction'). related_group=true pairs are trivially coupled (e.g. sleep
    stages vs total sleep) — never report them as findings. Follow up on
    interesting hits with correlate_metrics."""
    conn = ctx.read_only()
    try:
        if not _metric_exists(conn, target):
            return {"error": f"unknown metric {target!r}. Call list_available_metrics."}
        try:
            lag_list = tuple(sorted({max(0, min(int(p), 7))
                                     for p in lags.split(",") if p.strip()})) or (0,)
        except ValueError:
            return {"error": "lags must be comma-separated integers, e.g. '0,1'"}
        try:
            start_iso, end_iso = _corr_window(conn, target, period)
        except ValueError as e:
            return {"error": str(e)}
        tests = C.scan(conn, target, start_iso, end_iso, lag_list)
    finally:
        conn.close()
    n_pass = sum(1 for t in tests if t["passed_fdr"])
    max_results = max(1, min(int(max_results), 50))
    return {"target": target, "period": period, "window": [start_iso, end_iso],
            "lags": list(lag_list), "fdr_q": C.FDR_Q,
            "tested_count": len(tests), "passed_fdr_count": n_pass,
            "note": (f"{n_pass} of {len(tests)} tested associations passed FDR "
                     f"(q={C.FDR_Q}). Ignore related_group=true rows."),
            "results": tests[:max_results]}


# --------------------------------------------------------------------------- #
# write tools: write_insight (daily digests) + log_subjective (check-in)
# --------------------------------------------------------------------------- #
@tool
def write_insight(ctx: VaultContext, day: str, text: str, tags: str = "") -> dict:
    """Record a daily insight/summary (the only write operation). 'day' is the
    date the insight is about (YYYY-MM-DD); 'tags' is a comma-separated list.
    Shown on the Grafana dashboard."""
    try:
        date.fromisoformat(day)
    except ValueError:
        return {"ok": False, "error": "day must be YYYY-MM-DD"}
    if not text or not text.strip():
        return {"ok": False, "error": "text is required"}
    conn = ctx.connect()
    try:
        db.init_db(conn)
        rid = db.write_insight(conn, day, text.strip(), tags.strip())
    finally:
        conn.close()
    return {"ok": True, "id": rid, "day": day, "tags": tags}


@tool
def log_subjective(ctx: VaultContext, day: str, stress: int | None = None,
                   soreness: int | None = None, energy: int | None = None,
                   sleep_quality: int | None = None,
                   caffeine_drinks: float | None = None,
                   alcohol_drinks: float | None = None,
                   food_note: str = "", jog_niggle: str = "",
                   jog_niggle_detail: str = "", talk_test: str = "",
                   notes: str = "") -> dict:
    """Store the user's nightly subjective check-in for a day (YYYY-MM-DD).
    Scales: stress, soreness (muscle), energy, sleep_quality (last night's
    sleep) are integers 1=lowest/worst to 5=highest/best for energy and
    sleep_quality, 5=most severe for stress and soreness. caffeine_drinks /
    alcohol_drinks are counts of drinks (non-negative; 0 is a valid answer).
    'notes' is a catch-all for anything that doesn't fit (illness, travel,
    niggles, life events).

    Three further fields, kept deliberately cheap to answer — one free-text
    line every night, two taps on jog days:
      food_note          free text, EVERY night. What they ate or drank that is
                         worth noting.
      jog_niggle         'y' or 'n', JOG DAYS ONLY, with
      jog_niggle_detail  free text: where, and whether it changed the session.
                         This replaces the 1-5 soreness scale FOR RUNNING only.
                         Keep logging `soreness` as well — the 1-5 series is
                         still the right instrument for hikes, where it moves.
      talk_test          'comfortable' | 'not_sure' | 'not_comfortable',
                         JOG DAYS ONLY.
    An out-of-domain jog_niggle or talk_test is rejected rather than stored.

    Gather the answers first, then call this ONCE with every field given;
    omit fields that weren't answered. Upserts per day: a later correction
    ("actually 2 beers", including for a previous day) is another call with
    just that field — other fields keep their stored values. Numeric fields
    also become daily metrics (subjective_stress, subjective_soreness,
    subjective_energy, subjective_sleep_quality, caffeine_drinks,
    alcohol_drinks) usable in summaries and correlations. Notes are replaced
    only when provided — they cannot be cleared to empty. 'day' must be today or
    earlier: a check-in is a report of a day that happened."""
    if err := _bad_dates(day=day):
        return {"ok": False, "error": err}
    # A check-in about a day that hasn't happened is not a correction anyone can
    # make. Left open, a mis-parsed "tomorrow" writes a future row that then
    # mirrors into daily_metrics and shifts every baseline that reads forward.
    if day > date.today().isoformat():
        return {"ok": False,
                "error": f"day={day!r} is in the future; a check-in reports a "
                         "day that has happened. Today is "
                         f"{date.today().isoformat()}."}
    conn = ctx.connect()
    try:
        db.init_db(conn)
        row = subj.log(conn, day, stress=stress, soreness=soreness,
                       energy=energy, sleep_quality=sleep_quality,
                       caffeine_drinks=caffeine_drinks,
                       alcohol_drinks=alcohol_drinks,
                       food_note=food_note.strip() or None,
                       jog_niggle=jog_niggle.strip() or None,
                       jog_niggle_detail=jog_niggle_detail.strip() or None,
                       talk_test=talk_test.strip() or None,
                       notes=notes.strip() or None)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()
    return {"ok": True, "stored": row}


@tool
def get_subjective(ctx: VaultContext, start_date: str, end_date: str) -> dict:
    """Subjective check-in rows (stress/soreness/energy/sleep_quality 1-5,
    caffeine/alcohol drink counts, free-text notes) for a date range,
    ascending. This is the only way to read the notes; the numeric fields are
    also ordinary daily metrics queryable via summarize_metric etc.

    `get_subjective` keeps flat day fields and adds `period` equal to the day
    plus `field_metrics` for the non-null rating fields (`stress`, `soreness`,
    `energy`, `sleep_quality`); cite the direct field with its mapped
    `subjective_*` metric, and omit `metric` for fields absent from
    `field_metrics`."""
    if err := _bad_dates(start=start_date, end=end_date):
        return {"ok": False, "error": err}
    # Read-only, like every other read tool. This one opened the 3.6 GB DB
    # WRITABLE and ran init_db's DDL on it on every call — schema churn and a
    # write lock on the production database to answer a SELECT.
    conn = ctx.read_only()
    try:
        days = subj.get_range(conn, start_date, end_date)
    except sqlite3.OperationalError as e:
        # Read-only can no longer CREATE TABLE its way past a missing table, so
        # the missing table has to be an answer instead of a crash.
        if "no such table" not in str(e):
            raise
        return {"days": [], "count": 0,
                "note": "no check-ins have ever been recorded"}
    finally:
        conn.close()
    return {"days": days, "count": len(days)}


# --------------------------------------------------------------------------- #
# Food catalog — nutrition reference with provenance
# --------------------------------------------------------------------------- #
_TIER_HELP = (
    "source tiers, WORST to BEST: 'estimate' (no source; reasoned from a "
    "similar product), 'web' (a nutrition database or manufacturer page), "
    "'label_text' (the user typed the label numbers), 'label_photo' (the user "
    "sent a photo of the label). These are not interchangeable and every read "
    "returns the tier, so a guess can never present itself as a measurement."
)


def _empty_catalog(e: sqlite3.OperationalError):
    """The catalog table does not exist until the first write, and read tools
    open read-only so they cannot create it. Mirrors get_subjective above."""
    if "no such table" not in str(e):
        raise e
    return {"items": [], "count": 0,
            "note": "the food catalog is empty — nothing has been added yet"}


@tool
def food_lookup(ctx: VaultContext, query: str = "") -> dict:
    """Look up food items and their nutrition per serving from the user's catalog.

    Case-insensitive substring search over the item name and its aliases; an
    empty query returns the whole catalog, which is how you audit what is in
    it. Returns each item's serving description, kcal and macros, plus the
    provenance of those numbers.

    Always call this BEFORE estimating anything the user ate. On a miss, you may
    look the product up online and offer to add it with food_catalog_add — do
    not silently fall back to nutrition facts you remember.

    Every result carries: `source` (%s), `source_detail` (the URL, or what the
    photo showed, or the reasoning behind an estimate), `confirmed` (1 only if
    the user looked at these numbers and agreed), and `verified_at`. When
    you report a number from here, report where it came from.
    """ % _TIER_HELP
    conn = ctx.read_only()
    try:
        items = fd.search(conn, query)
    except sqlite3.OperationalError as e:
        return _empty_catalog(e)
    finally:
        conn.close()
    return {"items": items, "count": len(items)}


@tool
def food_catalog_add(ctx: VaultContext, item_key: str, display_name: str, serving_desc: str,
                     kcal: float, source: str, source_detail: str,
                     brand: str = "", aliases: str = "",
                     serving_g: float | None = None,
                     protein_g: float | None = None,
                     carb_g: float | None = None, fat_g: float | None = None,
                     confirmed: int = 0, notes: str = "") -> dict:
    """Add or update one item in the user's food catalog.

    THIS REPLACES THE WHOLE ROW. Every field is written from the arguments and
    anything you omit becomes empty — it does NOT keep its previous value. So
    supply the complete set of numbers on every write. This is deliberate: it
    is what stops a row being relabelled from 'estimate' to 'web' while its
    guessed macros quietly survive underneath the better label.

    `item_key` is a stable lowercase slug with no spaces, e.g.
    'tj-strained-greek-yogurt-plain'. `serving_desc` is the label's own serving
    ('3/4 cup', '2 tortillas', '4 oz raw') and every macro is PER THAT SERVING.
    `aliases` is pipe-separated and is what makes a casual mention findable:
    'greek yogurt|strained yogurt'.

    `source` is required and is one of — %s

    `source_detail` is required and must be specific enough that someone could
    re-check the number later: the URL, what the photo showed and when, or the
    reasoning behind an estimate. An entry nobody can audit does not get to
    exist, so a vague detail is worse than no entry.

    Set `confirmed=1` ONLY after you have shown the user the actual numbers and
    they agreed to them. The normal flow is: search the catalog, miss, look the
    product up online, show them what you found WITH the source, and add it
    once they confirm. If you could not find it, say so plainly and offer an
    'estimate' row with your reasoning written into source_detail — never
    present a guess as a lookup.
    """ % _TIER_HELP
    conn = ctx.connect()
    try:
        db.init_db(conn)
        row = fd.add(conn, item_key, display_name, serving_desc=serving_desc,
                     kcal=kcal, source=source, source_detail=source_detail,
                     brand=brand.strip() or None, aliases=aliases.strip() or None,
                     serving_g=serving_g, protein_g=protein_g, carb_g=carb_g,
                     fat_g=fat_g, confirmed=confirmed,
                     notes=notes.strip() or None)
    except (ValueError, TypeError, sqlite3.Error) as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()
    return {"ok": True, "stored": row}


@tool
def food_meal_total(ctx: VaultContext, items: list[dict]) -> dict:
    """Total a meal from catalog items. USE THIS RATHER THAN DOING THE
    ARITHMETIC YOURSELF — deriving numbers is not your job on this project.

    `items` is a list of {"item_key": str, "servings": number}, where servings
    is in units of that item's `serving_desc`. Six mini tortillas whose serving
    is '2 tortillas' means servings=3.

    Returns summed kcal and macros, a per-item breakdown, and two fields that
    matter more than the total:

      `weakest_source`  the worst provenance tier in the meal. This is the
                        headline. A meal containing one 'estimate' item is an
                        estimate, however precise the other rows are.
      `incomplete`      macros that could not be totalled because at least one
                        item is missing them. Those come back as null rather
                        than as a partial sum, because a partial sum reads as a
                        complete number and is not one.

    An unknown item_key is an error, not a skip — a total quietly missing a
    food is worse than no total. Look the item up and add it first.
    """
    try:
        pairs = []
        for it in items:
            if not isinstance(it, dict) or "item_key" not in it or "servings" not in it:
                raise ValueError(
                    'each item must be {"item_key": str, "servings": number}, '
                    f"got {it!r}")
            pairs.append((it["item_key"], it["servings"]))
    except (ValueError, TypeError) as e:
        return {"ok": False, "error": str(e)}

    conn = ctx.read_only()
    try:
        out = fd.totals(conn, pairs)
    except (ValueError, TypeError, sqlite3.Error) as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()
    return {"ok": True, **out}


@tool
def get_weekly_series(ctx: VaultContext, metric: str, start: str, end: str) -> dict:
    """Weekly means of a metric, each with its day count and its noise floor.

    Use this for any weekly figure. The reporting rules mandate weekly means —
    resting heart rate and VO2 max are never to be quoted as daily lows — and a
    weekly mean worked out by hand is exactly where a figure goes wrong: one
    that silently dropped its highest day became a published number that does
    not reproduce.

    `get_weekly_series` publishes each row's inclusive Monday-Sunday `period`
    as `YYYY-MM-DD:YYYY-MM-DD`; copy that exact string into a claim and never
    invent a period from `week_start`.

    Each row carries:
      period    the inclusive Monday-Sunday range, ``YYYY-MM-DD:YYYY-MM-DD``;
                copy it verbatim into a claim
      mean      the week's mean
      n_days    how many days it rests on. A mean of two days and a mean of
                seven are not the same claim. Say which you have.
      mdc95     the smallest week-to-week change that is NOT noise, given this
                metric's own day-to-day variability and autocorrelation.

    **Do not report a change smaller than mdc95 as a change.** The per-vault
    weekly resting-HR MDC is compared with the expected 0–3 bpm training effect
    reported by Reimers, Knapp & Reimers (2018, J Clin Med, DOI
    10.3390/jcm7120503). At weekly cadence that metric can miss a small effect,
    so the honest answer is "no change is measurable at this cadence", not a
    number. The returned `expected_training_effect_bpm` record carries this
    citation beside the literature figure.

    mdc95 is null when there is too little history to estimate a floor. A null
    floor means you cannot say whether a delta is real, not that it is.

    Metric ownership is per field: inherit a row's `metric` only for its own
    series-value leaves (`mean`, `median`, `min`, `max`, `std`, `latest`,
    `sum`, `recent_avg`, `baseline_avg`, `delta_pct`, `slope_per_week`) or a
    leaf whose field is exactly that metric; never inherit it for context
    fields such as `n_days`, `rho`, `sd_day`, `mdc95`, `unit`, dates, day
    counts, or other siblings.
    """
    if err := _bad_dates(start=start, end=end):
        return {"error": err}
    conn = ctx.read_only()
    try:
        if not A.mx.metric_exists(conn, metric):
            return {"error": f"no data for metric {metric!r}"}
        weeks = A.weekly_series(conn, metric, start, end)
        floor = A.metric_noise_floor(conn, metric, end)
    finally:
        conn.close()
    for row in weeks:
        _add_presentation(row, metric, row.get("period"), row.get("mean"),
                          field="mean")
    return {"metric": metric, "start": start, "end": end,
            "weeks": weeks, "count": len(weeks), "noise_floor": floor,
            "expected_training_effect_bpm": _literature_figure(
                "expected_training_effect_bpm")}


@tool
def get_block_structure(ctx: VaultContext, day: str) -> dict:
    """How long the user ran CONTINUOUSLY, and whether it was easy enough to count.

    This is the dial the impact ramp progresses on: the longest continuous jog
    block at or below the vault-configured block-mean heart-rate ceiling.
    Weekly jog minutes is a consequence of session
    shape, not a target — six two-minute reps and one twelve-minute rep are the
    same number of minutes and are not the same training.

    Returns per running session and for the day:
      longest_block_min      longest continuous block, bridge rule applied
      qualified_block_min    the same block only if its mean HR is at or below
                             the configured ceiling, else null. Null means
                             "nothing ran easy enough to count
                             toward the ramp" — it does NOT mean they did not run.
      unbridged_min          the block without the bridge rule, for comparison
      avg_hr_longest_block   mean HR over the longest block. This is one of
                             THREE different "average heart rates" and they run
                             up to 20 bpm apart: this one, avg_hr_all_jog (mean
                             over every jog bucket) and avg_hr_session (the
                             whole session including prescribed walk breaks).
                             Always say which you are quoting.
      reps                   every block in the session, longest first

    The bridge rule: up to two consecutive 16-18 min/mi buckets may join two jog
    segments when heart rate confirms the user was still running (>= 130 bpm)
    and no recording time is missing. Without it, one GPS wobble splits a run in
    two — a 9.3-minute continuous jog gets reported as 4.0.

    A treadmill session returns zeros: GymKit records no per-sample distance, so
    there are no buckets to chain. That is a known blind spot, not a rest day.
    """
    if err := _bad_dates(day=day):
        return {"error": err}
    conn = ctx.read_only()
    try:
        active = db.workout_mark_condition(conn, "w")
        marked = db.workout_mark_condition(conn, "w", marked=True)
        rows = conn.execute(
            "SELECT w.id, w.workout_type, w.start_utc, w.end_utc, w.duration_min, "
            "w.avg_heart_rate FROM workouts AS w WHERE w.local_date = ? "
            "AND w.workout_type IN ('running', 'walking', 'hiking') "
            f"AND {active} ORDER BY w.start_utc", (day,)).fetchall()
        excluded = conn.execute(
            f"SELECT COUNT(*) FROM workouts AS w WHERE w.local_date = ? "
            f"AND w.workout_type IN ('running', 'walking', 'hiking') AND {marked}",
            (day,)).fetchone()[0]
        sessions = []
        for w in rows:
            block = A.longest_block(conn, w["start_utc"], w["end_utc"])
            if not block["reps"] and w["workout_type"] != "running":
                continue          # a walk with no jog block is not a session here
            hr = A.mx.session_hr_figures(conn, w["start_utc"], w["end_utc"],
                                         w["avg_heart_rate"])
            sessions.append({
                "workout_type": w["workout_type"],
                "duration_min": A.mx.r(w["duration_min"], 1),
                "longest_block_min": block["bridged_min"],
                "qualified_block_min": block["qualified_min"],
                "unbridged_min": block["unbridged_min"],
                "avg_hr_longest_block": block["avg_hr_longest_block"],
                "avg_hr_all_jog": hr["avg_hr_all_jog"],
                "avg_hr_session": hr["avg_hr_session"],
                "reps": block["reps"],
            })
        hr_ceiling = A.mx.block_qualify_hr_max(conn)
    finally:
        conn.close()
    if not sessions:
        return {"day": day, "sessions": [], "longest_block_min": 0.0,
                "excluded_count": excluded,
                "qualified_block_min": None,
                "note": "no session with measurable block structure on this day"}
    best = max(sessions, key=lambda s: s["longest_block_min"])
    qualified = [s["qualified_block_min"] for s in sessions
                 if s["qualified_block_min"] is not None]
    return {
        "day": day, "sessions": sessions, "excluded_count": excluded,
        # Across a multi-session day the dial is the LONGEST block, not the sum:
        # the question it answers is "how long can they run continuously".
        "longest_block_min": best["longest_block_min"],
        "qualified_block_min": max(qualified) if qualified else None,
        "hr_ceiling_for_qualifying": hr_ceiling,
    }


@tool
def get_weekly_readiness(ctx: VaultContext, as_of: str = "") -> dict:
    """Readiness as a WEEK, plus continuity — the two numbers that survived review.

    The daily 0-100 composite was retired. Over 53 scored days it
    produced 40 amber, 13 green and red NEVER; red would have needed a 3-day
    mean resting HR of 77 against a 60 baseline, and both reachable bands
    licensed the same session. A number that cannot reach a third of its range,
    whose two attainable values imply the same action, was not informing a
    decision — so it is read weekly now, where seven days of averaging make it
    mean something.

    Returns:
      score          mean composite over the 7 days ending at `as_of`, or null
      n_days         how many of those days were scoreable — read this first
      components     each component's weekly mean subscore
      trend          week-over-week sentence, or null. REFUSES to compare across
                     2026-07-31, when SUBSCORE_K was halved:
                     the two sides are different instruments, and all four
                     mornings the brief said "red" recompute as amber.
      alert          the one thing worth interrupting a day for, or null. Fires
                     only when a component has been across its threshold on TWO
                     CONSECUTIVE days. Null is the expected answer.
      continuity     days with measurable movement, week to date. The
                     floor is an ACTION now — "a 15-minute walk, or 15 minutes
                     of deliberate movement at home" — and only the walk half
                     leaves a trace, so `unmeasured` days are days the watch
                     could not see, NOT missed days. Never report one as a miss.

    There is no band and no cue on purpose. Re-attaching a label to the weekly
    mean would put back the thing that was removed.
    """
    conn = ctx.read_only()
    try:
        as_of = as_of or A._as_of(conn, None)
        out = A.weekly_readiness(conn, as_of)
        out["trend"] = A.weekly_readiness_trend(conn, as_of)
        out["alert"] = A.readiness_alert(conn, as_of)
        days = A.movement_floor_days(conn, out["week_start"], out["week_end"])
        out["continuity"] = {
            "active_days": sum(1 for d in days if d["active"]),
            "days": len(days),
            "unmeasured": [d["date"] for d in days if not d["active"]],
            "walk_minutes_floor": A.FLOOR_WALK_MINUTES,
        }
    finally:
        conn.close()
    return out


@tool
def record_benchmark(ctx: VaultContext, date: str, stage: int, pace: str,
                     median_hr_last_two_min: float | None = None,
                     talk_test: str = "", temp_c: float | None = None,
                     dew_point_c: float | None = None, notes: str = "",
                     stage_start_utc: str = "", stage_end_utc: str = "") -> dict:
    """Store one stage of the monthly treadmill benchmark.

    Protocol: 4 x 4 min at 15:00 / 14:00 / 13:00 / 12:00 min/mi, the median HR
    of each stage's LAST TWO MINUTES, a spoken sentence per stage, stop at HR
    170 or loss of speech.
    Call once per completed stage. A run stopped early stores the stages that
    were completed and nothing for the rest — never a zero.

    `median_hr_last_two_min` is a FALLBACK only. Whenever heart-rate records
    cover the stage, Python recomputes the median and ignores what you pass,
    because a typed number and a measured one are not the same evidence. The
    stored `median_source` says which happened:
      records:explicit  you gave stage_start_utc, so the window is measured
      records:protocol  the window was INFERRED from the published stage
                        structure (8 min warmup, then 4-on/2-off). A session
                        that ran to a different shape yields a plausible wrong
                        median — pass stage_start_utc when you know it.
      typed             no records covered the stage; the number is yours.
    """
    if err := _bad_dates(date=date):
        return {"ok": False, "error": err}
    conn = ctx.connect()
    try:
        db.init_db(conn)
        benchmark.record(conn, date=date, stage=stage, pace=pace,
                         median_hr_last_two_min=median_hr_last_two_min,
                         talk_test=talk_test or None, temp_c=temp_c,
                         dew_point_c=dew_point_c, notes=notes or None,
                         stage_start_utc=stage_start_utc or None,
                         stage_end_utc=stage_end_utc or None)
        rows = [r for r in benchmark.series(conn) if r["date"] == date]
    finally:
        conn.close()
    return {"ok": True, "date": date, "stages": rows}


@tool
def get_benchmark_series(ctx: VaultContext) -> dict:
    """Every benchmark stage ever recorded, in date/stage order.

    This is the instrument to reach for because uncontrolled weekly wearable
    data cannot resolve training adaptation at this horizon: on measurement, 5
    of 7 tracked metrics could not detect change, and heat alone (~1 bpm/°C) is
    larger than the training signal (Pandolf KB, Cafarelli E, Noble BJ & Metz
    KF, 1975, Arch Phys Med Rehabil, PMID 1200826). A treadmill holds pace and
    grade fixed, so HR at a fixed pace is finally comparable month to month.

    The returned `heat_effect_bpm_per_c` record carries that citation beside
    the literature figure.

    Compare stages ACROSS dates, never stages within one date — the four paces
    are four different efforts. Read `median_source` before comparing: a
    "records:protocol" median came from an inferred window.
    """
    conn = ctx.read_only()
    try:
        rows = benchmark.series(conn)
    finally:
        conn.close()
    return {"stages": rows, "count": len(rows),
            "dates": sorted({r["date"] for r in rows}),
            "heat_effect_bpm_per_c": _literature_figure(
                "heat_effect_bpm_per_c")}


@tool
def get_monthly_running_power(ctx: VaultContext, month: str) -> dict:
    """Mean running power at matched pace for a month, or null.

    `month` is "YYYY-MM". Watts are the one instrument here that can say
    whether easy running is getting CHEAPER independent of HR lag, heat and
    drift — the three confounders that make speed-at-HR unusable over this
    dataset. Read it monthly, beside the benchmark, and nowhere else.

    Restricted to buckets at least 3 minutes into a continuous block, because
    any speed-at-HR comparison over this dataset is confounded by HR lag
    otherwise, and to outdoor route-backed runs, because Apple
    Watch reports running power OUTDOORS ONLY — it cannot come from the
    treadmill benchmark itself.

    Returns null when the month holds fewer than two qualifying outdoor runs.
    That is a refusal, not a gap: one run is a session, not a month.
    """
    conn = ctx.read_only()
    try:
        watts = RF.monthly_running_power(conn, month)
    finally:
        conn.close()
    return {"month": month, "mean_power_w": watts,
            "pace_band_min_per_mi": list(RF.REFERENCE_PACE_BAND),
            "note": (None if watts is not None else
                     "fewer than two qualifying outdoor runs this month")}


@tool
def log_manual_jog_minutes(ctx: VaultContext, day: str, jog_minutes: float, source_note: str,
                           why: str) -> dict:
    """Record jog minutes for a session the watch could not measure.

    Use this ONLY when the minutes genuinely happened and the data cannot show
    them. The standard case is a treadmill session, where GymKit produces no
    per-sample distance so `impact_volume` scores 0.0 for a run that really
    happened. Any indoor or unmeasured session has the same problem.

    This is the only tool here that can manufacture training volume. Three
    guards, all enforced:
      - A manual value NEVER overwrites a measured one. If the day already has
        real distance samples, this entry is stored and ignored.
      - Every consumer that uses a manual value says so: impact_volume returns
        `jog_minutes_source` = manual / partly_manual / measured.
      - Every entry is stored in its own table and stays listable, so a wrong
        one is findable rather than permanent.

    `why` is REQUIRED and stored. Do not invent minutes to fill a gap, and do
    not estimate from a prescription — if what they actually did is not recorded
    somewhere, say so and ask them rather than entering a number.

    Upserts on (day, source_note): a correction replaces, never accumulates.
    """
    if err := _bad_dates(day=day):
        return {"ok": False, "error": err}
    if day > date.today().isoformat():
        return {"ok": False, "error": f"day={day!r} is in the future"}
    conn = ctx.connect()
    try:
        db.init_db(conn)
        db.log_manual_jog(conn, day, jog_minutes=jog_minutes,
                          source_note=source_note, why=why)
        rows = A.impact_volume(conn, day, day, by="day")
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()
    return {"ok": True, "day": day, "day_totals": rows}


def main(argv: "list[str] | None" = None) -> int:
    """Serve one session over stdio. The vault arrives on argv, not from the
    environment: an env var is ambient, and ambient is how a worker ends up
    serving the vault it happened to inherit."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m health_advisor.mcp_server")
    ap.add_argument("--vault", "--db", dest="vault", required=True,
                    help="path to the vault this session serves")
    ap.add_argument("--user", default="local", help="user id this vault belongs to")
    args = ap.parse_args(argv)
    ctx = VaultContext.local(args.vault, user_id=args.user, writable=True)
    build_server(ctx).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
