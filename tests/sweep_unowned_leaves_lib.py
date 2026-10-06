"""Measurement harness for the unowned-leaf sweep (see test_sweep_unowned_leaves).

Everything here is synthetic: the vault is the engine's own invented demo
vault plus a few invented rows, and every number a tool returns is derived
from them.

The OPPONENT of the publisher is :func:`oracle_leaves`: a plain JSON walk
that knows nothing of ``deepdive_verify._ledger_scopes`` ownership. A leaf is
"published" only when a fact built by the engine's own builders names that
leaf's exact path as its source.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from health_advisor import chat, deepdive_verify, demo, fact_template, llm, normalize
from health_advisor.context import VaultContext

DAYS = 120
WIN = ("2026-07-27", "2026-08-30")

# Tools the ask path binds (llm.COACH_TOOLS) that the engine itself implements,
# plus two read-only engine tools the ask path does NOT expose; the report
# keeps them apart so a gap in the second group is not mistaken for a live one.
ASK_EXPOSED_SKIP = ("analyst_query", "cite")
NOT_ASK_EXPOSED = ("get_workout_coverage", "get_ingest_diagnostics")


def build_vault(path: Path) -> tuple[VaultContext, tuple[str, str]]:
    """The demo vault, plus the rows the demo does not generate."""
    demo.build_demo_vault(path, days=DAYS)
    path.chmod(0o644)
    conn = sqlite3.connect(path)
    try:
        (run_day, wkey), (other_day, _) = conn.execute(
            "SELECT local_date, dedupe_key FROM workouts "
            "WHERE workout_type='running' ORDER BY local_date DESC LIMIT 2"
        ).fetchall()
        # Two rival partitions, a pause/resume pair, an auto-pause: the shapes
        # get_workout_segments distinguishes.
        events = [
            ("segment", "09:45:00", "09:55:00", 10.0),
            ("segment", "09:55:00", "10:10:00", 15.0),
            ("segment", "10:10:00", "10:25:00", 15.0),
            ("lap", "09:45:00", "10:05:00", 20.0),
            ("lap", "10:05:00", "10:25:00", 20.0),
            ("pause", "10:00:00", None, None),
            ("resume", "10:02:00", None, None),
            ("motion_paused", "10:12:00", "10:13:00", 1.0),
        ]
        for i, (kind, s, e, dur) in enumerate(events):
            conn.execute(
                "INSERT INTO workout_events (workout_key, event_type, "
                "start_utc, end_utc, duration_min, dedupe_key) "
                "VALUES (?,?,?,?,?,?)",
                (wkey, kind, f"{run_day}T{s}Z",
                 f"{run_day}T{e}Z" if e else None, dur, f"sweep|{i}"))
        for d, stage, pace, hr in (
                ("2026-08-03", 1, 15.0, 118.0), ("2026-08-03", 2, 14.0, 126.0),
                ("2026-08-03", 3, 13.0, 133.0), ("2026-08-17", 1, 15.0, 116.0),
                ("2026-08-17", 2, 14.0, 124.0), ("2026-08-17", 3, 13.0, 131.0)):
            conn.execute(
                "INSERT INTO benchmark (date, stage, pace_min_per_mi, "
                "median_hr_last_two_min, talk_test, temp_c, dew_point_c, "
                "notes, median_source) VALUES (?,?,?,?,?,?,?,?,?)",
                (d, stage, pace, hr, "full sentences", 21.5, 12.0, "", "typed"))
        for k, name, kcal in (("sweep-oats", "Sweep oats", 150.0),
                              ("sweep-yogurt", "Sweep yogurt", 110.0)):
            conn.execute(
                "INSERT INTO food_catalog (item_key, display_name, brand, "
                "aliases, serving_desc, serving_g, kcal, protein_g, carb_g, "
                "fat_g, source, source_detail, confirmed, verified_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (k, name, None, "", "1 cup", 100.0, kcal, 5.0, 20.0, 3.0,
                 "estimate", "invented for the sweep", 1, "2026-08-01"))
        conn.commit()
    finally:
        conn.close()
    return VaultContext.local(path, user_id="sweep", writable=False), (run_day, other_day)


def calls(days: tuple[str, str]) -> list[tuple[str, str, dict]]:
    """(tool, argument-shape label, arguments), in a stable order."""
    RUN_DAY, OTHER_DAY = days
    s, e = WIN
    out: list[tuple[str, str, dict]] = [
        ("list_available_metrics", "default", {}),
    ]
    for m in ("step_count", "resting_heart_rate", "vo2_max"):
        out += [
            ("get_daily_series", f"metric={m}", {"metric": m}),
            ("summarize_metric", f"metric={m}", {"metric": m}),
            ("get_weekly_series", f"metric={m} window", {"metric": m, "start": s, "end": e}),
            ("get_latest", f"metric={m}", {"metric": m}),
            ("get_block_comparison", f"metric={m} 2wk",
             {"metric": m, "block_weeks": 2, "as_of": e}),
        ]
    out += [
        ("get_daily_series", "metric=step_count window", {"metric": "step_count", "start": s, "end": e}),
        ("summarize_metric", "metric=step_count 7d", {"metric": "step_count", "period": "7d"}),
        ("summarize_metric", "metric=step_count range", {"metric": "step_count", "period": f"{s}:{e}"}),
        ("get_latest", "metric=heart_rate", {"metric": "heart_rate"}),
        ("compare_periods", "step_count 14d vs range",
         {"metric": "step_count", "period_a": "14d", "period_b": "2026-08-01:2026-08-14"}),
        ("compare_periods", "vo2_max 14d vs range",
         {"metric": "vo2_max", "period_a": "14d", "period_b": "2026-08-01:2026-08-14"}),
        ("get_intraday", "heart_rate day", {"metric": "heart_rate", "day": RUN_DAY}),
        ("get_intraday", "step_count day", {"metric": "step_count", "day": RUN_DAY}),
        ("get_hr_zones", "run day", {"day": RUN_DAY}),
        ("get_hr_zones", "run day type+thresholds",
         {"day": RUN_DAY, "workout_type": "running", "thresholds": "120,140,160"}),
        ("list_workouts", "default", {}),
        ("list_workouts", "window", {"start": s, "end": e}),
        ("list_workouts", "limit=3", {"limit": 3}),
        ("get_workout_segments", "day with events", {"day": RUN_DAY}),
        ("get_workout_segments", "day without events", {"day": OTHER_DAY}),
        ("get_impact_volume", "by week", {"start": s, "end": e}),
        ("get_impact_volume", "by day", {"start": s, "end": e, "by": "day"}),
        ("get_impact_volume", "by block", {"start": s, "end": e, "weeks_per_block": 2}),
        ("get_sleep_regularity", "default", {}),
        ("get_sleep_regularity", "window", {"start": s, "end": e}),
        ("get_training_load_detail", "default", {}),
        ("get_training_load_detail", "window", {"start": s, "end": e}),
        ("get_run_form", "latest session", {}),
        ("get_run_form", "trend window", {"start": s, "end": e}),
        ("get_briefing", "daily", {}),
        ("get_briefing", "scope=deep", {"scope": "deep"}),
        ("correlate_metrics", "sleep vs rhr",
         {"metric_x": "sleep_asleep", "metric_y": "resting_heart_rate"}),
        ("scan_correlations", "target=rhr", {"target": "resting_heart_rate"}),
        ("get_subjective", "window", {"start_date": s, "end_date": e}),
        ("food_lookup", "default", {}),
        ("food_lookup", "query", {"query": "oats"}),
        ("food_meal_total", "two items",
         {"items": [{"item_key": "sweep-oats", "servings": 1.5},
                    {"item_key": "sweep-yogurt", "servings": 1}]}),
        ("get_block_structure", "run day", {"day": RUN_DAY}),
        ("get_weekly_readiness", "default", {"as_of": e}),
        ("get_benchmark_series", "default", {}),
        ("get_benchmark_series", "runs_recorded=4", {"runs_recorded": 4}),
        ("get_monthly_running_power", "month (stubbed watts)", {"month": "2026-08"}),
        ("get_workout_coverage", "window", {"start": s, "end": e}),
        ("get_ingest_diagnostics", "step_count window",
         {"metric": "step_count", "start": s, "end": e}),
    ]
    return out


def base_registry(ctx: VaultContext) -> dict:
    """The ask path's own tool registry (provider-facing, COACH_TOOLS)."""
    names = [n for n in llm.COACH_TOOLS
             if n not in ASK_EXPOSED_SKIP and n not in llm.HOST_SUPPLIED_TOOLS]
    names += list(NOT_ASK_EXPOSED)
    return llm._registry(ctx, include=names)


def run_one(registry: dict, tool: str, args: dict, ledger_path: Path) -> list[dict]:
    """Call one tool through the ask path's own ledger recorder; return the
    ledger exactly as ``chat._read_ledger`` reads it."""
    ledger_path.unlink(missing_ok=True)
    llm._ledgered({tool: registry[tool]}, ledger_path)[tool][0](**args)
    return chat._read_ledger(str(ledger_path))


# --------------------------------------------------------------------------- #
# The oracle: a plain JSON walk, deliberately ignorant of metric ownership.
# --------------------------------------------------------------------------- #
def _render(path: tuple) -> str:
    out = "$.result"
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def pattern(path: str) -> str:
    """A leaf path with list indexes and date-like keys made generic."""
    path = re.sub(r"\[\d+\]", "[]", path)
    return re.sub(r"\.\d{4}-\d{2}-\d{2}(?=\.|$)", ".<date>", path)


def oracle_leaves(result) -> list[dict]:
    """Every numeric leaf of a tool result, with whether a presentation sits
    beside it (a ``presentation`` sibling, or ``presentations[<leaf name>]``)."""
    out: list[dict] = []

    def walk(node, path, parent):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + (k,), node)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + (i,), node)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            leaf = path[-1] if path else ""
            single = parent.get("presentation") if isinstance(parent, dict) else None
            plural = parent.get("presentations") if isinstance(parent, dict) else None
            # A singular ``presentation`` is the row's headline rendering: it
            # belongs to ``value`` or to the leaf named by its own metric, not
            # to every sibling.
            has_pres = bool(
                (isinstance(single, dict)
                 and (leaf == "value" or leaf == single.get("metric")))
                or (isinstance(plural, dict) and leaf in plural))
            out.append({"path": _render(path), "pattern": pattern(_render(path)),
                        "leaf": str(leaf), "value": node,
                        "has_presentation": has_pres})

    walk(result, (), None)
    return out


def fact_builders() -> dict:
    return {
        "metric": fact_template.build_fact_set,
        "attachment": fact_template.build_attachment_facts,
        "workout": fact_template.build_workout_facts,
        "briefing": fact_template.build_briefing_facts,
        "declared": fact_template.build_declared_facts,
        "citation": fact_template.build_citation_facts,
    }


def fact_sources(ledger: list[dict]) -> tuple[dict[str, dict[str, str]], dict]:
    """({builder: {source path: key}}, union of every published fact)."""
    out: dict[str, dict[str, str]] = {}
    union: dict[str, dict] = {}
    for name, build in fact_builders().items():
        paths: dict[str, str] = {}
        facts = build(ledger)
        union.update(facts)
        for key, fact in facts.items():
            src = fact.get("source") or {}
            for p in ([src["path"]] if src.get("path") else []) + list(src.get("paths") or []):
                paths[p] = key
        out[name] = paths
    return out, union


def gate_seen_without_slot(ledger: list[dict], published_paths: set[str]) -> list[str]:
    """Metrics for which the denial gate sees a value but none of the leaves it
    sees is a published slot (the #576 shape).

    ``chat._question_metric`` offers every catalogue metric; the gate
    (``_ledger_has_asked_metric_value``) then asks the ledger. A metric is
    reported only when the gate answers yes and no leaf it counts as that
    metric's value is the source of any published fact.
    """
    out = []
    for metric in normalize.known_metrics():
        if not chat._ledger_has_asked_metric_value(ledger, metric):
            continue
        visible = [
            e["path"] for record in ledger
            for e in deepdive_verify._ledger_scopes(record)
            if e.get("kind") == "result" and e.get("value") is not None
            and (e.get("metric") == metric
                 or (_metricless(e.get("metric")) and e.get("field") == metric))]
        if not any(p in published_paths for p in visible):
            out.append(metric)
    return sorted(out)


def _metricless(metric) -> bool:
    return deepdive_verify._is_metricless_metric(metric)


def owned_entries(ledger: list[dict]) -> dict[str, dict]:
    """{path: engine scope entry} for leaves the engine gives a metric
    (diagnosis and duplicate resolution only; never part of the oracle)."""
    return {e["path"]: e for record in ledger
            for e in deepdive_verify._ledger_scopes(record)
            if e.get("kind") == "result" and not _metricless(e.get("metric"))}


def _same_figure_published(leaf: dict, facts: dict[str, dict]) -> bool:
    """A leaf whose figure another path already publishes under its own name.

    ``_publish_unambiguous`` keeps the first of several agreeing candidates,
    and a tool may restate one figure at two paths (a day's longest block at
    the top level and again per session), so a later source is unpublished by
    path yet citable. The test is deliberately narrow: same number AND the
    published fact's field, or its source path's last segment, is this leaf's
    own name.
    """
    for fact in facts.values():
        if fact.get("value") != leaf["value"] or isinstance(fact["value"], str):
            continue
        src = fact.get("source") or {}
        tail = str(src.get("path", "")).rsplit(".", 1)[-1]
        if leaf["leaf"] in (fact.get("field"), tail):
            return True
    return False


def analyse(ledger: list[dict]) -> dict:
    """Oracle vs publisher for a one-call ledger."""
    record = ledger[0]
    leaves = [] if record.get("result_elided") else oracle_leaves(record["result"])
    sources, union = fact_sources(ledger)
    published = {p for paths in sources.values() for p in paths}
    owned = owned_entries(ledger)
    for leaf in leaves:
        leaf["owned"] = leaf["path"] in owned
        leaf["published"] = (leaf["path"] in published
                             or _same_figure_published(leaf, union))
        leaf["published_by_metric_builder"] = leaf["path"] in sources["metric"]
    return {"leaves": leaves, "facts": union,
            "unpublished": [l for l in leaves if not l["published"]],
            "gate_no_slot": gate_seen_without_slot(ledger, published),
            "elided": bool(record.get("result_elided")),
            "bytes": record.get("result_bytes")}
