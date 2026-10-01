"""Every tool result must survive json.dumps (#563).

`llm._encode_tool_result` refuses to stringify what json cannot represent, so a
tool that returns a `date` is announced as failed and the model never sees its
figures. That is silent on every other path (the MCP transport coerces), which
is how `get_block_comparison` carried raw `date` objects on its
insufficient-coverage branch without a single test noticing.

The sweep runs every tool the engine registers against a synthetic vault with
valid minimal arguments. A new tool must be added to `SWEEP_ARGS`, so the
surface cannot grow without being encoded at least once.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from health_advisor import demo, llm
from health_advisor import mcp_server as S
from health_advisor.context import VaultContext

END = demo.DEFAULT_END_DATE
START = "2026-08-01"
DAY = "2026-08-30"
METRIC = "resting_heart_rate"

# Minimal valid arguments per tool, on a vault ending at `END`.
SWEEP_ARGS: dict[str, dict] = {
    "mark_workout_not_a_session": {"workout_key": "none|none|none|none"},
    "list_available_metrics": {},
    "get_daily_series": {"metric": METRIC, "start": START, "end": END},
    "summarize_metric": {"metric": METRIC, "period": "30d"},
    "compare_periods": {"metric": METRIC, "period_a": "2026-08-01:2026-08-15",
                        "period_b": "2026-08-16:2026-08-30"},
    "get_intraday": {"metric": "heart_rate", "day": DAY},
    "get_hr_zones": {"day": DAY},
    "list_workouts": {"start": START, "end": END},
    "get_workout_coverage": {"start": START, "end": END},
    "get_workout_segments": {"day": DAY},
    "get_impact_volume": {"start": "2026-06-01", "end": END},
    "get_sleep_regularity": {"start": START, "end": END},
    "get_training_load_detail": {"start": START, "end": END},
    "get_run_form": {"start": START, "end": END},
    "get_briefing": {"scope": "daily", "day": DAY},
    "get_latest": {"metric": METRIC},
    "correlate_metrics": {"metric_x": METRIC, "metric_y": "step_count",
                          "period": "60d"},
    "scan_correlations": {"target": METRIC, "period": "60d"},
    "write_insight": {"day": DAY, "text": "sweep"},
    "log_subjective": {"day": DAY, "stress": 3},
    "get_subjective": {"start_date": START, "end_date": END},
    "food_lookup": {"query": "egg"},
    "food_catalog_add": {"item_key": "sweep_item", "display_name": "Sweep item",
                         "serving_desc": "1 each", "kcal": 100.0,
                         "source": "estimate", "source_detail": "sweep"},
    "food_meal_total": {"items": [{"item_key": "sweep_item", "servings": 1}]},
    "get_ingest_diagnostics": {"metric": METRIC, "start": START, "end": END},
    "get_weekly_series": {"metric": METRIC, "start": "2026-06-01", "end": END},
    "get_block_comparison": {"metric": METRIC, "block_weeks": 4, "as_of": END},
    "get_block_structure": {"day": DAY},
    "get_weekly_readiness": {"as_of": END},
    "record_benchmark": {"date": DAY, "stage": 1, "pace": "12:00"},
    "get_benchmark_series": {},
    "get_monthly_running_power": {"month": "2026-08"},
    "log_manual_jog_minutes": {"day": DAY, "jog_minutes": 5.0,
                               "source_note": "sweep", "why": "sweep"},
}

# Calls beyond the one-per-tool table, for tools whose result takes more than
# one shape. Each is (tool, args, why).
EXTRA_CALLS = [
    ("get_block_comparison", {"metric": METRIC, "block_weeks": 52, "as_of": END},
     "insufficient_coverage branch"),
    ("get_block_comparison", {"metric": "no_such_metric", "block_weeks": 4,
                              "as_of": END}, "unknown metric"),
    ("get_block_comparison", {"metric": METRIC, "block_weeks": 4,
                              "as_of": "2020-01-01"}, "window before the data"),
    ("get_block_comparison", {"metric": "body_fat_percentage", "block_weeks": 4,
                              "as_of": END}, "derived metric"),
]


# A full vault, and a thin one where every four-week block is below the
# coverage floor, so the refusal branches of the tools are swept too.
VAULT_DAYS = {"full": 120, "thin": 20}


@pytest.fixture(scope="module", params=sorted(VAULT_DAYS))
def swept_vault(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("sweep") / "demo.db"
    demo.build_demo_vault(path, days=VAULT_DAYS[request.param], read_only=False)
    return path


@pytest.fixture
def swept_tools(swept_vault, tmp_path):
    """Tools bound to a writable copy, so the write tools can run too."""
    import shutil
    copy = tmp_path / "copy.db"
    shutil.copy(swept_vault, copy)
    ctx = VaultContext.local(copy, user_id="sweep", writable=True)
    return SimpleNamespace(**S.build_tools(ctx))


def test_sweep_table_names_every_registered_tool():
    registered = {fn.__name__ for fn in S._TOOLS}
    assert set(SWEEP_ARGS) == registered, (
        f"missing: {sorted(registered - set(SWEEP_ARGS))}, "
        f"stale: {sorted(set(SWEEP_ARGS) - registered)}")


def test_the_coach_surface_is_inside_the_sweep():
    engine = {fn.__name__ for fn in S._TOOLS}
    coach_engine = (set(llm.COACH_TOOLS) - set(llm.HOST_SUPPLIED_TOOLS)
                    - {llm.ANALYST_QUERY_NAME, llm.EVIDENCE_CITE_NAME})
    assert coach_engine <= engine
    assert coach_engine <= set(SWEEP_ARGS)


def test_every_registered_tool_result_is_json_encodable(swept_tools):
    failures = {}
    for name in sorted(SWEEP_ARGS):
        result = getattr(swept_tools, name)(**SWEEP_ARGS[name])
        try:
            json.dumps(result)
        except (TypeError, ValueError) as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    assert not failures, failures


@pytest.mark.parametrize("name,args,why", EXTRA_CALLS,
                         ids=[c[2] for c in EXTRA_CALLS])
def test_extra_result_shapes_are_json_encodable(swept_tools, name, args, why):
    json.dumps(getattr(swept_tools, name)(**args))


def test_block_comparison_insufficient_coverage_is_encodable(swept_tools):
    """The branch that returned raw `date` objects (#563)."""
    result = swept_tools.get_block_comparison(
        METRIC, block_weeks=52, as_of=END)
    assert result["status"] == "insufficient_coverage"
    for block in result["blocks"].values():
        assert isinstance(block["period"], str)
    assert llm._encode_tool_result(result, "get_block_comparison") == json.dumps(
        result)
    assert "error" not in json.loads(
        llm._encode_tool_result(result, "get_block_comparison"))


def test_block_comparison_dates_are_iso_strings(swept_tools):
    result = swept_tools.get_block_comparison(
        METRIC, block_weeks=52, as_of=END)
    for block in result["blocks"].values():
        for key in ("window_start", "window_end"):
            assert isinstance(block[key], str)
            assert len(block[key]) == 10 and block[key][4] == "-"
