"""Binding an ask's as-of to the tools changes nothing for production
(downstream issue #596).

Production asks without an as-of. The golden below was captured from the
engine BEFORE the as-of binding existed (commit 3800826): every coach tool,
called with a fixed argument list on the audit vault, its JSON hashed byte for
byte. A session without an as-of must still produce exactly those bytes.

The wall clock is frozen at :data:`WALL_CLOCK` for the capture and for every
comparison, because two tools take a default window end from it
(``get_sleep_regularity``'s ``end``, ``get_training_load_detail``'s ``end`` and
the ACWR computed at it). Freezing it is what makes a golden possible at all;
the code path that reads the clock is still the one under test.

Regenerate only from the pre-change engine:
``HA_REGEN_AS_OF_GOLDEN=1 python -m pytest tests/test_as_of_byte_identity.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date
from pathlib import Path

import pytest

from health_advisor import demo
from health_advisor import mcp_server as S
from health_advisor.context import VaultContext

END = demo.DEFAULT_END_DATE          # the vault's last day of data
VAULT_DAYS = 60
METRIC = "resting_heart_rate"
WALL_CLOCK = date(2026, 9, 15)       # a frozen "today", after the data ends
GOLDEN = Path(__file__).parent / "fixtures" / "as_of_byte_identity_3800826.json"

# (call id, tool, kwargs). The default-argument call is the one the audit makes
# (only required arguments, dates a model told "today is END" would send);
# where a tool takes a window, one explicit-window call follows it.
CALLS: list[tuple[str, str, dict]] = [
    ("list_available_metrics", "list_available_metrics", {}),
    ("get_daily_series", "get_daily_series", {"metric": METRIC}),
    ("get_daily_series/window", "get_daily_series",
     {"metric": METRIC, "start": "2026-07-20", "end": "2026-08-20"}),
    ("summarize_metric", "summarize_metric", {"metric": METRIC}),
    ("summarize_metric/window", "summarize_metric",
     {"metric": METRIC, "period": "2026-07-20:2026-08-20"}),
    ("compare_periods", "compare_periods",
     {"metric": METRIC, "period_a": "7d", "period_b": "28d"}),
    ("compare_periods/window", "compare_periods",
     {"metric": METRIC, "period_a": "2026-08-07:2026-08-20",
      "period_b": "2026-07-24:2026-08-06"}),
    ("get_intraday", "get_intraday", {"metric": "heart_rate", "day": END}),
    ("get_hr_zones", "get_hr_zones", {"day": END}),
    ("list_workouts", "list_workouts", {}),
    ("list_workouts/window", "list_workouts",
     {"start": "2026-07-20", "end": "2026-08-20"}),
    ("get_workout_segments", "get_workout_segments", {"day": END}),
    ("get_impact_volume", "get_impact_volume",
     {"start": "2026-08-04", "end": END}),
    ("get_impact_volume/window", "get_impact_volume",
     {"start": "2026-08-01", "end": "2026-08-20", "by": "day"}),
    ("get_sleep_regularity", "get_sleep_regularity", {}),
    ("get_sleep_regularity/window", "get_sleep_regularity",
     {"start": "2026-07-10", "end": "2026-08-20"}),
    ("get_training_load_detail", "get_training_load_detail", {}),
    ("get_training_load_detail/window", "get_training_load_detail",
     {"start": "2026-07-20", "end": "2026-08-20"}),
    ("get_run_form", "get_run_form", {}),
    ("get_run_form/window", "get_run_form",
     {"start": "2026-07-20", "end": "2026-08-20"}),
    ("get_briefing", "get_briefing", {}),
    ("get_briefing/window", "get_briefing", {"day": "2026-08-20"}),
    ("get_latest", "get_latest", {"metric": METRIC}),
    ("correlate_metrics", "correlate_metrics",
     {"metric_x": METRIC, "metric_y": "step_count"}),
    ("scan_correlations", "scan_correlations", {"target": METRIC}),
    ("get_subjective", "get_subjective",
     {"start_date": "2026-08-04", "end_date": END}),
    ("food_lookup", "food_lookup", {}),
    ("food_meal_total", "food_meal_total", {"items": []}),
    ("get_weekly_series", "get_weekly_series",
     {"metric": METRIC, "start": "2026-07-07", "end": END}),
    ("get_block_comparison", "get_block_comparison",
     {"metric": METRIC, "block_weeks": 2, "as_of": END}),
    ("get_block_structure", "get_block_structure", {"day": END}),
    ("get_weekly_readiness", "get_weekly_readiness", {}),
    ("get_weekly_readiness/window", "get_weekly_readiness",
     {"as_of": "2026-08-20"}),
    ("get_benchmark_series", "get_benchmark_series", {}),
    ("get_monthly_running_power", "get_monthly_running_power",
     {"month": END[:7]}),
]


def canonical(result) -> str:
    """The bytes compared. Keys are sorted because the engine's own key order
    is not stable across processes: ``presentations`` is filled from a set, so
    its order follows the interpreter's hash seed. Every value, every key and
    every float's repr are still compared exactly."""
    return json.dumps(result, ensure_ascii=False, sort_keys=True, default=repr)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def capture(ctx: VaultContext) -> dict[str, str]:
    """``{call id: canonical JSON}`` for every call in :data:`CALLS`."""
    tools = S.build_tools(ctx)
    return {cid: canonical(tools[name](**kwargs)) for cid, name, kwargs in CALLS}


@pytest.fixture(scope="module")
def built_vault(tmp_path_factory):
    path = tmp_path_factory.mktemp("as_of_identity") / "demo.db"
    demo.build_demo_vault(path, days=VAULT_DAYS, read_only=False)
    return path


@pytest.fixture
def session(built_vault, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_today", lambda local_timezone=None: WALL_CLOCK)
    copy = tmp_path / "vault.db"
    shutil.copy(built_vault, copy)
    return VaultContext.local(copy, user_id="test").provider_facing()


def test_every_call_is_a_coach_tool():
    from health_advisor import llm
    engine = {fn.__name__ for fn in S._TOOLS}
    coach_engine = {n for n in llm.COACH_TOOLS if n in engine}
    assert {name for _, name, _ in CALLS} == coach_engine


def test_no_as_of_is_byte_identical_to_the_pre_change_engine(session):
    got = {cid: {"sha256": digest(text), "bytes": len(text.encode("utf-8"))}
           for cid, text in capture(session).items()}
    if os.environ.get("HA_REGEN_AS_OF_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n",
                          encoding="utf-8")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(got) == set(golden)
    changed = sorted(cid for cid in got if got[cid] != golden[cid])
    assert not changed, f"no-as-of output moved for: {changed}"
