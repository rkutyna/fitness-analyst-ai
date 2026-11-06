"""A session's as-of lives in its VaultContext, and an explicit date after it is
clamped to it (downstream issue #596).

``VaultContext.at(day)`` returns a copy carrying the as-of, and every derived
context carries it forward. With an as-of set, an explicit date argument later
than it is replaced by the as-of: the result is the as-of window's result,
plus ``clamped_to_as_of: true`` and ``requested_dates`` naming what was asked
for. Without an as-of nothing is clamped.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest

from health_advisor import demo
from health_advisor import mcp_server as S
from health_advisor.context import WRITE, VaultContext

END = demo.DEFAULT_END_DATE
AS_OF = (date.fromisoformat(END) - timedelta(days=20)).isoformat()
LATE = (date.fromisoformat(END) - timedelta(days=6)).isoformat()
METRIC = "resting_heart_rate"


# --------------------------------------------------------------------------- #
# the context
# --------------------------------------------------------------------------- #
def test_at_returns_a_copy_carrying_the_as_of(tmp_path):
    ctx = VaultContext.local(tmp_path / "v.db")
    pinned = ctx.at("2026-08-11")
    assert pinned.as_of == "2026-08-11"
    assert ctx.as_of is None
    assert (pinned.user_id, pinned.db_path, pinned.capabilities) == \
        (ctx.user_id, ctx.db_path, ctx.capabilities)


@pytest.mark.parametrize("bad", [
    None, "", "2026-8-11", "2026-08-1", "2026-02-30", "2026-08-11T00:00",
    " 2026-08-11", "2026/08/11", "yesterday", 20260811, date(2026, 8, 11),
    "２０２６-08-11",
])
def test_at_rejects_anything_but_a_real_yyyy_mm_dd_day(tmp_path, bad):
    with pytest.raises(ValueError):
        VaultContext.local(tmp_path / "v.db").at(bad)


def test_the_constructor_validates_too(tmp_path):
    with pytest.raises(ValueError):
        VaultContext(user_id="u", db_path=tmp_path / "v.db", as_of="2026-13-01")


def test_every_derived_context_keeps_the_as_of(tmp_path):
    path = tmp_path / "v.db"
    ctx = VaultContext.local(path, writable=True)
    ctx.claim()
    pinned = ctx.at("2026-08-11")
    for derived in (pinned.provider_facing(), pinned.granting(WRITE),
                    pinned.revoking(WRITE), pinned.pinned(),
                    pinned.provider_facing().pinned()):
        assert derived.as_of == "2026-08-11"


def test_repr_is_unchanged_without_an_as_of(tmp_path):
    ctx = VaultContext.local(tmp_path / "v.db")
    assert "as_of" not in repr(ctx)
    assert "as_of=2026-08-11" in repr(ctx.at("2026-08-11"))


# --------------------------------------------------------------------------- #
# clamping
# --------------------------------------------------------------------------- #
# (tool, kwargs with a date after AS_OF, the same kwargs at AS_OF, the
# argument names clamped, keys that echo the caller's raw spec)
CASES = [
    ("get_daily_series", {"metric": METRIC, "start": "2026-07-20", "end": LATE},
     {"metric": METRIC, "start": "2026-07-20", "end": AS_OF}, {"end"}, set()),
    ("summarize_metric", {"metric": METRIC, "period": f"2026-07-20:{LATE}"},
     {"metric": METRIC, "period": f"2026-07-20:{AS_OF}"}, {"period"},
     {"requested_period"}),
    ("compare_periods", {"metric": METRIC, "period_a": f"2026-08-01:{LATE}",
                         "period_b": "2026-07-03:2026-07-31"},
     {"metric": METRIC, "period_a": f"2026-08-01:{AS_OF}",
      "period_b": "2026-07-03:2026-07-31"}, {"period_a"}, {"period_a.spec"}),
    ("list_workouts", {"start": "2026-07-20", "end": LATE},
     {"start": "2026-07-20", "end": AS_OF}, {"end"}, set()),
    ("get_impact_volume", {"start": "2026-07-20", "end": LATE},
     {"start": "2026-07-20", "end": AS_OF}, {"end"}, set()),
    ("get_sleep_regularity", {"start": "2026-07-10", "end": LATE},
     {"start": "2026-07-10", "end": AS_OF}, {"end"}, set()),
    ("get_training_load_detail", {"start": "2026-07-20", "end": LATE},
     {"start": "2026-07-20", "end": AS_OF}, {"end"}, set()),
    ("get_run_form", {"start": "2026-07-20", "end": LATE},
     {"start": "2026-07-20", "end": AS_OF}, {"end"}, set()),
    ("get_run_form", {"workout_date": LATE}, {"workout_date": AS_OF},
     {"workout_date"}, set()),
    ("get_briefing", {"day": LATE}, {"day": AS_OF}, {"day"}, set()),
    ("get_weekly_readiness", {"as_of": LATE}, {"as_of": AS_OF}, {"as_of"},
     set()),
    # The rest of the date-taking read tools (clamped by ``_clamps_dates``).
    ("get_intraday", {"metric": "heart_rate", "day": LATE},
     {"metric": "heart_rate", "day": AS_OF}, {"day"}, set()),
    ("get_hr_zones", {"day": LATE}, {"day": AS_OF}, {"day"}, set()),
    ("get_workout_segments", {"day": LATE}, {"day": AS_OF}, {"day"}, set()),
    ("get_block_structure", {"day": LATE}, {"day": AS_OF}, {"day"}, set()),
    ("get_workout_coverage", {"start": "2026-07-20", "end": LATE},
     {"start": "2026-07-20", "end": AS_OF}, {"end"}, set()),
    ("get_subjective", {"start_date": "2026-07-20", "end_date": LATE},
     {"start_date": "2026-07-20", "end_date": AS_OF}, {"end_date"}, set()),
    ("get_ingest_diagnostics", {"metric": METRIC, "start": "2026-07-20",
                                "end": LATE},
     {"metric": METRIC, "start": "2026-07-20", "end": AS_OF}, {"end"}, set()),
    ("get_weekly_series", {"metric": METRIC, "start": "2026-07-06",
                           "end": LATE},
     {"metric": METRIC, "start": "2026-07-06", "end": AS_OF}, {"end"}, set()),
    ("get_weekly_series", {"metric": METRIC, "start": LATE, "end": END},
     {"metric": METRIC, "start": AS_OF, "end": AS_OF}, {"start", "end"},
     set()),
    ("get_block_comparison", {"metric": METRIC, "block_weeks": 2,
                              "as_of": LATE},
     {"metric": METRIC, "block_weeks": 2, "as_of": AS_OF}, {"as_of"}, set()),
]
_ISO_DAY = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def _drop(result: dict, paths) -> dict:
    out = json.loads(json.dumps(result, default=repr))
    for path in paths:
        node, *rest = path.split(".")
        target = out
        while rest:
            target = target[node]
            node, *rest = rest
        target.pop(node, None)
    return out


@pytest.fixture(scope="module")
def built_vault(tmp_path_factory):
    path = tmp_path_factory.mktemp("as_of_clamps") / "demo.db"
    demo.build_demo_vault(path, days=60, read_only=False)
    return path


@pytest.fixture
def session(built_vault, tmp_path):
    copy = tmp_path / "vault.db"
    shutil.copy(built_vault, copy)
    return VaultContext.local(copy, user_id="test").provider_facing()


@pytest.mark.parametrize("name,late,at,clamped,echo", CASES,
                         ids=[f"{c[0]}:{sorted(c[3])[0]}" for c in CASES])
def test_an_explicit_date_after_the_as_of_is_clamped(session, name, late, at,
                                                     clamped, echo):
    tools = S.build_tools(session.at(AS_OF))
    got = tools[name](**late)
    assert "error" not in got, got
    assert got.get("clamped_to_as_of") is True
    assert got.get("requested_dates") == {k: late[k] for k in clamped}
    # The clamped call IS the as-of call, flag aside.
    want = tools[name](**at)
    assert "clamped_to_as_of" not in want and "requested_dates" not in want
    assert _drop(got, {"clamped_to_as_of", "requested_dates", *echo}) == \
        _drop(want, echo)
    # Nothing after the as-of is read: no later day anywhere but the echoes.
    rest = json.dumps(_drop(got, {"requested_dates", *echo}), default=repr)
    assert all(day <= AS_OF for day in _ISO_DAY.findall(rest)), name


@pytest.mark.parametrize("name,late", [(c[0], c[1]) for c in CASES],
                         ids=[f"{c[0]}:{sorted(c[3])[0]}" for c in CASES])
def test_without_an_as_of_nothing_is_clamped(session, name, late):
    got = S.build_tools(session)[name](**late)
    assert "clamped_to_as_of" not in got and "requested_dates" not in got


@pytest.mark.parametrize("given,bound", [
    (None, None), ("2026-08-11", "2026-08-11"), ("20260811", "2026-08-11"),
    ("2026-W33-2", "2026-08-11"), ("2026-8-11", None), ("", None),
])
def test_the_ask_binds_what_its_calendar_block_reads(tmp_path, given, bound):
    """The ask binds the day ``date.fromisoformat`` reads from its as_of, the
    same reading its calendar block has always made; an unreadable as_of is
    left unbound so the ask fails exactly where it did before."""
    from health_advisor import chat
    ctx = VaultContext.local(tmp_path / "v.db")
    assert chat._ctx_at_ask_as_of(ctx, given).as_of == bound


@pytest.mark.parametrize("name,kwargs", [
    # start after end, both after the as-of: clamping both to the as-of would
    # turn a refused window into an answered one.
    ("get_weekly_series", {"metric": METRIC, "start": END, "end": LATE}),
    ("get_workout_coverage", {"start": END, "end": LATE}),
    ("get_subjective", {"start_date": END, "end_date": LATE}),
    # not a day at all
    ("get_intraday", {"metric": "heart_rate", "day": "yesterday"}),
    ("get_block_comparison", {"metric": METRIC, "block_weeks": 2,
                              "as_of": "2026-13-01"}),
])
def test_a_refused_date_is_refused_the_same_way_under_an_as_of(session, name,
                                                               kwargs):
    unpinned = S.build_tools(session)[name](**kwargs)
    pinned = S.build_tools(session.at(AS_OF))[name](**kwargs)
    assert "error" in pinned
    assert pinned == unpinned
