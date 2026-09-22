"""A plan week's day span is explicit: ``week_end`` on the Week and its log row.

A start date alone can only express seven days.  A first week that begins
mid-cycle is shorter, and one extended to the following boundary is longer;
both must round-trip through the log with every day, and only those days,
in the projection.  All dates here are synthetic.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

import pytest

from health_advisor import db, plan_log, plan_model
from health_advisor.context import VaultContext

from tests.fixtures.plan_week import SYNTHETIC_FIXTURE_WEEK

POLICY = SYNTHETIC_FIXTURE_WEEK.grading_policy
START = date(2030, 1, 4)


def _ctx(tmp_path) -> VaultContext:
    ctx = VaultContext.local(tmp_path / "span.db", user_id="fixture-user",
                             writable=True)
    conn = ctx.connect()
    db.init_db(conn)
    conn.close()
    return ctx


def _session(day: date) -> plan_model.Rule:
    return plan_model.Rule(
        kind=plan_model.RuleKind.SESSION,
        scope=plan_model.Scope(week=START.isoformat(), days=(day,)),
        stated=plan_model.EffectiveInterval(start=day, end=day),
        statement=plan_model.Stated(f"session on {day.isoformat()}"),
        provenance=plan_model.ParsedProvenance("fixture/span.md", 1),
        payload={"date": day.isoformat()},
    )


def _week(start: date, week_end: date | None, sessions: range) -> plan_model.Week:
    return plan_model.Week(
        week_start=start,
        rules=tuple(_session(start + timedelta(days=offset)) for offset in sessions),
        provenance=plan_model.ParsedProvenance(f"fixture/week-{start}.md", 1),
        grading_policy=POLICY,
        week_end=week_end,
    )


def _projected_days(week: plan_model.Week) -> list[date]:
    return sorted(date.fromisoformat(rule.payload["date"]) for rule in week.rules)


def test_null_end_reads_as_seven_days(tmp_path):
    ctx = _ctx(tmp_path)
    # Sessions on day offsets 0..8: only the first seven belong to the week.
    plan_log.append_week(ctx, _week(START, None, range(9)))

    conn = ctx.read_only()
    try:
        stored = conn.execute(
            "SELECT week_end FROM plan_week_log WHERE week_start = ?",
            (START.isoformat(),)).fetchone()
    finally:
        conn.close()
    assert stored["week_end"] is None

    projected = plan_log.project_week(ctx, START)
    assert projected.week_end is None
    assert projected.end == START + timedelta(days=6)
    assert len(projected.days) == 7
    assert _projected_days(projected) == [START + timedelta(days=offset)
                                          for offset in range(7)]
    assert "week_end" not in projected.to_dict()


@pytest.mark.parametrize("length", [3, 9])
def test_short_and_long_weeks_round_trip_every_day_and_only_those(tmp_path, length):
    ctx = _ctx(tmp_path)
    week_end = START + timedelta(days=length - 1)
    # One session beyond each edge of the span proves the edge is honoured.
    plan_log.append_week(ctx, _week(START, week_end, range(length + 2)))

    projected = plan_log.project_week(ctx, START)
    assert projected.week_end == week_end
    assert projected.end == week_end
    assert len(projected.days) == length
    assert _projected_days(projected) == [START + timedelta(days=offset)
                                          for offset in range(length)]

    rebuilt = plan_log.rebuild_week_projection(ctx, START)
    assert rebuilt.to_dict() == projected.to_dict()

    serial = projected.to_dict()
    assert serial["week_end"] == week_end.isoformat()
    assert plan_model.Week.from_dict(serial) == projected
    assert plan_model.Week.from_json(projected.to_json()) == projected


def test_week_end_before_start_is_refused():
    with pytest.raises(ValueError, match="week_end must not precede"):
        _week(START, START - timedelta(days=1), range(0))


def test_overlapping_declared_weeks_are_refused_at_write(tmp_path):
    ctx = _ctx(tmp_path)
    plan_log.append_week(ctx, _week(START, START + timedelta(days=8), range(0)))
    overlapping = START + timedelta(days=5)
    with pytest.raises(plan_log.OverlappingWeeks, match="overlaps the stored week"):
        plan_log.append_week(ctx, _week(overlapping, overlapping + timedelta(days=6),
                                        range(0)))
    # The refusal wrote nothing.
    assert [span.start for span in plan_log.week_spans(ctx)] == [START]
    # The adjacent week, starting the day after the declared end, is fine.
    after = START + timedelta(days=9)
    plan_log.append_week(ctx, _week(after, None, range(0)))
    assert [span.start for span in plan_log.week_spans(ctx)] == [START, after]


def test_a_day_inside_two_stored_weeks_fails_closed(tmp_path):
    ctx = _ctx(tmp_path)
    plan_log.append_week(ctx, _week(START, START + timedelta(days=8), range(0)))
    # Force a conflicting row past the write-time refusal, as a damaged or
    # hand-edited vault could hold one.
    later = START + timedelta(days=4)
    conn = ctx.connect()
    conn.execute(
        "INSERT INTO plan_week_log (week_start, grading_policy_json, "
        "provenance_kind, parsed_file, parsed_line, week_end, recorded_at) "
        "SELECT ?, grading_policy_json, 'parsed', 'fixture/forced.md', 1, ?, "
        "recorded_at FROM plan_week_log WHERE week_start = ?",
        (later.isoformat(), (later + timedelta(days=6)).isoformat(),
         START.isoformat()),
    )
    conn.commit()
    conn.close()

    assert plan_log.week_containing(ctx, START).start == START
    with pytest.raises(plan_log.OverlappingWeeks, match="refusing to choose"):
        plan_log.week_containing(ctx, later + timedelta(days=1))
    assert plan_log.week_containing(ctx, START + timedelta(days=20)) is None


def test_an_undeclared_seven_day_default_yields_to_the_next_start(tmp_path):
    ctx = _ctx(tmp_path)
    # A week that never declared its end, followed three days later by the
    # next week: the default tail overlaps, and it is the default that yields.
    next_start = START + timedelta(days=3)
    plan_log.append_week(ctx, _week(START, None, range(0)))
    plan_log.append_week(ctx, _week(next_start, None, range(0)))
    assert plan_log.week_containing(ctx, START + timedelta(days=1)).start == START
    assert plan_log.week_containing(ctx, next_start).start == next_start
    assert plan_log.week_containing(ctx, START + timedelta(days=5)).start == next_start
    # A declared end never yields: a later week inside it is refused.
    declared = START + timedelta(days=30)
    plan_log.append_week(ctx, _week(declared, declared + timedelta(days=3), range(0)))
    with pytest.raises(plan_log.OverlappingWeeks):
        plan_log.append_week(ctx, _week(declared + timedelta(days=2), None, range(0)))


def test_an_existing_vault_gains_the_column_and_its_weeks_read_as_seven_days(tmp_path):
    path = tmp_path / "legacy.db"
    ctx = VaultContext.local(path, user_id="fixture-user", writable=True)
    conn = ctx.connect()
    db.init_db(conn)
    conn.close()

    # Rebuild plan_week_log without week_end, as a vault written before the
    # column existed holds it, with one week declared in it.
    raw = sqlite3.connect(path)
    raw.executescript("""
        DROP TRIGGER IF EXISTS plan_week_log_no_update;
        DROP TRIGGER IF EXISTS plan_week_log_no_delete;
        DROP TABLE plan_week_log;
        CREATE TABLE plan_week_log (
            week_start TEXT PRIMARY KEY,
            grading_policy_json TEXT NOT NULL,
            provenance_kind TEXT NOT NULL,
            conversation_turn_id TEXT,
            parsed_file TEXT,
            parsed_line INTEGER,
            recorded_at TEXT NOT NULL
        );
    """)
    raw.execute(
        "INSERT INTO plan_week_log VALUES (?, ?, 'parsed', NULL, ?, 1, ?)",
        (START.isoformat(),
         json.dumps(plan_model.grading_policy_to_dict(POLICY)),
         "fixture/legacy.md", "2030-01-01T00:00:00+00:00"),
    )
    raw.commit()
    columns = {row[1] for row in raw.execute("PRAGMA table_info(plan_week_log)")}
    raw.close()
    assert "week_end" not in columns

    conn = ctx.connect()
    db.init_db(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(plan_week_log)")}
    row = conn.execute("SELECT week_end FROM plan_week_log").fetchone()
    conn.close()
    assert "week_end" in columns
    assert row[0] is None

    projected = plan_log.project_week(ctx, START)
    assert projected.week_end is None
    assert len(projected.days) == 7
    span = plan_log.week_containing(ctx, START + timedelta(days=6))
    assert (span.start, span.end, span.declared_end) == (
        START, START + timedelta(days=6), False)
