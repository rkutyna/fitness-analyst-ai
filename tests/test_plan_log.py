from __future__ import annotations

import inspect
import json
import re
import sqlite3
from dataclasses import fields
from datetime import date

import pytest

from health_advisor import db, plan_log, plan_model
from health_advisor.context import VaultContext
from tests.fixtures.plan_week import SYNTHETIC_FIXTURE_WEEK


def _ctx(tmp_path) -> VaultContext:
    ctx = VaultContext.local(tmp_path / "plan-log.db", user_id="fixture-user",
                             writable=True)
    conn = ctx.connect()
    db.init_db(conn)
    conn.close()
    return ctx


def _round_trip_rule() -> plan_model.Rule:
    return plan_model.Rule(
        kind=plan_model.RuleKind.CONDITIONAL,
        scope=plan_model.Scope(
            week=8, days=(date(2026, 8, 18),), session="tempo", modality="running"
        ),
        stated=plan_model.EffectiveInterval(
            start=date(2026, 8, 18), end=date(2026, 8, 24),
            include_start=False, include_end=True,
        ),
        statement=plan_model.Stated({"minutes": 40, "pace": {"unit": "min/mi"}}),
        provenance=plan_model.ConversationTurnProvenance("turn-fixture-1"),
        enforced_from=date(2026, 8, 20),
        acceptance_date=date(2026, 8, 19),
        payload={"threshold": 2, "labels": ["a", "b"]},
    )


def test_rule_round_trip_covers_every_declared_field(tmp_path):
    ctx = _ctx(tmp_path)
    rule = _round_trip_rule()
    # The FK is intentional: conversation-turn provenance must point at a
    # durable turn rather than becoming an uncheckable string.
    conn = ctx.connect()
    conn.execute(
        "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
        ("conversation-fixture", "2026-08-19T00:00:00+00:00",
         "2026-08-19T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO conversation_turns (id, conversation_id, sequence, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("turn-fixture-1", "conversation-fixture", 1, "user",
         "typed fixture statement", "2026-08-19T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    plan_log.append_rule(ctx, rule, statement_id="statement-fixture-1")
    read_back = plan_log.read_rules(ctx)[0]
    declared = tuple(field.name for field in fields(plan_model.Rule))
    covered = tuple(field.name for field in fields(plan_model.Rule)
                    if getattr(read_back, field.name) == getattr(rule, field.name))
    print(f"[Done-when 1] Rule fields covered: {len(covered)}; "
          f"plan_model declares: {len(declared)}; fields: {', '.join(declared)}")
    assert covered == declared
    assert len(covered) == len(declared)


def test_synthetic_week_round_trips_all_four_fields_from_log(tmp_path):
    ctx = _ctx(tmp_path)
    plan_log.append_week(ctx, SYNTHETIC_FIXTURE_WEEK)

    projected = plan_log.project_week(ctx, SYNTHETIC_FIXTURE_WEEK.week_start)
    assert projected.to_dict() == SYNTHETIC_FIXTURE_WEEK.to_dict()
    print("[Done-when 2] projected Week.to_dict():", projected.to_dict())


def test_projection_is_independent_of_rule_append_order(tmp_path):
    """Append order must not leak into the Week's METADATA.

    Each log gets the same Week declaration, while the two rules carry
    different provenance and are appended in opposite orders. A projection
    that takes provenance from its last rule cannot pass this test -- that was
    the measured v1 defect.

    What is deliberately NOT asserted is rule ORDER. Rules come back in append
    order (read_rules reads ORDER BY sequence) because D6 renders markdown FROM
    this log, so a projected week must reproduce the order its document was
    authored in. Asserting whole-Week equality here instead was measured to
    force a content-canonicalising sort that reordered a real 30-rule week --
    the same rules as a multiset, the wrong document.
    """
    first_rule = SYNTHETIC_FIXTURE_WEEK.rules[0]
    second_rule = plan_model.Rule(
        kind=plan_model.RuleKind.SESSION,
        scope=plan_model.Scope(
            week="2026-08-17", days=(date(2026, 8, 19),),
            session="long-run", modality="running",
        ),
        stated=plan_model.EffectiveInterval(
            start=date(2026, 8, 19), end=date(2026, 8, 24),
        ),
        statement=plan_model.Stated({"minutes": 60, "intensity": "steady"}),
        provenance=plan_model.ParsedProvenance(
            "fixture/plan-week-2026-08-17.md", 13
        ),
        acceptance_date=date(2026, 8, 17),
        payload={"source": "fixture", "priority": 2},
    )
    rules = (first_rule, second_rule)
    weeks = []
    for index, order in enumerate((rules, tuple(reversed(rules)))):
        ctx = _ctx(tmp_path / f"order-{index}")
        plan_log.append_week_metadata(ctx, SYNTHETIC_FIXTURE_WEEK)
        for rule_index, rule in enumerate(order):
            plan_log.append_rule(
                ctx, rule, statement_id=f"order-{index}-statement-{rule_index}"
            )
        weeks.append(plan_log.project_week(ctx, SYNTHETIC_FIXTURE_WEEK.week_start))

    first, second = (week.to_dict() for week in weeks)
    # The metadata is what must be order-independent.
    assert first["provenance"] == second["provenance"]
    assert first["grading_policy"] == second["grading_policy"]
    assert first["week_start"] == second["week_start"]
    # The same rules survive, as a multiset...
    canonical = lambda week: sorted(
        json.dumps(rule, sort_keys=True) for rule in week["rules"])
    assert canonical(first) == canonical(second)
    # ...and their order follows append order, so these two logs differ in it.
    # This half fails if a canonicalising sort is ever reintroduced.
    assert [rule["provenance"] for rule in first["rules"]] != \
           [rule["provenance"] for rule in second["rules"]]


def test_rebuild_uses_mutated_log_and_never_updates_projection_row(tmp_path):
    ctx = _ctx(tmp_path)
    plan_log.append_week_metadata(ctx, SYNTHETIC_FIXTURE_WEEK)
    stated = SYNTHETIC_FIXTURE_WEEK.rules[0]
    plan_log.append_rule(ctx, stated, statement_id="fixture-statement-stated")
    first = plan_log.rebuild_week_projection(ctx, SYNTHETIC_FIXTURE_WEEK.week_start)

    withdrawn = plan_model.Rule(
        kind=stated.kind,
        scope=stated.scope,
        stated=stated.stated,
        statement=plan_model.Withdrawal("fixture withdrawal"),
        provenance=stated.provenance,
        acceptance_date=stated.acceptance_date,
        payload={"source": "fixture", "priority": 1},
    )
    plan_log.append_rule(ctx, withdrawn, statement_id="fixture-statement-withdrawal")
    second = plan_log.rebuild_week_projection(ctx, SYNTHETIC_FIXTURE_WEEK.week_start)
    assert first.rules
    assert second.rules == ()
    assert first != second

    conn = ctx.connect()
    rows = conn.execute(
        "SELECT projection_id, payload_json FROM plan_projections "
        "WHERE week_start = ? ORDER BY created_at, projection_id",
        ("2026-08-17",),
    ).fetchall()
    assert len(rows) == 2
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE plan_projections SET payload_json = ? WHERE projection_id = ?",
            ("{}", rows[0][0]),
        )
    conn.close()


def test_no_projection_update_path_exists_and_guard_would_fail_if_added():
    source = inspect.getsource(plan_log) + inspect.getsource(db)
    assert not re.search(r"UPDATE\s+plan_projections", source, re.IGNORECASE)
    assert not re.search(r"(?:INSERT\s+OR\s+REPLACE|REPLACE\s+INTO)\s+plan_projections",
                         source, re.IGNORECASE)


def test_withdrawal_for_same_scope_removes_rule_from_week(tmp_path):
    ctx = _ctx(tmp_path)
    plan_log.append_week_metadata(ctx, SYNTHETIC_FIXTURE_WEEK)
    stated = SYNTHETIC_FIXTURE_WEEK.rules[0]
    withdrawal = plan_model.Rule(
        kind=stated.kind,
        scope=stated.scope,
        stated=stated.stated,
        statement=plan_model.Withdrawal("no longer applies"),
        provenance=stated.provenance,
        acceptance_date=stated.acceptance_date,
        payload={"source": "fixture", "priority": 1},
    )
    plan_log.append_rule(ctx, stated, statement_id="withdrawal-test-stated")
    plan_log.append_rule(ctx, withdrawal, statement_id="withdrawal-test-withdrawal")
    projected = plan_log.project_week(ctx, SYNTHETIC_FIXTURE_WEEK.week_start)
    assert projected.rules == ()
    print("[Done-when 4] Stated followed by Withdrawal for the same scope: no rule")
