"""Append-only typed plan statements and rebuildable Week projections.

The statement and Week-metadata logs are the source of truth.
``plan_projections`` is only a cache of a rebuild, so this module never reads
it while projecting a week and never updates one of its rows in place.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, timedelta
from typing import Any, Mapping

from . import db
from . import plan_model
from .context import VaultContext


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _provenance_columns(provenance: plan_model.Provenance) -> tuple[
        str, str | None, str | None, int | None]:
    if isinstance(provenance, plan_model.ConversationTurnProvenance):
        return "conversation_turn", provenance.conversation_turn_id, None, None
    if isinstance(provenance, plan_model.ParsedProvenance):
        return "parsed", None, provenance.file, provenance.line
    raise TypeError("rule provenance must be conversation-turn or parsed")


def _rule_values(rule: plan_model.Rule) -> tuple[Any, ...]:
    provenance_kind, conversation_turn_id, parsed_file, parsed_line = (
        _provenance_columns(rule.provenance)
    )
    statement = plan_model.statement_to_dict(rule.statement)
    assert statement is not None
    return (
        rule.kind,
        _json(rule.scope.to_dict()),
        statement["type"],
        _json(statement),
        provenance_kind,
        conversation_turn_id,
        parsed_file,
        parsed_line,
        rule.stated.start.isoformat(),
        rule.stated.end.isoformat() if rule.stated.end is not None else None,
        int(rule.stated.include_start),
        int(rule.stated.include_end),
        rule.enforced_from.isoformat() if rule.enforced_from else None,
        rule.acceptance_date.isoformat() if rule.acceptance_date else None,
        _json(rule.to_dict()["payload"]),
    )


def _week_metadata_values(week: plan_model.Week) -> tuple[Any, ...]:
    if isinstance(week.provenance, plan_model.ConversationTurnProvenance):
        return (
            week.week_start.isoformat(),
            _json(plan_model.grading_policy_to_dict(week.grading_policy)),
            "conversation_turn", week.provenance.conversation_turn_id,
            None, None,
        )
    if isinstance(week.provenance, plan_model.ParsedProvenance):
        return (
            week.week_start.isoformat(),
            _json(plan_model.grading_policy_to_dict(week.grading_policy)),
            "parsed", None, week.provenance.file, week.provenance.line,
        )
    raise TypeError("week provenance must be conversation-turn or parsed")


def _insert_rule(conn, rule: plan_model.Rule, statement_id: str) -> None:
    conn.execute(
        """
        INSERT INTO plan_statement_log
            (statement_id, kind, scope_json, statement_kind,
             statement_json, provenance_kind, conversation_turn_id,
             parsed_file, parsed_line, effective_start, effective_end,
             include_start, include_end, enforced_from, acceptance_date,
             payload_json, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (statement_id, *_rule_values(rule), db.utcnow_iso()),
    )


def _insert_week_metadata(conn, week: plan_model.Week) -> None:
    conn.execute(
        """
        INSERT INTO plan_week_log
            (week_start, grading_policy_json, provenance_kind,
             conversation_turn_id, parsed_file, parsed_line, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (*_week_metadata_values(week), db.utcnow_iso()),
    )


def append_rule(
    ctx: VaultContext,
    rule: plan_model.Rule,
    *,
    statement_id: str | None = None,
) -> str:
    """Append one typed rule statement to the vault's log."""
    if not isinstance(ctx, VaultContext):
        raise TypeError("append_rule requires a VaultContext")
    if not isinstance(rule, plan_model.Rule):
        raise TypeError("rule must be a plan_model.Rule")
    rule.validate()
    statement_id = statement_id or uuid.uuid4().hex
    if not statement_id:
        raise ValueError("statement_id must be non-empty")
    conn = ctx.connect()
    try:
        db.init_db(conn)
        _insert_rule(conn, rule, statement_id)
        conn.commit()
    finally:
        conn.close()
    return statement_id


def append_week_metadata(
    ctx: VaultContext,
    week: plan_model.Week,
) -> None:
    """Append the immutable Week-level policy and provenance declaration."""
    if not isinstance(ctx, VaultContext):
        raise TypeError("append_week_metadata requires a VaultContext")
    if not isinstance(week, plan_model.Week):
        raise TypeError("week must be a plan_model.Week")
    conn = ctx.connect()
    try:
        db.init_db(conn)
        _insert_week_metadata(conn, week)
        conn.commit()
    finally:
        conn.close()


def append_week(ctx: VaultContext, week: plan_model.Week) -> None:
    """Append a Week declaration and its typed rule statements atomically."""
    if not isinstance(ctx, VaultContext):
        raise TypeError("append_week requires a VaultContext")
    if not isinstance(week, plan_model.Week):
        raise TypeError("week must be a plan_model.Week")
    conn = ctx.connect()
    try:
        db.init_db(conn)
        _insert_week_metadata(conn, week)
        for index, rule in enumerate(week.rules):
            rule.validate()
            _insert_rule(conn, rule, f"{week.week_start.isoformat()}-{index}")
        conn.commit()
    finally:
        conn.close()


def _rule_from_row(row: Mapping[str, Any]) -> plan_model.Rule:
    statement = plan_model.statement_from_dict(json.loads(row["statement_json"]))
    if statement is None:
        raise ValueError("plan statement row has no typed statement")
    if row["statement_kind"] != plan_model.statement_to_dict(statement)["type"]:
        raise ValueError("plan statement kind disagrees with its typed value")
    if row["provenance_kind"] == "conversation_turn":
        provenance: plan_model.Provenance = plan_model.ConversationTurnProvenance(
            row["conversation_turn_id"]
        )
    elif row["provenance_kind"] == "parsed":
        provenance = plan_model.ParsedProvenance(
            row["parsed_file"], int(row["parsed_line"])
        )
    else:
        raise ValueError(f"unknown plan statement provenance: {row['provenance_kind']!r}")
    return plan_model.Rule(
        kind=row["kind"],
        scope=plan_model.Scope.from_dict(json.loads(row["scope_json"])),
        stated=plan_model.EffectiveInterval(
            start=date.fromisoformat(row["effective_start"]),
            end=(date.fromisoformat(row["effective_end"])
                 if row["effective_end"] else None),
            include_start=bool(row["include_start"]),
            include_end=bool(row["include_end"]),
        ),
        statement=statement,
        provenance=provenance,
        enforced_from=(date.fromisoformat(row["enforced_from"])
                        if row["enforced_from"] else None),
        acceptance_date=(date.fromisoformat(row["acceptance_date"])
                         if row["acceptance_date"] else None),
        payload=json.loads(row["payload_json"]),
    )


def read_rules(ctx: VaultContext) -> list[plan_model.Rule]:
    """Read all log statements in append order."""
    if not isinstance(ctx, VaultContext):
        raise TypeError("read_rules requires a VaultContext")
    conn = ctx.read_only()
    try:
        rows = conn.execute(
            "SELECT * FROM plan_statement_log ORDER BY sequence"
        ).fetchall()
        return [_rule_from_row(row) for row in rows]
    finally:
        conn.close()


def _read_week_metadata(ctx: VaultContext, week_start: date) -> tuple[
        plan_model.Provenance, plan_model.GradingPolicy]:
    conn = ctx.read_only()
    try:
        row = conn.execute(
            "SELECT * FROM plan_week_log WHERE week_start = ?",
            (week_start.isoformat(),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(
            f"the plan log has no Week metadata declaration for {week_start}"
        )
    if row["provenance_kind"] == "conversation_turn":
        provenance: plan_model.Provenance = (
            plan_model.ConversationTurnProvenance(row["conversation_turn_id"])
        )
    elif row["provenance_kind"] == "parsed":
        provenance = plan_model.ParsedProvenance(
            row["parsed_file"], int(row["parsed_line"])
        )
    else:
        raise ValueError(
            f"unknown Week metadata provenance: {row['provenance_kind']!r}"
        )
    return provenance, plan_model.grading_policy_from_dict(
        json.loads(row["grading_policy_json"])
    )


def _scope_matches(scope: plan_model.Scope, week_start: date) -> bool:
    if scope.week is not None:
        week_values = {
            week_start.isoformat(),
            week_start.strftime("%G-W%V"),
            str(week_start.isocalendar().week),
        }
        # The markdown parser uses authored ordinal labels such as
        # ``week-08``.  The effective interval remains the date authority for
        # those labels; date and ISO-week selectors are checked directly.
        week_selector = str(scope.week)
        if not week_selector.startswith("week-") and week_selector not in week_values:
            return False
    if scope.days:
        week_days = {week_start + timedelta(days=i) for i in range(7)}
        if not week_days.intersection(
            date.fromisoformat(day) for day in scope.days
        ):
            return False
    return True


def _applies_to_week(rule: plan_model.Rule, week_start: date) -> bool:
    if not _scope_matches(rule.scope, week_start):
        return False
    return any(
        rule.stated.contains(week_start + timedelta(days=i))
        for i in range(7)
    )


def _scope_key(scope: plan_model.Scope) -> str:
    return _json(scope.to_dict())


def _project(rules: list[plan_model.Rule], week_start: date,
             provenance: plan_model.Provenance,
             grading_policy: plan_model.GradingPolicy) -> plan_model.Week:
    events = [rule for rule in rules if _applies_to_week(rule, week_start)]
    active: list[plan_model.Rule] = []
    for rule in events:
        key = _scope_key(rule.scope)
        if isinstance(rule.statement, plan_model.Withdrawal):
            active = [existing for existing in active
                      if _scope_key(existing.scope) != key]
        else:
            active.append(rule)
    # Rule ORDER is append order, deliberately: read_rules reads ORDER BY
    # sequence, and D6 makes markdown a view rendered FROM this log, so a week
    # projected back must reproduce the order the document was authored in.
    # Canonicalising by content here was measured to reorder a real 30-rule week
    # (anchors ahead of that week's sessions) while preserving the multiset --
    # the same rules, the wrong document. What must NOT depend on append order
    # is the Week's METADATA, and that no longer can: provenance and
    # grading_policy are read from the stored week declaration, not from
    # whichever statement happened to be last.
    return plan_model.Week(
        week_start=week_start,
        rules=tuple(active),
        provenance=provenance,
        grading_policy=grading_policy,
    )


def project_week(
    ctx: VaultContext,
    week_start: date | str,
) -> plan_model.Week:
    """Build a Week from the statement log, never from plan_projections."""
    week_start = _as_date(week_start)
    provenance, grading_policy = _read_week_metadata(ctx, week_start)
    return _project(read_rules(ctx), week_start, provenance, grading_policy)


def rebuild_week_projection(
    ctx: VaultContext,
    week_start: date | str,
) -> plan_model.Week:
    """Rebuild and append a projection cache row from the current log."""
    if not isinstance(ctx, VaultContext):
        raise TypeError("rebuild_week_projection requires a VaultContext")
    week_start = _as_date(week_start)
    conn = ctx.connect()
    try:
        db.init_db(conn)
        rows = conn.execute(
            "SELECT * FROM plan_statement_log ORDER BY sequence"
        ).fetchall()
        metadata = conn.execute(
            "SELECT * FROM plan_week_log WHERE week_start = ?",
            (week_start.isoformat(),),
        ).fetchone()
        if metadata is None:
            raise ValueError(
                f"the plan log has no Week metadata declaration for {week_start}"
            )
        if metadata["provenance_kind"] == "conversation_turn":
            provenance: plan_model.Provenance = (
                plan_model.ConversationTurnProvenance(
                    metadata["conversation_turn_id"]
                )
            )
        elif metadata["provenance_kind"] == "parsed":
            provenance = plan_model.ParsedProvenance(
                metadata["parsed_file"], int(metadata["parsed_line"])
            )
        else:
            raise ValueError(
                f"unknown Week metadata provenance: {metadata['provenance_kind']!r}"
            )
        week = _project(
            [_rule_from_row(row) for row in rows], week_start, provenance,
            plan_model.grading_policy_from_dict(
                json.loads(metadata["grading_policy_json"])
            ),
        )
        db.save_week_projection(conn, week, projection_id=uuid.uuid4().hex)
        return week
    finally:
        conn.close()


# Readable aliases for callers using the vocabulary from the issue.
write_rule = append_rule
load_rules = read_rules
project_week_from_log = project_week
rebuild_projection = rebuild_week_projection


__all__ = [
    "append_rule", "append_week_metadata", "append_week", "write_rule",
    "read_rules", "load_rules",
    "project_week", "project_week_from_log", "rebuild_week_projection",
    "rebuild_projection",
]
