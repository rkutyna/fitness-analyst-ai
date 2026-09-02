"""Vault-scoped durable user facts.

Facts hold meaning that sensors cannot see.  The model may help propose one,
but only an explicitly stated or confirmed row is rendered as context.  This
module never exposes fact text as a numeric input: callers receive strings for
prompt context and must not feed them into analysis or grounding payloads.
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from . import db
from .context import VaultContext


FACT_STATES = frozenset({"stated", "confirmed", "proposed", "rejected"})
CONTEXT_STATES = frozenset({"stated", "confirmed"})
# A confirmation request is a user-facing interruption, not a free resource.
MAX_CONFIRMATION_REQUESTS_PER_CONVERSATION = 1


def _has_table(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'user_facts'"
    ).fetchone() is not None


def _validate_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_confidence(confidence: float) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        raise ValueError("confidence must be between 0 and 1") from None
    if not 0 <= value <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return value


def _conversation_for_turn(conn: sqlite3.Connection,
                           conversation_turn_id: str | None) -> str | None:
    if conversation_turn_id is None:
        return None
    row = conn.execute(
        "SELECT conversation_id FROM conversation_turns WHERE id = ?",
        (conversation_turn_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown conversation turn: {conversation_turn_id}")
    return row["conversation_id"]


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _insert(
    conn: sqlite3.Connection,
    *,
    text: str,
    source: str,
    evidence: str,
    stated_at: str,
    scope: str,
    confidence: float,
    state: str,
    supersedes_fact_id: str | None = None,
    conversation_id: str | None = None,
    conversation_turn_id: str | None = None,
    fact_id: str | None = None,
) -> dict[str, Any]:
    fact = {
        "id": fact_id or uuid.uuid4().hex,
        "text": text,
        "source": source,
        "evidence": evidence,
        "stated_at": stated_at,
        "supersedes_fact_id": supersedes_fact_id,
        "scope": scope,
        "confidence": confidence,
        "state": state,
        "conversation_id": conversation_id,
        "conversation_turn_id": conversation_turn_id,
    }
    conn.execute(
        "INSERT INTO user_facts "
        "(id, text, source, evidence, stated_at, supersedes_fact_id, scope, "
        "confidence, state, conversation_id, conversation_turn_id) "
        "VALUES (:id, :text, :source, :evidence, :stated_at, "
        ":supersedes_fact_id, :scope, :confidence, :state, "
        ":conversation_id, :conversation_turn_id)",
        fact,
    )
    return fact


def record_stated(
    ctx: VaultContext,
    text: str,
    *,
    source: str = "chat",
    stated_at: str | None = None,
    scope: str = "user",
    confidence: float = 1.0,
    conversation_turn_id: str | None = None,
    supersedes_fact_id: str | None = None,
    fact_id: str | None = None,
) -> dict[str, Any]:
    """Persist a fact the user explicitly stated.

    This is the only ordinary creation path for a context-eligible fact.
    Inference must use :func:`propose`, which creates ``proposed`` instead.
    """
    text = _validate_text(text, "text")
    source = _validate_text(source, "source")
    scope = _validate_text(scope, "scope")
    stated_at = stated_at or db.utcnow_iso()
    confidence = _validate_confidence(confidence)
    evidence = f"user-stated: {text}"

    conn = ctx.connect()
    try:
        db.init_db(conn)
        conversation_id = _conversation_for_turn(conn, conversation_turn_id)
        conn.execute("BEGIN IMMEDIATE")
        if supersedes_fact_id is not None:
            _current_fact(conn, supersedes_fact_id)
        fact = _insert(
            conn, text=text, source=source, evidence=evidence,
            stated_at=stated_at, scope=scope, confidence=confidence,
            state="stated", supersedes_fact_id=supersedes_fact_id,
            conversation_id=conversation_id,
            conversation_turn_id=conversation_turn_id, fact_id=fact_id,
        )
        conn.commit()
        return fact
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def propose(
    ctx: VaultContext,
    text: str,
    *,
    source: str,
    evidence: str,
    scope: str = "user",
    confidence: float = 0.5,
    conversation_id: str | None = None,
    conversation_turn_id: str | None = None,
    stated_at: str | None = None,
    fact_id: str | None = None,
) -> dict[str, Any] | None:
    """Store one inferred fact as ``proposed`` or return ``None``.

    The same evidence cannot be proposed again after rejection, permanently.
    A conversation may create at most one confirmation request.  Returning
    ``None`` for either refusal keeps the caller from accidentally presenting
    a proposal that the store did not accept.
    """
    text = _validate_text(text, "text")
    source = _validate_text(source, "source")
    evidence = _validate_text(evidence, "evidence")
    scope = _validate_text(scope, "scope")
    confidence = _validate_confidence(confidence)
    stated_at = stated_at or db.utcnow_iso()

    conn = ctx.connect()
    try:
        db.init_db(conn)
        turn_conversation = _conversation_for_turn(conn, conversation_turn_id)
        if conversation_id is not None and turn_conversation not in (None, conversation_id):
            raise ValueError("conversation_turn_id belongs to another conversation")
        conversation_id = conversation_id or turn_conversation
        conn.execute("BEGIN IMMEDIATE")

        prior = conn.execute(
            "SELECT * FROM user_facts WHERE evidence = ? "
            "ORDER BY stated_at DESC, id DESC LIMIT 1",
            (evidence,),
        ).fetchone()
        if prior is not None and prior["state"] == "rejected":
            conn.rollback()
            return None
        if prior is not None and prior["state"] == "proposed":
            conn.rollback()
            return _row(prior)

        if conversation_id is not None:
            requests = conn.execute(
                "SELECT COUNT(*) FROM user_facts "
                "WHERE conversation_id = ? AND state = 'proposed'",
                (conversation_id,),
            ).fetchone()[0]
            if requests >= MAX_CONFIRMATION_REQUESTS_PER_CONVERSATION:
                conn.rollback()
                return None

        fact = _insert(
            conn, text=text, source=source, evidence=evidence,
            stated_at=stated_at, scope=scope, confidence=confidence,
            state="proposed", conversation_id=conversation_id,
            conversation_turn_id=conversation_turn_id, fact_id=fact_id,
        )
        conn.commit()
        return fact
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Descriptive aliases make the state transition explicit at call sites.
record_proposed = propose
propose_fact = propose


def _current_fact(conn: sqlite3.Connection, fact_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM user_facts f WHERE f.id = ? AND NOT EXISTS ("
        "SELECT 1 FROM user_facts newer WHERE newer.supersedes_fact_id = f.id)"
        , (fact_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown or superseded fact: {fact_id}")
    return row


def confirm(ctx: VaultContext, fact_id: str, *, confidence: float | None = None,
            stated_at: str | None = None) -> dict[str, Any]:
    """Append a confirmed row for a proposed fact; never mutate the proposal."""
    conn = ctx.connect()
    try:
        db.init_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        proposal = _current_fact(conn, fact_id)
        if proposal["state"] != "proposed":
            raise ValueError("only a current proposed fact can be confirmed")
        fact = _insert(
            conn, text=proposal["text"], source=proposal["source"],
            evidence=proposal["evidence"], stated_at=stated_at or db.utcnow_iso(),
            scope=proposal["scope"],
            confidence=(proposal["confidence"] if confidence is None
                        else _validate_confidence(confidence)),
            state="confirmed", supersedes_fact_id=fact_id,
            conversation_id=proposal["conversation_id"],
            conversation_turn_id=proposal["conversation_turn_id"],
        )
        conn.commit()
        return fact
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reject(ctx: VaultContext, fact_id: str, *, stated_at: str | None = None,
           source: str = "user") -> dict[str, Any]:
    """Append a rejection marker, retaining the proposal and its evidence."""
    source = _validate_text(source, "source")
    conn = ctx.connect()
    try:
        db.init_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        proposal = _current_fact(conn, fact_id)
        if proposal["state"] != "proposed":
            raise ValueError("only a current proposed fact can be rejected")
        fact = _insert(
            conn, text=proposal["text"], source=source,
            evidence=proposal["evidence"], stated_at=stated_at or db.utcnow_iso(),
            scope=proposal["scope"], confidence=proposal["confidence"],
            state="rejected", supersedes_fact_id=fact_id,
            conversation_id=proposal["conversation_id"],
            conversation_turn_id=proposal["conversation_turn_id"],
        )
        conn.commit()
        return fact
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


confirm_fact = confirm
reject_fact = reject


def _current_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT f.* FROM user_facts f WHERE NOT EXISTS ("
        "SELECT 1 FROM user_facts newer WHERE newer.supersedes_fact_id = f.id) "
        "ORDER BY f.stated_at ASC, f.id ASC"
    ).fetchall()
    return [dict(row) for row in rows]


def list_facts(ctx: VaultContext, *, include_history: bool = False) -> list[dict[str, Any]]:
    """List facts held by this vault; optionally include superseded history."""
    if not ctx.db_path.exists():
        return []
    conn = ctx.read_only()
    try:
        if not _has_table(conn):
            return []
        if not include_history:
            return _current_rows(conn)
        return [dict(row) for row in conn.execute(
            "SELECT * FROM user_facts ORDER BY stated_at ASC, id ASC"
        ).fetchall()]
    finally:
        conn.close()


def context_facts(ctx: VaultContext) -> list[dict[str, Any]]:
    """Return only current, context-eligible facts."""
    return [fact for fact in list_facts(ctx)
            if fact["state"] in CONTEXT_STATES]


def render_context(ctx: VaultContext) -> str:
    """Render durable facts as text, never as a numeric payload."""
    facts = context_facts(ctx)
    if not facts:
        return ""
    lines = [
        "--- BEGIN DURABLE USER CONTEXT (MEANING ONLY; NOT MEASUREMENTS) ---",
        "These are user-stated or user-confirmed facts. Use them as context, "
        "not as sensor data or inputs to calculations.",
    ]
    lines.extend(f"- {fact['text']}" for fact in facts)
    lines.append("--- END DURABLE USER CONTEXT ---")
    return "\n".join(lines)


def delete_fact(ctx: VaultContext, fact_id: str) -> None:
    """Delete a fact and its complete supersession lineage from this vault.

    Deleting the whole lineage prevents an older correction from becoming
    current again. It is intentional that deletion also removes the evidence;
    a later inference may then be proposed afresh if the user chooses.
    """
    conn = ctx.connect()
    try:
        db.init_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT id, supersedes_fact_id FROM user_facts"
        ).fetchall()
        by_id = {row["id"]: row for row in rows}
        if fact_id not in by_id:
            raise KeyError(f"unknown fact: {fact_id}")
        children: dict[str, list[str]] = {}
        for row in rows:
            parent = row["supersedes_fact_id"]
            if parent is not None:
                children.setdefault(parent, []).append(row["id"])

        # Remove descendants first (so foreign keys permit removing their
        # parents), then ancestors. This also supports deleting a historical
        # row shown by list_facts(include_history=True).
        descendants: list[str] = []
        stack = list(children.get(fact_id, []))
        while stack:
            child = stack.pop()
            descendants.append(child)
            stack.extend(children.get(child, []))
        ancestors: list[str] = []
        parent = by_id[fact_id]["supersedes_fact_id"]
        while parent is not None:
            ancestors.append(parent)
            parent = by_id[parent]["supersedes_fact_id"]
        deletion_order = list(reversed(descendants)) + [fact_id] + ancestors
        conn.executemany(
            "DELETE FROM user_facts WHERE id = ?",
            ((fact,) for fact in deletion_order),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


remove_fact = delete_fact
add_fact = record_stated
get_facts = list_facts
render_facts = render_context


__all__ = [
    "FACT_STATES", "CONTEXT_STATES",
    "MAX_CONFIRMATION_REQUESTS_PER_CONVERSATION",
    "record_stated", "propose", "propose_fact", "record_proposed",
    "confirm", "confirm_fact", "reject", "reject_fact",
    "list_facts", "get_facts", "context_facts", "render_context",
    "render_facts", "delete_fact", "remove_fact", "add_fact",
]
