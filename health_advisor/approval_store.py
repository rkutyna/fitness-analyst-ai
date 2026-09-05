"""Durable, per-vault storage for opaque approval tokens.

The client owns approval policy and token contents. This module only persists
the values and reports their storage state; spending is coupled to the plan
log in :func:`health_advisor.plan_log.append_rule_spending_token`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import db
from .context import VaultContext


class ApprovalTokenError(RuntimeError):
    """Base class for approval-token storage failures."""


class TokenSpendRefused(ApprovalTokenError):
    """The token was not available for a single-use spend."""


@dataclass(frozen=True)
class ApprovalToken:
    """One stored token, including its current spend state."""

    token_id: str
    statement_hash: str
    turn_id: str
    flow: str
    minted_at: str
    spent_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "statement_hash": self.statement_hash,
            "turn_id": self.turn_id,
            "flow": self.flow,
            "minted_at": self.minted_at,
            "spent_at": self.spent_at,
        }


@dataclass(frozen=True)
class ApprovalAudit:
    """Whether any token approved a statement for a conversation turn."""

    approved: bool
    minted_at: str | None


def _require_context(ctx: VaultContext, operation: str) -> None:
    if not isinstance(ctx, VaultContext):
        raise TypeError(f"{operation} requires a VaultContext")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _from_row(row: Mapping[str, Any]) -> ApprovalToken:
    return ApprovalToken(
        token_id=row["token_id"],
        statement_hash=row["statement_hash"],
        turn_id=row["turn_id"],
        flow=row["flow"],
        minted_at=row["minted_at"],
        spent_at=row["spent_at"],
    )


def issue_token(
    ctx: VaultContext,
    token_id: str,
    statement_hash: str,
    turn_id: str,
    flow: str,
    *,
    minted_at: str | None = None,
) -> ApprovalToken:
    """Persist one opaque token in the context's vault.

    The client supplies the token identity and policy values. ``minted_at``
    may be supplied by that client; otherwise the engine records its UTC
    insertion time.
    """
    _require_context(ctx, "issue_token")
    token_id = _require_text(token_id, "token_id")
    statement_hash = _require_text(statement_hash, "statement_hash")
    turn_id = _require_text(turn_id, "turn_id")
    flow = _require_text(flow, "flow")
    minted_at = db.utcnow_iso() if minted_at is None else _require_text(
        minted_at, "minted_at"
    )

    conn = ctx.connect()
    try:
        db.init_db(conn)
        conn.execute(
            """
            INSERT INTO approval_tokens
                (token_id, statement_hash, turn_id, flow, minted_at, spent_at)
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (token_id, statement_hash, turn_id, flow, minted_at),
        )
        conn.commit()
    finally:
        conn.close()
    return ApprovalToken(token_id, statement_hash, turn_id, flow, minted_at, None)


def read_token(ctx: VaultContext, token_id: str) -> ApprovalToken | None:
    """Read one token from the context's vault, or return ``None``."""
    _require_context(ctx, "read_token")
    token_id = _require_text(token_id, "token_id")
    conn = ctx.read_only()
    try:
        row = conn.execute(
            "SELECT * FROM approval_tokens WHERE token_id = ?", (token_id,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _from_row(row)


def audit_approval(
    ctx: VaultContext,
    statement_hash: str,
    turn_id: str,
) -> ApprovalAudit:
    """Report whether a statement/turn pair was approved and when."""
    _require_context(ctx, "audit_approval")
    statement_hash = _require_text(statement_hash, "statement_hash")
    turn_id = _require_text(turn_id, "turn_id")
    conn = ctx.read_only()
    try:
        row = conn.execute(
            """
            SELECT MIN(minted_at) AS minted_at
            FROM approval_tokens
            WHERE statement_hash = ? AND turn_id = ?
            """,
            (statement_hash, turn_id),
        ).fetchone()
    finally:
        conn.close()
    minted_at = None if row is None else row["minted_at"]
    return ApprovalAudit(minted_at is not None, minted_at)


# Descriptive aliases for callers that name the operation as a store/read.
store_token = issue_token
get_token = read_token
read_approval = audit_approval
