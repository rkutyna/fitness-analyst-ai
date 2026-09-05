from __future__ import annotations

import inspect
import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
from datetime import date

import pytest

from health_advisor import approval_store, db, plan_log, plan_model
from health_advisor.context import VaultContext


def _ctx(path) -> VaultContext:
    ctx = VaultContext.local(path / "approval.db", user_id="approval-test",
                             writable=True)
    conn = ctx.connect()
    db.init_db(conn)
    conn.close()
    return ctx


def _parsed_rule(label: str) -> plan_model.Rule:
    return plan_model.Rule(
        kind=plan_model.RuleKind.SESSION,
        scope=plan_model.Scope(week="2026-08-17", session=label,
                               modality="running"),
        stated=plan_model.EffectiveInterval(
            start=date(2026, 8, 17), end=date(2026, 8, 23),
        ),
        statement=plan_model.Stated({"minutes": 30}),
        provenance=plan_model.ParsedProvenance("fixture/plan.md", 1),
    )


def test_token_crosses_process_boundary_and_is_spendable(tmp_path):
    ctx = _ctx(tmp_path)
    token = approval_store.issue_token(
        ctx, "token-cross-process", "hash-cross-process", "turn-cross-process",
        "opaque-flow", minted_at="2026-08-19T12:00:00+00:00",
    )
    issuer_pid = os.getpid()
    child = r'''
import json
import os
import sys
from datetime import date
from health_advisor import approval_store, plan_log, plan_model
from health_advisor.context import VaultContext

ctx = VaultContext.local(sys.argv[1], user_id="approval-test", writable=True)
before = approval_store.read_token(ctx, "token-cross-process")
rule = plan_model.Rule(
    kind=plan_model.RuleKind.SESSION,
    scope=plan_model.Scope(week="2026-08-17", session="child", modality="running"),
    stated=plan_model.EffectiveInterval(start=date(2026, 8, 17), end=date(2026, 8, 23)),
    statement=plan_model.Stated({"minutes": 30}),
    provenance=plan_model.ParsedProvenance("fixture/plan.md", 1),
)
statement_id = plan_log.append_rule_spending_token(
    ctx, rule, "token-cross-process", statement_id="statement-cross-process"
)
after = approval_store.read_token(ctx, "token-cross-process")
print(json.dumps({
    "pid": os.getpid(),
    "read_before": before.to_dict() if before else None,
    "statement_id": statement_id,
    "spent_after": after.spent_at if after else None,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", child, str(ctx.db_path)],
        check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    print(f"[Done-when 1] issuer_pid={issuer_pid}; spender_pid={result['pid']}; "
          f"read_before={result['read_before'] is not None}; "
          f"outcome={result['statement_id']}")
    assert result["pid"] != issuer_pid
    assert result["read_before"]["spent_at"] is None
    assert result["spent_after"] is not None


def test_single_use_has_exactly_one_winner_when_twenty_races_contend(tmp_path,
                                                                      monkeypatch):
    ctx = _ctx(tmp_path)
    for trial in range(20):
        token_id = f"race-token-{trial}"
        approval_store.issue_token(ctx, token_id, f"hash-{trial}",
                                   f"turn-{trial}", "flow")

    # Both worker connections are opened before the lock is installed. The
    # barrier action then holds a separate SQLite write transaction, so both
    # workers reach and trace BEGIN IMMEDIATE while that lock is held. Seeing
    # two distinct trace events proves the calls really contended; a launch
    # barrier alone would not establish that.
    lock_conn = sqlite3.connect(ctx.db_path, timeout=30,
                                check_same_thread=False)
    lock_conn.execute("PRAGMA journal_mode = DELETE")
    lock_conn.execute("PRAGMA busy_timeout = 30000")
    begins: queue.Queue[int] = queue.Queue()
    real_connect = db.connect

    def acquire_lock() -> None:
        lock_conn.execute("BEGIN IMMEDIATE")

    opened = threading.Barrier(2, action=acquire_lock)

    def connected(path, *, read_only=False):
        conn = real_connect(path, read_only=read_only)
        if not read_only:
            conn.set_trace_callback(
                lambda statement: begins.put(threading.get_ident())
                if statement.strip().upper() == "BEGIN IMMEDIATE" else None
            )
            opened.wait(timeout=10)
        return conn

    monkeypatch.setattr(plan_log.db, "connect", connected)
    monkeypatch.setattr(plan_log.db, "init_db", lambda conn: None)
    wins = 0
    refusals = 0
    for trial in range(20):
        results: list[str] = []
        result_lock = threading.Lock()

        def spend() -> None:
            try:
                plan_log.append_rule_spending_token(
                    ctx, _parsed_rule(f"race-{trial}"), f"race-token-{trial}",
                    statement_id=f"race-statement-{trial}-{threading.get_ident()}",
                )
                outcome = "won"
            except approval_store.TokenSpendRefused:
                outcome = "refused"
            with result_lock:
                results.append(outcome)

        workers = [threading.Thread(target=spend) for _ in range(2)]
        for worker in workers:
            worker.start()
        observed = {begins.get(timeout=10), begins.get(timeout=10)}
        assert len(observed) == 2
        lock_conn.commit()
        for worker in workers:
            worker.join(timeout=10)
            assert not worker.is_alive()
        assert sorted(results) == ["refused", "won"]
        wins += results.count("won")
        refusals += results.count("refused")

    lock_conn.close()
    print(f"[Done-when 2] trials=20; wins={wins}; refusals={refusals}; "
          "contention=two worker BEGIN IMMEDIATE trace events while a write lock was held")
    assert wins == 20
    assert refusals == 20


def test_failed_rule_insert_rolls_back_spend_and_rule_row(tmp_path):
    ctx = _ctx(tmp_path)
    approval_store.issue_token(ctx, "rollback-token", "hash-rollback",
                               "turn-rollback", "flow")
    invalid_rule = plan_model.Rule(
        kind=plan_model.RuleKind.SESSION,
        scope=plan_model.Scope(week="2026-08-17", session="rollback",
                               modality="running"),
        stated=plan_model.EffectiveInterval(
            start=date(2026, 8, 17), end=date(2026, 8, 23),
        ),
        statement=plan_model.Stated({"minutes": 30}),
        provenance=plan_model.ConversationTurnProvenance("missing-turn"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        plan_log.append_rule_spending_token(ctx, invalid_rule, "rollback-token",
                                            statement_id="rollback-failed")

    after_failure = approval_store.read_token(ctx, "rollback-token")
    conn = ctx.connect()
    try:
        rule_rows = conn.execute(
            "SELECT COUNT(*) FROM plan_statement_log "
            "WHERE statement_id = 'rollback-failed'"
        ).fetchone()[0]
        assert after_failure is not None
        assert after_failure.spent_at is None
        assert rule_rows == 0
    finally:
        conn.close()

    statement_id = plan_log.append_rule_spending_token(
        ctx, _parsed_rule("rollback-retry"), "rollback-token",
        statement_id="rollback-success",
    )
    assert statement_id == "rollback-success"
    final = approval_store.read_token(ctx, "rollback-token")
    print(f"[Done-when 3] failed_insert_rule_rows={rule_rows}; "
          f"token_after_failure_spent={after_failure.spent_at is not None}; "
          f"second_spend={statement_id}; final_spent={final.spent_at is not None}")
    assert final is not None and final.spent_at is not None


def test_audit_reports_approved_and_never_approved(tmp_path):
    ctx = _ctx(tmp_path)
    approval_store.issue_token(
        ctx, "audit-token", "hash-approved", "turn-approved", "flow",
        minted_at="2026-08-19T12:00:00+00:00",
    )
    approved = approval_store.audit_approval(ctx, "hash-approved", "turn-approved")
    never = approval_store.audit_approval(ctx, "hash-never", "turn-never")
    print(f"[Done-when 5] approved=({approved.approved}, {approved.minted_at}); "
          f"never=({never.approved}, {never.minted_at})")
    assert approved == approval_store.ApprovalAudit(
        True, "2026-08-19T12:00:00+00:00"
    )
    assert never == approval_store.ApprovalAudit(False, None)


def test_no_module_level_token_state():
    source = inspect.getsource(approval_store)
    assert "_TOKENS" not in source
    assert "TOKEN_STORE =" not in source
