"""The analyst tool seam, with no live provider or codex process."""
from __future__ import annotations

import asyncio
import json
import threading

import httpx
from fastapi.responses import JSONResponse

from health_advisor import chat, llm, receiver
from health_advisor.analyst_envelope import Envelope


def _envelope() -> Envelope:
    return Envelope(
        run_id="run-id", question="q", code_sha256="code-sha",
        vault_sha256="vault-sha", vault_version=4,
        ledger={"query_count": 1, "tables_read": ["daily_metrics"],
                "rows_read": 2, "parent_observed": True},
        tables=({
            "name": "resting_rate",
            "columns": ("day", "rate"),
            "units": ("date", "count/min"),
            "rows": (("2026-08-01", 61.25), ("2026-08-02", 60.5)),
            "row_count": 2,
        },),
        counts={"rows": 2, "cells": 4, "numeric_tokens": 2, "bytes": 100},
    )


def _model_tool_call(name, arguments):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}
    ]}}


def _analyst_payload():
    table = _envelope().to_dict()["tables"][0]
    return {
        "refused": False,
        "tables": [table],
        "provenance": {
            "run_id": "run-id", "vault_sha256": "vault-sha",
            "vault_user_version": 4, "code_sha256": "code-sha",
            "ledger": {"query_count": 1, "tables_read": ["daily_metrics"],
                        "rows_read": 2, "provenance": "parent-observed"},
        },
        "code": "emit('resting_rate', ['day', 'rate'], rows)",
    }


def test_analyst_query_runs_run_analyst_and_returns_validated_tables(
        vault, conn, monkeypatch):
    seen = {}

    def run_code(code, vault_path, run_dir, executor, *, limits=None):
        seen.update(code=code, vault_path=vault_path, executor=executor)
        return _envelope()

    response = receiver._run_analyst(
        vault, "How has my resting rate looked?",
        complete_fn=lambda prompt: "```python\nemit('resting_rate', rows)\n```",
        run_code_fn=run_code, executor_factory=lambda: "fake-executor")

    assert response.status_code == 200
    body = json.loads(response.body)
    assert seen["vault_path"] == vault.db_path
    assert body["tables"] == _envelope().to_dict()["tables"]
    assert body["provenance"]["ledger"]["provenance"] == "parent-observed"


def test_analyst_table_cell_claim_round_trips_through_ask_gate():
    ledger = [{
        "sequence": 1, "tool_name": "analyst_query", "arguments": {},
        "result": {"tables": [{
            "name": "resting_rate", "columns": ["day", "rate"],
            "units": ["date", "count/min"],
            "rows": [["2026-08-01", 61.25]], "row_count": 1,
        }]},
    }]
    claim = {
        "metric": None, "period": None, "field": "1", "value": 61.25,
        "source": {"sequence": 1,
                    "path": "$.result.tables[0].rows[0][1]"},
    }
    verdict = chat._verify_ask_answer(
        None, "My resting rate was 61.25 count/min.", [claim], ledger)
    assert verdict["ok"] is True
    assert verdict["tier_counts"] == {"path": 1, "metric": 0}


def test_codex_tool_config_excludes_analyst_query(vault, monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "codex")
    config = llm._deepdive_mcp_config(
        vault, include=llm.COACH_TOOLS)
    joined = " ".join(config)
    assert "analyst_query" in {
        schema["function"]["name"]
        for schema in llm.tool_schemas(vault, include=llm.COACH_TOOLS)
    }
    assert "analyst_query" not in joined
    assert "get_latest" in joined


def test_chat_analyst_holds_permit_while_direct_route_gets_429(
        vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "secret")
    started = threading.Event()
    release = threading.Event()
    payload = _analyst_payload()

    def blocked_run(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return JSONResponse(payload)

    monkeypatch.setattr(receiver, "_run_analyst", blocked_run)
    calls = {"count": 0}
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, json=_model_tool_call(
                "analyst_query", {"question": "resting rate"}))
        return httpx.Response(200, json=_model_tool_call(
            "submit_answer", {"text": "I found the table.", "claims": []}))

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: 100)

    async def exercise():
        app = receiver.create_app(
            vault, analyst_executor_factory=lambda: "unused")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            ask_task = asyncio.create_task(client.post(
                "/v1/ask", json={"question": "resting rate"},
                headers={"x-health-secret": "secret"}))
            await asyncio.to_thread(started.wait, 5)
            direct = await client.post(
                "/v1/analyst", json={"question": "direct"},
                headers={"x-health-secret": "secret"})
            release.set()
            ask = await ask_task
        return ask, direct

    ask, direct = asyncio.run(exercise())
    assert direct.status_code == 429
    assert ask.status_code == 200
    tool_message = next(message for message in bodies[1]["messages"]
                        if message["role"] == "tool")
    tool_result = json.loads(tool_message["content"])
    assert tool_result.pop("_ledger") == {"sequence": 1}
    assert tool_result == {"tables": payload["tables"]}
    assert ask.json()["attachments"] == [{
        "type": "table",
        "name": payload["tables"][0]["name"],
        "columns": payload["tables"][0]["columns"],
        "units": payload["tables"][0]["units"],
        "rows": payload["tables"][0]["rows"],
        "row_count": payload["tables"][0]["row_count"],
        "provenance": payload["provenance"], "code": payload["code"],
    }]


def test_chat_analyst_wait_is_bounded_while_direct_run_holds_permit(
        vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "secret")
    monkeypatch.setattr(receiver, "ANALYST_INTERNAL_WAIT_SECONDS", 0.02)
    started = threading.Event()
    release = threading.Event()
    direct_payload = {"refused": True, "reason": "released"}

    def slow_run(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return JSONResponse(direct_payload)

    monkeypatch.setattr(receiver, "_run_analyst", slow_run)
    calls = {"count": 0}
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, json=_model_tool_call(
                "analyst_query", {"question": "queued"}))
        return httpx.Response(200, json=_model_tool_call(
            "submit_answer", {"text": "No measured result.", "claims": []}))

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: 100)

    async def exercise():
        app = receiver.create_app(vault)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            direct_task = asyncio.create_task(client.post(
                "/v1/analyst", json={"question": "direct"},
                headers={"x-health-secret": "secret"}))
            await asyncio.to_thread(started.wait, 5)
            begin = asyncio.get_running_loop().time()
            ask = await client.post(
                "/v1/ask", json={"question": "queued"},
                headers={"x-health-secret": "secret"})
            elapsed = asyncio.get_running_loop().time() - begin
            release.set()
            direct = await direct_task
        return ask, direct, elapsed

    ask, direct, elapsed = asyncio.run(exercise())
    assert direct.status_code == 200
    assert ask.status_code == 200
    assert elapsed < 1.0
    tool_message = next(message for message in bodies[1]["messages"]
                        if message["role"] == "tool")
    refusal = json.loads(tool_message["content"])
    assert refusal["refused"] is True
    assert "timed out waiting" in refusal["reason"]


def test_conversation_reload_returns_analyst_attachment_with_turn(vault):
    conversation = chat.create_conversation(vault, conversation_id="analyst-c1")
    attachment = {
        "type": "table", "name": "resting_rate", "columns": ["rate"],
        "units": ["count/min"], "rows": [[61.25]], "row_count": 1,
        "provenance": {"ledger": {"provenance": "parent-observed"}},
        "code": "emit('resting_rate', rows)",
    }
    question = chat.append_turn(vault, conversation["id"], "user", "q")
    chat.append_turn(vault, conversation["id"], "assistant", "table answer",
                     answers_turn_id=question["id"], attachments=[attachment])

    reloaded = chat.get_conversation(vault, conversation["id"])
    assistant = reloaded["turns"][1]
    assert assistant["attachments"] == [attachment]


def test_legacy_check_rebuild_carries_a_stored_attachment(tmp_path):
    # The corrected data-loss path: a vault whose conversation_turns still has
    # the old assistant-only answers CHECK, but which already stores
    # attachments, must come through init_db's table rebuild with the
    # attachment intact. The original patch copied a fixed nine-column list
    # and would have silently dropped the column.
    import sqlite3
    from health_advisor import db

    path = tmp_path / "vault.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    db.init_db(conn)

    # Downgrade to the legacy shape (old CHECK), keeping attachments_json.
    conn.execute("PRAGMA foreign_keys = OFF")
    for trig in ("conversation_turns_no_update", "conversation_turns_no_delete",
                 "conversation_turns_supersedes_same_conversation",
                 "conversation_turns_answers_same_conversation"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
    conn.execute("ALTER TABLE conversation_turns RENAME TO ct_current")
    conn.execute("""
        CREATE TABLE conversation_turns (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL
                REFERENCES conversations(id) ON DELETE RESTRICT,
            sequence INTEGER NOT NULL CHECK (sequence > 0),
            role TEXT NOT NULL CHECK (length(trim(role)) > 0),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            supersedes_turn_id TEXT,
            answers_turn_id TEXT,
            client_disconnected_at TEXT,
            attachments_json TEXT,
            UNIQUE (conversation_id, sequence),
            CHECK (answers_turn_id IS NULL OR role = 'assistant')
        )
    """)
    conn.execute("DROP TABLE ct_current")
    conn.execute(
        "INSERT INTO conversations (id, created_at, updated_at) "
        "VALUES ('c1', 't0', 't0')")
    stored = json.dumps([{"type": "table", "name": "rhr_weekly",
                          "columns": ["w"], "units": ["count/min"],
                          "rows": [[61.0]], "row_count": 1}])
    conn.execute(
        "INSERT INTO conversation_turns "
        "(id, conversation_id, sequence, role, content, created_at, "
        "attachments_json) VALUES ('t1', 'c1', 1, 'assistant', 'a', 't0', ?)",
        (stored,))
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")

    assert db._conversation_turns_need_migration(conn)
    db.init_db(conn)
    assert not db._conversation_turns_need_migration(conn)

    row = conn.execute(
        "SELECT attachments_json FROM conversation_turns WHERE id = 't1'"
    ).fetchone()
    conn.close()
    assert row is not None and row["attachments_json"] == stored
