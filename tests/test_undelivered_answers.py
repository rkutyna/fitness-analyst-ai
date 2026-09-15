"""Server-side recovery of answers produced after an ask client disconnects."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from health_advisor import chat, db, receiver


HEADERS = {"x-health-secret": "ask-secret"}


def _disconnected_answer(vault, conversation_id, question, answer,
                         *, progress_id=None, mode=None):
    question_turn = chat.append_turn(vault, conversation_id, "user", question)
    return chat.append_turn(
        vault,
        conversation_id,
        "assistant",
        answer,
        answers_turn_id=question_turn["id"],
        client_disconnected_at="2026-09-06T12:00:00+00:00",
        progress_id=progress_id,
        mode=mode,
    )


def test_get_returns_a_disconnected_undelivered_turn(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    conversation = chat.create_conversation(vault, conversation_id="recover-one")
    answer = _disconnected_answer(
        vault, conversation["id"], "q", "answer",
        progress_id="progress-one", mode="narration")

    with TestClient(receiver.create_app(vault)) as client:
        response = client.get(
            "/v1/ask/undelivered",
            params={"progress_id": "progress-one"}, headers=HEADERS)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == answer["id"]
    assert body["conversation_id"] == conversation["id"]
    assert body["answers_turn_id"] == answer["answers_turn_id"]
    assert body["content"] == "answer"
    assert body["client_disconnected_at"]
    assert body["delivered_at"] is None
    assert body["progress_id"] == "progress-one"
    assert body["mode"] == "narration"


def test_get_does_not_return_a_delivered_turn(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    conversation = chat.create_conversation(vault, conversation_id="recover-two")
    answer = _disconnected_answer(vault, conversation["id"], "q", "answer")

    with TestClient(receiver.create_app(vault)) as client:
        marked = client.post(
            "/v1/ask/delivered", json={"turn_id": answer["id"]}, headers=HEADERS
        )
        response = client.get("/v1/ask/undelivered", headers=HEADERS)

    assert marked.status_code == 200, marked.text
    assert response.status_code == 200, response.text
    assert response.json() == {}


def test_get_never_guesses_from_a_multi_conversation_backlog(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")

    conversations = [
        chat.create_conversation(vault, conversation_id="legacy-a"),
        chat.create_conversation(vault, conversation_id="legacy-b"),
        chat.create_conversation(vault, conversation_id="legacy-c"),
    ]
    legacy = [
        ("legacy-1", "legacy-a", "2026-08-01T00:00:00+00:00"),
        ("legacy-2", "legacy-a", "2026-08-01T00:00:01+00:00"),
        ("legacy-3", "legacy-b", "2026-08-15T00:00:00+00:00"),
        ("legacy-4", "legacy-b", "2026-08-15T00:00:01+00:00"),
        ("legacy-5", "legacy-c", "2026-09-01T00:00:00+00:00"),
        ("legacy-6", "legacy-c", "2026-09-01T00:00:01+00:00"),
        ("legacy-7", "legacy-c", "2026-09-01T00:00:02+00:00"),
    ]
    conn = vault.connect()
    try:
        for sequence, (turn_id, conversation_id, created_at) in enumerate(
                legacy, start=1):
            conn.execute(
                "INSERT INTO conversation_turns "
                "(id, conversation_id, sequence, role, content, created_at, "
                "client_disconnected_at, delivered_at, progress_id, mode) "
                "VALUES (?, ?, ?, 'assistant', ?, ?, ?, NULL, NULL, NULL)",
                (turn_id, conversation_id, sequence, f"legacy answer {sequence}",
                 created_at, created_at),
            )
        conn.commit()
    finally:
        conn.close()
    keyed = _disconnected_answer(
        vault, conversations[0]["id"], "keyed question", "keyed answer",
        progress_id="keyed-progress", mode="narration")

    with TestClient(receiver.create_app(vault)) as client:
        no_parameter = client.get("/v1/ask/undelivered", headers=HEADERS)
        unknown = client.get(
            "/v1/ask/undelivered", params={"progress_id": "unknown"},
            headers=HEADERS)
        keyed_response = client.get(
            "/v1/ask/undelivered", params={"progress_id": "keyed-progress"},
            headers=HEADERS)
        marked = client.post(
            "/v1/ask/delivered", json={"turn_id": keyed["id"]}, headers=HEADERS)
        after_delivery = client.get(
            "/v1/ask/undelivered", params={"progress_id": "keyed-progress"},
            headers=HEADERS)

    assert no_parameter.status_code == 200
    assert unknown.status_code == 200
    assert {
        "no_parameter": no_parameter.json(),
        "unknown": unknown.json(),
    } == {"no_parameter": {}, "unknown": {}}
    assert keyed_response.status_code == 200
    assert keyed_response.json() == keyed
    assert keyed_response.json()["mode"] == "narration"
    assert marked.status_code == 200
    assert after_delivery.status_code == 200
    assert after_delivery.json() == {}


def test_mark_delivered_is_idempotent(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    conversation = chat.create_conversation(vault, conversation_id="recover-idempotent")
    answer = _disconnected_answer(vault, conversation["id"], "q", "answer")

    with TestClient(receiver.create_app(vault)) as client:
        first = client.post(
            "/v1/ask/delivered", json={"turn_id": answer["id"]}, headers=HEADERS
        )
        delivered_at = first.json()["delivered_at"]
        second = client.post(
            "/v1/ask/delivered", json={"turn_id": answer["id"]}, headers=HEADERS
        )

    assert first.status_code == second.status_code == 200
    assert second.json()["delivered_at"] == delivered_at
    stored = {turn["id"]: turn for turn in chat.list_turns(vault, conversation["id"])}
    assert stored[answer["id"]]["content"] == "answer"
    assert stored[answer["id"]]["client_disconnected_at"]
    assert stored[answer["id"]]["delivered_at"] == delivered_at


def test_history_excludes_undelivered_and_includes_delivered_disconnect_answers(
        vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    conversation = chat.create_conversation(vault, conversation_id="recover-history")
    undelivered = _disconnected_answer(
        vault, conversation["id"], "unseen question", "unseen answer"
    )
    delivered = _disconnected_answer(
        vault, conversation["id"], "seen question", "seen answer"
    )

    with TestClient(receiver.create_app(vault)) as client:
        marked = client.post(
            "/v1/ask/delivered", json={"turn_id": delivered["id"]}, headers=HEADERS
        )
    assert marked.status_code == 200, marked.text

    rendered = chat._render_history(chat.list_turns(vault, conversation["id"]))
    assert "unseen question" in rendered
    assert "unseen answer" not in rendered
    assert "seen question" in rendered
    assert "seen answer" in rendered
    assert undelivered["id"] != delivered["id"]


def test_turn_select_reads_a_vault_without_delivered_at(tmp_path):
    path = tmp_path / "pre-delivery.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE conversation_turns (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            supersedes_turn_id TEXT,
            answers_turn_id TEXT,
            client_disconnected_at TEXT,
            attachments_json TEXT
        );
        INSERT INTO conversation_turns
            (id, conversation_id, sequence, role, content, created_at)
        VALUES ('old-turn', 'c1', 1, 'assistant', 'old answer', 't0');
    """)

    assert "NULL AS delivered_at" in chat._turn_select(conn)
    assert "NULL AS progress_id" in chat._turn_select(conn)
    assert "NULL AS mode" in chat._turn_select(conn)
    turns = chat._turn_rows(conn, "c1")
    assert turns[0]["id"] == "old-turn"
    assert turns[0]["delivered_at"] is None
    conn.close()


def test_conversation_turns_rebuild_preserves_delivered_at(tmp_path):
    path = tmp_path / "rebuild.db"
    conn = db.connect(path)
    db.init_db(conn)
    conn.execute("PRAGMA foreign_keys = OFF")
    for trigger in (
        "conversation_turns_no_update",
        "conversation_turns_no_delete",
        "conversation_turns_supersedes_same_conversation",
        "conversation_turns_answers_same_conversation",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("ALTER TABLE conversation_turns RENAME TO ct_current")
    conn.executescript("""
        CREATE TABLE conversation_turns (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id),
            sequence INTEGER NOT NULL CHECK (sequence > 0),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            supersedes_turn_id TEXT,
            answers_turn_id TEXT,
            client_disconnected_at TEXT,
            delivered_at TEXT,
            attachments_json TEXT,
            UNIQUE (conversation_id, sequence),
            CHECK (answers_turn_id IS NULL OR role = 'assistant')
        );
        INSERT INTO conversations (id, created_at, updated_at)
        VALUES ('rebuild-c', 't0', 't0');
        INSERT INTO conversation_turns
            (id, conversation_id, sequence, role, content, created_at,
             client_disconnected_at, delivered_at)
        VALUES ('rebuild-a', 'rebuild-c', 1, 'assistant', 'answer', 't0',
                'disconnect', 'delivery');
    """)
    conn.execute("DROP TABLE ct_current")
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")

    db.init_db(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(conversation_turns)")}
    stored = conn.execute(
        "SELECT delivered_at FROM conversation_turns WHERE id = 'rebuild-a'"
    ).fetchone()[0]
    conn.close()
    assert "delivered_at" in columns
    assert stored == "delivery"


def test_turn_metadata_remains_append_only(vault):
    conversation = chat.create_conversation(vault, conversation_id="immutable-meta")
    answer = _disconnected_answer(
        vault, conversation["id"], "synthetic question", "synthetic answer",
        progress_id="immutable-progress", mode="fallback")

    mutations = [
        ("content", "UPDATE conversation_turns SET content = ? WHERE id = ?",
         ("changed", answer["id"])),
        ("disconnect", "UPDATE conversation_turns "
         "SET client_disconnected_at = NULL WHERE id = ?", (answer["id"],)),
        ("created_at", "UPDATE conversation_turns SET created_at = ? WHERE id = ?",
         ("2026-09-15T00:00:00+00:00", answer["id"])),
    ]
    conn = vault.connect()
    try:
        for _, statement, params in mutations:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement, params)
            conn.rollback()
    finally:
        conn.close()

    delivered = chat.mark_turn_delivered(vault, answer["id"])
    assert delivered["delivered_at"]
    conn = vault.connect()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE conversation_turns SET delivered_at = NULL WHERE id = ?",
                (answer["id"],),
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM conversation_turns WHERE id = ?",
                         (answer["id"],))
        conn.rollback()
    finally:
        conn.close()
