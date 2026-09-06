"""Server-side recovery of answers produced after an ask client disconnects."""

import sqlite3

from fastapi.testclient import TestClient

from health_advisor import chat, db, receiver


HEADERS = {"x-health-secret": "ask-secret"}


def _disconnected_answer(vault, conversation_id, question, answer):
    question_turn = chat.append_turn(vault, conversation_id, "user", question)
    return chat.append_turn(
        vault,
        conversation_id,
        "assistant",
        answer,
        answers_turn_id=question_turn["id"],
        client_disconnected_at="2026-09-06T12:00:00+00:00",
    )


def test_get_returns_a_disconnected_undelivered_turn(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    conversation = chat.create_conversation(vault, conversation_id="recover-one")
    answer = _disconnected_answer(vault, conversation["id"], "q", "answer")

    with TestClient(receiver.create_app(vault)) as client:
        response = client.get("/v1/ask/undelivered", headers=HEADERS)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == answer["id"]
    assert body["conversation_id"] == conversation["id"]
    assert body["answers_turn_id"] == answer["answers_turn_id"]
    assert body["content"] == "answer"
    assert body["client_disconnected_at"]
    assert body["delivered_at"] is None


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


def test_get_returns_oldest_undelivered_turn(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    conversation = chat.create_conversation(vault, conversation_id="recover-order")
    first = _disconnected_answer(vault, conversation["id"], "q1", "first")
    second = _disconnected_answer(vault, conversation["id"], "q2", "second")

    with TestClient(receiver.create_app(vault)) as client:
        response = client.get("/v1/ask/undelivered", headers=HEADERS)

    assert response.status_code == 200, response.text
    assert response.json()["id"] == first["id"]
    assert response.json()["id"] != second["id"]


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
