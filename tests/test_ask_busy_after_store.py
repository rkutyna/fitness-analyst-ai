"""/v1/ask never answers a retryable 503 once the question is stored.

downstream #734: a busy vault answers 503 + Retry-After, which tells a client
it may re-send. For /v1/ask that is only true before the question is stored:
after it, a re-send stores the question twice and pays for a second model run.
Each test below makes one post-store step meet a busy vault; before the fix
each of them answered 503 with the question already stored.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from health_advisor import analyst_corpus, chat, db, receiver

HEADERS = {"x-health-secret": "s3cret"}
QUESTION = "How did I sleep this week?"


def _busy(*_a, **_k):
    raise sqlite3.OperationalError("database is locked")


def _answer(*_a, **_k):
    return {"text": "Fine.", "mode": "narration", "tool_trace": [],
            "figures": [],
            "verification": {"ok": True, "grounded": True, "unsupported": [],
                             "reason": "", "figures_verified": 0,
                             "figures_total": 0}}


@pytest.fixture
def app(vault, conn, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "s3cret")
    monkeypatch.setattr(db, "BUSY_TIMEOUT_MS", 300)
    monkeypatch.setattr(chat, "answer_question", _answer)
    return receiver.create_app(vault)


def _turns(path) -> list[tuple[str, str, str | None]]:
    c = sqlite3.connect(path)
    try:
        return c.execute("SELECT role, content, mode FROM conversation_turns "
                         "ORDER BY rowid").fetchall()
    finally:
        c.close()


def _ask(app, *, consumer_handler=False):
    if consumer_handler:
        # A consumer mounts its own busy handler after create_app, replacing
        # the engine's: the route must not rely on the handler to tell a
        # stored question from an unstored one.
        async def consumer_busy(request, exc):
            return JSONResponse(status_code=503, content={"detail": "busy"},
                                headers={"Retry-After": "5"})
        app.app.add_exception_handler(sqlite3.OperationalError, consumer_busy)
    with TestClient(app, raise_server_exceptions=False) as client:
        return client.post("/v1/ask", json={"question": QUESTION},
                           headers=HEADERS)


def _question_stored_once(path) -> list:
    turns = _turns(path)
    assert [t for t in turns if t[0] == "user"] == [("user", QUESTION, None)]
    return turns


@pytest.mark.parametrize("consumer_handler", [False, True])
def test_busy_while_answering_stores_and_returns_a_fallback_answer(
        app, vault_path, monkeypatch, consumer_handler):
    monkeypatch.setattr(chat, "answer_question", _busy)
    r = _ask(app, consumer_handler=consumer_handler)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "fallback"
    assert body["text"] == receiver.ASK_VAULT_BUSY_ANSWER
    assert body["verification"]["cause"] == receiver.ASK_VAULT_BUSY_CAUSE
    assert body["verification"]["ok"] is False
    turns = _question_stored_once(vault_path)
    assert turns[-1] == ("assistant", receiver.ASK_VAULT_BUSY_ANSWER, "fallback")


@pytest.mark.parametrize("consumer_handler", [False, True])
def test_busy_while_answering_and_storing_the_answer_is_a_plain_500(
        app, vault_path, monkeypatch, consumer_handler):
    monkeypatch.setattr(chat, "answer_question", _busy)
    monkeypatch.setattr(chat, "append_turn", _busy)
    r = _ask(app, consumer_handler=consumer_handler)
    assert r.status_code == 500, r.text
    assert "retry-after" not in r.headers
    assert r.json() == {"detail": receiver.ASK_ANSWER_NOT_STORED_DETAIL}
    assert _question_stored_once(vault_path)[-1][0] == "user"


@pytest.mark.parametrize("consumer_handler", [False, True])
def test_busy_freshness_returns_the_stored_answer_with_null_freshness(
        app, vault_path, monkeypatch, consumer_handler):
    monkeypatch.setattr(receiver, "_ask_freshness", _busy)
    r = _ask(app, consumer_handler=consumer_handler)
    assert r.status_code == 200, r.text
    assert r.json()["freshness"] is None
    assert r.json()["text"] == "Fine."
    assert _question_stored_once(vault_path)[-1] == ("assistant", "Fine.",
                                                     "narration")


def test_freshness_errors_other_than_busy_still_fail(app, monkeypatch):
    def broken(*_a, **_k):
        raise sqlite3.OperationalError("no such table: nowhere")
    monkeypatch.setattr(receiver, "_ask_freshness", broken)
    assert _ask(app).status_code == 500


@pytest.mark.parametrize("consumer_handler", [False, True])
def test_any_other_busy_step_after_the_question_is_stored_is_a_plain_500(
        app, vault_path, monkeypatch, consumer_handler):
    # A step the route does not handle by name (here the evidence gate, which
    # runs between storing the question and answering it).
    monkeypatch.setattr(analyst_corpus, "evidence_gate", _busy)
    r = _ask(app, consumer_handler=consumer_handler)
    assert r.status_code == 500, r.text
    assert "retry-after" not in r.headers
    assert r.json() == {"detail": receiver.ASK_ANSWER_NOT_STORED_DETAIL}
    assert _question_stored_once(vault_path)[-1][0] == "user"


@pytest.mark.parametrize("consumer_handler", [False, True])
def test_a_busy_step_after_the_answer_is_stored_says_so(
        app, vault_path, monkeypatch, consumer_handler):
    monkeypatch.setattr(receiver, "_ask_freshness_or_none", _busy)
    r = _ask(app, consumer_handler=consumer_handler)
    assert r.status_code == 500, r.text
    assert "retry-after" not in r.headers
    assert r.json() == {"detail": receiver.ASK_AFTER_ANSWER_DETAIL}
    assert _question_stored_once(vault_path)[-1][0] == "assistant"


@pytest.mark.parametrize("consumer_handler", [False, True])
def test_busy_before_the_question_is_stored_is_still_503(
        app, vault_path, monkeypatch, consumer_handler):
    monkeypatch.setattr(chat, "start_question", _busy)
    r = _ask(app, consumer_handler=consumer_handler)
    assert r.status_code == 503, r.text
    assert "retry-after" in r.headers
    assert _turns(vault_path) == []
