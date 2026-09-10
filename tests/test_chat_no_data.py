from __future__ import annotations

import re

from fastapi.testclient import TestClient

from health_advisor import chat, llm, receiver
from tests.conftest import seed_metric


def _raise_if_called(*args, **kwargs):
    raise AssertionError("the model must not be called for an empty vault")


def test_empty_vault_returns_python_status_without_model(monkeypatch, vault, conn):
    monkeypatch.setattr(llm, "tool_loop", _raise_if_called)

    result = chat.answer_question(vault, "Am I ready for a workout?",
                                  as_of="2026-09-09")

    assert result["mode"] == "status"
    assert result["verification"] == {
        "ok": True,
        "grounded": True,
        "unsupported": [],
        "figures_total": 0,
        "figures_verified": 0,
        "tool_calls": 0,
        "cause": "no_data_yet",
    }
    assert result["tool_trace"] == []
    assert result["days_of_history"] == 0
    assert "readiness starts after" in result["text"]
    assert not re.search(r"\d+\s*min", result["text"], re.IGNORECASE)
    assert "RPE" not in result["text"]
    assert "warm-up" not in result["text"]
    assert "session" not in result["text"]


def test_empty_vault_without_as_of_also_skips_model(monkeypatch, vault, conn):
    monkeypatch.setattr(llm, "tool_loop", _raise_if_called)

    result = chat.answer_question(vault, "Am I ready for a workout?")

    assert result["mode"] == "status"
    assert result["verification"]["cause"] == "no_data_yet"
    assert result["verification"]["tool_calls"] == 0


def test_one_daily_metric_row_takes_normal_model_path(monkeypatch, vault, conn):
    seed_metric(conn, "step_count", "2026-09-09", [1000])
    calls = []

    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "canned answer",
    )

    chat.answer_question(vault, "How am I doing?", as_of="2026-09-09")

    assert calls


def test_receiver_records_the_deterministic_answer_turn(monkeypatch, vault):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    monkeypatch.setattr(llm, "tool_loop", _raise_if_called)

    with TestClient(receiver.create_app(vault)) as client:
        response = client.post(
            "/v1/ask",
            json={"question": "Am I ready for a workout?", "as_of": "2026-09-09"},
            headers={"x-health-secret": "ask-secret"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "status"
    turns = chat.list_turns(vault, body["conversation_id"])
    assert [turn["role"] for turn in turns] == ["user", "assistant"]
    assert turns[1]["content"] == body["text"]


def test_no_data_cause_is_closed_and_accepted_by_cause_helper():
    assert "no_data_yet" in chat.ASK_CAUSES
    assert chat._ask_cause(
        {"ok": True}, ledger=[], loop_outcomes=[], no_data_yet=True
    ) == "no_data_yet"
