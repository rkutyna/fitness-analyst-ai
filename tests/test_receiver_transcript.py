"""Concurrent ask pairing and observed client disconnects."""
from __future__ import annotations

import asyncio
import threading
import time

import httpx

from health_advisor import chat, receiver


def test_concurrent_asks_render_question_answer_pairs_over_20_repetitions(
        monkeypatch, vault):
    """Three concurrent asks are paired correctly in every one of 20 runs."""
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    barrier = threading.Barrier(3)

    def fake_answer(ctx, question, **kwargs):
        barrier.wait(timeout=5)
        time.sleep((2 - int(question[-1])) * 0.01)
        return {
            "text": f"answer-for-{question}", "mode": "fallback",
            "tool_trace": [], "verification": {},
        }

    monkeypatch.setattr(chat, "answer_question", fake_answer)

    async def exercise() -> int:
        app = receiver.create_app(vault)
        transport = httpx.ASGITransport(app=app)
        passed = 0
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for repetition in range(20):
                conversation = chat.create_conversation(
                    vault, conversation_id=f"concurrent-{repetition}")
                questions = [f"question-{i}" for i in range(3)]
                responses = await asyncio.gather(*(
                    client.post(
                        "/v1/ask",
                        json={"conversation_id": conversation["id"],
                              "question": question},
                        headers={"x-health-secret": "ask-secret"},
                    ) for question in questions
                ))
                assert all(response.status_code == 200 for response in responses)
                turns = chat.list_turns(vault, conversation["id"])
                rendered = chat._render_history(turns)
                lines = [line for line in rendered.splitlines()
                         if line.startswith(("USER:", "ASSISTANT:"))]
                user_turns = [turn for turn in turns if turn["role"] == "user"]
                expected = []
                for turn in user_turns:
                    expected.extend([
                        f"USER: {turn['content']}",
                        f"ASSISTANT: answer-for-{turn['content']}",
                    ])
                assert lines == expected
                assert all(
                    turn["answers_turn_id"] in {question["id"] for question in user_turns}
                    for turn in turns if turn["role"] == "assistant"
                )
                passed += 1
        return passed

    assert asyncio.run(exercise()) == 20


def test_disconnect_is_recorded_as_an_event_and_derived_from_rendering(
        monkeypatch, vault):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")

    def fake_answer(ctx, question, **kwargs):
        return {
            "text": "answer not observed by the client", "mode": "fallback",
            "tool_trace": [], "verification": {},
        }

    monkeypatch.setattr(chat, "answer_question", fake_answer)

    async def exercise():
        app = receiver.create_app(vault)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/ask", json={"question": "question with disconnect"},
                headers={"x-health-secret": "ask-secret"})
        return response

    # The endpoint asks Request.is_disconnected() after the answer. This is an
    # async callable, matching Starlette's interface.
    async def _disconnected(request):
        return True

    monkeypatch.setattr(receiver.Request, "is_disconnected", _disconnected)
    response = asyncio.run(exercise())
    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    turns = chat.list_turns(vault, conversation_id)
    answer = next(turn for turn in turns if turn["role"] == "assistant")
    assert answer["client_disconnected_at"]
    assert "undelivered" not in answer
    assert "ASSISTANT: answer not observed by the client" not in chat._render_history(turns)
