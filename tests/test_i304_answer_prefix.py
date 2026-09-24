"""health_advisor#304: the prompt-cache layout of the fact-template answer call.

Captures the real request bodies the OpenRouter transport would send for a
scripted gather + answer turn, and pins the two properties that decide what the
answer call can cache: its instruction block leads and is identical across
questions, and the per-turn material (question, closed fact set) is the tail.
"""
from __future__ import annotations

import json

import httpx
import pytest

from health_advisor import chat, demo, llm
from health_advisor.context import VaultContext

END = "2026-09-09"


def _lcp(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@pytest.fixture
def capture_turn(tmp_path, monkeypatch):
    db_path = tmp_path / "demo.db"
    demo.build_demo_vault(db_path, days=60, end_date=END, read_only=False)
    ctx = VaultContext.local(db_path, user_id="demo-i304")
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setattr(llm, "OPENROUTER_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "coreweave/fp8")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_SORT", "")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")

    def run(question: str) -> list[dict]:
        bodies: list[dict] = []

        def respond(message):
            return httpx.Response(200, json={
                "choices": [{"message": message, "finish_reason": "stop"}],
                "provider": "CoreWeave",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

        def handler(request):
            bodies.append(json.loads(request.content))
            if len(bodies) == 1:
                return respond({"role": "assistant", "content": "", "tool_calls": [{
                    "id": "c1", "type": "function", "function": {
                        "name": "get_impact_volume",
                        "arguments": json.dumps({"start": "2026-08-27",
                                                 "end": END, "by": "week"})}}]})
            if len(bodies) == 2:
                return respond({"role": "assistant", "content": "Gathered."})
            return respond({"role": "assistant",
                            "content": "Your recent running looks steady."})

        monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
        llm._reset_loop_status()
        chat.answer_question(ctx, question, as_of=END)
        return bodies

    return run


def _answer_request(bodies: list[dict]) -> dict:
    # The gather loop is every request carrying tools; the first request
    # without them is the answer call.
    return next(b for b in bodies if not b.get("tools"))


def test_answer_call_leads_with_a_question_independent_instruction_block(
        capture_turn):
    first = _answer_request(capture_turn("How much have I been running?"))
    second = _answer_request(capture_turn("Am I running more than last month?"))
    a = first["messages"][0]["content"]
    b = second["messages"][0]["content"]
    question_at = a.index("USER QUESTION:")
    facts_at = a.index("CLOSED FACT SET (Python ledger facts")

    # The cacheable head is everything before the question; it must be
    # byte-identical across questions and substantial (the full output
    # contract), or a prefix cache has nothing stable to reuse.
    assert question_at >= 2000
    assert _lcp(a, b) >= question_at
    # Per-turn material is the tail, fact set last.
    assert question_at < facts_at
    assert "How much have I been running?" not in a[:question_at]
    assert '"fact|metric=' in a[facts_at:]


def test_answer_call_is_a_fresh_tool_less_prompt(capture_turn):
    bodies = capture_turn("How much have I been running?")
    gather = bodies[0]
    answer = _answer_request(bodies)
    assert gather["tools"] and gather["tool_choice"] == "auto"
    # The narration turn cannot call tools, so the ledger stays closed, and it
    # sees one message: the contract plus the closed fact set, not the raw
    # gather transcript.
    assert "tools" not in answer and "tool_choice" not in answer
    assert [m["role"] for m in answer["messages"]] == ["user"]
    assert bodies.index(answer) == 2
