"""The agentic loops and the ask path report a provider HTTP refusal as a typed
outcome (health_advisor#697).

A refused OpenRouter key (401/402/403) is ``provider_key_refused``, which is not
retryable: research_loop stops after one request. A 429 is ``rate_limited``.
Every request is served by an httpx.MockTransport; nothing reaches a network.
"""
from __future__ import annotations

import httpx
import pytest

from health_advisor import chat, llm
from tests.conftest import seed_metric

SECRET_MARKER = "SECRET-BODY-MARKER"


@pytest.fixture
def openrouter(monkeypatch, vault):
    """Approved openrouter backend whose every request gets ``code``."""
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "coreweave/fp8")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_SORT", "")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")
    monkeypatch.setattr(llm, "_registry", lambda ctx, include=None: {})
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    llm._reset_loop_status()

    def serve(code):
        """Answer every request with ``code``; return the list of requests seen."""
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(code, json={"error": {"message": SECRET_MARKER}})

        monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
        return seen

    yield serve
    llm._reset_loop_status()


@pytest.mark.parametrize("code", [401, 402, 403])
def test_tool_loop_refused_key_is_provider_key_refused(openrouter, vault, code):
    openrouter(code)
    answer = llm.tool_loop("gather", ctx=vault, tools=[], max_turns=4)
    assert answer == ""
    status = llm.last_loop_status()
    assert status["outcome"] == "provider_key_refused"
    assert status["detail"] == f"HTTP {code}"
    assert SECRET_MARKER not in repr(status)


def test_tool_loop_rate_limit_is_rate_limited(openrouter, vault):
    openrouter(429)
    assert llm.tool_loop("gather", ctx=vault, tools=[], max_turns=4) == ""
    status = llm.last_loop_status()
    assert status["outcome"] == "rate_limited"
    assert status["detail"] == "HTTP 429"
    assert SECRET_MARKER not in repr(status)


def test_tool_loop_server_error_stays_tool_loop_error(openrouter, vault):
    openrouter(500)
    assert llm.tool_loop("gather", ctx=vault, tools=[], max_turns=4) == ""
    status = llm.last_loop_status()
    assert status["outcome"] == "tool_loop_error"
    assert SECRET_MARKER not in repr(status)


def test_research_loop_refused_key_stops_after_one_request(openrouter, vault):
    seen = openrouter(401)
    out = llm.research_loop("investigate", ctx=vault, extra_tools={},
                            compact_state=lambda: "S", max_turns=10)
    assert out == ""
    status = llm.last_loop_status()
    assert status["outcome"] == "provider_key_refused"
    assert status["detail"] == "HTTP 401"
    assert SECRET_MARKER not in repr(status)
    # A refused key is not retryable: exactly one request, not the retry budget.
    assert len(seen) == 1


def test_research_loop_rate_limit_retries_then_announces_rate_limited(
        openrouter, vault):
    seen = openrouter(429)
    llm.research_loop("investigate", ctx=vault, extra_tools={},
                      compact_state=lambda: "S", max_turns=10)
    status = llm.last_loop_status()
    assert status["outcome"] == "rate_limited"
    assert status["detail"] == "HTTP 429"
    # The 429 is retried up to the turn-error budget, then reported.
    assert len(seen) == 1 + llm.TURN_ERROR_BUDGET


def test_complete_rate_limit_is_rate_limited(openrouter, vault):
    openrouter(429)
    assert llm.complete("prompt") == ""
    status = llm.last_complete_status()
    assert status["outcome"] == "rate_limited"
    assert status["detail"] == "HTTP 429"
    assert status["response_received"] is False


@pytest.fixture
def seeded_vault(conn):
    seed_metric(conn, "fixture_metric", "2026-01-01", [1])


@pytest.mark.parametrize("code", [401, 402])
def test_ask_with_refused_key_says_so(openrouter, vault, seeded_vault, code):
    openrouter(code)
    result = chat.answer_question(vault, "How has my resting heart rate changed?")
    assert result["mode"] == "fallback"
    assert result["verification"]["cause"] == "provider_key_refused"
    assert "refused this server's access key" in result["text"]
    assert SECRET_MARKER not in repr(result)


def test_ask_with_rate_limit_is_transport_failed(openrouter, vault, seeded_vault):
    openrouter(429)
    result = chat.answer_question(vault, "How has my resting heart rate changed?")
    assert result["mode"] == "fallback"
    assert result["verification"]["cause"] == "transport_failed"
    assert "refused this server's access key" not in result["text"]


def test_research_loop_refused_key_on_the_finalize_call(openrouter, vault,
                                                         monkeypatch):
    """The wrap-up request after the last turn sits outside the per-turn retry;
    a refused key there is still typed, not a generic research_loop_error."""
    openrouter(401)
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            # Turn 1: the model asks for a tool, so the loop runs out of turns
            # unfinalized and makes the separate finalize request.
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": "",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {
                        "name": "no_such_tool", "arguments": "{}"}}]}}],
                "usage": {"prompt_tokens": 10}})
        return httpx.Response(401, json={"error": {"message": SECRET_MARKER}})

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    out = llm.research_loop("investigate", ctx=vault, extra_tools={},
                            compact_state=lambda: "S", max_turns=1)
    assert out == ""
    status = llm.last_loop_status()
    assert status["outcome"] == "provider_key_refused"
    assert status["detail"] == "HTTP 401"
    assert SECRET_MARKER not in repr(status)
    assert len(seen) == 2
