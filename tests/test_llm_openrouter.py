import json

import httpx
import pytest

from health_advisor import llm


@pytest.fixture
def openrouter(monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")

    def set_handler(handler):
        monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))

    return set_handler


def test_complete_happy_path_sends_openrouter_request(openrouter):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "  answer  "}}],
            "provider": "CoreWeave"})

    openrouter(handler)
    assert llm.complete("prompt") == "answer"
    assert seen["url"] == f"{llm.OPENROUTER_URL}/chat/completions"
    assert seen["authorization"] == "Bearer unit-test-key"
    assert seen["body"]["model"] == llm.OPENROUTER_MODEL
    assert seen["body"]["reasoning"] == {"enabled": False}


def test_openrouter_reasoning_setting_controls_payload(openrouter, monkeypatch):
    seen = _capture(openrouter)
    assert llm.complete("prompt") == "answer"
    assert seen["body"]["reasoning"] == {"enabled": False}

    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "on")
    seen = _capture(openrouter)
    assert llm.complete("prompt") == "answer"
    assert seen["body"]["reasoning"] == {"enabled": True}


def test_openrouter_reasoning_controls_timeout_even_when_think_is_true(
        openrouter, monkeypatch):
    timeouts = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "answer"}}],
                "provider": "CoreWeave"})

    monkeypatch.setattr(llm, "_client",
                        lambda timeout: (timeouts.append(timeout) or Client()))
    llm.complete("prompt", think=True)
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "on")
    llm.complete("prompt", think=False)
    assert timeouts == [llm.TIMEOUT_BRIEF, llm.TIMEOUT_THINK]


def test_complete_strips_think_block(openrouter):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {
                "content": "<think>private reasoning</think>Real answer."}}],
            "provider": "CoreWeave"})

    openrouter(handler)
    assert llm.complete("prompt") == "Real answer."


def test_complete_non200_returns_empty(openrouter):
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})

    openrouter(handler)
    assert llm.complete("prompt") == ""


def test_complete_malformed_json_returns_empty(openrouter):
    def handler(request):
        return httpx.Response(200, content=b"not json")

    openrouter(handler)
    assert llm.complete("prompt") == ""


def test_complete_empty_or_missing_choices_returns_empty(openrouter):
    for body in ({"choices": [], "provider": "CoreWeave"},
                 {"provider": "CoreWeave"}):
        def handler(request, body=body):
            return httpx.Response(200, json=body)

        openrouter(handler)
        assert llm.complete("prompt") == ""


def test_complete_status_distinguishes_empty_content_from_real_short_answer(openrouter):
    replies = iter(["", "brief"])

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": next(replies)}}],
            "provider": "CoreWeave"})

    openrouter(handler)
    assert llm.complete("prompt") == ""
    empty_status = llm.last_complete_status()
    assert empty_status["outcome"] == "empty_response"
    assert empty_status["response_received"] is False

    assert llm.complete("prompt") == "brief"
    brief_status = llm.last_complete_status()
    assert brief_status["outcome"] == "success"
    assert brief_status["response_received"] is True
    assert brief_status["call_id"] != empty_status["call_id"]


def test_openrouter_credits_reports_balance_without_bypassing_approval(openrouter,
                                                                        monkeypatch):
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "coreweave/fp8")
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url),
                     request.headers["Authorization"]))
        return httpx.Response(200, json={
            "data": {"total_credits": 15, "total_usage": 5.123183311}})

    openrouter(handler)
    assert llm.openrouter_credits() == {
        "total_credits": 15.0, "total_usage": 5.123183311,
        "remaining": 9.876816689,
    }
    assert seen == [("GET", f"{llm.OPENROUTER_URL}/credits",
                     "Bearer unit-test-key")]


def test_complete_transport_error_returns_empty(openrouter):
    def handler(request):
        raise httpx.ConnectError("down")

    openrouter(handler)
    assert llm.complete("prompt") == ""


def test_complete_refuses_an_unapproved_served_provider_and_announces(
        openrouter, monkeypatch):
    events = []

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "leaked"}}],
            "provider": "Groq"})

    original_announce = llm._announce

    def announce(event, detail="", **kwargs):
        events.append((event, detail))
        original_announce(event, detail, **kwargs)

    monkeypatch.setattr(llm, "_announce", announce)
    openrouter(handler)
    assert llm.complete("prompt") == ""
    assert events and events[0][0] == "openrouter_provider_mismatch"
    assert "Groq" in events[0][1]
    assert llm.last_complete_status()["outcome"] == "backend_error"


def test_complete_without_api_key_makes_no_request(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "unexpected"}}]})

    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")
    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    assert llm.complete("prompt") == ""
    assert calls == []


def test_tool_loop_openrouter_unpinned_makes_no_request(monkeypatch):
    """The tool path is live on openrouter since #128, but D15 still gates it:
    an unpinned process is a different thing with the same name, so it sends
    nothing — and, unlike before, says so. The working path lives in
    tests/test_llm_openrouter_tools.py."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={})

    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")
    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    assert llm.tool_loop("prompt", ctx=None, tools=[]) == ""
    assert calls == []
    assert llm.last_loop_status()["outcome"] == "openrouter_not_approved"


def _capture(openrouter):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "answer"}}],
            "provider": "CoreWeave"})

    openrouter(handler)
    return seen


def test_no_provider_block_when_unconfigured(openrouter, monkeypatch):
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_SORT", "")
    seen = _capture(openrouter)
    assert llm.complete("prompt") == "answer"
    assert "provider" not in seen["body"]


def test_provider_sort_is_forwarded(openrouter, monkeypatch):
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_SORT", "throughput")
    seen = _capture(openrouter)
    llm.complete("prompt")
    assert seen["body"]["provider"] == {"sort": "throughput"}


def test_explicit_provider_list_wins_and_disables_fallbacks(openrouter, monkeypatch):
    # A pin that can silently fail over to an unpinned provider is not a pin,
    # and what is pinned here is who receives the health data.
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", " DeepInfra , Together ")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_SORT", "throughput")
    seen = _capture(openrouter)
    llm.complete("prompt")
    assert seen["body"]["provider"] == {
        "order": ["DeepInfra", "Together"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
