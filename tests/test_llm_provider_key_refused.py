"""complete() tells a refused OpenRouter key apart from other backend failures.

HTTP 401/402/403 from OpenRouter (expired, revoked, out of credit) record the
outcome ``provider_key_refused``; any other status records ``backend_error``.
The detail carries the status code only, never the response body.
"""

import httpx
import pytest

from health_advisor import llm

SECRET_MARKER = "SECRET-BODY-MARKER"


@pytest.fixture
def openrouter(monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")

    def set_handler(handler):
        monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))

    return set_handler


@pytest.mark.parametrize("code", [401, 402, 403])
def test_refused_key_is_provider_key_refused(openrouter, code):
    openrouter(lambda request: httpx.Response(code, json={"error": "nope"}))
    assert llm.complete("prompt") == ""
    status = llm.last_complete_status()
    assert status["outcome"] == "provider_key_refused"
    assert status["detail"] == f"HTTP {code}"
    assert status["response_received"] is False
    assert status["request_made"] is True


def test_server_error_stays_backend_error(openrouter):
    openrouter(lambda request: httpx.Response(500, json={"error": "boom"}))
    assert llm.complete("prompt") == ""
    status = llm.last_complete_status()
    assert status["outcome"] == "backend_error"
    assert status["detail"] == "HTTP 500"


def test_refusal_body_never_reaches_status(openrouter):
    openrouter(lambda request: httpx.Response(
        401, json={"error": {"message": SECRET_MARKER}},
        headers={"X-Echo": SECRET_MARKER}))
    assert llm.complete("prompt") == ""
    status = llm.last_complete_status()
    assert status["outcome"] == "provider_key_refused"
    assert SECRET_MARKER not in repr(status)


def test_success_still_reads_success(openrouter):
    openrouter(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": "answer"}}],
        "provider": "CoreWeave"}))
    assert llm.complete("prompt") == "answer"
    status = llm.last_complete_status()
    assert status["outcome"] == "success"
    assert status["response_received"] is True
