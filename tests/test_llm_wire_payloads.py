"""The request bodies llm.py puts on the wire, pinned byte for byte.

Every POST site in ``llm.py`` builds a JSON body: the Ollama ``/api/chat``
payload in ``complete``, ``tool_loop`` and ``research_loop``, and the
OpenAI-dialect payload in ``complete`` and ``_openrouter_post`` (OpenRouter
and the bring-your-own endpoint). The ``reasoning`` form and the ``provider``
pin are data-egress controls, so a refactor of how those bodies are assembled
must not change a single byte of what a provider receives.

``fixtures/llm_wire_payloads_golden.json`` holds, for a fixed set of synthetic
inputs, the URL and the raw request body text of every POST. The test replays
the same inputs through a mocked transport and compares the raw text, so key
order, separators and omitted-versus-null fields all count. Nothing here
touches a network, a real key or real data.

To re-record after an intended wire change, run the test with
``HA_RECORD_WIRE_GOLDEN=1`` and review the diff of the golden file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from health_advisor import llm

GOLDEN = Path(__file__).parent / "fixtures" / "llm_wire_payloads_golden.json"

FLASH = "deepseek/deepseek-v4-flash-0731"
PRO = "deepseek/deepseek-v4-pro"
PINNED = "coreweave/fp8"
PINNED_MANY = "coreweave/fp8,together"
PRO_PINNED = "novita/fp8,deepinfra/fp8"

TOOL_SCHEMA = {"type": "function", "function": {
    "name": "echo", "description": "echo", "parameters": {
        "type": "object", "properties": {"a": {"type": "integer"}}}}}
CUSTOM_OPTS = {"temperature": 0.3, "top_p": 0.8, "presence_penalty": 0.5,
               "num_predict": 77, "seed": 1}


def _echo(a=0):
    return {"echo": a}


class Recorder:
    """A transport that records every POST and serves a fixed two-turn script."""

    def __init__(self, dialect: str):
        self.dialect = dialect
        self.posts: list[dict] = []
        self.turn = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.posts.append({"url": str(request.url),
                           "body": request.content.decode("utf-8")})
        self.turn += 1
        first = self.turn == 1
        if self.dialect == "ollama":
            message = {"role": "assistant", "content": "answer"}
            if first:
                message = {"role": "assistant", "content": "",
                           "tool_calls": [{"function": {
                               "name": "echo", "arguments": {"a": 1}}}]}
            return httpx.Response(200, json={"message": message,
                                             "prompt_eval_count": 3})
        message = {"role": "assistant", "content": "answer"}
        if first:
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_a", "type": "function", "function": {
                    "name": "echo", "arguments": "{\"a\": 1}"}}]}
        return httpx.Response(200, json={
            "choices": [{"message": message, "finish_reason": "stop"}],
            "provider": "CoreWeave",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}})


def _configure(monkeypatch, backend: str, *, reasoning="off",
               providers=PINNED, sort="", model=FLASH):
    monkeypatch.setattr(llm, "BACKEND", backend)
    monkeypatch.setattr(llm, "MODEL", "wire-test-local-model")
    monkeypatch.setattr(llm, "KEEP_ALIVE", "7m")
    monkeypatch.setattr(llm, "OLLAMA_URL", "http://ollama.invalid")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", model)
    monkeypatch.setattr(llm, "OPENROUTER_URL", "https://openrouter.invalid/api/v1")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", providers)
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_SORT", sort)
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", reasoning)
    monkeypatch.setattr(llm, "PLAN_MODEL", None)
    monkeypatch.setattr(llm, "PLAN_PROVIDERS", None)
    monkeypatch.setattr(llm, "OPENAI_COMPAT_URL", "http://compat.invalid/v1")
    monkeypatch.setattr(llm, "OPENAI_COMPAT_MODEL", "compat-model")
    monkeypatch.setattr(llm, "OPENAI_COMPAT_API_KEY", "")
    if backend == llm.OPENAI_COMPATIBLE:
        monkeypatch.setattr(llm, "assert_backend_approved", lambda: None)
    monkeypatch.setattr(llm, "_registry", lambda ctx, **kw: {})
    llm._reset_loop_status()


def _run_complete(monkeypatch, backend, kwargs, **config):
    _configure(monkeypatch, backend, **config)
    recorder = Recorder("ollama" if backend == "ollama" else "openai")
    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(recorder))
    recorder.turn = 1   # serve the plain answer, not the scripted tool call
    llm.complete("prompt text", **kwargs)
    return recorder.posts


def _run_tool_loop(monkeypatch, backend, kwargs, **config):
    _configure(monkeypatch, backend, **config)
    monkeypatch.setattr(llm, "_registry",
                        lambda ctx, **kw: {"echo": (_echo, TOOL_SCHEMA)})
    recorder = Recorder("ollama" if backend == "ollama" else "openai")
    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(recorder))
    llm.tool_loop("loop prompt", ctx=None, tools=[TOOL_SCHEMA],
                  claim_instructions="[claims block]", **kwargs)
    return recorder.posts


def _run_research_loop(monkeypatch, backend, kwargs, **config):
    _configure(monkeypatch, backend, **config)
    recorder = Recorder("ollama" if backend == "ollama" else "openai")
    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(recorder))
    # The claim-channel prose is prompt policy, not wire assembly.
    monkeypatch.setattr(llm, "_RESEARCH_CLAIM_INSTRUCTIONS", "[claims block]")
    llm.research_loop("research prompt", ctx=None,
                      extra_tools={"echo": (_echo, TOOL_SCHEMA)},
                      compact_state=lambda: "STATE", max_turns=4, **kwargs)
    return recorder.posts


def _run_openrouter_post(monkeypatch, backend, kwargs, **config):
    _configure(monkeypatch, backend, **config)
    recorder = Recorder("openai")
    recorder.turn = 1
    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(recorder))
    messages = [{"role": "user", "content": "direct"},
                {"role": "assistant", "content": "", "_private": "dropped"}]
    llm._openrouter_post(messages, **kwargs)
    return recorder.posts


OR = "openrouter"
OC = llm.OPENAI_COMPATIBLE

SCENARIOS = {
    # --- complete, Ollama
    "complete_ollama_default": (_run_complete, "ollama", {}, {}),
    "complete_ollama_think": (_run_complete, "ollama", {"think": True}, {}),
    "complete_ollama_options": (
        _run_complete, "ollama", {"options": CUSTOM_OPTS}, {}),
    # --- complete, OpenRouter
    "complete_or_off_pinned": (_run_complete, OR, {}, {}),
    "complete_or_on_two_pins": (
        _run_complete, OR, {}, {"reasoning": "on", "providers": PINNED_MANY}),
    "complete_or_low_unpinned": (
        _run_complete, OR, {}, {"reasoning": "low", "providers": ""}),
    "complete_or_sort_and_max_tokens": (
        _run_complete, OR, {"max_tokens": 123, "options": CUSTOM_OPTS},
        {"sort": "throughput"}),
    "complete_or_plan_flow": (
        _run_complete, OR, {"flow": "plan"}, {"model": PRO}),
    # --- complete, bring-your-own endpoint
    "complete_compat_default": (_run_complete, OC, {}, {}),
    "complete_compat_max_tokens": (
        _run_complete, OC, {"max_tokens": 55, "options": CUSTOM_OPTS}, {}),
    # --- tool_loop
    "tool_loop_ollama": (_run_tool_loop, "ollama", {"think": True}, {}),
    "tool_loop_ollama_nothink": (_run_tool_loop, "ollama", {"think": False}, {}),
    "tool_loop_or_off": (_run_tool_loop, OR, {}, {}),
    "tool_loop_or_on_max_tokens": (
        _run_tool_loop, OR, {"max_tokens": 500},
        {"reasoning": "on", "providers": PINNED_MANY}),
    "tool_loop_or_low_unpinned": (
        _run_tool_loop, OR, {}, {"reasoning": "low", "providers": ""}),
    "tool_loop_compat": (_run_tool_loop, OC, {}, {}),
    "tool_loop_compat_max_tokens": (
        _run_tool_loop, OC, {"max_tokens": 64}, {}),
    # --- research_loop
    "research_loop_ollama": (_run_research_loop, "ollama", {}, {}),
    "research_loop_ollama_num_ctx": (
        _run_research_loop, "ollama", {"num_ctx": 4096, "think": False}, {}),
    "research_loop_or_off": (_run_research_loop, OR, {}, {}),
    "research_loop_or_on_sorted": (
        _run_research_loop, OR, {},
        {"reasoning": "on", "providers": PINNED_MANY, "sort": "price"}),
    "research_loop_compat": (_run_research_loop, OC, {}, {}),
    # --- _openrouter_post called directly
    "post_or_no_tools": (
        _run_openrouter_post, OR,
        {"tools": None, "timeout": 30}, {}),
    "post_or_tools_default_cap": (
        _run_openrouter_post, OR,
        {"tools": [TOOL_SCHEMA], "timeout": 30, "options": CUSTOM_OPTS},
        {"reasoning": "low"}),
    "post_or_tools_explicit_cap": (
        _run_openrouter_post, OR,
        {"tools": [TOOL_SCHEMA], "timeout": 30, "max_tokens": 9}, {}),
    "post_compat_no_tools_cap": (
        _run_openrouter_post, OC,
        {"tools": None, "timeout": 30, "max_tokens": 11}, {}),
    "post_compat_tools": (
        _run_openrouter_post, OC,
        {"tools": [TOOL_SCHEMA], "timeout": 30}, {}),
}


def _capture(monkeypatch) -> dict:
    captured = {}
    for name, (runner, backend, kwargs, config) in SCENARIOS.items():
        captured[name] = runner(monkeypatch, backend, kwargs, **config)
        assert captured[name], f"{name} sent no request"
    return captured


def test_every_post_body_is_byte_identical_to_the_recorded_wire(monkeypatch):
    captured = _capture(monkeypatch)
    if os.environ.get("HA_RECORD_WIRE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(captured, indent=1, sort_keys=True) + "\n",
                          encoding="utf-8")
        pytest.skip("golden re-recorded")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(golden) == set(captured)
    for name in sorted(golden):
        assert captured[name] == golden[name], name


def test_the_provider_pin_and_reasoning_field_are_on_the_openrouter_wire(
        monkeypatch):
    """Not a regression pin: the recorded bodies must actually exercise the
    egress controls, or a byte comparison of them proves nothing."""
    posts = _run_complete(monkeypatch, OR, {}, reasoning="low",
                          providers=PINNED_MANY)
    body = json.loads(posts[0]["body"])
    assert body["reasoning"] == {"effort": "low"}
    assert body["provider"]["only"] == ["coreweave/fp8", "together"]
    assert body["provider"]["allow_fallbacks"] is False
    posts = _run_complete(monkeypatch, OR, {}, providers="")
    assert "provider" not in json.loads(posts[0]["body"])
    posts = _run_complete(monkeypatch, OC, {})
    body = json.loads(posts[0]["body"])
    assert "provider" not in body and "reasoning" not in body

