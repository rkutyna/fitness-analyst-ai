# tests/test_tool_args_decode.py
"""A tool call whose arguments are not valid JSON must fail visibly.

The model proposes a tool call; Python decides what runs. A call whose
``arguments`` string cannot be decoded has no defined meaning, so it must not
be dispatched with an empty argument set (a tool whose parameters all have
defaults would run and its result would go back to the model as if the model
had asked for exactly that). The model is told the arguments were invalid and
the tool callable is never invoked.
"""
import json

import httpx
import pytest

from health_advisor import llm

BAD_ARGS = "{not json"


def _drive(monkeypatch, vault, runner):
    """Script two turns: a tool call with undecodable arguments, then a final
    answer. Return (calls, request_bodies, output)."""
    calls = []

    def probe(limit=5):
        calls.append(limit)
        return {"ok": True, "limit": limit}

    schema = {"type": "function", "function": {"name": "probe"}}
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(200, json={
                "message": {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {
                                "name": "probe", "arguments": BAD_ARGS}}]},
                "prompt_eval_count": 5})
        return httpx.Response(200, json={"message": {"content": "DONE"},
                                         "prompt_eval_count": 6})

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    out = runner(probe, schema)
    return calls, bodies, out


def _tool_messages(body):
    return [m for m in body["messages"] if m.get("role") == "tool"]


def test_tool_loop_does_not_run_a_tool_on_undecodable_arguments(
        monkeypatch, vault):
    def runner(probe, schema):
        monkeypatch.setattr(
            llm, "_registry", lambda ctx, **k: {"probe": (probe, schema)})
        return llm.tool_loop("question", ctx=vault, tools=[schema],
                             max_turns=4, claim_instructions=None)

    calls, bodies, out = _drive(monkeypatch, vault, runner)

    assert calls == []
    assert len(bodies) == 2
    messages = _tool_messages(bodies[1])
    assert len(messages) == 1
    assert json.loads(messages[0]["content"]) == {
        "error": "arguments were not valid JSON"}
    assert out == "DONE"


def test_research_loop_does_not_run_a_tool_on_undecodable_arguments(
        monkeypatch, vault):
    def runner(probe, schema):
        monkeypatch.setattr(llm, "_registry", lambda ctx, **k: {})
        return llm.research_loop(
            "investigate", ctx=vault, extra_tools={"probe": (probe, schema)},
            compact_state=lambda: "STATE", max_turns=5)

    calls, bodies, out = _drive(monkeypatch, vault, runner)

    assert calls == []
    assert len(bodies) == 2
    messages = _tool_messages(bodies[1])
    assert len(messages) == 1
    assert json.loads(messages[0]["content"]) == {
        "error": "arguments were not valid JSON"}
    assert out == "DONE"


def test_valid_string_arguments_still_dispatch(monkeypatch, vault):
    """A successful call is unchanged: decodable arguments reach the tool."""
    calls = []

    def probe(limit=5):
        calls.append(limit)
        return {"ok": True}

    schema = {"type": "function", "function": {"name": "probe"}}
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json={
                "message": {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {
                                "name": "probe",
                                "arguments": json.dumps({"limit": 9})}}]},
                "prompt_eval_count": 5})
        return httpx.Response(200, json={"message": {"content": "DONE"},
                                         "prompt_eval_count": 6})

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(
        llm, "_registry", lambda ctx, **k: {"probe": (probe, schema)})
    llm.tool_loop("question", ctx=vault, tools=[schema], max_turns=4,
                  claim_instructions=None)
    assert calls == [9]
