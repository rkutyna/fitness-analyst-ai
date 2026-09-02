# tests/test_llm_openrouter_tools.py
"""The OpenAI-dialect tool path for the openrouter backend (#128).

The orchestrator gets exactly one live run, so these tests pin the WIRE SHAPE
rather than only the return value: a result keyed by tool name instead of
tool_call_id, or an assistant turn echoed back with a provider's private fields
attached, is a 400 that only appears against the real endpoint.

Nothing here touches the network or the API key — every request is served by an
httpx.MockTransport and the key is a literal fixture string.
"""
import json

import httpx
import pytest

from health_advisor import llm

# Captured before any fixture replaces it, so the one seam test that needs the
# real tool surface can put it back.
_REAL_REGISTRY = llm._registry


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

@pytest.fixture
def openrouter(monkeypatch):
    """An approved-under-D15 openrouter backend with a mocked transport."""
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setattr(llm, "OPENROUTER_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "coreweave/fp8")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_SORT", "")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")
    monkeypatch.setattr(llm, "_registry", lambda ctx, include=None: {})
    llm._LAST_LOOP_STATUS.clear()
    llm._LAST_LOOP_STATUS.update({"outcome": "not_called", "backend": None,
                                  "detail": ""})

    def set_handler(handler):
        monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))

    return set_handler


def _assistant(content="", tool_calls=None, usage=1, extra=None,
               provider="CoreWeave"):
    """One OpenRouter chat-completions response body."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if extra:
        message.update(extra)
    return {"choices": [{"message": message, "finish_reason": "stop"}],
            "provider": provider,
            "usage": {"prompt_tokens": usage, "completion_tokens": 1}}


def _call(name, arguments, call_id="call_0"):
    """One OpenAI-shaped tool call: arguments are a JSON *string* on this wire."""
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def _tool(fn, name="probe"):
    return {name: (fn, {"type": "function", "function": {"name": name}})}


_OLLAMA_ONLY_KEYS = ("think", "options", "keep_alive", "num_ctx", "prompt_eval_count")


def _assert_valid_openai_request(body):
    """Every rule the /chat/completions endpoint enforces on a tool exchange."""
    assert set(_OLLAMA_ONLY_KEYS) & set(body) == set(), body.keys()
    seen_ids, answered = [], []
    for m in body["messages"]:
        assert m["role"] in {"user", "assistant", "tool"}, m
        assert "tool_name" not in m, "tool_name is Ollama's key, not OpenAI's"
        assert not any(k.startswith("_") for k in m), m
        if m["role"] == "assistant" and m.get("tool_calls"):
            assert set(m) <= {"role", "content", "tool_calls"}, m
            for c in m["tool_calls"]:
                assert c.get("id"), c
                seen_ids.append(c["id"])
        if m["role"] == "tool":
            assert m.get("tool_call_id") in seen_ids, m
            answered.append(m["tool_call_id"])
    assert sorted(seen_ids) == sorted(answered), (seen_ids, answered)


# --------------------------------------------------------------------------
# tool_loop — wire shape
# --------------------------------------------------------------------------

def test_tool_loop_posts_openai_dialect_and_keys_results_by_tool_call_id(openrouter):
    """The whole point of the third transport: results are paired by id."""
    bodies, urls, auth = [], [], []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        urls.append(str(request.url))
        auth.append(request.headers["Authorization"])
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {"days": 7}, "call_abc")]))
        return httpx.Response(200, json=_assistant("Steps averaged 5413."))

    openrouter(handler)
    out = llm.tool_loop("question", ctx=None, tools=[{"type": "function"}],
                        max_turns=4)
    assert out == "Steps averaged 5413."
    assert urls[0] == "https://openrouter.ai/api/v1/chat/completions"
    assert auth[0] == "Bearer unit-test-key"
    for body in bodies:
        _assert_valid_openai_request(body)
    second = bodies[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": "",
                          "tool_calls": [_call("probe", {"days": 7}, "call_abc")]}
    assert second[-1]["tool_call_id"] == "call_abc"
    assert second[-1]["name"] == "probe"
    assert json.loads(second[-1]["content"])["error"].startswith("unknown tool")


def test_tool_loop_pairs_two_calls_of_the_same_tool_by_id_not_by_name(openrouter):
    """Keying by name silently mispairs when one turn calls a tool twice —
    the concrete reason the Ollama loop cannot be pointed at OpenRouter."""
    bodies = []
    turns = {"n": 0}
    seen_args = []

    def probe(**kw):
        seen_args.append(kw)
        return {"days": kw["days"], "mean": 100 * kw["days"]}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(tool_calls=[
                _call("probe", {"days": 7}, "call_1"),
                _call("probe", {"days": 30}, "call_2")]))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm._registry = lambda ctx, include=None: _tool(probe)
    out = llm.tool_loop("q", ctx=None, tools=[], max_turns=4)
    assert out == "done"
    _assert_valid_openai_request(bodies[1])
    results = [m for m in bodies[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["call_1", "call_2"]
    assert json.loads(results[0]["content"])["mean"] == 700
    assert json.loads(results[1]["content"])["mean"] == 3000
    assert seen_args == [{"days": 7}, {"days": 30}]


def test_tool_loop_synthesises_an_id_when_the_provider_omits_one(openrouter):
    """Some OpenAI-compatible providers return tool_calls without an id. The
    pair still has to be well-formed, and both halves must use the same one."""
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(tool_calls=[
                {"function": {"name": "probe", "arguments": "{}"}}]))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm._registry = lambda ctx, include=None: _tool(lambda **k: {"ok": 1})
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=4) == "done"
    _assert_valid_openai_request(bodies[1])
    msgs = bodies[1]["messages"]
    assert msgs[-2]["tool_calls"][0]["id"] == msgs[-1]["tool_call_id"]
    assert msgs[-2]["tool_calls"][0]["type"] == "function"


def test_tool_loop_echoes_only_the_wire_fields_of_an_assistant_turn(openrouter):
    """A provider's private fields (reasoning traces, native ids) are not part
    of the request schema; echoing them back is a 400 on a strict validator."""
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {}, "c1")],
                extra={"reasoning": "chain of thought",
                       "reasoning_details": [{"text": "secret"}],
                       "native_finish_reason": "tool_calls", "refusal": None}))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm._registry = lambda ctx, include=None: _tool(lambda **k: {"ok": 1})
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=4) == "done"
    echoed = [m for m in bodies[1]["messages"] if m["role"] == "assistant"][0]
    assert set(echoed) == {"role", "content", "tool_calls"}
    assert "chain of thought" not in json.dumps(bodies[1])


def test_tool_loop_parses_json_string_arguments(openrouter):
    """OpenAI serialises arguments as a JSON string; Ollama sends an object."""
    got = {}

    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(tool_calls=[
                {"id": "c1", "type": "function", "function": {
                    "name": "probe", "arguments": '{"metric": "step_count", "n": 3}'}}]))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm._registry = lambda ctx, include=None: _tool(lambda **k: got.update(k) or {"ok": 1})
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=4) == "done"
    assert got == {"metric": "step_count", "n": 3}


def test_tool_loop_tool_exception_becomes_a_result_not_a_raise(openrouter):
    bodies = []
    turns = {"n": 0}

    def boom(**kw):
        raise ValueError("no such metric")

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {}, "c1")]))
        return httpx.Response(200, json=_assistant("recovered"))

    openrouter(handler)
    llm._registry = lambda ctx, include=None: _tool(boom)
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=4) == "recovered"
    result = [m for m in bodies[1]["messages"] if m["role"] == "tool"][0]
    assert json.loads(result["content"]) == {"error": "no such metric"}
    assert result["tool_call_id"] == "c1"


def test_tool_loop_survives_a_tool_result_json_cannot_encode(openrouter):
    """#151. The tool CALL was guarded; the ENCODE was not.

    A tool that returns a value ``json`` cannot represent used to raise
    TypeError at the ``json.dumps`` outside the per-call guard, escape to
    tool_loop's outer handler, and discard the entire conversation — every
    earlier tool call with it. The user saw the deterministic fallback, which
    is byte-indistinguishable from a genuinely ungrounded answer.

    Measured live 2026-08-27: ``Object of type GradingPolicy is not JSON
    serializable`` on /v1/ask, answer returned with figures_verified 0/0.
    """
    class Unencodable:
        pass

    bodies = []
    turns = {"n": 0}

    def leaks(**kw):
        return {"policy": Unencodable(), "jog_minutes": 55.3}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {}, "c1")]))
        return httpx.Response(200, json=_assistant("recovered"))

    openrouter(handler)
    llm._registry = lambda ctx, include=None: _tool(leaks)

    # Before #151 this returned "" — the loop died and turn 2 never happened.
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=4) == "recovered"
    assert len(bodies) == 2, "the loop must reach a second turn, not die"

    result = [m for m in bodies[1]["messages"] if m["role"] == "tool"][0]
    payload = json.loads(result["content"])
    assert "error" in payload and "not JSON-encodable" in payload["error"]
    assert result["tool_call_id"] == "c1"
    # The model is told the tool failed. It is NOT handed a stringified object
    # that reads like data — default=str would have shipped one.
    assert "55.3" not in result["content"]


def test_unencodable_tool_result_is_announced_not_silent(openrouter):
    """#128 made every empty return announce; #151 died before reaching that."""
    seen = []

    class Unencodable:
        pass

    llm._registry = lambda ctx, include=None: {}
    content = llm._encode_tool_result({"p": Unencodable()}, "get_week_plan")
    assert "not JSON-encodable" in content
    assert llm._encode_tool_result({"a": 1}, "ok") == '{"a": 1}'


def test_tool_loop_multi_turn_exchange_terminates(openrouter):
    """Three tool turns then a final answer — history stays valid throughout."""
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] <= 3:
            return httpx.Response(200, json=_assistant(tool_calls=[
                _call("probe", {"i": turns["n"]}, f"c{turns['n']}")]))
        return httpx.Response(200, json=_assistant("finished"))

    openrouter(handler)
    llm._registry = lambda ctx, include=None: _tool(lambda **k: {"ok": k})
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=10) == "finished"
    assert turns["n"] == 4
    for body in bodies:
        _assert_valid_openai_request(body)
    assert len([m for m in bodies[3]["messages"] if m["role"] == "tool"]) == 3


def test_tool_loop_forwards_the_provider_pin_with_fallbacks_off(openrouter):
    """D15: the allow-list is who sees the health data, and a pin that can fail
    over is not a pin. The tool path must carry it exactly as complete() does."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_assistant("answer"))

    openrouter(handler)
    llm.tool_loop("q", ctx=None, tools=[{"type": "function"}], max_turns=2)
    assert seen["body"]["provider"] == {
        "order": ["coreweave/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert seen["body"]["reasoning"] == {"enabled": False}
    assert seen["body"]["model"] == "deepseek/deepseek-v4-flash-0731"
    assert seen["body"]["tool_choice"] == "auto"
    assert seen["body"]["tools"] == [{"type": "function"}]


# --------------------------------------------------------------------------
# tool_loop — refusals, all of which must announce themselves
# --------------------------------------------------------------------------

def test_tool_loop_refuses_an_unapproved_provider_and_sends_nothing(openrouter,
                                                                    monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_assistant("leaked"))

    openrouter(handler)
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "cheapo/int4")
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=2) == ""
    assert calls == []
    status = llm.last_loop_status()
    assert status["outcome"] == "openrouter_not_approved"
    assert "cheapo/int4" in status["detail"]


def test_tool_loop_refuses_an_unknown_served_provider_name_and_announces(
        openrouter, monkeypatch):
    calls, events = [], []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_assistant(
            "leaked", provider="TogetherAI-Evil"))

    original_announce = llm._announce

    def announce(event, detail="", **kwargs):
        events.append((event, detail))
        original_announce(event, detail, **kwargs)

    monkeypatch.setattr(llm, "_announce", announce)
    openrouter(handler)
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=2) == ""
    assert len(calls) == 1
    assert events and events[0][0] == "openrouter_provider_mismatch"
    assert "TogetherAI-Evil" in events[0][1]


def test_tool_loop_refuses_an_unapproved_endpoint_host_and_sends_nothing(
        openrouter, monkeypatch):
    """#138's axis: the pin says who OpenRouter routes to, not who receives the
    request. The new transport must not be a way around that check."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_assistant("leaked"))

    openrouter(handler)
    monkeypatch.setattr(llm, "OPENROUTER_URL", "https://openrouter.ai.attacker.example/v1")
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=2) == ""
    assert calls == []
    assert llm.last_loop_status()["outcome"] == "openrouter_not_approved"


def test_tool_loop_without_an_api_key_announces_and_sends_nothing(openrouter,
                                                                 monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_assistant("leaked"))

    openrouter(handler)
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "")
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=2) == ""
    assert calls == []
    assert llm.last_loop_status()["outcome"] == "openrouter_no_api_key"


def test_tool_loop_turn_cap_holds_and_announces(openrouter):
    """A model that only ever calls tools must stop at max_turns, and the empty
    return must be distinguishable from a researcher that found nothing."""
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        return httpx.Response(200, json=_assistant(
            tool_calls=[_call("probe", {"i": turns["n"]}, f"c{turns['n']}")]))

    openrouter(handler)
    llm._registry = lambda ctx, include=None: _tool(lambda **k: {"ok": 1})
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=3) == ""
    assert turns["n"] == 3
    assert llm.last_loop_status()["outcome"] == "tool_loop_turns_exhausted"


def test_tool_loop_deadline_holds_before_any_request(openrouter):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_assistant("answer"))

    openrouter(handler)
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=5, deadline=-1) == ""
    assert calls == []
    assert llm.last_loop_status()["outcome"] == "tool_loop_deadline"


def test_tool_loop_transport_error_returns_empty_and_announces(openrouter):
    def handler(request):
        raise httpx.ConnectError("down")

    openrouter(handler)
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=3) == ""
    status = llm.last_loop_status()
    assert status["outcome"] == "tool_loop_error"
    assert "ConnectError" in status["detail"]


def test_tool_loop_non_200_returns_empty_and_announces(openrouter):
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})

    openrouter(handler)
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=3) == ""
    assert llm.last_loop_status()["outcome"] == "tool_loop_error"


# --------------------------------------------------------------------------
# provenance — the ledger the claim channel cites
# --------------------------------------------------------------------------

def test_tool_loop_writes_the_call_ledger_and_publishes_the_sequence(openrouter,
                                                                     tmp_path):
    """A research claim cites _ledger.sequence; without the ledger the claim is
    unverifiable and dropped, so the openrouter path must build the same one the
    MCP server does."""
    ledger = tmp_path / "run_ledger.jsonl"
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {"days": 7}, "c1")]))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm._registry = lambda ctx, include=None: _tool(lambda days: {"mean": 5413})
    assert llm.tool_loop("q", ctx=None, tools=[], max_turns=4,
                         ledger_path=str(ledger)) == "done"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "probe" and rows[0]["sequence"] == 1
    assert rows[0]["arguments"] == {"days": 7}
    published = json.loads([m for m in bodies[1]["messages"]
                            if m["role"] == "tool"][0]["content"])
    assert published["_ledger"]["sequence"] == 1
    assert published["mean"] == 5413


def test_research_loop_builds_the_ledger_beside_the_scratchpad(openrouter, tmp_path):
    """agent_loop reads <scratch>_ledger.jsonl back by that exact name, which is
    what _deepdive_mcp_config derives for codex. The local path must agree."""
    scratch = tmp_path / "run.json"
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {}, "c1")]))
        return httpx.Response(200, json=_assistant("DONE"))

    openrouter(handler)
    out = llm.research_loop("investigate", ctx=None,
                            extra_tools=_tool(lambda: {"mean": 1}),
                            compact_state=lambda: "S", max_turns=4,
                            scratch_path=str(scratch), task_id=1)
    assert out == "DONE"
    assert (tmp_path / "run_ledger.jsonl").exists()


# --------------------------------------------------------------------------
# research_loop — context management on the OpenAI dialect
# --------------------------------------------------------------------------

def test_research_loop_dispatches_a_tool_then_finishes(openrouter):
    recorded = {}
    turns = {"n": 0}
    bodies = []

    def record_finding(claim, numbers=None, **k):
        recorded["claim"] = claim
        return {"ok": True, "n_findings": 1}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(tool_calls=[
                _call("record_finding", {"claim": "Steps up.", "numbers": []}, "c1")]))
        return httpx.Response(200, json=_assistant("DONE"))

    openrouter(handler)
    out = llm.research_loop("investigate", ctx=None,
                            extra_tools=_tool(record_finding, "record_finding"),
                            compact_state=lambda: "STATE", max_turns=5)
    assert out == "DONE"
    assert recorded["claim"] == "Steps up."
    for body in bodies:
        _assert_valid_openai_request(body)


def test_research_loop_compacts_on_usage_prompt_tokens(openrouter):
    """Ollama reports prompt_eval_count; OpenRouter reports usage.prompt_tokens.
    Reading the wrong one means compaction never fires and the run dies on
    context length instead."""
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {}, "c1")], usage=2000))
        return httpx.Response(200, json=_assistant("DONE", usage=5))

    openrouter(handler)
    out = llm.research_loop("ORIGINAL", ctx=None, extra_tools=_tool(lambda: {"ok": 1}),
                            compact_tokens=1000, compact_state=lambda: "STATELINE",
                            max_turns=4)
    assert out == "DONE"
    second = bodies[1]["messages"]
    assert second[0]["content"] == "ORIGINAL"
    assert "[context compacted]" in second[1]["content"]
    assert "STATELINE" in second[1]["content"]
    _assert_valid_openai_request(bodies[1])


def test_research_loop_compaction_window_never_contains_a_half_turn(openrouter):
    """The kept tail is a fixed-size slice and can cut an assistant turn away
    from its results. On this dialect either half alone is a 400."""
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] <= 4:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {"i": turns["n"]}, f"c{turns['n']}")],
                usage=2000 if turns["n"] == 4 else 5))
        return httpx.Response(200, json=_assistant("DONE", usage=5))

    openrouter(handler)
    out = llm.research_loop("ORIGINAL", ctx=None, extra_tools=_tool(lambda **k: {"ok": 1}),
                            compact_tokens=1000, keep_last_k=1,
                            compact_state=lambda: "S", max_turns=8)
    assert out == "DONE"
    assert len(bodies) >= 5
    for body in bodies:
        _assert_valid_openai_request(body)
    assert "[context compacted]" in bodies[4]["messages"][1]["content"]


def test_research_loop_elision_keeps_the_tool_call_id(openrouter):
    """Eliding a big result is a size reduction, never a re-pairing: dropping
    the id would orphan the assistant turn that called it and make the retry a
    400. Driven through the turn-error retry, which elides in place."""
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] <= 2:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {"i": turns["n"]}, f"c{turns['n']}")]))
        if turns["n"] == 3:
            raise httpx.ReadTimeout("slot busy")
        return httpx.Response(200, json=_assistant("DONE"))

    openrouter(handler)
    out = llm.research_loop("p", ctx=None,
                            extra_tools=_tool(lambda **k: {"blob": "R" * 200}),
                            keep_last_k=1, compact_state=lambda: "S", max_turns=8)
    assert out == "DONE"
    final = bodies[-1]["messages"]
    elided = [m for m in final if m["role"] == "tool"
              and m["content"].startswith("[probe result elided")]
    assert elided, [m for m in final if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in elided] == ["c1"]
    assert all("tool_name" not in m for m in elided)
    _assert_valid_openai_request(bodies[-1])


def test_research_loop_truncates_a_long_tool_result_with_a_marker(openrouter):
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(
                tool_calls=[_call("probe", {}, "c1")]))
        return httpx.Response(200, json=_assistant("DONE"))

    openrouter(handler)
    out = llm.research_loop("p", ctx=None,
                            extra_tools=_tool(lambda: {"blob": "Z" * 10000}),
                            compact_state=lambda: "S", max_tool_chars=100, max_turns=4)
    assert out == "DONE"
    result = [m for m in bodies[1]["messages"] if m["role"] == "tool"][0]
    assert result["content"].endswith("…[truncated]")
    assert len(result["content"]) == 100 + len("…[truncated]")


def test_research_loop_stall_detection_and_finalize(openrouter):
    events = []

    def handler(request):  # always the SAME call
        return httpx.Response(200, json=_assistant(
            tool_calls=[_call("probe", {"a": 1}, "c1")]))

    openrouter(handler)
    out = llm.research_loop("p", ctx=None, extra_tools=_tool(lambda **k: {"ok": 1}),
                            compact_state=lambda: "S", max_turns=12, stall_budget=2,
                            on_log=lambda t, ev, d: events.append(ev))
    assert "loop" in events
    assert "finalize" in events
    assert isinstance(out, str)


def test_research_loop_turn_error_budget_bounds_retries_and_announces(openrouter):
    calls = {"n": 0}
    events = []

    def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("dead")

    openrouter(handler)
    out = llm.research_loop("p", ctx=None, extra_tools={}, compact_state=lambda: "S",
                            max_turns=10, on_log=lambda t, ev, d: events.append(ev))
    assert out == ""
    assert calls["n"] == 1 + llm.TURN_ERROR_BUDGET
    assert "research_loop_turn_error_budget" in events
    assert llm.last_loop_status()["outcome"] == "research_loop_turn_error_budget"


def test_research_loop_deadline_injects_finalize_before_the_first_post(openrouter):
    events, bodies = [], []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_assistant("wrapped up"))

    openrouter(handler)
    out = llm.research_loop("p", ctx=None, extra_tools={}, compact_state=lambda: "S",
                            deadline=-1, max_turns=5,
                            on_log=lambda t, ev, d: events.append(ev))
    assert out == "wrapped up"
    assert "finalize" in events
    assert any(llm._FINALIZE_MSG in m["content"]
               for m in bodies[0]["messages"] if m["role"] == "user")


def test_research_loop_refusal_reaches_on_log_and_makes_no_request(openrouter,
                                                                   monkeypatch):
    """research_loop already logged openrouter_tool_path_disabled; the refusal
    that replaces it must be at least as loud."""
    calls, events = [], []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_assistant("leaked"))

    openrouter(handler)
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "")
    out = llm.research_loop("p", ctx=None, extra_tools={}, compact_state=lambda: "S",
                            on_log=lambda t, ev, d: events.append((ev, d)))
    assert out == ""
    assert calls == []
    assert events and events[0][0] == "openrouter_not_approved"
    assert "HA_OPENROUTER_PROVIDERS" in events[0][1]


def test_research_loop_transport_error_returns_empty_and_announces(openrouter):
    def handler(request):
        raise httpx.ConnectError("down")

    openrouter(handler)
    monkeypatchless = llm.research_loop("p", ctx=None, extra_tools={},
                                        compact_state=lambda: "S", max_turns=1)
    assert monkeypatchless == ""


# --------------------------------------------------------------------------
# the /v1/ask seam — the other surface the early return took down
# --------------------------------------------------------------------------

def test_ask_path_produces_a_tool_call_ledger_on_openrouter(openrouter, monkeypatch,
                                                            vault, conn):
    """Measured on a Linux host 2026-08-27: POST /v1/ask returned HTTP 200,
    mode=fallback, tool_calls=0, reason "ask answer has no tool-call ledger", in
    0.32 s with the openrouter backend correctly configured — no model was
    contacted at all. The polite refusal is indistinguishable from the system's
    DESIGNED refusals, which is why the count is what this pins.

    This drives the REAL llm.tool_loop through chat.answer_question, so the seam
    that was dead is the seam under test: a tool call the coach makes must reach
    the ledger that the ask verification gate reads back.
    """
    from health_advisor import chat

    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)  # the real tool surface
    monkeypatch.setattr(llm, "BACKEND", "openrouter")

    def handler(request):
        body = json.loads(request.content)
        messages = body.get("messages", [])
        if any(m.get("role") == "tool" for m in messages):
            return httpx.Response(200, json=_assistant('{"text": "Checked.", "claims": []}'))
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("list_available_metrics", {}, "call_metrics")]))

    openrouter(handler)
    result = chat.answer_question(vault, "Which metrics do you have?")
    assert result["tool_trace"], "the ask gate reads this back; empty is the defect"
    assert result["tool_trace"][0]["tool_name"] == "list_available_metrics"
    assert result["tool_trace"][0]["sequence"] == 1
    reasons = json.dumps(result["verification"])
    assert "no tool-call ledger" not in reasons


# --------------------------------------------------------------------------
# unit: the dialect helpers
# --------------------------------------------------------------------------

def test_prune_orphaned_tool_turns_drops_both_halves():
    window = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "ra"},   # "b" was sliced off
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
        {"role": "tool", "tool_call_id": "c", "content": "rc"},
        {"role": "tool", "tool_call_id": "zz", "content": "orphan"},
    ]
    kept = llm._prune_orphaned_tool_turns(window)
    assert kept == window[2:4]


def test_elide_preserves_the_openai_pairing_key():
    msgs = [{"role": "user", "content": "p"}]
    for i in range(3):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": f"t{i}",
                     "content": "X" * 100})
    llm._elide_old_tool_results(msgs, keep_last_k=1)
    tools = [m for m in msgs if m["role"] == "tool"]
    assert tools[0]["tool_call_id"] == "c0"
    assert tools[0]["content"].startswith("[t0 result elided")
    assert "tool_name" not in tools[0]
    assert tools[2]["content"] == "X" * 100


def test_openai_sampling_drops_ollama_only_keys():
    assert llm._openai_sampling({"temperature": 0.7, "top_p": 0.95,
                                 "num_ctx": 65536, "keep_alive": "10m"}) == {
        "temperature": 0.7, "top_p": 0.95}


def test_derived_ledger_path_matches_the_mcp_config():
    assert llm._derived_ledger_path("/x/run.json", None) == "/x/run_ledger.jsonl"
    assert llm._derived_ledger_path("/x/run.json", "/y/explicit.jsonl") == "/y/explicit.jsonl"
    assert llm._derived_ledger_path(None, None) is None
