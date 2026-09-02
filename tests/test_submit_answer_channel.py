# tests/test_submit_answer_channel.py
"""The ask path's claim channel as a typed tool call, and the prompt it rides on.

Two defects are pinned here.

The first is a prompt defect: `chat.answer_question` appends
`ASK_CLAIM_INSTRUCTIONS`, and `llm.tool_loop` used to append
`_RESEARCH_CLAIM_INSTRUCTIONS` unconditionally on top of it. Every /v1/ask call
carried both — 929 words of two schemas that CONTRADICT, the research block
permitting a claim source path rooted at `$.arguments...` that the ask verifier
refuses outright.

The second is the channel itself. Terminating by parsing a free-text final
message for a {text, claims} object fails whenever the model writes bare prose,
which with reasoning off it frequently does; `claims` comes back None, every
figure is unsupported, and the user gets "Answer withheld". A synthetic
`submit_answer` tool makes the channel typed.

The guard that matters is that NONE of this weakens grounding: `submit_answer`
is not an MCP tool, writes nothing to the tool-call ledger, and a model that
calls only it still meets the ask gate's empty-ledger refusal.

Nothing here touches the network. Every request is served by an
httpx.MockTransport and the key is a literal fixture string.
"""
import json

import httpx
import pytest

from health_advisor import chat, llm

_REAL_REGISTRY = llm._registry


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


def _assistant(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _call(name, arguments, call_id="call_0"):
    """One OpenAI-shaped tool call; `arguments` may be a JSON string or raw."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


_CLAIMS = [{"metric": "steps", "period": "2026-08-01:2026-08-07",
            "field": "mean", "value": 5413,
            "source": {"sequence": 1, "path": "$.result.mean"}}]


# --------------------------------------------------------------------------
# Defect 2 — the typed terminal call
# --------------------------------------------------------------------------

def test_submit_answer_terminates_the_loop_with_a_typed_claim_channel(openrouter):
    """A submit_answer call ends the loop and its claims survive verbatim."""
    def handler(request):
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("submit_answer",
                  {"text": "Steps averaged 5413.", "claims": _CLAIMS})]))

    openrouter(handler)
    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True)
    assert isinstance(out, llm.ResearchResponse)
    assert out == "Steps averaged 5413."
    assert out.text == "Steps averaged 5413."
    assert out.claims == _CLAIMS
    assert out.claims is not None


def test_submit_answer_tool_is_offered_on_the_wire_only_when_opted_in(openrouter):
    """The schema reaches the model when asked for, and never by default."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm.tool_loop("q", ctx=None, tools=[{"type": "function"}], max_turns=1)
    llm.tool_loop("q", ctx=None, tools=[{"type": "function"}], max_turns=1,
                  submit_tool=True)

    assert bodies[0]["tools"] == [{"type": "function"}]
    assert bodies[1]["tools"] == [{"type": "function"}, llm.SUBMIT_ANSWER_TOOL]
    fn = llm.SUBMIT_ANSWER_TOOL["function"]
    assert fn["name"] == "submit_answer"
    assert sorted(fn["parameters"]["required"]) == ["claims", "text"]
    assert fn["parameters"]["properties"]["text"]["type"] == "string"
    assert fn["parameters"]["properties"]["claims"]["type"] == "array"


def test_opting_in_does_not_mutate_the_caller_s_tool_list(openrouter):
    """The caller's list is shared with other calls; appending in place would
    grow the wire payload by one synthetic tool per invocation."""
    tools = [{"type": "function"}]

    openrouter(lambda request: httpx.Response(200, json=_assistant("done")))
    llm.tool_loop("q", ctx=None, tools=tools, max_turns=1, submit_tool=True)
    assert tools == [{"type": "function"}]


# --------------------------------------------------------------------------
# the guards — grounding is not weakened anywhere
# --------------------------------------------------------------------------

def test_submit_answer_is_not_an_mcp_tool(vault):
    """It must not be reachable as data, in any tool selection."""
    registry = _REAL_REGISTRY(vault)
    assert "submit_answer" not in registry
    coach = _REAL_REGISTRY(vault, include=llm.COACH_TOOLS)
    assert "submit_answer" not in coach
    assert "submit_answer" not in llm.COACH_TOOLS
    assert "submit_answer" not in llm.RESEARCHER_TOOLS


def test_a_submit_answer_only_answer_still_fails_the_empty_ledger_gate(
        openrouter, monkeypatch, vault, conn):
    """The zero-tool-call loophole stays closed.

    submit_answer produces no ledger entry, so a model that calls nothing else
    files a perfectly well-formed claim channel and is still refused — the
    verifier never sees evidence it did not read.
    """
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)

    def handler(request):
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("submit_answer",
                  {"text": "Steps averaged 5413.", "claims": _CLAIMS})]))

    openrouter(handler)
    result = chat.answer_question(vault, "How many steps?")
    assert result["mode"] == "fallback"
    assert result["tool_trace"] == []
    assert result["verification"]["reason"] == "ask answer has no tool-call ledger"


def test_submit_answer_leaves_no_entry_in_the_tool_call_ledger(openrouter,
                                                               monkeypatch,
                                                               vault, tmp_path):
    """A real data call is ledgered; the terminal call beside it is not."""
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    ledger_path = str(tmp_path / "ledger.jsonl")
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(tool_calls=[
                _call("list_available_metrics", {}, "call_metrics")]))
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("submit_answer", {"text": "Checked.", "claims": []},
                  "call_submit")]))

    openrouter(handler)
    out = llm.tool_loop("question", ctx=vault, tools=[],
                        ledger_path=ledger_path,
                        tool_names=llm.COACH_TOOLS, submit_tool=True)
    assert out.text == "Checked."
    entries = [json.loads(line) for line in
               open(ledger_path).read().splitlines() if line.strip()]
    assert [e["tool_name"] for e in entries] == ["list_available_metrics"]


@pytest.mark.parametrize("arguments, event", [
    ("{not json at all", "submit_answer_unparseable_arguments"),
    ('["a", "list"]', "submit_answer_bad_arguments"),
    ('{"text": 17, "claims": []}', "submit_answer_bad_text"),
    ('{"claims": []}', "submit_answer_bad_text"),
    ('{"text": "ok", "claims": {"not": "a list"}}', "submit_answer_bad_claims"),
    ('{"text": "ok"}', "submit_answer_bad_claims"),
])
def test_malformed_submit_answer_announces_and_does_not_raise(openrouter,
                                                              arguments, event):
    """Every shape failure degrades to the loops' empty return, announced (#128)."""
    def handler(request):
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("submit_answer", arguments)]))

    openrouter(handler)
    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True)
    assert out == ""
    assert out.claims is None
    assert llm.last_loop_status()["outcome"] == event


def test_a_data_tool_beside_submit_answer_is_not_run_before_terminating(openrouter,
                                                                       monkeypatch,
                                                                       vault, tmp_path):
    """An answer written in the same turn as a data call cannot have read it,
    so running that call would put a ledger entry behind nothing."""
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    ledger_path = str(tmp_path / "ledger.jsonl")

    def handler(request):
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("list_available_metrics", {}, "call_metrics"),
            _call("submit_answer", {"text": "Answered.", "claims": []},
                  "call_submit")]))

    openrouter(handler)
    out = llm.tool_loop("question", ctx=vault, tools=[],
                        ledger_path=ledger_path,
                        tool_names=llm.COACH_TOOLS, submit_tool=True)
    assert out.text == "Answered."
    import os
    assert not os.path.exists(ledger_path) or open(ledger_path).read().strip() == ""


def test_submit_answer_is_inert_without_the_opt_in(openrouter):
    """Off by default, the name is just an unknown tool — no terminal shortcut."""
    turns = {"n": 0}

    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(tool_calls=[
                _call("submit_answer", {"text": "x", "claims": []})]))
        return httpx.Response(200, json=_assistant("plain prose"))

    openrouter(handler)
    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4)
    assert out == "plain prose"
    assert out.claims is None
    assert turns["n"] == 2


# --------------------------------------------------------------------------
# Defect 1 — one claim schema per prompt, not two
# --------------------------------------------------------------------------

def test_default_tool_loop_still_appends_the_research_claim_instructions(openrouter):
    """Unchanged for research_loop, the deep dive and coach_brief."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm.tool_loop("QUESTION BODY", ctx=None, tools=[], max_turns=1)
    messages = bodies[0]["messages"]
    assert messages[0] == {"role": "user", "content": "QUESTION BODY"}
    assert messages[1] == {"role": "user",
                           "content": llm._RESEARCH_CLAIM_INSTRUCTIONS}


def test_claim_instructions_none_sends_exactly_one_user_message(openrouter):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm.tool_loop("QUESTION BODY", ctx=None, tools=[], max_turns=1,
                  claim_instructions=None)
    messages = bodies[0]["messages"]
    assert messages == [{"role": "user", "content": "QUESTION BODY"}]


def test_the_ask_prompt_carries_one_claim_schema_not_two(openrouter, monkeypatch,
                                                         vault, conn):
    """The measured defect: both blocks, 929 words, on every /v1/ask call.

    The research block is not merely redundant — it tells the model a claim
    source path may be rooted at `$.arguments...`, which `_verify_ask_answer`
    rejects. Whatever else the ask prompt says, it must not say that.
    """
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("submit_answer", {"text": "Checked.", "claims": []})]))

    openrouter(handler)
    chat.answer_question(vault, "Which metrics do you have?")

    sent = "\n".join(m.get("content") or "" for m in bodies[0]["messages"])
    assert llm._RESEARCH_CLAIM_INSTRUCTIONS not in sent
    assert "RESEARCH CLAIM CHANNEL" not in sent
    assert "$.arguments" not in sent
    assert "COACH CLAIM CHANNEL" in sent
    assert sum(1 for m in bodies[0]["messages"] if m["role"] == "user") == 1


def test_the_ask_instructions_teach_the_submit_answer_call(openrouter):
    """Nothing makes the model use the tool except the prose that names it."""
    assert "submit_answer" in chat.ASK_CLAIM_INSTRUCTIONS


def test_a_double_encoded_claims_array_is_decoded_not_refused(openrouter):
    """The claims ARRAY can arrive as JSON inside the JSON `arguments` string.

    Measured 2026-08-28 against `coreweave/fp8` reasoning off: 3 of the 6
    ask-battery questions came back this way, each announcing
    `submit_answer_bad_claims` — "claims was str, not a list" — and refusing an
    answer whose claims were fully present and well formed.

    Decoding it is a WIRE concern, not a verification one. `arguments` is
    already decoded by the same `json.loads` one step earlier, so a nested
    array encoded the same way is the same class of thing. The decoded list is
    handed to `_verify_ask_answer` untouched; nothing here decides whether a
    claim is true.
    """
    claims = [{"metric": "jog_minutes", "period": "2026-08-21",
               "field": "jog_minutes", "value": 29.3,
               "source": {"sequence": 1, "path": "$.result.periods[0].jog_minutes"}}]

    def handler(request):
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("submit_answer",
                  # claims double-encoded, exactly as the provider sent it
                  json.dumps({"text": "29.3 jog minutes.",
                              "claims": json.dumps(claims)}))]))

    openrouter(handler)
    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True)
    assert out.text == "29.3 jog minutes."
    assert out.claims == claims


def test_a_claims_string_that_is_not_json_is_still_refused(openrouter):
    """Decoding is not tolerance: a string that is not an array still fails."""
    def handler(request):
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("submit_answer",
                  json.dumps({"text": "ok", "claims": "not json at all"}))]))

    openrouter(handler)
    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True)
    assert out == ""
    assert out.claims is None
    assert llm.last_loop_status()["outcome"] == "submit_answer_unparseable_claims"


def test_a_claims_string_decoding_to_a_non_list_is_still_refused(openrouter):
    """`"{}"` decodes cleanly and is still not a claims array."""
    def handler(request):
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("submit_answer",
                  json.dumps({"text": "ok", "claims": "{\"not\": \"a list\"}"}))]))

    openrouter(handler)
    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True)
    assert out == ""
    assert llm.last_loop_status()["outcome"] == "submit_answer_bad_claims"
