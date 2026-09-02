# tests/test_research_loop.py
import json
import httpx
import pytest
from health_advisor import llm
from health_advisor import deepdive_memory as M


def _t(handler):
    return httpx.MockTransport(handler)


def test_research_loop_dispatches_extra_tool_then_done(monkeypatch, vault):
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})  # isolate from the read tools
    recorded = {}
    def record_finding(claim, numbers=None, **k):
        recorded["claim"] = claim
        return {"ok": True, "n_findings": 1}
    extra = {"record_finding": (record_finding, {"type": "function",
             "function": {"name": "record_finding"}})}
    turns = {"n": 0}
    def handler(request):
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": "record_finding",
                    "arguments": {"claim": "Steps up.", "numbers": []}}}]},
                "prompt_eval_count": 10})
        return httpx.Response(200, json={"message": {"content": "DONE"},
                                         "prompt_eval_count": 12})
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("investigate", ctx=vault, extra_tools=extra,
                            compact_state=lambda: "STATE", max_turns=5)
    assert out == "DONE"
    assert recorded["claim"] == "Steps up."


def test_research_loop_unknown_tool_surfaces_error(monkeypatch, vault):
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    turns = {"n": 0}
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content)); turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json={"message": {"content": "",
                "tool_calls": [{"function": {"name": "nope", "arguments": {}}}]},
                "prompt_eval_count": 5})
        return httpx.Response(200, json={"message": {"content": "DONE"},
                                         "prompt_eval_count": 6})
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("x", ctx=vault, extra_tools={}, compact_state=lambda: "S", max_turns=5)
    assert out == "DONE"
    tool_msgs = [m for m in bodies[1]["messages"] if m.get("role") == "tool"]
    assert json.loads(tool_msgs[0]["content"])["error"].startswith("unknown tool")


def test_research_loop_survives_turn_timeout_and_finalizes(monkeypatch, vault):
    """A per-turn transport timeout must not abort the run (the 2026-07-05
    0-findings failure): the loop logs turn_error, injects the finalize
    instruction, and retries so gathered work can still be recorded."""
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    events = []
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content)); turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json={"message": {"content": "",
                "tool_calls": [{"function": {"name": "x", "arguments": {}}}]},
                "prompt_eval_count": 5})
        if turns["n"] == 2:
            raise httpx.ReadTimeout("slot busy")
        return httpx.Response(200, json={"message": {"content": "wrapped with what I have"},
                                         "prompt_eval_count": 6})

    extra = {"x": ((lambda **k: {"ok": 1}), {"type": "function", "function": {"name": "x"}})}
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("x", ctx=vault, extra_tools=extra, compact_state=lambda: "S",
                            max_turns=6,
                            on_log=lambda turn, ev, detail: events.append(ev))
    assert out == "wrapped with what I have"
    assert "turn_error" in events
    # the retry POST carries the finalize instruction
    assert any(llm._FINALIZE_MSG in m.get("content", "")
               for m in bodies[2]["messages"] if m.get("role") == "user")


def test_research_loop_turn_error_budget_bounds_retries(monkeypatch, vault):
    """Persistent per-turn failures stop after the error budget (no infinite
    retry against a dead model) and still return "" without raising."""
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("dead")

    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("x", ctx=vault, extra_tools={}, compact_state=lambda: "S",
                            max_turns=10)
    assert out == ""
    assert calls["n"] == 1 + llm.TURN_ERROR_BUDGET


def test_research_loop_maps_error_to_empty(monkeypatch, vault):
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    def handler(request):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    assert llm.research_loop("x", ctx=vault, extra_tools={}, compact_state=lambda: "S") == ""


def test_research_loop_finalize_on_deadline(monkeypatch, vault):
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    events = []
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "wrapped up"},
                                         "prompt_eval_count": 7})
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("x", ctx=vault, extra_tools={}, compact_state=lambda: "S",
                            deadline=0, max_turns=5,
                            on_log=lambda turn, ev, detail: events.append(ev))
    assert out == "wrapped up"
    assert "finalize" in events
    # finalize message was injected before the first POST
    assert any(llm._FINALIZE_MSG in m["content"]
               for m in bodies[0]["messages"] if m["role"] == "user")


def test_elide_old_tool_results_stubs_all_but_last_k():
    msgs = [{"role": "user", "content": "p"}]
    for i in range(4):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [1]})
        msgs.append({"role": "tool", "tool_name": f"t{i}",
                     "content": "X" * 100})
    llm._elide_old_tool_results(msgs, keep_last_k=2)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("[t0 result elided")
    assert tool_msgs[1]["content"].startswith("[t1 result elided")
    assert tool_msgs[2]["content"] == "X" * 100   # last 2 kept full
    assert tool_msgs[3]["content"] == "X" * 100


def test_research_loop_keeps_history_append_only_for_cache_reuse(monkeypatch, vault):
    """Old tool results must NOT be rewritten between turns: mutating history
    invalidates llama.cpp's SWA/hybrid context checkpoints and forces a full
    prompt reprocess every turn (the 2026-07-05 slowdown). Elision is reserved
    for compaction and timeout-retry, which pay a full reprocess anyway."""
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content)); turns["n"] += 1
        if turns["n"] <= 3:  # three tool turns with distinct args (no stall nudge)
            return httpx.Response(200, json={"message": {"content": "",
                "tool_calls": [{"function": {"name": "x", "arguments": {"i": turns["n"]}}}]},
                "prompt_eval_count": 5})
        return httpx.Response(200, json={"message": {"content": "DONE"},
                                         "prompt_eval_count": 6})

    extra = {"x": ((lambda **k: {"blob": "R" * 50}),
                   {"type": "function", "function": {"name": "x"}})}
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("x", ctx=vault, extra_tools=extra, compact_state=lambda: "S",
                            keep_last_k=1, max_turns=8)
    assert out == "DONE"
    tool_msgs = [m for m in bodies[3]["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    # every earlier tool result is still verbatim — the prompt is a strict prefix
    assert all(json.loads(m["content"])["blob"] == "R" * 50 for m in tool_msgs)


def test_research_loop_compacts_at_absolute_token_threshold(monkeypatch, vault):
    """Compaction keys off the absolute compact_tokens prompt-token count, not a
    fraction of num_ctx: with append-only history the only full reprocess left
    is the post-compaction one, so its trigger must have a known token cost no
    matter how large the context window is."""
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content)); turns["n"] += 1
        if turns["n"] == 1:  # above compact_tokens, far below any num_ctx fraction
            return httpx.Response(200, json={"message": {"content": "",
                "tool_calls": [{"function": {"name": "x", "arguments": {}}}]},
                "prompt_eval_count": 2000})
        return httpx.Response(200, json={"message": {"content": "DONE"},
                                         "prompt_eval_count": 5})

    extra = {"x": ((lambda **k: {"ok": 1}), {"type": "function", "function": {"name": "x"}})}
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("ORIGINAL", ctx=vault, extra_tools=extra, compact_tokens=1000,
                            compact_state=lambda: "STATELINE", max_turns=4)
    assert out == "DONE"
    assert "[context compacted]" in bodies[1]["messages"][1]["content"]


def test_research_loop_compacts_on_high_pec(monkeypatch, vault):
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    bodies = []
    turns = {"n": 0}
    def handler(request):
        bodies.append(json.loads(request.content)); turns["n"] += 1
        if turns["n"] == 1:  # over threshold + a tool call to force another turn
            return httpx.Response(200, json={"message": {"content": "",
                "tool_calls": [{"function": {"name": "x", "arguments": {}}}]},
                "prompt_eval_count": 10**9})
        return httpx.Response(200, json={"message": {"content": "DONE"},
                                         "prompt_eval_count": 5})
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("ORIGINAL", ctx=vault, extra_tools={},
                            compact_state=lambda: "STATELINE", max_turns=4)
    assert out == "DONE"
    second = bodies[1]["messages"]
    assert second[0]["content"] == "ORIGINAL"
    assert "[context compacted]" in second[1]["content"] and "STATELINE" in second[1]["content"]


def test_research_loop_loop_detection_and_stall_finalize(monkeypatch, vault):
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    events = []
    def handler(request):  # always repeats the SAME tool call
        return httpx.Response(200, json={"message": {"content": "",
            "tool_calls": [{"function": {"name": "x", "arguments": {"a": 1}}}]},
            "prompt_eval_count": 5})
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("x", ctx=vault, extra_tools={"x": ((lambda **k: {"ok": 1}),
                            {"type": "function", "function": {"name": "x"}})},
                            compact_state=lambda: "S", max_turns=12, stall_budget=2,
                            on_log=lambda turn, ev, detail: events.append(ev))
    assert "loop" in events       # repeated call was nudged
    assert "finalize" in events   # stall budget tripped finalize
    assert isinstance(out, str)   # terminates without raising


def test_research_loop_records_through_real_memory_tools(monkeypatch, tmp_path, vault):
    """Seam test: the REAL research_loop dispatch path drives the REAL
    build_memory_tools closure, so a model tool_call for record_finding actually
    writes the scratchpad — the production composition that was otherwise covered
    only by the skipped @live test. Also guards that think=True is forwarded on the
    researcher path (the dropped test_deep_dive_calls_all_use_think used to)."""
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})  # isolate from the read tools
    scratch = str(tmp_path / "s.json")
    M.new_scratchpad(scratch, "2026-06-29", [{"id": 1, "question": "How are steps?"}])
    tools = M.build_memory_tools(scratch, 1)
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content)); turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json={"message": {"content": "", "tool_calls": [
                {"function": {"name": "record_finding", "arguments": {
                    "claim": "Mean steps 5413.",
                    "numbers": [{"metric": "step_count", "field": "mean", "value": 5413}]}}}]},
                "prompt_eval_count": 10})
        return httpx.Response(200, json={"message": {"content": "DONE"},
                                         "prompt_eval_count": 12})

    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("investigate", ctx=vault, extra_tools=tools,
                            compact_state=lambda: M.compact_state(M.load(scratch), 1),
                            think=True, max_turns=5)
    assert out == "DONE"
    # the real closure mutated the real scratchpad (not a hand-rolled stub)
    findings = M.load(scratch)["findings"]
    assert len(findings) == 1
    assert findings[0]["claim"] == "Mean steps 5413."
    assert findings[0]["numbers"][0]["value"] == 5413
    # think=True forwarded on every researcher POST
    assert bodies and all(b["think"] is True for b in bodies)


def test_compaction_tail_forward_trims_leading_orphan_tool():
    """The compaction window must never begin with an orphaned tool-result whose
    assistant tool_calls parent was sliced off the front."""
    msgs = [{"role": "user", "content": "p"},
            {"role": "assistant", "tool_calls": [1, 2]},
            {"role": "tool", "tool_name": "a", "content": "r1"},
            {"role": "assistant", "tool_calls": [3]},
            {"role": "tool", "tool_name": "b", "content": "r2"},
            {"role": "assistant", "tool_calls": [4]},
            {"role": "tool", "tool_name": "c", "content": "r3"},
            {"role": "tool", "tool_name": "c2", "content": "r4"}]
    assert msgs[-6:][0]["role"] == "tool"          # raw slice would start orphaned
    tail = llm._compaction_tail(msgs, keep_last_k=3)
    assert tail[0]["role"] != "tool"               # forward-trimmed
    assert tail == msgs[-5:]                        # only the leading orphan dropped


def test_research_loop_truncates_long_tool_result_with_marker(monkeypatch, vault):
    monkeypatch.setattr(llm, "_registry", lambda ctx: {})
    big = {"blob": "Z" * 10000}
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content)); turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json={"message": {"content": "", "tool_calls": [
                {"function": {"name": "big", "arguments": {}}}]}, "prompt_eval_count": 5})
        return httpx.Response(200, json={"message": {"content": "DONE"},
                                         "prompt_eval_count": 6})

    extra = {"big": ((lambda **k: big), {"type": "function", "function": {"name": "big"}})}
    monkeypatch.setattr(llm, "_TRANSPORT", _t(handler))
    out = llm.research_loop("x", ctx=vault, extra_tools=extra, compact_state=lambda: "S",
                            max_tool_chars=100, max_turns=4)
    assert out == "DONE"
    tool_msg = [m for m in bodies[1]["messages"] if m.get("role") == "tool"][0]
    assert tool_msg["content"].endswith("…[truncated]")
    assert len(tool_msg["content"]) == 100 + len("…[truncated]")
