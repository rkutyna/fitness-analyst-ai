
# tests/test_submit_answer_repair.py
"""The in-loop repair turn for a malformed `submit_answer`.

Measured 2026-08-28 over eight ask-battery runs (reasoning off,
deepseek-v4-flash-0731 on coreweave/fp8), the /v1/ask loop-failure breakdown was
`submit_answer_unparseable_claims` 21, `submit_answer_bad_claims` 3,
`tool_loop_empty_answer` 2 — so 24 of 26 loop failures were the claims channel
arriving MALFORMED, and `no_model_response` was the most common outcome of every
arm. The corruption is garbage injected mid-JSON, e.g.
`{"sequence":  ript 19, "path": "$.result.points[2].value"}` at char 555, not
double-encoding of a valid array. Since the model usually gets the content
right, this is a serialisation failure and the fix is to hand back a tool error
saying what was wrong and let it re-emit inside the same loop.

What is pinned here is that the repair changes NOTHING about grounding:
`submit_answer` stays out of `_registry` and out of the ledger across a repair,
the repaired claims reach `_verify_ask_answer` untouched, a submit-only answer
is still refused for an empty ledger, and the whole thing is off by default.

Nothing here touches the network. Every request is served by an
httpx.MockTransport and the key is a literal fixture string.
"""
import json
import os

import httpx
import pytest

from health_advisor import chat, llm

_REAL_REGISTRY = llm._registry

# The corruption measured on 2026-08-28, shortened: a `claims` ARRAY handed back
# as a string with a token injected mid-JSON.
_CORRUPT_CLAIMS = '[{"sequence":  ript 19, "path": "$.result.points[2].value"}]'

_CLAIMS = [{"metric": "steps", "period": "2026-08-01:2026-08-07",
            "field": "mean", "value": 5413,
            "source": {"sequence": 1, "path": "$.result.mean"}}]


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
    llm._LAST_LOOP_STATUS.update({"call_id": 0, "outcome": "not_called",
                                  "backend": None, "detail": ""})
    monkeypatch.setattr(llm, "_SUBMIT_REPAIRS", {"attempted": 0, "succeeded": 0})

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


def _scripted(openrouter, responses):
    """Serve `responses` in order, recording every request body sent."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        index = min(len(bodies) - 1, len(responses) - 1)
        return httpx.Response(200, json=responses[index])

    openrouter(handler)
    return bodies


# --------------------------------------------------------------------------
# the repair itself
# --------------------------------------------------------------------------

def test_a_malformed_submit_answer_is_repaired_and_the_retry_is_returned(
        openrouter):
    """A corrupt claims string no longer ends the loop: the model is told what
    was wrong and the well-formed call that follows is what comes back."""
    bodies = _scripted(openrouter, [
        _assistant(tool_calls=[_call(
            "submit_answer",
            {"text": "Steps averaged 5413.", "claims": _CORRUPT_CLAIMS},
            "call_bad")]),
        _assistant(tool_calls=[_call(
            "submit_answer",
            {"text": "Steps averaged 5413.", "claims": _CLAIMS},
            "call_good")]),
    ])

    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True, submit_repair=True)

    assert out.text == "Steps averaged 5413."
    assert out.claims == _CLAIMS
    assert len(bodies) == 2
    status = llm.last_loop_status()
    assert status["submit_repairs_attempted"] == 1
    assert status["submit_repairs_succeeded"] == 1


def test_with_the_flag_off_the_same_exchange_still_terminates_empty(openrouter):
    """Today's behaviour, unchanged: the loop never gets to the second turn."""
    bodies = _scripted(openrouter, [
        _assistant(tool_calls=[_call(
            "submit_answer",
            {"text": "Steps averaged 5413.", "claims": _CORRUPT_CLAIMS},
            "call_bad")]),
        _assistant(tool_calls=[_call(
            "submit_answer",
            {"text": "Steps averaged 5413.", "claims": _CLAIMS},
            "call_good")]),
    ])

    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True)

    assert out == ""
    assert out.claims is None
    assert len(bodies) == 1
    status = llm.last_loop_status()
    assert status["outcome"] == "submit_answer_unparseable_claims"
    assert status["submit_repairs_attempted"] == 0


def test_the_repair_budget_is_enforced(openrouter):
    """A model that never recovers cannot spend the whole turn budget trying.

    Every turn is malformed. With a budget of 2 the loop makes exactly three
    requests — the original and two repairs — then announces and returns empty,
    well short of max_turns=8.
    """
    bodies = _scripted(openrouter, [
        _assistant(tool_calls=[_call(
            "submit_answer",
            {"text": "Steps averaged 5413.", "claims": _CORRUPT_CLAIMS},
            "call_bad")]),
    ])

    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=8,
                        submit_tool=True, submit_repair=True,
                        submit_repair_budget=2)

    assert out == ""
    assert out.claims is None
    assert len(bodies) == 3
    status = llm.last_loop_status()
    assert status["outcome"] == "submit_answer_repair_exhausted"
    assert "submit_answer_unparseable_claims" in status["detail"]
    assert status["submit_repairs_attempted"] == 2
    assert status["submit_repairs_succeeded"] == 0


def test_the_default_repair_budget_is_two(openrouter):
    """The bound is the module default, not something only tests pass."""
    assert llm.SUBMIT_ANSWER_REPAIR_BUDGET == 2

    bodies = _scripted(openrouter, [
        _assistant(tool_calls=[_call(
            "submit_answer", {"text": "x", "claims": _CORRUPT_CLAIMS},
            "call_bad")]),
    ])
    llm.tool_loop("question", ctx=None, tools=[], max_turns=8,
                  submit_tool=True, submit_repair=True)
    assert len(bodies) == 3


# --------------------------------------------------------------------------
# the guards — grounding is not weakened by the repair
# --------------------------------------------------------------------------

def test_a_repair_writes_no_ledger_entry_for_submit_answer(openrouter,
                                                           monkeypatch, vault,
                                                           tmp_path):
    """The synthetic call must never buy provenance, least of all by failing.

    A real data tool called in the SAME turn as the malformed submit_answer is
    ledgered; the repaired submit_answer contributes nothing to the ledger on
    either the malformed turn or the successful one.
    """
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    ledger_path = str(tmp_path / "ledger.jsonl")

    _scripted(openrouter, [
        _assistant(tool_calls=[
            _call("list_available_metrics", {}, "call_metrics"),
            _call("submit_answer", {"text": "Checked.",
                                    "claims": _CORRUPT_CLAIMS}, "call_bad"),
        ]),
        _assistant(tool_calls=[_call(
            "submit_answer", {"text": "Checked.", "claims": []}, "call_good")]),
    ])

    out = llm.tool_loop("question", ctx=vault, tools=[],
                        ledger_path=ledger_path, tool_names=llm.COACH_TOOLS,
                        submit_tool=True, submit_repair=True)

    assert out.text == "Checked."
    entries = [json.loads(line) for line in
               open(ledger_path).read().splitlines() if line.strip()]
    assert [e["tool_name"] for e in entries] == ["list_available_metrics"]
    assert "submit_answer" not in {e["tool_name"] for e in entries}


def test_a_data_tool_beside_a_malformed_submit_answer_is_actually_run(
        openrouter, monkeypatch, vault, tmp_path):
    """The contrast with the terminal-success path.

    On success the loop returns before running anything, because the answer was
    already written and must not get a ledger entry behind it. On a REPAIR
    there is no answer yet — the model will write a fresh one — so the other
    calls in the turn are legitimate reads and are run, and their results are
    in front of the model when it re-emits.
    """
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    ledger_bad = str(tmp_path / "repair.jsonl")
    ledger_good = str(tmp_path / "terminal.jsonl")

    _scripted(openrouter, [
        _assistant(tool_calls=[
            _call("list_available_metrics", {}, "call_metrics"),
            _call("submit_answer", {"text": "Checked.",
                                    "claims": _CORRUPT_CLAIMS}, "call_bad"),
        ]),
        _assistant(tool_calls=[_call(
            "submit_answer", {"text": "Checked.", "claims": []}, "call_good")]),
    ])
    llm.tool_loop("question", ctx=vault, tools=[], ledger_path=ledger_bad,
                  tool_names=llm.COACH_TOOLS, submit_tool=True,
                  submit_repair=True)

    _scripted(openrouter, [
        _assistant(tool_calls=[
            _call("list_available_metrics", {}, "call_metrics"),
            _call("submit_answer", {"text": "Checked.", "claims": []},
                  "call_good"),
        ]),
    ])
    llm.tool_loop("question", ctx=vault, tools=[], ledger_path=ledger_good,
                  tool_names=llm.COACH_TOOLS, submit_tool=True,
                  submit_repair=True)

    repaired = [json.loads(line) for line in
                open(ledger_bad).read().splitlines() if line.strip()]
    assert [e["tool_name"] for e in repaired] == ["list_available_metrics"]
    # Nothing ran on the terminal-success path, so its ledger was never even
    # opened for writing.
    assert not os.path.exists(ledger_good)


def test_a_repaired_answer_with_no_data_call_still_fails_the_empty_ledger_gate(
        openrouter, monkeypatch, vault, conn):
    """A successful repair is not evidence of anything.

    The model corrupts its channel, is corrected, and re-emits a perfectly
    well-formed claim set — having read nothing. The ask gate refuses it for an
    empty ledger exactly as it refuses an uncorrupted submit-only answer.
    """
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    monkeypatch.setenv("HA_ASK_SUBMIT_REPAIR", "1")

    _scripted(openrouter, [
        _assistant(tool_calls=[_call(
            "submit_answer",
            {"text": "Steps averaged 5413.", "claims": _CORRUPT_CLAIMS},
            "call_bad")]),
        _assistant(tool_calls=[_call(
            "submit_answer",
            {"text": "Steps averaged 5413.", "claims": _CLAIMS},
            "call_good")]),
    ])

    result = chat.answer_question(vault, "How many steps?")
    assert result["mode"] == "fallback"
    assert result["tool_trace"] == []
    assert result["verification"]["reason"] == "ask answer has no tool-call ledger"


def test_submit_answer_is_still_absent_from_every_registry(vault):
    """Unchanged by the repair: it is not reachable as data anywhere."""
    assert "submit_answer" not in _REAL_REGISTRY(vault)
    assert "submit_answer" not in _REAL_REGISTRY(vault, include=llm.COACH_TOOLS)
    assert "submit_answer" not in llm.COACH_TOOLS
    assert "submit_answer" not in llm.RESEARCHER_TOOLS


# --------------------------------------------------------------------------
# the wire — the corrective result is a tool result like any other
# --------------------------------------------------------------------------

def test_the_corrective_tool_result_is_keyed_by_tool_call_id(openrouter):
    """OpenAI requires every assistant tool_calls id to be answered; an
    unanswered one makes the next request a 400. The repair is answered by id,
    beside the data call's own result, in the turn that called them."""
    bodies = _scripted(openrouter, [
        _assistant(tool_calls=[
            _call("list_available_metrics", {}, "call_metrics"),
            _call("submit_answer", {"text": "Checked.",
                                    "claims": _CORRUPT_CLAIMS}, "call_bad"),
        ]),
        _assistant(tool_calls=[_call(
            "submit_answer", {"text": "Checked.", "claims": []}, "call_good")]),
    ])

    llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                  submit_tool=True, submit_repair=True)

    sent = bodies[1]["messages"]
    assistant = [m for m in sent if m.get("role") == "assistant"][0]
    called_ids = {c["id"] for c in assistant["tool_calls"]}
    results = {m["tool_call_id"]: m for m in sent if m.get("role") == "tool"}
    assert called_ids == {"call_metrics", "call_bad"}
    assert called_ids <= set(results)
    assert results["call_bad"]["name"] == "submit_answer"
    assert "submit_answer was NOT accepted" in results["call_bad"]["content"]


def test_the_ollama_dialect_keys_the_corrective_result_by_name(monkeypatch):
    """The other dialect pairs a result by tool_name, and so does the repair."""
    monkeypatch.setattr(llm, "BACKEND", "ollama")
    monkeypatch.setattr(llm, "_registry", lambda ctx, include=None: {})
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(200, json={"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"function": {
                    "name": "submit_answer",
                    "arguments": {"text": "ok",
                                  "claims": _CORRUPT_CLAIMS}}}]}})
        return httpx.Response(200, json={"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {
                "name": "submit_answer",
                "arguments": {"text": "ok", "claims": []}}}]}})

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True, submit_repair=True)

    assert out.text == "ok"
    results = [m for m in bodies[1]["messages"] if m.get("role") == "tool"]
    assert [m["tool_name"] for m in results] == ["submit_answer"]


# --------------------------------------------------------------------------
# the corrective text, per failure shape
# --------------------------------------------------------------------------

_SHAPE = ("Required shape: call submit_answer again with exactly two arguments "
          "— `text`, a string of coach prose, and `claims`, a JSON ARRAY of "
          "claim objects (not a string containing an array, not an object). "
          "Re-send the same answer, correctly encoded.")


@pytest.mark.parametrize("arguments, event, problem", [
    ("{not json at all", "submit_answer_unparseable_arguments",
     "its arguments were not valid JSON (JSONDecodeError: Expecting property "
     "name enclosed in double quotes: line 1 column 2 (char 1)). The text "
     "around character 1 is: {not json at all"),
    ('["a", "list"]', "submit_answer_bad_arguments",
     "its arguments decoded to list, not an object."),
    ('{"text": 17, "claims": []}', "submit_answer_bad_text",
     "`text` was int, not a string."),
    ('{"claims": []}', "submit_answer_bad_text",
     "`text` was NoneType, not a string."),
    ('{"text": "ok", "claims": {"not": "a list"}}', "submit_answer_bad_claims",
     "`claims` was dict, not an array."),
    ('{"text": "ok"}', "submit_answer_bad_claims",
     "`claims` was NoneType, not an array."),
])
def test_the_corrective_message_names_the_shape_that_failed(arguments, event,
                                                            problem):
    """One specific sentence per failure shape, then the required shape."""
    response, got_event, _detail, repair = llm._submit_answer_decode(
        {"function": {"name": "submit_answer", "arguments": arguments}})
    assert response is None
    assert got_event == event
    assert repair == (
        f"submit_answer was NOT accepted and your answer was not delivered: "
        f"{problem} {_SHAPE}")


def test_the_corrective_message_excerpts_the_failure_offset_and_not_the_payload():
    """The offset is what repairs the serialisation; the corrupt payload is
    large, is corrupt, and echoing it invites the same garbage again."""
    padding = "x" * 400
    corrupt = ('[{"note": "' + padding + '", "sequence":  ript 19, '
               '"path": "$.result.points[2].value"}]')
    response, event, _detail, repair = llm._submit_answer_decode(
        {"function": {"name": "submit_answer",
                      "arguments": json.dumps({"text": "ok",
                                               "claims": corrupt})}})

    assert response is None
    assert event == "submit_answer_unparseable_claims"
    assert repair == (
        "submit_answer was NOT accepted and your answer was not delivered: "
        "`claims` arrived as a string, and that string was not valid JSON "
        "(JSONDecodeError: Expecting value: line 1 column 428 (char 427)). "
        "The text around character 427 is: ...xxxxxxxxxxxxxx"
        '", "sequence":  ript 19, "path": "$.result.poi... ' + _SHAPE)
    assert padding not in repair
    # The message does not grow with the payload: ten times the garbage gives
    # the same excerpt, at the same length.
    bigger = ('[{"note": "' + "x" * 4000 + '", "sequence":  ript 19, '
              '"path": "$.result.points[2].value"}]')
    _r, _e, _d, repair_bigger = llm._submit_answer_decode(
        {"function": {"name": "submit_answer",
                      "arguments": json.dumps({"text": "ok",
                                               "claims": bigger})}})
    # Only the printed offset's digits differ; the excerpt itself is identical.
    assert repair_bigger.count("x") == repair.count("x")
    # The only growth is the extra digit in the three printed offsets.
    assert len(repair_bigger) - len(repair) == 3
    assert len(repair_bigger) < 600


def test_a_well_formed_call_decodes_with_no_event_and_no_repair():
    """The repair path is reached only by a SHAPE failure, never on success."""
    response, event, _detail, repair = llm._submit_answer_decode(
        {"function": {"name": "submit_answer",
                      "arguments": json.dumps({"text": "Steps averaged 5413.",
                                               "claims": _CLAIMS})}})
    assert response.text == "Steps averaged 5413."
    assert response.claims == _CLAIMS
    assert event is None
    assert repair is None


def test_an_empty_text_answer_is_not_a_shape_failure_and_is_not_repaired(
        openrouter):
    """It decodes; it is simply empty. Repairing it would change an outcome the
    loop already reports, and the measured 2 of 26 are not a wire fault."""
    bodies = _scripted(openrouter, [
        _assistant(tool_calls=[_call(
            "submit_answer", {"text": "   ", "claims": []}, "call_blank")]),
        _assistant(tool_calls=[_call(
            "submit_answer", {"text": "recovered", "claims": []}, "call_two")]),
    ])

    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True, submit_repair=True)

    assert out.text == "   "
    assert len(bodies) == 1
    assert llm.last_loop_status()["outcome"] == "tool_loop_empty_answer"
    assert llm.last_loop_status()["submit_repairs_attempted"] == 0


# --------------------------------------------------------------------------
# the env flag, and the default that keeps every existing caller identical
# --------------------------------------------------------------------------

def test_the_env_flag_defaults_off_and_reads_1(monkeypatch):
    monkeypatch.delenv("HA_ASK_SUBMIT_REPAIR", raising=False)
    assert chat._submit_repair_enabled() is False
    monkeypatch.setenv("HA_ASK_SUBMIT_REPAIR", "0")
    assert chat._submit_repair_enabled() is False
    monkeypatch.setenv("HA_ASK_SUBMIT_REPAIR", " 1 ")
    assert chat._submit_repair_enabled() is True


@pytest.mark.parametrize("value, expected", [(None, False), ("1", True)])
def test_answer_question_passes_one_arm_to_both_attempts(monkeypatch, vault,
                                                         conn, value, expected):
    """A retry that saw a different arm than the draft it is fixing would not
    be measuring one thing — so both attempts carry the same flag."""
    if value is None:
        monkeypatch.delenv("HA_ASK_SUBMIT_REPAIR", raising=False)
    else:
        monkeypatch.setenv("HA_ASK_SUBMIT_REPAIR", value)

    seen = []

    def fake_tool_loop(prompt, **kwargs):
        seen.append(kwargs.get("submit_repair"))
        return llm.ResearchResponse("", None)

    monkeypatch.setattr(llm, "tool_loop", fake_tool_loop)
    chat.answer_question(vault, "How many steps?")
    assert seen == [expected, expected]


def test_the_flag_off_leaves_the_wire_payload_untouched(openrouter):
    """Byte-identical: the repair adds nothing to what is sent when it is off."""
    off = _scripted(openrouter, [_assistant("done")])
    llm.tool_loop("q", ctx=None, tools=[{"type": "function"}], max_turns=1,
                  submit_tool=True)

    on = _scripted(openrouter, [_assistant("done")])
    llm.tool_loop("q", ctx=None, tools=[{"type": "function"}], max_turns=1,
                  submit_tool=True, submit_repair=True)

    assert off[0] == on[0]


# --------------------------------------------------------------------------
# #214 — the TRUNCATED-ARGUMENTS signature measured live on 2026-08-29
#
# A different failure from the 2026-08-28 corruption above. There the outer
# `arguments` object decoded and the `claims` STRING inside it did not; here
# the provider truncates the tool-call arguments mid-value, so `arguments`
# does not decode at all and the loop lands on
# `submit_answer_unparseable_arguments`.
#
# Measured on `ae443f7`, live against together + deepseek/deepseek-v4-flash-0731
# with reasoning off: 2 of 18 ask calls (11%) arrived this way, both with the
# byte-identical error below, and both with `submit_repairs_attempted: 0`
# because HA_ASK_SUBMIT_REPAIR defaults off. These tests are what stands in for
# a live battery: the failure is stochastic, so it is INJECTED here rather than
# waited for, and nothing below touches the network or spends a credit.
# --------------------------------------------------------------------------

# Reproduced byte-for-byte from the live `detail`, not paraphrased:
#   JSONDecodeError: Unterminated string starting at: line 1 column 9 (char 8)
# `{"text":"` puts the opening quote of the truncated value at char 8.
_TRUNCATED_ARGUMENTS = '{"text":"Steps averaged 5413 last week'

_LIVE_DETAIL = ("JSONDecodeError: Unterminated string starting at: "
                "line 1 column 9 (char 8)")

# Every shape a truncating provider produced or could produce for one
# submit_answer: the live signature first, then truncations at other offsets.
# All of them fail the OUTER decode, which is the path #214 is about.
_TRUNCATIONS = [
    _TRUNCATED_ARGUMENTS,
    '{"text": "Steps averaged 5413.", "claims": [{"metric": "steps"',
    '{"text": "Steps averaged 5413.", "claims',
    '{"text": "Steps averaged 5413.", "claims": [',
    '{"text":"',
    '{',
]


def _good_submit(call_id="call_good"):
    return _assistant(tool_calls=[_call(
        "submit_answer",
        {"text": "Steps averaged 5413.", "claims": _CLAIMS}, call_id)])


def _truncated_then_good(openrouter, truncated):
    """Turn 1 is a truncated submit_answer; turn 2 is the same answer, intact."""
    return _scripted(openrouter, [
        _assistant(tool_calls=[_call("submit_answer", truncated, "call_bad")]),
        _good_submit(),
    ])


def test_the_injected_truncation_reproduces_the_live_char_8_signature():
    """The injection is the observed failure, not a lookalike.

    If this ever stops matching the live `detail` string, the rest of this
    section is testing something #214 did not measure.
    """
    response, event, detail, repair = llm._submit_answer_decode(
        {"function": {"name": "submit_answer",
                      "arguments": _TRUNCATED_ARGUMENTS}})

    assert response is None
    assert event == "submit_answer_unparseable_arguments"
    assert detail == _LIVE_DETAIL
    # The corrective text names the failure and shows the offset, so the model
    # is told WHERE it was cut off rather than handed its own garbage back.
    assert "not valid JSON" in repair
    assert "character 8" in repair
    assert "call submit_answer again" in repair


def test_a_truncated_submit_answer_is_repaired_and_the_answer_is_delivered(
        openrouter):
    """Done-when 1, with the flag ON: the loop recovers and re-emits.

    The truncated call is answered with a corrective tool result instead of
    ending the loop, and the well-formed submit_answer that follows is what
    the caller receives.
    """
    bodies = _truncated_then_good(openrouter, _TRUNCATED_ARGUMENTS)

    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True, submit_repair=True)

    assert out.text == "Steps averaged 5413."
    assert out.claims == _CLAIMS
    assert len(bodies) == 2
    status = llm.last_loop_status()
    # A REPAIRED failure is deliberately not announced as a loop non-answer:
    # the loop did answer. It stays visible only in the repair counters, which
    # is the channel `last_loop_status` documents for it.
    assert status["outcome"] == "not_called"
    assert status["submit_repairs_attempted"] == 1
    assert status["submit_repairs_succeeded"] == 1

    # The second request carries the correction as a tool result keyed to the
    # truncated call, and nothing else about the exchange changed.
    correction = [m for m in bodies[1]["messages"]
                  if m.get("role") == "tool" and m.get("tool_call_id") == "call_bad"]
    assert len(correction) == 1
    assert "was NOT accepted" in correction[0]["content"]


def test_with_the_flag_off_the_same_truncation_loses_the_answer(openrouter):
    """The companion that makes the test above mean something.

    Same injection, same scripted recovery available on turn 2 — with the flag
    off the loop never asks for it. This is the live 11%: the answer existed
    and did not reach the user.
    """
    bodies = _truncated_then_good(openrouter, _TRUNCATED_ARGUMENTS)

    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                        submit_tool=True)

    assert out == ""
    assert out.claims is None
    assert len(bodies) == 1
    status = llm.last_loop_status()
    assert status["outcome"] == "submit_answer_unparseable_arguments"
    assert status["detail"] == _LIVE_DETAIL
    assert status["submit_repairs_attempted"] == 0


def test_every_injected_unparseable_submit_is_delivered_with_the_flag_on(
        openrouter):
    """Done-when 2, derived rather than asserted.

    Runs the whole truncation table twice — once with the repair on, once off —
    and counts how many end in a delivered answer. The recovery share is 6/6
    (100%) with the flag on and 0/6 (0%) with it off. Deterministic: the model
    turn that follows the correction is scripted to be well formed, so this is
    the ceiling the repair can reach, not a prediction of what the live model
    will re-emit.
    """
    delivered_on = 0
    delivered_off = 0
    for truncated in _TRUNCATIONS:
        _truncated_then_good(openrouter, truncated)
        assert llm._submit_answer_decode(
            {"function": {"name": "submit_answer",
                          "arguments": truncated}})[1] == \
            "submit_answer_unparseable_arguments"
        on = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                           submit_tool=True, submit_repair=True)
        if on.text == "Steps averaged 5413." and on.claims == _CLAIMS:
            delivered_on += 1

        _truncated_then_good(openrouter, truncated)
        off = llm.tool_loop("question", ctx=None, tools=[], max_turns=4,
                            submit_tool=True)
        if off.text == "Steps averaged 5413.":
            delivered_off += 1

    assert delivered_on == len(_TRUNCATIONS) == 6
    assert delivered_off == 0


def test_the_repair_budget_bounds_a_provider_that_always_truncates(openrouter):
    """The invariant that keeps the repair from becoming a spend loop.

    A provider that truncates EVERY submit_answer is exactly the case an
    unbounded retry would burn credit on. The default budget is 2, so the loop
    makes three requests and stops — not the eight max_turns allows.
    """
    bodies = _scripted(openrouter, [
        _assistant(tool_calls=[_call("submit_answer", _TRUNCATED_ARGUMENTS,
                                     "call_bad")]),
    ])

    out = llm.tool_loop("question", ctx=None, tools=[], max_turns=8,
                        submit_tool=True, submit_repair=True)

    assert out == ""
    assert out.claims is None
    assert len(bodies) == 1 + llm.SUBMIT_ANSWER_REPAIR_BUDGET == 3
    status = llm.last_loop_status()
    assert status["outcome"] == "submit_answer_repair_exhausted"
    assert "submit_answer_unparseable_arguments" in status["detail"]
    assert status["submit_repairs_attempted"] == llm.SUBMIT_ANSWER_REPAIR_BUDGET
    assert status["submit_repairs_succeeded"] == 0


def test_a_truncated_repair_writes_no_ledger_entry_for_submit_answer(
        openrouter, monkeypatch, vault, tmp_path):
    """The invariant: a repair turn adds no provenance the model can cite.

    `submit_answer` is deliberately absent from `_registry`, so the corrective
    result goes straight into `messages` and never through `_ledgered`. A real
    tool called in the same turn IS ledgered; the truncated submit_answer
    contributes nothing, on either the failed turn or the recovered one.
    """
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    ledger_path = str(tmp_path / "ledger.jsonl")

    _scripted(openrouter, [
        _assistant(tool_calls=[
            _call("list_available_metrics", {}, "call_metrics"),
            _call("submit_answer", _TRUNCATED_ARGUMENTS, "call_bad"),
        ]),
        _assistant(tool_calls=[_call(
            "submit_answer", {"text": "Checked.", "claims": []},
            "call_good")]),
    ])

    out = llm.tool_loop("question", ctx=vault, tools=[],
                        ledger_path=ledger_path, tool_names=llm.COACH_TOOLS,
                        submit_tool=True, submit_repair=True)

    assert out.text == "Checked."
    entries = [json.loads(line) for line in
               open(ledger_path).read().splitlines() if line.strip()]
    assert [e["tool_name"] for e in entries] == ["list_available_metrics"]
    assert "submit_answer" not in {e["tool_name"] for e in entries}


def test_a_truncated_repair_still_faces_the_empty_ledger_gate(
        openrouter, monkeypatch, vault, conn):
    """Python still owns the truth after a repair.

    The model is cut off, corrected, and re-emits a perfectly well-formed claim
    set having read nothing. The ask gate refuses it for an empty ledger,
    exactly as it refuses an uncorrupted submit-only answer — so the repair
    cannot be used as a channel for unverified claims.
    """
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    monkeypatch.setenv("HA_ASK_SUBMIT_REPAIR", "1")

    _scripted(openrouter, [
        _assistant(tool_calls=[_call("submit_answer", _TRUNCATED_ARGUMENTS,
                                     "call_bad")]),
        _good_submit(),
    ])

    result = chat.answer_question(vault, "How many steps?")
    assert result["mode"] == "fallback"
    assert result["tool_trace"] == []
    assert result["verification"]["reason"] == "ask answer has no tool-call ledger"
