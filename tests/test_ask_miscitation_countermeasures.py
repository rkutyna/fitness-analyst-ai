"""Three countermeasures against the /v1/ask miscitation defect.

Measured 2026-08-28, reasoning off, deepseek-v4-flash on coreweave/fp8: the ask
path files STRUCTURALLY PERFECT claims that cite the WRONG tool call. One
attempt produced 11 well-formed claims of which 2 verified — it cited sequence
13 for `sleep_time_in_bed` and sequence 2 for `sleep_asleep`, having mixed up
which of its 13 tool calls returned what. `_resolve_ledger_value` looks up only
the record with the cited sequence, so a number published verbatim one record
away is refused with "claim metric does not match ledger field".

Separately, the retry attempt frequently corrupted its own output: `claims`
arrived as a JSON *string* containing malformed JSON, with a stray token
injected mid-object. 3 of 6 retries returned nothing at all that way.

Three things are pinned here, and none of them moves a verification decision
out of Python:

1. `HA_OPENROUTER_REASONING=low` — a third mode the run may STATE. D15's
   fail-closed shape is unchanged (see test_llm_backend_approval.py).
2. The ledger index — Python tells the model which sequence published which
   metric, instead of asking it to remember across a dozen turns. Computed from
   the ledger Python wrote, using the verifier's own `_ledger_scopes`.
3. A typed `claims` item schema — a serialisation hint for the provider, with
   no `required`, no enum and no constraint that could decide legality.

Nothing here touches the network: every request is served by an
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
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def _bodies_for(openrouter, **loop_kwargs):
    """Run one tool-less turn and return the request bodies that went out."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm.tool_loop("q", ctx=None, tools=[], max_turns=1, **loop_kwargs)
    return bodies


# ---------------------------------------------------------------------------
# 1 — HA_OPENROUTER_REASONING=low, on the wire
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode, expected", [
    ("on", {"enabled": True}),
    ("off", {"enabled": False}),
    ("low", {"effort": "low"}),
])
def test_each_stated_reasoning_mode_sends_its_own_wire_form(openrouter,
                                                            monkeypatch,
                                                            mode, expected):
    """`low` is an EFFORT level; the boolean form cannot express it.

    `on`/`off` keep `{"enabled": ...}` byte-for-byte, so every existing run's
    payload is unchanged; only a stated `low` sends `{"effort": "low"}`.
    """
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", mode)
    bodies = _bodies_for(openrouter)
    assert bodies[0]["reasoning"] == expected


@pytest.mark.parametrize("mode, expected", [
    ("on", {"enabled": True}),
    ("off", {"enabled": False}),
    ("low", {"effort": "low"}),
])
def test_complete_sends_the_same_reasoning_form_as_the_tool_loop(openrouter,
                                                                 monkeypatch,
                                                                 mode, expected):
    """The single-shot path and the loop must not disagree about the wire."""
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", mode)
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_assistant("hello"))

    openrouter(handler)
    assert llm.complete("q") == "hello"
    assert bodies[0]["reasoning"] == expected


@pytest.mark.parametrize("value", [None, "", "medium", "LOW"])
def test_an_unstated_or_invalid_reasoning_mode_still_refuses(monkeypatch, value):
    """No default in code. Adding a third stated value added no fallback."""
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", value)
    with pytest.raises(RuntimeError, match="HA_OPENROUTER_REASONING"):
        llm._openrouter_reasoning_mode()


def test_the_reasoning_mode_parser_does_not_return_a_bool(monkeypatch):
    """Three states do not fit in two.

    A bool would make `low` indistinguishable from `on` at the call sites that
    choose a timeout and build the wire field.
    """
    for mode in ("on", "off", "low"):
        monkeypatch.setattr(llm, "OPENROUTER_REASONING", mode)
        parsed = llm._openrouter_reasoning_mode()
        assert parsed == mode
        assert not isinstance(parsed, bool)


# ---------------------------------------------------------------------------
# 2 — the Python-computed ledger index
# ---------------------------------------------------------------------------

def _write_ledger(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    return str(path)


_MISCITED_LEDGER = [
    {"sequence": 2, "tool_name": "get_metric_summary",
     "arguments": {"metric": "sleep_asleep", "days": 7},
     "result": {"metric": "sleep_asleep", "period": "2026-08-01:2026-08-07",
                "mean": 6.82, "n_days": 7, "unit": "hours"},
     "result_elided": False},
    {"sequence": 13, "tool_name": "get_metric_summary",
     "arguments": {"metric": "sleep_time_in_bed", "days": 7},
     "result": {"metric": "sleep_time_in_bed",
                "period": "2026-08-01:2026-08-07",
                "mean": 7.51, "n_days": 7, "unit": "hours"},
     "result_elided": False},
]


def test_the_index_states_the_true_sequence_to_metric_mapping(tmp_path):
    """The exact confusion measured: 2 is sleep_asleep, 13 is time_in_bed.

    The model swapped them. Python never had to guess.
    """
    path = _write_ledger(tmp_path / "ledger.jsonl", _MISCITED_LEDGER)
    text = llm._ledger_index_text(path)

    assert "sequence 2 — get_metric_summary" in text
    assert ("  metric=sleep_asleep period=2026-08-01:2026-08-07 field=mean "
            "value=6.82 path=$.result.mean") in text
    assert "sequence 13 — get_metric_summary" in text
    assert ("  metric=sleep_time_in_bed period=2026-08-01:2026-08-07 "
            "field=mean value=7.51 path=$.result.mean") in text


def test_the_index_is_derived_from_the_verifier_s_own_scope_function(tmp_path,
                                                                     monkeypatch):
    """It must show the SAME vocabulary the verifier will check against.

    A second scope extractor would drift from `_resolve_ledger_value`, and a
    vocabulary the model cannot read is the #93 defect one layer up. Pinned by
    replacing `_ledger_scopes` and watching the index change with it.
    """
    from health_advisor import deepdive_verify as DV

    monkeypatch.setattr(DV, "_ledger_scopes", lambda record: [
        {"metric": "sentinel", "period": None, "field": "f", "value": 1,
         "path": "$.result.f", "kind": "result"}])
    path = _write_ledger(tmp_path / "ledger.jsonl", _MISCITED_LEDGER)

    text = llm._ledger_index_text(path)
    assert "metric=sentinel field=f value=1 path=$.result.f" in text
    assert "sleep_asleep" not in text


def test_the_index_never_offers_an_arguments_path(tmp_path):
    """`_resolve_ledger_value` refuses `$.arguments...` outright.

    `_ledger_scopes` walks it anyway, so listing it would be teaching a
    citation that is rejected by construction.
    """
    path = _write_ledger(tmp_path / "ledger.jsonl", _MISCITED_LEDGER)
    text = llm._ledger_index_text(path)
    assert "$.arguments" not in text
    assert "$.result" in text


def test_the_index_truncates_visibly_at_the_per_record_cap(tmp_path):
    """A record can publish hundreds of leaves; the index is re-sent per turn.

    Truncation must be stated in the text, never silent — the model is told how
    many values it cannot see and what to do about it.
    """
    leaves = 3 * llm.LEDGER_INDEX_MAX_LEAVES
    record = {"sequence": 1, "tool_name": "get_metric_series",
              "arguments": {"metric": "steps"},
              "result": {"metric": "steps",
                         "points": [{"value": i} for i in range(leaves)]},
              "result_elided": False}
    path = _write_ledger(tmp_path / "ledger.jsonl", [record])

    text = llm._ledger_index_text(path)
    listed = [line for line in text.splitlines()
              if line.startswith("  ") and "path=$." in line]
    assert len(listed) == llm.LEDGER_INDEX_MAX_LEAVES
    hidden = leaves + 1 - llm.LEDGER_INDEX_MAX_LEAVES  # +1 for $.result.metric
    assert f"and {hidden} more citable values in this call are NOT listed" in text
    assert str(llm.LEDGER_INDEX_MAX_LEAVES) in text


def test_an_elided_record_says_so_instead_of_looking_empty(tmp_path):
    """`_resolve_ledger_value` refuses an elided record; saying nothing about
    it would read as "this call published nothing"."""
    path = _write_ledger(tmp_path / "ledger.jsonl", [
        {"sequence": 4, "tool_name": "list_workouts", "arguments": {},
         "result": {"_elided": True, "bytes": 900000}, "result_elided": True}])

    text = llm._ledger_index_text(path)
    assert "sequence 4 — list_workouts" in text
    assert "nothing in this call is citable" in text


def test_an_unreadable_or_empty_ledger_yields_no_index_and_no_raise(tmp_path):
    """The loop promises never to lose a conversation to a file error."""
    assert llm._ledger_index_text(str(tmp_path / "missing.jsonl")) == ""
    empty = _write_ledger(tmp_path / "empty.jsonl", [])
    assert llm._ledger_index_text(empty) == ""


def test_the_index_message_is_appended_only_when_the_flag_is_on(openrouter,
                                                                monkeypatch,
                                                                vault, tmp_path):
    """Default off: an existing caller's second-turn payload is unchanged."""
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    bodies = []

    def run(ledger_index):
        bodies.clear()
        turns = {"n": 0}

        def handler(request):
            bodies.append(json.loads(request.content))
            turns["n"] += 1
            if turns["n"] == 1:
                return httpx.Response(200, json=_assistant(tool_calls=[
                    _call("list_available_metrics", {}, "call_metrics")]))
            return httpx.Response(200, json=_assistant("done"))

        openrouter(handler)
        llm.tool_loop("q", ctx=vault, tools=[], max_turns=3,
                      ledger_path=str(tmp_path / f"l-{ledger_index}.jsonl"),
                      tool_names=llm.COACH_TOOLS, ledger_index=ledger_index)
        return [m["content"] for m in bodies[1]["messages"]]

    off_contents = run(False)
    on_contents = run(True)

    assert not any("LEDGER INDEX" in str(c) for c in off_contents)
    index = [c for c in on_contents if "LEDGER INDEX" in str(c)]
    assert len(index) == 1
    assert "sequence 1 — list_available_metrics" in index[0]


def test_the_index_is_computed_from_the_ledger_not_from_the_model(openrouter,
                                                                  monkeypatch,
                                                                  vault,
                                                                  tmp_path):
    """A tool the model names but that never ran leaves no ledger record, so
    it cannot appear in the index — the index is evidence, not echo."""
    monkeypatch.setattr(llm, "_registry", _REAL_REGISTRY)
    bodies = []
    turns = {"n": 0}

    def handler(request):
        bodies.append(json.loads(request.content))
        turns["n"] += 1
        if turns["n"] == 1:
            return httpx.Response(200, json=_assistant(tool_calls=[
                _call("no_such_tool", {}, "call_ghost")]))
        return httpx.Response(200, json=_assistant("done"))

    openrouter(handler)
    llm.tool_loop("q", ctx=vault, tools=[], max_turns=3,
                  ledger_path=str(tmp_path / "ledger.jsonl"),
                  tool_names=llm.COACH_TOOLS, ledger_index=True)

    assert not any("LEDGER INDEX" in str(m.get("content"))
                   for m in bodies[1]["messages"])


def test_answer_question_reads_the_arm_from_the_env_flag(monkeypatch):
    """`HA_ASK_LEDGER_INDEX` is read per call, and defaults OFF."""
    monkeypatch.delenv("HA_ASK_LEDGER_INDEX", raising=False)
    assert chat._ledger_index_enabled() is False
    monkeypatch.setenv("HA_ASK_LEDGER_INDEX", "0")
    assert chat._ledger_index_enabled() is False
    monkeypatch.setenv("HA_ASK_LEDGER_INDEX", "1")
    assert chat._ledger_index_enabled() is True


def test_the_ask_path_passes_the_flag_through_to_the_loop(openrouter,
                                                          monkeypatch, vault,
                                                          conn):
    """Both attempts run the same arm; a retry on a different prompt shape
    would not be measuring one thing."""
    seen = []
    real_tool_loop = llm.tool_loop

    def spy(prompt, **kwargs):
        seen.append(kwargs.get("ledger_index"))
        return real_tool_loop(prompt, **kwargs)

    monkeypatch.setattr(llm, "tool_loop", spy)
    openrouter(lambda request: httpx.Response(200, json=_assistant("")))

    monkeypatch.setenv("HA_ASK_LEDGER_INDEX", "1")
    chat.answer_question(vault, "How did I sleep?")
    assert seen == [True, True]

    seen.clear()
    monkeypatch.setenv("HA_ASK_LEDGER_INDEX", "0")
    chat.answer_question(vault, "How did I sleep?")
    assert seen == [False, False]


# ---------------------------------------------------------------------------
# 3 — the typed claims item schema
# ---------------------------------------------------------------------------

def test_the_claim_item_schema_names_the_keys_the_grammar_uses():
    """A serialisation hint: the provider is given a shape to fill."""
    item = llm.SUBMIT_ANSWER_TOOL["function"]["parameters"]["properties"][
        "claims"]["items"]
    assert item is llm.CLAIM_ITEM_SCHEMA
    assert item["type"] == "object"
    assert set(item["properties"]) == {"metric", "period", "field", "value",
                                       "source"}
    assert item["properties"]["metric"]["type"] == "string"
    assert item["properties"]["field"]["type"] == "string"
    source = item["properties"]["source"]
    assert source["type"] == "object"
    assert source["properties"]["sequence"]["type"] == "integer"
    assert source["properties"]["path"]["type"] == "string"


def test_the_claim_item_schema_decides_nothing_about_legality():
    """No `required`, no enum, no format, no constraint anywhere in it.

    The claim grammar stays in prose and the verdict stays in Python. `period`
    and `value` carry no `type` at all, because both are legitimately several
    JSON types and a type here would refuse a legal claim at the wire instead
    of at the gate.
    """
    def walk(node):
        if isinstance(node, dict):
            for key in ("required", "enum", "const", "format", "pattern",
                        "minimum", "maximum", "minItems", "additionalProperties"):
                assert key not in node, f"{key} encodes legality in the schema"
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(llm.CLAIM_ITEM_SCHEMA)
    assert "type" not in llm.CLAIM_ITEM_SCHEMA["properties"]["period"]
    assert "type" not in llm.CLAIM_ITEM_SCHEMA["properties"]["value"]


def test_a_claim_with_metric_omitted_still_round_trips(openrouter):
    """The grammar REQUIRES omitting `metric` for a list_workouts row.

    A schema that made it required would have made every legal workout claim
    unsendable.
    """
    workout_claim = {"period": None, "field": "duration_min", "value": 42,
                     "source": {"sequence": 3,
                                "path": "$.result.workouts[11].duration_min"}}

    def handler(request):
        return httpx.Response(200, json=_assistant(tool_calls=[
            _call("submit_answer", {"text": "That session ran 42 minutes.",
                                    "claims": [workout_claim]})]))

    openrouter(handler)
    out = llm.tool_loop("q", ctx=None, tools=[], max_turns=4, submit_tool=True)
    assert out.claims == [workout_claim]
    assert "metric" not in out.claims[0]


def test_the_submit_answer_required_list_is_unchanged():
    """The item schema is new; what the CALL requires is not."""
    params = llm.SUBMIT_ANSWER_TOOL["function"]["parameters"]
    assert sorted(params["required"]) == ["claims", "text"]


def test_submit_answer_is_still_absent_from_the_tool_registry(vault):
    """The typed schema is a wire hint, not a data tool. The zero-tool-call
    loophole stays closed."""
    assert "submit_answer" not in _REAL_REGISTRY(vault)
    assert "submit_answer" not in _REAL_REGISTRY(vault, include=llm.COACH_TOOLS)
    assert "submit_answer" not in llm.COACH_TOOLS
    assert "submit_answer" not in llm.RESEARCHER_TOOLS
