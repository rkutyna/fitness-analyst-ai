"""Explicit, opt-in replacement of one ENGINE tool on the ask path (issue #650).

A host replaces an engine tool by name with ``llm.register_replacement``. The
collision refusal on the default path (``llm.register_tools``) is unchanged:
registering an engine tool's name through it still raises.

Ledger coverage: a replacement is put into the same registry dict that
``_registry`` returns, and the ask path wraps that whole dict in ``_ledgered``
(llm.py, the ``reg = _ledgered(base_registry, ledger_path)`` line). So a
replaced tool's call is recorded in the call ledger exactly like an engine
tool's. ``test_replacement_call_is_ledgered_like_an_engine_tool`` checks it.
"""
from __future__ import annotations

import json

import pytest

from health_advisor import context, llm

ENGINE_TOOL = "get_sleep_regularity"
REPLACED_DOC = "REPLACED DOCSTRING: the host's own sleep-regularity default."

_captured_ctx: list = []


def get_sleep_regularity(ctx, start: str | None = None, end: str | None = None) -> dict:
    """REPLACED DOCSTRING: the host's own sleep-regularity default."""
    _captured_ctx.append(ctx)
    return {"replaced": True, "start": start, "end": end}


@pytest.fixture(autouse=True)
def _clean_registries():
    before = set(llm.registered_replacements())
    before_host = set(llm.registered_host_tools())
    yield
    llm.unregister_replacements(*(set(llm.registered_replacements()) - before))
    llm.unregister_tools(*(set(llm.registered_host_tools()) - before_host))
    _captured_ctx.clear()


def _schema(registry, name):
    return registry[name][1]["function"]


def test_without_a_replacement_the_engine_tool_is_offered_and_not_replaced(conn, vault):
    assert llm.registered_replacements() == ()
    reg = llm._registry(vault, include={ENGINE_TOOL})
    assert set(reg) == {ENGINE_TOOL}
    assert "REPLACED DOCSTRING" not in _schema(reg, ENGINE_TOOL)["description"]
    result = reg[ENGINE_TOOL][0]()
    assert isinstance(result, dict) and "replaced" not in result


def test_registered_replacement_supplies_description_and_callable(vault):
    llm.register_replacement(get_sleep_regularity)
    reg = llm._registry(vault, include={ENGINE_TOOL})
    assert "REPLACED DOCSTRING" in _schema(reg, ENGINE_TOOL)["description"]
    result = reg[ENGINE_TOOL][0]()
    assert result["replaced"] is True
    assert result == {"replaced": True, "start": None, "end": None}
    # The schema comes from the replacement's own signature: ctx is hidden,
    # and the parameters are the replacement's start/end.
    assert set(_schema(reg, ENGINE_TOOL)["parameters"]["properties"]) == {"start", "end"}


def test_replacement_leaves_the_registry_key_set_unchanged(vault):
    include = {ENGINE_TOOL, "get_latest"}
    before = set(llm._registry(vault, include=include))
    llm.register_replacement(get_sleep_regularity)
    after = set(llm._registry(vault, include=include))
    assert before == after == {ENGINE_TOOL, "get_latest"}


def test_replacement_receives_the_provider_facing_context(vault):
    assert context.RAW_SAMPLES in vault.capabilities        # not vacuous
    assert context.WRITE in vault.capabilities
    llm.register_replacement(get_sleep_regularity)
    reg = llm._registry(vault, include={ENGINE_TOOL})
    reg[ENGINE_TOOL][0]()
    (seen,) = _captured_ctx
    assert seen == vault.provider_facing()
    assert context.RAW_SAMPLES not in seen.capabilities
    assert context.WRITE not in seen.capabilities


def test_unselected_replacement_is_not_present(vault):
    llm.register_replacement(get_sleep_regularity)
    reg = llm._registry(vault, include={"get_latest"})
    assert set(reg) == {"get_latest"}
    assert ENGINE_TOOL not in reg
    assert _captured_ctx == []


def test_register_replacement_refuses_a_non_engine_name():
    def get_not_an_engine_tool(ctx) -> dict:
        """Not in the engine."""
        return {}
    with pytest.raises(ValueError, match="is not an engine tool; use register_tools"):
        llm.register_replacement(get_not_an_engine_tool)
    assert "get_not_an_engine_tool" not in llm.registered_replacements()


def test_a_second_different_function_under_the_same_name_is_refused():
    def other(ctx, start=None, end=None) -> dict:
        """A different function under the engine name."""
        return {}
    other.__name__ = ENGINE_TOOL
    llm.register_replacement(get_sleep_regularity)
    with pytest.raises(ValueError, match="already registered by another function"):
        llm.register_replacement(other)
    assert llm.registered_replacements() == (ENGINE_TOOL,)


def test_registering_the_same_function_twice_is_idempotent():
    assert llm.register_replacement(get_sleep_regularity) == ENGINE_TOOL
    assert llm.register_replacement(get_sleep_regularity) == ENGINE_TOOL
    assert llm.registered_replacements() == (ENGINE_TOOL,)


def test_register_tools_still_refuses_an_engine_name_with_a_replacement_present():
    llm.register_replacement(get_sleep_regularity)
    with pytest.raises(ValueError, match="collides with an engine tool"):
        llm.register_tools(get_sleep_regularity)
    assert ENGINE_TOOL not in llm.registered_host_tools()


def test_register_tools_still_refuses_an_engine_name_without_a_replacement():
    with pytest.raises(ValueError, match="collides with an engine tool"):
        llm.register_tools(get_sleep_regularity)
    assert llm.registered_replacements() == ()


def test_unregister_replacements_restores_the_engine_tool(conn, vault):
    llm.register_replacement(get_sleep_regularity)
    llm.unregister_replacements(ENGINE_TOOL, "never_registered")   # unknown ignored
    assert llm.registered_replacements() == ()
    reg = llm._registry(vault, include={ENGINE_TOOL})
    assert "REPLACED DOCSTRING" not in _schema(reg, ENGINE_TOOL)["description"]
    assert "replaced" not in reg[ENGINE_TOOL][0]()


def test_replacement_call_is_ledgered_like_an_engine_tool(vault, tmp_path):
    llm.register_replacement(get_sleep_regularity)
    ledger_path = tmp_path / "ledger.jsonl"
    reg = llm._ledgered(llm._registry(vault, include={ENGINE_TOOL}), ledger_path)
    result = reg[ENGINE_TOOL][0]()
    assert result["replaced"] is True
    assert "_ledger" in result and "sequence" in result["_ledger"]
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [r["tool_name"] for r in records] == [ENGINE_TOOL]
