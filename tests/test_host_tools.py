"""Host-supplied coach tools and the fail-loud include list (issue #555).

``llm.COACH_TOOLS`` names three plan tools the engine does not implement. Until
a host could register them, ``tool_schemas(include=COACH_TOOLS)`` dropped them
silently and the ask path answered plan questions from metrics alone.
"""
from __future__ import annotations

import json

import pytest

from health_advisor import context, llm, mcp_server


def host_probe(ctx, day: str = "today") -> dict:
    """A host tool: reports whether the context it was bound to can write."""
    return {"found": True, "requested": day,
            "can_write": ctx.can(context.WRITE),
            "can_raw": ctx.can(context.RAW_SAMPLES)}


def get_week_plan(ctx, day: str = "today", detail: bool = False) -> dict:
    """Stands in for a host's declared plan tool."""
    return {"found": True, "requested": day}


@pytest.fixture(autouse=True)
def _clean_host_registry():
    before = llm.registered_host_tools()
    yield
    llm.unregister_tools(*(set(llm.registered_host_tools()) - set(before)))


def _offered(vault, include):
    return {s["function"]["name"] for s in llm.tool_schemas(vault, include=include)}


def test_unknown_include_name_raises_and_names_it(vault):
    with pytest.raises(llm.UnregisteredToolError, match="get_wek_plan"):
        llm.tool_schemas(vault, include=("get_latest", "get_wek_plan"))


def test_unknown_include_name_raises_on_the_tool_loop_path_too(vault):
    with pytest.raises(llm.UnregisteredToolError, match="no_such_tool"):
        llm._registry(vault, include=("no_such_tool",))


def test_one_registered_engine_tool_is_not_an_error(vault):
    assert _offered(vault, ("get_latest",)) == {"get_latest"}


def test_declared_host_tool_absent_from_a_bare_engine_is_omitted_not_raised(vault):
    # An engine with no host is a legitimate configuration, so this cannot raise;
    # the host's own startup check (require_host_tools) is what refuses.
    assert "get_week_plan" not in llm.registered_host_tools()
    assert _offered(vault, ("get_latest", "get_week_plan")) == {"get_latest"}


def test_bare_engine_start_up_check_raises_and_names_every_missing_tool():
    assert llm.missing_host_tools() == llm.HOST_SUPPLIED_TOOLS
    with pytest.raises(llm.UnregisteredToolError) as err:
        llm.require_host_tools()
    for name in llm.HOST_SUPPLIED_TOOLS:
        assert name in str(err.value)


def test_registered_host_tool_is_offered_with_a_schema_that_hides_ctx(vault):
    llm.register_tools(get_week_plan)
    schemas = {s["function"]["name"]: s for s in
               llm.tool_schemas(vault, include=("get_latest", "get_week_plan"))}
    assert set(schemas) == {"get_latest", "get_week_plan"}
    params = schemas["get_week_plan"]["function"]["parameters"]
    assert set(params["properties"]) == {"day", "detail"}     # no ctx


def test_require_host_tools_passes_once_registered(vault):
    llm.register_tools(host_probe)
    llm.require_host_tools(("host_probe",))
    assert llm.missing_host_tools(("host_probe", "get_week_plan")) == ("get_week_plan",)


def test_host_tool_is_bound_to_the_provider_facing_context(vault):
    assert vault.can(context.WRITE)
    llm.register_tools(host_probe)
    registry = llm._registry(vault, include=("host_probe",))
    result = registry["host_probe"][0](day="tomorrow")
    assert result == {"found": True, "requested": "tomorrow",
                      "can_write": False, "can_raw": False}


def test_host_tool_calls_are_written_to_the_same_ledger_as_engine_tools(
        vault, tmp_path):
    llm.register_tools(host_probe)
    ledger = tmp_path / "ledger.jsonl"
    registry = llm._ledgered(
        llm._registry(vault, include=("host_probe", "get_latest")), str(ledger))
    registry["host_probe"][0](day="today")
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [(r["sequence"], r["tool_name"]) for r in records] == [(1, "host_probe")]
    assert records[0]["result"]["found"] is True


def test_registration_is_idempotent_and_refuses_collisions(vault):
    assert llm.register_tools(host_probe) == ("host_probe",)
    assert llm.register_tools(host_probe) == ("host_probe",)

    def other(ctx):
        return {}
    other.__name__ = "host_probe"
    with pytest.raises(ValueError, match="already registered"):
        llm.register_tools(other)

    def get_latest(ctx):
        return {}
    with pytest.raises(ValueError, match="engine tool"):
        llm.register_tools(get_latest)

    def cite(ctx):
        return {}
    with pytest.raises(ValueError, match="synthetic"):
        llm.register_tools(cite)


def test_registering_does_not_touch_the_engines_own_tool_register():
    before = [fn.__name__ for fn in mcp_server._TOOLS]
    llm.register_tools(host_probe)
    assert [fn.__name__ for fn in mcp_server._TOOLS] == before


def test_the_coach_surface_is_fully_accounted_for():
    """Every COACH_TOOLS name is an engine tool, a synthetic tool or declared
    host-supplied. A typo'd or orphaned name fails here, not silently in prod."""
    engine = {fn.__name__ for fn in mcp_server._TOOLS}
    synthetic = {llm.ANALYST_QUERY_NAME, llm.EVIDENCE_CITE_NAME}
    assert set(llm.HOST_SUPPLIED_TOOLS) <= set(llm.COACH_TOOLS)
    unaccounted = (set(llm.COACH_TOOLS) - engine - synthetic
                   - set(llm.HOST_SUPPLIED_TOOLS))
    assert unaccounted == set()
    assert not set(llm.HOST_SUPPLIED_TOOLS) & engine


def test_the_full_coach_surface_resolves_on_a_bare_engine(vault):
    names = _offered(vault, llm.COACH_TOOLS)
    assert names == (set(llm.COACH_TOOLS) - set(llm.HOST_SUPPLIED_TOOLS)
                     - {llm.EVIDENCE_CITE_NAME})
