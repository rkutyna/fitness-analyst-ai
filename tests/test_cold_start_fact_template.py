from __future__ import annotations

import json

from health_advisor import chat, cold_start, demo, deepdive_mcp, fact_template, llm, mcp_server
from health_advisor.context import VaultContext


SURFACES = ("readiness", "training_load")
COLD_FIELDS = ("status_text", "starts_on_day", "day_now", "starts_on_date", "status")


def _demo_briefing(tmp_path, days: int):
    db_path = tmp_path / f"demo-{days}.db"
    end_date = "2026-09-09"
    demo.build_demo_vault(db_path, days=days, end_date=end_date,
                          read_only=False)
    ctx = VaultContext.local(db_path, user_id=f"demo-{days}")
    ledger_path = tmp_path / f"calls-{days}.jsonl"
    ledger = deepdive_mcp._CallLedger(ledger_path)
    tools = mcp_server.build_tools(ctx)
    deepdive_mcp._ledger_wrapper(
        "get_briefing", tools["get_briefing"], ledger
    )(scope="deep", day=end_date)
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    return ctx, end_date, records


def _path(surface: str, field: str) -> str:
    return f"$.result.{surface}.cold_start.{field}"


def test_refusing_demo_surfaces_publish_all_cold_start_leaves(tmp_path):
    ctx, end_date, records = _demo_briefing(tmp_path, days=9)
    facts = fact_template.build_fact_set(records)
    rendered = json.loads(fact_template.render_fact_set(facts))
    with ctx.read_only() as conn:
        measured = cold_start.describe(conn, end_date,
                                       surfaces=set(SURFACES))

    for surface in SURFACES:
        block = measured[surface]
        assert block["status"] in {
            "establishing_baseline", "insufficient_history", "insufficient_data",
            "partial", "nothing_moved",
        }
        for field in COLD_FIELDS:
            key = _path(surface, field)
            assert key in facts
            assert key in rendered
            assert facts[key]["value"] == block[field]
            assert facts[key]["display"] == str(block[field])
            assert facts[key]["source"]["path"] == key

    readiness = measured["readiness"]
    assert facts[_path("readiness", "starts_on_day")]["value"] == readiness["starts_on_day"]
    assert facts[_path("readiness", "day_now")]["value"] == readiness["day_now"]
    template = (
        f"{{{_path('readiness', 'status_text')}}} "
        f"(start day {{{_path('readiness', 'starts_on_day')}}}; "
        f"current day {{{_path('readiness', 'day_now')}}})"
    )
    assert fact_template.scan_template(template, facts)["ok"]
    assert fact_template.interpolate_template(template, facts) == (
        f"{readiness['status_text']} "
        f"(start day {readiness['starts_on_day']}; "
        f"current day {readiness['day_now']})"
    )


def test_cold_start_publication_completeness_catches_a_dropped_leaf(tmp_path):
    _, _, records = _demo_briefing(tmp_path, days=9)
    published = fact_template.build_fact_set(records)
    dropped = _path("readiness", "status_text")
    del published[dropped]

    assert dropped in fact_template.publish_completeness(records, published)


def test_refusal_guidance_names_the_status_text_placeholder(tmp_path):
    _, _, records = _demo_briefing(tmp_path, days=9)
    facts = fact_template.build_fact_set(records)

    guidance = fact_template.cold_start_guidance(facts)

    assert _path("readiness", "status_text") in guidance
    assert "sentence" in guidance.lower()


def test_readiness_question_prompt_guides_model_to_status_text_leaf(
        tmp_path, monkeypatch):
    ctx, _, records = _demo_briefing(tmp_path, days=9)
    status_text = _path("readiness", "status_text")
    responses = iter(["acknowledged", f"Readiness: {{{status_text}}}."])
    prompts = []
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(chat, "_read_ledger", lambda path: records)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        llm, "tool_loop",
        lambda prompt, **kwargs: prompts.append(prompt) or next(responses),
    )

    result = chat.answer_question(
        ctx, "How ready am I to train today?", as_of="2026-09-09")

    assert result["mode"] == "narration"
    assert status_text in prompts[1]
    assert "use {" + status_text + "} as the sentence" in prompts[1]


def test_healthy_readiness_publishes_no_readiness_cold_start_leaves(tmp_path):
    ctx, end_date, records = _demo_briefing(tmp_path, days=60)
    facts = fact_template.build_fact_set(records)
    with ctx.read_only() as conn:
        measured = cold_start.describe(conn, end_date, surfaces={"readiness"})
    assert measured["readiness"]["status"] == "ok"

    readiness_paths = {
        _path("readiness", field) for field in COLD_FIELDS
    }
    assert readiness_paths.isdisjoint(facts)

    # Keep the test tied to the measured surface state rather than a demo-day
    # constant: this is the absence contract when readiness is not refusing.
    assert not any(
        fact.get("path") in readiness_paths for fact in facts.values()
    )
