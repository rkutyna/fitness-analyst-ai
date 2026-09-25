"""The figure-to-call join on narration answers (health_advisor#116 DW2).

A narration answer states figures that Python interpolated from a closed fact
set, and every fact records the ledger sequence that produced it. The answer
publishes that join as ``figures`` so a client can show which tool call each
stated figure came from. These tests pin the invariant that makes it
trustworthy: every figure's sequence resolves to a record in ``tool_trace``.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from health_advisor import chat, fact_template, llm, receiver
from tests.conftest import seed_metric

PERIOD = "2026-08-10:2026-08-16"


@pytest.fixture(autouse=True)
def _seed_model_path_vault(conn):
    seed_metric(conn, "fixture_metric", "2026-01-01", [1])


def _ledger() -> list[dict]:
    return [
        {
            "sequence": 1, "tool_name": "summarize_metric",
            "arguments": {"metric": "jog_minutes", "period": PERIOD},
            "result": {
                "metric": "jog_minutes", "unit": "min", "period": PERIOD,
                "mean": 50.1,
                "presentation": {"metric": "jog_minutes", "period": PERIOD,
                                  "field": "presentation", "value": "50 m"},
            },
        },
        {
            "sequence": 2, "tool_name": "analyst_query",
            "arguments": {"question": "synthetic"},
            "result": {"tables": [{
                "name": "synthetic_table", "columns": ["item", "value"],
                "units": ["text", "count"], "rows": [["synthetic-a", 11]],
                "row_count": 1,
            }]},
        },
    ]


MEAN_KEY = fact_template.fact_key("jog_minutes", PERIOD, "mean")
LABEL_KEY = fact_template.fact_key("jog_minutes", PERIOD, "period_label")
CELL_KEY = fact_template.attachment_fact_key(
    "synthetic_table", "value", "synthetic-a")
TEMPLATE = ("For {" + LABEL_KEY + "} you averaged {" + MEAN_KEY + "}, "
            "and the table shows {" + CELL_KEY + "}; again {" + MEAN_KEY + "}.")


def _stub(monkeypatch, responses):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    ledger = _ledger()
    replies = iter(responses)
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(replies))


def _assert_figures_resolve(result: dict) -> None:
    sequences = {record["sequence"] for record in result["tool_trace"]}
    assert result["figures"], "a narration with figures must publish them"
    for figure in result["figures"]:
        assert figure["sequence"] in sequences, figure
        assert figure["display"] in result["text"], figure


def test_narration_publishes_each_stated_figure_once_with_its_call(
        monkeypatch, vault):
    _stub(monkeypatch, ["acknowledged", TEMPLATE])

    result = chat.answer_question(vault, "How was my running?")

    assert result["mode"] == "narration"
    assert result["text"] == ("For the week of August 10 you averaged 50 m, "
                              "and the table shows 11; again 50 m.")
    assert [f["key"] for f in result["figures"]] == [
        LABEL_KEY, MEAN_KEY, CELL_KEY]
    by_key = {f["key"]: f for f in result["figures"]}
    assert by_key[MEAN_KEY] == {
        "key": MEAN_KEY, "display": "50 m", "unit": "min", "sequence": 1,
        "path": "$.result.mean", "metric": "jog_minutes", "period": PERIOD,
        "field": "mean",
    }
    assert by_key[CELL_KEY]["sequence"] == 2
    assert by_key[CELL_KEY]["table"] == "synthetic_table"
    assert by_key[LABEL_KEY]["field"] == "period_label"
    _assert_figures_resolve(result)


def test_retry_narration_publishes_the_retry_templates_figures(
        monkeypatch, vault):
    # Attempt 1 carries a bare digit outside any placeholder and is refused;
    # the retry states only the table cell. `figures` must follow the
    # template that published, not the one that was refused.
    _stub(monkeypatch, ["acknowledged",
                        "You ran 5 times: {" + MEAN_KEY + "}.",
                        "The table shows {" + CELL_KEY + "}."])

    result = chat.answer_question(vault, "How was my running?")

    assert result["mode"] == "narration"
    assert result["verification"]["retry"] is True
    assert [f["key"] for f in result["figures"]] == [CELL_KEY]
    _assert_figures_resolve(result)


def test_fallback_publishes_no_figures(monkeypatch, vault):
    bad = "You ran 5 times: {" + MEAN_KEY + "}."
    _stub(monkeypatch, ["acknowledged", bad, bad])

    result = chat.answer_question(vault, "How was my running?")

    assert result["mode"] == "fallback"
    assert not result.get("figures")


def test_answer_figures_skips_unknown_keys_and_unsequenced_facts():
    facts = {
        "a": {"display": "1", "source": {"sequence": 3, "path": "$.x"}},
        "b": {"display": "2", "source": {}},
        "c": {"display": "3", "source": {"sequence": True}},
    }

    figures = chat._answer_figures(["missing", "b", "a", "c", "a"], facts)

    assert figures == [{"key": "a", "display": "1", "unit": None,
                        "sequence": 3, "path": "$.x"}]


def _post(client):
    return client.post("/v1/ask", json={"question": "How was my running?"},
                       headers={"x-health-secret": "ask-secret"})


def test_ask_response_carries_figures_that_resolve_into_tool_trace(
        monkeypatch, vault):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    _stub(monkeypatch, ["acknowledged", TEMPLATE])

    with TestClient(receiver.create_app(vault)) as client:
        response = _post(client)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "narration"
    assert [f["key"] for f in body["figures"]] == [LABEL_KEY, MEAN_KEY, CELL_KEY]
    _assert_figures_resolve(body)


def test_ask_fallback_response_carries_an_empty_figures_list(
        monkeypatch, vault):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: "")

    with TestClient(receiver.create_app(vault)) as client:
        body = _post(client).json()

    assert body["mode"] == "fallback"
    assert body["figures"] == []


def test_ask_extra_adds_fields_but_never_replaces_engine_fields(
        monkeypatch, vault):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    _stub(monkeypatch, ["acknowledged", TEMPLATE])
    seen = []

    def extra(response):
        seen.append(response)
        return {"sources": [{"n": len(response["figures"])}],
                "text": "replaced", "figures": []}

    with TestClient(receiver.create_app(vault, ask_extra=extra)) as client:
        body = _post(client).json()

    assert body["sources"] == [{"n": 3}]
    assert body["text"].startswith("For the week of August 10")
    assert len(body["figures"]) == 3
    assert seen and seen[0]["tool_trace"] == body["tool_trace"]


def test_a_failing_ask_extra_costs_the_decoration_not_the_answer(
        monkeypatch, vault, capsys):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    _stub(monkeypatch, ["acknowledged", TEMPLATE])

    def extra(response):
        raise RuntimeError("synthetic hook failure")

    with TestClient(receiver.create_app(vault, ask_extra=extra)) as client:
        response = _post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "narration"
    assert "sources" not in body
    assert "ask-extra failed: RuntimeError" in capsys.readouterr().err
