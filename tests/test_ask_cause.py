"""Response-boundary tests for the diagnostic cause taxonomy (#145/#266)."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from health_advisor import chat, llm, receiver


LEDGER_FIXTURE = Path(__file__).parent / "fixtures/jog_ledger_live_20260824_claims.jsonl"


def _ledger() -> list[dict]:
    return [json.loads(LEDGER_FIXTURE.read_text(encoding="utf-8"))]


def _valid_draft() -> llm.ResearchResponse:
    ledger = _ledger()
    period = ledger[0]["result"]["block_comparison"]["blocks"]["recent"]["period"]
    draft = llm.ResearchResponse("Your recent jogging averaged 50.1 minutes.")
    draft.claims = [{
        "metric": "jog_minutes", "period": period, "field": "mean",
        "value": 50.1,
        "source": {"sequence": 1,
                    "path": "$.result.block_comparison.blocks.recent.mean"},
    }]
    return draft


def _post(vault, monkeypatch, loop, *, judge=None):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", loop)
    if judge is not None:
        monkeypatch.setattr(chat, "_ask_judge", judge)
    with TestClient(receiver.create_app(vault)) as client:
        response = client.post(
            "/v1/ask", json={"question": "How did my recent jogging compare?"},
            headers={"x-health-secret": "ask-secret"})
    assert response.status_code == 200, response.text
    return response.json()


def test_ask_cause_ok_and_reason_is_unchanged(monkeypatch, vault):
    ledger_line = LEDGER_FIXTURE.read_text(encoding="utf-8")
    draft = _valid_draft()

    def loop(*args, **kwargs):
        Path(kwargs["ledger_path"]).write_text(ledger_line, encoding="utf-8")
        return draft

    body = _post(vault, monkeypatch, loop, judge=lambda *a, **k: 95)

    assert body["verification"]["cause"] == "ok"
    assert body["verification"]["reason"] == ""


def test_ask_cause_transport_failed_keeps_no_ledger_reason(monkeypatch, vault):
    def loop(*args, **kwargs):
        llm._announce("tool_loop_error", "connection reset")
        return ""

    body = _post(vault, monkeypatch, loop)

    assert body["verification"]["cause"] == "transport_failed"
    assert body["verification"]["reason"] == "ask answer has no tool-call ledger"


def test_ask_cause_backend_unavailable_keeps_no_ledger_reason(monkeypatch, vault):
    def loop(*args, **kwargs):
        llm._announce("openrouter_no_api_key", "key is unset")
        return ""

    body = _post(vault, monkeypatch, loop)

    assert body["verification"]["cause"] == "backend_unavailable"
    assert body["verification"]["reason"] == "ask answer has no tool-call ledger"


def test_ask_cause_empty_gather_keeps_no_ledger_reason(monkeypatch, vault):
    body = _post(vault, monkeypatch, lambda *a, **k: "")

    assert body["verification"]["cause"] == "empty_gather"
    assert body["verification"]["reason"] == "ask answer has no tool-call ledger"


def test_ask_cause_gate_refused_keeps_python_gate_reason(monkeypatch, vault):
    ledger_line = LEDGER_FIXTURE.read_text(encoding="utf-8")

    def loop(*args, **kwargs):
        Path(kwargs["ledger_path"]).write_text(ledger_line, encoding="utf-8")
        return llm.ResearchResponse("Your recent jogging averaged 999 minutes.")

    body = _post(vault, monkeypatch, loop)

    assert body["verification"]["cause"] == "gate_refused"
    assert body["verification"]["reason"] == (
        "numbered coach prose has no structured claims")


def test_ask_cause_no_gather_needed_keeps_advice_answer(monkeypatch, vault):
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    responses = iter(["", "Circuit: {advice:3 rounds} of squats."])
    body = _post(vault, monkeypatch, lambda *a, **k: next(responses))

    assert body["verification"]["cause"] == "no_gather_needed"
    assert body["verification"]["reason"] == "ask answer has no tool-call ledger"


def test_ask_cause_judge_refused_keeps_python_gate_reason(monkeypatch, vault):
    ledger_line = LEDGER_FIXTURE.read_text(encoding="utf-8")
    draft = _valid_draft()

    def loop(*args, **kwargs):
        Path(kwargs["ledger_path"]).write_text(ledger_line, encoding="utf-8")
        return draft

    body = _post(vault, monkeypatch, loop, judge=lambda *a, **k: 55)

    assert body["verification"]["cause"] == "judge_refused"
    assert body["verification"]["reason"] == ""


def test_a_kept_answer_with_no_judge_is_ok_not_judge_refused():
    """The fact-template arm never runs the judge; its kept answers are `ok`.

    A narrated template answer with figures 1/1 once carried
    cause=judge_refused because the arm passed no score
    and the default of 0 read as a failing judge.
    """
    from health_advisor import chat
    kept = {"ok": True}
    ledger = [{"tool": "get_impact_volume", "result": {"x": 1}}]
    assert chat._ask_cause(kept, ledger=ledger, loop_outcomes=[]) == "ok"
    assert chat._ask_cause(kept, ledger=ledger, loop_outcomes=[],
                           judge_score=None) == "ok"
    assert chat._ask_cause(kept, ledger=ledger, loop_outcomes=[],
                           judge_score=10) == "judge_refused"
    assert chat._ask_cause(kept, ledger=ledger, loop_outcomes=[],
                           judge_score=70) == "ok"
