from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from health_advisor import chat, llm, receiver
from health_advisor import agents, analysis
from health_advisor import deepdive_verify as DV


LEDGER_FIXTURE = Path(__file__).parent / "fixtures/jog_ledger_live_20260824_claims.jsonl"


def _ledger() -> list[dict]:
    return [json.loads(LEDGER_FIXTURE.read_text())]


def _claim(metric: str, period, field: str, value, path: str) -> dict:
    return {"metric": metric, "period": period, "field": field,
            "value": value, "source": {"sequence": 1, "path": path}}


def test_fabrication_battery_is_rejected_by_the_scoped_ledger_gate():
    ledger = _ledger()
    blocks = ledger[0]["result"]["block_comparison"]["blocks"]
    recent = blocks["recent"]["period"]
    cases = [
        ("Your resting heart rate is 201 bpm.", _claim(
            "resting_heart_rate", "2026-06-29", "jog_minutes", 201,
            "$.result.periods[0].jog_minutes")),
        ("You slept 45.3 hours.", _claim(
            "sleep_hours", "2026-06-29", "jog_minutes", 45.3,
            "$.result.periods[0].jog_minutes")),
        ("Your VO2max is 68.3 ml/kg/min.", _claim(
            "vo2_max", "2026-08-17", "value", 68.3,
            "$.result.block_comparison.blocks.recent.weeks[3].value")),
        ("You weigh 46.7 kg and your body fat is 32.7%.", [
            _claim("body_mass", "2026-08-17", "value", 46.7,
                   "$.result.block_comparison.blocks.recent.weeks[3].value"),
            _claim("body_fat", "2026-08-03", "value", 32.7,
                   "$.result.block_comparison.blocks.recent.weeks[1].value"),
        ]),
        ("Jogging increased 100%.", _claim(
            "jog_minutes", recent, "mean_delta", 100,
            "$.result.block_comparison.change.mean_delta")),
        ("You ran 12 km yesterday.", _claim(
            "running_distance", "2026-08-17", "jog_miles", 12,
            "$.result.periods[7].jog_miles")),
        ("Your sleep debt is 7 hours.", _claim(
            "sleep_debt", "2026-08-17", "days_covered", 7,
            "$.result.periods[7].days_covered")),
    ]

    verdicts = []
    for prose, claims in cases:
        verdicts.append(chat._verify_ask_answer(
            None, prose, claims if isinstance(claims, list) else [claims], ledger))

    assert len(verdicts) == 7
    assert [verdict["ok"] for verdict in verdicts] == [False] * 7


def test_a_correct_answer_passes_and_reports_its_verified_figure():
    ledger = _ledger()
    period = ledger[0]["result"]["block_comparison"]["blocks"]["recent"]["period"]
    claim = _claim(
        "jog_minutes", period, "mean", 50.1,
        "$.result.block_comparison.blocks.recent.mean")

    verdict = chat._verify_ask_answer(
        None, "Your recent jogging averaged 50.1 minutes.", [claim], ledger)

    assert verdict["ok"] is True
    assert verdict["figures_verified"] == 1
    assert verdict["figures_total"] == 1
    assert verdict["unsupported"] == []


def test_answer_reports_path_and_metric_binding_tiers_separately():
    ledger = [{
        "sequence": 1, "tool_name": "synthetic", "arguments": {},
        "result": {
            "path_only": 12.0,
            "metric": "step_count", "period": "2026-08-21",
            "field": "mean", "value": 42.0,
        },
        "result_elided": False,
    }]
    claims = [
        {"metric": None, "period": None, "field": "path_only", "value": 12,
         "source": {"sequence": 1, "path": "$.result.path_only"}},
        {"metric": "step_count", "period": "2026-08-21", "field": "mean",
         "value": 42, "source": {"sequence": 1, "path": "$.result.value"}},
    ]

    verdict = chat._verify_ask_answer(
        None, "The values are 12 and 42.", claims, ledger)

    assert verdict["ok"] is True
    assert verdict["tier_counts"] == {"path": 1, "metric": 1}
    assert verdict["tier1_path_bound"] == 1
    assert verdict["tier2_metric_recomputed"] == 1
    assert [n["tier"] for n in verdict["verdict"]["numbers"]] == [
        "path", "metric"]


def test_empty_tool_trace_is_rejected_before_the_shared_gate(monkeypatch):
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("the resolver gate must not run for zero calls")

    monkeypatch.setattr(DV, "verify_coach_claims", should_not_run)
    verdict = chat._verify_ask_answer(None, "No number.", [], [])

    assert verdict["ok"] is False
    assert verdict["reason"] == "ask answer has no tool-call ledger"
    assert called is False


def test_empty_model_response_is_a_visible_fallback_not_a_500(monkeypatch, vault, conn):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: "")

    with TestClient(receiver.create_app(vault)) as client:
        response = client.post(
            "/v1/ask", json={"question": "How am I doing?"},
            headers={"x-health-secret": "ask-secret"})

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["mode"] == "fallback"
    assert result["text"].startswith("Fallback:")
    assert "How am I doing?" not in result["text"]


def test_ask_reports_uncovered_as_of_per_metric_without_new_numbers(
        monkeypatch, vault, conn):
    """Freshness is Python truth, exposed without changing the claim pool."""
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: "")

    # Seed one vital on the requested day and the other eight only before it.
    for metric in analysis.VITALS:
        conn.execute(
            "INSERT INTO daily_metrics "
            "(metric, date, count, sum, avg, min, max, last, unit) "
            "VALUES (?, ?, 1, 1, 1, 1, 1, 1, ?)",
            (metric, "2026-08-21" if metric == analysis.VITALS[0]
             else "2026-08-20", "count"),
        )
    conn.commit()

    with TestClient(receiver.create_app(vault)) as client:
        response = client.post(
            "/v1/ask",
            json={"question": "How did I sleep this week?", "as_of": "2026-08-21"},
            headers={"x-health-secret": "ask-secret"})

    assert response.status_code == 200, response.text
    body = response.json()
    freshness = body["freshness"]
    assert freshness == {
        "as_of": "2026-08-21",
        "metrics": [
            {
                "metric": row["metric"],
                "status": row["status"],
                "last_date": row["last_date"],
                "covers_as_of": row["covers_as_of"],
                "behind": row["behind"],
            }
            for row in analysis.coverage(conn, as_of="2026-08-21")
        ],
    }
    assert sum(row["covers_as_of"] for row in freshness["metrics"]) == 1
    assert sum(not row["covers_as_of"] for row in freshness["metrics"]) == 8
    assert all(
        isinstance(value, (str, bool)) or value is None
        for row in freshness["metrics"]
        for value in row.values()
    )
    assert len(agents._numbers_in(body)) == len(
        agents._numbers_in({key: value for key, value in body.items()
                            if key != "freshness"}))


def test_ask_requires_a_non_empty_secret_and_ingest_keeps_optional_behavior(
        monkeypatch, vault):
    request = object()
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    with pytest.raises(Exception) as exc:
        receiver._require_ask_secret(None)
    assert getattr(exc.value, "status_code", None) == 401

    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    with pytest.raises(Exception) as exc:
        receiver._require_ask_secret("wrong")
    assert getattr(exc.value, "status_code", None) == 401

    # The ingest helper intentionally retains its historical optional-secret
    # behavior; an empty configured secret does not reject a valid empty batch.
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    payload = {
        "protocol_version": 1,
        "device": {"id": "d", "name": "n", "model": "m"},
        "app_version": "1", "batch_id": "ask-auth-control",
        "batch_sequence": 1, "sent_at": "2026-08-22T00:00:00Z",
        "anchors": [], "samples": [], "deletions": [], "workouts": [],
    }
    import json as json_module
    response = receiver._healthkit_ingest(
        vault, request, json_module.dumps(payload).encode(), None)
    assert response.status_code == 200


def test_endpoint_records_user_and_assistant_turn_and_provenance(monkeypatch, vault):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    fixture_line = LEDGER_FIXTURE.read_text()

    def fake_loop(*args, **kwargs):
        Path(kwargs["ledger_path"]).write_text(fixture_line, encoding="utf-8")
        period = _ledger()[0]["result"]["block_comparison"]["blocks"]["recent"]["period"]
        claim = _claim("jog_minutes", period, "mean", 50.1,
                       "$.result.block_comparison.blocks.recent.mean")
        return json.dumps({"text": "Your recent jogging averaged 50.1 minutes.",
                           "claims": [claim]})

    monkeypatch.setattr(llm, "tool_loop", fake_loop)
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs:
                        '{"score": 95}')

    with TestClient(receiver.create_app(vault)) as client:
        response = client.post(
            "/v1/ask", json={"question": "How did my recent jogging compare?"},
            headers={"x-health-secret": "ask-secret"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "narration"
    assert body["provenance"]["tool_calls"] == 1
    assert body["verification"]["figures_verified"] == 1
    turns = chat.list_turns(vault, body["conversation_id"])
    assert [turn["role"] for turn in turns] == ["user", "assistant"]
    assert turns[1]["content"] == body["text"]


def test_followup_prompt_contains_prior_turns_once_and_reads_before_append(
        monkeypatch, vault):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    prompts = []
    events = []

    def fake_loop(prompt, **kwargs):
        prompts.append(prompt)
        Path(kwargs["ledger_path"]).write_text(
            json.dumps({"sequence": 1, "tool_name": "get_latest",
                        "arguments": {}, "result": {"status": "ok"}}) + "\n",
            encoding="utf-8")
        return "I checked the data."

    original_append = chat.append_turn
    original_begin = chat.append_question_and_history

    def recorded_begin(*args, **kwargs):
        events.append(("begin", args[1]))
        return original_begin(*args, **kwargs)

    def recorded_append(*args, **kwargs):
        events.append(("append", args[2], args[3]))
        return original_append(*args, **kwargs)

    monkeypatch.setattr(chat, "append_question_and_history", recorded_begin)
    monkeypatch.setattr(chat, "append_turn", recorded_append)
    monkeypatch.setattr(llm, "tool_loop", fake_loop)
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: '{"score": 95}')

    with TestClient(receiver.create_app(vault)) as client:
        first = client.post(
            "/v1/ask", json={"question": "TURN_ONE_QUESTION"},
            headers={"x-health-secret": "ask-secret"})
        second = client.post(
            "/v1/ask", json={"conversation_id": first.json()["conversation_id"],
                             "question": "TURN_TWO_QUESTION"},
            headers={"x-health-secret": "ask-secret"})

    assert first.status_code == second.status_code == 200
    second_prompt = next(prompt for prompt in prompts
                         if "TURN_ONE_QUESTION" in prompt
                         and "I checked the data." in prompt)
    assert "I checked the data." in second_prompt
    assert second_prompt.count("TURN_TWO_QUESTION") == 1
    begin_indices = [i for i, event in enumerate(events)
                     if event == ("begin", second.json()["conversation_id"])]
    assert len(begin_indices) == 2
    assert events[begin_indices[1] + 1][:2] == ("append", "assistant")


def test_history_cannot_launder_a_figure_from_an_earlier_turn(
        monkeypatch, vault, conn, tmp_path):
    """An unrelated non-empty turn-2 ledger cannot authorize turn-1's figure."""
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    unrelated = {
        "sequence": 1,
        "tool_name": "get_sleep_summary",
        "arguments": {"period": "2026-08-17"},
        "result": {"sleep_minutes": 480},
    }
    claim = _claim("jog_minutes", "2026-08-17", "sleep_minutes", 50.1,
                   "$.result.sleep_minutes")

    def unrelated_loop(prompt, **kwargs):
        Path(kwargs["ledger_path"]).write_text(
            json.dumps(unrelated) + "\n", encoding="utf-8")
        return json.dumps({
            "text": "Your jogging was 50.1 minutes.",
            "claims": [claim],
        })

    monkeypatch.setattr(llm, "tool_loop", unrelated_loop)
    result = chat.answer_question(
        vault,
        "What about that?",
        history=[{"role": "assistant",
                  "content": "Your jogging was 50.1 minutes."}],
        ledger_path=str(tmp_path / "turn-2.jsonl"),
    )

    assert result["mode"] == "fallback"
    assert result["verification"]["ok"] is False
    assert result["verification"]["tool_calls"] == 1


def test_history_does_not_change_resolvable_scope_pool_width(
        monkeypatch, vault, conn, tmp_path):
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    ledger_record = {
        "sequence": 1,
        "tool_name": "get_latest",
        "arguments": {},
        "result": {"metric": "jog_minutes", "period": "2026-08-17",
                    "field": "value", "value": 50.1},
    }
    captured_ledgers = []

    def fake_loop(prompt, **kwargs):
        Path(kwargs["ledger_path"]).write_text(
            json.dumps(ledger_record) + "\n", encoding="utf-8")
        return "No numbered claims."

    def capture_verifier(conn, prose, claims, ledger, as_of=None):
        captured_ledgers.append(ledger)
        return {"ok": False, "grounded": False, "unsupported": [],
                "reason": "test capture", "tool_calls": len(ledger)}

    monkeypatch.setattr(llm, "tool_loop", fake_loop)
    monkeypatch.setattr(chat, "_verify_ask_answer", capture_verifier)

    chat.answer_question(vault, "follow up", history=None,
                         ledger_path=str(tmp_path / "without.jsonl"))
    without_history = len(DV._ledger_scopes(captured_ledgers[-1][0]))
    chat.answer_question(
        vault, "follow up",
        history=[{"role": "assistant", "content": "50.1 is reference only"}],
        ledger_path=str(tmp_path / "with.jsonl"),
    )
    with_history = len(DV._ledger_scopes(captured_ledgers[-1][0]))

    assert without_history == with_history


def test_retry_is_absent_on_a_first_pass_success_and_true_after_a_retry(
        monkeypatch, vault, conn):
    """`retry` is absent/True, never false — pin it so nobody reads it as a flag.

    The first-pass success branch does not set `retry` at all; only the two
    post-retry branches do, and they set it True. So a client decoding it as an
    optional boolean gets nil on the happy path, and nil means "no retry
    happened" rather than "unknown". Nothing asserted this, which is how the
    ambiguity survived long enough for two readers to disagree about it.
    """
    ledger = _ledger()
    period = ledger[0]["result"]["block_comparison"]["blocks"]["recent"]["period"]
    good = _claim("jog_minutes", period, "mean", 50.1,
                  "$.result.block_comparison.blocks.recent.mean")

    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(chat, "_ask_judge", lambda *args, **kwargs: 100)
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])

    first = llm.ResearchResponse("Your recent jogging averaged 50.1 minutes.")
    first.claims = [good]
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: first)
    passed = chat.answer_question(vault, "How is my jogging?")

    assert passed["mode"] == "narration"
    assert "retry" not in passed["verification"], \
        "a first-pass success must not claim a retry happened"

    # Now force the first draft to fail so the retry branch is taken.
    drafts = iter([llm.ResearchResponse(""), first])
    monkeypatch.setattr(llm, "tool_loop", lambda *a, **k: next(drafts))
    retried = chat.answer_question(vault, "How is my jogging?")

    assert retried["verification"]["retry"] is True


def test_the_response_carries_no_fabricated_completion_fields(
        monkeypatch, vault, conn):
    """ok/status/cancelled were literals and are gone; mode carries the truth."""
    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: "")

    with TestClient(receiver.create_app(vault)) as client:
        response = client.post(
            "/v1/ask", json={"question": "How am I doing?"},
            headers={"x-health-secret": "ask-secret"})

    body = response.json()
    for fabricated in ("ok", "status", "cancelled"):
        assert fabricated not in body, (
            f"{fabricated!r} was a literal that could never report failure")
    # The fields that ARE computed, and that a client should bind to instead.
    assert body["mode"] == "fallback"
    assert body["conversation_id"]
    assert "verification" in body and "provenance" in body
