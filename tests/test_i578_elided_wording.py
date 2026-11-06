"""A turn whose only figure-bearing result was elided answers in Python's words
(downstream issue #578, Done-when 4).

The call ledger elides any result over ``deepdive_mcp.LEDGER_RESULT_MAX_BYTES``
(256 KiB), and an elided result publishes no fact. A ten-year by-week impact
volume is that case on a real vault. When elision leaves a turn with no figure
at all, ``answer_question`` returns a fixed Python-authored sentence and does
not ask the narration model; when another call published a figure, the turn
narrates as before.

The limit is lowered for the test so a 60-day demo vault's daily series
crosses it; the mechanism under test is the same one a 256 KiB result trips.
"""
from __future__ import annotations

import json
import re
import shutil

import pytest

from fastapi.testclient import TestClient

from health_advisor import chat, deepdive_mcp, demo, llm, receiver
from health_advisor.context import VaultContext

QUESTION = "How has my resting heart rate looked over the last 90 days?"
# The answer, hand-written here rather than read from chat.py.
WANT = ("What I looked up for this question was too large to check figure by "
        "figure, so I can't quote a number from it. Ask about a shorter "
        "period, or group it by week or month, and I can give you the figures.")
SERIES = {"name": "get_daily_series",
          "arguments": {"metric": "resting_heart_rate"}}
LATEST = {"name": "get_latest", "arguments": {"metric": "resting_heart_rate"}}
LIMIT = 4096          # bytes: the daily series is over it, get_latest under it


@pytest.fixture(scope="module")
def built_vault(tmp_path_factory):
    path = tmp_path_factory.mktemp("i578_elided") / "demo.db"
    demo.build_demo_vault(path, days=60, read_only=False)
    return path


@pytest.fixture
def ctx(built_vault, tmp_path):
    copy = tmp_path / "vault.db"
    shutil.copy(built_vault, copy)
    return VaultContext.local(copy, user_id="test").provider_facing()


def _ask(ctx, tmp_path, monkeypatch, calls, limit=LIMIT):
    """One real ask with a fake model whose gather turn makes ``calls``.

    Returns ``(result, ledger records, number of model posts)``. The gather
    loop takes two posts (the tool calls, then the model's closing reply), so
    a narration call shows up as a third. Every post after the first answers
    with a one-figure sentence."""
    posts = []

    def fake_post(payload, timeout):
        posts.append(payload)
        if len(posts) == 1:
            return {"message": {"role": "assistant", "content": "",
                                "tool_calls": [{"function": c} for c in calls]}}
        return {"message": {"role": "assistant",
                            "content": "Your resting heart rate is 58 bpm."}}

    monkeypatch.setattr(deepdive_mcp, "LEDGER_RESULT_MAX_BYTES", limit)
    monkeypatch.setattr(llm, "_ollama_chat_post", fake_post)
    ledger_path = tmp_path / "ledger.jsonl"
    result = chat.answer_question(ctx, QUESTION, ledger_path=str(ledger_path))
    with open(ledger_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    return result, records, len(posts)


def test_only_an_elided_result_gets_the_python_sentence(ctx, tmp_path,
                                                        monkeypatch):
    result, records, posts = _ask(ctx, tmp_path, monkeypatch, [SERIES])
    assert [(r["tool_name"], r["result_elided"]) for r in records] == \
        [("get_daily_series", True)]
    assert result["text"] == WANT
    # "status", not "fallback" (Roger, 2026-11-06): a kept Python answer.
    assert result["mode"] == "status"
    assert result["verification"]["cause"] == "result_elided"
    assert result["verification"]["ok"] is False
    assert posts == 2                      # the narration model was not asked
    assert not re.search(r"\d", result["text"])


def test_a_figure_from_another_call_still_narrates(ctx, tmp_path, monkeypatch):
    result, records, posts = _ask(ctx, tmp_path, monkeypatch, [SERIES, LATEST])
    assert sorted((r["tool_name"], r["result_elided"]) for r in records) == \
        [("get_daily_series", True), ("get_latest", False)]
    assert posts >= 3                      # the narration model was asked
    assert result["text"] != WANT
    assert result["verification"].get("cause") != "result_elided"


def test_nothing_elided_is_untouched(ctx, tmp_path, monkeypatch):
    result, records, posts = _ask(ctx, tmp_path, monkeypatch, [SERIES],
                                  limit=256 * 1024)
    assert [(r["tool_name"], r["result_elided"]) for r in records] == \
        [("get_daily_series", False)]
    assert posts >= 3
    assert result["text"] != WANT
    assert result["verification"].get("cause") != "result_elided"


def test_the_client_receives_and_stores_status(built_vault, tmp_path,
                                               monkeypatch):
    """The HTTP body and the stored turn both say "status" (the phone keys
    on the body; history and recovery key on the stored turn)."""
    copy = tmp_path / "vault.db"
    shutil.copy(built_vault, copy)
    vault = VaultContext.local(copy, user_id="test", writable=True)
    posts = []

    def fake_post(payload, timeout):
        posts.append(payload)
        if len(posts) == 1:
            return {"message": {"role": "assistant", "content": "",
                                "tool_calls": [{"function": SERIES}]}}
        return {"message": {"role": "assistant",
                            "content": "Your resting heart rate is 58 bpm."}}

    monkeypatch.setattr(receiver, "SHARED_SECRET", "ask-secret")
    monkeypatch.setattr(deepdive_mcp, "LEDGER_RESULT_MAX_BYTES", LIMIT)
    monkeypatch.setattr(llm, "_ollama_chat_post", fake_post)
    headers = {"x-health-secret": "ask-secret"}
    with TestClient(receiver.create_app(vault)) as client:
        body = client.post("/v1/ask", json={"question": QUESTION},
                           headers=headers).json()
        turns = client.get("/v1/conversation", headers=headers).json()["turns"]

    assert body["text"] == WANT
    assert body["mode"] == "status"
    assert body["verification"]["cause"] == "result_elided"
    assert [(t["answer"], t["mode"]) for t in turns
            if t["question"] == QUESTION] == [(WANT, "status")]


def test_an_audit_replaces_the_elided_sentence(vault, monkeypatch):
    """The elided sentence talks about a chat question; an audit narration
    that hits it gets the audit's own fallback text, as a fallback does."""
    monkeypatch.setattr(chat, "_audit_flag_ledger", lambda report: [])
    monkeypatch.setattr(chat, "_audit_fallback",
                        lambda report: "the audit's own words")
    monkeypatch.setattr(
        chat, "_answer_fact_template",
        lambda *a, **k: {"text": WANT, "mode": "status", "tool_trace": [],
                         "verification": {"cause": "result_elided"}})
    result = chat._narrate_audit(vault, {})
    assert result["text"] == "the audit's own words"
    assert result["mode"] == "status"
