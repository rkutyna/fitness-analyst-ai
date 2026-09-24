"""The loop side channel is attributed per thread (health_advisor#487).

`chat` attributes a loop outcome to one call by comparing
`llm.last_loop_status()["call_id"]` before and after it. With a process-global
status, a concurrent ask on another thread that announced `tool_loop_truncated`
between those two reads was charged to an answer that finished cleanly: the
clean answer was marked "answer truncated", paid for a repair call, and could
end in a fallback. These tests drive the real `chat.answer_question`
fact-template path with only the model transport (`llm.tool_loop`) stubbed.
"""
from __future__ import annotations

import threading

import pytest

from health_advisor import chat, fact_template, llm
from tests.conftest import seed_metric


WAIT = 10.0


@pytest.fixture(autouse=True)
def _seed_and_reset(conn, monkeypatch):
    seed_metric(conn, "fixture_metric", "2026-01-01", [1])
    monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", "1")
    monkeypatch.setattr(llm, "tool_schemas", lambda *args, **kwargs: [])
    llm._reset_loop_status()
    yield
    llm._reset_loop_status()


def _ledger():
    return [{
        "sequence": 1,
        "tool_name": "analyst_query",
        "arguments": {},
        "result": {"tables": [{
            "name": "resting_rate",
            "columns": ["day", "rate"],
            "units": ["date", "count/min"],
            "rows": [["2026-08-01", 63], ["2026-08-02", 60]],
            "row_count": 2,
        }]},
    }]


def _template():
    first_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-01")
    last_key = fact_template.attachment_fact_key(
        "resting_rate", "rate", "2026-08-02")
    return f"Your resting heart rate went from {{{first_key}}} to {{{last_key}}}."


def _ask(vault, monkeypatch, narration_call):
    """Run one real ask; ``narration_call`` stands in for the answer turn."""
    ledger = _ledger()
    calls = []

    def tool_loop(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return "acknowledged"
        return narration_call()

    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_loop", tool_loop)
    capture = []
    result = chat.answer_question(vault, "How has my resting heart rate changed?",
                                  capture=capture)
    return result, calls, capture


def test_a_concurrent_asks_truncation_is_not_charged_to_this_answer(
        monkeypatch, vault):
    """A neighbour thread announces tool_loop_truncated INSIDE this ask's
    answer call (between its before/after status reads). This answer finished
    cleanly, so it must publish on the first attempt with no repair."""
    in_answer_call = threading.Event()
    neighbour_announced = threading.Event()
    neighbour_saw = {}

    def neighbour_loop():
        # Another ask's answer turn, on its own worker thread, ending in a
        # length finish — timed to land inside ours.
        if not in_answer_call.wait(WAIT):
            return
        llm._announce("tool_loop_truncated", "neighbour length finish")
        neighbour_saw.update(llm.last_loop_status())
        neighbour_announced.set()

    def narration_call():
        in_answer_call.set()
        assert neighbour_announced.wait(WAIT), "neighbour never announced"
        return _template()

    events_before = llm.loop_event_count()
    neighbour = threading.Thread(target=neighbour_loop, name="neighbour-ask")
    neighbour.start()
    try:
        result, calls, capture = _ask(vault, monkeypatch, narration_call)
    finally:
        in_answer_call.set()
        neighbour.join(WAIT)
    assert not neighbour.is_alive()

    # The neighbour's event really happened, on its own thread, inside our
    # window: the process-wide count moved and the neighbour sees its event.
    assert neighbour_announced.is_set()
    assert neighbour_saw["outcome"] == "tool_loop_truncated"
    assert llm.loop_event_count() == events_before + 1

    # ...and it was not charged to this answer.
    assert result["verification"]["reason"] != "answer truncated"
    assert result["mode"] == "narration", result["verification"]
    assert "63" in result["text"] and "60" in result["text"]
    assert len(calls) == 2, "a clean answer must not pay for a repair call"
    assert len(capture) == 1
    assert llm.last_loop_status()["outcome"] == "not_called"


def test_an_earlier_failure_on_this_thread_is_not_a_later_answers_cause(
        monkeypatch, vault):
    """Serial guarantee of `_ask_loop_outcome`: the status is not cleared by a
    successful loop, and an event from BEFORE the call is not its cause."""
    llm._announce("tool_loop_truncated", "an earlier call's length finish")
    assert llm.last_loop_status()["outcome"] == "tool_loop_truncated"

    result, calls, _capture = _ask(vault, monkeypatch, _template)

    assert result["verification"]["reason"] != "answer truncated"
    assert result["mode"] == "narration", result["verification"]
    assert len(calls) == 2


def test_this_threads_own_truncation_is_still_attributed(monkeypatch, vault):
    """Positive control: the same announce, made by THIS ask's own answer call,
    is still charged to it — so the negative tests above are not passing
    because truncation detection is dead."""
    def narration_call():
        llm._announce("tool_loop_truncated", "our own length finish")
        return _template()

    result, calls, capture = _ask(vault, monkeypatch, narration_call)

    assert capture[0]["verification"]["reason"] == "answer truncated"
    assert len(calls) == 3, "our own truncation must trigger the repair call"


def test_each_thread_reads_its_own_status_and_ids_stay_unique():
    workers, per_worker = 8, 200
    start = threading.Barrier(workers)
    seen: dict[int, list[dict]] = {}

    def worker(index):
        start.wait(WAIT)
        mine = []
        for n in range(per_worker):
            llm._announce(f"event_{index}", str(n))
            mine.append(llm.last_loop_status())
        seen[index] = mine

    before = llm.loop_event_count()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)

    ids = [status["call_id"] for rows in seen.values() for status in rows]
    assert llm.loop_event_count() - before == workers * per_worker
    assert len(set(ids)) == workers * per_worker
    for index, rows in seen.items():
        assert all(status["outcome"] == f"event_{index}" for status in rows)
        assert [status["detail"] for status in rows] == [
            str(n) for n in range(per_worker)]
        own_ids = [status["call_id"] for status in rows]
        assert own_ids == sorted(own_ids)
    # The main thread announced nothing, so it still reads `not_called`.
    assert llm.last_loop_status()["outcome"] == "not_called"
