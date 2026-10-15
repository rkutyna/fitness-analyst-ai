# tests/test_complete_status_threads.py
"""The single-shot completion status belongs to the thread that made the call.

Callers attribute an outcome to their own call by reading
``last_complete_status()`` right after ``complete()`` returns. With one
process-global dict, a call finishing on another thread in between replaced the
status, so a timeout was reported as a neighbour's success. The status is now
per thread; ``call_id`` still comes from one process-wide counter.
"""
import threading

import httpx
import pytest

from health_advisor import llm


@pytest.fixture
def openrouter(monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")

    def set_handler(handler):
        monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))

    return set_handler


def _run(target):
    thread = threading.Thread(target=target)
    thread.start()
    return thread


def test_each_thread_reads_its_own_completion_status(openrouter):
    """Thread A times out and then waits while thread B completes a call
    successfully; A must still read its own timeout, B its own success."""
    a_finished_call = threading.Event()
    b_finished_call = threading.Event()
    seen = {}

    def handler(request):
        if b"prompt-a" in request.content:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "answer"}}],
            "provider": "CoreWeave"})

    openrouter(handler)

    def thread_a():
        seen["a_text"] = llm.complete("prompt-a")
        a_finished_call.set()
        assert b_finished_call.wait(10)
        seen["a"] = llm.last_complete_status()

    def thread_b():
        assert a_finished_call.wait(10)
        seen["b_text"] = llm.complete("prompt-b")
        seen["b"] = llm.last_complete_status()
        b_finished_call.set()

    threads = [_run(thread_a), _run(thread_b)]
    for thread in threads:
        thread.join(15)
        assert not thread.is_alive()

    assert seen["a_text"] == "" and seen["b_text"] == "answer"
    assert seen["a"]["outcome"] == "timeout"
    assert seen["a"]["response_received"] is False
    assert seen["b"]["outcome"] == "success"
    assert seen["b"]["response_received"] is True
    assert seen["a"]["call_id"] != seen["b"]["call_id"]


def test_a_thread_that_never_called_complete_reads_not_called(openrouter):
    openrouter(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": "answer"}}],
        "provider": "CoreWeave"}))
    llm.complete("prompt")
    seen = {}

    def other():
        seen["status"] = llm.last_complete_status()

    _run(other).join(10)
    assert seen["status"]["outcome"] == "not_called"
    assert seen["status"]["call_id"] == 0
    assert seen["status"]["request_made"] is False


def test_call_ids_stay_unique_and_monotonic_across_threads(openrouter):
    openrouter(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": "answer"}}],
        "provider": "CoreWeave"}))
    ids, lock = [], threading.Lock()
    start = threading.Barrier(8)

    def worker():
        start.wait(10)
        mine = []
        for _ in range(10):
            llm.complete("prompt")
            mine.append(llm.last_complete_status()["call_id"])
        with lock:
            ids.extend(mine)
        assert mine == sorted(mine)

    threads = [_run(worker) for _ in range(8)]
    for thread in threads:
        thread.join(30)
        assert not thread.is_alive()
    assert len(ids) == 80
    assert len(set(ids)) == 80
