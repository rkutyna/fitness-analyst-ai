"""Public, identity-free progress narration for /v1/ask."""
from __future__ import annotations

import asyncio
from datetime import datetime
import threading

import httpx
import pytest

from health_advisor import ask_progress, llm, mcp_server, receiver


HEADERS = {"x-health-secret": "progress-secret"}


@pytest.fixture(autouse=True)
def isolated_progress(monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "progress-secret")
    receiver.PROGRESS_REGISTRY.clear()
    yield
    receiver.PROGRESS_REGISTRY.clear()


def _answer(*, mode="narration", callbacks=()):
    def answer(ctx, question, **kwargs):
        callback = kwargs.get("on_tool_call")
        for sequence, tool_name in enumerate(callbacks, 1):
            if callback is not None:
                callback(tool_name, sequence)
        return {"text": "synthetic answer", "mode": mode,
                "tool_trace": [], "verification": {}}
    return answer


async def _post_and_get(app, body):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/ask", json=body, headers=HEADERS)
        progress = await client.get(
            "/v1/ask/progress", params={"progress_id": body["progress_id"]},
            headers=HEADERS)
    return response, progress


def test_progress_id_records_ordered_steps_and_absent_id_records_nothing(
        monkeypatch, vault):
    monkeypatch.setattr(receiver.chat, "answer_question",
                        _answer(callbacks=("get_daily_series", "get_latest")))
    app = receiver.create_app(vault)

    response, progress = asyncio.run(_post_and_get(
        app, {"question": "synthetic question", "progress_id": "turn-one"}))
    assert response.status_code == 200
    assert progress.status_code == 200
    entry = progress.json()
    assert entry["state"] == "done"
    assert [step["seq"] for step in entry["steps"]] == [1, 2]
    assert [step["phrase"] for step in entry["steps"]] == [
        ask_progress.TOOL_PROGRESS_PHRASES["get_daily_series"],
        ask_progress.TOOL_PROGRESS_PHRASES["get_latest"],
    ]
    assert entry["finished_at"] is not None

    seen_callback_kwarg = []

    def no_progress_answer(ctx, question, **kwargs):
        seen_callback_kwarg.append("on_tool_call" in kwargs)
        return {"text": "synthetic answer", "mode": "fallback",
                "tool_trace": [], "verification": {}}

    monkeypatch.setattr(receiver.chat, "answer_question", no_progress_answer)
    transport = httpx.ASGITransport(app=app)

    async def post_without_id():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/ask", json={"question": "no progress"},
                                     headers=HEADERS)

    response = asyncio.run(post_without_id())
    assert response.status_code == 200
    assert seen_callback_kwarg == [False]
    assert receiver.PROGRESS_REGISTRY.get("no-progress") is None


def test_running_then_done_is_observable_without_sleep(monkeypatch, vault):
    entered_tool = threading.Event()
    release_tool = threading.Event()

    def scripted_answer(ctx, question, **kwargs):
        callback = kwargs["on_tool_call"]
        callback("get_daily_series", 1)
        entered_tool.set()
        assert release_tool.wait(5), "scripted tool was not released"
        callback("get_sleep_regularity", 2)
        return {"text": "synthetic answer", "mode": "narration",
                "tool_trace": [], "verification": {}}

    monkeypatch.setattr(receiver.chat, "answer_question", scripted_answer)
    app = receiver.create_app(vault)

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            post = asyncio.create_task(client.post(
                "/v1/ask", json={"question": "pause", "progress_id": "paused"},
                headers=HEADERS))
            assert await asyncio.to_thread(entered_tool.wait, 5)
            running = await client.get(
                "/v1/ask/progress", params={"progress_id": "paused"},
                headers=HEADERS)
            assert running.status_code == 200
            assert running.json()["state"] == "running"
            assert len(running.json()["steps"]) == 1
            release_tool.set()
            response = await post
            done = await client.get(
                "/v1/ask/progress", params={"progress_id": "paused"},
                headers=HEADERS)
            return response, done

    response, done = asyncio.run(exercise())
    assert response.status_code == 200
    assert done.json()["state"] == "done"
    assert len(done.json()["steps"]) == 2


@pytest.mark.parametrize(
    ("mode", "expected_state"),
    [("fallback", "fallback")],
)
def test_fallback_turn_ends_fallback(monkeypatch, vault, mode, expected_state):
    monkeypatch.setattr(receiver.chat, "answer_question", _answer(mode=mode))
    app = receiver.create_app(vault)
    _, progress = asyncio.run(_post_and_get(
        app, {"question": "fallback", "progress_id": "fallback-turn"}))
    assert progress.json()["state"] == expected_state


def test_exception_turn_ends_error(monkeypatch, vault):
    def broken_answer(ctx, question, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(receiver.chat, "answer_question", broken_answer)
    app = receiver.create_app(vault)

    async def exercise():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/ask", json={"question": "error", "progress_id": "error-turn"},
                headers=HEADERS)
            progress = await client.get(
                "/v1/ask/progress", params={"progress_id": "error-turn"},
                headers=HEADERS)
            return response, progress

    response, progress = asyncio.run(exercise())
    assert response.status_code == 500
    assert progress.json()["state"] == "error"


def test_vocabulary_is_complete_and_carries_no_figure_or_placeholder():
    names = {fn.__name__ for fn in mcp_server._TOOLS}
    assert len(names) == 32
    assert set(ask_progress.TOOL_PROGRESS_PHRASES) == names
    for phrase in ask_progress.TOOL_PROGRESS_PHRASES.values():
        assert not any(character.isdigit() for character in phrase)
        assert "{" not in phrase
    assert not any(character.isdigit() for character in ask_progress.GENERIC_PROGRESS_PHRASE)
    assert "{" not in ask_progress.GENERIC_PROGRESS_PHRASE


def _ollama_script(monkeypatch, tool_result, on_tool_call):
    responses = iter([
        {"message": {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_latest", "arguments": {}}}]}},
        {"message": {"role": "assistant", "content": "synthetic answer"}},
    ])

    def handler(request):
        return httpx.Response(200, json=next(responses))

    monkeypatch.setattr(
        llm, "_client",
        lambda timeout: httpx.Client(timeout=timeout,
                                     transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(
        llm, "_registry",
        lambda ctx, include=None: {
            "get_latest": (lambda **kwargs: tool_result,
                            {"type": "function", "function": {"name": "get_latest"}})
        },
    )
    return llm.tool_loop("synthetic", ctx=None, tools=[], max_turns=2,
                         on_tool_call=on_tool_call)


def test_tool_result_number_does_not_change_step_phrase(monkeypatch):
    registry = ask_progress.ProgressRegistry()
    registry.start("numbered")
    result = _ollama_script(
        monkeypatch, {"value": 8675309},
        lambda tool_name, sequence: registry.add_step("numbered", sequence, tool_name),
    )
    assert result == "synthetic answer"
    entry = registry.get("numbered")
    assert entry["steps"][0]["phrase"] == ask_progress.TOOL_PROGRESS_PHRASES["get_latest"]
    assert "8675309" not in entry["steps"][0]["phrase"]


def test_progress_hook_error_cannot_fail_tool_loop(monkeypatch):
    result = _ollama_script(
        monkeypatch, {"value": 42},
        lambda tool_name, sequence: (_ for _ in ()).throw(RuntimeError("hook")),
    )
    assert result == "synthetic answer"


def test_registry_is_bounded_and_expiring():
    now = [0.0]
    registry = ask_progress.ProgressRegistry(
        ttl_seconds=10, max_entries=64, clock=lambda: now[0],
        timestamp_fn=lambda: now[0])
    for index in range(65):
        registry.start(f"progress-{index}")
    assert registry.get("progress-0") is None
    assert registry.get("progress-1") is not None
    now[0] = 11.0
    assert registry.get("progress-64") is None


def test_three_tool_turn_reports_registry_time_to_first_step(monkeypatch, vault):
    monkeypatch.setattr(receiver.chat, "answer_question",
                        _answer(callbacks=("get_daily_series", "get_latest",
                                           "get_sleep_regularity")))
    app = receiver.create_app(vault)
    response, progress = asyncio.run(_post_and_get(
        app, {"question": "timed", "progress_id": "timed-turn"}))
    assert response.status_code == 200
    entry = progress.json()
    started = datetime.fromisoformat(entry["started_at"].replace("Z", "+00:00"))
    first = datetime.fromisoformat(entry["steps"][0]["at"].replace("Z", "+00:00"))
    finished = datetime.fromisoformat(entry["finished_at"].replace("Z", "+00:00"))
    first_offset = (first - started).total_seconds()
    total_wall = (finished - started).total_seconds()
    print(f"time-to-first-step={first_offset:.3f}s total-wall={total_wall:.3f}s")
    assert len(entry["steps"]) == 3
    assert first_offset >= 0
    assert total_wall >= first_offset
