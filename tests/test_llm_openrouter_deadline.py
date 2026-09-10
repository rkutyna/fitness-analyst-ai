"""Total-deadline and provider-pin coverage for OpenRouter requests.

Two fixtures back this file:

- ``deadline_stub`` (an ``httpx.MockTransport``) for assertions that do not
  depend on real socket behaviour: log content, per-flow provider demotion,
  and thread cleanup on a normal response.
- ``openrouter_loopback`` (a real loopback TCP server, one thread per
  connection) for every TIMING assertion. A mock transport has no socket, so
  it cannot exercise ``close_request()``'s socket-shutdown path at all: a
  stub that blocks on its own internal sleep/wait bounds the mock tests by
  that sleep, not by the deadline. Only a real socket can be shut down out
  from under a blocked read.
"""

import json
import socket
import threading
import time

import httpx
import pytest

from health_advisor import llm


FLASH_MODEL = "deepseek/deepseek-v4-flash-0731"


class _DeadlineStubBody(httpx.SyncByteStream):
    def __init__(self, mode: str, timeout: float):
        self.mode = mode
        self.timeout = timeout
        self.closed = threading.Event()

    def __iter__(self):
        if self.mode == "A":
            self.closed.wait(self.timeout + 1)
            raise httpx.ReadTimeout("silent stub read timeout")
        yield b" "
        if self.mode == "B":
            while not self.closed.wait(0.1):
                yield b" "
                time.sleep(0.1)
        else:
            time.sleep(self.timeout - 0.1)
            yield b" "
            if not self.closed.wait(self.timeout + 1):
                raise httpx.ReadTimeout("closed stub read")
            raise httpx.ReadTimeout("closed stub read")

    def close(self):
        self.closed.set()


@pytest.fixture
def deadline_stub(monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", FLASH_MODEL)
    monkeypatch.setattr(llm, "PLAN_MODEL", FLASH_MODEL)
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "first,second")
    monkeypatch.setattr(llm, "PLAN_PROVIDERS", "plan-first,plan-second")
    monkeypatch.setattr(llm, "OPENROUTER_MIN_THROUGHPUT", 0)
    mode = {"value": "C"}
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if mode["value"] in {"A", "B", "C"}:
            return httpx.Response(
                200,
                stream=_DeadlineStubBody(mode["value"], 1),
                request=request,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}]},
            request=request,
        )

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    return mode, requests


def _serve(mode: str, timeout: float) -> socket.socket:
    """One-thread-per-connection loopback HTTP stub.

    mode "A": read the request, send nothing (not even headers).
    mode "B": send headers, then one space every 0.5s forever.
    mode "C": send headers, one space at ``timeout - 0.3``s, then silence.
    """
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)

    def accept_loop():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return

            def handle(conn=conn):
                try:
                    conn.recv(65536)
                    if mode in ("B", "C"):
                        conn.sendall(
                            b"HTTP/1.1 200 OK\r\n"
                            b"Content-Type: application/json\r\n"
                            b"Content-Length: 1000000\r\n\r\n"
                        )
                    if mode == "C":
                        time.sleep(max(0.0, timeout - 0.3))
                        conn.sendall(b" ")
                    while True:
                        time.sleep(0.5)
                        if mode == "B":
                            conn.sendall(b" ")
                except OSError:
                    pass

            threading.Thread(target=handle, daemon=True).start()

    threading.Thread(target=accept_loop, daemon=True).start()
    return listener


@pytest.fixture
def openrouter_loopback(monkeypatch):
    """Point ``llm`` at a real loopback server instead of the mock transport.

    Returns a ``start(mode, timeout)`` callable; each call spins up a fresh
    server bound to a free port and repoints ``llm.OPENROUTER_URL`` at it.
    Every socket opened this way is closed in teardown.
    """
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", FLASH_MODEL)
    monkeypatch.setattr(llm, "PLAN_MODEL", FLASH_MODEL)
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "first,second")
    monkeypatch.setattr(llm, "PLAN_PROVIDERS", "plan-first,plan-second")
    monkeypatch.setattr(llm, "OPENROUTER_MIN_THROUGHPUT", 0)
    monkeypatch.setattr(llm, "_TRANSPORT", None)

    listeners: list[socket.socket] = []

    def start(mode: str, timeout: float) -> None:
        listener = _serve(mode, timeout)
        listeners.append(listener)
        port = listener.getsockname()[1]
        monkeypatch.setattr(llm, "OPENROUTER_URL", f"http://127.0.0.1:{port}/api/v1")

    yield start

    for listener in listeners:
        listener.close()


# The margin must be tighter than the defect it pins. A timer that closes the
# response instead of shutting the socket down returns after about one extra read
# timeout (measured 5.5-5.7 s at a 3 s deadline, i.e. ~1.9x). At deadline 1 with
# a +1 s margin that defect PASSED, so these tests run at 2 s with a 1.25x margin,
# which the fix meets (~2.02 s) and the defect cannot (~3.8 s).
DEADLINE = 2
SLACK = 1.25


@pytest.mark.parametrize("mode", ["A", "B", "C"])
@pytest.mark.parametrize("flow", ["daily", "plan"])
def test_complete_bounded_against_real_server(openrouter_loopback, capsys, mode, flow):
    """complete() against a real stalling socket is bounded by the deadline,
    not by however long the peer chooses to sit on the connection, and logs
    the stall exactly once."""
    openrouter_loopback(mode, DEADLINE)
    started = time.monotonic()

    assert llm.complete("prompt", flow=flow, timeout=DEADLINE) == ""

    assert time.monotonic() - started <= DEADLINE * SLACK
    assert llm.last_complete_status()["outcome"] == "timeout"
    assert capsys.readouterr().err.count("openrouter_deadline") == 1


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_post_bounded_against_real_server(openrouter_loopback, mode):
    """The tool-loop path (_openrouter_post) is bounded the same way and
    raises rather than degrading to an empty string."""
    openrouter_loopback(mode, DEADLINE)
    started = time.monotonic()

    with pytest.raises(httpx.TimeoutException):
        llm._openrouter_post([], tools=[], timeout=DEADLINE)

    assert time.monotonic() - started <= DEADLINE * SLACK


def test_four_c_calls_fit_chain_budget(openrouter_loopback):
    """Four sequential stalled calls each pay at most one deadline, not an
    accumulating or compounding penalty."""
    openrouter_loopback("C", DEADLINE)
    started = time.monotonic()

    for _ in range(4):
        assert llm.complete("prompt", timeout=DEADLINE) == ""
        assert llm.last_complete_status()["outcome"] == "timeout"

    assert time.monotonic() - started <= 4 * DEADLINE * SLACK


def test_plan_stall_demotes_only_plan_pin(deadline_stub):
    mode, requests = deadline_stub
    assert llm.complete("prompt", flow="plan", timeout=1) == ""
    assert llm.last_complete_status()["outcome"] == "timeout"

    assert llm.OPENROUTER_PROVIDERS == "first,second"
    assert llm.PLAN_PROVIDERS == "plan-second,plan-first"
    assert requests[0]["provider"]["order"] == ["plan-first", "plan-second"]

    mode["value"] = "normal"
    assert llm.complete("prompt", timeout=1) == "answer"
    assert requests[1]["provider"]["order"] == ["first", "second"]


def test_daily_stall_demotes_only_daily_pin(deadline_stub):
    mode, requests = deadline_stub
    with pytest.raises(httpx.TimeoutException):
        llm._openrouter_post([], tools=[], timeout=1)

    assert llm.OPENROUTER_PROVIDERS == "second,first"
    assert llm.PLAN_PROVIDERS == "plan-first,plan-second"
    assert requests[0]["provider"]["order"] == ["first", "second"]

    mode["value"] = "normal"
    assert llm.complete("prompt", flow="plan", timeout=1) == "answer"
    assert requests[1]["provider"]["order"] == ["plan-first", "plan-second"]


def test_normal_response_cancels_deadline_timer(deadline_stub):
    mode, _ = deadline_stub
    mode["value"] = "normal"
    before = {thread.ident for thread in threading.enumerate()
              if thread.name.startswith("Thread-")}

    assert llm.complete("prompt", timeout=1) == "answer"
    time.sleep(0.05)
    after = {thread.ident for thread in threading.enumerate()
             if thread.name.startswith("Thread-")}
    assert after <= before
