"""Bring-your-own OpenAI-compatible endpoint (#79).

Four properties, one per Done-when:

1. ``openai_compatible`` pointed at a non-OpenRouter endpoint -- here a real
   HTTP server on loopback serving the chat-completions shape -- completes
   ``complete``, ``tool_loop`` and ``research_loop``, and sends it none of
   OpenRouter's private request fields.
2. An unset or unmatched approval config refuses at startup. The engine ships
   no approved host for this backend, loopback included, and configuring it
   widens nothing for the ``openrouter`` profile.
3. A config reproducing the pre-#79 OpenRouter pin produces byte-identical
   requests: the golden file was captured by ``_capture_openrouter_wire`` on
   the commit before #79 and is compared byte for byte (bodies, URLs and every
   header; only httpx's own version string is normalised).
4. is the README, not a test.
"""
from __future__ import annotations

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

GOLDEN = Path(__file__).parent / "fixtures" / "openrouter_wire_golden.json"

_LLM_ENV = (
    "HA_LLM_BACKEND", "HA_CODEX_BIN",
    "OPENROUTER_API_KEY", "HA_OPENROUTER_API_KEY_FILE",
    "HA_OPENROUTER_MODEL", "HA_OPENROUTER_PROVIDERS",
    "HA_OPENROUTER_PROVIDER_SORT", "HA_OPENROUTER_REASONING",
    "HA_OPENROUTER_URL", "HA_OPENROUTER_MIN_THROUGHPUT",
    "HA_PLAN_MODEL", "HA_PLAN_PROVIDERS", "HA_OLLAMA_URL",
    "HA_OPENAI_COMPAT_URL", "HA_OPENAI_COMPAT_MODEL",
    "HA_OPENAI_COMPAT_APPROVED_HOSTS", "HA_OPENAI_COMPAT_API_KEY",
    "HA_OPENAI_COMPAT_API_KEY_FILE",
    # A proxy variable would route the loopback double through a proxy.
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


@pytest.fixture(autouse=True)
def _restore_llm_module():
    """Reload llm back under the ambient environment after each test: its
    config is read at import, and other modules hold references to it."""
    yield
    import health_advisor.llm as llm
    importlib.reload(llm)


def _llm(monkeypatch, **env):
    for key in _LLM_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import health_advisor.llm as llm
    return importlib.reload(llm)


# --------------------------------------------------------------------------
# Done-when 3 -- the OpenRouter profile's wire is byte-identical to pre-#79
# --------------------------------------------------------------------------

PINNED_ENV = {
    "HA_LLM_BACKEND": "openrouter",
    "OPENROUTER_API_KEY": "wire-test-key",
    "HA_OPENROUTER_MODEL": "deepseek/deepseek-v4-flash-0731",
    "HA_OPENROUTER_PROVIDERS": "coreweave/fp8,nextbit/fp8",
    "HA_OPENROUTER_PROVIDER_SORT": "throughput",
    "HA_OPENROUTER_REASONING": "low",
    "HA_PLAN_MODEL": "deepseek/deepseek-v4-pro",
    "HA_PLAN_PROVIDERS": "novita/fp8,parasail/fp8",
}


def _reply(content="", tool_calls=None, provider="CoreWeave"):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "stop"}],
            "provider": provider,
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def _call(name, arguments, call_id):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


_PROBE_SCHEMA = {"type": "function",
                 "function": {"name": "probe", "description": "probe",
                              "parameters": {"type": "object",
                                             "properties": {"days": {"type": "integer"}}}}}


def _probe(days=7):
    return {"mean": 5413, "days": days}


def _run_script(llm, monkeypatch, serve):
    """The fixed call script both halves of this file drive.

    ``serve(replies)`` queues the replies the endpoint will return next.
    """
    monkeypatch.setattr(llm, "_registry",
                        lambda ctx, include=None, **k: {"probe": (_probe, _PROBE_SCHEMA)})
    serve([_reply("daily text")])
    assert llm.complete("daily prompt") == "daily text"
    serve([_reply("judge text")])
    assert llm.complete("judge prompt", think=True, options=llm.JUDGE_OPTS,
                        max_tokens=llm.ANSWER_COMPLETION_MAX_TOKENS) == "judge text"
    serve([_reply("plan text", provider="Novita")])
    assert llm.complete("plan prompt", flow="plan", max_tokens=900) == "plan text"

    serve([_reply(tool_calls=[_call("probe", {"days": 14}, "c1")]),
           _reply("tool loop answer")])
    out = llm.tool_loop("gather prompt", ctx=None, tools=[_PROBE_SCHEMA],
                        max_turns=4)
    assert str(out) == "tool loop answer", llm.last_loop_status()

    serve([_reply(tool_calls=[_call("probe", {"days": 30}, "r1")]),
           _reply("research answer")])
    out = llm.research_loop("research prompt", ctx=None,
                            extra_tools={"probe": (_probe, _PROBE_SCHEMA)},
                            compact_state=lambda: "STATE", max_turns=4)
    assert str(out) == "research answer", llm.last_loop_status()


def _capture_openrouter_wire(monkeypatch) -> list[dict]:
    llm = _llm(monkeypatch, **PINNED_ENV)
    llm.assert_backend_approved()
    records: list[dict] = []
    queue: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        records.append({
            "method": request.method,
            "url": str(request.url),
            "headers": [[k, v] for k, v in request.headers.multi_items()],
            "body": request.content.decode("utf-8"),
        })
        return httpx.Response(200, json=queue.pop(0) if queue else _reply("done"))

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    _run_script(llm, monkeypatch, lambda replies: queue.__setitem__(slice(None), replies))
    return records


def test_openrouter_profile_requests_are_byte_identical_to_pre_79(monkeypatch):
    records = _capture_openrouter_wire(monkeypatch)
    agent = f"python-httpx/{httpx.__version__}"
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for record in golden:
        record["headers"] = [[k, agent if k == "user-agent" else v]
                             for k, v in record["headers"]]
    assert len(records) == len(golden) == 7
    for got, want in zip(records, golden):
        assert got["url"] == want["url"]
        assert got["headers"] == want["headers"]
        assert got["body"].encode("utf-8") == want["body"].encode("utf-8")


def test_golden_wire_carries_the_openrouter_fields(monkeypatch):
    """The oracle is not vacuous: it pins the fields #79 must keep sending to
    OpenRouter, so a regression that dropped them could not match it."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for record in golden:
        body = json.loads(record["body"])
        assert body["reasoning"] == {"effort": "low"}
        assert body["provider"]["allow_fallbacks"] is False
        assert body["provider"]["require_parameters"] is True
        assert record["url"] == "https://openrouter.ai/api/v1/chat/completions"


# --------------------------------------------------------------------------
# Done-when 1 -- a loopback OpenAI-compatible double serves all three paths
# --------------------------------------------------------------------------

class _Double:
    """A real HTTP server on 127.0.0.1 speaking the chat-completions shape."""

    def __init__(self):
        self.requests: list[dict] = []
        self.queue: list[dict] = []
        double = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 -- http.server's name
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length)
                double.requests.append({"path": self.path,
                                        "headers": dict(self.headers.items()),
                                        "body": json.loads(raw)})
                reply = double.queue.pop(0) if double.queue else _reply("done")
                reply = {k: v for k, v in reply.items() if k != "provider"}
                out = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def serve(self, replies):
        self.queue[:] = replies

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def double():
    server = _Double()
    yield server
    server.close()


def _compat_env(double, **extra):
    return {"HA_LLM_BACKEND": "openai_compatible",
            "HA_OPENAI_COMPAT_URL": f"http://127.0.0.1:{double.port}/v1",
            "HA_OPENAI_COMPAT_MODEL": "local-test-model",
            "HA_OPENAI_COMPAT_APPROVED_HOSTS": "127.0.0.1",
            **extra}


def test_loopback_double_serves_complete_tool_loop_and_research_loop(
        monkeypatch, double):
    llm = _llm(monkeypatch, **_compat_env(double))
    llm.assert_backend_approved()
    _run_script(llm, monkeypatch, double.serve)

    assert len(double.requests) == 7
    for request in double.requests:
        assert request["path"] == "/v1/chat/completions"
        body = request["body"]
        assert body["model"] == "local-test-model"
        # OpenRouter's private fields go to OpenRouter only.
        assert "provider" not in body
        assert "reasoning" not in body
        assert "require_parameters" not in json.dumps(body)
        # No key configured: no Authorization header at all.
        assert "authorization" not in {k.lower() for k in request["headers"]}
    tool_turns = [r["body"] for r in double.requests if "tools" in r["body"]]
    assert tool_turns and all(b["tool_choice"] == "auto" for b in tool_turns)
    # The tool result went back keyed by tool_call_id, in the OpenAI dialect.
    replies = [m for b in tool_turns for m in b["messages"] if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in replies} >= {"c1", "r1"}


def test_configured_key_is_sent_as_a_bearer_token(monkeypatch, double):
    llm = _llm(monkeypatch, **_compat_env(
        double, HA_OPENAI_COMPAT_API_KEY="compat-test-key"))
    double.serve([_reply("ok")])
    assert llm.complete("p") == "ok"
    headers = {k.lower(): v for k, v in double.requests[0]["headers"].items()}
    assert headers["authorization"] == "Bearer compat-test-key"


def test_key_file_follows_the_openrouter_key_rules(monkeypatch, tmp_path, double):
    key = tmp_path / "compat.key"
    key.write_text("from-file\n")
    key.chmod(0o644)
    with pytest.raises(RuntimeError, match="D16 requires 600"):
        _llm(monkeypatch, **_compat_env(
            double, HA_OPENAI_COMPAT_API_KEY_FILE=str(key)))
    key.chmod(0o600)
    llm = _llm(monkeypatch, **_compat_env(
        double, HA_OPENAI_COMPAT_API_KEY_FILE=str(key),
        HA_OPENAI_COMPAT_API_KEY="something-else"))
    with pytest.raises(RuntimeError, match="disagree"):
        llm.assert_backend_approved()


def test_unconfigured_complete_makes_no_request(monkeypatch, double):
    """A direct caller that skipped the startup gate still cannot reach an
    endpoint whose host nobody approved."""
    env = _compat_env(double)
    del env["HA_OPENAI_COMPAT_APPROVED_HOSTS"]
    llm = _llm(monkeypatch, **env)
    assert llm.complete("p") == ""
    status = llm.last_complete_status()
    assert status["request_made"] is False
    assert "HA_OPENAI_COMPAT_APPROVED_HOSTS is unset" in status["detail"]
    assert double.requests == []


def test_unconfigured_loops_make_no_request(monkeypatch, double):
    env = _compat_env(double, HA_OPENAI_COMPAT_APPROVED_HOSTS="localhost")
    llm = _llm(monkeypatch, **env)
    monkeypatch.setattr(llm, "_registry", lambda ctx, include=None, **k: {})
    assert str(llm.tool_loop("q", ctx=None, tools=[])) == ""
    assert str(llm.research_loop("q", ctx=None, extra_tools={},
                                 compact_state=lambda: "S")) == ""
    assert double.requests == []
    assert llm.last_loop_status()["outcome"] == "openai_compatible_not_approved"


# --------------------------------------------------------------------------
# Done-when 2 -- unset or unmatched approval config refuses at startup
# --------------------------------------------------------------------------

BASE = {"HA_LLM_BACKEND": "openai_compatible",
        "HA_OPENAI_COMPAT_URL": "https://models.example.test/v1",
        "HA_OPENAI_COMPAT_MODEL": "some-model",
        "HA_OPENAI_COMPAT_APPROVED_HOSTS": "models.example.test"}


def _without(key, **extra):
    env = {k: v for k, v in BASE.items() if k != key}
    env.update(extra)
    return env


def test_a_fully_stated_remote_endpoint_passes(monkeypatch):
    _llm(monkeypatch, **BASE).assert_backend_approved()


@pytest.mark.parametrize("env,match", [
    (_without("HA_OPENAI_COMPAT_APPROVED_HOSTS"),
     "HA_OPENAI_COMPAT_APPROVED_HOSTS is unset"),
    (_without("HA_OPENAI_COMPAT_APPROVED_HOSTS",
              HA_OPENAI_COMPAT_APPROVED_HOSTS=" , "),
     "HA_OPENAI_COMPAT_APPROVED_HOSTS is empty"),
    (_without("HA_OPENAI_COMPAT_URL"), "HA_OPENAI_COMPAT_URL is unset"),
    (_without("HA_OPENAI_COMPAT_MODEL"), "HA_OPENAI_COMPAT_MODEL is unset"),
], ids=["hosts-unset", "hosts-empty", "url-unset", "model-unset"])
def test_unset_config_refuses(monkeypatch, env, match):
    llm = _llm(monkeypatch, **env)
    with pytest.raises(RuntimeError, match=match):
        llm.assert_backend_approved()


@pytest.mark.parametrize("url", [
    "https://other.example.test/v1",
    "https://models.example.test.attacker.example/v1",   # substring, not host
    "https://api.models.example.test/v1",                # a subdomain
    "https://models.example.tes/v1",                     # a prefix
])
def test_unmatched_host_refuses(monkeypatch, url):
    llm = _llm(monkeypatch, **_without("HA_OPENAI_COMPAT_URL",
                                       HA_OPENAI_COMPAT_URL=url))
    with pytest.raises(RuntimeError, match="not in the approved set"):
        llm.assert_backend_approved()


@pytest.mark.parametrize("hosts", ["*", "*.example.test",
                                   "https://models.example.test",
                                   "models.example.test/v1"])
def test_patterns_in_the_approved_list_are_refused_not_interpreted(
        monkeypatch, hosts):
    llm = _llm(monkeypatch, **_without("HA_OPENAI_COMPAT_APPROVED_HOSTS",
                                       HA_OPENAI_COMPAT_APPROVED_HOSTS=hosts))
    with pytest.raises(RuntimeError, match="not a bare hostname"):
        llm.assert_backend_approved()


def test_plain_http_to_a_remote_host_refuses(monkeypatch):
    llm = _llm(monkeypatch, **_without(
        "HA_OPENAI_COMPAT_URL", HA_OPENAI_COMPAT_URL="http://models.example.test/v1"))
    with pytest.raises(RuntimeError, match="is not https"):
        llm.assert_backend_approved()


def test_loopback_is_not_approved_by_default(monkeypatch):
    """No default approved host, loopback included: a local URL is refused
    until the operator lists it."""
    llm = _llm(monkeypatch, **_without(
        "HA_OPENAI_COMPAT_URL", HA_OPENAI_COMPAT_URL="http://127.0.0.1:8080/v1"))
    with pytest.raises(RuntimeError, match="not in the approved set"):
        llm.assert_backend_approved()


@pytest.mark.parametrize("url,hosts", [
    ("http://127.0.0.1:8080/v1", "127.0.0.1"),
    ("http://localhost:1234/v1", "localhost"),
    ("http://[::1]:11434/v1", "::1"),
    ("http://[::1]:11434/v1", "[::1]"),
])
def test_listed_loopback_passes_over_plain_http(monkeypatch, url, hosts):
    _llm(monkeypatch, **_without(
        "HA_OPENAI_COMPAT_URL", HA_OPENAI_COMPAT_URL=url,
        HA_OPENAI_COMPAT_APPROVED_HOSTS=hosts)).assert_backend_approved()


def test_a_listed_lan_host_still_needs_tls(monkeypatch):
    """TLS is exempt for the written-out loopback set only, not for whatever
    host the operator happens to list."""
    llm = _llm(monkeypatch, **_without(
        "HA_OPENAI_COMPAT_URL", HA_OPENAI_COMPAT_URL="http://192.168.1.20:8000/v1",
        HA_OPENAI_COMPAT_APPROVED_HOSTS="192.168.1.20"))
    with pytest.raises(RuntimeError, match="is not https"):
        llm.assert_backend_approved()


def test_compat_hosts_do_not_widen_the_openrouter_profile(monkeypatch):
    """Configuring the new backend's list is not an approval for any other."""
    llm = _llm(monkeypatch, **PINNED_ENV,
               HA_OPENROUTER_URL="https://models.example.test/api/v1",
               HA_OPENAI_COMPAT_APPROVED_HOSTS="models.example.test")
    with pytest.raises(RuntimeError, match="not in the approved set"):
        llm.assert_backend_approved()


def test_the_profile_is_never_selected_by_absence(monkeypatch):
    """With no backend named the process is on codex, not on the OpenRouter
    profile; the profile is reached only by naming it."""
    llm = _llm(monkeypatch)
    assert llm.BACKEND == "codex"
