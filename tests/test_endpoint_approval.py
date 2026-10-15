"""D15 ranges over the destination, not only over who is meant to be at it (#138).

Measured 2026-08-26, before the fix: `HA_OPENROUTER_URL='https://not-openrouter.example/v1'`
with an approved provider pin **passed** `assert_backend_approved()`. The provider
allow-list decides who OpenRouter routes to; it says nothing about who receives
the request, and the endpoint is an environment variable. Same shape as the hole
#76 closed for the backend name, on the axis it did not range over.

The cases below are #138's `Not done when` lines turned into tests: a substring
match would let `openrouter.ai.attacker.example` through, and "local" must be
checked rather than assumed of `HA_OLLAMA_URL`.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _restore_llm_module():
    """`importlib.reload` rebinds llm's module-level config, and other test
    modules hold references to it. Reload it back under the ambient environment
    once each test is done, so nothing here depends on or leaks test order.
    """
    yield
    import health_advisor.llm as llm
    importlib.reload(llm)


def _llm(monkeypatch, **env):
    """Re-import llm with a patched environment — its config is read at import."""
    for key in ("HA_LLM_BACKEND", "HA_CODEX_BIN",
                "HA_OPENROUTER_PROVIDERS",
                "HA_OPENROUTER_REASONING", "HA_OPENROUTER_MODEL",
                "HA_OPENROUTER_URL", "HA_OLLAMA_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import health_advisor.llm as llm
    return importlib.reload(llm)


OPENROUTER_OK = {"HA_LLM_BACKEND": "openrouter",
                 "HA_OPENROUTER_MODEL": "deepseek/deepseek-v4-flash-0731",
                 "HA_OPENROUTER_PROVIDERS": "coreweave/fp8",
                 "HA_OPENROUTER_REASONING": "off"}


def test_the_documented_sweep_configuration_still_passes(monkeypatch):
    """The command in the skill must keep working. A tightening discovered by a
    user rather than by running it is #138's second `Not done when`."""
    _llm(monkeypatch, **OPENROUTER_OK).assert_backend_approved()


def test_default_ollama_endpoint_passes(monkeypatch):
    _llm(monkeypatch, HA_LLM_BACKEND="ollama").assert_backend_approved()


def test_codex_is_unaffected(monkeypatch):
    """HA_CODEX_BIN is a separate axis, recorded on #138, not covered here."""
    _llm(monkeypatch, HA_LLM_BACKEND="codex",
         HA_CODEX_BIN="codex-test").assert_backend_approved()


def test_unset_backend_is_refused_even_with_codex_on_path(monkeypatch, tmp_path):
    """#99: no default backend. A codex binary being findable must not make an
    unset HA_LLM_BACKEND pass the gate."""
    fake = tmp_path / "codex"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    llm = _llm(monkeypatch)
    assert llm.BACKEND == ""
    with pytest.raises(RuntimeError) as err:
        llm.assert_backend_approved()
    message = str(err.value)
    assert "HA_LLM_BACKEND" in message
    for name in ("ollama", "openrouter", "openai_compatible", "codex"):
        assert name in message


def test_empty_backend_variable_is_refused_like_an_unset_one(monkeypatch):
    llm = _llm(monkeypatch, HA_LLM_BACKEND="")
    with pytest.raises(RuntimeError, match="HA_LLM_BACKEND"):
        llm.assert_backend_approved()


def test_unset_backend_complete_returns_empty_without_a_request(monkeypatch):
    """The per-call paths must not fall through to the Ollama transport."""
    llm = _llm(monkeypatch)

    def _no_network(*args, **kwargs):
        raise AssertionError("an unnamed backend reached a transport")

    monkeypatch.setattr(llm.httpx, "Client", _no_network)
    monkeypatch.setattr(llm, "_codex_exec", _no_network)
    assert llm.complete("hello") == ""
    status = llm.last_complete_status()
    assert status["outcome"] == "backend_error"
    assert status["request_made"] is False
    assert "HA_LLM_BACKEND" in status["detail"]


def test_unset_backend_loops_refuse_without_a_request(monkeypatch):
    llm = _llm(monkeypatch)

    def _no_network(*args, **kwargs):
        raise AssertionError("an unnamed backend reached a transport")

    monkeypatch.setattr(llm.httpx, "Client", _no_network)
    monkeypatch.setattr(llm, "_codex_exec", _no_network)
    answer = llm.tool_loop("q", ctx=None, tools=[])
    assert not answer.text
    assert llm.last_loop_status()["outcome"] == "unset_not_approved"
    answer = llm.research_loop("q", ctx=None, extra_tools={}, compact_state=None)
    assert not answer.text
    assert llm.last_loop_status()["outcome"] == "unset_not_approved"


def test_codex_is_chosen_only_by_naming_it(monkeypatch):
    """#99 keeps codex approved as an explicit opt-in."""
    assert "codex" in _llm(monkeypatch, HA_LLM_BACKEND="codex",
                           HA_CODEX_BIN="codex-test").APPROVED_BACKENDS


def test_codex_without_binary_is_refused_at_approval(monkeypatch, tmp_path):
    """No configured or discoverable binary reaches subprocess execution."""
    monkeypatch.setenv("PATH", str(tmp_path))
    llm = _llm(monkeypatch, HA_LLM_BACKEND="codex")
    with pytest.raises(RuntimeError, match="HA_CODEX_BIN"):
        llm.assert_backend_approved()


@pytest.mark.parametrize("url,reason", [
    ("https://not-openrouter.example/v1", "an unrelated host"),
    ("https://openrouter.ai.attacker.example/v1", "substring, not the host"),
    ("https://api.openrouter.ai/v1", "a different host, however plausible"),
    ("https://openrouter.ai.evil.test/api/v1", "substring again, deeper"),
])
def test_redirected_openrouter_endpoint_is_refused(monkeypatch, url, reason):
    llm = _llm(monkeypatch, **OPENROUTER_OK, HA_OPENROUTER_URL=url)
    with pytest.raises(RuntimeError, match="not approved under D15"):
        llm.assert_backend_approved()


def test_plain_http_to_the_real_host_is_refused(monkeypatch):
    llm = _llm(monkeypatch, **OPENROUTER_OK,
               HA_OPENROUTER_URL="http://openrouter.ai/api/v1")
    with pytest.raises(RuntimeError, match="is not https"):
        llm.assert_backend_approved()


def test_endpoint_with_no_host_is_refused(monkeypatch):
    llm = _llm(monkeypatch, **OPENROUTER_OK, HA_OPENROUTER_URL="not-a-url")
    with pytest.raises(RuntimeError, match="has no host"):
        llm.assert_backend_approved()


@pytest.mark.parametrize("url", ["http://ollama.example.com:11434",
                                 "https://ollama.example.com"])
def test_remote_ollama_is_refused(monkeypatch, url):
    """The default being local is not the same as the variable being local."""
    llm = _llm(monkeypatch, HA_LLM_BACKEND="ollama", HA_OLLAMA_URL=url)
    with pytest.raises(RuntimeError, match="not approved under D15"):
        llm.assert_backend_approved()


@pytest.mark.parametrize("url", ["http://localhost:9999", "http://127.0.0.1:1234",
                                 "http://[::1]:11434"])
def test_loopback_ollama_on_any_port_passes(monkeypatch, url):
    """Plain HTTP to loopback is fine — there is no network hop to protect."""
    _llm(monkeypatch, HA_LLM_BACKEND="ollama",
         HA_OLLAMA_URL=url).assert_backend_approved()


def test_unapproved_provider_is_still_refused(monkeypatch):
    """The control: the pre-existing half of the check must not have moved."""
    llm = _llm(monkeypatch, HA_LLM_BACKEND="openrouter",
               HA_OPENROUTER_MODEL="deepseek/deepseek-v4-flash-0731",
               HA_OPENROUTER_PROVIDERS="some/other",
               HA_OPENROUTER_REASONING="off")
    with pytest.raises(RuntimeError, match="HA_OPENROUTER_PROVIDERS names"):
        llm.assert_backend_approved()


def test_missing_provider_pin_is_still_refused(monkeypatch):
    llm = _llm(monkeypatch, HA_LLM_BACKEND="openrouter",
               HA_OPENROUTER_MODEL="deepseek/deepseek-v4-flash-0731",
               HA_OPENROUTER_REASONING="off")
    with pytest.raises(RuntimeError, match="must be set and non-empty"):
        llm.assert_backend_approved()
