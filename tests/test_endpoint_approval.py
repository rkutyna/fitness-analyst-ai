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
    for key in ("HA_LLM_BACKEND", "HA_OPENROUTER_PROVIDERS",
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
    _llm(monkeypatch, HA_LLM_BACKEND="codex").assert_backend_approved()


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
