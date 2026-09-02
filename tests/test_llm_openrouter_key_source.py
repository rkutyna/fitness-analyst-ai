"""The OpenRouter credential can leave the receiver environment (#145)."""
from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient

from health_advisor import llm as imported_llm
from health_advisor import receiver


MODEL = "deepseek/deepseek-v4-flash-0731"
FILE_KEY = "file-key-used-by-the-test"
ENV_KEY = "env-key-used-by-the-test"


@pytest.fixture(autouse=True)
def _restore_llm_module():
    names = ("HA_LLM_BACKEND", "HA_OPENROUTER_MODEL",
             "HA_OPENROUTER_PROVIDERS", "HA_OPENROUTER_REASONING",
             "HA_OPENROUTER_API_KEY_FILE", "OPENROUTER_API_KEY")
    ambient = {name: os.environ.get(name) for name in names}
    yield
    for name, value in ambient.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    importlib.reload(imported_llm)


def _reload(monkeypatch, **env):
    for name in ("HA_LLM_BACKEND", "HA_OPENROUTER_MODEL",
                 "HA_OPENROUTER_PROVIDERS", "HA_OPENROUTER_REASONING",
                 "HA_OPENROUTER_API_KEY_FILE", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return importlib.reload(imported_llm)


def _approved_env():
    return {
        "HA_LLM_BACKEND": "openrouter",
        "HA_OPENROUTER_MODEL": MODEL,
        "HA_OPENROUTER_PROVIDERS": "coreweave/fp8",
        "HA_OPENROUTER_REASONING": "off",
    }


def test_file_key_wins_when_environment_key_is_absent(monkeypatch, tmp_path):
    key_file = tmp_path / "openrouter.key"
    key_file.write_text(f"  {FILE_KEY}\n")
    key_file.chmod(0o600)

    llm = _reload(monkeypatch, **_approved_env(),
                  HA_OPENROUTER_API_KEY_FILE=str(key_file))

    assert llm.OPENROUTER_API_KEY == FILE_KEY
    assert llm.OPENROUTER_API_KEY_SOURCE == "file"


def test_environment_key_still_works_without_a_file(monkeypatch):
    llm = _reload(monkeypatch, **_approved_env(), OPENROUTER_API_KEY=ENV_KEY)

    assert llm.OPENROUTER_API_KEY == ENV_KEY
    assert llm.OPENROUTER_API_KEY_SOURCE == "env"


def test_disagreeing_file_and_environment_keys_are_refused_by_d15(
        monkeypatch, tmp_path):
    key_file = tmp_path / "openrouter.key"
    key_file.write_text(FILE_KEY)
    key_file.chmod(0o600)
    llm = _reload(monkeypatch, **_approved_env(),
                  HA_OPENROUTER_API_KEY_FILE=str(key_file),
                  OPENROUTER_API_KEY=ENV_KEY)

    with pytest.raises(RuntimeError, match="both set but disagree") as excinfo:
        llm.assert_backend_approved()

    assert FILE_KEY not in str(excinfo.value)
    assert ENV_KEY not in str(excinfo.value)


def test_health_reports_the_openrouter_key_source(monkeypatch, vault, tmp_path):
    key_file = tmp_path / "openrouter.key"
    key_file.write_text(FILE_KEY)
    key_file.chmod(0o600)
    llm = _reload(monkeypatch, **_approved_env(),
                  HA_OPENROUTER_API_KEY_FILE=str(key_file))
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")

    with TestClient(receiver.create_app(vault)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["openrouter_api_key_source"] == "file"
    assert FILE_KEY not in response.text
    assert llm.OPENROUTER_API_KEY_SOURCE == "file"
