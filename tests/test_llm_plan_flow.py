import importlib
import json

import httpx
import pytest

from health_advisor import llm


PRO_MODEL = "deepseek/deepseek-v4-pro"
FLASH_MODEL = "deepseek/deepseek-v4-flash-0731"
PRO_PROVIDERS = "novita/fp8,deepinfra/fp8,parasail/fp8"
DAILY_PROVIDERS = "coreweave/fp8,together,reka/fp4"


@pytest.fixture(autouse=True)
def restore_llm_module():
    yield
    importlib.reload(llm)


def _configure_openrouter(monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", FLASH_MODEL)
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "coreweave/fp8")
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "off")


def test_v4_pro_has_exact_admitted_provider_set():
    assert llm.APPROVED_OPENROUTER_PROVIDERS[PRO_MODEL] == frozenset({
        "novita/fp8", "fireworks", "deepinfra/fp8", "siliconflow/fp8",
        "baseten/fp4", "nextbit/fp8", "digitalocean", "parasail/fp8",
        "coreweave/fp8", "together",
    })


def test_pro_pin_rejects_provider_only_approved_for_flash(monkeypatch):
    _configure_openrouter(monkeypatch)
    monkeypatch.setattr(llm, "PLAN_MODEL", PRO_MODEL)
    monkeypatch.setattr(llm, "PLAN_PROVIDERS", DAILY_PROVIDERS)

    with pytest.raises(RuntimeError) as excinfo:
        llm.assert_backend_approved()

    message = str(excinfo.value)
    assert "HA_PLAN_MODEL" in message
    assert "HA_PLAN_PROVIDERS" in message
    assert "reka/fp4" in message


def test_plan_model_with_inherited_daily_pin_is_refused_from_environment(
        monkeypatch):
    monkeypatch.setenv("HA_LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setenv("HA_OPENROUTER_MODEL", FLASH_MODEL)
    monkeypatch.setenv("HA_OPENROUTER_PROVIDERS", DAILY_PROVIDERS)
    monkeypatch.setenv("HA_OPENROUTER_REASONING", "off")
    monkeypatch.setenv("HA_PLAN_MODEL", PRO_MODEL)
    monkeypatch.delenv("HA_PLAN_PROVIDERS", raising=False)
    importlib.reload(llm)

    with pytest.raises(RuntimeError) as excinfo:
        llm.assert_backend_approved()

    message = str(excinfo.value)
    assert "HA_PLAN_MODEL" in message
    assert "HA_PLAN_PROVIDERS" in message
    assert "reka/fp4" in message


def test_daily_and_plan_pairs_are_checked_with_plan_pin(monkeypatch):
    _configure_openrouter(monkeypatch)
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", DAILY_PROVIDERS)
    monkeypatch.setattr(llm, "PLAN_MODEL", PRO_MODEL)
    monkeypatch.setattr(llm, "PLAN_PROVIDERS", PRO_PROVIDERS)

    llm.assert_backend_approved()


def test_plan_configuration_defaults_to_daily_environment(monkeypatch):
    monkeypatch.setenv("HA_OPENROUTER_MODEL", FLASH_MODEL)
    monkeypatch.setenv("HA_OPENROUTER_PROVIDERS", DAILY_PROVIDERS)
    monkeypatch.delenv("HA_PLAN_MODEL", raising=False)
    monkeypatch.delenv("HA_PLAN_PROVIDERS", raising=False)

    importlib.reload(llm)

    assert llm.PLAN_MODEL is None and llm.PLAN_PROVIDERS is None
    assert llm._plan_model() == llm.OPENROUTER_MODEL == FLASH_MODEL
    assert llm._plan_providers() == llm.OPENROUTER_PROVIDERS == DAILY_PROVIDERS


def test_unset_plan_pair_follows_the_daily_pair_at_call_time(monkeypatch):
    # Regression: the plan pair once snapshotted the daily pair at import, so a
    # daily pair changed afterwards was checked against a stale plan pair.
    monkeypatch.setenv("HA_OPENROUTER_MODEL", FLASH_MODEL)
    monkeypatch.setenv("HA_OPENROUTER_PROVIDERS", DAILY_PROVIDERS)
    monkeypatch.delenv("HA_PLAN_MODEL", raising=False)
    monkeypatch.delenv("HA_PLAN_PROVIDERS", raising=False)
    importlib.reload(llm)
    _configure_openrouter(monkeypatch)
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "baseten/fp8")

    llm.assert_backend_approved()
    assert llm._plan_model() == "z-ai/glm-5.3-flash"
    assert llm._plan_providers() == "baseten/fp8"


def test_unset_daily_model_message_names_the_environment_variable(monkeypatch):
    _configure_openrouter(monkeypatch)
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", None)

    with pytest.raises(RuntimeError, match="HA_OPENROUTER_MODEL is unset"):
        llm.assert_backend_approved()


def test_baseten_response_name_is_resolved_for_each_model(monkeypatch):
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
    llm._assert_openrouter_response_provider({"provider": "BaseTen"})

    monkeypatch.setattr(llm, "OPENROUTER_MODEL", PRO_MODEL)
    monkeypatch.setattr(llm, "PLAN_MODEL", PRO_MODEL)
    llm._assert_openrouter_response_provider(
        {"provider": "BaseTen"}, model=PRO_MODEL)


def test_unknown_response_display_name_is_refused(monkeypatch):
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", PRO_MODEL)

    with pytest.raises(RuntimeError, match="Unknown Provider"):
        llm._assert_openrouter_response_provider(
            {"provider": "Unknown Provider"}, model=PRO_MODEL)


def test_complete_selects_daily_and_plan_payloads_and_max_tokens(monkeypatch):
    _configure_openrouter(monkeypatch)
    monkeypatch.setattr(llm, "PLAN_MODEL", PRO_MODEL)
    monkeypatch.setattr(llm, "PLAN_PROVIDERS", "novita/fp8")
    requests = []

    def handler(request):
        requests.append(request.content)
        body = json.loads(request.content)
        provider = "Novita" if body["model"] == PRO_MODEL else "CoreWeave"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "answer"}}],
            "provider": provider,
        })

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    assert llm.complete("prompt") == "answer"
    daily_without_flow = requests.pop()
    assert llm.complete("prompt", flow="daily", max_tokens=111) == "answer"
    daily_with_flow = requests.pop()
    assert daily_with_flow != daily_without_flow
    assert json.loads(daily_with_flow)["model"] == FLASH_MODEL
    assert json.loads(daily_with_flow)["max_tokens"] == 111

    assert llm.complete("prompt", flow="plan", max_tokens=222) == "answer"
    plan_body = json.loads(requests.pop())
    assert plan_body["model"] == PRO_MODEL
    assert plan_body["provider"]["order"] == ["novita/fp8"]
    assert plan_body["provider"]["only"] == ["novita/fp8"]
    assert plan_body["max_tokens"] == 222


def test_daily_flow_payload_is_byte_identical_to_omitted_flow(monkeypatch):
    _configure_openrouter(monkeypatch)
    requests = []

    def handler(request):
        requests.append(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "answer"}}],
            "provider": "CoreWeave",
        })

    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))
    assert llm.complete("prompt") == "answer"
    assert llm.complete("prompt", flow="daily") == "answer"
    assert requests[0] == requests[1]


def test_complete_rejects_unknown_flow(monkeypatch):
    with pytest.raises(ValueError, match="flow"):
        llm.complete("prompt", flow="weekly")
