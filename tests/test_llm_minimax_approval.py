"""MiniMax M3.1 Flash's D15 approval: exactly two endpoint tags, both directions.

The request side (``assert_backend_approved``) accepts the exact pin as the
daily pair, as the plan pair, and as the daily pair the plan flow inherits, and
refuses a pin naming any tag outside the set -- including tags approved for a
DIFFERENT model and near misses of the two approved ones, which is the point of
keying the table by model. The response side accepts only the two display names
that map into the set. The request body is the engine's ordinary one: the full
sampling profile and the hard provider filters, nothing omitted.

Nothing here touches a network, a real key or real data.
"""
from __future__ import annotations

import importlib
import json

import httpx
import pytest

from health_advisor import llm

MINIMAX = "minimax/minimax-m3.1-flash"
PIN = "together,parasail/fp8"
APPROVED = frozenset({"together", "parasail/fp8"})
FLASH = "deepseek/deepseek-v4-flash-0731"
FLASH_PIN = "coreweave/fp8,together"
DISPLAY = {"Together": "together", "Parasail": "parasail/fp8"}

# Near misses of the approved tags, and tags OpenRouter lists for the model
# that the benchmark did not pick. Every one must be refused.
NEAR_MISSES = ("parasail", "parasail/fp4", "together/fp8", "Together",
               "minimax", "minimax/fp8", "nebius/fp4", "novita/fp16", "wafer")


@pytest.fixture(autouse=True)
def restore_llm_module():
    yield
    importlib.reload(llm)


def _approve(monkeypatch, *, model, pin, plan_model=None, plan_pin=None):
    monkeypatch.setattr(llm, "BACKEND", "openrouter")
    monkeypatch.setattr(llm, "OPENROUTER_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(llm, "_OPENROUTER_API_KEY_FILE_VALUE", None)
    monkeypatch.setattr(llm, "OPENROUTER_REASONING", "low")
    monkeypatch.setattr(llm, "OPENROUTER_MODEL", model)
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", pin)
    monkeypatch.setattr(llm, "PLAN_MODEL", plan_model)
    monkeypatch.setattr(llm, "PLAN_PROVIDERS", plan_pin)
    llm.assert_backend_approved()


def _tags_approved_only_elsewhere() -> list[str]:
    others = set().union(*(tags for model, tags in
                           llm.APPROVED_OPENROUTER_PROVIDERS.items()
                           if model != MINIMAX))
    return sorted(others - APPROVED)


def test_minimax_has_exactly_the_benchmark_pin_as_its_approved_set():
    assert llm.APPROVED_OPENROUTER_PROVIDERS[MINIMAX] == APPROVED
    assert frozenset(PIN.split(",")) == APPROVED


def test_both_display_names_were_already_mapped_and_map_to_one_tag_each():
    for name, tag in DISPLAY.items():
        assert llm.OPENROUTER_PROVIDER_TAGS[name] == frozenset({tag}), name


def test_the_exact_pin_is_accepted_as_daily_and_as_plan_pair(monkeypatch):
    _approve(monkeypatch, model=MINIMAX, pin=PIN)
    _approve(monkeypatch, model=MINIMAX, pin="parasail/fp8,together")
    _approve(monkeypatch, model=FLASH, pin=FLASH_PIN,
             plan_model=MINIMAX, plan_pin=PIN)
    _approve(monkeypatch, model=MINIMAX, pin=PIN,
             plan_model=MINIMAX, plan_pin=PIN)


def test_the_daily_pin_from_the_environment_is_inherited_by_the_plan_flow(
        monkeypatch):
    """The deployment shape: HA_PLAN_MODEL and HA_PLAN_PROVIDERS unset."""
    monkeypatch.setenv("HA_LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.delenv("HA_OPENROUTER_API_KEY_FILE", raising=False)
    monkeypatch.delenv("HA_OPENROUTER_URL", raising=False)
    monkeypatch.setenv("HA_OPENROUTER_MODEL", MINIMAX)
    monkeypatch.setenv("HA_OPENROUTER_PROVIDERS", PIN)
    monkeypatch.setenv("HA_OPENROUTER_REASONING", "low")
    monkeypatch.delenv("HA_PLAN_MODEL", raising=False)
    monkeypatch.delenv("HA_PLAN_PROVIDERS", raising=False)
    importlib.reload(llm)

    llm.assert_backend_approved()
    assert (llm._plan_model(), llm._plan_providers()) == (MINIMAX, PIN)


def test_there_are_tags_approved_for_other_models_and_not_for_minimax():
    """Non-vacuous: the refusal test below ranges over a real list."""
    outsiders = _tags_approved_only_elsewhere()
    assert {"coreweave/fp8", "relace/fp4", "nextbit/fp8",
            "baseten/fp8", "novita/fp8"} <= set(outsiders)


@pytest.mark.parametrize("outsider",
                         _tags_approved_only_elsewhere() + list(NEAR_MISSES))
def test_a_pin_naming_any_other_tag_is_refused(monkeypatch, outsider):
    assert outsider not in llm.APPROVED_OPENROUTER_PROVIDERS[MINIMAX]
    for pin in (outsider, f"together,{outsider}", f"{PIN},{outsider}"):
        with pytest.raises(RuntimeError) as excinfo:
            _approve(monkeypatch, model=MINIMAX, pin=pin)
        message = str(excinfo.value)
        assert "HA_OPENROUTER_PROVIDERS" in message, pin
        assert repr(outsider) in message, pin
        with pytest.raises(RuntimeError) as excinfo:
            _approve(monkeypatch, model=FLASH, pin=FLASH_PIN,
                     plan_model=MINIMAX, plan_pin=pin)
        assert "HA_PLAN_PROVIDERS" in str(excinfo.value), pin


def test_the_response_check_accepts_only_the_two_display_names(monkeypatch):
    monkeypatch.setattr(llm, "_announce", lambda *a, **k: None)
    for name in DISPLAY:
        llm._assert_openrouter_response_provider({"provider": name},
                                                 model=MINIMAX)
    foreign = [name for name, tags in llm.OPENROUTER_PROVIDER_TAGS.items()
               if not tags & APPROVED]
    assert {"CoreWeave", "Relace", "NextBit", "Novita", "BaseTen"} <= set(
        foreign)
    for name in foreign + ["MiniMax", "Parasail [fp8]", "together", ""]:
        with pytest.raises(RuntimeError, match="not approved for model"):
            llm._assert_openrouter_response_provider({"provider": name},
                                                     model=MINIMAX)


def test_the_request_body_is_the_engines_ordinary_one(monkeypatch):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ready"},
                         "finish_reason": "stop"}],
            "provider": "Parasail",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    _approve(monkeypatch, model=MINIMAX, pin=PIN)
    monkeypatch.setattr(llm, "OPENROUTER_MIN_THROUGHPUT", 0)
    monkeypatch.setattr(llm, "_TRANSPORT", httpx.MockTransport(handler))

    assert llm.complete("prompt text") == "ready"
    assert llm.last_complete_status()["outcome"] == "success"
    (body,) = bodies
    assert body["model"] == MINIMAX
    assert {key: body[key] for key in llm.GROUNDED_OPTS} == llm.GROUNDED_OPTS
    assert body["reasoning"] == {"effort": "low"}
    assert body["provider"] == {
        "order": ["together", "parasail/fp8"],
        "only": ["together", "parasail/fp8"],
        "allow_fallbacks": False, "require_parameters": True,
        "zdr": True, "data_collection": "deny"}
