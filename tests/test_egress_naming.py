"""A grep for egress returns one kind of answer (#92, Done-when 3).

There used to be two unrelated "hermes" in this package: `agents.run_hermes`,
the **LLM seam**, and `checkin.HERMES_BIN`, the **Telegram CLI**. Anyone
auditing egress by grepping got both and had to know which was which — and
anyone who assumed `run_hermes` was the Telegram path audited the wrong thing
and concluded the wrong answer.

The LLM seam was the one renamed, because the asymmetry decides it: `HERMES_BIN`
names a binary genuinely called `hermes`, while `run_hermes`'s own docstring
admitted the name was "kept so the existing monkeypatch-based tests keep
working". This test holds that line — it is a naming invariant, so it is checked
against module symbols rather than by running anything.
"""
from __future__ import annotations

import pkgutil

import pytest

import health_advisor


def _module_names(module) -> set[str]:
    return {name for name in vars(module) if not name.startswith("__")}


def test_the_llm_seam_is_not_named_after_the_telegram_gateway():
    from health_advisor import agents
    assert hasattr(agents, "run_model"), "the model entrypoint is agents.run_model"
    assert not hasattr(agents, "run_hermes"), (
        "run_hermes is the Telegram gateway's name; the model entrypoint is "
        "run_model (#92). A compatibility alias defeats the purpose — it leaves "
        "both names greppable, which is the defect.")


def test_only_the_telegram_module_carries_hermes_symbols():
    """Any module gaining a `hermes` symbol is either the gateway or a mistake."""
    offenders: dict[str, set[str]] = {}
    for info in pkgutil.iter_modules(health_advisor.__path__):
        if info.name == "checkin":          # the Telegram gateway itself
            continue
        try:
            module = __import__(f"health_advisor.{info.name}", fromlist=["_"])
        except Exception:                    # optional deps are not this test's business
            continue
        named = {n for n in _module_names(module) if "hermes" in n.lower()}
        if named:
            offenders[info.name] = named
    assert not offenders, (
        f"modules other than checkin now define hermes-named symbols: {offenders}. "
        "Egress auditing depends on that name meaning exactly one thing.")
