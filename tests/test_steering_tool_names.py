# tests/test_steering_tool_names.py
"""The internal-tool-name rule must cover every tool the model is told about.

``steering.tool_names()`` is the set of names a narration may not repeat. It
used to miss ``submit_answer`` (the schema names it through a constant, not a
literal) and it silently dropped every name when the module import failed.
"""
import logging
import sys

import pytest

import health_advisor
from health_advisor import llm, steering

# The tool names the guard must still know when the live enumeration is
# unavailable. A test below pins this equal to the live set, so it cannot drift.
KNOWN = "get_daily_series"


@pytest.fixture(autouse=True)
def _no_host_tools(monkeypatch):
    monkeypatch.setattr(llm, "_HOST_TOOLS", {})


def _break_enumeration(monkeypatch):
    """Make ``from . import llm, mcp_server`` raise ImportError."""
    monkeypatch.delattr(health_advisor, "llm")
    monkeypatch.setitem(sys.modules, "health_advisor.llm", None)
    monkeypatch.setattr(steering, "_FALLBACK_WARNED", False, raising=False)


def test_submit_answer_is_an_internal_tool_name():
    assert llm.SUBMIT_ANSWER_NAME in steering.tool_names()
    assert steering.leak(
        "I will call submit_answer now to finish the answer."
    ) == "internal tool name"


def test_fallback_set_still_guards_when_the_enumeration_import_fails(
        monkeypatch, caplog):
    _break_enumeration(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=steering.__name__):
        names = steering.tool_names()
        steering.tool_names()          # a second call must not warn again
    assert KNOWN in names
    assert "submit_answer" in names
    assert steering.leak(f"Next I would use {KNOWN} for you.") == (
        "internal tool name")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1




def test_fallback_set_equals_the_live_set():
    """The static fallback cannot drift from what the live import yields."""
    assert frozenset(steering._FALLBACK_TOOL_NAMES) == steering.tool_names()
    assert all("_" in name for name in steering._FALLBACK_TOOL_NAMES)
