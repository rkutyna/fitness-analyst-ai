"""Calendar phrase resolution and Python-owned ask-window enforcement."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from health_advisor import chat
from health_advisor import deepdive_mcp as D
from health_advisor import vault as vaultmod
from health_advisor.calendar_window import CalendarWindow, resolve_window
from tests.conftest import seed_metric


@pytest.mark.parametrize("today", [date(2026, 8, 31), date(2026, 9, 6),
                                    date(2026, 9, 1)])
def test_required_calendar_phrases_are_monday_anchored(today):
    expected = {
        "this week": ("2026-08-31", today.isoformat(), "week"),
        "last week": ("2026-08-24", "2026-08-30", "week"),
        "today": (today.isoformat(), today.isoformat(), "day"),
        "yesterday": ((today - timedelta(days=1)).isoformat(),
                      (today - timedelta(days=1)).isoformat(), "day"),
        "this month": ("2026-09-01", today.isoformat(), None),
        "last month": ("2026-08-01", "2026-08-31", None),
    }
    # The August Monday case crosses the month boundary; keep the expected
    # week dates explicit for the three required as-of dates.
    if today == date(2026, 8, 31):
        expected["this month"] = ("2026-08-01", today.isoformat(), None)
        expected["last month"] = ("2026-07-01", "2026-07-31", None)
    for phrase, (start, end, by_hint) in expected.items():
        window = resolve_window("How much did I do " + phrase + "?", today)
        assert isinstance(window, CalendarWindow)
        assert (window.start, window.end, window.by_hint) == (start, end, by_hint)
        assert window.matched_phrase == phrase


def test_multiple_calendar_phrases_return_all_matches_without_collapsing_them():
    matches = resolve_window("this week versus last week", date(2026, 9, 1))

    assert isinstance(matches, tuple)
    assert [window.matched_phrase for window in matches] == [
        "this week", "last week"]


def test_past_days_is_a_day_window():
    window = resolve_window("my past 5 days", date(2026, 9, 1))

    assert window == CalendarWindow("2026-08-28", "2026-09-01",
                                    "past 5 days", "day")


def test_chat_as_of_comes_from_daily_metrics_not_vault_timezone(vault, conn):
    seed_metric(conn, "step_count", "2026-08-14", [1, 2])
    vaultmod.set_local_timezone(conn, "Pacific/Honolulu")
    conn.commit()

    assert chat._ask_calendar_today(vault, None) == date(2026, 8, 15)


def _make_wrapped_registry(sidecar, ledger_path, *, multiple=False):
    config, _ = chat._calendar_window_config(
        None, "How many jog minutes this week?" if not multiple
        else "this week versus last week", "2026-09-01")
    chat._write_window_override(str(sidecar), config)
    ledger = D._CallLedger(str(ledger_path))
    seen = []

    def get_impact_volume(start, end, by="week"):
        seen.append({"start": start, "end": end, "by": by})
        return {"periods": []}

    registry = {"get_impact_volume": D._ledger_wrapper(
        "get_impact_volume", get_impact_volume, ledger)}
    return registry, seen


def test_single_phrase_fake_registry_overrides_window_and_records_ledger(
        tmp_path):
    ledger_path = tmp_path / "calls.jsonl"
    registry, seen = _make_wrapped_registry(
        tmp_path / "calls.jsonl.window_override.json", ledger_path)

    registry["get_impact_volume"](
        start="2026-08-01", end="2026-08-31", by="day")

    assert seen == [{"start": "2026-08-31", "end": "2026-09-01",
                     "by": "week"}]
    record = json.loads(ledger_path.read_text().strip())
    assert record["arguments"] == {"start": "2026-08-31",
                                    "end": "2026-09-01", "by": "week"}
    assert record["window_override"] == {
        "phrase": "this week",
        "model_sent_window": {"start": "2026-08-01", "end": "2026-08-31",
                               "by": "day"},
        "applied_window": {"start": "2026-08-31", "end": "2026-09-01",
                            "by": "week"},
    }


def test_two_phrases_fake_registry_does_not_override_and_ledger_says_so(tmp_path):
    ledger_path = tmp_path / "calls.jsonl"
    registry, seen = _make_wrapped_registry(
        tmp_path / "calls.jsonl.window_override.json", ledger_path,
        multiple=True)

    registry["get_impact_volume"](
        start="2026-08-01", end="2026-08-31", by="day")

    assert seen == [{"start": "2026-08-01", "end": "2026-08-31", "by": "day"}]
    record = json.loads(ledger_path.read_text().strip())
    assert record["window_override"] == {
        "applied": False,
        "reason": "multiple_calendar_phrases",
        "phrases": ["this week", "last week"],
    }
