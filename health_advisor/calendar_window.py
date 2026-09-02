"""Resolve calendar phrases into explicit, inclusive date windows.

This module only parses the question and performs calendar arithmetic.  It
never consults the vault; the caller supplies the as-of date so Python owns
the window independently of model wording and data lag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from .analysis import _week_start


@dataclass(frozen=True, slots=True)
class CalendarWindow:
    """One calendar phrase and its inclusive ISO date window."""

    start: str
    end: str
    matched_phrase: str
    by_hint: str | None

    @property
    def phrase(self) -> str:
        """Compatibility spelling for callers that call it simply ``phrase``."""
        return self.matched_phrase


_PHRASE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<phrase>"
    r"past\s+[1-9]\d*\s+days?|this\s+week|last\s+week|"
    r"this\s+month|last\s+month|yesterday|today"
    r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _window_for(phrase: str, today: date) -> CalendarWindow:
    canonical = " ".join(phrase.lower().split())
    if canonical == "today":
        start = end = today
        by_hint = "day"
    elif canonical == "yesterday":
        start = end = today - timedelta(days=1)
        by_hint = "day"
    elif canonical in ("this week", "last week"):
        monday = date.fromisoformat(_week_start(today.isoformat()))
        if canonical == "last week":
            monday -= timedelta(days=7)
        start = monday
        end = (monday + timedelta(days=6)
               if canonical == "last week" else today)
        by_hint = "week"
    elif canonical in ("this month", "last month"):
        first = date(today.year, today.month, 1)
        if canonical == "last month":
            end = first - timedelta(days=1)
            start = date(end.year, end.month, 1)
        else:
            start, end = first, today
        by_hint = None
    else:
        days = int(canonical.split()[1])
        start = today - timedelta(days=days - 1)
        end = today
        by_hint = "day"
    return CalendarWindow(start.isoformat(), end.isoformat(), canonical, by_hint)


def resolve_window(
    text: str, today: date,
) -> CalendarWindow | tuple[CalendarWindow, ...] | None:
    """Resolve one or more supported calendar phrases in ``text``.

    A single match returns its :class:`CalendarWindow`; no match returns
    ``None``.  Multiple matches return all windows in question order so the
    caller can decline to override an ambiguous comparison.  ``today`` is
    injectable for tests; the ask path supplies its vault as-of date explicitly.
    """
    if not isinstance(text, str):
        return None
    matches = tuple(_window_for(match.group("phrase"), today)
                    for match in _PHRASE_RE.finditer(text))
    if not matches:
        return None
    return matches[0] if len(matches) == 1 else matches
