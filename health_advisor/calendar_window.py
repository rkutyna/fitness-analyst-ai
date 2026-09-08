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

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}
_MONTH_NAMES = ("January|February|March|April|May|June|July|August|"
                "September|October|November|December|"
                "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
# Keep the token deliberately date-shaped. The parser, rather than the model,
# owns the year for a month/day with no year.
_DATE_TOKEN_RE = (rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|"
                  rf"(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:,?\s+\d{{4}})?)")
_WEEK_START_RE = re.compile(
    rf"\b(?:the\s+)?week\s+starting\s+(?P<date>{_DATE_TOKEN_RE})\b",
    re.IGNORECASE,
)
_DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_TOKEN_RE})\s+(?:through|to|-)\s+"
    rf"(?P<end>{_DATE_TOKEN_RE})",
    re.IGNORECASE,
)
_ABSOLUTE_DATE_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<date>{_DATE_TOKEN_RE})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _parse_date_token(token: str, today: date, *, year: int | None = None) -> date | None:
    """Parse one absolute token, choosing a deterministic past year if absent."""
    cleaned = " ".join(token.strip().rstrip("?,.! ").split())
    iso = re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned)
    if iso:
        try:
            return date.fromisoformat(cleaned)
        except ValueError:
            return None

    named = re.fullmatch(
        rf"(?P<month>{_MONTH_NAMES})\s+(?P<day>\d{{1,2}})"
        rf"(?:,?\s+(?P<year>\d{{4}}))?",
        cleaned,
        re.IGNORECASE,
    )
    if not named:
        return None
    month_name = named.group("month").lower()
    month = _MONTHS.get(month_name) or {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
        "nov": 11, "dec": 12,
    }.get(month_name)
    explicit_year = named.group("year")
    day = int(named.group("day"))
    chosen_year = int(explicit_year) if explicit_year else year
    if chosen_year is not None:
        try:
            return date(chosen_year, month, day)
        except ValueError:
            return None
    # A bare month/day means the nearest occurrence on or before today. This
    # makes "August 31" on 2026-09-06 resolve to 2026, not 2025, and handles
    # a non-leap current year by continuing to the nearest valid February 29.
    for candidate_year in range(today.year, 0, -1):
        try:
            candidate = date(candidate_year, month, day)
        except ValueError:
            continue
        if candidate <= today:
            return candidate
    return None


def _absolute_windows(text: str, today: date) -> list[tuple[int, int, CalendarWindow]]:
    """Find absolute date/range phrases, consuming nested date tokens once."""
    found: list[tuple[int, int, CalendarWindow]] = []
    occupied: list[tuple[int, int]] = []

    def available(start: int, end: int) -> bool:
        return not any(start < other_end and end > other_start
                       for other_start, other_end in occupied)

    for match in _WEEK_START_RE.finditer(text):
        if not available(match.start(), match.end()):
            continue
        start = _parse_date_token(match.group("date"), today)
        if start is None:
            continue
        window = CalendarWindow(
            start.isoformat(), (start + timedelta(days=6)).isoformat(),
            " ".join(match.group(0).split()), "week")
        found.append((match.start(), match.end(), window))
        occupied.append((match.start(), match.end()))

    for match in _DATE_RANGE_RE.finditer(text):
        if not available(match.start(), match.end()):
            continue
        start = _parse_date_token(match.group("start"), today)
        if start is None:
            continue
        # An unqualified range has one calendar year. In particular, the
        # future end of "Sep 1 through Sep 7" must not roll back separately.
        start_has_year = bool(re.search(r"\d{4}", match.group("start")))
        end = _parse_date_token(
            match.group("end"), today,
            year=start.year if not start_has_year else None)
        if end is None or end < start:
            continue
        window = CalendarWindow(
            start.isoformat(), end.isoformat(),
            " ".join(match.group(0).split()), None)
        found.append((match.start(), match.end(), window))
        occupied.append((match.start(), match.end()))

    for match in _ABSOLUTE_DATE_RE.finditer(text):
        if not available(match.start(), match.end()):
            continue
        start = _parse_date_token(match.group("date"), today)
        if start is None:
            continue
        window = CalendarWindow(start.isoformat(), start.isoformat(),
                                " ".join(match.group(0).split()), "day")
        found.append((match.start(), match.end(), window))
        occupied.append((match.start(), match.end()))
    return found


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
    matches_with_positions = [
        (match.start(), _window_for(match.group("phrase"), today))
        for match in _PHRASE_RE.finditer(text)
    ]
    matches_with_positions.extend(
        (start, window) for start, _end, window in _absolute_windows(text, today)
    )
    matches_with_positions.sort(key=lambda item: item[0])
    matches = tuple(window for _start, window in matches_with_positions)
    if not matches:
        return None
    return matches[0] if len(matches) == 1 else matches
