"""The vault's two local-date horizons.

These are deliberately different measurements, not competing implementations
of one number:

``gradeable_through`` reads ``workouts.local_date`` and answers whether the
vault has workout evidence far enough out to grade a session.
``known_through`` reads ``daily_metrics.date`` and answers whether any daily
metric has arrived for a day.

The module owns both SQL queries and the vocabulary for the gap between them.
Callers supply an already-open connection; no database path or ambient vault
is available here.
"""
from __future__ import annotations

import sqlite3
from datetime import date


FUTURE_DAY = "future_day"
SESSION_NOT_RECORDED = "session_not_recorded"


def _date(value: date | str | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


def gradeable_through(conn: sqlite3.Connection) -> date | None:
    """Return the latest local date for which workout evidence exists."""
    row = conn.execute("SELECT MAX(local_date) FROM workouts").fetchone()
    return _date(row[0]) if row and row[0] is not None else None


def known_through(conn: sqlite3.Connection) -> date | None:
    """Return the latest local date represented by any daily metric."""
    row = conn.execute("SELECT MAX(date) FROM daily_metrics").fetchone()
    return _date(row[0]) if row and row[0] is not None else None


def state_for_day(
    day: date | str,
    *,
    gradeable: date | str | None,
    known: date | str | None,
) -> str | None:
    """Classify only the two horizon states; ``None`` is gradeable territory.

    The caller still applies plan precedence (for example, a rest day wins
    before this classification) and can choose the appropriate response shape.
    """
    target = _date(day)
    known_date = _date(known)
    gradeable_date = _date(gradeable)
    if known_date is None or target > known_date:
        return FUTURE_DAY
    if gradeable_date is None or target > gradeable_date:
        return SESSION_NOT_RECORDED
    return None


__all__ = [
    "FUTURE_DAY", "SESSION_NOT_RECORDED", "gradeable_through",
    "known_through", "state_for_day",
]
