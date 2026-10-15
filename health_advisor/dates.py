"""Leaf date and clock helpers shared across the engine.

Imports only the standard library, so ``metrics`` (which ``analysis`` imports)
can use the same week grouping without an import cycle.
"""
from __future__ import annotations

import time
from datetime import date, timedelta


def parse_date(value, name: str = "local_date") -> date:
    """Parse a strict YYYY-MM-DD string; the error names the field."""
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be YYYY-MM-DD") from None


def week_start(day: date) -> date:
    """The Monday that starts the week containing ``day``."""
    return day - timedelta(days=day.weekday())


def utc_now() -> str:
    """The current UTC instant as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
