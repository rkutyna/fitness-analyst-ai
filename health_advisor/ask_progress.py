"""In-memory progress narration for long-running coach questions.

Only Python-owned phrases and timestamps live here.  Tool arguments, results,
questions, and answer text deliberately never enter the registry.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import threading
import time
from typing import Callable


PROGRESS_TTL_SECONDS = 15 * 60
PROGRESS_MAX_ENTRIES = 64
GENERIC_PROGRESS_PHRASE = "gathering another figure…"


# Keep this map explicit and Python-owned.  The names are the complete tool
# surface in health_advisor.mcp_server; phrases describe the work, never its
# result.
TOOL_PROGRESS_PHRASES = {
    "mark_workout_not_a_session": "checking your workout label…",
    "list_available_metrics": "checking which measures you track…",
    "get_daily_series": "reading your daily trend…",
    "summarize_metric": "summarizing your measure…",
    "compare_periods": "comparing your two periods…",
    "get_intraday": "reading your within-day pattern…",
    "get_hr_zones": "checking your heart-rate zones…",
    "list_workouts": "reading your workouts…",
    "get_workout_segments": "checking your workout segments…",
    "get_impact_volume": "reading your training volume…",
    "get_sleep_regularity": "checking your sleep regularity…",
    "get_training_load_detail": "checking your training load…",
    "get_run_form": "checking your running form…",
    "get_briefing": "reading your daily briefing…",
    "get_latest": "checking your latest measure…",
    "correlate_metrics": "comparing your measures…",
    "scan_correlations": "looking for patterns in your measures…",
    "write_insight": "saving your coaching note…",
    "log_subjective": "recording your check-in…",
    "get_subjective": "reading your check-ins…",
    "food_lookup": "checking your food entry…",
    "food_catalog_add": "saving your food entry…",
    "food_meal_total": "adding your meal items…",
    "get_ingest_diagnostics": "checking your sync status…",
    "get_weekly_series": "reading your weekly trend…",
    "get_block_comparison": "comparing your training blocks…",
    "get_block_structure": "checking your workout structure…",
    "get_weekly_readiness": "checking your weekly readiness…",
    "record_benchmark": "saving your benchmark…",
    "get_benchmark_series": "reading your benchmark trend…",
    "get_monthly_running_power": "reading your monthly running power…",
    "log_manual_jog_minutes": "recording your jog minutes…",
}


def progress_phrase(tool_name: str) -> str:
    """Return the fixed narration for a tool name."""
    return TOOL_PROGRESS_PHRASES.get(tool_name, GENERIC_PROGRESS_PHRASE)


def _timestamp(timestamp_fn: Callable[[], float]) -> str:
    return (datetime.fromtimestamp(timestamp_fn(), timezone.utc)
            .isoformat(timespec="milliseconds").replace("+00:00", "Z"))


class ProgressRegistry:
    """A bounded, expiring, thread-safe registry of public progress entries."""

    def __init__(self, *, ttl_seconds: float = PROGRESS_TTL_SECONDS,
                 max_entries: int = PROGRESS_MAX_ENTRIES,
                 clock: Callable[[], float] | None = None,
                 timestamp_fn: Callable[[], float] | None = None):
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._clock = clock or time.monotonic
        self._timestamp_fn = timestamp_fn or time.time
        self._entries: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, (expires_at, _) in self._entries.items()
                   if expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)

    @staticmethod
    def _public(entry: dict) -> dict:
        return {
            "state": entry["state"],
            "started_at": entry["started_at"],
            "steps": [dict(step) for step in entry["steps"]],
            "finished_at": entry["finished_at"],
        }

    def start(self, progress_id: str) -> dict:
        """Start or replace one client-supplied progress id."""
        now = self._clock()
        entry = {
            "state": "running",
            "started_at": _timestamp(self._timestamp_fn),
            "steps": [],
            "finished_at": None,
        }
        with self._lock:
            self._purge_expired(now)
            self._entries.pop(progress_id, None)
            self._entries[progress_id] = (now + self.ttl_seconds, entry)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return self._public(entry)

    def add_step(self, progress_id: str, sequence: int, tool_name: str) -> None:
        """Record one tool step while its turn is running."""
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            stored = self._entries.get(progress_id)
            if stored is None:
                return
            _, entry = stored
            if entry["state"] != "running":
                return
            entry["steps"].append({
                "seq": sequence,
                "phrase": progress_phrase(tool_name),
                "at": _timestamp(self._timestamp_fn),
            })

    def finish(self, progress_id: str, state: str) -> None:
        """Finish a turn without changing its recorded steps."""
        if state not in {"done", "fallback", "error"}:
            raise ValueError(f"invalid progress state: {state!r}")
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            stored = self._entries.get(progress_id)
            if stored is None:
                return
            _, entry = stored
            entry["state"] = state
            entry["finished_at"] = _timestamp(self._timestamp_fn)

    def get(self, progress_id: str) -> dict | None:
        """Return a detached public entry, or ``None`` if unknown/expired."""
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            stored = self._entries.get(progress_id)
            return None if stored is None else self._public(stored[1])

    def clear(self) -> None:
        """Clear entries; intended for process-local test isolation."""
        with self._lock:
            self._entries.clear()
