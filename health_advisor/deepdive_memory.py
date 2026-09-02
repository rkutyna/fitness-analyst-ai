"""Durable per-run scratchpad for the autonomous deep-dive: the researcher's memory +
incremental findings sink, living OUTSIDE the model's context window so a long
unsupervised run can be compacted without losing what it learned. Inspectable and
crash-resumable (atomic writes). Designed as the substrate for a future 'blackboard'
loop. The run-scoped memory tools (Task 2) are injected ONLY into llm.research_loop —
never registered on mcp_server (the interactive gateway must not write findings)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scratch_path(trace_dir: str, day: str) -> str:
    return os.path.join(trace_dir, f"deepdive_{day}_scratch.json")


def _save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)  # atomic


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def new_scratchpad(path: str, as_of: str, tasks: list[dict]) -> dict:
    data = {
        "as_of": as_of,
        "tasks": [{"id": t["id"], "question": t.get("question", ""), "status": "open"}
                  for t in tasks],
        "findings": [], "notes": [], "log": [],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _save(path, data)
    return data


def append_finding(path, task_id, claim, numbers=None, tools_used=None,
                   confidence=None) -> dict:
    data = load(path)
    # Dedup on (task_id, claim): a finalize/compaction nudge can prompt a model that
    # already recorded a finding to record it again — don't inflate the board (it would
    # be double-scored downstream by the judge/filter). First record wins.
    if not any(f["task_id"] == task_id and f["claim"] == claim
               for f in data["findings"]):
        data["findings"].append({"task_id": task_id, "claim": claim,
                                 "numbers": numbers or [], "tools_used": tools_used or [],
                                 "confidence": confidence, "ts": _now()})
        _save(path, data)
    return {"ok": True, "n_findings": len(data["findings"])}


def append_note(path, task_id, text) -> dict:
    data = load(path)
    data["notes"].append({"task_id": task_id, "text": text, "ts": _now()})
    _save(path, data)
    return {"ok": True}


def append_log(path, task_id, turn, event, detail) -> None:
    data = load(path)
    data["log"].append({"turn": turn, "task_id": task_id, "event": event,
                        "detail": str(detail)[:500], "ts": _now()})
    _save(path, data)


def compact_state(scratch: dict, task_id) -> str:
    lines: list[str] = []
    task = next((t for t in scratch.get("tasks", []) if t["id"] == task_id), None)
    if task:
        lines.append(f"QUESTION (task {task_id}): {task.get('question', '')}")
    fnd = [f for f in scratch.get("findings", []) if f["task_id"] == task_id]
    if fnd:
        lines.append("Findings recorded so far:")
        for f in fnd:
            nums = ", ".join(f"{n.get('metric', '')} {n.get('field', '')}={n.get('value')}"
                             for n in f.get("numbers", []))
            lines.append(f"- {f['claim']}" + (f" [{nums}]" if nums else ""))
    else:
        lines.append("No findings recorded yet.")
    notes = [n for n in scratch.get("notes", []) if n["task_id"] == task_id]
    if notes:
        lines.append("Notes:")
        lines += [f"- {n['text']}" for n in notes]
    return "\n".join(lines)


_RECORD_SCHEMA = {"type": "function", "function": {
    "name": "record_finding",
    "description": ("Record ONE confirmed finding to your durable notepad the moment you "
                    "verify it with a real number from a tool. Call this as you go; you "
                    "may call it several times. 'numbers' is a list of scoped "
                    "objects: raw figures use {metric,period,field,value}; a "
                    "derived figure additionally names {operation,operands}, "
                    "with every operand itself {metric,period,field}."),
    "parameters": {"type": "object", "properties": {
        "claim": {"type": "string"},
        "numbers": {"type": "array", "items": {"type": "object",
            "properties": {
                "metric": {"type": "string"},
                "period": {},
                "field": {"type": "string"},
                "value": {"type": "number"},
                "operation": {"type": "string"},
                "operands": {"type": "array", "items": {"type": "object"}},
            }}},
        "tools_used": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"}},
        "required": ["claim"]}}}

_NOTE_SCHEMA = {"type": "function", "function": {
    "name": "note",
    "description": "Jot a short free-form note (e.g. something to check next) to your notepad.",
    "parameters": {"type": "object",
                   "properties": {"text": {"type": "string"}}, "required": ["text"]}}}

_READ_STATE_SCHEMA = {"type": "function", "function": {
    "name": "read_state",
    "description": ("Recall what you have already found for this question — your earlier "
                    "messages may have been compacted away."),
    "parameters": {"type": "object", "properties": {}}}}


def build_memory_tools(path: str, task_id) -> dict:
    """Run-scoped {name: (callable, ollama_schema)} bound to one run's scratchpad +
    task. Injected into llm.research_loop ONLY (never registered on mcp_server)."""
    def record_finding(claim, numbers=None, tools_used=None, confidence=None):
        return append_finding(path, task_id, claim, numbers, tools_used, confidence)

    def note(text):
        return append_note(path, task_id, text)

    def read_state():
        return {"task_id": task_id, "state": compact_state(load(path), task_id)}

    return {"record_finding": (record_finding, _RECORD_SCHEMA),
            "note": (note, _NOTE_SCHEMA),
            "read_state": (read_state, _READ_STATE_SCHEMA)}
