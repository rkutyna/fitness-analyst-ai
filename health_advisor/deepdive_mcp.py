"""Provider-facing MCP server for codex-run deep dives and coach chat.

By default it exposes only the read-only research tools
(``llm.RESEARCHER_TOOLS``); an explicit include set supplies the broader coach
surface. When a scratchpad is given it also exposes run-scoped notepad tools.
Every exposed tool
is registered on this server, with an optional append-only JSONL call ledger
around the callable.  The ledger is written here, where the real tool result
exists; it is never assembled from model output.  The vault remains read-only
and the ledger is a separate artifact.

Run: ``python deepdive_mcp_launch.py --vault PATH [--scratch PATH
--task-id ID] [--ledger PATH]`` (stdio; spawned by ``codex exec``).
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

from . import deepdive_memory as mem
from . import llm
from . import mcp_server as S
from .context import VaultContext


LEDGER_RESULT_MAX_BYTES = 256 * 1024
"""Maximum serialized result retained inline in one ledger record."""


class _CallLedger:
    """Append one JSON object per call and keep sequence numbers monotonic."""

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self.window_override = self._read_window_override()
        self._next_sequence = self._read_next_sequence()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def _read_window_override(self):
        """Read the ask window sidecar, which may be made by another process."""
        try:
            with open(self.path + ".window_override.json", encoding="utf-8") as fh:
                value = json.load(fh)
            return value if isinstance(value, dict) else None
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def _read_next_sequence(self) -> int:
        """Continue a ledger if a caller deliberately reuses its run path."""
        try:
            with open(self.path, encoding="utf-8") as fh:
                sequences = []
                for line in fh:
                    try:
                        value = json.loads(line).get("sequence")
                    except (json.JSONDecodeError, AttributeError):
                        continue
                    if isinstance(value, int) and not isinstance(value, bool):
                        sequences.append(value)
                return max(sequences, default=0) + 1
        except FileNotFoundError:
            return 1

    @staticmethod
    def _json_value(value):
        """Make an MCP value safe to persist without changing normal JSON."""
        try:
            json.dumps(value, allow_nan=False)
            return value
        except (TypeError, ValueError):
            return {"_unserializable_type": type(value).__name__,
                    "_repr": repr(value)}

    def _result_value(self, result):
        safe = self._json_value(result)
        encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        if len(encoded) <= LEDGER_RESULT_MAX_BYTES:
            return safe, False, len(encoded)
        return {
            "_elided": True,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "preview": encoded[:4096].decode("utf-8", errors="replace"),
        }, True, len(encoded)

    def append(self, tool_name: str, arguments: dict, result,
               *, window_override: dict | None = None) -> int:
        sequence = self._next_sequence
        result_value, result_elided, result_bytes = self._result_value(result)
        entry = {
            "sequence": sequence,
            "tool_name": tool_name,
            "arguments": self._json_value(arguments),
            "result": result_value,
            "result_elided": result_elided,
            "result_bytes": result_bytes,
        }
        if window_override is not None:
            entry["window_override"] = self._json_value(window_override)
        with open(self.path, "a", encoding="utf-8") as fh:
            json.dump(entry, fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._next_sequence += 1
        return sequence


def _call_arguments(fn, args, kwargs) -> dict:
    """Represent positional calls using the tool's named parameters."""
    bound = inspect.signature(fn).bind(*args, **kwargs)
    return dict(bound.arguments)


def _claim_period_vocabulary(result) -> list[dict]:
    """Publish the exact display periods the researcher may copy into claims.

    Block payloads intentionally store ``end`` as the start of the last week.
    That is useful for recomputation but ambiguous in prose, where the same
    block is naturally written through the end of that week. Publish that
    display spelling beside the ledger sequence so the model does not invent
    either interpretation.
    """
    vocabulary = []

    def walk(node, inherited_metric=None):
        if isinstance(node, dict):
            period = node.get("period")
            starts = period.get("period_starts") if isinstance(period, dict) else None
            metric = node.get("metric", inherited_metric)
            if metric and isinstance(starts, list) and starts:
                try:
                    # Span the last bucket by the spacing the payload itself
                    # shows, never by an assumed week. Measured 2026-08-24: a
                    # hardcoded +6 published '2026-06-29:2026-07-08' for a
                    # day-spaced block whose period ended 2026-07-02 -- six days
                    # long, and the model is told to copy this string verbatim,
                    # so the gate would have ACCEPTED a claim whose period
                    # overstated its own window. A verified figure with a wrong
                    # scope is the failure this whole channel exists to stop.
                    first = date.fromisoformat(str(starts[0]))
                    last = date.fromisoformat(str(starts[-1]))
                    if len(starts) > 1:
                        step = (date.fromisoformat(str(starts[1])) - first).days
                    else:
                        step = 0
                    if step > 0:
                        last_end = (last + timedelta(days=step - 1)).isoformat()
                    elif isinstance(period.get("end"), str):
                        # One bucket: the payload's own end is the only honest
                        # answer, and it is already inclusive for a single row.
                        last_end = period["end"]
                    else:
                        last_end = last.isoformat()
                    item = {"metric": str(metric),
                            "claim_period": f"{starts[0]}:{last_end}",
                            "ledger_period": period}
                    if item not in vocabulary:
                        vocabulary.append(item)
                except (TypeError, ValueError):
                    pass
            elif (metric and isinstance(period, str)
                  and "week_start" in node):
                item = {"metric": str(metric),
                        "claim_period": period,
                        "ledger_period": period}
                if item not in vocabulary:
                    vocabulary.append(item)
            for child in node.values():
                walk(child, metric)
        elif isinstance(node, list):
            for child in node:
                walk(child, inherited_metric)

    walk(result)
    return vocabulary


def _ledger_wrapper(tool_name: str, fn, ledger: _CallLedger | None):
    if ledger is None:
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        signature = inspect.signature(fn)
        partial = signature.bind_partial(*args, **kwargs)
        model_arguments = dict(partial.arguments)
        effective_arguments = dict(model_arguments)
        override = _window_override_for_call(ledger, signature,
                                             model_arguments, effective_arguments)
        arguments = _call_arguments(fn, (), effective_arguments)
        try:
            result = fn(**effective_arguments)
        except Exception as exc:
            # A failed call is still a call observed by the server.  Recording
            # it makes a missing result explicit without changing MCP errors.
            ledger.append(tool_name, arguments, {
                "_error_type": type(exc).__name__,
                "_error": str(exc),
            }, window_override=override)
            raise
        sequence = ledger.append(tool_name, arguments, result,
                                 window_override=override)
        # Keep the evidence record exactly as the server observed it, while
        # publishing its citation key to the model.  The private key is not a
        # result field and therefore cannot accidentally become claim evidence.
        if isinstance(result, dict):
            surfaced = dict(result)
        else:
            surfaced = {"result": result}
        ledger_info = {"sequence": sequence}
        period_vocabulary = _claim_period_vocabulary(result)
        if period_vocabulary:
            ledger_info["period_vocabulary"] = period_vocabulary
        surfaced["_ledger"] = ledger_info
        return surfaced

    wrapped.__signature__ = inspect.signature(fn)
    return wrapped


def _window_override_for_call(ledger: _CallLedger, signature, model_arguments,
                              effective_arguments) -> dict | None:
    """Apply the chat's single resolved window to one eligible tool call."""
    config = ledger.window_override
    if config is None:
        return None
    if config.get("status") != "single":
        return {"applied": False, "reason": config.get("reason", "no_override"),
                **({"phrases": config["phrases"]}
                   if isinstance(config.get("phrases"), list) else {})}
    window = config.get("window")
    if (not isinstance(window, dict)
            or not {"start", "end"} <= set(window)):
        return {"applied": False, "reason": "invalid_window_override"}
    parameters = signature.parameters
    if "start" not in parameters or "end" not in parameters:
        return {"applied": False, "reason": "tool_has_no_start_end",
                "phrase": window.get("matched_phrase")}
    model_sent = {name: model_arguments.get(name)
                  for name in ("start", "end")
                  if name in model_arguments}
    if "by" in model_arguments:
        model_sent["by"] = model_arguments["by"]
    effective_arguments["start"] = window["start"]
    effective_arguments["end"] = window["end"]
    if window.get("by_hint") == "week" and "by" in parameters:
        effective_arguments["by"] = "week"
    applied = {"start": effective_arguments["start"],
               "end": effective_arguments["end"]}
    if "by" in effective_arguments:
        applied["by"] = effective_arguments["by"]
    return {"phrase": window.get("matched_phrase"),
            "model_sent_window": model_sent,
            "applied_window": applied}


def build_server(ctx: VaultContext, *, scratch_path: str = "",
                 task_id: int | str = 1, ledger_path: str = "", include=None) -> FastMCP:
    """Build one provider-facing, optionally ledgered tool server.

    ``include=None`` retains the researcher's deliberately small default. A
    caller such as interactive coach chat supplies its own read-only surface;
    it must not silently inherit the researcher's narrower question set.
    """
    if ledger_path and os.path.realpath(os.fspath(ledger_path)) == os.path.realpath(
            os.fspath(ctx.db_path)):
        raise ValueError("the tool-call ledger must be separate from the vault")
    # Same narrowing as llm._registry: this server's answers reach a provider.
    # Build from plain bound callables so the ledger wraps the actual tool
    # result, rather than observing a model-side representation of the call.
    mcp = FastMCP("health-deepdive")
    ledger = _CallLedger(ledger_path) if ledger_path else None
    selected = frozenset(llm.RESEARCHER_TOOLS if include is None else include)
    for tool_name, bound in S.build_tools(ctx.provider_facing()).items():
        if tool_name in selected:
            mcp.tool(name=tool_name)(_ledger_wrapper(tool_name, bound, ledger))

    if not scratch_path:
        return mcp

    def record_finding(claim: str, numbers: list[dict] | None = None,
                       tools_used: list[str] | None = None,
                       confidence: float | None = None) -> dict:
        """Record ONE confirmed finding to your durable notepad the moment you
        verify it with a real number from a tool. Call this as you go; you may
        call it several times. 'numbers' uses the shared scoped claim shape
        {metric,period,field,value}; researcher claims additionally include
        source {sequence,path}. 'tools_used' is the researcher's self-report,
        not verification: the append-only call ledger is the measurement."""
        return mem.append_finding(scratch_path, task_id, claim, numbers,
                                  tools_used, confidence)
    mcp.tool()(_ledger_wrapper("record_finding", record_finding, ledger))

    def note(text: str) -> dict:
        """Jot a short free-form note (e.g. something to check next) to your
        notepad."""
        return mem.append_note(scratch_path, task_id, text)
    mcp.tool()(_ledger_wrapper("note", note, ledger))

    def read_state() -> dict:
        """Recall what you have already found for this question — your earlier
        messages may have been compacted away."""
        return {"task_id": task_id,
                "state": mem.compact_state(mem.load(scratch_path), task_id)}
    mcp.tool()(_ledger_wrapper("read_state", read_state, ledger))

    return mcp


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="deepdive_mcp_launch.py")
    ap.add_argument("--vault", "--db", dest="vault", required=True,
                    help="path to the vault this research run reads")
    ap.add_argument("--user", default="local", help="user id this vault belongs to")
    ap.add_argument("--scratch", default="", help="run-scoped notepad path")
    ap.add_argument("--task-id", dest="task_id", default="1")
    ap.add_argument("--ledger", default="", help="run-scoped tool-call ledger path")
    ap.add_argument("--include", default="",
                    help="comma-separated tool names (default: researcher surface)")
    args = ap.parse_args(argv)
    task_id = int(args.task_id) if str(args.task_id).isdigit() else args.task_id
    ctx = VaultContext.local(args.vault, user_id=args.user)
    include = (frozenset(name for name in args.include.split(",") if name)
               if args.include else None)
    build_server(ctx, scratch_path=args.scratch, task_id=task_id,
                 ledger_path=args.ledger, include=include).run(transport="stdio")
    return 0
