"""health_advisor/analyst.py -- the analyst-mode CLI entry point.

    ./.venv/bin/python -m health_advisor.analyst \\
        --vault path/to/health.db --question "did a long session produce
        elevated resting heart rate the next day?"

Analyst mode lets a language model write Python that runs sandboxed against
the user's health vault and returns a validated table with provenance. It
exists because the curated tools cannot answer novel questions -- ones that
need a join or a shape none of the fixed tools compute.

This module is the ONLY place those four building blocks meet:

- ``analyst_prompt`` builds the model's instructions from a live schema
  summary and the canonical metric/unit vocabulary.
- ``llm.complete`` gets one turn of model-written Python back.
- ``analyst_sandbox.default_executor()`` (via ``analyst_runner.run_analyst_code``)
  runs that code against a read-only vault connection, confined, timed, and
  bounded.
- ``analyst_envelope`` validates whatever the sandboxed run wrote to fd 3 into
  either a typed ``Envelope`` or a typed ``Refusal`` -- nothing in between is
  trusted.

THE ONE RULE THIS MODULE EXISTS TO HONOUR: "Python owns the truth. The model
is only ever a text transformer." There is deliberately NO second model turn
that narrates the result. Every number a user sees here is a cell out of a
validated ``Envelope``, rendered by this module's own Python into a plain
aligned text table. The human reader -- or a Claude session running the
weekly review -- does the narrating; this tool only supplies the numbers and
where they came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from health_advisor import db as dbmod
from health_advisor import llm
from health_advisor.analyst_envelope import Envelope, Refusal
from health_advisor.analyst_prompt import (
    REFUSAL_REMEDIATION,
    build_analyst_prompt,
    schema_summary,
)
from health_advisor import analyst_envelope
from health_advisor import analyst_sandbox
from health_advisor.analyst_sandbox import RunLimits, default_executor

__all__ = ["main", "run_analyst", "build_arg_parser"]


# --------------------------------------------------------------------------- #
# analyst_runner.py is being built concurrently elsewhere in this project and
# may not exist in a given checkout yet. This Protocol documents the contract
# this module codes against (its parent-side entry point,
# ``run_analyst_code(code, vault_path, run_dir, executor, *, limits) ->
# Envelope | Refusal``) WITHOUT importing the module at load time, so
# `python -m health_advisor.analyst --help` and every test that injects its
# own `run_code_fn` work whether or not the module exists on disk. The real
# import happens lazily, inside `_load_run_analyst_code`, only when a live run
# actually needs it.
# --------------------------------------------------------------------------- #
class _AnalystRunner(Protocol):
    def __call__(self, code: str, vault_path: str, run_dir: str, executor,
                 *, limits: RunLimits | None = None) -> "Envelope | Refusal":
        ...


def _load_run_analyst_code() -> _AnalystRunner:
    """Import ``health_advisor.analyst_runner.run_analyst_code`` lazily."""
    from health_advisor.analyst_runner import run_analyst_code  # noqa: F401 -- may not exist yet
    return run_analyst_code


# --------------------------------------------------------------------------- #
# Backend approval -- analyst mode is the widest provider-facing surface in
# the product (a model-written program runs against the vault), so it must
# not be the one path that opts out of the D15 destination check, and it adds
# a stricter LOCAL rule on top: codex is approved under D15/D17 for narration,
# but analyst mode refuses it by name regardless.
# --------------------------------------------------------------------------- #
CODEX_REFUSAL_MESSAGE = (
    "analyst mode refuses the 'codex' backend by name. Analyst mode "
    "is the widest provider-facing surface in this product -- it hands a "
    "model direct code-generation power over the vault, not just narration "
    "-- and must not be the one path that opts out of the D15 destination "
    "check. Set HA_LLM_BACKEND=openrouter, pin "
    "HA_OPENROUTER_PROVIDERS=coreweave/fp8, and set HA_OPENROUTER_REASONING, "
    "then rerun."
)


def assert_analyst_backend_approved() -> None:
    """Approve the backend before anything else. Raises on any refusal.

    Calls the shared D15 gate first -- so an unapproved backend name, an
    unapproved OpenRouter provider, or an unapproved endpoint host are all
    refused exactly as they are everywhere else in the product -- and then
    adds the one rule specific to this module: 'codex' is never acceptable
    here, even though D15/D17 approve it for narration elsewhere.
    """
    llm.assert_backend_approved()
    if llm.BACKEND == "codex":
        raise RuntimeError(CODEX_REFUSAL_MESSAGE)


# --------------------------------------------------------------------------- #
# Prompt caps -- analyst_prompt.build_analyst_prompt wants its caps under
# different names than analyst_envelope.CAPS uses, and asks for a wall-clock
# figure analyst_envelope has no opinion on. Both trace back to the SAME
# constants (never a second hardcoded copy), so a change to a cap in
# analyst_envelope or analyst_sandbox is what the model is told, not a stale
# echo of it.
# --------------------------------------------------------------------------- #
def _caps_for_prompt() -> dict[str, Any]:
    return {
        "rows_per_table": analyst_envelope.MAX_ROWS_PER_TABLE,
        "tables_per_run": analyst_envelope.MAX_TABLES,
        "cells_total": analyst_envelope.MAX_CELLS,
        "distinct_numeric_tokens": analyst_envelope.MAX_NUMERIC_TOKENS,
        "envelope_bytes": analyst_envelope.MAX_ENVELOPE_BYTES,
        "wall_clock_seconds": analyst_sandbox.DEFAULT_WALL_CLOCK_S,
    }


# --------------------------------------------------------------------------- #
# Refusal -> remediation. Refusal.reason is free text (analyst_envelope never
# hands back a bare stable code), so this is a best-effort match against the
# REFUSAL_CODES vocabulary rather than a guaranteed lookup. An unmatched
# reason still gets useful guidance: the whole remediation table, rather than
# nothing.
# --------------------------------------------------------------------------- #
_REFUSAL_CODE_HINTS: dict[str, tuple[str, ...]] = {
    "ZERO_READ": ("vault tables and", "reads"),
    "CAP_ROWS": ("exceeds row cap",),
    "CAP_CELLS": ("exceeds cell cap",),
    "BAD_UNIT": ("unit",),
    "BAD_COLUMN": ("naming grammar",),
    "TIMEOUT": ("timed out",),
}


def _guess_refusal_code(reason: str) -> str | None:
    lower = reason.lower()
    for code, needles in _REFUSAL_CODE_HINTS.items():
        if all(needle in lower for needle in needles):
            return code
    return None


def refusal_guidance(reason: str) -> str:
    """The human-actionable remediation for one refusal reason.

    Tries to match ``reason`` to a stable ``REFUSAL_CODES`` entry first; when
    no match is found (a JSON/grammar-shaped refusal from analyst_envelope
    that doesn't fit any of the known code shapes, or a sandbox-level
    failure), falls back to showing the whole remediation table rather than
    nothing.
    """
    code = _guess_refusal_code(reason)
    if code is not None:
        return REFUSAL_REMEDIATION[code]
    return "\n".join(f"- {c}: {r}" for c, r in REFUSAL_REMEDIATION.items())


def build_repair_prompt(question: str, schema: str, caps: dict[str, Any],
                        refusal_reason: str) -> str:
    """A second prompt carrying the first run's refusal and its remediation."""
    base = build_analyst_prompt(question, schema, caps=caps)
    return (
        f"{base}\n\n"
        "Your previous code was refused by the runtime.\n"
        f"Refusal reason: {refusal_reason}\n"
        "Remediation guidance:\n"
        f"{refusal_guidance(refusal_reason)}\n\n"
        "Write corrected Python code that avoids this refusal."
    )


# --------------------------------------------------------------------------- #
# Extracting code from a model reply
# --------------------------------------------------------------------------- #
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    """Pull Python out of a fenced code block, or treat the whole reply as
    code when there is no fence."""
    if not reply:
        return ""
    match = _CODE_FENCE_RE.search(reply)
    if match:
        return match.group(1).strip("\n")
    return reply.strip()


# --------------------------------------------------------------------------- #
# Vault access -- the vault path enters exactly once, from argv (T-003). No
# module-level default, no environment-variable fallback: every caller in
# this module receives the path as an explicit argument.
# --------------------------------------------------------------------------- #
def _open_vault_readonly(vault_path: str) -> sqlite3.Connection:
    conn = dbmod.connect(vault_path, read_only=True)
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Ledger provenance labelling -- honest about what has and hasn't been
# verified. Until the mediated runner lands (GitHub #226), the ledger is
# CHILD-ASSERTED: counters the sandboxed child reports about its own queries,
# not something the parent independently observed. This function never
# upgrades that claim on its own authority; it only relays a
# `parent_observed` flag if a future run_analyst_code actually sets one.
# --------------------------------------------------------------------------- #
def ledger_trust_label(ledger: dict) -> str:
    if ledger.get("parent_observed"):
        return "parent-observed"
    return ("child-asserted (see GitHub #226 -- not yet independently "
            "verified by the parent)")


# --------------------------------------------------------------------------- #
# Rendering -- deterministic, Python-owned. No model narration turn: every
# number below is a cell copied out of a validated Envelope.
# --------------------------------------------------------------------------- #
def _format_cell(value) -> str:
    return str(value)


def render_table(table) -> str:
    """table is a mapping with name/columns/units/rows/row_count, exactly the
    shape analyst_envelope._validate_table (and Envelope.to_dict) produce."""
    columns = list(table["columns"])
    units = list(table["units"])
    rows = [list(row) for row in table["rows"]]
    headers = [f"{c} ({u})" for c, u in zip(columns, units)]
    str_rows = [[_format_cell(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    lines = [f"Table: {table['name']} ({table['row_count']} rows)"]
    lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in str_rows:
        lines.append("  ".join(v.ljust(widths[i]) for i, v in enumerate(row)))
    return "\n".join(lines)


def _provenance_dict(*, run_id: str, vault_sha256: str, vault_version: int,
                      code_sha256: str, ledger: dict, record_path: str) -> dict:
    return {
        "run_id": run_id,
        "vault_sha256": vault_sha256,
        "vault_user_version": vault_version,
        "code_sha256": code_sha256,
        "ledger": {
            "query_count": ledger.get("query_count"),
            "tables_read": list(ledger.get("tables_read") or ()),
            "rows_read": ledger.get("rows_read"),
            "provenance": ledger_trust_label(ledger),
        },
        "run_record_path": str(record_path),
    }


def render_provenance(**kwargs) -> str:
    d = _provenance_dict(**kwargs)
    ledger = d["ledger"]
    lines = [
        "Provenance:",
        f"  run_id: {d['run_id']}",
        f"  vault_sha256: {d['vault_sha256']}",
        f"  vault_user_version: {d['vault_user_version']}",
        f"  code_sha256: {d['code_sha256']}",
        f"  ledger.query_count: {ledger['query_count']}",
        f"  ledger.tables_read: {', '.join(ledger['tables_read']) or '(none)'}",
        f"  ledger.rows_read: {ledger['rows_read']}",
        f"  ledger provenance: {ledger['provenance']}",
        f"  run_record: {d['run_record_path']}",
    ]
    return "\n".join(lines)


def _print_envelope(envelope: Envelope, *, run_id: str, vault_sha256: str,
                     vault_version: int, code_sha256: str, record_path: str,
                     question: str, code: str, json_output: bool, out) -> None:
    if json_output:
        payload = {
            "question": question,
            "tables": [
                {
                    "name": t["name"],
                    "columns": list(t["columns"]),
                    "units": list(t["units"]),
                    "rows": [list(row) for row in t["rows"]],
                    "row_count": t["row_count"],
                }
                for t in envelope.tables
            ],
            "provenance": _provenance_dict(
                run_id=run_id, vault_sha256=vault_sha256,
                vault_version=vault_version, code_sha256=code_sha256,
                ledger=envelope.ledger, record_path=record_path),
            "code": code,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
        return

    print(f"Question: {question}", file=out)
    print(file=out)
    for table in envelope.tables:
        print(render_table(table), file=out)
        print(file=out)
    print(render_provenance(
        run_id=run_id, vault_sha256=vault_sha256, vault_version=vault_version,
        code_sha256=code_sha256, ledger=envelope.ledger,
        record_path=record_path), file=out)
    print(file=out)
    print("Code:", file=out)
    print(code, file=out)


def _print_refusal(refusal: Refusal, *, question: str, json_output: bool, out) -> None:
    remediation = refusal_guidance(refusal.reason)
    if json_output:
        payload = {
            "question": question,
            "remediation": remediation,
            **refusal.to_dict(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
        return
    print(f"Question: {question}", file=out)
    print(file=out)
    print(f"Refused: {refusal.reason}", file=out)
    if refusal.diagnostic is not None:
        print(f"Diagnostic: {refusal.diagnostic}", file=out)
    print(f"Remediation: {remediation}", file=out)


# --------------------------------------------------------------------------- #
# The run record -- the provenance artifact. Parent-authored: only this
# module writes it, and the model-written code never sees or touches it.
# --------------------------------------------------------------------------- #
def _result_to_record_dict(result: "Envelope | Refusal") -> dict:
    if isinstance(result, Refusal):
        return result.to_dict()
    return result.to_dict()


def _write_run_record(run_dir: str, *, run_id: str, question: str,
                       prompt1: str, prompt2: str | None,
                       code1: str, code2: str | None,
                       result: "Envelope | Refusal", vault_sha256: str,
                       vault_version: int, started_at: str, finished_at: str) -> Path:
    ledger = {} if isinstance(result, Refusal) else dict(result.ledger)
    record = {
        "run_id": run_id,
        "question": question,
        "prompts": {"initial": prompt1, "repair": prompt2},
        "code": {"initial": code1, "repair": code2},
        "result": _result_to_record_dict(result),
        "ledger": ledger,
        "vault_sha256": vault_sha256,
        "vault_user_version": vault_version,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    run_dir_p = Path(run_dir)
    run_dir_p.mkdir(parents=True, exist_ok=True)
    path = run_dir_p / "run_record.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The flow
# --------------------------------------------------------------------------- #
def run_analyst(question: str, vault_path: str, run_dir: str, *,
                 complete_fn=None, run_code_fn: _AnalystRunner | None = None,
                 executor=None, limits: RunLimits | None = None,
                 json_output: bool = False, out=None) -> int:
    """Run one analyst-mode question end to end. Returns a process exit code.

    ``complete_fn`` defaults to ``llm.complete``; ``run_code_fn`` defaults to
    the real ``analyst_runner.run_analyst_code`` (imported lazily, only when
    no fake was injected); ``executor`` defaults to whichever real sandbox
    this host supports, via ``default_executor()``. Tests inject fakes for all
    three so no real model and no real sandbox is ever touched in the suite.

    ``default_executor()`` rather than ``SeatbeltExecutor()``: the CLI is how
    a session on a Linux deployment host actually reaches analyst mode, and
    hard-coding the macOS executor made it unusable there the moment a Linux
    one existed. It fails closed the same way the HTTP route does —
    ``NoExecutorAvailable`` is a ``RuntimeError`` naming the platform, never a
    fallback to running model-written code unsandboxed.
    """
    out = out if out is not None else sys.stdout
    complete_fn = complete_fn or llm.complete
    if run_code_fn is None:
        run_code_fn = _load_run_analyst_code()
    exec_obj = executor if executor is not None else default_executor()
    limits = limits or RunLimits()

    started_at = _now_iso()
    run_id = uuid.uuid4().hex

    conn = _open_vault_readonly(vault_path)
    try:
        vault_sha256 = _hash_file(vault_path)
        vault_version = conn.execute("PRAGMA user_version").fetchone()[0]
        schema = schema_summary(conn)
    finally:
        conn.close()

    caps = _caps_for_prompt()
    prompt1 = build_analyst_prompt(question, schema, caps=caps)
    code1 = extract_code(complete_fn(prompt1))
    code1_sha256 = hashlib.sha256(code1.encode("utf-8")).hexdigest()

    result: "Envelope | Refusal" = run_code_fn(
        code1, vault_path, run_dir, exec_obj, limits=limits)

    prompt2: str | None = None
    code2: str | None = None
    final_code = code1
    final_code_sha256 = code1_sha256

    # One bounded repair retry, and only one: a Refusal here builds exactly
    # one second prompt, runs exactly one more time, and whatever comes back
    # -- Envelope or a second Refusal -- is final. Never loop.
    # A SYNTAX_ERROR is the exception to "exactly one retry", and deliberately
    # so. It is decided by `compile()` in the parent before any sandbox runs,
    # so a further attempt costs one model call and nothing else -- no vault
    # read, no child process, no judgement about whether the answer is good.
    # It also happens to be the common failure: measured against the real
    # vault, the model available under D15's provider pin returned unparseable
    # Python on four consecutive questions (#194, #230). Retrying a fact the
    # parent can check is not the same as retrying until we like the answer.
    MAX_SYNTAX_ATTEMPTS = 4          # total model turns when only parsing fails
    MAX_SUBSTANTIVE_ATTEMPTS = 2     # the original "exactly one repair"
    attempts = 1
    while isinstance(result, Refusal):
        syntax = result.reason.startswith("SYNTAX_ERROR")
        budget = MAX_SYNTAX_ATTEMPTS if syntax else MAX_SUBSTANTIVE_ATTEMPTS
        if attempts >= budget:
            break
        prompt2 = build_repair_prompt(question, schema, caps, result.reason)
        code2 = extract_code(complete_fn(prompt2))
        code2_sha256 = hashlib.sha256(code2.encode("utf-8")).hexdigest()
        result = run_code_fn(code2, vault_path, run_dir, exec_obj, limits=limits)
        final_code = code2
        final_code_sha256 = code2_sha256
        attempts += 1

    finished_at = _now_iso()

    record_path = _write_run_record(
        run_dir, run_id=run_id, question=question,
        prompt1=prompt1, prompt2=prompt2, code1=code1, code2=code2,
        result=result, vault_sha256=vault_sha256, vault_version=vault_version,
        started_at=started_at, finished_at=finished_at)

    if isinstance(result, Refusal):
        _print_refusal(result, question=question, json_output=json_output, out=out)
        return 1

    _print_envelope(
        result, run_id=run_id, vault_sha256=vault_sha256,
        vault_version=vault_version, code_sha256=final_code_sha256,
        record_path=str(record_path), question=question, code=final_code,
        json_output=json_output, out=out)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m health_advisor.analyst",
        description=("Analyst mode: ask a novel question against a health "
                     "vault, answered by model-written code sandboxed "
                     "against a read-only connection and validated before "
                     "any number reaches you."),
    )
    parser.add_argument("--vault", required=True,
                        help="Path to the health vault SQLite file.")
    parser.add_argument("--question", required=True,
                        help="The question to answer.")
    parser.add_argument("--run-dir", default=None,
                        help="Directory for run artifacts (default: a fresh "
                             "temp directory).")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of the "
                             "human-rendered tables.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse's own --vault/--question required-arg failure: report it
        # as a clean return code rather than letting SystemExit propagate.
        return exc.code if isinstance(exc.code, int) else 2

    # Approve the backend before anything else -- before the vault is even
    # opened. Analyst mode's own codex-by-name refusal rides on top of the
    # shared D15 gate; see assert_analyst_backend_approved's docstring.
    try:
        assert_analyst_backend_approved()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run_dir = args.run_dir or tempfile.mkdtemp(prefix="ha_analyst_")

    try:
        return run_analyst(args.question, args.vault, run_dir,
                           json_output=args.json)
    except Exception as exc:  # a clean failure, not a bare traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
