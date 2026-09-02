#!/usr/bin/env python
"""Run one researcher question and report a necessary prose traceability check.

Result measurements and numeric arguments are reported separately.  Calendar
dates are context, not figures, so they are removed before the shared numeric
tokenizer is applied.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_advisor.agents import _PROSE_DATE_RE  # noqa: E402
from health_advisor import llm  # noqa: E402
from health_advisor.context import VaultContext  # noqa: E402
from health_advisor.numeric_tokens import NUM_RE  # noqa: E402


def _as_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _numeric_values(value: Any) -> Iterator[str]:
    """Yield numeric representations from one named result value."""
    if isinstance(value, str):
        yield from NUM_RE.findall(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)


def _result_fields(value: Any, path: tuple[Any, ...] = ()) \
        -> Iterator[tuple[str, str, str]]:
    """Yield ``(field, path, numeric_value)`` entries from a result tree.

    A numeric leaf is evidence only when it has a field path.  Tool payloads
    that explicitly publish ``{"field": ..., "value": ...}`` use the
    published field name while retaining the JSON path for the audit report.
    """
    if isinstance(value, dict):
        declared_field = value.get("field")
        if isinstance(declared_field, str) and declared_field.strip() and "value" in value:
            value_path = path + ("value",)
            for candidate in _numeric_values(value["value"]):
                yield declared_field.strip(), _format_path(value_path), candidate
            skipped = {"field", "value"}
        else:
            skipped = set()
        for key, child in value.items():
            if key in skipped:
                continue
            yield from _result_fields(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _result_fields(child, path + (index,))
    elif path:
        field = str(path[-1])
        for candidate in _numeric_values(value):
            yield field, _format_path(path), candidate


def _format_path(path: tuple[Any, ...]) -> str:
    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}"
    return rendered


_FIELD_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _matching_fields(token: str,
                     records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return every named result field supporting ``token``.

    A bare result key that merely echoes an argument is configuration echo,
    not measurement evidence.  Explicit ``{"field": ..., "value": ...}``
    publications remain eligible because the tool has declared the measured
    field rather than exposing an argument-shaped dict key.
    """
    expected = _as_decimal(token)
    matches: list[dict[str, str]] = []
    for record in records:
        if record.get("result_elided"):
            continue
        sequence = record.get("sequence")
        tool_name = record.get("tool_name")
        if sequence is None or not isinstance(tool_name, str) or not tool_name:
            continue
        for field, path, candidate in _result_fields(record.get("result")):
            if not (candidate == token or (
                    expected is not None and
                    _as_decimal(candidate) == expected)):
                continue
            arguments = record.get("arguments")
            if (isinstance(arguments, dict) and field in arguments
                    and not path.endswith(".value")):
                continue
            matches.append({
                "sequence": str(sequence),
                "tool_name": tool_name,
                "field": field,
                "path": path,
            })
    return matches


def _matching_arguments(token: str, context: str,
                        records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return contextual numeric argument matches, separately from results.

    The argument name must be present near the token.  This keeps an unrelated
    argument such as ``days=100`` from grounding a prose measurement while
    allowing ``weeks_per_block=4`` to document the requested comparison
    window.  Argument values are never presented as tool-returned results.
    """
    expected = _as_decimal(token)
    context_words = {
        word.lower() for word in _FIELD_WORD_RE.findall(context)
    }
    matches: list[dict[str, str]] = []
    for record in records:
        sequence = record.get("sequence")
        tool_name = record.get("tool_name")
        arguments = record.get("arguments")
        if (sequence is None or not isinstance(tool_name, str)
                or not tool_name or not isinstance(arguments, dict)):
            continue
        for argument, value in arguments.items():
            if not isinstance(argument, str):
                continue
            candidates = _numeric_values(value)
            if not any(candidate == token or (
                    expected is not None and
                    _as_decimal(candidate) == expected)
                       for candidate in candidates):
                continue
            argument_words = {
                word.lower() for word in _FIELD_WORD_RE.findall(argument)
            }
            argument_words.discard("n")
            if not argument_words.intersection(context_words):
                continue
            matches.append({
                "sequence": str(sequence),
                "tool_name": tool_name,
                "argument": argument,
            })
    return matches


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Read the server-written JSONL ledger, preserving record order."""
    with Path(path).open(encoding="utf-8") as ledger_file:
        return [json.loads(line) for line in ledger_file if line.strip()]


def _traceable(token: str, records: list[dict[str, Any]],
               context: str = "") -> bool:
    return bool(_matching_fields(token, records) or
                _matching_arguments(token, context, records))


def _claim_text(answer: str) -> str:
    """Remove ISO and prose calendar dates before numeric tokenization."""
    cleaned = re.sub(r"\d{4}-\d{2}-\d{2}", " ", answer)
    return _PROSE_DATE_RE.sub(" ", cleaned)


def _context(answer: str, start: int, end: int, width: int = 30) -> str:
    left = max(0, start - width)
    right = min(len(answer), end + width)
    snippet = answer[left:right].replace("\n", " ")
    if left:
        snippet = "…" + snippet
    if right < len(answer):
        snippet += "…"
    return snippet


def _trace_answer(answer: str, records: list[dict[str, Any]]) \
        -> tuple[int, list[dict[str, str]], list[dict[str, str]]]:
    """Return count, misses, and every named provenance match per figure."""
    misses: list[dict[str, str]] = []
    matches: list[dict[str, str]] = []
    traceable = 0
    claim_text = _claim_text(answer)
    for match in NUM_RE.finditer(claim_text):
        token = match.group(0)
        context = _context(claim_text, match.start(), match.end())
        result_matches = _matching_fields(token, records)
        argument_matches = _matching_arguments(token, context, records)
        if result_matches or argument_matches:
            traceable += 1
            for ledger_match in result_matches:
                matches.append({
                    "token": token,
                    "context": context,
                    "category": "TRACEABLE-RESULT",
                    **ledger_match,
                })
            for argument_match in argument_matches:
                matches.append({
                    "token": token,
                    "context": context,
                    "category": "TRACEABLE-ARG",
                    **argument_match,
                })
        else:
            misses.append({
                "token": token,
                "context": context,
            })
    return traceable, misses, matches


def trace_answer(answer: str, records: list[dict[str, Any]]) \
        -> tuple[int, list[dict[str, str]]]:
    """Return the traceable occurrence count and contextual misses."""
    traceable, misses, _ = _trace_answer(answer, records)
    return traceable, misses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="vault path")
    parser.add_argument("--question", required=True, help="literal researcher question")
    parser.add_argument("--backend", required=True,
                        choices=["codex", "ollama", "openrouter"])
    parser.add_argument("--ledger-out", required=True, help="JSONL ledger output path")
    parser.add_argument("--user-id", default="local")
    args = parser.parse_args(argv)

    llm.BACKEND = args.backend
    ctx = VaultContext.local(args.db, user_id=args.user_id)
    answer = llm.tool_loop(
        args.question,
        ctx=ctx,
        tools=llm.tool_schemas(ctx),
        ledger_path=args.ledger_out,
    )
    records = read_ledger(args.ledger_out)
    total = sum(1 for _ in NUM_RE.finditer(_claim_text(answer)))
    traceable, misses, matches = _trace_answer(answer, records)

    print(f"ANSWER: {answer}")
    print(f"TRACEABILITY: {traceable} of {total} figures traceable "
          "(NECESSARY CONDITION ONLY)")
    for matched in matches:
        if matched["category"] == "TRACEABLE-RESULT":
            print(f"TRACEABLE-RESULT: token={matched['token']!r} "
                  f"sequence={matched['sequence']} tool={matched['tool_name']!r} "
                  f"field={matched['path']!r} ({matched['field']!r})")
        else:
            print(f"TRACEABLE-ARG: token={matched['token']!r} "
                  f"sequence={matched['sequence']} tool={matched['tool_name']!r} "
                  f"argument={matched['argument']!r}")
    for miss in misses:
        print(f"UNTRACEABLE: token={miss['token']!r} context={miss['context']!r}")
    if misses:
        print("RESULT: FAIL — necessary prose condition failed; claim verification is separate")
        return 1
    print("RESULT: PASS — necessary prose condition; claim verification is separate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
