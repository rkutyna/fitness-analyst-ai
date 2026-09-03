#!/usr/bin/env python3
"""Measure the bag-of-numbers grounding gate against MCP tool payloads.

This is deliberately an instrument, not a copy of the gate.  Every probe is
sent through :func:`health_advisor.agents.grounding_check`; this module only
defines the probe spaces and counts the returned verdicts.

    ./.venv/bin/python scripts/false_accept_rate.py \
        --db /path/to/health.db --as-of 2026-08-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_advisor import agents, mcp_server  # noqa: E402
from health_advisor.context import VaultContext  # noqa: E402
from health_advisor.numeric_tokens import NUM_RE  # noqa: E402


INTEGER_PROBES = tuple(range(1, 201))
DECIMAL_PROBES = tuple(index / 10 for index in range(1, 1001))


def numeric_occurrence_count(value: Any) -> int:
    """Count numeric leaves, retaining repeated occurrences and excluding bool."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return 1
    if isinstance(value, dict):
        return sum(numeric_occurrence_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(numeric_occurrence_count(item) for item in value)
    return 0


def presentation_leaf_count(value: Any) -> int:
    """Count the engine's presentation leaves without interpreting their text."""
    if isinstance(value, dict):
        is_leaf = (value.get("field") == "presentation"
                   and isinstance(value.get("value"), str))
        return int(is_leaf) + sum(presentation_leaf_count(item)
                                  for item in value.values())
    if isinstance(value, list):
        return sum(presentation_leaf_count(item) for item in value)
    return 0


def _probe_text(token: str) -> str:
    """Build one-number prose and verify it uses the shared tokenizer exactly."""
    if NUM_RE.findall(token) != [token]:
        raise ValueError(f"probe is not one NUM_RE token: {token!r}")
    # Leave whitespace after the token: NUM_RE intentionally permits a
    # trailing decimal point, so ``... is 1.`` would tokenize as ``1.``.
    prose = f"The reported value is {token} today."
    if NUM_RE.findall(prose) != [token]:
        raise ValueError(f"probe prose has unexpected numeric tokens: {prose!r}")
    return prose


def _accepted_count(payload: dict, values: Iterable[float], decimals: int) -> int:
    accepted = 0
    for value in values:
        token = str(value) if decimals == 0 else f"{value:.1f}"
        ok, _ = agents.grounding_check(_probe_text(token), payload)
        accepted += int(ok)
    return accepted


def measure_payload(payload: dict,
                    integer_values: Iterable[int] = INTEGER_PROBES,
                    decimal_values: Iterable[float] = DECIMAL_PROBES) -> dict[str, Any]:
    """Measure one payload; optional iterables make the instrument unit-testable."""
    integers = tuple(integer_values)
    decimals = tuple(decimal_values)
    integer_accepted = _accepted_count(payload, integers, decimals=0)
    decimal_accepted = _accepted_count(payload, decimals, decimals=1)
    return {
        "number_count": numeric_occurrence_count(payload),
        "presentation_leaf_count": presentation_leaf_count(payload),
        "integers": {
            "accepted": integer_accepted,
            "total": len(integers),
            "rate": integer_accepted / len(integers),
        },
        "one_decimal": {
            "accepted": decimal_accepted,
            "total": len(decimals),
            "rate": decimal_accepted / len(decimals),
        },
    }

def _print_payload(name: str, result: dict[str, Any]) -> None:
    print(f"{name}: {result['number_count']} numbers")
    print(f"  presentation leaves: {result['presentation_leaf_count']}")
    integers = result["integers"]
    print(f"  integers 1..200: {integers['accepted']}/{integers['total']} = "
          f"{integers['rate']:.1%}")
    decimals = result["one_decimal"]
    print(f"  one-decimal 0.1..100.0: {decimals['accepted']}/{decimals['total']} = "
          f"{decimals['rate']:.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="vault SQLite path")
    parser.add_argument("--as-of", required=True, help="deep-dive snapshot date")
    args = parser.parse_args(argv)

    # The payload comes from an engine-native MCP tool over the supplied vault.
    # The probe spaces and `measure_payload` remain independent of that source
    # and are useful (and tested) on any payload.
    ctx = VaultContext.local(args.db, user_id="demo")
    tools = mcp_server.build_tools(ctx)
    control_payload = tools["get_briefing"](scope="deep", day=args.as_of)
    presentation_payload = tools["get_sleep_regularity"](end=args.as_of)
    if presentation_leaf_count(presentation_payload) == 0:
        raise RuntimeError("get_sleep_regularity returned no presentation leaves")

    print("N numbers = numeric leaves only; digits inside presentation strings "
          "are not counted.")
    for name, payload in (
        ("control payload (get_briefing)", control_payload),
        ("presentation payload (get_sleep_regularity)", presentation_payload),
    ):
        result = measure_payload(payload)
        _print_payload(name, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
