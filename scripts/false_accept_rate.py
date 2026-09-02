#!/usr/bin/env python3
"""Measure the bag-of-numbers grounding gate against a deep-dive payload.

This is deliberately an instrument, not a copy of the gate.  Every probe is
sent through :func:`health_advisor.agents.grounding_check`; this module only
defines the probe spaces and counts the returned verdicts.

    ./.venv/bin/python scripts/false_accept_rate.py \
        --db /path/to/health.db --as-of 2026-08-20
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_advisor import agents, mcp_server  # noqa: E402
from health_advisor.context import VaultContext  # noqa: E402
from health_advisor.numeric_tokens import NUM_RE  # noqa: E402


INTEGER_PROBES = tuple(range(1, 201))
DECIMAL_PROBES = tuple(index / 10 for index in range(1, 1001))
CONTROL_SIZE = 15
CONTROL_SEED = 91


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


def scoreboard_payload(scoreboard: str) -> tuple[dict[str, list[float]], list[str]]:
    """Turn scoreboard tokens into a numeric payload without deduplicating them."""
    tokens = NUM_RE.findall(scoreboard)
    values = [float(token.replace(",", "")) for token in tokens]
    return {"scoreboard_numbers": values}, tokens


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


def control(payload: dict, scoreboard_tokens: list[str]) -> dict[str, Any]:
    """Check a reproducible sample of figures copied from the scoreboard.

    Read the result narrowly. Measured 2026-08-25, four of the fifteen sampled
    tokens are date fragments — `2026`, `09`, `2026`, `10` — because NUM_RE
    splits a rendered date into components. So this demonstrates "numbers
    present in the payload are licensed", which is weaker than "real figures
    verify". It is left as-is because the published 15/15 reproduces against it,
    but the calendar numbers in the licence pool are themselves a false-accept
    surface worth carrying into #39: a model writing "10" of anything is
    licensed by a date.
    """
    if len(scoreboard_tokens) < CONTROL_SIZE:
        raise ValueError("scoreboard has fewer than 15 numeric tokens")
    sample = random.Random(CONTROL_SEED).sample(scoreboard_tokens, CONTROL_SIZE)
    accepted = sum(
        agents.grounding_check(_probe_text(token), payload)[0]
        for token in sample
    )
    return {"accepted": accepted, "total": CONTROL_SIZE, "sample": sample}


def _print_payload(name: str, result: dict[str, Any]) -> None:
    print(f"{name}: {result['number_count']} numbers")
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
    # The probe spaces, `measure_payload` and `control` remain independent of
    # that source and are useful (and tested) on any payload.
    ctx = VaultContext.local(args.db, user_id="demo")
    tools = mcp_server.build_tools(ctx)
    result = tools["get_briefing"](scope="deep", day=args.as_of)
    scoreboard = str(result)

    whole = measure_payload(result)
    rendered, scoreboard_tokens = scoreboard_payload(scoreboard)
    scoreboard_result = measure_payload(rendered)
    scoreboard_control = control(rendered, scoreboard_tokens)

    _print_payload("whole result dict", whole)
    _print_payload("rendered scoreboard", scoreboard_result)
    print("control: scoreboard's own 15 sampled figures: "
          f"{scoreboard_control['accepted']}/{scoreboard_control['total']} accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
