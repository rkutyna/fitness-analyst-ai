#!/usr/bin/env python
"""Stdio launcher for the deep-dive researcher MCP server (read-only research
tools + run-scoped notepad + optional append-only tool-call ledger), spawned by
`codex exec` during the weekly deep dive. Mirrors mcp_launch.py: a plain script
path so no -m/cwd gymnastics, and the project root on sys.path so the package's
relative imports resolve.

The vault, scratchpad, task id, and ledger path arrive on argv — see
health_advisor.context for why none of them is read from the environment."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from health_advisor.deepdive_mcp import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
