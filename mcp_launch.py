#!/usr/bin/env python
"""Stdio launcher for the Health MCP server, used by Hermes (`hermes mcp add`).
A plain script path avoids passing `-m` through Hermes' arg parser, and adding
the project root to sys.path lets the package's relative imports resolve.

Takes `--vault PATH` (see health_advisor.mcp_server.main): the server serves one
session's vault, and which one is not something the environment gets to decide."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from health_advisor.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
