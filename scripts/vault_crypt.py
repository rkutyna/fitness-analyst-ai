#!/usr/bin/env python3
"""Command-line wrapper for Health Advisor vault envelope encryption."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from health_advisor.vault_crypto import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
