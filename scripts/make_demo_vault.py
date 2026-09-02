#!/usr/bin/env python3
"""Generate a synthetic demo vault — thin wrapper over `health_advisor.demo`.

Exists so the demo generator is reachable the same way every other maintenance
script in this directory is, from a checkout that has not been pip-installed:

    ./.venv/bin/python scripts/make_demo_vault.py --out data/demo.db --days 730

`python -m health_advisor.demo` is the same entry point and takes the same
flags. All of the generation logic lives in the package, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from health_advisor.demo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
