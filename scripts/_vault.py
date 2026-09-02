"""The repo-local vault path, for developer scripts only.

`health_advisor` deliberately has no default database path: one process serves
many users' vaults, and a default is how a session reads the wrong one (see
health_advisor/context.py). These are one-off maintenance and audit scripts run
by hand against this checkout, where a default is a convenience rather than an
ambient global — so it lives out here, outside the package, where nothing that
serves a user session can reach it.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DB_PATH = REPO_ROOT / "data" / "health.db"
