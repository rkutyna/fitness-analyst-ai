"""Build a D3-filtered SQLite vault from a full Health Advisor snapshot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from health_advisor.vault import build_vault, format_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="full snapshot SQLite path")
    parser.add_argument(
        "--vault", "--output", dest="vault", required=True,
        help="destination vault SQLite path",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10_000,
        help="rows held per INSERT batch (default: 10000)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="atomically replace an existing destination vault",
    )
    parser.add_argument(
        "--owner",
        help="user id this vault belongs to; a session with a different one is "
             "refused at connect. Omit for an unclaimed development copy.",
    )
    args = parser.parse_args(argv)
    report = build_vault(
        args.source,
        args.vault,
        batch_size=args.batch_size,
        replace=args.force,
        owner=args.owner,
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
