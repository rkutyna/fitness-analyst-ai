#!/usr/bin/env python
"""Replay preserved /v1/ask captures through the offline claim verifier.

The capture's top-level mode/reason is the before snapshot. The after result is
computed from the final attempt's prose, claims, and ledger with ``conn=None``;
this deliberately excludes the SQL cross-check tier and all model calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from health_advisor.deepdive_verify import verify_coach_claims  # noqa: E402


METRIC_REASON = "claim metric does not match ledger field"
CONTROL_ARMS = frozenset(("control1", "control2"))

_LEGACY_FACTOR_METRICS = {"hrv": "heart_rate_variability",
                          "rhr": "resting_heart_rate",
                          "sleep": "sleep_asleep"}


def _bridge_legacy_labels(record: dict) -> None:
    """Inject the ``field_metrics`` today's publishers declare (#158, #202)
    into ledger records captured before those publishers existed.

    This lives in the replay harness, NOT in the verifier: the runtime keeps
    exactly one source of ownership truth (the publishers), and frozen
    evidence is upgraded to the modern shape here instead. ``setdefault``
    only — a capture that already carries labels is untouched, so the bridge
    is a no-op on post-#202 captures and this harness stays valid for them.
    """
    if record.get("result_elided") or not isinstance(record.get("result"), dict):
        return
    result = record["result"]
    tool = record.get("tool_name")
    if tool == "get_briefing":
        readiness = result.get("readiness")
        if isinstance(readiness, dict):
            readiness.setdefault("field_metrics", {}).setdefault(
                "score", "readiness")
            components = readiness.get("components")
            if isinstance(components, dict):
                components.setdefault("field_metrics", {}).setdefault(
                    "sleep", "readiness")
            for factor in readiness.get("factors") or []:
                if isinstance(factor, dict):
                    metric = _LEGACY_FACTOR_METRICS.get(factor.get("component"))
                    if metric:
                        labels = factor.setdefault("field_metrics", {})
                        labels.setdefault("current", metric)
                        labels.setdefault("baseline", metric)
    elif tool == "get_sleep_regularity":
        variability = result.get("midpoint_variability")
        if isinstance(variability, dict):
            variability.setdefault("field_metrics", {}).setdefault(
                "latest_sd_hours", "sleep_midpoint_sd_28d")


def _captures(root: Path):
    for path in sorted(root.glob("*/q*_s*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        attempt = data["attempts"][-1]
        for record in attempt["ledger"]:
            _bridge_legacy_labels(record)
        after = verify_coach_claims(
            None, attempt["prose"], attempt["claims"],
            as_of=data.get("as_of"), payload=attempt["ledger"])
        yield {
            "arm": path.parent.name,
            "question": path.name.split("_", 1)[0],
            "path": path,
            "before_ok": data.get("mode") == "narration",
            "before_reason": data.get("reason", ""),
            "after_ok": bool(after.get("ok")),
            "after_reason": after.get("reason", ""),
        }


def _summary(rows):
    groups = {
        "control": [row for row in rows if row["arm"] in CONTROL_ARMS],
        "ledgerA": [row for row in rows if row["arm"] not in CONTROL_ARMS],
    }
    print("Replay: before=capture mode, after=offline verify_coach_claims(conn=None)")
    for group, group_rows in groups.items():
        before_fallback = sum(not row["before_ok"] for row in group_rows)
        after_fallback = sum(not row["after_ok"] for row in group_rows)
        before_metric = sum(row["before_reason"] == METRIC_REASON
                            for row in group_rows)
        after_metric = sum(row["after_reason"] == METRIC_REASON
                           for row in group_rows)
        before_sleep = [row for row in group_rows
                        if row["question"] in {"q05", "q06"}]
        after_sleep = [row for row in group_rows
                       if row["question"] in {"q05", "q06"}]
        print(f"{group}: all {len(group_rows)}; fallback "
              f"{before_fallback}->{after_fallback}; metric-mismatch "
              f"{before_metric}->{after_metric}; sleep "
              f"{len(before_sleep)} fallback "
              f"{sum(not row['before_ok'] for row in before_sleep)}->"
              f"{sum(not row['after_ok'] for row in after_sleep)}; sleep "
              f"metric-mismatch "
              f"{sum(row['before_reason'] == METRIC_REASON for row in before_sleep)}->"
              f"{sum(row['after_reason'] == METRIC_REASON for row in after_sleep)}")
        for question in ("q01", "q02", "q03", "q04", "q05", "q06"):
            qs = [row for row in group_rows if row["question"] == question]
            if not qs:
                continue
            print(f"  {question}: fallback "
                  f"{sum(not row['before_ok'] for row in qs)}->"
                  f"{sum(not row['after_ok'] for row in qs)}; metric-mismatch "
                  f"{sum(row['before_reason'] == METRIC_REASON for row in qs)}->"
                  f"{sum(row['after_reason'] == METRIC_REASON for row in qs)}")
    regressed = [row["path"] for row in rows
                 if row["before_ok"] and not row["after_ok"]]
    print(f"previously verified that stopped verifying: {len(regressed)}")
    for path in regressed:
        print(f"  REGRESSION {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = list(_captures(args.captures_dir))
    if not rows:
        parser.error(f"no capture files found under {args.captures_dir}")
    _summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
