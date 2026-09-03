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

from health_advisor import agents  # noqa: E402
from health_advisor.deepdive_verify import verify_coach_claims  # noqa: E402


METRIC_REASON = "claim metric does not match ledger field"
CONTROL_ARMS = frozenset(("control1", "control2"))

_LEGACY_FACTOR_METRICS = {"hrv": "heart_rate_variability",
                          "rhr": "resting_heart_rate",
                          "sleep": "sleep_asleep"}


def _bridge_legacy_labels(record: dict) -> None:
    """Inject the ``field_metrics`` today's publishers declare (#14, #158, #202)
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
    elif tool == "get_impact_volume":
        for row in result.get("jog_threshold_sensitivity") or []:
            if isinstance(row, dict):
                row.setdefault("field_metrics", {}).setdefault(
                    "jog_minutes", "jog_minutes")


def _figure_key(value):
    """Use the verifier's numeric spelling rules for comparison only."""
    try:
        return "number", f"{float(str(value).replace(',', '')):.12g}"
    except (TypeError, ValueError):
        return "text", str(value)


def _numeric_figures(value) -> list:
    """Extract only canonical numeric tokens from one rejected value."""
    if isinstance(value, str):
        return agents._numeric_tokens(value)
    try:
        float(str(value).replace(',', ''))
    except (TypeError, ValueError):
        return []
    return [value]


def _rejected_figures(verification: dict) -> list:
    """Return rejected claim values and unsupported tokens once each."""
    if not isinstance(verification, dict):
        return []
    rejected = []
    failed_keys = set()
    verdict = verification.get("verdict")
    numbers = verdict.get("numbers") or [] if isinstance(verdict, dict) else []
    for number in numbers:
        if not isinstance(number, dict) or number.get("ok"):
            continue
        value = number.get("claimed")
        if value is None:
            value = number.get("value")
        for figure in _numeric_figures(value):
            rejected.append(figure)
            failed_keys.add(_figure_key(figure))
    unsupported = verification.get("unsupported") or []
    if not isinstance(unsupported, list):
        unsupported = []
    for token in unsupported:
        for figure in _numeric_figures(token):
            if _figure_key(figure) not in failed_keys:
                rejected.append(figure)
    return rejected


def _classify_rejected_figures(attempts: list[dict],
                               final_verification: dict | None = None) -> list[dict]:
    """Classify rejected figures against prose seen before each attempt.

    Later attempts may be retries or redacted regenerations. A rejected value
    found only in the current attempt is therefore not evidence that an
    earlier draft contained it. The initial attempt's prose is the draft for
    its own rejection; later attempts see all prose before them. This is
    diagnostic only: it never changes the verifier's verdict or licenses a
    value for publication. The final attempt uses freshly replayed
    verification when supplied; earlier attempts use their captured Python
    verification.
    """
    if not isinstance(attempts, list) or not attempts:
        return []
    seen_keys = set()
    classified = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            continue
        prose_keys = {_figure_key(token) for token in
                      agents._numeric_tokens(attempt.get("prose", ""))}
        if index == 0:
            seen_keys.update(prose_keys)
        verification = (final_verification if index == len(attempts) - 1
                        and final_verification is not None
                        else attempt.get("verification", {}))
        for value in _rejected_figures(verification):
            present = _figure_key(value) in seen_keys
            classified.append({
                "attempt": attempt.get("attempt", index + 1),
                "figure": value,
                "classification": ("present-in-draft" if present
                                    else "never-in-draft"),
                "present_in_draft": present,
            })
        seen_keys.update(prose_keys)
    return classified


def _captures(root: Path):
    for path in sorted(root.glob("*/q*_s*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        attempts = data.get("attempts") or []
        if not attempts:
            continue
        attempt = attempts[-1]
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
            "rejected_figures": _classify_rejected_figures(
                attempts, final_verification=after),
        }


def _summary(rows):
    groups = {
        "control": [row for row in rows if row["arm"] in CONTROL_ARMS],
        "ledgerA": [row for row in rows if row["arm"] not in CONTROL_ARMS],
    }
    print("Replay: before=capture mode, after=offline verify_coach_claims(conn=None)")
    rejected = [figure for row in rows
                for figure in row.get("rejected_figures", [])]
    present = sum(figure["classification"] == "present-in-draft"
                  for figure in rejected)
    never = sum(figure["classification"] == "never-in-draft"
                for figure in rejected)
    print(f"captures: {len(rows)}")
    print(f"rejected figures: {len(rejected)}; present-in-draft: {present}; "
          f"never-in-draft: {never}")
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
    parser.add_argument("--captures-dir", default=Path("data/ask_captures"),
                        type=Path)
    args = parser.parse_args(argv)
    rows = list(_captures(args.captures_dir))
    _summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
