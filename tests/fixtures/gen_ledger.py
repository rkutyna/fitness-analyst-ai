#!/usr/bin/env python
"""Generate the synthetic jog ledger/answer fixtures.

Everything here is invented. The numbers were chosen to satisfy the
assertions in the consuming tests; no value came from a real vault.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

RECENT_STARTS = ["2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17"]
PRIOR_STARTS = ["2026-06-29", "2026-07-06", "2026-07-13", "2026-07-20"]
RECENT_VALUES = [44.6, 55.2, 58.3, 42.3]   # sum 200.4, mean 50.1
PRIOR_VALUES = [47.5, 53.1, 56.9, 43.3]    # sum 200.8, mean 50.2

WEEK_END = {
    "2026-06-29": "2026-07-05", "2026-07-06": "2026-07-12",
    "2026-07-13": "2026-07-19", "2026-07-20": "2026-07-26",
    "2026-07-27": "2026-08-02", "2026-08-03": "2026-08-09",
    "2026-08-10": "2026-08-16", "2026-08-17": "2026-08-23",
}

# Miles/walk context per week start; deliberately none equal 4, 50.1, 50.2
# or 0.1, so the traceability fixture has exactly one numeric occurrence of
# each prose figure.
MILES = {
    "2026-06-29": (5.82, 213.7, 6.91), "2026-07-06": (6.35, 198.2, 6.24),
    "2026-07-13": (7.02, 226.8, 7.13), "2026-07-20": (5.19, 187.6, 5.88),
    "2026-07-27": (5.37, 205.3, 6.42), "2026-08-03": (6.71, 219.1, 6.87),
    "2026-08-10": (7.16, 231.4, 7.35), "2026-08-17": (5.03, 176.9, 5.51),
}


def week(start: str, value: float) -> dict:
    return {
        "days_covered": 7,
        "days_expected": 7,
        "field": "jog_minutes",
        "metric": "jog_minutes",
        "no_data": False,
        "partial": False,
        "period": f"{start}:{WEEK_END[start]}",
        "value": value,
    }


def block(starts, values, mean, total) -> dict:
    return {
        "mean": mean,
        "metric": "jog_minutes",
        # `end` is the EXCLUSIVE query bound; the model-facing spelling of the
        # block is period_starts[0]:period_starts[-1]+6 days.
        "period": {"start": starts[0], "end": "2026-08-24",
                   "period_starts": list(starts)},
        "period_starts": list(starts),
        "total": total,
        "weeks": [week(s, v) for s, v in zip(starts, values)],
    }


def period_row(start: str, jog_minutes: float) -> dict:
    jog_miles, walk_minutes, walk_miles = MILES[start]
    return {
        "days_covered": 7,
        "jog_change_note": "no prior week in range",
        "jog_change_pct": None,
        "jog_miles": jog_miles,
        "jog_minutes": jog_minutes,
        "jog_minutes_source": "cadence",
        "jog_pace_min_per_mi": round(jog_minutes / jog_miles, 2),
        "manual_note": None,
        "no_data": False,
        "partial": False,
        "period_start": start,
        "walk_miles": walk_miles,
        "walk_minutes": walk_minutes,
    }


prior = block(PRIOR_STARTS, PRIOR_VALUES, 50.2, 200.8)
recent = block(RECENT_STARTS, RECENT_VALUES, 50.1, 200.4)
# `period.end` for the prior block is the recent block's start.
prior["period"]["end"] = "2026-07-27"

result = {
    "block_comparison": {
        "anchor": "2026-08-24",
        "anchor_end": "2026-08-23",
        "blocks": {"prior": prior, "recent": recent},
        "change": {
            "mean_delta": -0.1,
            "total_delta": -0.4,
            "total_delta_pct": None,
            "total_delta_pct_note": (
                "Percent change is withheld because the block mean moved by "
                "only 0.1 min/week, which is inside the reporting floor."),
        },
        "completeness": {
            "as_of": "2026-08-24",
            "end_default": True,
            "partial_trailing_week": {
                "days_covered": 1,
                "days_expected": 7,
                "included": None,
                "partial": True,
                "period_start": "2026-08-24",
                "reason": "trailing week is incomplete and was excluded",
            },
            "rule": "complete Monday-Sunday weeks only",
        },
        "metric": "jog_minutes",
        "requested_range": {"start": "2026-06-29", "end": "2026-08-23"},
        "weeks_per_block": 4,
    },
    "by": "week",
    "count": 8,
    "end": "2026-08-23",
    "jog_near_threshold": {
        "buckets_near_cutoff": 63,
        "jog_buckets": 806,
        "note": ("buckets whose pace sits within within_min_per_mi of the "
                 "live cutoff"),
        "pct_of_jog_buckets": 7.82,
        "within_min_per_mi": 0.5,
    },
    "jog_pace_threshold_min_per_mi": 16.0,
    "jog_threshold_sensitivity": [
        {"jog_buckets": 355, "jog_minutes": 88.7,
         "live_cutoff": False, "pace_max_min_per_mi": 14.0},
        {"jog_buckets": 570, "jog_minutes": 142.6,
         "live_cutoff": False, "pace_max_min_per_mi": 15.0},
        {"jog_buckets": 806, "jog_minutes": 201.6,
         "live_cutoff": True, "pace_max_min_per_mi": 16.0},
        {"jog_buckets": 1076, "jog_minutes": 268.9,
         "live_cutoff": False, "pace_max_min_per_mi": 17.0},
        {"jog_buckets": 1327, "jog_minutes": 331.7,
         "live_cutoff": False, "pace_max_min_per_mi": 18.0},
    ],
    "periods": [period_row(s, v) for s, v in
                zip(PRIOR_STARTS + RECENT_STARTS,
                    PRIOR_VALUES + RECENT_VALUES)],
    "start": "2026-06-29",
}

record = {
    "sequence": 1,
    "tool_name": "get_impact_volume",
    "arguments": {
        "anchor": "2026-08-24",
        "by": "week",
        "end": "2026-08-23",
        "start": "2026-06-29",
        "weeks_per_block": 4,
    },
    "result": result,
    "result_bytes": len(json.dumps(result, sort_keys=True)),
    "result_elided": False,
}

line = json.dumps(record, sort_keys=True)
(OUT / "jog_ledger_live_20260824_claims.jsonl").write_text(line + "\n")
(OUT / "jog_ledger_live_20260824.jsonl").write_text(line + "\n")

TEXT = (
    "Average weekly jog minutes:\n"
    "- Last 4 complete weeks (Jul 27-Aug 23): 50.1 min/week\n"
    "- The 4 weeks before (Jun 29-Jul 26): 50.2 min/week\n"
    "Essentially unchanged: a decrease of 0.1 min/week."
)

recent_period = recent["period"]
prior_period = prior["period"]
operands = [
    {"metric": "jog_minutes", "period": recent_period, "field": "mean",
     "value": 50.1,
     "source": {"sequence": 1,
                "path": "$.result.block_comparison.blocks.recent.mean"}},
    {"metric": "jog_minutes", "period": prior_period, "field": "mean",
     "value": 50.2,
     "source": {"sequence": 1,
                "path": "$.result.block_comparison.blocks.prior.mean"}},
]

answer = {
    "captured": "synthetic",
    "note": (
        "SYNTHETIC. The live capture this file replaced held real personal "
        "figures and was not published. Every number here is invented and "
        "chosen so the shape, and the verifier behaviour it pins, are "
        "unchanged: three figures, one of them a derived difference, against "
        "the committed jog ledger."),
    "text": TEXT,
    "claims": [
        {"metric": "jog_minutes", "period": recent_period, "field": "mean",
         "value": 50.1,
         "source": {"sequence": 1,
                    "path": "$.result.block_comparison.blocks.recent.mean"}},
        {"metric": "jog_minutes", "period": prior_period, "field": "mean",
         "value": 50.2,
         "source": {"sequence": 1,
                    "path": "$.result.block_comparison.blocks.prior.mean"}},
        {"metric": "jog_minutes", "period": "comparison", "field": "mean_delta",
         "value": -0.1, "operation": "difference", "operands": operands,
         "source": {"sequence": 1,
                    "path": "$.result.block_comparison.change.mean_delta"}},
    ],
}
(OUT / "jog_answer_live_20260824.json").write_text(
    json.dumps(answer, indent=1, sort_keys=True) + "\n")
print("written")
