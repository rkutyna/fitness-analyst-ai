#!/usr/bin/env python
"""Author the prose date/name strip corpus and check it against the gate.

The token list in each case is written by hand from the documented rule, then
compared against `strip_dates_and_names` + `_NUM_RE`. A disagreement is a real
finding to reason about, not something to overwrite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from health_advisor import agents as G  # noqa: E402

CASES = [
    # --- what the date and name patterns take out of the gate -------------
    ("iso_date", "strip",
     "On 2026-08-21 you logged 42 jog minutes.", ["42"],
     "A full ISO date locates a point in time; only the 42 is a claim."),
    ("iso_year_month", "strip",
     "The 2026-04 quarter averaged 38.5 jog minutes a week.", ["38.5"],
     "The bare year-month form, without which the label shreds into 2026 "
     "and -04."),
    ("prose_month_day", "strip",
     "Aug 21 was the longest session at 6.2 miles.", ["6.2"],
     "Month plus day is a calendar reference."),
    ("prose_month_day_year", "strip",
     "Jul 9, 2026 was the first of 12 planned tempo sessions.", ["12"],
     "The optional year must not leave 2026 behind as a figure."),
    ("prose_month_year", "strip",
     "July 2026 totalled 214 jog minutes.", ["214"],
     "The month-year alternative is tried first; the day form would match "
     "'July 20' and leave the token 26."),
    ("prose_day_range", "strip",
     "July 6-13 you averaged 48.9 minutes a session.", ["48.9"],
     "A day range in the first element is a date, not a measured range."),
    ("clock_with_meridiem", "strip",
     "You fell asleep at 11:41 PM and slept 7.4 hours.", ["7.4"],
     "A 12-hour clock time with AM/PM is a time of day."),
    ("day_list_terminated_by_bracket", "strip",
     "The rest of the block (Aug 19, 21, 22, 23) recorded walking only.", [],
     "#187: a comma-separated day list that terminates cleanly is one date."),
    ("day_list_terminated_by_period", "strip",
     "The walking-only days were Aug 19, 21 and 23.", [],
     "'and' is a separator and a sentence-final period terminates the run."),
    ("zone_name", "strip",
     "Zone 2 work made up 63 of those minutes.", ["63"],
     "Zone 2 is a name from the plan's heart-rate model."),
    ("race_distance_name", "strip",
     "Your 5k pace held at 9.4 min/mi.", ["9.4"],
     "5k and 10k are race distances, not counts."),
    ("authored_week_label", "strip",
     "Week 7 closed with 3 hard sessions.", ["3"],
     "A capitalised Week label is an authored plan ordinal."),
    ("scale_denominator_five", "strip",
     "You rated sleep 3 out of 5 and energy 4 out of 5.", ["3", "4"],
     "#217: the denominator is a constant of the check-in scale; the "
     "numerator stays graded."),
    ("scale_denominator_hundred", "strip",
     "Readiness scored 82 out of 100 this morning.", ["82"],
     "The readiness composite is the other of the two published scales."),

    # --- what must keep being graded --------------------------------------
    ("unit_suffix", "keep",
     "You weighed 84.2kg at the last weigh-in.", ["84.2"],
     "Units are open-ended, so a digit followed by letters stays a figure."),
    ("pace_lane_range", "keep",
     "Your easy lane is 14-15 min/mi.", ["14", "15"],
     "A hyphen between measurements is a range; both ends stay graded and "
     "neither is folded into a negative."),
    ("bare_clock_is_a_pace", "keep",
     "You held 7:30 per mile through the final rep.", ["7", "30"],
     "A bare mm:ss is a pace or a duration; only a marked 12-hour clock is "
     "stripped."),
    ("digit_inside_a_name", "keep",
     "Your VO2max estimate is 48.3.", ["48.3"],
     "The leading guard keeps the 2 of VO2max out while the real figure "
     "stays."),
    ("a1c_name", "keep",
     "Your A1c reading was 5.2 this quarter.", ["5.2"],
     "Same guard, a different name."),
    ("lowercase_week_determiner", "keep",
     "This week 3 sessions are logged.", ["3"],
     "The Week label is case-sensitive on purpose: the determiner phrase "
     "must not strip a real count."),
    ("day_shaped_guard_is_a_range", "keep",
     "On Aug 19, 44 jog minutes were recorded.", ["44"],
     "44 is not a day, so the continuation tail cannot eat it."),
    ("day_list_guard_decimal", "keep",
     "On Aug 19, 44.0 minutes of that were jogging.", ["44.0"],
     "A decimal disqualifies a continuation element; half-eating a figure "
     "would fabricate a different number."),
    ("day_list_guard_thousands", "keep",
     "On Aug 19, 12,500 steps were recorded.", ["12,500"],
     "A thousands separator disqualifies it for the same reason."),
    ("day_list_terminated_by_word", "keep",
     "Aug 19, 21 and 22 were rest days.", ["21", "22"],
     "A run followed by a word is not a date list; a bare day left graded "
     "is the tolerable failure."),
    ("day_list_all_or_nothing", "keep",
     "Since Aug 19, 8, 9, 12 were your resting HR readings",
     ["8,", "9,", "12"],
     "The atomic run cannot succeed partially: the trailing word protects "
     "the interior elements too. The trailing comma rides along in the "
     "token because the tokenizer treats it as a thousands separator; "
     "`_rule_r_matches` strips it before comparing."),
    ("range_continuation_unsupported", "keep",
     "Aug 19, 21-23 were travel days.", ["21", "23"],
     "A range as a continuation element is deliberately not supported."),
    ("scale_without_anchor", "keep",
     "3 out of 5 planned sessions were completed.", ["3", "5"],
     "No rating anchor, so the denominator is a genuine published count."),
    ("scale_other_denominator", "keep",
     "You scored 3 out of 500 on the composite.", ["3", "500"],
     "Only the two published constants are blanked; any other value stays "
     "graded."),
    ("bare_five_is_not_a_scale", "keep",
     "You slept 5 h.", ["5"],
     "The pattern strips a denominator slot, never a digit."),
    ("negative_figure", "keep",
     "Your weight trend is -0.4 lb per week.", ["-0.4"],
     "A minus sign is part of the figure and must reach the gate."),
    ("visible_range_is_not_folded", "keep",
     "Recovery walks ran 29-30 minutes.", ["29", "30"],
     "Removing the separator would fabricate 2930."),
    ("percent_figure", "keep",
     "Jog volume rose 12.5% over the block.", ["12.5"],
     "A percentage is a measurement."),
]

note = (
    "SYNTHETIC. #187's amended criterion 5 kept the corpus in the repo "
    "because the battery it was measured against was lost (#172). The prose "
    "here is invented for this public repo -- no captured answer text -- but "
    "each case is one documented branch of `strip_dates_and_names`, and the "
    "assertion is the exact surviving token list. `kind` records intent: "
    "'strip' means the case exists because a date or name term is removed, "
    "'keep' because a figure must survive.")

fail = 0
for case_id, kind, prose, tokens, why in CASES:
    got = G._NUM_RE.findall(G.strip_dates_and_names(prose))
    if got != tokens:
        fail += 1
        print(f"MISMATCH {case_id}: expected {tokens} got {got}")
print(f"{len(CASES)} cases, {fail} mismatches")

if not fail and "--write" in sys.argv:
    payload = {
        "note": note,
        "cases": [{"id": i, "kind": k, "prose": p, "tokens": t, "why": w}
                  for i, k, p, t, w in CASES],
    }
    (ROOT / "tests/fixtures/prose_date_strip_cases.json").write_text(
        json.dumps(payload, indent=1) + "\n")
    print("written")
