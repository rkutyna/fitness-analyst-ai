"""Method C: throw the model's citation away and re-match on content.

`HA_ASK_LEDGER_RESOLVE` (Method B) widened *which record* is searched but kept
asking `_resolve_ledger_value_in_record(record, claim, path)` with the model's
own path. A claim that names the wrong leaf as well as the wrong record
therefore fails in every record — the search was widened over a broken
predicate.

`HA_ASK_VALUE_REBIND=1` (Method C) discards `{sequence, path}` entirely and asks
the only question the ledger can answer: is there EXACTLY ONE citable result
entry whose value, field, metric and period all agree with the claim?

Every test here exists to hold one line: the candidate set widened and nothing
else. No float tolerance, no case-insensitive metric, no period fuzzing, and —
the one that matters most — no tie-break. Two candidates are refused, because
an ambiguous rebind is a coin flip presented to the athlete as a verified
figure, which is strictly worse than the refusal it replaces.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from health_advisor import db as dbmod
from health_advisor import deepdive_verify as DV


REBIND = "HA_ASK_VALUE_REBIND"
SEARCH = "HA_ASK_LEDGER_RESOLVE"

# The pointer the model got wrong in BOTH coordinates: a record that does not
# publish this number, at a leaf that does not exist anywhere.
WRONG_PATH = "$.result.absent"


@pytest.fixture(autouse=True)
def both_flags_off(monkeypatch):
    """No test inherits a flag it did not ask for."""
    monkeypatch.delenv(REBIND, raising=False)
    monkeypatch.delenv(SEARCH, raising=False)


@pytest.fixture
def rebind_on(monkeypatch):
    monkeypatch.setenv(REBIND, "1")


def _series_record(sequence: int, metric: str, mean: float,
                   period: str = "30d") -> dict:
    return {
        "sequence": sequence,
        "tool_name": "get_metric_series",
        "arguments": {"metric": metric, "period": period},
        "result": {"metric": metric, "period": period, "mean": mean},
        "result_elided": False,
    }


def _ledger() -> list[dict]:
    """The two sleep series calls the live battery mixed up."""
    return [
        _series_record(2, "sleep_asleep", 6.42),
        _series_record(13, "sleep_time_in_bed", 7.31),
    ]


def _claim(*, metric, value, sequence, field="mean", period="30d",
           path=WRONG_PATH) -> dict:
    return {
        "metric": metric, "period": period, "field": field, "value": value,
        "source": {"sequence": sequence, "path": path},
    }


def _workout_ledger() -> tuple[list[dict], str]:
    workout_key = dbmod.workout_key(
        "running", "2026-08-15T12:00:00Z", "2026-08-15T13:08:48Z")
    return _ledger() + [{
        "sequence": 5,
        "tool_name": "list_workouts",
        "arguments": {"start": "2026-08-01", "end": "2026-08-31"},
        "result": {"workouts": [{
            "workout_key": workout_key,
            "date": "2026-08-15",
            "type": "running",
            "duration_min": 68.8,
        }]},
        "result_elided": False,
    }], workout_key


# --- the fix itself ------------------------------------------------------

def test_wrong_sequence_and_wrong_path_rebinds_only_with_the_flag_on(monkeypatch):
    """The case Method B cannot reach: BOTH coordinates of the pointer wrong.

    6.42 is sleep_asleep's 30d mean and is published as such — by sequence 2, at
    `$.result.mean`. The claim names sequence 13 and a leaf that exists nowhere.
    Method B searches every record for `$.result.absent` and finds nothing.
    Method C ignores the pointer and finds exactly one content match.
    """
    ledger = _ledger()
    claim = _claim(metric="sleep_asleep", value=6.42, sequence=13)

    off = DV._resolve_ledger_value(ledger, claim)
    monkeypatch.setenv(REBIND, "1")
    on = DV._resolve_ledger_value(ledger, claim)

    assert off["ok"] is False
    assert off["reason"] == "ledger path not found"
    assert "resolved_by" not in off

    assert on["ok"] is True
    assert on["resolved_by"] == "value_rebind"
    assert on["resolved_sequence"] == 2
    assert on["claimed_sequence"] == 13
    assert on["claimed_path"] == WRONG_PATH
    # The entry carries the path actually matched, not the one claimed, and the
    # tier semantics are untouched: a metric-labelled claim is still tier 2.
    assert on["path"] == "$.result.mean"
    assert on["tier"] == "metric"
    assert on["value"] == 6.42


def test_method_b_search_also_fails_on_this_claim(monkeypatch):
    """Method C is not a re-implementation of B: B genuinely cannot do this.

    With only `HA_ASK_LEDGER_RESOLVE=1`, the same claim still fails, because
    every candidate record is probed with the same wrong path.
    """
    monkeypatch.setenv(SEARCH, "1")

    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is False
    assert verdict["reason"] == "ledger path not found"


def test_a_correctly_cited_claim_never_reaches_the_rebind(rebind_on):
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=2,
                          path="$.result.mean"))

    assert verdict["ok"] is True
    assert "resolved_by" not in verdict
    assert "claimed_path" not in verdict


def test_rebind_can_correct_the_leaf_without_moving_the_sequence(rebind_on):
    """The model cited the right CALL and the wrong leaf inside it.

    This still rebinds, but `resolved_sequence == claimed_sequence`, which is
    the distinction the instrumentation below is built to keep.
    """
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=2))

    assert verdict["ok"] is True
    assert verdict["resolved_by"] == "value_rebind"
    assert verdict["resolved_sequence"] == 2
    assert verdict["claimed_sequence"] == 2


def test_a_sequence_that_does_not_exist_at_all_still_rebinds(rebind_on):
    """The model can invent a call number outright, not just swap two.

    `ledger sequence not found` is not an ambiguity, so the rebind is allowed to
    answer it — the value is still published exactly once.
    """
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=99))

    assert verdict["ok"] is True
    assert verdict["resolved_by"] == "value_rebind"
    assert verdict["resolved_sequence"] == 2
    assert verdict["claimed_sequence"] == 99


def test_verify_number_carries_the_rebind_marker_up(rebind_on):
    verdict = DV.verify_number(
        None, _claim(metric="sleep_asleep", value=6.42, sequence=13),
        payload=_ledger())

    assert verdict["ok"] is True
    assert verdict["tier"] == "metric"
    assert verdict["resolved_by"] == "value_rebind"
    assert verdict["resolved_sequence"] == 2
    assert verdict["claimed_sequence"] == 13
    assert verdict["claimed_path"] == WRONG_PATH


# --- the uniqueness gate -------------------------------------------------

def test_the_same_number_in_two_records_refuses_as_ambiguous(rebind_on):
    """Two calls publish 6.42 as sleep_asleep's 30d mean. Which one the claim
    means is unknowable, and the gate declines rather than guesses."""
    ledger = [record for record in _ledger() if record["sequence"] != 2]
    ledger += [_series_record(21, "sleep_asleep", 6.42),
               _series_record(22, "sleep_asleep", 6.42)]

    verdict = DV._resolve_ledger_value(
        ledger, _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is False
    assert verdict["reason"] == "ambiguous value rebind"
    assert sorted(verdict["sequences"]) == [21, 22]
    assert "resolved_by" not in verdict
    assert "resolved_sequence" not in verdict


def test_ambiguity_is_not_broken_by_proximity_to_the_claimed_sequence(rebind_on):
    """Sequence 12 is adjacent to the claimed 13 and sequence 2 is far from it.

    A nearest-sequence tie-break would bind 12. There is no tie-break.
    """
    ledger = _ledger() + [_series_record(12, "sleep_asleep", 6.42)]

    verdict = DV._resolve_ledger_value(
        ledger, _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is False
    assert verdict["reason"] == "ambiguous value rebind"
    assert sorted(verdict["sequences"]) == [2, 12]


def test_two_matching_entries_inside_one_record_are_also_ambiguous(rebind_on):
    """The gate counts candidate ENTRIES, not candidate records: one call that
    publishes the number twice under the same label is just as unknowable."""
    ledger = [record for record in _ledger() if record["sequence"] != 2]
    doubled = _series_record(21, "sleep_asleep", 6.42)
    doubled["result"]["nested"] = {"mean": 6.42}
    ledger.append(doubled)

    verdict = DV._resolve_ledger_value(
        ledger, _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is False
    assert verdict["reason"] == "ambiguous value rebind"
    assert verdict["sequences"] == [21, 21]
    assert sorted(verdict["paths"]) == ["$.result.mean", "$.result.nested.mean"]


# --- what the rebind must still refuse -----------------------------------

def test_a_value_published_nowhere_still_fails(rebind_on):
    """A fabricated figure matches no entry, so the original verdict stands."""
    claim = _claim(metric="sleep_asleep", value=9.99, sequence=13)

    verdict = DV._resolve_ledger_value(_ledger(), claim)

    assert verdict["ok"] is False
    assert verdict["reason"] == "ledger path not found"
    assert "resolved_by" not in verdict


def test_zero_matches_returns_the_original_verdict_unchanged(rebind_on,
                                                             monkeypatch):
    """Not merely "still fails" — the SAME dict the flag-off arm produced."""
    ledger = _ledger()
    claim = _claim(metric="sleep_asleep", value=9.99, sequence=2,
                   path="$.result.mean")

    monkeypatch.delenv(REBIND, raising=False)
    off = DV._resolve_ledger_value(ledger, claim)
    monkeypatch.setenv(REBIND, "1")
    on = DV._resolve_ledger_value(ledger, claim)

    assert on == off
    assert off["reason"] == "claim value does not match ledger field"


def test_the_same_value_under_a_different_metric_label_still_fails(rebind_on):
    """6.42 is real, but it is sleep_asleep's mean and nothing else's.

    No case folding, no alias table: the record that publishes 6.42 does not
    carry `sleep_time_in_bed`, and the record that does publishes 7.31.
    """
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_time_in_bed", value=6.42, sequence=13))

    assert verdict["ok"] is False
    assert "resolved_by" not in verdict


def test_a_wrong_period_is_not_rescued(rebind_on):
    """The published period vocabulary is compared exactly, as before."""
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13,
                          period="90d"))

    assert verdict["ok"] is False
    assert "resolved_by" not in verdict


def test_a_wrong_field_is_not_rescued(rebind_on):
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13,
                          field="median"))

    assert verdict["ok"] is False
    assert "resolved_by" not in verdict


def test_a_near_miss_value_is_not_rescued_by_any_tolerance(rebind_on):
    """`_close`'s 0.5% relative tolerance would accept 6.44. Exact equality is
    the comparison here and it stays exact."""
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.44, sequence=13))

    assert verdict["ok"] is False
    assert "resolved_by" not in verdict


def test_list_workouts_row_claim_naming_a_metric_still_fails(rebind_on):
    """The metric-must-be-omitted rule survives the rebind.

    The control is what makes the refusal meaningful: the identical claim with
    `metric` omitted DOES rebind, so the failure above is the metric rule doing
    its job rather than the value simply being unfindable.
    """
    ledger, _key = _workout_ledger()
    common = dict(value=68.8, sequence=13, field="duration_min", period=None)

    metric_bearing = DV._resolve_ledger_value(
        ledger, _claim(metric="jog_minutes", **common))
    metricless = DV._resolve_ledger_value(ledger, _claim(metric=None, **common))

    assert metric_bearing["ok"] is False
    assert "resolved_by" not in metric_bearing

    assert metricless["ok"] is True
    assert metricless["resolved_by"] == "value_rebind"
    assert metricless["resolved_sequence"] == 5
    assert metricless["scope"] == "workout"
    assert metricless["tier"] == "path"
    assert metricless["path"] == "$.result.workouts[0].duration_min"


def test_an_elided_record_is_never_a_rebind_candidate(rebind_on):
    """The rebind skips elided records; it does not read through them."""
    ledger = _ledger()
    ledger[0]["result"] = {"_elided": True}
    ledger[0]["result_elided"] = True

    verdict = DV._resolve_ledger_value(
        ledger, _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is False
    assert "resolved_by" not in verdict


def test_the_argument_path_refusal_is_upstream_and_unmoved(rebind_on):
    verdict = DV._resolve_ledger_value(_ledger(), _claim(
        metric=None, value=6.42, sequence=13, field="metric", period=None,
        path="$.arguments.metric"))

    assert verdict["ok"] is False
    assert verdict["reason"].startswith(
        "claim cites a tool argument, not a result:")


def test_a_tool_argument_is_not_a_rebind_candidate(rebind_on):
    """Ignoring the pointer must not make arguments reachable for the first
    time. `$.arguments...` is refused by path today, and a model's own tool
    input is not evidence for the model's own claim."""
    ledger = _ledger()
    ledger[1]["arguments"] = {"metric": "sleep_time_in_bed", "days": 30}

    verdict = DV._resolve_ledger_value(ledger, _claim(
        metric=None, value=30, sequence=13, field="days", period=None))

    assert verdict["ok"] is False
    assert "resolved_by" not in verdict


def test_a_claim_with_no_source_is_still_refused(rebind_on):
    verdict = DV._resolve_ledger_value(
        _ledger(), {"metric": "sleep_asleep", "period": "30d",
                    "field": "mean", "value": 6.42})

    assert verdict["ok"] is False
    assert verdict["reason"] == "ledger claim has no source"


# --- the flag-off arm ----------------------------------------------------

@pytest.mark.parametrize("claim_kwargs, reason", [
    (dict(metric="sleep_asleep", value=6.42, sequence=13),
     "ledger path not found"),
    (dict(metric="sleep_asleep", value=6.42, sequence=13,
          path="$.result.mean"),
     "claim metric does not match ledger field"),
    (dict(metric="sleep_asleep", value=9.99, sequence=2,
          path="$.result.mean"),
     "claim value does not match ledger field"),
    (dict(metric="sleep_asleep", value=6.42, sequence=99),
     "ledger sequence not found"),
    (dict(metric="sleep_asleep", value=6.42, sequence=2, field="median",
          path="$.result.mean"),
     "claim field does not match ledger path"),
])
def test_flag_off_behaviour_is_unchanged(claim_kwargs, reason):
    verdict = DV._resolve_ledger_value(_ledger(), _claim(**claim_kwargs))

    assert verdict["ok"] is False
    assert verdict["reason"] == reason
    assert "resolved_by" not in verdict
    assert "claimed_path" not in verdict


def test_flag_off_still_accepts_an_honestly_cited_claim():
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=2,
                          path="$.result.mean"))

    assert verdict["ok"] is True
    assert verdict["tier"] == "metric"
    assert "resolved_by" not in verdict


@pytest.mark.parametrize("raw", ["0", "", "no", "off", "false", "  "])
def test_the_flag_is_off_for_every_falsey_spelling(monkeypatch, raw):
    monkeypatch.setenv(REBIND, raw)

    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is False


def test_flag_off_verdicts_are_byte_identical_across_a_claim_battery(
        monkeypatch):
    """Every claim in this file's vocabulary, resolved with the flag absent and
    then with the flag explicitly falsey: the dicts must be equal."""
    ledger = _ledger()
    claims = [
        _claim(metric="sleep_asleep", value=6.42, sequence=13),
        _claim(metric="sleep_asleep", value=6.42, sequence=2,
               path="$.result.mean"),
        _claim(metric="sleep_time_in_bed", value=6.42, sequence=13),
        _claim(metric="sleep_asleep", value=9.99, sequence=13),
        _claim(metric="sleep_asleep", value=6.42, sequence=13, period="90d"),
    ]

    monkeypatch.delenv(REBIND, raising=False)
    absent = [DV._resolve_ledger_value(ledger, claim) for claim in claims]
    monkeypatch.setenv(REBIND, "0")
    falsey = [DV._resolve_ledger_value(ledger, claim) for claim in claims]

    assert absent == falsey


def test_the_public_coach_verdict_gains_no_key_with_the_flag_off():
    """`/v1/ask`'s verification shape is a gated public contract. With Method C
    off it must not grow a field."""
    verdict = DV.verify_coach_claims(
        None, "Your sleep averaged 6.42 hours.",
        [_claim(metric="sleep_asleep", value=6.42, sequence=2,
                path="$.result.mean")],
        payload=_ledger())

    assert "rebind_counts" not in verdict


# --- interaction with Method B -------------------------------------------

def test_method_b_wins_when_both_flags_are_on(monkeypatch):
    """B is pointer-preserving and stricter, so it is tried first. A claim B
    can resolve is marked `search`, exactly as with B alone."""
    monkeypatch.setenv(SEARCH, "1")
    monkeypatch.setenv(REBIND, "1")

    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13,
                          path="$.result.mean"))

    assert verdict["ok"] is True
    assert verdict["resolved_by"] == "search"
    assert verdict["resolved_sequence"] == 2
    assert "claimed_path" not in verdict


def test_method_c_picks_up_what_method_b_cannot_when_both_are_on(monkeypatch):
    monkeypatch.setenv(SEARCH, "1")
    monkeypatch.setenv(REBIND, "1")

    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is True
    assert verdict["resolved_by"] == "value_rebind"
    assert verdict["resolved_sequence"] == 2


def test_method_c_alone_covers_method_bs_case_too(rebind_on):
    """C ignores the path, so the wrong-sequence/right-path claim rebinds under
    C with B off — the two flags are genuinely independent."""
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13,
                          path="$.result.mean"))

    assert verdict["ok"] is True
    assert verdict["resolved_by"] == "value_rebind"
    assert verdict["resolved_sequence"] == 2


def test_method_bs_ambiguity_is_not_reopened_by_method_c(monkeypatch):
    """When B refuses two path-matching records, C is not consulted: it would
    see the same two candidates and refuse them too, and B's reason is the more
    precise diagnosis. The refusal must not be relabelled."""
    monkeypatch.setenv(SEARCH, "1")
    ledger = [record for record in _ledger() if record["sequence"] != 2]
    ledger += [_series_record(21, "sleep_asleep", 6.42),
               _series_record(22, "sleep_asleep", 6.42)]
    claim = _claim(metric="sleep_asleep", value=6.42, sequence=13,
                   path="$.result.mean")

    b_only = DV._resolve_ledger_value(ledger, claim)
    monkeypatch.setenv(REBIND, "1")
    both = DV._resolve_ledger_value(ledger, claim)

    assert b_only["reason"] == "ambiguous ledger record"
    assert both == b_only


def test_an_ambiguous_path_in_the_named_record_is_not_relabelled(rebind_on,
                                                                 monkeypatch):
    """The named record itself publishes the claimed path twice.

    That is already "the evidence is not unique", and a wider candidate set
    cannot make it unique. Method C must not overwrite the sharper diagnosis
    with its own — the claim is refused either way, but only one of the two
    reasons tells you where to look.
    """
    ledger = [{
        "sequence": 1, "tool_name": "synthetic", "arguments": {},
        "result": {"value": 68.8}, "result_elided": False,
    }]
    entry = {"metric": None, "period": None, "field": "value",
             "value": 68.8, "workout_key": None,
             "path": "$.result.value", "kind": "result"}
    monkeypatch.setattr(DV, "_ledger_scopes", lambda record: [entry, entry])
    claim = {"metric": None, "period": None, "field": "value", "value": 68.8,
             "source": {"sequence": 1, "path": "$.result.value"}}

    verdict = DV._resolve_ledger_value(ledger, claim)

    assert verdict["ok"] is False
    assert verdict["reason"] == "ambiguous ledger path"


def test_both_flags_on_still_refuses_an_ambiguous_rebind(monkeypatch):
    monkeypatch.setenv(SEARCH, "1")
    monkeypatch.setenv(REBIND, "1")
    ledger = [record for record in _ledger() if record["sequence"] != 2]
    ledger += [_series_record(21, "sleep_asleep", 6.42),
               _series_record(22, "sleep_asleep", 6.42)]

    verdict = DV._resolve_ledger_value(
        ledger, _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is False
    assert verdict["reason"] == "ambiguous value rebind"
    assert sorted(verdict["sequences"]) == [21, 22]


# --- instrumentation -----------------------------------------------------

def _research(prose: str, claims: list[dict], ledger: list[dict]) -> dict:
    return DV.verify_research_claims(prose, claims, ledger)


def test_a_rebound_claim_is_counted_and_flagged_as_sequence_changed(rebind_on):
    verdict = _research(
        "Your sleep averaged 6.42 hours.",
        [_claim(metric="sleep_asleep", value=6.42, sequence=13)], _ledger())

    assert verdict["rebind_counts"] == {
        "rebound": 1, "ambiguous": 0, "sequence_changed": 1}


def test_a_leaf_only_correction_is_counted_but_not_sequence_changed(rebind_on):
    """The number that decides whether this is safe to leave on is how often
    Python moved the citation to ANOTHER call. Correcting the leaf inside the
    call the model already named is a much weaker claim, and is counted apart."""
    verdict = _research(
        "Your sleep averaged 6.42 hours.",
        [_claim(metric="sleep_asleep", value=6.42, sequence=2)], _ledger())

    assert verdict["rebind_counts"] == {
        "rebound": 1, "ambiguous": 0, "sequence_changed": 0}


def test_an_ambiguous_rebind_is_counted_as_a_refusal(rebind_on):
    ledger = [record for record in _ledger() if record["sequence"] != 2]
    ledger += [_series_record(21, "sleep_asleep", 6.42),
               _series_record(22, "sleep_asleep", 6.42)]

    verdict = _research(
        "Your sleep averaged 6.42 hours.",
        [_claim(metric="sleep_asleep", value=6.42, sequence=13)], ledger)

    assert verdict["ok"] is False
    assert verdict["reason"] == "ambiguous value rebind"
    assert verdict["rebind_counts"] == {
        "rebound": 0, "ambiguous": 1, "sequence_changed": 0}


def test_an_honestly_cited_claim_reports_zero_rebinds(rebind_on):
    """The denominator is visible: the counters appear whenever the flag is on,
    so a run with no rebinds is distinguishable from a run with no reporting."""
    verdict = _research(
        "Your sleep averaged 6.42 hours.",
        [_claim(metric="sleep_asleep", value=6.42, sequence=2,
                path="$.result.mean")], _ledger())

    assert verdict["rebind_counts"] == {
        "rebound": 0, "ambiguous": 0, "sequence_changed": 0}


def test_the_counters_are_absent_with_the_flag_off():
    verdict = _research(
        "Your sleep averaged 6.42 hours.",
        [_claim(metric="sleep_asleep", value=6.42, sequence=2,
                path="$.result.mean")], _ledger())

    assert "rebind_counts" not in verdict


def test_the_flag_on_verdict_gains_exactly_one_key(monkeypatch):
    """The public `/v1/ask` verification shape is gated by an exact key set in
    test_chat.py. Turning Method C on adds `rebind_counts` and nothing else;
    this test is what makes that a declared field rather than a leak."""
    args = (None, "Your sleep averaged 6.42 hours.",
            [_claim(metric="sleep_asleep", value=6.42, sequence=2,
                    path="$.result.mean")])

    off = DV.verify_coach_claims(*args, payload=_ledger())
    monkeypatch.setenv(REBIND, "1")
    on = DV.verify_coach_claims(*args, payload=_ledger())

    assert set(on) - set(off) == {"rebind_counts"}
    assert {key: on[key] for key in off} == off


# --- the two contracts Method C deliberately changes ----------------------
#
# Both are asserted as refusals elsewhere in the suite, pinned to the flag-off
# arm. They are restated here with the flag on so the change is documented in
# one place rather than discovered by whoever flips the flag.

def test_a_live_claim_naming_an_absent_leaf_now_binds(rebind_on):
    """`test_researcher_claim_channel.py` pins this as a refusal with the flag
    off. It is the wrong-leaf case on a REAL recorded ledger: 50.1 is the
    recent block's mean under that metric and period, and the model pointed at
    `.recent.missing`."""
    fixture = (Path(__file__).parent
               / "fixtures/jog_ledger_live_20260824_claims.jsonl")
    ledger = [json.loads(fixture.read_text(encoding="utf-8"))]
    blocks = ledger[0]["result"]["block_comparison"]["blocks"]
    claim = {
        "metric": "jog_minutes", "period": blocks["recent"]["period"],
        "field": "mean", "value": 50.1,
        "source": {"sequence": 1,
                   "path": "$.result.block_comparison.blocks.recent.missing"},
    }

    verdict = DV._resolve_ledger_value(ledger, claim)

    assert verdict["ok"] is True
    assert verdict["resolved_by"] == "value_rebind"
    assert verdict["path"] == "$.result.block_comparison.blocks.recent.mean"


def test_a_right_value_behind_a_wrong_pointer_binds_and_that_is_the_point(
        rebind_on):
    """The suite's metricless false-accept check includes `field=score,
    value=42, path=$.result.other`, refused with the flag off.

    With Method C on it binds — and it should. The claim asserts that `score`
    is 42, and `score` IS 42. Only the pointer was wrong, which is the entire
    thing this method exists to stop punishing. The sibling cases, where the
    VALUE is wrong, still fail.
    """
    ledger = [{
        "sequence": 1, "tool_name": "synthetic",
        "arguments": {"answer": 42},
        "result": {"score": 42, "other": 7},
        "result_elided": False,
    }]

    def verdict_for(value, path):
        return DV.verify_number(None, {
            "metric": None, "period": None, "field": "score", "value": value,
            "source": {"sequence": 1, "path": path},
        }, payload=ledger)

    wrong_pointer = verdict_for(42, "$.result.other")
    wrong_value = verdict_for(43, "$.result.score")
    invented_value = verdict_for(99, "$.result.score")

    assert wrong_pointer["ok"] is True
    assert wrong_pointer["resolved_by"] == "value_rebind"
    assert wrong_value["ok"] is False
    assert invented_value["ok"] is False


def test_the_coach_verdict_carries_the_counters_too(rebind_on):
    verdict = DV.verify_coach_claims(
        None, "Your sleep averaged 6.42 hours.",
        [_claim(metric="sleep_asleep", value=6.42, sequence=13)],
        payload=_ledger())

    assert verdict["rebind_counts"] == {
        "rebound": 1, "ambiguous": 0, "sequence_changed": 1}
    # Tier accounting is untouched by the rebind: this is still a tier-2 bind.
    assert verdict["tier_counts"] == {"path": 0, "metric": 1}
