"""Python, not the model, decides which ledger record a number came from.

Measured 2026-08-28 on the six-question /v1/ask battery: one answer filed 11
well-formed claims and 2 verified. The numbers were real and published verbatim
at real paths with the right field, metric and period — the model simply mixed
up which of its 13 tool calls had returned which, citing sequence 13 for
`sleep_time_in_bed` and sequence 2 for `sleep_asleep`.

`HA_ASK_LEDGER_RESOLVE=1` lets the resolver look for the record the number is
actually in. Everything these tests care about is that the search is a search
over *records* and nothing else: a fabricated number, a mislabelled metric, a
metric-bearing list_workouts row and an ambiguous double publication must all
still be refused, and with the flag off nothing may move at all.
"""
from __future__ import annotations

import pytest

from health_advisor import db as dbmod
from health_advisor import deepdive_verify as DV


FLAG = "HA_ASK_LEDGER_RESOLVE"
# Method C (`tests/test_ledger_value_rebind.py`) resolves several of the claims
# below by ignoring the pointer entirely. Every test in this file is about
# Method B in isolation, so it must hold its sibling flag down rather than
# inherit whatever the ambient environment happens to be exporting.
REBIND_FLAG = "HA_ASK_VALUE_REBIND"


@pytest.fixture(autouse=True)
def value_rebind_off(monkeypatch):
    monkeypatch.delenv(REBIND_FLAG, raising=False)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


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
    """Two sleep series calls — the exact shape the battery got wrong."""
    return [
        _series_record(2, "sleep_asleep", 6.42),
        _series_record(13, "sleep_time_in_bed", 7.31),
    ]


def _claim(*, metric, value, sequence, field="mean", period="30d",
           path="$.result.mean") -> dict:
    return {
        "metric": metric, "period": period, "field": field, "value": value,
        "source": {"sequence": sequence, "path": path},
    }


# --- the fix itself ------------------------------------------------------

def test_wrong_sequence_right_number_resolves_only_with_the_flag_on(monkeypatch):
    """The dominant measured failure: right number, wrong record cited."""
    ledger = _ledger()
    # sleep_asleep's mean IS published, at this path, with this field, metric
    # and period — but by sequence 2, not the sequence 13 the model named.
    claim = _claim(metric="sleep_asleep", value=6.42, sequence=13)

    monkeypatch.delenv(FLAG, raising=False)
    off = DV._resolve_ledger_value(ledger, claim)

    monkeypatch.setenv(FLAG, "1")
    on = DV._resolve_ledger_value(ledger, claim)

    assert off["ok"] is False
    assert off["reason"] == "claim metric does not match ledger field"
    assert "resolved_by" not in off

    assert on["ok"] is True
    assert on["resolved_by"] == "search"
    assert on["resolved_sequence"] == 2
    assert on["claimed_sequence"] == 13
    # Tier semantics are untouched: a metric-labelled claim is still tier
    # "metric", and the entry still carries the real published path.
    assert on["tier"] == "metric"
    assert on["path"] == "$.result.mean"
    assert on["value"] == 6.42


def test_correctly_cited_claim_is_untouched_by_the_search(flag_on):
    """The model's own citation short-circuits; no marker, no search."""
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=2))

    assert verdict["ok"] is True
    assert "resolved_by" not in verdict
    assert "resolved_sequence" not in verdict


def test_verify_number_carries_the_search_marker_up(flag_on):
    ledger = _ledger()
    claim = _claim(metric="sleep_asleep", value=6.42, sequence=13)

    verdict = DV.verify_number(None, claim, payload=ledger)

    assert verdict["ok"] is True
    assert verdict["tier"] == "metric"
    assert verdict["resolved_by"] == "search"
    assert verdict["resolved_sequence"] == 2


def test_verify_number_adds_no_marker_key_when_the_citation_was_right(flag_on):
    verdict = DV.verify_number(
        None, _claim(metric="sleep_asleep", value=6.42, sequence=2),
        payload=_ledger())

    assert verdict["ok"] is True
    assert "resolved_by" not in verdict
    assert "resolved_sequence" not in verdict


# --- what the search must still refuse -----------------------------------

def test_number_published_nowhere_still_fails_with_the_flag_on(flag_on):
    """A fabricated figure resolves in no record, so it resolves at all."""
    ledger = _ledger()

    named_wrong = DV._resolve_ledger_value(
        ledger, _claim(metric="sleep_asleep", value=9.99, sequence=13))
    named_right = DV._resolve_ledger_value(
        ledger, _claim(metric="sleep_asleep", value=9.99, sequence=2))

    assert named_wrong["ok"] is False
    assert "resolved_by" not in named_wrong
    assert named_right["ok"] is False
    assert named_right["reason"] == "claim value does not match ledger field"


def test_number_carrying_a_different_metric_label_still_fails(flag_on):
    """6.42 is real, but it is sleep_asleep's mean and nothing else's.

    Relabelling it `sleep_time_in_bed` must not be rescued by the search: the
    record that publishes 6.42 does not carry that metric, and the record that
    carries that metric does not publish 6.42.
    """
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_time_in_bed", value=6.42, sequence=2))

    assert verdict["ok"] is False
    assert "resolved_by" not in verdict


def test_number_resolvable_in_two_records_fails_as_ambiguous(flag_on):
    """Two publications of the same labelled value are not evidence for each
    other, exactly as two matching paths inside one record are not."""
    ledger = _ledger() + [_series_record(21, "sleep_asleep", 6.42)]
    ledger.append(_series_record(22, "sleep_asleep", 6.42))
    # Drop the honestly-citable record so the named sequence is the wrong one
    # and both remaining sleep_asleep records are candidates.
    ledger = [record for record in ledger if record["sequence"] != 2]

    verdict = DV._resolve_ledger_value(
        ledger, _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is False
    assert verdict["reason"] == "ambiguous ledger record"
    assert sorted(verdict["sequences"]) == [21, 22]


def test_list_workouts_row_claim_naming_a_metric_still_fails(flag_on):
    """The metric-must-be-omitted rule survives the record search.

    The control matters: the same claim without a metric DOES resolve by
    search, so the refusal below is the metric rule doing its job and not the
    path simply being unfindable.
    """
    workout_key = dbmod.workout_key(
        "running", "2026-08-15T12:00:00Z", "2026-08-15T13:08:48Z")
    ledger = _ledger() + [{
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
    }]
    path = "$.result.workouts[0].duration_min"

    metric_bearing = DV._resolve_ledger_value(ledger, _claim(
        metric="jog_minutes", value=68.8, sequence=13,
        field="duration_min", period=None, path=path))
    metricless = DV._resolve_ledger_value(ledger, _claim(
        metric=None, value=68.8, sequence=13,
        field="duration_min", period=None, path=path))

    assert metric_bearing["ok"] is False
    assert "resolved_by" not in metric_bearing
    assert metricless["ok"] is True
    assert metricless["resolved_by"] == "search"
    assert metricless["resolved_sequence"] == 5
    assert metricless["scope"] == "workout"
    assert metricless["tier"] == "path"


def test_argument_path_is_still_refused_with_the_flag_on(flag_on):
    """The $.arguments refusal is upstream of the search and stays there."""
    ledger = _ledger()
    verdict = DV._resolve_ledger_value(ledger, _claim(
        metric=None, value=6.42, sequence=13, field="metric",
        period=None, path="$.arguments.metric"))

    assert verdict["ok"] is False
    assert verdict["reason"].startswith(
        "claim cites a tool argument, not a result:")


def test_elided_result_with_no_other_copy_still_fails(flag_on):
    """The search skips elided records; it does not read through them."""
    ledger = _ledger()
    ledger[0]["result"] = {"_elided": True}
    ledger[0]["result_elided"] = True

    verdict = DV._resolve_ledger_value(
        ledger, _claim(metric="sleep_asleep", value=6.42, sequence=2))

    assert verdict["ok"] is False
    assert verdict["reason"] == "ledger result is elided"


def test_wrong_field_is_not_rescued_by_the_search(flag_on):
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13,
                          field="median"))

    assert verdict["ok"] is False
    assert "resolved_by" not in verdict


def test_wrong_period_is_not_rescued_by_the_search(flag_on):
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13,
                          period="90d"))

    assert verdict["ok"] is False
    assert "resolved_by" not in verdict


def test_unknown_sequence_with_no_matching_record_keeps_its_reason(flag_on):
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=9.99, sequence=99))

    assert verdict["ok"] is False
    assert verdict["reason"] == "ledger sequence not found"


# --- the flag-off arm ----------------------------------------------------

@pytest.mark.parametrize("claim_kwargs, reason", [
    (dict(metric="sleep_asleep", value=6.42, sequence=13),
     "claim metric does not match ledger field"),
    (dict(metric="sleep_asleep", value=9.99, sequence=2),
     "claim value does not match ledger field"),
    (dict(metric="sleep_asleep", value=6.42, sequence=99),
     "ledger sequence not found"),
    (dict(metric="sleep_asleep", value=6.42, sequence=13, field="median"),
     "claim field does not match ledger path"),
    (dict(metric="sleep_asleep", value=6.42, sequence=13,
          path="$.result.absent"),
     "ledger path not found"),
])
def test_flag_off_behaviour_is_unchanged(flag_off, claim_kwargs, reason):
    verdict = DV._resolve_ledger_value(_ledger(), _claim(**claim_kwargs))

    assert verdict["ok"] is False
    assert verdict["reason"] == reason
    assert "resolved_by" not in verdict


def test_flag_off_still_accepts_an_honestly_cited_claim(flag_off):
    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=2))

    assert verdict["ok"] is True
    assert verdict["tier"] == "metric"


@pytest.mark.parametrize("raw", ["0", "", "no", "off", "false"])
def test_flag_is_off_for_every_falsey_spelling(monkeypatch, raw):
    monkeypatch.setenv(FLAG, raw)

    verdict = DV._resolve_ledger_value(
        _ledger(), _claim(metric="sleep_asleep", value=6.42, sequence=13))

    assert verdict["ok"] is False
