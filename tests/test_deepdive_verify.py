"""Regression tests for result-local metric ownership in ledger claims."""

from health_advisor import deepdive_verify as DV


def _claim(metric, period, field, value, path):
    return {
        "metric": metric,
        "period": period,
        "field": field,
        "value": value,
        "source": {"sequence": 1, "path": path},
    }


def _record(result, tool_name="test_tool"):
    return [{
        "sequence": 1,
        "tool_name": tool_name,
        "arguments": {},
        "result": result,
        "result_elided": False,
    }]


def test_points_value_is_owned_by_the_enclosing_result_metric():
    ledger = _record({
        "metric": "sleep_asleep",
        "period": "2026-08-20",
        "points": [{"date": "2026-08-20", "value": 439.69}],
    })
    claim = _claim("sleep_asleep", "2026-08-20", "value", 439.69,
                   "$.result.points[0].value")

    verdict = DV.verify_coach_claims(
        None, "I slept 439.69 minutes.", [claim], payload=ledger)

    assert verdict["ok"] is True, verdict
    assert verdict["tier_counts"] == {"path": 0, "metric": 1}
    point = next(entry for entry in DV._ledger_scopes(ledger[0])
                 if entry["path"] == "$.result.points[0].value")
    assert point["metric"] == "sleep_asleep"


def test_surplus_metric_on_result_context_field_verifies_at_path_tier():
    period = "2026-08-10:2026-08-16"
    ledger = _record({
        "metric": "sleep_asleep",
        "weeks": [{"period": period, "mean": 461.57, "n_days": 7}],
    })
    claim = _claim("sleep_asleep", period, "n_days", 7,
                   "$.result.weeks[0].n_days")

    verdict = DV.verify_coach_claims(
        None, "There were 7 recorded days.", [claim], payload=ledger)

    assert verdict["ok"] is True, verdict
    assert verdict["tier_counts"] == {"path": 1, "metric": 0}
    context = next(entry for entry in DV._ledger_scopes(ledger[0])
                   if entry["path"] == "$.result.weeks[0].n_days")
    assert context["metric"] is None


def test_surplus_context_label_does_not_bypass_path_or_value_checks():
    period = "2026-08-10:2026-08-16"
    ledger = _record({
        "metric": "sleep_asleep",
        "weeks": [{"period": period, "mean": 461.57, "n_days": 7}],
    })
    wrong_value = _claim("sleep_asleep", period, "n_days", 6,
                         "$.result.weeks[0].n_days")
    wrong_path = _claim("sleep_asleep", period, "n_days", 7,
                        "$.result.weeks[0].missing")

    assert DV._resolve_ledger_value(ledger, wrong_value)["reason"] == (
        "claim value does not match ledger field")
    assert DV._resolve_ledger_value(ledger, wrong_path)["reason"] == (
        "ledger path not found")


def test_points_value_claim_with_a_different_metric_still_refuses():
    ledger = _record({
        "metric": "sleep_asleep",
        "period": "2026-08-20",
        "points": [{"date": "2026-08-20", "value": 439.69}],
    })
    claim = _claim("sleep_time_in_bed", "2026-08-20", "value", 439.69,
                   "$.result.points[0].value")

    verdict = DV.verify_coach_claims(
        None, "I slept 439.69 minutes.", [claim], payload=ledger)

    assert verdict["ok"] is False
    assert verdict["reason"] == "claim metric does not match ledger field"


def test_value_is_not_owned_outside_a_points_array():
    ledger = _record({
        "metric": "sleep_asleep",
        "period": "2026-08-20",
        "value": 439.69,
    })
    claim = _claim("sleep_asleep", "2026-08-20", "value", 439.69,
                   "$.result.value")

    verdict = DV._resolve_ledger_value(ledger, claim)

    assert verdict["ok"] is False
    assert verdict["reason"] == "claim metric does not match ledger field"


def test_unlabelled_legacy_readiness_score_refuses_and_the_harness_bridges_it():
    """The verifier grants nothing a publisher did not label (one source of
    truth); the replay harness upgrades pre-#202 captures to the modern shape
    instead. Both halves are the contract."""
    from tests.replay_ask_captures import _bridge_legacy_labels

    ledger = _record({"readiness": {"score": 69}}, "get_briefing")
    claim = _claim("readiness", "2026-08-20", "score", 69,
                   "$.result.readiness.score")

    refused = DV.verify_coach_claims(
        None, "Readiness was 69.", [claim], payload=ledger)
    assert refused["ok"] is False
    assert refused["reason"] == "claim metric does not match ledger field"

    for record in ledger:
        _bridge_legacy_labels(record)
    verdict = DV.verify_coach_claims(
        None, "Readiness was 69.", [claim], payload=ledger)
    assert verdict["ok"] is True, verdict


def test_legacy_readiness_components_sleep_bridges_through_the_harness():
    from tests.replay_ask_captures import _bridge_legacy_labels

    ledger = _record({"readiness": {"components": {"sleep": 100}}},
                     "get_briefing")
    claim = _claim("readiness", "2026-08-20", "sleep", 100,
                   "$.result.readiness.components.sleep")

    assert DV.verify_coach_claims(
        None, "The readiness sleep component was 100.", [claim],
        payload=ledger)["ok"] is False

    for record in ledger:
        _bridge_legacy_labels(record)
    verdict = DV.verify_coach_claims(
        None, "The readiness sleep component was 100.", [claim], payload=ledger)
    assert verdict["ok"] is True, verdict


def test_legacy_readiness_factor_current_bridges_through_the_harness():
    from tests.replay_ask_captures import _bridge_legacy_labels

    ledger = _record({"readiness": {"factors": [
        {"component": "sleep", "current": 475.5}] }}, "get_briefing")
    claim = _claim("sleep_asleep", "2026-08-20", "current", 475.5,
                   "$.result.readiness.factors[0].current")

    assert DV.verify_coach_claims(
        None, "I slept 475.5 minutes.", [claim], payload=ledger)["ok"] is False

    for record in ledger:
        _bridge_legacy_labels(record)
    verdict = DV.verify_coach_claims(
        None, "I slept 475.5 minutes.", [claim], payload=ledger)
    assert verdict["ok"] is True, verdict


def test_legacy_midpoint_variability_bridges_through_the_harness():
    from tests.replay_ask_captures import _bridge_legacy_labels

    ledger = _record({"midpoint_variability": {"latest_sd_hours": 1.019}},
                     "get_sleep_regularity")
    claim = _claim("sleep_midpoint_sd_28d", "2026-08-20", "latest_sd_hours",
                   1.019, "$.result.midpoint_variability.latest_sd_hours")

    assert DV.verify_coach_claims(
        None, "Sleep midpoint variability was 1.019 hours.", [claim],
        payload=ledger)["ok"] is False

    for record in ledger:
        _bridge_legacy_labels(record)
    verdict = DV.verify_coach_claims(
        None, "Sleep midpoint variability was 1.019 hours.", [claim],
        payload=ledger)
    assert verdict["ok"] is True, verdict


def test_legacy_impact_sensitivity_jog_minutes_bridges_through_the_harness():
    from tests.replay_ask_captures import _bridge_legacy_labels

    ledger = _record({"jog_threshold_sensitivity": [{
        "cadence_min_steps_per_min": 140.0,
        "jog_buckets": 37,
        "jog_minutes": 12.3,
        "live_cutoff": True,
    }]}, "get_impact_volume")
    claim = _claim("jog_minutes", None, "jog_minutes", 12.3,
                   "$.result.jog_threshold_sensitivity[0].jog_minutes")

    assert DV.verify_coach_claims(
        None, "Jogging was 12.3 minutes.", [claim], payload=ledger)["ok"] is False

    for record in ledger:
        _bridge_legacy_labels(record)
    verdict = DV.verify_coach_claims(
        None, "Jogging was 12.3 minutes.", [claim], payload=ledger)
    assert verdict["ok"] is True, verdict

    context_claim = _claim("jog_minutes", None, "cadence_min_steps_per_min", 140.0,
                           "$.result.jog_threshold_sensitivity[0]."
                           "cadence_min_steps_per_min")
    refused = DV.verify_coach_claims(
        None, "The cutoff was 140 steps per minute.", [context_claim],
        payload=ledger)
    assert refused["ok"] is False
    assert refused["reason"] == "claim metric does not match ledger field"


def test_metricless_workout_count_does_not_accept_a_surplus_metric_label():
    ledger = _record({"workout_counts": [{"type": "running", "count": 5}]})
    claim = _claim("running", None, "count", 5,
                   "$.result.workout_counts[0].count")

    verdict = DV._resolve_ledger_value(ledger, claim)

    assert verdict["ok"] is False
    assert verdict["reason"] == "claim metric does not match ledger field"
