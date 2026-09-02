from __future__ import annotations

import copy
import json
from pathlib import Path

from health_advisor import chat
from health_advisor import deepdive_verify as DV


FIXTURES = Path(__file__).parent / "fixtures"


def _captured() -> tuple[dict, list[dict]]:
    answer = json.loads((FIXTURES / "jog_answer_live_20260824.json").read_text())
    ledger = [json.loads(line) for line in
              (FIXTURES / "jog_ledger_live_20260824_claims.jsonl").read_text().splitlines()
              if line.strip()]
    return answer, ledger


def test_captured_running_answer_verifies_with_a_source_backed_structural_claim():
    answer, ledger = _captured()

    verdict = chat._verify_ask_answer(
        None, answer["text"], answer["claims"], ledger)

    assert verdict["ok"] is True, verdict
    assert verdict["figures_verified"] == 3
    assert verdict["figures_total"] == 3
    assert verdict["unsupported"] == []
    assert verdict["structural_claims"] == [{
        "metric": "jog_minutes",
        "period": None,
        "field": "weeks_per_block",
        "value": 4,
        "source": {
            "sequence": 1,
            "path": "$.result.block_comparison.weeks_per_block",
        },
    }]


def test_arguments_sourced_reproducer_is_refused_with_argument_path_reason():
    _, ledger = _captured()
    claim = {
        "metric": "jog_minutes",
        "period": "2026-07-27:2026-08-23",
        "field": "weeks_per_block",
        "value": 4,
        "source": {"sequence": 1, "path": "$.arguments.weeks_per_block"},
    }

    verdict = chat._verify_ask_answer(
        None,
        "You completed 4 hard sessions and your longest run was 4 miles.",
        [claim],
        ledger,
    )

    assert verdict["ok"] is False
    assert verdict["reason"] == (
        "claim cites a tool argument, not a result: $.arguments.weeks_per_block"
    )


def test_arguments_sourced_claims_cannot_launder_metric_or_period():
    _, ledger = _captured()
    base_claim = {
        "metric": "jog_minutes",
        "period": "2026-07-27:2026-08-23",
        "field": "weeks_per_block",
        "value": 4,
        "source": {"sequence": 1, "path": "$.arguments.weeks_per_block"},
    }

    verdicts = []
    for changes in (
        {"metric": "steps"},
        {"period": "1999-01-01:1999-01-02"},
    ):
        claim = {**base_claim, **changes}
        verdicts.append(chat._verify_ask_answer(
            None,
            "You completed 4 hard sessions and your longest run was 4 miles.",
            [claim],
            ledger,
        ))

    assert [verdict["ok"] for verdict in verdicts] == [False, False]
    assert [verdict["reason"] for verdict in verdicts] == [
        "claim cites a tool argument, not a result: $.arguments.weeks_per_block",
        "claim cites a tool argument, not a result: $.arguments.weeks_per_block",
    ]


def test_wrong_direction_battery_refuses_all_one_hundred_answers():
    answer, ledger = _captured()
    prose = answer["text"].replace("a decrease of", "an increase of")

    refused = sum(
        not chat._verify_ask_answer(
            None, prose, answer["claims"], ledger)["ok"]
        for _ in range(100)
    )

    assert refused == 100


def test_wrong_value_battery_accepts_zero_of_one_hundred_one_answers():
    answer, ledger = _captured()
    accepted = 0
    for index in range(101):
        claims = copy.deepcopy(answer["claims"])
        claims[-1]["value"] = index + 0.2
        accepted += chat._verify_ask_answer(
            None, answer["text"], claims, ledger)["ok"]

    assert accepted == 0


def test_rule_r_fixture_fabrication_table_refuses_every_unlicensed_perturbation():
    answer, ledger = _captured()
    expected = {
        "50.4": False,
        "50.5": False,
        "50.6": False,
        "51.0": False,
        "55.0": False,
        "50.3": False,
        "50.15": False,
        "50.1": True,
        "50.2": True,
        "50": True,
    }

    observed = {}
    for token in expected:
        prose = answer["text"].replace("50.1", token, 1)
        observed[token] = chat._verify_ask_answer(
            None, prose, answer["claims"], ledger)["ok"]

    assert observed == expected


def test_rule_r_rounding_table_and_one_end_to_end_case():
    cases = [
        ("50.1", 50.14, True),
        ("50.1", 50.16, False),
        ("3", 3.4, True),
        ("3", 2.4, False),
        ("987", 987.3, True),
        ("988", 987.3, False),
    ]
    assert [DV._rule_r_matches(token, claim) for token, claim, _ in cases] == [
        expected for _, _, expected in cases]

    ledger, claim = _synthetic_claim(value=50.14)
    verdict = chat._verify_ask_answer(None, "Average 50.1.", [claim], ledger)
    assert verdict["ok"] is True


def test_rule_r_near_zero_integer_floor_and_direction_bridge():
    cases = [("0", 0.49, False), ("1", 0.5, False),
             ("1", 1.4, True)]
    assert [DV._rule_r_matches(token, claim) for token, claim, _ in cases] == [
        expected for _, _, expected in cases]

    ledger, claim = _synthetic_claim(value=-0.1, field="delta")
    verdict = chat._verify_ask_answer(
        None, "There was a decrease of 0.1.", [claim], ledger)
    assert verdict["ok"] is True

    answer, fixture_ledger = _captured()
    verdict = chat._verify_ask_answer(
        None, "The result is 0.", answer["claims"], fixture_ledger)
    assert verdict["ok"] is False


def test_rule_r_value_domain_ignores_source_sequence_and_keeps_string_values():
    answer, ledger = _captured()
    verdict = chat._verify_ask_answer(
        None, "The result is 1.", answer["claims"], ledger)
    assert verdict["ok"] is False

    string_ledger, string_claim = _synthetic_claim(
        value="987 kcal", field="max", metric="active_energy")
    verdict = chat._verify_ask_answer(
        None, "Active energy peaked at 987 kcal.", [string_claim],
        string_ledger)
    assert verdict["ok"] is True


def test_rule_r_token_normalization_handles_sentence_final_comma_grouping():
    ledger, claim = _synthetic_claim(value=1000, metric="active_energy")
    verdict = chat._verify_ask_answer(
        None, "Active energy peaked at 1,000.", [claim], ledger)
    assert verdict["ok"] is True

    ledger, claim = _synthetic_claim(value=1400, metric="active_energy")
    verdict = chat._verify_ask_answer(
        None, "Active energy peaked at 1,000.", [claim], ledger)
    assert verdict["ok"] is False

    assert DV._rule_r_matches("1,000.", 1000)


def test_non_ledger_path_keeps_proximity_until_61():
    payload = [{"metric": "jog_minutes", "period": "30d",
                "field": "mean", "value": 50.2}]
    claim = {key: payload[0][key] for key in DV.SCOPED_CLAIM_FIELDS}
    verdict = DV.verify_coach_claims(
        None, "Average 50.4.", [claim], payload=payload)
    assert verdict["ok"] is True


def _derived_claim(*, value, recent, prior):
    """A `difference` claim over two scoped operands, in both payload shapes."""
    facts = [{"metric": "jog_minutes", "period": "recent",
              "field": "mean", "value": recent},
             {"metric": "jog_minutes", "period": "prior",
              "field": "mean", "value": prior}]
    ledger = [{"sequence": 1, "tool_name": "synthetic", "arguments": {},
               "result": {"facts": facts}, "result_elided": False}]
    operands = [dict(fact, source={"sequence": 1,
                                   "path": f"$.result.facts[{i}].value"})
                for i, fact in enumerate(facts)]
    claim = {"metric": "jog_minutes", "period": "recent",
             "field": "difference", "value": value,
             "operation": "difference", "operands": operands}
    return facts, ledger, claim


def test_derived_non_ledger_path_keeps_proximity_until_61():
    """#89 tightened the ledgered derivation only. The coach path is live here —
    coach_brief asks for `operation` and scoped operands (coach_brief.py:1178) —
    so a derivation on the non-ledger payload must keep `_close` until #61."""
    facts, _, claim = _derived_claim(value=1000, recent=1104.0, prior=100.0)

    verdict = DV.verify_number(None, claim, payload=facts)

    # actual is 1004.0; _close licenses it (|4| <= 1004 * 0.005), rule R does not.
    assert verdict["actual"] == 1004.0
    assert verdict["ok"] is True


def test_derived_ledger_path_refuses_what_proximity_allowed():
    """The same numbers through the ledger: rule R reads the claim's own 0dp
    spelling, so a licence of +/-0.5 refuses a 4.0 gap that `_close` allowed."""
    _, ledger, claim = _derived_claim(value=1000, recent=1104.0, prior=100.0)

    verdict = DV.verify_number(None, claim, payload=ledger)

    assert verdict["actual"] == 1004.0
    assert verdict["ok"] is False
    assert verdict["reason"] == "claimed 1000.0, recomputed 1004.0"


def test_derived_claim_precision_comes_from_the_literal_not_the_float():
    """`3` and `3.0` are the same float and different claims. Rule R reads the
    spelling, so the integer licenses 3.18 at 0dp and the 1dp form does not.
    This is why the fix passes `num["value"]` and never `base["claimed"]`."""
    _, ledger_int, claim_int = _derived_claim(value=3, recent=4.0, prior=0.82)
    _, ledger_1dp, claim_1dp = _derived_claim(value=3.0, recent=4.0, prior=0.82)

    assert DV.verify_number(None, claim_int, payload=ledger_int)["ok"] is True
    assert DV.verify_number(None, claim_1dp, payload=ledger_1dp)["ok"] is False


def _synthetic_claim(*, value, metric="jog_minutes", field="mean"):
    ledger = [{
        "sequence": 1,
        "tool_name": "synthetic",
        "arguments": {},
        "result": {"metric": metric, "period": "30d", "field": field,
                    "value": value},
        "result_elided": False,
    }]
    claim = {"metric": metric, "period": "30d", "field": field,
             "value": value,
             "source": {"sequence": 1, "path": "$.result.value"}}
    return ledger, claim


def test_every_published_operation_is_one_the_verifier_understands():
    """#93: the vocabulary is closed, so the model has to be told what is in it.

    The prompt sentence and the verifier's branches read the same constant.
    This asserts the loop actually closes — every word published is accepted,
    and a word that is not published is refused.
    """
    from health_advisor import chat

    published = DV.DERIVATION_OPERATIONS
    assert published, "the vocabulary must not be empty"

    sentence = DV.operation_vocabulary_sentence()
    for operation in published:
        assert operation in sentence, f"{operation} is accepted but unpublished"
    assert sentence in chat.ASK_CLAIM_INSTRUCTIONS, \
        "the coach prompt no longer carries the vocabulary sentence"

    # Every published word computes something rather than falling through to
    # "unsupported derivation operation".
    for operation in published:
        _, ledger, claim = _derived_claim(value=2.0, recent=4.0, prior=2.0)
        claim["operation"] = operation
        verdict = DV.verify_number(None, claim, payload=ledger)
        assert "unsupported derivation operation" not in (verdict["reason"] or ""), \
            (operation, verdict["reason"])

    # And the shape the live model actually emitted stays refused.
    _, ledger, claim = _derived_claim(value=2.0, recent=4.0, prior=2.0)
    claim["operation"] = "(recent total - prior total)"
    verdict = DV.verify_number(None, claim, payload=ledger)
    assert verdict["ok"] is False
    assert "unsupported derivation operation" in verdict["reason"]
