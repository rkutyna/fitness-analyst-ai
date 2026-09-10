from __future__ import annotations

from datetime import date, timedelta

from health_advisor import chat
from health_advisor import deepdive_verify as DV


def _record(sequence: int, tool_name: str, result: dict) -> dict:
    return {
        "sequence": sequence,
        "tool_name": tool_name,
        "arguments": {},
        "result": result,
        "result_elided": False,
    }


def _empty_sleep_case() -> tuple[str, list[dict], list[dict]]:
    end = date.today()
    start = end - timedelta(days=6)
    prose = (
        f"I can't tell you how you slept last week ({start} through {end}) "
        "— there's no sleep data for that window at all. The sleep-timing "
        "record for those days shows 0 nights (plan compliance reports "
        "n_nights of 0), and the deep health briefing for "
        f"{end.strftime('%b')} {end.day} lists sleep_asleep as missing with "
        "no readings in the last 14 days. The metric catalog itself is "
        "empty, so there's no sleep duration or stage data to summarize "
        "either. In short: the vault has no sleep readings covering last "
        "week, so any number I gave you for hours slept or sleep quality "
        "would be invented."
    )
    ledger = [
        _record(1, "get_briefing", {"window_days": 14}),
        _record(2, "get_sleep_regularity", {"n_nights": 0}),
    ]
    claims = [{
        "field": "n_nights",
        "value": 0,
        "source": {"sequence": 2, "path": "$.result.n_nights"},
    }]
    return prose, ledger, claims


def test_recorded_empty_sleep_answer_binds_payload_window_literal():
    prose, ledger, claims = _empty_sleep_case()

    verdict = DV.verify_coach_claims(None, prose, claims, payload=ledger)

    assert verdict["unsupported"] == []
    assert verdict["grounded"] is True
    binding = next(item for item in verdict["payload_literals"]
                   if item["token"] == "14")
    assert binding["tool_name"] == "get_briefing"
    assert binding["key_path"] == "get_briefing.window_days"
    assert binding["path"] == "$.result.window_days"
    assert binding["source"] == "payload_literal"
    assert binding["tier"] == "payload_literal"


def test_absent_payload_literal_remains_unsupported():
    prose, ledger, claims = _empty_sleep_case()

    verdict = DV.verify_coach_claims(
        None, prose.replace("last 14 days", "last 15 days"), claims,
        payload=ledger)

    assert verdict["unsupported"] == ["15"]
    assert verdict["grounded"] is False


def test_nested_payload_literal_in_dict_inside_list_is_bound():
    ledger = [_record(1, "get_briefing", {
        "n_nights": 0,
        "sections": [{"sleep": {"window_days": 21}}],
    })]
    claims = [{
        "field": "n_nights",
        "value": 0,
        "source": {"sequence": 1, "path": "$.result.n_nights"},
    }]

    verdict = DV.verify_coach_claims(
        None, "The briefing covers 21 days and reports 0 nights.", claims,
        payload=ledger)

    assert verdict["unsupported"] == []
    assert verdict["grounded"] is True
    assert verdict["payload_literals"][0]["key_path"] == (
        "get_briefing.sections[0].sleep.window_days")


def test_payload_literal_binding_does_not_accept_non_numeric_strings():
    day = date.today()
    ledger = [_record(1, "get_briefing", {
        "window": "14 d",
        "day": day.isoformat(),
    })]

    verdict = DV.verify_coach_claims(
        None,
        f"The unsupported figures are 14 and {day.day} and {day.year} and "
        f"{day.month} in this draft.",
        [], payload=ledger)

    assert verdict["payload_literals"] == []
    assert verdict["unsupported"] == [
        "14", str(day.day), str(day.year), str(day.month)]


def test_exact_numeric_string_is_a_payload_literal():
    ledger = [_record(1, "get_briefing", {"window_days": "14"})]

    bindings = []
    grounded, unsupported = DV._coach_grounding(
        "The window is 14 days.", [], payload=ledger,
        literal_bindings_out=bindings)

    assert grounded is True
    assert unsupported == []
    assert bindings[0]["key_path"] == "get_briefing.window_days"


def test_fallback_counts_unsupported_figures_and_never_repeats_them():
    """The refusal names HOW MANY figures it dropped, never the figures.

    An unsupported token is a number the model derived and Python could not
    bind; repeating it inside the refusal would still put an unverified
    number in front of the user (the one rule).  This guard existed in
    test_chat.py before #370 and stays.
    """
    rendered = chat._fallback_answer({
        "unsupported": ["14"],
        "verdict": {"n_checkable": 1, "n_failed": 0},
    })

    assert "14" not in rendered
    assert "one figure" in rendered
    two = chat._fallback_answer({"unsupported": ["14", "97"],
                                 "verdict": {"n_checkable": 1, "n_failed": 0}})
    assert "97" not in two and "14" not in two and "2 figures" in two
    assert "couldn't verify a grounded answer" not in rendered
    assert "numeric verification verdict passed" in rendered
