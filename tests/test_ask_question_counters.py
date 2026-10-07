"""The ask question log counts what a turn was offered, cited and withheld.

consumer #579: ``figures_total`` is the number of figures the model STATED, so
the log could not say whether a narrated answer left out a figure the tools had
returned. The fact-template arm now puts Python-computed counters on its
``verification`` and ``_record_question`` copies them, with a closed ``cause``,
onto the row. Everything here is synthetic: invented values, a stubbed model.
"""
from __future__ import annotations

import ast
import inspect
import json
import textwrap

import pytest

from health_advisor import chat, fact_template, llm
from tests.conftest import seed_metric

AS_OF = "2026-08-21"
ASK_VO2 = "What is my VO2 max right now?"
ASK_NO_METRIC = "How am I doing overall?"

# Every key a row may carry. The first block is the pre-existing shape; the
# second is what this change adds. A key outside both is a leak until proven
# otherwise, which is the point of the privacy test.
OLD_ROW_KEYS = {
    "asked_at", "question", "as_of", "mode", "reason", "attempt_1_reason",
    "figures_verified", "figures_total", "elapsed_seconds", "python_seconds",
    "model_call_count", "model_calls", "accounting_anomaly",
}
NEW_ROW_KEYS = {
    "facts_offered", "facts_cited", "facts_withheld", "asked_metric_offered",
    "asked_metric_cited", "retried", "cause", "tool_calls",
    "template_compliant",
}
COUNTER_KEYS = {"facts_offered", "facts_cited", "facts_withheld",
                "asked_metric_offered", "asked_metric_cited", "retried"}


@pytest.fixture
def ledger(conn, tools):
    """A model that listed the metrics and read the latest of two of them."""
    seed_metric(conn, "vo2_max", "2026-08-10", [40.0, 41.0, 41.5, 42.0])
    seed_metric(conn, "step_count", "2026-08-10", [8000, 9000, 10234])
    return [
        {"sequence": 1, "tool_name": "list_available_metrics",
         "arguments": {}, "result": tools.list_available_metrics()},
        {"sequence": 2, "tool_name": "get_latest",
         "arguments": {"metric": "vo2_max"},
         "result": tools.get_latest("vo2_max")},
        {"sequence": 3, "tool_name": "get_latest",
         "arguments": {"metric": "step_count"},
         "result": tools.get_latest("step_count")},
    ]


def _key(ledger_, sequence: int, field: str = "value") -> str:
    day = ledger_[sequence - 1]["result"]["latest_day"]["date"]
    metric = ledger_[sequence - 1]["arguments"]["metric"]
    return fact_template.fact_key(metric, day, field)


def _ask(monkeypatch, tmp_path, vault, ledger_, responses, *,
         question=ASK_VO2, template="1"):
    """One ask through the real path with a stubbed model; return (result, row)."""
    if template is None:
        monkeypatch.delenv("HA_ASK_FACT_TEMPLATE", raising=False)
    else:
        monkeypatch.setenv("HA_ASK_FACT_TEMPLATE", template)
    log_path = tmp_path / "questions.jsonl"
    monkeypatch.setenv("HA_ASK_QUESTION_LOG", str(log_path))
    replies = iter(responses)
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger_)
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(llm, "tool_loop", lambda prompt, **k: next(replies))
    result = chat.answer_question(vault, question, as_of=AS_OF)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return result, json.loads(lines[0])


# --- the counters, one turn shape each --------------------------------------


def test_narration_citing_the_asked_metric_counts_offered_and_cited(
        monkeypatch, tmp_path, vault, ledger):
    template = "Your aerobic fitness reading was {%s} {%s}." % (
        _key(ledger, 2), _key(ledger, 2, "period_label"))

    out, row = _ask(monkeypatch, tmp_path, vault, ledger,
                    ["acknowledged", template])

    assert out["mode"] == "narration"
    assert row["mode"] == "narration"
    # Two metrics were published (a value each, plus a label that is not
    # counted); the prose cited one value, and the label is not a figure.
    assert row["facts_offered"] == 2
    assert row["facts_cited"] == 1
    assert row["facts_withheld"] == 0
    assert row["asked_metric_offered"] is True
    assert row["asked_metric_cited"] is True
    assert row["retried"] is False
    assert row["cause"] == "ok"
    assert row["tool_calls"] == 3
    assert row["template_compliant"] is True
    # The pre-existing meaning is unchanged: figures STATED.
    assert row["figures_total"] == 1
    # One sample row, for a consumer-side summary script.
    print("SAMPLE ROW", json.dumps(
        {k: v for k, v in row.items() if k not in ("model_calls",)}))


def test_narration_that_cites_only_the_asked_metrics_label_drops_the_figure(
        monkeypatch, tmp_path, vault, ledger):
    """The reachable 'narrated but did not cite the asked metric' shape.

    The denial gate counts any placeholder of the asked metric, and a period
    label is one. So prose that cites the asked metric's LABEL and a different
    metric's value passes every gate and is narrated, with the asked value
    never stated. Counting data facts only is what makes this visible.
    """
    template = "On {%s} your step count was {%s}." % (
        _key(ledger, 2, "period_label"), _key(ledger, 3))

    out, row = _ask(monkeypatch, tmp_path, vault, ledger,
                    ["acknowledged", template])

    assert out["mode"] == "narration"
    assert out["verification"]["cause"] == "ok"
    assert row["mode"] == "narration"
    assert row["asked_metric_offered"] is True
    assert row["asked_metric_cited"] is False
    assert row["facts_offered"] == 2
    assert row["facts_cited"] == 1


def test_denial_over_an_offered_figure_is_a_fallback_with_the_metric_offered(
        monkeypatch, tmp_path, vault, ledger):
    denial = ("That is not among the figures this lookup returned. "
              "{status:gathered_data_uncited}")

    out, row = _ask(monkeypatch, tmp_path, vault, ledger,
                    ["acknowledged", denial, denial])

    assert out["mode"] == "fallback"
    assert row["mode"] == "fallback"
    assert row["cause"] == "denied_available_figure"
    assert row["asked_metric_offered"] is True
    assert row["asked_metric_cited"] is False
    assert row["facts_cited"] == 0
    assert row["facts_offered"] == 2
    assert row["retried"] is True


def test_citing_a_different_metric_while_the_asked_one_is_offered_is_refused(
        monkeypatch, tmp_path, vault, ledger):
    """Cited-a-different-value is what the denial gate already refuses.

    Documented here so the tracked 'dropped figure' rate is read correctly: the
    gate turns this into a fallback after the retry, with the asked metric
    offered and not cited.
    """
    other = "You took {%s} steps." % _key(ledger, 3)

    out, row = _ask(monkeypatch, tmp_path, vault, ledger,
                    ["acknowledged", other, other])

    assert out["mode"] == "fallback"
    assert row["cause"] == "denied_available_figure"
    assert row["asked_metric_offered"] is True
    assert row["asked_metric_cited"] is False
    assert row["facts_cited"] == 1


def test_asked_metric_the_tool_returned_nothing_for_is_not_offered(
        monkeypatch, tmp_path, vault, conn, tools):
    seed_metric(conn, "step_count", "2026-08-10", [8000, 9000, 10234])
    short = [
        {"sequence": 1, "tool_name": "get_latest",
         "arguments": {"metric": "step_count"},
         "result": tools.get_latest("step_count")},
    ]
    template = "You took {%s} steps." % _key(short, 1)

    out, row = _ask(monkeypatch, tmp_path, vault, short,
                    ["acknowledged", template])

    assert row["asked_metric_offered"] is False
    assert row["asked_metric_cited"] is False
    assert row["facts_offered"] == 1
    assert row["facts_cited"] == 1


def test_a_question_naming_no_metric_leaves_both_asked_fields_null(
        monkeypatch, tmp_path, vault, ledger):
    template = "You took {%s} steps." % _key(ledger, 3)

    out, row = _ask(monkeypatch, tmp_path, vault, ledger,
                    ["acknowledged", template], question=ASK_NO_METRIC)

    assert row["asked_metric_offered"] is None
    assert row["asked_metric_cited"] is None
    assert row["facts_offered"] == 2
    assert row["facts_cited"] == 1


def test_withheld_count_is_a_length_and_the_turn_is_a_fallback(
        monkeypatch, tmp_path, vault, ledger):
    invented = {"invented_metric|2026-01-01|value", "another|2026-01-02|value"}
    monkeypatch.setattr(fact_template, "publish_completeness",
                        lambda *a, **k: set(invented))
    template = "Your aerobic fitness reading was {%s}." % _key(ledger, 2)

    out, row = _ask(monkeypatch, tmp_path, vault, ledger,
                    ["acknowledged", template, template])

    assert out["mode"] == "fallback"
    assert row["cause"] == "withheld_available_figure"
    assert row["facts_withheld"] == 2
    assert row["asked_metric_cited"] is True


def test_the_retry_describes_the_last_attempt(
        monkeypatch, tmp_path, vault, ledger):
    first = "Your reading was {status:gathered_data_uncited}."
    second = "Your aerobic fitness reading was {%s}." % _key(ledger, 2)

    out, row = _ask(monkeypatch, tmp_path, vault, ledger,
                    ["acknowledged", first, second])

    assert out["mode"] == "narration"
    assert row["retried"] is True
    assert row["facts_cited"] == 1
    assert row["asked_metric_cited"] is True


def test_the_conversational_branch_counts_nothing_offered(
        monkeypatch, tmp_path, vault, conn):
    seed_metric(conn, "vo2_max", "2026-08-10", [40.0])
    out, row = _ask(monkeypatch, tmp_path, vault, [], ["Hello there!"],
                    question="Thanks!")

    assert out["mode"] == "narration"
    assert row["cause"].startswith("conversational")
    assert row["facts_offered"] == 0
    assert row["facts_cited"] == 0
    assert row["facts_withheld"] == 0
    assert row["asked_metric_offered"] is None
    assert row["asked_metric_cited"] is None


def test_a_non_template_arm_still_writes_a_valid_row_without_the_counters(
        monkeypatch, tmp_path, vault, conn):
    seed_metric(conn, "vo2_max", "2026-08-10", [40.0, 41.0])
    verifications = iter([
        {"ok": False, "grounded": False, "unsupported": [], "reason": "no",
         "figures_verified": 0, "figures_total": 1, "tool_calls": 1},
        {"ok": False, "grounded": False, "unsupported": [], "reason": "no",
         "figures_verified": 0, "figures_total": 1, "tool_calls": 1},
    ])
    drafts = iter([llm.ResearchResponse("draft"), llm.ResearchResponse("retry")])
    monkeypatch.delenv("HA_ASK_FACT_TEMPLATE", raising=False)
    monkeypatch.setattr(chat, "_verify_ask_answer",
                        lambda *a, **k: next(verifications))
    monkeypatch.setattr(chat, "_read_ledger", lambda path: [{"sequence": 1}])
    monkeypatch.setattr(llm, "tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(llm, "tool_loop", lambda *a, **k: next(drafts))
    log_path = tmp_path / "questions.jsonl"
    monkeypatch.setenv("HA_ASK_QUESTION_LOG", str(log_path))

    chat.answer_question(vault, "How did I sleep?")

    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert OLD_ROW_KEYS <= set(row)
    assert not (COUNTER_KEYS & set(row))
    assert "template_compliant" not in row
    assert row["figures_total"] == 1


# --- the privacy invariant --------------------------------------------------


def _assert_row_is_closed(row: dict, published_keys: set[str]) -> None:
    """The invariant: nothing but ints, bools, null and a closed cause is new."""
    assert set(row) <= OLD_ROW_KEYS | NEW_ROW_KEYS, (
        f"unexpected row fields: {sorted(set(row) - OLD_ROW_KEYS - NEW_ROW_KEYS)}")
    for name in sorted(NEW_ROW_KEYS & set(row)):
        value = row[name]
        if name == "cause":
            assert value is None or value in chat._ASK_CAUSES
            # Bounded in length and charset even if the set were widened.
            assert value is None or (
                len(value) <= 40 and value.replace("_", "").isalpha()
                and value == value.lower())
        elif name in ("asked_metric_offered", "asked_metric_cited"):
            assert value is None or type(value) is bool
        elif name in ("retried", "template_compliant"):
            assert type(value) is bool
        else:
            assert type(value) is int, f"{name} is {type(value).__name__}"
    # No fact key, value or prose may appear anywhere outside the question,
    # which is the one field that legitimately holds the user's own words.
    rest = json.dumps({k: v for k, v in row.items() if k != "question"})
    for key in published_keys:
        assert key not in rest
        parsed = fact_template.parse_fact_key(key)
        if parsed is not None:
            assert parsed[0] not in rest  # the metric name
    assert "withheld_fact_keys" not in rest


def test_no_fact_key_metric_value_or_prose_reaches_the_row(
        monkeypatch, tmp_path, vault, ledger):
    invented = {"invented_metric|2026-01-01|value", "another|2026-01-02|value"}
    monkeypatch.setattr(fact_template, "publish_completeness",
                        lambda *a, **k: set(invented))
    template = "Your secret reading was {%s}." % _key(ledger, 2)

    out, row = _ask(monkeypatch, tmp_path, vault, ledger,
                    ["acknowledged", template, template])

    # The verification really does carry the list; the row must not.
    assert out["verification"]["withheld_fact_keys"] == sorted(invented)
    assert set(row) & NEW_ROW_KEYS == NEW_ROW_KEYS
    published = set(fact_template.build_fact_set(ledger)) | invented
    _assert_row_is_closed(row, published)
    assert "secret" not in json.dumps(row)


def test_a_hostile_verification_cannot_widen_the_row():
    """The copy is typed: a list, a string or a bool-as-int never passes."""
    fields = chat._question_log_counters({
        "facts_offered": ["key|a|b"], "facts_cited": True,
        "facts_withheld": "3", "asked_metric_offered": "yes",
        "asked_metric_cited": None, "retry": True,
        "cause": "a free-form sentence about the user's heart rate",
        "tool_calls": 1.5, "template_compliant": "yes",
        "withheld_fact_keys": ["some|key|value"],
    })
    assert fields == {"cause": None}


def test_cause_in_the_row_is_a_known_enum_value_for_every_cause_ask_cause_returns():
    source = textwrap.dedent(inspect.getsource(chat._ask_cause))
    returned = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Return):
            returned |= {sub.value for sub in ast.walk(node.value)
                         if isinstance(sub, ast.Constant)
                         and isinstance(sub.value, str)}
    assert returned
    # The one cause set directly by the scripted conversational reply.
    assert returned | {"conversational_scripted"} == set(chat._ASK_CAUSES)
