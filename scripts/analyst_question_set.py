"""Synthetic analyst cross-check harness; it never calls a live model.

Usage:
    python scripts/analyst_question_set.py --vault data/demo.db

The default completion function is intentionally scripted.  A caller may inject
another ``complete_fn`` for a local experiment, but the question set itself is
identity-free and every oracle is independent SQL over the supplied vault.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Make the repository package win when this file is invoked directly rather
# than as ``python -m scripts.analyst_question_set``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from health_advisor import analyst
from health_advisor import db
from health_advisor import normalize
from health_advisor.analyst_envelope import Envelope


@dataclass(frozen=True)
class Question:
    question: str
    metric: str
    declared_name: str
    mode: str = "correct"


QUESTIONS = (
    Question("What was my total jogging time in each of the last eight complete weeks?", "jog_minutes", "jog_minutes_per_week"),
    Question("What was my average daily resting heart rate in each of the last eight complete weeks?", "resting_heart_rate", "weekly_resting_heart_rate"),
    Question("What was my average daily heart-rate variability in each of the last eight complete weeks?", "heart_rate_variability", "weekly_heart_rate_variability"),
    Question("How many minutes did I sleep on average per night in each of the last eight complete weeks?", "sleep_asleep", "sleep_asleep_per_week"),
    Question("What was my average daily step count in each of the last eight complete weeks?", "step_count", "weekly_step_count"),
    Question("What was my average daily active energy in each of the last eight complete weeks?", "active_energy", "weekly_active_energy"),
    Question("What was my average body mass in each of the last eight complete weeks?", "body_mass", "weekly_body_mass"),
    Question("What was my average VO2 max in each of the last eight complete weeks?", "vo2_max", "weekly_vo2_max"),
    Question("What was my average respiratory rate in each of the last eight complete weeks?", "respiratory_rate", "weekly_respiratory_rate"),
    Question("What was my average exercise time in each of the last eight complete weeks?", "apple_exercise_time", "answer", "undeclared"),
    # Deliberately wrong, but still declared: it retains one daily row per
    # week. The parent must refuse it rather than trust its plausible shape.
    Question("What was my total jogging time in each of the last eight complete weeks (negative control)?", "jog_minutes", "jog_minutes_per_week", "one_day_per_week"),
)


def _value_column(metric: str) -> str:
    if metric == "jog_minutes":
        return "sum"
    aggregation = normalize.agg_for(metric)
    return "sum" if aggregation == "sum" else "last" if aggregation == "last" else "avg"


def _weekly_sql(metric: str, *, jog: bool = False, one_day: bool = False) -> str:
    value = _value_column(metric)
    if jog and one_day:
        expression = "MIN(v)"
    elif jog:
        expression = "SUM(v)"
    else:
        expression = f"AVG(v)"
    if jog:
        source = f"SELECT date, {value} AS v FROM daily_metrics WHERE metric = ?"
    else:
        source = f"SELECT date, {value} AS v FROM daily_metrics WHERE metric = ?"
    week_start = ("date(date, '-' || ((CAST(strftime('%w', date) AS INTEGER) "
                  "+ 6) % 7) || ' days')")
    return f"""
        WITH latest AS (SELECT MAX(date) AS d FROM daily_metrics),
        bounded AS (
            {source} AND date BETWEEN date((SELECT d FROM latest), '-63 days')
            AND date((SELECT d FROM latest), '-8 days')
        )
        SELECT {week_start} AS week_start, {expression} AS value
        FROM bounded
        WHERE v IS NOT NULL
        GROUP BY strftime('%Y-W%W', date)
        ORDER BY week_start
    """


def _code_for(question: Question) -> str:
    if question.metric == "jog_minutes":
        sql = _weekly_sql("jog_minutes", jog=True,
                          one_day=question.mode == "one_day_per_week")
        units = "min"
    else:
        sql = _weekly_sql(question.metric)
        units = normalize.canonical_unit(question.metric, "")
    return (
        f"rows = conn.execute({sql!r}, [{question.metric!r}]).fetchall()\n"
        f"emit({question.declared_name!r}, ['date_yyyymmdd', 'value'], "
        f"['count', {units!r}], [[int(row[0].replace('-', '')), row[1]] for row in rows])"
    )


def scripted_complete_fn() -> Callable[[str], str]:
    """Return a deterministic completion function for the fixed set."""
    by_question = {q.question: q for q in QUESTIONS}

    def complete(prompt: str, **kwargs) -> str:
        match = re.search(r"Question:\n(.*?)\n\nAvailable interface", prompt, re.DOTALL)
        question = match.group(1).strip() if match else ""
        return _code_for(by_question[question])

    return complete


def _oracle(conn: sqlite3.Connection, question: Question) -> list[dict]:
    rows = conn.execute(
        _weekly_sql(question.metric, jog=question.metric == "jog_minutes"),
        (question.metric,),
    ).fetchall()
    return [{"week": row[0], "value": row[1]} for row in rows]


def _scripted_run_code(code, vault_path, run_dir, executor, *, limits=None):
    """Execute only the harness's fixed child code against a read-only vault.

    This is the no-credentials default for the corpus.  A production caller
    can inject the real runner; the harness deliberately does not claim that
    these deterministic fixtures measured a live model or sandbox.
    """
    del run_dir, executor, limits
    tables = []

    def emit(name, columns, units, rows):
        tables.append({"name": name, "columns": tuple(columns),
                       "units": tuple(units), "rows": tuple(tuple(row) for row in rows),
                       "row_count": len(rows)})

    conn = db.connect(vault_path, read_only=True)
    try:
        exec(compile(code, "<scripted-analyst>", "exec"),
             {"__builtins__": __builtins__, "conn": conn, "emit": emit})
        rows_read = sum(1 for _ in conn.execute("SELECT 1 FROM daily_metrics"))
    finally:
        conn.close()
    return Envelope(
        run_id="scripted", question="", code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        vault_sha256="scripted", vault_version=0,
        ledger={"query_count": 1, "tables_read": ["daily_metrics"], "rows_read": rows_read},
        tables=tuple(tables),
        counts={"rows": sum(t["row_count"] for t in tables),
                "cells": sum(t["row_count"] * len(t["columns"]) for t in tables),
                "numeric_tokens": 0, "bytes": 0},
    )


def _envelope_values(payload: dict) -> list[dict]:
    # A refusal carries no tables; the harness must record that as an empty
    # envelope rather than crash the whole batch on the first refused answer.
    if not payload.get("tables"):
        return []
    if "tables" not in payload:
        return []
    def as_date(value: int) -> str:
        text = str(int(value))
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"

    return [{"week": as_date(row[0]), "value": row[1]}
            for row in payload["tables"][0]["rows"]]


def _correct_first_attempt(oracle: list[dict], payload: dict) -> bool:
    actual = _envelope_values(payload)
    if len(actual) != len(oracle):
        return False
    # The oracle's week label is `%Y-W%W`; compare by the Monday encoded in the
    # child table, avoiding any dependence on a model's display formatting.
    for expected, found in zip(oracle, actual):
        if expected["week"] != found["week"]:
            return False
        if abs(float(expected["value"]) - float(found["value"])) > 0.05:
            return False
    return True


def run_question_set(vault_path: str | Path, *, complete_fn=None,
                     run_code_fn=None, executor=None) -> list[dict]:
    """Run every fixed question through ``run_analyst`` and return records."""
    complete_fn = complete_fn or scripted_complete_fn()
    run_code_fn = run_code_fn or _scripted_run_code
    conn = db.connect(vault_path, read_only=True)
    results = []
    # The system temp root, never beside the vault: the sandbox profile denies
    # file-read-data under the user's home directory by design, so a run root
    # inside a checkout fails every child with "Operation not permitted".
    run_root = Path(tempfile.mkdtemp(prefix="analyst_question_set_"))
    try:
        for question in QUESTIONS:
            output = io.StringIO()
            kwargs = {
                "complete_fn": complete_fn,
                "json_output": True,
                "out": output,
            }
            if run_code_fn is not None:
                kwargs["run_code_fn"] = run_code_fn
            if executor is not None:
                kwargs["executor"] = executor
            run_dir = run_root / f"q{len(results) + 1:02d}"
            rc = analyst.run_analyst(question.question, str(vault_path),
                                     str(run_dir),
                                     **kwargs)
            payload = json.loads(output.getvalue())
            oracle = _oracle(conn, question)
            results.append({
                "question": question.question,
                "declared_quantity": payload.get("tables", [{}])[0].get("name")
                    if payload.get("tables") else question.declared_name,
                "cross_check_verdict": payload.get("verification"),
                "oracle_value": oracle,
                "envelope_value": _envelope_values(payload),
                "first_attempt_correct": _correct_first_attempt(oracle, payload),
                "return_code": rc,
            })
    finally:
        conn.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True)
    args = parser.parse_args(argv)
    results = run_question_set(args.vault)
    correct = sum(row["first_attempt_correct"] for row in results)
    for number, row in enumerate(results, 1):
        print(f"{number}. {row['question']}")
        print(f"   declared quantity: {row['declared_quantity']}")
        print(f"   cross-check verdict: {row['cross_check_verdict']}")
        print(f"   oracle value: {row['oracle_value']}")
        print(f"   envelope value: {row['envelope_value']}")
        print(f"   first-attempt correct: {'yes' if row['first_attempt_correct'] else 'no'}")
    print(f"First-attempt correct: {correct}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
