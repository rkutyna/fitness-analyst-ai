"""Correctness gates for declared analyst quantities."""
from __future__ import annotations

import io
import json

from health_advisor import analyst
from health_advisor import analyst_crosscheck as check
from health_advisor.analyst_envelope import Envelope
from scripts.analyst_question_set import QUESTIONS, scripted_complete_fn
from tests.conftest import seed_metric


def _envelope(name, columns, units, rows):
    table = {"name": name, "columns": tuple(columns), "units": tuple(units),
             "rows": tuple(tuple(row) for row in rows), "row_count": len(rows)}
    return Envelope(
        run_id="synthetic", question="synthetic", code_sha256="code",
        vault_sha256="vault", vault_version=0,
        ledger={"query_count": 1, "tables_read": ["daily_metrics"], "rows_read": 1},
        tables=(table,), counts={"rows": len(rows), "cells": len(rows) * len(columns),
                                 "numeric_tokens": len(rows) * len(columns), "bytes": 1},
    )


def test_jog_declaration_recomputes_all_eight_week_values(vault_path):
    """The issue's oracle is a weekly total, not one selected day."""
    from health_advisor import db

    conn = db.connect(vault_path)
    db.init_db(conn)
    expected = [
        {"period_start": "2026-06-29", "jog_minutes": 127.4},
        {"period_start": "2026-07-06", "jog_minutes": 120.3},
        {"period_start": "2026-07-13", "jog_minutes": 137.0},
        {"period_start": "2026-07-20", "jog_minutes": 95.4},
        {"period_start": "2026-07-27", "jog_minutes": 127.0},
        {"period_start": "2026-08-03", "jog_minutes": 146.3},
        {"period_start": "2026-08-10", "jog_minutes": 78.6},
        {"period_start": "2026-08-17", "jog_minutes": 145.3},
    ]
    # Keep this fixture entirely synthetic but exercise the real deterministic
    # path: daily_metrics is the independent SQL source, while impact_volume
    # obtains the same periods through its documented manual fallback.
    for row in expected:
        conn.execute(
            "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, last, unit) "
            "VALUES ('jog_minutes', ?, 1, ?, ?, ?, ?, ?, 'min')",
            (row["period_start"], row["jog_minutes"], row["jog_minutes"],
             row["jog_minutes"], row["jog_minutes"], row["jog_minutes"]),
        )
        db.log_manual_jog(conn, row["period_start"], jog_minutes=row["jog_minutes"],
                          source_note="synthetic fixture", why="cross-check test")
    conn.close()
    envelope = _envelope(
        "jog_minutes_per_week", ["date_yyyymmdd", "jog_minutes"],
        ["count", "min"],
        [[int(row["period_start"].replace("-", "")), row["jog_minutes"]]
         for row in expected],
    )
    result = check.cross_check(vault_path, envelope)
    assert result.verification == "cross_checked"
    assert [row["jog_minutes"] for row in result.oracle] == [
        127.4, 120.3, 137.0, 95.4, 127.0, 146.3, 78.6, 145.3]


def test_declared_one_day_per_week_answer_is_refused(monkeypatch, vault_path):
    from health_advisor import db

    conn = db.connect(vault_path)
    db.init_db(conn)
    conn.close()
    monkeypatch.setattr(check.analysis, "impact_volume", lambda *args, **kwargs: [
        {"period_start": "2026-06-29", "jog_minutes": 127.4},
        {"period_start": "2026-07-06", "jog_minutes": 120.3},
    ])
    envelope = _envelope(
        "jog_minutes_per_week", ["date_yyyymmdd", "jog_minutes"], ["count", "min"],
        [[20260629, 17.0], [20260706, 20.0]],
    )
    result = check.cross_check(vault_path, envelope)
    assert result.verification == "refused_disagreement"
    assert "disagreement" in (result.reason or "")


def test_weekly_series_uses_analysis_and_undeclared_is_unverified(conn, vault_path):
    seed_metric(conn, "resting_heart_rate", "2026-08-01", [60, 61, 62, 63, 64, 65, 66, 67])
    expected = check.analysis.weekly_series(
        conn, "resting_heart_rate", "2026-08-01", "2026-08-08"
    )
    envelope = _envelope(
        "weekly_resting_heart_rate", ["date_yyyymmdd", "mean"],
        ["count", "count/min"],
        [[int(row["week_start"].replace("-", "")), row["mean"]]
         for row in expected],
    )
    result = check.cross_check(vault_path, envelope)
    assert result.verification == "cross_checked"

    undeclared = _envelope("answer", ["date_yyyymmdd", "value"], ["count", "count/min"],
                           [[20260801, 1.0]])
    assert check.cross_check(vault_path, undeclared).verification == "unverified"


def test_run_analyst_labels_json_and_refuses_before_render(tmp_path, vault_path, monkeypatch):
    from health_advisor import db

    conn = db.connect(vault_path)
    db.init_db(conn)
    conn.close()
    envelope = _envelope("jog_minutes_per_week", ["date_yyyymmdd", "jog_minutes"],
                          ["count", "min"], [[20260629, 1.0]])
    monkeypatch.setattr(check, "cross_check", lambda *args: check.CrossCheck(
        "refused_disagreement", "jog_minutes_per_week", "value disagreement"))

    out = io.StringIO()
    rc = analyst.run_analyst(
        "q", vault_path, str(tmp_path / "run"),
        complete_fn=lambda prompt: "emit('x', [], [], [])",
        run_code_fn=lambda *args, **kwargs: envelope,
        executor=object(), json_output=True, out=out,
    )
    payload = json.loads(out.getvalue())
    assert rc == 1
    assert payload["verification"] == "refused_disagreement"
    assert payload["refused"] is True


def test_question_set_has_scripted_correct_wrong_and_undeclared_attempts():
    assert len(QUESTIONS) >= 10
    complete = scripted_complete_fn()
    prompts = [f"Question:\n{q.question}\n\nAvailable interface" for q in QUESTIONS]
    codes = [complete(prompt) for prompt in prompts]
    assert "jog_minutes_per_week" in codes[0]
    assert "SUM(v)" in codes[0]                 # correct weekly total
    assert "MIN(v)" in codes[-1]                # one day per week negative control
    assert "emit('answer'" in codes[-2]         # undeclared result


def _init_vault(vault_path):
    from health_advisor import db

    conn = db.connect(vault_path)
    db.init_db(conn)
    conn.close()


def test_empty_declared_table_is_unverified_not_cross_checked(vault_path):
    """Zero comparisons must not read as a clean cross-check (#23 gate)."""
    _init_vault(vault_path)
    result = check.cross_check(
        vault_path, _envelope("jog_minutes_per_week",
                              ["date_yyyymmdd", "jog_minutes"], ["count", "min"], []))
    assert result.verification == "unverified"
    assert "nothing was recomputed" in (result.reason or "")
    empty_blocks = check.cross_check(
        vault_path, _envelope("block_structure",
                              ["date_yyyymmdd", "longest_block_min", "qualified_block_min"],
                              ["count", "min", "min"], []))
    assert empty_blocks.verification == "unverified"


def test_declared_table_with_unparseable_periods_is_unverified(vault_path):
    """A declaration whose period column cannot be read compares nothing."""
    _init_vault(vault_path)
    result = check.cross_check(
        vault_path, _envelope("jog_minutes_per_week",
                              ["date_yyyymmdd", "jog_minutes"], ["count", "min"],
                              [[1, 12.0], [2, 13.0]]))
    assert result.verification == "unverified"
    assert result.reason and "parseable period" in result.reason
