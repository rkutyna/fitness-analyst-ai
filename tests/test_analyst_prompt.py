"""Contract tests for analyst prompt and schema vocabulary construction."""
from __future__ import annotations

from health_advisor.analyst_prompt import (
    REFUSAL_CODES,
    REFUSAL_REMEDIATION,
    SQLITE_TEMP_STORE_NOTE,
    build_analyst_prompt,
    render_refusal,
    schema_summary,
)
from health_advisor import normalize
from tests.conftest import seed_metric, seed_workout


CAPS = {
    "rows_per_table": 200,
    "tables_per_run": 4,
    "cells_total": 2_000,
    "distinct_numeric_tokens": 200,
    "envelope_bytes": 65_536,
    "wall_clock_seconds": 60,
}


def test_schema_summary_is_live_compact_and_training_focused(conn):
    for metric in ("step_count", "heart_rate", "resting_heart_rate"):
        seed_metric(conn, metric, "2026-08-01", [1.0, 2.0])
    seed_workout(conn, "running", "2026-08-01", 30.0, 3.0)
    seed_workout(conn, "rowing", "2026-08-02", 45.0, 5.0)

    summary = schema_summary(conn)

    assert len(summary) < 4_000
    for table in (
        "records",
        "daily_metrics",
        "workouts",
        "subjective",
        "workout_weather",
    ):
        assert table in summary
    assert "rows=6" in summary
    assert "rows=2" in summary
    assert "(metric, local_date)" in summary
    assert "(metric, start_utc)" in summary
    assert "metrics" in summary
    assert all(metric in summary for metric in ("heart_rate", "resting_heart_rate", "step_count"))


def test_prompt_pins_contract_aggregation_and_all_performance_facts():
    prompt_text = build_analyst_prompt(
        "Compare outcomes after sessions.",
        "schema summary",
        caps=CAPS,
    )

    required = (
        "Query the vault only through the read-only SQLite connection conn.",
        "Aggregate in Python, then call `emit(table_name, columns, units, rows)` once per result table.",
        "Never emit raw records rows.",
        "one row per session for session analyses",
        "one row per week for trend questions",
        "one row per comparison group for outcome checks",
        "daily_metrics (65,677 rows) and workouts (805 rows): ~0.00 s to query.",
        "records filtered by metric and date: 0.18 s.",
        "A full scan with an unindexed predicate: 2.4 s.",
        "A window function scoped to ONE metric over all history: 0.7 s.",
        "A window function partitioned across ALL metrics: over 25 minutes, and will be killed by the run timeout.",
        "216,674 buckets in 58.7 seconds",
        "scoped to seven days took 0.7 s, and scoped to twelve weeks took 11.1 s",
        "Always constrain records by metric",
        "prefer daily_metrics for trends and history",
        "never write an unscoped `PARTITION BY metric`",
        "scope any impact-shaped or bucketed query to a bounded date range",
        "records is indexed on (metric, local_date) and (metric, start_utc) and has no date-only index",
        "Two mistakes that produce a confidently WRONG answer, both measured on this\n"
        "vault. Neither is caught by any check -- the vault is read, the provenance is\n"
        "perfect, and the number is simply wrong. Re-read your code against both before\n"
        "you emit:",
        "- `daily_metrics` has separate `count`, `sum`, `avg`, `min`, `max` and `last`\n"
        "  columns for each (metric, date). **`last` means \"the value at the latest\n"
        "  timestamp that day\", NOT the daily total.** For a cumulative metric such as\n"
        "  jog_minutes, step_count or a distance, the day's figure is `sum`. Selecting\n"
        "  `last` for one of those is wrong by a large factor and looks plausible.",
        "- When you group days into weeks or any other bucket, **aggregate every day in\n"
        "  the bucket.** A dictionary insert guarded by `if key not in seen` keeps ONE\n"
        "  day and silently discards the rest, reporting a single day as the bucket's\n"
        "  total. Sum (or average) the whole group explicitly.",
    )
    for phrase in required:
        assert phrase in prompt_text

    for key, value in CAPS.items():
        assert f"- {key}: {value}" in prompt_text
    for metric, definition in normalize.CATALOG.items():
        assert f"- {metric}: unit={definition['unit']}; aggregation={definition['agg']}" in prompt_text


def test_prompt_states_metric_aggregation_conventions_and_daily_rule():
    prompt_text = build_analyst_prompt(
        "How has my health changed over the last month?",
        "schema summary",
        caps=CAPS,
    )

    required = (
        "Aggregation conventions (minimum meaningful time window):",
        "`resting_heart_rate`: a weekly mean is the minimum; four-week blocks are",
        "`vo2_max`: aggregate monthly at minimum; use an annual cadence for trends.",
        "`body_mass`: use a weekly trend; never report day-over-day deltas.",
        "`step_count` and `apple_exercise_time`: daily is fine for totals; use weekly",
        "`jog_minutes` (impact volume): aggregate weekly — it is the volume dial, not",
        "`sleep_*` and `sleep_timing`: report by session, attributing each session to",
        "the date the session ends.",
        "Produce a daily series only when the question explicitly asks for daily",
        "values.",
    )
    for phrase in required:
        assert phrase in prompt_text


def test_refusal_codes_all_have_actionable_renderings():
    assert set(REFUSAL_REMEDIATION) == set(REFUSAL_CODES)
    for code in REFUSAL_CODES:
        remediation = REFUSAL_REMEDIATION[code]
        rendered = render_refusal(code)
        assert code in rendered
        assert remediation in rendered

    assert "query conn" in REFUSAL_REMEDIATION["ZERO_READ"].lower()
    assert "emit derived rows" in REFUSAL_REMEDIATION["ZERO_READ"].lower()
    assert "aggregate" in REFUSAL_REMEDIATION["CAP_ROWS"].lower()
    assert "split" in REFUSAL_REMEDIATION["CAP_ROWS"].lower()
    assert "fewer columns" in REFUSAL_REMEDIATION["CAP_CELLS"]
    assert "vocabulary" in REFUSAL_REMEDIATION["BAD_UNIT"]
    assert "vocabulary" in REFUSAL_REMEDIATION["BAD_COLUMN"]
    assert "daily_metrics" in REFUSAL_REMEDIATION["TIMEOUT"]
    assert "metric" in REFUSAL_REMEDIATION["TIMEOUT"]
    assert "date" in REFUSAL_REMEDIATION["TIMEOUT"]


def test_sqlite_temp_storage_hazard_is_exported():
    assert "disk I/O error" in SQLITE_TEMP_STORE_NOTE
    assert "PRAGMA temp_store=MEMORY" in SQLITE_TEMP_STORE_NOTE
    assert "TMPDIR" in SQLITE_TEMP_STORE_NOTE


def test_prompt_states_the_naming_grammar_and_cap():
    # #255: the model cannot honour a cap it is never told. The regex, the
    # 31-char limit, and the short-name example must all appear.
    prompt_text = build_analyst_prompt("q", "schema summary", caps=CAPS)
    for phrase in (
        "^[a-z][a-z0-9_]{0,30}$",
        "most 31 characters, starting with a lowercase letter",
        "(`rhr_weekly`, not `resting_heart_rate_weekly_summary`)",
    ):
        assert phrase in prompt_text


def test_prompt_states_the_stored_date_formats():
    # Measured 2026-08-30: the model filtered August with
    # substr(date, 5, 2) = '08' against dashed YYYY-MM-DD dates and got a
    # clean, parent-observed, zero-row answer. The stored formats must be
    # stated, with the failing guess named.
    prompt_text = build_analyst_prompt("q", "schema summary", caps=CAPS)
    for phrase in (
        "Date formats, exact",
        "`YYYY-MM-DD` WITH dashes",
        "substr(date, 6, 2)",
        "2019-07-02T21:19:08+00:00",
        "`yyyymmdd`",
    ):
        assert phrase in prompt_text
