"""Tests for the model-free parts of scripts/benchmark_tool_turn.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import benchmark_tool_turn as benchmark


def test_nearest_rank_p95_and_median_are_hand_computable():
    summary = benchmark.aggregate_durations([1, 2, 3, 4, 5])

    assert summary["n"] == 5
    assert summary["median_seconds"] == 3
    # ceil(.95 * 5) = rank 5, so this deliberately small-sample p95 is max.
    assert summary["p95_rank"] == 5
    assert summary["p95_seconds"] == 5
    assert summary["min_seconds"] == 1
    assert summary["max_seconds"] == 5
    assert "n=5" in summary["p95_note"]


def test_empty_tool_loop_result_is_failure_and_not_a_timing_sample():
    calls = {question.name: 0 for question in benchmark.QUESTION_SHAPES}

    def invoke(question):
        calls[question.name] += 1
        if question.name == "period-comparison" and calls[question.name] == 1:
            return benchmark.Invocation("", {"outcome": "auth_failure"})
        return benchmark.Invocation("answer", {"outcome": "success"})

    now = [0.0]

    def clock():
        value = now[0]
        now[0] += 1.0
        return value

    report = benchmark.run_benchmark(
        repeat=2, backend="codex", invoke=invoke, clock=clock
    )
    comparison = report["shapes"]["period-comparison"]
    assert comparison["n"] == 1
    assert comparison["failures"] == 1
    assert comparison["failure_reasons"] == {"auth_failure": 1}
    assert report["overall"]["failures"] == 1
    assert report["overall"]["n"] == 7


def test_dry_run_produces_complete_human_and_json_reports(capsys):
    assert benchmark.main(["--dry-run", "--repeat", "2"]) == 0
    human = capsys.readouterr().out
    assert "single-metric-lookup" in human
    assert "period-comparison" in human
    assert "correlation" in human
    assert "plan-question" in human
    assert "overall" in human
    assert "n=8 successful timings" in human
    assert "p95" in human

    assert benchmark.main(["--dry-run", "--repeat", "2", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["attempts"] == 8
    assert report["overall"]["n"] == 8
    assert report["overall"]["failures"] == 0
    assert set(report["shapes"]) == {
        "single-metric-lookup",
        "period-comparison",
        "correlation",
        "plan-question",
    }
    assert all(report["shapes"][name]["n"] == 2 for name in report["shapes"])


def test_the_real_run_can_actually_import_the_package(tmp_path):
    """The expensive path's imports, exercised without spending a model call.

    --dry-run imports nothing from health_advisor, so the whole suite was green
    while `--backend codex` died on `from health_advisor import llm`: a script
    in scripts/ needs the repo root on sys.path and did not add it. A benchmark
    that only works from one working directory is not a reproducible command,
    which was the entire point of the issue that asked for it.

    Run from tmp_path so a bare `import health_advisor` cannot succeed by
    accident through the current working directory.
    """
    script = Path(__file__).parents[1] / "scripts" / "benchmark_tool_turn.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--self-check", "--backend", "codex"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "import ok" in proc.stdout
    assert "model=" in proc.stdout
