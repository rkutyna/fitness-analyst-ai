#!/usr/bin/env python3
"""Measure grounded health-coach tool turns, rather than quoting one sample.

The harness asks four representative questions through ``llm.tool_loop`` and
records wall-clock seconds for each non-empty, successfully completed answer.
Python computes the median, nearest-rank p95, minimum, and maximum; the model
does not calculate any health figure. Blank answers and backend failures are
counted separately and never enter those timing statistics. A correlation
refusal is a valid timed answer when the model returns it as non-empty text.

``--dry-run`` exercises the same question selection, repetition, timing,
aggregation, and rendering path with a deterministic clock and stub answer;
it does not need a model or a database. Real runs need an explicit ``--db``
path and are read-only. The harness cannot distinguish cold from warm model
state because ``llm.tool_loop`` exposes neither model residency nor load
events: codex calls are ephemeral and Ollama only exposes a keep-alive hint.

The p95 is deliberately a nearest-rank order statistic: sort the *n*
successful durations and select rank ``ceil(0.95 * n)`` (one-based). Thus, for
the default n=5, p95 is simply the maximum observed run, not a precise estimate
of a population tail. Every report states the n used.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

# A script in scripts/ is not run from inside the package, so the repo root has
# to be on sys.path before `health_advisor` resolves — the same bootstrap
# scripts/test_mcp_tools.py does. Its absence was invisible to the whole test
# suite: --dry-run imports nothing from the package, so every test passed while
# the only path that measures anything died on `from health_advisor import llm`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class QuestionShape:
    """A named coach question included in every benchmark run."""

    name: str
    prompt: str


@dataclass(frozen=True)
class Invocation:
    """The model result plus the status side-channel captured immediately."""

    answer: Any
    status: Mapping[str, Any] | None = None


QUESTION_SHAPES: tuple[QuestionShape, ...] = (
    QuestionShape(
        "single-metric-lookup",
        "Use the available read-only health tools to answer this coach question: "
        "What was my average resting heart rate over the last week? Use one "
        "appropriate metric lookup, report only the value and period returned by "
        "the tool, and do not estimate or calculate from raw data.",
    ),
    QuestionShape(
        "period-comparison",
        "Use the available read-only health tools to answer this coach question: "
        "How did my average weekly jog minutes over the last four weeks compare "
        "with the four weeks before that? Use the period-comparison tool for the "
        "arithmetic and report only facts it returns; do not estimate or calculate "
        "a health figure yourself.",
    ),
    QuestionShape(
        "correlation",
        "Use the available read-only health tools to answer this coach question: "
        "Is there a meaningful relationship between my sleep duration and jog "
        "minutes over the available matching weeks? Use the correlation tool. If "
        "there are too few pairs and it refuses, report that refusal as the answer; "
        "do not loosen a threshold or calculate a correlation yourself.",
    ),
    QuestionShape(
        "plan-question",
        "Read the applicable human-authored guidance in docs/fitness/ and answer "
        "this coach question: What should I focus on in my next training week "
        "according to the plan? Quote or paraphrase the plan's guidance, use no "
        "health arithmetic, and do not invent a recommendation when the plan is "
        "silent. A plan refusal or missing-plan answer is a valid answer if the "
        "backend returns it as non-empty text.",
    ),
)

COLD_WARM_NOTE = (
    "Cold and warm runs are not separated: llm.tool_loop exposes no model-load "
    "or residency event; codex calls are ephemeral and Ollama exposes only a "
    "keep-alive hint, not whether a model was resident."
)


def p95_nearest_rank(values: list[float]) -> tuple[float | None, int | None]:
    """Return nearest-rank p95 and its one-based rank, without rounding."""
    if not values:
        return None, None
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1], rank


def _p95_note(rank: int | None, n: int) -> str:
    """Say what the reported p95 actually is for THIS n.

    A templated sentence that names a sample size it did not use is a wrong
    number in the output, which is the one thing this benchmark exists to stop.
    At small n the nearest-rank p95 IS the maximum and must say so; once rank
    falls below n it is an order statistic and still not a tail estimate.
    """
    if not n or rank is None:
        return "p95 unavailable because n=0 successful timings were recorded."
    if rank >= n:
        return (f"nearest-rank p95 used rank {rank} of n={n} successful "
                "timings, which is the maximum observed run — not an estimate "
                "of a population tail.")
    return (f"nearest-rank p95 used rank {rank} of n={n} successful timings. "
            f"It is the {rank}th slowest run, an order statistic, not a fitted "
            "tail estimate.")


def aggregate_durations(
    durations: list[float],
    *,
    failures: int = 0,
    failure_reasons: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Summarize successful durations; failed calls are never included."""
    values = [float(value) for value in durations]
    p95, rank = p95_nearest_rank(values)
    return {
        "n": len(values),
        "failures": int(failures),
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": p95,
        "min_seconds": min(values) if values else None,
        "max_seconds": max(values) if values else None,
        "p95_rank": rank,
        "p95_note": _p95_note(rank, len(values)),
        "failure_reasons": dict(failure_reasons or {}),
    }


def _normalise_invocation(value: Invocation | str) -> Invocation:
    if isinstance(value, Invocation):
        return value
    return Invocation(value, {})


def _failure_reason(
    invocation: Invocation,
    *,
    backend: str,
    raised: BaseException | None = None,
) -> str | None:
    """Classify a result without turning an empty answer into a fast success."""
    if raised is not None:
        return f"runner_exception:{type(raised).__name__}"
    answer = invocation.answer
    if not isinstance(answer, str):
        return "invalid_response"
    if answer.strip():
        if backend == "codex":
            outcome = (invocation.status or {}).get("outcome")
            if outcome not in {None, "success"}:
                return f"status_{outcome}"
        return None

    status = invocation.status or {}
    outcome = status.get("outcome")
    if backend == "codex" and outcome and outcome != "not_called":
        # Includes timeout, auth_failure, rate_limited, binary_missing,
        # nonzero_exit, process_error, and empty_success from llm.py.
        return "empty_answer" if outcome == "success" else str(outcome)
    # llm.py does not update last_codex_status for Ollama, so a blank Ollama
    # result cannot safely be called a fast empty answer or a specific failure.
    return "empty_response_or_backend_failure"


def run_benchmark(
    *,
    repeat: int,
    backend: str,
    invoke: Callable[[QuestionShape], Invocation | str],
    clock: Callable[[], float] = time.perf_counter,
    db_path: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run all question shapes sequentially and return a JSON-safe report."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")

    shape_durations: dict[str, list[float]] = {
        question.name: [] for question in QUESTION_SHAPES
    }
    shape_failures: dict[str, Counter[str]] = {
        question.name: Counter() for question in QUESTION_SHAPES
    }
    failure_details: list[dict[str, Any]] = []

    for question in QUESTION_SHAPES:
        for iteration in range(1, repeat + 1):
            started = clock()
            raised: BaseException | None = None
            try:
                invocation = _normalise_invocation(invoke(question))
            except Exception as exc:  # a harness failure must be visible
                invocation = Invocation("", {"outcome": "runner_exception"})
                raised = exc
            elapsed = clock() - started
            reason = _failure_reason(invocation, backend=backend, raised=raised)
            if reason is None:
                shape_durations[question.name].append(elapsed)
            else:
                shape_failures[question.name][reason] += 1
                failure_details.append({
                    "shape": question.name,
                    "iteration": iteration,
                    "reason": reason,
                    "elapsed_seconds": elapsed,
                })

    shapes: dict[str, dict[str, Any]] = {}
    for question in QUESTION_SHAPES:
        shapes[question.name] = aggregate_durations(
            shape_durations[question.name],
            failures=sum(shape_failures[question.name].values()),
            failure_reasons=shape_failures[question.name],
        )

    all_durations = [
        duration
        for question in QUESTION_SHAPES
        for duration in shape_durations[question.name]
    ]
    all_failures = Counter()
    for reasons in shape_failures.values():
        all_failures.update(reasons)

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "model_id": None,
        "db_path": db_path,
        "dry_run": dry_run,
        "repeat_requested": repeat,
        "attempts": len(QUESTION_SHAPES) * repeat,
        "cold_warm": {"separated": False, "note": COLD_WARM_NOTE},
        "questions": [
            {"name": question.name, "prompt": question.prompt}
            for question in QUESTION_SHAPES
        ],
        "shapes": shapes,
        "overall": aggregate_durations(
            all_durations,
            failures=sum(all_failures.values()),
            failure_reasons=all_failures,
        ),
        "failure_details": failure_details,
        "failure_reporting_note": (
            "A returned refusal is a successful timed answer when non-empty. "
            "Blank/failed returns are excluded from n. Codex statuses come from "
            "llm.last_codex_status(); Ollama blank returns cannot be distinguished "
            "further because llm.py does not expose an Ollama status side-channel."
        ),
    }


def _dry_run_invoker(clock: "_DryClock") -> Callable[[QuestionShape], Invocation]:
    calls: Counter[str] = Counter()
    base_seconds = {
        "single-metric-lookup": 0.20,
        "period-comparison": 0.40,
        "correlation": 0.30,
        "plan-question": 0.25,
    }

    def invoke(question: QuestionShape) -> Invocation:
        iteration = calls[question.name]
        calls[question.name] += 1
        clock.advance(base_seconds[question.name] + iteration * 0.01)
        return Invocation("dry-run stub answer", {"outcome": "success"})

    return invoke


class _DryClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def render_human(report: Mapping[str, Any]) -> str:
    """Render a copy/pasteable report for a product document."""
    def fmt(value: Any) -> str:
        return "-" if value is None else f"{value:.3f}"

    lines = [
        "Grounded tool-turn benchmark",
        f"backend={report['backend']}  model={report['model_id'] or 'unknown'}  "
        f"repeat_requested={report['repeat_requested']}  attempts={report['attempts']}",
        f"database={report['db_path'] or '(dry-run stub; no database opened)'}",
        f"cold/warm: {report['cold_warm']['note']}",
        "",
        "shape                    n  failures  median(s)  p95(s)  min(s)  max(s)",
        "-----------------------  -  --------  ---------  ------  ------  ------",
    ]
    for name, summary in report["shapes"].items():
        lines.append(
            f"{name:23}  {summary['n']:1d}  {summary['failures']:8d}  "
            f"{fmt(summary['median_seconds']):>9}  {fmt(summary['p95_seconds']):>6}  "
            f"{fmt(summary['min_seconds']):>6}  {fmt(summary['max_seconds']):>6}"
        )
    overall = report["overall"]
    lines.extend([
        "",
        "overall",
        f"n={overall['n']} successful timings; failures={overall['failures']}; "
        f"median={fmt(overall['median_seconds'])} s; "
        f"p95={fmt(overall['p95_seconds'])} s; "
        f"min={fmt(overall['min_seconds'])} s; "
        f"max={fmt(overall['max_seconds'])} s",
        overall["p95_note"],
    ])
    reasons = report["overall"]["failure_reasons"]
    if reasons:
        lines.append("failure reasons: " + ", ".join(
            f"{reason}={count}" for reason, count in sorted(reasons.items())
        ))
    lines.extend(["", report["failure_reporting_note"]])
    return "\n".join(lines)


def self_check(backend: str) -> str:
    """Resolve everything a real run needs, without calling a model.

    Exists because the expensive path and the tested path were disjoint: the
    real run's imports had never executed under test. This is the cheap half of
    a real run — imports, the module-global backend switch, the model id — and
    it is what the subprocess test exercises. It cannot tell you the token is
    alive; only a real run does that.
    """
    from health_advisor import llm
    from health_advisor.context import VaultContext  # noqa: F401
    return _model_id(llm, backend)


def _model_id(llm_module: Any, backend: str) -> str:
    return str(llm_module.CODEX_MODEL if backend == "codex" else llm_module.MODEL)


def _real_invoker(
    *,
    backend: str,
    db_path: Path,
) -> tuple[Callable[[QuestionShape], Invocation], str]:
    from health_advisor import llm
    from health_advisor.context import VaultContext

    llm.BACKEND = backend
    context = VaultContext.local(db_path, user_id="benchmark", writable=False)
    # Schema construction is setup, not a model turn. The bound tools still
    # carry this read-only context and llm.tool_loop uses it for every call.
    tools = llm.tool_schemas(context)

    def invoke(question: QuestionShape) -> Invocation:
        answer = llm.tool_loop(
            question.prompt,
            ctx=context,
            tools=tools,
            think=True,
        )
        # Capture immediately: any subsequent model call could overwrite it.
        status = llm.last_codex_status() if backend == "codex" else {}
        return Invocation(answer, status)

    return invoke, _model_id(llm, backend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure grounded llm.tool_loop wall time for four real coach "
            "question shapes. Each shape runs --repeat times. n is the count "
            "of non-empty successful answers; failures and blank answers are "
            "reported separately and excluded from median/p95/min/max. p95 is "
            "nearest-rank, so with n=5 it is the maximum observed run, not a "
            "precise population-tail estimate. Python times and aggregates; "
            "the model only narrates tool-produced facts."
        )
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        help="read-only SQLite vault path (required unless --dry-run is used)",
    )
    parser.add_argument(
        "--backend",
        choices=("codex", "ollama"),
        default="codex",
        help="model backend to benchmark (default: codex)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        metavar="N",
        help="number of attempts per question shape; default: 5; failures are "
        "counted and do not become timing samples",
    )
    parser.add_argument(
        "--model-id",
        help="override the backend model id for this process and record it",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="resolve the imports and model id a real run needs, print them and "
        "exit; no model call, no database. Answers 'can this machine run the "
        "benchmark at all' without spending a turn finding out",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the complete harness against a deterministic timing stub; no "
        "model or database is opened",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="render the complete report as JSON instead of the human table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="also write the selected human/JSON report to PATH (never data/ by default)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        print(f"import ok; backend={args.backend} "
              f"model={args.model_id or self_check(args.backend)}")
        return 0
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if not args.dry_run and not args.db:
        raise SystemExit("--db PATH is required unless --dry-run is used")
    if not args.dry_run and not Path(args.db).is_file():
        raise SystemExit(f"database does not exist: {args.db}")
    if args.output:
        repo_data = (Path(__file__).resolve().parent.parent / "data").resolve()
        output_path = args.output.resolve()
        if output_path == repo_data or repo_data in output_path.parents:
            raise SystemExit("--output may not write under the repository data/ directory")

    if args.dry_run:
        clock = _DryClock()
        report = run_benchmark(
            repeat=args.repeat,
            backend=args.backend,
            invoke=_dry_run_invoker(clock),
            clock=clock,
            dry_run=True,
        )
        report["model_id"] = args.model_id or "dry-run-stub"
    else:
        from health_advisor import llm

        invoker, model_id = _real_invoker(
            backend=args.backend,
            db_path=Path(args.db).resolve(),
        )
        if args.model_id:
            if args.backend == "codex":
                llm.CODEX_MODEL = args.model_id
            else:
                llm.MODEL = args.model_id
            model_id = args.model_id
        report = run_benchmark(
            repeat=args.repeat,
            backend=args.backend,
            invoke=invoker,
            db_path=str(Path(args.db).resolve()),
        )
        report["model_id"] = model_id

    rendered = json.dumps(report, indent=2, sort_keys=False) if args.json else render_human(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
