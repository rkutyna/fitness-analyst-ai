"""Literature figures must remain machine-readable and cited."""
from __future__ import annotations

import ast
from pathlib import Path

from health_advisor import analysis as A
from health_advisor import mcp_server as M
from tests.conftest import seed_metric


ANALYSIS_PATH = Path(A.__file__)


def _literature_constant_names() -> list[str]:
    tree = ast.parse(ANALYSIS_PATH.read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        names.extend(
            target.id for target in targets
            if isinstance(target, ast.Name) and target.id.startswith("LITERATURE_")
        )
    return names


def test_literature_typed_constants_are_listed_and_cited():
    """The source scan catches a new literature container before publication."""
    names = _literature_constant_names()
    assert names, "literature-typed constants: none found"

    missing = []
    todo = []
    for constant_name in names:
        value = getattr(A, constant_name)
        if not isinstance(value, dict):
            missing.append(constant_name)
            continue
        for figure_name, figure in value.items():
            citation = figure.get("citation") if isinstance(figure, dict) else None
            if citation == "TODO(citation)":
                todo.append(f"{constant_name}.{figure_name}")
                continue
            if not isinstance(citation, dict):
                missing.append(f"{constant_name}.{figure_name}")
                continue
            required = {"author", "year", "venue"}
            if not all(citation.get(field) for field in required) or not (
                citation.get("doi") or citation.get("pmid")
            ):
                missing.append(f"{constant_name}.{figure_name}")

    if todo:
        print(f"TODO(citation) literature figures: {todo}")
    assert not missing, (
        f"literature-typed constants: {names}; missing citations: {missing}; "
        f"TODO(citation): {todo}"
    )


def test_affected_tool_payloads_publish_their_literature_citations(conn, tools):
    seed_metric(conn, "resting_heart_rate", "2026-08-31", [60.0])

    weekly = tools.get_weekly_series(
        "resting_heart_rate", "2026-08-31", "2026-08-31")
    benchmark = tools.get_benchmark_series()

    assert weekly["expected_training_effect_bpm"]["citation"] == (
        A.LITERATURE_FIGURES["expected_training_effect_bpm"]["citation"]
    )
    assert benchmark["heat_effect_bpm_per_c"]["citation"] == (
        A.LITERATURE_FIGURES["heat_effect_bpm_per_c"]["citation"]
    )


def test_prose_sites_name_the_same_sources_and_trends_explain_conventions():
    noise_doc = A.metric_noise_floor.__doc__ or ""
    weekly_doc = M.get_weekly_series.__doc__ or ""
    benchmark_doc = M.get_benchmark_series.__doc__ or ""

    assert "Reimers" in noise_doc and "10.3390/jcm7120503" in noise_doc
    assert "Reimers" in weekly_doc and "10.3390/jcm7120503" in weekly_doc
    assert "Pandolf" in benchmark_doc and "1200826" in benchmark_doc
    assert "operational convention" in (A.trends.__doc__ or "")
