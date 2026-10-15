"""mcp_server helpers and the build_server/main parameters (#98).

The unknown-metric refusal and the unit-system lookup are factored into one
helper each; the refusal text and the lookup result are pinned here so the
factoring cannot change what a caller sees. build_server and main accept an
explicit tool mapping (and main a program name) so a downstream server can
reuse them without copying either function.
"""
from __future__ import annotations

import pytest

from health_advisor import mcp_server, vault
from tests.conftest import seed_metric

UNKNOWN = "no_such_metric"
REFUSAL = {"error": "unknown metric 'no_such_metric'. Call list_available_metrics."}


@pytest.fixture
def seeded(vault, conn):
    seed_metric(conn, "step_count", "2026-09-01", [1000.0])
    return vault


# One row per unknown-metric guard site (seven in mcp_server), each with the
# arguments that reach the guard before any other refusal could fire.
GUARDED = [
    ("get_daily_series", {"metric": UNKNOWN}),
    ("summarize_metric", {"metric": UNKNOWN, "period": "30d"}),
    ("compare_periods", {"metric": UNKNOWN, "period_a": "30d", "period_b": "30d"}),
    # get_intraday's D3 check refuses non-raw series before the guard, so the
    # unknown name must be one the vault ships raw samples for.
    ("get_intraday", {"metric": "heart_rate", "day": "2026-10-01", "bucket_hours": 1}),
    ("get_latest", {"metric": UNKNOWN}),
    ("correlate_metrics", {"metric_x": UNKNOWN, "metric_y": "step_count"}),
    ("correlate_metrics", {"metric_x": "step_count", "metric_y": UNKNOWN}),
    ("scan_correlations", {"target": UNKNOWN}),
]


def _refusal(metric):
    """The text every guard site returned before the helper (captured on the
    unchanged code)."""
    return {"error": f"unknown metric {metric!r}. Call list_available_metrics."}


@pytest.mark.parametrize("name,kwargs", GUARDED,
                         ids=[f"{n}-{'-'.join(k)}" for n, k in GUARDED])
def test_every_guard_site_refuses_with_the_same_dict(seeded, name, kwargs):
    tools = mcp_server.build_tools(seeded)
    unknown = [v for k, v in kwargs.items() if k.startswith(("metric", "target"))
               and v != "step_count"]
    assert tools[name](**kwargs) == _refusal(unknown[0])


def test_unknown_metric_helper_returns_the_same_dict():
    assert mcp_server._unknown_metric(UNKNOWN) == REFUSAL == _refusal(UNKNOWN)


@pytest.mark.parametrize("system", ["metric", "imperial"])
def test_unit_system_lookup_matches_inline_lookup(seeded, system):
    conn = seeded.connect()
    try:
        vault.set_unit_system(conn, system)
        conn.commit()
    finally:
        conn.close()
    settings = seeded.settings()
    assert settings["unit_system"] == system
    assert mcp_server._unit_system(settings) == settings["unit_system"]


def test_unit_system_lookup_on_a_vault_with_no_declared_system(seeded):
    settings = seeded.settings()
    assert mcp_server._unit_system(settings) == settings["unit_system"]


# --- build_server ------------------------------------------------------------

def _names(server):
    return sorted(t.name for t in server._tool_manager.list_tools())


def get_latest(metric: str) -> dict:
    return {"metric": metric}


def summarize_metric(metric: str, period: str = "30d") -> dict:
    return {"metric": metric, "period": period}


def _fake_tools():
    return {"get_latest": get_latest, "summarize_metric": summarize_metric}


def test_build_server_default_registers_the_engine_tools(seeded):
    server = mcp_server.build_server(seeded)
    assert _names(server) == sorted(mcp_server.build_tools(seeded))


def test_build_server_tools_mapping_replaces_the_default_set(seeded):
    server = mcp_server.build_server(seeded, tools=_fake_tools())
    assert _names(server) == ["get_latest", "summarize_metric"]


def test_build_server_include_still_filters_an_explicit_mapping(seeded):
    server = mcp_server.build_server(
        seeded, include=frozenset({"get_latest"}), tools=_fake_tools())
    assert _names(server) == ["get_latest"]


def test_build_server_include_still_filters_the_default_set(seeded):
    server = mcp_server.build_server(seeded, include=("get_latest",))
    assert _names(server) == ["get_latest"]


def test_build_server_keeps_its_name_default(seeded):
    server = mcp_server.build_server(seeded, tools=_fake_tools())
    assert server.name == "health-advisor"


# --- main --------------------------------------------------------------------

def test_main_passes_tools_and_prog_through(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run(self, transport="stdio", **kwargs):
        captured["names"] = _names(self)
        captured["transport"] = transport

    monkeypatch.setattr(mcp_server.FastMCP, "run", fake_run)
    vault_path = str(tmp_path / "health.db")
    assert mcp_server.main(["--vault", vault_path], tools=_fake_tools()) == 0
    assert captured == {"names": ["get_latest", "summarize_metric"], "transport": "stdio"}


def test_main_default_prog_is_unchanged(capsys):
    with pytest.raises(SystemExit):
        mcp_server.main(["--no-such-flag"])
    assert "python -m health_advisor.mcp_server" in capsys.readouterr().err


def test_main_prog_parameter_names_the_program(capsys):
    with pytest.raises(SystemExit):
        mcp_server.main(["--no-such-flag"], prog="coach-server")
    assert "coach-server" in capsys.readouterr().err
