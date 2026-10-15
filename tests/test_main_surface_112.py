"""Pinned behaviour of mcp_server.main with no new arguments (#112).

Recorded on the unchanged code before the `make_tools` keyword was added, and
asserted unchanged after. A downstream server that calls main() with its
existing arguments must see the same server name, tool set, usage text and exit
codes as before.
"""
from __future__ import annotations

import pytest

from health_advisor import mcp_server


def _captured_run(monkeypatch):
    seen = {}

    def fake_run(self, transport="stdio", **kwargs):
        seen["name"] = self.name
        seen["names"] = sorted(t.name for t in self._tool_manager.list_tools())
        seen["transport"] = transport

    monkeypatch.setattr(mcp_server.FastMCP, "run", fake_run)
    return seen


def test_main_default_serves_the_engine_tool_set(monkeypatch, tmp_path):
    seen = _captured_run(monkeypatch)
    vault_path = str(tmp_path / "health.db")
    assert mcp_server.main(["--vault", vault_path]) == 0
    assert seen["name"] == "health-advisor"
    assert seen["transport"] == "stdio"
    expected = sorted(mcp_server.build_tools(
        mcp_server.VaultContext.local(vault_path, user_id="local", writable=True)))
    assert seen["names"] == expected
    assert len(seen["names"]) > 0


def test_main_default_user_is_local(monkeypatch, tmp_path):
    seen = _captured_run(monkeypatch)
    assert mcp_server.main(["--vault", str(tmp_path / "h.db"), "--user", "u2"]) == 0
    assert seen["name"] == "health-advisor"


def test_main_default_usage_text_names_the_module(capsys):
    with pytest.raises(SystemExit) as exc:
        mcp_server.main(["--vault", "v.db", "--no-such-flag"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: python -m health_advisor.mcp_server")
    assert "error: unrecognized arguments: --no-such-flag" in err


def test_main_missing_vault_exits_two(capsys):
    with pytest.raises(SystemExit) as exc:
        mcp_server.main([])
    assert exc.value.code == 2
    assert "--vault" in capsys.readouterr().err


def test_main_help_lists_the_vault_and_user_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        mcp_server.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: python -m health_advisor.mcp_server")
    assert "--vault VAULT" in out
    assert "--user USER" in out


# --- make_tools (#112) -------------------------------------------------------

def get_latest(metric: str) -> dict:
    return {"metric": metric}


def summarize_metric(metric: str, period: str = "30d") -> dict:
    return {"metric": metric, "period": period}


def _stub_tools():
    return {"get_latest": get_latest, "summarize_metric": summarize_metric}


def test_make_tools_receives_the_context_built_from_argv(monkeypatch, tmp_path):
    seen = _captured_run(monkeypatch)
    received = []
    vault_path = str(tmp_path / "health.db")

    def factory(ctx):
        received.append(ctx)
        return _stub_tools()

    assert mcp_server.main(["--vault", vault_path, "--user", "u7"],
                           prog="x", make_tools=factory) == 0
    assert len(received) == 1
    assert str(received[0].db_path) == vault_path
    assert received[0].user_id == "u7"
    assert seen["names"] == ["get_latest", "summarize_metric"]
    assert seen["name"] == "health-advisor"


def test_make_tools_result_is_what_gets_served_with_prog_kept(monkeypatch, tmp_path, capsys):
    seen = _captured_run(monkeypatch)
    assert mcp_server.main(["--vault", str(tmp_path / "h.db")],
                           prog="coach-server",
                           make_tools=lambda ctx: {"get_latest": get_latest}) == 0
    assert seen["names"] == ["get_latest"]
    with pytest.raises(SystemExit):
        mcp_server.main(["--no-such-flag"], prog="coach-server",
                        make_tools=lambda ctx: {})
    assert "coach-server" in capsys.readouterr().err


def test_tools_and_make_tools_together_is_a_type_error(tmp_path):
    with pytest.raises(TypeError):
        mcp_server.main(["--vault", str(tmp_path / "h.db")],
                        tools=_stub_tools(), make_tools=lambda ctx: _stub_tools())


def test_make_tools_default_none_leaves_main_unchanged(monkeypatch, tmp_path):
    seen = _captured_run(monkeypatch)
    assert mcp_server.main(["--vault", str(tmp_path / "h.db")], make_tools=None) == 0
    expected = sorted(mcp_server.build_tools(
        mcp_server.VaultContext.local(str(tmp_path / "h.db"), user_id="local", writable=True)))
    assert seen["names"] == expected
