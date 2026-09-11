from __future__ import annotations

import json
import shutil
import socket
import sqlite3
import sys

import pytest

from health_advisor import analyst_corpus as ac
from health_advisor import analyst_prompt
from health_advisor import analyst_runner as runner
from health_advisor import analyst_sandbox as sandbox
from health_advisor import receiver
from tests.test_analyst_corpus import _build_test_corpus
from tests.test_analyst_runner import LocalChannelExecutor, _build_vault


def _frame(payload: dict) -> bytes:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return len(body).to_bytes(4, "big") + body


def _read_frame(sock: socket.socket) -> dict:
    header = sock.recv(4)
    size = int.from_bytes(header, "big")
    body = sock.recv(size)
    while len(body) < size:
        body += sock.recv(size - len(body))
    return json.loads(body.decode("utf-8"))


def test_cite_frame_is_served_before_sql_and_refusals_continue(tmp_path):
    corpus = ac.open_corpus(_build_test_corpus(tmp_path / "corpus.db"))
    state = ac.CiteState()
    conn = sqlite3.connect(":memory:")
    parent, child = socket.socketpair()
    try:
        child.sendall(_frame({"op": "cite", "query": "built corpus"}))
        pending = bytearray()
        more, reason = runner._service_query(
            parent, conn, 10, pending, corpus, state)
        assert (more, reason) == (True, None)
        cite_response = _read_frame(child)
        assert cite_response["ok"] is True
        assert cite_response["passages"][0]["span"]

        child.sendall(_frame({"sql": "select 1"}))
        more, reason = runner._service_query(
            parent, conn, 10, pending, corpus, state)
        assert (more, reason) == (True, None)
        sql_response = _read_frame(child)
        assert sql_response["ok"] is True
        assert sql_response["rows"] == [[1]]
        assert sql_response["description"][0][0] == "1"

        child.sendall(_frame({"op": "cite", "query": 3}))
        more, reason = runner._service_query(
            parent, conn, 10, pending, corpus, state)
        assert (more, reason) == (True, None)
        refusal = _read_frame(child)
        assert refusal["ok"] is False
        assert refusal["error_type"] == "CiteRefusal"

        child.sendall(_frame({"sql": "select 1"}))
        more, reason = runner._service_query(
            parent, conn, 10, pending, corpus, state)
        assert (more, reason) == (True, None)
        assert _read_frame(child)["ok"] is True
    finally:
        parent.close()
        child.close()
        conn.close()
        corpus.close()


def test_no_corpus_keeps_source_closed_and_result_explicit(tmp_path):
    source = runner._runner_source("code.py")
    assert runner._runner_source("code.py") == (
        runner.RUNNER_TEMPLATE.replace("__code_path__", repr("code.py"))
        .replace("__query_fd__", "4").replace("__out_fd__", "3")
    )
    assert "def cite(" not in source
    assert '"cite": cite' not in source

    vault = tmp_path / "vault.db"
    _build_vault(vault)
    result = runner.run_analyst_code(
        "assert 'cite' not in globals()\nrows = conn.execute('select count(*) from t').fetchall()\n"
        "emit('probe', ['n'], ['count'], [[rows[0][0]]])",
        str(vault), str(tmp_path / "run"), LocalChannelExecutor())
    assert isinstance(result, runner.AnalystEnvelope)
    assert result.citation_ledger == {
        "corpus_configured": False,
        "corpus_version": None,
        "retrieval_channel_closed": True,
    }


def test_corpus_child_has_two_cites_one_vault_query_and_separate_ledgers(tmp_path):
    vault = tmp_path / "vault.db"
    corpus = _build_test_corpus(tmp_path / "corpus.db")
    _build_vault(vault)
    result = runner.run_analyst_code(
        "first = cite('integrity check', 1)\n"
        "second = cite('built corpus', 1)\n"
        "assert first and second\n"
        "rows = conn.execute('select count(*) from t').fetchall()\n"
        "emit('probe', ['n'], ['count'], [[rows[0][0]]])\n",
        str(vault), str(tmp_path / "run"), LocalChannelExecutor(),
        corpus_path=str(corpus),
    )
    assert isinstance(result, runner.AnalystEnvelope)
    assert result.ledger["query_count"] == 1
    citations = result.citation_ledger
    assert citations["corpus_configured"] is True
    assert citations["corpus_version"] == 1
    assert citations["cite_calls"] == 2
    assert citations["passages"]
    passage = citations["passages"][0]
    assert passage["span"]
    assert passage["title"] == "Built document"
    assert passage["license"] == "CC-BY-4.0"
    assert result.ledger is not citations


def _real_executor_or_skip():
    if sys.platform == "darwin":
        try:
            return sandbox.SeatbeltExecutor()
        except RuntimeError as exc:
            pytest.skip(str(exc))
    if sys.platform.startswith("linux"):
        if not shutil.which("bwrap"):
            pytest.skip("requires the platform sandbox executor")
        return sandbox.BwrapExecutor()
    pytest.skip("requires the platform sandbox executor")


def _namespace_unavailable(result) -> bool:
    diagnostic = (result.diagnostic or "").lower() if hasattr(result, "diagnostic") else ""
    return ("namespace" in diagnostic or "sandbox_apply" in diagnostic) and any(
        word in diagnostic for word in ("permission", "not allowed", "operation")
    )


def test_real_platform_sandbox_child_can_cite_and_query(tmp_path):
    executor = _real_executor_or_skip()
    vault = tmp_path / "vault.db"
    corpus = _build_test_corpus(tmp_path / "corpus.db")
    _build_vault(vault)
    result = runner.run_analyst_code(
        "first = cite('integrity check', 1)\n"
        "second = cite('built corpus', 1)\n"
        "assert first and second\n"
        "rows = conn.execute('select count(*) from t').fetchall()\n"
        "emit('probe', ['n'], ['count'], [[rows[0][0]]])\n",
        str(vault), str(tmp_path / "run"), executor,
        corpus_path=str(corpus),
    )
    if _namespace_unavailable(result):
        pytest.skip("platform sandbox cannot create a namespace here")
    assert isinstance(result, runner.AnalystEnvelope)
    assert result.ledger["query_count"] == 1
    assert result.citation_ledger["cite_calls"] == 2
    assert all(item["span"] for item in result.citation_ledger["passages"])
    assert result.ledger is not result.citation_ledger


def test_prompt_has_cite_only_for_configured_corpus():
    caps = {
        "rows_per_table": 1, "tables_per_run": 1, "cells_total": 1,
        "distinct_numeric_tokens": 1, "envelope_bytes": 1,
        "wall_clock_seconds": 1,
    }
    closed = analyst_prompt.build_analyst_prompt("q", "schema", caps=caps)
    open_prompt = analyst_prompt.build_analyst_prompt(
        "q", "schema", caps=caps, corpus_configured=True)
    assert "cite(" not in closed
    assert "cite(query, k=5, doc_id=None)" in open_prompt
    assert "verbatim corpus text" in open_prompt
    assert "never comes from memory" in open_prompt


def test_receiver_main_threads_cli_and_environment_corpus(tmp_path, monkeypatch):
    selected = []
    monkeypatch.setattr(receiver.VaultContext, "local",
                        lambda *args, **kwargs: object())
    monkeypatch.setenv("HEALTH_ADVISOR_CORPUS", "environment-corpus.db")
    monkeypatch.setitem(sys.modules, "uvicorn", type(
        "Uvicorn", (), {"run": staticmethod(lambda app, **kwargs: None)}))

    def fake_create_app(ctx, **kwargs):
        selected.append(kwargs)
        return object()

    assert receiver.main([
        "--vault", str(tmp_path / "vault.db"), "--corpus", "cli-corpus.db",
    ], app_factory=fake_create_app) == 0
    assert selected[-1]["analyst_corpus_path"] == "cli-corpus.db"

    selected.clear()
    assert receiver.main([
        "--vault", str(tmp_path / "vault.db"),
    ], app_factory=fake_create_app) == 0
    assert selected[-1]["analyst_corpus_path"] == "environment-corpus.db"
