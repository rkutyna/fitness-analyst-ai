"""Focused tests for issue #22's local LSA index and RRF seam."""
from __future__ import annotations

import inspect
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from health_advisor import analyst_corpus as ac
from health_advisor.corpus_build import CORPUS_SCHEMA


def _make_corpus(path: Path, chunks: int, *, chunks_per_doc: int = 1) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(CORPUS_SCHEMA)
    conn.execute("INSERT INTO corpus_meta VALUES ('corpus_version', '1')")
    for ix in range(chunks):
        doc_number = ix // chunks_per_doc
        doc_id = f"d{doc_number:04d}"
        if ix % chunks_per_doc == 0:
            conn.execute(
                "INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, f"Document {doc_number}", "Author", 2026,
                 None, None, "https://example.test", "2026-09-12",
                 "0" * 64, "1" * 64, "CC-BY-4.0", None, 1,
                 "test", "2026-09-12", None),
            )
        body = (
            f"chunk {ix} injury biomechanics clinical training recovery "
            f"aerobic endurance evidence context unique{ix}"
        )
        conn.execute(
            "INSERT INTO chunks(doc_id, chunk_ix, body) VALUES (?,?,?)",
            (doc_id, ix % chunks_per_doc, body),
        )
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def large_corpus(tmp_path):
    conn = ac.open_corpus(_make_corpus(tmp_path / "large.db", 300))
    try:
        yield conn
    finally:
        conn.close()


def _keys(passages):
    return [(passage.doc_id, passage.chunk_ix) for passage in passages]


def test_rrf_changes_the_order_and_bm25_mode_is_reachable(large_corpus):
    bm25 = ac.cite(large_corpus, "unique1", state=ac.CiteState(),
                   ranking="bm25")
    fused = ac.cite(large_corpus, "unique1", state=ac.CiteState())
    assert _keys(bm25) != _keys(fused)
    assert all(p.score is not None for p in bm25)
    assert any(p.score is None for p in fused)


def test_similarity_is_not_a_constant_and_fused_scores_are_not_exposed(
        large_corpus):
    index = ac._embedding_index(large_corpus)
    ranked = index.rank(["injury", "biomechanics"], limit=20)
    assert index.usable
    assert len({round(score, 12) for _, score in ranked}) > 1
    passages = ac.cite(large_corpus, "unique1",
                       state=ac.CiteState())
    assert all("score" in passage.as_dict() for passage in passages)
    assert all(p.score is None or p.score < 0 for p in passages)


def test_same_process_and_subprocess_rankings_are_identical(large_corpus,
                                                             tmp_path):
    query = "injury biomechanics"
    first = _keys(ac.cite(large_corpus, query, state=ac.CiteState()))
    second = _keys(ac.cite(large_corpus, query, state=ac.CiteState()))
    assert first == second

    db_path = tmp_path / "cross-process.db"
    _make_corpus(db_path, 300)
    script = (
        "import json, sys; "
        "from health_advisor import analyst_corpus as a; "
        "c=a.open_corpus(sys.argv[1]); "
        "print(json.dumps([(p.doc_id,p.chunk_ix) for p in "
        "a.cite(c, 'injury biomechanics', state=a.CiteState())]))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    output = subprocess.check_output(
        [sys.executable, "-c", script, str(db_path)], env=env, text=True)
    assert first == [tuple(item) for item in json.loads(output)]


@pytest.mark.parametrize(
    ("name", "chunks", "chunks_per_doc", "expected"),
    [
        ("empty", 0, 1, "empty"),
        ("one-document", 2, 2, "bm25"),
        ("one-chunk", 1, 1, "bm25"),
        ("fewer-than-requested-dimensions", 3, 1, "bm25"),
    ],
)
def test_degenerate_corpora_have_designed_answers(
        tmp_path, name, chunks, chunks_per_doc, expected):
    conn = ac.open_corpus(_make_corpus(
        tmp_path / f"{name}.db", chunks, chunks_per_doc=chunks_per_doc))
    try:
        index = ac._embedding_index(conn)
        passages = ac.cite(conn, "injury", state=ac.CiteState())
        if expected == "empty":
            assert passages == []
            assert index.fallback_reason == "empty_corpus"
        else:
            assert passages
            assert not index.usable
            assert index.fallback_reason == "fewer_chunks_than_svd_dimensions"
    finally:
        conn.close()


def test_open_builds_in_memory_without_mutating_the_corpus(tmp_path):
    path = _make_corpus(tmp_path / "immutable.db", 300)
    before = path.read_bytes()
    conn = ac.open_corpus(path)
    try:
        index = ac._embedding_index(conn)
        assert index.build_seconds >= 0.0
    finally:
        conn.close()
    assert path.read_bytes() == before


def test_cite_has_no_score_threshold_parameter():
    signature = inspect.signature(ac.cite)
    assert sum(name in str(signature) for name in
               ("min_score", "max_score", "score_cutoff")) == 0
