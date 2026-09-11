"""Synthetic, identity-free acceptance tests for citation claim verification."""
from __future__ import annotations

import shutil
import sqlite3
import statistics
import time
from pathlib import Path

from health_advisor import agents
from health_advisor import deepdive_verify as DV
from health_advisor.corpus_build import build_corpus, sha256_text


def _entry(doc_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": "Synthetic document 0400",
        "authors": "Synthetic Author",
        "year": 2020,
        "doi": "10.0000/doc0400",
        "pmid": None,
        "source_url": "https://example.org/doc0400",
        "retrieved_at": "2026-08-30T09:00:00Z",
        "source_sha256": "0" * 64,
        "text_sha256": sha256_text(text),
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "redistributable": 1,
        "approver": "synthetic-reviewer",
        "approved_at": "2026-08-30",
        "notes": None,
    }


def _build_test_corpus(path: Path) -> Path:
    text = ("Doc0400 chunk zero evidence. " * 55
            + "Doc0400 chunk one evidence. " * 55)
    build_corpus(
        [_entry("doc0400", text)], [text], path,
        corpus_version=5, built_at="2026-08-30T00:00:00+00:00",
        read_only=False,
    )
    return path


def _source(*, doc_id="doc0400", chunk_ix=0,
            span="Doc0400 chunk zero evidence.", corpus_version=5) -> dict:
    return {"doc_id": doc_id, "chunk_ix": chunk_ix, "span": span,
            "corpus_version": corpus_version}


def _claim(source: dict) -> dict:
    return {"assertion": "The synthetic document states this.",
            "source": source}


def test_citation_refusal_set_is_seven_of_seven_with_exact_reasons(tmp_path):
    corpus = _build_test_corpus(tmp_path / "corpus.db")
    with sqlite3.connect(corpus) as conn:
        chunk_one_span = conn.execute(
            "SELECT body FROM chunks WHERE doc_id = ? AND chunk_ix = 1",
            ("doc0400",),
        ).fetchone()[0]
    chunk_one_span = "Doc0400 chunk one evidence."

    cases = [
        _source(doc_id="doc0400-missing"),
        _source(chunk_ix=91),
        _source(span="This sentence is not in the synthetic chunk."),
        _source(span=chunk_one_span),
        _source(span="The document claims that the evidence is different."),
        _source(corpus_version=4),
        {"span": "Doc0400 chunk zero evidence.", "corpus_version": 5},
    ]
    expected = [
        "citation does not resolve: doc0400-missing chunk 0",
        "citation does not resolve: doc0400 chunk 91",
        "citation span not found in doc0400 chunk 0",
        "citation span not found in doc0400 chunk 0",
        "citation span not found in doc0400 chunk 0",
        "citation minted against corpus_version 4, current is 5",
        "citation source has neither doc_id nor sequence",
    ]

    verdicts = [DV.verify_citation_claims("", [_claim(case)], corpus)
                for case in cases]
    assert [verdict["ok"] for verdict in verdicts] == [False] * 7
    assert [verdict["reason"] for verdict in verdicts] == expected


def test_good_citation_returns_renderer_metadata_and_verbatim_span(tmp_path):
    corpus = _build_test_corpus(tmp_path / "corpus.db")
    span = "  Doc0400   chunk zero evidence.  "
    verdict = DV.verify_citation_claims(
        "The synthetic document states this.",
        [_claim(_source(span=span))], corpus)

    assert verdict["ok"] is True
    resolved = verdict["citations"][0]
    assert {key: resolved[key] for key in
            ("title", "year", "doi", "license", "span")} == {
                "title": "Synthetic document 0400",
                "year": 2020,
                "doi": "10.0000/doc0400",
                "license": "CC-BY-4.0",
                "span": span,
            }


def test_orphan_chunk_is_refused_by_the_inner_join(tmp_path):
    built = _build_test_corpus(tmp_path / "built.db")
    orphan = tmp_path / "orphan.db"
    shutil.copy(built, orphan)
    # open_corpus intentionally refuses an orphaned file; this verifier must
    # instead open the forged copy raw and refuse the claim at its JOIN.
    with sqlite3.connect(orphan) as writer:
        writer.execute(
            "INSERT INTO chunks(doc_id, chunk_ix, body) VALUES (?, ?, ?)",
            ("orphan-doc", 0, "Synthetic orphan evidence."),
        )
    # Read the forged copy raw: open_corpus refuses an orphan before the
    # verifier can refuse the claim, so this test uses SQLite's read-only URI.
    raw = sqlite3.connect(f"file:{orphan}?mode=ro", uri=True)
    try:
        assert raw.execute(
            "SELECT body FROM chunks WHERE doc_id = ?", ("orphan-doc",)
        ).fetchone()[0] == "Synthetic orphan evidence."
    finally:
        raw.close()
    verdict = DV.verify_citation_claims(
        "Synthetic orphan evidence.",
        [_claim(_source(doc_id="orphan-doc", span="Synthetic orphan evidence."))],
        orphan,
    )
    assert verdict["ok"] is False
    assert verdict["reason"] == "citation does not resolve: orphan-doc chunk 0"


def test_split_claim_channel_keeps_figure_claims_unchanged():
    claim = {"metric": "synthetic", "value": 7,
             "source": {"sequence": 3, "path": "$.result.value"}}
    raw = '{"text":"Seven.","claims":[' + str(claim).replace("'", '"') + ']}'
    prose, claims = agents.split_claim_channel(raw)
    assert prose == "Seven."
    assert claims == [claim]


def test_orphan_goes_green_when_the_join_is_mutated_and_is_restored(tmp_path,
                                                                    monkeypatch):
    """Pin the acceptance mutation: deleting the JOIN makes the orphan hit."""
    built = _build_test_corpus(tmp_path / "built.db")
    orphan = tmp_path / "orphan.db"
    shutil.copy(built, orphan)
    with sqlite3.connect(orphan) as writer:
        writer.execute(
            "INSERT INTO chunks(doc_id, chunk_ix, body) VALUES (?, ?, ?)",
            ("orphan-doc", 0, "Synthetic orphan evidence."),
        )

    original = DV._CITATION_ROW_SQL
    monkeypatch.setattr(
        DV, "_CITATION_ROW_SQL",
        original.replace(
            "SELECT chunks.body, d.title, d.year, d.doi, d.license, d.approver",
            "SELECT chunks.body, NULL, NULL, NULL, 'CC-BY-4.0', 'synthetic-reviewer'",
        ).replace("  FROM chunks INNER JOIN docs d ON d.doc_id = chunks.doc_id",
                   "  FROM chunks"),
    )
    try:
        mutated = DV.verify_citation_claims(
            "Synthetic orphan evidence.",
            [_claim(_source(doc_id="orphan-doc",
                           span="Synthetic orphan evidence."))],
            orphan,
        )
    finally:
        # Restore the SQL text in the test itself so the mutation is
        # byte-identical before pytest tears the fixture down.
        monkeypatch.undo()
    assert mutated["ok"] is True
    assert DV._CITATION_ROW_SQL == original
    print("mutation: orphan GREEN without docs JOIN; SQL restored byte-identically")


def test_citation_verifier_median_latency_for_fifty_claims(tmp_path):
    corpus = _build_test_corpus(tmp_path / "corpus.db")
    claim = _claim(_source())
    samples = []
    for _ in range(50):
        started = time.perf_counter()
        verdict = DV.verify_citation_claims("", [claim], corpus)
        samples.append((time.perf_counter() - started) * 1000)
        assert verdict["ok"] is True
    print(f"citation median latency ms: {statistics.median(samples):.3f}")



def test_partition_claim_sources_routes_by_source_shape():
    from health_advisor.agents import partition_claim_sources, split_claim_channel

    figure = {"assertion": "46.3 jog minutes", "source": {"sequence": 2}}
    citation = {"assertion": "volume raises risk",
                "source": {"doc_id": "doc0400", "chunk_ix": 0, "span": "x", "corpus_version": 1}}
    both = {"assertion": "both", "source": {"sequence": 1, "doc_id": "doc0400"}}
    other = {"assertion": "neither", "source": {}}
    figures, citations, others = partition_claim_sources([figure, citation, both, other, "junk"])
    assert figures == [figure, both]
    assert citations == [citation]
    assert others == [other, "junk"]
    # split_claim_channel is untouched: every claim comes back in order.
    import json
    text, claims = split_claim_channel(json.dumps({"text": "t", "claims": [figure, citation]}))
    assert (text, claims) == ("t", [figure, citation])
