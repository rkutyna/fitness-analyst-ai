"""B1 — the corpus builder: chunk determinism, schema, versioning, ship filter.

The property this file exists to defend is `Done when` 4: rebuilding from the
same inputs must produce identical `text_sha256` for every doc and identical
`(doc_id, chunk_ix)` boundaries. If a boundary moves, every stored citation
silently breaks and nothing raises — so it is checked programmatically against
a second build at a different path, never by eye.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from health_advisor.corpus_build import (
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    BuildResult,
    RegistryRefusal,
    build_corpus,
    check_corpus_integrity,
    chunk_boundaries,
    chunk_text,
    corpus_file_sha256,
    extract_text,
    next_corpus_version,
    read_corpus_version,
    sha256_text,
)


# --------------------------------------------------------------------------- #
# Fixture helpers — every corpus in this file is built under tmp_path. Nothing
# here may go near data/health.db; conftest's autouse guard makes naming it an
# AssertionError, and these tests never name any path they did not create.
# --------------------------------------------------------------------------- #

def make_text(seed: str, chars: int) -> str:
    """Deterministic pseudo-prose of a given length. No RNG: an RNG in a test
    for determinism would be testing the seed, not the chunker."""
    words = [f"{seed}{i:05d}" for i in range(chars // 6 + 2)]
    return " ".join(words)[:chars]


def entry(doc_id: str, text: str, **over) -> dict:
    row = {
        "doc_id": doc_id,
        "title": f"Title of {doc_id}",
        "authors": "Author A; Author B",
        "year": 2019,
        "doi": f"10.0000/{doc_id}",
        "pmid": None,
        "source_url": f"https://example.org/{doc_id}",
        "retrieved_at": "2026-08-30T09:00:00Z",
        "source_sha256": "0" * 64,
        "text_sha256": sha256_text(text),
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "redistributable": 1,
        "approver": "reviewer",
        "approved_at": "2026-08-30",
        "notes": None,
    }
    row.update(over)
    return row


def build(tmp_path: Path, pairs, name="corpus.db", **kw) -> BuildResult:
    entries = [e for e, _ in pairs]
    texts = [t for _, t in pairs]
    kw.setdefault("corpus_version", 1)
    kw.setdefault("built_at", "2026-08-30T00:00:00+00:00")
    return build_corpus(entries, texts, tmp_path / name, **kw)


# --------------------------------------------------------------------------- #
# Chunking — the pure function
# --------------------------------------------------------------------------- #

def test_chunk_parameters_are_the_documented_ones():
    assert CHUNK_CHARS == 1200
    assert CHUNK_OVERLAP == 200


def test_short_text_is_one_chunk():
    text = make_text("a", 500)
    assert chunk_text(text) == [text]


def test_empty_text_is_no_chunks():
    assert chunk_text("") == []


def test_chunk_size_and_overlap_hold():
    text = make_text("b", 5000)
    chunks = chunk_text(text)
    assert all(len(c) <= CHUNK_CHARS for c in chunks)
    # Consecutive chunks share their overlap verbatim.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-CHUNK_OVERLAP:] == nxt[:CHUNK_OVERLAP]


def test_chunks_cover_the_text_without_loss():
    """Every character of the document appears in at least one chunk, and each
    chunk is a verbatim slice at its documented offset."""
    text = make_text("c", 7321)
    chunks = chunk_text(text)
    stride = CHUNK_CHARS - CHUNK_OVERLAP
    covered = bytearray(len(text))
    for i, chunk in enumerate(chunks):
        start = i * stride
        assert text[start:start + len(chunk)] == chunk
        for j in range(start, start + len(chunk)):
            covered[j] = 1
    assert text.startswith(chunks[0])
    assert text.endswith(chunks[-1])
    assert all(covered), "a character of the document is in no chunk"


def test_chunking_is_a_pure_function_of_the_input():
    """Same input, many calls, byte-identical output every time."""
    text = make_text("d", 9000)
    first = chunk_text(text)
    for _ in range(50):
        assert chunk_text(text) == first


def test_chunking_does_not_depend_on_locale_or_environment(monkeypatch):
    text = make_text("e", 6000)
    before = chunk_text(text)
    monkeypatch.setenv("LC_ALL", "tr_TR.UTF-8")
    monkeypatch.setenv("LANG", "tr_TR.UTF-8")
    monkeypatch.setenv("PYTHONHASHSEED", "12345")
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    assert chunk_text(text) == before


def test_chunk_boundaries_are_stable_under_unicode():
    """Multi-byte characters must not shift a boundary: slicing is on code
    points, so a chunk index means the same thing for any script."""
    text = ("αβγδε" * 400) + ("日本語テキスト" * 300)
    chunks = chunk_text(text)
    assert chunks == chunk_text(text)
    assert "".join(c[:CHUNK_CHARS - CHUNK_OVERLAP] for c in chunks[:-1]) == \
        text[:(len(chunks) - 1) * (CHUNK_CHARS - CHUNK_OVERLAP)]


# --------------------------------------------------------------------------- #
# Schema and build product
# --------------------------------------------------------------------------- #

def test_built_corpus_has_the_design_schema(tmp_path):
    t = make_text("f", 3000)
    result = build(tmp_path, [(entry("d1", t), t)])
    conn = sqlite3.connect(f"file:{result.path}?mode=ro", uri=True)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        assert {"corpus_meta", "docs", "chunks"} <= names
        cols = [r[1] for r in conn.execute("PRAGMA table_info(docs)")]
        assert cols == [
            "doc_id", "title", "authors", "year", "doi", "pmid",
            "source_url", "retrieved_at", "source_sha256", "text_sha256",
            "license", "license_url", "redistributable",
            "approver", "approved_at", "notes"]
        notnull = {r[1]: r[3] for r in conn.execute("PRAGMA table_info(docs)")}
        for required in ("title", "source_url", "retrieved_at", "source_sha256",
                         "text_sha256", "license", "redistributable",
                         "approver", "approved_at"):
            assert notnull[required] == 1, f"{required} must be NOT NULL"
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='chunks'").fetchone()[0]
        assert "fts5" in sql.lower()
        assert "porter unicode61" in sql
    finally:
        conn.close()


def test_build_runs_and_exposes_the_corpus_integrity_check(tmp_path):
    text = make_text("integrity", 3000)
    result = build(tmp_path, [(entry("d1", text), text)])
    assert result.orphan_chunk_count == 0
    conn = sqlite3.connect(f"file:{result.path}?mode=ro", uri=True)
    try:
        assert check_corpus_integrity(conn) == 0
    finally:
        conn.close()


def test_corpus_meta_carries_version_and_provenance(tmp_path):
    t = make_text("g", 2000)
    result = build(tmp_path, [(entry("d1", t), t)], corpus_version=7)
    conn = sqlite3.connect(f"file:{result.path}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT key, value FROM corpus_meta"))
    finally:
        conn.close()
    assert meta["corpus_version"] == "7"
    assert meta["chunk_chars"] == "1200"
    assert meta["chunk_overlap"] == "200"
    assert len(meta["builder_sha"]) == 64
    assert meta["built_at"]
    assert read_corpus_version(result.path) == 7


def test_built_corpus_is_mode_0444(tmp_path):
    t = make_text("h", 2000)
    result = build(tmp_path, [(entry("d1", t), t)])
    observed = os.stat(result.path).st_mode & 0o777
    assert oct(observed) == "0o444", oct(observed)
    assert result.mode == 0o444


def test_a_0444_corpus_refuses_a_write(tmp_path):
    """The corpus is read-only in the filesystem, not only by convention."""
    t = make_text("i", 2000)
    result = build(tmp_path, [(entry("d1", t), t)])
    with pytest.raises(sqlite3.OperationalError):
        conn = sqlite3.connect(f"file:{result.path}?mode=ro", uri=True)
        try:
            conn.execute(
                "INSERT INTO chunks(doc_id, chunk_ix, body) VALUES ('x', 0, 'y')")
        finally:
            conn.close()


def test_chunk_ix_is_zero_based_and_contiguous_per_doc(tmp_path):
    ta, tb = make_text("j", 5000), make_text("k", 3000)
    result = build(tmp_path, [(entry("d1", ta), ta), (entry("d2", tb), tb)])
    conn = sqlite3.connect(f"file:{result.path}?mode=ro", uri=True)
    try:
        for doc_id in ("d1", "d2"):
            ixs = sorted(int(r[0]) for r in conn.execute(
                "SELECT chunk_ix FROM chunks WHERE doc_id = ?", (doc_id,)))
            assert ixs == list(range(len(ixs)))
            assert ixs[0] == 0
    finally:
        conn.close()


def test_the_indexed_body_is_verbatim_source_text(tmp_path):
    """A citation span is verified as a verbatim substring of a chunk, so the
    chunk must be a verbatim substring of the document."""
    t = make_text("l", 4000)
    result = build(tmp_path, [(entry("d1", t), t)])
    conn = sqlite3.connect(f"file:{result.path}?mode=ro", uri=True)
    try:
        bodies = [r[0] for r in conn.execute(
            "SELECT body FROM chunks WHERE doc_id='d1' ORDER BY chunk_ix")]
    finally:
        conn.close()
    assert bodies
    for body in bodies:
        assert body in t


def test_fts5_match_and_bm25_work_on_a_built_corpus(tmp_path):
    t = "vo2max improves with high-intensity interval training. " + make_text("m", 3000)
    result = build(tmp_path, [(entry("d1", t), t)])
    conn = sqlite3.connect(f"file:{result.path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT doc_id, chunk_ix FROM chunks WHERE chunks MATCH ? "
            "ORDER BY bm25(chunks) LIMIT 5", ("vo2max",)).fetchall()
    finally:
        conn.close()
    assert rows and rows[0][0] == "d1"


# --------------------------------------------------------------------------- #
# `Done when` 4 — rebuild determinism
# --------------------------------------------------------------------------- #

def _doc_shas(path: Path) -> dict[str, str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {r[0]: r[1] for r in conn.execute(
            "SELECT doc_id, text_sha256 FROM docs")}
    finally:
        conn.close()


def test_rebuild_from_identical_inputs_is_identical(tmp_path):
    pairs = []
    for i in range(25):
        t = make_text(f"doc{i}", 900 + i * 613)
        pairs.append((entry(f"d{i:03d}", t), t))

    a = build(tmp_path, pairs, name="a.db")
    b = build(tmp_path, pairs, name="b.db")

    assert a.path != b.path
    assert _doc_shas(a.path) == _doc_shas(b.path)
    assert a.doc_count == b.doc_count == 25

    ba, bb = chunk_boundaries(a.path), chunk_boundaries(b.path)
    differences = [x for x in ba if x not in set(bb)] + \
                  [x for x in bb if x not in set(ba)]
    assert differences == [], differences
    assert len(ba) == len(bb) == a.chunk_count


def test_rebuild_is_identical_when_the_registry_order_changes(tmp_path):
    """`chunk_ix` is per doc, so it must not depend on where a doc sat in the
    registry a human happened to write."""
    pairs = []
    for i in range(10):
        t = make_text(f"o{i}", 2000 + i * 311)
        pairs.append((entry(f"e{i:03d}", t), t))
    a = build(tmp_path, pairs, name="fwd.db")
    b = build(tmp_path, list(reversed(pairs)), name="rev.db")
    assert chunk_boundaries(a.path) == chunk_boundaries(b.path)
    assert _doc_shas(a.path) == _doc_shas(b.path)


def test_builder_sha_and_corpus_sha_are_reported(tmp_path):
    t = make_text("p", 2500)
    result = build(tmp_path, [(entry("d1", t), t)])
    assert len(result.builder_sha) == 64
    assert len(result.corpus_sha256) == 64
    assert result.corpus_sha256 == corpus_file_sha256(result.path)


# --------------------------------------------------------------------------- #
# Ship filter — design §4.3, licensing is a column
# --------------------------------------------------------------------------- #

def test_local_corpus_admits_non_redistributable(tmp_path):
    t1, t2 = make_text("q", 1500), make_text("r", 1500)
    result = build(tmp_path, [
        (entry("open", t1, redistributable=1), t1),
        (entry("paywalled", t2, redistributable=0, license="publisher-proprietary"), t2),
    ])
    assert result.doc_count == 2
    assert result.excluded_non_redistributable == 0


def test_shippable_corpus_excludes_and_reports(tmp_path):
    t1, t2, t3 = (make_text("s", 1500), make_text("t", 1500), make_text("u", 1500))
    result = build(tmp_path, [
        (entry("open1", t1, redistributable=1), t1),
        (entry("paywalled", t2, redistributable=0, license="publisher-proprietary"), t2),
        (entry("open2", t3, redistributable=1), t3),
    ], shippable=True)
    assert result.doc_count == 2
    assert result.excluded_non_redistributable == 1
    assert result.excluded_doc_ids == ("paywalled",)
    conn = sqlite3.connect(f"file:{result.path}?mode=ro", uri=True)
    try:
        ids = {r[0] for r in conn.execute("SELECT doc_id FROM docs")}
        chunk_ids = {r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM chunks")}
    finally:
        conn.close()
    assert ids == {"open1", "open2"}
    # An excluded document must not leave chunks behind — that would ship its
    # text while hiding its provenance row.
    assert chunk_ids == {"open1", "open2"}


# --------------------------------------------------------------------------- #
# Versioning
# --------------------------------------------------------------------------- #

def test_next_corpus_version_starts_at_one_and_increments(tmp_path):
    assert next_corpus_version(None) == 1
    assert next_corpus_version(tmp_path / "absent.db") == 1
    t = make_text("v", 1500)
    first = build(tmp_path, [(entry("d1", t), t)], name="v1.db", corpus_version=1)
    assert next_corpus_version(first.path) == 2
    second = build(tmp_path, [(entry("d1", t), t)], name="v2.db",
                   corpus_version=next_corpus_version(first.path))
    assert read_corpus_version(second.path) == 2


def test_corpus_version_must_be_a_positive_int(tmp_path):
    t = make_text("w", 1500)
    for bad in (0, -3, "2", True):
        with pytest.raises(RegistryRefusal) as exc:
            build(tmp_path, [(entry("d1", t), t)], corpus_version=bad)
        assert exc.value.code == "bad_corpus_version"


def test_a_rebuild_replaces_a_read_only_corpus_in_place(tmp_path):
    """0444 must not make the next build fail — the builder owns the file."""
    t = make_text("x", 1500)
    build(tmp_path, [(entry("d1", t), t)], name="same.db", corpus_version=1)
    again = build(tmp_path, [(entry("d1", t), t)], name="same.db", corpus_version=2)
    assert read_corpus_version(again.path) == 2
    assert oct(os.stat(again.path).st_mode & 0o777) == "0o444"
    assert not (tmp_path / "same.db.building").exists()


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def test_extract_text_reads_plain_text_verbatim(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("line one\nline two\n", encoding="utf-8")
    assert extract_text(p) == "line one\nline two\n"


def test_extract_text_walks_xml_in_document_order(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(
        "<article><front><title>T</title></front>"
        "<body><p>alpha </p><p>beta</p></body></article>", encoding="utf-8")
    out = extract_text(p)
    assert out == "Talpha beta"
    assert extract_text(p) == out  # deterministic


def test_extract_text_refuses_a_pdf(tmp_path):
    """No PDF branch exists, and its absence is a security decision (§1.5)."""
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.7\n")
    with pytest.raises(RegistryRefusal) as exc:
        extract_text(p)
    assert exc.value.code == "unsupported_text_format"


# --------------------------------------------------------------------------- #
# Nothing here may set temp_store as a "fix"
# --------------------------------------------------------------------------- #

def test_the_builder_does_not_set_temp_store_as_a_performance_fix():
    """Measured: FTS5 ORDER BY bm25() over 20,000 chunks needs no writable temp
    space, and temp_store=MEMORY was ~2x slower. A pragma justified by a hazard
    that does not exist is a pessimisation with a comment on it."""
    import ast
    src = Path(__file__).resolve().parents[1] / "health_advisor" / "corpus_build.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    executed = [
        node.value
        for call in ast.walk(tree) if isinstance(call, ast.Call)
        for node in ast.walk(call)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and "temp_store" in node.value.lower()
    ]
    assert executed == [], executed
