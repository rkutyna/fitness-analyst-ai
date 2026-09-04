"""Build a vetted evidence corpus: schema, chunking, FTS5 index, versioning.

The one rule, transposed (design §0.2):

    The corpus owns the evidence. The model retrieves and cites; it never
    states a finding from memory.

A citation is keyed ``(doc_id, chunk_ix, span)`` and is verified later by
re-reading this file and requiring ``span`` to be a verbatim substring of the
chunk. That verification is only worth anything if two things hold, and this
module exists to make both of them mechanical rather than procedural:

1. **Chunking is a pure deterministic function of the input text.** No clock,
   no randomness, no dict iteration order, no locale. ``chunk_ix`` is half the
   citation key; if a boundary moves between builds of the same input, every
   stored citation silently breaks and nothing raises. See `chunk_text`.
2. **The registry cannot be bypassed.** `docs` is `NOT NULL` on approver,
   license and both checksums, and `validate_entry` refuses anything the
   constraint would otherwise have to tolerate. "Vetted" is a schema
   constraint for `docs`; `chunks` are enforced at build time and open time
   (design §1.3, §4.2).

This module is the library half of B1. The entry point is
`scripts/corpus_ingest.py`, which is run by a human and — deliberately — has no
argument that accepts a health-vault path (design §4.1). Nothing in the analyst
loop imports this module: building is a write path, and the corpus is opened
read-only everywhere else.

On ``PRAGMA temp_store``: it is deliberately NOT set here, and its absence is
the considered choice rather than an oversight. Measured 2026-08-30 against a
corpus built by this module — 420 docs / 23,100 chunks / 54.0 MB — FTS5
``ORDER BY bm25()`` returned in **0.2 ms with TMPDIR pointed at an unwritable
directory**. There is therefore no writable-temp hazard for a pragma to close.
An earlier independent measurement additionally found ``temp_store=MEMORY``
about 2x SLOWER on this workload. A pragma justified by a hazard that does not
exist is a pessimisation with a comment on it, so there is neither.
`tests/test_corpus_build.py` asserts no executed statement mentions it.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

__all__ = [
    "CHUNK_CHARS",
    "CHUNK_OVERLAP",
    "CORPUS_SCHEMA",
    "BuildResult",
    "RegistryRefusal",
    "build_corpus",
    "check_corpus_integrity",
    "chunk_text",
    "corpus_file_sha256",
    "extract_text",
    "next_corpus_version",
    "read_corpus_version",
    "sha256_text",
    "validate_entry",
]

# --------------------------------------------------------------------------- #
# Schema — design §1.3, verbatim. Do not reorder or relax a `docs` NOT NULL:
# those constraints are the registry's vetting policy; `chunks` are checked by
# the builder and by the parent when the corpus opens.
# --------------------------------------------------------------------------- #

CORPUS_SCHEMA = """
CREATE TABLE corpus_meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE docs(
  doc_id TEXT PRIMARY KEY,
  title TEXT NOT NULL, authors TEXT, year INTEGER,
  doi TEXT, pmid TEXT,
  source_url TEXT NOT NULL, retrieved_at TEXT NOT NULL,
  source_sha256 TEXT NOT NULL,
  text_sha256 TEXT NOT NULL,
  license TEXT NOT NULL, license_url TEXT, redistributable INTEGER NOT NULL,
  approver TEXT NOT NULL, approved_at TEXT NOT NULL,
  notes TEXT);
CREATE VIRTUAL TABLE chunks USING fts5(
  doc_id UNINDEXED, chunk_ix UNINDEXED, body,
  tokenize='porter unicode61');
"""

# Design §1.4. Large enough that a retrieved span has surrounding context to be
# read against; small enough that a cited chunk is checkable in one glance.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
_STRIDE = CHUNK_CHARS - CHUNK_OVERLAP  # 1000

# Columns of `docs`, in schema order. Insert order is fixed here rather than
# taken from a dict, so the build never depends on mapping iteration order.
DOC_COLUMNS: tuple[str, ...] = (
    "doc_id", "title", "authors", "year", "doi", "pmid",
    "source_url", "retrieved_at", "source_sha256", "text_sha256",
    "license", "license_url", "redistributable",
    "approver", "approved_at", "notes",
)

# The subset the registry may not omit. Design §4.2: approver, license and both
# checksums are the fields the citation verifier would otherwise have to
# tolerate the absence of.
REQUIRED_FIELDS: tuple[str, ...] = (
    "doc_id", "title", "source_url", "retrieved_at",
    "source_sha256", "text_sha256", "license", "redistributable",
    "approver", "approved_at",
)


class RegistryRefusal(Exception):
    """A typed refusal from the vetting registry.

    Carries a stable machine-readable ``code`` and a one-line ``reason``. It is
    an exception so a build cannot continue past one by ignoring a return
    value, but callers at an entry point print ``.reason`` and exit — a
    traceback is a data channel in this codebase and is treated as a defect.
    """

    def __init__(self, code: str, detail: str, *, doc_id: str | None = None):
        self.code = code
        self.doc_id = doc_id
        where = f" [{doc_id}]" if doc_id else ""
        self.reason = f"corpus.registry.{code}{where}: {detail}"
        super().__init__(self.reason)


@dataclass(frozen=True)
class BuildResult:
    """What a build produced. Every number a `Done when` asks for is here."""
    path: Path
    corpus_version: int
    built_at: str
    builder_sha: str
    doc_count: int
    chunk_count: int
    orphan_chunk_count: int
    excluded_non_redistributable: int
    excluded_doc_ids: tuple[str, ...]
    corpus_sha256: str
    mode: int
    shippable: bool


# --------------------------------------------------------------------------- #
# Chunking — the pure function the whole design turns on
# --------------------------------------------------------------------------- #

def chunk_text(text: str) -> list[str]:
    """Split ``text`` into ~1,200-char chunks with 200 chars of overlap.

    PURE AND DETERMINISTIC BY CONSTRUCTION. It is character slicing on a fixed
    stride and nothing else: no regex whose behaviour varies with locale, no
    sentence splitter, no clock, no randomness, no set or dict iteration. Given
    the same `str` it returns the same list on any machine, in any locale, in
    any interpreter run — which is what makes ``chunk_ix`` a citation key
    rather than a build artifact.

    Boundaries are at multiples of ``_STRIDE`` (1000). The final chunk is
    whatever remains and may overlap its predecessor by more than 200 chars;
    that is deliberate, because the alternative (a short tail chunk) makes the
    last boundary depend on ``len(text) % stride``, which is harder to reason
    about and no more stable.

    Whitespace is NOT normalised. ``text_sha256`` pins exactly the bytes that
    were indexed (design §1.5: "the thing cited must be the thing checked"), so
    any normalisation has to happen in extraction, before the checksum, not
    here.
    """
    if not text:
        return []
    n = len(text)
    if n <= CHUNK_CHARS:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += _STRIDE
    return chunks


def sha256_text(text: str) -> str:
    """SHA-256 of the UTF-8 encoding of the text actually indexed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_text(path: Path) -> str:
    """Read one already-acquired artifact as plain text.

    Extraction happens HERE, on a human's machine, outside the sandbox, before
    anything is indexed — design §1.5. No PDF branch exists and none should be
    added: a PDF parser is a binary-format attack surface fed documents fetched
    from the web, and no PDF library is installed. Convert to text first.

    ``.xml`` (PubMed Central OA / JATS) is walked with `xml.etree`, which is
    stdlib and whose document-order traversal is deterministic.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xml":
        root = ET.parse(path).getroot()
        # `itertext()` is document order — deterministic, no dict iteration.
        parts = [t for t in root.itertext()]
        return "".join(parts)
    if suffix in (".txt", ".md", ".text", ""):
        return path.read_text(encoding="utf-8")
    raise RegistryRefusal(
        "unsupported_text_format",
        f"cannot extract text from {path.name!r}; supported: .txt, .md, .xml "
        f"(no PDF branch exists by design — convert to text first)",
    )


# --------------------------------------------------------------------------- #
# Registry validation — a constraint plus a refusal, not a review step
# --------------------------------------------------------------------------- #

def _is_iso8601(value: str) -> bool:
    """True when `value` parses as an ISO-8601 date or datetime.

    `datetime.fromisoformat` on 3.11 accepts the full ISO-8601 grammar
    including a trailing ``Z``. It does NOT accept ``2026/08/30``, a
    month-name form, or an out-of-range component, which is exactly the
    discrimination the registry needs.
    """
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return False
    return True


def validate_entry(entry: dict, text: str, *, seen_doc_ids: set[str]) -> None:
    """Refuse a registry entry that a corpus must not be able to contain.

    Raises `RegistryRefusal` with a typed code, or returns None. Checks run in
    the fixed order below so that an entry with two defects always refuses on
    the same one — a refusal reason that varies run to run is not a typed
    string, it is a coin flip.

    ``seen_doc_ids`` is mutated: the FIRST occurrence of a ``doc_id`` is
    admitted and every later one is refused, so the refusal names the copy, not
    the original.
    """
    doc_id = entry.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise RegistryRefusal(
            "missing_doc_id",
            "registry entry has no doc_id; a citation key cannot be minted without one",
        )
    doc_id = doc_id.strip()

    for field_name in REQUIRED_FIELDS:
        if field_name == "doc_id":
            continue
        value = entry.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise RegistryRefusal(
                f"missing_{field_name}",
                f"required registry field {field_name!r} is absent or empty; "
                f"docs.{field_name} is NOT NULL and vetting requires it",
                doc_id=doc_id,
            )

    if doc_id in seen_doc_ids:
        raise RegistryRefusal(
            "duplicate_doc_id",
            f"doc_id {doc_id!r} already admitted in this build; doc_ids are "
            f"stable and never reused, so a duplicate would collide the citation key",
            doc_id=doc_id,
        )

    if not _is_iso8601(str(entry["retrieved_at"])):
        raise RegistryRefusal(
            "bad_retrieved_at",
            f"retrieved_at {entry['retrieved_at']!r} is not ISO-8601; "
            f"provenance timestamps must be machine-comparable",
            doc_id=doc_id,
        )
    if not _is_iso8601(str(entry["approved_at"])):
        raise RegistryRefusal(
            "bad_approved_at",
            f"approved_at {entry['approved_at']!r} is not ISO-8601; "
            f"provenance timestamps must be machine-comparable",
            doc_id=doc_id,
        )

    redistributable = entry["redistributable"]
    if isinstance(redistributable, bool) or not isinstance(redistributable, int):
        raise RegistryRefusal(
            "bad_redistributable",
            f"redistributable {redistributable!r} is not an int 0 or 1; "
            f"licensing is a column, not a convention",
            doc_id=doc_id,
        )
    if redistributable not in (0, 1):
        raise RegistryRefusal(
            "bad_redistributable",
            f"redistributable {redistributable!r} is not 0 or 1",
            doc_id=doc_id,
        )

    year = entry.get("year")
    if year is not None and not isinstance(year, int):
        raise RegistryRefusal(
            "bad_year", f"year {year!r} is not an integer", doc_id=doc_id)

    if not text or not text.strip():
        raise RegistryRefusal(
            "empty_text",
            "extracted text is empty; an unretrievable document cannot be cited "
            "and must not occupy a doc_id",
            doc_id=doc_id,
        )

    actual = sha256_text(text)
    declared = str(entry["text_sha256"]).strip().lower()
    if actual != declared:
        raise RegistryRefusal(
            "text_sha256_mismatch",
            f"declared text_sha256 {declared[:16]}... does not match the text "
            f"being indexed ({actual[:16]}...); the thing cited must be the thing checked",
            doc_id=doc_id,
        )

    seen_doc_ids.add(doc_id)


# --------------------------------------------------------------------------- #
# Versioning — design §1.6
# --------------------------------------------------------------------------- #

def read_corpus_version(corpus_path: str | Path) -> int | None:
    """The `corpus_version` stamped into an existing corpus, or None."""
    path = Path(corpus_path)
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT value FROM corpus_meta WHERE key = 'corpus_version'").fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def next_corpus_version(previous_corpus: str | Path | None) -> int:
    """1 for a first build, else one past the previous corpus's version.

    Monotonic because `(doc_id, chunk_ix)` is only stable WITHIN a version
    (design §1.6): a citation minted under version N must never be silently
    re-checked against version N+1, and the only way a verifier can notice is
    if the number went up.
    """
    if previous_corpus is None:
        return 1
    prev = read_corpus_version(previous_corpus)
    return 1 if prev is None else prev + 1


def corpus_file_sha256(path: str | Path) -> str:
    """SHA-256 of the whole finished corpus file.

    Not stored inside the file it hashes — it goes in the run record beside
    `vault_version` (design §1.6). Returned by `build_corpus` for that purpose.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def builder_sha256() -> str:
    """SHA-256 of this module's source — the `builder_sha` in `corpus_meta`.

    The chunker is the thing a citation depends on. Recording which chunker
    built a corpus is how a later verifier can tell "the boundaries moved"
    from "the builder changed".
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


_ORPHAN_CHUNK_COUNT_SQL = (
    "SELECT COUNT(*) FROM chunks WHERE doc_id NOT IN "
    "(SELECT doc_id FROM docs)"
)


def check_corpus_integrity(conn: sqlite3.Connection) -> int:
    """Return the number of indexed chunks without a vetted ``docs`` row.

    This is deliberately a query over the opened corpus, not a check of the
    builder's counters.  The builder runs it before publishing a file and the
    parent runs it again when opening one.
    """
    return int(conn.execute(_ORPHAN_CHUNK_COUNT_SQL).fetchone()[0])


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build_corpus(
    entries: Sequence[dict],
    texts: Sequence[str],
    out_path: str | Path,
    *,
    corpus_version: int,
    shippable: bool = False,
    built_at: str | None = None,
    read_only: bool = True,
) -> BuildResult:
    """Validate every entry, then write a fresh corpus at `out_path`.

    `entries` and `texts` are positionally paired; `texts[i]` is the extracted
    plain text for `entries[i]`. Both are sequences, never mappings, so the
    build order cannot depend on dict iteration.

    Validation runs over ALL entries first and refuses on the first defect.
    A partially built corpus is never left behind: the file is written to a
    temporary sibling and renamed only once every row is in.

    `shippable=True` applies the ship filter of design §4.3 — only
    `redistributable = 1` is admitted, and the count excluded is reported.
    Exclusion is not refusal: a local corpus may legitimately hold documents
    that may not be redistributed.

    The finished file is chmod 444 unless `read_only=False`.
    """
    out_path = Path(out_path)
    if len(entries) != len(texts):
        raise RegistryRefusal(
            "entry_text_length_mismatch",
            f"{len(entries)} registry entries against {len(texts)} texts",
        )
    if not isinstance(corpus_version, int) or isinstance(corpus_version, bool):
        raise RegistryRefusal(
            "bad_corpus_version", f"corpus_version {corpus_version!r} is not an int")
    if corpus_version < 1:
        raise RegistryRefusal(
            "bad_corpus_version",
            f"corpus_version {corpus_version} is not >= 1; versions increase monotonically")

    # Pass 1: vetting. Every entry is checked before anything is written, so a
    # refusal never leaves a half-built corpus on disk.
    seen: set[str] = set()
    for entry, text in zip(entries, texts):
        validate_entry(entry, text, seen_doc_ids=seen)

    # Pass 2: the ship filter. A column, not a convention (design §4.3).
    admitted: list[tuple[dict, str]] = []
    excluded_ids: list[str] = []
    for entry, text in zip(entries, texts):
        if shippable and int(entry["redistributable"]) != 1:
            excluded_ids.append(str(entry["doc_id"]).strip())
            continue
        admitted.append((entry, text))

    # Sorted by doc_id so the on-disk row order is a property of the corpus
    # contents and not of the order a human happened to list them in. This
    # cannot move a chunk boundary — chunk_ix is per doc — but it makes two
    # builds of the same set of documents lay out identically.
    admitted.sort(key=lambda pair: str(pair[0]["doc_id"]).strip())

    built_at = built_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    b_sha = builder_sha256()

    tmp_path = out_path.with_name(out_path.name + ".building")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for stale in (tmp_path, out_path):
        if stale.exists():
            stale.chmod(0o644)
            stale.unlink()

    chunk_count = 0
    orphan_chunk_count = 0
    conn = sqlite3.connect(tmp_path)
    try:
        conn.executescript(CORPUS_SCHEMA)
        conn.executemany(
            "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
            [("corpus_version", str(corpus_version)),
             ("built_at", built_at),
             ("builder_sha", b_sha),
             ("chunk_chars", str(CHUNK_CHARS)),
             ("chunk_overlap", str(CHUNK_OVERLAP)),
             ("shippable", "1" if shippable else "0")],
        )
        placeholders = ", ".join("?" for _ in DOC_COLUMNS)
        insert_doc = (f"INSERT INTO docs({', '.join(DOC_COLUMNS)}) "
                      f"VALUES ({placeholders})")
        for entry, text in admitted:
            row = tuple(
                str(entry["doc_id"]).strip() if col == "doc_id"
                else int(entry["redistributable"]) if col == "redistributable"
                else entry.get(col)
                for col in DOC_COLUMNS
            )
            conn.execute(insert_doc, row)
            doc_id = row[0]
            for chunk_ix, body in enumerate(chunk_text(text)):
                conn.execute(
                    "INSERT INTO chunks(doc_id, chunk_ix, body) VALUES (?, ?, ?)",
                    (doc_id, chunk_ix, body),
                )
                chunk_count += 1
        conn.commit()
        conn.execute("PRAGMA optimize")
        conn.commit()
        orphan_chunk_count = check_corpus_integrity(conn)
        if orphan_chunk_count:
            raise RegistryRefusal(
                "orphan_chunks",
                f"integrity check found {orphan_chunk_count} chunk(s) without "
                "a docs row",
            )
    finally:
        conn.close()

    os.replace(tmp_path, out_path)
    mode = 0o444 if read_only else 0o644
    out_path.chmod(mode)

    return BuildResult(
        path=out_path,
        corpus_version=corpus_version,
        built_at=built_at,
        builder_sha=b_sha,
        doc_count=len(admitted),
        chunk_count=chunk_count,
        orphan_chunk_count=orphan_chunk_count,
        excluded_non_redistributable=len(excluded_ids),
        excluded_doc_ids=tuple(excluded_ids),
        corpus_sha256=corpus_file_sha256(out_path),
        mode=os.stat(out_path).st_mode & 0o777,
        shippable=shippable,
    )


def chunk_boundaries(corpus_path: str | Path) -> list[tuple[str, int, str]]:
    """Every `(doc_id, chunk_ix, sha256(body))` in a built corpus, sorted.

    The comparison primitive for `Done when` 4: two builds of the same inputs
    must return equal lists. Hashing the body rather than returning it keeps
    the comparison cheap and makes an inequality mean "a boundary moved or the
    text changed" rather than "the diff is 40 MB".
    """
    conn = sqlite3.connect(f"file:{Path(corpus_path)}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT doc_id, chunk_ix, body FROM chunks").fetchall()
    finally:
        conn.close()
    return sorted(
        (str(doc_id), int(chunk_ix), sha256_text(body))
        for doc_id, chunk_ix, body in rows
    )
