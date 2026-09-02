#!/usr/bin/env python3
"""Build a vetted evidence corpus from a registry file. Run by a human.

THIS PROGRAM HAS NO WAY TO OPEN A HEALTH VAULT, AND THAT IS THE DESIGN.

Design §4.1: the separation of duties is enforced by argument surface, not by
discipline. Ingest is the one path that may reach the network (to acquire
sources); the analyst path is the one that may open the vault. Neither property
depends on anyone remembering it, because ingest defines no flag that takes a
vault path — no ``--vault``, no ``--db``, no ``--health-db``, nothing that
could be one. `tests/test_corpus_registry.py` measures that over the parser's
own option strings, so the property fails a test rather than a code review.

T-003 — no ambient path. ``--corpus`` is REQUIRED and has no default and no
environment variable. A ``HA_CORPUS`` fallback is precisely the defect that
rule exists to prevent: one process serves more than one corpus, and a
module-level default is how a session ends up writing the wrong one.

Registry format — a JSON list (or ``{"docs": [...]}``), one object per
document, with the `docs` columns of design §1.3 plus a ``text_path``:

    [{"doc_id": "pmc-123", "title": "...", "authors": "...", "year": 2019,
      "doi": null, "pmid": "31000000",
      "source_url": "https://...", "retrieved_at": "2026-08-30T09:00:00Z",
      "source_sha256": "<of the fetched artifact>",
      "text_sha256":   "<of the extracted text>",
      "license": "CC-BY-4.0", "license_url": "https://...",
      "redistributable": 1,
      "approver": "your-name", "approved_at": "2026-08-30",
      "notes": null,
      "text_path": "texts/pmc-123.txt"}]

``text_path`` resolves relative to the registry file's own directory.
Extraction is deliberately restricted to plain text and JATS XML: no PDF is
parsed anywhere in this repo (design §1.5).

Exit codes: 0 built, 2 typed refusal, 1 usage error. A refusal prints one
typed line on stderr and never a traceback — a traceback is a data channel in
this codebase and is treated as a defect.

    ./.venv/bin/python scripts/corpus_ingest.py \\
        --corpus data/corpus.db --registry corpus/registry.json --shippable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_advisor.corpus_build import (  # noqa: E402
    RegistryRefusal,
    build_corpus,
    extract_text,
    next_corpus_version,
    read_corpus_version,
)

# Any option string or dest matching one of these would hand this program a
# path into the user's health data. The test asserts the parser defines none of
# them; the list is here so the intent is legible at the definition site too.
FORBIDDEN_ARG_TOKENS: tuple[str, ...] = (
    "vault", "health", "db", "snapshot", "export", "hk", "metrics", "workout",
)


def build_parser() -> argparse.ArgumentParser:
    """The whole argument surface of the ingest path.

    Every option here is about the corpus or the registry. Adding one that
    takes a vault path breaks `test_ingest_exposes_no_vault_path`, which is the
    point.
    """
    parser = argparse.ArgumentParser(
        prog="corpus_ingest.py",
        description="Build a vetted evidence corpus from a registry file.",
    )
    parser.add_argument(
        "--corpus", required=True,
        help="output corpus path (REQUIRED; no default, no environment "
             "variable — T-003, there is no ambient path)")
    parser.add_argument(
        "--registry", required=True,
        help="JSON registry of vetted documents (the docs rows plus text_path)")
    parser.add_argument(
        "--corpus-version", type=int, default=None,
        help="corpus_version to stamp; defaults to one past --previous-corpus, "
             "or 1 for a first build")
    parser.add_argument(
        "--previous-corpus", default=None,
        help="an existing corpus whose corpus_version the new build succeeds")
    parser.add_argument(
        "--shippable", action="store_true",
        help="admit only redistributable = 1, and report the count excluded")
    parser.add_argument(
        "--writable", action="store_true",
        help="leave the built corpus mode 0644 instead of 0444 (for a build "
             "pipeline that will chmod it itself; the default is read-only)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the registry and report, writing nothing")
    return parser


def load_registry(registry_path: Path) -> list[dict]:
    """Parse the registry JSON into a list of entries, in file order.

    A list, never a mapping keyed by doc_id: build order must not depend on
    dict iteration, and the registry's own order is the order a human wrote.
    """
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RegistryRefusal(
            "registry_not_found", f"no registry file at {registry_path}")
    except json.JSONDecodeError as exc:
        raise RegistryRefusal(
            "registry_not_json",
            f"{registry_path.name} is not valid JSON: {exc.msg} at line {exc.lineno}")
    if isinstance(raw, dict):
        raw = raw.get("docs")
    if not isinstance(raw, list):
        raise RegistryRefusal(
            "registry_not_a_list",
            "registry must be a JSON list of document objects, or an object "
            "with a 'docs' list")
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise RegistryRefusal(
                "registry_entry_not_an_object",
                f"registry item {i} is {type(entry).__name__}, not an object")
    return raw


def load_texts(entries: list[dict], base_dir: Path) -> list[str]:
    """Extract the plain text for each entry, in the same order.

    Missing ``text_path`` is a refusal rather than an empty document, because
    an empty document would be refused later for the wrong reason and the
    operator would go looking at the file instead of at the registry.
    """
    texts: list[str] = []
    for entry in entries:
        doc_id = str(entry.get("doc_id", "<no doc_id>"))
        text_path = entry.get("text_path")
        if not isinstance(text_path, str) or not text_path.strip():
            raise RegistryRefusal(
                "missing_text_path",
                "registry entry has no text_path; extracted text is what gets "
                "indexed and there is nothing to index without it",
                doc_id=doc_id)
        resolved = (base_dir / text_path).resolve()
        if not resolved.is_file():
            raise RegistryRefusal(
                "text_file_not_found",
                f"text_path {text_path!r} does not resolve to a file",
                doc_id=doc_id)
        try:
            texts.append(extract_text(resolved))
        except UnicodeDecodeError:
            raise RegistryRefusal(
                "text_not_utf8",
                f"{text_path!r} is not valid UTF-8; extraction happens before "
                f"ingest and must produce text",
                doc_id=doc_id)
        except RegistryRefusal as exc:
            raise RegistryRefusal(exc.code, str(exc).split(": ", 1)[-1], doc_id=doc_id)
    return texts


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry_path = Path(args.registry).expanduser()
    corpus_path = Path(args.corpus).expanduser()

    entries = load_registry(registry_path)
    texts = load_texts(entries, registry_path.resolve().parent)

    version = args.corpus_version
    if version is None:
        version = next_corpus_version(args.previous_corpus)
    elif args.previous_corpus is not None:
        previous = read_corpus_version(args.previous_corpus)
        if previous is not None and version <= previous:
            raise RegistryRefusal(
                "corpus_version_not_monotonic",
                f"--corpus-version {version} does not exceed the previous "
                f"corpus's {previous}; (doc_id, chunk_ix) is only stable within "
                f"a version, so the number must go up")

    if args.dry_run:
        from health_advisor.corpus_build import validate_entry
        seen: set[str] = set()
        for entry, text in zip(entries, texts):
            validate_entry(entry, text, seen_doc_ids=seen)
        print(f"dry-run OK: {len(entries)} entries vetted, corpus_version would "
              f"be {version}, nothing written")
        return 0

    result = build_corpus(
        entries, texts, corpus_path,
        corpus_version=version,
        shippable=args.shippable,
        read_only=not args.writable,
    )
    print(f"built {result.path}")
    print(f"  corpus_version : {result.corpus_version}")
    print(f"  built_at       : {result.built_at}")
    print(f"  builder_sha    : {result.builder_sha}")
    print(f"  docs           : {result.doc_count}")
    print(f"  chunks         : {result.chunk_count}")
    print(f"  excluded (non-redistributable): "
          f"{result.excluded_non_redistributable}"
          + (f" {list(result.excluded_doc_ids)}"
             if result.excluded_doc_ids else ""))
    print(f"  mode           : {oct(result.mode)}")
    print(f"  corpus_sha256  : {result.corpus_sha256}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Turn every expected failure into one typed line, never a traceback."""
    try:
        return run(argv)
    except RegistryRefusal as refusal:
        print(refusal.reason, file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"corpus.ingest.io_error: {exc.strerror or exc} "
              f"({getattr(exc, 'filename', None)})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
