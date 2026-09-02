"""B1 — the vetting registry, and the ingest path's argument surface.

Two properties are measured here, both of which are supposed to be mechanical
rather than procedural:

1. **The registry cannot be bypassed** (`Done when` 1). A corpus must not be
   able to come into existence in a state the citation verifier would later
   have to tolerate, so a malformed registry entry is a typed refusal, not a
   warning and not a review step.
2. **Separation of duties** (`Done when` 2). `scripts/corpus_ingest.py` is the
   one program that may reach the network, and it has no argument that accepts
   a health-vault path. That is asserted over the parser's own option strings,
   so a future `--vault` fails a test rather than a code review.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from health_advisor.corpus_build import (
    RegistryRefusal,
    build_corpus,
    sha256_text,
    validate_entry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST_PATH = REPO_ROOT / "scripts" / "corpus_ingest.py"


def _load_ingest():
    spec = importlib.util.spec_from_file_location("corpus_ingest_under_test",
                                                  INGEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest()


GOOD_TEXT = "Endurance training raises VO2max. " * 60


def good(doc_id: str = "d1", text: str = GOOD_TEXT, **over) -> dict:
    row = {
        "doc_id": doc_id,
        "title": "A Vetted Paper",
        "authors": "Author A",
        "year": 2019,
        "doi": "10.0000/x",
        "pmid": "31000000",
        "source_url": "https://example.org/x",
        "retrieved_at": "2026-08-30T09:00:00Z",
        "source_sha256": "a" * 64,
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


def test_a_well_formed_entry_is_admitted():
    validate_entry(good(), GOOD_TEXT, seen_doc_ids=set())


# --------------------------------------------------------------------------- #
# `Done when` 1 — the eight malformed entries, 8/8 refused
# --------------------------------------------------------------------------- #

def _drop(field: str, **over) -> dict:
    row = good(**over)
    row.pop(field)
    return row


#  (label, entry, text, prior_doc_ids, expected code)
MALFORMED: list[tuple[str, dict, str, set, str]] = [
    ("missing approver",
     _drop("approver"), GOOD_TEXT, set(), "missing_approver"),
    ("missing license",
     _drop("license"), GOOD_TEXT, set(), "missing_license"),
    ("missing source_sha256",
     _drop("source_sha256"), GOOD_TEXT, set(), "missing_source_sha256"),
    ("text_sha256 mismatches the indexed text",
     good(text_sha256="b" * 64), GOOD_TEXT, set(), "text_sha256_mismatch"),
    ("duplicate doc_id",
     good("dup"), GOOD_TEXT, {"dup"}, "duplicate_doc_id"),
    ("redistributable absent",
     _drop("redistributable"), GOOD_TEXT, set(), "missing_redistributable"),
    ("empty extracted text",
     good(text_sha256=sha256_text("")), "", set(), "empty_text"),
    ("retrieved_at is not ISO-8601",
     good(retrieved_at="30/08/2026"), GOOD_TEXT, set(), "bad_retrieved_at"),
]


@pytest.mark.parametrize(
    "label, entry, text, seen, code",
    MALFORMED, ids=[m[0] for m in MALFORMED])
def test_malformed_registry_entries_are_refused(label, entry, text, seen, code):
    with pytest.raises(RegistryRefusal) as exc:
        validate_entry(entry, text, seen_doc_ids=set(seen))
    assert exc.value.code == code, exc.value.reason
    assert exc.value.reason.startswith("corpus.registry.")


def test_all_eight_malformed_entries_are_refused_as_a_corpus():
    """The count, not the individual cases: 8 malformed in, 8 refused."""
    refused = 0
    reasons = []
    for _label, entry, text, seen, _code in MALFORMED:
        try:
            validate_entry(entry, text, seen_doc_ids=set(seen))
        except RegistryRefusal as exc:
            refused += 1
            reasons.append(exc.reason)
    assert len(MALFORMED) >= 8
    assert refused == len(MALFORMED) == 8
    assert len(set(reasons)) == 8, "each refusal must be distinguishable"


def test_a_build_refuses_rather_than_writing_a_partial_corpus(tmp_path):
    """One bad entry among good ones leaves no corpus on disk at all."""
    out = tmp_path / "corpus.db"
    entries = [good("ok1"), _drop("approver", doc_id="bad"), good("ok2")]
    texts = [GOOD_TEXT, GOOD_TEXT, GOOD_TEXT]
    with pytest.raises(RegistryRefusal) as exc:
        build_corpus(entries, texts, out, corpus_version=1)
    assert exc.value.code == "missing_approver"
    assert not out.exists()
    assert not (tmp_path / "corpus.db.building").exists()


def test_refusal_names_the_document(tmp_path):
    with pytest.raises(RegistryRefusal) as exc:
        validate_entry(_drop("license", doc_id="pmc-999"), GOOD_TEXT,
                       seen_doc_ids=set())
    assert "pmc-999" in exc.value.reason


def test_the_first_occurrence_wins_and_the_copy_is_refused():
    seen: set[str] = set()
    validate_entry(good("same"), GOOD_TEXT, seen_doc_ids=seen)
    assert "same" in seen
    with pytest.raises(RegistryRefusal) as exc:
        validate_entry(good("same"), GOOD_TEXT, seen_doc_ids=seen)
    assert exc.value.code == "duplicate_doc_id"


@pytest.mark.parametrize("value", [
    "30/08/2026", "August 30 2026", "2026-13-01", "", "yesterday", "20260830T"])
def test_non_iso8601_timestamps_are_refused(value):
    with pytest.raises(RegistryRefusal) as exc:
        validate_entry(good(retrieved_at=value), GOOD_TEXT, seen_doc_ids=set())
    assert exc.value.code in ("bad_retrieved_at", "missing_retrieved_at")


@pytest.mark.parametrize("value", [
    "2026-08-30", "2026-08-30T09:00:00Z", "2026-08-30T09:00:00+00:00",
    "2026-08-30T09:00:00.123456+02:00"])
def test_iso8601_timestamps_are_admitted(value):
    validate_entry(good(retrieved_at=value, approved_at=value), GOOD_TEXT,
                   seen_doc_ids=set())


@pytest.mark.parametrize("value", [2, -1, "1", 1.0, None, True])
def test_redistributable_must_be_int_0_or_1(value):
    with pytest.raises(RegistryRefusal) as exc:
        validate_entry(good(redistributable=value), GOOD_TEXT, seen_doc_ids=set())
    assert exc.value.code in ("bad_redistributable", "missing_redistributable")


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_required_strings_are_refused_like_absent_ones(blank):
    with pytest.raises(RegistryRefusal) as exc:
        validate_entry(good(approver=blank), GOOD_TEXT, seen_doc_ids=set())
    assert exc.value.code == "missing_approver"


def test_whitespace_only_text_is_empty_text():
    with pytest.raises(RegistryRefusal) as exc:
        validate_entry(good(text_sha256=sha256_text("   \n\t  ")), "   \n\t  ",
                       seen_doc_ids=set())
    assert exc.value.code == "empty_text"


# --------------------------------------------------------------------------- #
# `Done when` 2 — separation of duties, measured over the argument surface
# --------------------------------------------------------------------------- #

def _option_strings(parser: argparse.ArgumentParser) -> list[str]:
    out: list[str] = []
    for action in parser._actions:
        out.extend(action.option_strings)
    return out


def test_ingest_exposes_no_vault_path_argument():
    """`Done when` 2. The count that must be 0.

    Checked against BOTH the option strings and the dests, because a
    `--corpus`-spelled flag with `dest="vault"` would pass a check on either
    one alone.
    """
    parser = ingest.build_parser()
    surface = []
    for action in parser._actions:
        surface.extend(s.lstrip("-").replace("-", "_") for s in action.option_strings)
        if action.dest:
            surface.append(str(action.dest))
    offending = [
        name for name in surface
        for token in ingest.FORBIDDEN_ARG_TOKENS
        if token in name.lower()
    ]
    assert offending == [], offending
    assert len(offending) == 0


def test_ingest_rejects_a_vault_flag_at_the_command_line(tmp_path):
    parser = ingest.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--corpus", str(tmp_path / "c.db"),
                           "--registry", str(tmp_path / "r.json"),
                           "--vault", str(tmp_path / "health.db")])


def test_ingest_source_contains_no_vault_import():
    """It must not be able to reach a vault by import either."""
    src = INGEST_PATH.read_text(encoding="utf-8")
    code_lines = [
        ln for ln in src.splitlines()
        if ln.startswith(("import ", "from ")) or " import " in ln]
    for line in code_lines:
        for banned in ("health_advisor.db", "health_advisor.vault",
                       "health_advisor.context", "health_advisor.receiver",
                       "health_advisor.normalize"):
            assert banned not in line, line


def test_ingest_has_no_default_corpus_path_and_no_env_var(monkeypatch):
    """T-003 — no ambient path. `--corpus` is required, with no default and no
    environment fallback."""
    parser = ingest.build_parser()
    corpus_action = next(a for a in parser._actions if "--corpus" in a.option_strings)
    assert corpus_action.required is True
    assert corpus_action.default is None
    monkeypatch.setenv("HA_CORPUS", "/tmp/should-never-be-read.db")
    with pytest.raises(SystemExit):
        parser.parse_args(["--registry", "/tmp/r.json"])
    src = INGEST_PATH.read_text(encoding="utf-8")
    assert "getenv(" not in src and "os.environ" not in src


def test_the_documented_argument_surface_is_the_whole_surface():
    parser = ingest.build_parser()
    assert sorted(_option_strings(parser)) == sorted([
        "-h", "--help",
        "--corpus", "--registry", "--corpus-version", "--previous-corpus",
        "--shippable", "--writable", "--dry-run",
    ])


# --------------------------------------------------------------------------- #
# The ingest entry point end to end
# --------------------------------------------------------------------------- #

def write_registry(tmp_path: Path, rows: list[tuple[dict, str]]) -> Path:
    texts_dir = tmp_path / "texts"
    texts_dir.mkdir(exist_ok=True)
    entries = []
    for row, text in rows:
        name = f"{row['doc_id']}.txt"
        (texts_dir / name).write_text(text, encoding="utf-8")
        entry = dict(row)
        entry["text_path"] = f"texts/{name}"
        entries.append(entry)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return registry


def test_ingest_builds_a_corpus(tmp_path, capsys):
    registry = write_registry(tmp_path, [(good("d1"), GOOD_TEXT),
                                         (good("d2", text=GOOD_TEXT + "extra"), GOOD_TEXT + "extra")])
    out = tmp_path / "corpus.db"
    code = ingest.main(["--corpus", str(out), "--registry", str(registry)])
    assert code == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert "corpus_version : 1" in printed
    assert "mode           : 0o444" in printed


def test_ingest_refusal_is_a_typed_line_not_a_traceback(tmp_path, capsys):
    registry = write_registry(tmp_path, [(_drop("approver", doc_id="bad"), GOOD_TEXT)])
    out = tmp_path / "corpus.db"
    code = ingest.main(["--corpus", str(out), "--registry", str(registry)])
    assert code == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("corpus.registry.missing_approver")
    assert "Traceback" not in captured.err
    assert captured.err.count("\n") == 1
    assert not out.exists()


def test_ingest_missing_registry_is_a_typed_refusal(tmp_path, capsys):
    code = ingest.main(["--corpus", str(tmp_path / "c.db"),
                        "--registry", str(tmp_path / "nope.json")])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("corpus.registry.registry_not_found")
    assert "Traceback" not in err


def test_ingest_bad_json_is_a_typed_refusal(tmp_path, capsys):
    registry = tmp_path / "registry.json"
    registry.write_text("{not json", encoding="utf-8")
    code = ingest.main(["--corpus", str(tmp_path / "c.db"),
                        "--registry", str(registry)])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("corpus.registry.registry_not_json")


def test_ingest_dry_run_writes_nothing(tmp_path, capsys):
    registry = write_registry(tmp_path, [(good("d1"), GOOD_TEXT)])
    out = tmp_path / "corpus.db"
    code = ingest.main(["--corpus", str(out), "--registry", str(registry),
                        "--dry-run"])
    assert code == 0
    assert not out.exists()
    assert "dry-run OK" in capsys.readouterr().out


def test_ingest_shippable_reports_the_exclusion(tmp_path, capsys):
    registry = write_registry(tmp_path, [
        (good("open"), GOOD_TEXT),
        (good("closed", text=GOOD_TEXT + "z", redistributable=0,
              license="proprietary"), GOOD_TEXT + "z"),
    ])
    out = tmp_path / "corpus.db"
    code = ingest.main(["--corpus", str(out), "--registry", str(registry),
                        "--shippable"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "excluded (non-redistributable): 1 ['closed']" in printed
    assert "docs           : 1" in printed


def test_ingest_refuses_a_non_monotonic_version(tmp_path, capsys):
    registry = write_registry(tmp_path, [(good("d1"), GOOD_TEXT)])
    first = tmp_path / "v1.db"
    assert ingest.main(["--corpus", str(first), "--registry", str(registry),
                        "--corpus-version", "5"]) == 0
    capsys.readouterr()
    code = ingest.main(["--corpus", str(tmp_path / "v2.db"),
                        "--registry", str(registry),
                        "--previous-corpus", str(first),
                        "--corpus-version", "5"])
    assert code == 2
    assert capsys.readouterr().err.startswith(
        "corpus.registry.corpus_version_not_monotonic")


def test_ingest_missing_text_path_is_a_typed_refusal(tmp_path, capsys):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([good("d1")]), encoding="utf-8")
    code = ingest.main(["--corpus", str(tmp_path / "c.db"),
                        "--registry", str(registry)])
    assert code == 2
    assert capsys.readouterr().err.startswith("corpus.registry.missing_text_path")


def test_ingest_runs_as_a_subprocess_without_a_traceback(tmp_path):
    """The real command line, not an in-process call."""
    registry = write_registry(tmp_path, [(_drop("license", doc_id="bad"), GOOD_TEXT)])
    proc = subprocess.run(
        [sys.executable, str(INGEST_PATH),
         "--corpus", str(tmp_path / "c.db"), "--registry", str(registry)],
        capture_output=True, text=True)
    assert proc.returncode == 2
    assert proc.stderr.strip().startswith("corpus.registry.missing_license")
    assert "Traceback" not in proc.stderr
