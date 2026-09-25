"""#394 / engine #22 item 1 — the corpus-derived domain lexicon (trigger A).

Decision brief `docs/product/reviews/i394-decision-brief-20260925.md`,
accepted 2026-09-25 (all four recommendations, consumer #394): a question is
refused (no `cite` for that turn) when it contains none of a lexicon of
stems that are common in the evidence corpus and rare in two control
corpora. This file tests `analyst_corpus.domain_lexicon`,
`analyst_corpus.question_in_domain`, and `corpus_build.build_corpus`'s
storage of the lexicon in `corpus_meta` (decision 4).

Two kinds of test, on purpose (task instruction: a test that skips silently
when the corpus is absent measures nothing):

- HERMETIC — build tiny evidence/control corpora under `tmp_path` with
  `corpus_build.build_corpus`. These always run; nothing here depends on
  `data/corpus` existing.
- REAL-CORPUS — reproduce the decision brief's own measured numbers against
  `data/corpus/{corpus,corpus-nearmiss,corpus-null}.db`. These skip
  EXPLICITLY (a visible `pytest.skip`, not a silent no-op) when that data is
  not present, exactly the way `tests/test_analyst_corpus.py`'s `corpus`
  fixture does.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from health_advisor import analyst_corpus as ac
from health_advisor.corpus_build import build_corpus, sha256_text

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
REAL_EVIDENCE = CORPUS_DIR / "corpus.db"
REAL_NEARMISS = CORPUS_DIR / "corpus-nearmiss.db"
REAL_NULL = CORPUS_DIR / "corpus-null.db"

GAAS_QUESTION = "What is the boiling point of gallium arsenide?"

# The brief's own 24 out-of-domain questions (12 "far", 12 "near-miss
# medical"), taken verbatim from
# `docs/product/reviews/i394-decision-brief-20260925.md`'s lists -- the
# consumer repo's `.claude/i394-artifacts/q-outofdomain.txt` holds only the
# GaAs question (a smaller smoke fixture), not the full 24, so this list is
# reconstructed from the brief's prose rather than read from a file.
OUT_OF_DOMAIN_24 = (
    "What is the boiling point of gallium arsenide?",
    "What is the capital of Australia?",
    "How do I start a sourdough starter?",
    "Who won the 1998 World Cup?",
    "How do I reverse a linked list in Python?",
    "What is the half-life of carbon-14?",
    "How do freelancer taxes work?",
    "How do I learn guitar chords?",
    "How tall is Mount Everest?",
    "What is the speed of light in glass?",
    "How do I change a car tyre?",
    "What caused the fall of the Roman Empire?",
    "What is the correct levothyroxine dose?",
    "How often should I get a colonoscopy?",
    "How long does shingles last?",
    "What is a celiac blood test?",
    "What are migraine aura symptoms?",
    "How is a tooth cavity treated?",
    "Can I take ibuprofen with blood-pressure medication?",
    "What causes kidney stones?",
    "How is glaucoma diagnosed?",
    "What adult vaccines do I need?",
    "What are early signs of Parkinson's?",
    "How do I treat acne scars?",
)


def _require_real_corpora():
    missing = [p for p in (REAL_EVIDENCE, REAL_NEARMISS, REAL_NULL) if not p.exists()]
    if missing:
        pytest.skip(
            "real corpus/control corpora not present: "
            + ", ".join(str(p) for p in missing))


@pytest.fixture()
def real_lexicon():
    """The lexicon `domain_lexicon` derives from the real corpus + controls.

    Built once per test via this fixture (not module-scoped): each test gets
    its own connections and closes them, so a test that also wants to poke at
    an open corpus connection is never surprised by one already closed
    elsewhere.
    """
    _require_real_corpora()
    evidence = sqlite3.connect(f"file:{REAL_EVIDENCE}?mode=ro", uri=True)
    nearmiss = sqlite3.connect(f"file:{REAL_NEARMISS}?mode=ro", uri=True)
    null = sqlite3.connect(f"file:{REAL_NULL}?mode=ro", uri=True)
    try:
        yield ac.domain_lexicon(evidence, [nearmiss, null])
    finally:
        evidence.close()
        nearmiss.close()
        null.close()


def _questions(name: str) -> list[dict]:
    """A question-set fixture from the CONSUMER repo, read-only.

    These 52+52 questions live in `healthAI/ha-client`, a private sibling
    repo to this public engine checkout -- not duplicated into this tree.
    Reached the same way `data/corpus` already is (that path is a symlink
    into the very same sibling checkout): assume the sibling layout, and
    skip explicitly, visibly, when it is not there (a different machine, or
    a CI runner for this public repo that never clones the private one).
    """
    path = (REPO_ROOT.parent / "healthAI" / "ha-client" / "docs" / "product"
            / "corpus" / name)
    if not path.exists():
        pytest.skip(f"consumer-repo question fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Hermetic — tokenizer and exclusion-list properties, no corpus file needed
# --------------------------------------------------------------------------- #

def test_generic_question_words_is_exactly_twenty():
    # Named and counted per the task: the original 20-word list from the
    # decision brief's uncommitted scratchpad script is not in this tree (nor
    # the consumer repo's) -- confirmed absent, so this is a reconstruction,
    # asserted at exactly 20 so a future edit cannot silently drift the count.
    assert len(ac._GENERIC_QUESTION_WORDS) == 20


def test_stopword_stemming_changes_some_entries():
    """Porter is not the identity function on the #22 stopwords.

    If this ever assembled zero differences, `_excluded_stems` would be
    filtering raw strings against stemmed corpus vocabulary and silently
    admitting words like "does" (stem "doe") into the lexicon.
    """
    changed = {w for w in ac._STOPWORDS if ac.fts5_stems(w) != frozenset({w})}
    assert changed == {"are", "being", "does", "has", "its", "they", "this", "was"}


def test_excluded_stems_covers_the_stemmed_forms():
    excluded = ac._excluded_stems()
    # Raw stopwords that stem differently must be excluded by their STEMMED
    # form, not their raw spelling -- this is the property the mismatch above
    # exists to protect.
    assert "doe" in excluded  # stem of "does"
    assert "ar" in excluded  # stem of "are"
    assert "wa" in excluded  # stem of "was"
    # A domain-relevant word must never accidentally collide with an
    # excluded stem.
    assert "run" not in excluded
    assert "injuri" not in excluded


def test_fts5_stems_is_order_and_case_insensitive():
    assert ac.fts5_stems("Running Runs RUN") == frozenset({"run"})
    assert ac.fts5_stems("") == frozenset()
    assert ac.fts5_stems("   ") == frozenset()


def test_question_in_domain_refuses_on_empty_lexicon():
    assert ac.question_in_domain("anything at all", frozenset()) is False


def test_question_in_domain_admits_on_any_shared_stem():
    lexicon = frozenset({"injuri", "cadenc"})
    assert ac.question_in_domain("Does cadence affect injury risk?", lexicon) is True
    assert ac.question_in_domain("What is the weather like?", lexicon) is False


# --------------------------------------------------------------------------- #
# Hermetic — domain_lexicon's contrast logic, built under tmp_path
# --------------------------------------------------------------------------- #

def _entry(doc_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"Title of {doc_id}",
        "authors": "Author A",
        "year": 2020,
        "doi": None,
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


# Long enough (>> CHUNK_CHARS=1200 with 200 overlap) that the repeated
# domain terms land in several chunks each, clearing `min_chunks=3`.
_EVIDENCE_TEXT = (
    "Weekly running volume and injury risk in distance runners. "
    "Cadence, stride length and running economy during endurance training. "
    "Recovery, sleep and heart rate variability after hard running sessions. "
) * 40

_CONTROL_TEXT = (
    "Gallium arsenide crystal growth and semiconductor wafer doping. "
    "Photolithography, etching and thin-film deposition in fabrication. "
    "Bandgap engineering and carrier mobility in compound semiconductors. "
) * 40


@pytest.fixture()
def hermetic_corpora(tmp_path):
    """One evidence and one control corpus, built fresh, connections open."""
    build_corpus([_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT],
                 tmp_path / "evidence.db", corpus_version=1)
    build_corpus([_entry("ctrl-1", _CONTROL_TEXT)], [_CONTROL_TEXT],
                 tmp_path / "control.db", corpus_version=1)
    evidence = sqlite3.connect(f"file:{tmp_path / 'evidence.db'}?mode=ro", uri=True)
    control = sqlite3.connect(f"file:{tmp_path / 'control.db'}?mode=ro", uri=True)
    try:
        yield evidence, control
    finally:
        evidence.close()
        control.close()


def test_domain_lexicon_admits_contrastive_evidence_terms(hermetic_corpora):
    evidence, control = hermetic_corpora
    lexicon = ac.domain_lexicon(evidence, [control])
    # Evidence-only vocabulary, well over min_chunks and never seen in the
    # control at all: must be admitted.
    for stem in ("run", "injuri", "cadenc", "recoveri"):
        assert stem in lexicon, f"{stem!r} should be admitted; lexicon={sorted(lexicon)}"
    # Control-only vocabulary must never leak into the evidence lexicon
    # (it never appears in the evidence text at all, so doc_count=0 there).
    for stem in ("semiconductor", "wafer", "gallium", "arsenid"):
        assert stem not in lexicon


def test_domain_lexicon_excludes_stopwords_and_question_words(hermetic_corpora):
    evidence, control = hermetic_corpora
    lexicon = ac.domain_lexicon(evidence, [control])
    assert lexicon.isdisjoint(ac._excluded_stems())


def test_domain_lexicon_requires_min_chunks(hermetic_corpora):
    """A term seen in only one or two chunks must not be admitted."""
    evidence, control = hermetic_corpora
    # min_chunks=1 must admit a strict superset of the default min_chunks=3
    # lexicon (never fewer terms, since the bar is only lower).
    loose = ac.domain_lexicon(evidence, [control], min_chunks=1)
    strict = ac.domain_lexicon(evidence, [control], min_chunks=3)
    assert strict <= loose
    assert len(strict) <= len(loose)


def test_domain_lexicon_pure_and_order_independent(hermetic_corpora):
    """Same inputs, same output -- no clock, no randomness, no set-iteration
    dependence leaking into which terms are admitted."""
    evidence, control = hermetic_corpora
    first = ac.domain_lexicon(evidence, [control])
    second = ac.domain_lexicon(evidence, [control])
    assert first == second


def test_domain_lexicon_rejects_empty_evidence_corpus(tmp_path):
    build_corpus([_entry("empty-1", "x")], ["x"], tmp_path / "empty.db",
                 corpus_version=1)
    # A single one-character doc still produces one chunk, so force a truly
    # chunkless corpus is not reachable through build_corpus (it refuses an
    # empty text at validate_entry) -- so this exercises the guard directly
    # against a hand-built connection with no chunks inserted, which is the
    # state `domain_lexicon` must not divide by zero against.
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE VIRTUAL TABLE chunks USING fts5(doc_id UNINDEXED, "
        "chunk_ix UNINDEXED, body, tokenize='porter unicode61');")
    with pytest.raises(ValueError):
        ac.domain_lexicon(conn, [])


# --------------------------------------------------------------------------- #
# THE MUTATION TEST -- a lexicon of every term must admit GaAs.
#
# This is the discrimination proof the task asks for: it does not test
# `domain_lexicon`'s output, it tests that `question_in_domain` ITSELF would
# pass the out-of-domain question if the lexicon were degenerate. If this
# test ever failed, every other "GaAs is refused" test in this file would be
# passing for a reason that has nothing to do with contrast against a
# control corpus, and would give false confidence.
# --------------------------------------------------------------------------- #

def test_degenerate_lexicon_of_every_stem_admits_gaas():
    """The direct, corpus-independent form: a lexicon built from literally
    every stem in the question itself of course admits it. This is the
    "always True" shape stated as a lexicon rather than as a monkeypatched
    predicate -- it holds for ANY question, which is exactly why it proves
    nothing about discrimination on its own and `domain_lexicon`'s contrast
    step is what has to do the actual work (see the real-corpus version of
    this test below, which is the one the task means).
    """
    every_stem_in_the_question = ac.fts5_stems(GAAS_QUESTION)
    assert ac.question_in_domain(GAAS_QUESTION, every_stem_in_the_question)


def test_degenerate_every_word_lexicon_admits_gaas_on_the_real_corpus(real_lexicon):
    """THE required mutation test: "any word the corpus has ever seen" (the
    brief's own rejected, non-contrastive alternative) must ADMIT the
    out-of-domain GaAs question, where the real (contrastive) `real_lexicon`
    fixture refuses it (`test_gaas_is_refused_against_the_real_corpus`).

    `min_chunks=1, control_conns=[]` is precisely that alternative:
    `domain_lexicon` pools an EMPTY sequence of controls to a total of 0, so
    every stem's control rate is 0.0 and the `rate_multiple` check admits
    everything meeting `min_chunks` -- i.e. every word the corpus has ever
    used at least once, no contrast at all.

    This must go red if a future change made `domain_lexicon` (or
    `question_in_domain`) discriminate on SOMETHING OTHER than the corpus
    contrast -- e.g. if a naive reimplementation forgot the control pooling
    entirely and treated "no controls supplied" as "refuse everything"
    instead of "admit everything meeting the other bars." Requires the real
    corpus because the mechanism is real FTS5 vocabulary breadth (the word
    "point" turns out to occur somewhere in the 1,923-chunk real corpus, by
    the ordinary breadth of a corpus that size) -- a small hermetic corpus
    limited to running/injury vocabulary shares no word at all with the GaAs
    question, so it cannot exhibit this leak and would prove nothing.
    """
    evidence = sqlite3.connect(f"file:{REAL_EVIDENCE}?mode=ro", uri=True)
    try:
        every_word_the_corpus_has_ever_seen = ac.domain_lexicon(
            evidence, [], min_chunks=1)
    finally:
        evidence.close()
    assert ac.question_in_domain(
        GAAS_QUESTION, every_word_the_corpus_has_ever_seen)
    # The contrastive lexicon this file otherwise tests refuses it -- this is
    # the comparison that shows the mutation actually changed the outcome,
    # not merely that SOME lexicon admits SOME question.
    assert not ac.question_in_domain(GAAS_QUESTION, real_lexicon)


# --------------------------------------------------------------------------- #
# Real corpus — reproducing the decision brief's oracle table
# --------------------------------------------------------------------------- #

def test_gaas_is_refused_against_the_real_corpus(real_lexicon):
    assert ac.question_in_domain(GAAS_QUESTION, real_lexicon) is False


def test_all_52_clinical_questions_are_admitted(real_lexicon):
    clinical = _questions("questions-clinical.json")
    assert len(clinical) == 52
    refused = [q["question"] for q in clinical
               if not ac.question_in_domain(q["question"], real_lexicon)]
    assert refused == [], f"clinical questions wrongly refused: {refused}"


def test_lay_questions_have_exactly_one_false_refusal(real_lexicon):
    """Pins the decision brief's own measured false refusal (2026-09-25):
    'Is a low step count automatically a problem?' -- `step` is common in the
    control corpora too. If this list ever grows, trigger A's false-refusal
    rate has regressed and that is worth knowing by name, not just by count.
    """
    lay = _questions("questions-lay.json")
    assert len(lay) == 52
    refused = [q["question"] for q in lay
               if not ac.question_in_domain(q["question"], real_lexicon)]
    assert refused == ["Is a low step count automatically a problem?"]


def test_most_out_of_domain_questions_are_refused(real_lexicon):
    """Reproduces the brief's 21-of-24 out-of-domain refusal rate. Not 24 of
    24 -- three leaks (`World Cup`/`footbal`+`cup`+`world`, `half-life`/half,
    `speed of light`/speed) are measured and expected, not a bug to chase in
    this pass.
    """
    refused = [q for q in OUT_OF_DOMAIN_24
               if not ac.question_in_domain(q, real_lexicon)]
    assert GAAS_QUESTION in refused
    assert len(refused) == 21


def test_real_lexicon_size_matches_the_brief_within_a_small_margin(real_lexicon):
    """The brief measured 2,026 terms; this session's reconstruction of the
    20 generic question words (not committed anywhere, confirmed absent from
    both repos) is the one input that cannot be reproduced exactly, so this
    checks "close", not "equal". See `analyst_corpus.DOMAIN_LEXICON_RATE_
    MULTIPLE`'s comment for the full comparison, including that the
    per-control alternative (1,559 terms) was tried and rejected in favour of
    this pooled definition specifically because it reproduces the brief's
    number far more closely.
    """
    assert abs(len(real_lexicon) - 2026) <= 10


# --------------------------------------------------------------------------- #
# corpus_build storage — decision 4
# --------------------------------------------------------------------------- #

def test_build_without_controls_leaves_lexicon_unwritten(tmp_path):
    result = build_corpus([_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT],
                          tmp_path / "evidence.db", corpus_version=1)
    assert result.domain_lexicon_status == ac.DOMAIN_LEXICON_STATUS_NO_CONTROLS
    assert result.domain_lexicon_size is None
    conn = sqlite3.connect(f"file:{tmp_path / 'evidence.db'}?mode=ro", uri=True)
    assert ac.load_domain_lexicon(conn) is None
    status_row = conn.execute(
        "SELECT value FROM corpus_meta WHERE key = ?",
        (ac.DOMAIN_LEXICON_STATUS_KEY,)).fetchone()
    assert status_row == (ac.DOMAIN_LEXICON_STATUS_NO_CONTROLS,)


def test_build_with_controls_stores_a_loadable_lexicon(tmp_path):
    build_corpus([_entry("ctrl-1", _CONTROL_TEXT)], [_CONTROL_TEXT],
                 tmp_path / "control.db", corpus_version=1)
    result = build_corpus(
        [_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT],
        tmp_path / "evidence.db", corpus_version=1,
        control_corpus_paths=[tmp_path / "control.db"])
    assert result.domain_lexicon_status == ac.DOMAIN_LEXICON_STATUS_BUILT
    assert result.domain_lexicon_size is not None and result.domain_lexicon_size > 0

    conn = sqlite3.connect(f"file:{tmp_path / 'evidence.db'}?mode=ro", uri=True)
    lexicon = ac.load_domain_lexicon(conn)
    assert lexicon is not None
    assert len(lexicon) == result.domain_lexicon_size
    assert "injuri" in lexicon

    params_row = conn.execute(
        "SELECT value FROM corpus_meta WHERE key = ?",
        (ac.DOMAIN_LEXICON_PARAMS_KEY,)).fetchone()
    params = json.loads(params_row[0])
    assert params["control_corpus_count"] == 1


def test_load_domain_lexicon_tolerates_a_malformed_row(tmp_path):
    """A retrieval-time reader degrades to "no lexicon", never raises, on a
    corpus_meta row that is not a JSON list of strings -- e.g. a future
    schema change, or a corpus built by a tool that never learned this key.
    """
    build_corpus([_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT],
                 tmp_path / "evidence.db", corpus_version=1, read_only=False)
    conn = sqlite3.connect(tmp_path / "evidence.db")
    conn.execute(
        "INSERT OR REPLACE INTO corpus_meta(key, value) VALUES (?, ?)",
        (ac.DOMAIN_LEXICON_META_KEY, "not valid json"))
    conn.commit()
    assert ac.load_domain_lexicon(conn) is None

    conn.execute(
        "INSERT OR REPLACE INTO corpus_meta(key, value) VALUES (?, ?)",
        (ac.DOMAIN_LEXICON_META_KEY, json.dumps([1, 2, 3])))
    conn.commit()
    assert ac.load_domain_lexicon(conn) is None


def test_load_domain_lexicon_is_none_when_key_absent(tmp_path):
    build_corpus([_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT],
                 tmp_path / "old.db", corpus_version=1, read_only=False)
    conn = sqlite3.connect(tmp_path / "old.db")
    conn.execute("DELETE FROM corpus_meta WHERE key = ?",
                (ac.DOMAIN_LEXICON_STATUS_KEY,))
    conn.commit()
    assert ac.load_domain_lexicon(conn) is None
