"""#394 / engine #22 item 1 — the corpus-derived domain lexicon (trigger A).

Decision brief `docs/product/reviews/i394-decision-brief-20260925.md`
(consumer repo, not present in this engine checkout), accepted 2026-09-25
(all four recommendations, consumer #394): a question is refused (no `cite`
for that turn) when it contains none of a lexicon of stems that are common
in the evidence corpus and rare in two control corpora. This file tests
`analyst_corpus.domain_lexicon`, `analyst_corpus.question_in_domain`, and
`corpus_build.build_corpus`'s storage of the lexicon in `corpus_meta`
(decision 4).

HERMETIC ONLY, DELIBERATELY. Every test here builds its own tiny evidence
and/or control corpus under `tmp_path` with `corpus_build.build_corpus` and
touches nothing outside this checkout. Two things that do NOT belong here,
on purpose:

1. `data/corpus/{corpus,corpus-nearmiss,corpus-null}.db` -- gitignored, real,
   and specific to whichever machine has them checked out.
2. The consumer repo's question fixtures (`questions-lay.json`,
   `questions-clinical.json`) and the decision brief's own out-of-domain
   question list -- these live in the consuming application's repository,
   not here. A public engine test that hard-codes a sibling checkout's path
   only resolves on one machine and publishes another repo's layout.

The real-data reproduction of the decision brief's oracle table (lexicon
size ~2,024, lay 1/52 refused, clinical 0/52, out-of-domain 21/24 including
gallium arsenide) lives in the consuming application's own tests instead -- that is where the private
question fixtures and the gitignored corpora already are, and it is the
right side of the boundary for a test that only makes sense with both.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from health_advisor import analyst_corpus as ac
from health_advisor.corpus_build import build_corpus, sha256_text

GAAS_QUESTION = "What is the boiling point of gallium arsenide?"


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


def test_fts5_stems_works_from_a_second_thread():
    """`fts5_stems` (and therefore `question_in_domain`) is called from the
    receiver's request-thread pool at retrieval time, never only from the
    thread that first imported this module. A tokenizer connection created
    on the main thread with `sqlite3.connect`'s default
    `check_same_thread=True` raises `ProgrammingError` the first time any
    OTHER thread touches it -- the same CLASS of defect (a cross-thread
    SQLite handle) that got #394's first `cite` landing reverted, one layer
    down in this module instead of in the corpus connection.
    """

    result: dict[str, object] = {}
    error: list[BaseException] = []

    def worker():
        try:
            result["stems"] = ac.fts5_stems("Running Runs RUN")
            result["in_domain"] = ac.question_in_domain(
                "Does running increase injury risk?", frozenset({"injuri"}))
        except BaseException as exc:  # noqa: BLE001 -- capture to assert on
            error.append(exc)

    main_thread_stems = ac.fts5_stems("Running Runs RUN")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive(), "worker thread did not finish"
    assert error == [], f"worker thread raised: {error}"
    assert result["stems"] == main_thread_stems == frozenset({"run"})
    assert result["in_domain"] is True


def test_fts5_stems_is_consistent_across_concurrent_threads():
    """Several threads calling in at once must each get the tokenization of
    THEIR OWN text, never another thread's -- the property a shared,
    cleared-and-reinserted table would put at risk even with
    `check_same_thread=False`, because "clear the table, insert my row, read
    it back" is three separate statements with no lock between them.
    """

    words = ["running", "training", "recovery", "cadence", "injury",
             "sleeping", "pacing", "fueling", "stretching", "climbing"]
    expected = [ac.fts5_stems(w) for w in words]  # sequential ground truth

    results: list = [None] * len(words)
    errors: list[BaseException] = []

    def worker(index: int, word: str) -> None:
        try:
            got = None
            for _ in range(20):  # repeat to make a race likelier to surface
                got = ac.fts5_stems(word)
                if got != expected[index]:
                    errors.append(AssertionError(
                        f"thread for {word!r} got {got!r}, expected "
                        f"{expected[index]!r}"))
                    return
            results[index] = got
        except BaseException as exc:  # noqa: BLE001 -- capture to assert on
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i, w))
               for i, w in enumerate(words)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not any(t.is_alive() for t in threads)
    assert errors == [], f"{len(errors)} thread error(s): {errors[:3]}"
    assert results == expected


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
# The 2x -> 3x threshold raise (consumer #522, the owner's decision 2026-09-26)
# --------------------------------------------------------------------------- #

def _hand_built_chunks_conn(rows: list[str]) -> sqlite3.Connection:
    """An in-memory connection with a `chunks` FTS5 table, one chunk per
    string in `rows` -- built directly (not through `corpus_build.build_corpus`,
    whose chunker splits on character count, not on a caller-chosen number of
    chunks) so a test can put an exact term in an exact fraction of chunks.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE VIRTUAL TABLE chunks USING fts5(doc_id UNINDEXED, "
        "chunk_ix UNINDEXED, body, tokenize='porter unicode61');")
    conn.executemany(
        "INSERT INTO chunks(doc_id, chunk_ix, body) VALUES ('d', ?, ?)",
        [(i, row) for i, row in enumerate(rows)])
    return conn


def test_rate_multiple_default_is_now_three():
    # Named and asserted directly, per Method: the default IS the decision.
    assert ac.DOMAIN_LEXICON_RATE_MULTIPLE == 3.0


def test_a_term_at_exactly_2_5x_contrast_moves_with_the_threshold():
    """The threshold raise's whole point, made concrete: a stem that clears
    the OLD 2x bar but not the NEW 3x default.

    20 evidence chunks, "widgetword" in exactly 5 of them (rate 0.25). Two
    control corpora of 10 chunks each, "widgetword" in exactly 1 chunk of
    ONE of them (pooled: 1 of 20 chunks, rate 0.05)... adjusted below to land
    the ratio at exactly 2.5x (0.25 / 0.10), not 5x, so the test exercises
    the boundary the #522 change actually moved, not an unrelated one.
    """
    evidence_rows = (
        ["widgetword fillerword domainword"] * 5
        + ["fillerword domainword only"] * 15)
    assert len(evidence_rows) == 20
    evidence = _hand_built_chunks_conn(evidence_rows)

    # Pooled control: 20 chunks total, "widgetword" in exactly 2 of them ->
    # pooled rate 2/20 = 0.10. Evidence rate 5/20 = 0.25. 0.25 / 0.10 = 2.5x
    # exactly: admitted when rate_multiple <= 2.5, excluded above it.
    control_a = _hand_built_chunks_conn(
        ["widgetword offtopic"] * 1 + ["offtopic only"] * 9)
    control_b = _hand_built_chunks_conn(
        ["widgetword offtopic"] * 1 + ["offtopic only"] * 9)

    lexicon_at_2x = ac.domain_lexicon(
        evidence, [control_a, control_b], rate_multiple=2.0)
    lexicon_at_3x = ac.domain_lexicon(
        evidence, [control_a, control_b], rate_multiple=3.0)

    assert "widgetword" in lexicon_at_2x, (
        "setup error: the term must clear the OLD 2x bar")
    assert "widgetword" not in lexicon_at_3x, (
        "the #522 raise from 2x to 3x must exclude a term at exactly 2.5x "
        "contrast that the old default admitted")

    # And the shipped DEFAULT (no explicit rate_multiple passed) must behave
    # like the 3x call above, because the default IS 3.0 -- this is the line
    # the required mutation test below flips back to red.
    default_lexicon = ac.domain_lexicon(evidence, [control_a, control_b])
    assert "widgetword" not in default_lexicon


# THE MUTATION (Method: "a mutation ... turning it red"), performed and
# recorded rather than encoded as a second in-process test:
# `domain_lexicon(rate_multiple: float = DOMAIN_LEXICON_RATE_MULTIPLE)` binds
# its default at function-definition time, i.e. at module import -- once per
# fresh interpreter, which is once per pytest run. So the mutation that
# actually exercises "the default reverted to 2.0" is editing the source
# constant back to 2.0 and re-running the suite in a FRESH process (a
# monkeypatch of the module attribute inside a running process would not
# reach an already-bound default, and a test built to route around that by
# passing the constant explicitly would not be testing the default at all).
# Performed manually for consumer #522: with `DOMAIN_LEXICON_RATE_MULTIPLE`
# changed back to `2.0` on this branch and the suite re-run fresh,
# `test_a_term_at_exactly_2_5x_contrast_moves_with_the_threshold`'s last
# assertion (`"widgetword" not in default_lexicon`) FAILED as expected --
# "widgetword" was admitted at the reverted default, exactly reproducing the
# pre-#522 behaviour the raise was meant to change. Reverted back to 3.0
# immediately after; see the consumer issue for the transcript.


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
    step is what has to do the actual work (see the next test, which is the
    one the task means).
    """
    every_stem_in_the_question = ac.fts5_stems(GAAS_QUESTION)
    assert ac.question_in_domain(GAAS_QUESTION, every_stem_in_the_question)


# An evidence corpus built to contain exactly one incidental, off-topic
# mention of GaAs-question vocabulary -- landing in exactly ONE chunk
# (verified: `chunk_text` puts it only in the final chunk of 8) alongside
# heavy running/injury content spread across all of them. This is what makes
# the mutation test below hermetic: `min_chunks=3` (the shipped default)
# excludes a term seen in only one chunk, so the CONTRASTIVE reading refuses
# GaAs, while `min_chunks=1` (equivalent to "any word the corpus has ever
# seen", the brief's own rejected alternative) admits it. No real corpus
# needed -- the same mechanism the brief measured against 1,923 real chunks,
# reproduced at a scale a unit test can construct on purpose instead of
# stumbling into.
_EVIDENCE_WITH_INCIDENTAL_GAAS_MENTION = (
    "Weekly running volume and injury risk in distance runners. "
    "Cadence, stride length and running economy during endurance training. "
    "Recovery, sleep and heart rate variability after hard running sessions. "
) * 40 + (
    "Incidentally, someone once asked about the boiling point of gallium "
    "arsenide at trivia night."
)


def test_degenerate_every_word_lexicon_admits_gaas(tmp_path):
    """THE required mutation test: "any word the corpus has ever seen" (the
    brief's own rejected, non-contrastive alternative) must ADMIT the
    out-of-domain GaAs question, where the shipped, contrastive default
    (`min_chunks=3`) refuses it on the SAME corpus.

    `min_chunks=1` is precisely that alternative when paired with no control
    corpora (`domain_lexicon` pools an empty sequence of controls to a total
    of 0, so every stem's control rate is 0.0 and the `rate_multiple` check
    admits everything meeting `min_chunks`) -- i.e. every word the corpus has
    used at least once, no contrast and no repetition bar at all.

    This must go red if a future change made `domain_lexicon` (or
    `question_in_domain`) discriminate on something other than the corpus
    contrast -- e.g. a reimplementation that forgot `min_chunks` entirely, or
    one that treats "no controls supplied" as "refuse everything" instead of
    "admit everything meeting the other bars."
    """
    build_corpus([_entry("ev-1", _EVIDENCE_WITH_INCIDENTAL_GAAS_MENTION)],
                 [_EVIDENCE_WITH_INCIDENTAL_GAAS_MENTION],
                 tmp_path / "leaky.db", corpus_version=1)
    conn = sqlite3.connect(f"file:{tmp_path / 'leaky.db'}?mode=ro", uri=True)
    try:
        every_word_the_corpus_has_ever_seen = ac.domain_lexicon(
            conn, [], min_chunks=1)
        contrastive_default = ac.domain_lexicon(conn, [])  # min_chunks=3
    finally:
        conn.close()

    assert ac.question_in_domain(
        GAAS_QUESTION, every_word_the_corpus_has_ever_seen)
    # The shipped default, on the identical corpus, refuses it -- the
    # comparison that shows the mutation actually changed the outcome, not
    # merely that SOME lexicon admits SOME question.
    assert not ac.question_in_domain(GAAS_QUESTION, contrastive_default)


# --------------------------------------------------------------------------- #
# Hermetic — the shipped default refuses an out-of-domain question
# --------------------------------------------------------------------------- #

def test_gaas_is_refused_by_the_contrastive_default(hermetic_corpora):
    """The property `test_degenerate_every_word_lexicon_admits_gaas` exists to
    show is NOT free: it must come from the contrast, not from `cite`-style
    query sanitisation or from the question simply sharing no words at all
    with a small corpus. `hermetic_corpora`'s evidence text never mentions
    gallium/arsenide/boiling, so this also confirms the ordinary "no overlap
    at all" case refuses, which is necessary but (per the test above) not
    sufficient on its own.
    """
    evidence, control = hermetic_corpora
    lexicon = ac.domain_lexicon(evidence, [control])
    assert ac.question_in_domain(GAAS_QUESTION, lexicon) is False


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


def test_domain_lexicon_params_records_control_provenance(tmp_path):
    """A lexicon must be traceable to the control corpora it was contrasted
    against -- a count alone cannot distinguish "the same two controls" from
    "two different controls that happen to also number two". Each control's
    `corpus_file_sha256` (identifies the exact bytes) and `corpus_version`
    (may be `None` -- a control need not be a versioned corpus at all) are
    recorded, in `control_corpus_paths` order.
    """
    from health_advisor.corpus_build import corpus_file_sha256, read_corpus_version

    build_corpus([_entry("ctrl-1", _CONTROL_TEXT)], [_CONTROL_TEXT],
                 tmp_path / "control.db", corpus_version=7)
    result = build_corpus(
        [_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT],
        tmp_path / "evidence.db", corpus_version=1,
        control_corpus_paths=[tmp_path / "control.db"])
    assert result.domain_lexicon_status == ac.DOMAIN_LEXICON_STATUS_BUILT

    conn = sqlite3.connect(f"file:{tmp_path / 'evidence.db'}?mode=ro", uri=True)
    params = json.loads(conn.execute(
        "SELECT value FROM corpus_meta WHERE key = ?",
        (ac.DOMAIN_LEXICON_PARAMS_KEY,)).fetchone()[0])

    assert params["control_corpus_count"] == 1
    assert len(params["control_corpora"]) == 1
    provenance = params["control_corpora"][0]
    assert provenance["corpus_sha256"] == corpus_file_sha256(tmp_path / "control.db")
    assert provenance["corpus_version"] == 7 == read_corpus_version(tmp_path / "control.db")
    # No local path or filename leaks into a record that ships with the
    # corpus file.
    assert "control.db" not in json.dumps(provenance)
    assert str(tmp_path) not in json.dumps(provenance)


def test_domain_lexicon_params_control_version_is_none_when_unversioned(tmp_path):
    """A control corpus need not be one this module's own versioning scheme
    ever touched -- `read_corpus_version` returns `None` for a file with no
    readable `corpus_version` row, and the provenance record must say so
    plainly rather than coercing it to 0 or omitting the key.
    """
    build_corpus([_entry("ctrl-1", _CONTROL_TEXT)], [_CONTROL_TEXT],
                 tmp_path / "control.db", corpus_version=1, read_only=False)
    unversioned = sqlite3.connect(tmp_path / "control.db")
    unversioned.execute("DELETE FROM corpus_meta WHERE key = 'corpus_version'")
    unversioned.commit()
    unversioned.close()

    result = build_corpus(
        [_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT],
        tmp_path / "evidence.db", corpus_version=1,
        control_corpus_paths=[tmp_path / "control.db"])
    assert result.domain_lexicon_status == ac.DOMAIN_LEXICON_STATUS_BUILT

    conn = sqlite3.connect(f"file:{tmp_path / 'evidence.db'}?mode=ro", uri=True)
    params = json.loads(conn.execute(
        "SELECT value FROM corpus_meta WHERE key = ?",
        (ac.DOMAIN_LEXICON_PARAMS_KEY,)).fetchone()[0])
    assert params["control_corpora"][0]["corpus_version"] is None


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
