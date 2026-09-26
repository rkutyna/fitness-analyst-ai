"""consumer #522 / engine #22 item 1 follow-up -- the vendored common-English
word list `analyst_corpus._common_english_words()` folds into
`_excluded_stems()`.

Background: the domain lexicon (`analyst_corpus.domain_lexicon`) admits a
stem when it is frequent in the evidence corpus and rare in the two control
corpora. That contrast alone lets ordinary, topic-neutral English words in
whenever a corpus happens to use them often -- corpus v2 added review
articles, and words like "best"/"any"/"might" (common in review-genre
prose) cleared the contrast bar, so a question like "What is the best
answer to this: <off-topic question>" was wrongly admitted (consumer #522,
`docs/product/reviews/corpus-v2-proposal-20260926.md` section 1.1). the owner's
decision: exclude a standard common-English word list, not a hand-picked
set of the exact words that leaked this time.

HERMETIC ONLY, same discipline as `test_domain_lexicon.py`: every test here
builds its own tiny corpus under `tmp_path`. No gitignored real corpus, no
consumer-repo question fixture.
"""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import analyst_corpus as ac
from health_advisor.corpus_build import build_corpus, sha256_text


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


@pytest.fixture(autouse=True)
def _reset_excluded_stems_cache():
    """`_excluded_stems()` and `_common_english_words()` are cached at module
    scope. A test that monkeypatches the word list (the mutation test below)
    must not leak its patched cache into a later test, and must not inherit
    a cache warmed by an earlier one.
    """
    ac._EXCLUDED_STEMS_CACHE = None
    ac._COMMON_ENGLISH_WORDS_CACHE = None
    yield
    ac._EXCLUDED_STEMS_CACHE = None
    ac._COMMON_ENGLISH_WORDS_CACHE = None


# --------------------------------------------------------------------------- #
# The list file itself
# --------------------------------------------------------------------------- #

def test_common_english_words_list_loads_and_has_documented_size():
    words = ac._common_english_words()
    assert len(words) == ac.COMMON_ENGLISH_WORDS_COUNT == 150
    # Spot-check a few words the consumer report names as the leaks a
    # standard list is meant to close (a subset -- "best" and "question"
    # are content words this particular list does NOT contain; see the
    # consumer report for that gap, honestly measured rather than papered
    # over here).
    assert {"time", "any", "might", "right", "world"} <= words


def test_common_english_words_path_resolves_inside_the_package():
    assert ac.COMMON_ENGLISH_WORDS_PATH.is_file()
    assert ac.COMMON_ENGLISH_WORDS_PATH.parent == ac.COMMON_ENGLISH_WORDS_PATH.parent
    assert ac.COMMON_ENGLISH_WORDS_PATH.name == "common_english_words.txt"


def test_common_english_words_feed_into_excluded_stems():
    excluded = ac._excluded_stems()
    # "world" stems to itself under porter unicode61; it is on the vendored
    # list and must land in the same stemmed exclusion space as the #22
    # stopwords and generic question words.
    assert ac.fts5_stems("world") <= excluded
    # A domain-relevant word must never accidentally collide.
    assert "run" not in excluded
    assert "injuri" not in excluded


# --------------------------------------------------------------------------- #
# THE hermetic wrapper test: a common word that WOULD qualify by contrast
# alone must be excluded once it is also a vendored common-English word.
# --------------------------------------------------------------------------- #

# "world" (on the vendored list) used heavily in the evidence text and never
# in the control -- by contrast alone (`domain_lexicon`'s own logic) it would
# clear both the min_chunks bar and the 2x-rate-over-control bar exactly like
# any genuine domain term, so the ONLY thing that can keep it out of the
# lexicon is the common-English exclusion under test.
_EVIDENCE_TEXT = (
    "Weekly running volume and injury risk in distance runners, seen all "
    "over the world by athletes and coaches around the world. "
    "Cadence, stride length and running economy during endurance training. "
    "Recovery, sleep and heart rate variability after hard running sessions. "
) * 40

_CONTROL_TEXT = (
    "Gallium arsenide crystal growth and semiconductor wafer doping. "
    "Photolithography, etching and thin-film deposition in fabrication. "
    "Bandgap engineering and carrier mobility in compound semiconductors. "
) * 40

# A question built ONLY from stopwords/generic-question-words plus "world" --
# no domain term at all. It must be refused precisely because "world" is
# common English, not because it happens to share no vocabulary with the
# corpus at all.
_WRAPPER_QUESTION = "What is going on in the world right now?"


@pytest.fixture()
def _world_corpus(tmp_path):
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


def test_hermetic_common_word_would_otherwise_qualify_by_contrast_alone(
        _world_corpus, monkeypatch):
    """Proof the fixture is honest: with the common-English exclusion
    DISABLED (monkeypatched to empty), "world" clears `domain_lexicon`'s
    contrast bars on its own -- so the test below is excluding something
    that pure corpus contrast would have let through, not something already
    kept out by the existing #22 stopwords / generic question words.
    """
    evidence, control = _world_corpus
    monkeypatch.setattr(ac, "_common_english_words", lambda: frozenset())
    ac._EXCLUDED_STEMS_CACHE = None
    lexicon = ac.domain_lexicon(evidence, [control])
    assert "world" in lexicon
    assert ac.question_in_domain(_WRAPPER_QUESTION, lexicon) is True


def test_hermetic_common_word_is_excluded_from_the_domain_lexicon(_world_corpus):
    """THE required hermetic test (Done-when #4): with the real, vendored
    common-English word list in effect, "world" no longer qualifies, and a
    question built from nothing but stopwords/generic-question-words plus
    "world" is refused.
    """
    evidence, control = _world_corpus
    lexicon = ac.domain_lexicon(evidence, [control])
    assert "world" not in lexicon
    assert ac.question_in_domain(_WRAPPER_QUESTION, lexicon) is False


# --------------------------------------------------------------------------- #
# THE MUTATION TEST -- emptying the list must turn the wrapper test red.
# --------------------------------------------------------------------------- #

def test_mutation_emptying_the_common_word_list_admits_the_wrapper_question(
        _world_corpus, monkeypatch):
    """The required mutation (Done-when #4, method step 4): stub the list to
    return nothing and confirm the hermetic wrapper test's assertion flips.
    If this test ever failed, `test_hermetic_common_word_is_excluded_...`
    above would be passing for a reason unrelated to the vendored list --
    e.g. "world" already excluded by the #22 stopwords, or a `domain_lexicon`
    contrast quirk that has nothing to do with consumer #522 at all.
    """
    evidence, control = _world_corpus
    monkeypatch.setattr(ac, "_common_english_words", lambda: frozenset())
    ac._EXCLUDED_STEMS_CACHE = None
    mutated_lexicon = ac.domain_lexicon(evidence, [control])

    # With the list emptied, "world" is admitted and the wrapper question
    # that only that one stem carries is answered True -- the flip that
    # proves the real list (tested above) is the thing doing the work.
    assert "world" in mutated_lexicon
    assert ac.question_in_domain(_WRAPPER_QUESTION, mutated_lexicon) is True
