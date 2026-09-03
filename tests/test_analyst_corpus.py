"""Tests for `health_advisor.analyst_corpus` -- ``cite()``.

Everything asserted here that can be run against the real corpus is run
against the real corpus (``data/corpus/corpus.db``, 45 docs / 1,923 chunks,
opened read-only), because the properties at stake -- a span being verbatim, a
sort being the right way round, an orphan being unreachable -- are properties
of real FTS5 behaviour and a synthetic fixture can be built to satisfy them by
accident.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import os
import re
import shutil
import socket
import sqlite3
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from health_advisor import analyst_corpus as ac
from health_advisor import analyst_envelope, analyst_ledger, analyst_runner

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.db"

# Queries that retrieve broadly across the real corpus. Written the way a coach
# asks (`RELEVANCE-MEASUREMENT.md` calls this the "lay" phrasing), because that
# is the phrasing the product actually has to survive.
LIVE_QUERIES = [
    "does poor sleep increase injury risk in runners",
    "how much should i increase my weekly mileage",
    "what does heart rate variability tell me about recovery",
    "is running cadence related to injury",
    "how long does it take to improve vo2max with training",
    "does resting heart rate go down as i get fitter",
    "carbohydrate intake before a long run",
    "strength training for distance runners",
    "how much protein do endurance athletes need",
    "what is lactate threshold and why does it matter",
]


def _require_real_corpus():
    if not REAL_CORPUS.exists():  # pragma: no cover - environment guard
        pytest.skip(f"real corpus not present at {REAL_CORPUS}")


@pytest.fixture()
def corpus():
    _require_real_corpus()
    conn = ac.open_corpus(REAL_CORPUS)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def orphan_corpus(tmp_path):
    """A copy of the real corpus carrying one chunk with no ``docs`` row.

    #229: FTS5 virtual tables cannot carry a foreign key, so a chunk can
    outlive its ``docs`` row. Forging one is the only way to test the defence,
    and it is forged in a temp copy -- the real corpus stays ``chmod 444``.
    """
    _require_real_corpus()
    path = tmp_path / "orphan.db"
    shutil.copy(REAL_CORPUS, path)
    path.chmod(0o644)
    writer = sqlite3.connect(path)
    writer.execute(
        "INSERT INTO chunks(doc_id, chunk_ix, body) VALUES (?, ?, ?)",
        ("orphan-doc", 0,
         "Sleep deprivation orphanmarker substantially increases the injury "
         "risk of distance runners across every training volume studied."),
    )
    writer.commit()
    writer.close()
    conn = ac.open_corpus(path)
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# 1. Zero parameters accept a score threshold
# --------------------------------------------------------------------------- #

# Every spelling a threshold could arrive under. The point of the list is that
# it is checked mechanically: a future parameter called `min_relevance` is
# caught by the test rather than by a reviewer remembering
# `RELEVANCE-MEASUREMENT.md`.
_THRESHOLD_WORDS = re.compile(
    r"(min_score|max_score|score_cutoff|score_threshold|threshold|cutoff|"
    r"min_relevance|min_bm25|max_bm25|min_rank|relevance_floor|score_floor|"
    r"score_min|score_max)", re.IGNORECASE)


def _public_callables():
    return [(name, getattr(ac, name)) for name in ac.__all__
            if callable(getattr(ac, name))]


def test_no_public_parameter_accepts_a_score_threshold():
    """`Done when` 1. Zero parameters across the public surface, measured."""
    offenders = []
    signatures = {}
    for name, obj in _public_callables():
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover - builtins only
            continue
        signatures[name] = str(sig)
        for param in sig.parameters:
            if _THRESHOLD_WORDS.search(param):
                offenders.append(f"{name}({param})")
    # Dataclass fields are part of the surface too: CiteCaps is where a
    # threshold would most plausibly be smuggled in as "just another cap".
    for field_name in ac.CiteCaps.__dataclass_fields__:
        if _THRESHOLD_WORDS.search(field_name):
            offenders.append(f"CiteCaps.{field_name}")
    for field_name in ac.CiteState.__dataclass_fields__:
        if _THRESHOLD_WORDS.search(field_name):
            offenders.append(f"CiteState.{field_name}")
    assert offenders == [], f"score-threshold parameters found: {offenders}"
    assert len(signatures) >= 8, signatures


def test_score_is_returned_but_never_filtered_on():
    """A threshold could also arrive as a bare comparison, with no parameter.

    `RELEVANCE-MEASUREMENT.md` S2: 12 of 16 passages from a corpus of plant
    genomics and materials science outscored the worst genuinely relevant hit.
    No bm25 value separates the two, so nothing in this module may branch on
    one.
    """
    tree = ast.parse(Path(ac.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert "score" not in names, ast.dump(node)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            for gen in node.generators:
                for cond in gen.ifs:
                    src = ast.dump(cond)
                    assert "score" not in src, src


def test_score_is_still_reported():
    """Returned, just never acted on -- it is real and it is bm25."""
    _require_real_corpus()
    conn = ac.open_corpus(REAL_CORPUS)
    try:
        passages = ac.cite(conn, LIVE_QUERIES[0], state=ac.CiteState())
    finally:
        conn.close()
    assert passages
    assert all(isinstance(p.score, float) and p.score < 0 for p in passages)


# --------------------------------------------------------------------------- #
# 2. The hostile-query battery: typed refusals, zero tracebacks
# --------------------------------------------------------------------------- #

HOSTILE_QUERIES = [
    ("bare_numeric", "1.6"),
    ("unbalanced_quote", 'sleep "injury'),
    ("fts5_operators", "AND AND ("),
    ("over_length", "sleep " * 33 + "abc"),        # 201 characters
    ("star_alone", "*"),
    ("nul_byte", "sleep\x00injury"),
    ("sql_injection", "sleep'; DROP TABLE docs; --"),
    ("empty", ""),
    ("stopwords_only", "the of a and"),
    ("non_string", 3.14),
    ("only_operators", "AND OR NOT NEAR"),
    ("column_filter", "body:sleep"),
    ("prefix_glob", "sleep* OR injur*"),
    ("whitespace_only", "   \t  "),
]


def test_the_over_length_query_really_is_201_characters():
    """The cap is 200; the fixture has to be exactly one past it to test it."""
    query = dict(HOSTILE_QUERIES)["over_length"]
    assert len(query) == 201


def test_every_hostile_query_is_a_typed_refusal_and_never_a_traceback(corpus):
    """`Done when` 2. Each of these raises `CiteRefusal` and nothing else.

    "Nothing else" is the load-bearing half: `sqlite3.OperationalError` carries
    the offending query text, and proposal S5.3 treats a traceback reaching a
    repair turn as an unclosed data channel.
    """
    reasons = {}
    for label, query in HOSTILE_QUERIES:
        state = ac.CiteState()
        try:
            result = ac.cite(corpus, query, state=state)
        except ac.CiteRefusal as exc:
            reasons[label] = exc.reason
            assert exc.reason.startswith("corpus.cite.")
            assert exc.code and " " not in exc.code
            assert state.refused_calls == 1
            assert state.calls == 0
            continue
        except Exception as exc:  # pragma: no cover - the thing under test
            pytest.fail(f"{label!r} raised {type(exc).__name__}: {exc}")
        pytest.fail(f"{label!r} was not refused; it returned {len(result)} passages")
    assert len(reasons) == len(HOSTILE_QUERIES)
    # The ten the task names, each with its own distinguishable reason.
    named = ["bare_numeric", "unbalanced_quote", "fts5_operators",
             "over_length", "star_alone", "nul_byte", "sql_injection",
             "empty", "stopwords_only", "non_string"]
    assert len({reasons[label] for label in named}) == 10


def test_a_refused_query_never_reaches_fts5_as_syntax(corpus):
    """The escaping claim, checked directly rather than through `cite`.

    Measured raw, without the harness: ``1.6`` -> ``fts5: syntax error near
    "."``; ``sleep "injury`` -> ``unterminated string``; ``AND AND (`` ->
    ``syntax error near "AND"``; ``*`` -> ``unknown special query``. Those are
    the errors this module exists to prevent from ever being raised.
    """
    for _, query in HOSTILE_QUERIES:
        with pytest.raises(ac.CiteRefusal):
            ac.build_match_expression(query)


def test_terms_that_would_break_fts5_are_quoted_rather_than_refused(corpus):
    """The harness owns syntax, so a legitimate awkward term still works.

    ``1.6`` alone is refused as a figure-shaped query, but ``1.6`` beside a
    content word is a term and must survive: quoting is what turns the measured
    ``fts5: syntax error near "."`` into an ordinary phrase match.
    """
    expression = ac.build_match_expression("running economy 1.6 percent")
    assert expression == '"running" OR "economy" OR "1.6" OR "percent"'
    # And it executes.
    ac.cite(corpus, "running economy 1.6 percent", state=ac.CiteState())
    assert ac.build_match_expression("high-intensity interval training") == (
        '"high-intensity" OR "interval" OR "training"')


def test_or_is_the_measured_strategy_and_no_term_is_dropped():
    """`RELEVANCE-MEASUREMENT.md` S3: OR-of-all-terms, 81%; AND collapses it."""
    expression = ac.build_match_expression("the sleep and injury risk of runners")
    assert " AND " not in expression
    assert expression.count(" OR ") == 6
    # Stopwords stay in the executed query. They are used only to detect a
    # query that has nothing else in it.
    assert '"the"' in expression and '"of"' in expression


# --------------------------------------------------------------------------- #
# 3. Caps
# --------------------------------------------------------------------------- #

def test_k_of_fifty_returns_five(corpus):
    """`Done when` 3a. Clamped, not refused."""
    state = ac.CiteState()
    passages = ac.cite(corpus, LIVE_QUERIES[0], k=50, state=state)
    assert len(passages) == 5 == ac.CITE_CAPS.passages_per_call
    assert state.passages_returned == 5


def test_k_below_one_is_refused(corpus):
    for bad in (0, -1, 2.5, "5", True):
        with pytest.raises(ac.CiteRefusal) as caught:
            ac.cite(corpus, LIVE_QUERIES[0], k=bad, state=ac.CiteState())
        assert caught.value.code == "bad_k"


def test_the_ninth_cite_call_in_a_run_is_refused(corpus):
    """`Done when` 3b.

    The evidence-byte cap is the binding one against the real corpus (see
    `test_the_evidence_byte_cap_trips`), so the call cap is isolated by
    raising the byte budget. Raising a cap is a parent-side act: `CiteCaps` is
    frozen and the child cannot construct a `CiteState` at all.
    """
    state = ac.CiteState(caps=replace(ac.CITE_CAPS, evidence_bytes=10 ** 9,
                                      docs_per_run=10 ** 6))
    for index in range(8):
        assert ac.cite(corpus, LIVE_QUERIES[index], state=state)
    assert state.calls == 8
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.cite(corpus, LIVE_QUERIES[8], state=state)
    assert caught.value.code == "call_cap"
    assert state.calls == 8, "a refused call must not consume budget"


def test_the_evidence_byte_cap_trips(corpus):
    """`Done when` 3c, with the default 16,384-byte budget."""
    state = ac.CiteState(caps=replace(ac.CITE_CAPS, docs_per_run=10 ** 6))
    spent = []
    for index in range(ac.CITE_CAPS.calls_per_run):
        try:
            ac.cite(corpus, LIVE_QUERIES[index], state=state)
        except ac.CiteRefusal as exc:
            assert exc.code == "evidence_byte_cap"
            assert state.evidence_bytes <= ac.CITE_CAPS.evidence_bytes
            assert spent, "the cap must not trip on the first call"
            return
        spent.append(state.evidence_bytes)
    pytest.fail(f"byte cap never tripped; spent {state.evidence_bytes}")


def test_the_distinct_document_cap_trips(corpus):
    state = ac.CiteState(caps=replace(ac.CITE_CAPS, evidence_bytes=10 ** 9))
    for index in range(ac.CITE_CAPS.calls_per_run):
        try:
            ac.cite(corpus, LIVE_QUERIES[index], state=state)
        except ac.CiteRefusal as exc:
            assert exc.code == "doc_cap"
            assert len(state.docs_cited) <= ac.CITE_CAPS.docs_per_run
            return
    pytest.fail(f"doc cap never tripped; {len(state.docs_cited)} docs cited")


def test_a_refused_call_leaves_the_run_record_unchanged(corpus):
    state = ac.CiteState()
    ac.cite(corpus, LIVE_QUERIES[0], state=state)
    before = state.citation_ledger()
    with pytest.raises(ac.CiteRefusal):
        ac.cite(corpus, "1.6", state=state)
    after = state.citation_ledger()
    for key in ("cite_calls", "passages_returned", "evidence_bytes",
                "docs_cited", "queries", "chunks_returned"):
        assert before[key] == after[key]
    assert after["cite_refusals"] == 1


def test_the_span_is_capped_at_sixty_four_tokens(corpus):
    """FTS5's own documented ceiling, applied by us because 3.51.0 does not.

    Measured 2026-08-30: ``snippet(..., 100)`` returned 666 characters against
    ``snippet(..., 64)``'s 438. Nothing raised and nothing clamped, so the cap
    is enforced by never passing anything but `DEFAULT_SNIPPET_TOKENS`.
    """
    assert ac.DEFAULT_SNIPPET_TOKENS == 64 == ac.CITE_CAPS.span_tokens
    assert "snippet(chunks, 2, '', '', '', ?)" in ac._SELECT
    state = ac.CiteState(caps=replace(ac.CITE_CAPS, evidence_bytes=10 ** 9,
                                      docs_per_run=10 ** 6))
    seen = 0
    for query in LIVE_QUERIES[:8]:
        for passage in ac.cite(corpus, query, state=state):
            # unicode61 tokenises on runs of alphanumerics, so a bare "."
            # between sentences is whitespace-separated but is not a token --
            # `split()` overcounts and would fail a correct 64-token span.
            assert len(re.findall(r"\w+", passage.span)) <= 64
            assert len(passage.span) < 1200  # never a whole chunk
            seen += 1
    assert seen >= 30


def test_the_query_length_cap_is_two_hundred(corpus):
    assert ac.CITE_CAPS.query_chars == 200
    at_the_cap = "sleep " * 33 + "ab"
    assert len(at_the_cap) == 200
    assert ac.cite(corpus, at_the_cap, state=ac.CiteState())
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.cite(corpus, at_the_cap + "c", state=ac.CiteState())
    assert caught.value.code == "query_too_long"


def test_the_wall_clock_cap_is_enforced_inside_sqlite(corpus):
    """A zero-second budget must abort the query, not be checked afterwards."""
    state = ac.CiteState(caps=replace(ac.CITE_CAPS, wall_clock_s=-1.0))
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.cite(corpus, LIVE_QUERIES[0], state=state)
    assert caught.value.code == "retrieval_timeout"
    # And the connection is left usable: the handler is cleared in `finally`.
    assert ac.cite(corpus, LIVE_QUERIES[0], state=ac.CiteState())


# --------------------------------------------------------------------------- #
# 4. The span is verbatim
# --------------------------------------------------------------------------- #

def test_every_returned_span_is_verbatim_in_the_corpus(corpus):
    """`Done when` 4, over well past 100 passages against the real corpus.

    This is the mechanism the whole citation design rests on (design S2.3):
    the model selects a quotation, it never types one. A paraphrase produces a
    span that will not resolve, and resolution is string comparison, not
    judgement.
    """
    bodies = {(doc_id, chunk_ix): body for doc_id, chunk_ix, body in
              corpus.execute("SELECT doc_id, chunk_ix, body FROM chunks")}
    checked = 0
    exact = 0
    caps = replace(ac.CITE_CAPS, evidence_bytes=10 ** 9, calls_per_run=10 ** 6,
                   docs_per_run=10 ** 6)
    for query in LIVE_QUERIES * 4:
        for passage in ac.cite(corpus, query, k=5, state=ac.CiteState(caps=caps)):
            body = bodies[(passage.doc_id, passage.chunk_ix)]
            assert ac.span_is_verbatim(passage.span, body), (
                passage.doc_id, passage.chunk_ix, passage.span[:80])
            exact += passage.span in body
            checked += 1
    assert checked >= 100, checked
    assert exact == checked, (
        f"{checked - exact} of {checked} spans needed whitespace normalisation")


def test_a_paraphrase_does_not_resolve():
    body = "Sleep deficiency and poor sleep quality are common in athletes."
    assert ac.span_is_verbatim("poor sleep quality", body)
    assert not ac.span_is_verbatim("poor sleep habits", body)


def test_normalize_span_does_not_case_fold_or_strip_punctuation():
    assert ac.normalize_span("  a   b\nc ") == "a b c"
    assert ac.normalize_span("Injury.") == "Injury."
    assert not ac.span_is_verbatim("injury", "Injury risk")


# --------------------------------------------------------------------------- #
# 5. Orphan chunks
# --------------------------------------------------------------------------- #

def test_the_forged_orphan_really_is_an_orphan(orphan_corpus):
    """Without the join it is retrievable, and its span is genuinely verbatim.

    That is #229's whole point: an orphan passes every check except a join.
    """
    raw = orphan_corpus.execute(
        "SELECT doc_id FROM chunks WHERE chunks MATCH ?",
        ('"orphanmarker"',)).fetchall()
    assert raw == [("orphan-doc",)]
    assert orphan_corpus.execute(
        "SELECT count(*) FROM docs WHERE doc_id = ?",
        ("orphan-doc",)).fetchone() == (0,)


def test_an_orphan_chunk_is_never_returned(orphan_corpus):
    """`Done when` 5, primary mechanism: the inner join makes it unreachable."""
    caps = replace(ac.CITE_CAPS, evidence_bytes=10 ** 9, docs_per_run=10 ** 6)
    state = ac.CiteState(caps=caps)
    assert ac.cite(orphan_corpus, "orphanmarker", state=state) == []
    for query in ("sleep injury risk runners orphanmarker",
                  "orphanmarker deprivation distance runners training volume"):
        for passage in ac.cite(orphan_corpus, query, state=state):
            assert passage.doc_id != "orphan-doc"
            assert passage.title is not None


def test_an_orphan_does_not_consume_a_result_slot(orphan_corpus):
    """The join filters before LIMIT, so the orphan costs nothing."""
    state = ac.CiteState()
    passages = ac.cite(
        orphan_corpus, "sleep injury risk runners orphanmarker", k=5, state=state)
    assert len(passages) == 5


def test_a_named_orphan_gets_its_own_typed_refusal(orphan_corpus):
    """`Done when` 5, secondary mechanism -- and secondary on purpose."""
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.cite(orphan_corpus, "sleep injury", state=ac.CiteState(),
                doc_id="orphan-doc")
    assert caught.value.code == "orphan_doc"
    assert "no docs row" in caught.value.reason


def test_a_named_unknown_doc_is_a_different_refusal(orphan_corpus):
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.cite(orphan_corpus, "sleep injury", state=ac.CiteState(),
                doc_id="no-such-doc-at-all")
    assert caught.value.code == "unknown_doc"


def test_a_named_doc_filter_still_goes_through_match_and_the_caps(corpus):
    state = ac.CiteState()
    unfiltered = ac.cite(corpus, LIVE_QUERIES[0], state=ac.CiteState())
    target = unfiltered[0].doc_id
    passages = ac.cite(corpus, LIVE_QUERIES[0], k=50, state=state, doc_id=target)
    assert passages
    assert len(passages) <= ac.CITE_CAPS.passages_per_call
    assert {p.doc_id for p in passages} == {target}


# --------------------------------------------------------------------------- #
# 6. bm25 is negative; the sort must be ascending
# --------------------------------------------------------------------------- #

def test_reversed_sort_would_be_caught(corpus):
    """`Done when` 6. THIS is the test that fails if the comparison flips.

    ``bm25()`` in SQLite is NEGATIVE and more negative is better. A reversed
    sort returns the *worst* matches while looking perfectly correct -- same
    shape, same count, plausible spans -- so a test that only checks
    monotonicity would pass under the flip. This one pins the returned set
    against the true minima of a wide fetch, which a reversed sort cannot
    satisfy.
    """
    query = LIVE_QUERIES[0]
    expression = ac.build_match_expression(query)
    wide = corpus.execute(
        "SELECT chunks.doc_id, chunks.chunk_ix, bm25(chunks) "
        "  FROM chunks JOIN docs d ON d.doc_id = chunks.doc_id "
        " WHERE chunks MATCH ? ORDER BY bm25(chunks) ASC",
        (expression,)).fetchall()
    assert len(wide) > 5, "need a wide result for the comparison to bite"
    assert wide[0][2] < 0, "bm25 is negative; this whole test assumes it"
    assert wide[0][2] < wide[-1][2], "ascending means best first"

    passages = ac.cite(corpus, query, k=5, state=ac.CiteState())
    got = [(p.doc_id, p.chunk_ix) for p in passages]
    best_five = [(row[0], int(row[1])) for row in wide[:5]]
    worst_five = [(row[0], int(row[1])) for row in reversed(wide[-5:])]
    assert got == best_five
    assert got != worst_five, "a reversed sort would return exactly these"

    scores = [p.score for p in passages]
    assert scores == sorted(scores), "returned order must be ascending bm25"
    assert scores[0] == min(row[2] for row in wide)


def test_ascending_is_spelled_out_in_the_sql():
    """A default sort direction is a comment away from being flipped."""
    assert "ORDER BY bm25(chunks) ASC" in ac._SELECT


# --------------------------------------------------------------------------- #
# 7. Latency
# --------------------------------------------------------------------------- #

def test_retrieval_latency(corpus):
    """`Done when` 7. Reported as a median; asserted only loosely.

    The assertion is deliberately far from the measurement: a per-call bound
    tight enough to be interesting is a bound tight enough to flake on a loaded
    machine, and the number that matters is printed rather than asserted.
    """
    caps = replace(ac.CITE_CAPS, evidence_bytes=10 ** 9, calls_per_run=10 ** 6,
                   docs_per_run=10 ** 6)
    state = ac.CiteState(caps=caps)
    timings = []
    for index in range(60):
        query = LIVE_QUERIES[index % len(LIVE_QUERIES)]
        start = time.perf_counter()
        ac.cite(corpus, query, state=state)
        timings.append((time.perf_counter() - start) * 1000.0)
    median = statistics.median(timings)
    print(f"\ncite() latency over {len(timings)} calls: "
          f"median {median:.3f} ms, p95 {sorted(timings)[56]:.3f} ms, "
          f"max {max(timings):.3f} ms")
    assert len(timings) >= 50
    assert median < 1000.0 * ac.CITE_CAPS.wall_clock_s


# --------------------------------------------------------------------------- #
# 9 + 10. The citation ledger is NOT the vault ledger
# --------------------------------------------------------------------------- #

VAULT_LEDGER_KEYS = frozenset(
    analyst_ledger.LedgerSummary(0, 0, (), ()).as_dict())


def test_the_two_ledgers_share_no_key(corpus):
    """A careless ``dict.update`` must not be able to satisfy the zero-read gate."""
    state = ac.CiteState()
    ac.cite(corpus, LIVE_QUERIES[0], state=state)
    citation = state.citation_ledger()
    assert VAULT_LEDGER_KEYS == {"query_count", "rows_read", "tables_read",
                                 "columns_read"}
    assert VAULT_LEDGER_KEYS.isdisjoint(citation)
    assert citation["parent_observed"] is True


def test_this_module_never_touches_a_vault_read_ledger():
    """`Done when` 10, checked on the AST so prose cannot make it pass.

    Comments and docstrings are allowed to name ``tables_read`` -- the comment
    on `CiteState` has to name the bypass to be worth anything. What must be
    zero is executable references.
    """
    source = Path(ac.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))

    forbidden = set(VAULT_LEDGER_KEYS) | {"analyst_ledger", "open_ledgered",
                                          "LedgerSummary", "LedgeredConnection"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            hits.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in forbidden:
            hits.append(node.attr)
        elif isinstance(node, ast.keyword) and node.arg in forbidden:
            hits.append(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            hits.extend(n for n in names if n in forbidden)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings and node.value in forbidden:
            hits.append(node.value)
    assert hits == [], f"executable references to the vault ledger: {hits}"
    assert "analyst_ledger" not in {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}


def test_no_public_symbol_merges_the_two_ledgers():
    """`Done when` (constraint 3): no convenience that combines them exists."""
    for name, obj in _public_callables():
        assert "merge" not in name.lower()
    for name in dir(ac.CiteState):
        if name.startswith("__"):
            continue
        assert "merge" not in name.lower()
        assert "ledger" not in name.lower() or name == "citation_ledger"


def test_a_cite_call_leaves_the_vault_ledger_empty(corpus):
    """The corpus read is invisible to the vault's accumulator, by construction.

    `cite` is handed a corpus connection and a `CiteState` and nothing else --
    there is no argument through which a vault ledger could be reached.
    """
    vault_state = analyst_ledger._LedgerState()
    before = vault_state.summary().as_dict()
    cite_state = ac.CiteState()
    passages = ac.cite(corpus, LIVE_QUERIES[0], state=cite_state)
    assert passages
    after = vault_state.summary().as_dict()
    assert before == after == {"query_count": 0, "rows_read": 0,
                               "tables_read": [], "columns_read": []}
    assert cite_state.citation_ledger()["cite_calls"] == 1
    assert cite_state.citation_ledger() is not after


def _numeric_envelope_bytes() -> bytes:
    unit = sorted(analyst_envelope.ALLOWED_UNITS)[0]
    return json.dumps({"tables": [{
        "name": "weekly", "columns": ["mileage"], "units": [unit],
        "rows": [[42.0], [43.5]]}]}).encode("utf-8")


def _validate_with(ledger: dict):
    return analyst_envelope.validate(
        _numeric_envelope_bytes(), run_id="r" * 32, question="",
        code_sha256="a" * 64, vault_sha256="b" * 64, vault_version=1,
        ledger=ledger)


def _numeric_spelling_payload(spellings: list[str]) -> bytes:
    """Build JSON while preserving each numeric literal's source spelling."""
    unit = json.dumps(sorted(analyst_envelope.ALLOWED_UNITS)[0])
    first = spellings[:analyst_envelope.MAX_ROWS_PER_TABLE]
    second = spellings[analyst_envelope.MAX_ROWS_PER_TABLE:]

    def table(name, literals):
        rows = ",".join(f"[{literal}]" for literal in literals)
        return ("{\"name\":" + json.dumps(name) +
                ",\"columns\":[\"value\"],\"units\":[" + unit +
                "],\"rows\":[" + rows + "]}")

    return ("{\"tables\":[" + table("spelling_one", first) + "," +
            table("spelling_two", second) + "]}").encode("ascii")


def test_numeric_token_cap_is_explicitly_on_distinct_values():
    """Equivalent numeric spellings share one parsed grounding value.

    The 201 spellings below are intentionally all ``0.1`` after JSON parsing;
    the value cap therefore accepts them, while 201 distinct parsed values
    still refuse at the same boundary.
    """
    ledger = {"query_count": 1, "rows_read": 201,
              "tables_read": ["source"], "columns_read": []}
    spellings = ["1e-" + ("0" * i) + "1" for i in range(201)]
    accepted = analyst_envelope.validate(
        _numeric_spelling_payload(spellings), run_id="r" * 32, question="",
        code_sha256="a" * 64, vault_sha256="b" * 64, vault_version=1,
        ledger=ledger)
    assert isinstance(accepted, analyst_envelope.Envelope)
    assert accepted.counts["numeric_tokens"] == 1

    distinct = [str(i) for i in range(201)]
    refused = analyst_envelope.validate(
        _numeric_spelling_payload(distinct), run_id="r" * 32, question="",
        code_sha256="a" * 64, vault_sha256="b" * 64, vault_version=1,
        ledger=ledger)
    assert isinstance(refused, analyst_envelope.Refusal)
    assert refused.reason == "envelope exceeds distinct numeric-token cap: 201 > 200"


def test_the_zero_read_gate_still_refuses_after_a_successful_cite(corpus):
    """`Done when` 9. The bypass, run.

    A child that calls `cite` successfully, never reads the vault, and then
    emits numeric cells must still be refused. The gate lives in a file this
    task may not edit, so the ledger state is constructed directly and the gate
    is invoked on it.
    """
    cite_state = ac.CiteState()
    assert ac.cite(corpus, LIVE_QUERIES[0], state=cite_state)

    empty_vault = analyst_ledger._LedgerState().summary().as_dict()
    refusal = _validate_with(empty_vault)
    assert isinstance(refusal, analyst_envelope.Refusal), refusal
    print(f"\nzero-read gate refusal: {refusal.reason}")

    # And the stronger form: even a careless merge of the two records cannot
    # satisfy the gate, because the key sets are disjoint by construction.
    merged = dict(empty_vault)
    merged.update(cite_state.citation_ledger())
    still_refused = _validate_with(merged)
    assert isinstance(still_refused, analyst_envelope.Refusal), still_refused
    assert still_refused.reason == refusal.reason


def test_the_gate_passes_once_the_vault_actually_was_read():
    """The control: the gate is not simply refusing everything."""
    vault_state = analyst_ledger._LedgerState()
    vault_state.query_count = 1
    vault_state.rows_read = 2
    vault_state.tables_read.add("daily_metrics")
    vault_state.columns_read.add(("daily_metrics", "value"))
    result = _validate_with(vault_state.summary().as_dict())
    assert isinstance(result, analyst_envelope.Envelope), result


# --------------------------------------------------------------------------- #
# The child-side proxy and the runner dispatch
# --------------------------------------------------------------------------- #

def _compose_child_source(code_path: str, query_fd: int, out_fd: int) -> str:
    base = (analyst_runner.RUNNER_TEMPLATE
            .replace("__code_path__", repr(code_path))
            .replace("__query_fd__", str(query_fd))
            .replace("__out_fd__", str(out_fd)))
    return ac.child_source_with_cite(base)


def test_the_child_source_compiles_and_binds_cite():
    source = _compose_child_source("/tmp/code.py", 4, 3)
    compile(source, "<child>", "exec")
    assert '"cite": cite' in source
    assert "def cite(query, k=5, doc_id=None):" in source
    # The child gets a function and nothing else: no connection, no path.
    assert "sqlite3" not in ac.CHILD_CITE_SNIPPET
    assert "corpus" not in ac.CHILD_CITE_SNIPPET.lower()


def test_child_source_with_cite_fails_loudly_on_template_drift():
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.child_source_with_cite("nothing that looks like the runner")
    assert caught.value.code == "runner_template_drift"


def test_the_documented_one_liner_matches_serve_cite_frame():
    """The docstring's wiring instruction has to stay true to the signature."""
    assert ac.DISPATCH_ONE_LINER in ac.__doc__
    sig = inspect.signature(ac.serve_cite_frame)
    assert list(sig.parameters) == ["payload", "send", "corpus_conn", "state"]
    assert list(inspect.signature(ac.serve_cite).parameters) == [
        "payload", "corpus_conn", "state"]


def test_serve_cite_returns_a_json_safe_dict_and_never_raises(corpus):
    state = ac.CiteState()
    ok = ac.serve_cite({"op": "cite", "query": LIVE_QUERIES[0], "k": 5},
                       corpus, state)
    assert ok["ok"] is True and len(ok["passages"]) == 5
    json.dumps(ok, allow_nan=False)
    for payload in ({"op": "cite", "query": "1.6"},
                    {"op": "cite", "query": None},
                    {"op": "cite", "query": "sleep", "k": "many"},
                    {"op": "cite", "query": {"__unencodable__": "object"}},
                    "not a dict"):
        bad = ac.serve_cite(payload, corpus, ac.CiteState())
        assert bad["ok"] is False
        assert bad["error_type"] == "CiteRefusal"
        assert bad["error"].startswith("corpus.cite.")
        json.dumps(bad, allow_nan=False)


def test_serve_cite_frame_never_aborts_the_run(corpus):
    sent = []
    more, cap_reason = ac.serve_cite_frame(
        {"op": "cite", "query": "1.6"}, sent.append, corpus, ac.CiteState())
    assert (more, cap_reason) == (True, None)
    assert sent[0]["ok"] is False


def test_the_child_can_call_cite_end_to_end(corpus, tmp_path):
    """The whole wire path: model code -> fd -> parent retrieval -> passages.

    Run as a real subprocess against the real corpus, because the thing being
    tested is that the child gets passages over a file descriptor and nothing
    else.
    """
    code_path = tmp_path / "code.py"
    code_path.write_text(
        "ps = cite('does poor sleep increase injury risk in runners', k=3)\n"
        "print(len(ps), ps[0].doc_id, ps[0].chunk_ix, len(ps[0].span))\n"
        "print('title', bool(ps[0].title), 'license', ps[0].license)\n"
        "try:\n"
        "    cite('1.6')\n"
        "except CiteRefusal as exc:\n"
        "    print('refused', exc)\n"
        "print('has_conn_attr', hasattr(ps[0], 'execute'))\n",
        encoding="utf-8")

    parent_sock, child_sock = socket.socketpair()
    child_fd = child_sock.detach()
    out_r, out_w = os.pipe()
    runner_path = tmp_path / "runner.py"
    runner_path.write_text(
        _compose_child_source(str(code_path), child_fd, out_w), encoding="utf-8")

    state = ac.CiteState()
    proc = subprocess.Popen(
        [sys.executable, "-I", str(runner_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=(out_w, child_fd), close_fds=True)
    os.close(out_w)
    os.close(child_fd)

    parent_sock.settimeout(20)
    pending = bytearray()
    try:
        while True:
            try:
                chunk = parent_sock.recv(65536)
            except (socket.timeout, OSError):
                break
            if not chunk:
                break
            pending.extend(chunk)
            while len(pending) >= 4:
                size = struct.unpack("!I", bytes(pending[:4]))[0]
                if len(pending) < 4 + size:
                    break
                request = json.loads(bytes(pending[4:4 + size]).decode("utf-8"))
                del pending[:4 + size]
                assert request.get("op") == "cite", request
                response = ac.serve_cite(request, corpus, state)
                body = json.dumps(response, separators=(",", ":")).encode()
                parent_sock.sendall(struct.pack("!I", len(body)) + body)
    finally:
        parent_sock.close()
    stdout, stderr = proc.communicate(timeout=30)
    os.close(out_r)

    text = stdout.decode()
    assert stderr == b"", stderr.decode()
    assert text.startswith("3 "), text
    assert "title True license" in text
    assert "refused corpus.cite.numeric_only_query" in text
    assert "has_conn_attr False" in text

    # And the parent -- not the child -- is what recorded the retrieval.
    ledger = state.citation_ledger()
    assert ledger["cite_calls"] == 1
    assert ledger["cite_refusals"] == 1
    assert ledger["passages_returned"] == 3
    assert len(ledger["chunks_returned"]) == 3
    assert ledger["parent_observed"] is True


# --------------------------------------------------------------------------- #
# Corpus handling
# --------------------------------------------------------------------------- #

def test_open_corpus_has_no_default_path_and_no_env_var():
    """T-003: the path is passed in, never defaulted, never ambient."""
    sig = inspect.signature(ac.open_corpus)
    assert sig.parameters["corpus_path"].default is inspect.Parameter.empty
    # Checked on the AST: the docstring is allowed to say "environment
    # variable", the code is not allowed to read one.
    tree = ast.parse(Path(ac.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("environ", "getenv"), node.attr
        if isinstance(node, ast.Name):
            assert node.id not in ("environ", "getenv"), node.id
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            names.append(getattr(node, "module", "") or "")
            assert "os" not in names, names


def test_the_corpus_is_opened_read_only(corpus):
    with pytest.raises(sqlite3.OperationalError):
        corpus.execute("INSERT INTO corpus_meta(key, value) VALUES ('x','y')")


def test_open_corpus_refuses_a_missing_file(tmp_path):
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.open_corpus(tmp_path / "nope.db")
    assert caught.value.code == "no_such_corpus"


def test_temp_store_is_never_set():
    """Measured absence, matching `corpus_build`: no sort spill, no hazard.

    ``corpus_build``'s docstring records the measurement (no spill to 800,000
    chunks / 2.4 GB with an unwritable TMPDIR, and ``temp_store=MEMORY`` about
    2x slower on this workload). A pragma justified by a hazard that does not
    exist is a pessimisation with a comment on it.
    """
    source = Path(ac.__file__).read_text(encoding="utf-8")
    assert "temp_store" not in source


# --------------------------------------------------------------------------- #
# #232. A refused cite() call costs something, and the record of it is bounded
#
# Before this section existed, measured against the real corpus:
#
#   30 malformed queries:    cite_calls=0  cite_refusals=30   the 8-call cap
#                            never reached; a valid cite() afterwards still
#                            returned 5 passages
#   5000 malformed queries:  refusal_codes entries=5000
#                            citation_ledger JSON = 1,211 KB (`query_syntax`)
#                                                 = 1,016 KB (`numeric_only_query`)
#
# Validation happens before any MATCH, so that loop costs the child a function
# call and costs the PARENT an allocation -- growth in the parent process,
# which is why the sandbox never sees it.
# --------------------------------------------------------------------------- #

# The two refusals the loop is written against. ``query_syntax`` is the one the
# issue names; ``numeric_only_query`` is a second code so that the per-code
# tally is tested with more than one key in it.
_LOOP_QUERY = "sleep* OR injur*"        # -> query_syntax
_LOOP_QUERY_2 = "1.6"                   # -> numeric_only_query

_LOOP_ATTEMPTS = 5000


def test_the_loop_queries_are_refused_before_any_match_reaches_fts5(corpus):
    """The premise of the whole section: these cost validation and nothing else.

    If either of these ever became a *retrievable* query the measurements below
    would be measuring a different thing, and the bound would look like it was
    holding when it was only being made expensive.
    """
    for query, code in ((_LOOP_QUERY, "query_syntax"),
                        (_LOOP_QUERY_2, "numeric_only_query")):
        with pytest.raises(ac.CiteRefusal) as caught:
            ac.cite(corpus, query, state=ac.CiteState())
        assert caught.value.code == code
    # And no MATCH expression can be built from them at all.
    for query in (_LOOP_QUERY, _LOOP_QUERY_2):
        with pytest.raises(ac.CiteRefusal):
            ac.build_match_expression(query)


def test_five_thousand_malformed_queries_terminate_at_the_allowance(corpus):
    """`Done when` 1. The loop stops being served on attempt 25.

    Measured: the first ``refusal_cap`` is attempt 25 -- 24 refusals served,
    then the channel closes -- and it stays closed for the remaining 4,976.
    """
    state = ac.CiteState()
    first_terminal = None
    for attempt in range(1, _LOOP_ATTEMPTS + 1):
        with pytest.raises(ac.CiteRefusal) as caught:
            ac.cite(corpus, _LOOP_QUERY, state=state)
        if caught.value.code == "refusal_cap" and first_terminal is None:
            first_terminal = attempt

    assert first_terminal == ac.CITE_CAPS.refusals_per_run + 1 == 25
    assert state.retrieval_channel_closed is True
    assert state.refused_calls == _LOOP_ATTEMPTS
    assert state.calls == 0
    # The exact string, because a typed refusal whose text drifts is not typed.
    assert caught.value.reason == (
        "corpus.cite.refusal_cap: this run has made 24 refused cite() calls, "
        "the refusal allowance; the retrieval channel is closed for the rest "
        "of this run and no further cite() call will be served")


def test_the_terminal_refusal_is_terminal_for_a_good_query_too(corpus):
    """It closes the CHANNEL, not just the malformed path.

    A cap that only stopped the malformed query would be the same defect the
    issue names one level down: the loop would simply alternate.
    """
    state = ac.CiteState()
    for _ in range(ac.CITE_CAPS.refusals_per_run):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY, state=state)
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.cite(corpus, LIVE_QUERIES[0], state=state)
    assert caught.value.code == "refusal_cap"
    assert state.calls == 0


def test_the_refusal_record_is_bounded_regardless_of_attempts(corpus):
    """`Done when` 2. 5,000 attempts, 8 retained entries, ~2 KB of ledger.

    Against 1,211 KB before the bound existed: a 500-fold reduction, and --
    the property that actually matters -- a size that no longer depends on
    how many times the child called.
    """
    state = ac.CiteState()
    for _ in range(_LOOP_ATTEMPTS):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY, state=state)

    ledger = state.citation_ledger()
    assert len(ledger["refusal_codes"]) == ac.CITE_CAPS.refusals_retained == 8
    assert ledger["cite_refusals"] == _LOOP_ATTEMPTS
    size_kb = len(json.dumps(ledger, separators=(",", ":")).encode("utf-8")) / 1024
    assert size_kb < 10, f"citation_ledger is {size_kb:.1f} KB"

    # Ten times the attempts must not make the record bigger.
    ten_x = ac.CiteState()
    for _ in range(_LOOP_ATTEMPTS * 10):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY, state=ten_x)
    bigger = ten_x.citation_ledger()
    assert len(bigger["refusal_codes"]) == len(ledger["refusal_codes"])
    ten_x_kb = len(json.dumps(bigger, separators=(",", ":")).encode("utf-8")) / 1024
    assert ten_x_kb - size_kb < 0.1, (
        f"{ten_x_kb:.3f} KB at 10x the attempts against {size_kb:.3f} KB -- "
        f"the record is still a function of child behaviour")


def test_nothing_is_dropped_silently(corpus):
    """`Done when` 2, the honesty half.

    A truncated list with no count reads identically to a run that made four
    typos. The record has to say a loop happened.
    """
    state = ac.CiteState()
    for _ in range(30):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY, state=state)
    for _ in range(20):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY_2, state=state)

    ledger = state.citation_ledger()
    retained = len(ledger["refusal_codes"])
    assert retained == 8
    assert ledger["refusal_codes_dropped"] == 50 - retained == 42
    assert retained + ledger["refusal_codes_dropped"] == ledger["cite_refusals"]
    # The aggregate stays COMPLETE while the detail is capped.
    assert sum(ledger["refusal_code_counts"].values()) == ledger["cite_refusals"]
    assert ledger["refusal_code_counts"] == {"query_syntax": 24,
                                             "refusal_cap": 26}
    assert ledger["retrieval_channel_closed"] is True


def test_the_per_code_tally_is_keyed_by_this_modules_own_vocabulary(corpus):
    """The tally is bounded because the child cannot mint a key for it.

    Every ``CiteRefusal`` code is a string literal in `analyst_corpus`, so the
    dict has a fixed maximum size no matter what the child sends. If a future
    refusal ever interpolated child text into ``code``, this fails.
    """
    source = Path(ac.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    literal_codes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "CiteRefusal" and node.args:
            first = node.args[0]
            assert isinstance(first, ast.Constant) and isinstance(first.value, str), (
                "a CiteRefusal code must be a literal; a computed code would "
                "make refusal_code_counts child-controlled")
            literal_codes.add(first.value)
    assert len(literal_codes) < 40

    state = ac.CiteState()
    for query in ("sleep* OR injur*", "1.6", "", "the of a and", "\x00"):
        for _ in range(50):
            with pytest.raises(ac.CiteRefusal):
                ac.cite(corpus, query, state=state)
    assert set(state.refusal_counts) <= literal_codes


def test_one_retained_refusal_cannot_hold_an_unbounded_query(corpus):
    """The hole the list bound alone does not close.

    ``query_too_long`` refuses a query for being over 200 characters and is
    minted with the whole string still in hand. Retaining 8 entries verbatim
    therefore retained 8 arbitrarily large strings: measured at 8 calls with a
    1.2 MB query, the pre-bound record was **9,377 KB** -- well inside any
    refusal allowance, so the allowance would not have caught it either.
    """
    big = "sleep " * 200_000
    state = ac.CiteState()
    for _ in range(8):
        with pytest.raises(ac.CiteRefusal) as caught:
            ac.cite(corpus, big, state=state)
        assert caught.value.code == "query_too_long"

    ledger = state.citation_ledger()
    size_kb = len(json.dumps(ledger, separators=(",", ":")).encode("utf-8")) / 1024
    assert size_kb < 10, f"{size_kb:.1f} KB retained from 8 calls"
    entry = ledger["refusal_codes"][0]
    assert len(entry["query"]) == ac.CITE_CAPS.query_chars == 200
    assert entry["query_chars_dropped"] == len(big) - 200
    assert entry["query"] == big[:200], "the retained prefix is verbatim"


def test_a_single_typo_does_not_burn_the_run(corpus):
    """`Done when` 3. The recoverable case, which is the point of the design.

    One malformed query, one typed refusal, one corrected query, five
    passages -- and the call budget untouched by the mistake.
    """
    state = ac.CiteState()
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.cite(corpus, "sleep injury*", state=state)
    assert caught.value.code == "query_syntax"
    assert state.calls == 0, "a typo must not consume a cite() call"

    passages = ac.cite(corpus, "sleep injury", state=state)
    assert len(passages) == 5
    assert state.calls == 1
    assert state.refused_calls == 1
    # And all 8 calls remain available after the typo.
    roomy = ac.CiteState(caps=replace(ac.CITE_CAPS, evidence_bytes=10 ** 9,
                                      docs_per_run=10 ** 6))
    with pytest.raises(ac.CiteRefusal):
        ac.cite(corpus, "sleep injury*", state=roomy)
    for index in range(ac.CITE_CAPS.calls_per_run):
        assert ac.cite(corpus, LIVE_QUERIES[index], state=roomy)
    assert roomy.calls == 8


def test_twenty_three_malformed_queries_still_permit_a_successful_cite(corpus):
    """`Done when` 3, at the boundary -- the allowance is generous by design.

    23 refusals leave the channel open; the 24th closes it, so 23 is the most
    a child may waste and still retrieve. That is ~3 failed attempts for every
    call the run is allowed, which is a loop's worth of slack and nowhere near
    a mistake's worth of punishment.
    """
    state = ac.CiteState()
    for _ in range(23):
        with pytest.raises(ac.CiteRefusal) as caught:
            ac.cite(corpus, _LOOP_QUERY, state=state)
        assert caught.value.code == "query_syntax"
    assert state.retrieval_channel_closed is False
    assert ac.cite(corpus, LIVE_QUERIES[0], state=state), (
        "23 malformed queries must not cost the run its retrieval")

    # One more refusal and it is gone -- the boundary is exact, not fuzzy.
    one_more = ac.CiteState()
    for _ in range(24):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY, state=one_more)
    assert one_more.retrieval_channel_closed is True
    with pytest.raises(ac.CiteRefusal) as caught:
        ac.cite(corpus, LIVE_QUERIES[0], state=one_more)
    assert caught.value.code == "refusal_cap"


def test_the_vault_ledger_is_untouched_by_five_thousand_cite_refusals(corpus):
    """`Done when` 4. The #229-adjacent separation survives the fix.

    The refusal accounting added for #232 lives entirely in `CiteState`. It
    must not have grown a path into the vault's accumulator -- a corpus read
    (or a corpus REFUSAL) satisfying `analyst_envelope`'s zero-read gate is
    exactly the two-database bypass the `CiteState` comment exists to stop.
    """
    vault_state = analyst_ledger._LedgerState()
    empty = {"query_count": 0, "rows_read": 0,
             "tables_read": [], "columns_read": []}
    assert vault_state.summary().as_dict() == empty

    state = ac.CiteState()
    for _ in range(_LOOP_ATTEMPTS):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY, state=state)
    assert state.refused_calls == _LOOP_ATTEMPTS

    assert vault_state.summary().as_dict() == empty
    citation = state.citation_ledger()
    assert VAULT_LEDGER_KEYS.isdisjoint(citation), (
        "a key added for #232 collides with the vault ledger")
    # Not one of the new keys is a synonym for a vault field either.
    for key in ("refusal_codes_dropped", "refusal_code_counts",
                "retrieval_channel_closed"):
        assert key in citation
        assert "read" not in key and "quer" not in key.replace("queries", "")


def test_the_new_caps_are_frozen_and_unreachable_from_the_child(corpus):
    """A cap the model's code can reach is not a cap (proposal S4.2)."""
    for name in ("refusals_per_run", "refusals_retained"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(ac.CITE_CAPS, name, 10 ** 9)
    assert ac.CITE_CAPS.refusals_per_run == 24
    assert ac.CITE_CAPS.refusals_retained == 8
    assert ac.CITE_CAPS.refusals_per_run == 3 * ac.CITE_CAPS.calls_per_run
    assert ac.CITE_CAPS.refusals_retained == ac.CITE_CAPS.calls_per_run
    # And the child-side snippet cannot name the state, the caps, or a way to
    # reset either: it sends {"op": "cite", ...} and reads a response.
    for forbidden in ("CiteState", "CiteCaps", "refusals_per_run",
                      "refusals_retained", "refusal_counts", "refusals_dropped",
                      "retrieval_channel_closed"):
        assert forbidden not in ac.CHILD_CITE_SNIPPET


def test_serve_cite_returns_the_terminal_refusal_as_a_response(corpus):
    """The framed path refuses terminally too, and still never raises.

    ``serve_cite_frame`` keeps returning ``cap_reason=None``: #232 closes the
    retrieval channel, not the run. Killing the run would make a
    malformed-query loop a way to destroy an analysis rather than a way to
    lose retrieval, and the memory bound is enforced in `CiteState` regardless.
    """
    state = ac.CiteState()
    for _ in range(ac.CITE_CAPS.refusals_per_run):
        response = ac.serve_cite({"op": "cite", "query": _LOOP_QUERY}, corpus, state)
        assert response["ok"] is False
    terminal = ac.serve_cite({"op": "cite", "query": LIVE_QUERIES[0]},
                             corpus, state)
    assert terminal == {"ok": False, "error_type": "CiteRefusal",
                        "code": "refusal_cap", "error": terminal["error"]}
    assert terminal["error"].startswith("corpus.cite.refusal_cap: ")

    sent = []
    more, cap_reason = ac.serve_cite_frame(
        {"op": "cite", "query": LIVE_QUERIES[0]}, sent.append, corpus, state)
    assert (more, cap_reason) == (True, None)
    assert sent[0]["code"] == "refusal_cap"


def test_the_loop_is_cheaper_after_the_bound_not_merely_smaller(corpus):
    """`Done when` 5, as a property rather than a wall-clock number.

    Once the channel is closed no child-supplied data is inspected at all, so
    the per-attempt cost falls. The risk this rules out is the opposite
    outcome -- a bound that makes the loop cheap to run and expensive to
    serialise -- so the serialisation is timed too.
    """
    state = ac.CiteState()
    for _ in range(_LOOP_ATTEMPTS):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY, state=state)

    started = time.perf_counter()
    for _ in range(_LOOP_ATTEMPTS):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY, state=state)
    closed_channel_s = time.perf_counter() - started

    fresh = ac.CiteState(caps=replace(ac.CITE_CAPS,
                                      refusals_per_run=10 ** 9,
                                      refusals_retained=10 ** 9))
    started = time.perf_counter()
    for _ in range(_LOOP_ATTEMPTS):
        with pytest.raises(ac.CiteRefusal):
            ac.cite(corpus, _LOOP_QUERY, state=fresh)
    unbounded_s = time.perf_counter() - started

    assert closed_channel_s < unbounded_s, (
        f"closed {closed_channel_s:.4f}s vs unbounded {unbounded_s:.4f}s")

    started = time.perf_counter()
    blob = json.dumps(state.citation_ledger(), separators=(",", ":"))
    bounded_serialise_s = time.perf_counter() - started
    started = time.perf_counter()
    unbounded_blob = json.dumps(fresh.citation_ledger(), separators=(",", ":"))
    unbounded_serialise_s = time.perf_counter() - started
    assert bounded_serialise_s < unbounded_serialise_s
    assert len(blob) * 100 < len(unbounded_blob), (
        f"{len(blob)} bytes against {len(unbounded_blob)}")
