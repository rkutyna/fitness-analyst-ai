"""analyst_corpus.py -- ``cite()``, the parent-side corpus retrieval interface.

The one rule, transposed again (``corpus_build.py`` docstring, design S0.2):

    The corpus owns the evidence. The model retrieves and cites; it never
    states a finding from memory, and it never types the quotation.

``cite()`` is bound into the analyst run **parent-side, exactly like ``conn``**.
The child gets a function over the existing framed query channel
(``analyst_runner._QueryProxy``); it never gets a corpus connection, a cursor,
or a path. Three reasons, each a mechanism rather than a preference:

1. **A raw handle allows ``SELECT body FROM chunks``** -- the whole corpus into
   the envelope and out to the model provider, defeating every cap in one
   statement. A function that only ever returns ``snippet()`` spans under a
   byte budget cannot be made to do that.
2. **A child-side handle makes retrieval provenance forgeable.** This is #226's
   shape applied to the corpus: a child holding the connection could claim it
   retrieved a passage it invented. Parent-side retrieval means the parent
   *observed* which queries ran and which chunks came back --
   ``CiteState.citation_ledger()`` is that observation, not the child's report.
3. **FTS5 syntax errors must become typed refusals, not tracebacks.** Tracebacks
   are treated as a data channel in this codebase (proposal S5.3,
   ``analyst_runner._reduce_diagnostic``). Every failure below raises
   `CiteRefusal`, which carries a stable ``code`` and a one-line ``reason`` and
   nothing else.

Wiring this in -- what the parallel session that owns ``analyst_runner.py``
must add. Nothing here edits that file.

**One line**, inside ``_service_query``'s ``try:`` block, immediately before
``sql = request["sql"]``::

    if request.get("op") == "cite": return analyst_corpus.serve_cite_frame(request, lambda p: _send_frame(sock, p), corpus_conn, cite_state)

`serve_cite_frame` returns the same ``(more, cap_reason)`` tuple
``_service_query`` returns, so the line is a drop-in. `serve_cite` is the pure
half (decoded request in, JSON-safe dict out) for a caller that prefers to do
its own framing.

**One line**, to give the child the name -- wrap the return of
``_runner_source``::

    return analyst_corpus.child_source_with_cite(RUNNER_TEMPLATE.replace(...))

`child_source_with_cite` splices `CHILD_CITE_SNIPPET` into an already-built
runner source and extends the ``_globals`` mapping. It raises if either anchor
has moved, so a template drift fails loudly instead of silently shipping a
child with no ``cite``.

The remaining plumbing -- opening the corpus (`open_corpus`), constructing a
`CiteState` per run, and carrying both into ``_service_query`` -- belongs to
that file and is deliberately not attempted here.

On measurement. Everything asserted below was measured on 2026-08-30 against
the real corpus at ``data/corpus/corpus.db`` (45 docs, 1,923 chunks,
SQLite 3.51.0); the numbers are in the module's tests rather than in prose that
can go stale.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CHILD_CITE_SNIPPET",
    "CITE_CAPS",
    "DEFAULT_SNIPPET_TOKENS",
    "CiteCaps",
    "CiteRefusal",
    "CiteState",
    "Passage",
    "build_match_expression",
    "child_source_with_cite",
    "cite",
    "normalize_span",
    "open_corpus",
    "serve_cite",
    "serve_cite_frame",
    "span_is_verbatim",
]


# --------------------------------------------------------------------------- #
# Caps -- design S2.2, asserted here as constants rather than argued about.
# Every one is enforced in THIS process, against state the child cannot reach.
# A cap the model's code can reach is not a cap (proposal S4.2).
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CiteCaps:
    """Design S2.2's table. Frozen: a run may not raise its own ceiling.

    There is deliberately **no score cap of any kind** on this object, and
    there is no parameter anywhere in this module that accepts one. See
    `cite` for the measurement that rules it out.
    """

    passages_per_call: int = 5
    calls_per_run: int = 8
    span_tokens: int = 64
    docs_per_run: int = 10
    evidence_bytes: int = 16_384
    wall_clock_s: float = 1.0
    query_chars: int = 200
    # -- #232. The two caps below bound REFUSED calls. They exist because
    # ``calls_per_run`` does not: validation happens before any ``MATCH``, so a
    # refused call costs the child one function call and costs the PARENT one
    # allocation, and a child that only ever sends malformed queries never
    # touches the 8-call budget at all. Measured before this pair existed:
    # 5,000 malformed queries in one run gave ``cite_calls=0``,
    # ``cite_refusals=5000``, and a 1,167 KB ``citation_ledger``.
    #
    # ``refusals_per_run = 3 * calls_per_run``, and that ratio is the whole
    # argument. The failure mode being bounded is a LOOP, not a MISTAKE. A
    # model that typos, reads the typed refusal and corrects itself is doing
    # exactly what the typed-refusal design (module docstring, reason 3) exists
    # to enable -- so a refusal must NOT be charged against ``calls_per_run``
    # one for one, which would let one legitimate typo cost a citation. Three
    # refusals per permitted call means every one of the 8 calls may be
    # preceded by two failed attempts and still land. Nothing short of a loop
    # reaches 24; a loop reaches it on attempt 25, and there the channel closes.
    refusals_per_run: int = 24
    # ``refusals_retained = calls_per_run``: the record keeps the first 8
    # refusals in full and counts the rest by code. Eight is enough to show a
    # human what a run whose entire budget is 8 calls was failing at; the
    # twentieth identical ``query_syntax`` refusal carries nothing the first
    # did not. Nothing is dropped silently -- ``citation_ledger`` reports both
    # how many were dropped and a per-code tally, and that tally is bounded by
    # this module's own fixed code vocabulary rather than by child behaviour.
    refusals_retained: int = 8


CITE_CAPS = CiteCaps()

# FTS5's own documented ceiling on ``snippet()``'s token count, not one we
# chose -- design S2.2 calls this "the cheapest kind of cap to defend".
#
# MEASURED CORRECTION, 2026-08-30, SQLite 3.51.0: the ceiling is documented but
# NOT enforced. ``snippet(..., 100)`` returned 666 characters where
# ``snippet(..., 64)`` returned 438; nothing raised and nothing clamped. So the
# cap is defensible but it is ours to apply, and `cite` applies it by never
# passing anything but this constant.
DEFAULT_SNIPPET_TOKENS = 64


# --------------------------------------------------------------------------- #
# Query sanitisation -- the harness owns query syntax; the model supplies terms
# --------------------------------------------------------------------------- #

# Characters a search *term* may contain. An allowlist, not a denylist: a new
# FTS5 metacharacter in some future version is then refused by default rather
# than admitted by default.
#
# ``-`` and ``'`` are here because they occur inside real terms
# (``high-intensity``, ``athlete's``) and are harmless once the term is quoted
# -- measured: ``"high-intensity"`` and ``"don't"`` both match. ``.`` is here
# because ``1.6`` must reach FTS5 as a quoted phrase rather than as the bare
# token that raises ``fts5: syntax error near "."``.
#
# Deliberately absent, each because FTS5 gives it a meaning the model must not
# be able to reach: ``"`` (string), ``*`` (prefix / special query), ``(``/``)``
# (grouping), ``:`` (column filter), ``^`` (initial-token), ``{``/``}`` (column
# lists). ``;`` is absent because it is SQL's statement separator and appears
# in no search term.
_ALLOWED_EXTRA = "-'’.,%/"

# FTS5's operator keywords are uppercase-only in its grammar, so this strips
# exactly what FTS5 would treat as syntax and leaves the ordinary English words
# ``and``/``or``/``not`` alone.
_FTS5_OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})

# Used ONLY for the degenerate-query refusal below -- never to drop a term from
# the search itself. Issue #22 in this repository (no bm25 threshold separates
# relevant from irrelevant retrieval) carries the measurement: "OR of all terms"
# at 81% top-5 hit rate and every narrower strategy far
# worse, so the executed query is that strategy with nothing removed. A query
# that consists of nothing BUT these is not a narrower strategy; it is a query
# with no content word in it at all.
_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have how i if in into is it its me my no nor not of on or our so such than that
the their them then there these they this those to too was we were what when
where which while who whom why will with would you your
""".split())


class CiteRefusal(Exception):
    """A typed refusal from the retrieval interface.

    Carries a stable machine-readable ``code`` and a one-line ``reason``, the
    same shape as `corpus_build.RegistryRefusal`. It is an exception so a
    caller cannot continue past one by ignoring a return value, and ``reason``
    is the only thing that reaches the child -- never a SQLite message, never a
    fragment of the corpus, never a traceback.
    """

    def __init__(self, code: str, detail: str):
        self.code = code
        self.reason = f"corpus.cite.{code}: {detail}"
        super().__init__(self.reason)


def _describe_char(ch: str) -> str:
    """A refusal-safe description of one rejected character."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "unnamed"
    return f"U+{ord(ch):04X} ({name})"


def _terms(query, caps: CiteCaps) -> list[str]:
    """Every check that can refuse a query, in a fixed order.

    Fixed order because a query with two defects must always refuse on the same
    one -- ``corpus_build.validate_entry`` makes the same argument: a refusal
    reason that varies run to run is not a typed string, it is a coin flip.
    """
    if not isinstance(query, str):
        raise CiteRefusal(
            "bad_query_type",
            f"query must be a string, not {type(query).__name__}; cite() takes "
            f"search terms, and only a string can carry them")

    for ch in query:
        if ch in "\t\n\r":
            continue
        if unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn"):
            raise CiteRefusal(
                "control_character",
                f"query contains the control character {_describe_char(ch)}; "
                f"a search term is printable text")

    if len(query) > caps.query_chars:
        raise CiteRefusal(
            "query_too_long",
            f"query is {len(query)} characters, over the {caps.query_chars}-"
            f"character cap; a retrieval query is a handful of terms")

    if not query.strip():
        raise CiteRefusal(
            "empty_query",
            "query is empty; there is nothing to retrieve against")

    for ch in query:
        if ch.isspace() or ch.isalnum() or ch == "_" or ch in _ALLOWED_EXTRA:
            continue
        raise CiteRefusal(
            "query_syntax",
            f"query contains {_describe_char(ch)}, which is query syntax "
            f"rather than a search term; cite() escapes and quotes the terms "
            f"itself, so supply words and numbers only")

    raw = [tok for tok in query.split() if tok not in _FTS5_OPERATORS]
    # A token with no alphanumeric character produces no FTS5 token at all
    # (``"--"`` is a zero-token phrase); keeping it would put an empty phrase
    # into the expression for no benefit.
    terms = [tok for tok in raw if any(c.isalnum() for c in tok)]
    if not terms:
        raise CiteRefusal(
            "no_searchable_terms",
            "query reduces to no searchable term once FTS5 operators and "
            "punctuation are removed")

    if not any(any(c.isalpha() for c in tok) for tok in terms):
        raise CiteRefusal(
            "numeric_only_query",
            "query has no term containing a letter; the corpus is asked for "
            "claims and context, and every figure comes from the vault")

    if all(tok.strip(_ALLOWED_EXTRA).lower() in _STOPWORDS for tok in terms):
        raise CiteRefusal(
            "stopword_only_query",
            "query contains no content word; a retrieval ranked on function "
            "words alone is a ranking of nothing")

    return terms


def build_match_expression(query, *, caps: CiteCaps = CITE_CAPS) -> str:
    """The FTS5 MATCH expression for ``query``: every term quoted, OR-ed.

    OR because it is what was measured. Issue #22 in this repository (no bm25
    threshold separates relevant from irrelevant retrieval) carries the
    measurement: OR-of-all-terms retrieves 81% top-5 against the real corpus;
    AND of the two rarest present terms collapses that to 19% and AND of three
    to 6%. Both AND strategies were rejected there and are not offered here.

    Every term is wrapped in FTS5 double quotes with any embedded quote
    doubled. The escape is belt-and-braces -- `_terms` has already refused a
    query containing ``"`` -- and it stays because the quoting is what turns
    the measured ``fts5: syntax error near "."`` on a bare ``1.6`` into an
    ordinary phrase match.
    """
    return " OR ".join(
        '"' + term.replace('"', '""') + '"' for term in _terms(query, caps))


# --------------------------------------------------------------------------- #
# The parent-observed retrieval record -- NOT the vault ledger
# --------------------------------------------------------------------------- #

@dataclass
class CiteState:
    """One run's corpus-retrieval accumulator, and its caps.

    ###################################################################
    # THIS IS A SEPARATE NAMESPACE FROM THE VAULT LEDGER, ON PURPOSE. #
    ###################################################################

    ``analyst_envelope._validate`` carries a *zero-read gate*: an envelope
    holding any numeric cell is refused when the vault ledger shows
    ``query_count == 0``, ``rows_read == 0``, or no vault table read. It exists
    to stop a model inventing figures without touching the vault.

    That gate asks *"was a database read?"* while meaning *"was the VAULT
    read?"*. Adding a second database to the process is exactly what makes the
    ambiguity load-bearing. If corpus reads accumulated into the vault ledger,
    this would pass::

        cite(query="anything")   # a corpus read; the vault ledger is now non-empty
        emit("weekly", cols, units, [[fabricated numbers]])

    -- the child never touched the vault, invented every figure, and the gate
    is satisfied.

    So: this object never writes into, extends, or is merged with
    `analyst_ledger.LedgerSummary`. This module imports nothing from
    `analyst_ledger`, holds no vault connection, and its `citation_ledger`
    keys are chosen to be **disjoint** from that summary's
    (``query_count``/``rows_read``/``tables_read``/``columns_read``) so that a
    ``dict.update`` of one over the other cannot silently satisfy the gate
    either. There is no function, parameter, or convenience here that merges
    them; a caller wanting both takes both objects.

    **The failure mode this comment exists to stop is a future tidy-up** that
    notices two accumulators which look redundant and merges them. They are not
    redundant. They answer different questions, and one of them is a security
    gate.
    """

    caps: CiteCaps = CITE_CAPS
    calls: int = 0
    refused_calls: int = 0
    passages_returned: int = 0
    evidence_bytes: int = 0
    # Ordered and deduplicated by insertion, so the record is deterministic
    # without depending on set iteration order.
    docs_cited: dict[str, None] = field(default_factory=dict)
    queries: list[dict] = field(default_factory=list)
    chunks_returned: list[list] = field(default_factory=list)
    # #232: BOUNDED at ``caps.refusals_retained`` entries. The three fields
    # below are the whole retained record of refusal, and none of their sizes
    # is a function of how many times the child called ``cite()``:
    # ``refusals`` stops growing at the cap, ``refusals_dropped`` is one int,
    # and ``refusal_counts`` is keyed by this module's own fixed vocabulary of
    # ``CiteRefusal`` codes -- every one of which is a literal in this file, so
    # the child cannot mint a new key.
    refusals: list[dict] = field(default_factory=list)
    refusals_dropped: int = 0
    refusal_counts: dict[str, int] = field(default_factory=dict)

    @property
    def retrieval_channel_closed(self) -> bool:
        """Whether this run has spent its refusal allowance (#232).

        Once true it stays true and every subsequent ``cite()`` -- well-formed
        or not -- is refused with ``refusal_cap``. That is the termination
        property: an unbounded refusal loop stops being served rather than
        being served ever more cheaply.
        """
        return self.refused_calls >= self.caps.refusals_per_run

    def citation_ledger(self) -> dict:
        """The parent's own record of what retrieval did this run.

        ``parent_observed`` is true in the same sense
        ``analyst_runner`` means it: every field here was written by THIS
        process while servicing a request, from the rows THIS process got back.
        Nothing in it depends on what the child said it did.
        """
        return {
            "cite_calls": self.calls,
            "cite_refusals": self.refused_calls,
            "passages_returned": self.passages_returned,
            "evidence_bytes": self.evidence_bytes,
            "docs_cited": list(self.docs_cited),
            "queries": [dict(q) for q in self.queries],
            "chunks_returned": [list(c) for c in self.chunks_returned],
            "refusal_codes": [dict(r) for r in self.refusals],
            # #232: the record must SAY it dropped things. A truncated list
            # with no count is a record that lies about a loop by omission --
            # it reads identically to a run that made four typos. These two
            # keys are what make the bound honest, and they are also what a
            # reviewer reads to tell "the model kept correcting itself" from
            # "the model was in a loop".
            "refusal_codes_dropped": self.refusals_dropped,
            "refusal_code_counts": dict(self.refusal_counts),
            "retrieval_channel_closed": self.retrieval_channel_closed,
            "parent_observed": True,
        }

    def _record_refusal(self, query, exc: "CiteRefusal") -> None:
        """Record one refusal without letting the record grow without bound.

        The per-code tally is incremented for EVERY refusal, retained or not,
        so ``sum(refusal_counts.values()) == refused_calls`` always holds and
        the aggregate stays complete while the detail is capped.
        """
        self.refused_calls += 1
        self.refusal_counts[exc.code] = self.refusal_counts.get(exc.code, 0) + 1
        if len(self.refusals) >= self.caps.refusals_retained:
            self.refusals_dropped += 1
            return
        record = {"code": exc.code, "reason": exc.reason, "query": None,
                  "query_chars_dropped": 0}
        if isinstance(query, str):
            # A retained entry is bounded too. ``query_too_long`` refuses a
            # query for being over ``query_chars`` and is minted with the whole
            # string still in hand, so storing it verbatim would let 8 retained
            # entries hold an arbitrary number of bytes -- the same defect the
            # list bound fixes, one level down.
            record["query"] = query[:self.caps.query_chars]
            record["query_chars_dropped"] = max(
                0, len(query) - self.caps.query_chars)
        self.refusals.append(record)


@dataclass(frozen=True)
class Passage:
    """One retrieved passage. Design S2.1's shape, verbatim.

    ``score`` is bm25 and is present because ordering and the run record both
    want it. It is **returned, never filtered on** -- see `cite`.
    """

    doc_id: str
    chunk_ix: int
    span: str
    score: float
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    pmid: str | None = None
    license: str | None = None

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "chunk_ix": self.chunk_ix,
            "span": self.span, "score": self.score, "title": self.title,
            "authors": self.authors, "year": self.year, "doi": self.doi,
            "pmid": self.pmid, "license": self.license,
        }


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

# The INNER JOIN is the whole orphan defence -- see the comment in `cite`.
#
# ``snippet()`` is called with EMPTY start/end markers and an EMPTY ellipsis.
# Design S2.3's example used ``'['``/``']'``/``'...'``, but S3.3 resolves a
# citation by exact substring match against the corpus body, and any marker or
# ellipsis is text the corpus body does not contain. With all three empty the
# span is a verbatim substring of ``body`` and the resolver is a single ``in``
# -- measured 200/200 on the real corpus, where the dotted form needs a strip
# that is itself a guess about which dots were the corpus's own.
#
# Aux functions take the FTS5 table's own name; an alias fails with
# ``no such column: c`` (measured), so ``chunks`` is spelled out.
_SELECT = """
SELECT chunks.doc_id, chunks.chunk_ix,
       snippet(chunks, 2, '', '', '', ?),
       bm25(chunks),
       d.title, d.authors, d.year, d.doi, d.pmid, d.license
  FROM chunks JOIN docs d ON d.doc_id = chunks.doc_id
 WHERE chunks MATCH ?
 ORDER BY bm25(chunks) ASC
 LIMIT ?
"""

_ORPHAN_PROBE = "SELECT 1 FROM chunks WHERE doc_id = ? LIMIT 1"
_DOC_PROBE = "SELECT 1 FROM docs WHERE doc_id = ? LIMIT 1"


def open_corpus(corpus_path: str | Path) -> sqlite3.Connection:
    """Open a corpus read-only. The path is passed in, never defaulted.

    No default argument, no environment variable, no module-level constant
    holding a path -- the repo's T-003 rule. A caller that does not know which
    corpus it means does not get one.
    """
    if not isinstance(corpus_path, (str, Path)) or not str(corpus_path):
        raise CiteRefusal("no_corpus_path",
                          "a corpus path must be supplied explicitly")
    path = Path(corpus_path)
    if not path.exists():
        raise CiteRefusal("no_such_corpus",
                          f"no corpus file at {path.name!r}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _guard_doc_id(corpus_conn: sqlite3.Connection, doc_id) -> str:
    """Refuse a ``doc_id`` that names an orphan, or nothing at all.

    **This is the SECONDARY orphan mechanism, and it is secondary on purpose.**
    The primary one is the ``JOIN docs`` in `_SELECT`, which makes an orphan
    chunk *unreachable* rather than merely unrequestable. Refusing only a
    *named* orphan would leave it reachable by every path that does not name
    it -- which is how #229 came to exist in the first place: FTS5 virtual
    tables cannot carry a foreign key, so a chunk can outlive its ``docs`` row,
    is returned by ``MATCH``, and its span is genuinely verbatim. It passes
    every check except a join. The join must be the mechanism; this refusal
    exists only so that naming one gets an honest answer instead of an empty
    list.
    """
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise CiteRefusal("bad_doc_id",
                          f"doc_id must be a non-empty string, not "
                          f"{type(doc_id).__name__}")
    doc_id = doc_id.strip()
    if corpus_conn.execute(_DOC_PROBE, (doc_id,)).fetchone() is not None:
        return doc_id
    if corpus_conn.execute(_ORPHAN_PROBE, (doc_id,)).fetchone() is not None:
        raise CiteRefusal(
            "orphan_doc",
            f"doc_id {doc_id!r} has indexed chunks but no docs row, so nothing "
            f"about it is vetted -- no title, no license, no approver. It is "
            f"unreachable through retrieval and it is not citable by name")
    raise CiteRefusal("unknown_doc",
                      f"doc_id {doc_id!r} is not in this corpus")


def cite(corpus_conn: sqlite3.Connection, query: str, k: int = 5, *,
         state: CiteState, doc_id: str | None = None) -> list[Passage]:
    """Retrieve up to ``k`` passages for ``query``. Parent-side. No thresholds.

    **There is no score threshold and no parameter that accepts one**, and this
    is the single most important property of this signature. Measured over 16
    real coaching questions against the 45-document corpus plus a 20-document
    null control of plant genomics and materials science
    (issue #22 in this repository, which carries the measurement): all 16 questions
    retrieved passages from the irrelevant corpus, and **12 of 16 of those
    outscored the worst genuinely relevant hit** -- best null bm25 -14.08
    against worst true -6.82. No absolute bm25 value separates relevant from
    irrelevant. A ``min_score`` would look like a quality gate and would not be
    one; the mitigation that survives measurement is showing the span to a
    human (S6.3), which is why the span is returned and the score is not acted
    on.

    ``bm25()`` in SQLite is NEGATIVE and more negative is better, so `_SELECT`
    sorts ASC and ``result[0]`` is the best match. Reversing that silently
    returns the *worst* matches while looking correct;
    ``test_reversed_sort_would_be_caught`` is the assertion that fails if it
    is flipped.

    ``k`` is clamped to `CiteCaps.passages_per_call`, not refused -- a model
    asking for 50 gets 5, which is the cap doing its job rather than an error.
    Everything else that a run can exhaust is a refusal, in full, on the
    ``analyst_envelope`` principle that a half-state is worse than none.
    """
    caps = state.caps

    # #232, and it is FIRST on purpose: this is the one refusal that must not
    # depend on anything about the request. Every check below it -- k, the call
    # cap, `build_match_expression` -- inspects child-supplied data, and the
    # point of the allowance is that once it is spent no child-supplied data is
    # looked at again. It is also the only TERMINAL refusal: a well-formed
    # query arriving after this point is refused too, because what is being
    # bounded is the loop and not the query.
    if state.retrieval_channel_closed:
        raise _refuse(state, query, CiteRefusal(
            "refusal_cap",
            f"this run has made {caps.refusals_per_run} refused cite() calls, "
            f"the refusal allowance; the retrieval channel is closed for the "
            f"rest of this run and no further cite() call will be served"))

    if isinstance(k, bool) or not isinstance(k, int):
        raise _refuse(state, query, CiteRefusal(
            "bad_k", f"k must be an integer, not {type(k).__name__}"))
    if k < 1:
        raise _refuse(state, query, CiteRefusal(
            "bad_k", f"k is {k}; a retrieval returns at least one passage"))
    k = min(k, caps.passages_per_call)

    if state.calls >= caps.calls_per_run:
        raise _refuse(state, query, CiteRefusal(
            "call_cap",
            f"this run has already made {state.calls} cite() calls, the cap "
            f"({caps.calls_per_run}); {caps.calls_per_run * caps.passages_per_call} "
            f"passages is the most a single finding can honestly rest on"))

    try:
        expression = build_match_expression(query, caps=caps)
        target = _guard_doc_id(corpus_conn, doc_id) if doc_id is not None else None
    except CiteRefusal as exc:
        raise _refuse(state, query, exc) from None

    sql, params = _SELECT, [DEFAULT_SNIPPET_TOKENS, expression, k]
    if target is not None:
        # Filtering inside the FTS5 scan, so the doc filter cannot be used to
        # walk the corpus: it still goes through MATCH, snippet() and the caps.
        sql = _SELECT.replace("WHERE chunks MATCH ?",
                              "WHERE chunks MATCH ? AND chunks.doc_id = ?")
        params = [DEFAULT_SNIPPET_TOKENS, expression, target, k]

    rows = _execute_bounded(corpus_conn, sql, params, caps, state, query)

    passages = [
        Passage(doc_id=str(row[0]), chunk_ix=int(row[1]), span=row[2],
                score=float(row[3]), title=row[4], authors=row[5],
                year=row[6], doi=row[7], pmid=row[8], license=row[9])
        for row in rows
    ]

    # Both remaining run-level caps are checked BEFORE any of this call's
    # passages is admitted, so a call either lands whole or does not land.
    payload = [p.as_dict() for p in passages]
    cost = len(json.dumps(payload, separators=(",", ":"),
                          allow_nan=False).encode("utf-8"))
    if state.evidence_bytes + cost > caps.evidence_bytes:
        raise _refuse(state, query, CiteRefusal(
            "evidence_byte_cap",
            f"this call would spend {cost} evidence bytes on top of "
            f"{state.evidence_bytes}, over the {caps.evidence_bytes}-byte "
            f"cap for the run; the evidence budget is separate from the data "
            f"envelope budget and neither may crowd out the other"))

    would_cite = dict(state.docs_cited)
    for passage in passages:
        would_cite[passage.doc_id] = None
    if len(would_cite) > caps.docs_per_run:
        raise _refuse(state, query, CiteRefusal(
            "doc_cap",
            f"this call would bring the run to {len(would_cite)} distinct "
            f"cited documents, over the cap of {caps.docs_per_run}"))

    state.calls += 1
    state.evidence_bytes += cost
    state.passages_returned += len(passages)
    state.docs_cited = would_cite
    state.queries.append({"query": query, "expression": expression,
                          "k": k, "doc_id": target, "returned": len(passages)})
    state.chunks_returned.extend([p.doc_id, p.chunk_ix] for p in passages)
    return passages


def _refuse(state: CiteState, query, exc: CiteRefusal) -> CiteRefusal:
    """Record a refusal in the run's own record and hand it back to be raised.

    A refused call is still something the parent observed, and it counts
    against ``cite_refusals`` and its own allowance -- never against
    ``calls_per_run``. A run cannot be starved of its call budget by writing
    bad queries, and it cannot buy extra budget with them either.

    #232: it does now cost SOMETHING, which it did not before. The allowance
    it spends (`CiteCaps.refusals_per_run`) is separate and larger, so a typo
    still costs a typo rather than a citation, but a loop terminates. And the
    record this writes is bounded -- see `CiteState._record_refusal`.
    """
    state._record_refusal(query, exc)
    return exc


def _execute_bounded(corpus_conn, sql, params, caps: CiteCaps,
                     state: CiteState, query) -> list:
    """Run one retrieval under a real wall-clock bound.

    ``set_progress_handler`` aborts inside SQLite rather than checking the
    clock after the fact, so a pathological query is stopped rather than
    reported. Measured: the handler raises ``OperationalError: interrupted``,
    and clearing it restores the connection (a following ``SELECT count(*)``
    returned 1,923). Normal retrieval is nowhere near this -- median 1.3 ms
    over 60 calls against the real corpus -- exactly as design S2.2 says: the
    cap exists for the pathological query, not the normal one.
    """
    deadline = time.monotonic() + caps.wall_clock_s
    corpus_conn.set_progress_handler(
        lambda: 1 if time.monotonic() > deadline else 0, 2000)
    try:
        return corpus_conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "interrupt" in str(exc).lower():
            raise _refuse(state, query, CiteRefusal(
                "retrieval_timeout",
                f"retrieval exceeded the {caps.wall_clock_s}s per-call wall "
                f"clock and was aborted")) from None
        # Anything else from FTS5 is a defect in this module's escaping, and it
        # still must not reach the child as a traceback or as a SQLite string:
        # a SQLite message can quote the query and, through it, whatever the
        # child put in the query.
        raise _refuse(state, query, CiteRefusal(
            "retrieval_failed",
            "the corpus refused this retrieval; the query reached FTS5 in a "
            "form it could not parse")) from None
    except sqlite3.DatabaseError:
        raise _refuse(state, query, CiteRefusal(
            "corpus_unreadable",
            "the corpus could not be read")) from None
    finally:
        corpus_conn.set_progress_handler(None, 0)


# --------------------------------------------------------------------------- #
# Span verification -- the mechanism the whole citation design rests on
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def normalize_span(text: str) -> str:
    """Collapse runs of whitespace and strip. The ONLY normalisation allowed.

    Deliberately not case-folding and not stripping punctuation: a citation
    resolves by exact substring match (design S3.3), and every additional
    normalisation is another paraphrase the resolver would accept.
    """
    return _WS.sub(" ", text).strip()


def span_is_verbatim(span: str, body: str) -> bool:
    """Whether ``span`` is text the corpus actually contains.

    Exact substring first, because with empty snippet markers that is what
    ``snippet()`` returns -- measured 200/200 on the real corpus. The
    normalised comparison is the fallback for a body whose whitespace a future
    extractor rewrites, and it is second so that the strict answer is the one
    normally given.
    """
    if span in body:
        return True
    return normalize_span(span) in normalize_span(body)


# --------------------------------------------------------------------------- #
# The dispatch handler -- one line for the runner
# --------------------------------------------------------------------------- #

def serve_cite(payload: dict, corpus_conn, state: CiteState) -> dict:
    """Decode one ``{"op": "cite", ...}`` request into a JSON-safe response.

    Pure in the sense that matters: request dict in, response dict out, no
    socket and no framing. Never raises for a bad request -- a refusal is a
    response, because the alternative is a traceback crossing the boundary.
    """
    if not isinstance(payload, dict):
        return _refusal_response(CiteRefusal(
            "bad_request", "a cite request must be a JSON object"))
    query = payload.get("query")
    if isinstance(query, dict) and "__unencodable__" in query:
        # The child could not JSON-encode what it was handed. It sends the type
        # name instead, so the refusal is still minted HERE rather than in the
        # child, and the parent's record still shows the call.
        return _refusal_response(_refuse(state, None, CiteRefusal(
            "bad_query_type",
            f"query must be a string, not "
            f"{str(query['__unencodable__'])[:40]}; cite() takes search "
            f"terms, and only a string can carry them")))
    k = payload.get("k", CITE_CAPS.passages_per_call)
    try:
        passages = cite(corpus_conn, query, k, state=state,
                        doc_id=payload.get("doc_id"))
    except CiteRefusal as exc:
        return _refusal_response(exc)
    return {"ok": True, "passages": [p.as_dict() for p in passages]}


def _refusal_response(exc: CiteRefusal) -> dict:
    return {"ok": False, "error_type": "CiteRefusal", "code": exc.code,
            "error": exc.reason}


def serve_cite_frame(payload: dict, send, corpus_conn, state: CiteState
                     ) -> tuple[bool, str | None]:
    """`serve_cite`, framed and returning ``_service_query``'s own tuple.

    ``send`` is a one-argument callable taking the response dict -- in the
    runner, ``lambda p: _send_frame(sock, p)``. The second element of the tuple
    is ``_service_query``'s ``cap_reason`` and is always ``None`` here **on
    purpose**: a cite refusal must not abort the whole run the way a vault row
    cap does. The child gets the refusal, can retry with a better query, and
    the parent's record shows both attempts.

    #232 does not change that, including for the terminal ``refusal_cap``.
    What that cap terminates is the RETRIEVAL CHANNEL, not the run: a child
    that has burned its refusal allowance may still have vault work worth
    finishing, and killing the run would make a malformed-query loop a way to
    destroy an analysis rather than merely to lose retrieval. The bound the
    issue asks for is on the parent's memory, and `CiteState._record_refusal`
    is where that is enforced -- returning a non-``None`` ``cap_reason`` here
    would buy nothing for it and would change ``analyst_runner``'s contract.
    """
    send(serve_cite(payload, corpus_conn, state))
    return True, None


# --------------------------------------------------------------------------- #
# The child-side proxy -- a constant to inject, not an edit to the runner
# --------------------------------------------------------------------------- #

# Built the way ``analyst_runner.RUNNER_TEMPLATE`` is built, and it depends on
# that template's helpers: ``_json``, ``_struct``, ``_write_all``,
# ``_read_exact``, ``_QUERY_LOCK``, ``_QUERY_FD``, ``AnalystQueryError``. It is
# spliced in after those exist -- see `child_source_with_cite`.
#
# What the child gets is a function and a namedtuple-ish record. No connection,
# no cursor, no path, and no way to ask for more than the parent will give.
CHILD_CITE_SNIPPET = r'''
class CiteRefusal(AnalystQueryError):
    pass

class Passage:
    __slots__ = ("doc_id", "chunk_ix", "span", "score", "title", "authors",
                 "year", "doi", "pmid", "license")

    def __init__(self, _d):
        for _name in self.__slots__:
            setattr(self, _name, _d.get(_name))

    def __repr__(self):
        return "Passage(%r, %r, %r)" % (self.doc_id, self.chunk_ix,
                                        self.span[:40] if self.span else "")

def cite(query, k=5, doc_id=None):
    _payload = {"op": "cite", "query": query, "k": k, "doc_id": doc_id}
    try:
        _body = _json.dumps(_payload, separators=(",", ":"),
                            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        _payload["query"] = {"__unencodable__": type(query).__name__}
        _body = _json.dumps(_payload, separators=(",", ":"),
                            allow_nan=False).encode("utf-8")
    _frame = _struct.pack("!I", len(_body)) + _body
    with _QUERY_LOCK:
        _write_all(_QUERY_FD, _frame)
        _size = _struct.unpack("!I", _read_exact(_QUERY_FD, 4))[0]
        if _size > 1048576:
            raise AnalystQueryError("cite response is too large")
        _response = _json.loads(_read_exact(_QUERY_FD, _size).decode("utf-8"))
    if not _response.get("ok"):
        raise CiteRefusal(_response.get("error", "cite failed"))
    return [Passage(_p) for _p in _response["passages"]]
'''

# The two anchors in ``analyst_runner.RUNNER_TEMPLATE`` that
# `child_source_with_cite` splices against.
_SPLICE_AFTER = "conn = _QueryProxy(_QUERY_FD)"
_GLOBALS_ANCHOR = '"conn": conn, "emit": emit'

DISPATCH_ONE_LINER = (
    'if request.get("op") == "cite": return analyst_corpus.serve_cite_frame('
    'request, lambda p: _send_frame(sock, p), corpus_conn, cite_state)')


def child_source_with_cite(runner_source: str) -> str:
    """Splice `CHILD_CITE_SNIPPET` into an already-built runner source.

    Raises `CiteRefusal` if either anchor has moved. Loud on drift by design:
    a silent no-op here ships a child with no ``cite`` name, the model's code
    dies on ``NameError``, and the run comes back as an ordinary EXEC_FAILED
    with nothing pointing at the real cause.
    """
    if _SPLICE_AFTER not in runner_source:
        raise CiteRefusal(
            "runner_template_drift",
            f"the runner source has no {_SPLICE_AFTER!r} line to splice after")
    if _GLOBALS_ANCHOR not in runner_source:
        raise CiteRefusal(
            "runner_template_drift",
            f"the runner source has no {_GLOBALS_ANCHOR!r} mapping to extend")
    spliced = runner_source.replace(
        _SPLICE_AFTER, _SPLICE_AFTER + "\n" + CHILD_CITE_SNIPPET, 1)
    # ``CiteRefusal`` goes into the child's globals beside ``cite``: model code
    # that cannot name the exception cannot catch a refusal, and an uncaught
    # refusal ends the run with a traceback -- which is the one outcome this
    # whole module exists to avoid.
    return spliced.replace(
        _GLOBALS_ANCHOR,
        _GLOBALS_ANCHOR + ', "cite": cite, "CiteRefusal": CiteRefusal', 1)
