"""LLM seam + deterministic helpers. run_model is a thin delegator over
health_advisor.llm (direct Ollama /api/chat); the model is only ever a text
transformer here — Python owns the truth. extract_json / grounding_check /
render_fallback are the gates that make a wrong or empty model output safe.
The numeric grounding pattern is shared with the other narration gates.

``grounding_check`` is retained as a legacy, unscoped pre-filter for callers
outside this package. Its result is not a grounding verdict: a nested briefing
is a bag of values with no metric, period, field, unit, or provenance identity.
Publishing paths must use ``deepdive_verify``'s scoped claim resolver instead."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import llm
from .numeric_tokens import NUM_RE as _NUM_RE, strip_invisible

# The judge's pass mark, defined once here rather than in each caller.  This is
# the gates module and both consumers (coach_brief, pipeline) already import it,
# so the name has one definition and one value.  It was 70 in two places (#127
# F-50); two copies of a threshold agree until one of them is tuned.
JUDGE_PASS = 70

# Sampling profile re-exported for judge call sites (temp 0 for determinism).
JUDGE_OPTS = llm.JUDGE_OPTS

# Keep the historical module-level name for callers and tests; the tokenizer
# itself is owned by the shared numeric tokenizer.
# Full ISO dates AND bare year-months. The year-month alternative matters: the
# arc deep dive labels quarters "2026-04", and without it _NUM_RE shredded that
# into the tokens '2026' and '-04', which then failed grounding as two invented
# numbers. Order matters — the 3-part form must be tried first or it is eaten
# as a year-month followed by a stray "-DD".
_DATE_RE = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")
# Prose calendar AND clock references ("July 6–13", "Jul 9, 2026", "July 2026",
# "11:41 PM") — like ISO dates, these locate a point in time; they are not
# quantitative claims. The capable model writes these; qwen never did.
#
# The month-year alternative is tried BEFORE the day form and the day form is
# guarded with (?!\d), because a 1-2 digit match that can sit inside a longer
# digit run half-eats the year: measured live 2026-08-27, "(July 2026)" matched
# as "July 20" and left the token "26", for which the gate then demanded a
# claim. That single artifact produced 4 of the 12-call battery's failures
# while every real figure in those answers verified.
#
# The AM/PM marker on the clock form is MANDATORY and is the whole reason the
# clock form is safe. A bare mm:ss is a pace or a duration — a measurement that
# must keep tokenizing — so only a marked 12-hour time of day is stripped.
#
# THE DAY-LIST TAIL (#187) is the one part of this pattern that makes the gate
# WEAKER, so its boundary is drawn explicitly rather than left to the regex.
# The coach writes several days under one month — "The other four days (Aug 19,
# 21, 22, 23) recorded walking only." — and before this tail only "Aug 19" was
# stripped, leaving 21/22/23 to be graded as unsupported quantitative
# claims and the answer refused for listing dates.
#
# Every token this tail stops grading is a figure the model could then state
# unchecked, so a continuation day is recognised ONLY when all five hold. Each
# one exists to keep a real figure out:
#
#  1. SEPARATOR. It follows the previous day across a comma, "and", "&", or
#     ", and" — the punctuation of a list. Bare whitespace is NOT a separator
#     ("Aug 19 30 minutes" is prose, not a list).
#  2. DAY-SHAPED VALUE, 1..31 inclusive, enumerated (`0?[1-9]|[12]\d|3[01]`)
#     rather than matched as `\d{1,2}`. This alone retires the issue's own
#     counter-example class: "On Aug 19, 44 jog minutes" keeps grading 44,
#     because 44 is not a day. Values 1..31 remain ambiguous, which is what
#     guards 3-5 are for.
#  3. NO DECIMAL, THOUSANDS SEPARATOR, SLASH, OR LONGER DIGIT RUN —
#     `(?!\d)(?![.,/]\d)`, the same guard `_SCALE_DENOM_RE` uses. "Aug 19,
#     44.0" and "Aug 19, 12,500 steps" must not have their leading digits
#     eaten; half-eating a figure is worse than not stripping it, because it
#     fabricates a different number (12,500 -> 500).
#  4. THE WHOLE RUN TERMINATES CLEANLY, and this guard is ALL-OR-NOTHING over
#     the run rather than per element. The maximal run of separator+day is
#     consumed atomically and must then be followed (after spaces/tabs only)
#     by sentence or bracket punctuation, a closing quote, a newline, or end
#     of text — NOT by a word, and NOT by a separator. If it is not, the tail
#     matches nothing at all and every element stays graded.
#
#     The all-or-nothing part is the correction that a per-element version of
#     this guard needed, and it is the fail-open the issue's Invariants block
#     names. Measured on the first #187 patch, where the guard was a lookahead
#     on each element:
#         "Since Aug 19, 8, 9, 12 were your resting HR readings"
#             -> survivors ['12']; 8 and 9 SILENTLY UNGRADED
#     The trailing word protected only the LAST element; the interior ones had
#     already been committed one at a time, so two real resting-HR readings
#     stopped being checked. The atomic group `(?>...)` is what forbids that:
#     the run cannot succeed partially, cannot backtrack to a shorter run that
#     happens to end at an interior comma, and so either the whole thing is a
#     date list or none of it is. (Atomic groups need Python 3.11+, which is
#     this project's floor.)
#
#     The terminator set is deliberately NARROW for the same reason: "," and
#     "and" are separators, and admitting either as a terminator reopens the
#     hole one comma further along — "Since Aug 19, 8, 9, 44.0 were readings"
#     would end the run at the comma before 44.0 and strip the 8 and the 9.
#     So a list followed by a comma or by a word is simply not stripped:
#     "Aug 19, 21 and 22 were rest days" keeps grading 21 and 22. A bare day
#     left behind is the old false refusal; a real figure silently ungraded is
#     the failure this whole comment exists to prevent.
#  5. IT ONLY EXTENDS A DATE. The tail cannot match on its own — it hangs off
#     an already-matched month+day (or month+range, or month+day+year), so no
#     free-standing "21, 22, 23" is ever stripped.
#
# Deliberately NOT supported: a range as a continuation element ("Aug 19,
# 21–23" strips "Aug 19" and grades 21/23 as before). Ranges keep working in
# the first element only; widening further has to earn it with evidence.
#
# The substitution is a SPACE, as for every other form here, so removing a list
# can never join two visible numbers into a fabricated one.
_PROSE_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"(?:\d{4}(?!\d)"
    r"|\d{1,2}(?!\d)(?:\s*[–—-]\s*\d{1,2}(?!\d))?(?:,\s*\d{4}(?!\d))?"
    r"(?:(?>(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)"
    r"(?:0?[1-9]|[12]\d|3[01])(?!\d)(?![.,/]\d))+)"
    r"(?=[ \t]*(?:[;:.!?)\]}\"'’”\n]|$)))?)"
    r"|\b\d{1,2}:\d{2}\s*[AaPp]\.?[Mm](?![A-Za-z])\.?")
# Domain names whose digit cannot be recognised by position alone: a bare digit
# in "Zone 2", and a digit that leads a name in "5k". Stripped before tokenizing,
# exactly as the date patterns above are.
#
# Keep this list explicit and short. Every entry removes a digit from the gate,
# so a term earns its place by being a name this project actually tracks — the
# zones are the plan's heart-rate model, 5k/10k are race distances, and Week
# labels are authored plan ordinals. A term earns its place by being a name
# this project actually tracks; a term that merely contains a digit does not.
#
# "Week" is deliberately CASE-SENSITIVE while the rest of this pattern is not.
# The authored label is capitalised ("drafts Week 7"); the determiner phrase is
# not ("this week 3 sessions are logged"). Under re.I the second form strips a
# real measurement and the gate silently stops checking it — measured
# 2026-08-24: "This week 3 sessions are logged." -> unsupported=[], while the
# same claim as "You logged 3 sessions this week." -> ['3']. That is exactly the
# "allowlisting Zone 2 quietly means stop checking the digit 2" failure this
# comment block exists to prevent, so the case distinction is load-bearing.
_NAME_TERM_RE = re.compile(r"\b(?:Zone\s*[25]|(?:5|10)k|(?-i:Week)\s*\d+)\b", re.I)

# The DENOMINATOR of a subjective scale, which is a constant of the display
# contract and not a measurement. `api_daily.figure(..., scale=N)` is the only
# thing in this codebase that emits one, it has exactly two call sites, and so
# this pattern accepts exactly those two constants: `scale=5` for the check-in
# ratings (stress/soreness/energy/sleep_quality) and `scale=100` for the
# readiness composite. Nothing publishes either as a claim -- `_numbers_in`
# walks numeric leaves and the rendered "3/5" is a string -- so a rating
# reported in natural English was unverifiable by construction (#217).
#
# Three properties carry the safety argument, and each is load-bearing:
#
#  1. It strips a PATTERN, never a digit. A bare 5 is untouched; only a 5 (or
#     100) sitting in the denominator slot of a rating expression is blanked.
#     "You slept 5 h" still grades. That is the "Zone 2 must not mean stop
#     checking the digit 2" rule applied to a denominator.
#  2. The numerator is KEPT (the `keep` group is re-emitted verbatim) and the
#     separator becomes a SPACE, never nothing. Removing a visible separator
#     fabricates a number -- `29-30` -> `2930` -- which is the failure the
#     module docstring above exists to prevent.
#  3. A rating ANCHOR is required before the numerator, within 12 non-digit
#     characters on one line. Without it, "3 out of 5 planned sessions" would
#     stop grading a genuine published count. The anchor vocabulary is the
#     check-in field names plus the rate/score verbs, i.e. the words the tool
#     surface itself uses; it is kept short for the same reason
#     `_NAME_TERM_RE` is.
#
# The residual hole is exactly one value wide: a model could write the literal
# 5 (or 100) as the denominator of an anchored ratio that is really a count.
# Any OTHER value there is still tokenized and still demands a claim, which is
# why the constants are enumerated rather than matched as `\d+`.
#
# The trailing guard is `(?!\d)(?![.,/]\d)` rather than `(?![\d.,/])`: a
# sentence-final "5." must still be recognised (it is how the live failure
# presented), while "out of 500", "out of 5.5", "3/5,000" and "4/5/2026" must
# not match at all and are left fully graded.
_SCALE_DENOM_RE = re.compile(
    r"(?P<keep>\b(?:rate[ds]?|rating|scor(?:e[ds]?|ing)|stress|soreness"
    r"|energy|(?:sleep\s+)?quality|readiness)\b[^.;!?\d\n]{0,12}\d+)"
    r"(?:\s*out\s+of\s+|\s*/\s*)(?:100|5)(?!\d)(?![.,/]\d)",
    re.I)

# A real finding has to put a value in context.  These patterns deliberately
# describe observable language rather than trying to understand the claim: the
# gate is a cheap deterministic backstop, not a second judge.  In particular,
# "currently", "latest", and "today" alone do not count as a time window; that
# distinction prevents the 2026-07-12 HRV filler from passing again.
_COMPARISON_RE = re.compile(
    r"\b(?:versus|vs\.?|compared\s+(?:with|to)|relative\s+to|"
    r"from\b.{0,80}\bto\b|between\b.{0,80}\band\b|"
    r"more|less|higher|lower|older|newer|increased|decreased|declined|"
    r"dropped|rose|fell|improved|worsened|up|down|delta|difference|"
    r"trend|trending|correlat\w*|association|unchanged|stable|"
    r"passed\s+fdr|no\s+(?:clear|verified|reliable|actionable)\b|"
    # Superlatives ARE comparisons — against every other value in the set.
    # "active energy peaked at 987 kcal" locates the number among all the
    # others; "heart rate variability is currently 46.34" does not, which is
    # the distinction this gate exists to draw. None of these appear in the
    # 2026-07-12 filler, so adding them does not reopen that hole.
    r"peak\w*|highest|lowest|maximum|minimum|best|worst|record\b|"
    r"longest|shortest|fastest|slowest|most|fewest|"
    r"did\s+not\b)", re.I | re.S)
_WINDOW_RE = re.compile(
    r"\b(?:over|across|during|throughout|within|for)\s+(?:the\s+)?"
    r"(?:(?:last|past|recent|current|prior|previous|next|this)\s+)?"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)?\s*"
    r"(?:-\s*)?(?:day|days|week|weeks|month|months|period|periods|window)\b|"
    r"\b(?:last|past|recent|current|prior|previous|next|this)\s+"
    r"(?:\d+\s*[- ]?)?(?:day|days|week|weeks|month|months|period)\b|"
    r"\b\d+\s*(?:d|w|m)\b|\b\d+\s*[- ](?:day|days|week|weeks|month|months)\b|"
    r"\b\d{4}-\d{2}-\d{2}\s*(?::|\bto\b)\s*\d{4}-\d{2}-\d{2}\b",
    re.I,
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_FILLER_RE = re.compile(
    r"\b(?:as it stands today|latest tool reading confirms|"
    r"continue monitoring(?: this metric)?|tracking your present load state|"
    r"for recovery insights)\b", re.I)
# Both footer shapes: the historical "confidence N/100 · N claim(s) dropped"
# still present in archived reports, and the current "N claim(s) dropped" —
# the confidence score was removed once it turned out to score the vacuous
# 2026-07-12 report 100 and a genuinely useful one 92.
_REPORT_FOOTER_RE = re.compile(
    r"\s*[—–-]\s*(?:confidence\s+\d+\s*/\s*100\s*[·•]\s*)?"
    r"\d+\s+claim\(s\)\s+dropped\.?\s*$", re.I)


@dataclass(frozen=True)
class SubstanceResult:
    """Explain the deterministic substance decision made for one text.

    ``signals`` contains the categories that made the text contextual rather
    than merely numeric.  ``evidence`` contains short matched snippets for
    logging/debugging.  The object is truthy when ``ok`` is true, so callers can
    use either ``result.ok`` or ``if result`` without losing the explanation.
    """

    ok: bool
    reason: str
    signals: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok

    def as_dict(self) -> dict:
        """Return a JSON/log-friendly representation."""
        return {"ok": self.ok, "reason": self.reason,
                "signals": list(self.signals), "evidence": list(self.evidence)}


def substance_check(text: str) -> SubstanceResult:
    """Reject numerically grounded prose that still says nothing useful.

    A finding must contain at least one comparative/change relation (for
    example ``versus``, ``increased``, ``from ... to``, a delta, or a failed
    association) or a real multi-day/period context (for example ``over twelve
    weeks``, ``last day``, or two dated observations).  A numeral by itself,
    ``currently``/``latest``/``today``, and advice-shaped padding such as
    ``continue monitoring`` do not qualify.  This is intentionally lexical:
    it catches the known vacuity pattern quickly and deterministically, but it
    cannot tell whether a valid comparison is insightful, causal, or medically
    important.  The caller should still run grounding and any domain judge.

    The return value is structured so a caller can log the rejected reason.
    """
    if not isinstance(text, str) or not text.strip():
        return SubstanceResult(False, "empty finding", evidence=())

    # The deep-dive appends this operational footer to reports.  It is not
    # evidence, and words such as "dropped" in it must not rescue a vacuous
    # report (the 2026-07-12 incident had confidence 100/100).
    content = _REPORT_FOOTER_RE.sub("", text).strip()
    comparison = _COMPARISON_RE.search(content)
    window = _WINDOW_RE.search(content)
    if not window and len(_ISO_DATE_RE.findall(content)) >= 2:
        window = _ISO_DATE_RE.search(content)

    signals = []
    evidence = []
    if comparison:
        signals.append("comparison")
        evidence.append(comparison.group(0).strip()[:80])
    if window:
        signals.append("window")
        evidence.append(window.group(0).strip()[:80])

    filler = _FILLER_RE.findall(content)
    if not signals:
        detail = "no comparative or temporal context beyond a value"
        if filler:
            detail += "; filler/advice language is not substance"
        return SubstanceResult(False, detail, evidence=tuple(filler))
    if filler and not comparison and not window:
        return SubstanceResult(False, "filler/advice language without a finding",
                               evidence=tuple(filler))
    return SubstanceResult(True, "contains contextual finding signals",
                           tuple(signals), tuple(evidence))


# ---------------------------------------------------------------------------
# #16 — a stated DAY COUNT against the answer's OWN itemisation.
#
# Scope, decided on measured evidence 2026-09-04: build a check for the one
# observed failure shape only — a day count overstated by one, contradicted by
# the sentence's own itemisation. Explicitly ruled out, and NOT built here: a
# word-number-to-digit tokenizer, and any widening of `numeric_tokens`. Over
# 158 published narrations the exposure was 219 word-number hits adjacent to a
# unit or metric noun; the harm was 2 arithmetically wrong against 119 correct.
# Roughly 140 of those 219 were fixed comparison windows ("the last four
# weeks"), which is ordinary coaching prose — a check that refuses them is
# worse than the hole it closes. Every guard below exists to leave those alone.
#
# `_COUNT_WORDS` is a closed twelve-entry lookup used ONLY to read the integer
# on the LEFT of this one comparison. It is not a tokenizer, nothing else
# imports it, and `_numeric_tokens` / `strip_dates_and_names` are untouched:
# word-numbers remain invisible to every grounding gate, exactly as decided.
#
# WHY IT LIVES HERE, beside `substance_check`, and not in `deepdive_verify`:
# this is a SELF-CONSISTENCY check, not a grounding check. It never reads the
# ledger, the payload or a claim, because it does not need one — the payload
# cannot supply "three" either way, and the contradiction is entirely inside
# the text. `deepdive_verify`'s scanners all reconcile prose against a scoped
# Python verdict; this one reconciles prose against itself, which is the same
# category as the other text-only deterministic gate in this module. It shares
# `_PROSE_DATE_RE` / `_DATE_RE` rather than defining a second date pattern, so
# the "one tokenizer definition, shared" invariant holds: there is no new
# numeric pattern in this repo.
#
# IT FIRES ONLY WHEN BOTH HALVES ARE IN THE TEXT: a count of days, and an
# enumeration of dated entries that the count phrase ITSELF introduces. The
# introducer (a colon, an OPENING spaced dash, an opening paren, "namely") is what makes
# the list exhaustive and therefore comparable. Without it, "you ran five days
# this week; Mar 3 was the hardest" is a partial mention, not a contradiction,
# and refusing it would cost far more than the two errors this closes.
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12}

# The count, then at most two adjective-ish words ("running", "easy"), then the
# day noun. The leading guard is the tokenizer's own: a digit after a letter,
# a dot or a hyphen belongs to something else ("68.3", "VO2max", "3-day").
_DAY_COUNT_RE = re.compile(
    r"(?<![\w.-])(?P<count>\d{1,2}|"
    + "|".join(sorted(_COUNT_WORDS, key=len, reverse=True))
    + r")\s+(?P<gap>(?:[A-Za-z][A-Za-z-]*\s+){0,2})(?P<noun>days?)(?![\w-])",
    re.I)
# Determiners that must never appear between the count and the noun: each one
# turns the phrase into something other than "a count of the listed days".
_GAP_STOP = frozenset({
    "of", "the", "a", "an", "and", "or", "out", "to", "in", "on", "at", "for",
    "by", "with", "your", "my", "our", "their", "his", "her", "its", "this",
    "that", "these", "those", "is", "are", "was", "were", "has", "have", "had",
    "been", "more", "less", "fewer", "than", "as", "per", "every", "each"})
# A FIXED COMPARISON WINDOW, an approximation, or a bound — never a count of a
# set the sentence is about to enumerate. This is the guard that protects the
# ~140/219 measured idiom hits: "over the last four days", "about three days",
# "at least two days". "other" is deliberately absent: "the other four days
# (Mar 4, 5, 7, 9)" is a real published shape and IS an exhaustive itemisation.
#
# The `a full` / `an entire` group is the TOTALITY idiom, added 2026-09-04 on a
# measured false positive over the private narration corpus: "the week of <date>
# — a full 7 days — totaled <n> jogging minutes". "a full N days" states the
# LENGTH of a window; it never counts a set the sentence is about to list.
# `all` is already in the list above and already stands "all three days: Mar 3,
# Mar 5" down, so this is that same class written with a determiner instead of
# a bare quantifier — not a new kind of leniency. It stays a CLOSED lexical
# list for exactly that reason: full/whole/entire/complete after a/an/the, and
# nothing generic ("a good 7 days"). The moment this becomes a modifier
# wildcard it starts eating the real hits.
_WINDOW_LEAD_RE = re.compile(
    r"\b(?:last|past|previous|prior|next|recent|coming|upcoming|first|final|"
    r"preceding|following|another|any|all|about|around|approximately|roughly|"
    r"nearly|almost|least|most|than|under|over|within|up\s+to|every|each|"
    r"(?:an?|the)\s+(?:full|whole|entire|complete))"
    r"\s+(?:the\s+)?$", re.I)
# A PARTITIVE: "one of your three easy days" counts a subset out of the set,
# so the enumeration that follows is not the count's own itemisation.
_PARTITIVE_LEAD_RE = re.compile(
    r"\b(?:\d{1,2}|" + "|".join(sorted(_COUNT_WORDS, key=len, reverse=True))
    + r"|some|several|many|few|none|any|all|each|one)\s+(?:out\s+)?of\s+"
    r"(?:(?:your|my|our|their|his|her|its|the|these|those)\s+)?$", re.I)
# A count of PLANNED days is a count of intent, not of the entries beside it:
# "two out of four planned days: Mar 3 and Mar 5" is a correct sentence in
# which the mismatch IS the point. These words in the gap stand the check down.
_PLANNED_GAP = frozenset({
    "planned", "scheduled", "prescribed", "target", "targeted", "intended",
    "remaining", "possible", "available", "potential", "upcoming", "key",
    "recommended", "suggested", "optional", "ideal"})
# The enumeration introducer. The optional lead is a TIGHT allowlist rather
# than "up to N words", because a permissive bridge reads "three days later —
# Mar 6 felt easier" as a count with a one-item itemisation and fires on
# perfectly good prose.
_ENUM_INTRO_RE = re.compile(
    # No `\A`: this pattern is applied with `.match(prose, pos)`, which already
    # anchors at pos, while `\A` would only ever match at offset 0.
    r"(?:\s+(?:this|that|last|the)?\s*(?:week|month))?"
    r"(?:\s+(?:in\s+total|total|altogether|overall|so\s+far))?"
    r"\s*(?::|\s+[—–]\s+|\s+-\s+|\(|,\s+(?:namely|specifically)[,:]?\s)")
# A spaced dash that CLOSES a dash-delimited aside is not an introducer. An
# aside is bounded on BOTH sides, and what follows its closing dash belongs to
# the sentence the aside interrupted — not to the count sitting inside it. This
# is the structural half of the 2026-09-04 false positive: in "the week of
# <date> — a full 7 days — totaled <n> jogging minutes, and ...", the trailing
# dash was read as "this count introduces what follows" and the window length
# was then compared against the remainder of the sentence.
#
# Parity inside the segment is what separates the two dashes, and it is exactly
# right for the shape that matters: a genuine introduction ("four cycling days
# — 30.0 minutes on Mar 3 and 47.5 minutes on Mar 6") has no unmatched dash
# before the count, so it still fires. Only the dash form of the introducer is
# affected; a colon or a parenthesis inside an aside still introduces normally,
# because neither is half of a pair that the count is sitting between.
#
# HORIZONTAL whitespace only, and that is load-bearing rather than tidiness: a
# markdown bullet ("\n- 30.0 minutes on Mar 3") is a spaced dash too, and with
# `\s` on the left an ODD number of bullets earlier in the same segment flipped
# the parity and stood a later, genuine dash-introduced hit down. An aside is
# never opened by a dash that begins a line, so a leading newline disqualifies
# the dash from the parity count and the bullets fall out.
_SPACED_DASH_RE = re.compile(r"[^\S\n\r][—–][^\S\n\r]|[^\S\n\r]-[^\S\n\r]")
# A segment ends at a sentence terminator or a blank line. The digit lookbehind
# keeps "44.0" and "Mar 3." from ending it; the capitalised-abbreviation
# lookbehinds keep "Aug." and "Sept." from ending it.
_SEGMENT_END_RE = re.compile(
    r"(?<!\d)(?<![A-Z][a-z]{2})(?<![A-Z][a-z]{3})[.!?](?=[\s\"'’”)\]]|$)"
    r"|\n[ \t]*\n")
# An itemisation that ADMITS it is partial is not comparable with the count.
_PARTIAL_LIST_RE = re.compile(
    r"\b(?:including|includ(?:es|ed|ing)|such\s+as|for\s+example|e\.?g\.?|"
    r"among\s+(?:them|others|other)|others|notably|especially|mostly|chiefly|"
    r"at\s+least|and\s+so\s+on|etc\.?|plus\s+others|and\s+more)\b|\.\.\.|…",
    re.I)
# A range denotes an unknown number of days, so the list cannot be counted.
_RANGE_HINT_RE = re.compile(
    r"\b(?:between|through|thru)\b"
    r"|\d{4}-\d{2}-\d{2}\s*(?:to|through|[–—-])\s*\d{4}-\d{2}-\d{2}", re.I)
_WEEKDAY_RE = re.compile(
    r"\b(?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?\b\.?",
    re.I)
_ORDINAL_DAY_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.I)
# A BARE CONTINUATION DAY that `_PROSE_DATE_RE` did not absorb. That pattern's
# day-list tail (#187) is deliberately all-or-nothing: "Mar 3, 5 and 7 were
# easy" ends in a word, so the tail matches nothing and only "Mar 3" is a date.
# For the grounding gate that is the right fail-open — 5 and 7 keep being
# graded. For THIS check it is the dangerous direction: two real days would go
# uncounted and a correct sentence would be called a contradiction. So when the
# run is present but unabsorbed, the check stands down instead of counting.
# The day-shape guards are copied in intent from `_PROSE_DATE_RE`'s tail: 1..31,
# and never the head of a decimal or a longer digit run, so "on Mar 3, 47.5
# minutes on Mar 6" is not mistaken for a continuation and still gets checked.
_CONT_DAY_RUN_RE = re.compile(
    r"(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)"
    r"(?:0?[1-9]|[12]\d|3[01])(?!\d)(?![.,/]\d))+")
_RELATIVE_DAY_RE = re.compile(r"\b(?:today|yesterday|tonight)\b", re.I)
# An itemisation does not have to label every entry: "three key days:
# Tuesday's intervals, Thursday's tempo, and the long run" names three entries
# and dates two, and it is perfectly correct prose. So the comparison is
# against an UPPER BOUND — dated days plus undated entries that could be days —
# and not against the dated days alone. An entry that only closes the list
# ("and the rest of the week had no cycling recorded") is not a day and does
# not raise the bound; that clause is exactly what makes the observed failure
# self-refuting.
_ITEM_SPLIT_RE = re.compile(r"[;,\n]|\band\b|\bplus\b|&", re.I)
_CLOSING_CLAUSE_RE = re.compile(
    r"\b(?:no|none|nothing|not|never|zero|nil|else|otherwise|rest\s+of|"
    r"remainder|remaining|only|all)\b", re.I)


@dataclass(frozen=True)
class DayCountResult:
    """Explain the deterministic day-count decision made for one answer.

    ``ok`` is false only when the text states a day count and then itemises
    FEWER entries under it than the count claims. ``findings`` carries one
    dict per contradiction — ``stated``, ``itemised`` (distinct dated days),
    ``undated`` (listed entries carrying no day label, which still count
    against the claim), ``delta``, the matched ``span`` and the day ``labels``
    — so a hit can be audited by hand without re-running the check. ``notes`` records why each near-miss was left
    alone, which is what makes a refusal-cost sweep readable.

    The object is truthy when ``ok`` is true, as ``SubstanceResult`` is.
    """

    ok: bool
    reason: str
    findings: tuple[dict, ...] = ()
    evidence: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok

    def as_dict(self) -> dict:
        """Return a JSON/log-friendly representation."""
        return {"ok": self.ok, "reason": self.reason,
                "findings": [dict(f) for f in self.findings],
                "evidence": list(self.evidence), "notes": list(self.notes)}


def _day_labels(span: str) -> tuple[set, str | None]:
    """Count the DISTINCT days a span itemises, or say why it cannot be counted.

    Distinct, because two entries on one date are one day and the count on the
    left of the comparison is a count of days. Every ambiguity returns a reason
    instead of a number: an uncountable list must make the check stand down,
    never fire. Overcounting is the safe direction (it only suppresses a hit);
    undercounting invents a contradiction, which is the failure this whole
    check is scoped to avoid.

    Dates are recognised with the module's existing `_PROSE_DATE_RE` and
    `_DATE_RE` — no second date pattern is defined for this check, so the
    day-list tail (#187) and the half-eaten-year guards apply here unchanged.
    """
    if _RANGE_HINT_RE.search(span):
        return set(), "the itemisation spans a range"
    labels: set = set()
    # LINE BY LINE, because `_PROSE_DATE_RE`'s range alternative accepts any
    # whitespace around the dash — so a markdown list ("Mar 3\n- 20.0 minutes")
    # reads as the range "Mar 3 - 20" and the whole itemisation would stand the
    # check down. A range never spans a line break in prose, and every other
    # pattern here is line-local anyway.
    for line in span.splitlines() or [span]:
        consumed: list[tuple[int, int]] = []
        for match in _PROSE_DATE_RE.finditer(line):
            chunk = match.group(0)
            consumed.append(match.span())
            if ":" in chunk:
                continue                  # a clock time is not a day label
            if re.search(r"\d\s*[–—-]\s*\d", chunk):
                return set(), "the itemisation spans a range"
            month = re.search(r"[A-Za-z]+", chunk)
            days = [tok for tok in re.findall(r"\d+", chunk) if len(tok) <= 2]
            if month is None or not days:
                continue                  # "July 2026" is a month, not a day
            if _CONT_DAY_RUN_RE.match(line, match.end()):
                return set(), "an unterminated day list cannot be counted"
            for day in days:
                labels.add(("md", month.group(0)[:3].lower(), int(day)))
        for match in _DATE_RE.finditer(line):
            if any(start <= match.start() < end for start, end in consumed):
                continue
            if len(match.group(0)) == 7:
                continue                  # a bare year-month is not a day
            consumed.append(match.span())
            labels.add(("iso", match.group(0)))
        for pattern, kind in ((_WEEKDAY_RE, "wd"), (_ORDINAL_DAY_RE, "ord"),
                              (_RELATIVE_DAY_RE, "rel")):
            for match in pattern.finditer(line):
                if any(start <= match.start() < end for start, end in consumed):
                    continue
                token = match.group(0).strip(".").lower()
                labels.add((kind, token[:3] if kind != "ord" else token))
    return labels, None


def _closes_a_dash_aside(prose: str, pos: int) -> bool:
    """True when the count at ``pos`` sits inside an unclosed dash aside.

    Counted within the current sentence only, because an aside never spans a
    sentence terminator and a stray dash in an earlier sentence must not flip
    the parity of this one. An odd number of spaced dashes between the start of
    the segment and the count means one of them opened an aside that is still
    open, so the next spaced dash closes it rather than introducing a list.
    """
    start = 0
    for match in _SEGMENT_END_RE.finditer(prose):
        if match.end() > pos:
            break
        start = match.end()
    return len(_SPACED_DASH_RE.findall(prose[start:pos])) % 2 == 1


def _undated_entries(span: str) -> int:
    """Count entries in an itemisation that carry no day label but could be one.

    These raise the ceiling the stated count is compared against. An entry that
    merely closes the list — "and the rest of the week had no cycling recorded",
    "nothing else was logged" — is not a candidate day, and a bare year (the
    tail of "Mar 3, 2031") is not an entry at all.
    """
    undated = 0
    for item in _ITEM_SPLIT_RE.split(span):
        entry = item.strip(" \t\r-*•·:.!?…()[]\"'")
        if not entry or not any(char.isalnum() for char in entry):
            continue
        if re.fullmatch(r"\d{1,4}(?:st|nd|rd|th)?", entry, re.I):
            continue    # a bare number: a year, or a continuation day already
                        # counted as a label by the date scan above
        if _day_labels(entry)[0]:
            continue
        if _CLOSING_CLAUSE_RE.search(entry):
            continue
        undated += 1
    return undated


def day_count_check(text: str) -> DayCountResult:
    """Reject an answer whose stated day count contradicts its own itemisation.

    The observed defect, twice, on otherwise correct figures: the answer says
    it ran on three days, then itemises two of them and adds that the rest of
    the week had none. Every quantity is right; only the day count is wrong,
    and the sentence refutes it without any external reference. So this check
    takes no ledger, no payload and no claim — call it on a bare answer string.

    It fires only when a count of days is IMMEDIATELY followed by an
    enumeration it introduces (colon, spaced dash, parenthesis, "namely") and
    the enumeration cannot account for the count even generously: the stated
    number is compared against distinct dated days PLUS undated entries in the
    same list that could be days, so "three key days: Tuesday's intervals,
    Thursday's tempo, and the long run" is left alone. It is
    deliberately one-directional: a stated count LOWER than the itemisation is
    not flagged, because a list may legitimately run past the window the count
    describes, and only overstatement was observed.

    Returns a ``DayCountResult``; ``ok`` is true when nothing contradicts,
    including when no countable day claim is present at all.
    """
    if not isinstance(text, str) or not text.strip():
        return DayCountResult(True, "no stated day count to check")

    # Scan the invisible-stripped view, as every other gate here does; the
    # reported offsets are offsets into that view.
    prose = strip_invisible(text)
    findings: list[dict] = []
    notes: list[str] = []

    for match in _DAY_COUNT_RE.finditer(prose):
        raw = match.group("count").lower()
        stated = _COUNT_WORDS.get(raw, int(raw) if raw.isdigit() else 0)
        if not 1 <= stated <= 31:
            continue
        gap = [word.lower() for word in match.group("gap").split()]
        if any(word in _GAP_STOP for word in gap):
            continue
        if any(word in _PLANNED_GAP for word in gap):
            notes.append(f"a planned or prospective count: {match.group(0)!r}")
            continue
        lead = prose[max(0, match.start() - 40):match.start()]
        if _PARTITIVE_LEAD_RE.search(lead):
            notes.append(f"partitive, not a count: {match.group(0)!r}")
            continue
        if _WINDOW_LEAD_RE.search(lead):
            notes.append(f"comparison window or bound: {match.group(0)!r}")
            continue
        intro = _ENUM_INTRO_RE.match(prose, match.end())
        if intro is None:
            notes.append(f"no itemisation introduced: {match.group(0)!r}")
            continue
        # Ask the INTRODUCER what form it took, rather than re-matching the
        # aside pattern over it: `_ENUM_INTRO_RE`'s dash alternatives accept a
        # newline, which the parity pattern above deliberately does not.
        if (intro.group(0).rstrip()[-1:] in "—–-"
                and _closes_a_dash_aside(prose, match.start())):
            notes.append(
                f"a dash aside closes rather than introduces: "
                f"{match.group(0)!r}")
            continue
        end = _SEGMENT_END_RE.search(prose, intro.end())
        span = prose[intro.end():end.start() if end else len(prose)]
        if _PARTIAL_LIST_RE.search(span):
            notes.append(f"itemisation declares itself partial: {match.group(0)!r}")
            continue
        labels, why = _day_labels(span)
        if why is not None:
            notes.append(f"{why}: {match.group(0)!r}")
            continue
        itemised = len(labels)
        if itemised == 0:
            notes.append(f"nothing dated is itemised: {match.group(0)!r}")
            continue
        undated = _undated_entries(span)
        if stated <= itemised + undated:
            continue
        findings.append({
            "stated": stated, "itemised": itemised, "undated": undated,
            "delta": stated - itemised - undated,
            "span": prose[match.start():end.start() if end else len(prose)],
            "start": match.start(), "end": end.start() if end else len(prose),
            "labels": sorted(str(label) for label in labels),
        })

    if findings:
        return DayCountResult(
            False,
            "a stated day count exceeds the days the same sentence itemises",
            tuple(findings), tuple(f["span"] for f in findings), tuple(notes))
    return DayCountResult(
        True, "no stated day count contradicts its own itemisation",
        notes=tuple(notes))


def run_model(prompt: str, *, tools: str | None = None, ctx=None,
               max_turns: int | None = None,
               timeout: int | None = None, think: bool = False,
               options: dict | None = None) -> str:
    """The pipeline's single model entrypoint.

    Renamed 2026-08-26: the old name collided with the consuming layer's
    messaging-gateway binary, so a grep for egress returned this function too
    and a reader auditing the delivery path could audit the wrong thing. This
    function never delivers anything; it only calls the model.

    tools=None -> tool-less
    narration/judge via llm.complete; tools set -> deep-dive researcher via
    llm.tool_loop. Always returns a string ("" on any model error).
    When tools is set the call delegates to tool_loop, which manages its own
    turn budget and timeouts; `timeout` and `options` apply only to the
    tool-less (complete) path."""
    if tools:
        if ctx is None:
            raise ValueError("the researcher path needs a VaultContext: "
                             "run_model(..., tools=..., ctx=ctx)")
        return llm.tool_loop(prompt, ctx=ctx, tools=llm.tool_schemas(ctx),
                             think=think, max_turns=max_turns or 12)
    return llm.complete(prompt, think=think, timeout=timeout, options=options)


def _strip_json_invisibles(candidate: str) -> str:
    """Strip invisible corruption from JSON syntax, preserving string values."""
    clean = []
    in_string = False
    escaped = False
    for char in candidate:
        if in_string:
            clean.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            clean.append(char)
            in_string = True
        else:
            clean.append(strip_invisible(char))
    return "".join(clean)


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM response.

    Fences and surrounding prose are tolerated. At the JSON parse boundary,
    invisible Unicode marks/format characters are removed from JSON syntax and
    scalar regions; quoted string values are retained exactly. Returns the
    parsed value or None.
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
        if start == -1:
            return None
        candidate = text[start:]
    candidate = _strip_json_invisibles(candidate)
    for end in range(len(candidate), 0, -1):
        try:
            return json.loads(candidate[:end])
        except json.JSONDecodeError:
            continue
    return None


def split_claim_channel(text: str) -> tuple[str, list[dict] | None]:
    """Separate coach prose from an optional structured claim channel."""
    parsed = extract_json(text)
    if (isinstance(parsed, dict) and isinstance(parsed.get("text"), str)
            and "claims" in parsed):
        claims = parsed.get("claims")
        return parsed["text"], claims if isinstance(claims, list) else None
    return text or "", None


def _numbers_in(obj) -> set[str]:
    """All numeric tokens appearing anywhere in a briefing (as normalized strings)."""
    found = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            found.add(_norm_num(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return found


def _norm_num(s) -> str:
    f = float(str(s).replace(",", ""))
    return str(int(f)) if f == int(f) else f"{f:.2f}"


def strip_dates_and_names(prose: str) -> str:
    r"""Blank out dates and known name-terms before a figure scan.

    The ONE definition of this pipeline; `grounding_check` and both
    `deepdive_verify` scanners call it rather than repeating the three
    substitutions, which is how they drifted (F-86).

    Invisible characters are removed FIRST. `_PROSE_DATE_RE` needs `\s`
    between the month and the day, and a format character is not whitespace —
    so `Aug<U+200F>10` did not match, the date was not stripped, and the day
    number was graded as an unclaimed figure. Only Mn/Mc/Me/Cf are removed:
    a reader cannot see them, so one inside a date or a figure is corruption.
    A VISIBLE separator is never touched, because removing one would fabricate
    a number (`29-30` -> `2930`).

    Subjective-scale denominators are blanked last (#217). They are the same
    category as a date -- a constant of the display contract rather than a
    quantitative claim -- and the argument for why the pattern cannot swallow a
    real figure is at `_SCALE_DENOM_RE`. Only the denominator and its separator
    are replaced, with a SPACE; the numerator is a claim and keeps grading.
    """
    cleaned = strip_invisible(prose)
    cleaned = _PROSE_DATE_RE.sub(" ", _DATE_RE.sub(" ", cleaned))
    cleaned = _NAME_TERM_RE.sub(" ", cleaned)
    return _SCALE_DENOM_RE.sub(lambda m: m.group("keep") + " ", cleaned)


def _numeric_tokens(prose: str) -> list[str]:
    """Return quantitative prose tokens after the shared date/name cleanup.

    This is only a scanner. It deliberately does not license a value; callers
    that publish prose must reconcile each token with a scoped Python verdict.
    """
    return _NUM_RE.findall(strip_dates_and_names(prose or ""))


def grounding_check(prose: str, briefing: dict, rel_tol: float = 0.005,
                    abs_floor: float = 0.05):
    """Coarsely pre-filter prose against an unscoped briefing bag.

    Tolerance is RELATIVE (within rel_tol of the matched magnitude, with a tiny
    abs_floor for rounding) so neighbours of small integers (e.g. a fabricated
    99 vs a real 100) are caught while legitimate rounding of large values still
    passes. ISO and prose dates are stripped first — they are not quantitative
    claims. Returns ``(ok: bool, unsupported: list[str])``.

    This function is not safe evidence for publication. A value such as a day
    count can coincide with a percentage or a sleep delta because this legacy
    interface has no field/metric/period/unit identity. Publishing callers must
    use the scoped resolver in ``deepdive_verify``; this function is only a
    pre-filter for unscoped external callers.
    """
    allowed = _numbers_in(briefing)
    allowed_floats = {float(a) for a in allowed}
    bad = []
    for tok in _numeric_tokens(prose):
        try:
            val = float(tok.replace(",", ""))
        except ValueError:
            continue
        if _norm_num(val) in allowed:
            continue
        if any(abs(val - a) <= max(abs_floor, abs(a) * rel_tol) for a in allowed_floats):
            continue
        bad.append(tok)
    return (len(bad) == 0, bad)


def render_fallback(briefing: dict) -> str:
    """Deterministic, guaranteed-correct prose from the briefing. Used when the
    model's narration fails verification."""
    lines = [tp["seed"] for tp in briefing.get("talking_points", [])]
    body = "; ".join(lines).strip()
    text = (body[:1].upper() + body[1:] + ".") if body else "No notable changes today."
    for s in briefing.get("suggestions", []):
        text += " " + s["text"]
    return text
