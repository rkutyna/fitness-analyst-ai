"""LLM seam + deterministic helpers. run_model is a thin delegator over
health_advisor.llm (direct Ollama /api/chat); the model is only ever a text
transformer here — Python owns the truth. extract_json / grounding_check /
render_fallback are the gates that make a wrong or empty model output safe.
The numeric grounding pattern is shared with the other narration gates."""
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


def run_model(prompt: str, *, tools: str | None = None, ctx=None,
               max_turns: int | None = None,
               timeout: int | None = None, think: bool = False,
               options: dict | None = None) -> str:
    """The pipeline's single model entrypoint.

    Renamed from ``run_hermes`` 2026-08-26 (#92). That name was the Telegram
    gateway's, kept — as the old docstring said — only so monkeypatch-based
    tests would keep working, while ``checkin.HERMES_BIN`` names a binary
    genuinely called ``hermes``. Two unrelated "hermes" meant a grep for egress
    returned both and the reader had to know which was which; anyone assuming
    this one was the Telegram path audited the wrong thing.

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


def grounding_check(prose: str, briefing: dict, rel_tol: float = 0.005,
                    abs_floor: float = 0.05):
    """Every numeric token in `prose` must match a number in `briefing`. Tolerance
    is RELATIVE (within rel_tol of the matched magnitude, with a tiny abs_floor for
    rounding) so neighbours of small integers (e.g. a fabricated 99 vs a real 100)
    are caught while legitimate rounding of large values still passes. ISO and
    prose dates are stripped first — they are not quantitative claims. Returns
    (ok: bool, unsupported: list[str])."""
    allowed = _numbers_in(briefing)
    allowed_floats = {float(a) for a in allowed}
    bad = []
    cleaned = strip_dates_and_names(prose)
    for tok in _NUM_RE.findall(cleaned):
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
