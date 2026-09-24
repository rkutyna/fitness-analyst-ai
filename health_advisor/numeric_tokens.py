"""The shared numeric tokenizer for deterministic grounding gates.

The leading side is guarded because a digit following a letter is part of a
name: VO2max, Zone2, and A1c must not shed a bare numeric claim. The trailing
side is deliberately unguarded. Units are open-ended, so a digit followed by
letters remains a number; otherwise figures such as 52kg or 52.5kg could pass
through a grounding gate unchecked. Callers that know a closed domain-name
vocabulary must strip those names before using :data:`NUM_RE`.
"""
from __future__ import annotations

import re
import unicodedata


# Do not add a trailing unit allowlist. That asymmetry is intentional: names
# are a closed set and are stripped by their caller, while measurement units
# are open-ended and must continue to tokenize (52kg, 52km, 52cm, 52mm, 52oz,
# 52kj, 52rpm, 52kph, 52bps, 52kcal, 52bpm, 52ms, 52lbs, 52mi, 52ft, 52.5kg,
# 1st, 2nd, and -13,900.25 are all numeric forms).
_RAW_NUM_RE = re.compile(r"(?<![A-Za-z0-9])-?\d[\d,]*\.?\d*")


# The line is INVISIBLE vs VISIBLE, and it is the whole safety argument.
#
# Mn/Mc/Me (marks) and Cf (format) are zero-width: a reader cannot see them, so
# one sitting inside a figure is corruption and merging the halves is what the
# reader already sees. Measured 2026-08-28 with the artifact placed INSIDE the
# number rather than beside it: U+200B split '154' into '1' and '54' exactly as
# a combining macron did, and #180's evidence recorded U+200B three times in
# no-reasoning output.
#
# VISIBLE characters must never be stripped, however artifact-like they look.
# Katakana U+30FC (Lm), em dash (Pd) and every space (Zs) separate figures a
# reader sees as distinct, so removing one would FABRICATE a number: '29—30'
# would become '2930'. That is the same failure mode that rules out NFKC, which
# turns '1/2' into 1 and 2 and would reject correct briefings.
_INVISIBLE = frozenset({"Mn", "Mc", "Me", "Cf"})


def _is_invisible(char: str) -> bool:
    """Return whether *char* belongs to the shared invisible category set."""
    return unicodedata.category(char) in _INVISIBLE


def strip_invisible(text: str) -> str:
    """Remove only Unicode marks/format characters, without normalization.

    This is the JSON-boundary sanitizer's operation. Unlike the tokenizer's
    scan view, it deliberately does not perform NFD decomposition and therefore
    leaves visible text, including precomposed characters, unchanged.
    """
    return "".join(char for char in text if not _is_invisible(char))

# A typographic minus is a SIGN, not decoration, and dropping it is the one
# artifact here that can corrupt a number without rejecting it.
#
# Measured 2026-08-28 in real reasoning-off output ("temperature at U+2212 0.06"):
#   '<U+2212>0.06'  ->  ['0.06']      the sign is silently dropped
# Two consequences, and the second is the serious one:
#   - a correctly grounded NEGATIVE claim is refused, like every other artifact;
#   - prose saying -0.06 PASSES against a filed claim of +0.06, because the gate
#     never saw a sign. The coach can state the opposite sign to what Python
#     computed and the grounding gate approves it.
#
# Only characters that unambiguously MEAN minus are folded. EN DASH and EM DASH
# are deliberately NOT in this map: they separate ranges ("14-15 min/mi"), and
# folding one would turn a range into a negative number -- fabricating a value,
# the same failure that rules out NFKC.
_MINUS_FORMS = {
    "\u2212": "-",   # MINUS SIGN
    "\uff0d": "-",   # FULLWIDTH HYPHEN-MINUS
}


def _normalise_for_tokenising(text: str) -> tuple[str, list[int]]:
    """Return text with combining marks removed and normalized offsets.

    The returned text is a scan-only view. Callers must retain the original
    prose: model output is evidence, not something this module rewrites. NFD
    makes canonically equivalent sequences consistent, then only Unicode mark
    characters are removed. In particular, this does not turn a decimal comma,
    a thin-space thousands separator, or a degree sign into another character.
    """
    clean_chars = []
    original_offsets = []
    for original_index, original_char in enumerate(text):
        for char in unicodedata.normalize("NFD", original_char):
            if _is_invisible(char):
                continue
            clean_chars.append(_MINUS_FORMS.get(char, char))
            original_offsets.append(original_index)
    return "".join(clean_chars), original_offsets


class _NumericMatch:
    """A regex-match facade whose token is scan-normalized.

    ``finditer`` callers use ``group()``, ``start()`` and ``end()``. The
    offsets point into the original string, while the group is the numeric
    token after mark sanitisation. This lets neighborhood checks continue to
    inspect the exact model output.
    """

    def __init__(self, match, source: str, offsets: list[int]):
        self._match = match
        self.string = source
        self._offsets = offsets

    def group(self, *groups):
        if not groups or groups == (0,):
            return self._match.group(0)
        return self._match.group(*groups)

    def start(self, group=0):
        start = self._match.start(group)
        return self._offsets[start] if start >= 0 else start

    def end(self, group=0):
        end = self._match.end(group)
        if end < 0:
            return end
        if end == len(self._offsets):
            return len(self.string)
        return self._offsets[end]

    def span(self, group=0):
        return self.start(group), self.end(group)

    def __getattr__(self, name):
        return getattr(self._match, name)


class _NumericPattern:
    """The one shared numeric tokenizer, with scan-only mark sanitisation."""

    def __init__(self, pattern: re.Pattern):
        self._pattern = pattern
        self.pattern = pattern.pattern
        self.flags = pattern.flags

    def finditer(self, text: str):
        clean, offsets = _normalise_for_tokenising(text)
        for match in self._pattern.finditer(clean):
            yield _NumericMatch(match, text, offsets)

    def findall(self, text: str):
        return [match.group(0) for match in self.finditer(text)]

    def search(self, text: str):
        clean, offsets = _normalise_for_tokenising(text)
        match = self._pattern.search(clean)
        return _NumericMatch(match, text, offsets) if match else None


# Keep this object canonical: every grounding call site imports this exact
# tokenizer. The raw regex remains private so no caller can bypass the shared
# mark handling accidentally.
NUM_RE = _NumericPattern(_RAW_NUM_RE)


# health_advisor#495: the PUBLISH-time normaliser. The scan view above never
# rewrites model output; this is the one place that does, and it runs on model
# prose BEFORE any gate reads it, so the gate checks exactly what publishes.
#
# Observed: a double space, then U+0308 COMBINING DIAERESIS (or U+200B), then
# a digit. The tokenizer already reads straight through such marks, so the
# figure verified and the stray diacritic reached the reader. Removing only
# what a reader cannot see as intended text keeps the argument above intact:
# the digits a gate scanned are the digits that publish.
_STRAY_MARKS = range(0x0300, 0x0370)   # Combining Diacritical Marks (Mn)
_DIGITS = frozenset("0123456789")


def _is_attachable(char: str) -> bool:
    """A combining mark or format character: part of a cluster, not a base."""
    return unicodedata.category(char) in _INVISIBLE


def _drop_stray_mark(text: str, index: int) -> bool:
    """Return whether the combining mark at *index* is attached to no letter.

    Its base is the nearest preceding character that is not itself a mark or a
    format character. A mark is dropped when that base is absent, whitespace or
    an ASCII digit, or when the mark sits directly before an ASCII digit on a
    non-letter base. A mark on a LETTER always survives: a decomposed 'é' is
    'e' + U+0301 and is legitimate text.
    """
    back = index - 1
    while back >= 0 and _is_attachable(text[back]):
        back -= 1
    base = text[back] if back >= 0 else None
    if base is not None and unicodedata.category(base).startswith("L"):
        return False
    if base is None or base.isspace() or base in _DIGITS:
        return True
    ahead = index + 1
    while ahead < len(text) and _is_attachable(text[ahead]):
        ahead += 1
    return ahead < len(text) and text[ahead] in _DIGITS


# ZERO WIDTH (NON-)JOINER between two non-ASCII, non-space characters is how
# an emoji ZWJ sequence (a runner with a gender sign) and joining scripts are
# written; removing it splits one glyph into two. Measured in the 2026-09-2x
# ask-battery captures: 2 of the 8 prose fields carrying a format character
# were exactly this. Beside a space, a digit or any ASCII it is the artifact.
_JOINERS = frozenset("\u200c\u200d")


def _is_legitimate_joiner(text: str, index: int) -> bool:
    if text[index] not in _JOINERS or index == 0 or index + 1 >= len(text):
        return False
    before, after = text[index - 1], text[index + 1]
    return all(ord(c) > 0x7F and not c.isspace() for c in (before, after))


def normalise_model_prose(text):
    """Remove invisible artifacts from model prose before it is verified.

    * Every format character (category Cf: U+200B-U+200D, U+2060, U+FEFF,
      bidi marks, soft hyphen) is removed wherever it occurs -- except a
      joiner (U+200C/U+200D) between two non-ASCII, non-space characters,
      which is an emoji ZWJ sequence or joining-script text.
    * A combining diacritical mark (U+0300-U+036F) attached to no letter --
      after a space, on a digit, or right before a digit -- is removed.
    * A run of 2+ ASCII spaces touching a removal collapses to one space.
      Newlines, tabs and untouched spacing are left exactly as written.

    Everything else is untouched, deliberately without NFC/NFKC: precomposed
    and decomposed accented letters, dashes, degree, multiplication, approx,
    check and bullet characters all survive byte for byte. Non-string input is
    returned as is, so callers can apply it to an optional reply.
    """
    if not isinstance(text, str) or not text:
        return text
    pieces: list[str | None] = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        format_artifact = (category == "Cf"
                           and not _is_legitimate_joiner(text, index))
        stray_mark = (category == "Mn" and ord(char) in _STRAY_MARKS
                      and _drop_stray_mark(text, index))
        if format_artifact or stray_mark:
            pieces.append(None)      # a removal site
        else:
            pieces.append(char)
    if None not in pieces:
        return text
    out: list[str] = []
    run: list[str | None] = []

    def flush() -> None:
        if None in run:
            if " " in run:
                out.append(" ")
        else:
            out.extend(run)            # type: ignore[arg-type]
        run.clear()

    for piece in pieces:
        if piece is None or piece == " ":
            run.append(piece)
        else:
            flush()
            out.append(piece)
    flush()
    return "".join(out)
