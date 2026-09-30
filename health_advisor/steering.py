"""A deterministic guard: model-facing instruction text never reaches a user.

The ask path talks to the model in two registers. Facts are data the user may
read; steering (prompts, guidance, repair notes, refusal detail) is an
instruction to the model and means nothing to a person. A model that repeats
steering back into its answer is not narrating anything Python published, yet
every other gate can pass it: it carries no figure, no claim and nothing to
contradict (health_advisor#557 -- a "nothing cleared the bar" note reached a
user as narration with ``verification.ok``).

The rule enforced here is the property rather than a list of past offenders:
**a narration may not repeat the engine's own instruction text.** Three
independent tests, any of which rejects:

* a run of ``SHINGLE_WORDS`` consecutive words that also appears in any
  registered steering text (:func:`register`), so a verbatim or lightly
  edited echo of ANY registered instruction is caught without a curated list;
* a fixed set of marker phrases for the placeholder and status vocabulary the
  model is told about, including a placeholder that was never interpolated;
* the name of an internal tool (any registered name containing an
  underscore -- ``cite`` and the like are ordinary English and are not
  matched).

The guard is a filter over a candidate answer. It never rewrites it: a leak
sends the answer to the deterministic fallback.
"""
from __future__ import annotations

import re
import threading

SHINGLE_WORDS = 6

REASON = "answer repeated internal instructions"

_WORD_RE = re.compile(r"[a-z0-9]+")

# Phrases that only exist in prompts. Case-insensitive.
_MARKER_RE = re.compile(
    "|".join((
        r"cleared the bar",
        r"\bdo not say\b",
        r"\bclosed fact set\b",
        r"\bfact set\b",
        r"\bplaceholders?\b",
        r"\badvice slot\b",
        r"\btool[- ]call ledger\b",
        r"\bstatus:gathered_data_uncited\b",
        r"\bdata_gathered_not_cited\b",
        r"\bout_of_corpus_domain\b",
        r"\bcorpus_domain_unverified\b",
        # A placeholder the interpolator did not consume: {fact|...},
        # {pub:...}, {status:...}, {scale:...}, {advice:...}, {cite:...}.
        r"\{\s*(?:fact|pub|status|evidence|advice|cite|scale)\s*[:|]",
    )),
    re.IGNORECASE,
)

_LOCK = threading.Lock()
_TEXTS: dict[str, frozenset] = {}
_EXTRA_TOOL_NAMES: set[str] = set()


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(str(text).lower())


def _shingles(text: str) -> frozenset:
    """Word runs of a steering text, minus runs that are mostly numbers.

    A run like ``2026 08 10 2026 08 16`` comes from an example date range in a
    prompt and appears verbatim in any honest answer about that week, so it
    carries no instruction; requiring words keeps the check on prose.
    """
    words = _words(text)
    out = set()
    for i in range(len(words) - SHINGLE_WORDS + 1):
        run = tuple(words[i:i + SHINGLE_WORDS])
        if sum(1 for word in run if not word.isdigit()) >= SHINGLE_WORDS - 2:
            out.add(run)
    return frozenset(out)


def register(*texts: str) -> None:
    """Register fixed steering text. Idempotent; whitespace-insensitive."""
    with _LOCK:
        for text in texts:
            if isinstance(text, str) and text.strip():
                key = " ".join(_words(text))
                if key not in _TEXTS:
                    _TEXTS[key] = _shingles(text)


def registered_texts() -> tuple[str, ...]:
    with _LOCK:
        return tuple(_TEXTS)


def register_tool_names(*names: str) -> None:
    with _LOCK:
        _EXTRA_TOOL_NAMES.update(n for n in names if isinstance(n, str))


def tool_names() -> frozenset:
    """Every tool name a user must never be shown (underscore names only)."""
    with _LOCK:
        names = set(_EXTRA_TOOL_NAMES)
    try:
        from . import llm, mcp_server
        names.update(llm.COACH_TOOLS)
        names.update(llm.RESEARCHER_TOOLS)
        names.update(fn.__name__ for fn in mcp_server._TOOLS)
        names.update(llm.registered_host_tools())
    except Exception:                                # noqa: BLE001
        # A guard that cannot enumerate must still guard with what it has.
        pass
    return frozenset(name for name in names if "_" in name)


def leak(text) -> str | None:
    """Return why ``text`` repeats internal instructions, or ``None``.

    The returned string is a short label naming the kind of match; it
    contains none of the matched text, so it is safe to log.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if _MARKER_RE.search(text):
        return "marker phrase"
    lowered = text.lower()
    for name in tool_names():
        if re.search(r"(?<![a-z0-9_])" + re.escape(name.lower())
                     + r"(?![a-z0-9_])", lowered):
            return "internal tool name"
    words = _words(text)
    if len(words) >= SHINGLE_WORDS:
        seen = {tuple(words[i:i + SHINGLE_WORDS])
                for i in range(len(words) - SHINGLE_WORDS + 1)}
        with _LOCK:
            registered = list(_TEXTS.values())
        for shingles in registered:
            if seen & shingles:
                return "repeated instruction text"
    return None
