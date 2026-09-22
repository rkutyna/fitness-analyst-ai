"""Closed, Python-owned facts and safe template rendering for the ask path.

The model is allowed to choose words and placeholder locations only.  This
module publishes facts from the current call's result ledger, keeps the
ordinary ledger's ``(metric, period, field)`` or attachment table identity in
every key, and interpolates the already-published presentation string without
reformatting it.
"""
from __future__ import annotations

import copy
import json
import re
from datetime import date, timedelta
from urllib.parse import quote, unquote

from . import claim_contract as _CLAIM_CONTRACT
from . import deepdive_verify as _verify
from . import metrics
from . import normalize


_KEY_SEPARATOR = "|"
_KEY_PART_RE = re.compile(r"^(metric|period|field)=(.*)$")
_ATTACHMENT_KEY_PART_RE = re.compile(r"^(table|column|row|trend)=(.*)$")
_WORKOUT_KEY_PART_RE = re.compile(r"^(workout|field)=(.*)$")
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
_ADVICE_PREFIX = "advice:"
_COLD_START_FIELDS = frozenset({
    "status_text", "starts_on_day", "day_now", "starts_on_date", "status",
})
_COLD_START_REFUSAL_STATUSES = frozenset({
    "establishing_baseline", "insufficient_history", "insufficient_data",
    "partial", "nothing_moved",
})
_COLD_START_PATH_RE = re.compile(
    r"^\$\.result\.([^.]+)\.cold_start\.([^\.]+)$"
)


def _advice_metric_names(facts: dict[str, dict] | None) -> list[str]:
    """Return metric spellings that an advice slot must not smuggle through."""
    names = set(normalize.known_metrics())
    for key in (facts or {}):
        parsed = parse_fact_key(key)
        if parsed is not None:
            names.add(parsed[0])
    return sorted(names, key=len, reverse=True)


def _advice_violation(content: str, facts: dict[str, dict] | None) -> str:
    """Reject advice text that turns the coaching exemption into a data claim."""
    if re.search(r"\byour\b", content, re.IGNORECASE):
        return "advice slot references the user's own data"
    for metric in _advice_metric_names(facts):
        words = re.escape(metric).replace(r"_", r"(?:[_ -]+)")
        if re.search(r"(?<![\w])" + words + r"(?![\w])",
                     content, re.IGNORECASE):
            return "advice slot references vault metric " + metric
    return ""


# --- Number words (#49) -------------------------------------------------------
#
# A digit outside a placeholder is refused, so a model told twice that digits
# are forbidden has one obvious evasion: spell the figure. "You ran twelve
# miles" and "fifty-two beats per minute" carry exactly the claim "12" and
# "52" would, and nothing grounds them. This closed, Python-owned list gives
# the spelled forms the digit treatment, in prose outside slots only.
#
# What is deliberately NOT in it, and why:
#   * "one" alone. It is a pronoun far more often than a count ("one of the
#     best", "the one you asked about", "no one"). It counts only when a unit
#     or count noun follows it ("one mile", "one more run").
#   * "first" and "second". "First," opens sentences and "second" is a unit
#     and a rank; both are overwhelmingly non-quantitative. "third" onwards
#     carry a count or a date ("your third run", "on the twelfth").
#   * "once", "a couple", "a few", "several". Vague quantifiers state no
#     figure a reader could check.
#   * "half" alone ("half-marathon" is an event, "the second half of the
#     week" a window). It counts in "and a half" and "half a/an <noun>".
# Known metric spellings are removed before the scan, as the shared numeric
# tokenizer's contract requires of callers with a closed name vocabulary:
# "six minute walk test distance" names a metric, it does not state a six.
_CARDINAL_WORDS = (
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "hundreds",
    "thousand", "thousands", "million", "dozen", "dozens", "twice", "thrice",
)
_ORDINAL_WORDS = (
    "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
    "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth",
    "eightieth", "ninetieth", "hundredth", "thousandth",
)
_COUNT_NOUNS = (
    r"(?:mile|kilometer|kilometre|km|minute|min|hour|hr|second|sec|day|"
    r"night|week|month|year|run|jog|walk|ride|session|workout|rep|set|lap|"
    r"step|beat|bpm|pound|lb|kilogram|kg|percent|point|time)s?"
)
_NUMBER_WORD_RE = re.compile(
    r"(?<![\w-])(?:"
    + "|".join(sorted(_CARDINAL_WORDS + _ORDINAL_WORDS, key=len,
                      reverse=True))
    + r")(?![\w])"
    r"|(?<![\w-])one\s+(?:(?:more|extra|full|whole|single|hard|easy|long|"
    r"short|quick|rest|last)\s+)?" + _COUNT_NOUNS + r"(?![\w])"
    r"|\band\s+a\s+half\b|(?<![\w-])half\s+an?\s+[a-z]+",
    re.IGNORECASE)
_ZERO_IN_RE = re.compile(r"\bzero\s+in\b", re.IGNORECASE)


def _metric_spelling_patterns(facts: dict[str, dict] | None) -> list[str]:
    """Regex spellings of every metric name, ``_`` matching space or hyphen."""
    return [re.escape(metric).replace(r"_", r"(?:[_ -]+)")
            for metric in _advice_metric_names(facts)]


def number_words(text: str, facts: dict[str, dict] | None = None) -> list[str]:
    """Return the spelled-number spans in *text*, in order (#49).

    Callers pass prose with placeholders already removed. Public so the
    repair prompt can name the offending spans, exactly as it does digits.
    """
    scan = text or ""
    for words in _metric_spelling_patterns(facts):
        scan = re.sub(r"(?<![\w])" + words + r"(?![\w])", " ", scan,
                      flags=re.IGNORECASE)
    scan = _ZERO_IN_RE.sub(" ", scan)
    return [match.group(0) for match in _NUMBER_WORD_RE.finditer(scan)]


# --- Conversational assertions (#69 item 3) -------------------------------------
#
# A conversational reply (no ledger, no tool call, allowlisted message) is
# exempt from numeric verification because it states nothing about the vault.
# The rule that enforced that used to refuse any canonical metric spelling.
# Measured on the deployed shape (#69, 36 samples) it refused 7 replies and
# all 7 were capability menus ("ask me about sleep, heart rate, training
# load"), while "You ran twelve miles on Saturday." passed because it named no
# canonical spelling. Naming a metric is not the risk; asserting a fact about
# the user's data is. A claim needs a subject and a predicate, so this rule
# refuses the three shapes a claim about the user takes:
#
#   1. The user's data is the SUBJECT of a statement: "your", a metric
#      spelling, a data noun ("sleep", "training") or a recent-time reference
#      ("last night", "this week"), then up to five words with no punctuation,
#      then a stative, change or reporting verb ("is", "has", "looks",
#      "dropped", "improving"). "Your sleep was better" is refused; "ask me
#      about your sleep, heart rate or runs" is admitted, because the comma
#      ends the phrase before any verb arrives: the metric is a topic on
#      offer, not a subject.
#   2. The user is the subject of a past, perfect, habitual or progressive
#      verb: "you ran", "you've been running", "you usually", "you're
#      recovering". "You're welcome", "would you like" and "if you want" are
#      admitted; a conditional or wh- lead ("if you slept badly", "what you
#      decided") states a hypothetical, not a fact. A progressive aimed at a
#      goal ("a goal you're working toward") is an intention and is admitted.
#   3. Praise for a recorded result: "great job", "well done", "personal
#      best", "new record". Congratulation presupposes the result happened.
#
# Shapes 1 and 2 are read one CLAUSE at a time (split at . ! ? ; : , brackets,
# line breaks and dashes), and a clause that OPENS with how/what/whether/if/
# when/which/why/where is an indirect question or a hypothetical and is
# skipped: "- How your recent training load is trending" and "what would you
# like to know — how your training's been going" offer a topic. Measured
# 2026-09-22 on the live battery: all four replies still refused after the
# first version of this rule were capability menus, three of them this shape.
# The lead must open the clause; "I noticed how your sleep improved" is still
# refused. Quoted example questions and sentences ending in "?" are NOT
# exempt: neither fired on the recorded set, and "Did you know your resting
# heart rate dropped?" is a claim in question form.
#
# Digits and number words stay refused unconditionally, through
# ``scan_template`` and ``number_words``, before any of this runs. This is a
# closed pattern list and it fails open on shapes it does not name ("You run
# more on weekends"); the bias is toward refusing, because a refused greeting
# costs a fallback card and a fabricated claim costs the user's trust. It is
# used on the conversational path ONLY: ledger-backed narration asserts facts by design, through
# placeholders, and is governed by ``scan_template`` alone.
_ASSERT_VERBS = (
    r"(?:is|was|are|were|has|have|had|looks?|looked|looking|seems?|seemed|"
    r"appears?|appeared|remains?|remained|stays?|stayed|improved|improves|"
    r"improving|dropped|drops|dropping|rose|rises|rising|fell|falls|falling|"
    r"increased|increasing|decreased|decreasing|declined|declining|climbed|"
    r"climbing|dipped|spiked|jumped|trended|trending|went\s+(?:up|down)|"
    r"came\s+(?:up|down|in)|shows?|showed|suggests?|suggested|indicates?|"
    r"indicated|averaged|peaked|hit|reached|topped|got|gets|getting)"
)
_DATA_NOUNS = (
    r"sleep|recovery|training|readiness|fitness|progress|pace|mileage|"
    r"volume|hrv|last\s+night|yesterday|today|this\s+(?:week|morning|month)"
)
_HYPOTHETICAL_LEADS = frozenset({
    "if", "when", "whenever", "whether", "what", "whatever", "unless", "once",
    "how",
})
_NOT_PAST = r"(?!(?:need|feed|seed|speed|proceed|exceed|succeed)\b)"
_USER_ACTION_RE = re.compile(
    r"(?:\b(?P<lead>[a-z]+)\s+)?\byou(?:"
    # perfect: you've been / you have logged / you had run
    r"(?:['’]ve|\s+have|\s+had|['’]d)\s+(?:been|done|gone|got|gotten|had|"
    r"made|run|slept|hit|set|kept|" + _NOT_PAST + r"[a-z]{2,}ed)"
    # simple past or habitual, with an optional frequency/stance adverb
    r"|\s+(?:(?:usually|often|always|typically|generally|consistently|"
    r"regularly|rarely|never|clearly|really|just|also|definitely)\s+)?"
    r"(?:ran|slept|hit|did|were|was|went|got|rode|swam|beat|made|took|kept|"
    r"felt|held|lost|won|tend|seem|" + _NOT_PAST + r"[a-z]{2,}ed)"
    # progressive or state: you're recovering / you are on track
    r"|(?:['’]re|\s+are)\s+(?:(?:really|clearly|definitely|still|now)\s+)?"
    # An -ing verb aimed at a goal ("a goal you're working toward", "what
    # you're training for") states an intention, not a measurement.
    r"(?:[a-z]+ing(?![\w])(?!\s+(?:toward|towards|for|to|at|on)\b)|"
    r"on\s+track|ahead|behind|in\s+(?:good|great|solid)\s+shape|"
    r"recovered|fitter|faster|stronger)"
    r")(?![\w])",
    re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;:,()\n\u2014\u2013]+|\s-\s")
_CLAUSE_MARKUP_RE = re.compile(r"^[\s*#>\"'\u201c\u201d\u2022-]+")
_INDIRECT_QUESTION_LEADS = frozenset({
    "how", "what", "whether", "if", "when", "whenever", "which", "why",
    "where", "whatever", "unless",
})
_PRAISE_RE = re.compile(
    r"\b(?:(?:great|good|nice|awesome|amazing|excellent)\s+job|well\s+done|"
    r"congrat\w*|personal\s+(?:best|record)|new\s+(?:best|record|pb|pr))\b",
    re.IGNORECASE)


def _data_subject_re(facts: dict[str, dict] | None) -> re.Pattern:
    heads = "|".join(["your", _DATA_NOUNS] + _metric_spelling_patterns(facts))
    return re.compile(
        r"(?<![\w])(?:" + heads + r")(?![\w])"
        r"(?:\s+\w[\w-]*){0,5}?"
        r"(?:\s+" + _ASSERT_VERBS + r"|['’]s\s+(?:been|looking|trending|"
        r"improving|dropping|getting|up|down))(?![\w])",
        re.IGNORECASE)


def conversational_assertion(text: str,
                             facts: dict[str, dict] | None = None) -> str:
    """Return why *text* asserts something about the user's data, or ``""``.

    The three refused shapes and the reasoning are in the comment block above
    ``_ASSERT_VERBS``. In short: a metric named as a topic on offer is
    admitted; the user or the user's data as the subject of a statement is
    refused.
    """
    text = (text if isinstance(text, str) else "").replace("\u2019", "'")
    subject_re = _data_subject_re(facts)
    for clause in _CLAUSE_SPLIT_RE.split(text):
        clause = _CLAUSE_MARKUP_RE.sub("", clause)
        first = re.match(r"[a-z]+", clause, re.IGNORECASE)
        if first and first.group(0).lower() in _INDIRECT_QUESTION_LEADS:
            continue
        if subject_re.search(clause):
            return ("conversational answer makes the user data the subject "
                    "of a statement")
        for match in _USER_ACTION_RE.finditer(clause):
            if (match.group("lead") or "").lower() not in _HYPOTHETICAL_LEADS:
                return "conversational answer states something the user did"
    if _PRAISE_RE.search(text):
        return "conversational answer praises a recorded result"
    return ""


def _period_token(period) -> str:
    if isinstance(period, str):
        return "s:" + period
    return "j:" + json.dumps(period, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, default=str)


def _period_from_token(token: str):
    if token.startswith("s:"):
        return token[2:]
    if token.startswith("j:"):
        try:
            return json.loads(token[2:])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return None


def _period_identity(period) -> str:
    """Canonical identity used to compare period objects without guessing."""
    return _period_token(period)


_PERIOD_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_MONTH_ABBREVIATIONS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _period_date(value: object) -> date | None:
    """Parse only the ISO day spelling accepted as a period component."""
    if not isinstance(value, str) or not _PERIOD_DAY_RE.fullmatch(value):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _short_period_day(value: date) -> str:
    weekdays = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    return f"{weekdays[value.weekday()]} {_MONTH_ABBREVIATIONS[value.month - 1]} {value.day}"


def _full_period_day(value: date) -> str:
    return f"{_MONTH_NAMES[value.month - 1]} {value.day}"


def _date_range_period_label(start: date, end: date) -> str | None:
    """Name a validated inclusive range without inferring its metric."""
    if end < start:
        return None
    if end - start == timedelta(days=6):
        return f"the week of {_full_period_day(start)}"
    return f"from {_short_period_day(start)} to {_short_period_day(end)}"


def _period_label(period) -> str | None:
    """Return a human label only for period shapes with explicit date meaning.

    Weekly block periods carry their bucket starts, so their count and cadence
    can be named directly (for example, ``the last 4 weeks``). Other shapes
    are labelled from their explicit day or inclusive date range. Unknown or
    malformed shapes return ``None`` rather than turning arbitrary structure
    into a guessed date.
    """
    if isinstance(period, str):
        day = _period_date(period)
        if day is not None:
            return _short_period_day(day)
        match = re.fullmatch(
            r"(\d{4}-\d{2}-\d{2}):(\d{4}-\d{2}-\d{2})", period)
        if not match:
            return None
        start, end = (_period_date(match.group(index)) for index in (1, 2))
        if start is None or end is None:
            return None
        return _date_range_period_label(start, end)

    if isinstance(period, dict):
        starts_raw = period.get("period_starts")
        if starts_raw is not None:
            if not isinstance(starts_raw, list) or not starts_raw:
                return None
            starts = [_period_date(value) for value in starts_raw]
            if any(value is None for value in starts):
                return None
            starts = [value for value in starts if value is not None]
            if len(starts) > 1:
                steps = [(right - left).days
                         for left, right in zip(starts, starts[1:])]
                if all(step == 7 for step in steps):
                    return f"the last {len(starts)} weeks"
                if all(step == 1 for step in steps):
                    return f"the last {len(starts)} days"
                return None

        start = _period_date(period.get("start"))
        end = _period_date(period.get("end"))
        if start is None or end is None:
            return None
        return _date_range_period_label(start, end)

    if isinstance(period, (list, tuple)) and period:
        starts = [_period_date(value) for value in period]
        if any(value is None for value in starts):
            return None
        starts = [value for value in starts if value is not None]
        if len(starts) < 2:
            return None
        steps = [(right - left).days
                 for left, right in zip(starts, starts[1:])]
        if all(step == 7 for step in steps):
            return f"the last {len(starts)} weeks"
        if all(step == 1 for step in steps):
            return f"the last {len(starts)} days"
    return None


def _add_period_label_facts(facts: dict[str, dict]) -> None:
    """Add one Python-owned label leaf for each closed metric/period pair."""
    seen: set[tuple[str, str]] = set()
    for fact in list(facts.values()):
        metric = fact.get("metric")
        period = fact.get("period")
        if _CLAIM_CONTRACT.is_metricless_metric(metric) or period is None:
            continue
        identity = (str(metric), _period_identity(period))
        if identity in seen:
            continue
        seen.add(identity)
        label = _period_label(period)
        if label is None:
            continue
        key = fact_key(metric, period, "period_label")
        if key in facts:
            continue
        facts[key] = {
            "key": key,
            "metric": metric,
            "period": copy.deepcopy(period),
            "field": "period_label",
            "value": label,
            "unit": None,
            "display": label,
            "source": {"sequence": (fact.get("source") or {}).get("sequence"),
                        "period": copy.deepcopy(period)},
        }


def fact_key(metric: str, period, field: str) -> str:
    """Return an unambiguous key derived only from a ledger identity tuple.

    Components are labeled and percent-escaped.  Strings stay readable (for
    example, a date period remains visible); structured periods retain their
    complete JSON, so parsing a key cannot collapse two ledger periods into
    one.  The ``fact`` prefix makes accidental ordinary prose placeholders
    distinguishable from published keys.
    """
    if not str(metric).strip() or not str(field).strip() or period is None:
        raise ValueError("metric, period, and field are required")
    enc = lambda value: quote(str(value), safe="-_.~:")
    return (_KEY_SEPARATOR.join(("fact", "metric=" + enc(metric),
                                 "period=" + enc(_period_token(period)),
                                 "field=" + enc(field))))


def parse_fact_key(key: str) -> tuple[str, object, str] | None:
    """Parse a key made by :func:`fact_key`, or return ``None``."""
    if not isinstance(key, str):
        return None
    parts = key.split(_KEY_SEPARATOR)
    if len(parts) != 4 or parts[0] != "fact":
        return None
    values = {}
    for part in parts[1:]:
        match = _KEY_PART_RE.match(part)
        if not match:
            return None
        values[match.group(1)] = unquote(match.group(2))
    if set(values) != {"metric", "period", "field"}:
        return None
    period = _period_from_token(values["period"])
    if period is None:
        return None
    return values["metric"], period, values["field"]


def attachment_fact_key(table: str, column: str, row_key) -> str:
    """Return a key for one verbatim cell in an analyst result table."""
    if not str(table).strip() or not str(column).strip():
        raise ValueError("table and column are required")
    enc = lambda value: quote(str(value), safe="-_.~:")
    return _KEY_SEPARATOR.join((
        "fact", "table=" + enc(table), "column=" + enc(column),
        "row=" + enc(row_key),
    ))


def parse_attachment_fact_key(key: str) -> tuple[str, str, str] | None:
    """Parse a key made by :func:`attachment_fact_key`, or return ``None``."""
    if not isinstance(key, str):
        return None
    parts = key.split(_KEY_SEPARATOR)
    if len(parts) != 4 or parts[0] != "fact":
        return None
    values = {}
    for part in parts[1:]:
        match = _ATTACHMENT_KEY_PART_RE.match(part)
        if not match:
            return None
        values[match.group(1)] = unquote(match.group(2))
    if set(values) != {"table", "column", "row"}:
        return None
    return values["table"], values["column"], values["row"]


def attachment_trend_key(table: str, column: str, stat: str) -> str:
    """Return a key for a Python-computed table trend statistic.

    ``stat`` is one of ``first``, ``last``, ``delta``, or ``direction``.
    Direction facts use ``increased``, ``decreased``, or ``unchanged`` based
    only on the sign of ``last - first``.
    """
    if not str(table).strip() or not str(column).strip():
        raise ValueError("table and column are required")
    if stat not in {"first", "last", "delta", "direction"}:
        raise ValueError("unknown trend statistic")
    enc = lambda value: quote(str(value), safe="-_.~:")
    return _KEY_SEPARATOR.join((
        "fact", "table=" + enc(table), "column=" + enc(column),
        "trend=" + enc(stat),
    ))


def parse_attachment_trend_key(key: str) -> tuple[str, str, str] | None:
    """Parse a key made by :func:`attachment_trend_key`, or return ``None``."""
    if not isinstance(key, str):
        return None
    parts = key.split(_KEY_SEPARATOR)
    if len(parts) != 4 or parts[0] != "fact":
        return None
    values = {}
    for part in parts[1:]:
        match = _ATTACHMENT_KEY_PART_RE.match(part)
        if not match:
            return None
        values[match.group(1)] = unquote(match.group(2))
    if set(values) != {"table", "column", "trend"}:
        return None
    if values["trend"] not in {"first", "last", "delta", "direction"}:
        return None
    return values["table"], values["column"], values["trend"]


def workout_fact_key(workout: str, field: str) -> str:
    """Return a key for one field of a single workout session.

    A workout is not a metric series (it has no ``metric``/``period`` pair to
    key on), so its identity is the tool-reported session date instead --
    the one identifier every run-answering tool already returns.
    """
    if not str(workout).strip() or not str(field).strip():
        raise ValueError("workout and field are required")
    enc = lambda value: quote(str(value), safe="-_.~:")
    return _KEY_SEPARATOR.join((
        "fact", "workout=" + enc(workout), "field=" + enc(field),
    ))


def parse_workout_fact_key(key: str) -> tuple[str, str] | None:
    """Parse a key made by :func:`workout_fact_key`, or return ``None``."""
    if not isinstance(key, str):
        return None
    parts = key.split(_KEY_SEPARATOR)
    if len(parts) != 3 or parts[0] != "fact":
        return None
    values = {}
    for part in parts[1:]:
        match = _WORKOUT_KEY_PART_RE.match(part)
        if not match:
            return None
        values[match.group(1)] = unquote(match.group(2))
    if set(values) != {"workout", "field"}:
        return None
    return values["workout"], values["field"]


def _path_parts(path: str) -> list[str | int]:
    """Parse the JSON path form emitted by ``_ledger_scopes``."""
    if not isinstance(path, str) or not path.startswith("$.result"):
        return []
    parts: list[str | int] = []
    for token in re.finditer(r"\.([^.\[]+)|\[(\d+)\]", path[len("$.result"):]):
        parts.append(token.group(1) if token.group(1) is not None
                    else int(token.group(2)))
    return parts


def _unit_by_path(record: dict) -> dict[str, str]:
    """Collect payload units without deriving or converting a measurement."""
    result = record.get("result")
    found: dict[str, str] = {}

    def walk(node, path: tuple[str | int, ...], inherited: str | None = None):
        unit = node.get("unit", inherited) if isinstance(node, dict) else inherited
        if isinstance(unit, str) and unit:
            found[_json_path(path)] = unit
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, path + (key,), unit)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, path + (index,), unit)

    walk(result, ())
    return found


def _json_path(parts: tuple[str | int, ...]) -> str:
    rendered = "$.result"
    for part in parts:
        rendered += f"[{part}]" if isinstance(part, int) else "." + str(part)
    return rendered


def _unit_for(entry: dict, units: dict[str, str]) -> str:
    parts = _path_parts(entry.get("path", ""))
    for length in range(len(parts), -1, -1):
        unit = units.get(_json_path(tuple(parts[:length])))
        if unit:
            return unit
    return normalize.canonical_unit(str(entry["metric"]), None)


def _display_value(value, *, metric: str | None = None,
                   unit: str | None = None, field: str | None = None,
                   signed: bool = False) -> str:
    """Return the one published display string for a numeric fact.

    Metric facts use the canonical renderer. Attachment facts have explicit
    table units but no canonical metric, so they use the shared unit renderer.
    A non-unit-preserving metric field deliberately remains unitless. Numeric
    values with no applicable renderer still get deterministic rounding rather
    than Python's float repr; the owned ``value`` is never changed.
    """
    if metric is not None:
        rendered = metrics.format_presentation(metric, value, field=field or "value")
        if rendered is not None:
            return rendered
        signed = signed or field in metrics._SIGNED_UNIT_PRESERVING_FIELDS
    elif unit:
        rendered = metrics.format_unit_value(value, unit, signed=signed)
        if rendered is not None:
            return rendered
    rendered = metrics.format_numeric(value, signed=signed)
    return rendered if rendered is not None else str(value)


def _same_scope(left: dict, right: dict) -> bool:
    return (left.get("metric") == right.get("metric")
            and _period_identity(left.get("period"))
            == _period_identity(right.get("period")))


def _presentation_for(raw: dict, presentations: list[dict]) -> dict | None:
    """Find the exact sibling presentation leaf for one raw field."""
    raw_path = str(raw.get("path") or "")
    raw_parent = raw_path.rsplit(".", 1)[0]
    field = str(raw.get("field") or "")
    candidates = [entry for entry in presentations
                  if entry.get("metric") == raw.get("metric")
                  and (raw.get("period") is None or _same_scope(raw, entry))
                  and isinstance(entry.get("value"), str)]
    exact = []
    for entry in candidates:
        path = str(entry.get("path") or "")
        if (path == raw_parent + ".presentation.value"
                or path == raw_parent + ".presentations." + field + ".value"):
            exact.append(entry)
    if len(exact) == 1:
        return exact[0]
    # A result may publish one canonical presentation for a metric/period
    # without a field-specific presentations map.  A sole candidate is safe;
    # multiple candidates are intentionally not guessed across.
    return candidates[0] if len(candidates) == 1 else None


def _weekly_period_for_entry(record: dict, entry: dict) -> str | None:
    """Recover a weekly mean's period from its enclosing ``week_start``."""
    if entry.get("field") != "mean":
        return None
    week_start = entry.get("week_start")
    if _period_date(week_start) is not None:
        return week_start
    path = _path_parts(entry.get("path", ""))
    node = record.get("result")
    try:
        for part in path[:-1]:
            node = node[part]
    except (KeyError, IndexError, TypeError):
        return None
    week_start = node.get("week_start") if isinstance(node, dict) else None
    return week_start if _period_date(week_start) is not None else None


def _publish_unambiguous(candidates: list[tuple[str, dict]]) -> dict[str, dict]:
    """Collapse duplicate keys when, and only when, their values agree.

    ``value`` is the fact's owned measurement. Presentation, units, source,
    and all other record metadata are intentionally excluded from identity.
    Equality is evaluated on the owned values themselves, not on their
    surrounding records.

    When the values agree but their presentations differ, the **first
    candidate's** record is published -- first by ledger order. That choice is
    user-visible: it decides which display string reaches the model, so two
    tool calls returning one value in different units would otherwise make the
    narrated string depend on call order. It is pinned by test rather than left
    implicit, so a later reordering of candidates fails loudly instead of
    quietly changing what a user reads.
    """
    grouped: dict[str, list[dict]] = {}
    for key, fact in candidates:
        grouped.setdefault(key, []).append(fact)

    published: dict[str, dict] = {}
    for key, facts in grouped.items():
        value = facts[0].get("value")
        if all(fact.get("value") == value for fact in facts[1:]):
            published[key] = facts[0]
    return published


def _cold_start_entries(entries: list[dict], *, sequence=None) -> list[tuple[str, dict]]:
    """Return refusal-only cold-start leaves as exact JSON-path facts.

    Cold-start values are structural facts, not metric-series values: they have
    no metric or period identity. Their ledger JSON path is nevertheless a
    stable, source-backed identity and is the spelling the template can copy.
    Only the five public leaves needed to explain a refusal enter the closed
    set; nearby ``reason`` and ``as_of`` context remains out of the narration
    vocabulary.
    """
    by_path = {
        str(entry.get("path")): entry for entry in entries
        if isinstance(entry, dict)
    }
    candidates: list[tuple[str, dict]] = []
    for path, status_entry in by_path.items():
        match = _COLD_START_PATH_RE.fullmatch(path)
        if not match or match.group(2) != "status":
            continue
        if status_entry.get("value") not in _COLD_START_REFUSAL_STATUSES:
            continue
        surface = match.group(1)
        prefix = f"$.result.{surface}.cold_start."
        for field in _COLD_START_FIELDS:
            leaf_path = prefix + field
            entry = by_path.get(leaf_path)
            if entry is None or entry.get("value") is None:
                continue
            candidates.append((leaf_path, {
                "key": leaf_path,
                "path": leaf_path,
                "surface": surface,
                "field": field,
                "metric": None,
                "period": None,
                "value": entry["value"],
                "unit": None,
                "display": _display_value(entry["value"]),
                "source": {"sequence": sequence,
                           "path": leaf_path},
            }))
    return candidates


def _weekly_period_for_eligibility(record: dict, entry: dict) -> str | None:
    """Recover a weekly mean period without reading the publisher helpers."""
    if entry.get("field") != "mean":
        return None
    path = _path_parts(entry.get("path", ""))
    node = record.get("result")
    try:
        for part in path[:-1]:
            node = node[part]
    except (KeyError, IndexError, TypeError):
        return None
    period = node.get("week_start") if isinstance(node, dict) else None
    return period if _period_date(period) is not None else None


def _presentation_period_for_eligibility(entry: dict,
                                         presentations: list[dict]):
    """Resolve a raw leaf's period from its one unambiguous sibling."""
    candidates = [candidate for candidate in presentations
                  if candidate.get("metric") == entry.get("metric")
                  and isinstance(candidate.get("value"), str)]
    raw_path = str(entry.get("path") or "")
    raw_parent = raw_path.rsplit(".", 1)[0]
    exact = [candidate for candidate in candidates
             if candidate.get("path") in {
                 raw_parent + ".presentation.value",
                 raw_parent + ".presentations."
                 + str(entry.get("field") or "") + ".value",
             }]
    if len(exact) == 1:
        return exact[0].get("period")
    return candidates[0].get("period") if len(candidates) == 1 else None


def eligible_fact_keys(ledger: list[dict]) -> set[str]:
    """Return independently re-derived metric keys eligible for publication.

    This walk is intentionally independent of :func:`build_fact_set`'s
    candidate grouping and publishing helpers. It reads result entries and
    re-derives eligibility from their metric, value, period, and constructible
    key. That independence is the instrument: two readers that can disagree
    can expose a publisher that is broken instead of checking itself. It is
    also required for historical validation: ``_publish_unambiguous`` did not
    exist before #42, so a check written against that helper could never run
    against the code state where the bug lived.

    Duplicate identities are eligible only when every owned value agrees.
    A conflicting identity is ambiguous evidence, not a withholding event.
    """
    if not isinstance(ledger, list):
        return set()

    values_by_key: dict[str, list[object]] = {}
    cold_values_by_path: dict[str, list[object]] = {}
    for record in ledger:
        if not isinstance(record, dict) or record.get("result_elided"):
            continue
        try:
            entries = _verify._ledger_scopes(record)
        except (AttributeError, TypeError, ValueError):
            continue
        # Cold-start facts are deliberately checked separately from metric
        # ownership: they have no metric/period identity, but their paths are
        # still eligible, source-backed publication keys.
        by_path = {str(entry.get("path")): entry for entry in entries}
        for path, status_entry in by_path.items():
            match = _COLD_START_PATH_RE.fullmatch(path)
            if not match or match.group(2) != "status":
                continue
            if status_entry.get("value") not in _COLD_START_REFUSAL_STATUSES:
                continue
            prefix = f"$.result.{match.group(1)}.cold_start."
            for field in _COLD_START_FIELDS:
                leaf = by_path.get(prefix + field)
                if leaf is not None and leaf.get("value") is not None:
                    cold_values_by_path[prefix + field] = (
                        cold_values_by_path.get(prefix + field, [])
                        + [leaf.get("value")])
        presentations = [entry for entry in entries
                         if entry.get("field") == "presentation"]
        for entry in entries:
            if (entry.get("kind") != "result"
                    or entry.get("field") == "presentation"):
                continue
            metric = entry.get("metric")
            if metric is None or not str(metric).strip():
                continue
            value = entry.get("value")
            if value is None:
                continue
            period = entry.get("period")
            if period is None:
                period = _presentation_period_for_eligibility(
                    entry, presentations)
            if period is None:
                period = _weekly_period_for_eligibility(record, entry)
            if period is None:
                continue
            try:
                key = fact_key(metric, period, entry.get("field"))
            except (TypeError, ValueError):
                continue
            values_by_key.setdefault(key, []).append(value)

    return {
        key for key, values in values_by_key.items()
        if all(value == values[0] for value in values[1:])
    } | {
        path for path, values in cold_values_by_path.items()
        if all(value == values[0] for value in values[1:])
    }


def publish_completeness(ledger: list[dict], published_facts) -> set[str]:
    """Return eligible keys missing from the fact set actually published.

    ``published_facts`` is supplied by the caller so this check can compare
    the real publisher output with the independent eligibility walk. A
    returned key is a withholding event; an empty set means the two readers
    agree for this ledger.
    """
    published_keys = set(published_facts) if isinstance(published_facts, dict) \
        else set(published_facts or ())
    return eligible_fact_keys(ledger) - published_keys


def build_fact_set(ledger: list[dict]) -> dict[str, dict]:
    """Build the closed fact set from result leaves in this call's ledger.

    Only metric-owned result leaves with an explicit period participate;
    weekly mean leaves may use their enclosing ``week_start``.
    Arguments, context fields, elided results, and metricless workout rows are
    excluded because they cannot round-trip through the natural identity tuple.
    Duplicate identities publish when their owned values **agree**, and are
    withheld only when they disagree (#42). A key never chooses between two
    different values -- but two calls returning the same value, which a retry
    routinely produces, no longer silence it. Where the values agree and their
    presentations differ, the first candidate by ledger order is published;
    ``_publish_unambiguous`` states why that choice is user-visible and pins it.

    This sentence used to state the pre-#42 rule -- that duplicates were simply
    removed -- two lines above code doing the opposite (#45). It was not an
    ordinary stale comment: health_advisor#272's diagnosis turned on an empty
    fact set, and a reader auditing that path was told here that the emptying
    was deliberate policy.
    """
    if not isinstance(ledger, list):
        return {}
    candidates: list[tuple[dict, dict, list[dict], dict[str, str]]] = []
    for record in ledger:
        if not isinstance(record, dict) or record.get("result_elided"):
            continue
        try:
            entries = [entry for entry in _verify._ledger_scopes(record)
                       if entry.get("kind") == "result"]
        except (AttributeError, TypeError, ValueError):
            continue
        presentations = [entry for entry in entries
                         if entry.get("field") == "presentation"]
        units = _unit_by_path(record)
        for entry in entries:
            if (_CLAIM_CONTRACT.is_metricless_metric(entry.get("metric"))
                    or entry.get("field") == "presentation"
                    or entry.get("value") is None):
                continue
            presentation = _presentation_for(entry, presentations)
            period = (entry["period"] if entry["period"] is not None
                      else (presentation or {}).get("period"))
            if period is None:
                period = _weekly_period_for_entry(record, entry)
            if period is None:
                continue
            entry = {**entry, "period": period,
                     "_presentation": presentation}
            try:
                fact_key(entry["metric"], entry["period"], entry["field"])
            except (TypeError, ValueError):
                continue
            candidates.append((entry, {"sequence": record.get("sequence"),
                                       "path": entry.get("path")},
                               presentations, units))

    resolved_candidates: list[tuple[str, dict]] = []
    for entry, source, presentations, units in candidates:
        key = fact_key(entry["metric"], entry["period"], entry["field"])
        presentation = entry.get("_presentation")
        if presentation is None:
            presentation = _presentation_for(entry, presentations)
        if presentation is None:
            display = _display_value(
                entry["value"], metric=entry["metric"],
                unit=_unit_for(entry, units), field=entry["field"])
        else:
            display = presentation["value"]
        resolved_candidates.append((key, {
                "key": key,
                "metric": entry["metric"],
                "period": copy.deepcopy(entry["period"]),
                "field": entry["field"],
                "value": entry["value"],
                "unit": _unit_for(entry, units),
                "display": display,
                "source": source,
            }))
    facts = _publish_unambiguous(resolved_candidates)
    cold_candidates: list[tuple[str, dict]] = []
    for record in ledger:
        if not isinstance(record, dict) or record.get("result_elided"):
            continue
        try:
            entries = [entry for entry in _verify._ledger_scopes(record)
                       if entry.get("kind") == "result"]
        except (AttributeError, TypeError, ValueError):
            continue
        cold_candidates.extend(_cold_start_entries(
            entries, sequence=record.get("sequence")))
    facts.update(_publish_unambiguous(cold_candidates))
    _add_period_label_facts(facts)
    return facts


def _key_order(rows) -> str | None:
    """``ascending``/``descending`` if the key column is strictly monotonic.

    Trend facts are meaningful only over a time-ordered series, and the table
    arrives in whatever row order the analyst's code emitted. The key column
    decides: strictly monotonic one way or the other names the chronology (a
    descending table is read newest-first, and its trend facts are computed
    oldest-to-newest regardless); anything else — duplicate keys, mixed
    types, an unordered categorical key — yields ``None`` and the table gets
    cell facts but no trend facts. Without this, a newest-first table would
    flip ``direction`` and narrate an increase over a chronological decline,
    with every value verbatim and no gate able to see it.
    """
    keys = [row[0] for row in rows]
    numeric = all(isinstance(k, (int, float)) and not isinstance(k, bool)
                  for k in keys)
    stringy = all(isinstance(k, str) for k in keys)
    if not (numeric or stringy):
        return None
    if all(a < b for a, b in zip(keys, keys[1:])):
        return "ascending"
    if all(a > b for a, b in zip(keys, keys[1:])):
        return "descending"
    return None


def build_attachment_facts(ledger: list[dict]) -> dict[str, dict]:
    """Build closed facts for analyst table cells and deterministic trends.

    Cell values and units are copied from the table. Numeric displays use the
    shared unit renderer. For numeric
    columns with at least two rows and a strictly monotonic key column,
    ``first``, ``last``, and ``delta`` are Python-owned values computed
    oldest-to-newest, while ``direction`` is the constant ``increased``,
    ``decreased``, or ``unchanged`` selected by the sign of ``delta``.
    Ambiguous duplicate keys are omitted rather than choosing a candidate,
    and a table whose key column is not strictly monotonic publishes cells
    but no trends.
    """
    if not isinstance(ledger, list):
        return {}

    candidates: list[tuple[str, dict]] = []
    for record in ledger:
        # Keep "analyst_query" synchronized with llm.ANALYST_QUERY_NAME
        # without adding a dependency from this module to llm. "run_audit"
        # is chat.run_audit's deterministic battery — its flag table rides
        # the same attachment-fact channel under its own honest name.
        if (not isinstance(record, dict)
                or record.get("tool_name") not in ("analyst_query", "run_audit")):
            continue
        result = record.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("tables"), list):
            continue

        for table_index, table in enumerate(result["tables"]):
            required = ("name", "columns", "units", "rows")
            if (not isinstance(table, dict)
                    or any(field not in table for field in required)):
                continue
            name = table["name"]
            columns = table["columns"]
            units = table["units"]
            rows = table["rows"]
            if (not isinstance(columns, (list, tuple))
                    or not isinstance(units, (list, tuple))
                    or not isinstance(rows, (list, tuple))
                    or len(columns) < 2
                    or len(units) < len(columns)
                    or any(not isinstance(row, (list, tuple))
                           or len(row) < len(columns) for row in rows)):
                continue
            if (not str(name).strip()
                    or any(not str(column).strip() for column in columns)):
                continue

            key_order = _key_order(rows)
            for column_index in range(1, len(columns)):
                column = columns[column_index]
                for row_index, row in enumerate(rows):
                    key = attachment_fact_key(name, column, row[0])
                    path = (f"$.result.tables[{table_index}].rows["
                            f"{row_index}][{column_index}]")
                    candidates.append((key, {
                        "key": key,
                        "table": name,
                        "column": column,
                        "row": row[0],
                        "value": row[column_index],
                        "unit": units[column_index],
                        "display": _display_value(
                            row[column_index], unit=units[column_index]),
                        "source": {"sequence": record.get("sequence"),
                                   "path": path},
                    }))

                numeric = (len(rows) >= 2 and all(
                    isinstance(row[column_index], (int, float))
                    and not isinstance(row[column_index], bool)
                    for row in rows))
                if not numeric or key_order is None:
                    continue
                first_row = 0 if key_order == "ascending" else len(rows) - 1
                last_row = len(rows) - 1 if key_order == "ascending" else 0
                first = rows[first_row][column_index]
                last = rows[last_row][column_index]
                delta = last - first
                direction = ("increased" if delta > 0 else
                             "decreased" if delta < 0 else "unchanged")
                first_path = (f"$.result.tables[{table_index}].rows["
                              f"{first_row}][{column_index}]")
                last_path = (f"$.result.tables[{table_index}].rows["
                             f"{last_row}][{column_index}]")
                trend_source = {
                    "sequence": record.get("sequence"),
                    "paths": [first_path, last_path],
                }
                trend_values = {
                    "first": (first, units[column_index]),
                    "last": (last, units[column_index]),
                    "delta": (delta, units[column_index]),
                    "direction": (direction, None),
                }
                for stat, (value, unit) in trend_values.items():
                    key = attachment_trend_key(name, column, stat)
                    candidates.append((key, {
                        "key": key,
                        "table": name,
                        "column": column,
                        "trend": stat,
                        "value": value,
                        "unit": unit,
                        "display": _display_value(
                            value, unit=unit, signed=(stat == "delta")),
                        "source": trend_source,
                    }))

    return _publish_unambiguous(candidates)


# Tools that answer "how was my run/workout" questions but return
# per-session rows with no metric/period identity (health_advisor#469). Each
# entry below maps a raw JSON field on that tool's result to the published
# workout-fact field name and the unit Python recorded for it.
#
# Two fields deliberately do NOT come from every tool that could name them:
#
# * ``avg_heart_rate`` comes only from ``get_block_structure``'s per-session
#   ``avg_hr_session`` (one decimal place, computed by
#   ``metrics.session_hr_figures`` from the same stored value
#   ``list_workouts`` also reports, but rounded to zero decimals there). Two
#   tools naming the same physical reading at different rounding depths are
#   not "the same value" to :func:`_publish_unambiguous` -- they read as a
#   conflict and the fact is withheld entirely. Picking the one precise
#   source avoids manufacturing that conflict.
# * ``get_briefing``'s ``workout_focus`` restates ``duration_min`` and
#   distance at a coarser rounding than ``list_workouts`` for the same
#   reason, so this module does not republish them from there -- doing so
#   would withhold the more precise ``list_workouts`` figure whenever both
#   tools ran. ``workout_focus`` is kept in the allowlist below (a caller
#   that only ran that one tool should still be able to extend this table)
#   but contributes nothing today.
_WORKOUT_TOOLS = frozenset({
    "list_workouts", "get_run_form", "get_block_structure", "get_briefing",
})
_WORKOUT_ROW_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("duration_min", "duration_min", "min"),
    ("max_heart_rate", "max_heart_rate", "bpm"),
    ("distance_mi", "distance_mi", "mi"),
    ("distance_km", "distance_km", "km"),
)


def _workout_identity(date, workout_type, start_time=None) -> str | None:
    """One workout's identity: its date and type, plus its start time only
    when that is needed to tell two same-type sessions on the same date
    apart (health_advisor#469 follow-up).

    A day is routinely a run AND a walk (or a ride) -- 14 of 17 run days in
    one measured stretch also carried another workout -- so ``date`` alone
    collides constantly and :func:`_publish_unambiguous` would then withhold
    every field on that day as an unresolved conflict. ``date + type`` is
    the floor. It is also what every one of the four run-answering tools can
    name: ``get_run_form`` only ever describes a running session,
    ``get_block_structure``'s ``sessions`` and ``get_briefing``'s
    ``workout_focus`` each carry their own ``workout_type`` -- so two tools
    naming the SAME single workout of a type on a day publish under the SAME
    key without needing a start time at all, and duration from one tool
    joins heart rate from another. Only ``list_workouts`` can return more
    than one workout of the same type on the same date (two walks, or two
    runs); only there is a start time added, and only for the rows that
    actually collide -- see the ambiguity check in the ``list_workouts``
    branch below.
    """
    if not date or not workout_type:
        return None
    parts = [str(date), str(workout_type)]
    if start_time:
        parts.append(str(start_time))
    return "|".join(parts)


def _workout_candidate(workout: str | None, field: str, value, unit: str, *,
                       sequence, path: str) -> tuple[str, dict] | None:
    if workout is None or not str(workout).strip():
        return None
    if value is None or isinstance(value, bool) or not isinstance(
            value, (int, float)):
        return None
    key = workout_fact_key(workout, field)
    return key, {
        "key": key,
        "workout": workout,
        "field": field,
        "value": value,
        "unit": unit,
        "display": _display_value(value, unit=unit),
        "source": {"sequence": sequence, "path": path},
    }


def _workout_row_candidates(row: dict, workout: str | None, *, sequence,
                            path_prefix: str) -> list[tuple[str, dict]]:
    """Duration/distance/max-heart-rate fields shared by workout rows."""
    if not isinstance(row, dict) or workout is None:
        return []
    out = []
    for raw_field, out_field, unit in _WORKOUT_ROW_FIELDS:
        candidate = _workout_candidate(
            workout, out_field, row.get(raw_field), unit,
            sequence=sequence, path=f"{path_prefix}.{raw_field}")
        if candidate is not None:
            out.append(candidate)
    return out


def build_workout_facts(ledger: list[dict]) -> dict[str, dict]:
    """Build closed facts for the run-answering tools' per-session numbers.

    ``list_workouts``, ``get_run_form``, ``get_block_structure``, and
    ``get_briefing``'s workout focus each describe one workout session, not a
    metric series, so their numbers cannot carry a ``(metric, period)``
    identity and :func:`build_fact_set` excludes them by design. This
    publishes an allowlisted set of their fields instead, keyed by the
    workout's own identity (:func:`_workout_identity` --
    date, type, and a start time only where two sessions of one type share a
    date) -- duration, distance, max heart rate, the session's own
    jog-minute count, and the longest continuous running block, wherever a
    call actually returned one. A field a tool did not return for a session
    is simply absent, never invented. Duplicate reports of the same
    workout/field publish only when their values agree, exactly as
    :func:`build_fact_set` handles duplicates -- see the module-level
    comment above ``_WORKOUT_TOOLS`` for the two fields this withholds by
    deliberate design, not by accident.

    Keying by date alone made any day with a second workout -- a walk beside
    the run, most days -- collide every field on that day into one withheld
    identity (health_advisor#469 follow-up). Type is what tells them apart.

    ``session_jog_minutes`` (from ``get_run_form``) is named apart from the
    vault's canonical ``jog_minutes`` metric on purpose: it counts jog
    buckets under ``running_form``'s own halves-comparison rule, which is a
    different classification from the one behind the published metric
    series (``get_impact_volume``), and the two can disagree. Publishing it
    under the metric's name would assert an agreement Python has not
    checked.
    """
    if not isinstance(ledger, list):
        return {}

    candidates: list[tuple[str, dict]] = []
    for record in ledger:
        if (not isinstance(record, dict) or record.get("result_elided")
                or record.get("tool_name") not in _WORKOUT_TOOLS):
            continue
        result = record.get("result")
        if not isinstance(result, dict):
            continue
        tool_name = record["tool_name"]
        sequence = record.get("sequence")

        if tool_name == "list_workouts":
            rows = result.get("workouts")
            if not isinstance(rows, list):
                continue
            # Only list_workouts can return more than one session of the
            # same type on the same date (two walks, two runs); add a start
            # time to break that tie, and only for the rows that need it, so
            # a day's single run still keys identically to what
            # get_run_form/get_block_structure/get_briefing publish for it.
            type_counts: dict[tuple, int] = {}
            for row in rows:
                if isinstance(row, dict):
                    dkey = (row.get("date"), row.get("type"))
                    type_counts[dkey] = type_counts.get(dkey, 0) + 1
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                dkey = (row.get("date"), row.get("type"))
                start_time = (row.get("start_time_local")
                             if type_counts.get(dkey, 0) > 1 else None)
                workout = _workout_identity(
                    row.get("date"), row.get("type"), start_time)
                candidates.extend(_workout_row_candidates(
                    row, workout, sequence=sequence,
                    path_prefix=f"$.result.workouts[{row_index}]"))

        elif tool_name == "get_run_form":
            if result.get("mode") != "session" or not result.get("found"):
                continue
            # get_run_form only ever describes a running session.
            workout = _workout_identity(result.get("date"), "running")
            efficiency = result.get("efficiency_change")
            if isinstance(efficiency, dict) and efficiency.get("status") == "ok":
                candidate = _workout_candidate(
                    workout, "session_jog_minutes",
                    efficiency.get("jog_minutes"), "min", sequence=sequence,
                    path="$.result.efficiency_change.jog_minutes")
                if candidate is not None:
                    candidates.append(candidate)

        elif tool_name == "get_block_structure":
            day = result.get("day")
            sessions = result.get("sessions")
            sessions = sessions if isinstance(sessions, list) else []
            # The day-level "best across sessions" fields belong to exactly
            # one session's type; only safe to name when that is unambiguous
            # (one session that day). A multi-session day withholds these
            # two rather than guessing whose block it was.
            if len(sessions) == 1 and isinstance(sessions[0], dict):
                solo_workout = _workout_identity(
                    day, sessions[0].get("workout_type"))
                for raw_field, unit in (("longest_block_min", "min"),
                                        ("qualified_block_min", "min")):
                    candidate = _workout_candidate(
                        solo_workout, raw_field, result.get(raw_field), unit,
                        sequence=sequence, path=f"$.result.{raw_field}")
                    if candidate is not None:
                        candidates.append(candidate)
            for session_index, session in enumerate(sessions):
                if not isinstance(session, dict):
                    continue
                workout = _workout_identity(day, session.get("workout_type"))
                candidate = _workout_candidate(
                    workout, "avg_heart_rate",
                    session.get("avg_hr_session"), "bpm",
                    sequence=sequence,
                    path=f"$.result.sessions[{session_index}]"
                         ".avg_hr_session")
                if candidate is not None:
                    candidates.append(candidate)

        elif tool_name == "get_briefing":
            # See the module-level comment above _WORKOUT_TOOLS: this tool's
            # only numeric fields duplicate list_workouts at a coarser
            # rounding, so nothing from it is republished here today.
            continue

    return _publish_unambiguous(candidates)


def render_fact_set(facts: dict[str, dict]) -> str:
    """Render facts for the final model turn in deterministic JSON."""
    return json.dumps(facts or {}, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def cold_start_guidance(facts: dict[str, dict]) -> str:
    """Describe how a refusal surface must use its published status sentence."""
    grouped: dict[str, dict[str, str]] = {}
    for key, fact in (facts or {}).items():
        path = fact.get("path", key) if isinstance(fact, dict) else key
        match = _COLD_START_PATH_RE.fullmatch(str(path))
        if not match:
            continue
        grouped.setdefault(match.group(1), {})[match.group(2)] = str(key)

    lines = []
    for surface in sorted(grouped):
        leaves = grouped[surface]
        status_key = leaves.get("status")
        status_value = (facts.get(status_key, {}).get("value")
                        if status_key else None)
        if status_value not in _COLD_START_REFUSAL_STATUSES:
            continue
        status_text = leaves.get("status_text")
        if status_text:
            lines.append(
                f"- {surface}: use {{{status_text}}} as the sentence stating "
                "why this surface is refusing; do not replace it with a "
                "generic 'unavailable' paraphrase."
            )
    if not lines:
        return ""
    return (
        "COLD-START REFUSAL GUIDANCE: when a surface is refusing, use its "
        "Python-owned status_text placeholder as the sentence to state the "
        "measured start and current day.\n" + "\n".join(lines)
    )


def scan_template(template: str, facts: dict[str, dict]) -> dict:
    """Check fact/advice slots and digits outside their spans.

    ``{advice:...}`` is the only literal-content exemption. Its contents are
    model-authored coaching guidance, not Python-owned facts, and therefore
    are returned separately for the response/UI label. A slot may not mention
    a canonical vault metric or the user's own data.
    """
    text = template if isinstance(template, str) else ""
    matches = list(_PLACEHOLDER_RE.finditer(text))
    stripped = _PLACEHOLDER_RE.sub("", text)
    advice_quantities = []
    keys = []
    advice_errors = []
    for match in matches:
        token = match.group(1)
        if token.startswith(_ADVICE_PREFIX):
            content = token[len(_ADVICE_PREFIX):].strip()
            if not content:
                advice_errors.append("empty advice slot")
            elif not re.search(r"\d", content):
                # A digit-free span was always legal as plain prose, so a slot
                # around it earns no exemption and no label — unwrap it rather
                # than refuse. Refusing was measured live 2026-08-31 to kill
                # the flagship advice question when the model wrapped
                # encouragement at both attempts; unwrapping is behaviorally
                # identical to the model never typing the slot.
                pass
            else:
                advice_quantities.append(content)
                violation = _advice_violation(content, facts)
                if violation:
                    advice_errors.append(violation)
        else:
            keys.append(token)
    unresolved = [key for key in keys if key not in (facts or {})]
    malformed = "{" in stripped or "}" in stripped
    digits = bool(re.search(r"\d", stripped))
    reason = ("malformed placeholder" if malformed else
              advice_errors[0] if advice_errors else
              "unresolvable placeholder" if unresolved else
              "digit outside placeholder" if digits else "")
    return {
        "ok": (not (malformed or unresolved or digits or advice_errors)
               and bool(text.strip())),
        "placeholders": keys,
        "advice_quantities": advice_quantities,
        "unresolved": unresolved,
        "digits_outside_placeholders": digits,
        "reason": reason,
    }


def conversational_violation(text: str,
                             facts: dict[str, dict] | None = None) -> str:
    """Return a reason a model-authored conversational reply is unsafe.

    A conversational reply has no Python-owned facts, so it may contain no
    digits, number words, placeholders or advice slots (``scan_template`` and
    ``number_words``, unconditionally), and it may not assert anything about
    the user's data (``conversational_assertion``).

    It MAY name a metric. The previous rule refused any canonical metric
    spelling; measured on the deployed shape (#69) that refused seven
    capability menus out of seven refusals and admitted "You ran twelve miles
    on Saturday.", because the risk is a claim, and a claim is a subject plus
    a predicate, not a noun. See ``conversational_assertion``.
    """
    text = text if isinstance(text, str) else ""
    scan = scan_template(text, facts or {})
    if not scan["ok"]:
        return scan["reason"] or "empty conversational answer"
    if _PLACEHOLDER_RE.search(text):
        return "conversational answer contains a placeholder"
    # Checked here, not only in scan_template: whether ledger-backed
    # narration refuses number words is #49's decision, and this exemption
    # must not depend on it. A reply that states nothing states no "twelve".
    if number_words(text, facts):
        return "number word outside placeholder"
    return conversational_assertion(text, facts)


def template_refused(template: str, facts: dict[str, dict]) -> bool:
    """Return whether a template must be refused before interpolation."""
    return not scan_template(template, facts)["ok"]


def interpolate_template(template: str, facts: dict[str, dict], *,
                          advice_quantities: list[str] | None = None) -> str | None:
    """Interpolate a valid template and optionally collect advice spans.

    The optional list keeps the established string-returning API intact while
    allowing the ask arm to publish the exact advice contents alongside its
    verification result.
    """
    scan = scan_template(template, facts)
    if not scan["ok"]:
        return None
    if advice_quantities is not None:
        advice_quantities.extend(scan["advice_quantities"])
    return _PLACEHOLDER_RE.sub(
        lambda match: (match.group(1)[len(_ADVICE_PREFIX):].strip()
                       if match.group(1).startswith(_ADVICE_PREFIX)
                       else str(facts[match.group(1)]["display"])), template)


# Verbose aliases make the two safety boundaries easy to discover at call sites.
resolve_template = interpolate_template
refuse_template = template_refused
