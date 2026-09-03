"""Verify a research finding's numbers against the entry each one CLAIMS.

Why this replaces the old bag-of-floats key (measured 2026-08-01 against the
real deepdive_2026-07-26 trace):

    answer_key: 11,314 rederived floats + 732 workout floats, 9.0 s to build
    random integer in [0,100] passing grounding ......... 93.2%

The anti-hallucination gate was a no-op. Both `_filter` and the compiler's
report check used that key, so both were compromised, and it cost 95 KB inside
the judge prompt — the single most expensive call in the pipeline.

The instinct is to shrink the bag. That is not sufficient, and the numbers say
so plainly:

    rederived only (11,314 floats) ..... 73.7%
    workouts only (732 floats) ......... 67.4%
    briefing only ...................... 20.2%

Even the briefing alone — an honest, small, non-shotgun set — accepts a fifth of
arbitrary integers, because ANY dense set of small floats covers the integers.
Membership in a bag is simply not evidence about a specific claim.

So verification here is per-claim: a finding's number declares
``{metric, period, field, value}``, and we recompute THAT metric over THAT
period and compare against THAT field. Scoping to one (metric, period) leaves
about a dozen candidate values instead of twelve thousand, which drops the
false-accept rate to 0.8-6.2% even before the exact-field match, and an exact
field match makes it a genuine test.

Field names are model-authored and messy. Across every historical run: 31 of
153 numbers used the field 'value', 10 had no field at all, and models invented
names like 'spearman_rho_sleep_awake_lag1' and 'above_150_minutes'. A resolver
that demanded canonical names would reject a large share of legitimate findings,
so an unrecognised field falls back to the SCOPED candidate set for the metric
and period it named — still a real constraint, just a weaker one. The result
records which happened, so a caller can tell a verified claim from a merely
plausible one.
"""
from __future__ import annotations

import os
import re
from datetime import date, timedelta

from . import correlate as C
from . import agents as G
from . import metrics as mx
from .numeric_tokens import NUM_RE as _NUM_RE

# Recomputable directly from one metric's series over one window.
_SERIES_FIELDS = ("mean", "median", "min", "max", "std", "latest", "n_days",
                  "sum", "recent_avg", "baseline_avg", "delta_pct",
                  "slope_per_week")
# ``n_days`` remains in _SERIES_FIELDS because the SQL route can derive it for
# a metric. It is not, however, the metric's series value when it appears as
# context beside a published stats row. This set is for inherited JSON labels;
# explicit {field, value} claim objects remain authoritative.
_INHERITED_SERIES_FIELDS = (
    frozenset(_SERIES_FIELDS) - {"n_days"}
    | frozenset({"total", "mean_delta", "total_delta", "total_delta_pct",
                 "weeks_per_block"})
)
# Numeric context emitted beside a stats series. These remain unowned; the
# ledger resolver may only ignore a surplus claim label on this known context
# vocabulary, not on arbitrary numeric siblings such as `days_covered`.
_SURPLUS_LABEL_CONTEXT_FIELDS = frozenset(("n_days", "rho", "sd_day", "mdc95"))


def _metric_owns_field(metric, field) -> bool:
    """Whether an inherited metric label belongs on this leaf.

    The extra names are the explicitly published impact-block outputs retained
    for the existing claim vocabulary; they are not blanket inheritance of all
    sibling context fields.
    """
    return bool(metric) and (
        field == metric or field in _INHERITED_SERIES_FIELDS)
# Fields that only exist as correlation output; they need the metric PAIR, so
# they resolve against every pair involving the cited metric over the window.
_CORR_RE = re.compile(r"rho|pearson|spearman|n_pairs|q_value|p_value|"
                      r"passed_fdr|tested_count|corr", re.I)
# Per-session values from list_workouts, which live in `workouts`, not
# daily_metrics, and so are never derivable from a metric series.
_WORKOUT_RE = re.compile(r"heart_rate|duration|distance|energy|pace|speed", re.I)

# Counts produced by a whole scan_correlations sweep (how many pairs were
# tested, how many survived FDR). Recomputing one means re-running ~106
# correlations — the cost that made the old answer key take 9 seconds — and the
# result depends on the exact candidate set the researcher's sweep used, which
# is not recorded. So these are reported as NOT CHECKED rather than guessed at.
# Calling a number wrong when we cannot check it is its own kind of fabrication.
_SCAN_COUNT_RE = re.compile(r"tested_count|passed_fdr_count|n_tests", re.I)

_NUM_IN_STR = re.compile(r"-?\d[\d,]*\.?\d*")

# A model can spend a whole response emitting the same punctuation-only token
# as a markdown bullet. Keep this detector intentionally narrow: meaningful
# repeated coaching lines (for example, a week of "Rest day" entries) contain
# alphanumeric content and must remain ordinary prose.
_DEGENERATE_BULLET_RE = re.compile(r"^\s*[-*+]\s+(\S+)\s*$")


def _is_degenerate_repeated_bullets(prose: str) -> bool:
    """Identify a repeated-token generation without treating repetition as bad.

    The shape is only considered degenerate when the entire non-empty response
    is at least eight identical markdown bullets whose token contains no
    alphanumeric character. This excludes normal repetitive coaching output
    while matching the observed ``- ......`` generation.
    """
    if not isinstance(prose, str) or _NUM_RE.findall(prose):
        return False
    lines = [line for line in prose.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    tokens = []
    for line in lines:
        match = _DEGENERATE_BULLET_RE.fullmatch(line)
        if not match:
            return False
        tokens.append(match.group(1))
    token = tokens[0]
    return (bool(token) and not any(char.isalnum() for char in token)
            and all(candidate == token for candidate in tokens))

# The derived-claim operation vocabulary. Closed on purpose — parsing an
# operation the verifier does not understand would be the model deriving a
# computation. Because it is closed, the model has to be told what is in it:
# these names are the single source both `_verify_derivation` and the coach's
# claim-channel prompt read, so the set cannot drift away from the sentence
# describing it (#93).
PERCENT_CHANGE_OPERATIONS = frozenset({
    "percent_change", "percentage_change", "pct_change", "relative_change",
})
PERCENT_DOWN_OPERATIONS = frozenset({
    "percent_down", "percentage_down", "decrease_pct",
})
DIFFERENCE_OPERATIONS = frozenset({"delta", "difference", "subtract"})
DERIVATION_OPERATIONS = (
    PERCENT_CHANGE_OPERATIONS | PERCENT_DOWN_OPERATIONS | DIFFERENCE_OPERATIONS)

_PERCENT_OPERATIONS = PERCENT_CHANGE_OPERATIONS | PERCENT_DOWN_OPERATIONS


def operation_vocabulary_sentence() -> str:
    """The one sentence that publishes the closed vocabulary to the model."""
    return ("`operation` must be exactly one of these words, copied verbatim: "
            + ", ".join(sorted(DERIVATION_OPERATIONS))
            + ". Do not describe the arithmetic in words or symbols; a formula "
              "such as \"(recent total - prior total)\" is not an operation.")


def weekly_claim_metadata_sentence() -> str:
    return ("`get_weekly_series` publishes each row's inclusive Monday-Sunday "
            "`period` as `YYYY-MM-DD:YYYY-MM-DD`; copy that exact string into "
            "a claim and never invent a period from `week_start`.")


def metric_ownership_sentence() -> str:
    return ("Metric ownership is per field: inherit a row's `metric` only for "
            "its own series-value leaves (`mean`, `median`, `min`, `max`, "
            "`std`, `latest`, `sum`, `recent_avg`, `baseline_avg`, "
            "`delta_pct`, `slope_per_week`) or a leaf whose field is exactly "
            "that metric; never inherit it for context fields such as "
            "`n_days`, `rho`, `sd_day`, `mdc95`, `unit`, dates, day counts, or "
            "other siblings.")


def subjective_claim_metadata_sentence() -> str:
    return ("`get_subjective` keeps flat day fields and adds `period` equal to "
            "the day plus `field_metrics` for the non-null rating fields "
            "(`stress`, `soreness`, `energy`, `sleep_quality`); cite the "
            "direct field with its mapped `subjective_*` metric, and omit "
            "`metric` for fields absent from `field_metrics`.")


def workout_count_claim_metadata_sentence() -> str:
    return ("`list_workouts` publishes full-range per-type counts as "
            "`workout_counts: [{type, count}]`; cite the `count` leaf at its "
            "exact path with `metric` omitted, and never count the possibly "
            "truncated `workouts` rows.")


_DECREASE_WORD_RE = re.compile(
    r"\b(?:down|decrease(?:d|s|ing)?|decline(?:d|s|ing)?|"
    r"drop(?:ped|s|ping)?|fall(?:en|s|ing)?|fell|lower|"
    r"less|reduc(?:e|ed|es|ing)|shrank|shrink)\b", re.I)
_INCREASE_WORD_RE = re.compile(
    r"\b(?:up|increase(?:d|s|ing)?|rise|rises|rose|rising|"
    r"higher|gain(?:ed|s|ing)?|grow|grew|growing|growth|more|expanded)\b", re.I)


def _as_float(value) -> float | None:
    """Findings carry values as floats OR as strings like '987 kcal'."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _NUM_IN_STR.search(str(value))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _claimable_value(value):
    """Values the publisher may expose as a claimable fact."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        return value
    return None


def _close(a: float, b: float, rel_tol: float = 0.005,
           abs_floor: float = 0.05) -> bool:
    """Same tolerance policy as agents.grounding_check: relative, with a small
    absolute floor so rounding of large values passes while a neighbouring
    small integer (a fabricated 99 against a real 100) does not."""
    return abs(a - b) <= max(abs_floor, abs(b) * rel_tol)


def _rule_r_matches(token: str, claim_value) -> bool:
    """Return whether one prose token is exact or legally rounded to a claim."""
    normalized = str(token).strip().rstrip(".,")
    try:
        prose_value = float(normalized.replace(",", ""))
    except ValueError:
        return False
    claimed = _as_float(claim_value)
    if claimed is None:
        return False
    if prose_value == claimed:
        return True
    decimals = (len(normalized.split(".", 1)[1])
                if "." in normalized else 0)
    if decimals == 0 and abs(claimed) < 1:
        return False
    return abs(prose_value - claimed) <= (
        0.5 * 10 ** (-decimals) + 1e-9)


def _is_percent_claim(claim: dict) -> bool:
    operation = str(claim.get("operation", claim.get("op", ""))).strip().lower()
    field = str(claim.get("field", "")).strip().lower()
    return operation in _PERCENT_OPERATIONS or "percent" in field or "pct" in field


def _is_signed_claim(claim: dict) -> bool:
    """Whether a claim carries a direction that prose may spell out in words."""
    operation = str(claim.get("operation", claim.get("op", ""))).strip().lower()
    field = str(claim.get("field", "")).strip().lower()
    value = _as_float(claim.get("value"))
    return (operation in DIFFERENCE_OPERATIONS
            or "delta" in field or "change" in field
            or (value is not None and value < 0))


def _signed_percentage_matches(prose: str, claims: list[dict],
                               *, signed_values: bool = False,
                               rule_r: bool = False) -> list[str]:
    """Return unsigned prose tokens licensed by a verified signed claim.

    Percentage and delta prose conventionally carries a decrease/increase sign
    in words such as ``down`` or ``up``. The claim remains the authority for the
    signed value; this helper only bridges that presentation difference when
    the local direction agrees with the claim's sign. ``signed_values`` keeps
    the older researcher-path behaviour opt-in for non-percentage fields.
    """
    cleaned = G.strip_dates_and_names(prose)
    matches = []
    for token_match in _NUM_RE.finditer(cleaned):
        token = token_match.group()
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if value < 0:
            continue
        left = max(0, token_match.start() - 48)
        neighborhood = cleaned[left:token_match.end() + 48]
        decreases = bool(_DECREASE_WORD_RE.search(neighborhood))
        increases = bool(_INCREASE_WORD_RE.search(neighborhood))
        if decreases == increases:
            continue
        direction = -1 if decreases else 1
        for claim in claims:
            if (not isinstance(claim, dict)
                    or (claim.get("field") == "presentation"
                        and isinstance(claim.get("value"), str))
                    or (not _is_percent_claim(claim)
                        and not (signed_values and _is_signed_claim(claim)))):
                continue
            claimed = _as_float(claim.get("value"))
            if claimed is None or claimed == 0 or (claimed < 0) != (direction < 0):
                continue
            licensed = (_rule_r_matches(token, abs(claimed))
                        if rule_r else _close(value, abs(claimed)))
            if licensed:
                matches.append(token)
                break
    return matches


def _structural_claims(payload) -> list[dict]:
    """Return source-backed structural claims published by a ledgered call.

    ``weeks_per_block`` is not a measurement and has no independent period, but
    it is still a fact in the tool's result. Representing it as a claim keeps
    the structural count inside the same source/value gate instead of silently
    whitelisting the prose token. Only the result copy is used: the argument
    says what was requested, while the result says what the tool published.
    """
    if not _is_ledger(payload):
        return []
    out = []
    seen = set()
    for record in payload:
        if not isinstance(record, dict):
            continue
        for entry in _ledger_scopes(record):
            if (entry.get("kind") != "result"
                    or entry.get("field") != "weeks_per_block"
                    or not entry.get("metric")):
                continue
            key = (record.get("sequence"), entry.get("path"))
            if key in seen:
                continue
            claim = {
                "metric": entry["metric"], "period": None,
                "field": entry["field"], "value": entry["value"],
                "source": {"sequence": record.get("sequence"),
                           "path": entry["path"]},
            }
            resolved = _resolve_ledger_value(payload, claim)
            if resolved.get("ok"):
                out.append(claim)
                seen.add(key)
    return out


def _structural_matches(prose: str, claims: list[dict], *, rule_r: bool = False) -> list[str]:
    """Match only source-backed window-shape values next to a week label."""
    cleaned = G.strip_dates_and_names(prose)
    matches = []
    for token_match in _NUM_RE.finditer(cleaned):
        token = token_match.group()
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if value < 0:
            continue
        neighborhood = cleaned[max(0, token_match.start() - 48):
                               token_match.end() + 48]
        if not re.search(r"\bweeks?\b", neighborhood, re.I):
            continue
        for claim in claims:
            if (isinstance(claim, dict)
                    and claim.get("field") == "weeks_per_block"
                    and (_rule_r_matches(token, claim.get("value"))
                         if rule_r else
                         _close(value, _as_float(claim.get("value"))))):
                matches.append(token)
                break
    return matches


def _scoped_grounding_claims(claims, payload) -> list[dict]:
    """Keep only claims resolved in the payload's own metric/period/field scope.

    A numeric set made from all claims has the same collision flaw as the old
    briefing bag. When a non-ledger payload is available, each claim therefore
    goes through ``resolve_payload_value`` independently. With no payload, the
    caller has already supplied claims that were verified against SQL, so their
    declared scopes are the available evidence. This helper never derives a
    value and never merges unrelated claim identities.
    """
    if not isinstance(claims, list):
        return []
    candidates = [claim for claim in claims if isinstance(claim, dict)]
    if payload is None or _is_ledger(payload):
        return candidates
    if not _payload_scopes(payload):
        return []
    return [claim for claim in candidates
            if (resolve_payload_value(payload, claim) or {}).get("ok")]


def _coach_grounding(prose: str, claims: list[dict], payload=None) -> tuple[bool, list[str]]:
    """Ground numeric occurrences in one complete answer.

    The ledger branch below counts numeric-token occurrences as a multiset;
    one presentation claim can retire only one matching occurrence. This
    function must therefore receive the complete answer, never a sentence or
    span evaluated independently and then combined: independent calls let
    the same claim pay for the same evidence once per fragment, weakening the
    whole-answer gate.
    """
    # Rule R runs exactly where the claim layer beneath it is exact: a ledger
    # payload. Non-ledger payloads use the same scoped claim resolver, with the
    # legacy tolerance because their published claim layer allows +/-0.5%;
    # demanding exact prose against a drifted claim would refuse prose stating
    # the true payload value.
    if _is_ledger(payload):
        cleaned = G.strip_dates_and_names(prose)
        remaining = [
            token_match.group()
            for token_match in _NUM_RE.finditer(cleaned)
            if not any(isinstance(claim, dict)
                       and not (claim.get("field") == "presentation"
                                and isinstance(claim.get("value"), str))
                       and _rule_r_matches(token_match.group(),
                                           claim.get("value"))
                       for claim in claims)
        ]
        for token in _signed_percentage_matches(
                prose, claims, signed_values=True, rule_r=True):
            if token in remaining:
                remaining.remove(token)
        for token in _structural_matches(
                prose, _structural_claims(payload), rule_r=True):
            if token in remaining:
                remaining.remove(token)
        for token in _presentation_matches(prose, claims):
            if token in remaining:
                remaining.remove(token)
        return not remaining, remaining

    # Do not turn the claims list into a bag of floats. The old call to
    # G.grounding_check did exactly that and allowed a value from one claim to
    # license prose for another claim. Resolve each claim in its own scope.
    scoped_claims = _scoped_grounding_claims(claims, payload)
    remaining = [token_match.group() for token_match in
                 _NUM_RE.finditer(G.strip_dates_and_names(prose))
                 if not any(
                     _close(float(token_match.group().replace(",", "")),
                            _as_float(claim.get("value")))
                     for claim in scoped_claims
                     if _as_float(claim.get("value")) is not None
                 )]
    for token in _presentation_matches(prose, scoped_claims):
        if token in remaining:
            remaining.remove(token)
    if not remaining:
        return True, []
    # Only remove tokens that are actually reconciled. In particular, an
    # unsigned token next to the opposite direction remains unsupported.
    for token in _signed_percentage_matches(prose, scoped_claims,
                                            signed_values=True):
        if token in remaining:
            remaining.remove(token)
    for token in _structural_matches(prose, _structural_claims(payload)):
        if token in remaining:
            remaining.remove(token)
    return not remaining, remaining


def _presentation_matches(prose: str, claims: list[dict]) -> list[str]:
    """Return numeric tokens covered by exact published presentation text."""
    matches = []
    for claim in claims:
        if (not isinstance(claim, dict)
                or claim.get("field") != "presentation"
                or not isinstance(claim.get("value"), str)
                or not claim["value"]):
            continue
        if re.search(re.escape(claim["value"]), prose or ""):
            matches.extend(_NUM_RE.findall(claim["value"]))
    return matches


def _research_grounding(prose: str, claims: list[dict],
                        payload=None) -> tuple[bool, list[str]]:
    """Ground signed prose such as ``down 0.1`` to a signed claim value."""
    prose = _RESEARCH_BLOCK_ORDINAL_RE.sub(" ", prose or "")
    rule_r = _is_ledger(payload)
    # Keep the complete research prose here. The occurrence multiset in
    # _coach_grounding is not sound when sentence fragments share claims.
    grounded, bad = _coach_grounding(prose, claims, payload=payload)
    if grounded:
        return True, []
    # This rescue pass must run under the same licence as the pass above:
    # with _close here, a signed token rule R refused would be re-licensed
    # by the old proximity tolerance on exactly this path.
    remaining = list(bad)
    for token in _signed_percentage_matches(prose, claims, signed_values=True,
                                            rule_r=rule_r):
        if token in remaining:
            remaining.remove(token)
    return not remaining, remaining


_RANGE_IN = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(?::|\bto\b)\s*(\d{4}-\d{2}-\d{2})")
_DATE_IN = re.compile(r"\d{4}-\d{2}-\d{2}")
_PERIOD_IN = re.compile(r"\b(\d+\s*[dwmy]|all|max|lifetime)\b", re.I)

# Research prose often names the two comparison blocks as "the last 4 complete
# weeks" and "the 4 weeks before that". Those 4s are authored block ordinals,
# not measurements. Keep this allowlist local: a generic "N weeks" rule would
# remove real quantities, just as broadening agents._NAME_TERM_RE would.
_RESEARCH_BLOCK_ORDINAL_RE = re.compile(
    r"\b(?:last|past|prior|previous)\s+\d+\s+(?:complete\s+)?weeks?\b|"
    r"\b(?:the\s+)?\d+\s+weeks?\s+before(?:\s+that)?\b", re.I)

# The base of every claim channel is deliberately the coach vocabulary.  A
# researcher claim adds only ``source``; it does not get a second metric/period
#/field/value dialect.
SCOPED_CLAIM_FIELDS = frozenset(("metric", "period", "field", "value"))


def _period_key(period):
    """Return a comparable representation of a scoped period.

    Tool payloads use a structured period for block aggregates and a string for
    individual weeks. Claims may use either representation; accepting the
    equivalent explicit ``start:end`` spelling keeps the scope explicit without
    making the model reproduce JSON key ordering.
    """
    if isinstance(period, dict):
        if "start" in period and "end" in period:
            return (str(period["start"]), str(period["end"]))
        starts = period.get("period_starts")
        if isinstance(starts, list) and starts:
            return tuple(str(v) for v in starts)
        return None
    if isinstance(period, (list, tuple)):
        return tuple(str(v) for v in period)
    text = str(period or "").strip()
    if not text:
        return None
    rng = _RANGE_IN.search(text)
    if rng:
        return (rng.group(1), rng.group(2))
    return text


def _same_period(left, right) -> bool:
    """Compare only explicit periods; never make an omitted scope implicit."""
    a, b = _period_key(left), _period_key(right)
    return a is not None and b is not None and a == b


def _published_period_alias(period) -> str | None:
    """Return the model-facing period spelling published for a block."""
    if not isinstance(period, dict):
        return None
    starts = period.get("period_starts")
    if not isinstance(starts, list) or not starts:
        return None
    try:
        last_end = (date.fromisoformat(str(starts[-1]))
                    + timedelta(days=6)).isoformat()
    except (TypeError, ValueError):
        return None
    return f"{starts[0]}:{last_end}"


def _payload_scopes(payload) -> list[dict]:
    """Flatten labelled values from one or more tool results.

    The half-A block payload labels each weekly value as
    ``{metric, period, field, value}``. Its block totals and means are labelled
    by their containing block rather than repeating that four-key shape, so
    those fields are made scoped entries here as well. Unlabelled numbers are
    deliberately ignored: a sibling such as ``days: 100`` cannot authorize a
    claim about a different metric or operation.
    """
    out: list[dict] = []

    def walk(node, inherited_metric=None, inherited_period=None):
        if isinstance(node, dict):
            metric = node.get("metric", inherited_metric)
            period = node.get("period", inherited_period)
            field_metrics = node.get("field_metrics")
            if not isinstance(field_metrics, dict):
                field_metrics = {}
            field = node.get("field")
            if metric and period is not None and field and "value" in node:
                value = _claimable_value(node.get("value"))
                if value is not None:
                    out.append({"metric": metric, "period": period,
                                "field": str(field), "value": value,
                                "source": "payload", "tier": "metric"})
            # Block aggregates are published as named fields alongside their
            # period. Do not flatten arbitrary numeric keys (e.g. days).
            if metric and period is not None:
                for name in ("mean", "total", "mean_delta", "total_delta",
                             "total_delta_pct"):
                    if (name in node and _metric_owns_field(metric, name)
                            and isinstance(node.get(name), (int, float))
                            and not isinstance(node.get(name), bool)):
                        value = _as_float(node.get(name))
                        if value is not None:
                            out.append({"metric": metric, "period": period,
                                        "field": name, "value": value,
                                        "source": "payload", "tier": "metric"})
            # get_impact_volume's ordinary period rows publish the canonical
            # jog-minute field in a flat shape. Keep this narrow: the sibling
            # impact fields are derived context, not the jog_minutes series.
            if (metric == "jog_minutes" and isinstance(period, str)
                    and "jog_minutes" in node):
                value = _as_float(node.get("jog_minutes"))
                if value is not None:
                    out.append({"metric": metric, "period": period,
                                "field": "jog_minutes", "value": value,
                                "source": "payload", "tier": "metric"})
            for key, child in node.items():
                if key == "field_metrics":
                    continue
                # A block's `change` has no period of its own and is not a
                # scoped operand. Its block fields above are sufficient.
                if key == "change":
                    continue
                walk(child, metric, period)
                if (key in field_metrics and period is not None
                        and isinstance(child, (int, float))
                        and not isinstance(child, bool)
                        and field_metrics[key]):
                    out.append({"metric": str(field_metrics[key]),
                                "period": period, "field": str(key),
                                "value": float(child),
                                "source": "payload", "tier": "metric"})
        elif isinstance(node, list):
            for child in node:
                walk(child, inherited_metric, inherited_period)

    walk(payload)
    return out


def _is_ledger(payload) -> bool:
    return (isinstance(payload, list) and any(
        isinstance(record, dict) and "sequence" in record
        and "tool_name" in record and "result" in record
        for record in payload))


def _path_text(path: tuple) -> str:
    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}"
    return rendered


def _ledger_scopes(record: dict) -> list[dict]:
    """Flatten one non-elided ledger record while retaining exact JSON paths."""
    out: list[dict] = []

    def walk(node, path, inherited_metric=None, inherited_period=None,
             inherited_workout_key=None, declared_field=None,
             field_metric_owned=False, root="result", points_metric=None):
        if isinstance(node, dict):
            metric = node.get("metric", inherited_metric)
            period = node.get("period", inherited_period)
            field_metrics = node.get("field_metrics")
            if not isinstance(field_metrics, dict):
                field_metrics = {}
            workout_key = node.get(
                "workout_key", node.get("dedupe_key", inherited_workout_key))
            field = node.get("field", declared_field)
            for key, child in node.items():
                if key == "field_metrics":
                    continue
                child_metric = field_metrics.get(key, metric)
                child_metric_owned = key in field_metrics
                # `points[N].value` is owned by the enclosing result metric;
                # keep this exception scoped to that exact result shape.
                child_points_metric = (
                    node.get("metric") if root == "result"
                    and key == "points" and node.get("metric") else None)
                walk(child, path + (key,), child_metric, period,
                     workout_key, field, child_metric_owned, root,
                     child_points_metric or points_metric)
            return
        if isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, path + (index,), inherited_metric, inherited_period,
                     inherited_workout_key, declared_field, field_metric_owned,
                     root, points_metric)
            return
        if not path:
            return
        leaf = str(path[-1])
        # A published {field, value} object names the measurement with its
        # declared field. Metadata siblings such as days_covered do not inherit
        # that measurement's ownership.
        explicit_measurement = declared_field and leaf == "value"
        field = str(declared_field if explicit_measurement else leaf)
        points_value_owned = bool(
            root == "result" and leaf == "value" and points_metric
            and len(path) >= 3 and path[-3] == "points"
            and isinstance(path[-2], int))
        metric_owned = (
            explicit_measurement or field_metric_owned or points_value_owned
            or (not declared_field
                and _metric_owns_field(inherited_metric, field)))
        metric = inherited_metric if metric_owned else None
        out.append({"metric": metric, "period": inherited_period,
                    "field": field, "value": node,
                    "workout_key": inherited_workout_key,
                    "path": _path_text((root,) + path),
                    "kind": root,
                    # A context leaf beside a metric result is distinct from
                    # a metricless result shape such as workout counts.
                    "context": (bool(inherited_metric) and not metric_owned
                                and field in _SURPLUS_LABEL_CONTEXT_FIELDS)})

    for root in ("result", "arguments"):
        if root in record and not (root == "result" and record.get("result_elided")):
            walk(record[root], (), root=root)
    return out


def _source_path_matches(entry_path: str, requested: str) -> bool:
    requested = str(requested or "").strip()
    if not requested:
        return False
    if not requested.startswith("$"):
        requested = "$." + requested
        requested = requested.replace("$.\u200b", "$.")
    # Result paths may be written relative to the result object (the natural
    # model-facing spelling) or rooted at the ledger record.
    if requested == "$.result":
        return entry_path == "$.result"
    if requested.startswith("$.result.") or requested.startswith("$.arguments."):
        return entry_path == requested
    return entry_path == "$.result" + requested[1:]


def _ledger_record_search_enabled() -> bool:
    """Is Python allowed to find the record a claim's number actually came from?

    Off by default, so the resolver's behaviour is byte-identical to the
    sequence-only version unless ``HA_ASK_LEDGER_RESOLVE=1``. Read per call, not
    captured at import, so both arms are measurable in one process.
    """
    raw = str(os.environ.get("HA_ASK_LEDGER_RESOLVE", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# A refusal that already means "the evidence is not unique". Widening the
# candidate set cannot turn one of these into a unique answer, so no
# looser-pointer method is allowed to overwrite the reason with its own.
_AMBIGUITY_REFUSALS = frozenset((
    "ambiguous ledger path",
    "ambiguous ledger record",
    "ambiguous payload scope",
    "ambiguous value rebind",
))


def _value_rebind_enabled() -> bool:
    """Is Python allowed to ignore the model's pointer and re-match on content?

    Method C, off by default and independent of ``HA_ASK_LEDGER_RESOLVE``. Read
    per call, not captured at import, so both arms are measurable in one
    process.
    """
    raw = str(os.environ.get("HA_ASK_VALUE_REBIND", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _entry_is_workout_scoped(entry: dict) -> bool:
    """A list_workouts row: server-stable workout identity, no metric owner."""
    return entry["kind"] == "result" and entry.get("workout_key") is not None


def _is_surplus_context_metric(entry: dict, claim: dict) -> bool:
    """Whether a claim label is surplus on a known result context leaf.

    Context fields remain unowned. This is only a path/value resolution rule
    for result leaves that sit under an enclosing metric result; it does not
    apply to metricless result shapes such as workout counts.
    """
    return (claim.get("metric") is not None
            and entry.get("metric") is None
            and entry.get("context") is True)


def _entry_claim_refusal(entry: dict, claim: dict) -> dict | None:
    """The whole per-ENTRY contract: field, metric, period, exact value.

    Returns the refusal verdict, or ``None`` when this entry satisfies the
    claim. Factored out of ``_resolve_ledger_value_in_record`` so that the
    record search (Method B) and the value rebind (Method C) apply *these*
    checks rather than a looser copy of them — the copy is how a "widened
    search" quietly becomes a weakened one.
    """
    field = str(claim.get("field") or "").strip()
    if not field:
        return {"ok": False, "reason": "claim has no field"}
    if field != entry["field"]:
        return {"ok": False, "reason": "claim field does not match ledger path",
                "actual_field": entry["field"]}
    metric = claim.get("metric")
    # list_workouts has no metric-series owner. Its row carries the server's
    # stable workout identity instead, so a metric-bearing claim is still
    # refused there; a workout field is never relabelled as a daily-metric
    # series. Metric-less result claims are handled by the general exact-path
    # rule below.
    workout_scoped = _entry_is_workout_scoped(entry)
    if ((workout_scoped and metric is not None) or (
            metric is not None and entry["metric"] != metric
            and not _is_surplus_context_metric(entry, claim))):
        return {"ok": False, "reason": "claim metric does not match ledger field",
                "actual_metric": entry["metric"]}
    if entry["period"] is not None and claim.get("period") is not None:
        claimed_period = claim.get("period")
        exact = _same_period(entry["period"], claimed_period)
        published = _published_period_alias(entry["period"])
        copied = (isinstance(claimed_period, str)
                  and claimed_period.strip() == published)
        if not exact and not copied:
            return {
                "ok": False,
                "reason": "claim period does not match published period vocabulary",
                "actual_period": entry["period"],
                "published_periods": [published] if published else [],
            }
    claimed_raw = _claimable_value(claim.get("value"))
    actual_raw = _claimable_value(entry["value"])
    if claimed_raw is None or actual_raw is None:
        return {"ok": False, "reason": "ledger field has no claimable value",
                "actual": entry["value"]}
    if claimed_raw != actual_raw:
        return {"ok": False, "reason": "claim value does not match ledger field",
                "actual": entry["value"]}
    return None


def _bind_entry(entry: dict, claim: dict) -> dict:
    """Turn a satisfied entry into the resolved verdict. Tier semantics live
    here and nowhere else, so no resolution path can invent its own."""
    resolved = {**entry, "ok": True}
    # A source/path claim is Tier 1: the exact result leaf proves the model did
    # not invent the figure, but Python did not rederive it. A labelled metric
    # claim is Tier 2 and is identified separately by verify_number when it
    # performs the SQL cross-check. Keep this explicit so the two assurances
    # cannot collapse into one boolean.
    resolved["tier"] = (
        "path" if claim.get("metric") is None
        or _is_surplus_context_metric(entry, claim) else "metric")
    if _entry_is_workout_scoped(entry):
        resolved["scope"] = "workout"
    return resolved


def _resolve_ledger_value_in_record(record: dict, claim: dict, path) -> dict:
    """Resolve one claim against ONE ledger record.

    This is the whole of the per-record contract — path uniqueness, then the
    per-entry contract above. It exists as a function so the record-search
    fallback below can apply *these* checks to a candidate record rather than a
    looser copy of them.
    """
    if record.get("result_elided"):
        return {"ok": False, "reason": "ledger result is elided"}
    matches = [entry for entry in _ledger_scopes(record)
               if _source_path_matches(entry["path"], path)]
    if not matches:
        return {"ok": False, "reason": "ledger path not found"}
    if len(matches) != 1:
        return {"ok": False, "reason": "ambiguous ledger path", "matches": matches}
    entry = matches[0]
    refusal = _entry_claim_refusal(entry, claim)
    if refusal is not None:
        return refusal
    return _bind_entry(entry, claim)


def _value_rebind(ledger, claim: dict, sequence, path) -> dict | None:
    """Method C: discard the model's pointer entirely and re-match on content.

    Method B widened *which record* is searched while still requiring the
    model's own ``path``. That helps only when the path was right and the
    sequence was wrong. The measured failure is not so tidy — a claim can name
    the wrong record AND the wrong leaf inside it, and then every record fails
    against the same broken predicate.

    So this method throws the whole ``{sequence, path}`` citation away and asks
    the only question the ledger can actually answer: is there exactly ONE
    citable result entry, anywhere in the ledger, whose value, field, metric
    and period all agree with the claim? Every one of those comparisons is the
    unmodified per-entry contract above — exact float equality, exact field,
    the list_workouts metric refusal, the published period vocabulary. Nothing
    is loosened; only the candidate set widened.

    ``None`` means no candidate, and the caller keeps its original verdict.

    Two candidates are REFUSED, never tie-broken. Numbers are the worst case
    for content matching: low entropy, rounded, and repeated across metrics and
    periods — a 7-day mean can equal a 30-day mean. Picking the nearest
    sequence would turn a coin flip into a figure the user reads as verified,
    which is strictly worse than the refusal it replaces.

    Only ``result`` entries are candidates. ``arguments`` entries are the
    model's own tool inputs, not evidence, and are unreachable by the pointer
    path (``_resolve_ledger_value`` refuses ``$.arguments...`` upstream).
    Ignoring the pointer must not be the thing that makes them reachable.
    """
    hits: list[tuple[dict, dict]] = []
    for record in ledger:
        if not isinstance(record, dict) or record.get("result_elided"):
            continue
        for entry in _ledger_scopes(record):
            if entry.get("kind") != "result":
                continue
            if _entry_claim_refusal(entry, claim) is None:
                hits.append((record, entry))
    if not hits:
        return None
    if len(hits) != 1:
        return {"ok": False, "reason": "ambiguous value rebind",
                "matches": [entry for _, entry in hits],
                "sequences": [record.get("sequence") for record, _ in hits],
                "paths": [entry["path"] for _, entry in hits]}
    found_record, entry = hits[0]
    return {**_bind_entry(entry, claim), "resolved_by": "value_rebind",
            "resolved_sequence": found_record.get("sequence"),
            "claimed_sequence": sequence,
            "claimed_path": path}


def _resolve_ledger_value(ledger, claim: dict) -> dict:
    """Bind a claim to the ledger record its number was actually published in.

    The model names a ``sequence``, and that naming is bookkeeping across its
    own tool calls — not evidence. Measured 2026-08-28 on a six-question
    battery: one answer filed 11 well-formed claims and 2 verified, because it
    cited sequence 13 for a value published by sequence 2 and vice versa. The
    numbers were real, at real published paths, with the right field, metric and
    period; only the record label was wrong.

    So when resolution against the named sequence fails, Python looks for the
    record the number is actually in. What it searches for is unchanged: a
    candidate must pass ``_resolve_ledger_value_in_record`` in full — one
    matching path, matching field, matching metric, matching period, exactly
    equal value. A fabricated number resolves nowhere; a mislabelled metric
    resolves nowhere. Two candidates are ambiguous and refused, exactly as two
    matching paths inside one record already are. Only the search space over
    records widened; no check was weakened.

    That method still trusts the model's ``path``, so a claim that names the
    wrong leaf as well as the wrong record fails in every record — the search
    was widened over a broken predicate. ``HA_ASK_VALUE_REBIND=1`` adds Method
    C beside it (``_value_rebind``), which discards the pointer entirely and
    re-matches on content under a uniqueness gate. The two flags are
    independent; when both are on, Method B is tried first and Method C only
    sees claims it found nothing for.
    """
    source = claim.get("source") if isinstance(claim, dict) else None
    if not isinstance(source, dict):
        return {"ok": False, "reason": "ledger claim has no source"}
    sequence = source.get("sequence")
    path = source.get("path")
    requested_path = str(path or "").strip()
    if requested_path and not requested_path.startswith("$"):
        requested_path = "$." + requested_path
    if (requested_path == "$.arguments" or
            requested_path.startswith("$.arguments.")):
        return {
            "ok": False,
            "reason": ("claim cites a tool argument, not a result: "
                       f"{requested_path}"),
        }
    records = [record for record in ledger
               if isinstance(record, dict) and record.get("sequence") == sequence]
    named = records[0] if records else None
    if named is None:
        named_verdict = {"ok": False, "reason": "ledger sequence not found"}
    else:
        named_verdict = _resolve_ledger_value_in_record(named, claim, path)
        if named_verdict.get("ok"):
            return named_verdict
    if _ledger_record_search_enabled():
        hits = []
        for record in ledger:
            if not isinstance(record, dict) or record is named:
                continue
            found = _resolve_ledger_value_in_record(record, claim, path)
            if found.get("ok"):
                hits.append((record, found))
        if hits:
            if len(hits) != 1:
                # Two records publish this number at this path under this
                # label. Which one the claim means is unknowable, and a lucky
                # pick is not evidence. Method C is not consulted here: it
                # would see the same two candidates and refuse them too, and
                # Method B's own refusal is the more precise diagnosis.
                return {"ok": False, "reason": "ambiguous ledger record",
                        "matches": [found for _, found in hits],
                        "sequences": [record.get("sequence")
                                      for record, _ in hits]}
            found_record, resolved = hits[0]
            return {**resolved, "resolved_by": "search",
                    "resolved_sequence": found_record.get("sequence"),
                    "claimed_sequence": sequence}
    # Method C runs only once the pointer-preserving methods have found
    # nothing, so an honest citation and a Method B hit are both untouched.
    # An ambiguity they already found is never re-litigated: "the evidence is
    # not unique" is the same answer Method C would reach with a wider
    # candidate set, and the stricter method's reason is the sharper diagnosis.
    if (_value_rebind_enabled()
            and named_verdict.get("reason") not in _AMBIGUITY_REFUSALS):
        rebound = _value_rebind(ledger, claim, sequence, path)
        if rebound is not None:
            return rebound
    return named_verdict


def resolve_payload_value(payload, claim: dict) -> dict | None:
    """Resolve one named claim through the shared scoped resolver.

    Coach payloads resolve by their published metric/period/field labels.  A
    ledger adds the required sequence/path citation, but uses the same claim
    shape and this same public resolver rather than a second verifier.
    """
    if _is_ledger(payload):
        return _resolve_ledger_value(payload, claim)
    metric = claim.get("metric")
    field = str(claim.get("field") or "").strip()
    period = claim.get("period")
    if not metric or not field or period is None:
        return None
    matches = [entry for entry in _payload_scopes(payload)
               if entry["metric"] == metric and entry["field"] == field
               and _same_period(entry["period"], period)]
    if not matches:
        return None
    # Duplicate labels are not evidence for one another. They are an
    # ambiguous payload and must not let a lucky match through.
    values = {entry["value"] for entry in matches}
    if len(values) != 1:
        return {"ok": False, "reason": "ambiguous payload scope", "matches": matches}
    return {**matches[0], "ok": True}


# Public name for callers on the researcher path.  Keep it as an alias, not a
# wrapper, so tests and future changes cannot silently create a second resolver.
resolve_ledger_value = resolve_payload_value


def resolve_window(conn, metric: str, period, as_of: str | None = None) -> tuple[str, str] | None:
    """A period spec ('30d', 'all'), an explicit 'YYYY-MM-DD:YYYY-MM-DD' range,
    or a single date. None when nothing resolvable is present — which is a
    rejection, not a default: guessing a window is how compare_periods once
    answered '0% change' over a window nobody asked for.

    Deliberately tolerant of ANNOTATION, because models write the period field
    as freeform prose. Real examples from historical runs: '12w lag 1 vs
    sleep_deep' and '2026-07-21 workout'. Both name a perfectly good window and
    were being rejected outright, which is a false rejection of a true claim —
    the opposite of this module's purpose. The recognisable token is extracted
    from anywhere in the string; if there is none, it still rejects.
    """
    end = mx.anchor_end(conn, metric)
    if not end:
        return None
    # Anchor to the RUN's date, not to now. A relative period like '12w'
    # resolves against the latest data, which keeps moving: the same claim
    # re-checked five days later spans a different window and a different
    # n_pairs (measured: 45 anchored at 2026-07-26, 50 at 2026-07-31). Without
    # this, verifying a stored finding silently grades it against data that did
    # not exist when it was made.
    if as_of and as_of < end:
        end = as_of
    spec = str(period or "").strip()
    if not spec:
        return None
    if spec.lower().startswith("latest"):
        # The metric's most recent day THAT HAS DATA at or before the anchor —
        # not the anchor day itself. vo2_max and body_mass are measured
        # irregularly, so [end, end] is usually empty for them and a perfectly
        # true "latest value" claim was being rejected for want of data.
        row = conn.execute(
            "SELECT MAX(date) FROM daily_metrics WHERE metric = ? AND date <= ?",
            (metric, end)).fetchone()
        if not row or not row[0]:
            return None
        return row[0], row[0]
    rng = _RANGE_IN.search(spec)
    if rng:
        try:
            return mx.parse_range(f"{rng.group(1)}:{rng.group(2)}", end)
        except ValueError:
            return None
    per = _PERIOD_IN.search(spec)
    if per:
        try:
            return mx.parse_period(per.group(1).replace(" ", ""), end)
        except ValueError:
            pass
    single = _DATE_IN.search(spec)
    if single:
        return single.group(0), single.group(0)
    return None


def series_values(conn, metric: str, start: str, end: str) -> dict:
    """Every value legitimately derivable for ONE metric over ONE window."""
    dates, vals, _ = mx.series(conn, metric, start, end)
    if not vals:
        return {}
    st = mx.stats(dates, vals)
    out = {k: float(st[k]) for k in ("mean", "median", "min", "max", "std")
           if isinstance(st.get(k), (int, float))}
    out["latest"] = float(vals[-1])
    out["n_days"] = float(len(vals))
    out["sum"] = float(sum(vals))
    # recent-vs-baseline, matching what the old _rederive_floats offered so
    # delta/pct findings stay verifiable.
    rn = max(1, min(7, len(vals) // 3))
    recent = sum(vals[-rn:]) / rn
    base_vals = vals[:-rn] or vals
    base = sum(base_vals) / len(base_vals)
    out["recent_avg"] = float(recent)
    out["baseline_avg"] = float(base)
    p = mx.pct_change(recent, base)
    if p is not None:
        out["delta_pct"] = float(p)
    sw = mx.slope_per_week(dates, vals)
    if sw is not None:
        out["slope_per_week"] = float(mx.r(sw))
    # The daily values themselves: a finding may legitimately cite one point.
    points = [float(v) for v in vals]
    # ALSO the stored per-day aggregates. mx.series returns only the catalog's
    # chosen aggregate (resting_heart_rate aggregates as 'last'), but avg/min/max
    # are equally real measurements of that day and other surfaces expose them.
    # Measured on the 2026-07-26 run: three resting-heart-rate claims matched
    # `avg` EXACTLY and were being rejected as fabrications. Rejecting a real
    # measurement is the worse error, and deciding which aggregate the model
    # ought to have quoted is a different question from whether the number is
    # real. The strict exact-field path above is unaffected.
    for row in conn.execute(
            "SELECT avg, min, max, last FROM daily_metrics "
            "WHERE metric = ? AND date BETWEEN ? AND ?", (metric, start, end)):
        points.extend(float(v) for v in tuple(row) if isinstance(v, (int, float)))
    out["_points"] = points
    return out


def correlation_values(conn, metric: str, start: str, end: str) -> dict:
    """Correlation outputs for every pair involving `metric` over the window,
    at lags 0 and 1. Scoped to the cited metric rather than every ordered pair
    of every cited metric — the old version ran that full product across three
    lag-sets and every window, which is where most of the 9 seconds went."""
    out: dict[str, list[float]] = {"_points": []}
    if not mx.metric_exists(conn, metric):
        return out
    others = [r[0] for r in conn.execute(
        "SELECT DISTINCT metric FROM daily_metrics WHERE metric NOT IN (?, 'wear_hours')",
        (metric,))]
    vals: list[float] = []
    for other in others:
        for lag in (0, 1):
            for x, y in ((metric, other), (other, metric)):
                xs, ys, meta = C.paired_series(conn, x, y, lag, start, end)
                res = C.correlate(xs, ys)
                if res.get("status") != "ok":
                    continue
                vals.append(float(res["n_pairs"]))
                for k in ("spearman_rho", "spearman_p", "pearson_r", "pearson_p"):
                    v = res.get(k)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                for key in ("pearson_ci95", "spearman_ci95"):
                    vals.extend(float(v) for v in (res.get(key) or [])
                                if isinstance(v, (int, float)))
    out["_points"] = vals
    return out


def workout_values(conn, start: str, end: str) -> dict:
    """Per-session numbers from workouts overlapping the window. Scoped by
    date: the old key pooled the most recent 200 sessions regardless of what
    the finding cited, and those 732 floats alone accepted 67.4% of random
    integers."""
    rows = conn.execute(
        "SELECT duration_min, distance_mi, energy_kcal, avg_heart_rate, "
        "max_heart_rate FROM workouts WHERE date(start_utc) BETWEEN ? AND ?",
        (start, end)).fetchall()
    return {"_points": [float(v) for r in rows for v in tuple(r)
                        if isinstance(v, (int, float))]}


def _impact_period_spec(period):
    """Extract the explicit date(s) used by an impact-volume scope."""
    if isinstance(period, dict):
        starts = period.get("period_starts")
        if isinstance(starts, list) and starts:
            return str(starts[0]), str(starts[-1]), [str(v) for v in starts]
        if period.get("start") and period.get("end"):
            return str(period["start"]), str(period["end"]), None
        return None
    text = str(period or "").strip()
    rng = _RANGE_IN.search(text)
    if rng:
        return rng.group(1), rng.group(2), None
    match = _DATE_IN.search(text)
    if match:
        return match.group(0), match.group(0), None
    return None


def _sql_impact_value(conn, claim: dict) -> tuple[float | None, list[str] | None,
                                                       str | None]:
    """Resolve the impact-volume fields exposed by get_impact_volume.

    The weekly rows and block comparison are computed by the same analysis and
    MCP helpers as the tool. This is a resolver, not a second jog-minute
    calculation: it only selects the field and scope the claim named.
    """
    if claim.get("metric") != "jog_minutes":
        return None, None, None
    spec = _impact_period_spec(claim.get("period"))
    if spec is None:
        return None, None, "unresolvable impact-volume period"
    start, end_or_last, starts = spec
    from . import analysis as A
    from . import mcp_server as MS
    if (claim.get("field") in {"mean", "total"}
            or starts):
        end = (date.fromisoformat(end_or_last) + timedelta(days=6)).isoformat()
    else:
        # A single Monday names one weekly row; an explicit range names the
        # exact range, whose end may itself be the last Monday of a block.
        end_date = date.fromisoformat(end_or_last)
        end = ((end_date + timedelta(days=6)).isoformat()
               if start == end_or_last else end_or_last)
    rows = A.impact_volume(conn, start, end, by="week")
    periods = MS._impact_periods(rows, start, end, "week")
    field = str(claim.get("field") or "")
    if field == "jog_minutes" and not starts:
        hit = next((row for row in periods if row["period_start"] == start), None)
        return ((float(hit["jog_minutes"]), [start], None) if hit else
                (None, None, f"no impact data for week {start}"))
    selected = (set(starts) if starts else
                {row["period_start"] for row in periods
                 if start <= row["period_start"] <= end_or_last})
    chosen = [row for row in periods if row["period_start"] in selected]
    if not chosen:
        return None, None, "no impact data for named block"
    values = [float(row["jog_minutes"]) for row in chosen]
    total = round(sum(values), 1)
    mean = round(total / len(values), 1)
    actuals = {"total": total, "mean": mean}
    if field not in actuals:
        return None, None, f"{field} is not an impact block field"
    return actuals[field], [row["period_start"] for row in chosen], None


def _sql_scoped_value(conn, claim: dict, as_of: str | None = None) -> dict:
    """Return a scoped actual without comparing it to a claimed value."""
    metric = claim.get("metric")
    field = str(claim.get("field") or "").strip()
    if metric == "jog_minutes":
        actual, window, reason = _sql_impact_value(conn, claim)
        return {"actual": actual, "window": window, "scope": "impact_volume",
                "reason": reason or ""}
    if not metric or not mx.metric_exists(conn, metric):
        return {"actual": None, "reason": f"unknown metric {metric!r}"}
    window = resolve_window(conn, metric, claim.get("period"), as_of)
    if window is None:
        return {"actual": None,
                "reason": f"unresolvable period {claim.get('period')!r}"}
    start, end = window
    if _SCAN_COUNT_RE.search(field):
        return {"actual": None,
                "window": [start, end], "reason":
                "scan-level count not independently recomputable"}
    if _CORR_RE.search(field):
        scope, values = "correlation", correlation_values(conn, metric, start, end)
    elif _WORKOUT_RE.search(field):
        scope, values = "workout", workout_values(conn, start, end)
    else:
        scope, values = "series", series_values(conn, metric, start, end)
    if not values or (not values.get("_points") and len(values) <= 1):
        return {"actual": None, "scope": scope, "window": [start, end],
                "reason": f"no data for {metric} over {start}..{end}"}
    if scope == "series" and field in _SERIES_FIELDS:
        return {"actual": values.get(field), "scope": scope,
                "window": [start, end],
                "reason": "" if values.get(field) is not None
                else f"{field} not derivable here"}
    candidates = [v for k, v in values.items()
                  if k != "_points" and isinstance(v, (int, float))]
    candidates += list(values.get("_points") or [])
    # A non-canonical field has no unique actual. Keep the old scoped fallback
    # for ordinary findings, but derivations must use named canonical fields.
    return {"actual": None, "candidates": candidates, "scope": scope,
            "window": [start, end], "reason": "field has no unique actual"}


def _resolve_operand(conn, operand: dict, as_of: str | None, payload) -> dict:
    required = ("metric", "period", "field")
    missing = [key for key in required
               if not operand.get(key) and operand.get(key) != 0]
    if missing:
        return {"ok": False, "reason":
                "derivation operand must name metric, period, and field",
                "missing": missing}
    if operand.get("operation") or operand.get("op") or operand.get("operands"):
        return {"ok": False, "reason": "nested derivations are not supported"}
    payload_hit = resolve_payload_value(payload, operand) if payload is not None else None
    if payload_hit and not payload_hit.get("ok", False):
        return payload_hit
    sql_hit = _sql_scoped_value(conn, operand, as_of) if conn is not None else {}
    payload_actual = payload_hit.get("value") if payload_hit else None
    sql_actual = sql_hit.get("actual")
    if payload_actual is not None and sql_actual is not None:
        if not _close(payload_actual, sql_actual):
            return {"ok": False, "reason": "payload/SQL scope disagreement",
                    "payload_actual": payload_actual, "sql_actual": sql_actual,
                    "gap": abs(payload_actual - sql_actual)}
    actual = payload_actual if payload_actual is not None else sql_actual
    if actual is None:
        return {"ok": False, "reason": sql_hit.get("reason") or
                "operand scope not found in payload or database"}
    return {"ok": True, "actual": float(actual),
            "source": "payload" if payload_actual is not None else "sql",
            "payload_actual": payload_actual, "sql_actual": sql_actual,
            "scope": operand}


def _verify_derivation(conn, num: dict, as_of: str | None, payload) -> dict:
    operation = str(num.get("operation", num.get("op", ""))).strip().lower()
    operands = num.get("operands")
    base = {"metric": num.get("metric"), "period": num.get("period"),
            "field": str(num.get("field") or "") or None,
            "claimed": _as_float(num.get("value")), "ok": False,
            "exact": True, "derived": True}
    if not num.get("metric") or num.get("period") is None or not num.get("field"):
        return {**base, "reason":
                "derived claim must name metric, period, and field"}
    if not operation or not isinstance(operands, list):
        return {**base, "reason":
                "derived claim must name an operation and its operands"}
    resolved = [_resolve_operand(conn, operand, as_of, payload)
                for operand in operands if isinstance(operand, dict)]
    if len(resolved) != len(operands) or not all(r.get("ok") for r in resolved):
        reason = next((r.get("reason") for r in resolved if not r.get("ok")),
                      "all derivation operands must be scoped")
        return {**base, "reason": reason, "operands": resolved}
    if operation in PERCENT_CHANGE_OPERATIONS:
        if len(resolved) != 2:
            return {**base, "reason": "percent change requires two operands"}
        actual = mx.pct_change(resolved[0]["actual"], resolved[1]["actual"])
        if actual is None:
            return {**base, "reason": "percent change from zero is undefined"}
    elif operation in PERCENT_DOWN_OPERATIONS:
        if len(resolved) != 2 or resolved[1]["actual"] == 0:
            return {**base, "reason": "percent down requires two nonzero operands"}
        actual = mx.r((resolved[1]["actual"] - resolved[0]["actual"])
                      / resolved[1]["actual"] * 100)
    elif operation in DIFFERENCE_OPERATIONS:
        if len(resolved) != 2:
            return {**base, "reason": "difference requires two operands"}
        actual = mx.r(resolved[0]["actual"] - resolved[1]["actual"])
    else:
        return {**base, "reason": f"unsupported derivation operation {operation!r}"}
    claimed = base["claimed"]
    if claimed is None:
        return {**base, "actual": actual, "reason": "no numeric value"}
    # Rule R against the claim's own literal, not against `claimed`: precision
    # is a property of the spelling, and _as_float has already eaten it (a
    # ledgered `3` licenses 3.18 at 0dp, `3.0` does not at 1dp). Gated on
    # _is_ledger for the same reason #85 gated it — coach_brief and pipeline
    # keep the legacy tolerance until #61.
    ok = (_rule_r_matches(str(num.get("value")), actual)
          if _is_ledger(payload) else _close(claimed, actual))
    return {**base, "actual": actual, "ok": ok,
            "reason": "" if ok else f"claimed {claimed}, recomputed {actual}",
            "operands": resolved, "operation": operation}


def _verify_presentation_claim(num: dict, payload) -> dict:
    """Verify an exact Python-owned presentation leaf without recomputing it."""
    metric = num.get("metric")
    field = str(num.get("field") or "").strip()
    period = num.get("period")
    claimed = num.get("value")
    base = {"metric": metric, "period": period, "field": field or None,
            "claimed": claimed, "ok": False, "exact": False, "reason": ""}
    if not isinstance(claimed, str) or not claimed:
        return {**base, "reason": "no presentation value"}
    payload_hit = resolve_payload_value(payload, num) if payload is not None else None
    if payload_hit and not payload_hit.get("ok", False):
        return {**base, **payload_hit}
    if not payload_hit:
        return {**base, "reason": "presentation leaf not found in payload"}
    actual = payload_hit["value"]
    ok = isinstance(actual, str) and claimed == actual
    verdict = {**base, "ok": ok, "exact": True, "actual": actual,
               "payload_actual": actual, "scope": "payload",
               "tier": payload_hit.get("tier"),
               "reason": "" if ok else
               f"claimed {claimed!r}, payload actual {actual!r}"}
    for key in ("resolved_by", "resolved_sequence", "claimed_sequence",
                "claimed_path"):
        if key in payload_hit:
            verdict[key] = payload_hit[key]
    return verdict


def verify_number(conn, num: dict, as_of: str | None = None,
                  payload=None, tool_results=None) -> dict:
    """Check ONE claimed number against what its own (metric, period, field)
    actually is.

    Returns a dict carrying the verdict and enough context to explain it. `ok`
    is True only when the claim matched something real in its declared scope;
    `exact` distinguishes a canonical-field match from the weaker scoped-set
    fallback used for model-invented field names.
    """
    if payload is None:
        payload = tool_results
    if num.get("operation") is not None or num.get("op") is not None \
            or num.get("operands") is not None:
        return _verify_derivation(conn, num, as_of, payload)

    # Presentation leaves are exact strings (for example ``7h 20m`` or
    # ``11:41 PM``).  They are facts published by Python, not values for this
    # verifier to convert back into raw units.
    if (str(num.get("field") or "").strip() == "presentation"
            and isinstance(num.get("value"), str)):
        return _verify_presentation_claim(num, payload)

    claimed = _as_float(num.get("value"))
    metric = num.get("metric")
    field = str(num.get("field") or "").strip()
    period = num.get("period")
    base = {"metric": metric, "period": period, "field": field or None,
            "claimed": claimed, "ok": False, "exact": False, "reason": ""}

    if claimed is None:
        return {**base, "reason": "no numeric value"}
    payload_hit = resolve_payload_value(payload, num) if payload is not None else None
    # Metric omission is general only for an exact, unambiguous result path in
    # a ledger. Non-ledger payloads still require a metric, and the ledger
    # resolver itself retains all source/path refusals.
    if metric is None and not _is_ledger(payload):
        return {**base, "reason": "no metric named — unverifiable"}
    if payload_hit and not payload_hit.get("ok", False):
        return {**base, **payload_hit}
    if payload_hit:
        payload_actual = payload_hit["value"]
        sql_hit = (_sql_scoped_value(conn, num, as_of)
                   if conn is not None else {})
        sql_actual = sql_hit.get("actual")
        if sql_actual is not None and not _close(payload_actual, sql_actual):
            return {**base, "payload_actual": payload_actual,
                    "sql_actual": sql_actual, "ok": False,
                    "reason": "payload/SQL scope disagreement"}
        ok = (_as_float(claimed) == _as_float(payload_actual)
              if _is_ledger(payload) else _close(claimed, payload_actual))
        verdict = {**base, "ok": ok, "exact": True, "actual": payload_actual,
                   "payload_actual": payload_actual, "sql_actual": sql_actual,
                   "scope": "payload", "tier": payload_hit.get("tier"),
                   "reason": "" if ok else
                   f"claimed {claimed}, payload actual {payload_actual}"}
        # Carry the record-search marker up only when there is one, so a
        # verdict is unchanged whenever the model's own citation resolved.
        for key in ("resolved_by", "resolved_sequence", "claimed_sequence",
                    "claimed_path"):
            if key in payload_hit:
                verdict[key] = payload_hit[key]
        return verdict
    if conn is None:
        return {**base, "reason": "scope not found in payload and no database fallback"}
    if metric == "jog_minutes":
        sql_hit = _sql_scoped_value(conn, num, as_of)
        actual = sql_hit.get("actual")
        if actual is None:
            return {**base, **sql_hit}
        ok = _close(claimed, actual)
        return {**base, "ok": ok, "exact": True, "actual": actual,
                "scope": sql_hit.get("scope"), "window": sql_hit.get("window"),
                "reason": "" if ok else f"claimed {claimed}, actual {actual}"}
    if not mx.metric_exists(conn, metric):
        return {**base, "reason": f"unknown metric {metric!r}"}
    window = resolve_window(conn, metric, period, as_of)
    if window is None:
        return {**base, "reason": f"unresolvable period {period!r}"}
    start, end = window

    if _SCAN_COUNT_RE.search(field):
        return {**base, "reason": "scan-level count not independently recomputable"}
    if _CORR_RE.search(field):
        scope, values = "correlation", correlation_values(conn, metric, start, end)
    elif _WORKOUT_RE.search(field):
        scope, values = "workout", workout_values(conn, start, end)
    else:
        scope, values = "series", series_values(conn, metric, start, end)
    base.update({"scope": scope, "window": [start, end]})

    if not values or (not values.get("_points") and len(values) <= 1):
        return {**base, "reason": f"no data for {metric} over {start}..{end}"}

    # Exact field match is the real test; everything else is a weaker check.
    if scope == "series" and field in _SERIES_FIELDS:
        actual = values.get(field)
        if actual is None:
            return {**base, "reason": f"{field} not derivable here"}
        ok = _close(claimed, actual)
        return {**base, "ok": ok, "exact": True, "actual": actual,
                "reason": "" if ok else f"claimed {claimed}, actual {actual}"}

    candidates = [v for k, v in values.items()
                  if k != "_points" and isinstance(v, (int, float))]
    candidates += list(values.get("_points") or [])
    ok = any(_close(claimed, c) for c in candidates)
    return {**base, "ok": ok, "exact": False, "n_candidates": len(candidates),
            "reason": "" if ok else
            f"no value within tolerance among {len(candidates)} for "
            f"{metric} {start}..{end}"}


def verify_finding(conn, finding: dict, as_of: str | None = None,
                   payload=None, tool_results=None) -> dict:
    """Verify every number a finding carries.

    A finding is verified only when it stakes at least one checkable claim AND
    every checkable claim holds. A finding whose numbers are all unverifiable
    is NOT quietly accepted: it asserts something about the athlete's body with
    nothing behind it, which is the case this gate exists for.
    """
    if payload is None:
        payload = tool_results
    results = [verify_number(conn, n, as_of, payload=payload)
               for n in (finding.get("numbers") or [])]
    _UNCHECKABLE = ("no numeric value", "no metric named — unverifiable",
                    "scan-level count not independently recomputable")
    checkable = [r for r in results
                 if r["reason"] not in _UNCHECKABLE
                 and not r["reason"].startswith("unknown metric")]
    ok = bool(checkable) and all(r["ok"] for r in checkable)
    return {"claim": finding.get("claim"), "ok": ok, "numbers": results,
            "n_checkable": len(checkable),
            "n_failed": sum(1 for r in checkable if not r["ok"])}


def _binding_tier_counts(numbers: list[dict]) -> dict[str, int]:
    """Count successful claims by provenance assurance tier.

    Tier 1 proves exact result-path provenance. Tier 2 additionally identifies
    a canonical metric that the verifier can cross-check against SQL. Failed
    claims are deliberately not counted as bound claims.
    """
    return {
        "path": sum(1 for number in numbers
                    if number.get("ok") and number.get("tier") == "path"),
        "metric": sum(1 for number in numbers
                      if number.get("ok") and number.get("tier") == "metric"),
    }


def _value_rebind_counts(numbers: list[dict]) -> dict[str, int]:
    """Count what Method C actually did to a response's claims.

    ``sequence_changed`` is the number that decides whether the method is safe
    to leave on: it is how often Python bound a claim to a DIFFERENT tool call
    than the model named. A rebind whose sequence is unchanged only corrected
    the leaf inside the record the model already cited, which is a much smaller
    claim about the model's honesty than moving the citation to another call.
    ``ambiguous`` is the refusal arm — claims the gate declined rather than
    guessed — and a rising one is the method working, not failing.
    """
    rebound = [number for number in numbers
               if number.get("ok") and number.get("resolved_by") == "value_rebind"]
    return {
        "rebound": len(rebound),
        "ambiguous": sum(1 for number in numbers
                         if number.get("reason") == "ambiguous value rebind"),
        "sequence_changed": sum(
            1 for number in rebound
            if number.get("resolved_sequence") != number.get("claimed_sequence")),
    }


def _rebind_instrumentation(numbers: list[dict]) -> dict:
    """Splat the Method C counters into a verdict, and only when C is on.

    Absent with the flag off, so a flag-off verdict — including the public
    ``/v1/ask`` verification shape built on top of it — is byte-identical to
    the version before Method C existed.
    """
    if not _value_rebind_enabled():
        return {}
    return {"rebind_counts": _value_rebind_counts(numbers)}


def verify_all(conn, findings: list[dict], as_of: str | None = None,
               payload=None, tool_results=None) -> list[dict]:
    if payload is None:
        payload = tool_results
    return [verify_finding(conn, f, as_of, payload=payload) for f in findings]


def verify_coach_claims(conn, prose: str, claims, as_of: str | None = None,
                        payload=None, tool_results=None) -> dict:
    """Verify numbered coach prose through the existing scoped verifier."""
    if payload is None:
        payload = tool_results
    # Number-free compatibility prose does not need a claim record. Once prose
    # contains a number, every number must be represented by a scoped claim.
    unsupported = G._numeric_tokens(prose)
    degenerate = _is_degenerate_repeated_bullets(prose)
    if not unsupported and not degenerate:
        return {"ok": True, "grounded": True, "unsupported": [],
                "claims": [], "tier_counts": {"path": 0, "metric": 0},
                **_rebind_instrumentation([])}
    if not isinstance(claims, list) or not claims:
        return {"ok": False, "grounded": False, "unsupported": unsupported,
                "reason": ("degenerate repeated-token generation"
                           if degenerate else
                           "numbered coach prose has no structured claims"),
                "tier_counts": {"path": 0, "metric": 0},
                **_rebind_instrumentation([])}

    structural_claims = _structural_claims(payload)
    verdict = verify_finding(
        conn, {"claim": prose, "numbers": claims}, as_of=as_of,
        payload=payload)
    verified_claims = [claim for claim, result in
                       zip(claims, verdict["numbers"])
                       if isinstance(claim, dict) and result.get("ok")]
    # Keep the complete ask prose here. The occurrence multiset in
    # _coach_grounding is not sound when sentence fragments share claims.
    grounded, bad = _coach_grounding(
        prose, verified_claims, payload=payload)
    tier_counts = _binding_tier_counts(verdict["numbers"])
    if not verdict["ok"]:
        failed = next((n for n in verdict["numbers"] if not n["ok"]), None)
        return {"ok": False, "grounded": False,
                "unsupported": bad,
                "reason": (failed or {}).get("reason", "claim verification failed"),
                "verdict": verdict, "structural_claims": structural_claims,
                "tier_counts": tier_counts,
                **_rebind_instrumentation(verdict["numbers"])}

    return {"ok": grounded, "grounded": grounded, "unsupported": bad,
            "reason": "" if grounded else "prose number is not in claims",
            "verdict": verdict, "structural_claims": structural_claims,
            "tier_counts": tier_counts,
            **_rebind_instrumentation(verdict["numbers"])}


def verify_research_claims(prose: str, claims, ledger) -> dict:
    """Verify researcher prose through the same scoped claim vocabulary.

    The ledger is evidence only; ``tools_used`` and prose are not provenance.
    A research response must contain a call ledger and at least one structured
    claim, even when its prose happens to contain no numbers.
    """
    cleaned_prose = G.strip_dates_and_names(prose or "")
    figure_total = len(G._NUM_RE.findall(cleaned_prose))
    if not _is_ledger(ledger):
        return {"ok": False, "grounded": False, "unsupported": [],
                "reason": "research answer has no tool-call ledger",
                "figures_verified": 0, "figures_total": figure_total}
    if not isinstance(claims, list) or not claims:
        return {"ok": False, "grounded": False,
                "unsupported": [],
                "reason": "research answer has no structured claims",
                "figures_verified": 0, "figures_total": figure_total}
    verdict = verify_finding(None, {"claim": prose, "numbers": claims},
                             payload=ledger)
    verified_claims = [claim for claim, result in
                       zip(claims, verdict["numbers"])
                       if isinstance(claim, dict) and result.get("ok")]
    grounded, bad = _research_grounding(prose, verified_claims, payload=ledger)
    if not verdict["ok"]:
        failed = next((number for number in verdict["numbers"]
                       if not number.get("ok")), None)
        return {"ok": False, "grounded": False, "unsupported": bad,
                "reason": (failed or {}).get("reason", "claim verification failed"),
                "verdict": verdict, "figures_verified": 0,
                "figures_total": figure_total,
                **_rebind_instrumentation(verdict["numbers"])}
    if not grounded:
        return {"ok": False, "grounded": False, "unsupported": bad,
                "reason": "prose number is not in verified claims",
                "verdict": verdict, "figures_verified": figure_total - len(bad),
                "figures_total": figure_total,
                **_rebind_instrumentation(verdict["numbers"])}
    return {"ok": True, "grounded": True, "unsupported": [],
            "reason": "", "verdict": verdict,
            "figures_verified": figure_total, "figures_total": figure_total,
            **_rebind_instrumentation(verdict["numbers"])}


def judge_evidence(verdicts: list[dict]) -> list[dict]:
    """Compact, LABELLED evidence for the judge prompt.

    Replaces the 95 KB of unlabeled floats the judge was shipped and then told
    to ignore. Labelled claim/actual pairs are the thing a judge can actually
    reason about; an anonymous float bag is not.
    """
    out = []
    for v in verdicts:
        nums = [{"metric": n.get("metric"), "field": n.get("field"),
                 "period": n.get("period"), "claimed": n.get("claimed"),
                 "actual": n.get("actual"), "verified": n["ok"],
                 "exact": n.get("exact", False)}
                for n in v["numbers"]]
        out.append({"claim": v["claim"], "verified": v["ok"], "numbers": nums})
    return out
