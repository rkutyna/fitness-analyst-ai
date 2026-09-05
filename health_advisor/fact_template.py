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

from . import deepdive_verify as _verify
from . import normalize


_KEY_SEPARATOR = "|"
_KEY_PART_RE = re.compile(r"^(metric|period|field)=(.*)$")
_ATTACHMENT_KEY_PART_RE = re.compile(r"^(table|column|row|trend)=(.*)$")
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
_ADVICE_PREFIX = "advice:"


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
        if metric is None or period is None:
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


def build_fact_set(ledger: list[dict]) -> dict[str, dict]:
    """Build the closed fact set from result leaves in this call's ledger.

    Only metric-owned result leaves with an explicit period participate;
    weekly mean leaves may use their enclosing ``week_start``.
    Arguments, context fields, elided results, and metricless workout rows are
    excluded because they cannot round-trip through the natural identity tuple.
    Duplicate identities are removed from the published set rather than
    allowing a key to choose between two ledger entries.
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
            if (not entry.get("metric")
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
            display = str(entry["value"])
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

    Cell values and units are copied verbatim from the table.  For numeric
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
                        "display": str(row[column_index]),
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
                        "display": str(value),
                        "source": trend_source,
                    }))

    return _publish_unambiguous(candidates)


def render_fact_set(facts: dict[str, dict]) -> str:
    """Render facts for the final model turn in deterministic JSON."""
    return json.dumps(facts or {}, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


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
