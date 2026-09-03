"""Parse the HealthKit-direct delta envelope into canonical record rows.

This module is deliberately a pure boundary: it validates and normalizes one
already-decoded dictionary and does not open a database or apply anchors and
tombstones.  Protocol versions other than the server's supported version are
refused, and anchors for types outside the persistence vocabulary are returned
as explicitly rejected status entries rather than as advanceable anchors.  The
vocabulary and unit arithmetic belong to ``normalize``;
``db.record_key`` supplies the existing window/value-aware identity.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

from . import db
from . import normalize as nz

# The origin tag this parser stamps on records.origin. Named for its writer
# rather than the bare name it had: subjective.py and scripts/reingest_vo2max.py
# each defined the same bare name with a DIFFERENT value ("checkin",
# "backfill"), so it read as "the" origin at a call site while meaning three
# different things. Renamed, not unified — the three values are all correct.
# #127 (F-50); found by tests/test_constant_collisions.py.
HEALTHKIT_ORIGIN = "healthkit"

_TOP_FIELDS = frozenset({
    "protocol_version", "device", "app_version", "batch_id",
    "batch_sequence", "sent_at", "anchors", "samples", "deletions",
    "workouts", "daily_totals",
})
_DEVICE_FIELDS = frozenset({"id", "name", "model"})
_SAMPLE_DEVICE_FIELDS = frozenset({"name", "model"})
# Every field the source_revision object MAY carry (unknown ones are rejected)…
_REVISION_FIELDS = frozenset({"source_name", "bundle_id", "version",
                              "product_type", "operating_system_version"})
# …and the two the contract actually requires. HealthKit does not always
# populate the rest, and demanding them drops real samples into `unhandled`.
_REVISION_REQUIRED = ("source_name", "bundle_id")
_SAMPLE_FIELDS = frozenset({
    "kind", "hk_uuid", "type_identifier", "start", "end", "value",
    "unit", "source_revision", "device", "local_date",
})
_WORKOUT_FIELDS = frozenset({
    "hk_uuid", "workout_activity_type", "start", "end", "duration_min",
    "energy_kcal", "distance_mi", "avg_heart_rate", "max_heart_rate",
    "source_revision",
})
_DAILY_TOTAL_FIELDS = frozenset({
    "type_identifier", "local_date", "value", "unit", "interval",
    "state", "queried_at",
})
_ANCHOR_FIELDS = frozenset({"type_identifier", "from", "to"})
_DELETION_FIELDS = frozenset({"hk_uuid", "type_identifier"})
_HK_WORKOUT_TYPE_IDENTIFIER = "HKWorkoutTypeIdentifier"

# The one shared list of metrics for which the phone pulls Apple's
# consolidated daily statistics.  The verifier imports this rather than
# maintaining a second list that could silently drift from the wire.
D19_TOTAL_METRICS = (
    "distance_walking_running", "flights_climbed", "step_count",
)
DAILY_TOTAL_METRICS = D19_TOTAL_METRICS


class PayloadError(ValueError):
    """The envelope shape is not a HealthKit-direct payload."""


def _mapping(value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise PayloadError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def _list(value: Any, what: str) -> list:
    if not isinstance(value, list):
        raise PayloadError(f"{what} must be a JSON array, got {type(value).__name__}")
    return value


def _reject_unknown(mapping: dict, allowed: frozenset[str], what: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise PayloadError(f"{what} has unknown field(s): {', '.join(unknown)}")


def _required(mapping: dict, fields: tuple[str, ...], what: str) -> list[str]:
    """The required fields this mapping lacks, in declaration order.

    Returns the list rather than a bool so the caller can name them. A sample
    routed to `unhandled` as "missing required field" is an absence nobody can
    explain, which is the one kind of answer this project is not allowed to
    give.
    """
    return [field for field in fields if field not in mapping]


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    # A server cannot derive the client's local calendar day from a naive time.
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_number(value: Any) -> float | None:
    """A JSON number, unlike the legacy sample parser's numeric coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return _number(value)


def _revision_json(revision: dict) -> str:
    return json.dumps(revision, sort_keys=True, separators=(",", ":"))


def _type_info(type_identifier: str, category_value: Any):
    """Resolve through normalize's catalog, never through a local map."""
    if type_identifier in nz.HK_QUANTITY:
        return "quantity", [nz.hk_quantity_to_canonical(type_identifier)]
    if type_identifier == nz.HK_SLEEP_TYPE_IDENTIFIER:
        return "sleep", nz.SLEEP_VALUE_MAP.get(category_value)
    if type_identifier in nz.HK_CATEGORY:
        return "category", [nz.HK_CATEGORY[type_identifier]["metric"]]
    return None, None


def _anchor_type_supported(type_identifier: str) -> bool:
    """Whether this server can persist samples for a HealthKit anchor type."""
    return (
        type_identifier in nz.HK_QUANTITY
        or type_identifier == nz.HK_SLEEP_TYPE_IDENTIFIER
        or type_identifier in nz.HK_CATEGORY
        or type_identifier == _HK_WORKOUT_TYPE_IDENTIFIER
    )


def _unhandled(unhandled: list[str], index: int, reason: str) -> None:
    unhandled.append(f"samples[{index}]: {reason}")


def _workout_unhandled(unhandled: list[str], index: int, reason: str) -> None:
    unhandled.append(f"workouts[{index}]: {reason}")


def _daily_total_unhandled(unhandled: list[str], index: int, reason: str) -> None:
    unhandled.append(f"daily_totals[{index}]: {reason}")


def _record(metric: str, value: float, unit: str | None,
            start_dt: datetime, end_dt: datetime, source: str,
            hk_uuid: str, type_identifier: str, revision_json: str | None,
            device_id: str, *, source_value: Any, source_metric: str) -> dict:
    start_utc = nz.to_utc_iso(start_dt)
    end_utc = nz.to_utc_iso(end_dt)
    # local_date is intentionally derived here.  A client field of the same
    # name is accepted for forwards compatibility but is never consulted.
    local_date = nz.local_date_of(end_dt if metric.startswith("sleep_") else start_dt)
    return {
        "metric": metric,
        "value": value,
        "unit": unit,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "start_local": nz.local_naive(start_dt),
        "local_date": local_date,
        "source": source,
        "origin": HEALTHKIT_ORIGIN,
        "dedupe_key": db.record_key(
            metric, start_utc, end_utc, value, unit or "", source,
            source_metric=source_metric, source_value=source_value,
        ),
        "hk_uuid": hk_uuid,
        "hk_type_identifier": type_identifier,
        "source_revision_json": revision_json,
        "hk_device_id": device_id,
    }


def _parse_sample(sample: dict, index: int, device_id: str,
                  records: list[dict], pairs: set[tuple[str, str]],
                  unhandled: list[str]) -> None:
    # `device` is NOT required: HealthKit returns no device for a great many
    # samples, and demanding one drops real data into `unhandled` silently.
    # The wire contract lists it as optional; this used to require it.
    required = ("kind", "hk_uuid", "type_identifier", "start", "end", "value",
                "source_revision")
    missing = _required(sample, required, f"samples[{index}]")
    if missing:
        _unhandled(unhandled, index,
                   f"missing required field(s): {', '.join(missing)}")
        return
    kind = sample["kind"]
    type_identifier = sample["type_identifier"]
    if kind not in ("quantity", "category"):
        _unhandled(unhandled, index, f"unknown sample kind {kind!r}")
        return
    if kind == "quantity" and "unit" not in sample:
        _unhandled(unhandled, index, "quantity sample is missing unit")
        return
    if kind == "category" and sample.get("unit") is not None:
        _unhandled(unhandled, index, "category unit must be null when present")
        return
    if not all(_text(sample.get(field)) for field in ("hk_uuid", "type_identifier")):
        _unhandled(unhandled, index, "missing or invalid identity")
        return
    start_dt, end_dt = _parse_dt(sample["start"]), _parse_dt(sample["end"])
    if start_dt is None or end_dt is None or end_dt < start_dt:
        _unhandled(unhandled, index, "unparseable or inverted start/end")
        return

    revision = sample["source_revision"]
    # `device` is absent for a great many real HealthKit samples — the watch
    # writes one, a manual entry does not. Absent and null are both fine and
    # mean "no device", not "malformed".
    sample_device = sample.get("device")
    if not isinstance(revision, dict):
        _unhandled(unhandled, index, "source_revision is not an object")
        return
    if sample_device is not None and not isinstance(sample_device, dict):
        _unhandled(unhandled, index, "device is present but is not an object")
        return
    # These are part of the sample's shape, so an unknown nested field is still
    # a whole-batch rejection rather than point attrition.
    _reject_unknown(revision, _REVISION_FIELDS,
                    f"samples[{index}].source_revision")
    if sample_device is not None:
        _reject_unknown(sample_device, _SAMPLE_DEVICE_FIELDS,
                        f"samples[{index}].device")
    if not all(_text(revision.get(field)) for field in _REVISION_REQUIRED):
        _unhandled(unhandled, index,
                   "source_revision needs a non-empty source_name and bundle_id")
        return

    source = revision["source_name"]
    if kind == "category" and not isinstance(sample["value"], str):
        _unhandled(unhandled, index, "category value is not a string")
        return
    info_kind, metrics = _type_info(type_identifier, sample["value"])
    if info_kind is None:
        _unhandled(unhandled, index,
                   f"unknown type_identifier {type_identifier!r}")
        return
    if not metrics:
        _unhandled(unhandled, index,
                   f"unknown category value {sample['value']!r} for {type_identifier!r}")
        return
    if kind == "quantity" and info_kind != "quantity":
        _unhandled(unhandled, index, "quantity kind does not match type_identifier")
        return
    if kind == "category" and info_kind == "quantity":
        _unhandled(unhandled, index, "category kind does not match type_identifier")
        return

    revision_json = _revision_json(revision)
    if info_kind == "quantity":
        raw_value = _number(sample["value"])
        if raw_value is None:
            _unhandled(unhandled, index, "quantity value is not a finite number")
            return
        raw_unit = sample.get("unit")
        if raw_unit is not None and not isinstance(raw_unit, str):
            _unhandled(unhandled, index, "quantity unit is not a string")
            return
        metric = metrics[0]
        try:
            convert, unit = nz.unit_converter(metric, raw_unit)
            # HealthKit-specific: '%' quantities arrive as a 0–1 ratio. Most are
            # recovered by value; walking asymmetry has to be keyed on the type,
            # because sub-1% readings there are real. See normalize (#143).
            value = nz.hk_canonical_value(type_identifier, metric,
                                          convert(raw_value))
        except nz.UnitError as exc:
            _unhandled(unhandled, index, f"{exc}; point dropped")
            return
        row = _record(metric, value, unit, start_dt, end_dt, source,
                      sample["hk_uuid"], type_identifier, revision_json, device_id,
                      source_value=sample["value"], source_metric=type_identifier)
        records.append(row)
        pairs.add((metric, row["local_date"]))
        return

    # Category samples are interval/event data. Sleep is the one category
    # family whose canonical rows must retain a numeric duration for derive.py;
    # other category rows follow the existing category encodings in normalize.
    if info_kind == "sleep":
        value = (end_dt - start_dt).total_seconds() / 60.0
        unit = "min"
    else:
        spec = nz.HK_CATEGORY[type_identifier]
        if spec["mode"] == "duration":
            value = (end_dt - start_dt).total_seconds() / 60.0
            unit = "min"
        elif spec["mode"] == "flag":
            value = 1.0 if sample["value"] in spec.get("positive", set()) else 0.0
            unit = "count"
        else:
            value, unit = 1.0, "count"
    for metric in metrics:
        row = _record(metric, value, unit, start_dt, end_dt, source,
                      sample["hk_uuid"], type_identifier, revision_json, device_id,
                      source_value=sample["value"],
                      source_metric=f"{type_identifier}:{metric}")
        records.append(row)
        pairs.add((metric, row["local_date"]))


def _parse_workout(workout: dict, index: int, workouts: list[dict],
                   workout_dates: set[str], unhandled: list[str]) -> None:
    required = ("hk_uuid", "workout_activity_type", "start", "end",
                "duration_min", "source_revision")
    missing = _required(workout, required, f"workouts[{index}]")
    if missing:
        _workout_unhandled(unhandled, index,
                           f"missing required field(s): {', '.join(missing)}")
        return
    if not all(_text(workout.get(field))
               for field in ("hk_uuid", "workout_activity_type")):
        _workout_unhandled(unhandled, index, "missing or invalid identity")
        return

    start_dt, end_dt = _parse_dt(workout["start"]), _parse_dt(workout["end"])
    if start_dt is None or end_dt is None or end_dt < start_dt:
        _workout_unhandled(unhandled, index, "unparseable or inverted start/end")
        return

    duration_min = _json_number(workout["duration_min"])
    if duration_min is None or duration_min < 0:
        _workout_unhandled(unhandled, index,
                           "duration_min is not a non-negative finite number")
        return

    revision = workout["source_revision"]
    if not isinstance(revision, dict):
        _workout_unhandled(unhandled, index, "source_revision is not an object")
        return
    _reject_unknown(revision, _REVISION_FIELDS,
                    f"workouts[{index}].source_revision")
    if not all(_text(revision.get(field)) for field in _REVISION_REQUIRED):
        _workout_unhandled(
            unhandled, index,
            "source_revision needs a non-empty source_name and bundle_id")
        return

    optional_numbers: dict[str, float | None] = {}
    for field in ("energy_kcal", "distance_mi", "avg_heart_rate", "max_heart_rate"):
        if field not in workout or workout[field] is None:
            optional_numbers[field] = None
            continue
        value = _json_number(workout[field])
        if value is None:
            _workout_unhandled(unhandled, index,
                               f"{field} is not a finite number or null")
            return
        optional_numbers[field] = value

    start_utc, end_utc = nz.to_utc_iso(start_dt), nz.to_utc_iso(end_dt)
    workout_type = nz.workout_label(workout["workout_activity_type"])
    row = {
        "workout_type": workout_type,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "local_date": nz.local_date_of(start_dt),
        "duration_min": duration_min,
        "energy_kcal": optional_numbers["energy_kcal"],
        "distance_mi": optional_numbers["distance_mi"],
        "unit_distance": "mi" if "distance_mi" in workout
        and workout["distance_mi"] is not None else None,
        "source": revision["source_name"],
        "route_ref": None,
        "avg_heart_rate": optional_numbers["avg_heart_rate"],
        "max_heart_rate": optional_numbers["max_heart_rate"],
        "dedupe_key": db.workout_key(workout_type, start_utc, end_utc),
        "hk_uuid": workout["hk_uuid"],
    }
    workouts.append(row)
    workout_dates.add(row["local_date"])


def _parse_daily_total(total: dict, index: int, device_id: str,
                       daily_totals: list[dict], daily_total_dates: set[str],
                       unhandled: list[str]) -> None:
    required = ("type_identifier", "local_date", "value", "unit", "interval",
                "state", "queried_at")
    missing = _required(total, required, f"daily_totals[{index}]")
    if missing:
        _daily_total_unhandled(
            unhandled, index,
            f"missing required field(s): {', '.join(missing)}")
        return
    if not _text(total["type_identifier"]):
        _daily_total_unhandled(unhandled, index,
                               "type_identifier must be a non-empty string")
        return
    if not _text(total["local_date"]):
        _daily_total_unhandled(unhandled, index,
                               "local_date must be a non-empty string")
        return
    try:
        parsed_date = date.fromisoformat(total["local_date"])
    except (TypeError, ValueError):
        parsed_date = None
    if parsed_date is None or parsed_date.isoformat() != total["local_date"]:
        _daily_total_unhandled(unhandled, index,
                               "local_date must be YYYY-MM-DD")
        return
    value = _json_number(total["value"])
    if value is None or value < 0:
        _daily_total_unhandled(
            unhandled, index,
            "value is not a non-negative finite number")
        return
    if not _text(total["unit"]):
        _daily_total_unhandled(unhandled, index,
                               "unit must be a non-empty string")
        return
    if total["interval"] != "day":
        _daily_total_unhandled(unhandled, index,
                               "interval must be 'day'")
        return
    if total["state"] not in ("provisional", "settled"):
        _daily_total_unhandled(unhandled, index,
                               "state must be 'provisional' or 'settled'")
        return
    queried_at = _parse_dt(total["queried_at"])
    if queried_at is None:
        _daily_total_unhandled(unhandled, index,
                               "queried_at must be an aware ISO timestamp")
        return

    type_identifier = total["type_identifier"]
    if type_identifier not in nz.HK_QUANTITY:
        _daily_total_unhandled(
            unhandled, index,
            f"unknown type_identifier {type_identifier!r}")
        return
    metric = nz.hk_quantity_to_canonical(type_identifier)
    try:
        convert, unit = nz.unit_converter(metric, total["unit"])
        value = nz.hk_canonical_value(type_identifier, metric, convert(value))
    except nz.UnitError as exc:
        _daily_total_unhandled(unhandled, index, f"{exc}; point dropped")
        return

    daily_totals.append({
        "metric": metric,
        "local_date": total["local_date"],
        "value": value,
        "unit": unit,
        "interval": total["interval"],
        "state": total["state"],
        "device_id": device_id,
        "queried_at": total["queried_at"],
    })
    daily_total_dates.add(total["local_date"])


def parse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return canonical rows, advanceable anchors, and envelope IDs.

    Unsupported protocol versions fail the whole payload. Unsupported anchor
    types remain visible in ``anchor_results`` but are excluded from the
    advanceable ``anchors`` collection.
    """
    envelope = _mapping(payload, "payload")
    _reject_unknown(envelope, _TOP_FIELDS, "payload")
    required = ("protocol_version", "device", "app_version", "batch_id",
                "batch_sequence", "sent_at", "anchors", "samples",
                "deletions", "workouts")
    missing = _required(envelope, required, "payload")
    if missing:
        raise PayloadError(
            f"payload is missing required envelope field(s): {', '.join(missing)}")
    if envelope["protocol_version"] != 1:
        raise PayloadError(f"unsupported protocol_version {envelope['protocol_version']!r}")
    device = _mapping(envelope["device"], "payload.device")
    _reject_unknown(device, _DEVICE_FIELDS, "payload.device")
    if not all(_text(device.get(field)) for field in _DEVICE_FIELDS):
        raise PayloadError("payload.device requires non-empty id, name, and model")
    if not all(_text(envelope.get(field)) for field in ("app_version", "batch_id", "sent_at")):
        raise PayloadError("payload app_version, batch_id, and sent_at must be strings")
    if (isinstance(envelope["batch_sequence"], bool)
            or not isinstance(envelope["batch_sequence"], int)
            or envelope["batch_sequence"] < 0):
        raise PayloadError("payload.batch_sequence must be a non-negative integer")

    anchors_wire = _list(envelope["anchors"], "payload.anchors")
    anchors: list[dict] = []
    anchor_results: list[dict] = []
    rejected_anchors: list[dict] = []
    unhandled: list[str] = []
    for i, anchor in enumerate(anchors_wire):
        anchor = _mapping(anchor, f"payload.anchors[{i}]")
        _reject_unknown(anchor, _ANCHOR_FIELDS, f"payload.anchors[{i}]")
        missing = _required(anchor, ("type_identifier", "from", "to"),
                            f"payload.anchors[{i}]")
        if missing:
            raise PayloadError(
                f"payload.anchors[{i}] is missing required field(s): "
                f"{', '.join(missing)}")
        if not _text(anchor["type_identifier"]):
            raise PayloadError(f"payload.anchors[{i}].type_identifier must be a string")
        if anchor["from"] is not None and not _text(anchor["from"]):
            raise PayloadError(f"payload.anchors[{i}].from must be a string or null")
        if not _text(anchor["to"]):
            raise PayloadError(f"payload.anchors[{i}].to must be a string")
        if not _anchor_type_supported(anchor["type_identifier"]):
            reason = (
                f"unknown type_identifier {anchor['type_identifier']!r}; "
                "anchor is not advanceable"
            )
            unhandled.append(
                f"anchors[{i}]: unknown type_identifier {anchor['type_identifier']!r}"
            )
            rejected_anchors.append(anchor)
            anchor_results.append({
                "index": i,
                "type_identifier": anchor["type_identifier"],
                "accepted": False,
                "reason": reason,
            })
        else:
            anchors.append(anchor)
            anchor_results.append({
                "index": i,
                "type_identifier": anchor["type_identifier"],
                "accepted": True,
            })

    deletions = _list(envelope["deletions"], "payload.deletions")
    for i, deletion in enumerate(deletions):
        deletion = _mapping(deletion, f"payload.deletions[{i}]")
        _reject_unknown(deletion, _DELETION_FIELDS, f"payload.deletions[{i}]")
        missing = _required(deletion, ("hk_uuid", "type_identifier"),
                            f"payload.deletions[{i}]")
        if missing:
            raise PayloadError(
                f"payload.deletions[{i}] is missing required field(s): "
                f"{', '.join(missing)}")
        if not _text(deletion["hk_uuid"]) or not _text(deletion["type_identifier"]):
            raise PayloadError(f"payload.deletions[{i}] has invalid identity")
        if (deletion["type_identifier"] not in nz.HK_QUANTITY
                and deletion["type_identifier"] != nz.HK_SLEEP_TYPE_IDENTIFIER
                and deletion["type_identifier"] not in nz.HK_CATEGORY):
            unhandled.append(
                f"deletions[{i}]: unknown type_identifier {deletion['type_identifier']!r}"
            )

    samples = _list(envelope["samples"], "payload.samples")
    records: list[dict] = []
    pairs: set[tuple[str, str]] = set()
    # Unknown sample fields reject the batch before any point can be emitted.
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            _unhandled(unhandled, i, "sample is not an object")
            continue
        sample = _mapping(sample, f"payload.samples[{i}]")
        _reject_unknown(sample, _SAMPLE_FIELDS, f"payload.samples[{i}]")
        _parse_sample(sample, i, device["id"], records, pairs, unhandled)

    workouts = _list(envelope["workouts"], "payload.workouts")
    parsed_workouts: list[dict] = []
    workout_dates: set[str] = set()
    for i, workout in enumerate(workouts):
        if not isinstance(workout, dict):
            raise PayloadError(f"payload.workouts[{i}] must be a JSON object")
        _reject_unknown(workout, _WORKOUT_FIELDS, f"payload.workouts[{i}]")
        _parse_workout(workout, i, parsed_workouts, workout_dates, unhandled)
    daily_totals_wire = _list(envelope.get("daily_totals", []),
                              "payload.daily_totals")
    daily_totals: list[dict] = []
    daily_total_dates: set[str] = set()
    for i, total in enumerate(daily_totals_wire):
        if not isinstance(total, dict):
            raise PayloadError(f"payload.daily_totals[{i}] must be a JSON object")
        _reject_unknown(total, _DAILY_TOTAL_FIELDS,
                        f"payload.daily_totals[{i}]")
        _parse_daily_total(total, i, device["id"], daily_totals,
                           daily_total_dates, unhandled)
    return {
        "records": records,
        "workouts": parsed_workouts,
        "workout_dates": workout_dates,
        "daily_totals": daily_totals,
        "daily_total_dates": daily_total_dates,
        "pairs": pairs,
        "unhandled": unhandled,
        "deletions": deletions,
        "anchors": anchors,
        "rejected_anchors": rejected_anchors,
        "anchor_results": anchor_results,
        "batch_id": envelope["batch_id"],
        "batch_sequence": envelope["batch_sequence"],
        "device_id": device["id"],
    }
