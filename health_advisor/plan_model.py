"""Immutable, serializable objects for the rule-based plan projection.

The conversation log is the source of truth.  This module describes the
typed envelope that can be rebuilt from that log; it deliberately does not
decide how a rule is graded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

PLAN_MODEL_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GradingPolicy:
    """Immutable rule values selected for a plan week.

    The class is generic: it says which dials exist and that a week pins them
    explicitly, so a historical projection cannot be rewritten by a later
    change of policy. It says nothing about what the values should be — that
    is the caller's prescription, not this module's.
    """

    version: str
    effective_date: date | None
    over_volume_factor: float
    under_volume_factor: float
    jog_credit_factor: float
    block_credit_factor: float
    qualify_min_minutes: int
    qualify_min_avg_hr: int
    qualify_min_kcal: int
    non_endurance_types: frozenset[str]


# A NEUTRAL DEFAULT, NOT A TRAINING PRESCRIPTION.
#
# `Week.grading_policy` is required, so the typed envelope needs some policy to
# exist before a caller supplies one. These are conservative, generic values;
# they are not tuned to any athlete, goal or training block, and nothing here
# should be read as advice. A caller that grades real work is expected to
# construct its own `GradingPolicy` and pin it on the `Week` — that pinning is
# the whole point of the field, and this default only keeps the round-trip
# (`to_dict`/`from_dict`) and the type check honest in its absence.
#
# `effective_date` is None: a default is not in force from any particular day.
DEFAULT_GRADING_POLICY = GradingPolicy(
    version="default",
    effective_date=None,
    over_volume_factor=1.25,
    under_volume_factor=0.5,
    jog_credit_factor=0.5,
    block_credit_factor=0.5,
    qualify_min_minutes=20,
    qualify_min_avg_hr=100,
    qualify_min_kcal=100,
    non_endurance_types=frozenset({
        "traditional_strength_training", "functional_strength_training",
        "core_training", "high_intensity_interval_training", "yoga", "pilates",
        "barre", "tai_chi", "mind_and_body", "flexibility", "cooldown",
        "preparation_and_recovery", "gymnastics", "wrestling", "boxing",
        "kickboxing", "martial_arts", "fencing", "fitness_gaming",
    }),
)


def _freeze(value: Any) -> Any:
    """Recursively make JSON-shaped values immutable."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _iso(value: date) -> str:
    return value.isoformat()


def _date(value: str) -> date:
    return date.fromisoformat(value)


class RuleKind(str, Enum):
    SESSION = "session"
    CONSTRAINT = "constraint"
    CONDITIONAL = "conditional"
    ANCHOR = "anchor"
    STANCE = "stance"


RULE_KINDS = frozenset(kind.value for kind in RuleKind)


@dataclass(frozen=True, slots=True)
class Scope:
    """A composable selector over the dimensions a rule may govern."""

    week: str | int | None = None
    days: tuple[str | date, ...] = ()
    session: str | None = None
    modality: str | None = None

    def __post_init__(self) -> None:
        if self.week is not None and not isinstance(self.week, (str, int)):
            raise TypeError("scope.week must be a string, integer, or None")
        days = tuple(_iso(day) if isinstance(day, date) else day for day in self.days)
        for day in days:
            if not isinstance(day, (str, date)):
                raise TypeError("scope days must be ISO strings or date objects")
            if isinstance(day, str):
                _date(day)
        object.__setattr__(self, "days", days)
        for name in ("session", "modality"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"scope.{name} must be a non-empty string")

    @classmethod
    def for_week(cls, week: str | int) -> "Scope":
        return cls(week=week)

    @classmethod
    def for_day(cls, day: str | date) -> "Scope":
        return cls(days=(day,))

    @classmethod
    def for_days(cls, days: tuple[str | date, ...] | list[str | date]) -> "Scope":
        return cls(days=tuple(days))

    @classmethod
    def for_session(cls, session: str) -> "Scope":
        return cls(session=session)

    @classmethod
    def for_modality(cls, modality: str) -> "Scope":
        return cls(modality=modality)

    # Readable aliases for callers that prefer the selector vocabulary itself.
    week_scope = for_week
    day = for_day
    days_scope = for_days
    session_scope = for_session
    modality_scope = for_modality

    def compose(self, other: "Scope") -> "Scope":
        """Return the intersection of two selectors, rejecting conflicts.

        Days intersect too: an empty day set constrains nothing, and two
        non-empty sets that share no day are a conflict. Composition may
        narrow a rule's scope, never widen it.
        """
        if not isinstance(other, Scope):
            raise TypeError("a scope can only compose with another scope")
        if self.week is not None and other.week is not None and self.week != other.week:
            raise ValueError("conflicting week selectors")
        if self.session is not None and other.session is not None and self.session != other.session:
            raise ValueError("conflicting session selectors")
        if self.modality is not None and other.modality is not None and self.modality != other.modality:
            raise ValueError("conflicting modality selectors")
        if self.days and other.days:
            keep = set(other.days)
            days = tuple(day for day in self.days if day in keep)
            if not days:
                raise ValueError("conflicting day selectors")
        else:
            days = self.days or other.days
        return Scope(
            week=self.week if self.week is not None else other.week,
            days=days,
            session=self.session if self.session is not None else other.session,
            modality=self.modality if self.modality is not None else other.modality,
        )

    __and__ = compose

    def to_dict(self) -> dict[str, Any]:
        return {
            "week": self.week,
            "days": [_iso(day) if isinstance(day, date) else day for day in self.days],
            "session": self.session,
            "modality": self.modality,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scope":
        return cls(
            week=data.get("week"),
            days=tuple(data.get("days", ())),
            session=data.get("session"),
            modality=data.get("modality"),
        )


@dataclass(frozen=True, slots=True)
class EffectiveInterval:
    """A date interval with independently explicit boundary semantics.

    ``end=None`` is an open interval — a rule in force with no decided end,
    which is the normal state of a live standing rule. A far-future sentinel
    date must never stand in for it: that is a value meaning "not set", the
    ``sleep_awakenings`` bug class.
    """

    start: date
    end: date | None = None
    include_start: bool = True
    include_end: bool = True

    def __post_init__(self) -> None:
        if self.end is not None:
            if self.end < self.start:
                raise ValueError("effective interval ends before it starts")
            if self.start == self.end and not (self.include_start and self.include_end):
                raise ValueError("a same-day interval must include both boundaries")

    def contains(self, day: date) -> bool:
        if not (day > self.start or (self.include_start and day == self.start)):
            return False
        if self.end is None:
            return True
        return day < self.end or (self.include_end and day == self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _iso(self.start),
            "end": _iso(self.end) if self.end is not None else None,
            "include_start": self.include_start,
            "include_end": self.include_end,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EffectiveInterval":
        return cls(
            start=_date(data["start"]),
            end=_date(data["end"]) if data.get("end") else None,
            include_start=bool(data.get("include_start", True)),
            include_end=bool(data.get("include_end", True)),
        )


# A shorter name is useful at call sites while the long name makes the
# boundary semantics clear in type annotations.
Interval = EffectiveInterval


@dataclass(frozen=True, slots=True)
class Stated:
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze(self.value))


@dataclass(frozen=True, slots=True)
class Withdrawal:
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("withdrawal reason must be a non-empty string")


Statement = Stated | Withdrawal
StatedValue = Stated


def statement_to_dict(statement: Statement | None) -> dict[str, Any] | None:
    """Serialize presence explicitly; ``None`` means absence, not withdrawal."""
    if statement is None:
        return None
    if isinstance(statement, Stated):
        return {"type": "stated", "value": _thaw(statement.value)}
    if isinstance(statement, Withdrawal):
        return {"type": "withdrawal", "reason": statement.reason}
    raise TypeError("statement must be Stated, Withdrawal, or None")


def statement_from_dict(data: Mapping[str, Any] | None) -> Statement | None:
    if data is None:
        return None
    kind = data.get("type")
    if kind == "stated":
        return Stated(data["value"])
    if kind == "withdrawal":
        return Withdrawal(data["reason"])
    raise ValueError(f"unknown statement type: {kind!r}")


def stated(value: Any) -> Stated:
    return Stated(value)


def withdrawal(reason: str) -> Withdrawal:
    return Withdrawal(reason)


@dataclass(frozen=True, slots=True)
class ConversationTurnProvenance:
    conversation_turn_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_turn_id, str) or not self.conversation_turn_id:
            raise ValueError("conversation_turn_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ParsedProvenance:
    file: str
    line: int

    def __post_init__(self) -> None:
        if not isinstance(self.file, str) or not self.file:
            raise ValueError("parsed provenance file must be a non-empty string")
        if not isinstance(self.line, int) or isinstance(self.line, bool) or self.line < 1:
            raise ValueError("parsed provenance line must be a positive integer")


ConversationTurn = ConversationTurnProvenance
Parsed = ParsedProvenance
Provenance = ConversationTurnProvenance | ParsedProvenance


def provenance_to_dict(provenance: Provenance) -> dict[str, Any]:
    if isinstance(provenance, ConversationTurnProvenance):
        return {"type": "conversation_turn", "conversation_turn_id": provenance.conversation_turn_id}
    if isinstance(provenance, ParsedProvenance):
        return {"type": "parsed", "file": provenance.file, "line": provenance.line}
    raise TypeError("invalid provenance")


def provenance_from_dict(data: Mapping[str, Any]) -> Provenance:
    kind = data.get("type")
    if kind == "conversation_turn":
        return ConversationTurnProvenance(data["conversation_turn_id"])
    if kind == "parsed":
        return ParsedProvenance(data["file"], int(data["line"]))
    raise ValueError(f"unknown provenance type: {kind!r}")


def validate_enforced_from(enforced_from: date | None, acceptance_date: date | None) -> None:
    """Enforce the no-retroactive-enforcement invariant when dated evidence exists."""
    if enforced_from is not None and acceptance_date is not None and enforced_from < acceptance_date:
        raise ValueError("enforced_from cannot precede the rule acceptance date")


@dataclass(frozen=True, slots=True)
class Rule:
    kind: str | RuleKind
    scope: Scope
    stated: EffectiveInterval
    statement: Statement
    provenance: Provenance
    enforced_from: date | None = None
    acceptance_date: date | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind.value if isinstance(self.kind, RuleKind) else self.kind
        if kind not in RULE_KINDS:
            raise ValueError(f"rule kind must be one of {sorted(RULE_KINDS)}")
        if not isinstance(self.scope, Scope):
            raise TypeError("rule scope must be a Scope")
        if not isinstance(self.stated, EffectiveInterval):
            raise TypeError("rule stated interval must be an EffectiveInterval")
        if self.statement is None:
            raise ValueError("absence is represented by no Rule, not statement=None")
        if not isinstance(self.statement, (Stated, Withdrawal)):
            raise TypeError("rule statement must be Stated or Withdrawal")
        if not isinstance(self.provenance, (ConversationTurnProvenance, ParsedProvenance)):
            raise TypeError("rule provenance must be conversation-turn or parsed")
        if self.enforced_from is not None and not isinstance(self.enforced_from, date):
            raise TypeError("enforced_from must be a date or None")
        if self.acceptance_date is not None and not isinstance(self.acceptance_date, date):
            raise TypeError("acceptance_date must be a date or None")
        validate_enforced_from(self.enforced_from, self.acceptance_date)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "payload", MappingProxyType({key: _freeze(value) for key, value in self.payload.items()}))

    @property
    def effective(self) -> EffectiveInterval:
        return self.stated

    @property
    def effective_interval(self) -> EffectiveInterval:
        return self.stated

    def validate(self, acceptance_date: date | None = None) -> None:
        """Validate this rule against the acceptance date supplied by its log turn."""
        validate_enforced_from(self.enforced_from, acceptance_date or self.acceptance_date)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "scope": self.scope.to_dict(),
            "stated": self.stated.to_dict(),
            "statement": statement_to_dict(self.statement),
            "provenance": provenance_to_dict(self.provenance),
            "enforced_from": _iso(self.enforced_from) if self.enforced_from else None,
            "acceptance_date": _iso(self.acceptance_date) if self.acceptance_date else None,
            "payload": _thaw(self.payload),
        }
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Rule":
        statement = statement_from_dict(data.get("statement"))
        if statement is None:
            raise ValueError("serialized rule has no statement; absence has no rule row")
        return cls(
            kind=data["kind"],
            scope=Scope.from_dict(data["scope"]),
            stated=EffectiveInterval.from_dict(data["stated"]),
            statement=statement,
            provenance=provenance_from_dict(data["provenance"]),
            enforced_from=_date(data["enforced_from"]) if data.get("enforced_from") else None,
            acceptance_date=_date(data["acceptance_date"]) if data.get("acceptance_date") else None,
            payload=data.get("payload", {}),
        )


@dataclass(frozen=True, slots=True)
class Week:
    week_start: date
    rules: tuple[Rule, ...]
    provenance: Provenance
    grading_policy: GradingPolicy = DEFAULT_GRADING_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.week_start, date):
            raise TypeError("week_start must be a date")
        object.__setattr__(self, "rules", tuple(self.rules))
        if not all(isinstance(rule, Rule) for rule in self.rules):
            raise TypeError("week rules must contain Rule values")
        if not isinstance(self.provenance, (ConversationTurnProvenance, ParsedProvenance)):
            raise TypeError("week provenance must be conversation-turn or parsed")
        if not isinstance(self.grading_policy, GradingPolicy):
            raise TypeError("week grading_policy must be a GradingPolicy")

    @property
    def start(self) -> date:
        return self.week_start

    @property
    def policy(self) -> GradingPolicy:
        return self.grading_policy

    def to_dict(self) -> dict[str, Any]:
        policy = self.grading_policy
        return {
            "schema_version": PLAN_MODEL_SCHEMA_VERSION,
            "week_start": _iso(self.week_start),
            "rules": [rule.to_dict() for rule in self.rules],
            "provenance": provenance_to_dict(self.provenance),
            "grading_policy": {
                "version": policy.version,
                "effective_date": (_iso(policy.effective_date)
                                   if policy.effective_date else None),
                "over_volume_factor": policy.over_volume_factor,
                "under_volume_factor": policy.under_volume_factor,
                "jog_credit_factor": policy.jog_credit_factor,
                "block_credit_factor": policy.block_credit_factor,
                "qualify_min_minutes": policy.qualify_min_minutes,
                "qualify_min_avg_hr": policy.qualify_min_avg_hr,
                "qualify_min_kcal": policy.qualify_min_kcal,
                "non_endurance_types": sorted(policy.non_endurance_types),
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Week":
        if data.get("schema_version") != PLAN_MODEL_SCHEMA_VERSION:
            raise ValueError(f"unsupported plan model schema version: {data.get('schema_version')!r}")
        policy_data = data["grading_policy"]
        policy = GradingPolicy(
            version=policy_data["version"],
            effective_date=(_date(policy_data["effective_date"])
                            if policy_data.get("effective_date") else None),
            over_volume_factor=policy_data["over_volume_factor"],
            under_volume_factor=policy_data["under_volume_factor"],
            jog_credit_factor=policy_data["jog_credit_factor"],
            block_credit_factor=policy_data["block_credit_factor"],
            qualify_min_minutes=policy_data["qualify_min_minutes"],
            qualify_min_avg_hr=policy_data["qualify_min_avg_hr"],
            qualify_min_kcal=policy_data["qualify_min_kcal"],
            non_endurance_types=frozenset(policy_data["non_endurance_types"]),
        )
        return cls(
            week_start=_date(data["week_start"]),
            rules=tuple(Rule.from_dict(item) for item in data["rules"]),
            provenance=provenance_from_dict(data["provenance"]),
            grading_policy=policy,
        )

    @classmethod
    def from_json(cls, payload: str) -> "Week":
        return cls.from_dict(json.loads(payload))


def serialize_absence() -> None:
    """The serial form of silence: no rule payload and therefore no row."""
    return statement_to_dict(None)


__all__ = [
    "PLAN_MODEL_SCHEMA_VERSION", "RULE_KINDS", "RuleKind", "Scope",
    "EffectiveInterval", "Interval", "Stated", "StatedValue", "Withdrawal",
    "Statement", "statement_to_dict", "statement_from_dict", "stated", "withdrawal",
    "ConversationTurnProvenance", "ConversationTurn", "ParsedProvenance", "Parsed",
    "Provenance", "provenance_to_dict", "provenance_from_dict",
    "validate_enforced_from", "Rule", "Week", "serialize_absence",
]
