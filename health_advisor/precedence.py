"""Precedence for plan governors.

The plan's controls are an ordered set, not a hard/diagnostic flag.  A
diagnostic target remains in the set so it can explain an observation, while
the resolver chooses the highest-ranked applicable governor.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Iterable


class PrecedenceRank(IntEnum):
    """The five plan ranks, in descending authority."""

    SAFETY_SIGNALS = 5
    PERCEIVED_EFFORT = 4
    NUMERIC_TARGETS = 3
    SCHEDULE = 2
    DIAGNOSTIC = 1


# Keep this explicit and declaration-ordered: it is the public precedence
# contract, not an incidental property of the enum's integer values.
RANKS_HIGHEST_FIRST = (
    PrecedenceRank.SAFETY_SIGNALS,
    PrecedenceRank.PERCEIVED_EFFORT,
    PrecedenceRank.NUMERIC_TARGETS,
    PrecedenceRank.SCHEDULE,
    PrecedenceRank.DIAGNOSTIC,
)


@dataclass(frozen=True, slots=True)
class Governor:
    """A plan control carrying its precedence rank.

    ``applicable`` is the result of matching the governor to the current
    session and observations.  It is intentionally separate from ``rank``:
    a target can be authored and retained for explanation without applying
    on a particular day.  There is no ``binding`` field.
    """

    rank: PrecedenceRank
    kind: str
    statement: str
    source: str
    applicable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.rank, PrecedenceRank):
            try:
                object.__setattr__(self, "rank", PrecedenceRank(self.rank))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown precedence rank: {self.rank!r}") from exc
        for name in ("kind", "statement", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.applicable, bool):
            raise TypeError("applicable must be a bool")

    @property
    def is_diagnostic(self) -> bool:
        return self.rank is PrecedenceRank.DIAGNOSTIC

    def demote_to_diagnostic(self) -> "Governor":
        """Retain this governor while making it observation-only."""
        return replace(self, rank=PrecedenceRank.DIAGNOSTIC)


# A target is represented by the same rank-bearing object as every other
# governor.  This compatibility name keeps the domain vocabulary available
# without reviving Target.binding.
Target = Governor


@dataclass(frozen=True, slots=True)
class Resolution:
    """The resolver's answer and the ordered context used to explain it."""

    selected: Governor | None
    ordered: tuple[Governor, ...]

    @property
    def lower_ranked(self) -> tuple[Governor, ...]:
        """Governors below the selected item, retained for explanation."""
        if self.selected is None:
            return self.ordered
        selected_index = self.ordered.index(self.selected)
        return self.ordered[selected_index + 1 :]

    @property
    def applicable(self) -> tuple[Governor, ...]:
        return tuple(governor for governor in self.ordered if governor.applicable)

    @property
    def governing(self) -> Governor | None:
        """The item allowed to decide an action; diagnostics never decide."""
        if self.selected is None or self.selected.is_diagnostic:
            return None
        return self.selected


def resolve(governors: Iterable[Governor]) -> Resolution:
    """Select the highest-ranked applicable governor.

    Sorting is stable, so multiple controls at one rank remain in authored
    order.  The complete ordered list is returned with the selection; callers
    can use ``lower_ranked`` to explain what was outranked or demoted.
    """
    ordered = tuple(sorted(governors, key=lambda governor: governor.rank, reverse=True))
    selected = next((governor for governor in ordered if governor.applicable), None)
    return Resolution(selected=selected, ordered=ordered)


__all__ = [
    "Governor",
    "PrecedenceRank",
    "RANKS_HIGHEST_FIRST",
    "Resolution",
    "Target",
    "resolve",
]
