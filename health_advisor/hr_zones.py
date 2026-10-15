"""Heart-rate zones from a person's own characteristics (#113).

Pure functions, no I/O. The birth date is a characteristic the person states
once; nothing here logs it, and nothing here reads it from a vault. The vault
owns storage (`vault.date_of_birth` / `vault.set_date_of_birth`) and calls
`check_date_of_birth` before it writes.

Two formulas, each named for its source:

* Tanaka et al. (2001) for an age-predicted maximum heart rate:
  208 - 0.7 x age. Its population is wide; it is an estimate, not a test.
* Karvonen for a zone ceiling by heart-rate reserve:
  rest + fraction x (max - rest).

`ZONE2_TOP_FRACTION` is the published default. A caller that has a measured
maximum should pass it in place of the age-predicted one.
"""
from __future__ import annotations

import math
from datetime import date

ZONE2_TOP_FRACTION = 0.70

MIN_AGE = 10
MAX_AGE = 100

_TANAKA_INTERCEPT = 208.0
_TANAKA_SLOPE = 0.7


def parse_date_of_birth(text: object) -> date:
    """A strict ``YYYY-MM-DD`` calendar date, or ValueError.

    The message names the expected form and never echoes the value, so a
    refused birth date cannot reach a log line or a response by way of an
    exception.
    """
    refusal = "date of birth must be a calendar date in YYYY-MM-DD form"
    if not isinstance(text, str):
        raise ValueError(refusal)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise ValueError(refusal) from None
    if parsed.isoformat() != text:
        # fromisoformat also accepts 19900615 and week dates; the stored and
        # wire form is the one spelling.
        raise ValueError(refusal)
    return parsed


def age_on(date_of_birth: date, on: date) -> int:
    """Whole years of age on the day ``on``.

    A birthday in a leap year is reached on 1 March in a common year.
    """
    if on < date_of_birth:
        raise ValueError("a date of birth cannot fall after the day it is measured on")
    years = on.year - date_of_birth.year
    if (on.month, on.day) < (date_of_birth.month, date_of_birth.day):
        years -= 1
    return years


def check_date_of_birth(date_of_birth: date, on: date) -> None:
    """Refuse a birth date after ``on`` or giving an age outside 10 to 100."""
    if date_of_birth > on:
        raise ValueError("date of birth is in the future")
    age = age_on(date_of_birth, on)
    if not MIN_AGE <= age <= MAX_AGE:
        raise ValueError(
            f"date of birth gives an age outside {MIN_AGE} to {MAX_AGE}")


def max_hr_from_age(age: int) -> float:
    """Tanaka age-predicted maximum heart rate in bpm, for ages 10 to 100."""
    if (isinstance(age, bool) or not isinstance(age, int)
            or not MIN_AGE <= age <= MAX_AGE):
        raise ValueError(f"age must be a whole number from {MIN_AGE} to {MAX_AGE}")
    return _TANAKA_INTERCEPT - _TANAKA_SLOPE * age


def zone2_top(rest_bpm: float, max_bpm: float,
              fraction: float = ZONE2_TOP_FRACTION) -> int:
    """Karvonen zone ceiling in whole bpm, rounded to the nearest bpm."""
    if not (math.isfinite(rest_bpm) and math.isfinite(max_bpm)):
        raise ValueError("resting and maximum heart rate must be finite")
    if not max_bpm > rest_bpm:
        raise ValueError("maximum heart rate must exceed resting heart rate")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be above 0 and at most 1")
    return round(rest_bpm + fraction * (max_bpm - rest_bpm))
