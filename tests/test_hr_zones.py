"""Pure heart-rate zone and birth-date functions (#113).

Everything here is synthetic; the engine repository is public.
"""
from __future__ import annotations

from datetime import date

import pytest

from health_advisor import hr_zones


# --------------------------------------------------------------------------- #
# age_on
# --------------------------------------------------------------------------- #
def test_age_is_whole_years_and_turns_on_the_birthday():
    dob = date(1990, 6, 15)
    assert hr_zones.age_on(dob, date(2020, 6, 14)) == 29
    assert hr_zones.age_on(dob, date(2020, 6, 15)) == 30


def test_leap_day_birthday_turns_on_march_first_in_a_common_year():
    dob = date(2000, 2, 29)
    assert hr_zones.age_on(dob, date(2021, 2, 28)) == 20
    assert hr_zones.age_on(dob, date(2021, 3, 1)) == 21


def test_age_on_a_date_before_birth_is_refused():
    with pytest.raises(ValueError):
        hr_zones.age_on(date(2000, 1, 2), date(2000, 1, 1))


# --------------------------------------------------------------------------- #
# max_hr_from_age (Tanaka)
# --------------------------------------------------------------------------- #
def test_tanaka_max_hr_for_a_23_year_old():
    assert hr_zones.max_hr_from_age(23) == pytest.approx(191.9)


@pytest.mark.parametrize("age", [10, 100])
def test_tanaka_accepts_the_bounds(age):
    assert hr_zones.max_hr_from_age(age) == pytest.approx(208 - 0.7 * age)


@pytest.mark.parametrize("age", [9, 101, 0, -3])
def test_tanaka_refuses_ages_outside_10_to_100(age):
    with pytest.raises(ValueError):
        hr_zones.max_hr_from_age(age)


@pytest.mark.parametrize("age", [True, 23.0, "23", None])
def test_tanaka_refuses_non_integer_ages(age):
    with pytest.raises(ValueError):
        hr_zones.max_hr_from_age(age)


# --------------------------------------------------------------------------- #
# zone2_top (Karvonen)
# --------------------------------------------------------------------------- #
def test_zone2_top_for_the_published_example():
    assert hr_zones.zone2_top(65, 201) == 160


def test_zone2_top_rounds_to_the_nearest_bpm_not_down():
    # 60 + 0.70 * 51 = 95.7: round gives 96, truncation would give 95.
    assert hr_zones.zone2_top(60, 111) == 96


def test_zone2_top_default_fraction_is_the_published_seventy_percent():
    assert hr_zones.ZONE2_TOP_FRACTION == 0.70
    assert hr_zones.zone2_top(60, 111) == hr_zones.zone2_top(60, 111, 0.70)


def test_zone2_top_returns_an_int():
    assert isinstance(hr_zones.zone2_top(65, 201), int)


def test_zone2_top_fraction_one_is_the_max():
    assert hr_zones.zone2_top(60, 180, 1.0) == 180


@pytest.mark.parametrize("rest, max_", [(60, 60), (70, 60)])
def test_zone2_top_refuses_max_not_above_rest(rest, max_):
    with pytest.raises(ValueError):
        hr_zones.zone2_top(rest, max_)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01, float("nan")])
def test_zone2_top_refuses_fraction_outside_zero_to_one(fraction):
    with pytest.raises(ValueError):
        hr_zones.zone2_top(60, 180, fraction)


@pytest.mark.parametrize("rest, max_", [(float("nan"), 180), (60, float("inf"))])
def test_zone2_top_refuses_non_finite_inputs(rest, max_):
    with pytest.raises(ValueError):
        hr_zones.zone2_top(rest, max_)


# --------------------------------------------------------------------------- #
# birth date: strict parse and the range check
# --------------------------------------------------------------------------- #
def test_parse_accepts_only_the_iso_calendar_spelling():
    assert hr_zones.parse_date_of_birth("1990-06-15") == date(1990, 6, 15)


@pytest.mark.parametrize("text", [
    "19900615", "1990-6-5", "1990-02-30", "", "1990/06/15",
    "1990-W24-5", "tomorrow", 19900615, None,
])
def test_parse_refuses_anything_else_with_a_value_free_message(text):
    with pytest.raises(ValueError) as info:
        hr_zones.parse_date_of_birth(text)
    if isinstance(text, str) and text:
        assert text not in str(info.value)


ON = date(2030, 1, 1)


def test_check_accepts_age_ten_and_age_one_hundred():
    hr_zones.check_date_of_birth(date(2020, 1, 1), ON)   # age 10
    hr_zones.check_date_of_birth(date(1929, 12, 31), ON)  # age 100


def test_check_refuses_age_nine():
    with pytest.raises(ValueError):
        hr_zones.check_date_of_birth(date(2020, 1, 2), ON)


def test_check_refuses_age_one_hundred_and_one():
    with pytest.raises(ValueError):
        hr_zones.check_date_of_birth(date(1928, 12, 31), ON)


def test_check_refuses_a_birth_date_after_the_measured_day():
    with pytest.raises(ValueError, match="future"):
        hr_zones.check_date_of_birth(date(2030, 1, 2), ON)
