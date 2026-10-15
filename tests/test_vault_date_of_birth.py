"""The vault stores one birth date in vault_meta, validated on the way in (#113).

Everything here is synthetic; the engine repository is public.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from health_advisor import vault as vaultmod


def _years_ago(years: int, extra_days: int = 0) -> date:
    return date.today() - timedelta(days=365 * years + extra_days)


def test_an_undeclared_vault_has_no_birth_date(conn):
    assert vaultmod.date_of_birth(conn) is None


def test_a_birth_date_round_trips_through_vault_meta(conn):
    vaultmod.set_date_of_birth(conn, date(1990, 6, 15))
    assert vaultmod.date_of_birth(conn) == date(1990, 6, 15)
    row = conn.execute(
        "SELECT value FROM vault_meta WHERE key = 'date_of_birth'").fetchone()
    assert row[0] == "1990-06-15"


def test_a_later_birth_date_replaces_the_stored_one(conn):
    vaultmod.set_date_of_birth(conn, date(1990, 6, 15))
    vaultmod.set_date_of_birth(conn, date(1988, 3, 2))
    assert vaultmod.date_of_birth(conn) == date(1988, 3, 2)


def test_clearing_the_birth_date_removes_it(conn):
    vaultmod.set_date_of_birth(conn, date(1990, 6, 15))
    vaultmod.set_date_of_birth(conn, None)
    assert vaultmod.date_of_birth(conn) is None


def test_a_future_birth_date_is_refused_and_not_stored(conn):
    with pytest.raises(ValueError):
        vaultmod.set_date_of_birth(conn, date.today() + timedelta(days=2))
    assert vaultmod.date_of_birth(conn) is None


def test_an_age_under_ten_is_refused_and_not_stored(conn):
    with pytest.raises(ValueError):
        vaultmod.set_date_of_birth(conn, _years_ago(9))
    assert vaultmod.date_of_birth(conn) is None


def test_an_age_over_one_hundred_is_refused_and_not_stored(conn):
    with pytest.raises(ValueError):
        vaultmod.set_date_of_birth(conn, _years_ago(101, extra_days=30))
    assert vaultmod.date_of_birth(conn) is None


def test_a_refused_value_leaves_the_previous_one_in_place(conn):
    vaultmod.set_date_of_birth(conn, date(1990, 6, 15))
    with pytest.raises(ValueError):
        vaultmod.set_date_of_birth(conn, _years_ago(9))
    assert vaultmod.date_of_birth(conn) == date(1990, 6, 15)


@pytest.mark.parametrize("value", ["1990-06-15", datetime(1990, 6, 15)])
def test_a_string_or_datetime_is_not_a_date_of_birth(conn, value):
    with pytest.raises(ValueError):
        vaultmod.set_date_of_birth(conn, value)
    assert vaultmod.date_of_birth(conn) is None
