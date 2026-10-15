"""set_local_timezone stores only canonical IANA zone names (#97)."""
import pytest

from health_advisor import vault as vaultmod

# Names ZoneInfo resolves on a case-insensitive host but no tz database lists
# in this spelling, plus the slash-less, tree-prefixed and padded forms.
REFUSED = [
    "europe/london",
    "EUROPE/LONDON",
    "America/new_york",
    "Factory",
    "posix/Europe/London",
    "right/UTC",
    "Europe/London ",
    "../x",
    "",
]

ACCEPTED = [
    "UTC",
    "GMT",
    "Europe/London",
    "America/New_York",
    "Pacific/Honolulu",
    "Europe/Berlin",
    "Etc/GMT+5",
]


@pytest.mark.parametrize("zone", REFUSED)
def test_non_canonical_zone_name_is_refused_and_not_stored(conn, zone):
    with pytest.raises(ValueError, match="unknown IANA time zone"):
        vaultmod.set_local_timezone(conn, zone)
    assert vaultmod.local_timezone(conn) is None


@pytest.mark.parametrize("zone", ACCEPTED)
def test_canonical_zone_name_is_stored(conn, zone):
    vaultmod.set_local_timezone(conn, zone)
    assert vaultmod.local_timezone(conn) == zone


def test_clearing_the_zone_still_works(conn):
    vaultmod.set_local_timezone(conn, "Europe/London")
    vaultmod.set_local_timezone(conn, None)
    assert vaultmod.local_timezone(conn) is None
