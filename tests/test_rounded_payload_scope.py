"""A payload figure rounded to its own precision is the SQL figure, not a new scope (#568).

The evening brief publishes a day's active energy as a whole number. On a
low-energy day the rounding gap is a larger share of the value than `_close`'s
0.5%, so the payload/SQL cross-check refused a correctly quoted figure on every
sedentary day: 23 against a stored 22.6 is 1.8%. Truncation stays refused.
"""
import pytest

from health_advisor import deepdive_verify as DV

DAY = "2026-08-31"


@pytest.fixture
def energy_day(conn):
    def put(total):
        conn.execute("DELETE FROM daily_metrics WHERE metric = 'active_energy'")
        conn.execute(
            "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max,"
            " last, unit, source_kind) VALUES ('active_energy', ?, 1, ?, ?, ?,"
            " ?, ?, 'kcal', 'records')", (DAY, total, total, total, total, total))
        conn.commit()
        return conn
    return put


def _verdict(conn, published):
    payload = [{"metric": "active_energy", "period": DAY, "field": "sum",
                "value": published}]
    claim = {"metric": "active_energy", "period": DAY, "field": "sum",
             "value": published}
    return DV.verify_number(conn, claim, as_of=DAY, payload=payload)


@pytest.mark.parametrize("stored,published", [
    (22.6, 23), (50.4, 50), (76.9, 77), (112.8, 113), (9.46, 9.5)])
def test_rounded_publication_binds(energy_day, stored, published):
    verdict = _verdict(energy_day(stored), published)
    assert verdict["ok"] is True, verdict
    assert verdict["sql_actual"] == pytest.approx(stored)


@pytest.mark.parametrize("stored,published", [
    (76.9, 76), (22.6, 22), (22.6, 24), (9.46, 9.4)])
def test_truncated_or_wrong_publication_is_refused(energy_day, stored, published):
    verdict = _verdict(energy_day(stored), published)
    assert verdict["ok"] is False
    assert verdict["reason"] == "payload/SQL scope disagreement"


def test_payload_decimals():
    assert DV._payload_decimals(77) == 0
    assert DV._payload_decimals(77.0) == 0
    assert DV._payload_decimals(9.5) == 1
    assert DV._payload_decimals(7.29) == 2
    assert DV._payload_decimals(True) is None
    assert DV._payload_decimals("77") is None
