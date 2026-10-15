"""Single definitions for the helpers that used to be copied across modules (#101).

Each collapsed helper gets two kinds of check: the retired private names are
gone (so a second copy cannot quietly reappear), and the surviving helper gives
the same answer the old copies gave over a table of inputs.
"""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import chat, db, facts


# --- table and column probes -------------------------------------------------

def _old_has_table(conn, name):
    """The body every retired copy shared (db, chat, facts, the two scripts)."""
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,)).fetchone() is not None


@pytest.mark.parametrize("row_factory", [None, sqlite3.Row])
def test_has_table_matches_old_copy(row_factory):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = row_factory
    conn.execute("CREATE TABLE present (a INTEGER)")
    conn.execute("CREATE VIEW only_a_view AS SELECT 1 AS a")
    for name in ["present", "absent", "only_a_view", "PRESENT", ""]:
        assert db.has_table(conn, name) == _old_has_table(conn, name), name
    assert db.has_table(conn, "present") is True
    assert db.has_table(conn, "absent") is False
    assert db.has_table(conn, "only_a_view") is False


@pytest.mark.parametrize("row_factory", [None, sqlite3.Row])
def test_has_column_reads_pragma_by_position(row_factory):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = row_factory
    conn.execute("CREATE TABLE daily_metrics (metric TEXT, source_kind TEXT)")
    assert db.has_column(conn, "daily_metrics", "source_kind") is True
    assert db.has_column(conn, "daily_metrics", "metric") is True
    assert db.has_column(conn, "daily_metrics", "missing") is False
    # A table that does not exist has no columns, so the probe answers False.
    assert db.has_column(conn, "no_such_table", "source_kind") is False


def test_retired_table_probes_have_no_definition():
    import scripts.build_source_provenance as provenance
    import scripts.verify_daily_metrics as verify
    assert not hasattr(db, "_has_table")
    assert not hasattr(chat, "_has_table")
    assert not hasattr(facts, "_has_table")
    assert not hasattr(provenance, "_has_table")
    assert not hasattr(verify, "_has_table")
    assert not hasattr(verify, "_has_column")


# --- UTC clock ---------------------------------------------------------------

def test_utc_now_matches_old_copies(monkeypatch):
    import time
    from health_advisor import dates

    real_gmtime = time.gmtime
    for epoch in (0, 1_700_000_000, 1_735_689_599, 951_782_400):
        fixed = real_gmtime(epoch)
        monkeypatch.setattr(time, "gmtime",
                            lambda *a, _f=fixed: _f if not a else real_gmtime(*a))
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        assert dates.utc_now() == old
    monkeypatch.setattr(time, "gmtime", real_gmtime)


def test_retired_clock_helpers_have_no_definition():
    from health_advisor import device_auth, device_secrets, enrol
    assert not hasattr(device_auth, "_utc_now")
    assert not hasattr(device_secrets, "_utc_now")
    assert not hasattr(enrol, "_utc_now_iso")


# --- base64url ---------------------------------------------------------------

@pytest.mark.parametrize("data", [b"", b"\x00", b"\xff" * 3, bytes(range(256)),
                                  b"\xfb\xff\xbf" * 7])
def test_b64url_encode_matches_old_copies(data):
    import base64
    from health_advisor import wire
    old_device = base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
    old_push = base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")
    assert wire.b64url_encode(data) == old_device == old_push
    assert wire.b64url_decode(old_device) == data


@pytest.mark.parametrize("bad", ["a+b", "a/b", "a=", "ab c", None, 7])
def test_b64url_decode_rejects_as_before(bad):
    from health_advisor import wire
    with pytest.raises(ValueError, match="not base64url"):
        wire.b64url_decode(bad)


def test_retired_b64url_names_have_no_definition():
    from health_advisor import device_auth, enrol, push
    assert not hasattr(push, "_b64url")
    for mod in (device_auth, enrol):
        assert not hasattr(mod, "_b64url")
        assert not hasattr(mod, "_b64url_decode")


# --- strict date parser ------------------------------------------------------

@pytest.mark.parametrize("value", ["2026-10-15", "2024-02-29", "2026-1-5", "2026-13-01",
                                   "not-a-date", "", None, 20261015])
def test_parse_date_matches_both_old_copies(value):
    from datetime import date
    from health_advisor import dates

    def old(v, name):
        try:
            return date.fromisoformat(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be YYYY-MM-DD") from None

    for name in ("local_date", "start", "as_of"):
        try:
            want = old(value, name)
        except ValueError as exc:
            with pytest.raises(ValueError) as got:
                dates.parse_date(value, name)
            assert str(got.value) == str(exc)
        else:
            assert dates.parse_date(value, name) == want
    with pytest.raises(ValueError, match="^local_date must be YYYY-MM-DD$"):
        dates.parse_date("nope")


def test_retired_date_parsers_have_no_definition():
    from health_advisor import observations, subjective
    assert not hasattr(observations, "_date")
    assert not hasattr(subjective, "_as_date")


# --- vault-zone today --------------------------------------------------------

@pytest.mark.parametrize("zone", [None, "Pacific/Auckland", "America/Los_Angeles"])
def test_local_today_matches_old_copies(monkeypatch, zone):
    import sqlite3
    from datetime import date, datetime
    from zoneinfo import ZoneInfo
    from health_advisor import vault

    monkeypatch.setattr(vault, "local_timezone", lambda conn: zone)
    conn = sqlite3.connect(":memory:")
    got = vault.local_today(conn)
    # Both sides read the wall clock, so allow the one midnight-crossing case.
    now = datetime.now(ZoneInfo(zone)).date() if zone else date.today()
    assert got in (now, datetime.now(ZoneInfo(zone)).date() if zone else date.today())


def test_retired_today_copies_have_no_definition():
    from health_advisor import analysis, chat
    assert not hasattr(chat, "_today")
    assert not hasattr(analysis, "_today")


# --- Monday week start -------------------------------------------------------

def test_week_start_agrees_across_every_caller_for_two_years():
    from datetime import date, timedelta
    from health_advisor import analysis, calendar_window, dates, mcp_server, metrics

    start = date(2025, 1, 1)
    days = [start + timedelta(days=i) for i in range(730)]
    for day in days:
        monday = day - timedelta(days=day.weekday())
        assert dates.week_start(day) == monday
        assert analysis._week_start(day.isoformat()) == monday.isoformat()
        assert calendar_window._week_start(day.isoformat()) == monday.isoformat()
    # The mcp impact grid lists Mondays, from the Monday of the range start.
    keys = mcp_server._impact_period_keys(days[0].isoformat(), days[-1].isoformat(), "week")
    assert keys == sorted({d - timedelta(days=d.weekday()) for d in days})
    # metrics.downsample_weekly groups by the same Monday.
    out = metrics.downsample_weekly([d.isoformat() for d in days[:21]], [1.0] * 21, "sum")
    assert [row["date"] for row in out] == sorted(
        {(d - timedelta(days=d.weekday())).isoformat() for d in days[:21]})
