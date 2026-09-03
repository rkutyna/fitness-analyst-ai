"""D19 storage: hk_daily_totals, its immutability triggers, and the revision log.

The design's assertions 1-7a live in this file. **Only assertion 1 and the
storage half of assertion 3 are here yet** — the rest need the ingest path
(`hk_parse` / `receiver`), which is step 3 of the build order. Step 1 is schema
and storage with no ingest path, so nothing in this file goes through HTTP.

Assertion 1 is deliberately asserted against the TRIGGER rather than against a
Python guard: #220 Done-when 2 is a property of the storage engine, so the next
script somebody writes cannot bypass it.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from health_advisor import db as dbmod
from health_advisor import receiver
from health_advisor import vault as vault_mod


def _total(metric="step_count", local_date="2026-08-25", value=10173.0,
           unit="count", state="provisional", queried_at="2026-08-26T09:00:00",
           interval="day", device_id="dev-1"):
    return {"metric": metric, "local_date": local_date, "value": value,
            "unit": unit, "interval": interval, "state": state,
            "device_id": device_id, "queried_at": queried_at}


_DEVICE = {"id": "dev-1", "name": "iPhone", "model": "iPhone17,1"}


def _daily_payload(*rows, batch_id="daily-batch", sequence=1):
    return {
        "protocol_version": 1,
        "device": _DEVICE,
        "app_version": "1.0",
        "batch_id": batch_id,
        "batch_sequence": sequence,
        "sent_at": "2026-08-29T09:00:00Z",
        "anchors": [], "samples": [], "deletions": [], "workouts": [],
        "daily_totals": list(rows),
    }


def _wire_total(**overrides):
    row = {
        "type_identifier": "HKQuantityTypeIdentifierStepCount",
        "local_date": "2026-08-25",
        "value": 10173.0,
        "unit": "count",
        "interval": "day",
        "state": "provisional",
        "queried_at": "2026-08-26T09:00:00-04:00",
    }
    row.update(overrides)
    return row


def _client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "hk-secret")
    return TestClient(receiver.create_app(vault))


def _post(client, *rows, batch_id="daily-batch", sequence=1):
    return client.post(
        "/v1/ingest",
        json=_daily_payload(*rows, batch_id=batch_id, sequence=sequence),
        headers={"x-health-secret": "hk-secret"},
    )


def _insights_digest(conn):
    """Digest the complete stored insight table, including its timestamps."""
    rows = [tuple(row) for row in conn.execute(
        "SELECT id, date, text, tags, created_at FROM insights ORDER BY id")]
    return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()


# --- assertion 1: a settled total is immutable, in the engine ---------------

def test_settled_row_rejects_a_direct_update(conn):
    dbmod.insert_daily_totals(conn, [_total(state="settled")], batch_id="b1")

    with pytest.raises(sqlite3.IntegrityError) as excinfo:
        conn.execute(
            "UPDATE hk_daily_totals SET value = 99.0 "
            "WHERE metric = 'step_count' AND local_date = '2026-08-25'")

    assert "a settled daily total is immutable" in str(excinfo.value)
    assert conn.execute(
        "SELECT value FROM hk_daily_totals").fetchone()["value"] == 10173.0


def test_settled_row_rejects_a_direct_delete(conn):
    dbmod.insert_daily_totals(conn, [_total(state="settled")], batch_id="b1")

    with pytest.raises(sqlite3.IntegrityError) as excinfo:
        conn.execute(
            "DELETE FROM hk_daily_totals "
            "WHERE metric = 'step_count' AND local_date = '2026-08-25'")

    assert "settled totals are not deletable" in str(excinfo.value)
    assert conn.execute("SELECT COUNT(*) c FROM hk_daily_totals").fetchone()["c"] == 1


def test_a_provisional_row_is_still_writable(conn):
    """The trigger is scoped to `state = 'settled'`, not to the table.

    Without this the immutability guard would also forbid the ordinary
    provisional overwrite the state machine is built on.
    """
    dbmod.insert_daily_totals(conn, [_total(value=9000.0)], batch_id="b1")
    dbmod.insert_daily_totals(
        conn, [_total(value=10173.0, queried_at="2026-08-27T09:00:00")],
        batch_id="b2")

    row = conn.execute("SELECT * FROM hk_daily_totals").fetchone()
    assert row["value"] == 10173.0
    assert row["state"] == "provisional"
    assert row["settled_at"] is None


# --- assertion 3, storage half: the revision log -----------------------------

def test_every_write_appends_a_revision_row_with_the_server_derived_lag(conn):
    """`lag_days` is derived here from `queried_at`, never sent as a count.

    A phone with a wrong clock then produces a visibly wrong lag rather than a
    plausible one. The first write carries NULL `from_value`/`from_state`.
    """
    dbmod.insert_daily_totals(conn, [_total(value=9000.0)], batch_id="b1")
    dbmod.insert_daily_totals(
        conn, [_total(value=10173.0, queried_at="2026-08-28T09:00:00")],
        batch_id="b2")

    revs = conn.execute(
        "SELECT * FROM hk_daily_total_revisions ORDER BY id").fetchall()
    assert len(revs) == 2
    assert (revs[0]["from_value"], revs[0]["to_value"]) == (None, 9000.0)
    assert (revs[0]["from_state"], revs[0]["to_state"]) == (None, "provisional")
    assert revs[0]["lag_days"] == 1          # queried 08-26 for day 08-25
    assert revs[0]["batch_id"] == "b1"
    assert (revs[1]["from_value"], revs[1]["to_value"]) == (9000.0, 10173.0)
    assert revs[1]["lag_days"] == 3          # queried 08-28 for day 08-25


def test_a_no_change_write_is_still_recorded(conn):
    """N is read off this table as "the lag by which values stop changing", so a
    write that changes nothing at lag k is the evidence, not noise. Duplicate
    suppression is the commit_log preflight's job, not this function's."""
    dbmod.insert_daily_totals(conn, [_total(value=10173.0)], batch_id="b1")
    dbmod.insert_daily_totals(
        conn, [_total(value=10173.0, queried_at="2026-08-27T09:00:00")],
        batch_id="b2")

    revs = conn.execute(
        "SELECT from_value, to_value, lag_days FROM hk_daily_total_revisions "
        "ORDER BY id").fetchall()
    assert [tuple(r) for r in revs] == [(None, 10173.0, 1), (10173.0, 10173.0, 2)]


def test_the_settle_write_stamps_settled_at_and_records_the_transition(conn):
    dbmod.insert_daily_totals(conn, [_total(value=10173.0)], batch_id="b1")
    dbmod.insert_daily_totals(
        conn, [_total(value=10173.0, state="settled",
                      queried_at="2026-08-28T09:00:00")], batch_id="b2")

    row = conn.execute("SELECT * FROM hk_daily_totals").fetchone()
    assert row["state"] == "settled"
    assert row["settled_at"] is not None
    assert row["first_seen_at"] is not None
    rev = conn.execute(
        "SELECT from_state, to_state FROM hk_daily_total_revisions "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert (rev["from_state"], rev["to_state"]) == ("provisional", "settled")


def test_expected_from_is_stamped_once_and_never_moves(conn):
    """Each metric gets its own check-6 epoch, and each epoch is write-once.

    INSERT OR IGNORE, deliberately: a later pull for a later day must not move
    that metric's epoch forward, while a metric whose first arrival is later
    gets its own independent start date.
    """
    dbmod.insert_daily_totals(conn, [_total(local_date="2026-08-25")],
                              batch_id="b1")
    dbmod.insert_daily_totals(
        conn, [_total(metric="flights_climbed", local_date="2026-08-26",
                      queried_at="2026-08-27T09:00:00")],
        batch_id="b2")
    dbmod.insert_daily_totals(
        conn, [_total(local_date="2026-08-28", queried_at="2026-08-29T09:00:00")],
        batch_id="b3")
    dbmod.insert_daily_totals(
        conn, [_total(metric="flights_climbed", local_date="2026-08-29",
                      queried_at="2026-08-30T09:00:00")],
        batch_id="b4")

    values = dict(conn.execute(
        "SELECT key, value FROM vault_meta "
        "WHERE key LIKE 'daily_totals_expected_from:%'"))
    assert values == {
        "daily_totals_expected_from:step_count": "2026-08-25",
        "daily_totals_expected_from:flights_climbed": "2026-08-26",
    }


def test_state_is_constrained_to_the_two_legal_values(conn):
    with pytest.raises(sqlite3.IntegrityError):
        dbmod.insert_daily_totals(conn, [_total(state="maybe")], batch_id="b1")


# --- assertions 2-7a: the server wire and guards ---------------------------

def test_settled_daily_total_post_is_rejected_with_evidence(
        conn, vault, vault_path, monkeypatch, capsys):
    """2. A fresh batch cannot rewrite a settled day or log its value."""
    dbmod.insert_daily_totals(conn, [_total(state="settled")], batch_id="seed")
    conn.commit()

    with _client(vault, monkeypatch) as client:
        response = _post(client, _wire_total(state="provisional", value=9999),
                         batch_id="settled-refusal")

    assert response.status_code == 409
    assert response.json()["detail"].startswith("daily total already settled")
    trace = capsys.readouterr().err
    assert "ingest-trace reject-409-settled" in trace
    assert "9999" not in trace
    assert conn.execute(
        "SELECT value, state FROM hk_daily_totals WHERE metric = 'step_count' "
        "AND local_date = '2026-08-25'").fetchone()[:] == (10173.0, "settled")
    assert conn.execute("SELECT COUNT(*) FROM hk_daily_total_revisions").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM commit_log WHERE key = 'healthkit:dev-1:settled-refusal'"
    ).fetchone()[0] == 0
    fresh = dbmod.connect(vault_path, read_only=True)
    try:
        evidence = fresh.execute(
            "SELECT source, kind, rows_seen, rows_added, detail FROM ingest_log"
        ).fetchone()
        assert tuple(evidence[:4]) == ("receiver", "reject", 0, 0)
        assert "settled_guard" in evidence["detail"]
        assert "settled-refusal" in evidence["detail"]
        assert "9999" not in evidence["detail"]
    finally:
        fresh.close()


def test_provisional_daily_total_post_accepts_update(conn, vault, monkeypatch):
    """3. A provisional update changes the fact and appends its revision."""
    dbmod.insert_daily_totals(conn, [_total(value=9000.0)], batch_id="seed")
    conn.commit()

    with _client(vault, monkeypatch) as client:
        response = _post(
            client,
            _wire_total(value=10173.0,
                        queried_at="2026-08-28T09:00:00-04:00"),
            batch_id="provisional-update")

    assert response.status_code == 200, response.text
    row = conn.execute(
        "SELECT value, state FROM hk_daily_totals WHERE metric = 'step_count' "
        "AND local_date = '2026-08-25'").fetchone()
    assert tuple(row) == (10173.0, "provisional")
    rev = conn.execute(
        "SELECT from_value, to_value, lag_days FROM hk_daily_total_revisions "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert tuple(rev) == (9000.0, 10173.0, 3)


def test_settle_transition_is_one_way(conn, vault, monkeypatch):
    """4. Settling stamps the row, after which a fresh pull is refused."""
    with _client(vault, monkeypatch) as client:
        first = _post(client, _wire_total(), batch_id="settle-provisional")
        settled = _post(
            client,
            _wire_total(state="settled", value=10173.0,
                        queried_at="2026-08-28T09:00:00-04:00"),
            batch_id="settle-final", sequence=2)
        refused = _post(
            client,
            _wire_total(state="settled", value=10173.0,
                        queried_at="2026-08-29T09:00:00-04:00"),
            batch_id="settle-after", sequence=3)

    assert first.status_code == 200
    assert settled.status_code == 200
    assert refused.status_code == 409
    row = conn.execute(
        "SELECT state, settled_at FROM hk_daily_totals WHERE metric = 'step_count' "
        "AND local_date = '2026-08-25'").fetchone()
    assert row["state"] == "settled"
    assert row["settled_at"] is not None


def test_settling_a_changed_total_leaves_prior_insights_byte_identical(conn):
    """A settle changes the total, never the already-written insight table.

    Cost, measured 2026-09-03 on the demo vault: apply_consolidated_totals
    over 2,190 hk_daily_totals rows took 3.7 ms in total (0.0017 ms per day);
    one scoped (metric, day) pair took 1.2 ms. Journal mode DELETE.
    """
    day = "2026-08-25"
    dbmod.write_insight(conn, day, "Synthetic daily summary.", tags="daily")
    before = _insights_digest(conn)

    dbmod.insert_daily_totals(
        conn, [_total(value=9000.0, queried_at="2026-08-26T09:00:00")],
        batch_id="provisional")
    dbmod.apply_consolidated_totals(conn, pairs=[("step_count", day)])
    dbmod.insert_daily_totals(
        conn, [_total(value=10173.0, state="settled",
                      queried_at="2026-08-28T09:00:00")],
        batch_id="settled")
    dbmod.apply_consolidated_totals(conn, pairs=[("step_count", day)])
    conn.commit()

    assert _insights_digest(conn) == before
    assert conn.execute(
        "SELECT state, value FROM hk_daily_totals").fetchone()[:] == \
        ("settled", 10173.0)
    revisions = conn.execute(
        "SELECT from_value, to_value, from_state, to_state, lag_days "
        "FROM hk_daily_total_revisions ORDER BY id").fetchall()
    assert [tuple(row) for row in revisions] == [
        (None, 9000.0, None, "provisional", 1),
        (9000.0, 10173.0, "provisional", "settled", 3),
    ]

    with pytest.raises(sqlite3.IntegrityError):
        dbmod.insert_daily_totals(
            conn, [_total(value=11111.0, state="settled",
                          queried_at="2026-08-29T09:00:00")],
            batch_id="after-settle")
    assert _insights_digest(conn) == before


def test_applied_settle_batch_retry_is_already_applied(conn, vault, monkeypatch):
    """5. The same settle key is idempotent before the settle guard."""
    payload = _wire_total(state="settled")
    with _client(vault, monkeypatch) as client:
        applied = _post(client, payload, batch_id="settle-retry")
        retry = _post(client, payload, batch_id="settle-retry")

    assert applied.status_code == 200
    assert retry.status_code == 200
    assert retry.json()["applied"] is False
    assert retry.json()["reason"] == "already_applied"


def test_daily_total_watermark_guard_wins(conn, vault, monkeypatch):
    """6. A total at/below D14's watermark is refused before any write."""
    vault_mod.set_history_imported_through(conn, "2026-08-21")
    conn.commit()
    with _client(vault, monkeypatch) as client:
        response = _post(
            client,
            _wire_total(local_date="2026-08-21"),
            batch_id="total-before-watermark")

    detail = response.json()["detail"]
    assert response.status_code == 409
    assert detail.startswith("history imported through 2026-08-21")
    assert "daily total" in detail
    assert conn.execute("SELECT COUNT(*) FROM hk_daily_totals").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM hk_daily_total_revisions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM commit_log").fetchone()[0] == 0


def test_daily_total_409_details_are_distinguishable_and_watermark_parseable(
        conn, vault, monkeypatch):
    """7. Both prefixes are stable; the client-compatible date slice is exact."""
    dbmod.insert_daily_totals(conn, [_total(state="settled")], batch_id="seed")
    vault_mod.set_history_imported_through(conn, "2026-08-21")
    conn.commit()
    with _client(vault, monkeypatch) as client:
        settled = _post(client, _wire_total(), batch_id="different-settled")
        watermark = _post(
            client, _wire_total(local_date="2026-08-21"),
            batch_id="different-watermark")

    settled_detail = settled.json()["detail"]
    watermark_detail = watermark.json()["detail"]
    assert settled_detail.startswith("daily total already settled")
    prefix = "history imported through "
    watermark = "2026-08-21"
    assert watermark_detail[:len(prefix) + len(watermark) + 1] == \
        f"{prefix}{watermark};"
    assert watermark_detail[len(prefix):len(prefix) + 10] == watermark
    assert settled_detail.split(" ", 1)[0] != watermark_detail.split(" ", 1)[0]


def test_mixed_daily_total_batch_is_refused_atomically(conn, vault, monkeypatch):
    """7a. One settled row refuses every other row in the same transaction."""
    dbmod.insert_daily_totals(
        conn, [_total(local_date="2026-08-26", state="settled")], batch_id="seed")
    conn.commit()
    with _client(vault, monkeypatch) as client:
        response = _post(
            client,
            _wire_total(local_date="2026-08-25", value=1111),
            _wire_total(local_date="2026-08-26", value=2222, state="provisional"),
            batch_id="mixed-settled")

    assert response.status_code == 409
    assert conn.execute(
        "SELECT COUNT(*) FROM hk_daily_totals WHERE local_date = '2026-08-25'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM hk_daily_total_revisions WHERE local_date = '2026-08-25'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM commit_log WHERE key = 'healthkit:dev-1:mixed-settled'"
    ).fetchone()[0] == 0
