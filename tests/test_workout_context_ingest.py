"""Workout context on the wire and in the vault (consumer #590, engine half).

A workout may state whether it was done indoors (``is_indoor``: true, false or
null) and which connected fitness machine recorded it (``fitness_machine``: a
short kind or null). Absence is not a fact: a missing or null value is stored
as NULL and never read as "outdoors" or "no machine".

Everything here is synthetic; the engine repository is public.
"""
from __future__ import annotations

import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, hk_parse, receiver, vault as vault_mod
from health_advisor import normalize as nz


DEVICE = {"id": "synthetic-device", "name": "synthetic-phone",
          "model": "synthetic-model"}
SOURCE = "Synthetic Source"
REVISION = {"source_name": SOURCE, "bundle_id": "synthetic.bundle"}
START = "2030-01-05T15:00:00Z"
END = "2030-01-05T15:30:00Z"
MACHINE_PREFIX = "com.apple.health.fitnessmachinemodel."


def _workout(*, uuid="synthetic-workout", start=START, end=END, **extra):
    row = {
        "hk_uuid": uuid,
        "workout_activity_type": "HKWorkoutActivityTypeRunning",
        "start": start,
        "end": end,
        "duration_min": 30.0,
        "source_revision": REVISION,
    }
    row.update(extra)
    return row


def _payload(*, batch_id="synthetic-batch", workouts=None, **sections):
    payload = {
        "protocol_version": 1,
        "device": DEVICE,
        "app_version": "synthetic-version",
        "batch_id": batch_id,
        "batch_sequence": 1,
        "sent_at": "2030-01-05T18:00:00Z",
        "anchors": [],
        "samples": [],
        "deletions": [],
        "workouts": workouts or [],
    }
    payload.update(sections)
    return payload


def _entry(*, uuid="synthetic-workout", start=START, end=END, source=SOURCE,
           is_indoor=True, machine="treadmill"):
    return {"hk_uuid": uuid, "start": start, "end": end, "source_name": source,
            "is_indoor": is_indoor, "fitness_machine": machine}


def _client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "")
    return TestClient(receiver.create_app(vault))


def _rows(vault):
    conn = vault.read_only()
    try:
        return conn.execute(
            "SELECT hk_uuid, is_indoor, fitness_machine, dedupe_key "
            "FROM workouts ORDER BY id").fetchall()
    finally:
        conn.close()


def _row(vault):
    rows = _rows(vault)
    assert len(rows) == 1
    return rows[0]


def _parse(*workouts):
    return hk_parse.parse_payload(_payload(workouts=list(workouts)))["workouts"]


# --------------------------------------------------------------------------- #
# Parse: is_indoor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sent, stored", [(True, 1), (False, 0), (None, None)])
def test_is_indoor_true_false_null_parse_to_1_0_none(sent, stored):
    (row,) = _parse(_workout(is_indoor=sent))
    assert row["is_indoor"] == stored
    assert (row["is_indoor"] is None) == (stored is None)


def test_absent_is_indoor_and_machine_are_not_stated():
    """The phone's Codable omits a nil key, as it does for distance_mi."""
    (row,) = _parse(_workout())
    assert row["is_indoor"] is None and row["fitness_machine"] is None
    (no_distance,) = _parse(_workout())
    assert no_distance["distance_mi"] is None  # the precedent being mirrored


@pytest.mark.parametrize("bad", [1, 0, 1.0, "true", "false", "yes", [], {}, [True]])
def test_wrong_typed_is_indoor_is_refused(bad):
    # The message must be the type refusal, not "unknown field(s)": a server
    # that does not know the key refuses it too, for a different reason.
    with pytest.raises(hk_parse.PayloadError,
                       match=r"is_indoor must be true, false or null"):
        _parse(_workout(is_indoor=bad))


# --------------------------------------------------------------------------- #
# Parse: fitness_machine
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sent, stored", [
    ("treadmill", "treadmill"),
    (MACHINE_PREFIX + "treadmill", "treadmill"),
    ("Treadmill", "treadmill"),
    ("indoorbike", "indoor_bike"),
    ("indoor-bike", "indoor_bike"),
    ("Indoor Bike", "indoor_bike"),
    ("elliptical", "elliptical"),
    (MACHINE_PREFIX + "stairstepper", "stair_stepper"),
    ("stair_stepper", "stair_stepper"),
    ("rower", "rower"),
])
def test_known_machine_kinds_are_canonical(sent, stored):
    (row,) = _parse(_workout(fitness_machine=sent))
    assert row["fitness_machine"] == stored
    assert stored in nz.FITNESS_MACHINE_KINDS


@pytest.mark.parametrize("sent, stored", [
    (MACHINE_PREFIX + "skierg", "skierg"),
    ("Cross Trainer", "cross trainer"),
    (MACHINE_PREFIX + "Some-New.Kind_2", "some-new.kind_2"),
])
def test_unknown_machine_kind_is_kept_lower_cased_as_sent(sent, stored):
    (row,) = _parse(_workout(fitness_machine=sent))
    assert row["fitness_machine"] == stored
    assert stored not in nz.FITNESS_MACHINE_KINDS


@pytest.mark.parametrize("bad", [
    5, True, False, 1.5, ["treadmill"], {"kind": "treadmill"},
    "", "   ", MACHINE_PREFIX, "x" * 65, "tread\nmill", "tread;mill",
    "tread<mill>", "-treadmill",
])
def test_wrong_typed_or_unusable_machine_is_refused(bad):
    with pytest.raises(hk_parse.PayloadError,
                       match=r"fitness_machine must be a machine kind"):
        _parse(_workout(fitness_machine=bad))


def test_machine_of_exactly_the_maximum_length_is_accepted():
    kind = "k" * nz.FITNESS_MACHINE_MAX_LENGTH
    (row,) = _parse(_workout(fitness_machine=kind))
    assert row["fitness_machine"] == kind


def test_a_manufacturer_or_device_field_is_refused_not_stored():
    """Only the kind is accepted: no manufacturer name, no device identifier."""
    (accepted,) = _parse(_workout(fitness_machine="treadmill"))
    assert accepted["fitness_machine"] == "treadmill"
    for extra in ({"fitness_machine_manufacturer": "Synthetic Maker"},
                  {"fitness_machine_device": {"model": "m"}}):
        with pytest.raises(hk_parse.PayloadError, match="unknown field"):
            _parse(_workout(fitness_machine="treadmill", **extra))
    assert "manufacturer" not in " ".join(db.WORKOUT_COLS)


def test_the_parsed_row_still_carries_exactly_the_workout_columns():
    (row,) = _parse(_workout(is_indoor=True, fitness_machine="treadmill"))
    assert set(row) == set(db.WORKOUT_COLS)


# --------------------------------------------------------------------------- #
# Migration: an old vault gains the columns with NULLs
# --------------------------------------------------------------------------- #
def _legacy_row(**extra):
    row = {
        "workout_type": "running", "start_utc": "2030-01-05T15:00:00+00:00",
        "end_utc": "2030-01-05T15:30:00+00:00", "local_date": "2030-01-05",
        "duration_min": 30.0, "energy_kcal": None, "distance_mi": 3.0,
        "unit_distance": "mi", "source": SOURCE, "dedupe_key": "legacy",
        "hk_uuid": "legacy-uuid",
    }
    row.update(extra)
    return row


def _drop_context_columns(path):
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE workouts DROP COLUMN is_indoor")
    conn.execute("ALTER TABLE workouts DROP COLUMN fitness_machine")
    conn.commit()
    names = {r[1] for r in conn.execute("PRAGMA table_info(workouts)")}
    conn.close()
    assert not names & {"is_indoor", "fitness_machine"}


def test_an_old_vault_gains_the_columns_with_nulls_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.init_db(conn)
    db.insert_workouts(conn, [_legacy_row()])
    conn.commit()
    conn.close()
    _drop_context_columns(path)

    conn = db.connect(path)
    assert not db.has_workout_context_columns(conn)
    db.init_db(conn)
    assert db.has_workout_context_columns(conn)
    row = conn.execute(
        "SELECT hk_uuid, distance_mi, is_indoor, fitness_machine "
        "FROM workouts").fetchone()
    assert tuple(row) == ("legacy-uuid", 3.0, None, None)
    # Re-running the migration changes nothing (idempotent).
    db.init_db(conn)
    assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 1
    conn.close()


def test_the_migration_is_not_a_schema_version_bump(tmp_path):
    """Adding nullable columns is not a bump, as for the elevation columns."""
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.init_db(conn)
    conn.close()
    _drop_context_columns(path)
    conn = db.connect(path)
    before = db.vault_schema_version(conn)
    db.init_db(conn)
    assert db.vault_schema_version(conn) == before == db.VAULT_SCHEMA_VERSION
    conn.close()


def test_a_declared_vault_opens_with_its_declaration_undisturbed(tmp_path):
    path = tmp_path / "declared.db"
    conn = db.connect(path)
    db.init_db(conn)
    vault_mod.declare_vault(conn)
    db.insert_workouts(conn, [_legacy_row()])
    conn.commit()
    through = vault_mod.compacted_through(conn)
    meta = dict(conn.execute("SELECT key, value FROM vault_meta").fetchall())
    conn.close()
    _drop_context_columns(path)

    conn = db.connect(path)
    db.init_db(conn)
    assert vault_mod.is_vault(conn)
    assert vault_mod.compacted_through(conn) == through
    after = dict(conn.execute("SELECT key, value FROM vault_meta").fetchall())
    assert {k: after.get(k) for k in meta} == meta
    assert conn.execute(
        "SELECT is_indoor, fitness_machine FROM workouts").fetchone()[:] == (None, None)
    conn.close()


def test_build_vault_carries_the_context_columns(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "vault.db"
    conn = db.connect(source)
    db.init_db(conn)
    db.insert_workouts(conn, [_legacy_row(is_indoor=1, fitness_machine="treadmill")])
    conn.commit()
    conn.close()
    vault_mod.build_vault(source, target, measure_gzip=False)
    conn = db.connect(target, read_only=True)
    try:
        assert tuple(conn.execute(
            "SELECT is_indoor, fitness_machine FROM workouts").fetchone()) == (
                1, "treadmill")
    finally:
        conn.close()


def test_build_vault_from_a_source_without_the_columns_reads_null(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "vault.db"
    conn = db.connect(source)
    db.init_db(conn)
    db.insert_workouts(conn, [_legacy_row()])
    conn.commit()
    conn.close()
    _drop_context_columns(source)
    vault_mod.build_vault(source, target, measure_gzip=False)
    conn = db.connect(target, read_only=True)
    try:
        assert tuple(conn.execute(
            "SELECT is_indoor, fitness_machine FROM workouts").fetchone()) == (
                None, None)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Ingest and storage
# --------------------------------------------------------------------------- #
def test_an_old_phone_payload_without_the_keys_still_ingests_unchanged(
        vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            workouts=[_workout()]))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["workouts_added"] == 1
    assert not [k for k in body if k.startswith("workout_context")]
    row = _row(vault)
    assert row["is_indoor"] is None and row["fitness_machine"] is None


def test_a_stated_context_is_stored(vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(workouts=[_workout(
            is_indoor=True, fitness_machine=MACHINE_PREFIX + "treadmill")]))
    assert response.status_code == 200, response.text
    row = _row(vault)
    assert row["is_indoor"] == 1 and row["fitness_machine"] == "treadmill"


def test_false_is_stored_as_zero_not_null(vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        client.post("/v1/ingest", json=_payload(
            workouts=[_workout(is_indoor=False)]))
    assert _row(vault)["is_indoor"] == 0


def test_a_wrong_typed_flag_is_refused_and_nothing_is_written(
        vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            workouts=[_workout(is_indoor="true")]))
    assert response.status_code == 400
    assert "is_indoor" in response.text
    assert _rows(vault) == []


def test_null_then_stated_fills_the_hole(vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        first = client.post("/v1/ingest", json=_payload(
            batch_id="b1", workouts=[_workout()]))
        assert first.json()["workouts_added"] == 1
        assert _row(vault)["is_indoor"] is None
        second = client.post("/v1/ingest", json=_payload(
            batch_id="b2", workouts=[_workout(
                is_indoor=True, fitness_machine="treadmill")]))
    assert second.status_code == 200, second.text
    assert second.json()["workouts_added"] == 0
    row = _row(vault)
    assert row["is_indoor"] == 1 and row["fitness_machine"] == "treadmill"


def test_a_later_different_stated_value_replaces_the_stored_one(
        vault, monkeypatch):
    """The ordinary path's rule, as for the elevation columns: the later stated
    value wins. A later upload that states nothing leaves it alone."""
    with _client(vault, monkeypatch) as client:
        client.post("/v1/ingest", json=_payload(batch_id="b1", workouts=[
            _workout(is_indoor=True, fitness_machine="treadmill")]))
        client.post("/v1/ingest", json=_payload(batch_id="b2", workouts=[
            _workout(is_indoor=False, fitness_machine="rower")]))
        assert (_row(vault)["is_indoor"], _row(vault)["fitness_machine"]) == (
            0, "rower")
        client.post("/v1/ingest", json=_payload(batch_id="b3", workouts=[
            _workout(is_indoor=None, fitness_machine=None)]))
        client.post("/v1/ingest", json=_payload(batch_id="b4", workouts=[
            _workout()]))
    row = _row(vault)
    assert (row["is_indoor"], row["fitness_machine"]) == (0, "rower")


def test_a_context_only_change_is_not_swallowed_by_the_upsert_where_clause(
        conn):
    """Everything else on the row is already filled; only the new column
    differs. The hole-filling WHERE clause must still let it through."""
    db.insert_workouts(conn, [_legacy_row(is_indoor=1)])
    db.insert_workouts(conn, [_legacy_row(is_indoor=0)])
    assert conn.execute("SELECT is_indoor FROM workouts").fetchone()[0] == 0
    db.insert_workouts(conn, [_legacy_row(fitness_machine="elliptical")])
    assert conn.execute(
        "SELECT is_indoor, fitness_machine FROM workouts").fetchone()[:] == (
            0, "elliptical")


def test_a_caller_that_omits_the_columns_leaves_null(conn):
    """Backfill and export callers never name the new columns."""
    db.insert_workouts(conn, [_legacy_row()])
    assert conn.execute(
        "SELECT is_indoor, fitness_machine FROM workouts").fetchone()[:] == (
            None, None)


def test_context_is_not_part_of_workout_identity(vault, monkeypatch):
    """The dedupe key stays type|start|end: the same session with and without
    context is one row, and the key is exactly what it was."""
    expected = db.workout_key("running", "2030-01-05T15:00:00+00:00",
                              "2030-01-05T15:30:00+00:00")
    with _client(vault, monkeypatch) as client:
        client.post("/v1/ingest", json=_payload(
            batch_id="b1", workouts=[_workout()]))
        client.post("/v1/ingest", json=_payload(
            batch_id="b2", workouts=[_workout(
                is_indoor=True, fitness_machine="treadmill")]))
    rows = _rows(vault)
    assert len(rows) == 1
    assert rows[0]["dedupe_key"] == expected
    (parsed_plain,) = _parse(_workout())
    (parsed_ctx,) = _parse(_workout(is_indoor=False, fitness_machine="rower"))
    assert parsed_plain["dedupe_key"] == parsed_ctx["dedupe_key"] == expected


# --------------------------------------------------------------------------- #
# The workout_context backfill section
# --------------------------------------------------------------------------- #
def _seed_one(vault, monkeypatch, **extra):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            batch_id="seed", workouts=[_workout(**extra)]))
    assert response.status_code == 200, response.text


def test_backfill_fills_nulls_and_reports_counts(vault, monkeypatch):
    _seed_one(vault, monkeypatch)
    entries = [
        _entry(),
        _entry(uuid="not-stored", start="2030-01-05T20:00:00Z",
               end="2030-01-05T20:30:00Z"),
    ]
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            batch_id="ctx-1", workout_context=entries))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["workout_context_seen"] == 2
    assert body["workout_context_matched"] == 1
    assert body["workout_context_updated"] == 1
    assert body["workout_context_unmatched"] == 1
    row = _row(vault)
    assert row["is_indoor"] == 1 and row["fitness_machine"] == "treadmill"
    # The elevation counters are absent: that section was not sent.
    assert "workout_elevation_seen" not in body


def test_backfill_never_overwrites_a_stated_value_or_clears_one(
        vault, monkeypatch):
    _seed_one(vault, monkeypatch, is_indoor=True, fitness_machine="treadmill")
    with _client(vault, monkeypatch) as client:
        different = client.post("/v1/ingest", json=_payload(
            batch_id="ctx-1", workout_context=[
                _entry(is_indoor=False, machine="rower")]))
        nulls = client.post("/v1/ingest", json=_payload(
            batch_id="ctx-2", workout_context=[
                _entry(is_indoor=None, machine=None)]))
    assert different.json()["workout_context_updated"] == 0
    assert nulls.json()["workout_context_updated"] == 0
    row = _row(vault)
    assert (row["is_indoor"], row["fitness_machine"]) == (1, "treadmill")


def test_backfill_fills_only_the_missing_column(vault, monkeypatch):
    _seed_one(vault, monkeypatch, is_indoor=True)
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            batch_id="ctx-1", workout_context=[
                _entry(is_indoor=False, machine="treadmill")]))
    assert response.json()["workout_context_updated"] == 1
    row = _row(vault)
    assert (row["is_indoor"], row["fitness_machine"]) == (1, "treadmill")


def test_backfill_matches_a_uuidless_legacy_row_by_overlap(conn):
    db.insert_workouts(conn, [_legacy_row(hk_uuid=None)])
    result = db.attach_workout_context(conn, [{
        "hk_uuid": "phone-uuid", "start_utc": "2030-01-05T15:00:00+00:00",
        "end_utc": "2030-01-05T15:30:00+00:00", "source_name": SOURCE,
        "is_indoor": 1, "fitness_machine": None}])
    assert result == {"seen": 1, "matched": 1, "updated": 1, "unmatched": 0}
    assert conn.execute("SELECT is_indoor FROM workouts").fetchone()[0] == 1


def test_backfill_with_the_section_present_but_empty_is_accepted(
        vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            batch_id="ctx-1", workout_context=[]))
    assert response.status_code == 200
    assert response.json()["workout_context_seen"] == 0


@pytest.mark.parametrize("mutate, needle", [
    (lambda e: e.pop("is_indoor"), "workout_context[0] is missing required"),
    (lambda e: e.pop("fitness_machine"), "workout_context[0] is missing required"),
    (lambda e: e.pop("hk_uuid"), "workout_context[0] is missing required"),
    (lambda e: e.update(is_indoor="yes"), "is_indoor must be true, false or null"),
    (lambda e: e.update(is_indoor=1), "is_indoor must be true, false or null"),
    (lambda e: e.update(fitness_machine=7), "fitness_machine must be a machine kind"),
    (lambda e: e.update(fitness_machine=""), "fitness_machine must be a machine kind"),
    (lambda e: e.update(manufacturer="x"), "workout_context[0] has unknown field"),
    (lambda e: e.update(source_name=""), "source_name must be a non-empty"),
    (lambda e: e.update(end=START), "unparseable or inverted start/end"),
])
def test_a_malformed_backfill_entry_refuses_the_payload(
        vault, monkeypatch, mutate, needle):
    entry = _entry()
    mutate(entry)
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            batch_id="ctx-bad", workout_context=[entry]))
    assert response.status_code == 400
    assert needle in response.text


def test_a_non_list_or_non_object_backfill_section_is_refused():
    for bad, needle in (("x", "must be a JSON array"),
                        ({"a": 1}, "must be a JSON array"),
                        ([1], "workout_context[0] must be a JSON object"),
                        ([None], "workout_context[0] must be a JSON object")):
        with pytest.raises(hk_parse.PayloadError, match=re.escape(needle)):
            hk_parse.parse_payload(_payload(workout_context=bad))


def test_a_replayed_backfill_batch_reports_zero_updates(vault, monkeypatch):
    _seed_one(vault, monkeypatch)
    payload = _payload(batch_id="ctx-1", workout_context=[_entry()])
    with _client(vault, monkeypatch) as client:
        first = client.post("/v1/ingest", json=payload)
        replay = client.post("/v1/ingest", json=payload)
    assert first.json()["workout_context_updated"] == 1
    body = replay.json()
    # As for workout_elevation, an already-applied replay is answered before
    # the sections are looked at: it reports no counters and writes nothing.
    assert body["applied"] is False and body["reason"] == "already_applied"
    assert "workout_context_updated" not in body
    row = _row(vault)
    assert (row["is_indoor"], row["fitness_machine"]) == (1, "treadmill")


# --------------------------------------------------------------------------- #
# Capability advertisement
# --------------------------------------------------------------------------- #
def test_health_advertises_context_support_next_to_elevation(vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["workout_context_supported"] is True
    assert body["workout_context_backfill_generation"] == (
        db.WORKOUT_CONTEXT_BACKFILL_GENERATION) == 1
    # What an older phone reads is untouched.
    assert body["workout_elevation_supported"] is True
    assert body["workout_routes_supported"] is True
    assert body["workout_elevation_backfill_generation"] >= 1
