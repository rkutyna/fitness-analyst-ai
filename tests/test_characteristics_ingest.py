"""The optional `characteristics` envelope section on /v1/ingest (#113).

A stated date of birth is stored in vault_meta. A refused one refuses the whole
batch with 422 before anything is written, and the value never reaches a log
line or the response. An absent section changes nothing.

Everything here is synthetic; the engine repository is public.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from health_advisor import hk_parse, receiver, vault as vault_mod
from tests.conftest import INGEST_HEADERS, INGEST_TEST_SECRET


DEVICE = {"id": "synthetic-device", "name": "synthetic-phone",
          "model": "synthetic-model"}
SOURCE = "Synthetic Source"
REVISION = {"source_name": SOURCE, "bundle_id": "synthetic.bundle"}
DOB_TEXT = "1990-06-15"


def _workout(uuid="synthetic-workout"):
    return {
        "hk_uuid": uuid,
        "workout_activity_type": "HKWorkoutActivityTypeRunning",
        "start": "2030-01-05T15:00:00Z",
        "end": "2030-01-05T15:30:00Z",
        "duration_min": 30.0,
        "source_revision": REVISION,
    }


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


def _client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", INGEST_TEST_SECRET)
    return TestClient(receiver.create_app(vault), headers=INGEST_HEADERS)


def _stored_dob(vault):
    conn = vault.read_only()
    try:
        return vault_mod.date_of_birth(conn)
    finally:
        conn.close()


def _workout_count(vault):
    conn = vault.read_only()
    try:
        return conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0]
    finally:
        conn.close()


def _vault_meta_has_dob(vault):
    conn = vault.read_only()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM vault_meta WHERE key = 'date_of_birth'"
        ).fetchone()[0] == 1
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #
def test_an_absent_section_parses_to_no_characteristics():
    assert hk_parse.parse_payload(_payload())["characteristics"] == {}


def test_a_stated_date_parses_to_a_date():
    parsed = hk_parse.parse_payload(_payload(
        characteristics={"date_of_birth": DOB_TEXT}))
    assert parsed["characteristics"] == {"date_of_birth": date(1990, 6, 15)}


@pytest.mark.parametrize("section", [
    {"date_of_birth": "19900615"},
    {"date_of_birth": "1990-02-30"},
    {"date_of_birth": None},
    {"date_of_birth": 19900615},
    {"date_of_birth": ""},
])
def test_a_bad_date_is_a_characteristics_refusal(section):
    with pytest.raises(hk_parse.CharacteristicsError) as info:
        hk_parse.parse_payload(_payload(characteristics=section))
    assert "1990" not in str(info.value)


def test_a_bad_date_refusal_is_still_a_payload_error():
    # Existing `except PayloadError` callers keep working.
    assert issubclass(hk_parse.CharacteristicsError, hk_parse.PayloadError)


@pytest.mark.parametrize("section", [
    {"sex": "unspecified"},
    {"date_of_birth": DOB_TEXT, "height_cm": 180},
])
def test_an_unknown_characteristic_field_is_refused(section):
    with pytest.raises(hk_parse.CharacteristicsError,
                       match=r"characteristics has unknown field"):
        hk_parse.parse_payload(_payload(characteristics=section))


@pytest.mark.parametrize("section", [[], "1990-06-15", None, 1])
def test_a_non_object_characteristics_section_is_refused(section):
    with pytest.raises(hk_parse.CharacteristicsError,
                       match=r"characteristics must be a JSON object"):
        hk_parse.parse_payload(_payload(characteristics=section))


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
def test_a_stated_date_is_stored_and_the_batch_applies(vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            characteristics={"date_of_birth": DOB_TEXT},
            workouts=[_workout()]))
    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert _stored_dob(vault) == date(1990, 6, 15)
    assert _workout_count(vault) == 1


def test_an_absent_section_stores_nothing_and_applies_as_before(
        vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            workouts=[_workout()]))
    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert _stored_dob(vault) is None
    assert not _vault_meta_has_dob(vault)


def test_a_later_stated_date_replaces_the_stored_one(vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        client.post("/v1/ingest", json=_payload(
            batch_id="c1", characteristics={"date_of_birth": DOB_TEXT}))
        client.post("/v1/ingest", json=_payload(
            batch_id="c2", characteristics={"date_of_birth": "1988-03-02"}))
    assert _stored_dob(vault) == date(1988, 3, 2)


@pytest.mark.parametrize("section", [
    {"date_of_birth": "19900615"},
    {"date_of_birth": "1990-02-30"},
    {"date_of_birth": None},
    {"sex": "unspecified"},
    [],
])
def test_a_malformed_section_is_422_and_the_whole_batch_is_refused(
        vault, monkeypatch, section):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            characteristics=section, workouts=[_workout()]))
    assert response.status_code == 422
    assert "1990" not in response.text
    assert _workout_count(vault) == 0
    assert not _vault_meta_has_dob(vault)


def _future():
    return (date.today() + timedelta(days=2)).isoformat()


def _aged(years, extra_days=0):
    return (date.today() - timedelta(days=365 * years + extra_days)).isoformat()


@pytest.mark.parametrize("text", [_future(), _aged(9), _aged(101, 30)])
def test_a_date_outside_the_vault_bounds_is_422_and_nothing_is_written(
        vault, monkeypatch, text):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=_payload(
            characteristics={"date_of_birth": text}, workouts=[_workout()]))
    assert response.status_code == 422
    assert text not in response.text
    assert _workout_count(vault) == 0
    assert not _vault_meta_has_dob(vault)


def test_a_refused_date_leaves_a_previously_stored_one_in_place(
        vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        client.post("/v1/ingest", json=_payload(
            batch_id="c1", characteristics={"date_of_birth": DOB_TEXT}))
        response = client.post("/v1/ingest", json=_payload(
            batch_id="c2", characteristics={"date_of_birth": _future()}))
    assert response.status_code == 422
    assert _stored_dob(vault) == date(1990, 6, 15)


def test_the_birth_date_never_reaches_stderr_or_the_log(
        vault, monkeypatch, capsys, caplog):
    caplog.set_level("DEBUG")
    with _client(vault, monkeypatch) as client:
        ok = client.post("/v1/ingest", json=_payload(
            batch_id="c1", characteristics={"date_of_birth": DOB_TEXT}))
        refused = client.post("/v1/ingest", json=_payload(
            batch_id="c2", characteristics={"date_of_birth": _future()}))
    assert ok.status_code == 200 and refused.status_code == 422
    captured = capsys.readouterr()
    assert DOB_TEXT not in captured.err + captured.out
    assert DOB_TEXT not in caplog.text
    assert DOB_TEXT not in ok.text and DOB_TEXT not in refused.text


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_advertises_characteristics_support_next_to_context(
        vault, monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["characteristics_supported"] is True
    assert body["workout_context_supported"] is True
    assert DOB_TEXT not in response.text
