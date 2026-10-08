"""The configurable enrolment token lifetime (consumer #586, task T2).

An invite that travels to a person needs up to 24 hours where a QR shown in
person needed 15 minutes. ``HA_ENROL_MAX_TTL_SECONDS`` raises the cap; unset
keeps 900; nothing exceeds ``TTL_CEILING_SECONDS``; an invalid value refuses
at startup (the startup half lives in ``test_enrol_route.py``).

Mutations, each applied by hand and observed red before this file was
finalised:

- removing the ``<= TTL_CEILING_SECONDS`` test in ``max_ttl_seconds`` (so
  86401 is accepted) turns ``test_an_invalid_cap_is_refused_naming_the_variable``
  and the startup refusal tests red;
- returning ``MAX_TTL_SECONDS`` instead of raising on a bad value turns the
  same refusal tests red;
- making a long-lived token outlive its TTL (``now > record.expires_at`` in
  ``_effective``, or an expiry computed from a fixed 900 added back on) turns
  ``test_a_day_long_token_expires_exactly_at_its_ttl`` red.
"""
from __future__ import annotations

import json
import time

import pytest

from health_advisor import enrol

VAR = "HA_ENROL_MAX_TTL_SECONDS"
DAY = 86400


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "state" / "enrol-tokens.json"
    path.parent.mkdir(parents=True)
    return enrol.TokenStore(path)


@pytest.fixture(autouse=True)
def _unset(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)


def _granted(store, requested, now=1_000_000.0):
    record, _ = store.mint(ttl_seconds=requested, now=now)
    return record.expires_at - now


def test_unset_keeps_the_default_clamp(store):
    assert _granted(store, DAY) == 900
    assert enrol.max_ttl_seconds() == enrol.MAX_TTL_SECONDS == 900


def test_configured_ceiling_grants_a_full_day(store, monkeypatch):
    monkeypatch.setenv(VAR, str(DAY))
    assert _granted(store, DAY) == DAY


def test_configured_cap_clamps_a_larger_request(store, monkeypatch):
    monkeypatch.setenv(VAR, "3600")
    assert _granted(store, DAY) == 3600
    assert _granted(store, 600) == 600  # a shorter request is untouched


@pytest.mark.parametrize("bad", ["86401", "0", "-1", "abc", "", "900.5", " 3600",
                                 "1e3", "+900", "٣٦٠٠"])
def test_an_invalid_cap_is_refused_naming_the_variable(bad, store, monkeypatch):
    monkeypatch.setenv(VAR, bad)
    with pytest.raises(RuntimeError, match=VAR):
        enrol.max_ttl_seconds()
    with pytest.raises(RuntimeError, match=VAR):
        store.mint(ttl_seconds=900)  # never silently defaulted at mint either


def test_a_day_long_token_expires_exactly_at_its_ttl(store, monkeypatch):
    monkeypatch.setenv(VAR, str(DAY))
    start = 5_000_000.0
    record, _ = store.mint(ttl_seconds=DAY, now=start)
    assert record.expires_at == start + DAY

    still = store.get_live(record.id, now=start + DAY - 1)
    assert still.state == "live"
    assert still.x25519_private is not None

    # The first instant treated as expired is expires_at itself (>=).
    with pytest.raises(enrol.EnrolError) as caught:
        store.get_live(record.id, now=start + DAY)
    assert caught.value.code == "enrol_token_expired"
    assert caught.value.status == 401

    with open(store.path, encoding="utf-8") as handle:
        stored = next(t for t in json.load(handle)["tokens"] if t["id"] == record.id)
    assert stored["state"] == "expired"
    assert stored["x25519_private"] is None


def test_a_day_long_token_cannot_be_burned_after_its_ttl(store, monkeypatch):
    monkeypatch.setenv(VAR, str(DAY))
    start = 5_000_000.0
    record, _ = store.mint(ttl_seconds=DAY, now=start)
    with pytest.raises(enrol.EnrolError) as caught:
        store.burn(record.id, kid="kid-x", now=start + DAY)
    assert caught.value.code == "enrol_token_expired"


def test_a_day_long_token_is_still_single_use(store, monkeypatch):
    monkeypatch.setenv(VAR, str(DAY))
    start = 5_000_000.0
    record, _ = store.mint(ttl_seconds=DAY, now=start)
    store.burn(record.id, kid="kid-1", now=start + DAY - 1)
    with pytest.raises(enrol.EnrolError) as caught:
        store.burn(record.id, kid="kid-2", now=start + DAY - 1)
    assert caught.value.code == "enrol_token_used"
    assert caught.value.status == 409


def test_the_cli_clamps_an_over_limit_request_silently_and_refuses_a_bad_cap(
        tmp_path, monkeypatch, capsys):
    path = tmp_path / "tokens.json"
    assert enrol.main(["--store", str(path), "mint", "--ttl", str(DAY)]) == 0
    first = json.loads(capsys.readouterr().out)
    record = enrol.TokenStore(path).get(first["id"])
    assert 890 <= record.expires_at - time.time() <= 900

    monkeypatch.setenv(VAR, "3600")
    assert enrol.main(["--store", str(path), "mint", "--ttl", str(DAY)]) == 0
    second = json.loads(capsys.readouterr().out)
    record = enrol.TokenStore(path).get(second["id"])
    assert 3590 <= record.expires_at - time.time() <= 3600

    monkeypatch.setenv(VAR, "86401")
    assert enrol.main(["--store", str(path), "mint"]) == 2
    captured = capsys.readouterr()
    assert VAR in captured.err and captured.out == ""
