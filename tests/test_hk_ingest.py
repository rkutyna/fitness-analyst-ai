"""End-to-end application of the parsed HealthKit delta contract (#27)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from health_advisor import db, receiver, vault as vault_mod


DEVICE = {"id": "watch-a", "name": "iPhone", "model": "iPhone17,1"}
REVISION = {"source_name": "Apple Watch", "bundle_id": "com.apple.Health"}
HEART = "HKQuantityTypeIdentifierHeartRate"
MASS = "HKQuantityTypeIdentifierBodyMass"


def _sample(kind, uuid, type_identifier, start, end, value, unit,
            revision=REVISION):
    return {
        "kind": kind, "hk_uuid": uuid, "type_identifier": type_identifier,
        "start": start, "end": end, "value": value, "unit": unit,
        "source_revision": revision,
    }


def _payload(*samples, batch_id="batch-1", sequence=1, deletions=None,
             anchors=None, workouts=None):
    return {
        "protocol_version": 1, "device": DEVICE, "app_version": "1.0",
        "batch_id": batch_id, "batch_sequence": sequence,
        "sent_at": "2026-08-22T13:04:05Z",
        "anchors": anchors if anchors is not None else [{
            "type_identifier": HEART, "from": None, "to": f"anchor-{sequence}"
        }],
        "samples": list(samples), "deletions": deletions or [],
        "workouts": workouts or [],
    }


def _client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "hk-secret")
    return TestClient(receiver.create_app(vault))


def _counts(path):
    conn = db.connect(path, read_only=True)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("records", "daily_metrics", "hk_sync_state",
                          "hk_deletions", "commit_log")
        }
    finally:
        conn.close()


def test_healthkit_ingest_aggregates_every_series_but_keeps_raw_allowlist(
    vault, vault_path, monkeypatch
):
    conn = vault.connect()
    db.init_db(conn)
    vault_mod.declare_vault(conn)
    conn.commit()
    conn.close()

    body = _sample("quantity", "mass-1", MASS,
                   "2026-08-22T08:00:00-04:00", "2026-08-22T08:00:01-04:00",
                   80.0, "lb")
    heart = _sample("quantity", "hr-1", HEART,
                    "2026-08-22T09:00:00-04:00", "2026-08-22T09:00:01-04:00",
                    145.0, "count/min")
    awkward = _sample("quantity", "unknown-1", "HKQuantityTypeIdentifierFuture",
                      "2026-08-22T10:00:00-04:00", "2026-08-22T10:00:01-04:00",
                      1.0, "count")
    # The awkward real-world shape is intentional: no sample device and only
    # the two required source_revision fields.
    payload = _payload(body, heart, awkward)

    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json=payload,
                               headers={"x-health-secret": "hk-secret"})
        assert response.status_code == 200, response.text
        assert response.json()["unhandled"]

    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM records WHERE metric = 'body_mass'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM records WHERE metric = 'heart_rate'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT avg, last FROM daily_metrics "
            "WHERE metric = 'body_mass' AND date = '2026-08-22'"
        ).fetchone()[:2] == (80.0, 80.0)
        assert conn.execute(
            "SELECT avg, last FROM daily_metrics "
            "WHERE metric = 'heart_rate' AND date = '2026-08-22'"
        ).fetchone()[:2] == (145.0, 145.0)
        state = conn.execute(
            "SELECT anchor_token, last_batch_id FROM hk_sync_state "
            "WHERE device_id = ? AND type_identifier = ?", (DEVICE["id"], HEART)
        ).fetchone()
        assert tuple(state) == ("anchor-1", "batch-1")
        before = {
            "counts": _counts(vault_path),
            "daily": conn.execute(
                "SELECT metric, count, avg, last FROM daily_metrics "
                "WHERE date = '2026-08-22' ORDER BY metric"
            ).fetchall(),
            "anchor": tuple(state),
        }
    finally:
        conn.close()

    with _client(vault, monkeypatch) as client:
        replay = client.post("/v1/ingest", json=payload,
                             headers={"x-health-secret": "hk-secret"})
        assert replay.status_code == 200
        assert replay.json()["applied"] is False

    conn = db.connect(vault_path, read_only=True)
    try:
        assert _counts(vault_path) == before["counts"]
        assert conn.execute(
            "SELECT metric, count, avg, last FROM daily_metrics "
            "WHERE date = '2026-08-22' ORDER BY metric"
        ).fetchall() == before["daily"]
        assert tuple(conn.execute(
            "SELECT anchor_token, last_batch_id FROM hk_sync_state "
            "WHERE device_id = ? AND type_identifier = ?", (DEVICE["id"], HEART)
        ).fetchone()) == before["anchor"]
    finally:
        conn.close()


def test_healthkit_three_batches_accumulate_non_allowlisted_raw(vault, vault_path,
                                                                monkeypatch):
    """A day's aggregate must include every HealthKit delivery, not its tail."""
    conn = vault.connect()
    db.init_db(conn)
    vault_mod.declare_vault(conn)
    conn.commit()
    conn.close()

    with _client(vault, monkeypatch) as client:
        for sequence, (hour, heart_value) in enumerate(
                zip((8, 10, 12), (150.0, 130.0, 140.0)), start=1):
            mass = _sample(
                "quantity", f"mass-{sequence}", MASS,
                f"2026-08-22T{hour:02d}:00:00-04:00",
                f"2026-08-22T{hour:02d}:00:01-04:00",
                80.0 + sequence, "lb",
            )
            heart = _sample(
                "quantity", f"hr-{sequence}", HEART,
                f"2026-08-22T{hour:02d}:30:00-04:00",
                f"2026-08-22T{hour:02d}:30:01-04:00",
                heart_value, "count/min",
            )
            response = client.post(
                "/v1/ingest",
                json=_payload(mass, heart, batch_id=f"batch-{sequence}",
                              sequence=sequence),
                headers={"x-health-secret": "hk-secret"},
            )
            assert response.status_code == 200, response.text

    conn = db.connect(vault_path, read_only=True)
    try:
        mass = conn.execute(
            "SELECT count, sum, avg, min, max, last FROM daily_metrics "
            "WHERE metric = 'body_mass' AND date = '2026-08-22'"
        ).fetchone()
        heart = conn.execute(
            "SELECT count, sum, avg, min, max, last FROM daily_metrics "
            "WHERE metric = 'heart_rate' AND date = '2026-08-22'"
        ).fetchone()
    finally:
        conn.close()

    assert tuple(mass) == (3, 246.0, 82.0, 81.0, 83.0, 83.0)
    assert tuple(heart) == (3, 420.0, 140.0, 130.0, 150.0, 140.0)


def test_healthkit_deletion_tombstone_and_tombstoned_add(vault, vault_path, monkeypatch):
    conn = vault.connect()
    db.init_db(conn)
    vault_mod.declare_vault(conn)
    conn.commit()
    conn.close()
    first = _sample("quantity", "hr-delete", HEART,
                    "2026-08-22T08:00:00-04:00", "2026-08-22T08:00:01-04:00",
                    100.0, "count/min")
    second = _sample("quantity", "hr-keep", HEART,
                     "2026-08-22T09:00:00-04:00", "2026-08-22T09:00:01-04:00",
                     120.0, "count/min")
    with _client(vault, monkeypatch) as client:
        assert client.post("/v1/ingest", json=_payload(first, second),
                           headers={"x-health-secret": "hk-secret"}).status_code == 200
        deletion = _payload(
            batch_id="batch-delete", sequence=2,
            deletions=[{"hk_uuid": "hr-delete", "type_identifier": HEART}],
        )
        assert client.post("/v1/ingest", json=deletion,
                           headers={"x-health-secret": "hk-secret"}).status_code == 200

        conn = db.connect(vault_path, read_only=True)
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM records WHERE hk_uuid = 'hr-delete'"
            ).fetchone()[0] == 0
            assert tuple(conn.execute(
                "SELECT count, avg, last FROM daily_metrics "
                "WHERE metric = 'heart_rate' AND date = '2026-08-22'"
            ).fetchone()) == (1, 120.0, 120.0)
            assert conn.execute(
                "SELECT COUNT(*) FROM hk_deletions WHERE hk_uuid = 'hr-delete'"
            ).fetchone()[0] == 1
            before = _counts(vault_path)
        finally:
            conn.close()

        replay = client.post("/v1/ingest", json=deletion,
                             headers={"x-health-secret": "hk-secret"})
        assert replay.json()["applied"] is False
        resurrect = _payload(batch_id="batch-stale-add", sequence=3)
        resurrect["samples"] = [first]
        assert client.post("/v1/ingest", json=resurrect,
                           headers={"x-health-secret": "hk-secret"}).status_code == 200

    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM records WHERE hk_uuid = 'hr-delete'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM hk_deletions WHERE hk_uuid = 'hr-delete'"
        ).fetchone()[0] == 1
        assert _counts(vault_path)["hk_deletions"] == before["hk_deletions"]
    finally:
        conn.close()


def test_healthkit_failure_rolls_back_data_and_anchor(vault, vault_path, monkeypatch):
    conn = vault.connect()
    db.init_db(conn)
    vault_mod.declare_vault(conn)
    conn.commit()
    conn.close()
    sample = _sample("quantity", "hr-fail", HEART,
                     "2026-08-22T08:00:00-04:00", "2026-08-22T08:00:01-04:00",
                     110.0, "count/min")

    def fail(*args, **kwargs):
        raise RuntimeError("injected insert failure")

    monkeypatch.setattr(receiver.db, "insert_records", fail)
    with pytest.raises(RuntimeError, match="injected insert failure"):
        with _client(vault, monkeypatch) as client:
            client.post("/v1/ingest", json=_payload(sample),
                        headers={"x-health-secret": "hk-secret"})

    assert _counts(vault_path) == {
        "records": 0, "daily_metrics": 0, "hk_sync_state": 0,
        "hk_deletions": 0, "commit_log": 0,
    }


def test_healthkit_malformed_batch_is_400_and_writes_nothing(vault, vault_path,
                                                              monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.post("/v1/ingest", json={"protocol_version": 1},
                               headers={"x-health-secret": "hk-secret"})
    assert response.status_code == 400
    assert _counts(vault_path) == {
        "records": 0, "daily_metrics": 0, "hk_sync_state": 0,
        "hk_deletions": 0, "commit_log": 0,
    }


def _set_history(vault, through):
    conn = vault.connect()
    db.init_db(conn)
    vault_mod.set_history_imported_through(conn, through)
    conn.commit()
    conn.close()


def _workout(uuid="workout-1", day="2026-08-22"):
    return {
        "hk_uuid": uuid,
        "workout_activity_type": "HKWorkoutActivityTypeRunning",
        "start": f"{day}T08:00:00-04:00",
        "end": f"{day}T08:30:00-04:00",
        "duration_min": 30.0,
        "energy_kcal": 250.0,
        "distance_mi": 3.0,
        "avg_heart_rate": 145.0,
        "max_heart_rate": 170.0,
        "source_revision": REVISION,
    }


def test_workout_only_ingest_writes_and_derives_its_day(vault, vault_path,
                                                         monkeypatch):
    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest",
            json=_payload(batch_id="workout-only", workouts=[_workout()]),
            headers={"x-health-secret": "hk-secret"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["records_seen"] == 0
    assert body["workouts_seen"] == 1
    assert body["workouts_added"] == 1
    assert body["dates"] == ["2026-08-22"]

    conn = db.connect(vault_path, read_only=True)
    try:
        row = conn.execute(
            "SELECT workout_type, local_date, dedupe_key FROM workouts"
        ).fetchone()
        assert tuple(row) == (
            "running", "2026-08-22",
            db.workout_key("running", "2026-08-22T12:00:00+00:00",
                           "2026-08-22T12:30:00+00:00"),
        )
        longest = conn.execute(
            "SELECT last FROM daily_metrics WHERE metric = 'longest_block_min' "
            "AND date = '2026-08-22'"
        ).fetchone()
        assert longest is not None
        commit_detail = conn.execute(
            "SELECT detail FROM commit_log WHERE key = ?",
            ("healthkit:watch-a:workout-only",),
        ).fetchone()[0]
        ingest_detail = conn.execute(
            "SELECT detail FROM ingest_log ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        conn.close()

    assert "workouts_added=1" in commit_detail
    assert "workouts_added=1" in ingest_detail

    with _client(vault, monkeypatch) as client:
        replay = client.post(
            "/v1/ingest",
            json=_payload(batch_id="workout-only", workouts=[_workout()]),
            headers={"x-health-secret": "hk-secret"},
        )
    assert replay.status_code == 200
    assert replay.json()["applied"] is False
    assert replay.json()["reason"] == "already_applied"
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 1
    finally:
        conn.close()


def test_late_workout_recomputes_arbitrated_distance_pair(
    vault, vault_path, monkeypatch
):
    """A later workout must re-derive the samples in its arbitration window."""
    day = "2026-08-22"
    gymkit_revision = {"source_name": "GymKit", "bundle_id": "com.apple.Health"}
    gymkit = _sample(
        "quantity", "distance-gymkit", "HKQuantityTypeIdentifierDistanceWalkingRunning",
        f"{day}T08:05:00-04:00", f"{day}T08:05:20-04:00", 0.1, "mi",
        revision=gymkit_revision,
    )
    iphone = _sample(
        "quantity", "distance-iphone", "HKQuantityTypeIdentifierDistanceWalkingRunning",
        f"{day}T08:10:00-04:00", f"{day}T08:10:20-04:00", 0.2, "mi",
        revision={"source_name": "Demo's iPhone", "bundle_id": "com.apple.Health"},
    )

    with _client(vault, monkeypatch) as client:
        first = client.post(
            "/v1/ingest",
            json=_payload(gymkit, iphone, batch_id="distance-samples", sequence=1),
            headers={"x-health-secret": "hk-secret"},
        )
        assert first.status_code == 200, first.text

        before = db.connect(vault_path, read_only=True)
        try:
            assert before.execute(
                "SELECT sum FROM daily_metrics WHERE metric = ? AND date = ?",
                ("distance_walking_running", day),
            ).fetchone()[0] == pytest.approx(0.3)
        finally:
            before.close()

        second = client.post(
            "/v1/ingest",
            json=_payload(batch_id="late-workout", sequence=2,
                          workouts=[_workout(day=day)]),
            headers={"x-health-secret": "hk-secret"},
        )
        assert second.status_code == 200, second.text

    conn = db.connect(vault_path, read_only=True)
    try:
        clause, params = db._arbitration(conn, "distance_walking_running", day)
        expected = conn.execute(
            "SELECT COUNT(*), SUM(value), AVG(value), MIN(value), MAX(value), "
            "(SELECT value FROM records WHERE metric = ? AND local_date = ?"
            f"{clause} ORDER BY start_utc DESC, end_utc DESC, id DESC LIMIT 1) "
            "FROM records WHERE metric = ? AND local_date = ?" + clause,
            ("distance_walking_running", day, *params,
             "distance_walking_running", day, *params),
        ).fetchone()
        stored = conn.execute(
            "SELECT count, sum, avg, min, max, last FROM daily_metrics "
            "WHERE metric = ? AND date = ?",
            ("distance_walking_running", day),
        ).fetchone()
    finally:
        conn.close()

    assert tuple(stored) == pytest.approx(tuple(expected))


def test_workout_before_history_watermark_returns_409(vault, vault_path,
                                                       monkeypatch):
    _set_history(vault, "2026-08-22")
    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest",
            json=_payload(batch_id="workout-before-watermark",
                          workouts=[_workout(day="2026-08-22")]),
            headers={"x-health-secret": "hk-secret"},
        )
    assert response.status_code == 409
    assert "workout dated 2026-08-22" in response.json()["detail"]
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM commit_log").fetchone()[0] == 0
    finally:
        conn.close()


def test_healthkit_history_watermark_refuses_whole_batch_without_evidence(
    vault, vault_path, monkeypatch
):
    """An imported-day replay cannot leave a partial tombstone or anchor."""
    _set_history(vault, "2026-08-21")
    sample = _sample(
        "quantity", "hr-before-watermark", HEART,
        "2026-08-21T08:00:00-04:00", "2026-08-21T08:00:01-04:00",
        110.0, "count/min",
    )
    payload = _payload(
        sample, batch_id="batch-before-watermark",
        deletions=[{"hk_uuid": "unrelated-delete", "type_identifier": HEART}],
    )

    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest", json=payload,
            headers={"x-health-secret": "hk-secret"},
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail.startswith(
        "history imported through 2026-08-21; refusing HealthKit batch "
        "containing record dated 2026-08-21"
    )
    # A wedged client is otherwise silent: it retries the identical batch
    # forever, because it commits its anchor only after a 2xx. The refusal has
    # to name the two ways out.
    assert "set_history_imported_through()" in detail
    assert "cutover" in detail
    assert _counts(vault_path) == {
        "records": 0, "daily_metrics": 0, "hk_sync_state": 0,
        "hk_deletions": 0, "commit_log": 0,
    }
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM ingest_log"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM commit_log "
            "WHERE key = 'healthkit:watch-a:batch-before-watermark'"
        ).fetchone()[0] == 0
    finally:
        conn.close()

    # M6 must opt into the historical rewrite explicitly. The same payload and
    # idempotency key is accepted because the refusal wrote no commit_log row.
    _set_history(vault, "2026-08-20")
    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest", json=payload,
            headers={"x-health-secret": "hk-secret"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["applied"] is True
    assert _counts(vault_path) == {
        "records": 1, "daily_metrics": 2, "hk_sync_state": 1,
        "hk_deletions": 1, "commit_log": 1,
    }


def test_healthkit_after_history_watermark_is_unaffected(
    vault, vault_path, monkeypatch
):
    _set_history(vault, "2026-08-21")
    sample = _sample(
        "quantity", "hr-after-watermark", HEART,
        "2026-08-22T08:00:00-04:00", "2026-08-22T08:00:01-04:00",
        120.0, "count/min",
    )
    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest", json=_payload(sample, batch_id="batch-after-watermark"),
            headers={"x-health-secret": "hk-secret"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["records_added"] == 1
    assert _counts(vault_path) == {
        "records": 1, "daily_metrics": 2, "hk_sync_state": 1,
        "hk_deletions": 0, "commit_log": 1,
    }


def test_healthkit_without_history_watermark_keeps_existing_behavior(
    vault, vault_path, monkeypatch
):
    sample = _sample(
        "quantity", "hr-unguarded-history", HEART,
        "2019-06-01T08:00:00-04:00", "2019-06-01T08:00:01-04:00",
        130.0, "count/min",
    )
    with _client(vault, monkeypatch) as client:
        response = client.post(
            "/v1/ingest", json=_payload(sample, batch_id="batch-no-watermark"),
            headers={"x-health-secret": "hk-secret"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["records_added"] == 1
    assert _counts(vault_path) == {
        "records": 1, "daily_metrics": 2, "hk_sync_state": 1,
        "hk_deletions": 0, "commit_log": 1,
    }


def test_a_tombstone_records_the_deleted_sample_date_and_metric(
    vault, vault_path, monkeypatch
):
    """The lag between a sample's date and its deletion has one chance to be
    captured: the `records` row is gone immediately afterwards, and the
    tombstone is all that survives. `deleted_at` alone gives when a deletion
    arrived, never how late it was — and how late is the figure the compaction
    window has to be designed against (#37).
    """
    conn = vault.connect()
    db.init_db(conn)
    vault_mod.declare_vault(conn)
    conn.commit()
    conn.close()

    sample = _sample("quantity", "hr-late", HEART,
                     "2026-08-11T08:00:00-04:00", "2026-08-11T08:00:01-04:00",
                     100.0, "count/min")
    with _client(vault, monkeypatch) as client:
        assert client.post("/v1/ingest", json=_payload(sample),
                           headers={"x-health-secret": "hk-secret"}).status_code == 200
        deletion = _payload(
            batch_id="batch-delete", sequence=2,
            deletions=[{"hk_uuid": "hr-late", "type_identifier": HEART}],
        )
        assert client.post("/v1/ingest", json=deletion,
                           headers={"x-health-secret": "hk-secret"}).status_code == 200

    conn = db.connect(vault_path, read_only=True)
    try:
        row = conn.execute(
            "SELECT sample_local_date, sample_metric, deleted_at "
            "FROM hk_deletions WHERE hk_uuid = 'hr-late'"
        ).fetchone()
    finally:
        conn.close()
    assert row["sample_local_date"] == "2026-08-11"
    assert row["sample_metric"] == "heart_rate"
    # The pair is what makes a lag computable at all.
    assert row["deleted_at"][:4].isdigit()


def test_a_tombstone_for_a_sample_this_vault_never_held_records_no_date(
    vault, vault_path, monkeypatch
):
    """A deletion for an unknown UUID still gets a tombstone — it is written
    before the add filter so a later stale add is still refused — but there is
    no sample to date it against. NULL keeps those rows OUT of the lag
    measurement; a zero would silently pull the distribution toward 'no lag'.
    """
    conn = vault.connect()
    db.init_db(conn)
    vault_mod.declare_vault(conn)
    conn.commit()
    conn.close()

    with _client(vault, monkeypatch) as client:
        deletion = _payload(
            batch_id="batch-unknown", sequence=1,
            deletions=[{"hk_uuid": "never-seen", "type_identifier": HEART}],
        )
        assert client.post("/v1/ingest", json=deletion,
                           headers={"x-health-secret": "hk-secret"}).status_code == 200

    conn = db.connect(vault_path, read_only=True)
    try:
        row = conn.execute(
            "SELECT sample_local_date, sample_metric FROM hk_deletions "
            "WHERE hk_uuid = 'never-seen'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "the tombstone itself must still be written"
    assert row["sample_local_date"] is None
    assert row["sample_metric"] is None


def test_an_existing_vault_gains_the_tombstone_columns_on_init(tmp_path):
    """Every vault ingesting before this existed has tombstones that cannot be
    back-filled, so the columns have to arrive by additive migration rather
    than by rebuild — the live device vault is one of them.
    """
    path = tmp_path / "old.db"
    conn = db.connect(path)
    conn.execute(
        "CREATE TABLE hk_deletions ("
        "device_id TEXT NOT NULL, type_identifier TEXT NOT NULL, "
        "hk_uuid TEXT NOT NULL, deleted_at TEXT NOT NULL, "
        "PRIMARY KEY (device_id, type_identifier, hk_uuid))")
    conn.execute(
        "INSERT INTO hk_deletions VALUES ('d', 't', 'u', '2026-08-01T00:00:00Z')")
    conn.commit()

    db.init_db(conn)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(hk_deletions)")}
    assert {"sample_local_date", "sample_metric"} <= columns
    # The pre-existing tombstone survives, undated — it cannot be recovered.
    row = conn.execute(
        "SELECT deleted_at, sample_local_date FROM hk_deletions").fetchone()
    conn.close()
    assert row["deleted_at"] == "2026-08-01T00:00:00Z"
    assert row["sample_local_date"] is None


def test_successful_ingest_leaves_operator_evidence_in_ingest_log(
    vault, vault_path, monkeypatch
):
    """A healthy sync must be visible from the server, not only to the phone.

    Until 2026-08-27 only the REJECT path wrote to `ingest_log`. Every counter a
    successful HealthKit ingest computes was returned in the HTTP response and
    then dropped, so `/health`'s `last_ingest` reported the final Health Auto
    Export batch of 2026-08-21 while 23,091 HealthKit rows were landing on
    2026-08-27 — the restored service looked like it had not ingested in six
    days. This pins the evidence rather than the wording.
    """
    conn = vault.connect()
    db.init_db(conn)
    vault_mod.declare_vault(conn)
    conn.commit()
    conn.close()

    client = _client(vault, monkeypatch)
    before = db.connect(vault_path, read_only=True)
    try:
        n_before = before.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0]
    finally:
        before.close()

    response = client.post(
        "/v1/ingest",
        json=_payload(_sample("quantity", "hr-log-1", HEART,
                              "2026-08-22T09:00:00-04:00",
                              "2026-08-22T09:00:01-04:00", 61.0, "count/min")),
        headers={"X-Health-Secret": "hk-secret"},
    )
    assert response.status_code == 200
    assert response.json()["applied"] is True

    after = db.connect(vault_path, read_only=True)
    try:
        rows = after.execute(
            "SELECT source, kind, detail FROM ingest_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        n_after = after.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0]
    finally:
        after.close()

    assert n_after > n_before, "a successful ingest wrote no operator evidence"
    source, kind, detail = rows[0], rows[1], rows[2]
    assert source == "healthkit"
    assert kind == "ingest"
    # The counters an operator needs to tell "nothing arrived" from "nothing was
    # new" apart. records_added is the one that distinguishes them.
    assert "records_seen=1" in detail
    assert "records_added=" in detail
    assert "unhandled=" in detail
