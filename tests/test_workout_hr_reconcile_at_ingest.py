"""Workout heart-rate reconciliation runs on the ingest paths (#96).

`db.reconcile_workout_heart_rate` corrects a workout's avg/max HR from its raw
heart_rate samples when the device summary contradicts them, behind unchanged
evidence gates. It used to run only from the maintenance entry point. Now:

* the live receiver runs it after each applied batch, scoped to the local dates
  that batch touched, after the batch is durable — a reconciliation failure is
  logged and never fails the ingest;
* the backfill runs it over the range it backfilled.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from health_advisor import backfill, db, receiver


DEVICE = {"id": "watch-a", "name": "iPhone", "model": "iPhone17,1"}
REVISION = {"source_name": "Apple Watch", "bundle_id": "com.apple.Health"}
HEART = "HKQuantityTypeIdentifierHeartRate"
SECRET = {"x-health-secret": "hk-secret"}

DAY = "2026-08-22"
# 30 samples, one a minute across a 30-minute run: avg 152.27, max 184.
HARD = [140] * 14 + [160] * 14 + [184, 184]
HARD_AVG = sum(HARD) / len(HARD)


def _payload(*, batch_id, sequence=1, samples=(), workouts=()):
    return {
        "protocol_version": 1, "device": DEVICE, "app_version": "1.0",
        "batch_id": batch_id, "batch_sequence": sequence,
        "sent_at": "2026-08-22T13:04:05Z",
        "anchors": [{"type_identifier": HEART, "from": None,
                     "to": f"anchor-{sequence}"}],
        "samples": list(samples), "deletions": [], "workouts": list(workouts),
    }


def _workout(day=DAY, avg=120.0, mx=130.0):
    return {
        "hk_uuid": f"workout-{day}",
        "workout_activity_type": "HKWorkoutActivityTypeRunning",
        "start": f"{day}T08:00:00-04:00", "end": f"{day}T08:30:00-04:00",
        "duration_min": 30.0, "energy_kcal": 250.0, "distance_mi": 3.0,
        "avg_heart_rate": avg, "max_heart_rate": mx,
        "source_revision": REVISION,
    }


def _hr_samples(day=DAY, values=HARD):
    t0 = datetime.fromisoformat(f"{day}T08:00:00-04:00")
    out = []
    for i, v in enumerate(values):
        t = (t0 + timedelta(minutes=i)).isoformat()
        out.append({"kind": "quantity", "hk_uuid": f"hr-{day}-{i}",
                    "type_identifier": HEART, "start": t, "end": t,
                    "value": float(v), "unit": "count/min",
                    "source_revision": REVISION})
    return out


def _client(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "hk-secret")
    return TestClient(receiver.create_app(vault))


def _post(client, payload):
    r = client.post("/v1/ingest", json=payload, headers=SECRET)
    assert r.status_code == 200, r.text
    assert r.json()["applied"] is True
    return r.json()


def _hr(vault_path, day=DAY):
    conn = db.connect(vault_path, read_only=True)
    try:
        r = conn.execute("SELECT avg_heart_rate, max_heart_rate FROM workouts "
                         "WHERE local_date = ?", (day,)).fetchone()
        return (r[0], r[1]) if r else None
    finally:
        conn.close()


def _last_ingest_log(vault_path, kind):
    conn = db.connect(vault_path, read_only=True)
    try:
        r = conn.execute("SELECT detail FROM ingest_log WHERE kind = ? "
                         "ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
        return r[0] if r else None
    finally:
        conn.close()


# ------------------------------------------------------------- receiver ----
def test_a_batch_whose_summary_contradicts_its_samples_is_reconciled(
        vault, vault_path, monkeypatch):
    with _client(vault, monkeypatch) as client:
        _post(client, _payload(batch_id="b1", samples=_hr_samples(),
                               workouts=[_workout()]))

    avg, mx = _hr(vault_path)
    assert abs(avg - HARD_AVG) < 0.01
    assert mx == 184.0
    assert "hr_reconciled=1" in _last_ingest_log(vault_path, "ingest")


def test_samples_arriving_in_a_later_batch_reconcile_the_earlier_workout(
        vault, vault_path, monkeypatch):
    # The phone ships each HealthKit type on its own anchor, so a workout and
    # its heart-rate series need not share a batch. The samples' dates are
    # touched dates too.
    with _client(vault, monkeypatch) as client:
        _post(client, _payload(batch_id="w", workouts=[_workout()]))
        assert _hr(vault_path) == (120.0, 130.0)   # no samples yet: no evidence
        _post(client, _payload(batch_id="s", sequence=2,
                               samples=_hr_samples()))

    assert abs(_hr(vault_path)[0] - HARD_AVG) < 0.01


def test_a_workout_outside_the_batch_dates_is_untouched(
        conn, vault, vault_path, monkeypatch):
    # Seed a drifted workout two days earlier, with samples that would
    # overrule it, so only the scoping keeps it as it is.
    other = "2026-08-20"
    with _client(vault, monkeypatch) as client:
        _post(client, _payload(batch_id="seed", samples=_hr_samples(other),
                               workouts=[_workout(other, avg=152.0, mx=184.0)]))
    conn.execute("UPDATE workouts SET avg_heart_rate = 100, max_heart_rate = 110 "
                 "WHERE local_date = ?", (other,))
    conn.commit()

    with _client(vault, monkeypatch) as client:
        _post(client, _payload(batch_id="b1", sequence=2,
                               samples=_hr_samples(), workouts=[_workout()]))

    assert abs(_hr(vault_path)[0] - HARD_AVG) < 0.01
    assert _hr(vault_path, other) == (100.0, 110.0)


def test_a_failing_reconciliation_leaves_the_ingest_committed_and_logs(
        vault, vault_path, monkeypatch, caplog):
    def boom(*_a, **_k):
        raise RuntimeError("reconcile exploded")
    monkeypatch.setattr(db, "reconcile_workout_heart_rate", boom)

    with caplog.at_level(logging.WARNING, logger=receiver.__name__):
        with _client(vault, monkeypatch) as client:
            body = _post(client, _payload(batch_id="b1", samples=_hr_samples(),
                                          workouts=[_workout()]))

    assert body["workouts_added"] == 1
    assert _hr(vault_path) == (120.0, 130.0)          # durable, unreconciled
    conn = db.connect(vault_path, read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM records WHERE metric = "
                            "'heart_rate'").fetchone()[0] == len(HARD)
        # The derive step after it still ran and did not fail. The workout has
        # no distance samples, so no block row is written for its day.
        assert conn.execute(
            "SELECT COUNT(*) FROM ingest_log WHERE source = 'derive_error'"
        ).fetchone()[0] == 0
    finally:
        conn.close()
    assert "reconcile exploded" in _last_ingest_log(vault_path,
                                                    "reconcile_hr_error")
    assert "workout heart-rate reconciliation failed" in caplog.text
    assert "hr_reconciled=0" in _last_ingest_log(vault_path, "ingest")


def test_a_second_pass_changes_nothing(vault, vault_path, monkeypatch):
    with _client(vault, monkeypatch) as client:
        _post(client, _payload(batch_id="b1", samples=_hr_samples(),
                               workouts=[_workout()]))
        first = _hr(vault_path)
        # A later batch touching the same day re-runs the pass.
        _post(client, _payload(batch_id="b2", sequence=2,
                               workouts=[_workout()]))
    assert _hr(vault_path) == first
    assert "hr_reconciled=0" in _last_ingest_log(vault_path, "ingest")

    conn = db.connect(vault_path)
    try:
        assert db.reconcile_workout_heart_rate(conn, local_dates={DAY}) == 0
    finally:
        conn.close()


# ------------------------------------------------------------- backfill ----
def _xml_hr(day, values=HARD):
    t0 = datetime.fromisoformat(f"{day}T08:00:00")
    out = []
    for i, v in enumerate(values):
        t = (t0 + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        out.append(f'<Record type="{HEART}" value="{v}" unit="count/min" '
                   f'sourceName="Watch" startDate="{t} -0400" '
                   f'endDate="{t} -0400"/>')
    return out


def _xml_workout(day):
    return (f'<Workout workoutActivityType="HKWorkoutActivityTypeRunning" '
            f'duration="30.0" durationUnit="min" sourceName="Watch" '
            f'startDate="{day} 08:00:00 -0400" endDate="{day} 08:30:00 -0400"/>')


def _export(tmp_path, items):
    xml = tmp_path / "export.xml"
    xml.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<HealthData>\n'
                   + "\n".join(items) + "\n</HealthData>\n")
    return str(xml)


def test_backfill_reconciles_the_range_it_backfilled(tmp_path, vault_path):
    # An export.zip workout carries no HR summary; its samples fill it in.
    summary = backfill.run(
        xml_path=_export(tmp_path, [_xml_workout(DAY), *_xml_hr(DAY)]),
        db_path=str(vault_path))

    assert summary["workouts_hr_reconciled"] == 1
    avg, mx = _hr(vault_path)
    assert abs(avg - HARD_AVG) < 0.01 and mx == 184.0

    again = backfill.run(
        xml_path=_export(tmp_path, [_xml_workout(DAY), *_xml_hr(DAY)]),
        db_path=str(vault_path))
    assert again["workouts_hr_reconciled"] == 0


def test_workouts_only_backfill_reconciles_and_leaves_other_years_alone(
        tmp_path, vault_path, conn):
    # Samples already in the vault (the receiver's), a workout from export.zip.
    old = "2025-01-10"
    backfill.run(xml_path=_export(tmp_path, [*_xml_hr(DAY), *_xml_hr(old),
                                             _xml_workout(old)]),
                 db_path=str(vault_path))
    conn.execute("UPDATE workouts SET avg_heart_rate = 100, max_heart_rate = 110 "
                 "WHERE local_date = ?", (old,))
    conn.commit()

    summary = backfill.run(xml_path=_export(tmp_path, [_xml_workout(DAY)]),
                           db_path=str(vault_path), workouts_only=True)

    assert summary["workouts_hr_reconciled"] == 1
    assert abs(_hr(vault_path)[0] - HARD_AVG) < 0.01
    assert _hr(vault_path, old) == (100.0, 110.0)     # outside the range


def test_samples_after_midnight_reach_a_workout_dated_the_day_before(
        vault, vault_path, monkeypatch):
    # A workout is dated by its start. Its first 20 minutes of samples arrive
    # with it but cover too little of it; the last 20 arrive alone, dated the
    # next day, and only then is the evidence complete.
    day, nxt = "2026-08-21", "2026-08-22"
    workout = dict(_workout(day), start=f"{day}T23:40:00-04:00",
                   end=f"{nxt}T00:20:00-04:00", duration_min=40.0)

    def samples(start, values, tag):
        t0 = datetime.fromisoformat(start)
        return [{"kind": "quantity", "hk_uuid": f"hr-{tag}-{i}",
                 "type_identifier": HEART,
                 "start": (t0 + timedelta(minutes=i)).isoformat(),
                 "end": (t0 + timedelta(minutes=i)).isoformat(),
                 "value": float(v), "unit": "count/min",
                 "source_revision": REVISION} for i, v in enumerate(values)]

    with _client(vault, monkeypatch) as client:
        _post(client, _payload(batch_id="a", workouts=[workout],
                               samples=samples(f"{day}T23:40:00-04:00",
                                               [150] * 20, "a")))
        assert _hr(vault_path, day) == (120.0, 130.0)
        _post(client, _payload(batch_id="b", sequence=2,
                               samples=samples(f"{nxt}T00:00:00-04:00",
                                               [160] * 20, "b")))

    assert _hr(vault_path, day) == (155.0, 160.0)
