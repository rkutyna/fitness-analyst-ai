"""#150 — sub-workouts arriving as top-level sessions.

2026-08-25 in the live vault carried THREE `running` rows for one treadmill
session: the real 57.6-min / 3.45-mi run, and two 4-min / 0.2-mi rows nested
inside it (one sharing its exact start_utc). Each is a distinct HealthKit
workout with its own `hk_uuid`, so nothing upstream refuses them — and
`workout_key` is type|start|end, so a contained row hashes differently from its
container BY CONSTRUCTION. No key over those three fields can collide a row
with a row that contains it, which is why the refusal is a containment rule in
the ingest path and not a re-key (a re-key would reach the 2019-2022 history).

The discriminator is same-source, and it is measured. The snapshot holds 48
same-type containment pairs; 47 are CROSS-source (ErgData nested inside the
Apple Watch, 2019-2021), two devices legitimately recording one session, and
both rows are real historical record. Exactly one is same-source, and it is
this defect. So the cross-source case must still store both rows.

The source strings below carry the real curly apostrophe and NO-BREAK SPACE.
Any rule comparing against a hardcoded literal silently never matches; the
implementation compares the two rows' stored values to each other.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from health_advisor import db, hk_parse, receiver
from health_advisor import vault as vault_mod

WATCH = "Demo’s Apple Watch"     # curly apostrophe + NBSP, as the phone sends it
ERG = "ErgData"

DEVICE = {"id": "device-1", "name": "iPhone", "model": "iPhone17,1"}


def _revision(source_name: str) -> dict:
    return {"source_name": source_name, "bundle_id": "com.apple.Health",
            "version": "26.0"}


# The three real rows, from the live vault (issue #150 and its second comment).
# Ends are start + the measured duration.
REAL_RUN = ("2026-08-25T10:06:41-04:00", "2026-08-25T11:04:17-04:00", 57.6, 3.45, 129, 172)
FRAG_A = ("2026-08-25T10:06:41-04:00", "2026-08-25T10:10:57-04:00", 4.27, 0.201, 95, 101)
FRAG_B = ("2026-08-25T10:11:05-04:00", "2026-08-25T10:15:32-04:00", 4.44, 0.201, 99, 103)


def _workout(uuid: str, spec, source: str = WATCH,
             activity: str = "HKWorkoutActivityTypeRunning") -> dict:
    start, end, minutes, miles, avg_hr, max_hr = spec
    return {
        "hk_uuid": uuid,
        "workout_activity_type": activity,
        "start": start, "end": end,
        "duration_min": minutes,
        "energy_kcal": minutes * 8.0,
        "distance_mi": miles,
        "avg_heart_rate": float(avg_hr),
        "max_heart_rate": float(max_hr),
        "source_revision": _revision(source),
    }


def _payload(*workouts) -> dict:
    return {
        "protocol_version": 1, "device": DEVICE, "app_version": "1.0.0+7",
        "batch_id": "batch-150", "batch_sequence": 1,
        "sent_at": "2026-08-25T16:00:00Z",
        "anchors": [], "samples": [], "deletions": [],
        "workouts": list(workouts),
    }


def _ingest(dbp, *workouts):
    """Parse a HealthKit page and apply its workouts, exactly as the receiver
    does. Returns (added, stored rows, refusals)."""
    conn = db.connect(dbp)
    db.init_db(conn)
    parsed = hk_parse.parse_payload(_payload(*workouts))
    assert parsed["unhandled"] == []
    refused: list[tuple[dict, dict]] = []
    added = db.insert_workouts(conn, parsed["workouts"],
                               report=lambda row, outer: refused.append((row, outer)))
    conn.commit()
    rows = [dict(r) for r in conn.execute(
        "SELECT workout_type, start_utc, end_utc, duration_min, distance_mi, "
        "source, hk_uuid FROM workouts ORDER BY start_utc, end_utc")]
    conn.close()
    return added, rows, refused


def test_the_three_aug_25_rows_become_one(tmp_path):
    """The real batch shape: one session, two fragments, one page."""
    added, rows, refused = _ingest(
        tmp_path / "h.db",
        _workout("frag-a", FRAG_A), _workout("real", REAL_RUN), _workout("frag-b", FRAG_B))

    assert added == 1
    assert len(rows) == 1
    assert rows[0]["hk_uuid"] == "real"
    assert round(rows[0]["duration_min"], 1) == 57.6
    assert round(rows[0]["distance_mi"], 2) == 3.45
    # It refuses OUT LOUD — a silent drop is the other half of the defect.
    assert len(refused) == 2
    assert {round(r["duration_min"], 2) for r, _ in refused} == {4.27, 4.44}
    assert all(round(outer["duration_min"], 1) == 57.6 for _, outer in refused)


def test_fragments_are_refused_in_any_arrival_order(tmp_path):
    """Container last in the page, and container in an earlier page."""
    _, rows, _ = _ingest(tmp_path / "order.db", _workout("frag-b", FRAG_B),
                         _workout("frag-a", FRAG_A), _workout("real", REAL_RUN))
    assert len(rows) == 1 and round(rows[0]["duration_min"], 1) == 57.6

    later = tmp_path / "later.db"
    _ingest(later, _workout("real", REAL_RUN))                    # session syncs first
    added, rows, refused = _ingest(later, _workout("frag-a", FRAG_A),
                                   _workout("frag-b", FRAG_B))    # fragments follow
    assert added == 0 and len(refused) == 2
    assert len(rows) == 1 and rows[0]["hk_uuid"] == "real"


def test_cross_source_contained_workout_still_stores(tmp_path):
    """ErgData inside an Apple Watch row: 47 of the snapshot's 48 pairs. Both
    rows are real record and neither may be refused."""
    outer = ("2019-11-05T13:00:00-05:00", "2019-11-05T13:40:00-05:00", 40.0, 2.30, 148, 171)
    inner = ("2019-11-05T13:04:00-05:00", "2019-11-05T13:34:00-05:00", 30.0, 2.10, 150, 168)
    added, rows, refused = _ingest(
        tmp_path / "cross.db",
        _workout("watch", outer, activity="HKWorkoutActivityTypeRowing"),
        _workout("erg", inner, source=ERG, activity="HKWorkoutActivityTypeRowing"))

    assert refused == []
    assert added == 2 and len(rows) == 2
    assert {r["source"] for r in rows} == {WATCH, ERG}


def test_source_is_compared_row_to_row_not_to_a_literal(tmp_path):
    """The stored source carries a NBSP and a curly apostrophe. A plain-ASCII
    twin is a different device string and must not read as the same source."""
    plain = "Demo's Apple Watch"
    assert plain != WATCH

    added, rows, _ = _ingest(tmp_path / "nbsp.db", _workout("real", REAL_RUN),
                             _workout("frag-a", FRAG_A, source=plain))
    assert added == 2 and len(rows) == 2       # different sources -> both kept

    added2, rows2, _ = _ingest(tmp_path / "nbsp2.db", _workout("real", REAL_RUN),
                               _workout("frag-a", FRAG_A))
    assert added2 == 1 and len(rows2) == 1     # identical sources -> refused


def test_a_resighting_of_a_stored_session_still_merges(tmp_path):
    """The containment rule must not break the one-directional column merge: a
    session we already hold is an update, never a fragment."""
    conn = db.connect(tmp_path / "merge.db")
    db.init_db(conn)
    first = hk_parse.parse_payload(_payload(_workout("real", REAL_RUN)))["workouts"]
    for row in first:
        row["avg_heart_rate"] = None                    # one half: no HR summary
    assert db.insert_workouts(conn, first) == 1
    second = hk_parse.parse_payload(_payload(_workout("real", REAL_RUN)))["workouts"]
    assert db.insert_workouts(conn, second) == 0        # other half: fills the hole
    conn.commit()
    got = conn.execute("SELECT COUNT(*) n, MAX(avg_heart_rate) hr FROM workouts").fetchone()
    conn.close()
    assert got["n"] == 1 and got["hr"] == 129.0


def test_the_receiver_stores_one_row_and_says_what_it_refused(
    vault, vault_path, monkeypatch
):
    """End to end over /v1/ingest: the day carries one running row, and the
    refusals are named in the response and in the ingest log. A silent drop
    would leave `workouts_added=1` looking identical to a clean batch."""
    conn = vault.connect()
    db.init_db(conn)
    vault_mod.declare_vault(conn)
    conn.commit()
    conn.close()

    payload = _payload(_workout("frag-a", FRAG_A), _workout("real", REAL_RUN),
                       _workout("frag-b", FRAG_B))
    monkeypatch.setattr(receiver, "SHARED_SECRET", "hk-secret")
    with TestClient(receiver.create_app(vault)) as client:
        response = client.post("/v1/ingest", json=payload,
                               headers={"x-health-secret": "hk-secret"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["workouts_seen"] == 3
    assert body["workouts_added"] == 1
    assert body["workouts_rejected"] == 2
    assert len(body["workouts_rejected_detail"]) == 2
    assert all("contained in" in line for line in body["workouts_rejected_detail"])

    conn = db.connect(vault_path, read_only=True)
    try:
        rows = conn.execute(
            "SELECT hk_uuid, duration_min FROM workouts "
            "WHERE local_date = '2026-08-25' AND workout_type = 'running'").fetchall()
        detail = conn.execute(
            "SELECT detail FROM ingest_log ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    assert [r["hk_uuid"] for r in rows] == ["real"]
    assert round(rows[0]["duration_min"], 1) == 57.6
    assert "workouts_contained_rejected=2" in detail["detail"]


def test_only_containment_is_refused(tmp_path):
    """Overlapping-but-not-contained, and a different activity type, both still
    store. Containment is the measured signature; overlap is a larger claim."""
    overlap = ("2026-08-25T11:00:00-04:00", "2026-08-25T11:20:00-04:00", 20.0, 1.2, 120, 150)
    added, rows, refused = _ingest(
        tmp_path / "overlap.db",
        _workout("real", REAL_RUN), _workout("next", overlap),
        _workout("walk", FRAG_A, activity="HKWorkoutActivityTypeWalking"))

    assert refused == []
    assert added == 3 and len(rows) == 3
