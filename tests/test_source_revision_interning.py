"""Source revisions are stored interned (downstream issue #702).

`records.source_revision_json` used to hold a per-sample copy of a HealthKit
source-revision object: a fifth of a synthetic one-year vault, from three
distinct values. New rows store an id into `source_revisions` instead and
leave the JSON column NULL; `db.intern_source_revisions` moves old rows
explicitly; `db.source_revision_json` reads either form.

The oracle here is the parse output itself, captured before anything is
inserted, and a control written with the pre-#702 INSERT statement copied
verbatim below -- not the code under test.
"""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import db, hk_parse


DEVICE = {"id": "device-1", "name": "iPhone", "model": "iPhone17,1"}
# Three revisions with every field the contract allows, as a watch, a phone
# and an app write them: the realistic shape is few distinct, long values.
REVISIONS = [
    {"source_name": "Synthetic Watch", "bundle_id": "com.apple.health.watch",
     "version": "11.0", "product_type": "Watch7,5",
     "operating_system_version": "11.0.0"},
    {"source_name": "Synthetic Phone", "bundle_id": "com.apple.health.phone",
     "version": "26.0", "product_type": "iPhone17,1",
     "operating_system_version": "26.0.1"},
    {"source_name": "Synthetic App", "bundle_id": "org.example.tracker",
     "version": "3.2.1"},
]

# The insert statement exactly as it was before downstream issue #702: the
# control writes every row the old way, JSON and all.
OLD_INSERT = (
    "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
    "start_local, local_date, source, origin, dedupe_key, hk_uuid, "
    "hk_type_identifier, source_revision_json, hk_device_id) "
    "VALUES (:metric, :value, :unit, :start_utc, :end_utc, :start_local, "
    ":local_date, :source, :origin, :dedupe_key, :hk_uuid, "
    ":hk_type_identifier, :source_revision_json, :hk_device_id) "
    "ON CONFLICT(dedupe_key) DO UPDATE SET "
    "value = excluded.value, unit = excluded.unit, origin = excluded.origin"
)

# The records table before downstream issue #702: no source_revision_id.
OLD_RECORDS_DDL = """
CREATE TABLE records (
    id INTEGER PRIMARY KEY, metric TEXT NOT NULL, value REAL, unit TEXT,
    start_utc TEXT NOT NULL, end_utc TEXT NOT NULL, start_local TEXT,
    local_date TEXT NOT NULL, source TEXT,
    origin TEXT NOT NULL DEFAULT 'backfill', dedupe_key TEXT NOT NULL UNIQUE,
    hk_uuid TEXT, hk_type_identifier TEXT, source_revision_json TEXT,
    hk_device_id TEXT
);
"""
OLD_COLUMNS = ("id", "metric", "value", "unit", "start_utc", "end_utc",
               "start_local", "local_date", "source", "origin", "dedupe_key",
               "hk_uuid", "hk_type_identifier", "source_revision_json",
               "hk_device_id")
REVISION_COLUMNS = {"source_revision_json", "source_revision_id"}


def _samples(n: int, *, uuid_prefix: str = "s") -> list[dict]:
    """``n`` heart-rate samples a minute apart, revisions in rotation, plus
    one sleep sample (one UUID, one canonical row per stage it maps to)."""
    samples = []
    for i in range(n):
        hour, minute = divmod(i, 60)
        day, hour = divmod(hour, 24)
        start = f"2026-08-{1 + day:02d}T{hour:02d}:{minute:02d}:00-04:00"
        samples.append({
            "kind": "quantity", "hk_uuid": f"{uuid_prefix}-{i:08d}",
            "type_identifier": "HKQuantityTypeIdentifierHeartRate",
            "start": start, "end": start, "value": 60 + i % 40,
            "unit": "count/min", "source_revision": REVISIONS[i % 3],
            "device": {"name": "Synthetic Watch", "model": "Watch7,5"},
        })
    samples.append({
        "kind": "category", "hk_uuid": f"{uuid_prefix}-sleep",
        "type_identifier": "HKCategoryTypeIdentifierSleepAnalysis",
        "start": "2026-07-31T23:50:00-04:00",
        "end": "2026-08-01T00:20:00-04:00",
        "value": "HKCategoryValueSleepAnalysisAsleepCore", "unit": None,
        "source_revision": REVISIONS[1],
    })
    return samples


def _parsed(samples: list[dict], batch_id: str = "batch-1") -> list[dict]:
    payload = {
        "protocol_version": 1, "device": DEVICE, "app_version": "1.0.0+7",
        "batch_id": batch_id, "batch_sequence": 1,
        "sent_at": "2026-08-30T13:04:05Z", "anchors": [],
        "samples": samples, "deletions": [], "workouts": [],
    }
    records = hk_parse.parse_payload(payload)["records"]
    assert records, "precondition: the payload parsed to rows"
    return records


def _fresh(path) -> sqlite3.Connection:
    conn = db.connect(path)
    db.init_db(conn)
    return conn


def _old_insert(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(OLD_INSERT, rows)
    conn.commit()


def _rows_by_key(conn: sqlite3.Connection) -> dict[str, dict]:
    return {row["dedupe_key"]: {k: row[k] for k in row.keys()
                                if k not in REVISION_COLUMNS}
            for row in conn.execute("SELECT * FROM records")}


def _recovered(conn: sqlite3.Connection) -> dict[str, str | None]:
    return {row["dedupe_key"]: db.source_revision_json(conn, row["id"])
            for row in conn.execute("SELECT id, dedupe_key FROM records")}


def test_interned_rows_recover_parse_output_and_match_old_rows(tmp_path):
    first = _parsed(_samples(300))
    # A re-send of the first 30 samples under a newer revision of the same
    # source (so the same dedupe key): the old upsert kept the stored row's
    # revision, and so must the interned one.
    resent = _parsed([dict(s, source_revision=dict(s["source_revision"],
                                                   version="99.0"))
                      for s in _samples(30)], batch_id="batch-2")
    # The oracle, captured from the parse output before any insert: the
    # first sighting of each key wins.
    expected: dict[str, str] = {}
    for row in first + resent:
        expected.setdefault(row["dedupe_key"], row["source_revision_json"])
    assert len(set(expected.values())) == 3, "precondition: three revisions"

    control = _fresh(tmp_path / "control.db")
    _old_insert(control, first)
    _old_insert(control, resent)
    interned = _fresh(tmp_path / "interned.db")
    db.insert_records(interned, first)
    db.insert_records(interned, resent)
    interned.commit()

    assert _recovered(interned) == expected
    assert _recovered(control) == expected       # the oracle itself holds
    # Identity and every other column are unchanged, row for row, id included.
    assert _rows_by_key(interned) == _rows_by_key(control)
    # The storage really is interned: no per-row JSON, three stored values.
    assert interned.execute(
        "SELECT COUNT(*) FROM records WHERE source_revision_json IS NOT NULL"
    ).fetchone()[0] == 0
    assert interned.execute(
        "SELECT COUNT(*) FROM records WHERE source_revision_id IS NULL"
    ).fetchone()[0] == 0
    assert interned.execute(
        "SELECT COUNT(DISTINCT source_revision_id) FROM records"
    ).fetchone()[0] == 3
    # The re-send's newer revisions were interned before its rows met the
    # conflict clause, so they are stored though no row names them: one
    # short row per distinct value, never per sample.
    assert interned.execute(
        "SELECT COUNT(*) FROM source_revisions").fetchone()[0] == 6


def test_a_row_without_a_revision_stays_null(conn):
    row = dict(_parsed(_samples(1))[0], source_revision_json=None)
    db.insert_records(conn, [row])
    rid, rev_id = conn.execute(
        "SELECT id, source_revision_id FROM records").fetchone()
    assert rev_id is None
    assert db.source_revision_json(conn, rid) is None
    assert conn.execute("SELECT COUNT(*) FROM source_revisions").fetchone()[0] == 0


def test_interned_vault_is_smaller_after_vacuum(tmp_path):
    rows = _parsed(_samples(6000))
    old_path, new_path = tmp_path / "old.db", tmp_path / "new.db"
    old = _fresh(old_path)
    _old_insert(old, rows)
    new = _fresh(new_path)
    db.insert_records(new, rows)
    new.commit()
    for conn in (old, new):
        conn.execute("VACUUM")
        conn.close()
    old_size, new_size = old_path.stat().st_size, new_path.stat().st_size
    reduction = 1 - new_size / old_size
    print(f"\n#702 size: {len(rows)} rows, old {old_size} B, interned "
          f"{new_size} B, {reduction:.2%} smaller")
    assert reduction >= 0.15


def _old_shape_vault(tmp_path, n: int = 53) -> tuple[sqlite3.Connection, list]:
    """A vault as the pre-#702 code left it, after the new init_db ran: every
    row's revision in source_revision_json, no id. Some rows carry none."""
    conn = _fresh(tmp_path / "old_shape.db")
    rows = _parsed(_samples(n))
    rows[5] = dict(rows[5], source_revision_json=None)
    _old_insert(conn, rows)
    return conn, rows


def test_migration_is_lossless_and_a_second_run_is_a_noop(tmp_path):
    conn, rows = _old_shape_vault(tmp_path)
    before = _recovered(conn)
    holding = sum(r["source_revision_json"] is not None for r in rows)
    other_before = _rows_by_key(conn)

    report = db.intern_source_revisions(conn, batch_rows=7)
    assert report == {"rows_interned": holding,
                      "batches": -(-holding // 7),
                      "revisions": 3, "remaining": 0}
    assert _recovered(conn) == before
    assert _rows_by_key(conn) == other_before
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE source_revision_json IS NOT NULL"
    ).fetchone()[0] == 0

    again = db.intern_source_revisions(conn, batch_rows=7)
    assert again == {"rows_interned": 0, "batches": 0, "revisions": 3,
                     "remaining": 0}
    assert _recovered(conn) == before


def test_an_interrupted_migration_resumes(tmp_path):
    conn, rows = _old_shape_vault(tmp_path)
    before = _recovered(conn)
    holding = sum(r["source_revision_json"] is not None for r in rows)
    real_commit = conn.commit

    def commit_then_die():
        real_commit()
        raise RuntimeError("interrupted after the first batch")

    conn.commit = commit_then_die
    with pytest.raises(RuntimeError):
        db.intern_source_revisions(conn, batch_rows=10)
    del conn.commit
    conn.rollback()
    # Exactly the first batch landed, and it is already readable.
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE source_revision_id IS NOT NULL"
    ).fetchone()[0] == 10
    assert _recovered(conn) == before

    report = db.intern_source_revisions(conn, batch_rows=10)
    assert report["rows_interned"] == holding - 10
    assert report["remaining"] == 0
    assert _recovered(conn) == before


def test_init_db_adds_the_shape_and_rewrites_no_record(tmp_path):
    path = tmp_path / "pre702.db"
    conn = sqlite3.connect(path)
    conn.executescript(OLD_RECORDS_DDL)
    rows = _parsed(_samples(40))
    conn.executemany(OLD_INSERT, rows)
    conn.commit()
    cols = ", ".join(OLD_COLUMNS)
    before = conn.execute(f"SELECT {cols} FROM records ORDER BY id").fetchall()
    conn.close()

    conn = db.connect(path)
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    db.init_db(conn)
    conn.set_trace_callback(None)

    assert db.has_table(conn, "source_revisions")
    assert db.has_column(conn, "records", "source_revision_id")
    after = [tuple(r) for r in conn.execute(
        f"SELECT {cols} FROM records ORDER BY id")]
    assert after == [tuple(r) for r in before]
    # The migration did not run: the JSON is where it was, no id was written,
    # and no statement init_db issued updated records.
    assert conn.execute(
        "SELECT COUNT(*) FROM records WHERE source_revision_id IS NOT NULL"
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM source_revisions").fetchone()[0] == 0
    assert not [s for s in statements
                if s.lstrip().upper().startswith("UPDATE RECORDS")]
    # And the old form reads back through the helper unchanged.
    expected = {r["dedupe_key"]: r["source_revision_json"] for r in rows}
    assert _recovered(conn) == expected
