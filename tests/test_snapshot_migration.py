"""Importing the ten-year snapshot into a vault (T-007).

The import half of D7 only. The re-derivation half is gated on #8, and the
distinction is load-bearing: **this migration must carry the two-device step
double-count forward untouched** — 1.12x (2019), 1.21x (2020), 1.45x (2021),
1.19x (2022). G-01 confirmed on device that Apple resolves overlap at interval
granularity with the winning source varying through the day, which matches no
simple rule; an independent methodology review returned NOT USABLE on
reconstructing it. A migration that "fixes" the double-count with a merge
heuristic is manufacturing data.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from health_advisor import db as dbmod
from health_advisor import vault as V
from health_advisor.context import VaultContext, VaultOwnershipError
from tests.conftest import seed_metric, seed_workout

from tests.test_d3_contract import _record, _two_instruments  # noqa: E402

PACKAGE = Path(dbmod.__file__).resolve().parent


def _seed_snapshot(conn):
    """A miniature snapshot: some series with raw samples, some without."""
    for hour in range(6):
        ts = f"2026-08-01T{10 + hour:02d}:00:00+00:00"
        for metric, unit, value in (("heart_rate", "count/min", 120.0),
                                    ("step_count", "count", 500.0),
                                    ("basal_energy", "kcal", 60.0)):
            conn.execute(
                "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
                "start_local, local_date, source, origin, dedupe_key) "
                "VALUES (?, ?, ?, ?, ?, ?, '2026-08-01', 'Watch', 'backfill', ?)",
                (metric, value, unit, ts, ts, ts, f"{metric}|{ts}"))
    seed_metric(conn, "body_mass", "2026-08-01", [188.8])
    seed_workout(conn, "running", "2026-08-01", 41.0, 2.04)
    dbmod.recompute_daily_metrics(conn, full=True)
    conn.commit()


@pytest.fixture
def snapshot(conn, vault_path):
    _seed_snapshot(conn)
    conn.close()
    return vault_path


def _content_hash(path: Path) -> str:
    """Everything a reader can see, independent of file layout.

    Hashing the file bytes would compare page allocation and free lists, which
    two identical builds are not obliged to agree on. What has to be identical
    is the data.
    """
    conn = dbmod.connect(path, read_only=True)
    try:
        digest = hashlib.sha256()
        tables = sorted(r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"))
        for table in tables:
            digest.update(table.encode())
            for row in conn.execute(f'SELECT * FROM "{table}"'):
                # `created_at` is when this file was built, which two builds are
                # entitled to disagree about. It is metadata, not data — the
                # claim is that a reader cannot distinguish the two vaults'
                # CONTENT, and a build timestamp is not content.
                if table == "vault_meta" and row[0] == "created_at":
                    continue
                digest.update(repr(tuple(row)).encode())
        return digest.hexdigest()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# the migration itself
# --------------------------------------------------------------------------- #
def test_the_vault_belongs_to_the_user_it_was_built_for(snapshot, tmp_path):
    out = tmp_path / "user1.db"
    report = V.build_vault(snapshot, out, owner="user-1", measure_gzip=False)

    assert report["owner"] == "user-1"
    assert VaultContext.local(out, user_id="user-1").owner() == "user-1"
    with pytest.raises(VaultOwnershipError):
        VaultContext.local(out, user_id="user-2").read_only()


def test_running_the_migration_twice_adds_nothing(snapshot, tmp_path):
    """Idempotent in the sense that matters: the second run leaves a vault a
    reader cannot distinguish from the first."""
    first = tmp_path / "a.db"
    second = tmp_path / "b.db"
    V.build_vault(snapshot, first, owner="user-1", measure_gzip=False)
    V.build_vault(snapshot, second, owner="user-1", measure_gzip=False)

    assert _content_hash(first) == _content_hash(second)

    # And rebuilding over an existing vault replaces rather than accumulates.
    before = _content_hash(first)
    V.build_vault(snapshot, first, owner="user-1", replace=True, measure_gzip=False)
    assert _content_hash(first) == before


def test_a_vault_is_never_overwritten_by_accident(snapshot, tmp_path):
    out = tmp_path / "user1.db"
    V.build_vault(snapshot, out, owner="user-1", measure_gzip=False)
    with pytest.raises(FileExistsError, match="already exists"):
        V.build_vault(snapshot, out, owner="user-1")


def test_a_failed_migration_leaves_neither_a_vault_nor_a_staging_file(
        snapshot, tmp_path, monkeypatch):
    """Better than resumable: there is nothing to resume from.

    The build goes to a sibling temp file and is moved into place only after it
    closes cleanly, so a crash leaves the destination absent rather than
    half-written. A half-written vault is the dangerous state — it opens, it
    answers queries, and the answers are short.
    """
    out = tmp_path / "user1.db"
    real_copy = V._copy_table

    def explode(source, target, table, **kw):
        if table == "records":
            raise RuntimeError("migration died mid-copy")
        return real_copy(source, target, table, **kw)

    monkeypatch.setattr(V, "_copy_table", explode)
    with pytest.raises(RuntimeError, match="died mid-copy"):
        V.build_vault(snapshot, out, owner="user-1")

    assert not out.exists(), "a failed migration left a vault behind"
    assert not list(tmp_path.glob(".*tmp")), "a staging file survived the failure"


# --------------------------------------------------------------------------- #
# what the migration must NOT do
# --------------------------------------------------------------------------- #
def test_the_migration_carries_the_double_count_forward_untouched(conn, vault_path,
                                                                 tmp_path):
    """A day written twice by two devices stays written twice.

    G-01: Apple resolves overlap at interval granularity with the winning source
    varying through the day — 2021-09-27 showed 10,173 against our stored 20,140,
    which is 2.77% above watch-only and 0.66% below phone-only. It matches no
    rule we could apply. So the import copies the defect forward and the
    one-instrument-class rule stays mandatory; inventing a merge here would
    replace a known-wrong number with an unknowably-wrong one.
    """
    day = "2021-09-27"
    for source, value in (("Demo's Apple Watch", 9899.0), ("Demo's iPhone", 10241.0)):
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) VALUES "
            "('step_count', ?, 'count', ?, ?, ?, ?, ?, 'backfill', ?)",
            (value, f"{day}T12:00:00+00:00", f"{day}T12:00:00+00:00",
             f"{day}T12:00:00", day, source, f"step|{source}|{day}"))
    conn.execute(
        "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, "
        "last, unit) VALUES ('step_count', ?, 2, 20140.0, 10070.0, 9899.0, "
        "10241.0, 10241.0, 'count')", (day,))
    conn.commit()
    conn.close()

    out = tmp_path / "user1.db"
    V.build_vault(vault_path, out, owner="user-1", measure_gzip=False)

    vconn = dbmod.connect(out, read_only=True)
    try:
        stored = vconn.execute(
            "SELECT sum FROM daily_metrics WHERE metric='step_count' AND date=?",
            (day,)).fetchone()["sum"]
    finally:
        vconn.close()

    assert stored == pytest.approx(20140.0), (
        "the migration silently merged a two-device day; G-01 established that "
        "the real answer matches no rule available to us"
    )


def test_backfill_is_reachable_only_as_a_migration_entry_point():
    """Nothing in the package may import it.

    A full re-run after the receiver has been live double-counts every day the
    receiver already replaced with fine samples. That makes it something a
    person invokes deliberately — never something a code path reaches on its
    own — and this is what keeps that true as the package grows.
    """
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "backfill.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[-1] == "backfill" for a in node.names):
                    offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if any(a.name == "backfill" for a in node.names) or \
                        (node.module or "").endswith("backfill"):
                    offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == [], (
        "backfill is a migration entry point; these import it at runtime: "
        + ", ".join(offenders)
    )


def test_a_rebuild_cannot_lower_the_fencing_epoch(snapshot, tmp_path):
    """A fresh SQLite file starts at user_version 0.

    Rebuilding a vault that had committed at epoch 5 would silently reset its
    fence, after which a worker still holding epoch 3 passes
    `landed >= lease.epoch` and overwrites the work that replaced it. The epoch
    may only move forward, including across a rebuild.
    """
    out = tmp_path / "user1.db"
    V.build_vault(snapshot, out, owner="user-1", measure_gzip=False)

    ctx = VaultContext.local(out, user_id="user-1", writable=True)
    ctx.set_version(5)
    assert ctx.current_version() == 5

    report = V.build_vault(snapshot, out, owner="user-1", replace=True,
                           measure_gzip=False)

    assert report["epoch"] == 5
    assert VaultContext.local(out, user_id="user-1").current_version() == 5, \
        "the rebuild reset the fence; a stale worker can now commit"


def test_provenance_is_rebuilt_not_merely_inherited(conn, vault_path, tmp_path):
    """A source whose `metric_source_months` was filled incrementally covers
    recent months only. Accepting it as complete because it is non-empty makes a
    historical instrument change disappear — F3-2 with no symptom."""
    _two_instruments(conn, "basal_energy")
    # A partial provenance table, exactly as an incremental receiver leaves it.
    dbmod.rebuild_metric_source_months(
        conn, pairs=[("basal_energy", "2022-12-01")])
    conn.commit()
    partial = conn.execute(
        "SELECT COUNT(*) FROM metric_source_months").fetchone()[0]
    conn.close()
    assert partial > 0, "the fixture must start non-empty or it tests nothing"

    out = tmp_path / "user1.db"
    V.build_vault(vault_path, out, owner="user-1", measure_gzip=False)

    vconn = dbmod.connect(out, read_only=True)
    try:
        months = vconn.execute(
            "SELECT COUNT(DISTINCT month) FROM metric_source_months "
            "WHERE metric='basal_energy'").fetchone()[0]
    finally:
        vconn.close()
    assert months == 12, f"only {months} months of provenance travelled"


def test_a_rebuild_keeps_the_commit_keys_already_applied(snapshot, tmp_path):
    """Forgetting an applied key means a retried commit re-applies it — a
    duplicate insight and a duplicate provider charge. The data is rebuilt from
    the source; the applied-key set is history the source cannot know."""
    out = tmp_path / "user1.db"
    V.build_vault(snapshot, out, owner="user-1", measure_gzip=False)
    conn = dbmod.connect(out)
    dbmod.log_ingest(conn, "t", "t", 0, 0, "")     # unrelated row, must not matter
    conn.execute("INSERT INTO commit_log (key, epoch, applied_at, detail) "
                 "VALUES ('brief:2026-08-01', 3, '2026-08-01T00:00:00Z', 'x')")
    conn.commit()
    conn.close()

    V.build_vault(snapshot, out, owner="user-1", replace=True, measure_gzip=False)

    conn = dbmod.connect(out, read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM commit_log WHERE key='brief:2026-08-01'"
        ).fetchone()[0] == 1, "the rebuild forgot a commit that had already landed"
    finally:
        conn.close()


def test_a_vault_says_what_shape_it_was_built_in(snapshot, tmp_path):
    """#11. Nothing reads this yet, and that is the point.

    A migration across every user's vault has to know what shape each one is in.
    A vault that never said cannot be asked — the answer would have to be
    inferred from its contents, per user, at the moment inference is least
    affordable.
    """
    out = tmp_path / "user1.db"
    V.build_vault(snapshot, out, owner="user-1", measure_gzip=False)

    conn = dbmod.connect(out, read_only=True)
    try:
        assert dbmod.vault_schema_version(conn) == dbmod.VAULT_SCHEMA_VERSION
        stamped = dict(conn.execute("SELECT key, value FROM vault_meta"))
    finally:
        conn.close()
    assert stamped["owner"] == "user-1"
    assert stamped["created_at"].startswith("20")


def test_init_db_migrates_a_preexisting_vault_before_advancing_its_stamp(snapshot):
    """An old declaration advances only after the legacy table is migrated."""
    conn = dbmod.connect(snapshot)
    try:
        # Model a pre-existing v1 vault with rows in several tables and the
        # v1 daily_metrics shape that lacks both additive columns.
        conn.execute("DROP TABLE daily_metrics")
        conn.execute(
            "CREATE TABLE daily_metrics ("
            "metric TEXT NOT NULL, date TEXT NOT NULL, count INTEGER NOT NULL, "
            "sum REAL, avg REAL, min REAL, max REAL, unit TEXT, "
            "PRIMARY KEY (metric, date))"
        )
        conn.execute(
            "INSERT INTO daily_metrics "
            "(metric, date, count, sum, avg, min, max, unit) "
            "VALUES ('synthetic_metric', '2026-08-01', 1, 1, 1, 1, 1, 'unit')"
        )
        conn.execute(
            "UPDATE vault_meta SET value = '1' WHERE key = 'schema_version'"
        )
        conn.commit()
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        )]
        before = {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }
        assert dbmod.vault_schema_version(conn) == 1
    finally:
        conn.close()

    conn = dbmod.connect(snapshot)
    try:
        dbmod.init_db(conn)
        assert dbmod.vault_schema_version(conn) == dbmod.VAULT_SCHEMA_VERSION
        columns = {row[1] for row in conn.execute("PRAGMA table_info(daily_metrics)")}
        assert {"last", "source_kind"} <= columns
        after = {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }
    finally:
        conn.close()

    assert after == before


def test_init_db_refuses_a_vault_newer_than_the_code(vault_path):
    conn = dbmod.connect(vault_path)
    dbmod.init_db(conn)
    conn.execute("UPDATE vault_meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    conn = dbmod.connect(vault_path)
    try:
        with pytest.raises(ValueError, match="newer than code version"):
            dbmod.init_db(conn)
        assert dbmod.vault_schema_version(conn) == 99
    finally:
        conn.close()


def test_init_db_does_not_stamp_a_read_only_vault(vault_path):
    conn = dbmod.connect(vault_path)
    dbmod.init_db(conn)
    conn.execute("UPDATE vault_meta SET value = '1' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    conn = dbmod.connect(vault_path, read_only=True)
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        dbmod.init_db(conn)
        assert dbmod.vault_schema_version(conn) == 1
        assert not any(
            "vault_meta" in sql.lower() and
            ("insert" in sql.lower() or "update" in sql.lower())
            for sql in statements
        )
    finally:
        conn.close()


def test_a_rebuild_keeps_the_owner_when_none_is_given(snapshot, tmp_path):
    """Owner is history the source cannot know — a snapshot has no idea whose
    vault it is about to become."""
    out = tmp_path / "user1.db"
    V.build_vault(snapshot, out, owner="user-1", measure_gzip=False)
    report = V.build_vault(snapshot, out, replace=True, measure_gzip=False)

    assert report["owner"] == "user-1"
    assert VaultContext.local(out, user_id="user-1").owner() == "user-1"


def test_a_rebuild_keeps_an_explicitly_moved_history_watermark(snapshot, tmp_path):
    out = tmp_path / "user1.db"
    V.build_vault(snapshot, out, owner="user-1", measure_gzip=False)

    conn = dbmod.connect(out)
    V.set_history_imported_through(conn, "2030-01-01")
    conn.commit()
    conn.close()

    report = V.build_vault(
        snapshot, out, owner="user-1", replace=True, measure_gzip=False
    )

    assert report["history_imported_through"] == "2030-01-01"
    conn = dbmod.connect(out, read_only=True)
    try:
        assert conn.execute(
            "SELECT value FROM vault_meta "
            "WHERE key = 'history_imported_through'"
        ).fetchone()[0] == "2030-01-01"
    finally:
        conn.close()
