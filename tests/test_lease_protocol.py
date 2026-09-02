"""Lease, fencing and the commit phase (T-005).

The failures here are the ones a passing test suite is least likely to notice:
a lost update, a duplicate provider charge, an upload that waited forty minutes
behind a research run. None of them produces a wrong number — they produce a
number that is right about the wrong state of the world.

Every test below fixes its own clock. Expiry that depends on `sleep` is a test
that is either slow or flaky, and usually both.
"""
from __future__ import annotations

import sqlite3

import pytest

from health_advisor import db as dbmod
from health_advisor import lease as L
from health_advisor.context import VaultContext, WRITE
from tests.conftest import seed_metric

T0 = 1_000_000.0          # a fixed "now"


@pytest.fixture
def store(tmp_path):
    return L.LeaseStore(tmp_path / "control" / "leases.db")


@pytest.fixture
def seeded(vault):
    conn = vault.connect()
    dbmod.init_db(conn)
    seed_metric(conn, "body_mass", "2026-08-01", [188.8])
    conn.close()
    return vault


def _insights(ctx) -> list[str]:
    conn = ctx.connect(read_only=True)
    try:
        return [r["text"] for r in conn.execute(
            "SELECT text FROM insights ORDER BY id")]
    finally:
        conn.close()


def _write(text: str):
    def apply(conn):
        dbmod.write_insight(conn, "2026-08-01", text, "test")
        return text
    return apply


# --------------------------------------------------------------------------- #
# 1. a worker crash mid-session leaves nothing behind
# --------------------------------------------------------------------------- #
def test_a_crash_before_the_commit_phase_writes_nothing(seeded, store):
    """The reason reads do not take the lease and writes are one transaction:
    there is no partial state to reconcile, because there was never a partial
    write. The worker below dies holding the lease, having read everything and
    committed nothing."""
    dead = store.acquire(seeded.vault_id, "worker-a", ttl_seconds=60, now=T0)
    snapshot = L.open_snapshot(seeded, seeded.db_path.parent / "snap.db")
    assert snapshot.read_only().execute(
        "SELECT COUNT(*) FROM daily_metrics").fetchone()[0] == 1
    # …and then it dies. No commit call.

    assert _insights(seeded) == []

    # The lease lapses and the next worker takes it, one epoch higher.
    live = store.acquire(seeded.vault_id, "worker-b", ttl_seconds=60, now=T0 + 61)
    assert live.epoch == dead.epoch + 1
    L.commit(seeded, live, _write("from B"), key="b-1", store=store, now=T0 + 61)
    assert _insights(seeded) == ["from B"]


def test_a_failure_inside_the_commit_phase_leaves_no_partial_write(seeded, store):
    """The commit is one transaction, so a writer that succeeded before the
    failure must not survive it — and the key must stay unused, or the retry
    that is supposed to recover the write would be refused as a replay."""
    lease = store.acquire(seeded.vault_id, "worker-a", now=T0)

    def apply(conn):
        dbmod.write_insight(conn, "2026-08-01", "half-written", "test")
        raise RuntimeError("worker died here")

    with pytest.raises(RuntimeError, match="worker died here"):
        L.commit(seeded, lease, apply, key="a-1", store=store, now=T0)

    assert _insights(seeded) == [], "a partial write survived the failure"
    assert L.already_applied(seeded, "a-1") is None, \
        "the key was consumed by a commit that did not land"

    # And the retry that recovers it is not mistaken for a replay.
    lease2 = store.acquire(seeded.vault_id, "worker-a", now=T0 + 1)
    L.commit(seeded, lease2, _write("retried"), key="a-1", store=store, now=T0 + 1)
    assert _insights(seeded) == ["retried"]


def test_a_writer_that_commits_underneath_the_phase_is_caught(seeded, store):
    """`db.write_insight` was a `with conn:` block until this task, which ends
    the caller's transaction and makes every later failure a partial write.
    Standalone behaviour is identical, so nothing else could have caught it."""
    lease = store.acquire(seeded.vault_id, "worker-a", now=T0)

    def apply(conn):
        with conn:                       # the old shape, on purpose
            conn.execute(
                "INSERT INTO insights (date, text, tags, created_at) "
                "VALUES ('2026-08-01', 'x', 'test', '2026-08-01T00:00:00Z')")
        return "x"

    with pytest.raises(RuntimeError, match="ended the transaction"):
        L.commit(seeded, lease, apply, key="a-1", store=store, now=T0)


# --------------------------------------------------------------------------- #
# 2. a stale worker is fenced
# --------------------------------------------------------------------------- #
def test_a_resumed_stale_worker_cannot_commit(seeded, store):
    """Expiry without fencing turns a stuck lease into a lost update, which is
    strictly worse: the stuck lease is visible and the lost update is not."""
    stale = store.acquire(seeded.vault_id, "worker-a", ttl_seconds=60, now=T0)
    fresh = store.acquire(seeded.vault_id, "worker-b", ttl_seconds=60, now=T0 + 61)
    assert fresh.epoch > stale.epoch

    L.commit(seeded, fresh, _write("from B"), key="b-1", store=store, now=T0 + 61)

    # worker-a wakes up believing it still holds the vault.
    with pytest.raises(L.LeaseExpired):
        L.commit(seeded, stale, _write("from A"), key="a-1", store=store,
                 now=T0 + 62)

    assert _insights(seeded) == ["from B"], "the stale worker overwrote a newer commit"


def test_the_epoch_fences_even_when_the_lease_store_is_not_consulted(seeded, store):
    """The store is a courtesy check; the vault's own epoch is the fence.

    A worker partitioned from the lease store, or one whose clock says it still
    holds the lease, gets no say — the vault refuses to move backwards.
    """
    stale = store.acquire(seeded.vault_id, "worker-a", ttl_seconds=60, now=T0)
    fresh = store.acquire(seeded.vault_id, "worker-b", ttl_seconds=60, now=T0 + 61)
    L.commit(seeded, fresh, _write("from B"), key="b-1", store=store, now=T0 + 61)

    with pytest.raises(L.Fenced, match="must not overwrite"):
        L.commit(seeded, stale, _write("from A"), key="a-1")   # no store at all

    assert _insights(seeded) == ["from B"]


def test_an_unexpired_lease_cannot_be_stolen(store):
    store.acquire("v", "worker-a", ttl_seconds=60, now=T0)
    with pytest.raises(L.LeaseHeld, match="worker-a"):
        store.acquire("v", "worker-b", now=T0 + 30)


def test_renewing_keeps_the_epoch_but_taking_over_raises_it(store):
    first = store.acquire("v", "worker-a", ttl_seconds=60, now=T0)
    same = store.acquire("v", "worker-a", ttl_seconds=60, now=T0 + 30)
    assert same.epoch == first.epoch, "renewal must not fence the renewer"

    later = store.acquire("v", "worker-b", ttl_seconds=60, now=T0 + 120)
    assert later.epoch == first.epoch + 1


def test_releasing_does_not_lower_the_epoch(seeded, store):
    """A worker that released and then came back to life is fenced exactly like
    one that expired — releasing is not a way to keep your turn."""
    released = store.acquire(seeded.vault_id, "worker-a", now=T0)
    store.release(released)
    nxt = store.acquire(seeded.vault_id, "worker-b", now=T0 + 1)
    assert nxt.epoch == released.epoch + 1

    L.commit(seeded, nxt, _write("from B"), key="b-1", store=store, now=T0 + 1)
    with pytest.raises((L.Fenced, L.LeaseExpired)):
        L.commit(seeded, released, _write("from A"), key="a-1")


# --------------------------------------------------------------------------- #
# 3. a phone delta lands while a long read session is open
# --------------------------------------------------------------------------- #
def test_an_upload_commits_while_a_long_read_session_is_open(seeded, store):
    """The concrete failure D4 was underspecified about: `llm.py` permits a
    900 s loop deadline and 2400 s per tool turn, so a session-long lease would
    have made a phone upload wait fifteen to forty minutes. Reads take no lease
    and read their own copy, so there is nothing to wait for."""
    reader = L.open_snapshot(seeded, seeded.db_path.parent / "snap.db")
    assert store.current(seeded.vault_id) is None, "a read session must take no lease"

    def land_a_delta(conn):
        # Written inline rather than through `seed_metric`, which commits — the
        # commit-phase guard rejects that, correctly.
        conn.execute(
            "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, "
            "max, last, unit) VALUES ('body_mass', '2026-08-02', 1, 187.0, "
            "187.0, 187.0, 187.0, 187.0, 'lb')")
        return "delta"

    upload = store.acquire(seeded.vault_id, "phone", now=T0)
    L.commit(seeded, upload, land_a_delta, key="upload-1", store=store, now=T0)

    live = seeded.read_only()
    try:
        assert live.execute(
            "SELECT COUNT(*) FROM daily_metrics").fetchone()[0] == 2
    finally:
        live.close()

    # The reader still sees one instant, which is what makes its report
    # internally consistent rather than half-old and half-new.
    snap = reader.read_only()
    try:
        assert snap.execute(
            "SELECT COUNT(*) FROM daily_metrics").fetchone()[0] == 1
    finally:
        snap.close()


def test_a_snapshot_cannot_be_written_to(seeded):
    """A snapshot that can be written to is a snapshot somebody will write to,
    and its writes go nowhere anyone will ever read."""
    reader = L.open_snapshot(seeded, seeded.db_path.parent / "snap.db")
    assert not reader.can(WRITE)
    with pytest.raises(Exception):
        reader.connect()


# --------------------------------------------------------------------------- #
# 4. replay applies once
# --------------------------------------------------------------------------- #
def test_replaying_a_commit_produces_one_insight_not_two(seeded, store):
    """Retrying is the recovery mechanism for a chunked write, so it has to be
    free. 'Idempotent' here means more than 'the same rows end up there': the
    second run also spends a provider call, so it must not run at all."""
    ran = []

    def apply(conn):
        ran.append(1)
        dbmod.write_insight(conn, "2026-08-01", "the brief", "test")
        return "brief"

    first = L.commit(seeded, store.acquire(seeded.vault_id, "w", now=T0), apply,
                     key="brief-2026-08-01", store=store, now=T0)
    second = L.commit(seeded, store.acquire(seeded.vault_id, "w", now=T0 + 120), apply,
                      key="brief-2026-08-01", store=store, now=T0 + 120)

    assert first["applied"] is True
    assert second["applied"] is False and second["reason"] == "already_applied"
    assert ran == [1], "the replay re-ran the work, which is where the charge is"
    assert _insights(seeded) == ["the brief"]


def test_two_workers_racing_the_same_key_apply_it_once(seeded, store):
    """The replay check is not under the same lock as the insert, so the
    PRIMARY KEY on commit_log is what actually decides the race."""
    a = store.acquire(seeded.vault_id, "worker-a", now=T0)
    L.commit(seeded, a, _write("from A"), key="same-key", store=store, now=T0)

    b = store.acquire(seeded.vault_id, "worker-b", now=T0 + 120)
    out = L.commit(seeded, b, _write("from B"), key="same-key", store=store,
                   now=T0 + 120)

    assert out["applied"] is False
    assert _insights(seeded) == ["from A"]


def test_a_different_key_still_applies(seeded, store):
    L.commit(seeded, store.acquire(seeded.vault_id, "w", now=T0), _write("one"),
             key="k1", store=store, now=T0)
    L.commit(seeded, store.acquire(seeded.vault_id, "w", now=T0 + 120), _write("two"),
             key="k2", store=store, now=T0 + 120)
    assert _insights(seeded) == ["two"], "same (date, tags) — the upsert rule still holds"
    conn = seeded.connect(read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM commit_log").fetchone()[0] == 2
    finally:
        conn.close()


def test_the_key_is_derived_from_the_work_not_the_attempt():
    """A key that varies per attempt makes every retry a new commit, which is
    the duplicate insight and the duplicate provider charge."""
    assert L.commit_key("morning", "2026-08-01") == \
        L.commit_key("morning", "2026-08-01")
    assert L.commit_key("morning", "2026-08-01") != \
        L.commit_key("evening", "2026-08-01")
    assert L.commit_key("morning", "2026-08-01") != \
        L.commit_key("morning", "2026-08-02")


def test_a_lease_for_another_vault_is_refused(seeded, store, tmp_path):
    """The only failure here that produces a *correct* write in the wrong place.

    The lease store fences correctly, the epoch check passes, the transaction
    commits — into somebody else's file. Nothing downstream catches it, which is
    why the check is the first thing `commit` does.
    """
    other = store.acquire("someone-else", "worker-a", now=T0)

    with pytest.raises(L.WrongVault, match="someone-else"):
        L.commit(seeded, other, _write("wrong vault"), key="k1", store=store,
                 now=T0)

    assert _insights(seeded) == []


def test_acquire_for_cannot_produce_a_mismatched_lease(seeded, store):
    lease = store.acquire_for(seeded, "worker-a", now=T0)
    assert lease.vault_id == seeded.vault_id
    L.commit(seeded, lease, _write("fine"), key="k1", store=store, now=T0)
    assert _insights(seeded) == ["fine"]


def test_the_snapshot_is_a_consistent_copy_not_a_file_copy(seeded, tmp_path):
    """`shutil.copyfile` reads pages while a writer may be committing to them,
    so it can produce a file holding pages from two transactions — inconsistent
    in a way nothing later can detect. The backup API takes a read lock.

    Pinned by behaviour rather than by naming the implementation: the snapshot
    must open cleanly and pass an integrity check.
    """
    snapshot = L.open_snapshot(seeded, tmp_path / "snap" / "s.db")
    conn = snapshot.read_only()
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute(
            "SELECT COUNT(*) FROM daily_metrics").fetchone()[0] == 1
    finally:
        conn.close()
    assert not (tmp_path / "snap" / "s.db.partial").exists(), \
        "the staging file survived; a reader could open a half-written snapshot"


def test_a_full_recompute_can_run_inside_a_commit(seeded, store):
    """`recompute_daily_metrics(full=True)` is the heaviest writer there is, and
    the migration path will want it inside a commit.

    It calls `rebuild_metric_source_months` and `log_ingest`, both of which used
    to commit on their own — so the derived tables became durable while the
    commit log and the epoch stamp were still pending. `lease.commit` would then
    detect the ended transaction and raise, having already let a partial write
    land. Both are transaction-neutral now, so the whole thing is one commit.
    """
    lease = store.acquire_for(seeded, "worker-a", now=T0)

    def rebuild(conn):
        dbmod.recompute_daily_metrics(conn, full=True)
        return "rebuilt"

    out = L.commit(seeded, lease, rebuild, key="rebuild-1", store=store, now=T0)

    assert out["applied"] is True
    conn = seeded.connect(read_only=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == lease.epoch
        assert conn.execute(
            "SELECT COUNT(*) FROM commit_log WHERE key='rebuild-1'").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_failed_full_recompute_leaves_nothing_behind(seeded, store):
    """The other half: if it is one transaction, a later failure must roll back
    the derived tables too."""
    before = seeded.read_only()
    try:
        rows = before.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()[0]
    finally:
        before.close()

    lease = store.acquire_for(seeded, "worker-a", now=T0)

    def rebuild_then_die(conn):
        dbmod.recompute_daily_metrics(conn, full=True)
        raise RuntimeError("died after the rebuild")

    with pytest.raises(RuntimeError, match="died after the rebuild"):
        L.commit(seeded, lease, rebuild_then_die, key="r-1", store=store, now=T0)

    conn = seeded.connect(read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM daily_metrics").fetchone()[0] == rows
        assert conn.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0] == 0, \
            "the rebuild's log row outlived the transaction it belonged to"
    finally:
        conn.close()
