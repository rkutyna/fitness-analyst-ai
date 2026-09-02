"""Leases, fencing epochs, and the commit phase for a per-user vault (T-005).

D4 leaves one worker holding a vault for the length of a session. That is unsafe
and the adversarial review was right about why: `llm.py` permits a 900 s loop
deadline and 2400 s per tool turn, so a scheduled analyst run would block a
phone upload for fifteen to forty minutes; a crashed worker strands the lease
with no expiry; and "re-run against the newer vault" is not idempotent, because
a coach session writes insights and spends provider calls.

The protocol here is the answer to those three, and only those three:

**Reads never take the lease.** A session reads from an immutable snapshot —
`open_snapshot` copies the vault once, and everything after that reads a file
nobody will write to. So a long research run and a phone upload do not contend
at all, rather than contending politely.

**The lease is held only for the commit phase**, which is one transaction. A
worker that dies between reading and committing has written nothing; its lease
expires; the next holder takes it. There is no partial state to reconcile
because there was never a partial write.

**Expiry alone would be a lost update, so the epoch fences.** Every acquisition
increments a counter. The winner stamps its epoch into the vault on commit. A
worker that comes back from the dead holding epoch 3, against a vault that has
since been committed at epoch 4, is refused — it does not get to overwrite what
replaced it. Expiry without fencing converts a stuck lease into silent data
loss, which is worse than the stuck lease.

**Every commit carries an idempotency key.** Replay is a no-op, not a second
insight and not a second provider charge.

*On the store.* The lease lives in a small SQLite file beside the vault, using
`BEGIN IMMEDIATE` for compare-and-set. In the deployed system it becomes a
conditional write against object storage (an `If-Match` on the object's ETag) or
a row in a control database. The protocol does not change with the store — what
it needs is an atomic read-modify-write and a clock, and both of those are
assumptions worth keeping visible.

*On the clock.* Expiry is wall-clock and therefore only as good as clock skew
between workers. The epoch is what makes that safe: a worker that wrongly
believes it still holds the lease is fenced at commit rather than trusted.
"""
from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from . import db as _db
from .context import WRITE

#: How long an acquisition is good for. Deliberately short: the lease covers a
#: commit, not a session, and a worker that needs longer should renew rather
#: than be granted a window that hides its own death.
DEFAULT_TTL_SECONDS = 60.0


class LeaseHeld(RuntimeError):
    """Somebody else holds an unexpired lease on this vault."""


class LeaseExpired(RuntimeError):
    """This worker's lease lapsed before it committed."""


class WrongVault(RuntimeError):
    """A lease was presented for a commit into a different vault.

    Nothing else catches this. The lease store fences correctly, the epoch
    check passes, the transaction commits — into the wrong user's file. It is
    the only failure here that produces a *correct* write in the wrong place.
    """


class Fenced(RuntimeError):
    """A newer epoch has already committed; this worker must not overwrite it.

    Raised in preference to letting a resumed worker win. The lost work is
    recoverable by re-running against the newer vault; the overwritten commit
    is not.
    """


@dataclass(frozen=True)
class Lease:
    vault_id: str
    holder: str
    epoch: int
    expires_at: float

    def expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires_at


_SCHEMA = """
CREATE TABLE IF NOT EXISTS leases (
    vault_id   TEXT PRIMARY KEY,
    holder     TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    expires_at REAL NOT NULL
);
"""


class LeaseStore:
    """Compare-and-set over one SQLite file. See the module docstring on stores."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    # ------------------------------------------------------------------ #
    def current(self, vault_id: str) -> Lease | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM leases WHERE vault_id = ?", (vault_id,)).fetchone()
        finally:
            conn.close()
        return _row_to_lease(row) if row else None

    def acquire(self, vault_id: str, holder: str, *,
                ttl_seconds: float = DEFAULT_TTL_SECONDS,
                now: float | None = None) -> Lease:
        """Take the lease, or raise :class:`LeaseHeld`.

        A new acquisition always increments the epoch — including a steal from
        an expired holder, which is the case the increment exists for. Renewing
        an unexpired lease you already hold keeps your epoch, because nothing
        has come between you and the vault.
        """
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM leases WHERE vault_id = ?", (vault_id,)).fetchone()
            if row is not None:
                held = _row_to_lease(row)
                if not held.expired(now) and held.holder != holder:
                    conn.execute("ROLLBACK")
                    raise LeaseHeld(
                        f"vault {vault_id!r} is held by {held.holder!r} at epoch "
                        f"{held.epoch} for another {held.expires_at - now:.1f}s"
                    )
                epoch = held.epoch if (held.holder == holder and not held.expired(now)) \
                    else held.epoch + 1
            else:
                epoch = 1
            lease = Lease(vault_id, holder, epoch, now + ttl_seconds)
            conn.execute(
                "INSERT INTO leases (vault_id, holder, epoch, expires_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(vault_id) DO UPDATE SET "
                "holder = excluded.holder, epoch = excluded.epoch, "
                "expires_at = excluded.expires_at",
                (vault_id, holder, epoch, lease.expires_at))
            conn.execute("COMMIT")
            return lease
        finally:
            conn.close()

    def acquire_for(self, ctx, holder: str, *,
                    ttl_seconds: float = DEFAULT_TTL_SECONDS,
                    now: float | None = None) -> Lease:
        """Take the lease on a context's vault. The ergonomic form — it cannot
        produce a lease that `commit` will then reject as the wrong vault."""
        return self.acquire(ctx.vault_id, holder, ttl_seconds=ttl_seconds, now=now)

    def renew(self, lease: Lease, *, ttl_seconds: float = DEFAULT_TTL_SECONDS,
              now: float | None = None) -> Lease:
        """Extend a lease you still hold. Raises if it already lapsed and was
        taken — a renew is not a way back in."""
        now = time.time() if now is None else now
        held = self.current(lease.vault_id)
        if held is None or held.epoch != lease.epoch or held.holder != lease.holder:
            raise LeaseExpired(
                f"lease on {lease.vault_id!r} at epoch {lease.epoch} is gone "
                f"(now {held.holder!r} at epoch {held.epoch})" if held else
                f"lease on {lease.vault_id!r} at epoch {lease.epoch} is gone"
            )
        if held.expired(now):
            raise LeaseExpired(
                f"lease on {lease.vault_id!r} at epoch {lease.epoch} expired "
                f"{now - held.expires_at:.1f}s ago"
            )
        return self.acquire(lease.vault_id, lease.holder,
                            ttl_seconds=ttl_seconds, now=now)

    def release(self, lease: Lease) -> None:
        """Give the lease up early. Releasing does NOT lower the epoch: the next
        holder still gets a higher one, so a worker that released and then came
        back to life is fenced exactly like one that expired."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE leases SET expires_at = 0 WHERE vault_id = ? AND epoch = ?",
                (lease.vault_id, lease.epoch))
            conn.execute("COMMIT")
        finally:
            conn.close()


def _row_to_lease(row: sqlite3.Row) -> Lease:
    return Lease(row["vault_id"], row["holder"], row["epoch"], row["expires_at"])


def commit_key(kind: str, *parts: str) -> str:
    """The idempotency key for a commit, built from what makes it unique.

    A key convention has to exist somewhere, and "somewhere" turns into each
    call site inventing one — at which point a retry of the morning brief looks
    like a different commit and writes a second insight. Keys are derived from
    the work, never from the attempt: `commit_key("morning", "2026-08-01")` is
    the same on every retry, which is the entire point.

    Deliberately not including the worker or a timestamp. If two workers both
    decide to write today's morning brief, one of them should lose.
    """
    return ":".join((kind, *parts))


def worker_id() -> str:
    """A fresh identity per worker. Two workers must never share one."""
    return f"worker-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Reads: an immutable snapshot, taken without the lease
# --------------------------------------------------------------------------- #
def open_snapshot(ctx, snapshot_path: str | Path):
    """Copy the vault to ``snapshot_path`` and return a context reading it.

    The copy is the point. A session that reads the live file either blocks the
    phone's upload or sees it land halfway through an analysis; a session that
    reads its own copy does neither, and the figures it reports are all as of one
    instant, which is what makes a report internally consistent.

    Cost is one file copy per session, against a vault measured at 496 MB. That
    is the price of a read path that never contends, and it is paid by the
    worker that already had to decrypt the thing.
    """
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    staging = snapshot_path.with_name(snapshot_path.name + ".partial")
    staging.unlink(missing_ok=True)

    # SQLite's online backup API, not a file copy. `shutil.copyfile` reads pages
    # while a writer may be committing to them, so it can produce a file with
    # pages from two different transactions — a snapshot that is internally
    # inconsistent in a way nothing later can detect. The backup API takes a
    # read lock and gives a transactionally consistent copy.
    source = _db.connect(ctx.db_path, read_only=True)
    try:
        destination = sqlite3.connect(staging)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    # And publish atomically, so a reader never opens a half-written snapshot.
    os.replace(staging, snapshot_path)
    # Same user, same version pin, but explicitly not writable: a snapshot that
    # can be written to is a snapshot somebody will write to.
    return replace(ctx, db_path=snapshot_path,
                   capabilities=ctx.capabilities - {WRITE})


# --------------------------------------------------------------------------- #
# Writes: a short commit phase, fenced by epoch, keyed for replay
# --------------------------------------------------------------------------- #
def already_applied(ctx, key: str) -> dict | None:
    """The commit_log row for ``key``, if this commit has already landed.

    A vault that does not exist yet, or has no schema, has applied nothing —
    both are "no", not an error a caller has to distinguish.
    """
    if not Path(ctx.db_path).exists():
        return None
    conn = None
    try:
        conn = ctx.connect(read_only=True)
        row = conn.execute(
            "SELECT key, epoch, applied_at, detail FROM commit_log WHERE key = ?",
            (key,)).fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None
    finally:
        if conn is not None:
            conn.close()


def commit(ctx, lease: Lease, apply: Callable[[sqlite3.Connection], str | None],
           *, key: str, store: LeaseStore | None = None,
           now: float | None = None) -> dict:
    """Run ``apply`` against the vault inside one fenced, idempotent transaction.

    ``apply`` receives a writable connection and returns an optional detail
    string for the log. It must not commit or close the connection.

    Order matters and is not arbitrary:

    1. **Replay check first**, before the lease and before any work. A retry of
       a commit that already landed must cost nothing — that is what makes retry
       a safe recovery mechanism for the chunked writes.
    2. **Lease still ours and unexpired**, or :class:`LeaseExpired`. Checked
       against the store, not against the local object, because the local object
       is exactly what a stale worker is wrong about.
    3. **Epoch not overtaken**, or :class:`Fenced`. The vault's `user_version`
       is the epoch of the last commit that landed; a worker may only move it
       forward.
    4. Apply, log, stamp the epoch, commit — all one transaction, so a failure
       anywhere leaves the vault exactly as it was and the key unused.
    """
    now = time.time() if now is None else now

    # Before anything else: is this lease even for this vault? A lease is a
    # claim on a vault, and a claim on a different one is not weaker evidence,
    # it is evidence about something else.
    if lease.vault_id != ctx.vault_id:
        raise WrongVault(
            f"lease is for vault {lease.vault_id!r} but this session is "
            f"{ctx.vault_id!r} at {ctx.db_path}"
        )

    if (prior := already_applied(ctx, key)) is not None:
        return {"applied": False, "reason": "already_applied", **prior}

    if store is not None:
        held = store.current(lease.vault_id)
        if held is None or held.holder != lease.holder or held.epoch != lease.epoch:
            raise LeaseExpired(
                f"lease on {lease.vault_id!r} at epoch {lease.epoch} is no longer "
                f"ours" + (f" ({held.holder!r} holds epoch {held.epoch})"
                           if held else "")
            )
        if held.expired(now):
            raise LeaseExpired(
                f"lease on {lease.vault_id!r} expired {now - held.expires_at:.1f}s "
                f"before the commit phase began"
            )

    conn = ctx.connect()
    try:
        _db.init_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        landed = conn.execute("PRAGMA user_version").fetchone()[0]
        if landed >= lease.epoch:
            conn.execute("ROLLBACK")
            raise Fenced(
                f"vault {ctx.db_path} is at epoch {landed}; this worker holds "
                f"epoch {lease.epoch} and must not overwrite a newer commit"
            )
        # A duplicate key raises IntegrityError inside the transaction, which
        # rolls the whole thing back — so two workers racing the same key cannot
        # both apply, even though the check above is not under the same lock.
        detail = apply(conn)
        # A writer that commits underneath us has already made its write
        # durable, and everything after this point could still fail — which is
        # the partial write this whole phase exists to prevent. `db.write_insight`
        # was exactly this bug (a `with conn:` block) until T-005 found it. Fail
        # loudly rather than let the next one be silent.
        if not conn.in_transaction:
            raise RuntimeError(
                f"commit {key!r}: apply() ended the transaction — a writer it "
                f"called committed or rolled back on its own. Writers used "
                f"inside a commit must leave transaction control to the commit."
            )
        conn.execute(
            "INSERT INTO commit_log (key, epoch, applied_at, detail) "
            "VALUES (?, ?, ?, ?)",
            (key, lease.epoch, _db.utcnow_iso(), detail))
        conn.execute(f"PRAGMA user_version = {int(lease.epoch)}")
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()

    if store is not None:
        store.release(lease)
    return {"applied": True, "key": key, "epoch": lease.epoch, "detail": detail}
