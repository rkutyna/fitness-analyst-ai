"""The session's vault: whose it is, where it is, and what may be done to it.

Every module below an entry point takes one of these instead of resolving a
database path for itself. There is deliberately **no default path and no
environment fallback**: the product serves many users from one process, and an
ambient default is exactly how one session ends up reading another user's data.
A path enters the process once, at an entry point, from argv — everywhere else
it arrives as a `VaultContext` argument.

Two of the four fields have teeth today:

- ``capabilities`` gates writes. :meth:`connect` refuses to open the vault
  writable without :data:`WRITE`, so a read-only session cannot become a writer
  by calling the wrong helper. D5 needs this asymmetry visible rather than
  implied: a scheduled run gets a smaller surface than an interactive one.
- ``vault_version`` fences a resumed worker. When it is pinned, every
  ``connect`` checks SQLite's own ``PRAGMA user_version`` and refuses a vault
  that moved underneath the session. That is the primitive T-005's lease and
  commit protocol needs; it is here rather than there because the check has to
  live at the connection, which is the one thing every caller shares.

``user_id`` is carried for logging and for the isolation assertions; it is not
a query filter, and it must never become one. Isolation here is the file, not
a ``WHERE user_id =`` that one query can forget (D4).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

from . import db as _db
from . import vault as _vault

#: May read the vault. Every context has it; it exists so a capability set
#: reads as a complete statement rather than "write or nothing".
READ = "read"
#: May open the vault writable. Insights, subjective entries, food logs and
#: every ingest path require it.
WRITE = "write"
#: May receive values that identify a SINGLE stored sample — one row of
#: `records`, with its own timestamp. Aggregates over a window are not this,
#: even sub-daily ones: a 20-minute mean does not reveal a reading.
#:
#: Withheld from any session whose output reaches a model provider. That is the
#: whole mechanism behind §5's privacy claim, and it is a capability rather than
#: a convention because the claim is the product's central differentiator and
#: "we were careful" is not a boundary.
RAW_SAMPLES = "raw_samples"

READ_ONLY: frozenset[str] = frozenset({READ})
READ_WRITE: frozenset[str] = frozenset({READ, WRITE})
#: A session on this machine, working the local snapshot: everything, including
#: sample-level reads. Audits and the desktop review workflow live here.
LOCAL_FULL: frozenset[str] = frozenset({READ, WRITE, RAW_SAMPLES})


class CapabilityError(PermissionError):
    """A session tried to do something its capabilities do not permit."""


class VaultOwnershipError(PermissionError):
    """A session tried to open a vault belonging to a different user.

    Under D4 isolation is the file: a session is handed one vault's path and
    that is the whole boundary. This is what makes the boundary a mechanism
    rather than a hope — the vault says whose it is, and a mismatched session is
    refused at connect instead of quietly answering with someone else's data.
    """


class VaultVersionMismatch(RuntimeError):
    """The vault moved underneath a session that had pinned its version.

    Raised on connect, not on commit, so a fenced worker fails before it spends
    a provider call on data it is not allowed to write back.
    """


@dataclass(frozen=True)
class VaultContext:
    """One user's vault, for the duration of one session.

    Frozen because a session's identity must not change halfway through a tool
    loop. Widen or narrow it with :meth:`granting` / :meth:`revoking`, which
    return a new context.
    """

    user_id: str
    db_path: Path
    capabilities: frozenset[str] = READ_ONLY
    vault_version: int | None = None

    @property
    def vault_id(self) -> str:
        """Which vault this session is for, independent of where it sits.

        D4 is one vault per user, so this is the user id — and it is a property
        rather than a field precisely so the two cannot drift apart. A lease is
        taken on the vault_id, and `lease.commit` refuses a lease whose
        vault_id is not this one: without that check a valid lease can fence
        one vault while the write lands in another user's file.

        `db_path` is deliberately not the identity. A snapshot copy is the same
        vault at a different path, and two workers holding the same vault at
        different mount points must contend with each other.
        """
        return self.user_id

    def __post_init__(self) -> None:
        object.__setattr__(self, "db_path", Path(self.db_path))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))

    # ----------------------------------------------------------------- #
    # construction
    # ----------------------------------------------------------------- #
    @classmethod
    def local(cls, db_path: str | Path, *, user_id: str = "local",
              writable: bool = False) -> "VaultContext":
        """A context for a single-user checkout — CLI tools, scripts, tests.

        Named `local` rather than `default` on purpose: it is a statement about
        which vault, not a fallback for having failed to say.
        """
        return cls(
            user_id=user_id,
            db_path=Path(db_path),
            capabilities=(LOCAL_FULL if writable else READ_ONLY | {RAW_SAMPLES}),
        )

    def provider_facing(self) -> "VaultContext":
        """This session, narrowed for work whose output reaches a model provider.

        Drops :data:`RAW_SAMPLES` and :data:`WRITE`. The researcher path builds
        its tools from one of these, so a provider-facing session cannot receive
        a single stored reading by any route — not because each tool remembers
        to check, but because the capability is not there to check against.
        """
        return self.revoking(RAW_SAMPLES, WRITE)

    def granting(self, *caps: str) -> "VaultContext":
        return replace(self, capabilities=self.capabilities | frozenset(caps))

    def revoking(self, *caps: str) -> "VaultContext":
        return replace(self, capabilities=self.capabilities - frozenset(caps))

    def claim(self) -> None:
        """Record this context's user as the vault's owner.

        Called once when a vault is created for a user. Claiming a vault that
        another user already owns raises rather than overwriting: an ownership
        record that can be reassigned by whoever opens the file next is not a
        boundary.
        """
        self.require(WRITE)
        conn = _db.connect(self.db_path)
        try:
            _db.init_db(conn)
            existing = _owner_of(conn)
            if existing is not None and existing != self.user_id:
                raise VaultOwnershipError(
                    f"vault {self.db_path} is already owned by {existing!r}; "
                    f"{self.user_id!r} cannot claim it"
                )
            conn.execute(
                "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('owner', ?)",
                (self.user_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def owner(self) -> str | None:
        """The user this vault belongs to, or None.

        None covers both "never claimed" and "does not exist yet"; neither is a
        vault anyone owns, and distinguishing them here would only tempt a
        caller to treat a missing file as an error it then has to handle.
        """
        if not self.db_path.exists():
            return None
        conn = _db.connect(self.db_path, read_only=True)
        try:
            return _owner_of(conn)
        finally:
            conn.close()

    def settings(self) -> dict[str, object]:
        """This vault's declared settings (T-032), read from the vault itself.

        `{"local_timezone": str|None, "unit_system": str|None,
          "units": dict|None, "workout_source_arbitration_from": str|None,
          "block_qualify_hr_max": float|None}`. A value is None when the vault
        has not declared
        it -- there is no default applied here. Computation callers resolve
        their documented legacy or explicit setting. One
        process serves many vaults, so a module-level fallback would read as
        correct right up until the second vault declares something different,
        which is the failure T-003 exists to prevent.

        Undeclared is not an error: the snapshot and every test vault are
        undeclared, and historical samples keep being attributed by the
        per-sample offset carried in the export.
        """
        blank: dict[str, object] = {
            "local_timezone": None, "unit_system": None, "units": None,
            "workout_source_arbitration_from": None,
            "block_qualify_hr_max": None,
        }
        if not self.db_path.exists():
            return blank
        conn = _db.connect(self.db_path, read_only=True)
        try:
            return {"local_timezone": _vault.local_timezone(conn),
                    "unit_system": _vault.unit_system(conn),
                    "units": _vault.units(conn),
                    "workout_source_arbitration_from":
                        _vault.workout_source_arbitration_from(conn),
                    "block_qualify_hr_max": _vault.block_qualify_hr_max(conn)}
        finally:
            conn.close()

    def pinned(self) -> "VaultContext":
        """Stamp the vault's current ``user_version`` onto the context.

        Every later connect then verifies it. Call this once at session start,
        after the lease is taken.
        """
        return replace(self, vault_version=self.current_version())

    # ----------------------------------------------------------------- #
    # capabilities
    # ----------------------------------------------------------------- #
    def can(self, capability: str) -> bool:
        return capability in self.capabilities

    def require(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise CapabilityError(
                f"session for user {self.user_id!r} lacks the {capability!r} "
                f"capability (has: {sorted(self.capabilities) or 'none'})"
            )

    # ----------------------------------------------------------------- #
    # connections
    # ----------------------------------------------------------------- #
    def connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        """Open this session's vault. Writable connections require :data:`WRITE`."""
        if read_only:
            self.require(READ)
        else:
            self.require(WRITE)
        conn = _db.connect(self.db_path, read_only=read_only)
        owner = _owner_of(conn)
        if owner is not None and owner != self.user_id:
            conn.close()
            raise VaultOwnershipError(
                f"vault {self.db_path} belongs to {owner!r}; this session is "
                f"{self.user_id!r}. An unclaimed vault is allowed through — a "
                f"claimed one is not."
            )
        if self.vault_version is not None:
            actual = conn.execute("PRAGMA user_version").fetchone()[0]
            if actual != self.vault_version:
                conn.close()
                raise VaultVersionMismatch(
                    f"vault {self.db_path} is at user_version {actual}, but the "
                    f"session for user {self.user_id!r} is pinned to "
                    f"{self.vault_version}; the lease is stale"
                )
        return conn

    def read_only(self) -> sqlite3.Connection:
        """The common case. Every tool that only reports reads through this."""
        return self.connect(read_only=True)

    def current_version(self) -> int:
        """The vault's ``PRAGMA user_version`` right now, ignoring any pin."""
        conn = _db.connect(self.db_path, read_only=True)
        try:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])
        finally:
            conn.close()

    def set_version(self, version: int) -> None:
        """Advance the vault's version. Writers only; T-005 owns when."""
        self.require(WRITE)
        conn = _db.connect(self.db_path)
        try:
            conn.execute(f"PRAGMA user_version = {int(version)}")
            conn.commit()
        finally:
            conn.close()

    def __repr__(self) -> str:  # never print the path's parent tree in logs
        v = "unpinned" if self.vault_version is None else f"v{self.vault_version}"
        return (f"VaultContext(user={self.user_id!r}, vault={self.db_path.name!r}, "
                f"{v}, caps={','.join(sorted(self.capabilities))})")


def _owner_of(conn: sqlite3.Connection) -> str | None:
    """The vault's declared owner, or None.

    Tolerates a database with no schema yet: `connect()` runs before `init_db`
    on a brand-new vault, and a missing table means unclaimed, not broken.
    """
    try:
        row = conn.execute(
            "SELECT value FROM vault_meta WHERE key = 'owner'").fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None
