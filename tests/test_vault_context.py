"""The per-session vault context (T-003).

The defect these pin down is not a wrong number — it is one user's session
reading another user's data because a path was resolved from somewhere ambient.
That failure is silent and total, so it gets its own tests rather than being
implied by the tools' own.
"""
from __future__ import annotations

import inspect

import pytest

from health_advisor import db as dbmod
from health_advisor import mcp_server as S
from health_advisor import vault as vaultmod
from health_advisor.context import (READ, READ_WRITE, WRITE, CapabilityError,
                                    VaultContext, VaultOwnershipError,
                                    VaultVersionMismatch)
from tests.conftest import PROD_DB_PATH, seed_metric


# --------------------------------------------------------------------------- #
# there is no ambient path left to fall back to
# --------------------------------------------------------------------------- #
def test_db_has_no_default_path():
    """`connect()` with no argument used to open the production database. The
    point of T-003 is that omitting the vault is now a TypeError, not a
    silently-correct call against whatever the process last had lying around."""
    assert not hasattr(dbmod, "DEFAULT_DB_PATH")
    with pytest.raises(TypeError):
        dbmod.connect()


def test_the_test_isolation_guard_rejects_the_production_path():
    """The migration suite must still fail loudly on an accidental live-DB open."""
    with pytest.raises(AssertionError, match="production DB"):
        dbmod.connect(PROD_DB_PATH)


def test_the_mcp_server_is_not_a_module_global():
    """A module-level FastMCP registers every tool at import against whatever
    path was in the environment, which is what made one process serving two
    users impossible."""
    assert not hasattr(S, "mcp")
    assert not hasattr(S, "DB_PATH")


# --------------------------------------------------------------------------- #
# two sessions, one process
# --------------------------------------------------------------------------- #
def test_two_sessions_in_one_process_each_read_their_own_vault(tmp_path):
    """The criterion T-003 exists for. Same process, same tool name, two
    vaults, two different answers."""
    figures = {"alice": 188.8, "bob": 143.2}
    sessions = {}
    for user, weight in figures.items():
        path = tmp_path / user / "health.db"
        ctx = VaultContext.local(path, user_id=user, writable=True)
        conn = ctx.connect()
        dbmod.init_db(conn)
        seed_metric(conn, "body_mass", "2026-08-01", [weight])
        conn.close()
        sessions[user] = S.build_tools(ctx)

    for user, weight in figures.items():
        got = sessions[user]["get_latest"]("body_mass")
        assert got["latest_day"]["value"] == pytest.approx(weight), user

    # And they are genuinely distinct callables over distinct vaults, not one
    # object that happened to be re-pointed between the two calls.
    assert sessions["alice"]["get_latest"] is not sessions["bob"]["get_latest"]


def _claimed(tmp_path, user: str, weight: float) -> VaultContext:
    """A vault that belongs to `user` and says so."""
    ctx = VaultContext.local(tmp_path / user / "health.db",
                             user_id=user, writable=True)
    ctx.claim()
    conn = ctx.connect()
    dbmod.init_db(conn)
    seed_metric(conn, "body_mass", "2026-08-01", [weight])
    conn.close()
    return ctx


def test_a_session_cannot_open_another_users_vault(tmp_path):
    """The guard, and the reason it has to exist.

    Under D4 isolation is the file, so the only thing standing between Alice's
    session and Bob's data was "the right path gets passed". That is a hope, not
    a mechanism: a worker that reuses a context, or a lease service that hands
    back the wrong row, produces a session that answers confidently with someone
    else's body weight. A claimed vault refuses at connect instead.

    Removing the ownership check in `VaultContext.connect` turns this red.
    """
    alice = _claimed(tmp_path, "alice", 188.8)
    bob = _claimed(tmp_path, "bob", 143.2)
    assert bob.owner() == "bob"

    # Alice's identity, Bob's file — the shape of every way this goes wrong.
    trespass = VaultContext(user_id="alice", db_path=bob.db_path,
                            capabilities=alice.capabilities)

    with pytest.raises(VaultOwnershipError, match="belongs to 'bob'"):
        trespass.read_only()
    with pytest.raises(VaultOwnershipError, match="belongs to 'bob'"):
        trespass.connect()

    # And it is refused before any tool can answer, not after.
    with pytest.raises(VaultOwnershipError):
        S.build_tools(trespass)["get_latest"]("body_mass")

    # Bob's own session is unaffected.
    assert S.build_tools(bob)["get_latest"]("body_mass")["latest_day"]["value"] \
        == pytest.approx(143.2)


def test_claiming_a_vault_someone_else_owns_is_refused(tmp_path):
    """An ownership record that the next opener can overwrite is not a boundary."""
    bob = _claimed(tmp_path, "bob", 143.2)
    thief = VaultContext(user_id="alice", db_path=bob.db_path,
                         capabilities=READ_WRITE)

    with pytest.raises(VaultOwnershipError, match="already owned by 'bob'"):
        thief.claim()

    assert bob.owner() == "bob"


def test_an_unclaimed_vault_is_allowed_through(vault_path):
    """The limit of the guarantee, stated rather than implied.

    The development snapshot and every test database are unclaimed, so the
    promise is "a claimed vault cannot be opened by the wrong session", not
    "every open is authorised". A guard whose scope is not written down is a
    guard people assume covers more than it does.
    """
    ctx = VaultContext.local(vault_path, user_id="whoever", writable=True)
    assert ctx.owner() is None
    ctx.connect().close()


def test_vault_context_opens_pre_79_conversation_turns(tmp_path):
    """Opening a pre-#79 vault adds the turn columns before schema.sql runs."""
    path = tmp_path / "pre-79.db"
    conn = dbmod.connect(path)
    conn.execute(
        "CREATE TABLE conversation_turns ("
        "id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, "
        "sequence INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, "
        "created_at TEXT NOT NULL, supersedes_turn_id TEXT)"
    )
    conn.execute(
        "INSERT INTO conversation_turns "
        "(id, conversation_id, sequence, role, content, created_at, "
        "supersedes_turn_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("turn-1", "conversation-1", 1, "user", "legacy", "2026-08-25", None),
    )
    conn.commit()
    conn.close()

    ctx = VaultContext.local(path, user_id="legacy", writable=True)
    conn = ctx.connect()
    try:
        dbmod.init_db(conn)
        columns = {row["name"] for row in conn.execute(
            "PRAGMA table_info(conversation_turns)"
        )}
        turn = conn.execute(
            "SELECT content FROM conversation_turns WHERE id = 'turn-1'"
        ).fetchone()
    finally:
        conn.close()

    assert {"answers_turn_id", "client_disconnected_at"} <= columns
    assert turn["content"] == "legacy"


def test_a_built_vault_can_be_stamped_with_its_owner(conn, vault_path, tmp_path):
    """Build time is where a per-user vault gets its identity."""
    seed_metric(conn, "body_mass", "2026-08-01", [188.8])
    conn.close()

    out = tmp_path / "alice.db"
    report = vaultmod.build_vault(vault_path, out, owner="alice")

    assert report["owner"] == "alice"
    assert VaultContext.local(out, user_id="alice").owner() == "alice"
    with pytest.raises(VaultOwnershipError):
        VaultContext.local(out, user_id="mallory").read_only()


def test_binding_hides_the_vault_from_the_tool_schema():
    """The model calls `get_latest(metric)`. If `ctx` reached the schema the
    model could name a vault, which is the same defect from the other side."""
    ctx = VaultContext.local("/nonexistent/health.db")
    bound = S.build_tools(ctx)["get_latest"]
    assert list(inspect.signature(bound).parameters) == ["metric"]
    # …while the real tool is still reachable for anything that introspects.
    assert inspect.unwrap(bound) is S.get_latest


def test_the_researcher_gets_a_strict_subset(vault):
    """D5: a scheduled run gets a smaller surface than an interactive one, and
    the narrowing happens in the factory rather than by convention."""
    from health_advisor import llm

    full = set(S.build_tools(vault))
    researcher = {t.name for t in
                  S.build_server(vault, include=frozenset(llm.RESEARCHER_TOOLS))
                  ._tool_manager.list_tools()}
    assert researcher == set(llm.RESEARCHER_TOOLS)
    assert researcher < full
    assert "write_insight" not in researcher


# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #
def test_a_read_only_session_cannot_open_the_vault_writable(vault_path):
    ctx = VaultContext.local(vault_path)          # not writable
    assert ctx.can(READ) and not ctx.can(WRITE)
    ctx.granting(WRITE).connect().close()          # seed it so :ro can open
    ctx.read_only().close()                        # reading is fine
    with pytest.raises(CapabilityError, match="write"):
        ctx.connect()


def test_a_read_only_session_cannot_write_an_insight(vault_path):
    """The capability is checked at the connection, so every writer inherits it
    rather than each one having to remember."""
    writable = VaultContext.local(vault_path, writable=True)
    conn = writable.connect()
    dbmod.init_db(conn)
    conn.close()

    with pytest.raises(CapabilityError):
        dbmod.write_insight_ctx(writable.revoking(WRITE), "2026-08-01", "x", "t")


# --------------------------------------------------------------------------- #
# version fencing (the primitive T-005's lease protocol needs)
# --------------------------------------------------------------------------- #
def test_a_pinned_session_refuses_a_vault_that_moved(vault_path):
    """A resumed worker must fail before it spends a provider call, not after
    it writes. So the check is on connect, not on commit."""
    ctx = VaultContext.local(vault_path, writable=True)
    ctx.connect().close()
    pinned = ctx.pinned()
    pinned.read_only().close()                     # still at the pinned version

    ctx.set_version(pinned.vault_version + 1)      # something else committed
    with pytest.raises(VaultVersionMismatch, match="stale"):
        pinned.read_only()
    ctx.read_only().close()                        # an unpinned session is fine


def test_an_unpinned_context_does_not_read_the_version(vault_path):
    """Pinning is opt-in: nothing pays for a PRAGMA it did not ask for."""
    ctx = VaultContext.local(vault_path, writable=True)
    assert ctx.vault_version is None
    ctx.connect().close()
    assert ctx.pinned().vault_version == 0


# --------------------------------------------------------------------------- #
# per-vault settings (T-032)
# --------------------------------------------------------------------------- #
def _vault_declaring(tmp_path, user, *, zone=None, units=None):
    ctx = VaultContext.local(tmp_path / user / "health.db",
                             user_id=user, writable=True)
    conn = ctx.connect()
    dbmod.init_db(conn)
    if zone is not None:
        vaultmod.set_local_timezone(conn, zone)
    if units is not None:
        vaultmod.set_unit_system(conn, units)
    conn.commit()
    conn.close()
    return ctx


def test_settings_are_per_vault_not_global(tmp_path):
    """The criterion T-032 exists for: one process, two vaults, two answers.

    A module-level default would pass every single-vault test and fail here,
    which is the whole reason this one is written with two.
    """
    alice = _vault_declaring(tmp_path, "alice",
                             zone="America/New_York", units="imperial")
    bob = _vault_declaring(tmp_path, "bob",
                           zone="Europe/Berlin", units="metric")

    a, b = alice.settings(), bob.settings()
    assert a["local_timezone"] == "America/New_York"
    assert b["local_timezone"] == "Europe/Berlin"
    assert a["units"]["distance"] == "mi" and a["units"]["mass"] == "lb"
    assert b["units"]["distance"] == "km" and b["units"]["mass"] == "kg"
    # Reading one must not have taught the other anything.
    assert alice.settings()["local_timezone"] == "America/New_York"


def test_an_undeclared_vault_reads_none_and_is_not_defaulted(tmp_path):
    """Undeclared is a legitimate state -- the snapshot and every test vault
    are undeclared. A convenience default here is the T-003 defect: it reads
    as correct until the second vault declares something different.
    """
    ctx = _vault_declaring(tmp_path, "carol")
    assert ctx.settings() == {
        "local_timezone": None, "unit_system": None, "units": None,
        "workout_source_arbitration_from": None, "block_qualify_hr_max": None,
    }


def test_a_typo_cannot_be_stored_as_a_timezone(tmp_path):
    """A stored typo would not fail here; it would mis-attribute a date later,
    somewhere with no connection to this call."""
    ctx = _vault_declaring(tmp_path, "dave")
    conn = ctx.connect()
    with pytest.raises(ValueError, match="unknown IANA time zone"):
        vaultmod.set_local_timezone(conn, "America/New_Yrok")
    with pytest.raises(ValueError, match="unknown unit system"):
        vaultmod.set_unit_system(conn, "furlongs")
    conn.close()
    assert ctx.settings()["local_timezone"] is None


def test_units_hands_back_a_copy(tmp_path):
    """UNIT_SYSTEMS is shared vocabulary; a caller mutating what it got back
    would re-define what 'imperial' means for every other vault in the process.
    """
    ctx = _vault_declaring(tmp_path, "erin", units="imperial")
    got = ctx.settings()["units"]
    got["distance"] = "furlong"
    assert ctx.settings()["units"]["distance"] == "mi"
    assert vaultmod.UNIT_SYSTEMS["imperial"]["distance"] == "mi"


def test_a_declaration_can_be_cleared(tmp_path):
    ctx = _vault_declaring(tmp_path, "fran",
                           zone="America/New_York", units="imperial")
    conn = ctx.connect()
    vaultmod.set_local_timezone(conn, None)
    vaultmod.set_unit_system(conn, None)
    conn.commit()
    conn.close()
    assert ctx.settings() == {
        "local_timezone": None, "unit_system": None, "units": None,
        "workout_source_arbitration_from": None, "block_qualify_hr_max": None,
    }
