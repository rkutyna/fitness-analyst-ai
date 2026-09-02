"""Shared test fixtures: a temp SQLite DB seeded with synthetic daily_metrics."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from health_advisor import db as dbmod
from health_advisor import mcp_server
from health_advisor import normalize as nz
from health_advisor.context import VaultContext

# The live 3.4 GB DB. The package has no default path any more (T-003), so this
# is spelled out here — the guard still has to know what it is defending.
PROD_DB_PATH = (dbmod.REPO_ROOT / "data" / "health.db").resolve()
_real_connect = dbmod.connect


def _value_col(metric: str) -> str:
    return "sum" if nz.agg_for(metric) == "sum" else "avg"


def seed_metric(conn: sqlite3.Connection, metric: str, start: str,
                values: list[float], unit: str | None = None,
                counts: int | list[int] = 1) -> None:
    """Insert one daily_metrics row per value, starting at `start` (inclusive).
    `counts` sets each row's sample count (the wear-density signal): a scalar
    applies to every day, or pass a per-day list the same length as `values`."""
    col = _value_col(metric)
    unit = unit or nz.canonical_unit(metric, "")
    if isinstance(counts, int):
        counts = [counts] * len(values)
    d0 = date.fromisoformat(start)
    for i, v in enumerate(values):
        d = (d0 + timedelta(days=i)).isoformat()
        conn.execute(
            "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, max, last, unit) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (metric, d, counts[i], v if col == "sum" else v,
             v if col == "avg" else v, v, v, v, unit),
        )
    conn.commit()


def seed_workout(conn: sqlite3.Connection, workout_type: str, local_date: str,
                 duration_min: float, distance_mi: float | None,
                 energy_kcal: float | None = None,
                 avg_heart_rate: float | None = None,
                 max_heart_rate: float | None = None) -> None:
    """Insert one workouts row. Synthesizes the NOT NULL utc/dedupe fields."""
    start_utc = f"{local_date}T12:00:00Z"
    end_utc = f"{local_date}T13:00:00Z"
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, energy_kcal, distance_mi, unit_distance, source, dedupe_key, "
        "avg_heart_rate, max_heart_rate) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'mi', 'test', ?, ?, ?)",
        (workout_type, start_utc, end_utc, local_date, duration_min, energy_kcal,
         distance_mi, f"{workout_type}|{start_utc}|{end_utc}|test",
         avg_heart_rate, max_heart_rate),
    )
    conn.commit()


@pytest.fixture(autouse=True)
def _isolate_db(request, tmp_path, monkeypatch):
    """Give every test its own vault, and make opening the production DB a hard
    error.

    Since T-003 there is no default path to redirect: `db.connect` requires one,
    every runner takes a `VaultContext`, and `mcp_server`'s tools are bound to a
    session by `build_server`/`build_tools`. So a test can no longer reach the
    real DB by *omission* — it would have to name it. This fixture defends
    against naming it.

    `db.connect` is the single choke point (nothing else calls sqlite3.connect
    on a non-temp path) and it CREATES a missing file rather than failing, so a
    stray absolute path would corrupt the real DB silently. The wrapper refuses
    that path loudly instead.

    The `vault` fixture below is what tests should use for the temp path; this
    one only guards. @pytest.mark.live tests are exempt: they deliberately read
    the real DB and are skipped unless `-m live`.
    """
    if request.node.get_closest_marker("live"):
        return

    def _guarded_connect(db_path, *, read_only=False):
        if Path(db_path).expanduser().resolve() == PROD_DB_PATH:
            raise AssertionError(
                f"test tried to open the production DB at {PROD_DB_PATH}; "
                "point it at tmp_path (see tests/conftest.py::_isolate_db)"
            )
        return _real_connect(db_path, read_only=read_only)

    monkeypatch.setattr(dbmod, "connect", _guarded_connect)


@pytest.fixture(autouse=True)
def _ncbi_contact(monkeypatch):
    """NCBI requires a real developer contact and the engine ships no default,
    so an unset HA_NCBI_EMAIL is a deliberate refusal (corpus_sources). Tests
    must not depend on the developer's environment for it."""
    monkeypatch.setenv("HA_NCBI_EMAIL", "tests@example.org")


@pytest.fixture
def vault_path(tmp_path):
    """This test's vault file — the same path `conn` and `vault` use, so a test
    can seed through the connection and then read through the bound tools."""
    return tmp_path / "health.db"


@pytest.fixture
def vault(vault_path):
    """A writable VaultContext for this test's own vault.

    Writable because most tests that want a context want to seed one. Narrow it
    with `vault.revoking(context.WRITE)` when the point of the test is that a
    read-only session cannot write.
    """
    return VaultContext.local(vault_path, user_id="test", writable=True)


@pytest.fixture
def tools(vault):
    """The MCP tools bound to this test's vault, as the model would get them.

    A SimpleNamespace rather than the module, so a test calls
    `tools.get_latest("body_mass")` and physically cannot reach another
    session's vault — which is the property T-003 bought.
    """
    return SimpleNamespace(**mcp_server.build_tools(vault))


@pytest.fixture(autouse=True)
def _ollama_backend(monkeypatch):
    """Pin the LLM backend to the in-process Ollama transport for every test —
    the suite's httpx.MockTransport fixtures model that path. Codex-backend
    tests opt back in by re-setting llm.BACKEND to "codex"."""
    from health_advisor import llm
    monkeypatch.setattr(llm, "BACKEND", "ollama")


@pytest.fixture
def conn(vault):
    """A writable connection to this test user's schema-initialized vault.

    Schema-initialized but NOT D3-declared: only `vault.build_vault` declares a
    vault, so a test database accepts any series exactly as a snapshot does.
    Tests that want the D3 write guard call `vault.declare_vault` themselves.
    """
    c = vault.connect()
    dbmod.init_db(c)
    yield c
    c.close()


@pytest.fixture
def ro_conn_factory(vault):
    """Returns functions that reopen this user's vault writable or read-only."""

    def _open_writable():
        c = vault.connect()
        dbmod.init_db(c)
        return c

    def _open_ro():
        return vault.read_only()

    return _open_writable, _open_ro


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.live tests unless the user explicitly selects markers
    (e.g. `pytest -m live`)."""
    if config.getoption("-m"):  # an explicit -m expression: let pytest filter
        return
    skip_live = pytest.mark.skip(reason="live model test; run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


# --- captured live figures ----------------------------------------------------
# Written by tests/fixtures/capture.py, which is a script, not a test: opening
# the production DB from a test is an AssertionError (see above), so the audits'
# validation numbers have to be captured once and committed. Each file carries a
# provenance block naming the query and the figure it should reproduce.

def load_fixture(name: str) -> dict:
    """The `data` payload of tests/fixtures/<name>.json."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent / "fixtures" / f"{name}.json"
    return json.loads(path.read_text())["data"]


@pytest.fixture
def ran_hot_sessions():
    """The seven sessions the easy-band warning actually fired on (F4-1)."""
    return load_fixture("ran_hot")


@pytest.fixture
def block_sessions():
    """Bucket series for the running workouts the block figures were measured on."""
    return load_fixture("blocks")


@pytest.fixture
def revision_series():
    """resting_heart_rate / walking_heart_rate_average day rows, plan window."""
    return load_fixture("revision_series")
