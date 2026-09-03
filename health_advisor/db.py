"""SQLite connection + idempotent helpers — the single source of truth.

Both ingestion paths (backfill.py, receiver.py) and the MCP server import from
here so there is exactly one place that knows the schema and the upsert rules.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
# schema.sql ships INSIDE the package. It used to sit at the repository root,
# resolved via REPO_ROOT — which works from a checkout and breaks under a
# non-editable `pip install`, where there is no repository root and the file
# was not packaged at all. Keeping it beside the code makes the installed
# artifact self-contained; pyproject ships it as package data.
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# The shape of a vault, stamped into `vault_meta` when one is created and
# advanced only after the migration sequence has completed. A migration across
# every user's vault has to know what shape each one is in, and a vault that
# never said cannot be asked: the answer would have to be inferred from its
# contents, per user, at the exact moment inference is least affordable. Three
# lines now against that, decided 2026-08-22 (#11).
#
# Bump when a change would make an older vault read wrongly rather than merely
# incompletely — a renamed column, a changed unit, a redefined key. Adding a
# table is not a bump; the code already tolerates absence.
VAULT_SCHEMA_VERSION = 2

# There is deliberately no default database path here. One process serves more
# than one user's vault, so a module-level default is how a session ends up
# reading the wrong one; every caller passes a path, and above this layer that
# path arrives inside a context.VaultContext. Entry points resolve it from argv.


# --------------------------------------------------------------------------- #
# Connection / init
# --------------------------------------------------------------------------- #
# How long a blocked connection waits for the lock before SQLITE_BUSY.
# DELETE journal mode means a writer holds EXCLUSIVE for its whole transaction
# and readers are locked out for the duration. At 5 s a reader that arrived
# during an ingest simply failed; the receiver's largest observed batch was
# 1,052,330 records. Writes are chunked now (see receiver.INGEST_CHUNK) so the
# lock is held in short bursts, and 30 s is far longer than any one burst —
# together they turn "reader errors out" into "reader waits a moment".
BUSY_TIMEOUT_MS = 30_000


class _VaultConnection(sqlite3.Connection):
    """Connection type used to retain the requested open mode locally."""


def connect(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with sane pragmas. read_only uses a URI immutable mode."""
    db_path = Path(db_path)
    if read_only:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, factory=_VaultConnection
        )
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, factory=_VaultConnection)
    conn._health_advisor_read_only = read_only
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        # DELETE (rollback) journal — NOT WAL. In WAL mode even a SELECT writes
        # the -shm file, which fails when Grafana opens the DB from a read-only
        # mount ("attempt to write a readonly database"). DELETE-mode reads never
        # write, so the :ro mount works. Writes here are infrequent (daily).
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


# Columns added after the initial schema. `CREATE TABLE IF NOT EXISTS` won't add
# them to a pre-existing table, so apply them as idempotent ALTERs on every init.
_ADDED_COLUMNS = {
    "records": {
        "hk_uuid": "TEXT",
        "hk_type_identifier": "TEXT",
        "source_revision_json": "TEXT",
        "hk_device_id": "TEXT",
    },
    "workouts": {
        "avg_heart_rate": "REAL",
        "max_heart_rate": "REAL",
        "hk_uuid": "TEXT",
    },
    # source_kind is the D19 discriminator: which provenance the row's `sum`
    # currently carries. A constant DEFAULT is legal in SQLite's ALTER TABLE ADD
    # COLUMN, so every existing vault gains it — labelled 'records', which is
    # what every pre-D19 row actually is — on its next init_db.
    "daily_metrics": {"last": "REAL",
                      "source_kind": "TEXT NOT NULL DEFAULT 'records'"},
    # Deletion lag is measurable only if the sample's date is captured at the
    # moment its row is deleted; every vault ingesting before this existed has
    # tombstones that cannot be back-filled. Additive so the live vault gains
    # them on its next ingest rather than needing a rebuild.
    "hk_deletions": {
        "sample_local_date": "TEXT",
        "sample_metric": "TEXT",
    },
    # These are additive because subjective rows are hand-entered and cannot
    # be regenerated (P1-3, W7-3). The nightly food line is convenience data
    # (P5-4); the running-only fields are deliberately separate from the
    # long-standing 1-5 soreness series (P8-3, P4-5).
    "subjective": {
        "food_note": "TEXT",
        "jog_niggle": "TEXT",
        "jog_niggle_detail": "TEXT",
        "talk_test": "TEXT",
    },
    "conversation_turns": {
        "answers_turn_id": "TEXT",
        "client_disconnected_at": "TEXT",
        "attachments_json": "TEXT",
    },
}

# Tables which were not present in older vaults.
#
# READ THIS BEFORE ADDING A KEY (corrected 2026-08-26, #124). The comment that
# used to sit here said schema.sql "carries the canonical fresh-database
# declaration", which reads as though it does not reach existing vaults. It
# does: init_db below runs `executescript(schema.sql)` unconditionally on every
# call, and every table there is CREATE TABLE IF NOT EXISTS. Measured on a vault
# built from the snapshot's real 11 tables: init_db takes it to 20, and emptying
# this dict entirely changes nothing.
#
# Two sessions reasoned from the old wording and both concluded a table declared
# only in schema.sql would be silently absent from existing vaults. It would not.
#
# Worse, the duplication runs the wrong way: _apply_table_migrations executes at
# the TOP of init_db, before the executescript, so for any table named here THIS
# copy wins and schema.sql's is a dead no-op -- on fresh vaults too. Editing the
# schema.sql declaration of a table listed here has no effect on any vault, with
# a green suite. tests/test_schema_declarations.py is what makes that fail loudly.
#
# So: a new table needs schema.sql only. This dict is now EMPTY, and #124's last
# open box -- "a table is declared in exactly one place" -- is what emptied it.
# plan_projections lived here and in schema.sql; the schema.sql copy was the dead
# one. Removing this copy leaves schema.sql as the single declaration, which is
# the file a reader goes to. Measured before removing: no entry in _ADDED_COLUMNS
# targets plan_projections, so the "must exist before the column migrations run"
# justification did not apply to it.
#
# Add a key here ONLY for a table that must exist before _apply_column_migrations
# runs -- and then it must NOT also be in schema.sql, or this copy silently wins
# again. tests/test_schema_declarations.py enforces that disjointness.
_ADDED_TABLES: dict[str, str] = {}


def _apply_column_migrations(conn: sqlite3.Connection) -> None:
    """Add columns that must exist before schema indexes are created."""
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    for table, cols in _ADDED_COLUMNS.items():
        if table not in tables:
            continue
        # ``init_db`` is also a public seam for callers holding a plain
        # sqlite3.Connection (without sqlite3.Row), including old vault
        # upgrade tooling. PRAGMA rows are tuples in that case.
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _conversation_turns_need_migration(conn: sqlite3.Connection) -> bool:
    """Return whether the old assistant-only answer constraint is present.

    SQLite has no ALTER CHECK. The decision queue needs both legal edges:
    assistant -> user for ordinary follow-ups and user -> assistant for a
    decision answering a proposal. A pre-existing vault therefore needs a
    table rebuild, while a fresh vault gets the declaration from schema.sql.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'conversation_turns'"
    ).fetchone()
    if row is None:
        return False
    sql = row[0] or ""
    # A legacy table with no answer CHECK is already compatible with the new
    # trigger and must not be rebuilt merely to normalize its DDL. Only the
    # explicit assistant-only CHECK requires a table migration.
    return re.search(
        r"answers_turn_id\s+IS\s+NULL\s+OR\s+role\s*=\s*'assistant'",
        sql, re.I,
    ) is not None


def _migrate_conversation_turns_answers_constraint(conn: sqlite3.Connection) -> None:
    """Rebuild old conversation_turns without losing its append-only log.

    This is deliberately separate from additive column migrations: changing a
    CHECK constraint is a table migration. The replacement keeps the same nine
    columns and all existing values, then the canonical schema recreates the
    triggers with the new opposite-role rule.
    """
    if not _conversation_turns_need_migration(conn):
        return

    columns = {row[1] for row in conn.execute("PRAGMA table_info(conversation_turns)")}
    copied = (
        "id", "conversation_id", "sequence", "role", "content", "created_at",
        "supersedes_turn_id", "answers_turn_id", "client_disconnected_at",
        "attachments_json",
    )
    expressions = [name if name in columns else "NULL" for name in copied]

    # Foreign-key enforcement cannot be toggled inside a transaction. init_db
    # owns the open operation, so make the rebuild atomic while temporarily
    # disabling checks; the copied rows are retained exactly and the pragma is
    # restored before returning.
    if conn.in_transaction:
        conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN")
        conn.execute("DROP TRIGGER IF EXISTS conversation_turns_no_update")
        conn.execute("DROP TRIGGER IF EXISTS conversation_turns_no_delete")
        conn.execute("DROP TRIGGER IF EXISTS conversation_turns_supersedes_same_conversation")
        conn.execute("DROP TRIGGER IF EXISTS conversation_turns_answers_same_conversation")
        conn.execute(
            """
            CREATE TABLE conversation_turns__new (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE RESTRICT,
                sequence INTEGER NOT NULL CHECK (sequence > 0),
                role TEXT NOT NULL CHECK (length(trim(role)) > 0),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                supersedes_turn_id TEXT
                    REFERENCES conversation_turns(id) ON DELETE RESTRICT,
                answers_turn_id TEXT
                    REFERENCES conversation_turns(id) ON DELETE RESTRICT,
                client_disconnected_at TEXT,
                attachments_json TEXT,
                UNIQUE (conversation_id, sequence),
                CHECK (supersedes_turn_id IS NULL OR supersedes_turn_id <> id),
                CHECK (answers_turn_id IS NULL OR role IN ('assistant', 'user')),
                CHECK (client_disconnected_at IS NULL OR role = 'assistant')
            )
            """
        )
        names = ", ".join(copied)
        conn.execute(
            f"INSERT INTO conversation_turns__new ({names}) "
            f"SELECT {', '.join(expressions)} FROM conversation_turns"
        )
        conn.execute("DROP TABLE conversation_turns")
        conn.execute("ALTER TABLE conversation_turns__new RENAME TO conversation_turns")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _review_questions_need_migration(conn: sqlite3.Connection) -> bool:
    """Return whether an older vault rejects the session confirmation kind."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'review_questions'"
    ).fetchone()
    if row is None:
        return False
    sql = (row[0] or "").lower()
    return "session_confirmation" not in sql


def _migrate_review_questions_kind_constraint(conn: sqlite3.Connection) -> None:
    """Add the weekly review's session-confirmation question kind."""
    if not _review_questions_need_migration(conn):
        return
    if conn.in_transaction:
        conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN")
        conn.execute("DROP INDEX IF EXISTS idx_review_questions_conversation")
        conn.execute(
            """
            CREATE TABLE review_questions__new (
                asked_turn_id TEXT PRIMARY KEY
                    REFERENCES conversation_turns(id) ON DELETE RESTRICT,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE RESTRICT,
                kind TEXT NOT NULL CHECK (
                    kind IN ('standing', 'session_confirmation', 'anomaly', 'followup')
                ),
                anomaly_ref TEXT,
                asked_at TEXT NOT NULL,
                CHECK ((kind = 'anomaly') = (anomaly_ref IS NOT NULL))
            )
            """
        )
        conn.execute(
            "INSERT INTO review_questions__new "
            "(asked_turn_id, conversation_id, kind, anomaly_ref, asked_at) "
            "SELECT asked_turn_id, conversation_id, kind, anomaly_ref, asked_at "
            "FROM review_questions"
        )
        conn.execute("DROP TABLE review_questions")
        conn.execute("ALTER TABLE review_questions__new RENAME TO review_questions")
        conn.execute(
            "CREATE INDEX idx_review_questions_conversation "
            "ON review_questions (conversation_id, kind)"
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _apply_table_migrations(conn: sqlite3.Connection) -> None:
    """Create additive projection tables before the canonical schema indexes."""
    for ddl in _ADDED_TABLES.values():
        conn.execute(ddl)


def init_db(conn: sqlite3.Connection) -> None:
    """Migrate a writable vault, then declare the shape that migration made.

    A read-only connection is intentionally a no-op after the compatibility
    check. Newer vaults are refused before any schema operation; older and
    undeclared writable vaults are stamped only after all migrations succeed.
    """
    declared = vault_schema_version(conn)
    if declared is not None and declared > VAULT_SCHEMA_VERSION:
        raise ValueError(
            f"vault declares schema version {declared}, newer than code version "
            f"{VAULT_SCHEMA_VERSION}"
        )
    if getattr(conn, "_health_advisor_read_only", False):
        return

    # Constraint changes are table migrations, not column migrations. Do this
    # before schema.sql so its IF NOT EXISTS triggers cannot preserve the old
    # assistant-only trigger on an existing vault.
    _migrate_conversation_turns_answers_constraint(conn)
    _migrate_review_questions_kind_constraint(conn)
    # CREATE TRIGGER IF NOT EXISTS cannot replace #108's assistant-only
    # trigger on a table that did not need a table rebuild.
    conn.execute("DROP TRIGGER IF EXISTS conversation_turns_answers_same_conversation")
    # schema.sql creates the partial HK indexes. Existing vaults need their
    # nullable columns first or SQLite would reject those CREATE INDEX lines.
    _apply_column_migrations(conn)
    _apply_table_migrations(conn)
    conn.executescript(SCHEMA_PATH.read_text())
    # W7-7: the treadmill benchmark is deliberately outside the raw workout
    # schema. Its four stage rows are a hand-recorded instrument, not Apple
    # Health workout data, and must survive re-ingestion of that data.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark (
            date TEXT NOT NULL,
            stage INTEGER NOT NULL CHECK (stage BETWEEN 1 AND 4),
            pace_min_per_mi REAL NOT NULL,
            median_hr_last_two_min REAL,
            talk_test TEXT,
            temp_c REAL,
            dew_point_c REAL,
            notes TEXT,
            -- How median_hr_last_two_min was arrived at. A protocol-derived
            -- window is a GUESS about session structure (8 min warmup, then
            -- 4-on/2-off), so a stage timed differently produces a plausible
            -- wrong number. Recording the provenance is what lets a reader
            -- tell those apart instead of trusting all four equally.
            median_source TEXT,
            PRIMARY KEY (date, stage)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_date ON benchmark (date)")
    # Keep this second pass for a partially-created database where the table
    # did not exist during the pre-schema pass.
    _apply_column_migrations(conn)
    _apply_table_migrations(conn)

    # The declaration is metadata about the shape after the complete
    # migration sequence above, not about the code that happened to open the
    # file. Preserve created_at while advancing an existing declaration;
    # insert it for a pre-stamp vault or a fresh one. This is intentionally the
    # last database operation before commit, so a failed migration cannot make
    # the vault claim a shape it does not have.
    if declared is None:
        conn.execute(
            "INSERT OR IGNORE INTO vault_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(VAULT_SCHEMA_VERSION)),
        )
    elif declared < VAULT_SCHEMA_VERSION:
        conn.execute(
            "UPDATE vault_meta SET value = ? WHERE key = 'schema_version'",
            (str(VAULT_SCHEMA_VERSION),),
        )
    conn.execute(
        "INSERT OR IGNORE INTO vault_meta (key, value) VALUES (?, ?)",
        ("created_at", utcnow_iso()),
    )
    conn.commit()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Dedupe keys (source-native identities -> idempotent INSERT OR IGNORE)
# --------------------------------------------------------------------------- #
def _value_identifies_sample(metric: str) -> bool:
    """Whether two same-window samples with different values are two real
    observations rather than one sample re-sent.

    Cumulative device samples (steps, distance, energy, minutes) measure a
    window, so the window IS the identity: Health Auto Export re-transmits
    overlapping windows on every sync and recomputes the quantity between
    sends, which used to insert a second row for one physical sample and
    inflate the daily sum by up to 12.7%.

    Everything else keeps value in the key. Instantaneous samples legitimately
    repeat a timestamp (150 such heart_rate groups in the backfill), and
    nutrition entries are per-food — two items logged in the same second are
    two foods, not a correction.
    """
    from . import normalize as nz  # local: normalize imports nothing from db

    return not (nz.agg_for(metric) == "sum" and not metric.startswith("dietary_"))


def record_key(
    metric: str,
    start_utc: str,
    end_utc: str,
    value,
    unit: str,
    source: str,
    *,
    source_metric: str | None = None,
    source_value=None,
) -> str:
    """Return the stable identity of one source sample.

    ``metric``, ``value`` and ``unit`` are normalized fields. They are not a
    durable identity: catalog additions can rename a metric, unit policy can
    change (``Cal`` -> ``kcal``), and canonicalization can rescale a value.
    Ingest callers therefore pass the source-native metric name and raw value.
    The optional arguments keep callers that have no native representation
    (for example subjective rows) compatible; their fallback deliberately
    omits the canonical unit so a unit-label edit alone cannot re-key them.
    """
    identity_metric = source_metric if source_metric is not None else metric
    identity_value = value if source_value is None else source_value
    if _value_identifies_sample(metric):
        raw = f"{identity_metric}|{start_utc}|{end_utc}|{identity_value}|{source}"
    else:
        raw = f"{identity_metric}|{start_utc}|{end_utc}|{source}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def vault_schema_version(conn: sqlite3.Connection) -> int | None:
    """The shape this vault currently declares, or None if it predates the stamp.

    None is a real answer and not an error: every vault built before
    2026-08-22 is in that state, and init_db handles it as a legacy vault by
    running the migration sequence before adding the declaration.
    """
    try:
        row = conn.execute(
            "SELECT value FROM vault_meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row else None


# --------------------------------------------------------------------------- #
# Plan projection storage
# --------------------------------------------------------------------------- #
def save_week_projection(
    conn: sqlite3.Connection,
    week,
    *,
    projection_id: str | None = None,
) -> str:
    """Persist one rebuildable Week envelope as a projection row.

    The import is local so the DB layer remains usable by the existing plan
    parser without creating an import cycle.  A parsed projection is a
    historical source and therefore cannot carry an enforced rule.
    """
    from . import plan_model

    if not isinstance(week, plan_model.Week):
        raise TypeError("week must be a plan_model.Week")
    if isinstance(week.provenance, plan_model.ParsedProvenance):
        if any(rule.enforced_from is not None for rule in week.rules):
            raise ValueError("historical parsed rules cannot be enforced retroactively")

    payload_json = week.to_json()
    if projection_id is None:
        projection_id = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    if isinstance(week.provenance, plan_model.ConversationTurnProvenance):
        conversation_turn_id = week.provenance.conversation_turn_id
        parsed_file = parsed_line = None
    else:
        conversation_turn_id = None
        parsed_file = week.provenance.file
        parsed_line = week.provenance.line

    conn.execute(
        """
        INSERT OR REPLACE INTO plan_projections
            (projection_id, week_start, payload_json, schema_version,
             conversation_turn_id, parsed_file, parsed_line, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            projection_id,
            week.week_start.isoformat(),
            payload_json,
            plan_model.PLAN_MODEL_SCHEMA_VERSION,
            conversation_turn_id,
            parsed_file,
            parsed_line,
            utcnow_iso(),
        ),
    )
    conn.commit()
    return projection_id


def load_week_projection(conn: sqlite3.Connection, week_start: str):
    """Load the newest projection for an ISO week start, if one exists."""
    from . import plan_model

    row = conn.execute(
        """
        SELECT payload_json, schema_version
        FROM plan_projections
        WHERE week_start = ?
        ORDER BY created_at DESC, projection_id DESC
        LIMIT 1
        """,
        (week_start,),
    ).fetchone()
    if row is None:
        return None
    if row["schema_version"] != plan_model.PLAN_MODEL_SCHEMA_VERSION:
        raise ValueError(f"unsupported plan projection schema version: {row['schema_version']}")
    payload = json.loads(row["payload_json"])
    return plan_model.Week.from_dict(payload)


def list_week_projections(conn: sqlite3.Connection) -> list:
    """Load all plan projections in deterministic storage order."""
    from . import plan_model

    rows = conn.execute(
        "SELECT payload_json FROM plan_projections ORDER BY week_start, created_at, projection_id"
    ).fetchall()
    return [plan_model.Week.from_json(row[0]) for row in rows]


# Explicit aliases make the projection boundary readable to callers that use
# "store" terminology while retaining the more precise save/load names.
store_week_projection = save_week_projection
load_plan_projection = load_week_projection


def bucket_key(metric: str, local_date: str, source: str, bucket: int,
               seconds: int) -> str:
    """Identity of an aggregated bucket, not of a sample.

    A bucketed row (vault.VAULT_BUCKET_SECONDS) has no sample identity to hash:
    it is a sum over a window. Its natural key is therefore the window itself —
    metric, local day, source, bucket index and width. `seconds` is part of it
    so a vault rebuilt at a different width cannot collide with the old rows.

    Same sha1 shape as the sample keys, because `records.dedupe_key` is
    documented as a hash and a plain composite string sitting among hashes is
    the kind of difference nobody notices until an upsert silently duplicates.
    """
    raw = f"bucket|{metric}|{local_date}|{source}|{seconds}|{bucket}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def workout_key(workout_type: str, start_utc: str, end_utc: str) -> str:
    # `source` is deliberately NOT part of the key: the two ingest paths label
    # one physical session differently (the export says "<name>'s Apple
    # Watch", HAE says "GymKit", "GymKit|<name>'s Apple Watch", "<name>'s
    # iPhone "), so including it made every fresh export.zip re-add workouts
    # the receiver had already stored. Type + exact start + exact end
    # identifies the session.
    raw = f"{workout_type}|{start_utc}|{end_utc}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def workout_event_key(workout_key: str, event_type: str, start_utc: str, duration_min) -> str:
    # duration is part of the key: concurrent segment streams can share a start
    # time (observed in real exports), differing only in duration.
    raw = f"{workout_key}|{event_type}|{start_utc}|{duration_min}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Upserts. Each row dict carries its own dedupe_key (callers build via *_key()).
# --------------------------------------------------------------------------- #
RECORD_COLS = (
    "metric", "value", "unit", "start_utc", "end_utc", "start_local",
    "local_date", "source", "origin", "dedupe_key", "hk_uuid",
    "hk_type_identifier", "source_revision_json", "hk_device_id",
)
WORKOUT_COLS = (
    "workout_type", "start_utc", "end_utc", "local_date", "duration_min",
    "energy_kcal", "distance_mi", "unit_distance", "source", "route_ref",
    "avg_heart_rate", "max_heart_rate", "dedupe_key", "hk_uuid",
)
# New identity fields are optional until the HealthKit-direct ingest exists.
_RECORD_OPTIONAL = (
    "hk_uuid", "hk_type_identifier", "source_revision_json", "hk_device_id",
)
# Optional workout columns default to NULL when a caller (e.g. backfill) omits them.
_WORKOUT_OPTIONAL = ("route_ref", "avg_heart_rate", "max_heart_rate", "hk_uuid")


def insert_records(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Upsert records. Returns the number of NEW rows added.

    A key collision now means "this sample again": for cumulative metrics the
    key is the window (see record_key), so the newer send carries the newer
    quantity and wins. For value-keyed metrics the colliding row is identical
    and the update is a no-op.

    Added rows are counted from the rowid high-water mark rather than
    total_changes, which would also count those updates.
    """
    sql = (
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key, hk_uuid, "
        "hk_type_identifier, source_revision_json, hk_device_id) "
        "VALUES (:metric, :value, :unit, :start_utc, :end_utc, :start_local, "
        ":local_date, :source, :origin, :dedupe_key, :hk_uuid, "
        ":hk_type_identifier, :source_revision_json, :hk_device_id) "
        "ON CONFLICT(dedupe_key) DO UPDATE SET "
        "value = excluded.value, unit = excluded.unit, origin = excluded.origin"
    )
    rows = [{**{k: None for k in _RECORD_OPTIONAL}, **r} for r in rows]
    if rows:
        # Import lazily: vault.py imports db for its copy/upsert helpers. The
        # declaration check is once per call, before executemany, not once per
        # row in a receiver chunk.
        from . import vault

        if vault.is_vault(conn):
            watermark = vault.compacted_through(conn)
            invalid = sorted({row["metric"] for row in rows
                              if watermark is not None
                              and row.get("origin") in vault.D3_GOVERNED_ORIGINS
                              and row["metric"] not in vault.VAULT_RAW_SERIES
                              and row["local_date"] <= watermark})
            if invalid:
                raise ValueError(
                    "vault raw records are behind compacted_through for metric(s) "
                    f"{', '.join(repr(metric) for metric in invalid)}; "
                    f"watermark is {watermark!r}; allowlist is "
                    f"{sorted(vault.VAULT_RAW_SERIES)}"
                )
    before = conn.execute("SELECT COALESCE(MAX(id), 0) FROM records").fetchone()[0]
    conn.executemany(sql, rows)
    after = conn.execute("SELECT COALESCE(MAX(id), 0) FROM records").fetchone()[0]
    return after - before


def delete_records_for_pairs(
    conn: sqlite3.Connection,
    pairs,
    *,
    origin: str = "backfill",
) -> int:
    """Delete records for the given (metric, local_date) pairs, optionally only
    of a given origin. Used by the receiver to evict coarse backfill rows for a
    day before inserting the fine-grained live samples (kills double-counting)."""
    n = 0
    for metric, date in set(pairs):
        cur = conn.execute(
            "DELETE FROM records WHERE metric = ? AND local_date = ? AND origin = ?",
            (metric, date, origin),
        )
        n += cur.rowcount
    return n


# Columns a second sighting of the same session may FILL but never overwrite.
_WORKOUT_MERGE = ("duration_min", "energy_kcal", "distance_mi", "unit_distance",
                  "route_ref", "avg_heart_rate", "max_heart_rate", "hk_uuid")


def _contains(outer_start: str, outer_end: str, start: str, end: str) -> bool:
    """True when [start, end] lies strictly inside [outer_start, outer_end].

    Strict: an identical span is the same session, not a fragment of one — and
    it hashes to the same `workout_key`, so it belongs on the merge path.
    """
    return (outer_start <= start and end <= outer_end
            and (outer_start < start or end < outer_end))


def contained_by(conn: sqlite3.Connection, row: dict,
                 accepted: Sequence[dict] = ()) -> dict | None:
    """The workout that wholly contains `row`, or None. Same type, SAME SOURCE.

    The phone ships workout objects that are sub-workouts of another object in
    the same batch — each with its own `hk_uuid`, so they are distinct
    HealthKit sessions as far as identity goes, and nothing upstream refuses
    them. #150: 2026-08-25 arrived as a 57.6-min run plus two 4-min, 0.2-mi
    rows nested inside it, one sharing its exact start. `workout_key` is
    type|start|end (see :func:`workout_key`), so a contained row hashes
    differently from its container BY CONSTRUCTION — no key over those three
    fields can ever collide a row with a row that contains it. The refusal has
    to be a containment rule, and it lives here because this is the one choke
    point every ingest path funnels through.

    `source` is the discriminator, and it is measured, not assumed. The
    snapshot holds 48 same-type containment pairs; 47 are CROSS-source —
    ErgData nested inside the Apple Watch, mostly rowing, 2019-2021 — two
    devices legitimately recording one session, and both rows are real
    historical record. Exactly one is same-source, and it is this defect. So
    same-source containment separates the bug from the history cleanly, and a
    cross-source contained row still stores.

    The two sources are compared TO EACH OTHER, never to a literal: the stored
    strings carry a curly apostrophe and a NO-BREAK SPACE (repr:
    '<name>\\u2019s Apple\\xa0Watch'), so any hardcoded comparand silently
    never matches.

    `accepted` are rows already admitted from the same batch, so a page that
    carries the container and its fragments together resolves without needing
    them committed first. A container found there is the candidate dict as the
    caller built it (no `id` yet); one found in the table is the stored row.
    Both carry type/start/end/duration/source, which is what a caller reporting
    the refusal needs.
    """
    src = row.get("source") or ""
    if not src:
        # Neither ingest path writes NULL here, but a source-less row cannot
        # prove same-source containment, and guessing would risk the 47.
        return None
    start, end = row["start_utc"], row["end_utc"]
    wtype = row["workout_type"]
    for a in accepted:
        if (a["workout_type"] == wtype and (a.get("source") or "") == src
                and _contains(a["start_utc"], a["end_utc"], start, end)):
            return a
    # A session we already hold is a re-sighting to be merged, not a new
    # fragment; refusing it would strand the half of the truth the other path
    # carries. Both identities count: the session key, and the HealthKit UUID.
    hk_uuid = row.get("hk_uuid")
    if conn.execute(
        "SELECT 1 FROM workouts WHERE dedupe_key = ? "
        "OR (hk_uuid IS NOT NULL AND hk_uuid = ?)",
        (row["dedupe_key"], hk_uuid),
    ).fetchone():
        return None
    hit = conn.execute(
        "SELECT id, workout_type, start_utc, end_utc, duration_min, source, dedupe_key "
        "FROM workouts WHERE workout_type = ? AND source = ? "
        "AND start_utc <= ? AND end_utc >= ? AND (start_utc < ? OR end_utc > ?) "
        # Outermost first, so the reported container is the real session and not
        # an intermediate fragment in a nested chain.
        "ORDER BY start_utc ASC, end_utc DESC LIMIT 1",
        (wtype, src, start, end, start, end),
    ).fetchone()
    return dict(hit) if hit is not None else None


def _span_seconds(row: dict) -> float:
    return _span_minutes(row["start_utc"], row["end_utc"]) * 60.0


def insert_workouts(conn: sqlite3.Connection, rows: Iterable[dict],
                    *, report=None) -> int:
    """Upsert workouts, merging a second sighting into the row we already have.

    This used to be INSERT OR IGNORE, which meant whichever ingest path saw a
    session first owned every column of it forever. The two paths carry
    disjoint halves of the truth — the export XML has duration/energy/distance
    and rarely HR, Health Auto Export has the GPS route, the HR summary and the
    per-second samples — so the second half was simply discarded.

    The merge is one-directional: COALESCE(existing, excluded), i.e. fill the
    holes and never clobber. `source` follows the same rule with '' treated as
    absent, since neither path writes NULL there.

    A row wholly contained in a same-type, same-source workout is refused
    outright (see :func:`contained_by`) — it is a sub-workout the phone emitted
    as a top-level session, and storing it makes every per-workout iteration
    count one session as two or three. `report(row, container)` is called once
    per refusal so the ingest can name what it dropped instead of dropping it
    silently, which is the half of #150 that let this run for two months.
    """
    sets = ", ".join(f"{c} = COALESCE({c}, excluded.{c})" for c in _WORKOUT_MERGE)
    sql = (
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, energy_kcal, distance_mi, unit_distance, source, route_ref, "
        "avg_heart_rate, max_heart_rate, dedupe_key, hk_uuid) "
        "VALUES (:workout_type, :start_utc, :end_utc, :local_date, :duration_min, "
        ":energy_kcal, :distance_mi, :unit_distance, :source, :route_ref, "
        ":avg_heart_rate, :max_heart_rate, :dedupe_key, :hk_uuid) "
        f"ON CONFLICT(dedupe_key) DO UPDATE SET {sets}, "
        "source = COALESCE(NULLIF(source, ''), excluded.source)"
    )
    # Tolerate rows missing optional keys (backfill) and ignore transient extras
    # like route_points that aren't columns.
    rows = [{**{k: None for k in _WORKOUT_OPTIONAL}, **r} for r in rows]
    # Longest span first so one batch carrying a session and its fragments
    # resolves the same way whatever order they arrive in. list.sort is stable,
    # so equal spans keep the caller's order — and two rows of one type with
    # equal spans share a dedupe_key anyway, which is the merge path.
    rows.sort(key=_span_seconds, reverse=True)
    kept: list[dict] = []
    for r in rows:
        outer = contained_by(conn, r, kept)
        if outer is not None:
            if report is not None:
                report(r, outer)
            continue
        kept.append(r)
    rows = kept
    # Count new rows from the rowid high-water mark: total_changes would also
    # count the merges above, and `workouts_added` is reported to the phone and
    # asserted on by the cross-source dedupe tests.
    before = conn.execute("SELECT COALESCE(MAX(id), 0) FROM workouts").fetchone()[0]
    conn.executemany(sql, rows)
    after = conn.execute("SELECT COALESCE(MAX(id), 0) FROM workouts").fetchone()[0]
    return after - before


# Reconciliation thresholds: how far the device summary may drift from the raw
# samples before the samples win (bpm).
HR_AVG_TOLERANCE = 5.0
HR_MAX_TOLERANCE = 8.0
# ...and how much sample series is needed before it may overrule the summary:
# enough samples, spanning enough of the workout to be representative of it.
HR_MIN_SAMPLES = 20
HR_MIN_COVERAGE = 0.8


def _span_minutes(lo: str, hi: str) -> float:
    try:
        return (datetime.fromisoformat(hi) - datetime.fromisoformat(lo)).total_seconds() / 60.0
    except (TypeError, ValueError):
        return 0.0


def reconcile_workout_heart_rate(
    conn: sqlite3.Connection,
    local_dates=None,
    *,
    since: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    report=None,
) -> int:
    """Correct workout avg/max HR from the raw heart_rate sample series.

    Device summaries occasionally contradict the samples shipped for the same
    window (HAE, 2026-07-22: 129/134 against samples averaging 152, peak 184 —
    a hard run graded easy). The samples win, but only when they're strong
    evidence: at least HR_MIN_SAMPLES of them, spanning HR_MIN_COVERAGE of the
    workout so a partial series can't misrepresent the whole. A summary that
    merely drifts within the tolerances is left alone as aggregation noise; a
    missing summary is always filled in.

    `local_dates` scopes the pass to those workout local_dates (None = all).
    `since`/`until` bound it by local_date instead, for the history-wide pass
    (see `python -m health_advisor.db --reconcile-hr`). `dry_run` counts what
    would change without writing.
    Returns the number of workouts updated.
    """
    sql = ("SELECT id, local_date, start_utc, end_utc, duration_min, avg_heart_rate, "
           "max_heart_rate FROM workouts")
    where: list[str] = []
    params: list = []
    if local_dates is not None:
        dates = sorted(local_dates)
        if not dates:
            return 0
        where.append(f"local_date IN ({','.join('?' * len(dates))})")
        params += dates
    if since:
        where.append("local_date >= ?")
        params.append(since)
    if until:
        where.append("local_date <= ?")
        params.append(until)
    if where:
        sql += " WHERE " + " AND ".join(where)
    params = tuple(params)
    updated = 0
    for w in conn.execute(sql, params).fetchall():
        s = conn.execute(
            "SELECT COUNT(*) n, AVG(value) a, MAX(value) m, "
            "MIN(start_utc) lo, MAX(start_utc) hi FROM records "
            "WHERE metric = 'heart_rate' AND start_utc BETWEEN ? AND ?",
            (w["start_utc"], w["end_utc"]),
        ).fetchone()
        if s["n"] < HR_MIN_SAMPLES:
            continue
        if _span_minutes(s["lo"], s["hi"]) < HR_MIN_COVERAGE * (w["duration_min"] or 0):
            continue
        if (w["avg_heart_rate"] is not None
                and abs(w["avg_heart_rate"] - s["a"]) <= HR_AVG_TOLERANCE
                and abs((w["max_heart_rate"] or 0) - s["m"]) <= HR_MAX_TOLERANCE):
            continue
        if report is not None:
            report(w, s)
        if not dry_run:
            conn.execute(
                "UPDATE workouts SET avg_heart_rate = ?, max_heart_rate = ? WHERE id = ?",
                (s["a"], s["m"], w["id"]),
            )
        updated += 1
    return updated


def insert_workout_events(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR IGNORE INTO workout_events (workout_key, event_type, start_utc, "
        "end_utc, duration_min, dedupe_key) VALUES (:workout_key, :event_type, "
        ":start_utc, :end_utc, :duration_min, :dedupe_key)"
    )
    before = conn.total_changes
    conn.executemany(sql, rows)
    return conn.total_changes - before


# --------------------------------------------------------------------------- #
# daily_metrics recompute (idempotent upsert from records)
# --------------------------------------------------------------------------- #
def _source_table(source_table: str) -> str:
    """Quote one of our SQL source-table names.

    This is an internal selector, not a SQL fragment supplied by a caller.
    Keeping the validation here makes the two aggregation queries below safe
    when the HealthKit path points them at its connection-local staging table.
    """
    if not source_table or not source_table.replace("_", "").isalnum():
        raise ValueError(f"invalid aggregation source table: {source_table!r}")
    return '"' + source_table.replace('"', '""') + '"'


def _arbitration(
    conn: sqlite3.Connection, metric: str, date: str, *,
    source_table: str = "records",
    arbitration_window: tuple[str, str] | None = None,
    arbitration_window_kind: str = "local_date",
) -> tuple[str, list]:
    """Extra WHERE clause excluding rows another source already accounts for.

    Only cumulative metrics can double-count, so everything else gets an empty
    clause. Three exclusions, all scoped to the data being recomputed (see
    normalize.MIRROR_SOURCES / SAMPLE_CEILING for why each exists):

      * a mirror source loses to the Apple devices from its cutoff date on, and
        wins before it — but only when the other source actually wrote that
        day. A coarse number beats no number.
      * a sample above the metric's ceiling is a whole-day estimate, dropped
        only when normal samples remain to take its place.
      * a post-2026-08-21 workout's lower-priority concurrent device samples
        lose to GymKit indoors or Apple Watch outdoors. This is workout-window
        device priority, not a new whole-day mirror source.
    """
    from . import normalize as nz
    source = _source_table(source_table)

    if nz.agg_for(metric) != "sum":
        return "", []

    clause, params = "", []
    sources = [r[0] for r in conn.execute(
        f"SELECT DISTINCT source FROM {source} WHERE metric = ? AND local_date = ?",
        (metric, date),
    )]
    mirrors = [s for s in sources if nz.is_mirror_source(s)]
    others = [s for s in sources if not nz.is_mirror_source(s)]
    if mirrors and others:
        losers = mirrors if date >= nz.mirror_loses_from(mirrors[0]) else others
        clause += f" AND source NOT IN ({','.join('?' * len(losers))})"
        params += losers

    workout_clause, workout_params = _workout_arbitration(
        conn, metric, date=date, source_table=source_table,
        arbitration_window=arbitration_window,
        arbitration_window_kind=arbitration_window_kind)
    clause += workout_clause
    params += workout_params

    ceiling = nz.SAMPLE_CEILING.get(metric)
    if ceiling is not None:
        has_normal = conn.execute(
            f"SELECT 1 FROM {source} WHERE metric = ? AND local_date = ? "
            "AND value <= ? LIMIT 1", (metric, date, ceiling),
        ).fetchone()
        if has_normal:
            clause += " AND value <= ?"
            params.append(ceiling)
    return clause, params


def _workout_source_labels(
    conn: sqlite3.Connection, *, source_table: str, date: str | None,
    arbitration_window: tuple[str, str] | None,
    arbitration_window_kind: str,
    scoped,
) -> dict[str, list[str]]:
    """Resolve the relevant raw source labels in the caller's window."""
    from . import normalize as nz

    source = _source_table(source_table)
    scope, scope_args = scoped(source)
    date_clause = ""
    date_args: list[str] = []
    if arbitration_window is None and date is not None:
        date_clause = f" AND {source}.local_date = ?"
        date_args.append(date)
    rows = conn.execute(
        f"SELECT DISTINCT source FROM {source} "
        "WHERE metric = ? AND local_date >= ?"
        f"{date_clause}{scope}",
        ("distance_walking_running", nz.WORKOUT_SOURCE_ARBITRATION_FROM,
         *date_args, *scope_args),
    ).fetchall()
    labels = {role: [] for role in ("watch", "iphone", "gymkit")}
    for row in rows:
        raw = row[0]
        normalized = " ".join((raw or "").replace("\xa0", " ").split())
        if not normalized or "|" in normalized:
            raise ValueError(
                "distance arbitration refuses post-cutoff source label "
                f"{raw!r}: pipe-joined and empty labels have no device "
                "identity")
        role = nz.workout_source_role(raw)
        if role in labels:
            labels[role].append(raw)
    for role in labels:
        labels[role].sort(key=lambda value: (value is not None, value or ""))
    return labels


def _workout_role_ctes(
    labels: dict[str, list[str]],
    arbitration_window: tuple[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Build one-bind-per-label SQL sets shared by all arbitration predicates."""
    ctes: list[str] = []
    args: list[str] = []
    for role in ("watch", "iphone", "gymkit"):
        values = labels[role]
        if values:
            ctes.append(
                f"{role}_sources(source) AS "
                f"(VALUES {','.join('( ?)' for _ in values)})")
            args.extend(values)
        else:
            ctes.append(f"{role}_sources(source) AS (SELECT NULL WHERE 0)")
    if arbitration_window is not None:
        ctes.append(
            "arbitration_window(start_value, end_value) AS "
            "(VALUES (?, ?))")
        args.extend(arbitration_window)
    return ", ".join(ctes), args


def _workout_arbitration(
    conn: sqlite3.Connection, metric: str, *, date: str | None = None,
    source_table: str = "records",
    arbitration_window: tuple[str, str] | None = None,
    arbitration_window_kind: str = "utc",
) -> tuple[str, list]:
    """Filter lower-priority device samples in the requested read window.

    Distinct post-cutoff source labels are classified in Python on every call.
    Their raw spellings are then bound once in role-specific SQL sets. The
    correlated filter remains a ``NOT EXISTS`` so it never binds one value per
    losing sample and is valid on raw sqlite connections without custom
    functions.
    """
    query, params = _workout_arbitration_query(
        conn, metric, date=date, source_table=source_table,
        arbitration_window=arbitration_window,
        arbitration_window_kind=arbitration_window_kind)
    if not query:
        return "", []
    return " AND NOT EXISTS (" + query + ")", params


def _workout_arbitration_query(
    conn: sqlite3.Connection, metric: str, *, date: str | None = None,
    source_table: str = "records",
    arbitration_window: tuple[str, str] | None = None,
    arbitration_window_kind: str = "utc",
) -> tuple[str, list]:
    """Build the shared loser subquery used by both arbitration directions."""
    from . import normalize as nz

    if metric != "distance_walking_running":
        return "", []
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'workouts'"
    ).fetchone() is None:
        return "", []
    if arbitration_window is not None:
        if len(arbitration_window) != 2:
            raise ValueError("arbitration_window must be (start, end)")
    if arbitration_window_kind not in ("utc", "local_date"):
        raise ValueError("arbitration_window_kind must be 'utc' or 'local_date'")
    if arbitration_window is not None:
        start, end = arbitration_window
        if (arbitration_window_kind == "utc" and start >= end) or \
                (arbitration_window_kind == "local_date" and start > end):
            relation = "start < end" if arbitration_window_kind == "utc" \
                else "start <= end"
            raise ValueError(f"arbitration_window must have {relation}")
    source = _source_table(source_table)

    def query_scoped(alias: str) -> tuple[str, list]:
        """Return the caller's read-window predicate for one SQL alias."""
        if arbitration_window is None:
            return "", []
        start, end = arbitration_window
        if arbitration_window_kind == "local_date":
            return f" AND {alias}.local_date >= ? AND {alias}.local_date <= ?", [start, end]
        return f" AND {alias}.start_utc >= ? AND {alias}.start_utc < ?", [start, end]

    def scoped(alias: str) -> tuple[str, list]:
        """Use the one shared caller-window bind set in the returned clause."""
        if arbitration_window is None:
            return "", []
        if arbitration_window_kind == "local_date":
            return (
                f" AND {alias}.local_date >= "
                "(SELECT start_value FROM arbitration_window)"
                f" AND {alias}.local_date <= "
                "(SELECT end_value FROM arbitration_window)", []
            )
        return (
            f" AND {alias}.start_utc >= "
            "(SELECT start_value FROM arbitration_window)"
            f" AND {alias}.start_utc < "
            "(SELECT end_value FROM arbitration_window)", []
        )

    def workout_scoped(alias: str) -> tuple[str, list]:
        """Bound workout candidates with the indexed date column first.

        UTC windows can straddle a source's local calendar date. One day on
        either side is therefore only a candidate bound; the exact UTC range
        below remains authoritative. The local-date bound is what prevents a
        query for a short UTC window from walking every post-floor workout via
        the unindexed start_utc predicate.
        """
        if arbitration_window is None or arbitration_window_kind == "local_date":
            return scoped(alias)
        start, end = arbitration_window
        start_day = (datetime.fromisoformat(start.replace("Z", "+00:00")).date()
                     - timedelta(days=1)).isoformat()
        end_day = (datetime.fromisoformat(end.replace("Z", "+00:00")).date()
                   + timedelta(days=1)).isoformat()
        return (
            f" AND {alias}.local_date >= ?"
            f" AND {alias}.local_date <= ?"
            + scoped(alias)[0], [start_day, end_day]
        )

    role_labels = _workout_source_labels(
        conn, source_table=source_table, date=date,
        arbitration_window=arbitration_window,
        arbitration_window_kind=arbitration_window_kind,
        scoped=query_scoped,
    )
    role_ctes, role_args = _workout_role_ctes(
        role_labels, arbitration_window=arbitration_window)

    def member(column: str, role: str) -> str:
        return f"{column} IN (SELECT source FROM {role}_sources)"

    cutoff = nz.WORKOUT_SOURCE_ARBITRATION_FROM.replace("'", "''")
    day_scope, day_scope_args = scoped("day_winner")
    workout_scope, workout_scope_args = workout_scoped("workout")
    winner_window = (
        "winner.metric = 'distance_walking_running' "
        "AND winner.start_utc >= workout.start_utc "
        "AND winner.start_utc < workout.end_utc"
    )
    has_gymkit = (
        f"EXISTS (SELECT 1 FROM {source} AS winner "
        f"WHERE {winner_window} AND {member('winner.source', 'gymkit')})"
    )
    has_watch = (
        f"EXISTS (SELECT 1 FROM {source} AS winner "
        f"WHERE {winner_window} AND {member('winner.source', 'watch')})"
    )
    day_loser = (
        f"({source}.local_date >= '{cutoff}' "
        f"AND {member(f'{source}.source', 'iphone')} "
        f"AND EXISTS (SELECT 1 FROM {source} AS day_winner "
        "WHERE day_winner.metric = 'distance_walking_running' "
        f"AND day_winner.local_date = {source}.local_date "
        f"{day_scope} AND {member('day_winner.source', 'watch')}))"
    )
    workout_loser = (
        "EXISTS (SELECT 1 FROM workouts AS workout "
        f"WHERE workout.local_date >= '{cutoff}' "
        f"{workout_scope} "
        f"AND {source}.start_utc >= workout.start_utc "
        f"AND {source}.start_utc < workout.end_utc "
        "AND (("
        f"{member(f'{source}.source', 'iphone')} AND ({has_gymkit} OR {has_watch})"
        ") OR ("
        f"{member(f'{source}.source', 'watch')} AND {has_gymkit}"
        ")))"
    )
    return (
        "WITH " + role_ctes + " "
        "SELECT 1 FROM (SELECT 1) AS arbitration WHERE ("
        f"{day_loser} OR {workout_loser})",
        [*role_args, *day_scope_args, *workout_scope_args],
    )


def arbitrated_pairs(
    conn: sqlite3.Connection, *, source_table: str = "records",
) -> list[tuple[str, str]]:
    """Every (metric, date) whose aggregate cross-source arbitration changes.

    The full rebuild aggregates in one bulk pass for speed; these few thousand
    pairs are then recomputed the careful way.
    """
    from . import normalize as nz
    source_ref = _source_table(source_table)

    pairs: set[tuple[str, str]] = set()
    for source in nz.MIRROR_SOURCES:
        pairs |= {(r["metric"], r["local_date"]) for r in conn.execute(
            f"SELECT DISTINCT metric, local_date FROM {source_ref} WHERE source LIKE ?",
            (f"%{source}%",),
        )}
    for metric, ceiling in nz.SAMPLE_CEILING.items():
        pairs |= {(metric, r["local_date"]) for r in conn.execute(
            f"SELECT DISTINCT local_date FROM {source_ref} WHERE metric = ? AND value > ?",
            (metric, ceiling),
        )}
    query, params = _workout_arbitration_query(
        conn, "distance_walking_running", source_table=source_table,
    )
    if query:
        pairs |= {("distance_walking_running", r["local_date"]) for r in conn.execute(
            f"SELECT DISTINCT local_date FROM {source_ref} "
            "WHERE metric = ? AND EXISTS (" + query + ")",
            ("distance_walking_running", *params),
        )}
    # Whole-day iPhone-vs-Watch arbitration has no workout row to correlate
    # with, so report its changed dates separately. Resolve the same scoped
    # raw labels as _workout_arbitration; the CTE keeps every label bound once.
    if query:
        def unscoped(alias: str) -> tuple[str, list]:
            return "", []

        role_labels = _workout_source_labels(
            conn, source_table=source_table, date=None,
            arbitration_window=None, arbitration_window_kind="local_date",
            scoped=unscoped,
        )
        role_ctes, role_args = _workout_role_ctes(role_labels)
        cutoff = nz.WORKOUT_SOURCE_ARBITRATION_FROM.replace("'", "''")
        day_loser = (
            f"{source_ref}.local_date >= '{cutoff}' AND "
            f"{source_ref}.source IN (SELECT source FROM iphone_sources) AND "
            f"EXISTS (SELECT 1 FROM {source_ref} AS day_winner "
            "WHERE day_winner.metric = 'distance_walking_running' "
            f"AND day_winner.local_date = {source_ref}.local_date "
            "AND day_winner.source IN (SELECT source FROM watch_sources))"
        )
        pairs |= {("distance_walking_running", r[0]) for r in conn.execute(
            f"WITH {role_ctes} SELECT DISTINCT local_date FROM {source_ref} "
            "WHERE metric = ? AND " + day_loser,
            (*role_args, "distance_walking_running"),
        )}
    return sorted(p for p in pairs if nz.agg_for(p[0]) == "sum")


def rebuild_metric_source_months(
    conn: sqlite3.Connection,
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
    full: bool = False,
) -> int:
    """Rebuild `metric_source_months` from `records`.

    full=True  -> every (metric, month) that has raw samples.
    pairs      -> (metric, local_date) tuples; the months they fall in.

    This is the only surviving consumer of raw `records.source`, and it exists
    so that consumer can stop being raw: under D3 most series' samples never
    reach a vault, and `analysis.instrument_eras` reading `records` directly
    would see no instrument change and report none. A missing boundary is worse
    than a wrong one here — `history.py` refuses to average across a boundary,
    so an unreported one means it averages straight through the watch leaving in
    2022 (F3-2).

    Returns the number of rows written.
    """
    outer = conn.in_transaction
    if full:
        conn.execute("DELETE FROM metric_source_months")
        conn.execute(
            "INSERT INTO metric_source_months (metric, month, source, n) "
            "SELECT metric, substr(local_date, 1, 7), source, COUNT(*) "
            "FROM records GROUP BY metric, substr(local_date, 1, 7), source"
        )
        if not outer:
            conn.commit()
        return conn.execute(
            "SELECT COUNT(*) FROM metric_source_months").fetchone()[0]

    if not pairs:
        return 0

    months = sorted({(metric, date[:7]) for metric, date in pairs})
    written = 0
    for metric, month in months:
        # Recount the whole month rather than incrementing: an ingest can
        # re-deliver samples already stored (every write here is idempotent by
        # dedupe_key), and an increment would count those twice.
        conn.execute(
            "DELETE FROM metric_source_months WHERE metric = ? AND month = ?",
            (metric, month),
        )
        cur = conn.execute(
            "INSERT INTO metric_source_months (metric, month, source, n) "
            "SELECT metric, substr(local_date, 1, 7), source, COUNT(*) "
            "FROM records WHERE metric = ? AND substr(local_date, 1, 7) = ? "
            "GROUP BY metric, substr(local_date, 1, 7), source",
            (metric, month),
        )
        written += cur.rowcount if cur.rowcount > 0 else 0
    if not outer:
        conn.commit()
    return written


def record_backed_series(conn: sqlite3.Connection) -> list[str]:
    """Metrics that actually have sample-level rows in THIS database.

    In the full snapshot that is every ingested metric; in a D3 vault it is the
    six in `vault.VAULT_RAW_SERIES`. Anything that rebuilds a cache from
    `records` has to scope itself to this, or it deletes rows it cannot
    reconstruct.
    """
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT metric FROM records ORDER BY metric")]


def recompute_daily_metrics(
    conn: sqlite3.Connection,
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
    full: bool = False,
    source_table: str = "records",
) -> int:
    """Recompute daily_metrics from records, then re-apply Apple's own totals.

    A thin wrapper on purpose (D19 §Q2). `_recompute_core` below is the whole of
    the old body; the one thing this adds is that `apply_consolidated_totals`
    runs after EVERY return path of it, including the two early ones
    (`if not rebuildable: return 0` and `if not pairs: return 0`). Both are
    states in which a consolidated total can be sitting in `hk_daily_totals`
    with a double-counted `daily_metrics` row in front of it, so a call placed
    at the end of either branch would miss them.

    The override is a property of "daily_metrics has just been recomputed", not
    of one branch of how it was recomputed — which is also why it cannot be an
    event that happens once at ingest: `_healthkit_ingest` recomputes
    (step_count, day) on every batch carrying a raw step sample, so a value
    written once would be rebuilt into the double-count by the next sync.

    NOT a rebuild's override, though: `build_vault` never calls this function at
    all, so a rebuild has to call `apply_consolidated_totals` itself (D19 §Q3).
    """
    written = _recompute_core(conn, pairs=pairs, full=full,
                              source_table=source_table)
    # `pairs=None` when full, because a full rebuild relabels every series it
    # touched. An empty `pairs` sequence means "no restriction was stated", the
    # same reading `_recompute_core`'s own `if not pairs` takes of it — D19's
    # test 12c calls recompute_daily_metrics(pairs=[]) and requires the override
    # to have run.
    apply_consolidated_totals(conn, pairs=None if full else pairs)
    return written


def _recompute_core(
    conn: sqlite3.Connection,
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
    full: bool = False,
    source_table: str = "records",
) -> int:
    """Recompute daily_metrics from one connection-local source table.

    Call `recompute_daily_metrics` instead unless you are it: this half knows
    nothing about Apple's consolidated totals, so on its own it leaves a
    double-counted `sum` behind wherever one exists.

    full=True  -> rebuild every series that HAS raw records (used after a
                  backfill). In a declared vault this is limited to the
                  allowlist, whose raw history is complete; series without raw
                  rows keep the daily aggregates they already have.
    pairs      -> only recompute these (metric, local_date) tuples (incremental).
    Returns the number of (metric, date) aggregate rows written.

    **`full` does not mean "the whole table".** It used to: `DELETE FROM
    daily_metrics` followed by a rebuild from `records`. That is safe only where
    `records` holds every series, which is true of the full snapshot and false
    of a D3 vault — there, `records` carries the complete allowlist plus a
    transient tail while `daily_metrics` carries about a hundred series. The
    old form deleted aggregates it could not reconstruct. Everything else would
    have come back as "no data", which is the one answer this project is not
    allowed to give when it means "not in this vault". So the rebuild is scoped
    to the series it can actually reconstruct, and it records in `ingest_log`
    which those were.
    """
    source = _source_table(source_table)
    if full:
        from . import vault

        scope = ""
        scope_params: list[str] = []
        if vault.is_vault(conn):
            allowed = sorted(vault.VAULT_RAW_SERIES)
            scope = " WHERE metric IN (" + ",".join("?" * len(allowed)) + ")"
            scope_params = allowed
        rebuildable = [r[0] for r in conn.execute(
            f"SELECT DISTINCT metric FROM {source}{scope} ORDER BY metric",
            scope_params,
        )]
        if not rebuildable:
            log_ingest(conn, "recompute", "rebuild", 0, 0,
                       "no raw records: nothing rebuilt, daily_metrics left intact")
            return 0
        placeholders = ",".join("?" * len(rebuildable))
        untouched = [r[0] for r in conn.execute(
            f"SELECT DISTINCT metric FROM daily_metrics "
            f"WHERE metric NOT IN ({placeholders}) ORDER BY metric", rebuildable)]
        conn.execute(
            f"DELETE FROM daily_metrics WHERE metric IN ({placeholders})",
            rebuildable)
        conn.execute(
            f"""
            INSERT INTO daily_metrics
                (metric, date, count, sum, avg, min, max, last, unit)
            WITH ranked AS (
                SELECT metric, local_date, value, unit,
                       ROW_NUMBER() OVER (
                           PARTITION BY metric, local_date
                           -- start_utc is the sample's observation time. The
                           -- end_utc and id tie-breakers make equal starts
                           -- deterministic without letting an end-time
                           -- ordering disagreement change the primary rule.
                           ORDER BY start_utc DESC, end_utc DESC, id DESC
                       ) AS rn
                FROM {source}
            )
            SELECT metric, local_date, COUNT(*), SUM(value), AVG(value),
                   MIN(value), MAX(value), MAX(value) FILTER (WHERE rn = 1),
                   MAX(unit)
            FROM ranked GROUP BY metric, local_date
            """
        )
        # The bulk pass above sums every row; redo the days where more than one
        # source describes the same movement.
        arbitration_pairs = arbitrated_pairs(conn, source_table=source_table)
        if vault.is_vault(conn):
            arbitration_pairs = [pair for pair in arbitration_pairs
                                 if pair[0] in rebuildable]
        # `_recompute_core`, NOT the wrapper: the wrapper applies the
        # consolidated override, and the wrapper that called us will apply it
        # again on the way out. Going through it here would run the override
        # twice per full rebuild. It is idempotent, so that is waste rather than
        # a defect — but do not "simplify" this back to the public name (D19).
        _recompute_core(conn, pairs=arbitration_pairs,
                        source_table=source_table)
        # Provenance travels with the aggregates: instrument_eras reads it, and
        # in a vault it is the only place source information survives.
        rebuild_metric_source_months(conn, full=True)
        written = conn.execute(
            "SELECT COUNT(*) FROM daily_metrics").fetchone()[0]
        # Say what was rebuilt and what was deliberately left alone. A silent
        # partial rebuild is indistinguishable from a complete one until
        # somebody asks a series that was not in scope.
        log_ingest(
            conn, "recompute", "rebuild", written, written,
            f"rebuilt={len(rebuildable)} series from records; "
            f"left_intact={len(untouched)} series with no raw rows here"
            + (" (" + ", ".join(untouched[:12])
               + (", …" if len(untouched) > 12 else "") + ")" if untouched else ""),
        )
        return written

    if not pairs:
        return 0

    written = 0
    source = _source_table(source_table)
    for metric, date in set(pairs):
        clause, extra = _arbitration(conn, metric, date,
                                     source_table=source_table)
        agg = conn.execute(
            f"""
            SELECT COUNT(*) c, SUM(value) s, AVG(value) a, MIN(value) mn,
                   MAX(value) mx, MAX(unit) u,
                    (SELECT value FROM {source}
                    WHERE metric = ? AND local_date = ?{clause}
                    -- Match the full rebuild's total order. id is stable for
                    -- otherwise identical timestamps and values.
                    ORDER BY start_utc DESC, end_utc DESC, id DESC LIMIT 1) AS last
            FROM {source} WHERE metric = ? AND local_date = ?{clause}
            """,
            (metric, date, *extra, metric, date, *extra),
        ).fetchone()
        if agg["c"] == 0:
            conn.execute("DELETE FROM daily_metrics WHERE metric = ? AND date = ?", (metric, date))
            continue
        conn.execute(
            """
            INSERT INTO daily_metrics
                (metric, date, count, sum, avg, min, max, last, unit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(metric, date) DO UPDATE SET
                count=excluded.count, sum=excluded.sum, avg=excluded.avg,
                min=excluded.min, max=excluded.max, last=excluded.last,
                unit=excluded.unit,
                -- The sum this statement just wrote came from `records`, so the
                -- label has to say so: source_kind describes the provenance of
                -- the sum in the row as it stands, and a stale
                -- 'apple_consolidated' on a records-derived sum is exactly the
                -- quiet lie D19 exists to prevent. recompute_daily_metrics
                -- re-applies the override on the way out, so a pair that still
                -- has a consolidated total gets its label straight back.
                -- (The full branch above needs no equivalent: it DELETEs and
                -- re-INSERTs, so those rows take the column DEFAULT.)
                source_kind='records'
            """,
            (metric, date, agg["c"], agg["s"], agg["a"], agg["mn"], agg["mx"],
             agg["last"], agg["u"]),
        )
        written += 1
    return written


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,)).fetchone() is not None


def apply_consolidated_totals(
    conn: sqlite3.Connection,
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> int:
    """Overwrite daily_metrics.sum with Apple's consolidated total (D19, #218).

    This must run after EVERY recompute — including the two that return early —
    not once at ingest: the receiver recomputes (step_count, day) on every batch
    that carries a raw step sample, so a value written once would be rebuilt
    into the double-count by the next sync. It is called from
    `recompute_daily_metrics`' wrapper, outside both branches, for exactly that
    reason. `build_vault` does not call recompute at all and must call this
    directly (D19 §Q3) — that call is step 4's, and is not here yet.

    Only `sum` and `unit` move, and `count` is NEVER synthesised: it stays the
    raw record count, 0 included, because it is one of the three columns
    verify_daily_metrics still compares on these rows. avg/min/max/last stay
    records-derived too. Note what that costs elsewhere: on a consolidated row
    `avg * count != sum`, and analysis._worn_rows must not read `count` as a
    wear-density signal (D19 §Q2).

    `pairs` restricts the update to those (metric, local_date) tuples. None —
    and an empty sequence, which states no restriction — means every row in
    `hk_daily_totals`. Idempotent: running it twice writes the same values.

    Returns the number of daily_metrics rows written. Guarded on the table's
    existence the way `_arbitration` guards on `workouts`, so a database that
    predates D19 still works.
    """
    if not _has_table(conn, "hk_daily_totals"):
        return 0
    # `pairs` is filtered in Python rather than bound into an IN list: it is
    # caller-sized (an ingest batch's whole affected set) and SQLite's bind limit
    # differs between the development Mac and the deployment host, so a query
    # that scales its binds passes here and breaks in production.
    # hk_daily_totals is small — one row per metric per day — so the full scan
    # costs nothing.
    wanted = {(m, d) for m, d in pairs} if pairs else None
    written = 0
    for row in conn.execute(
            "SELECT metric, local_date, value, unit FROM hk_daily_totals"
    ).fetchall():
        metric, day, value, unit = (row["metric"], row["local_date"],
                                    row["value"], row["unit"])
        if wanted is not None and (metric, day) not in wanted:
            continue
        updated = conn.execute(
            "UPDATE daily_metrics SET sum = ?, unit = ?, "
            "source_kind = 'apple_consolidated' WHERE metric = ? AND date = ?",
            (value, unit, metric, day)).rowcount
        if not updated:
            # A consolidated total for a day with no raw samples at all, which
            # is legitimate. count = 0 is the honest raw record count; the other
            # aggregates have nothing to be derived from and stay NULL.
            conn.execute(
                "INSERT INTO daily_metrics (metric, date, count, sum, avg, min, "
                "max, last, unit, source_kind) "
                "VALUES (?, ?, 0, ?, NULL, NULL, NULL, NULL, ?, "
                "'apple_consolidated')",
                (metric, day, value, unit))
        written += 1
    return written


def insert_daily_totals(
    conn: sqlite3.Connection,
    rows: Iterable[dict],
    *,
    batch_id: str,
) -> int:
    """Write received consolidated daily totals, with their revision history.

    One call writes all three things D19 §Q3 wants in one transaction:
    `hk_daily_totals`, one `hk_daily_total_revisions` row per write, and — on
    the first accepted row for each metric —
    `vault_meta.daily_totals_expected_from:<metric>`, the affirmative
    expectation `verify_daily_metrics`' check 6 reads. INSERT OR IGNORE for
    each key: it records where this deployment started receiving that metric's
    totals and must never move forward on a later pull. Per-metric keys matter
    because the phone may ship only a subset of the three D19 metrics.

    Each row is a dict with `metric`, `local_date`, `value`, `unit`, `interval`,
    `state`, `device_id` and `queried_at`.

    `lag_days` is derived HERE, from `queried_at`'s local date minus
    `local_date`, and is never taken from the phone as a count — a phone with a
    wrong clock then produces a visibly wrong lag rather than a plausible one.

    A revision row is written for every accepted write, including one that does
    not change the value. That is deliberate: N is read off this table as "the
    lag by which values stop changing", and a no-change write at lag k is the
    evidence for it. Duplicate suppression is the `commit_log` preflight's job,
    not this function's.

    Does NOT enforce the settle guard. A settled row is immutable in the storage
    engine (the `hk_daily_totals_settled_immutable` trigger), so an attempt to
    overwrite one raises sqlite3.IntegrityError from here; the receiver refuses
    it earlier and more politely, with a 409 and no database evidence.
    """
    now = utcnow_iso()
    written = 0
    for row in rows:
        metric, day = row["metric"], row["local_date"]
        state = row["state"]
        prior = conn.execute(
            "SELECT value, state, first_seen_at FROM hk_daily_totals "
            "WHERE metric = ? AND local_date = ?", (metric, day)).fetchone()
        lag_days = (datetime.fromisoformat(row["queried_at"][:10]).date()
                    - datetime.fromisoformat(day).date()).days
        if prior is None:
            conn.execute(
                "INSERT INTO hk_daily_totals (metric, local_date, value, unit, "
                "interval, state, device_id, queried_at, first_seen_at, "
                "settled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (metric, day, row["value"], row["unit"], row["interval"], state,
                 row["device_id"], row["queried_at"], now,
                 now if state == "settled" else None))
        else:
            conn.execute(
                "UPDATE hk_daily_totals SET value = ?, unit = ?, interval = ?, "
                "state = ?, device_id = ?, queried_at = ?, settled_at = ? "
                "WHERE metric = ? AND local_date = ?",
                (row["value"], row["unit"], row["interval"], state,
                 row["device_id"], row["queried_at"],
                 now if state == "settled" else None, metric, day))
        conn.execute(
            "INSERT INTO hk_daily_total_revisions (metric, local_date, "
            "from_value, to_value, from_state, to_state, lag_days, batch_id, "
            "recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (metric, day, prior["value"] if prior else None, row["value"],
             prior["state"] if prior else None, state, lag_days, batch_id, now))
        conn.execute(
            "INSERT OR IGNORE INTO vault_meta (key, value) VALUES "
            "(?, ?)", (f"daily_totals_expected_from:{metric}", day))
        written += 1
    return written


# --------------------------------------------------------------------------- #
# insights + ingest_log
# --------------------------------------------------------------------------- #
def write_insight(conn: sqlite3.Connection, date: str, text: str, tags: str = "") -> int:
    """Replace the canonical insight for a ``(date, tags)`` pair.

    Insight writes are retryable deliveries, not an append-only event log.  The
    schema's ``(date, created_at)`` uniqueness cannot express that rule, so it
    belongs here rather than in each caller's path-specific helper.

    **Transaction-neutral.** This used to be a ``with conn:`` block, which
    commits on exit — inside a caller's larger transaction that silently ended
    it, so the insight landed and everything the caller did afterwards could
    still be rolled back. That is the partial write T-005's commit phase exists
    to make impossible, and no test could have caught it because standalone
    behaviour is identical. A writer that may run inside a commit must leave
    transaction control to whoever opened one.
    """
    outer = conn.in_transaction
    try:
        conn.execute("DELETE FROM insights WHERE date = ? AND tags = ?", (date, tags))
        cur = conn.execute(
            "INSERT INTO insights (date, text, tags, created_at) VALUES (?, ?, ?, ?)",
            (date, text, tags, utcnow_iso()),
        )
    except Exception:
        if not outer:
            conn.rollback()
        raise
    if not outer:
        conn.commit()
    return cur.lastrowid or 0


def write_insight_ctx(ctx, date: str, text: str, tags: str = "") -> int:
    """:func:`write_insight` against a session's vault.

    Takes the context rather than a path so that a read-only session cannot
    write by holding a path it was handed for reading — ``ctx.connect()``
    refuses without the ``write`` capability. Every runner writes through here.
    """
    conn = ctx.connect()
    try:
        init_db(conn)
        return write_insight(conn, date, text, tags=tags)
    finally:
        conn.close()


def log_ingest(
    conn: sqlite3.Connection,
    source: str,
    kind: str,
    rows_seen: int,
    rows_added: int,
    detail: str = "",
) -> None:
    outer = conn.in_transaction
    conn.execute(
        "INSERT INTO ingest_log (source, kind, rows_seen, rows_added, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source, kind, rows_seen, rows_added, detail, utcnow_iso()),
    )
    # Only commit if we opened the transaction. Committing a caller's
    # transaction from inside a logging helper is how a partial write happens —
    # see write_insight, which was the same bug (T-005).
    if not outer:
        conn.commit()


# --------------------------------------------------------------------------- #
# Maintenance entry point
# --------------------------------------------------------------------------- #
def _main(argv: Sequence[str] | None = None) -> int:
    """`python -m health_advisor.db --reconcile-hr [--db P] [--since D] [--dry-run]`

    reconcile_workout_heart_rate is otherwise only ever called by the receiver,
    scoped to the days a POST touched, so a workout that predates the receiver
    — or that arrived by export.zip on a day the phone did not sync — is never
    revisited. This runs the same pass over history, with the same evidence
    gates (HR_MIN_SAMPLES / HR_MIN_COVERAGE) deliberately unchanged: the point
    is to reach the rows, not to lower the bar for overruling a device summary.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="python -m health_advisor.db")
    ap.add_argument("--reconcile-hr", action="store_true",
                    help="recompute workout avg/max HR from the raw heart_rate series")
    ap.add_argument("--db", required=True, help="path to the vault to reconcile")
    ap.add_argument("--since", help="only workouts with local_date >= this (YYYY-MM-DD)")
    ap.add_argument("--until", help="only workouts with local_date <= this (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--verbose", action="store_true", help="one line per workout changed")
    args = ap.parse_args(argv)

    if not args.reconcile_hr:
        ap.error("nothing to do: pass --reconcile-hr")

    conn = connect(args.db)
    try:
        def _show(w, s):
            if args.verbose:
                print(f"  {w['local_date']}  id={w['id']:<6} "
                      f"avg {w['avg_heart_rate']} -> {s['a']:.1f}  "
                      f"max {w['max_heart_rate']} -> {s['m']:.0f}  ({s['n']} samples)")

        n = reconcile_workout_heart_rate(
            conn, since=args.since, until=args.until,
            dry_run=args.dry_run, report=_show)
        if args.dry_run:
            print(f"dry run: {n} workouts would be updated (nothing written)")
        else:
            conn.commit()
            log_ingest(conn, "maintenance", "reconcile_hr", 0, n,
                       f"since={args.since} until={args.until} updated={n}")
            print(f"{n} workouts updated")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    raise SystemExit(_main())


# --- hand-entered jog minutes (F2-4) ---------------------------------------
#
# This is the only path in the system that can manufacture training volume that
# never happened, so it carries three guards, all of them required:
#   1. a manual value NEVER overwrites a measured one (analysis.impact_volume
#      prefers measured and falls back, and never blends the two — that is the
#      unlabelled-blend defect 19c5edc again);
#   2. every consumer that uses a manual value SAYS SO in its output; and
#   3. every entry ever made is listable, so a wrong one is findable rather
#      than permanent.
MANUAL_JOG_MAX_MINUTES = 300.0


def log_manual_jog(conn, local_date: str, *, jog_minutes: float,
                   source_note: str, why: str) -> None:
    """Record jog minutes for a session the watch could not measure.

    `why` is required and stored. A manual number with no reason attached is
    unauditable six weeks later, which is exactly when someone will be trying
    to work out where 22 minutes came from. Upserts on (local_date,
    source_note): a correction replaces, it does not accumulate.
    """
    minutes = float(jog_minutes)
    if minutes <= 0:
        raise ValueError("jog_minutes must be positive")
    if minutes > MANUAL_JOG_MAX_MINUTES:
        raise ValueError(
            f"jog_minutes={minutes} exceeds {MANUAL_JOG_MAX_MINUTES}; if that "
            "is real, it needs a second pair of eyes rather than a typo path")
    if not (source_note or "").strip():
        raise ValueError("source_note is required (e.g. 'treadmill')")
    if not (why or "").strip():
        raise ValueError("why is required — an unexplained manual entry is "
                         "indistinguishable from a mistake later")
    conn.execute(
        "INSERT INTO manual_jog (local_date, source_note, jog_minutes, why, "
        "entered_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(local_date, source_note) DO UPDATE SET "
        "jog_minutes = excluded.jog_minutes, why = excluded.why, "
        "entered_at = excluded.entered_at",
        (local_date, source_note.strip(), minutes, why.strip(),
         datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()


def manual_jog_entries(conn, start: str | None = None,
                       end: str | None = None) -> list[dict]:
    """Every hand-entered jog session, oldest first. Guard 3.

    Returns [] when the table does not exist. impact_volume calls this on every
    read, including against read-only connections to databases that predate the
    migration and against backups — and a missing optional table must not take
    out the number the whole plan is graded on.
    """
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' "
                        "AND name = 'manual_jog'").fetchone():
        return []
    where, args = "", []
    if start:
        where, args = " WHERE local_date >= ?", [start]
        if end:
            where, args = " WHERE local_date BETWEEN ? AND ?", [start, end]
    return [dict(r) for r in conn.execute(
        "SELECT local_date, source_note, jog_minutes, why, entered_at "
        f"FROM manual_jog{where} ORDER BY local_date, source_note", args)]


def manual_jog_by_day(conn, start: str, end: str) -> dict[str, dict]:
    """{local_date: {minutes, note}} over a range, summed across entries."""
    out: dict[str, dict] = {}
    for r in manual_jog_entries(conn, start, end):
        day = out.setdefault(r["local_date"], {"minutes": 0.0, "notes": []})
        day["minutes"] += r["jog_minutes"]
        day["notes"].append(f"{r['source_note']}: {r['why']}")
    return {d: {"minutes": round(v["minutes"], 1), "note": "; ".join(v["notes"])}
            for d, v in out.items()}
