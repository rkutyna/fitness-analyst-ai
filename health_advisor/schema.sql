-- Apple Health local analysis pipeline — single source of truth schema.
-- One SQLite DB. Idempotent design throughout. All timestamps stored UTC;
-- daily aggregates keyed by the LOCAL calendar day derived from the original offset.

-- Journal mode is set in db.connect() (DELETE, not WAL) so Grafana can read the
-- DB from a read-only mount — see the note there. Do not set journal_mode here:
-- init_db() runs on every connect and would override it.
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- vault_meta: facts about this database as a vault, not about its contents.
--
-- `owner` is the one that matters: the user whose vault this is. A session
-- carrying a different user_id is refused at connect (health_advisor/context.py).
-- Isolation under D4 is the FILE, not a `WHERE user_id =` that one query can
-- forget — but nothing stopped a worker from pointing Alice's context at Bob's
-- file, and "the paths are assigned correctly" is a hope, not a mechanism.
--
-- A vault with no `owner` row is unclaimed and is NOT refused: the development
-- snapshot and every test database are in that state. The guarantee is
-- therefore "a claimed vault cannot be opened by the wrong session", not "every
-- open is authorised". `VaultContext.claim()` is what makes it true of a vault.
--
-- `history_imported_through`, when present, is the inclusive final local date
-- imported by `vault.build_vault`. HealthKit-direct ingest refuses a whole batch
-- containing a record on or before that date. An absent key deliberately leaves
-- the vault unguarded; `vault.set_history_imported_through(conn, None)` clears it
-- and a deliberate migration may move it.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vault_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- conversations: one durable conversation identity per vault.
--
-- The turns are the record of the conversation. This row is deliberately only
-- identity and lifecycle metadata; it must not become a JSON transcript blob.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- conversation_turns: the append-only conversation event log.
--
-- A correction is another row, linked through supersedes_turn_id. UPDATE and
-- DELETE are refused by triggers below, so callers cannot replace history by
-- editing a turn in place.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversation_turns (
    id                  TEXT PRIMARY KEY,
    conversation_id     TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE RESTRICT,
    sequence            INTEGER NOT NULL CHECK (sequence > 0),
    role                TEXT NOT NULL CHECK (length(trim(role)) > 0),
    content             TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    supersedes_turn_id  TEXT
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    answers_turn_id     TEXT
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    client_disconnected_at TEXT,
    attachments_json    TEXT,
    UNIQUE (conversation_id, sequence),
    CHECK (supersedes_turn_id IS NULL OR supersedes_turn_id <> id),
    CHECK (answers_turn_id IS NULL OR role IN ('assistant', 'user')),
    CHECK (client_disconnected_at IS NULL OR role = 'assistant')
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation
    ON conversation_turns (conversation_id, sequence);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_answers
    ON conversation_turns (conversation_id, answers_turn_id)
    WHERE answers_turn_id IS NOT NULL;

CREATE TRIGGER IF NOT EXISTS conversation_turns_no_update
BEFORE UPDATE ON conversation_turns
BEGIN
    SELECT RAISE(ABORT,
        'conversation turns are append-only; insert a superseding turn');
END;

CREATE TRIGGER IF NOT EXISTS conversation_turns_no_delete
BEFORE DELETE ON conversation_turns
BEGIN
    SELECT RAISE(ABORT,
        'conversation turns are append-only; retain the log');
END;

CREATE TRIGGER IF NOT EXISTS conversation_turns_supersedes_same_conversation
BEFORE INSERT ON conversation_turns
WHEN NEW.supersedes_turn_id IS NOT NULL
 AND NOT EXISTS (
     SELECT 1 FROM conversation_turns
     WHERE id = NEW.supersedes_turn_id
       AND conversation_id = NEW.conversation_id
 )
BEGIN
    SELECT RAISE(ABORT,
        'a superseded turn must belong to the same conversation');
END;

CREATE TRIGGER IF NOT EXISTS conversation_turns_answers_same_conversation
BEFORE INSERT ON conversation_turns
WHEN NEW.answers_turn_id IS NOT NULL
 AND NOT EXISTS (
     SELECT 1 FROM conversation_turns
     WHERE id = NEW.answers_turn_id
       AND conversation_id = NEW.conversation_id
       AND role <> NEW.role
 )
BEGIN
    SELECT RAISE(ABORT,
        'an answer must link to a turn of the other role in the same conversation');
END;

-- ---------------------------------------------------------------------------
-- commit_log: one row per applied commit, keyed by the writer's idempotency key.
--
-- The recovery mechanism for a partial write here is retrying the same write —
-- records commit in chunks, not one atomic batch, because DELETE journal mode
-- makes a single large transaction hold EXCLUSIVE for its whole duration. Retry
-- is only a recovery mechanism if writes are idempotent, and "idempotent" has
-- to mean more than "the same rows end up there": a re-run that writes an
-- insight also spends a provider call, so the second run must not run at all.
--
-- The UNIQUE key is what makes replay a no-op. `epoch` records which lease
-- generation applied it, so a log row is also the evidence of who won.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS commit_log (
    key        TEXT PRIMARY KEY,
    epoch      INTEGER NOT NULL,
    applied_at TEXT NOT NULL,
    detail     TEXT
);

-- ---------------------------------------------------------------------------
-- records: cleaned per-sample quantity/category data from both ingestion paths.
--   metric     canonical metric name (see health_advisor/normalize.py)
--   value      numeric value (category records map to a numeric code/duration)
--   unit       canonical unit string
--   start_utc  ISO-8601 UTC, e.g. 2024-01-15T08:30:00+00:00
--   end_utc    ISO-8601 UTC
--   local_date YYYY-MM-DD in the sample's original local timezone
--   source     device/app that produced the sample (sourceName)
--   dedupe_key sha1 of the natural key; UNIQUE => INSERT OR IGNORE is idempotent
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS records (
    id         INTEGER PRIMARY KEY,
    metric     TEXT NOT NULL,
    value      REAL,
    unit       TEXT,
    start_utc  TEXT NOT NULL,
    end_utc    TEXT NOT NULL,
    start_local TEXT,                              -- naive local wall time for intraday bucketing
    local_date TEXT NOT NULL,
    source     TEXT,
    origin     TEXT NOT NULL DEFAULT 'backfill',  -- 'backfill' | 'receiver'
    dedupe_key TEXT NOT NULL UNIQUE,
    -- HealthKit identity is nullable because the existing backfill and HAE
    -- rows have only their window-based dedupe key.
    hk_uuid              TEXT,
    hk_type_identifier   TEXT,
    source_revision_json TEXT,
    hk_device_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_records_metric_localdate ON records (metric, local_date);
CREATE INDEX IF NOT EXISTS idx_records_metric_start     ON records (metric, start_utc);
-- One HealthKit sample may expand into several canonical metrics (notably
-- sleep), so UUID uniqueness must be scoped by canonical metric.
CREATE UNIQUE INDEX IF NOT EXISTS idx_records_metric_hk_uuid
    ON records (metric, hk_uuid) WHERE hk_uuid IS NOT NULL;

-- ---------------------------------------------------------------------------
-- workouts: one row per workout session.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS workouts (
    id              INTEGER PRIMARY KEY,
    workout_type    TEXT NOT NULL,       -- canonical activity name
    start_utc       TEXT NOT NULL,
    end_utc         TEXT NOT NULL,
    local_date      TEXT NOT NULL,
    duration_min    REAL,                -- minutes
    energy_kcal     REAL,                -- active energy burned
    distance_mi     REAL,                -- miles (matches records distance unit)
    unit_distance   TEXT,
    source          TEXT,
    route_ref       TEXT,                -- gpx filename if present (data/routes/)
    avg_heart_rate  REAL,                -- bpm, workout average
    max_heart_rate  REAL,                -- bpm, workout max
    dedupe_key      TEXT NOT NULL UNIQUE,
    hk_uuid         TEXT
);

CREATE INDEX IF NOT EXISTS idx_workouts_localdate ON workouts (local_date);
CREATE INDEX IF NOT EXISTS idx_workouts_type      ON workouts (workout_type, local_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_workouts_hk_uuid
    ON workouts (hk_uuid) WHERE hk_uuid IS NOT NULL;

-- ---------------------------------------------------------------------------
-- workout_session_marks: append-only user corrections to session identity.
--
-- A device-recorded workout is never rewritten or deleted when a user says it
-- was not a real session. This edge keeps the correction and its provenance
-- separate from the device row, while the stable workout key keeps it attached
-- across re-ingestion.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS workout_session_marks (
    id           INTEGER PRIMARY KEY,
    workout_id   INTEGER NOT NULL UNIQUE
        REFERENCES workouts(id) ON DELETE RESTRICT,
    workout_key  TEXT NOT NULL UNIQUE
        REFERENCES workouts(dedupe_key) ON DELETE RESTRICT,
    mark         TEXT NOT NULL CHECK (mark = 'not_a_session'),
    source       TEXT NOT NULL CHECK (length(trim(source)) > 0),
    marked_at    TEXT NOT NULL,
    reason       TEXT
);

CREATE INDEX IF NOT EXISTS idx_workout_session_marks_key
    ON workout_session_marks (workout_key);

CREATE TRIGGER IF NOT EXISTS workout_session_marks_no_update
BEFORE UPDATE ON workout_session_marks
BEGIN
    SELECT RAISE(ABORT,
        'workout session marks are append-only; a device row is never rewritten');
END;

CREATE TRIGGER IF NOT EXISTS workout_session_marks_no_delete
BEFORE DELETE ON workout_session_marks
BEGIN
    SELECT RAISE(ABORT,
        'workout session marks are append-only; the correction is retained');
END;

-- HealthKit operational state is mutable per device and type. It does not
-- belong in vault_meta, which describes the vault itself.
CREATE TABLE IF NOT EXISTS hk_sync_state (
    device_id           TEXT NOT NULL,
    type_identifier     TEXT NOT NULL,
    anchor_token        TEXT,
    last_batch_sequence INTEGER,
    last_batch_id       TEXT,
    updated_at          TEXT NOT NULL,
    PRIMARY KEY (device_id, type_identifier)
);

-- Deletions are retained as tombstones in this slice; applying them is a
-- later ingest concern.
CREATE TABLE IF NOT EXISTS hk_deletions (
    device_id       TEXT NOT NULL,
    type_identifier TEXT NOT NULL,
    hk_uuid         TEXT NOT NULL,
    deleted_at      TEXT NOT NULL,
    -- The deleted sample's own date and metric, captured at delete time
    -- because they are unrecoverable afterwards: the `records` row is gone,
    -- and the tombstone is all that survives. `deleted_at` alone gives the
    -- WHEN of a deletion but not the HOW LATE, and how late is the figure the
    -- compaction window has to be designed against (#37, D13). Both are NULL
    -- for a deletion whose sample this vault never held -- a tombstone is
    -- written before the add filter runs, so an unknown UUID still gets one --
    -- and those rows are outside the measurement rather than zeroes in it.
    sample_local_date TEXT,
    sample_metric     TEXT,
    PRIMARY KEY (device_id, type_identifier, hk_uuid)
);

-- ---------------------------------------------------------------------------
-- workout_events: <WorkoutEvent> children of a workout in the Apple export XML
-- (segments/laps/pauses). Only the full-export backfill produces these; the
-- live HAE payloads don't carry them, so a workout gains events when the next
-- export.zip is backfilled. Parent linkage is workouts.dedupe_key (stable
-- across re-runs), not the rowid.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS workout_events (
    id           INTEGER PRIMARY KEY,
    workout_key  TEXT NOT NULL,       -- workouts.dedupe_key of the parent session
    event_type   TEXT NOT NULL,       -- segment|lap|pause|resume|motion_paused|motion_resumed|marker
    start_utc    TEXT NOT NULL,
    end_utc      TEXT,                -- start + duration for events that have one
    duration_min REAL,                -- NULL for instantaneous events (pause/resume/marker)
    dedupe_key   TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_workout_events_key ON workout_events (workout_key);

-- ---------------------------------------------------------------------------
-- session_observation: the user's observation of what a session did.
--
-- This is a second observation beside the computed value, not a correction
-- applied to records or daily_metrics.  The statement time is part of the
-- key deliberately: a later statement is a new fact and the old one remains
-- available for comparison.  workout_key is the stable workouts identity,
-- never its rowid, so re-ingestion cannot detach an observation.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_observation (
    scope           TEXT NOT NULL CHECK (scope IN ('week','day','session')),
    local_date      TEXT NOT NULL,
    workout_key     TEXT NOT NULL DEFAULT '',
    field           TEXT NOT NULL,
    computed_value  REAL,
    stated_value    REAL,
    stated_text     TEXT,
    agrees          INTEGER NOT NULL CHECK (agrees IN (0,1)),
    computed_at     TEXT NOT NULL,
    stated_at      TEXT NOT NULL,
    evidence        TEXT NOT NULL CHECK (evidence IN ('recall','segments','device')),
    note            TEXT,
    PRIMARY KEY (scope, local_date, workout_key, field, stated_at)
);

CREATE INDEX IF NOT EXISTS idx_session_observation_date
    ON session_observation (local_date);

-- ---------------------------------------------------------------------------
-- daily_metrics: per-day-per-metric aggregates. THE table the agent & dashboard
-- read most. Recomputed from records for affected (metric, date) pairs only.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_metrics (
    metric TEXT NOT NULL,
    date   TEXT NOT NULL,                -- local YYYY-MM-DD
    count  INTEGER NOT NULL,
    sum    REAL,
    avg    REAL,
    min    REAL,
    max    REAL,
    last   REAL,                         -- final raw sample by observation time
    unit   TEXT,
    -- 'records' | 'apple_consolidated'. Which provenance the `sum` in THIS row
    -- currently carries (D19, #218). 'apple_consolidated' means `sum` came from
    -- hk_daily_totals — Apple's own consolidated daily total — while every other
    -- aggregate column stayed derived from `records`. Consequence, stated so it
    -- is never rediscovered as a bug: on such a row `avg * count != sum`, and
    -- `count` is the honest raw record count (0 when there are no raw samples),
    -- never a synthesised one. See db.apply_consolidated_totals.
    source_kind TEXT NOT NULL DEFAULT 'records',
    PRIMARY KEY (metric, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_metric_date ON daily_metrics (metric, date);

-- ---------------------------------------------------------------------------
-- hk_daily_totals: Apple's own consolidated daily total (D19, #218).
--
-- NOT a `records` row and deliberately not summable: a pre-aggregated day has
-- no window, no source and no hk_uuid, and putting it in `records` would let
-- recompute_daily_metrics add it to the raw samples it exists to replace.
--
-- `interval` is provenance, not a knob (D19 Q1): every row today says 'day',
-- and if the pinned interval ever moves, rows gathered under the old one stay
-- identifiable instead of silently mixing two populations.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hk_daily_totals (
    metric        TEXT NOT NULL,       -- canonical: step_count, flights_climbed, …
    local_date    TEXT NOT NULL,       -- local YYYY-MM-DD
    value         REAL NOT NULL,       -- Apple's consolidated total, canonical unit
    unit          TEXT NOT NULL,
    interval      TEXT NOT NULL,       -- 'day'
    state         TEXT NOT NULL CHECK (state IN ('provisional','settled')),
    device_id     TEXT NOT NULL,       -- which install pulled it
    queried_at    TEXT NOT NULL,       -- phone clock at the pull that wrote this value
    first_seen_at TEXT NOT NULL,       -- UTC, the provisional write
    settled_at    TEXT,                -- UTC, the settle write; NULL while provisional
    PRIMARY KEY (metric, local_date)
);

-- #220 Done-when 2, enforced in the storage engine rather than in a caller.
-- Python-level guards are bypassable by the next script somebody writes; a
-- trigger is not. Same shape as conversation_turns_no_update (schema.sql:79-86)
-- and retro_claims_no_update.
CREATE TRIGGER IF NOT EXISTS hk_daily_totals_settled_immutable
BEFORE UPDATE ON hk_daily_totals
WHEN OLD.state = 'settled'
BEGIN
    -- One string literal: SQLite has no adjacent-literal concatenation, so the
    -- two-line form this was written as is a syntax error, not a long message.
    SELECT RAISE(ABORT, 'hk_daily_totals: a settled daily total is immutable — re-pulling a settled day is a history violation, not a correction (D19/#220)');
END;

CREATE TRIGGER IF NOT EXISTS hk_daily_totals_settled_no_delete
BEFORE DELETE ON hk_daily_totals
WHEN OLD.state = 'settled'
BEGIN
    SELECT RAISE(ABORT, 'hk_daily_totals: settled totals are not deletable');
END;

-- ---------------------------------------------------------------------------
-- hk_daily_total_revisions: every write to hk_daily_totals, before and after.
--
-- This is the measurement #220 Done-when 1 asks for. It exists BEFORE the
-- settle pull is enabled: N is chosen from this table's distribution, not
-- guessed and then justified. The 0.00%-late-arrival figure on the live vault
-- predicts a small N; a prediction is what this tests, not what it assumes.
--
-- `lag_days` is derived on the SERVER from queried_at's local date minus
-- local_date, never sent by the phone as a count, so a phone with a wrong clock
-- produces a visibly wrong lag rather than a plausible one.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hk_daily_total_revisions (
    id          INTEGER PRIMARY KEY,
    metric      TEXT NOT NULL,
    local_date  TEXT NOT NULL,      -- the day being described
    from_value  REAL,               -- NULL on the first write for a day
    to_value    REAL NOT NULL,
    from_state  TEXT,               -- NULL on the first write
    to_state    TEXT NOT NULL,
    lag_days    INTEGER NOT NULL,   -- pull's local date minus local_date
    batch_id    TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hk_totals_rev_metric_date
    ON hk_daily_total_revisions (metric, local_date, lag_days);

-- ---------------------------------------------------------------------------
-- metric_source_months: which device recorded a metric, month by month.
--
-- Derived provenance, kept because D3 does not ship most raw `records` to the
-- vault and this is the only thing raw `source` was still needed for. Without
-- it, `analysis.instrument_eras` can see no instrument change and `history.py`
-- averages straight across one — the failure F3-2 is about: the watch left in
-- 2022 and changed the instrument under a series that never gapped, so a gap
-- test cannot find it.
--
-- One row per (metric, month, source) with its sample count. Rebuilt from
-- `records` wherever raw samples exist; carried into the vault as a table, so
-- era detection keeps working for every metric rather than only the six whose
-- samples survive the D3 filter.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS metric_source_months (
    metric TEXT NOT NULL,
    month  TEXT NOT NULL,                -- local YYYY-MM
    source TEXT NOT NULL,                -- exactly as stored in records.source
    n      INTEGER NOT NULL,
    PRIMARY KEY (metric, month, source)
);

CREATE INDEX IF NOT EXISTS idx_metric_source_months ON metric_source_months (metric, month);

-- ---------------------------------------------------------------------------
-- insights: agent-written daily summaries, shown on the dashboard.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS insights (
    id         INTEGER PRIMARY KEY,
    date       TEXT NOT NULL,            -- the day the insight is about (YYYY-MM-DD)
    text       TEXT NOT NULL,
    tags       TEXT,                     -- comma-separated
    created_at TEXT NOT NULL,            -- ISO-8601 UTC of when it was written
    UNIQUE (date, created_at)
);

CREATE INDEX IF NOT EXISTS idx_insights_date ON insights (date);

-- ---------------------------------------------------------------------------
-- ingest_log: what was ingested and when, for debugging daily syncs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_log (
    id           INTEGER PRIMARY KEY,
    source       TEXT NOT NULL,          -- 'backfill' | 'receiver'
    kind         TEXT,                   -- 'records' | 'workouts' | 'batch'
    rows_seen    INTEGER,
    rows_added   INTEGER,
    detail       TEXT,
    created_at   TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- subjective: nightly Telegram check-in, one row per local day. Partial
-- upsert (a write only overwrites the fields it provides). Numeric fields are
-- mirrored into records/daily_metrics (source/origin 'checkin') by
-- health_advisor/subjective.py so the analysis stack sees them; notes live
-- here only. sleep_quality is last night's sleep, stored on the wake day.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS subjective (
    date            TEXT PRIMARY KEY,   -- local day the entry describes
    stress          INTEGER,            -- 1-5
    soreness        INTEGER,            -- 1-5 muscle soreness
    energy          INTEGER,            -- 1-5 perceived energy
    sleep_quality   INTEGER,            -- 1-5, last night's sleep (wake day)
    caffeine_drinks REAL,               -- count of caffeinated drinks
    alcohol_drinks  REAL,               -- count of standard drinks
    notes           TEXT,               -- catch-all free text
    updated_at      TEXT NOT NULL       -- ISO-8601 UTC of last write
);

-- ---------------------------------------------------------------------------
-- workout_weather: outdoor conditions during a workout, joined via its GPX.
--
-- WHY THIS EXISTS. Audit part 1 (2026-08-15) called the plan's heat claims
-- "unmeasurable" and was wrong: workouts.route_ref points at a 1 Hz GPX, whose
-- trackpoints carry position and time, which is all a historical weather API
-- needs. The audit's own prototype found r=0.82 between dew point and cardiac
-- drift over five sessions -- a lead, not a result, but not one that should
-- have gone unmeasured for seven weeks.
--
-- WHY A TABLE AND NOT COLUMNS ON workouts. A 214-minute hike crosses real
-- weather; a 35-minute run does not. Long sessions get a sample every 30
-- minutes, so the relationship is one-to-many. offset_min is minutes from the
-- workout's start.
--
-- DEW POINT IS THE FIELD THAT MATTERS, not temperature -- it is what drives
-- heat strain, and it is why "run early" can be the wrong advice in a New
-- England August (Aug 11, 6:24 AM: 69F but 91% humidity). Heat index is
-- derivable from temp + humidity and is deliberately NOT stored.
--
-- PENDING ROWS. ERA5 lags ~5 days. A row whose readings are NULL but whose
-- fetched_utc is set means "asked, not yet published" -- distinguishable from
-- a workout never asked about at all, which has no row.
--
-- PRIVACY. lat/lon are stored and sent ROUNDED (see weather.COORD_PRECISION).
-- ERA5's native grid is coarser than the rounding, so nothing is lost.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS workout_weather (
    workout_id    INTEGER NOT NULL REFERENCES workouts(id) ON DELETE CASCADE,
    offset_min    INTEGER NOT NULL,     -- minutes from workout start; 0 = first point
    lat           REAL,                 -- rounded, see weather.COORD_PRECISION
    lon           REAL,
    observed_utc  TEXT,                 -- the hour this reading describes
    temp_f        REAL,
    humidity_pct  REAL,
    dew_point_f   REAL,                 -- the one that matters
    wind_kmh      REAL,
    source        TEXT NOT NULL,        -- e.g. 'open-meteo-era5'
    fetched_utc   TEXT NOT NULL,        -- set even when readings are NULL (pending)
    PRIMARY KEY (workout_id, offset_min)
);

-- Hand-entered jog minutes for sessions the watch cannot measure (F2-4).
--
-- Jun 27 and Jul 16 2026 were real running sessions that scored 0.0 jog
-- minutes: GymKit produces no per-sample distance, and impact_volume
-- classifies buckets by the pace their distance implies. PLAN.md's winter
-- strategy moves training indoors and the Week 9 monthly benchmark is a
-- treadmill session, so this stops being an edge case in October.
--
-- Kept in its own table rather than synthesised into `records`: a manufactured
-- distance sample would be indistinguishable from a measured one everywhere
-- downstream, forever. Here it can only be read by a consumer that asked for
-- it, and every consumer that uses it says so.
CREATE TABLE IF NOT EXISTS manual_jog (
    local_date  TEXT NOT NULL,
    source_note TEXT NOT NULL,     -- 'treadmill', 'track', ...
    jog_minutes REAL NOT NULL,
    why         TEXT NOT NULL,     -- free text; required, so an entry is auditable
    entered_at  TEXT NOT NULL,
    PRIMARY KEY (local_date, source_note)
);
CREATE INDEX IF NOT EXISTS idx_manual_jog_date ON manual_jog (local_date);

-- ---------------------------------------------------------------------------
-- user_facts: durable meaning supplied by the user or awaiting confirmation.
--
-- Facts are deliberately separate from measurements.  They are rendered as
-- text context only; no query in the analysis layer may use this table as an
-- input to a computation.  A proposal is retained when it is confirmed or
-- rejected, so the evidence and the user's decision remain auditable.  A
-- correction is a new row linked by supersedes_fact_id, like conversation
-- turns; the old row is never rewritten.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_facts (
    id                    TEXT PRIMARY KEY,
    text                  TEXT NOT NULL CHECK (length(trim(text)) > 0),
    source                TEXT NOT NULL CHECK (length(trim(source)) > 0),
    evidence              TEXT NOT NULL CHECK (length(trim(evidence)) > 0),
    stated_at             TEXT NOT NULL,
    supersedes_fact_id    TEXT
        REFERENCES user_facts(id) ON DELETE RESTRICT,
    scope                 TEXT NOT NULL CHECK (length(trim(scope)) > 0),
    confidence            REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    state                 TEXT NOT NULL CHECK (state IN
        ('stated', 'confirmed', 'proposed', 'rejected')),
    conversation_id       TEXT REFERENCES conversations(id) ON DELETE RESTRICT,
    conversation_turn_id  TEXT REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    CHECK (supersedes_fact_id IS NULL OR supersedes_fact_id <> id)
);

CREATE INDEX IF NOT EXISTS idx_user_facts_state
    ON user_facts (state, stated_at);
CREATE INDEX IF NOT EXISTS idx_user_facts_conversation
    ON user_facts (conversation_id, state);
CREATE INDEX IF NOT EXISTS idx_user_facts_evidence
    ON user_facts (evidence);

CREATE TRIGGER IF NOT EXISTS user_facts_no_update
BEFORE UPDATE ON user_facts
BEGIN
    SELECT RAISE(ABORT,
        'user facts are append-only; insert a superseding fact');
END;

-- ---------------------------------------------------------------------------
-- plan_statement_log: typed, append-only plan statements.
--
-- This is the source of truth for plan rules. Values stay typed: statement,
-- scope, provenance and the rule's interval are represented as structured
-- data, never rendered prose. A Week is rebuilt from these rows; it is not
-- reconciled with the cache below.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS plan_statement_log (
    sequence              INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_id          TEXT NOT NULL UNIQUE,
    kind                  TEXT NOT NULL CHECK (kind IN
        ('session', 'constraint', 'conditional', 'anchor', 'stance')),
    scope_json             TEXT NOT NULL,
    statement_kind         TEXT NOT NULL CHECK (statement_kind IN
        ('stated', 'withdrawal')),
    statement_json         TEXT NOT NULL,
    provenance_kind        TEXT NOT NULL CHECK (provenance_kind IN
        ('conversation_turn', 'parsed')),
    conversation_turn_id   TEXT
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    parsed_file            TEXT,
    parsed_line            INTEGER,
    effective_start        TEXT NOT NULL,
    effective_end          TEXT,
    include_start          INTEGER NOT NULL CHECK (include_start IN (0, 1)),
    include_end            INTEGER NOT NULL CHECK (include_end IN (0, 1)),
    enforced_from          TEXT,
    acceptance_date        TEXT,
    payload_json           TEXT NOT NULL,
    recorded_at            TEXT NOT NULL,
    CHECK (
        (provenance_kind = 'conversation_turn'
         AND conversation_turn_id IS NOT NULL
         AND parsed_file IS NULL AND parsed_line IS NULL)
        OR
        (provenance_kind = 'parsed'
         AND conversation_turn_id IS NULL
         AND parsed_file IS NOT NULL AND parsed_line IS NOT NULL)
    ),
    CHECK (parsed_line IS NULL OR parsed_line > 0)
);

CREATE INDEX IF NOT EXISTS idx_plan_statement_log_effective
    ON plan_statement_log (effective_start, effective_end, sequence);
CREATE INDEX IF NOT EXISTS idx_plan_statement_log_scope
    ON plan_statement_log (scope_json, sequence);

CREATE TRIGGER IF NOT EXISTS plan_statement_log_no_update
BEFORE UPDATE ON plan_statement_log
BEGIN
    SELECT RAISE(ABORT,
        'plan statement log is append-only; insert a superseding statement');
END;

CREATE TRIGGER IF NOT EXISTS plan_statement_log_no_delete
BEFORE DELETE ON plan_statement_log
BEGIN
    SELECT RAISE(ABORT,
        'plan statement log is append-only; retain the log');
END;

-- ---------------------------------------------------------------------------
-- approval_tokens: opaque, single-use approvals owned by this vault.
--
-- The client decides what the values mean and which values are eligible for
-- minting. The engine only stores them and atomically consumes one alongside
-- a plan statement append (see plan_log.append_rule_spending_token).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS approval_tokens (
    token_id       TEXT PRIMARY KEY,
    statement_hash TEXT NOT NULL,
    turn_id        TEXT NOT NULL,
    flow           TEXT NOT NULL,
    minted_at      TEXT NOT NULL,
    spent_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_approval_tokens_audit
    ON approval_tokens (statement_hash, turn_id, minted_at);

-- plan_week_log: one immutable metadata declaration per Week.
--
-- Policy and Week provenance are facts about the Week, not attributes of the
-- last rule appended to it. Keeping them here avoids duplicating policy on
-- every statement and makes a projection fail closed when its declaration is
-- absent rather than silently inventing metadata.
CREATE TABLE IF NOT EXISTS plan_week_log (
    week_start             TEXT PRIMARY KEY,
    grading_policy_json    TEXT NOT NULL,
    provenance_kind        TEXT NOT NULL CHECK (provenance_kind IN
        ('conversation_turn', 'parsed')),
    conversation_turn_id   TEXT
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    parsed_file            TEXT,
    parsed_line            INTEGER,
    recorded_at            TEXT NOT NULL,
    CHECK (
        (provenance_kind = 'conversation_turn'
         AND conversation_turn_id IS NOT NULL
         AND parsed_file IS NULL AND parsed_line IS NULL)
        OR
        (provenance_kind = 'parsed'
         AND conversation_turn_id IS NULL
         AND parsed_file IS NOT NULL AND parsed_line IS NOT NULL)
    ),
    CHECK (parsed_line IS NULL OR parsed_line > 0)
);

CREATE TRIGGER IF NOT EXISTS plan_week_log_no_update
BEFORE UPDATE ON plan_week_log
BEGIN
    SELECT RAISE(ABORT,
        'plan Week metadata is append-only; a week declaration cannot be updated');
END;

CREATE TRIGGER IF NOT EXISTS plan_week_log_no_delete
BEFORE DELETE ON plan_week_log
BEGIN
    SELECT RAISE(ABORT,
        'plan Week metadata is append-only; retain the declaration');
END;

-- ---------------------------------------------------------------------------
-- plan_projections: rebuildable typed plan envelopes, separate from the log.
--
-- The payload contains the complete Week projection, including its schema
-- version and rule provenance.  These columns keep the provenance edge
-- queryable and enforce the same two legal forms at the storage boundary.
-- A parsed historical projection has no conversation-turn edge and is never
-- retroactively enforceable; its rules carry enforced_from = NULL in payload.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS plan_projections (
    projection_id         TEXT PRIMARY KEY,
    week_start             TEXT NOT NULL,
    payload_json           TEXT NOT NULL,
    schema_version         INTEGER NOT NULL,
    conversation_turn_id   TEXT
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    parsed_file            TEXT,
    parsed_line            INTEGER,
    created_at             TEXT NOT NULL,
    CHECK (
        (conversation_turn_id IS NOT NULL AND parsed_file IS NULL AND parsed_line IS NULL)
        OR
        (conversation_turn_id IS NULL AND parsed_file IS NOT NULL AND parsed_line IS NOT NULL)
    ),
    CHECK (parsed_line IS NULL OR parsed_line > 0)
);

CREATE INDEX IF NOT EXISTS idx_plan_projections_week
    ON plan_projections (week_start, created_at, projection_id);
CREATE INDEX IF NOT EXISTS idx_plan_projections_conversation
    ON plan_projections (conversation_turn_id)
    WHERE conversation_turn_id IS NOT NULL;

CREATE TRIGGER IF NOT EXISTS plan_projections_no_update
BEFORE UPDATE ON plan_projections
BEGIN
    SELECT RAISE(ABORT,
        'plan projections are rebuild-only; insert a new projection');
END;

-- ---------------------------------------------------------------------------
-- Review workflow (M4, #106): durable checkpoints over the conversational
-- event log. Design decision (2026-08-25): **a review IS a
-- conversation.** There is no second event log — `conversation_turns` is it,
-- and every table below is a REBUILDABLE PROJECTION of that log, never a
-- second source of truth. Missing, corrupt or disagreeing: discarded and
-- rebuilt from the log, never merged, never repaired in place
-- (`health_advisor.review.rebuild_projections`).
--
-- Every column below must be derivable from `conversation_turns` alone; a
-- column that is not would be a silent data-loss bug on rebuild. Two kinds
-- of turn carry that log:
--   * literal dialogue — a question put to the user, the user's answer, an
--     agenda decision in the user's own words, the user's acceptance of next
--     week — stored as ordinary role='assistant'/'user' turns, so "user
--     turns" in a review means exactly what it means in chat.py, and the
--     projection stores metadata only, never a copy of the message text;
--   * structural bookkeeping with no natural dialogue line of its own (a
--     step transition, a close attempt, closing) — stored as a turn with
--     role='review_event' and a JSON `content` payload {"event": ..., ...}.
-- Every table here is insert-only: "answered", "decided", "accepted" and
-- "closed" are each a second row's presence, never an UPDATE of the first.
-- ---------------------------------------------------------------------------

-- review_checkpoints: one durable row per step entered (§7.3 steps 1-7; step
-- 8, the claims register, is explicitly out of scope for this table — see
-- the M4 gate, #114 decision 10). Current step for a conversation is
-- MAX(step); resuming after a restart is just reading this table.
CREATE TABLE IF NOT EXISTS review_checkpoints (
    conversation_turn_id  TEXT PRIMARY KEY
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    conversation_id       TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE RESTRICT,
    step                  INTEGER NOT NULL CHECK (step BETWEEN 1 AND 7),
    entered_at            TEXT NOT NULL,
    UNIQUE (conversation_id, step)
);
CREATE INDEX IF NOT EXISTS idx_review_checkpoints_conversation
    ON review_checkpoints (conversation_id, step);

-- review_questions: one row per question put to the user — the four standing
-- checks (pain, energy/stress, nutrition, catch-all), one per data anomaly,
-- or a follow-up thread the user opened mid-interview. `asked_turn_id` IS the
-- question's identity: the assistant turn carrying the literal prompt text.
-- The prompt itself is never duplicated here — read it via the turn.
CREATE TABLE IF NOT EXISTS review_questions (
    asked_turn_id    TEXT PRIMARY KEY
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    conversation_id  TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE RESTRICT,
    kind             TEXT NOT NULL CHECK (kind IN ('standing', 'session_confirmation', 'anomaly', 'followup')),
    anomaly_ref      TEXT,
    asked_at         TEXT NOT NULL,
    CHECK ((kind = 'anomaly') = (anomaly_ref IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_review_questions_conversation
    ON review_questions (conversation_id, kind);

-- review_question_answers: presence of a row means the question is answered.
-- Separate table (rather than an UPDATE of review_questions) so a question
-- row is never rewritten. answered_turn_id is the user's literal reply turn.
CREATE TABLE IF NOT EXISTS review_question_answers (
    asked_turn_id     TEXT PRIMARY KEY
        REFERENCES review_questions(asked_turn_id) ON DELETE RESTRICT,
    answered_turn_id  TEXT NOT NULL UNIQUE
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    conversation_id   TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE RESTRICT,
    answered_at       TEXT NOT NULL
);

-- review_agenda_items: one open-agenda item at a time (§7.3 step 5).
-- `opened_turn_id` carries the item text as its content, same pattern as
-- review_questions.asked_turn_id.
CREATE TABLE IF NOT EXISTS review_agenda_items (
    opened_turn_id   TEXT PRIMARY KEY
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    conversation_id  TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE RESTRICT,
    opened_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_agenda_items_conversation
    ON review_agenda_items (conversation_id);

-- review_agenda_decisions: presence of a row means the item is decided. The
-- decision is recorded "with the user's own words" (§7.3 step 5) by pointing
-- at their literal turn, never by copying it.
CREATE TABLE IF NOT EXISTS review_agenda_decisions (
    opened_turn_id   TEXT PRIMARY KEY
        REFERENCES review_agenda_items(opened_turn_id) ON DELETE RESTRICT,
    decided_turn_id  TEXT NOT NULL UNIQUE
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    conversation_id  TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE RESTRICT,
    decided_at       TEXT NOT NULL
);

-- review_week_acceptance: presence of a row means the user has accepted the
-- proposed next week at least once (§7.3 step 7). This alone does not permit
-- closing — attempt_close in review.py also requires no open question and no
-- undecided agenda item at the moment of the attempt, which is what lets
-- negotiation continue after this row already exists ("step 8 must not close
-- a review the user is still negotiating").
CREATE TABLE IF NOT EXISTS review_week_acceptance (
    conversation_id   TEXT PRIMARY KEY
        REFERENCES conversations(id) ON DELETE RESTRICT,
    accepted_turn_id  TEXT NOT NULL
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    accepted_at       TEXT NOT NULL
);

-- review_week_proposals: typed proposal/revision lifecycle projection. Each
-- row is one event-backed version state; no row is updated. The complete Week
-- and structured diff are in the event payload and mirrored here for a fast
-- client read. Rebuild replays these rows from conversation_turns alone.
CREATE TABLE IF NOT EXISTS review_week_proposals (
    projection_id       TEXT PRIMARY KEY,
    event_turn_id       TEXT NOT NULL
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    conversation_id     TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE RESTRICT,
    proposal_id         TEXT NOT NULL,
    version             INTEGER NOT NULL CHECK (version > 0),
    lifecycle           TEXT NOT NULL CHECK (lifecycle IN
        ('proposed', 'accepted', 'voided')),
    week_start          TEXT NOT NULL,
    week_json           TEXT NOT NULL,
    current_week_json   TEXT NOT NULL,
    diff_json           TEXT NOT NULL,
    classification_json TEXT NOT NULL,
    review_sections_json TEXT NOT NULL,
    review_turn_id      TEXT NOT NULL
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    supersedes_proposal_id TEXT,
    supersedes_version  INTEGER,
    recorded_at         TEXT NOT NULL,
    UNIQUE (conversation_id, proposal_id, version, lifecycle, event_turn_id)
);
CREATE INDEX IF NOT EXISTS idx_review_week_proposals_current
    ON review_week_proposals (conversation_id, proposal_id, version, recorded_at);

-- ---------------------------------------------------------------------------
-- retro_claims: attribution of an observed HealthKit workout to a proposed
-- session.  This is not a grading/enforcement projection: it records what
-- Python judged about the selected workout at claim time.  The source of the
-- claim is the conversation event turn, while proposal_id/version identify
-- the existing ProposedWeek being attributed to.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS retro_claims (
    claim_id              TEXT PRIMARY KEY,
    provenance_kind       TEXT NOT NULL CHECK (provenance_kind = 'conversation_turn'),
    provenance_turn_id    TEXT NOT NULL
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    conversation_id       TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE RESTRICT,
    proposal_id           TEXT NOT NULL,
    proposal_version      INTEGER NOT NULL CHECK (proposal_version > 0),
    week_start            TEXT NOT NULL,
    session_id            TEXT NOT NULL,
    proposed_day          TEXT NOT NULL,
    workout_id            INTEGER NOT NULL
        REFERENCES workouts(id) ON DELETE RESTRICT,
    verdict               TEXT NOT NULL CHECK (verdict IN
        ('supported', 'partial', 'unsupported')),
    proposed_jog_minutes  REAL,
    observed_jog_minutes  REAL,
    weekly_jog_minutes    REAL NOT NULL,
    weekly_jog_ceiling    REAL NOT NULL,
    volume_delta          REAL NOT NULL CHECK (volume_delta = 0),
    claim_text            TEXT NOT NULL,
    recorded_at           TEXT NOT NULL,
    UNIQUE (proposal_id, proposal_version, session_id, workout_id)
);
CREATE INDEX IF NOT EXISTS idx_retro_claims_proposal
    ON retro_claims (proposal_id, proposal_version, proposed_day);
CREATE INDEX IF NOT EXISTS idx_retro_claims_provenance
    ON retro_claims (provenance_turn_id);

CREATE TRIGGER IF NOT EXISTS retro_claims_no_update
BEFORE UPDATE ON retro_claims
BEGIN
    SELECT RAISE(ABORT,
        'retro claims are append-only; retain the original attribution');
END;

CREATE TRIGGER IF NOT EXISTS retro_claims_no_delete
BEFORE DELETE ON retro_claims
BEGIN
    SELECT RAISE(ABORT,
        'retro claims are append-only; retain the original attribution');
END;

-- review_status: presence of a row means the review is closed; absence means
-- in_progress. This is the only place "closed" is recorded, and it is never
-- updated — a review is closed at most once.
CREATE TABLE IF NOT EXISTS review_status (
    conversation_id  TEXT PRIMARY KEY
        REFERENCES conversations(id) ON DELETE RESTRICT,
    closed_turn_id   TEXT NOT NULL
        REFERENCES conversation_turns(id) ON DELETE RESTRICT,
    closed_at        TEXT NOT NULL
);

-- Nutrition reference: item -> macros per serving, with provenance.
--
-- Populated organically — an item enters the catalog the first time the user names
-- something that isn't in it. Rows may originate from a label photo, from typed
-- label text, from a web lookup, or from reasoning about a similar product, and
-- those four are NOT equally trustworthy. `source` records which, and every read
-- returns it, so a number derived from a guess can never present itself as a
-- measurement. `source_detail` is required for the same reason `manual_jog.why`
-- is required: an entry nobody can audit does not get to exist.
--
-- Writes are a FULL REPLACE, not the partial upsert `subjective` uses. That is
-- right for check-ins, which arrive as drip-fed corrections, and wrong here: a
-- catalog row is one complete statement read off one label. Under partial
-- semantics a row could be relabelled estimate -> web while its omitted macros
-- kept their guessed values, and the whole row would then read as web-sourced.
CREATE TABLE IF NOT EXISTS food_catalog (
    item_key      TEXT PRIMARY KEY,   -- slug: 'tj-strained-greek-yogurt-plain'
    display_name  TEXT NOT NULL,
    brand         TEXT,               -- "Trader Joe's"; NULL for generics
    aliases       TEXT,               -- pipe-separated: 'greek yogurt|strained yogurt'
    serving_desc  TEXT NOT NULL,      -- '3/4 cup', '2 tortillas', '4 oz raw'
    serving_g     REAL,               -- NULL when the label gives no gram weight
    kcal          REAL NOT NULL,
    protein_g     REAL,
    carb_g        REAL,
    fat_g         REAL,
    source        TEXT NOT NULL
        CHECK (source IN ('label_photo','label_text','web','estimate')),
    source_detail TEXT NOT NULL,      -- URL, what the photo showed, or the reasoning
    confirmed     INTEGER NOT NULL DEFAULT 0
        CHECK (confirmed IN (0,1)),   -- 1 = the user saw these numbers and said yes
    verified_at   TEXT NOT NULL,      -- when these numbers were last established
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_food_catalog_name ON food_catalog (display_name);
