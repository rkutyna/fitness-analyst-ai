"""Build and inspect the per-user SQLite vault.

The vault keeps every derived table and workout, plus allowlisted raw
``records`` and a transient tail of other live series until compaction. This
module is the one source of truth for that raw-series contract; the command-line
wrapper imports it rather than repeating SQL or a second allowlist. A build also
declares the source history's final local date so HealthKit-direct ingest cannot
silently append the imported history; the explicit setter is reserved for a
deliberate re-derivation migration.
"""
from __future__ import annotations

import gzip
import math
import os
import sqlite3
import tempfile
import time
from datetime import date, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import db
from . import metrics


# D3 raw-series contract.  Keep the reason beside each entry so adding a
# sample-level consumer requires an explicit review of what enters the vault.
VAULT_RAW_SERIES = frozenset({
    # metrics.bucket_series, HR zones, hr_load, and benchmark need raw heart rate.
    "heart_rate",
    # metrics.bucket_series, impact_volume, and longest_block classify movement buckets.
    "distance_walking_running",
    # running_form.monthly_running_power aggregates 20-second power buckets.
    "running_power",
    # derive.py sessionizes and re-attributes the sleep interval streams.
    "sleep_asleep",
    # derive.py counts awake interruptions inside sleep sessions.
    "sleep_awake",
    # derive.py uses the in-bed interval to establish the sleep session boundary.
    "sleep_in_bed",
    # get_intraday's time-of-day pattern. Nothing derives from raw steps — the
    # training dial reads distance and heart rate — so this is here for a user
    # question rather than a computation, and it is affordable only because it
    # is bucketed (D9): 884,431 samples become 142,419 five-minute buckets,
    # +53 MB once and +8 MB/year. Decided 2026-08-22, see #16.
    "step_count",
})

# Series the vault stores at a coarser resolution than the phone recorded them,
# with the bucket width in seconds. See ARCHITECTURE.md D9 for the decision and,
# more importantly, for why it is reversible: the vault is a COPY. HealthKit on
# the device is the source of truth, so a series that later needs its original
# resolution is recovered by re-ingesting it, not by regretting this.
#
# The bucket width is not a guess — it is the width every consumer of the series
# already reduces to (`metrics.IMPACT_BUCKET_SECONDS`). Storing finer than the
# coarsest consumer reads is storage nobody is using.
VAULT_BUCKET_SECONDS: dict[str, int] = {
    # 1 sample/second, ~2.7 ft each, 9,950 rows/day — 89% of all vault growth.
    # bucket_series, impact_volume and longest_block all reduce it to this same
    # width, which is why the constant is IMPORTED rather than repeated: two
    # copies of one number is how the vault ends up storing a resolution its
    # consumers no longer read.
    "distance_walking_running": metrics.IMPACT_BUCKET_SECONDS,
    # `get_intraday` rejects `bucket_hours < 1`, so an hour is the finest
    # question anything can ask of this series — five minutes is twelve times
    # finer than that and leaves room for a tool that wants better, at a cost
    # (+8 MB/year) small enough not to need re-deciding.
    "step_count": 300,
}


# --------------------------------------------------------------------------- #
# The D3 contract, as something code can ask rather than a rule in prose
# --------------------------------------------------------------------------- #
# The key `build_vault` writes into its destination, and the only thing that
# makes a database D3-filtered. It is deliberately NOT `schema_version` or
# `created_at`: `init_db` stamps those into every database it touches, so
# keying off them would conflate "has been initialised" with "is a vault" —
# and then a full snapshot opened by backfill, and every test database, would
# have to opt back out. A marker only one function writes needs no exceptions.
VAULT_DECLARATION = "d3_filtered"

# Origins the D3 allowlist governs: rows that came off a device, which is the
# firehose D3 exists to bound. `checkin` is deliberately absent — those six
# series are six rows a day the user typed, mirrored into `records` so the
# analysis stack can see them (subjective.py). They are not a copy of anything;
# `records` is where they originate, and no allowlist can be their gate without
# making the nightly check-in fail against a real vault.
#
# Consequence worth knowing: build_vault DROPS check-in raw rows (they are not
# in VAULT_RAW_SERIES) while their daily aggregates travel, and live check-ins
# then write raw rows again after the build. A vault therefore holds subjective
# raw rows only for days since its last build. That is harmless because these
# are daily data and daily_metrics carries the same values — but it is why the
# two facts look inconsistent if you go looking.
D3_GOVERNED_ORIGINS = frozenset({"backfill", "receiver", "healthkit"})


def declare_vault(conn: sqlite3.Connection) -> None:
    """Mark this database as D3-filtered. `build_vault` calls this; nothing else
    should. A snapshot that gains this key gets the D3 retention contract."""
    conn.execute(
        "INSERT OR IGNORE INTO vault_meta (key, value) VALUES (?, '1')",
        (VAULT_DECLARATION,),
    )


def is_vault(conn: sqlite3.Connection) -> bool:
    """Whether this database declares itself D3-filtered.

    A full snapshot has the same tables and columns as a vault and differs only
    in this declaration — which is why the declaration exists rather than a
    filename, a path, or an argument the caller passes, all of which are things
    a caller can get wrong.
    """
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM vault_meta WHERE key = ? AND value <> ''",
            (VAULT_DECLARATION,),
        ).fetchone()
    except sqlite3.Error:
        return False
    return bool(row[0])


def compacted_through(conn: sqlite3.Connection) -> str | None:
    """Return the D3 raw-compaction watermark, if this vault has one."""
    row = conn.execute(
        "SELECT value FROM vault_meta WHERE key = 'compacted_through'"
    ).fetchone()
    return row[0] if row else None


def history_imported_through(conn: sqlite3.Connection) -> str | None:
    """Return the declared imported-history watermark, if this vault has one."""
    row = conn.execute(
        "SELECT value FROM vault_meta WHERE key = 'history_imported_through'"
    ).fetchone()
    return row[0] if row else None


def set_history_imported_through(
    conn: sqlite3.Connection, through: str | None
) -> None:
    """Deliberately move or clear the imported-history watermark.

    ``None`` clears the declaration. The caller owns the surrounding
    transaction and must commit it. This is intentionally an explicit
    migration operation: normal HealthKit ingest has no switch that can bypass
    the guard.
    """
    if through is None:
        conn.execute(
            "DELETE FROM vault_meta WHERE key = 'history_imported_through'"
        )
        return
    try:
        parsed = date.fromisoformat(through)
    except (TypeError, ValueError):
        raise ValueError(
            "history_imported_through must be an ISO local date (YYYY-MM-DD)"
        ) from None
    if parsed.isoformat() != through:
        raise ValueError(
            "history_imported_through must be an ISO local date (YYYY-MM-DD)"
        )
    conn.execute(
        "INSERT INTO vault_meta (key, value) "
        "VALUES ('history_imported_through', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (through,),
    )


# Per-vault settings (T-032). These live in `vault_meta` beside `owner` and
# `history_imported_through` because they are the same kind of thing: a
# declaration the vault makes about itself.
#
# There is deliberately NO module-level fallback applied on read. An undeclared
# setting reads as None and the caller decides — exactly as `owner` does, where
# an unclaimed vault is allowed through rather than defaulted to somebody. A
# convenience default here would work perfectly until the second vault, which
# is the defect T-003 and D17 both forbid.

UNIT_SYSTEMS: dict[str, dict[str, str]] = {
    # What the stored values already ARE, not a display preference: `records`
    # and `workouts` carry miles, pounds and kcal in their column names
    # (`distance_mi`, `energy_kcal`). Declaring the system does not convert
    # anything -- changing what unit a stored value is in is a migration
    # against ten years of records, not a settings change.
    "imperial": {"distance": "mi", "mass": "lb", "energy": "kcal"},
    "metric": {"distance": "km", "mass": "kg", "energy": "kJ"},
}


def local_timezone(conn: sqlite3.Connection) -> str | None:
    """The IANA zone this vault's local dates are declared in, or None.

    Named `local_timezone` rather than `timezone` so that a later
    `from datetime import timezone` in this module cannot shadow it, and
    because it pairs with the `local_date` column it describes.

    None means undeclared, NOT UTC and not a default. Historical samples are
    attributed by the per-sample offset carried in the export, and that stays
    the fallback -- removing it would silently re-attribute a decade of data.
    """
    row = conn.execute(
        "SELECT value FROM vault_meta WHERE key = 'local_timezone'").fetchone()
    return row[0] if row else None


def set_local_timezone(conn: sqlite3.Connection, zone: str | None) -> None:
    """Declare (or clear) the vault's zone. Caller owns the transaction.

    Validated against the system tz database so a typo cannot be stored and
    then silently mis-attribute dates later.
    """
    if zone is None:
        conn.execute("DELETE FROM vault_meta WHERE key = 'local_timezone'")
        return
    try:
        ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise ValueError(f"unknown IANA time zone: {zone!r}") from None
    conn.execute(
        "INSERT INTO vault_meta (key, value) VALUES ('local_timezone', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (zone,))


def unit_system(conn: sqlite3.Connection) -> str | None:
    """The declared unit system name, or None if this vault has not said."""
    row = conn.execute(
        "SELECT value FROM vault_meta WHERE key = 'unit_system'").fetchone()
    return row[0] if row else None


def set_unit_system(conn: sqlite3.Connection, name: str | None) -> None:
    """Declare (or clear) the vault's unit system. Caller owns the transaction."""
    if name is None:
        conn.execute("DELETE FROM vault_meta WHERE key = 'unit_system'")
        return
    if name not in UNIT_SYSTEMS:
        raise ValueError(
            f"unknown unit system: {name!r} (known: {sorted(UNIT_SYSTEMS)})")
    conn.execute(
        "INSERT INTO vault_meta (key, value) VALUES ('unit_system', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (name,))


def _vault_meta_value(conn: sqlite3.Connection, key: str) -> str | None:
    """One vault_meta value, or None when unset OR when the table does not exist.

    A vault written before vault_meta existed, or any vault opened read-only
    before its first writable initialisation, has no vault_meta table at all.
    Such a vault has no settings by construction, and the legacy defaults
    apply; raising here would take every arbitration read down with it
    (measured 2026-09-03 on the reference snapshot).
    """
    try:
        row = conn.execute(
            "SELECT value FROM vault_meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return None
        raise
    return row[0] if row else None


def workout_source_arbitration_from(conn: sqlite3.Connection) -> str | None:
    """The vault's declared workout-arbitration start date, or ``None``."""
    row = _vault_meta_value(conn, "workout_source_arbitration_from")
    return row


def set_workout_source_arbitration_from(
    conn: sqlite3.Connection, through: str | None,
) -> None:
    """Declare (or clear) the vault's workout-arbitration start date."""
    if through is None:
        conn.execute(
            "DELETE FROM vault_meta WHERE key = 'workout_source_arbitration_from'"
        )
        return
    try:
        parsed = date.fromisoformat(through)
    except (TypeError, ValueError):
        raise ValueError(
            "workout_source_arbitration_from must be an ISO local date "
            "(YYYY-MM-DD)"
        ) from None
    if parsed.isoformat() != through:
        raise ValueError(
            "workout_source_arbitration_from must be an ISO local date "
            "(YYYY-MM-DD)"
        )
    conn.execute(
        "INSERT INTO vault_meta (key, value) VALUES "
        "('workout_source_arbitration_from', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (through,))


def block_qualify_hr_max(conn: sqlite3.Connection) -> float | None:
    """The vault's declared qualifying-block HR ceiling, or ``None``."""
    row = _vault_meta_value(conn, "block_qualify_hr_max")
    return float(row) if row is not None else None


def set_block_qualify_hr_max(
    conn: sqlite3.Connection, ceiling: float | None,
) -> None:
    """Declare (or clear) the vault's qualifying-block HR ceiling."""
    if ceiling is None:
        conn.execute("DELETE FROM vault_meta WHERE key = 'block_qualify_hr_max'")
        return
    try:
        ceiling = float(ceiling)
    except (TypeError, ValueError):
        raise ValueError("block_qualify_hr_max must be a finite number") from None
    if not math.isfinite(ceiling) or ceiling <= 0 or ceiling >= 300:
        raise ValueError("block_qualify_hr_max must be a finite bpm ceiling")
    conn.execute(
        "INSERT INTO vault_meta (key, value) VALUES ('block_qualify_hr_max', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (str(ceiling),))


def units(conn: sqlite3.Connection) -> dict[str, str] | None:
    """Concrete unit names for this vault, or None if undeclared.

    A copy, so a caller cannot mutate the shared vocabulary.
    """
    name = unit_system(conn)
    return dict(UNIT_SYSTEMS[name]) if name else None


def uncompacted_violations(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return non-allowlisted raw rows that are behind the compaction watermark."""
    through = compacted_through(conn)
    if through is None:
        return []
    placeholders = ",".join("?" * len(VAULT_RAW_SERIES))
    return conn.execute(
        "SELECT * FROM records "
        f"WHERE metric NOT IN ({placeholders}) AND local_date <= ? "
        "ORDER BY local_date, metric, id",
        (*sorted(VAULT_RAW_SERIES), through),
    ).fetchall()


def compaction_status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Report whether this vault has ever compacted and whether it is clean.

    ``uncompacted_violations`` cannot distinguish an unset watermark from a
    clean vault: without a watermark there is no boundary against which to
    compare. This status keeps that distinction explicit and counts both the
    complete non-allowlisted raw tail and the portion behind an existing
    watermark. An undeclared database is reported as a state rather than being
    treated as a failed vault check.
    """
    declared = is_vault(conn)
    try:
        through = compacted_through(conn)
    except sqlite3.Error:
        # A plain snapshot may predate vault_meta entirely. It is still useful
        # to report its raw-row evidence, but it is not a declared vault.
        through = None
    placeholders = ",".join("?" * len(VAULT_RAW_SERIES))
    allowlist = tuple(sorted(VAULT_RAW_SERIES))
    total = conn.execute(
        "SELECT COUNT(*) FROM records "
        f"WHERE metric NOT IN ({placeholders})",
        allowlist,
    ).fetchone()[0]
    behind = 0
    if through is not None:
        behind = conn.execute(
            "SELECT COUNT(*) FROM records "
            f"WHERE metric NOT IN ({placeholders}) AND local_date <= ?",
            (*allowlist, through),
        ).fetchone()[0]

    if not declared:
        status = "not_a_declared_vault"
    elif through is None:
        status = "never_compacted"
    elif behind:
        status = "compacted_violated"
    else:
        status = "compacted_clean"

    return {
        "status": status,
        "is_declared_vault": declared,
        "watermark_exists": through is not None,
        "compacted_through": through,
        "non_allowlisted_raw_total": total,
        "non_allowlisted_raw_behind_watermark": behind,
    }


COMPACTION_CHUNK = 10_000


def compact(conn: sqlite3.Connection, *, through: str) -> dict[str, int]:
    """Delete transient raw rows through ``through`` and advance the watermark.

    Rows are deleted in short transactions because the database uses SQLite's
    DELETE journal mode. A repeated or older compaction is a logged no-op; the
    watermark is monotonic and only moves after all selected rows are gone.
    """
    if not is_vault(conn):
        raise ValueError("compaction requires a declared vault")

    previous = compacted_through(conn)
    if previous is not None and through <= previous:
        db.log_ingest(
            conn, "vault", "compact", 0, 0,
            f"through={through} watermark={previous} deleted=none",
        )
        return {}

    placeholders = ",".join("?" * len(VAULT_RAW_SERIES))
    params = (*sorted(VAULT_RAW_SERIES), through)
    deleted: dict[str, int] = {}
    while True:
        rows = conn.execute(
            "SELECT id, metric FROM records "
            f"WHERE metric NOT IN ({placeholders}) AND local_date <= ? "
            "ORDER BY id LIMIT ?",
            (*params, COMPACTION_CHUNK),
        ).fetchall()
        if not rows:
            break
        conn.executemany("DELETE FROM records WHERE id = ?",
                         [(row["id"],) for row in rows])
        for row in rows:
            deleted[row["metric"]] = deleted.get(row["metric"], 0) + 1
        conn.commit()

    conn.execute(
        "INSERT INTO vault_meta (key, value) VALUES ('compacted_through', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value "
        "WHERE vault_meta.value < excluded.value",
        (through,),
    )
    total = sum(deleted.values())
    detail = "through=" + through + " " + (
        " ".join(f"{metric}={n}" for metric, n in sorted(deleted.items()))
        if deleted else "deleted=none"
    )
    db.log_ingest(conn, "vault", "compact", total, total, detail)
    conn.commit()
    return deleted


def raw_series_available(metric: str) -> bool:
    """Whether sample-level rows for ``metric`` exist in a D3 vault at all."""
    return metric in VAULT_RAW_SERIES


def raw_resolution_seconds(metric: str) -> int:
    """The finest resolution the vault holds for ``metric``. 0 means as recorded."""
    return VAULT_BUCKET_SECONDS.get(metric, 0)


def raw_unavailable(metric: str, *, needed_for: str) -> dict:
    """The answer a raw-dependent computation gives for a non-allowlisted series.

    Not an error and not an empty result. ``readiness`` returning
    ``establishing_baseline`` and ``coverage`` returning ``stale`` are answers;
    this joins that vocabulary. The distinction matters more here than anywhere:
    an empty series that actually means "these samples are not in this vault"
    reads as "you did not do that", which is how the VO2max ingest defect stayed
    invisible for weeks.
    """
    return {
        "status": "unavailable",
        "metric": metric,
        "needed_for": needed_for,
        "reason": "raw_series_not_in_vault",
        "detail": (
            f"{metric!r} has no sample-level rows here: the vault ships raw "
            f"samples only for the series that sample-level analysis needs "
            f"(D3). Daily aggregates for {metric!r} are present and usable — "
            f"use get_daily_series or summarize_metric. This is a property of "
            f"the vault, not of your data."
        ),
        "raw_series_in_vault": sorted(VAULT_RAW_SERIES),
    }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
# `vault_meta` is deliberately NOT here. It is the destination's identity —
# whose vault this is, what shape it was built in, when — and copying it from a
# source that may be a plain snapshot would import a stranger's answer to all
# three. Owner and epoch carry across a REPLACE instead, from the vault being
# replaced, because those are history rather than content.
#
# `hk_sync_state` is NOT here for a sharper version of the same reason, and it
# is the more dangerous of the two. An anchor is a claim: "this vault has
# consumed HealthKit type T up to point A, do not send it again." This build
# *drops* raw records for every metric outside `_streamed_raw_series()` (D3).
# Copy the anchor and the destination asserts it holds samples this very
# function threw away — the phone resumes past them and they are gone, per
# metric, permanently, presenting as a series that simply has no data. The
# anchor belongs to whoever actually holds the samples, so a rebuilt vault
# starts with none and resyncs.
#
# `hk_deletions` IS here, and the asymmetry is deliberate: a tombstone is a
# fact about what was removed, and losing it lets a stale replayed batch
# resurrect a sample the user deleted.
_COPY_ORDER = (
    # workout_weather has a foreign key to workouts.id.
    "commit_log",
    "hk_deletions",
    "workouts",
    "records",
    "workout_events",
    "daily_metrics",
    # Derived provenance. It is here rather than reconstructed on the far side
    # precisely because most raw `records` do not travel — see schema.sql.
    "metric_source_months",
    "insights",
    "ingest_log",
    "subjective",
    "workout_weather",
    "manual_jog",
    "food_catalog",
    "benchmark",
)


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _history_at(
    path: Path,
) -> tuple[int, list, str | None, str | None, str | None, int, list, list, dict]:
    """History declarations of an existing vault, to survive a rebuild.

    These are things the SOURCE cannot know: who the vault belongs to, how far
    its commit history got, which commits already landed, through which local
    date its imported history is declared complete, the earliest day on which
    it has already accepted a HealthKit-direct sample, and the live-acquired
    consolidated totals and their revision history. Rebuilding replaces the
    data, not the identity or the live facts received after the snapshot.
    """
    conn = db.connect(path, read_only=True)
    try:
        try:
            commits = [tuple(r) for r in conn.execute(
                "SELECT key, epoch, applied_at, detail FROM commit_log")]
        except sqlite3.Error:          # a vault predating commit_log
            commits = []
        try:
            row = conn.execute(
                "SELECT value FROM vault_meta WHERE key = 'owner'").fetchone()
        except sqlite3.Error:
            row = None
        owner = row[0] if row else None
        try:
            row = conn.execute(
                "SELECT value FROM vault_meta "
                "WHERE key = 'history_imported_through'"
            ).fetchone()
        except sqlite3.Error:
            row = None
        history = row[0] if row else None
        try:
            row = conn.execute(
                "SELECT MIN(local_date) FROM records "
                "WHERE hk_uuid IS NOT NULL").fetchone()
        except sqlite3.Error:
            row = None
        live_floor = row[0] if row and row[0] is not None else None
        try:
            existing_totals = conn.execute(
                "SELECT COUNT(*) FROM hk_daily_totals").fetchone()[0]
            total_rows = [tuple(r) for r in conn.execute(
                "SELECT metric, local_date, value, unit, interval, state, "
                "device_id, queried_at, first_seen_at, settled_at "
                "FROM hk_daily_totals")]
        except sqlite3.Error:          # a vault predating D19
            existing_totals, total_rows = 0, []
        try:
            revision_rows = [tuple(r) for r in conn.execute(
                "SELECT id, metric, local_date, from_value, to_value, "
                "from_state, to_state, lag_days, batch_id, recorded_at "
                "FROM hk_daily_total_revisions")]
        except sqlite3.Error:          # a vault predating D19
            revision_rows = []
        try:
            expected = dict(conn.execute(
                "SELECT key, value FROM vault_meta "
                "WHERE key LIKE 'daily_totals_expected_from:%'"))
        except sqlite3.Error:
            expected = {}
        return (
            _user_version(conn), commits, owner, history, live_floor,
            existing_totals, total_rows, revision_rows, expected,
        )
    finally:
        conn.close()


def _tuning_settings_at(path: Path) -> dict[str, str]:
    """Read the two analysis settings that a rebuild must carry forward."""
    conn = db.connect(path, read_only=True)
    try:
        try:
            rows = conn.execute(
                "SELECT key, value FROM vault_meta WHERE key IN "
                "('workout_source_arbitration_from', 'block_qualify_hr_max')"
            ).fetchall()
        except sqlite3.Error:
            return {}
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


def _streamed_raw_series() -> frozenset[str]:
    """Allowlisted series copied sample-for-sample; the rest are bucketed."""
    return VAULT_RAW_SERIES - VAULT_BUCKET_SECONDS.keys()


def _source_history_imported_through(
    source: sqlite3.Connection, source_tables: set[str]
) -> str | None:
    """Find the source's last historical local day for the new vault.

    Raw records are authoritative when present. A vault source can have no raw
    rows after filtering/compaction, while its derived daily history remains;
    in that case its latest daily date is the safe fallback. An entirely empty
    source has no history to guard and therefore returns ``None``.
    """
    if "records" in source_tables:
        row = source.execute("SELECT MAX(local_date) FROM records").fetchone()
        if row and row[0] is not None:
            return row[0]
    if "daily_metrics" in source_tables:
        row = source.execute("SELECT MAX(date) FROM daily_metrics").fetchone()
        if row and row[0] is not None:
            return row[0]
    return None


def _cap_history_to_live_floor(
    history: str | None, live_floor: str | None
) -> str | None:
    """Hold the watermark strictly below the first day of live HealthKit data.

    ``live_floor`` is the earliest ``local_date`` for which the destination has
    already accepted a HealthKit-direct row. Returns ``None`` when there is no
    day left to declare, which leaves the vault unguarded — correct, because a
    vault whose every day carries live data has no imported history to protect.
    """
    if history is None or live_floor is None or history < live_floor:
        return history
    capped = (date.fromisoformat(live_floor) - timedelta(days=1)).isoformat()
    return capped if capped >= "0001-01-01" else None


def _quote(identifier: str) -> str:
    """Quote an identifier after it came from SQLite metadata."""
    return '"' + identifier.replace('"', '""') + '"'


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({_quote(table)})")]


def _batches(rows: Iterable[Sequence[Any]], size: int) -> Iterable[list[Sequence[Any]]]:
    batch: list[Sequence[Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class _ByteCounter:
    """A file-like sink for measuring gzip output without creating an artifact."""

    def __init__(self) -> None:
        self.size = 0

    def write(self, data: bytes) -> int:
        self.size += len(data)
        return len(data)


def _gzip_size(path: Path) -> int:
    sink = _ByteCounter()
    with path.open("rb") as source, gzip.GzipFile(
        fileobj=sink, mode="wb", filename="", mtime=0
    ) as compressed:
        while chunk := source.read(1024 * 1024):
            compressed.write(chunk)
    return sink.size


def _copy_table(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    *,
    batch_size: int,
    allow_metrics: frozenset[str] | None = None,
) -> tuple[int, int, dict[str, int], dict[str, int]]:
    """Copy one table, optionally filtering ``records``.

    The source and target are schema versions from this repository.  Taking
    the common columns makes the copy tolerant of an additive source column;
    the target schema supplies its normal NULL/default for a new column.
    """
    source_columns = _columns(source, table)
    target_columns = _columns(target, table)
    columns = [column for column in target_columns if column in source_columns]
    if not columns:
        return 0, 0, {}, {}

    select = f"SELECT {', '.join(_quote(c) for c in columns)} FROM {_quote(table)}"
    if table == "records":
        select += " ORDER BY id"
    insert = (
        f"INSERT INTO {_quote(table)} ({', '.join(_quote(c) for c in columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})"
    )
    metric_index = columns.index("metric") if table == "records" else None
    seen = copied = 0
    copied_by_metric: dict[str, int] = {}
    dropped_by_metric: dict[str, int] = {}

    cursor = source.execute(select)

    def rows() -> Iterable[Sequence[Any]]:
        nonlocal seen, copied
        for row in cursor:
            if metric_index is None:
                yield tuple(row)
                continue
            seen += 1
            metric = row[metric_index]
            if metric not in allow_metrics:
                dropped_by_metric[metric] = dropped_by_metric.get(metric, 0) + 1
                continue
            copied += 1
            copied_by_metric[metric] = copied_by_metric.get(metric, 0) + 1
            yield tuple(row)

    for batch in _batches(rows(), batch_size):
        target.executemany(insert, batch)

    if metric_index is None:
        copied = target.execute(
            f"SELECT COUNT(*) FROM {_quote(table)}"
        ).fetchone()[0]
        seen = copied
    return seen, copied, copied_by_metric, dropped_by_metric


def _copy_bucketed(source: sqlite3.Connection, target: sqlite3.Connection,
                   metric: str, seconds: int) -> tuple[int, int]:
    """Copy one series into the vault as fixed-width buckets. (seen, written).

    **Grouped per source, not globally.** Collapsing two devices' samples into
    one row would destroy `db._arbitration`, which decides cross-source overlap
    by dropping a mirror source — and that decision is the difference between
    44,965 steps and 430x nonsense. It costs almost nothing to honour: 36,600
    per-source buckets against 36,479 global ones, 0.3% more rows.

    `local_date` is part of the key so a bucket cannot span two local days, and
    `start_utc` keeps the earliest sample in the bucket rather than the bucket's
    nominal start — any timestamp inside the window floors to the same bucket
    index, so a consumer re-bucketing on the same grid gets identical groups,
    and the real instant is more useful than a synthetic one.
    """
    seen = source.execute(
        "SELECT COUNT(*) FROM records WHERE metric = ?", (metric,)).fetchone()[0]
    cursor = source.execute(
        """
        SELECT metric,
               SUM(value)                                        AS value,
               MAX(unit)                                         AS unit,
               MIN(start_utc)                                    AS start_utc,
               MAX(end_utc)                                      AS end_utc,
               MIN(start_local)                                  AS start_local,
               local_date,
               source,
               MIN(origin)                                       AS origin,
               CAST(strftime('%s', start_utc) / ? AS INT)         AS bucket
          FROM records
         WHERE metric = ?
      GROUP BY metric, local_date, source,
               CAST(strftime('%s', start_utc) / ? AS INT)
        """,
        (seconds, metric, seconds),
    )

    def keyed(rows):
        """Swap the bucket index for its hashed key.

        `records.dedupe_key` is documented as a sha1 of the row's natural key,
        and a bucket's natural key is the window rather than a sample. A plain
        composite string sitting among hashes is the kind of difference nobody
        notices until an upsert silently duplicates.
        """
        for row in rows:
            fields = tuple(row)[:-1]
            yield fields + (db.bucket_key(row["metric"], row["local_date"],
                                          row["source"], row["bucket"], seconds),)

    written = 0
    for batch in _batches(keyed(cursor), 10_000):
        target.executemany(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
        written += len(batch)
    return seen, written


def build_vault(
    source_path: str | Path,
    vault_path: str | Path,
    *,
    batch_size: int = 10_000,
    replace: bool = False,
    owner: str | None = None,
    local_timezone: str | None = None,
    unit_system: str | None = None,
    measure_gzip: bool = True,
) -> dict[str, Any]:
    """Stream a full snapshot into a filtered vault and return measurements.

    ``source_path`` is opened through the project's read-only connection.  The
    destination is built in a sibling temporary file and atomically moved into
    place only after it is closed successfully.  Existing destinations require
    ``replace=True`` so a typo cannot silently destroy a vault.

    ``local_timezone`` and ``unit_system`` stamp the vault's declared settings
    (T-032).  Both are validated here, so a typo fails the build rather than
    sitting in `vault_meta` until something mis-attributes a date.  Leaving
    either unset produces a vault that has not declared it, which is what every
    existing snapshot is -- undeclared is a legitimate state, not a missing
    default.

    ``owner`` stamps the vault with the user it belongs to, after which a
    session carrying a different user id is refused at connect
    (:class:`context.VaultOwnershipError`).  Build time is where a per-user
    vault gets its identity; leaving it unset produces an unclaimed vault, which
    is right for a development copy and wrong for a user's.

    The destination also records ``history_imported_through`` from the source's
    latest raw local date (or latest derived date when a vault source has no raw
    rows). Rebuilding preserves the existing declaration and never lowers it.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    source_path = Path(source_path).expanduser().resolve()
    vault_path = Path(vault_path).expanduser().resolve()
    if source_path == vault_path:
        raise ValueError("source and vault must be different files")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if vault_path.exists() and not replace:
        raise FileExistsError(
            f"vault already exists: {vault_path} (pass replace=True/--force to replace it)"
        )

    # A replace inherits two things from the vault it replaces: the fencing
    # epoch, and the set of commit keys already applied. Both are history rather
    # than content — dropping the epoch lets a stale worker back in, and
    # dropping the keys means a retried commit re-applies, which is a duplicate
    # insight and a duplicate provider charge. The *data* is rebuilt from the
    # source; this is what the source cannot know.
    (
        existing_epoch, existing_commits, existing_owner, existing_history,
        live_floor, existing_totals, existing_total_rows,
        existing_revision_rows, existing_expected,
    ) = (
        _history_at(vault_path) if vault_path.exists()
        else (0, [], None, None, None, 0, [], [], {}))
    existing_tuning = (_tuning_settings_at(vault_path)
                       if vault_path.exists() else {})
    owner = owner or existing_owner
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging_name = tempfile.mkstemp(
        prefix=f".{vault_path.name}.", suffix=".tmp", dir=vault_path.parent
    )
    os.close(fd)
    staging_path = Path(staging_name)
    started = time.perf_counter()
    source = target = None
    try:
        source = db.connect(source_path, read_only=True)
        source_tables = _table_names(source)
        source_history = _source_history_imported_through(source, source_tables)
        # A replace inherits an existing declaration and also cannot lower it.
        # A newer source date may extend the declaration, but an older source
        # must not erase the protection deliberately established on the vault.
        history = max(
            (through for through in (existing_history, source_history) if through),
            default=None,
        )
        # A watermark may never reach a day on which this vault has ALREADY
        # accepted a HealthKit-direct sample, however fresh the source is.
        # Measured 2026-08-22: without this cap, refreshing the snapshot and
        # rebuilding moves the watermark onto the day the phone is currently
        # syncing, `_healthkit_ingest` then refuses every batch containing that
        # day, and the client — which commits its anchor only after a 2xx —
        # retries the identical batch forever. The sync wedges permanently and
        # silently, and only an explicit `set_history_imported_through` frees
        # it. Declaring history through a day we took live data for is a false
        # statement anyway; the cap keeps the declaration honest.
        history = _cap_history_to_live_floor(history, live_floor)
        target = db.connect(staging_path)
        db.init_db(target)
        target_tables = _table_names(target)

        unknown = source_tables - target_tables
        if unknown:
            raise ValueError(
                "source contains tables not present in the vault schema: "
                + ", ".join(sorted(unknown))
            )

        target.execute("BEGIN")
        # Inside the transaction: a vault that was interrupted mid-build is not
        # a vault, and must not come back declared. (This does not gate
        # build_vault's own copy — _copy_table writes raw SQL and never goes
        # through insert_records, which is where the D3 guard lives.)
        declare_vault(target)
        table_counts: dict[str, int] = {}
        records_seen = records_copied = 0
        copied_by_metric: dict[str, int] = {}
        dropped_by_metric: dict[str, int] = {}
        for table in _COPY_ORDER:
            if table not in source_tables:
                continue
            seen, copied, copied_metrics, dropped_metrics = _copy_table(
                source,
                target,
                table,
                batch_size=batch_size,
                allow_metrics=_streamed_raw_series() if table == "records" else None,
            )
            table_counts[table] = copied
            if table == "records":
                records_seen, records_copied = seen, copied
                copied_by_metric = copied_metrics
                dropped_by_metric = dropped_metrics

        # Bucketed series are aggregated rather than streamed, so they are
        # copied here instead of by _copy_table's row filter — which saw them,
        # counted them as dropped, and skipped them. Move that count across: a
        # bucketed series is stored coarser, not omitted, and a report that
        # calls it "dropped" is the kind of thing somebody later believes.
        bucketed_by_metric: dict[str, tuple[int, int]] = {}
        for metric, seconds in sorted(VAULT_BUCKET_SECONDS.items()):
            if metric not in VAULT_RAW_SERIES:
                continue
            written = _copy_bucketed(source, target, metric, seconds)[1]
            raw = dropped_by_metric.pop(metric, 0)
            if not raw and not written:
                continue          # the source has none of this series; say nothing
            bucketed_by_metric[metric] = (raw, written)
            records_copied += written
            copied_by_metric[metric] = written

        # Provenance is derived from raw `records`, and most of those do not
        # travel. Build it here from the SOURCE — after this the vault is the
        # only place the fact "the watch stopped recording steps in 2022"
        # survives at all, and rebuilding it inside the vault later would find
        # nothing to rebuild from. Copying the table above handles a source that
        # already has it; this fills a source that predates it.
        # Rebuild from the source's raw rows whenever it HAS raw rows, rather
        # than trusting a copied table that merely has some. A source whose
        # table was filled incrementally by the receiver covers recent months
        # only; accepting it as complete makes a historical instrument change
        # disappear, which is F3-2 with no symptom. When the source is itself a
        # vault it has no raw rows to rebuild from, and the copy is all there
        # is — which is correct, because that copy is already the full answer.
        if source.execute("SELECT 1 FROM records LIMIT 1").fetchone():
            target.execute("DELETE FROM metric_source_months")
            target.executemany(
                "INSERT INTO metric_source_months (metric, month, source, n) "
                "VALUES (?, ?, ?, ?)",
                source.execute(
                    "SELECT metric, substr(local_date, 1, 7), source, COUNT(*) "
                    "FROM records "
                    "GROUP BY metric, substr(local_date, 1, 7), source"),
            )
        table_counts["metric_source_months"] = target.execute(
            "SELECT COUNT(*) FROM metric_source_months").fetchone()[0]
        if owner is not None:
            target.execute(
                "INSERT OR REPLACE INTO vault_meta (key, value) "
                "VALUES ('owner', ?)", (owner,))
        if local_timezone is not None:
            set_local_timezone(target, local_timezone)
        if unit_system is not None:
            set_unit_system(target, unit_system)
        if existing_tuning.get("workout_source_arbitration_from") is not None:
            set_workout_source_arbitration_from(
                target, existing_tuning["workout_source_arbitration_from"])
        if existing_tuning.get("block_qualify_hr_max") is not None:
            set_block_qualify_hr_max(
                target, float(existing_tuning["block_qualify_hr_max"]))
        if history is not None:
            target.execute(
                "INSERT OR REPLACE INTO vault_meta (key, value) "
                "VALUES ('history_imported_through', ?)", (history,))

        # Carry the fencing epoch forward. A fresh SQLite file starts at
        # user_version 0, so rebuilding a vault that had committed at epoch 5
        # would silently reset its fence — after which a worker still holding
        # epoch 3 passes `landed >= lease.epoch` and overwrites work that
        # replaced it. The epoch may only ever move forward, including across a
        # rebuild, so take the highest of the source and whatever the
        # destination already reached.
        epoch = max(_user_version(source), existing_epoch)
        if epoch:
            target.execute(f"PRAGMA user_version = {int(epoch)}")
        if existing_commits:
            target.executemany(
                "INSERT OR IGNORE INTO commit_log (key, epoch, applied_at, detail) "
                "VALUES (?, ?, ?, ?)", existing_commits)

        if existing_total_rows:
            target.executemany(
                "INSERT INTO hk_daily_totals "
                "(metric, local_date, value, unit, interval, state, device_id, "
                "queried_at, first_seen_at, settled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", existing_total_rows)
        if existing_revision_rows:
            target.executemany(
                "INSERT INTO hk_daily_total_revisions "
                "(id, metric, local_date, from_value, to_value, from_state, "
                "to_state, lag_days, batch_id, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", existing_revision_rows)
        if existing_expected:
            target.executemany(
                "INSERT OR REPLACE INTO vault_meta (key, value) VALUES (?, ?)",
                existing_expected.items())
        # The source's daily_metrics rows carry records-derived sums. The
        # consolidated totals are live-acquired and were just restored above,
        # so apply Apple's figures again before the rebuilt vault is committed.
        db.apply_consolidated_totals(target)

        # An affirmative expectation. A rebuild may ADD consolidated totals; it may
        # never lose them. Checked here rather than by a later verify because here is
        # where the old vault still exists to be compared against — and because the
        # staging file is discarded on a raise, so the failure costs a rebuild
        # rather than a vault. (D19; the check `consolidated_diffs` cannot make.)
        kept = target.execute("SELECT COUNT(*) FROM hk_daily_totals").fetchone()[0]
        if kept < existing_totals:
            raise ValueError(
                f"rebuild would drop {existing_totals - kept} consolidated daily "
                f"total(s) ({existing_totals} in the vault being replaced, {kept} in "
                "the staging database) — these are live-acquired and have no source; "
                "see D19 §Q3")
        target.commit()
        target.close()
        target = None
        source.close()
        source = None
        os.replace(staging_path, vault_path)
        elapsed = time.perf_counter() - started
        size_bytes = vault_path.stat().st_size
        return {
            "source": str(source_path),
            "vault": str(vault_path),
            "build_seconds": elapsed,
            "size_bytes": size_bytes,
            "gzip_size_bytes": _gzip_size(vault_path) if measure_gzip else None,
            "records_seen": records_seen,
            "records_copied": records_copied,
            "records_dropped": records_seen - records_copied,
            "copied_by_metric": dict(sorted(copied_by_metric.items())),
            "bucketed_by_metric": {m: {"seconds": VAULT_BUCKET_SECONDS[m],
                                       "raw": raw, "buckets": n}
                                   for m, (raw, n) in bucketed_by_metric.items()},
            "dropped_by_metric": dict(sorted(dropped_by_metric.items())),
            "table_counts": table_counts,
            "owner": owner,
            "history_imported_through": history,
            "epoch": epoch,
        }
    finally:
        if target is not None:
            target.close()
        if source is not None:
            source.close()
        if staging_path.exists():
            staging_path.unlink()


def format_bytes(value: int) -> str:
    """Format a byte count in decimal units, matching the G-02 figures."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} GB"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} MB"
    if value >= 1_000:
        return f"{value / 1_000:.1f} kB"
    return f"{value} B"


def format_report(report: dict[str, Any]) -> str:
    """Render a complete human-readable build report."""
    lines = [
        f"vault: {report['vault']}",
        f"build time: {report['build_seconds']:.2f} s",
        f"size: {format_bytes(report['size_bytes'])}"
        + (f" ({format_bytes(report['gzip_size_bytes'])} gzipped)"
           if report.get("gzip_size_bytes") is not None else " (gzip not measured)"),
        f"records: {report['records_copied']:,} copied / "
        f"{report['records_seen']:,} source / {report['records_dropped']:,} dropped",
        "copied raw series:",
    ]
    lines.extend(
        f"  {metric}: {count:,}"
        for metric, count in report["copied_by_metric"].items()
    )
    if report.get("bucketed_by_metric"):
        lines.append("bucketed raw series (stored coarser than recorded):")
        lines.extend(
            f"  {metric}: {info['raw']:,} samples -> {info['buckets']:,} "
            f"{info['seconds']}s buckets "
            f"({info['raw'] / max(info['buckets'], 1):.1f}x)"
            for metric, info in report["bucketed_by_metric"].items()
        )
    lines.append("dropped raw series (visible omissions):")
    lines.extend(
        f"  {metric}: {count:,}"
        for metric, count in report["dropped_by_metric"].items()
    )
    lines.append("vault table rows:")
    lines.extend(
        f"  {table}: {count:,}"
        for table, count in report["table_counts"].items()
    )
    return "\n".join(lines)
