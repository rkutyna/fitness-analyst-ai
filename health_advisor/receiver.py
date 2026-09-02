"""FastAPI receiver for HealthKit-direct delta POSTs.

- Optional shared-secret header; rejects without it when set. The secret is
  read from HA_SECRET_FILE when set, else HA_SHARED_SECRET (#101).
- Idempotent upserts; recomputes daily_metrics for affected (metric, date) pairs.
- A vault that declares imported history refuses HealthKit samples on or before
  its watermark until an explicit re-derivation migration moves that marker.
- Binds to localhost by default (front with `tailscale serve`, never 0.0.0.0).

Run:  python -m health_advisor.receiver --vault PATH [--host H --port P]
Env:  HA_SECRET_FILE (preferred) or HA_SHARED_SECRET; HA_REQUIRE_SECRET=1 makes
      "no secret at all" a startup failure instead of an unauthenticated receiver
      HEALTH_ADVISOR_ANALYST_EXECUTOR=transient explicitly selects the
      user-systemd analyst executor; absent means the platform default

The vault is an argument, never an environment variable: `create_app(ctx)` binds
one receiver to one user's vault, and a process that serves two of them must not
be able to confuse which (T-003).
"""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from . import db
from . import chat
from .context import VaultContext
from . import analysis
from . import analyst
from . import derive
from . import hk_parse
from . import lease
from . import llm
from . import normalize as nz
from . import vault
from . import analyst_sandbox

def _load_shared_secret() -> tuple[str, str]:
    """The shared secret, preferring a file over the environment (#101, F-43).

    An environment variable is readable for the life of the process by anyone
    who can reach `/proc/<pid>/environ` — `docker exec`, root on the host, any
    process sharing the namespace. Reading the file here instead means the
    secret never has to enter this process's environment at all.

    Reading from a file does NOT by itself close that exposure. It is closed
    when `deploy/entrypoint.sh` stops exporting HA_SHARED_SECRET; until then
    both paths exist and `/proc` is unchanged. This function is step one of
    two, and `secret_source` on /health is how you tell which path a running
    container actually took — do not infer it.

    Fail-closed, and this is the part that matters. `receiver.py` treats an
    EMPTY secret as "no check" (see the auth callers below), so a file that is
    missing, unreadable, badly permissioned or too short must raise rather
    than fall through to the environment. Falling back would turn a
    misconfigured secret into a silently unauthenticated receiver that looks
    configured — the one way this change could be actively dangerous.

    The checks mirror `deploy/entrypoint.sh` and `scripts/run_receiver.sh`
    exactly, including stripping all whitespace the way `tr -d '[:space:]'`
    does. D16's bind conditions stopped being prose in those scripts; a
    receiver that read the file while skipping their validation would quietly
    turn them back into prose.
    """
    require = os.environ.get("HA_REQUIRE_SECRET", "").strip().lower() in {
        "1", "true", "yes", "on"}
    path = os.environ.get("HA_SECRET_FILE", "").strip()
    if not path:
        # No file configured: the historical environment path, unchanged,
        # empty-means-no-check included. Tests rely on that, and so does any
        # launcher that has not been migrated.
        env_secret = os.environ.get("HA_SHARED_SECRET", "")
        if require and not env_secret:
            # Step two of #101 removes `export HA_SHARED_SECRET` from
            # entrypoint.sh. After that, a HA_SECRET_FILE that fails to reach
            # Python leaves BOTH sources empty — and an empty secret means "no
            # check", so the receiver would serve unauthenticated on the
            # tailnet while looking configured. A deployment sets
            # HA_REQUIRE_SECRET so that state is a startup failure instead.
            raise RuntimeError(
                "refusing to start: HA_REQUIRE_SECRET is set but neither "
                "HA_SECRET_FILE nor HA_SHARED_SECRET provided a secret. An "
                "empty secret disables authentication entirely, which is why "
                "this raises rather than serving unauthenticated."
            )
        return env_secret, "env"

    try:
        raw = Path(path).read_text()
    except OSError as exc:
        raise RuntimeError(
            f"HA_SECRET_FILE={path!r} is set but could not be read ({exc}). "
            "Refusing to start: falling back to HA_SHARED_SECRET here could "
            "serve an unauthenticated receiver that looks configured."
        ) from exc

    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode not in (0o600, 0o400):
        raise RuntimeError(
            f"refusing to start: {path} is mode {mode:o}; D16 requires 600 or "
            "400. Run chmod 600 on it (host side)."
        )

    secret = "".join(raw.split())          # same as tr -d '[:space:]'
    if len(secret) < 16:
        raise RuntimeError(
            f"refusing to start: the secret in {path} is {len(secret)} chars "
            "after trimming; D16 requires >= 16. An empty one disables auth "
            "entirely, which is why this raises instead of falling back."
        )
    return secret, "file"


SHARED_SECRET, SHARED_SECRET_SOURCE = _load_shared_secret()

# Keep individual executemany calls bounded without giving up the HealthKit
# batch's one-transaction atomicity.
INGEST_CHUNK = int(os.environ.get("HA_INGEST_CHUNK", "10000"))

# Largest request body we will read. Derived from the unit's MemoryMax=2G, not
# picked round: the raw bytes, the decoded JSON and the built record dicts are
# all resident at once, roughly 8-10x the body, so ~256 MiB is about the most
# that can be parsed inside a 2 GB cgroup. The largest batch this receiver has
# ever seen is 1,052,330 records on 2026-06-24 (a one-time backlog drain), which
# is ~180 MB of JSON — so the cap admits every real sync with headroom while
# turning an unbounded one into a 413 instead of an OOM kill.
MAX_BODY_BYTES = int(os.environ.get("HA_MAX_BODY_BYTES", str(256 * 1024 * 1024)))

# An analyst tool call runs in the worker thread used by /v1/ask. It may not
# await an asyncio.Semaphore directly, so it submits the acquire coroutine to
# the receiver loop and waits here. A bounded wait turns contention into a
# typed refusal that the coach can relay instead of tying up the chat forever.
ANALYST_INTERNAL_WAIT_SECONDS = 120.0


def _require_ask_secret(x_health_secret: str | None) -> None:
    """Require a configured, non-empty secret for the interactive endpoint.

    This is intentionally stricter than ``/v1/ingest``'s historical optional
    check. An accidentally empty ask secret must never turn a health question
    endpoint into an unauthenticated data reader.
    """
    if not SHARED_SECRET or x_health_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="missing or bad shared secret")


def _ask_payload(raw: bytes) -> dict:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"malformed ask payload: {exc}")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="ask payload must be an object")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=422, detail="question must be a non-empty string")
    conversation_id = payload.get("conversation_id")
    if conversation_id is not None and (
            not isinstance(conversation_id, str) or not conversation_id.strip()):
        raise HTTPException(status_code=422,
                            detail="conversation_id must be a non-empty string")
    as_of = payload.get("as_of")
    if as_of is not None and not isinstance(as_of, str):
        raise HTTPException(status_code=422, detail="as_of must be a date string")
    return {"question": question.strip(), "conversation_id": conversation_id,
            "as_of": as_of}


def _run_analyst(ctx, question: str, *, complete_fn=None, run_code_fn=None,
                 executor_factory=analyst_sandbox.default_executor):
    """Run one analyst question and adapt the CLI JSON for HTTP.

    The sandbox is probed before ``run_analyst`` is called. That keeps an
    unavailable substrate from reaching the model, while the injectable seams
    keep this transport testable without changing the analyst core.
    """
    try:
        # Keep analyst.py's codex-by-name refusal in force for both the direct
        # endpoint and the in-process coach tool; the caller must not bypass
        # this check merely because it already selected a chat backend.
        analyst.assert_analyst_backend_approved()
    except RuntimeError as exc:
        return JSONResponse(status_code=200,
                            content={"refused": True, "reason": str(exc)})

    run_dir = tempfile.mkdtemp()
    try:
        try:
            executor = executor_factory()
        except RuntimeError as exc:
            # `detail`, not `reason`, and the key is load-bearing. FastAPI's
            # own HTTPException renders `{"detail": ...}` -- which is what the
            # 429 below sends -- and the iOS client reads `detail` for both.
            # Luna's patch sent `reason` here, so the one message the client
            # most needs to show intact was the one it would have dropped:
            # against the Linux host this 503 is the ONLY thing the Analysis
            # tab ever says, and a generic "server cannot run analyst mode"
            # without the sandbox's own words is a dead end for whoever reads
            # it. Caught at the review gate, 2026-08-30; the two halves were
            # built in parallel and each chose a defensible key.
            return JSONResponse(
                status_code=503,
                content={"detail": f"analyst sandbox unavailable: {exc}"},
            )

        output = io.StringIO()
        exit_code = analyst.run_analyst(
            question, ctx.db_path, run_dir,
            complete_fn=complete_fn, run_code_fn=run_code_fn,
            executor=executor, json_output=True, out=output)
        payload = json.loads(output.getvalue())
        if exit_code == 0:
            payload["refused"] = False
            payload["provenance"].pop("run_record_path", None)
        return JSONResponse(payload)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _analyst(ctx, request: Request, raw: bytes,
             x_health_secret: str | None = None, *, complete_fn=None,
             run_code_fn=None, executor_factory=analyst_sandbox.default_executor):
    """Handle one analyst request outside the FastAPI wiring."""
    _require_ask_secret(x_health_secret)
    payload = _ask_payload(raw)
    return _run_analyst(
        ctx, payload["question"], complete_fn=complete_fn,
        run_code_fn=run_code_fn, executor_factory=executor_factory)

def _health(ctx):
    conn = ctx.read_only()
    try:
        n = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        last = conn.execute("SELECT created_at, detail FROM ingest_log "
                            "ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    return {"ok": True, "records": n,
            "last_ingest": dict(last) if last else None,
            "secret_required": bool(SHARED_SECRET),
            # Which path the secret came from — "file" or "env". Never the
            # secret. #101 needs this observable: "we deployed the file
            # version" is a claim, and this is the check.
            "secret_source": SHARED_SECRET_SOURCE}


def _ask_freshness(ctx, as_of: str | None) -> dict:
    """Project the vault's per-vital coverage into the ask response.

    Keep this response contract limited to dates, labels, and booleans. The
    numeric coverage fields are useful to Python's analysis but would widen
    the model's grounding licence pool if they crossed the ask boundary.
    """
    conn = ctx.read_only()
    try:
        effective_as_of = analysis._as_of(conn, as_of)
        rows = analysis.coverage(conn, effective_as_of)
    finally:
        conn.close()
    return {
        "as_of": effective_as_of,
        "metrics": [
            {
                "metric": row["metric"],
                "status": row["status"],
                "last_date": row["last_date"],
                "covers_as_of": row["covers_as_of"],
                "behind": row["behind"],
            }
            for row in rows
        ],
    }


# TEMPORARY OPERATOR VISIBILITY: error codes and log detail are deliberately
# widened here to give visibility into sync attempts while the ingest path is
# still being brought up. Tightening encryption/opacity can wait until
# everything is working.
#
# This exists because a 409 was undiagnosable on 2026-08-27: `--no-access-log`
# keeps requests out of the journal, and the history-guard refusal deliberately
# rolls back, so neither the journal nor the database could say which date was
# refused. `_log_reject` cannot be used on that path — the ingest holds
# BEGIN IMMEDIATE, so a second connection would block on the writer lock.
# stderr has neither problem.
#
# Deliberately METADATA ONLY: dates, counts, metric names, batch and device ids.
# No sample values, ever — that restraint costs nothing and is why this is
# merely "temporary" rather than "a health-data trail in the journal".
# Re-tighten when ingest is trusted: see the tracker note filed with this change.
def _trace(event: str, **fields) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    print(f"ingest-trace {event} {parts}", file=sys.stderr, flush=True)


def _batch_span(parsed) -> tuple[str | None, str | None, int]:
    """(min local_date, max local_date, n) over all dated batch entries."""
    dates = [r["local_date"] for r in parsed.get("records", []) if r.get("local_date")]
    dates.extend(parsed.get("workout_dates", []))
    dates.extend(parsed.get("daily_total_dates", []))
    if not dates:
        return None, None, 0
    return min(dates), max(dates), len(dates)


def _log_reject(ctx, reason: str, nbytes: int) -> None:
    """Best-effort operator evidence for a request refused before parsing."""
    try:
        conn = ctx.connect()
        try:
            db.init_db(conn)
            db.log_ingest(conn, "receiver", "reject", 0, 0,
                          f"{reason} bytes={nbytes}")
        finally:
            conn.close()
    except Exception:                      # noqa: BLE001 - never mask the 413
        pass


async def _raw_body_for(ctx, request: Request) -> bytes:
    """Read the body in the event loop so the ENDPOINT can be a plain `def`.

    Dependencies are always resolved on the loop; a non-async endpoint is then
    handed to Starlette's threadpool. That is the whole trick: reading bytes off
    a socket is the only genuinely async part of /v1/ingest, and everything after
    it (json.loads over a multi-MB body, the parse, the sqlite writes) is
    blocking CPU/IO that must not sit on the loop.

    It is also where the body is bounded (MAX_BODY_BYTES). Content-Length is
    checked first so an oversized POST costs us nothing, but it is only a claim
    and a chunked upload omits it entirely — so the stream is counted as it
    arrives and abandoned the moment it crosses the cap.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_BODY_BYTES:
                _reject_too_large(ctx, int(declared))
        except ValueError:
            pass                               # unparseable: fall through to the stream cap

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            _reject_too_large(ctx, total)
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_too_large(ctx, nbytes: int) -> None:
    detail = f"body too large: {nbytes} bytes > {MAX_BODY_BYTES} limit"
    _log_reject(ctx, detail, nbytes)
    raise HTTPException(status_code=413, detail=detail)


def _healthkit_ingest(ctx, request: Request, raw: bytes,
                      x_health_secret: str | None = None):
    """Apply one parsed HealthKit delta to the vault bound to ``ctx``.

    This is the receiver's HealthKit-direct ingest path.
    HealthKit batches are one transaction in DELETE journal mode: the batch is
    small enough for the commit to be atomic, and an exception leaves records,
    tombstones, anchors, and the commit key all rolled back together.
    """
    if SHARED_SECRET:
        if x_health_secret != SHARED_SECRET:
            raise HTTPException(status_code=401, detail="missing or bad shared secret")

    try:
        payload = json.loads(raw)
        parsed = hk_parse.parse_payload(payload)
    except (json.JSONDecodeError, hk_parse.PayloadError) as exc:
        # A malformed HealthKit envelope must leave no
        # database evidence at all. In particular, do not write an ingest_log
        # row before the parser has accepted the batch.
        raise HTTPException(status_code=400, detail=f"malformed payload: {exc}")

    # "healthkit" here is the commit key's NAMESPACE, not records.origin.
    # It shares a spelling with hk_parse.HEALTHKIT_ORIGIN and with the
    # ingest_log source below, and the three are independent vocabularies
    # (#133). Do not "unify" them: changing this string changes every
    # idempotency key, so a batch mid-retry would stop matching its own
    # prior commit and be applied twice.
    key = lease.commit_key("healthkit", parsed["device_id"], parsed["batch_id"])
    if (prior := lease.already_applied(ctx, key)) is not None:
        return JSONResponse({
            "ok": True, "applied": False, "reason": "already_applied",
            "batch_id": parsed["batch_id"], "records_seen": len(parsed["records"]),
            "workouts_seen": len(parsed["workouts"]), "workouts_added": 0,
            "unhandled": parsed["unhandled"][:20], **prior,
        })

    accepted: list[dict] = []
    affected: set[tuple[str, str]] = set()
    rec_added = 0
    daily_totals_added = 0
    deleted = 0
    tombstones_added = 0
    moved = 0
    # Sub-workouts refused as contained in a same-source session (#150).
    workout_fragments: list[str] = []
    dm = 0
    applied = True

    conn = ctx.connect()
    try:
        db.init_db(conn)
        conn.execute("BEGIN IMMEDIATE")

        # The preflight is useful for the common replay case, but the check
        # inside this write transaction is the race-safe one.
        prior_row = conn.execute(
            "SELECT key, epoch, applied_at, detail FROM commit_log WHERE key = ?",
            (key,),
        ).fetchone()
        if prior_row is not None:
            conn.execute("ROLLBACK")
            applied = False
            prior = dict(prior_row)
        else:
            # Check before the first mutation. The rollback in the outer
            # exception handler is intentional: BEGIN IMMEDIATE has acquired
            # the writer lock, but this refusal leaves no database evidence.
            history = vault.history_imported_through(conn)
            if history is not None:
                record_offending = min(
                    (row["local_date"] for row in parsed["records"]
                     if row["local_date"] <= history), default=None)
                workout_offending = min(
                    (day for day in parsed["workout_dates"] if day <= history),
                    default=None)
                total_offending = min(
                    (day for day in parsed["daily_total_dates"] if day <= history),
                    default=None)
                offending = min(
                    (day for day in (record_offending, workout_offending,
                                     total_offending)
                     if day is not None), default=None)
                if offending is not None:
                    if record_offending == offending:
                        kind = "record"
                    elif workout_offending == offending:
                        kind = "workout"
                    else:
                        kind = "daily total"
                    lo, hi, n = _batch_span(parsed)
                    _trace("reject-409-history",
                           watermark=history, offending=offending,
                           batch_min=lo, batch_max=hi, records=n,
                           batch_id=parsed.get("batch_id"),
                           device=parsed.get("device_id"))
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"history imported through {history}; refusing "
                            f"HealthKit batch containing {kind} dated {offending}"
                            " — a client cannot retry past this, so move the"
                            " cutover after the watermark or move the watermark"
                            " with vault.set_history_imported_through()"
                        ),
                    )

            # This sibling guard is deliberately before every mutation. It is
            # batch-atomic: a settled row anywhere in a multi-day payload
            # refuses the whole transaction, including rows that were otherwise
            # still provisional.
            for row in parsed["daily_totals"]:
                prior = conn.execute(
                    "SELECT state FROM hk_daily_totals "
                    "WHERE metric = ? AND local_date = ?",
                    (row["metric"], row["local_date"]),
                ).fetchone()
                if prior is not None and prior["state"] == "settled":
                    _trace("reject-409-settled", metric=row["metric"],
                           day=row["local_date"], batch_id=parsed["batch_id"])
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"daily total already settled for {row['metric']} on "
                            f"{row['local_date']}; a settled consolidated total "
                            "is immutable (D19/#220) — this day is closed and the "
                            "client should advance past it, not retry"
                        ),
                    )

            # A tombstone is durable before the add filter is evaluated. A
            # deletion for an unknown UUID therefore still protects against a
            # later stale add, while replaying the deletion does no work.
            for deletion in parsed["deletions"]:
                dtype, uuid = deletion["type_identifier"], deletion["hk_uuid"]
                tombstone = conn.execute(
                    "SELECT 1 FROM hk_deletions WHERE device_id = ? "
                    "AND type_identifier = ? AND hk_uuid = ?",
                    (parsed["device_id"], dtype, uuid),
                ).fetchone()
                if tombstone is not None:
                    continue
                rows = conn.execute(
                    "SELECT metric, local_date FROM records "
                    "WHERE hk_uuid = ? AND hk_type_identifier = ? "
                    "AND hk_device_id = ?",
                    (uuid, dtype, parsed["device_id"]),
                ).fetchall()
                affected.update((row["metric"], row["local_date"]) for row in rows)
                # Capture the sample's own date BEFORE deleting it. This is the
                # only moment it exists: the row is about to go, and the
                # tombstone is all that survives. `deleted_at` gives when a
                # deletion arrived, never how late it was, and how late is the
                # figure the compaction window has to be designed against
                # (#37). Earliest row wins when a UUID somehow spans several,
                # so the value is deterministic rather than whichever row
                # SQLite returned first.
                oldest = min(rows, key=lambda r: r["local_date"]) if rows else None
                cur = conn.execute(
                    "DELETE FROM records WHERE hk_uuid = ? "
                    "AND hk_type_identifier = ? AND hk_device_id = ?",
                    (uuid, dtype, parsed["device_id"]),
                )
                deleted += cur.rowcount
                conn.execute(
                    "INSERT INTO hk_deletions "
                    "(device_id, type_identifier, hk_uuid, deleted_at, "
                    " sample_local_date, sample_metric) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (parsed["device_id"], dtype, uuid, db.utcnow_iso(),
                     oldest["local_date"] if oldest else None,
                     oldest["metric"] if oldest else None),
                )
                tombstones_added += 1

            # Filter tombstoned adds before constructing pairs. This matters
            # for a deletion of an unknown row followed by the same stale row:
            # it must not create either a raw row or a daily aggregate.
            for row in parsed["records"]:
                tombstone = conn.execute(
                    "SELECT 1 FROM hk_deletions WHERE device_id = ? "
                    "AND type_identifier = ? AND hk_uuid = ?",
                    (row["hk_device_id"], row["hk_type_identifier"], row["hk_uuid"]),
                ).fetchone()
                if tombstone is not None:
                    continue
                accepted.append(row)
                affected.add((row["metric"], row["local_date"]))

            # A HealthKit UUID is the source identity. Replace a prior copy of
            # that UUID in raw records so a source correction cannot trip the
            # metric/UUID uniqueness index. This applies to every metric now:
            # non-allowlisted rows are transient, not discarded.
            for row in accepted:
                old = conn.execute(
                    "SELECT metric, local_date FROM records "
                    "WHERE metric = ? AND hk_uuid = ?",
                    (row["metric"], row["hk_uuid"]),
                ).fetchall()
                affected.update((item["metric"], item["local_date"]) for item in old)
                conn.execute(
                    "DELETE FROM records WHERE metric = ? AND hk_uuid = ?",
                    (row["metric"], row["hk_uuid"]),
                )

            # Every accepted sample is durable now. Aggregation follows the
            # write and therefore sees this batch plus all earlier batches for
            # the day; only compaction removes transient raw rows later.
            for i in range(0, len(accepted), INGEST_CHUNK):
                rec_added += db.insert_records(
                    conn, accepted[i:i + INGEST_CHUNK]
                )
            # Workouts have their own session identity and merge point. They
            # are deliberately applied after records land and before the first
            # daily recompute, so a workout-only page still derives its day.
            # Sub-workouts the phone emits as top-level sessions are refused
            # here (#150), and each refusal is NAMED. The defect ran from June
            # to August precisely because nothing refused and nothing warned.
            def _note_fragment(row: dict, outer: dict) -> None:
                workout_fragments.append(
                    f"{row['workout_type']} {row['start_utc']}..{row['end_utc']} "
                    f"({(row.get('duration_min') or 0.0):.1f}min) contained in "
                    f"{outer['start_utc']}..{outer['end_utc']} "
                    f"({(outer.get('duration_min') or 0.0):.1f}min)")

            workouts_added = db.insert_workouts(
                conn, parsed["workouts"], report=_note_fragment)
            # A workout changes which already-ingested distance samples survive
            # workout-window arbitration. Re-derive that sole workout-arbitrated
            # metric when a workout arrives after its samples; history before
            # the arbitration cutoff is intentionally untouched.
            affected.update(
                ("distance_walking_running", day)
                for day in parsed["workout_dates"]
                if day >= nz.WORKOUT_SOURCE_ARBITRATION_FROM
            )
            for row in parsed["daily_totals"]:
                affected.add((row["metric"], row["local_date"]))
            daily_totals_added = db.insert_daily_totals(
                conn, parsed["daily_totals"], batch_id=parsed["batch_id"])
            dm = db.recompute_daily_metrics(conn, pairs=sorted(affected))

            # Session attribution is a records concern, so it runs after the
            # rows land, then recompute both the old and new dates.
            sleep_days = sorted({
                day for metric, day in affected
                if metric in derive._STAGE_METRICS
            })
            if sleep_days:
                moves = derive.reattribute_sleep(
                    conn, sleep_days[0], sleep_days[-1], apply=True
                )
                if moves:
                    moved = len(moves)
                    affected |= derive.pairs_for_moves(moves)
                    dm += db.recompute_daily_metrics(
                        conn, pairs=sorted(affected)
                    )

            db.rebuild_metric_source_months(conn, pairs=sorted(affected))

            for anchor in parsed["anchors"]:
                conn.execute(
                    "INSERT INTO hk_sync_state "
                    "(device_id, type_identifier, anchor_token, "
                    "last_batch_sequence, last_batch_id, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(device_id, type_identifier) DO UPDATE SET "
                    "anchor_token=excluded.anchor_token, "
                    "last_batch_sequence=excluded.last_batch_sequence, "
                    "last_batch_id=excluded.last_batch_id, "
                    "updated_at=excluded.updated_at",
                    (parsed["device_id"], anchor["type_identifier"], anchor["to"],
                     parsed["batch_sequence"], parsed["batch_id"], db.utcnow_iso()),
                )

            detail = (
                f"records_seen={len(parsed['records'])} records_added={rec_added} "
                f"workouts_seen={len(parsed['workouts'])} workouts_added={workouts_added} "
                f"workouts_contained_rejected={len(workout_fragments)} "
                + ("workouts_rejected_detail=" + "; ".join(workout_fragments[:5]) + " "
                   if workout_fragments else "")
                + f"deleted={deleted} tombstones={tombstones_added} "
                f"daily_totals_seen={len(parsed['daily_totals'])} "
                f"daily_totals_added={daily_totals_added} daily_pairs={dm} "
                f"history_imported_through={history or '-'} "
                f"batch_sequence={parsed['batch_sequence']}"
            )
            conn.execute(
                "INSERT INTO commit_log (key, epoch, applied_at, detail) "
                "VALUES (?, ?, ?, ?)",
                (key, 0, db.utcnow_iso(), detail),
            )
            conn.commit()
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()

    if not applied:
        return JSONResponse({
            "ok": True, "applied": False, "reason": "already_applied",
            "batch_id": parsed["batch_id"], "records_seen": len(parsed["records"]),
            "workouts_seen": len(parsed["workouts"]), "workouts_added": 0,
            "daily_totals_seen": len(parsed["daily_totals"]),
            "unhandled": parsed["unhandled"][:20], **prior,
        })

    # This helper deliberately runs after the data/anchor commit: it is
    # designed to swallow a derive failure and report it, while the HealthKit
    # batch itself is already durable and its replay key is already recorded.
    derive_errors: list[str] = []
    derive_conn = ctx.connect()
    try:
        derive_days = {day for _, day in affected} | parsed["workout_dates"]
        derived = derive.update_after_ingest(
            # This positional is derive's `source`, which lands in
            # ingest_log.source — not records.origin either (#133).
            derive_conn, derive_days, "healthkit",
            errors=derive_errors,
        )
        # Operator evidence for a SUCCESSFUL sync. Until 2026-08-27 only the
        # reject path wrote to ingest_log, so a healthy HealthKit ingest left no
        # server-side trace at all: `/health` kept reporting `last_ingest` from
        # the final HAE batch of 2026-08-21 while 23,091 HealthKit rows landed
        # on 08-27. Every counter below was already computed and returned in the
        # response — but only the phone ever saw it.
        #
        # Best-effort by construction: the batch is durable and its replay key
        # is recorded by this point, so a logging failure must not become a 500.
        # A 500 here would make the client retain and re-offer a batch that was
        # in fact applied.
        try:
            db.log_ingest(
                derive_conn, "healthkit", "ingest",
                len(parsed["records"]), rec_added,
                f"records_seen={len(parsed['records'])} records_added={rec_added} "
                f"workouts_seen={len(parsed['workouts'])} workouts_added={workouts_added} "
                f"workouts_contained_rejected={len(workout_fragments)} "
                + ("workouts_rejected_detail="
                   + "; ".join(workout_fragments[:5]) + " "
                   if workout_fragments else "")
                + f"deleted={deleted} tombstones={tombstones_added} "
                f"daily_totals_seen={len(parsed['daily_totals'])} "
                f"daily_totals_added={daily_totals_added} "
                f"daily_pairs={dm} derived={derived} "
                f"history_imported_through={history or '-'} "
                f"unhandled={len(parsed['unhandled'])} "
                f"batch_sequence={parsed['batch_sequence']}",
            )
        except Exception:                      # noqa: BLE001 - never fail a durable ingest
            pass
    finally:
        derive_conn.close()
    _dates = sorted({d for _, d in affected} | parsed["workout_dates"])
    _trace("ingest-ok",
           batch_id=parsed["batch_id"], device=parsed.get("device_id"),
           records_seen=len(parsed["records"]), records_added=rec_added,
           workouts_seen=len(parsed["workouts"]), workouts_added=workouts_added,
           workouts_rejected=len(workout_fragments),
           deleted=deleted, daily_pairs=dm, derived=derived,
           unhandled=len(parsed["unhandled"]),
           date_min=_dates[0] if _dates else None,
           date_max=_dates[-1] if _dates else None)
    return JSONResponse({
        "ok": True, "applied": True, "batch_id": parsed["batch_id"],
        "records_seen": len(parsed["records"]), "records_added": rec_added,
        "workouts_seen": len(parsed["workouts"]), "workouts_added": workouts_added,
        "daily_totals_seen": len(parsed["daily_totals"]),
        "daily_totals_added": daily_totals_added,
        # A refused fragment is not a workout the phone should retry (#150).
        # `workouts_added` alone cannot say whether one was refused or merged.
        "workouts_rejected": len(workout_fragments),
        "workouts_rejected_detail": workout_fragments[:10],
        "deleted": deleted, "tombstones_added": tombstones_added,
        "daily_pairs_updated": dm, "dates": _dates,
        "detail": detail,
        "unhandled": parsed["unhandled"][:20], "derived": derived,
        "derive_error": derive_errors[0] if derive_errors else None,
    })


def create_app(ctx, *, analyst_complete_fn=None, analyst_run_code_fn=None,
               analyst_executor_factory=analyst_sandbox.default_executor) -> FastAPI:
    """One receiver bound to one user's vault.

    A factory rather than a module-level `app` because the vault has to be
    chosen by the caller. The route body stays a module-level function taking
    `ctx` first; only the FastAPI wiring lives in here.
    """
    llm.assert_backend_approved()
    app = FastAPI(title="Health Advisor Receiver")
    analyst_permit = asyncio.Semaphore(1)

    @app.on_event("startup")
    def _ensure_db():
        """Create the DB + schema if missing so read-only /health always works."""
        chat.ensure_turn_schema(ctx)

    async def _raw_body(request: Request) -> bytes:
        return await _raw_body_for(ctx, request)

    @app.get("/health")
    def health():
        return _health(ctx)

    @app.post("/v1/ingest")
    def healthkit_ingest(request: Request, raw: bytes = Depends(_raw_body),
                         x_health_secret: str | None = Header(default=None)):
        return _healthkit_ingest(ctx, request, raw, x_health_secret)

    @app.post("/v1/analyst")
    async def analyst_route(request: Request, raw: bytes = Depends(_raw_body),
                            x_health_secret: str | None = Header(default=None)):
        _require_ask_secret(x_health_secret)
        payload = _ask_payload(raw)
        if analyst_permit.locked():
            raise HTTPException(status_code=429,
                                detail="an analyst run is already in flight")
        await analyst_permit.acquire()
        try:
            return await asyncio.to_thread(
                _run_analyst, ctx, payload["question"],
                complete_fn=analyst_complete_fn,
                run_code_fn=analyst_run_code_fn,
                executor_factory=analyst_executor_factory)
        finally:
            analyst_permit.release()

    @app.post("/v1/ask")
    async def ask(request: Request, raw: bytes = Depends(_raw_body),
                  x_health_secret: str | None = Header(default=None)):
        """Answer one authenticated question and persist its two turns.

        The route observes disconnect state after the blocking model/tool work
        completes, while that work runs off the event loop. The response
        carries explicit completion/provenance fields rather than hiding the
        turn's latency behind a fire-and-forget acknowledgement.
        """
        _require_ask_secret(x_health_secret)
        payload = _ask_payload(raw)
        conversation_id = payload["conversation_id"]
        if conversation_id is None:
            conversation_id = chat.create_conversation(ctx)["id"]
        elif chat.get_conversation(ctx, conversation_id) is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown conversation: {conversation_id}")

        question_turn, history = chat.append_question_and_history(
            ctx, conversation_id, payload["question"])
        loop = asyncio.get_running_loop()
        attachments: list[dict] = []

        def internal_analyst_query(question: str) -> dict:
            """Run analyst from chat while sharing /v1/analyst's permit.

            The chat work is in a worker thread. Scheduling the semaphore
            acquire/release onto the receiver's event loop keeps the async
            route's immediate 429 check and this internal wait on one permit,
            without attempting to use asyncio primitives across threads.
            """
            if loop.is_closed():
                return {"refused": True,
                        "reason": "analyst_query has no active receiver loop"}
            acquire = asyncio.run_coroutine_threadsafe(
                analyst_permit.acquire(), loop)
            try:
                acquire.result(timeout=ANALYST_INTERNAL_WAIT_SECONDS)
            except FutureTimeoutError:
                # If the timeout races with a successful acquire, cancel()
                # returns False and the permit must be returned explicitly.
                # Otherwise cancelling the pending coroutine removes it from
                # the semaphore waiters without consuming a permit.
                if not acquire.cancel():
                    try:
                        if acquire.result():
                            loop.call_soon_threadsafe(analyst_permit.release)
                    except Exception:
                        pass
                return {"refused": True,
                        "reason": "analyst_query timed out waiting for an analyst run"}
            except Exception as exc:
                return {"refused": True,
                        "reason": f"analyst_query could not acquire its run permit: {exc}"}

            try:
                response = _run_analyst(
                    ctx, question, complete_fn=analyst_complete_fn,
                    run_code_fn=analyst_run_code_fn,
                    executor_factory=analyst_executor_factory)
                if response.status_code != 200:
                    detail = getattr(response, "body", b"")
                    try:
                        detail = json.loads(detail).get("detail", detail)
                    except (TypeError, ValueError, AttributeError):
                        pass
                    return {"refused": True,
                            "reason": f"analyst_query failed: {detail}"}
                result = json.loads(response.body)
                if result.get("refused"):
                    return {"refused": True,
                            "reason": result.get("reason", "analyst run refused")}
                table_results = []
                for table in result.get("tables") or []:
                    # This is the exact validated table payload generated by
                    # analyst._print_envelope; no model output is involved.
                    table_results.append({
                        "name": table["name"],
                        "columns": table["columns"],
                        "units": table["units"],
                        "rows": table["rows"],
                        "row_count": table["row_count"],
                    })
                    attachments.append({
                        "type": "table",
                        "name": table["name"],
                        "columns": table["columns"],
                        "units": table["units"],
                        "rows": table["rows"],
                        "row_count": table["row_count"],
                        "provenance": result["provenance"],
                        "code": result["code"],
                    })
                return {"tables": table_results}
            except Exception as exc:
                return {"refused": True,
                        "reason": f"analyst_query failed: {type(exc).__name__}: {exc}"}
            finally:
                loop.call_soon_threadsafe(analyst_permit.release)

        result = await asyncio.to_thread(
            chat.answer_question, ctx, payload["question"],
            as_of=payload["as_of"], history=history,
            analyst_query_fn=internal_analyst_query,
            attachments=attachments)
        disconnected_at = db.utcnow_iso() if await request.is_disconnected() else None
        chat.append_turn(
            ctx, conversation_id, "assistant", result["text"],
            answers_turn_id=question_turn["id"],
            client_disconnected_at=disconnected_at,
            attachments=result.get("attachments", attachments),
        )
        # No "ok"/"status"/"cancelled" fields. They were literals — True,
        # "complete", False — under a docstring promising explicit completion
        # state, so a failed or cancelled turn could not have reported itself
        # through them. A status light wired to on is worse than no status
        # light: a client binds to it and believes it. Completion is carried by
        # `mode` ("narration" vs "fallback") and by `verification`, both of
        # which are computed. The store separately records an observed client
        # disconnect on the immutable assistant turn; it is not a response
        # status literal that a client could mistake for delivery.
        return {
            "request_id": uuid.uuid4().hex,
            "conversation_id": conversation_id,
            "text": result["text"],
            "answer": result["text"],
            "mode": result["mode"],
            "tool_trace": result["tool_trace"],
            "provenance": {"tool_calls": len(result["tool_trace"])},
            "verification": result["verification"],
            "attachments": result.get("attachments", attachments),
            "freshness": _ask_freshness(ctx, payload["as_of"]),
        }

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="python -m health_advisor.receiver")
    ap.add_argument("--vault", "--db", dest="vault", required=True,
                    help="path to the vault this receiver ingests into")
    ap.add_argument("--user", default="local", help="user id this vault belongs to")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; front with `tailscale serve`, never 0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--access-log", action="store_true",
                    help="TEMPORARY (2026-08-27): log every request line. Off by "
                         "default because request metadata in the journal is a "
                         "health-data trail; on while ingest is being restored.")
    ap.add_argument(
        "--analyst-executor", choices=("default", "transient"), default=None,
        help="explicit analyst substrate; otherwise use the platform default",
    )
    args = ap.parse_args(argv)
    ctx = VaultContext.local(args.vault, user_id=args.user, writable=True)
    # Resolve this once at process startup. In particular, do not silently
    # switch substrates by probing systemd-run on each analyst request.
    selected_executor = args.analyst_executor
    if selected_executor is None:
        selected_executor = os.environ.get("HEALTH_ADVISOR_ANALYST_EXECUTOR")
    if selected_executor not in (None, "", "default", "transient"):
        ap.error(
            "HEALTH_ADVISOR_ANALYST_EXECUTOR must be 'default' or 'transient'"
        )
    executor_factory = analyst_sandbox.default_executor
    if selected_executor == "transient":
        executor_factory = analyst_sandbox.TransientUnitExecutor
    # Keep access logging disabled so request metadata cannot become a health-data
    # trail in the journal.
    uvicorn.run(create_app(ctx, analyst_executor_factory=executor_factory),
                host=args.host, port=args.port,
                access_log=args.access_log, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
