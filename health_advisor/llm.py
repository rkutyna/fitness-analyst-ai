"""Robust model runtime for the health_advisor PIPELINE (briefings + deep-dive).
The ONLY module that talks to the model. Three backends, selected by
HA_LLM_BACKEND:

- "codex" (default): GPT via a `codex exec` subprocess (ChatGPT auth). Tool-less
  calls run sandboxed read-only with MCP disabled; the researcher paths run a
  single agentic `codex exec` with the curated health-deepdive MCP server.
- "ollama": the original direct Ollama /api/chat transport (local fallback;
  also what the httpx-level tests exercise).
- "openrouter": an OpenAI-compatible HTTP transport. `complete` is single-shot;
  `tool_loop`/`research_loop` run the same in-process agentic loop the ollama
  backend does, in the OpenAI dialect (tool results keyed by tool_call_id, not
  by tool name). Its reasoning mode is explicitly set by
  `HA_OPENROUTER_REASONING=on|off|low`; an unset or invalid value refuses
  startup. `on`/`off` send OpenRouter's boolean `{"reasoning": {"enabled": …}}`;
  `low` sends its effort form, `{"reasoning": {"effort": "low"}}`, which the
  boolean cannot express.
  Added on #128, because the rebuilt host runs neither of the other two
  backends.

Every error degrades to "" and never raises, so the callers'
grounding/judge/fallback gates always receive a clean string and a slow or down
model can never crash a briefing. Interactive chat is outside this model
transport."""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import urllib.parse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BACKEND = os.environ.get("HA_LLM_BACKEND", "codex")
CODEX_MODEL = os.environ.get("HA_CODEX_MODEL", "gpt-5.6-luna")
CODEX_BIN = os.environ.get("HA_CODEX_BIN") or shutil.which("codex")

OLLAMA_URL = os.environ.get("HA_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("HA_LLM_MODEL", "qwen3.5:9b-q4_K_M")
KEEP_ALIVE = os.environ.get("HA_LLM_KEEP_ALIVE", "10m")  # stay resident across narrate→judge→retry
OPENROUTER_URL = os.environ.get("HA_OPENROUTER_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("HA_OPENROUTER_MODEL")


def _read_openrouter_api_key_file(path: str) -> str:
    """Read and validate the explicitly configured OpenRouter key file."""
    try:
        raw = Path(path).read_text()
    except OSError as exc:
        raise RuntimeError(
            f"HA_OPENROUTER_API_KEY_FILE={path!r} is set but could not be "
            f"read ({exc}). Refusing to start rather than falling back to "
            "OPENROUTER_API_KEY.") from exc

    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode not in (0o600, 0o400):
        raise RuntimeError(
            f"refusing to start: {path} is mode {mode:o}; D16 requires 600 "
            "or 400. Run chmod 600 on it (host side).")

    key = "".join(raw.split())
    if not key:
        raise RuntimeError(
            f"refusing to start: the OpenRouter key in {path} is empty "
            "after trimming; refusing to fall back to OPENROUTER_API_KEY.")
    return key


OPENROUTER_API_KEY_FILE = os.environ.get(
    "HA_OPENROUTER_API_KEY_FILE", "").strip()
_OPENROUTER_API_KEY_ENV = os.environ.get("OPENROUTER_API_KEY")
_OPENROUTER_API_KEY_FILE_VALUE = (
    _read_openrouter_api_key_file(OPENROUTER_API_KEY_FILE)
    if OPENROUTER_API_KEY_FILE else None)
OPENROUTER_API_KEY = (_OPENROUTER_API_KEY_FILE_VALUE
                      if OPENROUTER_API_KEY_FILE
                      else (_OPENROUTER_API_KEY_ENV or ""))
OPENROUTER_API_KEY_SOURCE = (
    "file" if OPENROUTER_API_KEY_FILE else "env")
# Unlike sampling defaults, this changes both cost and model behaviour. It is
# deliberately absent when unstated: assert_backend_approved() refuses an
# OpenRouter process that does not name `on`, `off` or `low` at its entry point.
OPENROUTER_REASONING = os.environ.get("HA_OPENROUTER_REASONING")
# Route control. Not only a speed knob: pinning decides WHICH third party sees
# the health data in a prompt, which is the concern #41 raises about the codex
# path. HA_OPENROUTER_PROVIDERS is an explicit allow-list (fallbacks OFF, so a
# pin cannot silently degrade to whoever else is cheap); HA_OPENROUTER_PROVIDER_SORT
# ("throughput", "price", "latency") only reorders OpenRouter's own routing.
OPENROUTER_PROVIDERS = os.environ.get("HA_OPENROUTER_PROVIDERS", "")
OPENROUTER_PROVIDER_SORT = os.environ.get("HA_OPENROUTER_PROVIDER_SORT", "")

# D15 (2026-08-24 amendment): this is an explicit allow-list for every
# provider-facing entry point. Codex is approved for the solo phase only under
# D17; that approval expires when the first other person's vault exists.
APPROVED_BACKENDS = frozenset({"ollama", "openrouter", "codex"})

# D15 (2026-08-25 amendment, enforced by #104): the approved OpenRouter
# providers per model. This is the ONE place the mapping lives; both the
# entry-point gate and the response-side provider check below read the set for
# the configured model.
#
# Admission criteria (decided 2026-08-26) — a provider may be added only if it
# has **zero data retention and no training on prompts**, and either **similar
# throughput** to the entries here or is **significantly cheaper**. Adding one
# changes *who sees the health data*, which is what D15 is about, so a candidate
# is proposed with evidence of its published terms — never merged because it
# benchmarked well.
#
# This is deliberately NOT a default for HA_OPENROUTER_PROVIDERS. A default
# would supply the pin when it is absent and let an unpinned process look
# approved, which is exactly what #76 exists to prevent. This set only refuses a
# pin that is present and wrong; unset or empty still fails below.
# Admitted 2026-08-28, after verifying zero data retention and no training on
# prompts for both: `together` and `parasail/fp8`. They serve the
# same pinned model and cost the same as coreweave/fp8 ($0.14/$0.28 vs
# $0.13/$0.28 per M tokens), so they clear the "similar throughput or
# significantly cheaper" half of the criteria on price rather than on speed
# (55 and 48 tok/s against CoreWeave's 72).
#
# The reason they were sought is a CAPABILITY gap, measured 2026-08-28 against
# OpenRouter's own endpoint metadata. `coreweave/fp8` does NOT support
# `structured_outputs`, does NOT support `response_format`, and supports
# `tool_choice` only as "auto" — not "required" and not a named function. So
# nothing constrains tool-call arguments at generation time, and the terminal
# submit_answer call cannot be compelled. 24 of 26 measured loop failures were
# malformed submit_answer arguments, with garbage injected mid-JSON
# (`{"sequence":  ript 19, ...}`). Its context window is also 262,144 against
# these two at 1,048,576.
#
# NOTE the tag formats differ and are not interchangeable: Together's endpoint
# tag is bare `together`, Parasail's is `parasail/fp8`. A wrong tag fails
# closed here, which is the intended direction.
#
# CONSIDERED AND DECLINED — `relace/fp4` (decided 2026-08-30, #194). It is
# the cheapest endpoint serving the pinned model ($0.065/$0.18 per M against
# CoreWeave's $0.13/$0.28) at 1,048,576 context, so it clears the price half
# easily. It fails on both of the halves that matter here. Its capability
# profile is coreweave/fp8's exactly — no `structured_outputs`, no
# `response_format`, `tool_choice` "auto" only — so it is a third provider that
# cannot compel the terminal submit_answer call, which is the gap the two above
# were added to close. And the no-training half could not be established from
# published terms: OpenRouter's provider page states zero data retention, but
# OpenRouter's own docs treat retention and training as separate policies, and
# the privacy policy OpenRouter links for Relace 404s. Absent evidence is not
# admission. Do not re-propose it without terms to quote.
APPROVED_OPENROUTER_PROVIDERS = {
    "deepseek/deepseek-v4-flash-0731": frozenset({
        "coreweave/fp8", "together", "parasail/fp8"}),
    "z-ai/glm-5.3-flash": frozenset({
        "baseten/fp8", "novita/fp8", "together"}),
}

# OpenRouter reports the provider's display name in responses, while the
# request-side pin and D15 allow-list use endpoint tags. Keep this translation
# explicit and exact: an unfamiliar display name is not an approved provider.
OPENROUTER_PROVIDER_TAGS = {
    "CoreWeave": "coreweave/fp8",
    "Together": "together",
    "Parasail": "parasail/fp8",
    "BaseTen": "baseten/fp8",
    "Novita": "novita/fp8",
}

# D15 (#138): the provider pin says who OpenRouter routes to. It says nothing
# about who receives the request in the first place, and HA_OPENROUTER_URL is an
# env var. Measured 2026-08-26: HA_OPENROUTER_URL='https://not-openrouter.example/v1'
# with an approved provider pin PASSED assert_backend_approved(), so the check
# reported approval for a host D15 never named. This is the same shape #76
# closed for the backend name, on the axis it did not range over.
#
# Keyed by backend, compared on the PARSED HOSTNAME and never by substring:
# 'https://openrouter.ai.attacker.example/' contains "openrouter.ai" and is a
# different host. Loopback is written out rather than assumed, because the
# default being local is not the same as the variable being local.
APPROVED_ENDPOINT_HOSTS = {
    "openrouter": frozenset({"openrouter.ai"}),
    "ollama": frozenset({"127.0.0.1", "::1", "localhost"}),
}

# Remote endpoints must be TLS. Loopback is exempt: ollama serves plain HTTP on
# 127.0.0.1 by default and there is no network hop to protect.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _endpoint_host(url: str) -> str:
    """The hostname of a configured endpoint, lowercased, brackets stripped."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return host


def assert_endpoint_approved(backend: str | None = None) -> None:
    """Refuse an endpoint host D15 does not name, whatever the provider pin says.

    ``HA_CODEX_BIN`` is deliberately NOT covered here and is not exempt either:
    it names a local executable, and D15's question is who *receives* the data,
    which a filesystem path cannot establish — a check that a path exists would
    look like coverage and provide none. Recorded on #138 as its own axis.
    """
    backend = backend or BACKEND
    approved = APPROVED_ENDPOINT_HOSTS.get(backend)
    if approved is None:
        return
    url = {"openrouter": OPENROUTER_URL, "ollama": OLLAMA_URL}[backend]
    host = _endpoint_host(url)
    if not host:
        raise RuntimeError(
            f"LLM backend {backend!r} is not approved under D15: its endpoint "
            f"{url!r} has no host.")
    if host not in approved:
        raise RuntimeError(
            f"LLM backend {backend!r} is not approved under D15: endpoint host "
            f"{host!r} is not in the approved set "
            f"({', '.join(sorted(approved))}). Set the endpoint back to its "
            f"default, or propose the host with its published terms.")
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if host not in _LOOPBACK_HOSTS and scheme != "https":
        raise RuntimeError(
            f"LLM backend {backend!r} is not approved under D15: endpoint "
            f"{url!r} is not https.")


def _pinned_providers() -> list[str]:
    """HA_OPENROUTER_PROVIDERS parsed into the order sent to OpenRouter."""
    return [name.strip() for name in OPENROUTER_PROVIDERS.split(",") if name.strip()]


def _approved_openrouter_providers() -> frozenset[str] | None:
    """Return the D15 provider set belonging to the configured model."""
    return APPROVED_OPENROUTER_PROVIDERS.get(OPENROUTER_MODEL)


OPENROUTER_REASONING_MODES = ("on", "off", "low")


def _openrouter_reasoning_mode() -> str:
    """Parse the explicitly stated OpenRouter reasoning mode.

    Returns the stated mode itself — ``"on"``, ``"off"`` or ``"low"`` — and not
    a bool: three states do not fit in two, and a bool would have made ``low``
    indistinguishable from ``on`` at every call site that has to decide a
    timeout or a wire field.

    There is intentionally no fallback here. The entry-point guard calls this
    before allowing a provider-facing run, and direct completion calls use the
    same refusal rather than silently selecting the model's own default. Adding
    ``low`` adds a third value a run may STATE; it does not add a default, and
    an unset or misspelled value is refused exactly as before.
    """
    if OPENROUTER_REASONING in OPENROUTER_REASONING_MODES:
        return OPENROUTER_REASONING
    state = "unset" if OPENROUTER_REASONING is None else (
        f"set to invalid value {OPENROUTER_REASONING!r}")
    raise RuntimeError(
        "LLM backend 'openrouter' is not approved under D15: "
        f"HA_OPENROUTER_REASONING is {state}; set it to 'on', 'off' or 'low'."
    )


def _openrouter_reasoning_field(mode: str) -> dict:
    """The ``reasoning`` object one stated mode puts on the wire.

    ``on``/``off`` keep OpenRouter's boolean ``enabled`` form byte-for-byte, so
    every existing run's payload is unchanged. ``low`` is not expressible that
    way: it is an effort level, and OpenRouter reads it from ``effort``.
    """
    if mode == "low":
        return {"effort": "low"}
    return {"enabled": mode == "on"}


def assert_backend_approved() -> None:
    """Refuse a provider-facing process whose backend is outside D15's list."""
    if BACKEND not in APPROVED_BACKENDS:
        raise RuntimeError(
            f"LLM backend '{BACKEND}' is not approved under D15."
        )
    if BACKEND == "codex" and not CODEX_BIN:
        raise RuntimeError(
            "LLM backend 'codex' is not approved: no codex executable was "
            "configured or found on PATH; set HA_CODEX_BIN or put codex on "
            "PATH."
        )
    # The destination, not only the name of who is meant to be at it (#138).
    assert_endpoint_approved(BACKEND)
    if BACKEND != "openrouter":
        return
    if (_OPENROUTER_API_KEY_FILE_VALUE is not None
            and _OPENROUTER_API_KEY_ENV is not None
            and _OPENROUTER_API_KEY_FILE_VALUE != _OPENROUTER_API_KEY_ENV):
        raise RuntimeError(
            "LLM backend 'openrouter' is not approved under D15: "
            "HA_OPENROUTER_API_KEY_FILE and OPENROUTER_API_KEY are both set "
            "but disagree; remove one or make them identical. Refusing to "
            "start rather than silently choosing different credentials.")
    known_models = ", ".join(sorted(APPROVED_OPENROUTER_PROVIDERS))
    if OPENROUTER_MODEL is None:
        raise RuntimeError(
            "LLM backend 'openrouter' is not approved under D15: "
            "HA_OPENROUTER_MODEL is unset; set it to a pinned model "
            f"(known models: {known_models})."
        )
    if OPENROUTER_MODEL.startswith("~"):
        raise RuntimeError(
            "LLM backend 'openrouter' is not approved under D15: "
            f"OPENROUTER_MODEL {OPENROUTER_MODEL!r} is a floating model; "
            "set a pinned model "
            f"(known models: {known_models})."
        )
    approved_providers = _approved_openrouter_providers()
    if approved_providers is None:
        raise RuntimeError(
            "LLM backend 'openrouter' is not approved under D15: "
            f"OPENROUTER_MODEL {OPENROUTER_MODEL!r} has no approved provider "
            f"set (known models: {known_models})."
        )
    approved = ", ".join(sorted(approved_providers))
    pinned = _pinned_providers()
    if not pinned:
        raise RuntimeError(
            "LLM backend 'openrouter' is not approved under D15: "
            f"model {OPENROUTER_MODEL!r}: "
            "HA_OPENROUTER_PROVIDERS must be set and non-empty "
            f"(approved: {approved})."
        )
    # Every name has to be approved, not just one of them: fallbacks are off, so
    # the whole list is who may receive the health data.
    unapproved = [n for n in pinned if n not in approved_providers]
    if unapproved:
        raise RuntimeError(
            "LLM backend 'openrouter' is not approved under D15: "
            f"model {OPENROUTER_MODEL!r}: "
            f"HA_OPENROUTER_PROVIDERS names {', '.join(repr(n) for n in unapproved)}, "
            f"which D15 does not approve (approved: {approved})."
        )
    _openrouter_reasoning_mode()

TIMEOUT_BRIEF = 180       # tool-less, reasoning-OFF call (briefings)
TIMEOUT_THINK = 600       # tool-less, reasoning-ON call (deep-dive planner/compiler/judge)
TIMEOUT_TOOL_TURN = 2400  # per researcher turn (think-ON, tools). The weekly deep dive
                          # is not time-sensitive, so this covers the worst case: a cache
                          # eviction forcing a full reprocess of a compaction-threshold
                          # prompt (~39k tokens ≈ 30 min at this box's ~22 tok/s) + thinking
DEADLINE_TOOL_LOOP = 900  # overall wall-clock for one researcher task

# The ASK path is interactive and must not inherit the researcher's bounds.
# Measured 2026-08-28 on together/deepseek-v4-flash-0731: a whole question
# averages 38 s across ~7 tool turns, so ~5 s per turn. TIMEOUT_TOOL_TURN is
# 2400 s -- 480x that -- because it is sized for an unattended weekly deep dive
# on a ~22 tok/s LOCAL box, and `chat.answer_question` was silently inheriting
# it. A stalled request therefore blocked for up to 40 minutes, and
# DEADLINE_TOOL_LOOP could not cut it short because the deadline is only
# checked BETWEEN turns and cannot interrupt an in-flight request.
#
# Measured consequence: five concurrent battery processes hung for ~26 minutes
# with the provider healthy the whole time (a one-line probe returned in 1.4 s).
# Bounds are derived from the measurement, not guessed: 120 s per turn is 24x
# the observed mean turn, and 300 s per question is well above the slowest
# question yet recorded on this provider.
TIMEOUT_ASK_TURN = int(os.environ.get("HA_ASK_TIMEOUT_TURN", "120"))
DEADLINE_ASK_LOOP = int(os.environ.get("HA_ASK_DEADLINE", "300"))

# Keep enough stderr to explain a failed scheduled run without allowing a CLI
# diagnostic (which can contain a very large prompt/tool dump) to dominate the
# durable scratchpad trace.
CODEX_STDERR_MAX = 2000
CODEX_FAST_FAILURE_SECONDS = 60

# Sampling profiles override the model's poor Modelfile defaults (temp=1, pp=1.5).
GROUNDED_OPTS = {"temperature": 0.3, "top_p": 0.9, "presence_penalty": 0.0}
JUDGE_OPTS = {"temperature": 0.0, "top_p": 0.9, "presence_penalty": 0.0}
CREATIVE_OPTS = {"temperature": 0.7, "top_p": 0.95}  # deep-dive generative

# Tests set this to an httpx.MockTransport; None = real network transport.
_TRANSPORT: httpx.BaseTransport | None = None

# A single-shot completion has the same string contract as before, including
# the legitimate ``""`` result.  This side channel records whether that empty
# string came from a completed model response or from a backend that could not
# produce one.  ``call_id`` lets callers distinguish a fresh status from a
# status left by an earlier call without clearing the loop status semantics.
_COMPLETE_CALL_ID = 0
_LAST_COMPLETE_STATUS = {
    "call_id": 0,
    "outcome": "not_called",
    "backend": None,
    "response_received": None,
    "request_made": False,
    "detail": "",
    "text_length": 0,
}


def last_complete_status() -> dict:
    """Return the most recent single-shot completion status.

    The return value of :func:`complete` deliberately stays a string.  A
    non-empty answer has ``outcome == "success"`` and
    ``response_received == True``; an empty answer has a different outcome and
    ``response_received == False``.  This is an inspectable transport fact,
    independent of the coaching length gate or the model's interpretation.
    """
    return dict(_LAST_COMPLETE_STATUS)


def _begin_complete() -> int:
    global _COMPLETE_CALL_ID
    _COMPLETE_CALL_ID += 1
    _LAST_COMPLETE_STATUS.clear()
    _LAST_COMPLETE_STATUS.update({
        "call_id": _COMPLETE_CALL_ID,
        "outcome": "in_progress",
        "backend": BACKEND,
        "response_received": None,
        "request_made": False,
        "detail": "",
        "text_length": 0,
    })
    return _COMPLETE_CALL_ID


def _set_complete_status(*, outcome: str, response_received: bool,
                         request_made: bool, detail: str = "",
                         text_length: int = 0) -> None:
    _LAST_COMPLETE_STATUS.update({
        "outcome": outcome,
        "backend": BACKEND,
        "response_received": response_received,
        "request_made": request_made,
        "detail": str(detail)[:CODEX_STDERR_MAX],
        "text_length": text_length,
    })


# The researcher response remains string-compatible with the pipeline while
# carrying its structured claim channel. This side channel lets a caller decide
# whether an empty string was
# a legitimate empty answer or a backend failure, and is also emitted in the
# research scratchpad log below.
_LAST_CODEX_STATUS = {
    "outcome": "not_called",
    "backend_broken": False,
    "retryable": False,
    "fast_failure_signature": False,
    "returncode": None,
    "stderr": "",
    "elapsed_seconds": 0.0,
    "timeout_seconds": None,
}


def last_codex_status() -> dict:
    """Return a copy of the most recent codex call's JSON-safe status.

    ``complete()`` returns a string; researcher loops return a string-compatible
    ``ResearchResponse`` with claims. Callers that need failure policy inspect this
    status immediately after the call: ``outcome`` is ``success`` or
    ``empty_success`` for a completed process, while the other values identify
    backend failures. ``retryable`` is true only for a normal empty success;
    backend failures, including the fast-empty auth/rate-limit signature, are
    not safe to re-run immediately. ``stderr`` is already bounded.
    """
    return dict(_LAST_CODEX_STATUS)


# #128: the agentic loops return an empty ResearchResponse on every failure, and
# an empty answer is a LEGITIMATE result — the `discovery` deep dive is
# documented as one that may honestly find nothing. So "this backend cannot
# serve the call" was indistinguishable from "the researcher found nothing", and
# `tool_loop` produced no trace at all. Every non-answer now lands in three
# places at once: the caller's on_log when it has one, stderr so a scheduled
# run's journal carries it, and this side channel for a caller that wants to
# branch on it. It is a separate dict from _LAST_CODEX_STATUS because that one
# describes a subprocess and these are HTTP loops.
_LOOP_EVENT_ID = 0
_LAST_LOOP_STATUS = {"call_id": 0, "outcome": "not_called", "backend": None, "detail": ""}

# The submit_answer repair turn's counters. PROCESS-CUMULATIVE, never reset by a
# loop, because a caller like `chat.answer_question` runs two `tool_loop`s per
# question and a per-loop counter would report only the second. Readers that
# already snapshot `last_loop_status()` before and after a call (scripts/
# ask_battery.py does) get the per-question figure by subtraction. They ride on
# `last_loop_status` rather than in a new channel because that is the side
# channel a caller already reads to explain a non-answer.
_SUBMIT_REPAIRS = {"attempted": 0, "succeeded": 0}


def last_loop_status() -> dict:
    """The most recent `tool_loop`/`research_loop` non-answer, JSON-safe.

    ``outcome`` is ``not_called`` until a loop declines or fails. It is not
    cleared by a successful loop: a caller reads it immediately after receiving
    an empty answer, to tell a refusal from a genuinely empty finding.

    ``submit_repairs_attempted``/``submit_repairs_succeeded`` count the
    in-loop submit_answer repair turns (see :data:`_SUBMIT_REPAIRS`) since the
    process started, so they are read as a difference across a call.
    """
    status = dict(_LAST_LOOP_STATUS)
    status["submit_repairs_attempted"] = _SUBMIT_REPAIRS["attempted"]
    status["submit_repairs_succeeded"] = _SUBMIT_REPAIRS["succeeded"]
    return status


def _announce(event: str, detail: str = "", *, on_log=None, turn: int = 0) -> None:
    """Record and surface a loop outcome that is not a model answer."""
    global _LOOP_EVENT_ID
    _LOOP_EVENT_ID += 1
    _LAST_LOOP_STATUS.clear()
    _LAST_LOOP_STATUS.update({"call_id": _LOOP_EVENT_ID, "outcome": event,
                              "backend": BACKEND,
                              "detail": str(detail)[:CODEX_STDERR_MAX]})
    if on_log:
        try:
            on_log(turn, event, str(detail)[:CODEX_STDERR_MAX])
        except Exception:
            pass
    try:
        print(f"[llm.{event}] backend={BACKEND} {detail}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _stderr_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")[:CODEX_STDERR_MAX]
    return str(value)[:CODEX_STDERR_MAX]


def _set_codex_status(*, outcome: str, returncode=None, stderr="",
                      elapsed_seconds: float = 0.0, timeout_seconds=None) -> dict:
    fast = (outcome != "timeout" and not outcome == "success"
            and elapsed_seconds < CODEX_FAST_FAILURE_SECONDS
            and timeout_seconds is not None and timeout_seconds >= DEADLINE_TOOL_LOOP)
    broken = outcome not in {"success", "empty_success"}
    status = {
        "outcome": outcome,
        "backend_broken": broken,
        "retryable": outcome == "empty_success" and not fast,
        "fast_failure_signature": fast,
        "returncode": returncode,
        "stderr": _stderr_text(stderr),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "timeout_seconds": timeout_seconds,
    }
    _LAST_CODEX_STATUS.clear()
    _LAST_CODEX_STATUS.update(status)
    return status


def _diagnostic_outcome(returncode, stderr: str, *, has_output: bool,
                        exception=None) -> str:
    """Classify the failure using stderr before falling back to process status."""
    if isinstance(exception, FileNotFoundError):
        return "binary_missing"
    if isinstance(exception, subprocess.TimeoutExpired):
        return "timeout"
    lower = stderr.lower()
    if re.search(r"auth|unauthori[sz]ed|credential|api key|token|login|\b401\b", lower) \
            and (not has_output or returncode != 0):
        return "auth_failure"
    if re.search(r"rate limit|too many requests|\b429\b|quota|resource exhausted", lower) \
            and (not has_output or returncode != 0):
        return "rate_limited"
    if exception is not None:
        return "process_error"
    if returncode != 0:
        return "nonzero_exit"
    return "empty_success" if not has_output else "success"


# ---------------------------------------------------------------------------
# codex backend — every call is one `codex exec` subprocess
# ---------------------------------------------------------------------------

def _codex_exec(prompt: str, *, reasoning: str = "medium", timeout: int,
                config: tuple[str, ...] = ()) -> str:
    """Run one non-interactive ``codex exec`` and return its final message.

    The return value remains ``""`` on missing binary, timeout, non-zero exit,
    or empty output because existing callers use that fail-closed behavior and
    must not be taken down by a model outage. The side-channel
    :func:`last_codex_status` records which case occurred, including bounded
    stderr, so callers can avoid retrying a broken backend and traces can show
    why no answer was produced. Sandbox is read-only and sessions are
    ephemeral; ``config`` entries are extra ``-c key=value`` TOML overrides.
    """
    out_path = None
    started = time.monotonic()
    proc = None
    stderr = ""
    try:
        fd, out_path = tempfile.mkstemp(prefix="ha_codex_", suffix=".txt")
        os.close(fd)
        cmd = [CODEX_BIN, "exec", "--skip-git-repo-check", "--ephemeral",
               "--color", "never", "-C", REPO_ROOT,
               "-s", "read-only", "-m", CODEX_MODEL,
               "-c", 'approval_policy="never"',
               "-c", f'model_reasoning_effort="{reasoning}"',
               "-c", "project_doc_max_bytes=0"]
        for kv in config:
            cmd += ["-c", kv]
        cmd += ["-o", out_path, "-"]  # "-" = prompt on stdin
        proc = subprocess.run(cmd, input=prompt.encode(), capture_output=True,
                              timeout=timeout, cwd=REPO_ROOT)
        stderr = _stderr_text(getattr(proc, "stderr", ""))
        with open(out_path) as fh:
            text = fh.read().strip()
        outcome = _diagnostic_outcome(proc.returncode, stderr,
                                      has_output=bool(text))
        _set_codex_status(outcome=outcome, returncode=proc.returncode,
                          stderr=stderr, elapsed_seconds=time.monotonic() - started,
                          timeout_seconds=timeout)
        return text if outcome == "success" else ""
    except Exception as exc:
        stderr = _stderr_text(getattr(exc, "stderr", ""))
        outcome = _diagnostic_outcome(getattr(proc, "returncode", None), stderr,
                                      has_output=False, exception=exc)
        _set_codex_status(outcome=outcome,
                          returncode=getattr(proc, "returncode", None),
                          stderr=stderr, elapsed_seconds=time.monotonic() - started,
                          timeout_seconds=timeout)
        return ""
    finally:
        if out_path:
            try:
                os.unlink(out_path)
            except OSError:
                pass


def _deepdive_mcp_config(ctx, scratch_path: str | None = None,
                         task_id=None, ledger_path: str | None = None,
                         include=None) -> tuple[str, ...]:
    """-c overrides that hand codex exactly one MCP server: the curated
    health-deepdive stdio server (read-only research tools, plus the run-scoped
    notepad when a scratchpad is given and an append-only call ledger when a
    ledger path is given). Nothing else from ~/.codex/config.toml
    can reach the researcher because these overrides replace `mcp_servers`.

    The vault goes in the launcher's ARGV, not its environment. A subprocess
    inherits the environment; it does not inherit argv, so a spawned researcher
    can only ever open the vault this session named. A scratch-backed run gets a
    sibling ``*_ledger.jsonl`` path unless its caller supplies one explicitly."""
    launcher = os.path.join(REPO_ROOT, "deepdive_mcp_launch.py")
    launch_args = [launcher, "--vault", str(ctx.db_path), "--user", ctx.user_id]
    if scratch_path:
        launch_args += ["--scratch", scratch_path, "--task-id", str(task_id)]
        ledger_path = _derived_ledger_path(scratch_path, ledger_path)
    if ledger_path:
        launch_args += ["--ledger", ledger_path]
    if include is not None:
        # Defense in depth: analyst_query is an in-process-only seam and must
        # never be forwarded to the codex stdio MCP server, even if a caller
        # accidentally supplies the full coach tool tuple here.
        codex_include = set(include) - {ANALYST_QUERY_NAME}
        launch_args += ["--include", ",".join(sorted(codex_include))]
    args_toml = ", ".join(f'"{a}"' for a in launch_args)
    return ("mcp_servers={}",
            f'mcp_servers.health-deepdive.command="{sys.executable}"',
            f"mcp_servers.health-deepdive.args=[{args_toml}]",
            # exec mode auto-DENIES MCP calls that would prompt (stdin is
            # closed), so the read-only server must be pre-approved
            'mcp_servers.health-deepdive.default_tools_approval_mode="approve"')

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Defensively remove any <think>…</think> block. The native /api/chat with
    think:true keeps reasoning in message.thinking (content is already clean);
    this only guards a future config that inlines it."""
    return _THINK_RE.sub("", text or "")


def _openrouter_provider() -> dict:
    """The `provider` routing block, or {} to let OpenRouter choose.

    An explicit allow-list wins and disables fallbacks: a pin that silently
    fails over to an unpinned provider is not a pin, and here the thing being
    pinned is who receives the health data.
    """
    order = _pinned_providers()
    if order:
        return {"order": order, "allow_fallbacks": False,
                "require_parameters": True}
    if OPENROUTER_PROVIDER_SORT:
        return {"sort": OPENROUTER_PROVIDER_SORT}
    return {}


def _assert_openrouter_response_provider(data: dict) -> None:
    """Refuse a response served by a provider D15 does not approve.

    OpenRouter returns a display name, not the endpoint tag used in the
    request. A reported provider must be a known exact name; in particular,
    this never uses substring or prefix matching. Some existing OpenAI-
    compatible response shapes omit the optional field, so there is no
    provider identity to translate in that case.
    """
    if not isinstance(data, dict) or "provider" not in data:
        return
    display_name = data.get("provider")
    tag = (OPENROUTER_PROVIDER_TAGS.get(display_name)
           if isinstance(display_name, str) else None)
    approved_providers = _approved_openrouter_providers()
    if approved_providers is None:
        detail = (
            f"OpenRouter served provider {display_name!r} for model "
            f"{OPENROUTER_MODEL!r}, which has no approved provider set "
            f"(known models: {', '.join(sorted(APPROVED_OPENROUTER_PROVIDERS))})."
        )
        _announce("openrouter_provider_mismatch", detail)
        raise RuntimeError(detail)
    if tag not in approved_providers:
        detail = (
            f"OpenRouter served provider {display_name!r}, which maps to "
            f"{tag!r}; it is not approved for model {OPENROUTER_MODEL!r}. "
            f"Approved endpoint tags for this model are "
            f"{', '.join(sorted(approved_providers))}."
        )
        _announce("openrouter_provider_mismatch", detail)
        raise RuntimeError(detail)


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout, transport=_TRANSPORT)


def _openrouter_api_key() -> str:
    """Get the process-configured key without logging or exposing it."""
    return OPENROUTER_API_KEY


def openrouter_credits(timeout: float = 10) -> dict[str, float]:
    """Read the OpenRouter balance after enforcing the normal D15 admission.

    This is intentionally separate from model completion: a healthy balance is
    only a preflight signal, never evidence that a later generation returned
    content. The caller must still inspect completion/loop status.
    """
    assert_backend_approved()
    if BACKEND != "openrouter":
        raise RuntimeError("OpenRouter credits require the openrouter backend")
    api_key = _openrouter_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenRouter credits unavailable: OPENROUTER_API_KEY is unset. "
            "The key is stated at the entry point like the provider pin "
            "(D15); this module deliberately reads no key file, because a "
            "path baked in here is one deployment's layout and a credential "
            "read off disk is an implicit default.")
    with _client(timeout) as client:
        response = client.get(
            f"{OPENROUTER_URL}/credits",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()["data"]
    total = float(data["total_credits"])
    usage = float(data["total_usage"])
    return {"total_credits": total, "total_usage": usage,
            "remaining": total - usage}


def complete(prompt: str, *, think: bool = False, timeout: int | None = None,
             options: dict | None = None) -> str:
    """Single-shot text/JSON completion. Returns the model's message content, or
    "" on ANY error (timeout, transport, non-200, bad JSON, empty content).
    On the codex backend `options` (Ollama sampling) is ignored and `think`
    maps to reasoning effort."""
    _begin_complete()
    openrouter_reasoning = None
    if BACKEND == "openrouter":
        try:
            openrouter_reasoning = _openrouter_reasoning_mode()
        except RuntimeError as exc:
            _set_complete_status(outcome="backend_error", response_received=False,
                                 request_made=False, detail=str(exc))
            return ""
    if timeout is None:
        if BACKEND == "openrouter":
            # OpenRouter ignores `think`; its explicit wire setting owns the
            # ceiling. Codex and Ollama retain the think-based selection below.
            # `low` still reasons, so it takes the thinking ceiling: only a
            # stated `off` buys the short one.
            timeout = (TIMEOUT_BRIEF if openrouter_reasoning == "off"
                       else TIMEOUT_THINK)
        else:
            timeout = TIMEOUT_THINK if think else TIMEOUT_BRIEF
    if BACKEND == "codex":
        text = _codex_exec(prompt, reasoning="high" if think else "medium",
                           timeout=timeout, config=("mcp_servers={}",))
        codex_status = last_codex_status()
        _set_complete_status(
            outcome=codex_status["outcome"],
            response_received=bool(text),
            request_made=True,
            detail=codex_status.get("stderr", ""),
            text_length=len(text),
        )
        return text
    if BACKEND == "openrouter":
        api_key = _openrouter_api_key()
        if not api_key:
            _set_complete_status(outcome="no_api_key", response_received=False,
                                 request_made=False,
                                 detail="OPENROUTER_API_KEY is unset")
            return ""
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "reasoning": _openrouter_reasoning_field(openrouter_reasoning),
            **_openai_sampling(options),
        }
        provider = _openrouter_provider()
        if provider:
            payload["provider"] = provider
        try:
            with _client(timeout) as client:
                resp = client.post(
                    f"{OPENROUTER_URL}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
            _assert_openrouter_response_provider(data)
            content = data["choices"][0]["message"]["content"]
            text = _strip_think(content).strip()
            _set_complete_status(
                outcome="success" if text else "empty_response",
                response_received=bool(text), request_made=True,
                detail="" if text else "model content was empty",
                text_length=len(text),
            )
            return text
        except httpx.TimeoutException as exc:
            # A timeout is NOT a dead backend. The sweep aborts on the first
            # no-response job by design (#178); one slow generation must not
            # be able to trigger that, or a long tail ends the whole run.
            _set_complete_status(outcome="timeout", response_received=False,
                                 request_made=True,
                                 detail=f"{type(exc).__name__}: {exc}")
            return ""
        except Exception as exc:
            _set_complete_status(outcome="backend_error", response_received=False,
                                 request_made=True,
                                 detail=f"{type(exc).__name__}: {exc}")
            return ""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
        "options": options or GROUNDED_OPTS,
        "keep_alive": KEEP_ALIVE,
    }
    try:
        with _client(timeout) as client:
            resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = ((data.get("message") or {}).get("content")) or ""
        text = _strip_think(content).strip()
        _set_complete_status(
            outcome="success" if text else "empty_response",
            response_received=bool(text), request_made=True,
            detail="" if text else "model content was empty",
            text_length=len(text),
        )
        return text
    except httpx.TimeoutException as exc:
        _set_complete_status(outcome="timeout", response_received=False,
                             request_made=True,
                             detail=f"{type(exc).__name__}: {exc}")
        return ""
    except Exception as exc:
        _set_complete_status(outcome="backend_error", response_received=False,
                             request_made=True,
                             detail=f"{type(exc).__name__}: {exc}")
        return ""


# Read-only tools exposed to the deep-dive researcher. write_insight and the
# plan tools (get_planned_session/get_week_plan) are intentionally excluded —
# the researcher only reads health metrics.
RESEARCHER_TOOLS = ("list_available_metrics", "get_daily_series", "summarize_metric",
                    "compare_periods", "get_intraday", "list_workouts",
                    "get_latest", "get_briefing",
                    # PLAN.md's two central rules are the +15%/week impact ramp
                    # and the 150 bpm HR cap. The weekly deep dive was asked to
                    # judge both while holding neither instrument: workout
                    # duration counts walk breaks (~2x impact) and a session
                    # mean hides time over the cap.
                    "get_impact_volume", "get_hr_zones",
                    "get_sleep_regularity", "get_training_load_detail",
                    "correlate_metrics", "scan_correlations")

# The interactive question path is read-only, but deliberately broader than
# the autonomous researcher. It includes coach/planning and specialist read
# tools so the user's question, rather than a researcher's task template,
# chooses the useful instrument. The surface is bound from
# ``ctx.provider_facing()``; write tools and raw samples are absent by
# construction.
COACH_TOOLS = (
    "list_available_metrics", "get_daily_series", "summarize_metric",
    "compare_periods", "get_intraday", "get_hr_zones", "list_workouts",
    "get_workout_segments", "get_impact_volume", "get_sleep_regularity",
    "get_training_load_detail", "get_run_form", "get_briefing", "get_latest",
    "correlate_metrics", "scan_correlations", "get_planned_session",
    "get_week_plan", "get_plan_overview", "get_subjective", "food_lookup",
    "food_meal_total", "get_weekly_series", "get_block_structure",
    "get_weekly_readiness", "get_benchmark_series", "get_monthly_running_power",
    "analyst_query",
)

ANALYST_QUERY_NAME = "analyst_query"
_ANALYST_QUERY_SCHEMA = {
    "type": "function",
    "function": {
        "name": ANALYST_QUERY_NAME,
        "description": "Run a validated analyst query against the health vault.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


def _analyst_tool(question: str, *, analyst_query_fn=None) -> dict:
    """Dispatch the in-process analyst seam, never the codex MCP server."""
    if analyst_query_fn is None:
        return {"refused": True,
                "reason": "analyst_query is unavailable on this tool path"}
    if not isinstance(question, str) or not question.strip():
        return {"refused": True, "reason": "question must be a non-empty string"}
    return analyst_query_fn(question.strip())


def _registry(ctx, include=None, *, analyst_query_fn=None) -> dict:
    """{name: (callable, ollama_schema)} for one session's selected tools,
    sourced from real FastMCP tool objects so schemas can't drift from the
    signatures/docstrings.

    Deliberately NOT cached. The callables close over `ctx`, so a process-wide
    memo would hand a later session the earlier session's vault — which is the
    whole defect T-003 removed. Building costs one FastMCP construction per
    research run, against a run that then spends minutes in a model."""
    from . import mcp_server as S
    # Narrowed here, once, rather than trusted to each tool: the researcher's
    # answers go to a model provider, so it gets a session without RAW_SAMPLES
    # and without WRITE. A tool cannot forget a capability it was never handed.
    selected = frozenset(RESEARCHER_TOOLS if include is None else include)
    # analyst_query is deliberately not an MCP tool. Analyst mode is exempt
    # from the provider-boundary raw-sample guarantee until its service is
    # separately mediated; keeping it out of build_server preserves that
    # boundary's existing scope and keeps the codex subprocess unable to see it.
    mcp_selected = selected - {ANALYST_QUERY_NAME}
    server = S.build_server(ctx.provider_facing(), name="health-deepdive",
                            include=mcp_selected)
    registry = {t.name: (t.fn, {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.parameters,
        },
    }) for t in server._tool_manager.list_tools()}
    if ANALYST_QUERY_NAME in selected:
        registry[ANALYST_QUERY_NAME] = (
            lambda question: _analyst_tool(
                question, analyst_query_fn=analyst_query_fn),
            _ANALYST_QUERY_SCHEMA,
        )
    return registry


def tool_schemas(ctx, include=None) -> list[dict]:
    """Tool schemas for one provider-facing surface."""
    return [schema for _, schema in _registry(ctx, include=include).values()]


class ResearchResponse(str):
    """String-compatible researcher response with its structured claim channel."""

    def __new__(cls, text: str = "", claims=None):
        obj = super().__new__(cls, text or "")
        obj.text = text or ""
        obj.claims = claims if isinstance(claims, list) else None
        return obj

    def as_dict(self) -> dict:
        return {"text": self.text, "claims": self.claims}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.as_dict()[key]
        return super().__getitem__(key)


def _research_response(raw) -> ResearchResponse:
    """Parse the shared {text, claims} channel without adding a tokenizer."""
    from .agents import split_claim_channel

    if isinstance(raw, ResearchResponse):
        return raw
    text, claims = split_claim_channel(raw or "")
    return ResearchResponse(text, claims)


_RESEARCH_CLAIM_INSTRUCTIONS = (
    "\n\nRESEARCH CLAIM CHANNEL: When you finish, return a JSON object with keys "
    "text and claims. text is the prose answer. claims is a list of the same "
    "scoped objects used by the coach: {metric, period, field, value}; "
    "`value` may be the exact number or exact string from a Python-owned "
    "`presentation` leaf. Each research claim must additionally include "
    "source {sequence, path}. The "
    "sequence is the _ledger.sequence shown in the tool result. The path must "
    "be the exact JSON path in the ledger, rooted as $.result... or "
    "$.arguments.... Name the field and metric that the path actually contains; "
    "EXCEPTION - a row published by list_workouts carries workout_key and "
    "belongs to no metric series: for a number taken from such a row omit "
    "metric entirely. Naming a metric there is refused, and a per-session "
    "workout value must never be relabelled as a daily-metric series. "
    "For period, copy one of the exact strings in "
    "_ledger.period_vocabulary[*].claim_period verbatim; do not compose an ISO "
    "range or reinterpret the ledger's internal period.end. A structured period "
    "dict may be copied only when it is present in the published vocabulary. "
    "do not use prose to infer a source. Derived claims use operation and "
    "operands, with every operand carrying its own source. Do not report a "
    "number in text unless its claim is present.\n"
    "`get_weekly_series` publishes each row's inclusive Monday-Sunday `period` "
    "as `YYYY-MM-DD:YYYY-MM-DD`; copy that exact string into a claim and "
    "never invent a period from `week_start`. "
    "Metric ownership is per field: inherit a row's `metric` only for its own "
    "series-value leaves (`mean`, `median`, `min`, `max`, `std`, `latest`, "
    "`sum`, `recent_avg`, `baseline_avg`, `delta_pct`, `slope_per_week`) or a "
    "leaf whose field is exactly that metric; never inherit it for context "
    "fields such as `n_days`, `rho`, `sd_day`, `mdc95`, `unit`, dates, day "
    "counts, or other siblings. "
    "`get_subjective` keeps flat day fields and adds `period` equal to the day "
    "plus `field_metrics` for the non-null rating fields (`stress`, `soreness`, "
    "`energy`, `sleep_quality`); cite the direct field with its mapped "
    "`subjective_*` metric, and omit `metric` for fields absent from "
    "`field_metrics`. "
    "`list_workouts` publishes full-range per-type counts as "
    "`workout_counts: [{type, count}]`; cite the `count` leaf at its exact "
    "path with `metric` omitted, and never count the possibly truncated "
    "`workouts` rows."
)


# ---------------------------------------------------------------------------
# submit_answer — the claim channel as a typed tool call rather than as prose.
#
# The ask path used to terminate by parsing the model's final free-text message
# for a {text, claims} JSON object. With reasoning off the model frequently
# emitted bare prose (sometimes with a visible chain-of-thought preamble), so
# `claims` came back None, every figure was unsupported and the user got
# "Answer withheld" — a claim-CHANNEL failure, not a capability failure.
#
# This is a SYNTHETIC terminal tool. It is deliberately NOT in `_registry`: it
# reads nothing, touches no vault, and must never be wrapped by `_ledgered`,
# because a ledger entry for it would hand the model a way to satisfy the ask
# gate's "no tool-call ledger" refusal without ever reading data. The prose
# instructions still teach the claim grammar; the schema only guarantees the
# channel is PRESENT and TYPED. Encoding the full grammar in JSON Schema would
# move a verification decision out of Python, which is the one rule.
# ---------------------------------------------------------------------------

SUBMIT_ANSWER_NAME = "submit_answer"

# A SERIALISATION hint, not validation. `claims` used to be
# `{"type": "object"}` with no properties at all, and providers serialising an
# untyped object array frequently handed the array back as a *string* of
# malformed JSON (measured 2026-08-28: `{"sequence":  ript 19, ...}`, the token
# `ript` injected mid-JSON, which cost 3 of 6 retry attempts entirely). Naming
# the keys gives the provider's structured-output machinery a shape to fill.
#
# Deliberately absent: every `required`, enum, format and constraint. Nothing
# here decides whether a claim is TRUE, or even whether it is legal — the claim
# grammar stays in prose and the verdict stays in Python's verifier. `metric`
# in particular MUST stay omittable: the grammar requires omitting it for
# list_workouts rows. `period` and `value` carry no `type` because both are
# legitimately several JSON types (period is a string or null; value is a
# number or a string), and a type here would refuse a legal claim at the wire
# instead of at the gate.
CLAIM_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "metric": {
            "type": "string",
            "description": ("The metric key this number belongs to. OMIT it "
                            "where the claim grammar says to."),
        },
        "period": {
            "description": ("The display period the number covers, copied "
                            "verbatim from the tool result."),
        },
        "field": {
            "type": "string",
            "description": "The field within the result that holds the value.",
        },
        "value": {
            "description": ("The number exactly as the tool published it, not "
                            "rounded or recomputed."),
        },
        "source": {
            "type": "object",
            "description": ("The tool-call ledger record this number was read "
                            "from."),
            "properties": {
                "sequence": {
                    "type": "integer",
                    "description": ("The _ledger.sequence of the call that "
                                    "returned this number."),
                },
                "path": {
                    "type": "string",
                    "description": ("The exact JSON path within that call's "
                                    "result, e.g. $.result.mean."),
                },
            },
        },
    },
}

SUBMIT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_ANSWER_NAME,
        "description": (
            "Deliver your final answer. Call this INSTEAD of writing a final "
            "message — do not write the answer as JSON in a message. Calling "
            "it ends the conversation. It reads no data: call the read-only "
            "data tools first, then call this once with the prose and the "
            "claims backing every number in it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The coach prose the user will read.",
                },
                "claims": {
                    "type": "array",
                    "description": (
                        "One claim object per number in text, in the claim "
                        "grammar given in the instructions. Send an empty "
                        "array only if text states no number at all."
                    ),
                    "items": CLAIM_ITEM_SCHEMA,
                },
            },
            "required": ["text", "claims"],
        },
    },
}


def _submit_answer_call(calls) -> dict | None:
    """The terminal ``submit_answer`` call in one assistant turn, if any.

    Scanned across the WHOLE turn before any tool runs. A model that calls a
    data tool and ``submit_answer`` in the same turn wrote its answer without
    ever seeing that tool's result, so running the data tool first would put a
    ledger entry behind an answer it cannot have supported. Terminating instead
    leaves the ledger honest and lets the gate refuse on the evidence.
    """
    for call in calls or []:
        if ((call.get("function") or {}).get("name")) == SUBMIT_ANSWER_NAME:
            return call
    return None


# --------------------------------------------------------------------------
# The repair turn — a malformed submit_answer as a tool error, not a dead end
#
# Measured 2026-08-28 over eight ask-battery runs (reasoning off,
# deepseek-v4-flash-0731 on coreweave/fp8): 24 of 26 loop failures were the
# claims channel arriving malformed —
# submit_answer_unparseable_claims 21, submit_answer_bad_claims 3 — and
# `no_model_response` was the most common outcome of every arm. The corruption
# is not double-encoding of valid JSON: it is garbage injected mid-JSON, e.g.
# `{"sequence":  ript 19, "path": "$.result.points[2].value"}` at char 555.
#
# The model usually gets the CONTENT right (one measured attempt filed 11
# well-formed claims and the retry corrupted a similar set), so this is a
# SERIALISATION failure, not a reasoning one — which is why the fix is to say
# what was malformed and let the model re-emit, inside the same loop, as
# ordinary tool-error handling. It is far cheaper than `chat.answer_question`'s
# whole second attempt.
#
# What the repair deliberately does NOT do: nothing here filters, coerces,
# pre-validates or reconstructs a claim. A repaired submit_answer is not
# evidence of anything — its claims go to _verify_ask_answer exactly as an
# uncorrupted one's would, and a submit-only answer is still refused for an
# empty ledger. Python still owns the truth.
# --------------------------------------------------------------------------

# The excerpt shown around a JSON failure offset. Small on purpose: the offset
# is genuinely useful for repair, but pasting the whole corrupt payload back is
# large, is corrupt, and invites the same garbage a second time.
SUBMIT_ANSWER_EXCERPT_CHARS = 60

# Repairs allowed per loop. Bounded so it is impossible to spend the whole turn
# budget re-asking for a well-formed channel; the measured failure is one bad
# serialisation, not an endless one.
SUBMIT_ANSWER_REPAIR_BUDGET = 2

_SUBMIT_ANSWER_REPAIR_SHAPE = (
    "Required shape: call submit_answer again with exactly two arguments — "
    "`text`, a string of coach prose, and `claims`, a JSON ARRAY of claim "
    "objects (not a string containing an array, not an object). Re-send the "
    "same answer, correctly encoded."
)


def _json_error_excerpt(raw, exc) -> str:
    """The text around a JSONDecodeError's offset, or "" when there is none."""
    pos = getattr(exc, "pos", None)
    if not isinstance(raw, str) or not isinstance(pos, int):
        return ""
    half = SUBMIT_ANSWER_EXCERPT_CHARS // 2
    start = max(0, pos - half)
    end = min(len(raw), pos + half)
    return (f" The text around character {pos} is: "
            f"{'...' if start > 0 else ''}{raw[start:end]}"
            f"{'...' if end < len(raw) else ''}")


def _submit_answer_repair(problem: str, excerpt: str = "") -> str:
    """The corrective tool result handed back for one malformed shape."""
    return ("submit_answer was NOT accepted and your answer was not delivered: "
            f"{problem}{excerpt} {_SUBMIT_ANSWER_REPAIR_SHAPE}")


def _submit_answer_decode(call: dict):
    """Decode one ``submit_answer`` call, without announcing anything.

    Returns ``(response, event, detail, repair)``. ``response`` is None exactly
    when the channel's SHAPE was rejected, and ``repair`` then carries the
    corrective text for the model. A well-formed but empty-text call is not a
    shape failure: it returns its response and the ``tool_loop_empty_answer``
    event, with no repair, exactly as before.

    The claims are passed through UNTOUCHED — not filtered, repaired, coerced
    or pre-validated. Verification belongs to Python's gate downstream, and a
    ``submit_answer`` call is never itself evidence of grounding.
    """
    args = (call.get("function") or {}).get("arguments")
    if args is None:
        args = {}
    if isinstance(args, str):
        raw_args = args
        try:
            args = json.loads(args)
        except Exception as exc:
            return (None, "submit_answer_unparseable_arguments",
                    f"{type(exc).__name__}: {exc}",
                    _submit_answer_repair(
                        "its arguments were not valid JSON "
                        f"({type(exc).__name__}: {exc}).",
                        _json_error_excerpt(raw_args, exc)))
    if not isinstance(args, dict):
        return (None, "submit_answer_bad_arguments",
                f"arguments decoded to {type(args).__name__}, not an object",
                _submit_answer_repair(
                    f"its arguments decoded to {type(args).__name__}, not an "
                    "object."))
    text = args.get("text")
    claims = args.get("claims")
    if not isinstance(text, str):
        return (None, "submit_answer_bad_text",
                f"text was {type(text).__name__}, not a string",
                _submit_answer_repair(
                    f"`text` was {type(text).__name__}, not a string."))
    if isinstance(claims, str):
        # Measured 2026-08-28: 3 of 6 ask-battery questions arrived with the
        # claims ARRAY double-encoded — a JSON string nested inside the
        # already-JSON `arguments` string. Decoding it is a WIRE concern and
        # not a verification one: `arguments` is decoded three lines up by the
        # same call, and the result still goes to _verify_ask_answer untouched.
        # This does not loosen a gate; a claim that decodes is still a claim
        # that must survive Python, and an empty or bogus one still fails there.
        raw_claims = claims
        try:
            claims = json.loads(claims)
        except Exception as exc:
            return (None, "submit_answer_unparseable_claims",
                    f"claims was a string that is not JSON: "
                    f"{type(exc).__name__}: {exc}",
                    _submit_answer_repair(
                        "`claims` arrived as a string, and that string was not "
                        f"valid JSON ({type(exc).__name__}: {exc}).",
                        _json_error_excerpt(raw_claims, exc)))
    if not isinstance(claims, list):
        return (None, "submit_answer_bad_claims",
                f"claims was {type(claims).__name__}, not a list",
                _submit_answer_repair(
                    f"`claims` was {type(claims).__name__}, not an array."))
    if not text.strip():
        return (ResearchResponse(text, claims), "tool_loop_empty_answer",
                "submit_answer carried no text", None)
    return (ResearchResponse(text, claims), None, "", None)


def _submit_answer_response(call: dict) -> ResearchResponse:
    """Decode one ``submit_answer`` call into the claim channel.

    Only the channel's shape is checked, and a malformed call is announced
    (#128) and degraded to the loops' empty return rather than raised or
    papered over. The repair turn in :func:`tool_loop` is what may recover
    from that; this remains the terminal reading.
    """
    response, event, detail, _repair = _submit_answer_decode(call)
    if event:
        _announce(event, detail)
    return response if response is not None else ResearchResponse()


# ---------------------------------------------------------------------------
# OpenAI dialect (#128) — the shape the openrouter backend's tool path speaks.
#
# There is no provider-neutral loop that openrouter opted out of: below this
# point there was an Ollama loop and a codex loop. Ollama posts its own dialect
# to /api/chat (`think`, `options`, `keep_alive`) and keys a tool result by
# `tool_name`; OpenAI/OpenRouter post to /chat/completions and key a tool result
# by `tool_call_id` — the id of one specific call inside the assistant turn.
# These helpers are the whole of the difference; the loops themselves are shared,
# so the turn limit, deadline, compaction, stall detection and
# "return empty rather than something wrong" contract are the same code on
# every backend.
# ---------------------------------------------------------------------------

def _openai_sampling(options: dict | None) -> dict:
    """The subset of an Ollama sampling profile the OpenAI wire accepts."""
    source = options or GROUNDED_OPTS
    return {key: source[key] for key in
            ("temperature", "top_p", "presence_penalty") if key in source}


def _derived_ledger_path(scratch_path, ledger_path):
    """The run's ledger path: the caller's explicit one, else the scratchpad's
    sibling. ``agent_loop`` reads the artifact back by exactly this name, so the
    MCP config and the local transports have to derive it identically."""
    if ledger_path:
        return ledger_path
    if scratch_path:
        stem, _ = os.path.splitext(scratch_path)
        return stem + "_ledger.jsonl"
    return None


def _ledgered(reg: dict, ledger_path) -> dict:
    """Wrap a local tool registry in the MCP server's call ledger.

    The MCP server owns the canonical wrapper. Reuse it for the local transports
    so every backend produces the same append-only provenance — a research claim
    cites `_ledger.sequence`, and a claim that cannot cite one is dropped.
    """
    if not ledger_path:
        return reg
    from .deepdive_mcp import _CallLedger, _ledger_wrapper
    ledger = _CallLedger(ledger_path)
    return {name: (_ledger_wrapper(name, fn, ledger), schema)
            for name, (fn, schema) in reg.items()}


# ---------------------------------------------------------------------------
# The ledger index — what Python already knows, told to the model
# ---------------------------------------------------------------------------
# Measured 2026-08-28: /v1/ask files structurally perfect claims that cite the
# WRONG tool call. One attempt cited sequence 13 for sleep_time_in_bed and
# sequence 2 for sleep_asleep across a 13-call conversation; 11 well-formed
# claims verified 2. The model is being asked to REMEMBER, over a dozen turns,
# which call returned which metric. Python wrote the ledger and knows exactly.
#
# So it is computed here from the ledger Python wrote, never from anything the
# model said, and derived with deepdive_verify._ledger_scopes — the very
# function _resolve_ledger_value consults to decide what is citable. A second
# scope extractor would drift from the verifier, and a vocabulary the model
# cannot read is #93 one layer up.
#
# Only `result`-rooted leaves are listed. `_ledger_scopes` also walks
# `arguments`, but _resolve_ledger_value REFUSES an $.arguments path outright,
# so showing one would be teaching a citation that is rejected by construction.

# Per-record cap. A single record can publish hundreds of leaves — a daily
# series is three leaves per point — and the index is re-sent every turn, so an
# uncapped dump is the context cost this is meant to save. 24 covers every
# scalar-summary tool's full leaf set (the miscitations measured were all on
# summary scalars, which sit at the front of a record), while a long series is
# sampled rather than pasted. Truncation is stated in the text, never silent:
# the model is told how many leaves it cannot see and that calling the tool
# again is how to get one.
LEDGER_INDEX_MAX_LEAVES = 24

_LEDGER_INDEX_HEADER = (
    "LEDGER INDEX — computed by the server from the tool-call ledger, not by "
    "you, and authoritative. It lists every value each call published and the "
    "exact path that reaches it. When you cite a number, copy its sequence and "
    "path from the line below that carries it. Do not cite a sequence from "
    "memory, and do not cite one that is not listed here."
)


def _ledger_index_value(value) -> str:
    """One published value, rendered short enough to sit on an index line."""
    try:
        text = value if isinstance(value, str) else json.dumps(value)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _ledger_index_leaf(entry: dict) -> str:
    """One citable leaf: what it is, what it says, and how to cite it."""
    parts = []
    if entry.get("metric"):
        parts.append(f"metric={entry['metric']}")
    if entry.get("period"):
        parts.append(f"period={entry['period']}")
    parts.append(f"field={entry.get('field')}")
    parts.append(f"value={_ledger_index_value(entry.get('value'))}")
    parts.append(f"path={entry.get('path')}")
    return " ".join(parts)


def _ledger_index_text(ledger_path) -> str:
    """Index every ledger record written so far. "" when there is nothing yet.

    Never raises: an unreadable or half-written ledger yields the records it
    could parse, and the loop simply appends no index rather than losing the
    conversation to a file error.
    """
    from .deepdive_verify import _ledger_scopes

    records: list[dict] = []
    try:
        with open(os.fspath(ledger_path), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return ""
    if not records:
        return ""

    lines = [_LEDGER_INDEX_HEADER]
    for record in records:
        lines.append(f"sequence {record.get('sequence')} — "
                     f"{record.get('tool_name')}")
        if record.get("result_elided"):
            lines.append("  (result was too large to record; nothing in this "
                         "call is citable — call a narrower query)")
            continue
        try:
            entries = [scope for scope in _ledger_scopes(record)
                       if scope.get("kind") == "result"]
        except Exception as exc:  # a malformed record must not kill the loop
            _announce("ledger_index_scope_error", f"{type(exc).__name__}: {exc}")
            continue
        if not entries:
            lines.append("  (published no citable value)")
            continue
        for entry in entries[:LEDGER_INDEX_MAX_LEAVES]:
            lines.append("  " + _ledger_index_leaf(entry))
        hidden = len(entries) - LEDGER_INDEX_MAX_LEAVES
        if hidden > 0:
            lines.append(f"  ... and {hidden} more citable values in this call "
                         f"are NOT listed (the index shows at most "
                         f"{LEDGER_INDEX_MAX_LEAVES} per call); call the tool "
                         "again with a narrower query to cite one of them")
    return "\n".join(lines)


def _openrouter_ready(on_log=None) -> bool:
    """D15 gate and credential check for the OpenAI-dialect tool path.

    The gate is :func:`assert_backend_approved` itself, called here rather than
    reimplemented, so this transport cannot become a way around the provider pin
    or the endpoint-host check (#76, #104, #138). The loops promise never to
    raise, so a refusal is announced instead of propagating — loudly, because a
    silent empty return is the defect #128 is about.
    """
    try:
        assert_backend_approved()
    except Exception as exc:
        _announce("openrouter_not_approved", str(exc), on_log=on_log)
        return False
    if not _openrouter_api_key():
        _announce("openrouter_no_api_key",
                  "OPENROUTER_API_KEY is unset; no request was made", on_log=on_log)
        return False
    return True


def _openrouter_post(messages: list[dict], *, tools, timeout, options=None):
    """One OpenAI-dialect chat turn. Returns (assistant message, prompt tokens).

    Transport, headers, provider pin and error handling mirror ``complete()``'s
    openrouter branch, which is the working reference for this wire. The
    explicit ``HA_OPENROUTER_REASONING`` setting controls the reasoning field;
    ``think`` is not sent on this dialect. Raises on transport error or
    non-200; the loops own the degrade-to-empty contract.
    """
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": _payload_messages(messages),
        "stream": False,
        "reasoning": _openrouter_reasoning_field(_openrouter_reasoning_mode()),
        **_openai_sampling(options),
    }
    if tools:
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    provider = _openrouter_provider()
    if provider:
        payload["provider"] = provider
    with _client(timeout) as client:
        resp = client.post(f"{OPENROUTER_URL}/chat/completions", json=payload,
                           headers={"Authorization":
                                    f"Bearer {_openrouter_api_key()}"})
        resp.raise_for_status()
        data = resp.json()
    _assert_openrouter_response_provider(data)
    choices = data.get("choices") or []
    msg = (choices[0].get("message") if choices else None) or {}
    prompt_tokens = int((data.get("usage") or {}).get("prompt_tokens") or 0)
    return msg, prompt_tokens


def _openai_assistant_turn(msg: dict, turn: int) -> tuple[dict, list[dict]]:
    """Sanitize an assistant turn for echo-back and give every call an id.

    Only role/content/tool_calls go back on the wire: a provider's extra fields
    (reasoning traces, native ids, annotations) are not part of the request
    schema and a strict validator rejects them. The id matters more than it
    looks — it is the ONLY thing pairing a result to a call, so a turn that
    calls one tool twice with different arguments is mispaired without it. A
    provider that omits the id gets a synthetic one, used consistently in both
    halves of the pair.
    """
    calls: list[dict] = []
    for index, call in enumerate(msg.get("tool_calls") or []):
        call = dict(call or {})
        call.setdefault("type", "function")
        if not call.get("id"):
            call["id"] = f"call_{turn}_{index}"
        calls.append(call)
    echo = {"role": "assistant", "content": msg.get("content") or ""}
    if calls:
        echo["tool_calls"] = calls
    return echo, calls


def _tool_result_message(call: dict, name, content: str, *, openai_dialect: bool) -> dict:
    """One tool result, in the dialect the backend reads."""
    if not openai_dialect:
        return {"role": "tool", "tool_name": name, "content": content}
    return {"role": "tool", "tool_call_id": call.get("id"),
            "name": name, "content": content}


def _prune_orphaned_tool_turns(messages: list[dict]) -> list[dict]:
    """Drop half-turns from an OpenAI-dialect window.

    Compaction keeps a fixed-size trailing slice, which can cut a turn in two.
    OpenAI requires every assistant `tool_calls` id to be answered by a tool
    message and every tool message to answer a call in the window; either half
    alone is a 400. Dropping both is the version of the Ollama loop's
    forward-trim that this dialect needs.
    """
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    kept_ids: set = set()
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = [c.get("id") for c in m["tool_calls"]]
            if not all(i in answered for i in ids):
                continue
            kept_ids.update(ids)
        elif m.get("role") == "tool" and m.get("tool_call_id") not in kept_ids:
            continue
        out.append(m)
    return out


def tool_loop(prompt: str, *, ctx, tools: list[dict], think: bool = True,
              max_turns: int = 12, timeout: int = TIMEOUT_TOOL_TURN,
              deadline: int = DEADLINE_TOOL_LOOP,
              ledger_path: str | None = None, tool_names=None,
              claim_instructions: str | None = _RESEARCH_CLAIM_INSTRUCTIONS,
              submit_tool: bool = False,
              ledger_index: bool = False,
              submit_repair: bool = False,
              submit_repair_budget: int = SUBMIT_ANSWER_REPAIR_BUDGET,
              analyst_query_fn=None,
              ) -> ResearchResponse:
    """Researcher path: let the model call the read tools in-process until it
    returns a final text answer. Returns "" on any error, deadline/turn
    exhaustion, or an empty final answer (the deep-dive drops empty findings).
    On the codex backend the whole loop is one agentic `codex exec` run against
    the health-deepdive MCP server (codex manages its own turns/context). On
    ollama and openrouter it is this in-process loop, differing only in dialect
    (#128). Every empty return is announced through :func:`_announce`, because
    an empty answer is also a legitimate result and the two used to be
    indistinguishable.

    ``claim_instructions`` is the claim-channel block appended to the prompt.
    It defaults to the research block, which is what every historical caller
    got. Pass ``None`` when the caller has ALREADY put its own claim schema in
    ``prompt``: the ask path did not, so every /v1/ask call carried both blocks
    — 929 words of two schemas that contradict each other, the research one
    permitting a source path rooted at ``$.arguments...`` that the ask verifier
    rejects outright.

    ``submit_tool`` adds :data:`SUBMIT_ANSWER_TOOL` to the tools on the wire and
    makes a ``submit_answer`` call the loop's terminal condition, so the claim
    channel arrives typed instead of being parsed back out of prose. It is
    honoured only on the in-process (ollama/openrouter) loops: the codex backend
    is one `codex exec` run with no interception point, so it ignores the flag
    and keeps parsing the final message.

    ``ledger_index`` appends, after each turn's tool results, a Python-computed
    index of what every ledger record so far actually published — see
    :func:`_ledger_index_text`. It needs ``ledger_path``; without one there is
    no ledger to index and the flag is inert. Default off, so no existing
    caller's wire payload changes.

    ``submit_repair`` makes a MALFORMED ``submit_answer`` recoverable instead of
    terminal: the call is answered with a corrective tool result naming the
    shape failure, every other call in that turn is run normally, and the loop
    continues so the model can re-emit. ``submit_repair_budget`` bounds it per
    loop; past the budget the loop announces ``submit_answer_repair_exhausted``
    and returns empty exactly as it does today. Default off, so no existing
    caller's behaviour changes."""
    claim_prompt = prompt + (claim_instructions or "")
    if BACKEND == "codex":
        answer = _research_response(_codex_exec(
            claim_prompt, reasoning="high", timeout=deadline,
            config=_deepdive_mcp_config(ctx, ledger_path=ledger_path,
                                        include=tool_names)))
        if not answer:
            _announce("tool_loop_empty_answer",
                      json.dumps(last_codex_status(), sort_keys=True))
        return answer
    openai_dialect = BACKEND == "openrouter"
    if openai_dialect and not _openrouter_ready():
        return ResearchResponse()
    if analyst_query_fn is None:
        base_registry = (_registry(ctx) if tool_names is None else
                         _registry(ctx, include=tool_names))
    else:
        base_registry = (_registry(ctx, analyst_query_fn=analyst_query_fn)
                         if tool_names is None else
                         _registry(ctx, include=tool_names,
                                   analyst_query_fn=analyst_query_fn))
    reg = _ledgered(base_registry, ledger_path)
    messages: list[dict] = [{"role": "user", "content": prompt}]
    if claim_instructions:
        messages.append({"role": "user", "content": claim_instructions})
    wire_tools = (list(tools) + [SUBMIT_ANSWER_TOOL]) if submit_tool else tools
    repairs_spent = 0
    start = time.monotonic()
    try:
        for turn in range(max_turns):
            if time.monotonic() - start > deadline:
                _announce("tool_loop_deadline",
                          f"{deadline}s elapsed with no final answer")
                return ResearchResponse()
            if openai_dialect:
                msg, _ = _openrouter_post(messages, tools=wire_tools, timeout=timeout,
                                          options=CREATIVE_OPTS)
            else:
                payload = {
                    "model": MODEL,
                    "messages": _payload_messages(messages),
                    "stream": False,
                    "think": think,
                    "tools": wire_tools,
                    "options": CREATIVE_OPTS,
                    "keep_alive": KEEP_ALIVE,
                }
                with _client(timeout) as client:
                    resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
                    resp.raise_for_status()
                    msg = (resp.json().get("message")) or {}
            if msg.get("tool_calls"):
                # The assistant turn carrying the tool_calls, then one result per
                # call — paired by id on the OpenAI wire, by name on Ollama's.
                if openai_dialect:
                    echo, calls = _openai_assistant_turn(msg, turn)
                else:
                    echo, calls = msg, msg["tool_calls"]
                # Terminal, and checked before ANY tool runs: submit_answer is
                # synthetic, is absent from `reg`, and must leave no ledger
                # entry behind — a model that calls only it still faces the ask
                # gate's empty-ledger refusal.
                repair_pair = None   # (call, corrective text) for a repair turn
                if submit_tool:
                    terminal = _submit_answer_call(calls)
                    if terminal is not None:
                        response, event, detail, repair = \
                            _submit_answer_decode(terminal)
                        if response is None and repair and submit_repair:
                            if repairs_spent >= submit_repair_budget:
                                _announce(
                                    "submit_answer_repair_exhausted",
                                    f"{repairs_spent} repair(s) spent, budget "
                                    f"{submit_repair_budget}; last failure "
                                    f"{event}: {detail}")
                                return ResearchResponse()
                            repairs_spent += 1
                            _SUBMIT_REPAIRS["attempted"] += 1
                            repair_pair = (terminal, repair)
                        else:
                            if event:
                                _announce(event, detail)
                            if response is not None and repairs_spent:
                                _SUBMIT_REPAIRS["succeeded"] += 1
                            return (response if response is not None
                                    else ResearchResponse())
                messages.append(echo)
                for call in calls:
                    if repair_pair is not None and call is repair_pair[0]:
                        # The corrective result goes STRAIGHT into `messages`:
                        # submit_answer is not in `reg`, so it never reaches
                        # `_ledgered` and the repair adds no provenance the
                        # model could later cite. On the OpenAI wire it must be
                        # keyed by tool_call_id like any other result, or the
                        # next request is a 400 for an unanswered call.
                        messages.append(_tool_result_message(
                            call, SUBMIT_ANSWER_NAME, repair_pair[1],
                            openai_dialect=openai_dialect))
                        continue
                    # Every OTHER call in a repair turn IS run: they are
                    # legitimate reads, and the answer the model writes after
                    # the repair genuinely has their results in front of it.
                    # (The terminal-success path above still returns before
                    # running anything — there the answer was already written
                    # and must not get a ledger entry behind it.)
                    fn = call.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    entry = reg.get(name)
                    if entry is None:
                        result = {"error": f"unknown tool {name!r}"}
                    else:
                        try:
                            result = entry[0](**args)
                        except Exception as e:  # surface, don't raise (matches tools)
                            result = {"error": str(e)}
                    messages.append(_tool_result_message(
                        call, name, _encode_tool_result(result, name),
                        openai_dialect=openai_dialect))
                if ledger_index and ledger_path:
                    index_text = _ledger_index_text(ledger_path)
                    if index_text:
                        messages.append({"role": "user", "content": index_text})
                continue
            answer = _research_response(_strip_think(msg.get("content") or "").strip())
            if not answer:
                _announce("tool_loop_empty_answer",
                          f"model finished at turn {turn} with no text")
            return answer
        _announce("tool_loop_turns_exhausted", f"max_turns={max_turns}")
        return ResearchResponse()
    except Exception as exc:
        _announce("tool_loop_error", f"{type(exc).__name__}: {exc}")
        return ResearchResponse()


# ---------------------------------------------------------------------------
# research_loop — context-managed autonomous researcher (Task 3: core loop)
# Task 4 wires in tool-output elision, deterministic compaction, loop detection.
# ---------------------------------------------------------------------------

NUM_CTX                = int(os.environ.get("HA_LLM_NUM_CTX", "65536"))
# Absolute compaction point (prompt tokens), NOT a fraction of num_ctx, so the
# worst-case full reprocess (cache eviction near the threshold) has a known cost:
# ~39k tokens ≈ 30 min at ~22 tok/s, which TIMEOUT_TOOL_TURN is sized to absorb.
# History is append-only between compactions, so normal turns only pay the delta.
COMPACT_TOKENS         = int(os.environ.get("HA_LLM_COMPACT_TOKENS", "39000"))
KEEP_LAST_K_TURNS      = int(os.environ.get("HA_LLM_KEEP_LAST_K", "3"))
MAX_TOOL_RESULT_CHARS  = int(os.environ.get("HA_LLM_MAX_TOOL_CHARS", "4000"))
PER_TASK_DEADLINE      = int(os.environ.get("HA_LLM_TASK_DEADLINE", "7200"))  # 2 h/task
STALL_BUDGET           = int(os.environ.get("HA_LLM_STALL_BUDGET", "3"))
TURN_ERROR_BUDGET      = int(os.environ.get("HA_LLM_TURN_ERRORS", "2"))

_FINALIZE_MSG = ("Wrap up now: for anything you confirmed with a real number that you "
                 "have NOT already recorded, call record_finding (omit numbers you could "
                 "not obtain), then reply DONE.")


def _payload_messages(messages: list[dict]) -> list[dict]:
    """Strip private (underscore) keys before sending to Ollama."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


def _elide_old_tool_results(messages: list[dict], keep_last_k: int) -> None:
    """Replace all but the last keep_last_k tool-result messages with a one-line stub
    (the big dumps are the main context hog). Idempotent via the private _elided flag.

    The stub keeps whatever identifies the result to its backend: `tool_name` on
    Ollama, `tool_call_id` on the OpenAI wire. Dropping the id there would orphan
    the assistant turn that called it and make the next request a 400 — eliding
    is a size reduction, never a re-pairing."""
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    keep = set(tool_idxs[-keep_last_k:]) if keep_last_k > 0 else set()
    for i in tool_idxs:
        if i in keep or messages[i].get("_elided"):
            continue
        original = messages[i]
        name = original.get("tool_name") or original.get("name") or "tool"
        stub = {"role": "tool",
                "content": f"[{name} result elided — call again if needed]",
                "_elided": True}
        if "tool_call_id" in original:
            stub["tool_call_id"] = original["tool_call_id"]
            stub["name"] = name
        else:
            stub["tool_name"] = name
        messages[i] = stub


def _compaction_tail(messages: list[dict], keep_last_k: int) -> list[dict]:
    """The trailing window kept verbatim on compaction, forward-trimmed so it never
    begins with an orphaned tool-result message whose assistant tool_calls parent was
    sliced off the front (some chat templates reject a leading bare tool message)."""
    tail = messages[-(2 * keep_last_k):] if keep_last_k > 0 else []
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    return tail


def _encode_tool_result(result, name: str) -> str:
    """JSON-encode one tool result, degrading to an error payload for THAT call
    rather than killing the loop.

    The tool *call* is already guarded by its callers; this is the *encode*,
    which used to sit outside that guard. A tool returning a value json cannot
    represent raised TypeError here, escaped to the loop's outer handler, and
    discarded the whole conversation — every earlier tool call with it.
    Measured 2026-08-27 on a live /v1/ask: ``TypeError: Object of type
    GradingPolicy is not JSON serializable`` produced a designed-looking
    refusal with no data behind it, indistinguishable from an ungrounded
    answer (#151).

    ``default=str`` is deliberately NOT used. It would hand the model a
    stringified object as though it were data, in a system whose one rule is
    that the model never derives a figure. An unencodable result is a defect to
    surface, not to paper over — so it is announced, and the model is told this
    one tool failed instead of being handed something that reads like a value.
    """
    try:
        return json.dumps(result)
    except (TypeError, ValueError) as exc:
        _announce("tool_result_unencodable",
                  f"{name}: {type(exc).__name__}: {exc}")
        return json.dumps(
            {"error": f"tool result is not JSON-encodable: {exc}"})


def _tool_content(result, max_chars: int, name: str = "tool") -> str:
    """JSON-encode a tool result, truncating with an explicit marker when oversized."""
    payload = _encode_tool_result(result, name)
    if len(payload) > max_chars:
        payload = payload[:max_chars] + "…[truncated]"
    return payload


def research_loop(prompt, *, ctx, extra_tools, compact_state, think=True, num_ctx=NUM_CTX,
                  max_turns=40, timeout=TIMEOUT_TOOL_TURN, deadline=PER_TASK_DEADLINE,
                  keep_last_k=KEEP_LAST_K_TURNS, compact_tokens=COMPACT_TOKENS,
                  max_tool_chars=MAX_TOOL_RESULT_CHARS, stall_budget=STALL_BUDGET,
                  on_log=None, scratch_path=None, task_id=None,
                  ledger_path: str | None = None) -> ResearchResponse:
    """Context-managed free-form research agent. Merges the static read tools with the
    run-scoped extra_tools and lets the model call them until it finalizes, bounding the
    working context via deterministic compaction and guarding against loops. Between
    compactions the message history is strictly APPEND-ONLY: rewriting old messages
    invalidates llama.cpp's context checkpoints for SWA/hybrid models and forces a full
    prompt reprocess every turn (the 2026-07-05 failure), so tool-output elision happens
    only at compaction or before a timeout retry. A failed model turn (timeout/5xx) does
    not abort the run: the loop elides old tool output, asks the model to finalize, and
    retries, giving up only after TURN_ERROR_BUDGET failures. Findings are a side effect
    of the agent's record_finding calls; returns the final assistant text (or "" on any
    error — never raises).

    On the codex backend the run collapses to one agentic `codex exec` against the
    health-deepdive MCP server: codex owns turns/compaction/loop-guarding, and the
    notepad tools live in the MCP subprocess bound to (scratch_path, task_id) — the
    in-process extra_tools/compact_state closures are unused there.

    On openrouter it is this same in-process loop in the OpenAI dialect (#128):
    tool results keyed by tool_call_id, prompt tokens read from usage, and the
    call ledger built here so provenance matches the codex path. Every empty
    return is announced (on_log, stderr, last_loop_status) — an empty answer is
    a legitimate deep-dive result, so a refusal that looks like one is a defect,
    not a quiet default."""
    claim_prompt = prompt + _RESEARCH_CLAIM_INSTRUCTIONS
    if BACKEND == "codex":
        def _log(event, detail=""):
            if on_log:
                try:
                    on_log(0, event, detail)
                except Exception:
                    pass
        _log("codex_exec", f"model={CODEX_MODEL}")
        text = _codex_exec(claim_prompt, reasoning="high", timeout=deadline,
                           config=_deepdive_mcp_config(
                               ctx, scratch_path, task_id, ledger_path=ledger_path))
        _log("codex_done" if text else "codex_error",
             json.dumps({**last_codex_status(), "answer_length": len(text)},
                        sort_keys=True))
        answer = _research_response(text)
        if not answer:
            _announce("research_loop_empty_answer", json.dumps(
                {**last_codex_status(), "answer_length": len(text)},
                sort_keys=True), on_log=on_log)
        return answer
    openai_dialect = BACKEND == "openrouter"
    if openai_dialect and not _openrouter_ready(on_log):
        return ResearchResponse()
    reg = {**_registry(ctx), **extra_tools}
    if openai_dialect:
        # Provenance. The codex path gets its ledger inside the MCP subprocess;
        # the OpenAI-dialect path has to build the same one here, under the name
        # _deepdive_mcp_config would have derived, because agent_loop reads the
        # artifact back by that name and a research claim that cannot cite a
        # _ledger.sequence is dropped by the verifier.
        #
        # Deliberately NOT applied to the Ollama path: research_loop has never
        # written a ledger there (it ignores ledger_path), and switching it on
        # would change tool results under the existing local-fallback tests.
        # That gap is real and is recorded on #128 rather than fixed in passing.
        reg = _ledgered(reg, _derived_ledger_path(scratch_path, ledger_path))
    tools = [schema for _, schema in reg.values()]
    messages: list[dict] = [{"role": "user", "content": prompt},
                            {"role": "user", "content": _RESEARCH_CLAIM_INSTRUCTIONS}]
    start = time.monotonic()
    recent: list = []
    stalls = 0
    turn_errors = 0
    finalized = False

    def _log(turn, event, detail=""):
        if on_log:
            try:
                on_log(turn, event, detail)
            except Exception:
                pass

    def _post():
        """One model turn, in this backend's dialect. (message, prompt tokens)."""
        if openai_dialect:
            # think/num_ctx/keep_alive are Ollama's; the OpenAI wire has no
            # equivalent, so they map to nothing here exactly as in complete().
            return _openrouter_post(messages, tools=tools, timeout=timeout,
                                    options=CREATIVE_OPTS)
        payload = {"model": MODEL, "messages": _payload_messages(messages),
                   "stream": False, "think": think, "tools": tools,
                   "options": {**CREATIVE_OPTS, "num_ctx": num_ctx},
                   "keep_alive": KEEP_ALIVE}
        with _client(timeout) as client:
            resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return (data.get("message") or {}), int(data.get("prompt_eval_count") or 0)

    last_content = ""
    try:
        for turn in range(max_turns):
            if not finalized and time.monotonic() - start > deadline:
                messages.append({"role": "user", "content": _FINALIZE_MSG})
                finalized = True
                _log(turn, "finalize", "deadline")
            try:
                msg, pec = _post()
            except Exception as e:
                turn_errors += 1
                _log(turn, "turn_error", type(e).__name__)
                if turn_errors > TURN_ERROR_BUDGET:
                    _announce("research_loop_turn_error_budget",
                              f"{turn_errors} failed turns; last {type(e).__name__}: {e}",
                              on_log=on_log, turn=turn)
                    return _research_response(last_content)
                # Shrink the prompt (a too-slow reprocess is the usual culprit)
                # and switch to wrap-up so gathered work still gets recorded.
                _elide_old_tool_results(messages, keep_last_k)
                if not finalized:
                    messages.append({"role": "user", "content": _FINALIZE_MSG})
                    finalized = True
                continue
            if pec > compact_tokens:
                _elide_old_tool_results(messages, keep_last_k)
                tail = _compaction_tail(messages, keep_last_k)
                if openai_dialect:
                    tail = _prune_orphaned_tool_turns(tail)
                messages[:] = [
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": _RESEARCH_CLAIM_INSTRUCTIONS +
                     "\n[context compacted] Your state so far:\n"
                     + compact_state() + "\nCall read_state() if you need detail. Continue; "
                     "record new findings with record_finding."},
                ] + tail
                _log(turn, "compact", f"pec={pec}")
            if msg.get("tool_calls"):
                if openai_dialect:
                    echo, calls = _openai_assistant_turn(msg, turn)
                    messages.append(echo)
                else:
                    calls = msg["tool_calls"]
                    messages.append(msg)
                for call in calls:
                    fn = call.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    entry = reg.get(name)
                    if entry is None:
                        result = {"error": f"unknown tool {name!r}"}
                    else:
                        try:
                            result = entry[0](**args)
                        except Exception as e:
                            result = {"error": str(e)}
                    messages.append(_tool_result_message(
                        call, name, _tool_content(result, max_tool_chars, name),
                        openai_dialect=openai_dialect))
                    _log(turn, "tool", f"{name}({args})")
                    key = (name, json.dumps(args, sort_keys=True, default=str))
                    if key in recent:
                        stalls += 1
                        messages.append({"role": "user", "content":
                            f"You already called {name} with those arguments. Use that "
                            "result or investigate something else; do not repeat it."})
                        _log(turn, "loop", name)
                    recent.append(key)
                    del recent[:-6]
                if stalls >= stall_budget and not finalized:
                    messages.append({"role": "user", "content": _FINALIZE_MSG})
                    finalized = True
                    _log(turn, "finalize", "stall")
                continue
            last_content = _strip_think(msg.get("content") or "").strip()
            if finalized or last_content:
                return _research_response(last_content)
            messages.append({"role": "user", "content":
                "Call a tool to investigate, or record your findings and stop."})
        if not finalized:
            messages.append({"role": "user", "content": _FINALIZE_MSG})
            msg, _ = _post()
            final = _research_response(_strip_think(msg.get("content") or "").strip())
            if not final:
                _announce("research_loop_turns_exhausted", f"max_turns={max_turns}",
                          on_log=on_log, turn=max_turns)
            return final
        if not last_content:
            _announce("research_loop_turns_exhausted",
                      f"max_turns={max_turns} after finalize, no text",
                      on_log=on_log, turn=max_turns)
        return _research_response(last_content)
    except Exception as exc:
        _announce("research_loop_error", f"{type(exc).__name__}: {exc}", on_log=on_log)
        return ResearchResponse()
