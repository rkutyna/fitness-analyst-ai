# Fitness Analysis AI — an Apple Health analysis engine

Turn your Apple Health export into a deduplicated SQLite vault and let a language model answer questions about it — where every number is computed in Python, not by the model.

> **Project status: early, and honest about it.** The engine works, has a large
> test suite, and is extracted from a system that has run daily against ten
> years of real data. But it was extracted from a *personal* project, it is
> `0.1.0`, and the coaching layer is not included — see
> [Status and limitations](#status-and-limitations) before you plan around it.

---

## The one rule

> **Python owns the truth. The model is only ever a text transformer.**

Every number that reaches a user is computed in Python from SQLite and handed to
the model as fact. The model narrates it. It never derives it, never re-does the
arithmetic, and never fills in a figure it did not receive.

This is not a style preference. An LLM asked "what was my average resting heart
rate in July?" will confidently produce a number, and that number will sometimes
be wrong in a way nobody can see. So the model is never asked. Python runs the
query, computes the statistic, and gives the model a fact to put into a
sentence. If a change to this project would let a model produce a figure, that
change is wrong — see [CONTRIBUTING.md](CONTRIBUTING.md).

## What it does

- **Ingests** an Apple Health `export.zip` (the full XML history) and, optionally,
  live HealthKit deltas pushed to a small local REST receiver.
- **Deduplicates and arbitrates across devices.** If you wear a watch and carry a
  phone, both record steps for the same walk. The ingest layer decides which
  source owns which interval rather than summing them, and does it by comparing
  sources to each other — never against a hardcoded device name.
- **Normalizes a canonical vocabulary.** The XML export speaks
  `HKQuantityTypeIdentifierStepCount`; the Health Auto Export app speaks
  `step_count`. Both land on one canonical metric name with one canonical unit,
  with the arithmetic done explicitly and unplaceable units rejected rather than
  relabelled.
- **Computes analysis deterministically in Python** — daily metrics, trends,
  correlations with lag semantics, sleep regularity, running form, heart-rate
  load, benchmarks, horizons.
- **Exposes an MCP surface** so any MCP-capable model can ask bounded, curated
  questions. Each tool runs a bounded query and returns a small structured
  summary. The raw firehose never enters a model's context.
- **Runs "analyst mode"** for questions the fixed tools cannot answer: the model
  writes Python, and that Python runs in an OS-level sandbox against a read-only
  copy of your vault.

## Architecture in brief

```
Apple Health export.zip ──┐
                          ├──►  normalize  ──►  SQLite vault  ──►  deterministic
HealthKit deltas ─────────┘   (one canonical    (dedupe +          analysis
(local REST receiver)          vocabulary)       arbitration)      (Python/numpy/scipy)
                                                      │                   │
                                                      │                   ▼
                                                      │            ┌─────────────┐
                                                      │            │ MCP tools   │──► your LLM
                                                      │            │ (curated,   │    narrates
                                                      │            │  bounded)   │    the facts
                                                      │            └─────────────┘
                                                      │
                                                      └──► analyst mode:
                                                           model-written Python,
                                                           run in an OS sandbox
                                                           against a read-only vault
```

Four layers, and the boundary between them is the point:

1. **Ingest** (`backfill`, `hk_parse`, `normalize`, `receiver`, `routes`) — parse,
   canonicalize units, deduplicate, arbitrate sources.
2. **Vault** (`db`, `vault`, `vault_crypto`, `lease`, `context`) — one SQLite file
   per user, with envelope encryption at rest and no ambient database path
   anywhere: every operation is handed the vault it may touch, so one process can
   serve two users without either reaching the other's data.
3. **Analysis** (`analysis`, `metrics`, `correlate`, `history`, `horizons`,
   `hr_load`, `sleep_regularity`, `running_form`, `benchmark`, `facts`, …) —
   pure functions over a connection. This is where every number comes from.
4. **Model surface** (`llm`, `agents`, `chat`, `mcp_server`, `analyst*`) — the
   only place anything talks to a model, and the only place a model's output is
   accepted, gated, and rendered.

## Quickstart

Requires Python 3.11+ on macOS or Linux, and your own Apple Health export.

```bash
git clone https://github.com/rkutyna/fitness-analysis-ai.git
cd fitness-analysis-ai
python3.11 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

An editable install (`-e .`) is convenient while developing, but a plain
`pip install .` works too — the canonical schema ships inside the package as
`health_advisor/schema.sql`, so an installed copy is self-contained.

**Try it without your own data.** A synthetic vault takes seconds and needs no
Apple Health export:

```bash
python -m health_advisor.demo --out data/demo.db --days 730
```

That writes a deterministic multi-device vault — records, workouts, sleep
sessions and check-ins across several source devices — and leaves it read-only
(`--writable` opts out). The numbers are invented; the *shapes* are real, which
is what makes it useful for exercising dedupe, cross-source arbitration and the
analysis layer. Point the MCP server or analyst mode at it exactly as you would
a real vault.

**Get your data.** On your iPhone: Health app → your profile picture → *Export
All Health Data*. You get an `export.zip`. A decade of history is a few hundred
megabytes of XML and the first backfill takes a while.

**Build the vault.**

```bash
# Parse the export into a fresh SQLite vault.
./.venv/bin/python -m health_advisor.backfill --zip export.zip --db health.db

# Derive sleep-timing and wear metrics over the whole history.
./.venv/bin/python -m health_advisor.derive --db health.db --backfill
```

`health.db` is now a normal SQLite file. You can open it with `sqlite3` and read
every derived table directly — the vault is the product, and the model layer is
optional.

**Point a model at it** by running the MCP server over stdio and registering it
with your MCP client:

```bash
./.venv/bin/python -m health_advisor.mcp_server --vault health.db
```

The server takes the vault on the command line, not from the environment: which
data a session may reach is not a decision the environment gets to make.

**Ask an open-ended question** with analyst mode (requires a configured LLM
backend, below):

```bash
./.venv/bin/python -m health_advisor.analyst \
    --vault health.db \
    --question "did a long session produce elevated resting heart rate the next day?"
```

## Analyst mode

The curated MCP tools answer the questions someone thought of in advance.
Analyst mode answers the rest.

A model is shown a live summary of your vault's schema and the canonical metric
vocabulary, and writes one turn of Python. That Python is executed in an
OS-level sandbox — macOS `sandbox-exec` (seatbelt) or Linux `bubblewrap` — with:

- a read-only connection to the vault, and no network,
- an isolated interpreter (`python -I`) with a minimal environment: none of the
  parent process's environment, secrets included, is inherited,
- a **split run directory**, where only a `work/` subdirectory is writable by the
  child, so the sandboxed code cannot rewrite its own provenance record,
- results returned **only on file descriptor 3**, never as a file the child could
  rewrite afterwards,
- CPU, wall-clock, and output-size limits, with a process-group kill on timeout.

What comes back off fd 3 is treated as untrusted bytes and validated into either
a typed result envelope (columns, units, row provenance) or a typed refusal.
Nothing in between is trusted.

Then — and this is the part that matters — **there is deliberately no second
model turn that narrates the result.** Analyst mode prints an aligned text table
rendered by its own Python from the validated envelope. Every number you see is
a cell that Python computed. The one rule holds even here, where it would have
been easy to break.

If the host has no supported sandbox (neither macOS nor Linux-with-bubblewrap),
analyst mode **fails closed**. It does not fall back to running the code
unconfined.

> ⚠️ **The sandbox is defence in depth, not a guaranteed boundary.** Enforcement
> is OS-dependent, and it is measured rather than assumed: the attack corpus is
> fully blocked on macOS 26.x arm64 but `DYLD_INSERT_LIBRARIES` injection is
> **not** blocked on GitHub's `macos-latest` runner. Run
> `pytest tests/test_analyst_sandbox.py` on your own platform before relying on
> the confinement, and see [SECURITY.md](SECURITY.md) for the measured results.
> **On Linux the confinement is not yet verified at all** — the only measurement
> so far ran under a privileged container, which may itself defeat it (issue #3).

## LLM backends

The LLM layer is pluggable and is the only module that talks to a model. Select a
backend with `HA_LLM_BACKEND`:

| Backend | What it is | Where data goes |
|---|---|---|
| `ollama` | Direct `/api/chat` against a local Ollama server | Stays on your machine |
| `openrouter` | OpenAI-compatible HTTP transport | A third party you must explicitly pin |
| `codex` | A `codex exec` subprocess using ChatGPT auth | A third party |

The OpenRouter path will not start without an explicit provider pin, with
OpenRouter's own fallback routing **off**, and it validates the endpoint host
against an allow-list by parsed hostname. There is deliberately **no default
provider in code**, so an unpinned run fails closed rather than routing your
health data to whoever happens to be cheapest. See [SECURITY.md](SECURITY.md) for
the full design — it is the part of this project we would most like you to read
before configuring anything.

Nothing is sent anywhere until you configure a backend. Ingest, the vault, and
the whole deterministic analysis layer involve no network at all.

## Status and limitations

Things this README will not pretend about:

- **The demo vault is synthetic, and its numbers mean nothing.** It exists so
  the engine is runnable and testable without anyone's real export — the data is
  invented, so treat any figure it produces as a shape, not a finding.
- **The personal training-plan and coaching layer is not included.** It was
  deliberately left out of the extraction: it parsed one person's prose plan with
  hardcoded English regexes and would not have worked for anyone else. The typed
  plan data model (`plan_model`, `precedence`) ships, but nothing populates it
  yet. There is no coach in this repo.
- **The iOS sync client is not part of this repo.** The local REST receiver that
  accepts HealthKit deltas is here; the app that pushes them is not.
- **The user profile is not fully parameterized yet.** Some physiological
  constants (heart-rate thresholds, a sex-specific training-load coefficient, a
  reference pace band) are still module-level defaults set for one person rather
  than fields on a profile object. They are correct arithmetic with the wrong
  constants for you. Treat those specific outputs as provisional until the
  profile object lands.
- **The API is not stable.** `0.1.0` means what it says.

What *is* solid: ingest, dedupe and source arbitration, the canonical metric
vocabulary, the vault and its encryption, the deterministic analysis layer, the
MCP surface, and the analyst sandbox. Those carry the test suite.

## Roadmap

1. **Conversational onboarding to build a training plan** — talk to the model to
   populate the typed `plan_model` structure, with your profile and your
   preferences for how the coach should behave. Python owns the structure; the
   model conducts the conversation. Same rule, one layer up.
2. **The coach layer** — briefings and check-ins built on that structure, once
   there is a structure that is genuinely yours rather than someone else's.

## Development

```bash
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python -m pytest
```

Tests marked `live` hit a local Ollama model and are skipped by default; run them
with `-m live`. CI runs the suite on both `ubuntu-latest` and `macos-latest`,
because the analyst sandbox has a different executor per OS and one runner would
leave half of it untested.

See [CONTRIBUTING.md](CONTRIBUTING.md) — it is short, and one of its rules is
absolute.

## License

MIT. See [LICENSE](LICENSE).
