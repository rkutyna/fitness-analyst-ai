# CLAUDE.md — working in this repo

**Not to be confused with `SECURITY.md` or `CONTRIBUTING.md`**, which are for
users and contributors. This file is for an agent working *on* the code.

## The one rule

> **Python owns the truth. The model is only ever a text transformer.**

Every number that reaches a user is computed in Python from SQLite and handed to
the model as fact. The model narrates. It never derives a figure.
`agents.grounding_check`, `deepdive_verify` and the deterministic fallback
renderers exist to enforce exactly this. A change that lets a model produce a
number is wrong, however convenient.

## Environment

```bash
python3.11 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python -m pytest
```

Always `./.venv/bin/python` — never bare `python3`.

## There is no real data here, and there must never be

This repo was extracted from a private personal project. **No health data, no
device names, no hostnames, no personal identifiers.** CI enforces it: the
`no-personal-data` job fails the build on any of them, and it scans itself.

To run anything, build a synthetic vault:

```bash
./.venv/bin/python -m health_advisor.demo --out data/demo.db --days 730
```

Deterministic given `--seed`, chmod 0444, five source devices. It is what CI,
the quickstart and the test fixtures all rely on. `tests/fixtures/gen_*.py`
regenerate the committed fixtures the same way.

## Landmines

Read `health_advisor/db.py` and `normalize.py` module docstrings before touching
ingest or aggregation. The short version:

- Dedupe identity is **metric-class-dependent** — cumulative metrics key on the
  window with no value; instantaneous ones keep the value.
- `daily_metrics.last` means "value at the latest timestamp", not the largest.
- Journal mode is `DELETE`, not WAL, and that is load-bearing.
- Sleep is attributed **by session**, to the date the session ends.
- Jog minutes are classified by **20-second cadence** inside a workout window.
  Nothing coarser than 20-second samples produces a single jog minute.
- **`_workout_arbitration` matches device names as SQL literals**
  (`LIKE '%Apple Watch'`, seven places). Devices named anything else get no
  arbitration and silently double-count — issue #1, and the reason the demo
  vault's devices are called `Demo Apple Watch` rather than `Demo Watch`.
- **The sandbox attack corpus over-reports on Linux** — it scores a successful
  write as an escape, which is true under seatbelt and false under bubblewrap,
  where the child gets a namespace-local `/tmp`. Issue #3. Do not act on a
  Linux corpus failure without checking whether the artifact exists on the host.

## The tests are the specification

1549 passing. Any change to a computation ships with a test that would have
failed before it. A red suite is a stop-the-line event — this project's whole
claim is numerical correctness.

`tests/test_analyst_sandbox.py` skips without a vault; point `HA_TEST_VAULT` at
one to run it.

## Still open, deliberately

The HR model is eight module-level literals in `metrics.py` tuned to one
athlete, and `hr_load.py`'s `BANISTER_MALE_SCALE` has no sex field. These were
left exactly as extracted — **do not "fix" a constant in passing.** They become
a user profile in one deliberate change.
