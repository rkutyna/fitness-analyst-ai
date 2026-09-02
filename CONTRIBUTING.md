# Contributing

Thanks for looking. This is a small project with one non-negotiable rule; please
read that section before opening a PR.

## The one rule

> **Python owns the truth. The model is only ever a text transformer.**

Every number that reaches a user is computed in Python from SQLite and handed to
the model as fact. The model narrates it. It never derives it.

**As a contribution constraint: a PR that lets a model produce a number will be
rejected.** Not "will be reviewed carefully" — rejected. Concretely, this means:

- Do not ask a model to compute, sum, average, compare, convert units, or count.
  Compute it in Python and pass the result in.
- Do not accept a figure out of model output and render it. Numbers reaching a
  user come from Python that this repository owns.
- Do not add a "narrate the result" model turn after a computation. Analyst mode
  deliberately has none: it renders its validated result table in Python. Keep
  it that way.
- If a model's output must be trusted at all, it passes through an existing gate
  (typed envelope validation, grounding check, refusal) — not around one.

If a feature seems to need a model to produce a figure, that is worth a design
discussion in an issue first. There is usually a way to move the computation into
Python; if there genuinely is not, we would rather not ship the feature.

## Dev setup

Python 3.11+ on macOS or Linux.

```bash
git clone https://github.com/rkutyna/fitness-analysis-ai.git
cd fitness-analysis-ai
python3.11 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Install editable (`-e`) — `schema.sql` lives at the repository root and is
resolved relative to it.

## Running the tests

```bash
./.venv/bin/python -m pytest
```

The suite builds its own temporary vaults; it needs no personal data and no
network. Tests marked `live` hit a local Ollama model and are skipped by default:

```bash
./.venv/bin/python -m pytest -m live
```

CI runs the suite on both `ubuntu-latest` and `macos-latest`. That matrix is not
decoration: the analyst sandbox has a different executor per OS (bubblewrap on
Linux, seatbelt on macOS), so a change there is only half-tested on one runner.

## Before you open a PR

- **Tests pass on both OSes**, or you say which one you could not run.
- **New behaviour has a test.** The characteristic failure mode in this codebase
  is a plausible change that is numerically wrong, and the only thing that
  catches it is running the numbers.
- **No personal data.** CI fails the build if the tree contains identifying
  strings. Never commit a real vault, a real Apple Health export, or fixtures
  derived from real health records — generate fixtures synthetically.
- **No secrets.** Keys live in the environment or a keychain, never in the tree.
- **Keep the layer boundaries.** New arithmetic belongs in the analysis layer as
  a pure function over a connection. `llm.py` stays the only module that talks to
  a model.
- **Security-relevant changes** — the provider allow-list, endpoint validation,
  the analyst sandbox, vault encryption — should say in the PR description what
  the change makes possible that was not possible before. See
  [SECURITY.md](SECURITY.md); if you have found a vulnerability, report it
  privately rather than in a PR.

Small, focused PRs get reviewed faster than large ones. If you are planning
something substantial, open an issue first so we can agree on the shape.
