# Security Policy

This project handles medical-grade personal health data, and — if you configure
an LLM backend — it can send some of that data to a third party. This document
describes how to report a problem, what the data-handling posture actually is,
and how the provider allow-list is designed to fail closed.

## Reporting a vulnerability

Please report security issues **privately**, not as a public issue.

- Preferred: GitHub's private vulnerability reporting on this repository
  (*Security* → *Report a vulnerability*), which opens a private advisory thread.
- If that is unavailable to you, open a public issue containing **no details** —
  just a request for a private channel — and we will follow up.

Please include: what you found, how to reproduce it, and what an attacker gets.
If a proof of concept involves health data, describe the shape of the data rather
than attaching any.

**Do not include real health data, vault files, API keys, or master keys in a
report.** A synthetic reproduction is always preferred.

Expect an acknowledgement within a few days. This is a small project; there is no
paid bounty and no formal SLA. Fixes land on `main` and are noted in the release
notes.

### Scope

In scope: anything that lets health data leave the machine without the user
configuring it, that lets sandboxed analyst code escape its confinement or reach
the network, that bypasses the provider or endpoint allow-list, that leaks a key
or a master key into a log or a database, or that lets one user's session reach
another user's vault.

Out of scope: vulnerabilities in pinned third-party dependencies (report those
upstream; tell us so we can bump the pin), and anything requiring an attacker who
already has read access to the machine the vault lives on. The vault is local
data at rest: a local attacker with your files and your keychain wins, and no
design here claims otherwise.

## Data-handling posture

**All data is local.** Ingest reads your Apple Health export and writes a normal
SQLite file. The analysis layer is pure Python over that file. None of ingest,
normalization, dedupe, derivation, or the deterministic analysis layer makes any
network request at all.

**The vault supports encryption at rest.** `health_advisor/vault_crypto.py`
implements streaming envelope encryption over the SQLite file: AES-256-GCM with a
freshly generated 256-bit data key per vault, wrapped by a 256-bit master key
held by a pluggable key provider. Two providers ship — an environment variable
(`HEALTH_ADVISOR_MASTER_KEY`) and the macOS Keychain. Notes on the design:

- The SQLite layer knows nothing about encryption. A caller checks a vault out by
  decrypting it to a working path, uses it, and checks it back in. **While it is
  checked out, the plaintext vault is on disk.** Put that working path somewhere
  you are comfortable with (an encrypted volume, a private directory), and be
  aware that a crash can leave it there.
- Encryption and decryption are streamed in bounded chunks; the vault is never
  materialized in memory.
- The master key is never written into the vault, and access is recorded to an
  audit log (`HEALTH_ADVISOR_VAULT_AUDIT_LOG`).
- Encryption is **opt-in**. If you never configure a key provider, your vault is
  a plain SQLite file.

**No vault path comes from the environment.** Every entry point takes the vault
on the command line and every operation receives its vault context explicitly.
There is no module-global database path, so one process can serve two users
without either being able to reach the other's data, and no misconfigured
environment variable can silently repoint a session at the wrong vault.

**Nothing is sent anywhere unless you configure an LLM backend.** There is no
telemetry, no analytics, no crash reporting, and no "phone home". If you build a
vault and query it with SQL, this software never opens a socket.

## The LLM provider allow-list

Once you do configure a backend, your health data is in a prompt going to
somebody. That is the point where this project's design is most opinionated, and
it is deliberately built to **fail closed**: a misconfiguration produces a
refusal to start, not a silent route to an unvetted third party.

The rules are enforced in `health_advisor/llm.py` — the one module that talks to
a model — at the entry point of every provider-facing process, and again on the
response.

**1. The backend must be on an allow-list.** `ollama`, `openrouter`, and `codex`.
Anything else refuses to start.

**2. The endpoint host is validated, by parsed hostname, not by substring.** The
provider pin says who the API routes your request to; it says nothing about who
receives the request in the first place, and the endpoint URL is an environment
variable. So the URL's host is parsed and compared against a per-backend set —
`openrouter.ai` for OpenRouter, loopback only for Ollama. Substring matching is
explicitly rejected as unsafe: `https://openrouter.ai.attacker.example/` contains
the string `openrouter.ai` and is a different host. Non-loopback endpoints must
be HTTPS; loopback is exempt because Ollama serves plain HTTP locally.

**3. The OpenRouter provider must be explicitly pinned, and fallbacks are off.**
OpenRouter is a router: by default it picks an inference provider for you and
will silently fall back to another one. That means the party actually receiving
your health data is chosen at request time by price and availability. So:

- `HA_OPENROUTER_PROVIDERS` must be set and non-empty.
- The request sends `allow_fallbacks: false`. If the pinned provider cannot serve
  it, the request fails rather than degrading to whoever else is cheap.
- **Every** name in the pin must be approved, not just the first one — because
  with fallbacks off the whole list is the routing order, and an unapproved name
  anywhere in it is a provider that can serve the request.
- The approved set is keyed by model, since a provider's terms and its endpoint
  capabilities are per-model facts.

**4. There is deliberately no default provider in code.** This is the load-bearing
decision. A default would mean that forgetting to pin still sends your data
somewhere — to a provider that happens to be approved, chosen by accident. There
is no default, so an unpinned run raises at startup and nothing leaves the
machine. The allow-list exists only to *refuse* a pin, never to supply one.

**5. The provider that actually served the response is checked too.** A request-
side pin is a request; the response is evidence. OpenRouter reports the serving
provider as a display name, which is translated to an endpoint tag through an
explicit, exact mapping and re-checked against the approved set. An unfamiliar
display name is not an approved provider — again, no substring or prefix
matching. A mismatch raises and is announced.

**6. Reasoning mode must be stated explicitly.** `HA_OPENROUTER_REASONING` must
be `on`, `off`, or `low`. Unset or misspelled refuses startup rather than
silently accepting the model's own default, because reasoning mode changes both
cost and what is transmitted.

A provider is added to the allow-list only against its published terms — data
retention and training policies read and recorded — and absent evidence counts as
absent, not as consent.

### What the allow-list does not do

It constrains *where* data goes, not *what* goes. If you configure an approved
provider, your health data is in that provider's hands under their terms. If you
want no third party involved at all, use the `ollama` backend against a local
model, or use the vault and the deterministic analysis layer without any model.

## Analyst mode

Analyst mode executes model-written Python. That is an intentional capability,
and it is confined at the OS level rather than by inspecting the code:

- macOS `sandbox-exec` (seatbelt) or Linux `bubblewrap`, chosen per host. **If
  neither is available, analyst mode refuses to run.** There is no unconfined
  fallback path.
- The vault connection is read-only, and the sandbox has no network access.
- The child is an isolated interpreter (`python -I`) with a minimal environment
  built from scratch — `PATH`, `TMPDIR`, and `HOME` all point inside its own
  scratch directory. **None of the parent process's environment is inherited**,
  so no API key or master key is reachable from sandboxed code.
- Only a `work/` subdirectory is writable. The generated sandbox profile, the
  code, and every provenance record live outside the child's write grant, so the
  code cannot rewrite the record of what it did or race the parent reading it.
- Results arrive only on file descriptor 3, never as a file the child could
  rewrite after the parent reads it. Every path written into the sandbox profile
  is `realpath`-resolved first, because an unresolved symlink silently denies.
- CPU, wall-clock, and output-size limits are enforced, with a process-group kill
  on timeout.
- The bytes off fd 3 are treated as untrusted and validated into either a typed
  result envelope or a typed refusal. Nothing in between is accepted.

### Enforcement is OS-dependent — verify on your own platform

**The sandbox is defence in depth, not a guaranteed boundary against a hostile
model.** Its enforcement varies by operating system and version, and the project
measures this rather than assuming it.

`tests/test_analyst_sandbox.py` runs an attack corpus of 34+ attempts across
nine classes and requires **100% blocking per class**, with any accepted
exception named explicitly in a module-level `KNOWN_GAPS`. Measured results:

| Platform | Result |
|---|---|
| macOS 26.x, arm64 | Attack corpus **fully blocked** (one documented gap: `/Users/Shared` is readable) |
| GitHub `macos-latest` runner | **`DYLD_INSERT_LIBRARIES` injection is NOT blocked** |
| Linux (bubblewrap), unprivileged host — Ubuntu 26.04, bwrap 0.11.1 | Confining correctly; corpus **over-reports** 4 classes (issue #3) |

> ### Linux results from the attack corpus are **over-reported** — see issue #3
>
> On the bubblewrap executor the corpus reports four classes "NOT BLOCKED"
> (`write_into_home`, `write_into_tmp_outside_work`, `attach_and_write`,
> `vacuum_into_exfil`). **Verified on an unprivileged Ubuntu 26.04 host: none of
> them escape.** No probe file appears on the host filesystem or in the host
> `$HOME`, and the vault is byte-unchanged with `PRAGMA integrity_check` clean.
>
> The cause is a scoring assumption, not a hole. Each probe prints `UNSAFE` when
> its own write *succeeds*, which is a correct escape test under seatbelt —
> macOS denies by path, so a successful write really did escape. Under
> bubblewrap the child gets a namespace-local `/tmp` and `$HOME`, so the write
> succeeds harmlessly and vanishes with the namespace. The corpus cannot tell
> "confined" from "escaped" on Linux.
>
> It errs toward alarm rather than false comfort, which is the right direction
> for a security test — but it makes the Linux numbers unusable as a signal
> until each probe verifies from the parent that the artifact exists on the
> host.

The macOS discrepancy is a real open question, not a test artifact: the same
corpus passes on one macOS configuration and fails on another, which means the
seatbelt profile is relying on a protection that is not universally present.

The corpus assertion is deliberately left **strict and failing** where the gap
exists, rather than absorbed into `KNOWN_GAPS` to produce a green tick — that
would delete the finding instead of fixing it. The CI job that runs it is
informational for the same reason.

**What this means for you:** run the sandbox suite on your own platform before
relying on the confinement, and do not expose analyst mode to untrusted input on
a host where an escape would matter.

Note the residual risk honestly: the sandbox confines *execution*, and the
question and the schema summary still go to whichever model backend you
configured.

## Reminders for operators

- Keep API keys and vault master keys **out of the repository**. Read them from
  the environment or a keychain.
- A decrypted vault on disk is plaintext health data. Treat the working path
  accordingly, and clean it up.
- The local REST receiver accepts HealthKit deltas. Bind it to an interface you
  control and configure its shared secret; do not expose it to the internet.
- Never attach a real vault, a real export, or real fixtures to an issue or a PR.
