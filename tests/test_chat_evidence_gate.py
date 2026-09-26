"""#394 step 2 -- the per-turn domain gate around the chat `cite` tool, and
the cross-thread fix that lets `cite` run at all.

Two things landed together here, on the decision brief accepted 2026-09-25
(consumer #394):

1. Re-landing `0b553f4` (reverted by `6e5af41` after GaAs acquired citations
   3/3 live) with the cross-thread SQLite fix
   (`.claude/i394-artifacts/i394_fix.patch` in the consumer repo -- not
   present here, applied directly to `receiver.py`). `internal_citation` now
   opens and closes its own connection inside every call; nothing is held
   across the boundary between the worker thread `chat.answer_question` runs
   on and the event-loop thread that used to close it afterward.
2. Trigger A (`analyst_corpus.evidence_gate`): `cite` is registered for a
   turn only when a corpus is configured, carries a built domain lexicon,
   and the USER'S QUESTION -- never a model-constructed retrieval query --
   shares a stem with it. Every other configured-corpus case withholds
   `cite` for that turn only and publishes a Python-owned
   ``evidence:status`` fact ("absence is not a fact").

HERMETIC ONLY: every corpus here is built fresh under ``tmp_path`` with
`corpus_build.build_corpus`, mirroring `tests/test_domain_lexicon.py`'s own
evidence/control vocabulary split (running vs. semiconductor fabrication) so
this file needs no consumer-repo fixtures and no real corpus checkout.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import httpx
import pytest

from health_advisor import analyst_corpus as ac
from health_advisor import chat, fact_template, llm, receiver
from health_advisor.corpus_build import build_corpus, sha256_text

HEADERS = {"x-health-secret": "i394-gate-secret"}

GAAS_QUESTION = "What is the boiling point of gallium arsenide?"
FUELING_QUESTION = (
    "I have been feeling flat on my longer runs lately. What does the "
    "evidence say about carbohydrate intake and glycogen for endurance "
    "running?"
)
WEIGHT_QUESTION = "How has my weight changed over the last month?"

# Long enough (well over CHUNK_CHARS=1200 with 200 overlap, repeated 40x
# exactly as tests/test_domain_lexicon.py's own fixture) that the domain
# terms land in enough chunks to clear min_chunks=3.
_EVIDENCE_TEXT = (
    "Weekly running volume and injury risk in distance runners. "
    "Carbohydrate intake and glycogen depletion during endurance running. "
    "Cadence, stride length and running economy during endurance training. "
) * 40
_CONTROL_TEXT = (
    "Gallium arsenide crystal growth and semiconductor wafer doping. "
    "Photolithography, etching and thin-film deposition in fabrication. "
    "Bandgap engineering and carrier mobility in compound semiconductors. "
) * 40


def _entry(doc_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"Title of {doc_id}",
        "authors": "Author A",
        "year": 2020,
        "doi": None,
        "pmid": None,
        "source_url": f"https://example.org/{doc_id}",
        "retrieved_at": "2026-09-25T00:00:00Z",
        "source_sha256": "0" * 64,
        "text_sha256": sha256_text(text),
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "redistributable": 1,
        "approver": "reviewer",
        "approved_at": "2026-09-25",
        "notes": None,
    }


def _corpus_with_lexicon(tmp_path: Path, *, name: str = "evidence.db") -> Path:
    """An evidence corpus WITH a built, loadable domain lexicon."""
    control_path = tmp_path / f"{name}.control.db"
    build_corpus([_entry("ctrl-1", _CONTROL_TEXT)], [_CONTROL_TEXT],
                 control_path, corpus_version=1, read_only=False)
    evidence_path = tmp_path / name
    build_corpus([_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT],
                 evidence_path, corpus_version=1, read_only=False,
                 control_corpus_paths=[control_path])
    return evidence_path


def _corpus_without_lexicon(tmp_path: Path, *, name: str = "unverified.db") -> Path:
    """An evidence corpus built with NO control corpora -- no lexicon
    stored, exactly the shape of every deployed corpus.db as of 2026-09-25
    (the gap the last comment on #394 records: ``scripts/corpus_ingest.py``
    has no control-corpus flag yet)."""
    path = tmp_path / name
    build_corpus([_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT], path,
                 corpus_version=1, read_only=False)
    return path


# --------------------------------------------------------------------------- #
# The gate itself -- pure, hermetic, no ASGI, no model.
# --------------------------------------------------------------------------- #

def test_gate_refuses_gaas_and_publishes_out_of_domain(tmp_path):
    corpus_path = _corpus_with_lexicon(tmp_path)
    available, status = ac.evidence_gate(
        GAAS_QUESTION, corpus_path, cache=ac.DomainLexiconCache())
    assert available is False
    assert status == ac.EVIDENCE_STATUS_OUT_OF_DOMAIN


def test_gate_admits_the_in_domain_fueling_question(tmp_path):
    corpus_path = _corpus_with_lexicon(tmp_path)
    available, status = ac.evidence_gate(
        FUELING_QUESTION, corpus_path, cache=ac.DomainLexiconCache())
    assert available is True
    assert status is None


def test_gate_fails_closed_with_no_built_lexicon(tmp_path):
    """Even a heavily in-domain question is refused when the corpus carries
    no lexicon to check against. Fail CLOSED: never fall back to admitting
    everything just because there is nothing to verify against."""
    corpus_path = _corpus_without_lexicon(tmp_path)
    available, status = ac.evidence_gate(
        FUELING_QUESTION, corpus_path, cache=ac.DomainLexiconCache())
    assert available is False
    assert status == ac.EVIDENCE_STATUS_LEXICON_UNVERIFIED


def test_gate_with_no_corpus_configured_publishes_nothing():
    """Unchanged from `0b553f4`: no corpus wired up, nothing to say."""
    assert ac.evidence_gate(GAAS_QUESTION, None) == (False, None)
    assert ac.evidence_gate(FUELING_QUESTION, None) == (False, None)


def test_gate_loads_the_lexicon_once_per_corpus_not_per_turn(tmp_path, monkeypatch):
    corpus_path = _corpus_with_lexicon(tmp_path)
    calls = []
    original = ac.load_domain_lexicon

    def counting(conn):
        calls.append(1)
        return original(conn)

    monkeypatch.setattr(ac, "load_domain_lexicon", counting)
    cache = ac.DomainLexiconCache()
    for question in (FUELING_QUESTION, GAAS_QUESTION, FUELING_QUESTION,
                     FUELING_QUESTION, GAAS_QUESTION):
        ac.evidence_gate(question, corpus_path, cache=cache)
    assert len(calls) == 1, "the lexicon must load once per corpus, not per turn"


def test_gate_cache_is_thread_safe_under_concurrent_first_use(tmp_path):
    """Several threads racing on a COLD cache must all get the correct
    answer. `DomainLexiconCache.get` holds its lock only around the dict
    read/write, never across the corpus I/O that fills it, so this checks
    the property that actually needs checking -- not merely "there is a
    lock somewhere"."""
    corpus_path = _corpus_with_lexicon(tmp_path)
    cache = ac.DomainLexiconCache()
    results = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        result = ac.evidence_gate(FUELING_QUESTION, corpus_path, cache=cache)
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [(True, None)] * 8


def test_gate_rebuilds_after_a_corpus_version_change(tmp_path):
    """A cache entry is keyed by (path, corpus_version): a rebuild at the
    same path must not be served the stale entry."""
    control_path = tmp_path / "control.db"
    build_corpus([_entry("ctrl-1", _CONTROL_TEXT)], [_CONTROL_TEXT],
                 control_path, corpus_version=1, read_only=False)
    path = tmp_path / "evidence.db"
    cache = ac.DomainLexiconCache()

    build_corpus([_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT], path,
                 corpus_version=1, read_only=False)
    assert ac.evidence_gate(FUELING_QUESTION, path, cache=cache) == (
        False, ac.EVIDENCE_STATUS_LEXICON_UNVERIFIED)

    build_corpus([_entry("ev-1", _EVIDENCE_TEXT)], [_EVIDENCE_TEXT], path,
                 corpus_version=2, read_only=False,
                 control_corpus_paths=[control_path])
    assert ac.evidence_gate(FUELING_QUESTION, path, cache=cache) == (True, None)


# --------------------------------------------------------------------------- #
# The published fact (`evidence:status`) -- "absence is not a fact"
# --------------------------------------------------------------------------- #

def test_build_evidence_status_fact_is_empty_for_none():
    assert fact_template.build_evidence_status_fact(None) == {}


def test_build_evidence_status_fact_publishes_the_status_string():
    facts = fact_template.build_evidence_status_fact(
        ac.EVIDENCE_STATUS_OUT_OF_DOMAIN)
    assert list(facts) == ["evidence:status"]
    entry = facts["evidence:status"]
    assert entry["display"] == "evidence: out_of_corpus_domain"
    assert entry["value"] == ac.EVIDENCE_STATUS_OUT_OF_DOMAIN


def test_evidence_status_fact_resolves_in_a_template(vault, tmp_path, monkeypatch):
    """The status reaches the CLOSED FACT SET the model is handed, exactly
    the mechanism cold_start's status/status_text leaves already use --
    proven end to end by letting a (stubbed) model actually place the
    `{evidence:status}` slot and checking it resolves."""
    # A non-empty ledger with no usable figure -- close to the real GaAs
    # shape, where the model still calls a vault tool or two before finding
    # nothing relevant. An EMPTY ledger takes a different, older refusal path
    # (`_question_is_data_request` + no tool-call ledger => fallback) that
    # has nothing to do with this fact; that path is unaffected by this
    # feature and is not what is being tested here.
    ledger: list[dict] = [{
        "sequence": 1, "tool_name": "list_available_metrics",
        "result": {"metrics": []}, "result_elided": False,
    }]
    responses = iter(["ack", "I have no source for that: {evidence:status}."])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: next(responses))

    result = chat._answer_fact_template(
        vault, GAAS_QUESTION, "gather evidence", [], str(tmp_path / "ledger.jsonl"),
        evidence_status=ac.EVIDENCE_STATUS_OUT_OF_DOMAIN)

    assert "evidence: out_of_corpus_domain" in result["text"]


def test_evidence_status_does_not_force_a_retry_on_a_data_question(
        vault, tmp_path, monkeypatch):
    """Regression guard: merging the evidence:status fact into `facts` must
    NOT make `has_gathered_data` true on its own. Before this was excluded
    (`data_facts`, chat.py), a configured-but-out-of-domain corpus made
    every otherwise-successful data answer look like "gathered data but
    interpolated zero figures" and forced a needless, and for a real
    zero-figure case IMPOSSIBLE-to-satisfy, repair retry -- exactly the
    "answer is not blocked" requirement from the decision brief.
    """
    weight_key = "metric=weight_lb|period=2026-08-25|field=last"
    weight_fact = {
        weight_key: {
            "key": weight_key, "metric": "weight_lb", "period": "2026-08-25",
            "field": "last", "value": 168.2, "unit": "lb",
            "display": "168.2 lb",
            "source": {"sequence": 1, "path": "$.result.value"},
        },
    }
    ledger = [{"sequence": 1, "tool_name": "get_latest",
              "result": {"value": 168.2}, "result_elided": False}]
    responses = iter(["ack", "Your weight was last {" + weight_key + "}."])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(fact_template, "build_fact_set",
                        lambda ledger: dict(weight_fact))
    monkeypatch.setattr(llm, "tool_loop", lambda *args, **kwargs: next(responses))

    result = chat._answer_fact_template(
        vault, WEIGHT_QUESTION, "gather data", [], str(tmp_path / "ledger.jsonl"),
        evidence_status=ac.EVIDENCE_STATUS_OUT_OF_DOMAIN)

    assert result["mode"] == "narration"
    assert result["verification"]["ok"] is True
    # Only one model turn was consumed (the "ack" for gather, one narration
    # draft) -- `responses` would raise StopIteration on a third `next()`
    # call, so reaching here at all proves no repair retry fired.


# --------------------------------------------------------------------------- #
# End to end through the receiver: the gate decides `citation_fn` per turn.
# --------------------------------------------------------------------------- #

def _capturing_answer_question(bucket):
    def answer(ctx, question, **kwargs):
        bucket.append(kwargs)
        return {"text": "ok", "mode": "narration", "tool_trace": [],
                "verification": {}}
    return answer


async def _post(app, question):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/ask", json={"question": question},
                                 headers=HEADERS)


def test_receiver_gates_citation_fn_per_turn_on_the_question(
        monkeypatch, vault, tmp_path):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "i394-gate-secret")
    corpus_path = _corpus_with_lexicon(tmp_path)
    app = receiver.create_app(vault, analyst_corpus_path=str(corpus_path))
    calls: list[dict] = []
    monkeypatch.setattr(receiver.chat, "answer_question",
                        _capturing_answer_question(calls))

    gaas_response = asyncio.run(_post(app, GAAS_QUESTION))
    fueling_response = asyncio.run(_post(app, FUELING_QUESTION))

    assert gaas_response.status_code == 200
    assert fueling_response.status_code == 200
    assert calls[0].get("citation_fn") is None
    assert calls[0].get("evidence_status") == ac.EVIDENCE_STATUS_OUT_OF_DOMAIN
    assert calls[1].get("citation_fn") is not None
    assert calls[1].get("evidence_status") is None


def test_receiver_fails_closed_with_no_built_lexicon(monkeypatch, vault, tmp_path):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "i394-gate-secret")
    corpus_path = _corpus_without_lexicon(tmp_path)
    app = receiver.create_app(vault, analyst_corpus_path=str(corpus_path))
    calls: list[dict] = []
    monkeypatch.setattr(receiver.chat, "answer_question",
                        _capturing_answer_question(calls))

    # A question dense with the evidence corpus's own vocabulary -- if the
    # gate ever admitted by default when unverified, this is exactly the
    # question that would slip through.
    response = asyncio.run(_post(app, FUELING_QUESTION))

    assert response.status_code == 200
    assert calls[0].get("citation_fn") is None
    assert calls[0].get("evidence_status") == ac.EVIDENCE_STATUS_LEXICON_UNVERIFIED


def test_receiver_with_no_corpus_configured_is_unchanged(monkeypatch, vault):
    """No corpus at all: no `citation_fn`, no `evidence_status` -- identical
    to `0b553f4`'s original behaviour, never mind the gate."""
    monkeypatch.setattr(receiver, "SHARED_SECRET", "i394-gate-secret")
    app = receiver.create_app(vault, analyst_corpus_path=None)
    calls: list[dict] = []
    monkeypatch.setattr(receiver.chat, "answer_question",
                        _capturing_answer_question(calls))

    response = asyncio.run(_post(app, GAAS_QUESTION))

    assert response.status_code == 200
    assert calls[0].get("citation_fn") is None
    assert calls[0].get("evidence_status") is None


# --------------------------------------------------------------------------- #
# Thread affinity -- the cross-thread fix `i394_fix.patch` re-lands with
# `0b553f4`. `chat.answer_question` runs under `asyncio.to_thread` (a worker
# thread); the surrounding async handler runs on the event loop thread. The
# original bug lazily created `citation_conn` on the worker thread on first
# use and closed it from the event-loop thread afterward -- a same-thread
# violation SQLite raises as `sqlite3.ProgrammingError`. The fix opens and
# closes a fresh connection INSIDE every `citation_fn` call, so nothing is
# ever held across that boundary.
# --------------------------------------------------------------------------- #

def test_cite_connection_never_crosses_the_worker_thread_that_opened_it(
        monkeypatch, vault, tmp_path):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "i394-gate-secret")
    # The gate itself is not what this test is about; force it open so a
    # single call exercises `internal_citation` regardless of the domain
    # lexicon.
    monkeypatch.setattr(ac, "evidence_gate", lambda question, path, **kw: (True, None))
    corpus_path = _corpus_with_lexicon(tmp_path)
    app = receiver.create_app(vault, analyst_corpus_path=str(corpus_path))

    def fake_answer_question(ctx, question, **kwargs):
        # Runs inside `asyncio.to_thread`'s worker thread -- exactly where
        # the model's tool-calling loop would call `citation_fn` from in the
        # real path. `citation_conn` is created lazily on FIRST use in the
        # pre-fix code, on THIS thread.
        citation_fn = kwargs.get("citation_fn")
        assert citation_fn is not None, "gate forced open; cite must be wired"
        result = citation_fn(question)
        assert not result.get("refused"), result
        assert result["passages"]
        return {"text": "ok", "mode": "narration", "tool_trace": [],
                "verification": {}}

    monkeypatch.setattr(receiver.chat, "answer_question", fake_answer_question)

    # On the pre-fix code this raises inside the ASGI app when the event-loop
    # thread's `finally: citation_conn.close()` runs after `to_thread`
    # returns -- a different thread than the one that opened it. The
    # question must actually match the hermetic evidence corpus so `cite()`
    # returns a real passage rather than an empty (but not "refused") list.
    response = asyncio.run(_post(app, "running injury risk"))

    assert response.status_code == 200
