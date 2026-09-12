"""The coach evidence surface stays bounded by the parent corpus seam."""
from __future__ import annotations

import json
from pathlib import Path

from health_advisor import analyst_corpus as ac
from health_advisor import chat, deepdive_verify as DV, fact_template, llm
from health_advisor.corpus_build import build_corpus, sha256_text


QUESTION = "When should I worry about fueling while running?"


def _corpus(path: Path) -> Path:
    text = (
        "Fueling while running matters during prolonged continuous running. "
        "Fueling guidance for running is discussed in this evidence passage. "
    ) * 40
    build_corpus([{
        "doc_id": "evidence-doc",
        "title": "Running fueling evidence",
        "authors": "Research group",
        "year": 2020,
        "doi": "10.0000/fueling",
        "pmid": "123456",
        "source_url": "https://example.org/evidence-doc",
        "retrieved_at": "2026-09-11T00:00:00Z",
        "source_sha256": "0" * 64,
        "text_sha256": sha256_text(text),
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "redistributable": 1,
        "approver": "review group",
        "approved_at": "2026-09-11",
        "notes": None,
    }], [text], path, corpus_version=1, read_only=False)
    return path


def test_cite_schema_is_present_only_with_a_configured_callable(vault):
    retrieve = lambda query, **kwargs: {"passages": []}

    configured = llm.tool_schemas(
        vault, include=llm.COACH_TOOLS, citation_fn=retrieve)
    unconfigured = llm.tool_schemas(vault, include=llm.COACH_TOOLS)

    assert "cite" in {schema["function"]["name"] for schema in configured}
    assert "cite" not in {schema["function"]["name"] for schema in unconfigured}
    cite_schema = next(schema for schema in configured
                       if schema["function"]["name"] == "cite")
    assert "published literature" in cite_schema["function"]["description"]
    assert "vault" in cite_schema["function"]["description"]


def test_cite_boundary_strips_question_syntax_and_returns_passages(
        vault, tmp_path):
    corpus = ac.open_corpus(_corpus(tmp_path / "corpus.db"))
    seen = []
    state = ac.CiteState()

    def retrieve(query, *, k=5, doc_id=None):
        seen.append(query)
        return {
            "corpus_version": 1,
            "passages": [p.as_dict() for p in ac.cite(
                corpus, query, k, state=state, doc_id=doc_id)],
        }

    fn = llm._registry(
        vault, include=llm.COACH_TOOLS, citation_fn=retrieve)["cite"][0]
    result = fn(question=QUESTION)

    assert result["passages"]
    assert state.calls == 1
    assert len(result["passages"]) == 5
    assert seen == [QUESTION[:-1]]
    assert "query_syntax" not in json.dumps(result)
    corpus.close()


def _citation_ledger():
    return [{
        "sequence": 1,
        "tool_name": "cite",
        "arguments": {"question": QUESTION},
        "result": {
            "corpus_version": 1,
            "passages": [{
                "doc_id": "evidence-doc", "chunk_ix": 0,
                "span": "Fueling while running matters during prolonged continuous running.",
                "score": -3.0, "title": "Running fueling evidence",
                "authors": "Research group", "year": 2020,
                "doi": "10.0000/fueling", "pmid": "123456",
                "license": "CC-BY-4.0",
            }],
        },
        "result_elided": False,
    }]


def test_citation_slot_is_template_compliant_and_verified(
        vault, tmp_path, monkeypatch):
    ledger = _citation_ledger()
    facts = fact_template.build_citation_facts(ledger)
    key = next(iter(facts))
    corpus_path = _corpus(tmp_path / "corpus.db")
    responses = iter(["ack", "Evidence supports this {" + key + "}."])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    result = chat._answer_fact_template(
        vault, QUESTION, "gather evidence", [], str(tmp_path / "ledger.jsonl"),
        citation_fn=lambda *args, **kwargs: {},
        citation_verify_fn=lambda prose, claims: DV.verify_citation_claims(
            prose, claims, corpus_path))

    assert result["mode"] == "narration"
    assert result["verification"]["template_compliant"] is True
    assert result["verification"]["citations_verified"] == 1
    assert "2020" in result["text"]


def test_empty_retrieval_keeps_evidence_claim_unsourced(vault, tmp_path,
                                                        monkeypatch):
    ledger = [{
        "sequence": 1, "tool_name": "cite", "arguments": {},
        "result": {"corpus_version": 1, "passages": []},
        "result_elided": False,
    }]
    responses = iter(["ack", "This remains coaching guidance."])
    monkeypatch.setattr(chat, "_read_ledger", lambda path: ledger)
    monkeypatch.setattr(llm, "tool_loop",
                        lambda *args, **kwargs: next(responses))

    result = chat._answer_fact_template(
        vault, "an unrelated evidence question", "gather evidence", [],
        str(tmp_path / "ledger.jsonl"), citation_fn=lambda *a, **k: {})

    assert result["mode"] == "narration"
    assert result["verification"].get("citations_total", 0) == 0
    assert "citation" not in result["text"].lower()
