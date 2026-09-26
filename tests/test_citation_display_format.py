"""The citation ``display`` string is a compact, tappable markdown link.

Consumer issue #524: the old ``"[title; authors; year; doi:...; pmid:...]"``
form is replaced by ``"[<title> — <author short>, <year>](<url>)"`` so a
citation reads and taps like a normal inline link on a phone. Every string
below is checked against ``fact_template._citation_display`` directly
(the unit the format lives in) and, once, through the full
``build_citation_facts`` ledger path so the two stay in sync.
"""
from __future__ import annotations

from health_advisor import fact_template as ft


def test_citation_display_multi_author_with_doi_and_pmid():
    # Real corpus row (Llanos-Lagos et al. 2024, PMID 38165636).
    got = ft._citation_display(
        "Effect of Strength Training Programs in Middle- and Long-Distance "
        "Runners' Economy at Different Running Speeds: A Systematic Review "
        "with Meta-analysis.",
        "Llanos-Lagos C, Ramirez-Campillo R, Moran J, Sáez de Villarreal E.",
        2024, "10.1007/s40279-023-01978-y", "38165636")
    assert got == (
        "[Effect of Strength Training Programs in Middle- and Long-Distance "
        "Runners' Economy at Different Running Speeds: A Systematic Review "
        "with Meta-analysis — Llanos-Lagos et al., 2024]"
        "(https://doi.org/10.1007/s40279-023-01978-y)")


def test_citation_display_two_authors_non_ascii_surname():
    # Real corpus row (Trangmar & González-Alonso 2019, PMID 30671905).
    got = ft._citation_display(
        "Heat, Hydration and the Human Brain, Heart and Skeletal Muscles.",
        "Trangmar SJ, González-Alonso J.", 2019,
        "10.1007/s40279-018-1033-y", "30671905")
    assert got == (
        "[Heat, Hydration and the Human Brain, Heart and Skeletal Muscles "
        "— Trangmar and González-Alonso, 2019]"
        "(https://doi.org/10.1007/s40279-018-1033-y)")


def test_citation_display_many_authors_no_doi_no_pmid_has_no_link():
    # Real corpus row (Cozma et al. 2026), no DOI or PMID on file.
    got = ft._citation_display(
        "The Oxygen Imperative: Cardiorespiratory Fitness, Dose-Dependent "
        "Exercise Thresholds, and Longevity—A Narrative Review",
        "Cozma D, Gaita D, Crisan S, Tudoran C, Dumitrescu A, "
        "Văcărescu C.", 2026, None, None)
    assert got == (
        "[The Oxygen Imperative: Cardiorespiratory Fitness, Dose-Dependent "
        "Exercise Thresholds, and Longevity—A Narrative Review "
        "— Cozma et al., 2026]")
    assert "(" not in got  # no URL at all when neither doi nor pmid resolve


def test_citation_display_pmid_only_uses_pubmed_url():
    got = ft._citation_display("Some title.", "Smith AB.", 2021, None, "12345")
    assert got == "[Some title — Smith, 2021](https://pubmed.ncbi.nlm.nih.gov/12345/)"


def test_citation_display_single_author():
    got = ft._citation_display("Solo author paper.", "Smith AB.", 2021, None, None)
    assert got == "[Solo author paper — Smith, 2021]"


def test_citation_display_organisation_author_kept_whole():
    got = ft._citation_display(
        "Org paper.", "World Health Organization.", 2022, None, None)
    assert got == "[Org paper — World Health Organization, 2022]"


def test_citation_display_doi_with_parens_produces_parseable_url():
    got = ft._citation_display(
        "Paren DOI paper.", "Smith AB.", 2020,
        "10.1016/S0140-6736(20)30000-1", None)
    assert got == (
        "[Paren DOI paper — Smith, 2020]"
        "(https://doi.org/10.1016/S0140-6736%2820%2930000-1)")
    # The destination has no bare "(" or ")" left to close the link early.
    url = got.split("](", 1)[1].rstrip(")")
    assert "(" not in url and ")" not in url


def test_citation_display_title_with_brackets_is_escaped():
    got = ft._citation_display(
        "A [randomized] trial.", "Smith AB.", 2020, "10.1/x", None)
    assert got == r"[A \[randomized\] trial — Smith, 2020](https://doi.org/10.1/x)"


def test_citation_display_no_authors_omits_author_part():
    got = ft._citation_display("No authors title.", None, 2020, None, None)
    assert got == "[No authors title, 2020]"


def test_citation_display_no_year_omits_year():
    got = ft._citation_display("No year paper.", "Smith AB.", None, None, None)
    assert got == "[No year paper — Smith]"


def test_citation_display_no_authors_no_year():
    got = ft._citation_display("Bare title.", None, None, None, None)
    assert got == "[Bare title]"


def test_build_citation_facts_uses_the_new_display_format():
    ledger = [{
        "sequence": 1, "tool_name": "cite", "result_elided": False,
        "result": {
            "corpus_version": 1,
            "passages": [{
                "doc_id": "evidence-doc", "chunk_ix": 0,
                "span": "Fueling while running matters.",
                "title": "Running fueling evidence.",
                "authors": "Research group.", "year": 2020,
                "doi": "10.0000/fueling", "pmid": "123456",
            }],
        },
    }]
    facts = ft.build_citation_facts(ledger)
    (fact,) = facts.values()
    assert fact["display"] == (
        "[Running fueling evidence — Research group, 2020]"
        "(https://doi.org/10.0000/fueling)")
    # Only `display` changed shape; `source` stays the plain identity tuple.
    assert fact["source"] == {
        "doc_id": "evidence-doc", "chunk_ix": 0,
        "span": "Fueling while running matters.", "corpus_version": 1,
    }
