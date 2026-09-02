"""The acquisition path's invariants, measured rather than asserted in prose.

Four properties, each of which is the reason a defect class cannot recur:

1. **No vault on the network path.** `corpus_fetch.py` is the one program that
   may reach the internet, and it must have no way to open health data. Checked
   over the parser's option strings *and* dests, and over the file's import
   lines — a `--corpus`-spelled flag with ``dest="vault"`` passes a check on
   either alone, which is why both are here.
2. **The host allow-list is exact.** A suffix match would admit
   ``www.ebi.ac.uk.attacker.example``; a substring match on the raw URL would
   admit ``https://www.ebi.ac.uk@evil.example/``. Both are tested as refusals,
   because both are the mistake a hand-written check actually makes.
3. **The licence gate allow-lists.** An unrecognized string must score 0. The
   table below includes the strings SOURCES.md observed live and the ones its
   §"Gaps to close" warns may still exist unobserved.
4. **JATS extraction drops the citation apparatus** and is deterministic —
   `chunk_ix` is half a citation key, so a boundary that moves between builds
   silently breaks every stored citation.

No test here touches the network. The fetcher's one network chokepoint is
exercised through its disk cache, which is also how a rebuild works offline.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from health_advisor import corpus_sources as cs
from health_advisor.corpus_build import extract_text, sha256_text, validate_entry

FETCH_PATH = Path(__file__).resolve().parent.parent / "scripts" / "corpus_fetch.py"


def _load_fetch():
    spec = importlib.util.spec_from_file_location("corpus_fetch", FETCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load_fetch()


# --------------------------------------------------------------------------- #
# 1. No vault on the network path
# --------------------------------------------------------------------------- #

def test_fetch_exposes_no_vault_path_argument():
    """`Done when` 2. The count that must be 0.

    Checked over dests as well as spellings: `--corpus` with ``dest="vault"``
    would hand a network-capable process a path into health data while passing
    any grep over option strings.
    """
    parser = fetch.build_parser()
    surface = []
    for action in parser._actions:
        surface.extend(s.lstrip("-").replace("-", "_")
                       for s in action.option_strings)
        if action.dest:
            surface.append(str(action.dest))
    offending = [
        name for name in surface
        for token in fetch.FORBIDDEN_ARG_TOKENS
        if token in name.lower()
    ]
    assert offending == [], offending


def test_fetch_rejects_a_vault_flag_at_the_command_line(tmp_path):
    parser = fetch.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--out", str(tmp_path), "--approver", "reviewer",
                           "--approved-at", "2026-08-30",
                           "--vault", str(tmp_path / "health.db")])


def test_fetch_source_imports_nothing_that_can_open_a_vault():
    src = FETCH_PATH.read_text(encoding="utf-8")
    code_lines = [ln for ln in src.splitlines()
                  if ln.startswith(("import ", "from ")) or " import " in ln]
    for line in code_lines:
        for banned in fetch.FORBIDDEN_IMPORTS:
            assert banned not in line, line


def test_fetch_requires_an_output_directory(tmp_path):
    """T-003, no ambient path: --out has no default and no env fallback."""
    parser = fetch.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--approver", "reviewer", "--approved-at", "2026-08-30"])


# --------------------------------------------------------------------------- #
# 2. The host allow-list
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url,code", [
    ("https://www.ebi.ac.uk.attacker.example/x", "host_not_allowed"),
    ("https://evil.example/x", "host_not_allowed"),
    ("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv", "host_not_allowed"),
    ("https://www.ebi.ac.uk@evil.example/x", "host_not_allowed"),
    ("http://www.ebi.ac.uk/x", "not_https"),
    ("ftp://www.ebi.ac.uk/x", "not_https"),
    ("/europepmc/webservices/rest/search", "no_host"),
])
def test_disallowed_urls_are_refused(url, code):
    with pytest.raises(cs.SourceRefusal) as excinfo:
        cs.assert_url_allowed(url)
    assert excinfo.value.code == code
    assert excinfo.value.reason.startswith("corpus.source.")


@pytest.mark.parametrize("url,host", [
    ("https://www.ebi.ac.uk/europepmc/webservices/rest/search", "www.ebi.ac.uk"),
    ("https://pmc-oa-opendata.s3.amazonaws.com/metadata/PMC1.1.json",
     "pmc-oa-opendata.s3.amazonaws.com"),
    ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
     "eutils.ncbi.nlm.nih.gov"),
    ("https://odphp.health.gov/sites/default/files/x.pdf", "odphp.health.gov"),
])
def test_allow_listed_urls_pass(url, host):
    assert cs.assert_url_allowed(url) == host


def test_dead_ncbi_ftp_host_is_not_on_the_allow_list():
    """SOURCES.md landmine 1: valid cert, 404s, looks healthy, yields nothing."""
    assert "ftp.ncbi.nlm.nih.gov" not in cs.ALLOWED_HOSTS


def test_ncbi_identification_is_added_by_host_not_by_call_site(monkeypatch):
    monkeypatch.setenv(cs.NCBI_EMAIL_ENV, "dev@example.org")
    ncbi = cs.ncbi_params("eutils.ncbi.nlm.nih.gov", {"db": "pmc"})
    assert ncbi["tool"] == "health-advisor-corpus"
    assert ncbi["email"] == "dev@example.org"
    assert ncbi["db"] == "pmc"
    # Europe PMC is not NCBI and neither parameter is meaningful there.
    assert "tool" not in cs.ncbi_params("www.ebi.ac.uk", {"query": "x"})


def test_ncbi_request_is_refused_when_the_contact_address_is_unset(monkeypatch):
    """No default address ships, so an NCBI call without one must not be sent."""
    monkeypatch.delenv(cs.NCBI_EMAIL_ENV, raising=False)
    with pytest.raises(cs.SourceRefusal) as exc:
        cs.ncbi_params("eutils.ncbi.nlm.nih.gov", {"db": "pmc"})
    assert exc.value.code == "ncbi_email_unset"
    assert cs.NCBI_EMAIL_ENV in exc.value.reason
    # A caller that supplies its own address is unaffected.
    assert cs.ncbi_params(
        "eutils.ncbi.nlm.nih.gov", {"email": "me@example.org"}
    )["email"] == "me@example.org"
    # Non-NCBI hosts never needed it.
    assert "email" not in cs.ncbi_params("www.ebi.ac.uk", {"query": "x"})


# --------------------------------------------------------------------------- #
# 3. The licence gate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,normalized,flag", [
    ("cc by", "CC BY", 1),                 # the form Europe PMC REST returns
    ("CC BY", "CC BY", 1),                 # the form the S3 metadata returns
    ("CC-BY", "CC BY", 1),
    ("CC BY 4.0", "CC BY", 1),             # a version is not part of identity
    ("cc-by-4.0", "CC BY", 1),
    ("CC0", "CC0", 1),
    ("cc0 1.0", "CC0", 1),
    ("CC BY-SA", "CC BY SA", 0),           # copyleft: a product call, excluded
    ("CC BY-SA 4.0", "CC BY SA", 0),
    ("CC BY-ND", "CC BY ND", 0),           # chunking is arguably a derivative
    ("CC BY-NC", "CC BY NC", 0),
    ("cc by-nc-nd", "CC BY NC ND", 0),
    ("CC BY-NC-SA 3.0 IGO", "CC BY NC SA 3 0 IGO", 0),  # ported: extra terms
    ("NO-CC CODE", "NO CC CODE", 0),       # may survive the FTP retirement
    ("all rights reserved", "ALL RIGHTS RESERVED", 0),
    ("", "", 0),
    (None, "", 0),
    (123, "", 0),
    (True, "", 0),
])
def test_license_normalization_and_gate(raw, normalized, flag):
    assert cs.normalize_license_code(raw) == normalized
    assert cs.redistributable_flag(raw) == flag


def test_separator_folding_never_collapses_a_longer_licence_onto_a_shorter():
    """The bug this normalizer is most likely to have.

    Folding "-" to " " turns "CC BY-SA" into "CC BY SA". If the version-strip
    or the folding ever reduced that to "CC BY", a copyleft document would be
    admitted as CC BY and shipped. Every CC variant is checked to normalize to
    something distinct from the two admitted forms.
    """
    for variant in ("CC BY-SA", "CC BY-ND", "CC BY-NC", "CC BY-NC-SA",
                    "CC BY-NC-ND", "CC BY-SA 4.0", "CC BY-ND 3.0"):
        assert cs.normalize_license_code(variant) not in cs.REDISTRIBUTABLE_LICENSES


def test_unrecognized_licences_score_zero_rather_than_being_missed():
    """Allow-list, not blocklist. A blocklist scores a novel string as safe."""
    for invented in ("CC BY-XYZ", "PDDL", "OGL 3.0", "publisher-specific",
                     "© 2026 Elsevier", "unknown", "  "):
        assert cs.redistributable_flag(invented) == 0


def test_retraction_is_refused_independently_of_licence():
    d = cs.license_decision(s3_license_code="CC BY", is_retracted=True)
    assert not d.admitted and d.code == "retracted" and d.redistributable == 0


def test_license_is_taken_from_s3_then_rest_and_never_from_jats():
    """SOURCES.md landmine 2: ali:license_ref was present in 2 of 5 articles."""
    s3 = cs.license_decision(s3_license_code="CC BY", rest_license="cc by-nc")
    assert s3.admitted and s3.channel == "s3_metadata.license_code"

    rest = cs.license_decision(s3_license_code=None, rest_license="cc by")
    assert rest.admitted and rest.channel == "europepmc_rest.license"

    neither = cs.license_decision(s3_license_code=None, rest_license=None)
    assert not neither.admitted and neither.code == "license_absent"
    assert neither.redistributable == 0


# --------------------------------------------------------------------------- #
# 4. Rate discipline
# --------------------------------------------------------------------------- #

def test_rate_limiter_never_leaves_less_than_the_interval_between_calls():
    """Tested on an injected clock: a rate test that really sleeps is a rate
    test nobody runs, and it would take a third of a second to prove 3/s."""
    clock = {"t": 0.0}
    slept: list[float] = []

    def now():
        return clock["t"]

    def sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    limiter = cs.RateLimiter(3.0, now=now, sleep=sleep)
    for _ in range(4):
        limiter.wait("www.ebi.ac.uk")
    assert slept == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert limiter.waits == 3


def test_rate_limiter_is_per_host():
    clock = {"t": 0.0}

    def sleep(seconds):
        clock["t"] += seconds

    limiter = cs.RateLimiter(3.0, now=lambda: clock["t"], sleep=sleep)
    limiter.wait("www.ebi.ac.uk")
    limiter.wait("pmc-oa-opendata.s3.amazonaws.com")
    assert limiter.waits == 0, "distinct hosts must not queue behind each other"


def test_rate_limiter_refuses_a_nonpositive_rate():
    with pytest.raises(cs.SourceRefusal) as excinfo:
        cs.RateLimiter(0)
    assert excinfo.value.code == "bad_rate"


def test_default_rate_matches_ncbis_published_figure():
    assert cs.RATE_LIMIT_PER_SECOND == 3.0


# --------------------------------------------------------------------------- #
# 5. JATS extraction
# --------------------------------------------------------------------------- #

JATS = b"""<?xml version="1.0"?>
<article>
  <front>
    <journal-meta><journal-title>Journal Of Things</journal-title></journal-meta>
    <article-meta>
      <article-id pub-id-type="pmid">99999999</article-id>
      <permissions><license><license-p>Open Access under CC BY-NC</license-p></license></permissions>
      <abstract><p>Runners increased volume by ten percent weekly.</p></abstract>
      <abstract abstract-type="graphical"><p>GRAPHICAL CAPTION</p></abstract>
      <kwd-group><kwd>SHOULDNOTAPPEAR</kwd></kwd-group>
    </article-meta>
  </front>
  <body>
    <sec>
      <label>1.</label>
      <title>Introduction</title>
      <p>Injury risk rises with load <xref ref-type="bibr" rid="b1">[1]</xref>.</p>
      <p>See <xref ref-type="fig" rid="f1">Figure 1</xref> for the curve.</p>
      <fig id="f1"><label>Figure 1</label><caption><p>FIGCAPTION</p></caption></fig>
      <table-wrap><table><tr><td>TABLECELL</td></tr></table></table-wrap>
    </sec>
  </body>
  <back>
    <ref-list>
      <ref id="b1"><mixed-citation>Nielsen R. Excessive progression in weekly
      running distance and risk of running related injuries. J Orthop 2014.</mixed-citation></ref>
    </ref-list>
  </back>
</article>
"""


def test_jats_extraction_keeps_the_argument_and_drops_the_apparatus():
    text = fetch.extract_jats_text(JATS)
    assert "Runners increased volume by ten percent weekly." in text
    assert "Introduction" in text
    assert "Injury risk rises with load" in text

    for dropped in ("[1]", "Figure 1", "FIGCAPTION", "TABLECELL",
                    "Nielsen", "J Orthop", "Journal Of Things",
                    "SHOULDNOTAPPEAR", "GRAPHICAL CAPTION",
                    "CC BY-NC", "99999999"):
        assert dropped not in text, dropped


def test_jats_extraction_keeps_the_prose_around_a_dropped_xref():
    """A dropped element's tail belongs to the parent's flow.

    Losing it splices two sentences together, which is the failure mode of a
    naive "skip the subtree" walk and is invisible in a character count.
    """
    text = fetch.extract_jats_text(JATS)
    assert "for the curve." in text
    assert "See" in text


def test_jats_extraction_is_deterministic():
    """`chunk_ix` is half a citation key; a boundary that moves breaks every
    stored citation with nothing raising."""
    runs = {fetch.extract_jats_text(JATS) for _ in range(5)}
    assert len(runs) == 1


def test_extraction_never_reads_the_licence_from_the_xml():
    """SOURCES.md landmine 2, as a property of the code rather than a habit."""
    src = FETCH_PATH.read_text(encoding="utf-8")
    body = src.split('"""', 2)[-1]      # skip the module docstring
    for banned in ("license_ref", "license-p", "<permissions"):
        assert banned not in body, banned


def test_reference_survival_probe_measures_survival_not_absence():
    text = fetch.extract_jats_text(JATS)
    evidence = fetch.contains_reference_apparatus(JATS, text)
    assert evidence["xref_in_source"] == 2
    assert evidence["xref_bracketed_survived"] == 0
    assert evidence["refs_in_source"] == 1
    assert evidence["ref_prefixes_survived"] == 0
    # The whole point: naive itertext() would have indexed strictly more.
    assert evidence["naive_itertext_chars"] > evidence["extracted_chars"]


# --------------------------------------------------------------------------- #
# 6. The handoff to the ingest side
# --------------------------------------------------------------------------- #

def test_extracted_text_round_trips_through_the_file_the_ingest_reads(tmp_path):
    """`text_sha256` is computed on bytes and re-checked after a `read_text`.

    A newline translation between write and read would break every row for a
    reason no one would find, so the round trip is pinned by a test.
    """
    text = fetch.extract_jats_text(JATS)
    path = tmp_path / "doc.txt"
    path.write_bytes(text.encode("utf-8"))
    assert extract_text(path) == text
    assert sha256_text(extract_text(path)) == sha256_text(text)


def test_a_synthesised_row_satisfies_validate_entry(tmp_path):
    """The registry this program emits is the registry corpus_ingest consumes."""
    text = fetch.extract_jats_text(JATS)
    path = tmp_path / "pmc1.txt"
    path.write_bytes(text.encode("utf-8"))
    entry = {
        "doc_id": "pmc1", "title": "T", "authors": "A", "year": 2026,
        "doi": None, "pmid": "99999999",
        "source_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML",
        "retrieved_at": "2026-08-30T00:00:00Z",
        "source_sha256": "0" * 64,
        "text_sha256": sha256_text(text),
        "license": "CC BY", "license_url": None, "redistributable": 1,
        "approver": "reviewer", "approved_at": "2026-08-30",
        "notes": json.dumps({"topic": "t"}),
    }
    validate_entry(entry, extract_text(path), seen_doc_ids=set())


@pytest.mark.parametrize("raw,clean", [
    ("Genome-wide identification of the &lt;i&gt;CmHDZ&lt;/i&gt; gene family",
     "Genome-wide identification of the CmHDZ gene family"),
    ("Effects of VO&lt;sub&gt;2&lt;/sub&gt;max on runners",
     "Effects of VO2max on runners"),
    ("Sleep &amp; performance", "Sleep & performance"),
    ("  A plain title  ", "A plain title"),
    (None, ""),
])
def test_titles_are_stripped_of_escaped_jats_markup(raw, clean):
    """Measured on 5 of 65 fetched records: the REST ``title`` field carries
    the publisher's inline markup HTML-escaped, and it would otherwise reach
    `docs.title` and from there a citation shown to a human."""
    assert fetch._clean_title(raw) == clean


def test_title_cleaning_does_not_unescape_twice():
    """``&amp;lt;i&amp;gt;`` is a literal "&lt;i&gt;", not a tag. Unescaping in
    a loop would invent markup that was never in the document and delete it."""
    assert fetch._clean_title("Ratio of &amp;lt;i&amp;gt; to x") == \
        "Ratio of &lt;i&gt; to x"


def test_the_built_in_plans_cover_the_harvests_top_ranked_topics():
    """EVIDENCE-HARVEST.md ranks injury risk first by a wide margin — largest
    cluster and worst sourced — so it must carry the largest share."""
    plan = dict((name, target) for name, target, _ in fetch.EVIDENCE_PLAN)
    assert plan["injury-risk-volume-ramp"] == max(plan.values())
    top_three = ("injury-risk-volume-ramp", "sleep-performance",
                 "training-load-progression")
    assert sum(plan[t] for t in top_three) > sum(plan.values()) / 2
    assert sum(plan.values()) >= 40


def test_the_null_plan_shares_no_topic_with_the_evidence_plan():
    evidence = {name for name, _, _ in fetch.EVIDENCE_PLAN}
    null = {name for name, _, _ in fetch.NULL_PLAN}
    assert evidence & null == set()
    assert sum(t for _, t, _ in fetch.NULL_PLAN) >= 20


def test_search_filters_on_both_open_access_and_in_epmc():
    """SOURCES.md landmine 4: they are different properties, and filtering on
    only one yields records whose fullTextXML 404s."""
    src = FETCH_PATH.read_text(encoding="utf-8")
    assert "OPEN_ACCESS:Y AND IN_EPMC:Y" in src


def test_pagination_uses_cursormark_and_never_an_offset():
    """SOURCES.md landmine 5: an offset silently truncates or repeats."""
    src = FETCH_PATH.read_text(encoding="utf-8")
    assert "cursorMark" in src
    body = src.split('"""', 2)[-1]
    for banned in ('"offset"', "'offset'", '"page":', "&offset="):
        assert banned not in body, banned


# --------------------------------------------------------------------------- #
# 7. The fetch chokepoint
# --------------------------------------------------------------------------- #

def test_fetcher_refuses_a_disallowed_host_before_any_request(tmp_path):
    fetcher = fetch.Fetcher(tmp_path / "cache", offline=True)
    try:
        with pytest.raises(cs.SourceRefusal) as excinfo:
            fetcher.get("https://evil.example/x")
        assert excinfo.value.code == "host_not_allowed"
        assert fetcher.requests == 0
    finally:
        fetcher.close()


def test_offline_mode_refuses_on_a_cache_miss_rather_than_fetching(tmp_path):
    fetcher = fetch.Fetcher(tmp_path / "cache", offline=True)
    try:
        with pytest.raises(cs.SourceRefusal) as excinfo:
            fetcher.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search")
        assert excinfo.value.code == "offline_cache_miss"
        assert fetcher.requests == 0
    finally:
        fetcher.close()


def test_a_warm_cache_serves_without_a_request(tmp_path):
    cache = tmp_path / "cache"
    fetcher = fetch.Fetcher(cache, offline=True)
    try:
        (cache / "PMC1.xml").write_bytes(JATS)
        body, status = fetcher.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML",
            cache_name="PMC1.xml")
        assert status == 200 and body == JATS
        assert fetcher.requests == 0 and fetcher.cache_hits == 1
    finally:
        fetcher.close()
