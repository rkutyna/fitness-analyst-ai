#!/usr/bin/env python3
"""Acquire evidence documents from Europe PMC / PMC OA. Run by a human.

THIS PROGRAM CAN REACH THE INTERNET AND HAS NO WAY TO OPEN A HEALTH VAULT.
That is the whole point of it being a separate program from the analyst path.

Design §4.1 splits the two capabilities and enforces the split by *argument
surface* rather than by discipline: the process that can talk to the network
defines no flag that takes a vault path — no ``--vault``, no ``--health``, no
``--db``, nothing that could be one — and imports no module that could open
one. `tests/test_corpus_fetch.py` measures both over the parser's own option
strings and dests and over this file's import lines, so a future ``--corpus``
spelled with ``dest="vault"`` fails a test rather than a code review.

The counterpart is `scripts/corpus_ingest.py`, which builds the corpus from
what this program writes and never touches the network. The handoff between
them is a JSON registry plus a directory of extracted text files.

The acquisition contract is stated in `health_advisor.corpus_sources` (see issue #22 for its measurements); its policy —
hosts, licence gate, rate limit — is in `health_advisor.corpus_sources` and is
not restated here. Three of its findings shape this file directly:

* **NCBI's OA Web Service, ``oa_file_list.csv`` and the ``oa_comm/`` FTP tree
  are dead (404).** No code path here touches them and ``ftp.ncbi.nlm.nih.gov``
  is not on the allow-list.
* **The licence is not reliably in the JATS.** It is read from the S3 metadata
  ``license_code``, falling back to the Europe PMC REST ``license`` field.
  `extract_jats_text` never looks at ``<permissions>`` and must not learn to.
* **Pagination is by ``cursorMark``, never an offset**, and the search filters
  on ``OPEN_ACCESS:Y AND IN_EPMC:Y`` — different properties, and filtering on
  only one yields records whose ``fullTextXML`` 404s.

Deliberately NOT filtered on ``LICENSE:"cc by"`` in the query, though SOURCES.md
step 1 does: a query that pre-filters can never report an observed licence
distribution or a gate rejection count, and those are the numbers that tell us
whether the gate is doing anything. The gate runs here, on unfiltered results.

Every raw fetch is cached to disk under ``--cache-dir``, so a re-run costs no
requests and a rebuild is reproducible offline.

Exit codes: 0 fetched, 2 typed refusal, 1 usage error.

    ./.venv/bin/python scripts/corpus_fetch.py \\
        --plan evidence --out data/corpus --approver your-name \\
        --approved-at 2026-08-30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from html import unescape
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_advisor.corpus_sources import (  # noqa: E402
    EUROPE_PMC_BASE,
    PMC_S3_BASE,
    RATE_LIMIT_PER_SECOND,
    RateLimiter,
    SourceRefusal,
    assert_url_allowed,
    license_decision,
    ncbi_params,
)

# Same guard list as `scripts/corpus_ingest.py`. Any option string or dest
# containing one of these would hand a network-capable process a path into the
# user's health data. Duplicated rather than imported so that deleting the
# ingest script cannot silently disarm the check here.
FORBIDDEN_ARG_TOKENS: tuple[str, ...] = (
    "vault", "health", "db", "snapshot", "export", "hk", "metrics", "workout",
)

# Modules that can open health data. This program must import none of them,
# transitively or otherwise. Asserted over this file's import lines by
# `tests/test_corpus_fetch.py`.
FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "health_advisor.db", "health_advisor.vault", "health_advisor.context",
    "health_advisor.receiver", "health_advisor.normalize",
)


# --------------------------------------------------------------------------- #
# The topic plans — what to fetch, and in what proportion
# --------------------------------------------------------------------------- #
#
# Weighted from the maintainer's evidence-harvest ranking of topics (a design note; the weights below are the record)
# by how many load-bearing assertions depend on them. Rank 1 (injury risk and
# volume ramp) is both the largest cluster and the worst sourced — 12 of ~20
# assertions unsourced — so it gets the largest share. Query terms are the ones
# a coach would use, not MeSH: the retrieval test downstream asks coaching
# questions, and a corpus assembled from vocabulary the questions do not use
# measures the wrong thing.

EVIDENCE_PLAN: tuple[tuple[str, int, str], ...] = (
    ("injury-risk-volume-ramp", 12,
     '(("running injury" OR "running-related injury" OR "overuse injury") '
     'AND ("training volume" OR "load progression" OR "mileage" OR '
     '"acute chronic workload"))'),
    ("sleep-performance", 8,
     '(("sleep duration" OR "sleep quality" OR "sleep extension" OR '
     '"sleep deprivation") AND ("athletic performance" OR "exercise '
     'performance" OR "endurance performance" OR "injury"))'),
    ("training-load-progression", 7,
     '(("training load" OR "training progression" OR "periodization" OR '
     '"progressive overload") AND (running OR endurance OR runners))'),
    ("zone2-aerobic-base", 3,
     '(("aerobic base" OR "zone 2" OR "low intensity training" OR '
     '"polarized training" OR "lactate threshold") AND (endurance OR running))'),
    ("resting-heart-rate", 3,
     '(("resting heart rate" OR "resting pulse") AND ("aerobic training" OR '
     '"endurance training" OR "cardiorespiratory fitness"))'),
    ("heat-humidity", 3,
     '(("heat stress" OR "environmental heat" OR "humidity" OR '
     '"thermoregulation") AND ("endurance exercise" OR running OR "heart rate"))'),
    ("running-cadence", 3,
     '(("step rate" OR "cadence" OR "stride frequency") AND (running OR '
     '"running biomechanics" OR "tibial load"))'),
    ("hrv", 3,
     '(("heart rate variability" OR "HRV-guided training" OR "rMSSD") AND '
     '(training OR endurance OR recovery OR readiness))'),
    ("vo2max", 3,
     '(("VO2max" OR "maximal oxygen uptake" OR "cardiorespiratory fitness") '
     'AND ("endurance training" OR running OR "aerobic training"))'),
)

# The null control for the retrieval-relevance test. Real open-access papers on
# real subjects, chosen to share no vocabulary with exercise physiology — a
# retrieval system that surfaces these for a coaching question has a measurable
# defect, and synthetic filler would not test the same thing, because synthetic
# text has no incidental vocabulary to trip on.
NULL_PLAN: tuple[tuple[str, int, str], ...] = (
    ("plant-genomics", 5,
     '(("plant genome" OR "chloroplast genome" OR "Arabidopsis") AND '
     '("phylogenetic" OR "transcriptome" OR "gene family"))'),
    ("materials-science", 5,
     '(("perovskite" OR "graphene" OR "thin film" OR "alloy microstructure") '
     'AND ("crystal structure" OR "electrochemical" OR "mechanical properties"))'),
    ("marine-microbiology", 5,
     '(("marine bacteria" OR "coral reef" OR "phytoplankton" OR '
     '"deep-sea sediment") AND (microbiome OR "16S rRNA" OR biodiversity))'),
    ("astronomy-geoscience", 5,
     '(("exoplanet" OR "galaxy cluster" OR "seismic" OR "volcanic ash") AND '
     '("spectroscopy" OR "numerical model" OR observations))'),
)

PLANS = {"evidence": EVIDENCE_PLAN, "null": NULL_PLAN}


# --------------------------------------------------------------------------- #
# JATS text extraction — stdlib ElementTree only
# --------------------------------------------------------------------------- #
#
# No lxml, and no PDF branch anywhere: a PDF parser is a binary-format attack
# surface fed documents fetched from the web, and no PDF library is installed.

#: Elements dropped whole, children included. Three groups:
#:
#: * **Citation apparatus** — ``xref`` and the reference list. An inline
#:   ``<xref>`` renders as "[12]", which is noise in a chunk and actively
#:   harmful in a cited span; the reference list is other papers' titles, which
#:   is the single worst thing to have in a retrieval index because it matches
#:   every query about the topic while asserting nothing.
#: * **Non-prose figures** — tables, figures, formulas, media. Their text is
#:   numbers stripped of the layout that gave them meaning.
#: * **Metadata and back matter** — never part of the argument.
_DROP_TAGS: frozenset[str] = frozenset({
    "xref", "ref", "ref-list", "citation", "element-citation", "mixed-citation",
    "nlm-citation", "table", "table-wrap", "table-wrap-foot", "array",
    "fig", "fig-group", "graphic", "inline-graphic", "media", "alternatives",
    "disp-formula", "disp-formula-group", "inline-formula", "tex-math", "math",
    "supplementary-material", "fn", "fn-group", "author-notes", "ack",
    "app-group", "glossary", "back", "front-stub", "contrib-group", "aff",
    "funding-group", "permissions", "history", "pub-date", "article-id",
    "object-id", "label", "kwd-group", "journal-meta", "notes", "bio",
    "counts", "custom-meta-group", "conference", "product",
})

#: Elements whose content is a block: newline before and after, so paragraphs
#: and section titles do not run together into one wall of text. Chunking
#: downstream is a pure function of the text, so these boundaries are part of
#: the citation key and must be deterministic.
_BLOCK_TAGS: frozenset[str] = frozenset({
    "p", "title", "sec", "abstract", "list-item", "list", "disp-quote",
    "statement", "verse-group", "speech", "boxed-text",
})

_WS_RUN = re.compile(r"[ \t ]+")
_BLANK_RUN = re.compile(r"\n{3,}")
#: Brackets and parens left empty once their ``<xref>`` content was dropped —
#: "(  )", "[ ]", "[, ]". Cleaned so a chunk does not carry citation scars.
_EMPTY_BRACKETS = re.compile(r"[\(\[]\s*[,;:–—\-]*\s*[\)\]]")
#: Author-year citation scars. Measured on PMC13158899, whose body reads
#: ``(The Pharmacopoeia Commission..., <xref>2020</xref>; Y. Wang et al.,
#: <xref>2021</xref>)`` -- the surname is plain body prose and only the year is
#: an ``<xref>``, so dropping the xref leaves "..., ; Y. Wang et al., )". The
#: surname is legitimately part of the sentence and is kept; the orphaned
#: punctuation is not, and it would otherwise land inside a cited span.
_DANGLING_PUNCT = re.compile(r"[,;]\s*(?=[;\)\]])")


def _local(tag: object) -> str:
    """An element's local name, namespace stripped.

    JATS from Europe PMC is usually un-namespaced, but MathML and ``ali:``
    elements are not, and a tag test that ignores that silently stops dropping
    the very elements most worth dropping.
    """
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _walk(el: ET.Element, out: list[str]) -> None:
    """Append `el`'s prose to `out` in document order.

    Document order via explicit recursion rather than ``itertext()``, because
    ``itertext()`` cannot skip a subtree — it would pull every reference title
    and every table cell into the index. A dropped element still contributes
    its ``tail``: that text follows the element's close tag but belongs to the
    *parent's* flow, and losing it would splice two sentences together.
    """
    name = _local(el.tag)
    if name in _DROP_TAGS:
        if el.tail:
            out.append(el.tail)
        return
    block = name in _BLOCK_TAGS
    if block:
        out.append("\n")
    if el.text:
        out.append(el.text)
    for child in el:
        _walk(child, out)
    if block:
        out.append("\n")
    if el.tail:
        out.append(el.tail)


def _tidy(text: str) -> str:
    """Collapse the whitespace an XML tree leaves behind, deterministically."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Order matters: strip the punctuation an author-year xref left dangling
    # BEFORE collapsing empty brackets, so "(Author, ; Other, )" becomes
    # "(Author; Other)" rather than being half-cleaned into "(Author; Other,)".
    text = _DANGLING_PUNCT.sub("", text)
    text = _EMPTY_BRACKETS.sub("", text)
    text = _WS_RUN.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def extract_jats_text(xml_bytes: bytes) -> str:
    """Abstract plus body prose from a JATS article, references dropped.

    Only ``<abstract>`` (from ``<front>``) and ``<body>`` are walked. ``<back>``
    is never entered, so the reference list is excluded structurally rather
    than by a filter that could miss a variant spelling — and ``ref-list`` is
    in `_DROP_TAGS` besides, for the articles that put one inside the body.

    Graphical and teaser abstracts are skipped: they are figure captions with
    an ``abstract-type``, and their text is a caption, not an argument.
    """
    root = ET.fromstring(xml_bytes)
    parts: list[str] = []

    for child in root:
        if _local(child.tag) != "front":
            continue
        for el in child.iter():
            if _local(el.tag) != "abstract":
                continue
            if (el.get("abstract-type") or "").lower() in ("graphical", "teaser"):
                continue
            _walk(el, parts)

    for child in root:
        if _local(child.tag) == "body":
            _walk(child, parts)

    return _tidy("".join(parts))


def contains_reference_apparatus(xml_bytes: bytes, text: str) -> dict:
    """Evidence that the drop set worked, computed from the document itself.

    The measurement that matters is *survival*, not absence: "no reference text
    in the output" proves nothing about a document that carried none. So each
    probe is taken from this document's own source and looked for in the
    extracted text, and the source count is reported beside the survivor count.

    Reference entries are probed by a 60-character prefix of their rendered
    text rather than by a structured ``<article-title>``. Measured on real
    Europe PMC JATS: of the first documents fetched, references were
    ``<mixed-citation>`` free text with no ``<article-title>`` at all, so a
    structured probe reported "0 references in source" for a document with 125
    of them — a detector that cannot fail.

    ``<xref>`` renders as a bare number, and "12" occurs in prose, so the
    honest probe is the bracketed form a citation marker actually takes.
    A `naive_chars` figure is included for scale: what `itertext()` over the
    whole tree would have indexed, which is the failure this function guards.
    """
    root = ET.fromstring(xml_bytes)
    xref_texts: list[str] = []
    ref_probes: list[str] = []
    for el in root.iter():
        name = _local(el.tag)
        if name == "xref":
            s = _tidy("".join(el.itertext()))
            if s:
                xref_texts.append(s)
        elif name == "ref":
            s = _tidy(" ".join(el.itertext()))
            if len(s) >= 60:
                ref_probes.append(s[:60])

    survived_ref = sum(1 for p in ref_probes if p in text)
    # A bare "(1)" is an enumerator, not a citation. Measured on PMC13193517,
    # whose eligibility criteria read "(1) age >= 18 years; (2) registered
    # for the 2022 NYC Marathon; ..." — six digits that a paren test scores as
    # surviving xrefs and that are in fact the authors' own prose. Square
    # brackets are unambiguous; parentheses only count for a non-numeric label.
    survived_xref = sum(
        1 for t in set(xref_texts)
        if len(t) <= 12 and (f"[{t}]" in text
                             or (not t.isdigit() and f"({t})" in text)))
    return {
        "xref_in_source": len(xref_texts),
        "xref_bracketed_survived": survived_xref,
        "refs_in_source": len(ref_probes),
        "ref_prefixes_survived": survived_ref,
        "naive_itertext_chars": len(_tidy("".join(root.itertext()))),
        "extracted_chars": len(text),
    }


# --------------------------------------------------------------------------- #
# Fetching — one chokepoint, allow-listed, rate-limited, cached
# --------------------------------------------------------------------------- #

class Fetcher:
    """Every byte this program pulls from the network comes through `get`.

    One chokepoint so the allow-list, the rate limit, the NCBI identification
    parameters and the disk cache cannot be bypassed by a new call site that
    forgets one of them. A second `httpx` call anywhere in this file would be a
    defect.
    """

    def __init__(self, cache_dir: Path, *, rate: float = RATE_LIMIT_PER_SECOND,
                 timeout: float = 120.0, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(rate)
        self.offline = offline
        self.client = httpx.Client(
            timeout=timeout, follow_redirects=True,
            headers={"User-Agent": f"health-advisor-corpus (+mailto:"
                                   f"{ncbi_params('eutils.ncbi.nlm.nih.gov')['email']})"})
        self.requests = 0
        self.cache_hits = 0
        self.statuses: dict[int, int] = {}

    def close(self) -> None:
        self.client.close()

    def get(self, url: str, *, params: dict | None = None,
            cache_name: str | None = None) -> tuple[bytes, int]:
        """Fetch `url`, or serve it from disk. Returns (body, status).

        A cached body is returned with status 200; a cached 404 is recorded as
        an empty marker file so a re-run does not re-ask for something that
        does not exist. Status is returned rather than raised because a 404 on
        an S3 metadata version is an expected, informative answer, not a fault.
        """
        host = assert_url_allowed(url)
        params = ncbi_params(host, params)

        name = cache_name or hashlib.sha256(
            (url + "?" + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()[:24]
        blob = self.cache_dir / name
        miss = self.cache_dir / (name + ".404")
        if blob.exists():
            self.cache_hits += 1
            return blob.read_bytes(), 200
        if miss.exists():
            self.cache_hits += 1
            return b"", 404

        if self.offline:
            raise SourceRefusal(
                "offline_cache_miss",
                f"--offline was given and {url!r} is not in the cache")

        self.limiter.wait(host)
        resp = self.client.get(url, params=params)
        self.requests += 1
        self.statuses[resp.status_code] = self.statuses.get(resp.status_code, 0) + 1
        if resp.status_code == 200:
            blob.write_bytes(resp.content)
            return resp.content, 200
        if resp.status_code == 404:
            miss.write_bytes(b"")
            return b"", 404
        return resp.content, resp.status_code


def search(fetcher: Fetcher, query: str, *, want: int,
           page_size: int = 100) -> list[dict]:
    """Europe PMC records for `query`, paginated by ``cursorMark``.

    Never an offset: SOURCES.md records that offset paging is not supported
    here, and a script that uses one gets a truncated or repeating result set
    without any error to notice.
    """
    full = f'({query}) AND OPEN_ACCESS:Y AND IN_EPMC:Y'
    cursor, seen, out = "*", set(), []
    while len(out) < want:
        body, status = fetcher.get(EUROPE_PMC_BASE + "search", params={
            "query": full, "resultType": "core", "format": "json",
            "pageSize": page_size, "cursorMark": cursor,
        })
        if status != 200:
            raise SourceRefusal(
                "search_failed", f"Europe PMC search returned HTTP {status} "
                f"for {full!r}")
        data = json.loads(body)
        # `resultList.result[]`, NOT `result.result[]` — SOURCES.md.
        results = data.get("resultList", {}).get("result", []) or []
        for r in results:
            pmcid = (r.get("pmcid") or "").strip()
            if pmcid and pmcid not in seen:
                seen.add(pmcid)
                out.append(r)
        nxt = data.get("nextCursorMark")
        if not results or not nxt or nxt == cursor:
            break
        cursor = nxt
    return out


def s3_metadata(fetcher: Fetcher, pmcid: str) -> dict | None:
    """The PMC OA consolidated metadata JSON, trying article versions in turn.

    The bucket keys files ``PMC{id}.{version}``; the REST record does not carry
    the version, so it is discovered by asking. Versions beyond 3 are not
    tried — the cost is a request each and the yield is nil in practice.
    """
    for version in (1, 2, 3):
        url = f"{PMC_S3_BASE}metadata/{pmcid}.{version}.json"
        body, status = fetcher.get(url, cache_name=f"{pmcid}.{version}.meta.json")
        if status == 200 and body:
            try:
                meta = json.loads(body)
            except json.JSONDecodeError:
                return None
            meta["_metadata_url"] = url
            return meta
    return None


# --------------------------------------------------------------------------- #
# The acquisition loop
# --------------------------------------------------------------------------- #

def _doc_id(pmcid: str) -> str:
    return pmcid.strip().lower()


_TAG = re.compile(r"<[^>]+>")


def _clean_title(raw: object) -> str:
    """A Europe PMC title with its escaped JATS markup removed.

    The REST ``title`` field carries the publisher's inline markup HTML-escaped:
    measured on 5 of 65 fetched records, e.g. ``"…of the &lt;i&gt;CmHDZ&lt;/i&gt;
    gene family…"``. Left alone it reaches the `docs.title` column, and from
    there a citation the coach renders to a human. Unescaped once and then
    stripped of the tags that unescaping reveals — once, not in a loop, because
    repeated unescaping is how ``&amp;lt;`` becomes a tag that was never there.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    return _TAG.sub("", unescape(text)).strip()


def _year(record: dict) -> int | None:
    raw = record.get("pubYear")
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def acquire_topic(fetcher: Fetcher, *, topic: str, query: str, target: int,
                  texts_dir: Path, approver: str, approved_at: str,
                  taken: set[str], stats: dict, oversample: int = 6,
                  min_chars: int = 1500) -> list[dict]:
    """Fetch until `target` documents pass every gate, or the pool runs out.

    Oversampled because the gates reject: SOURCES.md measured ~75% CC BY in
    this domain, and ``fullTextXML`` 404s on a minority besides. The multiplier
    is generous rather than tuned — a second search page costs one request, and
    running short costs a re-run.
    """
    records = search(fetcher, query, want=target * oversample)
    stats["searched"] += len(records)
    rows: list[dict] = []

    for record in records:
        if len(rows) >= target:
            break
        pmcid = (record.get("pmcid") or "").strip()
        if not pmcid:
            continue
        doc_id = _doc_id(pmcid)
        if doc_id in taken:
            stats["skipped_duplicate"] += 1
            continue

        meta = s3_metadata(fetcher, pmcid)
        decision = license_decision(
            s3_license_code=(meta or {}).get("license_code"),
            rest_license=record.get("license"),
            is_retracted=(meta or {}).get("is_retracted"),
        )
        observed = decision.license or "(absent)"
        stats["licenses"][observed] = stats["licenses"].get(observed, 0) + 1
        if not decision.admitted:
            stats["rejected"][decision.code] = \
                stats["rejected"].get(decision.code, 0) + 1
            continue

        xml_url = f"{EUROPE_PMC_BASE}{pmcid}/fullTextXML"
        body, status = fetcher.get(xml_url, cache_name=f"{pmcid}.xml")
        if status != 200 or not body:
            stats["rejected"]["fulltext_unavailable"] = \
                stats["rejected"].get("fulltext_unavailable", 0) + 1
            continue

        try:
            text = extract_jats_text(body)
        except ET.ParseError as exc:
            stats["rejected"]["xml_parse_error"] = \
                stats["rejected"].get("xml_parse_error", 0) + 1
            stats["notes"].append(f"{pmcid}: {exc}")
            continue

        if len(text) < min_chars:
            # An extraction this short means the body was not in the XML —
            # an abstract-only record. It would occupy a doc_id and cite
            # nothing, which is the failure `empty_text` exists to prevent,
            # caught earlier and counted honestly.
            stats["rejected"]["text_too_short"] = \
                stats["rejected"].get("text_too_short", 0) + 1
            continue

        text_path = texts_dir / f"{doc_id}.txt"
        payload = text.encode("utf-8")
        # Bytes, not `write_text`: the ingest side re-reads this file and
        # re-hashes it, and a newline translation between write and read would
        # break `text_sha256` for a reason no one would find.
        text_path.write_bytes(payload)

        md5_url = ((meta or {}).get("xml_url") or "")
        publisher_md5 = md5_url.split("?md5=")[-1] if "?md5=" in md5_url else None

        rows.append({
            "doc_id": doc_id,
            "title": _clean_title(record.get("title")) or pmcid,
            "authors": _clean_title(record.get("authorString")) or None,
            "year": _year(record),
            "doi": (record.get("doi") or "").strip() or None,
            "pmid": (record.get("pmid") or "").strip() or None,
            "source_url": xml_url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds").replace("+00:00", "Z"),
            "source_sha256": hashlib.sha256(body).hexdigest(),
            "text_sha256": hashlib.sha256(payload).hexdigest(),
            "license": decision.license,
            "license_url": None,
            "redistributable": decision.redistributable,
            "approver": approver,
            "approved_at": approved_at,
            "notes": json.dumps({
                "topic": topic,
                "license_channel": decision.channel,
                "journal": ((record.get("journalInfo") or {}).get("journal")
                            or {}).get("title"),
                "source_bytes": len(body),
                "text_chars": len(text),
                "kept_fraction": round(len(text) / len(body), 4),
                "publisher_md5": publisher_md5,
            }, sort_keys=True),
            "text_path": f"texts/{doc_id}.txt",
        })
        taken.add(doc_id)

    return rows


def build_parser() -> argparse.ArgumentParser:
    """The whole argument surface of the acquisition path.

    Not one option here takes, or could take, a path into health data. That is
    the invariant `tests/test_corpus_fetch.py` measures over both the option
    strings and the dests — a `--corpus`-spelled flag with ``dest="vault"``
    would pass a check on either alone.
    """
    parser = argparse.ArgumentParser(
        prog="corpus_fetch.py",
        description="Fetch open-access evidence documents and emit a registry.")
    parser.add_argument(
        "--out", required=True,
        help="output directory for texts/, cache/ and the registry "
             "(REQUIRED; no default and no environment variable)")
    parser.add_argument(
        "--plan", choices=sorted(PLANS), default=None,
        help="a built-in topic plan: 'evidence' (weighted by "
             "EVIDENCE-HARVEST.md's ranking) or 'null' (the irrelevant control)")
    parser.add_argument(
        "--topic", action="append", default=None, dest="topics", metavar="NAME",
        help="ad-hoc topic name; pair each with one --query (repeatable)")
    parser.add_argument(
        "--query", action="append", default=None, dest="queries",
        help="ad-hoc Europe PMC query; OPEN_ACCESS:Y AND IN_EPMC:Y is added")
    parser.add_argument(
        "--target", type=int, default=None,
        help="documents wanted per topic; overrides the plan's own weighting")
    parser.add_argument(
        "--registry-name", default=None,
        help="registry filename written inside --out "
             "(default: registry-<plan>.json)")
    parser.add_argument(
        "--cache-dir", default=None,
        help="raw fetch cache (default: <out>/cache); a warm cache makes a "
             "re-run cost no requests")
    parser.add_argument(
        "--approver", required=True,
        help="who vetted these documents; written to every registry row")
    parser.add_argument(
        "--approved-at", required=True,
        help="ISO-8601 date or datetime of that approval")
    parser.add_argument(
        "--rate", type=float, default=RATE_LIMIT_PER_SECOND,
        help=f"max requests per second per host (default {RATE_LIMIT_PER_SECOND}; "
             f"NCBI publishes 3 and exceeding it risks an IP block)")
    parser.add_argument(
        "--offline", action="store_true",
        help="serve every fetch from the cache and refuse on a miss")
    parser.add_argument(
        "--report", default=None,
        help="write the acquisition statistics as JSON to this path")
    return parser


def resolve_plan(args) -> tuple[tuple[str, int, str], ...]:
    if args.plan:
        plan = PLANS[args.plan]
    elif args.topics and args.queries:
        if len(args.topics) != len(args.queries):
            raise SourceRefusal(
                "plan_mismatch",
                f"{len(args.topics)} --topic against {len(args.queries)} "
                f"--query; they pair positionally")
        plan = tuple((t, args.target or 5, q)
                     for t, q in zip(args.topics, args.queries))
    else:
        raise SourceRefusal(
            "no_plan", "give --plan, or pair --topic with --query")
    if args.target is not None:
        plan = tuple((t, args.target, q) for t, _, q in plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = resolve_plan(args)
        out_dir = Path(args.out).expanduser().resolve()
        texts_dir = out_dir / "texts"
        texts_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

        fetcher = Fetcher(cache_dir, rate=args.rate, offline=args.offline)
        stats = {
            "searched": 0, "skipped_duplicate": 0,
            "licenses": {}, "rejected": {}, "notes": [], "by_topic": {},
        }
        taken: set[str] = set()
        rows: list[dict] = []
        try:
            for topic, target, query in plan:
                got = acquire_topic(
                    fetcher, topic=topic, query=query, target=target,
                    texts_dir=texts_dir, approver=args.approver,
                    approved_at=args.approved_at, taken=taken, stats=stats)
                stats["by_topic"][topic] = len(got)
                rows.extend(got)
                print(f"  {topic:<28} {len(got):>3}/{target}", file=sys.stderr)
        finally:
            fetcher.close()

        name = args.registry_name or f"registry-{args.plan or 'adhoc'}.json"
        registry_path = out_dir / name
        registry_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        stats["documents"] = len(rows)
        stats["requests"] = fetcher.requests
        stats["cache_hits"] = fetcher.cache_hits
        stats["http_statuses"] = fetcher.statuses
        stats["registry"] = str(registry_path)
        if args.report:
            Path(args.report).write_text(
                json.dumps(stats, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        print(json.dumps({k: v for k, v in stats.items() if k != "notes"},
                         indent=2, sort_keys=True))
        return 0
    except SourceRefusal as refusal:
        print(refusal.reason, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
