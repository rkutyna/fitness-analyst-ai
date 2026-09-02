"""Acquisition policy for the evidence corpus: hosts, licences, rate limits.

This module is the machine-readable half of
the corpus acquisition contract (issue #22 in this repository carries its measurements),
verified live; this file is the part of it that a script cannot forget.

Three properties, each enforced here rather than by discipline:

1. **The host allow-list.** A fetch to any host not named in `ALLOWED_HOSTS`
   raises `SourceRefusal`. The comparison is on the *parsed* hostname and is
   exact, never a suffix test: ``www.ebi.ac.uk.attacker.example`` ends with an
   approved host and is refused. Plain ``http://`` is refused for the same
   reason a TLS check exists at all — an allow-listed host reached over a
   channel anyone can rewrite is not the allow-listed host. This mirrors
   `health_advisor.llm.assert_endpoint_approved`, which does the same job for
   the model endpoint.

   ``ftp.ncbi.nlm.nih.gov`` is deliberately absent. SOURCES.md records that it
   resolves with a valid certificate and returns 404 for the entire retired
   ``oa_comm``/``oa_file_list.csv`` layout, so a script pointed at it fails by
   returning nothing — which reads as "no results", not as an error. A dead
   host on an allow-list is worse than no allow-list.

2. **The licence gate, allow-listed.** ``redistributable = 1`` iff the
   normalized licence code is in `REDISTRIBUTABLE_LICENSES` = {CC0, CC BY}.
   Everything else is 0: NC variants, ND, SA, absent, unparseable, and
   anything simply unrecognized. SOURCES.md §"Gaps to close" asks for exactly
   this shape — *"handle defensively by allow-listing known-good strings
   rather than blocklisting bad ones"* — because a blocklist scores an
   unrecognized string as safe, and the retired FTP layout means nobody knows
   what strings still exist. A document whose licence cannot be positively
   identified is 0, never a guess.

   ``is_retracted: true`` is refused outright, whatever the licence says.

3. **Rate discipline.** NCBI publishes *"no more than three URL requests per
   second"* without an API key, and asks for a ``tool=`` and ``email=`` on
   every call. Exceeding it risks an IP block against the whole machine, which
   is a cost paid by the user and not by this process. Europe PMC allows more
   (semi-verified at 10/s per IP) and is held to 3/s anyway: the headroom is
   real but the politeness is free.

Nothing here opens a database, and nothing here performs I/O — `RateLimiter`
sleeps and that is the extent of its effects. The fetcher composes these
pieces; the policy lives here so it can be tested without a network.
"""
from __future__ import annotations

import os
import re
import time
import urllib.parse
from typing import Mapping

__all__ = [
    "ALLOWED_HOSTS",
    "NCBI_HOSTS",
    "NCBI_TOOL",
    "NCBI_EMAIL_ENV",
    "ncbi_email",
    "REDISTRIBUTABLE_LICENSES",
    "RATE_LIMIT_PER_SECOND",
    "EUROPE_PMC_BASE",
    "PMC_S3_BASE",
    "LicenseDecision",
    "RateLimiter",
    "SourceRefusal",
    "assert_url_allowed",
    "license_decision",
    "ncbi_params",
    "normalize_license_code",
    "redistributable_flag",
]


# --------------------------------------------------------------------------- #
# Hosts
# --------------------------------------------------------------------------- #

#: Every host this corpus may fetch from. Exact hostnames, lowercase, no
#: wildcards and no suffix matching. Adding one means adding its published
#: terms to SOURCES.md first — the list is the answer to "what does this
#: program talk to", and an entry without a terms citation makes that answer
#: worthless.
ALLOWED_HOSTS: frozenset[str] = frozenset({
    "www.ebi.ac.uk",                    # Europe PMC REST (search, fullTextXML)
    "pmc-oa-opendata.s3.amazonaws.com",  # PMC OA bucket (metadata, XML, TXT)
    "eutils.ncbi.nlm.nih.gov",          # E-utilities (efetch/esummary)
    "odphp.health.gov",                 # Physical Activity Guidelines
})

#: The subset of `ALLOWED_HOSTS` governed by NCBI's published rate rule and
#: its ``tool=``/``email=`` request. The S3 bucket is NCBI-operated data but is
#: served by S3 and documented as having no rate limit; it is held to the same
#: 3/s anyway by the caller, which costs nothing.
NCBI_HOSTS: frozenset[str] = frozenset({
    "eutils.ncbi.nlm.nih.gov",
})

#: NCBI asks for a no-spaces software identifier and the *developer's* address,
#: explicitly not an end user's. The tool identifier is a property of this
#: software, so it ships as a constant. The address is not: it identifies
#: whoever is *operating* this deployment, it is handed to a third party on
#: every request, and a shared default would attribute one operator's traffic
#: to another. So it has NO fallback — it is read from the environment at call
#: time, and an NCBI request without it is refused rather than sent.
NCBI_TOOL = "health-advisor-corpus"
NCBI_EMAIL_ENV = "HA_NCBI_EMAIL"

EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
PMC_S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com/"

#: Our ceiling, applied to every host. NCBI's published figure.
RATE_LIMIT_PER_SECOND = 3.0


class SourceRefusal(Exception):
    """A typed refusal from the acquisition policy.

    Carries a stable machine-readable ``code`` and a one-line ``reason``. It is
    an exception rather than a falsy return so a fetch cannot proceed past one
    by ignoring a value, and callers at an entry point print ``.reason`` and
    exit rather than raising a traceback — same convention as
    `health_advisor.corpus_build.RegistryRefusal`.
    """

    def __init__(self, code: str, detail: str):
        self.code = code
        self.reason = f"corpus.source.{code}: {detail}"
        super().__init__(self.reason)


def assert_url_allowed(url: str) -> str:
    """Return `url`'s hostname, or raise `SourceRefusal`.

    The check is on `urllib.parse`'s parsed hostname, which strips userinfo,
    port and brackets. That matters: ``https://www.ebi.ac.uk@evil.example/x``
    parses to host ``evil.example``, and a naive substring test on the raw URL
    would admit it.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as exc:                          # malformed IPv6, etc.
        raise SourceRefusal("unparseable_url", f"cannot parse {url!r}: {exc}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise SourceRefusal(
            "no_host", f"{url!r} has no host; only absolute https URLs to an "
            f"allow-listed host may be fetched")

    scheme = parsed.scheme.lower()
    if scheme != "https":
        raise SourceRefusal(
            "not_https",
            f"{url!r} uses scheme {scheme or '(none)'!r}, not https; an "
            f"allow-listed host reached over a rewritable channel is not the "
            f"allow-listed host")

    if host not in ALLOWED_HOSTS:
        raise SourceRefusal(
            "host_not_allowed",
            f"host {host!r} is not on the corpus allow-list "
            f"({', '.join(sorted(ALLOWED_HOSTS))}); the match is exact, so a "
            f"subdomain or a look-alike suffix does not qualify. Add the host "
            f"to SOURCES.md with its published terms first")
    return host


def ncbi_email() -> str:
    """The operator's contact address for NCBI, from ``$HA_NCBI_EMAIL``.

    Raises ``SourceRefusal`` when it is unset or blank. Read at call time, not
    at import, so setting the variable takes effect without re-importing.
    """
    value = (os.environ.get(NCBI_EMAIL_ENV) or "").strip()
    if not value:
        raise SourceRefusal(
            "ncbi_email_unset",
            f"NCBI requires a real developer contact address on every request "
            f"and this build ships no default. Set {NCBI_EMAIL_ENV} to an "
            f"address you monitor (export {NCBI_EMAIL_ENV}=you@example.org) "
            f"before fetching from {', '.join(sorted(NCBI_HOSTS))}")
    return value


def ncbi_params(host: str, params: Mapping[str, object] | None = None) -> dict:
    """`params` plus ``tool=``/``email=`` when `host` is an NCBI host.

    Applied by host rather than by call site so a new NCBI endpoint cannot be
    added without the identification NCBI asks for.
    """
    out = dict(params or {})
    if host.lower() in NCBI_HOSTS:
        out.setdefault("tool", NCBI_TOOL)
        if "email" not in out:
            # Only resolved when the caller has not supplied one, so a caller
            # that passes its own address never trips the unset refusal.
            out["email"] = ncbi_email()
    return out


# --------------------------------------------------------------------------- #
# Licence gate
# --------------------------------------------------------------------------- #

#: The only licences whose text this corpus may redistribute. Decided in
#: SOURCES.md §"Redistributability":
#:
#: * ``CC BY-ND`` is excluded despite sitting in NCBI's commercial tier —
#:   chunking text into FTS5 rows is arguably a derivative, and ND forbids
#:   derivatives. It cost one document in 500 sampled.
#: * ``CC BY-SA`` is excluded by default because it is copyleft and shipping SA
#:   text may oblige the derived database to carry compatible terms. That is a
#:   product decision for whoever ships the corpus, not one this module may
#:   take.
#:
#: Adding either is a one-line change *here* — which is the point of the set
#: being a named constant rather than a condition.
REDISTRIBUTABLE_LICENSES: frozenset[str] = frozenset({"CC0", "CC BY"})

# A trailing licence *version* is not part of the identity: "CC BY 4.0" and
# "cc-by-3.0" are both CC BY. Jurisdiction and port suffixes (IGO, deed, the
# 2.5 AU ports) are NOT stripped — they add terms a bare CC licence does not
# have, so a string carrying one is simply unrecognized and scores 0. That is
# why the version is stripped *token-wise* after separator folding rather than
# by a regex over the raw string: "4.0" has already become two tokens by then,
# and a tail pattern written against the raw form leaves "CC BY 4" behind.
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def _strip_version_tokens(tokens: list[str]) -> list[str]:
    """Drop trailing all-digit tokens — the folded remains of "4.0", "1.0".

    Only *trailing* ones, and only all-digit ones: "CC0" survives because it is
    not all digits, and "CC BY NC SA 3 0 IGO" survives intact because its last
    token is "IGO", so a ported licence can never collapse onto a bare one.
    """
    end = len(tokens)
    while end > 0 and tokens[end - 1].isdigit():
        end -= 1
    return tokens[:end]


def normalize_license_code(raw: object) -> str:
    """A licence string reduced to its comparable form, or ``""``.

    ``"cc by"``, ``"CC-BY"``, ``"CC BY 4.0"`` and ``"  cc_by  "`` all become
    ``"CC BY"``. ``"CC BY-SA"`` becomes ``"CC BY SA"``, which is *not* in the
    allow-list — separator folding must never collapse a longer licence onto a
    shorter one, and this is the case that would.

    Returns ``""`` for None, a non-string, or anything that normalizes empty.
    """
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        return ""
    tokens = _NON_ALNUM.sub(" ", raw.upper()).split()
    return " ".join(_strip_version_tokens(tokens))


def redistributable_flag(raw: object) -> int:
    """1 iff `raw` normalizes into `REDISTRIBUTABLE_LICENSES`, else 0.

    Allow-list, never blocklist. An unrecognized string, an absent value and a
    known-bad value are the same answer, because the corpus cannot tell them
    apart and must not pretend to.
    """
    return 1 if normalize_license_code(raw) in REDISTRIBUTABLE_LICENSES else 0


class LicenseDecision:
    """The outcome of the gate for one document, with its reason.

    ``admitted`` is the only field a caller should branch on; ``code`` and
    ``detail`` exist so a rejection can be counted by cause in a report rather
    than tallied as one undifferentiated "skipped" number.
    """

    __slots__ = ("admitted", "code", "detail", "license", "redistributable",
                 "channel")

    def __init__(self, *, admitted: bool, code: str, detail: str,
                 license: str, redistributable: int, channel: str):
        self.admitted = admitted
        self.code = code
        self.detail = detail
        self.license = license
        self.redistributable = redistributable
        self.channel = channel

    def __repr__(self) -> str:                                # pragma: no cover
        return (f"LicenseDecision(admitted={self.admitted!r}, "
                f"code={self.code!r}, license={self.license!r}, "
                f"redistributable={self.redistributable!r}, "
                f"channel={self.channel!r})")


def license_decision(
    *,
    s3_license_code: object = None,
    rest_license: object = None,
    is_retracted: object = None,
) -> LicenseDecision:
    """Gate one document on its licence and retraction status.

    The licence is taken from the S3 metadata ``license_code`` when present and
    from the Europe PMC REST ``license`` field otherwise. **Never from the
    JATS**: SOURCES.md measured ``ali:license_ref`` present in only 2 of 5
    sampled open-access articles, while the REST field was correct in all 5, so
    a JATS-trusting parser silently drops most documents or — worse — indexes
    them with an unknown licence.

    Retraction is checked *first* and independently of licence. A retracted
    paper is not a licensing question; a CC BY retracted paper is still a
    document the coach must never cite.
    """
    if is_retracted is True or (isinstance(is_retracted, str)
                                and is_retracted.strip().lower() == "true"):
        return LicenseDecision(
            admitted=False, code="retracted",
            detail="metadata says is_retracted; a retracted paper is excluded "
                   "regardless of licence",
            license=normalize_license_code(s3_license_code or rest_license),
            redistributable=0, channel="none")

    if normalize_license_code(s3_license_code):
        raw, channel = s3_license_code, "s3_metadata.license_code"
    elif normalize_license_code(rest_license):
        raw, channel = rest_license, "europepmc_rest.license"
    else:
        return LicenseDecision(
            admitted=False, code="license_absent",
            detail="neither the S3 metadata license_code nor the Europe PMC "
                   "REST license field carried a value; an unidentifiable "
                   "licence is all-rights-reserved, not a guess",
            license="", redistributable=0, channel="none")

    normalized = normalize_license_code(raw)
    flag = redistributable_flag(raw)
    if not flag:
        return LicenseDecision(
            admitted=False, code="license_not_redistributable",
            detail=f"licence {normalized!r} (from {channel}) is not in the "
                   f"allow-list {sorted(REDISTRIBUTABLE_LICENSES)}",
            license=normalized, redistributable=0, channel=channel)

    return LicenseDecision(
        admitted=True, code="ok",
        detail=f"licence {normalized!r} from {channel}",
        license=normalized, redistributable=1, channel=channel)


# --------------------------------------------------------------------------- #
# Rate discipline
# --------------------------------------------------------------------------- #

class RateLimiter:
    """A blocking minimum-interval limiter, per host.

    Deliberately not a token bucket: a bucket permits a burst, and a burst is
    the thing NCBI's rule forbids. The guarantee here is that two calls for the
    same host never leave less than ``1 / rate`` seconds between them.

    `sleep` and `now` are injectable so the limiter is testable without a test
    that actually waits — a rate-limit test that sleeps is a rate-limit test
    nobody runs.
    """

    def __init__(self, rate_per_second: float = RATE_LIMIT_PER_SECOND, *,
                 now=time.monotonic, sleep=time.sleep):
        if rate_per_second <= 0:
            raise SourceRefusal(
                "bad_rate", f"rate {rate_per_second!r} must be positive; "
                f"an unlimited fetcher risks an IP block on the whole machine")
        self.min_interval = 1.0 / float(rate_per_second)
        self._now = now
        self._sleep = sleep
        self._last: dict[str, float] = {}
        self.waits = 0
        self.total_wait = 0.0

    def wait(self, host: str) -> float:
        """Block until `host` may be called again. Returns seconds slept."""
        key = host.lower()
        last = self._last.get(key)
        slept = 0.0
        if last is not None:
            gap = self._now() - last
            if gap < self.min_interval:
                slept = self.min_interval - gap
                self._sleep(slept)
                self.waits += 1
                self.total_wait += slept
        self._last[key] = self._now()
        return slept


