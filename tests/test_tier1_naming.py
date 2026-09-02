"""Names in the Tier-1 modules must not borrow authority from constructs they
do not measure.

An earlier draft of this work named three measures after validated constructs
they only approximate — aerobic decoupling, SRI, overreaching — which is how an
unvalidated number acquires borrowed credibility. This project's entire
architecture exists to prevent that. See the design spec, section 11.2."""
from __future__ import annotations

import io
import pathlib
import tokenize

MODULES = ["running_form.py", "sleep_regularity.py", "hr_load.py"]

# Names that may not appear as identifiers. Prose may discuss them — the module
# docstrings explain exactly why each is NOT what is being computed — so this
# checks code, not comments.
FORBIDDEN = ["decoupling", "aerobic_decoupling", "sri", "overreaching",
             "trimp", "fitness", "fatigue"]


def _identifier_tokens(path: pathlib.Path):
    """Python identifier tokens, with comments and all string literals removed."""
    source = path.read_text()
    return [token for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.NAME]


def test_no_forbidden_names_as_identifiers():
    root = pathlib.Path(__file__).parent.parent / "health_advisor"
    offenders = []
    for name in MODULES:
        path = root / name
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        for token in _identifier_tokens(path):
            identifier = token.string.lower()
            for word in FORBIDDEN:
                if word in identifier:
                    line = lines[token.start[0] - 1].strip()
                    offenders.append(f"{name}:{token.start[0]}: {word!r} in {line!r}")
    assert not offenders, (
        "Forbidden construct names used as identifiers:\n" + "\n".join(offenders)
        + "\n\nSee the design spec section 11.2. These measures are proxies; "
        "naming them after the validated constructs they approximate lends "
        "them authority they have not earned."
    )


def test_the_persisted_metric_names_are_honest():
    from health_advisor import normalize as nz
    for metric in ("hr_load_proxy", "sleep_midpoint_sd_28d",
                   "sleep_timing_interval_regularity"):
        assert metric in nz.CATALOG
    for banned in ("trimp", "sri", "fitness", "fatigue"):
        assert not any(banned in m.lower() for m in nz.CATALOG), banned
