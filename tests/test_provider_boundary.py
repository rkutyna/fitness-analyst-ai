"""What a model provider may receive (F-02, #12).

ARCHITECTURE §5's privacy claim is the product's central differentiator, which
makes it the worst thing to be wrong about and the most likely to be checked.
The original wording — "the provider never receives raw health data" — was not
supported by the code: `get_latest` returned a single stored reading with its
timestamp, and it is in the researcher tool set.

The supportable claim is narrower and now enforced: **a session whose output
reaches a provider does not receive a value identifying one stored sample.**
Aggregates over a window are not that, even sub-daily ones — a 20-minute mean
does not reveal a reading.

The mechanism is a capability, not care: the researcher's server is built from
`ctx.provider_facing()`, so a tool cannot forget a check against a capability it
was never handed.
"""
from __future__ import annotations

import inspect

import pytest

from health_advisor import db as dbmod
from health_advisor import llm
from health_advisor import mcp_server as S
from health_advisor.context import RAW_SAMPLES, WRITE, VaultContext
from tests.conftest import seed_metric, seed_workout

SAMPLE_VALUE = 61.0
SAMPLE_TIME = "2026-08-01T09:17:33+00:00"
SAMPLE_LOCAL = "2026-08-01T09:17:33"


@pytest.fixture
def seeded(vault):
    """A vault holding exactly one heart_rate sample with a distinctive time."""
    conn = vault.connect()
    dbmod.init_db(conn)
    seed_metric(conn, "heart_rate", "2026-08-01", [SAMPLE_VALUE])
    seed_metric(conn, "step_count", "2026-08-01", [8000.0])
    seed_workout(conn, "running", "2026-08-01", 41.0, 2.04)
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) VALUES "
        "('heart_rate', ?, 'count/min', ?, ?, ?, '2026-08-01', 'w', 't', 'k1')",
        (SAMPLE_VALUE, SAMPLE_TIME, SAMPLE_TIME, SAMPLE_LOCAL))
    conn.commit()
    conn.close()
    return vault


def test_a_local_session_still_gets_the_sample(seeded):
    """The boundary is about provider egress, not about the user's own data.

    The desktop review workflow reads samples constantly, and breaking that to
    protect the user from themselves would be the wrong trade."""
    out = S.build_tools(seeded)["get_latest"]("heart_rate")
    assert out["latest_sample"]["value"] == pytest.approx(SAMPLE_VALUE)
    assert out["latest_sample"]["local_time"] == SAMPLE_LOCAL


def test_a_provider_facing_session_does_not_get_the_sample(seeded):
    out = S.build_tools(seeded.provider_facing())["get_latest"]("heart_rate")

    assert out["latest_sample"] is None
    # Withheld, not missing — the distinction T-006 exists for. A bare null
    # would be a claim about the data.
    assert out["latest_sample_status"]["status"] == "withheld"
    assert out["latest_sample_status"]["reason"] == "provider_facing_session"
    # And the aggregate is untouched, so the model is not left without an answer.
    assert out["latest_day"]["value"] == pytest.approx(SAMPLE_VALUE)


def test_provider_facing_drops_write_as_well_as_raw_samples(vault):
    narrowed = vault.provider_facing()
    assert not narrowed.can(RAW_SAMPLES)
    assert not narrowed.can(WRITE)


def test_the_researcher_server_is_built_provider_facing():
    """The mechanism, asserted at the seam rather than inferred from behaviour.

    If this narrowing is ever removed, every tool would silently regain
    sample-level egress and only the per-tool checks would stand between the
    provider and a reading.
    """
    source = inspect.getsource(llm._registry)
    assert "ctx.provider_facing()" in source, \
        "the researcher registry stopped narrowing its session"
    from health_advisor import deepdive_mcp
    assert "ctx.provider_facing()" in inspect.getsource(deepdive_mcp.build_server)


def test_no_researcher_tool_returns_a_stored_sample_timestamp(seeded):
    """The general check, not a list of known offenders.

    Every tool the researcher can call, invoked against a vault holding one
    sample with a distinctive second-level timestamp. If that timestamp appears
    anywhere in any result, some tool is handing a provider the identity of a
    single reading — which is the claim, whichever tool did it. The value case
    is recorded separately: a one-sample daily aggregate is expected to equal
    the sample, but must not carry its timestamp.
    """
    tools = S.build_tools(seeded.provider_facing())
    leaked, value_tools, invoked = [], [], []
    for name in llm.RESEARCHER_TOOLS:
        fn = tools[name]
        kwargs = _plausible_args(fn)
        if kwargs is None:
            continue
        invoked.append(name)
        try:
            result = fn(**kwargs)
        except Exception:          # a tool refusing is not a leak
            continue
        if SAMPLE_LOCAL in repr(result) or SAMPLE_TIME in repr(result):
            leaked.append(name)
        if _contains_value(result, SAMPLE_VALUE):
            value_tools.append(name)

    assert leaked == [], f"tools returned a stored sample's timestamp: {leaked}"
    assert set(value_tools) == {
        "get_daily_series", "summarize_metric", "compare_periods",
        "get_intraday", "get_latest", "get_hr_zones",
    }, f"unexpected one-sample value results: {sorted(set(value_tools))}"
    # A check that silently exercised nothing would pass forever.
    assert set(invoked) == set(llm.RESEARCHER_TOOLS), \
        f"not exercised: {sorted(set(llm.RESEARCHER_TOOLS) - set(invoked))}"


def _contains_value(result, expected: float) -> bool:
    if isinstance(result, dict):
        return any(_contains_value(value, expected) for value in result.values())
    if isinstance(result, (list, tuple)):
        return any(_contains_value(value, expected) for value in result)
    return result == expected


def _plausible_args(fn) -> dict | None:
    """Fill a tool's signature with arguments that will actually reach data."""
    defaults = {
        "metric": "heart_rate", "metric_x": "heart_rate", "metric_y": "step_count",
        "day": "2026-08-01", "period": "30d", "period_a": "30d", "period_b": "30d",
        "start": "2026-08-01", "end": "2026-08-01", "scope": "daily",
        "workout_date": "2026-08-01", "date": "2026-08-01",
        "target": "heart_rate",
    }
    kwargs = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name in defaults:
            kwargs[name] = defaults[name]
        elif param.default is not inspect.Parameter.empty:
            continue
        else:
            return None            # a required argument we cannot guess
    return kwargs
