"""qualified_block_min is the longest block that ran easy enough, which can be a
different and shorter block than longest_block_min (consumer issue #712).

A 59.0-minute run read qualified 0.3 because one long block ran over the ceiling
and a short separate rep under it qualified. The figure is the house-rule dial
and stays; the new qualified_is_longest flag says whether the qualifying block
is the longest block itself. Identity (start and end of the rep) decides it,
never minutes: two different blocks can tie in minutes.

Oracles are literals computed by hand from the bucket arithmetic: a 20-second
bucket is 1/3 minute, so 30 buckets = 10.0 min and 6 buckets = 2.0 min. The
ceiling is the legacy default 150 bpm.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from health_advisor import analysis
from health_advisor import mcp_server
from health_advisor import metrics as mx

BUCKET_MIN = mx.IMPACT_BUCKET_SECONDS / 60.0   # 1/3 minute
BASE = "2001-07-01T12:00:00Z"
DAY = "2026-07-06"
TOOL_BASE = "2026-07-06T12:00:00Z"

# Hand-computed from the bucket arithmetic above.
LONG_MIN = 10.0        # 30 buckets
REP_MIN = 2.0          # 6 buckets
RUN_UNDER_CEILING_MIN = 10.0


def _bucket(i, *, hr, is_jog, pace=10.0, cadence=150.0, base=BASE):
    start = datetime.fromisoformat(base.replace("Z", "+00:00"))
    return {
        "bucket_start_utc": (start + timedelta(seconds=mx.IMPACT_BUCKET_SECONDS * i)
                             ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_date": base[:10],
        "pace_min_per_mi": pace,
        "hr": hr,
        "cadence_spm": cadence,
        "is_jog": is_jog,
    }


def _series(*specs):
    """Consecutive buckets from BASE. Each spec is (count, hr, is_jog[, pace, cadence])."""
    out, i = [], 0
    for spec in specs:
        count, hr, is_jog, *rest = spec
        kw = {}
        if rest:
            kw["pace"] = rest[0]
        if len(rest) > 1:
            kw["cadence"] = rest[1]
        for _ in range(count):
            out.append(_bucket(i, hr=hr, is_jog=is_jog, **kw))
            i += 1
    return out


JOG = dict(is_jog=True)


def _walk_gap(count=3):
    """Not jog (cadence 105) and not a bridge (HR 110 < 130): it splits blocks."""
    return (count, 110.0, False, 17.0, 105.0)


def test_control_one_continuous_easy_block_is_its_own_qualifier():
    block = analysis.longest_block_from_buckets(_series((30, 140.0, True)))

    assert block["bridged_min"] == LONG_MIN
    assert block["qualified_min"] == LONG_MIN
    assert block["qualified_is_longest"] is True


def test_long_block_over_the_ceiling_plus_short_easy_rep_is_not_the_longest():
    # The 10-24 shape: 10 min at HR 160 (over the 150 ceiling), a walk gap that
    # cannot bridge, then a 2 min rep at HR 140 that qualifies.
    block = analysis.longest_block_from_buckets(_series(
        (30, 160.0, True), _walk_gap(), (6, 140.0, True)))

    assert block["bridged_min"] == LONG_MIN
    assert block["avg_hr_longest_block"] == 160.0
    assert block["qualified_min"] == REP_MIN
    assert block["qualified_is_longest"] is False
    assert len(block["reps"]) == 2


def test_no_qualifying_block_gives_null_figure_and_null_flag():
    block = analysis.longest_block_from_buckets(_series((30, 170.0, True)))

    assert block["bridged_min"] == LONG_MIN
    assert block["qualified_min"] is None
    assert block["qualified_is_longest"] is None


def test_heart_rate_not_synced_gives_null_flag():
    block = analysis.longest_block_from_buckets(_series((6, None, True)))

    assert block["hr"] == "not_synced"
    assert block["qualified_min"] is None
    assert block["qualified_is_longest"] is None


def test_equal_minutes_but_different_blocks_is_false_not_a_tie_win():
    # Two 2.0-minute reps. The earlier one runs over the ceiling (160), so it
    # is the longest by the stable order; the later one qualifies (140). They
    # tie in minutes, so only identity can say the qualifier is not the longest.
    block = analysis.longest_block_from_buckets(_series(
        (6, 160.0, True), _walk_gap(), (6, 140.0, True)))

    assert block["bridged_min"] == REP_MIN
    assert block["qualified_min"] == REP_MIN
    assert block["qualified_is_longest"] is False


def test_equal_minutes_when_the_first_block_qualifies_is_true():
    # Same tie, both under the ceiling: the qualifier is the first (longest) rep.
    block = analysis.longest_block_from_buckets(_series(
        (6, 140.0, True), _walk_gap(), (6, 140.0, True)))

    assert block["qualified_min"] == REP_MIN
    assert block["qualified_is_longest"] is True


# ---- the tool description ----

def test_tool_description_says_the_qualifier_can_be_a_shorter_block():
    doc = mcp_server.get_block_structure.__doc__
    assert "it can be a different, shorter block" in doc
    assert "the same block only if" not in doc
    assert "qualified_is_longest" in doc


# ---- through get_block_structure on a tmp vault ----

def _record(conn, metric, start, value, unit):
    conn.execute(
        "INSERT INTO records (metric, start_utc, end_utc, local_date, value, "
        "unit, source, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, 'test', ?)",
        (metric, start, start, start[:10], value, unit,
         f"{metric}|{start}|{value}"))


def _iso(i):
    t = datetime.fromisoformat(TOOL_BASE.replace("Z", "+00:00"))
    return (t + timedelta(seconds=mx.IMPACT_BUCKET_SECONDS * i)
            ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_session(conn, specs):
    """Write the per-bucket samples for a running session, then its workout row.

    specs: (count, hr, pace, cadence) per run of buckets. Distance = 20 s at
    the pace, steps = 20 s at the cadence, so the bucket pace and cadence match.
    """
    i = 0
    for count, hr, pace, cadence in specs:
        for _ in range(count):
            s = _iso(i)
            _record(conn, "distance_walking_running", s, BUCKET_MIN / pace, "mi")
            _record(conn, "step_count", s, cadence * BUCKET_MIN, "count")
            _record(conn, "heart_rate", s, hr, "count/min")
            i += 1
    end = _iso(i)
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, avg_heart_rate, source, dedupe_key) "
        "VALUES ('running', ?, ?, ?, ?, ?, 'test', ?)",
        (TOOL_BASE, end, DAY, i * BUCKET_MIN, None, f"w|{TOOL_BASE}|running"))


def test_session_entry_carries_the_flag_when_the_rep_is_not_the_longest(conn, tools):
    _run_session(conn, [
        (30, 160.0, 10.0, 150.0),
        (3, 110.0, 17.0, 105.0),
        (6, 140.0, 10.0, 150.0),
    ])
    conn.commit()

    out = tools.get_block_structure(day=DAY)

    session = out["sessions"][0]
    assert session["longest_block_min"] == LONG_MIN
    assert session["qualified_block_min"] == REP_MIN
    assert session["qualified_is_longest"] is False
    assert out["qualified_block_min"] == REP_MIN
    assert out["qualified_is_longest"] is False


def test_session_entry_and_day_flag_are_true_when_the_longest_qualifies(conn, tools):
    _run_session(conn, [(30, 140.0, 10.0, 150.0)])
    conn.commit()

    out = tools.get_block_structure(day=DAY)

    session = out["sessions"][0]
    assert session["qualified_block_min"] == LONG_MIN
    assert session["qualified_is_longest"] is True
    assert out["qualified_is_longest"] is True
