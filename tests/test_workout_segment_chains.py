"""get_workout_segments must report ONE partition of a workout (audit P1-1).

The watch stores several independent segmentations of the same session — the
2026-07-26 run carries a 5-split chain and a 4-split chain, each covering the
full 36.1 minutes. Concatenating them and numbering 1..n invented a 9-split
session totalling 72.2 minutes, with split 4 fully containing splits 5 and 6 and
their heart-rate samples counted twice. 302 of 370 workouts with segments were
affected, and AGENTS.md points the agent at this tool for negative-split and
HR-drift checks, so it would grade a prescribed session against a structure that
never happened.
"""
from __future__ import annotations

import pytest

from health_advisor import db as dbmod
from health_advisor import metrics as M

DAY = "2026-07-26"
KEY = "wk-run"

# Verbatim from the live DB: two chains, 12:17:15 -> 12:53:19, 36.09 min each.
EVENTS = [
    ("segment", "2026-07-26T12:17:15+00:00", "2026-07-26T12:24:47.271792+00:00", 7.54),
    ("segment", "2026-07-26T12:17:15+00:00", "2026-07-26T12:29:03.447422+00:00", 11.81),
    ("segment", "2026-07-26T12:24:47+00:00", "2026-07-26T12:32:50.478668+00:00", 8.06),
    ("segment", "2026-07-26T12:29:03+00:00", "2026-07-26T12:44:03.060012+00:00", 15.0),
    ("marker", "2026-07-26T12:29:19+00:00", None, None),
    ("segment", "2026-07-26T12:32:51+00:00", "2026-07-26T12:42:06.748918+00:00", 9.26),
    ("segment", "2026-07-26T12:42:06+00:00", "2026-07-26T12:51:08.793437+00:00", 9.05),
    ("segment", "2026-07-26T12:44:03+00:00", "2026-07-26T12:53:13.542752+00:00", 9.18),
    ("segment", "2026-07-26T12:51:09+00:00", "2026-07-26T12:53:19.681447+00:00", 2.18),
    ("segment", "2026-07-26T12:53:14+00:00", "2026-07-26T12:53:19.924076+00:00", 0.1),
]


def _seed(conn, events=EVENTS, duration_min=36.1):
    dbmod.insert_workouts(conn, [dict(
        workout_type="running", start_utc="2026-07-26T12:17:15+00:00",
        end_utc="2026-07-26T12:53:19+00:00", local_date=DAY,
        duration_min=duration_min, energy_kcal=400.0, distance_mi=3.1,
        unit_distance="mi", source="Watch", dedupe_key=KEY)])
    dbmod.insert_workout_events(conn, [
        dict(workout_key=KEY, event_type=t, start_utc=s, end_utc=e,
             duration_min=d, dedupe_key=f"{t}|{s}|{d}")
        for t, s, e, d in events])
    conn.commit()


@pytest.fixture
def served(conn, tools):
    _seed(conn)
    return lambda: tools.get_workout_segments(DAY)["workouts"][0]


# --------------------------------------------------------------------------- #
# The partitioner itself.
# --------------------------------------------------------------------------- #
def _ev(start, end, dur):
    return {"start_utc": start, "end_utc": end, "duration_min": dur}


def test_segment_chains_separates_two_partitions_of_one_workout():
    chains = M.segment_chains([_ev(s, e, d) for t, s, e, d in EVENTS
                               if t == "segment"])

    assert [len(c) for c in chains] == [5, 4]


def test_each_chain_covers_the_workout_once():
    chains = M.segment_chains([_ev(s, e, d) for t, s, e, d in EVENTS
                               if t == "segment"])

    for chain in chains:
        assert sum(c["duration_min"] for c in chain) == pytest.approx(36.09, abs=0.02)


def test_a_gap_does_not_start_a_new_chain():
    # A pause leaves a real hole in one partition; that is still one partition.
    chains = M.segment_chains([
        _ev("2026-07-26T12:00:00+00:00", "2026-07-26T12:10:00+00:00", 10.0),
        _ev("2026-07-26T12:15:00+00:00", "2026-07-26T12:25:00+00:00", 10.0),
    ])

    assert len(chains) == 1


def test_chains_are_ranked_by_coverage():
    chains = M.segment_chains([
        _ev("2026-07-26T12:00:00+00:00", "2026-07-26T12:05:00+00:00", 5.0),
        _ev("2026-07-26T12:00:00+00:00", "2026-07-26T12:30:00+00:00", 30.0),
    ])

    assert chains[0][0]["duration_min"] == 30.0


# --------------------------------------------------------------------------- #
# The tool.
# --------------------------------------------------------------------------- #
def test_tool_reports_one_partition_not_the_concatenation(served):
    w = served()

    assert w["n_segments"] == 5
    assert len(w["segments"]) == 5


def test_tool_splits_sum_to_the_workout_duration(served):
    w = served()

    assert w["covered_min"] == pytest.approx(36.09, abs=0.02)
    assert w["covered_min"] <= w["duration_min"] * 1.05


def test_tool_numbers_splits_within_the_chain(served):
    w = served()

    assert [s["n"] for s in w["segments"]] == [1, 2, 3, 4, 5]


def test_tool_returns_each_splits_end_time(served):
    w = served()

    assert w["segments"][0]["end_local"] is not None
    assert w["segments"][0]["end_local"] > w["segments"][0]["start_local"]


def test_tool_exposes_the_other_partition_separately(served):
    w = served()

    alts = w["alternate_segmentations"]
    assert len(alts) == 1
    assert alts[0]["n_segments"] == 4
    assert alts[0]["covered_min"] == pytest.approx(36.09, abs=0.02)


def test_list_workouts_agrees_with_the_segment_tool(conn, tools):
    _seed(conn)

    listed = tools.list_workouts(start=DAY, end=DAY)["workouts"][0]
    detailed = tools.get_workout_segments(DAY)["workouts"][0]

    assert listed["n_segments"] == detailed["n_segments"] == 5


# --------------------------------------------------------------------------- #
# duration_min is ACTIVE time; splits span wall clock. On a workout with a long
# pause the two legitimately disagree (2020-08-24: 8.6 min active, 38.2
# elapsed), so the completeness check has to measure against elapsed time or it
# calls every paused session broken.
# --------------------------------------------------------------------------- #
def test_paused_workout_is_not_called_partial(conn, tools):
    _seed(conn, events=[
        ("segment", "2026-07-26T12:17:15+00:00", "2026-07-26T12:35:00+00:00", 17.75),
        ("segment", "2026-07-26T12:35:00+00:00", "2026-07-26T12:53:19+00:00", 18.32),
    ], duration_min=8.6)

    w = tools.get_workout_segments(DAY)["workouts"][0]
    assert w["elapsed_min"] == pytest.approx(36.1, abs=0.05)
    assert w["duration_min"] == 8.6
    assert w["note"] is None


def test_genuinely_partial_splits_are_flagged(conn, tools):
    _seed(conn, events=[
        ("segment", "2026-07-26T12:17:15+00:00", "2026-07-26T12:22:15+00:00", 5.0),
    ], duration_min=36.1)

    w = tools.get_workout_segments(DAY)["workouts"][0]
    assert "partial" in w["note"]
