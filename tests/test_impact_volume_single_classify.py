"""get_impact_volume classifies its window's buckets once (#110).

The period rows and the cadence-sensitivity table are two views of one bucket
set. Computing that set twice cost the same raw-sample scan twice and left room
for the two to disagree; these tests pin that the tool classifies once, and that
handing a pre-classified set to either consumer changes no figure.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from health_advisor import analysis as A
from health_advisor import db, mcp_server
from health_advisor import metrics as mx
from health_advisor.context import VaultContext

DAYS = ("2026-07-27", "2026-07-28", "2026-08-04")


def _insert(conn, metric, value, unit, ts, day, key):
    conn.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 't', 't', ?)",
        (metric, value, unit, ts, ts, ts, day, key))


def _session(conn, day, start_hhmm, seconds, pace_min_per_mi, cadence_spm,
             hr=None):
    """One workout with per-second distance and 20-second step samples."""
    t0 = datetime.fromisoformat(f"{day}T{start_hhmm}:00+00:00")
    start = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (t0 + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('other', ?, ?, ?, ?, 'test', ?)",
        (start, end, day, seconds / 60.0, f"workout|{start}|{end}"))
    per_second = 1.0 / (pace_min_per_mi * 60.0)
    for i in range(seconds):
        ts = (t0 + timedelta(seconds=i)).isoformat()
        tag = f"{day}-{start_hhmm}-{i}"
        _insert(conn, "distance_walking_running", per_second, "mi", ts, day,
                f"d-{tag}")
        if hr is not None and i % 5 == 0:
            _insert(conn, "heart_rate", hr, "count/min", ts, day, f"h-{tag}")
        if cadence_spm and i % mx.IMPACT_BUCKET_SECONDS == 0:
            _insert(conn, "step_count", cadence_spm / 3.0, "count", ts, day,
                    f"s-{tag}")


@pytest.fixture
def vault(tmp_path):
    path = tmp_path / "vault.db"
    conn = db.connect(path)
    db.init_db(conn)
    # Clear jogging, a walk, a cadence just under the gait line (a "near" bucket
    # for the sensitivity table), and a heart-rate-confirmed tired stride.
    _session(conn, DAYS[0], "12:00", 300, 12.0, 160.0)
    _session(conn, DAYS[0], "12:10", 300, 22.0, 0.0)
    _session(conn, DAYS[1], "12:00", 300, 13.0, mx.IMPACT_JOG_CADENCE_MIN - 4)
    _session(conn, DAYS[1], "12:10", 300, 13.0, mx.IMPACT_JOG_EFFORT_CADENCE_MIN,
             hr=mx.IMPACT_JOG_HR_MIN + 5)
    _session(conn, DAYS[2], "12:00", 300, 12.5, 150.0)
    conn.commit()
    conn.close()
    ctx = VaultContext.local(path, user_id="t", writable=False)
    return SimpleNamespace(path=path, tools=SimpleNamespace(
        **mcp_server.build_tools(ctx)))


def _dump(value):
    return json.dumps(value, sort_keys=True)


def test_tool_classifies_the_window_once(vault, monkeypatch):
    calls = []
    real = mx.impact_bucket_rows

    def spy(conn, window_predicate, window_args, **kwargs):
        calls.append((window_predicate, tuple(window_args)))
        return real(conn, window_predicate, window_args, **kwargs)

    monkeypatch.setattr(mx, "impact_bucket_rows", spy)
    out = vault.tools.get_impact_volume(start=DAYS[0], end=DAYS[2], by="week")
    assert "error" not in out and out["periods"]
    assert out["jog_threshold_sensitivity"] and out["jog_near_threshold"]
    assert len(calls) == 1


@pytest.mark.parametrize("by", ["day", "week"])
@pytest.mark.parametrize("metric_units", [False, True])
def test_impact_volume_with_supplied_buckets_is_identical(
        vault, by, metric_units):
    conn = db.connect(vault.path, read_only=True)
    try:
        buckets = mx.impact_bucket_rows(
            conn, "local_date BETWEEN ? AND ?", (DAYS[0], DAYS[2]))
        snapshot = copy.deepcopy(buckets)
        fresh = A.impact_volume(conn, DAYS[0], DAYS[2], by=by,
                                metric_units=metric_units)
        shared = A.impact_volume(conn, DAYS[0], DAYS[2], by=by,
                                 metric_units=metric_units,
                                 bucket_rows=buckets)
    finally:
        conn.close()
    assert fresh and _dump(shared) == _dump(fresh)
    assert buckets == snapshot


def test_sensitivity_with_supplied_buckets_is_identical(vault):
    conn = db.connect(vault.path, read_only=True)
    try:
        buckets = mx.impact_bucket_rows(
            conn, "local_date BETWEEN ? AND ?", (DAYS[0], DAYS[2]))
        snapshot = copy.deepcopy(buckets)
        fresh = mcp_server._jog_threshold_sensitivity(conn, DAYS[0], DAYS[2])
        shared = mcp_server._jog_threshold_sensitivity(
            conn, DAYS[0], DAYS[2], bucket_rows=buckets)
    finally:
        conn.close()
    assert fresh and _dump(shared) == _dump(fresh)
    assert fresh["near"]["buckets_near_cutoff"] > 0
    assert buckets == snapshot


@pytest.mark.parametrize("by", ["day", "week"])
def test_published_payload_matches_two_independent_classifications(vault, by):
    """The tool's rows and sensitivity equal what two separate scans produce."""
    out = vault.tools.get_impact_volume(start=DAYS[0], end=DAYS[2], by=by)
    conn = db.connect(vault.path, read_only=True)
    try:
        rows = A.impact_volume(conn, DAYS[0], DAYS[2], by=by)
        sens = mcp_server._jog_threshold_sensitivity(conn, DAYS[0], DAYS[2])
    finally:
        conn.close()
    # The tool fills gap days with zero rows; every measured row must match.
    published = {p["period_start"]: p["jog_minutes"] for p in out["periods"]}
    assert rows and {r["period_start"]: r["jog_minutes"] for r in rows} == {
        start: published[start] for start in (r["period_start"] for r in rows)}
    assert _dump(out["jog_threshold_sensitivity"]) == _dump(sens["sensitivity"])
    assert _dump(out["jog_near_threshold"]) == _dump(sens["near"])


def test_bad_grouping_is_refused_before_any_scan(vault, monkeypatch):
    calls = []
    monkeypatch.setattr(mx, "impact_bucket_rows",
                        lambda *a, **k: calls.append(a) or [])
    out = vault.tools.get_impact_volume(start=DAYS[0], end=DAYS[2], by="month")
    assert out == {"error": "by must be 'week' or 'day'"}
    assert calls == []
