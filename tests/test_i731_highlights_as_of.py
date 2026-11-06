"""A briefing for a past day names no all-time high set after that day
(downstream issue #731).

``analysis.highlights`` took its all-time maximum over the whole vault, so
``get_briefing(day=D)`` could report a "new recent high" dated after D. The
vault below holds step counts on 2026-08-10..2026-08-16 (a high of 12,000 on
the 14th) and one later day, 2026-08-20, at 15,000.
"""
from __future__ import annotations

from health_advisor import analysis as A
from health_advisor import db
from health_advisor import mcp_server as S
from health_advisor.context import VaultContext

from tests.conftest import seed_metric

STEPS = [8000, 9000, 10000, 11000, 12000, 9500, 9000]   # 08-10 .. 08-16


def _vault(tmp_path):
    path = tmp_path / "v.db"
    conn = db.connect(str(path))
    db.init_db(conn)
    seed_metric(conn, "step_count", "2026-08-10", STEPS)
    seed_metric(conn, "step_count", "2026-08-20", [15000])
    return path, conn


def test_highlights_for_a_past_day_stop_at_that_day(tmp_path):
    _, conn = _vault(tmp_path)
    try:
        # As of the 16th the high is the 14th's 12,000, inside the last 7 days.
        assert A.highlights(conn, "2026-08-16") == [
            {"metric": "step_count", "kind": "all_time_high",
             "value": 12000, "date": "2026-08-14"}]
        # As of the latest day the 20th's 15,000 is the high, as before.
        assert A.highlights(conn, "2026-08-20") == [
            {"metric": "step_count", "kind": "all_time_high",
             "value": 15000, "date": "2026-08-20"}]
        assert A.highlights(conn) == A.highlights(conn, "2026-08-20")
        # As of the 12th the high so far is the 12th's own 10,000.
        assert A.highlights(conn, "2026-08-12") == [
            {"metric": "step_count", "kind": "all_time_high",
             "value": 10000, "date": "2026-08-12"}]
    finally:
        conn.close()


def test_get_briefing_for_a_past_day_names_no_later_high(tmp_path):
    path, conn = _vault(tmp_path)
    conn.close()
    tools = S.build_tools(VaultContext.local(path, user_id="test"))
    past = tools["get_briefing"](day="2026-08-16")
    assert [(h["value"], h["date"]) for h in past["highlights"]] == \
        [(12000, "2026-08-14")]
    latest = tools["get_briefing"]()
    assert [(h["value"], h["date"]) for h in latest["highlights"]] == \
        [(15000, "2026-08-20")]
