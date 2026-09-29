"""The synthetic demo vault: does it build, is it deterministic, is it coherent?

This is the file that makes `health_advisor.demo` usable as a correctness gate.
The generator's whole value is that a stranger — or CI — can run the analysis
layer against data with the same SHAPE as a real vault, so the assertions here
are about shape and coherence, not about any particular invented number.

Small windows on purpose: the generator is linear in `days`, and the properties
under test do not need two years to hold. Every build here uses a window ending
at demo.DEFAULT_END_DATE so it crosses the vault's arbitration setting and the
two-device path is reachable.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from health_advisor import db as dbmod
from health_advisor import demo, derive
from health_advisor import normalize as nz
from health_advisor import vault as vaultmod
from health_advisor.context import VaultContext

DAYS = 60


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    """One built vault for the whole module — building it is the slow part."""
    path = tmp_path_factory.mktemp("demo") / "demo.db"
    report = demo.build_demo_vault(path, days=DAYS)
    return path, report


def _daily_rows(path):
    """Every daily aggregate in a built vault, as comparable tuples."""
    c = dbmod.connect(path, read_only=True)
    try:
        return [tuple(r) for r in c.execute(
            "SELECT metric, date, count, sum, avg, last FROM daily_metrics "
            "ORDER BY metric, date")]
    finally:
        c.close()


@pytest.fixture
def conn(demo_db):
    path, _ = demo_db
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# It builds, and the tables a consumer reads are populated
# --------------------------------------------------------------------------- #
def test_vault_opens_and_core_tables_are_populated(conn):
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("records", "workouts", "daily_metrics",
                        "metric_source_months", "subjective")}
    for table, n in counts.items():
        assert n > 0, f"{table} is empty"
    # A vault with a handful of rows would satisfy "non-empty" while testing
    # nothing; these floors are what "a day of data" actually looks like.
    assert counts["records"] > 100 * DAYS
    assert counts["workouts"] >= DAYS // 7


def test_report_describes_what_was_built(demo_db):
    _, report = demo_db
    assert report["days"] == DAYS
    assert report["end_date"] == demo.DEFAULT_END_DATE
    assert report["rows"]["records"] > 0
    assert set(report["workout_types"]) >= {"running", "cycling"}
    assert demo.DEMO_WATCH in report["sources"]
    assert demo.DEMO_PHONE in report["sources"]


def test_vault_is_declared_but_not_d3_filtered(conn):
    from health_advisor import vault as vaultmod
    assert vaultmod.local_timezone(conn) == demo.DEMO_TIMEZONE
    assert vaultmod.unit_system(conn) == "imperial"
    # Not a D3 vault: it carries full-resolution raw records for every series
    # it generates, so `recompute_daily_metrics(full=True)` rebuilds all of them.
    assert not vaultmod.is_vault(conn)
    assert conn.execute(
        "SELECT value FROM vault_meta WHERE key = 'demo_vault'").fetchone()[0] == "1"


# --------------------------------------------------------------------------- #
# Determinism — the property that makes this usable as a gate
# --------------------------------------------------------------------------- #
def test_same_seed_gives_identical_content(tmp_path):
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    demo.build_demo_vault(a, days=25, seed=99)
    demo.build_demo_vault(b, days=25, seed=99)
    # Content, deliberately, not bytes: `vault_meta.created_at` is a wall clock
    # and SQLite page layout is not a documented function of the inserts, so
    # comparing files would fail on two logically identical vaults.
    assert demo.digest_file(a) == demo.digest_file(b)
    rows_a, rows_b = (_daily_rows(p) for p in (a, b))
    assert rows_a and rows_a == rows_b


def test_different_seed_gives_different_content(tmp_path):
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    demo.build_demo_vault(a, days=25, seed=99)
    demo.build_demo_vault(b, days=25, seed=100)
    assert demo.digest_file(a) != demo.digest_file(b)


def test_rebuild_replaces_rather_than_appends(tmp_path):
    path = tmp_path / "again.db"
    first = demo.build_demo_vault(path, days=25, seed=5)
    second = demo.build_demo_vault(path, days=25, seed=5)
    assert first["rows"]["records"] == second["rows"]["records"]
    with pytest.raises(FileExistsError):
        demo.build_demo_vault(path, days=25, seed=5, replace=False)


# --------------------------------------------------------------------------- #
# Coherence: daily_metrics is a cache of records and must reproduce from them
# --------------------------------------------------------------------------- #
def test_daily_metrics_reproduce_from_records(demo_db, tmp_path):
    """The strongest cheap check available — the same one
    scripts/verify_daily_metrics.py makes, run in-process on a copy.

    A stale or hand-written aggregate is the characteristic defect of a
    generator that writes `daily_metrics` directly instead of deriving it, and
    it is invisible to every other assertion in this file.
    """
    path, _ = demo_db
    copy = tmp_path / "copy.db"
    copy.write_bytes(path.read_bytes())
    c = dbmod.connect(copy)
    try:
        before = {(r["metric"], r["date"]): (r["count"], r["sum"], r["avg"],
                                             r["last"])
                  for r in c.execute("SELECT * FROM daily_metrics")}
        derived = set(derive.DERIVED_METRICS)
        dbmod.recompute_daily_metrics(c, full=True)
        after = {(r["metric"], r["date"]): (r["count"], r["sum"], r["avg"],
                                            r["last"])
                 for r in c.execute("SELECT * FROM daily_metrics")}
    finally:
        c.close()
    # Derived rows (sleep timing, wear, the training dial) are not rebuilt from
    # `records` by recompute; everything else must come back identical.
    raw_before = {k: v for k, v in before.items() if k[0] not in derived}
    raw_after = {k: v for k, v in after.items() if k[0] not in derived}
    assert raw_before.keys() == raw_after.keys()
    for key, values in raw_before.items():
        for lhs, rhs in zip(values, raw_after[key]):
            if lhs is None or rhs is None:
                assert lhs == rhs, key
            else:
                assert lhs == pytest.approx(rhs, rel=1e-9), key


def test_every_generated_metric_is_in_the_catalog(conn):
    """An uncatalogued metric silently defaults its unit, aggregation and
    correlation group (normalize.is_known_metric). A demo vault must not teach
    the engine a vocabulary it does not manage."""
    for table, column in (("records", "metric"), ("daily_metrics", "metric")):
        unknown = [r[0] for r in conn.execute(
            f"SELECT DISTINCT {column} FROM {table}")
            if not nz.is_known_metric(r[0])]
        assert unknown == [], f"{table}: {unknown}"


def test_no_real_device_or_person_names(conn):
    sources = {r[0] for r in conn.execute("SELECT DISTINCT source FROM records")}
    sources |= {r[0] for r in conn.execute("SELECT DISTINCT source FROM workouts")}
    assert sources <= {demo.DEMO_WATCH, demo.DEMO_PHONE, demo.DEMO_SCALE,
                       demo.CHECKIN_SOURCE, "Demo Food Log"}
    assert all("'" not in s for s in sources), (
        "Apple device names are possessive ('<name>'s Apple Watch'); a demo "
        "source string must never look like one")


# --------------------------------------------------------------------------- #
# The two correctness-critical shapes: cross-source arbitration, and sleep
# --------------------------------------------------------------------------- #
def test_two_devices_overlap_and_the_watch_wins(conn):
    """The reason a single-source demo vault would be worth little.

    From the vault's arbitration setting both devices record the same
    movement. `db._workout_arbitration` must resolve that to the watch alone —
    if it did not, the stored daily sum would be roughly the sum of both.
    """
    cutoff = vaultmod.workout_source_arbitration_from(conn)
    overlapping = [r["local_date"] for r in conn.execute(
        "SELECT local_date FROM records WHERE metric = 'distance_walking_running' "
        "AND local_date >= ? GROUP BY local_date "
        "HAVING COUNT(DISTINCT source) > 1", (cutoff,))]
    assert overlapping, "no day has two devices writing distance"

    for day in overlapping:
        watch, phone = (conn.execute(
            "SELECT COALESCE(SUM(value), 0) FROM records "
            "WHERE metric = 'distance_walking_running' AND local_date = ? "
            "AND source = ?", (day, src)).fetchone()[0]
            for src in (demo.DEMO_WATCH, demo.DEMO_PHONE))
        stored = conn.execute(
            "SELECT sum FROM daily_metrics WHERE metric = "
            "'distance_walking_running' AND date = ?", (day,)).fetchone()[0]
        assert watch > 0 and phone > 0
        assert stored == pytest.approx(watch, rel=1e-9), day
        assert stored < watch + phone * 0.5, f"{day}: both sources were summed"


def test_sleep_is_attributed_to_the_day_the_session_ends(conn):
    """`derive.reattribute_sleep` rewrites `local_date` to the session's end
    date. Run read-only over the whole window it must find nothing to move —
    which is only true if the generator got the invariant right in the first
    place."""
    span = conn.execute(
        "SELECT MIN(date), MAX(date) FROM daily_metrics").fetchone()
    moves = derive.reattribute_sleep(conn, span[0], span[1], apply=False)
    assert moves == []


def test_sleep_timing_metrics_are_plausible(conn):
    rows = {r["date"]: r["last"] for r in conn.execute(
        "SELECT date, last FROM daily_metrics WHERE metric = 'sleep_bedtime'")}
    assert len(rows) > DAYS // 3
    # Hours since the PREVIOUS day's noon: 22:00 is 10.0, 01:00 is 13.0.
    assert all(9.0 <= v <= 15.0 for v in rows.values()), sorted(rows.values())[:5]
    asleep = [r["sum"] for r in conn.execute(
        "SELECT sum FROM daily_metrics WHERE metric = 'sleep_asleep'")]
    assert all(3 * 60 <= v <= 11 * 60 for v in asleep)


def test_instrument_eras_are_visible_in_provenance(conn, demo_db):
    _, report = demo_db
    sources = {r["source"] for r in conn.execute(
        "SELECT DISTINCT source FROM metric_source_months "
        "WHERE metric = 'distance_walking_running'")}
    assert sources == {demo.DEMO_WATCH, demo.DEMO_PHONE}
    earliest_watch_hr = conn.execute(
        "SELECT MIN(local_date) FROM records WHERE metric = 'heart_rate'"
    ).fetchone()[0]
    assert earliest_watch_hr >= report["watch_from"], (
        "the phone-only era must carry no heart rate")


# --------------------------------------------------------------------------- #
# The point of the whole exercise: the analysis layer answers questions
# --------------------------------------------------------------------------- #
@pytest.fixture
def tools(demo_db):
    from types import SimpleNamespace
    from health_advisor import mcp_server
    path, _ = demo_db
    ctx = VaultContext.local(path, user_id="demo", writable=False)
    return SimpleNamespace(**mcp_server.build_tools(ctx))


def test_briefing_returns_a_populated_briefing(tools, demo_db):
    _, report = demo_db
    out = tools.get_briefing(scope="daily")
    assert "error" not in out
    assert out["as_of"] == report["end_date"]
    assert out["coverage"], "no metric reported coverage"
    assert out["readiness"]["score"] is not None
    assert out["talking_points"]


def test_impact_volume_finds_both_jogging_and_walking(tools, demo_db):
    _, report = demo_db
    out = tools.get_impact_volume(start=report["watch_from"],
                                  end=report["end_date"], by="week")
    assert "error" not in out
    weeks = out["periods"]
    assert weeks, "no impact-volume weeks"
    assert sum(w["jog_minutes"] for w in weeks) > 0, (
        "no jogging classified — the 20-second cadence samples are the whole "
        "reason the generator writes intraday records")
    assert sum(w["walk_minutes"] for w in weeks) > 0
    for w in weeks:
        if w["jog_pace_min_per_mi"] is not None:
            assert 5.0 < w["jog_pace_min_per_mi"] < 16.0


def test_demo_sensitivity_jog_minutes_values_are_unchanged(tools, demo_db, conn):
    """#14: metadata must not alter the Python-computed sensitivity values."""
    from health_advisor import mcp_server

    _, report = demo_db
    start, end = report["watch_from"], report["end_date"]
    expected = mcp_server._jog_threshold_sensitivity(conn, start, end)["sensitivity"]
    published = tools.get_impact_volume(start=start, end=end, by="week")
    fields = ("cadence_min_steps_per_min", "jog_buckets", "jog_minutes", "live_cutoff")
    assert [{field: row[field] for field in fields} for row in published[
        "jog_threshold_sensitivity"]] == [
        {field: row[field] for field in fields} for row in expected]


def test_correlate_metrics_runs_and_reports_honestly(tools):
    out = tools.correlate_metrics(metric_x="sleep_asleep",
                                  metric_y="subjective_energy", period="all")
    assert "error" not in out
    assert out["status"] == "ok"
    assert out["n_pairs"] > 10
    assert -1.0 <= out["pearson_r"] <= 1.0
    assert "correlation is not causation — report as association" in out["caveats"]


def test_get_latest_reads_a_last_valued_metric(tools):
    out = tools.get_latest(metric="vo2_max")
    assert "error" not in out
    assert 25.0 < out["latest_day"]["value"] < 60.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_module_cli_builds_a_vault(tmp_path, capsys):
    out = tmp_path / "cli.db"
    assert demo.main(["--out", str(out), "--days", "8", "--digest"]) == 0
    printed = capsys.readouterr().out
    assert str(out) in printed
    assert "digest" in printed
    assert out.exists()


def test_cli_rejects_a_nonsense_window(tmp_path):
    with pytest.raises(ValueError):
        demo.build_demo_vault(tmp_path / "x.db", days=0)
    with pytest.raises(ValueError):
        demo.build_demo_vault(tmp_path / "x.db", days=5, end_date="not-a-date")


# --------------------------------------------------------------------------- #
# Profiles: the default is pinned draw for draw; a non-default one moves content
# --------------------------------------------------------------------------- #
# The generator draws from ONE random stream, so a single extra (or missing)
# draw anywhere shifts everything after it. These digests are the whole guard on
# that: they were measured on the generator before it took a profile, and a
# change that moves them has changed the default vault, not just added a knob.
PINNED_DIGESTS = {
    (730, 42): "452d7a69d17d53d3c44dfb259812d897b933612581ceab08c07de4d7b54809b8",
    (180, 7): "f6515971dd437e3e4b977a5167cdfcd7437d89969f01ab68aee2d31f9ec1e45c",
}


@pytest.mark.parametrize("days,seed", sorted(PINNED_DIGESTS))
def test_default_vault_digest_is_pinned(tmp_path, days, seed):
    path = tmp_path / "pinned.db"
    demo.build_demo_vault(path, days=days, seed=seed)
    assert demo.digest_file(path) == PINNED_DIGESTS[(days, seed)]


def test_explicit_default_profile_is_the_historical_vault(tmp_path):
    path = tmp_path / "explicit.db"
    report = demo.build_demo_vault(path, days=180, seed=7,
                                   profile=demo.DemoProfile())
    assert demo.digest_file(path) == PINNED_DIGESTS[(180, 7)]
    assert "profile" not in report
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert c.execute("SELECT COUNT(*) FROM vault_meta "
                         "WHERE key = 'demo_profile'").fetchone()[0] == 0
    finally:
        c.close()


def _workout_days(path, kind):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [r[0] for r in c.execute(
            "SELECT local_date FROM workouts WHERE workout_type = ? "
            "ORDER BY local_date", (kind,))]
    finally:
        c.close()


def test_non_default_profile_changes_content_and_is_recorded(tmp_path):
    base, other = tmp_path / "base.db", tmp_path / "other.db"
    demo.build_demo_vault(base, days=40, seed=3)
    report = demo.build_demo_vault(
        other, days=40, seed=3,
        profile=demo.DemoProfile(name="couch", rhr_base=72.0, steps_base=3000.0,
                                 weekly_plan={}, watch_from=0))
    assert demo.digest_file(base) != demo.digest_file(other)
    assert report["profile"] == "couch"
    assert report["watch_from"] == report["start_date"]
    assert _workout_days(other, "running") == []
    assert _workout_days(base, "running")
    c = sqlite3.connect(f"file:{other}?mode=ro", uri=True)
    try:
        assert c.execute("SELECT value FROM vault_meta WHERE key = "
                         "'demo_profile'").fetchone()[0] == "couch"
    finally:
        c.close()


def test_callable_knobs_and_plan_resolve_per_day(tmp_path):
    """A runner who stops: the plan and a numeric knob both change over time."""
    path = tmp_path / "stops.db"
    report = demo.build_demo_vault(
        path, days=42, seed=11,
        profile=demo.DemoProfile(
            name="stops", watch_from=0, skip_probability=0.0,
            weekly_plan=lambda i, d: (
                {1: "running", 3: "running", 5: "running"} if i < 21 else
                {1: "rest", 3: "walking"}),
            rhr_base=lambda i, d: 52.0 if i < 21 else 66.0))
    start = date.fromisoformat(report["start_date"])
    cutoff = (start + timedelta(days=21)).isoformat()
    runs = _workout_days(path, "running")
    assert runs and all(day < cutoff for day in runs)
    assert any(day >= cutoff for day in _workout_days(path, "walking"))
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        def mean_rhr(lo, hi):
            return c.execute(
                "SELECT AVG(last) FROM daily_metrics WHERE metric = "
                "'resting_heart_rate' AND date >= ? AND date < ?",
                (lo, hi)).fetchone()[0]
        assert mean_rhr(cutoff, "9999") - mean_rhr("0000", cutoff) > 8.0
    finally:
        c.close()


def test_a_rest_day_is_the_same_as_an_absent_weekday(tmp_path):
    """"rest" must draw nothing, or a plan with explicit rest days would shift
    the whole stream against one that simply leaves them out."""
    plan = {1: "running", 3: "running"}
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    demo.build_demo_vault(a, days=30, seed=8,
                          profile=demo.DemoProfile(name="x", weekly_plan=plan))
    demo.build_demo_vault(b, days=30, seed=8, profile=demo.DemoProfile(
        name="x", weekly_plan={**plan, 0: "rest", 2: "rest", 4: "rest"}))
    assert demo.digest_file(a) == demo.digest_file(b)


def test_profile_rejects_an_unknown_activity_or_weekday():
    with pytest.raises(ValueError):
        demo.DemoProfile(weekly_plan={1: "skydiving"})
    with pytest.raises(ValueError):
        demo.DemoProfile(weekly_plan={9: "running"})
