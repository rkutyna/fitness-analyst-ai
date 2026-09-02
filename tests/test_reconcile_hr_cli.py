"""`python -m health_advisor.db --reconcile-hr` — run the HR reconciliation
over history, not just over the days a receiver POST happened to touch.

Audit P3-5: reconcile_workout_heart_rate is only ever called from the receiver,
scoped to the (metric, date) pairs in the payload just received. A workout that
predates the receiver, or that arrived via export.zip on a day the phone did not
sync, is never revisited — so its avg/max HR stays whatever the device summary
said, or stays NULL, even when a dense heart_rate series covering it is sitting
in `records`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from health_advisor import db

REPO = Path(__file__).resolve().parent.parent


def _seed(conn, *, n_samples: int, day: str = "2020-05-05"):
    """One workout with no HR summary, covered by `n_samples` heart_rate rows."""
    start, end = f"{day}T12:00:00+00:00", f"{day}T13:00:00+00:00"
    db.insert_workouts(conn, [dict(
        workout_type="running", start_utc=start, end_utc=end, local_date=day,
        duration_min=60.0, energy_kcal=400.0, distance_mi=5.0, unit_distance="mi",
        source="Watch", route_ref=None, avg_heart_rate=None, max_heart_rate=None,
        dedupe_key=db.workout_key("running", start, end))])
    rows = []
    for i in range(n_samples):
        mm = int(i * (59 / max(n_samples - 1, 1)))
        ts = f"{day}T12:{mm:02d}:00+00:00"
        rows.append(dict(metric="heart_rate", value=140.0 + (i % 7), unit="count/min",
                         start_utc=ts, end_utc=ts, start_local=f"{day} 12:{mm:02d}:00",
                         local_date=day, source="Watch", origin="backfill",
                         dedupe_key=f"hr-{day}-{i}"))
    db.insert_records(conn, rows)
    conn.commit()


def test_reconcile_over_all_history_fills_a_pre_receiver_workout(conn):
    _seed(conn, n_samples=40)
    assert conn.execute("SELECT avg_heart_rate FROM workouts").fetchone()[0] is None
    assert db.reconcile_workout_heart_rate(conn) == 1
    assert conn.execute("SELECT avg_heart_rate FROM workouts").fetchone()[0] is not None


def test_cli_reconciles_and_reports(tmp_path):
    p = tmp_path / "health.db"
    c = db.connect(p)
    db.init_db(c)
    _seed(c, n_samples=40)
    c.close()

    r = subprocess.run([sys.executable, "-m", "health_advisor.db", "--reconcile-hr",
                        "--db", str(p)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "1" in r.stdout, r.stdout

    c = db.connect(p, read_only=True)
    assert c.execute("SELECT avg_heart_rate FROM workouts").fetchone()[0] is not None
    c.close()


def test_cli_dry_run_changes_nothing(tmp_path):
    p = tmp_path / "health.db"
    c = db.connect(p)
    db.init_db(c)
    _seed(c, n_samples=40)
    c.close()

    r = subprocess.run([sys.executable, "-m", "health_advisor.db", "--reconcile-hr",
                        "--db", str(p), "--dry-run"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "dry" in r.stdout.lower()

    c = db.connect(p, read_only=True)
    assert c.execute("SELECT avg_heart_rate FROM workouts").fetchone()[0] is None, \
        "--dry-run must not write"
    c.close()


def test_cli_can_scope_to_a_date_range(tmp_path):
    p = tmp_path / "health.db"
    c = db.connect(p)
    db.init_db(c)
    _seed(c, n_samples=40, day="2020-05-05")
    _seed(c, n_samples=40, day="2021-06-06")
    c.close()

    r = subprocess.run([sys.executable, "-m", "health_advisor.db", "--reconcile-hr",
                        "--db", str(p), "--since", "2021-01-01"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    c = db.connect(p, read_only=True)
    got = {r["local_date"]: r["avg_heart_rate"] for r in
           c.execute("SELECT local_date, avg_heart_rate FROM workouts")}
    c.close()
    assert got["2020-05-05"] is None
    assert got["2021-06-06"] is not None


def test_sparse_samples_are_still_refused(conn):
    """The gate is unchanged: a handful of samples must not overrule or invent
    a summary. Most pre-receiver workouts fail exactly here."""
    _seed(conn, n_samples=4)
    assert db.reconcile_workout_heart_rate(conn) == 0
    assert conn.execute("SELECT avg_heart_rate FROM workouts").fetchone()[0] is None
