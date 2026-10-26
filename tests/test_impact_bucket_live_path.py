"""The public ``metrics.impact_bucket_rows`` runs the Python port, with the SQL as reference.

Two properties are pinned here:

* The public function is the port: when ``impact_buckets.impact_bucket_rows_py``
  is replaced by a function that raises, the public call raises too. A mutation
  that routes the live path back to the SQL turns this red.
* An unknown window predicate is delegated to the SQL reference
  (``metrics._impact_bucket_rows_sql``) and returns its rows through the public
  function. The delegation must not loop back into the port.
"""
from __future__ import annotations

import pytest

from health_advisor import db
from health_advisor import impact_buckets as ib
from health_advisor import metrics as mx

POST_DAY = "2026-09-02"      # after the default arbitration cutoff, 2026-08-21
LOCAL_WINDOW = "local_date BETWEEN ? AND ?"
UNKNOWN_WINDOW = "local_date >= ? AND local_date <= ?"


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "vault.db")
    db.init_db(connection)
    connection.execute(
        "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
        "start_local, local_date, source, origin, dedupe_key) "
        "VALUES ('distance_walking_running', 0.3, 'mi', ?, ?, ?, ?, "
        "'Apple Watch', 'test', 'live-path|d')",
        (f"{POST_DAY}T08:00:00Z", f"{POST_DAY}T08:00:00Z",
         f"{POST_DAY}T08:00:00Z", POST_DAY))
    connection.commit()
    yield connection
    connection.close()


def test_unknown_predicate_returns_the_sql_rows_without_recursing(conn):
    """An unknown predicate goes to the SQL reference once and comes back through the public call.

    The expected rows are the SQL reference's own output, and the one sample
    must produce exactly one bucket with its full 0.3 miles (a hand-checked
    literal, not the port's output).
    """
    public = mx.impact_bucket_rows(conn, UNKNOWN_WINDOW, (POST_DAY, POST_DAY))
    reference = mx._impact_bucket_rows_sql(conn, UNKNOWN_WINDOW,
                                           (POST_DAY, POST_DAY))
    assert public == reference
    assert len(public) == 1
    assert public[0]["mi"] == pytest.approx(0.3, abs=1e-12)


def test_public_path_really_runs_the_python_port(conn, monkeypatch):
    """If the port raises, the public call raises: the live path is the port, not the SQL."""
    def boom(*args, **kwargs):
        raise RuntimeError("port was called")

    monkeypatch.setattr(ib, "impact_bucket_rows_py", boom)
    with pytest.raises(RuntimeError, match="port was called"):
        mx.impact_bucket_rows(conn, LOCAL_WINDOW, (POST_DAY, POST_DAY))
