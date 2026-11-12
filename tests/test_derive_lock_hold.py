"""derive does its long reads before it takes the vault's write lock.

downstream #734: update_for_days wrote a day's rows first and only then read
365 days of heart rate (hr_max_estimate, via the HR load) and ran the impact
digest, so the write lock was held through ~0.6 s of reading per workout day.
The caller's transaction is deferred: the lock is taken by the first write.
"""
from __future__ import annotations

from health_advisor import derive


def _seed(conn, day):
    conn.execute(
        "INSERT INTO workouts (workout_type, start_utc, end_utc, local_date, "
        "duration_min, source, dedupe_key) VALUES ('walking', ?, ?, ?, 30, "
        "'x', ?)", (f"{day}T17:00:00+00:00", f"{day}T17:30:00+00:00", day, f"k|{day}"))
    for i in range(30):
        ts = f"{day}T17:{i:02d}:00+00:00"
        conn.execute(
            "INSERT INTO records (metric, value, unit, start_utc, end_utc, "
            "start_local, local_date, source, origin, dedupe_key) VALUES "
            "('heart_rate', 120, 'count/min', ?, ?, ?, ?, 'w', 'test', ?)",
            (ts, ts, f"{day}T12:{i:02d}:00", day, f"hr|{ts}"))
    conn.commit()


def test_the_long_reads_run_before_the_first_write(conn, monkeypatch):
    days = ["2026-10-20", "2026-10-21"]
    for day in days:
        _seed(conn, day)
    seen = []

    def outside_txn(name, fn):
        def wrapper(c, day):
            seen.append((name, day, c.in_transaction))
            return fn(c, day)
        return wrapper

    monkeypatch.setattr(derive, "_hr_load_for_day",
                        outside_txn("hr_load", derive._hr_load_for_day))
    monkeypatch.setattr(derive, "_dial_for_day",
                        outside_txn("dial", derive._dial_for_day))
    assert not conn.in_transaction
    written = derive.update_for_days(conn, days)
    assert written > 0 and conn.in_transaction   # it did write, deferred
    conn.commit()
    assert sorted((n, d) for n, d, _ in seen) == sorted(
        (n, d) for n in ("hr_load", "dial") for d in days)
    assert [s for s in seen if s[2]] == [], seen
