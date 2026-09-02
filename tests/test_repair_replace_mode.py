"""The replace path in scripts/repair_missing_days.py DELETES stored history.

The mid-July HAE outage left days holding a few hours of samples (2026-07-15 held
00:06-04:57 and 18.2 kcal against HealthKit's 828.8). Those days cannot be merged:
measured live on 2026-07-16 and -20, ZERO of 2,614 export records shared a
dedupe_key with the receiver rows already stored, because the receiver samples far
finer than the export. Adding them would double-count. So the day is rebuilt.

The invariant that makes that safe is: after a replace, the day equals the EXPORT
total — not export + what was there. And a day the export cannot supply is refused
rather than emptied, because deleting data and putting nothing back is the one
mistake here that a backup is the only recovery from.
"""
from datetime import datetime, timedelta

import pytest

from health_advisor import db as dbmod
from scripts import repair_missing_days as R

AE = "HKQuantityTypeIdentifierActiveEnergyBurned"


def _xml(tmp_path, samples):
    """samples: list of (start_datetime, value). Minimal Apple-export XML."""
    rows = []
    for ts, v in samples:
        end = ts + timedelta(minutes=1)
        f = "%Y-%m-%d %H:%M:%S +0000"
        rows.append(
            f'<Record type="{AE}" sourceName="Watch" unit="kcal" '
            f'value="{v}" startDate="{ts.strftime(f)}" endDate="{end.strftime(f)}"/>'
        )
    p = tmp_path / "export.xml"
    p.write_text("<HealthData>" + "".join(rows) + "</HealthData>")
    return str(p)


def _seed_partial_day(conn, day, hours, value):
    """Stand in for the receiver's partial ingest: real rows, finer windows than
    the export's, so their dedupe_keys cannot collide with it."""
    rows = []
    for h in hours:
        for minute in (0, 30):
            start = f"{day}T{h:02d}:{minute:02d}:00Z"
            end = f"{day}T{h:02d}:{minute:02d}:30Z"
            rows.append(dict(
                metric="active_energy", value=value, unit="kcal",
                start_utc=start, end_utc=end, start_local=f"{day} {h:02d}:{minute:02d}:00",
                local_date=day, source="Watch", origin="receiver",
                dedupe_key=dbmod.record_key("active_energy", start, end, value, "kcal", "Watch"),
            ))
    dbmod.insert_records(conn, rows)
    dbmod.recompute_daily_metrics(conn, pairs=[("active_energy", day)])
    conn.commit()


@pytest.fixture
def dbfile(tmp_path):
    p = tmp_path / "t.db"
    conn = dbmod.connect(p)
    dbmod.init_db(conn)
    conn.commit()
    yield conn, str(p)
    conn.close()


def test_replace_lands_on_the_export_total_not_the_sum_of_both(dbfile, tmp_path):
    conn, path = dbfile
    day = "2026-07-15"
    _seed_partial_day(conn, day, hours=[0, 1, 2], value=3.0)   # 6 rows x 3.0 = 18.0
    assert conn.execute("SELECT sum FROM daily_metrics WHERE date=?", (day,)).fetchone()[0] == 18.0

    # The export's view of the same day: whole-day coverage, coarser windows.
    d0 = datetime(2026, 7, 15, 0, 0, 0)
    xml = _xml(tmp_path, [(d0 + timedelta(hours=i), 40.0) for i in range(20)])  # 800.0

    out = R.run(None, ["active_energy"], [day], db_path=path, apply=True,
                replace=True, xml_path=xml)

    got = conn.execute("SELECT sum FROM daily_metrics WHERE date=?", (day,)).fetchone()[0]
    assert got == pytest.approx(800.0), "replace must land on the export total"
    assert got != pytest.approx(818.0), "18.0 of stale partial data survived the delete"
    assert out["deleted"] == 6
    assert out["added"] == 20


def test_replace_refuses_to_empty_a_day_the_export_cannot_supply(dbfile, tmp_path):
    conn, path = dbfile
    day = "2026-07-15"
    _seed_partial_day(conn, day, hours=[0, 1, 2], value=3.0)
    # Export holds a DIFFERENT day only.
    xml = _xml(tmp_path, [(datetime(2026, 7, 19, 8, 0, 0), 40.0)])

    out = R.run(None, ["active_energy"], [day], db_path=path, apply=True,
                replace=True, xml_path=xml)

    got = conn.execute("SELECT sum FROM daily_metrics WHERE date=?", (day,)).fetchone()[0]
    assert got == 18.0, "a day the export cannot supply must be left alone, not emptied"
    assert out["deleted"] == 0
    assert conn.execute("SELECT COUNT(*) FROM records WHERE local_date=?", (day,)).fetchone()[0] == 6


def test_fill_mode_still_refuses_a_day_that_has_records(dbfile, tmp_path):
    """The conservative default must not inherit replace's willingness to delete."""
    conn, path = dbfile
    day = "2026-07-15"
    _seed_partial_day(conn, day, hours=[0], value=3.0)
    d0 = datetime(2026, 7, 15, 0, 0, 0)
    xml = _xml(tmp_path, [(d0 + timedelta(hours=i), 40.0) for i in range(20)])

    out = R.run(None, ["active_energy"], [day], db_path=path, apply=True, xml_path=xml)

    assert out["skipped_occupied"] == 1
    assert out["added"] == 0
    assert conn.execute("SELECT sum FROM daily_metrics WHERE date=?", (day,)).fetchone()[0] == 6.0
