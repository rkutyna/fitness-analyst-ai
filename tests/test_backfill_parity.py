"""Backfill follows the same rules as the live receiver path.

Two rules the receiver applies and the backfill used not to: sleep-session
attribution (a stage sample belongs to the day its session ENDS), and real unit
conversion through `normalize.unit_converter` rather than a relabel.
"""
import sqlite3

import pytest

from health_advisor import backfill, db, derive
from health_advisor import normalize as nz


def _sleep(value, start, end):
    return (f'<Record type="HKCategoryTypeIdentifierSleepAnalysis" '
            f'value="HKCategoryValueSleepAnalysis{value}" sourceName="Watch" '
            f'startDate="{start} -0400" endDate="{end} -0400"/>')


# One night that crosses midnight. The first block ends BEFORE midnight, so its
# own end date is 07-01; the session it belongs to ends at 07:00 on 07-02.
NIGHT = [
    _sleep("AsleepCore", "2026-07-01 22:30:00", "2026-07-01 23:50:00"),
    _sleep("AsleepDeep", "2026-07-01 23:50:00", "2026-07-02 02:00:00"),
    _sleep("AsleepREM", "2026-07-02 02:00:00", "2026-07-02 07:00:00"),
]


def _export(tmp_path, records):
    xml = tmp_path / "export.xml"
    xml.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<HealthData>\n'
                   + "\n".join(records) + "\n</HealthData>\n")
    return xml


def _weight(value, unit):
    return (f'<Record type="HKQuantityTypeIdentifierBodyMass" value="{value}" '
            f'unit="{unit}" sourceName="Scale" '
            f'startDate="2026-07-02 08:00:00 -0400" endDate="2026-07-02 08:00:00 -0400"/>')


def _rows(path, sql, params=()):
    c = sqlite3.connect(path)
    try:
        return c.execute(sql, params).fetchall()
    finally:
        c.close()


# ----------------------------------------------------------------- sleep ----
def test_backfill_attributes_a_midnight_crossing_session_to_its_end_date(
        tmp_path, vault_path):
    backfill.run(xml_path=str(_export(tmp_path, NIGHT)), db_path=str(vault_path))
    rows = _rows(vault_path,
                 "SELECT metric, local_date FROM records "
                 "WHERE metric IN ('sleep_asleep','sleep_awake','sleep_in_bed') "
                 "ORDER BY start_utc")
    assert rows and {d for _, d in rows} == {"2026-07-02"}
    # The aggregate follows: the whole 8.5 h of sleep sits on the wake day and
    # nothing is left behind on the day before.
    daily = dict(_rows(vault_path,
                       "SELECT date, sum FROM daily_metrics "
                       "WHERE metric = 'sleep_asleep'"))
    assert daily == {"2026-07-02": pytest.approx(510.0)}


def test_second_reattribute_after_backfill_moves_nothing(tmp_path, vault_path):
    backfill.run(xml_path=str(_export(tmp_path, NIGHT)), db_path=str(vault_path))
    conn = db.connect(str(vault_path))
    try:
        assert derive.reattribute_sleep(
            conn, "2026-06-30", "2026-07-03", apply=True) == []
    finally:
        conn.close()
    # ...and a second backfill of the same export changes nothing either.
    summary = backfill.run(xml_path=str(tmp_path / "export.xml"),
                           db_path=str(vault_path))
    assert summary["records_added"] == 0
    assert summary["sleep_records_reattributed"] == 0


def test_backfill_keeps_the_per_sample_date_when_no_session_crosses(
        tmp_path, vault_path):
    recs = [_sleep("AsleepCore", "2026-07-02 01:00:00", "2026-07-02 06:00:00")]
    summary = backfill.run(xml_path=str(_export(tmp_path, recs)),
                           db_path=str(vault_path))
    assert summary["sleep_records_reattributed"] == 0
    assert {d for (d,) in _rows(vault_path,
                                "SELECT local_date FROM records")} == {"2026-07-02"}


def test_attribution_moves_only_local_date(tmp_path, vault_path):
    backfill.run(xml_path=str(_export(tmp_path, NIGHT)), db_path=str(vault_path))
    cols = ("metric, value, unit, start_utc, end_utc, start_local, source, "
            "dedupe_key")
    after = _rows(vault_path, f"SELECT {cols} FROM records ORDER BY id")
    # Rebuild the same export's rows by parsing only, with no attribution.
    expected = []
    for rec in NIGHT:
        import xml.etree.ElementTree as ET
        for r in backfill._record_rows(ET.fromstring(rec).attrib):
            expected.append((r["metric"], r["value"], r["unit"], r["start_utc"],
                             r["end_utc"], r["start_local"], r["source"],
                             r["dedupe_key"]))
    assert sorted(after) == sorted(expected)


# ----------------------------------------------------------------- units ----
def test_backfill_converts_a_kilogram_body_mass_like_the_live_path(
        tmp_path, vault_path):
    backfill.run(xml_path=str(_export(tmp_path, [_weight(80, "kg")])),
                 db_path=str(vault_path))
    (value, unit), = _rows(vault_path,
                           "SELECT value, unit FROM records WHERE metric='body_mass'")
    convert, target = nz.unit_converter("body_mass", "kg")
    assert (value, unit) == (nz.canonical_value("body_mass", convert(80.0)), target)
    assert unit == "lb" and value == pytest.approx(176.37, abs=0.005)


def _legacy_rows(attrib):
    """What the relabel-only path produced for a quantity record."""
    value = float(attrib["value"])
    metric = nz.hk_quantity_to_canonical(attrib["type"])
    value, _ = nz.apply_unit(value, attrib.get("unit"))
    return (metric, nz.canonical_value(metric, value),
            nz.canonical_unit(metric, attrib.get("unit")))


CANONICAL = [
    ("HKQuantityTypeIdentifierBodyMass", "80", "lb"),
    ("HKQuantityTypeIdentifierStepCount", "1200", "count"),
    ("HKQuantityTypeIdentifierActiveEnergyBurned", "37.5", "Cal"),   # relabel
    ("HKQuantityTypeIdentifierBasalEnergyBurned", "61", "kcal"),
    ("HKQuantityTypeIdentifierHeartRate", "62", "count/min"),
    ("HKQuantityTypeIdentifierOxygenSaturation", "0.96", "%"),
    ("HKQuantityTypeIdentifierVO2Max", "44.2", "mL/min·kg"),
]


@pytest.mark.parametrize("hk_type,value,unit", CANONICAL)
def test_canonical_unit_export_is_unchanged(hk_type, value, unit):
    attrib = {"type": hk_type, "value": value, "unit": unit, "sourceName": "Watch",
              "startDate": "2026-07-02 09:00:00 -0400",
              "endDate": "2026-07-02 09:05:00 -0400"}
    (row,) = backfill._record_rows(attrib)
    assert (row["metric"], row["value"], row["unit"]) == _legacy_rows(attrib)


def test_cal_is_still_relabelled_to_kcal():
    attrib = {"type": "HKQuantityTypeIdentifierActiveEnergyBurned", "value": "37.5",
              "unit": "Cal", "sourceName": "Watch",
              "startDate": "2026-07-02 09:00:00 -0400",
              "endDate": "2026-07-02 09:05:00 -0400"}
    (row,) = backfill._record_rows(attrib)
    assert (row["value"], row["unit"]) == (37.5, "kcal")


def test_unplaceable_unit_is_dropped_not_stored(tmp_path, vault_path):
    recs = [_weight(80, "furlongs"), _weight(81, "lb").replace("08:00", "09:00")]
    summary = backfill.run(xml_path=str(_export(tmp_path, recs)),
                           db_path=str(vault_path))
    assert _rows(vault_path, "SELECT value, unit FROM records") == [(81.0, "lb")]
    assert summary["skipped_records"] == 1
