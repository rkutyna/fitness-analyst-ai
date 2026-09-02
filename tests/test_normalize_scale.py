"""Fraction→percent canonicalization for percent-ratio metrics.

HealthKit reports OxygenSaturation and AppleWalkingSteadiness as a 0–1 ratio.
Canonical scale is percent ('%'), so a 0–1 sample is scaled up by 100 at ingest.
"""
from health_advisor import normalize as nz
from health_advisor import backfill
from health_advisor import hk_parse


def test_canonical_value_scales_blood_oxygen_fraction():
    assert nz.canonical_value("blood_oxygen_saturation", 0.96) == 96.0
    assert nz.canonical_value("blood_oxygen_saturation", 0.93) == 93.0


def test_canonical_value_scales_walking_steadiness_fraction():
    assert nz.canonical_value("walking_steadiness", 0.73) == 73.0
    assert nz.canonical_value("walking_steadiness", 1.0) == 100.0


def test_canonical_value_leaves_percent_scale_untouched():
    # already 0–100 (>1): idempotent, never re-scaled
    assert nz.canonical_value("blood_oxygen_saturation", 96.0) == 96.0
    assert nz.canonical_value("walking_steadiness", 73.0) == 73.0


def test_canonical_value_ignores_non_ratio_metrics():
    # a non-percent metric is never rescaled, even when its value is <= 1
    assert nz.canonical_value("heart_rate", 0.5) == 0.5
    assert nz.canonical_value("step_count", 1.0) == 1.0


def test_healthkit_parse_scales_oxygen_fraction_to_percent():
    payload = {
        "protocol_version": 1,
        "device": {"id": "watch", "name": "Watch", "model": "test"},
        "app_version": "1", "batch_id": "oxygen", "batch_sequence": 1,
        "sent_at": "2026-06-10T13:00:00Z", "anchors": [],
        "samples": [{
            "kind": "quantity", "hk_uuid": "oxygen-1",
            "type_identifier": "HKQuantityTypeIdentifierOxygenSaturation",
            "start": "2026-06-10T09:00:00-04:00",
            "end": "2026-06-10T09:00:01-04:00", "value": 0.96,
            "unit": "%",
            "source_revision": {"source_name": "Watch", "bundle_id": "test"},
        }], "deletions": [], "workouts": [],
    }
    rec = hk_parse.parse_payload(payload)["records"][0]
    assert rec["metric"] == "blood_oxygen_saturation"
    assert rec["value"] == 96.0 and rec["unit"] == "%"


def test_backfill_scales_oxygen_fraction_to_percent():
    attrib = {"type": "HKQuantityTypeIdentifierOxygenSaturation", "value": "0.96",
              "startDate": "2026-06-10 09:00:00 -0400",
              "endDate": "2026-06-10 09:00:00 -0400", "sourceName": "Watch"}
    rows = list(backfill._record_rows(attrib))
    assert len(rows) == 1
    assert rows[0]["metric"] == "blood_oxygen_saturation"
    assert rows[0]["value"] == 96.0 and rows[0]["unit"] == "%"


def test_backfill_scales_walking_steadiness_fraction():
    attrib = {"type": "HKQuantityTypeIdentifierAppleWalkingSteadiness", "value": "0.73",
              "startDate": "2026-06-02 12:00:00 -0400",
              "endDate": "2026-06-02 12:00:00 -0400", "sourceName": "iPhone"}
    rows = list(backfill._record_rows(attrib))
    assert rows[0]["metric"] == "walking_steadiness"
    assert rows[0]["value"] == 73.0


def test_migration_rescales_fraction_records_and_reaggregates(conn):
    """The one-off repair scales pre-fix fraction records to percent, leaves real
    percent rows alone, re-aggregates daily_metrics, and is idempotent."""
    from health_advisor import db as dbmod
    from scripts.fix_percent_ratio_scale import rescale_fraction_records

    def rec(metric, value, day, origin):
        start = f"{day}T12:00:00+00:00"
        return {"metric": metric, "value": value, "unit": "%", "start_utc": start,
                "end_utc": start, "start_local": f"{day} 08:00:00", "local_date": day,
                "source": "test", "origin": origin,
                "dedupe_key": dbmod.record_key(metric, start, start, value, "%", "test")}

    dbmod.insert_records(conn, [
        rec("blood_oxygen_saturation", 0.96, "2026-06-10", "backfill"),  # fraction → ×100
        rec("blood_oxygen_saturation", 96.0, "2026-06-11", "receiver"),  # already % → untouched
        rec("walking_steadiness", 0.73, "2026-06-02", "backfill"),       # fraction → ×100
    ])
    dbmod.recompute_daily_metrics(conn, full=True)
    conn.commit()

    stats = rescale_fraction_records(conn); conn.commit()
    assert stats["records"] == 2
    vals = {(r["metric"], r["local_date"]): r["value"]
            for r in conn.execute("SELECT metric, local_date, value FROM records")}
    assert vals[("blood_oxygen_saturation", "2026-06-10")] == 96.0   # rescaled
    assert vals[("blood_oxygen_saturation", "2026-06-11")] == 96.0   # left alone (already %)
    assert vals[("walking_steadiness", "2026-06-02")] == 73.0        # rescaled
    dm = conn.execute("SELECT avg FROM daily_metrics WHERE "
                      "metric='blood_oxygen_saturation' AND date='2026-06-10'").fetchone()
    assert dm["avg"] == 96.0                                          # aggregate updated
    assert rescale_fraction_records(conn)["records"] == 0            # idempotent


def test_canonical_value_scales_double_support_fraction():
    # HK backfill stored double support as a 0-1 ratio (observed 0.167-0.402).
    # Real double support is never <= 1%.
    assert nz.canonical_value("walking_double_support_percentage", 0.284) == 28.4
    assert nz.canonical_value("walking_double_support_percentage", 28.4) == 28.4


def test_canonical_value_never_scales_walking_asymmetry():
    # Asymmetry legitimately produces sub-1% percent readings on the live path,
    # so the <=1 fraction heuristic would corrupt them - deliberately excluded.
    assert nz.canonical_value("walking_asymmetry_percentage", 0.5) == 0.5
