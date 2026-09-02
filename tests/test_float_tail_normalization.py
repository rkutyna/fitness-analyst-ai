"""Float round-off is normalized at the HealthKit ingest boundary."""

from health_advisor import db
from health_advisor import hk_parse


DEVICE = {"id": "watch-a", "name": "iPhone", "model": "iPhone17,1"}
REVISION = {"source_name": "Apple Watch", "bundle_id": "com.apple.Health"}


def _sample(uuid, value):
    return {
        "kind": "quantity",
        "hk_uuid": uuid,
        "type_identifier": "HKQuantityTypeIdentifierRestingHeartRate",
        "start": "2026-08-29T07:00:00-04:00",
        "end": "2026-08-29T07:00:01-04:00",
        "value": value,
        "unit": "count/min",
        "source_revision": REVISION,
    }


def _payload(*samples):
    return {
        "protocol_version": 1,
        "device": DEVICE,
        "app_version": "1.0",
        "batch_id": "float-tail-test",
        "batch_sequence": 1,
        "sent_at": "2026-08-29T12:00:00Z",
        "anchors": [],
        "samples": list(samples),
        "deletions": [],
        "workouts": [],
    }


def test_single_resting_hr_tail_is_snapped_before_daily_metrics(conn):
    """A receiver value already in count/min is a source tail, not an average.

    The second sample proves the policy is not "make resting HR integral": a
    genuine fractional sample remains fractional and its daily mean remains so.
    """
    parsed = hk_parse.parse_payload(_payload(
        _sample("resting-tail", 61.99999999999999),
        _sample("resting-half", 62.5),
    ))

    assert [row["value"] for row in parsed["records"]] == [62.0, 62.5]
    db.insert_records(conn, parsed["records"])
    db.recompute_daily_metrics(conn, pairs=parsed["pairs"])

    daily = conn.execute(
        "SELECT count, avg, last FROM daily_metrics "
        "WHERE metric = 'resting_heart_rate' AND date = '2026-08-29'"
    ).fetchone()
    assert tuple(daily) == (2, 62.25, 62.5)


def test_percent_conversion_tail_is_snapped_after_scaling():
    """The same boundary also removes the one-ULP tail from 0.28 * 100."""
    sample = {
        **_sample("double-support", 0.28),
        "type_identifier": "HKQuantityTypeIdentifierWalkingDoubleSupportPercentage",
        "unit": "%",
    }
    parsed = hk_parse.parse_payload(_payload(sample))
    assert parsed["records"][0]["value"] == 28.0


def test_float_tail_policy_does_not_round_real_fraction_or_near_zero():
    from health_advisor import normalize as nz

    assert nz.canonical_value("resting_heart_rate", 62.0000000001) == 62.0000000001
    assert nz.canonical_value("distance_cycling", 6e-13) == 6e-13
