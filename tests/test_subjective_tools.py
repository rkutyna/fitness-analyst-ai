"""MCP log_subjective / get_subjective round-trip on a temp DB."""

from health_advisor import deepdive_verify as DV


def test_log_and_get_round_trip(tools):
    out = tools.log_subjective("2026-07-16", stress=2, soreness=4,
                             caffeine_drinks=2, notes="quads sore from hike")
    assert out["ok"] is True
    assert out["stored"]["soreness"] == 4

    got = tools.get_subjective("2026-07-15", "2026-07-17")
    assert got["count"] == 1
    assert got["days"][0]["notes"] == "quads sore from hike"


def test_log_validation_error_is_structured(tools):
    out = tools.log_subjective("2026-07-16", stress=9)
    assert out["ok"] is False
    assert "1-5" in out["error"]


def test_correction_upserts(tools):
    tools.log_subjective("2026-07-16", alcohol_drinks=1)
    out = tools.log_subjective("2026-07-16", alcohol_drinks=2)
    assert out["stored"]["alcohol_drinks"] == 2


def test_subjective_rating_fields_publish_per_field_metric_owners(tools, conn):
    tools.log_subjective("2026-07-16", stress=2, energy=4,
                         sleep_quality=4, notes="steady")
    payload = tools.get_subjective("2026-07-16", "2026-07-16")
    row = payload["days"][0]

    assert row["period"] == "2026-07-16"
    assert row["field_metrics"] == {
        "stress": "subjective_stress",
        "energy": "subjective_energy",
        "sleep_quality": "subjective_sleep_quality",
    }
    for field, metric in row["field_metrics"].items():
        claim = {"metric": metric, "period": row["period"],
                 "field": field, "value": row[field]}
        assert DV.verify_number(conn, claim, payload=payload)["ok"] is True

    scopes = DV._payload_scopes(payload)
    assert not any(entry["field"] == "notes" and entry["metric"]
                   for entry in scopes)
    assert not any(entry["field"] == "caffeine_drinks" and entry["metric"]
                   for entry in scopes)
