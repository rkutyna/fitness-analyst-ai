"""subjective.log: partial upsert + records/daily_metrics mirror."""
import pytest

from health_advisor import db as dbmod
from health_advisor import subjective as S


@pytest.fixture()
def wconn(tmp_path):
    conn = dbmod.connect(tmp_path / "t.db")
    dbmod.init_db(conn)
    yield conn
    conn.close()


DAY = "2026-07-16"


def test_log_writes_row_and_mirrors_to_daily_metrics(wconn):
    row = S.log(wconn, DAY, stress=3, caffeine_drinks=2, notes="long workday")
    assert row["stress"] == 3 and row["caffeine_drinks"] == 2
    assert row["notes"] == "long workday"

    dm = {r["metric"]: r for r in wconn.execute(
        "SELECT metric, sum, avg, unit FROM daily_metrics WHERE date = ?", (DAY,))}
    assert dm["subjective_stress"]["avg"] == 3
    assert dm["subjective_stress"]["unit"] == "score"
    assert dm["caffeine_drinks"]["sum"] == 2

    rec = wconn.execute(
        "SELECT source, origin, local_date FROM records WHERE metric = 'subjective_stress'"
    ).fetchone()
    assert (rec["source"], rec["origin"], rec["local_date"]) == ("checkin", "checkin", DAY)


def test_partial_update_preserves_other_fields(wconn):
    S.log(wconn, DAY, stress=3, notes="first")
    row = S.log(wconn, DAY, alcohol_drinks=1)
    assert row["stress"] == 3            # untouched
    assert row["alcohol_drinks"] == 1
    assert row["notes"] == "first"       # untouched


def test_new_checkin_fields_round_trip_and_keep_soreness_series(wconn):
    columns = {r["name"] for r in wconn.execute("PRAGMA table_info(subjective)")}
    assert columns >= {"food_note", "jog_niggle", "jog_niggle_detail", "talk_test"}

    row = S.log(wconn, DAY, soreness=4, food_note="late takeout",
                jog_niggle="y", jog_niggle_detail="left calf; no change",
                talk_test="comfortable")

    assert row["food_note"] == "late takeout"
    assert row["jog_niggle"] == "y"
    assert row["jog_niggle_detail"] == "left calf; no change"
    assert row["talk_test"] == "comfortable"
    assert row["soreness"] == 4
    soreness_record = wconn.execute(
        "SELECT value FROM records WHERE metric = 'subjective_soreness'"
    ).fetchone()
    assert soreness_record["value"] == 4


def test_new_checkin_fields_support_partial_upsert(wconn):
    S.log(wconn, DAY, food_note="breakfast was light", talk_test="not_sure")
    row = S.log(wconn, DAY, jog_niggle="n")
    assert row["food_note"] == "breakfast was light"
    assert row["talk_test"] == "not_sure"
    assert row["jog_niggle"] == "n"


def test_talk_test_domain_is_rejected(wconn):
    with pytest.raises(ValueError, match="talk_test"):
        S.log(wconn, DAY, talk_test="maybe")


def test_jog_niggle_domain_is_rejected(wconn):
    with pytest.raises(ValueError, match="jog_niggle"):
        S.log(wconn, DAY, jog_niggle="sometimes")


def test_relog_replaces_not_blends(wconn):
    S.log(wconn, DAY, caffeine_drinks=2)
    S.log(wconn, DAY, caffeine_drinks=4)
    dm = wconn.execute(
        "SELECT count, sum FROM daily_metrics WHERE metric = 'caffeine_drinks' AND date = ?",
        (DAY,)).fetchone()
    assert (dm["count"], dm["sum"]) == (1, 4)   # one record, replaced value


def test_validation_errors(wconn):
    with pytest.raises(ValueError):
        S.log(wconn, DAY, stress=7)              # rating out of range
    with pytest.raises(ValueError):
        S.log(wconn, DAY, caffeine_drinks=-1)    # negative count
    with pytest.raises(ValueError):
        S.log(wconn, "07/16/2026", stress=3)     # bad day format
    with pytest.raises(ValueError):
        S.log(wconn, DAY)                        # nothing to store


def test_get_range_ascending(wconn):
    S.log(wconn, "2026-07-15", stress=2)
    S.log(wconn, "2026-07-16", stress=4)
    days = [r["date"] for r in S.get_range(wconn, "2026-07-14", "2026-07-16")]
    assert days == ["2026-07-15", "2026-07-16"]
    assert S.get_day(wconn, "2026-07-14") is None
