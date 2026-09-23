"""Engine #77 items 2 and 5: `_period_label` must never give two different
periods the same text.

Two defects were fixed:

(A) `_date_range_period_label` dropped the year, so a 365-day window and a
    single day rendered indistinguishably close (e.g. "from Sat Sep 14 to
    Sun Sep 14").
(B) The weekly/daily block branches of `_period_label` said "the last N
    weeks"/"the last N days" purely from the spacing of `period_starts`,
    with no notion of "now" — so two different blocks of the same cadence
    and count produced the identical label text.

All dates below are synthetic (2029-2031), per the engine repo's no-real-data
rule.
"""
from datetime import date

from health_advisor import fact_template


# ---------------------------------------------------------------------------
# (1) Year rule for date-range labels
# ---------------------------------------------------------------------------

def test_year_added_when_endpoints_cross_a_calendar_year():
    label = fact_template._date_range_period_label(
        date(2030, 9, 14), date(2031, 9, 14))
    assert label == "from Sat Sep 14 2030 to Sun Sep 14 2031"


def test_year_added_for_a_two_year_span():
    label = fact_template._date_range_period_label(
        date(2029, 1, 1), date(2030, 12, 31))
    assert label == "from Mon Jan 1 2029 to Tue Dec 31 2030"


def test_year_added_for_an_eleven_year_span():
    label = fact_template._date_range_period_label(
        date(2020, 1, 1), date(2030, 12, 31))
    assert label == "from Wed Jan 1 2020 to Tue Dec 31 2030"


def test_year_added_when_span_crosses_new_years_eve():
    label = fact_template._date_range_period_label(
        date(2029, 12, 12), date(2030, 1, 8))
    assert label == "from Wed Dec 12 2029 to Tue Jan 8 2030"


def test_year_added_for_a_long_span_even_within_one_calendar_year():
    # 333 days, same year: the OR branch of the rule (span > 300 days) has
    # to fire on its own, independent of the cross-year branch.
    label = fact_template._date_range_period_label(
        date(2029, 1, 1), date(2029, 11, 30))
    assert label == "from Mon Jan 1 2029 to Fri Nov 30 2029"


def test_short_same_year_range_keeps_the_old_format():
    label = fact_template._date_range_period_label(
        date(2026, 8, 1), date(2026, 8, 20))
    assert label == "from Sat Aug 1 to Thu Aug 20"


def test_week_of_short_form_is_unchanged():
    label = fact_template._date_range_period_label(
        date(2026, 8, 10), date(2026, 8, 16))
    assert label == "the week of August 10"


def test_single_day_label_is_unchanged():
    assert fact_template._period_label("2026-08-07") == "Fri Aug 7"


# ---------------------------------------------------------------------------
# (2) Block labels never say "last", and name the real span
# ---------------------------------------------------------------------------

def test_weekly_block_label_names_the_span_not_last():
    recent_block_starts = [date(2029, 12, 2), date(2029, 12, 9),
                            date(2029, 12, 16), date(2029, 12, 23)]
    label = fact_template._period_starts_label(recent_block_starts)
    assert "last" not in label
    assert label == "the 4 weeks from Sun Dec 2 to Sat Dec 29"


def test_prior_weekly_block_gets_a_different_label_than_recent_block():
    recent_block_starts = [date(2029, 12, 2), date(2029, 12, 9),
                            date(2029, 12, 16), date(2029, 12, 23)]
    prior_block_starts = [date(2029, 11, 4), date(2029, 11, 11),
                           date(2029, 11, 18), date(2029, 11, 25)]

    recent_label = fact_template._period_starts_label(recent_block_starts)
    prior_label = fact_template._period_starts_label(prior_block_starts)

    assert prior_label == "the 4 weeks from Sun Nov 4 to Sat Dec 1"
    assert recent_label != prior_label


def test_daily_block_label_ends_on_the_last_start_not_last_plus_six():
    daily_starts = [date(2029, 6, 1), date(2029, 6, 2), date(2029, 6, 3)]
    label = fact_template._period_starts_label(daily_starts)
    assert "last" not in label
    assert label == "the 3 days from Fri Jun 1 to Sun Jun 3"


def test_irregular_spacing_still_returns_none():
    irregular_starts = [date(2029, 1, 1), date(2029, 1, 9), date(2029, 1, 20)]
    assert fact_template._period_starts_label(irregular_starts) is None


def test_weekly_block_label_via_period_label_dict_shape():
    period = {
        "start": "2029-12-02", "end": "2029-12-23",
        "period_starts": ["2029-12-02", "2029-12-09",
                           "2029-12-16", "2029-12-23"],
    }
    assert fact_template._period_label(period) == (
        "the 4 weeks from Sun Dec 2 to Sat Dec 29")


def test_weekly_block_label_via_period_label_list_shape():
    starts = ["2029-12-02", "2029-12-09", "2029-12-16", "2029-12-23"]
    assert fact_template._period_label(starts) == (
        "the 4 weeks from Sun Dec 2 to Sat Dec 29")


# ---------------------------------------------------------------------------
# (3) Invariant: distinct periods never collapse to the same label
# ---------------------------------------------------------------------------

def _labelled_periods():
    """Every period below is a DIFFERENT period; each must get a label, and
    no two labels may match."""
    return {
        "recent_4_week_block": fact_template._period_starts_label(
            [date(2029, 12, 2), date(2029, 12, 9),
             date(2029, 12, 16), date(2029, 12, 23)]),
        "prior_4_week_block": fact_template._period_starts_label(
            [date(2029, 11, 4), date(2029, 11, 11),
             date(2029, 11, 18), date(2029, 11, 25)]),
        "one_year_2030_to_2031": fact_template._date_range_period_label(
            date(2030, 9, 14), date(2031, 9, 14)),
        "two_year_2029_to_2030": fact_template._date_range_period_label(
            date(2029, 1, 1), date(2030, 12, 31)),
        "eleven_year_2020_to_2030": fact_template._date_range_period_label(
            date(2020, 1, 1), date(2030, 12, 31)),
        "cross_new_year_2029_to_2030": fact_template._date_range_period_label(
            date(2029, 12, 12), date(2030, 1, 8)),
        # Two 365-day windows exactly one year apart: same cadence and
        # length as "one_year_2030_to_2031" above, shifted a year earlier.
        "one_year_2029_to_2030": fact_template._date_range_period_label(
            date(2029, 9, 14), date(2030, 9, 14)),
    }


def test_all_example_periods_get_a_non_none_label():
    labels = _labelled_periods()
    for name, label in labels.items():
        assert label is not None, f"{name} unexpectedly got no label"


def test_distinct_periods_never_share_a_label():
    labels = _labelled_periods()
    names = list(labels)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert labels[left] != labels[right], (
                f"{left!r} and {right!r} both rendered as "
                f"{labels[left]!r}")


def test_the_three_long_windows_are_pairwise_distinct():
    # Restates the invariant narrowly over just the three long windows from
    # engine #77's background section, since those three alone were the
    # ones observed colliding on "one day" before the fix.
    a = fact_template._date_range_period_label(
        date(2030, 9, 14), date(2031, 9, 14))
    b = fact_template._date_range_period_label(
        date(2029, 1, 1), date(2030, 12, 31))
    c = fact_template._date_range_period_label(
        date(2020, 1, 1), date(2030, 12, 31))
    assert len({a, b, c}) == 3
