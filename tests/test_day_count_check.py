"""#16 — a stated day count against the answer's own itemisation.

Every fixture here is INVENTED. The two real failures came from one person's
real health data in a private repo; this engine repo is public, so the shapes
are reproduced with invented months, invented dates and invented figures. What
is preserved is the structure of the defect, which is what the check reads.

The negatives carry most of the weight. The decision that scoped this check was
made on measured exposure — roughly 140 of 219 word-number hits over 158
narrations were fixed comparison windows and similar coaching idioms — so a
check that refuses ordinary prose is worse than the hole it closes. Each
negative below names the idiom it pins.
"""
import pytest

from health_advisor import agents as G


# --------------------------------------------------------------------------
# Positives: the stated count exceeds what the same sentence itemises.
# --------------------------------------------------------------------------

def test_the_observed_failure_shape_is_flagged():
    """The real defect, with invented numbers: count says four, list says two,
    and the sentence's own tail says the rest of the week had none."""
    text = ("This week you logged 77.5 ride minutes. That came from four "
            "cycling days: 30.0 minutes on Mar 3, 47.5 minutes on Mar 6, and "
            "the rest of the week had no cycling recorded.")

    result = G.day_count_check(text)

    assert not result.ok
    assert not result
    finding = result.findings[0]
    assert finding["stated"] == 4
    assert finding["itemised"] == 2
    assert finding["delta"] == 2
    assert "four cycling days" in finding["span"]
    assert text[finding["start"]:finding["end"]] == finding["span"]


def test_overstated_by_exactly_one_is_flagged():
    """The measured shape: overstated by one, refuted by the itemisation."""
    result = G.day_count_check(
        "That came from three running days: 30.0 minutes on Mar 3, 20.0 "
        "minutes on Mar 5, and nothing else was logged.")

    assert not result
    assert (result.findings[0]["stated"], result.findings[0]["itemised"]) == (3, 2)


def test_a_digit_count_and_weekday_labels_are_flagged():
    result = G.day_count_check("You swam on 3 days last week: Tuesday and Friday.")

    assert not result
    assert result.findings[0]["itemised"] == 2


def test_iso_dated_entries_are_counted():
    result = G.day_count_check(
        "You had three strength days: 2031-03-02 and 2031-03-04.")

    assert not result
    assert result.findings[0]["labels"] == ["('iso', '2031-03-02')",
                                            "('iso', '2031-03-04')"]


def test_two_entries_on_one_date_are_one_day():
    """A day count is a count of DAYS; two sessions on one date are one day."""
    result = G.day_count_check(
        "That came from two lifting days: 20.0 minutes on Mar 3 and 15.0 "
        "minutes more on Mar 3.")

    assert not result
    assert result.findings[0]["itemised"] == 1


def test_a_bare_continuation_day_list_is_counted():
    """"Mar 3, 5 and 7" is three days, and the count says four."""
    result = G.day_count_check("You ran on four days: Mar 3, 5 and 7.")

    assert not result
    assert result.findings[0]["itemised"] == 3


def test_a_closing_clause_does_not_raise_the_ceiling():
    """The tail that makes the observed failure self-refuting: it says there
    were no more days, so it is not an undated entry that could be one."""
    result = G.day_count_check(
        "That came from three running days: 30.0 minutes on Mar 3, 20.0 "
        "minutes on Mar 5, and the rest of the week had no running recorded.")

    assert not result
    assert result.findings[0]["undated"] == 0


def test_a_parenthesised_list_is_an_itemisation():
    result = G.day_count_check("You lifted on three days (Mar 3, Mar 5).")

    assert not result


def test_a_dashed_list_is_an_itemisation():
    result = G.day_count_check(
        "That came from four cycling days — 30.0 minutes on Mar 3 and "
        "47.5 minutes on Mar 6.")

    assert not result


def test_a_bulleted_itemisation_is_counted():
    """`_PROSE_DATE_RE`'s range alternative accepts a newline, so a markdown
    bullet reads as `Mar 3 - 20`; the scan is line-local so the list counts."""
    result = G.day_count_check(
        "That came from three running days:\n"
        "- 30.0 minutes on Mar 3\n"
        "- 20.0 minutes on Mar 5\n"
        "Nothing else was logged.")

    assert not result
    assert result.findings[0]["itemised"] == 2


# --------------------------------------------------------------------------
# Negatives that must pass: correct answers.
# --------------------------------------------------------------------------

def test_an_exact_itemisation_passes():
    result = G.day_count_check(
        "That came from two cycling days: 30.0 minutes on Mar 3 and 47.5 "
        "minutes on Mar 6.")

    assert result.ok
    assert result.findings == ()


def test_an_exact_bulleted_itemisation_passes():
    result = G.day_count_check(
        "That came from three running days:\n"
        "- 30.0 minutes on Mar 3\n"
        "- 20.0 minutes on Mar 5\n"
        "- 10.0 minutes on Mar 8\n"
        "Nice consistency.")

    assert result.ok


def test_a_stated_count_below_the_itemisation_is_not_flagged():
    """One-directional on purpose: a list may legitimately run past the window
    the count describes, and only overstatement was ever observed."""
    result = G.day_count_check(
        "That came from two cycling days: 30.0 minutes on Mar 3, 47.5 minutes "
        "on Mar 6, and 12.0 minutes on Mar 8.")

    assert result.ok


def test_the_continuation_day_list_shape_passes():
    """#187's shape: several days under one month inside a parenthesis."""
    result = G.day_count_check(
        "The other four days (Mar 4, 5, 7, 9) recorded walking only.")

    assert result.ok


# --------------------------------------------------------------------------
# Negatives that must pass: the coaching idioms the decision protects.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,risk", [
    ("You got a couple of runs in this week, which is a couple more than "
     "last week.",
     "a couple of runs — no count word, no day noun"),
    ("A night or two of short sleep will not undo the base you built.",
     "a night or two — the noun is nights and there is no itemisation"),
    ("Mar 3 was one of your easy days: 20.0 minutes at a conversational pace.",
     "one of your easy days — a partitive with no count"),
    ("That was one of your three easy days: 20.0 minutes on Mar 3.",
     "one of your three easy days — a partitive that DOES carry a count"),
    ("Your mileage has climbed over the last four weeks: Mar 3 was the "
     "biggest single day.",
     "the last four weeks — a fixed comparison window, the largest measured "
     "share of the exposure"),
    ("Recovery has been steady over the last four days: Mar 6 was the "
     "standout.",
     "the last four DAYS — the same window idiom with the day noun"),
    ("You trained on about three days: Mar 6 was the hardest.",
     "an approximate count is not a count of a listed set"),
    ("Aim for at least two easy days: Mar 6 would be a good one.",
     "a bound, and prospective advice rather than a report"),
    ("You repeated the session three days later - Mar 6 felt easier.",
     "'three days later' — a bridge that a permissive introducer would read "
     "as a one-item itemisation"),
    ("You ran five days this week. Mar 3 was the hardest.",
     "a partial mention with no introducer — the biggest refusal risk"),
    ("You ran on three days, Mar 3 and Mar 5.",
     "a bare comma is not an enumeration introducer"),
    ("You ran on four days: Mar 3 and Mar 6 among others.",
     "an itemisation that declares itself partial"),
    ("You ran on four days: Mar 3-6.",
     "a range denotes an unknown number of days"),
    ("You ran on four days: Mar 3 through Mar 6.",
     "a spelled range denotes an unknown number of days"),
    ("You ran on three days: Mar 3, 5 and 7 were all easy.",
     "an unterminated bare day list — undercounting it would invent a "
     "contradiction"),
    ("That came from three running days: 30 minutes on Mar 3, 20 minutes on "
     "Mar 5.",
     "an integer quantity after a date is indistinguishable from a "
     "continuation day, so the check stands down"),
    ("That came from three runs: 30.0 minutes on Mar 3.",
     "the scope is a DAY count, not a session count"),
    ("You slept under 6 h on three nights: Mar 3 and Mar 5.",
     "the scope is a DAY count, not a night count"),
    ("That came from two swim days: 30.0 minutes on Mar 3 at 6:15 AM and "
     "25.0 minutes on Mar 6.",
     "a clock time is not a day label"),
    ("You have two hard days ahead: keep Mar 3 easy.",
     "prospective advice — one date mentioned, nothing itemised"),
    ("Three running days made the total: 30.0 minutes on Mar 3, 20.0 on "
     "Mar 5.",
     "the introducer must be adjacent to the count; a verb phrase between "
     "them means the list is not necessarily that count's itemisation"),
    ("You have three key days this week: Tuesday's intervals, Thursday's "
     "tempo, and the long run.",
     "an itemisation need not date every entry — three entries, two dated"),
    ("You hit one out of three easy days: 20.0 minutes on Mar 3.",
     "'one out of three' — a partitive written with 'out of'"),
    ("You made two out of four planned days: Mar 3 and Mar 5.",
     "a count of PLANNED days against what was achieved — the mismatch is "
     "the point of the sentence, not a defect"),
    ("Three quality days are on the plan: Mar 3 and Mar 5 are pencilled in.",
     "a prospective count whose colon follows a verb phrase, not the count — "
     "the introducer must be adjacent or the list is not that count's own"),
    ("Three days stand out: your best sleep, your best HRV, and Mar 3.",
     "undated entries that could be days raise the ceiling"),
    ("That came from three running days: 30.0 minutes on Mar 3, 20.0 minutes "
     "on Mar 5, which was your longest.",
     "an explanatory tail is an entry we cannot rule out as a day, so the "
     "check stands down rather than guessing"),
    ("The block ran a full 7 days: Mar 3 and Mar 6 were the hard ones.",
     "'a full N days' — the length of a window, not a count of a listed set; "
     "the same class as 'all 7 days', which the window guard already declined"),
    ("That was a whole 7 days: Mar 3 and Mar 6 stood out.",
     "'a whole N days' — the same totality idiom with a different determiner"),
    ("Recovery took an entire 4 days: Mar 3 and Mar 5 were the flat ones.",
     "'an entire N days' — the totality idiom in its 'an' form"),
    ("Your load rose steadily — you logged 7 running days — and Mar 3 was "
     "the standout.",
     "a count inside a PAIRED dash aside: the trailing dash closes the aside "
     "rather than introducing a list, and what follows it belongs to the "
     "sentence the aside interrupted"),
    ("Week 7 had three quality days: Mar 3, Mar 5 and Mar 7.",
     "an authored Week label beside a correct itemisation"),
    ("You held Zone 2 for two easy days: Mar 3 and Mar 5.",
     "a domain name-term beside a correct itemisation"),
])
def test_coaching_idioms_are_not_flagged(text, risk):
    result = G.day_count_check(text)

    assert result.ok, f"{risk}: {result.as_dict()}"
    assert result.findings == ()


# --------------------------------------------------------------------------
# The one measured false positive, and the two guards that close it.
#
# Found 2026-09-04 sweeping the check over the private narration corpus: 3
# flags in 189 rendered narrations, of which this was the only wrong one. The
# answer below is correct in every respect — it says five running sessions and
# then itemises exactly five dates — but the check reported stated=7,
# itemised=1, undated=2 against the parenthetical week-length aside.
#
# The text is REPRODUCED IN SHAPE ONLY. Every month, day and figure here is
# invented; the real answer was one person's health data in a private repo.
# What is preserved is what the check reads: a paired-dash aside carrying "a
# full N days", a following clause carrying a different count and a dated
# entry, and a closing itemisation that has nothing to do with either.
# --------------------------------------------------------------------------

_CORPUS_FALSE_POSITIVE = (
    "Over the last two weeks (Mar 2 through Mar 15) you did 5 running "
    "sessions. In jogging-volume terms, the week of Mar 2 — a full 7 days — "
    "totaled 41.0 jogging minutes, and the current week of Mar 9 has 55.0 "
    "jogging minutes across 4 days so far (it's still in progress, so that's "
    "a partial week). Your running sessions were: Mar 3 (2.10 mi), Mar 5 "
    "(3.40 mi), Mar 8 (2.60 mi), Mar 11 (3.90 mi), and Mar 14 (2.20 mi)."
)


def test_the_measured_corpus_false_positive_is_not_flagged():
    result = G.day_count_check(_CORPUS_FALSE_POSITIVE)

    assert result.ok, result.as_dict()
    assert result.findings == ()


def test_the_window_length_aside_is_declined_by_the_window_guard():
    """"a full 7 days" is the LENGTH of the week the aside describes. It is the
    same idiom as "all 7 days", which the window guard already declined."""
    result = G.day_count_check(
        "The week of Mar 2 — a full 7 days — totaled 41.0 jogging minutes, "
        "and Mar 9 was the biggest single day.")

    assert result.ok
    assert any("comparison window or bound" in note for note in result.notes)


def test_a_dash_aside_closes_rather_than_introduces():
    """The structural half, pinned WITHOUT the "a full" wording so it cannot
    pass on the window guard: a count between two spaced dashes introduces
    nothing, because what follows the closing dash belongs to the sentence the
    aside interrupted."""
    result = G.day_count_check(
        "Your load rose steadily — you logged 7 running days — and Mar 3 was "
        "the standout.")

    assert result.ok, result.as_dict()
    assert any("a dash aside closes rather than introduces" in note
               for note in result.notes), result.notes


def test_an_opening_dash_still_introduces_after_the_aside_guard():
    """The guard is parity, not "dashes are out". A count with no unmatched
    dash before it still introduces its list — including one that follows a
    COMPLETE aside earlier in the same sentence."""
    unpaired = G.day_count_check(
        "That came from four cycling days — 30.0 minutes on Mar 3 and 47.5 "
        "minutes on Mar 6.")
    after_closed_aside = G.day_count_check(
        "Your volume rose — nicely, too — on four cycling days — 30.0 "
        "minutes on Mar 3 and 47.5 minutes on Mar 6.")

    assert not unpaired
    assert not after_closed_aside, after_closed_aside.as_dict()


def test_markdown_bullets_do_not_flip_the_aside_parity():
    """A bullet is a spaced dash too. With an ODD number of them earlier in the
    segment, a naive parity scan reads the next dash as closing an aside and
    stands a genuine hit down — so a dash that begins a line is not counted."""
    result = G.day_count_check(
        "That came from three running days:\n"
        "- 30.0 minutes on Mar 3\n"
        "- 20.0 minutes on Mar 5\n"
        "- 10.0 minutes on Mar 8\n"
        "You also had four cycling days — Mar 4 and Mar 7.")

    assert not result, result.as_dict()
    assert result.findings[0]["stated"] == 4
    assert result.findings[0]["itemised"] == 2


def test_a_colon_inside_an_aside_still_introduces():
    """Only the DASH form of the introducer is halved by the parity guard: a
    colon is not one end of a pair the count is sitting between."""
    result = G.day_count_check(
        "Your volume rose — you ran on four days: Mar 3 and Mar 6 — which is "
        "a good sign.")

    assert not result, result.as_dict()
    assert result.findings[0]["stated"] == 4


@pytest.mark.parametrize("text", ["", "   ", None, 0])
def test_empty_or_non_text_input_is_not_flagged(text):
    result = G.day_count_check(text)

    assert result.ok
    assert result.findings == ()


# --------------------------------------------------------------------------
# The contract the refusal-cost sweep depends on.
# --------------------------------------------------------------------------

def test_the_result_is_auditable_by_hand():
    text = ("That came from three running days: 30.0 minutes on Mar 3, 20.0 "
            "minutes on Mar 5, and nothing else was logged.")

    payload = G.day_count_check(text).as_dict()

    assert payload["ok"] is False
    assert payload["reason"]
    assert payload["findings"][0]["stated"] == 3
    assert payload["findings"][0]["itemised"] == 2
    assert payload["findings"][0]["undated"] == 0
    assert payload["findings"][0]["delta"] == 1
    assert payload["findings"][0]["span"].startswith("three running days:")
    assert payload["evidence"] == [payload["findings"][0]["span"]]


def test_a_near_miss_records_why_it_was_left_alone():
    """The notes are what make a corpus sweep readable: every candidate that
    was seen and declined says which guard declined it."""
    result = G.day_count_check("You ran five days this week. Mar 3 was the hardest.")

    assert result.ok
    assert result.notes and "no itemisation introduced" in result.notes[0]


def test_every_count_phrase_in_a_long_answer_is_evaluated():
    text = ("That came from two cycling days: 30.0 minutes on Mar 3 and 47.5 "
            "minutes on Mar 6. Strength was three days: Mar 4 and Mar 7.")

    result = G.day_count_check(text)

    assert len(result.findings) == 1
    assert result.findings[0]["stated"] == 3


def test_an_invisible_character_inside_a_date_does_not_hide_it():
    """The same invisible-character sanitisation every other gate here uses."""
    result = G.day_count_check(
        "That came from two cycling days: 30.0 minutes on Mar​ 3 and "
        "47.5 minutes on Mar 6.")

    assert result.ok


# --------------------------------------------------------------------------
# The constraint the decision put on this work.
# --------------------------------------------------------------------------

def test_no_word_number_tokenizer_was_added_to_the_grounding_gate():
    """The decision ruled out a word-number tokenizer and any widening of
    `numeric_tokens`. Word-numbers must stay invisible to the figure scan: the
    count lookup exists only on the left of the day-count comparison."""
    assert G._numeric_tokens("You ran on three days this week.") == []
    assert G._numeric_tokens("Three easy days and 51.2 minutes.") == ["51.2"]
    ok, unsupported = G.grounding_check("You ran on three days.", {"a": 1})
    assert ok and unsupported == []


def test_the_check_needs_no_ledger_payload_or_claim():
    """It is a self-consistency check: the only argument is the answer text."""
    import inspect

    signature = inspect.signature(G.day_count_check)
    assert list(signature.parameters) == ["text"]


# --------------------------------------------------------------------------
# The two measured false positives of 2026-09-04, and the guards that close
# them. Both are ORDINARY COACHING PROSE that the check refused: one states a
# correct count and then names a qualified subset of it, the other states a
# PLANNED count and names two example days under it.
#
# Each negative is paired with a near-twin positive that differs only in the
# thing the guard reads. That pairing is the test: without it, either guard
# could pass by switching the check off for the whole shape.
#
# Every month, day and figure here is invented; the real answers were one
# person's health data in a private repo.
# --------------------------------------------------------------------------

def test_a_trailing_subset_qualifier_is_not_flagged():
    """"were the hard ones" predicates the listed days as a distinguished
    SUBSET of the four. The count is correct and the colon introduces a
    selection, not an enumeration of all four days."""
    result = G.day_count_check(
        "You trained on four days last week: Tuesday and Friday were the "
        "hard ones.")

    assert result.ok, result.as_dict()
    assert result.findings == ()
    assert any("qualified subset" in note for note in result.notes), result.notes


def test_the_same_sentence_without_the_qualifier_still_flags():
    """The near twin of the fixture above, differing only by the trailing
    qualifier. Drop "were the hard ones" and the list IS the itemisation of
    the four, so the contradiction is real and must still be caught."""
    result = G.day_count_check(
        "You trained on four days last week: Tuesday and Friday.")

    assert not result, result.as_dict()
    assert (result.findings[0]["stated"], result.findings[0]["itemised"]) == (4, 2)
    assert result.findings[0]["delta"] == 2


def test_a_qualifier_that_does_not_close_the_list_still_flags():
    """The exemption is a TRAILING qualifier, anchored at the end of the span.
    Mid-list, "was the hard one" qualifies ONE entry rather than the list, so
    an unanchored keyword search here would stand a real contradiction down."""
    result = G.day_count_check(
        "You trained on four days last week: Tuesday was the hard one and "
        "Friday was steady.")

    assert not result, result.as_dict()
    assert result.findings[0]["stated"] == 4
    assert result.findings[0]["itemised"] == 2


@pytest.mark.parametrize("text", [
    "Your plan has three quality days: Tuesday and Friday.",
    "The plan calls for three quality days: Tuesday and Friday.",
    "Your program includes three quality days: Tuesday and Friday.",
    "The schedule prescribes three quality days: Tuesday and Friday.",
    "You have three quality days scheduled: Tuesday and Friday.",
])
def test_a_plan_stating_the_count_is_not_flagged(text):
    """A PLANNED count is a count of intent, and the dates under it are
    examples of the prescription rather than its full itemisation.
    `_PLANNED_GAP` reads an adjective between the count and the noun; these
    shapes carry the intent in the verb phrase to the LEFT instead."""
    result = G.day_count_check(text)

    assert result.ok, result.as_dict()
    assert result.findings == ()


def test_the_same_count_reported_rather_than_planned_still_flags():
    """The near twin of the fixtures above: identical count phrase, identical
    itemisation, but reported rather than prescribed. The plan guard must not
    have switched the check off for "three quality days" as a shape."""
    result = G.day_count_check(
        "You trained on three quality days: Tuesday and Friday.")

    assert not result, result.as_dict()
    assert (result.findings[0]["stated"], result.findings[0]["itemised"]) == (3, 2)
    assert result.findings[0]["delta"] == 1


def test_a_plan_word_that_is_not_a_plan_statement_still_flags():
    """The plan guard is a closed subject+verb pair immediately before the
    count, not "the word plan appears nearby"."""
    result = G.day_count_check(
        "You changed the plan after three easy days: Mar 3 and Mar 5.")

    assert not result, result.as_dict()
    assert result.findings[0]["delta"] == 1


def test_both_ground_truth_positives_still_flag_by_exactly_one():
    """The only real evidence this check works: the two published narrations
    that overstated a day count by one and were refuted by their own
    itemisation, reproduced in shape with invented figures. Any exemption
    added to this check must leave both of them flagging."""
    first = G.day_count_check(
        "That came from three running days: 30.0 minutes on Mar 3, 20.0 "
        "minutes on Mar 5, and nothing else was logged.")
    second = G.day_count_check(
        "That came from three running days: 30.0 minutes on Mar 3, 20.0 "
        "minutes on Mar 5, and the rest of the week had no running recorded.")

    for result in (first, second):
        assert not result, result.as_dict()
        finding = result.findings[0]
        assert (finding["stated"], finding["itemised"], finding["undated"]) == (3, 2, 0)
        assert finding["delta"] == 1


def test_a_training_block_is_not_a_plan_statement():
    """"block" is kept out of the plan-lead subjects on purpose: unlike plan /
    programme / schedule / template it as often names the training that was
    actually done, so exempting it would hide a real self-contradiction."""
    result = G.day_count_check(
        "That block had three hard days: Mar 3 and Mar 5.")

    assert not result, result.as_dict()
    assert result.findings[0]["delta"] == 1
