import pytest

from health_advisor import agents as G


def test_extract_json_from_fenced_block():
    txt = "sure!\n```json\n{\"a\": 1, \"b\": [2,3]}\n```\ndone"
    assert G.extract_json(txt) == {"a": 1, "b": [2, 3]}


def test_extract_json_bare_object():
    assert G.extract_json('noise {"x": 5} tail') == {"x": 5}


def test_claim_channel_recovers_invisible_character_in_json_number():
    raw = (
        '{"text":"Readiness is stable.","claims":'
        '[{"source":{"sequence": \u200d10, '
        '"path":"$.result.workouts[12].distance_mi"}}]}'
    )

    prose, claims = G.split_claim_channel(raw)

    assert prose == "Readiness is stable."
    assert claims == [{"source": {"sequence": 10,
                                    "path": "$.result.workouts[12].distance_mi"}}]


def test_claim_channel_preserves_visible_separator_in_text():
    raw = (
        '{"text":"The range is 29-30 minutes.","claims":'
        '[{"source":{"sequence": \u200d10}}]}'
    )

    prose, claims = G.split_claim_channel(raw)

    assert claims == [{"source": {"sequence": 10}}]
    assert prose == "The range is 29-30 minutes."
    assert "2930" not in prose


def test_claim_channel_recovery_preserves_text_verbatim_without_nfkc_or_nfd():
    raw = (
        '{"text":"café: 29-30 minutes.","claims":'
        '[{"source":{"sequence": \u200d10}}]}'
    )

    prose, _ = G.split_claim_channel(raw)

    assert prose == "café: 29-30 minutes."


def test_grounding_check_passes_when_numbers_present():
    briefing = {"readiness": {"score": 72}, "movers": [{"pct": 18.0}]}
    ok, bad = G.grounding_check("Readiness is 72 and steps up 18%.", briefing)
    assert ok and bad == []


def test_grounding_check_flags_fabricated_number():
    briefing = {"readiness": {"score": 72}}
    ok, bad = G.grounding_check("Your HRV is 999 ms today.", briefing)
    assert not ok and "999" in bad


@pytest.mark.parametrize("prose", [
    "Your VO2max improved this week.",
    "Your VO2 improved this week.",
    "Your 5k pace improved.",
    "Your 10k pace improved.",
    "Zone2 work is going well.",
    "Zone 2 work is going well.",
    "Zone 5 work is going well.",
])
def test_grounding_check_ignores_digits_in_tracked_domain_names(prose):
    ok, bad = G.grounding_check(prose, {})
    assert ok and bad == []


def test_grounding_check_still_checks_a_bare_digit():
    ok, bad = G.grounding_check("You trained 2 days.", {})
    assert not ok and bad == ["2"]


def test_grounding_check_still_checks_an_attached_measurement():
    ok, bad = G.grounding_check("Your HRV was 52ms.", {})
    assert not ok and bad == ["52"]


def test_grounding_check_supported_number_and_unrelated_number_controls():
    ok, bad = G.grounding_check("You ran 7 miles.", {"jog_minutes": 31.0})
    assert not ok and bad == ["7"]

    ok, bad = G.grounding_check("Resting HR was 52.", {"resting_hr": 52.0})
    assert ok and bad == []


def test_grounding_check_numeric_token_shapes_are_unchanged():
    prose = "Values were -13,900.25 and 7%, then 0.5."
    assert G._NUM_RE.findall(prose) == ["-13,900.25", "7", "0.5"]


def test_grounding_check_generated_unsupported_numbers_stay_rejected():
    """Generated quantitative prose must not become passable through masking."""
    import random

    rng = random.Random(39)
    for _ in range(400):
        value = rng.uniform(-100_000, 100_000)
        token = f"{value:,.3f}" if rng.randrange(2) else f"{value:.3f}%"
        ok, bad = G.grounding_check(f"The metric changed by {token}.", {})
        assert not ok and bad, (token, bad)


def test_render_fallback_uses_talking_points():
    b = {"as_of": "2026-05-30",
         "talking_points": [{"seed": "recovery readiness 72/100 (green)"},
                            {"seed": "training load ACWR 1.1 (sweet-spot)"}],
         "suggestions": [{"text": "Good day to push."}]}
    text = G.render_fallback(b)
    assert "readiness 72/100" in text and "Good day to push" in text


def test_grounding_check_ignores_iso_dates():
    briefing = {"readiness": {"score": 72}}
    ok, bad = G.grounding_check("On 2026-05-28 readiness was 72.", briefing)
    assert ok and bad == []


def test_grounding_check_strips_year_month_and_full_iso_dates():
    ok, bad = G.grounding_check(
        "The 2026-04 review covered 2026-04-18.", {})
    assert ok and bad == []


def test_grounding_check_flags_near_miss_of_small_integer():
    briefing = {"readiness": {"score": 100}}
    ok, bad = G.grounding_check("Readiness is 99 today", briefing)
    assert not ok and "99" in bad


def test_grounding_check_allows_rounding_of_large_value():
    briefing = {"movers": [{"pct": 156.32}]}
    ok, bad = G.grounding_check("Active energy up 156%.", briefing)
    assert ok and bad == []


def test_grounding_strips_prose_dates_like_iso_dates():
    """Month-name dates and ranges ("July 6–13", "Jul 6, 2026") are calendar
    references, not quantitative claims — same treatment as ISO dates."""
    briefing = {"x": 42}
    ok, bad = G.grounding_check(
        "Coverage was complete July 6–13 (checked Jul 9, 2026): 42 metrics.",
        briefing)
    assert ok, bad


def test_a_month_year_strips_wholly_and_leaves_no_stray_token():
    """Measured live 2026-08-27: the day form's \\d{1,2} half-matched the year,
    consuming "July 20" and leaving "26" for the gate to demand a claim for.
    That artifact alone produced 4 of the 12-call ask battery's failures while
    every real figure in those answers verified."""
    ok, bad = G.grounding_check(
        "Your longest run last month (July 2026) was 3.04 miles on "
        "July 26, 2026", {"distance": 3.04})
    assert ok, bad

    ok, bad = G.grounding_check("Aug 2026 and July 2026 are covered.", {})
    assert ok and bad == []


def test_the_existing_prose_date_forms_still_strip_wholly():
    """The month-year alternative is tried first; the day forms it precedes
    must be unaffected by it."""
    for text in ("July 26, 2026", "Jul 9, 2026", "July 6–13", "Aug 7–21",
                 "Sept. 3", "Feb 28, 2026"):
        ok, bad = G.grounding_check(text, {})
        assert ok and bad == [], text


def test_clock_times_with_a_meridiem_marker_strip_like_dates():
    """A 12-hour time of day is a point-in-time reference, the clock analogue
    of a calendar date. Measured live: "from about 11:41 PM to 7:10 AM"
    tokenized as 11, 41, 7, 10 — four demanded claims for one bedtime."""
    ok, bad = G.grounding_check(
        "You slept from about 11:41 PM to 7:10 AM, with 0 min latency.",
        {"latency": 0})
    assert ok, bad

    for text in ("11:41 PM", "7:10AM", "11:41 p.m.", "7:10 am", "7:10 A.M."):
        ok, bad = G.grounding_check(text, {})
        assert ok and bad == [], text


def test_a_bare_clock_form_without_a_meridiem_marker_still_tokenizes():
    """The AM/PM marker is mandatory and is the whole reason the clock strip is
    safe: a bare mm:ss is a pace or a duration, which is a measurement the gate
    must keep checking."""
    ok, bad = G.grounding_check("You held a 5:30 pace.", {})
    assert not ok and bad == ["5", "30"]

    ok, bad = G.grounding_check("Your best split was 7:10 over the mile.", {})
    assert not ok and bad == ["7", "10"]


def test_the_calendar_and_clock_strips_do_not_swallow_measurements():
    """Every digit the strips do not cover is still demanded, including the
    bare year that no pattern here removes."""
    for text, expected in (("You ran 26 miles.", ["26"]),
                           ("You slept 11 hours.", ["11"]),
                           ("Your mileage in 2026 is up.", ["2026"]),
                           ("July 2026 held 41 sessions.", ["41"]),
                           ("At 7:10 am you weighed 41 kg.", ["41"])):
        ok, bad = G.grounding_check(text, {})
        assert not ok and bad == expected, text


def test_substance_gate_rejects_the_real_vacuous_report():
    text = ("Your heart rate variability is currently at 46.34 ms, tracking your "
            "present load state. The latest tool reading confirms this value as it "
            "stands today. We encourage you to continue monitoring this metric for "
            "recovery insights.")

    result = G.substance_check(text)

    assert not result.ok
    assert "comparative" in result.reason or "temporal" in result.reason
    assert result.signals == ()


def test_substance_gate_ignores_the_operational_confidence_footer():
    result = G.substance_check(
        "A current HRV reading is 46.34 ms. — confidence 100/100 · 0 claim(s) dropped"
    )

    assert not result


def test_substance_gate_accepts_the_real_useful_report():
    text = ("The clearest finding is a non-finding: more walking/running did not "
            "produce a verified sleep benefit. Mean distance was 3.94 mi versus "
            "3.01 mi, while deep sleep was 47.01 min versus 35.39 min... Over "
            "twelve weeks, associations were rho -0.028 (44 pairs) and -0.155 "
            "(45 pairs); neither passed FDR.")

    result = G.substance_check(text)

    assert result.ok
    assert "comparison" in result.signals
    assert "window" in result.signals


def test_substance_gate_rejects_a_score_without_context():
    result = G.substance_check("The recovery readiness metric returned a value of 69 out of 100.")

    assert not result
    assert result.reason


def test_substance_gate_accepts_a_real_time_window():
    result = G.substance_check(
        "Recent 7-day average morning wakefulness increased to 56.29 min "
        "versus baseline 21.97 min."
    )

    assert result.ok
    assert set(result.signals) == {"comparison", "window"}


def test_a_digit_followed_by_letters_is_still_a_number():
    """The trailing side of a numeric token is deliberately unguarded.

    Guarding it requires enumerating the unit suffixes that still count as
    measurements, and units are open-ended. The first attempt at this listed
    bpm/kcal/ms/lbs/mi/ft and silently stopped checking every unit below, so
    "You lost 52kg this week." passed against an empty briefing — a fabricated
    figure walking through the gate whose whole job is to stop them. Names are a
    closed set and go in _NAME_TERM_RE; units do not get an allowlist.
    """
    import re
    unguarded = re.compile(r"-?\d[\d,]*\.?\d*")     # the pre-fix tokenizer
    for text in ("52kg", "52km", "52cm", "52mm", "52oz", "52kj", "52rpm",
                 "52kph", "52bps", "52kcal", "52bpm", "52ms", "52lbs", "52mi",
                 "52ft", "52.5kg", "1st", "2nd", "-13,900.25"):
        assert G._NUM_RE.findall(text) == unguarded.findall(text), text

    ok, bad = G.grounding_check("You lost 52kg this week.", {})
    assert not ok and bad == ["52"]


def test_name_terms_are_stripped_not_tolerated():
    """A name's digit is removed before tokenizing; the same digit elsewhere in
    the same sentence is still checked. Otherwise allowlisting "Zone 2" would
    quietly mean "stop checking the digit 2"."""
    ok, bad = G.grounding_check("Zone 2 work rose 2 minutes.", {})
    assert not ok and bad == ["2"]


def test_week_labels_are_stripped_not_tolerated():
    """An authored Week label is removed, but the same digit elsewhere still
    needs a verified claim."""
    ok, bad = G.grounding_check("Week 2 work rose 2 minutes.", {})
    assert not ok and bad == ["2"]


def test_a_determiner_week_phrase_does_not_swallow_a_real_number():
    """"Week 7" is an authored plan label; "this week 3" is a measurement with a
    noun in front of it. Stripping the label must not stop the gate checking the
    measurement — the Zone 2 rule, applied to the ordinal vocabulary."""
    ok, bad = G.grounding_check("This week 3 sessions are logged.", {})
    assert not ok and bad == ["3"]
    ok, bad = G.grounding_check("Last week 12 km were run.", {})
    assert not ok and bad == ["12"]


def test_date_stripping_survives_an_invisible_between_month_and_day():
    """F-86: `_PROSE_DATE_RE` needs `\\s`, and a format character is not whitespace.

    Both cases are real coach output whose claims all verified and which were
    refused anyway, for citing a date.
    """
    joiner = G.strip_dates_and_names(
        "The most recent 4-week block (Jul ‍27 through Aug ‍23) "
        "totaled 200.4 jog minutes.")
    assert G._NUM_RE.findall(joiner) == ["4", "200.4"]

    rlm = G.strip_dates_and_names(
        "Last week (Monday Aug ‏10 through‏ Sunday Aug ‏16) "
        "you slept an average of 461.57 minutes per night.")
    assert G._NUM_RE.findall(rlm) == ["461.57"]


def test_date_stripping_never_joins_a_visible_separator():
    """A visible separator must survive: removing one FABRICATES a number."""
    cleaned = G.strip_dates_and_names("The range is 29-30 minutes.")
    assert "2930" not in cleaned
    assert "29" in cleaned and "30" in cleaned


def test_a_day_list_under_one_month_strips_completely():
    """#187 verbatim. Only the first day was stripped, so the rest were graded
    as unsupported quantitative claims and the answer was refused for listing
    dates."""
    cleaned = G.strip_dates_and_names(
        "The other four days (Aug 19, 21, 22, 23) recorded walking only.")
    assert G._NUM_RE.findall(cleaned) == []
    assert "21" not in cleaned and "22" not in cleaned and "23" not in cleaned

    ok, bad = G.grounding_check(
        "The other four days (Aug 19, 21, 22, 23) recorded walking only.", {})
    assert (ok, bad) == (True, [])


def test_a_day_list_does_not_swallow_the_figure_that_follows_it():
    """The whole risk of #187: the widened pattern fails OPEN. A figure sitting
    immediately after the list must still be graded."""
    cleaned = G.strip_dates_and_names(
        "(Aug 19, 21, 22) totalled 44.0 minutes")
    assert G._NUM_RE.findall(cleaned) == ["44.0"]

    ok, bad = G.grounding_check("(Aug 19, 21, 22) totalled 44.0 minutes", {})
    assert not ok and bad == ["44.0"]


def test_an_interior_list_element_is_not_stripped_when_the_run_ends_in_a_word():
    """The #187 review finding, and the sharpest form of the fail-open.

    The first patch guarded each element with its own trailing-word lookahead,
    which protected only the LAST element: interior days were committed one at
    a time, so "Since Aug 19, 8, 9, 12 were your resting HR readings" left only
    ['12'] and two real resting-HR readings silently stopped being graded.

    The run is all-or-nothing. It ends in a word here, so every element stays.
    """
    for prose, expected in (
            ("Since Aug 19, 8, 9, 12 were your resting HR readings",
             ["8,", "9,", "12"]),
            ("On Aug 19, 5, 10 and 15 minute intervals", ["5,", "10", "15"]),
            ("Aug 19, 3, 4 and 6 km were logged", ["3,", "4", "6"]),
            ("Aug 19, 2 and 3 hours of sleep", ["2", "3"]),
    ):
        assert G._NUM_RE.findall(
            G.strip_dates_and_names(prose)) == expected, prose

    ok, bad = G.grounding_check(
        "Since Aug 19, 8, 9, 12 were your resting HR readings.", {})
    assert not ok and bad == ["8,", "9,", "12"], bad


def test_a_run_that_ends_at_a_separator_strips_nothing():
    """Why "," and "and" are NOT terminators.

    Admitting either would end the run one element early and strip everything
    before it — the same fail-open, one comma further along. The cost is that a
    genuine day list followed by a comma clause is left graded, which is the
    old false refusal and therefore the safe direction.
    """
    assert G._NUM_RE.findall(G.strip_dates_and_names(
        "Since Aug 19, 8, 9, 44.0 were readings")) == ["8,", "9,", "44.0"]
    assert G._NUM_RE.findall(G.strip_dates_and_names(
        "Aug 19, 21, 22, 23, which were walking.")) == ["21,", "22,", "23,"]


def test_a_number_after_a_month_dated_clause_is_still_a_claim():
    """#187 Done-when 2, four ways the boundary is drawn.

    A decimal point, a thousands separator, a value outside 1..31, and a
    trailing unit word each disqualify a token from being read as a further
    day. Each of these silently stopping the gate would be worse than the
    false refusal the day-list tail removes.
    """
    for prose, expected in (
            ("On Aug 19, 44.0 jog minutes", ["44.0"]),
            ("On Aug 19, 22.5 jog minutes were logged.", ["22.5"]),
            ("On Aug 19, 44 jog minutes were logged.", ["44"]),
            ("On Aug 19, 22 jog minutes were logged.", ["22"]),
            ("On Aug 19, 12,500 steps.", ["12,500"]),
            ("On Aug 19, 44, 52, and 61 minutes were logged.",
             ["44,", "52,", "61"]),
    ):
        assert G._NUM_RE.findall(
            G.strip_dates_and_names(prose)) == expected, prose


def test_a_day_range_still_strips_after_the_day_list_change():
    """Ranges worked before #187 and must keep working, both dash forms."""
    for prose in ("Aug 19–23 was the block.", "Aug 19-23 was the block."):
        assert G._NUM_RE.findall(G.strip_dates_and_names(prose)) == [], prose
    assert G._NUM_RE.findall(
        G.strip_dates_and_names("The block ran Aug 19-23, 25, 26.")) == []


def test_a_year_is_not_eaten_as_the_head_of_a_day_list():
    """"Sunday Aug 16, 2026" is the one real shape that LOOKS like a day list
    (#187's amended banner found it as the corpus's only apparent hit). The
    year must be consumed as a year, and the figure after it must survive."""
    cleaned = G.strip_dates_and_names(
        "Sunday Aug 16, 2026, and 44.0 minutes followed.")
    assert G._NUM_RE.findall(cleaned) == ["44.0"]
    assert "2026" not in cleaned

    ok, bad = G.grounding_check(
        "Monday Aug 10 through Sunday Aug 16, 2026 you slept 461.57 minutes.",
        {"sleep": {"mean_minutes": 461.57}})
    assert (ok, bad) == (True, [])


def test_prose_date_strip_fixture_cases():
    """The committed corpus (#187 amended criterion 5).

    The battery captures this pattern was measured against were lost (#172),
    so the day-list examples live in the repo. `kind` is documentation of
    intent; the assertion is the exact surviving token list either way.
    """
    import json
    import pathlib
    fixture = json.loads(
        (pathlib.Path(__file__).parent / "fixtures"
         / "prose_date_strip_cases.json").read_text())
    cases = fixture["cases"]
    assert len(cases) >= 26, "cases may be added, never quietly dropped"
    for case in cases:
        assert case["kind"] in ("strip", "keep"), case["id"]
        got = G._NUM_RE.findall(G.strip_dates_and_names(case["prose"]))
        assert got == case["tokens"], f"{case['id']}: {got}"


def test_a_subjective_scale_denominator_is_not_a_claim():
    """#217, live on the first production /v1/ask after 24e7ff8 deployed.

    Every substantive figure verified and the answer was withheld in full,
    because the `5` in "3 out of 5" is the constant range of the check-in
    scale. `api_daily.figure(..., scale=5)` renders it into a display STRING,
    and `_numbers_in` walks numeric leaves only, so nothing can ever publish
    it — the sentence was unverifiable by construction.
    """
    ok, bad = G.grounding_check(
        "You rated sleep 3 out of 5 and energy 4 out of 5.", {"q": 3, "e": 4})
    assert (ok, bad) == (True, [])

    # The slash rendering the tool surface itself emits, and the readiness
    # composite's `scale=100` — the only other constant `figure()` can produce.
    assert G.grounding_check("Sleep quality 3/5, energy 4/5.",
                             {"q": 3, "e": 4}) == (True, [])
    assert G.grounding_check("Your readiness score was 71 out of 100.",
                             {"score": 71}) == (True, [])
    assert G.grounding_check("Readiness 71/100.", {"score": 71}) == (True, [])


def test_a_published_five_still_grades_and_a_neighbour_is_still_caught():
    """The denominator strip must not become a hole for a bare figure.

    A `5` outside the denominator slot is an ordinary measurement: published,
    it verifies; unpublished, it is still reported. Both halves matter — the
    first alone would pass with the gate deleted.
    """
    assert G.grounding_check("You slept 5 h.", {"sleep": 5}) == (True, [])

    ok, bad = G.grounding_check("You slept 9 h.", {"sleep": 5})
    assert not ok and bad == ["9"]

    # Same sentence shape as the fix's target, with one fabricated figure
    # alongside the rating: the rating passes, the invention does not.
    ok, bad = G.grounding_check(
        "You rated sleep 3 out of 5 and slept 9 h.", {"q": 3})
    assert not ok and bad == ["9"]


def test_the_denominator_strip_survives_a_colon_rendering():
    """A bulleted `Quality: 3 out of 5` is the shape the live coach actually
    writes, and a colon between the anchor and the numerator must not defeat
    the strip. Regression: the first version of `_SCALE_DENOM_RE` excluded `:`
    from the gap, so every colon-separated rating still died on its denominator
    while the prose form passed."""
    claims = {"quality": 3, "energy": 4, "readiness": 71}
    for prose in ("Quality: 3 out of 5.",
                  "- Sleep quality: 3 out of 5",
                  "Energy: 4 out of 5",
                  "Readiness: 71 out of 100"):
        assert G.grounding_check(prose, claims) == (True, []), prose
    # ...and a colon does NOT license a count ratio, because the anchor
    # vocabulary is still what gates it.
    ok, bad = G.grounding_check("Sessions completed: 3 out of 5.", claims)
    assert not ok and bad == ["5."]


def test_the_denominator_strip_needs_a_rating_anchor():
    """"3 out of 5 planned sessions" is a COUNT ratio, and 5 is a real claim.

    This is the "Zone 2 must not mean stop checking the digit 2" rule applied
    to a denominator: the pattern is anchored on the check-in vocabulary, so a
    ratio with no rating word in front of it keeps grading its denominator.
    """
    ok, bad = G.grounding_check(
        "You completed 3 out of 5 planned sessions.", {"done": 3})
    assert not ok and bad == ["5"]

    ok, bad = G.grounding_check(
        "Energy was low, so you ran 4 out of 5 planned workouts.", {"ran": 4})
    assert not ok and bad == ["5"]


def test_the_denominator_strip_only_accepts_the_two_scale_constants():
    """`figure(..., scale=N)` has exactly two call sites: 5 and 100.

    Any other denominator is a magnitude, not a constant of the display
    contract, and must still demand a claim. The digit guards matter as much
    as the enumeration: "out of 500" must not be read as "out of 5".
    """
    for prose, expected in (
        ("Your energy rating was 4 out of 50.", "50."),
        ("You rated soreness 2 out of 500.", "500."),
        ("Your energy rating was 4 out of 5.5.", "5.5"),
        ("Sleep quality 3/5,000.", "5,000."),
    ):  # the trailing dot in a sentence-final token is cosmetic (#217)
        ok, bad = G.grounding_check(prose, {"v": 4, "w": 2, "x": 3})
        assert not ok and expected in bad, (prose, bad)


def test_the_denominator_strip_keeps_the_numerator_and_the_separator():
    """The `29-30` -> `2930` rule, applied to this pattern.

    Only the denominator and its separator are blanked, and they are replaced
    with a SPACE. The numerator survives as its own token; nothing merges.
    """
    cleaned = G.strip_dates_and_names("You rated sleep 3 out of 5 today.")
    assert G._NUM_RE.findall(cleaned) == ["3"]
    assert "35" not in cleaned

    cleaned = G.strip_dates_and_names("Sleep quality 3/5, energy 4/5.")
    assert G._NUM_RE.findall(cleaned) == ["3", "4"]

    # A date that happens to sit beside a rating is not re-joined either.
    cleaned = G.strip_dates_and_names("Rated 3/5 on 8/5/2026.")
    assert "35" not in cleaned and "3" in cleaned


def test_grounding_check_accepts_a_dated_claim_carrying_an_invisible():
    """The end-to-end shape: a fully-grounded answer that cites a date."""
    briefing = {"sleep": {"mean_minutes": 461.57}}
    ok, bad = G.grounding_check(
        "Last week (Monday Aug ‏10 through‏ Sunday Aug ‏16) you "
        "slept an average of 461.57 minutes per night.", briefing)
    assert (ok, bad) == (True, [])


def test_no_module_open_codes_the_date_strip_pipeline():
    """F-86 was live on FIVE call sites and fixing three left it live on two.

    `strip_dates_and_names` is the one definition; a sixth open-coded copy
    fails here rather than drifting silently (the #53 one-tokenizer rule).
    `coach_brief` strips only ISO dates for activity-word matching, which has
    no month/day seam to break, so it is deliberately not in scope.
    """
    import pathlib
    root = pathlib.Path(G.__file__).parent
    offenders = [
        f"{path.name}:{n}"
        for path in sorted(root.glob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "_PROSE_DATE_RE.sub" in line and path.name != "agents.py"
    ]
    assert offenders == [], (
        "open-coded date stripping bypasses invisible-character removal: "
        + ", ".join(offenders))


def test_coach_grounding_does_not_flag_a_date_carrying_an_invisible():
    """The path that actually produces the `unsupported` list (dv:333).

    Real output: 'your longest run was on July <U+FE0F>26, at 3.04 miles'.
    The variation selector defeated the date strip, so `26` was graded as an
    unclaimed figure and a correct answer was refused.
    """
    from health_advisor import deepdive_verify as dv
    ledger = [{"sequence": 1, "tool": "list_workouts",
               "result": {"workouts": [{"distance_mi": 3.04}]}}]
    claims = [{"metric": "distance_mi", "field": "distance_mi", "value": 3.04,
               "source": {"sequence": 1,
                          "path": "$.result.workouts[0].distance_mi"}}]
    grounded, bad = dv._coach_grounding(
        "Your longest run was on July ️26, at 3.04 miles.",
        claims, payload=ledger)
    assert "26," not in bad and "26" not in bad, bad
