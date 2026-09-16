"""The checker: a stated answer stands only if its passage, its quote and its value all hold."""

from __future__ import annotations

import pytest

from techval.rag.passages import Passage
from techval.rag.readings import Reading
from techval.rag.verify import check_value, locate, numbers, verify

CUSTOMERS = (
    "As of December 31, 2025, we had approximately 32,700 customers spanning organizations "
    "of a broad range of sizes and industries, compared to approximately 30,000 as of December 31, 2024."
)
NRR = "As of December 31, 2025, our trailing 12-month dollar-based net retention rate was about 120%."
THRESHOLD = "We monitor our number of customers with ARR of $100,000 or more, and believe it is useful to investors"
PAYO = "will be converted into the right to receive $7.40 in cash, without interest (the “Merger Consideration”)."
WORK = (
    "the right to receive 0.0776 shares of Salesforce common stock and the right to receive "
    "$26.79 in cash, without interest"
)
AGREED = "On June 29, 2026, Iridium entered into an Agreement and Plan of Merger"


def _passage(text: str) -> Passage:
    return Passage("A:0", "A", None, 0, len(text), text)


def _stated(**fields) -> Reading:
    base = dict(task_id="T", reader="claude", status="stated", passage_id="A:0", reason="read")
    base.update(fields)
    return Reading(**base)


def test_a_hedged_count_is_kept_and_marked_hedged():
    r = verify(_stated(value=32700, unit="count", quote=CUSTOMERS), [_passage(CUSTOMERS)], "count")
    assert r.status == "stated" and r.hedged


def test_a_count_the_quote_does_not_state_is_refused():
    r = verify(_stated(value=32000, unit="count", quote=CUSTOMERS), [_passage(CUSTOMERS)], "count")
    assert r.status == "refused" and "does not state 32000" in r.reason


def test_a_rate_is_read_in_percentage_points():
    r = verify(_stated(value=120, unit="percent", quote=NRR), [_passage(NRR)], "percent")
    assert r.status == "stated" and r.hedged


def test_a_small_dollar_figure_with_no_scale_word_is_refused():
    r = verify(_stated(value=100000, unit="usd", quote=THRESHOLD), [_passage(THRESHOLD)], "usd")
    assert r.status == "refused" and "threshold" in r.reason


def test_a_quote_not_in_its_passage_is_refused():
    quote = "the right to receive $7.40 in cash"
    r = verify(_stated(value=7.4, unit="usd_per_share", quote=quote), [_passage(WORK)], "usd_per_share")
    assert r.status == "refused" and "not in the passage" in r.reason


def test_quote_marks_and_spacing_are_folded_and_nothing_else():
    straight = 'will be converted into the right to receive $7.40 in cash,  without interest (the "Merger Consideration").'
    r = verify(_stated(value=7.4, unit="usd_per_share", quote=straight), [_passage(PAYO)], "usd_per_share")
    assert r.status == "stated"
    altered = straight.replace("7.40", "7.50")
    r = verify(_stated(value=7.5, unit="usd_per_share", quote=altered), [_passage(PAYO)], "usd_per_share")
    assert r.status == "refused"


def test_a_passage_the_reader_was_not_given_is_refused():
    r = verify(
        _stated(value=7.4, unit="usd_per_share", quote=PAYO, passage_id="B:9"),
        [_passage(PAYO)],
        "usd_per_share",
    )
    assert r.status == "refused" and "not given" in r.reason


def test_an_answer_in_another_unit_is_refused():
    r = verify(_stated(value=7.4, unit="usd", quote=PAYO), [_passage(PAYO)], "usd_per_share")
    assert r.status == "refused" and "usd_per_share" in r.reason


def test_a_mixed_consideration_needs_both_legs_in_the_quote():
    assert verify(_stated(text_value="mixed", unit="text", quote=WORK), [_passage(WORK)], "text").status == "stated"
    assert verify(_stated(text_value="mixed", unit="text", quote=PAYO), [_passage(PAYO)], "text").status == "refused"
    assert verify(_stated(text_value="cash", unit="text", quote=PAYO), [_passage(PAYO)], "text").status == "stated"


def test_an_exchange_ratio_is_not_read_from_a_dollar_figure():
    assert verify(_stated(value=0.0776, unit="ratio", quote=WORK), [_passage(WORK)], "ratio").status == "stated"
    assert verify(_stated(value=26.79, unit="ratio", quote=WORK), [_passage(WORK)], "ratio").status == "refused"


def test_a_date_must_be_the_one_the_quote_states():
    p = [_passage(AGREED)]
    assert verify(_stated(text_value="2026-06-29", unit="date", quote=AGREED), p, "date").status == "stated"
    assert verify(_stated(text_value="2026-06-30", unit="date", quote=AGREED), p, "date").status == "refused"
    assert verify(_stated(text_value="June 29", unit="date", quote=AGREED), p, "date").status == "refused"


def test_a_name_must_appear_in_the_quote():
    quote = "entered into an Agreement and Plan of Merger with Rocket Lab Corporation, a Delaware corporation"
    p = [_passage(quote)]
    assert verify(_stated(text_value="Rocket Lab Corporation", unit="text", quote=quote), p, "text").status == "stated"
    assert verify(_stated(text_value="Rocket Lab USA", unit="text", quote=quote), p, "text").status == "refused"


def test_answers_that_are_not_stated_pass_through_untouched():
    reading = Reading("T", "claude", "ambiguous", reason="two ratios, set at closing")
    assert verify(reading, [], "ratio") is reading


def test_numbers_carry_their_scale_and_resolution():
    [n] = numbers("ARR of $1.2 billion")
    assert n.value == pytest.approx(1.2e9) and n.resolution == pytest.approx(1e8)
    assert n.dollars and n.scaled and not n.percent
    [m] = numbers("rate was 120%")
    assert m.value == 120 and m.percent and m.resolution == 1


def test_a_stated_number_needs_a_value():
    assert not check_value("count", None, None, CUSTOMERS).ok


def test_locate_finds_a_quote_across_spacing_and_marks():
    text = "the right to receive $7.40 in cash, without\ninterest (the “Merger Consideration”)."
    spans = locate('receive $7.40 in cash, without interest (the "Merger Consideration")', text, base=100)
    assert spans == [(100 + text.index("receive"), 100 + len(text) - 1)]
    assert locate("receive $7.50", text) == []
