"""The two commands that front the fitted half of the engine, end to end and offline.

A command is not tested by checking that it exits zero. It is tested by checking
that the sentence a reader will act on is the sentence the model actually
supports, and most of this file is about the cases where that sentence is bad
news.

Three things carry the weight here.

**The command must not launder a negative result.** The recorded EV/Revenue panel
scores a mean information coefficient of -0.0984 with a naive t-statistic of -2.65
and a Newey-West one of -1.62. A harness that printed the first of those and not
the second would be the single most common way a backtest lies, so the tests check
that both appear, that the words NOT SIGNIFICANT appear, and that the baseline's
own verdict of "Use the baseline." survives to the terminal.

**The counts must arrive in the honest order.** 2,604 company-dates and about 13
effective observations are the same sample described two ways, and only one of
them is the number a standard error should be read against. The test asserts the
effective count is printed above the company-date count, because a reader who sees
2,604 first has already formed a view by the time the 13 arrives.

**The point-in-time claim on ``--as-of`` must be enforced rather than asserted.**
``build_fade_panel`` runs to the present whatever date the valuation is struck at,
so a dated run would otherwise fit a growth curve on filings that postdate its own
valuation date. ``_truncate_panel`` is what stops that and the test pins both
halves of it: the observation that reaches the DCF must predate the knowledge
date, and the forward labels filed after it must be gone.

Everything runs against fixtures already committed for other modules. Nothing here
touches the network.
"""

from __future__ import annotations

import csv
import gzip
import json
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from techval.commands_forecast import _rebalance_dates, _truncate_panel, app
from techval.edgar import CompanyFacts
from techval.errors import ConfigError
from techval.ml.forecast import build_fade_panel

FIXTURES = Path(__file__).parent / "fixtures"
FADE_PANEL = FIXTURES / "fade_companyfacts.json.gz"
DDOG_FACTS = FIXTURES / "companyfacts_DDOG.json"
SIGNAL_SCORES = FIXTURES / "signals" / "ev_revenue.csv.gz"
SIGNAL_PRICES = FIXTURES / "signals" / "closes.csv.gz"

runner = CliRunner()


def flat(text: str) -> str:
    """Collapse whitespace before matching.

    Rich wraps at the console width, so a sentence this file cares about can be
    split across two lines with the break falling anywhere. Matching on the
    collapsed text asserts the sentence rather than the wrapping.
    """
    return " ".join(text.split())


def write_config(tmp_path: Path, **extra: object) -> Path:
    """An assumptions file that keeps the whole run offline.

    ``price_source: csv`` against the committed closes, and a pinned risk-free
    rate so that computing a WACC does not reach the Treasury.
    """
    lines = [
        "market:",
        "  risk_free_rate: 0.0483",
        "price_source: csv",
        f"price_csv_dir: {FIXTURES / 'prices'}",
    ]
    for key, value in extra.items():
        lines.append(f"{key}: {value}")
    path = tmp_path / "assumptions.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def blob() -> dict:
    with gzip.open(FADE_PANEL, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def small_panel_file(blob, tmp_path_factory) -> Path:
    """Forty filers out of the committed 224, Datadog among them.

    Derived from the committed fixture at test time rather than committed as a
    second binary, so it cannot drift away from the panel the model card was
    measured on. Forty is chosen because the fit refuses a training fold below
    ``ml.forecast.min_train_observations`` and forty filers clear it at every
    horizon, which is the smallest panel that still exercises the real code path.
    """
    picks = ["DDOG"] + [t for t in sorted(blob["payloads"]) if t != "DDOG"][:39]
    trimmed = {
        "payloads": {t: blob["payloads"][t] for t in picks},
        "meta": {t: blob["meta"][t] for t in picks},
    }
    path = tmp_path_factory.mktemp("fade") / "small_panel.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(trimmed, handle)
    return path


@pytest.fixture(scope="module")
def fade_output(tmp_path_factory) -> str:
    """One run of ``fade DDOG`` on the whole committed panel, reused by every test.

    Module scoped because fitting 224 filers across three horizons is the one
    genuinely expensive thing in this file, and every assertion below is about a
    different part of the same printout.
    """
    config = write_config(tmp_path_factory.mktemp("cfg"))
    result = runner.invoke(
        app,
        [
            "fade",
            "DDOG",
            "--config",
            str(config),
            "--panel",
            str(FADE_PANEL),
            "--facts",
            str(DDOG_FACTS),
        ],
    )
    assert result.exit_code == 0, result.output
    return result.output


@pytest.fixture(scope="module")
def signal_output(tmp_path_factory) -> str:
    """One run of ``signal`` on the recorded EV/Revenue panel, reused by every test."""
    config = write_config(tmp_path_factory.mktemp("cfg"))
    result = runner.invoke(
        app,
        [
            "signal",
            "--score",
            "cheapness",
            "--config",
            str(config),
            "--scores",
            str(SIGNAL_SCORES),
            "--prices",
            str(SIGNAL_PRICES),
        ],
    )
    assert result.exit_code == 0, result.output
    return result.output


# --------------------------------------------------------------------------- #
# fade: the comparison is the output
# --------------------------------------------------------------------------- #


def test_the_fade_command_values_the_assumed_path_and_the_fitted_one_side_by_side(
    fade_output,
):
    """The four valuations, which are the only form of this model worth printing.

    A fitted growth path on its own is a claim about a panel. These four numbers
    are a claim about Datadog in dollars a share, and they are the ones an analyst
    can argue with. The typed fade takes growth from 20% to 5% in a straight line
    and values the enterprise at 9,447mm, 36.65 a share. The fitted fade holds
    growth near 20% for three years and values it at 10,901mm, 40.61. The 10th and
    90th percentile paths value the same company at 26.97 and 66.69.

    All four have to be on the page. Printing 40.61 beside 36.65 without the band
    invites the reader to treat the middle of a forty point interval as a forecast.
    """
    text = flat(fade_output)
    assert "Assumed fade" in text
    assert "Fitted fade" in text
    assert "Fitted, low band" in text
    assert "Fitted, high band" in text
    for shown in ("36.65", "40.61", "26.97", "66.69"):
        assert shown in text, shown
    assert "9,447" in text and "10,901" in text


def test_the_band_is_printed_year_by_year_and_not_only_the_point_estimate(fade_output):
    """The honest output here is a range, so every fitted year carries its ends."""
    text = flat(fade_output)
    assert "Low" in text and "High" in text
    # The out-of-fold residual band at one year runs about minus twenty points to
    # plus twenty four around the estimate, so the first fitted year lands near 1%
    # at the low end and 45% at the high end on a 21% point estimate.
    assert "21.3%" in text and "1.0%" in text and "45.3%" in text


def test_every_year_says_whether_it_was_fitted_or_assumed(fade_output):
    """The handover is the thing that stops this being a black box.

    The fitted curve governs the first ``horizon_years``. What the filings support
    is reversion toward the mean growth of a growth sector, about 13%, and no
    terminal value can be built on that, so the later years fall back to the
    engine's own straight line. Blending the two into one path would hide exactly
    which of them is speaking.
    """
    text = flat(fade_output)
    assert "fitted" in text and "assumed" in text
    assert "years 4 to 5 are not fitted" in text


def test_the_baseline_is_printed_beside_the_model_at_every_horizon(fade_output):
    """Persistence is a hard baseline and the tie at one year is the honest headline.

    "Next year's growth equals this year's growth" scores 0.1510 and the ridge
    scores 0.1474, a lift of 0.0035 against a fold standard deviation of 0.0242.
    At one year this model does not beat doing nothing. At two and three it does,
    0.1476 against 0.1884 and 0.1391 against 0.1921, and both of those are outside
    the fold noise.
    """
    text = flat(fade_output)
    for horizon in ("1 year", "2 year", "3 year"):
        assert horizon in text, horizon
    assert "0.1474" in text and "0.1510" in text
    assert "0.1476" in text and "0.1884" in text
    assert "0.1391" in text and "0.1921" in text
    assert "ties it" in text and "beats it" in text
    assert "inside the fold-to-fold noise" in text


def test_all_three_baselines_are_shown_not_only_the_one_the_card_scores_against(
    fade_output,
):
    """The finding the brief for this command did not have, and it changes the reading.

    The model card scores against persistence, which is the hardest baseline at one
    year. It is not the hardest baseline at two or three: there the mean of the
    company's own sub-vertical scores 0.1578 and 0.1441 against the model's 0.1476
    and 0.1391, lifts of 0.0102 and 0.0050 against fold standard deviations of
    0.0162 and 0.0176. So the lift over the STRONGEST baseline is inside the fold
    noise at every horizon, even where the lift over persistence is not, and a
    reader who saw only the persistence column would take this curve for something
    it is not.

    ``techval.ml.forecast`` says as much in its own words: a model that beats
    persistence and loses to the mean of its own sub-vertical has learned something
    about the sector rather than about the company. The command prints the best
    baseline at each horizon so that nobody has to reconstruct which was in play.
    """
    text = flat(fade_output)
    assert "Training mean" in text and "Sub-vertical" in text
    assert "0.1693" in text and "0.1578" in text and "0.1441" in text
    assert "Best baseline" in text
    # Persistence wins the comparison only at one year; the sub-vertical mean is
    # the baseline to beat at two and three. The line is computed from the table
    # rather than written into the command, so it is safe to assert.
    assert (
        "Strongest baseline at each horizon: 1y persistence; 2y sub-vertical; "
        "3y sub-vertical." in text
    )
    assert text.count("inside fold noise") == 3


def test_the_fitted_fade_is_shown_against_the_sub_verticals_typical_fade(fade_output):
    """Whether this company is being treated as unusual is a question the path cannot answer.

    A fitted first year of 21% means one thing for a filer whose sub-vertical
    typically prints 17% and another for one whose sub-vertical typically prints
    28%. Datadog is faded 6.4 points where the average infrastructure software
    filer in this panel fades 4.2, which is close enough that the model is not
    claiming anything special about it, and the command says so rather than leaving
    the reader to guess.
    """
    text = flat(fade_output)
    assert "infrastructure software filer in this panel fades" in text
    assert "Sub-vertical" in text and "All TMT" in text
    assert "not treating this company as unusual" in text


def test_survivorship_is_stated_with_its_direction_on_every_run(fade_output):
    """The way this model is most likely to be wrong in a direction that costs money.

    A company whose growth collapses is bought or delisted and stops filing. The
    panel keeps 119 such filers in until the day they stop, which removes most of
    the bias and is why the gaps are tenths of a point rather than whole ones. What
    no panel can hold is the years after a filer left, and those are the years that
    would have faded hardest. So the residual runs one way, and naming the
    direction is the whole point of printing it.
    """
    text = flat(fade_output)
    assert "Survivorship" in text
    assert "fades TOO SLOWLY" in text
    assert "TOO HIGH" in text and "TOO LARGE" in text
    assert "+0.78%" in text  # the measured one-year gap between panel and survivors
    assert "9.8%" in text and "15.8%" in text


@pytest.fixture(scope="module")
def survivors_only_output(blob, tmp_path_factory) -> str:
    """``fade`` on a panel of filers that all still trade, which is the trap.

    Hand-picking a universe is the easiest thing in the world to do by accident:
    every ticker a person can name off the top of their head is a company that
    still exists. The panel that results has no leavers in it, and every
    survivorship number computed on it comes out at exactly zero.
    """
    picks = [
        t
        for t in sorted(blob["payloads"])
        if blob["meta"][t].get("status") == "listed"
    ][:45]
    assert "DDOG" in picks or picks, picks
    if "DDOG" not in picks:
        picks = ["DDOG"] + picks[:44]
    trimmed = {
        "payloads": {t: blob["payloads"][t] for t in picks},
        "meta": {t: blob["meta"][t] for t in picks},
    }
    path = tmp_path_factory.mktemp("fade") / "survivors.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(trimmed, handle)
    config = write_config(tmp_path_factory.mktemp("cfg"))
    result = runner.invoke(
        app,
        [
            "fade",
            "DDOG",
            "--config",
            str(config),
            "--panel",
            str(path),
            "--facts",
            str(DDOG_FACTS),
        ],
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_a_panel_with_no_leavers_is_flagged_rather_than_reported_as_unbiased(
    survivors_only_output,
):
    """Found on a live run, not reasoned about, and it would have read as reassurance.

    Fit the curve on a list of companies that all still trade and the survivorship
    gaps come out at +0.00% at every horizon, because the comparison is between a
    sample and itself. The full panel's paragraph, which says the leavers are kept
    in and that is why the gaps are small, is exactly wrong about such a run: the
    fix was not applied, the bias was not measured, and it is large. A reader who
    saw three zeros under a heading about survivorship would conclude the opposite
    of the truth, which is worse than printing nothing.
    """
    text = flat(survivors_only_output)
    assert "FLAG: not one filer in this panel ever left the tape" in text
    assert "zero by construction" in text
    assert "0 of the" in text  # the count of leavers, taken from this panel
    # The reassuring paragraph must not appear on a run that did not earn it.
    assert "Keeping the leavers in until the day they stop is most of the fix" not in text


def test_the_forecast_switch_is_reported_rather_than_silently_flipped(fade_output):
    """``ml.forecast.enabled`` defaults false and this command is what turns it on.

    The switch exists so that a fitted growth path can only reach a discounted cash
    flow through a decision somebody made on purpose. Typing this command is that
    decision, so the command honours it for its own run and says so. A run that
    flipped it silently would leave a reader unable to tell whether their ordinary
    ``techval value`` numbers had moved too.
    """
    text = flat(fade_output)
    assert "ml.forecast.enabled is false in the assumptions" in text
    assert "on for this run only" in text
    assert "ml.forecast.enabled true for this run" in text


def test_the_assumptions_that_drove_the_numbers_are_echoed(fade_output):
    text = flat(fade_output)
    assert "Assumptions in force" in text
    assert "dcf.projection_years 5" in text
    assert "ml.random_seed 7" in text
    assert "wacc used by the DCF" in text


# --------------------------------------------------------------------------- #
# fade: point in time
# --------------------------------------------------------------------------- #


def test_a_dated_run_fits_only_on_filings_that_predate_the_valuation_date(
    small_panel_file, tmp_path
):
    """The half of ``--as-of`` that nothing else in the engine does for us.

    ``build_fade_panel`` pins each fiscal year to a filing date, which makes every
    feature point in time, and then runs to the present whatever date the valuation
    is struck at. Without ``_truncate_panel`` a run dated 2024 would fit its growth
    curve on filings through 2026 and then value a company as at 2024, which is
    hindsight of the purest kind: the valuation would look excellent and the
    failure would be silent.
    """
    config = write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "fade",
            "DDOG",
            "--config",
            str(config),
            "--panel",
            str(small_panel_file),
            "--facts",
            str(DDOG_FACTS),
            "--as-of",
            "2024-06-28",
        ],
    )
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "Point in time" in text
    assert "observation(s) filed after 2024-06-28 were removed" in text
    assert "forward label(s) that had not been filed by then were detached" in text
    # The observation the growth path is built from has to be one that existed.
    built = [line for line in result.output.splitlines() if "as filed on" in line]
    assert built, result.output
    filed = date.fromisoformat(flat(" ".join(built)).split("as filed on ")[1][:10])
    assert filed <= date(2024, 6, 28)


def test_truncation_detaches_labels_that_were_not_public_yet(blob):
    """A label carries a filing date of its own, and that is the half people miss.

    The label for horizon h is the growth the filer printed in its 10-K h years
    later, so an observation filed in 2019 still holds a three year label that was
    not public until 2022. Dropping only the late OBSERVATIONS would leave the
    recent end of the training set in place with the answers attached.
    """
    ticker = "DDOG"
    panel = build_fade_panel(
        {ticker: blob["meta"][ticker]["sub_vertical"]},
        _OnePayload(blob["payloads"][ticker]),
    )
    knowledge = date(2023, 1, 1)
    assert any(
        any(d > knowledge for d in o.label_dates.values()) for o in panel.observations
    ), "the fixture should hold labels filed after the knowledge date, or this proves nothing"

    cut = _truncate_panel(panel, knowledge)
    assert all(o.as_of <= knowledge for o in cut.observations)
    for observation in cut.observations:
        assert all(d <= knowledge for d in observation.label_dates.values())
        assert set(observation.labels) == set(observation.label_dates)


def test_truncating_past_the_beginning_of_the_panel_refuses_rather_than_fits_nothing(
    blob,
):
    panel = build_fade_panel(
        {"DDOG": blob["meta"]["DDOG"]["sub_vertical"]},
        _OnePayload(blob["payloads"]["DDOG"]),
    )
    with pytest.raises(ConfigError) as exc:
        _truncate_panel(panel, date(1999, 1, 1))
    assert "nothing to fit" in str(exc.value)


class _OnePayload:
    """A one-company stand-in for the panel client, unpinned as the builder requires."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def company_facts(self, ticker: str) -> CompanyFacts:
        return CompanyFacts(self.payload, ticker)


# --------------------------------------------------------------------------- #
# fade: refusals
# --------------------------------------------------------------------------- #


def test_a_file_that_is_not_a_recorded_panel_is_refused_with_the_shape_it_wanted(
    tmp_path,
):
    bad = tmp_path / "not_a_panel.json"
    bad.write_text(json.dumps({"hello": "world"}))
    result = runner.invoke(
        app, ["fade", "DDOG", "--config", str(write_config(tmp_path)), "--panel", str(bad)]
    )
    assert result.exit_code == 1
    text = flat(result.output)
    assert "ConfigError" in text
    assert "'payloads'" in text and "'meta'" in text
    assert "Traceback" not in result.output


@pytest.fixture(scope="module")
def panel_without_datadog(blob, tmp_path_factory) -> Path:
    """The same forty filers with Datadog taken out, to prove the refusal."""
    picks = [t for t in sorted(blob["payloads"]) if t != "DDOG"][:40]
    trimmed = {
        "payloads": {t: blob["payloads"][t] for t in picks},
        "meta": {t: blob["meta"][t] for t in picks},
    }
    path = tmp_path_factory.mktemp("fade") / "no_ddog.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(trimmed, handle)
    return path


def test_a_ticker_outside_the_panel_is_refused_rather_than_extrapolated_to(
    panel_without_datadog, tmp_path
):
    """A fade curve is fitted on filers, so a filer it never saw gets nothing.

    The refusal matters more than it looks. Producing a growth path for a company
    that is not in the panel would mean scoring a feature vector nobody built, and
    the engine's cardinal rule is that a number nobody can trace does not get
    printed next to numbers that trace to filings. The message says how many filers
    the panel does hold, so the reader can tell a missing name from a typo.
    """
    result = runner.invoke(
        app,
        [
            "fade",
            "DDOG",
            "--config",
            str(write_config(tmp_path)),
            "--panel",
            str(panel_without_datadog),
            "--facts",
            str(DDOG_FACTS),
        ],
    )
    assert result.exit_code == 1
    text = flat(result.output)
    assert "ConfigError" in text
    assert "not in the panel this fade model was fitted on" in text
    assert "Traceback" not in result.output


def test_a_target_whose_filings_carry_no_revenue_is_refused_cleanly(
    small_panel_file, tmp_path
):
    """A ``TechvalError`` from the valuation half reaches the terminal as a sentence.

    The engine never interpolates a figure it could not find, so a payload with no
    revenue tag in the ladder raises. What matters for a command is that the raise
    arrives as the named concept and the tags that were tried rather than as a
    stack trace.
    """
    result = runner.invoke(
        app,
        [
            "fade",
            "NVDA",
            "--config",
            str(write_config(tmp_path)),
            "--panel",
            str(small_panel_file),
            "--facts",
            str(FIXTURES / "companyfacts_NVDA_shares.json"),
        ],
    )
    assert result.exit_code == 1
    text = flat(result.output)
    assert "MissingDataError" in text
    assert "could not source 'revenue' for NVDA" in text
    assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# signal: the adversary
# --------------------------------------------------------------------------- #


def test_the_recorded_cheapness_panel_reproduces_the_reference_result(signal_output):
    """The headline, pinned so it cannot drift through the command without being seen.

    Cheapness on trailing EV/Revenue, a hundred technology, media and
    telecommunications companies, thirty-five quarterly cross-sections, twelve
    month forward returns. Mean information coefficient -0.0984 against +0.0002 for
    a random score with the same cross-sectional shape. The cheap half of the
    technology universe underperformed the expensive half over this decade, which
    is the opposite of what the textbook says value does and exactly what anyone
    who lived through it remembers.
    """
    text = flat(signal_output)
    assert "-0.0984" in text
    assert "0.0002" in text
    assert "-2.65" in text and "-1.62" in text
    assert "1.64x" in text
    assert "2,604" in text
    assert "35" in text


def test_the_command_says_not_significant_in_those_words(signal_output):
    """The case the whole harness exists for.

    The naive statistic clears two and would be written up. The corrected one does
    not. A command that printed the first without the verdict would be doing the
    thing this module was built to stop, so the words are asserted rather than the
    number.
    """
    text = flat(signal_output)
    assert "NOT SIGNIFICANT" in text
    assert "clears two and would be written up" in text
    assert "Use the baseline." in text


def test_the_effective_count_is_printed_before_the_company_date_count(signal_output):
    """2,604 company-dates and 13 effective observations describe the same sample.

    Only one of them is the number a standard error should be read against. A
    hundred technology names in one quarter load on one factor and one rate cycle,
    and twelve month returns sampled quarterly share three quarters of their path,
    so neither a name nor a date is a draw. Order is the argument here: a reader
    who meets 2,604 first has formed a view before the 13 arrives.
    """
    text = flat(signal_output)
    effective = text.index("Effective observations, Newey-West")
    company_dates = text.index("Company-dates scored")
    assert effective < company_dates
    assert "13.0" in text
    assert "Independent 12 month periods" in text


def test_the_overlap_correction_is_explained_by_persistence_and_not_by_overlap_alone(
    signal_output,
):
    """The refinement the module measured, stated correctly or not stated at all.

    Overlapping windows do not on their own autocorrelate a cross-sectional
    coefficient: the correlation lives inside one date, so a market move common to
    every name cancels out of it, and a score redrawn from noise at each rebalance
    comes back with nothing worth correcting. What autocorrelates the series is a
    score that persists. Every score in this package persists, so the warning
    stands, and an explanation that omitted the second half would be teaching the
    reader something false.
    """
    text = flat(signal_output)
    assert "Overlap plus a persistent score is what inflates a t-statistic" in text
    assert "a market move common to every name cancels out of it" in text
    assert "+0.76" in text  # the measured first-order autocorrelation
    assert "the correction bites" in text


def test_the_information_coefficient_series_is_printed_date_by_date(
    tmp_path_factory,
):
    """The series is the statistic of record, so it is on the page and not summarised.

    A mean of -0.0984 can be one catastrophic quarter or thirty-five mildly
    negative ones, and the two are different claims. Printing the series is what
    lets a reader tell them apart without rerunning anything.
    """
    config = write_config(tmp_path_factory.mktemp("cfg"))
    result = runner.invoke(
        app,
        [
            "signal",
            "--config",
            str(config),
            "--scores",
            str(SIGNAL_SCORES),
            "--prices",
            str(SIGNAL_PRICES),
            "--universe",
            "DDOG,MDB,ZS,CRWD,NET,SNOW",
            "--min-names",
            "3",
            "--buckets",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "The coefficient series" in text
    assert "2022-03-15" in text and "2025-06-15" in text


def test_the_bucket_table_carries_its_monotonicity_and_its_turnover(signal_output):
    """A factor and a screen look identical until the buckets are laid out.

    A signal monotone across the cross-section is a different animal from one that
    lives entirely in its extreme bucket, and the second is a handful of names with
    a much shorter life expectancy. On this panel the ordering is monotone the
    wrong way: the rank correlation between bucket number and mean return is -0.90,
    the top less bottom spread is -5.38%, and there is no break-even cost to quote
    because the spread is negative before any cost at all.
    """
    text = flat(signal_output)
    assert "-5.38%" in text
    assert "-0.90" in text
    assert "Turnover per rebalance" in text
    assert "the spread is not positive before any cost at all" in text


def test_the_sign_convention_is_stated_before_the_coefficient_is_shown(signal_output):
    """Choosing the sign after seeing the coefficient is a second test in disguise.

    The name of the score fixes the sign, not the result. ``cheapness`` negates the
    multiple so that a high score is a cheap company, and the run prints that
    before it prints anything the sign could have been chosen to flatter.
    """
    text = flat(signal_output)
    convention = text.index("Sign convention")
    first_coefficient = text.index("-0.0984")
    assert convention < first_coefficient
    assert "a high score is a cheap company" in text


def test_flipping_the_sign_flips_the_coefficient(tmp_path_factory):
    """The harness reads an ordering and nothing else, which the flip demonstrates.

    A rank correlation against the reversed ordering is the same number negated, so
    ``--no-negate`` on a cheapness panel measures expensiveness and has to come back
    with the opposite sign. If it did not, the sign convention would not be doing
    what the help text claims.
    """
    config = write_config(tmp_path_factory.mktemp("cfg"))
    common = [
        "signal",
        "--config",
        str(config),
        "--scores",
        str(SIGNAL_SCORES),
        "--prices",
        str(SIGNAL_PRICES),
        "--universe",
        "DDOG,MDB,ZS,CRWD,NET,SNOW",
        "--min-names",
        "3",
        "--buckets",
        "2",
        "--no-series",
        "--draws",
        "20",
    ]
    negated = runner.invoke(app, common)
    plain = runner.invoke(app, common + ["--no-negate"])
    assert negated.exit_code == 0 and plain.exit_code == 0

    def mean_ic(output: str) -> float:
        for line in output.splitlines():
            if line.strip().startswith("Mean IC"):
                return float(line.split()[-1])
        raise AssertionError(output)

    assert mean_ic(negated.output) == pytest.approx(-mean_ic(plain.output), abs=1e-9)


def test_the_number_of_tests_run_changes_only_the_adjusted_p_value(tmp_path_factory):
    """Running twenty-four tests and reporting the winner is not a rounding error.

    Nothing about a winner's own arithmetic reveals how many attempts produced it;
    it is a property of the search rather than of the result and cannot be
    recovered afterwards. So the count is the caller's to state, it moves the Sidak
    p-value and nothing else, and the test pins that it moves nothing else.
    """
    config = write_config(tmp_path_factory.mktemp("cfg"))
    common = [
        "signal",
        "--config",
        str(config),
        "--scores",
        str(SIGNAL_SCORES),
        "--prices",
        str(SIGNAL_PRICES),
        "--universe",
        "DDOG,MDB,ZS,CRWD,NET,SNOW",
        "--min-names",
        "3",
        "--buckets",
        "2",
        "--no-series",
        "--draws",
        "20",
    ]
    one = runner.invoke(app, common)
    many = runner.invoke(app, common + ["--tests-run", "24"])
    assert one.exit_code == 0 and many.exit_code == 0

    def row(output: str, label: str) -> str:
        for line in output.splitlines():
            if line.strip().startswith(label):
                return line.split()[-1]
        raise AssertionError(f"{label} not in {output}")

    assert row(one.output, "Mean IC") == row(many.output, "Mean IC")
    assert row(one.output, "p, Newey-West") == row(many.output, "p, Newey-West")
    assert row(one.output, "p, Sidak-adjusted") != row(many.output, "p, Sidak-adjusted")
    assert "24 tests were run" in flat(many.output)


def test_the_company_date_detail_keeps_the_rows_that_could_not_be_scored(
    tmp_path_factory, tmp_path
):
    """Dropping the failures would make the table agree with the statistics.

    It would also make it disagree with the run, and the count of what could not be
    scored is part of the result. The recorded panel carries 2,992 company-dates of
    which 2,604 reached a coefficient, and the 388 censored at the edge of the
    price data have to be visible somewhere.
    """
    config = write_config(tmp_path_factory.mktemp("cfg"))
    out = tmp_path / "detail.csv"
    result = runner.invoke(
        app,
        [
            "signal",
            "--config",
            str(config),
            "--scores",
            str(SIGNAL_SCORES),
            "--prices",
            str(SIGNAL_PRICES),
            "--no-series",
            "--draws",
            "20",
            "--csv",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2992
    kinds = {r["Exit kind"] for r in rows}
    assert "censored" in kinds and "held" in kinds


# --------------------------------------------------------------------------- #
# signal: refusals
# --------------------------------------------------------------------------- #


def test_the_echoed_settings_are_the_ones_the_run_used_not_the_ones_on_file(
    tmp_path_factory,
):
    """A flag that overrode the file has to say so in the echo.

    This was a real defect found on the first live run. ``--min-names 12`` took
    effect and the assumptions block still printed the file's 20, so a reader
    working out why a run came back thin would have read the wrong number and
    concluded the flag had not landed. Printing the effective value with the
    configured one beside it is the fix, and a setting nobody overrode prints as a
    bare number so the annotation means something when it appears.
    """
    config = write_config(tmp_path_factory.mktemp("cfg"))
    result = runner.invoke(
        app,
        [
            "signal",
            "--config",
            str(config),
            "--scores",
            str(SIGNAL_SCORES),
            "--prices",
            str(SIGNAL_PRICES),
            "--universe",
            "DDOG,MDB,ZS,CRWD,NET,SNOW",
            "--min-names",
            "3",
            "--buckets",
            "2",
            "--no-series",
            "--draws",
            "20",
        ],
    )
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "ml.signals.min_names_per_date 3 (flag, 20 in the assumptions)" in text
    assert "ml.signals.buckets 2 (flag, 5 in the assumptions)" in text
    assert "ml.signals.horizon_months 12" in text
    assert "ml.signals.horizon_months 12 (flag" not in text


def test_a_score_panel_with_no_prices_behind_it_is_refused(tmp_path):
    result = runner.invoke(
        app,
        ["signal", "--config", str(write_config(tmp_path)), "--scores", str(SIGNAL_SCORES)],
    )
    assert result.exit_code == 1
    assert "has no forward return to be tested against" in flat(result.output)
    assert "Traceback" not in result.output


def test_a_fitted_score_cannot_be_built_by_the_harness_that_judges_it(tmp_path):
    """The refusal is the design, not a gap in it.

    Nothing in ``techval.ml.signals`` imports another model, so nothing in it can be
    tuned to flatter one, and the command inherits that. The warranted residual and
    the propensity score reach this harness as a CSV of ``(ticker, as_of, value)``
    like any other score, and the refusal says exactly which columns to write.
    """
    result = runner.invoke(
        app, ["signal", "--config", str(write_config(tmp_path)), "--score", "warranted"]
    )
    assert result.exit_code == 1
    text = flat(result.output)
    assert "imports no model it might be tuned to flatter" in text
    assert "residual_log" in text


def test_an_unknown_score_name_lists_the_ones_that_exist(tmp_path):
    result = runner.invoke(
        app, ["signal", "--config", str(write_config(tmp_path)), "--score", "alpha"]
    )
    assert result.exit_code == 1
    text = flat(result.output)
    assert "cheapness" in text and "propensity" in text and "warranted" in text


def test_a_missing_value_column_names_the_columns_the_file_has(tmp_path):
    result = runner.invoke(
        app,
        [
            "signal",
            "--config",
            str(write_config(tmp_path)),
            "--scores",
            str(SIGNAL_SCORES),
            "--prices",
            str(SIGNAL_PRICES),
            "--value-column",
            "alpha",
        ],
    )
    assert result.exit_code == 1
    text = flat(result.output)
    assert "has no 'alpha' column" in text
    assert "ev_revenue" in text


def test_a_live_cheapness_run_without_a_window_is_refused_rather_than_guessed_at(
    tmp_path,
):
    result = runner.invoke(
        app, ["signal", "--config", str(write_config(tmp_path)), "--score", "cheapness"]
    )
    assert result.exit_code == 1
    assert "needs --universe, --from and --to" in flat(result.output)


# --------------------------------------------------------------------------- #
# Rebalance dates
# --------------------------------------------------------------------------- #


def test_rebalance_dates_hold_the_day_of_month_and_stop_at_the_end():
    dates = _rebalance_dates(date(2020, 3, 15), date(2021, 3, 15), 3)
    assert dates == [
        date(2020, 3, 15),
        date(2020, 6, 15),
        date(2020, 9, 15),
        date(2020, 12, 15),
        date(2021, 3, 15),
    ]
    assert _rebalance_dates(date(2020, 3, 15), date(2020, 3, 14), 3) == []


def test_a_thirty_first_falls_back_inside_short_months():
    """A rebalance day that does not exist has to land somewhere stated.

    The 28th rather than the month end, because a month end picks up the volume of
    an index reconstitution in its entry price and the whole point of pinning to a
    mid-month day was to avoid that.
    """
    dates = _rebalance_dates(date(2021, 1, 31), date(2021, 7, 1), 1)
    assert date(2021, 2, 28) in dates
    assert all(d.day in (31, 28) for d in dates)
