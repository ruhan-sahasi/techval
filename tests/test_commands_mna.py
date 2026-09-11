"""The precedents and targets commands, rendered against the committed fixtures.

No test here touches the network. ``precedents`` runs against the nine real TMT
merger filings under ``tests/fixtures/merger`` and ``targets`` against the 493
registrants and 107 deals under ``tests/fixtures/mna``, both recorded on
2026-09-11 as their manifests say.

What is asserted is deliberately not that the numbers are right, which the module
tests already establish over the same fixtures. It is that the refusals survive
the trip to a terminal, because a rendering layer is exactly where a careful
engine stops being careful:

*The control premium warning is above the table, not below it.* Asserted by
character position, because a caution a reader reaches after the numbers has
already failed at the only job it has.

*Iridium's missing offer price stays missing.* The collar means no exchange ratio
exists at announcement, so ``offer_price`` is ``None``. The cash leg of 27.00
dollars must not appear in the offer column of the deal table, and the reason
must appear in words. A renderer that quietly substituted the cash leg would
report a 40 percent discount to a price nobody agreed to pay, and it would look
exactly like a working precedent table while doing it.

*Both premia are printed and the gap is flagged.* Roku at 11.3 percent against
27.0 and Payoneer at 9.6 against 37.2 are the same measurement problem in the
same direction, and Silicon Labs at +10.3 points is it in the other. One
convention hides all three.

*A refused statistic says refused.* A blank cell cannot be told from a rendering
bug, so a column below the deal floor prints the word in every cell and keeps its
``n``.

*The target screen prints its own score beside the list.* The AUC, its baseline,
the fold dispersion and the sentence that the lift sits inside the noise all
appear before the first ranked name.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from techval import commands_mna as C
from techval.edgar import (
    SEC_FACTS_URL,
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    EdgarClient,
    HttpCache,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MERGER = FIXTURES / "merger"
MNA = FIXTURES / "mna"

#: The date every fixture was recorded. Every claim about what was pending and
#: what had closed is fixed to it.
AS_OF = date(2026, 9, 11)

DEAL_TICKERS = ["SLAB", "RAMP", "PAYO", "ROKU", "IRDM", "SPLK", "MNDT", "ZEN", "WORK"]

runner = CliRunner()

_JSON: dict[str, dict] = {}
_TEXT: dict[str, str] = {}


def _read_json(path: Path) -> dict:
    key = str(path)
    if key not in _JSON:
        _JSON[key] = json.loads(path.read_text())
    return _JSON[key]


class MergerFixtureClient(EdgarClient):
    """The real EdgarClient with its three JSON endpoints served from disk.

    Written here rather than imported from the precedents test module so that
    this file does not break when that one is edited. Subclassing rather than
    reimplementing keeps ``filings``, ``ticker_to_cik`` and the knowledge-date
    filtering on the production code path, which is the whole reason the command
    can be trusted to behave offline the way it behaves live.
    """

    def __init__(self, cache=None, knowledge_date: date | None = None) -> None:
        # The same signature as EdgarClient, because the command constructs it
        # positionally the way it constructs the real one. A fixture double whose
        # signature has drifted from the class it stands in for tests the double.
        super().__init__(cache=HttpCache(enabled=False), knowledge_date=knowledge_date)
        self._by_cik: dict[int, str] = {}
        for ticker in DEAL_TICKERS:
            payload = _read_json(MERGER / f"submissions_{ticker}.json")
            self._by_cik[int(payload["cik"])] = ticker

    def _get_json(self, url: str) -> dict:
        if url == SEC_TICKERS_URL:
            raw = _read_json(MERGER / "company_tickers.json")
            return {k: v for k, v in raw.items() if k != "_fixture"}
        for cik, ticker in self._by_cik.items():
            if url == SEC_SUBMISSIONS_URL.format(cik=cik):
                return _read_json(MERGER / f"submissions_{ticker}.json")
            if url == SEC_FACTS_URL.format(cik=cik):
                return _read_json(MERGER / f"companyfacts_{ticker}.json")
        raise AssertionError(f"a test reached for {url}, which is not a fixture")

    def ticker_to_cik(self, ticker: str) -> int:
        for cik, known in self._by_cik.items():
            if known == ticker.upper():
                return cik
        return super().ticker_to_cik(ticker)

    def filing_text(self, ticker: str, filing: dict) -> str:
        accession = filing["accession"]
        if accession not in _TEXT:
            path = MERGER / "text" / f"{accession}.txt"
            # Most filings in the pruned index carry no committed text, which
            # stands in for a document holding no merger agreement.
            _TEXT[accession] = path.read_text() if path.exists() else ""
        return _TEXT[accession]


@pytest.fixture
def merger_config(tmp_path: Path, monkeypatch) -> Path:
    """An assumptions file pointing the price source at the committed closes.

    ``price_source: csv`` is not a test convenience. No public source serves
    price history for a delisted symbol, and a completed target is delisted by
    definition, so a real premium over a closed deal needs the closes supplied
    from disk. Driving the command through the same option a user would set
    keeps ``make_price_source`` on the production path.
    """
    monkeypatch.setattr(C, "EdgarClient", MergerFixtureClient)
    path = tmp_path / "assumptions.yaml"
    path.write_text(
        # Quoted deliberately. YAML reads a bare 2026-09-11 as a date object and
        # Assumptions.as_of is typed as a string, so an unquoted value fails
        # validation before the command is reached. See the PR body.
        'as_of: "2026-09-11"\n'
        "price_source: csv\n"
        f"price_csv_dir: {MERGER / 'prices'}\n"
        "ml:\n"
        "  mna:\n"
        # The fixture set is about extraction and arithmetic, so the default
        # 250mm floor is lifted and the size filter exercised separately.
        "    min_deal_size: 0.0\n"
    )
    return path


def _run(args: list[str]) -> str:
    result = runner.invoke(C.app, args)
    assert result.exit_code == 0, result.output
    return result.output


def _flat(text: str) -> str:
    """Whitespace collapsed to single spaces.

    Rich wraps prose at the console width, so a sentence the command prints as
    one string arrives split across lines with the break in an arbitrary place.
    Asserting on the wrapped form tests the terminal width; asserting on this
    tests the sentence.
    """
    return re.sub(r"\s+", " ", text).strip()


def _sections(output: str) -> dict[str, str]:
    """The output split at its rules, keyed by rule title.

    Needed because several section titles also occur inside ordinary prose: the
    header echoes "Statistics need 5 deals per column" well before the statistics
    rule, so a naive split on the word lands in the wrong half of the page.
    """
    out: dict[str, str] = {}
    title, buf = "header", []
    for line in output.splitlines():
        if "─" in line:
            out[title] = "\n".join(buf)
            title, buf = line.replace("─", "").strip(), []
        else:
            buf.append(line)
    out[title] = "\n".join(buf)
    return out


def _section(output: str, needle: str) -> str:
    for title, body in _sections(output).items():
        if needle in title:
            return body
    raise AssertionError(f"no section whose rule mentions {needle!r}")


@pytest.fixture
def precedent_output(merger_config) -> str:
    return _run(["precedents", ",".join(DEAL_TICKERS), "--config", str(merger_config)])


# --------------------------------------------------------------------------- #
# precedents: the warning that has to be above the numbers
# --------------------------------------------------------------------------- #


def test_the_control_premium_warning_prints_above_the_deal_table(precedent_output):
    """The single most misread number in the package, cautioned before it appears.

    A precedent multiple sits above a trading multiple by construction, so the
    gap between them is not upside. The caution is worth nothing below the table
    it qualifies.
    """
    flat = _flat(precedent_output)
    assert "control premium" in flat
    assert "roughly 25 to 40 percent across TMT" in flat
    warning_at = precedent_output.index("counts the control premium twice")
    first_deal_at = precedent_output.index("IRDM")
    assert warning_at < first_deal_at, "the caution landed under the table"


def test_the_warning_says_precedents_are_never_averaged_with_comps(precedent_output):
    flat = _flat(precedent_output)
    assert "not a trading comparable" in flat
    assert "never averaged" in flat


# --------------------------------------------------------------------------- #
# precedents: the deal with no offer price
# --------------------------------------------------------------------------- #


def _table_row(output: str, ticker: str) -> str:
    """The deal table's row for one ticker, out of the section holding the table."""
    head = _section(output, "Precedent transactions")
    rows = [ln for ln in head.splitlines() if re.search(rf"\b{ticker}\b", ln)]
    assert rows, f"{ticker} has no row in the deal table"
    return rows[0]


def test_iridium_reports_no_offer_price_at_all(precedent_output):
    """A collar means no exchange ratio exists at announcement, so no price does.

    Rocket Lab pays 27.00 in cash plus a number of its own shares set by a
    collar: one ratio below a reference price, a floating ratio inside the band,
    a third above it. The ratio is fixed from the buyer's price at closing. There
    is nothing to report at announcement and the command reports nothing.
    """
    row = _table_row(precedent_output, "IRDM")
    assert "see below" in row
    assert "27.00" not in row, "the cash leg was substituted for the offer price"
    assert "Deals with no offer price" in precedent_output
    assert "No offer price is reported and none is substituted." in precedent_output


def test_the_collar_is_explained_in_the_filing_s_own_terms(precedent_output):
    detail = _flat(_section(precedent_output, "Deals with no offer price"))
    assert "collar" in detail
    assert "does not exist at announcement" in detail


def test_the_cash_leg_is_shown_once_and_labelled_as_the_cash_leg(precedent_output):
    """Shown, because it is a fact about the deal. Labelled, because it is not the price."""
    detail = _flat(_section(precedent_output, "Deals with no offer price"))
    assert "27.00" in detail
    assert "CASH LEG and not the offer" in detail
    # And it never became a multiple or a premium anywhere.
    row = _table_row(precedent_output, "IRDM")
    assert row.count("n/a") >= 2


# --------------------------------------------------------------------------- #
# precedents: two premia, and where they disagree
# --------------------------------------------------------------------------- #


def test_both_premia_are_columns_on_every_row(precedent_output):
    assert "Prem. 1d" in precedent_output
    assert "Prem. 30d" in precedent_output
    roku = _table_row(precedent_output, "ROKU")
    assert "11.3%" in roku and "27.0%" in roku


def test_roku_s_disagreement_is_flagged_with_both_numbers_and_the_gap(precedent_output):
    """The stock ran twenty percent before a weekend signing, so the conventions split.

    Eleven percent against twenty seven. Only one of those is a control premium,
    and which one depends on whether the market already knew.
    """
    section = _section(precedent_output, "the two conventions disagree")
    assert "ROKU" in section
    assert "11.3%" in section and "27.0%" in section
    assert "-15.7" in section
    assert "ran INTO the announcement" in section


def test_payoneer_disagrees_by_more_than_roku_and_is_also_flagged(precedent_output):
    section = _section(precedent_output, "the two conventions disagree")
    assert "PAYO" in section
    assert "9.6%" in section and "37.2%" in section
    assert "-27.6" in section


def test_a_stock_that_fell_into_the_announcement_flags_the_other_way(precedent_output):
    """Silicon Labs is +10.3 points: there the one-day premium flatters the deal."""
    section = _section(precedent_output, "the two conventions disagree")
    slab = [ln for ln in section.splitlines() if "SLAB" in ln]
    assert slab, "SLAB should be flagged: its gap is +10.3 points"
    assert "+10.3" in slab[0]
    assert "FELL into the announcement" in slab[0]


def test_a_deal_inside_the_flag_threshold_is_not_in_the_disagreement_section(
    precedent_output,
):
    """LiveRamp is -0.8 points, which is the two conventions agreeing."""
    section = _section(precedent_output, "the two conventions disagree")
    assert "RAMP" not in section
    # It is still in the table above, with both premia on the row.
    ramp = _table_row(precedent_output, "RAMP")
    assert "29.8%" in ramp and "30.6%" in ramp


def test_a_premium_with_no_price_series_is_absent_rather_than_zero(precedent_output):
    """Splunk closed and was delisted, so no source serves its unaffected close."""
    splk = _table_row(precedent_output, "SPLK")
    assert "0.0%" not in splk
    assert "no price series" in precedent_output


# --------------------------------------------------------------------------- #
# precedents: statistics that refuse
# --------------------------------------------------------------------------- #


def test_a_column_below_the_deal_floor_prints_refused_in_every_cell(precedent_output):
    """Two EBITDA multiples out of nine deals is not a median."""
    stats = _section(precedent_output, "Statistics")
    assert "refused" in stats
    assert "column(s) refused" in stats
    assert "a median of three transactions is three transactions" in _flat(stats).lower()


def test_a_refused_column_keeps_its_n_so_the_refusal_is_legible(precedent_output):
    """The count stays visible. A blank row cannot be told from a rendering bug."""
    stats = _section(precedent_output, "Statistics")
    n_row = [ln for ln in stats.splitlines() if ln.strip().startswith("n ")]
    assert n_row, "the statistics table lost its n row"
    assert re.search(r"\b6\b", n_row[0]), "six revenue multiples were built"
    assert re.search(r"\b2\b", n_row[0]), "two EBITDA multiples were built"


def test_a_column_that_clears_the_floor_is_reported(precedent_output):
    """Six revenue multiples clears five, so EV/Revenue gets a median."""
    stats = _section(precedent_output, "Statistics")
    median = [ln for ln in stats.splitlines() if ln.strip().startswith("Median")]
    assert median
    assert "5.5x" in median[0]


def test_every_sub_vertical_is_refused_on_this_sample_and_says_so(precedent_output):
    section = _flat(_section(precedent_output, "By sub-vertical"))
    assert "No sub-vertical reached the 5-deal floor" in section


def test_the_statistics_floor_is_configurable_and_lowering_it_reports_more(
    merger_config,
):
    lowered = _run(
        ["precedents", ",".join(DEAL_TICKERS), "--config", str(merger_config),
         "--min-deals", "2"]
    )
    stats = _section(lowered, "Statistics")
    assert "refused" not in stats, "every column carries at least two values"
    assert "application_software" in _section(lowered, "By sub-vertical")


# --------------------------------------------------------------------------- #
# precedents: filters
# --------------------------------------------------------------------------- #


def test_the_sub_vertical_filter_narrows_and_recomputes(merger_config):
    out = _run(
        ["precedents", ",".join(DEAL_TICKERS), "--config", str(merger_config),
         "--sub-vertical", "internet"]
    )
    assert "2 of 9 ticker(s)" in out
    head = _section(out, "Precedent transactions")
    assert "RAMP" in head and "ZEN" in head
    assert "SPLK" not in head and "IRDM" not in head


def test_the_size_floor_drops_the_unsized_and_flags_the_count(merger_config):
    """A deal whose size could not be built cannot be shown to clear a floor.

    So it is excluded rather than waved through, and the exclusion is recorded
    with its count. Three of the nine here have no equity purchase price.
    """
    out = _run(
        ["precedents", ",".join(DEAL_TICKERS), "--config", str(merger_config),
         "--min-size", "5000"]
    )
    flat = _flat(out)
    assert "4 of 9 ticker(s)" in flat
    assert "dropped by the 5,000mm size floor" in flat
    assert "not because they were small" in flat


def test_a_filter_that_empties_the_set_is_an_ordinary_answer(merger_config):
    out = _run(
        ["precedents", ",".join(DEAL_TICKERS), "--config", str(merger_config),
         "--sub-vertical", "gaming"]
    )
    flat = _flat(out)
    assert "No transaction survived" in flat
    assert "Most companies are not acquired" in flat


# --------------------------------------------------------------------------- #
# precedents: provenance and errors
# --------------------------------------------------------------------------- #


def test_every_deal_names_the_document_it_came_from(precedent_output):
    section = _section(precedent_output, "Deal size and provenance")
    # The Splunk merger 8-K, by accession, as the module extracted it.
    assert "0001104659-23-102594" in section
    assert "Confidence is a coarse ladder" in _flat(precedent_output)


def test_the_size_a_floor_filters_on_is_printed_beside_the_floor(precedent_output):
    """A filter whose quantity is not on the page is one the reader takes on trust.

    Deal size is out of the wide table and in this one because a fifteenth
    column there shrank the price columns until 143.66 rendered as 143 and an
    ellipsis, and a truncated numeral still reads as a number.
    """
    section = _section(precedent_output, "Deal size and provenance")
    assert "Equity, mm" in section and "EV, mm" in section
    splk = [ln for ln in section.splitlines() if "SPLK" in ln]
    assert splk and "25,856" in splk[0]
    assert "ml.mna.min_deal_size" in _flat(section)
    assert "an unknown size cannot be shown to clear one" in _flat(section)


def test_no_number_in_the_deal_table_is_truncated(precedent_output):
    """Rich shrinks columns to fit, and a shrunk numeral is a wrong numeral.

    Roku's unaffected close is 143.66. Rendered as 143 with an ellipsis it still
    reads as a price, which is the failure mode a width bug produces and a test
    on the value alone would never catch.
    """
    table = _section(precedent_output, "Precedent transactions")
    rows = [ln for ln in table.splitlines() if re.search(r"^\d{4}-\d{2}-\d{2}", ln)]
    assert len(rows) == 9
    for row in rows:
        # The ellipsis is allowed in the acquirer name, which is prose, and
        # nowhere to the right of it, which is all numbers.
        after_names = row.split("pending")[-1].split("completed")[-1]
        assert "\u2026" not in after_names, f"a number was truncated: {row}"
    assert "143.66" in table

    # A crop with no ellipsis is worse than one with: nothing on the page says
    # it happened. The sub-vertical is the column rich reaches for last.
    assert "application_software" in table
    assert "media_entertainment" in table
    assert "application_softwa " not in table


def test_an_engine_refusal_prints_cleanly_rather_than_as_a_traceback(monkeypatch):
    monkeypatch.setattr(C, "EdgarClient", MergerFixtureClient)
    result = runner.invoke(C.app, ["precedents", "  "])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "No tickers given" in result.output


def test_the_run_echoes_the_assumptions_that_drove_it(precedent_output):
    flat = _flat(precedent_output)
    assert "Lookback 10 years" in flat
    assert "ml.mna.min_deal_size" in flat
    assert "Prices from csv" in flat
    assert "Statistics need 5 deals per column" in flat


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #


_TARGET_CACHE: dict[str, str] = {}


@pytest.fixture(scope="module")
def target_output() -> str:
    """One invocation of the screen, shared. The fit is the expensive part."""
    if "out" not in _TARGET_CACHE:
        result = runner.invoke(
            C.app,
            ["targets", "--dataset", str(MNA), "--as-of", AS_OF.isoformat(), "--top", "10"],
        )
        assert result.exit_code == 0, result.output
        _TARGET_CACHE["out"] = result.output
    return _TARGET_CACHE["out"]


def test_the_screen_prints_its_auc_and_its_baseline_before_the_first_name(
    target_output,
):
    """The score sits in the same block as the list, not in a footnote.

    A target screen is the output here most likely to be pasted into a deck, and
    a caveat on another page is a caveat nobody carries with them.
    """
    flat = _flat(target_output)
    assert "0.5685" in flat
    assert "0.5474" in flat
    # Twice: once in the block above the screen, and again inside the screen's
    # own header, so the number travels with the names if the page is cropped.
    assert flat.count("0.5685") >= 2
    screen = _section(target_output, "Acquisition target screen")
    assert "0.5685" in screen
    assert screen.index("0.5685") < screen.index("What put it on the list")


def test_the_baseline_is_the_size_sort_and_not_the_base_rate(target_output):
    """Beating a constant is not an achievement. Sorting by size costs one line."""
    flat = _flat(target_output)
    assert "size sort" in flat
    assert "smallest to largest, which costs one line of code" in flat


def test_the_lift_is_reported_as_inside_the_fold_noise(target_output):
    flat = _flat(target_output)
    assert "+0.0212" in flat
    assert "0.0910" in flat
    assert "the lift is INSIDE the fold-to-fold noise" in flat
    assert "beat_baseline is technically True" in flat
    assert "a different claim from the model working" in flat
    assert "the size sort is free" in flat


def test_the_blind_spot_is_printed_above_the_list(target_output):
    """The model is fundamentals only and the output must not let that pass unsaid."""
    assert "CANNOT SAY WHETHER CHEAP COMPANIES GET BOUGHT" in _flat(target_output)
    blind_at = target_output.index("What this model cannot see")
    list_at = target_output.index("Acquisition target screen")
    assert blind_at < list_at, "the blind spot landed under the list"
    assert "no valuation channel is in this ranking" in _flat(target_output)


def test_the_blind_spot_explains_why_rather_than_only_asserting_it(target_output):
    flat = _flat(target_output)
    assert "no public source serves price history for a delisted symbol" in flat
    assert "true, circular and useless" in flat


def test_the_ranked_list_is_printed_with_per_name_drivers(target_output):
    """A banker works a list, and a name on a list needs a sentence beside it."""
    section = _section(target_output, "Acquisition target screen")
    assert "What put it on the list" in section
    rows = [ln for ln in section.splitlines() if re.match(r"\s*\d+\s+\S", ln)]
    assert len(rows) >= 5
    assert "margin_gross" in section


def test_the_per_year_base_rate_is_printed_and_its_swing_stated(target_output):
    """It moves with the cycle, and a model scored across both can learn the calendar."""
    section = _section(target_output, "Base rate by year")
    for year in ("2019", "2020", "2021", "2022", "2023", "2024"):
        assert year in section
    flat = _flat(section)
    assert "2.05%" in flat and "4.53%" in flat
    assert "different worlds for technology M&A" in flat


def test_the_coefficients_that_change_sign_between_folds_are_counted(target_output):
    """Nine of fifteen. A coefficient that changes sign has been sampled, not measured."""
    section = _flat(_section(target_output, "Coefficient stability"))
    assert "9 of 15 coefficients change sign between folds" in section
    assert "has not been measured, it has been sampled" in section


def test_precision_at_k_is_reported_against_the_size_sort_on_the_same_dates(
    target_output,
):
    section = _section(target_output, "Precision and recall at k")
    assert "Prec. model" in section and "Prec. size" in section
    assert "is worth having and is not a result" in _flat(section)


def test_calibration_marks_the_thin_buckets_as_thin(target_output):
    """Ranking and calibration fail independently, and both are reported."""
    section = _section(target_output, "Calibration of the out-of-sample")
    assert "thin" in section
    assert "Ranking and calibration fail independently" in _flat(section)


def test_calibration_says_which_way_the_probabilities_miss(target_output):
    """A table of gaps is evidence. The direction of the miss is the finding.

    A class-weighted fit over a three percent base rate over-predicts, and the
    screen prints probabilities, so the command has to say the number on the
    list is a rank wearing a percentage sign.
    """
    section = _flat(_section(target_output, "Calibration of the out-of-sample"))
    assert "over-predict by more than five points" in section
    assert "ranks wearing a percentage sign" in section
    assert "do not quote a number off it as a likelihood" in section


def test_the_limitations_from_the_model_card_reach_the_page(target_output):
    flat = _flat(target_output)
    assert "Announcement is the label, not completion" in flat
    assert "A blocked deal is a positive" in flat
    assert "86 distinct companies were acquired in this sample" in flat


def test_the_verdict_rests_on_the_matrix_deal_count_not_the_observation_count(
    target_output,
):
    """86 deals reached the design matrix. The 284 positive rows are those again.

    Quoting the observation count beside a score the smaller sample produced
    would overstate the evidence by the companies the panel could not build.
    """
    flat = _flat(target_output)
    assert "86 distinct deals that reached the design matrix" in flat
    assert "284 positive observations" in flat
    assert "add no independent evidence" in flat


def test_the_run_echoes_the_sample_it_was_fitted_on(target_output):
    flat = _flat(target_output)
    assert "493 registrants" in flat
    assert "104 have left the filing record" in flat
    assert "107 announced deals" in flat
    assert "5 walk-forward folds, seed 7" in flat


def test_the_sub_vertical_filter_narrows_the_printed_list():
    result = runner.invoke(
        C.app,
        ["targets", "--dataset", str(MNA), "--as-of", AS_OF.isoformat(),
         "--top", "5", "--sub-vertical", "semiconductors"],
    )
    assert result.exit_code == 0, result.output
    body = _section(result.output, "Acquisition target screen")
    assert "semiconductors" in body
    assert "application_software" not in body


# --------------------------------------------------------------------------- #
# targets: the dataset loader
# --------------------------------------------------------------------------- #


def test_load_dataset_reads_the_panel_the_roster_and_the_deals():
    panel, universe, events, extras = C.load_dataset(MNA)
    assert len(universe.members) == 493
    assert len(events) == 107
    assert len(panel.rows) > 10_000
    assert panel.dates == sorted(panel.dates)


def test_load_dataset_keeps_the_departed_which_is_the_whole_experiment():
    """A roster whose departed count is zero is a survivor list.

    Every positive label is missing from one, and the model fitted on it is an
    all-negative classifier: a perfectly well-behaved object that measures
    nothing at all.
    """
    _panel, universe, _events, _extras = C.load_dataset(MNA)
    gone = universe.departed_members()
    assert len(gone) > 50
    assert 0.1 < len(gone) / len(universe.members) < 0.5


def test_load_dataset_carries_a_derived_extra_the_model_needs():
    """mna_net_cash_to_assets is not in FEATURE_NAMES and must not be dropped."""
    _panel, _universe, _events, extras = C.load_dataset(MNA)
    assert extras
    sample = next(iter(extras.values()))
    assert "mna_net_cash_to_assets" in sample


def test_an_empty_cell_stays_absent_rather_than_becoming_a_zero():
    """The model has a missing-share feature that depends on the difference."""
    panel, _universe, _events, _extras = C.load_dataset(MNA)
    with_gaps = [r for r in panel.rows if r.missing]
    assert with_gaps, "no row reported a gap, which cannot be true of this panel"
    row = with_gaps[0]
    assert row.values[row.missing[0]] is None


def test_a_missing_dataset_refuses_by_name_rather_than_raising(tmp_path):
    result = runner.invoke(C.app, ["targets", "--dataset", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "MissingDataError" in result.output


def test_a_dataset_missing_its_roster_names_the_file(tmp_path):
    (tmp_path / "events.json").write_text("[]")
    result = runner.invoke(C.app, ["targets", "--dataset", str(tmp_path)])
    assert result.exit_code == 1
    assert "universe.json" in result.output


def test_a_screen_date_the_panel_does_not_carry_is_refused_with_the_range(tmp_path):
    result = runner.invoke(
        C.app,
        ["targets", "--dataset", str(MNA), "--screen-on", "1999-01-01"],
    )
    assert result.exit_code == 1
    assert "ConfigError" in result.output
    assert "no rows dated 1999-01-01" in result.output
