"""The two commands that front the fitted models: the learned comp set and the screen.

Two kinds of test here and they answer two different questions, so they are kept
apart the way ``tests/ml/test_encoder.py`` keeps them apart.

**Does the command say the things it must not omit.** An invented universe of six
sectors, written out to the same recorded JSON the commands read in production,
because only an invented universe lets a test state which company is cold and
which is warm and then assert that the banner said so. Two companies in sector
five file no proxy of their own and are named only as somebody else's peers,
which is exactly the cold-start case: present in the panel, present in the
corpus, never a query the model was trained on. Nothing in that half of the file
is evidence about real filings.

**Does it work on filings.** ``tests/fixtures/warranted/observations.json.gz`` is
the recorded observation panel: ninety-four US TMT filers at quarter ends from
2021 to 2026, priced through ``build_observations`` against live SEC payloads and
daily closes on 2026-09-11, with the five hundred and forty-six company-dates the
engine refused carried beside them. The screen tests run against it, so the
assertions about what a screen prints are assertions about real companies.

Nothing here touches the network.
"""

from __future__ import annotations

import gzip
import json
import re
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from techval.commands_peers import (
    _label_sources,
    _render_thin_evidence,
    _tower_ablation,
    app,
)
from techval.config import Assumptions
from techval.ml.encoder import MIN_TRAIN_PAIRS, ablation_folds, build_dataset
from techval.ml.features import FEATURE_NAMES
from techval.ml.peer_labels import PeerGroup

FIXTURES = Path(__file__).parent / "fixtures"
REAL_PANEL = FIXTURES / "warranted" / "observations.json.gz"

runner = CliRunner()


# --------------------------------------------------------------------------- #
# An invented universe, written out as the artifacts the commands read
# --------------------------------------------------------------------------- #

# Disjoint vocabularies so the text tower has something to find. Real filings are
# nothing like this clean, which is what the real fixture below is for.
_VOCAB = {
    0: "spectrum cell sites postpaid churn handsets roaming coverage wireless subscribers",
    1: "observability monitoring telemetry traces dashboards ingest logs cloud platform",
    2: "wafer fabrication lithography nodes foundry analog yields packaging semiconductor",
    3: "towers ground leases backhaul colocation interconnection data centre kilowatts",
    4: "streaming titles subscribers content slate theatrical licensing studio catalogue",
    5: "merchants interchange authorisation settlement acquiring gateway payments volume",
}
_SECTORS = len(_VOCAB)
_PER_SECTOR = 6
_DATES = [date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1), date(2024, 1, 1)]

# Named by everybody regardless of sector: the popularity trap, present rather
# than hoped for.
_CELEBRITY = "S0C0"

# In the panel and in the corpus, named as peers by others, but never the filer
# of a group. That is precisely the cold-start case the banner has to catch.
_NEVER_FILES = {"S5C4", "S5C5"}

# Sector four lists late and files its first proxy in 2023, so an early fold's
# training window contains no group of theirs while a later test window does.
# Without a company like this every query in the sample is warm, ``cold`` on the
# evaluation is None, and the split the banner quotes would not exist to test.
_FIRST_FILES_IN = {f"S4C{i}": 2023 for i in range(_PER_SECTOR)}

# Its only proxy lands in June 2024, which is inside the training window and
# AFTER the last panel date the universe is embedded at. That gap is the reason
# the command compares a group against the filing cut rather than the fit date.
_FIRST_FILES_IN["S5C3"] = 2024

# Market capitalisation in USD millions, because the size gate reads a real one
# out of the panel and a feature drawn from a standard normal would put every
# company in the universe under a thousand million and empty the candidate list.
# S1C5 sits below the floor deliberately: it is the name the gate has to drop.
_MARKET_CAP = {
    f"S{s}C{i}": 4_000.0 + 1_500.0 * s + 400.0 * i
    for s in range(_SECTORS)
    for i in range(_PER_SECTOR)
}
_MARKET_CAP["S1C5"] = 300.0


def _ticker(sector: int, i: int) -> str:
    return f"S{sector}C{i}"


def _universe() -> list[str]:
    return [_ticker(s, i) for s in range(_SECTORS) for i in range(_PER_SECTOR)]


def _write_artifacts(root: Path) -> dict[str, Path]:
    """Write the three recorded inputs the peers command reads.

    Written out rather than built in memory because the readers are half of what
    is under test: a command that only ever sees objects a fixture handed it has
    not been shown to parse the file a user actually has.
    """
    rng = np.random.default_rng(11)
    root.mkdir(parents=True, exist_ok=True)

    centres = {s: rng.normal(0.0, 2.0, size=len(FEATURE_NAMES)) for s in range(_SECTORS)}
    rows = []
    for as_of in _DATES:
        for s in range(_SECTORS):
            for i in range(_PER_SECTOR):
                draw = centres[s] + rng.normal(0.0, 0.6, size=len(FEATURE_NAMES))
                values = {n: float(v) for n, v in zip(FEATURE_NAMES, draw)}
                values["scale_log_market_cap"] = float(
                    np.log(_MARKET_CAP[_ticker(s, i)])
                )
                missing: list[str] = []
                # One sector missing two features, so the missing-share
                # indicators are exercised rather than constant.
                if s == 2:
                    for name in ("margin_ebitda", "margin_fcf"):
                        values[name] = None
                        missing.append(name)
                rows.append(
                    {
                        "ticker": _ticker(s, i),
                        "as_of": as_of.isoformat(),
                        "knowledge_date": as_of.isoformat(),
                        "values": values,
                        "missing": missing,
                        "statement_date": None,
                        "error": None,
                    }
                )
    panel_path = root / "peer_panel.json"
    panel_path.write_text(json.dumps({"rows": rows, "random_seed": 7}))

    by_date: dict[str, dict] = {}
    for as_of in _DATES:
        companies = {}
        for s in range(_SECTORS):
            words = _VOCAB[s].split()
            for i in range(_PER_SECTOR):
                companies[_ticker(s, i)] = {
                    "item1": " ".join(rng.choice(words, size=120)),
                    "filed": as_of.isoformat(),
                    "accession": f"acc-{_ticker(s, i)}-{as_of}",
                }
        by_date[as_of.isoformat()] = companies
    text_path = root / "peer_item1.json"
    text_path.write_text(
        json.dumps(
            {
                "by_date": by_date,
                "entity_names": {t: f"{t} Holdings" for t in _universe()},
            }
        )
    )

    groups = []
    for as_of in _DATES:
        year = as_of.year
        for s in range(_SECTORS):
            for i in range(_PER_SECTOR):
                ticker = _ticker(s, i)
                if ticker in _NEVER_FILES:
                    continue
                if year < _FIRST_FILES_IN.get(ticker, 0):
                    continue
                peers = [_ticker(s, j) for j in range(_PER_SECTOR) if j != i]
                peers += [_ticker((s + 1) % _SECTORS, j) for j in range(4)]
                if ticker != _CELEBRITY:
                    peers.append(_CELEBRITY)
                groups.append(
                    {
                        "ticker": ticker,
                        "cik": 1000 + s * 100 + i,
                        "accession": f"{year}-{ticker}",
                        "filed": date(year, 3 + (i % 6), 1 + (s % 20)).isoformat(),
                        "fiscal_year": year - 1,
                        "peers": sorted(set(peers)),
                        "method": "fixture",
                        "confidence": 1.0,
                    }
                )
    groups_path = root / "peer_groups.json"
    groups_path.write_text(json.dumps({"groups": groups}))

    return {"groups": groups_path, "panel": panel_path, "text": text_path}


def _config(root: Path, name: str = "assumptions.yaml", **overrides) -> Path:
    """An assumptions file whose model cache points somewhere disposable.

    The default cache is the user's home directory. A suite that writes there is
    a suite that passes the second time for the wrong reason.
    """
    payload: dict = {
        "ml": {"cache_dir": str(root / "cache"), "random_seed": 7},
        "comps": {"peers": ["S1C1", "S1C2", "S3C0", "ZZZZ"]},
    }
    for key, value in overrides.items():
        payload.setdefault(key, {}).update(value)
    path = root / name
    path.write_text(yaml.safe_dump(payload))
    return path


def _flat(text: str) -> str:
    """Collapse the whitespace rich used to wrap a sentence across two lines.

    Every assertion below is about a sentence the command printed, and a table
    width is not. Flattening first keeps the tests from failing the day somebody
    widens a column.
    """
    return re.sub(r"\s+", " ", text)


def _tight(text: str) -> str:
    """All whitespace removed, for the assertions about a path.

    A temporary directory runs past the console width and rich folds it mid-word,
    so a path is the one thing that does not survive flattening to single spaces.
    """
    return re.sub(r"\s+", "", text)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("peers")
    paths = _write_artifacts(root)
    paths["config"] = _config(root)
    paths["root"] = root
    return paths


def _peers(paths: dict[str, Path], ticker: str, *extra: str):
    return runner.invoke(
        app,
        [
            "peers",
            ticker,
            "-c",
            str(paths["config"]),
            "--groups",
            str(paths["groups"]),
            "--panel",
            str(paths["panel"]),
            "--text",
            str(paths["text"]),
            "--k",
            "5",
            *extra,
        ],
    )


@pytest.fixture(scope="module")
def cold_run(synthetic):
    """One company that never filed a proxy, with the evaluation run."""
    result = _peers(synthetic, "S5C4", "--n", "6")
    assert result.exit_code == 0, result.output
    return _flat(result.output)


@pytest.fixture(scope="module")
def warm_run(synthetic):
    result = _peers(synthetic, "S1C1", "--n", "6")
    assert result.exit_code == 0, result.output
    return _flat(result.output)


# --------------------------------------------------------------------------- #
# What a ranked list is not allowed to hide
# --------------------------------------------------------------------------- #


def test_a_cold_target_is_labelled_cold_before_the_table(cold_run):
    """The whole point of the banner.

    S5C4 is in the panel and in the corpus and is named as a peer by its own
    sector, but it filed no proxy, so it was never a query the model trained on.
    A ranked list for it looks exactly like a ranked list for a company the model
    has seen a dozen times, and the only thing that separates them is this line.
    """
    assert "COLD START. S5C4 disclosed no peer group the model was fitted on" in cold_run
    assert "materially worse" in cold_run
    assert "WARM START" not in cold_run


def test_a_warm_target_says_the_overlap_with_its_own_proxy_is_recall(warm_run):
    """Agreement with a training label is not evidence, and the command says so."""
    assert "WARM START. S1C1 disclosed" in warm_run
    assert "fitted toward this company's own answer" in warm_run
    assert "treat the overlap as recall rather than as evidence" in warm_run


def test_warm_is_decided_by_the_filing_cut_and_not_by_the_embedding_date(synthetic):
    """The two dates on a fitted encoder are months apart and mean different things.

    Training pairs are selected on the date a proxy was FILED. The universe is
    then embedded at the newest PANEL date, which here is 1 January 2024. S5C3
    filed its only proxy that June, so it is inside the training window and after
    the embedding date, and comparing it against the wrong one of the two would
    label a company the model trained on as a company the model has never seen.
    """
    result = _peers(synthetic, "S5C3", "--no-evaluate")
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "embedded at 2024-01-01 on proxies filed through 2024-" in flat
    assert "WARM START. S5C3 disclosed 1 peer group(s)" in flat


def test_a_capped_training_window_moves_a_company_from_warm_to_cold(synthetic):
    """``--through`` is the point-in-time question asked of the model itself.

    S4C0 filed nothing before 2023. A fit that stops at the end of 2022 has
    therefore never seen it as a query, and the banner has to follow the window
    rather than the file. This is the same discipline the rest of the engine
    applies to a valuation dated in the past.
    """
    late = _flat(_peers(synthetic, "S4C0", "--no-evaluate", "--no-cache").output)
    early = _flat(
        _peers(
            synthetic, "S4C0", "--no-evaluate", "--no-cache", "--through", "2022-12-31"
        ).output
    )
    assert "WARM START. S4C0" in late
    assert "COLD START. S4C0" in early
    assert "proxies filed through 2022-12-31" in early


def test_a_training_window_with_too_few_pairs_refuses_to_fit(synthetic):
    result = _peers(
        synthetic, "S1C1", "--no-evaluate", "--no-cache", "--through", "2021-03-02"
    )
    assert result.exit_code == 1
    flat = _flat(result.output)
    assert "NotMeaningfulError" in flat
    assert "below the floor" in flat


def test_the_cold_and_warm_scores_both_reach_the_output(cold_run):
    """Both halves of the split, never the flattering one alone.

    The numbers are read off the model's own walk-forward evaluation rather than
    written here, so this asserts that both lines exist and that they are
    different numbers, not what either of them is.
    """
    cold = re.search(r"Cold start \(target never seen\) ([\d.]+)", cold_run)
    warm = re.search(r"Warm start \(target seen in training\) ([\d.]+)", cold_run)
    assert cold and warm
    assert float(cold.group(1)) != float(warm.group(1))


def test_without_the_evaluation_the_command_refuses_to_quote_a_score(synthetic):
    """An unmeasured claim is not made at all, rather than made quietly.

    Skipping the walk-forward pass is allowed, because it costs a minute on a
    real universe. Printing the ranking as though the model were equally good
    warm and cold is not, so the run that skipped it says which number is
    missing.
    """
    result = _peers(synthetic, "S1C1", "--no-evaluate", "--no-cache")
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "the warm and cold scores are unavailable" in flat
    assert "Cold start (target never seen)" not in flat


def test_every_baseline_is_printed_on_the_same_queries(warm_run):
    """The popularity prior is a degenerate solution and it has to be visible.

    A model can score respectably at retrieval by learning one number per
    candidate, how often anybody names it, and ignoring the query. Printing the
    encoder's score alone would leave that indistinguishable from comparability.
    """
    for method in (
        "popularity prior",
        "same sub-vertical",
        "size and growth",
        "fundamentals cosine",
        "text cosine",
    ):
        assert method in warm_run
    assert "Correlation with the popularity order" in warm_run


# --------------------------------------------------------------------------- #
# Label sources: how far outside its training distribution a ranked name sits
# --------------------------------------------------------------------------- #


#: Filler names that pad a group to ``MIN_PLAUSIBLE_PEERS``. A group of two is
#: refused by ``PeerGroup.count_plausible`` as a parse that found a fragment, so
#: a test that wrote one would be asserting on a group the fit never saw.
_PAD = [f"PAD{i}" for i in range(8)]


def _group(ticker: str, peers: list[str], *, year: int = 2023, usable: bool = True):
    return PeerGroup(
        ticker=ticker,
        cik=1,
        accession=f"{year}-{ticker}",
        filed=date(year, 3, 1),
        fiscal_year=year - 1 if usable else None,
        peers=[*peers, *_PAD],
        confidence=1.0 if usable else 0.0,
    )


def _groups(*groups) -> SimpleNamespace:
    """A dataset stand-in carrying only what ``_label_sources`` reads."""
    return SimpleNamespace(groups=list(groups))


def test_label_sources_counts_filers_and_not_mentions():
    """Four mentions from one committee is one source, which is the whole point.

    Procter and Gamble ranks fifth for Disney in the committed fit on four
    labelled appearances, and all four are Microsoft's proxy across four years.
    Counting mentions would report four and read as corroboration.
    """
    sources = _label_sources(
        _groups(
            _group("MSFT", ["PG", "DIS"], year=2021),
            _group("MSFT", ["PG", "DIS"], year=2022),
            _group("MSFT", ["PG", "DIS"], year=2023),
            _group("DIS", ["CMCSA", "NFLX"], year=2023),
        ),
        date(2026, 1, 1),
    )
    assert sources["PG"] == {"MSFT"}
    assert sources["DIS"] == {"MSFT", "DIS"}
    assert sources["CMCSA"] == {"DIS"}
    # A filer's own group counts as a source for itself and for nobody else.
    assert sources["MSFT"] == {"MSFT"}


def test_label_sources_honours_the_training_cut_and_the_usable_flag():
    dataset = _groups(
        _group("MSFT", ["PG"], year=2021),
        _group("ADI", ["PG"], year=2025),
        _group("ORCL", ["PG"], year=2022, usable=False),
    )
    assert _label_sources(dataset, date(2026, 1, 1))["PG"] == {"MSFT", "ADI"}
    # The 2025 group is outside an earlier window, and the unusable one never
    # became a training pair at all, so neither may count as supervision.
    assert _label_sources(dataset, date(2023, 1, 1))["PG"] == {"MSFT"}


def test_a_single_source_candidate_is_named_loudly_and_kept_at_its_rank(capsys):
    """The row stays where the model put it and carries the reason it is suspect.

    Dropping it would be the worse answer: a reader who cannot see that the model
    puts a consumer staples company fifth for a media conglomerate has been given
    a comp set with its most informative fact removed.
    """
    ranked = [("CMCSA", 0.9361), ("PG", 0.9161), ("NFLX", 0.9076)]
    sources = {"CMCSA": {"DIS", "NFLX"}, "PG": {"MSFT"}, "NFLX": {"DIS", "CMCSA", "MSFT"}}
    _render_thin_evidence(
        "DIS", ranked, sources, ["CMCSA", "PG", "NFLX", "T"], {"PG": "PROCTER & GAMBLE Co"}
    )
    out = _flat(capsys.readouterr().out)
    assert "OUT OF ITS DEPTH. PG (PROCTER & GAMBLE Co) is ranked 2 of 3 for DIS" in out
    assert "comes from one filer's proxy (MSFT)" in out
    assert "printed at its rank rather than dropped" in out
    # T is in the universe and in no group at all, so two of the four are thin.
    assert "2 of the 4 companies in the candidate universe" in out


def test_a_candidate_no_disclosed_group_mentions_at_all_says_so():
    ranked = [("XXXX", 0.5)]
    _render_thin_evidence("DIS", ranked, {}, ["XXXX"], {})


def test_a_ranking_every_name_of_which_is_supported_prints_no_banner(capsys):
    _render_thin_evidence(
        "DIS",
        [("CMCSA", 0.9)],
        {"CMCSA": {"DIS", "NFLX"}},
        ["CMCSA"],
        {},
    )
    assert capsys.readouterr().out == ""


def test_the_ranking_carries_a_label_source_count_for_every_name(warm_run):
    """The count is a column rather than a footnote, because it qualifies a row."""
    assert "Label sources" in warm_run


# The committed TMT peer artifacts: 173 companies, 220 disclosed groups read from
# DEF 14A proxies on 2026-09-11. Fitting on them costs about eight seconds, which
# is why only one test runs against them, and it is the one that has to: the
# result it pins is the measured cross-sector leak rather than an invented
# universe's version of it.
TMT_GROUPS = FIXTURES / "peer_groups_tmt.json"
TMT_PANEL = FIXTURES / "peer_panel_tmt.json"
TMT_TEXT = FIXTURES / "peer_item1_tmt.json"


@pytest.fixture(scope="module")
def disney_run(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("tmt")
    result = runner.invoke(
        app,
        [
            "peers",
            "DIS",
            "-c",
            str(_config(root)),
            "--groups",
            str(TMT_GROUPS),
            "--panel",
            str(TMT_PANEL),
            "--text",
            str(TMT_TEXT),
            "--no-evaluate",
        ],
    )
    assert result.exit_code == 0, result.output
    return _flat(result.output)


def test_procter_and_gamble_above_netflix_for_disney_is_flagged(disney_run):
    """The finding, on the filings it was found on.

    ``peers DIS`` ranks PROCTER & GAMBLE fifth at 0.9161, above NETFLIX at
    0.9076. That is a real output of a model fitted on 220 compensation peer
    groups, and it discredits the other seven rows unless the reader is told
    something about it. What the reader is told is the measurement: every
    labelled appearance P&G has in this fit comes from Microsoft's proxy.

    The ranking is not touched. The row is still fifth and still at 0.9161,
    because the argument for capping or filtering it is an argument for hiding
    the single most useful thing this page can say about the model's reach.
    """
    assert "PROCTER & GAMBLE Co 0.9161 1" in disney_run
    assert "NETFLIX INC 0.9076 13" in disney_run
    assert "OUT OF ITS DEPTH. PG (PROCTER & GAMBLE Co) is ranked 5 of 8 for DIS" in disney_run
    assert "comes from one filer's proxy (MSFT)" in disney_run
    assert "34 of the 173 companies in the candidate universe" in disney_run


def test_disney_is_warm_so_the_cold_start_flag_is_not_the_one_that_catches_this(disney_run):
    """The brief's hypothesis, checked and wrong in its first half.

    The cold-start signal was the obvious candidate for catching the P&G result
    and it does not fire: Disney disclosed two peer groups inside the training
    window and is WARM. The target is inside the distribution and the CANDIDATE
    is not, which is why the diagnostic had to be per-candidate.
    """
    assert "WARM START. DIS disclosed 2 peer group(s)" in disney_run
    assert "COLD START" not in disney_run
    assert "6 distinct filer(s) named DIS" in disney_run


def test_the_target_gets_its_own_source_count_beside_warm_or_cold(cold_run):
    """Two different questions, and printing one without the other is half an answer.

    Warm and cold is about whether the target was ever a QUERY. S5C4 was not, and
    is cold. It is still named as a peer by four filers in its own sector, so the
    labels do constrain where it sits, which the cold banner alone does not say.
    """
    assert re.search(r"\d+ distinct filer\(s\) named S5C4 in a usable disclosed group", cold_run)
    assert "only the filer of a group was ever a query, while anybody can name anybody" in cold_run


def test_the_hand_written_set_is_shown_with_where_the_model_put_each_name(warm_run):
    """The disagreement is the output worth reading, so it is printed both ways."""
    assert "The hand-written comp set, and where the model put it" in warm_run
    assert "The model adds" in warm_run
    assert "The model drops" in warm_run


def test_a_hand_written_peer_the_model_never_saw_is_named_rather_than_dropped(warm_run):
    """ZZZZ is in comps.peers and in no panel row.

    Silently omitting it would let a reader believe the model considered and
    rejected a name it could not encode at all.
    """
    assert "ZZZZ" in warm_run
    assert "not in the fitted universe" in warm_run


def test_the_target_itself_is_not_ranked_against_itself(synthetic):
    result = _peers(synthetic, "S1C1", "--no-evaluate")
    flat = _flat(result.output)
    assert "S1C1 - - the target itself" in flat


def test_a_company_outside_the_fitted_universe_is_refused_not_approximated(synthetic):
    """The engine's cardinal rule, at the command boundary."""
    result = _peers(synthetic, "NOPE", "--no-evaluate")
    assert result.exit_code == 1
    flat = _flat(result.output)
    assert "MissingDataError" in flat
    assert "this company was not in the fitted universe" in flat


def test_the_size_gate_drops_the_name_below_the_floor_and_says_it_is_on(synthetic):
    """The gate belongs in the product and not in the measurement, and both are said.

    S1C5 sits at 300mm against a floor of 1,000mm and is otherwise a sector peer
    of the target, so it is the one name whose presence separates a gated list
    from an ungated one. A banker does not put it in this comp set however
    similar the prose, and a measurement that did would be scoring the band
    rather than the model, which is why the evaluation below runs ungated.
    """
    on = _flat(_peers(synthetic, "S1C1", "--no-evaluate", "--n", "6").output)
    off = _flat(
        _peers(synthetic, "S1C1", "--no-evaluate", "--no-size-gate", "--n", "6").output
    )
    ranked_on = re.findall(r"\d+ (S\dC\d) S\dC\d Holdings", on)
    ranked_off = re.findall(r"\d+ (S\dC\d) S\dC\d Holdings", off)
    assert ranked_on and ranked_off
    assert "S1C5" not in ranked_on
    assert "S1C5" in ranked_off
    assert "Size gate on" in on
    assert "runs UNGATED" in on
    assert "Size gate on" not in off


def test_the_assumptions_that_drove_the_ranking_are_echoed(warm_run):
    for label in (
        "ml.peers.text_weight",
        "ml.peers.max_size_ratio",
        "ml.random_seed",
        "comps.peers",
    ):
        assert label in warm_run


def test_the_same_inputs_give_the_same_ranking(synthetic):
    """Seeded from ml.random_seed, so a rerun is a rerun and not a new opinion."""
    first = _flat(_peers(synthetic, "S1C1", "--no-evaluate", "--no-cache").output)
    second = _flat(_peers(synthetic, "S1C1", "--no-evaluate", "--no-cache").output)
    pattern = r"\d+ (S\dC\d) S\dC\d Holdings ([\d.]+)"
    assert re.findall(pattern, first) == re.findall(pattern, second)
    assert re.findall(pattern, first)


def test_the_fit_is_cached_and_the_second_run_reads_it(synthetic):
    """Named in the assumptions echo, because a cached number is still a number.

    A reader who cannot tell a fresh fit from a cached one cannot tell whether
    changing an input changed the answer.
    """
    first = _peers(synthetic, "S1C1", "--no-evaluate", "--refit")
    second = _peers(synthetic, "S1C1", "--no-evaluate")
    assert "fit fitted on this run" in _flat(first.output)
    assert "fit read from cache" in _flat(second.output)


# --------------------------------------------------------------------------- #
# The tower ablation
# --------------------------------------------------------------------------- #


def test_the_ablation_refits_without_each_tower_and_reports_all_three(synthetic):
    """The full model and one row per tower, with the fold spread beside each.

    What the numbers say on an invented universe is not evidence about anything,
    so this asserts the shape rather than the answer. The verdict rule that reads
    those numbers is tested on its own below, where the inputs can be stated.
    """
    # Two folds rather than the default five: three configurations refitted on
    # every fold is the most expensive thing in this file, and the shape under
    # test does not change with the number of splits.
    config = _config(
        synthetic["root"], name="two_folds.yaml", ml={"walk_forward_folds": 2}
    )
    result = runner.invoke(
        app,
        [
            "peers",
            "S1C1",
            "-c",
            str(config),
            "--groups",
            str(synthetic["groups"]),
            "--panel",
            str(synthetic["panel"]),
            "--text",
            str(synthetic["text"]),
            "--ablate",
            "--no-evaluate",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "What each tower is worth, refitted without it" in flat
    for row in ("all features", "without text", "without fundamentals"):
        assert row in flat
    assert "Fold sd" in flat


def test_a_tower_is_only_shown_to_help_when_it_clears_the_fold_noise(capsys):
    """Damage is judged against the fold standard deviation, not against zero.

    A tower whose absence costs less than the run-to-run spread has not been
    shown to help, and a ratio is printed rather than a yes so that 1.01 is
    distinguishable from 3.0. All three cases are here because all three happen:
    on the committed TMT universe the text tower clears the noise by a hair and
    the fundamentals tower does not clear it at all.
    """
    from techval.commands_peers import _render_ablation

    frame = pd.DataFrame(
        [
            {"group": "all features", "n_columns": 118, "score": 0.42, "damage": 0.0,
             "fold_sd": 0.08, "n_test": 100},
            {"group": "without text", "n_columns": 116, "score": 0.36, "damage": 0.064,
             "fold_sd": 0.064 / 1.01, "n_test": 100},
            {"group": "without fundamentals", "n_columns": 2, "score": 0.39,
             "damage": 0.039, "fold_sd": 0.069, "n_test": 100},
        ]
    )
    _render_ablation(frame)
    flat = _flat(capsys.readouterr().out)
    assert "without text 116 0.3600 0.0640" in flat
    assert "1.01x the fold noise: shown to help" in flat
    assert "0.57x the fold noise: NOT shown to help" in flat

    _render_ablation(
        pd.DataFrame(
            [
                {"group": "all features", "n_columns": 4, "score": 0.3, "damage": 0.0,
                 "fold_sd": 0.05, "n_test": 10},
                {"group": "without text", "n_columns": 2, "score": 0.36,
                 "damage": -0.06, "fold_sd": 0.05, "n_test": 10},
            ]
        )
    )
    assert "removing it did not hurt: this tower is not helping" in _flat(
        capsys.readouterr().out
    )


def test_a_run_without_the_ablation_says_the_tower_question_is_unanswered(warm_run):
    """The baseline table is about raw input spaces and is not a tower ablation.

    They are different questions and on the real universe they give different
    answers, so a run that answered only the first must not read as though it
    answered both.
    """
    assert "The tower ablation was not run" in warm_run
    assert "does not say which trained tower" in warm_run


def test_the_ablations_default_folds_now_clear_the_training_floor(real_dataset):
    """The defect this test used to pin is fixed in the module it belonged to.

    Cut into equal blocks of dates with no floor on the training window, the
    earliest walk-forward fold over the committed proxies held nine disclosed
    pairs, because the first filing season in the window is one filer.
    ``_train_encoder`` refused to fit a contrastive model on nine relationships,
    which was correct, and the refusal took the whole ablation with it rather
    than costing one fold. ``ablate_towers`` now asks for the same floor
    ``_tower_ablation`` below asks for, so the default path and the command's
    path cut the same folds. The fold shape is asserted rather than the
    ablation run, because the run is exercised in tests/ml/test_encoder.py and
    costs half a minute.
    """
    folds = ablation_folds(real_dataset, Assumptions())
    assert len(folds) >= 2
    pairs = [e.filed for e in real_dataset.examples]
    for fold in folds:
        cut = fold.test_start - timedelta(days=fold.embargo_days)
        assert sum(1 for d in pairs if d < cut) >= MIN_TRAIN_PAIRS


def test_the_folds_this_command_supplies_clear_the_training_floor(real_dataset):
    """The fix, which is the floor ``walk_forward_folds`` already takes.

    Asserted on the folds rather than by running the ablation, because running it
    on the real universe costs a minute and the property under test is the
    training window rather than the result.
    """
    import inspect

    source = inspect.getsource(_tower_ablation)
    assert "min_train=MIN_TRAIN_PAIRS" in source

    from techval.ml.evaluation import walk_forward_folds
    from techval.ml.encoder import EMBARGO_DAYS

    folds = walk_forward_folds(
        [e.filed for e in real_dataset.examples],
        Assumptions().ml.walk_forward_folds,
        min_train=MIN_TRAIN_PAIRS,
        embargo_days=EMBARGO_DAYS,
    )
    assert min(f.n_train for f in folds) >= MIN_TRAIN_PAIRS


@pytest.fixture(scope="module")
def real_dataset():
    """The TMT seed universe's disclosed peer groups, read from proxies 2026-09-11."""
    from datetime import date as _date

    from techval.ml.features import FeaturePanel, FeatureRow
    from techval.ml.peer_labels import PeerGroup
    from techval.ml.text import TextCorpus

    for name in ("peer_groups_tmt.json", "peer_panel_tmt.json", "peer_item1_tmt.json"):
        if not (FIXTURES / name).exists():
            pytest.skip(f"{name} is not committed")

    panel_payload = json.loads((FIXTURES / "peer_panel_tmt.json").read_text())
    panel = FeaturePanel(
        rows=[
            FeatureRow(
                ticker=r["ticker"],
                as_of=_date.fromisoformat(r["as_of"]),
                knowledge_date=_date.fromisoformat(r["knowledge_date"])
                if r["knowledge_date"]
                else None,
                values=r["values"],
                missing=r["missing"],
                statement_date=_date.fromisoformat(r["statement_date"])
                if r["statement_date"]
                else None,
                error=r["error"],
            )
            for r in panel_payload["rows"]
        ],
        random_seed=7,
    )
    item1 = json.loads((FIXTURES / "peer_item1_tmt.json").read_text())
    names = item1["entity_names"]
    corpora = {}
    for iso, companies in item1["by_date"].items():
        tickers = sorted(companies)
        corpora[_date.fromisoformat(iso)] = TextCorpus(
            tickers=tickers,
            documents=[companies[t]["item1"] for t in tickers],
            as_of_by_ticker={
                t: _date.fromisoformat(companies[t]["filed"]) for t in tickers
            },
            source_accessions={t: companies[t]["accession"] for t in tickers},
            entity_names={t: names[t] for t in tickers if names.get(t)},
        )
    groups = [
        PeerGroup(
            ticker=g["ticker"],
            cik=g["cik"],
            accession=g["accession"],
            filed=_date.fromisoformat(g["filed"]),
            fiscal_year=g["fiscal_year"],
            peers=g["peers"],
            method=g["method"],
            selection_criteria=g.get("selection_criteria"),
            confidence=g["confidence"],
        )
        for g in json.loads((FIXTURES / "peer_groups_tmt.json").read_text())["groups"]
    ]
    return build_dataset(groups, panel, corpora)


# --------------------------------------------------------------------------- #
# Reading the recorded artifacts
# --------------------------------------------------------------------------- #


def test_a_missing_artifact_names_the_file_and_the_recorder(tmp_path):
    config = _config(tmp_path)
    result = runner.invoke(
        app,
        ["peers", "AAA", "-c", str(config), "--groups", str(tmp_path / "nope.json")],
    )
    assert result.exit_code == 1
    assert "MissingDataError" in _flat(result.output)
    tight = _tight(result.output)
    assert "nope.json" in tight
    assert "record.py" in tight


def test_a_file_that_is_not_the_artifact_says_which_key_is_missing(tmp_path, synthetic):
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"something": "else"}))
    result = runner.invoke(
        app,
        [
            "peers",
            "S1C1",
            "-c",
            str(synthetic["config"]),
            "--groups",
            str(wrong),
            "--panel",
            str(synthetic["panel"]),
            "--text",
            str(synthetic["text"]),
        ],
    )
    assert result.exit_code == 1
    assert "'groups'" in _flat(result.output)


def test_a_non_positive_multiple_is_refused_rather_than_logged(tmp_path):
    """The panel is fitted in logs, and a zero multiple has no logarithm.

    Substituting anything for it would put a company in the ranking on a number
    nobody recorded, which is the one thing this engine will not do.
    """
    path = tmp_path / "obs.json"
    path.write_text(
        json.dumps(
            {
                "target": "ev_revenue",
                "observations": [
                    {
                        "ticker": "AAA",
                        "as_of": "2025-03-31",
                        "sub_vertical": "application_software",
                        "multiple": 0.0,
                        "enterprise_value": 100.0,
                        "equity_value": 90.0,
                        "denominator": 10.0,
                        "statement_date": None,
                        "basis_factor": 1.0,
                        "features": {},
                    }
                ],
                "skips": [],
            }
        )
    )
    result = runner.invoke(app, ["screen", "--panel", str(path)])
    assert result.exit_code == 1
    flat = _flat(result.output)
    assert "has no logarithm" in flat
    assert "recording error rather than a cheap company" in flat


# --------------------------------------------------------------------------- #
# The screen, on the recorded panel of real filers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def screen_run(tmp_path_factory):
    if not REAL_PANEL.exists():
        pytest.skip("the recorded observation panel is not committed")
    root = tmp_path_factory.mktemp("screen")
    result = runner.invoke(
        app,
        [
            "screen",
            "-c",
            str(_config(root)),
            "--panel",
            str(REAL_PANEL),
            "--n",
            "5",
            "--date",
            "2022-06-30",
        ],
    )
    assert result.exit_code == 0, result.output
    return _flat(result.output)


def test_the_reflexivity_limit_is_printed_before_the_ranking(screen_run):
    """The sentence that decides what the whole output means.

    A residual says a company is priced unlike its characteristics suggest the
    market prices characteristics. A reader who takes it for a valuation will
    read a positive residual as a sell, and it is not one.
    """
    limit = screen_run.index("It cannot say the market is wrong")
    ranking = screen_run.index("residual in within-date standard deviations")
    assert limit < ranking


def test_the_differenced_correlation_is_printed_beside_the_pooled_one(screen_run):
    """The number the brief for this module exists to stop anybody burying.

    Pooled, the model scores in the high sevens. Differenced against each
    company's own previous observation it scores four hundredths. Nearly all of
    the first number is company identity inherited from history, and the two
    belong in the same table.
    """
    pooled = re.search(r"Pooled rank correlation \+([\d.]+)", screen_run)
    change = re.search(
        r"Differenced against the company's own prior read \+([\d.]+)", screen_run
    )
    assert pooled and change
    assert float(pooled.group(1)) > 0.5
    assert float(change.group(1)) < 0.1
    assert "almost all of the headline is the level inherited from history" in screen_run


def test_the_pooled_score_is_shown_against_the_baseline_it_had_to_beat(screen_run):
    assert "comps.py OLS refit on the date, sub-vertical peers" in screen_run
    assert "Lift" in screen_run
    assert "fold sd" in screen_run


def test_a_refused_row_is_reported_by_name_with_its_reason(screen_run):
    """SHOP on 2022-06-30 is a share basis the engine would not resolve.

    Its filed share count and the price series may be on different split bases,
    so the equity value could be wrong by the split ratio. The row is refused
    rather than guessed at, and a screen that showed neither the name nor the
    reason would read as though the universe were ninety-four names on every
    date.
    """
    assert "Refused on 2022-06-30" in screen_run
    assert "SHOP share_basis_unresolved" in screen_run
    assert "may be on different bases" in screen_run


def test_the_whole_panel_is_summarised_by_what_it_lost(screen_run):
    """The count that says whether this is the universe or a corner of it."""
    assert "The whole panel, by outcome" in screen_run
    for category in ("observed", "not_built", "share_basis_unresolved"):
        assert category in screen_run


def test_the_residuals_are_out_of_sample_and_the_screen_says_so(screen_run):
    assert "Every residual is out of sample" in screen_run
    assert "the fold that produced it did not train on the row" in screen_run


def test_the_scale_a_residual_is_read_against_is_printed(screen_run):
    """A residual of 0.25 against a within-date sd of 0.55 is an argument.

    Against 0.15 it would be a trade. The z column and the standard deviation
    behind it travel together for that reason.
    """
    assert "Within-date standard deviation (log)" in screen_run
    assert "Share of variance that is the calendar" in screen_run


def test_the_screen_echoes_the_assumptions(screen_run):
    for label in (
        "ml.warranted.demean_by_date",
        "ml.warranted.min_train_observations",
        "ml.walk_forward_folds",
        "trained through",
    ):
        assert label in screen_run


def test_one_companys_own_read_can_be_pulled_out_of_the_screen(tmp_path):
    if not REAL_PANEL.exists():
        pytest.skip("the recorded observation panel is not committed")
    result = runner.invoke(
        app,
        [
            "screen",
            "-c",
            str(_config(tmp_path)),
            "--panel",
            str(REAL_PANEL),
            "--n",
            "3",
            "--ticker",
            "DDOG",
        ],
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "DDOG trades at" in flat
    assert "a statement about relative pricing, not about value" in flat


def test_a_sub_vertical_the_date_does_not_carry_is_refused_with_what_it_does(tmp_path):
    if not REAL_PANEL.exists():
        pytest.skip("the recorded observation panel is not committed")
    result = runner.invoke(
        app,
        [
            "screen",
            "-c",
            str(_config(tmp_path)),
            "--panel",
            str(REAL_PANEL),
            "--sub-vertical",
            "shipbuilding",
        ],
    )
    assert result.exit_code == 1
    flat = _flat(result.output)
    assert "No 'shipbuilding' name has an out-of-sample read" in flat
    assert "semiconductors" in flat


def test_a_panel_below_the_floor_reports_nothing_rather_than_noise(tmp_path):
    """A hundred and fifty observations is not a statistical nicety.

    Below it the cross-section cannot support a dozen fundamental coefficients
    and eleven sub-vertical dummies, and what comes out is the model's own noise
    with a decimal point on it.
    """
    payload = json.load(gzip.open(REAL_PANEL, "rt"))
    keep = [o for o in payload["observations"]][:40]
    path = tmp_path / "thin.json"
    path.write_text(json.dumps({**payload, "observations": keep, "skips": []}))
    result = runner.invoke(app, ["screen", "-c", str(_config(tmp_path)), "--panel", str(path)])
    assert result.exit_code == 1
    flat = _flat(result.output)
    assert "NotMeaningfulError" in flat
    assert "min_train_observations" in flat


def test_the_recorded_target_wins_over_the_configured_one(tmp_path):
    """A multiple cannot be changed after the rows were priced.

    The panel records which multiple every enterprise value was divided by. An
    assumptions file naming a different one is a disagreement worth printing and
    not a licence to relabel the column.
    """
    payload = json.load(gzip.open(REAL_PANEL, "rt"))
    path = tmp_path / "thin.json"
    path.write_text(json.dumps({**payload, "observations": payload["observations"][:5]}))
    config = _config(tmp_path, ml={"warranted": {"target": "ev_ebitda"}})
    result = runner.invoke(app, ["screen", "-c", str(config), "--panel", str(path)])
    flat = _flat(result.output)
    assert "The recorded panel is a ev_revenue panel" in flat
    assert "The panel wins" in flat
