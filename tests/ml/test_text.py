"""Tests for the text tower.

Two corpora, and they answer different questions.

The synthetic one is six invented filers in two invented sectors, written so
that one specific thing is true: a media company names a telecom company
repeatedly in its own filing. That is the shape of the name leak, and inventing
it is the only way to test the fix, because on a corpus of real filings each
company's name appears mostly in its own document, min_df removes it, and the
leak never fires. It fires constantly on a real universe of several hundred
filers where competitors, customers and acquirers are named across documents.

The real one is six committed Item 1 excerpts, four infrastructure and security
software companies and two wireless carriers, pulled from EDGAR on 2026-09-11
and frozen. It answers whether the thing works at all: whether a word count over
a business description puts a carrier next to a carrier.

Nothing here touches the network.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from techval.errors import ConfigError, MissingDataError
from techval.ml.text import (
    TextCorpus,
    baseline_neighbours,
    build_corpus,
    deglue,
    fit_text_features,
    item1_window,
    name_variants,
    nearest_neighbours,
    scrub,
    transform,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

SOFTWARE = ("DDOG", "CRWD", "ZS", "MDB")
CARRIERS = ("VZ", "TMUS")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def excerpts() -> dict:
    """Committed Item 1 excerpts, retrieved 2026-09-11. See the file header."""
    payload = json.loads((FIXTURES / "item1_excerpts.json").read_text())
    return payload["companies"]


@pytest.fixture
def real_corpus(excerpts) -> TextCorpus:
    tickers = list(excerpts)
    return TextCorpus(
        tickers=tickers,
        documents=[excerpts[t]["item1_excerpt"] for t in tickers],
        as_of_by_ticker={t: date.fromisoformat(excerpts[t]["filed"]) for t in tickers},
        source_accessions={t: excerpts[t]["accession"] for t in tickers},
        entity_names={t: excerpts[t]["entity_name"] for t in tickers},
    )


@pytest.fixture
def ml_assumptions(assumptions, tmp_path):
    """Assumptions with the model cache pointed somewhere disposable.

    The default cache directory is the user's home. A test suite that writes
    there is a test suite that passes on the second run for the wrong reason.
    """
    assumptions.ml.cache_dir = str(tmp_path / "mlcache")
    return assumptions


# The invented universe: three carriers and three media companies, each written
# in the register its own sector uses. Zenith is the leak. A streaming company
# names it fourteen times, which is unremarkable in a filing where a carrier is
# simultaneously your largest distribution partner and your largest competitor,
# and which a word count reads as evidence that a wireless carrier and a
# streaming service are the same kind of business.
_ZEN = (
    "Zenith is a wireless carrier. Zenith owns licensed spectrum and Zenith operates "
    "cell sites on that spectrum. Zenith sells postpaid phone plans to households. "
    "Zenith reports wireless subscribers, and Zenith reports the churn of those "
    "wireless subscribers, every quarter. Zenith raises average revenue per user by "
    "moving households onto unlimited plans. Zenith sells handsets in Zenith stores. "
    "Zenith extends coverage through roaming where Zenith spectrum is thin. Zenith "
    "spends capital on spectrum and on cell sites. Zenith bills households monthly."
)
_PEAK = (
    "Peak is a mobile network operator selling prepaid airtime. Peak buys wholesale "
    "capacity rather than owning licensed spectrum of its own. Peak moves handsets "
    "through independent dealers. Peak accounts top up in advance and Peak accounts "
    "lapse quickly, so the churn of Peak accounts runs high. Peak coverage rests on "
    "wholesale agreements with larger wireless carriers."
)
_ATL = (
    "Atlas owns towers and leases space on those towers to wireless carriers. Atlas "
    "builds fiber backhaul routes between towers. Atlas signs long lease terms with "
    "carriers extending coverage into rural counties. Atlas spends capital on towers "
    "and on fiber, and Atlas earns lease income rather than subscriber income."
)
_ORN = (
    "Orion runs a streaming service. Orion licenses film and television titles into a "
    "content library, and Orion amortises that content spend over the life of each "
    "title. Orion sells video advertising against the same content library. Orion "
    "membership churn falls as the content library deepens. Orion competes with Zenith "
    "for viewer attention. Zenith bundles streaming into its own offering, and Zenith "
    "markets that bundle hard. Zenith is a distribution partner of Orion and Zenith is "
    "a competitor of Orion. Payments from Zenith are material to Orion. Zenith may not "
    "renew the Zenith agreement. Zenith has scale and Zenith has capital. Orion watches "
    "Zenith pricing. Zenith reaches households that Orion cannot reach on its own. "
    "Zenith remains the largest single counterparty of Orion."
)
_NVA = (
    "Nova owns broadcast television stations. Nova sells video advertising to national "
    "brands. Nova collects affiliate fees from distributors. Nova licenses film and "
    "television titles from studios into a content library. Nova advertising revenue "
    "moves with the political cycle."
)
_VST = (
    "Vista is a film and television studio. Vista produces original titles and Vista "
    "licenses those titles to distributors. Vista amortises content spend over the life "
    "of each title. Vista sells titles into a streaming service and into theatrical "
    "windows. Vista content library income comes from licensing rather than from "
    "video advertising."
)

SYNTHETIC = {
    "ZEN": ("Zenith Wireless Holdings, Inc.", _ZEN),
    "PEAK": ("Peak Mobile Networks, Inc.", _PEAK),
    "ATL": ("Atlas Telecom Group, Inc.", _ATL),
    "ORN": ("Orion Streaming Media, Inc.", _ORN),
    "NVA": ("Nova Broadcast Entertainment, Inc.", _NVA),
    "VST": ("Vista Content Studios, Inc.", _VST),
}


@pytest.fixture
def synthetic_corpus() -> TextCorpus:
    tickers = list(SYNTHETIC)
    return TextCorpus(
        tickers=tickers,
        documents=[SYNTHETIC[t][1] for t in tickers],
        as_of_by_ticker={t: date(2026, 3, 1) for t in tickers},
        source_accessions={t: f"0000000000-26-0000{i:02d}" for i, t in enumerate(tickers)},
        entity_names={t: SYNTHETIC[t][0] for t in tickers},
    )


class FakeClient:
    """A filing client backed by the committed excerpts. Never fetches anything."""

    def __init__(self, excerpts: dict, *, text_by_accession: dict | None = None) -> None:
        self._excerpts = excerpts
        self._text = text_by_accession or {}
        self.text_calls: list[str] = []

    def filings(self, ticker, forms=("10-K",), since=None, limit=20):
        row = self._excerpts.get(ticker.upper())
        if row is None:
            return []
        out = [
            {
                "accession": row["accession"],
                "filed": date.fromisoformat(row["filed"]),
                "form": "10-K",
                "document": row["document"],
                "period": row["period"],
            }
        ]
        # A second, older filing, so "latest on or before as_of" has something to
        # choose between rather than something to accept.
        out.append(
            {
                "accession": row["accession"][:-1] + "0",
                "filed": date.fromisoformat(row["filed"]).replace(
                    year=date.fromisoformat(row["filed"]).year - 1
                ),
                "form": "10-K",
                "document": row["document"],
                "period": row["period"],
            }
        )
        return [f for f in out if since is None or f["filed"] >= since][:limit]

    def filing_text(self, ticker, filing):
        self.text_calls.append(ticker.upper())
        if filing["accession"] in self._text:
            return self._text[filing["accession"]]
        row = self._excerpts[ticker.upper()]
        # Wrapped so the fallback extractor has a heading and a terminator to
        # find, which is what a real primary document gives it.
        return (
            "TABLE OF CONTENTSPART I.Item 1.Business5Item 1A.Risk Factors13\n"
            + row["item1_excerpt"]
            + "\nItem 1A. Risk FactorsOur operations are subject to risks.\n"
        )

    def submissions(self, ticker):
        return {"name": self._excerpts[ticker.upper()]["entity_name"]}


class DeepFilingClient:
    """A filer with a long history, truncated the way the real client truncates.

    ``EdgarClient.filings`` sorts newest first and cuts at ``limit`` before
    anything downstream gets to filter by date. Reproducing that ordering here
    is the whole point of this fake: a client that returned everything would let
    a point-in-time bug pass, because the bug is not in the filtering, it is in
    what the filtering never sees.
    """

    def __init__(self, *, first_year: int = 2008, last_year: int = 2026) -> None:
        self.years = list(range(last_year, first_year - 1, -1))
        self.limits_seen: list[int] = []

    def filings(self, ticker, forms=("10-K",), since=None, limit=20):
        self.limits_seen.append(limit)
        out = [
            {
                "accession": f"0000000000-{y % 100:02d}-000001",
                "filed": date(y, 2, 20),
                "form": "10-K",
                "document": f"x-{y}1231.htm",
                "period": f"{y - 1}-12-31",
            }
            for y in self.years
        ]
        if since is not None:
            out = [f for f in out if f["filed"] >= since]
        return out[:limit]

    def filing_text(self, ticker, filing):
        return (
            "Item 1. BusinessOverview We operate a wireless network. "
            + ("We sell postpaid phone plans to households. " * 80)
            + "Item 1A. Risk Factors"
        )

    def submissions(self, ticker):
        return {"name": "Deep History Communications Inc"}


# --------------------------------------------------------------------------- #
# Finding Item 1
# --------------------------------------------------------------------------- #


def test_item1_window_takes_the_heading_not_the_contents_or_a_cross_reference():
    """The fixture is the trap: the string "Item 1. Business" occurs three times.

    Once in the table of contents, once as the heading, and once inside the risk
    factors as "described under Part I: Item 1. Business". The last of those is
    followed by a later mention of Item 1A, so the naive rule of taking the last
    occurrence returns a window made almost entirely of risk factors.
    """
    text = (FIXTURES / "tenk_excerpt_DDOG.txt").read_text()
    window = item1_window(text, ticker="DDOG")

    assert window.lower().startswith("item 1. business")
    assert "observability" in window.lower()
    # The contents entry would have produced a few dozen characters.
    assert len(window) > 5_000
    # The risk factors sit immediately after the heading this must stop at.
    assert "Risks Associated with our Growth" not in window
    assert "trading price of our Class A common stock could decline" not in window
    # And the whole of the spliced fixture is longer than what came back, which
    # is the point: most of a 10-K is not the business description.
    assert len(window) < len(text)


def test_item1_window_raises_rather_than_returning_the_filing():
    with pytest.raises(MissingDataError) as exc:
        item1_window("A filing with no item numbering at all. " * 200, ticker="XYZ")
    assert "Item 1" in str(exc.value)
    assert "XYZ" in str(exc.value)


def test_item1_window_ignores_item_1a_1b_1c_and_10_through_16():
    """Item 1A, 1B, 1C and 10 to 16 all begin with the characters "Item 1"."""
    body = "wireless subscribers and spectrum holdings. " * 200
    text = (
        "PART IItem 1B.Unresolved Staff CommentsItem 1C.CybersecurityItem 10.Directors"
        "Item 15.ExhibitsPART IItem 1. Business" + body + "Item 1A. Risk Factors here."
    )
    window = item1_window(text)
    assert window.startswith("Item 1. Business")
    assert "Unresolved Staff Comments" not in window


# --------------------------------------------------------------------------- #
# Removing the filer's identity
# --------------------------------------------------------------------------- #


def test_deglue_splits_run_together_words_and_spares_sector_vocabulary():
    assert deglue("BusinessOverviewDatadog is") == "Business Overview Datadog is"
    assert deglue("Strengths5Datadog was") == "Strengths5 Datadog was"
    # A capital that starts no word is not a word boundary.
    for intact in ("SaaS", "IoT", "MongoDB", "API", "5G", "EBITDA"):
        assert deglue(f"our {intact} platform") == f"our {intact} platform"


def test_name_variants_drops_legal_suffixes_and_keeps_industry_words():
    assert name_variants("CrowdStrike Holdings, Inc.") == [
        "CrowdStrike Holdings Inc",
        "CrowdStrike",
    ]
    # "Communications" describes an industry. Removing it from Verizon's filing
    # and leaving it in T-Mobile's would invent a difference between them.
    variants = name_variants("VERIZON COMMUNICATIONS INC")
    assert "VERIZON" in variants
    assert "COMMUNICATIONS" not in variants


def test_scrub_removes_the_name_even_where_markup_stripping_glued_it(excerpts):
    """The occurrences that survive a naive strip are the glued ones.

    Datadog's Item 1 opens "Item 1. BusinessOverviewDatadog is the AI-powered
    observability platform". There is no word boundary in front of that name, so
    a boundary-anchored substitution leaves it exactly where it does the most
    damage, in the first sentence.
    """
    document = excerpts["DDOG"]["item1_excerpt"]
    assert "BusinessOverviewDatadog" in document

    cleaned = scrub(document, name="Datadog, Inc.", ticker="DDOG")
    assert "datadog" not in cleaned.lower()
    # The description itself is untouched.
    assert "observability" in cleaned.lower()


def test_scrub_matches_the_ticker_case_sensitively():
    """A three-letter ticker is very often an ordinary word."""
    text = "All of our revenue comes from ALL branded devices sold to all customers."
    cleaned = scrub(text, name=None, ticker="ALL")
    assert "all customers" in cleaned
    assert "ALL branded" not in cleaned


# --------------------------------------------------------------------------- #
# Building the corpus point in time
# --------------------------------------------------------------------------- #


def test_build_corpus_takes_the_latest_filing_on_or_before_the_date(excerpts):
    client = FakeClient(excerpts)
    corpus = build_corpus(["DDOG", "VZ"], client, as_of=date(2026, 3, 1))

    assert corpus.tickers == ["DDOG", "VZ"]
    assert corpus.as_of_by_ticker["DDOG"] == date(2026, 2, 18)
    assert corpus.source_accessions["DDOG"] == excerpts["DDOG"]["accession"]
    assert corpus.entity_names["VZ"] == "VERIZON COMMUNICATIONS INC"
    assert "observability" in corpus.document("DDOG").lower()


def test_build_corpus_honours_an_earlier_as_of(excerpts):
    """Zscaler filed on 2026-09-03. A corpus dated before that must not see it."""
    client = FakeClient(excerpts)
    corpus = build_corpus(["ZS"], client, as_of=date(2026, 3, 1))
    assert corpus.as_of_by_ticker["ZS"] == date(2025, 9, 3)


def test_build_corpus_reaches_back_past_the_filing_index_window():
    """A fold dated 2015 must find its own 10-K, not conclude there is none.

    The client hands back its newest filings and cuts the list at ``limit``;
    the ``as_of`` filter runs after that cut. Ask for eight and a 2015 fit date
    sees only 2019 through 2026, discards all of them as too late, and reports a
    filer that has filed every February since 2008 as having no 10-K on record.
    Nothing about that failure looks like a failure downstream: the company is
    simply missing from the corpus, and the backtest it was fitted for quietly
    covers fewer companies the further back it goes, which is the direction that
    flatters a result.
    """
    client = DeepFilingClient()
    corpus = build_corpus(["DEEP"], client, as_of=date(2015, 6, 30))

    assert corpus.tickers == ["DEEP"]
    assert corpus.as_of_by_ticker["DEEP"] == date(2015, 2, 20)
    assert min(client.limits_seen) >= 25


def test_build_corpus_says_the_window_ran_out_rather_than_claiming_no_filing():
    """Exhausting the window is not the same fact as a filer never filing.

    When every filing the client returned postdates the fit date, an older one
    may still exist just beyond the window. Reporting that as "no 10-K filed on
    or before D" asserts an absence that was never checked, so the note names
    the window instead and says which knob moves it.
    """
    client = DeepFilingClient()
    corpus = build_corpus(["DEEP"], client, as_of=date(2015, 6, 30), filing_limit=4)

    assert corpus.tickers == []
    note = " ".join(corpus.notes)
    assert "4 most recent" in note
    assert "filing_limit" in note
    assert "no 10-K filed on or before" not in note


def test_build_corpus_skips_a_filer_it_cannot_read_and_says_why(excerpts):
    client = FakeClient(excerpts)
    corpus = build_corpus(["DDOG", "NOSUCH"], client, as_of=date(2026, 3, 1))

    assert corpus.tickers == ["DDOG"]
    assert any("NOSUCH" in n for n in corpus.notes)
    # Absent, not present and empty: an empty document becomes a zero vector,
    # and a zero vector reads as "similar to nothing".
    assert len(corpus.documents) == 1


def test_build_corpus_prefers_the_section_loader_when_one_is_given(excerpts):
    client = FakeClient(excerpts)
    calls: list[str] = []

    def loader(ticker, _client):
        calls.append(ticker)
        return "a business description supplied by nlp.sections"

    corpus = build_corpus(["DDOG"], client, section_loader=loader, as_of=date(2026, 3, 1))
    assert calls == ["DDOG"]
    assert corpus.documents == ["a business description supplied by nlp.sections"]
    assert client.text_calls == []
    assert any("section loader" in n for n in corpus.notes)


def test_corpus_without_filing_dates_is_refused():
    with pytest.raises(ConfigError, match="knowable"):
        TextCorpus(
            tickers=["AAA"],
            documents=["some text"],
            as_of_by_ticker={},
            source_accessions={},
        )


def test_corpus_to_frame_carries_the_provenance(real_corpus):
    frame = real_corpus.to_frame()
    assert list(frame.index) == real_corpus.tickers
    assert frame.loc["MDB", "accession"] == real_corpus.source_accessions["MDB"]
    assert frame.loc["MDB", "characters"] > 1_000
    assert dict(real_corpus.rows())["Companies"] == 6


# --------------------------------------------------------------------------- #
# Point in time, which is the part that fails silently
# --------------------------------------------------------------------------- #


def test_fit_refuses_a_document_filed_after_the_fit_date(real_corpus, ml_assumptions):
    """Zscaler's 10-K landed 2026-09-03. A fit dated February cannot have read it."""
    ml_assumptions.as_of = "2026-02-20"
    with pytest.raises(ConfigError) as exc:
        fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)

    message = str(exc.value)
    assert "vocabulary leakage" in message
    assert "ZS filed 2026-09-03" in message
    assert "transform()" in message


def test_fit_accepts_the_corpus_once_the_date_moves_past_it(real_corpus, ml_assumptions):
    ml_assumptions.as_of = "2026-09-10"
    features = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    assert features.fitted_through == date(2026, 9, 10)
    assert set(features.tickers) == set(real_corpus.tickers)


def test_fit_through_overrides_the_valuation_date(real_corpus, ml_assumptions):
    ml_assumptions.as_of = "2026-09-10"
    with pytest.raises(ConfigError, match="vocabulary leakage"):
        fit_text_features(
            real_corpus, ml_assumptions, dim=32, cache=False, fit_through=date(2026, 3, 1)
        )


def test_fit_without_a_valuation_date_stamps_the_latest_filing(real_corpus, ml_assumptions):
    assert ml_assumptions.as_of is None
    features = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    assert features.fitted_through == date(2026, 9, 3)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_two_fits_with_the_same_seed_agree_exactly(real_corpus, ml_assumptions):
    """Exactly, not approximately. Truncated SVD is randomised and is seeded."""
    first = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    second = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)

    assert np.array_equal(first.matrix, second.matrix)
    assert first.explained_variance == second.explained_variance
    assert first.vocabulary_size == second.vocabulary_size


def test_a_different_seed_moves_the_fit_but_not_the_conclusion(real_corpus, ml_assumptions):
    """A rotation of the same subspace, so the cosines survive it."""
    ml_assumptions.ml.random_seed = 7
    seven = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    ml_assumptions.ml.random_seed = 20260911
    other = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)

    assert seven.similarity("VZ", "TMUS") == pytest.approx(
        other.similarity("VZ", "TMUS"), abs=1e-6
    )


# --------------------------------------------------------------------------- #
# Bigrams
# --------------------------------------------------------------------------- #


def test_bigrams_reach_the_vocabulary(real_corpus, ml_assumptions):
    """Unigrams scatter the sector signal that a bigram holds together."""
    features = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    vocabulary = set(features.vectorizer.get_feature_names_out())

    bigrams = {term for term in vocabulary if " " in term}
    assert len(bigrams) > 50
    for term in ("artificial intelligence", "cloud native", "fixed wireless"):
        assert term in bigrams
    # The unigrams that make them up are there too, and separately mean less.
    assert "wireless" in vocabulary


def test_pure_numbers_are_not_features_but_alphanumerics_are(real_corpus, ml_assumptions):
    features = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    vocabulary = set(features.vectorizer.get_feature_names_out())
    assert not any(term.isdigit() for term in vocabulary)
    assert "5g" in vocabulary


# --------------------------------------------------------------------------- #
# The name leak, measured rather than asserted
# --------------------------------------------------------------------------- #


def test_stripping_the_company_name_changes_the_neighbour_ranking(
    synthetic_corpus, ml_assumptions
):
    """One media filer names one carrier fourteen times. That is the whole leak.

    Left in, "zenith" occurs in exactly two documents out of six, survives
    min_df, carries a high idf because it is rare, and makes a streaming service
    the nearest business in the universe to a wireless carrier. Stripped from
    its owner's own document it occurs in one document only, min_df discards it,
    and the carrier's nearest neighbour goes back to being a carrier.

    Both fits are run rather than one, because a claim about preprocessing that
    is asserted instead of measured is not worth making.
    """
    leaky = fit_text_features(
        synthetic_corpus, ml_assumptions, dim=8, cache=False, strip_names=False
    )
    clean = fit_text_features(synthetic_corpus, ml_assumptions, dim=8, cache=False)

    assert nearest_neighbours(leaky, "ZEN", 1)[0][0] == "ORN"
    assert nearest_neighbours(clean, "ZEN", 1)[0][0] in {"PEAK", "ATL"}

    # And the mechanism is what was claimed, not a coincidence of the projection.
    assert "zenith" in set(leaky.vectorizer.get_feature_names_out())
    assert "zenith" not in set(clean.vectorizer.get_feature_names_out())

    # The leaky fit says so about itself.
    assert any("NAME LEFT IN" in note for note in leaky.notes)


def test_the_leak_moves_the_similarity_and_not_only_the_order(
    synthetic_corpus, ml_assumptions
):
    leaky = fit_text_features(
        synthetic_corpus, ml_assumptions, dim=8, cache=False, strip_names=False
    )
    clean = fit_text_features(synthetic_corpus, ml_assumptions, dim=8, cache=False)
    assert leaky.similarity("ZEN", "ORN") > clean.similarity("ZEN", "ORN") + 0.10


# --------------------------------------------------------------------------- #
# Does it work: sector structure on real filings
# --------------------------------------------------------------------------- #


def test_within_sector_cosine_beats_every_across_sector_cosine(real_corpus, ml_assumptions):
    """Four software companies and two carriers, from their own Item 1 text.

    The claim is the strict one. The weakest pair inside a sector is further
    apart than nothing; it is still closer than the closest pair spanning the
    two sectors.
    """
    features = fit_text_features(real_corpus, ml_assumptions, dim=64, cache=False)

    within = [features.similarity(a, b) for a, b in _pairs(SOFTWARE)]
    within += [features.similarity(a, b) for a, b in _pairs(CARRIERS)]
    across = [features.similarity(a, b) for a in SOFTWARE for b in CARRIERS]

    assert min(within) > max(across)
    # The two carriers describe nearly the same business.
    assert features.similarity("VZ", "TMUS") > 0.85
    # The two endpoint security companies are the closest software pair.
    assert features.similarity("CRWD", "ZS") == max(
        features.similarity(a, b) for a, b in _pairs(SOFTWARE)
    )


def test_the_raw_tfidf_baseline_separates_the_sectors_too(real_corpus, ml_assumptions):
    """The baseline is a real one. A learned model has to beat this, not a straw man."""
    features = fit_text_features(real_corpus, ml_assumptions, dim=64, cache=False)

    assert baseline_neighbours(features, "VZ", 1)[0][0] == "TMUS"
    assert baseline_neighbours(features, "CRWD", 1)[0][0] == "ZS"
    assert {t for t, _ in baseline_neighbours(features, "MDB", 3)} == {"DDOG", "ZS", "CRWD"}


def test_neighbours_are_ranked_descending_and_exclude_self(real_corpus, ml_assumptions):
    features = fit_text_features(real_corpus, ml_assumptions, dim=64, cache=False)
    for space in ("lsa", "tfidf"):
        got = nearest_neighbours(features, "DDOG", 5, space=space)
        assert "DDOG" not in [t for t, _ in got]
        scores = [s for _, s in got]
        assert scores == sorted(scores, reverse=True)
        assert len(got) == 5


def test_an_unknown_space_is_refused(real_corpus, ml_assumptions):
    features = fit_text_features(real_corpus, ml_assumptions, dim=64, cache=False)
    with pytest.raises(ConfigError, match="lsa"):
        nearest_neighbours(features, "DDOG", 3, space="word2vec")


def _pairs(tickers):
    return [
        (a, b) for i, a in enumerate(tickers) for b in tickers[i + 1 :]
    ]


# --------------------------------------------------------------------------- #
# Transform: companies the fit never saw
# --------------------------------------------------------------------------- #


def test_transform_places_a_held_out_company_with_its_sector(excerpts, ml_assumptions):
    """MongoDB is removed from the fit entirely, then placed by transform alone."""
    kept = [t for t in excerpts if t != "MDB"]
    corpus = TextCorpus(
        tickers=kept,
        documents=[excerpts[t]["item1_excerpt"] for t in kept],
        as_of_by_ticker={t: date.fromisoformat(excerpts[t]["filed"]) for t in kept},
        source_accessions={t: excerpts[t]["accession"] for t in kept},
        entity_names={t: excerpts[t]["entity_name"] for t in kept},
    )
    features = fit_text_features(corpus, ml_assumptions, dim=64, cache=False)

    vector = transform(
        features,
        [excerpts["MDB"]["item1_excerpt"]],
        names=[excerpts["MDB"]["entity_name"]],
        tickers=["MDB"],
    )
    assert vector.shape == (1, features.dim)
    assert np.linalg.norm(vector[0]) == pytest.approx(1.0)

    sims = dict(zip(features.tickers, features.matrix @ vector[0]))
    assert min(sims[t] for t in SOFTWARE if t != "MDB") > max(sims[t] for t in CARRIERS)


def test_transform_refuses_to_leave_the_name_in_when_the_fit_took_it_out(
    real_corpus, ml_assumptions
):
    features = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    with pytest.raises(ConfigError) as exc:
        transform(features, ["some business description"])
    assert "names=" in str(exc.value)

    # The explicit opt out is accepted, because a caller may have text with no
    # identity in it to remove.
    anonymous = transform(features, ["wireless subscribers and spectrum"], names=[None])
    assert anonymous.shape == (1, features.dim)


def test_transform_rejects_ragged_inputs(real_corpus, ml_assumptions):
    features = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    with pytest.raises(ConfigError, match="parallel"):
        transform(features, ["a", "b"], names=["Only One, Inc."])


def test_a_document_with_no_shared_vocabulary_raises_rather_than_scoring_zero(
    real_corpus, ml_assumptions
):
    """A zero vector is an absence of measurement, not a similarity of zero."""
    features = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    with pytest.raises(MissingDataError) as exc:
        transform(features, ["zzzqqq xxxyyy wwwvvv"], names=[None], tickers=["NEW"])
    assert "NEW" in str(exc.value)
    assert "absence of measurement" in str(exc.value)


# --------------------------------------------------------------------------- #
# Reporting the fit honestly
# --------------------------------------------------------------------------- #


def test_explained_variance_is_reported_and_bounded(real_corpus, ml_assumptions):
    features = fit_text_features(real_corpus, ml_assumptions, dim=64, cache=False)
    assert 0.0 < features.explained_variance <= 1.0
    assert dict(features.rows())["Variance retained"] == features.explained_variance


def test_a_small_corpus_clamps_the_dimension_and_says_so(real_corpus, ml_assumptions):
    """Six documents span at most six directions, whatever was asked for."""
    features = fit_text_features(real_corpus, ml_assumptions, dim=256, cache=False)
    assert features.dim == 5
    assert features.matrix.shape == (6, 5)
    assert any("requested 256 dimensions" in note for note in features.notes)


def test_rows_and_frames_are_renderable(real_corpus, ml_assumptions):
    features = fit_text_features(real_corpus, ml_assumptions, dim=64, cache=False)
    rows = dict(features.rows())
    assert rows["Companies"] == 6
    assert rows["Own name stripped"] is True

    frame = features.to_frame()
    assert frame.shape == (6, features.dim)
    assert list(frame.index) == features.tickers

    block = features.similarity_frame()
    assert block.loc["VZ", "VZ"] == pytest.approx(1.0)
    assert block.loc["VZ", "TMUS"] == pytest.approx(block.loc["TMUS", "VZ"])


def test_vectors_are_unit_length_so_cosine_is_a_dot_product(real_corpus, ml_assumptions):
    features = fit_text_features(real_corpus, ml_assumptions, dim=64, cache=False)
    norms = np.linalg.norm(features.matrix, axis=1)
    assert np.allclose(norms, 1.0)
    assert features.similarity("DDOG", "DDOG") == pytest.approx(1.0)
    assert features.vector("DDOG") @ features.vector("MDB") == pytest.approx(
        features.similarity("DDOG", "MDB")
    )


def test_an_unfitted_company_is_named_in_the_error(real_corpus, ml_assumptions):
    features = fit_text_features(real_corpus, ml_assumptions, dim=32, cache=False)
    with pytest.raises(MissingDataError) as exc:
        features.vector("NFLX")
    assert "NFLX" in str(exc.value)
    assert "transform()" in str(exc.value)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_an_empty_corpus_is_refused(ml_assumptions):
    empty = TextCorpus(
        tickers=[], documents=[], as_of_by_ticker={}, source_accessions={},
        notes=["AAA: skipped, no 10-K on file"],
    )
    with pytest.raises(ConfigError, match="empty corpus"):
        fit_text_features(empty, ml_assumptions, cache=False)


def test_min_df_of_one_is_refused(real_corpus, ml_assumptions):
    with pytest.raises(ConfigError, match="single document"):
        fit_text_features(real_corpus, ml_assumptions, dim=32, min_df=1, cache=False)


def test_an_impossible_max_df_is_refused(real_corpus, ml_assumptions):
    with pytest.raises(ConfigError, match="max_df"):
        fit_text_features(real_corpus, ml_assumptions, dim=32, max_df=1.5, cache=False)


def test_a_corpus_too_small_to_project_is_refused(ml_assumptions):
    tiny = TextCorpus(
        tickers=["AAA", "BBB"],
        documents=[
            "wireless subscribers spectrum towers",
            "wireless subscribers spectrum fiber",
        ],
        as_of_by_ticker={"AAA": date(2026, 1, 1), "BBB": date(2026, 1, 1)},
        source_accessions={},
    )
    with pytest.raises(ConfigError, match="nothing to project onto"):
        fit_text_features(tiny, ml_assumptions, dim=8, cache=False)


def test_a_vocabulary_that_prunes_to_nothing_is_refused(ml_assumptions):
    corpus = TextCorpus(
        tickers=["AAA", "BBB", "CCC"],
        documents=["alpha alpha", "beta beta", "gamma gamma"],
        as_of_by_ticker={t: date(2026, 1, 1) for t in ("AAA", "BBB", "CCC")},
        source_accessions={},
    )
    with pytest.raises(ConfigError, match="no vocabulary survived"):
        fit_text_features(corpus, ml_assumptions, dim=2, cache=False)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_the_cache_returns_the_identical_fit(real_corpus, ml_assumptions):
    first = fit_text_features(real_corpus, ml_assumptions, dim=32)
    cached = list(Path(ml_assumptions.ml.cache_dir).glob("text_features_*.joblib"))
    assert len(cached) == 1

    second = fit_text_features(real_corpus, ml_assumptions, dim=32)
    assert np.array_equal(first.matrix, second.matrix)
    assert second.fitted_through == first.fitted_through
    assert second.vocabulary_size == first.vocabulary_size


def test_the_cache_key_follows_the_documents_not_only_the_tickers(
    real_corpus, ml_assumptions
):
    fit_text_features(real_corpus, ml_assumptions, dim=32)

    # Same tickers, same dates, different text: a section loader instead of the
    # fallback would do exactly this.
    shortened = TextCorpus(
        tickers=real_corpus.tickers,
        documents=[d[:4000] for d in real_corpus.documents],
        as_of_by_ticker=real_corpus.as_of_by_ticker,
        source_accessions=real_corpus.source_accessions,
        entity_names=real_corpus.entity_names,
    )
    fit_text_features(shortened, ml_assumptions, dim=32)

    cached = list(Path(ml_assumptions.ml.cache_dir).glob("text_features_*.joblib"))
    assert len(cached) == 2


def test_the_cache_key_follows_the_seed(real_corpus, ml_assumptions):
    fit_text_features(real_corpus, ml_assumptions, dim=32)
    ml_assumptions.ml.random_seed = 99
    fit_text_features(real_corpus, ml_assumptions, dim=32)
    cached = list(Path(ml_assumptions.ml.cache_dir).glob("text_features_*.joblib"))
    assert len(cached) == 2


def test_a_cache_that_cannot_be_written_is_noted_not_raised(
    real_corpus, assumptions, tmp_path
):
    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not a directory")
    assumptions.ml.cache_dir = str(blocked / "ml")

    features = fit_text_features(real_corpus, assumptions, dim=32)
    assert features.matrix.shape == (6, 5)
    assert any("not cached" in note for note in features.notes)
