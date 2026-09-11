"""Assumption schema.

Every judgment that moves a valuation lives here and is loaded from YAML. Nothing
in the numeric modules carries a hardcoded rate, growth path, margin or multiple.
Defaults exist so a run is possible without a full file, but each default is
documented in ``docs/methodology.md`` and echoed in CLI output, because an
undocumented default is indistinguishable from a hardcoded number.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import ConfigError


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketAssumptions(_Base):
    equity_risk_premium: float = Field(
        0.050,
        description=(
            "Forward-looking ERP over the 10Y Treasury. Practitioners commonly use "
            "4.5-6.0%; Damodaran's implied ERP and the Duff & Phelps / Kroll "
            "recommended rate both sit in that band. Pure assumption, not observable."
        ),
    )
    risk_free_rate: float | None = Field(
        None,
        description=(
            "Override the 10Y constant-maturity Treasury yield. Left null the engine "
            "pulls the latest published value from the US Treasury daily curve."
        ),
    )
    market_index: str = Field(
        "SPY",
        description="Market proxy for the beta regression. SPY is the S&P 500 tracker.",
    )
    beta_lookback_years: float = 2.0
    beta_frequency: Literal["weekly", "monthly"] = "weekly"
    beta_adjustment: Literal["raw", "blume"] = Field(
        "raw",
        description=(
            "'blume' applies 0.67*raw + 0.33*1.0, the Bloomberg adjusted-beta "
            "convention that shrinks toward the market. 'raw' is the OLS estimate."
        ),
    )
    peer_beta_method: Literal["median", "vasicek"] = Field(
        "median",
        description=(
            "How peer asset betas are pooled. 'median' is the sell-side default and "
            "treats a beta measured on an R-squared of 0.05 exactly like one measured "
            "on 0.40. 'vasicek' weights each peer by the precision of its own "
            "regression, shrinking noisy estimates toward the cross-sectional mean by "
            "an amount the data decides rather than by a fixed rule. Blume shrinks "
            "every beta by the same 33% no matter how well measured it was."
        ),
    )
    size_premium: float = Field(
        0.0,
        description="Additive CAPM adjustment for small-cap illiquidity. Default off.",
    )


class DilutionAssumptions(_Base):
    """How the share count in equity value is built."""

    method: Literal["waso", "treasury_stock"] = Field(
        "waso",
        description=(
            "'waso' uses trailing-twelve-month weighted-average diluted shares, which "
            "is an average over a past window and understates the count for a company "
            "issuing steadily. 'treasury_stock' counts basic shares outstanding plus "
            "the net new shares in-the-money awards would create at today's price, "
            "which is the point-in-time figure a live model uses."
        ),
    )
    include_rsus: bool = Field(
        True,
        description=(
            "Unvested RSUs have no strike, so they contribute their full count with no "
            "buyback offset. Excluding them understates dilution at any software "
            "company that has moved from options to units."
        ),
    )
    assumed_forfeiture_rate: float = Field(
        0.0,
        ge=0.0,
        le=0.5,
        description=(
            "Share of unvested awards assumed never to vest. Left at zero: the "
            "treasury stock method as written in ASC 260 does not haircut for "
            "forfeitures, and guessing one is a thumb on the scale."
        ),
    )


class NOLAssumptions(_Base):
    """Net operating loss carryforwards in the projection."""

    track: bool = Field(
        False,
        description=(
            "Off by default so the base case stays simple. On, a projected loss "
            "creates a carryforward that shelters later taxable income, which is "
            "worth real money to a company with a decade of accumulated losses."
        ),
    )
    opening_balance: float | None = Field(
        None,
        description=(
            "Federal NOL carryforward at the valuation date, USD millions. Null reads "
            "it from the filings where the filer tags it; the tax footnote carries the "
            "number when the tag is absent."
        ),
    )
    annual_limitation_pct: float = Field(
        0.80,
        description=(
            "Share of taxable income a carryforward may shelter in one year. Post-2017 "
            "federal losses carry forward indefinitely but offset only 80% of taxable "
            "income, so a profitable year still pays some cash tax."
        ),
    )


class TerminalAssumptions(_Base):
    """How the terminal value is constructed."""

    method: Literal["gordon", "value_driver"] = Field(
        "gordon",
        description=(
            "'gordon' capitalises the final projected cash flow and leaves the implied "
            "return on capital to be checked afterwards. 'value_driver' inverts that: "
            "you state the terminal ROIC you believe and the reinvestment needed to "
            "fund growth is derived from it, so the terminal value cannot embed a "
            "return nobody signed up for. Both are always reported; this picks the "
            "headline."
        ),
    )
    terminal_roic: float | None = Field(
        None,
        description=(
            "Return on incremental invested capital in perpetuity, used by the "
            "value-driver formula. Null falls back to the WACC, the competitive "
            "equilibrium assumption that growth is worth exactly nothing."
        ),
    )


class SimulationAssumptions(_Base):
    """Monte Carlo over the assumptions that actually move the answer."""

    enabled: bool = False
    draws: int = Field(10_000, ge=100, le=1_000_000)
    seed: int = Field(
        7,
        description="Fixed so a distribution is reproducible; this is a model, not a lottery.",
    )
    revenue_growth_sd: float = Field(
        0.04, description="Standard deviation on the year-one growth rate, absolute."
    )
    terminal_margin_sd: float = Field(
        0.04, description="Standard deviation on the terminal EBIT margin, absolute."
    )
    wacc_sd: float = Field(0.010, description="Standard deviation on the discount rate.")
    terminal_growth_sd: float = Field(0.005)


class APVAssumptions(_Base):
    """Adjusted present value, valued as unlevered firm plus financing side effects."""

    enabled: bool = Field(
        False,
        description=(
            "On, the engine values the business unlevered and adds the present value of "
            "the interest tax shield separately, then reconciles to the WACC answer. "
            "The two agree only when leverage is constant; where they diverge, the "
            "divergence is the point."
        ),
    )
    shield_discount_rate: Literal["cost_of_debt", "unlevered_cost_of_equity"] = Field(
        "cost_of_debt",
        description=(
            "What the tax shield is worth turns on how safe it is. Discounting at the "
            "cost of debt (Modigliani-Miller) assumes a fixed debt schedule; "
            "discounting at the unlevered cost of equity (Miles-Ezzell, Harris-Pringle) "
            "assumes debt rebalances with firm value, which is what a constant-WACC "
            "model already implies."
        ),
    )


class RegressionCompsAssumptions(_Base):
    """Warranted multiples from fundamentals rather than from a percentile band."""

    enabled: bool = Field(
        False,
        description=(
            "Fits EV/Revenue across the peer set against growth and margin, then reads "
            "off the multiple the target's own fundamentals warrant. A percentile band "
            "says what the neighbours trade at; this says what the company should trade "
            "at given what it is. Needs a real sample to mean anything."
        ),
    )
    min_observations: int = Field(
        8,
        description=(
            "Below this the fit is memorising the peer set. The engine reports the "
            "regression as unavailable rather than fitting two parameters to six points."
        ),
    )
    drivers: list[str] = Field(
        default_factory=lambda: ["revenue_growth", "ebitda_margin"],
        description="Explanatory variables, as PeerMetrics attribute names.",
    )


class PurchaseAccountingAssumptions(_Base):
    """The opening balance sheet a real merger model builds."""

    enabled: bool = Field(
        False,
        description=(
            "On, the merger module builds goodwill, writes up identifiable intangibles, "
            "books the deferred tax liability on the step-up, haircuts acquired deferred "
            "revenue and runs the deal forward with debt amortisation. Off, it stays the "
            "year-one screening tool it is documented as."
        ),
    )
    intangible_pct_of_excess: float = Field(
        0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Share of purchase price above book equity allocated to identifiable "
            "intangibles rather than goodwill. Software deals commonly land 30-50% here; "
            "the rest is goodwill, which is not amortised."
        ),
    )
    intangible_life_years: int = Field(7, ge=1, le=40)
    deferred_revenue_haircut: float = Field(
        0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Write-down of acquired deferred revenue to fair value, which is the cost of "
            "delivering the service plus a margin, not the amount billed. The haircut "
            "permanently destroys revenue the target would have recognised, which is why "
            "SaaS acquisitions look worse in year one than the run rate suggests."
        ),
    )
    projection_years: int = Field(3, ge=1, le=10)
    debt_repayment_pct_of_fcf: float = Field(
        0.50,
        ge=0.0,
        le=1.0,
        description="Share of free cash flow swept to repay acquisition debt each year.",
    )


class TaxAssumptions(_Base):
    marginal_tax_rate: float = Field(
        0.24,
        description=(
            "21% US federal plus a blended state burden. Used for NOPAT, the debt "
            "tax shield and synergy taxation. Deliberately not the effective rate: "
            "the GAAP effective rate of a loss-making or NOL-rich software company "
            "is a reporting artefact, not the rate an incremental dollar bears."
        ),
    )
    use_effective_rate: bool = Field(
        False,
        description="Use the filed effective rate instead. Off by default; see methodology.",
    )


class LeaseAssumptions(_Base):
    capitalize_operating_leases: bool = Field(
        False,
        description=(
            "False (default): operating lease liabilities are excluded from debt in "
            "the headline EV, which is the consistent pairing with a US-GAAP EBITDA "
            "that is already net of operating lease cost. True: liabilities go into "
            "debt, and every earnings metric is then taken pre-rent (EBITDAR). The "
            "engine will not let you mix the two."
        ),
    )
    include_finance_leases_in_debt: bool = Field(
        True,
        description=(
            "Finance lease liabilities are debt under both conventions: their "
            "interest and amortisation are already outside EBITDA."
        ),
    )


class ConvertibleAssumptions(_Base):
    treatment: Literal["if_converted", "debt", "auto"] = Field(
        "auto",
        description=(
            "How convertible notes enter the bridge. 'debt' adds principal to debt "
            "and assumes conversion shares are NOT in diluted WASO. 'if_converted' "
            "excludes them from debt because the shares already sit in diluted WASO "
            "(mandatory under ASU 2020-06 whenever dilutive). 'auto' uses the "
            "conversion price when supplied, otherwise treats them as debt and warns."
        ),
    )
    conversion_price: float | None = Field(
        None, description="Conversion price per share, from the notes footnote."
    )


class DCFAssumptions(_Base):
    projection_years: int = Field(5, ge=1, le=15)
    revenue_growth_start: float = Field(0.20, description="Year 1 revenue growth.")
    revenue_growth_terminal: float = Field(
        0.05, description="Year N growth; the path fades linearly from start to this."
    )
    ebit_margin_start: float | None = Field(
        None,
        description="Year 1 EBIT margin. Null anchors on the TTM realised margin.",
    )
    ebit_margin_terminal: float = Field(
        0.20, description="Year N EBIT margin; path interpolates linearly."
    )
    capex_pct_revenue: float = 0.030
    da_pct_revenue: float = 0.025
    nwc_pct_revenue: float = Field(
        -0.05,
        description=(
            "Non-cash working capital as a share of revenue. Negative for most "
            "subscription software: deferred revenue is billed ahead of delivery, so "
            "growth releases cash rather than consuming it."
        ),
    )
    sbc_treatment: Literal["expense", "addback"] = Field(
        "expense",
        description=(
            "'expense' (default) leaves SBC inside EBIT, so it is charged against "
            "free cash flow as the real compensation cost it is. 'addback' restores "
            "it as a non-cash charge, which overstates FCF unless you separately "
            "model the dilution it creates. See methodology for both camps."
        ),
    )
    mid_year_convention: bool = Field(
        True,
        description="Discount explicit-period flows at t-0.5. See methodology for the terminal-value interaction.",
    )
    terminal_growth: float = Field(0.025, description="Gordon g; must be below WACC.")
    exit_multiple: float | None = Field(
        None, description="EV/EBITDA exit multiple. Null uses the peer median."
    )
    wacc_override: float | None = None
    sensitivity_wacc_step: float = 0.015
    sensitivity_growth_step: float = 0.010

    sbc_dilution: bool = Field(
        True,
        description=(
            "Only bites under sbc_treatment: addback. Grows the share count each year "
            "by the stock compensation issued divided by the share price, which is the "
            "other half of that treatment: adding the cash back while holding the "
            "denominator flat counts the benefit of paying in equity and ignores its "
            "cost. It does NOT reconcile the two camps. Modelling the dilution closes "
            "only about an eighth of the gap, because the addback lands mostly in the "
            "terminal value, which capitalises it in perpetuity, while a five year "
            "projection issues shares against just five years of it. See "
            "docs/methodology.md; the number is reported in the DCF checks."
        ),
    )
    nol: NOLAssumptions = Field(default_factory=NOLAssumptions)
    terminal: TerminalAssumptions = Field(default_factory=TerminalAssumptions)

    @model_validator(mode="after")
    def _check(self) -> "DCFAssumptions":
        if self.terminal_growth >= 0.06:
            raise ConfigError(
                f"terminal_growth of {self.terminal_growth:.2%} exceeds any plausible "
                "long-run nominal growth of the economy. No firm outgrows its own "
                "economy forever."
            )
        return self


class CostOfDebtAssumptions(_Base):
    method: Literal["filings", "synthetic", "override"] = Field(
        "synthetic",
        description=(
            "'filings' divides interest expense by period-end total debt, which "
            "is a backward-looking book yield, not a market one; for a 0%-coupon "
            "convertible issuer it returns a near-zero cost of debt that is plainly "
            "wrong. 'synthetic' assigns a rating from interest coverage and adds the "
            "corresponding spread to the risk-free rate. 'override' takes the value below."
        ),
    )
    override_rate: float | None = None


class CompsAssumptions(_Base):
    peers: list[str] = Field(default_factory=list)
    ev_ebitda_nm_threshold: float = Field(
        100.0,
        description=(
            "Above this, EV/EBITDA and EV/EBIT are flagged NM rather than printed. A "
            "900x multiple on a two-percent GAAP EBITDA margin is arithmetic, not "
            "valuation: it measures how near the denominator sits to zero."
        ),
    )
    pe_nm_threshold: float = Field(
        75.0,
        description=(
            "Above this, P/E is flagged NM. Set lower than the enterprise cut-off "
            "because net income sits below interest, tax and every non-operating "
            "item, so it reaches zero sooner and the ratio destabilises sooner."
        ),
    )
    apply_percentiles: tuple[float, float] = (0.25, 0.75)
    regression: RegressionCompsAssumptions = Field(
        default_factory=RegressionCompsAssumptions
    )


class SynergyAssumptions(_Base):
    pretax_cost_synergies: float = 0.0
    pretax_revenue_synergies: float = 0.0
    phase_in_year_one: float = Field(
        1.0, ge=0.0, le=1.0, description="Share of run-rate synergies realised in year 1."
    )


class MergerAssumptions(_Base):
    offer_price_per_share: float | None = None
    offer_premium: float | None = Field(
        None, description="Premium to the unaffected price; alternative to a price."
    )
    pct_cash: float = Field(0.50, ge=0.0, le=1.0)
    cost_of_new_debt: float = 0.055
    foregone_cash_yield: float = Field(
        0.040, description="Pre-tax yield surrendered on balance-sheet cash spent."
    )
    balance_sheet_cash_used: float = Field(
        0.0, description="USD millions of existing cash funding the cash portion."
    )
    deal_fees: float = Field(0.0, description="Advisory and other one-time fees, USD mm.")
    financing_fees: float = 0.0
    financing_fee_amort_years: int = 7
    include_intangible_amortization: bool = Field(
        False,
        description=(
            "Off by default: purchase-accounting step-up amortisation is a non-cash "
            "artefact of the deal and the street quotes cash EPS. Turn on to see it."
        ),
    )
    intangible_step_up: float = 0.0
    intangible_life_years: int = 10
    synergies: SynergyAssumptions = Field(default_factory=SynergyAssumptions)
    purchase_accounting: PurchaseAccountingAssumptions = Field(
        default_factory=PurchaseAccountingAssumptions
    )


# --------------------------------------------------------------------------- #
# TMT sector layer and the models over it
# --------------------------------------------------------------------------- #


class TMTAssumptions(_Base):
    """Sector conventions. TMT is three businesses, not one."""

    sub_vertical: str | None = Field(
        None,
        description=(
            "Override the classified sub-vertical: infrastructure_software, "
            "application_software, internet, semiconductors, hardware, payments, "
            "media_entertainment, telecom, towers_fiber, gaming. Null classifies "
            "from the filer's SIC code and business description."
        ),
    )
    kpi_extraction: bool = Field(
        True,
        description=(
            "Read the operating metrics the sector is priced on out of the filing "
            "text and custom tags: ARR, net revenue retention, remaining performance "
            "obligation, billings, subscribers, ARPU, churn, content spend. None of "
            "these is a standard us-gaap concept, which is why a generic engine "
            "cannot see them."
        ),
    )
    segments: bool = Field(
        True,
        description=(
            "Pull segment and geography detail from dimensioned XBRL. A media "
            "conglomerate valued on its consolidated margin is being valued as an "
            "average of businesses that trade at different multiples."
        ),
    )
    sotp: bool = Field(
        False,
        description=(
            "Value each reported segment on its own peer multiple and sum. Off by "
            "default: it is the right answer only where segments are genuinely "
            "separable and separately disclosed."
        ),
    )


class PeerModelAssumptions(_Base):
    """Learned comparable company selection."""

    enabled: bool = Field(
        False,
        description=(
            "Rank a candidate universe by similarity to the target and propose a "
            "comp set, instead of taking the hand-written list. The engine's own "
            "limitations section has always said peer sets are hand-picked and "
            "comparability is the user's judgment; this is the attempt to measure it."
        ),
    )
    n_peers: int = Field(8, ge=3, le=30)
    text_weight: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Blend between business-description similarity and financial-profile "
            "similarity. Text alone groups companies that describe themselves alike; "
            "financials alone group companies that happen to be the same size. "
            "Neither is a comp set on its own."
        ),
    )
    min_market_cap: float = Field(
        1_000.0, description="Smallest candidate by market capitalisation, USD millions."
    )
    max_size_ratio: float = Field(
        10.0,
        description=(
            "Largest tolerated ratio between candidate and target market "
            "capitalisation, either way round. A banker does not put a 2bn company "
            "in a 200bn company's comp set however similar the prose."
        ),
    )
    require_same_sub_vertical: bool = Field(
        False,
        description=(
            "NOT IMPLEMENTED, and it refuses rather than being ignored. The "
            "encoder gates its candidate list on min_market_cap and "
            "max_size_ratio and on nothing else; there is no sub-vertical gate "
            "behind this name. Left as the only field in this file that read as "
            "a setting and reached no code, it would be a written claim the "
            "engine does not keep, which is the failure this package is built "
            "around avoiding."
        ),
    )

    @field_validator("require_same_sub_vertical")
    @classmethod
    def _refuse_an_unimplemented_gate(cls, value: bool) -> bool:
        if value:
            raise ConfigError(
                "ml.peers.require_same_sub_vertical is not implemented. The "
                "encoder applies min_market_cap and max_size_ratio to its "
                "candidate list and no sub-vertical gate, so setting this true "
                "would change nothing and the ranking would not be what the "
                "assumptions file says it is. Restrict the ranking by hand from "
                "the Company column of `techval peers TICKER`, or set "
                "tmt.sub_vertical to correct a misclassified target."
            )
        return value


class ForecastAssumptions(_Base):
    """Statistical revenue and margin forecasts, held to a baseline."""

    enabled: bool = False
    horizon_years: int = Field(3, ge=1, le=5)
    model: Literal["ridge", "gradient_boosting", "linear"] = "ridge"
    min_train_observations: int = Field(
        60,
        description=(
            "Below this the fit is memorising. The engine reports the forecast as "
            "unavailable rather than producing one nobody should act on."
        ),
    )


class MnaAssumptions(_Base):
    """Which companies get acquired, and on what terms precedent suggests."""

    propensity_enabled: bool = False
    precedents_enabled: bool = Field(
        False,
        description=(
            "Build a precedent transaction set from merger filings: multiples paid, "
            "premium to unaffected, consideration mix. Precedents are what anchor an "
            "M&A conversation, and they are not trading comps: a control premium and "
            "expected synergies are inside every one of them."
        ),
    )
    lookback_years: int = Field(10, ge=1, le=25)
    min_deal_size: float = Field(
        250.0, description="Smallest precedent by equity purchase price, USD millions."
    )


class WarrantedAssumptions(_Base):
    """A fitted warranted multiple, and the gap to where the company trades."""

    enabled: bool = False
    target: Literal["ev_revenue", "ev_gross_profit", "ev_ebitda"] = "ev_revenue"
    demean_by_date: bool = Field(
        True,
        description=(
            "Remove each date's cross-sectional mean before fitting. Multiples "
            "re-rate market wide: software traded at 15x revenue in 2021 and 5x in "
            "2023 on the same fundamentals, because the discount rate moved, not "
            "the companies. A model fitted across both without this learns to read "
            "the rate cycle off company characteristics and calls it a valuation."
        ),
    )
    min_train_observations: int = Field(
        150,
        description="Below this the residual is noise and the model reports nothing.",
    )


class SignalAssumptions(_Base):
    """Whether a score predicts anything, tested against forward returns."""

    enabled: bool = False
    horizon_months: int = Field(12, ge=1, le=36)
    buckets: int = Field(5, ge=2, le=10)
    min_names_per_date: int = Field(
        20,
        description=(
            "A rank correlation across eight companies is not a result. Dates with "
            "fewer names are dropped from the information coefficient series."
        ),
    )
    overlapping_windows: bool = Field(
        True,
        description=(
            "Twelve month returns sampled quarterly share eleven months of their "
            "path, so the coefficients are autocorrelated and a naive t-statistic "
            "on them is roughly double what it should be. The harness computes "
            "the Newey-West standard error unconditionally and prints it beside "
            "the naive one with the inflation ratio between them, so this is a "
            "statement of fact about the sampling scheme rather than a switch. "
            "It refuses if set false, because the correction cannot be turned "
            "off and a reader who asked for it off should be told so rather than "
            "left to believe a naive t-statistic is what came back."
        ),
    )

    @field_validator("overlapping_windows")
    @classmethod
    def _refuse_to_pretend_the_windows_do_not_overlap(cls, value: bool) -> bool:
        if not value:
            raise ConfigError(
                "ml.signals.overlapping_windows cannot be set false. The windows "
                "overlap whatever this file says: twelve month returns sampled "
                "quarterly share eleven months of their path, and the harness "
                "always reports the Newey-West standard error, the naive one and "
                "the ratio between them. Setting this false would quietly change "
                "nothing. If the intent is non-overlapping windows, sample the "
                "scores annually instead, which changes the data rather than the "
                "label on it."
            )
        return value


class MLAssumptions(_Base):
    """Shared settings for everything fitted rather than assumed."""

    random_seed: int = Field(
        7, description="Fixed so a fitted model is reproducible. Never left to chance."
    )
    cache_dir: str | None = Field(
        None,
        description="Where fitted models and text features are cached. Null uses ~/.techval/ml.",
    )
    walk_forward_folds: int = Field(
        5,
        ge=2,
        le=20,
        description=(
            "Out-of-sample folds, split by DATE rather than at random. A random "
            "split lets a model train on 2026 and test on 2024, which is the "
            "cleanest way to manufacture a result that cannot be repeated."
        ),
    )
    peers: PeerModelAssumptions = Field(default_factory=PeerModelAssumptions)
    forecast: ForecastAssumptions = Field(default_factory=ForecastAssumptions)
    mna: MnaAssumptions = Field(default_factory=MnaAssumptions)
    warranted: WarrantedAssumptions = Field(default_factory=WarrantedAssumptions)
    signals: SignalAssumptions = Field(default_factory=SignalAssumptions)


class Assumptions(_Base):
    ticker: str | None = None
    market: MarketAssumptions = Field(default_factory=MarketAssumptions)
    tax: TaxAssumptions = Field(default_factory=TaxAssumptions)
    leases: LeaseAssumptions = Field(default_factory=LeaseAssumptions)
    convertibles: ConvertibleAssumptions = Field(default_factory=ConvertibleAssumptions)
    cost_of_debt: CostOfDebtAssumptions = Field(default_factory=CostOfDebtAssumptions)
    dcf: DCFAssumptions = Field(default_factory=DCFAssumptions)
    comps: CompsAssumptions = Field(default_factory=CompsAssumptions)
    merger: MergerAssumptions = Field(default_factory=MergerAssumptions)
    dilution: DilutionAssumptions = Field(default_factory=DilutionAssumptions)
    tmt: TMTAssumptions = Field(default_factory=TMTAssumptions)
    ml: MLAssumptions = Field(default_factory=MLAssumptions)
    simulation: SimulationAssumptions = Field(default_factory=SimulationAssumptions)
    apv: APVAssumptions = Field(default_factory=APVAssumptions)

    as_of: str | None = Field(
        None,
        description=(
            "Value the company as it was knowable on this date (YYYY-MM-DD). Every "
            "fact filed after it is discarded and prices stop there, so a historical "
            "run uses only information that existed at the time. This is what makes a "
            "backtest a backtest rather than an exercise in hindsight."
        ),
    )

    price_source: Literal["nasdaq", "stooq", "csv"] = "nasdaq"
    price_csv_dir: str | None = None

    @field_validator("as_of", mode="before")
    @classmethod
    def _accept_an_unquoted_date(cls, value):
        """``as_of: 2026-09-11`` is a date to YAML and a string to everything else.

        YAML 1.1 resolves an unquoted ISO date to a native date object, so the
        obviously correct line above arrived here as ``datetime.date`` and was
        rejected for not being a string, on a line a reader would stare at for a
        while before suspecting the quotes. The rest of the engine wants the ISO
        text, because that is what it passes to the EDGAR client and prints in
        the point-in-time banner, so the date is normalised to its own ISO form
        rather than the field being widened to hold either.

        ``isoformat`` rather than ``str`` so a ``datetime`` written with a time
        on it loses the time instead of carrying it into a filename.
        """
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return value

    @classmethod
    def load(cls, path: str | Path | None) -> "Assumptions":
        """Read an assumptions file, or the defaults when none is named.

        A rejected field is reported as a ``ConfigError`` rather than allowed out
        as a ``ValidationError``. Every command in this package catches
        ``TechvalError`` and prints the reason; a pydantic exception is caught by
        none of them, so a mistyped key in a YAML file produced a forty line
        traceback through the pydantic internals with the useful sentence at the
        bottom. The engine's own rule is that a refusal carries a reason, and a
        traceback is not a refusal.
        """
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"assumptions file not found: {p}")
        raw = yaml.safe_load(p.read_text()) or {}
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(
                ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
                for error in exc.errors()
            )
            raise ConfigError(f"{p} was not accepted. {problems}") from exc
