"""Assumption schema.

Every judgment that moves a valuation lives here and is loaded from YAML. Nothing
in the numeric modules carries a hardcoded rate, growth path, margin or multiple.
Defaults exist so a run is possible without a full file, but each default is
documented in ``docs/methodology.md`` and echoed in CLI output, because an
undocumented default is indistinguishable from a hardcoded number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    size_premium: float = Field(
        0.0,
        description="Additive CAPM adjustment for small-cap illiquidity. Default off.",
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
            "'filings' divides interest expense by average total debt, which is a "
            "backward-looking book yield, not a market one; for a 0%-coupon "
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
    price_source: Literal["nasdaq", "stooq", "csv"] = "nasdaq"
    price_csv_dir: str | None = None

    @classmethod
    def load(cls, path: str | Path | None) -> "Assumptions":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"assumptions file not found: {p}")
        raw = yaml.safe_load(p.read_text()) or {}
        return cls.model_validate(raw)
