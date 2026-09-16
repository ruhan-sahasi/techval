"""Valuation engine: Datadog through the analyst exhibits, on the committed fixtures.

Not a model section. There is no baseline to beat here, so the section carries
no headline; it shows the standard exhibits of a valuation, each one computed by
the engine's own entry points from the fixtures under ``tests/fixtures``:

    cost_of_capital   the discount rate as a tile row, with the date first
    football          every method as a range against the price
    bridge            equity value to enterprise value
    sensitivity       the Gordon DCF across WACC and terminal growth
    montecarlo        correlated against independent draws of the four drivers
    apv               APV under each shield rate against the WACC DCF

**What the run is pinned to.** The target is Datadog and the run uses the test
suite's own construction (``tests/conftest.py``): the committed companyfacts
payloads, the committed closes, and the suite's peer set and risk-free rate
where the assumptions leave them unset. Offline there is no Treasury quote to
read, and ``MarketData.risk_free_rate`` would reach for the network, so a null
rate is filled with ``SUITE_RISK_FREE_RATE`` and the page says so on the tile.
Both constants are locked to the conftest fixture by a test, so they cannot
drift from the rate the engine's own tests are pinned to. The valuation date is
not typed either: it is the last close in the target's price file, unless
``as_of`` is set, and it leads the tile row because a DCF without its date is
not a number.

**What is refused, and why.** The peer set's GAAP EBITDA is thin or cannot be
formed, so no peer carries a meaningful EV/EBITDA, the comps produce no median,
and the engine will not invent an exit multiple. The exit-multiple terminal
value and the EV/EBITDA comps range are then refused by name with the engine's
own reasons, and nothing is drawn in their place. The third shield
convention is refused too: ``apv.shield_discount_rate`` offers the cost of debt
and the unlevered cost of equity, and Miles-Ezzell's one year at the cost of
debt is not computed anywhere in the engine.

**Two constructions this section adds, and how each is held to the engine.**
The CLI's football field gives the Gordon row the span of the sensitivity grid
and has no grid for the value-driver form. Here the value-driver band is the
span of ``run_dcf`` evaluated at every cell of that same grid, and every one of
those runs is checked to reproduce the grid's own Gordon cell, so the axes are
provably the engine's. ``SimulationResult`` carries percentiles and not draws,
and a histogram needs the draws, so they are regenerated through the same
``_prepare`` and ``draw_shocks`` path ``run_simulation`` takes and checked to
reproduce its percentiles exactly. A mismatch in either check is a
``ValueError``: it would mean the figure describes a different calculation from
the one the engine reports.

The part that turns measured results into figures is ``shape``, a pure function
of a ``Measured`` record, so the page can be tested without a single fit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

ID = "engine"
TITLE = "Valuation engine"

TICKER = "DDOG"

# The peer set and the risk-free rate tests/conftest.py pins. Used only where
# the assumptions leave them unset, and locked to the conftest fixture by
# tests/dashboard/test_section_engine.py.
SUITE_PEERS: tuple[str, ...] = ("CRWD", "MDB", "ZS")
SUITE_RISK_FREE_RATE = 0.0483

# The market index the beta regressions run against by default.
_INDEX = "SPY"

INPUTS: list[str] = [f"companyfacts_{t}.json" for t in (TICKER, *SUITE_PEERS)] + [
    f"prices/{t}.csv" for t in (TICKER, *SUITE_PEERS, _INDEX)
]

# The comps rows the CLI's football field draws, in its order.
_COMPS_ROWS = ("EV/Revenue", "EV/Gross Profit", "EV/EBITDA")

# The football-field row the exit-multiple terminal value would take.
EXIT_ROW = "DCF, exit multiple"

# A histogram with more bins than this stops reading as a shape.
_MAX_BINS = 36

# The chart kinds this section draws, all of them in the kit.
KINDS = ("tiles", "range", "waterfall", "heat", "hist", "dot")


# --------------------------------------------------------------------------- #
# what was measured
# --------------------------------------------------------------------------- #


@dataclass
class CostOfCapital:
    risk_free_rate: float
    risk_free_pinned: bool
    erp: float
    # Peers whose regressions reached the pool; empty when the WACC fell back to
    # the target's own regression.
    peers: list[str]
    beta_method: str
    beta: float
    # The same beta under the other pooling, and the WACC it would give.
    alternative_method: str | None
    alternative_beta: float | None
    alternative_wacc: float | None
    cost_of_equity: float
    wacc: float
    weight_equity: float
    weight_debt: float
    after_tax_cost_of_debt: float
    debt_to_equity: float


@dataclass
class Bridge:
    equity_value: float
    diluted_shares: float
    # The non-zero items between equity value and enterprise value, signed.
    steps: list[tuple[str, float]]
    enterprise_value: float
    convertible_debt: float
    convertible_treatment: str
    convertible_note: str
    lease_convention: str
    operating_lease: float
    operating_lease_in_debt: float


@dataclass
class Band:
    label: str
    group: str  # "dcf" or "market"
    low: float
    high: float
    mid: float | None = None


@dataclass
class Sensitivity:
    wacc_labels: list[str]
    growth_labels: list[str]
    values: list[list[float | None]]
    wacc_low: float
    wacc_high: float
    growth_low: float
    growth_high: float
    base_wacc: float
    base_growth: float
    base_value: float


@dataclass
class Simulation:
    draws: int
    drivers: int
    correlated: list[float]
    independent: list[float]
    sd_correlated: float
    sd_independent: float
    central: float
    terminal_method: str


@dataclass
class APV:
    terminal_method: str
    wacc: float
    wacc_enterprise_value: float
    unlevered_value: float
    ku: float
    kd: float
    at_cost_of_debt: float
    at_unlevered_cost_of_equity: float
    modelled_interest: float
    filed_interest: float | None


@dataclass
class Refusal:
    what: str
    why: str
    # The figure the refusal belongs under, or None for a section-wide note.
    figure: str | None = None


@dataclass
class Measured:
    ticker: str
    price: float
    price_date: str
    filings_through: str
    cost_of_capital: CostOfCapital
    bridge: Bridge
    bands: list[Band]
    comps_quantiles: tuple[float, float]
    peers_requested: list[str]
    per_share_gordon: float | None = None
    per_share_value_driver: float | None = None
    sensitivity: Sensitivity | None = None
    simulation: Simulation | None = None
    apv: APV | None = None
    refusals: list[Refusal] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# shaping: measured results into figures, with no model in sight
# --------------------------------------------------------------------------- #


def _figure(
    kind: str, title: str, subtitle: str, data: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    return {"kind": kind, "title": title, "subtitle": subtitle, "data": data, **extra}


def _round_step(raw: float) -> float:
    """The smallest of 1, 2, 2.5 and 5 times a power of ten at or above ``raw``."""
    raw = max(raw, 1e-9)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        if multiple * magnitude >= raw * (1 - 1e-12):
            return multiple * magnitude
    return 10 * magnitude


def bin_draws(
    first: Sequence[float], second: Sequence[float], max_bins: int = _MAX_BINS
) -> tuple[list[float], list[int], list[int]]:
    """Shared edges on a round step, and the count of each sample in each bin.

    Both samples go on one set of edges, so a bar in one series and the bar
    beside it cover the same values. The step is the smallest of 1, 2, 2.5 and 5
    times a power of ten that fits the combined range in ``max_bins``, and the
    edges sit on multiples of it, so every draw is counted and the table reads
    in round numbers.
    """
    import numpy as np

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("bin_draws needs two non-empty samples")
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    step = _round_step((hi - lo) / max_bins)
    while True:
        start = math.floor(lo / step)
        stop = max(math.ceil(hi / step), start + 1)
        if stop - start <= max_bins:
            break
        step = _round_step(step * 1.01)
    edges = [round(k * step, 10) for k in range(start, stop + 1)]
    counts_a = np.histogram(a, bins=edges)[0].tolist()
    counts_b = np.histogram(b, bins=edges)[0].tolist()
    return edges, [int(c) for c in counts_a], [int(c) for c in counts_b]


def _share(value: float, of: float) -> str:
    return f"{value / of:.0%}"


# How the engine's setting names read in a sentence.
_NAMES = {
    "median": "median",
    "vasicek": "Vasicek",
    "gordon": "Gordon",
    "value_driver": "value-driver",
}


def _name(setting: str) -> str:
    return _NAMES.get(setting, setting.replace("_", " "))


def _ordinal(q: float) -> str:
    n = round(q * 100)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _tiles(m: Measured) -> dict[str, Any]:
    c = m.cost_of_capital
    peers = ", ".join(c.peers)
    tiles: list[dict[str, Any]] = [
        {
            "label": "Valuation date",
            "value": m.price_date,
            "sub": f"{m.ticker} close of {m.price:,.2f}; filings through {m.filings_through}",
        },
        {
            "label": "Risk-free rate",
            "value": c.risk_free_rate,
            "format": "pct:2",
            "sub": (
                "Pinned as the test suite pins it; no Treasury quote is read offline"
                if c.risk_free_pinned
                else "From market.risk_free_rate in the assumptions"
            ),
        },
    ]
    if c.peers:
        tiles.append(
            {
                "label": f"Beta, peer {_name(c.beta_method)}",
                "value": c.beta,
                "format": "num:2",
                "sub": (
                    f"{peers} asset betas, relevered at {c.debt_to_equity:.1%} "
                    "debt to equity"
                ),
            }
        )
        if c.alternative_beta is not None and c.alternative_wacc is not None:
            tiles.append(
                {
                    "label": f"Beta, peer {_name(c.alternative_method or '')}",
                    "value": c.alternative_beta,
                    "format": "num:2",
                    "sub": f"Would give a {c.alternative_wacc:.2%} WACC; not used",
                }
            )
    else:
        tiles.append(
            {
                "label": "Beta, own regression",
                "value": c.beta,
                "format": "num:2",
                "sub": f"No peer regression survived, so {m.ticker}'s own is used",
            }
        )
    tiles += [
        {
            "label": "Equity risk premium",
            "value": c.erp,
            "format": "pct:2",
            "sub": "From market.equity_risk_premium",
        },
        {
            "label": "Cost of equity",
            "value": c.cost_of_equity,
            "format": "pct:2",
            "sub": f"CAPM on the {c.beta:.2f} beta",
        },
        {
            "label": "WACC",
            "value": c.wacc,
            "format": "pct:2",
            "sub": (
                f"{c.weight_equity:.1%} equity at {c.cost_of_equity:.2%}, "
                f"{c.weight_debt:.1%} debt at {c.after_tax_cost_of_debt:.2%} after tax"
            ),
        },
    ]
    title = f"A {c.wacc:.2%} WACC, with equity at {c.weight_equity:.1%} of capital"
    return _figure("tiles", title, "", {"tiles": tiles})


def _football(m: Measured) -> dict[str, Any] | None:
    bands = [b for b in m.bands if b.high > b.low]
    if not bands:
        return None
    price = m.price
    dcf = [b for b in bands if b.group == "dcf"]
    market = [b for b in bands if b.group == "market"]
    containing = [b for b in market if b.low <= price <= b.high]
    if dcf and all(b.high < price for b in dcf):
        top = max(b.high for b in dcf)
        # Rounded up, so "below" stays true of the unrounded share.
        ceiling = math.floor(top / price * 100) + 1
        title = f"Every DCF band tops out below {ceiling}% of the price"
        if market and len(containing) == len(market):
            title += "; every market-based range contains it"
        elif market:
            title += f"; {len(containing)} of {len(market)} market-based ranges contain it"
    elif dcf and all(b.low > price for b in dcf):
        title = "Every DCF band sits above the price"
    else:
        inside = [b for b in bands if b.low <= price <= b.high]
        title = f"{len(inside)} of {len(bands)} methods bracket the price"

    lo_q, hi_q = m.comps_quantiles
    parts = [f"{m.ticker}, USD per share at the {m.price_date} close."]
    if m.sensitivity is not None and dcf:
        s = m.sensitivity
        parts.append(
            f"DCF bands span WACC {s.wacc_low:.2%} to {s.wacc_high:.2%} and terminal "
            f"growth {s.growth_low:.1%} to {s.growth_high:.1%}, ticked at the base case."
        )
    if any(b.label.startswith("Comps") for b in market):
        parts.append(
            f"Comps bands apply the {_ordinal(lo_q)} to {_ordinal(hi_q)} percentile "
            f"peer multiple "
            f"({', '.join(m.peers_requested)})."
        )
    rows = [
        {
            "label": b.label,
            "lo": b.low,
            "mid": b.mid,
            "hi": b.high,
            "role": _band_role(b),
        }
        for b in bands
    ]
    return _figure(
        "range",
        title,
        " ".join(parts),
        {
            "rows": rows,
            "format": "num:2",
            "reference": [{"value": price, "label": f"Price {price:,.2f}"}],
            "roleLabels": {
                "baseline": "Trading range, for reference only",
                "alt": "Trading comps",
                "model": "DCF",
            },
            "labelHeader": "Method",
        },
        wide=True,
    )


def _band_role(b: Band) -> str:
    """The DCF is the engine's own; comps are the named comparison; the trading range is context, not a value."""
    if b.group == "dcf":
        return "model"
    return "alt" if b.label.startswith("Comps") else "baseline"


def _bridge(m: Measured) -> dict[str, Any]:
    b = m.bridge
    gap = b.enterprise_value - b.equity_value
    if gap < 0:
        title = (
            f"Cash outweighs debt, so enterprise value sits {-gap:,.0f}mm below "
            "equity value"
        )
    else:
        title = f"Claims ahead of the common add {gap:,.0f}mm to equity value"
    parts = [f"USD mm. {m.price:,.2f} a share on {b.diluted_shares:,.1f}mm diluted shares."]
    if b.convertible_note:
        parts.append(b.convertible_note)
    lease = f"Lease convention: {b.lease_convention}"
    if b.operating_lease > 0 and b.operating_lease_in_debt == 0:
        lease += (
            f", so the {b.operating_lease:,.0f}mm operating lease liability stays "
            "out of debt"
        )
    parts.append(lease + ".")
    return _figure(
        "waterfall",
        title,
        " ".join(parts),
        {
            "start": {"label": "Equity value", "value": b.equity_value},
            "steps": [{"label": label, "value": value} for label, value in b.steps],
            "total": {"label": "Enterprise value", "value": b.enterprise_value},
            "format": "num:0",
            "valueLabel": "USD mm",
            "upLabel": "Adds to enterprise value",
            "downLabel": "Subtracts",
            "totalLegend": "Equity or enterprise value",
        },
    )


def _heat(m: Measured) -> dict[str, Any]:
    s = m.sensitivity
    assert s is not None
    cells = [v for row in s.values for v in row if v is not None]
    richest = max(cells)
    clearing = sum(1 for v in cells if v >= m.price)
    if clearing == 0:
        title = (
            f"The richest cell, {richest:,.2f} a share, is "
            f"{_share(richest, m.price)} of the price"
        )
    else:
        title = f"{clearing} of {len(cells)} cells clear the {m.price:,.2f} price"
    subtitle = (
        f"Gordon DCF, implied value per share in USD, WACC down the side and "
        f"terminal growth across the top. The base case, a "
        f"{s.base_wacc:.2%} WACC and {s.base_growth:.1%} terminal growth, gives "
        f"{s.base_value:,.2f} against a price of {m.price:,.2f}."
    )
    data: dict[str, Any] = {
        "rows": _wacc_rows(s),
        "cols": s.growth_labels,
        "values": s.values,
        "scale": "plain",
        "format": "num:2",
        "valueLabel": "Value per share",
        "rowHeader": "WACC",
        "colTitle": "Terminal growth",
    }
    base = _base_cell(s)
    if base is not None:
        data["base"] = base
    return _figure("heat", title, subtitle, data)


def _wacc_rows(s: Sensitivity) -> list[str]:
    """WACC to two decimals, so the middle row reads as the base case it is (12.97%, not 13.0%)."""
    n = len(s.wacc_labels)
    if n < 2:
        return list(s.wacc_labels)
    step = (s.wacc_high - s.wacc_low) / (n - 1)
    return [f"{s.wacc_low + i * step:.2%}" for i in range(n)]


def _base_cell(s: Sensitivity) -> list[int] | None:
    """The grid is struck symmetrically around the base case, so it is the middle cell, if that cell holds the base value."""
    rows, cols = len(s.values), len(s.growth_labels)
    if rows % 2 == 0 or cols % 2 == 0:
        return None
    i, j = rows // 2, cols // 2
    v = s.values[i][j] if j < len(s.values[i]) else None
    return [i, j] if v is not None and abs(v - s.base_value) < 0.005 else None


def _hist(m: Measured) -> dict[str, Any]:
    s = m.simulation
    assert s is not None
    edges, correlated, independent = bin_draws(s.correlated, s.independent)
    ratio = s.sd_correlated / s.sd_independent
    if ratio >= 1:
        title = f"Correlating the drivers widens the spread by {ratio - 1:.0%}"
    else:
        title = f"Correlating the drivers narrows the spread by {1 - ratio:.0%}"
    subtitle = (
        f"Value per share, USD, {s.draws:,} seeded draws each on the "
        f"{_name(s.terminal_method)} terminal value. Standard deviation "
        f"{s.sd_correlated:,.2f} correlated against {s.sd_independent:,.2f} "
        f"independent, a ratio of {ratio:.2f}."
    )
    top = max(max(s.correlated), max(s.independent))
    if top < m.price:
        subtitle += f" The {m.price:,.2f} price lies above every draw."
    return _figure(
        "hist",
        title,
        subtitle,
        {
            "edges": edges,
            "series": [
                {"name": "Correlated drivers", "role": "model", "counts": correlated},
                {"name": "Independent drivers", "role": "baseline", "counts": independent},
            ],
            "format": "num:0",
            "binLabel": "Value per share",
            # Not drawn: the figures the subtitle quotes, kept as numbers.
            "sd": {
                "correlated": s.sd_correlated,
                "independent": s.sd_independent,
                "ratio": ratio,
            },
            "reference": [{"value": s.central, "label": f"Base case {s.central:,.2f}"}],
        },
        wide=True,
    )


def _apv(m: Measured) -> dict[str, Any]:
    a = m.apv
    assert a is not None
    rows = [
        ("No shield (unlevered)", a.unlevered_value),
        (f"Harris-Pringle, Ku {a.ku:.2%}", a.at_unlevered_cost_of_equity),
        (f"Modigliani-Miller, Kd {a.kd:.2%}", a.at_cost_of_debt),
    ]
    base = a.wacc_enterprise_value
    # The dots differ by a few percent of a large number, so the axis is fitted
    # to them on a round step with a quarter step of room either side.
    points = [value for _, value in rows] + [base]
    step = _round_step((max(points) - min(points)) / 4)
    domain = [
        math.floor((min(points) - step / 4) / step) * step,
        math.ceil((max(points) + step / 4) / step) * step,
    ]
    shielded = [a.at_unlevered_cost_of_equity, a.at_cost_of_debt]
    if a.unlevered_value > base:
        title = "APV clears the WACC DCF before any tax shield is added"
    elif all(v > base for v in shielded):
        title = "APV sits above the WACC DCF only once the shield is added"
    elif all(v < base for v in shielded):
        title = "APV sits below the WACC DCF under both shield rates"
    else:
        title = "The shield rate decides which side of the WACC DCF APV lands"
    method = _name(a.terminal_method)
    gap_bp = (a.ku - a.wacc) * 1e4
    if round(gap_bp) == 0:
        relation = "level with the WACC"
    else:
        relation = f"{abs(gap_bp):,.0f}bp {'above' if gap_bp > 0 else 'below'} the WACC"
    subtitle = (
        f"Enterprise value, USD mm, on the {method} terminal value: {base:,.0f} "
        f"from the WACC DCF at {a.wacc:.2%}. The unlevered cost of equity is "
        f"{a.ku:.2%}, {relation}."
    )
    if a.filed_interest and a.filed_interest > 0:
        subtitle += (
            f" Modelled interest of {a.modelled_interest:,.1f}mm is "
            f"{a.modelled_interest / a.filed_interest:,.1f}x the "
            f"{a.filed_interest:,.1f}mm filed."
        )
    return _figure(
        "dot",
        title,
        subtitle,
        {
            "rows": [
                {"label": label, "values": {"apv": value, "wacc": base}}
                for label, value in rows
            ],
            "series": [
                {"key": "apv", "name": "APV", "role": "alt"},
                {"key": "wacc", "name": "WACC DCF", "role": "model"},
            ],
            "format": "num:0",
            "gapLabel": "APV minus WACC DCF",
            "gapFormat": "signed:0",
            "labelHeader": "Shield convention",
            "domain": domain,
            "height": 190,
        },
        wide=True,
    )


_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _takeaway(m: Measured) -> str:
    c = m.cost_of_capital
    when = (
        f"At the {m.price_date} close of {m.price:,.2f}, on a {c.risk_free_rate:.2%} "
        f"risk-free rate and a {c.wacc:.2%} WACC"
    )
    if m.per_share_gordon is None:
        return f"{when}, no DCF could be formed for {m.ticker}."
    text = (
        f"{when}, the DCF values {m.ticker} at {m.per_share_gordon:,.2f} a share on "
        "Gordon growth"
    )
    if m.per_share_value_driver is not None:
        text += (
            f" and {m.per_share_value_driver:,.2f} on the value driver, "
            f"{_share(m.per_share_gordon, m.price)} and "
            f"{_share(m.per_share_value_driver, m.price)} of the price."
        )
    else:
        text += f", {_share(m.per_share_gordon, m.price)} of the price."
    if any(r.what == EXIT_ROW for r in m.refusals):
        text += " No exit-multiple value is drawn; the note under the football field says why."
    if m.simulation is not None:
        ratio = m.simulation.sd_correlated / m.simulation.sd_independent
        verb = "widens" if ratio >= 1 else "narrows"
        drivers = _WORDS.get(m.simulation.drivers, str(m.simulation.drivers))
        text += (
            f" Correlating the simulation's {drivers} drivers {verb} the spread of "
            f"value by {abs(ratio - 1):.0%}."
        )
    return text


def shape(m: Measured) -> dict[str, Any]:
    """The section as the collector returns it, built from measured values alone."""
    figures: dict[str, dict[str, Any]] = {"cost_of_capital": _tiles(m)}
    football = _football(m)
    if football is not None:
        figures["football"] = football
    figures["bridge"] = _bridge(m)
    if m.sensitivity is not None:
        figures["sensitivity"] = _heat(m)
    if m.simulation is not None:
        figures["montecarlo"] = _hist(m)
    if m.apv is not None:
        figures["apv"] = _apv(m)

    # A band built on one value is not a range: it would draw as a hairline and
    # read as a precise answer. _football leaves it out, and it is said here.
    collapsed = [
        Refusal(
            b.label,
            f"the range collapses to {b.low:,.2f}, a single value, which would draw as "
            "a hairline and read as a precise answer.",
            "football",
        )
        for b in m.bands
        if not b.high > b.low
    ]
    refusals = []
    for r in [*m.refusals, *collapsed]:
        refusals.append({"what": r.what, "why": r.why})
        # A refusal about a row inside a figure is named for the row, so the
        # renderer is told which card it belongs under.
        if r.figure in figures:
            figures[r.figure].setdefault("refused", []).append(r.what)
    return {
        "status": "ok",
        "takeaway": _takeaway(m),
        "refusals": refusals,
        "headline": None,
        "figures": figures,
    }


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #


class _FixtureClient:
    """The committed companyfacts payloads, read through the context so each is declared."""

    def __init__(self, ctx, knowledge_date: date | None) -> None:
        self.ctx = ctx
        self.knowledge_date = knowledge_date
        self._payloads: dict[str, dict] = {}

    def company_facts(self, ticker: str):
        from ...edgar import CompanyFacts

        key = ticker.upper()
        if key not in self._payloads:
            path = self.ctx.input(f"companyfacts_{key}.json")
            self._payloads[key] = json.loads(path.read_text(encoding="utf-8"))
        return CompanyFacts(self._payloads[key], ticker, knowledge_date=self.knowledge_date)

    def ticker_to_cik(self, ticker: str) -> int:
        return self.company_facts(ticker).cik


def _check_fixtures(peers: list[str], index: str) -> None:
    from ...errors import ConfigError

    missing = [
        name
        for name in (
            *(f"companyfacts_{t}.json" for t in (TICKER, *peers)),
            *(f"prices/{t}.csv" for t in (TICKER, *peers, index)),
        )
        if name not in INPUTS
    ]
    if missing:
        raise ConfigError(
            "the engine section runs on committed fixtures only, and the assumptions "
            f"ask for files it does not declare: {', '.join(missing)}. Leave "
            "comps.peers and market.market_index unset to use the suite's peer set."
        )


def _bridge_measured(bridge, assumptions) -> Bridge:
    steps = [
        ("Straight debt", bridge.straight_debt),
        ("Convertible notes", bridge.convertible_in_debt),
        ("Finance leases", bridge.finance_lease),
        ("Operating leases", bridge.operating_lease_in_debt),
        ("Preferred stock", bridge.preferred),
        ("Minority interest", bridge.nci),
        ("Cash", -bridge.cash),
        ("Short-term investments", -bridge.short_term_investments),
    ]
    treatment = bridge.convertible_treatment
    cd = bridge.convertible_debt
    if treatment == "debt":
        note = f"Convertible notes of {cd:,.1f}mm are carried as debt"
        if not assumptions.convertibles.conversion_price:
            note += (
                ", since no conversion price is configured to show their shares "
                "are in the diluted count"
            )
        note += "."
    elif treatment == "if_converted":
        note = (
            f"Convertible notes of {cd:,.1f}mm are carried as equity, their shares "
            "already in the diluted count."
        )
    else:
        note = ""
    return Bridge(
        equity_value=bridge.equity_value,
        diluted_shares=bridge.diluted_shares,
        steps=[(label, float(value)) for label, value in steps if value],
        enterprise_value=bridge.enterprise_value,
        convertible_debt=cd,
        convertible_treatment=treatment,
        convertible_note=note,
        lease_convention=bridge.lease_convention,
        operating_lease=bridge.operating_lease,
        operating_lease_in_debt=bridge.operating_lease_in_debt,
    )


def _note_starting(notes: Sequence[str], prefix: str) -> str | None:
    return next((n for n in notes if n.startswith(prefix)), None)


def _reproduce_draws(fin, bridge, wacc_result, assumptions, result, independent: bool):
    """The per-share draws behind a ``run_simulation`` result, checked against it."""
    from ... import simulation as S

    engine = S._prepare(fin, bridge, wacc_result, assumptions)
    per_share, _, alive = engine.values(S.draw_shocks(assumptions, independent=independent))
    values = per_share[alive]
    if S.Percentiles.of(values) != result.per_share:
        raise ValueError(
            "the draws regenerated for the histogram do not reproduce the percentiles "
            "run_simulation reported, so the two are not the same sample"
        )
    return values


def collect(ctx) -> dict:
    import numpy as np
    import pandas as pd

    from ... import cli
    from ...apv import run_apv
    from ...comps import run_comps
    from ...dcf import run_dcf, sensitivity_wacc_growth
    from ...edgar import HttpCache
    from ...errors import ConfigError, TechvalError
    from ...ev_bridge import build_ev_bridge
    from ...financials import build_financials
    from ...market import CsvSource, MarketData
    from ...simulation import DRIVERS, run_simulation
    from ...wacc import compute_wacc

    a = ctx.assumptions.model_copy(deep=True)
    if a.dilution.method != "waso":
        raise ConfigError(
            "dilution.method is 'treasury_stock', whose share count reads award tables "
            "the committed fixtures do not carry. The engine section values on diluted "
            "WASO only; leave dilution.method at 'waso'."
        )
    peers = [p.upper() for p in a.comps.peers] or list(SUITE_PEERS)
    a.comps.peers = peers
    _check_fixtures(peers, a.market.market_index.upper())
    pinned = a.market.risk_free_rate is None
    if pinned:
        a.market.risk_free_rate = SUITE_RISK_FREE_RATE

    knowledge = date.fromisoformat(a.as_of) if a.as_of else None
    source = CsvSource(ctx.input(f"prices/{TICKER}.csv").parent)
    today = knowledge or source.fetch(TICKER, date.min, date.max).last_date
    client = _FixtureClient(ctx, knowledge)

    target = [f"companyfacts_{TICKER}.json", f"prices/{TICKER}.csv"]
    everything = [
        *(f"companyfacts_{t}.json" for t in (TICKER, *peers)),
        *(f"prices/{t}.csv" for t in (TICKER, *peers, a.market.market_index.upper())),
    ]
    refusals: list[Refusal] = []

    # -- the bridge ---------------------------------------------------------- #
    with ctx.record("bridge", "techval.ev_bridge.build_ev_bridge", target):
        fin = build_financials(TICKER, facts=client.company_facts(TICKER))
        market = MarketData(source, HttpCache(enabled=False), today=today)
        price = market.spot(TICKER)
        bridge = build_ev_bridge(fin, price, a)
    series = market.prices(TICKER)

    # -- comps and the cost of capital --------------------------------------- #
    with ctx.record("football", "techval.comps.run_comps", everything):
        comps = run_comps(TICKER, a, client, market)

    with ctx.record("cost_of_capital", "techval.wacc.compute_wacc", everything):
        peer_betas = cli._peer_betas(comps, market, a)
        method = a.market.peer_beta_method
        alternative = None
        alt_wacc = None
        if peer_betas:
            alternative = "vasicek" if method == "median" else "median"
            b = a.model_copy(deep=True)
            b.market.peer_beta_method = alternative
            alt_wacc = compute_wacc(fin, bridge, market, b, peer_betas=peer_betas)
        w = compute_wacc(fin, bridge, market, a, peer_betas=peer_betas)
    cost = CostOfCapital(
        risk_free_rate=w.risk_free_rate,
        risk_free_pinned=pinned,
        erp=w.erp,
        peers=[e.ticker for e in peer_betas] if peer_betas else [],
        beta_method=method,
        beta=w.levered_beta,
        alternative_method=alternative,
        alternative_beta=alt_wacc.levered_beta if alt_wacc else None,
        alternative_wacc=alt_wacc.wacc if alt_wacc else None,
        cost_of_equity=w.cost_of_equity,
        wacc=w.wacc,
        weight_equity=w.weight_equity,
        weight_debt=w.weight_debt,
        after_tax_cost_of_debt=w.after_tax_cost_of_debt,
        debt_to_equity=bridge.total_debt / bridge.equity_value,
    )

    # -- market-based ranges -------------------------------------------------- #
    bands: list[Band] = []
    with ctx.record(
        "football", "techval.market.PriceSeries.fifty_two_week_range", [f"prices/{TICKER}.csv"]
    ):
        lo, hi = series.fifty_two_week_range()
    bands.append(Band("52-week trading range", "market", lo, hi))
    for label in _COMPS_ROWS:
        row = comps.implied.loc[label] if label in comps.implied.index else None
        low = row.get("Implied price low") if row is not None else None
        high = row.get("Implied price high") if row is not None else None
        formed = low is not None and high is not None and pd.notna(low) and pd.notna(high)
        if formed and low > 0:
            bands.append(Band(f"Comps, {label}", "market", float(low), float(high)))
        else:
            why = _note_starting(comps.notes, f"{label} is not applied") or (
                f"the comps produced no positive implied price range on {label}."
            )
            refusals.append(Refusal(f"Comps, {label}", why, "football"))

    measured = Measured(
        ticker=TICKER,
        price=price,
        price_date=series.last_date.isoformat(),
        filings_through=str(fin.as_of),
        cost_of_capital=cost,
        bridge=_bridge_measured(bridge, a),
        bands=bands,
        comps_quantiles=tuple(a.comps.apply_percentiles),
        peers_requested=peers,
        refusals=refusals,
    )

    # -- the DCF -------------------------------------------------------------- #
    median = comps.stats.loc["Median"].get("EV/EBITDA")
    median = float(median) if median is not None and pd.notna(median) else None
    try:
        with ctx.record("football", "techval.dcf.run_dcf", everything):
            dcf = run_dcf(fin, bridge, w, a, peer_median_ev_ebitda=median)
    except TechvalError as exc:
        refusals.append(Refusal("Discounted cash flow", str(exc)))
        return shape(measured)
    measured.per_share_gordon = dcf.per_share_gordon
    measured.per_share_value_driver = dcf.per_share_value_driver

    if dcf.per_share_exit is None:
        why = _note_starting(dcf.notes, "No exit multiple") or _note_starting(
            dcf.notes, "Terminal EBITDA of"
        )
        if median is None:
            flags = [
                f"{p.ticker}: {flag.removeprefix('EV/EBITDA NM: ')}"
                for p in comps.peers
                for flag in p.flags
                if flag.startswith("EV/EBITDA NM: ")
            ]
            why = (
                f"dcf.exit_multiple is unset and no peer in {', '.join(peers)} has a "
                "meaningful EV/EBITDA to take a median of"
                + (f" ({'; '.join(flags)})" if flags else "")
                + ". The engine does not invent a multiple, so there is no exit-multiple "
                "terminal value to draw."
            )
        why = why or "run_dcf formed no exit-multiple terminal value."
        refusals.append(Refusal(EXIT_ROW, why, "football"))

    # -- the sensitivity grid, and the DCF bands struck on it ------------------ #
    try:
        with ctx.record("sensitivity", "techval.dcf.sensitivity_wacc_growth", everything):
            grid = sensitivity_wacc_growth(fin, bridge, a, dcf.wacc)
    except TechvalError as exc:
        refusals.append(Refusal("DCF sensitivity grid", str(exc)))
        grid = None

    if grid is not None:
        cfg = a.dcf
        waccs = dcf.wacc + np.linspace(
            -cfg.sensitivity_wacc_step, cfg.sensitivity_wacc_step, grid.shape[0]
        )
        growths = cfg.terminal_growth + np.linspace(
            -cfg.sensitivity_growth_step, cfg.sensitivity_growth_step, grid.shape[1]
        )
        cells = grid.to_numpy(dtype=float)
        measured.sensitivity = Sensitivity(
            wacc_labels=[str(x) for x in grid.index],
            growth_labels=[str(x) for x in grid.columns],
            values=[[None if math.isnan(v) else float(v) for v in row] for row in cells],
            wacc_low=float(waccs[0]),
            wacc_high=float(waccs[-1]),
            growth_low=float(growths[0]),
            growth_high=float(growths[-1]),
            base_wacc=dcf.wacc,
            base_growth=cfg.terminal_growth,
            base_value=dcf.per_share_gordon,
        )
        finite = cells[~np.isnan(cells)]
        bands.append(
            Band(
                "DCF, Gordon growth",
                "dcf",
                float(finite.min()),
                float(finite.max()),
                dcf.per_share_gordon,
            )
        )

        # The value-driver form has no grid of its own, so run_dcf is evaluated
        # at every cell of the Gordon grid. Each run must reproduce that cell's
        # Gordon value, which proves the axes rebuilt here are the grid's.
        driver: list[float] = []
        with ctx.record("football", "techval.dcf.run_dcf", everything):
            for i, wi in enumerate(waccs):
                for j, gj in enumerate(growths):
                    if math.isnan(cells[i, j]):
                        continue
                    point = a.model_copy(deep=True)
                    point.dcf.wacc_override = float(wi)
                    point.dcf.terminal_growth = float(gj)
                    run = run_dcf(fin, bridge, w, point, peer_median_ev_ebitda=median)
                    if not math.isclose(run.per_share_gordon, cells[i, j], rel_tol=1e-9):
                        raise ValueError(
                            f"run_dcf at WACC {wi:.4%} and growth {gj:.4%} gives a Gordon "
                            f"value of {run.per_share_gordon} against the grid's "
                            f"{cells[i, j]}, so the rebuilt axes are not the grid's"
                        )
                    if run.per_share_value_driver is not None:
                        driver.append(run.per_share_value_driver)
        if dcf.per_share_value_driver is not None and driver:
            bands.append(
                Band(
                    "DCF, value driver",
                    "dcf",
                    min(driver),
                    max(driver),
                    dcf.per_share_value_driver,
                )
            )
        else:
            why = _note_starting(dcf.notes, "Steady-state NOPAT of") or (
                "run_dcf formed no value-driver terminal value."
            )
            refusals.append(Refusal("DCF, value driver", why, "football"))

    # -- Monte Carlo ------------------------------------------------------------ #
    try:
        with ctx.record("montecarlo", "techval.simulation.run_simulation", everything):
            correlated = run_simulation(fin, bridge, w, a, peer_median_ev_ebitda=median)
            independent = run_simulation(
                fin, bridge, w, a, peer_median_ev_ebitda=median, independent=True
            )
            draws_c = _reproduce_draws(fin, bridge, w, a, correlated, independent=False)
            draws_i = _reproduce_draws(fin, bridge, w, a, independent, independent=True)
        measured.simulation = Simulation(
            draws=correlated.draws,
            drivers=len(DRIVERS),
            correlated=[float(v) for v in draws_c],
            independent=[float(v) for v in draws_i],
            sd_correlated=correlated.per_share.sd,
            sd_independent=independent.per_share.sd,
            central=correlated.central_per_share,
            terminal_method=a.dcf.terminal.method,
        )
    except TechvalError as exc:
        refusals.append(Refusal("Monte Carlo", str(exc)))

    # -- APV ------------------------------------------------------------------- #
    try:
        with ctx.record("apv", "techval.apv.run_apv", everything):
            results = {}
            for rate in ("cost_of_debt", "unlevered_cost_of_equity"):
                point = a.model_copy(deep=True)
                point.apv.shield_discount_rate = rate
                results[rate] = run_apv(fin, bridge, w, dcf, point)
        kd_run = results["cost_of_debt"]
        measured.apv = APV(
            terminal_method=kd_run.terminal_method,
            wacc=kd_run.wacc,
            wacc_enterprise_value=kd_run.wacc_enterprise_value,
            unlevered_value=kd_run.unlevered_value,
            ku=kd_run.unlevered_cost_of_equity,
            kd=w.pretax_cost_of_debt,
            at_cost_of_debt=kd_run.total_value,
            at_unlevered_cost_of_equity=results["unlevered_cost_of_equity"].total_value,
            modelled_interest=w.pretax_cost_of_debt * kd_run.debt_balance,
            filed_interest=fin.interest_expense,
        )
        refusals.append(
            Refusal(
                "APV, Miles-Ezzell",
                "The engine computes two shield conventions, the cost of debt "
                "(Modigliani-Miller) and the unlevered cost of equity (Harris-Pringle), "
                "and does not compute Miles-Ezzell. Miles-Ezzell discounts each year's "
                "shield at the cost of debt for its last year and at Ku before that, "
                "which scales the Harris-Pringle value by (1 + Ku) / (1 + Kd). This is "
                "a convention the engine does not offer, not a figure the fixtures "
                "cannot reproduce, and the page draws only what the engine computes.",
                "apv",
            )
        )
    except TechvalError as exc:
        refusals.append(Refusal("APV", str(exc)))

    return shape(measured)
