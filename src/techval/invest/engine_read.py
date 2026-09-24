"""What the engine's models say about each holding, from the committed panels.

The fade curve and the warranted multiple both fit offline from panels the
repo already carries, so this costs two fits however many names are held. A
name outside a panel gets a refusal naming the pane; every figure travels with
the verdict its model earned, because a residual quoted without "the value
signal lost to a random score" is how a research tool becomes a tip sheet.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import Assumptions
from ..errors import TechvalError


@dataclass
class EngineRead:
    symbol: str
    covered: bool
    sub_vertical: str | None
    fade: dict | None = None
    warranted: dict | None = None
    dcf: dict | None = None
    refusals: list[dict] = field(default_factory=list)


@lru_cache(maxsize=2)
def fade_model(fixtures: Path):
    """One fade fit on the committed panel, shared by every read."""
    from ..edgar import CompanyFacts
    from ..ml.forecast import build_fade_panel, fade_universe, fit_fade

    blob = json.load(gzip.open(Path(fixtures) / "fade_companyfacts.json.gz", "rt"))

    class _Client:
        def company_facts(self, ticker: str) -> CompanyFacts:
            return CompanyFacts(blob["payloads"][ticker.upper()], ticker)

    universe = {t: v for t, v in fade_universe().items() if t in blob["payloads"]}
    panel = build_fade_panel(universe, _Client())
    assumptions = Assumptions()
    assumptions.ml.forecast.enabled = True
    return fit_fade(panel, assumptions)


@lru_cache(maxsize=2)
def warranted_model(fixtures: Path):
    """One warranted fit on the recorded panel, shared by every read."""
    from ..commands_peers import _load_observations
    from ..ml.warranted import fit_warranted

    panel = _load_observations(Path(fixtures) / "warranted" / "observations.json.gz")
    return fit_warranted(panel, Assumptions(), model="mlp")


def _assumed_line(assumptions: Assumptions) -> list[float]:
    """The engine's own straight fade, the line every DCF takes by default."""
    years = assumptions.dcf.projection_years
    start = assumptions.dcf.revenue_growth_start
    end = assumptions.dcf.revenue_growth_terminal
    if years == 1:
        return [start]
    step = (start - end) / (years - 1)
    return [start - step * i for i in range(years)]


def read_holdings(
    symbols: list[str],
    *,
    fixtures: Path,
    assumptions: Assumptions,
) -> dict[str, EngineRead]:
    from ..ml.forecast import fade_universe

    universe = fade_universe()
    fixtures = Path(fixtures)
    out: dict[str, EngineRead] = {}
    for raw in symbols:
        symbol = raw.upper()
        vertical = universe.get(symbol)
        read = EngineRead(symbol=symbol, covered=vertical is not None, sub_vertical=vertical)
        if vertical is None:
            read.refusals = [
                {"what": "Fade path", "why": f"{symbol} is outside the engine's TMT universe, so no fade curve was fitted for it"},
                {"what": "Warranted multiple", "why": f"{symbol} is outside the engine's TMT universe, so it has no warranted line to sit against"},
            ]
            out[symbol] = read
            continue
        read.fade = _fade_read(symbol, fixtures, assumptions, read.refusals)
        read.warranted = _warranted_read(symbol, fixtures, read.refusals)
        out[symbol] = read
    return out


def _fade_read(symbol: str, fixtures: Path, assumptions: Assumptions, refusals: list[dict]) -> dict | None:
    model = fade_model(fixtures)
    years = assumptions.dcf.projection_years
    try:
        path = model.path(symbol, years)
    except TechvalError as err:
        refusals.append({"what": "Fade path", "why": str(err)})
        return None
    latest = model.latest.get(symbol)
    trailing = None if latest is None or latest.growth is None else round(latest.growth, 6)
    return {
        "years": list(range(1, years + 1)),
        "fitted": [round(g, 6) for g in path.growth],
        "lower": [round(g, 6) for g in path.lower],
        "upper": [round(g, 6) for g in path.upper],
        "basis": list(path.basis),
        "assumed": [round(g, 6) for g in _assumed_line(assumptions)],
        "trailing": trailing,
        "as_of": path.as_of.isoformat(),
        "verdict": model.fits[min(model.fits)].evaluation.verdict(),
    }


def _warranted_read(symbol: str, fixtures: Path, refusals: list[dict]) -> dict | None:
    model = warranted_model(fixtures)
    try:
        read = model.warranted(symbol)
    except TechvalError as err:
        refusals.append({"what": "Warranted multiple", "why": str(err)})
        return None
    return {
        "traded": round(read.actual_multiple, 6),
        "warranted": round(read.warranted_multiple, 6),
        "residual_log": round(read.residual_log, 6),
        "residual_turns": round(read.residual_turns, 6),
        "z": round(read.z, 6),
        "call": "rich" if read.residual_log > 0 else "cheap",
        "as_of": model.latest.isoformat(),
        "verdict": model.verdict(),
    }


def attach_dcf(reads: dict[str, EngineRead], *, facts_for, market, assumptions: Assumptions) -> None:
    """Value each covered holding through the engine's own pipeline, in place.

    ``facts_for`` returns a ``CompanyFacts`` or None, and it is the caller who
    decides where facts may come from: the CLI passes a cached-or-live loader,
    the tests a fixture reader. Offline with nothing cached is a refusal, not
    an error, and so is every ``TechvalError`` the pipeline raises: a holding
    the engine cannot value honestly stays on the page with the reason.
    """
    from ..dcf import run_dcf
    from ..ev_bridge import build_ev_bridge
    from ..financials import build_financials
    from ..wacc import compute_wacc

    for symbol, read in reads.items():
        if not read.covered:
            continue
        try:
            facts = facts_for(symbol)
        except TechvalError as err:
            read.refusals.append({"what": "DCF", "why": f"{symbol}: {err}"})
            continue
        if facts is None:
            read.refusals.append(
                {"what": "DCF", "why": f"no filings cached for {symbol}; run without --offline to fetch them"}
            )
            continue
        try:
            fin = build_financials(symbol, facts=facts)
            spot = market.spot(symbol)
            bridge = build_ev_bridge(fin, spot, assumptions)
            wacc = compute_wacc(fin, bridge, market, assumptions)
            result = run_dcf(fin, bridge, wacc, assumptions)
        except TechvalError as err:
            read.refusals.append({"what": "DCF", "why": f"{symbol}: {err}"})
            continue
        read.dcf = {
            "per_share": round(result.per_share, 4),
            "price": round(spot, 4),
            "gap_pct": round(result.per_share / spot - 1.0, 6),
            "wacc": round(wacc.wacc, 6),
            "enterprise_value_mm": round(result.enterprise_value, 2),
        }
