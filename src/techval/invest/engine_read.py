"""What the engine's models say about each holding, from the committed panels.

The fade curve and the warranted multiple both fit offline from panels the
repo already carries, so this costs two fits however many names are held. A
name outside a panel gets a refusal naming the pane; every figure travels with
the verdict its model earned, because a residual quoted without "the value
signal lost to a random score" is how a research tool becomes a tip sheet.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
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


# Bumped whenever what a cached fit means changes without its inputs changing.
FIT_SCHEMA = 1

FADE_PANEL = "fade_companyfacts.json.gz"
WARRANTED_PANEL = "warranted/observations.json.gz"

# The modules whose source moves each fit. A change to any of them is a cache
# miss, the same rule the dashboard collector keys its sections on.
_FADE_MODULES = ("techval.ml.forecast", "techval.tmt.taxonomy", "techval.invest.engine_read")
_WARRANTED_MODULES = (
    "techval.ml.warranted",
    "techval.ml.nn",
    "techval.ml.features",
    "techval.commands_peers",
    "techval.invest.engine_read",
)


def fit_key(kind: str, inputs: list[Path], modules: tuple[str, ...]) -> str:
    """A key over the panel bytes, the fitting code and the cache schema."""
    h = hashlib.sha256(f"{kind}:{FIT_SCHEMA}".encode())
    for path in inputs:
        h.update(Path(path).read_bytes())
    for name in modules:
        h.update(Path(importlib.import_module(name).__file__).read_bytes())
    return h.hexdigest()[:16]


def cached_fit(kind: str, inputs: list[Path], modules: tuple[str, ...], build, cache_root: Path | None, refit: bool = False):
    """Load a fit from the disk cache, or build it and store it.

    The fits take nine seconds together and are pure functions of committed
    panels, so a second run of techval invest should not pay for them again.
    An unreadable or unwritable entry is a miss, never a failed run; with no
    cache root there is no disk cache at all, which is how the tests run.
    """
    if cache_root is None:
        return build()
    import joblib

    path = Path(cache_root).expanduser() / f"invest_{kind}_{fit_key(kind, inputs, modules)}.joblib"
    if path.is_file() and not refit:
        try:
            return joblib.load(path)
        except Exception:  # noqa: BLE001 - an unreadable cache is a miss
            pass
    model = build()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=3)
    except Exception:  # noqa: BLE001 - an unwritable cache slows the next run only
        pass
    return model


def missing_panels(fixtures: Path) -> dict[str, Path]:
    """Which committed panels are absent, by the pane that reads them."""
    fixtures = Path(fixtures)
    wanted = {"Fade path": fixtures / FADE_PANEL, "Warranted multiple": fixtures / WARRANTED_PANEL}
    return {pane: path for pane, path in wanted.items() if not path.is_file()}


def panel_refusal(path: Path) -> str:
    return (
        f"the model panel {path.name} is not at {path}. The panels ship in a techval "
        "checkout under tests/fixtures; point --panels at that directory"
    )


def _normal(fixtures, cache_root, refit) -> tuple:
    """One lru key per fit however the call spells its arguments."""
    return Path(fixtures), None if cache_root is None else Path(cache_root), bool(refit)


def fade_model(fixtures: Path, cache_root: Path | None = None, refit: bool = False):
    """One fade fit on the committed panel, shared by every read."""
    return _fade_model(*_normal(fixtures, cache_root, refit))


def warranted_model(fixtures: Path, cache_root: Path | None = None, refit: bool = False):
    """One warranted fit on the recorded panel, shared by every read."""
    return _warranted_model(*_normal(fixtures, cache_root, refit))


def fade_panel(fixtures: Path, cache_root: Path | None = None, refit: bool = False):
    """The point-in-time fade panel itself, for base rates as well as the fit."""
    return _fade_panel(*_normal(fixtures, cache_root, refit))


@lru_cache(maxsize=4)
def _fade_panel(fixtures: Path, cache_root: Path | None, refit: bool):
    def build():
        from ..edgar import CompanyFacts
        from ..ml.forecast import build_fade_panel, fade_universe

        blob = json.load(gzip.open(fixtures / FADE_PANEL, "rt"))

        class _Client:
            def company_facts(self, ticker: str) -> CompanyFacts:
                return CompanyFacts(blob["payloads"][ticker.upper()], ticker)

        universe = {t: v for t, v in fade_universe().items() if t in blob["payloads"]}
        return build_fade_panel(universe, _Client())

    return cached_fit("fade_panel", [fixtures / FADE_PANEL], _FADE_MODULES, build, cache_root, refit)


@lru_cache(maxsize=4)
def _fade_model(fixtures: Path, cache_root: Path | None, refit: bool):
    def build():
        from ..ml.forecast import fit_fade

        assumptions = Assumptions()
        assumptions.ml.forecast.enabled = True
        return fit_fade(_fade_panel(fixtures, cache_root, refit), assumptions)

    return cached_fit("fade", [fixtures / FADE_PANEL], _FADE_MODULES, build, cache_root, refit)


@lru_cache(maxsize=4)
def _warranted_model(fixtures: Path, cache_root: Path | None, refit: bool):
    def build():
        from ..commands_peers import _load_observations
        from ..ml.warranted import fit_warranted

        panel = _load_observations(fixtures / WARRANTED_PANEL)
        return fit_warranted(panel, Assumptions(), model="mlp")

    return cached_fit("warranted", [fixtures / WARRANTED_PANEL], _WARRANTED_MODULES, build, cache_root, refit)


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
    cache_root: Path | None = None,
    refit: bool = False,
) -> dict[str, EngineRead]:
    from ..ml.forecast import fade_universe

    universe = fade_universe()
    fixtures = Path(fixtures)
    absent = missing_panels(fixtures)
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
        # A missing panel refuses its pane for every holding; the other pane,
        # and the rest of the page, carry on without it.
        if "Fade path" in absent:
            read.refusals.append({"what": "Fade path", "why": f"{symbol}: {panel_refusal(absent['Fade path'])}"})
        else:
            read.fade = _fade_read(fade_model(fixtures, cache_root, refit), symbol, assumptions, read.refusals)
        if "Warranted multiple" in absent:
            read.refusals.append(
                {"what": "Warranted multiple", "why": f"{symbol}: {panel_refusal(absent['Warranted multiple'])}"}
            )
        else:
            read.warranted = _warranted_read(warranted_model(fixtures, cache_root, refit), symbol, read.refusals)
        out[symbol] = read
    return out


def _fade_read(model, symbol: str, assumptions: Assumptions, refusals: list[dict]) -> dict | None:
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


def _warranted_read(model, symbol: str, refusals: list[dict]) -> dict | None:
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


def attach_dcf(
    reads: dict[str, EngineRead], *, facts_for, market, assumptions: Assumptions, observations=None
) -> None:
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
            "expectations": _expectations(fin, bridge, wacc, assumptions, spot, read, observations),
        }


def _expectations(fin, bridge, wacc, assumptions, price, read, observations) -> dict:
    """What the holding's price assumes: the return it earns and the growth it needs.

    The two levers a holder reads first, from techval.reverse_dcf: the return
    a buyer at today's price earns if the base case holds, and the first-year
    growth the price needs, set against the fade panel's base rate when the
    panel is on hand.
    """
    from ..reverse_dcf import (
        BASE_RATE_HORIZON,
        Case,
        growth_base_rate,
        implied_discount_rate,
        implied_first_year_growth,
        path_cagr,
    )

    case = Case(fin=fin, bridge=bridge, wacc=wacc, assumptions=assumptions)
    rate = implied_discount_rate(case, price)
    growth = implied_first_year_growth(case, price)
    base = None
    if observations is not None and growth.reached:
        trailing = read.fade["trailing"] if read.fade else None
        base = growth_base_rate(
            observations,
            implied_cagr=path_cagr(growth.extras["path"], BASE_RATE_HORIZON),
            horizon=BASE_RATE_HORIZON,
            trailing=trailing,
        )
    if base is not None:
        base["implied_cagr"] = round(base["implied_cagr"], 6)
        for bucket in (base["all"], base["similar"]):
            if bucket and bucket["share"] is not None:
                bucket["share"] = round(bucket["share"], 6)
    return {
        "implied_return": None if rate.implied is None else round(rate.implied, 6),
        "implied_growth": None if growth.implied is None else round(growth.implied, 6),
        "base_rate": base,
        "sentences": [rate.sentence, growth.sentence],
    }
