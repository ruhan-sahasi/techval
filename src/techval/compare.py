"""techval compare: several companies through the same engine, side by side.

A tearsheet answers what one company is worth on the engine's terms. Set
beside another, the question becomes which prices ask the most, and that needs
every company valued the same way: the same assumptions, the same reverse DCF,
the same scenarios read off the same panel. ``measure_all`` measures each
company twice, once on the assumptions file and once on its own record
(``techval.anchors``), skipping the quarterly history neither comparison draws,
and keeps a company the engine cannot value as a named refusal rather than
dropping it from the set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .dashboard.sections import engine as E
from .errors import TechvalError


@dataclass
class Compared:
    """One company measured on the assumptions file and on its own record, or refused."""

    ticker: str
    base: E.Measured | None = None
    record: E.Measured | None = None
    refused: str | None = None


def measure_all(
    tickers: list[str],
    assumptions,
    sources: E.Sources,
    *,
    today: date,
    peers_from_set: bool = False,
) -> list[Compared]:
    """Each ticker measured twice, in the order given; a refusal keeps its place.

    With ``peers_from_set`` each company's peers are the rest of the set, which
    suits a set of comparables and not a mixed one; otherwise each is valued on
    its own beta, as a tearsheet with no peers is.
    """
    symbols = [t.upper() for t in tickers]
    out: list[Compared] = []
    for symbol in symbols:
        peers = [t for t in symbols if t != symbol] if peers_from_set else []
        row = Compared(symbol)
        try:
            for record, attr in ((False, "base"), (True, "record")):
                a = assumptions.model_copy(deep=True)
                a.dcf.from_record = record
                a.comps.peers = peers
                setattr(row, attr, E.measure(symbol, peers, a, sources, today=today, history=False))
        except TechvalError as err:
            # The whole message on one line: its hint names what was missing.
            row = Compared(symbol, refused=" ".join(line.strip() for line in str(err).splitlines()))
        out.append(row)
    return out


def _ratio(m: E.Measured | None) -> float | None:
    if m is None or m.per_share_gordon is None or not m.price:
        return None
    return m.per_share_gordon / m.price


def _implied(m: E.Measured | None) -> float | None:
    return m.expectations.implied_return if m is not None and m.expectations is not None else None


def value_figure(rows: list[Compared]) -> dict:
    """Each company's DCF as a multiple of its close, on the assumptions and on its own record."""
    valued = [c for c in rows if c.base is not None]
    above = sum(1 for c in valued if (_ratio(c.base) or 0) >= 1)
    above_record = sum(1 for c in valued if (_ratio(c.record) or 0) >= 1)
    title = (
        f"{above} of {len(valued)} base cases reach the close, and {above_record} "
        f"{'does' if above_record == 1 else 'do'} on the companies' own records"
    )
    return E._figure(
        "dot",
        title,
        "Gordon DCF value per share over the close. The base case is the assumptions file's; "
        "the record case sets growth, margin and reinvestment from each company's own filings.",
        {
            "series": [
                {"key": "base", "name": "Base case", "role": "model"},
                {"key": "record", "name": "Record case", "role": "alt"},
            ],
            "rows": [
                {"label": c.ticker, "values": {"base": _ratio(c.base), "record": _ratio(c.record)}}
                for c in valued
            ],
            "format": "mult:2",
            "reference": [{"value": 1.0, "label": "The close"}],
            "labelHeader": "Company",
        },
        wide=True,
    )


def returns_figure(rows: list[Compared]) -> dict:
    """What each close earns if its base case holds, beside its cost of capital."""
    valued = [c for c in rows if c.base is not None]
    earning = [(c, _implied(c.base)) for c in valued if _implied(c.base) is not None]
    if earning:
        low, rate = min(earning, key=lambda x: x[1])
        title = (
            f"{low.ticker}'s close earns least, {rate:.1%} a year against a "
            f"{low.base.cost_of_capital.wacc:.1%} cost of capital"
        )
    else:
        title = "No close in the set is reached by any discount rate"
    return E._figure(
        "dot",
        title,
        "The return a buyer at the close earns if the base case holds, the DCF run backwards "
        "for its discount rate, beside the cost of capital the engine discounts at.",
        {
            "series": [
                {"key": "implied", "name": "Implied return", "role": "model"},
                {"key": "wacc", "name": "Cost of capital", "role": "baseline"},
            ],
            "rows": [
                {"label": c.ticker, "values": {"implied": _implied(c.base), "wacc": c.base.cost_of_capital.wacc}}
                for c in valued
            ],
            "format": "pct:1",
            "labelHeader": "Company",
        },
        wide=True,
    )


def _scenarios(m: E.Measured) -> tuple[float | None, str | None]:
    """The weighted scenarios, and where the close sits against them, as a cell."""
    if m.scenarios is None:
        return None, None
    r = m.scenarios.result
    implied = r["implied"]
    if implied["side"] == "above":
        bull = next(s["value"] for s in r["scenarios"] if s["name"] == "bull")
        where = f"{m.price / bull:.1f}x the bull"
    elif implied["side"] == "below":
        where = "below the bear"
    else:
        where = f"{implied['weight']:.0%} {implied['side']}"
    return r["weighted"], where


def table_figure(rows: list[Compared]) -> dict:
    """Every company's reading in one table, in the order asked for."""
    valued = [c for c in rows if c.base is not None]
    out = []
    for c in valued:
        m, exp = c.base, c.base.expectations
        weighted, where = _scenarios(m)
        out.append(
            {
                "company": c.ticker,
                "close": m.price,
                "base": m.per_share_gordon,
                "record": c.record.per_share_gordon if c.record else None,
                "implied": exp.implied_return if exp else None,
                "wacc": m.cost_of_capital.wacc,
                "growth": exp.implied_growth if exp else None,
                "did": f"{exp.base_hits:,} of {exp.base_n:,}" if exp and exp.base_n else "n/a",
                "weighted": weighted,
                "against": where or "n/a",
            }
        )
    return E._figure(
        "table",
        f"{len(valued)} companies through the same engine, on the same assumptions",
        "Per share, at each company's last close. Growth needed is the first-year growth the "
        "close needs with everything else as assumed; the company-years beside it are the fade "
        "panel's that compounded that path.",
        {
            "columns": [
                {"key": "company", "label": "Company"},
                {"key": "close", "label": "Close", "align": "right", "format": "num:2"},
                {"key": "base", "label": "Base case", "align": "right", "format": "num:2"},
                {"key": "record", "label": "Record case", "align": "right", "format": "num:2"},
                {"key": "implied", "label": "Implied return", "align": "right", "format": "pct:1"},
                {"key": "wacc", "label": "Cost of capital", "align": "right", "format": "pct:1"},
                {"key": "growth", "label": "Growth needed", "align": "right", "format": "pct:0"},
                {"key": "did", "label": "Company-years that did it", "align": "right"},
                {"key": "weighted", "label": "Scenarios, weighted", "align": "right", "format": "num:2"},
                {"key": "against", "label": "Close against them", "align": "right"},
            ],
            "rows": out,
        },
        wide=True,
    )


def takeaway(rows: list[Compared], when: str) -> str:
    """The set in two sentences: how many prices the base case reaches, and which asks most."""
    valued = [c for c in rows if c.base is not None]
    if not valued:
        return "No company in the set could be valued."
    above = sum(1 for c in valued if (_ratio(c.base) or 0) >= 1)
    first = f"At the closes of {when}, the base case reaches the price of {above} of the {len(valued)} companies valued."
    needing = [c for c in valued if c.base.expectations and c.base.expectations.implied_growth is not None]
    if not needing:
        return first
    most = max(needing, key=lambda c: c.base.expectations.implied_growth)
    exp = most.base.expectations
    did = (
        f", a path {exp.base_hits:,} of {exp.base_n:,} TMT company-years compounded"
        if exp.base_n
        else ""
    )
    return first + (
        f" {most.ticker}'s price asks the most: {exp.implied_growth:.0%} revenue growth next year{did}."
    )
