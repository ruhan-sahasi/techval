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
