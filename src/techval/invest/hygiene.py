"""Portfolio hygiene: weights, concentration, exposure, drift, coverage.

Arithmetic on the owner's own book and targets, never a view on markets. The
buckets are honest about what the engine knows: a TMT filer takes its
sub-vertical from the fade universe, ETFs and crypto and cash are their own
buckets, and a stock outside the taxonomy says so instead of being guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .ledger import Ledger
from .quotes import Quotes


@lru_cache(maxsize=1)
def _universe() -> dict[str, str]:
    from ..ml.forecast import fade_universe

    return fade_universe()


def sub_vertical(symbol: str) -> str | None:
    """The engine's own bucket for a ticker, or None outside its universe."""
    return _universe().get(symbol.upper())


@dataclass
class Weight:
    symbol: str
    kind: str
    value: float
    weight: float


def weights(ledger: Ledger, quotes: Quotes) -> list[Weight]:
    """Every position at today's close plus a cash row, largest first."""
    rows = []
    for symbol, p in ledger.positions().items():
        if p.shares <= 0:
            continue
        rows.append((symbol, p.kind, p.shares * quotes.close_on(symbol, quotes.today, p.kind)))
    cash = ledger.cash()
    total = sum(v for _, _, v in rows) + cash
    out = [Weight(s, k, v, v / total if total else 0.0) for s, k, v in rows]
    out.sort(key=lambda w: (-w.value, w.symbol))
    out.append(Weight("cash", "cash", cash, cash / total if total else 0.0))
    return out


def top_share(rows: list[Weight], n: int = 5) -> float:
    return sum(sorted((r.weight for r in rows), reverse=True)[:n])


def exposure(rows: list[Weight]) -> dict[str, float]:
    """Weight by bucket: sub-verticals, index funds, crypto, cash, outside TMT."""
    buckets: dict[str, float] = {}
    for r in rows:
        if r.kind == "cash":
            bucket = "cash"
        elif r.kind == "etf":
            bucket = "index funds"
        elif r.kind == "crypto":
            bucket = "crypto"
        else:
            vertical = sub_vertical(r.symbol)
            bucket = vertical.replace("_", " ") if vertical else "outside TMT"
        buckets[bucket] = buckets.get(bucket, 0.0) + r.weight
    return buckets


def drift(rows: list[Weight], targets: dict[str, float]) -> list[dict]:
    """Weight against each stated target, in the targets' own order."""
    if not targets:
        return []
    by = {r.symbol: r.weight for r in rows}
    return [
        {"symbol": symbol, "weight": by.get(symbol, 0.0), "target": target, "gap": by.get(symbol, 0.0) - target}
        for symbol, target in targets.items()
    ]


def coverage(rows: list[Weight]) -> dict[str, float]:
    """How much of the book the engine can value at all."""
    held = [r for r in rows if r.kind != "cash"]
    covered = [r for r in held if r.kind == "stock" and sub_vertical(r.symbol) is not None]
    total = sum(r.value for r in rows)
    return {
        "positions": len(held),
        "covered_positions": len(covered),
        "covered_value_share": sum(r.value for r in covered) / total if total else 0.0,
    }
