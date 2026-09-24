"""Assemble and validate the portfolio snapshot the page renders from.

One dict, eight panes, every number rounded and sorted so two runs on the same
inputs write identical bytes. ``validate_snapshot`` is the page's contract: a
missing or mistyped key refuses here, at build time, rather than as a blank
region in the browser.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from ..config import Assumptions
from .engine_read import attach_dcf, read_holdings
from .hygiene import coverage, drift, exposure, top_share, weights
from .ideas import ideas
from .ledger import Ledger
from .performance import benchmark_growth, contributions, twr, value_series
from .quotes import Quotes

SCHEMA = 1

# The contract: every pane and the keys it must carry. Types are spot-checked
# where a wrong one would draw nonsense rather than crash.
_REQUIRED = {
    "meta": ("name", "benchmark", "generated", "first_date", "price_source", "schema"),
    "overview": (
        "value", "cash", "day_abs", "day_pct", "twr_pct", "benchmark_twr_pct",
        "n_positions", "cheap", "rich", "uncovered", "top5_share",
        "covered_value_share", "movers",
    ),
    "performance": (
        "dates", "growth", "benchmark_growth", "benchmark", "contributions",
        "lots", "realized_total", "dividends_total",
    ),
    "hygiene": ("weights", "top5_share", "exposure", "drift", "coverage"),
    "ideas": ("as_of", "cheap", "rich", "verdict"),
}

_POSITION_KEYS = (
    "symbol", "kind", "shares", "price", "day_pct", "value", "weight", "cost",
    "unrealized", "unrealized_pct", "realized", "dividends", "covered", "engine",
)


def build_snapshot(
    ledger: Ledger,
    quotes: Quotes,
    *,
    fixtures: Path,
    assumptions: Assumptions,
    today: date,
    facts_for=None,
    market=None,
) -> dict:
    positions = ledger.positions()
    held = [p for p in positions.values() if p.shares > 0]
    cash = ledger.cash()

    reads = read_holdings(
        [p.symbol for p in held if p.kind == "stock"],
        fixtures=fixtures,
        assumptions=assumptions,
    )
    for p in held:
        if p.kind != "stock":
            reads.setdefault(
                p.symbol,
                _uncovered(p.symbol, p.kind),
            )
    if facts_for is not None and market is not None:
        attach_dcf(reads, facts_for=facts_for, market=market, assumptions=assumptions)
    else:
        for read in reads.values():
            if read.covered:
                read.refusals.append(
                    {"what": "DCF", "why": f"no filings cached for {read.symbol}; run without --offline to fetch them"}
                )

    weight_rows = weights(ledger, quotes)
    weight_by = {w.symbol: w for w in weight_rows}
    series = value_series(ledger, quotes)
    growth = twr(series, ledger.flows())
    bench = benchmark_growth(quotes, ledger.benchmark, series.dates)

    rows = []
    movers = []
    for p in sorted(held, key=lambda p: -weight_by[p.symbol].value):
        price, previous = quotes.last_two(p.symbol, p.kind)
        value = weight_by[p.symbol].value
        day_pct = price / previous - 1.0 if previous else 0.0
        read = reads.get(p.symbol)
        engine = None
        if read is not None and read.warranted is not None:
            engine = {"call": read.warranted["call"], "residual_log": read.warranted["residual_log"]}
        rows.append(
            {
                "symbol": p.symbol,
                "kind": p.kind,
                "shares": round(p.shares, 6),
                "price": round(price, 4),
                "day_pct": round(day_pct, 6),
                "value": round(value, 2),
                "weight": round(weight_by[p.symbol].weight, 6),
                "cost": round(p.cost, 2),
                "unrealized": round(value - p.cost, 2),
                "unrealized_pct": round(value / p.cost - 1.0, 6) if p.cost else None,
                "realized": round(p.realized, 2),
                "dividends": round(p.dividends, 2),
                "covered": bool(read is not None and read.covered),
                "engine": engine,
            }
        )
        movers.append({"symbol": p.symbol, "day_pct": round(day_pct, 6), "day_abs": round(value - value / (1 + day_pct), 2) if day_pct > -1 else 0.0})

    movers.sort(key=lambda m: -abs(m["day_abs"]))
    total = sum(r["value"] for r in rows) + cash
    day_abs = sum(m["day_abs"] for m in movers)
    calls = [r["engine"]["call"] for r in rows if r["engine"]]
    cov = coverage(weight_rows)

    lots = []
    for p in held:
        price = quotes.close_on(p.symbol, today, p.kind)
        for lot in p.lots:
            lots.append(
                {
                    "symbol": p.symbol,
                    "opened": lot.opened.isoformat(),
                    "shares": round(lot.shares, 6),
                    "cost_per_share": round(lot.cost_per_share, 4),
                    "price": round(price, 4),
                    "unrealized": round(lot.shares * (price - lot.cost_per_share), 2),
                }
            )

    snapshot = {
        "meta": {
            "name": ledger.name,
            "benchmark": ledger.benchmark,
            "generated": today.isoformat(),
            "first_date": ledger.first_date.isoformat(),
            "price_source": quotes.source.name,
            "schema": SCHEMA,
        },
        "overview": {
            "value": round(total, 2),
            "cash": round(cash, 2),
            "day_abs": round(day_abs, 2),
            "day_pct": round(day_abs / (total - day_abs), 6) if total != day_abs else 0.0,
            "twr_pct": round(float(growth[-1]) - 1.0, 6),
            "benchmark_twr_pct": round(float(bench[-1]) - 1.0, 6),
            "n_positions": len(rows),
            "cheap": calls.count("cheap"),
            "rich": calls.count("rich"),
            "uncovered": len(rows) - len(calls),
            "top5_share": round(top_share(weight_rows, 5), 6),
            "covered_value_share": round(cov["covered_value_share"], 6),
            "movers": movers[:3],
        },
        "positions": rows,
        "performance": {
            "dates": [d.isoformat() for d in series.dates],
            "growth": [round(float(g), 6) for g in growth],
            "benchmark_growth": [round(float(g), 6) for g in bench],
            "benchmark": ledger.benchmark,
            "contributions": [
                {
                    "symbol": c.symbol,
                    "unrealized": round(c.unrealized, 2),
                    "realized": round(c.realized, 2),
                    "dividends": round(c.dividends, 2),
                    "total": round(c.total, 2),
                }
                for c in contributions(ledger, quotes)
            ],
            "lots": lots,
            "realized_total": round(sum(p.realized for p in held), 2),
            "dividends_total": round(sum(p.dividends for p in held), 2),
        },
        "hygiene": {
            "weights": [
                {"symbol": w.symbol, "kind": w.kind, "value": round(w.value, 2), "weight": round(w.weight, 6)}
                for w in weight_rows
            ],
            "top5_share": round(top_share(weight_rows, 5), 6),
            "exposure": {k: round(v, 6) for k, v in sorted(exposure(weight_rows).items())},
            "drift": [
                {**g, "weight": round(g["weight"], 6), "gap": round(g["gap"], 6)}
                for g in drift(weight_rows, ledger.targets)
            ],
            "coverage": {k: round(v, 6) for k, v in cov.items()},
        },
        "engine": {
            symbol: {
                "symbol": read.symbol,
                "covered": read.covered,
                "sub_vertical": read.sub_vertical,
                "fade": read.fade,
                "warranted": read.warranted,
                "dcf": read.dcf,
                "refusals": read.refusals,
            }
            for symbol, read in sorted(reads.items())
        },
        "ideas": _ideas_pane(fixtures, {p.symbol for p in held}),
        "activity": [
            {
                "date": t.date.isoformat(),
                "type": t.type,
                "symbol": t.symbol,
                "shares": t.shares,
                "price": t.price,
                "amount": t.amount,
                "ratio": t.ratio,
            }
            for t in reversed(ledger.transactions)
        ],
    }
    validate_snapshot(snapshot)
    return snapshot


def _uncovered(symbol: str, kind: str):
    from .engine_read import EngineRead

    what = {"etf": "an index fund", "crypto": "a crypto asset"}.get(kind, "outside the universe")
    read = EngineRead(symbol=symbol, covered=False, sub_vertical=None)
    read.refusals = [
        {"what": "Engine read", "why": f"{symbol} is {what}: it is priced and weighed here, never valued from filings"}
    ]
    return read


def _ideas_pane(fixtures: Path, held: set[str]) -> dict:
    from .engine_read import warranted_model

    pane = ideas(warranted_model(Path(fixtures)), held)
    return {
        "as_of": pane["as_of"],
        "cheap": [asdict(i) for i in pane["cheap"]],
        "rich": [asdict(i) for i in pane["rich"]],
        "verdict": pane["verdict"],
    }


def validate_snapshot(snap: dict) -> None:
    """Refuse a snapshot the page could not draw, naming what is wrong."""
    if not isinstance(snap, dict):
        raise ValueError("the snapshot is not a mapping")
    missing_panes = [k for k in ("meta", "overview", "positions", "performance", "hygiene", "engine", "ideas", "activity") if k not in snap]
    if missing_panes:
        raise ValueError(f"the snapshot is missing panes: {', '.join(missing_panes)}")
    for pane, keys in _REQUIRED.items():
        for key in keys:
            if key not in snap[pane]:
                raise ValueError(f"{pane} is missing {key}")
    if snap["meta"]["schema"] != SCHEMA:
        raise ValueError(f"snapshot schema {snap['meta']['schema']!r}, expected {SCHEMA}")
    for i, row in enumerate(snap["positions"]):
        for key in _POSITION_KEYS:
            if key not in row:
                raise ValueError(f"positions[{i}] is missing {key}")
    perf = snap["performance"]
    if not (len(perf["dates"]) == len(perf["growth"]) == len(perf["benchmark_growth"])):
        raise ValueError("performance series disagree on length")
    for symbol, read in snap["engine"].items():
        for key in ("covered", "sub_vertical", "fade", "warranted", "dcf", "refusals"):
            if key not in read:
                raise ValueError(f"engine[{symbol}] is missing {key}")


def write_snapshot(snap: dict, path: Path) -> None:
    validate_snapshot(snap)
    text = json.dumps(snap, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")
