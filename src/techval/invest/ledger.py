"""The portfolio ledger: a YAML transaction list, held to what can be true.

A positions list cannot support performance against a benchmark or cost basis
by lot, so the file is a ledger: buys, sells, deposits, withdrawals, dividends
and splits, each dated. Validation refuses at load, naming the transaction, in
preference to loading something almost right and valuing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from ..errors import ConfigError

TYPES = ("buy", "sell", "deposit", "withdraw", "dividend", "split")
KINDS = ("stock", "etf", "crypto")

# What each type must carry. Everything else on the row must be absent or null.
_NEEDS = {
    "buy": ("symbol", "shares", "price"),
    "sell": ("symbol", "shares", "price"),
    "deposit": ("amount",),
    "withdraw": ("amount",),
    "dividend": ("symbol", "amount"),
    "split": ("symbol", "ratio"),
}

# Fields that must be strictly positive when present; price alone may be zero,
# because a spin-off or an award can land shares at no cost.
_POSITIVE = ("shares", "amount", "ratio")


@dataclass(frozen=True)
class Transaction:
    date: date
    type: str
    symbol: str | None = None
    shares: float | None = None
    price: float | None = None
    amount: float | None = None
    ratio: float | None = None
    kind: str | None = None
    note: str | None = None


class Ledger:
    """The parsed file: transactions sorted by date, file order preserved inside a day."""

    def __init__(
        self,
        *,
        name: str,
        benchmark: str,
        transactions: list[Transaction],
        targets: dict[str, float],
        kinds: dict[str, str],
    ) -> None:
        self.name = name
        self.benchmark = benchmark
        self.transactions = transactions
        self.targets = targets
        self._kinds = kinds

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(
                f"no portfolio file at {path}. techval invest init writes a starter."
            )
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, ValueError) as err:
            # An impossible calendar date raises inside the YAML constructor.
            raise ConfigError(f"{path} did not parse: a transaction date or value is impossible ({err})") from err
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} is not a mapping")
        rows = raw.get("transactions") or []
        if not isinstance(rows, list) or not rows:
            raise ConfigError(f"{path} carries no transactions")
        transactions: list[Transaction] = []
        kinds: dict[str, str] = {}
        for i, row in enumerate(rows, start=1):
            transactions.append(_parse_row(row, i, kinds))
        transactions.sort(key=lambda t: t.date)
        targets = raw.get("targets") or {}
        if not isinstance(targets, dict):
            raise ConfigError("targets must map symbols to weights of value")
        clean_targets: dict[str, float] = {}
        for key, value in targets.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(f"target for {key} must be a non-negative weight, not {value!r}")
            clean_targets[str(key)] = float(value)
        return cls(
            name=str(raw.get("name") or "Portfolio"),
            benchmark=str(raw.get("benchmark") or "SPY").upper(),
            transactions=transactions,
            targets=clean_targets,
            kinds=kinds,
        )

    def kind(self, symbol: str) -> str:
        return self._kinds.get(symbol.upper(), "stock")

    def symbols(self) -> list[str]:
        """Non-cash symbols, in order of first appearance."""
        seen: list[str] = []
        for t in self.transactions:
            if t.symbol and t.symbol not in seen:
                seen.append(t.symbol)
        return seen

    @property
    def first_date(self) -> date:
        return self.transactions[0].date

    def positions(self, as_of: date | None = None) -> dict[str, "Position"]:
        return _walk(self, as_of)[0]

    def cash(self, as_of: date | None = None) -> float:
        return _walk(self, as_of)[1]

    def flows(self) -> dict[date, float]:
        """External money in and out, the flows a time-weighted return strips."""
        out: dict[date, float] = {}
        for t in self.transactions:
            if t.type == "deposit":
                out[t.date] = out.get(t.date, 0.0) + t.amount
            elif t.type == "withdraw":
                out[t.date] = out.get(t.date, 0.0) - t.amount
        return out


def _parse_row(row: object, index: int, kinds: dict[str, str]) -> Transaction:
    where = f"transaction {index}"
    if not isinstance(row, dict):
        raise ConfigError(f"{where} is not a mapping")
    kind_of = row.get("type")
    if kind_of not in TYPES:
        raise ConfigError(f"{where}: type {kind_of!r} is not one of {', '.join(TYPES)}")
    when = row.get("date")
    if isinstance(when, str):
        try:
            when = date.fromisoformat(when)
        except ValueError:
            when = None
    if not isinstance(when, date):
        raise ConfigError(f"{where}: date {row.get('date')!r} is not an ISO date")
    for field in _NEEDS[kind_of]:
        if row.get(field) is None:
            raise ConfigError(f"{where}: a {kind_of} needs {field}")
    for field in _POSITIVE:
        value = row.get(field)
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ConfigError(f"{where}: {field} must be positive, not {value!r}")
    price = row.get("price")
    if price is not None and (not isinstance(price, (int, float)) or price < 0):
        raise ConfigError(f"{where}: price must be zero or more, not {price!r}")
    stated = row.get("kind")
    if stated is not None and stated not in KINDS:
        raise ConfigError(f"{where}: kind {stated!r} is not one of {', '.join(KINDS)}")
    symbol = row.get("symbol")
    if symbol is not None:
        symbol = str(symbol).upper()
        if stated is not None:
            known = kinds.get(symbol)
            if known is not None and known != stated:
                raise ConfigError(
                    f"{where}: {symbol} was {known} earlier in the file and cannot become {stated}"
                )
            kinds[symbol] = stated
    return Transaction(
        date=when,
        type=kind_of,
        symbol=symbol,
        shares=_number(row.get("shares")),
        price=_number(price),
        amount=_number(row.get("amount")),
        ratio=_number(row.get("ratio")),
        kind=stated,
        note=None if row.get("note") is None else str(row.get("note")),
    )


def _number(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


@dataclass
class Lot:
    """One open purchase, split-adjusted as splits arrive."""

    symbol: str
    opened: date
    shares: float
    cost_per_share: float


@dataclass
class Position:
    """One symbol's open lots and the gains its closed ones realized."""

    symbol: str
    kind: str
    shares: float = 0.0
    lots: list[Lot] = field(default_factory=list)
    realized: float = 0.0
    dividends: float = 0.0

    @property
    def cost(self) -> float:
        return sum(lot.shares * lot.cost_per_share for lot in self.lots)


def _walk(ledger: "Ledger", as_of: date | None):
    """Replay the ledger to a date, refusing anything the record cannot support.

    The refusals live here rather than in ``load`` because they are stateful:
    whether a sell is an oversell depends on every row before it.
    """
    positions: dict[str, Position] = {}
    cash = 0.0
    for t in ledger.transactions:
        if as_of is not None and t.date > as_of:
            break
        if t.type in ("buy", "sell", "dividend", "split"):
            position = positions.setdefault(
                t.symbol, Position(symbol=t.symbol, kind=ledger.kind(t.symbol))
            )
        if t.type == "deposit":
            cash += t.amount
        elif t.type == "withdraw":
            cash -= t.amount
        elif t.type == "buy":
            position.lots.append(Lot(t.symbol, t.date, t.shares, t.price))
            position.shares += t.shares
            cash -= t.shares * t.price
        elif t.type == "sell":
            if t.shares > position.shares + 1e-9:
                raise ConfigError(
                    f"{t.date}: selling {t.shares:g} {t.symbol} with only "
                    f"{position.shares:g} held"
                )
            left = t.shares
            while left > 1e-12:
                lot = position.lots[0]
                taken = min(lot.shares, left)
                position.realized += taken * (t.price - lot.cost_per_share)
                lot.shares -= taken
                left -= taken
                if lot.shares <= 1e-12:
                    position.lots.pop(0)
            position.shares -= t.shares
            cash += t.shares * t.price
        elif t.type == "dividend":
            if not position.lots and position.shares <= 0:
                raise ConfigError(f"{t.date}: a dividend on {t.symbol}, which the ledger never bought")
            position.dividends += t.amount
            cash += t.amount
        elif t.type == "split":
            for lot in position.lots:
                lot.shares *= t.ratio
                lot.cost_per_share /= t.ratio
            position.shares *= t.ratio
        if cash < -1e-9:
            raise ConfigError(
                f"{t.date}: the {t.type} takes cash to {cash:,.2f}. The ledger "
                "is missing a deposit."
            )
    return positions, cash
