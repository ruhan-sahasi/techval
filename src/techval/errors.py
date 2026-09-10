"""Typed errors.

The engine never interpolates, guesses, or silently zero-fills a figure it needed
but could not find. When a required input is missing it raises, and the message
names the concept, the tags that were tried, and the period that was searched, so
the user can go look in the filing themselves.
"""

from __future__ import annotations


class TechvalError(Exception):
    """Base class for every error this package raises deliberately."""


class MissingDataError(TechvalError):
    """A required figure could not be sourced from any filing.

    Carries the structured detail a user needs to audit the gap by hand.
    """

    def __init__(
        self,
        concept: str,
        *,
        ticker: str | None = None,
        tags_tried: list[str] | None = None,
        period: str | None = None,
        hint: str | None = None,
    ) -> None:
        self.concept = concept
        self.ticker = ticker
        self.tags_tried = tags_tried or []
        self.period = period
        self.hint = hint

        parts = [f"could not source {concept!r}"]
        if ticker:
            parts.append(f"for {ticker}")
        if period:
            parts.append(f"over period {period}")
        msg = " ".join(parts)
        if self.tags_tried:
            msg += "\n  us-gaap tags tried (in order): " + ", ".join(self.tags_tried)
        if hint:
            msg += f"\n  hint: {hint}"
        super().__init__(msg)


class StaleDataError(MissingDataError):
    """A tag exists but only with facts far older than the balance-sheet date.

    Filers retire tags. ``ShortTermInvestments`` may sit in a company's fact set
    with its newest value years stale because they now report
    ``AvailableForSaleSecuritiesDebtSecuritiesCurrent``. Taking the stale value
    would be silently, badly wrong, so it is an error rather than a fallback.
    """


class DataSourceError(TechvalError):
    """A remote source failed, refused, or returned something unusable."""


class ConfigError(TechvalError):
    """The assumptions file is missing something, or is internally inconsistent."""


class NotMeaningfulError(TechvalError):
    """A multiple is arithmetically computable but economically meaningless.

    Raised, and caught by callers who then render ``NM``, when a denominator is
    negative or so close to zero that the ratio conveys nothing.
    """
