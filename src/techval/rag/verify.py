"""Hold an answer to its quote: the quote must be in the passage, and the value in the quote.

A language model can state a number that no passage contains, or quote a
sentence that says something else. Neither is caught by asking the model
again, so every stated answer is checked here, deterministically, against the
passages it was given. The same rules hold the owner's answer key to the filing,
so a typo in the key fails loudly instead of marking a correct reader wrong.

Only two things are folded before a quote is compared: runs of whitespace, and
typographic quote marks. Filings write “Merger Consideration” with curly marks,
and a reader that copies them straight has still quoted the filing. Digits,
words and punctuation otherwise stay exactly as they are.

A dollar figure below one million with no scale word is refused, as
``tmt.kpis`` already refuses it. "ARR of $100,000" is a threshold for counting
customers far more often than it is a company's ARR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from typing import Sequence

from .passages import Passage
from .readings import Reading

_MARKS = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})
_SCALE = {"thousand": 1e3, "million": 1e6, "mm": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}
_NUMBER = re.compile(
    r"(?P<dollar>\$\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<pct>%|percent\b))?"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|mm|bn)\b)?",
    re.IGNORECASE,
)
_HEDGES = (
    "approximately", "about", "more than", "over", "nearly", "roughly",
    "almost", "at least", "in excess of", "greater than",
)
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTHS = {name: i for i, name in enumerate(_MONTH_NAMES, start=1)}
_DATE_WORDS = re.compile(r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(\d{1,2}),\s*(\d{4})", re.IGNORECASE)
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_CATEGORY_EVIDENCE = {
    "cash": (("cash",),),
    "stock": (("share", "stock"),),
    "mixed": (("cash",), ("share", "stock")),
}
_NUMERIC_UNITS = ("count", "usd", "usd_per_share", "ratio", "percent")

MONEY_FLOOR = 1_000_000.0


def fold(text: str) -> str:
    return " ".join(text.translate(_MARKS).split())


def fold_words(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.translate(_MARKS).lower()).split())


@dataclass(frozen=True)
class Number:
    value: float
    resolution: float
    percent: bool
    dollars: bool
    scaled: bool
    start: int


def numbers(text: str) -> list[Number]:
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group("num")
        scale = _SCALE.get((m.group("scale") or "").lower(), 1.0)
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        out.append(
            Number(
                value=float(raw.replace(",", "")) * scale,
                resolution=(10.0**-decimals) * scale,
                percent=bool(m.group("pct")),
                dollars=bool(m.group("dollar")),
                scaled=scale != 1.0,
                start=m.start(),
            )
        )
    return out


def dates(text: str) -> list[date]:
    found = []
    for month, day, year in _DATE_WORDS.findall(text):
        try:
            found.append(date(int(year), _MONTHS[month.lower()], int(day)))
        except ValueError:
            continue
    for year, month, day in _DATE_ISO.findall(text):
        try:
            found.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return found


def locate(quote: str, text: str, base: int = 0) -> list[tuple[int, int]]:
    """Every place ``quote`` occurs in ``text``, as absolute ``(start, end)`` offsets."""
    parts = []
    for ch in fold(quote):
        if ch == " ":
            parts.append(r"\s+")
        elif ch == '"':
            parts.append("[\"“”„]")
        elif ch == "'":
            parts.append("['‘’]")
        else:
            parts.append(re.escape(ch))
    if not parts:
        return []
    pattern = re.compile("".join(parts))
    return [(base + m.start(), base + m.end()) for m in pattern.finditer(text)]


def _hedged_at(text: str, start: int) -> bool:
    before = text[max(0, start - 30) : start].lower()
    return any(h in before for h in _HEDGES)


@dataclass(frozen=True)
class Check:
    ok: bool
    why: str = ""
    resolution: float | None = None
    hedged: bool = False


def check_value(unit: str | None, value: float | None, text_value: str | None, quote: str) -> Check:
    """Whether ``quote`` states this value, in this unit."""
    if unit in _NUMERIC_UNITS:
        if value is None:
            return Check(False, "a stated number carries no value")
        for n in numbers(quote):
            if (unit == "percent") != n.percent:
                continue
            if n.dollars and unit not in ("usd", "usd_per_share"):
                continue
            if abs(n.value - value) > max(n.resolution / 2, 1e-9 * max(1.0, abs(value))):
                continue
            if unit == "usd" and value < MONEY_FLOOR and not n.scaled:
                return Check(
                    False,
                    f"the quote gives ${n.value:,.0f} with no scale word, and below $1,000,000 "
                    "that is as likely a threshold as a total",
                )
            return Check(True, resolution=n.resolution, hedged=_hedged_at(quote, n.start))
        return Check(False, f"the quote does not state {value:g}")
    if unit == "date":
        try:
            want = date.fromisoformat(text_value or "")
        except ValueError:
            return Check(False, f"{text_value!r} is not a date written YYYY-MM-DD")
        if want in dates(quote):
            return Check(True)
        return Check(False, f"the quote does not state {want.isoformat()}")
    if unit == "text":
        if not text_value:
            return Check(False, "a stated text answer carries no text")
        words = fold_words(quote)
        groups = _CATEGORY_EVIDENCE.get(text_value.strip().lower())
        if groups is not None:
            if all(any(w in words for w in group) for group in groups):
                return Check(True)
            return Check(False, f"the quote does not show {text_value.strip().lower()} consideration")
        if fold_words(text_value) in words:
            return Check(True)
        return Check(False, f"the quote does not contain {text_value!r}")
    return Check(False, f"no rule checks the unit {unit!r}")


def verify(reading: Reading, passages: Sequence[Passage], unit: str) -> Reading:
    """A stated reading kept only if its passage, its quote and its value all hold."""
    if reading.status != "stated":
        return reading
    if reading.unit != unit:
        return reading.refuse(
            f"the answer is in {reading.unit!r}, and this fact is read in {unit!r}"
        )
    passage = next((p for p in passages if p.id == reading.passage_id), None)
    if passage is None:
        return reading.refuse(
            f"the answer cites passage {reading.passage_id!r}, which it was not given"
        )
    if not reading.quote or fold(reading.quote) not in fold(passage.text):
        return reading.refuse("the quote is not in the passage it cites")
    check = check_value(unit, reading.value, reading.text_value, reading.quote)
    if not check.ok:
        return reading.refuse(check.why)
    return replace(reading, hedged=reading.hedged or check.hedged)
