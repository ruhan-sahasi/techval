"""What a reader returns for one task, whichever reader it is.

``stated``, ``not_stated`` and ``ambiguous`` are answers. ``refused`` is a reader,
or the checker behind it, declining to give one, and it always says why.
``missing`` is a Claude reading with no recording behind it: the reader did not
run, which is a different fact from a reader that ran and declined.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

READERS = ("claude", "regex_retrieved", "regex_native")
UNITS = ("count", "usd", "percent", "usd_per_share", "ratio", "date", "text")
STATUSES = ("stated", "not_stated", "ambiguous", "refused", "missing")


@dataclass(frozen=True)
class Reading:
    task_id: str
    reader: str
    status: str
    value: float | None = None
    text_value: str | None = None
    unit: str | None = None
    period_end: str | None = None
    passage_id: str | None = None
    quote: str | None = None
    hedged: bool = False
    reason: str = ""
    served_by: str | None = None

    def __post_init__(self) -> None:
        if self.reader not in READERS:
            raise ValueError(f"unknown reader {self.reader!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown reading status {self.status!r}")

    @property
    def answered(self) -> bool:
        return self.status in ("stated", "not_stated", "ambiguous")

    def refuse(self, why: str) -> Reading:
        return replace(self, status="refused", reason=why)
