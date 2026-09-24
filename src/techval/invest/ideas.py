"""The screen's cheap and rich, with the signal's measured verdict attached.

This is the closest thing the page has to a recommendations pane, so it leads
with the fact that matters most and flatters least: scored against forward
returns, cheapness on this residual LOST to a random score. The residuals are
still worth reading, as a description of where a name sits against its
fundamentals today; the verdict is what stops them being read as a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

# The recorded finding, verbatim from the signal harness on the recorded
# EV/Revenue panel. techval signal reprints it on every run.
SIGNAL_VERDICT = (
    "Scored against 12-month forward returns by techval signal, cheapness on "
    "this residual's own panel had a mean IC of -0.0984, the wrong sign, and a "
    "Newey-West t of -1.62: not significant. Read these rows as where a name "
    "sits against its fundamentals today, not as a forecast of where it goes."
)


@dataclass(frozen=True)
class Idea:
    ticker: str
    sub_vertical: str
    traded: float
    warranted: float
    residual_log: float
    z: float


def ideas(model, held: set[str], n: int = 8) -> dict:
    """The n cheapest and n richest names on the panel's latest date, ex holdings.

    Sorted by the within-date z of the residual, the column the screen itself
    sorts on, because turns are not comparable between a 2x telecom and a 20x
    security name. Ties break by ticker so two runs agree.
    """
    when = model.latest
    held = {h.upper() for h in held}
    reads = [
        r
        for (ticker, day), r in model.reads.items()
        if day == when and ticker not in held and r.out_of_sample
    ]

    def idea(r) -> Idea:
        return Idea(
            ticker=r.ticker,
            sub_vertical=r.sub_vertical,
            traded=round(r.actual_multiple, 6),
            warranted=round(r.warranted_multiple, 6),
            residual_log=round(r.residual_log, 6),
            z=round(r.z, 6),
        )

    cheap = sorted((r for r in reads if r.residual_log < 0), key=lambda r: (r.z, r.ticker))
    rich = sorted((r for r in reads if r.residual_log > 0), key=lambda r: (-r.z, r.ticker))
    return {
        "as_of": when.isoformat(),
        "cheap": [idea(r) for r in cheap[:n]],
        "rich": [idea(r) for r in rich[:n]],
        "verdict": SIGNAL_VERDICT,
    }
