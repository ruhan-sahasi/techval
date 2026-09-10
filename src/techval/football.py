"""Football field chart.

One horizontal bar per valuation methodology, a vertical rule at the current
share price, and nothing else. The chart's only job is to show where the market
price sits against the range each method produces, and whether the methods agree.

Deliberately absent: gridlines behind the bars, a legend restating the axis
labels, gradient fills, a title repeating what the caption says. Every one of
those competes with the four numbers a reader actually needs, which are the low
and high of each bar and where the price line crosses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

_INK = "#1a1a1a"
_MUTED = "#8a8a8a"
_BAR = "#4a6fa5"
_BAR_EDGE = "#2c4a73"
_PRICE = "#c1440e"


@dataclass
class FootballRow:
    """One methodology's range."""

    label: str
    low: float
    high: float
    note: str = ""

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


def football_field(
    rows: list[FootballRow],
    current_price: float,
    ticker: str,
    out_path: str | Path,
    *,
    subtitle: str = "",
) -> Path:
    """Render the chart and return where it was written.

    Rows are drawn top to bottom in the order given. The convention is to lead
    with what the market says (the trading range), then the market-relative
    methods, then the intrinsic ones, so the eye travels from observed to
    derived.
    """
    rows = [r for r in rows if r.low is not None and r.high is not None]
    # A range built from a single observation is not a range. It draws as a
    # hairline and reads as a precise answer, which is the opposite of what it
    # is, so it is dropped rather than shown.
    rows = [r for r in rows if r.high - r.low > 0.005 * max(abs(r.high), 1.0)]
    if not rows:
        raise ValueError("football field needs at least one methodology range")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    height = 1.6 + 0.62 * len(rows)
    fig, ax = plt.subplots(figsize=(9.5, height))

    y = list(range(len(rows)))[::-1]

    for yi, row in zip(y, rows):
        width = max(row.high - row.low, 1e-9)
        ax.barh(
            yi,
            width,
            left=row.low,
            height=0.44,
            color=_BAR,
            edgecolor=_BAR_EDGE,
            linewidth=0.8,
            zorder=3,
        )
        # Endpoint labels sit outside the bar so they never fight the fill.
        pad = (max(r.high for r in rows) - min(r.low for r in rows)) * 0.012
        ax.text(
            row.low - pad,
            yi,
            f"${row.low:,.0f}",
            va="center",
            ha="right",
            fontsize=9,
            color=_INK,
            zorder=4,
        )
        ax.text(
            row.high + pad,
            yi,
            f"${row.high:,.0f}",
            va="center",
            ha="left",
            fontsize=9,
            color=_INK,
            zorder=4,
        )

    ax.axvline(
        current_price, color=_PRICE, linewidth=1.6, linestyle="-", zorder=5
    )
    ax.annotate(
        f"Current  ${current_price:,.2f}",
        xy=(current_price, len(rows) - 0.35),
        xytext=(4, 0),
        textcoords="offset points",
        color=_PRICE,
        fontsize=9.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        zorder=6,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([r.label for r in rows], fontsize=10, color=_INK)
    ax.set_ylim(-0.7, len(rows) - 0.15)

    lo = min(min(r.low for r in rows), current_price)
    hi = max(max(r.high for r in rows), current_price)
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - span * 0.14, hi + span * 0.14)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.tick_params(axis="x", labelsize=9, colors=_MUTED)
    ax.tick_params(axis="y", length=0)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_MUTED)
    ax.spines["bottom"].set_linewidth(0.8)

    # The subtitle sits just above the axes and the title is padded clear of it,
    # so the two never collide however tall the figure grows with more rows.
    ax.set_title(
        f"{ticker}  implied value per share",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=_INK,
        pad=26 if subtitle else 10,
    )
    if subtitle:
        ax.text(
            0.0,
            1.008,
            subtitle,
            transform=ax.transAxes,
            fontsize=8.8,
            color=_MUTED,
            va="bottom",
            ha="left",
        )

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
