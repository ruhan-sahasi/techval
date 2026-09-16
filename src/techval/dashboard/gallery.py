"""A gallery page for the dashboard front end, drawn from synthetic data.

``render_gallery()`` returns one HTML page that exercises every chart in the
kit, every verdict chip, a figure carrying a refusal note, a refused section
and a section that has not been collected. It also exercises what the kit does
beyond drawing a series: figures sorted by an order number, whiskers on dots
and bars, row groups, direct labels a row chooses, coincident points drawn
apart, reference labels placed clear of the marks, a table figure and fuller
table views, a legend turned off, a waterfall with no subtracting step, a heat
grid of long identifiers on fixed breaks, a line on dates and a log scale with
shaded ranges and a labelled point, a caution under a figure, and tiles in a
card with a second part. The overview carries the three figures its collector
returns, in the collector's shapes and read from the gallery's own headlines,
so the page draws them with the real scoreboard renderer rather than the kit's
default drawing of an unknown figure. It is a development tool for looking at the kit in
both themes, not a results page: the page says so in a notice under its title,
records no fixtures, and every provenance entry names this module rather than a
model entry point.

The page is assembled exactly as the real dashboard is (the embedding contract
in ``render.py``): the stylesheets inlined in one ``<style>``, the snapshot in a
JSON data block with ``</`` and ``<!--`` escaped, then ``kit.js``, ``app.js``
and each section script in page order. So what the gallery shows is what the
real page will show, down to the shell app.js draws around the figures.

This module depends on nothing but the standard library and reads the assets
relative to its own file, so it can be loaded by path before the rest of the
dashboard package exists. Run it as a script to write the page to a file::

    python src/techval/dashboard/gallery.py out/gallery.html

then open ``out/gallery.html?theme=dark`` or ``?theme=light`` to force a theme.
"""

from __future__ import annotations

import html
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Page order. The package's sections module holds the same tuple; the gallery
# keeps its own copy so it can load without the package.
PAGE_ORDER = (
    "overview",
    "signal",
    "encoder",
    "warranted",
    "fade",
    "propensity",
    "engine",
    "tmt",
    "datalayer",
    "sample",
)

STYLESHEETS = ("tokens.css", "layout.css")
SCRIPTS = ("kit.js", "app.js")

FONTS_HREF = (
    "https://fonts.googleapis.com/css2"
    "?family=IBM+Plex+Mono:wght@400;500"
    "&family=Newsreader:ital,wght@0,400;0,500;1,400"
    "&family=Public+Sans:wght@400;500;650"
    "&display=swap"
)

GALLERY_ENTRY_POINT = "techval.dashboard.gallery.render_gallery"

NOTICE = (
    "Synthetic data. This page exercises the chart kit and the page shell; "
    "no figure on it is a techval result."
)


# Assembly ------------------------------------------------------------------------


def _read_asset(name: str) -> str:
    path = ASSETS_DIR.joinpath(*name.split("/"))
    return path.read_text(encoding="utf-8").rstrip("\n") + "\n"


def _jsonable(value: Any) -> Any:
    """Round floats to 6 places and turn non-finite floats into null, as the snapshot does."""
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _embed_json(data: Any) -> str:
    text = json.dumps(
        _jsonable(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


def asset_names() -> list[str]:
    """Every asset the page inlines, in the order it inlines them."""
    return [*STYLESHEETS, *SCRIPTS, *(f"sections/{sid}.js" for sid in PAGE_ORDER)]


def render_gallery() -> str:
    """The gallery page, as a string. The same assets always give the same bytes."""
    snapshot = gallery_snapshot()
    styles = "".join(_read_asset(name) for name in STYLESHEETS)
    scripts = [(name, _read_asset(name)) for name in (*SCRIPTS, *(f"sections/{sid}.js" for sid in PAGE_ORDER))]
    out = [
        "<!doctype html>\n",
        '<html lang="en">\n',
        "<head>\n",
        '<meta charset="utf-8">\n',
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
        f"<title>{html.escape(snapshot['title'])}</title>\n",
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n',
        f'<link rel="stylesheet" href="{html.escape(FONTS_HREF)}">\n',
        "<style>\n",
        styles,
        "</style>\n",
        "</head>\n",
        "<body>\n",
        "<noscript>This page draws its figures with JavaScript.</noscript>\n",
        f'<p class="tv-banner" id="tv-notice" hidden>{html.escape(NOTICE)}</p>\n',
        '<script type="application/json" id="tv-snapshot">',
        _embed_json(snapshot),
        "</script>\n",
    ]
    for name, text in scripts:
        out += [f'<script data-asset="{html.escape(name)}">\n', text, "</script>\n"]
    out += ["</body>\n", "</html>\n"]
    return "".join(out)


def write_gallery(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_gallery(), encoding="utf-8", newline="\n")
    return target


# Synthetic data ------------------------------------------------------------------
#
# Plausible shapes from this project's world, generated from fixed formulas so the
# page is identical on every run. None of it is a measured result.


def _wave(i: int, period: float, amp: float, phase: float = 0.0) -> float:
    return amp * math.sin(i * 2 * math.pi / period + phase)


def _figure(kind: str, title: str, subtitle: str, data: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A figure; extra carries the figure-level fields: order, wide, card, part, note."""
    return {"kind": kind, "title": title, "subtitle": subtitle, "data": data, **extra}


def _provenance(*figure_ids: str) -> list[dict[str, Any]]:
    return [
        {"figure": fid, "entry_point": GALLERY_ENTRY_POINT, "inputs": [], "seconds": 0.0}
        for fid in figure_ids
    ]


# A second entry point for the figures that show a card naming more than one.
SNAPSHOT_ENTRY_POINT = "techval.dashboard.gallery.gallery_snapshot"


def _section(
    sid: str,
    title: str,
    takeaway: str,
    *,
    status: str = "ok",
    headline: dict[str, Any] | None = None,
    figures: dict[str, Any] | None = None,
    refusals: list[dict[str, str]] | None = None,
    also_from: tuple[str, ...] = (),
) -> dict[str, Any]:
    figures = figures or {}
    provenance = _provenance(*figures)
    for fid in also_from or ():
        provenance.append({"figure": fid, "entry_point": SNAPSHOT_ENTRY_POINT, "inputs": [], "seconds": 0.0})
    return {
        "id": sid,
        "title": title,
        "takeaway": takeaway,
        "status": status,
        "refusals": refusals or [],
        "headline": headline,
        "figures": figures,
        "provenance": provenance,
    }


def _headline(
    metric: str,
    score: float,
    baseline_name: str,
    baseline_score: float,
    n: int,
    higher_is_better: bool,
    verdict_status: str,
    verdict_text: str,
) -> dict[str, Any]:
    lift = score - baseline_score if higher_is_better else baseline_score - score
    return {
        "metric": metric,
        "score": score,
        "baseline_name": baseline_name,
        "baseline_score": baseline_score,
        "lift": lift,
        "n": n,
        "higher_is_better": higher_is_better,
        "verdict_status": verdict_status,
        "verdict_text": verdict_text,
    }


TICKERS = ("MSFT", "GOOGL", "META", "AMZN", "AAPL", "NFLX", "ORCL", "CRM")


# The model sections, in page order, as the package's sections module names them.
MODEL_SECTIONS = ("signal", "encoder", "warranted", "fade", "propensity")


def _dp(v: float) -> int:
    """Decimal places that follow the magnitude, as the page's headline strip formats."""
    a = abs(v)
    return 0 if a >= 100 else 1 if a >= 10 else 2 if a >= 1 else 4


def _scoreboard_tile(section: dict[str, Any]) -> dict[str, Any]:
    """A model's tile in the shape the overview collector gives it, read from the section."""
    sid = section["id"]
    tile: dict[str, Any] = {"section": sid, "href": f"#{sid}", "label": section["title"]}
    h = section.get("headline")
    if section["status"] == "ok" and h:
        base = f"{h['baseline_score']:,.{_dp(h['baseline_score'])}f}"
        tile.update(
            {
                "state": "scored",
                "metric": h["metric"],
                "score": h["score"],
                "baseline_name": h["baseline_name"],
                "baseline_score": h["baseline_score"],
                "lift": h["lift"],
                "n": h["n"],
                "higher_is_better": h["higher_is_better"],
                "verdict_status": h["verdict_status"],
                "value": h["score"],
                "format": f"num:{_dp(h['score'])}",
                "sub": f"{h['metric']} against {base} for {h['baseline_name']}, n = {h['n']:,}",
                "status": h["verdict_status"],
                "delta": {"value": h["lift"], "format": f"signed:{_dp(h['lift'])}", "label": "lift"},
            }
        )
    elif section["status"] == "refused":
        first = (section.get("refusals") or [{"what": section["title"], "why": "No reason was recorded."}])[0]
        tile.update({"state": "refused", "value": "Refused", "status": "refused", "reason": first, "sub": first["why"]})
    elif section["status"] == "ok":
        tile.update({"state": "no_headline", "value": "No score", "sub": "Collected without a headline score against a baseline"})
    else:
        tile.update({"state": "not_built", "value": "Not collected", "sub": "This snapshot holds no results for this model"})
    return tile


def _overview(sections: list[dict[str, Any]]) -> dict[str, Any]:
    """The scoreboard's three figures, with the ids and shapes the overview collector returns."""
    by_id = {s["id"]: s for s in sections}
    tiles = [_scoreboard_tile(by_id[sid]) for sid in MODEL_SECTIONS]
    counts = {v: sum(t.get("verdict_status") == v for t in tiles) for v in ("beats", "inside_noise", "ties", "not_significant", "loses")}
    rows = [
        {
            "id": s["id"],
            "title": s["title"],
            "href": f"#{s['id']}",
            "status": s["status"],
            "verdict_status": s["headline"]["verdict_status"] if s.get("headline") else None,
            "figures": len(s["figures"]),
            "refusals": len(s["refusals"]),
            "provenance": len(s["provenance"]),
        }
        for s in sections
    ]
    figures = sum(r["figures"] for r in rows)
    refusals = sum(r["refusals"] for r in rows)
    refusing = [r for r in rows if r["refusals"]]
    return _section(
        "overview",
        "Scoreboard",
        "One model beats its baseline outside the noise; the others tie, sit inside the noise, lose, or are not significant.",
        figures={
            "scoreboard": _figure(
                "tiles",
                "One of five models beats its baseline outside the noise: the peer encoder",
                "Each model's headline score against the baseline its section names",
                {"tiles": tiles, "counts": counts, "returns_model": "signal"},
            ),
            "sections": _figure(
                "hbar",
                f"{figures} figures drawn and {refusals} refusals stated, across {len(refusing)} sections",
                "Every other section, in page order, with the figures it drew and the refusals it stated",
                {
                    "rows": [{"label": r["title"], "value": r["refusals"], "role": "baseline"} for r in rows],
                    "format": "int",
                    "valueLabel": "Refusals",
                    "labelHeader": "Section",
                    "table": rows,
                    "totals": {
                        "sections": len(rows),
                        "collected": sum(r["status"] == "ok" for r in rows),
                        "figures": figures,
                        "refusals": refusals,
                        "refusing_sections": len(refusing),
                    },
                },
            ),
            "collection": _figure(
                "tiles",
                "Every assumption is the engine default",
                "What every number on this page was computed from, and under what",
                {
                    "tiles": [
                        {"label": "Assumptions changed from the defaults", "value": 0, "format": "int", "sub": "No setting differs from the defaults"}
                    ],
                    "overrides": [],
                },
            ),
        },
    )


def _signal() -> dict[str, Any]:
    # Labels too wide for twelve to fit a half-width card, so the last fold's must be kept on purpose.
    folds = [
        {"label": f"Fold {i + 1}", "value": 0.011 + _wave(i, 5.3, 0.024, 0.6)} for i in range(12)
    ]
    mean_lift = round(sum(f["value"] for f in folds) / len(folds), 4)
    quarters = [f"{2020 + q // 4}Q{q % 4 + 1}" for q in range(16)]
    model, base = [], []
    m = b = 0.0
    for i, q in enumerate(quarters):
        m += 0.0042 + _wave(i, 6.0, 0.006)
        b += 0.0031 + _wave(i, 7.0, 0.005, 1.1)
        width = 0.004 + 0.0016 * i
        model.append({"x": q, "y": m, "lo": m - width, "hi": m + width})
        base.append({"x": q, "y": b})
    return _section(
        "signal",
        "Filing-text signal",
        "The signal's rank IC is positive but sits inside the fold-to-fold noise of plain momentum.",
        headline=_headline(
            "rank IC", 0.0412, "12-month momentum", 0.0300, 96, True, "inside_noise",
            "Rank IC of 0.0412 against 0.0300 for momentum, a lift smaller than one standard error across 12 folds.",
        ),
        figures={
            # The mean rule crosses the first folds' bars, so its label has to be placed clear of them.
            "ic_lift_by_fold": _figure(
                "column",
                "The lift changes sign across folds",
                "Rank IC, model minus momentum, per walk-forward fold",
                {
                    "rows": folds,
                    "diverging": True,
                    "format": "signed:3",
                    "valueLabel": "IC lift",
                    "labelHeader": "Fold",
                    "reference": [{"value": mean_lift, "label": f"Mean {mean_lift:+.3f}"}],
                },
            ),
            # A rule at the start of both lines, whose label must not sit on them.
            "spread": _figure(
                "line",
                "Cumulative long-short spread tracks momentum",
                "Top minus bottom quintile, cumulative return, 80% band on the model",
                {
                    "series": [
                        {"name": "Text signal", "role": "model", "values": model},
                        {"name": "Momentum", "role": "baseline", "values": base},
                    ],
                    "x": {"label": "Quarter"},
                    "format": "pct:1",
                    "zero": True,
                    "yTitle": "Cumulative return",
                    "reference": [{"value": round(base[1]["y"], 4), "label": "Second quarter"}],
                },
            ),
        },
    )


def _encoder() -> dict[str, Any]:
    sim = []
    for i in range(len(TICKERS)):
        row = []
        for j in range(len(TICKERS)):
            row.append(1.0 if i == j else round(0.42 + 0.4 * abs(math.cos((i + 1) * (j + 1) * 0.37)), 2))
        sim.append(row)
    for i in range(len(TICKERS)):
        for j in range(i):
            sim[i][j] = sim[j][i]
    # Telecom's two scores coincide, so its two dots must be drawn apart rather than one over the other.
    sectors = [("Software", 0.61, 0.24), ("Semiconductors", 0.57, 0.31), ("Internet", 0.55, 0.19), ("IT services", 0.48, 0.22), ("Media", 0.44, 0.27), ("Telecom", 0.39, 0.39)]
    cuts = [
        ("Walk-forward cut, 5 folds", "Without text", 0.064, 0.063),
        ("Walk-forward cut, 5 folds", "Without fundamentals", 0.039, 0.069),
        ("Headline cut, 5 folds", "Without text", 0.069, 0.068),
        ("Headline cut, 5 folds", "Without fundamentals", 0.027, 0.086),
    ]
    ablation_rows = [
        {
            "label": label,
            "group": group,
            "values": {"damage": damage},
            "lo": round(damage - sd, 4),
            "hi": round(damage + sd, 4),
            "text": ("clears by " if damage > sd else "inside by ") + f"{abs(damage - sd):.4f}",
            "tip": [{"label": "Fold standard deviation", "value": sd, "format": "num:3"}],
        }
        for group, label, damage, sd in cuts
    ]
    return _section(
        "encoder",
        "Peer encoder",
        "The encoder ranks true peers far above the popularity prior in every sector.",
        headline=_headline(
            "NDCG@10", 0.5407, "popularity prior", 0.2213, 162, True, "beats",
            # In lower case, as the model's verdict() names its metric: the strip shows NDCG@10.
            "ndcg@10 of 0.5407 against 0.2213 for the popularity prior, outside the fold noise on 162 query filers.",
        ),
        figures={
            "ndcg_by_method": _figure(
                "hbar",
                "The encoder leads every peer rule",
                "nDCG@10 on 162 query filers",
                {
                    "rows": [
                        {"label": "Peer encoder", "value": 0.5407, "role": "model", "lo": 0.5171, "hi": 0.5643},
                        {"label": "Same four-digit SIC", "value": 0.3310, "role": "alt", "lo": 0.3052, "hi": 0.3568},
                        {"label": "Popularity prior", "value": 0.2213, "role": "baseline", "lo": 0.2004, "hi": 0.2422},
                        {"label": "Random peers", "value": 0.0480, "role": "baseline", "lo": 0.0391, "hi": 0.0569},
                    ],
                    "intervalLabel": "One standard error",
                    "format": "num:3",
                    "valueLabel": "nDCG@10",
                    "labelHeader": "Method",
                    "roleLabels": {"model": "Peer encoder", "alt": "SIC rule", "baseline": "Baselines"},
                },
            ),
            "similarity": _figure(
                "heat",
                "Large-cap platforms cluster tightly",
                "Cosine similarity of encoder embeddings",
                {"rows": list(TICKERS), "cols": list(TICKERS), "values": sim, "scale": "sequential", "format": "num:2", "scaleLabel": "Cosine similarity", "rowHeader": "Ticker"},
            ),
            "by_sector": _figure(
                "dot",
                "The gap is widest in software",
                "nDCG@10 by sector, encoder against popularity prior",
                {
                    "rows": [{"label": s, "values": {"model": a, "prior": b}} for s, a, b in sectors],
                    "series": [
                        {"key": "model", "name": "Peer encoder", "role": "model"},
                        {"key": "prior", "name": "Popularity prior", "role": "baseline"},
                    ],
                    "format": "num:2",
                    "gapLabel": "Lift",
                    "labelHeader": "Sector",
                    "domain": [0, 0.8],
                    # Explicit label control: the prior's value, beside the prior's dot, on the first row only.
                    "labels": {"series": "prior", "rows": [0]},
                },
            ),
            "ablation": _figure(
                "dot",
                "Only the text tower clears its fold noise",
                "Damage from removing a tower, whisker one fold standard deviation either side, grouped by fold cut",
                {
                    "rows": ablation_rows,
                    "series": [{"key": "damage", "name": "Damage", "role": "model"}],
                    "format": "signed:3",
                    "valueLabel": "Damage",
                    "intervalLabel": "One fold standard deviation",
                    "labelHeader": "Tower removed",
                    "groupHeader": "Fold cut",
                    "reference": [{"value": 0, "label": "No damage"}],
                    "zero": True,
                    # A fuller table than the chart draws: the margin each whisker clears by.
                    "table": {
                        "columns": [
                            {"key": "group", "label": "Fold cut"},
                            {"key": "label", "label": "Tower removed"},
                            {"key": "damage", "label": "Damage", "align": "right", "format": "signed:3"},
                            {"key": "margin", "label": "Damage less deviation", "align": "right", "format": "signed:4"},
                        ],
                        "rows": [
                            {"group": g, "label": label, "damage": damage, "margin": round(damage - sd, 4)}
                            for g, label, damage, sd in cuts
                        ],
                    },
                },
                note="Seven of the panel's features are empty on every row, so the fundamentals tower is judged on a degraded input.",
            ),
        },
    )


def _warranted() -> dict[str, Any]:
    residuals = [("NVDA", 3.4), ("CRM", 1.6), ("ADBE", 0.9), ("ORCL", 0.4), ("MSFT", 0.1), ("CSCO", -0.3), ("IBM", -0.8), ("INTC", -1.2), ("T", -1.9), ("VZ", -2.4)]
    edges = [round(0.1 * i, 1) for i in range(13)]
    model_counts = [9, 21, 30, 27, 22, 17, 12, 9, 6, 4, 3, 2]
    base_counts = [8, 20, 32, 28, 21, 16, 13, 9, 6, 5, 2, 2]
    return _section(
        "warranted",
        "Warranted multiple",
        "The fitted EV/Revenue multiple ties a sector median, so the regression adds nothing a lookup would not.",
        headline=_headline(
            "log MAE", 0.4120, "sector median multiple", 0.4090, 162, False, "ties",
            # Several sentences, the first in lower case, as a model's own verdict() often is.
            "log MAE of 0.4120 against 0.4090 for the sector median, a lift of -0.0030 on 162 observations. "
            "Fold standard deviation 0.0110, so the difference is inside the fold-to-fold noise. "
            "Use the sector median.",
        ),
        figures={
            "residuals": _figure(
                "dot",
                "Semiconductors trade furthest above their warranted multiple",
                "Traded minus warranted EV/Revenue, turns, latest fiscal year",
                {
                    "rows": [{"label": t, "values": {"value": v}} for t, v in residuals],
                    "series": [{"key": "value", "name": "Traded minus warranted", "role": "model"}],
                    "format": "signed:1",
                    "reference": [{"value": 0, "label": "Fairly priced"}],
                    "labelHeader": "Ticker",
                    "zero": True,
                },
            ),
            "error_hist": _figure(
                "hist",
                "The two error distributions are near copies",
                "Absolute log error, 162 filer-years",
                {
                    "edges": edges,
                    "series": [
                        {"name": "Warranted multiple", "role": "model", "counts": model_counts},
                        {"name": "Sector median", "role": "baseline", "counts": base_counts},
                    ],
                    "format": "num:1",
                    "binLabel": "Error",
                },
            ),
            # The row that stands out is one where the model sits left of its baseline:
            # its label must sit beside the model's dot, never beside the baseline's.
            "fold_scores": _figure(
                "dot",
                "The model trails the sector median in the second fold",
                "Log MAE per walk-forward fold, model against sector median",
                {
                    "rows": [
                        {"label": "Fold 1", "values": {"model": 0.411, "median": 0.409}},
                        {"label": "Fold 2", "values": {"model": 0.352, "median": 0.452}},
                        {"label": "Fold 3", "values": {"model": 0.438, "median": 0.391}},
                    ],
                    "series": [
                        {"key": "model", "name": "Warranted multiple", "role": "model"},
                        {"key": "median", "name": "Sector median", "role": "baseline"},
                    ],
                    "format": "num:3",
                    "gapFormat": "signed:3",
                    "gapLabel": "Model minus median",
                    "labelHeader": "Fold",
                },
            ),
            # Every value of the first series is zero, so no row stands out and none is labelled.
            # Two rows in a row put three points at zero, which must be drawn apart without
            # running into each other.
            "misdated": _figure(
                "dot",
                "Dating a split at either end of its window breaks periods; the filing basis breaks none",
                "Periods whose filings disagree after adjustment, by how each split is dated",
                {
                    "rows": [
                        {"label": "MSFT", "values": {"filed": 0, "late": 0, "early": 0}},
                        {"label": "AMZN", "values": {"filed": 0, "late": 0, "early": 0}},
                        {"label": "NVDA", "values": {"filed": 0, "late": 3, "early": 0}},
                        {"label": "CRM", "values": {"filed": 0, "late": 0, "early": 5}},
                    ],
                    "series": [
                        {"key": "filed", "name": "Per-filing basis", "role": "model"},
                        {"key": "late", "name": "Late end", "role": "baseline"},
                        {"key": "early", "name": "Early end", "role": "alt"},
                    ],
                    "format": "int",
                    "labelHeader": "Filer",
                    "zero": True,
                },
            ),
        },
    )


def _fade() -> dict[str, Any]:
    years = [f"Y{i}" for i in range(6)]
    model = [0.34, 0.25, 0.19, 0.15, 0.12, 0.10]
    linear = [0.34, 0.28, 0.22, 0.16, 0.10, 0.05]
    realised = [0.34, 0.27, 0.23, 0.19, 0.16, 0.14]
    cohorts = ["2010 to 2012", "2013 to 2015", "2016 to 2018", "2019 to 2021"]
    grid = [[round(_wave(i * 5 + j, 7.0, 1.8, 0.4) - 0.3, 1) for j in range(5)] for i in range(len(cohorts))]
    grid[3][4] = None
    names = (
        "mna_opex_load",
        "margin_gross",
        "returns_deferred_revenue_to_revenue",
        "scale_log_revenue",
        "growth_revenue_1y",
        "mna_years_listed",
        "capital_current_ratio",
    )
    coefficients = [
        (name, [round(_wave(i * 3 + j, 4.3, 1.4, 0.9) + (1.6 if i >= 4 else 0.0), 2) for j in range(5)])
        for i, name in enumerate(names)
    ]
    # Rows that change sign first, so the two groups are contiguous.
    coefficients.sort(key=lambda c: not (min(c[1]) < 0 < max(c[1])))
    flipped = sum(1 for _, vals in coefficients if min(vals) < 0 < max(vals))
    return _section(
        "fade",
        "Growth fade",
        "The learned fade decays too fast and loses to a straight line toward GDP growth.",
        headline=_headline(
            "MAE, growth points", 4.12, "linear fade to GDP", 3.87, 138, False, "loses",
            "MAE of 4.12 growth points against 3.87 for a linear fade, worse at every horizon beyond year two.",
        ),
        figures={
            "paths": _figure(
                "line",
                "Realised growth holds up longer than either fade",
                "Median revenue growth after a year above 30%, 138 filers",
                {
                    "series": [
                        {"name": "Learned fade", "role": "model", "values": [{"x": x, "y": y} for x, y in zip(years, model)]},
                        {"name": "Linear fade", "role": "baseline", "values": [{"x": x, "y": y} for x, y in zip(years, linear)]},
                        {"name": "Realised", "role": "third", "values": [{"x": x, "y": y} for x, y in zip(years, realised)]},
                    ],
                    "x": {"label": "Year after"},
                    "format": "pct:0",
                    "domain": [0, 0.4],
                    "yTitle": "Revenue growth",
                    "points": [{"x": "Y3", "y": realised[3], "label": "Realised still 19%", "role": "third"}],
                },
                note={"what": "Survivors only", "why": "A filer whose growth collapses is often bought and stops filing, so the realised path runs high."},
            ),
            "error_grid": _figure(
                "heat",
                "The fade loses most at long horizons in recent cohorts",
                "Linear fade error minus learned fade error, growth points; blue is where the model wins",
                {
                    "rows": cohorts,
                    "cols": ["Y1", "Y2", "Y3", "Y4", "Y5"],
                    "values": grid,
                    "scale": "diverging",
                    "format": "signed:1",
                    "scaleLabel": "Baseline error minus model error",
                    "rowHeader": "Cohort",
                    "colTitle": "Years after the fade starts",
                },
            ),
            # Identifiers in the mono face at full length, grouped, on fixed colour breaks.
            "coefficients": _figure(
                "heat",
                f"{flipped} of {len(coefficients)} coefficients change sign between folds",
                "Standardised coefficient per walk-forward fit",
                {
                    "rows": [name for name, _ in coefficients],
                    "cols": [f"Fold {j + 1}" for j in range(5)],
                    "values": [vals for _, vals in coefficients],
                    "scale": "diverging",
                    "breaks": [0.1, 0.5],
                    "format": "signed:2",
                    "scaleLabel": "Standardised coefficient",
                    "valueLabel": "Coefficient",
                    "rowHeader": "Feature",
                    "groups": [
                        {"label": f"Changes sign, {flipped}", "count": flipped},
                        {"label": f"Same sign in every fit, {len(coefficients) - flipped}", "count": len(coefficients) - flipped},
                    ],
                    "mono": True,
                    "labelAlign": "start",
                    "cellMax": 88,
                    "cellHeight": 24,
                    "cellLabels": False,
                    "rowNotes": [
                        f"mean {sum(coefficients[0][1]) / 5:+.2f}".replace("-", "−")
                    ] + [None] * (len(coefficients) - 1),
                    "rowTips": [[{"label": "Row", "value": name}] for name, _ in coefficients],
                },
                wide=True,
            ),
            # Tiles on the page plane, the kit's tileset: no card, but the same
            # table view, data toggle and source line a card figure carries.
            "panel_reach": _figure(
                "tiles",
                "The panel behind every figure here",
                "What the fade was fitted on, before any figure was drawn",
                {"tiles": [
                    {"label": "Company-years", "value": 2334, "format": "int", "sub": "Filed 2009 to 2026"},
                    {"label": "Companies", "value": 223, "format": "int", "sub": "One revenue ladder each"},
                    {"label": "Median years a filer supports", "value": 12, "format": "int", "sub": "The deepest reach 19"},
                ]},
            ),
            # Tiles drawn in a card, with a second set of tiles as a part of the same card.
            "year_five": _figure(
                "tiles",
                "The learned fade ends year five 8% below the straight line",
                "Revenue in year five on each path, USD millions, from 1,000mm today",
                {"tiles": [
                    {"label": "Linear fade", "value": 2210.4, "format": "mm", "sub": "Year 5 revenue"},
                    {
                        "label": "Learned fade",
                        "value": 2031.9,
                        "format": "mm",
                        "sub": "8.1% below the line",
                        "delta": {"value": -0.081, "format": "pct:1", "label": "against the line"},
                    },
                ]},
                card=True,
            ),
            "year_five_value": _figure(
                "tiles",
                "The same DCF moves by 2.10 a share between the paths",
                "USD per share, one DCF on each path",
                {"tiles": [
                    {"label": "Linear fade", "value": 31.42, "format": "num:2", "sub": "EV 3,120mm"},
                    {"label": "Learned fade", "value": 29.32, "format": "num:2", "sub": "EV 2,911mm"},
                ]},
                part="year_five",
            ),
        },
        also_from=("year_five_value",),
    )


def _propensity() -> dict[str, Any]:
    edges = [round(0.02 * i, 2) for i in range(11)]
    counts = [212, 488, 402, 281, 180, 112, 71, 44, 28, 22]
    deciles = [{"label": f"D{i + 1}", "value": 0.012 + 0.0068 * i + _wave(i, 4.0, 0.004)} for i in range(10)]
    return _section(
        "propensity",
        "Acquisition propensity",
        "The model ranks targets slightly better than firm size alone, but not significantly so.",
        headline=_headline(
            "AUC", 0.561, "size-only logit", 0.548, 1840, True, "not_significant",
            "auc of 0.561 against 0.548 for a size-only logit; the DeLong test gives p = 0.21.",
        ),
        figures={
            "score_hist": _figure(
                "hist",
                "Most filers score under 6%",
                "Predicted 12-month acquisition probability, 1,840 filer-years",
                {"edges": edges, "series": [{"name": "Filer-years", "role": "model", "counts": counts}], "format": "pct:0", "binLabel": "Probability", "reference": [{"value": 0.043, "label": "Base rate 4.3%"}]},
            ),
            # Not "calibration": the real propensity renderer reads a figure of that name as its dumbbell.
            "decile_rates": _figure(
                "column",
                "Observed rates rise with the predicted decile",
                "Share of filers acquired within 12 months, by predicted decile",
                {"rows": deciles, "format": "pct:1", "valueLabel": "Acquired", "labelHeader": "Decile", "reference": [{"value": 0.043, "label": "Base rate"}]},
            ),
            # Bars whose row labels name them, so the legend is off.
            "precision": _figure(
                "hbar",
                "One in 15 of the model's top 20 was bought, one in 35 of the size sort's",
                "Precision at 20, mean over 19 screen dates",
                {
                    "rows": [
                        {"label": "Fitted model", "value": 0.0658, "role": "model"},
                        {"label": "Size-only sort", "value": 0.0289, "role": "baseline"},
                    ],
                    "format": "pct:1",
                    "valueLabel": "Precision at 20",
                    "labelHeader": "Screen",
                    "legend": False,
                    "reference": [{"value": 0.0378, "label": "Base rate 3.8%"}],
                },
            ),
        },
        refusals=[
            {"what": "decile_rates", "why": "Deciles 9 and 10 hold 11 and 7 events, under the 20 the reliability test needs, so no calibration slope is stated."},
            {"what": "Deal premium model", "why": "Only 41 completed deals carry a disclosed premium in the fixtures, too few to fit and score out of sample."},
        ],
    )


def _engine() -> dict[str, Any]:
    methods = [
        ("DCF, perpetuity growth", 342, 401, 468, "model"),
        ("DCF, exit multiple", 318, 379, 452, "model"),
        ("Comps, EV/Revenue", 296, 351, 410, "model"),
        ("Comps, EV/EBITDA", 305, 348, 395, "model"),
        ("Precedent transactions", 360, 432, 520, "model"),
        ("52-week trading range", 309, None, 468, "baseline"),
    ]
    refused = {"chip": "refused", "chipText": "Refused"}
    ledger = [
        {"method": {"text": "DCF", "sub": "one caution under the terminal value"}, "figures": 4, "refused": 0, "entry": "techval.dcf", "note": "none"},
        {"method": {"text": "Comparable companies", "sub": "multiples drawn under a pinned date"}, "figures": 3, "refused": 0, "entry": "techval.comps", "note": "none"},
        {"method": {"text": "Merger model", "sub": "refused at collection"}, "figures": 0, "refused": 2, "entry": "techval.merger", "note": refused},
    ]
    return _section(
        "engine",
        "Valuation engine",
        "Four of five methods bracket the current price; precedent transactions sit above it on control premia.",
        # The renderer names the football field first; the rest follow their order
        # numbers, not their keys: the bridge, the segments, then the ledger.
        figures={
            "methods": _figure(
                "table",
                "Two of three methods draw every figure they are asked for",
                "Figures per valuation method, one offline run; a refused cell is one the method declined to fill",
                {
                    "columns": [
                        {"key": "method", "label": "Method"},
                        {"key": "figures", "label": "Figures", "align": "right", "format": "int"},
                        {"key": "refused", "label": "Refused", "align": "right", "format": "int"},
                        {"key": "entry", "label": "Module", "mono": True},
                        {"key": "note", "label": "Figure", "nowrap": True},
                    ],
                    "rows": ledger,
                },
                order=3,
                wide=True,
            ),
            "football": _figure(
                "range",
                "Most methods bracket the price",
                "MSFT, USD per share, fiscal 2024 fixtures",
                {
                    "rows": [{"label": l, "lo": lo, "mid": mid, "hi": hi, "role": r} for l, lo, mid, hi, r in methods],
                    "format": "num:0",
                    "reference": [{"value": 415.2, "label": "Price 415"}],
                    "roleLabels": {"model": "Valuation method", "baseline": "Market context"},
                    "labelHeader": "Method",
                    "tableHeaders": {"lo": "Low, USD", "mid": "Mid, USD", "hi": "High, USD"},
                },
            ),
            "sotp": _figure(
                "waterfall",
                "Cloud carries most of the enterprise value",
                "Sum of the parts to equity value, USD bn",
                {
                    "steps": [
                        {"label": "Intelligent cloud", "value": 1452.3},
                        {"label": "Productivity", "value": 1108.9},
                        {"label": "Personal computing", "value": 402.5},
                        {"label": "Cash and investments", "value": 75.5},
                        {"label": "Debt and leases", "value": -97.8},
                        {"label": "Minority interests", "value": -12.4},
                    ],
                    "total": {"label": "Equity value", "value": 2929.0},
                    "format": "num:0",
                    "valueLabel": "USD bn",
                },
                order=1,
            ),
            # No step subtracts, so the legend carries no entry for a kind of bar the bridge never draws.
            "segment_income": _figure(
                "waterfall",
                "Operating income foots to the segments' own sum",
                "Operating income by segment, USD bn; nothing is taken off outside the segments",
                {
                    "steps": [
                        {"label": "Intelligent cloud", "value": 42.9},
                        {"label": "Productivity", "value": 40.5},
                        {"label": "Personal computing", "value": 19.3},
                        {"label": "Outside the segments", "value": 0.0},
                    ],
                    "total": {"label": "Operating income", "value": 102.7},
                    "format": "num:1",
                    "valueLabel": "USD bn",
                    "upLabel": "Segment operating income",
                    "downLabel": "Taken off outside the segments",
                    "totalLegend": "Operating income",
                },
                order=2,
            ),
        },
        also_from=("sotp",),
    )


def _tmt() -> dict[str, Any]:
    return _section(
        "tmt",
        "TMT operating metrics",
        "Refused: the committed fixtures do not carry the operating facts this section would chart.",
        status="refused",
        refusals=[
            {"what": "Subscribers and ARPU", "why": "No fact in the committed fixtures carries a subscriber or ARPU tag, and the TMT module does not estimate them."},
            {"what": "Spectrum-adjusted leverage", "why": "Licence carrying values appear only in filing footnotes, which the offline fixtures do not include."},
        ],
    )


def _datalayer() -> dict[str, Any]:
    years = list(range(2012, 2025))
    facts = [{"label": f"FY{str(y)[2:]}", "value": 18400 + 2150 * i + round(_wave(i, 5.0, 900))} for i, y in enumerate(years)]
    lag = [{"x": y, "y": 61 - 1.3 * i + _wave(i, 4.0, 2.5)} for i, y in enumerate(years)]
    # Weekly closes over four years, a price that multiplies about tenfold.
    start = date(2021, 1, 8)
    closes = [
        {"x": (start + timedelta(days=7 * i)).isoformat(), "y": round(12.0 * math.exp(0.011 * i + _wave(i, 26.0, 0.18)), 2)}
        for i in range(208)
    ]
    windows = [
        {"from": "2022-05-20", "to": "2022-08-12", "label": "4-for-1 window", "worst_day": -0.041},
        {"from": "2024-03-01", "to": "2024-05-24", "label": "10-for-1 window", "worst_day": -0.066},
    ]
    low = min(closes, key=lambda p: p["y"])
    return _section(
        "datalayer",
        "Data layer",
        "Parsed XBRL facts grow every year and filers report sooner than they did a decade ago.",
        figures={
            "facts_by_year": _figure(
                "column",
                "Facts parsed per year have roughly doubled",
                "XBRL facts across all fixture filers, by fiscal year",
                {"rows": facts, "format": "int", "valueLabel": "Facts", "labelHeader": "Fiscal year"},
            ),
            "filing_lag": _figure(
                "line",
                "Filing lag has shortened by about two weeks",
                "Median days from period end to 10-K filing",
                {"series": [{"name": "Median lag", "role": "model", "values": lag}], "x": {"label": "Fiscal year"}, "format": "num:0"},
            ),
            # Dates along x, a log scale, shaded windows and one labelled point.
            "closes": _figure(
                "line",
                "The closes carry no split step inside either window",
                "Close, USD, last close of each week",
                {
                    "series": [{"name": "Close", "role": "model", "values": closes}],
                    "x": {"label": "Week ending", "type": "date"},
                    "format": "num:2",
                    "yScale": "log",
                    "yTitle": "USD, log scale",
                    "endLabels": False,
                    "shade": [
                        {"from": w["from"], "to": w["to"], "label": w["label"], "tip": {"label": w["label"], "value": w["worst_day"], "format": "pct:1"}}
                        for w in windows
                    ],
                    "shadeLegend": "Split window, from the last filing on the old basis to the first on the new",
                    "points": [{"x": low["x"], "y": low["y"], "label": f"Lowest close, {low['y']:.2f}"}],
                    "table": [
                        {
                            "caption": "Closes",
                            "columns": [
                                {"key": "x", "label": "Week ending", "mono": True},
                                {"key": "y", "label": "Close, USD", "align": "right", "format": "num:2"},
                            ],
                            "rows": closes,
                        },
                        {
                            "caption": "Split windows",
                            "columns": [
                                {"key": "label", "label": "Window"},
                                {"key": "from", "label": "From", "mono": True},
                                {"key": "to", "label": "To", "mono": True},
                                {"key": "worst_day", "label": "Worst day", "align": "right", "format": "pct:1"},
                            ],
                            "rows": windows,
                        },
                    ],
                },
                wide=True,
            ),
        },
    )


def _sample() -> dict[str, Any]:
    return _section(
        "sample",
        "Sample",
        "The sample section has not been collected in this snapshot.",
        status="not_built",
    )


def gallery_snapshot() -> dict[str, Any]:
    """The synthetic snapshot the gallery page renders."""
    rest = [_signal(), _encoder(), _warranted(), _fade(), _propensity(), _engine(), _tmt(), _datalayer(), _sample()]
    sections = [_overview(rest), *rest]
    return {
        "schema": 1,
        "title": "techval chart kit gallery",
        "techval_commit": "none",
        "collected_at": "not collected",
        "fixtures": {},
        "sections": {s["id"]: s for s in sections},
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python gallery.py OUT.html")
    print(write_gallery(sys.argv[1]))
