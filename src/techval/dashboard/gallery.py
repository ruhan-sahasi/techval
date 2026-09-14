"""A gallery page for the dashboard front end, drawn from synthetic data.

``render_gallery()`` returns one HTML page that exercises every chart in the
kit, every verdict chip, a figure carrying a refusal note, a refused section
and a section that has not been collected. It is a development tool for looking
at the kit in both themes, not a results page: the page says so in a notice
under its title, records no fixtures, and every provenance entry names this
module rather than a model entry point.

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
    "&family=IBM+Plex+Sans:wght@400;500;600"
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


def _figure(kind: str, title: str, subtitle: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "title": title, "subtitle": subtitle, "data": data}


def _provenance(*figure_ids: str) -> list[dict[str, Any]]:
    return [
        {"figure": fid, "entry_point": GALLERY_ENTRY_POINT, "inputs": [], "seconds": 0.0}
        for fid in figure_ids
    ]


def _section(
    sid: str,
    title: str,
    takeaway: str,
    *,
    status: str = "ok",
    headline: dict[str, Any] | None = None,
    figures: dict[str, Any] | None = None,
    refusals: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    figures = figures or {}
    return {
        "id": sid,
        "title": title,
        "takeaway": takeaway,
        "status": status,
        "refusals": refusals or [],
        "headline": headline,
        "figures": figures,
        "provenance": _provenance(*figures),
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


def _overview() -> dict[str, Any]:
    tiles = [
        {"label": "Signal rank IC", "value": 0.0412, "format": "num:4", "sub": "Momentum baseline 0.0300", "status": "inside_noise"},
        {
            "label": "Peer encoder nDCG@10",
            "value": 0.5407,
            "format": "num:4",
            "sub": "Popularity prior 0.2213",
            "status": "beats",
            "delta": {"value": 0.3194, "format": "signed:4", "label": "lift"},
        },
        {
            "label": "Warranted multiple log MAE",
            "value": 0.412,
            "format": "num:3",
            "sub": "Sector median 0.409",
            "status": "ties",
            "delta": {"value": 0.003, "format": "signed:3", "label": "vs baseline", "higher_is_better": False},
        },
        {
            "label": "Growth fade MAE",
            "value": 4.12,
            "format": "num:2",
            "sub": "Linear fade 3.87 points",
            "status": "loses",
            "delta": {"value": 0.25, "format": "signed:2", "label": "vs baseline", "higher_is_better": False},
        },
        {"label": "Propensity AUC", "value": 0.561, "format": "num:3", "sub": "Size-only logit 0.548", "status": "not_significant"},
        {"label": "TMT operating KPIs", "value": None, "sub": "Subscriber tags absent from fixtures", "status": "refused"},
    ]
    seconds = [("encoder", 37.1), ("fade", 22.4), ("signal", 18.9), ("warranted", 12.6), ("propensity", 9.3), ("engine", 4.8), ("datalayer", 1.7)]
    return _section(
        "overview",
        "Overview",
        "One model beats its baseline outside the noise; the others tie, sit inside the noise, lose, or refuse.",
        figures={
            "verdicts": _figure("tiles", "Verdicts at a glance", "", {"tiles": tiles}),
            "collect_seconds": _figure(
                "hbar",
                "The peer encoder takes longest to collect",
                "Seconds per section, one offline run",
                {
                    "rows": [{"label": name, "value": s} for name, s in seconds],
                    "format": "num:1",
                    "valueLabel": "Seconds",
                    "labelHeader": "Section",
                },
            ),
        },
    )


def _signal() -> dict[str, Any]:
    folds = [
        {"label": f"F{i + 1}", "value": 0.011 + _wave(i, 5.3, 0.024, 0.6)} for i in range(12)
    ]
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
            "ic_lift_by_fold": _figure(
                "column",
                "The lift changes sign across folds",
                "Rank IC, model minus momentum, per walk-forward fold",
                {"rows": folds, "diverging": True, "format": "signed:3", "valueLabel": "IC lift", "labelHeader": "Fold"},
            ),
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
    sectors = [("Software", 0.61, 0.24), ("Semiconductors", 0.57, 0.31), ("Internet", 0.55, 0.19), ("IT services", 0.48, 0.22), ("Media", 0.44, 0.27), ("Telecom", 0.39, 0.33)]
    return _section(
        "encoder",
        "Peer encoder",
        "The encoder ranks true peers far above the popularity prior in every sector.",
        headline=_headline(
            "nDCG@10", 0.5407, "popularity prior", 0.2213, 162, True, "beats",
            "nDCG@10 of 0.5407 against 0.2213 for the popularity prior, outside the fold noise on 162 query filers.",
        ),
        figures={
            "ndcg_by_method": _figure(
                "hbar",
                "The encoder leads every peer rule",
                "nDCG@10 on 162 query filers",
                {
                    "rows": [
                        {"label": "Peer encoder", "value": 0.5407, "role": "model"},
                        {"label": "Same four-digit SIC", "value": 0.3310, "role": "alt"},
                        {"label": "Popularity prior", "value": 0.2213, "role": "baseline"},
                        {"label": "Random peers", "value": 0.0480, "role": "baseline"},
                    ],
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
                },
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
            "Log MAE of 0.4120 against 0.4090 for the sector median, a difference inside one standard error.",
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
                },
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
                },
            ),
        },
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
            "AUC of 0.561 against 0.548 for a size-only logit; the DeLong test gives p = 0.21.",
        ),
        figures={
            "score_hist": _figure(
                "hist",
                "Most filers score under 6%",
                "Predicted 12-month acquisition probability, 1,840 filer-years",
                {"edges": edges, "series": [{"name": "Filer-years", "role": "model", "counts": counts}], "format": "pct:0", "binLabel": "Probability", "reference": [{"value": 0.043, "label": "Base rate 4.3%"}]},
            ),
            "calibration": _figure(
                "column",
                "Observed rates rise with the predicted decile",
                "Share of filers acquired within 12 months, by predicted decile",
                {"rows": deciles, "format": "pct:1", "valueLabel": "Acquired", "labelHeader": "Decile", "reference": [{"value": 0.043, "label": "Base rate"}]},
            ),
        },
        refusals=[
            {"what": "calibration", "why": "Deciles 9 and 10 hold 11 and 7 events, under the 20 the reliability test needs, so no calibration slope is stated."},
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
    return _section(
        "engine",
        "Valuation engine",
        "Four of five methods bracket the current price; precedent transactions sit above it on control premia.",
        figures={
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
            ),
        },
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
    sections = [_overview(), _signal(), _encoder(), _warranted(), _fade(), _propensity(), _engine(), _tmt(), _datalayer(), _sample()]
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
