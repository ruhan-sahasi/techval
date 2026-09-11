"""Two commands that front the fitted half of the engine: the fade curve and the harness.

``techval fade TICKER`` puts a measured revenue growth path beside the typed one
and values the company on both. ``techval signal`` asks whether any score in this
package predicts anything at all, and is built so that the answer can be no.

They live in one module because they are two ends of the same argument. The fade
curve is the most consequential fitted number the package produces: a discounted
cash flow's revenue growth path moves the answer more than the discount rate, the
margin path and the terminal multiple put together, and in a default run it is
typed into a YAML file by whoever opened the file. ``techval.ml.forecast`` fits it
on 2,798 point-in-time company-years instead. The harness is the thing that stops
that from being a story. It takes a score, any score, and reports whether the
ordering it implies was related to what the stocks did next, and its reference
result on this package's own cheapness score is that it was not.

**What ``fade`` prints, and why it is a comparison rather than a forecast.** The
output an analyst can use is not a growth path. It is two growth paths and the
two valuations they imply, because the fitted number only means something against
the number it replaces. So the table carries the assumed fade from
``assumptions.dcf``, the fitted fade, and the two ends of the band the fit
actually supports, and it values all four. On Datadog those four come out at
36.65, 40.61, 26.97 and 66.69 a share. Quoting the 40.61 on its own, without the
26.97 and the 66.69 beside it, would be reporting a point estimate from a model
whose out-of-fold residual band is forty growth points wide.

**The baseline is printed at every horizon, and the model card's own verdict is
carried rather than summarised.** "Next year's growth equals this year's growth"
is a hard baseline and the fitted curve does not beat it at one year. It beats it
comfortably at two and three. Both facts are on the table, and so are the two
weaker baselines, the training mean and the mean of the company's own
sub-vertical, because a model that beats persistence and loses to its own sector
average has learned something about the sector rather than about the company. See
``_horizon_table`` for what that comparison turns up here, which is not quite what
the headline says.

**Survivorship is stated on every run, with its direction.** Companies whose
growth collapses get acquired or delisted and stop filing, so a curve fitted on
whoever is left fades too slowly and every valuation built on it is too
optimistic. ``techval.ml.forecast`` keeps 119 delisted filers in the panel until
the day they stop, which is most of the fix, and the residual bias is the years
after they left, which no panel can hold. The command prints the measured size of
the part that was fixed and names the direction of the part that was not.

**What ``signal`` prints, and why it leads with the counts.** A harness that
reports a t-statistic before it reports how many independent observations are
behind it has already made its case. So the counts come first, effective
observations before company-dates: the reference run on this package's cheapness
score has 2,604 company-dates, 35 quarterly cross-sections, about 8 independent
twelve month periods and about 13 effective observations once the overlap is
accounted for. The mean information coefficient is -0.0984 against +0.0002 for a
random score with the same cross-sectional shape. The naive t-statistic is -2.65
and would be written up; the Newey-West figure is -1.62 and would not. The
command prints NOT SIGNIFICANT in those words.

**The overlap correction, explained the way the module measured it rather than
the way it is usually stated.** The usual claim is that overlapping windows
inflate the t-statistic. ``techval.ml.signals`` tested that and it is not right on
its own. A cross-sectional rank correlation is computed inside a single date, so a
market move common to every name cancels out of it, and a score redrawn from noise
at each rebalance produces a coefficient series with no autocorrelation worth
correcting whatever the returns underneath are doing. What autocorrelates the
series is a score that persists: cheap this quarter is cheap next quarter, and
paired with returns that share three quarters of their path the first-order
autocorrelation lands on 0.76 against a theoretical 0.75. So the correct statement
is that overlapping windows inflate the t-statistic OF A PERSISTENT SCORE. Every
score in this package is persistent, so the warning stands, and the command prints
the measured autocorrelation beside the inflation so a reader can see which of the
two is doing the work on their own data.

**Where the data comes from.** Both commands run live by default and both take a
recorded panel instead. That is not a test affordance bolted on: fitting the fade
curve live is 224 ``companyfacts`` calls against the SEC at the fair-access
throttle, and the repository already commits the recorded panels that the model
cards were measured on. ``--panel tests/fixtures/fade_companyfacts.json.gz`` fits
the same curve in twelve seconds, and ``--scores`` with ``--prices`` reproduces
the reference signal result exactly.

Nothing here computes a number. Every figure on every table comes from a
``FadeModel``, a ``FadeComparison`` or a ``SignalResult``, with two exceptions
that are aggregates of a panel taken for display and are marked as such where they
are defined.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import Assumptions
from .edgar import CompanyFacts, EdgarClient, HttpCache
from .errors import ConfigError, MissingDataError, TechvalError
from .ev_bridge import build_ev_bridge
from .financials import build_financials
from .market import MarketData, PriceSeries, make_price_source
from .ml import signals as signals_module
from .ml.forecast import (
    DelistedAwareClient,
    FadeModel,
    FadePanel,
    build_fade_panel,
    compare_fade,
    fade_universe,
    fit_fade,
    is_delisted,
)
from .ml.signals import (
    _EXIT_ACQUIRED,
    _EXIT_CENSORED,
    _EXIT_DELISTED,
    _EXIT_HELD,
    _EXIT_NO_PRICE,
    Score,
    SignalResult,
    ev_revenue_scores,
)
from .wacc import compute_wacc

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fitted revenue fade curves, and the harness that tests whether a score predicts returns.",
)

# Same width rule as ``cli.py``: rich falls back to eighty columns when stdout is
# not a terminal, which crushes a seven column table into something unreadable in
# a piped or redirected run.
console = Console(width=None if sys.stdout.isatty() else 120)


# --------------------------------------------------------------------------- #
# Rendering helpers
#
# Deliberately duplicated from ``cli.py`` rather than imported from it. A later
# agent mounts this module into ``cli.py`` with ``add_typer``, so an import in the
# other direction would close a cycle. They are twenty lines of formatting and the
# cycle is not worth paying to avoid them.
# --------------------------------------------------------------------------- #


def _rule(title: str) -> None:
    console.print()
    console.rule(Text(title, style="bold"), style="dim")


def _money(v: float | None, dp: int = 0) -> str:
    if v is None:
        return "n/a"
    return f"({abs(v):,.{dp}f})" if v < 0 else f"{v:,.{dp}f}"


def _pct(v: float | None, dp: int = 1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{dp}%}"


def _num(v: float | None, dp: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.{dp}f}"


def _signed(v: float | None, fmt: str) -> str:
    """Signed formatting that prints ``n/a`` rather than ``+nan%``.

    A table cell reading ``+nan%`` is not a missing value, it is a missing value
    that has been given a sign and a unit, and a reader skimming a column of
    percentages will take it for one.
    """
    if v is None or not np.isfinite(v):
        return "n/a"
    return format(v, fmt)


def _notes(items: Sequence[str], *, heading: str = "Notes") -> None:
    if not items:
        return
    console.print(f"\n[bold]{heading}[/bold]")
    for n in items:
        style = "yellow" if n.upper().startswith(("FLAG", "WARN")) else "dim"
        console.print(Text(f"  - {n}", style=style))


def _kv(rows: Sequence[tuple[str, str]], *, title: str = "", bold: Sequence[str] = ()) -> Table:
    t = Table(title=title or None, title_justify="left", box=None, pad_edge=False)
    t.add_column("", no_wrap=True)
    t.add_column("", justify="right")
    for label, shown in rows:
        style = "bold" if label in bold else ""
        t.add_row(Text(label, style=style), Text(shown, style=style))
    return t


def _fail(exc: TechvalError) -> None:
    """The clean refusal ``cli.py`` prints instead of a traceback."""
    console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
    raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# Shared options
# --------------------------------------------------------------------------- #

_CFG = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file.")
_NOCACHE = typer.Option(False, "--no-cache", help="Bypass the HTTP cache.")
_ASOF = typer.Option(
    None,
    "--as-of",
    help=(
        "Fit and value as at this date (YYYY-MM-DD). Facts filed later are discarded, "
        "prices stop there, and the panel is cut back to the filings and the forward "
        "labels that were public by then."
    ),
)


def _setup(config: Path | None, no_cache: bool, as_of: str | None = None):
    """``cli._setup``, repeated here for the same reason the formatters are."""
    assumptions = Assumptions.load(config)
    if as_of:
        assumptions.as_of = as_of
    knowledge = date.fromisoformat(assumptions.as_of) if assumptions.as_of else None
    cache = HttpCache(enabled=not no_cache)
    client = EdgarClient(cache, knowledge_date=knowledge)
    source = make_price_source(assumptions.price_source, cache, assumptions.price_csv_dir)
    market = MarketData(source, cache, today=knowledge or date.today())
    return assumptions, client, market, cache, source


def _open_text(path: Path) -> io.TextIOBase:
    """Read a ``.gz`` the same way as a plain file.

    Every recorded panel in this repository is gzipped because the uncompressed
    ones run to tens of megabytes, and a user who unzips one to look at it should
    not then have to zip it again to feed it back in.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


# --------------------------------------------------------------------------- #
# The fade panel: live, or from a recorded blob
# --------------------------------------------------------------------------- #


class _RecordedFactsClient:
    """Serves companyfacts payloads out of a recorded blob, unpinned.

    Unpinned on purpose, and this is the one place in the engine where that is
    correct rather than sloppy. ``build_fade_panel`` pins the fact set once per
    fiscal year, to the filing date of the report that first carried that year,
    which is a different knowledge date for every row. A client handed in
    pre-pinned could only serve one of them, so the pinning happens inside the
    panel builder and the client stays open.
    """

    def __init__(self, payloads: dict[str, dict]) -> None:
        self.payloads = payloads

    def company_facts(self, ticker: str) -> CompanyFacts:
        payload = self.payloads.get(ticker.upper())
        if payload is None:
            raise MissingDataError(
                "companyfacts",
                ticker=ticker,
                hint="not present in the recorded panel file supplied to --panel",
            )
        return CompanyFacts(payload, ticker)

    def ticker_to_cik(self, ticker: str) -> int:
        return int(self.payloads[ticker.upper()].get("cik", 0))


def _recorded_panel(path: Path, assumptions: Assumptions, only: set[str] | None) -> FadePanel:
    """Build the panel from a recorded blob of companyfacts payloads.

    The file is JSON, optionally gzipped, with two keys. ``payloads`` maps ticker
    to the raw ``companyfacts`` payload the SEC served, and ``meta`` maps ticker
    to at least ``{"sub_vertical": "..."}``. ``tests/fixtures/fade_companyfacts.json.gz``
    is one such file, committed with the repository, and it is what the model card
    in ``techval.ml.forecast`` was measured on.

    The sub-vertical has to come from the file rather than from
    ``taxonomy.classify`` because the filers that matter most to this model are
    the ones that no longer exist, and ``classify`` resolves a ticker through the
    SEC's current ticker file, in which a delisted company does not appear.
    """
    with _open_text(path) as handle:
        blob = json.load(handle)
    payloads = blob.get("payloads")
    meta = blob.get("meta")
    if not isinstance(payloads, dict) or not isinstance(meta, dict):
        raise ConfigError(
            f"{path} is not a recorded fade panel. The file must be JSON with a "
            "'payloads' object mapping ticker to a companyfacts payload and a "
            "'meta' object mapping ticker to at least {'sub_vertical': ...}."
        )
    universe = {
        ticker.upper(): str(entry.get("sub_vertical", "other"))
        for ticker, entry in meta.items()
        if ticker.upper() in {k.upper() for k in payloads}
    }
    if only:
        universe = {t: v for t, v in universe.items() if t in only}
    if not universe:
        raise ConfigError(
            f"{path} carries no filer matching the requested universe, so there is "
            "nothing to fit a fade curve on."
        )
    client = _RecordedFactsClient({k.upper(): v for k, v in payloads.items()})
    return build_fade_panel(universe, client, assumptions)


def _truncate_panel(panel: FadePanel, knowledge: date) -> FadePanel:
    """Cut the panel back to what had been filed by ``knowledge``, labels included.

    This is the half of ``--as-of`` that nothing else does for us, and without it
    the option would be a lie. ``build_fade_panel`` pins each fiscal year to the
    filing date of the report that first carried it, which makes every FEATURE
    point in time, and it stops there: the panel runs to the present whatever date
    the valuation is struck at. Fit a curve on filings through 2026 and value a
    company as at 2020 and the growth path knows how the intervening six years
    went. The valuation would look excellent and the failure would be silent,
    which is the worst combination available.

    Two cuts are needed and only the first is obvious. An observation filed after
    the knowledge date goes entirely. A label also carries a filing date of its
    own, because the label for horizon h is the growth the filer printed in its
    10-K h years later, so an observation filed in 2019 still holds a three year
    label that was not public until 2022. ``FadeObservation.label_dates`` records
    it and this drops every label that had not been filed yet, which is what takes
    the recent end of the panel out of the training set rather than leaving it in
    with the answers attached.

    The cross-sectional winsorization needs no adjustment, and it is worth saying
    why rather than leaving it to be checked. It is grouped by the calendar year
    of the filing date, so an observation filed in 2019 was trimmed by the 2019
    cross-section and by nothing later. The bounds a surviving row was clipped to
    were all public by the knowledge date.
    """
    kept = []
    dropped_observations = 0
    dropped_labels = 0
    for observation in panel.observations:
        if observation.as_of > knowledge:
            dropped_observations += 1
            continue
        for horizon in [h for h, d in observation.label_dates.items() if d > knowledge]:
            observation.labels.pop(horizon, None)
            observation.label_dates.pop(horizon, None)
            dropped_labels += 1
        kept.append(observation)
    if not kept:
        raise ConfigError(
            f"no filing in this panel predates {knowledge}, so there is nothing to "
            "fit a fade curve on as at that date."
        )
    tickers = {o.ticker for o in kept}
    out = FadePanel(
        observations=kept,
        depth={t: n for t, n in panel.depth.items() if t in tickers},
        failures=list(panel.failures),
        winsorized=panel.winsorized,
        notes=list(panel.notes),
        random_seed=panel.random_seed,
    )
    out.notes.append(
        f"point in time: {dropped_observations:,} observation(s) filed after "
        f"{knowledge} were removed and {dropped_labels:,} forward label(s) that had "
        "not been filed by then were detached. The curve is fitted only on what a "
        "reader could have known on that date."
    )
    return out


def _live_panel(cache: HttpCache, assumptions: Assumptions, only: set[str] | None) -> FadePanel:
    """Build the panel from the SEC, through the client that can find the leavers.

    ``DelistedAwareClient`` rather than ``EdgarClient``, and the difference is the
    whole survivorship argument. ``EdgarClient.ticker_to_cik`` resolves through the
    SEC's current ``company_tickers.json``, so Splunk, VMware, Xilinx, Activision
    and 115 other filers raise ``MissingDataError`` that reads exactly like a
    mistyped ticker. A panel built by walking today's tickers backwards is a panel
    of winners, and the subclass falls back to a recorded CIK a name at a time.
    """
    universe = fade_universe()
    if only:
        universe = {t: v for t, v in universe.items() if t in only}
        missing = sorted(only - set(universe))
        if missing:
            raise ConfigError(
                "these tickers are not in the fade universe and carry no sub-vertical: "
                + ", ".join(missing)
                + ". Add them to techval.tmt.taxonomy.SEED, or supply a recorded "
                "panel with --panel whose meta names their sub-vertical."
            )
    if not universe:
        raise ConfigError("the fade universe is empty, so there is nothing to fit on")
    return build_fade_panel(universe, DelistedAwareClient(cache), assumptions)


# --------------------------------------------------------------------------- #
# fade: rendering
# --------------------------------------------------------------------------- #


def _render_curve(model: FadeModel, panel: FadePanel) -> None:
    """The fade curve as a banker would write it, in three numbers.

    Next year's growth equals an intercept plus a slope times this year's. One
    minus the slope is the share of the gap to the long-run level that closes in a
    year, and the ratio of the intercept to one minus the slope is the level it
    closes toward. Everything else in this module is a refinement of those three
    numbers, and they fit in a sentence.
    """
    _rule("The fade curve")
    console.print(
        _kv(
            [
                ("Persistence slope", _num(model.persistence_slope, 4)),
                ("Persistence intercept", _num(model.persistence_intercept, 4)),
                ("Fade per year, share of the gap closed", _pct(model.fade_per_year, 0)),
                ("Reverts toward", _pct(model.reversion_level)),
                ("Half-life, years", _num(model.half_life_years, 2)),
                ("Panel observations", f"{len(panel.observations):,}"),
                ("Filers with at least one fiscal year", f"{len(panel.tickers):,}"),
                ("Horizons fitted", ", ".join(f"{h}y" for h in sorted(model.fits))),
            ],
            bold=("Fade per year, share of the gap closed", "Reverts toward"),
        )
    )
    console.print(
        f"\n  [bold]Next year's growth = {model.persistence_intercept:+.4f} + "
        f"{model.persistence_slope:.4f} x this year's.[/bold]"
    )
    console.print(
        "  [dim]The straight line in assumptions.dcf fades a fifth of the gap a "
        "year. The filings fade about half of it.[/dim]"
    )


def _horizon_table(model: FadeModel) -> None:
    """Model against all three baselines at every horizon, and the honest reading.

    The brief for this command said the model ties persistence at one year and
    beats it at two and three, and that is exactly what the persistence column
    shows. It is also only two thirds of the comparison. At two and three years
    the baseline that matters is no longer persistence, it is the mean of the
    company's own sub-vertical, which scores 0.1578 and 0.1441 against the model's
    0.1476 and 0.1391. Those lifts are 0.0102 and 0.0050 against fold standard
    deviations of 0.0162 and 0.0176, so at every horizon the lift over the
    STRONGEST baseline is inside the fold-to-fold noise, even where the lift over
    persistence is not.

    That is not an argument against the fitted curve. The three-year model beats
    doing nothing by a wide margin and doing nothing is what a DCF's later years
    currently do. It is an argument against quoting the curve as though it had
    beaten everything, and the table prints the best baseline and the comparison
    against it so that nobody has to take the headline's word for which baseline
    was in play.

    The fold standard deviation is the dispersion of the model's own score across
    walk-forward folds, which is the convention ``EvalResult.verdict`` already uses
    for this comparison.
    """
    _rule("Model against the baselines, every horizon")
    t = Table(box=None, pad_edge=False)
    t.add_column("Horizon", no_wrap=True)
    t.add_column("Model MAE", justify="right")
    t.add_column("Persistence", justify="right")
    t.add_column("Training mean", justify="right")
    t.add_column("Sub-vertical mean", justify="right")
    t.add_column("Best baseline", justify="right", no_wrap=True)
    t.add_column("Lift on best", justify="right")
    t.add_column("Fold SD", justify="right")
    t.add_column("Reading", no_wrap=True)

    short = {
        "persistence": "persistence",
        "training_mean": "training mean",
        "sub_vertical_mean": "sub-vertical",
    }
    persistence_lines: list[str] = []
    best_by_horizon: list[tuple[int, str]] = []
    for horizon in sorted(model.fits):
        fit = model.fits[horizon]
        best_name, best_score = min(fit.baselines.items(), key=lambda kv: kv[1])
        lift = best_score - fit.evaluation.score
        sd = fit.evaluation.fold_sd
        if sd is None:
            reading = "no fold spread"
        elif lift <= 0:
            reading = "loses"
        elif lift < sd:
            reading = "inside fold noise"
        else:
            reading = "outside fold noise"
        style = "bold" if reading == "outside fold noise" else ""
        t.add_row(
            Text(f"{horizon} year", style=style),
            Text(_num(fit.evaluation.score), style=style),
            _num(fit.baselines.get("persistence")),
            _num(fit.baselines.get("training_mean")),
            _num(fit.baselines.get("sub_vertical_mean")),
            short.get(best_name, best_name.replace("_", " ")),
            Text(f"{lift:+.4f}", style=style),
            "n/a" if sd is None else _num(sd),
            Text(reading, style=style),
        )
        persistence = fit.baselines.get("persistence")
        if persistence is not None:
            # A lift smaller than the fold-to-fold dispersion is a tie whichever
            # way it points. Calling a lift of minus 0.0024 against a fold
            # standard deviation of 0.0490 a loss would be reading a sign off
            # noise, and calling the mirror image of it a win is the same mistake
            # in the direction that flatters the model.
            gap = persistence - fit.evaluation.score
            if sd is not None and abs(gap) < sd:
                verdict = "ties it"
            elif gap > 0:
                verdict = "beats it"
            else:
                verdict = "loses to it"
            persistence_lines.append(f"{horizon}y {gap:+.4f} and {verdict}")
        best_by_horizon.append((horizon, short.get(best_name, best_name)))
    console.print(t)
    console.print(
        "\n[dim]Mean absolute error on out-of-fold predictions, lower is better. "
        "Persistence is 'next year's growth equals this year's', which is the "
        "hardest of the three and the one the model card scores against. The fold "
        "standard deviation is the dispersion of the model's own score across "
        "walk-forward folds, which is the error bar EvalResult.verdict uses.[/dim]"
    )
    if persistence_lines:
        console.print(
            "\n  [bold]Against persistence alone:[/bold] "
            + "; ".join(persistence_lines)
            + "."
        )
    if best_by_horizon:
        console.print(
            "  [bold]Strongest baseline at each horizon:[/bold] "
            + "; ".join(f"{h}y {name}" for h, name in best_by_horizon)
            + ". Persistence is the hardest baseline at one year and not "
            "necessarily afterwards, so read the 'Reading' column, which compares "
            "the model against whichever baseline actually won, before quoting "
            "this curve as a model that beat the alternatives."
        )
    card = model.card.evaluation
    if card is not None:
        console.print(f"\n  [bold]One year, the model card's own verdict:[/bold] {card.verdict()}")


def _sub_vertical_fade(panel: FadePanel, sub_vertical: str, horizon: int) -> dict[str, float] | None:
    """Mean trailing and mean forward growth for one sub-vertical. Display only.

    This is an aggregate of the panel taken for the table and not a model output,
    which is why it lives in the command rather than in ``techval.ml.forecast``. It
    answers the one question the fitted path alone cannot: is this company being
    treated as unusual? A fitted first year of 21% means something different for a
    filer whose sub-vertical typically prints 17% than for one whose sub-vertical
    typically prints 28%, and without the comparison a reader cannot tell which
    they are looking at.

    No functional form is assumed. It is the arithmetic mean of what the filings
    said, over the observations of that sub-vertical which carry a label at this
    horizon.
    """
    rows = [
        o
        for o in panel.labelled(horizon)
        if str(o.sub_vertical) == str(sub_vertical)
    ]
    if not rows:
        return None
    trailing = float(np.mean([o.growth for o in rows]))
    forward = float(np.mean([o.labels[horizon] for o in rows]))
    return {
        "trailing": trailing,
        "forward": forward,
        # Points of growth lost, so a bigger number is a faster fade.
        "fade": trailing - forward,
        "n": float(len(rows)),
    }


def _render_paths(comparison, model: FadeModel, panel: FadePanel, ticker: str) -> None:
    """The assumed path, the fitted path, the band, and the sub-vertical beside it."""
    observation = model.latest[ticker]
    _rule(f"Growth path: assumed against fitted, {ticker}")

    t = Table(box=None, pad_edge=False)
    t.add_column("Year", no_wrap=True)
    t.add_column("Assumed", justify="right")
    t.add_column("Fitted", justify="right")
    t.add_column("Low", justify="right")
    t.add_column("High", justify="right")
    t.add_column("Sub-vertical", justify="right")
    t.add_column("All TMT", justify="right")
    t.add_column("Basis", no_wrap=True)

    path = comparison.fitted_path
    for i, (g, lo, hi, basis) in enumerate(
        zip(path.growth, path.lower, path.upper, path.basis)
    ):
        year = i + 1
        fit = model.fits.get(year)
        sub = (
            fit.sub_vertical_means.get(str(observation.sub_vertical))
            if fit is not None
            else None
        )
        pooled = fit.pooled_mean if fit is not None else None
        style = "" if basis == "fitted" else "dim"
        t.add_row(
            Text(f"Year {year}", style=style),
            Text(_pct(comparison.assumed_path[i]), style=style),
            Text(_pct(g), style="bold" if basis == "fitted" else "dim"),
            Text(_pct(lo), style=style),
            Text(_pct(hi), style=style),
            Text(_pct(sub), style=style),
            Text(_pct(pooled), style=style),
            Text(basis, style=style),
        )
    console.print(t)

    trailing = observation.growth
    console.print(
        f"\n  Trailing growth, as filed on {observation.as_of} for the fiscal year "
        f"ended {observation.fiscal_year_end}: [bold]{_pct(trailing)}[/bold]"
    )
    typical = _sub_vertical_fade(panel, str(observation.sub_vertical), 1)
    if typical is not None and trailing is not None:
        # Fade is a decline, so both figures are quoted as points lost. A signed
        # difference here reads as a growth rate and gets misread as one.
        own = trailing - path.growth[0]
        console.print(
            f"  The model fades it [bold]{own * 100:.1f} points[/bold] in the first "
            f"year. The average {str(observation.sub_vertical).replace('_', ' ')} "
            f"filer in this panel fades {typical['fade'] * 100:.1f} points, from "
            f"{typical['trailing']:.1%} to {typical['forward']:.1%} over "
            f"{int(typical['n']):,} company-years."
        )
        unusual = abs(own - typical["fade"])
        if unusual > 0.05:
            console.print(
                f"  [yellow]That is {unusual * 100:.1f} points away from the sub-vertical's "
                "typical fade, so this company is being treated as unusual. The "
                "features that did it are the trailing growth path, the size and "
                "the spend ratios, and they are in the panel rather than in a "
                "judgment.[/yellow]"
            )
        else:
            console.print(
                "  [dim]That is within five points of the sub-vertical's typical "
                "fade, so the model is not treating this company as unusual.[/dim]"
            )
    _notes(path.notes, heading="What the path is and is not")


def _render_valuation(comparison) -> None:
    """The four valuations, which is the only form of this model an analyst can argue with.

    Four rather than two. A point estimate printed beside the assumption it
    replaces invites the reader to believe the fitted number is the answer, and it
    is the middle of a band whose ends are different companies.
    """
    _rule(f"What the difference is worth: {comparison.ticker}")
    frame = comparison.to_frame()
    t = Table(box=None, pad_edge=False)
    t.add_column("Case", no_wrap=True)
    year_columns = [c for c in frame.columns if str(c).startswith("Y")]
    for c in year_columns:
        t.add_column(str(c), justify="right")
    for c in ("Terminal revenue", "Enterprise value", "Equity value", "Per share"):
        t.add_column(c, justify="right")

    for case, row in frame.iterrows():
        bold = case in ("Assumed fade", "Fitted fade")
        style = "bold" if bold else "dim"
        cells = [Text(_pct(row[c], 0), style=style) for c in year_columns]
        cells += [
            Text(_money(row["Terminal revenue"]), style=style),
            Text(_money(row["Enterprise value"]), style=style),
            Text(_money(row["Equity value"]), style=style),
            Text(f"{row['Per share']:,.2f}", style=style),
        ]
        t.add_row(Text(str(case), style=style), *cells)
    console.print(t)
    console.print("\n[dim]USD millions except per share.[/dim]")
    console.print(f"\n  {comparison.summary()}")


def _render_survivorship(model: FadeModel, panel: FadePanel) -> None:
    """The measured part of the bias, and the direction of the part that is left.

    On the full TMT panel the story is the one ``techval.ml.forecast`` tells. It
    keeps 119 delisted filers in until the day they stop filing, which removes
    most of the bias and is why the gaps below come out at tenths of a point. What
    no panel can hold is the years AFTER a filer left, and those are the years
    that would have faded hardest: the leavers' final observed year grows 9.8%
    against 15.8% for the filers still quoted, and then they are gone. The
    residual runs one way. The fitted curve fades too slowly, the growth path is
    too high, and every enterprise value struck on it is too large.

    That story is false for a panel with no leavers in it, and this was found on a
    live run rather than reasoned about. Fit the curve on a hand-picked list of
    companies that still trade and every gap in the table below comes out at
    exactly zero, because the comparison is between a sample and itself. Printing
    the full panel's paragraph over that would tell the reader the bias had been
    measured and found small, when what happened is that it was not measured at
    all and is large. So the count of leavers is taken from the panel in front of
    us and the paragraph follows the count.
    """
    survivorship = model.survivorship
    if not survivorship:
        return
    _rule("Survivorship: the direction this is wrong in")
    leavers = sorted(name for name in panel.tickers if is_delisted(name))
    t = Table(box=None, pad_edge=False)
    t.add_column("Horizon", no_wrap=True)
    t.add_column("Mean forward growth, whole panel", justify="right")
    t.add_column("Survivors only", justify="right")
    t.add_column("Gap", justify="right")
    t.add_column("Observations", justify="right")
    for horizon in sorted(model.fits):
        full = survivorship.get(f"full_{horizon}y")
        survivors = survivorship.get(f"survivors_{horizon}y")
        gap = survivorship.get(f"gap_{horizon}y")
        if full is None:
            continue
        t.add_row(
            f"{horizon} year",
            _pct(full),
            _pct(survivors),
            Text(_signed(gap, "+.2%"), style="yellow"),
            f"{int(survivorship.get(f'n_full_{horizon}y', 0)):,} / "
            f"{int(survivorship.get(f'n_survivors_{horizon}y', 0)):,}",
        )
    console.print(t)

    console.print(
        f"\n  {len(leavers):,} of the {len(panel.tickers):,} filers in this panel "
        "have since been acquired, taken private or wound up."
    )
    if not leavers:
        console.print(
            "\n  [yellow]FLAG: not one filer in this panel ever left the tape, so "
            "the gaps above are zero by construction. They are a comparison between "
            "a sample and itself and they measure nothing. This panel is a list of "
            "companies that survived, the fade curve fitted on it is fitted on "
            "survivors only, and the bias is not the tenths of a point the full TMT "
            "panel shows but whatever the missing failures would have been worth. "
            "Refit without --universe, or with a universe that includes the filers "
            "that stopped filing, before quoting any of this.[/yellow]"
        )
        return

    console.print(
        "  They are kept in the panel until the day they stop filing, and they are "
        "never a feature: a company is not delisted in 2014 because it was bought "
        "in 2024, and using the fact as an input would hand the model the future in "
        "its purest form."
    )
    final_leavers = survivorship.get("final_year_growth_leavers")
    final_survivors = survivorship.get("final_year_growth_survivors")
    if final_leavers is not None and final_survivors is not None:
        console.print(
            f"\n  In their last observed fiscal year the filers that later left the "
            f"tape grew [bold]{final_leavers:.1%}[/bold] against "
            f"[bold]{final_survivors:.1%}[/bold] for the filers still quoted. Then "
            "they stop filing."
        )
    console.print(
        "\n  [yellow]Residual bias, and its direction. Keeping the leavers in until "
        "the day they stop is most of the fix, and it is why the gaps above are "
        "small. What no panel can hold is the years after a filer left, and those "
        "are the years that would have faded hardest. So what is left runs one way: "
        "the fitted curve fades TOO SLOWLY, the growth path is TOO HIGH, and the "
        "enterprise value struck on it is TOO LARGE. Read the low end of the band as "
        "the more honest end.[/yellow]"
    )


def _render_assumptions(
    assumptions: Assumptions, extra: Sequence[tuple[str, str]], *, configured_years: int
) -> None:
    _rule("Assumptions in force")
    years = assumptions.dcf.projection_years
    rows = [
        (
            "dcf.projection_years",
            str(years)
            + ("" if years == configured_years else f" (flag, {configured_years} in the assumptions)"),
        ),
        ("dcf.revenue_growth_start", _pct(assumptions.dcf.revenue_growth_start)),
        ("dcf.revenue_growth_terminal", _pct(assumptions.dcf.revenue_growth_terminal)),
        ("ml.forecast.model", assumptions.ml.forecast.model),
        ("ml.forecast.horizon_years", str(assumptions.ml.forecast.horizon_years)),
        ("ml.random_seed", str(assumptions.ml.random_seed)),
        ("ml.walk_forward_folds", str(assumptions.ml.walk_forward_folds)),
        ("price_source", assumptions.price_source),
    ]
    rows.extend(extra)
    console.print(_kv(rows))


# --------------------------------------------------------------------------- #
# fade
# --------------------------------------------------------------------------- #


@app.command()
def fade(
    ticker: str,
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    as_of: str = _ASOF,
    panel_file: Path = typer.Option(
        None,
        "--panel",
        help=(
            "Recorded companyfacts blob to fit on, JSON or JSON.gz, instead of 224 "
            "live SEC calls. tests/fixtures/fade_companyfacts.json.gz is one."
        ),
    ),
    facts_file: Path = typer.Option(
        None,
        "--facts",
        help=(
            "Recorded companyfacts payload for TICKER, instead of a live SEC call. "
            "Used for the valuation half, not for the panel."
        ),
    ),
    universe: str = typer.Option(
        None,
        "--universe",
        help="Comma-separated tickers to fit the curve on, instead of the whole TMT panel.",
    ),
    years: int = typer.Option(
        None,
        "--years",
        help="Length of the growth path. Defaults to dcf.projection_years.",
    ),
    show_panel: bool = typer.Option(
        False,
        "--reversion-table",
        help="Print mean forward growth by decile of trailing growth, which assumes no functional form.",
    ),
) -> None:
    """Fit the revenue growth fade curve on filings and value TICKER on it.

    The output is a comparison, not a forecast. The assumed growth path from
    ``assumptions.dcf`` sits beside the fitted one, both are valued through the
    same discounted cash flow, and the two ends of the band the fit supports are
    valued too, because the point estimate is the middle of something forty growth
    points wide.

    ``ml.forecast.enabled`` gates whether a fitted path can reach an ordinary
    ``techval value`` run, and it defaults to false so that a base valuation is
    byte for byte what it was before the model existed. This command is the
    deliberate act that switch exists to require, so it turns it on for its own run
    and says so. Nothing else in the engine is affected.
    """
    try:
        assumptions, client, market, cache, _source = _setup(config, no_cache, as_of)
        symbol = ticker.upper()
        only = (
            {t.strip().upper() for t in universe.split(",") if t.strip()}
            if universe
            else None
        )
        if only is not None:
            only.add(symbol)

        if assumptions.as_of:
            console.print(
                f"[yellow]Point in time: fitting and valuing {symbol} as it was "
                f"knowable on {assumptions.as_of}.[/yellow]"
            )
        if not assumptions.ml.forecast.enabled:
            console.print(
                "[yellow]ml.forecast.enabled is false in the assumptions, so no "
                "other command in this engine is valuing on a fitted path. This "
                "command is what that switch exists to require, so it is on for "
                "this run only.[/yellow]"
            )
            assumptions.ml.forecast.enabled = True

        if panel_file is not None:
            console.print(f"[dim]Fitting on the recorded panel at {panel_file}.[/dim]")
            panel = _recorded_panel(panel_file, assumptions, only)
        else:
            console.print(
                "[dim]Fitting on live SEC filings. This is one companyfacts call "
                "per filer at the fair-access throttle.[/dim]"
            )
            panel = _live_panel(cache, assumptions, only)

        if assumptions.as_of:
            panel = _truncate_panel(panel, date.fromisoformat(assumptions.as_of))

        model = fit_fade(panel, assumptions)
        _render_curve(model, panel)
        _notes(panel.notes, heading="Panel")
        _horizon_table(model)

        if show_panel:
            _rule("Reversion table: no functional form at all")
            table = panel.reversion_table(1)
            t = Table(box=None, pad_edge=False)
            t.add_column("Decile", no_wrap=True)
            for column in table.columns:
                t.add_column(str(column), justify="right")
            for index, row in table.iterrows():
                t.add_row(
                    str(index),
                    f"{int(row['Observations']):,}",
                    _pct(row["Trailing growth"]),
                    _pct(row["Forward growth"]),
                    _pct(row["Forward median"]),
                    f"{row['Fade']:+.1%}",
                )
            console.print(t)
            console.print(
                "\n[dim]Sort every company-year by the growth it just reported, cut "
                "into ten equal buckets, read off what each did next. Both ends move "
                "toward the middle and nothing about that depends on a "
                "specification anybody could quarrel with.[/dim]"
            )

        # The valuation half. Facts from a recorded payload where one was given,
        # so the whole command can run with no network at all.
        if facts_file is not None:
            with _open_text(facts_file) as handle:
                payload = json.load(handle)
            knowledge = (
                date.fromisoformat(assumptions.as_of) if assumptions.as_of else None
            )
            fin = build_financials(
                symbol, facts=CompanyFacts(payload, symbol, knowledge_date=knowledge)
            )
        else:
            fin = build_financials(symbol, client=client)
        price = market.spot(symbol)
        bridge = build_ev_bridge(fin, price, assumptions)
        wacc_result = compute_wacc(fin, bridge, market, assumptions)

        configured_years = assumptions.dcf.projection_years
        if years is not None:
            if years < 1:
                raise ConfigError(f"a growth path needs at least one year, got {years}")
            assumptions.dcf.projection_years = years

        comparison = compare_fade(fin, bridge, wacc_result, assumptions, model, ticker=symbol)
        _render_paths(comparison, model, panel, symbol)
        _render_valuation(comparison)
        _render_survivorship(model, panel)

        _notes(model.notes, heading="Fit")
        _notes(model.card.limitations, heading="What this model cannot do")
        _render_assumptions(
            assumptions,
            [
                ("ml.forecast.enabled", "true for this run"),
                ("panel", str(panel_file) if panel_file else "live SEC filings"),
                ("wacc used by the DCF", _pct(comparison.assumed.wacc, 2)),
            ],
            configured_years=configured_years,
        )
    except TechvalError as exc:
        _fail(exc)


# --------------------------------------------------------------------------- #
# signal: loading a score panel
# --------------------------------------------------------------------------- #

# Score names this command recognises, with the value column each one arrives in
# and the sign convention that makes a positive coefficient mean the score was
# right. The sign is attached to the NAME rather than chosen after the
# coefficient is seen, which is the only way a one-sided reading of the answer is
# honest: picking the sign afterwards is a second test wearing a convention's
# clothes. Every one of them is printed on the run that uses it.
_KNOWN_SCORES: dict[str, dict[str, Any]] = {
    "cheapness": {
        "column": "ev_revenue",
        "negate": True,
        "label": "cheapness (negative trailing EV/Revenue)",
        "sign": (
            "the multiple is negated, so a high score is a cheap company and a "
            "positive coefficient means cheap beat expensive"
        ),
    },
    "warranted": {
        "column": "residual_log",
        "negate": True,
        "label": "cheapness against the warranted multiple (negative residual)",
        "sign": (
            "the residual is negated, so a high score is a company trading below "
            "the multiple its fundamentals warrant and a positive coefficient "
            "means the gap closed"
        ),
    },
    "propensity": {
        "column": "probability",
        "negate": False,
        "label": "acquisition propensity",
        "sign": (
            "the probability is used as it stands, so a high score is a likely "
            "target and a positive coefficient means likely targets earned more"
        ),
    },
    "raw": {
        "column": "value",
        "negate": False,
        "label": "signal",
        "sign": "the value is used exactly as the file states it",
    },
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with _open_text(path) as handle:
        return list(csv.DictReader(handle))


def _scores_from_csv(
    path: Path, column: str, negate: bool, limit_to: set[str] | None
) -> list[Score]:
    """Read a recorded score panel.

    Columns ``ticker`` and ``as_of`` are required and the value comes from
    ``column``. ``knowledge_date`` is set to the score's own date, which is the
    strongest claim a flat file can make: the harness then refuses the panel
    outright if any provenance it carries postdates it. A recorded panel that
    carries a ``max_filed`` column, as this repository's does, keeps that as the
    note so the point-in-time claim travels with the row.
    """
    rows = _read_rows(path)
    if not rows:
        raise ConfigError(f"{path} has no rows, so there is no score to test")
    header = set(rows[0])
    for needed in ("ticker", "as_of"):
        if needed not in header:
            raise ConfigError(
                f"{path} has no {needed!r} column. A score panel needs 'ticker', "
                "'as_of' and a value column; this file has "
                + ", ".join(sorted(header))
            )
    if column not in header:
        raise ConfigError(
            f"{path} has no {column!r} column. Pass --value-column to name the one "
            "that carries the score; this file has " + ", ".join(sorted(header))
        )
    out: list[Score] = []
    for row in rows:
        ticker = row["ticker"].upper()
        if limit_to and ticker not in limit_to:
            continue
        raw = row[column]
        if raw is None or raw == "":
            continue
        value = float(raw)
        when = date.fromisoformat(row["as_of"])
        out.append(
            Score(
                ticker=ticker,
                as_of=when,
                value=-value if negate else value,
                knowledge_date=when,
                note=row.get("max_filed") or None,
            )
        )
    if not out:
        raise ConfigError(
            f"{path} yielded no usable score. Every row was empty, or the "
            "--universe filter removed all of them."
        )
    return out


def _prices_from_csv(path: Path, limit_to: set[str] | None) -> dict[str, PriceSeries]:
    """Read a long-format close file: ticker, date, close.

    Long rather than one file per symbol, because a signal panel is a hundred
    symbols and ten years and a directory of a hundred CSVs is worse in every way
    than one gzipped file with a checksum beside it.
    """
    rows = _read_rows(path)
    if not rows:
        raise ConfigError(f"{path} has no rows, so there is no price history")
    header = set(rows[0])
    missing = {"ticker", "date", "close"} - header
    if missing:
        raise ConfigError(
            f"{path} is missing the column(s) {', '.join(sorted(missing))}. A price "
            "file is long format: ticker, date, close."
        )
    by_ticker: dict[str, list[tuple[date, float]]] = {}
    for row in rows:
        ticker = row["ticker"].upper()
        if limit_to and ticker not in limit_to:
            continue
        by_ticker.setdefault(ticker, []).append(
            (date.fromisoformat(row["date"]), float(row["close"]))
        )
    if not by_ticker:
        raise ConfigError(f"{path} carries no price series for the requested universe")
    out: dict[str, PriceSeries] = {}
    for ticker, pairs in by_ticker.items():
        pairs.sort()
        out[ticker] = PriceSeries(
            ticker,
            [d for d, _ in pairs],
            np.asarray([c for _, c in pairs], dtype=float),
            f"csv:{path.name}",
        )
    return out


def _rebalance_dates(first: date, last: date, months: int) -> list[date]:
    """Dates spaced ``months`` apart, inclusive of the first and no later than the last.

    Pinned to the day of month the caller gave rather than to a month end, so no
    rebalance lands on a quarter close and picks up the volume of an index
    reconstitution in its entry price.
    """
    out: list[date] = []
    year, month = first.year, first.month
    while True:
        try:
            when = date(year, month, first.day)
        except ValueError:  # a 31st in a 30 day month
            when = date(year, month, 28)
        if when > last:
            break
        out.append(when)
        month += months
        while month > 12:
            year, month = year + 1, month - 12
    return out


def _live_cheapness(
    tickers: Sequence[str],
    dates: Sequence[date],
    assumptions: Assumptions,
    cache: HttpCache,
    source: Any,
) -> tuple[list[Score], dict[str, PriceSeries], list[str]]:
    """Build the trailing EV/Revenue panel and the closes behind it, from live sources.

    One ``companyfacts`` call per filer rather than one per filer per date. The
    payload is fetched once and re-pinned with a different knowledge date for every
    rebalance, which is exactly what a point-in-time client does and saves a
    hundred-fold on the SEC's throttle. The pin is not a formality:
    ``ev_revenue_scores`` runs ``assert_no_lookahead`` on every row it builds and a
    payload left unpinned would fail it.

    Prices are fetched once per symbol over the whole span plus the horizon,
    because this is the one module in the package where a price series running past
    its row date is required rather than forbidden. The forward return is the label.
    """
    client = EdgarClient(cache)
    payloads: dict[str, dict] = {}
    notes: list[str] = []
    for ticker in tickers:
        try:
            payloads[ticker] = client.company_facts(ticker).raw
        except TechvalError as exc:
            notes.append(f"{ticker} served no companyfacts and is absent: {exc}")

    def facts_for(ticker: str, when: date) -> CompanyFacts | None:
        payload = payloads.get(ticker.upper())
        if payload is None:
            return None
        return CompanyFacts(payload, ticker, knowledge_date=when)

    span_start = min(dates) - timedelta(days=400)
    span_end = max(dates) + timedelta(days=int(400 + 31 * assumptions.ml.signals.horizon_months))
    prices: dict[str, PriceSeries] = {}
    for ticker in tickers:
        try:
            prices[ticker] = source.fetch(ticker, span_start, span_end)
        except TechvalError as exc:
            notes.append(f"{ticker} served no price history and is absent: {exc}")

    def price_at(ticker: str, when: date) -> float | None:
        series = prices.get(ticker.upper())
        if series is None:
            return None
        chosen = None
        for d, c in zip(series.dates, series.closes):
            if d <= when:
                chosen = float(c)
            else:
                break
        return chosen

    scores, score_notes = ev_revenue_scores(
        list(tickers),
        list(dates),
        facts_for=facts_for,
        price_at=price_at,
        assumptions=assumptions,
        cheap_is_high=True,
    )
    return scores, prices, notes + score_notes


# --------------------------------------------------------------------------- #
# signal: rendering
# --------------------------------------------------------------------------- #


def _render_headline(result: SignalResult) -> None:
    """The answer, in one styled block, including when the answer is no.

    ``SignalResult.verdict`` already prints NOT SIGNIFICANT in those words where
    the naive statistic clears two and the corrected one does not. This block
    repeats the finding as a single coloured line above the tables, because a
    verdict that arrives after seven tables has already lost the argument to the
    first impressive number in them.
    """
    ic = result.ic
    significant = result.significant
    naive_clears = abs(ic.t_naive) >= 1.96
    corrected_clears = abs(ic.t_newey_west) >= 1.96

    _rule(f"Verdict: {result.label}")
    if naive_clears and not corrected_clears:
        console.print(
            "  [red bold]NOT SIGNIFICANT.[/red bold] The naive t-statistic of "
            f"[bold]{_signed(ic.t_naive, '+.2f')}[/bold] clears two and would be "
            f"written up. The Newey-West figure of "
            f"[bold]{_signed(ic.t_newey_west, '+.2f')}[/bold] does not, and it is "
            "the one to quote."
        )
    elif significant:
        console.print(
            f"  [green bold]SIGNIFICANT[/green bold] after the overlap correction "
            f"and after {result.n_tests_run} test(s), at p = "
            f"{result.p_adjusted:.4f}. Newey-West t = "
            f"{_signed(ic.t_newey_west, '+.2f')}."
        )
    else:
        console.print(
            "  [red bold]NOT SIGNIFICANT.[/red bold] Newey-West t = "
            f"[bold]{_signed(ic.t_newey_west, '+.2f')}[/bold] at p = "
            f"{_signed(result.p_newey_west, '.3f')}, and the naive figure does not "
            "clear two either."
        )
    console.print(f"\n  [dim]{result.evaluation.verdict()}[/dim]")


def _render_counts(result: SignalResult) -> None:
    """Counts before coefficients, and the honest count before the flattering one.

    A hundred technology names in one quarter load on one factor and one rate
    cycle, so a single cross-section is closer to one observation than to a
    hundred, and twelve month returns sampled quarterly share three quarters of
    their path so a date is not a draw either. The number a standard error should
    reflect is the effective count, and it is printed first here for that reason.
    The 2,604 company-dates are printed last, where they belong.
    """
    ic = result.ic
    counts = result.exit_counts()
    _rule("How much data this actually is")
    console.print(
        _kv(
            [
                ("Effective observations, Newey-West", _num(ic.effective_n, 1)),
                (
                    f"Independent {result.horizon_months} month periods in the span",
                    _num(result.n_independent_periods, 1),
                ),
                ("Rebalance dates scored", f"{ic.n:,}"),
                ("Company-dates scored", f"{result.n_scored:,}"),
                ("Names per date, thinnest to widest", f"{min(ic.counts)} to {max(ic.counts)}"),
                ("Held to the horizon", f"{counts.get(_EXIT_HELD, 0):,}"),
                ("Terminated at a deal", f"{counts.get(_EXIT_ACQUIRED, 0):,}"),
                ("Delisted with no deal on record", f"{counts.get(_EXIT_DELISTED, 0):,}"),
                ("Censored at the edge of the price data", f"{counts.get(_EXIT_CENSORED, 0):,}"),
                ("No usable price", f"{counts.get(_EXIT_NO_PRICE, 0):,}"),
            ],
            bold=("Effective observations, Newey-West",),
        )
    )


def _render_ic(result: SignalResult, show_series: bool) -> None:
    """The coefficient statistics, both standard errors, and the series itself."""
    ic = result.ic
    _rule("Information coefficient")
    console.print(
        _kv(
            [
                ("Mean IC", _signed(ic.mean, "+.4f")),
                ("Standard deviation across dates", _num(ic.sd)),
                ("Share of dates positive", _pct(ic.share_positive, 0)),
                ("Naive standard error", _num(ic.naive_se)),
                ("Newey-West standard error", _num(ic.newey_west_se)),
                ("Standard error inflation", _signed(ic.inflation, ".2f") + "x"),
                ("t, naive", _signed(ic.t_naive, "+.2f")),
                ("t, Newey-West", _signed(ic.t_newey_west, "+.2f")),
                ("Newey-West lag", str(ic.lag)),
                (
                    "First-order autocorrelation of the series",
                    _signed(ic.autocorrelations[0], "+.2f") if ic.autocorrelations else "n/a",
                ),
                (
                    "Moving-block bootstrap standard error",
                    _num(ic.block_bootstrap_se) if ic.block_bootstrap_se else "n/a",
                ),
                ("p, Newey-West", _num(result.p_newey_west, 4)),
                ("p, Sidak-adjusted for tests run", _num(result.p_adjusted, 4)),
                (
                    "p, permutation within date",
                    _num(result.permutation_p, 4) if result.permutation_p is not None else "n/a",
                ),
            ],
            bold=("Mean IC", "t, Newey-West"),
        )
    )
    _render_overlap_note(result)

    if not show_series:
        return
    _rule("The coefficient series")
    t = Table(box=None, pad_edge=False)
    t.add_column("Date", no_wrap=True)
    t.add_column("Names", justify="right")
    t.add_column("IC", justify="right")
    t.add_column("", no_wrap=True)
    for when, names, value in zip(ic.dates, ic.counts, ic.values):
        # A rank correlation lives in [-1, 1], so twenty cells is the full scale
        # and nothing is clipped into looking like something it is not.
        bar = ("+" if value > 0 else "-") * min(20, int(round(abs(value) * 20)))
        t.add_row(
            str(when),
            f"{names:,}",
            f"{value:+.4f}",
            Text(bar, style="green" if value > 0 else "red"),
        )
    console.print(t)


def _render_overlap_note(result: SignalResult) -> None:
    """The correction, stated the way the module measured it.

    The usual claim is that overlapping windows inflate the t-statistic.
    ``techval.ml.signals`` tested that directly and it is not right on its own: a
    cross-sectional rank correlation lives inside one date, so a market move common
    to every name cancels out of it, and a score redrawn from noise at each
    rebalance produces a series with no autocorrelation worth correcting whatever
    the returns underneath are doing. What autocorrelates the series is a score
    that PERSISTS. Both halves are needed, and this note says which half the run in
    front of the reader actually had.
    """
    ic = result.ic
    if ic.lag == 0:
        console.print(
            "\n[dim]The rebalance spacing is at least the horizon, so the windows "
            "do not overlap and no correction is due. The two standard errors are "
            "the same number by construction.[/dim]"
        )
        return
    rho = ic.autocorrelations[0] if ic.autocorrelations else float("nan")
    console.print(
        "\n[bold]Why the two standard errors differ[/bold]\n"
        f"  [dim]Overlapping windows do not on their own autocorrelate a "
        f"cross-sectional coefficient: the correlation is computed inside one date, "
        f"so a market move common to every name cancels out of it. What survives "
        f"from one date to the next is the part of the ORDERING that persisted. "
        f"Overlap plus a persistent score is what inflates a t-statistic, and every "
        f"score in this package is persistent.[/dim]"
    )
    if np.isfinite(rho) and abs(rho) >= 0.3:
        console.print(
            f"  [dim]Measured here: first-order autocorrelation {rho:+.2f} at lag "
            f"{ic.lag}, so both halves are present and the correction bites. It "
            f"multiplied the standard error by {ic.inflation:.2f}x, leaving about "
            f"{ic.effective_n:.0f} effective observations out of {ic.n} dates.[/dim]"
        )
    elif np.isfinite(rho):
        console.print(
            f"  [dim]Measured here: first-order autocorrelation {rho:+.2f}, which is "
            "small. This score did not persist much across rebalances, so the "
            f"correction had little to correct and moved the standard error by "
            f"{ic.inflation:.2f}x.[/dim]"
        )
    if np.isfinite(ic.inflation) and ic.inflation < 1.0:
        console.print(
            "  [dim]The corrected standard error is SMALLER than the naive one. The "
            "coefficients alternate in sign, the autocovariances are negative, and "
            "the naive figure was the conservative one. That is a real result, not "
            "a failure of the correction.[/dim]"
        )


def _render_buckets(result: SignalResult) -> None:
    """The bucket table, which separates a factor from a screen.

    A signal monotone across the cross-section is a different animal from one that
    lives entirely in its extreme bucket. The second is a handful of names and a
    shorter life expectancy, and the monotonicity columns are what tell them apart.
    """
    buckets = result.buckets
    _rule("Returns by score bucket")
    frame = buckets.frame
    if frame.empty or not frame["Mean return"].notna().any():
        console.print(
            "[yellow]No bucket table was built. Every cross-section was too thin "
            "to fill the requested number of buckets, and a quantile of one name "
            "is a name rather than a quantile.[/yellow]"
        )
        _notes(buckets.notes)
        return
    t = Table(box=None, pad_edge=False)
    t.add_column("Bucket", no_wrap=True)
    t.add_column("Names per date", justify="right")
    t.add_column("Mean score", justify="right")
    t.add_column("Mean return", justify="right")
    t.add_column("Median of date means", justify="right")
    t.add_column("Share of dates positive", justify="right")
    for index, row in frame.iterrows():
        t.add_row(
            f"{index}" + (" (lowest score)" if index == frame.index[0] else
                          " (highest score)" if index == frame.index[-1] else ""),
            _num(row["Names per date"], 1),
            _num(row["Mean score"], 3),
            _pct(row["Mean return"]),
            _pct(row["Median of date means"]),
            _pct(row["Share of dates positive"], 0),
        )
    console.print(t)
    console.print(
        "\n[dim]Bucket one holds the lowest scores. Which end of the cross-section "
        "that is depends on the sign convention printed at the top of this run.[/dim]"
    )
    console.print()
    console.print(
        _kv(
            [
                (
                    f"Top less bottom, {result.horizon_months} months",
                    _signed(buckets.spread, "+.2%"),
                ),
                ("Spread t, Newey-West", _signed(buckets.spread_t, "+.2f")),
                (
                    "Monotonicity, bucket number against mean return",
                    _num(buckets.monotonicity, 2) if buckets.monotonicity is not None else "n/a",
                ),
                ("Share of adjacent steps that rise", _pct(buckets.monotone_steps, 0)),
                ("Turnover per rebalance, mean of the extremes", _pct(result.turnover.mean, 0)),
                ("Universe churn between rebalances", _pct(result.turnover.churn, 0)),
                (
                    "Break-even one-way cost",
                    f"{result.turnover.breakeven_cost_bps:.0f}bp"
                    if result.turnover.breakeven_cost_bps is not None
                    else "none: the spread is not positive before any cost at all",
                ),
            ],
            bold=(f"Top less bottom, {result.horizon_months} months",),
        )
    )
    _notes(buckets.notes)


def _render_signal_assumptions(
    assumptions: Assumptions,
    result: SignalResult,
    effective: dict[str, Any],
    extra: Sequence[tuple[str, str]],
) -> None:
    """Echo what actually drove the numbers, not what the file happens to say.

    ``buckets`` and ``min_names_per_date`` can be overridden on the command line,
    and printing the value out of the assumptions file where a flag overrode it
    would be worse than printing nothing: a reader checking why a run came out
    thin would read the wrong number and conclude the flag had not taken. Where a
    flag is in force the row says so.
    """
    _rule("Assumptions in force")

    def setting(label: str, value: Any, default: Any) -> tuple[str, str]:
        shown = str(value)
        if value != default:
            shown += f" (flag, {default} in the assumptions)"
        return label, shown

    signals_cfg = assumptions.ml.signals
    rows = [
        setting("ml.signals.horizon_months", result.horizon_months, signals_cfg.horizon_months),
        setting("ml.signals.buckets", effective["buckets"], signals_cfg.buckets),
        setting(
            "ml.signals.min_names_per_date",
            effective["min_names"],
            signals_cfg.min_names_per_date,
        ),
        ("ml.random_seed", str(assumptions.ml.random_seed)),
        ("tests run, for the Sidak adjustment", str(result.n_tests_run)),
        ("permutation draws behind the baseline", str(result.baseline_draws)),
    ]
    rows.extend(extra)
    console.print(_kv(rows))


# --------------------------------------------------------------------------- #
# signal
# --------------------------------------------------------------------------- #


@app.command()
def signal(
    config: Path = _CFG,
    no_cache: bool = _NOCACHE,
    score: str = typer.Option(
        "cheapness",
        "--score",
        help=(
            "Which score is being tested: cheapness, warranted, propensity or raw. "
            "The name fixes the value column and the sign convention, before the "
            "coefficient is seen rather than after it."
        ),
    ),
    scores_file: Path = typer.Option(
        None,
        "--scores",
        help=(
            "Recorded score panel, CSV or CSV.gz, columns ticker and as_of plus a "
            "value column. Without it, cheapness is built live from filings."
        ),
    ),
    prices_file: Path = typer.Option(
        None,
        "--prices",
        help="Long-format closes to score against: ticker, date, close. Required with --scores.",
    ),
    value_column: str = typer.Option(
        None, "--value-column", help="Column carrying the score. Defaults to the one --score implies."
    ),
    negate: bool = typer.Option(
        None,
        "--negate/--no-negate",
        help="Flip the sign of the value column. Defaults to what --score implies.",
    ),
    universe: str = typer.Option(
        None, "--universe", help="Comma-separated tickers to restrict the panel to."
    ),
    start: str = typer.Option(None, "--from", help="First rebalance date (YYYY-MM-DD), live builds."),
    end: str = typer.Option(None, "--to", help="Last rebalance date (YYYY-MM-DD), live builds."),
    every: int = typer.Option(3, "--every", help="Months between rebalances, live builds."),
    horizon: int = typer.Option(None, "--horizon", help="Forward return horizon in months."),
    buckets: int = typer.Option(None, "--buckets", help="Number of score buckets."),
    min_names: int = typer.Option(
        None, "--min-names", help="Scored names a date needs to enter the coefficient series."
    ),
    delisting: str = typer.Option(
        None,
        "--delisting",
        help=(
            "What a name that stopped trading with no deal on record earned: a "
            "number like -0.30, or one of total_loss, shumway_nyse, shumway_nasdaq. "
            "Omitted, such names are excluded and the count is flagged."
        ),
    ),
    tests_run: int = typer.Option(
        1,
        "--tests-run",
        help=(
            "How many tests were run behind the one being reported. It changes no "
            "coefficient and only the p-value, and it is yours to state honestly."
        ),
    ),
    draws: int = typer.Option(None, "--draws", help="Within-date permutations behind the baseline."),
    label: str = typer.Option(None, "--label", help="What to call the signal in the output."),
    show_series: bool = typer.Option(
        True, "--series/--no-series", help="Print the coefficient series date by date."
    ),
    csv_out: Path = typer.Option(
        None, "--csv", help="Write one row per attempted company-date, failures kept."
    ),
) -> None:
    """Test whether a score predicted forward returns. The answer is allowed to be no.

    The harness knows nothing about what produced the score, which is deliberate:
    a harness that imports the model it is judging can be tuned to flatter it. So a
    score arrives as ``(ticker, as_of, value)`` from a recorded file, or is built
    here for the one score in this package that depends on nothing fitted,
    trailing EV/Revenue.

    Reported: the information coefficient series, its mean with the Newey-West
    standard error beside the naive one, the share of dates positive, the bucket
    table with its monotonicity, turnover with the break-even cost that erases the
    spread, and the effective number of independent observations rather than the
    number of company-dates. Where the naive statistic clears two and the corrected
    one does not, the output says NOT SIGNIFICANT in those words.
    """
    try:
        assumptions, _client, _market, cache, source = _setup(config, no_cache)
        key = score.lower().strip()
        if key not in _KNOWN_SCORES:
            raise ConfigError(
                f"unknown score {score!r}. The names this command recognises are "
                + ", ".join(sorted(_KNOWN_SCORES))
                + ". Any other score is tested by writing it to a CSV and passing "
                "--scores with --value-column, which is how the warranted residual "
                "and the propensity score reach this harness: nothing here imports "
                "another model, so nothing here can be tuned to flatter one."
            )
        spec = _KNOWN_SCORES[key]
        column = value_column or spec["column"]
        flip = spec["negate"] if negate is None else negate
        title = label or spec["label"]
        only = (
            {t.strip().upper() for t in universe.split(",") if t.strip()}
            if universe
            else None
        )

        notes: list[str] = []
        if scores_file is not None:
            if prices_file is None:
                raise ConfigError(
                    "--scores needs --prices: a score panel with no price history "
                    "behind it has no forward return to be tested against."
                )
            parsed = _scores_from_csv(scores_file, column, flip, only)
            prices = _prices_from_csv(prices_file, None)
            notes.append(f"{len(parsed):,} scores read from {scores_file}.")
            notes.append(f"{len(prices):,} price series read from {prices_file}.")
        else:
            if key != "cheapness":
                raise ConfigError(
                    f"the {key} score cannot be built by this command. It is the "
                    "output of a fitted model, and this harness deliberately "
                    "imports no model it might be tuned to flatter. Write the score "
                    "to a CSV with columns ticker, as_of and "
                    f"{spec['column']!r}, then pass it with --scores."
                )
            if not (universe and start and end):
                raise ConfigError(
                    "a live cheapness run needs --universe, --from and --to. To "
                    "test a recorded panel instead, pass --scores and --prices."
                )
            dates = _rebalance_dates(
                date.fromisoformat(start), date.fromisoformat(end), every
            )
            if not dates:
                raise ConfigError("no rebalance date falls inside that window")
            console.print(
                f"[dim]Building trailing EV/Revenue for {len(only or [])} filers on "
                f"{len(dates)} dates from live filings and closes.[/dim]"
            )
            parsed, prices, build_notes = _live_cheapness(
                sorted(only), dates, assumptions, cache, source
            )
            notes.extend(build_notes)

        kwargs: dict[str, Any] = {"label": title, "n_tests_run": tests_run}
        if horizon is not None:
            kwargs["horizon_months"] = horizon
        if buckets is not None:
            kwargs["buckets"] = buckets
        if min_names is not None:
            kwargs["min_names_per_date"] = min_names
        if draws is not None:
            kwargs["baseline_draws"] = draws
        if delisting is not None:
            try:
                kwargs["delisting_return"] = float(delisting)
            except ValueError:
                kwargs["delisting_return"] = delisting

        result = signals_module.test_signal(parsed, prices, assumptions=assumptions, **kwargs)

        console.print(f"\n[dim]Sign convention: {spec['sign']}.[/dim]")
        _render_headline(result)
        _render_counts(result)
        _render_ic(result, show_series)
        _render_buckets(result)

        _notes(result.checks, heading="Cross-checks")
        _notes(list(result.notes) + notes)
        _notes(result.card.limitations, heading="What this test cannot do")
        _render_signal_assumptions(
            assumptions,
            result,
            {
                "buckets": kwargs.get("buckets", assumptions.ml.signals.buckets),
                "min_names": kwargs.get(
                    "min_names_per_date", assumptions.ml.signals.min_names_per_date
                ),
            },
            [
                ("score", key),
                ("value column", column),
                ("sign", "negated" if flip else "as stated"),
                ("delisting return", delisting or "excluded, and flagged"),
                ("scores", str(scores_file) if scores_file else "built live from filings"),
            ],
        )

        if csv_out is not None:
            result.frame().to_csv(csv_out, index=False)
            _rule("Output")
            console.print(f"Company-date detail written to [bold]{csv_out}[/bold]")
    except TechvalError as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
