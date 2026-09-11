"""Two commands that put the fitted models in front of a user: the learned comp
set, and the warranted-multiple screen.

    techval peers  TICKER    rank comparable companies with techval.ml.encoder
    techval screen           rank the universe by the residual from techval.ml.warranted

Both front models that already exist and neither one fits anything the modules
do not already know how to fit. What lives here is the part a command has to get
right that a library does not: deciding what a reader is entitled to see beside
a ranked list before the list means anything.

---

## Why both commands read a recorded artifact rather than the network

Every other command in this engine takes a ticker and goes to EDGAR. Neither of
these can, and the reason is not laziness.

``peers`` needs a fitted encoder, and an encoder is fitted on the compensation
peer groups of a universe of filers, the point-in-time feature panel behind them
and one business description per company per date. That is several thousand
filings. ``screen`` needs an observation panel, which is every company in the
universe priced at every quarter end over five years: ninety-odd names over
twenty-one dates, each one a full ``companyfacts`` parse and an enterprise value
bridge, and ``fit_warranted`` refuses below a hundred and fifty observations
precisely so that nobody runs it on a sample small enough to fetch live.

So both commands read a recorded input and both say so on every run. The inputs
are ordinary JSON, the recorders are committed beside the fixtures they wrote
(``tests/fixtures/warranted/record.py`` for the observation panel), and the
committed fixtures under ``tests/fixtures`` are real filings rather than
examples: the peer groups were read from DEF 14A proxies on 2026-09-11 and the
observation panel was priced against live SEC payloads the same day. Pointing
either command at those files is the fastest way to see what it prints.

The expensive half of ``peers`` is cached. Fitting the encoder on the committed
universe takes about eight seconds and the walk-forward evaluation behind the
cold-start number takes about a minute, so both are keyed on a digest of the
input files and the settings that change the answer, and written under
``assumptions.ml.cache_dir``. ``--refit`` throws the cache away. Nothing is
cached that a change to an input file would not invalidate.

---

## What ``peers`` refuses to let a reader miss

**Cold start.** The encoder scores materially worse on a company that had no
disclosed peer group in its training window than on one that did. On the
committed universe the gap is NDCG@10 of 0.37 cold against 0.63 warm, which is
not a footnote, it is nearly half the result. A ranked list looks identical in
both cases, so the command works out which one the target is, from the groups it
was fitted on, and says so in the header before the table. The numbers printed
beside it come from the model's own walk-forward evaluation and are never
hardcoded here.

**Which half of the model is working, and the two questions that get confused.**
The evaluation ranks the same queries with the encoder and with six baselines,
three of which are plain cosine in the encoder's own input spaces. That table is
printed in full, because it says how far the raw features get alone and how much
the contrastive fit added on top. It does not say which TRAINED tower carries the
ranking, and on the committed universe the two questions point different ways:
raw fundamentals cosine outranks raw text cosine, while refitting the model
without each tower in turn shows the text tower doing work outside the fold noise
and the fundamentals tower inside it. ``--ablate`` runs that refit and prints
both rows with the fold standard deviation each has to clear, so neither half of
the picture is available without the other.

**Label sources, which is where the cross-sector results come from.** Cold start
is a property of the target. The names that come out visibly wrong are on the
other side of the pair, and the measure that separates them is how many DIFFERENT
filers ever named the candidate. ``peers DIS`` ranks Procter and Gamble fifth,
above Netflix, and every labelled appearance P&G has in the committed fit comes
from Microsoft's proxy: one filer, no group of its own. ``peers NVDA`` ranks
Boston Scientific sixth, above Microsoft, on Analog Devices' proxy alone. A
candidate enters the universe by being named once, so the universe is always
wider than the supervision behind it, and the cross-sector name a compensation
committee reaches for is exactly the kind that enters on one mention. The count
is a column in the ranking and a single-source row gets a banner of its own.
Nothing is dropped: see ``_render_thin_evidence`` for why a bad row printed and
marked beats a bad row filtered out.

**The hand-written comp set, beside it.** ``assumptions.comps.peers`` is what a
human chose, and the disagreement between that list and the model's is the
interesting output. Agreement is not, particularly: if the target is warm, the
model was trained toward this very company's disclosed peer group, so it
agreeing with that group is a memory rather than a judgment, and the command
says which peers came from the target's own proxy for exactly that reason.

## What ``screen`` refuses to let a reader miss

**The 0.042.** The fitted model's pooled rank correlation on the committed panel
is 0.765 against 0.631 for the OLS the engine already had. Differenced against
each company's own previous observation it is 0.042. Nearly all of the pooled
number is company identity inherited from history rather than a view about what
changed, ``WarrantedModel`` carries that as a field rather than as a note, and
this command prints the two numbers in the same table. A screen that shows 0.765
and hides 0.042 is a lie by omission.

**Reflexivity.** The model is fitted on multiples, so it has learned the
market's own pricing rule. The most it can ever say is that a company is priced
unlike its characteristics suggest the market prices characteristics. It cannot
say the market is wrong: if the whole sector is mispriced the model is fitted on
the mispricing and reports everybody as fair. That is the difference between
this and a discounted cash flow, which has an outside anchor in the cash and the
discount rate. The banner is printed before the ranking rather than after it.

**The refusals.** A company-date the panel could not price is not absent, it is
refused, and the reason is a fact about the sample. ``share_basis_unresolved``
means the filed share count and the price series may be on different split
bases, so the equity value could be wrong by the split ratio and the row was
dropped rather than guessed at. Those rows are printed by name with their
reason. A screen quietly assembled from the names that happened to resolve is
the most common way a cross-sectional result gets quoted with more confidence
than it earned.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import Assumptions
from .errors import ConfigError, MissingDataError, TechvalError
from .ml.encoder import (
    EMBARGO_DAYS,
    MIN_TRAIN_PAIRS,
    PeerDataset,
    PeerEncoder,
    PeerEvaluation,
    ablate_towers,
    build_dataset,
    evaluate_peer_encoder,
    fit_peer_encoder,
    load_peer_encoder,
)
from .ml.evaluation import walk_forward_folds
from .ml.features import FeaturePanel, FeatureRow
from .ml.peer_labels import PeerGroup
from .ml.text import TextCorpus
from .ml.warranted import (
    TARGETS,
    Observation,
    ObservationPanel,
    Skip,
    WarrantedModel,
    fit_warranted,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Learned comparable companies, and the warranted-multiple screen.",
)

# Same reasoning as cli.py: rich falls back to eighty columns when stdout is not
# a terminal, and a redirected run would otherwise crush a nine-column table into
# three characters a column.
console = Console(width=None if sys.stdout.isatty() else 120)

# These four helpers are deliberately copies of the ones in cli.py rather than
# imports from it. cli.py mounts this module, so importing back into it would be
# a cycle, and four lines of formatting is a cheaper price than an import that
# only works in one direction.


def _rule(title: str) -> None:
    console.print()
    console.rule(Text(title, style="bold"), style="dim")


def _mult(v: float | None) -> str:
    return "NM" if v is None or not np.isfinite(v) else f"{v:,.1f}x"


def _num(v: float | None, dp: int = 3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:,.{dp}f}"


def _notes(items: list[str], *, heading: str = "Notes") -> None:
    if not items:
        return
    console.print(f"\n[bold]{heading}[/bold]")
    for n in items:
        style = "yellow" if n.upper().startswith(("FLAG", "WARN")) else "dim"
        console.print(Text(f"  - {n}", style=style))


def _echo_assumptions(rows: list[tuple[str, Any]]) -> None:
    """Every setting that moved a number on this page, printed under it.

    A ranked list and a rank correlation are both functions of settings the
    reader did not see chosen. The engine's convention is that they appear at the
    bottom of the run that used them rather than only in the file they came from.
    """
    _rule("Assumptions that drove these numbers")
    t = Table(box=None, pad_edge=False)
    t.add_column("", style="dim", no_wrap=True)
    # Folded rather than truncated: one of these rows is the path to the recorded
    # artifact every number above came from, and half a path is not provenance.
    t.add_column("", justify="right", style="dim", overflow="fold")
    for label, value in rows:
        t.add_row(label, str(value))
    console.print(t)


_CFG = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file.")


# --------------------------------------------------------------------------- #
# Reading the recorded inputs
# --------------------------------------------------------------------------- #


# The flag that supplies each artifact and the committed file that satisfies it,
# so a refusal can print the command line that works rather than the directory to
# go looking in. Nothing here is loaded by default: pointing the command at the
# test fixtures is a decision a reader makes, because a fixture recorded on one
# day is evidence about that day and substituting it for a missing input would be
# the engine inventing a universe.
_ARTIFACT_FLAGS = {
    "the disclosed peer groups": ("peers", "--groups", "tests/fixtures/peer_groups_tmt.json"),
    "the feature panel": ("peers", "--panel", "tests/fixtures/peer_panel_tmt.json"),
    "the business-description corpora": (
        "peers",
        "--text",
        "tests/fixtures/peer_item1_tmt.json",
    ),
    "the warranted-multiple observation panel": (
        "screen",
        "--panel",
        "tests/fixtures/warranted/observations.json.gz",
    ),
}


def _read_json(path: Path, *, what: str) -> Any:
    """Read a JSON document, gzipped or not, and name the file when it will not.

    ``.json.gz`` is accepted because the observation panel is two thousand rows
    of features and compresses fourteen to one. The suffix decides, so a file
    named ``.json`` that is really gzip is an error rather than a guess.

    The refusal names the flag and the committed file that satisfies it. Measured
    on a clean machine: ``peers DDOG`` with no flags fails, because
    ``~/.techval/ml/`` holds the joblib caches these commands write but none of
    the three inputs they read, and the old refusal pointed at ``tests/fixtures``
    without saying which of the files in it went with which of the three flags.
    A reader who has to guess that has been handed a puzzle rather than an
    instruction.
    """
    if not path.exists():
        command, flag, example = _ARTIFACT_FLAGS.get(what, ("", "", ""))
        fix = (
            f" Supply it with {flag}, for example: techval {command} "
            f"{'TICKER ' if command == 'peers' else ''}{flag} {example}."
            if flag
            else ""
        )
        raise MissingDataError(
            what,
            hint=(
                f"no file at {path}. These commands read a recorded artifact rather "
                "than the network, because fitting either model touches thousands of "
                "filings, and nothing is loaded by default: a fixture recorded on one "
                "day is evidence about that day." + fix + " The committed examples "
                "under tests/fixtures were read from real filings on 2026-09-11, and "
                "tests/fixtures/warranted/record.py is the script that wrote the "
                "observation panel"
            ),
        )
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as handle:
                return json.load(handle)
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, EOFError) as exc:
        raise ConfigError(f"{path} could not be read as {what}: {exc}") from None


def _need(payload: Any, key: str, path: Path) -> Any:
    if not isinstance(payload, dict) or key not in payload:
        raise ConfigError(
            f"{path} has no {key!r} key, so it is not the artifact this command "
            f"expects. See the module docstring for the shape of each input."
        )
    return payload[key]


def _iso(value: Any) -> date | None:
    return None if value in (None, "") else date.fromisoformat(str(value))


def _load_groups(path: Path) -> list[PeerGroup]:
    """Disclosed compensation peer groups, as ``peer_labels`` writes them.

    A group with no filing date or no fiscal year is dropped by ``build_dataset``
    rather than here, so the count of what the file held and the count of what
    could be used stay two different numbers.
    """
    payload = _read_json(path, what="the disclosed peer groups")
    raw = payload if isinstance(payload, list) else _need(payload, "groups", path)
    return [
        PeerGroup(
            ticker=str(g["ticker"]).upper(),
            cik=g.get("cik"),
            accession=g.get("accession"),
            filed=_iso(g.get("filed")),
            fiscal_year=g.get("fiscal_year"),
            peers=[str(p).upper() for p in g.get("peers", [])],
            unresolved=list(g.get("unresolved", [])),
            method=g.get("method", ""),
            selection_criteria=g.get("selection_criteria"),
            confidence=float(g.get("confidence", 0.0)),
        )
        for g in raw
    ]


def _load_panel(path: Path) -> FeaturePanel:
    """The point-in-time feature panel behind the fundamentals tower."""
    payload = _read_json(path, what="the feature panel")
    rows = _need(payload, "rows", path)
    return FeaturePanel(
        rows=[
            FeatureRow(
                ticker=str(r["ticker"]).upper(),
                as_of=date.fromisoformat(r["as_of"]),
                knowledge_date=_iso(r.get("knowledge_date")),
                values=r.get("values", {}),
                missing=list(r.get("missing", [])),
                statement_date=_iso(r.get("statement_date")),
                error=r.get("error"),
            )
            for r in rows
        ],
        random_seed=payload.get("random_seed"),
    )


def _load_corpora(path: Path) -> dict[date, TextCorpus]:
    """One business-description corpus per panel date, keyed by that date.

    ``as_of_by_ticker`` is the FILING date of the document rather than the panel
    date, which is what lets the encoder prove it fitted nothing it could not
    have read. A corpus missing it is refused by ``TextCorpus`` itself.
    """
    payload = _read_json(path, what="the business-description corpora")
    by_date = _need(payload, "by_date", path)
    names = payload.get("entity_names", {})
    out: dict[date, TextCorpus] = {}
    for iso, companies in by_date.items():
        tickers = sorted(companies)
        out[date.fromisoformat(iso)] = TextCorpus(
            tickers=tickers,
            documents=[companies[t]["item1"] for t in tickers],
            as_of_by_ticker={t: date.fromisoformat(companies[t]["filed"]) for t in tickers},
            source_accessions={t: companies[t].get("accession", "") for t in tickers},
            entity_names={t: names[t] for t in tickers if names.get(t)},
        )
    return out


def _load_observations(path: Path) -> ObservationPanel:
    """The recorded warranted-multiple panel, skips and all.

    The skips are read back with the observations and are not optional. They are
    the company-dates the engine refused to price, and a panel handed to
    ``fit_warranted`` without them still fits, but the screen printed from it
    could no longer say what it left out.
    """
    payload = _read_json(path, what="the warranted-multiple observation panel")
    rows = _need(payload, "observations", path)
    target = payload.get("target", "ev_revenue")
    if target not in TARGETS:
        raise ConfigError(
            f"{path} records target {target!r}, which is not a multiple "
            f"techval.ml.warranted fits; expected one of {', '.join(TARGETS)}"
        )
    observations: list[Observation] = []
    for r in rows:
        multiple = float(r["multiple"])
        if not np.isfinite(multiple) or multiple <= 0:
            raise ConfigError(
                f"{r['ticker']} on {r['as_of']} carries a multiple of {multiple}, "
                "which has no logarithm. The panel is fitted in logs and a "
                "non-positive multiple is a recording error rather than a cheap "
                "company"
            )
        observations.append(
            Observation(
                ticker=str(r["ticker"]).upper(),
                as_of=date.fromisoformat(r["as_of"]),
                sub_vertical=r["sub_vertical"],
                multiple=multiple,
                log_multiple=float(np.log(multiple)),
                enterprise_value=float(r["enterprise_value"]),
                equity_value=float(r["equity_value"]),
                denominator=float(r["denominator"]),
                statement_date=_iso(r.get("statement_date")),
                basis_factor=float(r.get("basis_factor", 1.0)),
                features=r.get("features", {}),
            )
        )
    skips = [
        Skip(
            ticker=str(s["ticker"]).upper(),
            as_of=date.fromisoformat(s["as_of"]),
            reason=s.get("reason", ""),
            category=s.get("category", "unknown"),
        )
        for s in payload.get("skips", [])
    ]
    return ObservationPanel(
        target=target,
        observations=observations,
        skips=skips,
        notes=list(payload.get("notes", [])),
        random_seed=payload.get("random_seed"),
    )


# --------------------------------------------------------------------------- #
# The fitted bundle, and its cache
# --------------------------------------------------------------------------- #


@dataclass
class _Bundle:
    """A fitted encoder, the sample it came from, and its walk-forward score."""

    dataset: PeerDataset
    encoder: PeerEncoder
    evaluation: PeerEvaluation | None
    ablation: pd.DataFrame | None
    cached_fit: bool
    cached_evaluation: bool
    cached_ablation: bool


def _cache_root(assumptions: Assumptions) -> Path:
    configured = assumptions.ml.cache_dir
    return Path(configured).expanduser() if configured else Path.home() / ".techval" / "ml"


def _digest(paths: list[Path], settings: dict[str, Any]) -> str:
    """A key over the inputs and over every setting that moves the fit.

    File contents rather than modification times, because a recorded artifact
    that is copied or regenerated byte for byte is the same evidence and should
    hit the same cache entry, and one that changed by a single feature should
    miss it.
    """
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _tower_ablation(
    dataset: PeerDataset, assumptions: Assumptions, n_folds: int
) -> pd.DataFrame:
    """Refit with each tower removed in turn, on folds this function supplies.

    ``ablate_towers`` takes a ``folds`` argument and it is passed here rather
    than left to default, for a reason that is a property of the sample rather
    than a preference. Cut into equal blocks of dates with no floor on the
    initial training window, the earliest walk-forward fold on the committed
    universe holds nine disclosed pairs, because proxies arrive in a season and
    the first season inside the window is one filer. ``_train_encoder`` then
    refuses, correctly, to fit a contrastive model on nine relationships, and the
    refusal takes the whole ablation with it rather than costing one fold.
    Reserving an initial window that holds at least
    ``MIN_TRAIN_PAIRS`` disclosed pairs is what ``walk_forward_folds(min_train=)``
    exists for, and the boundaries it returns are dates, so they apply unchanged
    to the expanded frame of positives and sampled negatives the ablation scores.
    """
    folds = walk_forward_folds(
        [e.filed for e in dataset.examples],
        n_folds,
        min_train=MIN_TRAIN_PAIRS,
        embargo_days=EMBARGO_DAYS,
    )
    return ablate_towers(dataset, assumptions, folds=folds)


def _bundle(
    assumptions: Assumptions,
    *,
    groups_path: Path,
    panel_path: Path,
    text_path: Path,
    through: date | None,
    k: int,
    evaluate: bool,
    ablate: bool,
    refit: bool,
    use_cache: bool,
) -> _Bundle:
    """Build the dataset, fit the encoder, and score it, reusing what it can.

    The fit and the evaluation are cached separately because they cost an order
    of magnitude apart and a reader who wants a quick ranking should not pay for
    a walk-forward evaluation they already have on disk.

    ``evaluate`` defaults on at the command because the cold-start split is not
    optional context for a ranked list. Turning it off is allowed and the command
    then says the split is unavailable rather than quietly printing the warm
    number, which is the flattering half.
    """
    dataset = build_dataset(
        _load_groups(groups_path), _load_panel(panel_path), _load_corpora(text_path)
    )

    root = _cache_root(assumptions)
    key = _digest(
        [groups_path, panel_path, text_path],
        {
            "text_weight": assumptions.ml.peers.text_weight,
            "seed": assumptions.ml.random_seed,
            "folds": assumptions.ml.walk_forward_folds,
            "through": through,
            "k": k,
        },
    )
    fit_file = root / f"peer_encoder_{key}.joblib"
    eval_file = root / f"peer_evaluation_{key}.joblib"
    ablation_file = root / f"peer_ablation_{key}.joblib"

    encoder: PeerEncoder | None = None
    cached_fit = False
    if use_cache and not refit and fit_file.exists():
        try:
            encoder = load_peer_encoder(fit_file)
            cached_fit = True
        except (TechvalError, OSError, ValueError):
            # A cache entry this version cannot read is a cache miss, never a
            # failed run. load_peer_encoder already refuses a foreign schema by
            # name; anything else unreadable is treated the same way.
            encoder = None

    evaluation: PeerEvaluation | None = None
    cached_evaluation = False
    if evaluate and use_cache and not refit and eval_file.exists():
        try:
            import joblib

            evaluation = joblib.load(eval_file)
            cached_evaluation = True
        except Exception:  # noqa: BLE001 - an unreadable cache is a miss
            evaluation = None

    if evaluate and evaluation is None:
        evaluation = evaluate_peer_encoder(dataset, assumptions, k=k)
        if use_cache:
            _write_cache(eval_file, evaluation)

    ablation: pd.DataFrame | None = None
    cached_ablation = False
    if ablate and use_cache and not refit and ablation_file.exists():
        try:
            import joblib

            ablation = joblib.load(ablation_file)
            cached_ablation = True
        except Exception:  # noqa: BLE001 - an unreadable cache is a miss
            ablation = None
    if ablate and ablation is None:
        ablation = _tower_ablation(
            dataset, assumptions, int(assumptions.ml.walk_forward_folds)
        )
        if use_cache:
            _write_cache(ablation_file, ablation)

    if encoder is None:
        encoder = fit_peer_encoder(
            dataset,
            assumptions,
            fit_through=through,
            # The whole evaluation rather than its headline, so the warm and
            # cold halves are saved with the fit and survive to whoever loads
            # the joblib next. See PeerEncoder.cold_start_note.
            evaluation=evaluation,
        )
        if use_cache:
            try:
                encoder.save(fit_file)
            except OSError:
                pass
    elif evaluation is not None and encoder.card.evaluation is None:
        # A fit cached before the evaluation existed still deserves its score.
        encoder.card.evaluation = evaluation.headline

    return _Bundle(
        dataset,
        encoder,
        evaluation,
        ablation,
        cached_fit,
        cached_evaluation,
        cached_ablation,
    )


def _write_cache(path: Path, payload: Any) -> None:
    try:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path, compress=3)
    except Exception:  # noqa: BLE001 - an unwritable cache slows the next run only
        pass


# --------------------------------------------------------------------------- #
# peers
# --------------------------------------------------------------------------- #


def _training_cut(dataset: PeerDataset, through: date | None) -> date:
    """The last FILING date the fit was allowed to read, which is not the fit date.

    ``fit_peer_encoder`` selects its training pairs on the date a proxy was
    filed and then embeds the universe at the newest PANEL date, so the two dates
    on a fitted model mean different things and are usually months apart. Warm
    and cold are properties of the training window, so the filing cut is the one
    to compare a group against. Using the embedding date instead would call a
    company cold whose proxy landed between the two, which is the only kind of
    company where getting this wrong would matter.
    """
    if through is not None:
        return through
    return max(e.filed for e in dataset.examples)


def _trained_filers(dataset: PeerDataset, through: date) -> dict[str, list[PeerGroup]]:
    """Every company that was a QUERY in the training window, with its groups.

    Being named as somebody else's peer does not make a company warm. The split
    the evaluation reports is about the query: a target the model has seen
    ranked before against one it has not, and only the filer of a group was ever
    a query.
    """
    out: dict[str, list[PeerGroup]] = {}
    for g in dataset.groups:
        if g.filed is not None and g.filed <= through:
            out.setdefault(g.ticker.upper(), []).append(g)
    return out


# How many DIFFERENT filers have to have asserted something about a name before
# its position in a ranking is treated as a result rather than as extrapolation.
# Two, because one committee is not a distribution: a name every one of whose
# labelled appearances comes from a single proxy has an embedding driven by its
# features and its prose, with the contrastive loss having pulled on it from one
# direction only.
MIN_LABEL_SOURCES = 2


def _label_sources(dataset: PeerDataset, through: date) -> dict[str, set[str]]:
    """For every company, the set of filers whose disclosed groups mention it.

    A company's own group counts as one source, so a filer that disclosed a group
    and was never named by anybody else has exactly one and is flagged, which is
    the right answer: nothing outside its own compensation committee has said
    where it sits.

    Why this and not the cold-start flag the command already prints. Warm and
    cold are properties of the TARGET and they were measured at the right place:
    ``peers DIS`` reports Disney WARM, because Disney disclosed two groups inside
    the training window. The names that come out wrong are on the other side of
    the pair. Measured on the committed universe of 173 companies and 220
    disclosed groups:

        PROCTER & GAMBLE ranks 5th for Disney at 0.9161, above NETFLIX at 0.9076.
        Every labelled appearance P&G has in this fit comes from Microsoft's
        proxy. One filer, four groups, no group of its own.

        BOSTON SCIENTIFIC ranks 6th for Nvidia at 0.7601, above MICROSOFT at
        0.7532. Every labelled appearance it has comes from Analog Devices'
        proxy. One filer, six groups, no group of its own.

    Those were the two cross-sector results the wave's verifiers found, and both
    are single-source names. 34 of the 173 candidates are in that state and the
    median name has four sources, so this is a real minority rather than a
    description of the whole universe. The measure needs no sector taxonomy,
    which matters: a sector label would be this command's opinion, and the
    question a comp set has to answer is what the labels support.
    """
    out: dict[str, set[str]] = {}
    for g in dataset.groups:
        if g.filed is None or g.filed > through or not g.usable:
            continue
        filer = g.ticker.upper()
        out.setdefault(filer, set()).add(filer)
        for peer in g.peers:
            out.setdefault(peer.upper(), set()).add(filer)
    return out


def _render_start_banner(
    ticker: str,
    groups: list[PeerGroup],
    evaluation: PeerEvaluation | None,
    sources: set[str],
) -> None:
    """Warm or cold, said before the table rather than in a footnote.

    The two scores beside it come from the model's own walk-forward evaluation
    and are printed only when that evaluation was run. Neither is hardcoded here
    and neither is a claim this command makes on its own.

    ``sources`` is the second, orthogonal question: warm or cold is about whether
    the target ever appeared as a QUERY, and this is about how many different
    filers said anything about it at all. A cold target named by fifteen proxies
    is a name the model has seen from fifteen angles without ever being asked to
    rank it, and a warm target named by nobody else is a company whose only
    supervision is its own compensation committee. They come apart often enough
    that printing one without the other is a half answer.
    """
    warm = bool(groups)
    filed = max(g.filed for g in groups) if warm else None
    scores = ""
    if evaluation is not None and evaluation.warm is not None and evaluation.cold is not None:
        scores = (
            f" On this fit the encoder scores NDCG@{evaluation.k} of "
            f"{evaluation.cold.score:.4f} on the {evaluation.cold.n_observations} cold "
            f"queries against {evaluation.warm.score:.4f} on the "
            f"{evaluation.warm.n_observations} warm ones."
        )
    if warm:
        console.print(
            Text(
                f"WARM START. {ticker} disclosed {len(groups)} peer group(s) inside the "
                f"training window, the most recent filed {filed}, so the model was "
                "fitted toward this company's own answer. Where the ranking agrees "
                "with that group it is partly recall rather than judgment." + scores,
                style="yellow",
            )
        )
    else:
        console.print(
            Text(
                f"COLD START. {ticker} disclosed no peer group the model was fitted "
                "on, so this ranking comes from its features and its prose alone. "
                "This is the harder case and the model is materially worse at it."
                + scores,
                style="red bold",
            )
        )
    n = len(sources)
    console.print(
        Text(
            f"{n} distinct filer(s) named {ticker} in a usable disclosed group "
            "inside the training window, counting its own. That is how much the "
            "labels constrain where this company sits, and it is a different "
            "question from warm or cold: only the filer of a group was ever a "
            "query, while anybody can name anybody.",
            style="red bold" if n < MIN_LABEL_SOURCES else "dim",
        )
    )


def _render_ranking(
    ticker: str,
    ranked: list[tuple[str, float]],
    hand: list[str],
    disclosed: set[str],
    names: dict[str, str],
    sources: dict[str, set[str]],
) -> None:
    t = Table(box=None, pad_edge=False)
    t.add_column("#", justify="right")
    t.add_column("Ticker", no_wrap=True)
    t.add_column("Company", overflow="ellipsis", max_width=34)
    t.add_column("Similarity", justify="right")
    t.add_column("Label sources", justify="right")
    t.add_column("In the hand-written set", justify="center")
    t.add_column(f"In {ticker}'s own proxy", justify="center")
    hand_set = {h.upper() for h in hand}
    for i, (peer, similarity) in enumerate(ranked, 1):
        n = len(sources.get(peer, ()))
        thin = n < MIN_LABEL_SOURCES
        style = "red" if thin else ""
        t.add_row(
            str(i),
            Text(peer, style="red bold" if thin else "bold"),
            Text(names.get(peer, ""), style=style),
            Text(f"{similarity:.4f}", style=style),
            Text(str(n), style="red bold" if thin else ""),
            "yes" if peer in hand_set else "-",
            "yes" if peer in disclosed else "-",
        )
    console.print(t)


def _render_thin_evidence(
    ticker: str,
    ranked: list[tuple[str, float]],
    sources: dict[str, set[str]],
    universe: list[str],
    names: dict[str, str],
) -> None:
    """The rows the labels barely constrain, named, kept and argued over.

    Nothing is dropped and nothing is capped. A ranking that quietly excluded its
    embarrassing rows would be a worse object than one that prints them: the
    reader would have no way to see that the model puts a consumer staples
    company fifth for a media conglomerate, which is the single most useful thing
    this output can tell them about how far the model can be trusted here.

    The argument for refusing the whole ranking instead was considered and
    rejected. On Disney seven of the eight names are Comcast, Verizon, AT&T,
    Charter, T-Mobile, Netflix and Microsoft, which is a defensible media and
    telecom comp set and is what the reader came for. Refusing all eight because
    one of them rests on one proxy throws away a right answer to avoid printing a
    wrong row that is now labelled as one. What IS refused is the silent version:
    the row keeps its rank and carries the count that earned the warning.
    """
    thin = [(p, s) for p, s in ranked if len(sources.get(p, ())) < MIN_LABEL_SOURCES]
    if not thin:
        return
    total_thin = sum(1 for t in universe if len(sources.get(t, ())) < MIN_LABEL_SOURCES)
    console.print()
    for peer, similarity in thin:
        who = sorted(sources.get(peer, ()))
        rank = next(i for i, (p, _s) in enumerate(ranked, 1) if p == peer)
        origin = (
            f"every labelled appearance it has comes from one filer's proxy ({who[0]})"
            if who
            else "no disclosed group in the training window mentions it at all"
        )
        console.print(
            Text(
                f"OUT OF ITS DEPTH. {peer} ({names.get(peer, 'name unknown')}) is "
                f"ranked {rank} of {len(ranked)} for {ticker} at {similarity:.4f}, and "
                f"{origin}. The contrastive loss pulled on this name from one "
                "direction, so its position here is driven by its features and its "
                "prose rather than by anything a committee asserted about it. The "
                "row is printed at its rank rather than dropped, because a ranking "
                "that hides the names it cannot support is worse than one that "
                "marks them.",
                style="red bold",
            )
        )
    console.print(
        f"\n[dim]{total_thin} of the {len(universe)} companies in the candidate "
        f"universe are named by fewer than {MIN_LABEL_SOURCES} distinct filers. A "
        "candidate enters the universe by being named once, so the universe is "
        "wider than the supervision behind it, and the cross-sector names a "
        "compensation committee reaches for are exactly the ones that enter on a "
        "single mention.[/dim]"
    )


def _render_hand_set(
    encoder: PeerEncoder, ticker: str, hand: list[str], shown: int
) -> None:
    """Where the model put every name a human already chose.

    The rank is taken over the WHOLE fitted universe with no size gate, because
    the question is where the model would have placed this company, not whether
    it survived a filter the hand-written list never passed through.
    """
    if not hand:
        console.print(
            "\n[dim]assumptions.comps.peers is empty, so there is no hand-written "
            "comp set to disagree with. Set it to see the comparison this command "
            "exists for.[/dim]"
        )
        return
    order = encoder.neighbours(
        ticker, max(1, len(encoder.tickers) - 1), apply_size_gate=False
    )
    rank_of = {peer: i for i, (peer, _s) in enumerate(order, 1)}
    similarity_of = dict(order)

    t = Table(box=None, pad_edge=False)
    t.add_column("Ticker", no_wrap=True)
    t.add_column("Model rank", justify="right")
    t.add_column("Similarity", justify="right")
    t.add_column("", overflow="fold")
    for peer in [h.upper() for h in hand]:
        if peer == ticker:
            t.add_row(peer, "-", "-", "the target itself")
            continue
        rank = rank_of.get(peer)
        if rank is None:
            t.add_row(
                peer,
                "-",
                "-",
                "not in the fitted universe: no feature row and business "
                "description at the fit date",
            )
            continue
        note = "" if rank <= shown else f"outside the model's top {shown}"
        t.add_row(peer, f"{rank}", f"{similarity_of[peer]:.4f}", note)
    console.print(t)


def _render_evaluation(evaluation: PeerEvaluation) -> None:
    """The walk-forward score, the cold split, and every baseline on one table."""
    _rule(f"What the model scored, walk-forward, NDCG@{evaluation.k}")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    t.add_column("", justify="right", style="dim")
    rows: list[tuple[str, str, str]] = [
        ("Queries scored", f"{evaluation.n_queries:,}", f"{len(evaluation.folds)} folds"),
        (
            "Encoder",
            f"{evaluation.headline.score:.4f}",
            f"fold sd {_num(evaluation.headline.fold_sd, 4)}",
        ),
        (
            "Popularity prior (query ignored)",
            f"{evaluation.headline.baseline_score:.4f}",
            "beaten" if evaluation.headline.beat_baseline else "NOT beaten",
        ),
    ]
    if evaluation.warm is not None:
        rows.append(
            (
                "Warm start (target seen in training)",
                f"{evaluation.warm.score:.4f}",
                f"{evaluation.warm.n_observations:,} queries",
            )
        )
    if evaluation.cold is not None:
        rows.append(
            (
                "Cold start (target never seen)",
                f"{evaluation.cold.score:.4f}",
                f"{evaluation.cold.n_observations:,} queries",
            )
        )
    rows.append(
        (
            "Correlation with the popularity order",
            f"{evaluation.collapse:.3f}",
            "1.0 would mean one list for every query",
        )
    )
    ceiling = evaluation.coverage.get("in_universe")
    if ceiling is not None and np.isfinite(ceiling):
        rows.append(
            ("Recall ceiling from universe coverage", f"{ceiling:.1%}", "a property of the universe")
        )
    for label, value, note in rows:
        bold = label.startswith(("Encoder", "Cold start"))
        t.add_row(
            Text(label, style="bold" if bold else ""),
            Text(value, style="bold" if bold else ""),
            note,
        )
    console.print(t)
    console.print(f"\n[dim]{evaluation.headline.verdict()}[/dim]")

    _rule("Every ranking method on the same queries")
    console.print(
        "[dim]Three of these are plain cosine in the encoder's own input spaces, "
        "with no learning on top, so the table says how far the raw features get "
        "on their own and how much the contrastive fit added. It is NOT the tower "
        "question: which of the two TRAINED towers is carrying the ranking is what "
        "--ablate answers by refitting without each one, and on this universe the "
        "two questions do not give the same answer.[/dim]\n"
    )
    frame = evaluation.scores
    t = Table(box=None, pad_edge=False)
    for column in frame.columns:
        t.add_column(str(column), justify="left" if column == "method" else "right")
    for _i, row in frame.iterrows():
        style = "bold" if row["method"] == "encoder" else ""
        cells = [
            Text(str(v) if isinstance(v, str) else _num(float(v), 4), style=style)
            for v in row
        ]
        t.add_row(*cells)
    console.print(t)


def _render_ablation(frame: pd.DataFrame) -> None:
    """What each tower's absence costs, against the fold noise it has to beat.

    ``damage`` is signed so positive means the tower was doing work, and the
    comparison that decides whether it was doing work is against ``fold_sd``
    beside it rather than against zero. A tower whose damage is smaller than the
    fold-to-fold standard deviation has not been shown to help, and printing the
    verdict in that column is the whole point of running this: a model card
    listing two towers invites a reader to assume both are earning their place.
    """
    _rule("What each tower is worth, refitted without it")
    t = Table(box=None, pad_edge=False)
    t.add_column("Configuration", no_wrap=True)
    t.add_column("Columns", justify="right")
    t.add_column("Spearman", justify="right")
    t.add_column("Damage", justify="right")
    t.add_column("Fold sd", justify="right")
    t.add_column("Verdict", overflow="fold")
    for _i, row in frame.iterrows():
        full = row["group"] == "all features"
        damage, sd = float(row["damage"]), float(row["fold_sd"])
        # The test is the damage against the fold-to-fold standard deviation
        # beside it, not against zero, and the ratio is printed rather than
        # collapsed into a yes so a reader can tell 1.01 from 3.0.
        ratio = damage / sd if np.isfinite(sd) and sd > 0 else float("nan")
        if full:
            verdict = "the model with both towers"
        elif damage <= 0:
            verdict = "removing it did not hurt: this tower is not helping"
        elif np.isfinite(ratio) and ratio < 1:
            verdict = f"damage is {ratio:.2f}x the fold noise: NOT shown to help"
        else:
            verdict = f"damage is {ratio:.2f}x the fold noise: shown to help"
        style = "" if full else ("dim" if np.isfinite(ratio) and ratio < 1 else "bold")
        t.add_row(
            Text(str(row["group"]), style=style),
            f"{int(row['n_columns']):,}",
            _num(float(row["score"]), 4),
            _num(damage, 4),
            _num(sd, 4),
            Text(verdict, style=style),
        )
    console.print(t)
    console.print(
        "\n[dim]A row here is a (target, candidate, date) pair, the outcome is 1 for "
        "a candidate the target's proxy named and 0 for one it did not, and the "
        "prediction is the cosine the model gives that pair. Spearman between the "
        "two is a rank measure of the same thing the NDCG above measures, and unlike "
        "NDCG it is defined per row, which is what refitting one group at a time "
        "needs.[/dim]"
    )


@app.command()
def peers(
    ticker: str,
    config: Path = _CFG,
    groups: Path = typer.Option(
        None, "--groups", help="Recorded DEF 14A peer groups (JSON)."
    ),
    panel: Path = typer.Option(
        None, "--panel", help="Recorded point-in-time feature panel (JSON)."
    ),
    text: Path = typer.Option(
        None, "--text", help="Recorded business descriptions by date (JSON)."
    ),
    n: int = typer.Option(None, "--n", help="Peers to show. Default ml.peers.n_peers."),
    k: int = typer.Option(10, "--k", help="Rank cut-off the evaluation reports."),
    through: str = typer.Option(
        None,
        "--through",
        "--as-of",
        help=(
            "Fit on pairs disclosed on or before this date (YYYY-MM-DD). This is "
            "what --as-of means here, and it pins less than --as-of pins on the "
            "commands that read EDGAR: see the note printed under the ranking."
        ),
    ),
    size_gate: bool = typer.Option(
        True, "--size-gate/--no-size-gate", help="Apply ml.peers size bands to candidates."
    ),
    evaluate: bool = typer.Option(
        True,
        "--evaluate/--no-evaluate",
        help="Run the walk-forward evaluation behind the cold-start split.",
    ),
    ablate: bool = typer.Option(
        False,
        "--ablate/--no-ablate",
        help="Refit without each tower to say which one carries the ranking.",
    ),
    refit: bool = typer.Option(
        False, "--refit", help="Ignore the cache and recompute fit, evaluation and ablation."
    ),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Use the model cache."),
) -> None:
    """Rank comparable companies with the learned encoder, beside the hand-written set.

    The encoder is two towers trained contrastively on peer groups companies
    disclosed in their own DEF 14A proxies, so no pair in its training set is
    anybody's opinion about comparability: each one is a dated assertion a
    compensation committee signed. What it returns is a cosine, and the cosine is
    the blend ``(1 - text_weight) * fundamentals + text_weight * text`` because
    that blend is inside the architecture rather than applied afterwards.

    Four things are printed beside the ranking and none of them is optional.
    Whether the target was warm or cold in the training window, since the model
    is much weaker cold and a ranked list looks the same either way. How many
    different filers ever named each name in the list, since a candidate resting
    on one proxy is where this model leaves the distribution it was fitted on and
    the ranked list looks the same there too. Where the hand-written comp set
    landed, since the disagreement is the output worth reading. And every
    baseline on the same queries, since the popularity prior is a degenerate
    solution that scores respectably by naming the same famous companies for
    every query, and a model that beat it by imitating it would otherwise be
    invisible.

    ``--ablate`` adds the fourth, at the cost of about a minute: the model
    refitted without each tower in turn, with the fold-to-fold standard deviation
    each tower's damage has to clear before it counts as help. It is off by
    default only because of what it costs, and the run that leaves it off says so
    rather than letting the baseline table stand in for it.
    """
    try:
        assumptions = Assumptions.load(config)
        root = _cache_root(assumptions)
        groups_path = Path(groups) if groups else root / "peer_groups.json"
        panel_path = Path(panel) if panel else root / "peer_panel.json"
        text_path = Path(text) if text else root / "peer_item1.json"
        cut = date.fromisoformat(through) if through else None
        target = ticker.upper()

        bundle = _bundle(
            assumptions,
            groups_path=groups_path,
            panel_path=panel_path,
            text_path=text_path,
            through=cut,
            k=k,
            evaluate=evaluate,
            ablate=ablate,
            refit=refit,
            use_cache=cache,
        )
        encoder = bundle.encoder
        shown = int(n) if n else int(assumptions.ml.peers.n_peers)
        ranked = encoder.neighbours(target, shown, apply_size_gate=size_gate)

        names: dict[str, str] = {}
        for corpus in bundle.dataset.corpora.values():
            names.update(corpus.entity_names)

        trained_through = _training_cut(bundle.dataset, cut)
        _rule(f"Learned comp set: {target}")
        console.print(
            f"[dim]{names.get(target, '')}   {len(encoder.tickers)} companies in the "
            f"fitted universe, embedded at {encoder.fit_date} on proxies filed through "
            f"{trained_through}, text weight {encoder.text_weight:.2f}.[/dim]"
        )
        # What --as-of means here, stated because it does not mean what it means
        # everywhere else in this engine. On a command that reads EDGAR it
        # discards facts filed later and stops the price series. Here there is no
        # EDGAR call: the labels are cut at the date, and the features and the
        # prose are whatever the recorded artifacts hold, because the panel and
        # the corpora were built once per panel date and this command cannot
        # rebuild them. `fit_peer_encoder` embeds at the newest panel date
        # whatever the cut, so a 2023 cut on the committed artifacts still
        # embeds at 2026-01-01. An alias that silently pinned less than the flag
        # pins elsewhere would be the worst of the three options available.
        console.print(
            f"[dim]--through / --as-of pins the LABELS only: pairs disclosed after "
            f"{trained_through} were not fitted on. The feature panel and the "
            f"business descriptions are recorded artifacts, and the universe is "
            f"embedded at {encoder.fit_date}, the newest date in the panel, "
            "whatever the cut. Point-in-time here is therefore weaker than "
            "--as-of on the commands that read EDGAR, and the fix is a panel "
            "recorded to the date rather than a flag.[/dim]\n"
        )

        filers = _trained_filers(bundle.dataset, trained_through)
        own = filers.get(target, [])
        sources = _label_sources(bundle.dataset, trained_through)
        _render_start_banner(target, own, bundle.evaluation, sources.get(target, set()))

        disclosed: set[str] = set()
        if own:
            latest = max(own, key=lambda g: g.filed or date.min)
            disclosed = {p.upper() for p in latest.peers}

        hand = [p.upper() for p in assumptions.comps.peers]
        console.print()
        _render_ranking(target, ranked, hand, disclosed, names, sources)
        _render_thin_evidence(target, ranked, sources, list(encoder.tickers), names)
        if size_gate:
            console.print(
                f"\n[dim]Size gate on: candidates below "
                f"{assumptions.ml.peers.min_market_cap:,.0f}mm market capitalisation, "
                f"or more than {assumptions.ml.peers.max_size_ratio:,.0f}x either side "
                "of the target, are excluded. The evaluation below runs UNGATED, "
                "because a gate lifts every ranker at once and would measure the band "
                "rather than the model.[/dim]"
            )

        _rule("The hand-written comp set, and where the model put it")
        _render_hand_set(encoder, target, hand, shown)

        model_top = [p for p, _s in ranked]
        if hand:
            added = [p for p in model_top if p not in set(hand)]
            dropped = [p for p in hand if p != target and p not in set(model_top)]
            console.print("\n[bold]Disagreement[/bold]")
            console.print(
                Text(
                    f"  The model adds  {', '.join(added) if added else 'nothing'}",
                    style="dim" if not added else "",
                )
            )
            console.print(
                Text(
                    f"  The model drops {', '.join(dropped) if dropped else 'nothing'}",
                    style="dim" if not dropped else "",
                )
            )

        if disclosed:
            found = [p for p in model_top if p in disclosed]
            console.print(
                f"\n[dim]{target}'s own most recent proxy named {len(disclosed)} peers; "
                f"{len(found)} of them are in the model's top {shown}. That is a "
                "training label for this company, so treat the overlap as recall "
                "rather than as evidence.[/dim]"
            )

        if bundle.evaluation is not None:
            _render_evaluation(bundle.evaluation)
            _notes(list(bundle.evaluation.notes), heading="What the evaluation found")
        else:
            console.print(
                "\n[yellow]The walk-forward evaluation was not run, so the warm and "
                "cold scores are unavailable. Rerun without --no-evaluate rather than "
                "reading the ranking as though the model were equally good on "
                "both.[/yellow]"
            )

        if bundle.ablation is not None:
            _render_ablation(bundle.ablation)
        else:
            console.print(
                "\n[dim]The tower ablation was not run. The method table above is "
                "about the raw input spaces and does not say which trained tower is "
                "carrying the ranking; --ablate refits without each one and answers "
                "that, at about a minute.[/dim]"
            )

        _notes(list(encoder.card.limitations), heading="Limitations")
        _notes(list(encoder.notes) + list(bundle.dataset.notes))

        _echo_assumptions(
            [
                ("ml.peers.enabled", assumptions.ml.peers.enabled),
                ("ml.peers.n_peers", assumptions.ml.peers.n_peers),
                ("ml.peers.text_weight", assumptions.ml.peers.text_weight),
                ("ml.peers.min_market_cap", assumptions.ml.peers.min_market_cap),
                ("ml.peers.max_size_ratio", assumptions.ml.peers.max_size_ratio),
                ("ml.random_seed", assumptions.ml.random_seed),
                ("ml.walk_forward_folds", assumptions.ml.walk_forward_folds),
                ("comps.peers", ", ".join(hand) if hand else "(empty)"),
                ("groups", groups_path),
                ("panel", panel_path),
                ("text", text_path),
                (
                    "fit",
                    "read from cache" if bundle.cached_fit else "fitted on this run",
                ),
                (
                    "evaluation",
                    "read from cache"
                    if bundle.cached_evaluation
                    else ("run on this run" if bundle.evaluation is not None else "not run"),
                ),
                (
                    "tower ablation",
                    "read from cache"
                    if bundle.cached_ablation
                    else ("run on this run" if bundle.ablation is not None else "not run"),
                ),
            ]
        )
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# screen
# --------------------------------------------------------------------------- #


_REFLEXIVITY = (
    "This says a company is priced unlike its characteristics suggest the market "
    "prices characteristics. It cannot say the market is wrong. The model is "
    "fitted on multiples, so it has learned the market's own pricing rule, and if "
    "the whole sector is mispriced it is fitted on the mispricing and reports "
    "everybody as fair. A discounted cash flow has an outside anchor in the cash "
    "and the discount rate and can tell you the market is wrong. This is not one."
)


def _render_score(model: WarrantedModel) -> None:
    """The pooled number and the differenced number, in one table, together.

    They are not two views of the same thing and separating them is how a screen
    gets oversold. The pooled correlation is mostly company identity: a feature
    vector barely moves in three months and neither does a relative multiple, so
    a model fitted on the past is rewarded for recognising a name. The
    differenced one asks whether it predicted the CHANGE, and it is the size of
    what is actually being added.
    """
    result = model.card.evaluation
    _rule("What the model scored, walk-forward")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    t.add_column("", justify="right", style="dim")
    if result is not None:
        t.add_row(
            Text("Pooled rank correlation", style="bold"),
            Text(f"{result.score:+.4f}", style="bold"),
            f"{result.n_observations:,} observations",
        )
        t.add_row(
            f"Baseline: {result.baseline_name}",
            f"{result.baseline_score:+.4f}",
            "beaten" if result.beat_baseline else "NOT beaten",
        )
        t.add_row("Lift", f"{result.lift:+.4f}", f"fold sd {_num(result.fold_sd, 4)}")
    if model.change_rank_correlation is not None:
        t.add_row(
            Text("Differenced against the company's own prior read", style="bold yellow"),
            Text(f"{model.change_rank_correlation:+.4f}", style="bold yellow"),
            f"{model.n_changes:,} observations, and the part "
            "that is not company identity",
        )
    console.print(t)
    if model.change_rank_correlation is not None and result is not None:
        console.print(
            Text(
                f"\nRead these two together. {result.score:+.4f} pooled and "
                f"{model.change_rank_correlation:+.4f} differenced means almost all of "
                "the headline is the level inherited from history rather than a view "
                "about what changed. Read every residual on this page as a description "
                "of where a company sits, not as a forecast that the gap will close.",
                style="yellow",
            )
        )

    ics = list(model.information_coefficients.values())
    if len(ics) > 1:
        console.print(
            f"\n[dim]Per-date rank correlation across "
            f"{len(ics)} dates: mean {np.mean(ics):+.3f}, standard deviation "
            f"{np.std(ics, ddof=1):.3f}, range {min(ics):+.3f} to {max(ics):+.3f}. "
            "Read the dispersion as well as the mean.[/dim]"
        )


def _render_rerating(model: WarrantedModel) -> None:
    _rule("The re-rating, and the scale a residual is read against")
    t = Table(box=None, pad_edge=False)
    t.add_column("")
    t.add_column("", justify="right")
    for label, value in model.rerating.rows():
        shown = f"{value:,}" if isinstance(value, int) else f"{value:,.4f}"
        if "Share" in label:
            shown = f"{float(value):.1%}"
        t.add_row(label, shown)
    console.print(t)
    console.print(f"\n[dim]{model.rerating.sentence()}[/dim]")


def _render_screen(frame: pd.DataFrame, when: date, sd: float) -> None:
    t = Table(box=None, pad_edge=False)
    t.add_column("Ticker", no_wrap=True)
    t.add_column("Sub-vertical", no_wrap=True)
    t.add_column("Actual", justify="right")
    t.add_column("Warranted", justify="right")
    t.add_column("Residual", justify="right")
    t.add_column("Log", justify="right")
    t.add_column("z", justify="right")
    t.add_column("", no_wrap=True)
    for _i, row in frame.iterrows():
        rich = row["verdict"] == "rich"
        style = "red" if rich else "green"
        t.add_row(
            Text(str(row["ticker"]), style="bold"),
            str(row["sub_vertical"]).replace("_", " "),
            _mult(row["actual"]),
            _mult(row["warranted"]),
            Text(f"{row['residual_turns']:+,.1f}x", style=style),
            f"{row['residual_log']:+.3f}",
            Text(f"{row['z']:+.2f}", style=style),
            Text(row["verdict"], style=style),
        )
    console.print(t)
    console.print(
        f"\n[dim]Priced on {when}. z is the residual in within-date standard "
        f"deviations ({sd:.3f} log points here) and is the column to sort on, "
        "because turns are not comparable between a 2x telecom and a 20x security "
        "name. Every residual is out of sample: the fold that produced it did not "
        "train on the row.[/dim]"
    )


def _render_refusals(panel: ObservationPanel, when: date) -> None:
    """What the engine would not price, by name, with the reason it gave.

    Nothing here is a failure of the screen. A refused row is the engine
    declining to divide by a number it could not defend, and the categories are
    not random: they cluster on the filers whose tagging is hardest and on the
    periods before a company listed. A reader who cannot see them will read the
    ranking as covering the universe.
    """
    _rule("Refused, not absent")
    console.print(_frame(panel.selection(), "The whole panel, by outcome"))

    on_date = sorted(
        (s for s in panel.skips if s.as_of == when), key=lambda s: (s.category, s.ticker)
    )
    if not on_date:
        console.print(f"\n[dim]No company-date was refused on {when}.[/dim]")
        return
    console.print(f"\n[bold]Refused on {when}[/bold]")
    t = Table(box=None, pad_edge=False)
    t.add_column("Ticker", no_wrap=True)
    t.add_column("Category", no_wrap=True)
    t.add_column("Reason", overflow="fold")
    for s in on_date:
        style = "yellow" if s.category == "share_basis_unresolved" else "dim"
        t.add_row(Text(s.ticker, style=style), Text(s.category, style=style), s.reason)
    console.print(t)


def _frame(frame: pd.DataFrame, title: str) -> Table:
    t = Table(title=title, title_justify="left", box=None, pad_edge=False)
    for column in frame.columns:
        t.add_column(str(column), justify="right" if column == "n" else "left")
    for _i, row in frame.iterrows():
        t.add_row(*[f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in row])
    return t


@app.command()
def screen(
    config: Path = _CFG,
    panel: Path = typer.Option(
        None, "--panel", help="Recorded warranted-multiple observation panel (JSON or .json.gz)."
    ),
    model: str = typer.Option("mlp", "--model", help="'mlp' or 'ridge'."),
    n: int = typer.Option(10, "--n", help="Names to show at each end."),
    when: str = typer.Option(
        None,
        "--date",
        "--as-of",
        help=(
            "Screen date (YYYY-MM-DD). Default: latest. This is what --as-of "
            "means here: the rows read are the ones the recorded panel priced on "
            "that date, and each residual comes from a walk-forward fold that "
            "did not train on it."
        ),
    ),
    ticker: str = typer.Option(None, "--ticker", help="Also print this company's own read."),
    sub_vertical: str = typer.Option(
        None, "--sub-vertical", help="Restrict the ranking to one sub-vertical."
    ),
) -> None:
    """Rank the universe by the gap between where a company trades and its warranted multiple.

    A comp table answers what similar companies trade at. This answers the
    sharper question: what does the market pay for this bundle of
    characteristics, and does this company sit above or below that line. The
    multiple is regressed on fundamentals across the whole TMT universe and
    across five years of quarter ends, the fitted value is the warranted multiple
    and the residual is the rich-or-cheap number.

    It is also the easiest thing here to over-claim, so three things are printed
    whether or not anybody wants them. The reflexivity limit, because a residual
    is a statement about relative pricing and never about value. The differenced
    rank correlation beside the pooled one, because the pooled number is mostly
    company identity and the differenced one is the part that is not. And every
    company-date the engine refused to price, with its reason, because a screen
    assembled from the rows that happened to resolve is not a screen over the
    universe.
    """
    try:
        assumptions = Assumptions.load(config)
        panel_path = (
            Path(panel) if panel else _cache_root(assumptions) / "observations.json.gz"
        )
        observations = _load_observations(panel_path)

        configured = assumptions.ml.warranted.target
        if configured != observations.target:
            console.print(
                f"[yellow]The recorded panel is a {observations.target} panel and "
                f"ml.warranted.target says {configured}. The panel wins: a multiple "
                "cannot be changed after the rows were priced.[/yellow]"
            )

        fitted = fit_warranted(observations, assumptions, model=model)
        as_of = date.fromisoformat(when) if when else fitted.latest

        _rule(f"Warranted {TARGETS[observations.target][1]} screen: {as_of}")
        console.print(Text(_REFLEXIVITY, style="yellow"))
        console.print(
            f"\n[dim]{len(observations.observations):,} observations over "
            f"{len(observations.dates)} dates and {len(observations.tickers)} "
            f"companies, from {panel_path}. Model: {model}. Demeaned by date: "
            f"{fitted.demeaned}.[/dim]"
        )

        frame = fitted.extremes(n, when=as_of)
        if sub_vertical:
            reads = [
                r
                for (_t, d), r in fitted.reads.items()
                if d == as_of and r.sub_vertical == sub_vertical
            ]
            if not reads:
                verticals = sorted({r.sub_vertical for (_t, d), r in fitted.reads.items() if d == as_of})
                console.print(
                    f"\n[red]No {sub_vertical!r} name has an out-of-sample read on "
                    f"{as_of}. That date carries: {', '.join(verticals)}[/red]"
                )
                raise typer.Exit(1)
            reads.sort(key=lambda r: r.residual_log, reverse=True)
            chosen = reads[:n] + reads[-n:] if len(reads) > 2 * n else reads
            frame = pd.DataFrame(
                [
                    {
                        "ticker": r.ticker,
                        "sub_vertical": r.sub_vertical,
                        "actual": r.actual_multiple,
                        "warranted": r.warranted_multiple,
                        "residual_turns": r.residual_turns,
                        "residual_log": r.residual_log,
                        "z": r.z,
                        "verdict": "rich" if r.residual_log > 0 else "cheap",
                    }
                    for r in chosen
                ]
            )

        console.print()
        _render_screen(frame, as_of, fitted.rerating.within_date_sd)

        if ticker:
            read = fitted.warranted(ticker, as_of)
            _rule(f"{read.ticker} on {read.as_of}")
            console.print(Text(read.sentence(), style="bold"))

        _render_score(fitted)
        _render_rerating(fitted)
        _render_refusals(observations, as_of)

        _notes(list(fitted.card.limitations), heading="Limitations")
        _notes(list(fitted.notes))

        _echo_assumptions(
            [
                ("ml.warranted.enabled", assumptions.ml.warranted.enabled),
                ("ml.warranted.target", observations.target),
                ("ml.warranted.demean_by_date", assumptions.ml.warranted.demean_by_date),
                (
                    "ml.warranted.min_train_observations",
                    assumptions.ml.warranted.min_train_observations,
                ),
                ("ml.walk_forward_folds", assumptions.ml.walk_forward_folds),
                ("ml.random_seed", assumptions.ml.random_seed),
                ("model", model),
                ("panel", panel_path),
                ("trained through", fitted.card.trained_through),
            ]
        )
    except TechvalError as exc:
        console.print(f"\n[red bold]{type(exc).__name__}[/red bold]\n{exc}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
