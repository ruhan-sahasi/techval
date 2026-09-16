"""The retrieval-augmented filing reader, at the command line.

Three commands, under one group:

    techval rag template   write the blank answer key for the owner to fill in
    techval rag record     record the Claude reader's answers
    techval rag score      score both readers against the answer key

Only ``record`` touches the network, and only through the Anthropic SDK, which
is the optional ``rag`` extra: ``pip install 'techval[rag]'``. It needs
credentials the SDK can find, either ``ANTHROPIC_API_KEY`` or an
``ant auth login`` profile.

A full recording is about sixty requests through the batch API, which costs
roughly one to two US dollars. ``template`` and ``score`` read committed files
and nothing else.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .config import Assumptions
from .errors import ConfigError, TechvalError
from .rag.evaluate import ERROR_KINDS, compare, verdict_text
from .rag.key import KEY_FILE, has_answers, load_key, write_template
from .rag.recording import Recorder
from .rag.run import prepare, read_all, requests
from .rag.store import FilingStore
from .rag.tasks import KPI_DOCUMENTS

app = typer.Typer(
    no_args_is_help=True,
    help="Read filing facts with retrieval and a recorded Claude reader, and score it against the regex readers.",
)
console = Console(width=None if sys.stdout.isatty() else 120)

_CFG = typer.Option(None, "--config", "-c", help="Path to an assumptions YAML file.")
_ROOT = typer.Option(Path("tests/fixtures"), "--ml-data", help="The committed fixtures the readers read.")
_KEY = typer.Option(None, "--key", help="The answer key. Default: <ml-data>/rag/answer_key.csv.")
_RECORDINGS = typer.Option(None, "--recordings", help="Where recordings live. Default: <ml-data>/rag/recordings.")


def _store(root: Path) -> FilingStore:
    return FilingStore.from_fixtures(root, annual_reports=KPI_DOCUMENTS.values())


def _client():
    try:
        import anthropic
    except ImportError as exc:
        raise ConfigError("recording needs the Anthropic SDK: pip install 'techval[rag]'") from exc
    try:
        return anthropic.Anthropic()
    except anthropic.AnthropicError as exc:
        raise ConfigError(
            f"the Anthropic SDK found no credentials ({exc}); set ANTHROPIC_API_KEY or run `ant auth login`"
        ) from exc


def _sdk_version() -> str:
    try:
        import anthropic
    except ImportError:
        return "unknown"
    return getattr(anthropic, "__version__", "unknown")


def _stop(exc: Exception) -> typer.Exit:
    # escape: messages name extras such as techval[rag], which Rich would read as markup.
    console.print(f"[red]{escape(str(exc))}[/red]")
    return typer.Exit(1)


@app.command()
def template(
    config: Path = _CFG,
    root: Path = _ROOT,
    key: Path = _KEY,
    force: bool = typer.Option(False, "--force", help="Overwrite a key that already holds answers."),
) -> None:
    """Write the blank answer key: one row per task, and nothing any reader said."""
    path = key or root / "rag" / KEY_FILE
    try:
        settings = Assumptions.load(config).ml.rag
        prepared, unresolved = prepare(_store(root), settings)
        if has_answers(path) and not force:
            raise ConfigError(f"{path} already holds answers; pass --force to overwrite them")
        written = write_template(path, prepared)
    except TechvalError as exc:
        raise _stop(exc) from exc
    console.print(f"Wrote {written} rows to {path}.")
    for line in unresolved:
        console.print(f"[yellow]Not a task yet[/yellow]: {escape(line)}")


@app.command()
def record(
    config: Path = _CFG,
    root: Path = _ROOT,
    recordings: Path = _RECORDINGS,
    live: bool = typer.Option(
        False, "--live",
        help="Send requests one at a time with server-side fallbacks on, instead of as a batch at half price.",
    ),
    only: str = typer.Option(None, "--tasks", help="Comma-separated task ids to record. Default: every task."),
) -> None:
    """Record the Claude reader's answer for every task that has no recording yet."""
    try:
        settings = Assumptions.load(config).ml.rag
        prepared, _ = prepare(_store(root), settings)
        pending = requests(prepared, settings)
        if only:
            wanted = {t.strip() for t in only.split(",") if t.strip()}
            unknown = sorted(wanted - {t for t, _ in pending})
            if unknown:
                raise ConfigError(f"no task with retrieved passages is called {', '.join(unknown)}")
            pending = [(t, p) for t, p in pending if t in wanted]
        recorder = Recorder(
            _client(), recordings or root / "rag" / "recordings",
            today=date.today().isoformat(), sdk_version=_sdk_version(),
        )
        written = recorder.record(pending, live=live)
    except TechvalError as exc:
        raise _stop(exc) from exc
    mode = "one at a time" if live else "as a batch"
    console.print(
        f"Recorded {len(written)} of {len(pending)} requests {mode}; "
        "the others were already recorded or failed."
    )
    for task_id, why in recorder.failures:
        console.print(f"[yellow]{escape(task_id)}[/yellow]: {escape(why)}")


@app.command()
def score(config: Path = _CFG, root: Path = _ROOT, key: Path = _KEY, recordings: Path = _RECORDINGS) -> None:
    """Score both readers against the answer key, or say what is still missing."""
    try:
        settings = Assumptions.load(config).ml.rag
        run = read_all(_store(root), settings, recordings or root / "rag" / "recordings")
        state = load_key(key or root / "rag" / KEY_FILE, run.prepared)
    except TechvalError as exc:
        raise _stop(exc) from exc
    console.print(f"{len(run.prepared)} tasks, {len(run.unresolved)} filings not yet committed.")
    waiting = []
    if run.missing:
        waiting.append(f"{len(run.missing)} Claude readings are not recorded; `techval rag record` records them.")
    if state is None:
        waiting.append("No answer key is committed; `techval rag template` writes the blank key.")
    elif not state.complete:
        waiting.append(
            f"The answer key is not complete: {len(state.unfilled)} rows blank, "
            f"{len(state.problems)} rows rejected, {len(state.missing)} tasks absent."
        )
        waiting.extend(f"  {line}" for line in state.problems[:20])
    if waiting:
        for line in waiting:
            console.print(f"[yellow]{escape(line)}[/yellow]")
        return
    comparison = compare(run, state)
    table = Table(title="Readers against the answer key")
    for column in ("Reader", "Right", "Accuracy", *ERROR_KINDS):
        table.add_column(column, justify="left" if column == "Reader" else "right")
    for s in (comparison.claude, *comparison.regex_runs.values()):
        table.add_row(
            s.reader, f"{s.correct} of {s.n}", f"{s.accuracy:.1%}",
            *(str(s.errors[kind]) for kind in ERROR_KINDS),
        )
    console.print(table)
    console.print(verdict_text(comparison))
