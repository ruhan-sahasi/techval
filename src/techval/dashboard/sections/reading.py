"""Dashboard section: reading filing facts with retrieval and a recorded Claude reader.

The section asks one question of one filing at a time: five deal terms of each
of nine merger announcements, and four operating figures from each committed
10-K. It answers each three ways:

- a Claude reader over the passages BM25 retrieves;
- the regex rules techval already has, run on those same passages;
- the same regex rules, run again on the whole input.

It scores nothing until two things are committed:

- a recording for every Claude request, which ``techval rag record`` makes;
- the owner's answer key, whose blank template ``techval rag template`` writes.

Until then it shows the readings side by side and says what is missing. Once
both are in, the headline is Claude's accuracy against the stronger regex run,
with an exact McNemar test on the tasks where exactly one of them is right.

One pass computes every figure before the provenance blocks run. The blocks
attach each figure to its entry point and time nothing, as the sample
section's do.
"""

from __future__ import annotations

from typing import Any

from ...rag.evaluate import ERROR_KINDS, Comparison, compare, verdict_text
from ...rag.key import KEY_FILE, KeyRow, KeyState, load_key
from ...rag.readings import Reading
from ...rag.run import Run, read_all
from ...rag.store import FilingStore
from ...rag.tasks import KPI_DOCUMENTS, METRICS

ID = "reading"
TITLE = "Reading filings"

RAG_DIR = "rag"
MERGER_DIR = "merger"
DDOG_FACTS = "companyfacts_DDOG.json"
INPUTS: list[str] = [RAG_DIR, MERGER_DIR, KPI_DOCUMENTS["DDOG"], DDOG_FACTS]

ENTRY_POINTS = {
    "readings": "techval.rag.run.read_all",
    "errors": "techval.rag.evaluate.compare",
    "retrieval": "techval.rag.evaluate.recall_at_k",
    "recording": "techval.rag.recording.Replayer",
}
FIGURE_INPUTS = {
    "readings": INPUTS,
    "errors": [RAG_DIR],
    "retrieval": [RAG_DIR],
    "recording": [RAG_DIR],
}
READER_NAMES = {
    "claude": "Claude reader",
    "regex_retrieved": "Regex, retrieved passages",
    "regex_native": "Regex, whole text",
}
KIND_NAMES = {
    "wrong_value": "wrong value accepted",
    "missed": "stated value missed",
    "invented": "value invented",
    "wrong_status": "wrong status",
}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _money(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v / 1e9:,.2f}bn"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.1f}mm"
    return f"${v:,.0f}"


def _value(unit: str, value: float | None, text_value: str | None) -> str:
    if unit in ("date", "text"):
        return text_value or ""
    if value is None:
        return ""
    if unit == "count":
        return f"{value:,.0f}"
    if unit == "usd":
        return _money(value)
    if unit == "percent":
        return f"{value:g}%"
    if unit == "usd_per_share":
        return f"${value:,.2f}"
    return f"{value:g}"


def show(reading: Reading, unit: str) -> str:
    if reading.status == "stated":
        return _value(unit, reading.value, reading.text_value) + (" (hedged)" if reading.hedged else "")
    if reading.status == "refused":
        return f"refused: {reading.reason}"
    return {"not_stated": "not stated", "ambiguous": "ambiguous", "missing": "not recorded"}[reading.status]


def show_key(row: KeyRow, unit: str) -> str:
    return _value(unit, row.value, row.text_value) if row.status == "stated" else row.status.replace("_", " ")


def _readings_figure(run: Run, key: KeyState | None, c: Comparison | None, top_k: int) -> dict[str, Any]:
    columns = [{"key": "subject", "label": "Filer"}, {"key": "fact", "label": "Fact"}]
    if c is not None:
        columns.append({"key": "key", "label": "Answer key"})
    columns += [{"key": reader, "label": name} for reader, name in READER_NAMES.items()]
    if c is not None:
        columns.append({"key": "verdict", "label": "Verdict"})
    rows = []
    for p in run.prepared:
        task, unit = p.task, METRICS[p.task.metric].unit
        row = {"subject": task.ticker, "fact": METRICS[task.metric].label}
        row.update({reader: show(run.readings[reader][task.task_id], unit) for reader in READER_NAMES})
        if c is not None:
            row["key"] = show_key(key.rows[task.task_id], unit)
            row["verdict"] = c.verdicts[task.task_id]
        rows.append(row)
    n = len(run.prepared)
    if c is not None:
        title = (
            f"The Claude reader is right on {c.claude.correct} of {n} filing facts, "
            f"the regex readers on {c.regex.correct}"
        )
    else:
        stated = sum(r.status == "stated" for r in run.readings["regex_native"].values())
        title = f"The regex rules state a value for {stated} of {n} filing facts on the whole text"
    subtitle = (
        "Each fact is asked of its own filing: the announcing 8-K for a deal term, Item 7 of the "
        f"10-K for an operating figure. Both readers see the {top_k} passages BM25 ranks highest "
        "for the question, and the regex rules are run again on the whole text. A Claude answer "
        "stands only if its quote is in the passage it cites and its value is in the quote."
    )
    return {
        "kind": "table",
        "title": title,
        "subtitle": subtitle,
        "data": {"columns": columns, "rows": rows},
        "wide": True,
    }


def _errors_figure(c: Comparison) -> dict[str, Any]:
    rows = [
        {
            "label": f"{READER_NAMES[score.reader]}, {KIND_NAMES[kind]}",
            "value": score.errors[kind],
            "role": role,
        }
        for score, role in ((c.claude, "model"), (c.regex, "baseline"))
        for kind in ERROR_KINDS
    ]
    return {
        "kind": "hbar",
        "title": (
            f"The Claude reader accepted {c.claude.errors['wrong_value']} wrong values, "
            f"the regex readers {c.regex.errors['wrong_value']}"
        ),
        "subtitle": (
            "Every wrong answer by kind, for the Claude reader and the stronger regex run. "
            "A wrong value accepted is the error that would reach a valuation."
        ),
        "data": {
            "rows": rows,
            "format": "int",
            "valueLabel": "Tasks",
            "labelHeader": "Reader and error",
            "roleLabels": {"model": "Claude reader", "baseline": "Regex readers"},
        },
    }


def _retrieval_figure(c: Comparison, top_k: int) -> dict[str, Any]:
    return {
        "kind": "tiles",
        "title": (
            f"The search put the answer in front of the readers on {c.recall_hits} "
            f"of {c.recall_n} stated facts"
        ),
        "subtitle": (
            f"A stated fact counts when the key's quote overlaps one of the {top_k} passages "
            "retrieved for it. A fact the search missed is missed by both passage readers alike."
        ),
        "data": {
            "tiles": [
                {
                    "label": f"Stated facts with the quote in the top {top_k} passages",
                    "value": c.recall,
                    "format": "pct:0",
                    "sub": f"{c.recall_hits} of {c.recall_n}",
                }
            ]
        },
    }


def _recording_figure(run: Run) -> dict[str, Any]:
    recs = list(run.recordings.values())
    models = sorted({r.served_by or "unknown" for r in recs})
    dates = sorted({r.recorded_at for r in recs})
    asked = sum(1 for p in run.prepared if p.passages)
    tiles = [
        {"label": "Answers replayed", "value": len(recs), "format": "int", "sub": f"of {asked} requests with passages"},
        {"label": "Answered by", "value": ", ".join(models)},
        {
            "label": "Recorded",
            "value": dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}",
            "sub": ", ".join(sorted({r.mode for r in recs})),
        },
        {"label": "Input tokens", "value": sum(int(r.usage.get("input_tokens") or 0) for r in recs), "format": "int"},
        {
            "label": "Output tokens",
            "value": sum(int(r.usage.get("output_tokens") or 0) for r in recs),
            "format": "int",
            "sub": "thinking included",
        },
    ]
    return {
        "kind": "tiles",
        "title": f"{len(recs)} recorded answers from {', '.join(models)}, replayed rather than requested",
        "subtitle": "What the Claude readings on this page were recorded from. No request is sent when the page is built.",
        "data": {"tiles": tiles},
    }


def _incomplete(key: KeyState) -> str:
    parts = []
    if key.unfilled:
        parts.append(_plural(len(key.unfilled), "blank row"))
    if key.problems:
        more = f", and {len(key.problems) - 1} more" if len(key.problems) > 1 else ""
        parts.append(f"{_plural(len(key.problems), 'rejected row')} ({key.problems[0]}{more})")
    if key.missing:
        parts.append(_plural(len(key.missing), "task") + " with no row")
    detail = ", ".join(parts) or "no answered rows"
    return f"The answer key is not complete: {detail}. Neither reader is scored until it is."


def _takeaway(run: Run, key: KeyState | None, c: Comparison | None) -> str:
    if c is not None:
        return (
            f"On {len(run.prepared)} filing facts the Claude reader is right on {c.claude.correct} and the "
            f"regex readers on {c.regex.correct}. They disagree on {c.claude_only + c.regex_only}, where an "
            f"exact McNemar test gives p = {c.mcnemar_p:.3f}, and the Claude reader accepted "
            f"{c.claude.errors['wrong_value']} wrong values against the regex readers' {c.regex.errors['wrong_value']}."
        )
    deals = len({p.task.accession for p in run.prepared if p.task.kind == "deal"})
    reports = len({p.task.accession for p in run.prepared if p.task.kind == "kpi"})
    readers = "the regex rules and by a recorded Claude reader over the same retrieved passages"
    if run.missing:
        readers = "the regex rules, over the retrieved passages and over the whole text"
        waiting = "the Claude reader's answers are not recorded yet"
    elif key is None:
        waiting = "no answer key is committed yet"
    else:
        waiting = "the answer key is not filled in yet"
    return (
        f"{len(run.prepared)} facts from {_plural(deals, 'merger announcement')} and "
        f"{_plural(reports, 'annual report')} are read by {readers}. "
        f"Neither reader is scored, because {waiting}."
    )


def _headline(c: Comparison | None) -> dict[str, Any] | None:
    if c is None:
        return None
    e = c.evaluation
    return {
        "metric": e.metric,
        "score": e.score,
        "baseline_name": e.baseline_name,
        "baseline_score": e.baseline_score,
        "lift": e.lift,
        "n": e.n_observations,
        "higher_is_better": True,
        "verdict_status": c.verdict_status,
        "verdict_text": verdict_text(c),
    }


def shape(run: Run, key: KeyState | None, top_k: int) -> dict[str, Any]:
    refusals = []
    if run.unresolved:
        refusals.append({"what": "Filings not yet committed", "why": "; ".join(run.unresolved) + "."})
    if run.missing:
        refusals.append(
            {
                "what": "Claude reader",
                "why": (
                    f"{len(run.missing)} of {len(run.prepared)} requests have no recording, so the Claude "
                    "reader is not scored. `techval rag record` records them; it needs the rag extra and "
                    "Anthropic credentials."
                ),
            }
        )
    if key is None:
        refusals.append(
            {
                "what": "Scoring",
                "why": (
                    "No answer key is committed, so neither reader is scored. `techval rag template` "
                    "writes the blank key for the owner to fill in."
                ),
            }
        )
    elif not key.complete:
        refusals.append({"what": "Scoring", "why": _incomplete(key)})
    if not run.prepared:
        return {
            "status": "refused",
            "takeaway": "No filing the reader asks about is committed.",
            "refusals": refusals or [{"what": "Tasks", "why": "No filing the reader asks about is committed."}],
            "headline": None,
            "figures": {},
        }
    scored = not run.missing and key is not None and key.complete
    c = compare(run, key) if scored else None
    figures = {"readings": _readings_figure(run, key, c, top_k)}
    if c is not None:
        figures["errors"] = _errors_figure(c)
        figures["retrieval"] = _retrieval_figure(c, top_k)
    if run.recordings:
        figures["recording"] = _recording_figure(run)
    return {
        "status": "ok",
        "takeaway": _takeaway(run, key, c),
        "refusals": refusals,
        "headline": _headline(c),
        "figures": figures,
    }


def collect(ctx) -> dict:
    settings = ctx.assumptions.ml.rag
    store = FilingStore.from_fixtures(ctx.root, annual_reports=KPI_DOCUMENTS.values(), merger_dir=MERGER_DIR)
    rag = ctx.input(RAG_DIR)
    run = read_all(store, settings, rag / "recordings")
    section = shape(run, load_key(rag / KEY_FILE, run.prepared), settings.top_k)
    for fid in section["figures"]:
        with ctx.record(fid, ENTRY_POINTS[fid], FIGURE_INPUTS[fid]):
            pass
    return section
