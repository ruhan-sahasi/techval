"""The owner's answer key: a blind template, and a loader that holds it to the filing.

The template names each task and the exact question both readers were asked,
and nothing a reader said, so the key cannot be shaped by the readings it will
judge.

A filled row goes through the same checks as a reader's answer:

- its quote must occur in the filing, and near the ``start_char`` it gives when
  it gives one;
- its value must be what the quote states.

A key that fails these checks is reported row by row and scores nothing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .tasks import METRICS
from .verify import check_value, locate

if TYPE_CHECKING:
    from .run import Prepared

KEY_FILE = "answer_key.csv"
PREFILLED = (
    "task_id", "kind", "subject", "metric", "unit", "accession",
    "form", "filed", "item", "filing_url", "question",
)
TO_FILL = ("status", "value", "text_value", "period_end", "quote", "start_char", "notes")
COLUMNS = PREFILLED + TO_FILL
KEY_STATUSES = ("stated", "not_stated", "ambiguous")
# How far a stated start_char may sit from where the quote actually begins.
START_SLACK = 200


@dataclass(frozen=True)
class KeyRow:
    task_id: str
    status: str
    value: float | None = None
    text_value: str | None = None
    period_end: str | None = None
    quote: str | None = None
    notes: str = ""
    resolution: float | None = None
    span: tuple[int, int] | None = None


@dataclass
class KeyState:
    rows: dict[str, KeyRow] = field(default_factory=dict)
    unfilled: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.rows) and not (self.unfilled or self.problems or self.missing)


def write_template(path: str | Path, prepared: Sequence[Prepared]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for p in sorted(prepared, key=lambda x: x.task.task_id):
            task, metric = p.task, METRICS[p.task.metric]
            writer.writerow(
                {
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "subject": task.ticker,
                    "metric": task.metric,
                    "unit": metric.unit,
                    "accession": task.accession,
                    "form": task.form,
                    "filed": task.filed.isoformat(),
                    "item": task.item or "",
                    "filing_url": task.url or "",
                    "question": metric.question,
                    **dict.fromkeys(TO_FILL, ""),
                }
            )
    return len(prepared)


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["task_id"]: row for row in csv.DictReader(handle)}


def has_answers(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() and any((row.get("status") or "").strip() for row in _read(path).values())


def _cell(row: dict[str, str], name: str) -> str | None:
    return (row.get(name) or "").strip() or None


def _row(p: Prepared, row: dict[str, str], status: str) -> KeyRow | str:
    task_id, unit = p.task.task_id, METRICS[p.task.metric].unit
    try:
        value = float(_cell(row, "value").replace(",", "")) if _cell(row, "value") else None
        start = int(_cell(row, "start_char")) if _cell(row, "start_char") else None
    except ValueError as exc:
        return f"value and start_char must be numbers ({exc})"
    notes = _cell(row, "notes") or ""
    if status != "stated":
        return KeyRow(task_id, status, notes=notes)
    quote = _cell(row, "quote")
    if not quote:
        return "a stated row needs the quote that states it"
    spans = locate(quote, p.text, p.base)
    if not spans:
        return "the quote is not in the filing"
    if start is not None:
        near = [s for s in spans if abs(s[0] - start) <= START_SLACK]
        if not near:
            return f"the quote occurs at {spans[0][0]}, not within {START_SLACK} characters of start_char {start}"
        spans = near
    text_value = _cell(row, "text_value")
    check = check_value(unit, value, text_value, quote)
    if not check.ok:
        return check.why
    return KeyRow(
        task_id, status, value, text_value, _cell(row, "period_end"), quote, notes,
        check.resolution, spans[0],
    )


def load_key(path: str | Path, prepared: Sequence[Prepared]) -> KeyState | None:
    path = Path(path)
    if not path.exists():
        return None
    raw = _read(path)
    state = KeyState()
    for p in prepared:
        task_id = p.task.task_id
        row = raw.get(task_id)
        if row is None:
            state.missing.append(task_id)
            continue
        status = _cell(row, "status")
        if status is None:
            state.unfilled.append(task_id)
        elif status not in KEY_STATUSES:
            state.problems.append(f"{task_id}: status {status!r} is not one of {', '.join(KEY_STATUSES)}")
        else:
            parsed = _row(p, row, status)
            if isinstance(parsed, str):
                state.problems.append(f"{task_id}: {parsed}")
            else:
                state.rows[task_id] = parsed
    return state
