"""One pass over every task: retrieve its passages once, then read them three ways.

Each filing, or each Item of a 10-K, is cut and indexed once and shared by every
task asked of it. A filing whose Item cannot be found becomes an unresolved
line with the splitter's reason, never a task with empty input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import MissingDataError
from .passages import Passage, chunk
from .reader import ClaudeReader, RegexReader, build_request
from .readings import READERS, Reading
from .recording import Recording, Replayer
from .retrieve import BM25
from .store import FilingStore
from .tasks import METRICS, Task, build_tasks


@dataclass(frozen=True)
class Prepared:
    task: Task
    text: str
    base: int
    passages: tuple[Passage, ...]


@dataclass
class Run:
    prepared: list[Prepared]
    unresolved: list[str]
    readings: dict[str, dict[str, Reading]] = field(default_factory=dict)
    recordings: dict[str, Recording] = field(default_factory=dict)

    @property
    def missing(self) -> list[str]:
        return sorted(t for t, r in self.readings.get("claude", {}).items() if r.status == "missing")


def prepare(store: FilingStore, settings: Any) -> tuple[list[Prepared], list[str]]:
    tasks, unresolved = build_tasks(store)
    indexed: dict[tuple[str, str | None], tuple[str, int, BM25]] = {}
    failed: set[tuple[str, str | None]] = set()
    out: list[Prepared] = []
    for task in tasks:
        slot = (task.accession, task.item)
        if slot not in indexed and slot not in failed:
            doc = store.get(task.accession)
            try:
                text, base = store.section(doc, task.item) if task.item else (store.text(doc), 0)
            except MissingDataError as exc:
                failed.add(slot)
                unresolved.append(f"{task.ticker}: {exc}")
            else:
                passages = chunk(
                    text, accession=task.accession, item=task.item, base_offset=base,
                    size=settings.passage_chars, overlap=settings.overlap_chars,
                )
                indexed[slot] = (text, base, BM25(passages))
        if slot in failed:
            continue
        text, base, index = indexed[slot]
        top = index.top_k(METRICS[task.metric].terms, settings.top_k)
        out.append(Prepared(task, text, base, tuple(top)))
    return out, unresolved


def requests(prepared: list[Prepared], settings: Any) -> list[tuple[str, dict]]:
    return [(p.task.task_id, build_request(p.task, p.passages, settings)) for p in prepared if p.passages]


def read_all(store: FilingStore, settings: Any, recordings_root: str | Path) -> Run:
    prepared, unresolved = prepare(store, settings)
    claude = ClaudeReader(Replayer(recordings_root), settings)
    regex = RegexReader()
    run = Run(prepared, unresolved, {reader: {} for reader in READERS})
    for p in prepared:
        tid = p.task.task_id
        reading, recording = claude.read(p.task, p.passages)
        run.readings["claude"][tid] = reading
        if recording is not None:
            run.recordings[tid] = recording
        joined = "\n\n".join(x.text for x in p.passages)
        run.readings["regex_retrieved"][tid] = regex.read(p.task, joined, "retrieved", p.passages)
        run.readings["regex_native"][tid] = regex.read(p.task, p.text, "native", p.passages)
    return run
