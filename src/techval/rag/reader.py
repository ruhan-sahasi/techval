"""The two readers: a recorded Claude reader, and the regex rules techval already has.

The Claude reader sends one request per task. The request holds:

- a fixed system prompt carrying the rules and every fact's definition, which is
  cacheable because no task changes it;
- the task's retrieved passages, each tagged with its id;
- a JSON schema the answer must follow.

The response comes from a recording, never from a live call, and every stated
answer then goes through ``verify``.

The regex reader is the incumbent, unchanged: ``tmt.kpis.extract_from_text``
for operating figures and ``tmt.precedents.extract_transaction`` for deal
terms. It runs twice, on the same retrieved passages and on its native input
(the whole Item or document), so the comparison is never rigged by cutting its
input down. It keeps its own refusal rules and is not put through ``verify``:
its deal readings carry no quoted span to check.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Sequence

from ..tmt.kpis import KPI, extract_from_text
from ..tmt.precedents import Transaction, extract_transaction
from .passages import Passage
from .readings import UNITS, Reading
from .recording import Recording, RecordingMissing, Replayer
from .tasks import METRICS, Task
from .verify import fold, verify

ANSWER_STATUSES = ("stated", "not_stated", "ambiguous")
FIELDS = ("status", "value", "text_value", "unit", "period_end", "passage_id", "quote", "hedged", "reason")


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(ANSWER_STATUSES)},
        "value": _nullable({"type": "number"}),
        "text_value": _nullable({"type": "string"}),
        "unit": _nullable({"type": "string", "enum": list(UNITS)}),
        "period_end": _nullable({"type": "string"}),
        "passage_id": _nullable({"type": "string"}),
        "quote": _nullable({"type": "string"}),
        "hedged": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}

RULES = """You read passages from SEC filings and answer one question about one fact.

Answer only from the passages. Do not use anything you know about the company or the deal from elsewhere, and do not work a figure out from other figures.

Set status to:
- stated when the passages state the answer. Give value for numbers, or text_value for names, dates and the form of consideration; the unit; the passage_id of the passage that states it; and quote, the sentence or clause that states it, copied exactly from that passage.
- not_stated when the passages do not state it. Leave value, text_value, passage_id and quote null.
- ambiguous when the passages give more than one candidate and do not say which the question means, or when the figure is not fixed, such as an exchange ratio set within a collar at closing. Quote the passage that shows why and leave value null.

Write numbers in these units:
- count: a whole number of customers or subscribers.
- usd: dollars with the scale applied, so $1.2 billion is 1200000000.
- percent: percentage points, so 120% is 120.
- usd_per_share: dollars per share, so $7.40 per share is 7.4.
- ratio: acquirer shares per target share, such as 0.0776.
- date: text_value written YYYY-MM-DD.
- text: text_value exactly as the filing writes it; for the form of consideration, one of cash, stock or mixed.

Set hedged to true when the filing qualifies the figure with a word such as approximately, about, over or more than.

Write reason as one sentence saying where the answer is, or why there is none.
"""


def system_prompt() -> str:
    definitions = "\n".join(
        f"- {m.name} ({m.unit}): {m.definition}"
        for m in sorted(METRICS.values(), key=lambda m: m.name)
    )
    return f"{RULES}\nThe facts you may be asked about:\n{definitions}\n"


def user_prompt(task: Task, passages: Sequence[Passage]) -> str:
    blocks = "\n\n".join(f'<passage id="{p.id}">\n{p.text}\n</passage>' for p in passages)
    return (
        f"Filing: {task.ticker} {task.form} filed {task.filed.isoformat()}, accession {task.accession}.\n"
        f"Question ({task.metric}): {METRICS[task.metric].question}\n\n"
        f"Passages:\n{blocks}"
    )


def build_request(task: Task, passages: Sequence[Passage], settings: Any) -> dict[str, Any]:
    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_prompt(task, passages)}],
        "output_config": {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
    }


def _well_formed(answer: Any) -> bool:
    if not isinstance(answer, dict) or set(answer) != set(FIELDS):
        return False
    value = answer["value"]
    return (
        answer["status"] in ANSWER_STATUSES
        and (answer["unit"] is None or answer["unit"] in UNITS)
        and (value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)))
        and isinstance(answer["hedged"], bool)
        and isinstance(answer["reason"], str)
        and all(answer[k] is None or isinstance(answer[k], str) for k in ("text_value", "period_end", "passage_id", "quote"))
    )


def parse_response(task: Task, recording: Recording) -> Reading:
    """The recorded response as a reading, before any check against the passages."""
    response = recording.response
    base = Reading(task.task_id, "claude", "refused", served_by=recording.served_by)
    stop = response.get("stop_reason")
    if stop == "refusal":
        category = (response.get("stop_details") or {}).get("category")
        return base.refuse("Claude declined the request" + (f" ({category})" if category else ""))
    if stop == "max_tokens":
        return base.refuse("the answer was cut off at max_tokens")
    text = next((b.get("text") for b in response.get("content") or [] if b.get("type") == "text"), None)
    try:
        answer = json.loads(text or "")
    except json.JSONDecodeError:
        return base.refuse("the answer was not JSON")
    if not _well_formed(answer):
        return base.refuse("the answer did not match the schema")
    return Reading(
        task_id=task.task_id,
        reader="claude",
        status=answer["status"],
        value=None if answer["value"] is None else float(answer["value"]),
        text_value=answer["text_value"],
        unit=answer["unit"],
        period_end=answer["period_end"],
        passage_id=answer["passage_id"],
        quote=answer["quote"],
        hedged=answer["hedged"],
        reason=answer["reason"],
        served_by=recording.served_by,
    )


class ClaudeReader:
    def __init__(self, replayer: Replayer, settings: Any) -> None:
        self.replayer = replayer
        self.settings = settings

    def read(self, task: Task, passages: Sequence[Passage]) -> tuple[Reading, Recording | None]:
        if not passages:
            return (
                Reading(task.task_id, "claude", "not_stated",
                        reason="no passage in the filing shares a term with the question, so nothing was asked"),
                None,
            )
        params = build_request(task, passages, self.settings)
        try:
            recording = self.replayer.recording(task.task_id, params)
        except RecordingMissing:
            return (
                Reading(task.task_id, "claude", "missing",
                        reason="this request has no recording; `techval rag record` records it"),
                None,
            )
        reading = parse_response(task, recording)
        return verify(reading, passages, METRICS[task.metric].unit), recording


# The regex KPI rules' names for each fact, in the order they are tried.
KPI_RULES = {
    "customers": ("customers",),
    "arr": ("arr",),
    "net_revenue_retention": ("net_revenue_retention",),
    "subscribers": ("paid_subscribers", "subscribers"),
}
# The KPI module's rung for a figure the filing hedged.
_TEXT_RUNG = 0.75
# extract_transaction looks names up through a client. Offline there is none;
# it catches the AttributeError and treats the lookup as unanswered.
_OFFLINE = object()


@lru_cache(maxsize=256)
def _kpis(text: str, as_of) -> dict[str, KPI]:
    return extract_from_text(text, as_of)


@lru_cache(maxsize=256)
def _transaction(text: str, ticker: str, form: str, accession: str, filed, name: str | None) -> Transaction | None:
    filing = {"form": form, "accession": accession, "filed": filed}
    return extract_transaction(text, ticker, _OFFLINE, filing, target_name=name)


def _passage_of(quote: str | None, passages: Sequence[Passage]) -> str | None:
    if not quote:
        return None
    want = fold(quote)
    return next((p.id for p in passages if want in fold(p.text)), None)


class RegexReader:
    def read(self, task: Task, text: str, mode: str, passages: Sequence[Passage]) -> Reading:
        reader = f"regex_{mode}"
        if not text.strip():
            return Reading(task.task_id, reader, "not_stated", reason="there was no text to read")
        if task.kind == "kpi":
            return self._kpi(task, text, reader, passages)
        return self._deal(task, text, reader)

    def _kpi(self, task: Task, text: str, reader: str, passages: Sequence[Passage]) -> Reading:
        found = _kpis(text, task.filed)
        kpi = next((found[n] for n in KPI_RULES[task.metric] if n in found), None)
        if kpi is None:
            return Reading(task.task_id, reader, "not_stated", reason="no rule for this fact matched the text")
        if kpi.confidence <= 0.0:
            return Reading(task.task_id, reader, "refused", quote=kpi.tag_or_phrase,
                           reason=kpi.notes or "the rule refused the figure")
        if kpi.unit == "usd_mm":
            value, unit = round(kpi.value * 1e6, 4), "usd"
        elif kpi.unit == "ratio":
            value, unit = round(kpi.value * 100.0, 10), "percent"
        else:
            value, unit = kpi.value, kpi.unit
        return Reading(
            task.task_id, reader, "stated", value=value, unit=unit, quote=kpi.tag_or_phrase,
            passage_id=_passage_of(kpi.tag_or_phrase, passages),
            hedged=kpi.confidence < _TEXT_RUNG, reason=kpi.notes,
        )

    def _deal(self, task: Task, text: str, reader: str) -> Reading:
        txn = _transaction(text, task.ticker, task.form, task.accession, task.filed, task.target_name)
        tid = task.task_id
        if txn is None:
            return Reading(tid, reader, "not_stated", reason="no merger agreement for this filer was read out of the text")
        # The notes are sentences; joined, they read as one with its stops removed.
        notes = "; ".join(n.strip().rstrip(".") for n in txn.notes if n.strip())
        metric = task.metric
        if metric == "cash_per_share" and txn.cash_per_share is not None:
            return Reading(tid, reader, "stated", value=txn.cash_per_share, unit="usd_per_share", reason=notes)
        if metric == "consideration_form" and txn.consideration:
            return Reading(tid, reader, "stated", text_value=txn.consideration, unit="text", reason=notes)
        if metric == "exchange_ratio":
            if txn.exchange_ratio is not None:
                return Reading(tid, reader, "stated", value=txn.exchange_ratio, unit="ratio", reason=notes)
            if txn.consideration in ("stock", "mixed"):
                return Reading(tid, reader, "refused", reason=notes or "the stock leg has no single ratio")
        if metric == "agreement_date" and txn.agreement_date is not None:
            return Reading(tid, reader, "stated", text_value=txn.agreement_date.isoformat(), unit="date", reason=notes)
        if metric == "acquirer" and txn.acquirer_name:
            return Reading(tid, reader, "stated", text_value=txn.acquirer_name, unit="text", reason=notes)
        return Reading(tid, reader, "not_stated", reason=notes or "the rules found no value for this term")
