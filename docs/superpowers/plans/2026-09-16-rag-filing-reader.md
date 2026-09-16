# Retrieval-Augmented Filing Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score a retrieval-augmented, recorded Claude reader against techval's regex readers on 61 filing facts, with every number on the dashboard reproducible offline.

**Architecture:**
- **New package `src/techval/rag/`.** It holds:
  - a point-in-time filing store;
  - a passage cutter;
  - BM25 search;
  - a verbatim-quote checker;
  - a recorded Claude reader beside a regex reader that wraps `tmt/kpis.py` and `tmt/precedents.py`;
  - the owner's answer key;
  - a paired scorer.
- **Offline by default.** Claude requests are keyed by a hash of their content and replayed from committed JSON. Only `techval rag record` touches the API.
- **Dashboard.** A new `reading` section shows both readers side by side and scores them once the key is filled.

**Tech Stack:** Python 3.11+, numpy, scipy (`binomtest`), pydantic, typer, the existing dashboard kit. `anthropic>=1.6,<2` is an optional extra used only for recording.

**Spec:** `docs/superpowers/specs/2026-09-16-rag-filing-reader-design.md`

## Global Constraints

- No em-dashes anywhere: code, comments, copy, commit messages.
- Prose reads as a person wrote it: plain and specific, no marketing.
- Commits:
  - `git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit`.
  - No `Co-Authored-By` trailer, no "Generated with" line.
- No live network call in collection, rendering or tests. Only `Recorder` imports `anthropic`.
- Claude model: `claude-opus-5`, set by `ml.rag.model`. Thinking and effort stay at the model defaults.
- Batch recording sends no `fallbacks` parameter; the batch API rejects it.
- Live recording (`--live`) sends `betas=["server-side-fallback-2026-07-01"]` and `extra_body={"fallbacks": "default"}`.
- Configuration defaults, from the spec's table:

  | Setting | Default |
  |---|---|
  | `passage_chars` | 1200 |
  | `overlap_chars` | 200 |
  | `top_k` | 6 |
  | `max_tokens` | 16000 |

- Confidence rungs are unchanged: text 0.75, hedged 0.4, refused 0.0.
- Tests run with `.venv/bin/python -m pytest`. The full suite takes about four minutes.
- Follow the surrounding style:
  - module docstrings that argue their rules;
  - `from __future__ import annotations`;
  - frozen dataclasses for values.

## Deviations from the spec, decided while planning

Each is small, and each is stated where it applies.

1. **A sixth shared module, `rag/readings.py`, holds the `Reading` value.** Without it, `verify.py` and `reader.py` would import each other.
2. **`verify` folds typographic quote marks (“ ” ‘ ’) as well as whitespace.** Filings use curly quotes, and a reader copying `“Merger Consideration”` as `"Merger Consideration"` has still quoted it. Digits and words are never folded.
3. **The verbatim check applies to Claude readings and to key rows.** Regex readings keep the refusal rules they already have. Deal readings have no quoted span to check, and the regex KPI reading's quote is the fragment its own rule matched.
4. **Scoring adds a fourth error kind, `wrong_status`,** for a non-stated answer with the wrong status: `ambiguous` where the key says `not_stated`, or the reverse. The spec's three kinds do not cover it.
5. **The five deal metrics are named:**
   - `cash_per_share`, the cash leg per target share;
   - `consideration_form`;
   - `exchange_ratio`;
   - `agreement_date`;
   - `acquirer`.

   "Offer value per share" is split into the cash leg and the exchange ratio, because a mixed deal states the two separately and never states one number.
6. **Replay is exposed as `Replayer.recording(task_id, params) -> Recording`,** so the served model travels with the response.
7. **KPI tasks exist only for 10-K texts that are committed.** The rest are listed as unresolved on the page. NET, NFLX and TMUS become tasks once Task 13's recorder has run, which needs `TECHVAL_SEC_EMAIL`.

## File map

| File | Responsibility |
|---|---|
| Create `src/techval/rag/__init__.py` | Package docstring only |
| Create `src/techval/rag/readings.py` | `Reading`, `READERS`, `UNITS`, `STATUSES` |
| Create `src/techval/rag/passages.py` | `Passage`, `chunk` |
| Create `src/techval/rag/retrieve.py` | `tokens`, `BM25` |
| Create `src/techval/rag/verify.py` | `fold`, `fold_words`, `numbers`, `dates`, `locate`, `check_value`, `verify` |
| Create `src/techval/rag/store.py` | `Document`, `FilingStore` |
| Create `src/techval/rag/tasks.py` | `Metric`, `METRICS`, `Task`, `DEALS`, `KPI_DOCUMENTS`, `build_tasks` |
| Create `src/techval/rag/recording.py` | `request_key`, `Recording`, `RecordingMissing`, `Replayer`, `Recorder` |
| Create `src/techval/rag/reader.py` | `ANSWER_SCHEMA`, `system_prompt`, `build_request`, `parse_response`, `ClaudeReader`, `RegexReader` |
| Create `src/techval/rag/run.py` | `Prepared`, `Run`, `prepare`, `requests`, `read_all` |
| Create `src/techval/rag/key.py` | `KEY_FILE`, `KeyRow`, `KeyState`, `write_template`, `has_answers`, `load_key` |
| Create `src/techval/rag/evaluate.py` | `ReaderScore`, `Comparison`, `IncompleteRun`, `compare`, `recall_at_k`, `verdict_text` |
| Create `src/techval/commands_rag.py` | `techval rag template / record / score` |
| Create `src/techval/dashboard/sections/reading.py` | The collector and `shape` |
| Create `src/techval/dashboard/assets/sections/reading.js` | The section renderer |
| Create `tests/fixtures/rag/record_filing_text.py` | Records the NET, NFLX and TMUS 10-K texts |
| Create `tests/fixtures/rag/recordings/.gitkeep` | Where recordings land |
| Modify `src/techval/config.py:688-713` | `RagAssumptions` block on `MLAssumptions` |
| Modify `pyproject.toml` | `rag` optional extra |
| Modify `src/techval/cli.py:133-144` | Mount the `rag` group |
| Modify `src/techval/dashboard/sections/__init__.py:25-36` | Add `reading` to `SECTION_IDS` |
| Modify `src/techval/dashboard/gallery.py:49-60, 1010-1013` | `PAGE_ORDER` and a synthetic `_reading()` |
| Modify `tests/dashboard/test_collect.py:125`, `tests/dashboard/test_frontend.py:30-41` | Section count and ids |
| Create `tests/rag/__init__.py`, `tests/rag/test_*.py`, `tests/dashboard/test_section_reading.py` | Tests |
| Modify `README.md` | A short "Reading filings" subsection |

---
### Task 1: Settings, the optional extra, and the shared reading value

**Files:**
- Modify: `src/techval/config.py` (add `RagAssumptions` above `MLAssumptions`; add one field to `MLAssumptions`)
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)
- Create: `src/techval/rag/__init__.py`, `src/techval/rag/readings.py`
- Test: `tests/rag/__init__.py`, `tests/rag/test_readings.py`

**Interfaces:**
- Produces: `Assumptions().ml.rag` with `.model: str`, `.passage_chars: int`, `.overlap_chars: int`, `.top_k: int`, `.max_tokens: int`
- Produces: `techval.rag.readings.Reading(task_id, reader, status, value=None, text_value=None, unit=None, period_end=None, passage_id=None, quote=None, hedged=False, reason="", served_by=None)`, which is frozen and has `.answered: bool` and `.refuse(why) -> Reading`
- Produces: `READERS = ("claude", "regex_retrieved", "regex_native")`
- Produces: `UNITS = ("count", "usd", "percent", "usd_per_share", "ratio", "date", "text")`
- Produces: `STATUSES = ("stated", "not_stated", "ambiguous", "refused", "missing")`

- [ ] **Step 1: Write the failing tests**

`tests/rag/__init__.py` is empty. `tests/rag/test_readings.py`:

```python
"""The reading value every reader returns, and the settings that shape a reading."""

from __future__ import annotations

import pytest

from techval.config import Assumptions
from techval.errors import ConfigError
from techval.rag.readings import READERS, STATUSES, UNITS, Reading


def test_the_rag_settings_default_to_the_spec():
    rag = Assumptions().ml.rag
    assert (rag.model, rag.passage_chars, rag.overlap_chars, rag.top_k, rag.max_tokens) == (
        "claude-opus-5",
        1200,
        200,
        6,
        16000,
    )


def test_an_unknown_rag_setting_is_refused(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("ml:\n  rag:\n    temperature: 0.2\n")
    with pytest.raises(ConfigError):
        Assumptions.load(path)


def test_a_reading_names_a_known_reader_and_status():
    with pytest.raises(ValueError):
        Reading("T", "gpt", "stated")
    with pytest.raises(ValueError):
        Reading("T", "claude", "maybe")


def test_refusing_keeps_the_reading_and_states_why():
    reading = Reading("T", "claude", "stated", value=1.0, unit="count", quote="1 customer")
    refused = reading.refuse("the quote is not in the passage it cites")
    assert refused.status == "refused" and refused.value == 1.0
    assert refused.reason == "the quote is not in the passage it cites"
    assert not refused.answered and reading.answered


def test_the_vocabularies_are_the_spec_s():
    assert READERS == ("claude", "regex_retrieved", "regex_native")
    assert STATUSES == ("stated", "not_stated", "ambiguous", "refused", "missing")
    assert "usd_per_share" in UNITS and "percent" in UNITS
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_readings.py -q -p no:cacheprovider`
Expected: collection error, `ModuleNotFoundError: No module named 'techval.rag'`.

- [ ] **Step 3: Implement**

In `src/techval/config.py`, directly above `class MLAssumptions(_Base):`, add:

```python
class RagAssumptions(_Base):
    """The retrieval-augmented filing reader: what it reads, and which model reads it.

    Every value here can change an answer, so every value is part of the request
    key a recording is filed under. Changing one means recording again, which is
    the point: a replayed answer is only evidence for the request that produced it.
    """

    model: str = Field(
        "claude-opus-5",
        description="The Claude model that reads the retrieved passages.",
    )
    passage_chars: int = Field(
        1200, ge=200, le=8000, description="Target length of one passage, in characters."
    )
    overlap_chars: int = Field(
        200, ge=0, le=2000, description="Characters each passage shares with the one before it."
    )
    top_k: int = Field(
        6, ge=1, le=20, description="Passages retrieved per task and given to every reader."
    )
    max_tokens: int = Field(
        16000, ge=1024, le=64000, description="Output ceiling for one reading, thinking included."
    )
```

In `MLAssumptions`, after the `signals:` field, add:

```python
    rag: RagAssumptions = Field(default_factory=RagAssumptions)
```

In `pyproject.toml`, under `[project.optional-dependencies]`, add the line:

```toml
rag = ["anthropic>=1.6,<2"]
```

Create `src/techval/rag/__init__.py`:

```python
"""A retrieval-augmented reader for facts stated in SEC filings.

The package asks one question of one filing at a time: how much cash a target's
holders receive, how many customers a company states, and so on. It answers it
three ways, over the same retrieved passages. A recorded Claude reader answers
once. The regex rules techval already has answer twice, on the passages and on
the whole Item or document. An owner-written answer key then says which was
right.

Nothing here is a live dependency. The Claude reader's requests and responses
are committed as fixtures and replayed, so the dashboard and the tests read
them offline with no SDK and no key. Only ``techval rag record`` calls the API.
Nothing here changes a valuation, a label or a model score either: the readings
are measured here, and used elsewhere only if the measurement supports it.
"""
```

Create `src/techval/rag/readings.py`:

```python
"""What a reader returns for one task, whichever reader it is.

``stated``, ``not_stated`` and ``ambiguous`` are answers. ``refused`` is a reader,
or the checker behind it, declining to give one, and it always says why.
``missing`` is a Claude reading with no recording behind it: the reader did not
run, which is a different fact from a reader that ran and declined.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

READERS = ("claude", "regex_retrieved", "regex_native")
UNITS = ("count", "usd", "percent", "usd_per_share", "ratio", "date", "text")
STATUSES = ("stated", "not_stated", "ambiguous", "refused", "missing")


@dataclass(frozen=True)
class Reading:
    task_id: str
    reader: str
    status: str
    value: float | None = None
    text_value: str | None = None
    unit: str | None = None
    period_end: str | None = None
    passage_id: str | None = None
    quote: str | None = None
    hedged: bool = False
    reason: str = ""
    served_by: str | None = None

    def __post_init__(self) -> None:
        if self.reader not in READERS:
            raise ValueError(f"unknown reader {self.reader!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown reading status {self.status!r}")

    @property
    def answered(self) -> bool:
        return self.status in ("stated", "not_stated", "ambiguous")

    def refuse(self, why: str) -> Reading:
        return replace(self, status="refused", reason=why)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_readings.py tests/test_config.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/techval/config.py pyproject.toml src/techval/rag/__init__.py src/techval/rag/readings.py tests/rag/__init__.py tests/rag/test_readings.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Add the filing reader's settings, its optional SDK extra, and the reading every reader returns"
```

---

### Task 2: Passages that quote back to the filing

**Files:**
- Create: `src/techval/rag/passages.py`
- Test: `tests/rag/test_passages.py`

**Interfaces:**
- Produces: `Passage(id: str, accession: str, item: str | None, start_char: int, end_char: int, text: str)`, frozen, with `id == f"{accession}:{start_char}"`
- Produces: `chunk(source, *, accession, item=None, base_offset=0, size=1200, overlap=200, snap=150) -> list[Passage]`. It raises `ValueError` unless `size > snap` and `0 <= overlap < size`.

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_passages.py`:

```python
"""Passages: every one is the document at its own range, and together they cover it."""

from __future__ import annotations

import pytest

from techval.rag.passages import chunk

TEXT = ("Alpha beta gamma. " * 30 + "Delta epsilon! " * 30).strip()


def test_every_passage_is_the_document_at_its_range():
    head = "HEADER " * 10
    document = head + TEXT
    for p in chunk(TEXT, accession="A", base_offset=len(head), size=200, overlap=40):
        assert document[p.start_char : p.end_char] == p.text
        assert p.id == f"A:{p.start_char}"


def test_windows_end_on_a_sentence_when_one_is_near():
    passages = chunk(TEXT, accession="A", size=200, overlap=40, snap=60)
    assert len(passages) > 2
    for p in passages[:-1]:
        assert p.text.rstrip()[-1] in ".!"


def test_the_text_is_covered_and_neighbours_overlap():
    passages = chunk(TEXT, accession="A", size=200, overlap=40)
    assert passages[0].start_char == 0 and passages[-1].end_char == len(TEXT)
    for a, b in zip(passages, passages[1:]):
        assert a.start_char < b.start_char <= a.end_char


def test_whitespace_only_windows_are_skipped():
    text = "Words here. " + " " * 500 + "More words."
    passages = chunk(text, accession="A", size=200, overlap=0, snap=50)
    assert passages and all(p.text.strip() for p in passages)
    assert passages[-1].text.strip().endswith("More words.")


@pytest.mark.parametrize("size,overlap", [(200, 200), (200, -1), (150, 10)])
def test_windows_that_cannot_advance_are_refused(size, overlap):
    with pytest.raises(ValueError):
        chunk(TEXT, accession="A", size=size, overlap=overlap)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_passages.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.passages'`.

- [ ] **Step 3: Implement**

`src/techval/rag/passages.py`:

```python
"""Filing text cut into passages that can always be quoted back to the filing.

A passage carries its character range in the whole document, and its text is
that range, byte for byte. Everything downstream relies on the identity: a
reader cites a passage, the checker finds the quote inside it, and the page can
point a reader at the exact place in the filing. ``nlp.sections`` keeps the same
identity for the Items it splits, so a passage cut from an Item is offset by the
Item's own start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A sentence ends at a full stop, question or exclamation mark, optionally
# followed by a closing quote or bracket, and then whitespace.
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]?\s")


@dataclass(frozen=True)
class Passage:
    id: str
    accession: str
    item: str | None
    start_char: int
    end_char: int
    text: str


def chunk(
    source: str,
    *,
    accession: str,
    item: str | None = None,
    base_offset: int = 0,
    size: int = 1200,
    overlap: int = 200,
    snap: int = 150,
) -> list[Passage]:
    """Windows of about ``size`` characters, each sharing ``overlap`` with the one before.

    A window that stops short of the end is cut back to the last sentence end
    inside its final ``snap`` characters, so a passage rarely stops mid-sentence.
    ``base_offset`` is where ``source`` starts in the document, so for every
    passage ``document[p.start_char:p.end_char] == p.text``. Windows holding
    nothing but whitespace are skipped.
    """
    if size <= snap or overlap < 0 or overlap >= size:
        raise ValueError(
            f"a window of {size} characters needs size above snap ({snap}) and "
            f"0 <= overlap < size (overlap is {overlap})"
        )
    out: list[Passage] = []
    start, n = 0, len(source)
    while start < n:
        end = min(start + size, n)
        if end < n:
            last = None
            for match in _SENTENCE_END.finditer(source, end - snap, end):
                last = match.end()
            if last is not None:
                end = last
        text = source[start:end]
        if text.strip():
            at = base_offset + start
            out.append(Passage(f"{accession}:{at}", accession, item, at, base_offset + end, text))
        if end >= n:
            break
        start = end - overlap if end - overlap > start else end
    return out
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_passages.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/passages.py tests/rag/test_passages.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Cut filing text into passages that quote back to their exact range"
```

---

### Task 3: BM25 search over one document

**Files:**
- Create: `src/techval/rag/retrieve.py`
- Test: `tests/rag/test_retrieve.py`

**Interfaces:**
- Consumes: `Passage` (Task 2)
- Produces: `tokens(text) -> list[str]`, which lowercases and keeps numbers such as `32,700` and `1.5` whole
- Produces: `BM25(passages, *, k1=1.5, b=0.75)`, with `.score(index, terms) -> float` and `.top_k(query: Sequence[str], k: int) -> list[Passage]`. Results come best first, ties go to the lower `start_char`, and passages that share no term with the query are never returned.

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_retrieve.py`:

```python
"""BM25 over one document: the passage that answers ranks first, and nothing else leaks in."""

from __future__ import annotations

from techval.rag.passages import Passage
from techval.rag.retrieve import BM25, tokens


def _p(at: int, text: str) -> Passage:
    return Passage(f"A:{at}", "A", None, at, at + len(text), text)


CORPUS = [
    _p(0, "The company had approximately 32,700 customers at year end."),
    _p(100, "Revenue grew and the company invested in research."),
    _p(200, "Customers with ARR of $100,000 or more numbered 4,310."),
    _p(300, "The company describes its platform and its products."),
]


def test_tokens_keep_numbers_whole():
    assert tokens("About 32,700 customers, 1.5 billion") == ["about", "32,700", "customers", "1.5", "billion"]


def test_the_passages_that_answer_rank_first_and_ties_go_to_the_earlier():
    top = BM25(CORPUS).top_k(["customers", "customer count"], 2)
    assert [p.start_char for p in top] == [0, 200]


def test_passages_sharing_no_term_are_never_returned():
    assert BM25(CORPUS).top_k(["subscribers"], 3) == []


def test_a_rare_term_outweighs_a_common_one():
    index = BM25(CORPUS)
    assert index.score(2, ["arr"]) > index.score(1, ["company"])


def test_the_same_query_gives_the_same_order_every_time():
    index = BM25(CORPUS)
    first = [p.id for p in index.top_k(["company", "customers"], 4)]
    assert all([p.id for p in index.top_k(["customers", "company"], 4)] == first for _ in range(3))
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_retrieve.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.retrieve'`.

- [ ] **Step 3: Implement**

`src/techval/rag/retrieve.py`:

```python
"""BM25 over one document's passages: which passages a question should read.

One document is one index. A question about Datadog's customers is asked of
Datadog's filing and of nothing else, so no other filing, earlier or later, can
weight the terms, and the point-in-time rule holds by construction. The scoring
is Okapi BM25 with the textbook ``k1`` and ``b``. It is a keyword search, it
knows nothing about meaning, and that is what makes its misses visible: the
page reports how often the answer's passage was among those retrieved.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from .passages import Passage

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, passages: Sequence[Passage], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = list(passages)
        self.k1, self.b = k1, b
        self._counts = [Counter(tokens(p.text)) for p in self.passages]
        self._lengths = [sum(c.values()) for c in self._counts]
        self._average = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        n = len(self._counts)
        frequency = Counter(t for counts in self._counts for t in counts)
        self._idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def score(self, index: int, terms: Sequence[str]) -> float:
        counts, length = self._counts[index], self._lengths[index]
        norm = 1.0 - self.b + self.b * (length / self._average if self._average else 0.0)
        total = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if tf:
                total += self._idf[term] * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        return total

    def top_k(self, query: Sequence[str], k: int) -> list[Passage]:
        """The ``k`` best passages for the query, best first, ties to the earlier passage.

        Passages that share no term with the query are never returned, so a
        filing that never mentions the metric yields nothing rather than its
        first few pages.
        """
        terms = sorted({t for phrase in query for t in tokens(phrase)})
        ranked = sorted(
            ((self.score(i, terms), p.start_char, i) for i, p in enumerate(self.passages)),
            key=lambda row: (-row[0], row[1]),
        )
        return [self.passages[i] for score, _, i in ranked[:k] if score > 0]
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_retrieve.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/retrieve.py tests/rag/test_retrieve.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Rank a filing's passages for a question with BM25 fitted on that filing alone"
```

---
### Task 4: Holding an answer to its quote

**Files:**
- Create: `src/techval/rag/verify.py`
- Test: `tests/rag/test_verify.py`

**Interfaces:**
- Consumes: `Passage` (Task 2) and `Reading` (Task 1)
- Produces, text folding:
  - `fold(text) -> str` collapses whitespace and folds typographic quote marks.
  - `fold_words(text) -> str` does the same, then lowercases and turns punctuation into spaces.
- Produces, number and date parsing:
  - `numbers(text) -> list[Number]`. A `Number` is frozen and carries `value`, `resolution`, `percent`, `dollars`, `scaled` and `start`.
  - `dates(text) -> list[date]`
- Produces, locating a quote: `locate(quote, text, base=0) -> list[tuple[int, int]]` returns absolute spans. The match tolerates any run of whitespace and either style of quote mark.
- Produces, checking a value: `check_value(unit, value, text_value, quote) -> Check`. A `Check` is frozen and carries `ok`, `why`, `resolution` and `hedged`.
- Produces, the whole check: `verify(reading, passages, unit) -> Reading`. It returns non-stated readings unchanged.
- Produces the constant `MONEY_FLOOR = 1_000_000.0`.

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_verify.py`. The first five sentences are copied from the committed DDOG 10-K and the PAYO and WORK 8-Ks:

```python
"""The checker: a stated answer stands only if its passage, its quote and its value all hold."""

from __future__ import annotations

import pytest

from techval.rag.passages import Passage
from techval.rag.readings import Reading
from techval.rag.verify import check_value, locate, numbers, verify

CUSTOMERS = (
    "As of December 31, 2025, we had approximately 32,700 customers spanning organizations "
    "of a broad range of sizes and industries, compared to approximately 30,000 as of December 31, 2024."
)
NRR = "As of December 31, 2025, our trailing 12-month dollar-based net retention rate was about 120%."
THRESHOLD = "We monitor our number of customers with ARR of $100,000 or more, and believe it is useful to investors"
PAYO = "will be converted into the right to receive $7.40 in cash, without interest (the “Merger Consideration”)."
WORK = (
    "the right to receive 0.0776 shares of Salesforce common stock and the right to receive "
    "$26.79 in cash, without interest"
)
AGREED = "On June 29, 2026, Iridium entered into an Agreement and Plan of Merger"


def _passage(text: str) -> Passage:
    return Passage("A:0", "A", None, 0, len(text), text)


def _stated(**fields) -> Reading:
    base = dict(task_id="T", reader="claude", status="stated", passage_id="A:0", reason="read")
    base.update(fields)
    return Reading(**base)


def test_a_hedged_count_is_kept_and_marked_hedged():
    r = verify(_stated(value=32700, unit="count", quote=CUSTOMERS), [_passage(CUSTOMERS)], "count")
    assert r.status == "stated" and r.hedged


def test_a_count_the_quote_does_not_state_is_refused():
    r = verify(_stated(value=32000, unit="count", quote=CUSTOMERS), [_passage(CUSTOMERS)], "count")
    assert r.status == "refused" and "does not state 32000" in r.reason


def test_a_rate_is_read_in_percentage_points():
    r = verify(_stated(value=120, unit="percent", quote=NRR), [_passage(NRR)], "percent")
    assert r.status == "stated" and r.hedged


def test_a_small_dollar_figure_with_no_scale_word_is_refused():
    r = verify(_stated(value=100000, unit="usd", quote=THRESHOLD), [_passage(THRESHOLD)], "usd")
    assert r.status == "refused" and "threshold" in r.reason


def test_a_quote_not_in_its_passage_is_refused():
    quote = "the right to receive $7.40 in cash"
    r = verify(_stated(value=7.4, unit="usd_per_share", quote=quote), [_passage(WORK)], "usd_per_share")
    assert r.status == "refused" and "not in the passage" in r.reason


def test_quote_marks_and_spacing_are_folded_and_nothing_else():
    straight = 'will be converted into the right to receive $7.40 in cash,  without interest (the "Merger Consideration").'
    r = verify(_stated(value=7.4, unit="usd_per_share", quote=straight), [_passage(PAYO)], "usd_per_share")
    assert r.status == "stated"
    altered = straight.replace("7.40", "7.50")
    r = verify(_stated(value=7.5, unit="usd_per_share", quote=altered), [_passage(PAYO)], "usd_per_share")
    assert r.status == "refused"


def test_a_passage_the_reader_was_not_given_is_refused():
    r = verify(
        _stated(value=7.4, unit="usd_per_share", quote=PAYO, passage_id="B:9"),
        [_passage(PAYO)],
        "usd_per_share",
    )
    assert r.status == "refused" and "not given" in r.reason


def test_an_answer_in_another_unit_is_refused():
    r = verify(_stated(value=7.4, unit="usd", quote=PAYO), [_passage(PAYO)], "usd_per_share")
    assert r.status == "refused" and "usd_per_share" in r.reason


def test_a_mixed_consideration_needs_both_legs_in_the_quote():
    assert verify(_stated(text_value="mixed", unit="text", quote=WORK), [_passage(WORK)], "text").status == "stated"
    assert verify(_stated(text_value="mixed", unit="text", quote=PAYO), [_passage(PAYO)], "text").status == "refused"
    assert verify(_stated(text_value="cash", unit="text", quote=PAYO), [_passage(PAYO)], "text").status == "stated"


def test_an_exchange_ratio_is_not_read_from_a_dollar_figure():
    assert verify(_stated(value=0.0776, unit="ratio", quote=WORK), [_passage(WORK)], "ratio").status == "stated"
    assert verify(_stated(value=26.79, unit="ratio", quote=WORK), [_passage(WORK)], "ratio").status == "refused"


def test_a_date_must_be_the_one_the_quote_states():
    p = [_passage(AGREED)]
    assert verify(_stated(text_value="2026-06-29", unit="date", quote=AGREED), p, "date").status == "stated"
    assert verify(_stated(text_value="2026-06-30", unit="date", quote=AGREED), p, "date").status == "refused"
    assert verify(_stated(text_value="June 29", unit="date", quote=AGREED), p, "date").status == "refused"


def test_a_name_must_appear_in_the_quote():
    quote = "entered into an Agreement and Plan of Merger with Rocket Lab Corporation, a Delaware corporation"
    p = [_passage(quote)]
    assert verify(_stated(text_value="Rocket Lab Corporation", unit="text", quote=quote), p, "text").status == "stated"
    assert verify(_stated(text_value="Rocket Lab USA", unit="text", quote=quote), p, "text").status == "refused"


def test_answers_that_are_not_stated_pass_through_untouched():
    reading = Reading("T", "claude", "ambiguous", reason="two ratios, set at closing")
    assert verify(reading, [], "ratio") is reading


def test_numbers_carry_their_scale_and_resolution():
    [n] = numbers("ARR of $1.2 billion")
    assert n.value == pytest.approx(1.2e9) and n.resolution == pytest.approx(1e8)
    assert n.dollars and n.scaled and not n.percent
    [m] = numbers("rate was 120%")
    assert m.value == 120 and m.percent and m.resolution == 1


def test_a_stated_number_needs_a_value():
    assert not check_value("count", None, None, CUSTOMERS).ok


def test_locate_finds_a_quote_across_spacing_and_marks():
    text = "the right to receive $7.40 in cash, without\ninterest (the “Merger Consideration”)."
    spans = locate('receive $7.40 in cash, without interest (the "Merger Consideration")', text, base=100)
    assert spans == [(100 + text.index("receive"), 100 + len(text) - 1)]
    assert locate("receive $7.50", text) == []
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_verify.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.verify'`.

- [ ] **Step 3: Implement**

`src/techval/rag/verify.py`:

```python
"""Hold an answer to its quote: the quote must be in the passage, and the value in the quote.

A language model can state a number that no passage contains, or quote a
sentence that says something else. Neither is caught by asking the model
again, so every stated answer is checked here, deterministically, against the
passages it was given. The same rules hold the owner's answer key to the filing,
so a typo in the key fails loudly instead of marking a correct reader wrong.

Only two things are folded before a quote is compared: runs of whitespace, and
typographic quote marks. Filings write “Merger Consideration” with curly marks,
and a reader that copies them straight has still quoted the filing. Digits,
words and punctuation otherwise stay exactly as they are.

A dollar figure below one million with no scale word is refused, as
``tmt.kpis`` already refuses it. "ARR of $100,000" is a threshold for counting
customers far more often than it is a company's ARR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from typing import Sequence

from .passages import Passage
from .readings import Reading

_MARKS = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})
_SCALE = {"thousand": 1e3, "million": 1e6, "mm": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}
_NUMBER = re.compile(
    r"(?P<dollar>\$\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<pct>%|percent\b))?"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|mm|bn)\b)?",
    re.IGNORECASE,
)
_HEDGES = (
    "approximately", "about", "more than", "over", "nearly", "roughly",
    "almost", "at least", "in excess of", "greater than",
)
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTHS = {name: i for i, name in enumerate(_MONTH_NAMES, start=1)}
_DATE_WORDS = re.compile(r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(\d{1,2}),\s*(\d{4})", re.IGNORECASE)
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_CATEGORY_EVIDENCE = {
    "cash": (("cash",),),
    "stock": (("share", "stock"),),
    "mixed": (("cash",), ("share", "stock")),
}
_NUMERIC_UNITS = ("count", "usd", "usd_per_share", "ratio", "percent")

MONEY_FLOOR = 1_000_000.0


def fold(text: str) -> str:
    return " ".join(text.translate(_MARKS).split())


def fold_words(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.translate(_MARKS).lower()).split())


@dataclass(frozen=True)
class Number:
    value: float
    resolution: float
    percent: bool
    dollars: bool
    scaled: bool
    start: int


def numbers(text: str) -> list[Number]:
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group("num")
        scale = _SCALE.get((m.group("scale") or "").lower(), 1.0)
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        out.append(
            Number(
                value=float(raw.replace(",", "")) * scale,
                resolution=(10.0**-decimals) * scale,
                percent=bool(m.group("pct")),
                dollars=bool(m.group("dollar")),
                scaled=scale != 1.0,
                start=m.start(),
            )
        )
    return out


def dates(text: str) -> list[date]:
    found = []
    for month, day, year in _DATE_WORDS.findall(text):
        try:
            found.append(date(int(year), _MONTHS[month.lower()], int(day)))
        except ValueError:
            continue
    for year, month, day in _DATE_ISO.findall(text):
        try:
            found.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return found


def locate(quote: str, text: str, base: int = 0) -> list[tuple[int, int]]:
    """Every place ``quote`` occurs in ``text``, as absolute ``(start, end)`` offsets."""
    parts = []
    for ch in fold(quote):
        if ch == " ":
            parts.append(r"\s+")
        elif ch == '"':
            parts.append("[\"“”„]")
        elif ch == "'":
            parts.append("['‘’]")
        else:
            parts.append(re.escape(ch))
    if not parts:
        return []
    pattern = re.compile("".join(parts))
    return [(base + m.start(), base + m.end()) for m in pattern.finditer(text)]


def _hedged_at(text: str, start: int) -> bool:
    before = text[max(0, start - 30) : start].lower()
    return any(h in before for h in _HEDGES)


@dataclass(frozen=True)
class Check:
    ok: bool
    why: str = ""
    resolution: float | None = None
    hedged: bool = False


def check_value(unit: str | None, value: float | None, text_value: str | None, quote: str) -> Check:
    """Whether ``quote`` states this value, in this unit."""
    if unit in _NUMERIC_UNITS:
        if value is None:
            return Check(False, "a stated number carries no value")
        for n in numbers(quote):
            if (unit == "percent") != n.percent:
                continue
            if n.dollars and unit not in ("usd", "usd_per_share"):
                continue
            if abs(n.value - value) > max(n.resolution / 2, 1e-9 * max(1.0, abs(value))):
                continue
            if unit == "usd" and value < MONEY_FLOOR and not n.scaled:
                return Check(
                    False,
                    f"the quote gives ${n.value:,.0f} with no scale word, and below $1,000,000 "
                    "that is as likely a threshold as a total",
                )
            return Check(True, resolution=n.resolution, hedged=_hedged_at(quote, n.start))
        return Check(False, f"the quote does not state {value:g}")
    if unit == "date":
        try:
            want = date.fromisoformat(text_value or "")
        except ValueError:
            return Check(False, f"{text_value!r} is not a date written YYYY-MM-DD")
        if want in dates(quote):
            return Check(True)
        return Check(False, f"the quote does not state {want.isoformat()}")
    if unit == "text":
        if not text_value:
            return Check(False, "a stated text answer carries no text")
        words = fold_words(quote)
        groups = _CATEGORY_EVIDENCE.get(text_value.strip().lower())
        if groups is not None:
            if all(any(w in words for w in group) for group in groups):
                return Check(True)
            return Check(False, f"the quote does not show {text_value.strip().lower()} consideration")
        if fold_words(text_value) in words:
            return Check(True)
        return Check(False, f"the quote does not contain {text_value!r}")
    return Check(False, f"no rule checks the unit {unit!r}")


def verify(reading: Reading, passages: Sequence[Passage], unit: str) -> Reading:
    """A stated reading kept only if its passage, its quote and its value all hold."""
    if reading.status != "stated":
        return reading
    if reading.unit != unit:
        return reading.refuse(
            f"the answer is in {reading.unit!r}, and this fact is read in {unit!r}"
        )
    passage = next((p for p in passages if p.id == reading.passage_id), None)
    if passage is None:
        return reading.refuse(
            f"the answer cites passage {reading.passage_id!r}, which it was not given"
        )
    if not reading.quote or fold(reading.quote) not in fold(passage.text):
        return reading.refuse("the quote is not in the passage it cites")
    check = check_value(unit, reading.value, reading.text_value, reading.quote)
    if not check.ok:
        return reading.refuse(check.why)
    return replace(reading, hedged=reading.hedged or check.hedged)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_verify.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/verify.py tests/rag/test_verify.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Hold every stated answer to a verbatim quote and a value the quote actually states"
```

---
### Task 5: The point-in-time filing store

**Files:**
- Create: `src/techval/rag/store.py`
- Test: `tests/rag/test_store.py`

**Interfaces:**
- Produces: `Document(accession, ticker, form, filed: date, path, url=None, entity_name=None)`, frozen. `path` is relative to the ml-data root.
- Produces: `FilingStore(root, documents)`, built with `FilingStore.from_fixtures(root, annual_reports=(...), merger_dir="merger")`. Its methods:
  - `.get(accession) -> Document | None`
  - `.by_path(path) -> Document | None`
  - `.documents(ticker, as_of, forms=None) -> list[Document]`: newest first, and never a document filed after `as_of`
  - `.text(doc) -> str`
  - `.section(doc, item) -> tuple[str, int]`: the Item text and its offset. It raises `MissingDataError` when the Item is not found.

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_store.py`:

```python
"""The filing store: dated from the filers' own records, and blind to anything filed later."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from techval.errors import MissingDataError
from techval.rag.store import FilingStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DDOG_TEXT = "filing_text_DDOG_2025.json.gz"


@pytest.fixture(scope="module")
def store() -> FilingStore:
    return FilingStore.from_fixtures(FIXTURES, annual_reports=(DDOG_TEXT, "rag/not_committed.json.gz"))


def test_a_merger_document_is_dated_from_its_filer_s_submissions(store):
    doc = store.get("0000950103-26-008945")
    assert (doc.ticker, doc.form, doc.filed) == ("PAYO", "8-K", date(2026, 6, 15))
    assert doc.url == "https://www.sec.gov/Archives/edgar/data/1845815/000095010326008945/dp248400_8k.htm"
    assert doc.entity_name == "Payoneer Global Inc."
    assert doc.path == "merger/text/0000950103-26-008945.txt"


def test_nothing_filed_after_the_date_asked_about_is_served(store):
    assert [d.accession for d in store.documents("SLAB", date(2026, 3, 1))] == ["0001193125-26-036712"]
    assert [d.form for d in store.documents("slab", date(2026, 12, 31))] == ["DEFM14A", "8-K"]
    assert store.documents("SLAB", date(2026, 2, 3)) == []
    assert [d.form for d in store.documents("SLAB", date(2026, 12, 31), forms=("8-K",))] == ["8-K"]


def test_a_compressed_merger_text_reads_like_a_plain_one(store):
    doc = store.get("0001193125-26-128959")
    assert doc.path.endswith(".txt.gz")
    assert "merger" in store.text(doc).lower()


def test_an_annual_report_is_split_into_its_items(store):
    doc = store.by_path(DDOG_TEXT)
    assert (doc.ticker, doc.form, doc.accession, doc.filed) == ("DDOG", "10-K", "0001628280-26-008819", date(2026, 2, 18))
    assert doc.url == "https://www.sec.gov/Archives/edgar/data/1561550/000162828026008819/"
    text, base = store.section(doc, "7")
    assert store.text(doc)[base : base + len(text)] == text
    assert "dollar-based net retention rate" in text


def test_an_annual_report_that_is_not_committed_is_not_served(store):
    assert store.by_path("rag/not_committed.json.gz") is None


def test_an_item_the_splitter_cannot_find_is_refused(store):
    with pytest.raises(MissingDataError):
        store.section(store.by_path(DDOG_TEXT), "99")
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_store.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.store'`.

- [ ] **Step 3: Implement**

`src/techval/rag/store.py`:

```python
"""Committed filing text, served point in time.

Two kinds of document are committed:

- **Annual reports.** Stored whole as ``filing_text_*.json.gz``, each payload
  carrying its own accession, form and filing date.
- **Merger filings.** The primary documents under ``merger/text/``. They are
  dated from the filer's own ``submissions_<TICKER>.json`` beside them, because
  a merger text with no date cannot be served point in time and is therefore
  not served at all.

``documents`` never returns a filing dated after the date asked about. A task
reads its own filing as of that filing's date, so a later amendment, proxy or
10-K cannot reach it.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from ..errors import MissingDataError
from ..nlp.sections import FilingSections, split_items

EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{bare}/"


@dataclass(frozen=True)
class Document:
    accession: str
    ticker: str
    form: str
    filed: date
    path: str
    url: str | None = None
    entity_name: str | None = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entity(root: Path, name: str) -> tuple[int | None, str | None]:
    path = root / name
    if not path.exists():
        return None, None
    facts = _read_json(path)
    return facts.get("cik"), facts.get("entityName")


def _annual_report(root: Path, rel: str) -> Document | None:
    path = root / rel
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        head = json.load(handle)
    ticker = str(head["ticker"]).upper()
    cik, name = _entity(root, f"companyfacts_{ticker}.json")
    url = head.get("url")
    if url is None and cik is not None:
        url = EDGAR_ARCHIVE.format(cik=cik, bare=head["accession"].replace("-", ""))
    return Document(
        accession=head["accession"],
        ticker=ticker,
        form=head["form"],
        filed=date.fromisoformat(head["filed"]),
        path=rel,
        url=url,
        entity_name=head.get("entity_name") or name,
    )


def _submissions(directory: Path) -> dict[str, tuple[str, str, date]]:
    """accession -> (ticker, form, filed), from every submissions file in the directory."""
    out: dict[str, tuple[str, str, date]] = {}
    for path in sorted(directory.glob("submissions_*.json")):
        ticker = path.stem.split("_", 1)[1].upper()
        recent = (_read_json(path).get("filings") or {}).get("recent") or {}
        for accession, form, filed in zip(
            recent.get("accessionNumber", []), recent.get("form", []), recent.get("filingDate", [])
        ):
            out[accession] = (ticker, form, date.fromisoformat(filed))
    return out


def _merger_documents(root: Path, merger_dir: str) -> list[Document]:
    directory = root / merger_dir
    manifest = directory / "MANIFEST.json"
    if not manifest.exists():
        return []
    dated = _submissions(directory)
    docs = []
    for name, entry in sorted(_read_json(manifest).get("files", {}).items()):
        if not name.startswith("text/"):
            continue
        accession = Path(name).name.split(".", 1)[0]
        if accession not in dated:
            continue
        ticker, form, filed = dated[accession]
        _, entity = _entity(directory, f"companyfacts_{ticker}.json")
        docs.append(
            Document(
                accession=accession,
                ticker=ticker,
                form=form,
                filed=filed,
                path=f"{merger_dir}/{name}",
                url=entry.get("source"),
                entity_name=entity,
            )
        )
    return docs


class FilingStore:
    def __init__(self, root: str | Path, documents: Iterable[Document]) -> None:
        self.root = Path(root)
        self._docs = {d.accession: d for d in documents}
        self._by_path = {d.path: d for d in self._docs.values()}
        self._text: dict[str, str] = {}
        self._splits: dict[str, FilingSections] = {}

    @classmethod
    def from_fixtures(
        cls,
        root: str | Path,
        *,
        annual_reports: Iterable[str] = (),
        merger_dir: str = "merger",
    ) -> FilingStore:
        root = Path(root)
        docs = [d for rel in annual_reports if (d := _annual_report(root, rel)) is not None]
        docs.extend(_merger_documents(root, merger_dir))
        return cls(root, docs)

    def get(self, accession: str) -> Document | None:
        return self._docs.get(accession)

    def by_path(self, path: str) -> Document | None:
        return self._by_path.get(path)

    def documents(
        self, ticker: str, as_of: date, forms: tuple[str, ...] | None = None
    ) -> list[Document]:
        """The ticker's documents filed on or before ``as_of``, newest first."""
        rows = [
            d
            for d in self._docs.values()
            if d.ticker == ticker.upper() and d.filed <= as_of and (forms is None or d.form in forms)
        ]
        return sorted(rows, key=lambda d: (d.filed, d.accession), reverse=True)

    def text(self, doc: Document) -> str:
        if doc.accession not in self._text:
            path = self.root / doc.path
            if doc.path.endswith(".json.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    body = json.load(handle)["text"]
            elif doc.path.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    body = handle.read()
            else:
                body = path.read_text(encoding="utf-8")
            self._text[doc.accession] = body
        return self._text[doc.accession]

    def section(self, doc: Document, item: str) -> tuple[str, int]:
        """One Item's text and where it starts in the document."""
        if doc.accession not in self._splits:
            self._splits[doc.accession] = split_items(self.text(doc), doc.form)
        found = self._splits[doc.accession].get(item)
        if found is None:
            raise MissingDataError(
                f"{doc.ticker} {doc.form} {doc.accession} has no Item {item} the section splitter could find"
            )
        return found.text, found.start_char
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_store.py -q -p no:cacheprovider`
Expected: all pass. If the PAYO `url` assertion fails, print `store.get("0000950103-26-008945").url` and compare it with the `source` field of `tests/fixtures/merger/MANIFEST.json` for `text/0000950103-26-008945.txt`. The test must assert exactly that manifest value.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/store.py tests/rag/test_store.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Serve committed filing text point in time, dated from the filers' own submissions"
```

---
### Task 6: The 61 tasks and what each one asks

**Files:**
- Create: `src/techval/rag/tasks.py`
- Test: `tests/rag/test_tasks.py`

**Interfaces:**
- Consumes: `FilingStore`, `Document` (Task 5) and `UNITS` (Task 1)
- Produces: `Metric(name, kind, unit, label, question, definition, terms: tuple[str, ...])`, frozen, and `METRICS: dict[str, Metric]`
- Produces: `DEAL_METRICS`, `KPI_METRICS`, `DEALS: tuple[tuple[str, str], ...]` (ticker, accession) and `KPI_DOCUMENTS: dict[str, str]` (ticker to fixture path)
- Produces: `Task(task_id, kind, ticker, metric, accession, form, filed: date, item: str | None, url: str | None, target_name: str | None)`, frozen, with `.as_of -> date`
- Produces: `build_tasks(store) -> tuple[list[Task], list[str]]`, which returns the sorted tasks and one line per unresolved filing

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_tasks.py`:

```python
"""The task list: every committed filing becomes its questions, and a missing one is named."""

from __future__ import annotations

from pathlib import Path

import pytest

from techval.rag.readings import UNITS
from techval.rag.store import FilingStore
from techval.rag.tasks import DEAL_METRICS, DEALS, KPI_DOCUMENTS, KPI_METRICS, METRICS, build_tasks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def built():
    store = FilingStore.from_fixtures(FIXTURES, annual_reports=KPI_DOCUMENTS.values())
    return build_tasks(store)


def test_the_nine_deals_become_forty_five_tasks(built):
    tasks, _ = built
    deals = [t for t in tasks if t.kind == "deal"]
    assert len(deals) == 45 == len(DEALS) * len(DEAL_METRICS)
    assert {t.ticker for t in deals} == {ticker for ticker, _ in DEALS}
    assert all(t.item is None and t.form == "8-K" for t in deals)


def test_kpi_tasks_exist_exactly_for_the_committed_annual_reports(built):
    tasks, unresolved = built
    kpis = [t for t in tasks if t.kind == "kpi"]
    assert all(t.item == "7" and t.form == "10-K" for t in kpis)
    for ticker, path in KPI_DOCUMENTS.items():
        committed = (FIXTURES / path).exists()
        mine = [t for t in kpis if t.ticker == ticker]
        assert len(mine) == (len(KPI_METRICS) if committed else 0), ticker
        assert any(line.startswith(f"{ticker}:") for line in unresolved) != committed, ticker
    assert "DDOG" in {t.ticker for t in kpis}


def test_task_ids_are_unique_sorted_and_read_as_of_their_own_filing(built):
    tasks, _ = built
    ids = [t.task_id for t in tasks]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    assert all(t.as_of == t.filed for t in tasks)
    assert next(t for t in tasks if t.task_id == "PAYO:acquirer").target_name == "Payoneer Global Inc."


def test_every_metric_is_fully_described():
    assert set(METRICS) == set(DEAL_METRICS) | set(KPI_METRICS)
    for name, m in METRICS.items():
        assert m.name == name and m.unit in UNITS
        assert m.question.endswith("?") and m.definition and m.terms and m.label
        assert m.kind == ("deal" if name in DEAL_METRICS else "kpi")
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_tasks.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.tasks'`.

- [ ] **Step 3: Implement**

`src/techval/rag/tasks.py`:

```python
"""The filing facts the readers are asked for, and the filings they are asked of.

The questions come in two groups:

- **Deals.** Each of the nine committed merger announcements, all 8-Ks, is asked
  for five deal terms.
- **KPIs.** Each committed 10-K is asked, in its Item 7, for four operating
  figures that filers state in prose rather than tag.

A fact a filing does not state is still a task: the right answer is
``not_stated``, and a reader that invents a value for it is wrong.

The deal terms follow what an announcement actually states. A mixed deal gives
a cash amount and an exchange ratio in one sentence and never a single price,
so the cash leg and the ratio are asked separately. A collar makes the ratio
depend on the buyer's price at closing, and the honest answer to the ratio
question is then ``ambiguous``.

KPI tasks exist only for 10-K texts that are committed. A 10-K that is not
committed is reported as unresolved rather than silently dropped;
``tests/fixtures/rag/record_filing_text.py`` records the missing ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .store import Document, FilingStore


@dataclass(frozen=True)
class Metric:
    name: str
    kind: str
    unit: str
    label: str
    question: str
    definition: str
    terms: tuple[str, ...]


METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        Metric(
            "cash_per_share", "deal", "usd_per_share", "Cash per share",
            "How much cash does each share of the target's common stock receive in the merger, in US dollars per share?",
            "the cash each target common share is converted into; if the consideration has no cash leg, not_stated",
            ("per share", "cash", "merger consideration", "converted", "right to receive", "without interest"),
        ),
        Metric(
            "consideration_form", "deal", "text", "Form of consideration",
            "Is the merger consideration for each target share cash only, stock only, or a mix of cash and stock?",
            "text_value cash, stock or mixed, judged from what each target common share is converted into",
            ("merger consideration", "cash", "shares", "common stock", "converted", "right to receive"),
        ),
        Metric(
            "exchange_ratio", "deal", "ratio", "Exchange ratio",
            "How many shares of the acquirer does each target share receive?",
            "the fixed number of acquirer shares per target share; ambiguous when a collar or formula sets it at closing; not_stated when there is no stock leg",
            ("exchange ratio", "shares", "common stock", "fraction", "collar", "converted"),
        ),
        Metric(
            "agreement_date", "deal", "date", "Agreement date",
            "On what date was the merger agreement entered into?",
            "the date the parties entered into the agreement and plan of merger, as text_value YYYY-MM-DD",
            ("entered into", "agreement and plan of merger", "dated as of", "merger agreement"),
        ),
        Metric(
            "acquirer", "deal", "text", "Acquirer",
            "Which company does the merger agreement name as the buyer's parent?",
            "the parent company party to the merger agreement, written exactly as the filing names it; not a merger subsidiary",
            ("parent", "merger sub", "agreement and plan of merger", "by and among", "acquire"),
        ),
        Metric(
            "customers", "kpi", "count", "Customers",
            "How many customers does the company state it had at the end of the fiscal year, in total?",
            "the total customer count; a count of customers above a spending threshold is a different figure",
            ("customers", "customer count", "approximately"),
        ),
        Metric(
            "arr", "kpi", "usd", "Annual recurring revenue",
            "What annual recurring revenue does the company state for the whole company at the end of the fiscal year, in US dollars?",
            "the company's total ARR; a threshold used to count customers, such as ARR of $100,000 or more, is not the company's ARR",
            ("annual recurring revenue", "arr", "annual run-rate revenue"),
        ),
        Metric(
            "net_revenue_retention", "kpi", "percent", "Net revenue retention",
            "What net revenue retention rate, or dollar-based net retention rate, does the company state for the end of the fiscal year, in percent?",
            "the net or dollar-based net retention rate for the latest period, in percentage points",
            ("net retention", "net revenue retention", "dollar-based", "retention rate"),
        ),
        Metric(
            "subscribers", "kpi", "count", "Subscribers",
            "How many paying subscribers, members or subscriptions does the company state it had in total at the end of the period?",
            "the total count of paying subscribers or members; net additions in a period are not the total",
            ("subscribers", "paid memberships", "members", "subscriptions", "customers"),
        ),
    )
}

DEAL_METRICS = ("cash_per_share", "consideration_form", "exchange_ratio", "agreement_date", "acquirer")
KPI_METRICS = ("customers", "arr", "net_revenue_retention", "subscribers")

# The announcing 8-K of each committed deal. SPLK's 2021 8-K and ZEN's 2021 S-4
# are committed too, as decoys for the precedent reader, and are not tasks.
DEALS: tuple[tuple[str, str], ...] = (
    ("IRDM", "0001104659-26-078482"),
    ("MNDT", "0001104659-22-031786"),
    ("PAYO", "0000950103-26-008945"),
    ("RAMP", "0001104659-26-062908"),
    ("ROKU", "0001140361-26-025115"),
    ("SLAB", "0001193125-26-036712"),
    ("SPLK", "0001104659-23-102594"),
    ("WORK", "0001193125-20-307385"),
    ("ZEN", "0001193125-22-181655"),
)

KPI_DOCUMENTS: dict[str, str] = {
    "DDOG": "filing_text_DDOG_2025.json.gz",
    "NET": "rag/filing_text_NET_2025.json.gz",
    "NFLX": "rag/filing_text_NFLX_2025.json.gz",
    "TMUS": "rag/filing_text_TMUS_2025.json.gz",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    ticker: str
    metric: str
    accession: str
    form: str
    filed: date
    item: str | None
    url: str | None
    target_name: str | None

    @property
    def as_of(self) -> date:
        return self.filed


def _task(doc: Document, metric: str, item: str | None) -> Task:
    return Task(
        task_id=f"{doc.ticker}:{metric}",
        kind=METRICS[metric].kind,
        ticker=doc.ticker,
        metric=metric,
        accession=doc.accession,
        form=doc.form,
        filed=doc.filed,
        item=item,
        url=doc.url,
        target_name=doc.entity_name,
    )


def build_tasks(store: FilingStore) -> tuple[list[Task], list[str]]:
    tasks: list[Task] = []
    unresolved: list[str] = []
    for ticker, accession in DEALS:
        doc = store.get(accession)
        if doc is None:
            unresolved.append(f"{ticker}: merger filing {accession} is not committed")
            continue
        tasks.extend(_task(doc, metric, None) for metric in DEAL_METRICS)
    for ticker, path in KPI_DOCUMENTS.items():
        doc = store.by_path(path)
        if doc is None:
            unresolved.append(
                f"{ticker}: the 10-K text {path} is not committed; "
                "tests/fixtures/rag/record_filing_text.py records it"
            )
            continue
        tasks.extend(_task(doc, metric, "7") for metric in KPI_METRICS)
    return sorted(tasks, key=lambda t: t.task_id), unresolved
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_tasks.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/tasks.py tests/rag/test_tasks.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Define the filing facts both readers are asked for, one task per committed filing"
```

---
### Task 7: Recording and replaying Claude requests

**Files:**
- Create: `src/techval/rag/recording.py`, `tests/fixtures/rag/recordings/.gitkeep`
- Test: `tests/rag/test_recording.py`

**Interfaces:**
- Produces, keys: `canonical(params) -> str` and `request_key(params) -> str`, the sha256 hex digest of the canonical JSON
- Produces, the stored record: `Recording(key, task_id, request, response, served_by, usage, recorded_at, mode, sdk_version)`, frozen, with `.to_json()` and `Recording.from_json(raw)`
- Produces, the missing-recording error: `RecordingMissing(task_id, key)`, a `TechvalError`
- Produces, replay: `Replayer(root)` with `.recording(task_id, params) -> Recording`. It raises `RecordingMissing` when the file is absent, and `DataSourceError` when the stored request differs from `params`.
- Produces, recording: `Recorder(client, root, *, today, sdk_version, sleep=time.sleep, poll_seconds=30)` with `.record(requests: Sequence[tuple[str, dict]], *, live=False) -> list[Recording]` and `.failures: list[tuple[str, str]]`
- Produces the constant `FALLBACK_BETA = "server-side-fallback-2026-07-01"`

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_recording.py`:

```python
"""Recordings: one file per request, found by the request's own content, never by a live call."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from techval.errors import DataSourceError
from techval.rag.recording import (
    FALLBACK_BETA,
    Recorder,
    Recording,
    RecordingMissing,
    Replayer,
    request_key,
)

PARAMS_A = {"model": "claude-opus-5", "max_tokens": 16000, "messages": [{"role": "user", "content": "A"}]}
PARAMS_B = {"model": "claude-opus-5", "max_tokens": 16000, "messages": [{"role": "user", "content": "B"}]}


def _message(text: str, model: str = "claude-opus-5") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class _Message:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return dict(self._payload)


class FakeBatches:
    """Returns results in reverse order, and leaves unanswered requests errored."""

    def __init__(self, answers: dict[str, dict], statuses=("in_progress", "ended")) -> None:
        self.answers = answers
        self.statuses = list(statuses)
        self.created: list[list[dict]] = []

    def create(self, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id="batch_1")

    def retrieve(self, batch_id):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return SimpleNamespace(processing_status=status)

    def results(self, batch_id):
        for request in reversed(self.created[-1]):
            payload = self.answers.get(request["custom_id"])
            if payload is None:
                yield SimpleNamespace(custom_id=request["custom_id"], result=SimpleNamespace(type="errored"))
            else:
                yield SimpleNamespace(
                    custom_id=request["custom_id"],
                    result=SimpleNamespace(type="succeeded", message=_Message(payload)),
                )


class FakeClient:
    def __init__(self, batches: FakeBatches | None = None, live: dict | None = None) -> None:
        self.messages = SimpleNamespace(batches=batches)
        self.calls: list[dict] = []

        def create(**kwargs):
            self.calls.append(kwargs)
            return _Message(live)

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def _recorder(client, root, sleeps=None):
    return Recorder(
        client, root, today="2026-09-16", sdk_version="1.6.0",
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


def test_the_key_ignores_key_order_and_follows_content():
    shuffled = {"messages": PARAMS_A["messages"], "max_tokens": 16000, "model": "claude-opus-5"}
    assert request_key(PARAMS_A) == request_key(shuffled)
    assert request_key(PARAMS_A) != request_key(PARAMS_B)
    assert len(request_key(PARAMS_A)) == 64


def test_a_missing_recording_is_an_error_not_a_live_call(tmp_path):
    with pytest.raises(RecordingMissing) as caught:
        Replayer(tmp_path).recording("PAYO:acquirer", PARAMS_A)
    assert caught.value.task_id == "PAYO:acquirer"
    assert caught.value.key == request_key(PARAMS_A)


def test_a_recording_of_a_different_request_is_refused(tmp_path):
    wrong = Recording(request_key(PARAMS_A), "T", PARAMS_B, _message("{}"), "claude-opus-5", {}, "2026-09-16", "batch", "1.6.0")
    (tmp_path / f"{request_key(PARAMS_A)}.json").write_text(json.dumps(wrong.to_json()))
    with pytest.raises(DataSourceError):
        Replayer(tmp_path).recording("T", PARAMS_A)


def test_a_batch_is_recorded_by_custom_id_whatever_order_results_arrive_in(tmp_path):
    batches = FakeBatches({request_key(PARAMS_A): _message("a"), request_key(PARAMS_B): _message("b")})
    sleeps: list[float] = []
    recorder = _recorder(FakeClient(batches), tmp_path, sleeps)
    written = recorder.record([("T:a", PARAMS_A), ("T:b", PARAMS_B)])
    assert sorted(r.task_id for r in written) == ["T:a", "T:b"]
    assert sleeps == [30]
    [sent] = batches.created
    assert all("fallbacks" not in r["params"] for r in sent)
    replayed = Replayer(tmp_path).recording("T:b", PARAMS_B)
    assert replayed.response["content"][0]["text"] == "b"
    assert (replayed.mode, replayed.served_by, replayed.recorded_at) == ("batch", "claude-opus-5", "2026-09-16")


def test_a_failed_batch_entry_is_reported_and_nothing_is_written_for_it(tmp_path):
    batches = FakeBatches({request_key(PARAMS_A): _message("a")})
    recorder = _recorder(FakeClient(batches), tmp_path)
    written = recorder.record([("T:a", PARAMS_A), ("T:b", PARAMS_B)])
    assert [r.task_id for r in written] == ["T:a"]
    assert recorder.failures == [("T:b", "errored")]
    assert not (tmp_path / f"{request_key(PARAMS_B)}.json").exists()


def test_a_request_already_recorded_is_not_sent_again(tmp_path):
    batches = FakeBatches({request_key(PARAMS_A): _message("a")})
    _recorder(FakeClient(batches), tmp_path).record([("T:a", PARAMS_A)])
    again = FakeBatches({})
    assert _recorder(FakeClient(again), tmp_path).record([("T:a", PARAMS_A)]) == []
    assert again.created == []


def test_a_live_recording_turns_on_server_side_fallbacks_and_names_the_model_that_answered(tmp_path):
    client = FakeClient(live=_message("x", model="claude-opus-4-8"))
    [rec] = _recorder(client, tmp_path).record([("T:a", PARAMS_A)], live=True)
    [call] = client.calls
    assert call["betas"] == [FALLBACK_BETA]
    assert call["extra_body"] == {"fallbacks": "default"}
    assert {k: call[k] for k in PARAMS_A} == PARAMS_A
    assert (rec.mode, rec.served_by) == ("live", "claude-opus-4-8")
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_recording.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.recording'`.

- [ ] **Step 3: Implement**

Create `tests/fixtures/rag/recordings/.gitkeep` as an empty file.

`src/techval/rag/recording.py`:

```python
"""Claude requests, recorded once and replayed forever.

A recording is filed under the sha256 of the request's canonical JSON: the
model, the settings, the system prompt, the passages and the question. Change
any of them and the key changes, so a replay can only ever return the answer to
the request actually being made. The replayer also compares the stored request
with the one asked for, byte for byte, before it trusts the file.

There is no live fallback. A missing recording raises ``RecordingMissing``, and
the page turns that into a refusal naming the command that records it.

Recording goes through the Message Batches API at half price, and batches do
not accept the server-side ``fallbacks`` parameter, so a declined batch request
is recorded as declined. ``live=True`` sends requests one at a time with
fallbacks on, and the recording names the model that answered.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..errors import DataSourceError, TechvalError

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def canonical(params: Mapping[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(params: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(params).encode("utf-8")).hexdigest()


class RecordingMissing(TechvalError):
    def __init__(self, task_id: str, key: str) -> None:
        super().__init__(f"no recording for task {task_id} (request {key[:12]}); run `techval rag record`")
        self.task_id = task_id
        self.key = key


@dataclass(frozen=True)
class Recording:
    key: str
    task_id: str
    request: dict
    response: dict
    served_by: str | None
    usage: dict
    recorded_at: str
    mode: str
    sdk_version: str

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Recording:
        missing = [f for f in cls.__dataclass_fields__ if f not in raw]
        if missing:
            raise DataSourceError(f"a recording is missing {', '.join(missing)}")
        return cls(**{f: raw[f] for f in cls.__dataclass_fields__})


class Replayer:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def recording(self, task_id: str, params: Mapping[str, Any]) -> Recording:
        key = request_key(params)
        path = self.path(key)
        if not path.exists():
            raise RecordingMissing(task_id, key)
        found = Recording.from_json(json.loads(path.read_text(encoding="utf-8")))
        if canonical(found.request) != canonical(params):
            raise DataSourceError(
                f"recording {key[:12]} holds a different request from the one task {task_id} sends"
            )
        return found


class Recorder:
    def __init__(
        self,
        client: Any,
        root: str | Path,
        *,
        today: str,
        sdk_version: str,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = 30,
    ) -> None:
        self.client = client
        self.root = Path(root)
        self.today = today
        self.sdk_version = sdk_version
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self.failures: list[tuple[str, str]] = []

    def record(self, requests: Sequence[tuple[str, dict]], *, live: bool = False) -> list[Recording]:
        replayer = Replayer(self.root)
        pending = [(t, p) for t, p in requests if not replayer.path(request_key(p)).exists()]
        if not pending:
            return []
        self.root.mkdir(parents=True, exist_ok=True)
        return self._live(pending) if live else self._batch(pending)

    def _batch(self, pending: Sequence[tuple[str, dict]]) -> list[Recording]:
        by_key = {request_key(p): (t, p) for t, p in pending}
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": key, "params": params} for key, (_, params) in by_key.items()]
        )
        while self.client.messages.batches.retrieve(batch.id).processing_status != "ended":
            self.sleep(self.poll_seconds)
        written = []
        for entry in self.client.messages.batches.results(batch.id):
            task_id, params = by_key[entry.custom_id]
            if entry.result.type != "succeeded":
                self.failures.append((task_id, entry.result.type))
                continue
            written.append(self._write(task_id, params, entry.result.message.to_dict(), "batch"))
        return sorted(written, key=lambda r: r.task_id)

    def _live(self, pending: Sequence[tuple[str, dict]]) -> list[Recording]:
        written = []
        for task_id, params in pending:
            message = self.client.beta.messages.create(
                **params, betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}
            )
            written.append(self._write(task_id, params, message.to_dict(), "live"))
        return written

    def _write(self, task_id: str, params: dict, message: dict, mode: str) -> Recording:
        recording = Recording(
            key=request_key(params),
            task_id=task_id,
            request=params,
            response=message,
            served_by=message.get("model"),
            usage=message.get("usage") or {},
            recorded_at=self.today,
            mode=mode,
            sdk_version=self.sdk_version,
        )
        Replayer(self.root).path(recording.key).write_text(
            json.dumps(recording.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return recording
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_recording.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/recording.py tests/rag/test_recording.py tests/fixtures/rag/recordings/.gitkeep
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Record Claude requests by batch or live, and replay them by the hash of their content"
```

---
### Task 8: The two readers

**Files:**
- Create: `src/techval/rag/reader.py`
- Test: `tests/rag/test_reader.py`

**Interfaces:**
- Consumes:
  - `Passage` (Task 2) and `Reading`, `UNITS` (Task 1)
  - `verify`, `fold` (Task 4)
  - `METRICS`, `Task`, `build_tasks` (Task 6)
  - `Recording`, `RecordingMissing`, `Replayer` (Task 7)
  - `techval.tmt.kpis.extract_from_text(text, as_of) -> dict[str, KPI]`
  - `techval.tmt.precedents.extract_transaction(text, ticker, client, filing, *, target_name=None) -> Transaction | None`
- Produces, the request:
  - `ANSWER_SCHEMA: dict`
  - `system_prompt() -> str`, identical for every task
  - `user_prompt(task, passages) -> str`
  - `build_request(task, passages, settings) -> dict`, with keys `model`, `max_tokens`, `system`, `messages` and `output_config`
- Produces, parsing: `parse_response(task, recording) -> Reading`
- Produces, the Claude reader: `ClaudeReader(replayer, settings)` with `.read(task, passages) -> tuple[Reading, Recording | None]`
- Produces, the regex reader: `RegexReader()` with `.read(task, text, mode, passages) -> Reading`. `mode` is `"retrieved"` or `"native"`, and the reading's reader is `f"regex_{mode}"`.

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_reader.py`:

```python
"""The two readers: a recorded Claude reader held to its quotes, and the regex rules as they stand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.rag.passages import Passage, chunk
from techval.rag.reader import (
    ANSWER_SCHEMA,
    ClaudeReader,
    RegexReader,
    build_request,
    parse_response,
    system_prompt,
)
from techval.rag.recording import Recording, RecordingMissing, request_key
from techval.rag.store import FilingStore
from techval.rag.tasks import KPI_DOCUMENTS, build_tasks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SETTINGS = Assumptions().ml.rag
PAYO_SENTENCE = "will be converted into the right to receive $7.40 in cash, without interest"


@pytest.fixture(scope="module")
def store() -> FilingStore:
    return FilingStore.from_fixtures(FIXTURES, annual_reports=KPI_DOCUMENTS.values())


@pytest.fixture(scope="module")
def tasks(store):
    return {t.task_id: t for t in build_tasks(store)[0]}


def _answer(**fields) -> dict:
    base = dict(
        status="stated", value=7.4, text_value=None, unit="usd_per_share", period_end=None,
        passage_id="A:0", quote="the right to receive $7.40 in cash", hedged=False,
        reason="the announcement states it",
    )
    base.update(fields)
    return base


def _response(answer, stop: str = "end_turn") -> dict:
    return {
        "model": "claude-opus-5",
        "stop_reason": stop,
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": json.dumps(answer)}],
    }


def _recording(task_id: str, response: dict, served_by: str = "claude-opus-5") -> Recording:
    return Recording("k" * 64, task_id, {}, response, served_by, {}, "2026-09-16", "batch", "1.6.0")


class _Replayer:
    def __init__(self, stored: Recording | None = None) -> None:
        self.stored = stored
        self.asked: list[dict] = []

    def recording(self, task_id, params):
        self.asked.append(params)
        if self.stored is None:
            raise RecordingMissing(task_id, request_key(params))
        return self.stored


def test_the_request_has_the_spec_s_shape(tasks):
    passages = [Passage("A:0", "A", None, 0, 10, "Some text.")]
    params = build_request(tasks["PAYO:cash_per_share"], passages, SETTINGS)
    assert (params["model"], params["max_tokens"]) == ("claude-opus-5", 16000)
    assert "thinking" not in params
    [system] = params["system"]
    assert system["cache_control"] == {"type": "ephemeral"} and system["text"] == system_prompt()
    assert params["output_config"] == {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
    [message] = params["messages"]
    assert message["role"] == "user"
    assert '<passage id="A:0">' in message["content"]
    assert "How much cash does each share" in message["content"]


def test_every_task_shares_one_cacheable_system_prompt(tasks):
    passages = [Passage("A:0", "A", None, 0, 4, "Text")]
    assert len({json.dumps(build_request(t, passages, SETTINGS)["system"]) for t in tasks.values()}) == 1


def test_the_schema_requires_every_field_and_allows_no_other():
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert set(ANSWER_SCHEMA["required"]) == set(ANSWER_SCHEMA["properties"])


def test_a_well_formed_answer_becomes_a_reading_with_its_model(tasks):
    task = tasks["PAYO:cash_per_share"]
    reading = parse_response(task, _recording(task.task_id, _response(_answer()), served_by="claude-opus-4-8"))
    assert (reading.status, reading.value, reading.unit, reading.served_by, reading.reader) == (
        "stated", 7.4, "usd_per_share", "claude-opus-4-8", "claude",
    )


@pytest.mark.parametrize(
    "response,why",
    [
        ({"model": "m", "stop_reason": "refusal", "stop_details": {"category": "cyber"}, "content": []}, "declined"),
        (_response(_answer(), stop="max_tokens"), "max_tokens"),
        ({"model": "m", "stop_reason": "end_turn", "content": [{"type": "text", "text": "no"}]}, "not JSON"),
        (_response({"status": "stated"}), "schema"),
        (_response(_answer(status="probably")), "schema"),
        (_response(_answer(value="7.40")), "schema"),
        (_response(_answer(unit="dollars")), "schema"),
    ],
)
def test_a_declined_or_malformed_answer_is_refused(tasks, response, why):
    task = tasks["PAYO:cash_per_share"]
    reading = parse_response(task, _recording(task.task_id, response))
    assert reading.status == "refused" and why in reading.reason


def test_no_passages_means_no_request_and_no_statement(tasks):
    replayer = _Replayer()
    reading, recording = ClaudeReader(replayer, SETTINGS).read(tasks["PAYO:cash_per_share"], [])
    assert reading.status == "not_stated" and recording is None and replayer.asked == []


def test_an_unrecorded_request_is_missing_rather_than_answered(tasks):
    passages = [Passage("A:0", "A", None, 0, 4, "Text")]
    reading, recording = ClaudeReader(_Replayer(), SETTINGS).read(tasks["PAYO:cash_per_share"], passages)
    assert reading.status == "missing" and "techval rag record" in reading.reason
    assert recording is None


def test_a_recorded_answer_is_held_to_its_quote(tasks):
    passage = Passage("A:0", "A", None, 0, len(PAYO_SENTENCE), PAYO_SENTENCE)
    good = _recording("PAYO:cash_per_share", _response(_answer()))
    reading, recording = ClaudeReader(_Replayer(good), SETTINGS).read(tasks["PAYO:cash_per_share"], [passage])
    assert reading.status == "stated" and recording is good
    bad = _recording("PAYO:cash_per_share", _response(_answer(value=7.5, quote="the right to receive $7.50 in cash")))
    reading, _ = ClaudeReader(_Replayer(bad), SETTINGS).read(tasks["PAYO:cash_per_share"], [passage])
    assert reading.status == "refused"


def test_the_regex_rules_read_datadog_s_item_7_as_they_always_have(store, tasks):
    text, _ = store.section(store.by_path(KPI_DOCUMENTS["DDOG"]), "7")
    regex = RegexReader()

    def read(metric):
        return regex.read(tasks[f"DDOG:{metric}"], text, "native", [])

    assert read("customers").status == "refused"
    assert read("arr").status == "refused"
    nrr = read("net_revenue_retention")
    assert (nrr.reader, nrr.status, nrr.value, nrr.unit, nrr.hedged) == (
        "regex_native", "stated", 120.0, "percent", True,
    )
    assert read("subscribers").status == "not_stated"


def test_the_regex_rules_read_the_payoneer_announcement(store, tasks):
    text = store.text(store.get("0000950103-26-008945"))
    regex = RegexReader()

    def read(metric):
        return regex.read(tasks[f"PAYO:{metric}"], text, "retrieved", [])

    cash = read("cash_per_share")
    assert (cash.reader, cash.status, cash.value, cash.unit) == ("regex_retrieved", "stated", 7.4, "usd_per_share")
    assert read("consideration_form").text_value == "cash"
    assert read("exchange_ratio").status == "not_stated"
    assert read("agreement_date").text_value == "2026-06-12"
    assert read("acquirer").text_value == "Neon Maple Parent Inc."


def test_a_collared_exchange_ratio_is_refused_by_the_regex_rules(store, tasks):
    text = store.text(store.get("0001104659-26-078482"))
    reading = RegexReader().read(tasks["IRDM:exchange_ratio"], text, "native", [])
    assert reading.status == "refused" and "collar" in reading.reason


def test_a_regex_reading_points_at_the_passage_holding_its_match(store, tasks):
    doc = store.by_path(KPI_DOCUMENTS["DDOG"])
    text, base = store.section(doc, "7")
    passages = chunk(text, accession=doc.accession, item="7", base_offset=base)
    reading = RegexReader().read(tasks["DDOG:net_revenue_retention"], text, "native", passages)
    cited = next(p for p in passages if p.id == reading.passage_id)
    assert reading.quote in cited.text


def test_blank_text_is_not_read(tasks):
    reading = RegexReader().read(tasks["PAYO:acquirer"], "  ", "retrieved", [])
    assert reading.status == "not_stated"
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_reader.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.reader'`.

- [ ] **Step 3: Implement**

`src/techval/rag/reader.py`:

```python
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
        notes = "; ".join(txn.notes)
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_reader.py -q -p no:cacheprovider`
Expected: all pass.

These tests pin values the regex rules produce today. PAYO's agreement date is 2026-06-12 and IRDM's refusal note names the collar; both were measured on the committed fixtures while planning. If a value differs, print the reading and check whether `tmt/precedents.py` changed since `bb16346`. Do not edit the rules to fit the test.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/reader.py tests/rag/test_reader.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Read each fact with a recorded Claude reader and with the regex rules, over the same passages"
```

---
### Task 9: One pass over every task

**Files:**
- Create: `src/techval/rag/run.py`
- Test: `tests/rag/test_run.py`

**Interfaces:**
- Consumes:
  - `chunk` (Task 2) and `BM25` (Task 3)
  - `FilingStore` (Task 5) and `METRICS`, `Task`, `build_tasks` (Task 6)
  - `Recording`, `Replayer` (Task 7)
  - `ClaudeReader`, `RegexReader`, `build_request` (Task 8)
- Produces: `Prepared(task, text, base, passages: tuple[Passage, ...])`, frozen. `text` is the native input, meaning the Item for a KPI and the whole document for a deal, and `base` is its offset in the document.
- Produces: `Run(prepared, unresolved, readings: dict[reader, dict[task_id, Reading]], recordings: dict[task_id, Recording])` with `.missing -> list[str]`
- Produces:
  - `prepare(store, settings) -> tuple[list[Prepared], list[str]]`
  - `requests(prepared, settings) -> list[tuple[str, dict]]`
  - `read_all(store, settings, recordings_root) -> Run`

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_run.py`:

```python
"""One pass: every committed task retrieved once, within its own filing, and read three ways."""

from __future__ import annotations

from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.rag.run import prepare, read_all, requests
from techval.rag.store import FilingStore
from techval.rag.tasks import KPI_DOCUMENTS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SETTINGS = Assumptions().ml.rag


@pytest.fixture(scope="module")
def store() -> FilingStore:
    return FilingStore.from_fixtures(FIXTURES, annual_reports=KPI_DOCUMENTS.values())


@pytest.fixture(scope="module")
def prepared(store):
    return prepare(store, SETTINGS)[0]


@pytest.fixture(scope="module")
def run(store, tmp_path_factory):
    return read_all(store, SETTINGS, tmp_path_factory.mktemp("recordings"))


def test_every_passage_comes_from_its_task_s_own_input(prepared):
    assert len(prepared) >= 49
    for p in prepared:
        assert len(p.passages) <= SETTINGS.top_k
        for x in p.passages:
            assert x.accession == p.task.accession
            assert p.text[x.start_char - p.base : x.end_char - p.base] == x.text


def test_kpis_read_item_7_and_deals_read_the_whole_announcement(store, prepared):
    ddog = next(p for p in prepared if p.task.task_id == "DDOG:customers")
    assert ddog.base > 0 and "dollar-based net retention" in ddog.text
    payo = next(p for p in prepared if p.task.task_id == "PAYO:cash_per_share")
    assert payo.base == 0 and payo.text == store.text(store.get(payo.task.accession))
    assert any("$7.40" in x.text for x in payo.passages)


def test_a_request_is_built_only_where_something_was_retrieved(prepared):
    assert [t for t, _ in requests(prepared, SETTINGS)] == [p.task.task_id for p in prepared if p.passages]


def test_without_recordings_the_claude_reader_is_missing_and_the_regex_readings_stand(run):
    assert run.missing == sorted(p.task.task_id for p in run.prepared if p.passages)
    assert set(run.readings) == {"claude", "regex_retrieved", "regex_native"}
    ids = {p.task.task_id for p in run.prepared}
    assert all(set(by_task) == ids for by_task in run.readings.values())
    assert run.readings["regex_native"]["PAYO:cash_per_share"].value == 7.4
    assert run.recordings == {}


def test_the_same_inputs_give_the_same_readings(store, run, tmp_path):
    assert read_all(store, SETTINGS, tmp_path).readings == run.readings
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_run.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.run'`.

- [ ] **Step 3: Implement**

`src/techval/rag/run.py`:

```python
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_run.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/run.py tests/rag/test_run.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Retrieve each task's passages once and read them with all three readers"
```

---
### Task 10: The answer key

**Files:**
- Create: `src/techval/rag/key.py`
- Test: `tests/rag/test_key.py`

**Interfaces:**
- Consumes: `Prepared` (Task 9), `METRICS` (Task 6) and `check_value`, `locate` (Task 4)
- Produces, constants: `KEY_FILE = "answer_key.csv"`, `PREFILLED`, `TO_FILL`, `COLUMNS`, `KEY_STATUSES` and `START_SLACK = 200`
- Produces: `KeyRow(task_id, status, value=None, text_value=None, period_end=None, quote=None, notes="", resolution=None, span=None)`, frozen
- Produces: `KeyState(rows, unfilled, problems, missing)` with `.complete -> bool`
- Produces, functions:
  - `write_template(path, prepared) -> int`
  - `has_answers(path) -> bool`
  - `load_key(path, prepared) -> KeyState | None`, which returns `None` when the file does not exist

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_key.py`:

```python
"""The answer key: a blind template, and a loader that holds every row to the filing."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from techval.config import Assumptions
from techval.rag.key import COLUMNS, KEY_FILE, TO_FILL, has_answers, load_key, write_template
from techval.rag.run import prepare
from techval.rag.store import FilingStore
from techval.rag.tasks import KPI_DOCUMENTS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def prepared():
    store = FilingStore.from_fixtures(FIXTURES, annual_reports=KPI_DOCUMENTS.values())
    return prepare(store, Assumptions().ml.rag)[0]


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fill(path: Path, answers: dict[str, dict]) -> None:
    rows = _rows(path)
    for row in rows:
        row.update(answers.get(row["task_id"], {}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_the_template_lists_every_task_with_nothing_answered(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    assert write_template(path, prepared) == len(prepared)
    rows = _rows(path)
    assert [r["task_id"] for r in rows] == sorted(p.task.task_id for p in prepared)
    assert list(rows[0]) == list(COLUMNS)
    assert all(r[c] == "" for r in rows for c in TO_FILL)
    assert not has_answers(path)
    payo = next(r for r in rows if r["task_id"] == "PAYO:cash_per_share")
    assert payo["unit"] == "usd_per_share" and payo["filing_url"].startswith("https://www.sec.gov/")
    assert payo["question"].startswith("How much cash does each share")


def test_the_template_shows_no_reader_output(tmp_path, prepared):
    write_template(tmp_path / KEY_FILE, prepared)
    body = (tmp_path / KEY_FILE).read_text(encoding="utf-8").lower()
    assert "claude" not in body and "regex" not in body and "refused" not in body


def test_no_key_file_means_no_key(tmp_path, prepared):
    assert load_key(tmp_path / KEY_FILE, prepared) is None


def test_an_unfilled_key_names_every_blank_row(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    write_template(path, prepared)
    _fill(path, {"PAYO:cash_per_share": {"status": "stated", "value": "7.40", "quote": "the right to receive $7.40 in cash"}})
    state = load_key(path, prepared)
    assert not state.complete and has_answers(path)
    assert set(state.rows) == {"PAYO:cash_per_share"}
    assert len(state.unfilled) == len(prepared) - 1
    assert state.rows["PAYO:cash_per_share"].resolution == pytest.approx(0.01)


def test_every_stated_row_is_held_to_the_filing(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    write_template(path, prepared)
    _fill(
        path,
        {
            "PAYO:cash_per_share": {"status": "stated", "value": "7.50", "quote": "the right to receive $7.40 in cash"},
            "PAYO:agreement_date": {"status": "stated", "text_value": "2026-06-12", "quote": "a sentence the filing never wrote"},
            "PAYO:exchange_ratio": {"status": "maybe"},
            "PAYO:consideration_form": {
                "status": "stated", "text_value": "cash",
                "quote": "the right to receive $7.40 in cash", "start_char": "5",
            },
            "PAYO:acquirer": {
                "status": "stated", "text_value": "Neon Maple Parent Inc.",
                "quote": "Neon Maple Parent Inc., a corporation incorporated pursuant to the laws of Canada",
            },
            "DDOG:arr": {"status": "stated", "value": "seven"},
        },
    )
    state = load_key(path, prepared)
    problems = "\n".join(state.problems)
    assert "PAYO:cash_per_share: the quote does not state 7.5" in problems
    assert "PAYO:agreement_date: the quote is not in the filing" in problems
    assert "PAYO:exchange_ratio: status 'maybe'" in problems
    assert "PAYO:consideration_form: the quote occurs at" in problems
    assert "DDOG:arr: value and start_char must be numbers" in problems
    acquirer = state.rows["PAYO:acquirer"]
    assert acquirer.text_value == "Neon Maple Parent Inc." and acquirer.span[1] > acquirer.span[0]


def test_a_fully_answered_key_is_complete(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    write_template(path, prepared)
    _fill(path, {p.task.task_id: {"status": "not_stated"} for p in prepared})
    state = load_key(path, prepared)
    assert state.complete and all(r.status == "not_stated" for r in state.rows.values())


def test_a_task_the_key_leaves_out_is_reported(tmp_path, prepared):
    path = tmp_path / KEY_FILE
    write_template(path, prepared[:-1])
    state = load_key(path, prepared)
    assert state.missing == [prepared[-1].task.task_id] and not state.complete
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_key.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.key'`.

- [ ] **Step 3: Implement**

`src/techval/rag/key.py`:

```python
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_key.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/key.py tests/rag/test_key.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Write a blind answer-key template and hold every filled row to the filing"
```

---
### Task 11: Scoring both readers against the key

**Files:**
- Create: `src/techval/rag/evaluate.py`
- Test: `tests/rag/test_evaluate.py`

**Interfaces:**
- Consumes:
  - `KeyRow`, `KeyState` (Task 10) and `Run`, `Prepared` (Task 9)
  - `METRICS` (Task 6) and `fold_words` (Task 4)
  - `EvalResult`, `PairedDelta` from `techval.ml.protocol`
- Produces, constants: `ERROR_KINDS = ("wrong_value", "missed", "invented", "wrong_status")`, `REGEX_READERS` and `SIGNIFICANCE = 0.05`
- Produces, single readings:
  - `value_matches(unit, reading, key) -> bool`
  - `is_correct(reading, key, unit) -> bool`
  - `error_kind(reading, key, unit) -> str | None`
- Produces, the test statistic: `mcnemar_p(claude_only, regex_only) -> float`
- Produces, retrieval recall: `recall_at_k(run, key) -> tuple[float | None, int, int]`, which returns the recall, the hits and the number of stated rows
- Produces, per-reader scores: `ReaderScore(reader, n, correct, errors)` with `.accuracy`
- Produces, the comparison: `Comparison(claude, regex, regex_runs, verdicts, claude_only, regex_only, mcnemar_p, verdict_status, evaluation, recall, recall_hits, recall_n)`
- Produces, the entry points:
  - `IncompleteRun(TechvalError)`
  - `compare(run, key) -> Comparison`
  - `verdict_text(comparison) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_evaluate.py`:

```python
"""Scoring: both readers against the key, on the same tasks, with their disagreements tested exactly."""

from __future__ import annotations

from datetime import date

import pytest

from techval.rag.evaluate import (
    IncompleteRun,
    compare,
    error_kind,
    is_correct,
    mcnemar_p,
    recall_at_k,
    verdict_text,
)
from techval.rag.key import KeyRow, KeyState
from techval.rag.passages import Passage
from techval.rag.readings import Reading
from techval.rag.run import Prepared, Run
from techval.rag.tasks import Task

RIGHT = ("stated", 100.0)
WRONG = ("stated", 150.0)
MISS = ("not_stated",)


def _task(i: int) -> Task:
    return Task(f"T{i:02d}:customers", "kpi", f"T{i:02d}", "customers", f"A{i}", "10-K", date(2026, 1, 1), "7", None, None)


def _reading(task: Task, reader: str, status: str, value: float | None = None) -> Reading:
    return Reading(task.task_id, reader, status, value=value, unit="count" if value is not None else None)


def _build(claude, native, retrieved=None, key=None):
    tasks = [_task(i) for i in range(len(claude))]
    passage = lambda t: Passage(f"{t.accession}:0", t.accession, "7", 0, 100, "x" * 100)  # noqa: E731
    prepared = [Prepared(t, "x" * 200, 0, (passage(t),)) for t in tasks]
    retrieved = retrieved or native
    readings = {
        "claude": {t.task_id: _reading(t, "claude", *c) for t, c in zip(tasks, claude)},
        "regex_native": {t.task_id: _reading(t, "regex_native", *c) for t, c in zip(tasks, native)},
        "regex_retrieved": {t.task_id: _reading(t, "regex_retrieved", *c) for t, c in zip(tasks, retrieved)},
    }
    key = key or [RIGHT] * len(tasks)
    rows = {
        t.task_id: KeyRow(
            t.task_id, status, value=value,
            resolution=1.0 if value is not None else None,
            span=(10, 20) if status == "stated" else None,
        )
        for t, (status, value) in zip(tasks, key)
    }
    return Run(prepared, [], readings), KeyState(rows=rows)


def test_a_value_within_half_the_quote_s_last_digit_is_right():
    key = KeyRow("T", "stated", value=100.0, resolution=1.0)
    assert is_correct(Reading("T", "claude", "stated", value=100.4, unit="count"), key, "count")
    assert not is_correct(Reading("T", "claude", "stated", value=100.6, unit="count"), key, "count")


def test_a_refusal_is_right_only_where_the_key_states_nothing():
    refused = Reading("T", "claude", "refused", reason="declined")
    assert is_correct(refused, KeyRow("T", "not_stated"), "count")
    assert is_correct(refused, KeyRow("T", "ambiguous"), "count")
    assert not is_correct(refused, KeyRow("T", "stated", value=1.0, resolution=1.0), "count")


def test_text_folds_case_and_punctuation_and_dates_do_not_fold():
    name = KeyRow("T", "stated", text_value="Neon Maple Parent Inc.")
    assert is_correct(Reading("T", "claude", "stated", text_value="neon maple parent inc", unit="text"), name, "text")
    day = KeyRow("T", "stated", text_value="2026-06-12")
    assert not is_correct(Reading("T", "claude", "stated", text_value="2026-06-13", unit="date"), day, "date")


def test_every_error_has_a_kind():
    stated = KeyRow("T", "stated", value=100.0, resolution=1.0)
    absent = KeyRow("T", "not_stated")
    assert error_kind(Reading("T", "claude", "stated", value=150.0, unit="count"), stated, "count") == "wrong_value"
    assert error_kind(Reading("T", "claude", "not_stated"), stated, "count") == "missed"
    assert error_kind(Reading("T", "claude", "stated", value=5.0, unit="count"), absent, "count") == "invented"
    assert error_kind(Reading("T", "claude", "ambiguous"), absent, "count") == "wrong_status"
    assert error_kind(Reading("T", "claude", "not_stated"), absent, "count") is None


def test_mcnemar_is_exact_and_two_sided():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(8, 0) == pytest.approx(2 * 0.5**8)
    assert mcnemar_p(3, 3) == pytest.approx(1.0)
    assert mcnemar_p(8, 2) == pytest.approx(112 / 1024)


def test_the_comparison_counts_the_disagreements_and_calls_the_verdict():
    claude = [RIGHT] * 10 + [RIGHT] * 8 + [WRONG] * 2
    native = [RIGHT] * 10 + [WRONG] * 8 + [RIGHT] * 2
    c = compare(*_build(claude, native))
    assert (c.claude.correct, c.regex.correct, c.claude_only, c.regex_only) == (18, 12, 8, 2)
    assert c.mcnemar_p == pytest.approx(112 / 1024)
    assert c.verdict_status == "not_significant"
    assert (c.claude.errors["wrong_value"], c.regex.errors["wrong_value"]) == (2, 8)
    assert c.verdicts["T10:customers"] == "Claude only" and c.verdicts["T00:customers"] == "both right"
    e = c.evaluation
    assert (e.metric, e.baseline_name, e.fold_unit, e.paired.n) == ("accuracy", "regex readers", "query", 20)
    text = verdict_text(c)
    assert text.startswith("accuracy of 0.9000 against 0.6000 for regex readers")
    assert "the Claude reader alone is right on 8 and the regex readers alone on 2" in text
    assert "McNemar p of 0.109" in text


def test_a_one_sided_run_of_wins_is_significant_either_way():
    assert compare(*_build([RIGHT] * 10, [WRONG] * 10)).verdict_status == "beats"
    assert compare(*_build([WRONG] * 10, [RIGHT] * 10)).verdict_status == "loses"


def test_the_stronger_regex_run_is_the_baseline():
    c = compare(*_build([RIGHT] * 4, [WRONG] * 4, [RIGHT] * 3 + [WRONG]))
    assert (c.regex.reader, c.regex.correct) == ("regex_retrieved", 3)
    assert set(c.regex_runs) == {"regex_native", "regex_retrieved"}


def test_a_run_with_unrecorded_readings_is_not_scored():
    with pytest.raises(IncompleteRun):
        compare(*_build([("missing",), RIGHT], [RIGHT, RIGHT]))


def test_an_incomplete_key_is_not_scored():
    run, key = _build([RIGHT], [RIGHT])
    key.unfilled.append("T99:customers")
    with pytest.raises(IncompleteRun):
        compare(run, key)


def test_recall_counts_the_stated_rows_whose_quote_a_retrieved_passage_covers():
    run, key = _build([RIGHT, RIGHT, MISS], [RIGHT, RIGHT, MISS], key=[RIGHT, RIGHT, ("not_stated", None)])
    tid = run.prepared[1].task.task_id
    rows = dict(key.rows)
    rows[tid] = KeyRow(tid, "stated", value=100.0, resolution=1.0, span=(150, 160))
    assert recall_at_k(run, KeyState(rows=rows)) == (0.5, 1, 2)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_evaluate.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'techval.rag.evaluate'`.

- [ ] **Step 3: Implement**

`src/techval/rag/evaluate.py`:

```python
"""Both readers scored against the owner's key, on the same tasks.

**What counts as right.**

- A reader is right on a task when its status is the key's and, for a stated
  fact, its value is the key's.
- A number matches within half a unit of the last digit the key's quote shows.
  A date must match exactly. Text is compared after folding case and
  punctuation.
- A refusal is right only where the key states nothing.

**What counts as wrong.** Every wrong answer has one kind:

- a wrong value accepted, the one that reaches a valuation;
- a stated value missed;
- a value invented where the filing states none;
- a wrong status between the answers that state nothing.

**The baseline.** The regex rules are scored twice, on the passages and on the
whole input, and the stronger score is the baseline, so the comparison is
never against a handicapped incumbent.

**The test.** Whether the difference is more than chance is an exact McNemar
test on the tasks where exactly one reader is right. With about sixty tasks, a
real gap can still come out not significant, and the page says so when it
does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

from ..errors import TechvalError
from ..ml.protocol import EvalResult, PairedDelta
from .key import KeyRow, KeyState
from .readings import Reading
from .run import Run
from .tasks import METRICS
from .verify import fold_words

ERROR_KINDS = ("wrong_value", "missed", "invented", "wrong_status")
REGEX_READERS = ("regex_native", "regex_retrieved")
SIGNIFICANCE = 0.05
_VERDICTS = {
    (True, True): "both right",
    (True, False): "Claude only",
    (False, True): "regex only",
    (False, False): "neither",
}


class IncompleteRun(TechvalError):
    """A comparison asked for before every reading is recorded and every key row is filled."""


def value_matches(unit: str, reading: Reading, key: KeyRow) -> bool:
    if unit == "date":
        return (reading.text_value or "") == (key.text_value or "")
    if unit == "text":
        return fold_words(reading.text_value or "") == fold_words(key.text_value or "")
    if reading.value is None or key.value is None:
        return False
    tolerance = key.resolution / 2 if key.resolution else 0.005 * abs(key.value)
    return abs(reading.value - key.value) <= max(tolerance, 1e-9)


def is_correct(reading: Reading, key: KeyRow, unit: str) -> bool:
    if reading.status == "refused":
        return key.status in ("not_stated", "ambiguous")
    if reading.status != key.status:
        return False
    return key.status != "stated" or value_matches(unit, reading, key)


def error_kind(reading: Reading, key: KeyRow, unit: str) -> str | None:
    if is_correct(reading, key, unit):
        return None
    if key.status == "stated":
        return "wrong_value" if reading.status == "stated" else "missed"
    return "invented" if reading.status == "stated" else "wrong_status"


def mcnemar_p(claude_only: int, regex_only: int) -> float:
    discordant = claude_only + regex_only
    if discordant == 0:
        return 1.0
    return float(binomtest(claude_only, discordant, 0.5).pvalue)


@dataclass(frozen=True)
class ReaderScore:
    reader: str
    n: int
    correct: int
    errors: dict[str, int]

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


@dataclass(frozen=True)
class Comparison:
    claude: ReaderScore
    regex: ReaderScore
    regex_runs: dict[str, ReaderScore]
    verdicts: dict[str, str]
    claude_only: int
    regex_only: int
    mcnemar_p: float
    verdict_status: str
    evaluation: EvalResult
    recall: float | None
    recall_hits: int
    recall_n: int


def _score(reader: str, run: Run, key: KeyState) -> tuple[ReaderScore, dict[str, bool]]:
    errors = dict.fromkeys(ERROR_KINDS, 0)
    marks: dict[str, bool] = {}
    for p in run.prepared:
        task_id, unit = p.task.task_id, METRICS[p.task.metric].unit
        reading, row = run.readings[reader][task_id], key.rows[task_id]
        marks[task_id] = is_correct(reading, row, unit)
        if not marks[task_id]:
            errors[error_kind(reading, row, unit)] += 1
    return ReaderScore(reader, len(marks), sum(marks.values()), errors), marks


def recall_at_k(run: Run, key: KeyState) -> tuple[float | None, int, int]:
    hits = n = 0
    for p in run.prepared:
        row = key.rows.get(p.task.task_id)
        if row is None or row.status != "stated" or row.span is None:
            continue
        n += 1
        start, end = row.span
        hits += any(x.start_char < end and start < x.end_char for x in p.passages)
    return (hits / n if n else None), hits, n


def _paired(differences: np.ndarray) -> PairedDelta | None:
    if differences.size < 2:
        return None
    sd = float(np.std(differences, ddof=1))
    standard_error = sd / float(np.sqrt(differences.size))
    return PairedDelta(
        mean=float(differences.mean()),
        sd=sd,
        standard_error=standard_error,
        t=float(differences.mean() / standard_error) if standard_error else float("inf"),
        win_rate=float((differences > 0).mean()),
        n=int(differences.size),
    )


def compare(run: Run, key: KeyState) -> Comparison:
    if run.missing:
        raise IncompleteRun(f"{len(run.missing)} Claude readings have no recording")
    if not key.complete:
        raise IncompleteRun("the answer key is not complete")
    claude, claude_marks = _score("claude", run, key)
    runs = {reader: _score(reader, run, key) for reader in REGEX_READERS}
    best = max(REGEX_READERS, key=lambda r: (runs[r][0].correct, r == "regex_native"))
    regex, regex_marks = runs[best]
    ids = [p.task.task_id for p in run.prepared]
    verdicts = {t: _VERDICTS[(claude_marks[t], regex_marks[t])] for t in ids}
    claude_only = sum(claude_marks[t] and not regex_marks[t] for t in ids)
    regex_only = sum(regex_marks[t] and not claude_marks[t] for t in ids)
    p = mcnemar_p(claude_only, regex_only)
    if p < SIGNIFICANCE:
        status = "beats" if claude_only > regex_only else "loses"
    else:
        status = "not_significant"
    differences = np.array([float(claude_marks[t]) - float(regex_marks[t]) for t in ids])
    evaluation = EvalResult(
        metric="accuracy",
        score=claude.accuracy,
        baseline_name="regex readers",
        baseline_score=regex.accuracy,
        n_observations=len(ids),
        higher_is_better=True,
        fold_unit="query",
        paired=_paired(differences),
        notes=[f"The regex score is the stronger of its two runs, {best.replace('_', ' ')}."],
    )
    recall, hits, n = recall_at_k(run, key)
    return Comparison(
        claude=claude,
        regex=regex,
        regex_runs={reader: score for reader, (score, _) in runs.items()},
        verdicts=verdicts,
        claude_only=claude_only,
        regex_only=regex_only,
        mcnemar_p=p,
        verdict_status=status,
        evaluation=evaluation,
        recall=recall,
        recall_hits=hits,
        recall_n=n,
    )


def verdict_text(c: Comparison) -> str:
    return (
        f"{c.evaluation.verdict()} The readers disagree on {c.claude_only + c.regex_only} of "
        f"{c.claude.n} tasks: the Claude reader alone is right on {c.claude_only} and the "
        f"regex readers alone on {c.regex_only}, an exact McNemar p of {c.mcnemar_p:.3f}."
    )
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_evaluate.py -q -p no:cacheprovider`
Expected: all pass. If `text.startswith(...)` fails, print `c.evaluation.verdict()` and compare it with `EvalResult.verdict` in `src/techval/ml/protocol.py:154-166`. The expected head is `f"{metric} of {score:.4f} against {baseline_score:.4f} for {baseline_name}, a lift of ..."`.

- [ ] **Step 5: Commit**

```bash
git add src/techval/rag/evaluate.py tests/rag/test_evaluate.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Score both readers against the key, with an exact McNemar test on their disagreements"
```

---
### Task 12: The `techval rag` commands

**Files:**
- Create: `src/techval/commands_rag.py`
- Modify: `src/techval/cli.py`. Import beside the other command modules at lines 49-52, and mount after `_mount(_mna_app, "M&A")` at line 144.
- Test: `tests/rag/test_commands.py`

**Interfaces:**
- Consumes: every earlier task
- Produces: the typer app `techval.commands_rag.app`, mounted at `techval rag`, with three commands:
  - `template [--config] [--ml-data] [--key] [--force]`
  - `record [--config] [--ml-data] [--recordings] [--live] [--tasks]`
  - `score [--config] [--ml-data] [--key] [--recordings]`
- Produces two seams that tests monkeypatch: `_client()`, which raises `ConfigError` when the SDK or credentials are missing, and `Recorder`

- [ ] **Step 1: Write the failing tests**

`tests/rag/test_commands.py`:

```python
"""The commands: template and score read committed files only; record is the one door to the API."""

from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from techval import cli as C
from techval import commands_rag as R
from techval.errors import ConfigError

runner = CliRunner()
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _invoke(*args):
    return runner.invoke(C.app, ["rag", *args])


def test_the_rag_group_is_mounted_with_its_three_commands():
    result = _invoke("--help")
    assert result.exit_code == 0
    for name in ("template", "record", "score"):
        assert name in result.output


def test_template_writes_a_blank_key(tmp_path):
    key = tmp_path / "answer_key.csv"
    result = _invoke("template", "--ml-data", str(FIXTURES), "--key", str(key))
    assert result.exit_code == 0, result.output
    assert key.exists() and "rows to" in result.output


def test_template_keeps_answers_unless_forced(tmp_path):
    key = tmp_path / "answer_key.csv"
    _invoke("template", "--ml-data", str(FIXTURES), "--key", str(key))
    with key.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["status"] = "not_stated"
    with key.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = _invoke("template", "--ml-data", str(FIXTURES), "--key", str(key))
    assert result.exit_code == 1 and "--force" in result.output
    assert _invoke("template", "--ml-data", str(FIXTURES), "--key", str(key), "--force").exit_code == 0


def test_score_says_what_it_is_waiting_for(tmp_path):
    result = _invoke(
        "score", "--ml-data", str(FIXTURES),
        "--key", str(tmp_path / "absent.csv"), "--recordings", str(tmp_path / "recordings"),
    )
    assert result.exit_code == 0, result.output
    assert "not recorded" in result.output and "No answer key" in result.output


def test_record_without_the_sdk_says_how_to_install_it(tmp_path, monkeypatch):
    def refuse():
        raise ConfigError("recording needs the Anthropic SDK: pip install 'techval[rag]'")

    monkeypatch.setattr(R, "_client", refuse)
    result = _invoke("record", "--ml-data", str(FIXTURES), "--recordings", str(tmp_path))
    assert result.exit_code == 1 and "techval[rag]" in result.output


def test_record_sends_only_the_tasks_asked_for(tmp_path, monkeypatch):
    sent: list[tuple[str, dict]] = []

    class FakeRecorder:
        def __init__(self, client, root, **kwargs):
            self.failures = [("PAYO:acquirer", "errored")]

        def record(self, requests, live=False):
            sent.extend(requests)
            return []

    monkeypatch.setattr(R, "_client", lambda: object())
    monkeypatch.setattr(R, "Recorder", FakeRecorder)
    result = _invoke(
        "record", "--ml-data", str(FIXTURES), "--recordings", str(tmp_path),
        "--tasks", "PAYO:acquirer, DDOG:arr",
    )
    assert result.exit_code == 0, result.output
    assert sorted(t for t, _ in sent) == ["DDOG:arr", "PAYO:acquirer"]
    assert "PAYO:acquirer" in result.output and "errored" in result.output


def test_record_refuses_a_task_it_does_not_know(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_client", lambda: object())
    result = _invoke("record", "--ml-data", str(FIXTURES), "--recordings", str(tmp_path), "--tasks", "NOPE:arr")
    assert result.exit_code == 1 and "NOPE:arr" in result.output
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/rag/test_commands.py -q -p no:cacheprovider`
Expected: `ImportError` on `from techval import commands_rag`.

- [ ] **Step 3: Implement**

`src/techval/commands_rag.py`:

```python
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
```

In `src/techval/cli.py`, add the import beside the other command modules:

```python
from .commands_rag import app as _rag_app
```

After `_mount(_mna_app, "M&A")`, add:

```python
# The filing reader is a group, not a merge: "template", "record" and "score"
# are too generic to stand at the top level beside "value" and "comps".
app.add_typer(_rag_app, name="rag", rich_help_panel="Filing reader")
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/rag/test_commands.py tests/test_cli_integration.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/techval/commands_rag.py src/techval/cli.py tests/rag/test_commands.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Add techval rag template, record and score"
```

---
### Task 13: The dashboard section "Reading filings"

**Files:**
- Create: `src/techval/dashboard/sections/reading.py`, `src/techval/dashboard/assets/sections/reading.js`
- Modify: `src/techval/dashboard/sections/__init__.py`. In `SECTION_IDS`, insert `"reading"` after `"tmt"`.
- Modify: `src/techval/dashboard/gallery.py`. Insert `"reading"` after `"tmt"` in `PAGE_ORDER`, add `_reading()`, and add it to `gallery_snapshot`'s `rest` after `_tmt()`.
- Modify: `tests/dashboard/test_collect.py:125`. Change `== 10` to `== 11`.
- Modify: `tests/dashboard/test_frontend.py:30-41`. Insert `"reading",` after `"tmt",`.
- Test: `tests/dashboard/test_section_reading.py`

**Interfaces:**
- Consumes: `read_all`, `Run` (Task 9), `load_key`, `KeyState`, `KeyRow` (Task 10) and `compare`, `verdict_text`, `ERROR_KINDS`, `Comparison` (Task 11)
- Produces:
  - `ID = "reading"`, `TITLE = "Reading filings"` and `INPUTS`
  - `ENTRY_POINTS`, `collect(ctx) -> dict`, `shape(run, key, top_k) -> dict`
  - `show(reading, unit) -> str` and `show_key(row, unit) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/dashboard/test_section_reading.py`:

```python
"""The reading section: the readings side by side until a key and recordings exist, a scored comparison after."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from techval.dashboard import collect as C
from techval.dashboard.collect import CollectContext, entry_module
from techval.dashboard.sections import reading as S
from techval.dashboard.snapshot import to_jsonable, validate_section
from techval.rag.key import KeyRow, KeyState
from techval.rag.passages import Passage
from techval.rag.readings import Reading
from techval.rag.recording import Recording
from techval.rag.run import Prepared, Run
from techval.rag.tasks import Task

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
UNRESOLVED = "NET: the 10-K text rag/filing_text_NET_2025.json.gz is not committed"


def _task(i: int) -> Task:
    return Task(f"T{i}:customers", "kpi", f"T{i}", "customers", f"A{i}", "10-K", date(2026, 1, 1), "7", None, None)


def _run(claude_status: str = "stated") -> Run:
    tasks = [_task(i) for i in range(4)]
    prepared = [
        Prepared(t, "x" * 200, 0, (Passage(f"{t.accession}:0", t.accession, "7", 0, 100, "x" * 100),))
        for t in tasks
    ]

    def stated(t, reader, right):
        return Reading(t.task_id, reader, "stated", value=100.0 if right else 150.0, unit="count")

    claude = {
        t.task_id: stated(t, "claude", True) if claude_status == "stated"
        else Reading(t.task_id, "claude", claude_status, reason="not run")
        for t in tasks
    }
    readings = {
        "claude": claude,
        "regex_retrieved": {t.task_id: stated(t, "regex_retrieved", i < 2) for i, t in enumerate(tasks)},
        "regex_native": {t.task_id: stated(t, "regex_native", i < 1) for i, t in enumerate(tasks)},
    }
    recordings = {}
    if claude_status == "stated":
        recordings = {
            t.task_id: Recording(
                "k", t.task_id, {}, {}, "claude-opus-5",
                {"input_tokens": 100, "output_tokens": 20}, "2026-09-16", "batch", "1.6.0",
            )
            for t in tasks
        }
    return Run(prepared, [UNRESOLVED], readings, recordings)


def _key() -> KeyState:
    return KeyState(
        rows={
            f"T{i}:customers": KeyRow(f"T{i}:customers", "stated", value=100.0, resolution=1.0, span=(10, 20))
            for i in range(4)
        }
    )


def _as_section(shaped: dict) -> dict:
    section = to_jsonable(
        {
            "id": S.ID,
            "title": S.TITLE,
            "provenance": [
                {"figure": f, "entry_point": S.ENTRY_POINTS[f], "inputs": [], "seconds": 0.0}
                for f in shaped["figures"]
            ],
            **shaped,
        }
    )
    validate_section(section)
    return section


def _why(section: dict) -> str:
    return " ".join(r["why"] for r in section["refusals"])


def test_a_scored_section_carries_a_headline_and_all_four_figures():
    section = _as_section(S.shape(_run(), _key(), 6))
    assert set(section["figures"]) == {"readings", "errors", "retrieval", "recording"}
    head = section["headline"]
    assert (head["metric"], head["score"], head["baseline_score"], head["n"]) == ("accuracy", 1.0, 0.5, 4)
    assert head["verdict_status"] == "not_significant" and "McNemar" in head["verdict_text"]
    assert section["takeaway"].startswith("On 4 filing facts the Claude reader is right on 4 and the regex readers on 2")
    table = section["figures"]["readings"]["data"]
    assert [c["key"] for c in table["columns"]] == [
        "subject", "fact", "key", "claude", "regex_retrieved", "regex_native", "verdict",
    ]
    assert (table["rows"][0]["claude"], table["rows"][3]["regex_native"]) == ("100", "150")
    assert [r["what"] for r in section["refusals"]] == ["Filings not yet committed"]


def test_without_a_key_the_readings_stand_and_nothing_is_scored():
    section = _as_section(S.shape(_run(), None, 6))
    assert section["headline"] is None
    assert set(section["figures"]) == {"readings", "recording"}
    assert "No answer key is committed" in _why(section)
    assert "key" not in [c["key"] for c in section["figures"]["readings"]["data"]["columns"]]
    assert "no answer key is committed yet" in section["takeaway"]


def test_without_recordings_the_claude_reader_is_named_as_missing():
    section = _as_section(S.shape(_run(claude_status="missing"), _key(), 6))
    assert section["headline"] is None and "recording" not in section["figures"]
    assert "4 of 4 requests have no recording" in _why(section)
    assert "techval rag record" in _why(section)
    assert section["figures"]["readings"]["data"]["rows"][0]["claude"] == "not recorded"


def test_an_incomplete_key_says_what_is_wrong_with_it():
    key = KeyState(unfilled=["T0:customers"], problems=["T1:customers: the quote is not in the filing"])
    why = _why(_as_section(S.shape(_run(), key, 6)))
    assert "1 blank row" in why and "the quote is not in the filing" in why


def test_values_are_shown_in_their_units():
    assert S.show(Reading("T", "claude", "stated", value=1.2e9, unit="usd"), "usd") == "$1.20bn"
    hedged = Reading("T", "claude", "stated", value=120.0, unit="percent", hedged=True)
    assert S.show(hedged, "percent") == "120% (hedged)"
    assert S.show(Reading("T", "claude", "stated", value=7.4, unit="usd_per_share"), "usd_per_share") == "$7.40"
    assert S.show(Reading("T", "claude", "refused", reason="two ratios"), "ratio") == "refused: two ratios"
    assert S.show(Reading("T", "claude", "not_stated"), "count") == "not stated"
    assert S.show_key(KeyRow("T", "ambiguous"), "ratio") == "ambiguous"


def test_collect_runs_offline_on_the_committed_fixtures():
    section, run = C.collect_section(S, CollectContext(root=FIXTURES, repo_root=ROOT), use_cache=False)
    assert run.status == "ok"
    assert len(section["figures"]["readings"]["data"]["rows"]) >= 49
    assert {r["figure"] for r in section["provenance"]} == set(section["figures"])
    for row in section["provenance"]:
        entry_module(row["entry_point"])
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/dashboard/test_section_reading.py -q -p no:cacheprovider`
Expected: `ImportError` on `from techval.dashboard.sections import reading`.

- [ ] **Step 3: Implement the collector**

`src/techval/dashboard/sections/reading.py`:

```python
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
    if run.missing:
        waiting = "the Claude reader's answers are not recorded yet"
    elif key is None:
        waiting = "no answer key is committed yet"
    else:
        waiting = "the answer key is not filled in yet"
    return (
        f"{len(run.prepared)} facts from {_plural(deals, 'merger announcement')} and "
        f"{_plural(reports, 'annual report')} are read by the regex rules and by a recorded Claude "
        f"reader over the same retrieved passages. Neither is scored, because {waiting}."
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
```

`src/techval/dashboard/assets/sections/reading.js`:

```js
/*
 * Section renderer: reading.
 *
 * Four figures, in the order the argument runs:
 *   - every fact as each reader read it;
 *   - the errors by kind;
 *   - how often the search put the answer in front of the readers;
 *   - what the Claude readings were recorded from.
 * Each is a kit kind as it stands (table, hbar, tiles), so the kit's figure
 * loop draws them with their source lines and refusal routing.
 */
(function (TV) {
  "use strict";

  var ORDER = ["readings", "errors", "retrieval", "recording"];

  TV.sections.register("reading", function (root, data) {
    return TV.sections.figures(root, data, { order: ORDER });
  });
})(window.TV);
```

In `src/techval/dashboard/sections/__init__.py`, `SECTION_IDS` becomes:

```python
SECTION_IDS: tuple[str, ...] = (
    "overview",
    "signal",
    "encoder",
    "warranted",
    "fade",
    "propensity",
    "engine",
    "tmt",
    "reading",
    "datalayer",
    "sample",
)
```

In `src/techval/dashboard/gallery.py`, make the same insertion in `PAGE_ORDER`. Then add this function above `gallery_snapshot`:

```python
def _reading() -> dict[str, Any]:
    columns = [
        {"key": "subject", "label": "Filer"},
        {"key": "fact", "label": "Fact"},
        {"key": "key", "label": "Answer key"},
        {"key": "claude", "label": "Claude reader"},
        {"key": "regex_retrieved", "label": "Regex, retrieved passages"},
        {"key": "regex_native", "label": "Regex, whole text"},
        {"key": "verdict", "label": "Verdict"},
    ]
    facts = [
        ("MSFT", "Customers", "41,000", "41,000 (hedged)", "refused: the text states 3 different values", "refused: the text states 3 different values", "Claude only"),
        ("GOOGL", "Cash per share", "$52.00", "$52.00", "$52.00", "$52.00", "both right"),
        ("META", "Exchange ratio", "ambiguous", "ambiguous", "refused: two ratios set a collar", "refused: two ratios set a collar", "both right"),
        ("AMZN", "Annual recurring revenue", "not stated", "$2.40bn", "not stated", "not stated", "regex only"),
        ("NFLX", "Acquirer", "Example Parent Inc.", "Example Parent Inc.", "not stated", "Example Parent Inc.", "both right"),
    ]
    rows = [dict(zip([c["key"] for c in columns], fact)) for fact in facts]
    errors = [
        {"label": f"{reader}, {kind}", "value": value, "role": role}
        for reader, role, values in (("Claude reader", "model", (1, 0, 1, 0)), ("Regex, whole text", "baseline", (0, 1, 0, 0)))
        for kind, value in zip(("wrong value accepted", "stated value missed", "value invented", "wrong status"), values)
    ]
    return _section(
        "reading",
        "Reading filings",
        "On 5 filing facts the Claude reader is right on 4 and the regex readers on 4; they disagree on 2.",
        headline=_headline(
            "accuracy", 0.8, "regex readers", 0.8, 5, True, "not_significant",
            "accuracy of 0.8000 against 0.8000 for regex readers: the model does NOT beat the baseline on 5 "
            "observations. Use the baseline. The readers disagree on 2 of 5 tasks.",
        ),
        figures={
            "readings": _figure(
                "table", "The Claude reader is right on 4 of 5 filing facts, the regex readers on 4",
                "Each fact is asked of its own filing, by both readers, over the same passages",
                {"columns": columns, "rows": rows}, wide=True,
            ),
            "errors": _figure(
                "hbar", "The Claude reader accepted 1 wrong value, the regex readers 0",
                "Every wrong answer by kind",
                {"rows": errors, "format": "int", "valueLabel": "Tasks", "labelHeader": "Reader and error",
                 "roleLabels": {"model": "Claude reader", "baseline": "Regex readers"}},
            ),
            "retrieval": _figure(
                "tiles", "The search put the answer in front of the readers on 3 of 3 stated facts",
                "A stated fact counts when the key's quote overlaps a retrieved passage",
                {"tiles": [{"label": "Stated facts with the quote in the top 6 passages", "value": 1.0, "format": "pct:0", "sub": "3 of 3"}]},
            ),
            "recording": _figure(
                "tiles", "5 recorded answers from claude-opus-5, replayed rather than requested",
                "What the Claude readings were recorded from",
                {"tiles": [
                    {"label": "Answers replayed", "value": 5, "format": "int"},
                    {"label": "Answered by", "value": "claude-opus-5"},
                    {"label": "Input tokens", "value": 15250, "format": "int"},
                ]},
            ),
        },
    )
```

In `gallery_snapshot`, add `_reading()` to the `rest` list, directly after `_tmt()`.

Make the two test edits: 10 becomes 11 in `tests/dashboard/test_collect.py`, and `"reading",` goes after `"tmt",` in the `SECTION_IDS` tuple of `tests/dashboard/test_frontend.py`.

- [ ] **Step 4: Run the dashboard tests**

Run: `.venv/bin/python -m pytest tests/dashboard -q -p no:cacheprovider`
Expected: all pass. The verdict test in `test_frontend.py` holds the gallery's first sentence to the verdict text verbatim. If it fails on the reading section, make the gallery's `verdict_text` start exactly as `EvalResult.verdict()` would for those numbers.

- [ ] **Step 5: Commit**

```bash
git add src/techval/dashboard/sections/reading.py src/techval/dashboard/assets/sections/reading.js src/techval/dashboard/sections/__init__.py src/techval/dashboard/gallery.py tests/dashboard/test_section_reading.py tests/dashboard/test_collect.py tests/dashboard/test_frontend.py
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Add a Reading filings section that shows both readers and scores them once the key and recordings exist"
```

---
### Task 14: Record the NET, NFLX and TMUS 10-K texts

**Files:**
- Create: `tests/fixtures/rag/record_filing_text.py`
- Create, by running the script: `tests/fixtures/rag/filing_text_NET_2025.json.gz`, `tests/fixtures/rag/filing_text_NFLX_2025.json.gz`, `tests/fixtures/rag/filing_text_TMUS_2025.json.gz`
- Test: `tests/rag/test_fixture_recorder.py`

**Interfaces:**
- Consumes:
  - `EdgarClient(knowledge_date=...)`, with `.filings(ticker, forms, limit)`, `.filing_text(ticker, filing)` and `.ticker_to_cik(ticker)`
  - `KPI_DOCUMENTS` (Task 6)
- Produces: three payloads in the DDOG layout. Each carries `retrieved`, `source`, `url`, `ticker`, `accession`, `form`, `filed`, `period` and `text`.

- [ ] **Step 1: Write the failing test**

`tests/rag/test_fixture_recorder.py`:

```python
"""The fixture recorder writes exactly the files the KPI tasks look for."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from techval.rag.tasks import KPI_DOCUMENTS

SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "rag" / "record_filing_text.py"


def test_the_recorder_writes_the_files_the_tasks_expect():
    spec = importlib.util.spec_from_file_location("record_filing_text", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = {f"rag/filing_text_{t}_{module.FISCAL_YEAR}.json.gz" for t in module.TICKERS}
    assert written == {path for ticker, path in KPI_DOCUMENTS.items() if ticker != "DDOG"}
```

- [ ] **Step 2: Run the test to see it fail**

Run: `.venv/bin/python -m pytest tests/rag/test_fixture_recorder.py -q -p no:cacheprovider`
Expected: FAIL. `spec_from_file_location` finds no file, so `spec.loader` raises.

- [ ] **Step 3: Implement the script**

`tests/fixtures/rag/record_filing_text.py`:

```python
"""Record the 10-K texts the filing reader's KPI tasks read: NET, NFLX and TMUS.

Run from the repository root, with the contact address EDGAR's fair-access
policy asks for:

    TECHVAL_SEC_EMAIL=you@example.com .venv/bin/python tests/fixtures/rag/record_filing_text.py

Each file holds the newest 10-K the filer made on or before AS_OF. The text is
stripped of markup by techval.edgar.strip_markup and stored in full, in the
layout of tests/fixtures/filing_text_DDOG_2025.json.gz, plus the document's
URL. The gzip header carries no timestamp, so a re-recording of an unchanged
filing is byte-identical.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import date
from pathlib import Path

from techval.edgar import EdgarClient

AS_OF = date(2026, 9, 11)
FISCAL_YEAR = 2025
TICKERS = ("NET", "NFLX", "TMUS")
HERE = Path(__file__).resolve().parent


def record(client: EdgarClient, ticker: str) -> Path:
    [filing] = client.filings(ticker, forms=("10-K",), limit=1)
    if not str(filing["period"]).startswith(str(FISCAL_YEAR)):
        raise SystemExit(
            f"{ticker}: the newest 10-K on or before {AS_OF} covers {filing['period']}, not fiscal {FISCAL_YEAR}"
        )
    bare = filing["accession"].replace("-", "")
    payload = {
        "retrieved": date.today().isoformat(),
        "source": "SEC EDGAR primary document, markup stripped by techval.edgar.strip_markup, stored verbatim and in full",
        "url": f"https://www.sec.gov/Archives/edgar/data/{client.ticker_to_cik(ticker)}/{bare}/{filing['document']}",
        "ticker": ticker,
        "accession": filing["accession"],
        "form": filing["form"],
        "filed": filing["filed"].isoformat(),
        "period": filing["period"],
        "text": client.filing_text(ticker, filing),
    }
    path = HERE / f"filing_text_{ticker}_{FISCAL_YEAR}.json.gz"
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return path


def main() -> int:
    client = EdgarClient(knowledge_date=AS_OF)
    for ticker in TICKERS:
        print(record(client, ticker))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to see it pass**

Run: `.venv/bin/python -m pytest tests/rag/test_fixture_recorder.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Record the three texts. This step needs the owner.**

This makes live requests to SEC EDGAR and needs `TECHVAL_SEC_EMAIL`, which is not set in this environment. Ask the owner whether to run it and which contact address to use. Do not invent one. If they agree, run:

```bash
TECHVAL_SEC_EMAIL=<the owner's address> .venv/bin/python tests/fixtures/rag/record_filing_text.py
```

Expected output: three written paths. Then check that each file splits and has an Item 7:

```bash
.venv/bin/python -c "
from pathlib import Path
from techval.rag.store import FilingStore
from techval.rag.tasks import KPI_DOCUMENTS
store = FilingStore.from_fixtures(Path('tests/fixtures'), annual_reports=KPI_DOCUMENTS.values())
for t, p in KPI_DOCUMENTS.items():
    d = store.by_path(p)
    print(t, d and (d.accession, d.filed, len(store.section(d, '7')[0])))
"
```

Expected: four lines, each with an accession, a 2026 filing date and an Item 7 length above 10,000.

If the owner declines, skip this step. The page and the key template then list NET, NFLX and TMUS as unresolved, and everything else still runs.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/rag/record_filing_text.py tests/rag/test_fixture_recorder.py tests/fixtures/rag/filing_text_*_2025.json.gz
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Record the NET, NFLX and TMUS 10-K texts the KPI reading tasks ask of"
```

If Step 5 was skipped, the `git add` of the `.json.gz` glob fails. Leave it out of the command.

---

### Task 15: Template, recordings, documentation, and the page

**Files:**
- Create, by running a command: `tests/fixtures/rag/answer_key.csv`
- Create, only if the owner approves: `tests/fixtures/rag/recordings/*.json`
- Modify: `README.md`. Add a `### Reading filings` subsection at the end of `## The results dashboard`, directly before `## Install` (line 1415 at `bb16346`).
- Modify: `docs/dashboard/snapshot.json`, `docs/dashboard/index.html`

- [ ] **Step 1: Write the blank key template**

Run: `.venv/bin/techval rag template`
Expected: `Wrote N rows to tests/fixtures/rag/answer_key.csv.`, with N = 49 plus 12 when Task 14 Step 5 ran. Any unresolved filings are listed after it.

- [ ] **Step 2: Record the Claude answers. This step needs the owner.**

This spends money on the owner's Anthropic account (about $1 to $2) and needs `pip install 'techval[rag]'` plus credentials. Neither is present in this environment. Ask the owner whether to record now. If they agree and provide credentials, run:

```bash
uv pip install --python .venv/bin/python 'anthropic>=1.6,<2'
.venv/bin/techval rag record
```

Expected: `Recorded N of N requests as a batch; ...`, with any failures listed after it. If the owner declines, skip this step. The section then shows the regex readings and the "not recorded" refusal.

- [ ] **Step 3: Document the commands**

Insert into `README.md`, directly before the line `## Install`:

```markdown
### Reading filings

`techval rag` reads facts that filings state in prose: operating figures in a
10-K's Item 7, and the terms of a merger announcement. It answers each question
three ways. A Claude reader reads the passages BM25 retrieves, and the regex
rules techval already has run on the same passages and again on the whole
text. A Claude answer counts only if its quote is in the passage it cites and
its value is in that quote.

    techval rag template   write the blank answer key to tests/fixtures/rag/answer_key.csv
    techval rag record     record the Claude reader's answers (pip install 'techval[rag]',
                           Anthropic credentials, about one to two dollars a full run)
    techval rag score      score both readers against the filled-in key

The Claude reader's requests and responses are committed under
`tests/fixtures/rag/recordings/` and replayed, so the Reading filings section and
the tests run offline with no SDK and no key. The section scores nothing until
the key is filled in and every request is recorded. The design is in
`docs/superpowers/specs/2026-09-16-rag-filing-reader-design.md`.
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests -q -p no:cacheprovider`
Expected: every test passes. That is 2,106 plus the new rag and section tests.

- [ ] **Step 5: Recollect and render the dashboard**

Run: `.venv/bin/techval dashboard --config docs/dashboard/assumptions.yaml --collect`
Expected: a `reading ok` line, and `Wrote docs/dashboard/snapshot.json` and `Wrote docs/dashboard/index.html`.

Then check the section:

```bash
.venv/bin/python -c "
import json
s = json.load(open('docs/dashboard/snapshot.json'))['sections']['reading']
print(s['status'], len(s['figures']['readings']['data']['rows']))
print([r['what'] for r in s['refusals']])
"
```

Expected: `ok`, the task count, and refusals that match what Steps 1 and 2 left undone.

Screenshot the section in both themes with headless Chrome and read the images. Check for clipping or overflow in the wide table. `?theme=light` and `?theme=dark` force the theme; do not use a `#reading` hash with a tall window, because that screenshot comes out blank.

- [ ] **Step 6: Commit, push, and update PR #59**

```bash
git add tests/fixtures/rag/answer_key.csv README.md docs/dashboard/snapshot.json docs/dashboard/index.html
git add tests/fixtures/rag/recordings  # only if Step 2 ran
git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit -m "Commit the blank answer key, document techval rag, and render the Reading filings section"
git push origin rag/filing-reader
```

Update the PR description with `gh pr edit 59 --body-file <file>`. The new body should:
- say the implementation is on the branch beside the design;
- list what was recorded and what is still waiting on the owner (the key, and the recordings if Step 2 was skipped);
- carry no Claude attribution.

Do not merge until the owner says so.

---

## Self-review against the spec

| Spec requirement | Task |
|---|---|
| Point-in-time store, `filed <= as_of` | 5 |
| 1,200-character passages, 200 overlap, sentence snapping, range identity | 2 |
| BM25 with in-document IDF, top 6, offset tie-break | 3 |
| Claude request: `claude-opus-5`, default thinking, cached system block, JSON schema | 8 |
| Batch recording without fallbacks, live recording with `fallbacks: "default"` | 7 |
| Request key over the full parameters, stored-request check, `RecordingMissing` | 7 |
| Verbatim quote, value in quote, money floor, hedge rung | 4 |
| 61 tasks (45 deal, 16 KPI) and unresolved filings named | 6, 14 |
| Blind template, key checked by the same rules | 10 |
| Correctness, error kinds, recall at k, McNemar, `EvalResult` headline, strongest regex run | 11 |
| Dashboard section with refusals before the key and recordings exist | 13 |
| CLI template, record, score; `anthropic` optional extra | 1, 12 |
| No live calls in tests or collection | 7, 8, 9, 13 |
| Slice 2 excluded: nothing feeds valuations, labels or models | all (no task touches kpis, precedents or mna outputs) |
