# Retrieval-augmented filing reader: design

Date: 2026-09-16. Status: approved in conversation, awaiting review of this document.

## Decisions taken

| Question | Decision |
|---|---|
| Does the retrieval layer include a language model? | Yes. Claude reads retrieved passages. Every request and response is recorded as a committed fixture, so the dashboard and the tests reproduce offline with no SDK and no key. A key is needed only to re-record. |
| Which process does the first slice improve? | Extraction: operating KPIs stated only in prose, and merger deal terms. |
| How is the reader judged? | Against an answer key written by the owner of this repository, from a blank template that shows no reader output. |
| How large is the first key? | The committed filings only: 61 rows. |
| How does the Claude reader relate to the regex readers? | Side by side. Both read the same retrieved passages for every task and both are scored. Readings reach valuations and labels only in a second slice, and only if the score supports it. |

## Goal

Measure whether a retrieval-augmented Claude reader extracts filing facts more accurately than the regex readers techval uses today, on a fixed, owner-written answer key, with every number on the page reproducible offline.

## Non-goals

- Feeding reader output into `kpis.build_kpis`, `precedents.extract_transaction`, the M&A labels or any model. That is slice 2, with its own spec, and it depends on this slice's score.
- Passage retrieval for the peer encoder. The store and the retriever built here are meant to serve it later.
- Any Claude output used as a model feature. Claude has read about events after most fold dates, so a generated feature can carry hindsight into a walk-forward score even with names removed.
- Live calls during collection, rendering or tests.

## Background

Facts measured before this design, read-only, at `bb16346`:

- `tmt/kpis.py` reads prose with ordered regex rules (`_TEXT_RULES`, kpis.py:1357-1489) and refuses:
  - a rate with no percent sign;
  - money with no scale word;
  - two distinct values in one text.

  Two known misreads led to a downstream gate that refuses every population count read from prose (commands_tmt.py:319-363):
  - Disney's "27.1 million subscribers" is the prior-year half of a bundle-overlap footnote.
  - T-Mobile's "added 3,287,000 postpaid phone customers" is net additions.
- `tmt/precedents.py` reads merger filings with regex:
  - the conversion clause (precedents.py:538-553, 1079-1121);
  - cash from the granting sentence;
  - exchange ratios, refusing on more than one distinct ratio (1144-1179);
  - the acquirer from a 900-character window;
  - the agreement date with a 220-character gap allowance.

  IRDM's offer price is refused because the agreement states two exchange ratios as a collar.
- Only one 10-K text among the KPI filers is committed: `filing_text_DDOG_2025.json.gz`, 405,446 characters. The KPI filers are DDOG, NET, NFLX and TMUS (dashboard/sections/tmt.py:70-73).
- Merger texts are committed under `tests/fixtures/merger/text/`, 10 files, 1.2 MB, for the 9 deals IRDM, MNDT, PAYO, RAMP, ROKU, SLAB, SPLK, WORK and ZEN.
- `nlp/sections.py` splits a 10-K into Items and keeps `text == raw[start_char:end_char]` for every section (sections.py:211-226), so any passage can be quoted back by character range.
- No labelled set exists for either reader.
- The project depends on requests, pandas, numpy, matplotlib, pydantic, typer, pyyaml, rich, scikit-learn, scipy and lxml, and on no LLM provider.

## Architecture

A new package, `src/techval/rag/`. Each module has one job and a small interface.

| Module | Responsibility | Interface | Depends on |
|---|---|---|---|
| `store.py` | Committed filing texts with their metadata. | `FilingStore.from_fixtures(root)`; `store.documents(subject, as_of, forms=None) -> list[Document]` returns only documents with `filed <= as_of`; `store.section(document, item)` for 10-K Items. | fixtures, `nlp.sections` |
| `passages.py` | Cuts a document or section into passages. | `chunk(text, *, base_offset, size, overlap) -> list[Passage]`. A `Passage` has `id`, `accession`, `item`, `start_char`, `end_char`, `text`. | none |
| `retrieve.py` | BM25 keyword search over one document's passages. | `BM25(passages).top_k(query_terms, k) -> list[Passage]` | `passages` |
| `reader.py` | The two readers, one interface. | `Reader.read(task, passages) -> Reading`. `ClaudeReader(replayer)` and `RegexReader()`. | `retrieve`, `recording`, `tmt.kpis`, `tmt.precedents` |
| `recording.py` | Request keys, replay and record. | `request_key(params) -> str`; `Replayer(root).response(params)`; `Recorder(client, root).record(requests, live=False)` | none (the SDK only inside `Recorder`) |
| `verify.py` | Holds a reading, or a key row, to its quote. | `verify(reading, passages) -> Reading` (possibly refused, with a reason) | `passages` |
| `tasks.py` | The 61 tasks. | `TASKS: tuple[Task, ...]` | `store` |
| `key.py` | The answer key's template, loader and checker. | `write_template(path)`; `load_key(path, store) -> dict[task_id, KeyRow]` | `store`, `verify` |
| `evaluate.py` | Scores both readers against the key. | `score(readings_by_reader, key) -> ReaderScore` wrapping `ml.protocol.EvalResult` | `ml.protocol`, `scipy` |

Around the package:

- **CLI**, in a new `commands_rag.py` registered on the existing typer app:
  - `techval rag template` writes the key template.
  - `techval rag record [--live] [--tasks ...]` records Claude responses.
  - `techval rag score` prints both readers' scores.
- **Dashboard section `reading`**, titled "Reading filings", with a collector in `dashboard/sections/reading.py` and a renderer in `assets/sections/reading.js`.
- **Fixtures** under `tests/fixtures/rag/`:
  - `recordings/`: one JSON file per request.
  - `answer_key.csv`: the template, filled in by the owner.
  - `filing_text_NET_*.json.gz`, `filing_text_NFLX_*.json.gz` and `filing_text_TMUS_*.json.gz`: the latest 10-K on or before the TMT collection date, in the same format as DDOG's, written by a committed `record_filing_text.py` that uses `EdgarClient` and requires `TECHVAL_SEC_EMAIL`.
- **Configuration.** A `RagAssumptions` block on `MLAssumptions` in `config.py`:

  | Key | Default |
  |---|---|
  | `model` | `claude-opus-5` |
  | `passage_chars` | 1,200 |
  | `overlap_chars` | 200 |
  | `top_k` | 6 |
  | `max_tokens` | 16,000 |

  Every value that can change an answer enters the request key.
- **Dependencies.** `anthropic` becomes the optional extra `techval[rag]`. It is imported inside `Recorder` only. Replay reads plain JSON.

## Data flow

1. `tasks.py` defines each task: `task_id`, `kind` (`kpi` or `deal`), `subject` (a ticker or a deal), `accession`, `as_of` (the document's filing date), `item` (`7` for KPIs, none for merger texts), `metric`, and `question`, the exact wording both readers get.
2. `store.documents(subject, as_of)` returns the document, and `store.section` narrows a 10-K to its Item.
3. `passages.chunk` cuts it into windows of about 1,200 characters with 200 characters of overlap. Window ends move back to the nearest sentence end inside the last 150 characters. Each passage checks `text == source[start_char:end_char]`, or construction fails.
4. `BM25` ranks the passages with a fixed query vocabulary per metric, for example `customers`: customer, customers, organizations, paying, count. IDF is computed over that one document's passages, and ties break on `start_char`. The top 6 go to both readers.
5. `RegexReader` runs the existing rule for the metric over the concatenated top passages:
   - `kpis.extract_from_text` for KPIs;
   - the `precedents` sentence and clause readers for deal terms.

   A value the rules produce is mapped back to the passage and the character range of its match.

   The regex readers are also run on their native input, the whole Item or document, since cutting their input to six passages could handicap them. Both runs are scored, and the headline takes whichever regex score is higher, so the baseline is always the strongest version of the regex readers.
6. `ClaudeReader` builds the request, fetches the response from the `Replayer`, and parses the JSON.
7. `verify` holds every reading to its quote.
8. `evaluate.score` compares the readings with the key. The dashboard collector records each step with `ctx.record`, and the inputs are the text fixtures, the recordings directory and the key.

## The Claude request

- `model`: from configuration, `claude-opus-5`. Thinking is left at the model's default (adaptive), and effort at the default.
- `system`: a fixed block marked `cache_control: {"type": "ephemeral"}`. It holds:
  - the reading rules: answer only from the passages; quote the sentence verbatim; say `not_stated` when the passages do not state the metric; say `ambiguous` when they state more than one candidate and do not say which is meant; flag hedges;
  - the definitions of every metric in `tasks.py`.

  If the block is shorter than the model's minimum cacheable prefix, it is sent uncached, which changes cost and nothing else.
- `messages`: one user turn with the passages, each tagged with its `passage_id`, then the question.
- `output_config.format`: a JSON schema with `additionalProperties: false`:

  | Field | Type |
  |---|---|
  | `status` | enum `stated`, `not_stated`, `ambiguous` |
  | `value` | number or null |
  | `text_value` | string or null, for names, dates and forms of consideration |
  | `unit` | enum `count`, `usd`, `usd_per_share`, `percent`, `ratio`, `date`, `text`, or null |
  | `period_end` | string or null |
  | `passage_id` | string or null |
  | `quote` | string or null |
  | `hedged` | boolean |
  | `reason` | string |

  Structured output cannot be combined with the API's citations feature, so grounding is checked locally by `verify`.
- **Recording by batch**, the default:
  - `client.messages.batches.create` with one request per task, `custom_id` set to the request key.
  - Results are matched by `custom_id`, never by position.
  - The batch API does not accept the server-side fallback option, so a declined request is recorded as declined and scored as a refusal.
- **Recording live** (`--live`), one request at a time:
  - server-side fallbacks on: beta `server-side-fallback-2026-07-01` with `fallbacks: "default"`;
  - the recording stores the model that served the answer;
  - intended for re-recording a handful of tasks.
- **Cost:** 61 requests of roughly 3,000 input and 1,500 output tokens, at batch prices, is about 1 to 2 US dollars per full recording.
- **Credentials:** `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile, resolved by the SDK. `Recorder` fails with instructions when none is found. No credential is ever written to a recording or a log.

## Recordings

- The request key is the sha256 of the canonical JSON of the full request parameters, with sorted keys and no whitespace. The parameters are model, max_tokens, thinking, output_config, system and messages.
- Each file is `tests/fixtures/rag/recordings/<key>.json` and holds:

  | Field | Contents |
  |---|---|
  | `key` | the request key |
  | `task_id` | the task |
  | `request` | the full request parameters |
  | `response` | the full message as a dict |
  | `served_by` | the model that answered |
  | `usage` | token usage |
  | `recorded_at` | recording date |
  | `mode` | `batch` or `live` |
  | `sdk_version` | the SDK version used |

- `Replayer.response(params)` loads the file for `request_key(params)`, checks that the stored request equals `params`, and returns the response. A missing file raises `RecordingMissing(task_id, key)`, and nothing falls back to a live call.
- The dashboard section turns `RecordingMissing` into a refusal naming the tasks and the command that records them.

## Verification

`verify` applies the same rules to a reading and to a key row. A `stated` answer is held only if all four hold:

1. `passage_id` names one of the passages given (a key row cites a character range in the filing instead).
2. `quote` appears in that passage. Whitespace runs are collapsed on both sides, and nothing else is normalised.
3. The value parses out of the quote:
   - numbers with thousands separators;
   - scale words (thousand, million, billion) and the `mm`/`bn` shorthand;
   - percentages;
   - per-share dollar amounts;
   - dates in the forms filings use.

   Text values must appear in the quote.
4. For money, a figure below 1,000,000 with no scale word is refused, as `kpis.py` already does.

Otherwise the reading becomes `refused` with the failed rule as its reason. A `hedged` reading, or one whose quote carries "approximately", "about", "more than", "over" or "nearly" before the value, takes the existing 0.4 confidence rung. `not_stated` and `ambiguous` pass through with their reason.

## Tasks and the answer key

61 tasks:

- **45 deal rows:** 9 deals times 5 terms. The terms are:
  - offer value per target share (`usd_per_share`);
  - form of consideration (`text`: cash, stock or mixed);
  - exchange ratio or collar terms (`ratio` or `text`);
  - agreement date (`date`);
  - acquirer (`text`).
- **16 KPI rows:** DDOG, NET, NFLX and TMUS times 4 metrics: customer count, annual recurring revenue, net revenue retention, and subscriber count. Rows a filer does not disclose are kept. Their true answer is `not_stated`.

`techval rag template` writes `tests/fixtures/rag/answer_key.csv`.

- **Pre-filled:** `task_id`, `kind`, `subject`, `accession`, `form`, `filed`, `item`, `metric`, `question` and `filing_url`.
- **Blank for the owner:** `status`, `value`, `text_value`, `unit`, `period_end`, `quote`, `start_char` and `notes`.
- **Blind:** the template never contains any reader output.

`load_key` refuses the key unless:

- every row has a status;
- every `stated` row has a quote that appears in the task's document, or its Item for a 10-K, and, where `start_char` is filled, an occurrence of the quote begins within 200 characters of it;
- the value parses out of that quote under the rules above.

An unfilled key is reported as unfilled, row by row.

## Scoring

- **A reader is correct on a task when:**
  - its status equals the key's; and
  - for `stated` rows, its value equals the key's:
    - numbers after unit and scale, with a tolerance of half a unit in the last digit the quote shows;
    - dates exactly;
    - text after case and punctuation folding.

  A refused reading is correct only where the key says `not_stated` or `ambiguous`.
- **Errors are counted in three kinds:**

  | Kind | Meaning |
  |---|---|
  | wrong value accepted | the most dangerous |
  | stated value missed | the reader refused or said `not_stated` where the key states a value |
  | value invented | the reader stated a value where the key says `not_stated` |

- **Retrieval recall at 6** is the share of `stated` key rows whose quoted range overlaps a retrieved passage. It separates search failures from reading failures.
- **The headline** is an `EvalResult`:
  - `metric="accuracy"`, `score` the Claude reader's accuracy;
  - `baseline_name="regex readers"`, `baseline_score` the higher of their two scores;
  - `n_observations=61`, `fold_unit="query"`;
  - `paired` the `PairedDelta` of per-task correctness, Claude minus regex.
- **The section's verdict** uses an exact McNemar test on the tasks where exactly one reader is correct (`scipy.stats.binomtest`):
  - p < 0.05 with Claude ahead: `beats`;
  - p < 0.05 with the regex readers ahead: `loses`;
  - otherwise: `not_significant`.

## Dashboard section: Reading filings

- **Takeaway:** both accuracies, the number of disagreements, the McNemar p, and the count of wrong values each reader accepted.
- **Headline strip:** the `EvalResult` above.
- **Figures:**
  - a per-task table: subject, metric, key value, regex reading on the retrieved passages, regex reading on the whole input, Claude reading with its quote, and verdict;
  - an error breakdown by kind, per reader;
  - retrieval recall at 6;
  - a recording basis: model, recording date, total input and output tokens, and which model served each answer.
- **Refusals:**
  - No key committed: "No answer key is committed, so neither reader is scored." The table still shows both readings.
  - Missing recordings: the tasks, and the command to record them.
- **Wiring:** the section is added to `SECTION_IDS` and to the gallery's page order with a synthetic section.

## Doctrine and failure handling

- **Point in time:** `store.documents` filters `filed <= as_of`, and each task's `as_of` is its document's filing date. Merger texts are read for deal terms only and never reach M&A features.
- **Offline:** collection, rendering and tests replay only. Nothing outside `Recorder` imports the SDK.
- **Refusals are stated:** failed verification, schema mismatch, a declined request, a missing recording and a disagreement between readers each carry a reason on the page.
- **No new confidence rungs:** stated readings take the text rung (0.75), hedged readings 0.4, refusals 0.0.
- **Determinism:** the same recordings and fixtures produce the same page bytes.

## Testing

No test calls the API. The tests cover:

- **`passages`:** the range invariant, window ends on sentence boundaries, and full coverage of the source.
- **`retrieve`:** a fixed ranking on a small corpus, tie-breaking by offset, and IDF computed within the document.
- **`verify`:** cases taken from the committed fixtures:
  - DDOG's three "approximately ... customers" counts, which must come out `ambiguous` or a hedged single value, never a silent pick;
  - DDOG's "$100,000" ARR with no scale word;
  - IRDM's two exchange ratios;
  - a quote not in the passage;
  - a value not in the quote.
- **`recording`:** key stability across dict orderings, `RecordingMissing` on a missing file, refusal on a stored request that differs, and batch recording against a fake client that returns results out of order.
- **`key`:** template columns and blindness, the unfilled report, and the rejection of a quote not in the filing.
- **`evaluate`:** correctness rules, the three error kinds, retrieval recall, and McNemar on known discordant counts.
- **`reader`:** `RegexReader` agrees with `kpis.extract_from_text` on DDOG's Item 7 where both see the same text. `ClaudeReader` parses a recorded response and refuses a malformed one.
- **Dashboard:** the `reading` section against a fake recording set and a fake key, the no-key refusal, and the gallery section drawn by the renderer.
- **Full suite:** green before every commit.

## Slice 2, for reference only

If the key shows the Claude reader beats the regex readers, a second spec will feed verified readings into:

- `kpis.build_kpis`, under the existing text rung;
- `precedents.extract_transaction`, for IRDM's collar and the refused deal terms;
- a reader-built M&A label set. The propensity model would be refit on it and reported beside the current labels, walk-forward.

Nothing in this slice changes a valuation, a label or a model score.

## Risks

- **A small key:** 61 rows can leave the comparison not significant. The page will say so rather than claim a win.
- **Retrieval misses cap both readers equally.** Recall at 6 is reported so the cap is visible.
- **Recording drift:** any change to the prompt, the passages or the configuration changes the keys and requires a new recording, which is the intended behaviour.
- **Cost:** a full recording costs about 1 to 2 US dollars, and re-recording is manual.
