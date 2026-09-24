# App Chrome and Investing Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the results dashboard into financial-app chrome, then build `techval invest`, a portfolio tracker and engine-read dashboard rendered as one self-contained page.

**Architecture:** Wave R restyles the existing dashboard in place: new ink-teal tokens (dark default), cards, sticky tabs, verdict tiles, delta colouring, gated motion; content and snapshot schema untouched. Wave I adds `src/techval/invest/`: a YAML transaction ledger, quotes through the existing market sources, deterministic performance/hygiene/engine/ideas computations, a validated snapshot, and a renderer that inlines the dashboard's `kit.js` plus new app assets into `portfolio/index.html`. A fictional fixture portfolio drives tests and a committed demo page at `docs/invest/`.

**Tech Stack:** Python 3.12, typer, pydantic config, numpy, pyyaml; vanilla JS on the existing `window.TV` chart kit; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-investing-dashboard-design.md`

## Global Constraints

- Commits: `git -c user.name="Ruhan Sahasi" -c user.email="ruhansahasi@icloud.com" commit`, no attribution trailers, no em-dashes anywhere.
- No network in tests; fixtures only. `tests/fixtures/prices/*.csv` and the committed panels are the only data.
- Frontend rules from `tests/dashboard/test_frontend.py` apply to new assets too: DOM via element constructors only, tokens define both themes, reduced-motion gate, node --check parses, no em-dashes, no dashed strokes.
- Every valuation figure ships beside its baseline and verdict; refusals are named, never blank.
- Private data: nothing under `portfolio/` is ever committed (gitignored already).
- The full suite must pass at every commit.

---

## Wave R: app chrome on the results dashboard

Branch: `dashboard/app-chrome`.

### Task R1: Ink-teal tokens, dark default palette

**Files:**
- Modify: `src/techval/dashboard/assets/tokens.css`
- Test: existing `tests/dashboard/test_frontend.py` token tests must stay green

Replace the palette in place, keeping every token name and the two-dark-block structure. Light theme becomes the secondary theme re-derived from the same hues. New values (dark): `--page:#0A1216`, `--surface:#101B21` (cards now differ from page), `--raised:#16232A`, `--line:#1B2A32`, ink `#E9F1F0`, muted `#87999F`, accent `--focus`/links `#7FE8D2`, `--c-model` teal `#2AA98C` family kept, pos `#5CD6A8`, neg (clay) `#E8927C`. Add `--radius-card:14px`, `--radius-chip:999px`, `--shadow-card`. Because `--surface` no longer equals `--page`, check every kit halo/ring usage against the new ground on screenshots in R7. Run the dataviz validator on the categorical set against both new grounds and fix any failing step. Commit: "Reset the dashboard tokens to the ink-teal app palette".

### Task R2: Cards and ground

**Files:**
- Modify: `src/techval/dashboard/assets/layout.css`

Exhibits (`.tv-figure`, `.tv-tileset`) become cards: `background:var(--surface); border:1px solid var(--line); border-radius:var(--radius-card); padding:var(--space-4)`; the rules-between-exhibits go; section spacing carries the rhythm instead. Page keeps its serif title and standfirst. Grid stays. Commit: "Set every exhibit on a card".

### Task R3: Sticky tab navigation

**Files:**
- Modify: `src/techval/dashboard/assets/app.js` (contents bar), `layout.css`

The existing contents bar becomes `position:sticky; top:0` on a `--page` background with a hairline underneath; the active section gets a 2px `--focus` underline driven by the existing scroll-spy; roman numerals drop, short titles stay. Keyboard reachability unchanged. Commit: "Pin the contents bar as a sticky tab bar".

### Task R4: Verdict tiles under each section title

**Files:**
- Modify: `src/techval/dashboard/assets/app.js`, `layout.css`
- Test: extend `tests/dashboard/test_frontend.py` app-shell test to assert the builder exists

For sections whose snapshot carries a `headline`, render a row of three tiles between the standfirst and the first figure: metric+score, baseline name+score, lift with the existing verdict chip. Values come straight from the already-rendered headline data (no snapshot change); the current prose headline line stays beneath as the sentence. Tiles are cards with `--fs-figure` numbers, tabular numerals. Commit: "Open each section with verdict tiles".

### Task R5: Delta colouring and table treatment

**Files:**
- Modify: `src/techval/dashboard/assets/kit.js` (`format`/table cell path), `layout.css`

In kit table views and `aside` columns, any value rendered through a `signed:*` format gains class `tv-pos` or `tv-neg` by sign; CSS colours them `--c-pos`/`--c-neg`. Zebra hover on table rows, header row in caps micro type. Guard: text still never wears a series colour for identity, only for sign. Add a node-driven test asserting `signedClass(-1)` and `signedClass(1)` outputs. Commit: "Colour signed values by their sign in every table".

### Task R6: Dark default and motion

**Files:**
- Modify: `src/techval/dashboard/assets/app.js` (theme stamp), `layout.css`
- Modify: `tests/dashboard/test_frontend.py` (theme-default test)

`?theme=` still forces; with no query the page now stamps `data-theme="dark"` unless the system asks for light (`prefers-color-scheme: light` keeps light users on light). Motion inside `@media (prefers-reduced-motion: no-preference)`: cards get a 120ms hover lift, tiles fade-slide in once per load, the sticky bar gets a shadow when scrolled. Commit: "Open dark by default and add gated motion".

### Task R7: Recollect, screenshot, and verify

**Files:**
- Modify: `docs/dashboard/index.html` (render only; snapshot numbers unchanged)

`techval dashboard --config docs/dashboard/assumptions.yaml --no-collect` (a pure re-render; the snapshot's figures are untouched by this wave; if any test pins rendered bytes, recollect instead). Screenshot both themes headless, read the images, fix clipping. Run the full suite. Commit: "Render the results page in the app chrome". Merge `dashboard/app-chrome` to main.

---

## Wave I: the investing dashboard

Branch: `invest/portfolio-app`. Package `src/techval/invest/` with `__init__.py` exporting nothing until wired.

### Task I1: Ledger parsing and validation

**Files:**
- Create: `src/techval/invest/__init__.py`, `src/techval/invest/ledger.py`
- Test: `tests/invest/__init__.py`, `tests/invest/test_ledger.py`

**Interfaces (produced):**
```python
TYPES = ("buy", "sell", "deposit", "withdraw", "dividend", "split")
KINDS = ("stock", "etf", "crypto")

@dataclass(frozen=True)
class Transaction:
    date: date; type: str; symbol: str | None = None; shares: float | None = None
    price: float | None = None; amount: float | None = None; ratio: float | None = None
    kind: str | None = None; note: str | None = None

class Ledger:
    name: str; benchmark: str; transactions: list[Transaction]  # sorted by (date, file order)
    targets: dict[str, float]
    @classmethod
    def load(cls, path: Path) -> "Ledger"           # ConfigError on any invalid row, naming it
    def kind(self, symbol: str) -> str
    def symbols(self) -> list[str]                   # non-cash, first-seen order
    @property
    def first_date(self) -> date
```

Validation in `load`: unknown type/kind; buy/sell need symbol+shares>0+price>=0; deposit/withdraw need amount>0; dividend needs symbol+amount>0; split needs symbol+ratio>0; a symbol's `kind` may be stated once (conflicts refuse); dates parse ISO. Tests: a good file loads sorted; each refusal path raises `ConfigError` whose message contains the row number and field. Steps: failing tests, run, implement, run, commit "Read the portfolio ledger and refuse what cannot be true".

### Task I2: Positions by FIFO lot

**Files:**
- Modify: `src/techval/invest/ledger.py`
- Test: `tests/invest/test_positions.py`

**Interfaces (produced):**
```python
@dataclass
class Lot:  symbol: str; opened: date; shares: float; cost_per_share: float

@dataclass
class Position:
    symbol: str; kind: str; shares: float; lots: list[Lot]
    realized: float; dividends: float
    @property
    def cost(self) -> float                          # sum over lots

# on Ledger:
def positions(self, as_of: date | None = None) -> dict[str, Position]
def cash(self, as_of: date | None = None) -> float
def flows(self) -> dict[date, float]                 # deposits + minus withdrawals only
```

Semantics: buys append lots at price; sells consume FIFO, realized += proceeds minus consumed basis; selling more than held raises `ConfigError` naming the date and symbol; splits multiply open lot shares and divide cost_per_share by ratio; dividends accrue to the position and to cash; buys draw cash, sells and deposits feed it; cash below zero after any transaction raises `ConfigError` with the running date. Tests hand-compute: two buys then a partial sell (FIFO basis), a 10:1 split then a sell, dividend flow, the overdraft refusal, the oversell refusal. Commit "Carry positions by FIFO lot with realized gain and cash".

### Task I3: Quotes through the market sources

**Files:**
- Create: `src/techval/invest/quotes.py`
- Test: `tests/invest/test_quotes.py` (CsvSource on `tests/fixtures/prices`)

**Interfaces (produced):**
```python
CRYPTO_STOOQ = {"BTC": "btcusd", "ETH": "ethusd", "SOL": "solusd", "DOGE": "dogeusd"}

class Quotes:
    def __init__(self, source, *, start: date, today: date) -> None
    def series(self, symbol: str, kind: str = "stock") -> PriceSeries   # crypto maps via CRYPTO_STOOQ
    def close_on(self, symbol: str, day: date, kind: str = "stock") -> float  # last close <= day; MissingDataError if none
    def last_two(self, symbol: str, kind: str = "stock") -> tuple[float, float]  # (last, previous)
```

Memoized per symbol; crypto with no mapping raises `MissingDataError` with a hint to add the pair; `close_on` binary-searches dates. Tests on committed CSVs: series loads, forward-fill picks the prior close on a weekend date, unknown crypto refuses. Commit "Quote holdings through the engine's own price sources".

### Task I4: Daily value series and time-weighted return

**Files:**
- Create: `src/techval/invest/performance.py`
- Test: `tests/invest/test_performance.py`

**Interfaces (produced):**
```python
@dataclass
class Series:  dates: list[date]; values: np.ndarray                    # portfolio value, oldest first

def value_series(ledger: Ledger, quotes: Quotes) -> Series               # calendar = benchmark trading days >= first_date
def twr(series: Series, flows: dict[date, float]) -> np.ndarray          # growth of 1, flows stripped
def benchmark_growth(quotes: Quotes, benchmark: str, dates: list[date]) -> np.ndarray
@dataclass
class Contribution: symbol: str; unrealized: float; realized: float; dividends: float; total: float
def contributions(ledger: Ledger, quotes: Quotes) -> list[Contribution]  # sorted by total desc
```

`value_series`: for each benchmark trading day, positions(as_of=day) valued at `close_on`, plus cash(day). `twr`: on a day with external flow F and values V_i, the sub-period return is `V_i / (V_{i-1} + F)`; chain the product. Test the load-bearing case with hand numbers: 100 deposited and invested, value doubles to 200, deposit another 200 (value 400), flat to the end: money doubled but TWR must read +100.0 percent, not +33.3. Also: benchmark growth aligns to the same dates; contributions sum to total gain. Commit "Value the portfolio daily and strip deposits out of the return".

### Task I5: Hygiene: weights, concentration, exposure, drift

**Files:**
- Create: `src/techval/invest/hygiene.py`
- Test: `tests/invest/test_hygiene.py`

**Interfaces (produced):**
```python
@dataclass
class Weight: symbol: str; kind: str; value: float; weight: float
def weights(ledger: Ledger, quotes: Quotes) -> list[Weight]              # includes a "cash" row
def top_share(rows: list[Weight], n: int = 5) -> float
def exposure(rows: list[Weight]) -> dict[str, float]                     # sub-vertical via fade_universe(); "index funds", "crypto", "cash", "outside TMT"
def drift(rows: list[Weight], targets: dict[str, float]) -> list[dict]   # symbol, weight, target, gap; [] without targets
def coverage(rows: list[Weight]) -> dict[str, float]                     # share of value the engine can value
```

`exposure` classifies through `techval.ml.forecast.fade_universe()` (offline). Tests: weights sum to 1, exposure buckets a mixed book, drift math, coverage on a half-covered book. Commit "Measure concentration, exposure and drift".

### Task I6: Engine read: fade path and warranted residual, offline

**Files:**
- Create: `src/techval/invest/engine_read.py`
- Test: `tests/invest/test_engine_read.py`

**Interfaces (produced):**
```python
@dataclass
class EngineRead:
    symbol: str; covered: bool; sub_vertical: str | None
    fade: dict | None        # {"years": [...], "fitted": [...], "assumed": [...], "basis": [...], "trailing": float}
    warranted: dict | None   # {"traded": float, "warranted": float, "residual": float, "as_of": "YYYY-MM-DD"}
    dcf: dict | None         # filled by I7
    refusals: list[dict]     # {"what","why"} per missing pane

def read_holdings(symbols, *, fixtures: Path, assumptions: Assumptions) -> dict[str, EngineRead]
def fade_model(fixtures: Path, assumptions: Assumptions)                 # cached fit on fade_companyfacts.json.gz
def warranted_model(fixtures: Path, assumptions: Assumptions)            # cached fit on warranted/observations.json.gz
```

Fade: build the panel once from the committed blob (the pattern in `tests/ml/test_forecast.py::FadeFixtureClient`), `fit_fade`, `model.path(ticker, projection_years)`; a ticker outside the panel refuses "not in the fade panel". Warranted: `fit_warranted` on the recorded panel, residual for the ticker at `model.latest`, with the panel's recorded-age caveat carried as a note; outside the panel refuses. Both fits are module-level caches so twelve holdings cost one fit each. Every `EngineRead` for a covered name carries the model card verdict sentences. Tests: DDOG gets a fade path and a warranted residual; a non-TMT symbol is uncovered with two named refusals; determinism (two calls equal). Commit "Read the fade and the warranted line for what the portfolio holds".

### Task I7: Engine read: DCF against price

**Files:**
- Modify: `src/techval/invest/engine_read.py`
- Test: `tests/invest/test_engine_dcf.py` (DDOG fixtures: `companyfacts_DDOG.json`, csv prices)

**Interfaces (produced):**
```python
def dcf_read(symbol: str, *, facts: CompanyFacts, market: MarketData, assumptions: Assumptions) -> dict
# {"per_share": float, "price": float, "gap_pct": float, "wacc": float}
def attach_dcf(reads: dict[str, EngineRead], *, client, market, assumptions, live: bool) -> None
```

`attach_dcf` tries `client.company_facts(symbol)`; offline (`live=False`) it only proceeds when the payload is already cached, else appends a refusal "facts not cached; run with the network". The valuation is the existing pipeline: `build_financials`, `build_ev_bridge`, `compute_wacc`, `run_dcf`; every `TechvalError` becomes a named refusal, never a crash. Test: DDOG from fixtures reproduces the pinned 36.65 per share against the fixture close; a symbol with no facts refuses. Commit "Value covered holdings through the engine's own DCF".

### Task I8: Ideas from the warranted screen, verdict attached

**Files:**
- Create: `src/techval/invest/ideas.py`
- Test: `tests/invest/test_ideas.py`

**Interfaces (produced):**
```python
@dataclass
class Idea: ticker: str; traded: float; warranted: float; residual: float
def ideas(model, held: set[str], n: int = 8) -> dict
# {"cheap": [Idea...], "rich": [Idea...], "as_of": iso, "verdict": SIGNAL_VERDICT}
SIGNAL_VERDICT: str  # the recorded finding: mean IC -0.0984, Newey-West t -1.62, NOT significant; use the baseline
```

Held names excluded; ties broken by ticker for determinism. The verdict string names its source (`techval signal` on the recorded EV/Revenue panel) and is asserted verbatim in the test so it cannot drift from honest. Commit "Rank the screen's cheap and rich, with the signal's own failure on top".

### Task I9: Snapshot assembly and validation

**Files:**
- Create: `src/techval/invest/snapshot.py`
- Test: `tests/invest/test_snapshot.py`

**Interfaces (produced):**
```python
def build_snapshot(ledger, quotes, *, fixtures: Path, assumptions, today: date,
                   client=None, market=None, live: bool = False) -> dict
def validate_snapshot(snap: dict) -> None            # ValueError naming the missing/mistyped key
def write_snapshot(snap: dict, path: Path) -> None   # sorted keys, floats rounded to 6, trailing newline
```

Top-level keys: `meta` (name, generated, today, benchmark, price_source), `overview` (value, day_abs, day_pct, twr_pct, benchmark_twr_pct, cash, n_positions, cheap/rich/uncovered counts, movers), `positions` (rows with value/weight/day/unrealized/realized/engine chip), `performance` (dates, growth, benchmark_growth, contributions, lots), `hygiene`, `engine` (per-symbol EngineRead dicts), `ideas`, `activity` (ledger rows). `validate_snapshot` walks a declared schema of required keys and types, the `validate_section` discipline. Tests: fixture portfolio builds a valid snapshot; validator names a deleted key; write is byte-stable across two runs. Commit "Assemble and validate the portfolio snapshot".

### Task I10: Fixture portfolio and the demo snapshot

**Files:**
- Create: `tests/fixtures/invest/portfolio.yaml` (fictional; only tickers with committed closes: DDOG, NET, CRWD, SNOW, MDB, HUBS, ZS, DIS, SPY as the ETF; benchmark SPY; deposits, buys, one sell, one dividend, targets)
- Create: `tests/fixtures/invest/record_demo.py` (rebuilds `docs/invest/snapshot.json` offline from the fixture; header documents that)
- Create: `docs/invest/snapshot.json`
- Test: `tests/invest/test_demo_snapshot.py`

Dates must sit inside the committed CSV ranges; check `head/tail` of each CSV first and choose 2024-01 through the last common date. Test: `build_snapshot` on the fixture with CsvSource equals the committed `docs/invest/snapshot.json` (proving the public demo regenerates from public inputs). Commit "Commit the fictional demo portfolio and its snapshot".

### Task I11: The CLI: techval invest

**Files:**
- Create: `src/techval/commands_invest.py`
- Modify: `src/techval/cli.py` (mount after the dashboard command's panel)
- Test: `tests/test_commands_invest.py`

**Interfaces (produced):** typer app `invest` with `init` subcommand and a default action: `techval invest [--portfolio portfolio/portfolio.yaml] [--out <dir>] [--config ...] [--offline] [--render]`. Default action: load ledger (refuse politely when the file is missing, pointing at `techval invest init`), build quotes from `assumptions.price_source` (csv when `--offline`), build snapshot (live DCF only when not offline), write snapshot and page, print where. `init` writes `portfolio/portfolio.yaml` with commented starter content and refuses to overwrite. Tests through `CliRunner` with the fixture portfolio and csv config: exit 0, files written, `--render` touches no quotes, missing-file refusal text, init scaffolds and refuses twice. Commit "Mount techval invest: init, refresh, render".

### Task I12: App tokens and the parity test

**Files:**
- Create: `src/techval/invest/assets/tokens.css` (dark on bare :root, light under `[data-theme="light"]`; same shared token names as the dashboard)
- Create: `tests/invest/test_assets.py`

Shared-name parity test: parse both tokens files; every token name they share must carry the same value in the corresponding theme blocks; a curated list (`--page`, `--surface`, `--raised`, `--line`, inks, `--c-pos`, `--c-neg`, `--c-model`, radii) must be present in both. Also port the asset rules: no em-dash, no dashed strokes, no markup-from-strings (scans `src/techval/invest/assets`). Commit "One token vocabulary across both dashboards, tested".

### Task I13: Page renderer and shell assets

**Files:**
- Create: `src/techval/invest/render.py`, `src/techval/invest/assets/app.css`, `src/techval/invest/assets/invest.js`
- Test: `tests/invest/test_render.py`

**Interfaces (produced):**
```python
ASSET_NAMES: list[str]      # tokens.css, app.css, kit.js (read from dashboard assets), invest.js, panes/*.js
def render_page(snapshot: dict) -> str
```

`render_page` follows `gallery.render_gallery` byte-discipline: fonts link, inlined styles, inlined scripts in order, `<script type="application/json" id="iv-snapshot">` with the same `_embed_json` escaping (share the helpers by importing from a small new `techval.dashboard.pagemeta` if lifting is cleaner than copying). `invest.js` builds the shell on DOMContentLoaded: rail (seven destinations), sticky header with name + generated stamp + theme toggle, pane host; hash routing (`#holdings`); dark default with `?theme=` override and the light system preference respected; `IV.panes` registry `{id, title, render(body, snap)}`; motion helpers `IV.countUp(el, value, format)` and `IV.reveal(el)` no-ops under reduced motion. Tests: rendered page contains every asset once, snapshot block parses back, node --check on each js, two renders byte-identical. Commit "Render the app shell around an embedded snapshot".

### Task I14: Overview pane

**Files:**
- Create: `src/techval/invest/assets/panes/overview.js`
- Test: extend `tests/invest/test_assets.py` (registry names) and `tests/invest/test_render.py`

Hero value with count-up; day and total-return line with `TV.charts.line` (portfolio growth vs benchmark growth, both from `performance`); tile row: day move, vs benchmark gap, top-five share, engine coverage; cheap/rich/uncovered counts card; top movers list. Everything reads `overview`/`performance`/`hygiene` keys only. Commit "Overview: hero, growth against the benchmark, tiles".

### Task I15: Holdings pane

**Files:**
- Create: `src/techval/invest/assets/panes/holdings.js`

Dense sortable table (click a header to sort; aria-sort maintained): symbol, kind, shares, price, day, value, weight, cost, unrealized abs and pct, realized, engine chip (cheap/rich/uncovered from `engine`). Groups: stocks, ETFs, crypto, cash row. Signed cells wear `tv-pos`/`tv-neg`. Sorting is DOM re-order of built rows, no string HTML. Commit "Holdings: the dense sortable book".

### Task I16: Performance pane

**Files:**
- Create: `src/techval/invest/assets/panes/performance.js`

TWR growth vs benchmark line (shared x from snapshot dates), contribution `TV.charts.hbar` (positive right, negative left, clay), realized + dividends summary tiles, per-lot table behind the chart's Show data discipline. Commit "Performance: growth, contribution, lots".

### Task I17: Engine read pane

**Files:**
- Create: `src/techval/invest/assets/panes/engine.js`

One card per covered holding: DCF per share against price (dot pair with the gap), warranted traded vs warranted with the residual chip, fade path mini-line (fitted vs assumed with basis split), each figure footed by the model card verdict sentence from the snapshot. Uncovered names render their named refusals in the standard note style. Commit "Engine read: what the models say about each holding".

### Task I18: Hygiene, Ideas and Activity panes

**Files:**
- Create: `src/techval/invest/assets/panes/hygiene.js`, `panes/ideas.js`, `panes/activity.js`

Hygiene: concentration hbar, exposure bars by bucket, drift diverging bar when targets exist, coverage note. Ideas: the verdict banner first (verbatim `SIGNAL_VERDICT`), then cheap and rich tables with residuals; held names never appear. Activity: the ledger newest-first with type chips and running cash. Commit "Hygiene, ideas and activity panes".

### Task I19: Motion and polish pass

**Files:**
- Modify: `src/techval/invest/assets/app.css`, `invest.js`

Behind the reduced-motion gate: hero count-up (600ms, once), chart clip-path sweep on first draw, card hover lift, pane crossfade (120ms), slow gradient drift on the featured coverage tile; focus rings on everything interactive; phone width (390px) usable: rail collapses to a top tab strip under 720px. Screenshot both themes at desktop and 390px, read the images, fix overflow. Commit "Motion and small-screen polish".

### Task I20: Demo page, README, and the merge

**Files:**
- Create: `docs/invest/index.html` (rendered from the committed demo snapshot)
- Modify: `README.md` (a "### The investing dashboard" subsection near the results-dashboard section: what it is, `pip install`-free usage, the privacy line, the no-advice line)
- Test: `tests/invest/test_demo_page.py` (render_page on the committed snapshot equals the committed html)

Full suite; screenshots of the demo page both themes; commit "Ship the demo investing dashboard and its README section". Merge `invest/portfolio-app` to main, push, PRs per the house pattern.

## Self-review

Spec coverage: tokens parity (I12), ledger and validation (I1-I2), quotes and crypto (I3), TWR with the deposit trap (I4), hygiene (I5), engine read offline and DCF (I6-I7), ideas with verdict (I8), snapshot+schema (I9), fixture+demo (I10, I20), CLI (I11), shell+panes (I13-I18), motion (I19, R6), restyle (R1-R7). Types checked across tasks: `Ledger.positions -> dict[str, Position]` consumed in I4/I5; `Quotes.close_on` consumed in I4/I5; `EngineRead` consumed in I9/I17. No placeholders: asset tasks name exact selectors, data keys and behaviours; their CSS bodies are the executor's, verified by screenshot steps and the asset tests.
