# The investing dashboard, and the app chrome on the results page

Two deliverables, one visual system. First, the existing results dashboard is
restyled from a research document into app chrome. Second, a new investing
dashboard tracks a private portfolio, runs the engine over what it holds, and
renders one self-contained page in the register of a consumer finance app.

Decisions taken with the owner: full app chrome on the results page; private
holdings in a gitignored file with live prices; a static page rendered from a
snapshot, no server; all four content areas (engine read, hygiene, performance,
ideas); consumer register, dark first; a left-rail workspace layout; the
ink-teal skin that grows the existing techval identity. The owner delegated
every remaining decision and waived further review gates.

## One visual system

The ink-teal skin is a shared token vocabulary: blue-black ground (`#0A1216`
family), hairline-bordered cards one step lighter, teal accent on the active
thing, gains in the model teal-green, losses in warm clay rather than alarm
red, tabular numerals wherever a number sits in a column, radii of 12 to 14px,
and motion only behind `prefers-reduced-motion: no-preference`. The results
dashboard keeps `src/techval/dashboard/assets/tokens.css` and the investing app
gets `src/techval/invest/assets/tokens.css`; both define the same token names,
and a test asserts the shared names carry the same values, the way the two dark
blocks are already tested. One family, no code coupling.

Fonts stay the house set (Public Sans, Newsreader for the voice, IBM Plex Mono
for code), loaded from Google Fonts with system fallbacks, as the dashboard
already does.

## Part one: the results-page restyle

What changes: exhibits move onto cards; the contents bar becomes a sticky tab
bar; each section opens with verdict tiles (headline score, baseline, lift);
tables get delta colouring and verdict chips; dark becomes the default theme
with light kept; motion is added behind the reduced-motion gate. The serif
stays for the page title and standfirsts.

What does not change: every number, takeaway, refusal and source line; the
snapshot schema; the chart kit's marks and its one-tab-stop-per-figure
behaviour; the table view behind every figure.

Roughly seven commits: tokens, card chrome, sticky nav, verdict tiles, table
treatment plus dark default, motion and polish, recollect and screenshot both
themes.

## Part two: the investing dashboard

### Data

A private, gitignored directory `portfolio/` holds `portfolio.yaml`,
`snapshot.json` and `index.html`. Nothing in it is ever committed. The
committed example lives at `tests/fixtures/invest/portfolio.yaml`, a fictional
portfolio drawn only from tickers whose daily closes are already committed
under `tests/fixtures/prices/` (SPY is there for the benchmark), so tests and
the public demo page run offline and deterministically.

`portfolio.yaml` is a transaction ledger, not a positions list, because a
positions list cannot support performance against a benchmark or cost basis by
lot:

```yaml
name: Ruhan's portfolio
benchmark: SPY
transactions:
  - {date: 2024-01-15, type: buy, symbol: MSFT, shares: 10, price: 390.00}
  - {date: 2024-02-01, type: deposit, amount: 5000}
  - {date: 2024-03-10, type: sell, symbol: NVDA, shares: 5, price: 880.00}
  - {date: 2024-06-12, type: dividend, symbol: MSFT, amount: 7.50}
  - {date: 2024-08-05, type: split, symbol: NVDA, ratio: 10}
  - {date: 2025-01-02, type: buy, symbol: BTC, shares: 0.05, price: 42000, kind: crypto}
targets:            # optional, for drift; weights of value
  MSFT: 0.10
  VOO: 0.40
  cash: 0.10
```

Transaction types: `buy`, `sell`, `deposit`, `withdraw`, `dividend`, `split`.
`kind` classifies a symbol on first sight: `stock` (default), `etf`, `crypto`.
Cash is the running balance of deposits, withdrawals, dividends and trade
settlements; a negative balance is a validation error, as is selling shares
never bought. Validation failures name the transaction and refuse the run,
in the engine's usual voice.

### Prices

Quotes come through the existing `techval.market` sources (stooq by default,
nasdaq available, csv for offline runs) via the existing `HttpCache`. Crypto
maps to stooq pairs (`BTC` to `btcusd`, `ETH` to `ethusd`); a symbol with no
source coverage refuses with a hint rather than inventing a price. The
benchmark series comes through the same path.

### What the collector computes

`techval invest` builds a snapshot with these panes:

- **positions**: from the ledger by FIFO lots: shares held, cost basis,
  unrealized and realized gain, dividends received, weight, day move.
- **performance**: a daily portfolio value series from first transaction to
  today; time-weighted return so deposits do not masquerade as skill; the
  benchmark's TWR over the same window; contribution by position; a by-lot
  table.
- **hygiene**: weights and top-five concentration, exposure by sub-vertical
  through `techval.tmt.taxonomy` for covered names (ETFs and crypto are their
  own buckets), drift against `targets` when present, and how much of the
  portfolio the engine can value at all.
- **engine read**: for holdings inside the TMT universe: the fitted fade path
  against the typed one (offline, from the committed fade panel), the
  warranted-multiple residual at the panel's latest date (offline, from the
  recorded warranted panel), and a DCF per share against the market price when
  the filer's facts are cached or the network is allowed. Every figure carries
  its model card's verdict; names outside coverage get a refusal by name, not
  a blank.
- **ideas**: the recorded warranted panel's residuals across the universe at
  its latest date, cheapest and richest, excluding held names. The pane leads
  with the signal harness's own finding, that this score lost to a random one
  on forward returns, so it reads as the engine's opinion with its track
  record attached, never as advice.
- **activity**: the ledger itself, newest first.

The snapshot is validated against a schema the same way `validate_section`
guards the dashboard, and the collector is a pure function of the ledger, the
price series and the committed panels, so two runs on the same inputs agree.

### The command

`techval invest` mirrors `techval dashboard`:

- `techval invest init` scaffolds `portfolio/` with a commented starter file.
- `techval invest` refreshes quotes (unless `--offline`), rebuilds
  `portfolio/snapshot.json`, and renders `portfolio/index.html`.
- `--render` re-renders from the existing snapshot without touching the
  network. `--portfolio`, `--out`, `--config` override paths.

The demo page is rendered from the fixture portfolio with csv prices into
`docs/invest/index.html` and committed, so the public repo shows the app with
fictional holdings.

### The page

One self-contained HTML file, dark first with light kept, in the B layout: a
left rail with Overview, Holdings, Performance, Engine read, Hygiene, Ideas
and Activity; a workspace that swaps panes without a reload. The snapshot is
embedded as a JSON block; scripts build DOM through element constructors, no
markup from strings, matching the dashboard's frontend rules.

`kit.js` is reused as-is for charts (line, hbar, dot, heat all draw through
tokens, so the app skin restyles them for free), which keeps tooltips, table
views and the one-tab-stop rule. The app's own assets are `tokens.css`,
`app.css` and `invest.js` plus one small script per pane.

Aesthetics: hero number with a count-up on load, charts revealed with a
clip-path sweep, cards that lift on hover, a slow animated gradient on one
featured tile, crossfade on pane switches; all of it inside the
reduced-motion gate, and none of it on a second render of the same pane.
Numbers over sixty seconds old say when they were fetched.

### Testing

- Ledger, lots, performance, hygiene, ideas and snapshot: pytest against the
  fixture portfolio and committed closes; every computation deterministic.
- TWR checked against hand-computed values on a tiny ledger, including a
  deposit mid-window that must not inflate the return.
- Frontend: the dashboard's asset rules applied to the new assets (tokens
  parity test, no markup from strings, node --check, both themes defined),
  plus a rendered-demo test that the committed page rebuilds byte-identical.
- CLI: refusal paths (missing file, invalid ledger, unknown symbol) through
  the Typer runner.

### What this deliberately does not do

No server, no broker imports, no order placement, no alerts, no options, no
ETF look-through in v1, and no advice: every valuation figure ships beside its
baseline and its measured verdict, and the ideas pane leads with the signal's
failure. The engine values US TMT filers; everything else is priced, weighed
and reported, never fake-valued.
