# Market-implied expectations: the DCF run backwards

The engine's DCF answers "what is this worth on these assumptions". It never
answers the question a reader asks next when the answer is far from the
price: what does the price assume? Datadog's base case is 36.65 a share
against a 225.27 close, and nothing in the engine says what the market must
believe to pay the difference.

A reverse DCF answers it by solving the engine's own DCF for the price, one
lever at a time, and then asks how often the implied growth has actually
happened, using the 2,798 point-in-time company-years of the fade panel. That
second step is the part a spreadsheet reverse DCF cannot do and the reason it
belongs in this engine: an implied growth rate means little until it is set
against a base rate.

## Levers

Each solve changes one assumption, holds every other at its configured value,
and finds where `run_dcf(...).per_share` equals the market price.

- **Implied discount rate.** `dcf.wacc_override`, bracketed from just above
  terminal growth to 40%. Value falls monotonically as the rate rises and
  explodes as it nears terminal growth, so a root always exists for a positive
  price: the return a buyer at today's price earns if the base case is right.
- **Implied first-year growth.** `dcf.revenue_growth_start`, the engine's
  straight fade to terminal unchanged, bracketed from -50% to +300%.
- **Implied terminal margin.** `dcf.ebit_margin_terminal`, bracketed from -20%
  to 95%.
- **Implied duration.** The company's trailing growth held flat for k of 15
  projection years, then faded linearly to terminal: the shortest whole k
  whose value reaches the price.

A lever that cannot reach the price inside its bracket is reported as such,
with the value at the bracket's edge, never extrapolated: "no terminal margin
below 95% gets there; at 95% the DCF reaches 134" is the answer, not a
failure.

## Base rates

For an implied growth path, the implied h-year revenue CAGR (h = 5, or the
path's length if shorter) is compared with the fade panel: the share of
labelled company-years whose realised forward CAGR over h years, from the
growth each later 10-K printed, met or beat it. Reported twice: across the
whole panel, and among company-years whose trailing growth was within ten
points of the company's own. Each share carries its count, so "0 of 412" reads
as the evidence it is. The panel's labels are winsorized within each filing
year and the panel keeps delisted filers, which this states rather than
hides.

## Frontier

A table of the first-year growth implied at each terminal margin from 10% to
60%, so the trade-off between the two levers reads as a curve rather than two
unrelated numbers.

## Surfaces

- `techval.reverse_dcf`: the solver, the levers, the base rates, the frontier
  and a `MarketExpectations` result with plain sentences.
- `techval expectations TICKER`: the CLI, offline with `--facts` and a csv
  config like `techval fade`.
- `techval invest`: the implied return and implied growth with its base rate
  on each covered holding that has a DCF.
- The results dashboard's valuation engine section: Datadog's expectations as
  a figure.

## Not in scope

Joint solves over several levers beyond the frontier table, scenario weights,
and any statement that the price is wrong. The output is what the price
assumes and how rare that has been; whether to pay it is the reader's call.
