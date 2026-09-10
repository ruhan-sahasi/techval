# Methodology

Every formula the engine uses, and every judgment call behind it. Written for a
finance reader: the arithmetic is stated, but the space given to each section is
proportional to how much the choice moves the answer, not to how hard it was to
code.

Where a decision is contested in practice, both camps are stated, the default is
named, and the switch that changes it is given.

---

## 1. Scope and units

US-listed filers reporting under US GAAP to the SEC in XBRL. Money is carried in
USD millions, share counts in millions of shares, per-share figures in dollars.
`Decimal` appears only at the reporting boundary; everything internal is float64,
which is exact enough for a valuation whose largest input is a judgment about the
equity risk premium.

The engine is deterministic. The same ticker, the same assumptions file and the
same cached filings produce the same numbers, because every HTTP response is
cached by request URL under `~/.techval/cache`. A number in a memo can be
reproduced a year later from the cache alone.

Nothing is interpolated. When a required figure cannot be sourced, the engine
raises `MissingDataError` naming the concept, the us-gaap tags it tried in order,
and the period it searched. A valuation built on a silently invented number is
worse than no valuation, because it looks the same.

---

## 2. Sourcing fundamentals from XBRL

The SEC publishes every filer's tagged facts at
`data.sec.gov/api/xbrl/companyfacts/CIK##########.json`. Turning that into an
income statement is most of the work in this package, and it is where naive
implementations go quietly wrong. Four failure modes, each observed in the
filings committed as fixtures.

### 2.1 Retired tags

A tag that ever appeared in a filer's history appears in `companyfacts` forever,
carrying whatever value it last had. CrowdStrike's `ShortTermInvestments` last
held a value in early 2025. HubSpot's `ConvertibleLongTermNotesPayable` stops in
2021. Resolving a balance-sheet concept by "first tag in the fallback list that
exists" returns a number that is years stale, with nothing on screen to say so.

**Treatment.** Balance-sheet concepts resolve *as of a date*. A tag wins only if
it carries a fact within 20 days of the balance-sheet date being asked about. A
ladder entry whose newest fact is older than that is skipped, and if every entry
is stale the engine raises `StaleDataError` listing what it found and when.

This is not hypothetical. Zscaler reports its marketable securities under
`DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent`, a tag outside
the obvious list. Before that entry was added, the engine returned zero
short-term investments and overstated Zscaler's enterprise value by $2.5bn.

### 2.2 Absence means two different things

A concept can be missing because the filer has none of it, or because the engine
looked in the wrong place. Non-controlling interest and preferred stock are
absent from most software filers' fact sets because those companies have neither,
and zero is the correct reading. Revenue is never legitimately absent.

**Treatment.** Only concepts of the first kind carry a default, and when a
default is used the provenance record says `absent, defaulted` rather than
showing a sourced figure. Everything else raises.

As a backstop against a wrong zero, current assets are tied back to cash,
short-term investments and receivables. An unexplained residual above 25% of
current assets is reported as a warning naming the amount, because a
two-billion-dollar hole in an enterprise value bridge should never be invisible.

### 2.3 Trailing twelve months when quarters are not reported

Q4 is never a filed period: it is the annual figure less the first nine months.
Beyond that, filers differ enormously in what they tag. Snowflake reports
depreciation only cumulatively from the fiscal year start, so no discrete quarter
exists at all. CrowdStrike files a year-to-date half *and* a discrete Q2, leaving
Q1 recoverable only by subtraction. "Sum the last four quarterly values" finds
nothing for either company.

**Treatment.** Period algebra rather than quarter summing.

1. Deduplicate facts by period. A period is reported many times: when filed,
   again as a comparative in later filings, again in an amendment, and sometimes
   again in a proxy statement. Two rules pick between them.

   **Form class first.** An audited periodic report (10-K, 10-Q, 20-F and their
   amendments) beats anything else no matter when it was filed. A proxy
   summarises the financial statements, it does not restate them, and letting a
   DEF 14A override a 10-K because it happened to be filed six weeks later swaps
   an audited figure for a summary of one. CrowdStrike's fiscal 2024 net income
   is 72.2mm in the 10-K and 73.4mm in the proxy.

   **Filing date second.** Within the same class the most recently filed value
   wins, because that is the genuine restatement and the number the company now
   stands behind. The same CrowdStrike year read 89.3mm in the 10-K filed in 2024
   and 72.2mm in the one filed in 2026, and the later figure is the right one.
2. Derive every period recoverable by subtraction. Where two reported windows
   share a **start** date, their difference is the tail period. Where they share
   an **end** date, their difference is the head period. Applied repeatedly, this
   recovers discrete quarters that were never filed, including the fourth.
3. Tile the twelve months ending on the balance-sheet date exactly, using
   non-overlapping contiguous periods, preferring longer pieces. A filed annual
   period ending on that date is used directly when one exists.

A few days of slack are allowed at the window start, because a 13-week fiscal
quarter does not land on the calendar anniversary.

When a concept still cannot be tiled, the engine either raises, or for
non-critical items falls back to the most recent full fiscal year and says so in
both the provenance record and a warning. Operating lease cost is the usual case:
several filers disclose it only annually.

Some concepts are simply not there. Where a filer publishes no combined
depreciation and amortisation line, the ladder tries summing the components, but
only over periods where every component is tagged, so a partial sum is never
passed off as a total. CrowdStrike tags amortisation of intangibles and no
depreciation line at all, so its EBITDA cannot be built from company facts.
The engine reports that rather than publishing an EBITDA missing its
depreciation, and the comp table keeps the company with its revenue multiples
intact and a flag on the one it could not form.

### 2.4 Stock splits

This is the subtlest of the four and the one most likely to produce a confidently
wrong answer.

A split makes every historical share count and per-share figure inconsistent with
every later one. Filings after the split restate the comparative periods they
happen to display, but earlier periods that no later filing repeats keep their
pre-split values in `companyfacts` permanently. A trailing twelve months built
across that boundary silently mixes units.

CrowdStrike split four-for-one in mid-2026. Its fact set shows the quarter ended
31 July 2025 as 249.9mm shares as originally filed and 999.6mm as restated in the
August 2026 10-Q. Mixing restated and unrestated quarters produced a TTM diluted
share count of 271mm against a true post-split figure of 1,023mm, an error of
roughly 4x flowing straight into equity value.

**Treatment.** Splits are detected from the filings themselves, with no external
data. Where one period carries two filed values whose ratio is within 0.5% of a
ratio a board would actually declare (2, 3, 4, 5, 10, 20 and the common reverse
ratios), that is a split, and the filing that introduced the new value dates it.
Requiring at least two periods to move by the identical ratio in the identical
filing separates a split from an ordinary restatement, which moves numbers by a
few percent rather than by exactly four.

Facts filed before that date are then restated into current units: share counts
multiplied by the factor, per-share figures divided by it.

**Control.** Net income divided by TTM diluted shares must reproduce diluted EPS
summed from the filings. A gap above 10% is reported, because that is precisely
what mixing units on either side of a corporate action looks like.

### 2.5 Averages are not sums

Weighted-average shares outstanding is a duration fact, but it is an average over
its window, not an accumulation. Subtracting a first-quarter average from a
half-year average is meaningless arithmetic, and summing four quarterly averages
gives four times the share count.

**Treatment.** Average-type concepts are converted to *integrals* (average times
days, the sum of the daily share count over the window) before any period algebra
runs, then divided by total days at the end. The integral is additive, so every
subtraction and tiling rule applies unchanged.

### 2.6 What `companyfacts` does not contain

Only undimensioned facts are published. A dual-class issuer tags shares
outstanding by class, so `dei:EntityCommonStockSharesOutstanding` does not exist
for it at all. Datadog is one such filer.

Period-end share counts are therefore not reliably available from this endpoint,
and the engine uses TTM weighted-average diluted shares throughout. See §4.3 for
what that approximation costs.

### 2.7 Provenance

Every line item carries a record of the tag that won, the periods used, the forms
they came from, the filing date, and the method (tiled, derived by subtraction,
defaulted, annual fallback). `techval fetch <TICKER>` prints the whole table.
This is the audit trail: any number in the output can be walked back to a filing.

---

## 3. Market data

**Prices.** Nasdaq's public quote API, which needs no key and no account and
carries roughly three years of daily closes for equities and ETFs. Sources sit
behind one interface, and a local CSV source exists for offline and fully
reproducible runs.

This package originally targeted Stooq. Stooq now answers plain HTTP clients with
a JavaScript proof-of-work challenge rather than CSV. Defeating a bot check the
operator deliberately erected is not something this package will do, so the Stooq
source raises an explanatory error pointing at the alternatives.

**Risk-free rate.** The latest 10-year constant-maturity Treasury yield from the
Treasury's daily yield curve feed. The ten-year point is the conventional anchor
for a US equity discount rate: long enough to match the duration of the cash
flows being discounted, liquid enough that the quote means something. Overridable
in the assumptions file to pin a run to a fixed date.

**Market proxy.** SPY rather than the S&P 500 index itself, because the index is
not available from a keyless source. The substitution is close to free for beta
estimation: SPY's return differs from the index return by its dividend
distributions, and subtracting a roughly constant yield from the market series
shifts the regression **intercept**, not its slope. It moves alpha, not beta.

---

## 4. Enterprise value bridge

```
EV = equity value
   + straight debt + convertibles (see 4.2) + finance leases
   + operating leases (see 4.1)
   + preferred stock + non-controlling interest
   - cash and equivalents - short-term investments
```

Simple to write down. The judgment is entirely in what counts as debt, and two
items decide it for a technology company.

### 4.1 Operating leases and ASC 842

ASC 842 put the lease liability on the balance sheet but left the expense inside
operating income, as a single straight-line rent charge. It is **not** split into
depreciation and interest the way IFRS 16 requires, so it is not part of the D&A
that gets added back.

The consequence: **a US filer's EBITDA is already after rent.** Adding the lease
liability to debt while pairing it with that EBITDA counts the same obligation
twice, once in the numerator as a claim on the enterprise and again in the
denominator as a charge that suppressed earnings. The multiple comes out too
high, and the error scales with how lease-heavy the company is.

Either convention is defensible so long as it is applied consistently:

| Enterprise value | pairs with | Earnings metric |
|---|---|---|
| Excluding lease liabilities | → | EBITDA (after rent) |
| Including lease liabilities | → | EBITDAR (before rent) |

**Default: operating leases excluded, EBITDA basis.** That is how a US software
comp set is normally quoted, and it avoids inventing an interest rate to unwind
the rent charge.

The engine reports both enterprise values regardless of the setting, and the
earnings figure that pairs with the chosen one is only reachable through
`EVBridge.multiple_denominator()`. A mismatch cannot happen by accident. If
leases are capitalised and the filer does not disclose operating lease cost,
EBITDAR cannot be formed and the affected multiples are suppressed rather than
shown on an inconsistent basis.

Finance leases are debt under either convention: their interest and amortisation
already sit outside EBITDA.

### 4.2 Convertible notes

ASU 2020-06 removed the treasury-stock option and made the if-converted method
mandatory for convertible instruments. Whenever a convertible is dilutive, its
**full conversion shares are already inside diluted weighted-average shares
outstanding.**

Equity value computed on that share count therefore already contains the
converted instrument. Adding the principal to debt as well counts it twice.

| Convertible | Shares in diluted WASO? | Treatment |
|---|---|---|
| In the money | Yes, if-converted | Equity, **not** debt |
| Out of the money | No, antidilutive | Debt |

This is worth real money. Datadog carries roughly $986mm of convertible notes
against a conversion price of about $148 with the stock above $200. Counting them
as debt on top of a diluted share count that already reflects conversion
overstates enterprise value by close to a billion dollars.

**Default: `auto`.** With a conversion price supplied in the assumptions file the
engine compares it to the current price and picks the matching treatment,
recording which and why. Without one it falls back to treating them as debt and
prints a warning stating exactly how much enterprise value may be overstated. The
warning is deliberately loud, because the silent version of this error is
invisible in the output.

### 4.3 Share count

Equity value uses **TTM weighted-average diluted shares**, consistent with the
earnings figures in every multiple built on it.

The alternative is a full treasury-stock-method count on currently outstanding
in-the-money options and RSUs at the current price. That is more precise for a
point-in-time equity value, and it is what a live model would use. It is not
implemented here, for two reasons: the strike-price detail sits in dimensioned
XBRL facts that `companyfacts` does not publish, and a partial implementation
that captured some awards and not others would be worse than a clearly labelled
approximation.

What the approximation costs: diluted WASO is a backward-looking average, so for
a company issuing shares steadily it understates the current count, and it
understates equity value in a rising market where more awards are in the money.
For a mid-cap software company with heavy stock compensation the gap is typically
low single-digit percent. It is listed in the README limitations.

---

## 5. Cost of capital

```
WACC = We * Ke + Wd * Kd * (1 - t)
Ke   = rf + beta_levered * ERP + size premium
```

### 5.1 Beta

Ordinary least squares of weekly stock returns on weekly market returns over two
years, roughly 104 observations. Weekly is the usual compromise: daily returns of
a mid-cap carry enough non-synchronous trading noise to bias the slope downward,
and monthly returns leave too few points for a two-year window to say anything.

Two details that matter more than the choice of frequency.

**Pairing is by date, never by position.** A peer that listed eighteen months ago
has fewer weeks than the index. Zipping the two return vectors together pairs its
first week against the index's first week two years earlier, and the slope that
comes back is arithmetic performed on unrelated numbers. It looks entirely
plausible on screen. The regression here joins on the dates the two series
actually share, and refuses below 30 common observations.

**R-squared and the standard error are reported, not hidden.** A beta of 1.42 on
an R-squared of 0.11 is not a number the market told you with any confidence.
Datadog's own two-year regression against SPY produces exactly that. The analyst
needs to see it before deciding whether to lean on the peer beta instead.

**Blume adjustment** (`beta_adjustment: blume`) applies `0.67 * raw + 0.33 * 1.0`,
the Bloomberg convention. The justification is that betas mean-revert toward one
as companies mature and diversify. Off by default, because the shrinkage is a
prior rather than a measurement and the raw estimate is the thing actually
observed. Applied after the OLS and before unlevering, so the leverage adjustment
operates on the beta the cost of equity will actually use.

### 5.2 Unlevering and relevering

```
beta_unlevered = beta_levered / (1 + (1 - t) * D/E)
```

The Hamada relation. The tax term is there because the debt tax shield is itself
a claim whose risk tracks the debt, so leverage raises equity risk by less than
gross D/E would suggest.

The peer approach, which is the standard sell-side one: unlever each peer at
**its own** D/E and tax rate, take the **median** of the resulting asset betas,
then relever at the target's capital structure. The point of the exercise is to
strip out capital-structure differences before averaging, so that what is being
pooled is business risk rather than financing policy. The median rather than the
mean, because one peer with a broken regression should not move the answer.

D/E uses market equity against book debt, matching the WACC weights.

### 5.3 Cost of debt

Three methods, and the default is not the obvious one.

**`filings`** divides interest expense by total debt at the latest balance
sheet date. Averaging opening and closing debt would be better, and is what a
credit analyst would do, but only one balance-sheet instant is normalized here,
so the period-end balance is what there is. For an issuer that borrowed heavily
during the year this understates the divisor and overstates the implied rate.
Either way it is a backward-looking *accounting* yield, not a market one. For a company whose
borrowings are zero-coupon convertible notes it returns something near 0.5%,
which is not a rate anyone would lend at. The engine therefore rejects any book
yield that lands below the risk-free rate and falls through to the synthetic
method, saying so in the output.

**`synthetic`** (default) forms interest coverage as EBIT over interest expense,
maps it to a rating on a Damodaran-style large-cap table, and adds that spread to
the risk-free rate. Where coverage cannot be formed at all, because EBIT is
negative or there is debt but no reported interest, the rating is pinned at the
lowest investment grade rather than the bottom of the table. Entering at the
bottom would price a cash-rich, unprofitable software company as a defaulted
credit at an eighteen-percent spread, which is not what its bonds would trade at.

**A limitation worth stating plainly.** Interest coverage is the wrong risk
metric for a net-cash issuer. Datadog holds about $5bn of cash and securities
against $986mm of zero-coupon converts, and its GAAP EBIT is thin enough that
coverage reads 1.4x, which maps to a speculative rating. Its actual default risk
is negligible. The saving grace is that the debt weight is near zero, so the
number barely reaches the WACC, and the output says as much. For a company with
meaningful debt *and* net cash, override the rate.

### 5.4 Weights

Market value of equity against **gross** book debt, not net. The tax shield
attaches to debt outstanding, not to a net position, so netting cash against debt
before weighting would understate the shield. Where a company is net cash the
output says so and notes that its WACC is its cost of equity to within a few
basis points.

Book value is used as a proxy for the market value of debt. For investment-grade
paper near par this is close. For distressed debt it is badly wrong, and that is
listed in the README limitations.

---

## 6. Discounted cash flow

### 6.1 Free cash flow to the firm

```
FCFF = EBIT * (1 - t) + D&A - capex - change in NWC
```

**Stock-based compensation is the largest judgment in a software DCF, and the
build specification for this package got it wrong.** The specification asks for
`EBIT * (1-t) + D&A - capex - dNWC - SBC`. GAAP EBIT is *already net of* stock
compensation, because it is an operating expense on the income statement.
Subtracting it again charges the same compensation twice. That formula is not
implemented.

What is implemented is a switch:

- **`expense` (default).** Leave SBC inside EBIT. Free cash flow is struck after
  the full cost of compensating employees. The case for this is simply that
  stock compensation is compensation: the company would have to pay cash to
  replace it, and calling it non-cash confuses the *form* of the payment with
  whether it is a cost.
- **`addback`.** Restore SBC as a non-cash charge. The case for this is that no
  cash leaves the business, so a cash-flow statement should not show it going
  out. The case *against* is that adding it back without separately modelling
  the shares it creates counts the benefit and ignores the cost. Anyone using
  this setting should also be growing the share count.

For Datadog the switch is worth roughly $823mm a year, about 21% of revenue, and
it moves the valuation by more than any other assumption in the file. That is why
it is a switch with both arguments written down rather than a default with none.

**Working capital.** NWC is taken as a percentage of revenue, and the year-one
change is measured against that same percentage applied to TTM revenue, rather
than against the reported balance. Anchoring on the reported balance would
introduce a one-time step in year one that was never an operating event. For
subscription software the percentage is negative, because customers are billed
ahead of delivery and deferred revenue grows with the business, so growth
releases cash rather than consuming it.

**Taxes** are `rate * max(EBIT, 0)`. A loss year creates a carryforward, not a
refund. The model does not track the NOL balance, so no cash benefit is booked in
the loss year and none is taken later. For a company with large accumulated
losses this understates value, and it is in the limitations.

### 6.2 Discounting

Mid-year convention by default: explicit-period flows are discounted at
`(1 + w)^-(t - 0.5)`, on the reasoning that revenue arrives through the year
rather than in a lump on 31 December.

### 6.3 Terminal value, and the timing distinction

Both methods are computed and both are shown, because agreement between them is
information and disagreement is more.

**Gordon growth.**

```
TV_N = FCFF_N * (1 + g) / (w - g)
```

This capitalises a perpetuity of flows starting in year N+1, valued as of the end
of year N. Under the mid-year convention those perpetuity flows also arrive
mid-year, so the terminal value carries a half-year uplift and is discounted at
`(1 + w)^-(N - 0.5)`.

**Exit multiple.**

```
TV_N = terminal EBITDA * multiple
```

This is an **observed market price struck at a date**, the end of year N. It is
not a stream received through the year. It is therefore discounted over whole
periods, at `(1 + w)^-N`, with **no** mid-year uplift.

Applying the mid-year factor to both is a common error, and it overstates the
exit-multiple valuation by about half a year of WACC, roughly 5% at a 12%
discount rate. The two methods run on separate code paths here for that reason,
and a test asserts the two discount ratios differ by exactly that half year.

Terminal EBITDA is terminal EBIT plus terminal D&A, on the same post-rent
definition the comps are quoted on, so the exit multiple and the peer median
describe the same quantity.

### 6.4 Cross-checks

These are where a DCF is either coherent or is not, and the engine prints them
rather than leaving them to be noticed.

1. **Implied exit multiple from the Gordon terminal value.** If Gordon implies
   7.3x and the peer set trades at 37.3x, the growth and margin assumptions do
   not support what the market is paying, and one of the two is wrong.
2. **Implied perpetuity growth from the exit multiple.** Solving the Gordon
   formula backwards. An exit multiple that implies 9.8% growth in perpetuity
   against an 11.9% WACC is a multiple no long-run assumption supports.
3. **Terminal value as a share of enterprise value**, flagged above 75%. Above
   that line the valuation is a bet on the terminal assumption rather than on
   the forecast, and the honest response is to extend the explicit period until
   the business is mature.
4. **Reinvestment consistency**, the sharpest of the four. In perpetuity,

   ```
   g = ROIC * reinvestment rate
   reinvestment rate = (capex + change in NWC - D&A) / NOPAT
   ```

   so the terminal year implies a return on capital. The engine backs it out and
   flags an implied ROIC that is negative, above 60%, or below the WACC. That
   last case is the interesting one: it means the model is capitalising growth
   that destroys value. A terminal assumption that survives this check is
   internally consistent; one that does not is arithmetic with a story attached.

---

## 7. Trading comparables

Multiples computed: EV/Revenue, EV/Gross Profit, EV/EBITDA, EV/EBIT and P/E, plus
revenue growth, EBITDA margin and Rule of 40.

**EV/Gross Profit** is included because for software it is the more honest scale
metric. Gross margin varies enormously across infrastructure and application
names, and EV/Revenue silently rewards a company for reselling low-margin
compute at scale. Gross profit is closer to the economics being bought.

**Rule of 40** is TTM revenue growth plus TTM EBITDA margin, in points. The free
cash flow margin variant is more common in growth equity; EBITDA margin is used
here because it is comparable across the set from filings alone.

**Not meaningful.** A multiple is suppressed and flagged, never quietly printed,
when its denominator is negative or when it exceeds a configured cut-off: 100x
for EV/EBITDA and EV/EBIT, 75x for P/E, which is lower because net income sits
below interest, tax and every non-operating item and so reaches zero sooner.
EV/Revenue and EV/Gross Profit carry no ceiling, because their denominators do
not approach zero the way an earnings line does; a high revenue multiple is a
statement about the company rather than an artefact of arithmetic. Past
those lines the ratio is measuring how close the denominator is to zero, not what
the market pays for the business, and letting it into a percentile drags the
whole distribution with it.

This matters more than it sounds for a GAAP software comp set. Of the six peers
in the example file, **not one** produces a meaningful GAAP EV/EBITDA, because
GAAP EBITDA margins run between minus twenty and plus two percent. That is a real
finding about the cohort, and the correct output is to say so and lean on
EV/Revenue and EV/Gross Profit, not to print a 3,000x median.

**Peers are never dropped silently.** A peer that fails to build appears in an
exclusions list with the reason. A peer whose EBITDA cannot be formed keeps its
revenue multiples: a company unprofitable this year is still evidence of what the
market pays for its revenue.

**The `n` row** in the statistics table is as important as the median. A column
where two of six peers contributed is a quotation from two companies, and the
table says so. Percentiles use numpy's linear interpolation between order
statistics; over a small set they are interpolated values, not observed trades.

**Implied range** applies the peer 25th and 75th percentiles to the target's own
metric, then walks back through the same EV bridge to a price per share, so a
comps value and a DCF value are reconcilable rather than merely similar. Rows
where the target's own metric is not positive are suppressed, because there is no
sense in applying an EV/EBITDA to a negative EBITDA.

---

## 8. Merger analysis

Year one, both sides on TTM figures, at the acquirer's marginal tax rate. The
acquirer's rate is the right one: the incremental dollar of synergy and the
incremental dollar of interest are earned and deducted inside the acquirer's tax
posture.

```
pro forma NI = acquirer NI
             + target NI
             + after-tax synergies
             - after-tax interest on new debt
             - after-tax interest foregone on balance sheet cash used
             - after-tax financing fee amortisation
             - after-tax intangible amortisation   (off by default)

pro forma shares = acquirer diluted shares + shares issued as consideration
```

**One-time deal fees are funded in sources and uses but kept out of the earnings
walk.** A fee paid once at close is not part of the run-rate earnings the market
capitalises, and including it would make every large deal look dilutive in year
one for a reason unrelated to whether the businesses fit. Financing fees go the
other way: they are capitalised and amortised over the life of the facility, so
their annual charge does belong in the walk.

**Percentage accretion is withheld when the base will not support it.** The
convention is to quote accretion as a percentage of standalone EPS, and that
percentage fails in two directions. A negative base inverts the sign, so a deal
that improves a loss prints as dilution. A base near zero explodes the ratio:
CrowdStrike's TTM diluted EPS of about four cents turns four cents of dilution
into minus eight hundred percent, which describes the denominator rather than the
transaction. Below a ten-cent floor the engine reports cents per share only and
says why.

**Breakeven synergies are solved, not searched.** With `S` the run-rate pre-tax
synergy, `phi` the phase-in, `t` the tax rate, `N_pf` pro forma shares and `D` the
sum of after-tax funding charges:

```
EPS_standalone * N_pf = NI_acq + NI_tgt + S * phi * (1 - t) - D
S = [EPS_standalone * N_pf - (NI_acq + NI_tgt - D)] / (1 - t) / phi
```

`N_pf` does not depend on `S`, so this is exact and needs no iteration.

**Contribution analysis** compares each side's share of combined revenue, EBITDA,
EBIT and net income against pro forma ownership, to show whether the target's
owners receive more of the combined company than the earnings they bring.
Percentages are withheld where either side is negative: dividing by a combined
figure that a loss has pulled toward zero produces shares above 100% on one side
and below zero on the other, and ranks the larger loss as the larger contributor.

### 8.1 The rule of thumb

A deal is accretive when what you buy yields more than what you pay with. What
you buy is always the target's earnings yield at the offer price. What you pay
with depends on the currency:

- **Stock.** The currency costs the acquirer's own earnings yield, `1 / P/E`,
  because every share printed hands a claim on those earnings to someone new. So
  the deal is accretive when the acquirer's P/E exceeds the P/E it is paying.
  Note the direction: a high-multiple acquirer has a *low* earnings yield, which
  is exactly why its paper is cheap currency. Reading the comparison backwards
  inverts the answer on every deal, and it is the commonest error in this
  calculation.
- **Cash.** The currency costs the after-tax coupon on new debt, or the after-tax
  yield surrendered on balance-sheet cash. Accretive when the target's yield at
  the offer exceeds it.

Both are computed, blended at the actual mix, and compared against the full
calculation. Where they disagree the difference is synergies and financing
charges, which is precisely what the rule leaves out. The rule is undefined when
either side loses money, and the engine says so rather than printing a negative
earnings yield.

### 8.2 What is not modelled

No opening balance sheet, no goodwill, no deferred tax liability on the step-up,
no deferred revenue haircut, no debt paydown or buyback over time. Purchase
accounting intangible amortisation is off by default, on the reasoning that it is
a non-cash artefact of the transaction and the street quotes cash EPS; a config
hook turns it on.

Year-one EPS built on two sets of trailing twelve month figures is a screening
tool. It tells you the shape of a deal and roughly what it costs. It is not a
merger model, and no one should take a bid to a board on it.
