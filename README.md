# techval

![Datadog football field](out/DDOG_football.png)

An open-source valuation engine that produces analyst-grade DCF, trading comps
and merger analysis for US-listed technology companies, using only free data:
SEC EDGAR for fundamentals, Nasdaq's public quote API for prices, and the US
Treasury daily curve for the risk-free rate. It builds normalized trailing
statements out of raw XBRL, an enterprise value bridge, a CAPM cost of capital
off peer beta regressions, an unlevered DCF carrying three terminal values side
by side, a comp set with a warranted multiple regressed on fundamentals, and an
accretion/dilution screen that will build the full opening balance sheet when
asked. Around that sit a point-in-time share count read from the filing instance
document, net operating loss carryforwards, a Monte Carlo over correlated
assumptions, an adjusted present value cross-check, and a backtest harness that
re-runs the whole model on past dates behind a knowledge cutoff so nothing filed
later can reach it. Every number it prints traces back to a filing, a quote, or
an assumption you wrote down. When a figure cannot be sourced it raises an error
naming the XBRL concept and the tags it tried, rather than interpolating
something plausible.

---

## Sample output

Struck on 10 September 2026 against filings through the June 2026 quarter. Every
figure quoted anywhere in this README was produced by running the code on that
date, so the market-dependent ones move with the price.

```
$ techval value DDOG --config assumptions.yaml

────────── Datadog, Inc. (DDOG)  normalized TTM, USD millions ──────────
CIK 1561550   twelve months ended 2026-06-30

Revenue                   3,966.7
Gross profit              3,154.0  79.5% of revenue
EBIT                         16.3   0.4% of revenue
D&A                          62.0
EBITDA                       78.3   2.0% of revenue
Stock-based compensation    823.0  20.7% of revenue
Net income                  177.6
Diluted shares (mm)         366.9

─────────────────── Enterprise value bridge ────────────────────
Share price                     221.7
Diluted shares (mm)             366.9
Equity value                 81,356.7
+ Straight debt                   0.0
+ Convertible notes               0.0
+ Operating leases                0.0
- Cash and equivalents        (435.0)
- Short-term investments    (4,550.5)
Enterprise value             76,371.2

Convention: operating leases excluded, EBITDA basis (ASC 842).
Notes
  - Convertible notes of 986mm are in the money at 221.72 against a
    148.15 conversion price. Under ASU 2020-06 their shares are already
    in diluted WASO, so they are carried as equity, not debt.
  - Checked: conversion would create 6.7mm shares and diluted WASO runs
    14.6mm above basic, so the shares are inside the count the equity
    value is built on.

──────────────────────── WACC buildup ────────────────────────
Input                    Value  Source
Risk-free rate           4.83%  US Treasury 10Y constant maturity, 2026-09-09
Equity risk premium      5.00%  assumption: market.equity_risk_premium
Levered beta             1.387  median of 6 peer unlevered betas, relevered
Pre-tax cost of debt    11.27%  synthetic: EBIT/interest of 1.4x maps to B3/B-
Weight of equity          1.00  E / (D + E) on 81,357mm
Cost of equity          11.76%  CAPM: risk-free + beta x ERP + size premium
WACC                    11.76%  100.0% x 11.76% + 0.0% x 8.57%

Notes
  - DDOG is net cash by 4,985mm. Weights still use gross debt, because the
    tax shield attaches to debt outstanding and not to a net position.
  - Peer regressions: R-squared runs 0.12 to 0.24 and the slope standard
    error 0.28 to 0.39 across 6 names, each on at least 104 weekly
    observations. The median asset beta pools these estimates; it does
    not sharpen any one of them.

────────────────────── Discounted cash flow ──────────────────────
Enterprise value, Gordon growth  10,642
Enterprise value, exit multiple  41,494
Enterprise value, value driver    9,045

  Implied per share, Gordon growth   $42.59  <- headline
  Implied per share, exit multiple  $126.67
  Implied per share, value driver    $38.24

Cross-checks
  - The Gordon terminal value implies an exit multiple of 7.9x terminal
    EBITDA of 1,906mm, restated onto the exit method's whole-period
    discount clock so the comparison is like for like. The peer median
    today is 36.1x, a gap of -78%.
  - The 36.1x exit multiple implies perpetuity growth of 9.60% against a
    WACC of 11.76%.
  - FLAG: Gordon terminal value is 81% of enterprise value, above the 75%
    mark. The valuation is a bet on the terminal assumption, not on the
    forecast.
  - Steady-state reinvestment at 2.50% growth is 69mm on 1,361mm of NOPAT,
    a reinvestment rate of 5.1%. It implies a terminal ROIC of 49.0%
    against a WACC of 11.76%, so terminal growth creates value and the
    assumption hangs together.
  - The value driver runs at ROIC equal to the 11.76% WACC, so its
    terminal value is NOPAT/WACC, 11,569mm, and the 2.50% growth rate
    changes it by nothing at all. Growth priced at the cost of capital is
    worth exactly what it costs.
```

The Datadog run is worth reading as a result, not just as a demo. On GAAP
economics with stock compensation expensed, the DCF lands far below the market
price, the implied exit multiple from the perpetuity is roughly a fifth of what
the peer set trades at, and four fifths of the value sits in the terminal
assumption. The engine says all of that on screen rather than printing a single
number, and it reads the g = ROIC x reinvestment identity off a g-consistent
steady state rather than off a terminal year still growing at eight percent. The
three terminal values span $38.24 to $126.67 on identical cash flows, which is
the measure of what the terminal method alone is carrying. Flipping
`sbc_treatment` to `addback` takes the headline from $42.59 to $79.80, which is
the honest measure of what that one accounting judgment is worth.

---

## The judgment calls

These decisions move the answer more than anything else in the code.
`docs/methodology.md` argues each one at length; this is the summary.

**Stock-based compensation is expensed, not added back, and the two camps do not
converge.** GAAP EBIT is already net of SBC, so the default adds nothing back
and free cash flow is struck after the full cost of paying employees. The
opposing camp adds it back as a non-cash charge. That is only honest alongside a
growing share count, so `sbc_dilution` is on by default: every projected year
issues `SBC dollars / share price` new shares and the count carried forward
grows by them. Datadog reads $42.59 expensed, $86.41 added back with no
dilution, and $79.80 added back with the dilution modelled. Modelling the shares
closes about 15% of the gap at this discount rate, and 13% at the 10.5% WACC the methodology worked through; the share of it that closes moves with the rate, but never far and the two camps are still a factor of two apart.

That failure to converge is structural, not a calibration problem, and it is the
most useful thing in this package. Roughly 70% of the addback's uplift in
enterprise value sits in the terminal value, which capitalises the addback in
perpetuity, while the modelled issuance stops at year five. Five years of
issuance is 8.3% of the register, and 8.3% cannot pay for a perpetuity. The
checks say so directly: the terminal year issues 1.87% of the share count in
stock compensation, so beside 2.50% terminal growth cash flow per share
compounds at 0.62%, not 2.50%. That is the number the addback camp has to
defend, and a model that reports $79.80 without it is reporting the generous
reading of its own assumption.

**The share count is diluted WASO by default and a treasury stock count on
request.** WASO is the right denominator for reported EPS and the wrong one for
a valuation struck today: it is an average over a past window, and it carries
the awards outstanding across that window rather than the ones outstanding now.
Set `dilution.method: treasury_stock` and the count is built from shares
actually outstanding in the latest filing plus the net new shares in-the-money
awards would create at today's price, under ASC 260. For Datadog that is 359.08mm
outstanding plus 1.55mm net from options plus 17.10mm unvested RSUs, so 377.73mm
against a 366.93mm diluted weighted average. The equity value was 2.9% light.
Options are valued by exercise-price band where the filer tags bands, because a
single weighted-average strike credits the company with exercise proceeds from
options nobody would exercise and so understates dilution, by six-fold on the
worked case in the tests. WASO stays the default because it is available for
every filer and needs no instance document.

**The terminal value is computed three ways on every run.** Gordon capitalises
the final projected flow and leaves the implied return on capital to be checked
afterwards. The exit multiple applies the peer median. The value driver inverts
the Gordon assumption: `TV = NOPAT_{N+1} * (1 - g/ROIC) / (WACC - g)`, so
reinvestment is derived from the return you will defend rather than assumed
separately, and the terminal value cannot embed a return nobody signed up for.
With `terminal_roic` null it falls back to the WACC, the competitive-equilibrium
view, and at ROIC equal to WACC it collapses to `NOPAT / WACC` for any g at all.
Datadog reads $38.24 there against $42.59 on Gordon. `terminal.method` picks
which one drives the headline; all three are always reported.

**Operating leases are excluded from debt, and EBITDA is the pairing.** Under
ASC 842 the lease charge stays inside operating income as straight-line rent. It
is not split into depreciation and interest the way IFRS 16 requires, so US-GAAP
EBITDA is already *after* rent. Adding the liability to debt while dividing by
that EBITDA charges the lease twice. Set `capitalize_operating_leases: true` and
every multiple switches to EBITDAR automatically; the engine will not let the two
conventions mix. Finance leases are debt either way.

**In-the-money convertibles are equity, not debt.** ASU 2020-06 made if-converted
mandatory, so a dilutive convertible's shares are already inside diluted
weighted-average shares. Counting the principal as debt on top of that
double-counts the instrument, by $986mm in Datadog's case.

**Beta is the median of peer asset betas, relevered.** Each peer's levered beta
is unlevered at its own D/E and tax rate, the median of those is taken, and the
result is relevered at the target's capital structure, so what gets pooled is
business risk rather than financing policy. Regressions are joined on the dates
the two series actually share, never zipped by position, and R-squared and the
standard error are printed because a beta of 1.42 on an R-squared of 0.11 is not
something the market told you with confidence. `peer_beta_method: vasicek`
shrinks each asset beta toward the cross-sectional mean by a weight that is the
ratio of the dispersion across peers to that dispersion plus the peer's own
sampling variance, so a noisy regression is pulled hard and a tight one is
barely moved. On the six committed peers it pools to an asset beta of 1.411
against 1.369 for the median, worth 21 basis points on the cost of equity, and
the reason is MongoDB: the widest standard error in the set, so it keeps least
of its own estimate and comes back from 1.914 to 1.602. The median could not see
that, because a median only reads the middle of the sorted list. Note also the
direction. Shrinkage pulled the highest beta down hard and the pooled figure
still came out above the median, because the mean of this set sits above its
middle. The median stays the default because one broken regression, a peer in
a takeover or three weeks of a short squeeze, can move a mean and cannot move a
median.

**Mid-year discounting, with one asymmetry.** Explicit flows discount at
`(1+w)^-(t-0.5)`. The Gordon terminal value carries the same half-year uplift,
because its perpetuity flows also arrive mid-year. The exit-multiple terminal
value does **not**: a multiple is a price struck at a date, so it discounts over
whole periods. Applying the mid-year factor to both overstates the exit-multiple
case by roughly half a year of WACC.

**The Monte Carlo draws jointly, and correlation widens the answer rather than
narrowing it.** The usual claim is that correlating the drivers narrows a DCF
distribution, on the reasoning that independence generates combinations that do
not occur. It generates them, but look at which ones. Value rises with growth,
rises with the terminal margin and falls with the WACC, so a positive
growth-margin correlation and negative correlations of both against the WACC all
point the same way: the good draws arrive together and so do the bad ones.
Independence puts mass on the offsetting middle, which is where the point
estimate already sits. The shipped matrix widens the standard deviation by about
21% against independent draws, on each of the four companies committed as
fixtures. Narrowing would require believing growth carries its own discount rate
with it. That is a defensible matrix and it is one line to enter, but it is not
the one this module argues.

**Purchase accounting: the deferred tax liability makes goodwill bigger, not
smaller.** With `merger.purchase_accounting.enabled`, the excess of the purchase
price over book equity acquired goes first to identifiable intangibles at the
configured share, and the rest is goodwill. A stock deal is tax free to the
seller and carries over the seller's tax basis, so the write-up exists for book
and not for tax, and that temporary difference is a liability assumed at close:

    DTL      = intangible write-up * tax rate
    goodwill = excess - intangibles + DTL

The plus sign is the part people get backwards. Goodwill is consideration less
the fair value of net assets acquired, and the DTL sits inside those net assets,
so booking it takes net assets down and pushes goodwill up. Screening Datadog
buying MongoDB at a 30% premium over the committed filings, the write-up is
$14,198mm, the DTL on it is $3,408mm at 24%, and goodwill comes out at $24,704mm
rather than the $21,297mm a model that skips the DTL would report. The liability
everybody forgets is a seventh of the goodwill it creates, and without it the
opening balance sheet does not balance. The amortisation of those intangibles is
a book expense with no tax basis behind it, so free cash flow gets back only
`(1 - t)` of the charge, and acquired deferred revenue is written down to fair
value, which permanently destroys revenue the target would have recognised.

---

## Beyond a single valuation

Every one of these is opt-in. A base case should be one set of assumptions you
can defend; these are the arguments about it.

**Monte Carlo.** `simulation.enabled` draws year-one revenue growth, the
terminal EBIT margin, the discount rate and terminal growth from one
multivariate normal, revalues on every draw, and reports percentiles of value
per share and of enterprise value. Ten thousand draws on Datadog put the 5th
percentile at $30.63, the median at $42.50 and the 95th at $60.12, against a
$221.72 market price, so P(value > price) is 0.0% and the 95th percentile does
not reach the quote. Draws where terminal growth lands at or above the discount
rate are rejected outright rather than clipped to the boundary, since clipping
would pile probability mass exactly where the perpetuity is largest. The
tornado is reported alongside and is deliberately univariate: it says which
assumption to argue about, the distribution says how wide the answer is once
they all move together.

**Adjusted present value.** `apv.enabled` values the business unlevered at the
cost of equity its asset beta implies, adds the tax shield as a stream in its
own right, and reconciles to the WACC answer line by line. The discount rate on
the shield is the whole argument: the cost of debt is Modigliani-Miller and
assumes a debt schedule fixed in dollars, while the unlevered cost of equity is
Harris-Pringle and assumes debt rebalanced to a constant share of value, which
is what a constant WACC already implies. Datadog is net cash, so the shield is
zero, Ku equals the levered cost of equity, and APV comes back at $10,642mm
against a WACC enterprise value of $10,642mm, a difference of +0.00%. That is
the identity case, and it is the one worth checking first.

**Point-in-time valuation.** `--as-of` on every command pushes a knowledge date
through the EDGAR client and the price series, so facts filed later are
discarded and prices stop there. It changes the company. Datadog valued as of
2026-03-01 shows the twelve months to 2025-12-31 and a $44.4mm operating loss;
with full knowledge the same company shows the twelve months to 2026-06-30 and a
$16.3mm operating profit. Those are different businesses to a model, and the
second one did not exist on 2026-03-01.

**Backtesting.** `techval backtest` values a list of names on past dates and
scores the upside against realised forward returns. Forward prices live in an
object the valuation path is never handed, so lookahead is structurally
impossible rather than a matter of discipline, and the provenance of every
`Financials` that comes out is walked afterwards to confirm no line item was
sourced from a filing dated after the valuation date. The statistic is Spearman
rather than Pearson, because the relationship between a DCF upside and a
subsequent return is monotonic at best and a handful of names at 300% upside
would carry a Pearson coefficient on their own. The harness is loud about
survivorship bias, overlapping windows and thin samples, and none of those is
fixed by it.

**Warranted multiples.** `comps.regression.enabled` fits EV/Revenue across the
peer set on growth and margin by OLS and reads off the multiple the target's own
fundamentals warrant, with the residual against the traded multiple as the thing
to argue about. It refuses below eight usable observations. The six-name peer
set shipped in `assumptions.example.yaml` produces five, and the engine declines
rather than fitting three parameters to five points.

---

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd techval
uv venv && uv pip install -e ".[dev]"
export TECHVAL_SEC_EMAIL=you@example.com   # SEC fair access requires a contact
cp assumptions.example.yaml assumptions.yaml
```

The SEC blocks automated clients that do not declare a contact address, so the
engine refuses to start without `TECHVAL_SEC_EMAIL`.

## Usage

```bash
uv run techval value DDOG --config assumptions.yaml   # full valuation + chart
uv run techval comps DDOG --config assumptions.yaml   # comp table only
uv run techval merger CRWD MDB --config assumptions.yaml
uv run techval fetch DDOG                             # warm cache, show provenance

# value the company as it was knowable on a past date
uv run techval value DDOG --config assumptions.yaml --as-of 2026-03-01

# re-run the model quarterly and score it on forward returns
uv run techval backtest DDOG,MDB,ZS,CRWD --config backtest.yaml \
    --from 2025-03-31 --to 2025-09-30 --horizon 252
```

`--as-of` is accepted by `value`, `comps`, `merger` and `fetch`. It discards
every fact filed after that date and stops the price series there, so the run
uses only information that existed at the time.

`backtest` takes valuation dates at quarter ends between `--from` and `--to`, or
an explicit list through `--dates`, and scores each against the close `--horizon`
calendar days later. It needs a risk-free rate it can date, so pin
`market.risk_free_rate` in the config or the run refuses to start rather than
applying today's yield to a 2025 valuation.

```
$ techval backtest DDOG,MDB,ZS,CRWD --config backtest.yaml \
      --from 2025-03-31 --to 2025-09-30 --horizon 252

──────────── Backtest: 4 names over 3 dates ────────────

(one row per name and date: price, value per share, upside, WACC, revenue,
 EBIT margin, exit date, exit price, forward return, and the reason where
 a valuation failed)

Observations valued                  10
Failed                                2
Skipped                               0
Scored against a forward return      10
Non-overlapping of those              4
Spearman rank IC                 +0.491
Hit rate                          40.0%

Cross-checks
  - 12 observations attempted: 10 valued, 2 failed, 0 skipped with no
    filings on record.
  - FLAG: the universe is survivorship biased. The honest fix is a
    point-in-time ticker list at each date; this run does not have one.
  - FLAG: n = 10, below the 30 mark. A rank correlation on a sample this
    size is dominated by sampling noise and will happily print a number
    that looks like skill. Treat anything below as an illustration of the
    machinery, not as evidence about the model.
  - FLAG: the 10 scored observations contain only 4 non-overlapping
    252-day windows per name. Overlapping holding periods share most of
    the same price path, so they are not independent draws and any
    significance computed from the raw n is overstated. Windows across
    different names in the same months are correlated too, through the
    market.
  - Spearman rank IC +0.491 on n = 10.
  - Hit rate 40% on n = 10 directional calls.
```

An IC of +0.491 on ten observations is noise, and the output says so three
different ways before it prints the number. The two failures are MongoDB at two
dates, where the filings on record then did not carry enough periods to tile a
trailing twelve months of diluted shares; a failure is recorded with its reason
rather than dropped, so the count of what was not valued stays visible.

`--no-cache` bypasses the HTTP cache. Everything else is in
`assumptions.yaml`: no rate, growth path, margin or multiple is hardcoded
anywhere in the engine.

Programmatic use returns dataclasses and DataFrames, with rendering confined to
the CLI:

```python
from techval.financials import build_financials
from techval.ev_bridge import build_ev_bridge
from techval.config import Assumptions

fin = build_financials("DDOG")
bridge = build_ev_bridge(fin, 225.27, Assumptions.load("assumptions.yaml"))
print(bridge.enterprise_value, bridge.multiple_denominator(fin))
```

---

## What the data layer handles that a naive one does not

Turning `companyfacts` into an income statement is most of the work, and these
are all observed in the four filings committed as test fixtures:

- **Retired tags.** A tag that ever appeared stays in the payload forever with a
  stale value. Balance-sheet concepts therefore resolve *as of a date*, and a
  tag whose newest fact is more than 20 days old is skipped. Zscaler reports
  marketable securities under `DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent`;
  resolving by first-tag-present returned zero and overstated its EV by $2.5bn.
- **Quarters that were never filed.** Q4 is always the year less nine months.
  Snowflake tags D&A only cumulatively; CrowdStrike files a year-to-date half
  plus a discrete Q2 with no discrete Q1. Periods are recovered by subtracting
  nested windows that share an endpoint, then tiled to twelve months exactly.
- **Stock splits.** CrowdStrike's four-for-one in mid-2026 leaves pre-split share
  counts in periods no later filing restates. Mixing them gave a TTM diluted
  count of 271mm against a true 1,023mm. Splits are detected from the filings
  themselves and earlier facts restated into current units.
- **Averages that are not sums.** Weighted-average share counts are aggregated as
  integrals (average times days) so they can be added and subtracted at all.
- **Dimensioned facts are absent from `companyfacts`.** That endpoint publishes
  undimensioned facts only, so option counts under the award-type and plan axes
  and a dual-class issuer's shares under the class-of-stock axis simply are not
  there. Datadog has no undimensioned `dei:EntityCommonStockSharesOutstanding` at
  all. `parse_instance` reads the filing's own instance document instead, which
  is where 334.90mm Class A and 24.17mm Class B are found and summed to 359.08mm.
  Reading either class alone understates the company by 7% or by 93%.
- **Tags that look like the one you want.** The award footnote is full of them.
  CrowdStrike tags an undimensioned
  `OptionsVestedAndExpectedToVestOutstandingWeightedAverageExercisePrice` at the
  same date as the real outstanding strike; Zscaler files its whole antidilutive
  securities table under `OptionsOutstandingNumber`, so 8.9mm RSUs and 1.8mm ESPP
  shares sit under an options tag next to the 150,000 options that are options.
  Tag matching is exact, and option counts are taken undimensioned or by
  exercise-price band and from nowhere else.
- **A roll-forward tags its opening balance with the closing balance's tag.**
  Datadog's option table carries 3,474,619 options at $7.26 and 1,610,360 at
  $7.83 under one tag, the first of them six months stale. Only the context date
  separates them, so the latest instant wins.

Every line item carries provenance: the tag that won, the periods used, the forms
they came from and the method. `techval fetch <TICKER>` prints the table.

---

## Limitations

Read these before quoting a number from this tool.

**The share count**

- **The treasury stock count degrades to WASO rather than guessing.** If the
  instance document cannot be read, if shares outstanding are not tagged, or if
  options are tagged only by award type, the result comes back as
  `diluted WASO fallback` with a flag naming what was missing. The award-type
  case is the common one, because a dimensioned option row cannot be told apart
  from an antidilutive-securities table reusing the same tag.
- **ESPP shares are never counted.** Neither is any award the filer discloses
  only in narrative text. Where bands are tagged but not all of them carry a
  strike, the single weighted-average strike is used and dilution is understated
  to the extent any band is out of the money.
- **The count is as of the last filing, not as of today.** Shares outstanding
  come off the latest cover page and awards off the latest balance sheet date,
  so anything issued since is missing. Forfeitures are not haircut either:
  `assumed_forfeiture_rate` defaults to zero because ASC 260 does not haircut
  and guessing a rate is a thumb on the scale.

**The discount rate**

- **Book value proxies the market value of debt**, in the EV bridge and in the
  WACC weights. Close for investment-grade paper near par, wrong for distressed.
- **Cost of debt is synthetic.** Interest coverage maps to a rating and a spread.
  Coverage is the wrong risk metric for a net-cash issuer: Datadog's 1.4x
  coverage implies a speculative rating against negligible actual default risk.
  The debt weight is near zero there, so it barely reaches the WACC, but for a
  company with real debt and real cash you should override the rate.
- **The equity risk premium is an assumption**, not a measurement, and it is the
  largest single unobservable in the output.
- **Peer sets are hand-picked** in YAML. There is no automated screen, and
  comparability is your judgment. The warranted-multiple regression needs eight
  usable observations and the shipped six-name example produces five, so it
  declines. Widening the set is your job, and a set wide enough to regress on is
  a different exercise from a set tight enough to take a median from.

**The projection**

- **The NOL model stops short of three things that matter.** Section 382, which
  caps annual use of a carryforward after a change of control, so any deal case
  here overstates the shield. State carryforwards, with their own expiry and
  apportionment. And the valuation allowance, so the balance used is the gross
  federal carryforward rather than the deferred tax asset the filer believes it
  will realise. The residual balance at year N is not carried into the terminal
  value either, which is struck at the full rate.
- **Working capital is anchored on a ratio, not on the balance sheet.** The year
  one change is measured against `nwc_pct_revenue` applied to TTM revenue rather
  than against the reported balance, which keeps the path internally consistent
  at the cost of ignoring where the balance sheet actually starts.
- **The Monte Carlo shocks only year-one revenue growth.** The path still fades
  to an unshocked terminal growth rate, so the spread on the final explicit year
  is narrower than a parallel shift of the whole path would give. The exit
  multiple is not simulated at all, because sampling a peer median alongside
  terminal growth would put two competing terminal assumptions into one
  histogram. And the distribution is over assumptions, not over outcomes:
  nothing here models the chance that the business is a different business than
  the one the projection describes.
- **The Monte Carlo and the treasury stock count do not compose.** The
  vectorised path divides by diluted WASO. Set `dilution.method: treasury_stock`
  and the simulation refuses to run, because its reconciliation against
  `run_dcf` catches the 2.94% share-count gap and will not centre a distribution
  on a figure the rest of the engine does not report. Refusing is the right
  behaviour and it is still a missing feature.

**APV**

- **Debt is held flat at the current balance across the explicit period**, and
  interest on it is the WACC's own pre-tax cost of debt rather than the filed
  interest expense. Nothing in the assumptions describes a paydown, and inventing
  an amortisation schedule would put a financing forecast inside the module whose
  job is to expose one. The cost is that a filer whose coupon sits far below its
  synthetic yield, which is the zero-coupon convertible signature, gets a modelled
  shield larger than the deduction it will claim. The checks report that gap.
- **The terminal shield grows at g while the explicit period held debt flat.**
  That inconsistency is inside the schedule rather than hidden by it, it is
  stated on every run, and it vanishes when g is zero.

**Merger**

- **Book equity acquired is understated, so goodwill is overstated.** It is built
  from working capital less borrowings and any preferred or minority interest.
  Lease liabilities are left out because their right-of-use assets are not
  carried here and deducting one side alone would invent goodwill, and
  non-current operating assets are missing for the same reason. The deal
  therefore looks more dilutive here than it should, not less.
- **The multi-year roll-forward needs D&A, capex and pre-tax income** and refuses
  where a filer does not tag them. CrowdStrike as an acquirer fails on the D&A
  tag, which is the same gap that makes its EV/EBITDA non-meaningful in the comp
  table.
- **Left at its default the merger module is a screen.** No opening balance
  sheet, no goodwill, no deferred tax on the step-up, no deferred revenue
  haircut, no debt paydown. Year-one EPS on two sets of TTM figures tells you the
  shape of a deal and roughly what it costs. It is not a merger model, and no one
  should take a bid to a board on it.

**The backtest**

- **The universe is survivorship biased and this harness does not fix it.** The
  ticker list is supplied by the caller, so it contains only companies that exist
  today. Names delisted, acquired at a discount or wound up during the sample are
  absent and are disproportionately the losers, so every statistic is biased
  upward by an amount the harness cannot measure. The honest fix is a
  point-in-time universe at each valuation date, and there is not one here.
- **Overlapping windows are not independent draws.** Valuing the same name
  quarterly on a one-year horizon shares most of the price path between
  observations, and names in the same months are correlated through the market,
  so any significance computed from the raw n is overstated. The count of
  non-overlapping windows is reported beside it.
- **Returns are price returns**, not total returns. No name in this universe pays
  a material dividend, but the convention is stated rather than assumed.
- **The risk-free rate is held flat across dates** unless you supply one per
  date, so measured upsides move with fundamentals and the share price and not
  with the rates that were actually quoted.

**Scope**

- **US GAAP only.** No foreign private issuers, no IFRS, no 20-F filers.

---

## Tests

```bash
uv run pytest
```

No test touches the network. Fixtures are frozen SEC `companyfacts` payloads,
pruned XBRL instance documents, and Nasdaq daily closes retrieved 2026-09-10,
cut to the concepts the engine reads and to facts from 2023 onward. The four
companies are chosen for what they break: Datadog for in-the-money convertibles,
thin GAAP EBITDA and a dual-class share count, CrowdStrike for a January year end
plus a four-for-one split and no combined D&A tag, MongoDB for finance leases,
Zscaler for a retired-tag balance and for out-of-the-money options filed under an
antidilutive-securities table.

The accretion/dilution test checks against a worked example whose arithmetic is
written out in the test docstring, and separately asserts that substituting the
solved breakeven synergy figure returns pro forma EPS to standalone. The
simulation reconciles its vectorised central draw against `run_dcf` on every
run and raises rather than prints when the two disagree. The APV reconciliation
is tested on a constructed constant-leverage firm where the identity has to hold
exactly, so a failure there is arithmetic rather than opinion.

---

## Data sources and SEC fair access

- **Fundamentals**: `data.sec.gov/api/xbrl/companyfacts` for undimensioned facts,
  and the filing's own XBRL instance document for the dimensioned ones that
  endpoint drops. Requests declare a contact in the `User-Agent` as the SEC
  requires, are throttled to 8 per second against a published limit of 10, and
  back off exponentially on 429 and 503.
- **Prices**: Nasdaq's public quote API. No key, no account. One request per
  ticker per run.
- **Risk-free rate**: US Treasury daily yield curve, 10-year constant maturity.

Nasdaq and Treasury both reject clients whose `User-Agent` they do not
recognise, and neither publishes a fair-access policy of the kind the SEC does,
so requests to those two hosts carry a browser string. That is stated here
rather than left in the code, because the SEC client deliberately declares a
real contact address and a reader should know the two are not held to the same
standard.

The package originally targeted Stooq for prices. Stooq now answers plain HTTP
clients with a JavaScript proof-of-work challenge instead of CSV. Defeating a bot
check the operator deliberately erected is not something this package will do, so
the Stooq source raises an explanatory error and points at the alternatives.
Sources sit behind one interface, and a local CSV source makes a valuation
reproducible offline years later, which is also what the backtest runs on.

All HTTP responses are cached by request URL under `~/.techval/cache`, so the
same ticker and the same assumptions produce identical output on every run.

## Licence

MIT.
