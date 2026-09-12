# techval

![Datadog football field](out/DDOG_football.png)

*Both DCF bands sitting far below the traded price ARE the result, not a broken
chart: on GAAP economics the model will not pay what the market pays, and that
gap is what the rest of this document is about. The chart carries two of the
engine's three terminal values; the third prints in the run below.*

A valuation engine for technology, media and telecommunications, built entirely
on free data. SEC EDGAR for fundamentals, Nasdaq's public quote API for prices,
the US Treasury daily curve for the risk-free rate. Fourteen commands, 1,701
tests, and no paid terminal anywhere in the stack.

One half is the analyst work, meant to be defensible line by line. Normalized
trailing statements out of raw XBRL. An enterprise value bridge. A CAPM cost of
capital off peer beta regressions. An unlevered DCF carrying three terminal
values side by side. Trading comps with a warranted multiple regressed on
fundamentals, and an accretion/dilution screen that will build the full opening
balance sheet when asked.

The other half is what a generic DCF tool does not have. A TMT data layer reads
what the `companyfacts` endpoint will not publish: segment and geography
economics out of dimensioned XBRL, operating KPIs out of tags and prose, a sum
of the parts, and a precedent transaction database built out of merger filings.
Four models sit on top of it, fitted on those same filings: a peer similarity
encoder, a warranted trading multiple, a revenue growth fade curve and an
acquisition propensity screen. Each is scored walk-forward against a baseline it
has to beat. The scoreboard below says which ones cleared it.

Every number traces back to a filing, a quote, or an assumption you wrote down.
Where a figure cannot be sourced the engine raises an error naming the XBRL
concept and the tags it tried. It does not interpolate.

> **Start with [the value signal](#the-value-signal).** The one thing here
> tested directly against forward returns came back negative, and then came back
> not significant once the overlapping windows were corrected. It is the result
> this harness was not built to produce, and everything else should be read in
> its light.

[Sample output](#sample-output) ·
[What the models are worth](#what-the-models-are-worth) ·
[The TMT data layer](#the-tmt-data-layer) ·
[The judgment calls](#the-judgment-calls) ·
[Beyond a single valuation](#beyond-a-single-valuation) ·
[Install](#install) ·
[Usage](#usage) ·
[What the data layer handles](#what-the-data-layer-handles-that-a-naive-one-does-not) ·
[Limitations](#limitations) ·
[Tests](#tests) ·
[Data sources](#data-sources-and-sec-fair-access)

---

## Sample output

Struck on 11 September 2026 against filings through the June 2026 quarter. Every
figure quoted in this README was produced by running the code on this tree, so
the market-dependent ones move with the price. Output blocks throughout are
trimmed for length and the figures in them are verbatim.

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
Capital expenditure          46.9
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
Risk-free rate           4.95%  US Treasury 10Y constant maturity, 2026-09-10
Equity risk premium      5.00%  assumption: market.equity_risk_premium
Levered beta             1.387  median of 6 peer unlevered betas, relevered
Pre-tax cost of debt    11.39%  synthetic: EBIT/interest of 1.4x maps to B3/B-
Weight of equity          1.00  E / (D + E) on 81,357mm
Cost of equity          11.88%  CAPM: risk-free + beta x ERP + size premium
WACC                    11.88%  100.0% x 11.88% + 0.0% x 8.66%

Notes
  - DDOG is net cash by 4,985mm. Weights still use gross debt, because the
    tax shield attaches to debt outstanding and not to a net position.
  - Peer regressions: R-squared runs 0.12 to 0.24 and the slope standard
    error 0.28 to 0.39 across 6 names, each on at least 104 weekly
    observations. The median asset beta pools these estimates; it does
    not sharpen any one of them.

────────────────────── Discounted cash flow ──────────────────────
Projection, USD mm
                 Year 1  Year 2  Year 3  Year 4  Year 5
Revenue           4,839   5,735   6,595   7,353   7,942
Growth            22.0%   18.5%   15.0%   11.5%    8.0%
EBIT               19.9     333     739   1,221   1,747
NOPAT              15.1     253     562     928   1,328
FCFF               19.1     250     547     900   1,284
PV                 18.1     211     413     607     774

Enterprise value, Gordon growth  10,484
Enterprise value, exit multiple  41,276
Enterprise value, value driver    8,934

  Implied per share, Gordon growth   $42.16  <- headline
  Implied per share, exit multiple  $126.08
  Implied per share, value driver    $37.93

Cross-checks
  - The Gordon terminal value implies an exit multiple of 7.8x terminal
    EBITDA of 1,906mm, restated onto the exit method's whole-period
    discount clock so the comparison is like for like. The peer median
    today is 36.1x, a gap of -78%.
  - The 36.1x exit multiple implies perpetuity growth of 9.72% against a
    WACC of 11.88%.
  - FLAG: Gordon terminal value is 81% of enterprise value, above the 75%
    mark. The valuation is a bet on the terminal assumption, not on the
    forecast.
  - Steady-state reinvestment at 2.50% growth is 69mm on 1,361mm of NOPAT,
    a reinvestment rate of 5.1%. It implies a terminal ROIC of 49.0%
    against a WACC of 11.88%, so terminal growth creates value and the
    assumption hangs together.
  - The value driver runs at ROIC equal to the 11.88% WACC, so its
    terminal value is NOPAT/WACC, 11,453mm, and the 2.50% growth rate
    changes it by nothing at all. Growth priced at the cost of capital is
    worth exactly what it costs.
```

Read the Datadog run as a result, not a demo. On GAAP economics with stock
compensation expensed, the DCF lands at a fifth of the market price, the implied
exit multiple from the perpetuity is roughly a fifth of what the peer set trades
at, and four fifths of the value sits in the terminal assumption. All of that
prints on screen. The engine also reads the g = ROIC x reinvestment identity off
a g-consistent steady state, not off a terminal year still growing at eight
percent. That is where most models quietly cheat.

Three terminal values spanning $37.93 to $126.08 on identical cash flows is the
measure of what the terminal method alone is carrying. Flip `sbc_treatment` to
`addback` and the headline goes from $42.16 to $78.88. That is what one
accounting judgment is worth here.

---

## What the models are worth

Four models are fitted on the filings this engine already aggregates. A fifth
piece, the signal harness, exists to test any score at all against forward
returns, including these. No score is quoted without the baseline it was scored
against, and every one is walk-forward by date with the training window cut
strictly before the test window less an embargo.

| Model | Metric | Score | Baseline | Baseline score | Verdict |
|---|---|---|---|---|---|
| Value signal, trailing EV/Revenue | mean IC, 12m forward | **-0.0984** | random score, same cross-sectional shape | +0.0002 | loses, and is **not significant** once overlap is corrected |
| Peer similarity encoder | NDCG@10 | **0.5407** | popularity prior, query ignored | 0.2213 | beats it on 84% of targets, paired t 13.6 |
| Warranted EV/Revenue | Spearman, level | **0.7651** | `comps.py` OLS refit on sub-vertical peers | 0.6313 | beats it on the 888 observations both can score, and adds only **+0.0420** on the change |
| Revenue fade, 1 year | mean absolute error | **0.1474** | last year's growth carried forward | 0.1510 | ties it, inside the fold noise |
| Revenue fade, 3 years | mean absolute error | **0.1391** | last year's growth carried forward | 0.1921 | beats it, outside the fold noise |
| Acquisition propensity | AUC | **0.5685** | sort the universe smallest first | 0.5474 | lift of +0.0212 against a fold sd of 0.0910, so it **ties** |

Two of the six beat their baseline cleanly, one beats it on the level and adds
almost nothing on the change, two tie, and one loses. That is about the hit rate
a panel this size should produce. A scoreboard on which every row won would be
evidence of a harness grading its own homework.

### The value signal

Cheapness on trailing EV/Revenue, 100 technology, media and telecommunications
companies, 35 quarterly cross-sections from December 2016 to June 2025, twelve
month forward returns, 2,604 scored company-dates. `ml.signals.test_signal` on
the recorded panel prints:

```
cheapness (negative trailing EV/Revenue): mean IC -0.0984 over 35 dates (2,604
company-dates, about 8 independent 12 month periods). Share of dates positive
43%. t = -2.65 naive, -1.62 Newey-West at lag 3; the correction moved the
standard error by 1.64x, leaving about 13 effective observations. The naive
statistic clears two and the corrected one does not, so the honest answer is
NOT SIGNIFICANT. The naive figure is printed above only so the size of the
difference is visible. Top-bottom spread -5.38% over 12 months (t = -0.79),
monotone across 25% of the steps. mean information coefficient of -0.0984
against 0.0002 for a random score with the same cross-sectional shape, 400
permutations within date: the model does NOT beat the baseline on 2,604
observations. Use the baseline.
```

Two findings sit in that paragraph and they point opposite ways.

Value was a negative signal in technology over this decade. The cheap half of
the universe underperformed the expensive half by about five points a year and
the bucket table is close to monotone in the wrong direction. Anyone who lived
through 2017 to 2021 remembers it. It is also the opposite of what the textbook
says value does.

And it is not evidence. The naive t-statistic is -2.65 and would have been
written up. The Newey-West figure at lag 3 is -1.62 and would not. Twelve month
returns sampled quarterly share three quarters of their path with the next
observation, so the coefficient series is autocorrelated by construction: the
measured first-order autocorrelation is +0.76 against a theoretical 0.75, about
as clean a confirmation of the design as this sample can give. Thirty five
overlapping quarterly readings are about 8.5 non-overlapping annual periods and
about 13 effective observations. Both t-statistics print, the verdict says NOT
SIGNIFICANT in those words, and neither number can be quoted without the other.

The corrected statistic is itself an undercorrection, and the module says so. A
Bartlett kernel truncated at the matched lag weights the three autocovariances
by 0.75, 0.50 and 0.25, so it recovers about 1.66 of the true factor of 2 with
infinite data and about 1.55 at this sample size. Reaching for a longer lag does
not help at forty observations, because the sample autocovariances at high lags
are estimated from a handful of products and shrink toward zero.
`test_the_bartlett_correction_recovers_most_but_not_all_of_the_inflation`
measures both regimes.

A permutation baseline runs beside the kernel and disagrees with it on this
panel. Not one of 400 within-date permutations reaches the observed coefficient,
so the permutation p-value hits its floor of 1/401 while Newey-West says p =
0.11. Neither is broken. Permuting inside a date leaves the dates independent of
each other by construction, so the permutation null reproduces the
cross-sectional dependence and not the time-series autocorrelation. It is the
right null for whether any cross-sectional relation exists and the wrong one for
how precisely the mean of 35 overlapping coefficients is measured. Both are
reported and the Newey-West interval is the statistic of record.

The checks that print beside the number are as much the result as the number:

```
35 rebalance dates spanning 2016-12-15 to 2025-06-15, about 8.5 non-
overlapping 12 month periods. The standard error is quoted against roughly 13
effective observations, not against the 2,604 company-dates.
Newey-West at lag 3 against a 92 day rebalance spacing. First-order
autocorrelation of the coefficient series +0.76; the correction multiplied the
standard error by 1.64x.
Moving-block bootstrap standard error 0.0624 against 0.0608 from Newey-West.
The two should agree to within a fifth; where they do not, the lag is the
thing to doubt.
2,604 holding periods ran their course, 0 were terminated at a deal, 0 ended
in a delisting with no deal on record, 388 were censored at the edge of the
price data, 0 had no usable price.
FLAG: not one name in this sample was acquired or delisted over the whole
period. That is not what happens to a technology universe over a decade, so
the universe is a list of today's survivors and every number here is biased
upward by the outcomes it cannot see.
```

That last flag is the limitation this package cannot fix. The delisting
machinery is built, tested and exercised, and on the real panel it finds nothing
to do, because the data sources delete a company the day it stops trading. Five
of the 110 names in the seed universe, EA, FI, FYBR, IPG and JNPR, are missing
from the SEC's own `company_tickers.json` and return no rows from the price
endpoint, and every one of the five left to an acquisition. Juniper's CIK still
resolves through the committed former-tickers index, which is why the taxonomy
section further down counts only four unresolved names, but a CIK is not a
price history. None of the five is in the panel at all, so the harness cannot
terminate them at a deal. The direction matters for reading the headline:
the missing names are takeouts, takeouts earn a premium and skew cheap, so their
absence flatters cheapness and the true coefficient is if anything more negative
than -0.098.

One correction to the received wisdom, and it is measured. Overlapping
windows alone do not inflate the t-statistic. Overlapping windows plus a
*persistent score* do. The coefficient is a rank correlation inside one date, so
a market move common to every name cancels out of it; what survives from one
date to the next is the part of the ordering that persisted. Hand the harness a
score redrawn from noise at every rebalance and the inflation factor comes back
at 1.0 whatever the returns are doing. That sharpens the warning, because a
valuation multiple is about as persistent as a company characteristic gets, and
so is every other score in this package.

### The peer similarity encoder

Two towers into one cosine space, trained with InfoNCE on the compensation peer
groups filers disclose in their own DEF 14A proxies. The label is the point. A
peer set built on my own opinion of which companies are comparable would make
the model a restatement of the prior and the backtest a tautology. A
compensation peer table is a dated, auditable assertion of comparability by a
board that had to sign it.

Walk forward by proxy filing date, five folds, a 365-day embargo, every
transform refitted inside each fold, 162 scorable targets:

```
             method  ndcg@10  ndcg_sd  precision@10  recall@10  n_queries
            encoder 0.540665 0.255327      0.479630   0.301489        162
fundamentals cosine 0.372876 0.234782      0.325309   0.201466        162
        text cosine 0.310841 0.191535      0.274691   0.173802        162
    text lsa cosine 0.282363 0.195556      0.239506   0.151051        162
    size and growth 0.248605 0.203738      0.236420   0.148277        162
   popularity prior 0.221331 0.240940      0.211111   0.131127        162
  same sub-vertical 0.201501 0.197943      0.181481   0.112323        162
```

The lift over the popularity prior is +0.3193 on 162 scorable targets. Read the
`ndcg_sd` column for what it is: the spread of per-query scores across targets,
dominated by how findable each target's peers are, which is variation every
method shares. It is a spread, not an error bar on the lift. The statistic that
answers whether the encoder beats a baseline on the same company is the paired,
query-by-query difference, and it is unambiguous: the encoder beats the
popularity prior on 84% of the 162 targets, paired t 13.6, and beats the
strongest baseline, fundamentals cosine, on 74% of them, paired t 7.9.

The popularity trap is measured, not merely avoided. A model can beat a weak
baseline by imitating it, so `popularity_collapse` reports the rank correlation
between the model's per-query orderings and the global popularity order. It is
0.146. At 1.0 the model returns the same list whatever it is asked. It does not.

The size gate is in the product and out of the measurement, and the gap is on
the record. A compensation committee picks inside a revenue and market
capitalisation band to begin with, so gating the candidate list on size hands
every method a large part of the answer. Gated, NDCG@10 is 0.5536 against 0.5407
ungated. That 0.013 is what the gate is worth, and it is a good deal less than
the free lift a gated evaluation would have claimed.

This table disagrees with the one recorded when the module merged, and the cause
is in the fixture. Pull request #17 measured 0.594 against the same 0.221
popularity prior on the live corpus. The committed fixture keeps only the first
2,500 characters of each Item 1, the file's stated pruning rule, so the text
tower here is reading an excerpt of a business description. Cutting further
confirms the mechanism:

| Item 1 truncated at | encoder | text cosine | text lsa | same sub-vertical | fundamentals cosine | size and growth | popularity |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2,500 characters | 0.5407 | 0.3108 | 0.2824 | 0.2015 | 0.3729 | 0.2486 | 0.2213 |
| 1,500 characters | 0.5342 | 0.3078 | 0.2729 | 0.1904 | 0.3729 | 0.2486 | 0.2213 |
| 800 characters | 0.5127 | 0.2791 | 0.2620 | 0.1561 | 0.3729 | 0.2486 | 0.2213 |

Every method that reads the text falls with the cut, including the sub-vertical
baseline, which classifies from the business description as well as from the SIC
code. The three that read no text at all are bit-identical across all three
columns. **0.5407 is the figure that reproduces offline from this repository and
it is the one quoted above.** 0.594 is what the same code scored on the
untruncated documents and cannot be reproduced without refetching them.

Two further slices belong beside the headline. Splitting the queries by whether
the target ever appeared in a training group separates warm start from cold
start, and the cold-start score is the one worth quoting for a new coverage
name. And `ablate_towers` refits without each tower. On the committed fixtures
the full model scores 0.4241 with a fold standard deviation of 0.0850; dropping
the text tower costs 0.0643 against a dispersion of 0.0636, and dropping the
fundamentals tower costs 0.0387 against 0.0690. So the text tower is the more
important of the two, its edge over the noise is a hair, and the fundamentals
tower is not shown to help but is not shown to hurt either. Part of its
weakness is the panel. It is built with no price feed at all, so 14 of the 50
features are missing on every row, market capitalisation among them. The
fundamentals tower is being judged on a degraded input and the model card says
so.

### The warranted trading multiple

What the market pays for a bundle of characteristics, and whether a company sits
above or below that line. 94 US TMT filers at 22 quarter ends from March 2021 to
June 2026, 1,764 company-quarters, enterprise value through the engine's own
bridge and features from the feature store. `WarrantedModel.verdict()` prints
the headline and the number that deflates it in the same paragraph,
deliberately, so the two cannot be quoted apart:

```
2% of the variance in the log multiple sits between the 22 dates rather than
inside them: a model given the date and nothing else scores an R-squared of
0.02 against the raw multiple. That is the re-rating, and it is not a
statement about any company. warranted EV/Revenue (mlp): fitted EV/Revenue
from fundamentals across the TMT universe, and the out-of-sample residual
against it, fitted on 1,509 observations through 2025-09-30 across 25
features. spearman of 0.7651 against 0.6313 for comps.py OLS refit on the
date, sub-vertical peers, a lift of +0.1339 on 888 observations. Fold standard
deviation 0.0841, so the lift is outside the fold-to-fold noise. Differenced
against each company's own previous observation the rank correlation is
+0.0420 on 1,548 observations, so almost all of the score above is the level
inherited from history rather than a view about the change. Treat the residual
as a description of where a company sits, not as a forecast of where it is
going.
```

0.7651 is the level, and it is the level on the 888 observations where the
incumbent OLS can be refit at all. Every baseline is scorable on a different
subsample, so the same model carries more than one Spearman in this section:
0.7651 against the sub-vertical OLS, 0.8391 against the OLS refit on the whole
cross-section, 0.8294 against the sub-vertical median, 0.8412 against
persistence. 0.0420 is the change, and it is the honest size of the
contribution. A feature vector barely moves in three months and neither does
a relative multiple, so a model fitted on the past is rewarded for recognising a
name as much as for understanding it. That is not lookahead, since the training
window is strictly earlier, but a reader shown 0.84 will believe far more than
the model earned. The practical consequence prints on every read: the residual
describes where a company sits, it does not forecast that the gap will close.
Testing whether it closes needs forward returns with the full horizon as an
embargo. That is what the signal harness above is for, and it is a different
piece of work.

The pooled score is also mostly the ordering of the buckets, so the card reports
the harder number beside it. Inside one sub-vertical on one date, averaged over
the 106 cross-sections carrying at least eight names, the model scores **+0.6773
against +0.3537** for the same OLS. That is the comparison a screen actually
lives on. The sub-vertical median scores -0.8828 there, for a structural reason:
a median assigns every member of its group the same value and so carries no
ordering inside it at all. That is the case against quoting one number for a
whole comp set, stated as a number.

Carrying a company's own multiple forward scores 0.9722, higher than the model,
and it is reported loudly with its own note because a reader deserves to know
the ceiling. It is not what the card is scored against, because it is built from
the company's own price, and a warranted multiple exists to be differenced
against that price. Separately, the features that *are* the answer are refused
outright: the target is log EV less log revenue, so log market capitalisation,
log enterprise value and the three capital ratios denominated in them sit in
`BANNED_FEATURES` with the reason, and `fit_warranted` raises on them instead of
returning a superb R-squared carrying no information.

One more thing this panel had to solve before any of it was measurable. 151 of
the 1,764 observations carry a share count on a pre-split basis against a price
series the vendor has already restated, so the equity value was converted onto
the price's basis before the multiple was struck. Without the conversion each of
those observations is understated by its whole split ratio.

### The revenue growth fade curve

Fitted on 224 TMT filers, 2,798 fiscal years ending between 2007 and 2026, with
an embargo of 365 days per year of horizon:

```
Next year's growth = +0.0693 + 0.4719 x this year's, so growth fades 53% of the
way to 13.1% each year, a half-life of 0.92 years.

224 filers, median 12 fiscal years of revenue each, deepest 19, 154 with ten
or more
h=1: model 0.1474, persistence 0.1510, training mean 0.1734, sub-vertical mean
0.1693 (mean absolute error, lower is better)
h=2: model 0.1476, persistence 0.1884, training mean 0.1583, sub-vertical mean
0.1578 (mean absolute error, lower is better)
h=3: model 0.1391, persistence 0.1921, training mean 0.1509, sub-vertical mean
0.1441 (mean absolute error, lower is better)
```

At one year out the model does not beat doing nothing, and a test asserts the
tie so it cannot drift away quietly. At two and three years it beats it clearly.
Read that pattern as the result. Persistence gets worse as the horizon lengthens
while the model gets better, because growth is sticky one year out and by year
three last year's number is actively misleading. A fade curve is a claim about
the second regime, and a five-year DCF spends four of its five years there.

The regression is a claim about a panel. The decile table assumes no functional
form at all and is the version to quote in an argument:

```
 Observations  Trailing growth  Forward growth  Forward median      Fade
          233        -0.156930        0.073345        0.021471  0.230275
          233        -0.024851        0.043805        0.027609  0.068657
          234         0.020572        0.054103        0.034160  0.033531
          233         0.055228        0.074949        0.056487  0.019721
          234         0.091277        0.072564        0.077876 -0.018713
          233         0.136007        0.129103        0.115606 -0.006905
          233         0.194555        0.178324        0.157865 -0.016232
          234         0.269022        0.222623        0.225660 -0.046399
          233         0.395265        0.284178        0.289149 -0.111087
          234         0.897151        0.474237        0.460731 -0.422915
```

Both ends move toward the middle and the crossing point sits between 9 and 14
percent. The fitted line has a fixed point at 13.1 percent, where growth settles
if nothing else changes, and the shipped default fades on a straight line from
20 percent to 5 percent over five years. The two differ less in speed than in
where they come to rest. The filings revert toward a growth sector's own mean
and no perpetuity can carry that; the typed schedule walks all the way down to
something a terminal value can. So the fitted models govern only
`ml.forecast.horizon_years` and the path then falls back to the engine's own
line toward `dcf.revenue_growth_terminal`, anchored on a measured level, with
`GrowthPath.basis` saying `fitted` or `assumed` on every single year so the
handover is visible instead of blended away.

The sample construction is worth as much as the model. Delisted TMT filers are
in the panel, kept until the day they stop filing and keyed on CIK, which
survives a delisting. They are 1,224 of the 2,798 observations, 44 percent of
it. Their last observed year grows 9.8 percent against 15.8 percent for the
filers still quoted, so dropping them lifts mean forward growth by 0.78 points
at one year and 1.52 points at three. The bias compounds with the horizon,
exactly the wrong direction for a fade curve.

Acquisitive years are kept, not excluded. Excluding them fits a fade curve for a
world in which nobody does M&A and then hands it to a DCF valuing a company that
will keep doing it. Acquisition spend over revenue is a feature instead. The
effect is real and is a timing artefact: a year of heavy acquisition spend grows
2.1 points faster than a quiet one, and the year *after* it grows 5.9 points
faster, because a deal closing in June contributes six months to this year and
twelve to the next.

One expectation the panel overturned. Restatement of a filed revenue figure is
**rarer** than it is usually assumed to be. Holding the us-gaap tag fixed, 0.89
percent of the 2,798 fiscal years read differently today than in the filing that
first reported them, and 0.71 percent differ by more than one percent. What is
commoner, by a factor of two and a half at the one percent threshold, and far
larger when it happens, is the filer moving revenue to a different tag: through
the full ladder 2.6 percent of years move and 1.75 percent move by more than a
percent. Both are reported separately, because only one of them is about
accounting.

### The acquisition propensity screen

493 registrants, 9,400 labelled observations, 336 positives over 97 distinct
deals, base rate 3.57 percent, five walk-forward folds with a 462-day embargo:

```
auc of 0.5685 against 0.5474 for size only (log revenue, smallest first),
walk-forward AUC 0.5474, a lift of +0.0212 on 5,881 observations. Fold
standard deviation 0.0910, so the lift is inside the fold-to-fold noise.

Observations                         9400
Positives                            336
Distinct deals                       97
Base rate                            0.03574468085106383
Dropped inside the gap               102
Dropped after announcement           394
Dropped with an unresolved window    1802
```

The model ties the size sort. The lift is a fifth of the fold-to-fold
dispersion, and there is a harder reason not to believe it than that. Fixing two
extraction bugs in the precedent scanner added 7 transactions to a label set of
100, and the headline moved from "does not beat the baseline, use the baseline"
to "beats the baseline by +0.0212". Same code, same panel, same seed: the label
set moved by 7 percent and the sign of the conclusion changed. Both numbers were
recorded. That measures how much this sample can support, and the answer is not
a conclusion of this size. `coefficient_stability()` says the same thing from
the other end: nine of the fifteen coefficients change sign somewhere across the
five folds.

What is usable is the top of the list, the part a coverage banker reads.
Averaged across the 19 walk-forward dates, precision at twenty runs **0.0658 for
the model against 0.0289 for the size sort and a base rate of 0.0378**. The
screen finds a target about one time in fifteen, against one in twenty-six by
chance and one in thirty-five by sorting on size.

It is a screen and not a probability, and the calibration table says so in the
output:

```
 low  high    n  predicted  realised       gap  thin
 0.0   0.1 5437   0.032609  0.035130  0.002521 False
 0.1   0.2  330   0.132128  0.069697 -0.062431 False
 0.2   0.3   60   0.238713  0.100000 -0.138713 False
 0.3   0.4   25   0.344703  0.040000 -0.304703 False
 0.4   0.5   11   0.456424  0.090909 -0.365515 False
```

Wherever the model speaks loudly it is badly overconfident. No accuracy figure
appears anywhere in the module, because a model predicting "never" is right 96.4
percent of the time.

The feature a reader will look for first is deliberately absent, and the reason
is a trap worth naming. No public price source serves history for a delisted
symbol, so every price-derived feature is missing for exactly the companies that
were acquired and present for the companies that were not. An indicator that is
simply "this company has since left the filing record" scores an AUC of 0.82 on
its own, against 0.57 for the real model, and
`test_the_absence_of_a_price_would_separate_the_classes_perfectly` measures it.
A panel carrying market capitalisation would have produced a spectacular score
and learned that companies without a share price get bought. The cost is real
and is not hidden: **the valuation channel is untested here, and this model
cannot say whether cheap companies get bought.**

### How the harness is built

`techval.ml.evaluation` is the single scoring path every model above runs
through, and its rules are what the scoreboard means.

Every fold splits by date, never at random. A random split lets a model train on
2026 and test on 2024. That is the cleanest way to manufacture a result nobody
can repeat.

The embargo is counted, not merely applied. Where the label is a forward return
or a forward growth rate, an observation dated t is not known until t plus the
horizon, so a training row a month before the test window shares eleven twelfths
of its label window with the rows being predicted. Training is cut strictly
before the test start less the embargo, and the rows that fall in the gap are
counted on the fold, so an embargo that throws away a third of the sample is
visible as a decision. A zero embargo is written into the notes so a reader can
see the protection was declined rather than forgotten. The central test builds a
panel whose label is a forward twelve month average and scores the same
prediction rule twice on identical test windows: against the window it beats a
constant comfortably, and with 365 days between them it loses to the training
mean, with nothing about the model changed.

No result without a baseline. Where the caller supplies none the module computes
the obvious one and names it: the training-period mean for a regression, the
base rate for a classifier whose AUC is then 0.5 by construction, and the
closed-form expected NDCG of a uniform random ordering for a ranking. A computed
baseline that has to peek at the answer key is named in sample, which makes it
the harder comparison and therefore the conservative choice.

Fold dispersion travels with every score. A lift of 0.02 against a fold standard
deviation of 0.09 reads as what it is, and where the folds are real folds
`verdict()` says "inside the fold-to-fold noise" in those words. A ranking
result carries one score per query instead, so its verdict names the spread as
per-query, declines that judgment, and cites the paired per-query difference,
the only spread that is an error bar on the lift. Honest error bars, not
significance tests, and the module does not pretend otherwise.

Ranking well and being calibrated are different claims. `calibration_table`
buckets on equal-width bins over [0, 1], not by quantile, because the question
is absolute: when this model says 70 percent, how often does it happen. Quantile
buckets would hide the fact that a model never says 70 percent at all, so empty
buckets are kept with a count of zero.

The training primitives underneath are written out by hand in numpy with
analytic gradients and no framework. A gradient a reader can check is worth more
beside a valuation than one they cannot. `gradient_check` differences every
layer and every loss against central finite differences in the suite at a bar of
1e-6 with measured errors around 1e-9, and two tests break a backward pass on
purpose to show the check would catch it.

---

## The TMT data layer

None of this is available from `companyfacts` alone, and none of it is in a
generic DCF tool. Every block below is real output from the code.

### The universe and its sub-verticals

The SEC assigns every filer one SIC code and publishes it in the submissions
payload. It is the only industry label in this pipeline that comes from a filing
rather than from a vendor with a licence fee, and it is not sufficient. The map
is built against 105 real submissions payloads retrieved on 2026-09-11 and
committed as a fixture. Every code is either annotated with the filers observed
under it or declared in `SIC_UNVERIFIED` as resting on the code definition
alone, and a test enforces the partition.

```python
from techval.tmt.taxonomy import tmt_universe

universe = tmt_universe(client, assumptions)
```

```
110 candidates, 106 classified
AMZN  SIC 5961  internet                0.55
      curated prior internet; unclassified: SIC 5961 (retail catalogue and mail order, the code Amazon has filed under since 1997) maps to no TMT sub-vertical and no business text was read [Retail-Catalog & Mail-Order Houses]
PANW  SIC 3577  infrastructure_software 0.55
      curated prior infrastructure_software overrides SIC 3577 alone; no business text was read [Computer Peripheral Equipment, NEC]
QCOM  SIC 3663  semiconductors          0.55
      curated prior semiconductors overrides SIC 3663 alone; no business text was read [Radio & Tv Broadcasting & Communications Equipment]
AMT   SIC 6798  towers_fiber            0.55
      curated prior towers_fiber; unclassified: SIC 6798 (REIT election, shared by American Tower, Crown Castle, SBA, Equinix and Digital Realty with every shopping mall and apartment landlord) maps to no TMT sub-vertical and no business text was read [Real Estate Investment Trusts]
V     SIC 7389  payments                0.55
      curated prior payments; unclassified: SIC 7389 (services not elsewhere classified, the widest code in the register: Visa, Mastercard, PayPal, FIS, Global Payments, Accenture, Uber, eBay and DoorDash all file under it) maps to no TMT sub-vertical and no business text was read [Services-Business Services, NEC]
EA    unresolved: could not source 'CIK' for EA
FI    unresolved: could not source 'CIK' for FI
FYBR  unresolved: could not source 'CIK' for FYBR
IPG   unresolved: could not source 'CIK' for IPG
```

The source string is as much the deliverable as the label. A peer set built on
an override nobody can see is a peer set nobody can defend. Classification runs
the code first, scores keyword evidence over the business description second,
and reconciles the two. Evidence overturns a mapped code only on a margin over
the runner-up, the margin is lower for the codes demonstrably holding several
industries at once, and the answer that lost is named. An exact tie is reported
as a tie with every tied candidate named, never broken by dictionary order. The
confidence is not a probability and does not come out of a fitted model: it is
the rung of the ladder the classification landed on, which is why the source
travels with it and the number is only a sort key.

103 of the 110 rows end in `no business text was read`, and that is the default.
`tmt_universe` does not fetch a hundred 10-Ks for a screen that mostly does not
need them, so most names in a default run are classified on the code plus a
curated prior and say so on the record. Pass a `business_text` reader and the
keyword evidence enters, the two sources are reconciled, and the string names
which of them won. A curated prior beating a classification made from strong
evidence is a human override and is recorded as one every time.

Two findings from building it against the register. **There is no SIC code for
payments.** Visa, Mastercard, PayPal, FIS and Global Payments all file under
7389 alongside Accenture, Uber, eBay and DoorDash, so mapping either code to
payments would be right about one filer and wrong about four. The map has no
payments entry at all. And first filing is the wrong admission date, because a
CIK exists from the first piece of paper filed under it, and for a
venture-backed company that is a Form D notice years before the listing.
Roblox's CIK carries submissions from 2005 against a 2021 IPO, so admission keys
on the first periodic report.

The four unresolved names are left in with the reason recorded. A candidate pool
that quietly loses names is a candidate pool nobody can check. All four left the
tape to an acquisition, the same survivorship hole the signal harness flags,
seen from the other end. Juniper used to be a fifth. It resolves now because
`former_tickers.FORMER_TICKERS` carries 100 TMT registrants that left between
2019 and 2026, Juniper among them at CIK 1043604 through 2025-05-09, and the
resolver falls back to that index when the live ticker file has forgotten a
symbol.

### Segment and geography economics

Segment detail is tagged under `StatementBusinessSegmentsAxis` and is invisible
to the `companyfacts` endpoint, which publishes undimensioned facts only.
`techval segments` reads it out of the instance document.

```
$ techval segments DIS --config assumptions.yaml

───────────── DIS  segment economics, USD millions ─────────────
Year ended 2025-09-27, from 0001744489-25-000155

Segment                     Revenue  % of total  Operating income  Margin    D&A  Capex
Entertainment                42,466       45.0%             4,674   11.0%    825  1,155
Experiences                  36,156       38.3%             9,995   27.6%  2,823    n/a
Sports                       17,672       18.7%             2,882   16.3%     48      3
Unallocated / eliminations  (1,869)
Consolidated revenue         94,425

──────────────────────── Reconciliation ────────────────────────
Sum of reported segments   96,294
Consolidated revenue       94,425
Residual                  (1,869)  -1.98% of consolidated revenue
Within 2% tolerance         foots

  Revenue concentration (Herfindahl)   0.369
  a genuine conglomerate: a sum of the parts values it better than any
  single multiple, because the blended multiple the market applies is a
  weighted average nobody chose.

───────────────────────── By geography ─────────────────────────
Geography                        Revenue  % of total  Long-lived assets
Americas                          76,430       80.9%             61,888
Europe                            11,090       11.7%             13,227
Asia Pacific                       6,905        7.3%             10,799
Consolidated revenue              94,425

How the table was read
  - Eliminations And Other is a reconciling item rather than a business,
    tagged at -1,869, and is left out of the table. The residual above is
    what accounts for it.
```

Experiences earns a 27.6 percent operating margin against 11.0 percent in
Entertainment, and the 18.6 percent company average describes neither. A single
EV/EBITDA struck on it prices a theme park like a streaming service.

The reconciliation is the control. Segment revenue must sum to consolidated
revenue over the same period from the same concept, and where it does not the
gap is a named residual instead of something spread across the segments. The
stronger control is whether the residual is *explained*: it is matched against
the reconciling items the filer actually tags, on magnitude alone, because
filers write intersegment revenue both ways. A gap that matches nothing
disclosed gets different words from a gap that is the eliminations working.

The double-counting trap is the reason the geography cut is readable at all.
Filers tag the same dollar several times over on different slices, so Disney
tags Entertainment revenue once whole, again split across three geographies, and
again third-party against intersegment on the product axis. Any fact carrying
more than one slicing axis is a cell of a matrix. It is not a total and it joins
neither table.

### Operating KPIs, and where each one came from

Not one of ARR, customer count, net revenue retention, subscribers, ARPU or
churn is a us-gaap concept. What filings carry instead is near misses:
`NumberOfCustomerAccountsImpacted`, `NumberofCustomerClasses`,
`NumberOfSegmentManagers`. A substring search for "customer" takes every one, so
the concept ladders are anchored full-match patterns and a test asserts the
traps really are in the fixture files before asserting they stay out of the
results.

```
$ techval kpis DDOG --config assumptions.yaml

────── Disclosed operating metrics, and where each came from ───────
Metric                       Value  Evidence       Conf  Tag or phrase
rpo                        3,471.4  XBRL, us-gaap  1.00  RevenueRemainingPerformanceObligation
revenue_prior              3,016.1  XBRL, us-gaap  1.00  RevenueFromContractWithCustomerExcludingAssessedTax
sales_and_marketing        1,094.4  XBRL, us-gaap  1.00  SellingAndMarketingExpense
billings                   4,308.0  derived        0.80  revenue + change in ContractWithCustomerLiabilityCurrent
                                                         + ContractWithCustomerLiabilityNoncurrent
arr                        refused  prose          0.00  ARR of $100,000
customers                  refused  prose          0.00  approximately 32,700 customers; approximately 4,310
                                                         customers; approximately 603 customers
net_revenue_retention      120.00%  prose, hedged  0.40  dollar-based net retention rate was about 120%

Extraction flags
  - net_revenue_retention: low confidence, hedged: the filing says 'about',
    so the figure is the company's own estimate rather than a measurement
  - arr: not usable, the phrase gives $100,000 with no scale word, and below
    $1,000,000 that is as likely to mean dollars as millions
  - customers: not usable, the text states 3 different values for this metric
    (3.27e+04, 4,310, 603), which are usually different periods the prose
    distinguishes and this parser does not

────────────────── Looked for and not disclosed ──────────────────
customers_over_100k, gross_revenue_retention, rpo_current: no element in
the filing's instance document matches this concept and no phrasing matched
in the text supplied, so this filer does not disclose it.
```

Two of those rows are refusals, which is the module working. A tagged figure
comes back at confidence 1.0 with the tag and the accession behind it, under the
source label `xbrl_extension`, which names the extraction path and not the
taxonomy: `RevenueRemainingPerformanceObligation` is a standard us-gaap concept
and the row's own note says so. A figure derived by arithmetic comes back at 0.8
with the arithmetic written out. A figure read out of prose comes back with the
sentence it came from and a confidence that falls for a hedge. "About 120%" is
the company's own estimate and drops to 0.4. A dollar figure with no scale word
below the floor is reported at 0.0 with the reason attached, because $100,000 is
as likely to mean dollars as millions. A filing that stated a metric the parser
could not read is a different finding from a filing that did not state it, and
absence is a finding too: a software company that discloses no customer count
and a carrier that discloses no churn are both different from a run that failed.

Nothing is silently corrected. A net revenue retention of 11500% would keep its
value, lose its confidence and say the percent sign was probably misread,
because dividing by a hundred would make three different errors look like one
clean number.

Billings is the derivation, and the convention is argued: `billings = revenue +
change in total deferred revenue`, including the non-current part, because a
multi-year prepayment is invoiced value whichever side of twelve months it is
delivered on. For Datadog the current-only convention runs 21.0mm lower and a
test pins the gap.

The extractor is handed Item 7, not the whole filing. Running these patterns
over a whole 10-K reads risk factors and forward-looking statements as though
they were results. Locating the boundaries is `nlp/sections.py`'s job:

```
0001628280-26-008819  405,446 characters, 89.9% assigned to an Item
  Item 1   Business                   5,397 words
  Item 1A  Risk Factors              22,289 words
```

Every 10-K names each Item twice, once in the contents table and once at the
section, so a first-match split returns a list of section names and page numbers
as the entire business description and nothing downstream notices, because a
list of section names is still text. Taking the last occurrence fails on the
same filing: Datadog's final "Item 1. Business" sits inside the risk factors, in
the sentence "described under Part I - Item 1. Business in this Annual Report".
Three rules decide it instead, each measured on six real 10-Ks. A contents entry
ends in a page number and a section ends in a sentence. A real header is
preceded by a full stop or a page number while a cross-reference is preceded by
a lowercase word or a comma. And Items run in the sequence Regulation S-K
prescribes, so a reference pointing forward cannot claim a section the scan has
not reached.

### The metrics each sub-vertical is priced on

The Rule of 40 is one rule and three different numbers, and which margin goes in
is the whole argument, so all three are computed:

```
Revenue growth                     31.5%
FCF margin                         29.8%
EBITDA margin                       2.0%
Operating margin                    0.4%
SBC / revenue                      20.7%
Rule of 40 (FCF variant)           61.3
Rule of 40 (EBITDA variant)        33.5
Rule of 40 (operating variant)     31.9
Rule of 40 variant spread          29.4

The Rule of 40 variants straddle 40: this company passes at 61.3 on the FCF
variant and fails at 31.9 on the operating variant, off one set of filings.
Which number gets quoted is a choice, so state it.
```

One set of filings, a 29.4 point spread, and the company passes on one variant
and fails on two. Stock compensation runs 20.7 percent of revenue and is most of
the distance between the cash and earnings variants, because it is added back
inside cash from operations and charged inside EBITDA. Every metric travels with
the prose definition of the variant that produced it, and the pack refuses to be
constructed if a metric arrives without one, so a printed table cannot carry a
number whose definition was left behind.

An unknown sub-vertical raises rather than falling back to the software pack. A
semiconductor company scored on the Rule of 40 is a worse answer than no answer.

### Sum of the parts

The multiple is an argument, not a number. Each segment is valued on a `(metric,
multiple, source)` triple supplied by the caller, the source sentence travels
all the way to the printed row, and a multiple with a blank source is refused.
There is no default: a peer set for cable has no business living in a config
file shared with a software DCF. `--emit-plan` writes a skeleton carrying the
filer's own segment names and stops.

The first thing the module does is refuse. The segment note is a fiscal year and
`build_financials` returns a trailing twelve months, and pairing the two is the
mistake nobody sees:

```
FLAG: the segments cover the year ended 2025-09-27 and the consolidated figures
cover the twelve months to 2026-06-27, because DIS has filed at least one
quarter since its annual report. Segment detail is annual, so the two cannot be
brought together by reading more filings. Either accept that the parts are
measured a period behind the whole, or pin the knowledge date so that the annual
report is the newest filing on record. That date is 2025-11-14, the day after
the 2025-09-27 annual report reached EDGAR on 2025-11-13: run with --as-of
2025-11-14, which also prices the company at that date's close. Do NOT pin to
2025-09-27. The annual report did not exist on the last day of the year it
covers, so that date reads the PRIOR year's segment note and the two windows are
a period apart again.

MissingDataError
could not source 'segment revenue reconciling to consolidated revenue' for DIS
  hint: the 3 segments supplied add to 96,294mm against consolidated revenue of
  98,861mm, which is 97.4% of the company. A sum of the parts over segments that
  do not cover the company values the missing part at zero. Add the missing
  segment, or the eliminations line if the segments overlap.
```

Pinned to the date it names, with three multiples the caller supplied and
labelled as illustrative, it runs:

```
$ techval sotp DIS --config assumptions.yaml --plan dis.yaml --as-of 2025-11-14

Entertainment at 10.0x EV/EBIT        46,740
Experiences at 14.0x EV/EBIT         139,930
Sports at 8.0x EV/EBIT                23,056
Gross value of the parts             209,726
Capitalised corporate cost          (12,449)
Conglomerate discount                      0
Enterprise value, sum of the parts   197,277
Equity value                         156,041
Value per share                       $86.16
Enterprise value, market             232,840
Gap, dollars                        (35,563)
Gap, % of market EV                   -15.3%
Share price                          $105.80
Gap per share                       $(19.64)

Cross-checks
  - FLAG: Largest segment 'Experiences' is 66.7% of the value of the parts at
    14.0x EV/EBIT and 37.5% of revenue, a wedge of +29.2%. The mix is the
    valuation: the consolidated multiple this company trades on is the wrong
    multiple for most of its revenue.
  - EV/EBIT multiples supplied span 8.0x to 14.0x, 1.8 times.
  - Capitalised corporate cost is 12,449mm, 5.9% of the gross value of the parts.
  - Sum of the parts is -15.3% against the market enterprise value, -35,563mm.
```

The multiples above are the caller's and carry that in their source string on
every row, so nothing here is the engine's opinion of what Disney's parts are
worth. What the engine contributes is the arithmetic and the checks.

Corporate cost is where most sums of the parts go wrong. Segment operating income
under ASC 280 is struck before unallocated corporate expense, so adding the
segments up and stopping values the holding company's own overhead at zero. It is
capitalised as a negative and subtracted, and absent a multiple from the caller
it is capitalised at the enterprise-value-weighted average of the segment
*earnings* multiples, because the overhead is an earnings charge on the businesses
it supports and averaging an EV/Revenue against an EV/EBITDA would average
different denominators. Where no corporate cost is supplied at all the total still
prints and the flags say it is overstated.

The conglomerate discount defaults to zero and both sides of the argument appear
in the notes on every run. The diversification literature, Lang and Stulz
through Berger and Ofek, finds something like 5 to 15 percent. Against that, a
discount assumed rather than observed is a fudge factor that can be tuned until
the sum of the parts agrees with whatever answer was wanted.

Negative EBITDA times a positive multiple claims the business is a liability,
which is almost never true of something that can be closed or sold. The default
refuses with `NotMeaningfulError` and the caller either values the segment on
revenue or passes `on_negative="nil"`, which is recorded on the row as an
assumption.

### Precedent transactions

A precedent price contains a control premium and whatever synergies the buyer
underwrote. A trading multiple contains neither. That sentence is in the module
docstring and in the notes of every result, along with the consequence: a
precedent multiple sits above a trading multiple for the same company by
construction, so reading the gap as evidence the company is cheap today is
reading the control premium twice.

```
$ techval precedents RAMP,PAYO,IRDM,SLAB --config assumptions.yaml

Announced   Target  Acquirer          Cons.  % cash    Offer  Unaff. 1d  Prem. 1d  Prem. 30d  Gap, pts  EV/Rev
2026-06-29  IRDM    Rocket Lab        mixed     n/a      n/a      43.52       n/a        n/a       n/a      NM
2026-06-15  PAYO    Neon Maple Parent  cash    100%     7.40       6.75      9.6%      37.2%     -27.6    2.9x
2026-05-18  RAMP    MMS USA Holdings   cash    100%    38.50      29.66     29.8%      30.6%      -0.8    2.6x
2026-02-04  SLAB    Texas Instruments  cash    100%   231.00     136.62     69.1%      58.8%     +10.3      NM

IRDM  Iridium Communications Inc.  <- Rocket Lab Corporation  announced 2026-06-29
  No offer price is reported and none is substituted.
  The cash leg alone is 27.00 per share. That is the CASH LEG and not the offer:
  quoting it as the price would understate the consideration by the whole of the
  stock leg and strike every premium and multiple against a number nobody agreed.
  - the agreement states 0.2400, 0.4000 as alternative exchange ratios, which is
    a collar: the ratio is fixed from the buyer's share price at closing and does
    not exist at announcement. No exchange ratio is recorded.

Target   Offer  Unaff. 1d  Prem. 1d  Unaff. 30d  Prem. 30d  Gap, pts  Reading
PAYO      7.40       6.75      9.6%        5.39      37.2%     -27.6  ran INTO the announcement
SLAB    231.00     136.62     69.1%      145.47      58.8%     +10.3  FELL into the announcement

Flags
  - RAMP: other parties named in the agreement: Publicis Groupe S.A.. A buyer
    that is a holding company formed for the deal is what the agreement names;
    the sponsor or ultimate parent is in this list.
  - SLAB: kept without being checked against the 250mm floor in
    ml.mna.min_deal_size, because no equity purchase price could be built for it.
  - 4 of 4 statistic column(s) refused: fewer than 5 of the 4 transactions
    carried a value for them.
```

Four deals, and the engine prices two of them completely. Iridium pays cash plus
a collared number of Rocket Lab shares, so no exchange ratio exists at
announcement; the transaction is kept, the cash leg is kept, and the offer price
is `None` with the reason in the notes. Reporting the cash leg as the offer
price would be the worst kind of wrong number, a real figure in the wrong role.
Silicon Labs has an offer price and a premium but no equity purchase price, so
it carries a premium and no multiple, and the size floor excludes it. An unknown
size cannot be shown to clear one.

Payoneer is the case for measuring the unaffected price two ways. The headline
is the last close strictly before the announcement, because any reader can
reconstruct it from one quote. The mean of the thirty calendar days before is
computed beside it. Nine and a half percent against thirty-seven is a gap of
27.6 points, and only one of those is a control premium. Silicon Labs runs the
other way, at 69.1 against 58.8: a stock that fell into the announcement, and a
one-day premium that flatters the deal.

The announcement date is neither the signing date nor the filing date. An 8-K
reports the event date and can be filed four business days later, and a signing
after the close is announced the next morning, so the date used is the earliest
filing across the announcement cluster, floored at the agreement date the
document states in words.

Four traps are each a real document in the fixtures. Item 1.01 is not a merger:
Splunk's June 2021 8-K is a convertible note sale and contains "an initial
conversion price of $160.00 per share". The filer is not always the target:
Zendesk's own S-4 says each share converts into 0.225 of a Zendesk share,
because Zendesk was buying Momentive. Par value is not an offer price, and every
merger 8-K in the set says "par value $0.001 per share" a few words from the
real consideration. And a price that cannot be fixed is not fixed.

---

## The judgment calls

These decisions move the answer more than anything else in the code.
`docs/methodology.md` argues each one at length; this is the summary.

### Stock-based compensation is expensed

GAAP EBIT is already net of SBC, so the default adds nothing back and free cash
flow is struck after the full cost of paying employees. The opposing camp adds
it back as a non-cash charge. That is only honest alongside a growing share
count, so `sbc_dilution` is on by default: every projected year issues `SBC
dollars / share price` new shares and the count carried forward grows by them.
Datadog reads $42.16 expensed, $85.41 added back with no dilution, and $78.88
added back with the dilution modelled. Modelling the shares closes about 15% of
the gap at this discount rate, and 13% at the 10.5% WACC the methodology worked
through. The share that closes moves with the rate, but never far, and the two
camps are still a factor of two apart.

That failure to converge is structural, not a calibration problem. Roughly 70%
of the addback's uplift in enterprise value sits in the terminal value, which
capitalises the addback in perpetuity, while the modelled issuance stops at year
five. Five years of issuance takes the count from 366.9mm to 397.3mm, 8.3% of
the register, and 8.3% cannot pay for a perpetuity. The checks say so directly:
the terminal year issues 1.87% of the share count in stock compensation, so
beside 2.50% terminal growth cash flow per share compounds at 0.62%, not 2.50%.
That is the number the addback camp has to defend, and a model reporting $78.88
without it is reporting the generous reading of its own assumption.

### Diluted WASO by default, a treasury stock count on request

WASO is the right denominator for reported EPS and the wrong one for a valuation
struck today. It is an average over a past window, and the awards it carries are
the ones outstanding across that window. Not the ones outstanding now. Set
`dilution.method: treasury_stock` and the count is built from shares actually
outstanding in the latest filing plus the net new shares in-the-money awards
would create at today's price, under ASC 260. For Datadog that is 359.08mm
outstanding plus 1.55mm net from options plus 17.10mm unvested RSUs, so 377.73mm
against a 366.93mm diluted weighted average. The equity value was 2.9% light.
Options are valued by exercise-price band where the filer tags bands, because a
single weighted-average strike credits the company with exercise proceeds from
options nobody would exercise: on the worked case in the tests, banding produces
3.6mm net new shares and the single average produces 0.6mm, a six-fold
understatement. WASO stays the default because it is available for every filer
and needs no instance document.

### Three terminal values on every run

Gordon capitalises the final projected flow and leaves the implied return on
capital to be checked afterwards. The exit multiple applies the peer median. The
value driver inverts the Gordon assumption: `TV = NOPAT_{N+1} * (1 - g/ROIC) /
(WACC - g)`, so reinvestment is derived from the return you will defend, and the
terminal value cannot embed a return nobody signed up for. With `terminal_roic`
null it falls back to the WACC, the competitive-equilibrium view, and at ROIC
equal to WACC it collapses to `NOPAT / WACC` for any g at all. Datadog reads
$37.93 there against $42.16 on Gordon. `terminal.method` picks which one drives
the headline; all three are always reported.

### Operating leases stay out of debt, and EBITDA is the pairing

Under ASC 842 the lease charge stays inside operating income as straight-line
rent. It is not split into depreciation and interest the way IFRS 16 requires,
so US-GAAP EBITDA is already *after* rent. Adding the liability to debt while
dividing by that EBITDA charges the lease twice. Set
`capitalize_operating_leases: true` and every multiple switches to EBITDAR
automatically; the engine will not let the two conventions mix. Finance leases
are debt either way.

### In-the-money convertibles are equity, not debt

ASU 2020-06 made if-converted mandatory, so a dilutive convertible's shares are
already inside diluted weighted-average shares. Counting the principal as debt
on top of that double-counts the instrument, by $986mm in Datadog's case.

### Beta is the median of peer asset betas, relevered

Each peer's levered beta is unlevered at its own D/E and tax rate, the median of
those is taken, and the result is relevered at the target's capital structure,
so what gets pooled is business risk and not financing policy. Regressions are
joined on the dates the two series actually share, never zipped by position, and
R-squared and the standard error are printed, because a beta of 1.42 on an
R-squared of 0.11 is not something the market told you with confidence.

`peer_beta_method: vasicek` shrinks each asset beta toward the cross-sectional
mean by a weight that is the ratio of the dispersion across peers to that
dispersion plus the peer's own sampling variance, so a noisy regression is
pulled hard and a tight one is barely moved. On the six committed peers it pools
to an asset beta of 1.436 against 1.387 for the median, worth 25 basis points on
the cost of equity. The reason is MongoDB: the widest standard error in the set
at 0.393, so it keeps only 0.367 of its own estimate and comes back from 1.964
to 1.644. The median could not see that, because a median only reads the middle
of the sorted list. Note the direction too. Shrinkage pulled the highest beta
down hard and the pooled figure still came out above the median, because the
mean of this set, 1.458, sits above its middle. The median stays the default
because one broken regression, a peer in a takeover or three weeks of a short
squeeze, can move a mean and cannot move a median.

### Mid-year discounting, with one asymmetry

Explicit flows discount at `(1+w)^-(t-0.5)`. The Gordon terminal value carries
the same half-year uplift, because its perpetuity flows also arrive mid-year.
The exit-multiple terminal value does **not**: a multiple is a price struck at a
date, so it discounts over whole periods. Applying the mid-year factor to both
overstates the exit-multiple case by roughly half a year of WACC.

### The Monte Carlo draws jointly, and correlation widens the answer

The usual claim is that correlating the drivers narrows a DCF distribution, on
the reasoning that independence generates combinations that do not occur. It
generates them, but look at which ones. Value rises with growth, rises with the
terminal margin and falls with the WACC, so a positive growth-margin correlation
and negative correlations of both against the WACC all point the same way: the
good draws arrive together and so do the bad ones. Independence puts mass on the
offsetting middle, where the point estimate already sits. The shipped matrix
widens the standard deviation by 21 percent against independent draws on all
four companies committed as fixtures, at 21.7% for Datadog, 21.2% for
CrowdStrike, 20.9% for MongoDB and 21.7% for Zscaler. Narrowing would require
believing growth carries its own discount rate with it. That is a defensible
matrix and it is one line to enter, but it is not the one this module argues.

### Purchase accounting: the deferred tax liability makes goodwill bigger

With `merger.purchase_accounting.enabled`, the excess of the purchase price over
book equity acquired goes first to identifiable intangibles at the configured
share, and the rest is goodwill. A stock deal is tax free to the seller and
carries over the seller's tax basis, so the write-up exists for book and not for
tax, and that temporary difference is a liability assumed at close:

    DTL      = intangible write-up * tax rate
    goodwill = excess - intangibles + DTL

The plus sign is the part people get backwards. Goodwill is consideration less
the fair value of net assets acquired, and the DTL sits inside those net assets,
so booking it takes net assets down and pushes goodwill up. Screening Datadog
buying MongoDB at a 30% premium, the excess over book equity is $37,136mm, the
write-up is $14,855mm, the DTL on it is $3,565mm at 24%, and goodwill comes out
at $25,847mm against the $22,281mm a model that skips the DTL would report. The
liability everybody forgets is a seventh of the goodwill it creates, and without
it the opening balance sheet does not balance. The amortisation of those
intangibles is a book expense with no tax basis behind it, so free cash flow
gets back only `(1 - t)` of the charge, and acquired deferred revenue is written
down to fair value, which permanently destroys revenue the target would have
recognised.

### Software is two sub-verticals and telecom is two, and both are arguable

Infrastructure software is consumption priced and its gross margin is
constrained by the cloud bill underneath it; application software is seat priced
and grows by landing and expanding. They do not trade at the same EV/Revenue for
the same growth rate, and pooling them is the commonest way a software comp set
goes quietly wrong. Towers are separated from telecom for the mirror-image
reason. A tower company is a landlord with escalating multi-decade leases and
three tenants; a carrier is an operating business with churn, spectrum and capex
per home passed. One is valued on AFFO and lease-up, the other on EBITDA and
subscriber economics. The case against both splits is that they halve an already
thin comp set, and the engine's own eight-observation floor then refuses more
often. Eleven buckets is the answer this package defends and the count is
visible everywhere it matters.

### The training labels come from filings, never from me

Every learned peer model needs ground truth, and the weakest thing to build one
on is the author's own opinion of which companies are comparable. The labels
here are the compensation peer groups filers disclose in their own proxies,
chosen by an independent consultant under stated revenue and market
capitalisation bands, and a pair means the filer told the SEC its board used
that company when setting pay for that year. The bias is real and is carried
rather than argued away: a compensation committee picks partly for competition
for executive talent, which is why Apple's and Comcast's tables contain Johnson
and Johnson, Merck, Procter and Gamble and Honeywell. `selection_criteria` is
captured with every group, because a reader who knows the bands knows what the
label is worth.

### Missing is not zero, and not meaningful is not missing

The feature store holds `None` rather than imputing, because filling a missing
gross margin with zero tells the model the company broke even. That is a
specific claim and usually a false one. Separately, a ratio whose denominator
has the wrong sign comes back `None` with the reason: net debt over a negative
EBITDA is a negative multiple that sorts to the conservative end of the column
and means the opposite. Winsorization is the answer to a denominator that is
small, never to one whose sign has flipped.

---

## Beyond a single valuation

Every one of these is opt-in. A base case should be one set of assumptions you
can defend; these are the arguments about it.

### Monte Carlo

`simulation.enabled` draws year-one revenue growth, the terminal EBIT margin,
the discount rate and terminal growth from one multivariate normal, revalues on
every draw, and reports percentiles of value per share and of enterprise value.
Ten thousand draws on Datadog put the 5th percentile at $30.41, the median at
$42.08 and the 95th at $59.30 against a $221.72 market price, so P(value >
price) is 0.0% and the 95th percentile does not reach the quote. The
distribution has a standard deviation of $8.92 on a median of $42.08, a
coefficient of variation of 21 percent. A DCF carries that much dispersion
before any argument about whether the central case is right. Draws where
terminal growth lands at or above the discount rate are rejected outright rather
than clipped to the boundary, since clipping would pile probability mass exactly
where the perpetuity is largest. The tornado is reported alongside and is
deliberately univariate: it says which assumption to argue about, and the
distribution says how wide the answer is once they all move together. Here it is
led by the terminal margin at $17.59 of swing across its 90% band, against $4.45
for terminal growth at the foot of it.

### Adjusted present value

`apv.enabled` values the business unlevered at the cost of equity its asset beta
implies, adds the tax shield as a stream in its own right, and reconciles to the
WACC answer line by line. The discount rate on the shield is the whole argument:
the cost of debt is Modigliani-Miller and assumes a debt schedule fixed in
dollars, while the unlevered cost of equity is Harris-Pringle and assumes debt
rebalanced to a constant share of value, the thing a constant WACC already
implies. Datadog is net cash, so the shield is zero, Ku equals the levered cost
of equity, and APV comes back at $10,484mm against a WACC enterprise value of
$10,484mm, a difference of +0.0%. That is the identity case, and it is the one
worth checking first.

### Point-in-time valuation

`--as-of` on every command pushes a knowledge date through the EDGAR client and
the price series, so facts filed later are discarded and prices stop there. It
changes the company. Datadog valued as of 2026-03-01 shows the twelve months to
2025-12-31 and a $44.4mm operating loss; with full knowledge the same company
shows the twelve months to 2026-06-30 and a $16.3mm operating profit. Those are
different businesses to a model, and the second one did not exist on 2026-03-01.

### Backtesting

`techval backtest` values a list of names on past dates and scores the upside
against realised forward returns. Forward prices live in an object the valuation
path is never handed, so lookahead is structurally impossible rather than a
matter of discipline, and the provenance of every `Financials` that comes out is
walked afterwards to confirm no line item was sourced from a filing dated after
the valuation date. The statistic is Spearman rather than Pearson, because the
relationship between a DCF upside and a subsequent return is monotonic at best
and a handful of names at 300% upside would carry a Pearson coefficient on their
own. The harness is loud about survivorship bias, overlapping windows and thin
samples, and none of those is fixed by it. `techval signal` is the same argument
taken further: it applies the Newey-West correction the backtest only warns
about, and it is what produced the negative result above.

### Warranted multiples, two of them

`comps.regression.enabled` fits EV/Revenue across the peer set on growth and
margin by OLS and reads off the multiple the target's own fundamentals warrant,
with the residual against the traded multiple as the thing to argue about. It
refuses below eight usable observations, and the six-name peer set shipped in
`assumptions.example.yaml` produces five, so the engine declines rather than
fitting three parameters to five points. Measured across 1,764 sub-vertical
cross-sections in the model panel, it refuses on 788 of them, a 45 percent
refusal rate. The engine is behaving exactly as the methodology documents: a
real comp set is six to ten names against a floor of eight. `techval screen` is
the non-linear version of the same question fitted across the whole TMT
cross-section, and it beats that OLS on ordering while adding almost nothing on
the change.

### A fitted growth path

`ml.forecast.enabled` replaces the straight-line fade in
the DCF with the one measured on filings, described under
[the fade curve](#the-revenue-growth-fade-curve) above. `growth_path_for_dcf`
returns `None` when the switch is off, so the base report is byte for byte what it
was, and `compare_fade` raises rather than quietly valuing on a path the
configuration says not to use.

---

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ruhan-sahasi/techval && cd techval
uv venv && uv pip install -e ".[dev]"
export TECHVAL_SEC_EMAIL=you@example.com   # SEC fair access requires a contact
cp assumptions.example.yaml assumptions.yaml
```

The SEC blocks automated clients that do not declare a contact address, so the
engine refuses to start without `TECHVAL_SEC_EMAIL`.

## Usage

Fourteen commands in five groups.

```bash
# Core valuation
uv run techval value DDOG --config assumptions.yaml     # statements, EV bridge, WACC, DCF, comps, football field
uv run techval comps DDOG --config assumptions.yaml     # comp table only
uv run techval merger CRWD MDB --config assumptions.yaml
uv run techval fetch DDOG                               # warm cache, show provenance
uv run techval backtest DDOG,MDB,ZS,CRWD --config backtest.yaml \
    --from 2025-03-31 --to 2025-09-30 --horizon 252

# TMT fundamentals
uv run techval kpis DDOG --config assumptions.yaml      # operating metrics, with the provenance of each
uv run techval segments DIS --config assumptions.yaml   # revenue and margin by segment and geography
uv run techval sotp DIS --config assumptions.yaml --emit-plan dis.yaml
uv run techval sotp DIS --config assumptions.yaml --plan dis.yaml --as-of 2025-11-14

# Learned comparables
uv run techval peers DDOG --config assumptions.yaml     # encoder ranking beside the hand-written set
uv run techval screen --config assumptions.yaml \
    --panel tests/fixtures/warranted/observations.json.gz   # traded against warranted multiple

# Forecasts and signal testing
uv run techval fade DDOG --config assumptions.yaml      # fit the fade curve and value on it
uv run techval signal --config assumptions.yaml         # test a score against forward returns

# M&A
uv run techval precedents RAMP,PAYO,IRDM,SLAB --config assumptions.yaml
uv run techval targets --dataset tests/fixtures/mna --config assumptions.yaml
```

`--as-of` is accepted by every command that reads a company. It discards every
fact filed after that date and stops the price series there, so the run uses
only information that existed at the time. `--no-cache` bypasses the HTTP cache.
`screen` and `targets` read a recorded panel, and the committed fixtures under
`tests/fixtures` are the shape they expect. Everything else is in
`assumptions.yaml`: no rate, growth path, margin or multiple is hardcoded
anywhere in the engine.

`backtest` takes valuation dates at quarter ends between `--from` and `--to`, or
an explicit list through `--dates`, and scores each against the close
`--horizon` calendar days later. It needs a risk-free rate it can date, so pin
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
trailing twelve months of diluted shares. A failure is recorded with its reason.
Nothing is dropped, so the count of what was not valued stays visible.

### The Python API

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

The `client` and `assumptions` the TMT calls take are built the same way the CLI
builds them, and a knowledge date passed here is what makes a run point in time:

```python
from datetime import date

from techval.config import Assumptions
from techval.edgar import EdgarClient, HttpCache
from techval.market import MarketData, make_price_source

assumptions = Assumptions.load("assumptions.yaml")
cache = HttpCache()
client = EdgarClient(cache)                      # add knowledge_date=... to pin it
market = MarketData(make_price_source(assumptions.price_source, cache,
                                      assumptions.price_csv_dir),
                    cache, today=date.today())
```

The entry points for the newer layers, each returning a dataclass carrying its
own notes, flags and provenance:

| Call | Returns |
|---|---|
| `tmt.taxonomy.tmt_universe(client, assumptions)` | the candidate universe, classified, with the unresolved kept |
| `tmt.segments.build_segments(ticker, client, assumptions)` | the segment and geography tables and their residual |
| `tmt.kpis.build_kpis(ticker, client, assumptions, mdna_text=...)` | operating metrics with a source and a confidence each |
| `tmt.metrics.build_metrics(fin, sub_vertical, assumptions, bridge=...)` | the metric pack that sub-vertical is priced on |
| `tmt.sotp.run_sotp(fin, bridge, segments, multiples, assumptions)` | the sum of the parts and its gap to the market |
| `tmt.precedents.build_precedents(tickers, client, assumptions)` | transactions, premia and multiples from merger filings |
| `nlp.sections.load_sections(ticker, client)` | a 10-K split into its Items |
| `ml.features.build_panel(...)` | the point-in-time feature store |
| `ml.encoder.fit_peer_encoder(dataset, assumptions)` | the peer similarity model and its card |
| `ml.warranted.fit_warranted(panel, assumptions)` | the warranted multiple and the residual table |
| `ml.forecast.fit_fade(panel, assumptions)` | the fade curve and the fitted growth path |
| `ml.mna.fit_propensity(...)` | the acquisition screen, its calibration and its labels |
| `ml.signals.test_signal(scores, prices)` | any score tested against forward returns |

Every model reads its switches and its seed from `assumptions.ml`, and the TMT
layer from `assumptions.tmt`. Both blocks have documented defaults, so an
assumptions file written before they existed still loads.

---

## What the data layer handles that a naive one does not

Turning `companyfacts` into an income statement is most of the work, and these
are all observed in filings committed as test fixtures:

- **Retired tags.** A tag that ever appeared stays in the payload forever with a
  stale value. Balance-sheet concepts therefore resolve *as of a date*, and a tag
  whose newest fact is more than 20 days old is skipped. Zscaler reports
  marketable securities under `DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent`;
  resolving by first-tag-present returned zero and overstated its EV by $2.5bn.
- **Quarters that were never filed.** Q4 is always the year less nine months.
  Snowflake tags D&A only cumulatively; CrowdStrike files a year-to-date half plus
  a discrete Q2 with no discrete Q1. Periods are recovered by subtracting nested
  windows that share an endpoint, then tiled to twelve months exactly.
- **Stock splits, and one corporate action seen many times.** CrowdStrike's
  four-for-one in mid-2026 leaves pre-split share counts in periods no later
  filing restates. Mixing them gave a TTM diluted count of 271mm against a true
  1,023mm. Worse, a split is not restated all at once: a quarterly report shows
  the current period and its prior-year comparative and nothing else, so one
  four-for-one is first reflected in one set of periods at the next 10-Q, in a
  different set at the one after that, and in more again at the 10-K. Counting
  filings rather than corporate actions multiplied the factor once per sighting,
  and Nvidia's two real splits came back as six: 4 cubed times 10 cubed, and a
  trailing diluted share count of 2.5 trillion against a true 24.9 billion. Market
  capitalisation is that count times a price, so it was wrong in the same
  proportion at every date before the splits. Each sighting is now treated as an
  *interval*, because a period whose value moves between a filing on A and the
  next on B says the split took effect somewhere in (A, B]. Every sighting of one
  split brackets the same day so their intervals intersect; two real splits at the
  same ratio are years apart and cannot. `tests/test_split_detection.py` fails
  before the fix on a committed cut of the real Nvidia fact set.
- **A split is a unit and not information, so it is deliberately not knowledge
  dated.** This is the one place the engine reads outside its own knowledge date,
  and the reason is arithmetic. The price vendor restates its whole history for a
  split, so a share count left on the basis of its own day paired with a price
  already divided by ten gives a market capitalisation wrong by ten. Both sides of
  a ratio have to be quoted in the same unit; only the facts have to be point in
  time. `test_split_units_are_deliberately_not_knowledge_dated` pins the argument,
  and `ml.warranted.share_basis_factor` reads the conversion off the filings rather
  than a vendor calendar, as the ratio between a fiscal period's diluted count as
  reported today and the same period's count as reported at the row date.
- **A disaggregation component that outranks the total.** `tags.REVENUE` ranks
  `RevenueFromContractWithCustomerIncludingAssessedTax` above `Revenues`. Charter
  Communications tags the first at 889mm for fiscal 2025 and the second at
  54,774mm, so `build_financials("CHTR")` returned revenue **62 times too small**
  and every multiple struck on it 62 times too high. An opt-in `component_guard`
  on `resolve_ttm` now resolves the remaining ladder entries too and swaps in one
  that covers the same window with a value more than the guard multiple of the
  winner's, recording which tag it beat and by how much. The guard is ten, not two, measured across 224
  filers: the genuine scope disagreements in this universe run two to five times,
  and picking the larger of those by rule would be a thumb on the scale. It is opt-in per concept, because on EBIT an order of
  magnitude is an ordinary year.
- **Averages that are not sums.** Weighted-average share counts are aggregated as
  integrals (average times days) so they can be added and subtracted at all.
- **Dimensioned facts are absent from `companyfacts`.** That endpoint publishes
  undimensioned facts only, so option counts under the award-type and plan axes, a
  dual-class issuer's shares under the class-of-stock axis, and every segment and
  geography row simply are not there. Datadog has no undimensioned
  `dei:EntityCommonStockSharesOutstanding` at all. `parse_instance` reads the
  filing's own instance document instead, where 334.90mm Class A and 24.17mm
  Class B are found and summed to 359.08mm, and it is the same path
  `tmt.segments` and `tmt.kpis` read.
- **Tags that look like the one you want.** The award footnote is full of them.
  CrowdStrike tags an undimensioned
  `OptionsVestedAndExpectedToVestOutstandingWeightedAverageExercisePrice` at the
  same date as the real outstanding strike; Zscaler files its whole antidilutive
  securities table under `OptionsOutstandingNumber`, so 8.9mm RSUs and 1.8mm ESPP
  shares sit under an options tag next to the 150,000 options that are options.
  Tag matching is exact, and option counts are taken undimensioned or by
  exercise-price band and from nowhere else.
- **A roll-forward tags its opening balance with the closing balance's tag.**
  Datadog's option table carries 3,474,619 options at $7.26 and 1,610,360 at $7.83
  under one tag, the first of them six months stale. Only the context date
  separates them, so the latest instant wins.
- **A registrant's own name is not what the SEC file says it is.** The Commission
  appends the state of incorporation to a title: `APPLIED MATERIALS INC /DE`,
  `CHARTER COMMUNICATIONS, INC. /MO/`, on 291 of 10,407 rows in the live ticker
  file. Tokenising the marker into a trailing `de` left the core name as
  `appliedmaterialsincde`, which a proxy naming "Applied Materials" can never
  meet, and it cannot be fixed by adding `de` to the suffix list because those are
  ordinary word fragments. And two symbols on one CIK are often not two share
  classes: Comcast files its common stock as CMCSA and an exchangeable debenture
  as CCZ under one CIK with the same title, so a shortest-symbol rule pointed every
  Comcast label at a debt security. Measured across the live ticker file, that rule
  disagreed with the Commission's own listing order for 224 of the 1,441
  registrants carrying more than one symbol.

Every line item carries provenance: the tag that won, the periods used, the
forms they came from and the method. `techval fetch <TICKER>` prints the table.

---

## Limitations

Read these before quoting a number from this tool.

### The share count and the discount rate

The treasury stock count degrades to diluted WASO rather than guessing. If the
instance document cannot be read, if shares outstanding are not tagged, or if
options are tagged only by award type, the result comes back as `diluted WASO
fallback` with a flag naming what was missing, and the award-type case is the
common one, because a dimensioned option row cannot be told apart from an
antidilutive-securities table reusing the same tag. ESPP shares are never
counted, and neither is any award the filer discloses only in narrative text.
Where bands are tagged but not all of them carry a strike, the single
weighted-average strike is used and dilution is understated to the extent any
band is out of the money. And the count is as of the last filing, not as of
today: shares outstanding come off the latest cover page, awards off the latest
balance sheet date, and anything issued since is missing. Forfeitures are not
haircut either. `assumed_forfeiture_rate` defaults to zero because ASC 260 does
not haircut and guessing a rate is a thumb on the scale.

The discount rate carries four approximations of its own. Book value proxies
the market value of debt, in the EV bridge and in the WACC weights, which is
close for investment-grade paper near par and wrong for distressed. The cost of
debt is synthetic: interest coverage maps to a rating and a spread, and
coverage is the wrong risk metric for a net-cash issuer. Datadog's 1.4x
coverage implies a speculative rating against negligible actual default risk,
and the debt weight is near zero there, so it barely reaches the WACC, but for
a company with real debt and real cash you should override the rate. The equity
risk premium is an assumption, not a measurement, and it is the largest single
unobservable in the output. And peer sets are hand-picked in YAML. There is no
automated screen in the valuation path, and comparability is your judgment. The
warranted-multiple regression needs eight usable observations where the shipped
six-name example produces five, so it declines, and `ml.encoder`, the attempt
to measure comparability, is not wired into `comps.py`.

### The projection, the simulation and the APV

The NOL model stops short of three things that matter: Section 382, which caps
annual use of a carryforward after a change of control, so any deal case here
overstates the shield; state carryforwards, with their own expiry and
apportionment; and the valuation allowance, so the balance used is the gross
federal carryforward rather than the deferred tax asset the filer believes it
will realise. The residual balance at year N is not carried into the terminal
value either, which is struck at the full rate. Working capital is anchored on
a ratio, not on the balance sheet: the year one change is measured against
`nwc_pct_revenue` applied to TTM revenue rather than against the reported
balance, which keeps the path internally consistent at the cost of ignoring
where the balance sheet actually starts.

The Monte Carlo shocks only year-one revenue growth. The path still fades to an
unshocked terminal growth rate, so the spread on the final explicit year is
narrower than a parallel shift of the whole path would give. The exit multiple
is not simulated at all, because sampling a peer median alongside terminal
growth would put two competing terminal assumptions into one histogram. And the
distribution is over assumptions, not over outcomes: nothing here models the
chance that the business is a different business than the one the projection
describes.

The APV holds debt flat at the current balance across the explicit period, and
interest on it is the WACC's own pre-tax cost of debt rather than the filed
interest expense. Nothing in the assumptions describes a paydown, and inventing
an amortisation schedule would put a financing forecast inside the module whose
job is to expose one. The cost is that a filer whose coupon sits far below its
synthetic yield, the zero-coupon convertible signature, gets a modelled shield
larger than the deduction it will claim, and the checks report that gap. The
terminal shield then grows at g while the explicit period held debt flat. That
inconsistency is inside the schedule, it is stated on every run, and it
vanishes when g is zero.

### The merger module

Book equity acquired is understated, so goodwill is overstated. It is built
from working capital less borrowings and any preferred or minority interest;
lease liabilities are left out because their right-of-use assets are not
carried here and deducting one side alone would invent goodwill, and
non-current operating assets are missing for the same reason. The deal
therefore looks more dilutive here than it should, not less. The multi-year
roll-forward needs D&A, capex and pre-tax income and refuses where a filer does
not tag them: CrowdStrike as an acquirer fails on the D&A tag, the same gap
that makes its EV/EBITDA non-meaningful in the comp table.

**The CLI cannot render the multi-year pro forma.** With
`merger.purchase_accounting.enabled` the model computes correctly and the
opening balance sheet prints, then `_render_purchase_accounting` passes each
`(year, dollars, percent)` tuple from `accretion_by_year` to a formatter
expecting a float and raises `TypeError`. The figures are reachable from the
Python API. The renderer is not fixed yet.

Left at its default the merger module is a screen: no opening balance sheet, no
goodwill, no deferred tax on the step-up, no deferred revenue haircut, no debt
paydown. Year-one EPS on two sets of TTM figures tells you the shape of a deal
and roughly what it costs. **It is not a merger model, and no one should take a
bid to a board on it.**

### The data layer

Segment detail is read from the latest 10-K only, so there is no segment time
series, and segment EBITDA is not built, because segment D&A is disclosed
inconsistently and adding a partial D&A back to segment EBIT would produce a
margin nobody reported. Geography rows carry long-lived assets, which is what
ASC 280 requires, so the two asset columns mean different things and each row
says which tag produced it. A sum of the parts, in turn, is a gross asset-value
argument. There is no tax leakage on a break-up, often the thing that kills the
pitch, and the discount when set is applied uniformly, so a partly owned
subsidiary or a listed tracking stake is not supported. Segment profit is
reconciled to consolidated revenue and not down to consolidated operating
income, so a corporate cost that does not actually close the footnote
reconciliation is accepted as given.

`tmt.metrics` and `tmt.taxonomy` speak different vocabularies. `build_metrics`
routes on `software, saas, internet, media, streaming, entertainment, telecom,
wireless, cable, fiber, towers` while `taxonomy.SubVertical` speaks in
`infrastructure_software, application_software, internet, semiconductors,
hardware, it_services, payments, media_entertainment, telecom, towers_fiber,
gaming`. The CLI bridges them through an explicit map in
`commands_tmt.SUB_VERTICAL_TO_METRIC_PACK`, but the two vocabularies have not
been reconciled at the source, and
`test_tmt_metrics_cannot_be_run_across_this_universe` is a tripwire as much as
a test. Reconciling them is a design decision about which pack a bucket belongs
to, so it is flagged. I have not guessed at it. Several inputs to the metric
packs have to be supplied as well: sales and marketing, content spend, content
amortisation, maintenance capex and subscriber counts are income statement and
footnote lines the normalized statement set does not separate, and anything
missing is `None` with a flag naming the exact key, never a zero. The media and
telecom packs are proven on synthetic cases, because no fixture company is a
streamer or a carrier, so the content gap and capex intensity are tested
against arithmetic that checks on paper.

Two limits sit under the precedent transactions. Premia over closed deals need
a local CSV price source: Nasdaq answers "Symbol not exists" for a delisted
target, and Stooq is behind a bot check this engine will not defeat, so a live
run over completed deals returns multiples and no premium, flagged and never
zero. Every priced deal in the fixture set is a pending transaction, and that
is exactly why its ticker still trades. And the acquirer on a precedent is the
entity the agreement names, which is often not the buyer a reader has in mind.
LiveRamp's comes back as MMS USA Holdings, Inc., the vehicle in the merger
agreement, with Publicis Groupe S.A. carried beside it in the other-parties
note. Every priced figure in a `Transaction` comes from the target side and
from the consideration clause, so the acquirer label reaches no premium and no
multiple, but it should not be printed in a table without being read.

### The models

**A universe of today's survivors runs under every one of them.** The SEC's
`company_tickers.json` is a snapshot of currently registered filers and the
price sources delete a company the day it stops trading, so five seed names are
missing on every date, including the dates on which they were live and cheap.
Two separate losses sit under the peer models and they are worth keeping apart.

The one you can measure: of the peers a proxy names and the resolver does
resolve, 2,201 of 2,573 (85.5%) are in the candidate universe at their own
date, and recall at any k is capped by that share. Most of that shortfall is
not committees reaching outside TMT. It is this repository's own extraction
failing on names the committed panel already carries, and the fixtures record
which name died where. Intel, named 61 times, and HubSpot, at 54, have no Item
1 in the committed corpus at any panel date; Intel answers Item 1 by
cross-reference to page numbers, so there is no section to extract, and the
splitter declines rather than returning the wrong span. MongoDB, at 47, and
Visa fail on `could not source 'diluted shares'` at every panel date. KLA, at
33, and IBM, at 20, fail on `could not source 'EBIT'`, and so do the REIT and
pharmaceutical names the proxies carry, Public Storage and Welltower through
Johnson & Johnson and Pfizer. Alphabet, at 25, is lost twice over, with no
clean feature row at any date and no Item 1 before the 2025 panel date. The
loss that is structural rather than a bug is the smaller share: the 20-F
filers a committee names, SAP, STMicroelectronics, Spotify, file no 10-K and
so have no Item 1 to split, and the odd retailer or airline never enters the
panel at all.

The one you cannot measure: a deregistered company is never resolved and so
never becomes a named peer, so it lands in `PeerGroup.unresolved` and never
reaches a survivorship count. The committed peer fixture holds only the spans
that resolved, so the size of that second loss cannot be reproduced from this
repository, and no figure for it is quoted here. Its shape is visible from the
other side: `former_tickers.FORMER_TICKERS` carries Splunk, Coupa, Xilinx,
Activision, VMware, Slack, Citrix, Zendesk and Juniper with the dates each left
the tape, 2021 through 2025. The refusal to guess is what makes the loss
auditable. The honest fix is a delisting-complete vendor file or an archived
ticker file per date, and this package has neither.

On the statistics themselves: `fold_sd` is dispersion across folds where the
result holds folds, and across queries where a ranking result holds per-query
scores, and it is not a confidence interval either way. On a handful of folds it is an honest error bar and not a
significance test. Quarterly readings of the same companies are not independent
draws, and nothing in the evaluation harness corrects for that; only
`ml.signals` does, and only for a score tested against forward returns, where
its Newey-West correction is itself an undercorrection, by a knowable amount of
roughly a fifth at this sample size, and the module says so.

The feature panel carries no market data at all. It was recorded that way
under a belief that Nasdaq's keyless quote API only reaches back three years,
which measurement later contradicted: the endpoint serves ten years for every
symbol tested, so the early panel dates could have carried prices and, as
committed, do not. Re-recording the panel with the feed is the obvious next
experiment. In the meantime 14 of the 50 features are missing on every row: market
capitalisation and enterprise value, the five capital ratios struck against
them, momentum at three, six and twelve months, relative momentum, the 52-week
range position, beta and realised volatility. The size gate and the
size-and-growth baseline fall back to log total assets, a different quantity.
This is the single largest limitation on the peer encoder and it is very likely
part of why its fundamentals tower is the weaker of the two in the ablation,
where the text tower's edge over the noise is a hair and the fundamentals
tower's contribution sits inside its own dispersion. The committed peer fixture
also truncates Item 1 to 2,500 characters, so the encoder score that reproduces
offline is 0.5407 against the 0.594 measured on the untruncated documents. Both
are stated above with the cause.

The acquisition screen cannot say whether cheap companies get bought, because
the valuation channel had to be removed to keep the delisting out of the
labels. Nine of its fifteen coefficients change sign somewhere across the five
folds and nothing in it should be quoted to two decimal places. Some of its
label sets carry false positives and were not hand-cleaned: the seven
transactions recovered by the consideration-clause fix were audited by hand,
three of them are an acquirer-side document or an internal reorganisation that
the extractor cannot yet tell from a sale, and the same error rate presumably
runs through the rest. A label set corrected by my opinion of which deals are
real is my opinion wearing a label set's clothes, so it ships as the code
produced it with the audit in the manifest.

The fitted growth path is not constrained to decay. Each horizon is fitted
directly rather than by iterating the one-year model, so nothing forces the
fitted years to fade monotonically, and `GrowthPath.notes` flags it. The
one-year slope is 0.47 and the three-year slope is 0.17, which is not 0.47
cubed, so a constant-decay model is wrong about the shape as well. And the
prediction intervals on the fade are unconditional, the same width for every
company, because the residual quantiles of a sub-vertical with 80 observations
are two observations deep at each tail.

### The backtest, and scope

The backtest universe is survivorship biased and the harness does not fix it.
The ticker list is supplied by the caller, so it contains only companies that
exist today; names delisted, acquired at a discount or wound up during the
sample are absent and are disproportionately the losers, so every statistic is
biased upward by an amount the harness cannot measure. Overlapping windows are
not independent draws either: valuing the same name quarterly on a one-year
horizon shares most of the price path between observations, names in the same
months are correlated through the market, and any significance computed from
the raw n is overstated. The count of non-overlapping windows is reported
beside it, and `ml.signals` is where the correction itself lives. Returns are
price returns, not total returns; no name in this universe pays a material
dividend, but the convention is stated. And the risk-free rate is held flat
across dates unless you supply one per date, so measured upsides move with
fundamentals and the share price and not with the rates that were actually
quoted.

The scope is US GAAP only. No foreign private issuers, no IFRS, no 20-F
filers.

---

## Tests

```bash
uv run pytest
```

1,701 tests in about three minutes, 496 of them over `techval.tmt` and 500 over
`techval.ml`. No test touches the network. The 14MB of fixtures are frozen SEC
payloads, pruned XBRL instance documents, real proxy statements, verbatim merger
filings and recorded daily closes, each carrying its retrieval date and the
exact pruning rule applied, and several carrying a `MANIFEST.json` with row
counts and SHA-256 digests. A test checks the facts in a fixture against the
rule its header states, so the provenance note cannot quietly stop being true.
Filing text is verbatim apart from markup stripping, punctuation included: a
fixture edited to suit a house style is no longer evidence of what the filer
wrote.

Five companies carry the core valuation tests and each is there for what it
breaks. Datadog for in-the-money convertibles, thin GAAP EBITDA and a dual-class
share count. CrowdStrike for a January year end plus a four-for-one split and no
combined D&A tag. MongoDB for finance leases. Zscaler for a retired-tag balance
and for out-of-the-money options filed under an antidilutive-securities table.
Verizon joins them for the debt ladder, where `LongTermDebtNoncurrent` has a
newest value dated 2013 and the fixture keeps the concepts back to the first
filing because the thirteen-year-old value *is* the test.

Around them sit the fixtures the newer layers need: 105 submissions payloads for
the taxonomy, Disney's fiscal 2025 instance facts for segments, four instance
documents chosen one per business model for the KPIs, two 10-Ks committed in
full rather than pruned because pruning would delete the contents-table problem
the splitter exists to solve, nine real TMT merger deals, the compensation peer
groups of the seed universe, and the recorded panels behind every model score
quoted above.

The accretion/dilution test checks against a worked example whose arithmetic is
written out in the test docstring, and separately asserts that substituting the
solved breakeven synergy figure returns pro forma EPS to standalone. The
simulation reconciles its vectorised central draw against `run_dcf` on every run
and raises rather than prints when the two disagree. The APV reconciliation is
tested on a constructed constant-leverage firm where the identity has to hold
exactly, so a failure there is arithmetic rather than opinion.

On the model side the tests are built to fail in the right direction. The
embargo test scores one prediction rule twice on identical windows and asserts
it loses to a constant once the gap is inserted. `gradient_check` is run over
every layer and every loss, and two tests break a backward pass on purpose so
the check is shown to catch a bug. Determinism is asserted bitwise across
processes and across `PYTHONHASHSEED` rather than to a tolerance. And the
results that embarrass the models are pinned:
`test_the_one_year_fit_does_not_beat_persistence_outside_the_noise` and
`test_value_in_technology_is_a_negative_signal_and_not_a_significant_one` both
exist so that a tie and a failure cannot drift away quietly.

---

## Data sources and SEC fair access

- **Fundamentals**: `data.sec.gov/api/xbrl/companyfacts` for undimensioned facts,
  and the filing's own XBRL instance document for the dimensioned ones that
  endpoint drops. Requests declare a contact in the `User-Agent` as the SEC
  requires, are throttled to 8 per second against a published limit of 10, and back
  off exponentially on 429 and 503.
- **Filing text**: the primary document of the filing itself, stripped of markup
  and split into its Items. Modern filings are inline XBRL, so the human-readable
  HTML and the tagged facts are the same file.
- **Registrant identity and industry**: `data.sec.gov/submissions` for the SIC
  code, the registered name and the filing history, and `company_tickers.json` for
  the symbol. Both are snapshots of the present, the survivorship limitation
  named throughout this README.
- **Prices**: Nasdaq's public quote API. No key, no account. One request per ticker
  per run.
- **Risk-free rate**: US Treasury daily yield curve, 10-year constant maturity.

Nasdaq and Treasury both reject clients whose `User-Agent` they do not
recognise, and neither publishes a fair-access policy of the kind the SEC does,
so requests to those two hosts carry a browser string. The SEC client declares a
real contact address. These two are not held to the same standard.

The package originally targeted Stooq for prices. Stooq now answers plain HTTP
clients with a JavaScript proof-of-work challenge instead of CSV. Defeating a
bot check the operator deliberately erected is not something this package will
do, so the Stooq source raises an explanatory error and points at the
alternatives. Sources sit behind one interface, and a local CSV source makes a
valuation reproducible offline years later. That is what the backtest runs on.

All HTTP responses are cached by request URL under `~/.techval/cache`, so the
same ticker and the same assumptions produce identical output on every run.
Fitted models and text features cache separately under
`assumptions.ml.cache_dir`, carrying a parser version so a stale cache from an
older parser is re-read rather than trusted.

## Licence

MIT.
