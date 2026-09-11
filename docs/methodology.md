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

The engine is deterministic, with one scope worth stating precisely. Filing
data is cached by request URL under `~/.techval/cache`, and those URLs carry
only the CIK, so the same ticker against the same cached filings reproduces the
same fundamentals indefinitely. Price requests embed the request window, whose
end is the run date, so a market-data run is byte-reproducible on the same day
and refreshes on the next. To pin a valuation completely, point
`price_source: csv` at exported price files; that is how the test suite and the
demo notebook stay identical forever.

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
it carries a fact within 20 days of the balance-sheet date being asked about,
and an explicit zero does not end the search: a filer can report zero under the
ladder's first tag while carrying the real balance under a later one, so a
non-zero balance at the date beats a zero above it, with the provenance saying
so. When every entry is stale, a concept that must exist raises
`StaleDataError` listing what was found and when; a concept whose absence
legitimately means zero (preferred, non-controlling interest) records "only
stale tags found, read as zero" in its provenance instead of raising.

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

   One asymmetry is accepted deliberately: recast statements furnished on Form
   8-K, after a segment change or a discontinued operation, rank below the
   periodic reports, so a recast that exists only in an 8-K is not picked up
   until the next 10-Q restates the comparatives. Ranking 8-K above the audited
   reports would let every unaudited earnings release override them, which is
   the worse trade.
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

**One corporate action gets one threshold.** A split is not restated all at once.
The first 10-Q after it restates the comparatives that quarter happens to show,
the next 10-Q restates its own, and the 10-K restates the annual periods, so a
single four-for-one is detected in three separate filings months apart. Counting
each detection as its own action multiplies a pre-split fact by four three times.
Nvidia is the case: two splits, a four-for-one in July 2021 and a ten-for-one in
June 2024, produce six detections, and its fiscal 2022 diluted share count came
back as 2,535,000mm against a true 25,350mm. Detections of the same ratio are
therefore one action unless a later one restates a period *ending after* the open
action's own filing date, since a period that closed after a split was first
reported on the new basis and cannot be that split restating itself. Arista is
the case that decides the rule, with two genuine four-for-one splits three years
apart across six filings.

**A split the knowledge date has not reached has not happened.** Detection reads
the same rows `facts` reads and filters them on the filing date in the same way.
Without that filter a run pinned to March 2022 reads Nvidia's June 2024 split off
the 2024 filings and applies it to a 2022 valuation, and every point-in-time
equity value for a company that later split is wrong by the split ratio.

**A price feed is on a different basis, and that is not fixed here.** Price
vendors restate their whole history for a split, so Nvidia's close for 31 March
2022 comes back as 27.29 rather than the 272.86 that printed. A point-in-time
share count is on the basis of its own day. Multiplying the two gives an equity
value a tenth of the truth. The engine's valuation path is unaffected, because a
live run prices today's share count against today's close, but any historical
run pairs the two bases: `backtest.py` values a splitter at a tenth or a
fortieth of its size on a date before the split. `ml/warranted.share_basis_factor`
computes the conversion, from the ratio between a period's share count as
reported today and the same period's count as reported at the row date, and
applies it where that module builds its panel. Lifting it into `ev_bridge` would
change the signature of a function the whole engine calls and has not been done.

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

### 2.6 Dimensioned facts, and the instance document

`companyfacts` publishes undimensioned facts only. Anything a filer reports along
an axis, by share class, by award type, by exercise-price band, by plan, is not
in that payload. It is not marked as missing either. The concept simply looks as
though the company does not report it.

Two consequences, and between them they decide a share count.

A dual-class issuer tags shares outstanding by class, so
`dei:EntityCommonStockSharesOutstanding` does not exist for it undimensioned.
Datadog is one. Its cover page carries 334,904,614 Class A shares and 24,170,410
Class B shares at 31 July 2026, and no total anywhere. Read one class alone and
the company is 24.2mm shares short or 334.9mm short; read `companyfacts` and the
concept is absent.

Option counts and exercise prices sit under the award-type and
exercise-price-range axes. A treasury-stock share count therefore cannot be built
from `companyfacts` at all, because the facts it needs were never published there.

**Treatment.** The XBRL instance document of the latest periodic filing is
fetched and parsed into facts that carry their axes: tag, value, unit, period,
and the member reported along each dimension. The knowledge date is honoured, so
a point-in-time run reaches the filing that was current then rather than the
newest one on file. Four handling rules, each of them a real failure in the
committed fixtures:

1. **Matching is on local names, not namespace prefixes.** Instance documents
   bind different prefixes and different taxonomy versions. A namespace map
   breaks on the next filer; a local name does not.
2. **Inline XBRL repeats a fact once for every place it is rendered.** Facts are
   deduplicated on period and dimensions before anything is summed, or a figure
   that appears twice in the rendered document is counted twice.
3. **A roll-forward tags its opening balance with the same tag as its closing
   balance.** Datadog's option table carries 3,474,619 options at a $7.26
   weighted-average strike and 1,610,360 at $7.83 under one tag, the first being
   the opening balance six months stale. Only the context date separates them,
   so the latest instant wins and everything earlier is ignored.
4. **Tag matching is exact, never a substring.** Zscaler tags its entire
   antidilutive securities table under `...OptionsOutstandingNumber`, dimensioned
   by award type, so 8,879,000 restricted stock units, 1,757,000 employee stock
   purchase plan shares and 997,000 performance shares sit under an options tag
   beside the 150,000 that are actually options. Option counts are taken
   undimensioned or by exercise-price band and from nowhere else.

Period-end share counts are therefore available, and §4.3 is written around them.
Trailing diluted weighted-average shares remain the documented fallback, used
whenever the instance document cannot be read or the award tags cannot be
trusted.

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

Two counts are available and they answer different questions.

**TTM weighted-average diluted shares** is the denominator of reported EPS. It is
the default, and it is consistent with the earnings figures in every multiple
built on it. It is also an average over a past window, so a company issuing
steadily ends that window above its own average, and it reflects the awards
outstanding across the window rather than the ones outstanding now.

**A treasury-stock count at today's price** is what a live model uses for a
point-in-time equity value. `dilution.method: treasury_stock` builds it from the
filing's own instance document (§2.6). Both counts are reported either way, with
the difference in shares and in percent, because the size of that difference is
the point.

**The method, ASC 260.**

```
  shares outstanding, summed across classes
+ net new shares from in-the-money options
+ unvested RSUs at their full count
= fully diluted shares
```

An in-the-money option is assumed exercised. The company issues the full option
count and spends the exercise proceeds buying its own shares back in the market,
so only the difference is genuinely new:

```
net new shares = count x (1 - strike / price),   when price > strike
```

An out-of-the-money option is antidilutive and contributes exactly zero. It is
dropped, not netted: past the strike the formula turns negative, and worthless
options would shrink the share count and lift per-share value. Zscaler's 150,000
options struck at $232.89 against a $166.10 close would otherwise subtract 60,316
shares. An RSU has no strike, so there are no proceeds and no buyback, and each
unvested unit adds a full share.

**Which price.** ASC 260 runs the test on the average market price over the
reporting period, because the diluted EPS denominator is itself a period average
and the numerator and denominator have to describe the same window. This is not
an EPS calculation. The count is struck at an instant and then multiplied by
today's price to reach equity value, so today's price is what decides which
options are in the money. Testing on a trailing average and valuing at spot mixes
two dates, and in a stock that has run it drops options that are in the money
right now.

**Datadog, at the 9 September 2026 close of $225.27:**

| | mm shares |
|---|---:|
| Shares outstanding, Class A and Class B | 359.08 |
| + net new from 1.61mm options at a $7.83 strike | 1.55 |
| + unvested units | 17.10 |
| **Fully diluted** | **377.73** |
| Memo: TTM diluted weighted average | 366.93 |
| Memo: difference | +10.79, +2.9% |

Equity value moves from $82.7bn to $85.1bn. The two counts sit apart for two
reasons pulling opposite ways: the weighted average is struck over a past window
and so falls below the count outstanding today, while it already carries a share
of the award overhang that the walk above adds back in full.

**Bands, not one average strike.** Where the filer tags options by exercise-price
range, each band is valued on its own terms, because a band is either in the
money or it is not and averaging across bands loses that. The error from
averaging runs one way. A single average strike below today's price implies the
whole grant is in the money; it credits the company with exercise proceeds from
options nobody would exercise, and those proceeds buy back shares that would
never be bought. Four million options at $10 and six million at $150 average to
$94.00, which is in the money against a $100 price: the average nets 0.6mm new
shares and the bands net 3.6mm, six times as much. A single average strike
therefore understates dilution whenever any band is out of the money. Datadog
tags no bands, so its own count uses the single average and the notes say which
happened.

**The fallback is real, and it is labelled.** The treasury-stock path degrades to
the weighted average rather than guessing. An instance document that cannot be
read, no share count tagged, an option count with no strike beside it, an award
tag that cannot be trusted: each returns the weighted average with a flag naming
what was missing. A weighted average known to be a weighted average is worth more
than a treasury-stock count built on an invented strike.

**What the treasury-stock method still cannot see.** The limitation moved, it did
not disappear.

- *Awards tagged only by award type.* Datadog reports 17,099,356 unvested units
  under one member covering restricted stock units, restricted stock and
  performance stock units together. Whatever performance condition sits inside
  that number enters at the count the filer tagged, and the instance document
  offers no way to separate the three. Where a filer tags several award-type
  members and no combined total they are summed, which is right where the members
  are distinct populations and double counts where one member's label spans
  another's. The members are named in the notes so the footnote can be checked
  rather than taken on trust.
- *Employee stock purchase plans.* ESPP shares are not counted. Zscaler's
  1,757,000 of them are tagged only inside the antidilutive table under an
  options tag, which is refused for the reason in §2.6, and Datadog's sit in a
  context combining a share class with a plan, which is dropped from shares
  outstanding because they are not outstanding shares. Both exclusions understate
  dilution.
- *The convertible interaction.* Conversion shares are deliberately absent: the
  EV bridge has already decided whether an instrument is a debt claim or
  converted equity (§4.2), and counting its shares here as well would count the
  same claim twice. Note which way the two conventions cut, though. Diluted WASO
  already contains the conversion shares of an in-the-money convertible under
  ASU 2020-06, and this count does not. Pairing a treasury-stock count with an
  if-converted bridge needs those shares added back by hand.
- *Forfeitures.* `dilution.assumed_forfeiture_rate` is zero by default, because
  ASC 260 does not haircut for forfeitures and guessing a rate is a thumb on the
  scale. Set it and the haircut reaches unvested awards only: options outstanding
  include vested ones that can no longer be forfeited, and the filing does not
  always separate them.
- *Two dates in one number.* Shares outstanding come from the cover page, struck
  weeks after the balance-sheet date, Datadog's on 31 July against a 30 June
  balance sheet. Awards come from the footnote at the balance-sheet date. Both
  are then valued at today's price, and anything issued since the filing is in
  neither.

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
**its own** D/E and tax rate, pool the resulting asset betas, then relever at the
target's capital structure. The point of the exercise is to strip out
capital-structure differences before averaging, so that what is being pooled is
business risk rather than financing policy. How the pooling is done is §5.3.

D/E uses market equity against book debt, matching the WACC weights.

### 5.3 Pooling the peer betas: median or Vasicek

**Default: the median**, because one peer with a broken regression, a recent IPO
or a takeover rumour in the window should not be able to move the answer.

The median and the Blume adjustment share a blind spot, though. Neither looks at
how well each beta was measured. The median treats a slope estimated on an
R-squared of 0.11 exactly like one estimated on 0.24, and Blume shrinks every
slope by the same third whether it came from a tight regression or from a
scatter.

**Vasicek (1973)** lets the data decide the shrinkage. Each peer's asset beta is
pulled toward the cross-sectional mean of the set by a weight that is the ratio
of the dispersion across peers to that dispersion plus the peer's own sampling
variance:

```
w_i      = sigma_cross^2 / (sigma_cross^2 + se_i^2)
beta_i*  = w_i * beta_i + (1 - w_i) * beta_cross_mean
```

Read the weight directly. A peer whose standard error is small against the spread
of the group keeps its own number. A peer whose standard error is as wide as the
spread of the group is told that half of what it thinks it knows is noise. The
pooled figure is then the precision-weighted mean of what comes out, on sampling
precision, `1 / se^2`. The posterior version, which would add `1 / sigma_cross^2`
to every peer, compresses the weights toward equal and is harder to defend at a
desk: the number quoted is the one that comes off each regression's own output.

**Applied at the asset beta, with the standard error unlevered alongside it.**
This is the step that is easy to get wrong. Both operations between the OLS slope
and the number being pooled are multiplications by constants rather than new
estimates: Blume multiplies the slope by 0.67, and unlevering divides by
`1 + (1-t) D/E`, which comes off the balance sheet. The standard error rides
through both on exactly the same scale. Skip the division and a levered peer's
asset beta is paired with the standard error of its equity beta, which is larger
by the leverage factor, so its weight comes out too low and it is dragged toward
the peer mean for no reason except that it carries debt. The shrinkage would then
be reading capital structure, which is the one thing unlevering exists to remove.

**Six software peers against SPY**, weekly returns over two years, each unlevered
at 0.05x D/E and a 24% tax rate, Datadog relevered at its own 0.01x:

| peer | raw beta | s.e. | R2 | asset beta | weight kept | shrunk |
|---|---:|---:|---:|---:|---:|---:|
| CRWD | 1.628 | 0.285 | 0.24 | 1.569 | 0.52 | 1.500 |
| MDB | 1.987 | 0.393 | 0.20 | 1.914 | 0.36 | 1.602 |
| ZS | 1.155 | 0.294 | 0.13 | 1.113 | 0.50 | 1.269 |
| NET | 1.407 | 0.332 | 0.15 | 1.356 | 0.44 | 1.395 |
| SNOW | 1.435 | 0.373 | 0.13 | 1.382 | 0.39 | 1.409 |
| HUBS | 1.270 | 0.352 | 0.11 | 1.224 | 0.41 | 1.342 |

Cross-sectional mean 1.426, standard deviation 0.284.

| | median | Vasicek |
|---|---:|---:|
| pooled asset beta | 1.369 | 1.411 |
| relevered | 1.381 | 1.424 |
| cost of equity | 11.74% | 11.95% |
| WACC | 11.70% | 11.91% |

Twenty-one basis points of cost of equity, and the reason is MongoDB. It carries
the widest standard error in the set, keeps the least of its own estimate at
0.36, and comes back from 1.914 to 1.602. The median cannot see that, because the
median reads only the middle of the sorted list and cannot tell whether the
estimate sitting there was the tightest regression in the set or the loosest.
Note also the direction. Shrinkage pulled the highest beta down hard and the
pooled figure still came out **above** the median, because the mean of this set
sits above its middle and the median was ignoring three quarters of the evidence
about where the centre is.

**The trade.** Vasicek buys information the median throws away, and pays for it
with the one thing a median is good at. The pooled mean uses every estimate, so
one genuinely broken regression moves it, and shrinkage narrows a bad estimate's
influence without ever removing it, where a median ignores it outright. On a set
of six that is why the median stays the default. Switch with
`market.peer_beta_method: vasicek`. A Vasicek request over a degenerate set,
fewer than three peers, no cross-sectional dispersion, or a standard error that
is zero or missing, falls back to the median and says so in the notes rather than
dividing by zero.

**Do not stack Blume on top of it.** The two are answers to the same question.
Running Blume-adjusted estimates into Vasicek shrinks each peer twice, two thirds
of the way toward 1.0 by a fixed rule and then toward the peer mean by its own
precision, which pulls dispersion out of the set beyond what either correction
intends. The engine says so in the notes whenever it sees both.
`market.beta_adjustment: raw` lets the data do the shrinking on its own.

### 5.4 Cost of debt

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

### 5.5 Weights

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

**Closing that loop: SBC dilution.** The addback argument only works if the
shares the compensation creates are modelled alongside it. `dcf.sbc_dilution`
does that and is on by default: every projected year issues
`SBC dollars / share price` new shares, and the count carried forward grows by
them. Under expensing it does nothing, because no benefit was taken and so there
is no cost to charge.

Two judgments sit inside that line, and a finding falls out of it.

The price is held at today's price for the whole forecast. A rising share price
would issue fewer shares for the same dollars, so a constant price is a choice
rather than a neutral assumption. Growing the price at the cost of equity would
make the share count depend on the answer the model is trying to produce, and the
exercise is to hold an intrinsic value up against the price quoted today. Where
the model's own value per share is far below that price, as it is across this
cohort, shares are being issued at a valuation the model does not believe, so the
dilution charged here is the smallest defensible one.

The per-share value divides by the share count at the **end** of the explicit
period, not by the opening count and not by the average. Every share the addback
pays for has been issued by then, and the terminal value belongs to the register
as it stands at that date. Against the year-one flows that is too many shares.
Against the terminal value it is far too few, for the reason below, and the
second error is much the larger of the two.

**The two camps do not converge.** This is worth stating plainly, because the
natural expectation is that modelling the shares closes the argument. It does
not. Datadog at a pinned 10.5% discount rate on the example assumptions:

| SBC treatment | value per share |
|---|---:|
| `expense` | $39.55 |
| `addback`, no dilution | $85.67 |
| `addback`, with dilution | $79.60 |

Modelling the shares closes 13% of the gap. The two camps are still a factor of
two apart.

The reason is structural rather than a matter of calibration. Five years of
issuance takes the count from 366.9mm to 394.9mm, cumulative dilution of 7.6%.
But 71% of the addback's uplift in enterprise value sits in the **terminal
value**, which capitalises the addback in perpetuity, and no modelled issuance
funds any of it: the register stops growing at year five while the flow it is
divided into is grossed up forever. A denominator 7.6% larger cannot pay for a
perpetuity.

Two further pressures push the same way at this company. Datadog's SBC runs at a
fifth of revenue against a GAAP EBIT margin under one percent, so the addback is
enormous relative to the earnings it is added to. And the shares are issued at
$225.27 while the model says the equity is worth $39.55, so each dollar of
compensation buys very few shares. Issuing at the model's own value would dilute
far harder.

The checks report the cumulative dilution and the terminal year's issuance rate
beside the terminal growth rate, because that pairing is what the addback camp
actually has to defend. Datadog's terminal year issues 1.66% of the share count
in stock compensation, so against 2.50% terminal growth, cash flow per share
compounds at 0.83% and not at 2.50%.

**Working capital.** NWC is taken as a percentage of revenue, and the year-one
change is measured against that same percentage applied to TTM revenue, rather
than against the reported balance. Anchoring on the reported balance would
introduce a one-time step in year one that was never an operating event. For
subscription software the percentage is negative, because customers are billed
ahead of delivery and deferred revenue grows with the business, so growth
releases cash rather than consuming it.

**Taxes** are `rate * max(EBIT, 0)` by default. A loss year creates a
carryforward, not a refund, and with the carryforward untracked no cash benefit
is booked in the loss year and none is taken later. For a company with large
accumulated losses that understates value, which is what §6.2 is for.

### 6.2 Net operating losses

Off by default, on with `dcf.nol.track`. On, the tax line becomes a cash tax
rather than a statutory charge on positive EBIT.

A projected loss adds to the carryforward balance and pays nothing. A profitable
year shelters the lesser of the balance and `nol.annual_limitation_pct` of
taxable income, and pays tax on the rest. NOPAT is struck on that cash tax, so
the shield reaches free cash flow in the year it is used rather than being
asserted in a footnote.

**The 80% limitation is the part people forget.** Post-2017 federal losses carry
forward indefinitely, but they offset only 80% of taxable income in any one year.
A company sitting on a decade of accumulated losses still writes a cheque the
moment it turns profitable, however large the balance. A model that shelters the
whole of taxable income overstates free cash flow by a fifth of the tax bill for
as long as the balance lasts. Datadog with a $3,000mm opening balance shelters
$2,714mm across the explicit period, worth $651mm of cash tax not paid and $460mm
discounted, and still pays cash tax in every one of the five years including the
first.

**The opening balance.** It comes from the assumptions file where the analyst
supplies one, and otherwise from `OperatingLossCarryforwards` where the filer
tags it undimensioned. An analyst reading the tax footnote knows more than the
tag does, so the assumption wins where both exist. Where neither exists the
engine raises rather than assuming zero, because the carryforward is commonly
reported only by jurisdiction and a dimensioned fact does not reach
`companyfacts` (§2.6).

**What the carryforward model does not do.** Section 382 caps the annual use of a
carryforward after a change of control at roughly the equity value times the
long-term tax-exempt rate, and it is not modelled, so any deal case here
overstates the shield. State carryforwards, with their own expiry and
apportionment rules, are not tracked. No valuation allowance is applied, so the
balance used is the gross federal carryforward and not the deferred tax asset the
filer believes it will realise. And the residual balance at year N is not carried
into the terminal value, which is struck at the full rate, so an unused balance
is worth something this model does not count. Datadog leaves $286mm unused at
year five in the example above, and the checks say so.

### 6.3 Discounting

Mid-year convention by default: explicit-period flows are discounted at
`(1 + w)^-(t - 0.5)`, on the reasoning that revenue arrives through the year
rather than in a lump on 31 December.

### 6.4 Terminal value, and the timing distinction

All three methods are computed and all three are shown, because agreement between
them is information and disagreement is more. `dcf.terminal.method` picks which
one drives the headline figures; the Gordon and exit-multiple fields keep their
own meanings whatever it is set to.

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

**Value driver.**

```
TV_N = NOPAT_{N+1} * (1 - g / ROIC) / (w - g)
```

`g / ROIC` is the reinvestment that growth of `g` requires at a return of `ROIC`,
so `(1 - g / ROIC)` is the share of NOPAT that reaches investors. The Gordon path
assumes capex and working capital in the terminal year and leaves the implied
return on capital to be checked afterwards (§6.5). This inverts the two: the
analyst states the return they will defend, and the reinvestment is derived from
it, so the terminal value cannot embed a return nobody signed up for.

The two are one formula with different inputs held fixed. Both are NOPAT less
reinvestment, over `w - g`, on the first perpetuity year. Set the ROIC to the one
the Gordon path implies and the two terminal values are the same number.

With `terminal.terminal_roic` null the return falls back to the WACC and the
bracket becomes `(1 - g / w)`, the competitive-equilibrium view that growth in
perpetuity creates exactly nothing. At ROIC equal to WACC the formula collapses
to `NOPAT / w` for any `g` at all, which is the cleanest statement of that idea
available: if a business earns its cost of capital and no more, how fast it grows
does not change what it is worth. A model in which `g` still moves the answer
under that assumption has a return assumption hidden inside it somewhere.

It capitalises a stream rather than observing a price at a date, so it takes the
Gordon timing and not the exit multiple's: the same half-year uplift under the
mid-year convention, discounted at `(1 + w)^-(N - 0.5)`.

Datadog at a 10.5% discount rate, with the return falling back to the WACC, puts
the terminal value at $10,548mm against Gordon's $13,605mm, and value per share
at $34.23 against $39.55. The difference is reinvestment, and the engine itemises
it. On the same steady-state year the Gordon formula gives $13,499mm, because it
spends only $28mm in the first perpetuity year, capex plus the working-capital
change less D&A, while growth of 2.50% at a 10.5% return on $1,108mm of NOPAT
requires $264mm. That $236mm of extra reinvestment is $2,952mm of terminal value.
The $106mm left between $13,499mm and the $13,605mm actually reported is the
shortcut flow, Gordon struck on the final explicit year grown at `g` rather than
on the steady state, which is a difference of input and not of method.

Read the other way round, those same Gordon assumptions imply a terminal return
on capital of 100.5%, a perpetual return that assumes no competitor ever arrives.
That is the check in §6.5 item 4 used as an input rather than as a warning.

**The walk back to equity never subtracts operating leases**, whichever bridge
convention is configured. The projected cash flows pay rent in every explicit
year and in the terminal perpetuity, so the lease obligation is serviced inside
the DCF; subtracting the liability as well would charge the same lease twice,
the DCF-side twin of the EV/EBITDA pairing trap. Finance leases stay in the
walk, because unlevered FCFF excludes their interest and principal and the
claim is therefore still outstanding.

### 6.5 Cross-checks

These are where a DCF is either coherent or is not, and the engine prints them
rather than leaving them to be noticed.

1. **Implied exit multiple from the Gordon terminal value**, restated onto the
   exit method's whole-period discount clock, since the two conventions differ
   by half a year under mid-year discounting and the comparison against a peer
   multiple has to be like for like. If Gordon implies 7.7x and the peer set
   trades at 37.3x, the growth and margin assumptions do not support what the
   market is paying, and one of the two is wrong.
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

   The subtlety is which year to read the rate from. The final explicit year
   still grows at the faded revenue rate, so its working-capital swing is sized
   for that growth, not for the perpetuity's g, and reading the identity off it
   mixes two growth rates. The engine instead constructs the first perpetuity
   year properly: revenue one notch of g beyond the terminal year, margins and
   capital intensity at their terminal settings, the working-capital change
   sized by g alone. That is the steady state the Gordon formula claims to
   capitalise, and it is Damodaran's construction for exactly this reason.

   The same construction exposes a quieter bias. The terminal value is computed
   off `FCFF_N x (1+g)`, the standard shortcut, and that flow inherits
   reinvestment sized for the faster explicit-period growth. Where the shortcut
   and the steady-state flow disagree by more than two percent the engine says
   so and by how much, because that gap sits inside the terminal value itself.

   The implied ROIC is then flagged when negative, above 60%, or below the
   WACC. That last case is the interesting one: it means the model is
   capitalising growth that destroys value. A terminal assumption that survives
   this check is internally consistent; one that does not is arithmetic with a
   story attached.

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

### 7.1 Warranted multiples from fundamentals

A percentile band says what the neighbours trade at. It knows nothing about the
company it is applied to, so it quietly penalises a name growing faster than its
set and rewards one growing slower. Across software, EV/Revenue is largely
explained by growth and margin, and that is a testable claim rather than a saying.

```
EV/Revenue_i = a + b1 * revenue growth_i + b2 * EBITDA margin_i + e_i
```

Ordinary least squares with an intercept across the peer set, evaluated at the
target's own growth and margin. The output is the multiple the target's
characteristics warrant, and the residual against what it actually trades at. The
residual is the number worth arguing about: what the market pays over or under
what the fundamentals in the regression explain. Off by default
(`comps.regression.enabled`), and it is a cross-check on the percentile band
rather than a replacement for it.

**It refuses rather than overfits, and on a normal comp set it refuses.** Fitting
two coefficients and an intercept to six points memorises the peer set. The plane
passes near every name by construction, the R-squared is high, and the warranted
multiple is a restatement of the inputs. The fit is therefore withheld below
`comps.regression.min_observations`, which defaults to eight, below three
observations per regressor whatever that minimum is set to, and where the design
matrix is singular because two drivers move together across the set or one does
not move at all. The example file's six peers fall below the floor and the engine
declines. On the three-name fixture set only two of the three carry a GAAP EBITDA
margin at all (§2.3), so the fit has two usable observations against a floor of
eight; the note says exactly that, names the peer it dropped and why, and points
at `comps.peers` as the fix.

That refusal is the honest output. A ten-name comp set is a cross-section of ten
points. It can say that the market pays for growth and roughly how much. It
cannot support a claim that a coefficient differs from zero at any particular
confidence, and no p-value is printed for that reason.

**What comes back when it does run.** Coefficients, t-statistics, R-squared and
adjusted R-squared, the standard error of the fit, the usable observation count,
the warranted multiple and the residual. The adjusted figure is the one to quote:
raw R-squared cannot fall when a regressor is added, so on a sample this size it
rewards spending a degree of freedom whether or not the driver carries
information. Below ten observations the t-statistics are labelled indicative
rather than inferential. They are shown so that an unsupported coefficient is
visible, not so that a confidence level can be quoted off them.

**Sign checks, not tuning.** Every driver here raises what a dollar of revenue is
worth. A negative loading on growth or on margin is not a view of the world, it
is a symptom that one peer with an extreme multiple is carrying the fit, and it
is flagged as such. A target sitting outside the peer range on any driver is
flagged too, because the fitted line is then being extrapolated rather than read.
A warranted multiple that comes out negative is refused outright.

**Nothing is imputed.** A peer missing the dependent variable or any driver is
dropped and named. Filling a missing margin with the peer mean would put that
name on the fitted line by construction and inflate the R-squared with a figure
nobody reported.

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

### 8.2 Purchase accounting

Two switches, and they are not the same thing.
`merger.include_intangible_amortization` adds a step-up amortisation charge to
the year-one walk and does nothing else: no goodwill, no deferred tax, no balance
sheet. It is off by default, on the reasoning that the charge is a non-cash
artefact of the transaction and the street quotes cash EPS.
`merger.purchase_accounting.enabled` is the full exercise, and is also off by
default. Switched on it builds the opening balance sheet and rolls the combined
company forward, adding a result rather than restating the year-one screen.

Four pieces of it are where a reviewer should start, because they are the ones
most often got wrong. Two more describe how the years after close are built.

**Goodwill is a plug, and the deferred tax liability makes it bigger.** The
excess of the equity purchase price over the book equity acquired is allocated
first to identifiable intangibles, at the configured share, and whatever is left
is goodwill. A stock deal is tax free to the seller and carries over the seller's
tax basis, so the write-up exists for book and not for tax. That temporary
difference is a liability assumed at close:

```
DTL      = intangible write-up * tax rate
goodwill = excess - intangibles + DTL
```

The plus sign is the part people get backwards. Goodwill is consideration less
the fair value of net assets acquired, and the DTL is a liability inside those
net assets, so booking it takes net assets down and pushes goodwill up. On a
$3,250mm equity purchase price against $600mm of book equity, with 40% of the
$2,650mm excess allocated to intangibles at a 24% rate: $1,060mm of intangibles,
a $254.4mm deferred tax liability and $1,844.4mm of goodwill. Book equity plus
intangibles plus goodwill less the DTL is $3,250mm exactly, and nothing else
balances. Omit the liability and goodwill is short by that $254.4mm, a quarter of
the write-up, and the opening balance sheet does not reconcile.

**Goodwill is not amortised, intangibles are.** ASC 350 stopped goodwill
amortisation in 2001 and replaced it with impairment testing, so no part of that
$1,844.4mm touches EPS until the day it is written down, all at once. The
identifiable intangibles, acquired technology and customer relationships, are
amortised over their useful lives and hit EPS every quarter.

**That amortisation has no cash tax deduction behind it.** The DTL unwinds as the
intangible amortises, which produces a deferred tax benefit, so GAAP tax expense
looks normal and the book charge is the usual `amortisation * (1 - t)`. Cash tax
is not reduced at all, because there was never any tax basis to amortise. Free
cash flow therefore gets back only `(1 - t)` of the charge and not the whole of
it. Adding back the full amortisation overstates cash generation by the tax on
it, and with it how fast the acquisition debt is repaid.

**The deferred revenue haircut destroys revenue outright.** Acquired deferred
revenue is remeasured at fair value, which is the cost of delivering the service
plus a normal margin rather than the amount billed. The difference is revenue the
target would have recognised and the combined company never will. For a
subscription business carrying a year of billings in deferred revenue that is a
visible hole in year-one revenue, and it is why SaaS deals look worse in the
first year than the run rate says they should. The write-down is shown as a memo
rather than inside the allocation, because a full fair value exercise would add
it to identifiable net assets and take the same amount back out of goodwill,
which nets to nothing for EPS. Its earnings effect, which does not net to
nothing, is charged against year-one revenue.

**The roll-forward and the sweep.** Year one covers the same trailing twelve
months as the screen above, so the two are directly comparable and the gap
between them is exactly the purchase accounting. Revenue then compounds at
`dcf.revenue_growth_start`, margins and capital intensity are held where the
trailing twelve months put them, and a configured share of free cash flow repays
the acquisition facility at the end of each year, with the next year's interest
charged on what is left. Interest is charged on the balance at the **start** of
the year: sweeping out of cash the year has not generated yet would credit the
deal with a repayment before the money arrives.

**Where book equity is understated, and which way that cuts.** `Financials`
carries the claims a valuation needs rather than a full balance sheet, so book
equity acquired is assembled from working capital less borrowings, preferred and
minority interest. Lease liabilities are deliberately not deducted: ASC 842 books
a right-of-use asset against the liability at inception and the two stay close,
so subtracting the liability while the matching asset is absent from the object
would conjure hundreds of millions of goodwill out of an accounting entry.
Non-current operating assets are missing for the same reason. Book equity is
therefore understated, and the excess, the write-up and the amortisation are
overstated, which makes the deal look more dilutive here than a full opening
balance sheet would make it, not less. A price at or below book equity is treated
as a bargain purchase under ASC 805, with the difference flagged and kept out of
the pro forma rather than carried as negative goodwill.

### 8.3 What is still not modelled

No revenue build and no margin path: the top line compounds at one rate and every
margin is held at its trailing twelve month level, which is generous to an
acquirer buying a faster-growing, lower-margin target. No section 382 limitation
on the target's carryforwards, which would cut whatever NOL shield a deal case
claims (§6.2), and no carryforwards at all inside the roll-forward, so a pro
forma loss books a full tax benefit at the acquirer's marginal rate. No
retention, integration or restructuring cost beyond the one-time fees in sources
and uses. No change-of-control repayment of the target's
borrowings: they are assumed at close on unchanged terms, and where a covenant
forces repayment instead, that amount has to be added to the acquisition facility
and the whole walk rerun on the larger balance. No share buyback. No second bid.

Left at its default the module stops before all of this, and the year-one screen
is exactly what it says it is: two sets of trailing twelve month figures, telling
you the shape of a deal and roughly what it costs. Purchase accounting turns it
into a three-year model of a combined company. Neither is a merger model, and no
one should take a bid to a board on either.

---

## 9. Monte Carlo

A discounted cash flow prints one number, and that number is a function of four
judgments nobody can observe. The honest output is a distribution and a
probability statement about it. Off by default (`simulation.enabled`). On, the
engine draws year-one revenue growth, the terminal EBIT margin, the discount rate
and terminal growth together, revalues the company on every draw, and reports
percentiles of enterprise value and of value per share alongside the probability
that value clears the quoted price.

### 9.1 The draw is joint, and that is the whole point

The commonest error in DCF Monte Carlo is four independent normals. Independence
asserts that a company growing twice as fast as expected is no more likely than
average to reach a better terminal margin, and that the market would discount the
two cases at the same rate. Neither is true of any business anyone has ever
covered.

The draws come from a multivariate normal with this correlation matrix, which is
a module-level constant so that a reader can see it and change it:

```
                 growth   margin     WACC   term g
growth             1.00     0.35    -0.25     0.10
margin             0.35     1.00    -0.20     0.05
WACC              -0.25    -0.20     1.00     0.00
term g             0.10     0.05     0.00     1.00
```

Each sign, in a line:

- **growth with margin, +0.35.** Operating leverage. Fixed cost is spread over a
  larger base, and the company that wins the market is also the one that stops
  discounting to win it.
- **growth with WACC, -0.25.** The market discounts a better business at a lower
  rate: scale buys index inclusion, cheaper funding and a lower equity risk
  premium on the name.
- **margin with WACC, -0.20.** The same mechanism, weaker. Cash generation
  reduces financing risk, though a high-margin business is not automatically a
  low-beta one.
- **terminal growth with everything else, +0.10 and +0.05.** Near enough to
  independent, deliberately. The perpetuity growth rate is a claim about the
  economy in fifteen years, not about this year's execution, and treating it as
  an echo of near-term momentum is how a cyclical peak gets capitalised forever.

The matrix is verified positive semi-definite before anything is drawn. A
correlation matrix that is not PSD describes no joint distribution at all: some
weighted combination of the drivers would have negative variance. The error names
the pair that the offending direction loads on, because in practice the failure
is a transitivity one, two strong correlations of opposite sign forcing a third
that was entered by hand.

### 9.2 Correlation widens the distribution here, and the common intuition is wrong

Analysts often say that correlating the drivers narrows the answer, on the
reasoning that independence generates combinations that do not occur. It does
generate them. Look at which ones.

Value rises with growth, rises with the terminal margin, and falls with the WACC.
The three correlations above therefore all point the same way: the good draw is
fast growth and a fat margin and a low discount rate together, and the bad draw
is all three against you. Independence puts mass on the offsetting middle, fast
growth at a thin margin, and the middle is where the point estimate already sits.

Write out the variance of a linear approximation, with `a` and `b` the
sensitivities of value to growth and to margin, and `c` the size of its
sensitivity to the WACC, which enters negatively:

```
Var = a^2 sG^2 + b^2 sM^2 + c^2 sW^2
      + 2ab corr(G,M) sG sM
      - 2ac corr(G,W) sG sW
      - 2bc corr(M,W) sM sW
```

With `a`, `b` and `c` all positive, every one of the three cross terms is
positive at the signs in the matrix. The correlated distribution is **wider**
than the independent one. On Datadog at the default spreads and 10,000 draws, the
correlated sample has a standard deviation of $9.93 a share against $8.15
independent, 22% wider, and it is wider for the right reason: the scenarios that
hurt arrive together.

Narrowing would require believing that growth carries its own discount rate with
it, that `corr(growth, WACC)` is positive, which is the riskier-company,
higher-beta view. That is a defensible matrix and it is one line to enter. It is
not the one argued above, and the engine reports the spread the stated signs
produce rather than the spread the folklore expects.

### 9.3 Truncation, not clipping

A draw where terminal growth lands at or above the discount rate is rejected
outright, not pushed back to the boundary. The Gordon formula divides by
`(w - g)`, so those draws are undefined rather than extreme. Clipping them to
`g = w - epsilon` would pile probability mass on the exact point where the
perpetuity is largest and manufacture a right tail that is pure artefact; the
signature of a clipped sample is a mean sitting above its own 95th percentile.

Rejections are counted and reported, and flagged above 5% of the sample, because
a rejection rate that high means the assumed spreads on the WACC and on `g` are
inconsistent with the central case rather than merely wide. At the example
assumptions nothing is rejected: terminal growth of 2.5% against a 10.5%
discount rate is eight points clear of the boundary and no draw reaches it. Move
terminal growth to 8.5% with a 1.1 point spread on it and half a point on the
discount rate, and 498 draws in 10,000 are rejected, which is where the flag
fires.

### 9.4 What the probabilities are, and what they are not

`P(value > price)` is the share of surviving draws whose value exceeds the quoted
price. It is a statement about a model, conditional on four standard deviations
an analyst typed in and a correlation matrix an analyst chose. The model does not
know whether those spreads are right, and a tight distribution around a wrong
central case is confidently wrong.

Read it as: given this view of the business and this much uncertainty about it,
the stock screens cheap in this fraction of cases. Nothing stronger is available
from it. For Datadog at the example assumptions the answer is 0.0%, and the check
beside it says why: the 95th percentile of value is $58.93 a share against a
quoted price of $225.27. No combination inside the assumed spreads reaches the
market price, so the disagreement with the market sits in the central case and
not in the uncertainty around it. The response to that is to argue about the
terminal margin, not to widen the distribution until it overlaps the price.

### 9.5 The tornado is univariate on purpose

Each driver is moved to the 5th and 95th percentile of its own marginal while the
other three are held at their central values, which ignores every correlation the
simulation is built on. That is not an oversight. The two outputs answer
different questions.

The tornado says which assumption to argue about in the meeting, because it
isolates one at a time and ranks them by the swing in price. For Datadog the
terminal margin swings the price by $19.04 a share across its 90% band, the
discount rate by $14.00, year-one growth by $8.19 and terminal growth by $5.30.
The distribution says how wide the answer is once the assumptions move the way
they actually move together. Reporting only the tornado understates the spread;
reporting only the distribution hides which knob produced it.

### 9.6 Vectorised, and reconciled against the scalar path

Ten thousand calls to the scalar DCF would rebuild the projection ten thousand
times. The projection arithmetic runs as array operations over the draw dimension
instead, looping only over the handful of forecast years. A vectorised valuation
is worth nothing unless it is the same valuation, so every run revalues the
central draw through the fast path and checks it against `run_dcf` on the same
inputs. For Datadog the two agree at a relative gap of zero. A disagreement
raises rather than prints: if the two paths have diverged, the distribution is
centred on a number the rest of the engine does not report.

**Limitations left in.** Only year-one revenue growth is shocked, matching the
assumption the config exposes, so the fade converges on an unshocked terminal
revenue growth rate and the spread on the final explicit year is narrower than a
parallel shift of the whole path would give. The exit-multiple terminal value is
not simulated, because the exit multiple is not one of the drawn drivers and
sampling a peer median alongside terminal growth would put two competing terminal
assumptions into one histogram. And the distribution is over assumptions, not
over outcomes: nothing here models the chance that the business is a different
business from the one the projection describes.

---

## 10. Adjusted present value

```
APV = unlevered enterprise value + present value of the financing side effects
```

A WACC discounted cash flow buries the tax benefit of debt inside the discount
rate. APV takes it back out and prices it as a stream in its own right. The two
are the same valuation seen from different angles, and the reason to run both is
the size of the gap between them: where they disagree, the disagreement is a
statement about financing policy rather than an arithmetic error. Off by default
(`apv.enabled`).

### 10.1 The flows are not re-forecast

The unlevered value re-discounts the free cash flows the DCF already built, at
the unlevered cost of equity instead of the WACC. The terminal value is recovered
the same way: whatever dollar of flow the DCF capitalised is backed out as
`TV * (w - g)` and re-capitalised at `Ku - g`, so the Gordon and value-driver
paths both carry over without being rebuilt.

Two valuations that each re-derive their own cash flows are two models, and the
difference between them is then a mixture of financing policy and forecasting
drift with no way to tell which is which. Here the only difference is the
discount rate.

The unlevered cost of equity is CAPM on the asset beta the WACC build already
produced:

```
Ku = rf + beta_unlevered * ERP + size premium
```

It sits below the levered cost of equity by exactly
`(beta_levered - beta_unlevered) * ERP`, and equals it when there is no debt.

### 10.2 Which rate discounts the shield

This is the whole argument, and both camps are genuinely defensible.

**Modigliani-Miller discounts the shield at the cost of debt.** It says the
shield is exactly as safe as the interest payment that creates it: the company
either pays the coupon and takes the deduction or it does neither, so the two
streams carry one risk. That is right when the debt schedule is **fixed in
dollars**, a term loan amortising on a stated schedule, because then next year's
interest is known today whatever happens to the enterprise.

**Harris-Pringle discounts it at the unlevered cost of equity.** It says the
shield is as risky as the firm. That is right when debt is **rebalanced** to a
constant percentage of firm value, because then the debt balance, and with it the
interest and the shield, moves with the enterprise and inherits its risk.
**Miles-Ezzell** is the same view with one year of grace: next year's debt is
already known, so the first year's shield is discounted at the cost of debt and
everything after it at `Ku`. The difference between Miles-Ezzell and
Harris-Pringle is one year of one rate on one stream, which is second order
beside the choice of camp.

**A constant-WACC model already assumes constant leverage.** That is what a
single discount rate applied to every year means: the weights never move, so the
debt is rebalanced to value in every period. The rate consistent with a WACC DCF
is therefore the second one, `Ku`. Since the cost of debt normally sits below
`Ku`, an APV run at the cost of debt comes out systematically **above** the WACC
answer.

That gap is not an error to be tuned away. It is the price of a fixed debt
schedule over a rebalanced one, and a company that really does hold its debt flat
in dollars while its equity compounds is genuinely worth more than a
constant-WACC model says.

**Default: `cost_of_debt`, Modigliani-Miller.** It is the textbook APV, and it is
not the rate that reconciles with a constant WACC, which is the argument above.
Leaving it as the default puts that gap on the page rather than assuming it away,
and the checks report what the other camp would give, to the dollar. Set
`apv.shield_discount_rate: unlevered_cost_of_equity` when the point of the
exercise is to reconcile against the WACC rather than to price a fixed schedule.
Zscaler, on $1,696mm of debt, values the shield at $616mm discounted at a 6.66%
cost of debt and $335mm discounted at a 10.34% `Ku`, a swing of $281mm against a
DCF enterprise value of $8,809mm.

### 10.3 The terminal shield, without which the comparison is rigged

A WACC DCF capitalises the tax benefit forever, because the rate that discounts
the terminal value is itself net of it. An APV that shields only the explicit
years is therefore guaranteed to come out lower, and an analyst reading it would
conclude that leverage is worth less than it is. The terminal shield here is the
final explicit year's shield capitalised as a perpetuity on the same growth rate
the terminal value uses. For Zscaler at the default shield rate it is $500mm of
the $616mm total, 81% of the whole financing side effect, and $228mm of $335mm at
`Ku`.

That does assume debt grows at `g` in perpetuity while the explicit period held
it flat, which is an inconsistency inside the schedule rather than a hidden one.
It is stated in the notes on every run, and it vanishes when `g` is zero.

### 10.4 The reconciliation, which is the point of the exercise

APV and the WACC answer agree when four conditions hold together, and each one
that fails is reported with the basis points or the dollars it costs.

1. **The shield is discounted at `Ku`, not at `Kd`.** Section 10.2. Zscaler:
   $281mm.
2. **The WACC is the one its own asset beta implies under rebalancing**, which is
   `w = Ku - (D/V) * t * Kd`. The WACC build unlevers with Hamada, the fixed-debt
   relation, and relevers with a debt beta of zero, so the WACC handed in is
   normally a few basis points away from that. On a levered filer this is usually
   the largest single term in the difference, and it is a property of the beta
   convention rather than of the APV. Zscaler: 10.24% implied against the 10.27%
   the DCF used, a gap of 3bp.
3. **The leverage in the WACC weights equals the leverage in the model.** The
   weights are market debt over market capitalisation; the model's own leverage
   is debt over the enterprise value the DCF produced. Zscaler's are 5.99% and
   19.26%, because the DCF values the company well below its market
   capitalisation. A model that disagrees with the market by that much is
   discounting at a capital structure it does not believe, and the check says so.
4. **The mid-year convention is off, or its effect is taken out.** Moving every
   flow half a year earlier multiplies each valuation by `(1 + r)^0.5` at its own
   rate, and the rates differ, so the identity picks up a factor of
   `((1 + Ku) / (1 + w))^0.5`. `mid_year_wedge` reports that amount exactly, and
   subtracting it restores the identity.

With all four satisfied, on a constructed constant-leverage firm, the two agree
to the last decimal. On a real filer they do not, and the residual is itemised
rather than described.

**Where there is no debt there is nothing to reconcile, and that is the test.**
Datadog, with its in-the-money convertibles treated as the equity they are
(§4.2), carries no debt at all. The shield is zero, `Ku` equals the levered cost
of equity because the asset beta equals the equity beta, and APV is the WACC DCF
run a second time: $9,407mm against $9,407mm, a difference of 0.00%. Anything
else there would be a bug in the discounting rather than a finding about
financing. MongoDB, whose only debt is a $26mm finance lease and whose debt
weight is therefore under a tenth of a percent, differs by 0.06%.

**One honest failure mode.** Debt is held flat at the bridge's current balance
across the explicit period, and the interest charged on it is the WACC's own
pre-tax cost of debt rather than the filed interest expense. The rate has to be
the WACC's `Kd`, or the reconciliation is comparing two different credit views
instead of two different financing policies. The cost is that a filer whose
coupon sits far below its synthetic yield, which is the zero-coupon convertible
signature, gets a modelled shield far larger than the deduction it will actually
claim. Zscaler's modelled interest of $113mm is 9.6x the $12mm it reports on the
same debt, and the checks say so rather than leaving it to be found.

---

## 11. Point-in-time valuation and backtesting

### 11.1 Lookahead is the only thing that matters

A valuation model is a claim about the future. The only way to find out whether
the claim has ever been worth anything is to re-run it on dates in the past,
using only what was knowable then, and score the answer against what the stock
actually did afterwards. A backtest that reads today's restated financials into a
2025 valuation will look brilliant and mean nothing.

Datadog makes it concrete. Valued as of 1 March 2026, the latest statements on
file end 31 December 2025 and show a $44mm operating loss on $3,427mm of revenue.
Run with full knowledge, the same company shows the twelve months to 30 June
2026, a $16mm operating profit on $3,967mm. Those are different companies to a
model, and the second one did not exist on 1 March 2026.

### 11.2 The separation is structural, and it is proved afterwards

`knowledge_date` runs through the whole data layer. `CompanyFacts` and the EDGAR
client discard every fact filed after it, including restatements, later
comparatives and split adjustments; the filing picker reaches for the periodic
report that was current then rather than the newest on file; `MarketData` stops
the price series at that date. `--as-of` on every command sets it, so any single
valuation can be struck at a past date and not only a backtest.

Forward prices are the part that discipline cannot handle. They live in a
separate object that the valuation path never receives. There is no route by
which a valuation can reach a price it should not have seen, because the thing
holding those prices is not in scope where the valuation runs.

The result is then proved rather than asserted. Every line item carries the
filing date behind it (§2.7), and the provenance of every observation is walked
afterwards to confirm that nothing reaching the valuation was filed after its
date. The test is the filing date, not the period end: a fiscal year ending 31
December is not knowable on 2 January, it becomes knowable when the 10-K is filed
in February. A client built without a knowledge date, a cached payload reused
from a later run, a fixture loaded by the wrong helper: all three pass every
configuration check and all three are caught here. A leak stops the run rather
than being recorded against one observation, because it invalidates the result
rather than costing one row of it.

### 11.3 Survivorship bias is present and is not fixed here

The ticker list is supplied by the caller, so it is a list of companies that
exist today. Every name delisted, acquired at a discount or wound up between the
valuation date and now is missing from it, and those are disproportionately the
losers. Results from this harness are biased upward, and the bias is not small in
a sector with as much delisting and acquisition as this one.

The honest fix is a point-in-time universe: the SEC's `company_tickers.json` as
it stood at each valuation date, or a delisting-complete vendor file. This
harness does not have one, and no statistic it produces should be read as though
it did. The caveat is attached to every result rather than left in the
documentation. Where a company had not filed by the valuation date, the
observation is recorded as skipped with the reason attached and never dropped, so
the count of what was not valued stays visible; a name that fails to parse is
recorded as failed, which is a different fact about the world and is counted
separately.

### 11.4 Spearman, not Pearson

The relationship between a DCF upside and a subsequent return is monotonic at
best and nothing like linear. A handful of names at 300% upside, which is usually
where the model has broken rather than where the opportunity is, would drive a
Pearson coefficient on their own. Ranking first throws the magnitudes away and
keeps the ordering, which is the only part of the prediction anyone trades on.
Ties share an averaged rank, so the answer does not depend on the order the
observations arrived in. A hit rate is reported beside it, counting only pairs
where both the upside and the return are non-zero, since a prediction of exactly
zero upside has no direction to be right about.

### 11.5 Sample size, and overlapping windows

With a handful of names over a few dates the sample is far too small for any of
this to mean anything, and a rank correlation on twelve observations will happily
print 0.4. Every statistic is reported with its `n` beside it, and below 30
paired observations the checks say plainly that the output is an illustration of
the machinery rather than evidence about the model.

Overlapping windows are the subtler problem. Valuing the same name quarterly on a
one-year horizon produces holding periods that share most of the same price path,
so those observations are not independent draws and any significance computed
from the raw `n` is overstated. The count of non-overlapping windows is reported
alongside the raw count, and for quarterly valuations on a 252-day horizon it is
a small fraction of it. Windows across different names overlap in time as well,
and that is not corrected for at all.

**Conventions.** Returns are price returns, entry at the last close on or before
the valuation date and exit at the first close on or after the horizon, counted
in calendar days. The price layer carries closes rather than a total-return
index, so a dividend-paying name's realised figure understates the holding-period
return by its yield. No name in the technology universe this engine targets pays
a material one, but the convention is stated rather than assumed. An observation
whose horizon runs past the end of the price series records no return and is
excluded, rather than being scored over a shorter holding period, which would mix
holding periods inside one coefficient. Where the valuation price and the scoring
entry price disagree by more than half a percent the run flags it: the two feeds
disagree about what the stock closed at, and the return is being measured from a
different point than the call was made.

**The risk-free rate.** Without a per-date mapping of the ten-year yield, one
rate from the assumptions file is applied across the whole sample. That is not
lookahead, but it is wrong in its own way: the risk-free rate moved several
hundred basis points across 2023 to 2026, so holding it flat suppresses the part
of the discount rate that was genuinely different at each date and pushes all the
variation in measured upside onto the fundamentals. Supplying `risk_free_by_date`
fixes it, and a missing date in that mapping is an error rather than a quiet
fallback to the file, because reverting to a different rate for some dates and
not others would make the sample incomparable with itself.

`techval backtest DDOG,MDB,ZS --from 2024-01-01 --to 2025-09-30 --horizon 252`
runs it over quarter ends, and `--dates` supplies an explicit list instead.
