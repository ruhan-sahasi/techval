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
ratios), that is a split. Requiring at least two distinct periods to move by the
identical ratio separates a split from an ordinary restatement, which moves
numbers by a few percent rather than by exactly four: one period moving by
exactly four could be a typo in one tagged fact, and two could not.

Facts filed before the split's effective date are then restated into current
units: share counts multiplied by the factor, per-share figures divided by it.

**One corporate action gets one threshold.** A split is not restated all at once.
The first 10-Q after it restates the comparatives that quarter happens to show,
the next 10-Q restates its own, and the 10-K restates the annual periods, so a
single four-for-one is detected in three separate filings months apart. Counting
each detection as its own action multiplies a pre-split fact by four three times.
Nvidia is the case: two splits, a four-for-one in July 2021 and a ten-for-one in
June 2024, produce six detections, so a naive count applies 4 cubed times 10
cubed. Its fiscal 2022 diluted share count came back as 2,535,000mm against a
true 25,350mm, and its fiscal 2009 second quarter as 35.5 trillion shares.

What each sighting actually pins down is an **interval**, not a date. A period
whose value moves between a filing on A and the next on B says the split took
effect somewhere in `(A, B]`. Every sighting of one split brackets the same day,
so their intervals intersect; two real splits at the same ratio are years apart
and cannot. Sightings of one ratio are therefore clustered by intersecting
interval, and each cluster is one action dated at the earliest filing that showed
the new units. Arista is the case that proves the rule has to be more than "same
ratio, merge": two genuine four-for-one splits three years apart across six
filings, which a same-ratio merge would fuse into one and leave every Arista
share count before November 2021 at a quarter of what it should be.

**The first contested judgment: split units are deliberately not knowledge
dated.** Every other fact in this package is filtered on its filing date and a
split looks like it should be no different. It is not, and the reason is
arithmetic rather than principle.

**The case for knowledge dating it.** A split that has not been announced has not
happened. A run pinned to March 2022 that reads Nvidia's June 2024 split off the
2024 filings is reading the future, which is exactly what §11 exists to prevent,
and the resulting share count is one nobody could have quoted on the day.

**The case against, which is the one the code takes.** A price vendor restates its
*whole* history for a split. Nvidia's close for 31 March 2022 comes back as 27.29
rather than the 272.86 that actually printed that day. So a share count left on
the basis of its own day, paired with a price already divided by ten, gives a
market capitalisation out by a factor of ten. **Both sides of a ratio have to be
quoted in the same unit. Only the facts have to be point in time.** A split
carries no information about value either: it multiplies the count and divides the
price by the same number, and the product is unchanged.

That the vendor really does restate its history is not assumed, it is measured off
the committed close series. Nvidia split four for one in 2021 and ten for one in
2024, and **the largest single-day move across 2,513 trading days is 1.298.** A
series left on the basis of each day would carry a close-to-close ratio of 0.25 on
the 2021 ex date and 0.10 on the 2024 one, because that is what a split does to a
quoted price. Neither is there. The vendor divided the whole history instead, and
the share count has to be carried into the same units to meet it.

So detection reads the whole fact set, and a test pins the property in the form
that matters: the same quarter reads the same however the client is pinned. That
is what "unit, not information" means in practice.

**The second contested judgment: dating a split inside its bracket, and the
residual defect.** The bracket is not always tight. The true ex date lies
somewhere inside `(lo, hi]`, and **neither end of it is the ex date.** Where a
filing falls between the two ends it reports whichever basis was current when it
was made, and the threshold can land on the wrong side of it.

Dating at `hi`, the earliest filing showing new units, is what the code does, and
Palo Alto is what it costs. Palo Alto split three for one in September 2022 and
the first filing to restate a comparative by three is dated May 2023, eight months
later, so the 10-Q filed in November 2022 sits inside the bracket. That 10-Q had
already reported post-split shares, 292.9mm for the quarter, and gets multiplied
by three a second time.

Dating at `lo` instead is not a fix, it is a trade, and the trade was measured
rather than argued: across every splitter in the fixtures it **repairs Palo Alto's
three affected periods and breaks five of CrowdStrike's**, whose own bracket is a
year wide. There is no date rule that is right for both. The real fix anchors each
period on its own latest filing and reads the restatement ratio off the filer
rather than deciding from a date at all, which is what
`ml.warranted.share_basis_factor` does across two fact sets.

**Nothing consumes an affected figure silently, and that is the part that makes the
defect acceptable.** `share_basis_factor` compares a fact set pinned to a date
against a current one and takes the ratio between the same fiscal period's diluted
share count in each. Where the units have been normalised correctly that ratio is
exactly 1.0. Where they have not, as at Palo Alto, it comes back as a third, which
is **not a product of any split Palo Alto declared** (3 then 2, admitting 1, 2 and
6), so it is returned unsnapped with a note saying the price and the share count
may be on different bases, and the caller drops the row with the reason attached
rather than training on it. A rounding restatement is separated from a corporate
action by a two percent tolerance, because share counts are reported to thousands
and a real ratio was never exactly the split ratio.

The cost is a row rather than a market capitalisation that is wrong by three times
while looking entirely ordinary. That is the same trade §2.2 makes about a wrong
zero and §7 makes about a not-meaningful multiple.

**What is still not fixed, and where.** The unit normalisation lives in
`CompanyFacts`, so every consumer of `facts()` gets it. The refusal built on
`share_basis_factor` lives in `ml.warranted`, where that module builds its panel,
and has deliberately not been lifted into `ev_bridge`. It belongs there, because
`backtest.py` has the same exposure on any date inside a bracket. Moving it changes
the signature of a function the whole engine calls, which is a decision for the
repository owner rather than one to take inside a model.

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

**Revenue growth fades on a straight line from `dcf.revenue_growth_start` to
`dcf.revenue_growth_terminal`, and nothing about that line is measured.** It is a
typed schedule, and it moves the answer more than the discount rate, the margin
path and the terminal multiple put together. §15.3 measures what growth actually
does in the filings and finds the typed schedule fades less than half as fast.
The fitted path reaches a valuation only through an explicit `growth_path`
argument and only with `assumptions.ml.forecast.enabled` set, so a run without
one is byte for byte what it was.

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

**Who is in the set is the user's judgment, and this whole section is downstream
of it.** `comps.peers` is a hand-written list, so every percentile above is a
quotation from companies somebody chose. That has always been the largest unstated
assumption in a comp table. §15.1 is the attempt to measure it rather than concede
it, against labels the filers themselves disclosed, and it reports what a learned
comp set is worth on a name the model has never seen before: NDCG@10 of 0.415,
which is +0.279 over ranking candidates by how often anyone names them. That is
better than every alternative tested and it is still not a comp set an analyst
should take unread.

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

**The same idea, fitted across a panel rather than a comp set, is §15.2.** Two
thousand company-quarters buy the degrees of freedom this refusal exists to
protect, and they introduce four traps a ten-name cross-section never meets. That
section also reports the first measurement of how often the refusal above
actually fires on real data: on 45% of sub-vertical cross-sections in the TMT
universe.

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

Every fitted model in §15 inherits this, and each one inherits it by a different
route and in a different direction. §14.4 tabulates the five, with the measured
size wherever it could be measured.

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

This module reports the overlap and does not correct the standard error for it.
§14.6 does correct it, on a Newey-West kernel with the lag matched to the
overlap, and reports by how much the correction itself falls short. It also
measures the mechanism, which turns out not to be the one the folklore states:
overlapping windows alone do not inflate a t-statistic, and overlapping windows
plus a persistent score do.

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

---

## 12. The TMT layer

Sections 1 to 11 describe an engine that would value any US filer. This section
describes the part that only makes sense for technology, media and
telecommunications, and the reason it exists is the first sentence of the
taxonomy module: TMT is three businesses, not one. A tower REIT and a games
publisher share an acronym and nothing else. Pooling them produces a comp set
whose median is arithmetic rather than evidence.

### 12.1 The universe and the eleven sub-verticals

`tmt.taxonomy` admits a candidate to the universe and assigns it to one of
eleven buckets: infrastructure software, application software, internet,
semiconductors, hardware, IT services, payments, media and entertainment,
telecom, towers and fibre, gaming.

Two of those splits are the ones that move a multiple most, and they are worth
stating because a reader will otherwise take them for pedantry.

**Infrastructure against application software.** Infrastructure software is
consumption priced and its gross margin is constrained by the cloud bill
underneath it. Application software is seat priced and grows by landing and
expanding. They do not trade at the same EV/Revenue for the same growth rate,
and pooling them is the commonest way a software comp set goes quietly wrong.

**Towers and fibre against telecom.** A tower company is a landlord with
escalating multi-decade leases and three tenants. A carrier is an operating
business with churn, spectrum and capex per home passed. One is quoted on AFFO
and lease-up, the other on EBITDA and subscriber economics.

**SIC is the spine, because it comes from a filing.** The Commission assigns
every registrant one code and publishes it in the submissions payload. It is the
only industry label anywhere in this pipeline that is not bought from a vendor,
and it carries a date. The map from code to sub-vertical was not written from
intuition: it was built against 105 real submissions payloads, and every code in
it is either annotated with the filers actually observed under it or declared as
resting on the code definition alone. A test enforces that partition, so the
comments cannot drift away from the register.

**SIC is also not sufficient, and the register says so loudly.** The codes were
written for a manufacturing economy and they have not been revised for the one
this engine covers:

| Filer | Code | What the code says | What it is |
|---|---|---|---|
| Palo Alto Networks, Fortinet | 3577 | computer peripherals | infrastructure software |
| Zscaler | 7371 | offshore programming services | infrastructure software |
| IBM | 3570 | computer and office equipment | IT services |
| Qualcomm | 3663 | broadcasting equipment | semiconductors |
| Warner Bros Discovery, Roku | 4841 | cable access | media |
| Workday | 7374 | data processing bureau | application software |
| American Tower, Equinix | 6798 | REIT | towers and fibre |
| Amazon | 5961 | mail-order catalogue | internet |
| Netflix | 7841 | video tape rental | media |

**Treatment.** Classification runs the code first, scores keyword evidence over
the business description second, and reconciles the two. Evidence overturns a
mapped code only on a margin over the runner-up; the margin required is lower
for the codes demonstrably holding several industries at once; and the source
string names the answer that lost, so a reader can see what was overruled. An
exact tie is reported as a tie with every tied candidate named, never broken by
dictionary order. An override starts out *less* confident than accepting the
code, because overturning the registrant's own filing is the larger claim.

**There is no SIC code for payments.** Visa, Mastercard, PayPal, FIS and Global
Payments all file under 7389, the services catch-all, alongside Accenture, Uber,
eBay and DoorDash. Toast files under 7374 with Workday and DXC. Mapping either
code to payments would be right about one filer and wrong about four, so the map
has no payments entry at all and every payments name is admitted on evidence or
on a stated human prior.

**The first filing is the wrong admission date.** A CIK exists from the first
piece of paper filed under it, and for a venture-backed company that is a Form D
years before the listing. Roblox's CIK carries submissions from 2005 against a
2021 IPO. Admission therefore keys on the first periodic report, which only a
reporting company files.

**Survivorship is left visible rather than tidied away.** The SEC ticker file is
today's list. Five seed names no longer resolve against it and are deliberately
kept, recorded as unresolved with the reason. The instructive one is Fiserv,
which is still listed and still fails, because the file carries it under its
pre-2023 symbol. Worse than absence, symbols get reused: PARA now resolves to an
unrelated small cap because Paramount files under a different one. Every record
therefore carries its CIK and the symbol is only a label. This is §11.3
appearing one layer up, for the same reason.

### 12.2 Segment economics, and the double-counting trap

Segment detail is tagged under `StatementBusinessSegmentsAxis` and is invisible
to `companyfacts`, which publishes undimensioned facts only (§2.6). It is read
out of the instance document instead.

Disney's fiscal 2025 is why it matters. Experiences earns a 27.6% operating
margin against 11.0% in Entertainment, and the 18.6% company average describes
neither. A single EV/EBITDA struck on that average prices a theme park like a
streaming service.

**The reconciliation is the control.** Segment revenue must sum to consolidated
revenue, over the same period and from the same concept. Where it does not, the
gap is a named residual rather than something spread quietly across the
segments. Disney's three segments sum to 96,294 against consolidated revenue of
94,425, and the 1,869 difference is the intersegment revenue the filer
eliminates. The stronger control is whether the residual is *explained*: it is
matched against the reconciling items the filer actually tags, on magnitude
rather than sign, because filers write intersegment revenue both ways. A gap
that matches nothing disclosed means a member was missed, and that gets
different words from a gap that is the eliminations working.

**Filers tag the same dollar several times over, on different cuts.** Disney
tags Entertainment revenue once whole at 42,466, again split across three
geographies, and again third-party against intersegment on the product axis.
Summing everything under the segment axis counts the same revenue three times.
Any fact carrying more than one slicing axis is a cell of a matrix rather than a
total, so it joins neither the segment table nor the geography table.
`ConsolidationItemsAxis` is treated as a qualifier rather than a slice, which is
what makes the geography table readable at all.

Three smaller rules, each a real failure in the committed filing. Periods never
mix: flows come from one annual duration and assets from the instant at that
same period end. A figure tagged only across two axes is left blank with the
reason rather than summed back up, which is how Experiences depreciation is
handled. And segment assets absent is a disclosure fact rather than a zero,
because ASC 280 requires them only where the chief operating decision maker
reviews them.

**One trap that decides a margin.** The undimensioned `OperatingIncomeLoss`
inside a segment note is often *total segment operating income* rather than GAAP
operating income. Disney's income statement carries no operating-income subtotal
at all, so the 17,551 sitting undimensioned is the segment sum, and reading it
as the consolidated figure understates corporate cost by the whole of it.

### 12.3 Sum of the parts

`herfindahl` is struck on shares of segment revenue so the shares sum to one and
the index keeps its meaning. Disney at 0.37 is three businesses on three
multiples, and a sum of the parts values it better than any single multiple can.

**The multiple is an argument, not a number.** Each segment is valued on a
`(metric, multiple, source)` triple supplied by the caller, and the source
sentence travels all the way to the printed row. There is no default multiple: a
peer set for cable has no business living in a config file shared with a
software DCF. A multiple with a blank source is refused, and so is a segment
with no multiple at all, because dropping a segment from the sum values that
business at zero without anyone deciding to.

**Corporate cost is where most sum-of-the-parts analyses go wrong.** Segment
operating income under ASC 280 is struck *before* unallocated corporate expense,
so adding the segments up and stopping values the holding company's own overhead
at zero. It is capitalised as a negative and subtracted. Absent a multiple from
the caller it is capitalised at the enterprise-value-weighted average of the
segment *earnings* multiples, because the overhead is an earnings charge on the
businesses it supports and averaging an EV/Revenue against an EV/EBITDA would
average different denominators. Where no segment carries an earnings multiple
there is nothing to average and the engine asks for one rather than inventing
it. Where no corporate cost is supplied at all the total still prints and the
flags say it is overstated.

**The conglomerate discount defaults to zero, and both camps appear in the notes
on every run.** The diversification literature, Lang and Stulz through Berger
and Ofek, finds something like 5 to 15 percent. Against that, a discount assumed
rather than observed is a fudge factor that can be tuned until the sum of the
parts agrees with whatever answer was wanted. Where it is set it applies to the
enterprise value *after* the corporate deduction, and the note says why that is
still not clean: those studies measured market value against segment sums that
mostly did not deduct capitalised overhead, so part of the same cost is charged
twice.

**Loss-making segments.** Negative EBITDA times a positive multiple claims the
business is a liability, which is almost never true of something that can be
closed or sold. The default refuses with `NotMeaningfulError`; the caller values
the segment on revenue instead, or asks explicitly for nil, and nil is recorded
on the row as an assumption rather than a measurement.

**The gap is the point.** The sum-of-the-parts enterprise value is reported
against the priced bridge in dollars, in percent and per share, and the equity
walk is the engine's own, so a sum-of-the-parts price per share and a DCF price
per share are reconcilable rather than merely similar. A gap wider than 25% is
flagged as a probable wrong multiple or a missing corporate line *before* it is
read as a mispricing.

What it does not do is listed rather than glossed. Segment profit is not
reconciled down to consolidated operating income, so a corporate cost that does
not actually close the footnote reconciliation is accepted as given. Nothing
checks that the segment period matches the consolidated period. There is no tax
leakage on a break-up, which is often what kills the pitch. And segment values
are struck on a post-rent convention because no segment carries an allocated
lease liability, so under the lease-inclusive bridge convention (§4.1) the gap
is measured across two conventions and is flagged rather than corrected.

### 12.4 The operating metrics a sector is actually priced on

**The Rule of 40 is one rule and three different numbers**, and which margin goes
in is the whole argument, so all three are computed rather than one being
chosen. Datadog, trailing twelve months to 30 June 2026:

| Variant | Points |
|---|---:|
| Growth plus free cash flow margin | 61.3 |
| Growth plus EBITDA margin | 33.5 |
| Growth plus operating margin | 31.9 |

One set of filings, a 29.4 point spread, and the company passes on one variant
and fails on two. The free cash flow variant is what growth investors quote, the
EBITDA variant is what §7 already reports because it comes off filed lines every
company has, and the operating variant is the strictest. The spread is printed
beside them and a note fires when the variants straddle 40.

The reason for the spread is stock compensation, and it is worth following. Free
cash flow margin is cash from operations less capex, over revenue, and stock
compensation is added back inside cash from operations. That is exactly why the
cash variant flatters a heavy issuer against the earnings variants. The same
margin with stock compensation reversed out sits beside it, and the difference
between the two is the stock compensation share of revenue by construction,
which is asserted as an identity rather than left to be noticed. At Datadog that
share is 20.7% of revenue and it is most of the distance between the cash and
earnings variants. The argument in §6.1 about what free cash flow means is the
same argument, arriving through a different door.

Two more software metrics, each with the definition that produced it attached to
the number. The magic number is computed on twelve months against twelve months,
so it needs no annualising, and the definition names the noisier quarterly
variant the sell side quotes. Customer acquisition payback runs off gross profit
rather than revenue, because a dollar of revenue that costs forty cents to serve
does not pay back a dollar of selling cost.

**Media.** Cash content spend against content amortisation, because the gap is
the sector's single most important adjustment: a company amortising less than it
spends is capitalising a growing library, and reported earnings flatter the cash
reality by the difference. The normalised statements cannot say whether content
amortisation sat inside the D&A added back to reach EBITDA, because that is a
tagging decision the filer made. Both cash-adjusted figures are therefore shown,
each naming the tagging it assumes, and where content amortisation exceeds total
D&A the ambiguity is settled arithmetically and the pack says which case
applies. Where it cannot be settled it is flagged rather than guessed.

**Telecom, towers and fibre.** Capex intensity, EBITDA less capex, and the
multiple on it, with the headline EV/EBITDA beside it so the difference can be
read. Pricing a network operator on EV/EBITDA treats the spending that keeps the
network competitive as optional, which is the whole reason the sector quotes the
other multiple. Towers get an AFFO-style proxy and the note that they are REITs
quoted on AFFO multiples rather than EV/EBITDA. Total capex is never substituted
for maintenance capex: a tower REIT's capex is mostly discretionary augmentation
and land, and the substitution would understate AFFO by the whole growth
programme. Without the disclosure the proxy is absent and the flag says why.

Where an EV bridge is supplied the metric pack takes its earnings denominator
from `EVBridge.multiple_denominator()`, so a lease-inclusive enterprise value
divides EBITDAR rather than a post-rent EBITDA. In towers and cable that trap is
worth more than a turn.

**An unknown sub-vertical raises rather than falling back to the software pack**,
because a semiconductor company scored on the Rule of 40 is a worse answer than
no answer.

**A known defect between two modules, stated rather than hidden.** The metric
packs route on a sub-vertical string whose vocabulary is `software, saas,
internet, media, streaming, entertainment, telecom, wireless, cable, fiber,
towers`. The taxonomy of §12.1 speaks in `infrastructure_software,
application_software, internet, semiconductors, hardware, it_services, payments,
media_entertainment, telecom, towers_fiber, gaming`. Exactly two values appear in
both, so handed the other nine the pack raises `ConfigError` and cannot be run
across the engine's own universe at all. This was found by measurement rather
than by reading, and there is a test that pins it as a tripwire. Reconciling the
two vocabularies is a design decision about which pack a bucket belongs to, and
it is deliberately not guessed at inside a model.

### 12.5 Operating metrics are not in the taxonomy, and that is the finding

The metrics TMT is priced on are not us-gaap concepts. Four real instance
documents, one per business model, were pruned into fixtures, and what they tag
decided the design:

| Filing | What it actually tags |
|---|---|
| Datadog | remaining performance obligation of 3,471.4mm, and nothing else the sector quotes |
| Cloudflare | the same, plus the RPO percentage due within a year |
| Netflix | additions to streaming content assets and their amortisation, both filer extensions |
| T-Mobile | three different undimensioned RPO values at one date: 694mm, 1,000mm and 2,300mm |

Not one of the four tags annual recurring revenue, a customer count, retention,
subscribers, average revenue per user or churn. **Netflix, whose entire equity
story is paid memberships, does not tag a membership count.** A parametrised test
asserts all nine absences across all four filings, so if a later taxonomy release
or a filer changes that, the test turns red and the module's central claim gets
rewritten rather than quietly surviving.

What the filings do carry, in quantity, is near misses:
`NumberOfCustomerAccountsImpacted`, `NumberofCustomerClasses`,
`NumberOfCustomerCategories`, `NumberOfSegmentManagers`,
`ProvisionForDoubtfulAccountsExcludingPayAsYouGoCustomers`. A substring search
for "customer" takes every one of them. The concept ladders are therefore
anchored full-match patterns, and a test asserts the traps are really in the
files before asserting they stay out of the results.

**Treatment, and it is mostly refusal.** A tagged value and a text value that
disagree are both kept, the tagged one under the metric name and the text one
beside it, with the gap flagged. Several undimensioned values for one element at
one date are refused outright, which is the T-Mobile case above: picking the
largest, the first or the sum would each be a guess dressed as a number. The row
survives for audit, carrying all three figures in its notes and a flag, and is
withheld from the machine-readable view a model or a table would read, so an
auditor sees the conflict and a consumer sees nothing at all. A
retention figure of 11500% keeps its value and loses its confidence, with a note
that the percent sign was probably misread, because dividing by a hundred would
make three different errors look like one clean number. A rate with no percent
sign, a dollar figure below the floor with no scale word, a fractional count:
each is reported at zero confidence with the reason attached and withheld from
the machine-readable output. **A filing that stated a metric the parser could not
read is a different finding from a filing that did not state it**, and the two
are never merged.

Hedges are classified rather than merely noted. "Over 120%" is a bound wrong in a
known direction; "approximately 108%" is the company's own estimate. Both drop to
0.4 confidence and the reader is told which kind it was.

**Billings uses total deferred revenue, not the current part.** Billings is
revenue plus the change in deferred revenue, and a multi-year prepayment is
invoiced value whichever side of twelve months it is delivered on. Datadog, on
the committed facts:

```
revenue                            3,966.725
deferred revenue at 2026-06-30     1,286.690 + 50.860 = 1,337.550
deferred revenue at 2025-06-30       966.442 + 29.866 =   996.308
billings = 3,966.725 + 1,337.550 - 996.308           = 4,307.967
```

The current-only convention runs 21.0mm lower, which a test pins.

**Sub-verticals with no pack search for nothing.** Semiconductors, hardware and
IT services get an empty set and a note. Searching a chip maker's filing for net
revenue retention produces noise and calls it coverage.

### 12.6 Precedent transactions

**Precedents are not trading comps, and the difference is a number.** A precedent
price contains a control premium and whatever synergies the buyer underwrote; a
trading multiple contains neither. The historical TMT control premium runs
roughly 25 to 40 percent, so a precedent multiple sits above a trading multiple
for the same company *by construction*. Reading the gap as evidence that the
company is cheap today is reading the control premium twice. That sentence is in
the notes of every precedent set rather than in a footnote here.

**Finding the deals.** The scan asks EDGAR for 8-K, PREM14A, DEFM14A, SC 14D9 and
S-4. Rule 425 and DEFA14A are deliberately excluded from that list, because both
are also used for routine annual meeting solicitations and neither establishes
that a deal exists. They are read afterwards, once a deal is established, and
only to date it.

Four traps, each a real document in the fixtures:

- **Item 1.01 is not a merger.** Splunk's June 2021 8-K is a convertible note
  sale and contains "an initial conversion price of $160.00 per share".
  Extraction requires merger agreement language *and* share conversion language
  before it will look for a number.
- **The filer is not always the target.** Zendesk's own December 2021 S-4 says
  each share converts into 0.225 of a Zendesk share, because Zendesk was buying
  Momentive. The extractor reads whose stock is being converted and discards the
  acquirer's side of somebody else's deal.
- **Par value is not an offer price.** Every merger 8-K in the set says "par
  value $0.001 per share" a few words from the real consideration.
- **A price that cannot be fixed is not fixed.** Iridium pays $27.00 in cash plus
  a collared number of Rocket Lab shares, one ratio below a reference price, a
  floating ratio inside the band and a third above it. No exchange ratio exists
  at announcement. The transaction is kept, the cash leg is kept, and the offer
  price is absent with the reason recorded. Reporting the $27.00 cash leg as the
  offer price would be the worst kind of wrong number: a real figure in the wrong
  role.

**The unaffected price, and the judgment stated.** The headline is the last close
strictly before the announcement, because it is the figure any reader can
reconstruct from one quote and because a longer window imports a month of sector
news into a number meant to isolate the deal. The mean of the thirty calendar
days before is computed beside it and both are reported. Where the two premia
differ by more than five points the deal is flagged, because that gap is the
market having already moved. Roku is the case the fixtures carry: the stock rose
twenty percent on Friday 12 June 2026, the agreement was signed over that
weekend and the 8-K came on the Monday. Premium to the last close, 11.3 percent.
Premium to the month before, 27.0 percent. Only one of those is a control
premium. The thirty-day figure is named an unweighted mean and not a VWAP,
because the price layer carries closes and no volume.

**The announcement date is neither the signing date nor the filing date.** An 8-K
reports the event date and can be filed four business days later, and a signing
after the close is announced the next morning. The date used is the earliest
filing across the announcement cluster, floored at the agreement date the
document states in words. Splunk: agreement 20 September 2023, announced 21
September, and the resulting unaffected close is the one the street quoted.

Target financials are rebuilt through a `CompanyFacts` pinned to the announcement
date, so the trailing twelve months are what a bidder could see rather than the
restatements that arrived afterwards, and enterprise value runs through the same
bridge as everything else so the target's assumed net debt is inside every
multiple. Statistics carry `n` beside every figure and refuse below five deals,
per column as well as per sub-vertical.

**What does not work, recorded rather than filled.** Delisted targets have no
price history: the quote source answers "Symbol not exists" for Splunk, Mandiant,
Zendesk and Slack, so a live run over completed deals returns multiples and no
premium, flagged rather than zero. Real premia over closed deals need the closes
supplied through a CSV source. Every priced deal in the fixture set is a pending
2026 transaction, which is exactly why its ticker still trades. That is §11.3
again, and it is the constraint that shapes most of §13 and §15.

---

## 13. What a fitted model is allowed to learn from

Everything below this line is fitted rather than assumed, and the engine's
cardinal rule does not relax for it. A model that cannot be traced back to
filings does not get printed next to numbers that can. Three things make that
possible and they are defined once, in `ml.protocol`, so that no model can skip
one: a computed baseline, a model card, and a knowledge date on every
observation.

The order of this section and the next is deliberate. What the labels are decides
what a score can mean, and how the score was computed decides whether to believe
it. Both come before any result.

### 13.1 Where the labels come from, and why that is the whole idea

**The weakest thing a learned peer model can be built on is the author's own
opinion of which companies are comparable.** Hand-label a few thousand pairs, fit
a model, backtest it against the same labels, and the result is a restatement of
the prior with a confidence interval attached. The backtest is a tautology: it
measures how well the model reproduces the person who wrote it. A high score
there is evidence of nothing except that the fitting worked.

So the labels here are taken out of the filings. A registrant's DEF 14A proxy
discloses the companies its compensation committee benchmarked pay against,
chosen by an independent consultant under stated revenue and market
capitalisation bands. That is a **dated, auditable assertion of comparability by
people with something to lose from getting it wrong**, and it is the only peer set
in the public record that carries a date.

**What a label means, exactly.** A pair `(filer, peer, fiscal_year)` means the
filer told the SEC that its board used that peer when setting pay for that fiscal
year. Nothing more.

**And what it does not mean.** It does not mean the two trade on the same
multiple. A compensation peer group is selected partly for competition for
executive talent, so it skews toward companies of similar size and toward the
filer's own labour market rather than toward its product market. That is not a
defect to be corrected away; it is the price of having a label with a date on it,
and it shows up in the output. Apple's and Comcast's disclosed tables contain
Johnson and Johnson, Merck, Procter and Gamble and Honeywell. Any model fitted on
these labels inherits that, and it is on the model card rather than buried here.
The selection criteria the proxy states are captured and carried alongside every
group, on the reasoning that a reader who knows the bands knows what the label is
worth.

**The parse is fiddly and the failures are silent, which is why it is documented
at length.** The rendered proxy is HTML and the peer list is a table. Stripping
markup joins the cells with no separator, so the list arrives as one run of proper
nouns:

```
AtlassianHubSpotThe Trade DeskCloudflareMongoDBVeeva SystemsCrowdStrike...
```

The obvious approach lowercases the blob, strips corporate suffix words and runs
a greedy dictionary match. It produces confident garbage, in two distinct ways.

*The stranded suffix.* Stripping suffixes from the whole blob at once lets a
suffix word from one name bridge into the next, and the leftover letters spell
other companies. `Veeva Systems` followed by `CrowdStrike` leaves the unconsumed
word `systems`, inside which `stem` is a real ticker. Three rules kill it: a match
may only begin where a name can begin, which is a capital letter or a delimiter,
and `stem` begins at the lowercase `t` inside `Systems`; every candidate span is
looked up both as written and suffix-stripped, so `Veeva Systems` matches its
registrant across the full length instead of leaving a tail; and the longest match
at a boundary wins.

*The bridge that resolves*, which is worse, because it produces a confident wrong
reading rather than a failure. `NV` and `SA` open NVIDIA and SAP SE and are
themselves corporate suffix words, so `Netflix, Inc.NV` normalises to `netflix`
and matches Netflix across fifteen characters where the correct read matches
thirteen. Longest-match prefers it and eats the first two letters of the next
company. Nothing looks wrong afterwards: every ticker returned is real and only
the count is quietly short. The resolution is that a longer span landing on the
same registrant earned nothing by being longer, so the shorter reading is taken,
but only when a registrant actually starts at the seam that opens. That condition
is what stops `Veeva Systems` from shrinking to `Veeva`.

**Ambiguity is never guessed.** Roughly eight thousand registrant names normalise
into a space where collisions happen. A colliding span prefers a candidate already
named elsewhere in the same proxy, then one inside the universe being collected,
and a tie that survives both is recorded as unresolved with both candidates named.
A missing label costs one training row. A wrong one teaches the model a
relationship nobody asserted.

**A group that is not the right shape is not a small group.** Confidence is the
share of spans that resolved, and it is zeroed when the count falls outside 8 to
30 names. Consultants build these tables to give a defensible median, which needs
enough names to be stable and few enough to stay comparable. A parse returning
three names or eighty has found a fragment or run past the end of the table.
Those groups are returned for audit and never become training pairs.

**Pairs are symmetric and year-stamped**, so a walk-forward split can respect
point in time. Training on the union across years would let a 2025 relationship
justify a 2021 ranking. Nothing in the module is random, every tie is broken by a
stated rule, and every list is sorted, so three separate processes produce a
byte-identical digest.

**Two bugs the measurement found, both of which had been silently costing
labels.**

*A symbol tie-break was handing labels to debentures.* Choosing "shortest symbol,
then alphabetical" among the symbols of one CIK rests on the belief that two
symbols on one CIK are two share classes. They are often not. Comcast files its
common stock and an exchangeable debenture under one CIK, both titled
`COMCAST CORP`, and the shorter symbol is the debenture. Every proxy naming
Comcast therefore produced a label pointing at a debt security, and because no
equity universe contains it the pair was then dropped as a peer outside the
universe. One of the largest filers in the sector disappeared from the training
set and nothing said so. Prudential and DTE Energy lost their common stock the
same way. Measured across the live ticker file, the rule disagreed with the
Commission's own listing order for **224 of the 1,441 registrants carrying more
than one symbol**. The Commission lists the primary symbol first, and that is now
what is taken. The cost is that Alphabet resolves to one arbitrary class rather
than the other, which was always arbitrary; what changed materially is the
debenture case, and Comcast now resolves in 21 groups.

*The state of incorporation was being read as part of the name.* The Commission
appends it to a registrant's title: `APPLIED MATERIALS INC /DE`, `QUALCOMM INC/DE`,
`CORNING INC /NY`. 291 of 10,407 titles carry one. Tokenising the marker produced
a trailing `de`, the suffix stripper stopped on it before reaching `inc`, and the
core form came out as something the proxy's "Applied Materials" can never meet. It
cannot be fixed by adding `de` to the suffix list, because those are ordinary word
fragments and stripping them would eat the tail of a real name; the marker is
removed by its slash, at the end of the string, and nowhere else. Measured cost:
**121 of the 1,205 unresolved spans**, across 27 names. Applied Materials went from
0 namings to 44, Qualcomm from 0 to 66, Corning from 0 to 32.

Both bugs carry a regression test that fails before the fix and passes after, and
the parser version is bumped so cached parses are re-read rather than trusted.

**The yield, measured rather than extrapolated.** Across the 110-name seed
universe: **320 usable groups from 75 filers, 8,638 symmetric year-stamped pairs**.
Thirty-five of the 110 yield nothing at all, either because they file no
compensation peer table or because the parse cannot find one. Of those pairs,
**2,534 directed pairs survive** the requirement that both legs carry a feature row
and a business description at the pair's own panel date. That last number, not the
first, is what a model actually trains on, and the gap between them is the honest
size of the label set.

**The largest loss in the label pipeline is survivorship, and it happens one step
earlier than anyone looks for it.** Measured directly: *zero* named peers are
missing from the SEC ticker file. Not one. The reason is that resolution runs
against a snapshot of currently registered filers, so a deregistered company
cannot be resolved at all and never becomes a named peer. It lands in the
unresolved list instead. Reading that list is reading the 2021 to 2026 technology
M&A wave:

```
Splunk 51, Juniper Networks 28, Coupa Software 17, Electronic Arts 15,
Maxim Integrated 12, Xilinx 11, Activision Blizzard 10, Zendesk 10,
VMware 8, Slack 8, Citrix 7, Alteryx 7
```

Every one acquired or taken private inside the sample. **18.4% of all disclosed
peer spans cannot be resolved to a ticker**, and after the two fixes above the
residue is dominated by exactly this. The module's refusal to guess is what makes
the loss auditable instead of silent.

### 13.2 The feature panel, and the three ways point in time is enforced

Fifty features per company per date, all of them ratios, growth rates or explicit
log-scale size terms, grouped so that an ablation can report by block: scale,
growth, margin, efficiency, capital, returns, market and quality. **A model given
raw revenue learns that large companies are large**, which is why nothing enters
in levels except a log.

The column list is a contract. What was computed is compared against it and the
builder raises rather than returning a row of a different width, so a feature
added to the arithmetic and not to the declaration cannot silently change every
design matrix downstream.

**Point in time is enforced three ways rather than assumed.** The builder refuses
a client with no knowledge date and refuses a price series built past the row
date. Then the filing date behind *every figure that actually reached the row* is
walked: the statements' own provenance (§2.7), the trailing twelve month
resolutions made at earlier anchors for growth and margin change, and the last
close on the price series. The test that matters seeds a fact set claiming a
knowledge date it does not honour. It passes every configuration check, produces
entirely ordinary-looking statements, and is caught only by the provenance walk.
That is the case the walk exists for.

**The reporting lag falls out of the construction rather than from a rule.** A row
dated 15 January 2025 for Datadog, a calendar-year filer, anchors on September
2024 and carries 2,536mm of trailing revenue. By 3 March the annual report is on
file and the same row anchors on December and carries 2,684mm. Both are pinned in
tests, so the lag is a different figure and not merely a different label.

**Missing is not zero, and not meaningful is not missing.** A missing gross margin
filled with zero tells the model the company broke even, which is a specific claim
and usually a false one. Missing values stay missing, the names go into a list,
and a missing-share indicator per group is emitted beside the design matrix, so a
consumer decides what to do and has to say so. Separately, a ratio whose
denominator has the wrong sign is kept apart from a figure that was never
reported: net debt over a negative EBITDA is a negative multiple that sorts to the
conservative end of the column and means the opposite, so it comes back absent
with the reason, in the same spirit as the NM the comp tables print. Winsorization
is the answer to a denominator that is small, never to one whose sign has flipped.

**Winsorization is cross-sectional at each date**, at the 1st and 99th percentiles,
and every value moved is counted by feature and by date. Pooling across dates would
set the bound from the level of whichever era dominates the sample, and a 2021
software cross-section and a 2026 one are not the same distribution. Below twenty
names a 1st percentile falls between the first and second order statistics and the
operation degenerates into pulling the single extreme to its neighbour, so it is
skipped and the skip is reported rather than performed quietly.

**Standardisation will not run without being told which rows to fit on.** Fitting
on the full panel and splitting afterwards moves the test period's mean and
dispersion into the transform, and the resulting out-of-sample score is better
than the model deserves by an amount nobody can estimate afterwards. The leak test
puts ordinary values in the training period and values three orders of magnitude
larger after it, and checks the fitted mean is 2.5 rather than 501. A column with
no variation across the fit rows is centred and left at unit scale, so the
transform stays exactly invertible: dividing by zero would put infinities in the
design matrix and dividing by an epsilon would put 1e15 there.

**A company that cannot be built at a date is a failed row carrying the reason**,
never dropped and never filled in. A lookahead error is *not* caught there, because
it says the panel is not evidence rather than costing one row of it.

### 13.3 The text tower: Item 1, not the 10-K

**The document is Item 1.** Datadog's annual report is 405,446 characters and its
Item 1 is 38,928 of them. The rest is risk factors, financial statements and
exhibits written to a template every filer in the country shares, so a term
frequency model over all of it measures securities counsel rather than the
business.

Splitting a 10-K into its Items is harder than it sounds and both obvious fixes
fail on the same filing. Every 10-K names each Item twice, once in the contents
table and once at the section, so a first-match split returns a list of section
names and page numbers as the entire business description, and nothing downstream
notices because a list of section names is still text. Taking the last occurrence
instead fails because a 10-K cross-references itself: in Datadog's filing the
final "Item 1. Business" sits at character 85,583, inside the risk factors, in the
sentence "described under Part I - Item 1. Business in this Annual Report".
Last-match hands back the back half of the document.

Three rules, each measured on six real 10-Ks rather than assumed. **Page numbers,
not density**: a contents entry ends in a page number and a section ends in a
sentence. Clustering alone is not enough and that is not theoretical, because the
Items at the back of a 10-K that answer by incorporating the proxy sit a median of
159 to 205 characters apart, inside any window wide enough to catch a contents
table. **Sentence boundary**: a real header is preceded by a full stop, a closing
bracket or a page number once the part label and the repeated running head are set
aside, where a cross-reference is preceded by a lowercase word, a comma, a dash or
an opening quote. One test rejects all eight of Datadog's self-references while
accepting all twenty-three of its real headers. **Order**: Items run in the
sequence Regulation S-K prescribes, so the scan walks that sequence and a
reference pointing forward cannot claim a section the scan has not reached.

**The company's own name is the leak.** Left in, the highest weighted term in a
document is the filer's own name, the nearest neighbour of any company becomes
whoever happens to mention it, and similarity collapses into a lookup that scores
well and is worthless for the question being asked. Names and tickers are
therefore stripped from their own documents. That first requires un-gluing words
that markup stripping ran together, because a name with no word boundary in front
of it survives every attempt to match it: Datadog's Item 1 opens as one run of
characters. Generic industry words inside a registered name are kept on purpose,
since removing "Communications" from Verizon and leaving it in T-Mobile would
manufacture the very difference the step exists to prevent.

**A zero vector is never returned.** A document sharing no vocabulary with the
corpus raises rather than scoring cosine zero against everything, because zero
reads as "resembles nothing" when the truth is "was not measured".

**One point-in-time bug found while testing, of a shape worth remembering.** The
filing index is returned newest first and was truncated at its limit *before* the
as-of filter ran, so reading only the eight most recent 10-Ks made a fold dated
2015 conclude that a filer which has filed every February since 2008 had no 10-K
on record. Nothing about that failure looks like a failure downstream: the company
is simply missing from the corpus, and coverage thins the further back the
backtest goes, which is the direction that flatters a result.

### 13.4 The fitting primitives, and why they are written out by hand

Layers, losses, Adam and a training loop, in numpy, with analytic gradients and
no framework underneath. Three reasons, in order of weight.

**A gradient a reader can check is worth more beside a valuation than one they
cannot.** The engine's claim is that every number traces to a filing or to a
stated assumption. A model whose fitting procedure cannot be inspected weakens
that claim even when the fit is correct. Here the forward pass and the derivative
of each layer sit twenty lines apart, and a gradient check differences every one
of them against central finite differences. It runs in the suite over every layer
and every loss at a bar of 1e-6, measured errors come in around 1e-9, and two
tests break a backward pass on purpose to show the check would catch it rather
than passing everything put in front of it. That headroom is why the module is
float64 throughout: in single precision the cancellation term of a central
difference alone is about 1e-2 and no useful bar exists.

**The models this dataset justifies are small.** A 462,080 parameter stack over
20,000 rows trains at 0.75 seconds an epoch. Throughput was never the binding
constraint, and buying it with a dependency nobody can audit would have been
paying for the wrong thing.

**And the framework does not fit on the machine.** That decides nothing the first
two had not already decided.

Two details that are easy to get wrong and are therefore pinned by tests. Adam's
weight decay is **decoupled**: L2 added into the gradient passes through the
division by the root second moment, so a parameter with a large noisy gradient
decays less than one with a quiet gradient, and how much a weight shrinks ends up
decided by the local shape of the loss surface instead of by the strength that was
asked for. And early stopping restores the **best** epoch rather than the last,
which is asserted bit for bit rather than assumed.

InfoNCE returns gradients with respect to all three of its inputs and the
temperature is documented rather than left as a magic number: low values sharpen
the distribution so the loss is dominated by the closest negative and the model
spends its capacity on hard cases, high values flatten it and it learns coarse
structure and stops. For normalised embeddings 0.05 to 0.1 is the working band,
and a test measures that claim by planting a hard negative and reading off its
share of the negative gradient at two temperatures.

Everything is seeded from `assumptions.ml.random_seed` and two runs at one seed
produce bitwise identical parameters, verified across separate processes as well
as within one.

---

## 14. How a fitted model is scored

This section is longer than the results it governs, which is the right
proportion. A score is an assertion about the future and the only thing standing
behind it is the protocol that produced it.

### 14.1 No result without a baseline, and the baseline is computed

**An R-squared of 0.4 on revenue growth sounds respectable until you learn that
carrying last year's growth forward scores 0.45**, at which point the model is
worse than doing nothing and should be reported as such. Every `EvalResult`
carries the naive alternative beside the model, names it, and computes
`beat_baseline` rather than asserting it. `verdict()` prints the negative case in
words: *the model does NOT beat the baseline. Use the baseline.*

Where the caller supplies no baseline the harness computes the obvious one and
names it: the training-period mean for a regression, the base rate for a
classifier whose AUC is then 0.5 by construction rather than by assumption, and
for a ranking the expected NDCG of a uniform random ordering. **The random NDCG is
computed in closed form rather than simulated**, so it is exact and there is no
seed to argue about.

One asymmetry is deliberate. A computed baseline that has to peek at the answer
key to exist is named *in sample*, which makes it a harder comparison and
therefore the conservative choice. A baseline that has seen the test window and
still loses is a baseline that lost properly.

The baselines are also chosen to be the ones a reader would actually run instead,
not straw men. The warranted multiple is scored against the existing OLS of §7.1
refit on each date's sub-vertical peers, which is the incumbent in this very
repository. The peer encoder is scored against five alternatives including raw
term-frequency cosine, which is the thing you get for free without any learning at
all. The fade curve is scored against persistence, which is free and, at one year,
unbeatable.

**Fold dispersion is the honest error bar and it is not a significance test.**
`fold_sd` is the standard deviation across folds. `verdict()` compares the lift to
it and says *inside* or *outside* the fold-to-fold noise in those words, because a
lift of 0.02 against a fold standard deviation of 0.09 is not a lift. On a handful
of folds that is an error bar and nothing more, and the module does not pretend
otherwise.

### 14.2 Folds are cut on dates, with an embargo, and never at random

**A random split lets a model train on 2026 and test on 2024**, which is the
cleanest way to manufacture a result nobody can repeat. Every fold trains on
everything up to a cut and tests on the window after it, and the cuts march
forward through the sample.

Blocks are cut on the sorted distinct observation dates rather than on equal
calendar spans, because filings arrive in clusters and equal calendar windows hand
one fold a reporting season and another an empty August.

**The embargo is the detail that separates a real backtest from a plausible one.**
Where the label is a forward return or a forward growth rate, the label of an
observation dated t is not known until t plus the horizon. Train on an observation
dated one month before the test window opens and its label window overlaps that
test window by eleven months out of twelve, so the model is shown most of the
answer before it is asked the question. The fit looks excellent and the failure is
silent.

Training is therefore cut strictly before the test start less the embargo, and the
rows that fall into the gap are counted on the fold, so an embargo that throws away
a third of the sample is visible as a decision rather than as an absence. **A zero
embargo is written into the notes**, so a reader can see the protection was
declined rather than forgotten. It is declined legitimately in one place: the
warranted multiple, whose label is the multiple observed on the observation date
and has no future inside it.

The demonstration is not an argument, it is a measurement. The same prediction
rule is scored twice on identical test windows against a label that is a forward
twelve month average. With the cut sitting against the window it beats a constant
comfortably; with 365 days between them **it loses to the training mean, and
nothing about the model changed**. A parametrised companion shows the error is
between 1.5 and 3 times larger once embargoed on every panel tried, so the effect
is a property of the split rather than one lucky seed.

Embargoes in use: `365 * horizon` days for the fade curve, so three years of
horizon buys three years of gap rather than one; 462 days for the propensity
screen, which is twelve months counted at 31 days each plus the 90 day
announcement gap, and which cost 7,267 observations; and 365 days for the peer
encoder.

**The encoder's embargo is there for a different reason and that reason is worth
stating.** A compensation peer group is known on the day it is filed, so there is
no forward label to overlap. The embargo exists because a filer's next proxy
repeats most of its previous one: without a year of separation the test question
is whether the model can recall a list it was shown eleven months earlier. Even
that is not enough on its own, which is why the encoder additionally reports warm
start and cold start separately (§15.1).

### 14.3 Everything fitted is fitted inside the fold

The transforms are models too. The term-frequency vocabulary, the SVD basis, the
feature mean and standard deviation and the imputation value are all *estimated
from data*, and estimating any of them on the full sample leaves no trace
whatsoever in the output while lifting every fold.

So all four are fitted on the training window and applied forward. This is
enforced twice over in the text path: once by the encoder, and once by the
vectoriser's own refusal to fit a corpus containing a document filed after the fit
date.

**Imputation is the one place the ML layer overrules a sibling, and the overruling
is partial.** The feature store refuses to impute and is right about both
objections: filling a missing gross margin with zero asserts the company broke
even, and filling it with the cross-sectional mean leaks the cross-section. A
neural network still cannot consume a missing value. What is done is to impute to
the *training fold's* mean, which after in-fold standardisation is exactly zero,
and to feed the missing-share indicators alongside. **That answers the leakage
objection completely and the "you are asserting a false fact" objection not at
all**, and nothing does. The encoder is told "this company sits at the training
average on this feature" beside "this share of this feature group was
unavailable", and is left to learn what the pair means together. Every row's
imputed-cell count is carried on the model card.

### 14.4 Survivorship, and which way it cuts for each model

**Survivorship is the bias that flatters every model in this package**, and it
arrives by a different route in each one. The universe is built from companies
that exist today, because the sources delete a company the day it stops trading.
There is no point-in-time ticker file here and no delisting-complete vendor file,
so this is a stated limitation rather than a solved problem. What is done instead
is to measure the size and to name the direction, per model:

| Model | How the survivor bias enters | Direction |
|---|---|---|
| Peer encoder | a deregistered company cannot be resolved from the ticker file, so it never becomes a named peer at all; 18.4% of disclosed spans are lost, dominated by the M&A wave | lands entirely in recall, so every ranker is capped at the same ceiling and the *comparison* stays fair while the *level* is overstated |
| Warranted multiple | five seed names have no price history at any date, including the dates on which they were live and cheap | biased by an amount nobody can estimate |
| Revenue fade | 119 delisted filers are deliberately kept, 44% of the panel; their last observed year grows 9.8% against 15.8% for survivors | dropping them would lift mean forward growth by 0.78 points at one year, 1.34 at two and 1.52 at three, and the bias compounds with horizon, which is the worst possible direction for a fade curve |
| M&A propensity | 77 of 107 deals are on companies that no longer file | not a bias but the destruction of the experiment: a survivor universe has almost no positive class left |
| Value signal | not one name in the panel was acquired or delisted over the whole period | the missing names are takeouts, takeouts earn a premium and skew cheap, so their absence flatters cheapness and the true coefficient is if anything more negative than reported |

Two of those deserve the extra sentence.

**The propensity screen is where survivorship stops being a bias and becomes a
category error.** A company that is acquired stops filing, is struck from the
exchange, and disappears from every list of tickers that exists today. Build the
panel from names listed now and every positive label vanishes with them: the fit
runs on an all-negative sample, reports a flawless in-sample accuracy, and means
nothing at all. The failure is silent, because an all-negative classifier is a
perfectly well-behaved object. The universe there is therefore constructed as of
each date from a roster that keeps the departed and is keyed on CIK, and a
`since_departed` column of zeros on any date means the experiment is void rather
than clean.

**The fade curve is the one place the size of the sample-construction decision was
measured against the size of the modelling decision, and they are comparable.** On
Datadog: moving from the typed straight-line fade to the fitted one is worth
+$3.96 a share. Fitting on survivors only rather than on everything is worth
+$3.75 a share. **The sample-construction decision nobody would have seen is 95% as
large as the entire modelling decision.**

One further channel is worth naming because it is invisible in the obvious place.
In the propensity panel, no public price source serves history for a delisted
symbol, so every price-derived feature is **absent for exactly the companies that
were acquired and present for the companies that were not**. That is not a missing
feature, it is a label in disguise: an indicator that is simply "this company has
since left the filing record" scores an AUC above 0.75 on its own, against 0.57
for the real model. The feature set there is built from filings alone for that
reason, size is log revenue rather than market capitalisation, and the cost is
that **the valuation channel is untested and that model cannot say whether cheap
companies get bought.**

### 14.5 Ranking well and being calibrated are different claims

A model can order takeout candidates beautifully and still say 70% where the
realised frequency is 20%. Ordering is what a screen needs; a stated probability
is what a memo needs. A calibration table buckets on equal-width bins over [0, 1]
rather than by quantile, because the question is absolute: when this model says
70%, how often does it happen. Quantile buckets would hide the fact that a model
never says 70% at all, so empty buckets are kept with a count of zero. The gap is
realised less predicted, so a negative gap is overconfidence.

**A model that ranks well and calibrates badly may be used to sort and must not be
used to state a number.** The propensity screen is exactly that case and says so
in its own output (§15.4).

For ranking, NDCG uses graded relevance with a log2 discount, normalised against
the ideal ordering over the candidates actually ranked. A disclosed peer that
never entered the universe is a screening failure rather than a ranking failure,
so it shows up in recall rather than as an NDCG penalty. With roughly 20 disclosed
peers in a universe of 100 names, precision@10 under a random ordering sits near
0.20, which is the bar, and the notes print the model's own figure beside it so a
reader does not have to know that.

Ablation refits without each feature group to show which ones are load bearing. It
measures the drop from *removing* a group, which is not the same as the group's
marginal contribution where features are collinear: two features carrying the same
signal will both look unimportant. That is stated on the result rather than left
to be discovered.

### 14.6 Overlapping windows, and why the correction is reported as incomplete

Section 11.5 introduced the problem: valuing the same name quarterly on a one-year
horizon produces holding periods that share most of the same price path. The
signal harness is where it is corrected, and the correction deserves three
qualifications.

**First, the mechanism is not what the folklore says.** Overlapping windows alone
do *not* inflate the t-statistic. Overlapping windows plus a **persistent score**
do. The information coefficient is a rank correlation *inside* one date, so a
market move common to every name cancels out of it completely. What survives from
one date to the next is the part of the *ordering* that persisted. Hand the
harness a score redrawn from noise at every rebalance and the coefficient series
comes back with an inflation factor of 1.0 whatever the returns underneath it are
doing; hand it a perfectly persistent score on the same returns and the
first-order autocorrelation lands on 0.75 and the inflation on 1.7. Both are
measured rather than argued. This does not weaken the warning, it sharpens it: a
valuation multiple is about as persistent as a company characteristic gets, and so
is every other score this package produces, so the correction bites on all of
them. But a harness that applied a lag-3 correction to a genuinely transient signal
would be throwing power away for nothing.

**Second, the kernel.** Bartlett weights with a lag of `ceil(horizon / rebalance
spacing) - 1`, which is three for a twelve month horizon sampled quarterly:
coefficients four steps apart no longer share any of their path. Bartlett rather
than the uniform weights of Hansen-Hodrick, because Bartlett guarantees a
non-negative variance estimate and Hansen-Hodrick does not, and a standard error
that comes out imaginary leaves the caller with nothing. The naive and corrected
figures come from one code path, with a lag of zero returning the ordinary
standard error exactly, so the two cannot drift apart through separate
implementations. The lag is measured in months and rounded, because quarterly
dates 90, 91 and 92 days apart make a day-count formula flip between 3 and 4
depending on which quarters the sample contains.

**Third, and this is the part usually left out: the correction is an
undercorrection, by a knowable amount.** On the canonical design the true standard
error is exactly twice the naive one. A Bartlett kernel truncated at the matched
lag of three weights the three autocovariances by 0.75, 0.50 and 0.25, and so
recovers `sqrt(1 + 2 * (0.5625 + 0.25 + 0.0625)) = 1.66` of that factor of 2 *even
with infinite data*. At the forty dates this package can offer it recovers about
1.55. **The corrected t-statistic is therefore still roughly a fifth too large**,
and the module says so rather than presenting the correction as a fix.

Reaching for a longer lag does not help at this sample size, and that was measured
rather than reasoned. At 2,000 observations the recovered factor rises
monotonically with the lag: 1.66, 1.78, 1.83, 1.89 at lags 3, 5, 7 and 11. At 40
observations it peaks near the matched lag and falls away again: 1.55, 1.60, 1.60,
1.54, because the sample autocovariances at high lags are estimated from a handful
of products and shrink toward zero. That measurement is what settles the lag
choice.

A moving-block bootstrap is reported beside the kernel as a second opinion, at
blocks of lag plus one, and a disagreement above a fifth is flagged. On the real
panel the two agree to 2.6%.

**A permutation baseline is also computed, and on the real panel the two nulls
disagree.** Not one of 400 within-date permutations reaches the observed
coefficient, so the permutation p-value hits its floor while the Newey-West figure
says 0.11. Neither is a bug. Permuting inside a date leaves the dates independent
of each other by construction, so the permutation null reproduces the
cross-sectional dependence and *not* the time-series autocorrelation, which makes
it anti-conservative for exactly the reason the naive t is. It is the right null
for whether any cross-sectional relation exists, and the wrong one for how
precisely the mean of 35 overlapping coefficients is measured. Both are reported
and the Newey-West interval is the statistic of record. Having a live case where
the two disagree on real data is worth more than a harness that agrees with
itself.

Finally, multiple comparisons. Running the same panel at three horizons is three
tests, and the number of tests run changes no coefficient and only the p-value.
The verdict prints the Sidak-adjusted figure and the sentence about
pre-registration beside it.

---

## 15. The five models, and what each of them is worth

Each of these is reported the way `verdict()` reports it, which means the negative
results are stated first and in the same typeface as the positive ones. **A reader
who cannot trust the negative results has no reason to believe the positive
ones**, and that property is the most valuable thing in this section.

Every model is off by default. The fitted layer adds outputs beside a valuation;
it never silently changes one.

**Where the numbers below come from, because they are not all the same kind of
number.** The test suite is offline and has to stay fast, so it runs each model at
a reduced configuration and asserts the *mechanism*: that the embargo bites, that
the baseline is the one the card claims, that two fits at one seed agree bit for
bit, that a residual came from a fold which did not train on it. The scores quoted
here come from a full run over the recorded panel, on data the fold never saw, and
the panels themselves are committed with a manifest carrying their provenance, row
counts and digests.

Two of the five headline figures are additionally pinned by the offline suite and
cannot drift without a test turning red: the fade curve's tie against persistence
at one year (§15.3) and the value signal's information coefficient and effective
sample (§15.5). The peer encoder's 0.594, the warranted multiple's 0.765 and the
propensity model's 0.5685 are not pinned to four decimal places, because
reproducing them inside the suite would mean re-running the full panel on every
commit. Reproducing them outside it needs the recorded fixtures and the module's
own entry point: `evaluate_peer_encoder`, `fit_warranted`, `fit_propensity`,
`fit_fade` and `test_signal`. That distinction is stated here rather than left for
a reader to discover from a test file.

### 15.1 The peer encoder: it works, and one of its two towers does not

Two encoders, one over what a company's financial profile looks like and one over
what the company says it does, trained with InfoNCE to put a filer and the peers
its own compensation committee disclosed near each other on the unit sphere.

**The blend is the architecture rather than a post-processing step.** Each tower
ends in an L2 normalisation, so each returns a point on a unit sphere and a dot
product there is a cosine. A company's embedding is the two towers concatenated,
the fundamentals half scaled by `sqrt(1 - w)` and the text half by `sqrt(w)`. Take
the dot product of two such vectors:

```
similarity(a, b) = (1 - w) * cos_fundamentals(a, b) + w * cos_text(a, b)
```

which is exactly what `assumptions.ml.peers.text_weight` is documented to mean,
and the concatenated vector still has unit length, so the whole embedding is
itself a cosine space and the contrastive loss runs on it directly. Each tower is
therefore trained knowing how much of the final similarity it is responsible for,
rather than being trained alone and averaged afterwards by a caller.

**The result.** Walk-forward by proxy filing date, five folds, a 365-day embargo,
every transform refitted inside each fold. 162 scorable targets, mean candidate
universe 162 names.

| Method | NDCG@10 | precision@10 | recall@10 |
|---|---:|---:|---:|
| **Encoder** | **0.594** | **0.535** | **0.337** |
| Text cosine, raw term frequency | 0.473 | 0.396 | 0.253 |
| Fundamentals cosine | 0.373 | 0.325 | 0.201 |
| Text cosine after SVD, no learning | 0.368 | 0.303 | 0.192 |
| Same sub-vertical (§12.1) | 0.301 | 0.283 | 0.175 |
| Nearest by size and growth | 0.249 | 0.236 | 0.148 |
| Popularity prior | 0.221 | 0.211 | 0.131 |
| Random order, closed form | 0.080 | 0.078 | |

**The popularity prior is the weakest baseline, not the strongest, and that was
the expectation it overturned.** The reason is visible in the label: a
compensation peer table is 8 to 30 names drawn from a pool of several hundred, so
no single company is named often enough to carry a ranking. The most-named company
in the sample is NetApp, followed by American Tower, Pinterest and Palo Alto.

The trap is real all the same, because a model can beat a weak baseline by
imitating it, and that is measured directly rather than hoped away: the rank
correlation between the encoder's per-query orderings and the global popularity
order is **0.166**. At 1.0 the model returns the same list whatever it is asked. It
does not.

**Did the learning add anything over the features, which is the question that
matters most.** By the across-target fold dispersion the lift over raw text cosine
is inside the noise, but that bar answers the wrong question: targets differ
enormously in how findable their peers are, so the across-target spread is
dominated by variation both methods share. The methods are scored on identical
queries, so the difference can be taken target by target:

| Baseline | Mean paired difference | SE | t | Encoder wins |
|---|---:|---:|---:|---:|
| Popularity prior | +0.372 | 0.023 | 16.1 | 93.2% |
| Size and growth | +0.345 | 0.023 | 15.2 | 88.3% |
| Same sub-vertical | +0.292 | 0.022 | 13.0 | 85.8% |
| Text cosine after SVD | +0.226 | 0.020 | 11.2 | 80.2% |
| Fundamentals cosine | +0.221 | 0.021 | 10.6 | 78.4% |
| **Text cosine, raw** | **+0.121** | 0.020 | **5.9** | **69.8%** |

So yes, on the hardest comparison, on 70% of targets individually. Both readings
are reported and neither is hidden.

One nuance worth the space. **The SVD alone makes the ranking worse**: the reduced
text cosine (0.368) loses to raw term-frequency cosine (0.473), because projecting
to 128 dimensions throws away ranking information. The contrastive fit on top of
that same projection recovers all of it and then some, +0.226 over the projection
it started from. The learning is doing real work rather than inheriting the
representation's.

**Warm start against cold start, which is the leakage question the date split does
not close.** Splitting by date alone still lets a filer's earlier proxy teach the
model its own later one, so queries are additionally split by whether the target
appeared in any training group:

| | NDCG@10 | Against popularity | n |
|---|---:|---:|---:|
| Warm start, target seen in training | 0.688 | +0.422 | 106 |
| Cold start, target never seen | **0.415** | +0.279 | 56 |

**Quote 0.415 for a new coverage name.** The 0.594 headline is a blend of the two.
Holding out whole filers instead of dates would have been the cleaner protocol and
would have cut the sample below the point where five folds are possible; reporting
both slices is what that choice costs, and it is reported rather than mentioned.

**Which tower carries the signal, and it is not the fundamentals tower.** Ablation
over 12,372 pair observations, rank correlation of pair similarity against a 0/1
relevance:

| Configuration | Score | Damage | Fold sd |
|---|---:|---:|---:|
| Both towers | 0.4535 | 0.0000 | 0.074 |
| Without text, fundamentals only | 0.3511 | **+0.1025** | 0.068 |
| Without fundamentals, text only | 0.4650 | **-0.0115** | 0.065 |

**The text tower carries all of it and the fundamentals tower is not shown to
help.** Its point estimate is negative: text alone scores slightly higher than both
towers together. The -0.0115 sits inside the fold standard deviation, so the honest
statement is "not shown to help, and the point estimate is that it hurts", not "it
hurts". Either way a reader should not believe the fundamentals tower is earning
its place.

Two reasons contribute and only one of them is about the world. A compensation
peer group is chosen for what a company does and for the executive labour market
it competes in, and not for its financial profile. And the panel is built with
**no market feed at all**, so 12 of the 50 features are missing everywhere,
including market capitalisation, enterprise value, the three market-cap-denominated
capital ratios, momentum, beta and realised volatility. The reason is itself a
point-in-time argument: the keyless quote endpoint serves roughly the last three
years, so a panel date in 2021 returns nothing, and a panel carrying market
features at its late dates and none at its early ones would hand the model a clean
proxy for the calendar. **This is the largest single limitation in the model** and
it is very likely part of why the fundamentals tower is not earning its place. The
size gate and the size-and-growth baseline fall back to log total assets, which is
a different quantity, and the code says so.

**Three judgment calls, each with the losing side stated.**

*Binary relevance.* Every disclosed peer grades 1. The graded alternative was to
score a company named in an earlier group but dropped this year at 1 and a current
member at 2. That asserts something no filer asserted, which is exactly the
author's-opinion label this project took from filings to avoid.

*Tanh rather than ReLU in the towers.* A ReLU layer can output exactly zero across
a whole row; with the output bias at zero that row reaches the normaliser as the
zero vector, which has no direction, and comes back with cosine zero against
everything. At He initialisation on a batch of eight it happens to one row in
eight, and the gradient check fails at 0.29 because of it. There is a test that
demonstrates the collapse and the fix, and ReLU remains available.

*The batch sampler flattens popularity.* In-batch negatives assume the off-diagonal
pairs are negatives, which this data violates twice over. A batch takes at most one
pair per disclosed group and uses any company as a positive at most once. That is
also a thumb on the scale against the popularity solution, and it is declared in
the sampler rather than discovered later.

**What could not be made to work.** The size gate turned out to be worth almost
nothing: gating the candidate list on minimum market capitalisation and maximum
size ratio moves NDCG@10 from 0.5937 to 0.6001, a lift of 0.006 where a large free
lift was expected, since a compensation committee selects inside a size band to
begin with. The evaluation therefore runs ungated and the neighbour lookup applies
the gate, with the gap on the record rather than assumed away. Two of the
most-named peers cannot be represented at all, because Intel (61 namings) and
Alphabet (25) answer Item 1 by cross reference to page numbers rather than inline,
so there is no Item 1 to extract; the splitter flags it and declines rather than
returning the wrong span. HubSpot (54) and MongoDB (47) are lost to a
tag-resolution failure at every panel date. The universe had to be widened by hand
from the seed to reach 94% of namings, and after encodability the measured
coverage is **85.5%**, which caps recall at any k for every ranker at once. And the
saved model is 74MB, almost entirely the SVD basis over a 50,000-term vocabulary.

### 15.2 The warranted multiple: it beats the incumbent OLS, and predicts the change at 0.04

What the market pays for a bundle of characteristics, and whether a company sits
above or below that line. Section 7.1 already does this on one comp set of eight
names on one date. This does it on 94 US TMT filers at 22 quarter ends from March
2021 to June 2026, 1,764 company-quarters, which changes what can be asked of it.

**Reflexivity decides what the output means, so it comes first.** The multiple is
the market's opinion. A model fitted on multiples learns the market's own pricing
rule, so the most it can ever say is that a company is priced unlike its
characteristics suggest the market prices characteristics. **It cannot say the
market is wrong.** If the whole sector is mispriced the model is fitted on the
mispricing and reports everything as fair. That is the difference between this and
a DCF: the DCF has an outside anchor in the cash and the discount rate and can tell
you the market is wrong, and this cannot, ever. The sentence every read prints ends
"That is a statement about relative pricing, not about value."

**The headline.**

| Baseline | Model | Baseline | Lift | Fold sd | n |
|---|---:|---:|---:|---:|---:|
| §7.1 OLS on sub-vertical peers | **+0.7651** | +0.6313 | +0.1339 | 0.0841 | 888 |
| §7.1 OLS on the whole TMT cross-section | +0.8391 | +0.4480 | +0.3910 | 0.0341 | 1,500 |
| Sub-vertical median on the date | +0.8294 | +0.6180 | +0.2115 | 0.0557 | 1,441 |
| The company's own multiple last quarter | +0.8412 | **+0.9722** | -0.1310 | 0.0406 | 1,548 |

The lift over the incumbent OLS is outside the fold-to-fold noise. Inside one
sub-vertical on one date, which is the number a screen actually lives on, the
network scores +0.68 against +0.35 for that OLS, averaged over the 106
cross-sections carrying at least eight names.

**And almost all of it is company identity rather than insight.** A feature vector
barely moves in three months and neither does a relative multiple, so a model
fitted on the past is rewarded for recognising a name as much as for understanding
it. That is not lookahead, the training window is strictly earlier, but a reader
shown 0.84 will believe far more than the model earned. Differencing both sides
against the company's own previous observation asks what was added on top of
knowing which company it is:

**the network scores +0.0420 on the change, and the ridge +0.0205.**

That is the honest size of the contribution. It is a field on the result rather
than a note, and `verdict()` prints it in the same paragraph as the headline so
the two cannot be quoted apart. The practical consequence: **the residual is a
description of where a company sits, not a forecast that the gap will close.**
Testing whether it closes needs forward returns with the full horizon as an
embargo, which is §15.5 and is a different piece of work.

**Persistence wins on ordering and is deliberately not the card's baseline.**
Carrying a company's own multiple forward scores 0.9722 against the model's 0.8412.
It is reported loudly, with its own note, because a reader deserves to know the
ceiling. It is not what the card is scored against, because it is built from the
company's own price: a warranted multiple exists to be differenced against that
price, so a construction that reproduces it has a residual of zero by definition
and no signal whatever. The card is scored against the strongest baseline that
answers the same question.

**The network beats the ridge**, 0.8399 against 0.7845 on the same 1,572
observations, so the relationship is not close to linear. No hyperparameter was
chosen by looking at the walk-forward score: width, depth, dropout, learning rate
and batch size were fixed before the first fit and have not moved. The only thing
the data chooses is when to stop, on a held-out tail of the training window.

**The re-rating is real, it is not about the companies, and it is 2% of the
variance rather than most of it.** The move itself is exactly as advertised: median
software EV/Revenue in this panel is 17.1x at the end of 2021, 8.7x a year later
and 7.7x by mid-2026, on fundamentals that did not move anything like that far,
because the discount rate did. But pooled across TMT the calendar is **2.1%** of
the variance of the log multiple, because the dispersion *between* sub-verticals
swamps the dispersion *through* time: IT services at 1.9x and infrastructure
software at 13.2x are four times further apart than either moved across the whole
rate cycle. Inside a single sub-vertical, where the claim actually lives, it rises
to 9% to 18%: application software 16.5%, infrastructure software 13.5%,
semiconductors 16.1%, telecom 10.8%. Still a minority.

`demean_by_date` therefore stays on by default, for the right reason rather than
the expected one: it removes a real effect that has nothing to do with any
company, and it is cheap. A model fitted without it would not have been swamped by
the rate cycle. The date mean is taken **leave-one-out**: it is contemporaneous and
so not lookahead, but a mean that includes the company being predicted hands back
1/N of its own answer, and at N near a hundred that is a small leak in a residual
that is itself small.

**Fitted in logs, and the selection is counted.** Multiples are right-skewed, so a
cross-section running 2x to 25x has a squared error dominated by its top decile.
Logs fix that and make a residual readable as a percentage at any level. The cost
is that the log needs a positive argument, which is a selection decision rather
than a technicality. On EV/Revenue almost nothing is lost. On EV/EBITDA the loss is
severe and structural: the unprofitable filers are exactly the high-growth names
the multiple is most often quoted for, and dropping them quietly restricts the
model to profitable companies and then reports a result as though it were about
all of them. The losses are counted by category so the restriction is visible.

**The feature that is the answer written on the other side of the page.** The
target is log EV less log revenue. Hand the model log market capitalisation or log
enterprise value and it will discover that log EV predicts log EV, report an
R-squared near 0.95 and mean nothing whatever. The same applies with one step of
laundering to every ratio carrying price in a denominator. Those five features are
banned by name with the reason attached and the fit raises on them. The price
momentum features are excluded on a weaker argument, stated as such: momentum
genuinely predicts next quarter's multiple, and a warranted multiple built partly
on the price stops being an independent read on the price.

**A wide universe breaks ladders a software comp set never tested.** The rest of
this engine is exercised on US software filers. Run across semiconductors,
carriers, agencies and tower REITs and three separate resolutions come back
confidently wrong. Each is now checked before the division, and each **refuses the
row rather than correcting it**, because every correction belongs in the resolution
path where it would repair the DCF and the comp tables too, and a model is the
wrong place to change what a ladder returns.

- *A lessor tags only its services revenue under ASC 606.* A tower REIT's tenant
  and ground lease income is lease revenue under ASC 842 and is not a contract with
  a customer, so only its site-services piece carries the concept the revenue
  ladder ranks first and the total sits in `Revenues`. At 31 March 2025 Crown
  Castle resolves **210.0mm against 6,568.0mm**, American Tower 774.6mm against
  10,127.2mm, SBA Communications 128.7mm against 2,083.1mm. Before the guard the
  whole towers and fibre bucket read as trading at 125x revenue. 70 observations
  refused.
- *A filer reporting `LongTermDebtAndCapitalLeaseObligations` has no debt.* That
  concept appears in none of the three debt ladders. Lumen's `LongTermDebt` exists
  but was last written in 2021, so a resolution that correctly asks for a value at
  the balance sheet date finds nothing and returns zero: at 30 September 2023
  **Lumen resolves zero straight debt against roughly twenty billion dollars of
  it**, and its enterprise value prints as 996mm. This is §2.1's retired-tag trap
  one ladder along. Adding the tag is not a one-line fix, because the concept
  includes capital lease obligations and the bridge already adds finance leases
  from their own tags, so importing it wholesale double-counts every finance lease.
  97 observations refused.
- *A price vendor returns the wrong security.* The series for Booking runs 71.39 at
  the start of 2018 to 174.33 in September 2026, against a share that traded near
  1,750 and 5,500 on those days. The ratio is not constant, so it is not a split
  adjustment; it is a different instrument. Nothing inside a valuation can catch
  this, because a wrong series is clean and monotone and plausible. The filer's own
  cover page can: `dei:EntityPublicFloat` is a number the company states rather
  than one the vendor computes, and **Booking's equity value on that series is
  7,796mm against a reported float of 133,100mm**. The tolerance is ten times
  either way and is meant never to fire on anything real: across this universe
  Palantir is the largest honest offender at 6.6 times and everything else lands
  between 0.6 and 2.3. 44 observations refused.

The cost of those three refusals is stated rather than absorbed: they remove seven
filers and cut the towers and fibre bucket from 105 observations to 37.

**One measurement about the engine itself falls out of this panel.** The OLS of
§7.1 refuses to fit on **788 of 1,764** sub-vertical cross-sections, a 45% refusal
rate. That is the engine behaving exactly as §7.1 documents, since a real comp set
is six to ten names against a floor of eight. It is the first time the claim has
been measured rather than asserted.

### 15.3 The revenue fade curve: it ties persistence at one year and beats it at two and three

Every DCF in this engine fades revenue growth from a start rate to a terminal rate
on a straight line (§6.1), and that line moves the answer more than the discount
rate, the margin path and the terminal multiple put together. Nothing about it is
measured. This measures it: given what a technology company looks like on the day
its 10-K is filed, how fast does its revenue growth actually decay?

**The verdict first, because it is a tie at the horizon most people care about.**
Mean absolute error on forward revenue growth, walk-forward by date with an
embargo of 365 days per year of horizon, 224 TMT filers, 2009 to 2026:

| Horizon | Model | Persistence | Training mean | Sub-vertical mean | Fold sd | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 1 year | **0.1474** | 0.1510 | 0.1734 | 0.1693 | 0.0242 | lift of 0.0035, **inside** the fold noise. A tie. |
| 2 years | **0.1476** | 0.1884 | 0.1583 | 0.1578 | 0.0162 | lift of 0.0407, outside the noise |
| 3 years | **0.1391** | 0.1921 | 0.1509 | 0.1441 | 0.0176 | lift of 0.0530, outside the noise |

**At one year out this model does not beat doing nothing, and a test asserts the
tie so it cannot drift away quietly.**

That pattern is the result rather than a disappointment. Persistence gets worse as
the horizon lengthens, 0.1510 then 0.1884 then 0.1921, while the model gets better.
Growth is sticky one year out, so there is almost nothing to add to last year's
number, and by year three last year's number is actively misleading. A fade curve
is a claim about the second regime, which is the regime a five-year DCF spends four
of its five years in.

Note also that the model beats the *sub-vertical* mean at every horizon but only by
0.005 at three years, which is well inside the fold noise. Read that as "the model
is a slightly better sector mean", not as stock picking.

**The curve in two numbers.** Forward growth `= 0.0693 + 0.4719 x` trailing growth,
trimmed at the 1st and 99th percentiles of both axes. Growth closes **53% of the gap
to a long-run TMT mean of 13.1% every year**, a half-life of eleven months. **The
default assumptions fade a fifth of the gap a year, so the typed schedule fades
less than half as fast as the filings do.**

Untrimmed the slope is 0.15, because one filer growing 2,026% in a year has more
leverage on a least-squares line than the other two thousand put together. The
decile table assumes no functional form at all and is the version to quote in an
argument:

| Trailing growth decile | Trailing | Forward | Fade |
|---|---:|---:|---:|
| 1, worst | -15.7% | +7.3% | +23.0 pts |
| 5 | +9.1% | +7.3% | -1.9 pts |
| 8 | +26.9% | +22.3% | -4.6 pts |
| 10, best | +89.7% | +47.4% | -42.3 pts |

Both ends move toward the middle and the crossing point sits between 9 and 14%.

**What it is worth on one company, because an R-squared is a claim about a panel
and this is a claim about a business.** Datadog:

| Case | Y1 | Y2 | Y3 | Y4 | Y5 | EV (mm) | Per share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Assumed fade | 20.0% | 16.2% | 12.5% | 8.8% | 5.0% | 9,447 | **36.65** |
| Fitted fade | 21.3% | 19.6% | 20.5% | 12.7% | 5.0% | 10,901 | **40.61** |
| Fitted, 10th percentile | 1.0% | -2.2% | 4.1% | 4.6% | 5.0% | 5,897 | 26.97 |
| Fitted, 90th percentile | 45.3% | 40.0% | 46.0% | 25.5% | 5.0% | 20,472 | 66.69 |

The fitted path is worth +$3.96 a share, 15.4% of enterprise value. **The band the
fit actually supports runs $26.97 to $66.69, and anyone quoting the middle of that
as a forecast has stopped reporting a measurement.**

**The fitted curve cannot reach a terminal value, and saying so is half the
module.** Mean reversion in the filings runs toward the average growth of a growth
sector, 13%, and no perpetuity can carry that. The fitted models govern the first
`horizon_years`; after that the path falls back to the engine's own straight line
toward `dcf.revenue_growth_terminal`, now anchored on a measured level rather than
a typed one. The basis is recorded on **every single year** as fitted or assumed,
so the handover is visible instead of blended away, and setting the horizon equal
to the projection length produces a note saying the path ends on a cliff. Nothing
constrains the fitted years to decay either, and Datadog's come out 21.3%, 19.6%,
20.5%: imposing monotonicity would be an assumption dressed as a result. The
one-year slope is 0.47 and the three-year slope is 0.17, which is not 0.47 cubed,
so a constant-decay model is wrong about the shape as well.

**Restatement, measured rather than assumed, and the expectation was wrong in an
interesting direction.** Restatement was expected to be commoner than people
assume. It is rarer. Holding the us-gaap tag fixed, **0.89%** of the 2,798 fiscal
years carry a different revenue today than in the filing that first reported them,
and 0.71% differ by more than one percent. What is two and a half times commoner,
and far larger when it happens, is the filer moving revenue to a different tag:
through the full ladder **2.6%** of years move and 1.75% move by more than a
percent. Every one of the largest gaps in the panel is a tag migration rather than
a restatement. Crown Castle's fiscal 2017 reads 88% lower through the ladder today
than in its own 10-K, and nothing was restated. Both numbers are reported
separately, because only one of them is about accounting.

**Acquisitive years are kept, and the reasoning is a valuation argument rather than
a statistical one.** 399 labelled observations spent more than a tenth of revenue
on acquisitions. They grow 2.1 points faster than quiet years in the year of the
deal and 5.9 points faster in the year after it, because a deal closing in June
contributes six months to this year and twelve to the next. Excluding acquisitive
years does not remove noise: it fits a fade curve for a world in which nobody does
M&A and then hands it to a DCF valuing a company that will keep doing M&A.
Acquisition spend over revenue is a feature instead. Excluding them changes nothing
worth having either, 0.1350 against 0.1328 at one year and 0.1314 against 0.1344 at
three.

**The trap that cost the most, and it is worth reading even by someone who will
never fit a model.** The first version of this module scored a mean absolute error
of 0.46 on the first fold against 0.16 for doing nothing, and forecast 370% revenue
growth for a company. Two defects that only bit together. First, **a feature that is
a clock**: "years of filing history" is not a company characteristic, it is the
calendar, and in an expanding-window walk-forward its test range never overlaps its
training range. Second, **standardising by the imputed column's dispersion**: a
column 75% missing in the training fold collapses to a quarter of its real spread
once the median fills it, and a test observation two real standard deviations out
then arrives at the model as sixty-six. The scaler now centres and scales on
observed training values only and clips the design matrix at five standard
deviations, and every feature's train-to-test shift in training standard deviations
is reported so the next clock is visible before it is fitted. The clock feature is
out on the mechanism rather than on the score: with the scaler fixed it *improves*
the one-year error by 0.002, which is the model reading the date off the feature.

**A bug in the revenue ladder, found here and fixed.**
`RevenueFromContractWithCustomerIncludingAssessedTax` ranked above `Revenues`.
Charter Communications tags the first at **889mm** for fiscal 2025 and the second at
**54,774mm**, so the engine returned revenue 62 times too small and every multiple
struck on it 62 times too high. Charter is inside this engine's own TMT universe and
reachable by the comp tables. The fix is an opt-in component guard on the trailing
twelve month resolution: with it set, every remaining ladder entry is also resolved,
and one that covers the same window with a value more than the guard multiple of the
winner's replaces it, with the provenance recording which tag it beat and by how
much. The guard is ten rather than two, measured across all 224 filers: the genuine
scope disagreements in this universe, revenue before or after billable expenses at
Interpublic and services revenue against total at Starz, run two to five times, and
picking the larger of those by rule would be a thumb on the scale. At ten the guard
fires on Charter and on SBA Communications and on nothing else. It is opt-in per
concept because the test is a ratio, and a ratio conveys nothing about a concept
that passes through zero: on EBIT an order of magnitude is an ordinary year.

**One more expectation overturned.** The panel was expected to reach 15 to 18 years
for mature names. Measured on 224 filers, the deepest is 19 fiscal years, **the
median is 12**, and only 34 reach 19. The consequence shows up in every fold table:
the first walk-forward fold trains on **64 observations** and is asked about 521.

### 15.4 M&A propensity: it ties the size sort, and the lift is inside the noise

Which technology companies get bought, scored on what was knowable before the bid.

**The headline.** Walk-forward AUC **0.5685** against **0.5474** for sorting the
universe smallest first. A lift of **+0.0212** with a fold-to-fold standard
deviation of **0.0910**, so `beat_baseline` is true and the honest reading is that
**the model ties the size sort**. Sample: 493 registrants, 9,400 labelled
observations, 336 positives over 97 distinct deals, base rate 3.57%, five folds, a
462-day embargo.

The part that is usable is the top of the list, which is the part a coverage banker
reads. Mean precision at twenty, per date:

| | precision@20 | recall@20 |
|---|---:|---:|
| Model | **0.0658** | 0.137 |
| Size sort | 0.0289 | 0.064 |
| That date's base rate | 0.0378 | |

The screen finds a target roughly one time in fifteen against a base rate of one in
twenty-six, and one in thirty-five for sorting by size. **It is a screen, not a
probability**: the calibration table has gaps of minus 30 to minus 55 points above
the 0.3 bucket, so the model is badly overconfident wherever it speaks loudly, and
the result says so in the output rather than in a footnote. Section 14.5 exists for
exactly this case.

**The number that should stop a reader, and it is the most useful thing in the
model.** Two extraction bugs in the precedent scanner were recovering seven extra
transactions. **Adding those 7 deals to a set of 100 moved the headline from "the
model does not beat the baseline, use the baseline" to "the model beats the
baseline by +0.0212".** Same code, same panel, same seed; the label set moved by 7%
and the sign of the conclusion changed. Both numbers are real and both are
recorded. That is not a claim that the model is good or bad, it is a measurement of
how much this sample can support, and the answer is: **not a conclusion of this
size.** Coefficient stability says the same thing from the other end. Six of fifteen
coefficients change sign between folds, and one has a mean of +0.65 with a standard
deviation of 1.28. Nothing here should be quoted to two decimal places.

**Targets, not completions, and the choice is not a coin toss.** A deal that
regulators block was still a bid, and the company was still a target. Xerox offered
for HP, HP filed a Schedule 14D-9, and everything that made HP worth bidding for
was true whether or not the bid succeeded. Predicting completion is a different
problem whose drivers sit with the acquirer, the financing market and the antitrust
division, none of which is a fact about the target and none of which is in this
feature set. It is also right-censored in a way that quietly corrupts the label: a
deal announced four months before the as-of date has neither completed nor broken,
and calling it a failure because no Form 25 has landed labels a live deal as a dead
one. The completion flag is carried for reporting and is never the label.

**Class imbalance.** At a base rate of 3.57% a model predicting "never" is right
96.4% of the time, so **no accuracy figure appears anywhere**. The positive class is
weighted by the ratio of negatives to positives during fitting and that weighting is
undone before any probability is quoted, which is why the mean predicted probability
lands near the base rate rather than near a half.

**Censoring has two directions and the second one is easy to get backwards.** The
familiar error is calling an unresolved window a zero, which labels every recent
deal as a non-deal. The mirror error is dropping unresolved observations only after
checking the label. A positive is settled the day the agreement is announced; a
negative is settled only when the whole window closes. Applying the test to the
negatives alone keeps every recent deal and discards the companies it should be
compared against, and **the 2025 base rate came out at 9.9% against a true rate near
3.5%**, so the last fold trained on a world that never existed. The fix is one line
moved: drop the observation before looking at its label.

**Base rates move with the cycle**, and that is part of the result rather than a
diagnostic: 2022 runs at 2.05% and 2023 at 4.53%, a factor of 2.2 inside one sample.

**Three findings about the filing record, each verified rather than assumed.**

*A Form 25 is not a departure.* The obvious test for whether a company has left is
the first Form 25 or Form 15 in its filing index, and it is wrong. A Form 25
delists a *security*, not a company, and an issuer files one when a class of
warrants expires or a note matures. Super Micro Computer filed one in March 2019 and
has filed a 10-Q every quarter since; a first roster built that way marked five live
companies dead. The test used is the Form 25 or Form 15 that is followed by no
further periodic report.

*Deregistration is not the only exit.* Avaya's last 10-Q was September 2023 and its
filing record simply stops, with no qualifying Form 25 at all. Left in, it sat in
every later cross-section carrying three-year-old numbers frozen at their worst and
screened as a permanent takeout candidate. Fifteen months after the last periodic
report a registrant therefore leaves the universe whether or not anything was filed
to say so.

*The forms that establish a deal are wider than a merger proxy implies.* A merger
proxy is a DEFM14A only when shareholders are asked to vote. When a holder with the
votes approves by written consent, Regulation 14C applies and the company files an
information statement and never a proxy at all. In a screen of 493 registrants, **13
of the 23 departed companies a proxy-only screen could not explain had filed a
DEFM14C**: PowerSchool, Instructure, Thoughtworks, Informatica, Paycor, SolarWinds,
Vizio. Sponsor and founder take-privates almost without exception.

**And one bug in the precedent extractor that was biased against the hypothesis it
was being used to test.** The clause that asks whose stock a conversion clause
converts exists to stop an acquirer's own S-4 becoming a precedent in which the
acquirer was acquired. But the regex reading the name after "share of" read
`Class B Common Stock` and reported a company called *Class B*, and the first such
clause aborted the whole extraction. **Every dual-class filer has a clause like
that**, because the class the public holds is never the only class. PowerSchool's
information statement converts Class B in one clause and Class A into cash in
another, and the Class B clause comes first, so a real $5.6bn transaction was
discarded unseen. This does not miss deals at random: **it misses founder and
sponsor controlled companies, which is the same population the model was built to
test.** The fix distinguishes a refusal naming another registrant, which stops the
scan, from one naming a share class, which does not.

**What could not be made to work.** The dual-class control feature is not in the
model: the instance parser does reach dimensioned share-class facts, but it reads
the latest filing's instance document, so building the feature point in time across
493 names and 30 dates needs one archived instance per company-date. Shipping
without the feature beat shipping a version of it that quietly reads today's share
counts. The label set also carries false positives and was not hand-cleaned: of the
7 recovered transactions, 4 are target-side acquisitions and 3 are an acquirer-side
document or an internal reorganisation the extractor cannot yet tell from a sale,
and the same error rate presumably runs through the other 100. It is shipped as the
code produced it, because **a label set corrected by the author's opinion of which
deals are real is the author's opinion wearing a label set's clothes.**

The model is linear on purpose. On 97 deals anything with more capacity fits the
noise, and it would have nothing to say when somebody asks why a name is on the
list. Every ranked name carries an attribution whose contributions sum to the logit.

### 15.5 The value signal: it is negative, and it is not significant

The adversary. Any score at all, as `(ticker, date, value)`, scored against forward
returns. Nothing in it imports another model, so nothing in it can be tuned to
flatter one, and every model in this package can be tested through the same harness
on the same conventions. **A harness that knows what it is scoring is a harness that
can be made to agree with it.**

The demonstration signal is the obvious one: cheapness on trailing EV/Revenue, built
from this engine's own bridge through a fact set pinned to each row date. 100 TMT
companies, 35 quarterly cross-sections from December 2016 to June 2025, twelve month
forward returns, 2,604 scored company-dates.

| | |
|---|---|
| Mean information coefficient | **-0.0984** |
| Share of dates positive | 43% |
| t, naive | **-2.65** |
| t, Newey-West at lag 3 | **-1.62** |
| Standard error inflation | 1.64x |
| Effective observations | **13**, against 35 dates and 2,604 company-dates |
| Top minus bottom bucket, 12 months | -5.4% (t = -0.79) |
| Bucket rank correlation | -0.90 |
| Baseline, random score with the same cross-sectional shape | +0.0002 |
| Verdict | does NOT beat the baseline. Use the baseline. |

**Value was a negative signal in technology over this decade.** The cheap half of
the universe underperformed the expensive half by about five points a year and the
bucket table is close to monotone in the wrong direction. That is what everyone who
lived through 2017 to 2021 remembers, and it is the opposite of what the textbook
says value does.

**And it is not significant.** The naive t-statistic is -2.65 and would have been
written up. The Newey-West figure is -1.62 and would not. Both numbers are in the
result and the verdict prints NOT SIGNIFICANT in those words. A clean negative that
is also not significant is the most credible thing this harness could have produced
on its first real signal, because it is a result nobody would have manufactured on
purpose.

**The correction is doing exactly what §14.6 says it does.** The first-order
autocorrelation of the real coefficient series is +0.76 against a theoretical 0.75
for a twelve month return sampled quarterly, which is about as clean a confirmation
of the design as this sample can give. Three horizons on the same panel, which is
also three tests and is reported as three:

| Horizon | Lag | rho(1) | SE inflation | t naive | t Newey-West | p, Sidak for 3 tests |
|---|---:|---:|---:|---:|---:|---:|
| 6 months | 1 | +0.38 | 1.16x | -2.11 | -1.81 | 0.195 |
| 12 months | 3 | +0.76 | 1.64x | **-2.65** | **-1.62** | 0.285 |
| 24 months | 7 | +0.84 | 2.14x | **-2.92** | **-1.36** | 0.435 |

**The naive statistic gets larger as the horizon lengthens and the corrected one
gets smaller.** A longer window is a smoother series with less dispersion to divide
by, so the naive figure improves for a reason that has nothing to do with evidence;
the correction tracks the overlap instead, and the overlap is worse. The two moving
in opposite directions over the same range is the clearest statement available that
one of them is measuring something other than skill.

**Turnover, reported because a small edge with high turnover is a costless-trading
illusion.** On the real panel, 11% of the top bucket and 16% of the bottom are
replaced per rebalance against 4.7% universe churn, with churn reported separately
so a signal is not charged for a name that left the universe rather than the bucket.
The break-even one-way cost is reported only where the spread is positive, because a
signal that loses money before costs has no break-even cost and printing a negative
one invites it to be read as a threshold.

**Acquisitions terminate the holding period rather than deleting the observation.** A
company acquired six months into a holding period has no twelve month return in the
usual sense, and dropping it deletes precisely the outcome with the largest return in
the sample: a takeout at a 40% premium. Every such name is terminated at the deal, at
the offer price from §12.6 where the filings pin one down and at the last traded close
where they do not, with the remaining months carried at a supplied benchmark or at
cash. Carrying the remainder keeps every observation on the same clock; the
alternative puts a six month holding period into the same statistic as a twelve month
one.

**Survivorship could not be corrected here. It could only be measured, and the
measurement is a flag.** The delisting machinery is built, tested and exercised, and
on the real panel it finds nothing to do, because the data sources delete a company
the day it stops trading. Five seed names return no CIK and no price rows, and every
one of the five left to an acquisition. They are not in the panel at all, so the
harness cannot terminate them at a deal, and the sensitivity to the delisting return
comes back flat because there are no delistings to be sensitive to. The check says so
in terms:

> FLAG: not one name in this sample was acquired or delisted over the whole period.
> That is not what happens to a technology universe over a decade, so the universe is
> a list of today's survivors and every number here is biased upward by the outcomes
> it cannot see.

The direction matters for reading the headline. The missing names are takeouts,
takeouts earn a premium, and takeouts skew cheap. **Their absence flatters cheapness,
so the true coefficient is if anything more negative than -0.098, not less.**

**Point in time is refused rather than warned about.** A score whose knowledge date
postdates its own date is an error, and so is a provenance carrying a filing later
than the score. This is also the one module in the package where a price series
running past its row date is *required* rather than forbidden, since the forward
return is the label, and the forward prices live in an object the scoring path
receives and the valuation path never does (§11.2).

**Coverage, stated because it is uneven.** 2,992 of a possible 3,900 company-dates
carry a multiple. 777 fail because the trailing twelve months will not tile out of
the quarterly facts on file at the date, which hits the December cross-sections
hardest because they anchor on a third quarter. 326 fail because a company had not
yet listed. Two filers report revenue under tags outside the ladder and resolve
almost nowhere. The thinnest scored cross-section has 31 names and the widest 96.

---

## 16. What the sample can support, which is less than it looks

Everything in sections 13 to 15 is fitted on the same underlying record, and this
section states its size in the terms a sceptical reader would use rather than the
terms that flatter it.

**Start with the honest count.** The universe is roughly 110 US-listed TMT filers.
The panels built on it run 10 to 15 years depending on which one, and every headline
in section 15 is quoted against a company-date count in the thousands. **Those counts
are not the sample size.**

- The price endpoint serves exactly ten years and no more, for every symbol tested.
  The signal panel is therefore 39 quarterly dates from December 2016, of which 35
  carry a full twelve month forward. That is **8.5 non-overlapping annual periods**,
  and after the overlap correction it is **about 13 effective observations** against
  2,604 company-dates. Both counts are on the result, with the small one first.
- The filing record is deeper but not by as much as expected. Measured across 224
  filers, the deepest history is 19 fiscal years, **the median is 12**, and only 34
  reach 19.
- The warranted panel is 94 filers at 22 quarter ends. A feature vector barely moves
  in three months and neither does a relative multiple, so consecutive quarters of
  one company are close to one observation repeated. That is why the differenced
  number, +0.042, is quoted in the same paragraph as the pooled 0.84.
- The propensity model rests on **97 distinct deals**. Seven of them moved the sign
  of its conclusion.
- The peer encoder trains on 2,534 directed pairs drawn from 320 disclosed groups
  filed by 75 filers, and it is scored on 162 targets of which 56 are genuinely cold
  start.

**Cross-sectional counts do not rescue any of this.** A hundred names inside one
quarter share a sector, a rate cycle and a market, so they are not a hundred draws
either. The correction applied here handles overlap *through time* and does nothing
about dependence *across names* within a date, and that is stated in §11.5 and not
fixed anywhere.

So: **roughly 110 companies over 10 to 15 years is a few dozen genuinely independent
observations.** Every score in section 15 should be read against that number rather
than against the company-date count beside it. It is why fold dispersion is printed
on every result, why no p-value is quoted off a t-statistic in §7.1, why the
propensity coefficients are not quoted to two decimal places, and why a lift of 0.02
against a fold standard deviation of 0.09 is printed as what it is.

**Five things no amount of care downstream fixes.**

1. **There is no point-in-time universe.** The SEC's ticker file is today's list, the
   price sources delete a company the day it stops trading, and this package has
   neither an archived ticker file per date nor a delisting-complete vendor file.
   Every model in section 15 is biased upward by the outcomes it cannot see, by the
   amounts and in the directions tabulated in §14.4. The harnesses are built to
   consume a delisting file the day one exists.
2. **There is no market feed in the peer panel.** Twelve of fifty features are
   missing everywhere for the reason in §15.1, and the fundamentals tower is being
   judged on a degraded input. Its failure to earn its place is therefore not a
   verdict on fundamentals in general.
3. **The labels are compensation peers rather than trading comparables.** A committee
   picks partly for competition for executive talent, and the model inherits that.
4. **A warranted multiple cannot say the market is wrong.** It is fitted on the
   market's own pricing rule, so a sector-wide mispricing is invisible to it by
   construction.
5. **A total-return series is not available.** The price layer carries closes, so a
   dividend payer's realised return is understated by its yield. No name in the TMT
   universe pays a material one, but the convention is stated rather than assumed.

**What the fitted layer is for, given all that.** Not for a number to put in a
valuation. A comp set proposed by a model that clears 0.415 cold start is a better
starting list than a hand-written one and still needs an analyst to strike names off
it. A warranted residual is a description of where a company sits, and §15.2 is
explicit that it is not a forecast that the gap will close. A fitted fade curve is an
argument that the typed straight line fades less than half as fast as the filings do,
and the band it supports is wide enough to print. A propensity screen is twenty names
that are at least arguable in a Monday meeting. **Each of those is worth having and
none of them is worth more than the filing it traces back to**, which is the same
standard section 2 applies to a revenue line.

The last word belongs to the protocol rather than to any result. Every model in this
package carries a card recording what it was trained on, over what period, with what
features, how it scored out of sample, against what alternative, and what it cannot
do. The card is where a small sample is said out loud. **A number nobody can audit
does not belong beside numbers that trace to filings**, and a fitted number with no
baseline beside it is exactly that.
