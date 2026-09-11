"""Point-in-time diluted share count under the treasury stock method.

The share count is the quietest large error in a valuation. Equity value is
price times shares, so a count that is three percent light makes every
per-share number three percent rich, and nothing in the output looks wrong.

The engine's default count is trailing-twelve-month diluted weighted-average
shares outstanding. That is the correct denominator for reported EPS and the
wrong one for a valuation struck today, for two reasons. It is an average over a
past window, so a company issuing steadily ends the window above its own
average. And it reflects the awards that were outstanding across that window
rather than the ones outstanding now. Datadog shows both effects: 352.3mm basic
weighted-average shares over the trailing year against 359.1mm actually
outstanding on the cover of the latest 10-Q.

This module builds the count a live model uses: shares actually outstanding at
the most recent reported date, plus the net new shares that in-the-money awards
would create at today's price.

**Treasury stock method, ASC 260.** An in-the-money option is assumed exercised.
The company issues the full option count and spends the exercise proceeds buying
its own shares back in the market, so only the difference is genuinely new:

    net new shares = count - (count x strike) / price,   when price > strike

An out-of-the-money option is antidilutive and contributes nothing. It must
never contribute a negative number: the formula above turns negative once the
strike exceeds the price, which would have worthless options shrinking the share
count and lifting per-share value. Out-of-the-money blocks are dropped, not
netted off the in-the-money ones.

**Which price.** ASC 260 uses the average market price over the reporting
period, because the diluted EPS denominator is itself a period average and the
numerator and denominator have to describe the same window. This is not an EPS
calculation. It is a count struck at an instant and then multiplied by today's
price to get equity value. Deciding which options are in the money on a
three-month average price and then valuing the resulting shares at today's price
mixes two dates, and in a stock that has run it drops options that are in the
money right now. Spot is used throughout.

Four traps, every one of them observed in the filings committed as fixtures:

*companyfacts does not carry these facts.* The SEC's companyfacts endpoint
publishes undimensioned facts only. Option counts sit under the award-type and
plan axes, and a dual-class issuer's shares outstanding sit under the
class-of-stock axis, so none of it appears there. Datadog has no undimensioned
``dei:EntityCommonStockSharesOutstanding`` at all: it reports 334.9mm Class A
and 24.2mm Class B, and the count is the sum. Reading either alone understates
the company by seven percent or by ninety-three.

*A roll-forward tags its opening balance with the same tag as its closing
balance.* Datadog's option table carries 3,474,619 options at 7.26 and 1,610,360
at 7.83 under one tag, the first being the opening balance six months stale. The
only thing separating them is the context date, so the latest instant wins here
and anything earlier is ignored.

*Near-miss tags.* The award footnote is full of tags that look like the one
wanted. CrowdStrike tags an undimensioned
``OptionsVestedAndExpectedToVestOutstandingWeightedAverageExercisePrice`` at the
same date as the real outstanding strike. Zscaler tags its whole antidilutive
securities table under ``OptionsOutstandingNumber``, dimensioned by award type,
so that 8.9mm RSUs and 1.8mm ESPP shares sit under an options tag next to the
150,000 options that are actually options. Tag matching is therefore exact, and
option counts are taken undimensioned or by exercise-price band and from
nowhere else.

*A weighted-average strike hides the bands inside it.* One average strike below
today's price implies the whole grant is in the money. Split into bands it often
is not, and the error runs one way. The average credits the company with
exercise proceeds from options nobody would exercise, and those proceeds buy
back shares that would never be bought, so a single average strike understates
dilution whenever any band is out of the money. Where the filer tags bands they
are valued one at a time; where it does not, the single average is used and the
notes say which happened.

Convertible notes are deliberately absent from this count. Under ASU 2020-06 the
EV bridge already decides whether an instrument is carried as debt or as
converted equity, and counting its shares here as well would count the same
claim twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

from .config import Assumptions
from .edgar import DimensionedFact
from .errors import MissingDataError, NotMeaningfulError, TechvalError
from .financials import Financials

_MM = 1e6

# Shares outstanding, most recent first. The dei cover-page count is struck
# weeks after the balance-sheet date, which is why it leads: Datadog's cover
# count is dated 2026-07-31 against a 2026-06-30 balance sheet.
_OUTSTANDING_TAGS = ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding")

# Exact tags, lowercased. Substring matching cannot be used here: see the
# near-miss trap in the module docstring. The plural "Arrangements" spellings are
# genuine alternates in the taxonomy, not typos.
_OPTION_COUNT_TAGS = frozenset(
    t.lower()
    for t in (
        "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingNumber",
        "ShareBasedCompensationArrangementsByShareBasedPaymentAwardOptionsOutstandingNumber",
        "ShareBasedCompensationSharesAuthorizedUnderStockOptionPlansExercisePriceRangeNumberOfOutstandingOptions",
    )
)
_OPTION_STRIKE_TAGS = frozenset(
    t.lower()
    for t in (
        "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingWeightedAverageExercisePrice",
        "ShareBasedCompensationArrangementsByShareBasedPaymentAwardOptionsOutstandingWeightedAverageExercisePrice",
    )
)
_RSU_COUNT_TAGS = frozenset(
    t.lower()
    for t in (
        "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber",
        "ShareBasedCompensationArrangementsByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber",
    )
)

_CLASS_AXIS = "classofstockaxis"
_BAND_AXIS = "exercisepricerange"


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AwardBlock:
    """One population of awards that dilutes on its own terms.

    A block is the unit the treasury stock method is applied to. Options are
    split by exercise-price band where the filer tags bands, because a band is
    either in the money or it is not and averaging across bands loses that.
    RSUs carry no strike at all.
    """

    kind: str  # 'option' or 'rsu'
    count: float  # millions of shares
    strike: float | None  # dollars per share, None for RSUs
    source_tag: str


@dataclass
class ShareCount:
    """A share count with the walk that produced it, in millions of shares."""

    ticker: str
    basic_outstanding: float | None
    option_awards: list[AwardBlock]
    rsu_awards: list[AwardBlock]
    net_new_from_options: float
    rsu_shares: float
    fully_diluted: float
    diluted_waso: float
    as_of: date
    accession: str | None
    method: str
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def gap_shares(self) -> float:
        """This count less the trailing diluted weighted average, in millions."""
        return self.fully_diluted - self.diluted_waso

    @property
    def gap_pct(self) -> float | None:
        """The same gap as a fraction of the weighted average, not a percentage."""
        if not self.diluted_waso:
            return None
        return self.gap_shares / self.diluted_waso

    def rows(self) -> list[tuple[str, float]]:
        if self.basic_outstanding is None:
            return [
                ("Diluted weighted-average shares (mm)", self.diluted_waso),
                ("Fully diluted shares (mm)", self.fully_diluted),
            ]
        return [
            ("Basic shares outstanding (mm)", self.basic_outstanding),
            ("+ Net new shares from in-the-money options (mm)", self.net_new_from_options),
            ("+ Unvested RSUs (mm)", self.rsu_shares),
            ("Fully diluted shares (mm)", self.fully_diluted),
            ("Memo: diluted weighted-average shares (mm)", self.diluted_waso),
            ("Memo: difference (mm)", self.gap_shares),
        ]


# --------------------------------------------------------------------------- #
# Fact handling
# --------------------------------------------------------------------------- #


def _dims(fact: DimensionedFact) -> tuple[tuple[str, str], ...]:
    """Dimensions as a sorted key, with namespace prefixes stripped.

    Instance documents differ in the prefix they bind a taxonomy to, so the
    local name is the only part worth comparing.
    """
    return tuple(
        sorted(
            (axis.split(":")[-1], member.split(":")[-1])
            for axis, member in fact.dimensions.items()
        )
    )


def _member(key: tuple[tuple[str, str], ...], axis_needle: str) -> str | None:
    for axis, member in key:
        if axis_needle in axis.lower():
            return member
    return None


def _instants(
    facts: list[DimensionedFact], tags: frozenset[str] | tuple[str, ...], unit: str, as_of: date
) -> dict[tuple[date, tuple[tuple[str, str], ...]], DimensionedFact]:
    """Instant facts for a set of tags, deduplicated by date and dimensions.

    Inline XBRL repeats the same fact once for each place it is rendered in the
    document, so the raw list carries duplicates that would double any sum.
    """
    wanted = {t.lower() for t in tags}
    out: dict[tuple[date, tuple[tuple[str, str], ...]], DimensionedFact] = {}
    for f in facts:
        if not f.is_instant or f.unit != unit or f.end > as_of:
            continue
        if f.tag.split(":")[-1].lower() not in wanted:
            continue
        out[(f.end, _dims(f))] = f
    return out


def _latest(rows: dict[tuple[date, tuple], DimensionedFact]) -> date | None:
    return max((d for d, _ in rows), default=None)


# --------------------------------------------------------------------------- #
# Shares outstanding
# --------------------------------------------------------------------------- #


def _basic_outstanding(
    facts: list[DimensionedFact], as_of: date
) -> tuple[float, date, str, list[str]]:
    """Shares outstanding, summed across share classes, with its date and tag."""
    notes: list[str] = []
    for tag in _OUTSTANDING_TAGS:
        rows = {
            key: f
            for key, f in _instants(facts, (tag,), "shares", as_of).items()
            # A class-of-stock member is the only dimension allowed. This drops
            # contexts that combine a class with a plan, such as Datadog's
            # employee stock purchase plan shares, which are not outstanding
            # shares and would be added twice if they were.
            if all(_CLASS_AXIS in axis.lower() for axis, _ in key[1])
        }
        when = _latest(rows)
        if when is None:
            continue
        at_date = {key[1]: f for key, f in rows.items() if key[0] == when}

        by_class = {
            _member(key, _CLASS_AXIS): f.value
            for key, f in at_date.items()
            if _member(key, _CLASS_AXIS) is not None
        }
        if by_class:
            total = sum(by_class.values())
            notes.append(
                f"Shares outstanding of {total / _MM:,.2f}mm at {when} are the sum of "
                f"{len(by_class)} share classes ({', '.join(sorted(by_class))}) tagged "
                f"{tag}. A dual-class issuer reports each class separately and "
                "companyfacts drops both, so the instance document is the only place "
                "the full count appears."
            )
            return total / _MM, when, tag, notes

        plain = at_date.get(())
        if plain is not None:
            notes.append(
                f"Shares outstanding of {plain.value / _MM:,.2f}mm at {when}, tagged "
                f"{tag} with no class dimension, so the filer has a single class."
            )
            return plain.value / _MM, when, tag, notes

    raise MissingDataError(
        "shares outstanding",
        tags_tried=list(_OUTSTANDING_TAGS),
        period=f"on or before {as_of}",
        hint=(
            "the instance document tags no cover-page or balance-sheet share count "
            "that carries either no dimension or a class-of-stock dimension only"
        ),
    )


def shares_outstanding(
    facts: list[DimensionedFact], as_of: date
) -> tuple[float, list[str]]:
    """Shares actually outstanding at the latest reported date, in millions.

    Summed across share classes. This is the reason the instance document has to
    be fetched at all: a dual-class issuer tags each class under the
    class-of-stock axis, and companyfacts publishes undimensioned facts only, so
    it drops every class and leaves the concept looking absent. Datadog is one,
    with 334.9mm Class A and 24.2mm Class B.

    The cover-page count is preferred over the balance-sheet count because it is
    struck later, typically within a week of the filing date rather than at the
    quarter end, and a point-in-time count should be as recent as the filing
    allows. Raises rather than falling back to a weighted average, since the
    caller has a weighted average already and needs to know it is using it.
    """
    value, _when, _tag, notes = _basic_outstanding(facts, as_of)
    return value, notes


# --------------------------------------------------------------------------- #
# Awards
# --------------------------------------------------------------------------- #


def _option_blocks(
    facts: list[DimensionedFact], as_of: date
) -> tuple[list[AwardBlock], list[str], list[str]]:
    """Options outstanding at the latest reported date, by band where tagged.

    Only two shapes are trusted: an undimensioned total, and rows dimensioned by
    an exercise-price-range axis. Anything else is refused, because Zscaler tags
    its antidilutive-securities table under the options-outstanding tag with
    award-type members, and summing those would report 12mm options against the
    150,000 it actually has.
    """
    counts = _instants(facts, _OPTION_COUNT_TAGS, "shares", as_of)
    when = _latest(counts)
    if when is None:
        return [], [], []

    # The award roll-forward tags its opening balance with the closing balance's
    # tag, so only the latest instant is the balance outstanding now.
    at_date = {key[1]: f for key, f in counts.items() if key[0] == when}
    strikes = {
        key[1]: f
        for key, f in _instants(facts, _OPTION_STRIKE_TAGS, "USD/shares", as_of).items()
        if key[0] == when
    }

    banded = {k: f for k, f in at_date.items() if _member(k, _BAND_AXIS) is not None}
    if banded and all(k in strikes for k in banded):
        blocks = [
            AwardBlock(
                kind="option",
                count=f.value / _MM,
                strike=strikes[k].value,
                source_tag=f"{f.tag} [{_member(k, _BAND_AXIS)}]",
            )
            for k, f in sorted(banded.items(), key=lambda kv: strikes[kv[0]].value)
        ]
        return (
            blocks,
            [
                f"Options at {when} are taken by exercise-price band, {len(blocks)} of "
                "them, each tested against the price on its own. Collapsing them into "
                "one weighted-average strike would credit the company with exercise "
                "proceeds from the bands nobody would exercise, buying back shares "
                "that would never be bought, and so understate dilution."
            ],
            [],
        )

    band_note = []
    if banded:
        band_note.append(
            f"{len(banded)} exercise-price bands are tagged at {when} but not all of "
            "them carry a strike, so the single weighted-average strike is used "
            "instead. Any band that is out of the money is then hidden inside the "
            "average, contributing phantom exercise proceeds, and dilution here is "
            "understated to that extent."
        )

    plain = at_date.get(())
    if plain is None:
        return (
            [],
            band_note,
            [
                f"options outstanding at {when} are tagged only by dimension "
                f"({len(at_date)} contexts, none undimensioned), and dimensioned "
                "option rows cannot be told apart from an antidilutive-securities "
                "table that reuses the same tag"
            ],
        )
    strike = strikes.get(())
    if strike is None:
        return (
            [],
            band_note,
            [
                f"options outstanding of {plain.value / _MM:,.2f}mm are tagged at "
                f"{when} but no weighted-average exercise price is tagged with them, "
                "and a strike is never guessed"
            ],
        )

    return (
        [
            AwardBlock(
                kind="option",
                count=plain.value / _MM,
                strike=strike.value,
                source_tag=plain.tag,
            )
        ],
        band_note
        + [
            f"Options at {when}: {plain.value / _MM:,.2f}mm at a single "
            f"weighted-average exercise price of {strike.value:,.2f}, the filer "
            "tagging no exercise-price bands to split it by."
        ],
        [],
    )


def _rsu_blocks(
    facts: list[DimensionedFact], as_of: date
) -> tuple[list[AwardBlock], list[str]]:
    """Unvested units at the latest reported date, summed across award types.

    An undimensioned total wins outright where the filer tags one. Otherwise the
    award-type rows are the only figures there are and they are summed, which is
    right where the members are distinct populations and overstates where they
    overlap. The members are named in the note so that can be checked against
    the footnote rather than taken on trust.
    """
    counts = _instants(facts, _RSU_COUNT_TAGS, "shares", as_of)
    when = _latest(counts)
    if when is None:
        return [], []

    at_date = {key[1]: f for key, f in counts.items() if key[0] == when}
    plain = at_date.get(())
    if plain is not None:
        return (
            [AwardBlock(kind="rsu", count=plain.value / _MM, strike=None, source_tag=plain.tag)],
            [f"Unvested units at {when}: {plain.value / _MM:,.2f}mm, tagged as one total."],
        )

    blocks = [
        AwardBlock(
            kind="rsu",
            count=f.value / _MM,
            strike=None,
            source_tag=f"{f.tag} [{'; '.join(m for _, m in k)}]",
        )
        for k, f in sorted(at_date.items(), key=lambda kv: -kv[1].value)
    ]
    total = sum(b.count for b in blocks)
    if len(blocks) == 1:
        return blocks, [
            f"Unvested units at {when}: {total:,.2f}mm, tagged under one award type."
        ]
    members = ", ".join(m for k in sorted(at_date) for _, m in k)
    return blocks, [
        f"Unvested units at {when}: {total:,.2f}mm, summed across {len(blocks)} award "
        f"types ({members}) because the filer tags no combined total. Check the "
        "footnote before relying on this: a member whose label spans two award types "
        "alongside a member for one of them would be counted twice."
    ]


# --------------------------------------------------------------------------- #
# The method
# --------------------------------------------------------------------------- #


def treasury_stock_shares(
    awards: list[AwardBlock], price: float, basic: float
) -> tuple[float, float, list[str]]:
    """Fully diluted shares under ASC 260, at a spot price.

    Returns the fully diluted count, the net new shares options contribute, and
    the notes. Every figure is in millions of shares; ``price`` is dollars.

    Options are assumed exercised only where the price exceeds the strike. The
    proceeds buy shares back at that same price, so the block adds
    ``count x (1 - strike / price)``. Below the strike the option is
    antidilutive and adds exactly zero: the arithmetic would otherwise return a
    negative number and a worthless option would shrink the share count.

    ASC 260 runs this on the average market price over the reporting period,
    because diluted EPS divides a period's earnings by a period's average
    shares. Here the count is being struck at an instant and multiplied by
    today's price, so today's price is what decides which options are in the
    money. Using a trailing average for the test and spot for the valuation
    would mix two dates in one number.

    RSUs have no strike, so there are no proceeds and no buyback. Each unit adds
    one share. Any forfeiture haircut has already been applied to the counts by
    the caller: it is an assumption, not part of the method.
    """
    if price <= 0:
        raise NotMeaningfulError(
            f"a share price of {price} cannot support the treasury stock method: "
            "the exercise proceeds buy back shares at the market price, and at zero "
            "or below there is no such price"
        )

    net_new = 0.0
    rsu_shares = 0.0
    in_money: list[AwardBlock] = []
    out_money: list[AwardBlock] = []

    for a in awards:
        if a.kind == "rsu":
            rsu_shares += a.count
            continue
        if a.strike is None:
            raise MissingDataError(
                "option exercise price",
                tags_tried=[a.source_tag],
                hint=(
                    "an option block reached the treasury stock method with no "
                    "strike; without one it cannot be tested for being in the money "
                    "and a strike is never assumed"
                ),
            )
        if price > a.strike:
            net_new += a.count * (1.0 - a.strike / price)
            in_money.append(a)
        else:
            out_money.append(a)

    notes: list[str] = []
    if in_money:
        gross = sum(a.count for a in in_money)
        notes.append(
            f"Treasury stock method at {price:,.2f}: {gross:,.2f}mm options in the "
            f"money across {len(in_money)} block(s) are exercised and the proceeds "
            f"buy back {gross - net_new:,.2f}mm shares, leaving {net_new:,.2f}mm net "
            "new shares."
        )
    if out_money:
        gross = sum(a.count for a in out_money)
        strikes = ", ".join(f"{a.strike:,.2f}" for a in out_money)
        notes.append(
            f"{gross:,.2f}mm options across {len(out_money)} block(s) are out of the "
            f"money at {price:,.2f} (strikes: {strikes}) and add nothing. They are "
            "dropped rather than netted: an antidilutive option cannot reduce the "
            "share count."
        )
    if rsu_shares:
        notes.append(
            f"Unvested units add their full {rsu_shares:,.2f}mm. An RSU has no "
            "exercise price, so there are no proceeds to buy shares back with and "
            "the treasury stock offset is zero by construction."
        )

    return basic + net_new + rsu_shares, net_new, notes


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

_CONVERTIBLE_NOTE = (
    "Convertible notes are not in this count. Under ASU 2020-06 the EV bridge "
    "decides whether an instrument is a debt claim or converted equity, and adding "
    "its shares here as well would count the same claim twice. Where the bridge "
    "carries an in-the-money convertible as equity, its conversion shares sit in "
    "diluted WASO but not in this count, so that pairing needs them added back."
)


def _waso_count(
    ticker: str,
    fin: Financials,
    *,
    method: str,
    accession: str | None = None,
    notes: list[str] | None = None,
    flags: list[str] | None = None,
) -> ShareCount:
    """The trailing diluted weighted average, used as the whole count."""
    return ShareCount(
        ticker=ticker,
        basic_outstanding=None,
        option_awards=[],
        rsu_awards=[],
        net_new_from_options=0.0,
        rsu_shares=0.0,
        fully_diluted=fin.diluted_shares,
        diluted_waso=fin.diluted_shares,
        as_of=fin.as_of,
        accession=accession,
        method=method,
        notes=notes or [],
        flags=flags or [],
    )


def build_share_count(
    ticker: str,
    price: float,
    fin: Financials,
    assumptions: Assumptions,
    client,
) -> ShareCount:
    """Build the share count equity value should be struck on.

    ``assumptions.dilution.method`` picks the default. Under ``'waso'`` the
    trailing diluted weighted average is used as it stands and no filing is
    fetched. Under ``'treasury_stock'`` the latest instance document is read and
    the count is built from shares outstanding plus in-the-money awards.

    The treasury stock path degrades to the weighted average rather than
    guessing. If the instance document cannot be read, if no share count is
    tagged, or if the award tags are absent or unusable, the result carries
    ``method='diluted WASO fallback'`` and a flag naming exactly what was
    missing. A weighted average that is known to be a weighted average is worth
    more than a treasury stock count built on an invented strike.

    The weighted average is reported alongside either way, with the difference
    in shares and in percent, because the point of this module is the size of
    that difference and a reader should be able to see it.
    """
    cfg = assumptions.dilution

    if cfg.method == "waso":
        return _waso_count(
            ticker,
            fin,
            method="diluted WASO (configured)",
            notes=[
                "Trailing diluted weighted-average shares are used as configured. "
                "This is an average over the past twelve months, so it sits below "
                "the count outstanding today at any company issuing steadily. Set "
                "dilution.method to 'treasury_stock' for a point-in-time count."
            ],
        )

    try:
        facts, accession, filed = client.instance_facts(ticker)
    except TechvalError as exc:
        return _waso_count(
            ticker,
            fin,
            method="diluted WASO fallback",
            flags=[f"the XBRL instance document could not be read: {exc}"],
        )

    try:
        basic, basic_date, _basic_tag, notes = _basic_outstanding(facts, filed)
    except TechvalError as exc:
        return _waso_count(
            ticker,
            fin,
            method="diluted WASO fallback",
            accession=accession,
            flags=[f"shares outstanding are not tagged in {accession}: {exc}"],
        )

    options, option_notes, option_flags = _option_blocks(facts, filed)
    rsus, rsu_notes = _rsu_blocks(facts, filed)

    if option_flags:
        return _waso_count(
            ticker,
            fin,
            method="diluted WASO fallback",
            accession=accession,
            flags=option_flags,
        )
    if not options and not rsus:
        return _waso_count(
            ticker,
            fin,
            method="diluted WASO fallback",
            accession=accession,
            flags=[
                f"{accession} tags no options outstanding and no unvested units, so "
                "there is nothing to dilute shares outstanding with"
            ],
        )

    notes += option_notes

    gross_rsu = sum(a.count for a in rsus)
    if not cfg.include_rsus:
        if rsus:
            notes.append(
                f"Unvested units of {gross_rsu:,.2f}mm are excluded by assumption. "
                "They have no strike and would each add a full share, so this count "
                "understates dilution by that amount."
            )
        rsus = []
    else:
        notes += rsu_notes
        rate = cfg.assumed_forfeiture_rate
        if rate:
            rsus = [replace(a, count=a.count * (1.0 - rate)) for a in rsus]
            notes.append(
                f"Unvested units are cut by the assumed forfeiture rate of {rate:.1%}, "
                f"from {gross_rsu:,.2f}mm to {sum(a.count for a in rsus):,.2f}mm. The "
                "haircut reaches unvested awards only. Options outstanding include "
                "vested ones that no longer can be forfeited, and the filing does not "
                "always separate them, so options are left whole. ASC 260 itself does "
                "not haircut for forfeitures."
            )

    fully_diluted, net_new, tsm_notes = treasury_stock_shares(
        [*options, *rsus], price, basic
    )
    notes += tsm_notes

    gap = fully_diluted - fin.diluted_shares
    pct = gap / fin.diluted_shares if fin.diluted_shares else 0.0
    notes.append(
        f"Fully diluted {fully_diluted:,.2f}mm against a trailing diluted "
        f"weighted average of {fin.diluted_shares:,.2f}mm, a difference of "
        f"{gap:+,.2f}mm shares or {pct:+.2%}. The two move apart for two reasons "
        f"pulling opposite ways: the weighted average is struck over a past window "
        f"and so sits below the {basic:,.2f}mm outstanding today, while it already "
        "carries a share of the award overhang that is added here in full."
    )
    notes.append(_CONVERTIBLE_NOTE)

    return ShareCount(
        ticker=ticker,
        basic_outstanding=basic,
        option_awards=options,
        rsu_awards=rsus,
        net_new_from_options=net_new,
        rsu_shares=sum(a.count for a in rsus),
        fully_diluted=fully_diluted,
        diluted_waso=fin.diluted_shares,
        as_of=basic_date,
        accession=accession,
        method="treasury stock method",
        notes=notes,
        flags=[],
    )
