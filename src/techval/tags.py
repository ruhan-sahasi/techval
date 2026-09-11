"""us-gaap tag ladders.

Ordered by how specific and how trustworthy each tag is, most specific first.
A tuple entry is a composite: its tags are summed over periods where all of them
are present.

The ladders are longer than the obvious ones because real filers do not use the
obvious tags. Observed in the fixtures committed with this package: Datadog
carries its convertible notes under ``ConvertibleLongTermNotesPayable`` and has
no ``LongTermDebt*`` tag at all; CrowdStrike reports revenue under the
*Including*AssessedTax variant; every one of the seven companies reports short
term investments under ``AvailableForSaleSecuritiesDebtSecuritiesCurrent`` rather
than ``ShortTermInvestments``.
"""

from __future__ import annotations

# --- income statement -------------------------------------------------------

REVENUE = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueServicesNet",
]

COST_OF_REVENUE = [
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfServices",
]

GROSS_PROFIT = ["GrossProfit"]

EBIT = ["OperatingIncomeLoss"]

NET_INCOME = [
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
]

PRETAX_INCOME = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
]

TAX_EXPENSE = ["IncomeTaxExpenseBenefit"]

INTEREST_EXPENSE = [
    "InterestExpenseDebt",
    "InterestExpense",
    "InterestExpenseNonoperating",
    "InterestIncomeExpenseNet",
]

INTEREST_INCOME = [
    "InvestmentIncomeInterest",
    "InterestIncomeOther",
    "InvestmentIncomeInterestAndDividend",
]

# --- cash flow --------------------------------------------------------------

# The combined tag first; then the components summed, for filers that publish
# depreciation and amortisation separately but never together.
#
# The composite is not a guarantee. It fires only where every component is
# tagged over the same periods, and some filers tag neither a combined line nor
# a separate depreciation one: CrowdStrike reports amortisation of intangibles
# but no depreciation tag at all, so its EBITDA genuinely cannot be built from
# company facts and the comps table flags it rather than inventing a figure.
DA = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "DepreciationAndAmortization",
    ("Depreciation", "AmortizationOfIntangibleAssets"),
    "DepreciationNonproduction",
]

SBC = [
    "ShareBasedCompensation",
    "AllocatedShareBasedCompensationExpense",
]

CAPEX = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsForCapitalImprovements",
]

CFO = ["NetCashProvidedByUsedInOperatingActivities"]

# --- shares -----------------------------------------------------------------

DILUTED_SHARES = ["WeightedAverageNumberOfDilutedSharesOutstanding"]

BASIC_SHARES = ["WeightedAverageNumberOfSharesOutstandingBasic"]

EPS_DILUTED = ["EarningsPerShareDiluted"]

# --- balance sheet ----------------------------------------------------------

CASH = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
]

SHORT_TERM_INVESTMENTS = [
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent",
    "MarketableSecuritiesCurrent",
    "ShortTermInvestments",
    "AvailableForSaleSecuritiesCurrent",
    "OtherShortTermInvestments",
]

RECEIVABLES = [
    "AccountsReceivableNetCurrent",
    "ReceivablesNetCurrent",
    "AccountsReceivableGrossCurrent",
]

LONG_TERM_INVESTMENTS = [
    "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent",
    "MarketableSecuritiesNoncurrent",
    "LongTermInvestments",
]

# --- debt -------------------------------------------------------------------
#
# Straight borrowings, in three ladders: the non-current line, the current line,
# and a combined figure for filers who publish only a total. Convertibles are
# resolved separately, because whether they belong in debt at all is a judgment
# the EV bridge has to make explicitly.
#
# These four lists are the most consequential in this file, and the order inside
# them is a judgment rather than a lookup. Three rules set it, and each was
# written against a filer that breaks the obvious answer.
#
# **Pure debt before debt-and-leases.** ``LongTermDebtAndCapitalLeaseObligations``
# is what Verizon, AT&T, Comcast, Accenture and Warner Bros Discovery tag their
# long-term debt line with, and its definition is "long-term debt AND lease
# obligation, noncurrent". The figure therefore already contains the finance
# leases ``FINANCE_LEASE_NONCURRENT`` would add a second time, so
# ``INCLUDES_FINANCE_LEASES`` marks it and ``financials`` suppresses the separate
# lease line whenever one of these wins. Ranking the pure tag above it means the
# suppression only ever fires on filers who genuinely bundle the two.
#
# **The broadest current line first.** ``DebtCurrent`` is "debt and lease
# obligation classified as current" and is the whole of the current-liability
# debt caption. ``LongTermDebtCurrent`` is only the current maturities inside it.
# Qualcomm at 2026-06-28 tags DebtCurrent 2,489mm for its balance-sheet line and
# LongTermDebtCurrent 1,991mm for the current-maturities component, with the
# 498mm balance commercial paper; ranking the narrow tag first read Qualcomm's
# short-term debt 20% light.
#
# **``LongTermDebt`` is a non-current line as often as it is a total, so it sits
# at the foot of the non-current ladder rather than at the head of the combined
# one.** The taxonomy does not settle it. Microsoft uses it as a total:
# LongTermDebtNoncurrent 31,067mm, LongTermDebtCurrent 9,227mm and LongTermDebt
# 40,294mm at 2026-06-30, which adds up. Adobe and Qualcomm tag the non-current
# line on the face of the balance sheet with it, 4,802mm and 12,781mm, with a
# separate current debt line above. Placing it last in the NON-CURRENT ladder
# resolves both camps, because every filer who uses it as a total also tags
# LongTermDebtNoncurrent or LongTermDebtAndCapitalLeaseObligations, which rank
# above it and win first, and ``DEBT_TOTAL_BY_DEFINITION`` below catches the
# filer who tags an explicit non-current zero instead.
#
# Counts below are filers out of the 110-name TMT seed universe reporting that
# concept at their own latest balance-sheet date, measured on 2026-09-11.
DEBT_NONCURRENT = [
    "LongTermDebtNoncurrent",  # 50 filers
    "LongTermDebtAndCapitalLeaseObligations",  # 19 filers, leases inside
    "LongTermDebt",  # 39 filers, see the note above
    "LongTermNotesPayable",  # 3 filers
]

DEBT_CURRENT = [
    "DebtCurrent",  # 17 filers
    "LongTermDebtAndCapitalLeaseObligationsCurrent",  # 10 filers, leases inside
    "LongTermDebtCurrent",  # 33 filers
    "ShortTermBorrowings",  # 11 filers
    "NotesPayableCurrent",  # 3 filers
    "LinesOfCreditCurrent",  # 2 filers
    "OtherShortTermBorrowings",  # 1 filer
]

# Consulted when the split resolves nothing, and as the tie-break when only one
# of the two legs resolves. Never added to the split: a combined figure overlaps
# its own components, and summing the two triple counts the current maturities.
DEBT_COMBINED = [
    "DebtLongtermAndShorttermCombinedAmount",  # 8 filers
    # 5 filers, leases inside
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "DebtAndCapitalLeaseObligations",  # 8 filers, leases inside
    "NotesPayable",  # 3 filers
    # Here as well as in DEBT_NONCURRENT, and the duplication is the point. A
    # ladder reaches this list only when one of the two legs failed, and the only
    # way to arrive here with LongTermDebt unused is that the non-current leg
    # already resolved from a strictly non-current concept, which makes this
    # figure the total rather than the same line again. PayPal at 2026-03-31:
    # LongTermDebtNoncurrent 9,409mm, LongTermDebt 10,876mm and no current debt
    # concept of any kind, because the 1,467mm current portion is tagged under a
    # dimension the companyfacts API does not return.
    "LongTermDebt",
]

# Concepts whose NAME says "debt and capital lease obligations". A figure read off
# one of them already contains the finance leases the lease ladders would add a
# second time, so ``financials`` folds the lease into the debt line rather than
# counting it twice, and the EV bridge labels the row for what is inside it.
#
# Verizon supplies the arithmetic. Its balance sheet carries 139,532mm of
# long-term debt and 18,618mm of current debt at 2025-12-31, 158,150mm together,
# while its fair-value note gives "debt excluding capital leases" as 155,639mm.
# The 2,511mm difference is exactly the FinanceLeaseLiabilityCurrent 943mm plus
# FinanceLeaseLiabilityNoncurrent 1,568mm the lease ladders resolve separately,
# so the two really are the same obligation twice.
INCLUDES_FINANCE_LEASES = frozenset(
    {
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "DebtAndCapitalLeaseObligations",
    }
)

# Concepts whose us-gaap definition ADMITS a lease without requiring one.
# ``DebtCurrent`` reads "amount of debt and lease obligation, classified as
# current", and filers use it both ways, so it cannot decide the question alone.
#
#   Micron, 2026-05-28. DebtCurrent is 582mm and FinanceLeaseLiabilityCurrent is
#   582mm as well: the whole of Micron's current debt caption is finance lease.
#   Adding the lease again reports 6,304mm of debt against the 5,722mm Micron
#   itself tags as DebtAndCapitalLeaseObligations for the same date.
#
#   Oracle, 2026-05-31. DebtCurrent 7,199mm is "notes payable and other
#   borrowings, current", and the lease note's extensible enumeration puts the
#   620mm current finance lease in other current liabilities instead. Suppressing
#   it would lose a real obligation.
#
# So one of these swallows its lease only where the filer has ALREADY been shown
# to bundle the two, by resolving the other leg of the same balance sheet from a
# concept in ``INCLUDES_FINANCE_LEASES``. Micron's non-current line is
# LongTermDebtAndCapitalLeaseObligations and Oracle's is LongTermDebtNoncurrent,
# whose own definition ends "Excludes lease obligation", so the test separates
# the two on the filer's tagging rather than on a guess.
MAY_INCLUDE_FINANCE_LEASES = frozenset(
    {"DebtCurrent", "DebtLongtermAndShorttermCombinedAmount"}
)

# ``LongTermDebt`` is in ``DEBT_NONCURRENT`` because most filers tag their
# non-current line with it, and the taxonomy does not settle whether current
# maturities are inside: the definition reads "amount, after deduction of
# unamortized premium (discount) and debt issuance cost, of long-term debt.
# Excludes lease obligation", and stops there. Where the filer states a
# non-current balance of ZERO at the same date the figure can only be a total,
# and adding a current leg to it would report the debt twice. Western Digital's
# 2026 10-K is the case: LongTermDebtNoncurrent 0, LongTermDebtCurrent 1,052mm
# and LongTermDebt 1,052mm, where summing the two slots gives 2,104mm of debt
# against 1,052mm on the balance sheet.
DEBT_TOTAL_BY_DEFINITION = frozenset({"LongTermDebt"})

# The concepts that mean non-current long-term debt and nothing else. An explicit
# zero under one of these at the balance-sheet date is the filer saying it has
# none, and that is what disambiguates the entry above.
DEBT_STRICTLY_NONCURRENT = (
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
)

# Concepts that mean the WHOLE of a filer's borrowings. Read only by the control
# that watches for a lease counted twice, so an entry here has to be a true
# total or the control fires on a balance sheet that is already right.
#
# ``LongTermDebt`` is deliberately absent even though it is a total for many
# filers. Cisco's 2026 10-K reports LongTermDebt 22,872mm, its long-term debt
# current and non-current together, against 29,533mm of borrowings once the
# 6,661mm of commercial paper inside the 10,161mm short-term debt line is
# counted. A concept that excludes commercial paper is not a ceiling on debt.
DEBT_TOTALS = [
    "DebtLongtermAndShorttermCombinedAmount",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "DebtAndCapitalLeaseObligations",
    "NotesPayable",
]

# Concepts the debt control compares the resolved figure against, to catch a
# ladder that missed a line. Each is bounded ABOVE by the filer's total
# borrowings whatever the ladder did, so one of them reporting materially more
# than was resolved is proof that something was missed.
#
# ``DebtInstrumentCarryingAmount`` is deliberately absent from this list and from
# ``DEBT_COMBINED``. It is a footnote concept whose scope varies by filer:
# Microsoft reports 46,136mm of it against a 40,294mm balance-sheet total because
# the note is struck before unamortised discount, and Meta reports 84,000mm of
# principal against 83,664mm of carrying value. A par figure is not a balance,
# and a control built on one fires on companies whose debt is already right.
DEBT_CROSSCHECK = DEBT_TOTALS + [
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
    "SeniorNotes",
    "SecuredDebt",
    "ConvertibleDebt",
]

CONVERTIBLE_NONCURRENT = [
    "ConvertibleLongTermNotesPayable",
    "ConvertibleDebtNoncurrent",
    "ConvertibleSubordinatedDebtNoncurrent",
]

CONVERTIBLE_CURRENT = [
    "ConvertibleNotesPayableCurrent",
    "ConvertibleDebtCurrent",
]

OPERATING_LEASE_NONCURRENT = ["OperatingLeaseLiabilityNoncurrent"]
OPERATING_LEASE_CURRENT = ["OperatingLeaseLiabilityCurrent"]
FINANCE_LEASE_NONCURRENT = [
    "FinanceLeaseLiabilityNoncurrent",
    "CapitalLeaseObligationsNoncurrent",
]
FINANCE_LEASE_CURRENT = [
    "FinanceLeaseLiabilityCurrent",
    "CapitalLeaseObligationsCurrent",
]

# Total lease liabilities, current and non-current together. Read only when the
# split resolves nothing, on the same rule the debt ladders follow, because a
# combined figure and its own components must never be added to each other.
#
# Not an edge case. Microsoft's 2026 10-K tags FinanceLeaseLiability at 66,594mm
# and neither of the split concepts at all, so a ladder without this entry reads
# the largest finance lease balance in the seed universe as zero: bigger than
# Microsoft's entire 40,294mm of straight debt, and the data-centre build is the
# reason it is there.
FINANCE_LEASE_COMBINED = ["FinanceLeaseLiability", "CapitalLeaseObligations"]
OPERATING_LEASE_COMBINED = ["OperatingLeaseLiability"]

OPERATING_LEASE_COST = [
    "OperatingLeaseCost",
    "OperatingLeasesRentExpenseNet",
    "LeaseCost",
]

NCI = ["MinorityInterest", "MinorityInterestInOperatingPartnerships"]

# Mezzanine first. Redeemable preferred lives in temporary equity at its
# carrying amount; ``PreferredStockValue`` is the par value inside permanent
# equity, often a few thousand dollars against a nine-figure liquidation
# preference. Par-first would read a real preferred stack as roughly zero.
# The temporary-equity tag can also carry redeemable NCI, which mislabels the
# line but not the bridge: either way it is a claim ahead of the common and
# belongs in enterprise value.
PREFERRED = [
    "TemporaryEquityCarryingAmountAttributableToParent",
    "PreferredStockValueOutstanding",
    "PreferredStockValue",
]

CURRENT_ASSETS = ["AssetsCurrent"]
CURRENT_LIABILITIES = ["LiabilitiesCurrent"]
TOTAL_ASSETS = ["Assets"]
EQUITY = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]

DEFERRED_REVENUE_CURRENT = [
    "ContractWithCustomerLiabilityCurrent",
    "DeferredRevenueCurrent",
]
DEFERRED_REVENUE_NONCURRENT = [
    "ContractWithCustomerLiabilityNoncurrent",
    "DeferredRevenueNoncurrent",
]

# --- concepts kept past their retirement, and why ---------------------------
#
# ``tests/test_tags.py`` reads ``tests/fixtures/tag_census.json``, a record of
# which of these concepts the TMT seed universe still reports at its own latest
# balance-sheet date, and fails when a ladder names a concept nobody uses any
# more. That test exists because of Verizon: it stopped tagging
# ``LongTermDebtNoncurrent`` in 2013, the ladder kept asking for it until 2026,
# and 143 billion dollars of long-term debt read as zero with nothing on the page
# to say so. A retired tag does not raise. It resolves to the default.
#
# Not every retired concept is dead weight, though, and that is what this map is
# for. ``--as-of`` pushes a knowledge date through the whole engine, so a
# valuation dated 2018 needs the concepts a 2018 filing used and will find
# nothing else. Every entry below is a concept no filer in the universe reports
# TODAY, kept deliberately, with the reason it earns its place. A concept that is
# retired and not named here is a bug, and a concept named here that has come
# back into live use should be taken out, so the exemption list cannot rot
# either.
#
# The three debt resolution ladders take no exemptions at all. Everywhere else a
# retired concept costs a missing row; there it costs an enterprise value that
# looks right and is not.
RETIRED_BUT_KEPT: dict[str, str] = {
    "SalesRevenueNet": (
        "the pre-ASC-606 top line, last used in the universe at 2018-12-31. A "
        "run dated before the 2019 adoption resolves revenue on this and on "
        "nothing else"
    ),
    "SalesRevenueServicesNet": (
        "the pre-ASC-606 services top line, same 2018 cutover, and the tag a "
        "software filer of that era used in place of SalesRevenueNet"
    ),
    "CostOfServices": (
        "the pre-ASC-606 cost line that pairs with SalesRevenueServicesNet. "
        "Gross profit for a 2018 run needs the pair or it needs neither"
    ),
    "OperatingLeasesRentExpenseNet": (
        "rent expense as it was disclosed before ASC 842 put the liability on "
        "the balance sheet in 2019. It is the only route to EBITDAR for a "
        "valuation dated earlier than that"
    ),
    "CapitalLeaseObligations": (
        "ASC 842 renamed capital leases to finance leases in 2019. The universe "
        "last reported this in 2024 and a pre-2019 run needs it"
    ),
    "CapitalLeaseObligationsCurrent": "the current half of the same 2019 rename",
    "CapitalLeaseObligationsNoncurrent": "the non-current half of the same rename",
    "AvailableForSaleSecuritiesCurrent": (
        "the pre-2019 marketable securities concept, superseded by the "
        "DebtSecuritiesAvailableForSale family. Leaving it out would read a 2018 "
        "securities portfolio as zero and overstate enterprise value by it"
    ),
    "ConvertibleNotesPayableCurrent": (
        "reported as recently as 2025-12-31 in the universe and simply not "
        "current at anyone's latest balance-sheet date, which is what a "
        "convertible that matured looks like rather than what a retired tag "
        "looks like"
    ),
    "ConvertibleSubordinatedDebtNoncurrent": (
        "last used in 2016, and kept because subordinated converts are rare "
        "rather than extinct. Reading one as zero would understate the claims "
        "ahead of the common by the whole instrument"
    ),
    "ConvertibleDebt": (
        "a cross-check entry rather than a resolution entry. It is read only to "
        "test a resolved figure from below, so an extra concept there can widen "
        "the net and cannot change a number"
    ),
    "DepreciationNonproduction": (
        "the last rung of the D&A ladder, reported at 2025-12-31 by one filer "
        "and read only where every rung above it is absent"
    ),
    "MinorityInterestInOperatingPartnerships": (
        "the partnership form of non-controlling interest, which is a REIT and "
        "tower-operator disclosure rather than a retired one"
    ),
    "PaymentsForCapitalImprovements": (
        "the last rung of the capex ladder, and the one a filer with no "
        "PaymentsToAcquirePropertyPlantAndEquipment line falls back to"
    ),
}
