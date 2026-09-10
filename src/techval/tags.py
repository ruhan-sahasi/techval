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

# The combined tag first; then the components summed, for filers such as
# CrowdStrike that never publish a combined depreciation and amortisation line.
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
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardCompensationCost1",
]

CAPEX = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsForCapitalImprovements",
]

CFO = ["NetCashProvidedByUsedInOperatingActivities"]

# --- shares -----------------------------------------------------------------

DILUTED_SHARES = ["WeightedAverageNumberOfDilutedSharesOutstanding"]

BASIC_SHARES = [
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfSharesOutstanding",
]

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
    "DebtSecuritiesAvailableForSaleCurrent",
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

# Straight borrowings. Convertibles are resolved separately: whether they belong
# in debt at all is a judgment the EV bridge has to make explicitly.
DEBT_NONCURRENT = [
    "LongTermDebtNoncurrent",
    "LongTermNotesPayable",
    "SecuredDebtNoncurrent",
    "NotesPayableNoncurrent",
]

DEBT_CURRENT = [
    "LongTermDebtCurrent",
    "NotesPayableCurrent",
    "ShortTermBorrowings",
    "OtherShortTermBorrowings",
]

DEBT_COMBINED = [
    "LongTermDebt",
    "DebtLongtermAndShorttermCombinedAmount",
    "DebtInstrumentCarryingAmount",
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

OPERATING_LEASE_COST = [
    "OperatingLeaseCost",
    "OperatingLeasesRentExpenseNet",
    "LeaseCost",
]

NCI = ["MinorityInterest", "MinorityInterestInOperatingPartnerships"]

PREFERRED = [
    "PreferredStockValue",
    "PreferredStockValueOutstanding",
    "TemporaryEquityCarryingAmountAttributableToParent",
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
