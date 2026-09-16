"""The filing facts the readers are asked for, and the filings they are asked of.

The questions come in two groups:

- **Deals.** Each of the nine committed merger announcements, all 8-Ks, is asked
  for five deal terms.
- **KPIs.** Each committed 10-K is asked, in its Item 7, for four operating
  figures that filers state in prose rather than tag.

A fact a filing does not state is still a task: the right answer is
``not_stated``, and a reader that invents a value for it is wrong.

The deal terms follow what an announcement actually states. A mixed deal gives
a cash amount and an exchange ratio in one sentence and never a single price,
so the cash leg and the ratio are asked separately. A collar makes the ratio
depend on the buyer's price at closing, and the honest answer to the ratio
question is then ``ambiguous``.

KPI tasks exist only for 10-K texts that are committed. A 10-K that is not
committed is reported as unresolved rather than silently dropped;
``tests/fixtures/rag/record_filing_text.py`` records the missing ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .store import Document, FilingStore


@dataclass(frozen=True)
class Metric:
    name: str
    kind: str
    unit: str
    label: str
    question: str
    definition: str
    terms: tuple[str, ...]


METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        Metric(
            "cash_per_share", "deal", "usd_per_share", "Cash per share",
            "How much cash does each share of the target's common stock receive in the merger, in US dollars per share?",
            "the cash each target common share is converted into; if the consideration has no cash leg, not_stated",
            ("per share", "cash", "merger consideration", "converted", "right to receive", "without interest"),
        ),
        Metric(
            "consideration_form", "deal", "text", "Form of consideration",
            "Is the merger consideration for each target share cash only, stock only, or a mix of cash and stock?",
            "text_value cash, stock or mixed, judged from what each target common share is converted into",
            ("merger consideration", "cash", "shares", "common stock", "converted", "right to receive"),
        ),
        Metric(
            "exchange_ratio", "deal", "ratio", "Exchange ratio",
            "How many shares of the acquirer does each target share receive?",
            "the fixed number of acquirer shares per target share; ambiguous when a collar or formula sets it at closing; not_stated when there is no stock leg",
            ("exchange ratio", "shares", "common stock", "fraction", "collar", "converted"),
        ),
        Metric(
            "agreement_date", "deal", "date", "Agreement date",
            "On what date was the merger agreement entered into?",
            "the date the parties entered into the agreement and plan of merger, as text_value YYYY-MM-DD",
            ("entered into", "agreement and plan of merger", "dated as of", "merger agreement"),
        ),
        Metric(
            "acquirer", "deal", "text", "Acquirer",
            "Which company does the merger agreement name as the buyer's parent?",
            "the parent company party to the merger agreement, written exactly as the filing names it; not a merger subsidiary",
            ("parent", "merger sub", "agreement and plan of merger", "by and among", "acquire"),
        ),
        Metric(
            "customers", "kpi", "count", "Customers",
            "How many customers does the company state it had at the end of the fiscal year, in total?",
            "the total customer count; a count of customers above a spending threshold is a different figure",
            ("customers", "customer count", "approximately"),
        ),
        Metric(
            "arr", "kpi", "usd", "Annual recurring revenue",
            "What annual recurring revenue does the company state for the whole company at the end of the fiscal year, in US dollars?",
            "the company's total ARR; a threshold used to count customers, such as ARR of $100,000 or more, is not the company's ARR",
            ("annual recurring revenue", "arr", "annual run-rate revenue"),
        ),
        Metric(
            "net_revenue_retention", "kpi", "percent", "Net revenue retention",
            "What net revenue retention rate, or dollar-based net retention rate, does the company state for the end of the fiscal year, in percent?",
            "the net or dollar-based net retention rate for the latest period, in percentage points",
            ("net retention", "net revenue retention", "dollar-based", "retention rate"),
        ),
        Metric(
            "subscribers", "kpi", "count", "Subscribers",
            "How many paying subscribers, members or subscriptions does the company state it had in total at the end of the period?",
            "the total count of paying subscribers or members; net additions in a period are not the total",
            ("subscribers", "paid memberships", "members", "subscriptions", "customers"),
        ),
    )
}

DEAL_METRICS = ("cash_per_share", "consideration_form", "exchange_ratio", "agreement_date", "acquirer")
KPI_METRICS = ("customers", "arr", "net_revenue_retention", "subscribers")

# The announcing 8-K of each committed deal. SPLK's 2021 8-K and ZEN's 2021 S-4
# are committed too, as decoys for the precedent reader, and are not tasks.
DEALS: tuple[tuple[str, str], ...] = (
    ("IRDM", "0001104659-26-078482"),
    ("MNDT", "0001104659-22-031786"),
    ("PAYO", "0000950103-26-008945"),
    ("RAMP", "0001104659-26-062908"),
    ("ROKU", "0001140361-26-025115"),
    ("SLAB", "0001193125-26-036712"),
    ("SPLK", "0001104659-23-102594"),
    ("WORK", "0001193125-20-307385"),
    ("ZEN", "0001193125-22-181655"),
)

KPI_DOCUMENTS: dict[str, str] = {
    "DDOG": "filing_text_DDOG_2025.json.gz",
    "NET": "rag/filing_text_NET_2025.json.gz",
    "NFLX": "rag/filing_text_NFLX_2025.json.gz",
    "TMUS": "rag/filing_text_TMUS_2025.json.gz",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    ticker: str
    metric: str
    accession: str
    form: str
    filed: date
    item: str | None
    url: str | None
    target_name: str | None

    @property
    def as_of(self) -> date:
        return self.filed


def _task(doc: Document, metric: str, item: str | None) -> Task:
    return Task(
        task_id=f"{doc.ticker}:{metric}",
        kind=METRICS[metric].kind,
        ticker=doc.ticker,
        metric=metric,
        accession=doc.accession,
        form=doc.form,
        filed=doc.filed,
        item=item,
        url=doc.url,
        target_name=doc.entity_name,
    )


def build_tasks(store: FilingStore) -> tuple[list[Task], list[str]]:
    tasks: list[Task] = []
    unresolved: list[str] = []
    for ticker, accession in DEALS:
        doc = store.get(accession)
        if doc is None:
            unresolved.append(f"{ticker}: merger filing {accession} is not committed")
            continue
        tasks.extend(_task(doc, metric, None) for metric in DEAL_METRICS)
    for ticker, path in KPI_DOCUMENTS.items():
        doc = store.by_path(path)
        if doc is None:
            unresolved.append(
                f"{ticker}: the 10-K text {path} is not committed; "
                "tests/fixtures/rag/record_filing_text.py records it"
            )
            continue
        tasks.extend(_task(doc, metric, "7") for metric in KPI_METRICS)
    return sorted(tasks, key=lambda t: t.task_id), unresolved
