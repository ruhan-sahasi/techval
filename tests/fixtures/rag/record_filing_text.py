"""Record the 10-K texts the filing reader's KPI tasks read: NET, NFLX and TMUS.

Run from the repository root, with the contact address EDGAR's fair-access
policy asks for:

    TECHVAL_SEC_EMAIL=you@example.com .venv/bin/python tests/fixtures/rag/record_filing_text.py

Each file holds the newest 10-K the filer made on or before AS_OF. The text is
stripped of markup by techval.edgar.strip_markup and stored in full, in the
layout of tests/fixtures/filing_text_DDOG_2025.json.gz, plus the document's
URL. The gzip header carries no timestamp, so a re-recording of an unchanged
filing is byte-identical.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import date
from pathlib import Path

from techval.edgar import EdgarClient

AS_OF = date(2026, 9, 11)
FISCAL_YEAR = 2025
TICKERS = ("NET", "NFLX", "TMUS")
HERE = Path(__file__).resolve().parent


def record(client: EdgarClient, ticker: str) -> Path:
    [filing] = client.filings(ticker, forms=("10-K",), limit=1)
    if not str(filing["period"]).startswith(str(FISCAL_YEAR)):
        raise SystemExit(
            f"{ticker}: the newest 10-K on or before {AS_OF} covers {filing['period']}, not fiscal {FISCAL_YEAR}"
        )
    bare = filing["accession"].replace("-", "")
    payload = {
        "retrieved": date.today().isoformat(),
        "source": "SEC EDGAR primary document, markup stripped by techval.edgar.strip_markup, stored verbatim and in full",
        "url": f"https://www.sec.gov/Archives/edgar/data/{client.ticker_to_cik(ticker)}/{bare}/{filing['document']}",
        "ticker": ticker,
        "accession": filing["accession"],
        "form": filing["form"],
        "filed": filing["filed"].isoformat(),
        "period": filing["period"],
        "text": client.filing_text(ticker, filing),
    }
    path = HERE / f"filing_text_{ticker}_{FISCAL_YEAR}.json.gz"
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return path


def main() -> int:
    client = EdgarClient(knowledge_date=AS_OF)
    for ticker in TICKERS:
        print(record(client, ticker))
    return 0


if __name__ == "__main__":
    sys.exit(main())
