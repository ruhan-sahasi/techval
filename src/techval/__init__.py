"""techval: DCF, trading comps and merger analysis for US-listed technology companies.

Built on filings and market data that cost nothing: SEC EDGAR company facts for
fundamentals, Nasdaq's public quote API for prices, and the US Treasury daily
curve for the risk-free rate.

Every figure the engine reports carries provenance back to a filing, a quote, or
a stated assumption. Nothing is interpolated to fill a gap.
"""

from .errors import (
    ConfigError,
    DataSourceError,
    MissingDataError,
    NotMeaningfulError,
    StaleDataError,
    TechvalError,
)

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "DataSourceError",
    "MissingDataError",
    "NotMeaningfulError",
    "StaleDataError",
    "TechvalError",
    "__version__",
]
