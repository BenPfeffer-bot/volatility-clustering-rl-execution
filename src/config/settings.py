"""
Configuration Module

This module handles project-wide configuration settings.
"""

import pandas as pd
from .paths import *

# Dow Jones Global Titans 50 Index
DJ_TITANS_50_TICKER = [
    "MMM",  # 3M
    "ABBV",  # AbbVie
    # "ALIZY",  # Allianz
    "GOOG",  # Alphabet
    "AMZN",  # Amazon
    "AMGN",  # Amgen
    "BUD",  # Anheuser-Busch InBev
    "AAPL",  # Apple
    "BHP",  # BHP
    "BA",  # Boeing
    "BP",  # BP
    "BTI",  # British American Tobacco
    "CVX",  # Chevron
    "CSCO",  # Cisco
    "C",  # Citigroup
    "KO",  # Coca-Cola
    "DD",  # DuPont
    "XOM",  # ExxonMobil
    "META",  # Meta
    "GE",  # General Electric
    "GSK",  # GlaxoSmithKline
    "HSBC",  # HSBC
    "INTC",  # Intel
    "IBM",  # IBM
    "JNJ",  # Johnson & Johnson
    "JPM",  # JPMorgan Chase
    "MA",  # Mastercard
    "MCD",  # McDonald's
    "MRK",  # Merck
    "MSFT",  # Microsoft
    # "NSRGY",  # Nestlé
    "NVS",  # Novartis
    "NVDA",  # Nvidia
    "ORCL",  # Oracle
    "PEP",  # PepsiCo
    "PFE",  # Pfizer
    "PM",  # Philip Morris
    "PG",  # Procter & Gamble
    # "RHHBY",  # Roche
    "RY",  # Royal Bank of Canada
    "SHEL",  # Shell
    # "SSNLF",  # Samsung
    # "SOFR",  # Amplify Samsung SOFR ETF
    "SNY",  # Sanofi
    # "SIEGY",  # Siemens
    "TSM",  # TSMC
    "TTE",  # TotalEnergies
    "V",  # Visa
    "TM",  # Toyota
    "WMT",  # Walmart
    "DIS",  # Disney
    "VIXM",  # VIX Index
    "VIXY",  # VIX Index
]


# Error handling
class DataError(Exception):
    """Base class for data-related errors."""

    pass


class InsufficientDataError(DataError):
    """Raised when there is insufficient historical data."""

    pass


class DataQualityError(DataError):
    """Raised when data quality does not meet requirements."""

    pass


# Data Management Settings
DATA_MANAGEMENT = {
    "cleanup_threshold_days": 30,
    "version_retention": 3,
    "compression": "snappy",
    "cache_expiry_hours": 24,
}

# Logging Settings
LOGGING = {
    "console_level": "INFO",
    "file_level": "DEBUG",
    "rotation_size_mb": 10,
    "backup_count": 5,
}
