"""
NSE Market Configuration Module
================================

Provides comprehensive configuration for NSE (National Stock Exchange of India)
market simulation including:

- Top 20 F&O (Futures & Options) stock specifications
- NIFTY 50 futures contract configuration
- Market session hours (IST)
- Intraday volatility profile (U-shaped curve)
- Sector classifications and correlation mappings
- Default simulation parameters

All prices are in INR (₹). Tick sizes follow NSE conventions.

References
----------
- NSE F&O lot sizes: https://www.nseindia.com/
- Market hours: 09:15 IST to 15:30 IST (continuous trading session)
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Top 20 F&O Stocks
# ---------------------------------------------------------------------------

NSE_FNO_STOCKS: Dict[str, Dict[str, Any]] = {
    "RELIANCE": {
        "symbol": "RELIANCE",
        "base_price": 2450.0,
        "lot_size": 250,
        "sector": "Energy",
        "tick_size": 0.05,
        "isin": "INE002A01018",
        "daily_volatility": 0.018,
        "avg_daily_volume": 8_500_000,
    },
    "HDFCBANK": {
        "symbol": "HDFCBANK",
        "base_price": 1650.0,
        "lot_size": 550,
        "sector": "Banking",
        "tick_size": 0.05,
        "isin": "INE040A01034",
        "daily_volatility": 0.014,
        "avg_daily_volume": 12_000_000,
    },
    "ICICIBANK": {
        "symbol": "ICICIBANK",
        "base_price": 1100.0,
        "lot_size": 1400,
        "sector": "Banking",
        "tick_size": 0.05,
        "isin": "INE090A01021",
        "daily_volatility": 0.016,
        "avg_daily_volume": 15_000_000,
    },
    "INFY": {
        "symbol": "INFY",
        "base_price": 1580.0,
        "lot_size": 400,
        "sector": "IT",
        "tick_size": 0.05,
        "isin": "INE009A01021",
        "daily_volatility": 0.017,
        "avg_daily_volume": 10_000_000,
    },
    "TCS": {
        "symbol": "TCS",
        "base_price": 3600.0,
        "lot_size": 175,
        "sector": "IT",
        "tick_size": 0.05,
        "isin": "INE467B01029",
        "daily_volatility": 0.013,
        "avg_daily_volume": 3_500_000,
    },
    "SBIN": {
        "symbol": "SBIN",
        "base_price": 780.0,
        "lot_size": 1500,
        "sector": "Banking",
        "tick_size": 0.05,
        "isin": "INE062A01020",
        "daily_volatility": 0.019,
        "avg_daily_volume": 20_000_000,
    },
    "TATAMOTORS": {
        "symbol": "TATAMOTORS",
        "base_price": 650.0,
        "lot_size": 1400,
        "sector": "Automobile",
        "tick_size": 0.05,
        "isin": "INE155A01022",
        "daily_volatility": 0.022,
        "avg_daily_volume": 18_000_000,
    },
    "BAJFINANCE": {
        "symbol": "BAJFINANCE",
        "base_price": 7200.0,
        "lot_size": 125,
        "sector": "NBFC",
        "tick_size": 0.05,
        "isin": "INE296A01024",
        "daily_volatility": 0.020,
        "avg_daily_volume": 5_000_000,
    },
    "AXISBANK": {
        "symbol": "AXISBANK",
        "base_price": 1050.0,
        "lot_size": 1200,
        "sector": "Banking",
        "tick_size": 0.05,
        "isin": "INE238A01034",
        "daily_volatility": 0.017,
        "avg_daily_volume": 14_000_000,
    },
    "BHARTIARTL": {
        "symbol": "BHARTIARTL",
        "base_price": 1450.0,
        "lot_size": 475,
        "sector": "Telecom",
        "tick_size": 0.05,
        "isin": "INE397D01024",
        "daily_volatility": 0.016,
        "avg_daily_volume": 7_000_000,
    },
    "ITC": {
        "symbol": "ITC",
        "base_price": 430.0,
        "lot_size": 1600,
        "sector": "FMCG",
        "tick_size": 0.05,
        "isin": "INE154A01025",
        "daily_volatility": 0.012,
        "avg_daily_volume": 16_000_000,
    },
    "LT": {
        "symbol": "LT",
        "base_price": 3400.0,
        "lot_size": 150,
        "sector": "Infrastructure",
        "tick_size": 0.05,
        "isin": "INE018A01030",
        "daily_volatility": 0.015,
        "avg_daily_volume": 4_000_000,
    },
    "TATASTEEL": {
        "symbol": "TATASTEEL",
        "base_price": 145.0,
        "lot_size": 5500,
        "sector": "Metals",
        "tick_size": 0.05,
        "isin": "INE081A01020",
        "daily_volatility": 0.023,
        "avg_daily_volume": 25_000_000,
    },
    "HCLTECH": {
        "symbol": "HCLTECH",
        "base_price": 1520.0,
        "lot_size": 350,
        "sector": "IT",
        "tick_size": 0.05,
        "isin": "INE860A01027",
        "daily_volatility": 0.016,
        "avg_daily_volume": 6_000_000,
    },
    "M&M": {
        "symbol": "M&M",
        "base_price": 2700.0,
        "lot_size": 350,
        "sector": "Automobile",
        "tick_size": 0.05,
        "isin": "INE101A01026",
        "daily_volatility": 0.018,
        "avg_daily_volume": 5_500_000,
    },
    "MARUTI": {
        "symbol": "MARUTI",
        "base_price": 10500.0,
        "lot_size": 100,
        "sector": "Automobile",
        "tick_size": 0.05,
        "isin": "INE585B01010",
        "daily_volatility": 0.015,
        "avg_daily_volume": 1_500_000,
    },
    "SUNPHARMA": {
        "symbol": "SUNPHARMA",
        "base_price": 1750.0,
        "lot_size": 350,
        "sector": "Pharma",
        "tick_size": 0.05,
        "isin": "INE044A01036",
        "daily_volatility": 0.017,
        "avg_daily_volume": 6_500_000,
    },
    "ADANIENT": {
        "symbol": "ADANIENT",
        "base_price": 2900.0,
        "lot_size": 250,
        "sector": "Conglomerate",
        "tick_size": 0.05,
        "isin": "INE423A01024",
        "daily_volatility": 0.028,
        "avg_daily_volume": 9_000_000,
    },
    "JSWSTEEL": {
        "symbol": "JSWSTEEL",
        "base_price": 870.0,
        "lot_size": 675,
        "sector": "Metals",
        "tick_size": 0.05,
        "isin": "INE019A01038",
        "daily_volatility": 0.021,
        "avg_daily_volume": 8_000_000,
    },
    "KOTAKBANK": {
        "symbol": "KOTAKBANK",
        "base_price": 1850.0,
        "lot_size": 400,
        "sector": "Banking",
        "tick_size": 0.05,
        "isin": "INE237A01028",
        "daily_volatility": 0.014,
        "avg_daily_volume": 7_500_000,
    },
}

# ---------------------------------------------------------------------------
# NIFTY 50 Futures Configuration
# ---------------------------------------------------------------------------

NIFTY_FUTURES_CONFIG: Dict[str, Any] = {
    "symbol": "NIFTY",
    "base_price": 22500.0,
    "tick_size": 0.05,
    "lot_size": 25,
    "daily_volatility": 0.011,
    "avg_daily_volume": 150_000_000,
    "exchange": "NSE",
    "segment": "F&O",
}

# ---------------------------------------------------------------------------
# Market Hours (IST — Indian Standard Time, UTC+05:30)
# ---------------------------------------------------------------------------

MARKET_OPEN: datetime.time = datetime.time(9, 15, 0)
MARKET_CLOSE: datetime.time = datetime.time(15, 30, 0)
PRE_OPEN_START: datetime.time = datetime.time(9, 0, 0)
PRE_OPEN_END: datetime.time = datetime.time(9, 8, 0)

TRADING_SECONDS: int = (
    (MARKET_CLOSE.hour * 3600 + MARKET_CLOSE.minute * 60 + MARKET_CLOSE.second)
    - (MARKET_OPEN.hour * 3600 + MARKET_OPEN.minute * 60 + MARKET_OPEN.second)
)  # 22500 seconds = 6 hours 15 minutes


# ---------------------------------------------------------------------------
# Intraday Volatility Profile (U-shaped)
# ---------------------------------------------------------------------------

def intraday_volatility_profile(
    elapsed_seconds: np.ndarray | float,
    total_seconds: int = TRADING_SECONDS,
) -> np.ndarray:
    """
    Compute a U-shaped intraday volatility multiplier.

    The profile captures the well-documented intraday pattern where
    volatility (and volume) is elevated at market open and close,
    with a trough around midday.

    The model uses a quadratic U-shape:

        v(t) = a * (t_norm - 0.5)^2 + b

    where ``t_norm = elapsed / total`` ∈ [0, 1], calibrated so that
    the peak multiplier at open/close ≈ 2.0 and the trough at midday ≈ 0.6.

    Parameters
    ----------
    elapsed_seconds : np.ndarray or float
        Seconds elapsed since market open (09:15 IST).
    total_seconds : int, optional
        Total trading session length in seconds. Default is 22500.

    Returns
    -------
    np.ndarray
        Volatility multiplier(s) for each input time. Values are clipped
        to [0.4, 2.5] for numerical safety.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0, 11250, 22500])  # open, midday, close
    >>> intraday_volatility_profile(t)
    array([2. , 0.6, 2. ])
    """
    t_norm = np.asarray(elapsed_seconds, dtype=np.float64) / total_seconds
    t_norm = np.clip(t_norm, 0.0, 1.0)

    # Quadratic U-shape: peaks of 2.0 at t=0 and t=1, trough of 0.6 at t=0.5
    # v(t) = a*(t - 0.5)^2 + b
    # v(0) = a*0.25 + b = 2.0
    # v(0.5) = b = 0.6  →  a = (2.0 - 0.6) / 0.25 = 5.6
    a = 5.6
    b = 0.6
    profile = a * (t_norm - 0.5) ** 2 + b

    return np.clip(profile, 0.4, 2.5)


# ---------------------------------------------------------------------------
# Sector Mappings & Correlation Structure
# ---------------------------------------------------------------------------

SECTOR_LIST: List[str] = [
    "Banking",
    "IT",
    "Automobile",
    "Energy",
    "NBFC",
    "Telecom",
    "FMCG",
    "Infrastructure",
    "Metals",
    "Pharma",
    "Conglomerate",
]

SECTOR_CORRELATION: Dict[Tuple[str, str], float] = {
    # Intra-sector correlations (higher)
    ("Banking", "Banking"): 0.80,
    ("IT", "IT"): 0.75,
    ("Automobile", "Automobile"): 0.65,
    ("Metals", "Metals"): 0.70,
    # Cross-sector correlations
    ("Banking", "NBFC"): 0.60,
    ("Banking", "IT"): 0.30,
    ("Banking", "Energy"): 0.35,
    ("Banking", "Automobile"): 0.40,
    ("Banking", "FMCG"): 0.25,
    ("Banking", "Telecom"): 0.30,
    ("Banking", "Infrastructure"): 0.45,
    ("Banking", "Metals"): 0.35,
    ("Banking", "Pharma"): 0.20,
    ("Banking", "Conglomerate"): 0.40,
    ("IT", "NBFC"): 0.25,
    ("IT", "Energy"): 0.20,
    ("IT", "Automobile"): 0.25,
    ("IT", "FMCG"): 0.20,
    ("IT", "Telecom"): 0.35,
    ("IT", "Infrastructure"): 0.20,
    ("IT", "Metals"): 0.15,
    ("IT", "Pharma"): 0.25,
    ("IT", "Conglomerate"): 0.25,
    ("Automobile", "Energy"): 0.30,
    ("Automobile", "NBFC"): 0.35,
    ("Automobile", "Telecom"): 0.20,
    ("Automobile", "FMCG"): 0.20,
    ("Automobile", "Infrastructure"): 0.35,
    ("Automobile", "Metals"): 0.45,
    ("Automobile", "Pharma"): 0.15,
    ("Automobile", "Conglomerate"): 0.35,
    ("Energy", "NBFC"): 0.25,
    ("Energy", "Telecom"): 0.20,
    ("Energy", "FMCG"): 0.15,
    ("Energy", "Infrastructure"): 0.40,
    ("Energy", "Metals"): 0.40,
    ("Energy", "Pharma"): 0.15,
    ("Energy", "Conglomerate"): 0.45,
    ("Metals", "Infrastructure"): 0.50,
    ("Metals", "Conglomerate"): 0.40,
    ("Metals", "NBFC"): 0.25,
    ("Metals", "Pharma"): 0.10,
    ("FMCG", "Pharma"): 0.30,
}


def get_sector_correlation(sector_a: str, sector_b: str) -> float:
    """
    Look up the pairwise correlation between two sectors.

    Parameters
    ----------
    sector_a : str
        First sector name (must be in ``SECTOR_LIST``).
    sector_b : str
        Second sector name (must be in ``SECTOR_LIST``).

    Returns
    -------
    float
        Correlation coefficient in [0, 1]. Returns 1.0 for identical
        sectors, the stored value for known pairs, or a default of 0.20
        for unmapped pairs.
    """
    if sector_a == sector_b:
        return SECTOR_CORRELATION.get((sector_a, sector_b), 1.0)
    # Try both orderings
    corr = SECTOR_CORRELATION.get((sector_a, sector_b))
    if corr is not None:
        return corr
    corr = SECTOR_CORRELATION.get((sector_b, sector_a))
    if corr is not None:
        return corr
    return 0.20  # default for unmapped pairs


def get_stocks_by_sector(sector: str) -> List[str]:
    """
    Return list of stock symbols belonging to a given sector.

    Parameters
    ----------
    sector : str
        Sector name (e.g., "Banking", "IT").

    Returns
    -------
    List[str]
        Sorted list of matching stock symbols.
    """
    return sorted(
        sym for sym, cfg in NSE_FNO_STOCKS.items() if cfg["sector"] == sector
    )


# ---------------------------------------------------------------------------
# Default Simulation Parameters
# ---------------------------------------------------------------------------

SIMULATION_DEFAULTS: Dict[str, Any] = {
    "num_trading_days": 126,           # ~6 months of trading days
    "snapshot_interval_ms": 100,       # order-book snapshot every 100 ms
    "snapshot_levels": 5,              # top-5 bid/ask levels in snapshot
    "cancel_to_trade_ratio": 5.0,      # ~5 cancellations per trade
    "order_size_mu": 1.5,              # log-normal μ for order-size multiplier
    "order_size_sigma": 0.8,           # log-normal σ for order-size multiplier
    "mean_reversion_kappa": 5.0,       # Ornstein-Uhlenbeck mean-reversion speed
    "gbm_drift": 0.0,                 # annualised drift (neutral)
    "random_seed": 42,                 # reproducibility seed
    "hawkes_mu": 50.0,                 # base arrival intensity (events/sec)
    "hawkes_alpha": 30.0,              # self-excitation jump
    "hawkes_omega": 60.0,              # exponential decay rate
    "limit_order_depth_ticks": 20,     # max distance from mid for limit orders
    "market_order_fraction": 0.15,     # fraction of orders that are market orders
    "output_dir": "data/raw",          # output directory for generated files
}

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_all_symbols() -> List[str]:
    """Return a sorted list of all 20 F&O stock symbols."""
    return sorted(NSE_FNO_STOCKS.keys())


def get_stock_config(symbol: str) -> Dict[str, Any]:
    """
    Retrieve the configuration dictionary for a given stock symbol.

    Parameters
    ----------
    symbol : str
        NSE stock symbol (e.g., "RELIANCE").

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary with keys: symbol, base_price, lot_size,
        sector, tick_size, isin, daily_volatility, avg_daily_volume.

    Raises
    ------
    KeyError
        If the symbol is not found in ``NSE_FNO_STOCKS``.
    """
    if symbol not in NSE_FNO_STOCKS:
        raise KeyError(
            f"Symbol '{symbol}' not found. Available: {get_all_symbols()}"
        )
    return NSE_FNO_STOCKS[symbol]


def elapsed_seconds_from_time(t: datetime.time) -> float:
    """
    Compute elapsed seconds since market open (09:15:00) for a given time.

    Parameters
    ----------
    t : datetime.time
        Time of day (assumed IST).

    Returns
    -------
    float
        Seconds since market open. Negative if before open, positive after.
    """
    t_sec = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    open_sec = MARKET_OPEN.hour * 3600 + MARKET_OPEN.minute * 60 + MARKET_OPEN.second
    return t_sec - open_sec


if __name__ == "__main__":
    # Quick sanity check
    print("=" * 60)
    print("NSE F&O Configuration — Sanity Check")
    print("=" * 60)
    print(f"Number of stocks : {len(NSE_FNO_STOCKS)}")
    print(f"Market open      : {MARKET_OPEN}")
    print(f"Market close     : {MARKET_CLOSE}")
    print(f"Trading seconds  : {TRADING_SECONDS:,}")
    print(f"NIFTY base price : ₹{NIFTY_FUTURES_CONFIG['base_price']:,.2f}")
    print()

    for sym in get_all_symbols():
        cfg = NSE_FNO_STOCKS[sym]
        notional = cfg["base_price"] * cfg["lot_size"]
        print(
            f"  {sym:<14s}  ₹{cfg['base_price']:>10,.2f}  "
            f"lot={cfg['lot_size']:>5d}  "
            f"notional=₹{notional:>14,.2f}  "
            f"sector={cfg['sector']}"
        )

    # Volatility profile check
    print()
    t_check = np.array([0, TRADING_SECONDS // 4, TRADING_SECONDS // 2,
                         3 * TRADING_SECONDS // 4, TRADING_SECONDS])
    v_check = intraday_volatility_profile(t_check)
    for t_val, v_val in zip(t_check, v_check):
        mins = t_val // 60
        print(f"  t={mins:>4d} min  →  volatility multiplier = {v_val:.3f}")
