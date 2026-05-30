"""
Unit tests for the NSE configuration module in python/data_generation/nse_config.py.
"""

from __future__ import annotations

import datetime
import unittest
import numpy as np
from python.data_generation.nse_config import (
    NSE_FNO_STOCKS,
    NIFTY_FUTURES_CONFIG,
    MARKET_OPEN,
    MARKET_CLOSE,
    TRADING_SECONDS,
    intraday_volatility_profile,
    get_sector_correlation,
    get_stocks_by_sector,
    get_all_symbols,
    get_stock_config,
    elapsed_seconds_from_time,
)


class TestNSEConfig(unittest.TestCase):
    def test_stocks_exist(self) -> None:
        self.assertEqual(len(NSE_FNO_STOCKS), 20)
        self.assertIn("RELIANCE", NSE_FNO_STOCKS)
        self.assertIn("HDFCBANK", NSE_FNO_STOCKS)
        self.assertIn("TCS", NSE_FNO_STOCKS)

    def test_nifty_config(self) -> None:
        self.assertEqual(NIFTY_FUTURES_CONFIG["symbol"], "NIFTY")
        self.assertEqual(NIFTY_FUTURES_CONFIG["lot_size"], 25)

    def test_market_hours(self) -> None:
        self.assertEqual(MARKET_OPEN, datetime.time(9, 15, 0))
        self.assertEqual(MARKET_CLOSE, datetime.time(15, 30, 0))
        self.assertEqual(TRADING_SECONDS, 22500)

    def test_volatility_profile(self) -> None:
        # Volatility multiplier at open (0s) and close (22500s) should be around 2.0
        v_open = intraday_volatility_profile(0.0)
        v_close = intraday_volatility_profile(float(TRADING_SECONDS))
        v_mid = intraday_volatility_profile(TRADING_SECONDS / 2.0)

        self.assertAlmostEqual(v_open, 2.0, places=3)
        self.assertAlmostEqual(v_close, 2.0, places=3)
        self.assertAlmostEqual(v_mid, 0.6, places=3)

        # Check numpy array support
        t = np.array([0.0, TRADING_SECONDS / 2.0, float(TRADING_SECONDS)])
        v = intraday_volatility_profile(t)
        self.assertEqual(len(v), 3)
        self.assertAlmostEqual(v[0], 2.0, places=3)
        self.assertAlmostEqual(v[1], 0.6, places=3)
        self.assertAlmostEqual(v[2], 2.0, places=3)

    def test_sector_correlation(self) -> None:
        self.assertEqual(get_sector_correlation("Banking", "Banking"), 0.80)
        self.assertEqual(get_sector_correlation("Energy", "Energy"), 1.0)
        self.assertEqual(get_sector_correlation("Banking", "NBFC"), 0.60)
        self.assertEqual(get_sector_correlation("NBFC", "Banking"), 0.60)
        # Unmapped sectors should return 0.20
        self.assertEqual(get_sector_correlation("Telecom", "Pharma"), 0.20)

    def test_get_stocks_by_sector(self) -> None:
        banking_stocks = get_stocks_by_sector("Banking")
        self.assertIn("HDFCBANK", banking_stocks)
        self.assertIn("ICICIBANK", banking_stocks)
        self.assertIn("SBIN", banking_stocks)
        self.assertNotIn("TCS", banking_stocks)

    def test_get_all_symbols(self) -> None:
        symbols = get_all_symbols()
        self.assertEqual(len(symbols), 20)
        self.assertEqual(symbols, sorted(list(NSE_FNO_STOCKS.keys())))

    def test_get_stock_config(self) -> None:
        config = get_stock_config("RELIANCE")
        self.assertEqual(config["symbol"], "RELIANCE")
        self.assertEqual(config["lot_size"], 250)
        self.assertEqual(config["tick_size"], 0.05)

        with self.assertRaises(KeyError):
            get_stock_config("INVALID_SYMBOL")

    def test_elapsed_seconds_from_time(self) -> None:
        t_open = datetime.time(9, 15, 0)
        self.assertEqual(elapsed_seconds_from_time(t_open), 0.0)

        t_close = datetime.time(15, 30, 0)
        self.assertEqual(elapsed_seconds_from_time(t_close), 22500.0)


if __name__ == "__main__":
    unittest.main()
