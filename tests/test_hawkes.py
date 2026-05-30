"""
Unit tests for the Hawkes Process simulation in python/data_generation/hawkes_process.py.
"""

from __future__ import annotations

import datetime
import unittest
import numpy as np
from python.data_generation.hawkes_process import (
    HawkesProcess,
    MultivariateHawkes,
    generate_intraday_arrivals,
)
from python.data_generation.nse_config import intraday_volatility_profile


class TestHawkesProcess(unittest.TestCase):
    def test_hawkes_initialization(self) -> None:
        # Stationary process (alpha < omega)
        hp = HawkesProcess(mu=1.5, alpha=0.5, omega=1.0)
        self.assertEqual(hp.mu, 1.5)
        self.assertEqual(hp.alpha, 0.5)
        self.assertEqual(hp.omega, 1.0)
        self.assertEqual(hp.branching_ratio, 0.5)

        # Invalid parameters
        with self.assertRaises(ValueError):
            HawkesProcess(mu=-0.1, alpha=0.5, omega=1.0)
        with self.assertRaises(ValueError):
            HawkesProcess(mu=1.5, alpha=-0.1, omega=1.0)
        with self.assertRaises(ValueError):
            HawkesProcess(mu=1.5, alpha=0.5, omega=-0.1)

        # Non-stationary process (alpha >= omega)
        with self.assertRaises(ValueError):
            HawkesProcess(mu=1.5, alpha=1.0, omega=1.0)
        with self.assertRaises(ValueError):
            HawkesProcess(mu=1.5, alpha=1.5, omega=1.0)

    def test_expected_rate(self) -> None:
        hp = HawkesProcess(mu=1.0, alpha=0.5, omega=1.0)
        # Expected rate = mu / (1 - alpha/omega) = 1.0 / (1 - 0.5) = 2.0
        self.assertEqual(hp.expected_rate(), 2.0)

    def test_simulate_univariate(self) -> None:
        hp = HawkesProcess(mu=2.0, alpha=0.4, omega=1.0)
        events = hp.simulate(T=10.0, seed=42)
        self.assertIsInstance(events, np.ndarray)
        self.assertEqual(events.dtype, np.float64)
        if len(events) > 0:
            # Event times must be sorted and within [0, T]
            self.assertTrue(np.all(np.diff(events) >= 0))
            self.assertTrue(np.all(events >= 0.0))
            self.assertTrue(np.all(events <= 10.0))

    def test_multivariate_initialization(self) -> None:
        mh = MultivariateHawkes()
        self.assertEqual(len(mh.STREAMS), 3)
        self.assertIn("limit_orders", mh.params)
        self.assertIn("market_orders", mh.params)
        self.assertIn("cancellations", mh.params)

        # Custom parameters setup
        custom_params = {
            "limit_orders": {"mu": 10.0, "alpha": 2.0, "omega": 5.0},
            "market_orders": {"mu": 2.0, "alpha": 1.0, "omega": 4.0},
            "cancellations": {"mu": 8.0, "alpha": 3.0, "omega": 6.0},
        }
        mh_custom = MultivariateHawkes(params=custom_params)
        self.assertEqual(mh_custom.params["limit_orders"]["mu"], 10.0)

        # Test non-stationarity validation
        bad_params = {
            "limit_orders": {"mu": 10.0, "alpha": 5.0, "omega": 5.0},  # non-stationary
            "market_orders": {"mu": 2.0, "alpha": 1.0, "omega": 4.0},
            "cancellations": {"mu": 8.0, "alpha": 3.0, "omega": 6.0},
        }
        with self.assertRaises(ValueError):
            MultivariateHawkes(params=bad_params)

    def test_simulate_multivariate(self) -> None:
        mh = MultivariateHawkes()
        result = mh.simulate_all(T=2.0, seed=42)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(mh.STREAMS))

        for stream, events in result.items():
            self.assertIsInstance(events, np.ndarray)
            if len(events) > 0:
                self.assertTrue(np.all(np.diff(events) >= 0))
                self.assertTrue(np.all(events >= 0.0))
                self.assertTrue(np.all(events <= 2.0))

    def test_generate_intraday_arrivals(self) -> None:
        symbol_cfg = {
            "symbol": "TEST",
            "base_price": 1000.0,
            "lot_size": 100,
            "sector": "Banking",
            "tick_size": 0.05,
            "avg_daily_volume": 10_000_000,
        }
        # Simulate a small window to speed up tests (mock TRADING_SECONDS in test if needed,
        # but the function imports TRADING_SECONDS from nse_config which is 22500)
        # To make it fast, we can use default parameters and test a seed
        # Let's verify it returns the dict of order streams
        date = datetime.date(2025, 3, 1)
        # Note: the full simulate runs TRADING_SECONDS/300 windows = 75 windows. Each window simulates MultivariateHawkes(T=300).
        # This takes very little time (under 0.2s) in total.
        arrivals = generate_intraday_arrivals(
            symbol_config=symbol_cfg,
            date=date,
            volatility_profile=intraday_volatility_profile,
            seed=42,
        )

        self.assertIsInstance(arrivals, dict)
        self.assertEqual(set(arrivals.keys()), {"limit_orders", "market_orders", "cancellations"})
        for stream, times in arrivals.items():
            self.assertIsInstance(times, np.ndarray)
            if len(times) > 0:
                self.assertTrue(np.all(np.diff(times) >= 0))
                self.assertTrue(np.all(times >= 0))
                # 22500 is TRADING_SECONDS
                self.assertTrue(np.all(times <= 22500.0))


if __name__ == "__main__":
    unittest.main()
