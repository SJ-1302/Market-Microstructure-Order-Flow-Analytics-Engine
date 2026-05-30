"""
Unit tests for the metrics calculator and strategy backtester in python/utils/.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from python.utils.compute_metrics import (
    clean_trades,
    compute_spread_metrics,
    compute_obi_metrics,
    compute_adverse_selection,
    compute_price_impact,
    compute_queue_decay,
)
from python.utils.run_backtest import run_backtest, compute_performance_metrics


class TestBacktestAndMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        # Build dummy trade prints
        np.random.seed(42)
        dates = pd.date_range("2025-03-03 09:15:00", periods=100, freq="min")
        self.df_trades = pd.DataFrame({
            "timestamp": dates,
            "symbol": "RELIANCE",
            "price": np.random.normal(2450.0, 1.0, 100),
            "size": np.random.randint(1, 10, 100) * 250,
            "trade_id": np.arange(1, 101),
            "order_id": np.arange(1001, 1101),
        })

        # Build dummy snapshots
        self.df_book = pd.DataFrame({
            "timestamp": dates,
            "symbol": "RELIANCE",
            "mid_price": self.df_trades["price"],
            "total_bid_volume": 10000,
            "total_ask_volume": 12000,
        })
        for i in range(1, 6):
            self.df_book[f"bid_price_{i}"] = self.df_book["mid_price"] - i * 0.05
            self.df_book[f"bid_size_{i}"] = np.random.randint(500, 2000, 100)
            self.df_book[f"bid_orders_{i}"] = np.random.randint(1, 5, 100)
            self.df_book[f"ask_price_{i}"] = self.df_book["mid_price"] + i * 0.05
            self.df_book[f"ask_size_{i}"] = np.random.randint(500, 2000, 100)
            self.df_book[f"ask_orders_{i}"] = np.random.randint(1, 5, 100)

        # Build dummy order events
        self.df_events = pd.DataFrame({
            "timestamp": dates,
            "symbol": "RELIANCE",
            "event_type": np.random.choice(["LIMIT_ORDER", "CANCEL", "FILL"], 100),
            "order_id": np.random.randint(1, 50, 100),
            "side": np.random.choice(["BID", "ASK"], 100),
            "price": self.df_trades["price"],
            "size": 250,
        })

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_clean_trades(self) -> None:
        cleaned = clean_trades(self.df_trades)
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertFalse(cleaned.empty)
        self.assertTrue("price" in cleaned.columns)

    def test_spread_metrics(self) -> None:
        df_book_res, df_trades_res = compute_spread_metrics(self.df_book, self.df_trades)
        self.assertIn("quoted_spread", df_book_res.columns)
        self.assertIn("effective_spread", df_trades_res.columns)
        self.assertIn("realized_spread", df_trades_res.columns)

    def test_obi_metrics(self) -> None:
        df_book_res = compute_obi_metrics(self.df_book)
        self.assertIn("oi_level1", df_book_res.columns)
        self.assertIn("obi_exp_weight", df_book_res.columns)

    def test_adverse_selection_metrics(self) -> None:
        # Pre-requisite columns for compute_adverse_selection
        self.df_trades["signed_volume"] = self.df_trades["size"]
        df_adverse, df_vpin = compute_adverse_selection(self.df_trades)
        self.assertIsInstance(df_adverse, pd.DataFrame)
        self.assertIsInstance(df_vpin, pd.DataFrame)

    def test_price_impact_metrics(self) -> None:
        self.df_trades["dollar_volume"] = self.df_trades["price"] * self.df_trades["size"]
        self.df_trades["effective_spread"] = 0.1
        self.df_trades["realized_spread"] = 0.05
        df_impact = compute_price_impact(self.df_trades)
        self.assertIn("amihud_illiq", df_impact.columns)
        self.assertIn("temp_impact_5", df_impact.columns)

    def test_queue_decay(self) -> None:
        df_decay = compute_queue_decay(self.df_events)
        self.assertIsInstance(df_decay, pd.DataFrame)
        self.assertFalse(df_decay.empty)
        self.assertIn("decay_rate_lambda", df_decay.columns)

    def test_backtester(self) -> None:
        # Create a mock entry_exit_signals.csv file
        sig_path = os.path.join(self.temp_dir, "entry_exit_signals.csv")
        mock_signals = pd.DataFrame({
            "timestamp": pd.date_range("2025-03-03 09:15:00", periods=50, freq="5min"),
            "symbol": "RELIANCE",
            "price": np.random.uniform(2400.0, 2500.0, 50),
            "signal_type": ["ENTRY_LONG", "ENTRY_SHORT", "EXIT_LONG", "EXIT_SHORT", "HOLD"] * 10
        })
        mock_signals.to_csv(sig_path, index=False)

        # Run backtest
        df_equity, df_trades = run_backtest(signals_path=sig_path)
        self.assertIsInstance(df_equity, pd.DataFrame)
        self.assertIsInstance(df_trades, pd.DataFrame)

        # Test performance metrics
        metrics = compute_performance_metrics(df_equity, df_trades)
        self.assertIn("total_return", metrics)
        self.assertIn("sharpe", metrics)
        self.assertIn("max_dd", metrics)


if __name__ == "__main__":
    unittest.main()
