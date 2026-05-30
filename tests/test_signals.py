"""
Unit tests for the signal generation, trading rules, and calibrator modules.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from python.signals.signal_generator import (
    SignalConfig,
    SignalGenerator,
    rolling_zscore,
    momentum_signal,
    regime_detector,
)
from python.signals.entry_exit_rules import RuleConfig, TradingRuleEngine
from python.signals.threshold_calibrator import ThresholdCalibrator, compute_max_drawdown


class TestSignalsAndCalibrator(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        # Build some mock data
        np.random.seed(42)
        self.n_ticks = 200
        self.timestamps = pd.date_range("2024-01-02 09:30:00", periods=self.n_ticks, freq="s")
        self.prices = 150.0 + np.cumsum(np.random.normal(0, 0.1, self.n_ticks))

        # Basic signal df for calibrator
        self.signals_df = pd.DataFrame({
            "timestamp": self.timestamps,
            "symbol": "TEST",
            "price": self.prices,
            "signal_strength": np.tanh(np.random.normal(0, 0.5, self.n_ticks)),
            "signal_direction": "NEUTRAL",
            "spread_z": np.random.normal(0, 0.1, self.n_ticks),
            "vpin_signal": np.abs(np.random.normal(0, 0.1, self.n_ticks)),
            "spread": np.abs(np.random.normal(0.05, 0.01, self.n_ticks)),
        })
        self.signals_df["signal_direction"] = np.where(
            self.signals_df["signal_strength"] > 0.3,
            "BUY",
            np.where(self.signals_df["signal_strength"] < -0.3, "SELL", "NEUTRAL")
        )
        self.signals_df["confidence"] = self.signals_df["signal_strength"].abs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_feature_helpers(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test rolling_zscore
        z = rolling_zscore(s, window=3)
        self.assertEqual(len(z), 5)
        self.assertAlmostEqual(z.iloc[0], 0.0)
        self.assertAlmostEqual(z.iloc[1], 0.70710678)
        self.assertAlmostEqual(z.iloc[2], 1.0)

        # Test momentum_signal
        mom = momentum_signal(s, lookback=2)
        self.assertEqual(len(mom), 5)
        # index 2: (3 - 1)/1 = 2.0
        self.assertEqual(mom.iloc[2], 2.0)

        # Test regime_detector
        vol = pd.Series(np.random.uniform(0.1, 0.2, 50))
        regimes = regime_detector(vol, threshold=1.0)
        self.assertEqual(len(regimes), 50)
        self.assertTrue(set(regimes.unique()).issubset({"HIGH_VOL", "LOW_VOL"}))

    def test_signal_generator(self) -> None:
        # Default config
        gen = SignalGenerator()
        self.assertEqual(gen.config.weight_obi, 0.35)

        # Generate on syntheticAAPL/MSFT standalone mode (when no metrics loaded)
        signals = gen.generate_signals()
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertTrue(len(signals) > 0)
        self.assertIn("signal_strength", signals.columns)
        self.assertIn("signal_direction", signals.columns)

        # Test save signals
        path = gen.save_signals(self.temp_dir)
        self.assertTrue(os.path.exists(path))

    def test_trading_rule_engine(self) -> None:
        engine = TradingRuleEngine()
        self.assertEqual(engine.config.entry_threshold, 0.30)

        # Evaluate entry
        row = pd.Series({
            "signal_strength": 0.5,
            "signal_direction": "BUY",
            "spread_z": 0.1,
            "vpin_signal": 0.2,
            "price": 100.0,
            "spread": 0.05,
        })
        entry = engine.evaluate_entry(row)
        self.assertEqual(entry["action"], "BUY")

        # Wider spread entry filter
        row_wide_spread = pd.Series({
            "signal_strength": 0.5,
            "signal_direction": "BUY",
            "spread_z": 0.1,
            "vpin_signal": 0.2,
            "price": 10.0,
            "spread": 1.0,  # 10% spread (max is 0.3%)
        })
        entry_fail = engine.evaluate_entry(row_wide_spread)
        self.assertEqual(entry_fail["action"], "NONE")
        self.assertEqual(entry_fail["reason"], "spread_too_wide")

        # Evaluate exit (SL)
        pos = {"entry_price": 100.0, "direction": "BUY", "bars_held": 5, "quantity": 10, "entry_time": "2024-01-02 09:30:00"}
        row_exit = pd.Series({"price": 99.0, "signal_strength": 0.4, "signal_direction": "BUY"})  # -1% drop triggers SL
        exit_dec = engine.evaluate_exit(pos, row_exit)
        self.assertEqual(exit_dec["action"], "EXIT")
        self.assertEqual(exit_dec["reason"], "stop_loss_hit")

        # Evaluate exit (TP)
        row_tp = pd.Series({"price": 102.0, "signal_strength": 0.4, "signal_direction": "BUY"})  # +2% rise triggers TP
        exit_dec_tp = engine.evaluate_exit(pos, row_tp)
        self.assertEqual(exit_dec_tp["action"], "EXIT")
        self.assertEqual(exit_dec_tp["reason"], "take_profit_hit")

        # Position sizing
        size_fixed = engine.compute_position_size(0.8, account_value=10000.0, method="fixed_fraction", price=100.0)
        # fixed fraction = 2% * 0.8 = 1.6% of 10000 = $160 / $100 = 1 unit
        self.assertEqual(size_fixed, 1)

        # Run rules simulation
        trades = engine.run_rules(self.signals_df)
        self.assertIsInstance(trades, pd.DataFrame)
        stats = engine.compute_trade_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_trades", stats)

    def test_threshold_calibrator(self) -> None:
        cal = ThresholdCalibrator(self.signals_df)
        self.assertEqual(cal.config["train_window"], 63)

        # Test compute sharpe
        rets = pd.Series(np.random.normal(0.001, 0.01, 100))
        sharpe = cal.compute_sharpe(rets)
        self.assertIsInstance(sharpe, float)

        # Test compute false signal rate
        preds = pd.Series([1, 1, -1, 1])
        actuals = pd.Series([1, -1, -1, -1])
        # Wrong: indices 1 and 3 -> 2/4 = 50%
        self.assertEqual(cal.compute_false_signal_rate(preds, actuals), 0.5)

        # Test max drawdown
        eq = pd.Series([1.0, 1.05, 1.02, 0.95, 1.01])
        # Max peak is 1.05, valley is 0.95 -> (0.95 - 1.05)/1.05 = -0.0952
        dd = compute_max_drawdown(eq)
        self.assertAlmostEqual(dd, -0.09523809, places=6)

        # Grid search (use a tiny param grid to keep tests quick)
        grid = {
            "entry_threshold": [0.20, 0.30],
            "exit_threshold": [0.10],
            "holding_period": [20, 50],
            "stop_loss": [0.005],
        }
        best = cal.grid_search(param_grid=grid)
        self.assertIn("entry_threshold", best)
        self.assertIn("sharpe", best)

        # Compare strategies
        fixed = {"entry_threshold": 0.20, "exit_threshold": 0.10, "holding_period": 20, "stop_loss": 0.005}
        opt = {"entry_threshold": 0.30, "exit_threshold": 0.10, "holding_period": 50, "stop_loss": 0.005}
        comp = cal.compare_strategies(fixed, opt)
        self.assertIn("fixed", comp)
        self.assertIn("optimized", comp)
        self.assertIn("sharpe_improvement_pct", comp)

        # Generate report
        report_path = cal.generate_report(self.temp_dir)
        self.assertTrue(os.path.exists(report_path))


if __name__ == "__main__":
    unittest.main()
