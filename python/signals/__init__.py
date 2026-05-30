"""
Signal Generation Module for Market Microstructure & Order Flow Analytics Engine.

This package provides order flow-based signal generation, automated entry/exit
trading rules, and adaptive threshold calibration via walk-forward optimization.
"""

from .signal_generator import SignalGenerator, rolling_zscore, momentum_signal, regime_detector
from .entry_exit_rules import TradingRuleEngine
from .threshold_calibrator import ThresholdCalibrator, compute_max_drawdown

__all__ = [
    "SignalGenerator",
    "TradingRuleEngine",
    "ThresholdCalibrator",
    "rolling_zscore",
    "momentum_signal",
    "regime_detector",
    "compute_max_drawdown",
]
