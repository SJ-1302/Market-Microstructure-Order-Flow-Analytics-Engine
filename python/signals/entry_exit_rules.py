"""
Automated Trading Rule Engine
==============================

Defines entry / exit logic and position-sizing rules that consume the
composite signals produced by :class:`SignalGenerator`.  The engine evaluates
each bar against configurable thresholds for signal strength, spread,
toxicity (VPIN), and holding-period limits, then outputs a full trade log.

Dependencies
------------
numpy, pandas (listed in requirements.txt)

Example
-------
>>> engine = TradingRuleEngine()
>>> trades = engine.run_rules(signals_df, prices_df)
>>> print(engine.compute_trade_stats(trades))
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RuleConfig:
    """All tunable parameters for entry / exit / sizing rules."""

    entry_threshold: float = 0.30
    exit_threshold: float = 0.10
    max_holding_periods: int = 50
    stop_loss_pct: float = 0.005
    take_profit_pct: float = 0.010
    max_spread_pct: float = 0.003
    max_toxicity: float = 0.70
    position_size_method: str = "fixed_fraction"
    fixed_fraction_pct: float = 0.02     # 2 % of account per trade
    kelly_win_rate: float = 0.55
    kelly_payoff_ratio: float = 1.5
    account_value: float = 100_000.0
    price_per_unit: float = 150.0        # default fallback


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TradingRuleEngine:
    """Automated entry / exit and position-sizing rule engine.

    Parameters
    ----------
    config : dict or RuleConfig, optional
        Override default settings.

    Attributes
    ----------
    config : RuleConfig
    trade_log : pd.DataFrame or None
        Populated after ``run_rules()``.
    """

    def __init__(self, config: Optional[Dict | RuleConfig] = None) -> None:
        if config is None:
            self.config = RuleConfig()
        elif isinstance(config, dict):
            self.config = RuleConfig(**{
                k: v for k, v in config.items()
                if k in RuleConfig.__dataclass_fields__
            })
        else:
            self.config = config

        self.trade_log: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------

    def evaluate_entry(self, signal_row: pd.Series) -> Dict:
        """Decide whether to enter a new position on this bar.

        Parameters
        ----------
        signal_row : pd.Series
            Must contain ``signal_strength``, ``signal_direction``,
            ``spread_z``, ``vpin_signal``, and ``price``.

        Returns
        -------
        dict
            ``{'action': 'BUY'|'SELL'|'NONE', 'reason': str,
            'confidence': float}``
        """
        cfg = self.config
        strength = abs(signal_row.get("signal_strength", 0.0))
        direction = signal_row.get("signal_direction", "NEUTRAL")
        spread_raw = abs(signal_row.get("spread_z", 0.0))
        vpin_raw = abs(signal_row.get("vpin_signal", 0.0))
        price = signal_row.get("price", cfg.price_per_unit)

        # Compute spread as fraction of price for the filter
        spread_pct = signal_row.get("spread", 0.0) / price if price > 0 else 0.0

        # --- Gate 1: signal strength ------------------------------------
        if strength < cfg.entry_threshold:
            return {"action": "NONE", "reason": "signal_below_threshold",
                    "confidence": strength}

        # --- Gate 2: spread filter --------------------------------------
        if spread_pct > cfg.max_spread_pct:
            return {"action": "NONE", "reason": "spread_too_wide",
                    "confidence": strength}

        # --- Gate 3: toxicity filter ------------------------------------
        if vpin_raw > cfg.max_toxicity:
            return {"action": "NONE", "reason": "high_toxicity",
                    "confidence": strength}

        # All gates passed
        if direction == "BUY":
            action = "BUY"
        elif direction == "SELL":
            action = "SELL"
        else:
            return {"action": "NONE", "reason": "neutral_direction",
                    "confidence": strength}

        return {"action": action, "reason": "signal_confirmed",
                "confidence": strength}

    # ------------------------------------------------------------------
    # Exit evaluation
    # ------------------------------------------------------------------

    def evaluate_exit(
        self,
        position: Dict,
        current_row: pd.Series,
    ) -> Dict:
        """Decide whether to close an open position.

        Parameters
        ----------
        position : dict
            Must contain ``entry_price``, ``direction`` (``'BUY'``/``'SELL'``),
            and ``bars_held``.
        current_row : pd.Series
            Current bar data with ``price`` and ``signal_strength``.

        Returns
        -------
        dict
            ``{'action': 'EXIT'|'HOLD', 'reason': str}``
        """
        cfg = self.config
        entry_price = position["entry_price"]
        direction = position["direction"]
        bars_held = position["bars_held"]
        current_price = current_row.get("price", entry_price)
        current_strength = current_row.get("signal_strength", 0.0)
        current_direction = current_row.get("signal_direction", "NEUTRAL")

        # P&L fraction relative to entry
        if direction == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # --- Stop-loss --------------------------------------------------
        if pnl_pct <= -cfg.stop_loss_pct:
            return {"action": "EXIT", "reason": "stop_loss_hit"}

        # --- Take-profit ------------------------------------------------
        if pnl_pct >= cfg.take_profit_pct:
            return {"action": "EXIT", "reason": "take_profit_hit"}

        # --- Max holding period -----------------------------------------
        if bars_held >= cfg.max_holding_periods:
            return {"action": "EXIT", "reason": "max_holding_exceeded"}

        # --- Signal reversal --------------------------------------------
        if direction == "BUY" and current_direction == "SELL":
            return {"action": "EXIT", "reason": "signal_reversal"}
        if direction == "SELL" and current_direction == "BUY":
            return {"action": "EXIT", "reason": "signal_reversal"}

        # --- Signal weakening below exit threshold ----------------------
        if abs(current_strength) < cfg.exit_threshold:
            return {"action": "EXIT", "reason": "signal_weakened"}

        return {"action": "HOLD", "reason": "position_maintained"}

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        signal_strength: float,
        account_value: Optional[float] = None,
        method: Optional[str] = None,
        price: float = 150.0,
    ) -> int:
        """Calculate the number of units/shares to trade.

        Parameters
        ----------
        signal_strength : float
            Absolute signal strength in [0, 1].
        account_value : float, optional
            Portfolio value.  Defaults to config value.
        method : str, optional
            ``'fixed_fraction'`` or ``'kelly'``.  Defaults to config.
        price : float, default 150.0
            Price per share/unit.

        Returns
        -------
        int
            Number of units to trade (≥ 0).
        """
        cfg = self.config
        acct = account_value or cfg.account_value
        method = method or cfg.position_size_method

        if method == "kelly":
            # Kelly fraction = (p * b - q) / b
            p = cfg.kelly_win_rate
            q = 1.0 - p
            b = cfg.kelly_payoff_ratio
            kelly_f = max(0.0, (p * b - q) / b)
            # Scale by signal strength and half-Kelly for conservatism
            fraction = 0.5 * kelly_f * abs(signal_strength)
        else:
            # fixed_fraction: flat 2 % scaled by signal strength
            fraction = cfg.fixed_fraction_pct * abs(signal_strength)

        dollar_amount = acct * fraction
        units = int(dollar_amount / max(price, 0.01))
        return max(units, 0)

    # ------------------------------------------------------------------
    # Full simulation
    # ------------------------------------------------------------------

    def run_rules(
        self,
        signals_df: pd.DataFrame,
        prices_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Run the rule engine over a signals DataFrame.

        Parameters
        ----------
        signals_df : pd.DataFrame
            Output of ``SignalGenerator.generate_signals()``.
        prices_df : pd.DataFrame, optional
            If supplied, ``price`` column is taken from here; otherwise it
            must already exist in *signals_df*.

        Returns
        -------
        pd.DataFrame
            Trade log with columns: ``timestamp``, ``symbol``, ``action``,
            ``quantity``, ``entry_price``, ``exit_price``, ``pnl_pct``,
            ``bars_held``, ``reason``.
        """
        if prices_df is not None and "price" not in signals_df.columns:
            signals_df = signals_df.merge(
                prices_df[["timestamp", "price"]], on="timestamp", how="left"
            )

        # Ensure a spread column exists for the entry filter
        if "spread" not in signals_df.columns:
            signals_df["spread"] = 0.0

        trades: List[Dict] = []
        symbols = signals_df["symbol"].unique()

        for sym in symbols:
            sym_df = signals_df[signals_df["symbol"] == sym].reset_index(drop=True)
            position: Optional[Dict] = None

            for idx in range(len(sym_df)):
                row = sym_df.iloc[idx]

                if position is None:
                    # --- No open position: evaluate entry -----------------
                    entry = self.evaluate_entry(row)
                    if entry["action"] in ("BUY", "SELL"):
                        qty = self.compute_position_size(
                            signal_strength=row["signal_strength"],
                            price=row["price"],
                        )
                        if qty > 0:
                            position = {
                                "direction": entry["action"],
                                "entry_price": row["price"],
                                "entry_time": row["timestamp"],
                                "quantity": qty,
                                "bars_held": 0,
                            }
                else:
                    # --- Open position: evaluate exit ---------------------
                    position["bars_held"] += 1
                    exit_decision = self.evaluate_exit(position, row)

                    if exit_decision["action"] == "EXIT":
                        exit_price = row["price"]
                        entry_price = position["entry_price"]
                        if position["direction"] == "BUY":
                            pnl_pct = (exit_price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - exit_price) / entry_price

                        trades.append(
                            {
                                "timestamp": position["entry_time"],
                                "exit_timestamp": row["timestamp"],
                                "symbol": sym,
                                "action": position["direction"],
                                "quantity": position["quantity"],
                                "entry_price": round(entry_price, 4),
                                "exit_price": round(exit_price, 4),
                                "pnl_pct": round(pnl_pct, 6),
                                "bars_held": position["bars_held"],
                                "reason": exit_decision["reason"],
                            }
                        )
                        position = None

        self.trade_log = pd.DataFrame(trades)
        return self.trade_log

    # ------------------------------------------------------------------
    # Trade statistics
    # ------------------------------------------------------------------

    def compute_trade_stats(
        self,
        trades_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Compute summary statistics for a set of trades.

        Parameters
        ----------
        trades_df : pd.DataFrame, optional
            Defaults to ``self.trade_log``.

        Returns
        -------
        dict
            Keys: ``total_trades``, ``win_rate``, ``avg_pnl_pct``,
            ``sharpe_approx``, ``profit_factor``, ``max_drawdown_pct``,
            ``avg_bars_held``, ``exit_reasons``.
        """
        df = trades_df if trades_df is not None else self.trade_log
        if df is None or df.empty:
            return {"total_trades": 0}

        wins = df[df["pnl_pct"] > 0]
        losses = df[df["pnl_pct"] <= 0]

        gross_profit = wins["pnl_pct"].sum() if not wins.empty else 0.0
        gross_loss = abs(losses["pnl_pct"].sum()) if not losses.empty else 1e-9

        # Equity curve for max drawdown
        equity = (1 + df["pnl_pct"]).cumprod()
        running_max = equity.cummax()
        drawdowns = (equity - running_max) / running_max
        max_dd = drawdowns.min()

        avg_pnl = df["pnl_pct"].mean()
        std_pnl = df["pnl_pct"].std()
        sharpe = (avg_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0

        return {
            "total_trades": len(df),
            "win_rate": round(len(wins) / len(df), 4),
            "avg_pnl_pct": round(avg_pnl, 6),
            "sharpe_approx": round(sharpe, 4),
            "profit_factor": round(gross_profit / gross_loss, 4),
            "max_drawdown_pct": round(max_dd, 6),
            "avg_bars_held": round(df["bars_held"].mean(), 2),
            "exit_reasons": df["reason"].value_counts().to_dict(),
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Import the signal generator to create demo data
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from signals.signal_generator import SignalGenerator

    print("=" * 72)
    print(" Trading Rule Engine — Demo")
    print("=" * 72)

    # 1. Generate signals
    gen = SignalGenerator()
    signals = gen.generate_signals()
    print(f"\nLoaded {len(signals):,} signal rows.")

    # 2. Run the rule engine
    engine = TradingRuleEngine()
    trades = engine.run_rules(signals)
    print(f"Executed {len(trades)} trades.\n")

    # 3. Compute statistics
    stats = engine.compute_trade_stats()
    print("--- Trade statistics ---")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    # 4. Position sizing demo
    print("\n--- Position sizing demo ---")
    for method in ("fixed_fraction", "kelly"):
        for strength in (0.3, 0.5, 0.8):
            qty = engine.compute_position_size(
                signal_strength=strength,
                method=method,
                price=150.0,
            )
            print(f"  method={method:15s}  strength={strength:.1f}  → {qty:>5d} shares")

    # 5. Sample trades
    if not trades.empty:
        print("\n--- Sample trades (first 10) ---")
        print(trades.head(10).to_string(index=False))

    print("\nTrading rule engine demo complete.")
