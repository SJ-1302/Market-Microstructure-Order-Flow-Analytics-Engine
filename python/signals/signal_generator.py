"""
Order Flow-Based Signal Generation Engine
==========================================

Generates composite trading signals from market microstructure metrics computed
by the R analytics pipeline.  Each raw metric (OBI, spread, Kyle's lambda,
VPIN) is converted to a standardized z-score, then combined into a single
composite signal normalized to [-1, +1].

Dependencies
------------
numpy, pandas, scipy (all listed in requirements.txt)

Example
-------
>>> gen = SignalGenerator()
>>> signals = gen.generate_signals()
>>> print(signals.head())
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Feature-engineering helpers (module-level, reusable)
# ---------------------------------------------------------------------------

def rolling_zscore(series: pd.Series, window: int = 500) -> pd.Series:
    """Compute a rolling z-score for *series*.

    Parameters
    ----------
    series : pd.Series
        Raw numeric series (e.g. order-book imbalance).
    window : int, default 500
        Lookback window in ticks / rows.

    Returns
    -------
    pd.Series
        Standardized z-score.  The first ``window - 1`` values are ``NaN``.
    """
    rolling_mean = series.rolling(window=window, min_periods=max(1, window // 4)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(1, window // 4)).std()
    # Avoid divide-by-zero: replace zero std with NaN, then forward-fill
    rolling_std = rolling_std.replace(0.0, np.nan).ffill().fillna(1.0)
    return (series - rolling_mean) / rolling_std


def momentum_signal(series: pd.Series, lookback: int = 20) -> pd.Series:
    """Rate-of-change momentum indicator.

    Parameters
    ----------
    series : pd.Series
        Price or metric series.
    lookback : int, default 20
        Number of periods for the rate-of-change calculation.

    Returns
    -------
    pd.Series
        Fractional change over *lookback* periods.
    """
    return series.pct_change(periods=lookback)


def regime_detector(
    volatility_series: pd.Series,
    threshold: float = 1.0,
) -> pd.Series:
    """Classify the market volatility regime.

    Parameters
    ----------
    volatility_series : pd.Series
        Rolling volatility (e.g. 20-period std of returns).
    threshold : float, default 1.0
        Number of standard deviations above the mean at which the regime
        switches from ``LOW_VOL`` to ``HIGH_VOL``.

    Returns
    -------
    pd.Series
        Categorical series with values ``'HIGH_VOL'`` or ``'LOW_VOL'``.
    """
    vol_mean = volatility_series.expanding(min_periods=20).mean()
    vol_std = volatility_series.expanding(min_periods=20).std().replace(0.0, np.nan).ffill().fillna(1.0)
    z = (volatility_series - vol_mean) / vol_std
    return pd.Series(
        np.where(z > threshold, "HIGH_VOL", "LOW_VOL"),
        index=volatility_series.index,
        name="regime",
    )


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    """Configuration for the signal generator."""

    obi_window: int = 500
    spread_window: int = 500
    lambda_window: int = 500
    vpin_window: int = 500
    weight_obi: float = 0.35
    weight_spread: float = 0.25
    weight_lambda: float = 0.20
    weight_vpin: float = 0.20
    composite_clip: float = 1.0
    classify_threshold: float = 0.3
    momentum_lookback: int = 20
    vol_regime_threshold: float = 1.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SignalGenerator:
    """Order flow-based signal generation engine.

    Loads market-microstructure metrics (typically computed in R and saved as
    CSVs), converts each metric into a z-score, and builds a weighted
    composite signal.

    Parameters
    ----------
    config : dict or SignalConfig, optional
        Override default settings.  Accepts either a ``SignalConfig`` instance
        or a plain ``dict`` whose keys match ``SignalConfig`` fields.

    Attributes
    ----------
    config : SignalConfig
    metrics : dict[str, pd.DataFrame]
        Loaded metric DataFrames keyed by symbol.
    signals : pd.DataFrame or None
        Generated signals (populated after ``generate_signals``).
    """

    def __init__(self, config: Optional[Dict | SignalConfig] = None) -> None:
        if config is None:
            self.config = SignalConfig()
        elif isinstance(config, dict):
            self.config = SignalConfig(**{
                k: v for k, v in config.items() if k in SignalConfig.__dataclass_fields__
            })
        else:
            self.config = config

        self.metrics: Dict[str, pd.DataFrame] = {}
        self.signals: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_metrics(self, metrics_dir: str = "data/processed/") -> Dict[str, pd.DataFrame]:
        """Load R-computed microstructure metrics from CSV files.

        Parameters
        ----------
        metrics_dir : str, default ``'data/processed/'``
            Directory containing CSV files.  Each file is expected to include
            columns for ``timestamp``, ``price``, ``obi``, ``spread``,
            ``kyle_lambda``, and ``vpin``.  The symbol name is inferred from
            the filename.

        Returns
        -------
        dict[str, pd.DataFrame]
            DataFrames keyed by symbol name.
        """
        csv_files = glob.glob(os.path.join(metrics_dir, "*.csv"))

        for fpath in csv_files:
            symbol = os.path.splitext(os.path.basename(fpath))[0].upper()
            df = pd.read_csv(fpath, parse_dates=["timestamp"] if "timestamp" in
                             pd.read_csv(fpath, nrows=0).columns else False)

            # Ensure required columns exist, fill with NaN if missing
            for col in ("obi", "spread", "kyle_lambda", "vpin", "price"):
                if col not in df.columns:
                    df[col] = np.nan

            if "timestamp" not in df.columns:
                df["timestamp"] = pd.date_range(
                    start="2024-01-02 09:30:00", periods=len(df), freq="s"
                )
            self.metrics[symbol] = df

        return self.metrics

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def compute_composite_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standardized component signals and the composite signal.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``obi``, ``spread``, ``kyle_lambda``, ``vpin``.

        Returns
        -------
        pd.DataFrame
            Input frame augmented with ``obi_z``, ``spread_z``,
            ``lambda_signal``, ``vpin_signal``, ``composite_signal``.
        """
        cfg = self.config

        # --- OBI z-score (positive = buy pressure) ----------------------
        df["obi_z"] = rolling_zscore(df["obi"], window=cfg.obi_window)

        # --- Spread z-score (positive = wide spread = unfavourable) ------
        df["spread_z"] = -rolling_zscore(df["spread"], window=cfg.spread_window)

        # --- Kyle's lambda signal (high lambda = low liquidity) ----------
        # Invert so that positive = good liquidity
        df["lambda_signal"] = -rolling_zscore(df["kyle_lambda"], window=cfg.lambda_window)

        # --- VPIN signal (high VPIN = toxicity → avoid) ------------------
        df["vpin_signal"] = -rolling_zscore(df["vpin"], window=cfg.vpin_window)

        # --- Weighted composite -----------------------------------------
        composite = (
            cfg.weight_obi * df["obi_z"].fillna(0.0)
            + cfg.weight_spread * df["spread_z"].fillna(0.0)
            + cfg.weight_lambda * df["lambda_signal"].fillna(0.0)
            + cfg.weight_vpin * df["vpin_signal"].fillna(0.0)
        )

        # Normalize to [-1, +1] using tanh squashing
        df["composite_signal"] = np.tanh(composite)

        return df

    # ------------------------------------------------------------------

    @staticmethod
    def classify_signal(
        composite: float | pd.Series,
        threshold: float = 0.3,
    ) -> str | pd.Series:
        """Classify a composite signal into BUY / SELL / NEUTRAL.

        Parameters
        ----------
        composite : float or pd.Series
            Composite signal value(s) in [-1, 1].
        threshold : float, default 0.3
            Absolute value above which the signal triggers a direction.

        Returns
        -------
        str or pd.Series
            ``'BUY'``, ``'SELL'``, or ``'NEUTRAL'``.
        """
        if isinstance(composite, pd.Series):
            return pd.Series(
                np.where(
                    composite > threshold,
                    "BUY",
                    np.where(composite < -threshold, "SELL", "NEUTRAL"),
                ),
                index=composite.index,
                name="signal_direction",
            )
        if composite > threshold:
            return "BUY"
        if composite < -threshold:
            return "SELL"
        return "NEUTRAL"

    # ------------------------------------------------------------------

    def generate_signals(
        self,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate trading signals for all (or selected) loaded symbols.

        Parameters
        ----------
        symbols : list[str], optional
            Subset of symbols to process.  Defaults to all loaded symbols.
            If no symbols have been loaded, synthetic demo data is created.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``symbol``, ``signal_strength``,
            ``signal_direction``, ``confidence``, ``obi_z``, ``spread_z``,
            ``lambda_signal``, ``vpin_signal``.
        """
        if not self.metrics:
            # Generate synthetic data so the engine is demonstrable standalone
            self.metrics = self._generate_synthetic_data()

        targets = symbols or list(self.metrics.keys())
        frames: List[pd.DataFrame] = []

        for sym in targets:
            if sym not in self.metrics:
                continue
            df = self.metrics[sym].copy()
            df = self.compute_composite_signal(df)

            direction = self.classify_signal(
                df["composite_signal"], self.config.classify_threshold
            )
            confidence = df["composite_signal"].abs()

            out = pd.DataFrame(
                {
                    "timestamp": df["timestamp"],
                    "symbol": sym,
                    "price": df["price"],
                    "signal_strength": df["composite_signal"],
                    "signal_direction": direction,
                    "confidence": confidence,
                    "obi_z": df["obi_z"],
                    "spread_z": df["spread_z"],
                    "lambda_signal": df["lambda_signal"],
                    "vpin_signal": df["vpin_signal"],
                }
            )
            frames.append(out)

        self.signals = pd.concat(frames, ignore_index=True)
        return self.signals

    # ------------------------------------------------------------------

    def save_signals(self, output_dir: str = "data/signals/") -> str:
        """Persist generated signals to CSV.

        Parameters
        ----------
        output_dir : str, default ``'data/signals/'``
            Target directory (created if it doesn't exist).

        Returns
        -------
        str
            Path to the saved CSV file.
        """
        if self.signals is None:
            raise RuntimeError("No signals generated yet. Call generate_signals() first.")

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "generated_signals.csv")
        self.signals.to_csv(out_path, index=False)
        return out_path

    # ------------------------------------------------------------------
    # Synthetic data for standalone demo
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_synthetic_data(
        n_ticks: int = 2000,
        seed: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """Create realistic synthetic microstructure data.

        Returns
        -------
        dict[str, pd.DataFrame]
            One DataFrame per symbol.
        """
        rng = np.random.default_rng(seed)
        symbols = ["AAPL", "MSFT"]
        result: Dict[str, pd.DataFrame] = {}

        for sym in symbols:
            timestamps = pd.date_range(
                start="2024-06-03 09:30:00", periods=n_ticks, freq="s"
            )
            # Random walk price
            log_returns = rng.normal(0.0, 0.0003, size=n_ticks)
            price = 150.0 * np.exp(np.cumsum(log_returns))

            # Order-book imbalance: auto-correlated
            obi_noise = rng.normal(0.0, 0.1, size=n_ticks)
            obi = np.zeros(n_ticks)
            obi[0] = obi_noise[0]
            for i in range(1, n_ticks):
                obi[i] = 0.95 * obi[i - 1] + obi_noise[i]

            # Spread: positive, mean-reverting
            spread = np.abs(rng.normal(0.02, 0.008, size=n_ticks))

            # Kyle's lambda: positive
            kyle_lambda = np.abs(rng.normal(0.0005, 0.0002, size=n_ticks))

            # VPIN: between 0 and 1
            vpin_raw = rng.beta(2, 5, size=n_ticks)
            vpin = pd.Series(vpin_raw).rolling(20, min_periods=1).mean().values

            result[sym] = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "price": price,
                    "obi": obi,
                    "spread": spread,
                    "kyle_lambda": kyle_lambda,
                    "vpin": vpin,
                }
            )
        return result


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print(" Market Microstructure Signal Generator — Demo")
    print("=" * 72)

    gen = SignalGenerator()
    signals = gen.generate_signals()

    print(f"\nGenerated {len(signals):,} signal rows for symbols: "
          f"{signals['symbol'].unique().tolist()}")
    print("\n--- Signal distribution ---")
    print(signals["signal_direction"].value_counts().to_string())

    print("\n--- Signal statistics ---")
    print(signals[["signal_strength", "confidence", "obi_z", "spread_z",
                    "lambda_signal", "vpin_signal"]].describe().round(4).to_string())

    print("\n--- Sample signals (first 10 rows) ---")
    print(signals.head(10).to_string(index=False))

    # Regime detection demo
    price = signals.loc[signals["symbol"] == "AAPL", "price"].reset_index(drop=True)
    returns = price.pct_change().dropna()
    vol = returns.rolling(20).std().dropna()
    regimes = regime_detector(vol, threshold=1.0)
    print(f"\n--- Volatility regime counts (AAPL) ---")
    print(regimes.value_counts().to_string())

    # Momentum demo
    mom = momentum_signal(price, lookback=20)
    print(f"\nMomentum signal (AAPL, last 5): {mom.tail().round(6).tolist()}")

    print("\nSignal generation complete.")
