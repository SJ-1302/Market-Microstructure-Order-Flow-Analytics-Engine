"""
Entry/Exit Signal Generator for R Backtester
============================================

Loads processed microstructure metrics from `data/processed/`, computes
composite signals using `SignalGenerator`, runs an entry/exit state machine,
and writes `data/signals/entry_exit_signals.csv` for the R backtester.
"""

import os
import datetime
import pandas as pd
import numpy as np
from python.signals.signal_generator import SignalGenerator
from python.utils.helpers import ensure_dir, setup_logging

logger = setup_logging("EntryExitSignalGenerator")


def load_and_merge_metrics(processed_dir: str = "data/processed/") -> pd.DataFrame:
    """Loads all processed CSVs and aggregates them to a 5-minute frequency."""
    logger.info("Loading processed files for merging...")
    
    df_spread = pd.read_csv(os.path.join(processed_dir, "spread_metrics.csv"), parse_dates=["timestamp"])
    df_obi = pd.read_csv(os.path.join(processed_dir, "obi_metrics.csv"), parse_dates=["timestamp"])
    df_trades = pd.read_csv(os.path.join(processed_dir, "classified_trades.csv"), parse_dates=["timestamp"])
    df_vpin = pd.read_csv(os.path.join(processed_dir, "vpin_metrics.csv"), parse_dates=["timestamp"])
    df_adverse = pd.read_csv(os.path.join(processed_dir, "adverse_selection.csv"))

    logger.info("Binning metrics into 5-minute intervals...")
    # Floor to 5-minute intervals
    df_spread["timestamp_5m"] = df_spread["timestamp"].dt.floor("5min")
    df_obi["timestamp_5m"] = df_obi["timestamp"].dt.floor("5min")
    df_trades["timestamp_5m"] = df_trades["timestamp"].dt.floor("5min")
    df_vpin["timestamp_5m"] = df_vpin["timestamp"].dt.floor("5min")

    # Aggregate
    spread_5m = df_spread.groupby(["symbol", "timestamp_5m"])["pct_spread"].mean().reset_index()
    obi_5m = df_obi.groupby(["symbol", "timestamp_5m"])["obi_exp_weight"].mean().reset_index()
    trades_5m = df_trades.groupby(["symbol", "timestamp_5m"])["price"].mean().reset_index()
    vpin_5m = df_vpin.groupby(["symbol", "timestamp_5m"])["vpin"].mean().reset_index()

    # Merge
    logger.info("Merging datasets...")
    m1 = pd.merge(spread_5m, obi_5m, on=["symbol", "timestamp_5m"], how="outer")
    m2 = pd.merge(m1, trades_5m, on=["symbol", "timestamp_5m"], how="outer")
    m3 = pd.merge(m2, vpin_5m, on=["symbol", "timestamp_5m"], how="outer")
    m4 = pd.merge(m3, df_adverse[["symbol", "kyle_lambda"]], on="symbol", how="left")

    m4 = m4.rename(columns={
        "timestamp_5m": "timestamp",
        "pct_spread": "spread",
        "obi_exp_weight": "obi",
        "vpin": "vpin",
    })

    # Sort and clean
    m4 = m4.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    for col in ["spread", "obi", "price", "vpin", "kyle_lambda"]:
        m4[col] = m4.groupby("symbol")[col].ffill().bfill().fillna(0.0)

    return m4


def generate_entry_exit_signals(df: pd.DataFrame, entry_thresh: float = 0.35, exit_thresh: float = 0.10) -> pd.DataFrame:
    """Runs the state machine to generate ENTRY_LONG, ENTRY_SHORT, EXIT_LONG, EXIT_SHORT."""
    logger.info(f"Generating signals (entry_thresh={entry_thresh}, exit_thresh={exit_thresh})...")
    
    signals_list = []
    
    for sym, group in df.groupby("symbol"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        position = 0  # 0: flat, 1: long, -1: short
        
        signal_types = []
        for idx in range(len(group)):
            row = group.iloc[idx]
            sig_strength = row["composite_signal"]
            
            sig_type = "HOLD"
            
            if position == 0:
                if sig_strength > entry_thresh:
                    sig_type = "ENTRY_LONG"
                    position = 1
                elif sig_strength < -entry_thresh:
                    sig_type = "ENTRY_SHORT"
                    position = -1
            elif position == 1:
                # Exit Long conditions
                if sig_strength < exit_thresh:
                    sig_type = "EXIT_LONG"
                    position = 0
                elif sig_strength < -entry_thresh:
                    sig_type = "EXIT_LONG"  # Signal reversal exit
                    position = 0
            elif position == -1:
                # Exit Short conditions
                if sig_strength > -exit_thresh:
                    sig_type = "EXIT_SHORT"
                    position = 0
                elif sig_strength > entry_thresh:
                    sig_type = "EXIT_SHORT"  # Signal reversal exit
                    position = 0
                    
            signal_types.append(sig_type)
            
        group["signal_type"] = signal_types
        signals_list.append(group)
        
    return pd.concat(signals_list, ignore_index=True)


def main():
    # 1. Load and merge
    df_merged = load_and_merge_metrics()
    
    # 2. Run signal generator with shorter rolling windows for 5min interval data
    config_override = {
        "obi_window": 10,
        "spread_window": 10,
        "lambda_window": 10,
        "vpin_window": 10
    }
    gen = SignalGenerator(config=config_override)
    # Populate SignalGenerator's internal metrics dict so it doesn't fall back to synthetic data
    for sym, group in df_merged.groupby("symbol"):
        gen.metrics[sym] = group
        
    df_signals = gen.generate_signals()
    
    # Merge composite signal back to our df
    df_merged["composite_signal"] = df_signals["signal_strength"]
    
    # 3. Generate entry exit signals
    # We use optimized thresholds (e.g. entry=0.35, exit=0.10)
    df_final = generate_entry_exit_signals(df_merged, entry_thresh=0.35, exit_thresh=0.10)
    
    # 4. Save to data/signals/entry_exit_signals.csv
    signals_dir = "data/signals"
    ensure_dir(signals_dir)
    output_path = os.path.join(signals_dir, "entry_exit_signals.csv")
    
    df_final[["timestamp", "symbol", "signal_type", "composite_signal", "price"]].to_csv(output_path, index=False)
    logger.info(f"Successfully saved entry/exit signals to: {output_path}")


if __name__ == "__main__":
    main()
