"""
Python Visualization Engine
===========================

Generates all publication-quality, dark-themed microstructure and backtesting
plots in Python (replicating the R visualization suite) and saves them
to `reports/figures/`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from python.utils.helpers import ensure_dir, setup_logging

logger = setup_logging("VisualizationEngine")

# Define dark theme constants matching R theme
BG_COLOR = "#1a1a2e"
GRID_COLOR = "#2d2d44"
TEXT_COLOR = "#e0e0e0"
ACCENT_GREEN = "#00D4AA"
ACCENT_PURPLE = "#7B2FBE"
ACCENT_ORANGE = "#FFA726"
ACCENT_RED = "#FF5252"
ACCENT_BLUE = "#2196F3"


def apply_dark_theme():
    """Sets matplotlib styling parameters for a cohesive dark theme."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.5,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": "#c0c0c0",
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": "#a0a0a0",
        "ytick.color": "#a0a0a0",
        "font.family": "sans-serif",
        "font.size": 10,
        "legend.facecolor": BG_COLOR,
        "legend.edgecolor": "#444444",
        "savefig.facecolor": BG_COLOR,
        "savefig.edgecolor": BG_COLOR,
    })


def plot_spread_ushape(df_book: pd.DataFrame, output_dir: str):
    """Plot intraday U-shape spread profile."""
    logger.info("Plotting intraday spread profile (U-shape)...")
    
    df = df_book.copy()
    df["time_bin"] = df["timestamp"].dt.strftime("%H:%M")
    
    # Floor timestamps to 15-minute bins
    df["time_bin_15m"] = df["timestamp"].apply(
        lambda dt: f"{dt.hour:02d}:{(dt.minute // 15) * 15:02d}"
    )
    
    # Calculate stats
    grouped = df.groupby("time_bin_15m")["spread_bps"].agg(
        median="median",
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75)
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(
        grouped["time_bin_15m"], grouped["q25"], grouped["q75"],
        color=ACCENT_PURPLE, alpha=0.3, label="Interquartile Range"
    )
    ax.plot(
        grouped["time_bin_15m"], grouped["median"],
        color=ACCENT_GREEN, linewidth=2, marker="o", markersize=4, label="Median Spread"
    )
    
    ax.set_title("Intraday Spread Profile — U-Shape Pattern", pad=15, weight="bold", size=14)
    ax.set_xlabel("Time of Day (IST)", labelpad=10)
    ax.set_ylabel("Spread (basis points)", labelpad=10)
    
    # Format labels
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_intraday_ushape.png"), dpi=300)
    plt.close()


def plot_spread_cross_section(df_book: pd.DataFrame, output_dir: str):
    """Plot cross-sectional comparison of spreads by symbol."""
    logger.info("Plotting cross-sectional spread comparison...")
    
    symbols = df_book["symbol"].unique()
    data_to_plot = [df_book[df_book["symbol"] == sym]["spread_bps"].dropna() for sym in symbols]
    
    # Order by median spread
    medians = [np.median(d) for d in data_to_plot]
    sorted_indices = np.argsort(medians)
    symbols_sorted = [symbols[i] for i in sorted_indices]
    data_sorted = [data_to_plot[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw horizontal violin plot
    parts = ax.violinplot(data_sorted, vert=False, showmedians=True)
    
    # Style violins
    for pc in parts["bodies"]:
        pc.set_facecolor(ACCENT_PURPLE)
        pc.set_edgecolor("#9c27b0")
        pc.set_alpha(0.6)
    parts["cmedians"].set_colors(ACCENT_GREEN)
    parts["cmedians"].set_linewidth(2)
    
    ax.set_yticks(np.arange(1, len(symbols_sorted) + 1))
    ax.set_yticklabels(symbols_sorted)
    
    ax.set_title("Cross-Sectional Spread Distribution by Symbol", pad=15, weight="bold", size=14)
    ax.set_xlabel("Spread (basis points)", labelpad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_cross_section.png"), dpi=300)
    plt.close()


def plot_spread_rolling(df_book: pd.DataFrame, output_dir: str):
    """Plot rolling spread time series for the first symbol."""
    logger.info("Plotting rolling spread time series...")
    
    sym = df_book["symbol"].unique()[0]
    sub = df_book[df_book["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
    
    # Recompute rolling windows for clean plot
    sub["spread_roll_100"] = sub["pct_spread"].rolling(100, min_periods=1).mean() * 10000
    sub["spread_roll_500"] = sub["pct_spread"].rolling(500, min_periods=1).mean() * 10000
    sub["spread_roll_1000"] = sub["pct_spread"].rolling(1000, min_periods=1).mean() * 10000
    sub["spread_bps_raw"] = sub["pct_spread"] * 10000
    
    # Downsample if too large for performance
    if len(sub) > 5000:
        sub = sub.iloc[::len(sub)//5000 + 1].reset_index(drop=True)
        
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(sub["timestamp"], sub["spread_bps_raw"], color="#555555", alpha=0.25, label="Raw Spread")
    ax.plot(sub["timestamp"], sub["spread_roll_100"], color=ACCENT_ORANGE, linewidth=1.0, label="MA-100")
    ax.plot(sub["timestamp"], sub["spread_roll_500"], color=ACCENT_GREEN, linewidth=1.2, label="MA-500")
    ax.plot(sub["timestamp"], sub["spread_roll_1000"], color=ACCENT_RED, linewidth=1.5, label="MA-1000")
    
    ax.set_title(f"Rolling Spread Analysis — {sym}", pad=15, weight="bold", size=14)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Spread (basis points)", labelpad=10)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=30)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_rolling_timeseries.png"), dpi=300)
    plt.close()


def plot_obi_vs_returns(df_merged: pd.DataFrame, output_dir: str):
    """Plot OBI vs future returns correlation."""
    logger.info("Plotting OBI vs future returns correlation...")
    
    df = df_merged.copy()
    # Compute 5-minute lead returns as proxy for future returns
    df["future_return"] = df.groupby("symbol")["price"].pct_change().shift(-1)
    df = df.dropna(subset=["obi", "future_return"])
    
    # Downsample if too large
    if len(df) > 1000:
        df = df.sample(1000, random_seed=42).reset_index(drop=True)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(df["obi"], df["future_return"] * 100, color=ACCENT_PURPLE, alpha=0.5, edgecolors="none")
    
    # Fit regression line
    if len(df) > 5:
        m, c = np.polyfit(df["obi"], df["future_return"] * 100, 1)
        x_vals = np.linspace(df["obi"].min(), df["obi"].max(), 100)
        ax.plot(x_vals, m * x_vals + c, color=ACCENT_GREEN, linewidth=2, label=f"Trendline (slope: {m:.4f})")
        
    ax.set_title("Order Book Imbalance (OBI) vs. Future Returns", pad=15, weight="bold", size=14)
    ax.set_xlabel("OBI [-1.0, 1.0] (Exponentially Weighted)", labelpad=10)
    ax.set_ylabel("Future 5-Min Return (%)", labelpad=10)
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "obi_vs_returns.png"), dpi=300)
    plt.close()


def plot_adverse_selection_pin(df_adverse: pd.DataFrame, output_dir: str):
    """Plot Kyle's Lambda & PIN Estimates side-by-side."""
    logger.info("Plotting adverse selection PIN & Kyle's Lambda...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: PIN by Symbol
    df_adverse_sorted = df_adverse.sort_values("pin", ascending=True)
    ax1.barh(df_adverse_sorted["symbol"], df_adverse_sorted["pin"], color=ACCENT_PURPLE, edgecolor="#9c27b0", height=0.6)
    ax1.set_title("Probability of Informed Trading (PIN)", pad=15, weight="bold", size=12)
    ax1.set_xlabel("PIN Value", labelpad=10)
    
    # Plot 2: Kyle's Lambda by Symbol
    df_kyle_sorted = df_adverse.sort_values("kyle_lambda", ascending=True)
    ax2.barh(df_kyle_sorted["symbol"], df_kyle_sorted["kyle_lambda"] * 1000, color=ACCENT_GREEN, edgecolor="#00b0ff", height=0.6)
    ax2.set_title("Kyle's Lambda (Price Impact Coefficient)", pad=15, weight="bold", size=12)
    ax2.set_xlabel("Lambda (x10^-3)", labelpad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adverse_selection_pin.png"), dpi=300)
    plt.close()


def plot_price_impact_decay(df_impact: pd.DataFrame, output_dir: str):
    """Plot price impact decay curve."""
    logger.info("Plotting price impact decay profile...")
    
    # Average impact components across all trades
    horizons = ["1", "5", "10"]
    temp_impacts = [df_impact[f"temp_impact_{h}"].mean() * 10000 for h in horizons]
    perm_impacts = [df_impact[f"perm_impact_{h}"].mean() * 10000 for h in horizons]
    
    x = np.arange(len(horizons))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, temp_impacts, width, label="Temporary Impact (Slippage)", color=ACCENT_PURPLE, edgecolor="#9c27b0")
    rects2 = ax.bar(x + width/2, perm_impacts, width, label="Permanent Impact (Adverse Selection)", color=ACCENT_GREEN, edgecolor="#00b4aa")
    
    ax.set_ylabel("Price Impact (basis points)", labelpad=10)
    ax.set_title("Price Impact Decomposition by Trade Horizon", pad=15, weight="bold", size=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} Trades Ahead" for h in horizons])
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "price_impact_decay.png"), dpi=300)
    plt.close()


def plot_queue_survival(df_decay: pd.DataFrame, output_dir: str):
    """Plot empirical queue survival probabilities."""
    logger.info("Plotting queue position survival curves...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t_vals = np.linspace(0, 10000, 200)  # lifetimes in ms
    
    for idx, row in df_decay.iterrows():
        sym = row["symbol"]
        lambd = row["decay_rate_lambda"]
        # S(t) = exp(-λ * t)
        s_vals = np.exp(-lambd * t_vals)
        ax.plot(t_vals / 1000.0, s_vals, linewidth=1.5, label=sym)
        
    ax.set_title("Queue Survival Curves by Instrument", pad=15, weight="bold", size=14)
    ax.set_xlabel("Queue Lifetime (seconds)", labelpad=10)
    ax.set_ylabel("Survival Probability S(t)", labelpad=10)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "queue_survival_by_symbol.png"), dpi=300)
    plt.close()


def plot_backtest_equity(df_equity: pd.DataFrame, output_dir: str):
    """Plot walk-forward backtest equity curve."""
    logger.info("Plotting backtest equity curve...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Scale to Lakhs/Crores
    equity_lakhs = df_equity["equity"] / 100000.0
    
    ax.plot(df_equity["timestamp"], equity_lakhs, color=ACCENT_GREEN, linewidth=2, label="Strategy Equity (Adaptive Thresholds)")
    
    ax.set_title("Strategy Equity Curve — Walk-Forward Backtest", pad=15, weight="bold", size=14)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Portfolio Value (Lakhs)", labelpad=10)
    
    # Format axes
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{y:,.0f} L"))
    plt.xticks(rotation=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_equity_curve.png"), dpi=300)
    plt.close()


def plot_pnl_distribution(df_trades: pd.DataFrame, output_dir: str):
    """Plot trade PNL distribution."""
    logger.info("Plotting trade PNL distribution...")
    
    if df_trades.empty:
        logger.warning("No trades to plot for PNL distribution.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pnl_thousands = df_trades["pnl"] / 1000.0
    
    # Color bins based on profit/loss
    counts, bins, patches = ax.hist(pnl_thousands, bins=30, alpha=0.8, edgecolor="#1a1a2e")
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        if left >= 0:
            patch.set_facecolor(ACCENT_GREEN)
        else:
            patch.set_facecolor(ACCENT_RED)
            
    ax.set_title("Trade P&L Distribution", pad=15, weight="bold", size=14)
    ax.set_xlabel("P&L (₹ Thousands)", labelpad=10)
    ax.set_ylabel("Trade Count", labelpad=10)
    
    # Add vertical line at 0
    ax.axvline(0, color="#ffffff", linestyle="--", linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trade_pnl_distribution.png"), dpi=300)
    plt.close()


def plot_metric_correlation(df_cor: pd.DataFrame, output_dir: str):
    """Plot correlation matrix of microstructure metrics."""
    logger.info("Plotting correlation matrix heatmap...")
    
    cols = [c for c in df_cor.columns if c not in ["Unnamed: 0", "symbol"]]
    corr_matrix = df_cor[cols].values
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Draw heatmap
    im = ax.imshow(corr_matrix, cmap="plasma", vmin=-1, vmax=1)
    
    # Add annotations
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.4 else "black")
            
    # Add colorbar
    fig.colorbar(im, ax=ax, label="Correlation Coefficient")
    
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(cols)
    plt.xticks(rotation=45, ha="right")
    
    ax.set_title("Microstructure Metric Correlation Matrix", pad=15, weight="bold", size=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_metric_correlation.png"), dpi=300)
    plt.close()


def main():
    processed_dir = "data/processed"
    output_dir = "reports/figures"
    ensure_dir(output_dir)
    apply_dark_theme()
    
    # Load processed data
    logger.info("Loading processed files for plotting...")
    df_book = pd.read_csv(os.path.join(processed_dir, "spread_metrics.csv"), parse_dates=["timestamp"])
    df_obi = pd.read_csv(os.path.join(processed_dir, "obi_metrics.csv"), parse_dates=["timestamp"])
    df_trades = pd.read_csv(os.path.join(processed_dir, "classified_trades.csv"), parse_dates=["timestamp"])
    df_adverse = pd.read_csv(os.path.join(processed_dir, "adverse_selection.csv"))
    df_vpin = pd.read_csv(os.path.join(processed_dir, "vpin_metrics.csv"), parse_dates=["timestamp"])
    df_decay = pd.read_csv(os.path.join(processed_dir, "queue_decay.csv"))
    df_cor = pd.read_csv(os.path.join(processed_dir, "correlation_matrix.csv"))
    
    # Merged data for OBI correlation
    # We combine them by timestamp using our pre-constructed merged file
    df_merged = pd.read_csv(os.path.join(processed_dir, "microstructure_merged_5min.csv"), parse_dates=["timestamp"])
    df_signals = pd.read_csv("data/signals/entry_exit_signals.csv", parse_dates=["timestamp"])
    df_merged = pd.merge(df_merged, df_signals[["timestamp", "symbol", "price"]], on=["timestamp", "symbol"], how="left")
    df_merged["price"] = df_merged["price"].ffill().bfill()
    
    # Backtest data
    df_equity = pd.read_csv(os.path.join(processed_dir, "backtest_equity_curve.csv"), parse_dates=["timestamp"])
    df_backtest_trades = pd.read_csv(os.path.join(processed_dir, "backtest_trades.csv"), parse_dates=["entry_time", "exit_time"])
    
    # Generate all plots
    plot_spread_ushape(df_book, output_dir)
    plot_spread_cross_section(df_book, output_dir)
    plot_spread_rolling(df_book, output_dir)
    plot_obi_vs_returns(df_merged, output_dir)
    plot_adverse_selection_pin(df_adverse, output_dir)
    
    df_impact = pd.read_csv(os.path.join(processed_dir, "price_impact.csv"), parse_dates=["timestamp"])
    
    # For price impact, we use price_impact.csv as it contains permanent/temporary impact components
    plot_price_impact_decay(df_impact, output_dir)
    plot_queue_survival(df_decay, output_dir)
    
    # Backtest plots
    plot_backtest_equity(df_equity, output_dir)
    plot_pnl_distribution(df_backtest_trades, output_dir)
    
    # Correlation plot
    plot_metric_correlation(df_cor, output_dir)
    
    logger.info("Successfully generated all visual diagnostic figures and saved to reports/figures/")


if __name__ == "__main__":
    main()
