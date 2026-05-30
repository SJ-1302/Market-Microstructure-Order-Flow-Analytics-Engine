"""
Synthetic Market Data Generator
===============================

Orchestrates the generation of realistic, tick-by-tick financial market data
by combining:
1. Multivariate Hawkes processes (for order arrival timing and clustering)
2. A price-time priority Limit Order Book (LOB) matching engine
3. A stochastic mid-price process (Geometric Brownian Motion + Mean Reversion)

Outputs raw order book snapshots, trade prints, and order events in both
Parquet and CSV formats.
"""

from __future__ import annotations

import datetime
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from python.data_generation.hawkes_process import generate_intraday_arrivals
from python.data_generation.nse_config import (
    NSE_FNO_STOCKS,
    TRADING_SECONDS,
    intraday_volatility_profile,
)
from python.data_generation.order_book_engine import OrderBook, Side
from python.utils.helpers import ensure_dir, setup_logging

logger = setup_logging("SyntheticGenerator")


class SyntheticDataGenerator:
    """
    Main orchestrator for generating synthetic Level-2 order book and trade data.
    """

    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        
        # Extract parameters with fallbacks
        self.num_days = self.config.get("simulation", {}).get("num_days", 5)
        self.snapshot_interval_ms = self.config.get("simulation", {}).get("snapshot_interval_ms", 100)
        self.num_levels = self.config.get("simulation", {}).get("num_order_book_levels", 5)
        self.output_dir = self.config.get("paths", {}).get("data_raw", "data/raw")
        ensure_dir(self.output_dir)

    def generate_single_day(
        self, symbol: str, date: datetime.date, seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate one day of tick-by-tick data for a single symbol.
        """
        rng = np.random.default_rng(seed)
        random.seed(seed)
        
        stock_cfg = NSE_FNO_STOCKS.get(symbol, {
            "base_price": 1000.0,
            "lot_size": 100,
            "daily_volatility": 0.015,
            "avg_daily_volume": 10_000_000,
            "tick_size": 0.05,
        })
        base_price = stock_cfg["base_price"]
        tick_size = stock_cfg["tick_size"]
        lot_size = stock_cfg["lot_size"]

        # 1. Generate arrival times for limit, market, and cancellation events
        hawkes_params = {
            "limit_orders": {"mu": 1.5, "alpha": 0.4, "omega": 2.0},
            "market_orders": {"mu": 0.3, "alpha": 0.08, "omega": 1.5},
            "cancellations": {"mu": 1.2, "alpha": 0.3, "omega": 1.8},
        }

        arrivals = generate_intraday_arrivals(
            symbol_config=stock_cfg,
            date=date,
            volatility_profile=intraday_volatility_profile,
            hawkes_params=hawkes_params,
            seed=seed,
        )

        # 2. Combine and sort events
        event_list: List[Tuple[float, str]] = []
        for event_type, times in arrivals.items():
            for t in times:
                event_list.append((t, event_type))
        event_list.sort(key=lambda x: x[0])

        # 3. Setup Order Book
        book = OrderBook(symbol, tick_size=tick_size)
        active_order_ids: Dict[Side, List[int]] = {Side.BID: [], Side.ASK: []}

        # Setup Stochastic Mid-Price Process (Ornstein-Uhlenbeck)
        mid_price = base_price
        kappa = 0.5
        theta = base_price
        sigma = stock_cfg["daily_volatility"] / np.sqrt(TRADING_SECONDS)

        # 4. Initialize Order Book Depth around starting mid-price
        for i in range(1, 10):
            p_bid = mid_price - i * tick_size
            p_ask = mid_price + i * tick_size
            qty_bid = int(rng.lognormal(mean=4.0, sigma=0.5)) * lot_size
            qty_ask = int(rng.lognormal(mean=4.0, sigma=0.5)) * lot_size
            
            oid_bid = book.add_limit_order(Side.BID, p_bid, qty_bid, 0.0)
            oid_ask = book.add_limit_order(Side.ASK, p_ask, qty_ask, 0.0)
            
            active_order_ids[Side.BID].append(oid_bid)
            active_order_ids[Side.ASK].append(oid_ask)

        snapshots: List[Dict[str, Any]] = []
        next_snapshot_time = 0.0
        snapshot_step = self.snapshot_interval_ms / 1000.0

        last_time = 0.0

        # Loop through events
        for t, event_type in event_list:
            dt = t - last_time
            last_time = t

            # Update Mid-Price
            dw = rng.normal(0, np.sqrt(max(dt, 1e-6)))
            mid_price += kappa * (theta - mid_price) * dt + sigma * mid_price * dw
            mid_price = max(mid_price, tick_size)

            # Take periodic snapshots (highly optimized)
            if next_snapshot_time <= t:
                snap = book.get_snapshot(levels=self.num_levels)
                # Parse to schema format
                snap_row = {
                    "symbol": symbol,
                    "mid_price": snap["mid_price"] or mid_price,
                    "spread": snap["spread"] or tick_size,
                    "total_bid_volume": snap["total_bid_volume"],
                    "total_ask_volume": snap["total_ask_volume"],
                }
                for lv in range(self.num_levels):
                    snap_row[f"bid_price_{lv+1}"] = snap["bid_prices"][lv] if lv < len(snap["bid_prices"]) else np.nan
                    snap_row[f"bid_size_{lv+1}"] = snap["bid_quantities"][lv] if lv < len(snap["bid_quantities"]) else 0
                    snap_row[f"bid_orders_{lv+1}"] = snap["bid_order_counts"][lv] if lv < len(snap["bid_order_counts"]) else 0
                    snap_row[f"ask_price_{lv+1}"] = snap["ask_prices"][lv] if lv < len(snap["ask_prices"]) else np.nan
                    snap_row[f"ask_size_{lv+1}"] = snap["ask_quantities"][lv] if lv < len(snap["ask_quantities"]) else 0
                    snap_row[f"ask_orders_{lv+1}"] = snap["ask_order_counts"][lv] if lv < len(snap["ask_order_counts"]) else 0
                
                # Compute microprice
                if snap["bid_prices"] and snap["ask_prices"]:
                    b1, a1 = snap["bid_prices"][0], snap["ask_prices"][0]
                    bs1, as1 = snap["bid_quantities"][0], snap["ask_quantities"][0]
                    total_sz = bs1 + as1
                    snap_row["microprice"] = (b1 * as1 + a1 * bs1) / total_sz if total_sz > 0 else mid_price
                else:
                    snap_row["microprice"] = mid_price

                base_dt = datetime.datetime.combine(date, datetime.time(9, 15, 0))
                while next_snapshot_time <= t:
                    snap_row_copy = snap_row.copy()
                    snap_row_copy["timestamp"] = (base_dt + datetime.timedelta(seconds=next_snapshot_time)).strftime("%Y-%m-%d %H:%M:%S.%f")
                    snapshots.append(snap_row_copy)
                    next_snapshot_time += snapshot_step

            # Process the event
            if event_type == "limit_orders":
                side = Side.BID if rng.random() < 0.5 else Side.ASK
                offset = int(rng.exponential(scale=3.0)) + 1
                offset = min(offset, 30)

                qty = int(rng.lognormal(mean=3.5, sigma=0.8)) * lot_size
                qty = max(qty, lot_size)

                if side == Side.BID:
                    p = book._round_price(mid_price - offset * tick_size)
                else:
                    p = book._round_price(mid_price + offset * tick_size)

                oid = book.add_limit_order(side, p, qty, t)
                active_order_ids[side].append(oid)

            elif event_type == "cancellations":
                side = Side.BID if rng.random() < 0.5 else Side.ASK
                if active_order_ids[side]:
                    oid_to_cancel = rng.choice(active_order_ids[side])
                    book.cancel_order(oid_to_cancel, t)
                    active_order_ids[side].remove(oid_to_cancel)

            elif event_type == "market_orders":
                side = Side.BID if rng.random() < 0.5 else Side.ASK
                qty = int(rng.lognormal(mean=3.8, sigma=0.7)) * lot_size
                qty = max(qty, lot_size)
                
                book.add_market_order(side, qty, t)

        # 5. Build final dataframes
        snapshots_df = pd.DataFrame(snapshots)
        base_dt = datetime.datetime.combine(date, datetime.time(9, 15, 0))
        
        trades_list = []
        for tr in book.trades:
            trades_list.append({
                "timestamp": (base_dt + datetime.timedelta(seconds=tr.timestamp)).strftime("%Y-%m-%d %H:%M:%S.%f"),
                "symbol": symbol,
                "price": tr.price,
                "size": tr.quantity,
                "side": tr.aggressor_side.value,
                "trade_id": tr.trade_id,
                "order_id": tr.maker_order_id,
            })
        trades_df = pd.DataFrame(trades_list)

        events_list = []
        for ev in book.events:
            events_list.append({
                "timestamp": (base_dt + datetime.timedelta(seconds=ev.timestamp)).strftime("%Y-%m-%d %H:%M:%S.%f"),
                "symbol": symbol,
                "event_type": ev.event_type,
                "order_id": ev.order_id,
                "side": ev.side,
                "price": ev.price,
                "size": ev.quantity,
                "remaining_size": ev.quantity if ev.event_type == "LIMIT_ORDER" else 0,
                "queue_position": 1,
            })
        events_df = pd.DataFrame(events_list)

        return snapshots_df, trades_df, events_df

    def generate_dataset(self, symbols: List[str] = None, num_days: int = None) -> None:
        """
        Generate raw dataset for all selected symbols over multiple days.
        """
        if symbols is None:
            symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
        if num_days is None:
            num_days = self.num_days

        logger.info(f"Generating data for {len(symbols)} symbols over {num_days} days...")
        
        start_date = datetime.date(2025, 3, 1)

        all_snapshots = []
        all_trades = []
        all_events = []

        total_runs = len(symbols) * num_days
        pbar = tqdm(total=total_runs, desc="Simulating Markets")

        seed_base = 1000
        for i in range(num_days):
            date = start_date + datetime.timedelta(days=i)
            # Skip weekends
            if date.weekday() >= 5:
                continue
            
            for symbol in symbols:
                seed = seed_base + i * 100 + list(NSE_FNO_STOCKS.keys()).index(symbol)
                snap_df, trade_df, event_df = self.generate_single_day(symbol, date, seed)
                
                all_snapshots.append(snap_df)
                all_trades.append(trade_df)
                all_events.append(event_df)
                
                pbar.update(1)
        pbar.close()

        # Combine
        snapshots_df = pd.concat(all_snapshots, ignore_index=True)
        trades_df = pd.concat(all_trades, ignore_index=True)
        events_df = pd.concat(all_events, ignore_index=True)

        # Convert timestamp strings to datetime64[ns]
        for df in [snapshots_df, trades_df, events_df]:
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")

        # Save to Parquet
        logger.info("Saving datasets to data/raw/ as Parquet...")
        snapshots_df.to_parquet(os.path.join(self.output_dir, "order_book_snapshots.parquet"), index=False)
        trades_df.to_parquet(os.path.join(self.output_dir, "trade_prints.parquet"), index=False)
        events_df.to_parquet(os.path.join(self.output_dir, "order_events.parquet"), index=False)

        # Save to CSV for R script compatibility (R's data_loader uses fread on CSVs)
        logger.info("Saving datasets to data/raw/ as CSV...")
        snapshots_df.to_csv(os.path.join(self.output_dir, "orderbook.csv"), index=False)
        trades_df.to_csv(os.path.join(self.output_dir, "trades.csv"), index=False)
        events_df.to_csv(os.path.join(self.output_dir, "order_events.csv"), index=False)
        
        logger.info("Data generation complete!")


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_dataset(symbols=["RELIANCE", "TCS", "HDFCBANK"], num_days=3)
