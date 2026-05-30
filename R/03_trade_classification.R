###############################################################################
# 03_trade_classification.R — Classified Trades and TQR Analysis
# Market Microstructure & Order Flow Analytics Engine
#
# Analysis:
#   1. Read enriched trades data
#   2. Apply Lee-Ready classification (quote test + tick test)
#   3. Calculate Tick test classification
#   4. Compute signed volume: trade_sign * trade_qty
#   5. Compute Trade-to-Quote Ratio (TQR) per time interval (e.g., 5-minute)
#   6. Save classified trades and generate plots
###############################################################################

# ── Source utilities ─────────────────────────────────────────────────────────
source("R/utils/helpers.R")
source("R/utils/metrics.R")
setup_project()

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
})

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

load_trade_class_data <- function() {
  trade_path <- file.path(DATA_PROCESSED, "trades_enriched.rds")
  book_path <- file.path(DATA_PROCESSED, "orderbook_processed.rds")

  if (!file.exists(trade_path)) {
    log_msg("Processed trade data not found — running data loader", level = "WARN")
    source("R/01_data_loader.R")
  }

  dt_trades <- readRDS(trade_path)
  dt_book <- readRDS(book_path)
  list(trades = dt_trades, book = dt_book)
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS TRADE CLASSIFICATION & TQR
# ═══════════════════════════════════════════════════════════════════════════════

classify_and_compute_tqr <- function(dt_trades, dt_book) {
  log_msg("Classifying trades using Lee-Ready and Tick tests...")

  # Lee-Ready classification (if not already computed)
  if (!"trade_sign" %in% names(dt_trades)) {
    dt_trades[, trade_sign := classify_trade_side(trade_price, bid1, ask1), by = symbol]
  }
  dt_trades[, lr_sign := trade_sign]

  # Tick-test classification
  dt_trades[, price_diff := c(0, diff(trade_price)), by = symbol]
  dt_trades[, tick_sign := 0L]
  dt_trades[price_diff > 0, tick_sign := 1L]
  dt_trades[price_diff < 0, tick_sign := -1L]
  
  # Fill zero price diff (flat tick) with last non-zero sign
  # Uses data.table shift/locf logic
  dt_trades[, tick_sign := as.integer(nafill(shift(tick_sign, type = "lag"), type = "locf")), by = symbol]
  dt_trades[is.na(tick_sign) | tick_sign == 0, tick_sign := 1L]
  dt_trades[, price_diff := NULL]

  # Compute signed volume
  dt_trades[, signed_volume := lr_sign * trade_qty]

  # Compute Trade-to-Quote Ratio (TQR) per 5-minute interval
  log_msg("Computing Trade-to-Quote Ratios...")
  dt_book[, interval_5min := as.POSIXct(cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata")]
  dt_trades[, interval_5min := as.POSIXct(cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata")]

  quote_counts <- dt_book[, .(quotes = .N), by = .(symbol, interval_5min)]
  trade_counts <- dt_trades[, .(trades = .N), by = .(symbol, interval_5min)]

  setkey(quote_counts, symbol, interval_5min)
  setkey(trade_counts, symbol, interval_5min)
  
  tqr_dt <- merge(quote_counts, trade_counts, all = TRUE)
  tqr_dt[is.na(quotes), quotes := 0]
  tqr_dt[is.na(trades), trades := 0]
  tqr_dt[, tqr := quotes / pmax(trades, 1)]

  # Add TQR back to trades
  dt_trades <- merge(dt_trades, tqr_dt[, .(symbol, interval_5min, tqr)], by = c("symbol", "interval_5min"), all.x = TRUE)
  
  # Add BVC (Bulk Volume Classification) proxy - percentage of buy volume per interval
  tqr_dt <- merge(tqr_dt, dt_trades[, .(
    buy_vol = sum(trade_qty[lr_sign == 1]),
    total_vol = sum(trade_qty)
  ), by = .(symbol, interval_5min)], by = c("symbol", "interval_5min"), all.x = TRUE)
  tqr_dt[, bvc_buy_pct := buy_vol / pmax(total_vol, 1)]
  tqr_dt[is.na(bvc_buy_pct), bvc_buy_pct := 0.5]

  dt_trades <- merge(dt_trades, tqr_dt[, .(symbol, interval_5min, bvc_buy_pct)], by = c("symbol", "interval_5min"), all.x = TRUE)

  list(trades = dt_trades, tqr = tqr_dt)
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

generate_trade_plots <- function(dt_trades, tqr_dt) {
  log_msg("Generating trade classification and TQR plots...")

  # Sample a single symbol
  sym_sample <- dt_trades[, unique(symbol)][1]
  dt_sample <- dt_trades[symbol == sym_sample]
  
  # Plot 1: Cumulative flow vs Price
  dt_sample[, cum_signed_vol := cumsum(signed_volume)]
  
  p1 <- ggplot(dt_sample, aes(x = timestamp)) +
    geom_line(aes(y = trade_price, colour = "Price"), linewidth = 0.8) +
    geom_line(aes(y = cum_signed_vol / 100 + trade_price[1], colour = "Cum Signed Vol (scaled)"), linewidth = 1.0) +
    scale_colour_manual(
      values = c("Price" = "#00D4AA", "Cum Signed Vol (scaled)" = "#FF5252"),
      name = "Series"
    ) +
    labs(
      title = paste("Cumulative Signed Order Flow vs. Price —", sym_sample),
      subtitle = "Dynamic visual overlay demonstrating price response to order flow pressure",
      x = "Time",
      y = "Price / Cumulative Flow"
    ) +
    THEME_DARK
  
  save_plot(p1, "cumulative_flow_vs_price.png", width = 14, height = 7)

  # Plot 2: TQR Heatmap
  tqr_dt[, hour := as.integer(format(interval_5min, "%H"))]
  tqr_hourly <- tqr_dt[, .(avg_tqr = mean(tqr)), by = .(symbol, hour)]
  
  p2 <- ggplot(tqr_hourly, aes(x = factor(hour), y = symbol, fill = avg_tqr)) +
    geom_tile() +
    scale_fill_viridis_c(option = "inferno", name = "Avg TQR") +
    labs(
      title = "Trade-to-Quote Ratio (TQR) Heatmap",
      subtitle = "Average TQR values across symbols and hourly bins (HFT activity proxy)",
      x = "Hour of Day",
      y = "Symbol"
    ) +
    THEME_DARK

  save_plot(p2, "tqr_heatmap.png", width = 12, height = 8)
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main_trade_classification <- function() {
  log_msg("=" |> rep(60) |> paste(collapse = ""))
  log_msg("TRADE CLASSIFICATION & TQR — Starting")
  log_msg("=" |> rep(60) |> paste(collapse = ""))

  # Load data
  data <- load_trade_class_data()

  # Classify and compute TQR
  results <- classify_and_compute_tqr(data$trades, data$book)

  # Save results
  classified_trades_out <- results$trades[, .(
    timestamp, symbol, price = trade_price, size = trade_qty, lr_sign, tick_sign, signed_volume, tqr, bvc_buy_pct
  )]
  fwrite(classified_trades_out, file.path(DATA_PROCESSED, "classified_trades.csv"))
  
  # Save back the enriched trades binary
  saveRDS(results$trades, file.path(DATA_PROCESSED, "trades_enriched.rds"))
  
  log_msg("Classified trades saved to data/processed/classified_trades.csv")

  # Generate plots
  generate_trade_plots(results$trades, results$tqr)

  log_msg("TRADE CLASSIFICATION & TQR — Complete")
  invisible(results$trades)
}

# Run if executed directly
if (sys.nframe() == 0L || !interactive()) {
  main_trade_classification()
}
