###############################################################################
# 02_order_imbalance.R — Order Imbalance & OBI Metrics
# Market Microstructure & Order Flow Analytics Engine
#
# Analysis:
#   1. Compute Level-1 OBI: (bid_qty1 - ask_qty1) / (bid_qty1 + ask_qty1)
#   2. Compute multi-level weighted OBI across 5 levels
#   3. Compute volume imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol)
#   4. Compute rolling OBI metrics over multiple tick windows (50, 200, 1000)
#   5. Analyze predictive power: correlation of OBI(t) with future returns
#   6. Save OBI metrics and generate visualizations
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

load_obi_data <- function() {
  book_path <- file.path(DATA_PROCESSED, "orderbook_processed.rds")
  trade_path <- file.path(DATA_PROCESSED, "trades_enriched.rds")

  if (!file.exists(book_path)) {
    log_msg("Processed order book not found — running data loader", level = "WARN")
    source("R/01_data_loader.R")
  }
  
  dt_book <- readRDS(book_path)
  dt_trades <- if (file.exists(trade_path)) readRDS(trade_path) else NULL

  list(book = dt_book, trades = dt_trades)
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE OBI METRICS
# ═══════════════════════════════════════════════════════════════════════════════

compute_all_obi <- function(dt_book, dt_trades = NULL) {
  log_msg("Computing order book imbalance metrics...")

  # Level-1 OBI
  dt_book[, oi_level1 := compute_obi(bid_qty1, ask_qty1)]

  # Multi-level weighted OBI (distance decay)
  bid_cols <- paste0("bid_qty", 1:5)
  ask_cols <- paste0("ask_qty", 1:5)
  
  dt_book[, obi_exp_weight := compute_weighted_obi(
    .SD[, ..bid_cols], .SD[, ..ask_cols], levels = 5L
  )]
  
  # Equal weights OBI
  dt_book[, obi_equal_weight := (rowSums(.SD[, ..bid_cols]) - rowSums(.SD[, ..ask_cols])) / 
           (rowSums(.SD[, ..bid_cols]) + rowSums(.SD[, ..ask_cols]))]
  dt_book[is.nan(obi_equal_weight), obi_equal_weight := 0]

  # Linear decay weights OBI
  weights_linear <- c(5, 4, 3, 2, 1) / 15
  bid_weighted_lin <- as.matrix(dt_book[, ..bid_cols]) %*% weights_linear
  ask_weighted_lin <- as.matrix(dt_book[, ..ask_cols]) %*% weights_linear
  dt_book[, obi_linear_weight := (bid_weighted_lin - ask_weighted_lin) / (bid_weighted_lin + ask_weighted_lin)]
  dt_book[is.nan(obi_linear_weight), obi_linear_weight := 0]

  # Total volumes
  dt_book[, total_bid_volume := as.integer(rowSums(.SD[, ..bid_cols]))]
  dt_book[, total_ask_volume := as.integer(rowSums(.SD[, ..ask_cols]))]
  dt_book[, depth_ratio := total_bid_volume / total_ask_volume]

  # Rolling OBI momentum
  dt_book[, obi_momentum_50 := frollmean(oi_level1, n = 50L, align = "right", na.rm = TRUE), by = symbol]
  dt_book[, obi_momentum_200 := frollmean(oi_level1, n = 200L, align = "right", na.rm = TRUE), by = symbol]
  dt_book[, obi_volatility := frollapply(oi_level1, n = 200L, FUN = sd, align = "right"), by = symbol]

  log_msg("OBI calculations complete")
  dt_book
}

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIVE POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

analyze_predictive_power <- function(dt_book) {
  log_msg("Analyzing OBI predictive power for future returns...")

  # Compute future returns over multiple horizons (ticks): 1, 5, 10, 50 ticks ahead
  dt_book[, ret_fwd_1 := shift(log_return, n = 1L, type = "lead"), by = symbol]
  dt_book[, ret_fwd_5 := shift(rollsum(log_return, k = 5L, fill = NA), n = 5L, type = "lead"), by = symbol]
  dt_book[, ret_fwd_10 := shift(rollsum(log_return, k = 10L, fill = NA), n = 10L, type = "lead"), by = symbol]

  # Compute correlation per symbol
  cor_summary <- dt_book[, .(
    cor_1 = cor(oi_level1, ret_fwd_1, use = "pairwise.complete.obs"),
    cor_5 = cor(oi_level1, ret_fwd_5, use = "pairwise.complete.obs"),
    cor_10 = cor(oi_level1, ret_fwd_10, use = "pairwise.complete.obs")
  ), by = symbol]

  log_msg("Correlation between OBI and future returns:")
  print(cor_summary)
  
  cor_summary
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

generate_obi_plots <- function(dt_book) {
  log_msg("Generating OBI plots...")

  # Sample a single symbol for visualization efficiency
  sym_sample <- dt_book[, unique(symbol)][1]
  dt_sample <- dt_book[symbol == sym_sample]
  if (nrow(dt_sample) > 5000) {
    dt_sample <- dt_sample[seq(1, .N, length.out = 5000)]
  }

  # Plot 1: OBI Comparison (Level 1 vs Multi-Level Weighted)
  p1 <- ggplot(dt_sample, aes(x = timestamp)) +
    geom_line(aes(y = oi_level1, colour = "Level-1 OBI"), alpha = 0.3, linewidth = 0.5) +
    geom_line(aes(y = obi_exp_weight, colour = "Exponentially Weighted OBI"), linewidth = 0.8) +
    scale_colour_manual(
      values = c("Level-1 OBI" = "#7B2FBE", "Exponentially Weighted OBI" = "#00D4AA"),
      name = "Metric"
    ) +
    labs(
      title = paste("Order Book Imbalance (OBI) Comparison —", sym_sample),
      subtitle = "Top-of-book (Level-1) OBI vs. Multi-level Exponentially Weighted OBI",
      x = "Time",
      y = "Imbalance [-1, 1]"
    ) +
    THEME_DARK
  
  save_plot(p1, "obi_metric_comparison.png", width = 14, height = 7)

  # Plot 2: OBI vs Future Returns
  # Clean data
  dt_plot2 <- dt_book[is.finite(oi_level1) & is.finite(ret_fwd_5)]
  if (nrow(dt_plot2) > 10000) {
    dt_plot2 <- dt_plot2[sample(.N, 10000)]
  }
  
  p2 <- ggplot(dt_plot2, aes(x = oi_level1, y = ret_fwd_5 * 10000)) +
    geom_point(alpha = 0.1, colour = "#00D4AA") +
    geom_smooth(method = "loess", colour = "#FF5252", linewidth = 1.2) +
    labs(
      title = "Order Book Imbalance (OBI) vs. Future Returns",
      subtitle = "OBI at t vs. 5-tick forward log return (bps)",
      x = "Level-1 OBI",
      y = "Future Return (basis points)"
    ) +
    THEME_DARK

  save_plot(p2, "obi_predictive_power.png", width = 10, height = 8)

  log_msg("OBI plots generated successfully")
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main_order_imbalance <- function() {
  log_msg("=" |> rep(60) |> paste(collapse = ""))
  log_msg("ORDER IMBALANCE METRICS — Starting")
  log_msg("=" |> rep(60) |> paste(collapse = ""))

  # Load data
  data <- load_obi_data()

  # Compute all OBI metrics
  result_book <- compute_all_obi(data$book, data$trades)

  # Analyze predictive power
  analyze_predictive_power(result_book)

  # Save results
  obi_metrics <- result_book[, .(
    timestamp, symbol, oi_level1, obi_equal_weight, obi_exp_weight, 
    obi_linear_weight, total_bid_volume, total_ask_volume, depth_ratio,
    obi_momentum_50, obi_momentum_200, obi_volatility
  )]
  fwrite(obi_metrics, file.path(DATA_PROCESSED, "obi_metrics.csv"))
  
  # Save processed order book back
  saveRDS(result_book, file.path(DATA_PROCESSED, "orderbook_processed.rds"))
  
  log_msg("OBI metrics saved to data/processed/obi_metrics.csv")

  # Generate plots
  generate_obi_plots(result_book)

  log_msg("ORDER IMBALANCE METRICS — Complete")
  invisible(result_book)
}

# Helper rollsum function
rollsum <- function(x, k, fill = NA) {
  res <- frollsum(x, n = k, align = "right")
  res
}

# Run if executed directly
if (sys.nframe() == 0L || !interactive()) {
  main_order_imbalance()
}
