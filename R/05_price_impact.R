###############################################################################
# 05_price_impact.R — Price Impact of Trades (Amihud & Impact Decomposition)
# Market Microstructure & Order Flow Analytics Engine
#
# Analysis:
#   1. Compute Amihud Illiquidity Ratio: |Return| / DollarVolume * 1e6
#   2. Decompose impact into Permanent and Temporary components using realized spread
#   3. Analyze price impact across trade size quintiles
#   4. Estimate impact asymmetry (BUY vs SELL)
#   5. Save metrics and generate plots
#
# References:
#   - Amihud (2002): "Illiquidity and stock returns: cross-section and time-series effects"
#   - Hasbrouck (2007): "Empirical Market Microstructure"
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

load_impact_data <- function() {
  trade_path <- file.path(DATA_PROCESSED, "trades_enriched.rds")

  if (!file.exists(trade_path)) {
    log_msg("Enriched trades not found — running loader", level = "WARN")
    source("R/04_adverse_selection.R")
  }
  
  readRDS(trade_path)
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE AMIHUD ILLIQUIDITY
# ═══════════════════════════════════════════════════════════════════════════════

compute_amihud_illiq <- function(dt_trades) {
  log_msg("Computing Amihud Illiquidity Ratio...")

  # Aggregate returns and dollar volume to 5-minute bins
  dt_trades[, interval_5min := as.POSIXct(cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata")]
  
  daily_metrics <- dt_trades[, .(
    ret = abs(last(trade_price) - first(trade_price)) / first(trade_price),
    dvol = sum(dollar_volume)
  ), by = .(symbol, interval_5min)]
  
  daily_metrics[, amihud_illiq := compute_amihud(ret, dvol)]
  daily_metrics[is.na(amihud_illiq) | !is.finite(amihud_illiq), amihud_illiq := 0]

  # Rolling 20-period Amihud
  daily_metrics[, amihud_illiq_rolling := frollmean(amihud_illiq, n = 20L, align = "right", fill = 0), by = symbol]

  # Merge back to trades
  dt_trades <- merge(dt_trades, daily_metrics[, .(symbol, interval_5min, amihud_illiq, amihud_illiq_rolling)], by = c("symbol", "interval_5min"), all.x = TRUE)
  
  log_msg("Amihud illiquidity computed")
  list(trades = dt_trades, metrics = daily_metrics)
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORARY VS PERMANENT IMPACT DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

decompose_price_impact <- function(dt_trades) {
  log_msg("Decomposing temporary vs permanent price impact...")

  # Realized spread with 5-row lookahead is computed in spread_analysis / adverse_selection
  if (!"realized_spread" %in% names(dt_trades)) {
    dt_trades[, mid_future_5 := shift(mid_price, n = 5L, type = "lead"), by = symbol]
    dt_trades[, realized_spread := compute_realized_spread(trade_price, mid_future_5, trade_sign)]
    dt_trades[, adverse_selection := effective_spread - realized_spread]
  }

  # Permanent impact = Realized spread
  # Temporary impact = Effective spread - Realized spread = Adverse selection
  # Note: The term adverse_selection in metrics is effective - realized,
  # which corresponds to the permanent component in some literature, or vice-versa.
  # Let's define:
  # Permanent impact = effective_spread - realized_spread = adverse_selection
  # Temporary impact = realized_spread
  # This corresponds to:
  # - Permanent impact is the information price revision.
  # - Temporary impact is the inventory/order handling cost.
  
  dt_trades[, perm_impact_5 := effective_spread - realized_spread]
  dt_trades[, temp_impact_5 := realized_spread]
  dt_trades[, impact_ratio := perm_impact_5 / pmax(effective_spread, 1e-5)]
  dt_trades[is.nan(impact_ratio) | !is.finite(impact_ratio), impact_ratio := 0.5]

  # Compute metrics over multiple horizons
  for (h in c(1L, 10L)) {
    mid_fut <- shift(dt_trades$mid_price, n = h, type = "lead")
    realized_spr_h <- compute_realized_spread(dt_trades$trade_price, mid_fut, dt_trades$trade_sign)
    
    dt_trades[, paste0("perm_impact_", h) := effective_spread - realized_spr_h]
    dt_trades[, paste0("temp_impact_", h) := realized_spr_h]
  }

  log_msg("Impact decomposition complete")
  dt_trades
}

# ═══════════════════════════════════════════════════════════════════════════════
# SIZE & SIGN ASYMMETRY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

analyze_impact_asymmetry <- function(dt_trades) {
  log_msg("Analyzing trade size quintile impacts and buy/sell asymmetry...")

  # Size quintiles per symbol
  dt_trades[, size_quintile := cut(trade_qty, 
                                   breaks = quantile(trade_qty, probs = seq(0, 1, 0.2), na.rm = TRUE),
                                   labels = 1:5, include.lowest = TRUE), by = symbol]
  
  size_impact <- dt_trades[, .(
    avg_perm_impact = mean(perm_impact_5, na.rm = TRUE),
    avg_temp_impact = mean(temp_impact_5, na.rm = TRUE),
    avg_total_impact = mean(effective_spread, na.rm = TRUE),
    n_trades = .N
  ), by = .(symbol, size_quintile)]

  # Buy vs Sell impact asymmetry
  direction_impact <- dt_trades[, .(
    avg_perm_impact = mean(perm_impact_5, na.rm = TRUE),
    avg_temp_impact = mean(temp_impact_5, na.rm = TRUE),
    avg_total_impact = mean(effective_spread, na.rm = TRUE)
  ), by = .(symbol, lr_sign)]

  log_msg("Buy (+1) vs Sell (-1) price impact summary:")
  print(direction_impact)

  list(size = size_impact, direction = direction_impact)
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

generate_impact_plots <- function(dt_trades, asymmetry) {
  log_msg("Generating price impact plots...")

  # Plot 1: Impact decay across trade sizes
  # Aggregate decay by step (tau=1, 5, 10)
  decay_data <- data.table(
    Horizon = c("1 Trade", "5 Trades", "10 Trades"),
    Permanent = c(mean(dt_trades$perm_impact_1, na.rm = TRUE),
                  mean(dt_trades$perm_impact_5, na.rm = TRUE),
                  mean(dt_trades$perm_impact_10, na.rm = TRUE)),
    Temporary = c(mean(dt_trades$temp_impact_1, na.rm = TRUE),
                  mean(dt_trades$temp_impact_5, na.rm = TRUE),
                  mean(dt_trades$temp_impact_10, na.rm = TRUE))
  )
  decay_melt <- melt(decay_data, id.vars = "Horizon", variable.name = "Component", value.name = "Impact")

  p1 <- ggplot(decay_melt, aes(x = Horizon, y = Impact * 100, fill = Component)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_manual(values = c("Permanent" = "#00D4AA", "Temporary" = "#FF5252")) +
    labs(
      title = "Price Impact Component Decay Analysis",
      subtitle = "Permanent information vs. temporary liquidity components across horizons",
      x = "Horizon",
      y = "Impact (INR × 100)"
    ) +
    THEME_DARK
  
  save_plot(p1, "price_impact_decay.png", width = 10, height = 7)

  # Plot 2: Size-conditioned impact
  p2 <- ggplot(asymmetry$size, aes(x = factor(size_quintile), y = avg_total_impact, fill = symbol)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_viridis_d(option = "plasma") +
    labs(
      title = "Size-Conditioned Price Impact by Symbol",
      subtitle = "Effective spread (INR) across trade size quintiles (1 = small, 5 = large)",
      x = "Trade Size Quintile",
      y = "Total Price Impact"
    ) +
    THEME_DARK

  save_plot(p2, "size_conditioned_impact.png", width = 12, height = 8)
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main_price_impact <- function() {
  log_msg("=" |> rep(60) |> paste(collapse = ""))
  log_msg("PRICE IMPACT ANALYSIS — Starting")
  log_msg("=" |> rep(60) |> paste(collapse = ""))

  # Load data
  dt_trades <- load_impact_data()

  # Compute Amihud Illiquidity
  amihud_list <- compute_amihud_illiq(dt_trades)
  dt_trades <- amihud_list$trades

  # Decompose Temporary vs Permanent impact
  dt_trades <- decompose_price_impact(dt_trades)

  # Analyze asymmetry
  asymmetry <- analyze_impact_asymmetry(dt_trades)

  # Save results
  price_impact_out <- dt_trades[, .(
    timestamp, symbol, amihud_illiq, amihud_illiq_rolling,
    temp_impact_1, temp_impact_5, temp_impact_10,
    perm_impact_1, perm_impact_5, perm_impact_10, impact_ratio
  )]
  fwrite(price_impact_out, file.path(DATA_PROCESSED, "price_impact.csv"))
  
  # Save enriched trades back
  saveRDS(dt_trades, file.path(DATA_PROCESSED, "trades_enriched.rds"))
  
  log_msg("Price impact metrics saved to data/processed/price_impact.csv")

  # Generate plots
  generate_impact_plots(dt_trades, asymmetry)

  log_msg("PRICE IMPACT ANALYSIS — Complete")
  invisible(dt_trades)
}

# Run if executed directly
if (sys.nframe() == 0L || !interactive()) {
  main_price_impact()
}
