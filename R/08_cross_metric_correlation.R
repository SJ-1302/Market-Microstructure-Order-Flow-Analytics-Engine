###############################################################################
# 08_cross_metric_correlation.R — Cross-Metric Correlation
# Market Microstructure & Order Flow Analytics Engine
#
# Analysis:
#   1. Load processed metrics (spread, OBI, trades, adverse selection, price impact)
#   2. Merge all metrics into a unified dataframe by symbol and time interval
#   3. Compute pairwise Pearson correlation matrix across all key metrics
#   4. Save correlation matrix and generate correlation heatmap visualization
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
# LOAD AND ALIGN DATA
# ═══════════════════════════════════════════════════════════════════════════════

load_and_merge_metrics <- function() {
  log_msg("Loading and merging all microstructure metrics...")
  
  # Load CSVs
  spread_metrics <- fread(file.path(DATA_PROCESSED, "spread_metrics.csv"))
  obi_metrics <- fread(file.path(DATA_PROCESSED, "obi_metrics.csv"))
  
  # Classified trades has TQR and BVC
  classified_trades <- fread(file.path(DATA_PROCESSED, "classified_trades.csv"))
  
  # Adverse selection has Kyle's Lambda and PIN (cross-sectional, we join or use rolling timeseries if available)
  adverse_sel <- fread(file.path(DATA_PROCESSED, "adverse_selection.csv"))
  vpin_metrics <- fread(file.path(DATA_PROCESSED, "vpin_metrics.csv"))
  
  # Price impact has Amihud
  price_impact <- fread(file.path(DATA_PROCESSED, "price_impact.csv"))

  # Align to 5-minute bins
  spread_metrics[, interval_5min := as.POSIXct(cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata")]
  obi_metrics[, interval_5min := as.POSIXct(cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata")]
  classified_trades[, interval_5min := as.POSIXct(cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata")]
  vpin_metrics[, interval_5min := as.POSIXct(cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata")]
  price_impact[, interval_5min := as.POSIXct(cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata")]

  # Aggregate metrics to 5-minute mean/median per symbol
  spread_agg <- spread_metrics[, .(quoted_spread = mean(quoted_spread, na.rm = TRUE),
                                   pct_spread = mean(pct_spread, na.rm = TRUE)), by = .(symbol, interval_5min)]
  obi_agg <- obi_metrics[, .(obi = mean(obi_exp_weight, na.rm = TRUE)), by = .(symbol, interval_5min)]
  trade_agg <- classified_trades[, .(tqr = mean(tqr, na.rm = TRUE),
                                     bvc_buy_pct = mean(bvc_buy_pct, na.rm = TRUE)), by = .(symbol, interval_5min)]
  vpin_agg <- vpin_metrics[, .(vpin = mean(vpin, na.rm = TRUE)), by = .(symbol, interval_5min)]
  impact_agg <- price_impact[, .(amihud = mean(amihud_illiq, na.rm = TRUE),
                                 perm_impact = mean(perm_impact_5, na.rm = TRUE)), by = .(symbol, interval_5min)]

  # Sequentially merge
  m1 <- merge(spread_agg, obi_agg, by = c("symbol", "interval_5min"))
  m2 <- merge(m1, trade_agg, by = c("symbol", "interval_5min"))
  m3 <- merge(m2, vpin_agg, by = c("symbol", "interval_5min"))
  m_final <- merge(m3, impact_agg, by = c("symbol", "interval_5min"))

  # Merge with cross-sectional adverse selection (Kyle's Lambda & PIN)
  m_final <- merge(m_final, adverse_sel[, .(symbol, kyle_lambda, pin)], by = "symbol", all.x = TRUE)

  m_final
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

compute_correlation_matrix <- function(merged_dt) {
  log_msg("Computing correlation matrix...")

  # Select columns for correlation
  cols <- c("quoted_spread", "pct_spread", "obi", "tqr", "bvc_buy_pct", "vpin", "amihud", "perm_impact", "kyle_lambda", "pin")
  
  # Subset and clean
  corr_data <- merged_dt[, ..cols]
  corr_data <- na.omit(corr_data)
  
  # Compute matrix
  cor_matrix <- cor(as.matrix(corr_data), method = "pearson")
  
  log_msg("Correlation matrix:")
  print(cor_matrix)
  
  cor_matrix
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

generate_correlation_plots <- function(cor_matrix) {
  log_msg("Generating correlation matrix heatmap...")

  # Convert matrix to long format data.table for ggplot
  cor_dt <- as.data.table(cor_matrix, keep.rownames = "Metric1")
  cor_long <- melt(cor_dt, id.vars = "Metric1", variable.name = "Metric2", value.name = "Correlation")

  p1 <- ggplot(cor_long, aes(x = Metric1, y = Metric2, fill = Correlation)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", Correlation)), color = "#e0e0e0", size = 3) +
    scale_fill_gradient2(low = "#FF5252", mid = "#1a1a2e", high = "#00D4AA", midpoint = 0, limits = c(-1, 1)) +
    labs(
      title = "Microstructure Metrics Correlation Heatmap",
      subtitle = "Pairwise Pearson correlation matrix showing cross-metric relationships",
      x = "Metric",
      y = "Metric"
    ) +
    THEME_DARK +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  save_plot(p1, "microstructure_metric_correlation.png", width = 12, height = 10)
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main_cross_metric_correlation <- function() {
  log_msg("=" |> rep(60) |> paste(collapse = ""))
  log_msg("CROSS-METRIC CORRELATION — Starting")
  log_msg("=" |> rep(60) |> paste(collapse = ""))

  # Load and merge metrics
  merged_dt <- load_and_merge_metrics()

  # Compute matrix
  cor_matrix <- compute_correlation_matrix(merged_dt)

  # Save matrix
  write.csv(cor_matrix, file.path(DATA_PROCESSED, "correlation_matrix.csv"), row.names = TRUE)
  
  # Save merged time-series dataset for signal generation
  fwrite(merged_dt, file.path(DATA_PROCESSED, "microstructure_merged_5min.csv"))
  
  log_msg("Correlation matrix and merged metrics saved to data/processed/")

  # Generate plots
  generate_correlation_plots(cor_matrix)

  log_msg("CROSS-METRIC CORRELATION — Complete")
  invisible(cor_matrix)
}

# Run if executed directly
if (sys.nframe() == 0L || !interactive()) {
  main_cross_metric_correlation()
}
