###############################################################################
# 01_spread_analysis.R — Bid-Ask Spread Dynamics at Microsecond Resolution
# Market Microstructure & Order Flow Analytics Engine
#
# Analysis:
#   1. Compute all spread metrics (quoted, percentage, effective, realized)
#   2. Rolling spread analysis over multiple tick windows
#   3. Time-weighted average spread (TWAS) per 5-minute intervals
#   4. Intraday spread profile (U-shape detection)
#   5. Cross-sectional comparison across symbols
#   6. Statistical tests for spread differences
#
# References:
#   - Hasbrouck (2007): "Empirical Market Microstructure"
#   - Foucault, Pagano & Röell (2013): "Market Liquidity"
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

load_spread_data <- function() {
  book_path <- file.path(DATA_PROCESSED, "orderbook_processed.rds")
  trade_path <- file.path(DATA_PROCESSED, "trades_enriched.rds")

  if (file.exists(book_path)) {
    dt_book <- readRDS(book_path)
    log_msg(sprintf("Loaded order book data: %s rows", format(nrow(dt_book), big.mark = ",")))
  } else {
    log_msg("Processed order book not found — running data loader", level = "WARN")
    source("R/01_data_loader.R")
    dt_book <- readRDS(book_path)
  }

  dt_trades <- NULL
  if (file.exists(trade_path)) {
    dt_trades <- readRDS(trade_path)
    log_msg(sprintf("Loaded enriched trades: %s rows", format(nrow(dt_trades), big.mark = ",")))
  }

  list(book = dt_book, trades = dt_trades)
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL SPREAD METRICS
# ═══════════════════════════════════════════════════════════════════════════════

compute_all_spreads <- function(dt_book, dt_trades = NULL) {
  log_msg("Computing spread metrics...")

  # ── Quoted and percentage spreads (already computed in loader, ensure present)
  if (!"quoted_spread" %in% names(dt_book)) {
    dt_book[, quoted_spread := compute_quoted_spread(bid1, ask1)]
  }
  if (!"pct_spread" %in% names(dt_book)) {
    dt_book[, pct_spread := compute_pct_spread(bid1, ask1)]
  }
  if (!"mid_price" %in% names(dt_book)) {
    dt_book[, mid_price := (bid1 + ask1) / 2]
  }

  # ── Effective spread (from enriched trades) ──
  if (!is.null(dt_trades) && "effective_spread" %in% names(dt_trades)) {
    log_msg("Effective spread already computed in trade data")
  }

  # ── Realized spread: requires mid price τ seconds into the future ──
  # We use τ = 5 seconds (common in literature)
  if (!is.null(dt_trades) && "trade_sign" %in% names(dt_trades)) {
    # Forward-looking mid price (5 rows ahead as proxy for ~5 seconds)
    dt_trades[, mid_future_5 := shift(mid_price, n = 5L, type = "lead"), by = symbol]
    dt_trades[, realized_spread := compute_realized_spread(
      trade_price, mid_future_5, trade_sign
    )]
    # Adverse selection component = effective - realized
    dt_trades[, adverse_selection := effective_spread - realized_spread]
    log_msg("Computed realized spread and adverse selection component")
  }

  # ── Spread in basis points for interpretability ──
  dt_book[, spread_bps := pct_spread * 10000]

  log_msg(sprintf("Spread metrics computed: %d book rows", nrow(dt_book)))
  list(book = dt_book, trades = dt_trades)
}

# ═══════════════════════════════════════════════════════════════════════════════
# ROLLING SPREAD ANALYSIS
# Compute rolling averages over 100, 500, and 1000 tick windows.
# This captures the dynamic evolution of liquidity conditions.
# ═══════════════════════════════════════════════════════════════════════════════

compute_rolling_spreads <- function(dt_book) {
  log_msg("Computing rolling spread averages...")

  windows <- c(100L, 500L, 1000L)

  for (w in windows) {
    col_name <- paste0("spread_roll_", w)
    dt_book[, (col_name) := frollmean(pct_spread, n = w, align = "right",
                                       na.rm = TRUE), by = symbol]
  }

  # Rolling standard deviation (spread volatility) — 500-tick window
  dt_book[, spread_vol_500 := frollapply(pct_spread, n = 500L,
                                          FUN = sd, align = "right"), by = symbol]

  log_msg(sprintf("Rolling spreads computed: windows = %s",
                  paste(windows, collapse = ", ")))
  dt_book
}

# ═══════════════════════════════════════════════════════════════════════════════
# TIME-WEIGHTED AVERAGE SPREAD (TWAS)
# Computes TWAS per 5-minute interval per symbol.
# Weights each spread by the duration it was active, avoiding
# overweighting periods of quote updates.
# ═══════════════════════════════════════════════════════════════════════════════

compute_twas_intervals <- function(dt_book) {
  log_msg("Computing time-weighted average spread per 5-minute interval...")

  # Compute duration each quote was active (seconds until next update)
  dt_book[, duration := as.numeric(
    difftime(shift(timestamp, type = "lead"), timestamp, units = "secs")
  ), by = symbol]

  # Cap extreme durations (e.g., gaps between trading sessions)
  dt_book[is.na(duration) | duration > 600, duration := 1]
  dt_book[duration <= 0, duration := 0.001]

  # Create 5-minute interval bins
  dt_book[, interval_5min := as.POSIXct(
    cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata"
  )]

  # Compute TWAS per interval and symbol
  twas_dt <- dt_book[, .(
    twas         = compute_twas(pct_spread, duration),
    twas_bps     = compute_twas(pct_spread, duration) * 10000,
    n_updates    = .N,
    total_time   = sum(duration, na.rm = TRUE),
    mean_spread  = mean(pct_spread, na.rm = TRUE),
    median_spread = median(pct_spread, na.rm = TRUE)
  ), by = .(symbol, interval_5min)]

  setkey(twas_dt, symbol, interval_5min)
  log_msg(sprintf("TWAS computed: %d intervals across %d symbols",
                  nrow(twas_dt), uniqueN(twas_dt$symbol)))
  twas_dt
}

# ═══════════════════════════════════════════════════════════════════════════════
# INTRADAY SPREAD PROFILE
# Group by time-of-day to detect the U-shape:
# ═══════════════════════════════════════════════════════════════════════════════

compute_intraday_profile <- function(dt_book) {
  log_msg("Computing intraday spread profile...")

  # Create 15-minute time-of-day bins
  dt_book[, time_of_day := format(timestamp, "%H:%M")]
  dt_book[, time_bin := {
    h <- as.integer(format(timestamp, "%H"))
    m <- as.integer(format(timestamp, "%M"))
    bin_m <- (m %/% 15) * 15
    sprintf("%02d:%02d", h, bin_m)
  }]

  # Aggregate spread statistics per time bin
  intraday <- dt_book[, .(
    median_spread_bps = median(spread_bps, na.rm = TRUE),
    mean_spread_bps   = mean(spread_bps, na.rm = TRUE),
    q25_spread_bps    = quantile(spread_bps, 0.25, na.rm = TRUE),
    q75_spread_bps    = quantile(spread_bps, 0.75, na.rm = TRUE),
    sd_spread_bps     = sd(spread_bps, na.rm = TRUE),
    n_obs             = .N
  ), by = .(symbol, time_bin)]

  # Overall (across all symbols)
  intraday_all <- dt_book[, .(
    median_spread_bps = median(spread_bps, na.rm = TRUE),
    mean_spread_bps   = mean(spread_bps, na.rm = TRUE),
    q25_spread_bps    = quantile(spread_bps, 0.25, na.rm = TRUE),
    q75_spread_bps    = quantile(spread_bps, 0.75, na.rm = TRUE),
    n_obs             = .N
  ), by = time_bin]

  setkey(intraday, symbol, time_bin)
  log_msg("Intraday profile computed — U-shape expected in results")

  list(by_symbol = intraday, overall = intraday_all)
}

# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

cross_sectional_analysis <- function(dt_book) {
  log_msg("Performing cross-sectional spread analysis...")

  # Summary statistics by symbol
  cross_section <- dt_book[, .(
    n_obs           = .N,
    mean_spread_bps = round(mean(spread_bps, na.rm = TRUE), 4),
    median_spread_bps = round(median(spread_bps, na.rm = TRUE), 4),
    sd_spread_bps   = round(sd(spread_bps, na.rm = TRUE), 4),
    min_spread_bps  = round(min(spread_bps, na.rm = TRUE), 4),
    max_spread_bps  = round(max(spread_bps, na.rm = TRUE), 4),
    iqr_spread_bps  = round(IQR(spread_bps, na.rm = TRUE), 4)
  ), by = symbol]

  setorder(cross_section, median_spread_bps)

  # Kruskal-Wallis test
  kw_test <- kruskal.test(spread_bps ~ symbol, data = dt_book)
  log_msg(sprintf("Kruskal-Wallis test: χ² = %.2f, df = %d, p-value = %s",
                  kw_test$statistic, kw_test$parameter,
                  format.pval(kw_test$p.value, digits = 4)))

  list(summary = cross_section, kruskal_wallis = kw_test)
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

generate_spread_plots <- function(dt_book, intraday, twas_dt) {
  log_msg("Generating spread analysis plots...")

  # ── Plot 1: Intraday Spread Profile (U-Shape) ──
  p1 <- ggplot(intraday$overall, aes(x = time_bin, y = median_spread_bps)) +
    geom_ribbon(aes(ymin = q25_spread_bps, ymax = q75_spread_bps, group = 1),
                fill = "#7B2FBE", alpha = 0.3) +
    geom_line(aes(group = 1), colour = "#00D4AA", linewidth = 1.2) +
    geom_point(colour = "#00D4AA", size = 2) +
    labs(
      title = "Intraday Spread Profile — U-Shape Pattern",
      subtitle = "Median bid-ask spread (bps) with IQR band across all symbols",
      x = "Time of Day (IST)",
      y = "Spread (basis points)",
      caption = "Shaded region: interquartile range | Source: NSE Level-2 Data"
    ) +
    THEME_DARK +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

  save_plot(p1, "spread_intraday_ushape.png", width = 14, height = 7)

  # ── Plot 2: Cross-Sectional Spread Comparison ──
  spread_dist <- dt_book[, .(spread_bps = spread_bps, symbol = symbol)]

  p2 <- ggplot(spread_dist, aes(x = reorder(symbol, spread_bps, FUN = median),
                                 y = spread_bps)) +
    geom_violin(aes(fill = symbol), alpha = 0.6, colour = NA) +
    geom_boxplot(width = 0.15, fill = "#1a1a2e", colour = "#e0e0e0",
                 outlier.shape = NA) +
    scale_fill_viridis_d(option = "plasma", begin = 0.2, end = 0.9) +
    coord_cartesian(ylim = c(0, quantile(spread_dist$spread_bps, 0.99,
                                          na.rm = TRUE))) +
    labs(
      title = "Cross-Sectional Spread Distribution by Symbol",
      subtitle = "Violin + box plots | Ordered by median spread",
      x = "Symbol",
      y = "Spread (basis points)"
    ) +
    THEME_DARK +
    theme(legend.position = "none")

  save_plot(p2, "spread_cross_section.png", width = 12, height = 8)

  # ── Plot 3: Rolling Spread Time Series ──
  dt_sample <- dt_book[symbol == dt_book[, unique(symbol)][1]]
  if (nrow(dt_sample) > 5000) {
    dt_sample <- dt_sample[seq(1, .N, length.out = 5000)]
  }

  p3 <- ggplot(dt_sample, aes(x = timestamp)) +
    geom_line(aes(y = spread_bps, colour = "Raw"), alpha = 0.15, linewidth = 0.3) +
    geom_line(aes(y = spread_roll_100 * 10000, colour = "MA-100"),
              linewidth = 0.7, na.rm = TRUE) +
    geom_line(aes(y = spread_roll_500 * 10000, colour = "MA-500"),
              linewidth = 1.0, na.rm = TRUE) +
    geom_line(aes(y = spread_roll_1000 * 10000, colour = "MA-1000"),
              linewidth = 1.2, na.rm = TRUE) +
    scale_colour_manual(
      values = c("Raw" = "#555555", "MA-100" = "#FFA726",
                 "MA-500" = "#00D4AA", "MA-1000" = "#FF5252"),
      name = "Series"
    ) +
    labs(
      title = paste("Rolling Spread Analysis —", dt_sample$symbol[1]),
      subtitle = "Raw spread with 100/500/1000-tick moving averages",
      x = "Time",
      y = "Spread (basis points)"
    ) +
    THEME_DARK

  save_plot(p3, "spread_rolling_timeseries.png", width = 14, height = 7)

  log_msg("All spread plots generated successfully")
  invisible(list(p1 = p1, p2 = p2, p3 = p3))
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main_spread_analysis <- function() {
  log_msg("=" |> rep(60) |> paste(collapse = ""))
  log_msg("SPREAD ANALYSIS — Starting")
  log_msg("=" |> rep(60) |> paste(collapse = ""))

  # Load data
  data <- load_spread_data()

  # Compute all spread metrics
  result <- compute_all_spreads(data$book, data$trades)

  # Rolling spreads
  result$book <- compute_rolling_spreads(result$book)

  # TWAS per 5-minute interval
  twas_dt <- compute_twas_intervals(result$book)

  # Intraday profile
  intraday <- compute_intraday_profile(result$book)

  # Cross-sectional analysis
  cross <- cross_sectional_analysis(result$book)

  # Save results
  spread_metrics <- result$book[, .(
    timestamp, symbol, bid1, ask1, mid_price,
    quoted_spread, pct_spread, spread_bps
  )]

  fwrite(spread_metrics, file.path(DATA_PROCESSED, "spread_metrics.csv"))
  fwrite(twas_dt, file.path(DATA_PROCESSED, "twas_5min.csv"))
  
  # Save processed order book back (with the new columns)
  saveRDS(result$book, file.path(DATA_PROCESSED, "orderbook_processed.rds"))
  if (!is.null(result$trades)) {
    saveRDS(result$trades, file.path(DATA_PROCESSED, "trades_enriched.rds"))
  }
  
  log_msg("Spread metrics saved to data/processed/")

  # Generate plots
  generate_spread_plots(result$book, intraday, twas_dt)

  log_msg("SPREAD ANALYSIS — Complete")
  invisible(list(book = result$book, trades = result$trades,
                 twas = twas_dt, intraday = intraday, cross = cross))
}

# Run if executed directly
if (sys.nframe() == 0L || !interactive()) {
  main_spread_analysis()
}
