###############################################################################
# 04_adverse_selection.R — Adverse Selection Metrics (Kyle's Lambda & PIN)
# Market Microstructure & Order Flow Analytics Engine
#
# Analysis:
#   1. Compute Kyle's Lambda per symbol (regression of price change on signed flow)
#   2. Compute PIN (Probability of Informed Trading) using numerical MLE
#   3. Compute VPIN (Volume-Synchronized Probability of Informed Trading)
#   4. Save metrics and generate plots
#
# References:
#   - Kyle (1985): "Continuous Auctions and Informed Trader"
#   - Easley, Kiefer, O'Hara & Paperman (1996): "Liquidity, Information, and Less Frequently Traded Stocks"
#   - Easley, Lopez de Prado & O'Hara (2012): "Flow Toxicity and Liquidity in a High-Frequency World"
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

load_adverse_data <- function() {
  trade_path <- file.path(DATA_PROCESSED, "trades_enriched.rds")

  if (!file.exists(trade_path)) {
    log_msg("Processed trades enriched not found — running loader", level = "WARN")
    source("R/03_trade_classification.R")
  }
  
  readRDS(trade_path)
}

# ═══════════════════════════════════════════════════════════════════════════════
# ESTIMATE KYLE'S LAMBDA
# ═══════════════════════════════════════════════════════════════════════════════

estimate_kyle_lambda_all <- function(dt_trades) {
  log_msg("Estimating Kyle's Lambda per symbol...")
  
  # Aggregate price change and signed volume per 5-minute interval
  dt_trades[, interval_5min := as.POSIXct(cut(timestamp, breaks = "5 min"), tz = "Asia/Kolkata")]
  
  kyle_data <- dt_trades[, .(
    delta_p = last(trade_price) - first(trade_price),
    signed_flow = sum(signed_volume),
    total_volume = sum(trade_qty)
  ), by = .(symbol, interval_5min)]
  
  kyle_results <- kyle_data[, {
    kl <- compute_kyles_lambda(delta_p, signed_flow)
    list(
      kyle_lambda = kl$lambda,
      kyle_lambda_tstat = kl$tstat,
      kyle_r_squared = kl$rsq,
      n_obs = kl$nobs
    )
  }, by = symbol]
  
  log_msg("Estimated Kyle's Lambda coefficients:")
  print(kyle_results)
  
  list(results = kyle_results, data = kyle_data)
}

# ═══════════════════════════════════════════════════════════════════════════════
# PIN MODEL MLE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

# Simplified PIN Log-Likelihood objective function
pin_neg_log_lik <- function(params, B, S) {
  alpha  <- params[1]
  delta  <- params[2]
  mu     <- params[3]
  epsilon_b <- params[4]
  epsilon_s <- params[5]
  
  if (alpha < 0 || alpha > 1 || delta < 0 || delta > 1 || mu < 0 || epsilon_b < 0 || epsilon_s < 0) {
    return(1e10)
  }
  
  ll <- 0
  n_days <- length(B)
  
  for (d in 1:n_days) {
    b <- B[d]
    s <- S[d]
    
    # EHO (Easley-Kiefer-O'Hara) factorization to prevent underflow/overflow
    term1 <- (1 - alpha) * exp(-epsilon_b) * (epsilon_b^b / factorial(b)) * exp(-epsilon_s) * (epsilon_s^s / factorial(s))
    term2 <- alpha * delta * exp(-epsilon_b) * (epsilon_b^b / factorial(b)) * exp(-(epsilon_s + mu)) * ((epsilon_s + mu)^s / factorial(s))
    term3 <- alpha * (1 - delta) * exp(-(epsilon_b + mu)) * ((epsilon_b + mu)^b / factorial(b)) * exp(-epsilon_s) * (epsilon_s^s / factorial(s))
    
    val <- term1 + term2 + term3
    if (is.na(val) || val <= 0) {
      # Use high-precision approximation or large penalty
      ll <- ll - 1000
    } else {
      ll <- ll + log(val)
    }
  }
  
  -ll
}

estimate_pin_all <- function(dt_trades) {
  log_msg("Estimating PIN parameters...")

  # Aggregate trades to daily buys and sells
  dt_trades[, date := as.Date(timestamp, tz = "Asia/Kolkata")]
  daily_trades <- dt_trades[, .(
    buys = sum(lr_sign == 1),
    sells = sum(lr_sign == -1)
  ), by = .(symbol, date)]
  
  symbols <- daily_trades[, unique(symbol)]
  pin_list <- list()
  
  for (sym in symbols) {
    sym_data <- daily_trades[symbol == sym]
    B <- sym_data$buys
    S <- sym_data$sells
    
    # Starting values (informed guesses)
    mean_b <- mean(B)
    mean_s <- mean(S)
    init_params <- c(alpha = 0.3, delta = 0.5, mu = abs(mean_b - mean_s), epsilon_b = min(mean_b, mean_s), epsilon_s = min(mean_b, mean_s))
    
    fit <- tryCatch({
      optim(par = init_params, fn = pin_neg_log_lik, B = B, S = S, method = "Nelder-Mead")
    }, error = function(e) {
      list(par = c(0.3, 0.5, 50, 100, 100))
    })
    
    params <- fit$par
    a <- clip_val(params[1], 0, 1)
    d <- clip_val(params[2], 0, 1)
    m <- max(params[3], 0)
    eb <- max(params[4], 1)
    es <- max(params[5], 1)
    
    pin_val <- (a * m) / (a * m + eb + es)
    
    pin_list[[sym]] <- data.table(
      symbol = sym,
      pin = pin_val,
      pin_alpha = a,
      pin_delta = d,
      pin_mu = m,
      pin_epsilon_b = eb,
      pin_epsilon_s = es
    )
  }
  
  rbindlist(pin_list)
}

clip_val <- function(x, min_v, max_v) {
  pmax(pmin(x, max_v), min_v)
}

# ═══════════════════════════════════════════════════════════════════════════════
# ESTIMATE VPIN (Volume-Synchronized PIN)
# ═══════════════════════════════════════════════════════════════════════════════

estimate_vpin_all <- function(dt_trades, bucket_size = 5000) {
  log_msg(sprintf("Computing VPIN with bucket size = %d...", bucket_size))
  
  vpin_list <- list()
  symbols <- dt_trades[, unique(symbol)]
  
  for (sym in symbols) {
    sym_trades <- dt_trades[symbol == sym]
    setorder(sym_trades, timestamp)
    
    # Construct volume buckets
    sym_trades[, cum_vol := cumsum(trade_qty)]
    sym_trades[, bucket_idx := cum_vol %/% bucket_size]
    
    # Aggregate buy and sell volume in each bucket
    buckets <- sym_trades[, .(
      timestamp = last(timestamp),
      buy_vol = sum(trade_qty[lr_sign == 1]),
      sell_vol = sum(trade_qty[lr_sign == -1]),
      total_vol = sum(trade_qty)
    ), by = bucket_idx]
    
    # Compute VPIN = sum(|buy_vol - sell_vol|) / (N * bucket_size)
    # Using a rolling estimation over 50 buckets
    N <- 50
    buckets[, vol_imbalance := abs(buy_vol - sell_vol)]
    buckets[, vpin := frollsum(vol_imbalance, n = N) / (N * bucket_size)]
    buckets[is.na(vpin), vpin := 0]
    
    # Compute CDF rank
    buckets[, vpin_cdf := rank(vpin) / .N]
    
    buckets[, symbol := sym]
    vpin_list[[sym]] <- buckets[, .(timestamp, symbol, vpin, vpin_cdf)]
  }
  
  rbindlist(vpin_list)
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

generate_adverse_plots <- function(pin_dt, vpin_dt) {
  log_msg("Generating adverse selection plots...")

  # Plot 1: PIN Estimates Comparison
  p1 <- ggplot(pin_dt, aes(x = reorder(symbol, pin), y = pin, fill = symbol)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    scale_fill_viridis_d(option = "plasma") +
    coord_flip() +
    labs(
      title = "Probability of Informed Trading (PIN) by Symbol",
      subtitle = "Adverse selection metric derived via MLE from daily signed trade volume",
      x = "Symbol",
      y = "PIN Value"
    ) +
    THEME_DARK +
    theme(legend.position = "none")

  save_plot(p1, "pin_estimates.png", width = 12, height = 8)

  # Plot 2: VPIN timeseries
  sym_sample <- vpin_dt[, unique(symbol)][1]
  dt_sample <- vpin_dt[symbol == sym_sample]
  
  p2 <- ggplot(dt_sample, aes(x = timestamp, y = vpin)) +
    geom_line(colour = "#00D4AA", linewidth = 1.0) +
    geom_point(aes(colour = vpin_cdf > 0.9), size = 1.5) +
    scale_colour_manual(values = c("FALSE" = "#00D4AA", "TRUE" = "#FF5252"), name = "Toxicity Alert") +
    labs(
      title = paste("VPIN (Volume-Synchronized PIN) —", sym_sample),
      subtitle = "Real-time toxic flow monitoring | Red flags indicate high toxicity percentile (>90%)",
      x = "Time",
      y = "VPIN"
    ) +
    THEME_DARK

  save_plot(p2, "vpin_toxicity.png", width = 14, height = 7)
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main_adverse_selection <- function() {
  log_msg("=" |> rep(60) |> paste(collapse = ""))
  log_msg("ADVERSE SELECTION ANALYSIS — Starting")
  log_msg("=" |> rep(60) |> paste(collapse = ""))

  # Load data
  dt_trades <- load_adverse_data()

  # Kyle's Lambda
  kyle_list <- estimate_kyle_lambda_all(dt_trades)
  
  # PIN
  pin_dt <- estimate_pin_all(dt_trades)
  
  # VPIN
  vpin_dt <- estimate_vpin_all(dt_trades)

  # Combine R results
  # We merge Kyle's Lambda and PIN into a single dataset
  merged_results <- merge(kyle_list$results, pin_dt, by = "symbol")
  
  # VPIN is a timeseries, we match it back to trades or save separately
  # Save cross-sectional results
  fwrite(merged_results, file.path(DATA_PROCESSED, "adverse_selection.csv"))
  
  # Save VPIN timeseries results
  fwrite(vpin_dt, file.path(DATA_PROCESSED, "vpin_metrics.csv"))

  # Save enriched trades with VPIN mapped to trades
  # Rolling join VPIN to trades
  setkey(dt_trades, symbol, timestamp)
  setkey(vpin_dt, symbol, timestamp)
  dt_trades_enriched <- vpin_dt[dt_trades, roll = TRUE]
  saveRDS(dt_trades_enriched, file.path(DATA_PROCESSED, "trades_enriched.rds"))
  
  log_msg("Adverse selection metrics saved successfully")

  # Generate plots
  generate_adverse_plots(pin_dt, vpin_dt)

  log_msg("ADVERSE SELECTION ANALYSIS — Complete")
  invisible(merged_results)
}

# Run if executed directly
if (sys.nframe() == 0L || !interactive()) {
  main_adverse_selection()
}
