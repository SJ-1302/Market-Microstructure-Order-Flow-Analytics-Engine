###############################################################################
# 06_queue_decay.R — Queue Position Decay & Order Survival Analysis
# Market Microstructure & Order Flow Analytics Engine
#
# Analysis:
#   1. Read raw order events data
#   2. Match order entry (NEW) with exits (CANCEL or FILL) to compute lifetime
#   3. Perform Kaplan-Meier survival analysis to get survival probability
#   4. Fit parametric exponential decay model to estimate queue decay rate (lambda)
#   5. Compute queue survival statistics across liquidity tiers
#   6. Save queue decay metrics and generate survival curve plots
#
# References:
#   - Cont, Stoikov & Talreja (2010): "A Stochastic Model for Order Book Dynamics"
#   - Survival Analysis methodology (Kaplan & Meier, 1958)
###############################################################################

# ── Source utilities ─────────────────────────────────────────────────────────
source("R/utils/helpers.R")
source("R/utils/metrics.R")
setup_project()

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(survival)
  library(scales)
})

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

load_queue_data <- function() {
  events_path <- file.path(DATA_RAW, "order_events.csv")
  
  if (!file.exists(events_path)) {
    log_msg("Raw order events not found — creating synthetic order events", level = "WARN")
    # Run data generation if needed
    system("python python/data_generation/synthetic_generator.py")
  }
  
  dt_events <- fread(events_path)
  dt_events[, timestamp := as.POSIXct(timestamp, tz = "Asia/Kolkata")]
  dt_events
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE ORDER LIFETIMES AND SURVIVAL
# ═══════════════════════════════════════════════════════════════════════════════

analyze_queue_survival <- function(dt_events) {
  log_msg("Analyzing order lifetimes and matching entry/exit events...")

  # Separate NEW events from exits (FILL or CANCEL)
  new_orders <- dt_events[event_type == "LIMIT_ORDER", .(order_id, symbol, side, price, size, timestamp_entry = timestamp)]
  exits <- dt_events[event_type %in% c("CANCEL", "FILL"), .(
    timestamp_exit = max(timestamp),
    event_final = last(event_type)
  ), by = order_id]

  # Merge entry and exit
  orders_lifecycle <- merge(new_orders, exits, by = "order_id", all.x = TRUE)
  
  # For censored observations (no exit event found), use the end of day timestamp
  end_of_day <- max(dt_events$timestamp)
  orders_lifecycle[is.na(timestamp_exit), `:=`(
    timestamp_exit = end_of_day,
    event_final = "CENSORED"
  )]

  # Compute lifetime in milliseconds
  orders_lifecycle[, lifetime_ms := as.numeric(difftime(timestamp_exit, timestamp_entry, units = "secs")) * 1000]
  orders_lifecycle[lifetime_ms < 0, lifetime_ms := 0.1]

  # Status flag for survival analysis: 1 = event (fill or cancel), 0 = censored
  orders_lifecycle[, status := 1L]
  orders_lifecycle[event_final == "CENSORED", status := 0L]

  # Event type flag: 1 = filled, 2 = cancelled (competing risks context)
  orders_lifecycle[, event_type_num := 0L]
  orders_lifecycle[event_final == "FILL", event_type_num := 1L]
  orders_lifecycle[event_final == "CANCEL", event_type_num := 2L]

  orders_lifecycle
}

# ═══════════════════════════════════════════════════════════════════════════════
# ESTIMATE DECAY AND KAPLAN-MEIER STATS
# ═══════════════════════════════════════════════════════════════════════════════

estimate_decay_rates <- function(orders_lifecycle) {
  log_msg("Estimating Kaplan-Meier survival and exponential decay rates...")

  symbols <- orders_lifecycle[, unique(symbol)]
  results <- list()

  for (sym in symbols) {
    sym_orders <- orders_lifecycle[symbol == sym]
    
    # Fit Kaplan-Meier survival curves
    km_fit <- survfit(Surv(lifetime_ms, status) ~ 1, data = sym_orders)
    
    # Quantiles of survival
    km_summary <- quantile(km_fit, probs = c(0.25, 0.50, 0.75))
    
    t25 <- km_summary$quantile[1]
    t50 <- km_summary$quantile[2]
    t75 <- km_summary$quantile[3]

    # Fit parametric exponential survival model (S(t) = exp(-lambda * t))
    # Under exponential model, mean survival time is 1/lambda.
    # We estimate lambda as: n_events / sum(lifetimes)
    total_events <- sum(sym_orders$status)
    sum_lifetimes <- sum(sym_orders$lifetime_ms)
    
    lambda <- total_events / sum_lifetimes
    mean_lifetime <- 1 / lambda
    median_lifetime <- log(2) / lambda

    # Probability of fill vs cancel
    total_non_censored <- sym_orders[event_final %in% c("FILL", "CANCEL"), .N]
    fill_prob <- sym_orders[event_final == "FILL", .N] / max(total_non_censored, 1)
    cancel_prob <- 1 - fill_prob

    results[[sym]] <- data.table(
      symbol = sym,
      side = "ALL",
      price_level = 1L,
      decay_rate_lambda = lambda,
      mean_survival_time_ms = mean_lifetime,
      median_survival_time_ms = median_lifetime,
      fill_probability = fill_prob,
      cancel_probability = cancel_prob,
      num_observations = nrow(sym_orders),
      km_survival_25pct = t25,
      km_survival_50pct = t50,
      km_survival_75pct = t75
    )
  }

  rbindlist(results)
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

generate_queue_plots <- function(orders_lifecycle) {
  log_msg("Generating queue survival plots...")

  # Clean extreme values for visualization
  dt_plot <- orders_lifecycle[lifetime_ms < 60000] # under 1 minute
  
  # Plot 1: Survival probability curves by symbol
  p1 <- ggplot(dt_plot, aes(x = lifetime_ms / 1000, color = symbol)) +
    stat_ecdf(geom = "step", linewidth = 1.0) +
    scale_y_reverse(label = percent) +
    scale_color_viridis_d(option = "plasma") +
    labs(
      title = "Limit Order Survival Functions (Kaplan-Meier Style)",
      subtitle = "Empirical probability of order survival as a function of time elapsed (seconds)",
      x = "Time (seconds)",
      y = "Survival Probability"
    ) +
    THEME_DARK

  save_plot(p1, "queue_survival_curves.png", width = 12, height = 8)
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main_queue_decay <- function() {
  log_msg("=" |> rep(60) |> paste(collapse = ""))
  log_msg("QUEUE POSITION DECAY ANALYSIS — Starting")
  log_msg("=" |> rep(60) |> paste(collapse = ""))

  # Load raw events
  dt_events <- load_queue_data()

  # Analyze lifecycle
  orders_lifecycle <- analyze_queue_survival(dt_events)

  # Estimate stats
  decay_results <- estimate_decay_rates(orders_lifecycle)

  # Save results
  fwrite(decay_results, file.path(DATA_PROCESSED, "queue_decay.csv"))
  log_msg("Queue decay metrics saved to data/processed/queue_decay.csv")

  # Generate plots
  generate_queue_plots(orders_lifecycle)

  log_msg("QUEUE POSITION DECAY ANALYSIS — Complete")
  invisible(decay_results)
}

# Run if executed directly
if (sys.nframe() == 0L || !interactive()) {
  main_queue_decay()
}
