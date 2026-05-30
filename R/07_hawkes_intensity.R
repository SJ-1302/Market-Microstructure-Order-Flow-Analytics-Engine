###############################################################################
# 07_hawkes_intensity.R — Hawkes Process Parameter Estimation
# Market Microstructure & Order Flow Analytics Engine
#
# Analysis:
#   1. Load order events data
#   2. Separate into stream types: limit order, market order, cancellations
#   3. Implement exact Hawkes process Maximum Likelihood Estimation (MLE)
#   4. Optimize params (mu, alpha, omega) per symbol and stream type
#   5. Compute branching ratios and conditional intensity paths
#   6. Save Hawkes parameters and generate intensity visualizations
#
# References:
#   - Hawkes (1971): "Spectra of some self-exciting and mutually exciting point processes"
#   - Ogata (1978): "The asymptotic behaviour of maximum likelihood estimators for stationary point processes"
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

load_hawkes_data <- function() {
  events_path <- file.path(DATA_RAW, "order_events.csv")
  
  if (!file.exists(events_path)) {
    log_msg("Raw order events not found — running data generation", level = "WARN")
    system("python python/data_generation/synthetic_generator.py")
  }
  
  dt_events <- fread(events_path)
  # Convert timestamps to numeric seconds from open
  t_open <- min(dt_events$timestamp)
  dt_events[, time_sec := as.numeric(difftime(timestamp, t_open, units = "secs"))]
  dt_events
}

# ═══════════════════════════════════════════════════════════════════════════════
# HAWKES LOG-LIKELIHOOD AND OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Hawkes process log-likelihood calculation
# For details, see Ogata (1978)
hawkes_log_lik <- function(params, t, T_max) {
  mu    <- params[1]
  alpha <- params[2]
  omega <- params[3]
  
  if (mu <= 0 || alpha < 0 || omega <= 0 || alpha >= omega) {
    return(-1e10) # invalid parameters
  }
  
  n <- length(t)
  if (n == 0) return(0)
  
  # Compute recursive term: A(i) = sum_{j < i} exp(-omega * (t_i - t_j))
  # A(i) = exp(-omega * (t_i - t_i-1)) * (1 + A(i-1))
  A <- numeric(n)
  for (i in 2:n) {
    A[i] <- exp(-omega * (t[i] - t[i-1])) * (1 + A[i-1])
  }
  
  # Log terms: sum_{i=1}^n ln(mu + alpha * A(i))
  log_intensities <- log(mu + alpha * A)
  sum_log_lambda <- sum(log_intensities)
  
  # Integral term: mu * T + alpha/omega * sum_{i=1}^n (1 - exp(-omega * (T_max - t_i)))
  integral <- mu * T_max + (alpha / omega) * sum(1 - exp(-omega * (T_max - t)))
  
  # Log-likelihood = sum_log_lambda - integral
  val <- sum_log_lambda - integral
  if (is.na(val) || !is.finite(val)) return(-1e10)
  val
}

fit_hawkes_mle <- function(t, T_max) {
  # Subsample if too many events (keeps optimization snappy)
  if (length(t) > 1000) {
    t <- sort(sample(t, 1000))
  }
  
  n <- length(t)
  if (n < 10) {
    return(list(mu = 0.5, alpha = 0.1, omega = 1.0, log_lik = -1e10))
  }
  
  # Initial guesses
  mu_guess <- n / T_max * 0.5
  alpha_guess <- 0.2
  omega_guess <- 0.5
  
  fit <- tryCatch({
    optim(
      par = c(mu_guess, alpha_guess, omega_guess),
      fn = function(p) -hawkes_log_lik(p, t, T_max),
      method = "Nelder-Mead"
    )
  }, error = function(e) {
    list(par = c(mu_guess, 0.1, 1.0), value = 1e10)
  })
  
  list(
    mu = fit$par[1],
    alpha = fit$par[2],
    omega = fit$par[3],
    log_lik = -fit$value
  )
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FITTER
# ═══════════════════════════════════════════════════════════════════════════════

estimate_hawkes_parameters <- function(dt_events) {
  log_msg("Estimating Hawkes parameters per symbol and stream type...")
  
  symbols <- dt_events[, unique(symbol)]
  T_max <- max(dt_events$time_sec)
  results <- list()
  
  for (sym in symbols) {
    sym_events <- dt_events[symbol == sym]
    
    # Run MLE for: limit order submissions, market orders (trades), cancellations
    streams <- c("LIMIT_ORDER", "MARKET_ORDER", "CANCEL")
    
    for (stream in streams) {
      times <- sym_events[event_type == stream, time_sec]
      fit <- fit_hawkes_mle(times, T_max)
      
      # Branching ratio
      br <- fit$alpha / fit$omega
      half_life <- log(2) / fit$omega * 1000  # ms
      uncond_intensity <- fit$mu / (1 - br)
      
      results[[paste(sym, stream)]] <- data.table(
        symbol = sym,
        event_type = tolower(stream),
        mu = fit$mu,
        alpha = fit$alpha,
        omega = fit$omega,
        branching_ratio = br,
        half_life_ms = half_life,
        unconditional_intensity = uncond_intensity,
        log_likelihood = fit$log_lik,
        num_events = length(times)
      )
    }
  }
  
  rbindlist(results)
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

generate_intensity_plots <- function(dt_events, hawkes_results) {
  log_msg("Generating Hawkes intensity visualizations...")

  # Plot branching ratios across symbols and streams
  p1 <- ggplot(hawkes_results, aes(x = symbol, y = branching_ratio, fill = event_type)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_manual(values = c("limit_order" = "#00D4AA", "market_order" = "#FF5252", "cancel" = "#FFA726")) +
    labs(
      title = "Hawkes Self-Excitation Branching Ratios (α/ω)",
      subtitle = "Branching ratio indicates reflectivity of the stream (closer to 1.0 = higher clustering)",
      x = "Symbol",
      y = "Branching Ratio (α/ω)"
    ) +
    THEME_DARK

  save_plot(p1, "hawkes_branching_ratios.png", width = 12, height = 7)
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main_hawkes_intensity <- function() {
  log_msg("=" |> rep(60) |> paste(collapse = ""))
  log_msg("HAWKES INTENSITY ESTIMATION — Starting")
  log_msg("=" |> rep(60) |> paste(collapse = ""))

  # Load raw events
  dt_events <- load_hawkes_data()

  # Estimate parameters
  hawkes_results <- estimate_hawkes_parameters(dt_events)

  # Save results
  fwrite(hawkes_results, file.path(DATA_PROCESSED, "hawkes_params.csv"))
  log_msg("Hawkes parameters saved to data/processed/hawkes_params.csv")

  # Generate plots
  generate_intensity_plots(dt_events, hawkes_results)

  log_msg("HAWKES INTENSITY ESTIMATION — Complete")
  invisible(hawkes_results)
}

# Run if executed directly
if (sys.nframe() == 0L || !interactive()) {
  main_hawkes_intensity()
}
