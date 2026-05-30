###############################################################################
# metrics.R — Shared Metric Computation Functions
# Market Microstructure & Order Flow Analytics Engine
#
# Contains all core microstructure metric computations used across modules.
# Every function is vectorised for efficient use with data.table columns.
#
# References:
#   - Lee & Ready (1991): Trade classification
#   - Kyle (1985): Lambda / price impact
#   - Amihud (2002): Illiquidity ratio
#   - Easley, Kiefer, O'Hara & Paperman (1996): PIN model
###############################################################################

# ═══════════════════════════════════════════════════════════════════════════════
# SPREAD METRICS
# ═══════════════════════════════════════════════════════════════════════════════

#' Quoted Spread (absolute)
#' The raw difference between the best ask and best bid.
#' This is the cost a market-taker pays for immediate execution of a
#' round-trip (buy + sell) at the top of the book.
#'
#' @param bid Numeric vector of best bid prices
#' @param ask Numeric vector of best ask prices
#' @return Numeric vector: ask - bid
compute_quoted_spread <- function(bid, ask) {
  ask - bid
}

#' Percentage Spread (relative)
#' Normalises the quoted spread by the midpoint, making it comparable
#' across stocks with different price levels.
#'   pct_spread = (ask - bid) / midpoint
#'
#' @param bid Numeric vector of best bid prices
#' @param ask Numeric vector of best ask prices
#' @return Numeric vector of percentage spreads
compute_pct_spread <- function(bid, ask) {
  mid <- (ask + bid) / 2
  result <- (ask - bid) / mid
  result[!is.finite(result)] <- NA_real_
  result
}

#' Effective Spread
#' Measures the actual cost paid by a trader, accounting for trades
#' inside the quoted spread (price improvement).
#'   effective_spread = 2 × |trade_price − mid_price|
#' The factor of 2 converts a one-sided cost into a round-trip cost.
#'
#' @param trade_price Numeric vector of trade execution prices
#' @param mid_price Numeric vector of mid-prices at time of trade
#' @return Numeric vector of effective spreads
compute_effective_spread <- function(trade_price, mid_price) {
  2 * abs(trade_price - mid_price)
}

#' Realized Spread
#' Decomposes the effective spread into adverse selection and realized
#' components. The realized spread is the market maker's actual profit
#' after prices move to reflect information.
#'   realized_spread = 2 × trade_sign × (trade_price − mid_price_future)
#' where mid_price_future is the mid τ seconds after the trade.
#'
#' @param trade_price Numeric vector of trade execution prices
#' @param mid_price_future Numeric vector of mid-prices τ seconds after trade
#' @param trade_sign Numeric vector: +1 for buyer-initiated, -1 for seller-initiated
#' @return Numeric vector of realized spreads
compute_realized_spread <- function(trade_price, mid_price_future, trade_sign) {
  2 * trade_sign * (trade_price - mid_price_future)
}

# ═══════════════════════════════════════════════════════════════════════════════
# ORDER IMBALANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

#' Order Book Imbalance (Level-1)
#' Measures the relative pressure between buyers and sellers at the
#' top of the order book.
#'   OBI = (bid_qty − ask_qty) / (bid_qty + ask_qty)
#' Range: [-1, +1]. Positive → more bid-side pressure (bullish).
#'
#' @param bid_qty Numeric vector of bid-side quantities at best bid
#' @param ask_qty Numeric vector of ask-side quantities at best ask
#' @return Numeric vector of OBI values in [-1, 1]
compute_obi <- function(bid_qty, ask_qty) {
  total <- bid_qty + ask_qty
  result <- (bid_qty - ask_qty) / total
  result[!is.finite(result)] <- 0
  result
}

#' Weighted Order Book Imbalance (Multi-Level)
#' Extends OBI across multiple depth levels, with exponentially
#' decaying weights by distance from the best price.
#'   w_k = exp(-0.5 × k)  for level k = 0, 1, ..., levels-1
#' This gives more importance to queues closer to the touch.
#'
#' @param bid_qtys Matrix or list of bid quantities (rows = observations, cols = levels)
#' @param ask_qtys Matrix or list of ask quantities (rows = observations, cols = levels)
#' @param levels Integer number of depth levels to include (default 5)
#' @return Numeric vector of weighted OBI values in [-1, 1]
compute_weighted_obi <- function(bid_qtys, ask_qtys, levels = 5L) {
  # Ensure inputs are matrices
  bid_mat <- as.matrix(bid_qtys)
  ask_mat <- as.matrix(ask_qtys)

  # Use only up to 'levels' columns
  n_levels <- min(ncol(bid_mat), ncol(ask_mat), levels)
  bid_mat  <- bid_mat[, seq_len(n_levels), drop = FALSE]
  ask_mat  <- ask_mat[, seq_len(n_levels), drop = FALSE]

  # Exponential distance-decay weights: nearer levels get more weight
  # w_k = exp(-0.5 * k), so level 0 → weight 1.0, level 4 → weight 0.135
  weights <- exp(-0.5 * (seq_len(n_levels) - 1))
  weights <- weights / sum(weights)  # normalise to sum to 1


  # Weighted bid and ask totals
  weighted_bid <- bid_mat %*% weights
  weighted_ask <- ask_mat %*% weights

  # Compute imbalance
  total  <- weighted_bid + weighted_ask
  result <- (weighted_bid - weighted_ask) / total
  result[!is.finite(result)] <- 0
  as.numeric(result)
}

#' Volume Imbalance
#' Measures the directional imbalance of traded volume.
#'   VI = (buy_volume − sell_volume) / (buy_volume + sell_volume)
#'
#' @param buy_vol Numeric vector of buyer-initiated volume
#' @param sell_vol Numeric vector of seller-initiated volume
#' @return Numeric vector of volume imbalance in [-1, 1]
compute_volume_imbalance <- function(buy_vol, sell_vol) {
  total <- buy_vol + sell_vol
  result <- (buy_vol - sell_vol) / total
  result[!is.finite(result)] <- 0
  result
}

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

#' Lee-Ready Trade Classification Algorithm
#'
#' Classifies each trade as buyer-initiated (+1) or seller-initiated (-1).
#'
#' Algorithm (Lee & Ready, 1991):
#'   1. Quote test: If trade_price > mid → BUY (+1)
#'                  If trade_price < mid → SELL (-1)
#'   2. Tick test (when trade_price == mid):
#'      Compare to previous trade price:
#'        - uptick or zero-uptick → BUY (+1)
#'        - downtick or zero-downtick → SELL (-1)
#'
#' Empirical accuracy: ~85% for NYSE/NASDAQ, ~80% for NSE.
#'
#' @param trade_price Numeric vector of trade prices
#' @param bid Numeric vector of best bid at time of trade
#' @param ask Numeric vector of best ask at time of trade
#' @return Integer vector: +1 (buy), -1 (sell)
classify_trade_side <- function(trade_price, bid, ask) {
  n <- length(trade_price)
  mid <- (bid + ask) / 2
  side <- rep(0L, n)

  # Step 1: Quote test — classify trades above/below midpoint
  side[trade_price > mid] <- 1L   # buyer-initiated
  side[trade_price < mid] <- -1L  # seller-initiated

  # Step 2: Tick test — for trades exactly at the midpoint
  at_mid <- which(side == 0L)
  if (length(at_mid) > 0L) {
    for (i in at_mid) {
      if (i == 1L) {
        # First observation: no prior price to compare → default to BUY
        side[i] <- 1L
        next
      }
      # Walk backward to find the last trade at a different price
      prev_idx <- i - 1L
      while (prev_idx >= 1L && trade_price[prev_idx] == trade_price[i]) {
        prev_idx <- prev_idx - 1L
      }
      if (prev_idx < 1L) {
        side[i] <- 1L  # no prior different price found → default BUY
      } else if (trade_price[i] > trade_price[prev_idx]) {
        side[i] <- 1L  # uptick → BUY
      } else {
        side[i] <- -1L # downtick → SELL
      }
    }
  }

  side
}

# ═══════════════════════════════════════════════════════════════════════════════
# PRICE IMPACT & LIQUIDITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

#' Kyle's Lambda (Price Impact Coefficient)
#'
#' Estimates the permanent price impact per unit of signed order flow
#' via OLS regression:
#'   ΔP_t = λ × Q_t + ε_t
#'
#' where ΔP is the price change and Q is the signed order flow
#' (positive = net buying). Higher λ implies more adverse selection
#' or lower market depth.
#'
#' @param delta_p Numeric vector of price changes
#' @param signed_flow Numeric vector of signed order flow (net buy volume)
#' @return List with lambda (coefficient), tstat, rsq, nobs
compute_kyles_lambda <- function(delta_p, signed_flow) {
  # Remove NA / Inf observations
  valid <- is.finite(delta_p) & is.finite(signed_flow)
  dp <- delta_p[valid]
  sf <- signed_flow[valid]
  n  <- length(dp)

  if (n < 10L) {
    return(list(lambda = NA_real_, tstat = NA_real_,
                rsq = NA_real_, nobs = n))
  }

  # OLS: ΔP = λ × Q + ε
  fit   <- lm(dp ~ sf)
  coefs <- summary(fit)$coefficients

  list(
    lambda = coefs[2, 1],   # slope = Kyle's lambda
    tstat  = coefs[2, 3],   # t-statistic
    rsq    = summary(fit)$r.squared,
    nobs   = n
  )
}

#' Amihud Illiquidity Ratio (2002)
#'
#' Measures price impact per unit of dollar volume:
#'   ILLIQ = (|R_t| / DVOL_t) × 10^6
#'
#' Higher values indicate less liquid (more illiquid) securities.
#' The 10^6 scaling makes the ratio interpretable for typical
#' stock-level dollar volumes.
#'
#' @param returns Numeric vector of log returns
#' @param dollar_volume Numeric vector of dollar trading volume
#' @return Numeric vector of Amihud ratios (× 10^6)
compute_amihud <- function(returns, dollar_volume) {
  result <- (abs(returns) / dollar_volume) * 1e6
  result[!is.finite(result)] <- NA_real_
  result
}

#' Time-Weighted Average Spread (TWAS)
#'
#' Weights each spread observation by the duration it was in effect.
#' This avoids overweighting periods of rapid quote updates.
#'   TWAS = Σ(spread_i × duration_i) / Σ(duration_i)
#'
#' @param spreads Numeric vector of quoted spreads
#' @param durations Numeric vector of durations (seconds or ticks) each spread was active
#' @return Scalar: time-weighted average spread
compute_twas <- function(spreads, durations) {
  valid <- is.finite(spreads) & is.finite(durations) & durations > 0
  if (sum(valid) == 0L) return(NA_real_)
  sum(spreads[valid] * durations[valid]) / sum(durations[valid])
}

# ═══════════════════════════════════════════════════════════════════════════════
log_msg <- function(msg, level = "INFO") {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] [%-5s] %s\n", ts, toupper(level), msg))
  invisible(NULL)
}

log_msg("metrics.R loaded — all microstructure metric functions available")
