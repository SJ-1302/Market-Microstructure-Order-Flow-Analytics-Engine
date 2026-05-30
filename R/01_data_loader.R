###############################################################################
# 01_data_loader.R — Data Ingestion, Cleaning & Feature Engineering
# Market Microstructure & Order Flow Analytics Engine
#
# Pipeline:
#   1. Load raw order book snapshots and trade prints
#   2. Clean: remove outliers, filter market hours, handle duplicates
#   3. Merge order book + trades → compute derived fields
#   4. Save processed data for downstream analysis
#
# Input:  data/raw/  (CSV files from exchange data feeds)
# Output: data/processed/  (.rds and .csv files)
###############################################################################

# ── Source utilities ─────────────────────────────────────────────────────────
source(file.path(dirname(sys.frame(1)$ofile %||% "R/01_data_loader.R"), "utils", "helpers.R"),
       local = FALSE)
source(file.path(dirname(sys.frame(1)$ofile %||% "R/01_data_loader.R"), "utils", "metrics.R"),
       local = FALSE)

setup_project()

suppressPackageStartupMessages({
  library(data.table)
  library(TTR)
})

# ═══════════════════════════════════════════════════════════════════════════════
# load_orderbook_data(filepath)
#
# Reads raw order book snapshot data from CSV.
# Expected columns: timestamp, symbol, bid1..bid5, ask1..ask5,
#                    bid_qty1..bid_qty5, ask_qty1..ask_qty5
#
# If no file exists, generates realistic synthetic data for demonstration.
# ═══════════════════════════════════════════════════════════════════════════════

load_orderbook_data <- function(filepath = NULL) {
  if (!is.null(filepath) && file.exists(filepath)) {
    log_msg(paste("Loading order book data from:", filepath))
    dt <- fread(filepath, showProgress = FALSE)
    # Parse timestamps — handle multiple formats
    if ("timestamp" %in% names(dt)) {
      dt[, timestamp := as.POSIXct(timestamp, format = "%Y-%m-%d %H:%M:%OS",
                                   tz = "Asia/Kolkata")]
    }
    return(dt)
  }

  # ── Generate synthetic order book data for demonstration ──
  log_msg("No raw data file found — generating synthetic order book data", level = "WARN")

  set.seed(42)
  symbols   <- c("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")
  n_per_sym <- 10000L
  n_total   <- n_per_sym * length(symbols)

  # Base prices for NSE large-cap stocks (INR)
  base_prices <- c(RELIANCE = 2450, TCS = 3600, HDFCBANK = 1650,
                   INFY = 1480, ICICIBANK = 980)

  # Generate timestamps across a trading day: 09:15 to 15:30 IST
  base_date <- as.POSIXct("2025-03-15 09:15:00", tz = "Asia/Kolkata")
  trading_seconds <- 6 * 3600 + 15 * 60  # 6h 15m in seconds

  dt_list <- lapply(symbols, function(sym) {
    base_px <- base_prices[sym]

    # Random timestamps within trading hours
    ts_offsets <- sort(runif(n_per_sym, 0, trading_seconds))
    timestamps <- base_date + ts_offsets

    # Price process: geometric Brownian motion with mean-reversion
    log_returns <- rnorm(n_per_sym, mean = 0, sd = 0.0003)
    log_px      <- log(base_px) + cumsum(log_returns)
    mid_prices  <- exp(log_px)

    # Spread: wider at open/close (U-shape), tighter mid-day
    hours_elapsed  <- ts_offsets / 3600
    # U-shape: high at 0 and 6.25, low at ~3.1
    spread_factor  <- 1 + 0.8 * ((hours_elapsed - 3.125)^2 / 3.125^2)
    base_spread    <- base_px * 0.0005  # 5 bps base spread
    half_spreads   <- (base_spread * spread_factor) / 2

    # 5 levels of depth
    tick_size <- round(base_px * 0.05 / 100, 2)  # ~5 paise tick
    tick_size <- pmax(tick_size, 0.05)

    dt <- data.table(
      timestamp  = timestamps,
      symbol     = sym,
      bid1       = round(mid_prices - half_spreads, 2),
      ask1       = round(mid_prices + half_spreads, 2),
      bid_qty1   = as.integer(rpois(n_per_sym, lambda = 500)),
      ask_qty1   = as.integer(rpois(n_per_sym, lambda = 500)),
      bid2       = round(mid_prices - half_spreads - tick_size, 2),
      ask2       = round(mid_prices + half_spreads + tick_size, 2),
      bid_qty2   = as.integer(rpois(n_per_sym, lambda = 350)),
      ask_qty2   = as.integer(rpois(n_per_sym, lambda = 350)),
      bid3       = round(mid_prices - half_spreads - 2 * tick_size, 2),
      ask3       = round(mid_prices + half_spreads + 2 * tick_size, 2),
      bid_qty3   = as.integer(rpois(n_per_sym, lambda = 250)),
      ask_qty3   = as.integer(rpois(n_per_sym, lambda = 250)),
      bid4       = round(mid_prices - half_spreads - 3 * tick_size, 2),
      ask4       = round(mid_prices + half_spreads + 3 * tick_size, 2),
      bid_qty4   = as.integer(rpois(n_per_sym, lambda = 200)),
      ask_qty4   = as.integer(rpois(n_per_sym, lambda = 200)),
      bid5       = round(mid_prices - half_spreads - 4 * tick_size, 2),
      ask5       = round(mid_prices + half_spreads + 4 * tick_size, 2),
      bid_qty5   = as.integer(rpois(n_per_sym, lambda = 150)),
      ask_qty5   = as.integer(rpois(n_per_sym, lambda = 150))
    )
    dt
  })

  dt <- rbindlist(dt_list)
  setkey(dt, symbol, timestamp)

  log_msg(sprintf("Generated %s order book snapshots for %d symbols",
                  format(nrow(dt), big.mark = ","), length(symbols)))
  dt
}

# ═══════════════════════════════════════════════════════════════════════════════
# load_trade_data(filepath)
#
# Reads raw trade print data from CSV.
# Expected columns: timestamp, symbol, price, quantity, exchange_order_id
# ═══════════════════════════════════════════════════════════════════════════════

load_trade_data <- function(filepath = NULL) {
  if (!is.null(filepath) && file.exists(filepath)) {
    log_msg(paste("Loading trade data from:", filepath))
    dt <- fread(filepath, showProgress = FALSE)
    if ("timestamp" %in% names(dt)) {
      dt[, timestamp := as.POSIXct(timestamp, format = "%Y-%m-%d %H:%M:%OS",
                                   tz = "Asia/Kolkata")]
    }
    return(dt)
  }

  # ── Generate synthetic trade data correlated with order book ──
  log_msg("No raw trade file found — generating synthetic trade data", level = "WARN")

  set.seed(123)
  symbols   <- c("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")
  n_per_sym <- 8000L

  base_prices <- c(RELIANCE = 2450, TCS = 3600, HDFCBANK = 1650,
                   INFY = 1480, ICICIBANK = 980)

  base_date       <- as.POSIXct("2025-03-15 09:15:00", tz = "Asia/Kolkata")
  trading_seconds <- 6 * 3600 + 15 * 60

  dt_list <- lapply(symbols, function(sym) {
    base_px <- base_prices[sym]

    ts_offsets <- sort(runif(n_per_sym, 0, trading_seconds))
    timestamps <- base_date + ts_offsets

    # Trade prices: GBM with slight noise around the mid
    log_returns <- rnorm(n_per_sym, mean = 0, sd = 0.0004)
    log_px      <- log(base_px) + cumsum(log_returns)
    trade_px    <- round(exp(log_px) + rnorm(n_per_sym, 0, base_px * 0.0002), 2)

    # Trade quantities: power-law distributed (many small, few large)
    quantities <- as.integer(ceiling(rlnorm(n_per_sym, meanlog = 4, sdlog = 1.2)))
    quantities <- pmin(quantities, 50000L)  # cap extreme values

    data.table(
      timestamp = timestamps,
      symbol    = sym,
      price     = trade_px,
      quantity  = quantities
    )
  })

  dt <- rbindlist(dt_list)
  setkey(dt, symbol, timestamp)

  log_msg(sprintf("Generated %s trade records for %d symbols",
                  format(nrow(dt), big.mark = ","), length(symbols)))
  dt
}

# ═══════════════════════════════════════════════════════════════════════════════
# clean_trades(dt)
#
# Removes spurious trades:
#   1. Outlier prices: >3σ from a rolling median (window = 50 trades)
#   2. Non-market-hours trades: outside 09:15–15:30 IST
#   3. Zero/negative prices or quantities
# ═══════════════════════════════════════════════════════════════════════════════

clean_trades <- function(dt) {
  log_msg("Cleaning trade data...")
  n_before <- nrow(dt)

  # Remove zero/negative price or quantity

  dt <- dt[price > 0 & quantity > 0]

  # Filter to market hours (IST): 09:15:00 to 15:30:00
  dt[, hour_min := as.numeric(format(timestamp, "%H")) * 100 +
       as.numeric(format(timestamp, "%M"))]
  dt <- dt[hour_min >= 915 & hour_min <= 1530]
  dt[, hour_min := NULL]

  # Rolling median outlier detection per symbol
  # A trade is an outlier if |price - rolling_median| > 3 × rolling_MAD
  dt[, `:=`(
    roll_median = frollmean(price, n = 50L, align = "center", na.rm = TRUE),
    roll_mad    = frollapply(price, n = 50L, FUN = mad, align = "center")
  ), by = symbol]

  # Fill leading/trailing NAs with global symbol median/mad
  dt[is.na(roll_median), roll_median := median(price, na.rm = TRUE), by = symbol]
  dt[is.na(roll_mad) | roll_mad == 0, roll_mad := mad(price, na.rm = TRUE), by = symbol]

  # Flag and remove outliers (>3σ from rolling median)
  dt[, is_outlier := abs(price - roll_median) > 3 * roll_mad]
  n_outliers <- sum(dt$is_outlier, na.rm = TRUE)
  dt <- dt[is_outlier == FALSE]

  # Clean up temporary columns
  dt[, c("roll_median", "roll_mad", "is_outlier") := NULL]

  n_after <- nrow(dt)
  log_msg(sprintf("Cleaning complete: %d → %d rows (%d removed, %d outliers)",
                  n_before, n_after, n_before - n_after, n_outliers))
  dt
}

# ═══════════════════════════════════════════════════════════════════════════════
# merge_concurrent_trades(dt)
#
# Aggregates multiple trade prints at the exact same timestamp and symbol
# into a single record using VWAP and total quantity.
# This avoids double-counting in downstream analysis.
# ═══════════════════════════════════════════════════════════════════════════════

merge_concurrent_trades <- function(dt) {
  log_msg("Merging concurrent trades at same timestamp...")
  n_before <- nrow(dt)

  dt_merged <- dt[, .(
    price    = sum(price * quantity) / sum(quantity),  # VWAP
    quantity = sum(quantity),
    n_trades = .N
  ), by = .(timestamp, symbol)]

  # Round price to 2 decimal places (NSE tick precision)
  dt_merged[, price := round(price, 2)]
  setkey(dt_merged, symbol, timestamp)

  log_msg(sprintf("Merged: %d → %d rows (%d concurrent groups collapsed)",
                  n_before, nrow(dt_merged), n_before - nrow(dt_merged)))
  dt_merged
}

# ═══════════════════════════════════════════════════════════════════════════════
# compute_derived_fields(dt_book, dt_trades)
#
# Enriches the merged dataset with derived microstructure fields:
#   - mid_price: (bid1 + ask1) / 2
#   - quoted_spread, pct_spread
#   - log_returns: log(mid_t / mid_{t-1})
#   - trade_sign: Lee-Ready classification (+1 / -1)
#   - signed_volume: trade_sign × quantity
#   - dollar_volume: price × quantity
# ═══════════════════════════════════════════════════════════════════════════════

compute_derived_fields <- function(dt_book, dt_trades) {
  log_msg("Computing derived fields...")

  # ── Order book enrichment ──
  dt_book[, mid_price := (bid1 + ask1) / 2]
  dt_book[, quoted_spread := compute_quoted_spread(bid1, ask1)]
  dt_book[, pct_spread := compute_pct_spread(bid1, ask1)]
  dt_book[, log_return := c(NA_real_, diff(log(mid_price))), by = symbol]

  # OBI at level 1
  dt_book[, obi := compute_obi(bid_qty1, ask_qty1)]

  # Weighted OBI across 5 levels
  bid_cols <- paste0("bid_qty", 1:5)
  ask_cols <- paste0("ask_qty", 1:5)
  if (all(c(bid_cols, ask_cols) %in% names(dt_book))) {
    dt_book[, weighted_obi := compute_weighted_obi(
      .SD[, ..bid_cols], .SD[, ..ask_cols], levels = 5L
    )]
  }

  log_msg(sprintf("Order book enriched: %d rows, %d columns",
                  nrow(dt_book), ncol(dt_book)))

  # ── Trade enrichment ──
  # Rolling join: for each trade, find the most recent order book snapshot
  # This gives us bid/ask at the time of each trade for Lee-Ready classification
  setkey(dt_book, symbol, timestamp)
  setkey(dt_trades, symbol, timestamp)

  dt_merged <- dt_book[dt_trades, roll = TRUE, nomatch = NA]

  # Compute trade side using Lee-Ready algorithm
  dt_merged[, trade_sign := classify_trade_side(
    trade_price = i.price,
    bid = bid1,
    ask = ask1
  ), by = symbol]

  # Use original trade price where available
  if ("i.price" %in% names(dt_merged)) {
    dt_merged[, trade_price := i.price]
  }
  if ("i.quantity" %in% names(dt_merged)) {
    dt_merged[, trade_qty := i.quantity]
  }

  # Signed volume: positive for buys, negative for sells
  dt_merged[, signed_volume := trade_sign * trade_qty]

  # Dollar volume for Amihud calculation
  dt_merged[, dollar_volume := trade_price * trade_qty]

  # Effective spread
  dt_merged[, effective_spread := compute_effective_spread(trade_price, mid_price)]

  log_msg(sprintf("Merged dataset: %d rows, %d columns",
                  nrow(dt_merged), ncol(dt_merged)))

  list(book = dt_book, trades = dt_trades, merged = dt_merged)
}

# ═══════════════════════════════════════════════════════════════════════════════
# save_processed(dt, filename)
#
# Saves to both .rds (fast R binary) and .csv (portable) formats.
# ═══════════════════════════════════════════════════════════════════════════════

save_processed <- function(dt, filename) {
  if (!dir.exists(DATA_PROCESSED)) {
    dir.create(DATA_PROCESSED, recursive = TRUE)
  }

  rds_path <- file.path(DATA_PROCESSED, paste0(filename, ".rds"))
  csv_path <- file.path(DATA_PROCESSED, paste0(filename, ".csv"))

  saveRDS(dt, rds_path)
  fwrite(dt, csv_path)

  log_msg(sprintf("Saved: %s (%s rows) → .rds + .csv",
                  filename, format(nrow(dt), big.mark = ",")))
  invisible(list(rds = rds_path, csv = csv_path))
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main_data_loader <- function() {
  log_msg("=" |> rep(60) |> paste(collapse = ""))
  log_msg("DATA LOADER — Starting pipeline")
  log_msg("=" |> rep(60) |> paste(collapse = ""))

  # Step 1: Load raw data
  raw_book_path  <- file.path(DATA_RAW, "orderbook.csv")
  raw_trade_path <- file.path(DATA_RAW, "trades.csv")

  dt_book   <- load_orderbook_data(raw_book_path)
  dt_trades <- load_trade_data(raw_trade_path)

  # Step 2: Clean trades
  dt_trades <- clean_trades(dt_trades)

  # Step 3: Merge concurrent trades
  dt_trades <- merge_concurrent_trades(dt_trades)

  # Step 4: Compute derived fields
  result <- compute_derived_fields(dt_book, dt_trades)

  # Step 5: Save processed datasets
  save_processed(result$book,   "orderbook_processed")
  save_processed(result$merged, "trades_enriched")

  # Summary statistics
  log_msg("── Summary Statistics ──")
  log_msg(sprintf("Symbols: %s", paste(unique(result$book$symbol), collapse = ", ")))
  log_msg(sprintf("Order book snapshots: %s", format(nrow(result$book), big.mark = ",")))
  log_msg(sprintf("Enriched trades: %s", format(nrow(result$merged), big.mark = ",")))
  log_msg(sprintf("Date range: %s to %s",
                  min(result$book$timestamp, na.rm = TRUE),
                  max(result$book$timestamp, na.rm = TRUE)))

  # Per-symbol summary
  sym_summary <- result$merged[, .(
    n_trades      = .N,
    avg_price     = round(mean(trade_price, na.rm = TRUE), 2),
    avg_spread_bp = round(mean(pct_spread, na.rm = TRUE) * 10000, 2),
    buy_pct       = round(mean(trade_sign == 1, na.rm = TRUE) * 100, 1)
  ), by = symbol]

  log_msg("Per-symbol summary:")
  print(sym_summary)

  log_msg("DATA LOADER — Pipeline complete")
  invisible(result)
}

# Run if executed directly (not sourced)
if (sys.nframe() == 0L || !interactive()) {
  main_data_loader()
}
