###############################################################################
# helpers.R — Utility Functions & Configuration
# Market Microstructure & Order Flow Analytics Engine
#
# Provides:
#   - Project setup and directory management
#   - Formatted logging with severity levels
#   - Package management
#   - Custom dark ggplot2 theme (THEME_DARK)
#   - Standardised plot saving
#   - Path constants for data pipeline
###############################################################################

# ── Required packages ────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
})

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONSTANTS
# All paths are relative to the project root so scripts work on any machine.
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."), "..", ".."),
                              winslash = "/", mustWork = FALSE)

# Fallback: if sourced interactively, use working directory
if (!dir.exists(PROJECT_ROOT) || nchar(PROJECT_ROOT) < 3) {
  PROJECT_ROOT <- normalizePath(getwd(), winslash = "/")
}

DATA_RAW        <- file.path(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED  <- file.path(PROJECT_ROOT, "data", "processed")
DATA_SIGNALS    <- file.path(PROJECT_ROOT, "data", "signals")
REPORTS_FIGURES <- file.path(PROJECT_ROOT, "reports", "figures")

# ═══════════════════════════════════════════════════════════════════════════════
# setup_project()
# Creates the directory tree and sets the working directory to PROJECT_ROOT.
# Safe to call multiple times (idempotent).
# ═══════════════════════════════════════════════════════════════════════════════

setup_project <- function() {
  dirs <- c(DATA_RAW, DATA_PROCESSED, DATA_SIGNALS, REPORTS_FIGURES,
            file.path(PROJECT_ROOT, "reports", "tables"),
            file.path(PROJECT_ROOT, "R", "utils"))
  for (d in dirs) {
    if (!dir.exists(d)) {
      dir.create(d, recursive = TRUE, showWarnings = FALSE)
      log_msg(paste("Created directory:", d))
    }
  }
  setwd(PROJECT_ROOT)
  log_msg(paste("Working directory set to:", PROJECT_ROOT))
  invisible(PROJECT_ROOT)
}

# ═══════════════════════════════════════════════════════════════════════════════
# log_msg(msg, level)
# Formatted console logging with ISO-8601 timestamps.
# Levels: INFO, WARN, ERROR, DEBUG
# ═══════════════════════════════════════════════════════════════════════════════

log_msg <- function(msg, level = "INFO") {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] [%-5s] %s\n", ts, toupper(level), msg))
  invisible(NULL)
}

# ═══════════════════════════════════════════════════════════════════════════════
# ensure_packages(pkgs)
# Check for missing packages and install them from CRAN.
# Returns a character vector of packages that were newly installed.
# ═══════════════════════════════════════════════════════════════════════════════

ensure_packages <- function(pkgs) {
  missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0L) {
    log_msg(paste("Installing missing packages:", paste(missing, collapse = ", ")))
    install.packages(missing, repos = "https://cloud.r-project.org", quiet = TRUE)
  } else {
    log_msg("All required packages are already installed.")
  }
  invisible(missing)
}

# ═══════════════════════════════════════════════════════════════════════════════
# THEME_DARK — Custom dark ggplot2 theme
#
# Design rationale:
#   - Dark background (#1a1a2e) reduces eye strain on trading screens
#   - Light text (#e0e0e0) provides high contrast against dark background
#   - Grid lines (#2d2d44) are visible but unobtrusive
#   - Viridis palette is colour-blind friendly and prints well in greyscale
#   - Sans-serif font for clean, modern readability
# ═══════════════════════════════════════════════════════════════════════════════

THEME_DARK <- theme(
  # ── Background ──
  plot.background    = element_rect(fill = "#1a1a2e", colour = NA),
  panel.background   = element_rect(fill = "#1a1a2e", colour = NA),
  legend.background  = element_rect(fill = "#1a1a2e", colour = NA),
  legend.key         = element_rect(fill = "#1a1a2e", colour = NA),


  # ── Grid lines ──

  panel.grid.major   = element_line(colour = "#2d2d44", linewidth = 0.3),
  panel.grid.minor   = element_line(colour = "#2d2d44", linewidth = 0.15),


  # ── Text ──
  text               = element_text(family = "sans", colour = "#e0e0e0"),
  plot.title         = element_text(size = 16, face = "bold", colour = "#e0e0e0",
                                    hjust = 0, margin = margin(b = 8)),
  plot.subtitle      = element_text(size = 11, colour = "#b0b0b0",
                                    hjust = 0, margin = margin(b = 12)),
  plot.caption       = element_text(size = 8, colour = "#888888",
                                    hjust = 1, margin = margin(t = 8)),
  axis.title         = element_text(size = 11, colour = "#c0c0c0"),
  axis.text          = element_text(size = 9, colour = "#a0a0a0"),
  legend.title       = element_text(size = 10, colour = "#c0c0c0"),
  legend.text        = element_text(size = 9, colour = "#b0b0b0"),
  strip.text         = element_text(size = 10, colour = "#e0e0e0", face = "bold"),

  # ── Facet strips ──
  strip.background   = element_rect(fill = "#2d2d44", colour = NA),

  # ── Axes ──
  axis.ticks         = element_line(colour = "#555555"),
  axis.line          = element_line(colour = "#555555", linewidth = 0.4),

  # ── Legend position ──
  legend.position    = "bottom",

  # ── Margins ──
  plot.margin        = margin(12, 16, 12, 12)
)

# ═══════════════════════════════════════════════════════════════════════════════
# save_plot(p, filename, width, height)
# Saves a ggplot object to REPORTS_FIGURES with sensible defaults for
# publication-quality output (300 dpi PNG).
# ═══════════════════════════════════════════════════════════════════════════════

save_plot <- function(p, filename, width = 12, height = 8) {
  # Ensure the output directory exists
  if (!dir.exists(REPORTS_FIGURES)) {
    dir.create(REPORTS_FIGURES, recursive = TRUE, showWarnings = FALSE)
  }

  filepath <- file.path(REPORTS_FIGURES, filename)

  ggsave(
    filename = filepath,
    plot     = p,
    width    = width,
    height   = height,
    dpi      = 300,
    bg       = "#1a1a2e"
  )

  log_msg(paste("Plot saved:", filepath))
  invisible(filepath)
}

# ═══════════════════════════════════════════════════════════════════════════════
# format_number(x, digits)
# Pretty-print large numbers with commas and fixed decimal places.
# ═══════════════════════════════════════════════════════════════════════════════

format_number <- function(x, digits = 2) {
  formatC(round(x, digits), format = "f", digits = digits, big.mark = ",")
}

# ═══════════════════════════════════════════════════════════════════════════════
# pct(x, digits)
# Express a proportion as a percentage string, e.g. 0.0532 → "5.32%"
# ═══════════════════════════════════════════════════════════════════════════════

pct <- function(x, digits = 2) {
  paste0(format_number(x * 100, digits), "%")
}

# ═══════════════════════════════════════════════════════════════════════════════
# safe_div(num, denom, fill)
# Division guarded against zero / NA denominators.
# ═══════════════════════════════════════════════════════════════════════════════

safe_div <- function(num, denom, fill = NA_real_) {
  result <- num / denom
  result[!is.finite(result)] <- fill
  result
}

# ── Auto-setup when sourced ──────────────────────────────────────────────────
log_msg("helpers.R loaded — Market Microstructure & Order Flow Analytics Engine")
