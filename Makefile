# ==============================================================================
# Market Microstructure & Order Flow Analytics Engine
# Pipeline Orchestration Makefile
# ==============================================================================

.PHONY: all setup data analyze signals backtest clean help

# Default target
all: data analyze signals backtest
	@echo ""
	@echo "========================================"
	@echo "  Pipeline completed successfully!"
	@echo "========================================"
	@echo ""

# ------------------------------------------------------------------------------
# Setup: Install R and Python dependencies
# ------------------------------------------------------------------------------
setup:
	@echo "========================================"
	@echo "  Installing dependencies..."
	@echo "========================================"
	@echo ""
	@echo "[1/2] Installing Python dependencies..."
	pip install -r requirements.txt
	@echo ""
	@echo "[2/2] Installing R dependencies..."
	Rscript -e "install.packages(c('data.table','ggplot2','TTR','zoo','xts','yaml','arrow','survival','foreach','doParallel','moments','PerformanceAnalytics'), repos='https://cran.r-project.org')"
	@echo ""
	@echo "  Setup complete."
	@echo ""

# ------------------------------------------------------------------------------
# Data Generation: Synthetic order book data via Hawkes processes
# ------------------------------------------------------------------------------
data:
	@echo "========================================"
	@echo "  Stage 1: Data Generation"
	@echo "========================================"
	@echo ""
	python python/data_generation/synthetic_generator.py
	@echo ""
	@echo "  Data generation complete."
	@echo ""

# ------------------------------------------------------------------------------
# Analysis: Run all R microstructure analysis scripts (01-08)
# ------------------------------------------------------------------------------
analyze:
	@echo "========================================"
	@echo "  Stage 2: Microstructure Analysis"
	@echo "========================================"
	@echo ""
	@echo "[1/8] Spread Analysis..."
	Rscript R/01_spread_analysis.R
	@echo "[2/8] Order Imbalance..."
	Rscript R/02_order_imbalance.R
	@echo "[3/8] Trade Classification..."
	Rscript R/03_trade_classification.R
	@echo "[4/8] Adverse Selection..."
	Rscript R/04_adverse_selection.R
	@echo "[5/8] Price Impact..."
	Rscript R/05_price_impact.R
	@echo "[6/8] Queue Decay..."
	Rscript R/06_queue_decay.R
	@echo "[7/8] Hawkes Intensity..."
	Rscript R/07_hawkes_intensity.R
	@echo "[8/8] Cross-Metric Correlation..."
	Rscript R/08_cross_metric_correlation.R
	@echo ""
	@echo "  Analysis complete."
	@echo ""

# ------------------------------------------------------------------------------
# Signals: Generate composite trading signals with adaptive calibration
# ------------------------------------------------------------------------------
signals:
	@echo "========================================"
	@echo "  Stage 3: Signal Generation"
	@echo "========================================"
	@echo ""
	@echo "[1/3] Composite Signal Generator..."
	python python/signals/signal_generator.py
	@echo "[2/3] Entry/Exit Rules..."
	python python/signals/entry_exit_rules.py
	@echo "[3/3] Threshold Calibrator..."
	python python/signals/threshold_calibrator.py
	@echo ""
	@echo "  Signal generation complete."
	@echo ""

# ------------------------------------------------------------------------------
# Backtest: Walk-forward backtesting with performance analytics
# ------------------------------------------------------------------------------
backtest:
	@echo "========================================"
	@echo "  Stage 4: Backtesting"
	@echo "========================================"
	@echo ""
	Rscript R/09_backtest.R
	@echo ""
	@echo "  Backtest complete."
	@echo ""

# ------------------------------------------------------------------------------
# Clean: Remove all generated data and reports
# ------------------------------------------------------------------------------
clean:
	@echo "========================================"
	@echo "  Cleaning generated files..."
	@echo "========================================"
	@echo ""
	rm -f data/raw/*.csv data/raw/*.parquet
	rm -f data/processed/*.csv data/processed/*.parquet
	rm -f data/signals/*.csv data/signals/*.parquet
	rm -f reports/figures/*.png reports/figures/*.pdf reports/figures/*.html
	@echo "  Clean complete. (.gitkeep files preserved)"
	@echo ""

# ------------------------------------------------------------------------------
# Help: Display available targets
# ------------------------------------------------------------------------------
help:
	@echo ""
	@echo "================================================================"
	@echo "  Market Microstructure & Order Flow Analytics Engine"
	@echo "  Available Make Targets"
	@echo "================================================================"
	@echo ""
	@echo "  make all        Run the complete pipeline (data → backtest)"
	@echo "  make setup      Install all R and Python dependencies"
	@echo "  make data       Generate synthetic market data"
	@echo "  make analyze    Run all R analysis scripts (01-08)"
	@echo "  make signals    Generate trading signals with calibration"
	@echo "  make backtest   Execute walk-forward backtest"
	@echo "  make clean      Remove all generated data and reports"
	@echo "  make help       Display this help message"
	@echo ""
	@echo "================================================================"
	@echo "  Pipeline Order:  data → analyze → signals → backtest"
	@echo "================================================================"
	@echo ""
