#!/usr/bin/env bash
# ==============================================================================
# Market Microstructure & Order Flow Analytics Engine
# Full Pipeline Runner
# ==============================================================================
#
# Usage: ./run_pipeline.sh [stage]
#
# Stages: all, data, analyze, signals, backtest, setup, clean
# Default: all (runs the complete pipeline)
#
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Color codes
# ------------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ------------------------------------------------------------------------------
# Logging functions
# ------------------------------------------------------------------------------
log_info()    { echo -e "${BLUE}[INFO]${NC}    $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC}   $1"; }
log_stage()   { echo -e "\n${MAGENTA}${BOLD}══════════════════════════════════════════════════════════════${NC}"; \
                echo -e "${MAGENTA}${BOLD}  $1${NC}"; \
                echo -e "${MAGENTA}${BOLD}══════════════════════════════════════════════════════════════${NC}\n"; }

# ------------------------------------------------------------------------------
# Timer functions
# ------------------------------------------------------------------------------
PIPELINE_START=$(date +%s)
STAGE_START=0

start_timer() { STAGE_START=$(date +%s); }

end_timer() {
    local end=$(date +%s)
    local elapsed=$((end - STAGE_START))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))
    echo -e "${CYAN}  ⏱  Stage completed in ${minutes}m ${seconds}s${NC}\n"
}

total_timer() {
    local end=$(date +%s)
    local elapsed=$((end - PIPELINE_START))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))
    echo -e "${CYAN}  ⏱  Total pipeline time: ${minutes}m ${seconds}s${NC}"
}

# ------------------------------------------------------------------------------
# Error handler
# ------------------------------------------------------------------------------
on_error() {
    local exit_code=$?
    local line_no=$1
    log_error "Pipeline failed at line ${line_no} with exit code ${exit_code}"
    log_error "Check the output above for details."
    total_timer
    exit $exit_code
}

trap 'on_error ${LINENO}' ERR

# ------------------------------------------------------------------------------
# Prerequisite checks
# ------------------------------------------------------------------------------
check_prerequisites() {
    log_stage "Checking Prerequisites"

    local all_ok=true

    # Check Python
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 --version 2>&1)
        log_success "Python: ${py_version}"
    elif command -v python &> /dev/null; then
        local py_version=$(python --version 2>&1)
        log_success "Python: ${py_version}"
    else
        log_error "Python not found. Please install Python 3.10+"
        all_ok=false
    fi

    # Check R
    if command -v Rscript &> /dev/null; then
        local r_version=$(Rscript --version 2>&1 | head -1)
        log_success "R: ${r_version}"
    else
        log_error "R / Rscript not found. Please install R 4.3+"
        all_ok=false
    fi

    # Check required directories
    local dirs=("data/raw" "data/processed" "data/signals" "reports/figures")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: ${dir}"
        fi
    done

    if [ "$all_ok" = false ]; then
        log_error "Prerequisites check failed. Please install missing tools."
        exit 1
    fi

    log_success "All prerequisites satisfied."
}

# ------------------------------------------------------------------------------
# Pipeline stages
# ------------------------------------------------------------------------------
run_data() {
    log_stage "Stage 1/4: Data Generation"
    start_timer
    log_info "Generating synthetic order book data via Hawkes processes..."
    python python/data_generation/synthetic_generator.py
    log_success "Data generation complete."
    log_info "Output: data/raw/"
    end_timer
}

run_analyze() {
    log_stage "Stage 2/4: Microstructure Analysis"
    start_timer

    local scripts=(
        "01_spread_analysis.R:Spread Analysis"
        "02_order_imbalance.R:Order Imbalance"
        "03_trade_classification.R:Trade Classification"
        "04_adverse_selection.R:Adverse Selection"
        "05_price_impact.R:Price Impact"
        "06_queue_decay.R:Queue Decay"
        "07_hawkes_intensity.R:Hawkes Intensity"
        "08_cross_metric_correlation.R:Cross-Metric Correlation"
    )

    local total=${#scripts[@]}
    local current=0

    for entry in "${scripts[@]}"; do
        IFS=':' read -r script name <<< "$entry"
        current=$((current + 1))
        log_info "[${current}/${total}] Running ${name}..."
        Rscript "R/${script}"
        log_success "${name} complete."
    done

    log_info "Output: data/processed/"
    end_timer
}

run_signals() {
    log_stage "Stage 3/4: Signal Generation"
    start_timer

    log_info "[1/3] Generating composite signals..."
    python python/signals/signal_generator.py
    log_success "Composite signals generated."

    log_info "[2/3] Computing entry/exit rules..."
    python python/signals/entry_exit_rules.py
    log_success "Entry/exit rules computed."

    log_info "[3/3] Calibrating thresholds..."
    python python/signals/threshold_calibrator.py
    log_success "Threshold calibration complete."

    log_info "Output: data/signals/"
    end_timer
}

run_backtest() {
    log_stage "Stage 4/4: Backtesting"
    start_timer
    log_info "Running walk-forward backtest..."
    Rscript R/09_backtest.R
    log_success "Backtest complete."
    log_info "Output: reports/figures/"
    end_timer
}

run_setup() {
    log_stage "Setting Up Dependencies"
    start_timer

    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    log_success "Python dependencies installed."

    log_info "Installing R dependencies..."
    Rscript -e "install.packages(c('data.table','ggplot2','TTR','zoo','xts','yaml','arrow','survival','foreach','doParallel','moments','PerformanceAnalytics'), repos='https://cran.r-project.org')"
    log_success "R dependencies installed."

    end_timer
}

run_clean() {
    log_stage "Cleaning Generated Files"
    start_timer

    rm -f data/raw/*.csv data/raw/*.parquet
    rm -f data/processed/*.csv data/processed/*.parquet
    rm -f data/signals/*.csv data/signals/*.parquet
    rm -f reports/figures/*.png reports/figures/*.pdf reports/figures/*.html

    log_success "All generated files removed. (.gitkeep files preserved)"
    end_timer
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
main() {
    echo -e "${WHITE}${BOLD}"
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║   Market Microstructure & Order Flow Analytics Engine   ║"
    echo "  ║                   Pipeline Runner                      ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    local stage="${1:-all}"

    case "$stage" in
        all)
            check_prerequisites
            run_data
            run_analyze
            run_signals
            run_backtest
            ;;
        data)      check_prerequisites && run_data ;;
        analyze)   check_prerequisites && run_analyze ;;
        signals)   check_prerequisites && run_signals ;;
        backtest)  check_prerequisites && run_backtest ;;
        setup)     run_setup ;;
        clean)     run_clean ;;
        *)
            echo "Usage: $0 [stage]"
            echo ""
            echo "Stages:"
            echo "  all       Run the complete pipeline (default)"
            echo "  data      Generate synthetic market data"
            echo "  analyze   Run R analysis scripts (01-08)"
            echo "  signals   Generate trading signals"
            echo "  backtest  Run walk-forward backtest"
            echo "  setup     Install all dependencies"
            echo "  clean     Remove all generated files"
            exit 1
            ;;
    esac

    echo ""
    log_stage "Pipeline Complete"
    log_success "All stages finished successfully!"
    total_timer
    echo ""
}

main "$@"
