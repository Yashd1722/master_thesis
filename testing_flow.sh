#!/bin/bash
#SBATCH --job-name=testing_flow
#SBATCH --output=logs/slurm/testing_flow_%j.out
#SBATCH --error=logs/slurm/testing_flow_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# ============================================================
# Required via --export
# Example:
# sbatch --export=ALL,CHECKPOINT=checkpoints/cnn_lstm_ts_500_acc.pt,TEST_DATASET=pangaea_923197 testing_flow.sh
# ============================================================
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${TEST_DATASET:?TEST_DATASET is required}"

# ============================================================
# Optional config
# ============================================================
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"

PROGRESSIVE_START_FRAC="${PROGRESSIVE_START_FRAC:-0.60}"
PROGRESSIVE_END_FRAC="${PROGRESSIVE_END_FRAC:-1.00}"
PROGRESSIVE_NUM_STEPS="${PROGRESSIVE_NUM_STEPS:-40}"
MIN_PREFIX_LEN="${MIN_PREFIX_LEN:-64}"

ROLLING_WINDOW_FRAC="${ROLLING_WINDOW_FRAC:-0.50}"
GAUSSIAN_SMOOTH_SIGMA="${GAUSSIAN_SMOOTH_SIGMA:--1}"

POSITIVE_CLASS_INDEX="${POSITIVE_CLASS_INDEX:-1}"
POSITIVE_CLASS_NAME="${POSITIVE_CLASS_NAME:-}"
NEUTRAL_LABEL="${NEUTRAL_LABEL:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-results}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
MAKE_PER_SERIES_PLOTS="${MAKE_PER_SERIES_PLOTS:-1}"
SAVE_SUMMARY_CSV="${SAVE_SUMMARY_CSV:-1}"
SAVE_SERIES_CSV="${SAVE_SERIES_CSV:-1}"
VERBOSE="${VERBOSE:-1}"

PROJECT_DIR="${PROJECT_DIR:-$HOME/Master_thesis/master_thesis}"
VENV_PATH="${VENV_PATH:-$HOME/Master_thesis/myenv}"

# ------------------------------------------------------------
# Normalize old path variants
# ------------------------------------------------------------
if [[ "$PROJECT_DIR" == *"/Main"* || "$PROJECT_DIR" == *"/Main/"* ]]; then
    PROJECT_DIR="${PROJECT_DIR//\/Main\//\/master_thesis/}"
    PROJECT_DIR="${PROJECT_DIR%/Main}"
fi

if [[ "$CHECKPOINT" == *"/Main/"* ]]; then
    CHECKPOINT="${CHECKPOINT//\/Main\//\/master_thesis/}"
fi

# If checkpoint is relative, resolve relative to project dir
if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="$PROJECT_DIR/$CHECKPOINT"
fi

# ============================================================
# Move to project
# ============================================================
cd "$PROJECT_DIR"

mkdir -p logs/slurm

echo "========================================================"
echo "SLURM JOB ID         : ${SLURM_JOB_ID:-N/A}"
echo "PROJECT_DIR          : $PROJECT_DIR"
echo "CHECKPOINT           : $CHECKPOINT"
echo "TEST_DATASET         : $TEST_DATASET"
echo "SPLIT                : $SPLIT"
echo "DEVICE               : $DEVICE"
echo "RESULTS_ROOT         : $RESULTS_ROOT"
echo "========================================================"

# ============================================================
# Activate environment
# ============================================================
if [[ -d "$VENV_PATH" ]]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment not found at: $VENV_PATH"
    exit 1
fi

# ============================================================
# Derive run names
# ============================================================
CKPT_BASE="$(basename "$CHECKPOINT" .pt)"
DL_RUN_NAME="${DL_RUN_NAME:-${CKPT_BASE}_to_${TEST_DATASET}}"
CSD_RUN_NAME="${CSD_RUN_NAME:-csd_${TEST_DATASET}}"
COMPARE_RUN_NAME="${COMPARE_RUN_NAME:-compare_${DL_RUN_NAME}_vs_${CSD_RUN_NAME}}"

echo "DL_RUN_NAME          : $DL_RUN_NAME"
echo "CSD_RUN_NAME         : $CSD_RUN_NAME"
echo "COMPARE_RUN_NAME     : $COMPARE_RUN_NAME"
echo "========================================================"

# ============================================================
# Step 1: Deep learning testing
# ============================================================
DL_CMD=(
    python testing/test.py
    --checkpoint "$CHECKPOINT"
    --dataset "$TEST_DATASET"
    --split "$SPLIT"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device "$DEVICE"
    --seed "$SEED"
    --progressive-start-frac "$PROGRESSIVE_START_FRAC"
    --progressive-end-frac "$PROGRESSIVE_END_FRAC"
    --progressive-num-steps "$PROGRESSIVE_NUM_STEPS"
    --min-prefix-len "$MIN_PREFIX_LEN"
    --results-root "$RESULTS_ROOT"
    --run-name "$DL_RUN_NAME"
)

if [[ "$SAVE_SUMMARY_CSV" == "1" ]]; then
    DL_CMD+=(--save-summary-csv)
fi

if [[ "$SAVE_SERIES_CSV" == "1" ]]; then
    DL_CMD+=(--save-series-csv)
fi

if [[ "$MAKE_PLOTS" == "1" ]]; then
    DL_CMD+=(--make-plots)
fi

if [[ "$VERBOSE" == "1" ]]; then
    DL_CMD+=(--verbose)
fi

echo "========================================================"
echo "[1/3] Running DL testing"
echo "========================================================"
printf '%q ' "${DL_CMD[@]}"
echo
"${DL_CMD[@]}"

# ============================================================
# Step 2: CSD testing
# ============================================================
CSD_CMD=(
    python testing/test_csd.py
    --dataset "$TEST_DATASET"
    --split "$SPLIT"
    --seed "$SEED"
    --rolling-window-frac "$ROLLING_WINDOW_FRAC"
    --progressive-start-frac "$PROGRESSIVE_START_FRAC"
    --progressive-end-frac "$PROGRESSIVE_END_FRAC"
    --progressive-num-steps "$PROGRESSIVE_NUM_STEPS"
    --min-prefix-len "$MIN_PREFIX_LEN"
    --gaussian-smooth-sigma "$GAUSSIAN_SMOOTH_SIGMA"
    --results-root "$RESULTS_ROOT"
    --run-name "$CSD_RUN_NAME"
)

if [[ "$SAVE_SERIES_CSV" == "1" ]]; then
    CSD_CMD+=(--save-series-csv)
fi

if [[ "$MAKE_PLOTS" == "1" ]]; then
    CSD_CMD+=(--make-plots)
fi

if [[ "$VERBOSE" == "1" ]]; then
    CSD_CMD+=(--verbose)
fi

echo "========================================================"
echo "[2/3] Running CSD testing"
echo "========================================================"
printf '%q ' "${CSD_CMD[@]}"
echo
"${CSD_CMD[@]}"

# ============================================================
# Step 3: Compare DL vs CSD
# For pangaea_923197 this is still useful because it creates
# combined per-series panels. ROC/AUC will simply be skipped
# inside compare_dl_vs_csd.py if labels are unavailable.
# ============================================================
COMPARE_CMD=(
    python testing/compare_dl_vs_csd.py
    --dl-run-dir "$RESULTS_ROOT/testing/$DL_RUN_NAME"
    --csd-run-dir "$RESULTS_ROOT/testing_csd/$CSD_RUN_NAME"
    --positive-class-index "$POSITIVE_CLASS_INDEX"
    --neutral-label "$NEUTRAL_LABEL"
    --results-root "$RESULTS_ROOT"
    --run-name "$COMPARE_RUN_NAME"
)

if [[ -n "$POSITIVE_CLASS_NAME" ]]; then
    COMPARE_CMD+=(--positive-class-name "$POSITIVE_CLASS_NAME")
fi

if [[ "$MAKE_PER_SERIES_PLOTS" == "1" ]]; then
    COMPARE_CMD+=(--make-per-series-plots)
fi

if [[ "$VERBOSE" == "1" ]]; then
    COMPARE_CMD+=(--verbose)
fi

echo "========================================================"
echo "[3/3] Running DL vs CSD comparison"
echo "========================================================"
printf '%q ' "${COMPARE_CMD[@]}"
echo
"${COMPARE_CMD[@]}"

echo "========================================================"
echo "Testing flow completed successfully"
echo "DL results      : $RESULTS_ROOT/testing/$DL_RUN_NAME"
echo "CSD results     : $RESULTS_ROOT/testing_csd/$CSD_RUN_NAME"
echo "Comparison      : $RESULTS_ROOT/comparison/$COMPARE_RUN_NAME"
echo "========================================================"
