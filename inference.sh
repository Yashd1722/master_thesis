#!/usr/bin/env bash
#SBATCH --job-name=thesis_infer_all
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=test_logs/thesis_infer_all_%j.out
#SBATCH --error=test_logs/thesis_infer_all_%j.err

set -euo pipefail

# ----------------------------------------
# Paths / environment
# ----------------------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/Master_thesis/Main}"
VENV_DIR="$HOME/Master_thesis/myenv"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

ROOT_DIR="$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$ROOT_DIR/test_logs"
mkdir -p "$ROOT_DIR/test_results"
mkdir -p "$ROOT_DIR/inference_logs"

echo "========================================"
echo "PROJECT_DIR     : $PROJECT_DIR"
echo "ROOT_DIR        : $ROOT_DIR"
echo "PWD             : $(pwd)"
echo "PYTHON_BIN      : $PYTHON_BIN"
echo "PYTHON_PATH     : $(which "$PYTHON_BIN")"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "========================================"

# ----------------------------------------
# User input
# ----------------------------------------
TEST_DATASET="${TEST_DATASET:-ts_500}"          # ts_500 | ts_1500 | pangaea_923197
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"

WINDOW_FRAC="${WINDOW_FRAC:-0.5}"
MIN_WINDOW="${MIN_WINDOW:-20}"
FEATURE_MODE="${FEATURE_MODE:-first}"

GENERATE_NULL="${GENERATE_NULL:-1}"             # 1=yes, 0=no
N_NULL="${N_NULL:-100}"
NULL_FIT_FRACTION="${NULL_FIT_FRACTION:-0.2}"

DEVICE="${DEVICE:-cuda}"

# ----------------------------------------
# Dataset-specific config
# ----------------------------------------
FIXED_LENGTH=""
EXPECTED_TRAIN_TOKENS=""

case "$TEST_DATASET" in
  ts_500)
    FIXED_LENGTH=500
    EXPECTED_TRAIN_TOKENS="ts_500"
    ;;
  ts_1500)
    FIXED_LENGTH=1500
    EXPECTED_TRAIN_TOKENS="ts_1500"
    ;;
  pangaea_923197)
    FIXED_LENGTH=""
    # empirical dataset can be tested with both checkpoint families
    EXPECTED_TRAIN_TOKENS="ts_500 ts_1500"
    ;;
  *)
    echo "Unsupported TEST_DATASET: $TEST_DATASET"
    echo "Supported datasets: ts_500 | ts_1500 | pangaea_923197"
    exit 1
    ;;
esac

echo "========================================"
echo "Test dataset   : $TEST_DATASET"
echo "Checkpoint dir : $CHECKPOINT_DIR"
echo "Window frac    : $WINDOW_FRAC"
echo "Min window     : $MIN_WINDOW"
echo "Feature mode   : $FEATURE_MODE"
echo "Generate null  : $GENERATE_NULL"
echo "N null         : $N_NULL"
echo "Null fit frac  : $NULL_FIT_FRACTION"
echo "Device         : $DEVICE"
echo "========================================"

# ----------------------------------------
# Functions
# ----------------------------------------
run_dl() {
  local model="$1"
  local metric="$2"
  local train_dataset="$3"
  local experiment="$4"
  local run_tag="$5"

  echo "[1/3] DL inference/testing"

  local cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/testing/test.py"
    --dataset "$TEST_DATASET"
    --train_dataset "$train_dataset"
    --model "$model"
    --metric "$metric"
    --device "$DEVICE"
  )

  if [[ -n "$FIXED_LENGTH" ]]; then
    cmd+=(--fixed_length "$FIXED_LENGTH")
  fi

  # pass experiment only for non-base runs
  if [[ "$experiment" != "base" ]]; then
    cmd+=(--experiment "$experiment")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$ROOT_DIR/inference_logs/${run_tag}_dl.log"
}

run_csd() {
  local model="$1"
  local metric="$2"
  local train_dataset="$3"
  local experiment="$4"
  local run_tag="$5"

  echo "[2/3] CSD testing"

  local cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/testing/test_csd.py"
    --dataset "$TEST_DATASET"
    --train_dataset "$train_dataset"
    --model "$model"
    --metric "$metric"
    --window_frac "$WINDOW_FRAC"
    --min_window "$MIN_WINDOW"
    --feature_mode "$FEATURE_MODE"
  )

  if [[ -n "$FIXED_LENGTH" ]]; then
    cmd+=(--fixed_length "$FIXED_LENGTH")
  fi

  if [[ "$experiment" != "base" ]]; then
    cmd+=(--experiment "$experiment")
  fi

  if [[ "$GENERATE_NULL" == "1" ]]; then
    cmd+=(
      --generate_null
      --n_null "$N_NULL"
      --null_fit_fraction "$NULL_FIT_FRACTION"
    )
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$ROOT_DIR/inference_logs/${run_tag}_csd.log"
}

run_compare() {
  local model="$1"
  local metric="$2"
  local train_dataset="$3"
  local experiment="$4"
  local run_tag="$5"

  echo "[3/3] Comparing DL vs CSD"

  local cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/testing/compare_dl_vs_csd.py"
    --dataset "$TEST_DATASET"
    --train_dataset "$train_dataset"
    --model "$model"
    --metric "$metric"
  )

  if [[ "$experiment" != "base" ]]; then
    cmd+=(--experiment "$experiment")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$ROOT_DIR/inference_logs/${run_tag}_compare.log"
}

# ----------------------------------------
# Parse checkpoint name
# Expected formats:
#   cnn_ts_500_acc.pt
#   cnn_ts_500_acc_trend.pt
#   cnn_ts_500_acc_season.pt
#   cnn_ts_500_acc_trend_season.pt
#   lstm_ts_1500_f1_macro.pt
#   lstm_ts_1500_f1_macro_trend.pt
#   cnn_lstm_ts_500_f1_macro_trend_season.pt
#
# Rule:
#   <model>_<train_dataset>_<metric>[_trend][_season].pt
# ----------------------------------------
parse_checkpoint_name() {
  local checkpoint_base="$1"

  PARSED_MODEL=""
  PARSED_TRAIN_DATASET=""
  PARSED_METRIC=""
  PARSED_EXPERIMENT="base"

  local base="$checkpoint_base"
  local experiment="base"

  # Detect experiment suffix
  if [[ "$base" == *_trend_season ]]; then
    experiment="trend_season"
    base="${base%_trend_season}"
  elif [[ "$base" == *_season ]]; then
    experiment="season"
    base="${base%_season}"
  elif [[ "$base" == *_trend ]]; then
    experiment="trend"
    base="${base%_trend}"
  fi

  local metric=""
  local prefix=""

  if [[ "$base" == *_f1_macro ]]; then
    metric="f1_macro"
    prefix="${base%_f1_macro}"
  elif [[ "$base" == *_acc ]]; then
    metric="acc"
    prefix="${base%_acc}"
  else
    return 1
  fi

  local train_dataset=""
  local model=""

  if [[ "$prefix" == *_ts_500 ]]; then
    train_dataset="ts_500"
    model="${prefix%_ts_500}"
  elif [[ "$prefix" == *_ts_1500 ]]; then
    train_dataset="ts_1500"
    model="${prefix%_ts_1500}"
  else
    return 1
  fi

  if [[ -z "$model" || -z "$train_dataset" || -z "$metric" ]]; then
    return 1
  fi

  PARSED_MODEL="$model"
  PARSED_TRAIN_DATASET="$train_dataset"
  PARSED_METRIC="$metric"
  PARSED_EXPERIMENT="$experiment"
  return 0
}

# ----------------------------------------
# Collect checkpoints automatically
# ----------------------------------------
shopt -s nullglob
checkpoint_files=("$CHECKPOINT_DIR"/*.pt)
shopt -u nullglob

if [[ ${#checkpoint_files[@]} -eq 0 ]]; then
  echo "No checkpoint files found in: $CHECKPOINT_DIR"
  exit 1
fi

echo "Found ${#checkpoint_files[@]} checkpoint(s)."
echo

# ----------------------------------------
# Main loop over all checkpoints
# ----------------------------------------
for ckpt_path in "${checkpoint_files[@]}"; do
  checkpoint_file="$(basename "$ckpt_path")"
  checkpoint_name="${checkpoint_file%.pt}"

  if ! parse_checkpoint_name "$checkpoint_name"; then
    echo "WARNING: could not parse checkpoint name: $checkpoint_file"
    echo "Expected examples:"
    echo "  cnn_ts_500_acc.pt"
    echo "  cnn_ts_500_acc_trend.pt"
    echo "  cnn_ts_500_acc_season.pt"
    echo "  cnn_ts_500_acc_trend_season.pt"
    echo "  lstm_ts_1500_f1_macro.pt"
    echo "  cnn_lstm_ts_500_f1_macro_trend_season.pt"
    echo "Skipping..."
    echo
    continue
  fi

  model="$PARSED_MODEL"
  train_dataset="$PARSED_TRAIN_DATASET"
  metric="$PARSED_METRIC"
  experiment="$PARSED_EXPERIMENT"

  valid_train=0
  for token in $EXPECTED_TRAIN_TOKENS; do
    if [[ "$train_dataset" == "$token" ]]; then
      valid_train=1
      break
    fi
  done

  if [[ "$valid_train" -ne 1 ]]; then
    echo "Skipping checkpoint due to dataset mismatch: $checkpoint_file"
    continue
  fi

  run_tag="${model}_${train_dataset}_${metric}_${experiment}_on_${TEST_DATASET}"

  echo "----------------------------------------"
  echo "Checkpoint     : $checkpoint_file"
  echo "Model          : $model"
  echo "Train dataset  : $train_dataset"
  echo "Metric         : $metric"
  echo "Experiment     : $experiment"
  echo "Test dataset   : $TEST_DATASET"
  echo "Run tag        : $run_tag"
  echo "----------------------------------------"

  {
    echo "========================================"
    echo "START: $(date)"
    echo "Checkpoint     : $checkpoint_file"
    echo "Model          : $model"
    echo "Train dataset  : $train_dataset"
    echo "Metric         : $metric"
    echo "Experiment     : $experiment"
    echo "Test dataset   : $TEST_DATASET"
    echo "Device         : $DEVICE"
    echo "========================================"

    run_dl "$model" "$metric" "$train_dataset" "$experiment" "$run_tag"
    run_csd "$model" "$metric" "$train_dataset" "$experiment" "$run_tag"
    run_compare "$model" "$metric" "$train_dataset" "$experiment" "$run_tag"

    echo "========================================"
    echo "END: $(date)"
    echo "Completed: $run_tag"
    echo "========================================"
  } 2>&1 | tee "$ROOT_DIR/test_logs/${run_tag}.log"

  echo
done

echo "========================================"
echo "Inference pipeline completed successfully"
echo "========================================"
