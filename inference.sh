#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Config
# ----------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

TEST_DATASET="${TEST_DATASET:-ts_500}"          # ts_500 | ts_1500 | pangaea_923197
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"

# Comma-separated or space-separated lists
MODELS_DEFAULT="cnn lstm cnn_lstm"
METRICS_DEFAULT="acc f1_macro"

MODELS="${MODELS:-$MODELS_DEFAULT}"
METRICS="${METRICS:-$METRICS_DEFAULT}"

WINDOW_FRAC="${WINDOW_FRAC:-0.5}"
MIN_WINDOW="${MIN_WINDOW:-20}"
FEATURE_MODE="${FEATURE_MODE:-first}"

GENERATE_NULL="${GENERATE_NULL:-1}"             # 1=yes, 0=no
N_NULL="${N_NULL:-100}"
NULL_FIT_FRACTION="${NULL_FIT_FRACTION:-0.2}"

DEVICE="${DEVICE:-cpu}"

# For synthetic datasets, enforce fixed input length.
FIXED_LENGTH=""
TRAIN_DATASET_TOKEN=""

case "$TEST_DATASET" in
  ts_500)
    FIXED_LENGTH=500
    TRAIN_DATASET_TOKEN="ts_500"
    ;;
  ts_1500)
    FIXED_LENGTH=1500
    TRAIN_DATASET_TOKEN="ts_1500"
    ;;
  pangaea_923197)
    FIXED_LENGTH=""
    # choose which checkpoint family to use for empirical inference
    TRAIN_DATASET_TOKEN="${TRAIN_DATASET_TOKEN:-ts_500}"
    ;;
  *)
    echo "Unsupported TEST_DATASET: $TEST_DATASET"
    echo "Supported: ts_500 | ts_1500 | pangaea_923197"
    exit 1
    ;;
esac

# Normalize separators: allow commas or spaces
MODELS="$(echo "$MODELS" | tr ',' ' ')"
METRICS="$(echo "$METRICS" | tr ',' ' ')"

echo "========================================"
echo "Test dataset   : $TEST_DATASET"
echo "Checkpoint dir : $CHECKPOINT_DIR"
echo "Models         : $MODELS"
echo "Metrics        : $METRICS"
echo "Window frac    : $WINDOW_FRAC"
echo "Min window     : $MIN_WINDOW"
echo "Feature mode   : $FEATURE_MODE"
echo "Generate null  : $GENERATE_NULL"
echo "========================================"

run_dl() {
  local model="$1"
  local metric="$2"
  local train_dataset="$3"

  echo "[1/3] DL inference/testing"

  cmd=(
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

  "${cmd[@]}"
}

run_csd() {
  local model="$1"
  local metric="$2"

  echo "[2/3] CSD testing"

  cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/testing/test_csd.py"
    --dataset "$TEST_DATASET"
    --model "$model"
    --metric "$metric"
    --window_frac "$WINDOW_FRAC"
    --min_window "$MIN_WINDOW"
    --feature_mode "$FEATURE_MODE"
  )

  if [[ -n "$FIXED_LENGTH" ]]; then
    cmd+=(--fixed_length "$FIXED_LENGTH")
  fi

  if [[ "$GENERATE_NULL" == "1" ]]; then
    cmd+=(
      --generate_null
      --n_null "$N_NULL"
      --null_fit_fraction "$NULL_FIT_FRACTION"
    )
  fi

  "${cmd[@]}"
}

run_compare() {
  local model="$1"
  local metric="$2"

  echo "[3/3] DL vs CSD comparison"

  cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/testing/compare_dl_vs_csd.py"
    --dataset "$TEST_DATASET"
    --model "$model"
    --metric "$metric"
  )

  if [[ "$GENERATE_NULL" == "1" ]]; then
    cmd+=(--use_null)
  fi

  "${cmd[@]}"
}

for model in $MODELS; do
  for metric in $METRICS; do
    echo
    echo "----------------------------------------"
    echo "Checkpoint     : ${model}_${TRAIN_DATASET_TOKEN}_${metric}.pt"
    echo "Model          : $model"
    echo "Train dataset  : $TRAIN_DATASET_TOKEN"
    echo "Metric         : $metric"
    echo "Test dataset   : $TEST_DATASET"
    echo "----------------------------------------"
    echo

    run_dl "$model" "$metric" "$TRAIN_DATASET_TOKEN"
    run_csd "$model" "$metric"
    run_compare "$model" "$metric"

    echo
    echo "Finished: model=$model | metric=$metric | test_dataset=$TEST_DATASET"
    echo
  done
done

echo "========================================"
echo "All experiments finished successfully."
echo "========================================"
