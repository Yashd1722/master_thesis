#!/usr/bin/env bash
#SBATCH --job-name=thesis_infer_all
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=test_logs/thesis_infer_all_%j.out
#SBATCH --error=test_logs/thesis_infer_all_%j.err

set -euo pipefail

# ----------------------------------------
# Paths / environment
# ----------------------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/Master_thesis/master_thesis}"
VENV_DIR="$HOME/Master_thesis/myenv"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

ROOT_DIR="$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$ROOT_DIR/test_logs"
mkdir -p "$ROOT_DIR/test_results"

echo "========================================"
echo "PROJECT_DIR     : $PROJECT_DIR"
echo "PWD             : $(pwd)"
echo "PYTHON          : $(which "$PYTHON_BIN")"
echo "========================================"

# ----------------------------------------
# Config
# ----------------------------------------
TEST_DATASET="${TEST_DATASET:-pangaea_923197}"   # ts_500 | ts_1500 | pangaea_923197
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"
DEVICE="${DEVICE:-cuda}"

WINDOW_FRAC="${WINDOW_FRAC:-0.5}"
MIN_WINDOW="${MIN_WINDOW:-20}"
FEATURE_MODE="${FEATURE_MODE:-first}"

GENERATE_NULL="${GENERATE_NULL:-1}"
N_NULL="${N_NULL:-100}"
NULL_FIT_FRACTION="${NULL_FIT_FRACTION:-0.2}"

echo "========================================"
echo "Test dataset   : $TEST_DATASET"
echo "Checkpoint dir : $CHECKPOINT_DIR"
echo "Device         : $DEVICE"
echo "========================================"

# ----------------------------------------
# Parse checkpoint name
# Supports:
# cnn_ts_500_acc.pt
# lstm_ts_500_f1_macro_season.pt
# cnn_lstm_ts_1500_f1_macro_trend_season.pt
# ----------------------------------------
parse_checkpoint() {
  local fname="$1"
  fname="${fname%.pt}"

  # Detect model prefix BEFORE splitting on underscores,
  # so that cnn_lstm is not mistaken for model=cnn, train=lstm_ts
  if [[ "$fname" == cnn_lstm* ]]; then
    model="cnn_lstm"
    remainder="${fname#cnn_lstm_}"
  elif [[ "$fname" == cnn* ]]; then
    model="cnn"
    remainder="${fname#cnn_}"
  elif [[ "$fname" == lstm* ]]; then
    model="lstm"
    remainder="${fname#lstm_}"
  else
    echo "ERROR: Unknown model prefix in checkpoint name: $fname" >&2
    return 1
  fi

  # Split the remainder: <ds_part1>_<ds_part2>_<metric...>_<exp...>
  # e.g. ts_500_acc  or  ts_1500_f1_macro_trend_season
  IFS="_" read -r -a parts <<< "$remainder"

  train_dataset="${parts[0]}_${parts[1]}"   # e.g. ts_500 or ts_1500

  # metric can be f1_macro (2 tokens) or acc (1 token)
  if [[ "${parts[2]}" == "f1" ]]; then
    metric="f1_macro"
    exp_parts=("${parts[@]:4}")
  else
    metric="${parts[2]}"
    exp_parts=("${parts[@]:3}")
  fi

  if [[ ${#exp_parts[@]} -eq 0 ]]; then
    experiment="base"
  else
    experiment=$(IFS=_; echo "${exp_parts[*]}")
  fi
}

# ----------------------------------------
# Main loop over ALL checkpoints
# ----------------------------------------
for ckpt in "$CHECKPOINT_DIR"/*.pt; do
  fname=$(basename "$ckpt")

  parse_checkpoint "$fname"

  echo
  echo "========================================"
  echo "Checkpoint     : $fname"
  echo "Model          : $model"
  echo "Train dataset  : $train_dataset"
  echo "Metric         : $metric"
  echo "Experiment     : $experiment"
  echo "Test dataset   : $TEST_DATASET"
  echo "========================================"

  # ----------------------------------------
  # DL
  # ----------------------------------------
  echo "[1/3] DL testing"
  "$PYTHON_BIN" "$ROOT_DIR/testing/test.py" \
    --dataset "$TEST_DATASET" \
    --train_dataset "$train_dataset" \
    --model "$model" \
    --metric "$metric" \
    --experiment "$experiment" \
    --device "$DEVICE"

  # ----------------------------------------
  # CSD
  # ----------------------------------------
  echo "[2/3] CSD testing"
  cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/testing/test_csd.py"
    --dataset "$TEST_DATASET"
    --train_dataset "$train_dataset"
    --model "$model"
    --metric "$metric"
    --experiment "$experiment"
    --window_frac "$WINDOW_FRAC"
    --min_window "$MIN_WINDOW"
    --feature_mode "$FEATURE_MODE"
    --normalize
  )

  if [[ "$GENERATE_NULL" == "1" ]]; then
    cmd+=(
      --generate_null
      --n_null "$N_NULL"
      --null_fit_fraction "$NULL_FIT_FRACTION"
    )
  fi

  "${cmd[@]}"

  # ----------------------------------------
  # Compare
  # ----------------------------------------
  echo "[3/3] DL vs CSD comparison"
  "$PYTHON_BIN" "$ROOT_DIR/testing/compare_dl_vs_csd.py" \
    --dataset "$TEST_DATASET" \
    --train_dataset "$train_dataset" \
    --model "$model" \
    --metric "$metric" \
    --experiment "$experiment"

done

echo
echo "========================================"
echo "Inference pipeline completed"
echo "========================================"
