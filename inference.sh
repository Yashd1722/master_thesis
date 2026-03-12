#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TEST_DATASET="pangaea_923197"
CHECKPOINT_DIR="$ROOT_DIR/checkpoints"

echo "========================================"
echo "Test dataset   : $TEST_DATASET"
echo "Checkpoint dir : $CHECKPOINT_DIR"
echo "========================================"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

shopt -s nullglob
CHECKPOINT_FILES=("$CHECKPOINT_DIR"/*.pt)

if [ ${#CHECKPOINT_FILES[@]} -eq 0 ]; then
    echo "ERROR: No checkpoint files found in $CHECKPOINT_DIR"
    exit 1
fi

for CKPT in "${CHECKPOINT_FILES[@]}"; do
    BASENAME="$(basename "$CKPT" .pt)"

    TRAIN_DATASET=""
    MODEL_NAME=""
    METRIC_NAME=""

    if [[ "$BASENAME" == *"_ts_500_"* ]]; then
        TRAIN_DATASET="ts_500"
        MODEL_NAME="${BASENAME%%_ts_500_*}"
        METRIC_NAME="${BASENAME#*_ts_500_}"
    elif [[ "$BASENAME" == *"_ts_1500_"* ]]; then
        TRAIN_DATASET="ts_1500"
        MODEL_NAME="${BASENAME%%_ts_1500_*}"
        METRIC_NAME="${BASENAME#*_ts_1500_}"
    else
        echo "WARNING: Skipping checkpoint with unsupported name format: $BASENAME.pt"
        continue
    fi

    echo
    echo "----------------------------------------"
    echo "Checkpoint     : $(basename "$CKPT")"
    echo "Model          : $MODEL_NAME"
    echo "Train dataset  : $TRAIN_DATASET"
    echo "Metric         : $METRIC_NAME"
    echo "Test dataset   : $TEST_DATASET"
    echo "----------------------------------------"

    echo "[1/3] DL inference"
    python testing/test.py \
        --dataset "$TEST_DATASET" \
        --train_dataset "$TRAIN_DATASET" \
        --model "$MODEL_NAME" \
        --metric "$METRIC_NAME"

    echo "[2/3] CSD inference"
    python testing/test_csd.py \
        --dataset "$TEST_DATASET" \
        --model "$MODEL_NAME" \
        --metric "$METRIC_NAME"

    echo "[3/3] Comparison"
    python testing/compare_dl_vs_csd.py \
        --dataset "$TEST_DATASET" \
        --model "$MODEL_NAME" \
        --metric "$METRIC_NAME"

    echo "Finished: $(basename "$CKPT")"
done

echo
echo "All inference jobs completed successfully."
