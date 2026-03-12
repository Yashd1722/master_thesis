#!/bin/bash
# inference.sh
# Testing-only launcher for existing checkpoints

set -euo pipefail

usage() {
    cat <<EOF
Usage: bash inference.sh --dataset DATASET --metric METRIC [options]

Required:
  --dataset NAME           Empirical dataset name, e.g. pangaea_923197
  --metric NAME            Metric name in checkpoint, e.g. f1_macro, acc

Optional:
  --model NAME             Optional model filter, e.g. lstm, gru, cnn_lstm
  --train_dataset NAME     Optional training dataset filter, e.g. ts_500 or ts_1500
  --label_col NAME         Optional label column for compare script
  --window_frac FLOAT      CSD rolling window fraction (default: 0.5)
  --min_window INT         CSD minimum rolling window size (default: 20)
  --decision_threshold F   Threshold for comparison (default: 0.5)
  --num_classes INT        Number of classes (default: 4)
  --model_class NAME       Optional exact model class name
  --model_kwargs JSON      Optional JSON dict of model kwargs
  --python CMD             Python executable (default: python3)
  --root_dir PATH          Project root (default: current working directory)
  --help                   Show this help

Expected checkpoint naming:
  model_trainDataset_metric.pt

Examples:
  lstm_ts_500_f1_macro.pt
  cnn_lstm_ts_500_f1_macro.pt
EOF
}

DATASET=""
METRIC=""
MODEL=""
TRAIN_DATASET=""
LABEL_COL=""
WINDOW_FRAC="0.5"
MIN_WINDOW="20"
DECISION_THRESHOLD="0.5"
NUM_CLASSES="4"
MODEL_CLASS=""
MODEL_KWARGS=""
PYTHON_BIN="python3"
ROOT_DIR="$(pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --metric) METRIC="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --train_dataset) TRAIN_DATASET="$2"; shift 2 ;;
        --label_col) LABEL_COL="$2"; shift 2 ;;
        --window_frac) WINDOW_FRAC="$2"; shift 2 ;;
        --min_window) MIN_WINDOW="$2"; shift 2 ;;
        --decision_threshold) DECISION_THRESHOLD="$2"; shift 2 ;;
        --num_classes) NUM_CLASSES="$2"; shift 2 ;;
        --model_class) MODEL_CLASS="$2"; shift 2 ;;
        --model_kwargs) MODEL_KWARGS="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --root_dir) ROOT_DIR="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required."
    usage
    exit 1
fi

if [[ -z "$METRIC" ]]; then
    echo "Error: --metric is required."
    usage
    exit 1
fi

cd "$ROOT_DIR"

if [[ ! -d "checkpoints" ]]; then
    echo "Error: checkpoints/ folder not found in $ROOT_DIR"
    exit 1
fi

mkdir -p test_logs
mkdir -p test_results/dl
mkdir -p test_results/csd
mkdir -p test_results/comparison

echo "========================================"
echo "Testing-only inference launcher"
echo "ROOT_DIR            : $ROOT_DIR"
echo "DATASET             : $DATASET"
echo "METRIC              : $METRIC"
echo "MODEL FILTER        : ${MODEL:-<all>}"
echo "TRAIN DATASET FILTER: ${TRAIN_DATASET:-<all>}"
echo "========================================"

shopt -s nullglob
CHECKPOINT_FILES=(checkpoints/*.pt checkpoints/*.pth)
shopt -u nullglob

if [[ ${#CHECKPOINT_FILES[@]} -eq 0 ]]; then
    echo "Error: no checkpoint files found in checkpoints/"
    exit 1
fi

MATCHED=0

parse_checkpoint() {
    local stem="$1"
    local parsed_model=""
    local parsed_train_dataset=""
    local parsed_metric=""

    for ds in ts_500 ts_1500; do
        local marker="_${ds}_"
        if [[ "$stem" == *"$marker"* ]]; then
            parsed_model="${stem%%$marker*}"
            parsed_train_dataset="$ds"
            parsed_metric="${stem#*$marker}"
            echo "${parsed_model}|${parsed_train_dataset}|${parsed_metric}"
            return 0
        fi
    done

    return 1
}

for CKPT in "${CHECKPOINT_FILES[@]}"; do
    CKPT_FILE="$(basename "$CKPT")"
    STEM="${CKPT_FILE%.*}"

    if ! PARSED="$(parse_checkpoint "$STEM")"; then
        echo "Skipping malformed or unsupported checkpoint name: $CKPT_FILE"
        continue
    fi

    MODEL_PART="${PARSED%%|*}"
    REST="${PARSED#*|}"
    TRAIN_DATASET_PART="${REST%%|*}"
    METRIC_PART="${REST#*|}"

    if [[ -n "$MODEL" && "$MODEL_PART" != "$MODEL" ]]; then
        continue
    fi

    if [[ -n "$TRAIN_DATASET" && "$TRAIN_DATASET_PART" != "$TRAIN_DATASET" ]]; then
        continue
    fi

    if [[ "$METRIC_PART" != "$METRIC" ]]; then
        continue
    fi

    MATCHED=$((MATCHED + 1))

    echo
    echo "----------------------------------------"
    echo "Checkpoint #$MATCHED"
    echo "File           : $CKPT_FILE"
    echo "Model          : $MODEL_PART"
    echo "Train dataset  : $TRAIN_DATASET_PART"
    echo "Metric         : $METRIC_PART"
    echo "Test dataset   : $DATASET"
    echo "----------------------------------------"

    DL_CMD=(
        "$PYTHON_BIN" testing/test.py
        --dataset "$DATASET"
        --train_dataset "$TRAIN_DATASET_PART"
        --model "$MODEL_PART"
        --metric "$METRIC_PART"
        --num_classes "$NUM_CLASSES"
    )

    if [[ -n "$MODEL_CLASS" ]]; then
        DL_CMD+=( --model_class "$MODEL_CLASS" )
    fi

    if [[ -n "$MODEL_KWARGS" ]]; then
        DL_CMD+=( --model_kwargs "$MODEL_KWARGS" )
    fi

    echo "[1/3] DL inference"
    "${DL_CMD[@]}"

    echo "[2/3] CSD inference"
    "$PYTHON_BIN" testing/test_csd.py \
        --dataset "$DATASET" \
        --model "$MODEL_PART" \
        --metric "$METRIC_PART" \
        --window_frac "$WINDOW_FRAC" \
        --min_window "$MIN_WINDOW"

    echo "[3/3] Comparison"
    if [[ -n "$LABEL_COL" ]]; then
        "$PYTHON_BIN" testing/compare_dl_vs_csd.py \
            --dataset "$DATASET" \
            --model "$MODEL_PART" \
            --metric "$METRIC_PART" \
            --label_col "$LABEL_COL" \
            --decision_threshold "$DECISION_THRESHOLD"
    else
        "$PYTHON_BIN" testing/compare_dl_vs_csd.py \
            --dataset "$DATASET" \
            --model "$MODEL_PART" \
            --metric "$METRIC_PART" \
            --decision_threshold "$DECISION_THRESHOLD"
    fi
done

if [[ "$MATCHED" -eq 0 ]]; then
    echo "Error: no matching checkpoints found."
    echo "Filters used:"
    echo "  model         = ${MODEL:-<all>}"
    echo "  train_dataset = ${TRAIN_DATASET:-<all>}"
    echo "  metric        = $METRIC"
    exit 1
fi

echo
echo "========================================"
echo "Finished testing for $MATCHED checkpoint(s)."
echo "Logs saved in        : test_logs/"
echo "DL results in        : test_results/dl/"
echo "CSD results in       : test_results/csd/"
echo "Comparison results in: test_results/comparison/"
echo "========================================"
