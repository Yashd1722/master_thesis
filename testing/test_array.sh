#!/bin/bash
# =============================================================================
#  testing/test_array.sh
#  SLURM array job — one job per model, tests on all PANGAEA cores.
#
#  Each job does in sequence:
#    1. evaluate.py    → runs inference, saves prediction CSVs
#    2. compute_metrics.py → computes ROC/AUC, saves metrics JSONs
#    3. plot_figures.py    → generates Fig 2, 3, 4, 5
#
#  Array:  0 → cnn_lstm
#          1 → lstm
#          2 → cnn
#
#  Dependency (run after training completes):
#      sbatch --dependency=afterok:<train_job_id> testing/test_array.sh
#
#  Or run independently after training:
#      sbatch testing/test_array.sh
#
#  Monitor:
#      squeue -u $USER
#      tail -f test_logs/cnn_lstm_MS21_inference.log
# =============================================================================

#SBATCH --job-name=ews_test
#SBATCH --array=0-2
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=inference_logs/slurm_test_%x_%a_%j.out
#SBATCH --error=inference_logs/slurm_test_%x_%a_%j.err

# ── Map array index → model name ─────────────────────────────────────────────
declare -A MODEL_MAP
MODEL_MAP[0]="cnn_lstm"
MODEL_MAP[1]="lstm"
MODEL_MAP[2]="cnn"

MODEL=${MODEL_MAP[$SLURM_ARRAY_TASK_ID]}

if [ -z "$MODEL" ]; then
    echo "ERROR: No model mapped for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Default: use ts_500 checkpoint. Change to ts_1500 if needed.
DATASET="ts_500"

# ── Environment ──────────────────────────────────────────────────────────────
cd ~/Master_thesis/master_thesis || exit 1
source ~/myenv/bin/activate

echo "=============================================="
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Array task    : $SLURM_ARRAY_TASK_ID"
echo "  Model         : $MODEL"
echo "  Dataset       : $DATASET"
echo "  Node          : $SLURMD_NODENAME"
echo "  GPUs          : $CUDA_VISIBLE_DEVICES"
echo "  Start time    : $(date)"
echo "=============================================="

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p results test_results test_logs inference_logs metrics

# ── Step 1: Inference on all PANGAEA cores ────────────────────────────────────
echo ""
echo ">>> Step 1/3: Running inference — $MODEL on all cores ..."
echo ""

python testing/evaluate.py \
    --model   "$MODEL"   \
    --dataset "$DATASET" \
    --config  config.yaml

if [ $? -ne 0 ]; then
    echo "ERROR: evaluate.py failed for $MODEL"
    exit 1
fi

# ── Step 2: Compute ROC/AUC metrics ──────────────────────────────────────────
echo ""
echo ">>> Step 2/3: Computing metrics — $MODEL ..."
echo ""

python testing/compute_metrics.py \
    --model   "$MODEL"   \
    --dataset "$DATASET" \
    --config  config.yaml

if [ $? -ne 0 ]; then
    echo "ERROR: compute_metrics.py failed for $MODEL"
    exit 1
fi

# ── Step 3: Generate all figures ─────────────────────────────────────────────
echo ""
echo ">>> Step 3/3: Generating figures — $MODEL ..."
echo ""

python testing/plot_figures.py \
    --model   "$MODEL"   \
    --dataset "$DATASET" \
    --config  config.yaml

if [ $? -ne 0 ]; then
    echo "ERROR: plot_figures.py failed for $MODEL"
    exit 1
fi

echo ""
echo "=============================================="
echo "  All done: $MODEL"
echo "  Predictions : results/"
echo "  Metrics     : metrics/"
echo "  Figures     : test_results/"
echo "  End time    : $(date)"
echo "=============================================="

exit 0
