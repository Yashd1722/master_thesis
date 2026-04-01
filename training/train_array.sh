#!/bin/bash
# =============================================================================
#  training/train_array.sh
#  SLURM array job — trains one model per job on ts_500 then ts_1500.
#
#  Array:  0 → cnn_lstm
#          1 → lstm
#          2 → cnn
#
#  Submit:
#      cd ~/Master_thesis/master_thesis
#      sbatch training/train_array.sh
#
#  Monitor:
#      squeue -u $USER
#      tail -f logs/cnn_lstm_ts_500_v1_train.log
# =============================================================================

#SBATCH --job-name=ews_train
#SBATCH --array=0-2
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_train_%x_%a_%j.out
#SBATCH --error=logs/slurm_train_%x_%a_%j.err

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

# ── Environment ──────────────────────────────────────────────────────────────
cd ~/Master_thesis/master_thesis || exit 1
source ~/myenv/bin/activate

echo "=============================================="
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Array task    : $SLURM_ARRAY_TASK_ID"
echo "  Model         : $MODEL"
echo "  Node          : $SLURMD_NODENAME"
echo "  GPUs          : $CUDA_VISIBLE_DEVICES"
echo "  Start time    : $(date)"
echo "=============================================="

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p logs checkpoints metrics

# ── Train: ts_500 then ts_1500 (both pad variants each) ──────────────────────
echo ""
echo ">>> Training $MODEL on ts_500 and ts_1500 ..."
echo ""

python training/train.py \
    --model  "$MODEL"   \
    --mode   bury       \
    --config config.yaml

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "  Finished: $MODEL"
echo "  Exit code : $EXIT_CODE"
echo "  End time  : $(date)"
echo "=============================================="

exit $EXIT_CODE
