#!/bin/bash
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ts_train
#SBATCH --output=/home/%u/Master_thesis/Main/logs/slurm_%j.out
#SBATCH --error=/home/%u/Master_thesis/Main/logs/slurm_%j.err

set -e

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Go to project directory
cd /home/$USER/Master_thesis/Main

# Create logs directory (needed for slurm output + your python logger)
mkdir -p logs results checkpoints

# Activate virtual environment
echo "Activating virtual environment..."
source /home/$USER/Master_thesis/myenv/bin/activate

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
echo ""

echo "Starting training..."

# ---------------------------
# Option 1: Run a single experiment (like before)
# ---------------------------
# python3 training/train.py \
#     --dataset ts_500 \
#     --model lstm \
#     --metric f1_macro \
#     --epochs 20 \
#     --early_stop \
#     --patience 5

# ---------------------------
# Option 2: Run ALL combinations (models × datasets × metrics)
# (DON'T pass --dataset/--model/--metric)
# ---------------------------
python3 training/train.py \
  --dataset ts_500 \
  --epochs 20 \
  --early_stop \
  --patience 5 \
  --seq_len 500 \
  --bury_padding \
  --bury_pad_mode both \
  --bury_norm \
  --data_seed 42
echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="
